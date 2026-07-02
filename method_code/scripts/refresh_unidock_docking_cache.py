from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.pocket_dataset import PocketDatasetPaths, _read_pdb_atoms, _vina_box_from_coords
from mobo.config_utils import load_config
from mobo.io_utils import _is_valid_dock_score
from mobo.oracle import (
    _cache_fetch_conformer,
    _cache_store_docking,
    _default_oracle_cache_root,
    _ensure_oracle_cache,
    _ensure_receptor_pdbqt,
    _hash_file,
    _hash_json,
    _load_reference_ligand_fast,
    _materialize_unidock_pose_file,
    _reference_ligand_box,
    _run_unidock_service_batch,
)


def _section(cfg: dict, key: str) -> dict:
    val = cfg.get(key, {}) if isinstance(cfg, dict) else {}
    return val if isinstance(val, dict) else {}


def _resolve_target_context(
    dataset_root: Path,
    pocket_radius: float,
    *,
    meeko_allow_bad_res: bool,
    meeko_default_altloc: str | None,
) -> dict:
    paths = PocketDatasetPaths.from_root(dataset_root, pocket_radius=float(pocket_radius))
    if not paths.protein_pdb.exists():
        raise FileNotFoundError(f"protein.pdb not found: {paths.protein_pdb}")
    if not paths.pocket_pdb.exists():
        raise FileNotFoundError(f"pocket.pdb not found: {paths.pocket_pdb}")

    ref_mol = None
    if paths.reference_ligand is not None:
        ref_mol = _load_reference_ligand_fast(paths)

    pocket_coords, _ = _read_pdb_atoms(paths.pocket_pdb)
    receptor_pdbqt = paths.protein_pdb.with_suffix(".pdbqt")
    _ensure_receptor_pdbqt(
        paths.protein_pdb,
        receptor_pdbqt,
        meeko_allow_bad_res=meeko_allow_bad_res,
        meeko_default_altloc=meeko_default_altloc,
    )

    box_padding = 4.0
    if ref_mol is not None and ref_mol.GetNumConformers():
        center, box_size = _reference_ligand_box(ref_mol, padding=box_padding)
    else:
        center, raw_box_size = _vina_box_from_coords(pocket_coords)
        box_size = [float(v) + box_padding for v in raw_box_size]

    target_key = _hash_json(
        {
            "receptor_pdbqt_sha256": _hash_file(receptor_pdbqt),
            "center": [float(v) for v in center],
            "box_size": [float(v) for v in box_size],
            "pocket_radius": float(pocket_radius),
            "box_padding": box_padding,
        }
    )

    return {
        "paths": paths,
        "receptor_pdbqt": receptor_pdbqt,
        "center": tuple(float(v) for v in center),
        "box_size": tuple(float(v) for v in box_size),
        "target_key": target_key,
    }


def _load_cache_rows(
    conn: sqlite3.Connection,
    *,
    target_key: str,
    backend: str,
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT docking_key, canonical_smiles, target_key, conformer_key, oracle_json, vina_score, meta_rel, created_at
        FROM docking_cache
        WHERE target_key = ?
        ORDER BY docking_key
        """,
        (target_key,),
    ).fetchall()

    out: list[dict] = []
    for row in rows:
        oracle_json = json.loads(str(row[4]))
        if str(oracle_json.get("backend", "")).strip().lower() != backend:
            continue
        out.append(
            {
                "docking_key": str(row[0]),
                "canonical_smiles": str(row[1]),
                "target_key": str(row[2]),
                "conformer_key": str(row[3]),
                "oracle_json": oracle_json,
                "old_score": float(row[5]),
                "meta_rel": str(row[6]),
                "created_at": str(row[7]),
            }
        )
    return out


def _load_meta(cache_root: Path, meta_rel: str) -> dict:
    meta_path = cache_root / meta_rel
    if not meta_path.exists():
        raise FileNotFoundError(f"meta file missing: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _build_summary(df: pd.DataFrame) -> dict:
    success = df.loc[df["status"] == "ok"].copy()
    dry_run = df.loc[df["status"] == "dry_run"].copy()
    summary = {
        "rows_total": int(df.shape[0]),
        "rows_ok": int(success.shape[0]),
        "rows_dry_run": int(dry_run.shape[0]),
        "rows_failed": int(((df["status"] != "ok") & (df["status"] != "dry_run")).sum()),
    }
    if success.empty:
        return summary

    delta = success["delta_new_minus_old"].to_numpy(dtype=np.float64)
    summary.update(
        {
            "old_score_mean": float(success["old_score"].mean()),
            "new_score_mean": float(success["new_score"].mean()),
            "delta_new_minus_old_mean": float(np.mean(delta)),
            "delta_new_minus_old_median": float(np.median(delta)),
            "delta_new_minus_old_min": float(np.min(delta)),
            "delta_new_minus_old_max": float(np.max(delta)),
            "higher_raw_score_count": int((delta > 0).sum()),
            "lower_raw_score_count": int((delta < 0).sum()),
            "unchanged_score_count": int((delta == 0).sum()),
            "improved_if_lower_is_better_count": int((success["new_score"] < success["old_score"]).sum()),
            "worsened_if_lower_is_better_count": int((success["new_score"] > success["old_score"]).sum()),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
        }
    )
    return summary


def _load_allowed_docking_keys(path: str | None) -> set[str] | None:
    if path in (None, ""):
        return None
    key_path = Path(str(path)).resolve()
    if not key_path.exists():
        raise FileNotFoundError(f"docking key file not found: {key_path}")
    keys = {
        line.strip()
        for line in key_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not keys:
        raise ValueError(f"no docking keys found in: {key_path}")
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh Uni-Dock docking_cache in place using cached conformers only.")
    ap.add_argument("--config", default="config/surrogate/config.yaml")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--target-key", default=None)
    ap.add_argument("--service-url", default=None)
    ap.add_argument("--wsl-distro", default=None)
    ap.add_argument("--backend", default="unidock_service")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--analysis-dir", default=None)
    ap.add_argument("--docking-key-file", default=None)
    ap.add_argument("--scoring", default=None)
    ap.add_argument("--search-mode", default=None)
    ap.add_argument("--num-modes", type=int, default=None)
    ap.add_argument("--timeout-sec", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    general_cfg = _section(cfg, "general")
    oracle_cfg = _section(cfg, "oracle")
    objective_cfg = _section(cfg, "objective")

    dataset_root = Path(args.dataset_root or general_cfg.get("dataset_root", "dataset/8UN4")).resolve()
    cache_root = Path(args.cache_root).resolve() if args.cache_root else _default_oracle_cache_root()
    backend = str(args.backend).strip().lower()
    if backend != "unidock_service":
        raise ValueError(f"Only unidock_service is supported, got backend={backend}")

    pocket_radius = float(oracle_cfg.get("oracle_pocket_radius", 10.0))
    ctx = _resolve_target_context(
        dataset_root=dataset_root,
        pocket_radius=pocket_radius,
        meeko_allow_bad_res=bool(oracle_cfg.get("meeko_allow_bad_res", False)),
        meeko_default_altloc=oracle_cfg.get("meeko_default_altloc", None),
    )
    target_key = str(args.target_key).strip() if args.target_key else str(ctx["target_key"])
    if target_key != str(ctx["target_key"]):
        raise RuntimeError(
            f"Target key mismatch: resolved={ctx['target_key']} requested={target_key}. "
            "Use the dataset_root matching the cache target."
        )

    service_url = str(args.service_url or oracle_cfg.get("unidock_service_url", "")).strip()
    if not service_url:
        raise ValueError("unidock service url is required")
    wsl_distro = args.wsl_distro or oracle_cfg.get("unidock_service_wsl_distro", None)
    scoring = str(args.scoring or oracle_cfg.get("unidock_scoring", "vina"))
    search_mode = str(args.search_mode or oracle_cfg.get("unidock_search_mode", "balance"))
    num_modes = int(args.num_modes if args.num_modes is not None else oracle_cfg.get("unidock_num_modes", 1))
    timeout_sec = int(args.timeout_sec if args.timeout_sec is not None else oracle_cfg.get("unidock_timeout_sec", 3600))
    dock_valid_max = objective_cfg.get("dock_valid_max", 0.0)
    dock_valid_max = None if dock_valid_max is None else float(dock_valid_max)

    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else (
        Path("analysis")
        / "docking_cache_refresh"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    conn = _ensure_oracle_cache(cache_root)
    rows = _load_cache_rows(conn, target_key=target_key, backend=backend)
    target_total_rows = len(rows)
    allowed_keys = _load_allowed_docking_keys(args.docking_key_file)
    if allowed_keys is not None:
        rows = [row for row in rows if row["docking_key"] in allowed_keys]
    total_rows = len(rows)
    if args.offset > 0:
        rows = rows[int(args.offset) :]
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    plan = {
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "analysis_dir": str(analysis_dir),
        "target_key": target_key,
        "target_key_total_rows": target_total_rows,
        "filtered_rows": total_rows,
        "selected_rows": len(rows),
        "service_url": service_url,
        "backend": backend,
        "scoring": scoring,
        "search_mode": search_mode,
        "num_modes": num_modes,
        "timeout_sec": timeout_sec,
        "batch_size": int(args.batch_size),
        "dry_run": bool(args.dry_run),
    }
    (analysis_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print(json.dumps(plan, indent=2))
    if not rows:
        conn.close()
        print("no matching docking_cache rows")
        return 0

    results: list[dict] = []
    tmp_dir = cache_root / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(rows), max(1, int(args.batch_size))):
        chunk = rows[start : start + max(1, int(args.batch_size))]
        requests_batch: list[dict] = []
        active: list[dict] = []

        for row in chunk:
            meta = _load_meta(cache_root, row["meta_rel"])
            conf = _cache_fetch_conformer(conn, cache_root, row["conformer_key"])
            if conf is None:
                results.append(
                    {
                        "docking_key": row["docking_key"],
                        "canonical_smiles": row["canonical_smiles"],
                        "old_score": row["old_score"],
                        "new_score": np.nan,
                        "delta_new_minus_old": np.nan,
                        "status": "missing_conformer_cache",
                        "error": f"conformer cache missing for {row['conformer_key']}",
                    }
                )
                continue

            active.append(
                {
                    "row": row,
                    "meta": meta,
                    "conf": conf,
                }
            )
            requests_batch.append(
                {
                    "lig_id": row["docking_key"],
                    "docking_key": row["docking_key"],
                    "ligand_sdf": conf["pre_sdf"],
                }
            )

        if not active:
            continue

        if args.dry_run:
            for item in active:
                row = item["row"]
                results.append(
                    {
                        "docking_key": row["docking_key"],
                        "canonical_smiles": row["canonical_smiles"],
                        "old_score": row["old_score"],
                        "new_score": np.nan,
                        "delta_new_minus_old": np.nan,
                        "status": "dry_run",
                        "error": "",
                    }
                )
            continue

        service_results = _run_unidock_service_batch(
            service_url=service_url,
            target_key=target_key,
            pocket_pdbqt=ctx["receptor_pdbqt"],
            center=list(ctx["center"]),
            box_size=list(ctx["box_size"]),
            requests_batch=requests_batch,
            scoring=scoring,
            search_mode=search_mode,
            num_modes=num_modes,
            timeout_sec=timeout_sec,
        )
        result_by_key = {str(item.get("docking_key", "")): item for item in service_results}

        for item in active:
            row = item["row"]
            meta_old = dict(item["meta"])
            res = result_by_key.get(row["docking_key"])
            if res is None:
                results.append(
                    {
                        "docking_key": row["docking_key"],
                        "canonical_smiles": row["canonical_smiles"],
                        "old_score": row["old_score"],
                        "new_score": np.nan,
                        "delta_new_minus_old": np.nan,
                        "status": "missing_service_result",
                        "error": "service result missing",
                    }
                )
                continue

            try:
                if str(res.get("status", "")).lower() != "ok":
                    raise RuntimeError(str(res.get("error_message", "service reported failure")))

                score = float(res.get("score"))
                if not _is_valid_dock_score(score, dock_valid_max=dock_valid_max):
                    raise RuntimeError(f"invalid dock score returned: {score}")

                output_path = res.get("output_path")
                input_mode = str(res.get("input_mode", "sdf"))
                pose_format = "none"
                pose_file_name = "score_only.txt"
                pose_warning = None
                tmp_pose = None

                if output_path not in (None, ""):
                    try:
                        pose_text, pose_suffix = _materialize_unidock_pose_file(
                            output_path=str(output_path),
                            wsl_distro=wsl_distro,
                        )
                        tmp_pose = tmp_dir / f"{row['docking_key']}{pose_suffix}"
                        tmp_pose.write_text(pose_text, encoding="utf-8")
                        pose_format = pose_suffix.lstrip(".")
                        pose_file_name = f"pose{tmp_pose.suffix}"
                    except FileNotFoundError as exc:
                        pose_warning = str(exc)
                else:
                    pose_warning = "service did not return output_path"

                meta_new = dict(meta_old)
                meta_new.update(
                    {
                        "vina_score": float(score),
                        "target_key": target_key,
                        "conformer_key": row["conformer_key"],
                        "docking_key": row["docking_key"],
                        "center": list(ctx["center"]),
                        "box_size": list(ctx["box_size"]),
                        "backend": backend,
                        "unidock_output_path": None if output_path in (None, "") else str(output_path),
                        "unidock_input_mode": input_mode,
                        "pose_format": pose_format,
                        "pose_warning": pose_warning,
                        "refresh_old_vina_score": float(row["old_score"]),
                        "refresh_timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                )

                _cache_store_docking(
                    conn,
                    cache_root,
                    docking_key=row["docking_key"],
                    canonical_smiles=row["canonical_smiles"],
                    target_key=target_key,
                    conformer_key=row["conformer_key"],
                    oracle_params=row["oracle_json"],
                    vina_score=float(score),
                    pose_sdf_src=tmp_pose,
                    meta=meta_new,
                    pose_file_name=pose_file_name,
                )
                if tmp_pose is not None:
                    tmp_pose.unlink(missing_ok=True)

                results.append(
                    {
                        "docking_key": row["docking_key"],
                        "canonical_smiles": row["canonical_smiles"],
                        "old_score": row["old_score"],
                        "new_score": float(score),
                        "delta_new_minus_old": float(score) - float(row["old_score"]),
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "docking_key": row["docking_key"],
                        "canonical_smiles": row["canonical_smiles"],
                        "old_score": row["old_score"],
                        "new_score": np.nan,
                        "delta_new_minus_old": np.nan,
                        "status": "refresh_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        done = len(results)
        ok = sum(1 for r in results if r["status"] == "ok")
        failed = done - ok
        print(f"[refresh] {done}/{len(rows)} processed ok={ok} non_ok={failed}")

    result_df = pd.DataFrame(results)
    result_csv = analysis_dir / "refresh_results.csv"
    result_df.to_csv(result_csv, index=False)

    summary = _build_summary(result_df)
    summary.update(
        {
            "result_csv": str(result_csv),
            "target_key": target_key,
            "target_key_total_rows": target_total_rows,
            "filtered_rows": total_rows,
            "selected_rows": len(rows),
        }
    )
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
