from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from mobo.io_utils import _read_smiles_csv, _is_valid_dock_score


def resplit_smiles_csv(
    csv_path: str,
    split_ratio: Tuple[float, float, float],
    seed: int,
) -> None:
    df = _read_smiles_csv(csv_path)
    total = len(df)
    if total == 0:
        raise RuntimeError(f"Cannot resplit empty smiles.csv: {csv_path}")
    p_train, p_valid, p_test = split_ratio
    p_sum = max(p_train + p_valid + p_test, 1e-8)
    p_train, p_valid, p_test = (p_train / p_sum, p_valid / p_sum, p_test / p_sum)
    n_train = int(round(total * p_train))
    n_valid = int(round(total * p_valid))
    if n_train + n_valid > total:
        n_valid = max(0, total - n_train)
    n_test = max(0, total - n_train - n_valid)
    splits = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
    while len(splits) < total:
        splits.append("train")
    rng = np.random.RandomState(seed)
    order = rng.permutation(total)
    split_arr = np.array(splits[:total], dtype=object)
    df = df.copy()
    df["split"] = split_arr[order]
    df.to_csv(csv_path, index=False)


def clear_processed(root: str, splits: Sequence[str] = ("train", "valid", "test")) -> None:
    proc_dir = Path(root) / "processed"
    for split in splits:
        base = proc_dir / f"pocket_dataset_{split}.pt"
        for suffix in ("", ".meta", ".atom_per_mol.json"):
            path = Path(str(base) + suffix)
            if path.exists():
                path.unlink()


def _as_float(val):
    try:
        num = float(val)
    except Exception:
        raise ValueError(f"Failed to parse float value: {val!r}")
    if not np.isfinite(num):
        raise ValueError(f"Non-finite float value: {val!r}")
    return num


def save_top_pose_conformers(
    csv_path: Path,
    dataset_root: Path,
    run_dir: Path,
    top_frac: float = 0.1,
    dock_abs_max: float | None = 100.0,
    dock_valid_max: float | None = 0.0,
) -> int:
    if top_frac <= 0:
        return 0
    df = _read_smiles_csv(str(csv_path))
    if df.empty or "dock_score" not in df.columns:
        return 0
    pose_col = "dock_pose" if "dock_pose" in df.columns else None
    if pose_col is None:
        return 0
    smiles_col = None
    for cand in ("smiles", "SMILES", "smile"):
        if cand in df.columns:
            smiles_col = cand
            break
    qed_col = "qed" if "qed" in df.columns else ("QED" if "QED" in df.columns else None)
    sa_col = "sa_score" if "sa_score" in df.columns else None
    lig_col = "ligand_id" if "ligand_id" in df.columns else None

    rows = []
    for _, row in df.iterrows():
        dock = _as_float(row.get("dock_score"))
        if dock is None or not _is_valid_dock_score(dock, dock_valid_max=dock_valid_max):
            continue
        if dock_abs_max is not None and abs(dock) > dock_abs_max:
            continue
        pose_rel = str(row.get(pose_col, "")).strip()
        if not pose_rel or pose_rel.lower() in {"nan", "none"}:
            continue
        rows.append(
            {
                "ligand_id": str(row.get(lig_col, "")) if lig_col else "",
                "dock_score": float(dock),
                "qed": _as_float(row.get(qed_col)) if qed_col else None,
                "sa_score": _as_float(row.get(sa_col)) if sa_col else None,
                "smiles": str(row.get(smiles_col, "")) if smiles_col else "",
                "dock_pose": pose_rel,
            }
        )
    if not rows:
        return 0
    rows.sort(key=lambda r: r["dock_score"])
    top_n = max(1, int(len(rows) * top_frac))
    top_rows = rows[:top_n]

    out_dir = run_dir / "top10pct_poses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "top10pct_poses.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ligand_id",
                "dock_score",
                "qed",
                "sa_score",
                "smiles",
                "dock_pose",
                "copied_pose",
                "status",
            ],
        )
        writer.writeheader()
        for entry in top_rows:
            pose_rel = entry["dock_pose"]
            src = dataset_root / pose_rel
            dest = out_dir / Path(pose_rel).name
            status = "missing"
            if src.exists():
                shutil.copy2(src, dest)
                status = "copied"
            entry["copied_pose"] = str(dest.relative_to(run_dir)) if dest.exists() else ""
            entry["status"] = status
            writer.writerow(entry)
    return len(top_rows)


def mark_abnormal_dock(
    csv_path: str | Path,
    dock_valid_max: float | None = 0.0,
    dock_abs_max: float | None = None,
    fail_col: str = "dock_fail",
    reason_col: str = "dock_fail_reason",
) -> dict:
    df = _read_smiles_csv(str(csv_path))
    if df.empty or "dock_score" not in df.columns:
        return {"total": 0, "flagged": 0}
    dock = np.asarray(pd.to_numeric(df["dock_score"], errors="coerce"), dtype=float)
    finite = np.isfinite(dock)
    mask_bad = np.zeros_like(finite, dtype=bool)
    if dock_valid_max is not None:
        mask_bad |= finite & (dock > float(dock_valid_max))
    if dock_abs_max is not None:
        mask_bad |= finite & (np.abs(dock) > float(dock_abs_max))
    flagged = int(mask_bad.sum())
    if flagged == 0:
        return {"total": int(len(df)), "flagged": 0}
    df.loc[mask_bad, "dock_score"] = np.nan
    if fail_col not in df.columns:
        df[fail_col] = ""
    if reason_col not in df.columns:
        df[reason_col] = ""
    df.loc[mask_bad, fail_col] = "1"
    df.loc[mask_bad, reason_col] = "abnormal_score"
    df.to_csv(csv_path, index=False)
    return {"total": int(len(df)), "flagged": flagged}
