from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import shlex
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from shutil import which
from typing import Sequence, Tuple, List
from collections import defaultdict
from urllib import request as urllib_request, error as urllib_error
import re

import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem

from data.pocket_dataset import (
    PocketDatasetPaths,
    _ensure_receptor_pdbqt,
    _extract_pocket_atoms,
    _load_ligand_conformer,
    _parse_protein_atoms,
    _pose_is_valid,
    _pose_to_sdf,
    _read_pdb_atoms,
    _run_vina_binary,
    _smiles_to_3d_mol,
    _vina_box_from_coords,
    _write_pdbqt_from_rdkit,
    _write_pre_sdf_from_mol,
    _write_pocket_pdb_from_atoms,
)
from mobo.io_utils import _is_valid_dock_score, _read_smiles_csv
from mobo.smiles_utils import canonicalize_smiles_noh


def resolve_vina_executable(
    vina_executable: str | None,
    dataset_root: str | Path,
    backend: str = "vina",
) -> str | Path:
    if vina_executable:
        return vina_executable

    root = Path(dataset_root).resolve()
    repo_bin = root / "bin"

    if backend == "vina_cuda":
        candidates = [
            repo_bin / "vina-cuda1.1.exe",
            repo_bin / "vina-cuda1.1",
        ]
    else:
        candidates = [
            repo_bin / "vina_1.2.7_win.exe",
            repo_bin / "vina.exe",
            repo_bin / "vina",
        ]

    for c in candidates:
        if c.exists():
            return c

    return "vina-cuda1.1.exe" if backend == "vina_cuda" else "vina"


def _subprocess_env_for_vina(exec_path: str | Path) -> dict:
    """
    为 Windows 下的 Vina / Vina-CUDA 运行时补 PATH。
    这样 bin 目录下的 Boost DLL 能被找到。
    """
    env = os.environ.copy()
    exec_path = Path(exec_path).resolve()
    bin_dir = str(exec_path.parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def _run_vina_cuda_binary(
    vina_exec: str | Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: List[float],
    box_size: List[float],
    thread: int = 8192,
    search_depth: int = 8,
    rilc_bfgs: int = 1,
    seed: int = 0,
    n_poses: int = 1,
    log_file: Path | None = None,
) -> Tuple[float, str]:
    exec_path = Path(vina_exec)
    if not exec_path.exists():
        resolved = which(str(vina_exec))
        if resolved is None:
            raise FileNotFoundError(f"Vina-CUDA executable not found: {vina_exec}")
        exec_path = Path(resolved)

    fd, pose_name = tempfile.mkstemp(suffix=".pdbqt")
    os.close(fd)
    pose_file = Path(pose_name)

    if log_file is None:
        fd_log, log_name = tempfile.mkstemp(suffix=".txt")
        os.close(fd_log)
        log_file = Path(log_name)

    cmd = [
        str(exec_path),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.3f}",
        "--size_y", f"{box_size[1]:.3f}",
        "--size_z", f"{box_size[2]:.3f}",
        "--thread", str(int(thread)),
        "--search_depth", str(int(search_depth)),
        "--rilc_bfgs", str(int(rilc_bfgs)),
        "--seed", str(int(seed)),
        "--num_modes", str(max(int(n_poses), 1)),
        "--out", str(pose_file),
        "--log", str(log_file),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_subprocess_env_for_vina(exec_path),
    )

    if result.returncode != 0:
        pose_file.unlink(missing_ok=True)
        log_text = result.stdout + "\n" + result.stderr
        if log_file.exists():
            try:
                log_text += "\n[log_file]\n" + log_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
        raise RuntimeError(f"Vina-CUDA docking failed:\n{log_text}")

    score = None
    text_sources = []
    if log_file.exists():
        try:
            text_sources.append(log_file.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    if result.stdout:
        text_sources.append(result.stdout)

    for text in text_sources:
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    score = float(parts[1])
                    break
                except ValueError:
                    continue
        if score is not None:
            break

    if score is None:
        score = float("nan")

    pose_data = pose_file.read_text(encoding="utf-8", errors="ignore")
    pose_file.unlink(missing_ok=True)
    return score, pose_data


def _resolve_receptor_pdbqt_fast(
    paths: PocketDatasetPaths,
    meeko_allow_bad_res: bool = False,
    meeko_default_altloc: str | None = None,
) -> Path:
    """
    优先使用数据集内现成的 pocket.pdbqt。
    只有在缺失时，才回退到从 pocket.pdb 生成。
    """
    pocket_pdbqt = paths.pocket_pdb.with_suffix(".pdbqt")
    if pocket_pdbqt.exists():
        return pocket_pdbqt

    _ensure_receptor_pdbqt(
        paths.pocket_pdb,
        pocket_pdbqt,
        meeko_allow_bad_res=meeko_allow_bad_res,
        meeko_default_altloc=meeko_default_altloc,
    )
    return pocket_pdbqt


def _load_reference_ligand_fast(paths: PocketDatasetPaths) -> Chem.Mol | None:
    """
    优先从 reference_ligand.sdf 读取参考配体。
    若失败，再回退到原逻辑。
    """
    if paths.reference_ligand is None:
        return None

    ref_path = Path(paths.reference_ligand)
    if ref_path.suffix.lower() == ".sdf" and ref_path.exists():
        suppl = Chem.SDMolSupplier(str(ref_path), removeHs=False)
        for mol in suppl:
            if mol is not None and mol.GetNumConformers():
                return mol

    return _load_ligand_conformer(ref_path)


def _parse_ref_token(ref_token: str) -> dict:
    """Parse reference-ligand selector.

    Supported forms:
    - "LIG"
    - "LIG:A"
    - "LIG:A:401"
    - "A:LIG:401"
    - "LIG:401"
    """
    token = str(ref_token or "").strip()
    if not token:
        raise ValueError("Empty ref_resname/ref_token.")
    parts = [p.strip() for p in token.split(":") if p.strip()]
    out = {"resname": None, "chain": None, "resid": None}
    if len(parts) == 1:
        out["resname"] = parts[0]
    elif len(parts) == 2:
        # Prefer LIG:A, but also accept LIG:401
        if parts[1].lstrip("+-").isdigit():
            out["resname"] = parts[0]
            out["resid"] = int(parts[1])
        else:
            out["resname"] = parts[0]
            out["chain"] = parts[1]
    elif len(parts) >= 3:
        # Accept both LIG:A:401 and A:LIG:401
        if len(parts[0]) <= 2 and len(parts[1]) >= 2:
            out["chain"] = parts[0]
            out["resname"] = parts[1]
            if parts[2].lstrip("+-").isdigit():
                out["resid"] = int(parts[2])
        else:
            out["resname"] = parts[0]
            out["chain"] = parts[1]
            if parts[2].lstrip("+-").isdigit():
                out["resid"] = int(parts[2])
    if out["resname"] is None:
        raise ValueError(f"Failed to parse reference token: {ref_token!r}")
    return out


def _extract_reference_ligand_sdf(
    protein_pdb: Path,
    ref_token: str,
    out_sdf: Path,
    default_altloc: str | None = None,
) -> bool:
    """Extract a reference ligand from protein.pdb into SDF.

    This mirrors the older mobo_qpmhi preparation logic: only protein.pdb is
    required in the runtime dataset root, and reference_ligand.sdf can be
    derived on demand from a configurable selector.
    """
    protein_pdb = Path(protein_pdb)
    out_sdf = Path(out_sdf)
    if not protein_pdb.exists():
        raise FileNotFoundError(protein_pdb)

    sel = _parse_ref_token(ref_token)
    lines = protein_pdb.read_text(encoding="utf-8", errors="ignore").splitlines()

    groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for line in lines:
        if not (line.startswith("HETATM") or line.startswith("ATOM  ")):
            continue
        resname = line[17:20].strip()
        chain = line[21].strip()
        resid = line[22:26].strip()
        icode = line[26].strip()
        altloc = line[16].strip()

        if resname in {"HOH", "WAT"}:
            continue
        if sel["resname"] and resname != sel["resname"]:
            continue
        if sel["chain"] and chain != sel["chain"]:
            continue
        if sel["resid"] is not None and resid != str(sel["resid"]):
            continue
        if altloc and default_altloc and altloc != str(default_altloc).strip():
            continue
        key = (resname, chain, resid, icode)
        groups[key].append(line)

    if not groups:
        return False

    # Deterministic selection: exact match on resid/chain preferred; otherwise first sorted group.
    def _group_key(item):
        (resname, chain, resid, icode), atom_lines = item
        return (resname, chain, int(resid) if resid.lstrip("+-").isdigit() else 10**9, icode, len(atom_lines))

    chosen_key, chosen_lines = sorted(groups.items(), key=_group_key)[0]
    pdb_block = "\n".join(chosen_lines) + "\nEND\n"

    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=True, proximityBonding=True)
    if mol is None:
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False, proximityBonding=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return False

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_sdf))
    try:
        writer.write(mol)
    finally:
        writer.close()
    return out_sdf.exists() and out_sdf.stat().st_size > 0


def _default_oracle_cache_root() -> Path:
    return Path(__file__).resolve().parents[1] / ".oracle_cache"


def _hash_json(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seedless_conformer_params(
    *,
    canonical_smiles: str,
    max_attempts: int,
    num_confs: int,
    max_opt_iters: int,
    optimize: bool,
    prefer_mmff: bool,
) -> dict:
    return {
        "canonical_smiles": str(canonical_smiles),
        "max_attempts": int(max_attempts),
        "num_confs": int(num_confs),
        "max_opt_iters": int(max_opt_iters),
        "optimize": bool(optimize),
        "prefer_mmff": bool(prefer_mmff),
    }


def _conformer_cache_key_from_params(params: dict) -> str:
    payload = dict(params)
    payload.pop("seed", None)
    return _hash_json(payload)


def _docking_cache_key(*, canonical_smiles: str, conformer_key: str, oracle_params: dict) -> str:
    return _hash_json({
        "canonical_smiles": str(canonical_smiles),
        "conformer_key": str(conformer_key),
        "oracle": dict(oracle_params),
    })


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


UNIDOCK_MAX_ACTIVE_TORSIONS = 48
_ACTIVE_TORSION_RE = re.compile(r"REMARK\s+(\d+)\s+active torsions:")


def _reference_ligand_box(ref_mol: Chem.Mol, padding: float = 4.0) -> tuple[list[float], list[float]]:
    if ref_mol is None or not ref_mol.GetNumConformers():
        raise ValueError("Reference ligand has no conformer for docking box construction.")
    coords = np.asarray(ref_mol.GetConformer().GetPositions(), dtype=np.float64)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = ((mins + maxs) * 0.5).tolist()
    raw_box_size = np.maximum(maxs - mins, 1.0)
    box_size = (raw_box_size + float(padding)).tolist()
    return [float(v) for v in center], [float(v) for v in box_size]


def _extract_active_torsions_from_pdbqt(path: Path) -> int | None:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    match = _ACTIVE_TORSION_RE.search(text)
    if match is not None:
        return int(match.group(1))
    for line in text.splitlines():
        if line.startswith("TORSDOF"):
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1].lstrip("+-").isdigit():
                return int(parts[1])
    return None


def _validate_unidock_ligand_pdbqt(path: Path, max_active_torsions: int = UNIDOCK_MAX_ACTIVE_TORSIONS) -> None:
    torsions = _extract_active_torsions_from_pdbqt(path)
    if torsions is not None and torsions > int(max_active_torsions):
        raise RuntimeError(
            f"Ligand {path} exceeds Uni-Dock active torsion limit: "
            f"torsions={torsions} > max={int(max_active_torsions)}"
        )


def _ensure_oracle_cache(cache_root: Path) -> sqlite3.Connection:
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "conformers").mkdir(parents=True, exist_ok=True)
    (cache_root / "dockings").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_root / "cache.sqlite3"))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conformer_cache (
            conformer_key TEXT PRIMARY KEY,
            canonical_smiles TEXT NOT NULL,
            params_json TEXT NOT NULL,
            pre_sdf_rel TEXT NOT NULL,
            pdbqt_rel TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS docking_cache (
            docking_key TEXT PRIMARY KEY,
            canonical_smiles TEXT NOT NULL,
            target_key TEXT NOT NULL,
            conformer_key TEXT NOT NULL,
            oracle_json TEXT NOT NULL,
            vina_score REAL NOT NULL,
            pose_sdf_rel TEXT NOT NULL,
            meta_rel TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS failure_cache (
            failure_key TEXT PRIMARY KEY,
            failure_scope TEXT NOT NULL,
            canonical_smiles TEXT NOT NULL,
            target_key TEXT,
            conformer_key TEXT,
            docking_key TEXT,
            stage TEXT NOT NULL,
            retryable INTEGER NOT NULL,
            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def _cache_fetch_conformer(conn: sqlite3.Connection, cache_root: Path, conformer_key: str) -> dict | None:
    row = conn.execute(
        "SELECT canonical_smiles, params_json, pre_sdf_rel, pdbqt_rel FROM conformer_cache WHERE conformer_key = ?",
        (conformer_key,),
    ).fetchone()
    if row is None:
        return None
    pre_sdf = cache_root / row[2]
    pdbqt = cache_root / row[3]
    if not pre_sdf.exists() or not pdbqt.exists():
        return None
    return {
        "canonical_smiles": row[0],
        "params_json": row[1],
        "pre_sdf": pre_sdf,
        "pdbqt": pdbqt,
    }


def _cache_store_conformer(
    conn: sqlite3.Connection,
    cache_root: Path,
    conformer_key: str,
    canonical_smiles: str,
    params: dict,
    pre_sdf_src: Path,
    pdbqt_src: Path,
) -> dict:
    conf_dir = cache_root / "conformers" / conformer_key
    conf_dir.mkdir(parents=True, exist_ok=True)
    pre_sdf_dst = conf_dir / "ligand.pre.sdf"
    pdbqt_dst = conf_dir / "ligand.pdbqt"
    shutil.copy2(pre_sdf_src, pre_sdf_dst)
    shutil.copy2(pdbqt_src, pdbqt_dst)
    conn.execute(
        """
        INSERT OR REPLACE INTO conformer_cache(
            conformer_key, canonical_smiles, params_json, pre_sdf_rel, pdbqt_rel, created_at
        )
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            conformer_key,
            canonical_smiles,
            json.dumps(params, sort_keys=True),
            os.path.relpath(str(pre_sdf_dst), str(cache_root)),
            os.path.relpath(str(pdbqt_dst), str(cache_root)),
        ),
    )
    conn.commit()
    return {"pre_sdf": pre_sdf_dst, "pdbqt": pdbqt_dst}


def _cache_fetch_docking(conn: sqlite3.Connection, cache_root: Path, docking_key: str) -> dict | None:
    row = conn.execute(
        "SELECT canonical_smiles, target_key, conformer_key, oracle_json, vina_score, pose_sdf_rel, meta_rel FROM docking_cache WHERE docking_key = ?",
        (docking_key,),
    ).fetchone()
    if row is None:
        return None
    pose_path = cache_root / row[5]
    meta_path = cache_root / row[6]
    if not pose_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    pose_format = str(meta.get("pose_format", pose_path.suffix.lstrip(".").lower())).lower()
    if pose_format == "sdf":
        if not _pose_is_valid(pose_path):
            return None
    elif pose_format not in {"none", "score_only"} and pose_path.stat().st_size == 0:
        return None

    return {
        "canonical_smiles": row[0],
        "target_key": row[1],
        "conformer_key": row[2],
        "oracle_json": row[3],
        "vina_score": float(row[4]),
        "pose_sdf": pose_path,
        "pose_path": pose_path,
        "pose_format": pose_format,
        "meta_path": meta_path,
    }


def _cache_store_docking(
    conn: sqlite3.Connection,
    cache_root: Path,
    docking_key: str,
    canonical_smiles: str,
    target_key: str,
    conformer_key: str,
    oracle_params: dict,
    vina_score: float,
    pose_sdf_src: Path | None,
    meta: dict,
    pose_file_name: str | None = None,
) -> dict:
    dock_dir = cache_root / "dockings" / docking_key
    dock_dir.mkdir(parents=True, exist_ok=True)
    pose_file_name = str(pose_file_name).strip() if pose_file_name is not None else ""
    if pose_sdf_src is None:
        if not pose_file_name:
            pose_file_name = "score_only.txt"
    else:
        if not pose_file_name:
            pose_suffix = pose_sdf_src.suffix if pose_sdf_src.suffix else ".sdf"
            pose_file_name = f"pose{pose_suffix}"
    pose_sdf_dst = dock_dir / pose_file_name
    meta_dst = dock_dir / "meta.json"
    if pose_sdf_src is None:
        pose_sdf_dst.write_text("score_only\n", encoding="utf-8")
    else:
        shutil.copy2(pose_sdf_src, pose_sdf_dst)
    meta_dst.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    conn.execute(
        """
        INSERT OR REPLACE INTO docking_cache(
            docking_key, canonical_smiles, target_key, conformer_key, oracle_json, vina_score, pose_sdf_rel, meta_rel, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            docking_key,
            canonical_smiles,
            target_key,
            conformer_key,
            json.dumps(oracle_params, sort_keys=True),
            float(vina_score),
            os.path.relpath(str(pose_sdf_dst), str(cache_root)),
            os.path.relpath(str(meta_dst), str(cache_root)),
        ),
    )
    conn.commit()
    return {"pose_sdf": pose_sdf_dst, "meta_path": meta_dst}


def _cache_fetch_failure(conn: sqlite3.Connection, failure_key: str) -> dict | None:
    row = conn.execute(
        "SELECT failure_scope, canonical_smiles, target_key, conformer_key, docking_key, stage, retryable, error_type, error_message FROM failure_cache WHERE failure_key = ?",
        (failure_key,),
    ).fetchone()
    if row is None:
        return None
    return {
        "failure_scope": row[0],
        "canonical_smiles": row[1],
        "target_key": row[2],
        "conformer_key": row[3],
        "docking_key": row[4],
        "stage": row[5],
        "retryable": bool(row[6]),
        "error_type": row[7],
        "error_message": row[8],
    }


def _cache_store_failure(
    conn: sqlite3.Connection,
    failure_key: str,
    failure_scope: str,
    canonical_smiles: str,
    target_key: str | None,
    conformer_key: str | None,
    docking_key: str | None,
    stage: str,
    retryable: bool,
    exc: Exception,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO failure_cache(
            failure_key, failure_scope, canonical_smiles, target_key, conformer_key, docking_key,
            stage, retryable, error_type, error_message, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            failure_key,
            failure_scope,
            canonical_smiles,
            target_key,
            conformer_key,
            docking_key,
            stage,
            1 if retryable else 0,
            type(exc).__name__,
            str(exc),
        ),
    )
    conn.commit()


def _cache_delete_failure(conn: sqlite3.Connection, failure_key: str) -> None:
    conn.execute("DELETE FROM failure_cache WHERE failure_key = ?", (failure_key,))
    conn.commit()


def _classify_oracle_failure(stage: str, exc: Exception) -> bool:
    if stage == "conformer":
        return False
    if stage == "pose":
        return False
    if stage == "docking":
        return True
    return True


def _windows_path_to_wsl(path: str | Path) -> str:
    s = str(path)
    if s.startswith("/"):
        return s
    try:
        wp = PureWindowsPath(s)
    except Exception:
        return s
    drive = wp.drive.rstrip(":")
    if drive:
        parts = list(wp.parts)[1:]
        tail = "/".join(part.strip("\\/") for part in parts if part and part not in {wp.drive})
        return f"/mnt/{drive.lower()}/{tail}" if tail else f"/mnt/{drive.lower()}"
    return s.replace("\\", "/")


def _load_dataset_box_override(dataset_root: Path) -> tuple[tuple[float, float, float], tuple[float, float, float], str] | None:
    cfg_path = Path(dataset_root) / "box_config.json"
    if not cfg_path.exists():
        return None
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    center_raw = raw.get("center")
    box_raw = raw.get("box_size")
    if not isinstance(center_raw, list) or len(center_raw) != 3:
        raise ValueError(f"Invalid center in {cfg_path}")
    if not isinstance(box_raw, list) or len(box_raw) != 3:
        raise ValueError(f"Invalid box_size in {cfg_path}")
    center = tuple(float(v) for v in center_raw)
    box_size = tuple(float(v) for v in box_raw)
    source = str(raw.get("source", "box_config.json"))
    return center, box_size, source


def _load_service_asset_rel_root(dataset_root: Path) -> str | None:
    manifest_path = Path(dataset_root) / "service_asset_manifest.json"
    if not manifest_path.exists():
        return None
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    rel_root = str(raw.get("service_asset_rel_root", "")).strip().replace("\\", "/")
    return rel_root or None


def _join_posix_path(root: str, rel_path: str) -> str:
    base = str(root).strip()
    rel = str(rel_path).strip().replace("\\", "/")
    if not base:
        return rel
    if rel.startswith("/"):
        return rel
    return str(PurePosixPath(base) / rel)


def _coerce_optional_path_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _resolve_prebuilt_ligand_assets(
    dataset_root: Path,
    row_data,
    *,
    unidock_service_local_input_root: str | None,
) -> dict | None:
    rel_sdf = _coerce_optional_path_text(row_data.get("pre_sdf_rel", ""))
    rel_pdbqt = _coerce_optional_path_text(row_data.get("pre_pdbqt_rel", ""))
    service_rel_sdf = _coerce_optional_path_text(row_data.get("service_rel_pre_sdf", ""))
    service_abs_sdf = _coerce_optional_path_text(row_data.get("unidock_local_pre_sdf", ""))
    if not rel_sdf:
        return None
    local_sdf = (Path(dataset_root) / rel_sdf).resolve()
    if not local_sdf.exists():
        raise FileNotFoundError(f"Missing prebuilt ligand SDF: {local_sdf}")
    local_pdbqt = None
    if rel_pdbqt:
        candidate = (Path(dataset_root) / rel_pdbqt).resolve()
        if candidate.exists():
            local_pdbqt = candidate

    service_sdf = ""
    if service_abs_sdf.startswith("/"):
        service_sdf = service_abs_sdf
    elif service_rel_sdf and unidock_service_local_input_root:
        service_sdf = _join_posix_path(unidock_service_local_input_root, service_rel_sdf)

    return {
        "local_sdf": local_sdf,
        "local_pdbqt": local_pdbqt,
        "service_sdf": service_sdf,
    }


def _wsl_path_to_windows_unc(wsl_path: str | Path, wsl_distro: str | None = None) -> Path:
    p = str(wsl_path).strip()
    if not p:
        raise ValueError("empty WSL path")

    distro = str(wsl_distro).strip() if wsl_distro else "Ubuntu"

    if p.startswith("\\\\"):
        return Path(p)

    if p.startswith("/"):
        parts = [seg for seg in p.split("/") if seg]
        unc = "\\\\wsl.localhost\\" + distro
        if parts:
            unc += "\\" + "\\".join(parts)
        return Path(unc)

    raise ValueError(f"not a recognized WSL path: {wsl_path}")


def _read_unidock_output_file(
    output_path: str,
    wsl_distro: str | None = None,
    retries: int = 5,
    retry_delay: float = 0.2,
) -> str:
    """
    读取 Uni-Dock 服务返回的输出文件内容。
    先尝试通过 \\\\wsl.localhost\\<distro>\\... 读取；
    如果失败，则回退到 `wsl.exe -d <distro> bash -lc "cat <path>"`。
    注意：虽然字段名叫 output_pdbqt_path，但这里既可能读取 .pdbqt，也可能读取 .sdf。
    """
    last_exc: Exception | None = None

    local_path = Path(str(output_path))
    try:
        for _ in range(max(1, int(retries))):
            if local_path.exists():
                return local_path.read_text(encoding="utf-8", errors="ignore")
            time.sleep(float(retry_delay))
    except Exception as exc:
        last_exc = exc

    try:
        unc_path = _wsl_path_to_windows_unc(output_path, wsl_distro)
        for _ in range(max(1, int(retries))):
            if unc_path.exists():
                return unc_path.read_text(encoding="utf-8", errors="ignore")
            time.sleep(float(retry_delay))
    except Exception as exc:
        last_exc = exc

    try:
        bash_cmd = f"cat {shlex.quote(str(output_path))}"
        if wsl_distro:
            cmd = ["wsl.exe", "-d", str(wsl_distro), "bash", "-lc", bash_cmd]
        else:
            cmd = ["wsl.exe", "bash", "-lc", bash_cmd]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout

        err = proc.stderr.strip() if proc.stderr else "unknown error"
        raise FileNotFoundError(
            f"Failed to read WSL output file via wsl.exe: {output_path} | {err}"
        )
    except Exception as exc:
        if last_exc is not None:
            raise FileNotFoundError(
                f"Uni-Dock pose path not accessible via UNC or wsl.exe: {output_path}"
            ) from exc
        raise


def _materialize_unidock_pose_file(
    output_path: str,
    wsl_distro: str | None = None,
) -> tuple[str, str]:
    """Read Uni-Dock output as raw text without pose-format conversion."""
    suffix = Path(str(output_path)).suffix.lower()
    raw_text = _read_unidock_output_file(
        output_path=str(output_path),
        wsl_distro=wsl_distro,
    )
    if not raw_text.strip():
        raise RuntimeError(f"Empty Uni-Dock output: {output_path}")
    if suffix not in {".sdf", ".pdbqt"}:
        raise RuntimeError(f"Unsupported Uni-Dock output suffix: {output_path}")
    return raw_text, suffix


def _run_unidock_service_batch(
    service_url: str,
    target_key: str,
    pocket_pdbqt: Path,
    center: List[float],
    box_size: List[float],
    requests_batch: list[dict],
    *,
    service_receptor_path: str | None = None,
    scoring: str = "vina",
    search_mode: str = "balance",
    num_modes: int = 1,
    timeout_sec: int = 3600,
) -> list[dict]:
    if not service_url:
        raise ValueError("unidock_service_url is required when docking_backend='unidock_service'.")
    payload = {
        "target": {
            "target_key": target_key,
            "receptor_path": str(service_receptor_path).strip() if service_receptor_path else _windows_path_to_wsl(pocket_pdbqt),
            "center": [float(v) for v in center],
            "box_size": [float(v) for v in box_size],
        },
        "params": {
            "scoring": str(scoring),
            "search_mode": str(search_mode) if search_mode is not None else None,
            "num_modes": int(num_modes),
            "timeout_sec": int(timeout_sec),
            "verbosity": 1,
        },
        "ligands": [
            {
                "ligand_id": str(item["lig_id"]),
                "docking_key": str(item["docking_key"]),
                "ligand_sdf_path": _windows_path_to_wsl(item["ligand_sdf"]),
            }
            for item in requests_batch
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url=service_url.rstrip("/") + "/dock/batch",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # The service uses timeout_sec for each subprocess. A batch request runs ligand
    # preparation plus three Uni-Dock rounds, so the HTTP wait budget must be
    # longer than a single subprocess timeout.
    http_timeout_sec = max(int(timeout_sec) * 4, int(timeout_sec) + 600, 60)
    try:
        with urllib_request.urlopen(req, timeout=http_timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"Uni-Dock service HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Uni-Dock service request failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Invalid Uni-Dock service JSON response: {raw[:1000]}") from exc

    if not isinstance(data, dict) or "results" not in data:
        raise RuntimeError(f"Unexpected Uni-Dock service response: {data}")
    return list(data.get("results", []))


def _run_unidock_service_single(
    service_url: str,
    target_key: str,
    pocket_pdbqt: Path,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    ligand_id: str,
    docking_key: str,
    ligand_sdf: Path,
    lig_mol: Chem.Mol,
    *,
    scoring: str,
    search_mode: str,
    num_modes: int,
    timeout_sec: int,
    unidock_service_wsl_distro: str | None = None,
    return_pose_sdf: bool = True,
    service_receptor_path: str | None = None,
) -> tuple[float, str | None]:
    results = _run_unidock_service_batch(
        service_url=service_url,
        target_key=target_key,
        pocket_pdbqt=pocket_pdbqt,
        center=center,
        box_size=box_size,
        service_receptor_path=service_receptor_path,
        requests_batch=[
            {
                "lig_id": ligand_id,
                "docking_key": docking_key,
                "ligand_sdf": ligand_sdf,
            }
        ],
        scoring=scoring,
        search_mode=search_mode,
        num_modes=num_modes,
        timeout_sec=timeout_sec,
    )

    if not results:
        raise RuntimeError("Uni-Dock service returned empty result set.")

    res = results[0]
    if str(res.get("status", "")).lower() != "ok":
        raise RuntimeError(str(res.get("error_message", "unidock service failed")))

    score = float(res["score"])
    if not return_pose_sdf:
        return score, None

    output_path = res.get("output_path")
    if not output_path:
        raise RuntimeError("Uni-Dock service did not return output_path")

    pose_text, _ = _materialize_unidock_pose_file(
        output_path=str(output_path),
        wsl_distro=unidock_service_wsl_distro,
    )
    return score, pose_text


def run_oracle_docking(
    dataset_root: str,
    ligand_ids: Sequence[str],
    vina_executable: str | None = None,
    docking_backend: str = "vina",
    vina_cuda_thread: int = 8192,
    vina_cuda_search_depth: int = 8,
    vina_cuda_rilc_bfgs: int = 1,
    overwrite: bool = False,
    exhaustiveness: int = 8,
    pocket_radius: float = 10.0,
    confgen_max_attempts: int = 10,
    confgen_seed: int = 0,
    confgen_num_confs: int = 10,
    confgen_max_opt_iters: int = 200,
    confgen_optimize: bool = True,
    confgen_prefer_mmff: bool = True,
    meeko_allow_bad_res: bool = False,
    meeko_default_altloc: str | None = None,
    evaluate_reference: bool = True,
    cache_root: str | Path | None = None,
    ref_resname: str | None = None,
    unidock_service_url: str | None = None,
    unidock_service_wsl_distro: str | None = None,
    unidock_scoring: str = "vina",
    unidock_search_mode: str = "balance",
    unidock_num_modes: int = 1,
    unidock_timeout_sec: int = 3600,
    unidock_service_local_input_root: str | None = None,
    dock_valid_max: float | None = 0.0,
) -> dict:
    stats = {
        "attempted": 0,
        "docked": 0,
        "skipped": 0,
        "failed": 0,
        "cache_hit_conformer": 0,
        "cache_hit_docking": 0,
        "cache_hit_failure": 0,
    }
    if ligand_ids is None:
        ligand_ids = []

    root = Path(dataset_root).resolve()
    paths = PocketDatasetPaths.from_root(root, pocket_radius=pocket_radius)
    paths.docking_dir.mkdir(parents=True, exist_ok=True)

    ref_selector = str(ref_resname).strip() if ref_resname not in (None, "", "null") else None
    ref_lig_path = root / "reference_ligand.sdf"
    if ref_selector and paths.protein_pdb.exists() and not ref_lig_path.exists():
        try:
            ok = _extract_reference_ligand_sdf(
                paths.protein_pdb,
                ref_selector,
                ref_lig_path,
                default_altloc=meeko_default_altloc,
            )
            if ok:
                print(f"[oracle_ref] extracted reference_ligand.sdf from protein.pdb using ref_resname={ref_selector}")
                try:
                    paths.reference_ligand = ref_lig_path
                except Exception:
                    pass
            else:
                print(f"[oracle_ref] failed to extract reference ligand from protein.pdb for ref_resname={ref_selector}; falling back to existing oracle logic")
        except Exception as exc:
            print(f"[oracle_ref] extraction error for ref_resname={ref_selector}: {type(exc).__name__}: {exc}")

    cache_root_path = Path(cache_root).resolve() if cache_root is not None else _default_oracle_cache_root()
    cache_conn = _ensure_oracle_cache(cache_root_path)

    print(f"[oracle_cache] root={cache_root_path}")
    print(
        "[oracle_pocket] "
        f"protein={paths.protein_pdb} "
        f"reference_ligand={paths.reference_ligand if paths.reference_ligand else 'none'} "
        f"radius={paths.pocket_radius}"
    )

    ref_mol = None
    ref_pose_mol = None
    ref_pocket_score = None

    if evaluate_reference and paths.reference_ligand is not None:
        try:
            ref_mol = _load_reference_ligand_fast(paths)
            if ref_mol is None:
                raise RuntimeError(f"Failed to load reference ligand from {paths.reference_ligand}")

            if docking_backend == "unidock_service":
                print("[oracle_full] skipped for unidock_service; using reference_ligand geometry directly")
            else:
                protein_coords, _ = _read_pdb_atoms(paths.protein_pdb)
                protein_pdbqt = paths.protein_pdb.with_suffix(".pdbqt")
                _ensure_receptor_pdbqt(
                    paths.protein_pdb,
                    protein_pdbqt,
                    meeko_allow_bad_res=meeko_allow_bad_res,
                    meeko_default_altloc=meeko_default_altloc,
                )
                full_center, full_box = _vina_box_from_coords(protein_coords)
                print(f"[oracle_full] box_center={full_center} box_size={full_box}")

                full_target_key = _hash_json({
                    "protein_pdbqt_sha256": _hash_file(protein_pdbqt),
                    "center": [float(v) for v in full_center],
                    "box_size": [float(v) for v in full_box],
                    "scope": "full_reference",
                })

                ref_pdbqt_full = paths.docking_dir / "reference_ligand_full.pdbqt"
                _write_pdbqt_from_rdkit(ref_mol, ref_pdbqt_full)

                if docking_backend == "vina_cuda":
                    full_score, full_pose = _run_vina_cuda_binary(
                        resolve_vina_executable(vina_executable, dataset_root, backend="vina_cuda"),
                        protein_pdbqt,
                        ref_pdbqt_full,
                        center=full_center,
                        box_size=full_box,
                        thread=vina_cuda_thread,
                        search_depth=vina_cuda_search_depth,
                        rilc_bfgs=vina_cuda_rilc_bfgs,
                    )
                else:
                    full_score, full_pose = _run_vina_binary(
                        resolve_vina_executable(vina_executable, dataset_root, backend="vina"),
                        protein_pdbqt,
                        ref_pdbqt_full,
                        center=full_center,
                        box_size=full_box,
                        exhaustiveness=exhaustiveness,
                    )

                print(f"[oracle_full] reference_ligand_vina_score={float(full_score):.4f}")
                use_full_pose = np.isfinite(full_score) and float(full_score) < 0.0
                try:
                    sdf_block = _pose_to_sdf([full_pose], ref_mol)
                    pose_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=False)
                    if pose_mol is None or not pose_mol.GetNumConformers():
                        pose_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False, sanitize=True)
                    if pose_mol is not None and pose_mol.GetNumConformers() and use_full_pose:
                        ref_pose_mol = pose_mol
                        print("[oracle_full] reference_ligand_pose=ok")
                    else:
                        if not use_full_pose:
                            print("[oracle_full] reference_ligand_pose=ignored (invalid score)")
                        else:
                            print("[oracle_full] reference_ligand_pose=invalid")
                except Exception as exc:
                    raise RuntimeError("Failed to convert reference ligand pose to SDF.") from exc
        except Exception as exc:
            raise RuntimeError("Reference ligand docking against full protein failed.") from exc

        print("[oracle_pocket] extracting pocket from reference_ligand")
        pose_src = ref_mol
        if pose_src is None or not pose_src.GetNumConformers():
            pose_src = ref_pose_mol if ref_pose_mol is not None else ref_mol
            print("[oracle_pocket] pocket_pose_source=fallback_redocked" if pose_src is ref_pose_mol else "[oracle_pocket] pocket_pose_source=missing")
        else:
            print("[oracle_pocket] pocket_pose_source=reference_ligand_file")

        lig_coords = np.asarray(pose_src.GetConformer().GetPositions(), dtype=np.float64)
        ligand_atoms = (lig_coords, [atom.GetSymbol() for atom in pose_src.GetAtoms()])
        protein_full = _parse_protein_atoms(paths.protein_pdb, include_hetatm=False)
        pocket_atoms = _extract_pocket_atoms(protein_full, ligand_atoms, radius=paths.pocket_radius)
        if not pocket_atoms:
            raise ValueError("Pocket extraction produced empty atom set. Increase radius.")
        _write_pocket_pdb_from_atoms(pocket_atoms, paths.pocket_pdb)

    elif not paths.pocket_pdb.exists():
        print("[oracle_pocket] pocket.pdb missing; using full protein as pocket")
        protein_full = _parse_protein_atoms(paths.protein_pdb, include_hetatm=False)
        pocket_atoms = protein_full
        _write_pocket_pdb_from_atoms(pocket_atoms, paths.pocket_pdb)

    if ref_mol is None and paths.reference_ligand is not None:
        ref_mol = _load_reference_ligand_fast(paths)
        if ref_mol is None:
            raise RuntimeError(f"Failed to load reference ligand from {paths.reference_ligand}")

    receptor_pdbqt = paths.protein_pdb.with_suffix(".pdbqt")
    _ensure_receptor_pdbqt(
        paths.protein_pdb,
        receptor_pdbqt,
        meeko_allow_bad_res=meeko_allow_bad_res,
        meeko_default_altloc=meeko_default_altloc,
    )
    box_override = _load_dataset_box_override(paths.root)
    if box_override is not None:
        center, box_size, box_source = box_override
        raw_box_size = [float(v) for v in box_size]
        box_padding = 0.0
    else:
        pocket_coords, _ = _read_pdb_atoms(paths.pocket_pdb)
        box_padding = 4.0
        box_source = "pocket_bbox"
        box_source_mol = ref_mol if ref_mol is not None and ref_mol.GetNumConformers() else ref_pose_mol
        if box_source_mol is not None and box_source_mol.GetNumConformers():
            center, box_size = _reference_ligand_box(box_source_mol, padding=box_padding)
            raw_box_size = [float(v) - box_padding for v in box_size]
            box_source = "reference_ligand_bbox"
        else:
            center, raw_box_size = _vina_box_from_coords(pocket_coords)
            box_size = [float(v) + box_padding for v in raw_box_size]
        box_size = tuple(float(v) for v in box_size)
    print(f"[oracle_pocket] receptor_pdbqt={receptor_pdbqt}")
    print(f"[oracle_pocket] box_source={box_source}")
    print(f"[oracle_pocket] box_center={center} box_size={box_size} raw_box_size={raw_box_size}")

    service_asset_rel_root = _load_service_asset_rel_root(paths.root)
    service_receptor_path = None
    if unidock_service_local_input_root and service_asset_rel_root:
        service_receptor_path = _join_posix_path(
            unidock_service_local_input_root,
            f"{service_asset_rel_root}/protein.pdbqt",
        )

    target_key = _hash_json({
        "receptor_pdbqt_sha256": _hash_file(receptor_pdbqt),
        "center": [float(v) for v in center],
        "box_size": [float(v) for v in box_size],
        "pocket_radius": float(pocket_radius),
        "box_padding": box_padding,
    })

    if evaluate_reference and paths.reference_ligand is not None:
        try:
            if ref_mol is None:
                ref_mol = _load_reference_ligand_fast(paths)
            ref_pdbqt = paths.docking_dir / "reference_ligand.pdbqt"
            if docking_backend != "unidock_service":
                _write_pdbqt_from_rdkit(ref_mol, ref_pdbqt)

            if docking_backend == "vina_cuda":
                ref_score, _ = _run_vina_cuda_binary(
                    resolve_vina_executable(vina_executable, dataset_root, backend="vina_cuda"),
                    receptor_pdbqt,
                    ref_pdbqt,
                    center=center,
                    box_size=box_size,
                    thread=vina_cuda_thread,
                    search_depth=vina_cuda_search_depth,
                    rilc_bfgs=vina_cuda_rilc_bfgs,
                )
            elif docking_backend == "unidock_service":
                ref_score, _ = _run_unidock_service_single(
                    service_url=str(unidock_service_url or ""),
                    target_key=f"{target_key}::pocket_reference",
                    pocket_pdbqt=receptor_pdbqt,
                    center=center,
                    box_size=box_size,
                    ligand_id="REF_LIG_POCKET",
                    docking_key=_hash_json({
                        "ligand_id": "REF_LIG_POCKET",
                        "target_key": f"{target_key}::pocket_reference",
                        "backend": "unidock_service",
                        "center": [float(v) for v in center],
                        "box_size": [float(v) for v in box_size],
                        "scoring": str(unidock_scoring),
                        "search_mode": str(unidock_search_mode),
                        "num_modes": int(unidock_num_modes),
                    }),
                    ligand_sdf=paths.reference_ligand,
                    lig_mol=ref_mol,
                    scoring=str(unidock_scoring),
                    search_mode=str(unidock_search_mode),
                    num_modes=int(unidock_num_modes),
                    timeout_sec=int(unidock_timeout_sec),
                    unidock_service_wsl_distro=unidock_service_wsl_distro,
                    return_pose_sdf=False,
                )
            else:
                ref_score, _ = _run_vina_binary(
                    resolve_vina_executable(vina_executable, dataset_root, backend="vina"),
                    receptor_pdbqt,
                    ref_pdbqt,
                    center=center,
                    box_size=box_size,
                    exhaustiveness=exhaustiveness,
                )

            ref_pocket_score = float(ref_score) if np.isfinite(ref_score) else None
            print(f"[oracle_pocket] reference_ligand_vina_score={float(ref_score):.4f}")
        except Exception as exc:
            raise RuntimeError("Reference ligand docking against pocket failed.") from exc

    if not evaluate_reference and paths.reference_ligand is not None and not paths.pocket_pdb.exists():
        raise FileNotFoundError(
            "pocket.pdb missing while evaluate_reference=False. "
            "Enable evaluate_reference once or provide a prepared pocket.pdb."
        )

    if not ligand_ids:
        cache_conn.close()
        return stats

    df = _read_smiles_csv(str(paths.smiles_csv))
    if df.empty or "ligand_id" not in df.columns:
        cache_conn.close()
        return stats

    if "dock_pose" not in df.columns:
        df["dock_pose"] = ""
    if "dock_score" not in df.columns:
        df["dock_score"] = np.nan
    if "dock_cache_key" not in df.columns:
        df["dock_cache_key"] = ""
    if "dock_source" not in df.columns:
        df["dock_source"] = ""

    def _clear_failed_row(row_idx: int, source: str) -> None:
        df.at[row_idx, "dock_pose"] = ""
        df.at[row_idx, "dock_score"] = np.nan
        df.at[row_idx, "dock_cache_key"] = ""
        df.at[row_idx, "dock_source"] = source

    df["ligand_id"] = df["ligand_id"].astype(str)
    id_to_rows: dict[str, list[int]] = {}
    for idx, lig in enumerate(df["ligand_id"].tolist()):
        id_to_rows.setdefault(str(lig), []).append(idx)

    failed_path = paths.docking_dir / "failed.json"
    failed_map: dict[str, str] = {}

    vina_exec = None
    if docking_backend != "unidock_service":
        vina_exec = resolve_vina_executable(
            vina_executable,
            dataset_root,
            backend=docking_backend,
        )

    print(
        "[oracle_ligand_3d] "
        f"num_confs={confgen_num_confs} max_attempts={confgen_max_attempts} "
        f"max_opt_iters={confgen_max_opt_iters} optimize={confgen_optimize} "
        f"prefer_mmff={confgen_prefer_mmff}"
    )

    updated = False
    pending_batch: list[dict] = []

    for lig_id in tqdm(ligand_ids, desc="Dock", unit="lig", leave=True):
        lig_id = str(lig_id)
        row_indices = id_to_rows.get(lig_id)
        if not row_indices:
            raise KeyError(f"ligand_id '{lig_id}' not found in smiles.csv")

        for row_idx in row_indices:
            stats["attempted"] += 1
            row_data = df.loc[row_idx]
            smiles = str(row_data.get("smiles", "")).strip()
            if not smiles:
                raise ValueError(f"Empty SMILES for ligand_id '{lig_id}'")

            is_ref = str(row_data.get("is_reference", "")).strip() in {"1", "true", "True"} or lig_id.startswith("REF_LIG")

            score = float(row_data.get("dock_score", np.nan))

            if is_ref and ref_pocket_score is not None:
                df.at[row_idx, "dock_score"] = ref_pocket_score
                updated = True
                print(f"[oracle_target] reference_ligand_gt_score={ref_pocket_score:.4f}")

            failure_stage = "conformer"
            canonical_smiles = smiles
            conformer_key = ""
            docking_key = ""

            try:
                canonical_smiles = canonicalize_smiles_noh(smiles) or smiles
                conformer_params = _seedless_conformer_params(
                    canonical_smiles=canonical_smiles,
                    max_attempts=int(confgen_max_attempts),
                    num_confs=int(confgen_num_confs),
                    max_opt_iters=int(confgen_max_opt_iters),
                    optimize=bool(confgen_optimize),
                    prefer_mmff=bool(confgen_prefer_mmff),
                )
                conformer_key = _conformer_cache_key_from_params(conformer_params)

                oracle_params = {
                    "target_key": target_key,
                    "backend": str(docking_backend),
                    "exhaustiveness": int(exhaustiveness) if docking_backend == "vina" else None,
                    "vina_cuda_thread": int(vina_cuda_thread) if docking_backend == "vina_cuda" else None,
                    "vina_cuda_search_depth": int(vina_cuda_search_depth) if docking_backend == "vina_cuda" else None,
                    "vina_cuda_rilc_bfgs": int(vina_cuda_rilc_bfgs) if docking_backend == "vina_cuda" else None,
                    "unidock_service_url": str(unidock_service_url).strip() if docking_backend == "unidock_service" and unidock_service_url else None,
                    "unidock_scoring": str(unidock_scoring) if docking_backend == "unidock_service" else None,
                    "unidock_search_mode": str(unidock_search_mode) if docking_backend == "unidock_service" else None,
                    "unidock_num_modes": int(unidock_num_modes) if docking_backend == "unidock_service" else None,

                }

                docking_key = _docking_cache_key(
                    canonical_smiles=canonical_smiles,
                    conformer_key=conformer_key,
                    oracle_params=oracle_params,
                )

                if not overwrite:
                    cached_conf_failure = _cache_fetch_failure(cache_conn, conformer_key)
                    if cached_conf_failure is not None and not cached_conf_failure["retryable"]:
                        msg = f"{cached_conf_failure['error_type']}: {cached_conf_failure['error_message']}"
                        if is_ref:
                            raise RuntimeError(f"Reference ligand cached conformer failure: {lig_id} | {msg}")
                        failed_map[lig_id] = msg
                        _clear_failed_row(row_idx, "global_failure_cache:conformer")
                        updated = True
                        stats["failed"] += 1
                        stats["cache_hit_failure"] += 1
                        print(f"[oracle][failed-cache] {lig_id} | conformer | {msg}")
                        continue

                    cached_dock_failure = _cache_fetch_failure(cache_conn, docking_key)
                    if cached_dock_failure is not None and not cached_dock_failure["retryable"]:
                        msg = f"{cached_dock_failure['error_type']}: {cached_dock_failure['error_message']}"
                        if is_ref:
                            raise RuntimeError(f"Reference ligand cached docking failure: {lig_id} | {msg}")
                        failed_map[lig_id] = msg
                        _clear_failed_row(row_idx, f"global_failure_cache:{cached_dock_failure['stage']}")
                        updated = True
                        stats["failed"] += 1
                        stats["cache_hit_failure"] += 1
                        print(f"[oracle][failed-cache] {lig_id} | {cached_dock_failure['stage']} | {msg}")
                        continue

                if (
                    not overwrite
                    and not is_ref
                    and str(row_data.get("dock_cache_key", "")).strip() == docking_key
                    and _is_valid_dock_score(score, dock_valid_max=dock_valid_max)
                ):
                    stats["skipped"] += 1
                    continue

                if not overwrite and not is_ref:
                    cached_dock = _cache_fetch_docking(cache_conn, cache_root_path, docking_key)
                    if cached_dock is not None:
                        cached_score = float(cached_dock["vina_score"])
                        if not _is_valid_dock_score(cached_score, dock_valid_max=dock_valid_max):
                            failure_stage = "docking"
                            raise RuntimeError(f"Invalid cached docking score: {cached_score}")
                        df.at[row_idx, "dock_pose"] = ""
                        df.at[row_idx, "dock_score"] = cached_score
                        df.at[row_idx, "dock_cache_key"] = docking_key
                        df.at[row_idx, "dock_source"] = "global_docking_cache"
                        updated = True
                        stats["cache_hit_docking"] += 1
                        continue

                cached_conf = None
                lig_mol = None
                prebuilt_assets = _resolve_prebuilt_ligand_assets(
                    paths.root,
                    row_data,
                    unidock_service_local_input_root=unidock_service_local_input_root,
                )
                if prebuilt_assets is not None:
                    cached_conf = {
                        "pre_sdf": prebuilt_assets["local_sdf"],
                        "pdbqt": prebuilt_assets["local_pdbqt"],
                    }
                    lig_mol = _load_ligand_conformer(prebuilt_assets["local_sdf"])
                elif not overwrite:
                    cached_conf = _cache_fetch_conformer(cache_conn, cache_root_path, conformer_key)
                    if cached_conf is not None:
                        lig_mol = _load_ligand_conformer(cached_conf["pre_sdf"])
                        stats["cache_hit_conformer"] += 1

                if lig_mol is None:
                    failure_stage = "conformer"
                    lig_mol = _smiles_to_3d_mol(
                        smiles,
                        max_attempts=confgen_max_attempts,
                        seed=confgen_seed,
                        num_confs=confgen_num_confs,
                        max_opt_iters=confgen_max_opt_iters,
                        optimize=confgen_optimize,
                        prefer_mmff=confgen_prefer_mmff,
                    )
                    tmp_dir = cache_root_path / "_tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp_pre = tmp_dir / f"{conformer_key}.pre.sdf"
                    tmp_pdbqt = tmp_dir / f"{conformer_key}.pdbqt"
                    _write_pre_sdf_from_mol(lig_mol, tmp_pre)
                    _write_pdbqt_from_rdkit(lig_mol, tmp_pdbqt)
                    cached_conf = _cache_store_conformer(
                        cache_conn,
                        cache_root_path,
                        conformer_key=conformer_key,
                        canonical_smiles=canonical_smiles,
                        params=conformer_params,
                        pre_sdf_src=tmp_pre,
                        pdbqt_src=tmp_pdbqt,
                    )
                    tmp_pre.unlink(missing_ok=True)
                    tmp_pdbqt.unlink(missing_ok=True)

                if cached_conf is None:
                    raise RuntimeError(f"Missing conformer cache payload after generation: {conformer_key}")

                if docking_backend == "unidock_service":
                    ligand_sdf = cached_conf.get("pre_sdf")
                    ligand_pdbqt = cached_conf.get("pdbqt")
                    if ligand_sdf is None:
                        raise RuntimeError(f"Missing pre_sdf for Uni-Dock service input: {conformer_key}")
                    if ligand_pdbqt is None:
                        raise RuntimeError(f"Missing pdbqt for Uni-Dock service input: {conformer_key}")
                    _validate_unidock_ligand_pdbqt(Path(ligand_pdbqt))
                    pending_batch.append({
                        "lig_id": lig_id,
                        "row_idx": row_idx,
                        "smiles": smiles,
                        "canonical_smiles": canonical_smiles,
                        "conformer_key": conformer_key,
                        "docking_key": docking_key,
                        "oracle_params": oracle_params,
                        "ligand_sdf": prebuilt_assets["service_sdf"] if prebuilt_assets is not None and prebuilt_assets["service_sdf"] else ligand_sdf,
                        "lig_mol": lig_mol,
                        "is_ref": bool(is_ref),
                    })
                    continue

                failure_stage = "docking"
                if docking_backend == "vina_cuda":
                    vina_score, pose_text = _run_vina_cuda_binary(
                        vina_exec,
                        receptor_pdbqt,
                        cached_conf["pdbqt"],
                        center=center,
                        box_size=box_size,
                        thread=vina_cuda_thread,
                        search_depth=vina_cuda_search_depth,
                        rilc_bfgs=vina_cuda_rilc_bfgs,
                        seed=confgen_seed,
                        n_poses=1,
                    )
                else:
                    vina_score, pose_text = _run_vina_binary(
                        vina_exec,
                        receptor_pdbqt,
                        cached_conf["pdbqt"],
                        center=center,
                        box_size=box_size,
                        exhaustiveness=exhaustiveness,
                        seed=confgen_seed,
                        n_poses=1,
                    )

                if not _is_valid_dock_score(float(vina_score), dock_valid_max=dock_valid_max):
                    failure_stage = "docking"
                    raise RuntimeError(f"Invalid docking score returned: {float(vina_score)}")

                failure_stage = "pose"
                sdf_block = _pose_to_sdf([pose_text], lig_mol)
                tmp_pose = cache_root_path / "_tmp" / f"{docking_key}.sdf"
                tmp_pose.parent.mkdir(parents=True, exist_ok=True)
                tmp_pose.write_text(sdf_block, encoding="utf-8")

                meta = {
                    "smiles": smiles,
                    "canonical_smiles": canonical_smiles,
                    "vina_score": float(vina_score),
                    "target_key": target_key,
                    "conformer_key": conformer_key,
                    "docking_key": docking_key,
                    "center": list(center),
                    "box_size": list(box_size),
                    "backend": docking_backend,
                }

                _cache_store_docking(
                    cache_conn,
                    cache_root_path,
                    docking_key=docking_key,
                    canonical_smiles=canonical_smiles,
                    target_key=target_key,
                    conformer_key=conformer_key,
                    oracle_params=oracle_params,
                    vina_score=float(vina_score),
                    pose_sdf_src=tmp_pose,
                    meta=meta,
                )
                tmp_pose.unlink(missing_ok=True)
                _cache_delete_failure(cache_conn, conformer_key)
                _cache_delete_failure(cache_conn, docking_key)

                if not is_ref:
                    df.at[row_idx, "dock_pose"] = ""
                    df.at[row_idx, "dock_score"] = float(vina_score)
                    df.at[row_idx, "dock_cache_key"] = docking_key
                    df.at[row_idx, "dock_source"] = f"fresh_docking:{docking_backend}"
                    updated = True
                    stats["docked"] += 1
                else:
                    print(f"[oracle_target] reference_ligand_regen_score={float(vina_score):.4f}")

            except Exception as exc:
                retryable = _classify_oracle_failure(failure_stage, exc)
                failure_key = conformer_key if failure_stage == "conformer" else docking_key
                if not failure_key:
                    failure_key = _hash_json({
                        "canonical_smiles": canonical_smiles,
                        "target_key": target_key,
                        "failure_stage": failure_stage,
                        "raw_smiles": smiles,
                        "backend": docking_backend,
                    })

                _cache_store_failure(
                    cache_conn,
                    failure_key=failure_key,
                    failure_scope="conformer" if failure_stage == "conformer" else "docking",
                    canonical_smiles=canonical_smiles,
                    target_key=target_key,
                    conformer_key=conformer_key,
                    docking_key=docking_key,
                    stage=failure_stage,
                    retryable=retryable,
                    exc=exc,
                )
                failed_map[lig_id] = f"{type(exc).__name__}: {exc}"
                _clear_failed_row(row_idx, f"failed:{failure_stage}:{docking_backend}")
                updated = True
                stats["failed"] += 1
                if is_ref:
                    raise RuntimeError(f"Reference ligand docking failed: {lig_id}") from exc
                print(f"[oracle][failed] {lig_id} | {failure_stage} | {type(exc).__name__}: {exc}")
                continue

    if docking_backend == "unidock_service" and pending_batch:
        try:
            service_results = _run_unidock_service_batch(
                service_url=str(unidock_service_url or ""),
                target_key=target_key,
                pocket_pdbqt=receptor_pdbqt,
                center=center,
                box_size=box_size,
                service_receptor_path=service_receptor_path,
                requests_batch=pending_batch,
                scoring=str(unidock_scoring),
                search_mode=str(unidock_search_mode),
                num_modes=int(unidock_num_modes),
                timeout_sec=int(unidock_timeout_sec),
            )
            result_by_key = {str(item.get("docking_key", "")): item for item in service_results}
        except Exception as exc:
            result_by_key = {}
            batch_exc = exc
        else:
            batch_exc = None

        for item in pending_batch:
            lig_id = str(item["lig_id"])
            row_idx = int(item["row_idx"])
            is_ref = bool(item["is_ref"])
            canonical_smiles = str(item["canonical_smiles"])
            conformer_key = str(item["conformer_key"])
            docking_key = str(item["docking_key"])
            lig_mol = item["lig_mol"]
            oracle_params = item["oracle_params"]
            smiles = str(item["smiles"])
            res = result_by_key.get(docking_key)

            try:
                if batch_exc is not None:
                    raise RuntimeError(f"Uni-Dock service batch failed: {batch_exc}")
                if res is None:
                    raise RuntimeError(f"Uni-Dock service missing result for docking_key={docking_key}")
                if str(res.get("status", "")).lower() != "ok":
                    err = str(res.get("error_message", "service reported failure"))
                    raise RuntimeError(err)

                vina_score_raw = res.get("score")
                if vina_score_raw is None:
                    raise RuntimeError("Uni-Dock service returned null score")
                vina_score = float(vina_score_raw)
                if not _is_valid_dock_score(vina_score, dock_valid_max=dock_valid_max):
                    raise RuntimeError(f"Invalid docking score returned: {vina_score}")

                output_path = res.get("output_path")
                input_mode = str(res.get("input_mode", "sdf"))
                pose_format = "none"
                pose_file_name = "score_only.txt"
                tmp_pose = None
                pose_warning = None
                if output_path not in (None, ""):
                    try:
                        pose_text, pose_suffix = _materialize_unidock_pose_file(
                            output_path=str(output_path),
                            wsl_distro=unidock_service_wsl_distro,
                        )
                        tmp_pose = cache_root_path / "_tmp" / f"{docking_key}{pose_suffix}"
                        tmp_pose.parent.mkdir(parents=True, exist_ok=True)
                        tmp_pose.write_text(pose_text, encoding="utf-8")
                        pose_format = pose_suffix.lstrip(".")
                        pose_file_name = f"pose{tmp_pose.suffix}"
                    except FileNotFoundError as exc:
                        pose_warning = str(exc)
                else:
                    pose_warning = "Uni-Dock service did not return output_path"

                meta = {
                    "smiles": smiles,
                    "canonical_smiles": canonical_smiles,
                    "vina_score": float(vina_score),
                    "target_key": target_key,
                    "conformer_key": conformer_key,
                    "docking_key": docking_key,
                    "center": list(center),
                    "box_size": list(box_size),
                    "backend": docking_backend,
                    "unidock_output_path": None if output_path in (None, "") else str(output_path),
                    "unidock_input_mode": input_mode,
                    "pose_format": pose_format,
                    "pose_warning": pose_warning,
                }

                _cache_store_docking(
                    cache_conn,
                    cache_root_path,
                    docking_key=docking_key,
                    canonical_smiles=canonical_smiles,
                    target_key=target_key,
                    conformer_key=conformer_key,
                    oracle_params=oracle_params,
                    vina_score=float(vina_score),
                    pose_sdf_src=tmp_pose,
                    meta=meta,
                    pose_file_name=pose_file_name,
                )
                if tmp_pose is not None:
                    tmp_pose.unlink(missing_ok=True)
                _cache_delete_failure(cache_conn, conformer_key)
                _cache_delete_failure(cache_conn, docking_key)

                if pose_warning is not None:
                    print(f"[oracle][pose-missing] {lig_id} | docking | {pose_warning}")

                if not is_ref:
                    df.at[row_idx, "dock_pose"] = ""
                    df.at[row_idx, "dock_score"] = float(vina_score)
                    df.at[row_idx, "dock_cache_key"] = docking_key
                    source_suffix = input_mode if pose_warning is None else f"{input_mode}:score_only"
                    df.at[row_idx, "dock_source"] = f"fresh_docking:{docking_backend}:{source_suffix}"
                    updated = True
                    stats["docked"] += 1
                else:
                    print(f"[oracle_target] reference_ligand_regen_score={float(vina_score):.4f}")
            except Exception as exc:
                retryable = _classify_oracle_failure("docking", exc)
                _cache_store_failure(
                    cache_conn,
                    failure_key=docking_key,
                    failure_scope="docking",
                    canonical_smiles=canonical_smiles,
                    target_key=target_key,
                    conformer_key=conformer_key,
                    docking_key=docking_key,
                    stage="docking",
                    retryable=retryable,
                    exc=exc,
                )
                failed_map[lig_id] = f"{type(exc).__name__}: {exc}"
                _clear_failed_row(row_idx, f"failed:docking:{docking_backend}")
                updated = True
                stats["failed"] += 1
                if is_ref:
                    raise RuntimeError(f"Reference ligand docking failed: {lig_id}") from exc
                print(f"[oracle][failed] {lig_id} | docking | {type(exc).__name__}: {exc}")

    if updated:
        df.to_csv(paths.smiles_csv, index=False)
    if failed_map:
        failed_path.write_text(json.dumps(failed_map, indent=2), encoding="utf-8")
    elif failed_path.exists():
        failed_path.unlink()
    cache_conn.close()
    return stats
