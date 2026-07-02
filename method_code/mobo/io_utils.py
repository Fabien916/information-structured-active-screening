from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from mobo.smiles_utils import canonicalize_smiles_noh


def _read_smiles_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {path}") from exc


def _pick_smiles_column(columns: Sequence[str]) -> str | None:
    col_map = {str(c).lower(): str(c) for c in columns}
    for key in ["smiles_canonical", "smiles", "smile", "smiles_clean", "smiles_cleaned"]:
        if key in col_map:
            return col_map[key]
    for key in ["smiles", "smile"]:
        if key in col_map:
            return col_map[key]
    return None


def _scan_ligand_vocab(smiles_list: Sequence[str]) -> list[str]:
    vocab = set()
    from rdkit import Chem
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            vocab.add(atom.GetSymbol())
    return sorted(vocab)


def _load_ligand_vocab_override(root: Path) -> Optional[list[str]]:
    json_path = root / "ligand_vocab.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise RuntimeError(f"Invalid ligand vocab JSON (expected list): {json_path}")
        vocab = [str(x).strip() for x in data if str(x).strip()]
        return sorted(set(vocab))
    txt_path = root / "ligand_vocab.txt"
    if txt_path.exists():
        raw = txt_path.read_text(encoding="utf-8").splitlines()
        vocab = [line.strip() for line in raw if line.strip()]
        if vocab:
            return sorted(set(vocab))
    return None


def _is_valid_dock_score(dock: float | None, dock_valid_max: float | None = 0.0) -> bool:
    if dock is None or not np.isfinite(dock):
        return False
    if dock_valid_max is None:
        return True
    return float(dock) <= float(dock_valid_max)


def sanitize_out_csv(path: str) -> Path:
    base = Path(path)
    suffix = base.suffix or ".csv"
    stem = base.stem
    stem = re.sub(r"(?:_iter\\d+)+$", "", stem)
    if not stem:
        stem = "qpmhi_selected"
    return base.with_name(stem + suffix)


def _make_run_dir(runs_root: Path, run_tag: str) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    base = runs_root / run_tag
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base
    idx = 1
    while True:
        cand = runs_root / f"{run_tag}_{idx}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        idx += 1


def append_iter_metrics(path: Path, row: dict) -> None:
    header = [
        "iteration",
        "total_rows",
        "total_with_dock",
        "total_points",
        "pareto_count",
        "new_points",
        "new_rank_min",
        "new_rank_median",
        "new_rank_mean",
        "new_dist_min",
        "new_dist_median",
        "new_dist_mean",
        "oracle_added",
        "oracle_matched",
        "oracle_missing",
        "rmse",
        "mae",
        "r2",
        "hv_before",
        "hv_after",
        "hvi",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})


def purge_generated_rows(csv_path: str, prefix: str, dataset_root: str) -> int:
    df = _read_smiles_csv(csv_path)
    if "ligand_id" not in df.columns:
        return 0
    ligand_ids = df["ligand_id"].astype(str)
    mask = ligand_ids.str.startswith(prefix)
    if not mask.any():
        return 0
    gen_ids = ligand_ids[mask].tolist()
    pose_paths = []
    if "dock_pose" in df.columns:
        for val in df.loc[mask, "dock_pose"].astype(str):
            v = val.strip()
            if v and v.lower() not in {"nan", "none"}:
                pose_paths.append(v)
    df = df.loc[~mask].reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    root = Path(dataset_root)
    docking_dir = root / "docking"
    for lig_id in gen_ids:
        for ext in [".json", ".sdf", ".pdbqt"]:
            path = docking_dir / f"{lig_id}{ext}"
            if path.exists():
                path.unlink()
    for rel in pose_paths:
        path = root / rel
        if path.exists():
            path.unlink()
    failed_path = docking_dir / "failed.json"
    if failed_path.exists():
        with failed_path.open("r", encoding="utf-8") as f:
            failed_map = json.load(f) or {}
        if not isinstance(failed_map, dict):
            raise RuntimeError(f"Invalid failed.json content (expected object): {failed_path}")
        changed = False
        for lig_id in gen_ids:
            if lig_id in failed_map:
                failed_map.pop(lig_id, None)
                changed = True
        if changed:
            with failed_path.open("w", encoding="utf-8") as f:
                json.dump(failed_map, f, indent=2)
    return len(gen_ids)


def load_existing_smiles_set(csv_path: str) -> set[str]:
    df = _read_smiles_csv(csv_path)
    existing = set()
    if "smiles_canonical" in df.columns:
        for val in df["smiles_canonical"].astype(str):
            if val and val.lower() not in {"nan", "none"}:
                can = canonicalize_smiles_noh(val)
                if can:
                    existing.add(can)
    else:
        smiles_col = _pick_smiles_column(df.columns)
        if smiles_col:
            for smi in df[smiles_col].astype(str):
                can = canonicalize_smiles_noh(smi)
                if can:
                    existing.add(can)
    return existing
