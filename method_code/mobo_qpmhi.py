#!/usr/bin/env python3
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple
import re

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit import rdBase
from rdkit.Chem import Crippen, QED
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3, CalcNumHBA, CalcNumHBD, CalcTPSA
from torch_geometric.loader import DataLoader

from smiles_vae_models import build_vae_model
from selfies_data import TokenVocab
from mobo.candidate_pool import build_candidate_pool
from mobo.config_utils import load_config
from mobo.constants import ATOM_EXTRA_DIM, BOND_EXTRA_DIM, BOND_TYPE_CLASSES
from mobo.dataset_utils import (
    clear_processed,
    mark_abnormal_dock,
    resplit_smiles_csv,
    save_top_pose_conformers,
)
from mobo.graphs import build_graphs, build_graphs_3d, compute_qed_sa
from mobo.io_utils import (
    _make_run_dir,
    _is_valid_dock_score,
    _pick_smiles_column,
    _read_smiles_csv,
    _scan_ligand_vocab,
    append_iter_metrics,
    load_existing_smiles_set,
    purge_generated_rows,
    sanitize_out_csv,
)
from mobo.logging_utils import (
    log_kv,
    log_kv_table,
    log_model_param_report,
    log_section,
    log_table,
    log_tensor_stats,
)
from mobo.metrics import (
    DominatedPartitioning,
    bt_is_non_dominated,
    build_ref_point_from_objective,
    collect_oracle_points,
    compute_hvi_from_csv,
    evaluate_oracle_accuracy_from_csv,
    qphv_prob_gaussian_analytic_3d,
    qnehvi_scores_from_samples_nd,
    select_pareto_mc_indices_from_samples,
    select_qphv_indices_from_samples_nd,
    _topk_oracle_by_dock_rows,
    _topk_oracle_pareto_rows,
    _topk_oracle_rows,
    _topk_oracle_rows_weighted_pct,
)
from mobo.oracle import run_oracle_docking
from mobo.plot_utils import plot_run_artifacts
from mobo.retrospective import build_train_distance_scores
from mobo.smiles_utils import canonicalize_smiles_noh
from mobo.surrogate import (
    load_surrogate,
    load_train_objectives,
    mc_samples,
    predict_dock_for_smiles,
    predict_gaussian_stats,
    train_surrogate_from_scratch,
)
from mobo.io_utils import _load_ligand_vocab_override
from mobo.smiles_utils import calc_sa_score_mol
from data.ligand_only_3d_dataset import LigandOnly3DStore

rdBase.DisableLog("rdApp.error")

def _cfg_optional_path(cfg: dict, *keys: str) -> str | None:
    """Return the first non-empty path-like value from cfg for the given keys."""
    for key in keys:
        if not key:
            continue
        val = cfg.get(key)
        if val is None:
            continue
        sval = str(val).strip()
        if not sval or sval.lower() == "null":
            continue
        return sval
    return None


def _resolve_cfg_path(path_str: str, *, base_dir: Path | None = None) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (base_dir or Path.cwd()) / p
    return p.resolve()


def _copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve() == dst.resolve():
            return
    except Exception:
        pass
    shutil.copy2(src, dst)


def _write_sdf_from_pdb(pdb_path: Path, sdf_out: Path) -> None:
    mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False, removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=True, removeHs=False)
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse ligand PDB: {pdb_path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    sdf_out.write_text(Chem.MolToMolBlock(mol), encoding="utf-8")


def _percentile_rank(values: np.ndarray, ascending: bool) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"Percentile ranking expects a 1D array, got shape {values.shape}.")
    n = int(values.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    order = np.argsort(values if ascending else -values, kind="mergesort")
    out = np.empty(n, dtype=np.float32)
    if n == 1:
        out[order[0]] = 1.0
        return out
    out[order] = np.linspace(0.0, 1.0, num=n, endpoint=True, dtype=np.float32)
    return out


def _load_train_smiles_for_fallback(dataset_root: str) -> list[str]:
    csv_path = Path(dataset_root) / "smiles.csv"
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise RuntimeError(f"Missing 'split' column in {csv_path}.")
    smiles_col = _pick_smiles_column(df)
    train_rows = df.loc[df["split"].astype(str).str.lower() == "train", smiles_col].astype(str).tolist()
    train_smiles = [canonicalize_smiles_noh(s) for s in train_rows if str(s).strip()]
    train_smiles = [s for s in train_smiles if s]
    if not train_smiles:
        raise RuntimeError(f"No train split SMILES available for fallback scoring in {csv_path}.")
    return train_smiles


def _uncertainty_fallback_order(
    *,
    candidate_smiles: Sequence[str],
    pred_dock_std: Sequence[float],
    qed_vals: Sequence[float],
    sa_vals: Sequence[float],
    train_smiles: Sequence[str],
) -> tuple[list[int], list[float]]:
    n = len(candidate_smiles)
    if len(pred_dock_std) != n or len(qed_vals) != n or len(sa_vals) != n:
        raise ValueError("Fallback inputs must have the same length.")
    if n == 0:
        return [], []
    if not train_smiles:
        raise RuntimeError("Fallback scoring requires non-empty train_smiles.")
    dist = build_train_distance_scores(
        train_smiles=train_smiles,
        candidate_smiles=list(candidate_smiles),
    )
    std_arr = np.asarray(pred_dock_std, dtype=np.float32)
    qed_arr = np.asarray(qed_vals, dtype=np.float32)
    sa_arr = np.asarray(sa_vals, dtype=np.float32)
    dist_arr = np.asarray(dist, dtype=np.float32)
    if not np.isfinite(std_arr).all():
        raise RuntimeError("Non-finite pred_dock_std encountered in fallback scoring.")
    if not np.isfinite(qed_arr).all():
        raise RuntimeError("Non-finite QED encountered in fallback scoring.")
    if not np.isfinite(sa_arr).all():
        raise RuntimeError("Non-finite SA encountered in fallback scoring.")
    if not np.isfinite(dist_arr).all():
        raise RuntimeError("Non-finite distance encountered in fallback scoring.")
    p_std = _percentile_rank(std_arr, ascending=False)
    p_dist = _percentile_rank(dist_arr, ascending=False)
    p_qed = _percentile_rank(qed_arr, ascending=False)
    p_sa = _percentile_rank(sa_arr, ascending=True)
    scores = p_std + 0.5 * p_dist + 0.25 * p_qed + 0.25 * p_sa
    order = np.argsort(-scores, kind="mergesort").tolist()
    return order, scores.astype(np.float32).tolist()


def _ensure_oracle_assets_from_config(dataset_root: Path, oracle_cfg: dict) -> None:
    """
    Ensure dataset_root contains required oracle assets.

    This complements the default dataset layout by allowing users to provide external
    paths via config when dataset_root only contains smiles.csv.

    Supported config keys (under `oracle:`):
    - `protein_pdb` / `protein` / `protein_path`
    - `pocket_pdb` / `pocket` / `pocket_path`
    - `reference_ligand_sdf` / `reference_ligand` / `ref_ligand` / `ref_ligand_sdf`
    - `pocket_pdbqt` (optional)

    Files are copied into dataset_root with canonical names:
    `protein.pdb`, `pocket.pdb`, `reference_ligand.sdf`, `pocket.pdbqt`.
    """
    base_dir = Path.cwd()
    protein_src = _cfg_optional_path(oracle_cfg, "protein_pdb", "protein", "protein_path")
    pocket_src = _cfg_optional_path(oracle_cfg, "pocket_pdb", "pocket", "pocket_path")
    ref_src = _cfg_optional_path(
        oracle_cfg,
        "reference_ligand_sdf",
        "reference_ligand",
        "ref_ligand",
        "ref_ligand_sdf",
    )
    pocket_pdbqt_src = _cfg_optional_path(oracle_cfg, "pocket_pdbqt", "pocket_pdbqt_path")

    if protein_src:
        src = _resolve_cfg_path(protein_src, base_dir=base_dir)
        if not src.exists():
            raise FileNotFoundError(f"oracle.protein_pdb not found: {src}")
        _copy_if_needed(src, dataset_root / "protein.pdb")
    if pocket_src:
        src = _resolve_cfg_path(pocket_src, base_dir=base_dir)
        if not src.exists():
            raise FileNotFoundError(f"oracle.pocket_pdb not found: {src}")
        _copy_if_needed(src, dataset_root / "pocket.pdb")
    if ref_src:
        src = _resolve_cfg_path(ref_src, base_dir=base_dir)
        if not src.exists():
            raise FileNotFoundError(f"oracle.reference_ligand not found: {src}")
        dst = dataset_root / "reference_ligand.sdf"
        if src.suffix.lower() == ".sdf":
            _copy_if_needed(src, dst)
        elif src.suffix.lower() == ".pdb":
            _write_sdf_from_pdb(src, dst)
        else:
            raise ValueError(f"Unsupported reference ligand format: {src} (expected .sdf or .pdb)")
    if pocket_pdbqt_src:
        src = _resolve_cfg_path(pocket_pdbqt_src, base_dir=base_dir)
        if not src.exists():
            raise FileNotFoundError(f"oracle.pocket_pdbqt not found: {src}")
        _copy_if_needed(src, dataset_root / "pocket.pdbqt")


def load_vae(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    vocab = TokenVocab.from_tokens(ckpt["vocab"])
    decoder_max_len = int(data_cfg["max_len"]) + 1
    model = build_vae_model(vocab=vocab, model_cfg=model_cfg, max_len=decoder_max_len)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, vocab, cfg


def _section(cfg: dict, key: str) -> dict:
    sect = cfg.get(key)
    return sect if isinstance(sect, dict) else {}


def _next_ligand_id(existing_ids: set[str], prefix: str, start: int = 0) -> str:
    idx = start
    while True:
        ligand_id = f"{prefix}_{idx:05d}"
        if ligand_id not in existing_ids:
            return ligand_id
        idx += 1


def _collect_missing_dock_ids(csv_path: str, dock_valid_max: float | None) -> list[str]:
    df = _read_smiles_csv(csv_path)
    if df.empty or "ligand_id" not in df.columns:
        return []
    if "dock_score" not in df.columns:
        return [str(x) for x in df["ligand_id"].astype(str).tolist()]
    missing: list[str] = []
    for lig_id, dock in zip(df["ligand_id"].astype(str), df["dock_score"]):
        val = None
        try:
            val = float(dock)
        except Exception:
            val = None
        if not _is_valid_dock_score(val, dock_valid_max=dock_valid_max):
            missing.append(str(lig_id))
    return missing


def _reference_smiles_from_sdf(path: Path) -> str | None:
    if not path.exists():
        return None
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = next(iter(supplier), None)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _mark_reference_in_csv(dataset_root: Path, ref_id: str) -> tuple[bool, str]:
    smiles_csv = dataset_root / "smiles.csv"
    reference_ligand = dataset_root / "reference_ligand.sdf"
    if not smiles_csv.exists() or not reference_ligand.exists():
        return False, "missing_smiles_or_reference"
    df = pd.read_csv(smiles_csv)
    if df.empty:
        return False, "empty_smiles"
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        return False, "missing_smiles_column"
    if smiles_col != "smiles":
        df["smiles"] = df[smiles_col].astype(str)
    if "smiles_canonical" not in df.columns:
        df["smiles_canonical"] = [
            canonicalize_smiles_noh(smi) or "" for smi in df["smiles"].astype(str).tolist()
        ]
    if "ligand_id" not in df.columns:
        df["ligand_id"] = [f"LIG_{i:05d}" for i in range(1, len(df) + 1)]
    if "is_reference" not in df.columns:
        df["is_reference"] = 0

    ref_smi = _reference_smiles_from_sdf(reference_ligand)
    if not ref_smi:
        return False, "reference_read_failed"
    ref_canon = canonicalize_smiles_noh(ref_smi) or ""
    if not ref_canon:
        return False, "reference_canon_failed"

    match = df["smiles_canonical"] == ref_canon
    existing_ids = set(df["ligand_id"].astype(str).tolist())
    if ref_id in existing_ids and not df.loc[match, "ligand_id"].eq(ref_id).all():
        idx = 2
        while f"{ref_id}_{idx}" in existing_ids:
            idx += 1
        ref_id = f"{ref_id}_{idx}"

    if match.any():
        df.loc[match, "is_reference"] = 1
        df.loc[match, "ligand_id"] = ref_id
        action = "marked"
    else:
        new_row = {col: np.nan for col in df.columns}
        new_row["ligand_id"] = ref_id
        new_row["smiles"] = ref_smi
        new_row["smiles_canonical"] = ref_canon
        if "qed" in df.columns:
            try:
                mol = Chem.MolFromSmiles(ref_smi)
                new_row["qed"] = float(QED.qed(mol)) if mol is not None else np.nan
            except Exception:
                new_row["qed"] = np.nan
        if "sa_score" in df.columns:
            try:
                mol = Chem.MolFromSmiles(ref_smi)
                new_row["sa_score"] = float(calc_sa_score_mol(mol)) if mol is not None else np.nan
            except Exception:
                new_row["sa_score"] = np.nan
        if "split" in df.columns:
            new_row["split"] = "train"
        new_row["is_reference"] = 1
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        action = "appended"

    df.to_csv(smiles_csv, index=False)
    return True, action


def _fill_qed_sa_in_csv(dataset_root: Path, force: bool = False) -> dict[str, int]:
    smiles_csv = dataset_root / "smiles.csv"
    stats = {"rows": 0, "qed_updated": 0, "sa_updated": 0, "fail": 0}
    if not smiles_csv.exists():
        return stats
    df = pd.read_csv(smiles_csv)
    if df.empty:
        return stats
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        return stats
    if smiles_col != "smiles":
        df["smiles"] = df[smiles_col].astype(str)
    if "qed" not in df.columns:
        df["qed"] = np.nan
    if "sa_score" not in df.columns:
        df["sa_score"] = np.nan
    stats["rows"] = len(df)
    for idx, smi in enumerate(df["smiles"].astype(str).tolist()):
        if not smi:
            continue
        need_qed = force or pd.isna(df.at[idx, "qed"])
        need_sa = force or pd.isna(df.at[idx, "sa_score"])
        if not (need_qed or need_sa):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["fail"] += 1
            continue
        if need_qed:
            try:
                df.at[idx, "qed"] = float(QED.qed(mol))
                stats["qed_updated"] += 1
            except Exception:
                stats["fail"] += 1
        if need_sa:
            try:
                df.at[idx, "sa_score"] = float(calc_sa_score_mol(mol))
                stats["sa_updated"] += 1
            except Exception:
                stats["fail"] += 1
    df.to_csv(smiles_csv, index=False)
    return stats


def _window_desirability(x: float, hard_lo: float, hard_hi: float, soft_lo: float, soft_hi: float) -> float:
    """
    Trapezoid desirability in [0,1]:
      - 0 outside [soft_lo, soft_hi]
      - 1 inside [hard_lo, hard_hi]
      - linear ramps between soft and hard bounds.
    Requires: soft_lo <= hard_lo <= hard_hi <= soft_hi
    """
    if not (soft_lo <= hard_lo <= hard_hi <= soft_hi):
        raise ValueError("invalid window bounds: require soft_lo <= hard_lo <= hard_hi <= soft_hi")
    if x <= soft_lo or x >= soft_hi:
        return 0.0
    if hard_lo <= x <= hard_hi:
        return 1.0
    if x < hard_lo:
        den = (hard_lo - soft_lo) or 1.0
        return max(0.0, min(1.0, (x - soft_lo) / den))
    den = (soft_hi - hard_hi) or 1.0
    return max(0.0, min(1.0, (soft_hi - x) / den))


def _combine2(a: float, b: float, mode: str) -> float:
    mode = str(mode or "geom").lower().strip()
    if mode == "geom":
        return float(np.sqrt(max(a, 0.0) * max(b, 0.0)))
    if mode == "mean":
        return float((a + b) / 2.0)
    if mode == "min":
        return float(min(a, b))
    raise ValueError(f"unknown combine mode: {mode}")


def _compute_win_score(mol: Chem.Mol, window_cfg: dict) -> tuple[float, float, float, float, float]:
    mol_noh = Chem.RemoveHs(mol, sanitize=True)
    logp = float(Crippen.MolLogP(mol_noh))
    tpsa = float(CalcTPSA(mol_noh))
    hard_lp = window_cfg.get("logp_hard", [1.0, 3.5])
    soft_lp = window_cfg.get("logp_soft", [0.0, 5.0])
    hard_t = window_cfg.get("tpsa_hard", [40.0, 120.0])
    soft_t = window_cfg.get("tpsa_soft", [20.0, 160.0])
    agg = window_cfg.get("agg", window_cfg.get("win_agg", "geom"))
    wl = _window_desirability(logp, float(hard_lp[0]), float(hard_lp[1]), float(soft_lp[0]), float(soft_lp[1]))
    wt = _window_desirability(tpsa, float(hard_t[0]), float(hard_t[1]), float(soft_t[0]), float(soft_t[1]))
    win = _combine2(wl, wt, str(agg))
    return logp, tpsa, float(wl), float(wt), float(win)


def _compute_hb_score(mol: Chem.Mol, hb_cfg: dict) -> tuple[int, int, float, float, float]:
    mol_noh = Chem.RemoveHs(mol, sanitize=True)
    hbd = int(CalcNumHBD(mol_noh))
    hba = int(CalcNumHBA(mol_noh))
    hard_d = hb_cfg.get("hbd_hard", [0.0, 2.0])
    soft_d = hb_cfg.get("hbd_soft", [0.0, 4.0])
    hard_a = hb_cfg.get("hba_hard", [2.0, 8.0])
    soft_a = hb_cfg.get("hba_soft", [1.0, 10.0])
    agg = hb_cfg.get("agg", hb_cfg.get("hb_agg", "geom"))
    dhbd = _window_desirability(float(hbd), float(hard_d[0]), float(hard_d[1]), float(soft_d[0]), float(soft_d[1]))
    dhba = _window_desirability(float(hba), float(hard_a[0]), float(hard_a[1]), float(soft_a[0]), float(soft_a[1]))
    hb = _combine2(dhbd, dhba, str(agg))
    return hbd, hba, float(dhbd), float(dhba), float(hb)


def _build_prior_state(repo_root: Path, objective_cfg: dict) -> dict | None:
    """
    Build a cached prior similarity state:
      - Morgan generator
      - prior fingerprints
      - aggregation mode
    """
    extra = objective_cfg.get("sim_prior", {}) if isinstance(objective_cfg.get("sim_prior", {}), dict) else {}
    prior = str(extra.get("prior", "crc_epigenetic")).strip().lower()
    if prior in {"", "none", "null", "off", "disable", "disabled"}:
        return None
    fp_radius = int(extra.get("fp_radius", 2))
    fp_nbits = int(extra.get("fp_nbits", extra.get("fp_nbits", 2048)))
    sim_agg = str(extra.get("sim_agg", "max")).strip().lower()
    sim_topk = int(extra.get("sim_topk", 3))

    # Resolve prior smiles.
    if prior == "crc_epigenetic":
        prior_path = repo_root / "data" / "prior_crc_epigenetic.smi"
    elif prior in {"hdac", "hdac_hydroxamate", "hydroxamate"}:
        prior_path = repo_root / "data" / "prior_hdac_hydroxamate.smi"
    elif prior == "file":
        pf = extra.get("prior_file", None)
        if pf in (None, "", "null"):
            raise ValueError("objective.sim_prior.prior_file is required when prior=file")
        prior_path = (Path(pf) if Path(pf).is_absolute() else (repo_root / str(pf))).resolve()
    else:
        raise ValueError(f"unknown objective.sim_prior.prior: {prior}")
    if not prior_path.exists():
        raise FileNotFoundError(f"prior file not found: {prior_path}")

    prior_smiles: list[str] = []
    with prior_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0].strip()
            if smi:
                prior_smiles.append(smi)
    if not prior_smiles:
        raise RuntimeError(f"empty prior smiles: {prior_path}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_nbits)
    fps = []
    bad = 0
    for smi in prior_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad += 1
            continue
        fps.append(gen.GetFingerprint(m))
    if not fps:
        raise RuntimeError(f"no valid prior fingerprints built from {prior_path}")
    return {
        "prior": prior,
        "prior_path": str(prior_path),
        "fp_radius": fp_radius,
        "fp_nbits": fp_nbits,
        "sim_agg": sim_agg,
        "sim_topk": sim_topk,
        "bad_smiles": bad,
        "gen": gen,
        "fps": fps,
    }


def _sim_prior_from_fp(fp, prior_state: dict) -> float:
    fps = prior_state["fps"]
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
    if not sims:
        return 0.0
    mode = str(prior_state.get("sim_agg", "max")).lower()
    if mode == "max":
        return float(max(sims))
    if mode == "mean":
        return float(sum(sims) / len(sims))
    if mode == "mean_topk":
        k = min(int(prior_state.get("sim_topk", 3)), len(sims))
        sims_sorted = sorted(sims, reverse=True)
        return float(sum(sims_sorted[:k]) / max(1, k))
    raise ValueError(f"unknown sim_agg: {mode}")


def _ensure_metric_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _fill_extra_metrics_in_csv(
    dataset_root: Path,
    objective_cfg: dict,
    prior_state: dict | None,
    force: bool = False,
) -> dict[str, int]:
    """
    Fill extra RDKit metrics into dataset_root/smiles.csv so that:
      - selection/acquisition can treat them as deterministic objectives
      - logging/analysis/metrics can read them from CSV
    """
    smiles_csv = dataset_root / "smiles.csv"
    stats = {"rows": 0, "updated": 0, "fail": 0}
    if not smiles_csv.exists():
        return stats
    df = pd.read_csv(smiles_csv)
    if df.empty:
        return stats
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        return stats
    if smiles_col != "smiles":
        df["smiles"] = df[smiles_col].astype(str)

    # Ensure columns exist.
    need_cols = [
        "qed",
        "sa_score",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "fsp3",
        "win_logp",
        "win_tpsa",
        "win",
        "hb_hbd",
        "hb_hba",
        "hb",
        "sim_prior",
    ]
    df = _ensure_metric_columns(df, need_cols)

    window_cfg = objective_cfg.get("window", {}) if isinstance(objective_cfg.get("window", {}), dict) else {}
    hb_cfg = objective_cfg.get("hbond_balance", {}) if isinstance(objective_cfg.get("hbond_balance", {}), dict) else {}
    stats["rows"] = int(len(df))

    for idx, smi in enumerate(df["smiles"].astype(str).tolist()):
        if not smi:
            continue
        # Decide whether any metric needs computation.
        needs = False
        for c in need_cols:
            if force or pd.isna(df.at[idx, c]):
                needs = True
                break
        if not needs:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["fail"] += 1
            continue
        try:
            mol_noh = Chem.RemoveHs(mol, sanitize=True)
        except Exception:
            mol_noh = mol

        # QED/SA
        if force or pd.isna(df.at[idx, "qed"]):
            try:
                df.at[idx, "qed"] = float(QED.qed(mol_noh))
                stats["updated"] += 1
            except Exception:
                stats["fail"] += 1
        if force or pd.isna(df.at[idx, "sa_score"]):
            try:
                sa_clamp_min = objective_cfg.get("sa_clamp_min", None)
                sa_clamp_max = objective_cfg.get("sa_clamp_max", None)
                sa_clamp_min = None if sa_clamp_min is None else float(sa_clamp_min)
                sa_clamp_max = None if sa_clamp_max is None else float(sa_clamp_max)
                df.at[idx, "sa_score"] = float(calc_sa_score_mol(mol_noh, clamp_min=sa_clamp_min, clamp_max=sa_clamp_max))
                stats["updated"] += 1
            except Exception:
                stats["fail"] += 1

        # Basic descriptors.
        if force or pd.isna(df.at[idx, "logp"]) or pd.isna(df.at[idx, "tpsa"]) or pd.isna(df.at[idx, "fsp3"]):
            try:
                df.at[idx, "logp"] = float(Crippen.MolLogP(mol_noh))
                df.at[idx, "tpsa"] = float(CalcTPSA(mol_noh))
                df.at[idx, "fsp3"] = float(CalcFractionCSP3(mol_noh))
                stats["updated"] += 1
            except Exception:
                stats["fail"] += 1
        if force or pd.isna(df.at[idx, "hbd"]) or pd.isna(df.at[idx, "hba"]):
            try:
                df.at[idx, "hbd"] = int(CalcNumHBD(mol_noh))
                df.at[idx, "hba"] = int(CalcNumHBA(mol_noh))
                stats["updated"] += 1
            except Exception:
                stats["fail"] += 1

        # Window desirability.
        if force or pd.isna(df.at[idx, "win"]):
            try:
                logp, tpsa, wl, wt, win = _compute_win_score(mol_noh, window_cfg)
                df.at[idx, "logp"] = float(logp)
                df.at[idx, "tpsa"] = float(tpsa)
                df.at[idx, "win_logp"] = float(wl)
                df.at[idx, "win_tpsa"] = float(wt)
                df.at[idx, "win"] = float(win)
                stats["updated"] += 1
            except Exception:
                stats["fail"] += 1

        # H-bond balance.
        if force or pd.isna(df.at[idx, "hb"]):
            try:
                hbd, hba, dhbd, dhba, hb = _compute_hb_score(mol_noh, hb_cfg)
                df.at[idx, "hbd"] = int(hbd)
                df.at[idx, "hba"] = int(hba)
                df.at[idx, "hb_hbd"] = float(dhbd)
                df.at[idx, "hb_hba"] = float(dhba)
                df.at[idx, "hb"] = float(hb)
                stats["updated"] += 1
            except Exception:
                stats["fail"] += 1

        # Similarity-to-prior.
        if prior_state is not None and (force or pd.isna(df.at[idx, "sim_prior"])):
            try:
                fp = prior_state["gen"].GetFingerprint(mol_noh)
                df.at[idx, "sim_prior"] = float(_sim_prior_from_fp(fp, prior_state))
                stats["updated"] += 1
            except Exception:
                stats["fail"] += 1

    df.to_csv(smiles_csv, index=False)
    return stats


def _compute_extra_metrics_for_smiles(
    smiles_list: Sequence[str],
    objective_cfg: dict,
    prior_state: dict | None,
) -> dict[str, list[float]]:
    """
    Compute deterministic RDKit metrics for a list of SMILES (one pass, no pandas).
    Returns lists aligned with smiles_list.
    """
    window_cfg = objective_cfg.get("window", {}) if isinstance(objective_cfg.get("window", {}), dict) else {}
    hb_cfg = objective_cfg.get("hbond_balance", {}) if isinstance(objective_cfg.get("hbond_balance", {}), dict) else {}
    out: dict[str, list[float]] = {
        "logp": [],
        "tpsa": [],
        "hbd": [],
        "hba": [],
        "fsp3": [],
        "win_logp": [],
        "win_tpsa": [],
        "win": [],
        "hb_hbd": [],
        "hb_hba": [],
        "hb": [],
        "sim_prior": [],
    }
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            for k in out:
                out[k].append(float("nan"))
            continue
        try:
            mol_noh = Chem.RemoveHs(mol, sanitize=True)
        except Exception:
            mol_noh = mol
        try:
            logp, tpsa, wl, wt, win = _compute_win_score(mol_noh, window_cfg)
        except Exception:
            logp, tpsa, wl, wt, win = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        try:
            hbd, hba, dhbd, dhba, hb = _compute_hb_score(mol_noh, hb_cfg)
        except Exception:
            hbd, hba, dhbd, dhba, hb = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        try:
            fsp3 = float(CalcFractionCSP3(mol_noh))
        except Exception:
            fsp3 = float("nan")
        sim = float("nan")
        if prior_state is not None:
            try:
                fp = prior_state["gen"].GetFingerprint(mol_noh)
                sim = float(_sim_prior_from_fp(fp, prior_state))
            except Exception:
                sim = float("nan")
        out["logp"].append(float(logp))
        out["tpsa"].append(float(tpsa))
        out["hbd"].append(float(hbd))
        out["hba"].append(float(hba))
        out["fsp3"].append(float(fsp3))
        out["win_logp"].append(float(wl))
        out["win_tpsa"].append(float(wt))
        out["win"].append(float(win))
        out["hb_hbd"].append(float(dhbd))
        out["hb_hba"].append(float(dhba))
        out["hb"].append(float(hb))
        out["sim_prior"].append(float(sim))
    return out


def _load_train_objectives_extended(
    dataset_root: str,
    split: str,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    use_sa: bool,
    extra_objectives: Sequence[str],
    sim_prior_sign: float,
    win_sign: float,
    hb_sign: float,
    fsp3_sign: float,
    dock_valid_max: float | None,
    sa_clamp_min: float | None,
    sa_clamp_max: float | None,
    objective_cfg: dict,
    prior_state: dict | None,
) -> torch.Tensor | None:
    df = _read_smiles_csv(str(Path(dataset_root) / "smiles.csv"))
    if df.empty or "dock_score" not in df.columns:
        return None
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == str(split).lower()]
    else:
        if str(split).lower() != "train":
            df = df.iloc[0:0]
    if df.empty:
        return None
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        return None

    window_cfg = objective_cfg.get("window", {}) if isinstance(objective_cfg.get("window", {}), dict) else {}
    hb_cfg = objective_cfg.get("hbond_balance", {}) if isinstance(objective_cfg.get("hbond_balance", {}), dict) else {}

    values: list[list[float]] = []
    for _, row in df.iterrows():
        dock = float(row.get("dock_score", float("nan")))
        if not _is_valid_dock_score(dock, dock_valid_max=dock_valid_max):
            continue
        smi = str(row.get(smiles_col, "")).strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            mol_noh = Chem.RemoveHs(mol, sanitize=True)
        except Exception:
            mol_noh = mol

        # QED / SA (prefer CSV values if present).
        qed_val = row.get("qed", row.get("QED", None))
        try:
            qed_val = float(qed_val)
        except Exception:
            qed_val = float("nan")
        if not np.isfinite(qed_val):
            try:
                qed_val = float(QED.qed(mol_noh))
            except Exception:
                qed_val = float("nan")
        if not np.isfinite(qed_val):
            continue

        sa_val = row.get("sa_score", None)
        try:
            sa_val = float(sa_val)
        except Exception:
            sa_val = float("nan")
        if use_sa and not np.isfinite(sa_val):
            try:
                sa_val = float(calc_sa_score_mol(mol_noh, clamp_min=sa_clamp_min, clamp_max=sa_clamp_max))
            except Exception:
                sa_val = float("nan")
        if use_sa and not np.isfinite(sa_val):
            continue

        obj = [dock_sign * float(dock), qed_sign * float(qed_val)]
        if use_sa:
            obj.append(sa_sign * float(sa_val))

        # Extra deterministic objectives are all maximized.
        if "sim_prior" in extra_objectives:
            sim = row.get("sim_prior", None)
            try:
                sim = float(sim)
            except Exception:
                sim = float("nan")
            if not np.isfinite(sim) and prior_state is not None:
                try:
                    fp = prior_state["gen"].GetFingerprint(mol_noh)
                    sim = float(_sim_prior_from_fp(fp, prior_state))
                except Exception:
                    sim = float("nan")
            if not np.isfinite(sim):
                sim = 0.0
            obj.append(sim_prior_sign * float(sim))

        if "win" in extra_objectives:
            win = row.get("win", None)
            try:
                win = float(win)
            except Exception:
                win = float("nan")
            if not np.isfinite(win):
                try:
                    _, _, _, _, win = _compute_win_score(mol_noh, window_cfg)
                except Exception:
                    win = float("nan")
            if not np.isfinite(win):
                win = 0.0
            obj.append(win_sign * float(win))

        if "hb" in extra_objectives:
            hb = row.get("hb", None)
            try:
                hb = float(hb)
            except Exception:
                hb = float("nan")
            if not np.isfinite(hb):
                try:
                    _, _, _, _, hb = _compute_hb_score(mol_noh, hb_cfg)
                except Exception:
                    hb = float("nan")
            if not np.isfinite(hb):
                hb = 0.0
            obj.append(hb_sign * float(hb))

        if "fsp3" in extra_objectives:
            fsp3 = row.get("fsp3", None)
            try:
                fsp3 = float(fsp3)
            except Exception:
                fsp3 = float("nan")
            if not np.isfinite(fsp3):
                try:
                    fsp3 = float(CalcFractionCSP3(mol_noh))
                except Exception:
                    fsp3 = float("nan")
            if not np.isfinite(fsp3):
                fsp3 = 0.0
            obj.append(fsp3_sign * float(fsp3))

        values.append(obj)
    if not values:
        return None
    return torch.tensor(values, dtype=torch.float32)


def _normalize_split_ratio(split_ratio: Tuple[float, float, float] | None) -> dict[str, float] | None:
    if split_ratio is None:
        return None
    try:
        p_train, p_valid, p_test = (float(split_ratio[0]), float(split_ratio[1]), float(split_ratio[2]))
    except Exception:
        return None
    total = p_train + p_valid + p_test
    if total <= 0:
        return None
    return {
        "train": p_train / total,
        "valid": p_valid / total,
        "test": p_test / total,
    }


def _parse_ref_token(token: str) -> tuple[str, str | None, int | None]:
    raw = token.strip().upper()
    if not raw:
        return "", None, None
    m = re.match(r"^([A-Z0-9]{3})([A-Z])(\d+)$", raw)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    m = re.match(r"^([A-Z0-9]{3})([A-Z])$", raw)
    if m:
        return m.group(1), m.group(2), None
    m = re.match(r"^([A-Z0-9]{3})$", raw)
    if m:
        return m.group(1), None, None
    # fallback: try DU9:A:1201-like tokens
    parts = re.split(r"[:_\\-]", raw)
    if len(parts) >= 3 and len(parts[0]) >= 3:
        resname = parts[0][:3]
        chain = parts[1][:1] if parts[1] else None
        try:
            resid = int(parts[2])
        except Exception:
            resid = None
        return resname, chain, resid
    return raw[:3], None, None


def _extract_reference_ligand_sdf(
    protein_pdb: Path,
    ref_token: str,
    out_path: Path,
    default_altloc: str | None = None,
) -> bool:
    resname, chain, resid = _parse_ref_token(ref_token)
    if not resname:
        return False
    altloc_keep = {""}
    if default_altloc:
        altloc_keep.add(str(default_altloc).strip().upper())
    kept = []
    with protein_pdb.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            rec = line[0:6].strip().upper()
            if rec not in {"ATOM", "HETATM"}:
                continue
            name = line[17:20].strip().upper()
            if name != resname:
                continue
            altloc = line[16].strip().upper()
            if altloc not in altloc_keep:
                continue
            if chain is not None:
                chain_id = line[21].strip().upper()
                if chain_id != chain:
                    continue
            if resid is not None:
                try:
                    rid = int(line[22:26].strip())
                except Exception:
                    rid = None
                if rid != resid:
                    continue
            kept.append(line.rstrip("\n"))
    if not kept:
        return False
    pdb_block = "\n".join(kept) + "\nEND\n"
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    sdf_block = Chem.MolToMolBlock(mol)
    out_path.write_text(sdf_block, encoding="utf-8")
    return True


def _ensure_added_iter(csv_path: str) -> None:
    df = _read_smiles_csv(csv_path)
    if df.empty:
        return
    if "added_iter" in df.columns:
        return
    df = df.copy()
    df["added_iter"] = 0
    df.to_csv(csv_path, index=False)


def _init_split_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    if "split" not in df.columns:
        return counts
    for val in df["split"].astype(str):
        key = str(val).strip().lower()
        if key in counts:
            counts[key] += 1
        else:
            counts["train"] += 1
    return counts


def _pick_incremental_split(counts: dict[str, int], ratios: dict[str, float]) -> str:
    order = ("train", "valid", "test")
    total_after = sum(counts.values()) + 1
    deficits = {k: ratios.get(k, 0.0) * total_after - counts.get(k, 0) for k in order}
    max_def = max(deficits.values())
    if max_def > 0:
        for key in order:
            if deficits.get(key, 0.0) == max_def:
                counts[key] += 1
                return key
    total_now = sum(counts.values())
    if total_now <= 0:
        counts["train"] += 1
        return "train"
    best_key = "train"
    best_score = float("inf")
    for key in order:
        target = ratios.get(key, 0.0)
        if target <= 0:
            continue
        score = counts.get(key, 0) / (target * total_now)
        if score < best_score:
            best_score = score
            best_key = key
    counts[best_key] += 1
    return best_key


def append_candidates_to_smiles_csv(
    csv_path: str,
    smiles_list: Sequence[str],
    id_prefix: str = "GEN",
    sa_clamp_min: float | None = None,
    sa_clamp_max: float | None = None,
    split_ratio: Tuple[float, float, float] | None = None,
    added_iter: int | None = None,
    molecule_origin: str = "generated",
) -> tuple[int, list[str], list[int]]:
    split = "train"
    df = _read_smiles_csv(csv_path)
    cols = list(df.columns)
    smiles_col = "smiles" if "smiles" in df.columns else None
    if smiles_col is None:
        for alt in ["SMILES", "smile"]:
            if alt in df.columns:
                smiles_col = alt
                break
    if smiles_col is None:
        df["smiles"] = ""
        smiles_col = "smiles"
        cols = list(df.columns)
    if "qed" not in df.columns and "QED" not in df.columns:
        df["qed"] = np.nan
        cols = list(df.columns)
    # Extra deterministic metrics (filled later by _fill_extra_metrics_in_csv, but we create columns here
    # so downstream tooling can rely on stable schema).
    for c in (
        "sa_score",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "fsp3",
        "win_logp",
        "win_tpsa",
        "win",
        "hb_hbd",
        "hb_hba",
        "hb",
        "sim_prior",
    ):
        if c not in df.columns:
            df[c] = np.nan
    cols = list(df.columns)
    if "added_iter" not in df.columns:
        df["added_iter"] = 0
        cols = list(df.columns)
    if "molecule_origin" not in df.columns:
        df["molecule_origin"] = ""
        cols = list(df.columns)
    if "is_generated" not in df.columns:
        df["is_generated"] = 0
        cols = list(df.columns)

    ratios = _normalize_split_ratio(split_ratio)
    if ratios is not None and "split" not in df.columns:
        df["split"] = split
        cols = list(df.columns)
    split_counts = _init_split_counts(df) if ratios is not None else {}

    existing_canon = set()
    if "smiles_canonical" in df.columns:
        for val in df["smiles_canonical"].astype(str):
            if val and val.lower() not in {"nan", "none"}:
                can = canonicalize_smiles_noh(val)
                if can:
                    existing_canon.add(can)
    if not existing_canon:
        for smi in df[smiles_col].astype(str):
            can = canonicalize_smiles_noh(smi)
            if can:
                existing_canon.add(can)

    if "ligand_id" in df.columns:
        existing_ids = set(df["ligand_id"].astype(str))
    else:
        df["ligand_id"] = [f"LIG_{i:05d}" for i in range(len(df))]
        cols = list(df.columns)
        existing_ids = set(df["ligand_id"].astype(str))

    new_rows = []
    new_ids: list[str] = []
    kept_indices: list[int] = []
    next_id_idx = 0
    for idx, smi in enumerate(smiles_list):
        canon = canonicalize_smiles_noh(smi)
        if not canon or canon in existing_canon:
            continue
        existing_canon.add(canon)
        ligand_id = _next_ligand_id(existing_ids, id_prefix, start=next_id_idx)
        existing_ids.add(ligand_id)
        next_id_idx += 1
        new_ids.append(ligand_id)
        kept_indices.append(idx)

        row = {c: ("" if c in {"dock_pose"} else np.nan) for c in cols}
        row[smiles_col] = canon
        if "smiles" in cols:
            row["smiles"] = canon
        if "SMILES" in cols:
            row["SMILES"] = canon
        if "smiles_canonical" in cols:
            row["smiles_canonical"] = canon
        if "ligand_id" in cols:
            row["ligand_id"] = ligand_id
        if "split" in cols:
            if ratios is not None:
                row["split"] = _pick_incremental_split(split_counts, ratios)
            else:
                row["split"] = split
        if "added_iter" in cols:
            row["added_iter"] = int(added_iter) if added_iter is not None else np.nan
        if "molecule_origin" in cols:
            row["molecule_origin"] = str(molecule_origin)
        if "is_generated" in cols:
            row["is_generated"] = 1 if str(molecule_origin) == "generated" else 0
        if "dock_pose" in cols:
            row["dock_pose"] = ""
        if "dock_score" in cols:
            row["dock_score"] = np.nan

        if (
            "sa_score" in cols
            or "logp" in cols
            or "logP" in cols
            or "log_p" in cols
            or "tpsa" in cols
            or "hbd" in cols
            or "hba" in cols
            or "fsp3" in cols
            or "qed" in cols
            or "QED" in cols
        ):
            mol = Chem.MolFromSmiles(canon)
            sa_val = float("nan")
            logp_val = float("nan")
            tpsa_val = float("nan")
            hbd_val = float("nan")
            hba_val = float("nan")
            fsp3_val = float("nan")
            qed_val = float("nan")
            if mol is not None:
                try:
                    mol_noh = Chem.RemoveHs(mol, sanitize=True)
                except Exception:
                    mol_noh = mol
                try:
                    sa_val = float(calc_sa_score_mol(mol_noh, clamp_min=sa_clamp_min, clamp_max=sa_clamp_max))
                except Exception:
                    sa_val = float("nan")
                try:
                    logp_val = float(Crippen.MolLogP(mol_noh))
                except Exception:
                    logp_val = float("nan")
                try:
                    tpsa_val = float(CalcTPSA(mol_noh))
                except Exception:
                    tpsa_val = float("nan")
                try:
                    hbd_val = float(CalcNumHBD(mol_noh))
                    hba_val = float(CalcNumHBA(mol_noh))
                except Exception:
                    hbd_val = float("nan")
                    hba_val = float("nan")
                try:
                    fsp3_val = float(CalcFractionCSP3(mol_noh))
                except Exception:
                    fsp3_val = float("nan")
                try:
                    qed_val = float(QED.qed(mol_noh))
                except Exception:
                    qed_val = float("nan")
            if "sa_score" in cols:
                row["sa_score"] = sa_val
            if "logp" in cols:
                row["logp"] = logp_val
            if "logP" in cols:
                row["logP"] = logp_val
            if "log_p" in cols:
                row["log_p"] = logp_val
            if "tpsa" in cols:
                row["tpsa"] = tpsa_val
            if "hbd" in cols:
                row["hbd"] = hbd_val
            if "hba" in cols:
                row["hba"] = hba_val
            if "fsp3" in cols:
                row["fsp3"] = fsp3_val
            if "qed" in cols:
                row["qed"] = qed_val
            if "QED" in cols:
                row["QED"] = qed_val

        new_rows.append(row)

    if not new_rows:
        return 0, [], []
    df = pd.concat([df, pd.DataFrame(new_rows, columns=cols)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return len(new_rows), new_ids, kept_indices




def main() -> int:
    cfg = load_config("config/surrogate/config.yaml")
    if not isinstance(cfg, dict):
        raise RuntimeError("config.yaml must be a mapping.")

    repo_root = Path(__file__).resolve().parent

    general_cfg = _section(cfg, "general")
    candidate_cfg = _section(cfg, "candidate")
    selection_cfg = _section(cfg, "selection")
    oracle_cfg = _section(cfg, "oracle")
    dataset_cfg = _section(cfg, "dataset")
    surrogate_cfg = _section(cfg, "surrogate")
    model_cfg = _section(cfg, "model")
    objective_cfg = _section(cfg, "objective")
    run_cfg = _section(cfg, "run")

    device_cfg = general_cfg.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    vae_ckpt = general_cfg.get("vae_ckpt", "checkpoints/selfie_vae.pt")
    dataset_root = general_cfg.get("dataset_root", "dataset/6KRO")
    pool_size = int(candidate_cfg.get("pool_size", 2000))
    sample_batch = int(candidate_cfg.get("sample_batch", min(500, pool_size)))
    batch_size = int(selection_cfg.get("batch_size", 100))
    mc_samples_n = int(selection_cfg.get("mc_samples", 64))
    temperature = float(candidate_cfg.get("temperature", 1.0))
    top_k = int(candidate_cfg.get("top_k", 0))
    out_csv = str(run_cfg.get("out_csv", "qpmhi_selected.csv"))

    weights = None
    cfg_weights = selection_cfg.get("weights")
    if cfg_weights:
        if isinstance(cfg_weights, (list, tuple)):
            weights = [float(x) for x in cfg_weights]
        else:
            weights = [float(x) for x in str(cfg_weights).split(",")]

    surrogate_ckpts: list[str] = []

    iterations = int(general_cfg.get("iterations", 1))
    run_oracle = bool(oracle_cfg.get("run_oracle", False))
    oracle_overwrite = bool(oracle_cfg.get("oracle_overwrite", False))
    oracle_exhaustiveness = int(oracle_cfg.get("oracle_exhaustiveness", 8))
    oracle_pocket_radius = float(oracle_cfg.get("oracle_pocket_radius", 10.0))
    oracle_id_prefix = str(oracle_cfg.get("oracle_id_prefix", "GEN"))
    oracle_meeko_allow_bad_res = bool(oracle_cfg.get("meeko_allow_bad_res", False))
    oracle_meeko_default_altloc = oracle_cfg.get("meeko_default_altloc", None)
    oracle_prepare_dataset = bool(oracle_cfg.get("prepare_dataset", True))
    oracle_ref_resname = oracle_cfg.get("ref_resname", None)
    oracle_ref_resname = str(oracle_ref_resname).strip() if oracle_ref_resname not in (None, "", "null") else None
    oracle_fill_qed_sa = bool(oracle_cfg.get("fill_qed_sa", True))
    oracle_mark_reference = bool(oracle_cfg.get("mark_reference_in_csv", True))
    oracle_ref_id = str(oracle_cfg.get("ref_id", "REF_LIG")).strip() or "REF_LIG"
    resplit = bool(dataset_cfg.get("resplit", False))
    split_ratio_cfg = dataset_cfg.get("split_ratio", [0.9, 0.1, 0.0])
    split_ratio = tuple(float(x) for x in split_ratio_cfg)
    split_seed = int(dataset_cfg.get("split_seed", 42))
    retrain = bool(surrogate_cfg.get("retrain", False))
    retrain_epochs = int(surrogate_cfg.get("retrain_epochs", 50))
    retrain_batch_size = int(surrogate_cfg.get("retrain_batch_size", 32))
    retrain_lr = float(surrogate_cfg.get("retrain_lr", 1e-3))
    retrain_weight_decay = float(surrogate_cfg.get("retrain_weight_decay", 1e-4))
    retrain_hidden_dim = int(surrogate_cfg.get("retrain_hidden_dim", 128))
    retrain_num_layers = int(surrogate_cfg.get("retrain_num_layers", 4))
    retrain_dropout = float(surrogate_cfg.get("retrain_dropout", 0.1))
    retrain_standardize = bool(surrogate_cfg.get("retrain_standardize", False))
    retrain_save_dir = str(surrogate_cfg.get("retrain_save_dir", "checkpoints"))
    retrain_use_edge_attr = bool(surrogate_cfg.get("retrain_use_edge_attr", True))
    retrain_use_ligand_mask = bool(surrogate_cfg.get("retrain_use_ligand_mask", True))
    retrain_fp_dim = int(surrogate_cfg.get("retrain_fp_dim", 2048))
    retrain_fp_radius = int(surrogate_cfg.get("retrain_fp_radius", 2))
    retrain_eval_samples = int(surrogate_cfg.get("retrain_eval_samples", 1))
    retrain_scheduler = str(surrogate_cfg.get("retrain_scheduler", "none"))
    retrain_warmup_epochs = int(surrogate_cfg.get("retrain_warmup_epochs", 0))
    retrain_min_lr = float(surrogate_cfg.get("retrain_min_lr", 1e-6))
    retrain_early_stop_patience = int(surrogate_cfg.get("retrain_early_stop_patience", 0))
    retrain_early_stop_min_delta = float(surrogate_cfg.get("retrain_early_stop_min_delta", 0.0))
    surrogate_backbone = str(model_cfg.get("surrogate_backbone", "gin")).lower()
    ligand3d_cache_dir_cfg = surrogate_cfg.get("ligand3d_cache_dir")
    torchmd_base = _section(model_cfg, "torchmd")
    torchmd_head_hidden = torchmd_base.get("head_hidden_dim", None)
    torchmd_head_hidden = None if torchmd_head_hidden in (None, "", "null") else int(torchmd_head_hidden)
    torchmd_cfg = {
        "embedding_dim": int(torchmd_base.get("embedding_dim", retrain_hidden_dim)),
        "num_layers": int(torchmd_base.get("num_layers", 2)),
        "num_rbf": int(torchmd_base.get("num_rbf", 32)),
        "rbf_type": str(torchmd_base.get("rbf_type", "expnorm")),
        "trainable_rbf": bool(torchmd_base.get("trainable_rbf", False)),
        "activation": str(torchmd_base.get("activation", "silu")),
        "cutoff_lower": float(torchmd_base.get("cutoff_lower", 0.0)),
        "cutoff_upper": float(torchmd_base.get("cutoff_upper", 4.5)),
        "node_feat_dim": int(torchmd_base.get("node_feat_dim") or 0),
        "max_num_neighbors": int(torchmd_base.get("max_num_neighbors", 64)),
        "dropout": float(torchmd_base.get("dropout", retrain_dropout)),
        "reduce_op": str(torchmd_base.get("reduce", torchmd_base.get("reduce_op", "sum"))),
        "equivariance_invariance_group": str(torchmd_base.get("equivariance_group", torchmd_base.get("equivariance_invariance_group", "O(3)"))),
        "static_shapes": bool(torchmd_base.get("static_shapes", True)),
        "check_errors": bool(torchmd_base.get("check_errors", True)),
        "head_hidden_dim": torchmd_head_hidden,
        "head_num_layers": int(torchmd_base.get("head_num_layers", 3)),
        "gaussian_warmup_epochs": int(torchmd_base.get("gaussian_warmup_epochs", 5)),
        "gaussian_var_reg_beta": float(torchmd_base.get("gaussian_var_reg_beta", 1e-4)),
        "gaussian_logvar_min": float(torchmd_base.get("gaussian_logvar_min", -8.0)),
        "gaussian_logvar_max": float(torchmd_base.get("gaussian_logvar_max", 4.0)),
        "gaussian_min_var": float(torchmd_base.get("gaussian_min_var", 1e-6)),
    }
    dock_sign = float(objective_cfg.get("dock_sign", -1.0))
    dock_valid_max = objective_cfg.get("dock_valid_max", 0.0)
    dock_valid_max = None if dock_valid_max is None else float(dock_valid_max)
    qed_sign = float(objective_cfg.get("qed_sign", 1.0))
    sa_sign = float(objective_cfg.get("sa_sign", -1.0))
    use_sa = bool(objective_cfg.get("use_sa", True))
    sa_clamp_min = objective_cfg.get("sa_clamp_min", None)
    sa_clamp_max = objective_cfg.get("sa_clamp_max", None)
    sa_clamp_min = None if sa_clamp_min is None else float(sa_clamp_min)
    sa_clamp_max = None if sa_clamp_max is None else float(sa_clamp_max)
    # Extra objectives beyond (dock, qed, sa). These are deterministic RDKit-based metrics.
    extra_obj_cfg = objective_cfg.get("extra_objectives", [])
    extra_objectives: list[str] = []
    if extra_obj_cfg not in (None, "", "null"):
        if isinstance(extra_obj_cfg, (list, tuple)):
            extra_objectives = [str(x).strip().lower() for x in extra_obj_cfg if str(x).strip()]
        else:
            extra_objectives = [x.strip().lower() for x in str(extra_obj_cfg).split(",") if x.strip()]
    # Keep only supported names in a stable order.
    supported_extra = ["sim_prior", "win", "hb", "fsp3"]
    extra_objectives = [x for x in supported_extra if x in set(extra_objectives)]
    sim_prior_sign = float(objective_cfg.get("sim_prior_sign", 1.0))
    win_sign = float(objective_cfg.get("win_sign", 1.0))
    hb_sign = float(objective_cfg.get("hb_sign", 1.0))
    fsp3_sign = float(objective_cfg.get("fsp3_sign", 1.0))
    ref_point_cfg = objective_cfg.get("ref_point", None)
    ref_point = None
    if ref_point_cfg not in (None, "", "null"):
        if isinstance(ref_point_cfg, (list, tuple)):
            ref_point = [float(x) for x in ref_point_cfg if x not in (None, "")]
        else:
            ref_point = [float(x.strip()) for x in str(ref_point_cfg).split(",") if x.strip()]
    if not ref_point:
        ref_point = build_ref_point_from_objective(
            dock_sign=dock_sign,
            qed_sign=qed_sign,
            sa_sign=sa_sign,
            use_sa=use_sa,
            dock_valid_max=dock_valid_max,
            sa_clamp_max=sa_clamp_max,
        )
    base_dim = 2 + (1 if use_sa else 0)
    expected_dim = base_dim + len(extra_objectives)
    objective_names = ["dock", "qed"] + (["sa"] if use_sa else []) + list(extra_objectives)
    if ref_point is not None and len(ref_point) != expected_dim:
        raise ValueError(f"objective.ref_point length {len(ref_point)} != expected {expected_dim}")
    if weights is not None and len(weights) != expected_dim:
        raise ValueError(f"selection.weights length {len(weights)} != expected {expected_dim}")
    fallback_weights = list(weights) if weights is not None else [1.0] * expected_dim
    # Acquisition (qPMHI/qNEHVI/HV) uses boolean 0/1 masks per objective.
    # Fallback ranking and oracle summaries must use the exact same active objective set.
    acq_weights = [1.0 if float(w) > 0.0 else 0.0 for w in fallback_weights]
    if not any(w > 0.0 for w in acq_weights):
        raise ValueError("selection.weights must contain at least one positive value.")
    active_hvi_objectives = [name for name, w in zip(objective_names, acq_weights) if w > 0.0]
    active_extra_objectives = [name for name in extra_objectives if name in active_hvi_objectives]
    active_selection_weights = [float(w) for name, w in zip(objective_names, fallback_weights) if name in active_hvi_objectives]
    hvi_ref_point = None if ref_point is None else [v for v, w in zip(ref_point, acq_weights) if w > 0.0]

    # Similarity-to-prior is only needed if it's part of the active objective space.
    prior_state = _build_prior_state(repo_root, objective_cfg) if "sim_prior" in active_extra_objectives else None
    acq_backend = str(selection_cfg.get("acq_backend", "auto")).lower()
    resample_max_rounds = int(selection_cfg.get("resample_max_rounds", 1))
    qnehvi_min_gain = selection_cfg.get("qnehvi_min_gain", None)
    qnehvi_min_gain = None if qnehvi_min_gain is None else float(qnehvi_min_gain)
    qnehvi_decay = float(selection_cfg.get("qnehvi_decay", 0.5))
    runs_root = Path(str(run_cfg.get("runs_dir", "runs")))
    run_tag_cfg = run_cfg.get("run_tag")
    run_tag = str(run_tag_cfg) if run_tag_cfg else datetime.now().strftime("%Y%m%d_%H%M%S")
    copy_dataset_to_run = bool(run_cfg.get("run_dataset_copy", False))
    save_smiles_each_iter = bool(run_cfg.get("save_smiles_each_iter", False))
    save_top_pose_enabled = bool(run_cfg.get("save_top_pose_enabled", True))
    save_top_pose_pct = float(run_cfg.get("save_top_pose_pct", 0.1))
    save_top_pose_absmax = run_cfg.get("save_top_pose_absmax", 100.0)
    save_top_pose_absmax = None if save_top_pose_absmax is None else float(save_top_pose_absmax)
    save_plots_on_exit = bool(run_cfg.get("save_plots_on_exit", True))
    plot_max_points = int(run_cfg.get("plot_max_points", 5000))
    plot_dock_absmax = run_cfg.get("plot_dock_absmax", 100.0)
    plot_dock_absmax = None if plot_dock_absmax is None else float(plot_dock_absmax)
    dock_fail_absmax = run_cfg.get("dock_fail_absmax", None)
    dock_fail_absmax = None if dock_fail_absmax is None else float(dock_fail_absmax)
    candidate_3d_cfg = _section(candidate_cfg, "candidate_3d")
    candidate_3d_num_confs = int(candidate_3d_cfg.get("num_confs", 1))
    candidate_3d_max_attempts = int(candidate_3d_cfg.get("max_attempts", 3))
    candidate_3d_max_opt_iters = int(candidate_3d_cfg.get("max_opt_iters", 50))
    candidate_3d_optimize = bool(candidate_3d_cfg.get("optimize", False))
    candidate_3d_prefer_mmff = bool(candidate_3d_cfg.get("prefer_mmff", False))
    candidate_3d_workers = int(candidate_3d_cfg.get("workers", 4))
    candidate_3d_chunksize = int(candidate_3d_cfg.get("chunksize", 16))
    oracle_3d_cfg = _section(oracle_cfg, "oracle_3d")
    oracle_3d_num_confs = int(oracle_3d_cfg.get("num_confs", candidate_3d_num_confs))
    oracle_3d_max_attempts = int(oracle_3d_cfg.get("max_attempts", candidate_3d_max_attempts))
    oracle_3d_max_opt_iters = int(oracle_3d_cfg.get("max_opt_iters", candidate_3d_max_opt_iters))
    oracle_3d_optimize = bool(oracle_3d_cfg.get("optimize", candidate_3d_optimize))
    oracle_3d_prefer_mmff = bool(oracle_3d_cfg.get("prefer_mmff", candidate_3d_prefer_mmff))
    run_dir = _make_run_dir(runs_root, run_tag)
    dataset_root_path = Path(dataset_root)
    orig_dataset_root = dataset_root_path
    run_out_csv = run_dir / sanitize_out_csv(out_csv).name
    retrain_save_dir_path = Path(retrain_save_dir)
    if not retrain_save_dir_path.is_absolute():
        retrain_save_dir = str(run_dir / retrain_save_dir_path)
    clear_generated = bool(run_cfg.get("clear_generated_before_run", False))
    generated_prefix = str(run_cfg.get("generated_prefix", "GEN"))
    ligand_vocab_override = cfg.get("ligand_vocab_override")
    if ligand_vocab_override:
        if isinstance(ligand_vocab_override, (list, tuple)):
            ligand_vocab_override = [str(x).strip() for x in ligand_vocab_override if str(x).strip()]
        else:
            ligand_vocab_override = [s.strip() for s in str(ligand_vocab_override).split(",") if s.strip()]
    else:
        vocab_path = Path(dataset_root) / "ligand_vocab.json"
        if vocab_path.exists():
            try:
                with vocab_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, list):
                    ligand_vocab_override = [str(x).strip() for x in data if str(x).strip()]
                else:
                    ligand_vocab_override = None
            except Exception:
                ligand_vocab_override = None
        else:
            ligand_vocab_override = None
    vina_executable = run_cfg.get("vina_executable")
    if vina_executable:
        vina_executable = str(Path(vina_executable).expanduser())

    # Reference ligand full/pocket diagnostic docking is expensive; run it once per MOBO run.
    oracle_reference_evaluated = False

    if oracle_prepare_dataset:
        # Allow specifying external oracle inputs via config (copied into dataset_root).
        _ensure_oracle_assets_from_config(orig_dataset_root, oracle_cfg)
        csv_path = str(orig_dataset_root / "smiles.csv")
        protein_pdb = orig_dataset_root / "protein.pdb"
        if not protein_pdb.exists():
            raise FileNotFoundError(f"protein.pdb missing at {protein_pdb}")
        pocket_pdb = orig_dataset_root / "pocket.pdb"
        pocket_pdbqt = pocket_pdb.with_suffix(".pdbqt")
        ref_lig = orig_dataset_root / "reference_ligand.sdf"
        if oracle_ref_resname and not ref_lig.exists():
            ok = _extract_reference_ligand_sdf(
                protein_pdb,
                oracle_ref_resname,
                ref_lig,
                default_altloc=oracle_meeko_default_altloc,
            )
            if not ok:
                raise RuntimeError(f"Failed to extract reference ligand '{oracle_ref_resname}' from {protein_pdb}")
        if oracle_mark_reference:
            ok, action = _mark_reference_in_csv(orig_dataset_root, oracle_ref_id)
            if not ok:
                raise RuntimeError(f"Failed to mark reference ligand in smiles.csv: {action}")
        if oracle_fill_qed_sa:
            stats = _fill_qed_sa_in_csv(orig_dataset_root, force=False)
            log_kv("qed_sa_rows", stats.get("rows", 0))
            log_kv("qed_updated", stats.get("qed_updated", 0))
            log_kv("sa_updated", stats.get("sa_updated", 0))
            log_kv("qed_sa_fail", stats.get("fail", 0))
        # Ensure extra deterministic metrics exist for objective/analysis (including sim_prior if enabled).
        extra_stats = _fill_extra_metrics_in_csv(orig_dataset_root, objective_cfg, prior_state, force=False)
        if extra_stats.get("rows", 0) > 0:
            log_kv("extra_rows", extra_stats.get("rows", 0))
            log_kv("extra_updated", extra_stats.get("updated", 0))
            log_kv("extra_fail", extra_stats.get("fail", 0))
        missing_ids = _collect_missing_dock_ids(csv_path, dock_valid_max=dock_valid_max)
        need_pocket = (not pocket_pdb.exists()) or (not pocket_pdbqt.exists())
        if need_pocket or missing_ids:
            log_section("oracle_prepare")
            log_kv("protein_pdb", str(protein_pdb))
            log_kv("pocket_missing", need_pocket)
            log_kv("missing_dock", len(missing_ids))
            log_kv("reference_ligand", str(ref_lig) if ref_lig.exists() else "missing")
            if oracle_mark_reference:
                log_kv("reference_id", oracle_ref_id)
            run_oracle_docking(
                str(orig_dataset_root),
                ligand_ids=missing_ids,
                vina_executable=vina_executable,
                overwrite=oracle_overwrite,
                exhaustiveness=oracle_exhaustiveness,
                pocket_radius=oracle_pocket_radius,
                confgen_max_attempts=oracle_3d_max_attempts,
                confgen_seed=0,
                confgen_num_confs=oracle_3d_num_confs,
                confgen_max_opt_iters=oracle_3d_max_opt_iters,
                confgen_optimize=oracle_3d_optimize,
                confgen_prefer_mmff=oracle_3d_prefer_mmff,
                meeko_allow_bad_res=oracle_meeko_allow_bad_res,
                meeko_default_altloc=oracle_meeko_default_altloc,
                evaluate_reference=not oracle_reference_evaluated,
            )
            oracle_reference_evaluated = True

    if copy_dataset_to_run:
        run_dataset_root = run_dir / orig_dataset_root.parent.name / orig_dataset_root.name
        log_section("dataset_copy")
        log_kv("src", str(orig_dataset_root))
        log_kv("dst", str(run_dataset_root))
        if run_dataset_root.exists():
            shutil.rmtree(run_dataset_root)
        shutil.copytree(orig_dataset_root, run_dataset_root)
        dataset_root = str(run_dataset_root)
        dataset_root_path = run_dataset_root

    vae, vocab, vae_cfg = load_vae(vae_ckpt, device)
    token_type = vae_cfg["data"].get("token_type", "selfies")
    max_len = vae_cfg["data"]["max_len"]
    latent_dim = vae_cfg["model"]["latent_dim"]

    log_section("mobo_config")
    log_kv("pool", pool_size)
    log_kv("batch", batch_size)
    log_kv("mc_samples", mc_samples_n)
    log_kv("sample_batch", sample_batch)
    log_kv("iters", iterations)
    log_kv("run_oracle", run_oracle)
    log_kv("retrain", retrain)
    log_kv("device", device)
    log_kv("surrogate_backbone", surrogate_backbone)
    log_kv("dock_valid_max", dock_valid_max if dock_valid_max is not None else "none")
    if ref_point is not None:
        log_kv("hv_ref_point", ",".join(f"{v:.6g}" for v in ref_point))
    log_kv("acq_backend", acq_backend)
    log_kv("resample_max_rounds", resample_max_rounds)
    log_kv("qnehvi_min_gain", qnehvi_min_gain if qnehvi_min_gain is not None else "none")
    log_kv("qnehvi_decay", qnehvi_decay)
    log_kv("fallback_objectives", ",".join(objective_names))
    log_kv("hvi_weights_01", ",".join(str(int(w > 0.0)) for w in acq_weights))
    log_kv("hvi_objectives_active", ",".join(active_hvi_objectives))
    log_kv("fallback_weights", ",".join(f"{w:.3g}" for w in fallback_weights))
    log_kv("active_objectives", ",".join(active_hvi_objectives))
    log_kv("active_selection_weights", ",".join(f"{w:.3g}" for w in active_selection_weights))
    if dock_fail_absmax is not None:
        log_kv("dock_fail_absmax", dock_fail_absmax)
    if run_oracle:
        log_kv("oracle_exhaustiveness", oracle_exhaustiveness)
        log_kv("oracle_pocket_radius", oracle_pocket_radius)
        if oracle_ref_resname:
            log_kv("oracle_ref_resname", oracle_ref_resname)
        if oracle_prepare_dataset:
            log_kv("oracle_prepare_dataset", oracle_prepare_dataset)
        if oracle_fill_qed_sa:
            log_kv("oracle_fill_qed_sa", oracle_fill_qed_sa)
        if oracle_mark_reference:
            log_kv("oracle_ref_id", oracle_ref_id)
        log_kv(
            "oracle_3d",
            f"num_confs={oracle_3d_num_confs} "
            f"max_attempts={oracle_3d_max_attempts} "
            f"max_opt_iters={oracle_3d_max_opt_iters} "
            f"optimize={oracle_3d_optimize} "
            f"prefer_mmff={oracle_3d_prefer_mmff}",
        )
    log_kv("dataset_root", dataset_root)
    log_kv("run_dir", str(run_dir))
    if save_top_pose_enabled:
        log_kv("save_top_pose_pct", save_top_pose_pct)
    if surrogate_backbone == "tensornet":
            log_kv(
                "candidate_3d",
                f"num_confs={candidate_3d_num_confs} "
                f"max_attempts={candidate_3d_max_attempts} "
                f"max_opt_iters={candidate_3d_max_opt_iters} "
                f"optimize={candidate_3d_optimize} "
                f"prefer_mmff={candidate_3d_prefer_mmff} "
                f"workers={candidate_3d_workers} "
                f"chunksize={candidate_3d_chunksize}",
            )

    # Snapshot config and starting smiles for reproducibility.
    try:
        shutil.copy2("config/surrogate/config.yaml", run_dir / "config.yaml")
    except Exception:
        pass
    try:
        smiles_src = Path(dataset_root) / "smiles.csv"
        if smiles_src.exists():
            shutil.copy2(smiles_src, run_dir / "smiles_start.csv")
    except Exception:
        pass
    if clear_generated:
        csv_path = str(Path(dataset_root) / "smiles.csv")
        removed = purge_generated_rows(csv_path, generated_prefix, dataset_root)
        log_section("cleanup")
        log_kv("removed", removed)
        log_kv("prefix", generated_prefix)
        base_out = run_out_csv
        out_dir = base_out.parent / f"{base_out.stem}_iters"
        if out_dir.exists():
            if out_dir.is_dir():
                shutil.rmtree(out_dir)
            else:
                out_dir.unlink()
            log_kv("removed_iter_dir", out_dir)
        else:
            log_kv("removed_iter_dir", "none")

    # remove test ratio: test split handled by current-iter docked molecules
    split_ratio_used = (float(split_ratio[0]), float(split_ratio[1]), 0.0)

    # One-time split (train/valid only). Test split is reserved for current-iter docked molecules.
    if resplit:
        log_section("split")
        csv_path = str(Path(dataset_root) / "smiles.csv")
        resplit_smiles_csv(csv_path, split_ratio_used, seed=split_seed)
        _ensure_added_iter(csv_path)
        log_kv("resplit_mode", "initial+incremental")
        log_kv("split_ratio", split_ratio_used)
        log_kv("split_seed", split_seed)

    # Build an in-memory 3D store once and incrementally update it after each oracle round.
    ligand3d_store = None
    if surrogate_backbone in {"tensornet", "tsa", "tensor"}:
        log_section("ligand3d_store")
        if ligand3d_cache_dir_cfg:
            ligand3d_cache_dir = Path(ligand3d_cache_dir_cfg)
        else:
            ligand3d_cache_dir = None
        ligand3d_store = LigandOnly3DStore(
            dataset_root,
            ligand_vocab_override=ligand_vocab_override,
            atom_extra_dim=ATOM_EXTRA_DIM,
            cache_dir=ligand3d_cache_dir,
            fp_dim=retrain_fp_dim,
            fp_radius=retrain_fp_radius,
            confgen_max_attempts=candidate_3d_max_attempts,
            confgen_seed=0,
            confgen_num_confs=candidate_3d_num_confs,
            confgen_max_opt_iters=candidate_3d_max_opt_iters,
            confgen_optimize=candidate_3d_optimize,
            confgen_prefer_mmff=candidate_3d_prefer_mmff,
        )
        log_kv("store_total", len(ligand3d_store.data_list))
        log_kv(
            "store_splits",
            f"train={len(ligand3d_store.split_indices.get('train', []))} "
            f"valid={len(ligand3d_store.split_indices.get('valid', []))} "
            f"test={len(ligand3d_store.split_indices.get('test', []))}",
        )

    log_section("init_surrogate")
    Path(retrain_save_dir).mkdir(parents=True, exist_ok=True)
    init_path = Path(retrain_save_dir) / f"{surrogate_backbone}_surrogate_iter0.pt"
    if ligand3d_cache_dir_cfg:
        ligand3d_cache_dir = Path(ligand3d_cache_dir_cfg)
    else:
        ligand3d_cache_dir = Path(dataset_root) / "processed" / "ligand3d_cache"
    surrogate_ckpts = [
        train_surrogate_from_scratch(
            dataset_root,
            save_path=str(init_path),
            epochs=retrain_epochs,
            batch_size=retrain_batch_size,
            lr=retrain_lr,
            weight_decay=retrain_weight_decay,
            hidden_dim=retrain_hidden_dim,
            num_layers=retrain_num_layers,
            dropout=retrain_dropout,
            use_edge_attr=retrain_use_edge_attr,
            use_ligand_mask=retrain_use_ligand_mask,
            standardize=retrain_standardize,
            fp_dim=retrain_fp_dim,
            fp_radius=retrain_fp_radius,
            eval_samples=retrain_eval_samples,
            scheduler=retrain_scheduler,
            warmup_epochs=retrain_warmup_epochs,
            min_lr=retrain_min_lr,
            early_stop_patience=retrain_early_stop_patience,
            early_stop_min_delta=retrain_early_stop_min_delta,
            device=device,
            dock_valid_max=dock_valid_max,
            backbone=surrogate_backbone,
            torchmd_cfg=torchmd_cfg,
            ligand3d_cache_dir=ligand3d_cache_dir,
            ligand_vocab_override=ligand_vocab_override,
            ligand3d_store=ligand3d_store,
            confgen_max_attempts=candidate_3d_max_attempts,
            confgen_seed=0,
            confgen_num_confs=candidate_3d_num_confs,
            confgen_max_opt_iters=candidate_3d_max_opt_iters,
            confgen_optimize=candidate_3d_optimize,
            confgen_prefer_mmff=candidate_3d_prefer_mmff,
        )
    ]
    log_kv("init_ckpt", surrogate_ckpts[0])

    for iteration in range(1, iterations + 1):
        iter_start = time.perf_counter()
        log_section(f"iteration {iteration}/{iterations}")
        if len(surrogate_ckpts) != 1:
            raise RuntimeError("Expected a single surrogate checkpoint.")
        (
            model,
            mean,
            std,
            model_node_dim,
            model_edge_dim,
            model_fp_dim,
            model_fp_radius,
            model_atom_extra_dim,
            model_bond_extra_dim,
            model_pocket_graph,
            surrogate_kind,
            surrogate_meta,
        ) = load_surrogate(surrogate_ckpts[0], device)
        log_kv("surrogate_backbone", surrogate_kind)
        if surrogate_kind != surrogate_backbone:
            raise RuntimeError(
                f"Surrogate backbone mismatch: config={surrogate_backbone} "
                f"but checkpoint={surrogate_kind}."
            )
        log_model_param_report(model)
        existing_smiles = load_existing_smiles_set(str(Path(dataset_root) / "smiles.csv"))
        selected_smiles: list[str] = []
        selected_pred_dock: list[float] = []
        seen_smiles: set[str] = set()
        sampled_smiles: set[str] = set()
        iter_rows: list[dict] = []
        last_kept_smiles: list[str] | None = None
        last_pred_dock: list[float] | None = None
        last_pred_dock_std: list[float] | None = None
        last_qed_vals: list[float] | None = None
        last_sa_vals: list[float] | None = None
        last_extra_metrics: dict[str, list[float | None]] | None = None
        min_gain_round = qnehvi_min_gain
        max_rounds = max(resample_max_rounds, 1)
        round_idx = 0

        # Resample rounds: keep drawing candidates until we fill the oracle batch or hit max rounds.
        while len(selected_smiles) < batch_size and round_idx < max_rounds:
            round_idx += 1
            log_section(f"round {round_idx}/{max_rounds}")
            log_kv("selected_so_far", len(selected_smiles))
            if min_gain_round is not None:
                log_kv("qnehvi_min_gain", f"{min_gain_round:.6g}")

            # Step 1: sample candidate molecules from the VAE.
            log_section("step 1/6 candidate_pool")
            step_start = time.perf_counter()
            exclude_smiles = set(existing_smiles) | sampled_smiles
            log_kv("existing_smiles", len(existing_smiles))
            log_kv("sampled_smiles", len(sampled_smiles))
            candidates = build_candidate_pool(
                vae,
                vocab,
                token_type,
                latent_dim,
                max_len,
                pool_size,
                sample_batch,
                temperature,
                top_k,
                device,
                exclude_smiles=exclude_smiles,
                log_prefix="candidate_pool: ",
            )
            if not candidates:
                raise RuntimeError("No valid SMILES generated from VAE.")
            sampled_smiles.update(candidates)
            log_kv("step_time_sec", f"{time.perf_counter() - step_start:.2f}")

            # Step 2: build graphs + compute QED/SA.
            log_section("step 2/6 graphs + qed")
            step_start = time.perf_counter()
            smiles_df = _read_smiles_csv(str(Path(dataset_root) / "smiles.csv"))
            smiles_col = _pick_smiles_column(smiles_df.columns)
            if smiles_col is None:
                raise RuntimeError("smiles.csv missing smiles column for vocab inference.")
            dataset_ligand_vocab = list(ligand_vocab_override)
            dataset_atom_extra_dim = ATOM_EXTRA_DIM
            dataset_node_dim = len(dataset_ligand_vocab) + dataset_atom_extra_dim
            dataset_edge_dim = BOND_TYPE_CLASSES + BOND_EXTRA_DIM
            dataset_residue_vocab = []
            log_kv("ligand_vocab", len(dataset_ligand_vocab))
            log_kv("node_dim", dataset_node_dim)
            log_kv("edge_dim", dataset_edge_dim)

            use_torchmd = surrogate_kind == "tensornet"
            if use_torchmd:
                dataset_atom_extra_dim = ATOM_EXTRA_DIM
                node_feat_dim = int(surrogate_meta.get("node_feat_dim", 0)) if isinstance(surrogate_meta, dict) else 0
                atom_index = {sym: i for i, sym in enumerate(dataset_ligand_vocab)}
                atom_feat_dim = len(dataset_ligand_vocab) + dataset_atom_extra_dim
                if node_feat_dim <= 0:
                    node_feat_dim = atom_feat_dim
                if node_feat_dim != atom_feat_dim:
                    raise RuntimeError(
                        f"TensorNet node_feat_dim mismatch: ckpt={node_feat_dim} dataset={atom_feat_dim}."
                    )
                model_atom_extra_dim = dataset_atom_extra_dim
                graphs_all, kept_smiles_all = build_graphs_3d(
                    candidates,
                    atom_index=atom_index,
                    atom_feat_dim=node_feat_dim,
                    atom_extra_dim=dataset_atom_extra_dim,
                    fp_dim=model_fp_dim,
                    fp_radius=model_fp_radius,
                    max_attempts=candidate_3d_max_attempts,
                    seed=0,
                    num_confs=candidate_3d_num_confs,
                    max_opt_iters=candidate_3d_max_opt_iters,
                    optimize=candidate_3d_optimize,
                    prefer_mmff=candidate_3d_prefer_mmff,
                    progress_desc="candidate_3d",
                    num_workers=candidate_3d_workers,
                    mp_chunksize=candidate_3d_chunksize,
                )
                log_kv("node_feat_dim", node_feat_dim)
            else:
                if dataset_node_dim != model_node_dim:
                    log_kv("node_dim_mismatch", f"dataset={dataset_node_dim} model={model_node_dim}")
                if dataset_edge_dim != model_edge_dim:
                    log_kv("edge_dim_mismatch", f"dataset={dataset_edge_dim} model={model_edge_dim}")

                dataset_atom_extra_dim = ATOM_EXTRA_DIM
                dataset_residue_dim = len(dataset_residue_vocab)
                if model_bond_extra_dim <= 0 and model_edge_dim > BOND_TYPE_CLASSES:
                    model_bond_extra_dim = model_edge_dim - BOND_TYPE_CLASSES
                inferred_pocket = model_pocket_graph
                if not inferred_pocket and dataset_residue_dim > 0:
                    expected_ligand_dim = len(dataset_ligand_vocab) + max(
                        model_atom_extra_dim, dataset_atom_extra_dim, 0
                    )
                    if model_node_dim > expected_ligand_dim:
                        inferred_pocket = True
                if inferred_pocket and dataset_residue_dim > 0:
                    model_ligand_feat_dim = max(model_node_dim - dataset_residue_dim, 0)
                else:
                    model_ligand_feat_dim = model_node_dim
                if model_atom_extra_dim <= 0 and model_ligand_feat_dim > len(dataset_ligand_vocab):
                    model_atom_extra_dim = max(model_ligand_feat_dim - len(dataset_ligand_vocab), 0)
                model_atom_extra_dim = min(model_atom_extra_dim, model_ligand_feat_dim)
                expected_atom_vocab_dim = max(int(model_ligand_feat_dim) - int(model_atom_extra_dim), 0)
                log_kv("atom_extra_dim", model_atom_extra_dim)
                log_kv("bond_extra_dim", model_bond_extra_dim)
                if inferred_pocket != model_pocket_graph:
                    log_kv("pocket_graph_infer", inferred_pocket)
                if ligand_vocab_override:
                    ligand_vocab = list(ligand_vocab_override)
                    if expected_atom_vocab_dim and len(ligand_vocab) != expected_atom_vocab_dim:
                        raise RuntimeError(
                            f"ligand_vocab_override length {len(ligand_vocab)} "
                            f"does not match expected atom vocab dim {expected_atom_vocab_dim}."
                        )
                else:
                    ligand_vocab = list(dataset_ligand_vocab)
                    if expected_atom_vocab_dim and len(ligand_vocab) != expected_atom_vocab_dim:
                        if expected_atom_vocab_dim < len(ligand_vocab):
                            raise RuntimeError(
                                f"Surrogate expects atom vocab dim {expected_atom_vocab_dim}, "
                                f"but dataset provides {len(ligand_vocab)}. "
                                "Please retrain surrogate or set mobo.ligand_vocab_override."
                            )
                        defaults = ["B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
                        for sym in sorted(defaults):
                            if sym not in ligand_vocab:
                                ligand_vocab.append(sym)
                                if len(ligand_vocab) >= expected_atom_vocab_dim:
                                    break
                        if len(ligand_vocab) != expected_atom_vocab_dim:
                            raise RuntimeError(
                                f"Could not infer ligand vocab to match expected dim {expected_atom_vocab_dim}. "
                                "Please set mobo.ligand_vocab_override."
                            )
                        ligand_vocab = sorted(ligand_vocab)
                        log_kv("ligand_vocab_inferred", ",".join(ligand_vocab))

                atom_index = {sym: i for i, sym in enumerate(ligand_vocab)}
                graphs_all, kept_smiles_all = build_graphs(
                    candidates,
                    atom_index,
                    model_node_dim,
                    model_edge_dim,
                    atom_extra_dim=model_atom_extra_dim,
                    bond_extra_dim=model_bond_extra_dim,
                    fp_dim=model_fp_dim,
                    fp_radius=model_fp_radius,
                )
            qed_tensor, sa_tensor = compute_qed_sa(
                graphs_all,
                kept_smiles_all,
                use_sa=use_sa,
                sa_clamp_min=sa_clamp_min,
                sa_clamp_max=sa_clamp_max,
            )
            extra_metrics = _compute_extra_metrics_for_smiles(kept_smiles_all, objective_cfg, prior_state)
            sim_prior_tensor = torch.tensor(extra_metrics["sim_prior"], dtype=torch.float32)
            win_tensor = torch.tensor(extra_metrics["win"], dtype=torch.float32)
            hb_tensor = torch.tensor(extra_metrics["hb"], dtype=torch.float32)
            fsp3_tensor = torch.tensor(extra_metrics["fsp3"], dtype=torch.float32)
            # Replace NaNs with conservative worst-case values for deterministic objectives in [0,1].
            sim_prior_tensor = torch.nan_to_num(sim_prior_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            win_tensor = torch.nan_to_num(win_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            hb_tensor = torch.nan_to_num(hb_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            fsp3_tensor = torch.nan_to_num(fsp3_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            graphs = list(graphs_all)
            kept_smiles = list(kept_smiles_all)
            if not graphs:
                raise RuntimeError("No candidate SMILES could be featurized with the surrogate vocab.")
            log_kv("graphs_built", len(graphs_all))
            log_kv("qed_scored", len(graphs))
            log_kv("dropped", 0)
            log_tensor_stats("qed", qed_tensor)
            if use_sa:
                log_tensor_stats("sa", sa_tensor)
            if "sim_prior" in active_extra_objectives:
                log_tensor_stats("sim_prior", sim_prior_tensor)
            if "fsp3" in active_extra_objectives:
                log_tensor_stats("fsp3", fsp3_tensor)
            if "win" in active_extra_objectives:
                log_tensor_stats("win", win_tensor)
            if "hb" in active_extra_objectives:
                log_tensor_stats("hb", hb_tensor)
            log_kv("step_time_sec", f"{time.perf_counter() - step_start:.2f}")

            # Step 3: surrogate posterior evaluation.
            log_section("step 3/6 surrogate posterior")
            step_start = time.perf_counter()
            log_kv("surrogate_ckpt", surrogate_ckpts[0])

            loader = DataLoader(graphs, batch_size=1024, shuffle=False)
            samples = None
            if acq_backend == "qpmhi":
                if active_extra_objectives:
                    raise RuntimeError("Analytic qPMHI currently supports only docking, QED, and -SA objectives.")
                pred_mean_raw, pred_std_raw = predict_gaussian_stats(model, loader, mean, std, device)
                mu = torch.stack(
                    [
                        pred_mean_raw * dock_sign,
                        qed_tensor * qed_sign,
                        sa_tensor * sa_sign,
                    ],
                    dim=-1,
                )
                sigma = torch.zeros_like(mu)
                sigma[:, 0] = pred_std_raw
                best = mu
                pred_dock_mean = pred_mean_raw.detach().cpu().tolist()
                pred_dock_best = list(pred_dock_mean)
                pred_dock_std = pred_std_raw.detach().cpu().tolist()
            else:
                def sampler(pool):
                    if len(pool) != len(kept_smiles):
                        raise ValueError("Candidate pool size mismatch after graph filtering.")
                    dock_draws = mc_samples(model, loader, mean, std, mc_samples_n, device)
                    dock_draws = dock_draws * dock_sign
                    qed_obj = (qed_tensor * qed_sign).view(1, -1).expand_as(dock_draws)
                    objs = [dock_draws, qed_obj]
                    if use_sa:
                        sa_obj = (sa_tensor * sa_sign).view(1, -1).expand_as(dock_draws)
                        objs.append(sa_obj)
                    if "sim_prior" in active_extra_objectives:
                        sim_obj = (sim_prior_tensor * sim_prior_sign).view(1, -1).expand_as(dock_draws)
                        objs.append(sim_obj)
                    if "win" in active_extra_objectives:
                        w_obj = (win_tensor * win_sign).view(1, -1).expand_as(dock_draws)
                        objs.append(w_obj)
                    if "hb" in active_extra_objectives:
                        hb_obj = (hb_tensor * hb_sign).view(1, -1).expand_as(dock_draws)
                        objs.append(hb_obj)
                    if "fsp3" in active_extra_objectives:
                        f_obj = (fsp3_tensor * fsp3_sign).view(1, -1).expand_as(dock_draws)
                        objs.append(f_obj)
                    return torch.stack(objs, dim=-1)

                samples = sampler(kept_smiles)
                log_kv("samples", f"S={samples.size(0)} N={samples.size(1)} D={samples.size(2)}")
                mu = samples.mean(0)
                best = samples.max(0).values
                sigma = samples.std(0, unbiased=False)
                pred_dock_mean = (mu[:, 0] * dock_sign).tolist()
                pred_dock_best = (best[:, 0] * dock_sign).tolist()
                pred_dock_std = sigma[:, 0].tolist()

            y_train = _load_train_objectives_extended(
                dataset_root=dataset_root,
                split="train",
                dock_sign=dock_sign,
                qed_sign=qed_sign,
                sa_sign=sa_sign,
                use_sa=use_sa,
                extra_objectives=active_extra_objectives,
                sim_prior_sign=sim_prior_sign,
                win_sign=win_sign,
                hb_sign=hb_sign,
                fsp3_sign=fsp3_sign,
                dock_valid_max=dock_valid_max,
                sa_clamp_min=sa_clamp_min,
                sa_clamp_max=sa_clamp_max,
                objective_cfg=objective_cfg,
                prior_state=prior_state,
            )
            log_tensor_stats("mu_dock", mu[:, 0])
            log_tensor_stats("sigma_dock", sigma[:, 0])
            log_kv("step_time_sec", f"{time.perf_counter() - step_start:.2f}")

            # Step 4: acquisition selection (analytic qPMHI / qNEHVI).
            log_section("step 4/6 qpmhi select")
            step_start = time.perf_counter()
            selected_round: list[int] = []
            acq_scores_round = None
            if acq_backend == "qnehvi":
                if samples is None:
                    raise RuntimeError("qNEHVI requires Monte Carlo samples.")
                scores = qnehvi_scores_from_samples_nd(
                    samples,
                    y_train=y_train,
                    weights=acq_weights,
                    ref_point=ref_point,
                    show_progress=True,
                )
                acq_scores_round = scores.detach().cpu().tolist()
                order = torch.argsort(scores, descending=True).tolist()
                order = sorted(order, key=lambda i: (-acq_scores_round[i], pred_dock_mean[i]))
                nonzero = int((scores > 0.0).sum().item())
                log_kv("qnehvi_nonzero", nonzero)
                log_kv("qnehvi_ranked_candidates", len(order))
                for idx_t in order:
                    if len(selected_smiles) >= batch_size:
                        break
                    score = float(scores[idx_t].item())
                    if min_gain_round is not None and score < min_gain_round:
                        continue
                    smi = kept_smiles[idx_t]
                    if smi in seen_smiles:
                        continue
                    selected_round.append(idx_t)
                    seen_smiles.add(smi)
                    selected_smiles.append(smi)
                    selected_pred_dock.append(pred_dock_mean[idx_t])
                backend_mode = "qnehvi"
                log_kv("acq_backend", backend_mode)
            else:
                if acq_backend == "qpmhi":
                    scores = qphv_prob_gaussian_analytic_3d(
                        dock_mu=mu[:, 0],
                        dock_sigma=sigma[:, 0],
                        exact_obj=mu[:, 1:3],
                        y_train=y_train,
                        weights=acq_weights,
                        ref_point=ref_point,
                    )
                    acq_scores_round = scores.detach().cpu().tolist()
                    order = torch.argsort(scores, descending=True)
                    pred_dock_vec = pred_dock_mean
                    selected_before = len(selected_smiles)
                    nonzero = int((scores > 0.0).sum().item())
                    log_kv("qpmhi_nonzero", nonzero)
                    for idx_t in order.tolist():
                        if len(selected_smiles) >= batch_size:
                            break
                        if float(scores[idx_t].item()) <= 0.0:
                            break
                        smi = kept_smiles[idx_t]
                        if smi in seen_smiles:
                            continue
                        selected_round.append(idx_t)
                        seen_smiles.add(smi)
                        selected_smiles.append(smi)
                        selected_pred_dock.append(pred_dock_vec[idx_t])

                    if len(selected_smiles) == selected_before:
                        log_kv("qpmhi_degenerate", "no_positive_scores")
                        round_idx = max_rounds

                    backend_mode = "qpmhi"
                    log_kv("acq_backend", backend_mode)
                else:
                    if samples is None:
                        raise RuntimeError("Discrete MC acquisitions require Monte Carlo samples.")
                    use_botorch = acq_backend in {"auto", "botorch", "botorch_hv", "qehvi"}
                    if use_botorch and DominatedPartitioning is not None and bt_is_non_dominated is not None:
                        selected = select_qphv_indices_from_samples_nd(
                            samples,
                            batch_size,
                            y_train=y_train,
                            weights=acq_weights,
                            ref_point=ref_point,
                        )
                        backend_mode = "botorch_hv"
                        log_kv("acq_backend", backend_mode)
                    else:
                        selected = select_pareto_mc_indices_from_samples(
                            samples,
                            batch_size,
                            weights=acq_weights,
                        )
                        backend_mode = "pareto_mc"
                        log_kv("acq_backend", backend_mode)
                    pred_dock_vec = pred_dock_best if backend_mode == "botorch_hv" else pred_dock_mean
                    for idx in selected:
                        if len(selected_smiles) >= batch_size:
                            break
                        smi = kept_smiles[idx]
                        if smi in seen_smiles:
                            continue
                        selected_round.append(idx)
                        seen_smiles.add(smi)
                        selected_smiles.append(smi)
                        selected_pred_dock.append(pred_dock_vec[idx])
            selected_round_set = set(selected_round)
            mu_cpu = mu.detach().cpu()
            if locals().get("backend_mode") == "botorch_hv":
                pred_dock = pred_dock_best
            else:
                pred_dock = pred_dock_mean
            pred_qed = (mu_cpu[:, 1] * qed_sign).tolist()
            pred_sa = (mu_cpu[:, 2] * sa_sign).tolist() if use_sa and mu_cpu.size(1) >= 3 else [None] * len(kept_smiles)
            qed_vals = qed_tensor.detach().cpu().tolist()
            sa_vals = sa_tensor.detach().cpu().tolist() if use_sa else [None] * len(kept_smiles)
            if acq_scores_round is None:
                acq_scores_round = [None] * len(kept_smiles)
            for i, smi in enumerate(kept_smiles):
                # Extra deterministic metrics are computed once per candidate (RDKit) and treated as
                # "predicted == actual" fields in the iter CSV for easier analysis.
                logp_val = extra_metrics["logp"][i] if i < len(extra_metrics.get("logp", [])) else None
                tpsa_val = extra_metrics["tpsa"][i] if i < len(extra_metrics.get("tpsa", [])) else None
                hbd_val = extra_metrics["hbd"][i] if i < len(extra_metrics.get("hbd", [])) else None
                hba_val = extra_metrics["hba"][i] if i < len(extra_metrics.get("hba", [])) else None
                fsp3_val = extra_metrics["fsp3"][i] if i < len(extra_metrics.get("fsp3", [])) else None
                win_logp_val = extra_metrics["win_logp"][i] if i < len(extra_metrics.get("win_logp", [])) else None
                win_tpsa_val = extra_metrics["win_tpsa"][i] if i < len(extra_metrics.get("win_tpsa", [])) else None
                win_val = extra_metrics["win"][i] if i < len(extra_metrics.get("win", [])) else None
                hb_hbd_val = extra_metrics["hb_hbd"][i] if i < len(extra_metrics.get("hb_hbd", [])) else None
                hb_hba_val = extra_metrics["hb_hba"][i] if i < len(extra_metrics.get("hb_hba", [])) else None
                hb_val = extra_metrics["hb"][i] if i < len(extra_metrics.get("hb", [])) else None
                sim_prior_val = (
                    extra_metrics["sim_prior"][i] if i < len(extra_metrics.get("sim_prior", [])) else None
                )
                iter_rows.append(
                    {
                        "smiles": smi,
                        "round": round_idx,
                        "selected": 1 if i in selected_round_set else 0,
                        "selected_fallback": 0,
                        "pred_dock": pred_dock[i],
                        "pred_dock_std": pred_dock_std[i],
                        "pred_qed": pred_qed[i],
                        "pred_sa": pred_sa[i],
                        "qed": qed_vals[i] if i < len(qed_vals) else None,
                        "sa": sa_vals[i] if i < len(sa_vals) else None,
                        "logp": logp_val,
                        "tpsa": tpsa_val,
                        "hbd": hbd_val,
                        "hba": hba_val,
                        "fsp3": fsp3_val,
                        "win_logp": win_logp_val,
                        "win_tpsa": win_tpsa_val,
                        "win": win_val,
                        "hb_hbd": hb_hbd_val,
                        "hb_hba": hb_hba_val,
                        "hb": hb_val,
                        "sim_prior": sim_prior_val,
                        "acq_backend": acq_backend,
                        "acq_score": acq_scores_round[i],
                        "fallback_score": None,
                    }
                )

            log_kv("selected_round", len(selected_round))
            log_kv("selected_total", len(selected_smiles))
            log_kv("step_time_sec", f"{time.perf_counter() - step_start:.2f}")

            last_kept_smiles = kept_smiles
            last_pred_dock = pred_dock
            last_pred_dock_std = pred_dock_std
            last_qed_vals = [float(v) for v in qed_vals]
            if not use_sa:
                raise RuntimeError("Current selection pipeline requires SA for uncertainty fallback.")
            last_sa_vals = [float(v) for v in sa_vals]
            last_extra_metrics = {name: list(extra_metrics.get(name, [])) for name in active_extra_objectives}

            if len(selected_smiles) < batch_size and min_gain_round is not None and acq_backend == "qnehvi":
                min_gain_round *= qnehvi_decay

        if len(selected_smiles) < batch_size:
            added_fallback = 0
            fallback_smiles: list[str] = []
            if (
                last_kept_smiles is not None
                and last_pred_dock is not None
                and last_pred_dock_std is not None
                and last_qed_vals is not None
                and last_sa_vals is not None
            ):
                train_smiles = _load_train_smiles_for_fallback(dataset_root)
                order, fallback_scores = _uncertainty_fallback_order(
                    candidate_smiles=last_kept_smiles,
                    pred_dock_std=last_pred_dock_std,
                    qed_vals=last_qed_vals,
                    sa_vals=last_sa_vals,
                    train_smiles=train_smiles,
                )
                log_kv("fallback_use", "uncertainty+coverage")
                row_by_smiles = {row.get("smiles"): row for row in iter_rows}
                for idx in order:
                    if len(selected_smiles) >= batch_size:
                        break
                    smi = last_kept_smiles[idx]
                    row = row_by_smiles.get(smi)
                    if row is None:
                        raise RuntimeError(f"Missing iter row for fallback candidate: {smi}")
                    row["fallback_score"] = float(fallback_scores[idx])
                    if smi in seen_smiles:
                        continue
                    seen_smiles.add(smi)
                    selected_smiles.append(smi)
                    selected_pred_dock.append(last_pred_dock[idx])
                    row["selected"] = 1
                    row["selected_fallback"] = 1
                    added_fallback += 1
                    fallback_smiles.append(smi)
            if added_fallback:
                log_kv("selected_fallback", added_fallback)
        if len(selected_smiles) < batch_size:
            log_kv("selected_shortfall", batch_size - len(selected_smiles))

            # Step 5: save per-iter candidates + predictions.
            log_section("step 5/6 save")
        step_start = time.perf_counter()
        base_out = run_out_csv
        out_dir = base_out.parent / f"{base_out.stem}_iters"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = base_out.suffix or ".csv"
        out_csv_iter = out_dir / f"iter{iteration}{suffix}"
        with open(out_csv_iter, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "smiles",
                "round",
                "selected",
                "selected_fallback",
                "pred_dock",
                "pred_dock_std",
                "pred_qed",
                "pred_sa",
                "qed",
                "sa",
                "logp",
                "tpsa",
                "hbd",
                "hba",
                "fsp3",
                "win_logp",
                "win_tpsa",
                "win",
                "hb_hbd",
                "hb_hba",
                "hb",
                "sim_prior",
                "acq_backend",
                "acq_score",
                "fallback_score",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in iter_rows:
                writer.writerow(row)

        if last_kept_smiles is not None:
            log_kv("candidate_pool", len(last_kept_smiles))
        log_kv("selected_final", len(selected_smiles))
        log_kv("saved", out_csv_iter)
        log_kv("step_time_sec", f"{time.perf_counter() - step_start:.2f}")

        # Removed surrogate-based top-5 logs (can be misleading vs oracle).

        if run_oracle:
            # Step 6: oracle docking + update dataset / metrics.
            log_section("step 6/6 oracle")
            step_start = time.perf_counter()
            csv_path = str(Path(dataset_root) / "smiles.csv")
            added, new_ids, kept_indices = append_candidates_to_smiles_csv(
                csv_path,
                selected_smiles,
                id_prefix=oracle_id_prefix,
                sa_clamp_min=sa_clamp_min,
                sa_clamp_max=sa_clamp_max,
                split_ratio=split_ratio_used if resplit else None,
                added_iter=iteration,
            )
            log_kv("oracle_append", added)
            log_kv("csv_path", csv_path)
            if kept_indices:
                selected_pred_added = [selected_pred_dock[i] for i in kept_indices]
            else:
                selected_pred_added = []

            dock_stats = run_oracle_docking(
                dataset_root,
                new_ids,
                vina_executable=vina_executable,
                overwrite=oracle_overwrite,
                exhaustiveness=oracle_exhaustiveness,
                pocket_radius=oracle_pocket_radius,
                confgen_max_attempts=oracle_3d_max_attempts,
                confgen_seed=0,
                confgen_num_confs=oracle_3d_num_confs,
                confgen_max_opt_iters=oracle_3d_max_opt_iters,
                confgen_optimize=oracle_3d_optimize,
                confgen_prefer_mmff=oracle_3d_prefer_mmff,
                meeko_allow_bad_res=oracle_meeko_allow_bad_res,
                meeko_default_altloc=oracle_meeko_default_altloc,
                evaluate_reference=not oracle_reference_evaluated,
            )
            oracle_reference_evaluated = True
            log_kv_table(
                "oracle_docking",
                [
                    ("attempted", dock_stats.get("attempted", 0)),
                    ("docked", dock_stats.get("docked", 0)),
                    ("skipped", dock_stats.get("skipped", 0)),
                    ("failed", dock_stats.get("failed", 0)),
                ],
                cols=4,
            )
            dock_flag_stats = mark_abnormal_dock(
                csv_path,
                dock_valid_max=dock_valid_max,
                dock_abs_max=dock_fail_absmax,
            )
            if dock_flag_stats.get("flagged", 0) > 0:
                log_kv_table(
                    "dock_abnormal",
                    [
                        ("total_rows", dock_flag_stats.get("total", 0)),
                        ("flagged", dock_flag_stats.get("flagged", 0)),
                    ],
                    cols=2,
                )

            if ligand3d_store is not None and new_ids:
                update_stats = ligand3d_store.append_from_csv(csv_path, new_ids)
                log_kv_table(
                    "ligand3d_update",
                    [
                        ("added", update_stats.get("added", 0)),
                        ("failed", update_stats.get("failed", 0)),
                        ("store_total", len(ligand3d_store.data_list)),
                    ],
                    cols=3,
                )

            try:
                smiles_src = Path(dataset_root) / "smiles.csv"
                if smiles_src.exists():
                    shutil.copy2(smiles_src, run_dir / "smiles.csv")
            except Exception:
                pass
            log_kv("step_time_sec", f"{time.perf_counter() - step_start:.2f}")

            if save_smiles_each_iter:
                try:
                    smiles_src = Path(dataset_root) / "smiles.csv"
                    if smiles_src.exists():
                        shutil.copy2(smiles_src, run_dir / f"smiles_iter{iteration}.csv")
                except Exception:
                    pass

            # After oracle docking, ensure extra deterministic metrics are filled for the new rows
            # so HV/HVI and subsequent analysis can use the full objective vector.
            try:
                _fill_extra_metrics_in_csv(dataset_root_path, objective_cfg, prior_state, force=False)
            except Exception:
                pass

            eval_stats = evaluate_oracle_accuracy_from_csv(
                csv_path,
                new_ids,
                selected_pred_added,
                dock_valid_max=dock_valid_max,
            )
            eval_rows = [
                ("total_added", eval_stats.get("total", 0)),
                ("with_dock_score", eval_stats.get("matched", 0)),
                ("missing_dock_score", eval_stats.get("missing", 0)),
            ]
            if eval_stats.get("matched", 0) > 0:
                eval_rows.append(("rmse", f"{eval_stats.get('rmse', 0.0):.4f}"))
                eval_rows.append(("mae", f"{eval_stats.get('mae', 0.0):.4f}"))
                r2 = eval_stats.get("r2", None)
                if r2 is not None:
                    eval_rows.append(("r2", f"{r2:.4f}"))
                spearman = eval_stats.get("spearman", None)
                if spearman is not None and np.isfinite(spearman):
                    eval_rows.append(("spearman", f"{spearman:.4f}"))
                pearson = eval_stats.get("pearson", None)
                if pearson is not None and np.isfinite(pearson):
                    eval_rows.append(("pearson", f"{pearson:.4f}"))
                kendall = eval_stats.get("kendall", None)
                if kendall is not None and np.isfinite(kendall):
                    eval_rows.append(("kendall", f"{kendall:.4f}"))
                hit10 = eval_stats.get("hit@10", None)
                if hit10 is not None and np.isfinite(hit10):
                    eval_rows.append(("hit@10", f"{hit10:.4f}"))
                hit50 = eval_stats.get("hit@50", None)
                if hit50 is not None and np.isfinite(hit50):
                    eval_rows.append(("hit@50", f"{hit50:.4f}"))
                hit100 = eval_stats.get("hit@100", None)
                if hit100 is not None and np.isfinite(hit100):
                    eval_rows.append(("hit@100", f"{hit100:.4f}"))
                ndcg10 = eval_stats.get("ndcg@10", None)
                if ndcg10 is not None and np.isfinite(ndcg10):
                    eval_rows.append(("ndcg@10", f"{ndcg10:.4f}"))
                ndcg50 = eval_stats.get("ndcg@50", None)
                if ndcg50 is not None and np.isfinite(ndcg50):
                    eval_rows.append(("ndcg@50", f"{ndcg50:.4f}"))
                ndcg100 = eval_stats.get("ndcg@100", None)
                if ndcg100 is not None and np.isfinite(ndcg100):
                    eval_rows.append(("ndcg@100", f"{ndcg100:.4f}"))
            log_kv_table("oracle_eval", eval_rows, cols=6)
            hvi_stats = compute_hvi_from_csv(
                csv_path,
                new_ids,
                dock_sign=dock_sign,
                qed_sign=qed_sign,
                sa_sign=sa_sign,
                use_sa=use_sa,
                sa_clamp_min=sa_clamp_min,
                sa_clamp_max=sa_clamp_max,
                dock_valid_max=dock_valid_max,
                ref_point=hvi_ref_point,
                extra_objectives=active_extra_objectives,
                enabled_objectives=active_hvi_objectives,
            )
            if hvi_stats is None:
                log_kv_table("oracle_hvi", [("hvi", "unavailable")], cols=1)
            else:
                hvi_rows = [
                    ("total_rows", hvi_stats["total_rows"]),
                    ("total_with_dock", hvi_stats["total_with_dock"]),
                    ("total_points", hvi_stats["total_count"]),
                    ("pareto_points", hvi_stats["pareto_count"]),
                    ("new_points", hvi_stats["new_count"]),
                ]
                rank_stats = hvi_stats.get("rank_stats")
                if rank_stats:
                    hvi_rows.extend(
                        [
                            ("new_rank_min", rank_stats["rank_min"]),
                            ("new_rank_median", f"{rank_stats['rank_median']:.2f}"),
                            ("new_rank_mean", f"{rank_stats['rank_mean']:.2f}"),
                            ("new_dist_min", f"{rank_stats['dist_min']:.6f}"),
                            ("new_dist_median", f"{rank_stats['dist_median']:.6f}"),
                            ("new_dist_mean", f"{rank_stats['dist_mean']:.6f}"),
                        ]
                    )
                hvi_rows.extend(
                    [
                        ("hv_before", f"{hvi_stats['hv_old']:.6f}"),
                        ("hv_after", f"{hvi_stats['hv_new']:.6f}"),
                        ("hvi", f"{hvi_stats['hvi']:.6f}"),
                    ]
                )
                log_kv_table("oracle_hvi", hvi_rows, cols=6)
                metrics_path = out_dir / "iter_metrics.csv"
                append_iter_metrics(
                    metrics_path,
                    {
                        "iteration": iteration,
                        "total_rows": hvi_stats["total_rows"],
                        "total_with_dock": hvi_stats["total_with_dock"],
                        "total_points": hvi_stats["total_count"],
                        "pareto_count": hvi_stats["pareto_count"],
                        "new_points": hvi_stats["new_count"],
                        "new_rank_min": rank_stats["rank_min"] if rank_stats else "",
                        "new_rank_median": f"{rank_stats['rank_median']:.2f}" if rank_stats else "",
                        "new_rank_mean": f"{rank_stats['rank_mean']:.2f}" if rank_stats else "",
                        "new_dist_min": f"{rank_stats['dist_min']:.6f}" if rank_stats else "",
                        "new_dist_median": f"{rank_stats['dist_median']:.6f}" if rank_stats else "",
                        "new_dist_mean": f"{rank_stats['dist_mean']:.6f}" if rank_stats else "",
                        "oracle_added": eval_stats.get("total", 0),
                        "oracle_matched": eval_stats.get("matched", 0),
                        "oracle_missing": eval_stats.get("missing", 0),
                        "rmse": f"{eval_stats.get('rmse', 0.0):.6f}" if eval_stats.get("matched", 0) > 0 else "",
                        "mae": f"{eval_stats.get('mae', 0.0):.6f}" if eval_stats.get("matched", 0) > 0 else "",
                        "r2": f"{eval_stats.get('r2', 0.0):.6f}" if eval_stats.get("matched", 0) > 0 and eval_stats.get(
                            "r2", None) is not None else "",
                        "hv_before": f"{hvi_stats['hv_old']:.6f}",
                        "hv_after": f"{hvi_stats['hv_new']:.6f}",
                        "hvi": f"{hvi_stats['hvi']:.6f}",
                    },
                )

            oracle_ids, oracle_smiles, oracle_obj, _, _, oracle_sa = collect_oracle_points(
                csv_path,
                new_ids,
                dock_sign=dock_sign,
                qed_sign=qed_sign,
                sa_sign=sa_sign,
                use_sa=use_sa,
                sa_clamp_min=sa_clamp_min,
                sa_clamp_max=sa_clamp_max,
                dock_valid_max=dock_valid_max,
                extra_objectives=active_extra_objectives,
            )
            if oracle_obj.numel() == 0:
                log_table(
                    "oracle_top5",
                    ["rank", "ligand_id", "dock", "pred_dock", "qed", "sa", "score", "smiles"],
                    [],
                )
            else:
                pred_dock_vals = predict_dock_for_smiles(
                    model,
                    oracle_smiles,
                    device,
                    mean,
                    std,
                    use_torchmd,
                    atom_index=atom_index,
                    node_dim=model_node_dim,
                    edge_dim=model_edge_dim,
                    atom_extra_dim=model_atom_extra_dim,
                    bond_extra_dim=model_bond_extra_dim,
                    fp_dim=model_fp_dim,
                    fp_radius=model_fp_radius,
                    confgen_max_attempts=candidate_3d_max_attempts,
                    confgen_seed=0,
                    confgen_num_confs=candidate_3d_num_confs,
                    confgen_max_opt_iters=candidate_3d_max_opt_iters,
                    confgen_optimize=candidate_3d_optimize,
                    confgen_prefer_mmff=candidate_3d_prefer_mmff,
                )
                top5_rows = _topk_oracle_rows_weighted_pct(
                    csv_path,
                    oracle_ids,
                    oracle_smiles,
                    oracle_obj,
                    dock_sign=dock_sign,
                    qed_sign=qed_sign,
                    sa_sign=sa_sign,
                    use_sa=use_sa,
                    dock_valid_max=dock_valid_max,
                    pred_dock_vals=pred_dock_vals,
                    weights=active_selection_weights,
                    extra_objectives=active_extra_objectives,
                    sim_prior_sign=sim_prior_sign,
                    win_sign=win_sign,
                    hb_sign=hb_sign,
                    fsp3_sign=fsp3_sign,
                    top_k=5,
                )
                top_cols = ["rank", "ligand_id", "score", "rank_ds", "dock", "pred_dock", "qed"]
                if use_sa:
                    top_cols.append("sa")
                for name in active_extra_objectives:
                    top_cols.append(name)
                log_table(
                    "oracle_top5",
                    [*top_cols, "smiles"],
                    top5_rows,
                )
                top5_dock_rows = _topk_oracle_by_dock_rows(
                    oracle_ids,
                    oracle_smiles,
                    oracle_obj,
                    dock_sign=dock_sign,
                    qed_sign=qed_sign,
                    sa_sign=sa_sign,
                    sa_vals=oracle_sa,
                    pred_dock_vals=pred_dock_vals,
                    top_k=5,
                )
                log_table(
                    "oracle_top5_by_dock",
                    ["rank", "ligand_id", "dock", "pred_dock", "qed", "sa", "smiles"],
                    top5_dock_rows,
                )
                top5_pareto_rows = _topk_oracle_pareto_rows(
                    oracle_ids,
                    oracle_smiles,
                    oracle_obj,
                    dock_sign=dock_sign,
                    qed_sign=qed_sign,
                    sa_sign=sa_sign,
                    sa_vals=oracle_sa,
                    pred_dock_vals=pred_dock_vals,
                    top_k=5,
                )
                log_table(
                    "oracle_top5_pareto",
                    ["rank", "ligand_id", "dock", "pred_dock", "qed", "sa", "smiles"],
                    top5_pareto_rows,
                )

            if resplit:
                # Re-split every iteration (train/valid only). Test is reserved for current-iter molecules.
                csv_path = str(Path(dataset_root) / "smiles.csv")
                resplit_smiles_csv(csv_path, split_ratio_used, seed=split_seed + iteration)
                log_kv("resplit_iter_seed", split_seed + iteration)
                if ligand3d_store is not None:
                    try:
                        df_split = _read_smiles_csv(csv_path)
                        split_map = {"train": [], "valid": [], "test": []}
                        test_set = {str(x).strip() for x in (new_ids or []) if str(x).strip()}
                        for _, row in df_split.iterrows():
                            lig_id = str(row.get("ligand_id", "")).strip()
                            if not lig_id:
                                continue
                            idx = ligand3d_store.id_to_idx.get(lig_id)
                            if idx is None:
                                continue
                            if lig_id in test_set:
                                split_map["test"].append(idx)
                                continue
                            split = str(row.get("split", "train")).strip().lower()
                            if split not in {"train", "valid"}:
                                split = "train"
                            split_map[split].append(idx)
                        ligand3d_store.set_split_indices(split_map)
                        log_kv(
                            "store_splits_iter",
                            f"train={len(split_map['train'])} valid={len(split_map['valid'])} test={len(split_map['test'])}",
                        )
                    except Exception as e:
                        log_kv("store_splits_iter", f"failed:{e}")

            if retrain:
                clear_processed(dataset_root)
                if len(surrogate_ckpts) != 1:
                    raise RuntimeError("Retrain requires a single surrogate checkpoint.")
                Path(retrain_save_dir).mkdir(parents=True, exist_ok=True)
                save_path = str(Path(retrain_save_dir) / f"{surrogate_backbone}_surrogate_iter{iteration}.pt")
                if ligand3d_cache_dir_cfg:
                    ligand3d_cache_dir = Path(ligand3d_cache_dir_cfg)
                else:
                    ligand3d_cache_dir = Path(dataset_root) / "processed" / "ligand3d_cache"
                surrogate_ckpts = [
                    train_surrogate_from_scratch(
                        dataset_root,
                        save_path=save_path,
                        epochs=retrain_epochs,
                        batch_size=retrain_batch_size,
                        lr=retrain_lr,
                        weight_decay=retrain_weight_decay,
                        hidden_dim=retrain_hidden_dim,
                        num_layers=retrain_num_layers,
                        dropout=retrain_dropout,
                        use_edge_attr=retrain_use_edge_attr,
                        use_ligand_mask=retrain_use_ligand_mask,
                        standardize=retrain_standardize,
                        fp_dim=retrain_fp_dim,
                        fp_radius=retrain_fp_radius,
                        eval_samples=retrain_eval_samples,
                        scheduler=retrain_scheduler,
                        warmup_epochs=retrain_warmup_epochs,
                        min_lr=retrain_min_lr,
                        early_stop_patience=retrain_early_stop_patience,
                        early_stop_min_delta=retrain_early_stop_min_delta,
                        device=device,
                        dock_valid_max=dock_valid_max,
                        backbone=surrogate_backbone,
                        torchmd_cfg=torchmd_cfg,
                        vina_executable=vina_executable,
                        ligand3d_cache_dir=ligand3d_cache_dir,
                        ligand_vocab_override=ligand_vocab_override,
                        ligand3d_store=ligand3d_store,
                        test_ligand_ids=new_ids,
                        confgen_max_attempts=candidate_3d_max_attempts,
                        confgen_seed=0,
                        confgen_num_confs=candidate_3d_num_confs,
                        confgen_max_opt_iters=candidate_3d_max_opt_iters,
                        confgen_optimize=candidate_3d_optimize,
                        confgen_prefer_mmff=candidate_3d_prefer_mmff,
                    )
                ]
        log_kv("iter_time_sec", f"{time.perf_counter() - iter_start:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
