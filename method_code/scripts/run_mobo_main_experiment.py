#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3, CalcNumHBA, CalcNumHBD, CalcTPSA
from rdkit.ML.Cluster import Butina

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data.ligand_only_3d_dataset import GLOBAL_LIGAND3D_CACHE_DIR
from pretrain_tensornet_fp_encoder import resolve_init_encoder_pretrain_config, resolve_round_encoder_refresh_config, run_tensornet_encoder_pretrain
from run_retrospective_active_screening import (  # type: ignore
    _apply_fallback,
    _append_jsonl,
    _device_from_cfg,
    _prepare_loader,
    _section,
    _save_vocab,
)
from mobo.candidate_pool import build_candidate_pool
from mobo.constants import ATOM_EXTRA_DIM
from mobo.config_utils import load_config
from mobo.dataset_utils import clear_processed
from mobo.generators import load_candidate_generator
from mobo.init_selection import load_latent_library_csv
from mobo.io_utils import _is_valid_dock_score, _load_ligand_vocab_override, _pick_smiles_column, _scan_ligand_vocab
from mobo.metrics import bt_is_non_dominated, evaluate_oracle_accuracy_from_csv
from mobo.analytic_hvi_fast import nehvi_gaussian_analytic_3d, qphv_prob_gaussian_analytic_3d
from mobo.oracle import run_oracle_docking
from mobo.retrospective import build_objective_tensor, compute_hypervolume
from mobo.smiles_utils import calc_sa_score_mol, canonicalize_smiles_noh
from mobo.surrogate import load_surrogate, predict_gaussian_decomposed_stats, train_surrogate_from_scratch
from mobo_qpmhi import _ensure_oracle_assets_from_config, _fill_extra_metrics_in_csv, append_candidates_to_smiles_csv


ASSET_FILES = [
    "protein.pdb",
]



def _log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _release_cuda_cache(label: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    reserved_after = torch.cuda.memory_reserved()
    _log(
        "cuda cache release | "
        f"label={label} "
        f"allocated_before_mb={allocated_before / 1024 / 1024:.1f} "
        f"reserved_before_mb={reserved_before / 1024 / 1024:.1f} "
        f"reserved_after_mb={reserved_after / 1024 / 1024:.1f}"
    )




def _deep_update(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _apply_generate_config(cfg: dict, generate_cfg: dict) -> tuple[dict, dict]:
    gen_root = _require_dict(generate_cfg.get("mobo_generate"), "mobo_generate")
    if not gen_root:
        return cfg, {}

    applied: dict = {}
    out_cfg = dict(cfg)

    generator_cfg = gen_root.get("generator")
    if isinstance(generator_cfg, dict) and generator_cfg:
        general_cfg = _require_dict(out_cfg.get("general"), "general")
        general_cfg["generator"] = _deep_update(
            _require_dict(general_cfg.get("generator"), "general.generator"),
            generator_cfg,
        )
        out_cfg["general"] = general_cfg
        applied["general.generator"] = dict(generator_cfg)

    sample_cfg = gen_root.get("sample")
    if isinstance(sample_cfg, dict) and sample_cfg:
        candidate_cfg = _require_dict(out_cfg.get("candidate"), "candidate")
        key_map = {
            "sample_batch": "sample_batch",
            "generator_batch_size": "generator_batch_size",
            "temperature": "temperature",
            "top_k": "top_k",
            "pool_size": "pool_size",
            "candidate_3d": "candidate_3d",
        }
        applied_sample = {}
        for src_key, dst_key in key_map.items():
            if src_key in sample_cfg:
                if src_key == "candidate_3d" and isinstance(sample_cfg[src_key], dict):
                    candidate_cfg[dst_key] = _deep_update(
                        _require_dict(candidate_cfg.get(dst_key), f"candidate.{dst_key}"),
                        sample_cfg[src_key],
                    )
                else:
                    candidate_cfg[dst_key] = sample_cfg[src_key]
                applied_sample[dst_key] = sample_cfg[src_key]
        out_cfg["candidate"] = candidate_cfg
        if applied_sample:
            applied["candidate"] = applied_sample

    return out_cfg, applied

def _resolve_cli_or_config(cli_value, cfg_value, name: str):
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    raise RuntimeError(f"Missing required MOBO setting: {name}.")


def _require_dict(section, name: str) -> dict:
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RuntimeError(f"{name} must be a mapping.")
    return dict(section)


def _validate_mobo_config(cfg: dict) -> None:
    allowed_top = {"paths", "init_selection", "experiment"}
    allowed = {
        "paths": {
            "init_library_csv",
            "init_library_smiles_col",
            "init_encoder_ckpt",
            "init_training_dataset_root",
            "init_ligand_vocab_file",
            "output_dir",
        },
        "init_selection": {
            "init_size",
            "init_clusters",
            "init_min_per_cluster",
            "encode_batch_size",
            "encode_workers",
            "kmeans_epochs",
            "kmeans_n_init",
            "library_max_rows",
            "butina_cutoff",
            "fp_radius",
            "fp_bits",
            "init_train_valid_total",
            "holdout_test_total",
            "selection_seed",
        },
        "experiment": {
            "methods",
            "rounds",
            "score_eps",
            "seed",
        },
    }
    unknown_top = sorted(set(cfg.keys()) - allowed_top)
    if unknown_top:
        raise RuntimeError(f"Unsupported top-level keys in MOBO config: {unknown_top}")
    for section_name, allowed_keys in allowed.items():
        section = _require_dict(cfg.get(section_name), f"mobo.{section_name}")
        unknown = sorted(set(section.keys()) - allowed_keys)
        if unknown:
            raise RuntimeError(f"Unsupported keys in mobo.{section_name}: {unknown}")



def _resolve_oracle_backend(oracle_cfg: dict) -> str:
    backend = str(oracle_cfg.get("docking_backend", "vina")).strip().lower()
    if backend not in {"vina", "vina_cuda", "unidock_service"}:
        raise ValueError(f"Unsupported oracle backend: {backend}")
    return backend


def _resolve_oracle_executable(run_cfg: dict, oracle_cfg: dict) -> str | None:
    backend = _resolve_oracle_backend(oracle_cfg)
    if backend == "vina_cuda":
        return run_cfg.get("vina_cuda_executable", None)
    if backend == "unidock_service":
        return None
    return run_cfg.get("vina_executable", None)


def _build_oracle_call_kwargs(run_cfg: dict, oracle_cfg: dict, evaluate_reference: bool, dock_valid_max: float | None) -> dict:
    backend = _resolve_oracle_backend(oracle_cfg)
    oracle_3d_cfg = _section(oracle_cfg, "oracle_3d")
    return {
        "vina_executable": _resolve_oracle_executable(run_cfg, oracle_cfg),
        "docking_backend": backend,
        "vina_cuda_thread": int(oracle_cfg.get("vina_cuda_thread", 8192)),
        "vina_cuda_search_depth": int(oracle_cfg.get("vina_cuda_search_depth", 8)),
        "vina_cuda_rilc_bfgs": int(oracle_cfg.get("vina_cuda_rilc_bfgs", 1)),
        "overwrite": bool(oracle_cfg.get("oracle_overwrite", False)),
        "exhaustiveness": int(oracle_cfg.get("oracle_exhaustiveness", 32)),
        "pocket_radius": float(oracle_cfg.get("oracle_pocket_radius", 10.0)),
        "confgen_max_attempts": int(oracle_3d_cfg.get("max_attempts", 3)),
        "confgen_seed": 0,
        "confgen_num_confs": int(oracle_3d_cfg.get("num_confs", 8)),
        "confgen_max_opt_iters": int(oracle_3d_cfg.get("max_opt_iters", 200)),
        "confgen_optimize": bool(oracle_3d_cfg.get("optimize", True)),
        "confgen_prefer_mmff": bool(oracle_3d_cfg.get("prefer_mmff", False)),
        "meeko_allow_bad_res": bool(oracle_cfg.get("meeko_allow_bad_res", False)),
        "meeko_default_altloc": oracle_cfg.get("meeko_default_altloc", None),
        "evaluate_reference": bool(evaluate_reference),
        "cache_root": oracle_cfg.get("cache_root", None),
        "ref_resname": oracle_cfg.get("ref_resname", None),
        "unidock_service_url": oracle_cfg.get("unidock_service_url", None),
        "unidock_service_wsl_distro": oracle_cfg.get("unidock_service_wsl_distro", None),
        "unidock_scoring": oracle_cfg.get("unidock_scoring", "vina"),
        "unidock_search_mode": oracle_cfg.get("unidock_search_mode", "balance"),
        "unidock_num_modes": int(oracle_cfg.get("unidock_num_modes", 1)),
        "unidock_timeout_sec": int(oracle_cfg.get("unidock_timeout_sec", 3600)),
        "dock_valid_max": dock_valid_max,
    }
def _copy_dataset_assets(source_root: Path, dest_root: Path) -> None:
    """Copy oracle-related raw/prepared assets into a method dataset root.

    Historical result files such as smiles.csv, docking/, failed.json and old
    docking outputs are run outputs rather than method dataset assets. Each
    experiment writes a fresh smiles.csv under its own result directory.

    We DO preserve existing receptor/reference artifacts so the oracle can
    reuse prepared pocket/receptor files instead of regenerating them.
    """
    dest_root.mkdir(parents=True, exist_ok=True)

    keep_files = [
        "protein.pdb",
        "protein.pdbqt",
        "pocket.pdb",
        "pocket.pdbqt",
        "reference_ligand.sdf",
        "reference_ligand.pdb",
        "meeko_receptor_config.json",
    ]

    copied = []
    for name in keep_files:
        src = source_root / name
        if src.exists() and src.is_file():
            shutil.copy2(src, dest_root / name)
            copied.append(name)

    # If the source dataset ships with bin/, copy it too so relative executable paths still resolve.
    src_bin = source_root / "bin"
    if src_bin.exists() and src_bin.is_dir():
        dst_bin = dest_root / "bin"
        dst_bin.mkdir(parents=True, exist_ok=True)
        for item in src_bin.iterdir():
            if item.is_file():
                shutil.copy2(item, dst_bin / item.name)

    _log(
        f"copy_dataset_assets | source={source_root} dest={dest_root} "
        f"copied={copied if copied else '[]'}"
    )


def _normalize_ratio_values(values: list[float] | tuple[float, ...], *, expected_len: int) -> tuple[float, ...]:
    if len(values) != int(expected_len):
        raise RuntimeError(f"Expected {expected_len} ratio values, got {values!r}")
    arr = np.asarray(values, dtype=np.float64)
    if (~np.isfinite(arr)).any():
        raise RuntimeError(f"Ratio values must be finite: {values!r}")
    if (arr < 0.0).any():
        raise RuntimeError(f"Ratio values must be non-negative: {values!r}")
    total = float(arr.sum())
    if total <= 0.0:
        raise RuntimeError(f"Ratio values must sum to > 0: {values!r}")
    arr = arr / total
    return tuple(float(x) for x in arr.tolist())


def _normalize_ratio_config(raw_value, *, default: tuple[float, ...], expected_len: int) -> tuple[float, ...]:
    if raw_value is None:
        return _normalize_ratio_values(list(default), expected_len=expected_len)
    if isinstance(raw_value, (int, float)):
        if expected_len != 2:
            raise RuntimeError(f"Scalar ratio is only supported for two-way splits, got expected_len={expected_len}")
        val = float(raw_value)
        return _normalize_ratio_values([val, 1.0 - val], expected_len=2)
    if not isinstance(raw_value, (list, tuple)):
        raise RuntimeError(f"Unsupported ratio config value: {raw_value!r}")
    values = [float(x) for x in raw_value]
    if expected_len == 2 and len(values) == 3:
        values = values[:2]
    return _normalize_ratio_values(values, expected_len=expected_len)


def _smiles_signature(smiles_list: list[str]) -> str:
    h = hashlib.sha256()
    for smi in smiles_list:
        h.update(str(smi).strip().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _load_cache_metadata(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Cache metadata must be a JSON object: {meta_path}")
    return data


def _meta_matches(actual: dict | None, expected: dict) -> bool:
    if actual is None:
        return False
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _write_cached_dataframe(df: pd.DataFrame, csv_path: Path, meta_path: Path, meta: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _canonicalize_library_df(df: pd.DataFrame, smiles_col: str, *, require_unique: bool = True) -> pd.DataFrame:
    if smiles_col not in df.columns:
        raise RuntimeError(f"Missing smiles column '{smiles_col}' in downstream library.")
    work = df.copy().reset_index(drop=True)
    work[smiles_col] = work[smiles_col].astype(str).str.strip()
    work = work.loc[work[smiles_col] != ""].copy().reset_index(drop=True)
    if work.empty:
        raise RuntimeError("Downstream library became empty after dropping blank SMILES.")
    canons = []
    for smi in work[smiles_col].tolist():
        canon = canonicalize_smiles_noh(str(smi))
        if not canon:
            raise RuntimeError(f"Failed to canonicalize downstream SMILES: {smi!r}")
        canons.append(canon)
    work["smiles_canonical"] = canons
    work["smiles"] = work["smiles_canonical"].astype(str)
    if require_unique:
        work = work.drop_duplicates(subset=["smiles_canonical"], keep="first").reset_index(drop=True)
    if work.empty:
        raise RuntimeError("Downstream library became empty after canonical deduplication.")
    return work


def _ensure_property_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    for col in ["qed", "sa_score", "logp", "tpsa", "hbd", "hba", "fsp3"]:
        if col not in work.columns:
            work[col] = np.nan
    for idx, smi in enumerate(work["smiles_canonical"].astype(str).tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse canonical SMILES: {smi}")
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not np.isfinite(float(pd.to_numeric(pd.Series([work.at[idx, "qed"]]), errors="coerce").iloc[0])):
            work.at[idx, "qed"] = float(QED.qed(mol))
        if not np.isfinite(float(pd.to_numeric(pd.Series([work.at[idx, "sa_score"]]), errors="coerce").iloc[0])):
            work.at[idx, "sa_score"] = float(calc_sa_score_mol(mol))
        if not np.isfinite(float(pd.to_numeric(pd.Series([work.at[idx, "logp"]]), errors="coerce").iloc[0])):
            work.at[idx, "logp"] = float(Crippen.MolLogP(mol))
        if not np.isfinite(float(pd.to_numeric(pd.Series([work.at[idx, "tpsa"]]), errors="coerce").iloc[0])):
            work.at[idx, "tpsa"] = float(CalcTPSA(mol))
        if not np.isfinite(float(pd.to_numeric(pd.Series([work.at[idx, "hbd"]]), errors="coerce").iloc[0])):
            work.at[idx, "hbd"] = float(CalcNumHBD(mol))
        if not np.isfinite(float(pd.to_numeric(pd.Series([work.at[idx, "hba"]]), errors="coerce").iloc[0])):
            work.at[idx, "hba"] = float(CalcNumHBA(mol))
        if not np.isfinite(float(pd.to_numeric(pd.Series([work.at[idx, "fsp3"]]), errors="coerce").iloc[0])):
            work.at[idx, "fsp3"] = float(CalcFractionCSP3(mol))
    return work


def _compute_bemis_murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse SMILES for scaffold generation: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=bool(include_chirality))
    scaffold = str(scaffold).strip()
    return scaffold if scaffold else "__NO_SCAFFOLD__"


def _build_scaffold_groups(df: pd.DataFrame, smiles_col: str, include_chirality: bool = False) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    work = df.copy().reset_index(drop=True)
    if "murcko_scaffold" in work.columns:
        scaffolds = work["murcko_scaffold"].astype(str).tolist()
        if any((not str(scaffold).strip()) for scaffold in scaffolds):
            raise RuntimeError("Existing murcko_scaffold column contains blank values.")
    else:
        scaffolds = [
            _compute_bemis_murcko_scaffold(smi, include_chirality=include_chirality)
            for smi in work[smiles_col].astype(str).tolist()
        ]
    work["murcko_scaffold"] = scaffolds
    groups: dict[str, list[int]] = {}
    for idx, scaffold in enumerate(scaffolds):
        groups.setdefault(str(scaffold), []).append(int(idx))
    return work, groups


def _scaffold_split_downstream_pool(
    df: pd.DataFrame,
    smiles_col: str,
    pool_ratio: tuple[float, float],
    seed: int,
    include_chirality: bool = False,
) -> pd.DataFrame:
    work, groups = _build_scaffold_groups(df, smiles_col=smiles_col, include_chirality=include_chirality)
    total = int(work.shape[0])
    if total < 2:
        raise RuntimeError("Downstream scaffold split requires at least two molecules.")
    target_opt = int(round(float(pool_ratio[0]) * total))
    target_opt = min(max(1, target_opt), total - 1)
    target_eval = total - target_opt
    tie_rng = np.random.default_rng(int(seed))
    group_rows = []
    for scaffold, indices in groups.items():
        group_rows.append({
            "scaffold": str(scaffold),
            "indices": list(indices),
            "size": int(len(indices)),
            "tie": int(tie_rng.integers(0, 2**31 - 1)),
        })
    group_rows = sorted(group_rows, key=lambda item: (-item["size"], item["tie"], item["scaffold"]))
    opt_indices: list[int] = []
    eval_indices: list[int] = []
    for item in group_rows:
        indices = list(item["indices"])
        size = int(item["size"])
        cost_opt = abs((len(opt_indices) + size) - target_opt) + abs(len(eval_indices) - target_eval)
        cost_eval = abs(len(opt_indices) - target_opt) + abs((len(eval_indices) + size) - target_eval)
        choose_opt = cost_opt < cost_eval
        if cost_opt == cost_eval:
            opt_frac = len(opt_indices) / max(target_opt, 1)
            eval_frac = len(eval_indices) / max(target_eval, 1)
            choose_opt = opt_frac <= eval_frac
        if choose_opt:
            opt_indices.extend(indices)
        else:
            eval_indices.extend(indices)
    if not opt_indices or not eval_indices:
        raise RuntimeError("Scaffold split failed to create non-empty pool_opt and pool_eval.")
    pool_labels = np.full(total, "pool_eval", dtype=object)
    pool_labels[np.asarray(opt_indices, dtype=np.int64)] = "pool_opt"
    work["downstream_pool"] = pool_labels.tolist()
    return work


def _compute_morgan_fingerprints(df: pd.DataFrame, smiles_col: str, radius: int, n_bits: int) -> list:
    fps = []
    for smi in df[smiles_col].astype(str).tolist():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse SMILES for Morgan fingerprint: {smi!r}")
        fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits)))
    return fps


def _butina_cluster_indices(fps: list, cutoff: float) -> list[tuple[int, ...]]:
    if not fps:
        raise RuntimeError("Butina clustering requires a non-empty fingerprint list.")
    dists: list[float] = []
    for idx in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[:idx])
        dists.extend([1.0 - float(sim) for sim in sims])
    clusters = Butina.ClusterData(dists, len(fps), float(cutoff), isDistData=True)
    return [tuple(int(i) for i in cluster) for cluster in clusters]


def _select_representatives_from_clusters(
    df: pd.DataFrame,
    clusters: list[tuple[int, ...]],
    target_count: int,
    seed: int,
    smiles_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = int(df.shape[0])
    if int(target_count) <= 0 or int(target_count) > total:
        raise RuntimeError(f"Invalid Butina target count {target_count} for pool size {total}.")
    rng = np.random.default_rng(int(seed))
    cluster_items = []
    smiles = df[smiles_col].astype(str).tolist()
    for cluster_id, cluster in enumerate(clusters):
        members = sorted([int(i) for i in cluster], key=lambda idx: smiles[idx])
        representative = int(cluster[0]) if int(cluster[0]) in members else int(members[0])
        ordered_members = [representative] + [idx for idx in members if idx != representative]
        cluster_items.append({
            "cluster_id": int(cluster_id),
            "members": ordered_members,
            "size": int(len(ordered_members)),
            "tie": int(rng.integers(0, 2**31 - 1)),
            "representative": int(representative),
        })
    cluster_items = sorted(cluster_items, key=lambda item: (-item["size"], item["tie"], smiles[item["representative"]]))

    selected_indices: list[int] = []
    used: set[int] = set()
    for item in cluster_items:
        rep = int(item["representative"])
        if rep in used:
            continue
        selected_indices.append(rep)
        used.add(rep)
        if len(selected_indices) >= int(target_count):
            break
    depth = 1
    while len(selected_indices) < int(target_count):
        progress = False
        for item in cluster_items:
            members = item["members"]
            if depth >= len(members):
                continue
            idx = int(members[depth])
            if idx in used:
                continue
            selected_indices.append(idx)
            used.add(idx)
            progress = True
            if len(selected_indices) >= int(target_count):
                break
        if not progress:
            break
        depth += 1
    if len(selected_indices) != int(target_count):
        raise RuntimeError(f"Butina selection failed: selected={len(selected_indices)} target={target_count}")

    assignment_rows = []
    selection_rank = {idx: rank for rank, idx in enumerate(selected_indices)}
    for item in cluster_items:
        for member_rank, idx in enumerate(item["members"]):
            assignment_rows.append({
                "smiles_canonical": str(smiles[idx]),
                "butina_cluster_id": int(item["cluster_id"]),
                "cluster_size": int(item["size"]),
                "cluster_member_rank": int(member_rank),
                "cluster_representative": bool(idx == int(item["representative"])),
                "selected": bool(idx in selection_rank),
                "selection_rank": int(selection_rank[idx]) if idx in selection_rank else -1,
            })
    assignments_df = pd.DataFrame(assignment_rows).sort_values(["butina_cluster_id", "cluster_member_rank"]).reset_index(drop=True)
    selected_df = df.iloc[selected_indices].copy().reset_index(drop=True)
    selected_df["selection_rank"] = list(range(selected_df.shape[0]))
    return assignments_df, selected_df


def _assign_pretrain_split(
    df: pd.DataFrame,
    *,
    split_ratio: tuple[float, float, float],
    seed: int,
    allow_existing_split: bool,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if allow_existing_split and "split" in out.columns:
        split_vals = out["split"].astype(str).str.lower()
        if split_vals.isin({"train", "valid", "test"}).all():
            out["split"] = split_vals
            return out
    rng = np.random.default_rng(int(seed))
    order = np.arange(out.shape[0], dtype=np.int64)
    rng.shuffle(order)
    n = int(out.shape[0])
    p_train, p_valid, p_test = _normalize_ratio_values(list(split_ratio), expected_len=3)
    n_train = int(round(p_train * n))
    n_valid = int(round(p_valid * n))
    n_train = min(max(1, n_train), max(1, n - 2)) if n >= 3 else max(1, min(n, n_train))
    n_valid = min(max(1 if p_valid > 0 else 0, n_valid), max(0, n - n_train - 1)) if n >= 3 else max(0, min(n - n_train, n_valid))
    n_test = n - n_train - n_valid
    if n >= 3 and n_test <= 0:
        n_test = 1
        if n_train >= n_valid and n_train > 1:
            n_train -= 1
        elif n_valid > 1:
            n_valid -= 1
    split = np.empty(n, dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train:n_train + n_valid]] = "valid"
    split[order[n_train + n_valid:]] = "test"
    out["split"] = split.tolist()
    return out


def _prepare_encoder_pretrain_dataset(
    dataset_root: Path,
    init_library_df: pd.DataFrame,
    *,
    split_ratio: tuple[float, float, float],
    seed: int,
    cache_dir: Path,
    source_csv_path: Path | None,
    allow_existing_split: bool,
) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pretrain_csv = cache_dir / "pretrain_random_split.csv"
    pretrain_meta = cache_dir / "pretrain_random_split.meta.json"
    pretrain_work = init_library_df.copy().reset_index(drop=True)
    if "smiles_canonical" not in pretrain_work.columns:
        smiles_col = _pick_smiles_column(pretrain_work.columns)
        if smiles_col is None:
            raise RuntimeError("Failed to resolve smiles column for encoder pretraining split.")
        pretrain_work = _canonicalize_library_df(pretrain_work, smiles_col=smiles_col, require_unique=True)
    else:
        pretrain_work["smiles_canonical"] = pretrain_work["smiles_canonical"].astype(str).str.strip()
        pretrain_work = pretrain_work.loc[pretrain_work["smiles_canonical"] != ""].copy().reset_index(drop=True)
        if "smiles" not in pretrain_work.columns:
            pretrain_work["smiles"] = pretrain_work["smiles_canonical"].astype(str)
    signature = _smiles_signature(pretrain_work["smiles_canonical"].astype(str).tolist())
    expected_meta = {
        "source_csv_path": str(source_csv_path.resolve()) if source_csv_path is not None else None,
        "rows": int(pretrain_work.shape[0]),
        "smiles_signature": signature,
        "seed": int(seed),
        "split_ratio": [float(x) for x in split_ratio],
        "allow_existing_split": bool(allow_existing_split),
    }
    cached_meta = _load_cache_metadata(pretrain_meta)
    if pretrain_csv.exists() and _meta_matches(cached_meta, expected_meta):
        pretrain_df = pd.read_csv(pretrain_csv)
        _log(f"pretrain random split loaded_from_cache | cache={pretrain_csv} rows={pretrain_df.shape[0]}")
    else:
        pretrain_df = _assign_pretrain_split(
            pretrain_work,
            split_ratio=split_ratio,
            seed=int(seed),
            allow_existing_split=bool(allow_existing_split),
        )
        _write_cached_dataframe(pretrain_df, pretrain_csv, pretrain_meta, expected_meta)
        _log(f"pretrain random split recomputed | cache={pretrain_csv} rows={pretrain_df.shape[0]}")
    cols = []
    for col in ["ligand_id", "smiles_canonical", "smiles", "split"]:
        if col in pretrain_df.columns and col not in cols:
            cols.append(col)
    if "smiles" not in cols:
        cols.append("smiles")
    if "smiles_canonical" not in cols:
        cols.append("smiles_canonical")
    if "split" not in cols:
        cols.append("split")
    out_df = pretrain_df.copy()
    if "smiles" not in out_df.columns:
        out_df["smiles"] = out_df["smiles_canonical"].astype(str)
    if "smiles_canonical" not in out_df.columns:
        out_df["smiles_canonical"] = out_df["smiles"].astype(str)
    if "ligand_id" not in out_df.columns:
        out_df["ligand_id"] = [f"PRE_{i:07d}" for i in range(1, out_df.shape[0] + 1)]
    out_df[cols].to_csv(dataset_root / "smiles.csv", index=False)
    return dataset_root


def _prepare_round_encoder_pretrain_dataset(
    dataset_root: Path,
    init_library_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    split_ratio: tuple[float, float, float],
    seed: int,
) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    init_work = init_library_df.copy()
    if "smiles_canonical" not in init_work.columns:
        raise RuntimeError("init_library_df missing smiles_canonical for round encoder pretraining.")
    if "smiles" not in init_work.columns:
        init_work["smiles"] = init_work["smiles_canonical"].astype(str)
    init_work = init_work[[c for c in ["ligand_id", "smiles_canonical", "smiles"] if c in init_work.columns]].copy()

    cand_work = candidate_df.copy()
    if "smiles_canonical" not in cand_work.columns:
        raise RuntimeError("candidate_df missing smiles_canonical for round encoder pretraining.")
    if "smiles" not in cand_work.columns:
        cand_work["smiles"] = cand_work["smiles_canonical"].astype(str)
    cand_work = cand_work[[c for c in ["smiles_canonical", "smiles"] if c in cand_work.columns]].copy()
    cand_work["ligand_id"] = [f"ROUND_{i:07d}" for i in range(1, cand_work.shape[0] + 1)]

    merged = pd.concat([init_work, cand_work], ignore_index=True)
    merged["smiles_canonical"] = merged["smiles_canonical"].astype(str).str.strip()
    merged["smiles"] = merged["smiles"].astype(str).str.strip()
    merged = merged.loc[merged["smiles_canonical"] != ""].copy().reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["smiles_canonical"], keep="first").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("Round encoder pretraining dataset is empty after deduplication.")

    pretrain_df = _assign_pretrain_split(
        merged,
        split_ratio=split_ratio,
        seed=int(seed),
        allow_existing_split=False,
    )
    pretrain_df[["ligand_id", "smiles_canonical", "smiles", "split"]].to_csv(dataset_root / "smiles.csv", index=False)
    return dataset_root


def _accumulate_candidate_pool_history(history_df: pd.DataFrame | None, candidate_df: pd.DataFrame) -> pd.DataFrame:
    work = candidate_df.copy()
    if "smiles_canonical" not in work.columns:
        raise RuntimeError("candidate_df missing smiles_canonical for candidate history accumulation.")
    if "smiles" not in work.columns:
        work["smiles"] = work["smiles_canonical"].astype(str)
    work["smiles_canonical"] = work["smiles_canonical"].astype(str).str.strip()
    work["smiles"] = work["smiles"].astype(str).str.strip()
    work = work.loc[work["smiles_canonical"] != ""].copy().reset_index(drop=True)
    work = work.drop_duplicates(subset=["smiles_canonical"], keep="first").reset_index(drop=True)
    if work.empty:
        raise RuntimeError("Candidate history accumulation produced an empty frame after deduplication.")
    if history_df is None or history_df.empty:
        return work.reset_index(drop=True)
    merged = pd.concat([history_df, work], ignore_index=True)
    merged["smiles_canonical"] = merged["smiles_canonical"].astype(str).str.strip()
    if "smiles" in merged.columns:
        merged["smiles"] = merged["smiles"].astype(str).str.strip()
    merged = merged.loc[merged["smiles_canonical"] != ""].copy().reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["smiles_canonical"], keep="first").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("Candidate history accumulation produced an empty merged frame.")
    return merged


def _load_vocab_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise RuntimeError(f"Expected a list in ligand vocab JSON: {path}")
        vocab = [str(x).strip() for x in data if str(x).strip()]
    else:
        vocab = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not vocab:
        raise RuntimeError(f"Ligand vocab file is empty: {path}")
    return sorted(set(vocab))


def _resolve_init_pretrain_ligand_vocab(
    *,
    init_training_dataset_root: str | Path | None,
    init_ligand_vocab_file: str | Path | None,
    init_library_df: pd.DataFrame,
) -> list[str]:
    if init_ligand_vocab_file:
        return _load_vocab_file(Path(init_ligand_vocab_file))
    if init_training_dataset_root:
        root = Path(init_training_dataset_root)
        if not root.exists():
            raise FileNotFoundError(root)
        vocab = _load_ligand_vocab_override(root)
        if vocab:
            return vocab
        smiles_path = root / "smiles.csv"
        if smiles_path.exists():
            df = pd.read_csv(smiles_path)
            smiles_col = _pick_smiles_column(df.columns)
            if smiles_col is None:
                raise RuntimeError(f"No smiles column found in {smiles_path}.")
            return _scan_ligand_vocab(df[smiles_col].astype(str).tolist())
    smiles_col = _pick_smiles_column(init_library_df.columns)
    if smiles_col is None:
        raise RuntimeError("Failed to resolve a ligand vocab for init encoder pretraining.")
    return _scan_ligand_vocab(init_library_df[smiles_col].astype(str).tolist())


def _checkpoint_node_feat_dim(ckpt_path: Path) -> int | None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    torchmd_cfg = dict(cfg.get("torchmd") or {})
    val = torchmd_cfg.get("node_feat_dim", cfg.get("node_feat_dim"))
    return None if val is None else int(val)


def _resolve_or_train_init_encoder_ckpt(
    *,
    args: argparse.Namespace,
    cfg_path: str,
    out_root: Path,
    init_library_df: pd.DataFrame,
    shared_ligand3d_cache: Path,
    candidate_3d_cfg: dict,
    init_training_dataset_root: str | Path | None,
    init_ligand_vocab_file: str | Path | None,
    source_csv_path: Path | None,
    pretrain_split_ratio: tuple[float, float, float],
    pretrain_split_seed: int,
    pretrain_use_existing_split: bool,
) -> Path:
    if args.init_encoder_ckpt:
        ckpt_path = Path(str(args.init_encoder_ckpt)).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        return ckpt_path
    pretrain_root = out_root / "init_encoder_pretrain_dataset"
    _prepare_encoder_pretrain_dataset(
        pretrain_root,
        init_library_df,
        split_ratio=pretrain_split_ratio,
        seed=int(pretrain_split_seed),
        cache_dir=out_root / "cache",
        source_csv_path=source_csv_path,
        allow_existing_split=bool(pretrain_use_existing_split),
    )
    pretrain_ligand_vocab = _resolve_init_pretrain_ligand_vocab(
        init_training_dataset_root=init_training_dataset_root,
        init_ligand_vocab_file=init_ligand_vocab_file,
        init_library_df=init_library_df,
    )
    expected_node_feat_dim = len(pretrain_ligand_vocab) + int(ATOM_EXTRA_DIM)
    ckpt_path = out_root / "init_encoder_pretrain" / "tensornet_fp_prop_encoder.pt"
    if ckpt_path.exists():
        ckpt_node_feat_dim = _checkpoint_node_feat_dim(ckpt_path)
        if ckpt_node_feat_dim == expected_node_feat_dim:
            _log(f"init_encoder auto_pretrain reuse | ckpt={ckpt_path} node_feat_dim={ckpt_node_feat_dim}")
            return ckpt_path
        _log(
            f"init_encoder auto_pretrain rebuild | ckpt={ckpt_path} "
            f"node_feat_dim={ckpt_node_feat_dim} expected={expected_node_feat_dim}"
        )
    _log(f"init_encoder auto_pretrain start | dataset={pretrain_root} save_path={ckpt_path}")
    init_pretrain_cfg = resolve_init_encoder_pretrain_config(load_config(cfg_path))
    run_tensornet_encoder_pretrain(
        config_path=cfg_path,
        dataset_root=pretrain_root,
        save_path=ckpt_path,
        epochs=int(init_pretrain_cfg["epochs"]),
        batch_size=int(init_pretrain_cfg["batch_size"]),
        lr=float(init_pretrain_cfg["lr"]),
        weight_decay=float(init_pretrain_cfg["weight_decay"]),
        fp_bits=int(init_pretrain_cfg["fp_bits"]),
        fp_radius=int(init_pretrain_cfg["fp_radius"]),
        device_arg=str(init_pretrain_cfg["device"]),
        ligand3d_cache_dir=str(shared_ligand3d_cache),
        ligand_vocab_override=pretrain_ligand_vocab,
        ligand3d_num_workers=int(candidate_3d_cfg.get("workers", 8)),
        ligand3d_mp_chunksize=int(candidate_3d_cfg.get("chunksize", 16)),
        confgen_seed=int(args.seed),
        fp_weight=float(init_pretrain_cfg["fp_weight"]),
        prop_weight=float(init_pretrain_cfg["prop_weight"]),
        scheduler_name=str(init_pretrain_cfg["scheduler"]),
        warmup_epochs=int(init_pretrain_cfg["warmup_epochs"]),
        min_lr=float(init_pretrain_cfg["min_lr"]),
        early_stop_patience=int(init_pretrain_cfg["early_stop_patience"]),
        early_stop_min_delta=float(init_pretrain_cfg["early_stop_min_delta"]),
    )
    return ckpt_path


def _build_exact_random_split(smiles_list: list[str], *, valid_ratio: float, seed: int) -> list[str]:
    n = len(smiles_list)
    if n <= 0:
        raise RuntimeError("Random split assignment requires a non-empty smiles list.")
    valid_ratio = float(valid_ratio)
    if not (0.0 <= valid_ratio < 1.0):
        raise RuntimeError(f"predictor_valid_ratio must be in [0, 1), got {valid_ratio}")
    n_valid = int(round(valid_ratio * n))
    if valid_ratio > 0.0 and n > 1:
        n_valid = min(max(1, n_valid), n - 1)
    else:
        n_valid = 0
    order = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    splits = np.full(n, "train", dtype=object)
    if n_valid > 0:
        splits[order[:n_valid]] = "valid"
    return splits.tolist()


def _write_predictor_init_split_cache(
    *,
    init_df: pd.DataFrame,
    cache_csv: Path,
    cache_meta: Path,
    source_csv_path: Path | None,
    valid_ratio: float,
    seed: int,
) -> pd.DataFrame:
    work = init_df.copy().reset_index(drop=True)
    signature = _smiles_signature(work["smiles_canonical"].astype(str).tolist())
    expected_meta = {
        "source_csv_path": str(source_csv_path.resolve()) if source_csv_path is not None else None,
        "rows": int(work.shape[0]),
        "smiles_signature": signature,
        "valid_ratio": float(valid_ratio),
        "seed": int(seed),
    }
    cached_meta = _load_cache_metadata(cache_meta)
    if cache_csv.exists() and _meta_matches(cached_meta, expected_meta):
        split_df = pd.read_csv(cache_csv)
        _log(f"predictor random split loaded_from_cache | cache={cache_csv} rows={split_df.shape[0]}")
        return split_df
    split_df = work[["smiles_canonical"]].copy()
    split_df["predictor_split"] = _build_exact_random_split(
        work["smiles_canonical"].astype(str).tolist(),
        valid_ratio=float(valid_ratio),
        seed=int(seed),
    )
    _write_cached_dataframe(split_df, cache_csv, cache_meta, expected_meta)
    counts = split_df["predictor_split"].value_counts().to_dict()
    _log(
        "predictor random split recomputed | "
        f"cache={cache_csv} train={int(counts.get('train', 0))} valid={int(counts.get('valid', 0))}"
    )
    return split_df


def _merge_protocol_columns(df: pd.DataFrame, protocol_df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    protocol_cols = [
        "smiles_canonical",
        "split",
        "split_role",
        "init_group",
        "selection_rank",
        "downstream_pool",
        "murcko_scaffold",
        "predictor_split",
    ]
    keep_cols = [col for col in protocol_cols if col in protocol_df.columns]
    merged = work.merge(protocol_df[keep_cols], on="smiles_canonical", how="left", suffixes=("", "_protocol"))
    for col in keep_cols:
        if col == "smiles_canonical":
            continue
        protocol_col = f"{col}_protocol"
        if protocol_col in merged.columns:
            merged[col] = merged[protocol_col]
            merged = merged.drop(columns=[protocol_col])
    if "predictor_split" in merged.columns:
        predictor_split = merged["predictor_split"].astype(str).str.lower()
        holdout_mask = merged.get("split_role", pd.Series([""] * merged.shape[0])).astype(str).str.lower().eq("holdout_test")
        train_valid_mask = predictor_split.isin(["train", "valid"])
        merged.loc[train_valid_mask & ~holdout_mask, "split"] = predictor_split.loc[train_valid_mask & ~holdout_mask]
        merged.loc[holdout_mask, "split"] = "test"
    return merged


def _assign_new_rows_random_predictor_split(
    dataset_root: Path,
    new_ids: list[str],
    *,
    valid_ratio: float,
    seed: int,
) -> tuple[int, int]:
    csv_path = dataset_root / "smiles.csv"
    df = pd.read_csv(csv_path)
    if "ligand_id" not in df.columns:
        raise RuntimeError(f"smiles.csv missing ligand_id: {csv_path}")
    mask = df["ligand_id"].astype(str).isin([str(x) for x in new_ids])
    if int(mask.sum()) != len(new_ids):
        raise RuntimeError(f"Failed to locate all newly appended ligand ids in {csv_path}: expected={len(new_ids)} found={int(mask.sum())}")
    splits = _build_exact_random_split(
        df.loc[mask, "smiles_canonical"].astype(str).tolist(),
        valid_ratio=float(valid_ratio),
        seed=int(seed),
    )
    df.loc[mask, "split"] = splits
    if "split_role" in df.columns:
        df.loc[mask, "split_role"] = "active_learning"
    if "init_group" in df.columns:
        df.loc[mask, "init_group"] = "active_learning"
    df.to_csv(csv_path, index=False)
    train_n = sum(1 for item in splits if item == "train")
    valid_n = sum(1 for item in splits if item == "valid")
    return int(train_n), int(valid_n)


def _prepare_init_dataset(
    source_root: Path | None,
    dest_root: Path,
    schema_df: pd.DataFrame,
    init_df: pd.DataFrame,
    oracle_cfg: dict,
    objective_cfg: dict,
    run_cfg: dict,
    global_vocab: list[str],
) -> dict:
    dest_root.mkdir(parents=True, exist_ok=True)
    if source_root is not None:
        _copy_dataset_assets(source_root, dest_root)
    _ensure_oracle_assets_from_config(dest_root, oracle_cfg)
    if not (dest_root / "protein.pdb").exists():
        raise FileNotFoundError(
            f"protein.pdb missing at {dest_root / 'protein.pdb'}; set oracle.protein_pdb or provide a resolvable oracle asset root."
        )
    protocol_df = init_df.copy().reset_index(drop=True)
    for required_col in ["smiles_canonical", "split_role", "split"]:
        if required_col not in protocol_df.columns:
            raise RuntimeError(f"Initialization dataframe missing required column: {required_col}")
    out_df = schema_df.head(0).copy()
    for col in ["smiles", "smiles_canonical", "ligand_id", "dock_score", "dock_pose", "dock_cache_key", "dock_source", "added_iter", "molecule_origin", "is_generated", "split", "split_role", "init_group", "downstream_pool", "murcko_scaffold", "predictor_split", "selection_rank"]:
        if col not in out_df.columns:
            out_df[col] = np.nan
    out_df.to_csv(dest_root / "smiles.csv", index=False)
    _save_vocab(dest_root, global_vocab)

    init_smiles = protocol_df["smiles_canonical"].astype(str).tolist()
    added, new_ids, kept_indices = append_candidates_to_smiles_csv(
        str(dest_root / "smiles.csv"),
        init_smiles,
        id_prefix="INIT",
        sa_clamp_min=float(objective_cfg.get("sa_clamp_min", -10.0)),
        sa_clamp_max=float(objective_cfg.get("sa_clamp_max", 20.0)),
        split_ratio=None,
        added_iter=0,
        molecule_origin="init_library",
    )
    if added != len(init_smiles) or kept_indices != list(range(len(init_smiles))):
        raise RuntimeError(
            f"Initial-set append mismatch: added={added}, expected={len(init_smiles)}, kept={kept_indices}"
        )
    oracle_kwargs = _build_oracle_call_kwargs(
        run_cfg=run_cfg,
        oracle_cfg=oracle_cfg,
        evaluate_reference=True,
        dock_valid_max=objective_cfg.get("dock_valid_max", 0.0),
    )

    _log(
        "init oracle start | "
        f"backend={oracle_kwargs['docking_backend']} "
        f"new_ids={len(new_ids)} "
        f"vina_executable={oracle_kwargs['vina_executable']}"
    )
    _release_cuda_cache(f"before_init_oracle:{dest_root.name}")

    dock_stats = run_oracle_docking(
        str(dest_root),
        new_ids,
        **oracle_kwargs,
    )

    _log(
        "init oracle done | "
        f"backend={oracle_kwargs['docking_backend']} "
        f"attempted={dock_stats.get('attempted', 0)} "
        f"docked={dock_stats.get('docked', 0)} "
        f"cache_hit_conformer={dock_stats.get('cache_hit_conformer', 0)} "
        f"cache_hit_docking={dock_stats.get('cache_hit_docking', 0)} "
        f"failed={dock_stats.get('failed', 0)}"
    )
    _fill_extra_metrics_in_csv(dest_root, objective_cfg, prior_state=None, force=False)
    scored_df = pd.read_csv(dest_root / "smiles.csv")
    scored_df = _merge_protocol_columns(scored_df, protocol_df)
    scored_df.to_csv(dest_root / "smiles.csv", index=False)

    opt_df = _load_observed_df(dest_root, dock_valid_max=objective_cfg.get("dock_valid_max", 0.0))
    holdout_df = _load_holdout_df(dest_root, dock_valid_max=objective_cfg.get("dock_valid_max", 0.0))
    init_dock_vals = pd.to_numeric(opt_df.get("dock_score"), errors="coerce").to_numpy(dtype=float)
    holdout_dock_vals = pd.to_numeric(holdout_df.get("dock_score"), errors="coerce").to_numpy(dtype=float)
    init_best_dock = float(np.min(init_dock_vals)) if init_dock_vals.size > 0 else float("nan")
    init_mean_dock = float(np.mean(init_dock_vals)) if init_dock_vals.size > 0 else float("nan")
    return {
        "dock_stats": dock_stats,
        "new_ids": new_ids,
        "init_n": int(protocol_df.loc[protocol_df["split_role"].astype(str) == "init_pool"].shape[0]),
        "holdout_n": int(protocol_df.loc[protocol_df["split_role"].astype(str) == "holdout_test"].shape[0]),
        "valid_init_dock_n": int(init_dock_vals.size),
        "valid_holdout_dock_n": int(holdout_dock_vals.size),
        "init_best_dock": init_best_dock,
        "init_mean_dock": init_mean_dock,
    }


def _is_holdout_mask(df: pd.DataFrame) -> np.ndarray:
    split_role = df.get("split_role", pd.Series([""] * df.shape[0])).astype(str).str.lower()
    init_group = df.get("init_group", pd.Series([""] * df.shape[0])).astype(str).str.lower()
    return ((split_role == "holdout_test") | (init_group == "holdout_test")).to_numpy(dtype=bool)


def _load_observed_df(dataset_root: Path, dock_valid_max: float | None) -> pd.DataFrame:
    df = pd.read_csv(dataset_root / "smiles.csv")
    holdout_mask = _is_holdout_mask(df)
    mask = []
    for is_holdout, val in zip(holdout_mask.tolist(), df["dock_score"].tolist()):
        try:
            num = float(val)
        except Exception:
            num = None
        mask.append((not bool(is_holdout)) and _is_valid_dock_score(num, dock_valid_max=dock_valid_max))
    out = df.loc[np.asarray(mask, dtype=bool)].copy().reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No valid optimization-side docked rows available in {dataset_root / 'smiles.csv'}.")
    return out


def _load_holdout_df(dataset_root: Path, dock_valid_max: float | None) -> pd.DataFrame:
    df = pd.read_csv(dataset_root / "smiles.csv")
    holdout_mask = _is_holdout_mask(df)
    if not holdout_mask.any():
        raise RuntimeError(f"No holdout rows found in {dataset_root / 'smiles.csv'}.")
    mask = []
    for is_holdout, val in zip(holdout_mask.tolist(), df["dock_score"].tolist()):
        try:
            num = float(val)
        except Exception:
            num = None
        mask.append(bool(is_holdout) and _is_valid_dock_score(num, dock_valid_max=dock_valid_max))
    out = df.loc[np.asarray(mask, dtype=bool)].copy().reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No valid holdout docked rows available in {dataset_root / 'smiles.csv'}.")
    return out


def _evaluate_holdout_surrogate(
    *,
    holdout_df: pd.DataFrame,
    model: torch.nn.Module,
    mean: float,
    std: float,
    device: torch.device,
    global_vocab: list[str],
    surrogate_kind: str,
    model_node_dim: int,
    model_edge_dim: int,
    model_atom_extra_dim: int,
    model_bond_extra_dim: int,
    model_fp_dim: int,
    model_fp_radius: int,
    surrogate_meta: dict,
    pred_batch_size: int,
    candidate_3d_cfg: dict,
    iter_idx: int,
    method_dir: Path,
) -> dict:
    scored = _score_with_predictions_only(
        holdout_df,
        model=model,
        mean=mean,
        std=std,
        device=device,
        ligand_vocab=global_vocab,
        surrogate_kind=surrogate_kind,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        pred_batch_size=pred_batch_size,
        candidate_3d_cfg=candidate_3d_cfg,
    )
    if scored.empty:
        raise RuntimeError(f"Holdout prediction produced an empty dataframe for iter {iter_idx}.")
    work = scored.copy().reset_index(drop=True)
    work["iter"] = int(iter_idx)
    work["abs_error"] = (pd.to_numeric(work["pred_dock_mean"], errors="coerce") - pd.to_numeric(work["dock_score"], errors="coerce")).abs()
    iter_dir = method_dir / f"iter{iter_idx:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    holdout_pred_path = iter_dir / f"holdout_predictions_iter{iter_idx:03d}.csv"
    work.to_csv(holdout_pred_path, index=False)
    pred = pd.to_numeric(work["pred_dock_mean"], errors="coerce").to_numpy(dtype=float)
    truth = pd.to_numeric(work["dock_score"], errors="coerce").to_numpy(dtype=float)
    pred_std = pd.to_numeric(work.get("pred_dock_std", pd.Series(np.nan, index=work.index)), errors="coerce").to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(np.square(pred - truth))))
    mae = float(np.mean(np.abs(pred - truth)))
    if work.shape[0] >= 2:
        spearman = float(work["pred_dock_mean"].corr(work["dock_score"], method="spearman"))
        kendall = float(work["pred_dock_mean"].corr(work["dock_score"], method="kendall"))
        std_err_spearman = float(work["pred_dock_std"].corr(work["abs_error"], method="spearman")) if "pred_dock_std" in work.columns else float("nan")
    else:
        spearman = float("nan")
        kendall = float("nan")
        std_err_spearman = float("nan")
    rmv = float(np.sqrt(np.mean(np.square(pred_std)))) if np.isfinite(pred_std).any() else float("nan")
    top5_idx = np.argsort(pred)[: min(5, truth.size)] if truth.size > 0 else np.array([], dtype=int)
    top5_true_idx = np.argsort(truth)[: min(5, truth.size)] if truth.size > 0 else np.array([], dtype=int)
    top5_pred_set = {int(x) for x in top5_idx.tolist()}
    top5_true_set = {int(x) for x in top5_true_idx.tolist()}
    if top5_true_set:
        hit_at_5 = float(len(top5_pred_set & top5_true_set) / len(top5_true_set))
    else:
        hit_at_5 = float("nan")
    return {
        "iter": int(iter_idx),
        "n": int(work.shape[0]),
        "spearman": spearman,
        "kendall": kendall,
        "rmse": rmse,
        "mae": mae,
        "rmv": rmv,
        "calibration_gap": float(abs(rmv - rmse)) if np.isfinite(rmv) else float("nan"),
        "std_error_spearman": std_err_spearman,
        "hit_at_5": hit_at_5,
        "top5_true_mean": float(np.mean(truth[top5_idx])) if top5_idx.size > 0 else float("nan"),
        "predictions_path": str(holdout_pred_path),
    }


def _candidate_frame(smiles_list: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    seen: set[str] = set()
    for smi in smiles_list:
        canon = canonicalize_smiles_noh(str(smi))
        if not canon:
            raise RuntimeError(f"Failed to canonicalize generated SMILES: {smi!r}")
        if canon in seen:
            continue
        seen.add(canon)
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse generated SMILES: {canon}")
        mol = Chem.RemoveHs(mol, sanitize=True)
        rows.append(
            {
                "smiles": canon,
                "smiles_canonical": canon,
                "qed": float(QED.qed(mol)),
                "sa_score": float(calc_sa_score_mol(mol)),
                "logp": float(Crippen.MolLogP(mol)),
                "tpsa": float(CalcTPSA(mol)),
                "hbd": float(CalcNumHBD(mol)),
                "hba": float(CalcNumHBA(mol)),
                "fsp3": float(CalcFractionCSP3(mol)),
            }
        )
    if not rows:
        raise RuntimeError("Generated candidate pool is empty after canonicalization.")
    return pd.DataFrame(rows)



def _score_with_qpmhi_analytic(
    candidate_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    model: torch.nn.Module,
    mean: float,
    std: float,
    device: torch.device,
    ligand_vocab: list[str],
    surrogate_kind: str,
    model_node_dim: int,
    model_edge_dim: int,
    model_atom_extra_dim: int,
    model_bond_extra_dim: int,
    model_fp_dim: int,
    model_fp_radius: int,
    surrogate_meta: dict,
    pred_batch_size: int,
    candidate_3d_cfg: dict,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    weights: list[float],
    ref_point: list[float],
) -> pd.DataFrame:
    loader, kept = _prepare_loader(
        candidate_df["smiles_canonical"].astype(str).tolist(),
        surrogate_kind=surrogate_kind,
        ligand_vocab=ligand_vocab,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        batch_size=pred_batch_size,
        candidate_3d_cfg=candidate_3d_cfg,
    )
    scored = candidate_df.set_index("smiles_canonical").loc[kept].reset_index()
    scored, pred_mean, pred_std_total = _attach_decomposed_predictions(scored, model, loader, mean=mean, std=std, device=device)
    pred_std_eff = _build_qpmhi_sigma_eff(scored)
    exact_obj = torch.stack(
        [
            qed_sign * torch.tensor(scored["qed"].to_numpy(dtype=np.float32)),
            sa_sign * torch.tensor(scored["sa_score"].to_numpy(dtype=np.float32)),
        ],
        dim=-1,
    )
    y_train = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    scores, qpmhi_meta = qphv_prob_gaussian_analytic_3d(
        dock_mu=dock_sign * pred_mean.to(torch.float32),
        #dock_sigma=pred_std_eff.to(torch.float32),
        dock_sigma=pred_std_total.to(torch.float32),
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
        return_metadata=True,
    )
    _log(
        "qpmhi analytic prefilter | "
        f"input_n={qpmhi_meta['input_n']} "
        f"scored_n={qpmhi_meta['scored_n']} "
        f"ref_rect_zero_n={qpmhi_meta['prefilter_ref_rect_zero_n']} "
        f"no_support_n={qpmhi_meta['prefilter_no_support_n']} "
        f"zero_quadrature_gain_n={qpmhi_meta['prefilter_zero_quadrature_gain_n']} "
        f"sigma_total_mean={float(pred_std_total.mean().item()):.4f} "
        f"sigma_eff_mean={float(pred_std_eff.mean().item()):.4f}"
    )
    scored["pred_dock_mean"] = pred_mean.detach().cpu().numpy()
    scored["pred_dock_std"] = pred_std_total.detach().cpu().numpy()
    scored["score_qpmhi"] = scores.detach().cpu().numpy()
    return scored


def _score_with_predictions_only(
    candidate_df: pd.DataFrame,
    model: torch.nn.Module,
    mean: float,
    std: float,
    device: torch.device,
    ligand_vocab: list[str],
    surrogate_kind: str,
    model_node_dim: int,
    model_edge_dim: int,
    model_atom_extra_dim: int,
    model_bond_extra_dim: int,
    model_fp_dim: int,
    model_fp_radius: int,
    surrogate_meta: dict,
    pred_batch_size: int,
    candidate_3d_cfg: dict,
) -> pd.DataFrame:
    loader, kept = _prepare_loader(
        candidate_df["smiles_canonical"].astype(str).tolist(),
        surrogate_kind=surrogate_kind,
        ligand_vocab=ligand_vocab,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        batch_size=pred_batch_size,
        candidate_3d_cfg=candidate_3d_cfg,
    )
    scored = candidate_df.set_index("smiles_canonical").loc[kept].reset_index()
    scored, pred_mean, pred_std_total = _attach_decomposed_predictions(scored, model, loader, mean=mean, std=std, device=device)
    if np.isfinite(pd.to_numeric(scored["pred_dock_std_epi"], errors="coerce").to_numpy(dtype=np.float64)).all() and np.isfinite(pd.to_numeric(scored["pred_dock_std_ale"], errors="coerce").to_numpy(dtype=np.float64)).all():
        _build_qpmhi_sigma_eff(scored)
    else:
        scored["pred_dock_std_eff"] = pred_std_total.detach().cpu().numpy()
    scored["score_qpmhi"] = np.nan
    scored["score_qnehvi"] = np.nan
    return scored


def _score_with_qnehvi(
    candidate_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    model: torch.nn.Module,
    mean: float,
    std: float,
    device: torch.device,
    ligand_vocab: list[str],
    surrogate_kind: str,
    model_node_dim: int,
    model_edge_dim: int,
    model_atom_extra_dim: int,
    model_bond_extra_dim: int,
    model_fp_dim: int,
    model_fp_radius: int,
    surrogate_meta: dict,
    pred_batch_size: int,
    candidate_3d_cfg: dict,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    weights: list[float],
    ref_point: list[float],
) -> pd.DataFrame:
    loader, kept = _prepare_loader(
        candidate_df["smiles_canonical"].astype(str).tolist(),
        surrogate_kind=surrogate_kind,
        ligand_vocab=ligand_vocab,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        batch_size=pred_batch_size,
        candidate_3d_cfg=candidate_3d_cfg,
    )
    kept_scored = candidate_df.set_index("smiles_canonical").loc[kept].reset_index()
    kept_scored, pred_mean, pred_std = _attach_decomposed_predictions(kept_scored, model, loader, mean=mean, std=std, device=device)
    exact_obj = torch.stack(
        [
            qed_sign * torch.tensor(kept_scored["qed"].to_numpy(dtype=np.float32)),
            sa_sign * torch.tensor(kept_scored["sa_score"].to_numpy(dtype=np.float32)),
        ],
        dim=-1,
    )
    y_train = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    scores, nehvi_meta = nehvi_gaussian_analytic_3d(
        dock_mu=dock_sign * pred_mean.to(torch.float32),
        dock_sigma=pred_std.to(torch.float32),
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
        return_metadata=True,
    )
    _log(
        "nehvi analytic prefilter | "
        f"input_n={nehvi_meta['input_n']} "
        f"scored_n={nehvi_meta['scored_n']} "
        f"ref_rect_zero_n={nehvi_meta['prefilter_ref_rect_zero_n']} "
        f"no_support_n={nehvi_meta['prefilter_no_support_n']} "
        f"sigma_mean={float(pred_std.mean().item()):.4f}"
    )
    kept_scored["score_qnehvi"] = scores.detach().cpu().numpy()
    kept_scored["score_qpmhi"] = np.nan
    return kept_scored


def _attach_decomposed_predictions(
    scored: pd.DataFrame,
    model: torch.nn.Module,
    loader,
    mean: float,
    std: float,
    device: torch.device,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    pred_mean, pred_std_total, pred_std_epi, pred_std_ale = predict_gaussian_decomposed_stats(
        model,
        loader,
        mean=mean,
        std=std,
        device=device,
    )
    scored["pred_dock_mean"] = pred_mean.detach().cpu().numpy()
    scored["pred_dock_std"] = pred_std_total.detach().cpu().numpy()
    scored["pred_dock_std_total"] = pred_std_total.detach().cpu().numpy()
    scored["pred_dock_std_epi"] = pred_std_epi.detach().cpu().numpy()
    scored["pred_dock_std_ale"] = pred_std_ale.detach().cpu().numpy()
    epi_var = np.square(scored["pred_dock_std_epi"].to_numpy(dtype=np.float64))
    total_var = np.square(scored["pred_dock_std_total"].to_numpy(dtype=np.float64))
    frac = np.full(total_var.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(epi_var) & np.isfinite(total_var) & (total_var > 0.0)
    frac[valid] = epi_var[valid] / total_var[valid]
    scored["pred_dock_var_epi_frac"] = frac
    return scored, pred_mean, pred_std_total


def _build_qpmhi_sigma_eff(scored: pd.DataFrame) -> torch.Tensor:
    required_cols = ["pred_dock_std_epi", "pred_dock_std_ale", "pred_dock_std_total"]
    missing = [col for col in required_cols if col not in scored.columns]
    if missing:
        raise RuntimeError(f"qPMHI sigma_eff requires decomposition columns, missing: {missing}")
    epi = pd.to_numeric(scored["pred_dock_std_epi"], errors="coerce").to_numpy(dtype=np.float64)
    ale = pd.to_numeric(scored["pred_dock_std_ale"], errors="coerce").to_numpy(dtype=np.float64)
    if not (np.isfinite(epi).all() and np.isfinite(ale).all()):
        raise RuntimeError("qPMHI sigma_eff encountered non-finite epi/ale uncertainty values.")
    epi_var = np.square(epi)
    ale_var = np.square(ale)
    total_var = epi_var + ale_var
    ale_weight = np.zeros_like(total_var, dtype=np.float64)
    valid = total_var > 0.0
    ale_weight[valid] = epi_var[valid] / total_var[valid]
    eff = np.sqrt(epi_var + ale_weight * ale_var)
    if not (np.isfinite(ale_weight).all() and np.isfinite(eff).all()):
        raise RuntimeError("qPMHI sigma_eff produced non-finite adaptive aleatoric weights.")
    scored["pred_dock_ale_weight_eff"] = ale_weight
    scored["pred_dock_std_eff"] = eff
    return torch.tensor(eff, dtype=torch.float32)


def _uncertainty_summary(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    def _mean_or_nan(col: str) -> float:
        if col not in df.columns:
            return float("nan")
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size > 0 else float("nan")

    return {
        f"{prefix}_pred_std_total_mean": _mean_or_nan("pred_dock_std_total"),
        f"{prefix}_pred_std_epi_mean": _mean_or_nan("pred_dock_std_epi"),
        f"{prefix}_pred_std_ale_mean": _mean_or_nan("pred_dock_std_ale"),
        f"{prefix}_pred_var_epi_frac_mean": _mean_or_nan("pred_dock_var_epi_frac"),
    }

def _rank_gate_from_scores(scores: pd.Series, eps: float, gamma: float) -> np.ndarray:
    values = pd.to_numeric(scores, errors="coerce")
    if values.isna().any():
        raise RuntimeError("Rank-gated set selection encountered non-finite qPMHI scores.")
    n = int(values.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        rank_norm = np.ones((1,), dtype=np.float64)
    else:
        ranks = values.rank(method="average", ascending=False).to_numpy(dtype=np.float64)
        rank_norm = 1.0 - (ranks - 1.0) / float(n - 1)
    return float(eps) + (1.0 - float(eps)) * np.power(rank_norm, float(gamma))


def _set_selection_point_from_row(
    row: pd.Series,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    conservative_lambda: float,
) -> tuple[torch.Tensor, float]:
    pred_dock_mean = float(pd.to_numeric(pd.Series([row.get("pred_dock_mean")]), errors="coerce").iloc[0])
    std_eff_raw = row.get("pred_dock_std_eff", row.get("pred_dock_std"))
    pred_dock_std_eff = float(pd.to_numeric(pd.Series([std_eff_raw]), errors="coerce").iloc[0])
    qed_val = float(pd.to_numeric(pd.Series([row.get("qed")]), errors="coerce").iloc[0])
    sa_val = float(pd.to_numeric(pd.Series([row.get("sa_score")]), errors="coerce").iloc[0])
    if not np.isfinite(pred_dock_mean):
        raise RuntimeError("Set-aware selection requires finite pred_dock_mean.")
    if not np.isfinite(pred_dock_std_eff):
        raise RuntimeError("Set-aware selection requires finite pred_dock_std_eff.")
    if not np.isfinite(qed_val) or not np.isfinite(sa_val):
        raise RuntimeError("Set-aware selection requires finite QED and SA values.")
    dock_cons = pred_dock_mean + float(conservative_lambda) * pred_dock_std_eff
    point = torch.tensor(
        [
            dock_sign * dock_cons,
            qed_sign * qed_val,
            sa_sign * sa_val,
        ],
        dtype=torch.float32,
    )
    return point, dock_cons


def _greedy_select_batch_set_hv(
    *,
    subpool: pd.DataFrame,
    labeled_df: pd.DataFrame,
    batch_size: int,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    ref_point: list[float],
    conservative_lambda: float,
    gate_values: np.ndarray,
    selection_mode: str,
) -> pd.DataFrame:
    if subpool.empty:
        raise RuntimeError(f"{selection_mode} received an empty candidate subpool.")
    if int(subpool.shape[0]) < int(batch_size):
        raise RuntimeError(
            f"{selection_mode} subpool size {int(subpool.shape[0])} is smaller than batch_size {int(batch_size)}."
        )
    if int(gate_values.shape[0]) != int(subpool.shape[0]):
        raise RuntimeError(
            f"{selection_mode} gate size mismatch: gate_n={int(gate_values.shape[0])} subpool_n={int(subpool.shape[0])}."
        )
    work = subpool.copy().reset_index(drop=True)
    work["score_qpmhi_gate"] = gate_values.astype(np.float64)
    current_points = build_objective_tensor(
        labeled_df,
        dock_sign=dock_sign,
        qed_sign=qed_sign,
        sa_sign=sa_sign,
    ).to(torch.float32)
    selected_rows: list[pd.Series] = []

    for step_idx in range(int(batch_size)):
        hv_before = float(compute_hypervolume(current_points, ref_point=ref_point))
        best_key = None
        best_idx = None
        best_row = None
        best_point = None
        best_dock_cons = None
        for cand_idx, row in work.iterrows():
            point, dock_cons = _set_selection_point_from_row(
                row,
                dock_sign=dock_sign,
                qed_sign=qed_sign,
                sa_sign=sa_sign,
                conservative_lambda=conservative_lambda,
            )
            hv_after = float(
                compute_hypervolume(
                    torch.cat([current_points, point.view(1, -1)], dim=0),
                    ref_point=ref_point,
                )
            )
            marginal_hv = max(0.0, hv_after - hv_before)
            gate = float(row["score_qpmhi_gate"])
            gate_times_hv = gate * marginal_hv
            score_qpmhi = row.get("score_qpmhi", np.nan)
            score_qpmhi = float(score_qpmhi) if pd.notna(score_qpmhi) else float("-inf")
            pred_dock_mean = float(row["pred_dock_mean"]) if pd.notna(row.get("pred_dock_mean", np.nan)) else float("inf")
            key = (
                gate_times_hv,
                marginal_hv,
                gate,
                score_qpmhi,
                -pred_dock_mean,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = int(cand_idx)
                best_row = row.copy()
                best_point = point
                best_dock_cons = float(dock_cons)
        if best_idx is None or best_row is None or best_point is None or best_dock_cons is None:
            raise RuntimeError(f"{selection_mode} failed to choose a candidate at step {step_idx + 1}.")
        best_row["selection_mode"] = selection_mode
        best_row["selection_step"] = int(step_idx + 1)
        best_row["pred_dock_mean_cons"] = best_dock_cons
        best_row["set_obj_dock"] = float(best_point[0].item())
        best_row["set_obj_qed"] = float(best_point[1].item())
        best_row["set_obj_sa"] = float(best_point[2].item())
        best_row["set_marginal_hv_at_selection"] = float(best_key[1])
        best_row["set_gate_times_hv"] = float(best_key[0])
        score_qpmhi_log = ""
        if pd.notna(best_row.get("score_qpmhi", np.nan)):
            score_qpmhi_log = f"{float(best_row['score_qpmhi']):.6f}"
        _log(
            f"{selection_mode} step {step_idx + 1}/{batch_size} | "
            f"subpool_n={work.shape[0]} smiles={best_row['smiles_canonical']} "
            f"score_qpmhi={score_qpmhi_log} "
            f"gate={float(best_row['score_qpmhi_gate']):.6f} "
            f"marginal_hv={float(best_key[1]):.6f} "
            f"gate_times_hv={float(best_key[0]):.6f}"
        )
        selected_rows.append(best_row)
        current_points = torch.cat([current_points, best_point.view(1, -1)], dim=0)
        work = work.drop(index=best_idx).reset_index(drop=True)

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _select_batch_qpmhi_set_hv(
    *,
    scored: pd.DataFrame,
    labeled_df: pd.DataFrame,
    batch_size: int,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    ref_point: list[float],
    topk_factor: int,
    gamma: float,
    gate_eps: float,
    conservative_lambda: float,
) -> pd.DataFrame:
    if "score_qpmhi" not in scored.columns:
        raise RuntimeError("qpmhi_set_hv requires score_qpmhi in the scored dataframe.")
    topk = min(int(scored.shape[0]), max(int(batch_size), int(topk_factor) * int(batch_size)))
    subpool = (
        scored.sort_values(["score_qpmhi", "pred_dock_mean"], ascending=[False, True])
        .head(topk)
        .copy()
        .reset_index(drop=True)
    )
    gates = _rank_gate_from_scores(subpool["score_qpmhi"], eps=gate_eps, gamma=gamma)
    return _greedy_select_batch_set_hv(
        subpool=subpool,
        labeled_df=labeled_df,
        batch_size=batch_size,
        dock_sign=dock_sign,
        qed_sign=qed_sign,
        sa_sign=sa_sign,
        ref_point=ref_point,
        conservative_lambda=conservative_lambda,
        gate_values=gates,
        selection_mode="qpmhi_set_hv",
    )


def _select_batch_mean_set_hv(
    *,
    scored: pd.DataFrame,
    labeled_df: pd.DataFrame,
    batch_size: int,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    ref_point: list[float],
    topk_factor: int,
    conservative_lambda: float,
) -> pd.DataFrame:
    topk = min(int(scored.shape[0]), max(int(batch_size), int(topk_factor) * int(batch_size)))
    subpool = (
        scored.sort_values(["pred_dock_mean", "pred_dock_std"], ascending=[True, False])
        .head(topk)
        .copy()
        .reset_index(drop=True)
    )
    return _greedy_select_batch_set_hv(
        subpool=subpool,
        labeled_df=labeled_df,
        batch_size=batch_size,
        dock_sign=dock_sign,
        qed_sign=qed_sign,
        sa_sign=sa_sign,
        ref_point=ref_point,
        conservative_lambda=conservative_lambda,
        gate_values=np.ones((subpool.shape[0],), dtype=np.float64),
        selection_mode="mean_set_hv",
    )


def _select_batch_main(
    method: str,
    scored: pd.DataFrame,
    labeled_df: pd.DataFrame,
    batch_size: int,
    score_eps: float,
    seed: int,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    ref_point: list[float],
    set_hv_topk_factor: int,
    qpmhi_set_hv_gamma: float,
    qpmhi_set_hv_gate_eps: float,
    set_hv_conservative_lambda: float,
) -> tuple[pd.DataFrame, int]:
    if method == "random":
        out = scored.sample(n=int(batch_size), random_state=int(seed), replace=False).copy().reset_index(drop=True)
        out["selection_mode"] = "random"
        return out, 0
    if method == "qnehvi":
        out = scored.sort_values(["score_qnehvi", "pred_dock_mean"], ascending=[False, True]).head(int(batch_size)).copy().reset_index(drop=True)
        out["selection_mode"] = "qnehvi"
        return out, 0
    if method == "qpmhi":
        out = scored.sort_values(["score_qpmhi", "pred_dock_mean"], ascending=[False, True]).head(int(batch_size)).copy().reset_index(drop=True)
        out["selection_mode"] = "qpmhi"
        return out, 0
    if method == "qpmhi_set_hv":
        out = _select_batch_qpmhi_set_hv(
            scored=scored,
            labeled_df=labeled_df,
            batch_size=batch_size,
            dock_sign=dock_sign,
            qed_sign=qed_sign,
            sa_sign=sa_sign,
            ref_point=ref_point,
            topk_factor=set_hv_topk_factor,
            gamma=qpmhi_set_hv_gamma,
            gate_eps=qpmhi_set_hv_gate_eps,
            conservative_lambda=set_hv_conservative_lambda,
        )
        return out.reset_index(drop=True), 0
    if method == "mean_set_hv":
        out = _select_batch_mean_set_hv(
            scored=scored,
            labeled_df=labeled_df,
            batch_size=batch_size,
            dock_sign=dock_sign,
            qed_sign=qed_sign,
            sa_sign=sa_sign,
            ref_point=ref_point,
            topk_factor=set_hv_topk_factor,
            conservative_lambda=set_hv_conservative_lambda,
        )
        return out.reset_index(drop=True), 0
    if method == "qpmhi_fallback":
        primary = scored.loc[scored["score_qpmhi"] > float(score_eps)].copy()
        primary = primary.sort_values(["score_qpmhi", "pred_dock_mean"], ascending=[False, True]).head(int(batch_size)).copy()
        primary["selection_mode"] = "qpmhi"
        fallback_n = max(0, int(batch_size) - primary.shape[0])
        if fallback_n <= 0:
            return primary.reset_index(drop=True), 0
        fb = _apply_fallback(scored, labeled_df=labeled_df, batch_size=fallback_n, already=set(primary["smiles_canonical"].tolist()))
        fb["selection_mode"] = "uncertainty_fallback"
        out = pd.concat([primary, fb], axis=0, ignore_index=True)
        return out, int(fb.shape[0])
    raise ValueError(f"Unsupported method: {method}")


def _batch_hv_gain(labeled_df: pd.DataFrame, batch_df: pd.DataFrame, dock_sign: float, qed_sign: float, sa_sign: float, ref_point: list[float]) -> float:
    base = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    batch = build_objective_tensor(batch_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    hv_before = compute_hypervolume(base, ref_point=ref_point)
    hv_after = compute_hypervolume(torch.cat([base, batch], dim=0), ref_point=ref_point)
    return float(hv_after - hv_before)


def _dynamic_ref_point_from_labeled(labeled_df: pd.DataFrame, dock_sign: float, qed_sign: float, sa_sign: float) -> list[float]:
    observed = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign).to(torch.float64)
    if observed.numel() == 0:
        raise RuntimeError('Cannot build dynamic acquisition reference point from an empty labeled set.')
    mins = observed.min(dim=0).values
    maxs = observed.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-3)
    margin = torch.maximum(0.05 * span, torch.full_like(span, 1e-3))
    return (mins - margin).detach().cpu().tolist()


def _method_seed(base_seed: int, method: str, iter_idx: int) -> int:
    return int(base_seed) + 1009 * int(iter_idx) + sum(ord(ch) for ch in str(method))


def _resolve_oracle_asset_root(oracle_cfg: dict) -> Path | None:
    for key in ("oracle_asset_root", "asset_root", "dataset_root"):
        val = oracle_cfg.get(key)
        if val:
            p = Path(str(val)).resolve()
            if not p.exists():
                raise FileNotFoundError(f"oracle asset root not found: {p}")
            return p
    protein = oracle_cfg.get("protein_pdb") or oracle_cfg.get("protein") or oracle_cfg.get("protein_path")
    if protein:
        p = Path(str(protein)).resolve().parent
        return p if p.exists() else None
    return None


def _default_init_library_candidates(init_training_dataset_root: str | Path | None, oracle_asset_root: Path | None) -> list[Path]:
    """Return a single default init-library reference.

    The various tcm_library / pretrain CSVs are treated as equivalent historical copies.
    We therefore keep exactly one default reference here to avoid silently switching among
    duplicated libraries.
    """
    if init_training_dataset_root:
        root = Path(init_training_dataset_root).resolve()
        return [root / "tcm_library.csv"]
    return [REPO / "dataset" / "tcm_merged_pretrain" / "smiles.csv"]


def _export_selected_batch_with_oracle(
    dataset_root: Path,
    selected_df: pd.DataFrame,
    new_ids: list[str],
    out_path: Path,
) -> pd.DataFrame:
    obs = pd.read_csv(dataset_root / "smiles.csv")
    obs["ligand_id"] = obs["ligand_id"].astype(str)

    batch = selected_df.copy().reset_index(drop=True)
    batch["ligand_id"] = [str(x) for x in new_ids]
    cols = [
        "ligand_id",
        "dock_score",
        "dock_source",
        "dock_pose",
        "qed",
        "sa_score",
        "split",
        "added_iter",
        "molecule_origin",
    ]
    cols = [c for c in cols if c in obs.columns]
    merged = batch.merge(obs[cols], on="ligand_id", how="left")

    if "pred_dock_mean" in merged.columns and "dock_score" in merged.columns:
        pred = pd.to_numeric(merged["pred_dock_mean"], errors="coerce")
        true = pd.to_numeric(merged["dock_score"], errors="coerce")
        merged["dock_error"] = pred - true
        merged["abs_error"] = (pred - true).abs()
        merged["sq_error"] = (pred - true) ** 2
        merged["oracle_success"] = np.isfinite(true)

    merged.to_csv(out_path, index=False)
    return merged


def _load_existing_iter_rows(metrics_path: Path, rounds: int) -> tuple[list[dict], int]:
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    if df.empty:
        return [], -1
    if "iter" not in df.columns:
        raise RuntimeError(f"iter_metrics missing 'iter' column: {metrics_path}")
    iter_series = pd.to_numeric(df["iter"], errors="raise").astype(int)
    iters = iter_series.tolist()
    if len(set(iters)) != len(iters):
        raise RuntimeError(f"Duplicate iter rows in {metrics_path}: {iters}")
    expected = list(range(max(iters) + 1))
    if sorted(iters) != expected:
        raise RuntimeError(f"Non-contiguous iter rows in {metrics_path}: {iters}")
    if max(iters) > int(rounds):
        raise RuntimeError(f"Existing iter row exceeds configured rounds in {metrics_path}: {max(iters)} > {rounds}")
    out_df = df.copy()
    out_df["iter"] = iter_series
    return out_df.to_dict("records"), int(iter_series.max())



def _load_existing_last_iter_row(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None
    if "iter" not in df.columns:
        raise RuntimeError(f"iter_metrics missing 'iter' column: {metrics_path}")
    row = df.iloc[-1].to_dict()
    row["iter"] = int(pd.to_numeric(pd.Series([row["iter"]]), errors="raise").iloc[0])
    return row



def _resolve_append_global_vocab(
    *,
    out_root: Path,
    existing_methods: list[str],
    init_df: pd.DataFrame,
    init_training_dataset_root: str | Path | None,
    init_ligand_vocab_file: str | Path | None,
) -> list[str]:
    for method in existing_methods:
        dataset_root = out_root / method / "dataset"
        vocab = _load_ligand_vocab_override(dataset_root)
        if vocab:
            return list(vocab)
    return _resolve_init_pretrain_ligand_vocab(
        init_training_dataset_root=init_training_dataset_root,
        init_ligand_vocab_file=init_ligand_vocab_file,
        init_library_df=init_df,
    )



def _require_cache_match(*, csv_path: Path, meta_path: Path, expected_meta: dict, label: str, append_mode: bool) -> pd.DataFrame | None:
    if csv_path.exists():
        actual_meta = _load_cache_metadata(meta_path)
        if not _meta_matches(actual_meta, expected_meta):
            if append_mode:
                raise RuntimeError(
                    f"Append mode requires metadata-matched cached {label}: {meta_path}"
                )
            return None
        return pd.read_csv(csv_path)
    if append_mode:
        raise FileNotFoundError(f"Append mode requires cached {label}: {csv_path}")
    return None



def _source_csv_cache_fingerprint(source_csv_path: Path | None, raw_rows: int, smiles_col: str) -> dict:
    payload = {
        "source_csv_path": str(source_csv_path.resolve()) if source_csv_path is not None else None,
        "raw_rows": int(raw_rows),
        "smiles_col": str(smiles_col),
    }
    if source_csv_path is not None and source_csv_path.exists():
        st = source_csv_path.stat()
        payload["source_mtime_ns"] = int(st.st_mtime_ns)
        payload["source_size"] = int(st.st_size)
    else:
        payload["source_mtime_ns"] = None
        payload["source_size"] = None
    return payload


def _build_or_load_downstream_protocol(
    *,
    out_root: Path,
    downstream_raw_df: pd.DataFrame,
    smiles_col: str,
    source_csv_path: Path | None,
    downstream_pool_ratio: tuple[float, float],
    downstream_pool_seed: int,
    scaffold_include_chirality: bool,
    butina_cutoff: float,
    fp_radius: int,
    fp_bits: int,
    selection_seed: int,
    init_train_valid_total: int,
    holdout_test_total: int,
    predictor_valid_ratio: float,
    predictor_valid_seed: int,
    append_mode: bool,
) -> dict:
    cache_dir = out_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _log("downstream protocol start | checking canonical library cache")
    canonical_csv = cache_dir / "canonical_init_library.csv"
    canonical_meta = cache_dir / "canonical_init_library.meta.json"
    canonical_expected = _source_csv_cache_fingerprint(
        source_csv_path=source_csv_path,
        raw_rows=int(downstream_raw_df.shape[0]),
        smiles_col=str(smiles_col),
    )
    canonical_df = _require_cache_match(
        csv_path=canonical_csv,
        meta_path=canonical_meta,
        expected_meta=canonical_expected,
        label="canonical init library",
        append_mode=append_mode,
    )
    if canonical_df is None:
        _log(
            "downstream canonical library recomputing | "
            f"raw_rows={int(downstream_raw_df.shape[0])} smiles_col={smiles_col}"
        )
        canonical_df = _canonicalize_library_df(downstream_raw_df, smiles_col=smiles_col, require_unique=True)
        _write_cached_dataframe(canonical_df, canonical_csv, canonical_meta, canonical_expected)
        _log(
            "downstream canonical library recomputed | "
            f"rows={int(canonical_df.shape[0])} cache={canonical_csv}"
        )
    else:
        _log(
            "downstream canonical library loaded_from_cache | "
            f"rows={int(canonical_df.shape[0])} cache={canonical_csv}"
        )

    init_library_csv = out_root / "init_library.csv"
    if (not init_library_csv.exists()) or append_mode:
        canonical_df.to_csv(init_library_csv, index=False)

    library_signature = _smiles_signature(canonical_df["smiles_canonical"].astype(str).tolist())
    base_meta = {
        "source_csv_path": str(source_csv_path.resolve()) if source_csv_path is not None else None,
        "rows": int(canonical_df.shape[0]),
        "smiles_col": str(smiles_col),
        "smiles_signature": library_signature,
    }

    _log("downstream protocol continue | checking scaffold annotation cache")
    scaffold_annotations_csv = cache_dir / "downstream_scaffold_annotations.csv"
    scaffold_annotations_meta = cache_dir / "downstream_scaffold_annotations.meta.json"
    scaffold_annotations_expected = {
        **base_meta,
        "include_chirality": bool(scaffold_include_chirality),
    }
    scaffold_annotations_df = _require_cache_match(
        csv_path=scaffold_annotations_csv,
        meta_path=scaffold_annotations_meta,
        expected_meta=scaffold_annotations_expected,
        label="downstream scaffold annotations",
        append_mode=append_mode,
    )
    if scaffold_annotations_df is None:
        scaffold_annotations_df, _ = _build_scaffold_groups(
            canonical_df,
            smiles_col="smiles_canonical",
            include_chirality=bool(scaffold_include_chirality),
        )
        _write_cached_dataframe(
            scaffold_annotations_df,
            scaffold_annotations_csv,
            scaffold_annotations_meta,
            scaffold_annotations_expected,
        )
        _log(
            "downstream scaffold annotations recomputed | "
            f"unique_scaffolds={int(scaffold_annotations_df['murcko_scaffold'].astype(str).nunique())} "
            f"cache={scaffold_annotations_csv}"
        )
    else:
        _log(
            "downstream scaffold annotations loaded_from_cache | "
            f"unique_scaffolds={int(scaffold_annotations_df['murcko_scaffold'].astype(str).nunique())} "
            f"cache={scaffold_annotations_csv}"
        )

    _log("downstream protocol continue | checking scaffold split cache")
    scaffold_csv = cache_dir / "downstream_scaffold_split.csv"
    scaffold_meta = cache_dir / "downstream_scaffold_split.meta.json"
    scaffold_expected = {
        **base_meta,
        "seed": int(downstream_pool_seed),
        "pool_ratio": [float(x) for x in downstream_pool_ratio],
        "include_chirality": bool(scaffold_include_chirality),
    }
    scaffold_df = _require_cache_match(
        csv_path=scaffold_csv,
        meta_path=scaffold_meta,
        expected_meta=scaffold_expected,
        label="downstream scaffold split",
        append_mode=append_mode,
    )
    if scaffold_df is None:
        scaffold_df = _scaffold_split_downstream_pool(
            scaffold_annotations_df,
            smiles_col="smiles_canonical",
            pool_ratio=downstream_pool_ratio,
            seed=int(downstream_pool_seed),
            include_chirality=bool(scaffold_include_chirality),
        )
        _write_cached_dataframe(scaffold_df, scaffold_csv, scaffold_meta, scaffold_expected)
        _log(
            "downstream scaffold split recomputed | "
            f"pool_opt={int((scaffold_df['downstream_pool'] == 'pool_opt').sum())} "
            f"pool_eval={int((scaffold_df['downstream_pool'] == 'pool_eval').sum())} cache={scaffold_csv}"
        )
    else:
        _log(
            "downstream scaffold split loaded_from_cache | "
            f"pool_opt={int((scaffold_df['downstream_pool'] == 'pool_opt').sum())} "
            f"pool_eval={int((scaffold_df['downstream_pool'] == 'pool_eval').sum())} cache={scaffold_csv}"
        )
    pool_opt_df = scaffold_df.loc[scaffold_df["downstream_pool"].astype(str) == "pool_opt"].copy().reset_index(drop=True)
    pool_eval_df = scaffold_df.loc[scaffold_df["downstream_pool"].astype(str) == "pool_eval"].copy().reset_index(drop=True)
    if pool_opt_df.empty or pool_eval_df.empty:
        raise RuntimeError("Downstream scaffold split produced an empty pool_opt or pool_eval.")
    pool_opt_csv = cache_dir / "pool_opt.csv"
    pool_eval_csv = cache_dir / "pool_eval.csv"
    if (not pool_opt_csv.exists()) or append_mode:
        pool_opt_df.to_csv(pool_opt_csv, index=False)
    if (not pool_eval_csv.exists()) or append_mode:
        pool_eval_df.to_csv(pool_eval_csv, index=False)

    def _run_cached_butina(pool_df: pd.DataFrame, *, label: str, cluster_csv: Path, selected_csv: Path, meta_path: Path, target_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        expected_meta = {
            **base_meta,
            "pool_label": str(label),
            "pool_rows": int(pool_df.shape[0]),
            "pool_signature": _smiles_signature(pool_df["smiles_canonical"].astype(str).tolist()),
            "target_count": int(target_count),
            "butina_cutoff": float(butina_cutoff),
            "fp_radius": int(fp_radius),
            "fp_bits": int(fp_bits),
            "selection_seed": int(selection_seed),
        }
        assignments_df = _require_cache_match(
            csv_path=cluster_csv,
            meta_path=meta_path,
            expected_meta=expected_meta,
            label=f"{label} Butina clusters",
            append_mode=append_mode,
        )
        selected_df = None
        if assignments_df is not None:
            if not selected_csv.exists():
                if append_mode:
                    raise FileNotFoundError(f"Append mode requires cached {label} selection: {selected_csv}")
            else:
                selected_df = pd.read_csv(selected_csv)
                _log(f"Butina selection loaded_from_cache | label={label} selected={selected_df.shape[0]} cache={selected_csv}")
        if assignments_df is None or selected_df is None:
            fps = _compute_morgan_fingerprints(pool_df, smiles_col="smiles_canonical", radius=int(fp_radius), n_bits=int(fp_bits))
            clusters = _butina_cluster_indices(fps, cutoff=float(butina_cutoff))
            assignments_df, selected_df = _select_representatives_from_clusters(
                pool_df,
                clusters,
                target_count=int(target_count),
                seed=int(selection_seed),
                smiles_col="smiles_canonical",
            )
            _write_cached_dataframe(assignments_df, cluster_csv, meta_path, expected_meta)
            selected_df.to_csv(selected_csv, index=False)
            _log(f"Butina selection recomputed | label={label} selected={selected_df.shape[0]} cache={selected_csv}")
        return assignments_df, selected_df

    pool_opt_assignments, init_selected_df = _run_cached_butina(
        pool_opt_df,
        label="pool_opt",
        cluster_csv=cache_dir / "pool_opt_butina_clusters.csv",
        selected_csv=cache_dir / "init_selected_480.csv",
        meta_path=cache_dir / "pool_opt_butina_clusters.meta.json",
        target_count=int(init_train_valid_total),
    )
    pool_eval_assignments, holdout_selected_df = _run_cached_butina(
        pool_eval_df,
        label="pool_eval",
        cluster_csv=cache_dir / "pool_eval_butina_clusters.csv",
        selected_csv=cache_dir / "holdout_selected_120.csv",
        meta_path=cache_dir / "pool_eval_butina_clusters.meta.json",
        target_count=int(holdout_test_total),
    )
    init_selected_df = _ensure_property_columns(init_selected_df)
    holdout_selected_df = _ensure_property_columns(holdout_selected_df)
    predictor_split_df = _write_predictor_init_split_cache(
        init_df=init_selected_df,
        cache_csv=cache_dir / "predictor_init_random_split.csv",
        cache_meta=cache_dir / "predictor_init_random_split.meta.json",
        source_csv_path=cache_dir / "init_selected_480.csv",
        valid_ratio=float(predictor_valid_ratio),
        seed=int(predictor_valid_seed),
    )

    init_protocol_df = init_selected_df.merge(predictor_split_df, on="smiles_canonical", how="left")
    init_protocol_df["split_role"] = "init_pool"
    init_protocol_df["init_group"] = "train_valid_init"
    init_protocol_df["downstream_pool"] = "pool_opt"
    init_protocol_df["split"] = init_protocol_df["predictor_split"].astype(str)

    holdout_protocol_df = holdout_selected_df.copy()
    holdout_protocol_df["predictor_split"] = "test"
    holdout_protocol_df["split_role"] = "holdout_test"
    holdout_protocol_df["init_group"] = "holdout_test"
    holdout_protocol_df["downstream_pool"] = "pool_eval"
    holdout_protocol_df["split"] = "test"

    protocol_init_df = pd.concat([init_protocol_df, holdout_protocol_df], ignore_index=True)
    protocol_init_df = protocol_init_df.sort_values(["split_role", "selection_rank", "smiles_canonical"]).reset_index(drop=True)
    protocol_init_df.to_csv(out_root / "init_set.csv", index=False)

    return {
        "canonical_df": canonical_df,
        "scaffold_df": scaffold_df,
        "pool_opt_df": pool_opt_df,
        "pool_eval_df": pool_eval_df,
        "pool_opt_assignments": pool_opt_assignments,
        "pool_eval_assignments": pool_eval_assignments,
        "init_selected_df": init_selected_df,
        "holdout_selected_df": holdout_selected_df,
        "protocol_init_df": protocol_init_df,
        "predictor_split_df": predictor_split_df,
        "cache_dir": cache_dir,
    }



def main() -> None:
    ap = argparse.ArgumentParser(description="Run the main MOBO experiment with a shared generated candidate pool.")
    ap.add_argument("--config", default="config/surrogate/config.yaml")
    ap.add_argument("--generate-config", default="config/generative/config_mobo_generate_transformer_lm.yaml")
    ap.add_argument("--mobo-config", default="config/mobo/main_experiment.yaml")

    ap.add_argument("--init-library-csv", default=None)
    ap.add_argument("--init-library-smiles-col", default=None)
    ap.add_argument("--init-encoder-ckpt", default=None)
    ap.add_argument("--init-training-dataset-root", default=None)
    ap.add_argument("--init-ligand-vocab-file", default=None)
    ap.add_argument("--init-encode-batch-size", type=int, default=None)
    ap.add_argument("--init-encode-workers", type=int, default=None)
    ap.add_argument("--init-kmeans-epochs", type=int, default=None)
    ap.add_argument("--init-kmeans-n-init", type=int, default=None)
    ap.add_argument("--init-library-max-rows", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--init-size", type=int, default=None)
    ap.add_argument("--init-clusters", type=int, default=None)
    ap.add_argument("--init-min-per-cluster", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--score-eps", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    generate_cfg = load_config(args.generate_config) if args.generate_config else {}
    cfg, applied_generate_cfg = _apply_generate_config(cfg, generate_cfg)
    mobo_cfg = load_config(args.mobo_config)
    _validate_mobo_config(mobo_cfg)
    mobo_paths_cfg = _require_dict(mobo_cfg.get("paths"), "mobo.paths")
    mobo_init_cfg = _require_dict(mobo_cfg.get("init_selection"), "mobo.init_selection")
    mobo_experiment_cfg = _require_dict(mobo_cfg.get("experiment"), "mobo.experiment")

    resolved = argparse.Namespace(
        init_library_csv=args.init_library_csv if args.init_library_csv is not None else mobo_paths_cfg.get("init_library_csv"),
        init_library_smiles_col=_resolve_cli_or_config(args.init_library_smiles_col, mobo_paths_cfg.get("init_library_smiles_col"), "paths.init_library_smiles_col"),
        init_encoder_ckpt=args.init_encoder_ckpt if args.init_encoder_ckpt is not None else mobo_paths_cfg.get("init_encoder_ckpt"),
        init_training_dataset_root=_resolve_cli_or_config(args.init_training_dataset_root, mobo_paths_cfg.get("init_training_dataset_root"), "paths.init_training_dataset_root"),
        init_ligand_vocab_file=args.init_ligand_vocab_file if args.init_ligand_vocab_file is not None else mobo_paths_cfg.get("init_ligand_vocab_file"),
        output_dir=_resolve_cli_or_config(args.output_dir, mobo_paths_cfg.get("output_dir"), "paths.output_dir"),
        init_library_max_rows=int(_resolve_cli_or_config(args.init_library_max_rows, mobo_init_cfg.get("library_max_rows", 0), "init_selection.library_max_rows")),
        rounds=int(_resolve_cli_or_config(args.rounds, mobo_experiment_cfg.get("rounds"), "experiment.rounds")),
        score_eps=float(_resolve_cli_or_config(args.score_eps, mobo_experiment_cfg.get("score_eps"), "experiment.score_eps")),
        seed=int(_resolve_cli_or_config(args.seed, mobo_experiment_cfg.get("seed"), "experiment.seed")),
        methods=args.methods if args.methods is not None else mobo_experiment_cfg.get("methods"),
        mobo_config=args.mobo_config,
    )

    general_cfg = _section(cfg, "general")
    dataset_cfg = _section(cfg, "dataset")
    selection_cfg = _section(cfg, "selection")
    surrogate_cfg = _section(cfg, "surrogate")
    model_cfg = _section(cfg, "model")
    objective_cfg = _section(cfg, "objective")
    candidate_cfg = _section(cfg, "candidate")
    oracle_cfg = _section(cfg, "oracle")
    run_cfg = _section(cfg, "run")
    candidate_3d_cfg = dict(_section(candidate_cfg, "candidate_3d"))
    candidate_3d_cfg["seed"] = int(resolved.seed)

    if not bool(objective_cfg.get("use_sa", True)):
        raise RuntimeError("Current main experiment requires SA as the third online objective.")
    extra_objectives = objective_cfg.get("extra_objectives", [])
    if extra_objectives not in (None, [], (), "", "null"):
        raise RuntimeError("Current main experiment only supports dock/QED/SA.")

    method_value = resolved.methods
    if isinstance(method_value, (list, tuple)):
        methods = [str(m).strip() for m in method_value if str(m).strip()]
    else:
        methods = [m.strip() for m in str(method_value).split(",") if m.strip()]
    if not methods:
        raise RuntimeError("--methods must list at least one selection method.")
    allowed_methods = {"random", "qnehvi", "qpmhi", "qpmhi_fallback", "qpmhi_set_hv", "mean_set_hv"}
    invalid_methods = [m for m in methods if m not in allowed_methods]
    if invalid_methods:
        raise RuntimeError(f"Unsupported methods: {invalid_methods}. Allowed methods: {sorted(allowed_methods)}")

    dock_sign = float(objective_cfg.get("dock_sign", -1.0))
    qed_sign = float(objective_cfg.get("qed_sign", 1.0))
    sa_sign = float(objective_cfg.get("sa_sign", -1.0))
    dock_valid_max = objective_cfg.get("dock_valid_max", 0.0)
    dock_valid_max = None if dock_valid_max is None else float(dock_valid_max)
    weights = [float(x) for x in selection_cfg.get("weights", [0.7, 0.1, 0.2])]
    fixed_ref_point = [float(x) for x in objective_cfg.get("ref_point", [0.0, 0.0, -20.0])]
    set_hv_topk_factor = int(selection_cfg.get("set_hv_topk_factor", 3))
    qpmhi_set_hv_gamma = float(selection_cfg.get("qpmhi_set_hv_gamma", 1.0))
    qpmhi_set_hv_gate_eps = float(selection_cfg.get("qpmhi_set_hv_gate_eps", 0.1))
    set_hv_conservative_lambda = float(selection_cfg.get("set_hv_conservative_lambda", 0.5))
    pred_batch_size = int(surrogate_cfg.get("pred_batch_size", surrogate_cfg.get("retrain_batch_size", 48)))
    batch_size = int(selection_cfg.get("batch_size", 20))
    candidate_pool_size = int(candidate_cfg.get("pool_size", 2000))
    train_epochs = int(surrogate_cfg.get("retrain_epochs", 180))
    rounds = int(resolved.rounds)
    backbone = str(model_cfg.get("surrogate_backbone", "tensornet")).lower()

    pretrain_split_ratio = _normalize_ratio_config(
        dataset_cfg.get("pretrain_split_ratio", dataset_cfg.get("split_ratio", [0.8, 0.1, 0.1])),
        default=(0.8, 0.1, 0.1),
        expected_len=3,
    )
    pretrain_split_seed = int(dataset_cfg.get("pretrain_split_seed", dataset_cfg.get("split_seed", resolved.seed)))
    pretrain_use_existing_split = bool(dataset_cfg.get("pretrain_use_existing_split", False))
    downstream_pool_ratio = _normalize_ratio_config(
        dataset_cfg.get("downstream_pool_split_ratio", [0.8, 0.2]),
        default=(0.8, 0.2),
        expected_len=2,
    )
    downstream_pool_seed = int(dataset_cfg.get("downstream_pool_split_seed", resolved.seed))
    scaffold_include_chirality = bool(dataset_cfg.get("scaffold_include_chirality", False))
    predictor_valid_ratio = float(surrogate_cfg.get("predictor_valid_ratio", dataset_cfg.get("predictor_valid_ratio", 0.2)))
    predictor_valid_seed = int(surrogate_cfg.get("predictor_valid_seed", dataset_cfg.get("split_seed", resolved.seed)))
    butina_cutoff = float(mobo_init_cfg.get("butina_cutoff", 0.6))
    fp_radius = int(mobo_init_cfg.get("fp_radius", 2))
    fp_bits = int(mobo_init_cfg.get("fp_bits", 2048))
    init_train_valid_total = int(mobo_init_cfg.get("init_train_valid_total", mobo_init_cfg.get("init_size", 480)))
    holdout_test_total = int(mobo_init_cfg.get("holdout_test_total", 120))
    selection_seed = int(mobo_init_cfg.get("selection_seed", resolved.seed))

    sample_batch = int(candidate_cfg.get("sample_batch", min(500, candidate_pool_size)))
    temperature = float(candidate_cfg.get("temperature", 1.0))
    sample_top_k = int(candidate_cfg.get("top_k", 0))

    device = _device_from_cfg(str(general_cfg.get("device", "auto")))
    out_root = Path(str(resolved.output_dir))
    out_root.mkdir(parents=True, exist_ok=True)
    append_mode = bool(args.append)
    existing_meta = None
    existing_meta_methods: list[str] = []
    if append_mode:
        meta_path = out_root / "experiment_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Append mode requires an existing experiment meta file: {meta_path}")
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(existing_meta, dict):
            raise RuntimeError(f"experiment_meta.json must contain an object: {meta_path}")
        existing_meta_methods = [str(x).strip() for x in existing_meta.get("methods", []) if str(x).strip()]

    oracle_asset_root = _resolve_oracle_asset_root(oracle_cfg)
    default_init_library_candidates = _default_init_library_candidates(
        resolved.init_training_dataset_root,
        oracle_asset_root,
    )
    if append_mode:
        append_source_csv = existing_meta.get("init_library_csv") if existing_meta is not None else None
        if not append_source_csv:
            raise RuntimeError("Append mode requires init_library_csv in experiment_meta.json.")
        init_library_csv = Path(str(append_source_csv)).resolve()
        if not init_library_csv.exists():
            raise FileNotFoundError(f"Append mode requires the original init library source CSV: {init_library_csv}")
        append_smiles_col = existing_meta.get("init_library_smiles_col") if existing_meta is not None else resolved.init_library_smiles_col
        init_library_raw_df, init_library_smiles_col = load_latent_library_csv(
            init_library_csv,
            smiles_col=None if not append_smiles_col else str(append_smiles_col),
        )
        protocol_source_csv_path = init_library_csv
    else:
        if resolved.init_library_csv:
            init_library_csv = Path(str(resolved.init_library_csv)).resolve()
        else:
            init_library_csv = None
            for candidate in default_init_library_candidates:
                if candidate.exists():
                    init_library_csv = candidate.resolve()
                    break
            if init_library_csv is None:
                raise RuntimeError(
                    "Failed to resolve the downstream initialization library. Pass --init-library-csv explicitly."
                )
        init_library_raw_df, init_library_smiles_col = load_latent_library_csv(
            init_library_csv,
            smiles_col=None if not resolved.init_library_smiles_col else str(resolved.init_library_smiles_col),
        )
        if int(resolved.init_library_max_rows) > 0:
            init_library_raw_df = init_library_raw_df.head(int(resolved.init_library_max_rows)).copy().reset_index(drop=True)
        protocol_source_csv_path = init_library_csv

    shared_ligand3d_cache = GLOBAL_LIGAND3D_CACHE_DIR
    shared_ligand3d_cache.mkdir(parents=True, exist_ok=True)
    candidate_3d_cfg["cache_dir"] = str(shared_ligand3d_cache)

    if append_mode:
        for key, current_value in {
            "rounds": rounds,
            "batch_size": batch_size,
            "candidate_pool_size": candidate_pool_size,
            "backbone": backbone,
        }.items():
            existing_value = existing_meta.get(key)
            if existing_value != current_value:
                raise RuntimeError(
                    f"Append mode requires matching experiment settings for {key}: existing={existing_value} current={current_value}"
                )
        existing_ckpt = existing_meta.get("init_encoder_ckpt")
        if not existing_ckpt:
            raise RuntimeError("Append mode requires init_encoder_ckpt in experiment_meta.json.")
        init_encoder_ckpt = Path(str(existing_ckpt)).resolve()
        if not init_encoder_ckpt.exists():
            raise FileNotFoundError(init_encoder_ckpt)
    else:
        init_encoder_ckpt = _resolve_or_train_init_encoder_ckpt(
            args=resolved,
            cfg_path=args.config,
            out_root=out_root,
            init_library_df=init_library_raw_df,
            shared_ligand3d_cache=shared_ligand3d_cache,
            candidate_3d_cfg=candidate_3d_cfg,
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
            source_csv_path=init_library_csv,
            pretrain_split_ratio=pretrain_split_ratio,
            pretrain_split_seed=pretrain_split_seed,
            pretrain_use_existing_split=pretrain_use_existing_split,
        )

    round_encoder_pretrain_cfg = resolve_round_encoder_refresh_config(cfg)
    round_encoder_refresh_epochs = int(round_encoder_pretrain_cfg["epochs"])
    current_encoder_ckpt = init_encoder_ckpt

    protocol = _build_or_load_downstream_protocol(
        out_root=out_root,
        downstream_raw_df=init_library_raw_df,
        smiles_col=init_library_smiles_col,
        source_csv_path=protocol_source_csv_path,
        downstream_pool_ratio=downstream_pool_ratio,
        downstream_pool_seed=downstream_pool_seed,
        scaffold_include_chirality=scaffold_include_chirality,
        butina_cutoff=butina_cutoff,
        fp_radius=fp_radius,
        fp_bits=fp_bits,
        selection_seed=selection_seed,
        init_train_valid_total=init_train_valid_total,
        holdout_test_total=holdout_test_total,
        predictor_valid_ratio=predictor_valid_ratio,
        predictor_valid_seed=predictor_valid_seed,
        append_mode=append_mode,
    )
    init_df = protocol["protocol_init_df"].copy().reset_index(drop=True)
    if append_mode:
        global_vocab = _resolve_append_global_vocab(
            out_root=out_root,
            existing_methods=existing_meta_methods,
            init_df=init_df,
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
        )
        _log(
            f"append init reuse | init_library={init_library_csv} init_set_n={init_df.shape[0]} "
            f"encoder={init_encoder_ckpt.name} vocab_n={len(global_vocab)}"
        )
    else:
        global_vocab = _resolve_init_pretrain_ligand_vocab(
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
            init_library_df=protocol["canonical_df"],
        )
        _log(
            "protocol init prepared | "
            f"init_train_valid_total={int((init_df['split_role'] == 'init_pool').sum())} "
            f"holdout_test_total={int((init_df['split_role'] == 'holdout_test').sum())}"
        )

    predictor_counts = init_df.loc[init_df["split_role"].astype(str) == "init_pool", "split"].value_counts().to_dict()
    _log(
        "predictor random split counts | "
        f"train={int(predictor_counts.get('train', 0))} valid={int(predictor_counts.get('valid', 0))}"
    )
    _log(
        "holdout exclusion confirmed | "
        f"holdout_n={int((init_df['split_role'] == 'holdout_test').sum())} excluded_from_predictor_train_valid=1 excluded_from_active_learning=1"
    )
    proposal_exclude_smiles = set(init_df["smiles_canonical"].astype(str).tolist())
    proposal_exclude_smiles.update(protocol["pool_eval_df"]["smiles_canonical"].astype(str).tolist())

    if append_mode:
        meta_methods = list(existing_meta_methods)
        for method in methods:
            if method not in meta_methods:
                meta_methods.append(method)
    else:
        meta_methods = list(methods)

    meta = {
        "methods": meta_methods,
        "rounds": rounds,
        "batch_size": batch_size,
        "candidate_pool_size": candidate_pool_size,
        "backbone": backbone,
        "init_library_csv": str(init_library_csv),
        "init_library_smiles_col": init_library_smiles_col,
        "init_encoder_ckpt": str(init_encoder_ckpt.resolve()),
        "init_training_dataset_root": resolved.init_training_dataset_root,
        "mobo_config": str(Path(resolved.mobo_config).resolve()),
        "generate_config": str(Path(args.generate_config).resolve()) if args.generate_config else None,
        "generate_settings": applied_generate_cfg,
        "protocol": {
            "pretrain_split_ratio": [float(x) for x in pretrain_split_ratio],
            "pretrain_split_seed": int(pretrain_split_seed),
            "downstream_pool_split_ratio": [float(x) for x in downstream_pool_ratio],
            "downstream_pool_split_seed": int(downstream_pool_seed),
            "scaffold_include_chirality": bool(scaffold_include_chirality),
            "butina_cutoff": float(butina_cutoff),
            "fp_radius": int(fp_radius),
            "fp_bits": int(fp_bits),
            "init_train_valid_total": int(init_train_valid_total),
            "holdout_test_total": int(holdout_test_total),
            "predictor_valid_ratio": float(predictor_valid_ratio),
            "predictor_valid_seed": int(predictor_valid_seed),
        },
    }
    if append_mode and existing_meta is not None and isinstance(existing_meta.get("generator"), dict):
        meta["generator"] = existing_meta["generator"]
    (out_root / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(
        f"main_experiment start | out_root={out_root} methods={methods} rounds={rounds} batch_size={batch_size} "
        f"candidate_pool_size={candidate_pool_size} generate_config={Path(args.generate_config).name if args.generate_config else 'none'} append={append_mode}"
    )

    existing_completed_max = -1
    for method in existing_meta_methods:
        last_row = _load_existing_last_iter_row(out_root / method / "iter_metrics.csv")
        if last_row is not None:
            existing_completed_max = max(existing_completed_max, int(last_row["iter"]))

    existing_summary_rows: list[dict] = []
    for method in existing_meta_methods:
        if method in methods:
            continue
        last_row = _load_existing_last_iter_row(out_root / method / "iter_metrics.csv")
        if last_row is not None:
            existing_summary_rows.append({"method": method, **last_row})

    method_states: dict[str, dict] = {}
    for method in methods:
        method_dir = out_root / method
        dataset_root = method_dir / "dataset"
        method_dir.mkdir(parents=True, exist_ok=True)
        dataset_smiles_path = dataset_root / "smiles.csv"
        if append_mode and method in existing_meta_methods:
            if not dataset_smiles_path.exists():
                raise FileNotFoundError(f"Append mode found a missing dataset for existing method {method}: {dataset_smiles_path}")
            iter_rows, completed_iter = _load_existing_iter_rows(method_dir / "iter_metrics.csv", rounds=rounds)
            holdout_metrics_path = method_dir / "holdout_eval_metrics.csv"
            holdout_rows = pd.read_csv(holdout_metrics_path).to_dict("records") if holdout_metrics_path.exists() else []
            method_states[method] = {
                "method_dir": method_dir,
                "dataset_root": dataset_root,
                "iter_rows": iter_rows,
                "holdout_rows": holdout_rows,
                "completed_iter": completed_iter,
                "oracle_reference_evaluated": True,
                "init_stats": None,
            }
            _log(
                f"append method reuse | method={method} completed_iter={completed_iter} "
                f"dataset={dataset_root} rows={len(iter_rows)}"
            )
            continue
        init_stats = _prepare_init_dataset(
            oracle_asset_root,
            dataset_root,
            schema_df=init_df,
            init_df=init_df,
            oracle_cfg=oracle_cfg,
            objective_cfg=objective_cfg,
            run_cfg=run_cfg,
            global_vocab=global_vocab,
        )
        method_states[method] = {
            "method_dir": method_dir,
            "dataset_root": dataset_root,
            "iter_rows": [],
            "holdout_rows": [],
            "completed_iter": -1,
            "oracle_reference_evaluated": True,
            "init_stats": init_stats,
        }
        _log(
            f"init dataset prepared | method={method} init_n={init_stats.get('init_n', 0)} "
            f"holdout_n={init_stats.get('holdout_n', 0)} valid_init_dock_n={init_stats.get('valid_init_dock_n', 0)} "
            f"init_best_dock={init_stats.get('init_best_dock', float('nan')):.4f} "
            f"init_mean_dock={init_stats.get('init_mean_dock', float('nan')):.4f}"
        )

    generator, generator_info = load_candidate_generator(cfg, device)

    if append_mode:
        existing_generate_settings = existing_meta.get("generate_settings")
        if existing_generate_settings != applied_generate_cfg:
            raise RuntimeError(
                "Append mode requires the same generate settings as the existing experiment: "
                f"existing={existing_generate_settings} current={applied_generate_cfg}"
            )
        existing_generator = existing_meta.get("generator")
        if not isinstance(existing_generator, dict) or not existing_generator.get("generator_kind"):
            existing_generate_kind = _require_dict(
                existing_generate_settings.get("general.generator"),
                "experiment_meta.generate_settings.general.generator",
            ).get("kind")
            existing_generator = {"generator_kind": existing_generate_kind}
        if existing_generator.get("generator_kind") != generator_info.get("generator_kind"):
            raise RuntimeError(
                "Append mode requires the same generator kind as the existing experiment: "
                f"existing={existing_generator.get('generator_kind')} current={generator_info.get('generator_kind')}"
            )
    meta_live = json.loads((out_root / "experiment_meta.json").read_text(encoding="utf-8"))
    meta_live["generator"] = generator_info
    (out_root / "experiment_meta.json").write_text(json.dumps(meta_live, indent=2), encoding="utf-8")

    progress_path = out_root / "progress.jsonl"
    if progress_path.exists() and not append_mode:
        progress_path.unlink()

    append_pool_completed_max = max((int(state["completed_iter"]) for state in method_states.values()), default=-1)

    round_pool_dir = out_root / "round_pools"
    cumulative_candidate_pool = None
    if append_mode and round_pool_dir.exists() and int(append_pool_completed_max) >= 0:
        for prev_iter_idx in range(int(append_pool_completed_max) + 1):
            prev_pool_path = round_pool_dir / f"iter{prev_iter_idx:03d}_candidate_pool.csv"
            if not prev_pool_path.exists():
                raise FileNotFoundError(
                    f"Append mode requires an existing round pool for candidate history preload: {prev_pool_path}"
                )
            cumulative_candidate_pool = _accumulate_candidate_pool_history(
                cumulative_candidate_pool,
                pd.read_csv(prev_pool_path),
            )
        if cumulative_candidate_pool is not None:
            _log(
                f"append candidate history preload | completed_iters={int(append_pool_completed_max) + 1} "
                f"unique_n={cumulative_candidate_pool.shape[0]}"
            )
    for iter_idx in range(int(rounds) + 1):
        pending_methods = [method for method in methods if int(method_states[method]["completed_iter"]) < int(iter_idx)]
        if not pending_methods:
            _log(f"iter {iter_idx}/{rounds} skip | all requested methods already completed")
            continue
        iter_started = time.perf_counter()
        _log(f"iter {iter_idx}/{rounds} start | pending_methods={pending_methods}")
        shared_candidate_pool = None
        if iter_idx < int(rounds):
            round_pool_dir.mkdir(parents=True, exist_ok=True)
            pool_path = round_pool_dir / f"iter{iter_idx:03d}_candidate_pool.csv"
            if append_mode and pool_path.exists():
                shared_candidate_pool = pd.read_csv(pool_path)
                _log(
                    f"iter {iter_idx}/{rounds} reuse_candidate_pool | n={shared_candidate_pool.shape[0]} "
                    f"pool={pool_path.name}"
                )
            else:
                if append_mode and int(iter_idx) <= int(append_pool_completed_max):
                    raise FileNotFoundError(
                        f"Append mode expected an existing round pool for completed iter {iter_idx}: {pool_path}"
                    )
                pool_started = time.perf_counter()
                _log(f"iter {iter_idx}/{rounds} build_candidate_pool start")
                generated_smiles, pool_stats = build_candidate_pool(
                    generator,
                    candidate_pool_size,
                    sample_batch,
                    temperature,
                    sample_top_k,
                    exclude_smiles=proposal_exclude_smiles,
                    log_prefix=f"iter{iter_idx:03d}: ",
                    return_stats=True,
                )
                shared_candidate_pool = _candidate_frame(generated_smiles)
                shared_candidate_pool.to_csv(pool_path, index=False)
                pool_rounds_df = pd.DataFrame(pool_stats.pop("round_rows", []))
                if not pool_rounds_df.empty:
                    pool_rounds_df.to_csv(round_pool_dir / f"iter{iter_idx:03d}_candidate_pool_rounds.csv", index=False)
                pool_stats.update({
                    "iter": int(iter_idx),
                    "candidate_pool_n": int(shared_candidate_pool.shape[0]),
                    "generator_kind": str(generator_info.get("generator_kind", "unknown")),
                })
                (round_pool_dir / f"iter{iter_idx:03d}_candidate_pool_summary.json").write_text(json.dumps(pool_stats, indent=2), encoding="utf-8")
                candidate_pool_metrics_path = out_root / "candidate_pool_metrics.csv"
                if candidate_pool_metrics_path.exists():
                    pool_metrics_df = pd.read_csv(candidate_pool_metrics_path)
                    pool_metrics_df = pd.concat([pool_metrics_df, pd.DataFrame([pool_stats])], ignore_index=True)
                else:
                    pool_metrics_df = pd.DataFrame([pool_stats])
                pool_metrics_df.to_csv(candidate_pool_metrics_path, index=False)
                _log(
                    f"iter {iter_idx}/{rounds} build_candidate_pool done | n={shared_candidate_pool.shape[0]} "
                    f"total_sampled={pool_stats['total_sampled']} valid={pool_stats['valid']} invalid={pool_stats['invalid']} "
                    f"dup={pool_stats['duplicate']} excluded={pool_stats['excluded']} "
                    f"validity_rate={pool_stats['validity_rate']:.4f} elapsed={time.perf_counter() - pool_started:.1f}s"
                )

        if shared_candidate_pool is not None:
            cumulative_candidate_pool = _accumulate_candidate_pool_history(cumulative_candidate_pool, shared_candidate_pool)

        if backbone == "tensornet" and shared_candidate_pool is not None:
            if cumulative_candidate_pool is None or cumulative_candidate_pool.empty:
                raise RuntimeError("Round encoder refresh requires a non-empty cumulative candidate pool.")
            round_encoder_root = out_root / "round_encoder_pretrain_dataset" / f"iter{iter_idx:03d}"
            round_encoder_ckpt = out_root / "round_encoder_pretrain" / f"encoder_iter{iter_idx:03d}.pt"
            if round_encoder_ckpt.exists():
                current_encoder_ckpt = round_encoder_ckpt
                _log(f"iter {iter_idx}/{rounds} reuse_round_encoder | ckpt={round_encoder_ckpt.name}")
            else:
                _prepare_round_encoder_pretrain_dataset(
                    round_encoder_root,
                    init_library_df=init_library_raw_df,
                    candidate_df=cumulative_candidate_pool,
                    split_ratio=pretrain_split_ratio,
                    seed=int(pretrain_split_seed + iter_idx),
                )
                round_encoder_ckpt.parent.mkdir(parents=True, exist_ok=True)
                refresh_started = time.perf_counter()
                _log(
                    f"iter {iter_idx}/{rounds} refresh_encoder start | pool_n={shared_candidate_pool.shape[0]} "
                    f"history_pool_n={cumulative_candidate_pool.shape[0]} "
                    f"epochs={round_encoder_refresh_epochs} init_ckpt={current_encoder_ckpt.name}"
                )
                run_tensornet_encoder_pretrain(
                    config_path=args.config,
                    dataset_root=round_encoder_root,
                    save_path=round_encoder_ckpt,
                    init_encoder_ckpt=current_encoder_ckpt,
                    epochs=int(round_encoder_refresh_epochs),
                    batch_size=int(round_encoder_pretrain_cfg["batch_size"]),
                    lr=float(round_encoder_pretrain_cfg["lr"]),
                    weight_decay=float(round_encoder_pretrain_cfg["weight_decay"]),
                    fp_bits=int(round_encoder_pretrain_cfg["fp_bits"]),
                    fp_radius=int(round_encoder_pretrain_cfg["fp_radius"]),
                    device_arg=str(round_encoder_pretrain_cfg["device"]),
                    ligand3d_cache_dir=str(shared_ligand3d_cache),
                    ligand_vocab_override=global_vocab,
                    ligand3d_num_workers=int(candidate_3d_cfg.get("workers", 8)),
                    ligand3d_mp_chunksize=int(candidate_3d_cfg.get("chunksize", 16)),
                    confgen_seed=int(candidate_3d_cfg.get("seed", resolved.seed)),
                    fp_weight=float(round_encoder_pretrain_cfg["fp_weight"]),
                    prop_weight=float(round_encoder_pretrain_cfg["prop_weight"]),
                    scheduler_name=str(round_encoder_pretrain_cfg["scheduler"]),
                    warmup_epochs=min(int(round_encoder_pretrain_cfg["warmup_epochs"]), int(round_encoder_refresh_epochs)),
                    min_lr=float(round_encoder_pretrain_cfg["min_lr"]),
                    early_stop_patience=min(int(round_encoder_pretrain_cfg["early_stop_patience"]), int(round_encoder_refresh_epochs)),
                    early_stop_min_delta=float(round_encoder_pretrain_cfg["early_stop_min_delta"]),
                )
                current_encoder_ckpt = round_encoder_ckpt
                _log(
                    f"iter {iter_idx}/{rounds} refresh_encoder done | ckpt={round_encoder_ckpt.name} "
                    f"elapsed={time.perf_counter() - refresh_started:.1f}s"
                )

        for method in methods:
            state = method_states[method]
            if int(state["completed_iter"]) >= int(iter_idx):
                _log(f"iter {iter_idx}/{rounds} method {method} skip | already completed")
                continue
            method_started = time.perf_counter()
            method_dir = state["method_dir"]
            dataset_root = state["dataset_root"]
            iter_dir = method_dir / f"iter{iter_idx:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            _log(f"iter {iter_idx}/{rounds} method {method} start")

            labeled_df = _load_observed_df(dataset_root, dock_valid_max=dock_valid_max)
            observed_obj = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
            hv = compute_hypervolume(
                observed_obj,
                ref_point=fixed_ref_point,
            )
            final_nd_count = int(bt_is_non_dominated(observed_obj).sum().item())
            acq_ref_point = _dynamic_ref_point_from_labeled(
                labeled_df,
                dock_sign=dock_sign,
                qed_sign=qed_sign,
                sa_sign=sa_sign,
            )
            iter_row = {
                "iter": iter_idx,
                "labeled_n": int(labeled_df.shape[0]),
                "best_observed_dock": float(labeled_df["dock_score"].min()),
                "hv": float(hv),
                "final_hv": float(hv),
                "final_nd_count": int(final_nd_count),
                "acq_ref_point_dock": float(acq_ref_point[0]),
                "acq_ref_point_qed": float(acq_ref_point[1]),
                "acq_ref_point_sa": float(acq_ref_point[2]),
            }
            if int(iter_idx) == 0:
                init_stats = dict(state.get("init_stats") or {})
                iter_row.update({
                    "init_n": int(init_stats.get("init_n", labeled_df.shape[0])),
                    "holdout_n": int(init_stats.get("holdout_n", 0)),
                    "init_valid_dock_n": int(init_stats.get("valid_init_dock_n", labeled_df.shape[0])),
                    "holdout_valid_dock_n": int(init_stats.get("valid_holdout_dock_n", 0)),
                    "init_best_dock": float(init_stats.get("init_best_dock", labeled_df["dock_score"].min())),
                    "init_mean_dock": float(init_stats.get("init_mean_dock", pd.to_numeric(labeled_df["dock_score"], errors="coerce").mean())),
                })

            clear_processed(str(dataset_root))
            ckpt_path = method_dir / "checkpoints" / f"surrogate_iter{iter_idx:03d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            train_started = time.perf_counter()
            _log(f"iter {iter_idx}/{rounds} method {method} train_surrogate start | labeled_n={labeled_df.shape[0]}")
            train_surrogate_from_scratch(
                root=str(dataset_root),
                save_path=str(ckpt_path),
                epochs=train_epochs,
                batch_size=int(surrogate_cfg.get("retrain_batch_size", 48)),
                lr=float(surrogate_cfg.get("retrain_head_lr", surrogate_cfg.get("retrain_lr", 1e-3))),
                encoder_lr=float(surrogate_cfg.get("retrain_encoder_lr", surrogate_cfg.get("retrain_head_lr", surrogate_cfg.get("retrain_lr", 1e-3)))),
                weight_decay=float(surrogate_cfg.get("retrain_weight_decay", 1e-6)),
                hidden_dim=int(surrogate_cfg.get("retrain_hidden_dim", 48)),
                num_layers=int(surrogate_cfg.get("retrain_num_layers", 3)),
                dropout=float(surrogate_cfg.get("retrain_dropout", 0.0)),
                use_edge_attr=bool(surrogate_cfg.get("retrain_use_edge_attr", True)),
                use_ligand_mask=bool(surrogate_cfg.get("retrain_use_ligand_mask", True)),
                standardize=bool(surrogate_cfg.get("retrain_standardize", False)),
                fp_dim=int(surrogate_cfg.get("retrain_fp_dim", 0)),
                fp_radius=int(surrogate_cfg.get("retrain_fp_radius", 2)),
                eval_samples=int(surrogate_cfg.get("retrain_eval_samples", 8)),
                scheduler=str(surrogate_cfg.get("retrain_scheduler", "cosine")),
                warmup_epochs=int(surrogate_cfg.get("retrain_warmup_epochs", 15)),
                min_lr=float(surrogate_cfg.get("retrain_min_lr", 1e-5)),
                early_stop_patience=int(surrogate_cfg.get("retrain_early_stop_patience", 100)),
                early_stop_min_delta=float(surrogate_cfg.get("retrain_early_stop_min_delta", 0.0)),
                device=device,
                dock_valid_max=dock_valid_max,
                backbone=backbone,
                uncertainty_mode=str(surrogate_cfg.get("retrain_uncertainty_mode", "nig" if backbone == "tensornet" else "gaussian")),
                torchmd_cfg=_section(model_cfg, "torchmd"),
                ensemble_heads=int(surrogate_cfg.get("retrain_ensemble_heads", 1 if backbone == "tensornet" else 1)),
                freeze_backbone=bool(surrogate_cfg.get("retrain_freeze_backbone", False)),
                pretrained_encoder_ckpt=str(current_encoder_ckpt) if backbone == "tensornet" else None,
                ensemble_scheme=str(surrogate_cfg.get("retrain_ensemble_scheme", "full")),
                ensemble_bootstrap=bool(surrogate_cfg.get("retrain_ensemble_bootstrap", True)),
                random_seed=int(resolved.seed) + int(iter_idx),
                ligand3d_cache_dir=str(shared_ligand3d_cache),
                ligand_vocab_override=global_vocab,
                confgen_max_attempts=int(candidate_3d_cfg.get("max_attempts", 3)),
                confgen_seed=int(candidate_3d_cfg.get("seed", resolved.seed)),
                confgen_num_confs=int(candidate_3d_cfg.get("num_confs", 4)),
                confgen_max_opt_iters=int(candidate_3d_cfg.get("max_opt_iters", 100)),
                confgen_optimize=bool(candidate_3d_cfg.get("optimize", True)),
                confgen_prefer_mmff=bool(candidate_3d_cfg.get("prefer_mmff", False)),
                ligand3d_num_workers=int(candidate_3d_cfg.get("workers", 8)),
                ligand3d_mp_chunksize=int(candidate_3d_cfg.get("chunksize", 16)),
                train_log_csv=str(method_dir / f"surrogate_iter{iter_idx:03d}_train_log.csv"),
                train_summary_json=str(method_dir / f"surrogate_iter{iter_idx:03d}_summary.json"),
            )
            model, mean, std, model_node_dim, model_edge_dim, model_fp_dim, model_fp_radius, model_atom_extra_dim, model_bond_extra_dim, _model_pocket_graph, surrogate_kind, surrogate_meta = load_surrogate(str(ckpt_path), device)
            _log(f"iter {iter_idx}/{rounds} method {method} train_surrogate done | elapsed={time.perf_counter() - train_started:.1f}s ckpt={ckpt_path.name}")

            holdout_df = _load_holdout_df(dataset_root, dock_valid_max=dock_valid_max)
            holdout_eval = _evaluate_holdout_surrogate(
                holdout_df=holdout_df,
                model=model,
                mean=mean,
                std=std,
                device=device,
                global_vocab=global_vocab,
                surrogate_kind=surrogate_kind,
                model_node_dim=model_node_dim,
                model_edge_dim=model_edge_dim,
                model_atom_extra_dim=model_atom_extra_dim,
                model_bond_extra_dim=model_bond_extra_dim,
                model_fp_dim=model_fp_dim,
                model_fp_radius=model_fp_radius,
                surrogate_meta=surrogate_meta,
                pred_batch_size=pred_batch_size,
                candidate_3d_cfg=candidate_3d_cfg,
                iter_idx=int(iter_idx),
                method_dir=method_dir,
            )
            state["holdout_rows"] = [row for row in state.get("holdout_rows", []) if int(row.get("iter", -1)) != int(iter_idx)]
            state["holdout_rows"].append(holdout_eval)
            state["holdout_rows"] = sorted(state["holdout_rows"], key=lambda row: int(row["iter"]))
            pd.DataFrame(state["holdout_rows"]).to_csv(method_dir / "holdout_eval_metrics.csv", index=False)
            iter_row.update({
                "holdout_pred_n": int(holdout_eval.get("n", 0)),
                "holdout_pred_rmse": float(holdout_eval.get("rmse", np.nan)),
                "holdout_pred_mae": float(holdout_eval.get("mae", np.nan)),
                "holdout_pred_spearman": float(holdout_eval.get("spearman", np.nan)),
                "holdout_pred_kendall": float(holdout_eval.get("kendall", np.nan)),
                "holdout_pred_calibration_gap": float(holdout_eval.get("calibration_gap", np.nan)),
                "holdout_pred_hit_at_5": float(holdout_eval.get("hit_at_5", np.nan)),
                "holdout_pred_top5_true_mean": float(holdout_eval.get("top5_true_mean", np.nan)),
                "holdout_hit_at_5_last": float(holdout_eval.get("hit_at_5", np.nan)),
                "holdout_top5_oracle_mean_last": float(holdout_eval.get("top5_true_mean", np.nan)),
            })
            _log(
                f"iter {iter_idx}/{rounds} method {method} holdout_eval | "
                f"n={holdout_eval.get('n', 0)} spearman={float(holdout_eval.get('spearman', np.nan)):.4f} "
                f"rmse={float(holdout_eval.get('rmse', np.nan)):.4f} mae={float(holdout_eval.get('mae', np.nan)):.4f}"
            )

            if iter_idx < int(rounds):
                if shared_candidate_pool is None:
                    raise RuntimeError(f"Missing shared candidate pool at iter {iter_idx}.")
                candidate_df = shared_candidate_pool.loc[
                    ~shared_candidate_pool["smiles_canonical"].isin(labeled_df["smiles_canonical"])
                ].copy().reset_index(drop=True)
                if candidate_df.empty:
                    raise RuntimeError(f"No candidate molecules available at iter {iter_idx} for method {method}.")
                select_started = time.perf_counter()
                _log(
                    f"iter {iter_idx}/{rounds} method {method} score_candidates start | "
                    f"candidate_n={candidate_df.shape[0]} "
                    f"acq_ref_point={[round(float(x), 4) for x in acq_ref_point]}"
                )
                if method == "qnehvi":
                    scored = _score_with_qnehvi(
                        candidate_df,
                        labeled_df,
                        model,
                        mean,
                        std,
                        device,
                        global_vocab,
                        surrogate_kind,
                        model_node_dim,
                        model_edge_dim,
                        model_atom_extra_dim,
                        model_bond_extra_dim,
                        model_fp_dim,
                        model_fp_radius,
                        surrogate_meta,
                        pred_batch_size,
                        candidate_3d_cfg,
                        dock_sign,
                        qed_sign,
                        sa_sign,
                        weights,
                        acq_ref_point,
                    )
                elif method in {"qpmhi", "qpmhi_fallback", "qpmhi_set_hv"}:
                    scored = _score_with_qpmhi_analytic(
                        candidate_df,
                        labeled_df=labeled_df,
                        model=model,
                        mean=mean,
                        std=std,
                        device=device,
                        ligand_vocab=global_vocab,
                        surrogate_kind=surrogate_kind,
                        model_node_dim=model_node_dim,
                        model_edge_dim=model_edge_dim,
                        model_atom_extra_dim=model_atom_extra_dim,
                        model_bond_extra_dim=model_bond_extra_dim,
                        model_fp_dim=model_fp_dim,
                        model_fp_radius=model_fp_radius,
                        surrogate_meta=surrogate_meta,
                        pred_batch_size=pred_batch_size,
                        candidate_3d_cfg=candidate_3d_cfg,
                        dock_sign=dock_sign,
                        qed_sign=qed_sign,
                        sa_sign=sa_sign,
                        weights=weights,
                        ref_point=acq_ref_point,
                    )
                elif method in {"random", "mean_set_hv"}:
                    scored = _score_with_predictions_only(
                        candidate_df,
                        model=model,
                        mean=mean,
                        std=std,
                        device=device,
                        ligand_vocab=global_vocab,
                        surrogate_kind=surrogate_kind,
                        model_node_dim=model_node_dim,
                        model_edge_dim=model_edge_dim,
                        model_atom_extra_dim=model_atom_extra_dim,
                        model_bond_extra_dim=model_bond_extra_dim,
                        model_fp_dim=model_fp_dim,
                        model_fp_radius=model_fp_radius,
                        surrogate_meta=surrogate_meta,
                        pred_batch_size=pred_batch_size,
                        candidate_3d_cfg=candidate_3d_cfg,
                    )
                else:
                    raise RuntimeError(f"Unsupported method in scoring: {method}")
                scored.to_csv(iter_dir / "candidate_scores.csv", index=False)
                _log(f"iter {iter_idx}/{rounds} method {method} score_candidates done | scored_n={scored.shape[0]} elapsed={time.perf_counter() - select_started:.1f}s")
                select_batch_started = time.perf_counter()
                selected, fallback_n = _select_batch_main(
                    method,
                    scored,
                    labeled_df=labeled_df,
                    batch_size=batch_size,
                    score_eps=float(resolved.score_eps),
                    seed=_method_seed(resolved.seed, method, iter_idx),
                    dock_sign=dock_sign,
                    qed_sign=qed_sign,
                    sa_sign=sa_sign,
                    ref_point=acq_ref_point,
                    set_hv_topk_factor=set_hv_topk_factor,
                    qpmhi_set_hv_gamma=qpmhi_set_hv_gamma,
                    qpmhi_set_hv_gate_eps=qpmhi_set_hv_gate_eps,
                    set_hv_conservative_lambda=set_hv_conservative_lambda,
                )
                selected = selected.copy().reset_index(drop=True)
                selected["selected_iter"] = iter_idx + 1
                candidate_uncertainty_stats = _uncertainty_summary(scored, prefix="candidate")
                selected_uncertainty_stats = _uncertainty_summary(selected, prefix="selected")
                selected.to_csv(iter_dir / "selected_batch.csv", index=False)
                _log(
                    f"iter {iter_idx}/{rounds} method {method} select_batch done | "
                    f"selected_n={selected.shape[0]} fallback_n={fallback_n} "
                    f"selected_total_std_mean={selected_uncertainty_stats['selected_pred_std_total_mean']:.4f} "
                    f"selected_epi_std_mean={selected_uncertainty_stats['selected_pred_std_epi_mean']:.4f} "
                    f"selected_ale_std_mean={selected_uncertainty_stats['selected_pred_std_ale_mean']:.4f} "
                    f"selected_epi_var_frac_mean={selected_uncertainty_stats['selected_pred_var_epi_frac_mean']:.4f} "
                    f"elapsed={time.perf_counter() - select_batch_started:.1f}s"
                )

                selected_smiles = selected["smiles_canonical"].astype(str).tolist()
                selected_pred_added = [None if pd.isna(v) else float(v) for v in selected["pred_dock_mean"].tolist()]
                added, new_ids, kept_indices = append_candidates_to_smiles_csv(
                    str(dataset_root / "smiles.csv"),
                    selected_smiles,
                    id_prefix=str(oracle_cfg.get("oracle_id_prefix", "GEN")),
                    sa_clamp_min=float(objective_cfg.get("sa_clamp_min", -10.0)),
                    sa_clamp_max=float(objective_cfg.get("sa_clamp_max", 20.0)),
                    split_ratio=None,
                    added_iter=iter_idx + 1,
                    molecule_origin="generated",
                )
                if added != len(new_ids) or added != len(kept_indices):
                    raise RuntimeError(f"Append bookkeeping mismatch at iter {iter_idx} for method {method}: added={added}, new_ids={len(new_ids)}, kept={kept_indices}")
                if kept_indices != list(range(selected.shape[0])):
                    dropped = selected.shape[0] - len(kept_indices)
                    _log(
                        f"iter {iter_idx}/{rounds} method {method} append dedup | "
                        f"selected={selected.shape[0]} kept={len(kept_indices)} dropped={dropped}"
                    )
                    selected = selected.iloc[kept_indices].copy().reset_index(drop=True)
                    selected_pred_added = [selected_pred_added[i] for i in kept_indices]

                oracle_started = time.perf_counter()
                oracle_kwargs = _build_oracle_call_kwargs(
                    run_cfg=run_cfg,
                    oracle_cfg=oracle_cfg,
                    evaluate_reference=not bool(state["oracle_reference_evaluated"]),
                    dock_valid_max=dock_valid_max,
                )

                _log(
                    f"iter {iter_idx}/{rounds} method {method} oracle start | "
                    f"backend={oracle_kwargs['docking_backend']} "
                    f"new_ids={len(new_ids)} "
                    f"vina_executable={oracle_kwargs['vina_executable']}"
                )
                _release_cuda_cache(f"before_iter_oracle:{method}:iter{iter_idx:03d}")

                dock_stats = run_oracle_docking(
                    str(dataset_root),
                    new_ids,
                    **oracle_kwargs,
                )

                state["oracle_reference_evaluated"] = True

                _log(
                    f"iter {iter_idx}/{rounds} method {method} oracle done | "
                    f"backend={oracle_kwargs['docking_backend']} "
                    f"elapsed={time.perf_counter() - oracle_started:.1f}s "
                    f"attempted={dock_stats.get('attempted', 0)} "
                    f"docked={dock_stats.get('docked', 0)} "
                    f"cache_hit_conformer={dock_stats.get('cache_hit_conformer', 0)} "
                    f"cache_hit_docking={dock_stats.get('cache_hit_docking', 0)} "
                    f"cache_hit_failure={dock_stats.get('cache_hit_failure', 0)} "
                    f"failed={dock_stats.get('failed', 0)}"
                )
                _fill_extra_metrics_in_csv(dataset_root, objective_cfg, prior_state=None, force=False)
                new_train_n, new_valid_n = _assign_new_rows_random_predictor_split(
                    dataset_root,
                    new_ids,
                    valid_ratio=float(predictor_valid_ratio),
                    seed=int(predictor_valid_seed + iter_idx + 1),
                )
                _log(
                    f"iter {iter_idx}/{rounds} method {method} predictor_split_new_rows | "
                    f"train={new_train_n} valid={new_valid_n} holdout_excluded=1"
                )

                _export_selected_batch_with_oracle(
                    dataset_root,
                    selected,
                    new_ids,
                    iter_dir / "selected_batch_with_oracle.csv",
                )

                smiles_all = pd.read_csv(dataset_root / "smiles.csv")
                smiles_all["ligand_id"] = smiles_all["ligand_id"].astype(str)
                new_df = smiles_all.loc[smiles_all["ligand_id"].isin([str(x) for x in new_ids])].copy().reset_index(drop=True)

                valid_mask = []
                for val in new_df.get("dock_score", pd.Series([], dtype=float)).tolist():
                    try:
                        num = float(val)
                    except Exception:
                        num = None
                    valid_mask.append(_is_valid_dock_score(num, dock_valid_max=dock_valid_max))
                valid_new_df = new_df.loc[np.asarray(valid_mask, dtype=bool)].copy().reset_index(drop=True) if len(valid_mask) else new_df.head(0).copy()

                oracle_eval = evaluate_oracle_accuracy_from_csv(
                    str(dataset_root / "smiles.csv"),
                    new_ids,
                    selected_pred_added,
                    dock_valid_max=dock_valid_max,
                )

                selected_qed_mean = float(pd.to_numeric(new_df["qed"], errors="coerce").mean()) if not new_df.empty else np.nan
                selected_sa_mean = float(pd.to_numeric(new_df["sa_score"], errors="coerce").mean()) if not new_df.empty else np.nan
                if valid_new_df.empty:
                    selected_true_dock_mean = np.nan
                    selected_true_dock_best = np.nan
                    selected_oracle_hv_gain = 0.0
                    best_log = "nan"
                else:
                    dock_vals = pd.to_numeric(valid_new_df["dock_score"], errors="coerce")
                    selected_true_dock_mean = float(dock_vals.mean())
                    selected_true_dock_best = float(dock_vals.min())
                    selected_oracle_hv_gain = float(_batch_hv_gain(labeled_df, valid_new_df, dock_sign, qed_sign, sa_sign, fixed_ref_point))
                    best_log = f"{selected_true_dock_best:.4f}"

                iter_row.update({
                    "candidate_pool_n": int(candidate_df.shape[0]),
                    "selected_n": int(selected.shape[0]),
                    "selected_fallback_n": int(fallback_n),
                    "selected_true_dock_mean": selected_true_dock_mean,
                    "selected_true_dock_best": selected_true_dock_best,
                    "selected_qed_mean": selected_qed_mean,
                    "selected_sa_mean": selected_sa_mean,
                    "selected_oracle_hv_gain": selected_oracle_hv_gain,
                    "oracle_attempted": int(dock_stats.get("attempted", 0)),
                    "oracle_docked": int(dock_stats.get("docked", 0)),
                    "oracle_failed": int(dock_stats.get("failed", 0)),
                    "selected_pred_n": int(oracle_eval.get("matched", 0)),
                    "selected_pred_skipped": int(oracle_eval.get("skipped", 0)),
                    "selected_pred_invalid_dock": int(oracle_eval.get("invalid_dock", 0)),
                    "selected_pred_invalid_pred": int(oracle_eval.get("invalid_pred", 0)),
                    "selected_pred_rmse": float(oracle_eval.get("rmse", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
                    "selected_pred_mae": float(oracle_eval.get("mae", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
                    "selected_pred_spearman": float(oracle_eval.get("spearman", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
                    "selected_pred_kendall": float(oracle_eval.get("kendall", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
                    **candidate_uncertainty_stats,
                    **selected_uncertainty_stats,
                })
                _log(f"iter {iter_idx}/{rounds} method {method} oracle done | docked={iter_row['oracle_docked']} failed={iter_row['oracle_failed']} best={best_log} elapsed={time.perf_counter() - oracle_started:.1f}s")

            state["iter_rows"].append(iter_row)
            state["completed_iter"] = int(iter_idx)
            pd.DataFrame(state["iter_rows"]).to_csv(method_dir / "iter_metrics.csv", index=False)
            summary_rows = list(existing_summary_rows)
            summary_rows.extend(
                [{"method": name, **sub_state["iter_rows"][-1]} for name, sub_state in method_states.items() if sub_state["iter_rows"]]
            )
            pd.DataFrame(summary_rows).to_csv(out_root / "method_summary.csv", index=False)
            _append_jsonl(progress_path, {"iter": int(iter_idx), "method": method, **iter_row})
            _log(f"iter {iter_idx}/{rounds} method {method} done | hv={iter_row['hv']:.4f} best_observed_dock={iter_row['best_observed_dock']:.4f} elapsed={time.perf_counter() - method_started:.1f}s")

        _log(f"iter {iter_idx}/{rounds} done | elapsed={time.perf_counter() - iter_started:.1f}s")


if __name__ == "__main__":
    main()

