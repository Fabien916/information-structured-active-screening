#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from rdkit import DataStructs
from scipy.cluster import vq as scipy_vq
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data.ligand_only_3d_dataset import GLOBAL_LIGAND3D_CACHE_DIR, LigandOnly3DStore
from mobo.analytic_hvi_fast import nehvi_gaussian_analytic_3d, qphv_prob_gaussian_analytic_3d
from mobo.config_utils import load_config
from mobo.init_selection import allocate_cluster_quota, select_cluster_members, select_fingerprint_init_set
from mobo.io_utils import _load_ligand_vocab_override, _scan_ligand_vocab
from mobo.metrics import qnehvi_scores_from_samples_nd, qphv_prob_scaled_nd_botorch, qphv_topk_prob_nd_botorch
from mobo.retrospective import build_objective_tensor, compute_hypervolume, write_round_dataset
from mobo.smiles_utils import compute_qed, compute_sa
from mobo.surrogate import load_surrogate, train_surrogate_from_scratch
from run_mobo_main_experiment import (
    _compute_morgan_fingerprints,
    _score_with_predictions_only,
    _score_with_qnehvi,
    _score_with_qpmhi_analytic,
)
from run_retrospective_active_screening import _prepare_loader

BENCHMARK_SCRIPT = REPO / "scripts" / "benchmark_initial_surrogate_ssl.py"
DEFAULT_OUT = Path("runs") / "dockstring_benchmark"
DEFAULT_DOCKSTRING_ROOT = Path("data") / "dockstring"
DEFAULT_DOCKSTRING_TARGETS = ("PARP1", "F2", "ESR2", "KIT", "JAK2")
DEFAULT_FP_RADIUS = 2
DEFAULT_FP_BITS = 2048
VIRTUAL_METHODS = {"random", "greedy_mean", "qnehvi_mc", "qpmhi_mc", "qnparego_mc", "analytic_ehvi", "analytic_pomhi"}
RANKING_METHODS = VIRTUAL_METHODS | {"pio"}


def _resolve_config_path_from_main(main_config_path: Path, raw_path: str | Path) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Config file not found: {candidate}")
        return candidate.resolve()
    for base in (main_config_path.parent, REPO):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Config file not found: {candidate}")


def _normalize_trial_spec(name: str, cfg_path: Path, raw_cfg: dict) -> dict:
    trial_name = str(raw_cfg.get("trial_name") or name).strip()
    if not trial_name:
        raise RuntimeError(f"Trial config {cfg_path} has empty trial_name")
    backbone = str(raw_cfg.get("backbone") or "").strip().lower()
    if backbone not in {"gin", "tensornet", "gp"}:
        raise RuntimeError(f"Trial {trial_name} has unsupported backbone '{backbone}' in {cfg_path}")
    uncertainty_mode = str(raw_cfg.get("uncertainty_mode") or "").strip().lower()
    if uncertainty_mode not in {"gaussian", "nig", "bayes"}:
        raise RuntimeError(f"Trial {trial_name} has unsupported uncertainty_mode '{uncertainty_mode}' in {cfg_path}")
    if backbone == "gp" and uncertainty_mode != "gaussian":
        raise RuntimeError(f"GP trial {trial_name} must use gaussian uncertainty_mode in {cfg_path}")
    model_cfg = dict(raw_cfg.get("model") or {})
    train_cfg = dict(raw_cfg.get("train") or {})
    hidden_dim = model_cfg.get("hidden_dim")
    num_layers = model_cfg.get("num_layers")
    dropout = model_cfg.get("dropout")
    if hidden_dim is None or num_layers is None or dropout is None:
        raise RuntimeError(f"Trial {trial_name} in {cfg_path} is missing model.hidden_dim/num_layers/dropout")
    epochs = train_cfg.get("epochs")
    batch_size = train_cfg.get("batch_size")
    lr = train_cfg.get("lr")
    weight_decay = train_cfg.get("weight_decay")
    if epochs is None or batch_size is None or lr is None or weight_decay is None:
        raise RuntimeError(f"Trial {trial_name} in {cfg_path} is missing train.epochs/batch_size/lr/weight_decay")
    trial = {
        "trial_name": trial_name,
        "trial_family": str(raw_cfg.get("trial_family") or name),
        "trial_config_path": str(cfg_path.resolve()),
        "backbone": backbone,
        "uncertainty_mode": uncertainty_mode,
        "needs_pretrain": bool(raw_cfg.get("needs_pretrain", False)),
        "freeze_backbone": bool(raw_cfg.get("freeze_backbone", False)),
        "ensemble_heads": int(raw_cfg.get("ensemble_heads", 1)),
        "ensemble_scheme": str(raw_cfg.get("ensemble_scheme", "full")),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "encoder_lr": None if train_cfg.get("encoder_lr") in (None, "", "null") else float(train_cfg.get("encoder_lr")),
        "weight_decay": float(weight_decay),
        "torchmd_cfg": None,
        "gp_fp_bits": None,
        "gp_fp_radius": None,
    }
    if backbone == "tensornet":
        torchmd_cfg = dict(model_cfg.get("torchmd") or {})
        if not torchmd_cfg:
            raise RuntimeError(f"Trial {trial_name} in {cfg_path} is missing model.torchmd")
        trial["torchmd_cfg"] = torchmd_cfg
    if backbone == "gp":
        fp_bits = model_cfg.get("fp_bits")
        fp_radius = model_cfg.get("fp_radius")
        if fp_bits is None or fp_radius is None:
            raise RuntimeError(f"GP trial {trial_name} in {cfg_path} is missing model.fp_bits/fp_radius")
        trial["gp_fp_bits"] = int(fp_bits)
        trial["gp_fp_radius"] = int(fp_radius)
    return trial


def _resolve_optional_path(raw: str | Path) -> Path | None:
    if raw in (None, "", "null"):
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else (REPO / path).resolve()


def _seed_virtual_shared_inputs(target_root: Path, source_target_root: Path, rounds: int) -> None:
    if not source_target_root.exists():
        raise FileNotFoundError(source_target_root)
    src_pool = source_target_root / "pool_with_properties.csv"
    if src_pool.exists() and not (target_root / "pool_with_properties.csv").exists():
        shutil.copy2(src_pool, target_root / "pool_with_properties.csv")
    for name in ["init_set.csv", "init_assignments.csv", "init_quotas.csv"]:
        src = source_target_root / name
        if src.exists() and not (target_root / name).exists():
            shutil.copy2(src, target_root / name)
    source_shared = source_target_root / "_shared"
    if not source_shared.exists():
        raise FileNotFoundError(source_shared)
    shared_root = target_root / "_shared"
    shared_root.mkdir(parents=True, exist_ok=True)
    shared_init = source_shared / "init_set.csv"
    if shared_init.exists() and not (shared_root / "init_set.csv").exists():
        shutil.copy2(shared_init, shared_root / "init_set.csv")
    for round_idx in range(1, int(rounds) + 1):
        src = source_shared / f"round{round_idx:02d}_candidate_pool.csv"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = shared_root / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def _load_reused_virtual_ligand_vocab(source_target_root: Path) -> list[str] | None:
    for method in ["analytic_ehvi", "analytic_pomhi"]:
        vocab_path = source_target_root / method / "round01_dataset" / "ligand_vocab.json"
        if vocab_path.exists():
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            if isinstance(vocab, list) and vocab:
                return [str(x) for x in vocab]
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_targets(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise RuntimeError(f"Expected NAME=CSV_PATH, got: {raw}")
        name, path_raw = raw.split("=", 1)
        path = Path(path_raw.strip())
        if not path.is_absolute():
            path = (REPO / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        out[name.strip()] = path
    if not out:
        raise RuntimeError("No targets configured.")
    return out


def resolve_repo_path(raw: str | Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = (REPO / path).resolve()
    return path


def resolve_dockstring_root(raw: str | Path) -> Path:
    root = resolve_repo_path(raw)
    if not root.exists():
        raise FileNotFoundError(root)
    required = [root / "dockstring-dataset.tsv", root / "cluster_split.tsv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing local DOCKSTRING files under {root}: {missing}")
    return root


@lru_cache(maxsize=4)
def load_dockstring_tables(root_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_str)
    dataset_df = pd.read_csv(root / "dockstring-dataset.tsv", sep="\t")
    split_df = pd.read_csv(root / "cluster_split.tsv", sep="\t")
    return dataset_df, split_df


def materialize_dockstring_target_csv(root: Path, target_name: str, split_name: str) -> Path:
    split_name = str(split_name).strip().lower()
    if split_name not in {"train", "test", "full"}:
        raise RuntimeError(f"Unsupported DOCKSTRING split: {split_name}")
    out_path = root / "prepared" / split_name / f"{target_name}.csv"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    dataset_df, split_df = load_dockstring_tables(str(root.resolve()))
    if target_name not in dataset_df.columns:
        raise RuntimeError(f"Target {target_name} not found in {root / 'dockstring-dataset.tsv'}")
    required_dataset_cols = ["inchikey", "smiles", target_name]
    missing_dataset_cols = [col for col in required_dataset_cols if col not in dataset_df.columns]
    if missing_dataset_cols:
        raise RuntimeError(f"DOCKSTRING dataset is missing required columns: {missing_dataset_cols}")
    required_split_cols = ["inchikey", "split"]
    missing_split_cols = [col for col in required_split_cols if col not in split_df.columns]
    if missing_split_cols:
        raise RuntimeError(f"DOCKSTRING cluster split file is missing required columns: {missing_split_cols}")
    merged = dataset_df[required_dataset_cols].merge(
        split_df[required_split_cols],
        on="inchikey",
        how="inner",
        validate="one_to_one",
    )
    if split_name == "full":
        work = merged.copy()
    else:
        work = merged.loc[merged["split"].astype(str).str.strip().str.lower() == split_name].copy()
    if work.empty:
        raise RuntimeError(f"Resolved empty DOCKSTRING split for target={target_name}, split={split_name}")
    work = work.rename(columns={"smiles": "smiles_canonical", target_name: "dock_score"})
    work["smiles_canonical"] = work["smiles_canonical"].astype(str).str.strip()
    work["dock_score"] = pd.to_numeric(work["dock_score"], errors="coerce")
    work = work.loc[work["dock_score"].notna()].copy().reset_index(drop=True)
    if work["smiles_canonical"].eq("").any():
        raise RuntimeError(f"DOCKSTRING {target_name} {split_name} split contains empty smiles.")
    if work.empty:
        raise RuntimeError(f"DOCKSTRING {target_name} {split_name} split has no valid docking scores after filtering.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(out_path, index=False)
    return out_path


def resolve_targets(items: list[str], dockstring_root: str, default_split: str) -> dict[str, Path]:
    if items:
        if all("=" in raw for raw in items):
            return parse_targets(items)
        if any("=" in raw for raw in items):
            raise RuntimeError("Mixed target specifications are not allowed: use either NAME=CSV_PATH for all targets or plain target names for all targets.")
        root = resolve_dockstring_root(dockstring_root)
        out: dict[str, Path] = {}
        for target_name in items:
            out[str(target_name).strip()] = materialize_dockstring_target_csv(root, str(target_name).strip(), default_split)
        return out
    root = resolve_dockstring_root(dockstring_root)
    out: dict[str, Path] = {}
    for target_name in DEFAULT_DOCKSTRING_TARGETS:
        out[target_name] = materialize_dockstring_target_csv(root, target_name, default_split)
    return out


def resolve_surrogate_target_pairs(items: list[str], dockstring_root: str) -> dict[str, tuple[Path, Path]]:
    root = resolve_dockstring_root(dockstring_root)
    train_targets = resolve_targets(items, dockstring_root, "train")
    out: dict[str, tuple[Path, Path]] = {}
    for target_name, train_csv in train_targets.items():
        test_csv = materialize_dockstring_target_csv(root, target_name, "test")
        out[target_name] = (train_csv, test_csv)
    return out


def parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise RuntimeError("Expected a non-empty integer list.")
    return vals


def _dir_has_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _csv_exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _check_surrogate_run_dir(out_dir: Path) -> str:
    if not out_dir.exists() or not _dir_has_contents(out_dir):
        return "missing"
    if _csv_exists_and_nonempty(out_dir / "trial_summary.csv") and (out_dir / "run_meta.json").exists():
        return "complete"
    return "incomplete"


def _resolve_requested_surrogate_trial_names(args: argparse.Namespace) -> list[str]:
    config_path = resolve_repo_path(args.config)
    cfg = load_config(str(config_path))
    surrogate_cfg = dict(cfg.get("surrogate") or {})
    model_map = dict(surrogate_cfg.get("benchmark_initial_models") or {})
    if not model_map:
        raise RuntimeError("surrogate.benchmark_initial_models is required and cannot be empty.")
    trial_names: list[str] = []
    seen: set[str] = set()
    for name, raw_path in model_map.items():
        cfg_path = _resolve_config_path_from_main(config_path, raw_path)
        trial = _normalize_trial_spec(str(name), cfg_path, load_config(str(cfg_path)))
        trial_name = str(trial["trial_name"])
        if trial_name in seen:
            raise RuntimeError(f"Duplicate trial_name '{trial_name}' in benchmark model configs")
        seen.add(trial_name)
        if args.skip_gin_small and trial_name == "gin_small_gaussian":
            continue
        if args.skip_scratch and "_scratch_" in trial_name:
            continue
        if args.skip_pretrained and bool(trial["needs_pretrain"]) and int(trial["ensemble_heads"]) == 1:
            continue
        if args.skip_ensemble and int(trial["ensemble_heads"]) > 1:
            continue
        trial_names.append(trial_name)
    if not trial_names:
        raise RuntimeError("No surrogate trials requested.")
    return trial_names


def _missing_requested_surrogate_trials(out_dir: Path, requested_trial_names: list[str]) -> list[str]:
    summary_path = out_dir / "trial_summary.csv"
    if not _csv_exists_and_nonempty(summary_path):
        return list(requested_trial_names)
    summary = pd.read_csv(summary_path)
    existing = {str(x).strip() for x in summary.get("trial_name", pd.Series(dtype=str)).tolist() if str(x).strip()}
    return [name for name in requested_trial_names if name not in existing]


def _check_ranking_run_dir(run_out: Path) -> str:
    if not run_out.exists() or not _dir_has_contents(run_out):
        return "missing"
    if _csv_exists_and_nonempty(run_out / "ranking_summary.csv") and _csv_exists_and_nonempty(run_out / "frozen_pool_scores.csv"):
        return "complete"
    return "incomplete"


def _check_virtual_run_dir(run_out: Path, expected_rounds: int) -> tuple[str, pd.DataFrame | None]:
    if not run_out.exists() or not _dir_has_contents(run_out):
        return "missing", None
    traj_path = run_out / "trajectory.csv"
    if not _csv_exists_and_nonempty(traj_path):
        return "incomplete", None
    try:
        traj = pd.read_csv(traj_path)
    except pd.errors.EmptyDataError:
        return "incomplete", None
    if traj.empty:
        return "incomplete", None
    max_round = int(pd.to_numeric(traj["round"], errors="raise").max())
    last_unlabeled = int(pd.to_numeric(traj.iloc[-1]["unlabeled_n"], errors="raise"))
    if max_round >= int(expected_rounds) or last_unlabeled == 0:
        return "complete", traj
    return "incomplete", traj


def _completed_virtual_rounds(run_out: Path, traj: pd.DataFrame, expected_rounds: int) -> int:
    if "round" not in traj.columns:
        raise RuntimeError(f"Cannot resume virtual-loop method without round column: {run_out}")
    rounds = pd.to_numeric(traj["round"], errors="raise").astype(int).tolist()
    if not rounds:
        raise RuntimeError(f"Cannot resume empty virtual-loop trajectory: {run_out}")
    completed_rounds = int(max(rounds))
    if completed_rounds >= int(expected_rounds):
        return int(expected_rounds)
    expected = list(range(1, completed_rounds + 1))
    if rounds != expected:
        raise RuntimeError(f"Cannot resume non-contiguous virtual-loop trajectory in {run_out}: rounds={rounds}")
    for round_idx in expected:
        selected_path = run_out / f"round{round_idx:02d}_selected.csv"
        if not _csv_exists_and_nonempty(selected_path):
            raise RuntimeError(f"Cannot resume {run_out}; missing completed selected batch: {selected_path}")
    return completed_rounds


def _cleanup_virtual_round_artifacts(run_out: Path, *, from_round: int, expected_rounds: int) -> None:
    if not run_out.exists():
        return
    pat = re.compile(r"^round(\d+)_")
    for child in list(run_out.iterdir()):
        match = pat.match(child.name)
        if match is None:
            continue
        round_idx = int(match.group(1))
        if int(from_round) <= round_idx <= int(expected_rounds):
            resolved = _ensure_within_root(child, run_out)
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()


def _load_virtual_method_state_from_selected(init_df: pd.DataFrame, run_out: Path, completed_rounds: int, existing_traj: pd.DataFrame) -> pd.DataFrame:
    pieces = [init_df.copy().reset_index(drop=True)]
    for round_idx in range(1, int(completed_rounds) + 1):
        selected_path = run_out / f"round{round_idx:02d}_selected.csv"
        selected = ensure_pool(pd.read_csv(selected_path), f"{run_out.name}_round_{round_idx:02d}_selected")
        selected = selected.copy().reset_index(drop=True)
        selected["added_iter"] = int(round_idx)
        pieces.append(selected)
    state = pd.concat(pieces, ignore_index=True)
    expected_labeled = int(pd.to_numeric(existing_traj.iloc[int(completed_rounds) - 1]["labeled_n"], errors="raise"))
    if int(state.shape[0]) != expected_labeled:
        raise RuntimeError(
            f"Cannot resume {run_out}; reconstructed labeled_n={state.shape[0]} "
            f"but trajectory reports labeled_n={expected_labeled}"
        )
    return state


def _ensure_within_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refuse to delete path outside output root: {resolved}") from exc
    return resolved


def _reset_incomplete_dir(path: Path, *, root: Path, label: str, dry_run: bool) -> None:
    resolved = _ensure_within_root(path, root)
    if dry_run:
        print(f"[{label}][reset-planned] output_dir={resolved}")
        return
    print(f"[{label}][reset] removing incomplete output_dir={resolved}")
    shutil.rmtree(resolved)


def _load_cached_virtual_init(target_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    init_set_path = target_root / "init_set.csv"
    init_assignments_path = target_root / "init_assignments.csv"
    init_quotas_path = target_root / "init_quotas.csv"
    if not (init_set_path.exists() and init_assignments_path.exists() and init_quotas_path.exists()):
        return None
    init_df = ensure_pool(pd.read_csv(init_set_path), f"{target_root.name}_cached_init")
    assignments_df = pd.read_csv(init_assignments_path)
    quotas_df = pd.read_csv(init_quotas_path)
    return init_df, assignments_df, quotas_df


def _load_or_build_virtual_pool(raw_df: pd.DataFrame, *, target_root: Path, label: str) -> tuple[pd.DataFrame, str]:
    pool_cache_path = target_root / "pool_with_properties.csv"
    if pool_cache_path.exists():
        pool_df = ensure_pool(pd.read_csv(pool_cache_path), f"{label}_pool_cache")
        return pool_df, "cache"
    has_qed = "qed" in raw_df.columns and not pd.to_numeric(raw_df["qed"], errors="coerce").isna().any()
    has_sa = "sa_score" in raw_df.columns and not pd.to_numeric(raw_df["sa_score"], errors="coerce").isna().any()
    pool_df = ensure_pool(raw_df, label)
    pool_df.to_csv(pool_cache_path, index=False)
    if has_qed and has_sa:
        return pool_df, "source_csv"
    return pool_df, "computed"


def _smiles_signature(smiles_list: list[str]) -> str:
    digest = hashlib.sha256()
    for smi in smiles_list:
        digest.update(str(smi).strip().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_or_build_dataset_fp_cache(
    train_pool: pd.DataFrame,
    *,
    train_csv: Path,
    radius: int,
    n_bits: int,
) -> tuple[np.ndarray, Path, str]:
    smiles = train_pool["smiles_canonical"].astype(str).tolist()
    source_stat = train_csv.stat()
    expected_meta = {
        "cache_format": "dockstring_fp_cache_v2",
        "source_csv": _relative_or_absolute(train_csv),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "pool_n": int(train_pool.shape[0]),
        "smiles_signature": _smiles_signature(smiles),
        "fp_radius": int(radius),
        "fp_bits": int(n_bits),
    }
    cache_dir = train_csv.parent / "_fp_cache" / train_csv.stem / f"morgan_r{int(radius)}_b{int(n_bits)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    fp_bytes_path = cache_dir / "fp_bytes.npy"
    if meta_path.exists() and fp_bytes_path.exists():
        actual_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(actual_meta, dict):
            raise RuntimeError(f"Fingerprint cache metadata must be a JSON object: {meta_path}")
        if actual_meta == expected_meta:
            fp_bytes = np.load(fp_bytes_path, allow_pickle=False)
            expected_shape = (int(train_pool.shape[0]), int(n_bits) // 8)
            if fp_bytes.dtype != np.uint8 or tuple(fp_bytes.shape) != expected_shape:
                raise RuntimeError(
                    f"Fingerprint cache array shape mismatch: got dtype={fp_bytes.dtype} shape={tuple(fp_bytes.shape)}, "
                    f"expected dtype=uint8 shape={expected_shape}"
                )
            fp_mat = np.unpackbits(fp_bytes, axis=1)
            if tuple(fp_mat.shape) != (int(train_pool.shape[0]), int(n_bits)):
                raise RuntimeError(
                    f"Unpacked fingerprint matrix shape mismatch: got {tuple(fp_mat.shape)}, "
                    f"expected {(int(train_pool.shape[0]), int(n_bits))}"
                )
            return fp_mat.astype(np.uint8, copy=False), cache_dir, "cache"
    fps = _compute_morgan_fingerprints(
        train_pool,
        smiles_col="smiles_canonical",
        radius=int(radius),
        n_bits=int(n_bits),
    )
    fp_bytes = np.stack(
        [np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.uint8) for fp in fps],
        axis=0,
    )
    np.save(fp_bytes_path, fp_bytes, allow_pickle=False)
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    fp_mat = np.unpackbits(fp_bytes, axis=1)
    if tuple(fp_mat.shape) != (int(train_pool.shape[0]), int(n_bits)):
        raise RuntimeError(
            f"Computed fingerprint matrix shape mismatch: got {tuple(fp_mat.shape)}, "
            f"expected {(int(train_pool.shape[0]), int(n_bits))}"
        )
    return fp_mat.astype(np.uint8, copy=False), cache_dir, "computed"


def _fit_scipy_kmeans(
    fp_mat: np.ndarray,
    *,
    n_clusters: int,
    n_init: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if fp_mat.ndim != 2:
        raise RuntimeError(f"Expected 2-D fingerprint matrix, got shape={tuple(fp_mat.shape)}")
    if int(n_clusters) <= 0:
        raise RuntimeError(f"n_clusters must be positive, got {n_clusters}")
    if fp_mat.shape[0] < int(n_clusters):
        raise RuntimeError(f"n_clusters={n_clusters} exceeds sample count={fp_mat.shape[0]}")
    x = fp_mat.astype(np.float32, copy=False)
    rng_master = np.random.default_rng(int(seed))
    best_centroids = None
    best_labels = None
    best_inertia = None
    for _ in range(int(n_init)):
        init_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        init_centroids = x[init_rng.choice(x.shape[0], size=int(n_clusters), replace=False)].copy()
        centroids, labels = scipy_vq.kmeans2(
            data=x,
            k=init_centroids,
            iter=int(max_iter),
            minit="matrix",
            missing="raise",
            check_finite=False,
        )
        labels = np.asarray(labels, dtype=np.int32)
        cluster_sizes = np.bincount(labels, minlength=int(n_clusters))
        if np.any(cluster_sizes == 0):
            continue
        deltas = x - centroids[labels]
        inertia = float(np.sum(deltas * deltas, dtype=np.float64))
        if best_inertia is None or inertia < best_inertia:
            best_centroids = np.asarray(centroids, dtype=np.float32)
            best_labels = labels.copy()
            best_inertia = inertia
    if best_centroids is None or best_labels is None or best_inertia is None:
        raise RuntimeError("SciPy kmeans failed to produce a non-empty assignment for every cluster.")
    return best_centroids, best_labels, float(best_inertia)


def _extend_selection_to_size(fp_mat: np.ndarray, pick_idx: np.ndarray, *, target_size: int, batch_size: int = 4096) -> np.ndarray:
    selected = [int(x) for x in np.asarray(pick_idx, dtype=np.int64).tolist()]
    selected_set = set(selected)
    total = int(fp_mat.shape[0])
    if len(selected_set) != len(selected):
        raise RuntimeError("Duplicate indices detected before diversity backfill.")
    if len(selected) > int(target_size):
        raise RuntimeError(f"Initial selection size {len(selected)} exceeds target_size={target_size}.")
    if len(selected) == int(target_size):
        return np.asarray(selected, dtype=np.int64)
    remaining = np.asarray([idx for idx in range(total) if idx not in selected_set], dtype=np.int64)
    if remaining.size == 0:
        raise RuntimeError("No remaining molecules available for diversity backfill.")
    fp_float = fp_mat.astype(np.float32, copy=False)
    min_dist = np.full(remaining.shape[0], np.inf, dtype=np.float32)
    for sel_idx in selected:
        sel_vec = fp_float[sel_idx : sel_idx + 1]
        for start in range(0, remaining.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), remaining.shape[0])
            sub = fp_float[remaining[start:stop]]
            dist = np.sum((sub - sel_vec) ** 2, axis=1, dtype=np.float32)
            min_dist[start:stop] = np.minimum(min_dist[start:stop], dist)
    while len(selected) < int(target_size):
        next_pos = int(np.argmax(min_dist))
        next_idx = int(remaining[next_pos])
        selected.append(next_idx)
        selected_set.add(next_idx)
        sel_vec = fp_float[next_idx : next_idx + 1]
        for start in range(0, remaining.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), remaining.shape[0])
            sub = fp_float[remaining[start:stop]]
            dist = np.sum((sub - sel_vec) ** 2, axis=1, dtype=np.float32)
            min_dist[start:stop] = np.minimum(min_dist[start:stop], dist)
        min_dist[next_pos] = -1.0
    return np.asarray(selected, dtype=np.int64)


def ensure_pool(df: pd.DataFrame, label: str) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    if "smiles_canonical" not in work.columns:
        if "smiles" in work.columns:
            work["smiles_canonical"] = work["smiles"].astype(str)
        elif "canonical_smiles" in work.columns:
            work["smiles_canonical"] = work["canonical_smiles"].astype(str)
        else:
            raise RuntimeError(f"{label}: missing smiles column.")
    if "dock_score" not in work.columns:
        raise RuntimeError(f"{label}: missing dock_score.")
    work["smiles_canonical"] = work["smiles_canonical"].astype(str).str.strip()
    work["dock_score"] = pd.to_numeric(work["dock_score"], errors="coerce")
    if work["dock_score"].isna().any():
        raise RuntimeError(f"{label}: non-numeric dock_score.")
    if "qed" not in work.columns:
        work["qed"] = work["smiles_canonical"].map(compute_qed)
    else:
        work["qed"] = pd.to_numeric(work["qed"], errors="coerce")
    if "sa_score" not in work.columns:
        work["sa_score"] = work["smiles_canonical"].map(compute_sa)
    else:
        work["sa_score"] = pd.to_numeric(work["sa_score"], errors="coerce")
    if work["qed"].isna().any() or work["sa_score"].isna().any():
        raise RuntimeError(f"{label}: QED/SA contains NaN.")
    if "ligand_id" not in work.columns:
        work["ligand_id"] = [f"{label}_{i + 1:06d}" for i in range(work.shape[0])]
    else:
        work["ligand_id"] = work["ligand_id"].astype(str)
    if "added_iter" not in work.columns:
        work["added_iter"] = 0
    return work


def resolve_python_exe(raw: str) -> str:
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)
    for probe in (REPO / "train_log.txt", REPO / "log"):
        if not probe.exists():
            continue
        text = probe.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"([A-Za-z]:\\[^\\\r\n]+\\python\.exe)", text)
        if m and Path(m.group(1)).exists():
            return m.group(1)
    return sys.executable


def resolve_ligand_vocab(dataset_root: Path, smiles: list[str]) -> list[str]:
    vocab = _load_ligand_vocab_override(dataset_root)
    if vocab:
        return list(vocab)
    vocab = _scan_ligand_vocab(smiles)
    if not vocab:
        raise RuntimeError("Resolved ligand vocab is empty.")
    return list(vocab)


def threshold_mask(df: pd.DataFrame, dock_thr: float, qed_thr: float, sa_thr: float) -> pd.Series:
    return (df["dock_score"] <= float(dock_thr)) & (df["qed"] >= float(qed_thr)) & (df["sa_score"] <= float(sa_thr))


def mc_objective_samples(scored: pd.DataFrame, num_samples: int, dock_sign: float, qed_sign: float, sa_sign: float, seed: int) -> torch.Tensor:
    mu = torch.tensor(scored["pred_dock_mean"].to_numpy(dtype=np.float32))
    sigma = torch.tensor(scored["pred_dock_std"].to_numpy(dtype=np.float32))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    eps = torch.randn((int(num_samples), mu.numel()), generator=g, dtype=torch.float32)
    dock = mu.view(1, -1) + sigma.view(1, -1) * eps
    qed = torch.tensor(scored["qed"].to_numpy(dtype=np.float32))
    sa = torch.tensor(scored["sa_score"].to_numpy(dtype=np.float32))
    return torch.stack(
        [
            float(dock_sign) * dock,
            float(qed_sign) * qed.view(1, -1).expand(int(num_samples), -1),
            float(sa_sign) * sa.view(1, -1).expand(int(num_samples), -1),
        ],
        dim=-1,
    )


def qnparego_scores_from_samples_nd(
    samples: torch.Tensor,
    y_train: torch.Tensor,
    weights: list[float],
    scalarizations: int,
    seed: int,
) -> torch.Tensor:
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")
    if int(scalarizations) <= 0:
        raise RuntimeError("qNParEGO scalarizations must be positive.")
    weight_t = torch.as_tensor(weights, dtype=samples.dtype, device=samples.device)
    y_train_w = y_train.to(device=samples.device, dtype=samples.dtype) * weight_t
    samples_w = samples * weight_t
    rng = np.random.default_rng(int(seed))
    scores = torch.zeros(int(samples.shape[1]), dtype=samples.dtype, device=samples.device)
    for _ in range(int(scalarizations)):
        raw_weights = rng.random(int(samples.shape[-1]))
        raw_weights = raw_weights / raw_weights.sum()
        scalarization = get_chebyshev_scalarization(
            weights=torch.as_tensor(raw_weights, dtype=samples.dtype, device=samples.device),
            Y=y_train_w,
        )
        best_f = scalarization(y_train_w).max()
        values = scalarization(samples_w)
        scores += torch.clamp(values - best_f, min=0.0).mean(dim=0)
    return scores / float(scalarizations)


def qnehvi_scores_from_samples_fast_3d(
    samples: torch.Tensor,
    y_train: torch.Tensor,
    weights: list[float],
    ref_point: list[float],
) -> torch.Tensor:
    if samples.dim() != 3 or samples.size(-1) != 3:
        raise ValueError(f"samples must be (S,N,3), got {tuple(samples.shape)}")
    exact_obj = samples[0, :, 1:3].to(dtype=torch.double)
    y_train_d = y_train.to(dtype=torch.double)
    zero_sigma = torch.zeros(int(samples.shape[1]), dtype=torch.double, device=samples.device)
    total = torch.zeros(int(samples.shape[1]), dtype=torch.double, device=samples.device)
    for sample_idx in range(int(samples.shape[0])):
        scores, _ = nehvi_gaussian_analytic_3d(
            dock_mu=samples[sample_idx, :, 0].to(dtype=torch.double),
            dock_sigma=zero_sigma,
            exact_obj=exact_obj,
            y_train=y_train_d,
            weights=weights,
            ref_point=ref_point,
            return_metadata=True,
            validate=False,
        )
        total += scores.to(device=samples.device, dtype=torch.double)
    return total / float(samples.shape[0])


def _candidate_prediction_loader(
    candidate_df: pd.DataFrame,
    *,
    surrogate_kind: str,
    ligand_vocab: list[str],
    model_node_dim: int,
    model_edge_dim: int,
    model_atom_extra_dim: int,
    model_bond_extra_dim: int,
    model_fp_dim: int,
    model_fp_radius: int,
    surrogate_meta: dict,
    pred_batch_size: int,
    candidate_3d_cfg: dict,
) -> tuple[pd.DataFrame, DataLoader]:
    if surrogate_kind == "tensornet":
        eval_rows = candidate_df.copy().reset_index(drop=True)
        eval_rows["split"] = "test"
        cache_dir = candidate_3d_cfg.get("cache_dir")
        cache_dir = Path(cache_dir) if cache_dir else GLOBAL_LIGAND3D_CACHE_DIR
        store = LigandOnly3DStore(
            root=REPO,
            ligand_vocab_override=list(ligand_vocab),
            cache_dir=cache_dir,
            fp_dim=int(model_fp_dim),
            fp_radius=int(model_fp_radius),
            rows=eval_rows.to_dict("records"),
            confgen_max_attempts=int(candidate_3d_cfg.get("max_attempts", 3)),
            confgen_seed=int(candidate_3d_cfg.get("seed", 0)),
            confgen_num_confs=int(candidate_3d_cfg.get("num_confs", 4)),
            confgen_max_opt_iters=int(candidate_3d_cfg.get("max_opt_iters", 100)),
            confgen_optimize=bool(candidate_3d_cfg.get("optimize", True)),
            confgen_prefer_mmff=bool(candidate_3d_cfg.get("prefer_mmff", False)),
            build_num_workers=int(candidate_3d_cfg.get("workers", 1)),
            build_mp_chunksize=int(candidate_3d_cfg.get("chunksize", 16)),
        )
        kept_ids = [lig_id for lig_id in eval_rows["ligand_id"].astype(str).tolist() if lig_id in store.id_to_idx]
        if not kept_ids:
            raise RuntimeError("Acquisition scoring retained no valid TensorNet candidates.")
        kept_indices = [int(store.id_to_idx[lig_id]) for lig_id in kept_ids]
        loader = DataLoader(store.get_dataset_from_indices(kept_indices), batch_size=int(pred_batch_size), shuffle=False)
        scored = eval_rows.set_index("ligand_id").loc[kept_ids].reset_index()
        return scored, loader

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
        batch_size=int(pred_batch_size),
        candidate_3d_cfg=candidate_3d_cfg,
    )
    scored = candidate_df.set_index("smiles_canonical").loc[kept].reset_index()
    if scored.empty:
        raise RuntimeError("Acquisition scoring retained no valid candidates.")
    return scored, loader


def _bayes_predictive_samples(
    candidate_df: pd.DataFrame,
    *,
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
    num_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, torch.Tensor]:
    scored, loader = _candidate_prediction_loader(
        candidate_df,
        surrogate_kind=surrogate_kind,
        ligand_vocab=ligand_vocab,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        pred_batch_size=int(pred_batch_size),
        candidate_3d_cfg=candidate_3d_cfg,
    )
    model.eval()
    scale = float(std)
    shift = float(mean)
    sample_chunks: list[torch.Tensor] = []
    fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            for _ in range(max(1, int(num_samples))):
                draw_parts: list[torch.Tensor] = []
                model.sample_bayes_params()
                for batch in loader:
                    batch = batch.to(device)
                    pred = model(batch).detach().view(-1)
                    draw_parts.append((pred * scale + shift).cpu())
                model.clear_bayes_params()
                sample_chunks.append(torch.cat(draw_parts, dim=0))
    dock_draws = torch.stack(sample_chunks, dim=0)
    dock_mean = dock_draws.mean(dim=0)
    if dock_draws.size(0) == 1:
        dock_var = torch.full_like(dock_mean, 1.0e-6)
    else:
        dock_var = dock_draws.var(dim=0, unbiased=False).clamp_min(1.0e-6)
    dock_std = torch.sqrt(dock_var)
    scored["pred_dock_mean"] = dock_mean.numpy()
    scored["pred_dock_std"] = dock_std.numpy()
    scored["pred_dock_std_total"] = dock_std.numpy()
    scored["pred_dock_std_epi"] = dock_std.numpy()
    scored["pred_dock_std_ale"] = np.zeros(int(scored.shape[0]), dtype=np.float32)
    scored["pred_dock_var_epi_frac"] = np.ones(int(scored.shape[0]), dtype=np.float32)
    qed = torch.tensor(scored["qed"].to_numpy(dtype=np.float32))
    sa = torch.tensor(scored["sa_score"].to_numpy(dtype=np.float32))
    objective_samples = torch.stack(
        [
            float(dock_sign) * dock_draws,
            float(qed_sign) * qed.view(1, -1).expand(dock_draws.size(0), -1),
            float(sa_sign) * sa.view(1, -1).expand(dock_draws.size(0), -1),
        ],
        dim=-1,
    )
    return scored, objective_samples


def _bayes_predictive_samples_with_prefix_timing(
    candidate_df: pd.DataFrame,
    *,
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
    sample_sizes: list[int],
    seed: int,
) -> tuple[pd.DataFrame, torch.Tensor, dict[int, float]]:
    requested = sorted({int(size) for size in sample_sizes})
    if not requested or requested[0] <= 0:
        raise RuntimeError("sample_sizes must contain positive integers.")
    start = time.perf_counter()
    scored, loader = _candidate_prediction_loader(
        candidate_df,
        surrogate_kind=surrogate_kind,
        ligand_vocab=ligand_vocab,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        pred_batch_size=int(pred_batch_size),
        candidate_3d_cfg=candidate_3d_cfg,
    )
    model.eval()
    scale = float(std)
    shift = float(mean)
    sample_chunks: list[torch.Tensor] = []
    prefix_elapsed: dict[int, float] = {}
    fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            for draw_idx in range(int(requested[-1])):
                draw_parts: list[torch.Tensor] = []
                model.sample_bayes_params()
                for batch in loader:
                    batch = batch.to(device)
                    pred = model(batch).detach().view(-1)
                    draw_parts.append((pred * scale + shift).cpu())
                model.clear_bayes_params()
                sample_chunks.append(torch.cat(draw_parts, dim=0))
                completed = draw_idx + 1
                if completed in requested:
                    prefix_elapsed[int(completed)] = float(time.perf_counter() - start)
    dock_draws = torch.stack(sample_chunks, dim=0)
    qed = torch.tensor(scored["qed"].to_numpy(dtype=np.float32))
    sa = torch.tensor(scored["sa_score"].to_numpy(dtype=np.float32))
    objective_samples = torch.stack(
        [
            float(dock_sign) * dock_draws,
            float(qed_sign) * qed.view(1, -1).expand(dock_draws.size(0), -1),
            float(sa_sign) * sa.view(1, -1).expand(dock_draws.size(0), -1),
        ],
        dim=-1,
    )
    return scored, objective_samples, prefix_elapsed


def select_topk(scored: pd.DataFrame, method: str, top_k: int, seed: int) -> pd.DataFrame:
    n = min(int(top_k), int(scored.shape[0]))
    if method == "random":
        return scored.sample(n=n, random_state=int(seed)).reset_index(drop=True)
    col_map = {
        "greedy_mean": "score_greedy_mean",
        "analytic_ehvi": "score_analytic_ehvi",
        "analytic_pomhi": "score_analytic_pomhi",
        "qnehvi_mc": "score_qnehvi_mc",
        "qpmhi_mc": "score_qpmhi_mc",
        "qnparego_mc": "score_qnparego_mc",
        "pio": "score_pio",
    }
    score_col = col_map[method]
    return scored.sort_values([score_col, "pred_dock_mean"], ascending=[False, True]).head(n).reset_index(drop=True)


def eval_batch(labeled_df: pd.DataFrame, batch_df: pd.DataFrame, dock_sign: float, qed_sign: float, sa_sign: float, ref_point: list[float], dock_thr: float, qed_thr: float, sa_thr: float) -> dict[str, float]:
    base = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    batch = build_objective_tensor(batch_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    hv0 = compute_hypervolume(base, ref_point=ref_point)
    hv1 = compute_hypervolume(torch.cat([base, batch], dim=0), ref_point=ref_point)
    return {
        "mean_dock": float(batch_df["dock_score"].mean()),
        "hv_gained": float(hv1 - hv0),
        "threshold_rate": float(threshold_mask(batch_df, dock_thr, qed_thr, sa_thr).mean()),
    }


def true_single_candidate_hvi_gain(candidate_df: pd.DataFrame, labeled_df: pd.DataFrame, dock_sign: float, qed_sign: float, sa_sign: float, weights: list[float], ref_point: list[float]) -> np.ndarray:
    y_train = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    exact_obj = torch.stack(
        [
            float(qed_sign) * torch.tensor(candidate_df["qed"].to_numpy(dtype=np.float32)),
            float(sa_sign) * torch.tensor(candidate_df["sa_score"].to_numpy(dtype=np.float32)),
        ],
        dim=-1,
    )
    dock_mu = float(dock_sign) * torch.tensor(candidate_df["dock_score"].to_numpy(dtype=np.float32))
    dock_sigma = torch.zeros_like(dock_mu)
    gains, _ = nehvi_gaussian_analytic_3d(
        dock_mu=dock_mu,
        dock_sigma=dock_sigma,
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
        return_metadata=True,
    )
    return gains.detach().cpu().numpy().astype(np.float64)


def mean_hvi_gain(candidate_df: pd.DataFrame, labeled_df: pd.DataFrame, dock_sign: float, qed_sign: float, sa_sign: float, weights: list[float], ref_point: list[float]) -> np.ndarray:
    y_train = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    exact_obj = torch.stack(
        [
            float(qed_sign) * torch.tensor(candidate_df["qed"].to_numpy(dtype=np.float32)),
            float(sa_sign) * torch.tensor(candidate_df["sa_score"].to_numpy(dtype=np.float32)),
        ],
        dim=-1,
    )
    dock_mu = float(dock_sign) * torch.tensor(candidate_df["pred_dock_mean"].to_numpy(dtype=np.float32))
    dock_sigma = torch.zeros_like(dock_mu)
    gains, _ = nehvi_gaussian_analytic_3d(
        dock_mu=dock_mu,
        dock_sigma=dock_sigma,
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
        return_metadata=True,
    )
    return gains.detach().cpu().numpy().astype(np.float64)


def acquisition_ranking_metrics(scored: pd.DataFrame, score_col: str, top_k: int, true_col: str = "true_hvi_gain") -> dict[str, float | int]:
    if score_col not in scored.columns:
        raise RuntimeError(f"Missing acquisition score column: {score_col}")
    if true_col not in scored.columns:
        raise RuntimeError(f"Missing ranking target column: {true_col}")
    work = scored.copy().reset_index(drop=True)
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[true_col] = pd.to_numeric(work[true_col], errors="coerce")
    work = work.loc[np.isfinite(work[score_col].to_numpy(dtype=np.float64)) & np.isfinite(work[true_col].to_numpy(dtype=np.float64))].copy().reset_index(drop=True)
    if work.empty:
        raise RuntimeError("Cannot compute acquisition ranking metrics on an empty finite score set.")

    score_values = work[score_col].to_numpy(dtype=np.float64)
    true_values = work[true_col].to_numpy(dtype=np.float64)
    spearman = float(pd.Series(score_values).corr(pd.Series(true_values), method="spearman")) if work.shape[0] >= 2 else float("nan")
    kendall = float(pd.Series(score_values).corr(pd.Series(true_values), method="kendall")) if work.shape[0] >= 2 else float("nan")

    k = min(int(top_k), int(work.shape[0]))
    pred_sort_cols = [score_col]
    pred_ascending = [False]
    if "pred_dock_mean" in work.columns:
        pred_sort_cols.append("pred_dock_mean")
        pred_ascending.append(True)
    pred_top = work.sort_values(pred_sort_cols, ascending=pred_ascending).head(k)
    true_top = work.sort_values([true_col, "dock_score"], ascending=[False, True]).head(k)
    pred_set = set(pred_top.index.tolist())
    true_set = set(true_top.index.tolist())

    true_rank = work[true_col].rank(method="min", ascending=False)
    pred_ranks = true_rank.loc[pred_top.index].to_numpy(dtype=np.float64)
    return {
        "score_spearman_true_hvi": spearman,
        "score_kendall_true_hvi": kendall,
        "topk_true_hvi_overlap": int(len(pred_set & true_set)),
        "topk_true_hvi_recall": float(len(pred_set & true_set) / max(k, 1)),
        "selected_true_hvi_mean": float(pred_top[true_col].mean()),
        "selected_true_hvi_median_rank": float(np.median(pred_ranks)),
        "score_positive_n": int(np.sum(score_values > 0.0)),
    }


def eval_prediction_frame(scored: pd.DataFrame) -> dict[str, float]:
    if scored.empty:
        raise RuntimeError("Cannot evaluate an empty prediction frame.")
    work = scored.copy().reset_index(drop=True)
    work["pred_dock_mean"] = pd.to_numeric(work["pred_dock_mean"], errors="coerce")
    work["dock_score"] = pd.to_numeric(work["dock_score"], errors="coerce")
    work["pred_dock_std"] = pd.to_numeric(work.get("pred_dock_std", pd.Series(np.nan, index=work.index)), errors="coerce")
    work = work.loc[work["pred_dock_mean"].notna() & work["dock_score"].notna()].copy().reset_index(drop=True)
    if work.empty:
        raise RuntimeError("Prediction frame has no rows with both pred_dock_mean and dock_score.")
    pred = work["pred_dock_mean"].to_numpy(dtype=float)
    truth = work["dock_score"].to_numpy(dtype=float)
    pred_std = work["pred_dock_std"].to_numpy(dtype=float)
    abs_err = np.abs(pred - truth)
    rmse = float(np.sqrt(np.mean(np.square(pred - truth))))
    mae = float(np.mean(abs_err))
    tot = float(np.sum(np.square(truth - float(np.mean(truth)))))
    r2 = float(1.0 - (np.sum(np.square(pred - truth)) / tot)) if tot > 0.0 else float("nan")
    sigma = np.where(np.isfinite(pred_std), np.clip(pred_std, 1.0e-6, None), np.nan)
    if np.isfinite(sigma).all():
        nll = float(np.mean(0.5 * np.log(2.0 * math.pi * np.square(sigma)) + 0.5 * np.square((truth - pred) / sigma)))
        cov1 = float(np.mean(np.abs(truth - pred) <= sigma))
        cov2 = float(np.mean(np.abs(truth - pred) <= 2.0 * sigma))
        mean_std = float(np.mean(sigma))
        std_err_spearman = float(pd.Series(pred_std).corr(pd.Series(abs_err), method="spearman"))
    else:
        nll = float("nan")
        cov1 = float("nan")
        cov2 = float("nan")
        mean_std = float("nan")
        std_err_spearman = float("nan")
    if work.shape[0] >= 2:
        spearman = float(work["pred_dock_mean"].corr(work["dock_score"], method="spearman"))
        kendall = float(work["pred_dock_mean"].corr(work["dock_score"], method="kendall"))
    else:
        spearman = float("nan")
        kendall = float("nan")
    return {
        "candidate_test_n": int(work.shape[0]),
        "candidate_test_rmse": rmse,
        "candidate_test_mae": mae,
        "candidate_test_r2": r2,
        "candidate_test_nll": nll,
        "candidate_test_spearman": spearman,
        "candidate_test_kendall": kendall,
        "candidate_test_cov1": cov1,
        "candidate_test_cov2": cov2,
        "candidate_test_mean_std": mean_std,
        "candidate_test_std_error_spearman": std_err_spearman,
    }


def build_fixed_surrogate_dataset(
    train_csv: Path,
    test_csv: Path,
    out_dir: Path,
    *,
    target_name: str,
    seed: int,
    init_size: int,
    init_clusters: int,
    init_min_per_cluster: int,
) -> Path:
    train_pool = ensure_pool(pd.read_csv(train_csv), f"{target_name}_train")
    test_pool = ensure_pool(pd.read_csv(test_csv), f"{target_name}_test")
    fp_mat, fp_cache_dir, fp_cache_source = _load_or_build_dataset_fp_cache(
        train_pool,
        train_csv=train_csv,
        radius=int(DEFAULT_FP_RADIUS),
        n_bits=int(DEFAULT_FP_BITS),
    )
    print(
        f"[surrogate][fp-cache] target={target_name} seed={int(seed)} "
        f"source={fp_cache_source} cache_dir={fp_cache_dir}"
    )
    centroids, labels, inertia = _fit_scipy_kmeans(
        fp_mat,
        n_clusters=int(init_clusters),
        n_init=2,
        seed=int(seed),
        max_iter=20,
    )
    cluster_sizes = np.bincount(labels.astype(np.int32), minlength=int(init_clusters))
    quotas = allocate_cluster_quota(
        cluster_sizes,
        total_size=int(init_size),
        min_per_cluster=int(init_min_per_cluster),
    )
    pick_idx = select_cluster_members(fp_mat, quotas=quotas, labels=labels, centroids=centroids)
    if pick_idx.shape[0] < int(init_size):
        pick_idx = _extend_selection_to_size(fp_mat, pick_idx, target_size=int(init_size))
    init_df = train_pool.iloc[pick_idx].copy().reset_index(drop=True)
    if init_df.shape[0] != int(init_size):
        raise RuntimeError(f"{target_name}: selected {init_df.shape[0]} rows, expected {int(init_size)}.")
    init_df["selection_rank"] = np.arange(1, init_df.shape[0] + 1, dtype=np.int32)
    assignments_df = train_pool.copy().reset_index(drop=True)
    assignments_df["cluster"] = labels.astype(np.int32)
    assignments_df["selected"] = False
    assignments_df["selection_rank"] = -1
    assignments_df.loc[pick_idx, "selected"] = True
    assignments_df.loc[pick_idx, "selection_rank"] = np.arange(1, pick_idx.shape[0] + 1, dtype=np.int32)
    quota_df = (
        pd.DataFrame({
            "cluster": np.arange(int(init_clusters), dtype=np.int32),
            "size": cluster_sizes.astype(np.int32),
            "quota": quotas.astype(np.int32),
        })
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    if init_df.empty:
        raise RuntimeError(f"{target_name}: selected empty training subset.")
    init_df["split_role"] = "init_pool"
    init_df["init_group"] = "train_valid_init"
    init_df["split"] = "train"
    test_pool = test_pool.copy().reset_index(drop=True)
    test_pool["split_role"] = "holdout_test"
    test_pool["init_group"] = "holdout_test"
    test_pool["split"] = "test"
    dataset_root = out_dir / "_fixed_surrogate_dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([init_df, test_pool], ignore_index=True)
    combined.to_csv(dataset_root / "smiles.csv", index=False)
    assignments_df.to_csv(dataset_root / "init_assignments.csv", index=False)
    quota_df.to_csv(dataset_root / "init_quotas.csv", index=False)
    ligand_vocab = resolve_ligand_vocab(REPO / "dataset" / target_name, combined["smiles_canonical"].astype(str).tolist())
    (dataset_root / "ligand_vocab.json").write_text(json.dumps(ligand_vocab, indent=2), encoding="utf-8")
    meta = {
        "protocol": "official_train_sample_plus_full_official_test_scipy_kmeans_init",
        "target": target_name,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "seed": int(seed),
        "train_pool_n": int(train_pool.shape[0]),
        "train_selected_n": int(init_df.shape[0]),
        "test_n": int(test_pool.shape[0]),
        "init_size_requested": int(init_size),
        "init_clusters": int(init_clusters),
        "init_min_per_cluster": int(init_min_per_cluster),
        "fp_radius": int(DEFAULT_FP_RADIUS),
        "fp_bits": int(DEFAULT_FP_BITS),
        "cluster_n": int(quota_df.shape[0]),
        "cluster_inertia": float(inertia),
        "clustering_backend": "scipy.cluster.vq.kmeans2",
        "fp_cache_dir": _relative_or_absolute(fp_cache_dir),
        "fp_cache_source": fp_cache_source,
    }
    (dataset_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dataset_root


def run_surrogate(args: argparse.Namespace) -> int:
    if str(args.split).strip().lower() != "train":
        raise RuntimeError("surrogate benchmark always samples labels from official train split and evaluates on the full official test split.")
    if abs(float(args.holdout_frac) - 0.2) > 1.0e-12:
        raise RuntimeError("surrogate benchmark uses the official train split for labels and the full official test split for evaluation; keep --holdout-frac at the default compatibility value.")
    requested_trial_names = _resolve_requested_surrogate_trial_names(args)
    targets = resolve_surrogate_target_pairs(args.target, args.dockstring_root)
    seeds = parse_int_list(args.seeds)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    python_exe = resolve_python_exe(args.python_exe)
    manifest = []
    for target_name, pair in targets.items():
        train_csv, test_csv = pair
        for seed in seeds:
            out_dir = out_root / target_name / f"seed{int(seed):02d}"
            status = _check_surrogate_run_dir(out_dir)
            if status == "complete":
                missing_trials = _missing_requested_surrogate_trials(out_dir, requested_trial_names)
                if not missing_trials:
                    print(f"[surrogate][skip] target={target_name} seed={seed} output_dir={out_dir}")
                    manifest.append({
                        "target": target_name,
                        "seed": int(seed),
                        "output_dir": str(out_dir),
                        "train_csv": str(train_csv),
                        "test_csv": str(test_csv),
                        "dataset_root": str(out_dir / "_fixed_surrogate_dataset"),
                        "train_size": int(args.init_size),
                        "test_size": int(pd.read_csv(test_csv, usecols=["dock_score"]).shape[0]),
                        "status": "skipped_complete",
                    })
                    continue
                print(f"[surrogate][append] target={target_name} seed={seed} output_dir={out_dir} missing_trials={','.join(missing_trials)}")
            if status == "incomplete":
                _reset_incomplete_dir(out_dir, root=out_root, label="surrogate", dry_run=bool(args.dry_run))
            out_dir.mkdir(parents=True, exist_ok=True)
            dataset_root = out_dir / "_fixed_surrogate_dataset"
            if status != "complete":
                dataset_root = build_fixed_surrogate_dataset(
                    train_csv,
                    test_csv,
                    out_dir,
                    target_name=target_name,
                    seed=int(seed),
                    init_size=int(args.init_size),
                    init_clusters=int(args.init_clusters),
                    init_min_per_cluster=int(args.init_min_per_cluster),
                )
            elif not dataset_root.exists():
                raise RuntimeError(f"Expected existing fixed surrogate dataset for append: {dataset_root}")
            cmd = [
                python_exe, str(BENCHMARK_SCRIPT), "--config", str(Path(args.config)), "--output-dir", str(out_dir),
                "--init-dataset-root", str(dataset_root), "--seed", str(int(seed)), "--valid-frac", str(float(args.valid_frac)),
                "--pred-batch-size", str(int(args.pred_batch_size)), "--eval-3d-workers", str(int(args.eval_3d_workers)),
            ]
            if bool(args.skip_gin_small):
                cmd.append("--skip-gin-small")
            if bool(args.skip_scratch):
                cmd.append("--skip-scratch")
            if bool(args.skip_pretrained):
                cmd.append("--skip-pretrained")
            if bool(args.skip_ensemble):
                cmd.append("--skip-ensemble")
            if args.pretrain_csv:
                cmd.extend(["--pretrain-csv", str(Path(args.pretrain_csv))])
            if args.pretrain_smiles_col:
                cmd.extend(["--pretrain-smiles-col", str(args.pretrain_smiles_col)])
            if args.pretrain_valid_frac is not None:
                cmd.extend(["--pretrain-valid-frac", str(float(args.pretrain_valid_frac))])
            if args.pretrain_epochs is not None:
                cmd.extend(["--pretrain-epochs", str(int(args.pretrain_epochs))])
            manifest.append({
                "target": target_name,
                "seed": int(seed),
                "output_dir": str(out_dir),
                "train_csv": str(train_csv),
                "test_csv": str(test_csv),
                "dataset_root": str(dataset_root),
                "train_size": int(args.init_size),
                "test_size": int(pd.read_csv(test_csv, usecols=["dock_score"]).shape[0]),
                "status": "planned" if args.dry_run else "ran",
                "command": cmd,
            })
            if not args.dry_run:
                print(f"[surrogate] target={target_name} seed={seed} output_dir={out_dir}")
                subprocess.run(cmd, check=True, cwd=str(REPO))
    (out_root / "surrogate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def load_benchmark_run(run_dir: Path, trial_name: str) -> tuple[pd.DataFrame, pd.DataFrame, str, pd.Series, Path, dict]:
    summary = pd.read_csv(run_dir / "trial_summary.csv")
    if summary.empty:
        raise RuntimeError(f"Empty trial_summary.csv: {run_dir}")
    if trial_name:
        row = summary.loc[summary["trial_name"].astype(str) == str(trial_name)]
        if row.empty:
            raise RuntimeError(f"Trial {trial_name} not found in {run_dir}")
        row = row.iloc[0]
    else:
        rec_path = run_dir / "recommendation.json"
        if rec_path.exists():
            recommended = json.loads(rec_path.read_text(encoding="utf-8"))["recommended_trial"]
            row = summary.loc[summary["trial_name"].astype(str) == str(recommended)].iloc[0]
        else:
            row = summary.sort_values(["spearman", "nll", "rmse"], ascending=[False, True, True]).iloc[0]
    name = str(row["trial_name"])
    trial_dir = run_dir / name
    holdout = ensure_pool(pd.read_csv(trial_dir / "holdout_predictions.csv"), f"{run_dir.name}_{name}_holdout")
    trial_cfg = json.loads((trial_dir / "trial_config.json").read_text(encoding="utf-8"))
    train_df = ensure_pool(pd.read_csv(Path(str(trial_cfg["dataset_root"])) / "smiles.csv"), f"{run_dir.name}_{name}_train")
    return holdout, train_df, name, row, trial_dir, trial_cfg


def run_ranking(args: argparse.Namespace) -> int:
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    bad = sorted(set(methods) - RANKING_METHODS)
    if bad:
        raise RuntimeError(f"Unsupported ranking methods: {bad}")
    cfg = load_config(args.config)
    objective_cfg = dict(cfg.get("objective") or {})
    selection_cfg = dict(cfg.get("selection") or {})
    candidate_3d_cfg = dict(dict(cfg.get("candidate") or {}).get("candidate_3d") or {})
    candidate_3d_cfg["workers"] = int(args.eval_3d_workers)
    general_cfg = dict(cfg.get("general") or {})
    device_name = str(general_cfg.get("device", "auto")).strip().lower()
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    dock_sign = float(objective_cfg.get("dock_sign", -1.0))
    qed_sign = float(objective_cfg.get("qed_sign", 1.0))
    sa_sign = float(objective_cfg.get("sa_sign", -1.0))
    weights = list(selection_cfg.get("weights", [1.0, 1.0, 1.0]))
    ref_point = list(objective_cfg.get("ref_point", [0.0, 0.0, -20.0]))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for raw in args.run_dir:
        run_dir = Path(raw).resolve()
        scored, labeled_df, trial_name, trial_row, trial_dir, trial_cfg = load_benchmark_run(run_dir, args.trial_name)
        if int(args.candidate_pool_size) > 0:
            take_n = min(int(args.candidate_pool_size), int(scored.shape[0]))
            if take_n <= 0:
                raise RuntimeError("candidate_pool_size must be positive when provided.")
            scored = scored.sample(n=take_n, random_state=int(args.seed)).reset_index(drop=True)
        uncertainty_mode = str(trial_row.get("uncertainty_mode", "")).strip().lower()
        bayes_samples = None
        if uncertainty_mode == "bayes" and any(method in methods for method in ("qnehvi_mc", "qpmhi_mc")):
            ckpt_path = trial_dir / "surrogate.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(ckpt_path)
            (
                model,
                mean,
                std,
                model_node_dim,
                model_edge_dim,
                model_atom_extra_dim,
                model_bond_extra_dim,
                model_fp_dim,
                model_fp_radius,
                _graph,
                surrogate_kind,
                surrogate_meta,
            ) = load_surrogate(str(ckpt_path), device=device)
            ligand_vocab = resolve_ligand_vocab(Path(str(trial_cfg["dataset_root"])), scored["smiles_canonical"].astype(str).tolist())
            scored, bayes_samples = _bayes_predictive_samples(
                scored,
                model=model,
                mean=float(mean),
                std=float(std),
                device=device,
                ligand_vocab=ligand_vocab,
                surrogate_kind=surrogate_kind,
                model_node_dim=int(model_node_dim),
                model_edge_dim=int(model_edge_dim),
                model_atom_extra_dim=int(model_atom_extra_dim),
                model_bond_extra_dim=int(model_bond_extra_dim),
                model_fp_dim=int(model_fp_dim),
                model_fp_radius=int(model_fp_radius),
                surrogate_meta=surrogate_meta,
                pred_batch_size=int(args.pred_batch_size),
                candidate_3d_cfg=candidate_3d_cfg,
                dock_sign=dock_sign,
                qed_sign=qed_sign,
                sa_sign=sa_sign,
                num_samples=int(args.mc_samples),
                seed=int(args.seed) + 409,
            )
        y_train = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
        exact_obj = torch.stack(
            [float(qed_sign) * torch.tensor(scored["qed"].to_numpy(dtype=np.float32)), float(sa_sign) * torch.tensor(scored["sa_score"].to_numpy(dtype=np.float32))],
            dim=-1,
        )
        dock_mu = float(dock_sign) * torch.tensor(scored["pred_dock_mean"].to_numpy(dtype=np.float32))
        dock_sigma = torch.tensor(scored["pred_dock_std"].to_numpy(dtype=np.float32))
        scored = scored.copy()
        if "greedy_mean" in methods:
            scored["score_greedy_mean"] = mean_hvi_gain(scored, labeled_df, dock_sign, qed_sign, sa_sign, weights, ref_point)
        if "analytic_ehvi" in methods:
            vals, _ = nehvi_gaussian_analytic_3d(dock_mu=dock_mu, dock_sigma=dock_sigma, exact_obj=exact_obj, y_train=y_train, weights=weights, ref_point=ref_point, return_metadata=True)
            scored["score_analytic_ehvi"] = vals.detach().cpu().numpy()
        if "analytic_pomhi" in methods:
            vals, _ = qphv_prob_gaussian_analytic_3d(dock_mu=dock_mu, dock_sigma=dock_sigma, exact_obj=exact_obj, y_train=y_train, weights=weights, ref_point=ref_point, return_metadata=True)
            scored["score_analytic_pomhi"] = vals.detach().cpu().numpy()
        if "qnehvi_mc" in methods:
            samples = bayes_samples if bayes_samples is not None else mc_objective_samples(scored, int(args.mc_samples), dock_sign, qed_sign, sa_sign, int(args.seed) + 11)
            vals = qnehvi_scores_from_samples_fast_3d(samples, y_train=y_train, weights=weights, ref_point=ref_point)
            scored["score_qnehvi_mc"] = vals.detach().cpu().numpy()
        if "qpmhi_mc" in methods:
            samples = bayes_samples if bayes_samples is not None else mc_objective_samples(scored, int(args.mc_samples), dock_sign, qed_sign, sa_sign, int(args.seed) + 29)
            vals = qphv_topk_prob_nd_botorch(samples, y_train=y_train, top_k=int(args.top_k), weights=weights, ref_point=ref_point)
            scored["score_qpmhi_mc"] = vals.detach().cpu().numpy()
        if "qnparego_mc" in methods:
            samples = bayes_samples if bayes_samples is not None else mc_objective_samples(scored, int(args.mc_samples), dock_sign, qed_sign, sa_sign, int(args.seed) + 37)
            vals = qnparego_scores_from_samples_nd(
                samples,
                y_train=y_train,
                weights=weights,
                scalarizations=int(args.nparego_scalarizations),
                seed=int(args.seed) + 41,
            )
            scored["score_qnparego_mc"] = vals.detach().cpu().numpy()
        if "pio" in methods:
            mu = scored["pred_dock_mean"].to_numpy(dtype=np.float64)
            sigma = scored["pred_dock_std"].to_numpy(dtype=np.float64)
            z = np.zeros_like(mu, dtype=np.float64)
            pos = sigma > 1.0e-12
            z[pos] = (float(args.dock_threshold) - mu[pos]) / sigma[pos]
            prob = np.zeros_like(mu, dtype=np.float64)
            if np.any(pos):
                z_t = torch.as_tensor(z[pos], dtype=torch.double)
                prob[pos] = (0.5 * (1.0 + torch.erf(z_t / math.sqrt(2.0)))).detach().cpu().numpy()
            det = ~pos
            if np.any(det):
                prob[det] = (mu[det] <= float(args.dock_threshold)).astype(np.float64)
            cheap = (scored["qed"].to_numpy(dtype=np.float64) >= float(args.qed_threshold)) & (scored["sa_score"].to_numpy(dtype=np.float64) <= float(args.sa_threshold))
            scored["score_pio"] = prob * cheap.astype(np.float64)

        run_out = out_root / run_dir.parent.name / run_dir.name / trial_name
        status = _check_ranking_run_dir(run_out)
        if status == "complete":
            print(f"[ranking][skip] run_dir={run_dir} output_dir={run_out}")
            rows = pd.read_csv(run_out / "ranking_summary.csv").to_dict(orient="records")
            all_rows.extend(rows)
            continue
        if status == "incomplete":
            _reset_incomplete_dir(run_out, root=out_root, label="ranking", dry_run=False)
        run_out.mkdir(parents=True, exist_ok=True)
        scored.to_csv(run_out / "candidate_pool.csv", index=False)
        scored.to_csv(run_out / "frozen_pool_scores.csv", index=False)
        rows = []
        for idx, method in enumerate(methods):
            t0 = time.perf_counter()
            picked = select_topk(scored, method, int(args.top_k), int(args.seed) + idx)
            metrics = eval_batch(labeled_df, picked, dock_sign, qed_sign, sa_sign, ref_point, float(args.dock_threshold), float(args.qed_threshold), float(args.sa_threshold))
            row = {"run_dir": str(run_dir), "trial_name": trial_name, "candidate_pool_n": int(scored.shape[0]), "method": method, "top_k": int(picked.shape[0]), "mean_dock": metrics["mean_dock"], "hv_gained": metrics["hv_gained"], "threshold_rate": metrics["threshold_rate"], "wall_time_sec": float(time.perf_counter() - t0)}
            rows.append(row)
            all_rows.append(row)
        pd.DataFrame(rows).to_csv(run_out / "ranking_summary.csv", index=False)
        print(f"[ranking] run_dir={run_dir} trial={trial_name} output_dir={run_out}")
    pd.DataFrame(all_rows).to_csv(out_root / "ranking_summary_all.csv", index=False)
    return 0


def run_acquisition_speed(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    objective_cfg = dict(cfg.get("objective") or {})
    selection_cfg = dict(cfg.get("selection") or {})
    candidate_3d_cfg = dict(dict(cfg.get("candidate") or {}).get("candidate_3d") or {})
    candidate_3d_cfg["workers"] = int(args.eval_3d_workers)
    general_cfg = dict(cfg.get("general") or {})
    device_name = str(general_cfg.get("device", "auto")).strip().lower()
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    dock_sign = float(objective_cfg.get("dock_sign", -1.0))
    qed_sign = float(objective_cfg.get("qed_sign", 1.0))
    sa_sign = float(objective_cfg.get("sa_sign", -1.0))
    weights = list(selection_cfg.get("weights", [1.0, 1.0, 1.0]))
    ref_point = list(objective_cfg.get("ref_point", [0.0, 0.0, -20.0]))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    analytic_run_dir = Path(args.analytic_run_dir).resolve()
    bayes_run_dir = Path(args.bayes_run_dir).resolve()
    analytic_holdout, analytic_train, analytic_trial_name, analytic_row, analytic_trial_dir, analytic_trial_cfg = load_benchmark_run(analytic_run_dir, args.analytic_trial_name)
    _bayes_holdout, _bayes_train, bayes_trial_name, bayes_row, bayes_trial_dir, bayes_trial_cfg = load_benchmark_run(bayes_run_dir, args.bayes_trial_name)
    if str(bayes_row.get("uncertainty_mode", "")).strip().lower() != "bayes":
        raise RuntimeError(f"BNN speed benchmark requires a bayes trial, got uncertainty_mode={bayes_row.get('uncertainty_mode')!r}.")

    candidate_df = analytic_holdout.copy().reset_index(drop=True)
    if int(args.candidate_pool_size) > 0:
        take_n = min(int(args.candidate_pool_size), int(candidate_df.shape[0]))
        if take_n <= 0:
            raise RuntimeError("candidate_pool_size must be positive when provided.")
        candidate_df = candidate_df.sample(n=take_n, random_state=int(args.seed)).reset_index(drop=True)
    y_train = build_objective_tensor(analytic_train, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    candidate_df["true_hvi_gain"] = true_single_candidate_hvi_gain(
        candidate_df,
        analytic_train,
        dock_sign=dock_sign,
        qed_sign=qed_sign,
        sa_sign=sa_sign,
        weights=weights,
        ref_point=ref_point,
    )
    candidate_df.to_csv(out_root / "fixed_candidate_pool.csv", index=False)

    analytic_ckpt = analytic_trial_dir / "surrogate.pt"
    bayes_ckpt = bayes_trial_dir / "surrogate.pt"
    if not analytic_ckpt.exists():
        raise FileNotFoundError(analytic_ckpt)
    if not bayes_ckpt.exists():
        raise FileNotFoundError(bayes_ckpt)

    (
        analytic_model,
        analytic_mean,
        analytic_std,
        analytic_node_dim,
        analytic_edge_dim,
        analytic_atom_extra_dim,
        analytic_bond_extra_dim,
        analytic_fp_dim,
        analytic_fp_radius,
        _analytic_graph,
        analytic_kind,
        analytic_meta,
    ) = load_surrogate(str(analytic_ckpt), device=device)
    analytic_vocab = resolve_ligand_vocab(Path(str(analytic_trial_cfg["dataset_root"])), candidate_df["smiles_canonical"].astype(str).tolist())

    pred_start = time.perf_counter()
    analytic_scored = _score_with_predictions_only(
        candidate_df,
        analytic_model,
        float(analytic_mean),
        float(analytic_std),
        device,
        analytic_vocab,
        analytic_kind,
        int(analytic_node_dim),
        int(analytic_edge_dim),
        int(analytic_atom_extra_dim),
        int(analytic_bond_extra_dim),
        int(analytic_fp_dim),
        int(analytic_fp_radius),
        analytic_meta,
        int(args.pred_batch_size),
        candidate_3d_cfg,
    )
    analytic_pred_time = float(time.perf_counter() - pred_start)
    exact_obj = torch.stack(
        [
            float(qed_sign) * torch.tensor(analytic_scored["qed"].to_numpy(dtype=np.float32)),
            float(sa_sign) * torch.tensor(analytic_scored["sa_score"].to_numpy(dtype=np.float32)),
        ],
        dim=-1,
    )
    dock_mu = float(dock_sign) * torch.tensor(analytic_scored["pred_dock_mean"].to_numpy(dtype=np.float32))
    dock_sigma = torch.tensor(analytic_scored["pred_dock_std"].to_numpy(dtype=np.float32))
    acq_start = time.perf_counter()
    analytic_scores, analytic_meta_qpmhi = qphv_prob_gaussian_analytic_3d(
        dock_mu=dock_mu,
        dock_sigma=dock_sigma,
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
        quadrature_points=int(args.quadrature_points),
        return_metadata=True,
    )
    analytic_acq_time = float(time.perf_counter() - acq_start)
    analytic_scored = analytic_scored.copy()
    analytic_scored["score_analytic_pomhi"] = analytic_scores.detach().cpu().numpy()
    analytic_selected = select_topk(analytic_scored, "analytic_pomhi", int(args.top_k), int(args.seed))
    analytic_selected.to_csv(out_root / "selected_analytic_pomhi.csv", index=False)
    analytic_rank_metrics = acquisition_ranking_metrics(analytic_scored, "score_analytic_pomhi", int(args.top_k))

    rows = [
        {
            "method": "analytic_pomhi",
            "surrogate_trial": analytic_trial_name,
            "kernel": "analytic_closed_form",
            "mc_samples": 0,
            "candidate_pool_n": int(analytic_scored.shape[0]),
            "prediction_time_s": analytic_pred_time,
            "acquisition_time_s": analytic_acq_time,
            "total_time_s": float(analytic_pred_time + analytic_acq_time),
            "selected_overlap_with_analytic": int(args.top_k),
            "qpmhi_scored_n": int(analytic_meta_qpmhi.get("scored_n", 0)),
            **analytic_rank_metrics,
        }
    ]

    (
        bayes_model,
        bayes_mean,
        bayes_std,
        bayes_node_dim,
        bayes_edge_dim,
        bayes_atom_extra_dim,
        bayes_bond_extra_dim,
        bayes_fp_dim,
        bayes_fp_radius,
        _bayes_graph,
        bayes_kind,
        bayes_meta,
    ) = load_surrogate(str(bayes_ckpt), device=device)
    bayes_vocab = resolve_ligand_vocab(Path(str(bayes_trial_cfg["dataset_root"])), analytic_scored["smiles_canonical"].astype(str).tolist())
    sample_sizes = sorted({int(size) for size in args.mc_samples})
    bayes_base, bayes_samples, bayes_pred_prefix = _bayes_predictive_samples_with_prefix_timing(
        analytic_scored,
        model=bayes_model,
        mean=float(bayes_mean),
        std=float(bayes_std),
        device=device,
        ligand_vocab=bayes_vocab,
        surrogate_kind=bayes_kind,
        model_node_dim=int(bayes_node_dim),
        model_edge_dim=int(bayes_edge_dim),
        model_atom_extra_dim=int(bayes_atom_extra_dim),
        model_bond_extra_dim=int(bayes_bond_extra_dim),
        model_fp_dim=int(bayes_fp_dim),
        model_fp_radius=int(bayes_fp_radius),
        surrogate_meta=bayes_meta,
        pred_batch_size=int(args.pred_batch_size),
        candidate_3d_cfg=candidate_3d_cfg,
        dock_sign=dock_sign,
        qed_sign=qed_sign,
        sa_sign=sa_sign,
        sample_sizes=sample_sizes,
        seed=int(args.seed) + 409,
    )
    if bayes_base["smiles_canonical"].astype(str).tolist() != analytic_scored["smiles_canonical"].astype(str).tolist():
        raise RuntimeError("BNN and analytic scoring retained different candidate pools; speed benchmark requires identical candidates.")

    analytic_selected_smiles = set(analytic_selected["smiles_canonical"].astype(str).tolist())
    for sample_count in sample_sizes:
        sample_prefix = bayes_samples[: int(sample_count)]
        scored_s = bayes_base.copy()
        dock_draws = sample_prefix[:, :, 0] / float(dock_sign)
        scored_s["pred_dock_mean"] = dock_draws.mean(dim=0).detach().cpu().numpy()
        if sample_prefix.size(0) == 1:
            dock_std = torch.full_like(dock_draws[0], 1.0e-6)
        else:
            dock_std = dock_draws.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        scored_s["pred_dock_std"] = dock_std.detach().cpu().numpy()

        acq_start = time.perf_counter()
        baseline_vals = qphv_prob_scaled_nd_botorch(sample_prefix, y_train=y_train, weights=weights, ref_point=ref_point)
        baseline_acq_time = float(time.perf_counter() - acq_start)
        baseline_scored = scored_s.copy()
        baseline_scored["score_qpmhi_mc_baseline"] = baseline_vals.detach().cpu().numpy()
        baseline_selected = (
            baseline_scored.sort_values(["score_qpmhi_mc_baseline", "pred_dock_mean"], ascending=[False, True], kind="mergesort")
            .head(int(args.top_k))
            .reset_index(drop=True)
        )
        baseline_selected.to_csv(out_root / f"selected_qpmhi_mc_bnn_baseline_S{int(sample_count)}.csv", index=False)
        baseline_overlap = len(analytic_selected_smiles & set(baseline_selected["smiles_canonical"].astype(str).tolist()))
        baseline_rank_metrics = acquisition_ranking_metrics(baseline_scored, "score_qpmhi_mc_baseline", int(args.top_k))
        rows.append(
            {
                "method": f"qpmhi_mc_bnn_baseline_S{int(sample_count)}",
                "surrogate_trial": bayes_trial_name,
                "kernel": "baseline_hv",
                "mc_samples": int(sample_count),
                "candidate_pool_n": int(baseline_scored.shape[0]),
                "prediction_time_s": float(bayes_pred_prefix[int(sample_count)]),
                "acquisition_time_s": baseline_acq_time,
                "total_time_s": float(bayes_pred_prefix[int(sample_count)] + baseline_acq_time),
                "selected_overlap_with_analytic": int(baseline_overlap),
                "qpmhi_scored_n": np.nan,
                **baseline_rank_metrics,
            }
        )

        acq_start = time.perf_counter()
        fast_vals = qphv_topk_prob_nd_botorch(sample_prefix, y_train=y_train, top_k=int(args.top_k), weights=weights, ref_point=ref_point)
        fast_acq_time = float(time.perf_counter() - acq_start)
        fast_scored = scored_s.copy()
        fast_scored["score_qpmhi_mc_fast"] = fast_vals.detach().cpu().numpy()
        fast_selected = (
            fast_scored.sort_values(["score_qpmhi_mc_fast", "pred_dock_mean"], ascending=[False, True], kind="mergesort")
            .head(int(args.top_k))
            .reset_index(drop=True)
        )
        fast_selected.to_csv(out_root / f"selected_qpmhi_mc_bnn_fast_S{int(sample_count)}.csv", index=False)
        fast_overlap = len(analytic_selected_smiles & set(fast_selected["smiles_canonical"].astype(str).tolist()))
        fast_rank_metrics = acquisition_ranking_metrics(fast_scored, "score_qpmhi_mc_fast", int(args.top_k))
        rows.append(
            {
                "method": f"qpmhi_mc_bnn_fast_S{int(sample_count)}",
                "surrogate_trial": bayes_trial_name,
                "kernel": "piecewise_fast",
                "mc_samples": int(sample_count),
                "candidate_pool_n": int(fast_scored.shape[0]),
                "prediction_time_s": float(bayes_pred_prefix[int(sample_count)]),
                "acquisition_time_s": fast_acq_time,
                "total_time_s": float(bayes_pred_prefix[int(sample_count)] + fast_acq_time),
                "selected_overlap_with_analytic": int(fast_overlap),
                "qpmhi_scored_n": np.nan,
                **fast_rank_metrics,
            }
        )

    summary = pd.DataFrame(rows)
    analytic_total = float(summary.loc[summary["method"] == "analytic_pomhi", "total_time_s"].iloc[0])
    summary["slowdown_vs_analytic"] = summary["total_time_s"].astype(float) / analytic_total
    summary["analytic_speedup_vs_method"] = summary["slowdown_vs_analytic"]
    summary.to_csv(out_root / "acquisition_speed_summary.csv", index=False)
    meta = {
        "analytic_run_dir": str(analytic_run_dir),
        "bayes_run_dir": str(bayes_run_dir),
        "analytic_trial": analytic_trial_name,
        "bayes_trial": bayes_trial_name,
        "candidate_pool_n_requested": int(args.candidate_pool_size),
        "top_k": int(args.top_k),
        "mc_samples": sample_sizes,
        "quadrature_points": int(args.quadrature_points),
        "device": str(device),
        "rows": summary.to_dict(orient="records"),
    }
    (out_root / "acquisition_speed_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"[acquisition-speed] output_dir={out_root}")
    return 0


def confgen_cfg(candidate_3d_cfg: dict, seed: int) -> dict:
    return {"max_attempts": int(candidate_3d_cfg.get("max_attempts", 3)), "seed": int(seed), "num_confs": int(candidate_3d_cfg.get("num_confs", 4)), "max_opt_iters": int(candidate_3d_cfg.get("max_opt_iters", 100)), "optimize": bool(candidate_3d_cfg.get("optimize", True)), "prefer_mmff": bool(candidate_3d_cfg.get("prefer_mmff", False)), "workers": int(candidate_3d_cfg.get("workers", 1)), "chunksize": int(candidate_3d_cfg.get("chunksize", 16))}


def train_trial_ckpt(dataset_root: Path, ckpt_path: Path, trial: dict, cfg: dict, ligand_vocab: list[str], seed: int, pretrained_encoder_ckpt: Path | None, ligand3d_cache_dir: Path | None) -> None:
    if str(trial["backbone"]) == "gp":
        raise RuntimeError("Virtual closed-loop currently does not support GP trial configs.")
    surrogate_cfg = dict(cfg.get("surrogate") or {})
    candidate_3d_cfg = dict(dict(cfg.get("candidate") or {}).get("candidate_3d") or {})
    cg = confgen_cfg(candidate_3d_cfg, seed)
    general_cfg = dict(cfg.get("general") or {})
    device_name = str(general_cfg.get("device", "auto")).strip().lower()
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    dock_valid_max = dict(cfg.get("objective") or {}).get("dock_valid_max", 0.0)
    dock_valid_max = None if dock_valid_max in (None, "", "null") else float(dock_valid_max)
    train_surrogate_from_scratch(
        root=str(dataset_root), save_path=str(ckpt_path), epochs=int(trial["epochs"]),
        batch_size=int(trial.get("batch_size", surrogate_cfg.get("retrain_batch_size", 32))), lr=float(trial.get("lr", surrogate_cfg.get("retrain_lr", 5.0e-4))),
        weight_decay=float(trial.get("weight_decay", surrogate_cfg.get("retrain_weight_decay", 1.0e-5))), hidden_dim=int(trial["hidden_dim"]), num_layers=int(trial["num_layers"]),
        dropout=float(trial["dropout"]), use_edge_attr=bool(surrogate_cfg.get("retrain_use_edge_attr", True)), use_ligand_mask=bool(surrogate_cfg.get("retrain_use_ligand_mask", True)),
        standardize=bool(surrogate_cfg.get("retrain_standardize", True)), fp_dim=int(surrogate_cfg.get("retrain_fp_dim", 0)), fp_radius=int(surrogate_cfg.get("retrain_fp_radius", 2)),
        eval_samples=int(surrogate_cfg.get("retrain_eval_samples", 8)), scheduler=str(surrogate_cfg.get("retrain_scheduler", "cosine")), warmup_epochs=int(surrogate_cfg.get("retrain_warmup_epochs", 5)),
        min_lr=float(surrogate_cfg.get("retrain_min_lr", 1.0e-5)), early_stop_patience=int(surrogate_cfg.get("retrain_early_stop_patience", 4)), early_stop_min_delta=float(surrogate_cfg.get("retrain_early_stop_min_delta", 1.0e-3)),
        device=device, dock_valid_max=dock_valid_max, backbone=str(trial["backbone"]), torchmd_cfg=dict(trial.get("torchmd_cfg") or {}), ensemble_heads=int(trial["ensemble_heads"]),
        freeze_backbone=bool(trial["freeze_backbone"]), pretrained_encoder_ckpt=(str(pretrained_encoder_ckpt) if pretrained_encoder_ckpt is not None and bool(trial["needs_pretrain"]) else None),
        uncertainty_mode=str(trial["uncertainty_mode"]), ensemble_scheme=str(trial["ensemble_scheme"]), ensemble_bootstrap=bool(surrogate_cfg.get("retrain_ensemble_bootstrap", False)),
        random_seed=int(seed), auto_prepare=False, ligand3d_cache_dir=(None if ligand3d_cache_dir is None else str(ligand3d_cache_dir)), ligand_vocab_override=ligand_vocab,
        ligand3d_store=None, test_ligand_ids=None, confgen_max_attempts=int(cg["max_attempts"]), confgen_seed=int(cg["seed"]), confgen_num_confs=int(cg["num_confs"]),
        confgen_max_opt_iters=int(cg["max_opt_iters"]), confgen_optimize=bool(cg["optimize"]), confgen_prefer_mmff=bool(cg["prefer_mmff"]), ligand3d_num_workers=int(cg["workers"]),
        ligand3d_mp_chunksize=int(cg["chunksize"]), encoder_lr=(None if trial.get("encoder_lr") is None else float(trial.get("encoder_lr"))), train_log_csv=str(ckpt_path.with_suffix(".train_log.csv")),
        train_summary_json=str(ckpt_path.with_suffix(".train_summary.json")),
    )


def score_virtual(scored_method: str, candidate_df: pd.DataFrame, labeled_df: pd.DataFrame, model_bundle: tuple, ligand_vocab: list[str], candidate_3d_cfg: dict, dock_sign: float, qed_sign: float, sa_sign: float, weights: list[float], ref_point: list[float], mc_samples: int, batch_size: int, pred_batch_size: int, seed: int, nparego_scalarizations: int = 8) -> pd.DataFrame:
    model, mean, std, model_node_dim, model_edge_dim, model_atom_extra_dim, model_bond_extra_dim, model_fp_dim, model_fp_radius, _graph, surrogate_kind, surrogate_meta = model_bundle
    device = next(model.parameters()).device
    if scored_method == "analytic_ehvi":
        scored = _score_with_qnehvi(candidate_df, labeled_df, model, mean, std, device, ligand_vocab, surrogate_kind, model_node_dim, model_edge_dim, model_atom_extra_dim, model_bond_extra_dim, model_fp_dim, model_fp_radius, surrogate_meta, int(pred_batch_size), candidate_3d_cfg, dock_sign, qed_sign, sa_sign, weights, ref_point)
        if "score_qnehvi" not in scored.columns:
            raise RuntimeError("analytic_ehvi virtual scoring expected score_qnehvi in scored dataframe.")
        scored = scored.copy()
        scored["score_analytic_ehvi"] = scored["score_qnehvi"]
        return scored
    if scored_method == "analytic_pomhi":
        scored = _score_with_qpmhi_analytic(candidate_df, labeled_df, model, mean, std, device, ligand_vocab, surrogate_kind, model_node_dim, model_edge_dim, model_atom_extra_dim, model_bond_extra_dim, model_fp_dim, model_fp_radius, surrogate_meta, int(pred_batch_size), candidate_3d_cfg, dock_sign, qed_sign, sa_sign, weights, ref_point)
        if "score_qpmhi" not in scored.columns:
            raise RuntimeError("analytic_pomhi virtual scoring expected score_qpmhi in scored dataframe.")
        scored = scored.copy()
        scored["score_analytic_pomhi"] = scored["score_qpmhi"]
        return scored
    uncertainty_mode = str(surrogate_meta.get("uncertainty_mode", getattr(model, "uncertainty_mode", "gaussian"))).strip().lower()
    bayes_samples = None
    if uncertainty_mode == "bayes" and scored_method in {"qnehvi_mc", "qpmhi_mc", "qnparego_mc"}:
        scored, bayes_samples = _bayes_predictive_samples(
            candidate_df,
            model=model,
            mean=float(mean),
            std=float(std),
            device=device,
            ligand_vocab=ligand_vocab,
            surrogate_kind=surrogate_kind,
            model_node_dim=int(model_node_dim),
            model_edge_dim=int(model_edge_dim),
            model_atom_extra_dim=int(model_atom_extra_dim),
            model_bond_extra_dim=int(model_bond_extra_dim),
            model_fp_dim=int(model_fp_dim),
            model_fp_radius=int(model_fp_radius),
            surrogate_meta=surrogate_meta,
            pred_batch_size=int(pred_batch_size),
            candidate_3d_cfg=candidate_3d_cfg,
            dock_sign=dock_sign,
            qed_sign=qed_sign,
            sa_sign=sa_sign,
            num_samples=int(mc_samples),
            seed=int(seed) + 701,
        )
    else:
        scored = _score_with_predictions_only(candidate_df, model, mean, std, device, ligand_vocab, surrogate_kind, model_node_dim, model_edge_dim, model_atom_extra_dim, model_bond_extra_dim, model_fp_dim, model_fp_radius, surrogate_meta, int(pred_batch_size), candidate_3d_cfg)
    if scored_method == "greedy_mean":
        scored["score_greedy_mean"] = mean_hvi_gain(scored, labeled_df, dock_sign, qed_sign, sa_sign, weights, ref_point)
    elif scored_method == "qnehvi_mc":
        samples = bayes_samples if bayes_samples is not None else mc_objective_samples(scored, int(mc_samples), dock_sign, qed_sign, sa_sign, seed + 101)
        y_train = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
        scored["score_qnehvi_mc"] = qnehvi_scores_from_samples_fast_3d(samples, y_train=y_train, weights=weights, ref_point=ref_point).detach().cpu().numpy()
    elif scored_method == "qpmhi_mc":
        samples = bayes_samples if bayes_samples is not None else mc_objective_samples(scored, int(mc_samples), dock_sign, qed_sign, sa_sign, seed + 131)
        y_train = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
        scored["score_qpmhi_mc"] = qphv_topk_prob_nd_botorch(samples, y_train=y_train, top_k=int(batch_size), weights=weights, ref_point=ref_point).detach().cpu().numpy()
    elif scored_method == "qnparego_mc":
        samples = bayes_samples if bayes_samples is not None else mc_objective_samples(scored, int(mc_samples), dock_sign, qed_sign, sa_sign, seed + 151)
        y_train = build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
        scored["score_qnparego_mc"] = qnparego_scores_from_samples_nd(
            samples,
            y_train=y_train,
            weights=weights,
            scalarizations=int(nparego_scalarizations),
            seed=seed + 157,
        ).detach().cpu().numpy()
    return scored


def run_virtual(args: argparse.Namespace) -> int:
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    bad = sorted(set(methods) - VIRTUAL_METHODS)
    if bad:
        raise RuntimeError(f"Unsupported virtual-loop methods for this virtual-loop entrypoint: {bad}.")
    if int(args.nparego_scalarizations) <= 0:
        raise RuntimeError("--nparego-scalarizations must be positive.")
    targets = resolve_targets(args.target, args.dockstring_root, args.split)
    cfg = load_config(args.config)
    reuse_shared_from = _resolve_optional_path(args.reuse_shared_from)
    if reuse_shared_from is not None and not reuse_shared_from.exists():
        raise FileNotFoundError(reuse_shared_from)
    trial_path = Path(args.trial_config)
    if not trial_path.is_absolute():
        trial_path = (REPO / trial_path).resolve()
    trial = _normalize_trial_spec(trial_path.stem, trial_path, load_config(str(trial_path)))
    pretrained_encoder_ckpt = None
    if args.pretrained_encoder_ckpt:
        pretrained_encoder_ckpt = Path(args.pretrained_encoder_ckpt)
        if not pretrained_encoder_ckpt.is_absolute():
            pretrained_encoder_ckpt = (REPO / pretrained_encoder_ckpt).resolve()
        if not pretrained_encoder_ckpt.exists():
            raise FileNotFoundError(pretrained_encoder_ckpt)
    ligand3d_cache_dir = None
    if args.ligand3d_cache_dir:
        ligand3d_cache_dir = Path(args.ligand3d_cache_dir)
        if not ligand3d_cache_dir.is_absolute():
            ligand3d_cache_dir = (REPO / ligand3d_cache_dir).resolve()
        ligand3d_cache_dir.mkdir(parents=True, exist_ok=True)
    objective_cfg = dict(cfg.get("objective") or {})
    selection_cfg = dict(cfg.get("selection") or {})
    candidate_3d_cfg = dict(dict(cfg.get("candidate") or {}).get("candidate_3d") or {})
    candidate_3d_cfg["workers"] = int(args.eval_3d_workers)
    if ligand3d_cache_dir is not None:
        candidate_3d_cfg["cache_dir"] = str(ligand3d_cache_dir)
    dock_sign = float(objective_cfg.get("dock_sign", -1.0))
    qed_sign = float(objective_cfg.get("qed_sign", 1.0))
    sa_sign = float(objective_cfg.get("sa_sign", -1.0))
    weights = list(selection_cfg.get("weights", [1.0, 1.0, 1.0]))
    ref_point = list(objective_cfg.get("ref_point", [0.0, 0.0, -20.0]))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    config_path = resolve_repo_path(args.config)
    trial_config_path = resolve_repo_path(args.trial_config)
    shutil.copy2(config_path, out_root / config_path.name)
    shutil.copy2(trial_config_path, out_root / trial_config_path.name)
    run_meta = {
        "cmd": "virtual-loop",
        "protocol_version": "shared_candidate_pool_v2",
        "config": str(config_path.relative_to(REPO)),
        "trial_config": str(trial_config_path.relative_to(REPO)),
        "dockstring_root": str(Path(args.dockstring_root)),
        "split": str(args.split),
        "targets": {name: str(path) for name, path in targets.items()},
        "methods": list(methods),
        "reuse_shared_from": None if reuse_shared_from is None else str(reuse_shared_from),
        "rounds": int(args.rounds),
        "init_size": int(args.init_size),
        "init_clusters": int(args.init_clusters),
        "init_min_per_cluster": int(args.init_min_per_cluster),
        "candidate_pool_size": int(args.candidate_pool_size),
        "batch_size": int(args.batch_size),
        "mc_samples": int(args.mc_samples),
        "nparego_scalarizations": int(args.nparego_scalarizations),
        "pred_batch_size": int(args.pred_batch_size),
        "eval_3d_workers": int(args.eval_3d_workers),
        "dock_threshold": float(args.dock_threshold),
        "qed_threshold": float(args.qed_threshold),
        "sa_threshold": float(args.sa_threshold),
        "seed": int(args.seed),
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(
        f"[virtual-loop][config] output_dir={out_root} split={args.split} rounds={int(args.rounds)} "
        f"init_size={int(args.init_size)} candidate_pool_size={int(args.candidate_pool_size)} "
        f"batch_size={int(args.batch_size)} methods={','.join(methods)} seed={int(args.seed)}"
    )
    final_rows = []
    for target_name, csv_path in targets.items():
        target_root = out_root / target_name
        meta_path = target_root / "virtual_loop_meta.json"
        expected_meta = {
            "protocol_version": "shared_candidate_pool_v2",
            "target": str(target_name),
            "split": str(args.split),
            "seed": int(args.seed),
            "init_size": int(args.init_size),
            "init_clusters": int(args.init_clusters),
            "init_min_per_cluster": int(args.init_min_per_cluster),
            "candidate_pool_size": int(args.candidate_pool_size),
            "batch_size": int(args.batch_size),
        }
        reset_target = False
        if target_root.exists() and any(target_root.iterdir()):
            if not meta_path.exists():
                reset_target = True
            else:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if existing_meta != expected_meta:
                    reset_target = True
            if reset_target:
                _reset_incomplete_dir(target_root, root=out_root, label="virtual-loop", dry_run=False)
        target_root.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
        if reuse_shared_from is not None:
            _seed_virtual_shared_inputs(target_root, reuse_shared_from / target_name, int(args.rounds))
            print(f"[virtual-loop][reuse-shared] target={target_name} source={reuse_shared_from / target_name}")
        raw_pool_df = pd.read_csv(csv_path)
        if int(args.smoke_max_pool) > 0:
            raw_pool_df = raw_pool_df.head(int(args.smoke_max_pool)).copy().reset_index(drop=True)
        pool_df, pool_property_source = _load_or_build_virtual_pool(raw_pool_df, target_root=target_root, label=target_name)
        print(f"[virtual-loop][pool-props] target={target_name} source={pool_property_source} pool_n={pool_df.shape[0]}")

        cached_init = _load_cached_virtual_init(target_root)
        if cached_init is None:
            init_res = select_fingerprint_init_set(
                pool_df,
                smiles_col="smiles_canonical",
                n_clusters=int(args.init_clusters),
                init_size=int(args.init_size),
                min_per_cluster=int(args.init_min_per_cluster),
                seed=int(args.seed),
            )
            init_df = ensure_pool(init_res.selected.copy().reset_index(drop=True), f"{target_name}_init")
            init_df["added_iter"] = 0
            init_assignments_df = init_res.assignments.copy().reset_index(drop=True)
            init_quotas_df = init_res.quotas.copy().reset_index(drop=True)
            init_df.to_csv(target_root / "init_set.csv", index=False)
            init_assignments_df.to_csv(target_root / "init_assignments.csv", index=False)
            init_quotas_df.to_csv(target_root / "init_quotas.csv", index=False)
            print(f"[virtual-loop][init] target={target_name} source=recomputed init_n={init_df.shape[0]}")
        else:
            init_df, init_assignments_df, init_quotas_df = cached_init
            init_df["added_iter"] = 0
            print(f"[virtual-loop][init] target={target_name} source=cache init_n={init_df.shape[0]}")

        init_ligand_ids = set(init_df["ligand_id"].astype(str).tolist())
        base_unlabeled = pool_df.loc[~pool_df["ligand_id"].astype(str).isin(init_ligand_ids)].copy().reset_index(drop=True)
        print(
            f"[virtual-loop][target] target={target_name} pool_n={pool_df.shape[0]} "
            f"init_n={init_df.shape[0]} unlabeled_n={base_unlabeled.shape[0]} "
            f"init_clusters={int(args.init_clusters)} init_min_per_cluster={int(args.init_min_per_cluster)}"
        )
        reused_ligand_vocab = None
        if reuse_shared_from is not None:
            reused_ligand_vocab = _load_reused_virtual_ligand_vocab(reuse_shared_from / target_name)
        ligand_vocab = reused_ligand_vocab or resolve_ligand_vocab(REPO / "dataset" / target_name, pool_df["smiles_canonical"].astype(str).tolist())
        target_meta = {
            "target": str(target_name),
            "source_csv": str(csv_path),
            "pool_n": int(pool_df.shape[0]),
            "init_n": int(init_df.shape[0]),
            "unlabeled_n": int(base_unlabeled.shape[0]),
            "ligand_vocab_size": int(len(ligand_vocab)),
            "ligand_vocab_source": "reuse_shared_from" if reused_ligand_vocab is not None else "resolved_from_dataset_or_pool",
            "pool_property_source": str(pool_property_source),
            "init_clusters": int(args.init_clusters),
            "init_min_per_cluster": int(args.init_min_per_cluster),
            "seed": int(args.seed),
            "init_source": "cache" if cached_init is not None else "recomputed",
        }
        (target_root / "target_meta.json").write_text(json.dumps(target_meta, indent=2), encoding="utf-8")
        shared_root = target_root / "_shared"
        shared_root.mkdir(parents=True, exist_ok=True)
        init_path = shared_root / "init_set.csv"
        init_df.to_csv(init_path, index=False)
        shared_remaining = base_unlabeled.copy().reset_index(drop=True)
        shared_pool_rows: list[dict] = []
        for round_idx in range(1, int(args.rounds) + 1):
            cand_path = shared_root / f"round{round_idx:02d}_candidate_pool.csv"
            if cand_path.exists():
                cand_df = ensure_pool(pd.read_csv(cand_path), f"{target_name}_shared_round_{round_idx}")
                cand_ligand_ids = set(cand_df["ligand_id"].astype(str).tolist())
                shared_pool_rows.append(
                    {
                        "round": int(round_idx),
                        "candidate_n": int(cand_df.shape[0]),
                        "remaining_after_round": int(shared_remaining.shape[0] - cand_df.shape[0]),
                        "candidate_pool_csv": str(cand_path.relative_to(out_root)),
                    }
                )
                shared_remaining = shared_remaining.loc[
                    ~shared_remaining["ligand_id"].astype(str).isin(cand_ligand_ids)
                ].copy().reset_index(drop=True)
                continue
            if shared_remaining.empty:
                raise RuntimeError(f"Shared candidate pool exhausted before round {round_idx}: {target_root}")
            shared_seed = int(args.seed) + sum(ord(ch) for ch in f"{target_name}_shared_round_{round_idx}")
            take_n = min(int(args.candidate_pool_size), int(shared_remaining.shape[0]))
            shared_rng = random.Random(shared_seed)
            cand_idx = sorted(shared_rng.sample(range(int(shared_remaining.shape[0])), take_n))
            cand_df = shared_remaining.iloc[cand_idx].copy().reset_index(drop=True)
            cand_df.to_csv(cand_path, index=False)
            shared_pool_rows.append(
                {
                    "round": int(round_idx),
                    "candidate_n": int(cand_df.shape[0]),
                    "remaining_after_round": int(shared_remaining.shape[0] - cand_df.shape[0]),
                    "candidate_pool_csv": str(cand_path.relative_to(out_root)),
                }
            )
            shared_remaining = shared_remaining.loc[
                ~shared_remaining["ligand_id"].astype(str).isin(set(cand_df["ligand_id"].astype(str).tolist()))
            ].copy().reset_index(drop=True)
        pd.DataFrame(shared_pool_rows).to_csv(target_root / "shared_candidate_pool_summary.csv", index=False)

        complete_methods: dict[str, pd.DataFrame] = {}
        active_methods: list[str] = []
        resume_trajs: dict[str, pd.DataFrame] = {}
        completed_round_by_method: dict[str, int] = {}
        for method in methods:
            run_out = target_root / method
            status, existing_traj = _check_virtual_run_dir(run_out, int(args.rounds))
            if status == "complete":
                if existing_traj is None or existing_traj.empty:
                    raise RuntimeError(f"Expected complete trajectory for {run_out}")
                complete_methods[method] = existing_traj
                continue
            if status == "incomplete" and existing_traj is not None and not existing_traj.empty:
                completed_rounds = _completed_virtual_rounds(run_out, existing_traj, int(args.rounds))
                _cleanup_virtual_round_artifacts(run_out, from_round=completed_rounds + 1, expected_rounds=int(args.rounds))
                active_methods.append(method)
                resume_trajs[method] = existing_traj.copy().reset_index(drop=True)
                completed_round_by_method[method] = int(completed_rounds)
                print(
                    f"[virtual-loop][resume] target={target_name} method={method} "
                    f"completed_rounds={completed_rounds} output_dir={run_out}"
                )
            else:
                if run_out.exists() and any(run_out.iterdir()):
                    _reset_incomplete_dir(run_out, root=target_root, label="virtual-loop method", dry_run=False)
                active_methods.append(method)
                completed_round_by_method[method] = 0

        for method, existing_traj in complete_methods.items():
            final = dict(existing_traj.iloc[-1].to_dict())
            final["trial_name"] = str(trial["trial_name"])
            final_rows.append(final)

        if not active_methods:
            print(f"[virtual-loop][skip] target={target_name} output_dir={target_root}")
            continue

        method_states: dict[str, pd.DataFrame] = {}
        method_rows: dict[str, list[dict]] = {}
        method_run_seed: dict[str, int] = {}
        for method in active_methods:
            completed_rounds = int(completed_round_by_method[method])
            run_out = target_root / method
            if completed_rounds > 0:
                existing_traj = resume_trajs[method]
                method_states[method] = _load_virtual_method_state_from_selected(init_df, run_out, completed_rounds, existing_traj)
                method_rows[method] = existing_traj.to_dict("records")
            else:
                method_states[method] = init_df.copy().reset_index(drop=True)
                method_rows[method] = []
            method_run_seed[method] = int(args.seed) + sum(ord(ch) for ch in f"{target_name}_{method}")
            run_out.mkdir(parents=True, exist_ok=True)

        shared_unlabeled_n = int(base_unlabeled.shape[0])
        for round_idx in range(1, int(args.rounds) + 1):
            cand_path = shared_root / f"round{round_idx:02d}_candidate_pool.csv"
            cand_df = ensure_pool(pd.read_csv(cand_path), f"{target_name}_shared_round_{round_idx}")
            shared_unlabeled_n -= int(cand_df.shape[0])
            print(
                f"[virtual-loop][round-start] target={target_name} round={round_idx} "
                f"shared_candidate_n={cand_df.shape[0]} shared_remaining_n={shared_unlabeled_n}"
            )
            for method in active_methods:
                if round_idx <= int(completed_round_by_method[method]):
                    continue
                run_seed = method_run_seed[method]
                labeled_df = method_states[method]
                run_out = target_root / method
                print(
                    f"[virtual-loop][method-start] target={target_name} method={method} round={round_idx} "
                    f"labeled_n={labeled_df.shape[0]}"
                )
                ds_root = run_out / f"round{round_idx:02d}_dataset"
                write_round_dataset(labeled_df, ds_root, valid_frac=float(args.valid_frac), seed=run_seed + round_idx, smiles_col="smiles_canonical")
                (ds_root / "ligand_vocab.json").write_text(json.dumps(ligand_vocab, indent=2), encoding="utf-8")
                ckpt_path = run_out / f"round{round_idx:02d}_surrogate.pt"
                t_train = time.perf_counter()
                train_trial_ckpt(ds_root, ckpt_path, trial, cfg, ligand_vocab, run_seed + round_idx, pretrained_encoder_ckpt, ligand3d_cache_dir)
                train_sec = float(time.perf_counter() - t_train)
                device_name = str(dict(cfg.get("general") or {}).get("device", "auto")).strip().lower()
                device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
                bundle = load_surrogate(str(ckpt_path), device=device)
                t_score = time.perf_counter()
                scored = score_virtual(method, cand_df, labeled_df, bundle, ligand_vocab, candidate_3d_cfg, dock_sign, qed_sign, sa_sign, weights, ref_point, int(args.mc_samples), int(args.batch_size), int(args.pred_batch_size), run_seed + round_idx, int(args.nparego_scalarizations))
                score_sec = float(time.perf_counter() - t_score)
                pred_eval = eval_prediction_frame(scored)
                scored.to_csv(run_out / f"round{round_idx:02d}_candidate_predictions.csv", index=False)
                picked = select_topk(scored, method, int(args.batch_size), run_seed + round_idx)
                picked_ligand_ids = set(picked["ligand_id"].astype(str).tolist())
                picked_true = cand_df.loc[cand_df["ligand_id"].astype(str).isin(picked_ligand_ids)].copy().reset_index(drop=True)
                if picked_true.shape[0] != picked.shape[0]:
                    raise RuntimeError("Selected batch could not be matched back to shared candidate rows.")
                picked_true["added_iter"] = int(round_idx)
                labeled_after = pd.concat([labeled_df, picked_true], ignore_index=True)
                picked_true.to_csv(run_out / f"round{round_idx:02d}_selected.csv", index=False)
                hv_after = compute_hypervolume(
                    build_objective_tensor(labeled_after, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign),
                    ref_point=ref_point,
                )
                row = {
                    "target": target_name,
                    "method": method,
                    "round": int(round_idx),
                    "candidate_n": int(cand_df.shape[0]),
                    "selected_n": int(picked_true.shape[0]),
                    "init_pool_n": int(init_df.shape[0]),
                    "hv": float(hv_after),
                    "best_dock": float(labeled_after["dock_score"].min()),
                    "mean_selected_dock": float(picked_true["dock_score"].mean()),
                    "threshold_rate": float(threshold_mask(picked_true, float(args.dock_threshold), float(args.qed_threshold), float(args.sa_threshold)).mean()),
                    "train_seconds": train_sec,
                    "score_seconds": score_sec,
                    "labeled_n": int(labeled_after.shape[0]),
                    "unlabeled_n": int(shared_unlabeled_n),
                }
                row.update(pred_eval)
                method_rows[method].append(row)
                method_states[method] = labeled_after
                pd.DataFrame(method_rows[method]).to_csv(run_out / "trajectory.csv", index=False)
                print(
                    f"[virtual-loop] target={target_name} method={method} round={round_idx} "
                    f"hv={row['hv']:.4f} best_dock={row['best_dock']:.4f} "
                    f"cand_spearman={row['candidate_test_spearman']:.4f} cand_rmse={row['candidate_test_rmse']:.4f} "
                    f"cand_nll={row['candidate_test_nll']:.4f} mean_selected_dock={row['mean_selected_dock']:.4f} "
                    f"threshold_rate={row['threshold_rate']:.4f}"
                )
        for method in active_methods:
            traj = pd.DataFrame(method_rows[method])
            traj.to_csv(target_root / method / "trajectory.csv", index=False)
            if not traj.empty:
                final = dict(traj.iloc[-1].to_dict())
                final["trial_name"] = str(trial["trial_name"])
                final_rows.append(final)
        method_summary_rows = []
        for method in methods:
            traj_path = target_root / method / "trajectory.csv"
            if (not traj_path.exists()) or traj_path.stat().st_size == 0:
                continue
            try:
                traj = pd.read_csv(traj_path)
            except pd.errors.EmptyDataError:
                continue
            if traj.empty:
                continue
            row = dict(traj.iloc[-1].to_dict())
            row["trial_name"] = str(trial["trial_name"])
            method_summary_rows.append(row)
        pd.DataFrame(method_summary_rows).to_csv(target_root / "method_summary.csv", index=False)
    pd.DataFrame(final_rows).to_csv(out_root / "virtual_loop_summary.csv", index=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Dockstring surrogate and finite-pool active-screening experiments.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("surrogate", help="Run surrogate-family ablation.")
    s.add_argument("--config", default="config/surrogate/config_initial_benchmark.yaml")
    s.add_argument("--target", action="append", default=[], help="NAME=CSV_PATH")
    s.add_argument("--dockstring-root", default=str(DEFAULT_DOCKSTRING_ROOT))
    s.add_argument("--split", choices=["train", "test"], default="train", help="For surrogate benchmark this must remain train; evaluation always uses the full official test split.")
    s.add_argument("--output-dir", default=str(DEFAULT_OUT / "surrogate"))
    s.add_argument("--seeds", default="42,43,44")
    s.add_argument("--init-size", type=int, default=8000)
    s.add_argument("--init-clusters", type=int, default=64)
    s.add_argument("--init-min-per-cluster", type=int, default=8)
    s.add_argument("--holdout-frac", type=float, default=0.2, help="Compatibility flag; surrogate benchmark uses official train/test splits.")
    s.add_argument("--valid-frac", type=float, default=0.2)
    s.add_argument("--pred-batch-size", type=int, default=256)
    s.add_argument("--eval-3d-workers", type=int, default=1)
    s.add_argument("--pretrain-csv", default="")
    s.add_argument("--pretrain-smiles-col", default="canonical_smiles")
    s.add_argument("--pretrain-valid-frac", type=float, default=0.1)
    s.add_argument("--pretrain-epochs", type=int, default=None)
    s.add_argument("--skip-gin-small", action="store_true")
    s.add_argument("--skip-scratch", action="store_true")
    s.add_argument("--skip-pretrained", action="store_true")
    s.add_argument("--skip-ensemble", action="store_true")
    s.add_argument("--python-exe", default="")
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("ranking", help="Run frozen-pool acquisition ranking.")
    r.add_argument("--config", default="config/surrogate/config_initial_benchmark.yaml")
    r.add_argument("--run-dir", action="append", required=True)
    r.add_argument("--output-dir", default=str(DEFAULT_OUT / "ranking"))
    r.add_argument("--trial-name", default="")
    r.add_argument("--methods", default="random,greedy_mean,qnehvi_mc,qpmhi_mc,qnparego_mc,analytic_ehvi,analytic_pomhi,pio")
    r.add_argument("--top-k", type=int, default=50)
    r.add_argument("--candidate-pool-size", type=int, default=3000)
    r.add_argument("--mc-samples", type=int, default=128)
    r.add_argument("--nparego-scalarizations", type=int, default=8)
    r.add_argument("--pred-batch-size", type=int, default=256)
    r.add_argument("--eval-3d-workers", type=int, default=1)
    r.add_argument("--dock-threshold", type=float, default=-10.0)
    r.add_argument("--qed-threshold", type=float, default=0.5)
    r.add_argument("--sa-threshold", type=float, default=4.0)
    r.add_argument("--seed", type=int, default=42)
    b = sub.add_parser("acquisition-speed", help="Benchmark analytic qPMHI against BNN posterior-sampling qPMHI on one fixed pool.")
    b.add_argument("--config", default="config/surrogate/config_initial_benchmark.yaml")
    b.add_argument("--analytic-run-dir", required=True)
    b.add_argument("--bayes-run-dir", required=True)
    b.add_argument("--output-dir", default=str(DEFAULT_OUT / "acquisition_speed"))
    b.add_argument("--analytic-trial-name", default="tensornet_scratch_gaussian")
    b.add_argument("--bayes-trial-name", default="tensornet_scratch_bayes_mlp")
    b.add_argument("--top-k", type=int, default=50)
    b.add_argument("--candidate-pool-size", type=int, default=2000)
    b.add_argument("--mc-samples", type=int, nargs="+", default=[64, 128, 256, 512])
    b.add_argument("--quadrature-points", type=int, default=16)
    b.add_argument("--pred-batch-size", type=int, default=256)
    b.add_argument("--eval-3d-workers", type=int, default=1)
    b.add_argument("--dock-threshold", type=float, default=-10.0)
    b.add_argument("--qed-threshold", type=float, default=0.5)
    b.add_argument("--sa-threshold", type=float, default=4.0)
    b.add_argument("--seed", type=int, default=42)
    v = sub.add_parser("virtual-loop", help="Run DOCKSTRING virtual closed-loop without PIO.")
    v.add_argument("--config", default="config/surrogate/config_initial_benchmark.yaml")
    v.add_argument("--trial-config", required=True)
    v.add_argument("--target", action="append", default=[], help="Either TARGET_NAME or NAME=CSV_PATH")
    v.add_argument("--dockstring-root", default=str(DEFAULT_DOCKSTRING_ROOT))
    v.add_argument("--split", choices=["train", "test", "full"], default="full")
    v.add_argument("--output-dir", default=str(DEFAULT_OUT / "virtual_loop"))
    v.add_argument("--reuse-shared-from", default="", help="Existing virtual-loop run root whose target/_shared pools and init files should be reused.")
    v.add_argument("--pretrained-encoder-ckpt", default="")
    v.add_argument("--ligand3d-cache-dir", default="")
    v.add_argument("--methods", default="random,greedy_mean,qnehvi_mc,qpmhi_mc,qnparego_mc,analytic_ehvi,analytic_pomhi")
    v.add_argument("--rounds", type=int, default=5)
    v.add_argument("--init-size", type=int, default=500)
    v.add_argument("--init-clusters", type=int, default=16)
    v.add_argument("--init-min-per-cluster", type=int, default=3)
    v.add_argument("--candidate-pool-size", type=int, default=2000)
    v.add_argument("--batch-size", type=int, default=100)
    v.add_argument("--valid-frac", type=float, default=0.2)
    v.add_argument("--mc-samples", type=int, default=128)
    v.add_argument("--nparego-scalarizations", type=int, default=8)
    v.add_argument("--pred-batch-size", type=int, default=256)
    v.add_argument("--eval-3d-workers", type=int, default=1)
    v.add_argument("--dock-threshold", type=float, default=-10.0)
    v.add_argument("--qed-threshold", type=float, default=0.5)
    v.add_argument("--sa-threshold", type=float, default=4.0)
    v.add_argument("--smoke-max-pool", type=int, default=0)
    v.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(int(getattr(args, "seed", 42)))
    if args.cmd == "surrogate":
        return run_surrogate(args)
    if args.cmd == "ranking":
        return run_ranking(args)
    if args.cmd == "acquisition-speed":
        return run_acquisition_speed(args)
    if args.cmd == "virtual-loop":
        return run_virtual(args)
    raise RuntimeError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
