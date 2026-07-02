#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_vae_bo_holdout_init import _load_holdout_df, _load_init_pool_df
from train_initial_surrogate_sweep import (
    _build_pretrain_torchmd_cfg,
    _device_from_cfg,
    _prepare_loader,
    _prepare_pretrain_dataset_from_csv,
    _run_tensornet_fp_pretrain,
    _section,
)

from data.ligand_only_3d_dataset import GLOBAL_LIGAND3D_CACHE_DIR, LigandOnly3DStore
from mobo.config_utils import load_config
from mobo.constants import ATOM_EXTRA_DIM
from mobo.init_selection import select_fingerprint_init_set
from mobo.io_utils import _load_ligand_vocab_override, _pick_smiles_column, _scan_ligand_vocab
from mobo.retrospective import rank_metrics, stratified_holdout_split, write_round_dataset
from mobo.smiles_utils import compute_qed, compute_sa
from mobo.surrogate import load_surrogate, train_surrogate_from_scratch
from train_gin_surrogate import (
    gaussian_variance_from_raw,
    nig_variances_from_params,
    split_gaussian_output,
    split_nig_output,
)

EXPECTED_COVERAGE_1SIGMA = 0.6826894921370859
EXPECTED_COVERAGE_2SIGMA = 0.9544997361036416


def _parse_bool_token(raw: str) -> bool:
    mapping = {
        "1": True,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
    }
    token = str(raw).strip().lower()
    if token not in mapping:
        raise ValueError(f"Unsupported boolean token '{raw}'. Use true/false.")
    return mapping[token]


def _parse_csv_list(raw: str) -> list[str]:
    vals = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma separated list.")
    return vals


def _parse_bool_list(raw: str) -> list[bool]:
    return [_parse_bool_token(item) for item in _parse_csv_list(raw)]


def _safe_corr(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, method: str = "pearson") -> float:
    xs = pd.to_numeric(pd.Series(x), errors="coerce")
    ys = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    xs = xs.loc[mask]
    ys = ys.loc[mask]
    if np.isclose(float(xs.std(ddof=0)), 0.0) or np.isclose(float(ys.std(ddof=0)), 0.0):
        return float("nan")
    return float(xs.corr(ys, method=method))


def _build_morgan_fp_matrix(smiles_list: Sequence[str], *, fp_bits: int, fp_radius: int) -> np.ndarray:
    if int(fp_bits) <= 0:
        raise RuntimeError("GP baseline requires fp_bits > 0.")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(fp_radius), fpSize=int(fp_bits))
    rows: list[np.ndarray] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise RuntimeError(f"Invalid SMILES for GP fingerprint baseline: {smi}")
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((int(fp_bits),), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    if not rows:
        raise RuntimeError("No fingerprints were built for the GP baseline.")
    return np.stack(rows, axis=0)


def _fit_docking_gp_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.utils.transforms import normalize
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except Exception as exc:
        raise RuntimeError("Fingerprint GP baseline requires botorch and gpytorch in the current environment.") from exc

    train_x = torch.as_tensor(x_train, dtype=torch.float64, device=device)
    train_y_raw = torch.as_tensor(y_train, dtype=torch.float64, device=device).view(-1, 1)
    test_x = torch.as_tensor(x_test, dtype=torch.float64, device=device)

    bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])
    span = (bounds[1] - bounds[0]).clamp_min(1e-6)
    norm_bounds = torch.stack([bounds[0], bounds[0] + span])
    train_x_norm = normalize(train_x, norm_bounds)
    test_x_norm = normalize(test_x, norm_bounds)

    y_mean = train_y_raw.mean(dim=0, keepdim=True)
    y_std = train_y_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_y = (train_y_raw - y_mean) / y_std

    model = SingleTaskGP(train_x_norm, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    with torch.no_grad():
        posterior = model.posterior(test_x_norm)
        mean = (posterior.mean * y_std) + y_mean
        std = posterior.variance.clamp_min(1e-12).sqrt() * y_std

    mean_np = mean.detach().cpu().numpy().reshape(-1)
    std_np = std.detach().cpu().numpy().reshape(-1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean_np, std_np


def _nig_nll_terms(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    two_blambda = 2.0 * beta * (1.0 + nu)
    err2 = (target - gamma) ** 2
    return (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(two_blambda)
        + (alpha + 0.5) * torch.log(nu * err2 + two_blambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )


def _ensure_required_columns(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    work = df.copy()
    if "smiles_canonical" not in work.columns:
        if "smiles" in work.columns:
            work["smiles_canonical"] = work["smiles"].astype(str)
        else:
            raise RuntimeError(f"{label} is missing smiles_canonical/smiles columns.")
    if "dock_score" not in work.columns:
        raise RuntimeError(f"{label} is missing dock_score column.")
    smiles = work["smiles_canonical"].astype(str).str.strip()
    if (smiles == "").any():
        raise RuntimeError(f"{label} contains empty smiles_canonical values.")
    dock = pd.to_numeric(work["dock_score"], errors="coerce")
    if dock.isna().any():
        raise RuntimeError(f"{label} contains non-numeric dock_score values.")
    work["smiles_canonical"] = smiles
    work["dock_score"] = dock.astype(float)
    if "qed" not in work.columns:
        work["qed"] = work["smiles_canonical"].map(compute_qed)
    else:
        work["qed"] = pd.to_numeric(work["qed"], errors="coerce")
    if "sa_score" not in work.columns:
        work["sa_score"] = work["smiles_canonical"].map(compute_sa)
    else:
        work["sa_score"] = pd.to_numeric(work["sa_score"], errors="coerce")
    if work["qed"].isna().any():
        raise RuntimeError(f"{label} contains non-finite QED values.")
    if work["sa_score"].isna().any():
        raise RuntimeError(f"{label} contains non-finite SA values.")
    return work.reset_index(drop=True)


def _standardize_init_frame(df: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    work = _ensure_required_columns(df, label=prefix)
    if "ligand_id" not in work.columns:
        work["ligand_id"] = [f"{prefix}_{idx + 1:05d}" for idx in range(work.shape[0])]
    else:
        work["ligand_id"] = work["ligand_id"].astype(str)
    work["added_iter"] = 0
    return work.reset_index(drop=True)


def _resolve_ligand3d_cache_dir(args: argparse.Namespace) -> Path:
    cache_dir = Path(args.ligand3d_cache_dir) if args.ligand3d_cache_dir else GLOBAL_LIGAND3D_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _resolve_confgen_cfg(args: argparse.Namespace, candidate_3d_cfg: dict) -> dict:
    return {
        "max_attempts": int(candidate_3d_cfg.get("max_attempts", 3)),
        "seed": int(args.confgen_seed) if args.confgen_seed is not None else int(candidate_3d_cfg.get("seed", args.seed)),
        "num_confs": int(candidate_3d_cfg.get("num_confs", 4)),
        "max_opt_iters": int(candidate_3d_cfg.get("max_opt_iters", 100)),
        "optimize": bool(candidate_3d_cfg.get("optimize", True)),
        "prefer_mmff": bool(candidate_3d_cfg.get("prefer_mmff", False)),
        "workers": int(candidate_3d_cfg.get("workers", 1)),
        "chunksize": int(candidate_3d_cfg.get("chunksize", 16)),
    }


def _merge_ligand_vocab(base_vocab: Sequence[str] | None, extra_vocab: Sequence[str] | None) -> list[str]:
    merged = {str(x).strip() for x in (base_vocab or []) if str(x).strip()}
    merged.update(str(x).strip() for x in (extra_vocab or []) if str(x).strip())
    out = sorted(x for x in merged if x)
    if not out:
        raise RuntimeError("Resolved ligand vocab is empty.")
    return out


def _scan_vocab_from_smiles_csv(dataset_root: Path) -> list[str]:
    csv_path = dataset_root / "smiles.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing smiles.csv for vocab scan: {csv_path}")
    df = pd.read_csv(csv_path)
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        raise RuntimeError(f"No smiles column found in {csv_path}")
    return _scan_ligand_vocab(df[smiles_col].astype(str).tolist())


def _resolve_ligand_vocab(
    *,
    init_dataset_root: Path | None,
    init_smiles: Sequence[str],
    pretrain_dataset_root: Path | None,
) -> list[str]:
    base_vocab: list[str] | None = None
    if init_dataset_root is not None:
        base_vocab = _load_ligand_vocab_override(init_dataset_root)
    # If the source dataset already ships a ligand vocab file, treat it as authoritative.
    # This keeps atom_feat_dim aligned with existing 3D cache templates.
    if base_vocab:
        return sorted(base_vocab)
    base_vocab = _scan_ligand_vocab(list(init_smiles))
    extra_vocab: list[str] | None = None
    if pretrain_dataset_root is not None:
        extra_vocab = _load_ligand_vocab_override(pretrain_dataset_root)
        if not extra_vocab:
            extra_vocab = _scan_vocab_from_smiles_csv(pretrain_dataset_root)
    return _merge_ligand_vocab(base_vocab, extra_vocab)


def _resolve_pretrain_dataset_root(args: argparse.Namespace, out_dir: Path) -> Path | None:
    dataset_root = args.pretrain_dataset_root
    csv_path = args.pretrain_csv
    if dataset_root and csv_path:
        raise RuntimeError("Use either --pretrain-dataset-root or --pretrain-csv, not both.")
    if dataset_root:
        root = Path(dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"Pretrain dataset root not found: {root}")
        return root
    if csv_path:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Pretrain CSV not found: {csv_file}")
        return _prepare_pretrain_dataset_from_csv(
            csv_path=csv_file,
            smiles_col=str(args.pretrain_smiles_col),
            valid_frac=float(args.pretrain_valid_frac),
            seed=int(args.seed),
            out_root=out_dir / "pretrain_dataset",
        )
    return None


def _checkpoint_node_feat_dim(ckpt_path: Path) -> int | None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    torchmd_cfg = dict(cfg.get("torchmd") or {})
    val = torchmd_cfg.get("node_feat_dim", cfg.get("node_feat_dim"))
    return None if val is None else int(val)


def _resolve_maybe_relative_path(raw_path: str | Path, *bases: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (REPO / path).resolve()


def _resolve_iter0_surrogate_ckpt(init_dataset_root: Path | None) -> Path | None:
    if init_dataset_root is None:
        return None
    candidate = init_dataset_root.resolve().parent / "checkpoints" / "surrogate_iter000.pt"
    return candidate.resolve() if candidate.exists() else None


def _load_iter0_surrogate_config(init_dataset_root: Path | None) -> tuple[dict[str, object] | None, Path | None]:
    ckpt_path = _resolve_iter0_surrogate_ckpt(init_dataset_root)
    if ckpt_path is None:
        return None, None
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    if not cfg:
        raise RuntimeError(f"surrogate_iter000 checkpoint is missing config: {ckpt_path}")
    return cfg, ckpt_path



def _resolve_existing_iter0_encoder_ckpt(
    *,
    init_dataset_root: Path | None,
    expected_node_feat_dim: int,
) -> tuple[Path | None, dict[str, object]]:
    info: dict[str, object] = {
        "reused_existing_pretrain": False,
        "reuse_source": None,
        "reuse_candidate_count": 0,
        "reuse_rejections": [],
    }
    if init_dataset_root is None:
        return None, info

    dataset_root = init_dataset_root.resolve()
    method_root = dataset_root.parent
    run_root = method_root.parent
    candidates: list[tuple[str, Path]] = []

    iter0_surrogate_ckpt = _resolve_iter0_surrogate_ckpt(init_dataset_root)
    if iter0_surrogate_ckpt is not None and iter0_surrogate_ckpt.exists():
        surrogate_ckpt = torch.load(str(iter0_surrogate_ckpt), map_location="cpu", weights_only=False)
        surrogate_cfg = dict(surrogate_ckpt.get("config") or {})
        pretrained_encoder = surrogate_cfg.get("pretrained_encoder_ckpt")
        if pretrained_encoder:
            candidates.append((
                "iter000_surrogate.pretrained_encoder_ckpt",
                _resolve_maybe_relative_path(str(pretrained_encoder), REPO, method_root, run_root),
            ))

    direct_iter0_encoder = run_root / "round_encoder_pretrain" / "encoder_iter000.pt"
    if direct_iter0_encoder.exists():
        candidates.append(("round_encoder_pretrain.encoder_iter000", direct_iter0_encoder.resolve()))

    experiment_meta = run_root / "experiment_meta.json"
    if experiment_meta.exists():
        meta = json.loads(experiment_meta.read_text(encoding="utf-8"))
        init_encoder_ckpt = meta.get("init_encoder_ckpt")
        if init_encoder_ckpt:
            candidates.append((
                "experiment_meta.init_encoder_ckpt",
                _resolve_maybe_relative_path(str(init_encoder_ckpt), REPO, method_root, run_root),
            ))

    deduped: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for source_name, candidate_path in candidates:
        resolved = candidate_path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((source_name, resolved))

    info["reuse_candidate_count"] = int(len(deduped))
    rejection_rows: list[dict[str, object]] = []
    for source_name, candidate_path in deduped:
        if not candidate_path.exists():
            rejection_rows.append({
                "source": source_name,
                "path": str(candidate_path),
                "reason": "missing",
            })
            continue
        ckpt_node_feat_dim = _checkpoint_node_feat_dim(candidate_path)
        if ckpt_node_feat_dim != int(expected_node_feat_dim):
            rejection_rows.append({
                "source": source_name,
                "path": str(candidate_path),
                "reason": "node_feat_dim_mismatch",
                "node_feat_dim": None if ckpt_node_feat_dim is None else int(ckpt_node_feat_dim),
                "expected_node_feat_dim": int(expected_node_feat_dim),
            })
            continue
        info["reused_existing_pretrain"] = True
        info["reuse_source"] = source_name
        info["reuse_rejections"] = rejection_rows
        return candidate_path, info

    info["reuse_rejections"] = rejection_rows
    return None, info


def _prepare_benchmark_data(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    dock_valid_max: float | None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict]:
    if bool(args.pool_csv) == bool(args.init_dataset_root):
        raise RuntimeError("Use exactly one of --pool-csv or --init-dataset-root.")

    if args.pool_csv:
        pool = pd.read_csv(args.pool_csv)
        pool = _ensure_required_columns(pool, label="pool_csv")
        active_df, holdout_df = stratified_holdout_split(pool, holdout_frac=float(args.holdout_frac), seed=int(args.seed))
        init_result = select_fingerprint_init_set(
            active_df,
            smiles_col="smiles_canonical",
            n_clusters=int(args.init_clusters),
            init_size=int(args.init_size),
            min_per_cluster=int(args.init_min_per_cluster),
            seed=int(args.seed),
        )
        init_df = _standardize_init_frame(init_result.selected, prefix="INIT")
        holdout_df = _standardize_init_frame(holdout_df, prefix="HOLDOUT")
        active_df = _standardize_init_frame(active_df, prefix="ACTIVE")
        active_df.to_csv(out_dir / "active_pool.csv", index=False)
        init_df.to_csv(out_dir / "init_set.csv", index=False)
        holdout_df.to_csv(out_dir / "holdout_pool.csv", index=False)
        init_result.assignments.to_csv(out_dir / "init_assignments.csv", index=False)
        init_result.quotas.to_csv(out_dir / "init_quotas.csv", index=False)
        source_meta = {
            "source_mode": "pool_csv",
            "pool_csv": str(Path(args.pool_csv).resolve()),
            "n_pool_total": int(pool.shape[0]),
            "n_active": int(active_df.shape[0]),
            "n_init": int(init_df.shape[0]),
            "n_holdout": int(holdout_df.shape[0]),
        }
    else:
        dataset_root = Path(args.init_dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Init dataset root not found: {dataset_root}")
        init_df = _load_init_pool_df(dataset_root, dock_valid_max=dock_valid_max)
        holdout_df = _load_holdout_df(dataset_root, dock_valid_max=dock_valid_max)
        init_df = _standardize_init_frame(init_df, prefix="INIT")
        holdout_df = _standardize_init_frame(holdout_df, prefix="HOLDOUT")
        init_df.to_csv(out_dir / "init_set.csv", index=False)
        holdout_df.to_csv(out_dir / "holdout_pool.csv", index=False)
        source_meta = {
            "source_mode": "init_dataset_root",
            "init_dataset_root": str(dataset_root.resolve()),
            "n_init": int(init_df.shape[0]),
            "n_holdout": int(holdout_df.shape[0]),
        }

    dataset_root = write_round_dataset(init_df, out_dir / "dataset_init", valid_frac=float(args.valid_frac), seed=int(args.seed))
    source_info = dict(source_meta)
    source_info["init_source_dataset_root"] = str(Path(args.init_dataset_root).resolve()) if args.init_dataset_root else None
    return dataset_root, init_df, holdout_df, source_info

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


def _build_trials(args: argparse.Namespace, *, main_config_path: Path, surrogate_cfg: dict) -> list[dict]:
    model_map = dict(surrogate_cfg.get("benchmark_initial_models") or {})
    if not model_map:
        raise RuntimeError("surrogate.benchmark_initial_models is required and cannot be empty.")
    trials: list[dict] = []
    seen_trial_names: set[str] = set()
    for name, raw_path in model_map.items():
        cfg_path = _resolve_config_path_from_main(main_config_path, raw_path)
        trial = _normalize_trial_spec(str(name), cfg_path, load_config(str(cfg_path)))
        trial_name = str(trial["trial_name"])
        if trial_name in seen_trial_names:
            raise RuntimeError(f"Duplicate trial_name '{trial_name}' in benchmark model configs")
        seen_trial_names.add(trial_name)
        if args.skip_gin_small and trial_name == "gin_small_gaussian":
            continue
        if args.skip_scratch and "_scratch_" in trial_name:
            continue
        if args.skip_pretrained and bool(trial["needs_pretrain"]) and int(trial["ensemble_heads"]) == 1:
            continue
        if args.skip_ensemble and int(trial["ensemble_heads"]) > 1:
            continue
        trials.append(trial)
    if not trials:
        raise RuntimeError("No trials configured.")
    return trials


def _resolve_shared_pretrain_torchmd_cfg(trials: Sequence[dict], surrogate_cfg: dict) -> dict | None:
    pretrain_trials = [trial for trial in trials if bool(trial.get("needs_pretrain"))]
    if not pretrain_trials:
        return None
    torchmd_cfgs: list[tuple[str, dict]] = []
    for trial in pretrain_trials:
        if str(trial.get("backbone")) != "tensornet":
            raise RuntimeError(f"Pretrained trial {trial['trial_name']} must use tensornet backbone")
        cfg = dict(trial.get("torchmd_cfg") or {})
        if not cfg:
            raise RuntimeError(f"Pretrained trial {trial['trial_name']} is missing torchmd_cfg")
        pretrain_cfg = _build_pretrain_torchmd_cfg(cfg, surrogate_cfg)
        torchmd_cfgs.append((str(trial["trial_name"]), pretrain_cfg))
    ref_name, ref_cfg = torchmd_cfgs[0]
    ref_json = json.dumps(ref_cfg, sort_keys=True)
    for trial_name, cfg in torchmd_cfgs[1:]:
        if json.dumps(cfg, sort_keys=True) != ref_json:
            raise RuntimeError(
                f"Pretrained trials require identical encoder-pretrain torchmd configs, but {ref_name} and {trial_name} differ."
            )
    return ref_cfg


def _is_completed_trial_dir(trial_dir: Path) -> bool:
    required = (
        trial_dir / "trial_metrics.json",
        trial_dir / "holdout_predictions.csv",
        trial_dir / "train_summary.json",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def _load_completed_trial_metrics(trial_dir: Path, *, expected_trial_name: str) -> dict:
    metrics_path = trial_dir / "trial_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    row = json.loads(metrics_path.read_text(encoding="utf-8"))
    actual_trial_name = str(row.get("trial_name") or "").strip()
    if actual_trial_name != str(expected_trial_name):
        raise RuntimeError(
            f"trial_metrics.json mismatch under {trial_dir}: expected {expected_trial_name}, got {actual_trial_name}"
        )
    return row


def _collect_completed_trial_rows(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for child in sorted(out_dir.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if not _is_completed_trial_dir(child):
            continue
        metrics_path = child / "trial_metrics.json"
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        trial_name = str(row.get("trial_name") or "").strip()
        if not trial_name:
            raise RuntimeError(f"Completed trial_metrics.json under {child} is missing trial_name")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed trial rows found under {out_dir}")
    return rows


def _topk_overlap_rate(truth: np.ndarray, pred: np.ndarray, k: int) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if truth.ndim != 1 or pred.ndim != 1 or truth.shape[0] != pred.shape[0]:
        raise RuntimeError("top-k overlap expects 1-D truth/pred arrays with the same length.")
    n = int(truth.shape[0])
    if n <= 0:
        raise RuntimeError("top-k overlap received an empty array.")
    k = max(1, min(int(k), n))
    truth_top = set(np.argsort(truth, kind="mergesort")[:k].tolist())
    pred_top = set(np.argsort(pred, kind="mergesort")[:k].tolist())
    return float(len(truth_top & pred_top) / float(k))


def _summarize_holdout_frame(work: pd.DataFrame, *, predictions_path: Path) -> dict[str, float | int | str | None]:
    pred = pd.to_numeric(work["pred_dock_mean"], errors="coerce").to_numpy(dtype=np.float64)
    pred_std_total = pd.to_numeric(work["pred_dock_std"], errors="coerce").to_numpy(dtype=np.float64)
    pred_std_epi = pd.to_numeric(work["pred_dock_std_epi"], errors="coerce").to_numpy(dtype=np.float64)
    pred_std_ale = pd.to_numeric(work["pred_dock_std_ale"], errors="coerce").to_numpy(dtype=np.float64)
    truth = pd.to_numeric(work["dock_score"], errors="coerce").to_numpy(dtype=np.float64)
    abs_error = pd.to_numeric(work["abs_error"], errors="coerce").to_numpy(dtype=np.float64)
    sq_error = pd.to_numeric(work["sq_error"], errors="coerce").to_numpy(dtype=np.float64)
    nll_values = pd.to_numeric(work["nll"], errors="coerce").to_numpy(dtype=np.float64)
    epi_frac = pd.to_numeric(work["pred_dock_var_epi_frac"], errors="coerce").to_numpy(dtype=np.float64)

    true_tensor = torch.tensor(truth.astype(np.float32))
    pred_tensor = torch.tensor(pred.astype(np.float32))
    metrics = rank_metrics(true_tensor, pred_tensor, ks=(10, 50, 100))
    top1pct_k = max(1, int(round(int(work.shape[0]) * 0.01)))
    hit_top1pct = _topk_overlap_rate(truth, pred, k=top1pct_k)
    rmse = float(np.sqrt(np.mean(sq_error)))
    mae = float(np.mean(abs_error))
    rmv = float(np.sqrt(np.mean(np.square(pred_std_total)))) if np.isfinite(pred_std_total).any() else float("nan")
    coverage_1sigma = float(np.mean(abs_error <= pred_std_total))
    coverage_2sigma = float(np.mean(abs_error <= (2.0 * pred_std_total)))
    mean_epi_std = float(np.nanmean(pred_std_epi)) if np.isfinite(pred_std_epi).any() else float("nan")
    mean_ale_std = float(np.nanmean(pred_std_ale)) if np.isfinite(pred_std_ale).any() else float("nan")

    out: dict[str, float | int | str | None] = {
        "holdout_requested_n": int(work.shape[0]),
        "holdout_predicted_n": int(work.shape[0]),
        "holdout_skipped_n": 0,
        "rmse": rmse,
        "mae": mae,
        "nll": float(np.nanmean(nll_values)),
        "rmv": rmv,
        "calibration_gap": float(abs(rmv - rmse)) if np.isfinite(rmv) else float("nan"),
        "coverage_1sigma": coverage_1sigma,
        "coverage_1sigma_gap": float(abs(coverage_1sigma - EXPECTED_COVERAGE_1SIGMA)),
        "coverage_2sigma": coverage_2sigma,
        "coverage_2sigma_gap": float(abs(coverage_2sigma - EXPECTED_COVERAGE_2SIGMA)),
        "top1pct_k": int(top1pct_k),
        "hit@1pct": float(hit_top1pct),
        "mean_std": float(np.nanmean(pred_std_total)),
        "median_std": float(np.nanmedian(pred_std_total)),
        "p90_std": float(np.nanquantile(pred_std_total, 0.9)),
        "mean_ale_std": mean_ale_std,
        "mean_epi_std": mean_epi_std,
        "mean_epi_fraction": float(np.nanmean(epi_frac)) if np.isfinite(epi_frac).any() else float("nan"),
        "corr_abs_error_std": _safe_corr(abs_error, pred_std_total, method="pearson"),
        "corr_abs_error_ale_std": _safe_corr(abs_error, pred_std_ale, method="pearson"),
        "corr_abs_error_epi_std": _safe_corr(abs_error, pred_std_epi, method="pearson"),
        "std_error_spearman": _safe_corr(abs_error, pred_std_total, method="spearman"),
        "predictions_path": str(predictions_path.resolve()),
    }
    out.update(metrics)
    return out


def _run_fingerprint_gp_baseline(
    *,
    init_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    device: torch.device,
    fp_bits: int,
    fp_radius: int,
    output_csv: Path,
    artifact_path: Path,
) -> dict[str, float | int | str | None]:
    train_x = _build_morgan_fp_matrix(
        init_df["smiles_canonical"].astype(str).tolist(),
        fp_bits=int(fp_bits),
        fp_radius=int(fp_radius),
    )
    test_x = _build_morgan_fp_matrix(
        holdout_df["smiles_canonical"].astype(str).tolist(),
        fp_bits=int(fp_bits),
        fp_radius=int(fp_radius),
    )
    train_y = pd.to_numeric(init_df["dock_score"], errors="raise").to_numpy(dtype=np.float64)
    mean_np, std_np = _fit_docking_gp_and_predict(train_x, train_y, test_x, device=device)
    pred_var = np.square(std_np)
    truth = pd.to_numeric(holdout_df["dock_score"], errors="raise").to_numpy(dtype=np.float64)
    abs_error = np.abs(mean_np - truth)
    sq_error = np.square(mean_np - truth)
    nll = 0.5 * (np.log(2.0 * np.pi * pred_var) + (sq_error / pred_var))

    work = holdout_df.copy().reset_index(drop=True)
    work["pred_dock_mean"] = mean_np
    work["pred_dock_std"] = std_np
    work["pred_dock_std_total"] = std_np
    work["pred_dock_std_epi"] = np.full_like(std_np, np.nan)
    work["pred_dock_std_ale"] = std_np
    work["pred_dock_var_epi_frac"] = np.full_like(std_np, np.nan)
    work["abs_error"] = abs_error
    work["sq_error"] = sq_error
    work["nll"] = nll
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(output_csv, index=False)
    torch.save(
        {
            "kind": "fingerprint_gp",
            "fp_bits": int(fp_bits),
            "fp_radius": int(fp_radius),
            "train_n": int(init_df.shape[0]),
            "holdout_n": int(holdout_df.shape[0]),
        },
        str(artifact_path),
    )
    return _summarize_holdout_frame(work, predictions_path=output_csv)


def _evaluate_holdout_predictions(
    holdout_df: pd.DataFrame,
    *,
    ckpt_path: Path,
    device: torch.device,
    pred_batch_size: int,
    candidate_3d_cfg: dict,
    ligand_vocab: Sequence[str],
    output_csv: Path,
    ligand3d_cache_dir: Path,
    confgen_cfg: dict,
    bayes_eval_samples_override: int | None = None,
) -> dict[str, float | int | str | None]:
    loaded = load_surrogate(str(ckpt_path), device)
    model = loaded[0]
    mean = float(loaded[1])
    std = float(loaded[2])
    model_node_dim = int(loaded[3])
    model_edge_dim = int(loaded[4])
    model_atom_extra_dim = int(loaded[5])
    model_bond_extra_dim = int(loaded[6])
    model_fp_dim = int(loaded[7])
    model_fp_radius = int(loaded[8])
    surrogate_kind = str(loaded[10])
    surrogate_meta = dict(loaded[11]) if isinstance(loaded[11], dict) else {}

    if surrogate_kind == "tensornet":
        model_fp_dim = int(surrogate_meta.get("fp_dim", model_fp_dim))
        model_fp_radius = int(surrogate_meta.get("fp_radius", model_fp_radius))

    if surrogate_kind == "tensornet":
        eval_rows = holdout_df.copy().reset_index(drop=True)
        eval_rows["split"] = "test"
        store = LigandOnly3DStore(
            root=ckpt_path.parent,
            ligand_vocab_override=list(ligand_vocab),
            cache_dir=ligand3d_cache_dir,
            fp_dim=model_fp_dim,
            fp_radius=model_fp_radius,
            rows=eval_rows.to_dict("records"),
            confgen_max_attempts=int(confgen_cfg["max_attempts"]),
            confgen_seed=int(confgen_cfg["seed"]),
            confgen_num_confs=int(confgen_cfg["num_confs"]),
            confgen_max_opt_iters=int(confgen_cfg["max_opt_iters"]),
            confgen_optimize=bool(confgen_cfg["optimize"]),
            confgen_prefer_mmff=bool(confgen_cfg["prefer_mmff"]),
            build_num_workers=int(confgen_cfg["workers"]),
            build_mp_chunksize=int(confgen_cfg["chunksize"]),
        )
        kept_ids = [lig_id for lig_id in eval_rows["ligand_id"].astype(str).tolist() if lig_id in store.id_to_idx]
        if not kept_ids:
            raise RuntimeError("Holdout evaluation produced no retained molecules.")
        kept_indices = [int(store.id_to_idx[lig_id]) for lig_id in kept_ids]
        loader = DataLoader(store.get_dataset_from_indices(kept_indices), batch_size=int(pred_batch_size), shuffle=False)
        eval_df = eval_rows.set_index("ligand_id").loc[kept_ids].reset_index()
    else:
        loader, kept = _prepare_loader(
            holdout_df["smiles_canonical"].astype(str).tolist(),
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
        eval_df = holdout_df.set_index("smiles_canonical").loc[kept].reset_index()
        if eval_df.empty:
            raise RuntimeError("Holdout evaluation produced no retained molecules.")

    mode = str(getattr(model, "uncertainty_mode", surrogate_meta.get("uncertainty_mode", "gaussian"))).strip().lower()
    scale = abs(float(std)) if abs(float(std)) > 0 else 1.0
    shift = float(mean)

    pred_mean_chunks = []
    pred_std_total_chunks = []
    pred_std_epi_chunks = []
    pred_std_ale_chunks = []
    nll_chunks = []
    cursor = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_size = int(batch.num_graphs)
            y_true = torch.tensor(
                eval_df.iloc[cursor : cursor + batch_size]["dock_score"].to_numpy(dtype=np.float32),
                device=device,
            )
            y_z = (y_true - shift) / scale

            if mode == "bayes":
                draws = []
                eval_samples = (
                    max(1, int(bayes_eval_samples_override))
                    if bayes_eval_samples_override is not None
                    else max(1, int(surrogate_meta.get("eval_samples", 8)))
                )
                for _ in range(eval_samples):
                    model.sample_bayes_params()
                    draws.append(model(batch).detach())
                    model.clear_bayes_params()
                stacked = torch.stack(draws, dim=0)
                mu_z = stacked.mean(dim=0).view(-1)
                var_z = (
                    torch.full_like(mu_z, 1.0e-6)
                    if stacked.size(0) == 1
                    else stacked.var(dim=0, unbiased=False).view(-1).clamp_min(1.0e-6)
                )
                mu = mu_z * scale + shift
                total_std = torch.sqrt(var_z * (scale ** 2))
                epi_std = total_std
                ale_std = torch.zeros_like(total_std)
                nll = 0.5 * (torch.log(var_z) + ((y_z - mu_z) ** 2) / var_z)
            elif mode == "nig":
                raw = model(batch)
                gamma, nu, alpha, beta = split_nig_output(raw)
                ale_var_z, epi_var_z, total_var_z = nig_variances_from_params(nu, alpha, beta, min_var=1e-6)
                mu = gamma * scale + shift
                total_std = torch.sqrt(total_var_z * (scale ** 2))
                epi_std = torch.sqrt(epi_var_z * (scale ** 2))
                ale_std = torch.sqrt(ale_var_z * (scale ** 2))
                nll = _nig_nll_terms(gamma, nu, alpha, beta, y_z)
            elif hasattr(model, "decompose"):
                mu_z, ale_var_z, epi_var_z, total_var_z = model.decompose(batch)
                total_var_z = total_var_z.clamp_min(1e-6)
                ale_var_z = ale_var_z.clamp_min(1e-6)
                epi_var_z = epi_var_z.clamp_min(0.0)
                mu = mu_z * scale + shift
                total_std = torch.sqrt(total_var_z * (scale ** 2))
                epi_std = torch.sqrt(epi_var_z * (scale ** 2))
                ale_std = torch.sqrt(ale_var_z * (scale ** 2))
                nll = 0.5 * (torch.log(total_var_z) + ((y_z - mu_z.view(-1)) ** 2) / total_var_z)
            else:
                raw = model(batch)
                mu_z, raw_log_var = split_gaussian_output(raw)
                var_z = gaussian_variance_from_raw(raw_log_var, logvar_min=-8.0, logvar_max=4.0, min_var=1e-6)
                mu = mu_z * scale + shift
                total_std = torch.sqrt(var_z * (scale ** 2))
                epi_std = torch.full_like(total_std, float("nan"))
                ale_std = total_std
                nll = 0.5 * (torch.log(var_z) + ((y_z - mu_z.view(-1)) ** 2) / var_z)

            pred_mean_chunks.append(mu.detach().cpu().view(-1))
            pred_std_total_chunks.append(total_std.detach().cpu().view(-1))
            pred_std_epi_chunks.append(epi_std.detach().cpu().view(-1))
            pred_std_ale_chunks.append(ale_std.detach().cpu().view(-1))
            nll_chunks.append(nll.detach().cpu().view(-1))
            cursor += batch_size

    pred = torch.cat(pred_mean_chunks, dim=0).numpy()
    pred_std_total = torch.cat(pred_std_total_chunks, dim=0).numpy()
    pred_std_epi = torch.cat(pred_std_epi_chunks, dim=0).numpy()
    pred_std_ale = torch.cat(pred_std_ale_chunks, dim=0).numpy()
    nll_values = torch.cat(nll_chunks, dim=0).numpy()
    truth = eval_df["dock_score"].to_numpy(dtype=np.float64)
    abs_error = np.abs(pred - truth)
    sq_error = np.square(pred - truth)

    work = eval_df.copy().reset_index(drop=True)
    work["pred_dock_mean"] = pred
    work["pred_dock_std"] = pred_std_total
    work["pred_dock_std_total"] = pred_std_total
    work["pred_dock_std_epi"] = pred_std_epi
    work["pred_dock_std_ale"] = pred_std_ale
    epi_var = np.square(pred_std_epi)
    total_var = np.square(pred_std_total)
    epi_frac = np.full(total_var.shape, np.nan, dtype=np.float64)
    valid_frac = np.isfinite(epi_var) & np.isfinite(total_var) & (total_var > 0.0)
    epi_frac[valid_frac] = epi_var[valid_frac] / total_var[valid_frac]
    work["pred_dock_var_epi_frac"] = epi_frac
    work["abs_error"] = abs_error
    work["sq_error"] = sq_error
    work["nll"] = nll_values
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(output_csv, index=False)
    out = _summarize_holdout_frame(work, predictions_path=output_csv)
    out["holdout_requested_n"] = int(holdout_df.shape[0])
    out["holdout_predicted_n"] = int(work.shape[0])
    out["holdout_skipped_n"] = int(holdout_df.shape[0] - work.shape[0])
    return out



def _recommend(summary_df: pd.DataFrame) -> dict:
    ranked = summary_df.sort_values(
        by=["spearman", "nll", "calibration_gap", "coverage_1sigma_gap", "rmse", "train_seconds"],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()
    return {
        "recommended_trial": best["trial_name"],
        "reason": "sorted by spearman, nll, calibration gap, 1-sigma coverage gap, rmse, then train_seconds",
        "row": best,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark initial surrogate semi-supervised training on a fixed holdout set. "
            "Compares scratch vs encoder-pretrained TensorNet, small GIN, NIG, heteroscedastic Gaussian, and ensemble baselines."
        )
    )
    ap.add_argument("--config", default="config/surrogate/config.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pool-csv", default=None)
    ap.add_argument("--init-dataset-root", default=None)
    ap.add_argument("--init-size", type=int, default=120)
    ap.add_argument("--init-clusters", type=int, default=16)
    ap.add_argument("--init-min-per-cluster", type=int, default=3)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred-batch-size", type=int, default=256)
    ap.add_argument("--eval-3d-workers", type=int, default=1)
    ap.add_argument("--ligand3d-cache-dir", default=None)
    ap.add_argument("--confgen-seed", type=int, default=None)
    ap.add_argument("--override-bayes-eval-samples", type=int, default=None)
    ap.add_argument("--reevaluate-completed-bayes", action="store_true")

    ap.add_argument("--pretrain-dataset-root", default=None)
    ap.add_argument("--pretrain-csv", default=None)
    ap.add_argument("--pretrain-smiles-col", default="canonical_smiles")
    ap.add_argument("--pretrain-valid-frac", type=float, default=0.1)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--pretrain-batch-size", type=int, default=None)
    ap.add_argument("--pretrain-lr", type=float, default=None)
    ap.add_argument("--pretrain-weight-decay", type=float, default=None)
    ap.add_argument("--pretrain-fp-bits", type=int, default=None)
    ap.add_argument("--pretrain-fp-radius", type=int, default=None)

    ap.add_argument("--skip-gin-small", action="store_true")
    ap.add_argument("--skip-scratch", action="store_true")
    ap.add_argument("--skip-pretrained", action="store_true")
    ap.add_argument("--skip-ensemble", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_maybe_relative_path(args.config, REPO, Path.cwd())
    cfg = load_config(str(config_path))
    surrogate_cfg = _section(cfg, "surrogate")
    candidate_3d_cfg = dict(_section(_section(cfg, "candidate"), "candidate_3d"))
    candidate_3d_cfg["workers"] = int(args.eval_3d_workers)
    confgen_cfg = _resolve_confgen_cfg(args, candidate_3d_cfg)
    ligand3d_cache_dir = _resolve_ligand3d_cache_dir(args)
    device = _device_from_cfg(_section(cfg, "general").get("device", "auto"))
    dock_valid_max = _section(cfg, "objective").get("dock_valid_max", 0.0)
    dock_valid_max = None if dock_valid_max in (None, "", "null") else float(dock_valid_max)

    dataset_root, init_df, holdout_df, source_meta = _prepare_benchmark_data(
        args,
        out_dir=out_dir,
        dock_valid_max=dock_valid_max,
    )
    pretrain_dataset_root = _resolve_pretrain_dataset_root(args, out_dir)
    init_source_dataset_root = Path(args.init_dataset_root).resolve() if args.init_dataset_root else None
    ligand_vocab = _resolve_ligand_vocab(
        init_dataset_root=init_source_dataset_root,
        init_smiles=pd.concat([
            init_df["smiles_canonical"].astype(str),
            holdout_df["smiles_canonical"].astype(str),
        ], ignore_index=True).tolist(),
        pretrain_dataset_root=pretrain_dataset_root.resolve() if pretrain_dataset_root is not None else None,
    )
    (dataset_root / "ligand_vocab.json").write_text(json.dumps(ligand_vocab, indent=2), encoding="utf-8")
    trials = _build_trials(args, main_config_path=config_path, surrogate_cfg=surrogate_cfg)
    shared_pretrain_torchmd_cfg = _resolve_shared_pretrain_torchmd_cfg(trials, surrogate_cfg)

    needs_pretrain = any(bool(trial["needs_pretrain"]) for trial in trials)
    pretrain_metrics: dict[str, float | int | str] = {}
    pretrained_encoder_ckpt: Path | None = None

    if needs_pretrain:
        if pretrain_dataset_root is None:
            raise RuntimeError("Pretrained/ensemble trials require --pretrain-dataset-root or --pretrain-csv.")
        pretrain_cfg = dict(_section(surrogate_cfg, "init_encoder_pretrain"))
        resolved_pretrain_fp_bits = int(args.pretrain_fp_bits or pretrain_cfg.get("fp_bits", 512))
        resolved_pretrain_fp_radius = int(args.pretrain_fp_radius or pretrain_cfg.get("fp_radius", 2))
        expected_node_feat_dim = int(len(ligand_vocab) + ATOM_EXTRA_DIM)
        pretrain_metrics_path = out_dir / "pretrain" / "pretrain_metrics.json"
        local_pretrain_ckpt = out_dir / "pretrain" / "encoder.pt"
        if local_pretrain_ckpt.exists() and pretrain_metrics_path.exists():
            ckpt_node_feat_dim = _checkpoint_node_feat_dim(local_pretrain_ckpt)
            if ckpt_node_feat_dim != int(expected_node_feat_dim):
                raise RuntimeError(
                    f"Existing local pretrain encoder has node_feat_dim={ckpt_node_feat_dim}, expected {expected_node_feat_dim}: {local_pretrain_ckpt}"
                )
            pretrained_encoder_ckpt = local_pretrain_ckpt.resolve()
            pretrain_metrics = json.loads(pretrain_metrics_path.read_text(encoding="utf-8"))
            pretrain_metrics["pretrain_ckpt"] = str(pretrained_encoder_ckpt)
            pretrain_metrics["pretrain_reused_existing"] = True
            pretrain_metrics["pretrain_reuse_source"] = "out_dir.pretrain.encoder"
        else:
            reused_encoder_ckpt, reuse_info = _resolve_existing_iter0_encoder_ckpt(
                init_dataset_root=init_source_dataset_root,
                expected_node_feat_dim=expected_node_feat_dim,
            )
            if reused_encoder_ckpt is not None:
                pretrained_encoder_ckpt = reused_encoder_ckpt
                pretrain_metrics = {
                    "pretrain_ckpt": str(pretrained_encoder_ckpt),
                    "pretrain_reused_existing": True,
                    "pretrain_reuse_source": reuse_info.get("reuse_source"),
                    "pretrain_expected_node_feat_dim": expected_node_feat_dim,
                    "pretrain_fp_bits": int(resolved_pretrain_fp_bits),
                    "pretrain_fp_radius": int(resolved_pretrain_fp_radius),
                    "pretrain_candidate_count": int(reuse_info.get("reuse_candidate_count", 0)),
                    "pretrain_reuse_rejections": list(reuse_info.get("reuse_rejections", [])),
                }
            else:
                resolved_pretrain_epochs = int(args.pretrain_epochs or pretrain_cfg.get("epochs", 0))
                if resolved_pretrain_epochs <= 0:
                    raise RuntimeError("Pretrained/ensemble trials require pretrain epochs > 0.")
                resolved_pretrain_batch_size = int(args.pretrain_batch_size or pretrain_cfg.get("batch_size", surrogate_cfg.get("retrain_batch_size", 32)))
                resolved_pretrain_lr = float(args.pretrain_lr or pretrain_cfg.get("lr", 2.0e-4))
                resolved_pretrain_weight_decay = float(args.pretrain_weight_decay or pretrain_cfg.get("weight_decay", 1.0e-5))
                if shared_pretrain_torchmd_cfg is None:
                    raise RuntimeError("Pretrain requested but no shared pretrain torchmd config was resolved.")
                pretrain_torchmd_cfg = _build_pretrain_torchmd_cfg(shared_pretrain_torchmd_cfg, surrogate_cfg)
                pretrained_encoder_ckpt = out_dir / "pretrain" / "encoder.pt"
                pretrained_encoder_ckpt.parent.mkdir(parents=True, exist_ok=True)
                pretrain_device = _device_from_cfg(str(pretrain_cfg.get("device", _section(cfg, "general").get("device", "auto"))))
                pretrain_metrics = _run_tensornet_fp_pretrain(
                    dataset_root=Path(pretrain_dataset_root),
                    save_path=pretrained_encoder_ckpt,
                    device=pretrain_device,
                    batch_size=resolved_pretrain_batch_size,
                    lr=resolved_pretrain_lr,
                    weight_decay=resolved_pretrain_weight_decay,
                    epochs=resolved_pretrain_epochs,
                    fp_bits=resolved_pretrain_fp_bits,
                    fp_radius=resolved_pretrain_fp_radius,
                    torchmd_cfg=pretrain_torchmd_cfg,
                    candidate_3d_cfg=confgen_cfg,
                    ligand_vocab=ligand_vocab,
                    ligand3d_cache_dir=str(ligand3d_cache_dir),
                    confgen_seed=int(confgen_cfg["seed"]),
                )
                pretrain_metrics["pretrain_reused_existing"] = False
                pretrain_metrics["pretrain_reuse_source"] = None
                pretrain_metrics["pretrain_expected_node_feat_dim"] = expected_node_feat_dim
                pretrain_metrics["pretrain_candidate_count"] = int(reuse_info.get("reuse_candidate_count", 0))
                pretrain_metrics["pretrain_reuse_rejections"] = list(reuse_info.get("reuse_rejections", []))
        pretrain_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pretrain_metrics_path.write_text(json.dumps(pretrain_metrics, indent=2), encoding="utf-8")

    summary_rows: list[dict] = []
    for trial in trials:
        trial_dir = out_dir / str(trial["trial_name"])
        trial_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = trial_dir / "surrogate.pt"

        if _is_completed_trial_dir(trial_dir):
            should_reevaluate_bayes = (
                bool(args.reevaluate_completed_bayes)
                and str(trial.get("uncertainty_mode", "")).strip().lower() == "bayes"
            )
            if should_reevaluate_bayes:
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Cannot re-evaluate completed Bayes trial without checkpoint: {ckpt_path}")
                eval_samples_text = (
                    int(args.override_bayes_eval_samples)
                    if args.override_bayes_eval_samples is not None
                    else "checkpoint"
                )
                print(
                    f"[surrogate][reeval] trial={trial['trial_name']} output_dir={trial_dir} "
                    f"bayes_eval_samples={eval_samples_text}"
                )
                existing_row = _load_completed_trial_metrics(trial_dir, expected_trial_name=str(trial["trial_name"]))
                holdout_metrics = _evaluate_holdout_predictions(
                    holdout_df=holdout_df,
                    ckpt_path=ckpt_path,
                    device=device,
                    pred_batch_size=int(args.pred_batch_size),
                    candidate_3d_cfg=candidate_3d_cfg,
                    ligand_vocab=ligand_vocab,
                    output_csv=trial_dir / "holdout_predictions.csv",
                    ligand3d_cache_dir=ligand3d_cache_dir,
                    confgen_cfg=confgen_cfg,
                    bayes_eval_samples_override=args.override_bayes_eval_samples,
                )
                row = dict(existing_row)
                row.update(holdout_metrics)
                (trial_dir / "trial_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                summary_rows.append(row)
                continue
            print(f"[surrogate][reuse] trial={trial['trial_name']} output_dir={trial_dir}")
            summary_rows.append(_load_completed_trial_metrics(trial_dir, expected_trial_name=str(trial["trial_name"])))
            continue

        trial_record = dict(trial)
        trial_record["dataset_root"] = str(dataset_root.resolve())
        trial_record["pretrained_encoder_ckpt"] = str(pretrained_encoder_ckpt.resolve()) if pretrained_encoder_ckpt is not None and bool(trial["needs_pretrain"]) else None
        (trial_dir / "trial_config.json").write_text(json.dumps(trial_record, indent=2), encoding="utf-8")

        trial_torchmd_cfg = dict(trial.get("torchmd_cfg") or {}) if str(trial["backbone"]) == "tensornet" else None
        t0 = time.perf_counter()
        if str(trial["backbone"]) == "gp":
            holdout_metrics = _run_fingerprint_gp_baseline(
                init_df=init_df,
                holdout_df=holdout_df,
                device=device,
                fp_bits=int(trial["gp_fp_bits"]),
                fp_radius=int(trial["gp_fp_radius"]),
                output_csv=trial_dir / "holdout_predictions.csv",
                artifact_path=ckpt_path,
            )
            train_seconds = float(time.perf_counter() - t0)
            (trial_dir / "train_summary.json").write_text(
                json.dumps(
                    {
                        "model_kind": "fingerprint_gp",
                        "fp_bits": int(trial["gp_fp_bits"]),
                        "fp_radius": int(trial["gp_fp_radius"]),
                        "train_n": int(init_df.shape[0]),
                        "holdout_n": int(holdout_df.shape[0]),
                        "train_seconds": train_seconds,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            train_surrogate_from_scratch(
                root=str(dataset_root),
                save_path=str(ckpt_path),
                epochs=int(trial["epochs"]),
                batch_size=int(trial.get("batch_size", surrogate_cfg.get("retrain_batch_size", 32))),
                lr=float(trial.get("lr", surrogate_cfg.get("retrain_lr", 5.0e-4))),
                weight_decay=float(trial.get("weight_decay", surrogate_cfg.get("retrain_weight_decay", 1.0e-5))),
                hidden_dim=int(trial["hidden_dim"]),
                num_layers=int(trial["num_layers"]),
                dropout=float(trial["dropout"]),
                use_edge_attr=bool(surrogate_cfg.get("retrain_use_edge_attr", True)),
                use_ligand_mask=bool(surrogate_cfg.get("retrain_use_ligand_mask", True)),
                standardize=bool(surrogate_cfg.get("retrain_standardize", True)),
                fp_dim=int(surrogate_cfg.get("retrain_fp_dim", 0)),
                fp_radius=int(surrogate_cfg.get("retrain_fp_radius", 2)),
                eval_samples=int(surrogate_cfg.get("retrain_eval_samples", 8)),
                scheduler=str(surrogate_cfg.get("retrain_scheduler", "cosine")),
                warmup_epochs=int(surrogate_cfg.get("retrain_warmup_epochs", 5)),
                min_lr=float(surrogate_cfg.get("retrain_min_lr", 1.0e-5)),
                early_stop_patience=int(surrogate_cfg.get("retrain_early_stop_patience", 4)),
                early_stop_min_delta=float(surrogate_cfg.get("retrain_early_stop_min_delta", 1.0e-3)),
                device=device,
                dock_valid_max=dock_valid_max,
                backbone=str(trial["backbone"]),
                torchmd_cfg=trial_torchmd_cfg,
                ensemble_heads=int(trial["ensemble_heads"]),
                freeze_backbone=bool(trial["freeze_backbone"]),
                pretrained_encoder_ckpt=(str(pretrained_encoder_ckpt) if pretrained_encoder_ckpt is not None and bool(trial["needs_pretrain"]) else None),
                uncertainty_mode=str(trial["uncertainty_mode"]),
                ensemble_scheme=str(trial["ensemble_scheme"]),
                ensemble_bootstrap=bool(surrogate_cfg.get("retrain_ensemble_bootstrap", True)),
                random_seed=int(args.seed),
                auto_prepare=False,
                ligand3d_cache_dir=str(ligand3d_cache_dir),
                ligand_vocab_override=ligand_vocab,
                ligand3d_store=None,
                test_ligand_ids=None,
                confgen_max_attempts=int(confgen_cfg["max_attempts"]),
                confgen_seed=int(confgen_cfg["seed"]),
                confgen_num_confs=int(confgen_cfg["num_confs"]),
                confgen_max_opt_iters=int(confgen_cfg["max_opt_iters"]),
                confgen_optimize=bool(confgen_cfg["optimize"]),
                confgen_prefer_mmff=bool(confgen_cfg["prefer_mmff"]),
                ligand3d_num_workers=int(confgen_cfg["workers"]),
                ligand3d_mp_chunksize=int(confgen_cfg["chunksize"]),
                encoder_lr=(None if trial.get("encoder_lr") is None else float(trial.get("encoder_lr"))),
                train_log_csv=str(trial_dir / "train_log.csv"),
                train_summary_json=str(trial_dir / "train_summary.json"),
            )
            train_seconds = float(time.perf_counter() - t0)

            holdout_metrics = _evaluate_holdout_predictions(
                holdout_df=holdout_df,
                ckpt_path=ckpt_path,
                device=device,
                pred_batch_size=int(args.pred_batch_size),
                candidate_3d_cfg=candidate_3d_cfg,
                ligand_vocab=ligand_vocab,
                output_csv=trial_dir / "holdout_predictions.csv",
                ligand3d_cache_dir=ligand3d_cache_dir,
                confgen_cfg=confgen_cfg,
                bayes_eval_samples_override=args.override_bayes_eval_samples,
            )

        trial_torchmd_cfg_used = dict(trial.get("torchmd_cfg") or {})
        row = {
            "trial_name": str(trial["trial_name"]),
            "trial_family": str(trial.get("trial_family", "")),
            "backbone": str(trial["backbone"]),
            "uncertainty_mode": str(trial["uncertainty_mode"]),
            "ensemble_heads": int(trial["ensemble_heads"]),
            "ensemble_scheme": str(trial["ensemble_scheme"]),
            "used_pretrain": bool(trial["needs_pretrain"]),
            "freeze_backbone": bool(trial["freeze_backbone"]),
            "epochs": int(trial["epochs"]),
            "train_batch_size": int(trial.get("batch_size", surrogate_cfg.get("retrain_batch_size", 32))),
            "train_lr": float(trial.get("lr", surrogate_cfg.get("retrain_lr", 5.0e-4))),
            "encoder_lr": None if trial.get("encoder_lr") is None else float(trial.get("encoder_lr")),
            "weight_decay": float(trial.get("weight_decay", surrogate_cfg.get("retrain_weight_decay", 1.0e-5))),
            "trial_config_path": str(trial.get("trial_config_path", "")),
            "tensornet_embedding_dim": trial_torchmd_cfg_used.get("embedding_dim") if bool(trial_torchmd_cfg_used) else None,
            "tensornet_num_layers": trial_torchmd_cfg_used.get("num_layers") if bool(trial_torchmd_cfg_used) else None,
            "tensornet_head_hidden_dim": trial_torchmd_cfg_used.get("head_hidden_dim") if bool(trial_torchmd_cfg_used) else None,
            "tensornet_head_num_layers": trial_torchmd_cfg_used.get("head_num_layers") if bool(trial_torchmd_cfg_used) else None,
            "gaussian_warmup_epochs": trial_torchmd_cfg_used.get("gaussian_warmup_epochs") if bool(trial_torchmd_cfg_used) else None,
            "gaussian_var_reg_beta": trial_torchmd_cfg_used.get("gaussian_var_reg_beta") if bool(trial_torchmd_cfg_used) else None,
            "gaussian_select_by": trial_torchmd_cfg_used.get("gaussian_select_by") if bool(trial_torchmd_cfg_used) else None,
            "gaussian_select_nll_weight": trial_torchmd_cfg_used.get("gaussian_select_nll_weight") if bool(trial_torchmd_cfg_used) else None,
            "nig_warmup_epochs": trial_torchmd_cfg_used.get("nig_warmup_epochs") if bool(trial_torchmd_cfg_used) else None,
            "nig_reg_lambda": trial_torchmd_cfg_used.get("nig_reg_lambda") if bool(trial_torchmd_cfg_used) else None,
            "gp_fp_bits": int(trial["gp_fp_bits"]) if trial.get("gp_fp_bits") is not None else None,
            "gp_fp_radius": int(trial["gp_fp_radius"]) if trial.get("gp_fp_radius") is not None else None,
            "train_size": int(init_df.shape[0]),
            "holdout_size": int(holdout_df.shape[0]),
            "train_seconds": train_seconds,
            "pretrain_dataset_root": str(pretrain_dataset_root.resolve()) if pretrain_dataset_root is not None else None,
        }
        row.update(pretrain_metrics if bool(trial["needs_pretrain"]) else {})
        row.update(holdout_metrics)
        summary_rows.append(row)
        (trial_dir / "trial_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary_rows = _collect_completed_trial_rows(out_dir)
    summary = pd.DataFrame(summary_rows).sort_values("trial_name").reset_index(drop=True)
    summary.to_csv(out_dir / "trial_summary.csv", index=False)
    recommendation = _recommend(summary)
    (out_dir / "recommended_config.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    run_meta = {
        "config": str(config_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "device": str(device),
        "dataset_root": str(dataset_root.resolve()),
        "ligand_vocab_size": int(len(ligand_vocab)),
        "ligand_vocab": list(ligand_vocab),
        "ligand3d_cache_dir": str(ligand3d_cache_dir.resolve()),
        "confgen_cfg": dict(confgen_cfg),
        "trials": [str(item["trial_name"]) for item in trials],
        "benchmark_initial_models": {str(item["trial_name"]): str(item.get("trial_config_path", "")) for item in trials},
        "shared_pretrain_torchmd_cfg": shared_pretrain_torchmd_cfg,
    }
    run_meta.update(source_meta)
    run_meta["pretrain_dataset_root"] = str(pretrain_dataset_root.resolve()) if pretrain_dataset_root is not None else None
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
