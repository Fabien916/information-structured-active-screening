from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from data.ligand_only_3d_dataset import LigandOnly3DDataset, LigandOnly3DStore
from mobo.constants import ATOM_EXTRA_DIM, BOND_EXTRA_DIM, BOND_TYPE_CLASSES
from mobo.graphs import (
    build_ligand_graphs_from_smiles_csv,
    smiles_to_ligand_3d,
    smiles_to_ligand_graph,
)
from mobo.io_utils import (
    _read_smiles_csv,
    _pick_smiles_column,
    _scan_ligand_vocab,
    _is_valid_dock_score,
    _load_ligand_vocab_override,
)
from mobo.logging_utils import log_model_param_report
from mobo.smiles_utils import compute_qed, compute_sa
from train_gin_surrogate import (
    GINSurrogate,
    TensorNetBayesSurrogate,
    TensorNetEnsembleSurrogate,
    TensorNetNIGSurrogate,
    TensorNetSurrogate,
    evaluate,
    fit_target_stats,
    gaussian_nll_loss,
    gaussian_variance_from_raw,
    get_total_kl_loss,
    get_targets,
    nig_evidence_regularizer,
    nig_nll_loss,
    nig_variances_from_params,
    split_gaussian_output,
    split_nig_output,
)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _compute_lr(epoch: int, base_lr: float, total_epochs: int, warmup_epochs: int, min_lr: float) -> float:
    warmup_epochs = max(0, int(warmup_epochs))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * float(epoch) / float(warmup_epochs)
    if total_epochs <= 1:
        return base_lr
    t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    t = min(max(t, 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * t))
    return min_lr + (base_lr - min_lr) * cosine


class _EncodedTargetDataset(TorchDataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if features.dim() != 2:
            raise ValueError(f"Encoded features must be (N,F), got {tuple(features.shape)}")
        if targets.dim() != 1:
            raise ValueError(f"Encoded targets must be (N,), got {tuple(targets.shape)}")
        if features.size(0) != targets.size(0):
            raise ValueError("Encoded features and targets must have matching lengths.")
        self.features = features.contiguous()
        self.targets = targets.contiguous()

    def __len__(self) -> int:
        return int(self.targets.size(0))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class _IndexedDataset(TorchDataset):
    def __init__(self, base_dataset: TorchDataset, indices: Sequence[int]) -> None:
        if len(indices) == 0:
            raise ValueError("Bootstrap dataset indices must be non-empty.")
        self.base_dataset = base_dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]



def _load_independent_tensornet_ensemble(ckpt: dict, device: torch.device) -> torch.nn.Module:
    from mobo.tensornet_independent_ensemble import IndependentTensorNetEnsemble, _build_model

    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("Independent ensemble checkpoint missing config dictionary.")
    torchmd_cfg = dict(cfg.get("torchmd", {}))
    pretrained_encoder_ckpt = cfg.get("pretrained_encoder_ckpt")
    node_feat_dim = int(torchmd_cfg.get("node_feat_dim", 0))
    if node_feat_dim <= 0:
        raise RuntimeError("Independent ensemble checkpoint missing TensorNet node_feat_dim.")

    member_paths = ckpt.get("member_paths") or []
    if not member_paths:
        ensemble_dir = ckpt.get("ensemble_dir") or cfg.get("ensemble_dir")
        if not ensemble_dir:
            raise RuntimeError("Independent ensemble checkpoint missing member_paths.")
        member_paths = sorted(str(path) for path in Path(ensemble_dir).glob("member_*.pt"))
    if not member_paths:
        raise RuntimeError("No member checkpoints found for independent ensemble.")

    members = []
    for member_path in member_paths:
        payload = torch.load(member_path, map_location=device)
        model = _build_model(
            node_feat_dim=node_feat_dim,
            torchmd_cfg=torchmd_cfg,
            fp_dim=int(cfg.get("fp_dim", 0)),
            init_encoder_ckpt=(None if not pretrained_encoder_ckpt else Path(pretrained_encoder_ckpt)),
            device=device,
        )
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        members.append(model)

    return IndependentTensorNetEnsemble(
        members,
        logvar_min=float(torchmd_cfg.get("gaussian_logvar_min", -8.0)),
        logvar_max=float(torchmd_cfg.get("gaussian_logvar_max", 4.0)),
        min_var=float(torchmd_cfg.get("gaussian_min_var", 1e-6)),
    ).to(device)


def _train_independent_tensornet_ensemble(
    *,
    root: str,
    save_path: str,
    train_dataset,
    val_loader,
    test_loader,
    node_feat_dim: int,
    torchmd_cfg: dict,
    fp_dim: int,
    fp_radius: int,
    pretrained_encoder_ckpt: str | None,
    device: torch.device,
    ensemble_scheme: str,
    ensemble_heads: int,
    epochs: int,
    batch_size: int,
    lr: float,
    encoder_lr: float,
    weight_decay: float,
    warmup_epochs: int,
    min_lr: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    mean: float,
    std: float,
    dock_valid_max: float | None,
    train_log_csv: str | None,
    train_summary_json: str | None,
    ensemble_bootstrap: bool,
    random_seed: int,
) -> str:
    from mobo.tensornet_independent_ensemble import _train_variant

    ckpt_path = Path(save_path)
    variant_dir = ckpt_path.parent / f"{ckpt_path.stem}_{ensemble_scheme}_ensemble"
    summary = _train_variant(
        name=ensemble_scheme,
        train_dataset=train_dataset,
        val_loader=val_loader,
        test_loader=test_loader,
        node_feat_dim=int(node_feat_dim),
        torchmd_cfg=dict(torchmd_cfg),
        fp_dim=int(fp_dim),
        init_encoder_ckpt=(None if pretrained_encoder_ckpt is None else Path(pretrained_encoder_ckpt)),
        out_dir=variant_dir,
        device=device,
        ensemble_size=int(ensemble_heads),
        epochs=int(epochs),
        lr_head=float(lr),
        lr_encoder=float(encoder_lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        min_lr=float(min_lr),
        early_stop_patience=int(early_stop_patience),
        early_stop_min_delta=float(early_stop_min_delta),
        gaussian_warmup_epochs=int(torchmd_cfg.get("gaussian_warmup_epochs", 5)),
        gaussian_var_reg_beta=float(torchmd_cfg.get("gaussian_var_reg_beta", 1e-4)),
        gaussian_logvar_min=float(torchmd_cfg.get("gaussian_logvar_min", -8.0)),
        gaussian_logvar_max=float(torchmd_cfg.get("gaussian_logvar_max", 4.0)),
        gaussian_min_var=float(torchmd_cfg.get("gaussian_min_var", 1e-6)),
        mean=float(mean),
        std=float(std),
        dock_valid_max=dock_valid_max,
        select_nll_weight=float(torchmd_cfg.get("gaussian_select_nll_weight", 0.05)),
        batch_size=int(batch_size),
        bootstrap=bool(ensemble_bootstrap),
        seed=int(random_seed),
    )

    member_paths = sorted(str(path) for path in variant_dir.glob("member_*.pt"))
    if not member_paths:
        raise RuntimeError(f"Independent ensemble training produced no member checkpoints under {variant_dir}.")

    state = {
        "mean": float(mean),
        "std": float(std),
        "ensemble_dir": str(variant_dir),
        "member_paths": member_paths,
        "config": {
            "backbone": "tensornet",
            "torchmd": dict(torchmd_cfg),
            "fp_dim": int(fp_dim),
            "fp_radius": int(fp_radius),
            "ensemble_heads": int(ensemble_heads),
            "ensemble_type": "independent",
            "ensemble_scheme": str(ensemble_scheme),
            "ensemble_bootstrap": bool(ensemble_bootstrap),
            "pretrained_encoder_ckpt": None if pretrained_encoder_ckpt is None else str(pretrained_encoder_ckpt),
            "lr": float(lr),
            "encoder_lr": float(encoder_lr),
            "weight_decay": float(weight_decay),
            "warmup_epochs": int(warmup_epochs),
            "min_lr": float(min_lr),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
            "gaussian_select_by": str(torchmd_cfg.get("gaussian_select_by", "rmse")).lower(),
            "gaussian_select_nll_weight": float(torchmd_cfg.get("gaussian_select_nll_weight", 0.05)),
        },
    }
    torch.save(state, save_path)

    if train_log_csv:
        log_frames = []
        for member_idx, member_log_path in enumerate(sorted(variant_dir.glob("member_*_train_log.csv"))):
            frame = pd.read_csv(member_log_path)
            frame.insert(0, "member_idx", int(member_idx))
            log_frames.append(frame)
        log_path = Path(train_log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_frames:
            pd.concat(log_frames, ignore_index=True).to_csv(log_path, index=False)
        else:
            pd.DataFrame().to_csv(log_path, index=False)

    if train_summary_json:
        summary_payload = {
            "save_path": str(save_path),
            "root": str(root),
            "backbone": "tensornet",
            "ensemble_type": "independent",
            "ensemble_scheme": str(ensemble_scheme),
            "ensemble_bootstrap": bool(ensemble_bootstrap),
            "target_stats": {
                "mean": float(mean),
                "std": float(std),
            },
            "variant_dir": str(variant_dir),
            "member_paths": member_paths,
            "summary": summary,
        }
        Path(train_summary_json).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    test_metrics = dict(summary.get("test_metrics", {}))
    print(
        "surrogate_retrain: "
        f"saved={save_path} "
        f"scheme={ensemble_scheme} "
        f"ensemble={ensemble_heads} "
        f"test_rmse={float(test_metrics.get('rmse', float('nan'))):.4f} "
        f"test_nll={float(test_metrics.get('nll', float('nan'))):.4f}"
    )
    return save_path


def _bootstrap_index_sets(n_items: int, num_heads: int) -> list[list[int]]:
    if int(n_items) <= 0:
        raise ValueError("Cannot bootstrap from an empty dataset.")
    if int(num_heads) <= 0:
        raise ValueError("num_heads must be positive for bootstrap training.")
    return [
        torch.randint(0, int(n_items), (int(n_items),), dtype=torch.long).tolist()
        for _ in range(int(num_heads))
    ]



def _build_head_loaders(dataset, batch_size: int, num_heads: int, loader_cls, shuffle: bool):
    loaders = []
    for indices in _bootstrap_index_sets(len(dataset), int(num_heads)):
        head_ds = _IndexedDataset(dataset, indices)
        loaders.append(loader_cls(head_ds, batch_size=int(batch_size), shuffle=bool(shuffle)))
    return loaders



def _iter_head_batches(loaders: Sequence):
    if not loaders:
        raise RuntimeError("Expected at least one head loader.")
    lengths = [len(loader) for loader in loaders]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Head loaders have mismatched lengths: {lengths}")
    return zip(*loaders)


def _set_frozen_encoder_eval(model: torch.nn.Module) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False


def _set_tensornet_single_model_train_scheme(model: torch.nn.Module, scheme: str) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("Single-model TensorNet train-scheme selection requires an encoder.")
    scheme = str(scheme).lower().strip()
    for param in encoder.parameters():
        param.requires_grad = False
    if scheme == "last_layer":
        modules = [
            encoder.repr.layers[-1],
            encoder.repr.out_norm,
            encoder.repr.linear,
        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        return
    if scheme == "full":
        for param in encoder.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"Unsupported single-model TensorNet train scheme: {scheme}")


def _encode_dataset_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> _EncodedTargetDataset:
    if not hasattr(model, "encode"):
        raise RuntimeError("Encoded-feature training requires a model with encode().")
    if not hasattr(model, "encoder"):
        raise RuntimeError("Encoded-feature training requires an encoder module.")

    feature_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    _set_frozen_encoder_eval(model)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y = get_targets(batch)
            if y is None:
                raise RuntimeError("Missing target tensor while pre-encoding frozen-backbone features.")
            feat = model.encode(batch)
            feature_chunks.append(feat.detach().cpu())
            target_chunks.append(y.detach().cpu().view(-1))

    if not feature_chunks:
        raise RuntimeError("No encoded features were produced for frozen-backbone training.")

    features = torch.cat(feature_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0).view(-1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _EncodedTargetDataset(features=features, targets=targets)
def _safe_tensor_corr(x: torch.Tensor, y: torch.Tensor, method: str = "spearman") -> float:
    """
    Compute a correlation on two 1D tensors safely.
    Returns NaN if there are too few valid points or if variance is zero.
    Supported methods: spearman, pearson, kendall
    """
    x = torch.as_tensor(x).detach().cpu().reshape(-1)
    y = torch.as_tensor(y).detach().cpu().reshape(-1)

    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return float("nan")

    xs = x[mask].numpy()
    ys = y[mask].numpy()

    if xs.size < 3:
        return float("nan")

    # zero variance -> correlation undefined
    if np.allclose(xs.std(), 0.0) or np.allclose(ys.std(), 0.0):
        return float("nan")

    s1 = pd.Series(xs)
    s2 = pd.Series(ys)
    try:
        return float(s1.corr(s2, method=method))
    except Exception:
        return float("nan")

def _evaluate_encoded(
    model: torch.nn.Module,
    loader: TorchDataLoader,
    device: torch.device,
    mean: float,
    std: float,
    dock_valid_max: float | None = None,
    eval_samples: int = 1,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
    min_var: float = 1e-6,
) -> dict[str, float]:
    model.eval()
    _set_frozen_encoder_eval(model)

    preds = []
    targets = []
    nll_terms = []

    total_std_terms = []
    ale_std_terms = []
    epi_std_terms = []

    # NEW: raw NIG params
    nu_terms = []
    alpha_terms = []
    beta_terms = []

    with torch.no_grad():
        for feat, y in loader:
            feat = feat.to(device)
            y = y.to(device).view(-1)

            mask = torch.isfinite(y)
            if dock_valid_max is not None:
                mask = mask & (y <= dock_valid_max)
            if mask.sum() == 0:
                continue

            y_masked = y[mask]
            y_z = (y_masked - mean) / std

            uncertainty_mode = str(getattr(model, "uncertainty_mode", "gaussian")).strip().lower()

            # 1) Bayesian encoded head
            if uncertainty_mode == "bayes":
                draws = []
                for _ in range(max(1, int(eval_samples))):
                    model.sample_bayes_params()
                    draws.append(model.forward_encoded(feat).detach())
                    model.clear_bayes_params()
                stacked = torch.stack(draws, dim=0)
                mu_z = stacked.mean(dim=0)[mask]
                var_z = (
                    torch.full_like(mu_z, float(min_var))
                    if stacked.size(0) == 1
                    else stacked.var(dim=0, unbiased=False)[mask].clamp_min(min_var)
                )
                mu = mu_z * std + mean
                nll = 0.5 * (torch.log(var_z) + ((y_z - mu_z) ** 2) / var_z)
                total_std = torch.sqrt(var_z * (std ** 2))
                ale_std = torch.zeros_like(total_std)
                epi_std = total_std

            # 2) NIG encoded head
            elif uncertainty_mode == "nig":
                raw = model.forward_encoded(feat)
                gamma, nu, alpha, beta = split_nig_output(raw)

                gamma = gamma[mask]
                nu = nu[mask]
                alpha = alpha[mask]
                beta = beta[mask]

                ale_var_z, epi_var_z, total_var_z = nig_variances_from_params(
                    nu, alpha, beta, min_var=min_var
                )
                mu = gamma * std + mean

                # keep compatibility with current nig_nll_loss behavior
                nll = nig_nll_loss(gamma, nu, alpha, beta, y_z)

                total_std = torch.sqrt(total_var_z * (std ** 2))
                ale_std = torch.sqrt(ale_var_z * (std ** 2))
                epi_std = torch.sqrt(epi_var_z * (std ** 2))

                nu_terms.append(torch.as_tensor(nu.detach().cpu()).reshape(-1))
                alpha_terms.append(torch.as_tensor(alpha.detach().cpu()).reshape(-1))
                beta_terms.append(torch.as_tensor(beta.detach().cpu()).reshape(-1))

            # 3) explicit decomposed encoded head
            elif hasattr(model, "decompose_encoded"):
                mu_z, ale_var_z, epi_var_z, total_var_z = model.decompose_encoded(feat)
                mu = mu_z[mask] * std + mean

                total_var_z = total_var_z[mask].clamp_min(min_var)
                ale_var_z = ale_var_z[mask].clamp_min(min_var)
                epi_var_z = epi_var_z[mask].clamp_min(0.0)

                nll = 0.5 * (torch.log(total_var_z) + ((y_z - mu_z[mask]) ** 2) / total_var_z)
                total_std = torch.sqrt(total_var_z * (std ** 2))
                ale_std = torch.sqrt(ale_var_z * (std ** 2))
                epi_std = torch.sqrt(epi_var_z * (std ** 2))

            # 4) plain gaussian encoded head
            else:
                raw = model.forward_encoded(feat)
                mu_z, raw_log_var = split_gaussian_output(raw)
                var_z = gaussian_variance_from_raw(
                    raw_log_var[mask],
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                    min_var=min_var,
                )
                mu = mu_z[mask] * std + mean
                nll = 0.5 * (torch.log(var_z) + ((y_z - mu_z[mask]) ** 2) / var_z)
                total_std = torch.sqrt(var_z * (std ** 2))
                ale_std = total_std
                epi_std = torch.full_like(total_std, float("nan"))

            preds.append(torch.as_tensor(mu.detach().cpu()).reshape(-1))
            targets.append(torch.as_tensor(y_masked.detach().cpu()).reshape(-1))
            nll_terms.append(torch.as_tensor(nll.detach().cpu()).reshape(-1))
            total_std_terms.append(torch.as_tensor(total_std.detach().cpu()).reshape(-1))
            ale_std_terms.append(torch.as_tensor(ale_std.detach().cpu()).reshape(-1))
            epi_std_terms.append(torch.as_tensor(epi_std.detach().cpu()).reshape(-1))

    if not preds:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "nll": float("nan"),
            "mean_std": float("nan"),
            "median_std": float("nan"),
            "p90_std": float("nan"),
            "corr_abs_error_std": float("nan"),
            "coverage_1sigma": float("nan"),
            "coverage_2sigma": float("nan"),
            "mean_ale_std": float("nan"),
            "mean_epi_std": float("nan"),
            "corr_abs_error_ale_std": float("nan"),
            "corr_abs_error_epi_std": float("nan"),
            "corr_dock_ale_std": float("nan"),
            "corr_dock_epi_std": float("nan"),
            "mean_epi_over_ale": float("nan"),
            "median_epi_over_ale": float("nan"),
            "std_log_epi_over_ale": float("nan"),
            "spearman_epi_ale": float("nan"),
            "corr_abs_error_epi_over_ale": float("nan"),
            "corr_dock_epi_over_ale": float("nan"),
            "mean_nu": float("nan"),
            "mean_alpha": float("nan"),
            "mean_beta": float("nan"),
        }

    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(targets, dim=0)
    total_std = torch.cat(total_std_terms, dim=0)
    ale_std = torch.cat(ale_std_terms, dim=0)
    epi_std = torch.cat(epi_std_terms, dim=0)

    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(y_pred - y_true)).item()

    denom = torch.sum((y_true - y_true.mean()) ** 2).item()
    r2 = 1.0 - (torch.sum((y_pred - y_true) ** 2).item() / denom) if denom > 0 else float("nan")

    # support both per-sample NLL and batch-scalar NLL
    nll_sum = 0.0
    nll_count = 0
    for nll_flat in nll_terms:
        nll_flat = torch.as_tensor(nll_flat).reshape(-1)
        if nll_flat.numel() == 0:
            continue
        if nll_flat.numel() > 1:
            nll_sum += float(nll_flat.sum().item())
            nll_count += int(nll_flat.numel())
        else:
            nll_sum += float(nll_flat.item())
            nll_count += 1
    nll = float(nll_sum / max(nll_count, 1))

    abs_err = torch.abs(y_pred - y_true)
    corr_abs_error_std = _safe_tensor_corr(abs_err, total_std)
    corr_abs_error_ale_std = _safe_tensor_corr(abs_err, ale_std)

    epi_finite_mask = torch.isfinite(epi_std)
    epi_finite = epi_std[epi_finite_mask]
    mean_epi_std = epi_finite.mean().item() if epi_finite.numel() > 0 else float("nan")
    corr_abs_error_epi_std = _safe_tensor_corr(abs_err[epi_finite_mask], epi_finite) if epi_finite.numel() > 2 else float("nan")

    coverage_1sigma = torch.mean((abs_err <= total_std).to(torch.float32)).item()
    coverage_2sigma = torch.mean((abs_err <= 2.0 * total_std).to(torch.float32)).item()

    finite_pair = torch.isfinite(epi_std) & torch.isfinite(ale_std) & (ale_std > 0)
    if finite_pair.sum().item() > 2:
        epi_pair = epi_std[finite_pair]
        ale_pair = ale_std[finite_pair]
        abs_pair = abs_err[finite_pair]
        y_pair = y_true[finite_pair]

        epi_over_ale = epi_pair / ale_pair.clamp_min(1e-12)
        log_epi_over_ale = torch.log(epi_over_ale.clamp_min(1e-12))

        mean_epi_over_ale = epi_over_ale.mean().item()
        median_epi_over_ale = epi_over_ale.median().item()
        std_log_epi_over_ale = log_epi_over_ale.std().item() if log_epi_over_ale.numel() > 1 else 0.0
        spearman_epi_ale = _safe_tensor_corr(epi_pair, ale_pair)
        corr_abs_error_epi_over_ale = _safe_tensor_corr(abs_pair, epi_over_ale)
        corr_dock_epi_over_ale = _safe_tensor_corr(y_pair, epi_over_ale)
    else:
        mean_epi_over_ale = float("nan")
        median_epi_over_ale = float("nan")
        std_log_epi_over_ale = float("nan")
        spearman_epi_ale = float("nan")
        corr_abs_error_epi_over_ale = float("nan")
        corr_dock_epi_over_ale = float("nan")

    corr_dock_ale_std = _safe_tensor_corr(y_true, ale_std)
    corr_dock_epi_std = _safe_tensor_corr(y_true[epi_finite_mask], epi_finite) if epi_finite.numel() > 2 else float("nan")

    if nu_terms:
        nu_all = torch.cat(nu_terms, dim=0)
        alpha_all = torch.cat(alpha_terms, dim=0)
        beta_all = torch.cat(beta_terms, dim=0)
        mean_nu = nu_all.mean().item()
        mean_alpha = alpha_all.mean().item()
        mean_beta = beta_all.mean().item()
    else:
        mean_nu = float("nan")
        mean_alpha = float("nan")
        mean_beta = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nll": nll,
        "mean_std": total_std.mean().item(),
        "median_std": total_std.median().item(),
        "p90_std": torch.quantile(total_std, 0.9).item(),
        "corr_abs_error_std": corr_abs_error_std,
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma,
        "mean_ale_std": ale_std.mean().item(),
        "mean_epi_std": mean_epi_std,
        "corr_abs_error_ale_std": corr_abs_error_ale_std,
        "corr_abs_error_epi_std": corr_abs_error_epi_std,
        "corr_dock_ale_std": corr_dock_ale_std,
        "corr_dock_epi_std": corr_dock_epi_std,
        "mean_epi_over_ale": mean_epi_over_ale,
        "median_epi_over_ale": median_epi_over_ale,
        "std_log_epi_over_ale": std_log_epi_over_ale,
        "spearman_epi_ale": spearman_epi_ale,
        "corr_abs_error_epi_over_ale": corr_abs_error_epi_over_ale,
        "corr_dock_epi_over_ale": corr_dock_epi_over_ale,
        "mean_nu": mean_nu,
        "mean_alpha": mean_alpha,
        "mean_beta": mean_beta,
    }


def load_surrogate(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    backbone = str(cfg.get("backbone", "gin")).lower() if isinstance(cfg, dict) else "gin"
    surrogate_meta: dict = {}

    if backbone in {"tensornet", "tsa", "tensor"}:
        torchmd_cfg = cfg.get("torchmd", {}) if isinstance(cfg, dict) else {}
        fp_dim = int(cfg.get("fp_dim", 0)) if isinstance(cfg, dict) else 0
        fp_radius = int(cfg.get("fp_radius", 2)) if isinstance(cfg, dict) else 2
        embedding_dim = int(torchmd_cfg.get("embedding_dim", cfg.get("torchmd_embedding_dim", 128)))
        num_layers = int(torchmd_cfg.get("num_layers", cfg.get("torchmd_num_layers", 2)))
        num_rbf = int(torchmd_cfg.get("num_rbf", cfg.get("torchmd_num_rbf", 32)))
        rbf_type = str(torchmd_cfg.get("rbf_type", cfg.get("torchmd_rbf_type", "expnorm")))
        trainable_rbf = bool(torchmd_cfg.get("trainable_rbf", cfg.get("torchmd_trainable_rbf", False)))
        activation = str(torchmd_cfg.get("activation", cfg.get("torchmd_activation", "silu")))
        cutoff_lower = float(torchmd_cfg.get("cutoff_lower", cfg.get("torchmd_cutoff_lower", 0.0)))
        cutoff_upper = float(torchmd_cfg.get("cutoff_upper", cfg.get("torchmd_cutoff_upper", 4.5)))
        node_feat_dim = torchmd_cfg.get("node_feat_dim", cfg.get("torchmd_node_feat_dim", 0))
        node_feat_dim = int(node_feat_dim) if node_feat_dim else 0
        if node_feat_dim <= 0:
            raise RuntimeError("TensorNet checkpoint missing node_feat_dim; retrain with atom-vocab features.")
        max_num_neighbors = int(torchmd_cfg.get("max_num_neighbors", cfg.get("torchmd_max_num_neighbors", 64)))
        equivariance_group = str(
            torchmd_cfg.get(
                "equivariance_invariance_group",
                torchmd_cfg.get(
                    "equivariance_group",
                    cfg.get(
                        "torchmd_equivariance_invariance_group",
                        cfg.get("torchmd_equivariance_group", "O(3)"),
                    ),
                ),
            )
        )
        static_shapes = bool(torchmd_cfg.get("static_shapes", cfg.get("torchmd_static_shapes", True)))
        check_errors = bool(torchmd_cfg.get("check_errors", cfg.get("torchmd_check_errors", True)))
        dropout = float(torchmd_cfg.get("dropout", cfg.get("torchmd_dropout", 0.1)))
        reduce_op = str(torchmd_cfg.get("reduce_op", cfg.get("torchmd_reduce", "sum")))
        head_hidden_dim = torchmd_cfg.get("head_hidden_dim", cfg.get("torchmd_head_hidden_dim", None))
        head_hidden_dim = None if head_hidden_dim in (None, "", "null") else int(head_hidden_dim)
        head_num_layers = int(torchmd_cfg.get("head_num_layers", cfg.get("torchmd_head_num_layers", 3)))
        ensemble_heads = int(cfg.get("ensemble_heads", torchmd_cfg.get("ensemble_heads", 1)))
        uncertainty_mode = str(cfg.get("uncertainty_mode", torchmd_cfg.get("uncertainty_mode", "gaussian"))).lower().strip()
        if uncertainty_mode in {"nig", "bayes"}:
            ensemble_heads = 1
        if str(cfg.get("ensemble_type", "")).lower() == "independent":
            model = _load_independent_tensornet_ensemble(ckpt, device)
        else:
            model_kwargs = dict(
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_type=rbf_type,
                trainable_rbf=trainable_rbf,
                activation=activation,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                node_feat_dim=node_feat_dim,
                max_num_neighbors=max_num_neighbors,
                equivariance_invariance_group=equivariance_group,
                static_shapes=static_shapes,
                check_errors=check_errors,
                dropout=dropout,
                reduce_op=reduce_op,
                fp_dim=int(cfg.get("fp_dim", 0)) if isinstance(cfg, dict) else 0,
                head_hidden_dim=head_hidden_dim,
                head_num_layers=head_num_layers,
            )
            if uncertainty_mode == "nig":
                model = TensorNetNIGSurrogate(**model_kwargs)
            elif uncertainty_mode == "bayes":
                model = TensorNetBayesSurrogate(**model_kwargs)
            elif ensemble_heads > 1:
                model = TensorNetEnsembleSurrogate(num_heads=ensemble_heads, **model_kwargs)
            else:
                model = TensorNetSurrogate(**model_kwargs)
            model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device)
        mean = float(ckpt.get("mean", 0.0))
        std = float(ckpt.get("std", 1.0))
        surrogate_meta = {"backbone": "tensornet", "node_feat_dim": node_feat_dim, "fp_dim": fp_dim, "fp_radius": fp_radius, "ensemble_heads": ensemble_heads, "ensemble_type": str(cfg.get("ensemble_type", "shared")).lower(), "ensemble_scheme": str(cfg.get("ensemble_scheme", "single")).lower(), "uncertainty_mode": uncertainty_mode, "eval_samples": int(cfg.get("eval_samples", 8))}
        return (
            model,
            mean,
            std,
            0,
            0,
            fp_dim,
            fp_radius,
            0,
            0,
            False,
            "tensornet",
            surrogate_meta,
        )
    if backbone not in {"gin"}:
        raise RuntimeError(f"Unsupported surrogate backbone '{backbone}'. Use 'gin' or 'tensornet'.")

    node_dim = int(ckpt["node_dim"])
    edge_dim = int(ckpt["edge_dim"])
    fp_dim = int(cfg.get("fp_dim", 0)) if isinstance(cfg, dict) else 0
    fp_radius = int(cfg.get("fp_radius", 2)) if isinstance(cfg, dict) else 2
    atom_extra_dim = int(cfg.get("atom_extra_dim", 0)) if isinstance(cfg, dict) else 0
    bond_extra_dim = int(cfg.get("bond_extra_dim", 0)) if isinstance(cfg, dict) else 0
    model_pocket_graph = bool(cfg.get("pocket_graph", False)) if isinstance(cfg, dict) else False
    model = GINSurrogate(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layers=int(cfg.get("num_layers", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_edge_attr=not bool(cfg.get("no_edge_attr", False)),
        use_ligand_mask=not bool(cfg.get("no_ligand_mask", False)),
        fp_dim=fp_dim,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    mean = float(ckpt.get("mean", 0.0))
    std = float(ckpt.get("std", 1.0))
    return (
        model,
        mean,
        std,
        node_dim,
        edge_dim,
        atom_extra_dim,
        bond_extra_dim,
        fp_dim,
        fp_radius,
        model_pocket_graph,
        "gin",
        surrogate_meta,
    )


def _gaussian_moments_from_model(
    model: torch.nn.Module,
    batch,
    *,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
    min_var: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "decompose"):
        mu_z, _ale_var_z, _epi_var_z, total_var_z = model.decompose(batch)
        return mu_z, total_var_z.clamp_min(min_var)
    raw = model(batch)
    mu_z, raw_log_var = split_gaussian_output(raw)
    var_z = gaussian_variance_from_raw(raw_log_var, logvar_min=logvar_min, logvar_max=logvar_max, min_var=min_var)
    return mu_z, var_z


def predict_dock_for_smiles(
    model: torch.nn.Module,
    smiles_list: Sequence[str],
    device: torch.device,
    mean: float,
    std: float,
    use_torchmd: bool,
    atom_index: Dict[str, int] | None = None,
    node_dim: int = 0,
    edge_dim: int = 0,
    atom_extra_dim: int | None = None,
    bond_extra_dim: int | None = None,
    fp_dim: int = 0,
    fp_radius: int = 2,
    confgen_max_attempts: int = 10,
    confgen_seed: int = 0,
    confgen_num_confs: int = 10,
    confgen_max_opt_iters: int = 200,
    confgen_optimize: bool = True,
    confgen_prefer_mmff: bool = True,
) -> list[float]:
    if atom_index is None:
        raise ValueError("atom_index is required for surrogate prediction.")
    preds = [float("nan")] * len(smiles_list)
    data_list: list[Data] = []
    idxs: list[int] = []
    for i, smi in enumerate(smiles_list):
        if use_torchmd:
            atom_feat_dim = len(atom_index) + (atom_extra_dim or 0)
            data = smiles_to_ligand_3d(
                smi,
                atom_index=atom_index,
                atom_feat_dim=atom_feat_dim,
                atom_extra_dim=int(atom_extra_dim or 0),
                fp_dim=fp_dim,
                fp_radius=fp_radius,
                max_attempts=confgen_max_attempts,
                seed=confgen_seed,
                num_confs=confgen_num_confs,
                max_opt_iters=confgen_max_opt_iters,
                optimize=confgen_optimize,
                prefer_mmff=confgen_prefer_mmff,
            )
        else:
            data = smiles_to_ligand_graph(
                smi,
                atom_index,
                node_dim,
                edge_dim,
                atom_extra_dim=atom_extra_dim,
                bond_extra_dim=bond_extra_dim,
                fp_dim=fp_dim,
                fp_radius=fp_radius,
            )
        data_list.append(data)
        idxs.append(i)
    if not data_list:
        raise RuntimeError("No valid molecules converted for surrogate prediction.")
    loader = DataLoader(data_list, batch_size=256, shuffle=False)
    model.eval()
    out_vals: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu_z, _var_z = _gaussian_moments_from_model(model, batch)
            pred = mu_z * std + mean
            out_vals.extend(pred.detach().cpu().view(-1).tolist())
    for i, val in zip(idxs, out_vals):
        preds[i] = float(val)
    return preds


def mc_samples(
    model: torch.nn.Module,
    loader: DataLoader,
    mean: float,
    std: float,
    num_samples: int,
    device: torch.device,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
    min_var: float = 1e-6,
) -> torch.Tensor:
    model.eval()
    mu_chunks = []
    var_chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu_z, var_z = _gaussian_moments_from_model(model, batch, logvar_min=logvar_min, logvar_max=logvar_max, min_var=min_var)
            mu_chunks.append(mu_z.detach().cpu())
            var_chunks.append(var_z.detach().cpu())
    if not mu_chunks:
        raise RuntimeError("No predictions were produced for Gaussian surrogate sampling.")
    mu = torch.cat(mu_chunks, dim=0)
    var = torch.cat(var_chunks, dim=0)
    stddev = torch.sqrt(var)
    draws = []
    for _ in range(int(num_samples)):
        eps = torch.randn_like(mu)
        sample_z = mu + stddev * eps
        draws.append(sample_z * std + mean)
    return torch.stack(draws, dim=0)


def predict_gaussian_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
    min_var: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    mu_chunks = []
    var_chunks = []
    scale = float(std)
    shift = float(mean)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu_z, var_z = _gaussian_moments_from_model(model, batch, logvar_min=logvar_min, logvar_max=logvar_max, min_var=min_var)
            mu_chunks.append((mu_z * scale + shift).detach().cpu())
            var_chunks.append((var_z * (scale ** 2)).detach().cpu())
    if not mu_chunks:
        raise RuntimeError("No predictions were produced for Gaussian surrogate statistics.")
    mu = torch.cat(mu_chunks, dim=0).view(-1)
    var = torch.cat(var_chunks, dim=0).view(-1).clamp_min(min_var * (scale ** 2))
    return mu, torch.sqrt(var)
def predict_gaussian_decomposed_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
    min_var: float = 1e-6,
    eval_samples: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    uncertainty_mode = str(getattr(model, "uncertainty_mode", "gaussian")).strip().lower()
    if uncertainty_mode == "bayes":
        model.eval()
        mu_chunks = []
        total_chunks = []
        ale_chunks = []
        epi_chunks = []
        scale = float(std)
        shift = float(mean)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                draws = []
                for _ in range(max(1, int(eval_samples))):
                    model.sample_bayes_params()
                    draws.append(model(batch).detach().view(-1))
                    model.clear_bayes_params()
                stacked = torch.stack(draws, dim=0)
                mu_z = stacked.mean(dim=0)
                if stacked.size(0) == 1:
                    epi_var_z = torch.full_like(mu_z, float(min_var))
                else:
                    epi_var_z = stacked.var(dim=0, unbiased=False).clamp_min(min_var)
                zero_ale_var_z = torch.zeros_like(epi_var_z)
                mu_chunks.append((mu_z * scale + shift).detach().cpu())
                ale_chunks.append((zero_ale_var_z * (scale ** 2)).detach().cpu())
                epi_chunks.append((epi_var_z * (scale ** 2)).detach().cpu())
                total_chunks.append((epi_var_z * (scale ** 2)).detach().cpu())

        if not mu_chunks:
            raise RuntimeError("No predictions were produced for Bayesian surrogate statistics.")
        mu = torch.cat(mu_chunks, dim=0).view(-1)
        total_var = torch.cat(total_chunks, dim=0).view(-1).clamp_min(min_var * (scale ** 2))
        ale_var = torch.cat(ale_chunks, dim=0).view(-1)
        epi_var = torch.cat(epi_chunks, dim=0).view(-1).clamp_min(0.0)
        return mu, torch.sqrt(total_var), torch.sqrt(epi_var), torch.sqrt(ale_var)

    if not hasattr(model, "decompose"):
        mu, std_total = predict_gaussian_stats(model, loader, mean=mean, std=std, device=device, min_var=min_var)
        nan = torch.full_like(std_total, float("nan"))
        return mu, std_total, nan, nan

    model.eval()
    mu_chunks = []
    total_chunks = []
    ale_chunks = []
    epi_chunks = []
    scale = float(std)
    shift = float(mean)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu_z, ale_var_z, epi_var_z, total_var_z = model.decompose(batch)
            mu_chunks.append((mu_z * scale + shift).detach().cpu())
            ale_chunks.append((ale_var_z * (scale ** 2)).detach().cpu())
            epi_chunks.append((epi_var_z * (scale ** 2)).detach().cpu())
            total_chunks.append((total_var_z * (scale ** 2)).detach().cpu())

    mu = torch.cat(mu_chunks, dim=0).view(-1)
    total_var = torch.cat(total_chunks, dim=0).view(-1).clamp_min(min_var * (scale ** 2))
    ale_var = torch.cat(ale_chunks, dim=0).view(-1).clamp_min(min_var * (scale ** 2))
    epi_var = torch.cat(epi_chunks, dim=0).view(-1).clamp_min(0.0)

    return mu, torch.sqrt(total_var), torch.sqrt(epi_var), torch.sqrt(ale_var)

def load_train_objectives(
    dataset_root: str,
    split: str,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float | None = None,
    use_sa: bool = False,
    sa_clamp_min: float | None = None,
    sa_clamp_max: float | None = None,
    dock_valid_max: float | None = 0.0,
    split_ratio: Tuple[float, float, float] = (0.9, 0.1, 0.0),
    split_seed: int = 42,
) -> torch.Tensor | None:
    df = _read_smiles_csv(str(Path(dataset_root) / "smiles.csv"))
    if df.empty:
        raise RuntimeError(f"Empty smiles.csv in dataset root: {dataset_root}")
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()]
    else:
        if split.lower() != "train":
            df = df.iloc[0:0]
    if df.empty:
        raise RuntimeError(f"No rows found for split='{split}' in smiles.csv: {dataset_root}")
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        raise RuntimeError("smiles.csv missing smiles column.")
    values = []
    for _, row in df.iterrows():

        dock = float(row.get("dock_score", float("nan")))

        if not _is_valid_dock_score(dock, dock_valid_max=dock_valid_max):
            raise RuntimeError(f"Invalid dock_score encountered in split='{split}': {dock}")
        smi = str(row.get(smiles_col, "")).strip()
        if not smi:
            raise RuntimeError("Encountered empty SMILES while loading train objectives.")
        qed_val = compute_qed(smi)
        if not np.isfinite(qed_val):
            raise RuntimeError(f"Non-finite QED for SMILES: {smi}")
        sa_val = compute_sa(smi, clamp_min=sa_clamp_min, clamp_max=sa_clamp_max)
        if not np.isfinite(sa_val):
            raise RuntimeError(f"Non-finite SA for SMILES: {smi}")
        sign = sa_sign if sa_sign is not None else -1.0
        values.append([dock_sign * dock, qed_sign * qed_val, sign * sa_val])
    if not values:
        raise RuntimeError("No objective rows were produced from smiles.csv.")
    return torch.tensor(values, dtype=torch.float32)


def train_surrogate_from_scratch(
    root: str,
    save_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    use_edge_attr: bool,
    use_ligand_mask: bool,
    standardize: bool,
    fp_dim: int,
    fp_radius: int,
    eval_samples: int,
    scheduler: str,
    warmup_epochs: int,
    min_lr: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    device: torch.device,
    dock_valid_max: float | None = 0.0,
    backbone: str = "gin",
    torchmd_cfg: dict | None = None,
    ensemble_heads: int = 1,
    freeze_backbone: bool = False,
    pretrained_encoder_ckpt: str | None = None,
    uncertainty_mode: str = "nig",
    ensemble_scheme: str = "full",
    ensemble_bootstrap: bool = True,
    random_seed: int = 0,
    auto_prepare: bool = False,
    vina_executable: str | Path | None = None,
    ligand3d_cache_dir: str | Path | None = None,
    ligand_vocab_override: Sequence[str] | None = None,
    ligand3d_store: LigandOnly3DStore | None = None,
    test_ligand_ids: Sequence[str] | None = None,
    confgen_max_attempts: int = 10,
    confgen_seed: int = 0,
    confgen_num_confs: int = 10,
    confgen_max_opt_iters: int = 200,
    confgen_optimize: bool = True,
    confgen_prefer_mmff: bool = True,
    ligand3d_num_workers: int = 1,
    ligand3d_mp_chunksize: int = 16,
    encoder_lr: float | None = None,
    train_log_csv: str | None = None,
    train_summary_json: str | None = None,
) -> str:
    backbone = str(backbone).lower()
    if backbone not in {"tensornet", "tsa", "tensor"}:
        if int(ensemble_heads) != 1:
            raise ValueError("ensemble_heads > 1 is only supported for TensorNet backbones.")
        if freeze_backbone:
            raise ValueError("freeze_backbone is only supported for TensorNet backbones.")
        if pretrained_encoder_ckpt is not None:
            raise ValueError("pretrained_encoder_ckpt is only supported for TensorNet backbones.")
    ensemble_scheme = str(ensemble_scheme).lower().strip()
    gaussian_select_by = str(((torchmd_cfg or {}) if isinstance(torchmd_cfg, dict) else {}).get("gaussian_select_by", "rmse")).lower()
    gaussian_select_nll_weight = float(((torchmd_cfg or {}) if isinstance(torchmd_cfg, dict) else {}).get("gaussian_select_nll_weight", 0.05))
    torchmd_cfg_used = None
    if backbone in {"tensornet", "tsa", "tensor"}:
        cfg = torchmd_cfg or {}
        ensemble_heads = int(ensemble_heads)
        if ensemble_heads <= 0:
            raise ValueError("ensemble_heads must be >= 1.")
        if uncertainty_mode in {"nig", "bayes"}:
            if ensemble_heads != 1:
                raise ValueError("TensorNet NIG/Bayesian surrogate does not support ensemble_heads > 1.")
        if ensemble_heads == 1 and ensemble_scheme not in {"full", "last_layer"}:
            raise ValueError("Single-model TensorNet ensemble_scheme must be 'full' or 'last_layer'.")
        if ensemble_heads == 1 and ensemble_scheme == "last_layer" and pretrained_encoder_ckpt is None:
            raise ValueError("Single-model TensorNet with ensemble_scheme='last_layer' requires pretrained_encoder_ckpt.")
        if ensemble_heads == 1 and ensemble_scheme == "last_layer" and freeze_backbone:
            raise ValueError("freeze_backbone conflicts with ensemble_scheme='last_layer' for single-model TensorNet.")
        if ensemble_heads > 1 and ensemble_scheme not in {"full", "last_layer"}:
            raise ValueError("TensorNet ensemble_scheme must be 'full' or 'last_layer' when ensemble_heads > 1.")
        if ensemble_heads > 1 and freeze_backbone:
            raise ValueError("independent TensorNet ensembles train member-specific backbones; use ensemble_scheme='last_layer' for frozen-backbone ensembles.")
        local_ligand3d_store = ligand3d_store
        if local_ligand3d_store is None:
            if ligand_vocab_override:
                shared_ligand_vocab = list(ligand_vocab_override)
            else:
                smiles_df = _read_smiles_csv(str(Path(root) / "smiles.csv"))
                smiles_col = _pick_smiles_column(smiles_df.columns)
                if smiles_col is None:
                    raise RuntimeError("smiles.csv missing smiles column for TensorNet vocab inference.")
                shared_ligand_vocab = _load_ligand_vocab_override(Path(root)) or _scan_ligand_vocab(
                    smiles_df[smiles_col].astype(str).tolist()
                )
            local_ligand3d_store = LigandOnly3DStore(
                root,
                ligand_vocab_override=shared_ligand_vocab,
                cache_dir=ligand3d_cache_dir,
                fp_dim=fp_dim,
                fp_radius=fp_radius,
                confgen_max_attempts=confgen_max_attempts,
                confgen_seed=confgen_seed,
                confgen_num_confs=confgen_num_confs,
                confgen_max_opt_iters=confgen_max_opt_iters,
                confgen_optimize=confgen_optimize,
                confgen_prefer_mmff=confgen_prefer_mmff,
                build_num_workers=ligand3d_num_workers,
                build_mp_chunksize=ligand3d_mp_chunksize,
            )
        test_indices = []
        if test_ligand_ids:
            for lig_id in test_ligand_ids:
                idx = local_ligand3d_store.id_to_idx.get(str(lig_id))
                if idx is not None:
                    test_indices.append(idx)
        train_indices = list(local_ligand3d_store.split_indices.get("train", []))
        valid_indices = list(local_ligand3d_store.split_indices.get("valid", []))
        if test_indices:
            test_set = set(test_indices)
            train_indices = [i for i in train_indices if i not in test_set]
            valid_indices = [i for i in valid_indices if i not in test_set]
        train_ds = local_ligand3d_store.get_dataset_from_indices(train_indices)
        val_ds = local_ligand3d_store.get_dataset_from_indices(valid_indices)
        test_ds = (
            local_ligand3d_store.get_dataset_from_indices(test_indices)
            if test_indices
            else local_ligand3d_store.get_split_dataset("test")
        )
        if len(train_ds) == 0:
            raise RuntimeError(
                "No training samples available for 3D surrogate. "
                "Ensure smiles.csv has docking poses/scores or enable auto_prepare."
            )
        if len(val_ds) == 0:
            val_ds = train_ds
        if len(test_ds) == 0:
            test_ds = train_ds
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        ensemble_train_loaders = (
            _build_head_loaders(train_ds, batch_size=batch_size, num_heads=ensemble_heads, loader_cls=DataLoader, shuffle=True)
            if ensemble_heads > 1
            else None
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        node_dim = 0
        edge_dim = 0
        atom_extra_dim = 0
        bond_extra_dim = 0

        torchmd_cfg_used = {
            "embedding_dim": int(cfg.get("embedding_dim", hidden_dim)),
            "num_layers": int(cfg.get("num_layers", 2)),
            "num_rbf": int(cfg.get("num_rbf", 32)),
            "rbf_type": str(cfg.get("rbf_type", "expnorm")),
            "trainable_rbf": bool(cfg.get("trainable_rbf", False)),
            "activation": str(cfg.get("activation", "silu")),
            "cutoff_lower": float(cfg.get("cutoff_lower", 0.0)),
            "cutoff_upper": float(cfg.get("cutoff_upper", 4.5)),
            "node_feat_dim": int(getattr(train_ds, "num_node_classes_", 0)),
            "max_num_neighbors": int(cfg.get("max_num_neighbors", 64)),
            "equivariance_invariance_group": str(
                cfg.get("equivariance_invariance_group", cfg.get("equivariance_group", "O(3)"))
            ),
            "static_shapes": bool(cfg.get("static_shapes", True)),
            "check_errors": bool(cfg.get("check_errors", True)),
            "dropout": float(cfg.get("dropout", dropout)),
            "reduce_op": str(cfg.get("reduce_op", "sum")),
            "head_hidden_dim": None
            if cfg.get("head_hidden_dim", None) in (None, "", "null")
            else int(cfg.get("head_hidden_dim")),
            "head_num_layers": int(cfg.get("head_num_layers", 3)),
            "bayes_beta": float(cfg.get("bayes_beta", 1.0e-5)),
            "gaussian_warmup_epochs": int(cfg.get("gaussian_warmup_epochs", 5)),
            "gaussian_var_reg_beta": float(cfg.get("gaussian_var_reg_beta", 1e-4)),
            "gaussian_logvar_min": float(cfg.get("gaussian_logvar_min", -8.0)),
            "gaussian_logvar_max": float(cfg.get("gaussian_logvar_max", 4.0)),
            "gaussian_min_var": float(cfg.get("gaussian_min_var", 1e-6)),
            "ensemble_heads": ensemble_heads,
            "gaussian_select_by": str(cfg.get("gaussian_select_by", "rmse")).lower(),
            "gaussian_select_nll_weight": float(cfg.get("gaussian_select_nll_weight", 0.05)),
        }
        gaussian_select_by = str((torchmd_cfg_used or {}).get("gaussian_select_by", "rmse")).lower()
        gaussian_select_nll_weight = float((torchmd_cfg_used or {}).get("gaussian_select_nll_weight", 0.05))
        if ensemble_heads > 1:
            if standardize:
                mean, std = fit_target_stats(train_loader, device, dock_valid_max=dock_valid_max)
            else:
                mean, std = 0.0, 1.0
            return _train_independent_tensornet_ensemble(
                root=root,
                save_path=save_path,
                train_dataset=train_ds,
                val_loader=val_loader,
                test_loader=test_loader,
                node_feat_dim=int(torchmd_cfg_used["node_feat_dim"]),
                torchmd_cfg=dict(torchmd_cfg_used),
                fp_dim=int(fp_dim),
                fp_radius=int(fp_radius),
                pretrained_encoder_ckpt=(None if pretrained_encoder_ckpt is None else str(pretrained_encoder_ckpt)),
                device=device,
                ensemble_scheme=ensemble_scheme,
                ensemble_heads=int(ensemble_heads),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                encoder_lr=float(lr if encoder_lr is None else encoder_lr),
                weight_decay=float(weight_decay),
                warmup_epochs=int(warmup_epochs),
                min_lr=float(min_lr),
                early_stop_patience=int(early_stop_patience),
                early_stop_min_delta=float(early_stop_min_delta),
                mean=float(mean),
                std=float(std),
                dock_valid_max=dock_valid_max,
                train_log_csv=train_log_csv,
                train_summary_json=train_summary_json,
                ensemble_bootstrap=bool(ensemble_bootstrap),
                random_seed=int(random_seed),
            )
        if torchmd_cfg_used["head_hidden_dim"] is None:
            torchmd_cfg_used["head_hidden_dim"] = torchmd_cfg_used["embedding_dim"]
        model_kwargs = dict(
            embedding_dim=int(cfg.get("embedding_dim", hidden_dim)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_rbf=int(cfg.get("num_rbf", 32)),
            rbf_type=str(cfg.get("rbf_type", "expnorm")),
            trainable_rbf=bool(cfg.get("trainable_rbf", False)),
            activation=str(cfg.get("activation", "silu")),
            cutoff_lower=float(cfg.get("cutoff_lower", 0.0)),
            cutoff_upper=float(cfg.get("cutoff_upper", 4.5)),
            node_feat_dim=int(getattr(train_ds, "num_node_classes_", 0)),
            max_num_neighbors=int(cfg.get("max_num_neighbors", 64)),
            equivariance_invariance_group=str(
                cfg.get("equivariance_invariance_group", cfg.get("equivariance_group", "O(3)"))
            ),
            static_shapes=bool(cfg.get("static_shapes", True)),
            check_errors=bool(cfg.get("check_errors", True)),
            dropout=float(cfg.get("dropout", dropout)),
            reduce_op=str(cfg.get("reduce_op", "sum")),
            fp_dim=int(fp_dim),
            head_hidden_dim=torchmd_cfg_used["head_hidden_dim"],
            head_num_layers=torchmd_cfg_used["head_num_layers"],
        )
        if uncertainty_mode == "nig":
            model = TensorNetNIGSurrogate(**model_kwargs).to(device)
        elif uncertainty_mode == "bayes":
            model = TensorNetBayesSurrogate(**model_kwargs).to(device)
        elif ensemble_heads > 1:
            model = TensorNetEnsembleSurrogate(num_heads=ensemble_heads, **model_kwargs).to(device)
        else:
            model = TensorNetSurrogate(**model_kwargs).to(device)
        if pretrained_encoder_ckpt is not None:
            encoder_ckpt = torch.load(pretrained_encoder_ckpt, map_location=device)
            encoder_state = encoder_ckpt.get("encoder_state")
            if encoder_state is None:
                raise RuntimeError("Pretrained encoder checkpoint missing encoder_state.")
            model.encoder.load_state_dict(encoder_state, strict=True)
        if ensemble_heads == 1 and ensemble_scheme == "last_layer":
            _set_tensornet_single_model_train_scheme(model, ensemble_scheme)
        if freeze_backbone:
            for param in model.encoder.parameters():
                param.requires_grad = False
        log_model_param_report(model)
    else:
        if backbone not in {"gin"}:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use 'gin' or 'tensornet'.")
        smiles_df = _read_smiles_csv(str(Path(root) / "smiles.csv"))
        smiles_col = _pick_smiles_column(smiles_df.columns)
        if smiles_col is None:
            raise RuntimeError("smiles.csv missing smiles column for GIN vocab inference.")
        if ligand_vocab_override:
            ligand_vocab = list(ligand_vocab_override)
        else:
            ligand_vocab = _load_ligand_vocab_override(Path(root)) or _scan_ligand_vocab(
                smiles_df[smiles_col].astype(str).tolist()
            )
        atom_extra_dim = ATOM_EXTRA_DIM
        bond_extra_dim = BOND_EXTRA_DIM
        train_ds = build_ligand_graphs_from_smiles_csv(
            root,
            "train",
            ligand_vocab,
            atom_extra_dim,
            bond_extra_dim,
            fp_dim,
            fp_radius,
            exclude_ids=test_ligand_ids,
        )
        val_ds = build_ligand_graphs_from_smiles_csv(
            root,
            "valid",
            ligand_vocab,
            atom_extra_dim,
            bond_extra_dim,
            fp_dim,
            fp_radius,
            exclude_ids=test_ligand_ids,
        )
        test_ds = (
            build_ligand_graphs_from_smiles_csv(
                root,
                "train",
                ligand_vocab,
                atom_extra_dim,
                bond_extra_dim,
                fp_dim,
                fp_radius,
                include_ids=test_ligand_ids,
            )
            if test_ligand_ids
            else build_ligand_graphs_from_smiles_csv(
                root, "test", ligand_vocab, atom_extra_dim, bond_extra_dim, fp_dim, fp_radius
            )
        )
        if len(train_ds) == 0:
            raise RuntimeError(
                "No training samples available for surrogate. "
                "Ensure smiles.csv has docking poses/scores or enable auto_prepare."
            )
        if len(val_ds) == 0:
            val_ds = list(train_ds)
        if len(test_ds) == 0:
            test_ds = list(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        ensemble_train_loaders = None
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        node_dim = len(ligand_vocab) + atom_extra_dim
        sample = train_ds[0]
        if use_edge_attr and hasattr(sample, "edge_attr") and sample.edge_attr is not None:
            edge_dim = int(sample.edge_attr.size(-1))
        else:
            edge_dim = BOND_TYPE_CLASSES + bond_extra_dim

        model = GINSurrogate(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_edge_attr=use_edge_attr,
            use_ligand_mask=use_ligand_mask,
            fp_dim=fp_dim,
        ).to(device)
        log_model_param_report(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable surrogate parameters remain after freezing.")
    encoder_lr_value = float(lr if encoder_lr is None else encoder_lr)
    optimizer_group_labels = ["main"]
    if backbone in {"tensornet", "tsa", "tensor"}:
        encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
        encoder_param_ids = {id(param) for param in encoder_params}
        main_params = [param for param in trainable_params if id(param) not in encoder_param_ids]
        param_groups = []
        optimizer_group_labels = []
        if main_params:
            param_groups.append({"params": main_params, "lr": float(lr)})
            optimizer_group_labels.append("main")
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": float(encoder_lr_value)})
            optimizer_group_labels.append("encoder")
        if not param_groups:
            raise RuntimeError("No optimizer parameter groups were constructed for surrogate training.")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler_name = str(scheduler or "none").lower()
    warmup_epochs = int(warmup_epochs)
    min_lr = float(min_lr)
    early_stop_patience = int(early_stop_patience)
    early_stop_min_delta = float(early_stop_min_delta)
    gaussian_warmup_epochs = int((torchmd_cfg_used or {}).get("gaussian_warmup_epochs", 5))
    gaussian_var_reg_beta = float((torchmd_cfg_used or {}).get("gaussian_var_reg_beta", 1e-4))
    gaussian_logvar_min = float((torchmd_cfg_used or {}).get("gaussian_logvar_min", -8.0))
    gaussian_logvar_max = float((torchmd_cfg_used or {}).get("gaussian_logvar_max", 4.0))
    gaussian_min_var = float((torchmd_cfg_used or {}).get("gaussian_min_var", 1e-6))
    nig_warmup_epochs = int((torchmd_cfg_used or {}).get("nig_warmup_epochs", gaussian_warmup_epochs))
    nig_reg_lambda = float(
        (torchmd_cfg_used or {}).get("nig_reg_lambda", (torchmd_cfg_used or {}).get("evidential_reg_lambda", 1e-2))
    )
    bayes_beta = float((torchmd_cfg_used or {}).get("bayes_beta", 1.0e-5))

    if backbone not in {"tensornet", "tsa", "tensor"}:
        ensemble_train_loaders = None

    if standardize:
        mean, std = fit_target_stats(train_loader, device, dock_valid_max=dock_valid_max)
    else:
        mean, std = 0.0, 1.0

    use_encoded_training = bool(freeze_backbone and backbone in {"tensornet", "tsa", "tensor"})
    eval_fn = evaluate
    if use_encoded_training:
        encoded_batch_size = max(256, int(batch_size) * 8)
        if ensemble_train_loaders is not None:
            ensemble_train_loaders = [
                TorchDataLoader(
                    _encode_dataset_features(model, head_loader, device),
                    batch_size=encoded_batch_size,
                    shuffle=True,
                )
                for head_loader in ensemble_train_loaders
            ]
        train_loader = TorchDataLoader(
            _encode_dataset_features(model, train_loader, device),
            batch_size=encoded_batch_size,
            shuffle=True,
        )
        val_loader = TorchDataLoader(
            _encode_dataset_features(model, val_loader, device),
            batch_size=encoded_batch_size,
            shuffle=False,
        )
        test_loader = TorchDataLoader(
            _encode_dataset_features(model, test_loader, device),
            batch_size=encoded_batch_size,
            shuffle=False,
        )
        eval_fn = _evaluate_encoded

    def _mean_or_nan(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def _finite_or_inf(x: float) -> float:
        x = float(x)
        return x if np.isfinite(x) else float("inf")

    def _selection_score(metrics: dict) -> float:
        rmse = _finite_or_inf(metrics.get("rmse", float("inf")))
        nll = _finite_or_inf(metrics.get("nll", float("inf")))

        if gaussian_select_by == "nll":
            return nll
        if gaussian_select_by == "hybrid":
            return rmse + gaussian_select_nll_weight * nll
        return rmse
    history_rows: list[dict] = []
    train_started = time.perf_counter()
    best_val = None
    best_epoch = None
    epoch_iter = tqdm(range(1, epochs + 1), desc="surrogate[epochs]", unit="epoch", leave=False)
    for epoch in epoch_iter:
        head_lr_epoch = float(lr)
        encoder_lr_epoch = float(encoder_lr_value)
        if scheduler_name == "cosine":
            head_lr_epoch = _compute_lr(epoch, lr, epochs, warmup_epochs, min_lr)
            encoder_lr_epoch = _compute_lr(epoch, encoder_lr_value, epochs, warmup_epochs, min_lr)
            for group, label in zip(optimizer.param_groups, optimizer_group_labels):
                group["lr"] = head_lr_epoch if label == "main" else encoder_lr_epoch
        model.train()
        losses = []
        train_mse_vals = []
        train_nll_vals = []
        train_reg_vals = []
        train_mean_std_vals = []
        train_mean_ale_std_vals = []
        train_mean_epi_std_vals = []
        if use_encoded_training:
            _set_frozen_encoder_eval(model)
            if ensemble_train_loaders is not None:
                if not hasattr(model, "readouts"):
                    raise RuntimeError("Bootstrap head training requires TensorNetEnsembleSurrogate.readouts.")
                for head_batches in _iter_head_batches(ensemble_train_loaders):
                    optimizer.zero_grad()
                    head_losses = []
                    head_mses = []
                    head_nlls = []
                    head_regs = []
                    head_stds = []
                    for head_idx, (feat, y_raw) in enumerate(head_batches):
                        feat = feat.to(device)
                        y_raw = y_raw.to(device)
                        mask = torch.isfinite(y_raw)
                        if dock_valid_max is not None:
                            mask = mask & (y_raw <= dock_valid_max)
                        if mask.sum() == 0:
                            raise RuntimeError("All targets in encoded bootstrap head batch are invalid after dock filtering.")
                        feat = feat[mask]
                        y = (y_raw[mask] - mean) / std
                        raw = model.readouts[head_idx](feat)
                        mu, raw_log_var = split_gaussian_output(raw)
                        mse = torch.nn.functional.mse_loss(mu, y)
                        nll = gaussian_nll_loss(
                            mu,
                            raw_log_var,
                            y,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        reg = gaussian_var_reg_beta * torch.mean(raw_log_var ** 2)
                        loss_h = mse if epoch <= gaussian_warmup_epochs else nll + reg
                        var = gaussian_variance_from_raw(
                            raw_log_var,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        head_losses.append(loss_h)
                        head_mses.append(float(mse.item()))
                        head_nlls.append(float(nll.item()))
                        head_regs.append(float(reg.item()))
                        head_stds.append(float(torch.sqrt(var).mean().item()))
                    loss = torch.stack(head_losses).mean()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
                    train_mse_vals.append(float(np.mean(head_mses)))
                    train_nll_vals.append(float(np.mean(head_nlls)))
                    train_reg_vals.append(float(np.mean(head_regs)))
                    train_mean_std_vals.append(float(np.mean(head_stds)))
                    train_mean_ale_std_vals.append(float(np.mean(head_stds)))
                    train_mean_epi_std_vals.append(float("nan"))
            else:
                for feat, y_raw in train_loader:
                    feat = feat.to(device)
                    y_raw = y_raw.to(device)
                    mask = torch.isfinite(y_raw)
                    if dock_valid_max is not None:
                        mask = mask & (y_raw <= dock_valid_max)
                    if mask.sum() == 0:
                        raise RuntimeError("All targets in encoded batch are invalid after dock filtering.")
                    y = (y_raw[mask] - mean) / std
                    if hasattr(model, "forward_heads_from_encoded"):
                        raw_heads = model.forward_heads_from_encoded(feat)
                        mu = raw_heads[..., 0][mask]
                        raw_log_var = raw_heads[..., 1][mask]
                        y_expanded = y.unsqueeze(1).expand_as(mu)

                        mse = torch.nn.functional.mse_loss(mu, y_expanded)
                        nll = gaussian_nll_loss(
                            mu,
                            raw_log_var,
                            y_expanded,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        reg = gaussian_var_reg_beta * torch.mean(raw_log_var ** 2)

                        var = gaussian_variance_from_raw(
                            raw_log_var,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        mu_mean = mu.mean(dim=1)
                        ale_var = var.mean(dim=1)
                        epi_var = ((mu - mu_mean.unsqueeze(1)) ** 2).mean(dim=1)
                        total_var = ale_var + epi_var

                        if epoch <= gaussian_warmup_epochs:
                            loss = mse
                        else:
                            loss = nll + reg

                        train_mse_vals.append(float(mse.item()))
                        train_nll_vals.append(float(nll.item()))
                        train_reg_vals.append(float(reg.item()))
                        train_mean_std_vals.append(float(torch.sqrt(total_var).mean().item()))
                        train_mean_ale_std_vals.append(float(torch.sqrt(ale_var).mean().item()))
                        train_mean_epi_std_vals.append(float(torch.sqrt(epi_var).mean().item()))
                    elif getattr(model, "uncertainty_mode", "gaussian") == "bayes":
                        pred = model.forward_encoded(feat)[mask]
                        mse = torch.nn.functional.mse_loss(pred, y)
                        reg = bayes_beta * model.kl_loss() / max(len(train_loader.dataset), 1)
                        loss = mse + reg
                        train_mse_vals.append(float(mse.item()))
                        train_nll_vals.append(float("nan"))
                        train_reg_vals.append(float(reg.item()))
                        train_mean_std_vals.append(float("nan"))
                        train_mean_ale_std_vals.append(float("nan"))
                        train_mean_epi_std_vals.append(float("nan"))
                    elif getattr(model, "uncertainty_mode", "gaussian") == "nig":
                        raw = model.forward_encoded(feat)
                        gamma, nu, alpha, beta = split_nig_output(raw)
                        gamma = gamma[mask]
                        nu = nu[mask]
                        alpha = alpha[mask]
                        beta = beta[mask]
                        ale_var, epi_var, total_var = nig_variances_from_params(nu, alpha, beta, min_var=gaussian_min_var)
                        mse = torch.nn.functional.mse_loss(gamma, y)
                        nll = nig_nll_loss(gamma, nu, alpha, beta, y)
                        reg = nig_reg_lambda * nig_evidence_regularizer(gamma, nu, alpha, y)
                        if epoch <= nig_warmup_epochs:
                            loss = mse
                        else:
                            loss = nll + reg
                        train_mse_vals.append(float(mse.item()))
                        train_nll_vals.append(float(nll.item()))
                        train_reg_vals.append(float(reg.item()))
                        train_mean_std_vals.append(float(torch.sqrt(total_var).mean().item()))
                        train_mean_ale_std_vals.append(float(torch.sqrt(ale_var).mean().item()))
                        train_mean_epi_std_vals.append(float(torch.sqrt(epi_var).mean().item()))
                    else:
                        raw = model.forward_encoded(feat)
                        mu, raw_log_var = split_gaussian_output(raw)
                        mu = mu[mask]
                        raw_log_var = raw_log_var[mask]

                        mse = torch.nn.functional.mse_loss(mu, y)
                        nll = gaussian_nll_loss(
                            mu,
                            raw_log_var,
                            y,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        reg = gaussian_var_reg_beta * torch.mean(raw_log_var ** 2)

                        var = gaussian_variance_from_raw(
                            raw_log_var,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )

                        if epoch <= gaussian_warmup_epochs:
                            loss = mse
                        else:
                            loss = nll + reg

                        train_mse_vals.append(float(mse.item()))
                        train_nll_vals.append(float(nll.item()))
                        train_reg_vals.append(float(reg.item()))
                        train_mean_std_vals.append(float(torch.sqrt(var).mean().item()))
                        train_mean_ale_std_vals.append(float(torch.sqrt(var).mean().item()))
                        train_mean_epi_std_vals.append(float("nan"))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
        else:
            if ensemble_train_loaders is not None:
                if not hasattr(model, "readouts"):
                    raise RuntimeError("Bootstrap head training requires TensorNetEnsembleSurrogate.readouts.")
                for head_batches in _iter_head_batches(ensemble_train_loaders):
                    optimizer.zero_grad()
                    head_losses = []
                    head_mses = []
                    head_nlls = []
                    head_regs = []
                    head_stds = []
                    for head_idx, batch in enumerate(head_batches):
                        batch = batch.to(device)
                        y = get_targets(batch)
                        if y is None:
                            raise RuntimeError("Missing target tensor in surrogate bootstrap head batch.")
                        y = y.to(device)
                        mask = torch.isfinite(y)
                        if dock_valid_max is not None:
                            mask = mask & (y <= dock_valid_max)
                        if mask.sum() == 0:
                            raise RuntimeError("All targets in bootstrap head batch are invalid after dock filtering.")
                        feat = model.encode(batch)[mask]
                        y = (y[mask] - mean) / std
                        raw = model.readouts[head_idx](feat)
                        mu, raw_log_var = split_gaussian_output(raw)
                        mse = torch.nn.functional.mse_loss(mu, y)
                        nll = gaussian_nll_loss(
                            mu,
                            raw_log_var,
                            y,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        reg = gaussian_var_reg_beta * torch.mean(raw_log_var ** 2)
                        loss_h = mse if epoch <= gaussian_warmup_epochs else nll + reg
                        var = gaussian_variance_from_raw(
                            raw_log_var,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        head_losses.append(loss_h)
                        head_mses.append(float(mse.item()))
                        head_nlls.append(float(nll.item()))
                        head_regs.append(float(reg.item()))
                        head_stds.append(float(torch.sqrt(var).mean().item()))
                    loss = torch.stack(head_losses).mean()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
                    train_mse_vals.append(float(np.mean(head_mses)))
                    train_nll_vals.append(float(np.mean(head_nlls)))
                    train_reg_vals.append(float(np.mean(head_regs)))
                    train_mean_std_vals.append(float(np.mean(head_stds)))
                    train_mean_ale_std_vals.append(float(np.mean(head_stds)))
                    train_mean_epi_std_vals.append(float("nan"))
            else:
                for batch in train_loader:
                    batch = batch.to(device)
                    y = get_targets(batch)
                    if y is None:
                        raise RuntimeError("Missing target tensor in surrogate training batch.")
                    y = y.to(device)
                    mask = torch.isfinite(y)
                    if dock_valid_max is not None:
                        mask = mask & (y <= dock_valid_max)
                    if mask.sum() == 0:
                        raise RuntimeError("All targets in batch are invalid after dock filtering.")
                    y = (y[mask] - mean) / std
                    if hasattr(model, "forward_heads"):
                        raw_heads = model.forward_heads(batch)
                        mu = raw_heads[..., 0][mask]
                        raw_log_var = raw_heads[..., 1][mask]
                        y_expanded = y.unsqueeze(1).expand_as(mu)

                        mse = torch.nn.functional.mse_loss(mu, y_expanded)
                        nll = gaussian_nll_loss(
                            mu,
                            raw_log_var,
                            y_expanded,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        reg = gaussian_var_reg_beta * torch.mean(raw_log_var ** 2)

                        var = gaussian_variance_from_raw(
                            raw_log_var,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        mu_mean = mu.mean(dim=1)
                        ale_var = var.mean(dim=1)
                        epi_var = ((mu - mu_mean.unsqueeze(1)) ** 2).mean(dim=1)
                        total_var = ale_var + epi_var

                        if epoch <= gaussian_warmup_epochs:
                            loss = mse
                        else:
                            loss = nll + reg

                        train_mse_vals.append(float(mse.item()))
                        train_nll_vals.append(float(nll.item()))
                        train_reg_vals.append(float(reg.item()))
                        train_mean_std_vals.append(float(torch.sqrt(total_var).mean().item()))
                        train_mean_ale_std_vals.append(float(torch.sqrt(ale_var).mean().item()))
                        train_mean_epi_std_vals.append(float(torch.sqrt(epi_var).mean().item()))
                    elif getattr(model, "uncertainty_mode", "gaussian") == "bayes":
                        pred = model(batch)[mask]
                        mse = torch.nn.functional.mse_loss(pred, y)
                        reg = bayes_beta * model.kl_loss() / max(len(train_loader.dataset), 1)
                        loss = mse + reg
                        train_mse_vals.append(float(mse.item()))
                        train_nll_vals.append(float("nan"))
                        train_reg_vals.append(float(reg.item()))
                        train_mean_std_vals.append(float("nan"))
                        train_mean_ale_std_vals.append(float("nan"))
                        train_mean_epi_std_vals.append(float("nan"))
                    elif getattr(model, "uncertainty_mode", "gaussian") == "nig":
                        raw = model(batch)
                        gamma, nu, alpha, beta = split_nig_output(raw)
                        gamma = gamma[mask]
                        nu = nu[mask]
                        alpha = alpha[mask]
                        beta = beta[mask]
                        ale_var, epi_var, total_var = nig_variances_from_params(nu, alpha, beta, min_var=gaussian_min_var)
                        mse = torch.nn.functional.mse_loss(gamma, y)
                        nll = nig_nll_loss(gamma, nu, alpha, beta, y)
                        reg = nig_reg_lambda * nig_evidence_regularizer(gamma, nu, alpha, y)
                        if epoch <= nig_warmup_epochs:
                            loss = mse
                        else:
                            loss = nll + reg
                        train_mse_vals.append(float(mse.item()))
                        train_nll_vals.append(float(nll.item()))
                        train_reg_vals.append(float(reg.item()))
                        train_mean_std_vals.append(float(torch.sqrt(total_var).mean().item()))
                        train_mean_ale_std_vals.append(float(torch.sqrt(ale_var).mean().item()))
                        train_mean_epi_std_vals.append(float(torch.sqrt(epi_var).mean().item()))
                    else:
                        raw = model(batch)
                        mu, raw_log_var = split_gaussian_output(raw)
                        mu = mu[mask]
                        raw_log_var = raw_log_var[mask]

                        mse = torch.nn.functional.mse_loss(mu, y)
                        nll = gaussian_nll_loss(
                            mu,
                            raw_log_var,
                            y,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )
                        reg = gaussian_var_reg_beta * torch.mean(raw_log_var ** 2)

                        var = gaussian_variance_from_raw(
                            raw_log_var,
                            logvar_min=gaussian_logvar_min,
                            logvar_max=gaussian_logvar_max,
                            min_var=gaussian_min_var,
                        )

                        if epoch <= gaussian_warmup_epochs:
                            loss = mse
                        else:
                            loss = nll + reg

                        train_mse_vals.append(float(mse.item()))
                        train_nll_vals.append(float(nll.item()))
                        train_reg_vals.append(float(reg.item()))
                        train_mean_std_vals.append(float(torch.sqrt(var).mean().item()))
                        train_mean_ale_std_vals.append(float(torch.sqrt(var).mean().item()))
                        train_mean_epi_std_vals.append(float("nan"))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

        val_metrics = eval_fn(
            model,
            val_loader,
            device,
            mean,
            std,
            dock_valid_max=dock_valid_max,
            eval_samples=eval_samples,
            logvar_min=gaussian_logvar_min,
            logvar_max=gaussian_logvar_max,
            min_var=gaussian_min_var,
        )

        current_lr = float(head_lr_epoch if optimizer_group_labels else optimizer.param_groups[0]["lr"])
        current_encoder_lr = float(encoder_lr_epoch if "encoder" in optimizer_group_labels else current_lr)
        current_selection_score = _selection_score(val_metrics)

        is_best = False
        if best_val is None or current_selection_score < best_val:
            best_val = current_selection_score
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            is_best = True
            state = {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "mean": float(mean),
                "std": float(std),
                "best_epoch": int(epoch),
                "best_selection_score": float(current_selection_score),
                "best_val_metrics": dict(val_metrics),
                "config": {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "use_edge_attr": use_edge_attr,
                    "use_ligand_mask": use_ligand_mask,
                    "fp_dim": fp_dim,
                    "fp_radius": fp_radius,
                    "atom_extra_dim": atom_extra_dim,
                    "bond_extra_dim": bond_extra_dim,
                    "backbone": backbone,
                    "torchmd": torchmd_cfg_used,
                    "eval_samples": eval_samples,
                    "scheduler": scheduler_name,
                    "warmup_epochs": warmup_epochs,
                    "min_lr": min_lr,
                    "early_stop_patience": early_stop_patience,
                    "early_stop_min_delta": early_stop_min_delta,
                    "ensemble_heads": ensemble_heads,
                    "freeze_backbone": freeze_backbone,
                    "pretrained_encoder_ckpt": pretrained_encoder_ckpt,
                    "uncertainty_mode": uncertainty_mode,
                    "lr": float(lr),
                    "encoder_lr": float(encoder_lr_value),
                    "gaussian_select_by": gaussian_select_by,
                    "gaussian_select_nll_weight": gaussian_select_nll_weight,
                },
            }

        if train_log_csv:
            history_rows.append(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": _mean_or_nan(losses),
                    "train_mse": _mean_or_nan(train_mse_vals),
                    "train_nll": _mean_or_nan(train_nll_vals),
                    #"train_pred_var": train_var,

                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_r2": val_metrics["r2"],
                    "val_nll": val_metrics["nll"],

                    "val_mean_std": val_metrics["mean_std"],
                    "val_mean_ale_std": val_metrics["mean_ale_std"],
                    "val_mean_epi_std": val_metrics["mean_epi_std"],

                    "val_corr_abs_error_std": val_metrics["corr_abs_error_std"],
                    "val_corr_abs_error_ale_std": val_metrics["corr_abs_error_ale_std"],
                    "val_corr_abs_error_epi_std": val_metrics["corr_abs_error_epi_std"],

                    "val_corr_dock_ale_std": val_metrics["corr_dock_ale_std"],
                    "val_corr_dock_epi_std": val_metrics["corr_dock_epi_std"],

                    "val_mean_epi_over_ale": val_metrics["mean_epi_over_ale"],
                    "val_median_epi_over_ale": val_metrics["median_epi_over_ale"],
                    "val_std_log_epi_over_ale": val_metrics["std_log_epi_over_ale"],
                    "val_spearman_epi_ale": val_metrics["spearman_epi_ale"],
                    "val_corr_abs_error_epi_over_ale": val_metrics["corr_abs_error_epi_over_ale"],
                    "val_corr_dock_epi_over_ale": val_metrics["corr_dock_epi_over_ale"],

                    "val_mean_nu": val_metrics["mean_nu"],
                    "val_mean_alpha": val_metrics["mean_alpha"],
                    "val_mean_beta": val_metrics["mean_beta"],

                    "val_coverage_1sigma": val_metrics["coverage_1sigma"],
                    "val_coverage_2sigma": val_metrics["coverage_2sigma"],
                }
            )

        if (
                early_stop_patience > 0
                and best_epoch is not None
                and (epoch - best_epoch) >= early_stop_patience
                and current_selection_score > (best_val - early_stop_min_delta)
        ):
            break

    model.load_state_dict(state["model_state"], strict=True)
    torch.save(state, save_path)

    test_metrics = eval_fn(
        model,
        test_loader,
        device,
        mean,
        std,
        dock_valid_max=dock_valid_max,
        eval_samples=eval_samples,
        logvar_min=gaussian_logvar_min,
        logvar_max=gaussian_logvar_max,
        min_var=gaussian_min_var,
    )
    elapsed_sec = time.perf_counter() - train_started

    if train_log_csv:
        log_path = Path(train_log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_rows).to_csv(log_path, index=False)

    if train_summary_json:
        summary = {
            "save_path": str(save_path),
            "root": str(root),
            "backbone": str(backbone),
            "dataset_sizes": {
                "train": int(len(train_ds)),
                "valid": int(len(val_ds)),
                "test": int(len(test_ds)),
            },
            "target_stats": {
                "mean": float(mean),
                "std": float(std),
            },
            "training": {
                "epochs_requested": int(epochs),
                "epochs_completed": int(len(history_rows)),
                "elapsed_sec": float(elapsed_sec),
                "scheduler": str(scheduler_name),
                "warmup_epochs": int(warmup_epochs),
                "early_stop_patience": int(early_stop_patience),
                "early_stop_min_delta": float(early_stop_min_delta),
            },
            "gaussian": {
                "warmup_epochs": int(gaussian_warmup_epochs),
                "var_reg_beta": float(gaussian_var_reg_beta),
                "logvar_min": float(gaussian_logvar_min),
                "logvar_max": float(gaussian_logvar_max),
                "min_var": float(gaussian_min_var),
                "select_by": str(gaussian_select_by),
                "select_nll_weight": float(gaussian_select_nll_weight),
            },
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_selection_score": float(best_val) if best_val is not None else None,
            "best_val_metrics": dict(best_val_metrics) if "best_val_metrics" in locals() else None,
            "test_metrics": dict(test_metrics) if isinstance(test_metrics, dict) else {"rmse": float(test_metrics)},
        }
        Path(train_summary_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    test_rmse = test_metrics["rmse"] if isinstance(test_metrics, dict) else test_metrics
    test_rmse = test_metrics["rmse"] if isinstance(test_metrics, dict) else test_metrics
    test_nll = test_metrics.get("nll", float("nan")) if isinstance(test_metrics, dict) else float("nan")

    print(
        "surrogate_retrain: "
        f"saved={save_path} "
        f"best_epoch={best_epoch} "
        f"select_by={gaussian_select_by} "
        f"best_score={best_val:.4f} "
        f"best_val_rmse={best_val_metrics.get('rmse', float('nan')):.4f} "
        f"best_val_nll={best_val_metrics.get('nll', float('nan')):.4f} "
        f"test_rmse={test_rmse:.4f} "
        f"test_nll={test_nll:.4f}"
    )
    return save_path
