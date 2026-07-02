from __future__ import annotations

import csv
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader

from mobo.logging_utils import log_model_param_report
from train_gin_surrogate import (
    TensorNetSurrogate,
    evaluate,
    gaussian_nll_loss,
    gaussian_variance_from_raw,
    get_targets,
    split_gaussian_output,
)


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _IndexedDataset(TorchDataset):
    def __init__(self, base_dataset, indices: Sequence[int]) -> None:
        if not indices:
            raise ValueError("indices must be non-empty")
        self.base_dataset = base_dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


class IndependentTensorNetEnsemble(torch.nn.Module):
    def __init__(self, members: Sequence[TensorNetSurrogate], logvar_min: float, logvar_max: float, min_var: float) -> None:
        super().__init__()
        if not members:
            raise ValueError("IndependentTensorNetEnsemble requires at least one member.")
        self.members = torch.nn.ModuleList(members)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.min_var = float(min_var)

    def decompose(self, batch):
        mus = []
        vars_ = []
        for member in self.members:
            raw = member(batch)
            mu_z, raw_log_var = split_gaussian_output(raw)
            var_z = gaussian_variance_from_raw(
                raw_log_var,
                logvar_min=self.logvar_min,
                logvar_max=self.logvar_max,
                min_var=self.min_var,
            )
            mus.append(mu_z)
            vars_.append(var_z)
        mu = torch.stack(mus, dim=1)
        ale_var = torch.stack(vars_, dim=1).mean(dim=1)
        mu_mean = mu.mean(dim=1)
        epi_var = ((mu - mu_mean.unsqueeze(1)) ** 2).mean(dim=1)
        total_var = ale_var + epi_var
        return mu_mean, ale_var, epi_var, total_var


def _bootstrap_indices(n_items: int, seed: int) -> list[int]:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randint(0, int(n_items), (int(n_items),), generator=g, dtype=torch.long).tolist()


def _count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _set_trainable_scheme(model: TensorNetSurrogate, scheme: str) -> dict[str, list[str]]:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.readout.parameters():
        param.requires_grad = True

    encoder_param_names = [name for name, _ in model.encoder.named_parameters()]
    unfrozen: list[str] = []

    if scheme == "frozen":
        return {"unfrozen": unfrozen, "frozen": list(encoder_param_names)}

    if scheme == "full":
        for name, param in model.encoder.named_parameters():
            param.requires_grad = True
            unfrozen.append(f"encoder.{name}")
        return {"unfrozen": unfrozen, "frozen": []}

    if scheme != "last_layer":
        raise ValueError(f"Unsupported scheme: {scheme}")

    modules: list[tuple[str, torch.nn.Module]] = [
        ("encoder.repr.layers[-1]", model.encoder.repr.layers[-1]),
        ("encoder.repr.out_norm", model.encoder.repr.out_norm),
        ("encoder.repr.linear", model.encoder.repr.linear),
    ]
    for prefix, module in modules:
        for name, param in module.named_parameters():
            param.requires_grad = True
            unfrozen.append(f"{prefix}.{name}" if name else prefix)

    frozen = [name for name in encoder_param_names if f"encoder.{name}" not in unfrozen]
    return {"unfrozen": unfrozen, "frozen": frozen}


def _build_model(
    *,
    node_feat_dim: int,
    torchmd_cfg: dict,
    fp_dim: int,
    init_encoder_ckpt: Path | None,
    device: torch.device,
) -> TensorNetSurrogate:
    head_hidden_dim = torchmd_cfg.get("head_hidden_dim")
    head_hidden_dim = None if head_hidden_dim in (None, "", "null") else int(head_hidden_dim)
    if head_hidden_dim is None:
        head_hidden_dim = int(torchmd_cfg["embedding_dim"])

    model = TensorNetSurrogate(
        embedding_dim=int(torchmd_cfg["embedding_dim"]),
        num_layers=int(torchmd_cfg["num_layers"]),
        num_rbf=int(torchmd_cfg["num_rbf"]),
        rbf_type=str(torchmd_cfg["rbf_type"]),
        trainable_rbf=bool(torchmd_cfg["trainable_rbf"]),
        activation=str(torchmd_cfg["activation"]),
        cutoff_lower=float(torchmd_cfg["cutoff_lower"]),
        cutoff_upper=float(torchmd_cfg["cutoff_upper"]),
        node_feat_dim=int(node_feat_dim),
        max_num_neighbors=int(torchmd_cfg["max_num_neighbors"]),
        equivariance_invariance_group=str(torchmd_cfg["equivariance_invariance_group"]),
        static_shapes=bool(torchmd_cfg["static_shapes"]),
        check_errors=bool(torchmd_cfg["check_errors"]),
        dropout=float(torchmd_cfg["dropout"]),
        reduce_op=str(torchmd_cfg["reduce_op"]),
        fp_dim=int(fp_dim),
        head_hidden_dim=int(head_hidden_dim),
        head_num_layers=int(torchmd_cfg["head_num_layers"]),
    ).to(device)

    if init_encoder_ckpt is not None:
        ckpt = torch.load(init_encoder_ckpt, map_location=device)
        encoder_state = ckpt.get("encoder_state")
        if encoder_state is None:
            raise RuntimeError(f"encoder_state missing from {init_encoder_ckpt}")
        model.encoder.load_state_dict(encoder_state, strict=True)
    return model


def _selection_score(metrics: dict[str, float], nll_weight: float) -> float:
    rmse = float(metrics.get("rmse", float("inf")))
    nll = float(metrics.get("nll", float("inf")))
    if not math.isfinite(rmse):
        rmse = float("inf")
    if not math.isfinite(nll):
        nll = float("inf")
    return rmse + float(nll_weight) * max(0.0, nll)


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


def _train_one_member(
    *,
    model: TensorNetSurrogate,
    train_loader,
    val_loader,
    device: torch.device,
    mean: float,
    std: float,
    epochs: int,
    lr_head: float,
    lr_encoder: float,
    weight_decay: float,
    warmup_epochs: int,
    min_lr: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    gaussian_warmup_epochs: int,
    gaussian_var_reg_beta: float,
    gaussian_logvar_min: float,
    gaussian_logvar_max: float,
    gaussian_min_var: float,
    dock_valid_max: float | None,
    select_nll_weight: float,
) -> tuple[dict, dict, list[dict[str, float]]]:
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    readout_params = [p for p in model.readout.parameters() if p.requires_grad]
    if not readout_params and not encoder_params:
        raise RuntimeError("No trainable parameters remain.")

    param_groups = []
    if readout_params:
        param_groups.append({"params": readout_params, "lr": float(lr_head)})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": float(lr_encoder)})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(weight_decay))

    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_metrics = evaluate(
        model,
        val_loader,
        device=device,
        mean=mean,
        std=std,
        dock_valid_max=dock_valid_max,
        logvar_min=gaussian_logvar_min,
        logvar_max=gaussian_logvar_max,
        min_var=gaussian_min_var,
    )
    best_score = _selection_score(best_metrics, select_nll_weight)
    patience_left = int(early_stop_patience)
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        lr_enc = _compute_lr(epoch, float(lr_encoder), int(epochs), int(warmup_epochs), float(min_lr))
        lr_head_now = _compute_lr(epoch, float(lr_head), int(epochs), int(warmup_epochs), float(min_lr))
        for idx, group in enumerate(optimizer.param_groups):
            group["lr"] = lr_head_now if idx == 0 else lr_enc

        epoch_loss_sum = 0.0
        epoch_loss_steps = 0
        for batch in train_loader:
            batch = batch.to(device)
            y = get_targets(batch)
            if y is None:
                continue
            y = y.to(device)
            mask = torch.isfinite(y)
            if dock_valid_max is not None:
                mask = mask & (y <= dock_valid_max)
            if mask.sum() == 0:
                continue

            raw = model(batch)
            mu_z, raw_log_var = split_gaussian_output(raw)
            y_z = (y - mean) / std
            mu_z = mu_z[mask]
            raw_log_var = raw_log_var[mask]
            y_z = y_z[mask]

            if epoch <= int(gaussian_warmup_epochs):
                loss = torch.mean((mu_z - y_z) ** 2)
            else:
                loss = gaussian_nll_loss(
                    mu_z,
                    raw_log_var,
                    y_z,
                    logvar_min=gaussian_logvar_min,
                    logvar_max=gaussian_logvar_max,
                    min_var=gaussian_min_var,
                )
                var = gaussian_variance_from_raw(
                    raw_log_var,
                    logvar_min=gaussian_logvar_min,
                    logvar_max=gaussian_logvar_max,
                    min_var=gaussian_min_var,
                )
                loss = loss + float(gaussian_var_reg_beta) * var.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach().item())
            epoch_loss_steps += 1

        train_loss = float(epoch_loss_sum / epoch_loss_steps) if epoch_loss_steps > 0 else float("nan")
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            mean=mean,
            std=std,
            dock_valid_max=dock_valid_max,
            logvar_min=gaussian_logvar_min,
            logvar_max=gaussian_logvar_max,
            min_var=gaussian_min_var,
        )
        score = _selection_score(val_metrics, select_nll_weight)
        improved = score < (best_score - float(early_stop_min_delta))
        history_row = {
            "epoch": int(epoch),
            "lr_head": float(lr_head_now),
            "lr_encoder": float(lr_enc if encoder_params else 0.0),
            "train_loss": float(train_loss),
            "selection_score": float(score),
            "is_best": float(1.0 if improved else 0.0),
        }
        for key, value in val_metrics.items():
            history_row[f"val_{key}"] = float(value)
        history_rows.append(history_row)
        if improved:
            best_score = score
            best_metrics = dict(val_metrics)
            best_state = deepcopy(model.state_dict())
            best_epoch = int(epoch)
            patience_left = int(early_stop_patience)
        elif early_stop_patience > 0:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state, strict=True)
    return (
        {
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_score),
            "best_val_metrics": best_metrics,
            "epochs_completed": int(len(history_rows)),
        },
        {"state_dict": best_state},
        history_rows,
    )


def _summarize_epistemic(
    ensemble: IndependentTensorNetEnsemble,
    loader,
    *,
    device: torch.device,
    mean: float,
    std: float,
    dock_valid_max: float | None,
) -> dict[str, float]:
    ensemble.eval()
    epi_stds = []
    ale_stds = []
    total_stds = []
    epi_frac = []
    abs_err = []
    total_std_for_err = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y = get_targets(batch)
            if y is None:
                continue
            y = y.to(device)
            mask = torch.isfinite(y)
            if dock_valid_max is not None:
                mask = mask & (y <= dock_valid_max)
            if mask.sum() == 0:
                continue
            mu_z, ale_var_z, epi_var_z, total_var_z = ensemble.decompose(batch)
            mu = mu_z[mask] * std + mean
            y_valid = y[mask]
            ale_var = ale_var_z[mask].clamp_min(1e-12) * (std ** 2)
            epi_var = epi_var_z[mask].clamp_min(0.0) * (std ** 2)
            total_var = total_var_z[mask].clamp_min(1e-12) * (std ** 2)
            ale_std = torch.sqrt(ale_var)
            epi_std = torch.sqrt(epi_var)
            total_std = torch.sqrt(total_var)

            frac = torch.zeros_like(total_var)
            positive = total_var > 0
            frac[positive] = epi_var[positive] / total_var[positive]

            epi_stds.append(epi_std.detach().cpu())
            ale_stds.append(ale_std.detach().cpu())
            total_stds.append(total_std.detach().cpu())
            epi_frac.append(frac.detach().cpu())
            abs_err.append(torch.abs(mu - y_valid).detach().cpu())
            total_std_for_err.append(total_std.detach().cpu())

    if not epi_stds:
        return {
            "mean_epi_std": float("nan"),
            "mean_ale_std": float("nan"),
            "mean_total_std": float("nan"),
            "mean_epi_var_frac": float("nan"),
            "corr_abs_error_total_std": float("nan"),
        }

    epi = torch.cat(epi_stds)
    ale = torch.cat(ale_stds)
    total = torch.cat(total_stds)
    frac = torch.cat(epi_frac)
    err = torch.cat(abs_err).to(torch.float64)
    total_std = torch.cat(total_std_for_err).to(torch.float64)
    x = err - err.mean()
    y = total_std - total_std.mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).item()
    corr = float(torch.sum(x * y).item() / denom) if math.isfinite(denom) and denom > 0 else float("nan")
    return {
        "mean_epi_std": float(epi.mean().item()),
        "mean_ale_std": float(ale.mean().item()),
        "mean_total_std": float(total.mean().item()),
        "mean_epi_var_frac": float(frac.mean().item()),
        "corr_abs_error_total_std": corr,
    }


def _train_variant(
    *,
    name: str,
    train_dataset,
    val_loader,
    test_loader,
    node_feat_dim: int,
    torchmd_cfg: dict,
    fp_dim: int,
    init_encoder_ckpt: Path | None,
    out_dir: Path,
    device: torch.device,
    ensemble_size: int,
    epochs: int,
    lr_head: float,
    lr_encoder: float,
    weight_decay: float,
    warmup_epochs: int,
    min_lr: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    gaussian_warmup_epochs: int,
    gaussian_var_reg_beta: float,
    gaussian_logvar_min: float,
    gaussian_logvar_max: float,
    gaussian_min_var: float,
    mean: float,
    std: float,
    dock_valid_max: float | None,
    select_nll_weight: float,
    batch_size: int,
    bootstrap: bool,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    members: list[TensorNetSurrogate] = []
    member_summaries = []
    for member_idx in range(int(ensemble_size)):
        member_seed = int(seed) + 1000 * int(member_idx)
        _set_seed(member_seed)
        model = _build_model(
            node_feat_dim=node_feat_dim,
            torchmd_cfg=torchmd_cfg,
            fp_dim=fp_dim,
            init_encoder_ckpt=init_encoder_ckpt,
            device=device,
        )
        freeze_info = _set_trainable_scheme(model, name)

        if bootstrap:
            train_indices = _bootstrap_indices(len(train_dataset), member_seed)
            member_ds = _IndexedDataset(train_dataset, train_indices)
        else:
            member_ds = train_dataset
        train_loader = DataLoader(member_ds, batch_size=int(batch_size), shuffle=True)

        log_model_param_report(model)
        train_info, train_payload, train_history = _train_one_member(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            mean=mean,
            std=std,
            epochs=epochs,
            lr_head=lr_head,
            lr_encoder=lr_encoder,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            gaussian_warmup_epochs=gaussian_warmup_epochs,
            gaussian_var_reg_beta=gaussian_var_reg_beta,
            gaussian_logvar_min=gaussian_logvar_min,
            gaussian_logvar_max=gaussian_logvar_max,
            gaussian_min_var=gaussian_min_var,
            dock_valid_max=dock_valid_max,
            select_nll_weight=select_nll_weight,
        )
        model.load_state_dict(train_payload["state_dict"], strict=True)
        train_log_path = out_dir / f"member_{member_idx:02d}_train_log.csv"
        if train_history:
            with train_log_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(train_history[0].keys()))
                writer.writeheader()
                writer.writerows(train_history)
        ckpt_path = out_dir / f"member_{member_idx:02d}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "scheme": name,
                "member_idx": member_idx,
                "seed": member_seed,
                "train_info": train_info,
                "node_feat_dim": int(node_feat_dim),
                "torchmd_cfg": dict(torchmd_cfg),
            },
            ckpt_path,
        )
        members.append(model)
        member_summaries.append(
            {
                "member_idx": int(member_idx),
                "seed": int(member_seed),
                "trainable_params": _count_trainable_params(model),
                "unfrozen_encoder_modules": freeze_info["unfrozen"],
                "train_log_csv": str(train_log_path),
                **train_info,
            }
        )

    ensemble = IndependentTensorNetEnsemble(
        members,
        logvar_min=gaussian_logvar_min,
        logvar_max=gaussian_logvar_max,
        min_var=gaussian_min_var,
    ).to(device)
    val_metrics = evaluate(
        ensemble,
        val_loader,
        device=device,
        mean=mean,
        std=std,
        dock_valid_max=dock_valid_max,
        logvar_min=gaussian_logvar_min,
        logvar_max=gaussian_logvar_max,
        min_var=gaussian_min_var,
    )
    test_metrics = evaluate(
        ensemble,
        test_loader,
        device=device,
        mean=mean,
        std=std,
        dock_valid_max=dock_valid_max,
        logvar_min=gaussian_logvar_min,
        logvar_max=gaussian_logvar_max,
        min_var=gaussian_min_var,
    )
    val_epi = _summarize_epistemic(
        ensemble,
        val_loader,
        device=device,
        mean=mean,
        std=std,
        dock_valid_max=dock_valid_max,
    )
    test_epi = _summarize_epistemic(
        ensemble,
        test_loader,
        device=device,
        mean=mean,
        std=std,
        dock_valid_max=dock_valid_max,
    )
    summary = {
        "scheme": name,
        "ensemble_size": int(ensemble_size),
        "member_summaries": member_summaries,
        "training_overview": {
            "best_epoch_mean": float(np.mean([m["best_epoch"] for m in member_summaries])) if member_summaries else float("nan"),
            "best_epoch_min": int(min(m["best_epoch"] for m in member_summaries)) if member_summaries else 0,
            "best_epoch_max": int(max(m["best_epoch"] for m in member_summaries)) if member_summaries else 0,
            "epochs_completed_mean": float(np.mean([m.get("epochs_completed", 0) for m in member_summaries])) if member_summaries else float("nan"),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_epistemic_summary": val_epi,
        "test_epistemic_summary": test_epi,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
