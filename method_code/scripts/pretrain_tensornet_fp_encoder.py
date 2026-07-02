#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch_geometric.loader import DataLoader
from data.ligand_only_3d_dataset import LigandOnly3DStore
from mobo.config_utils import load_config
from mobo.io_utils import _load_ligand_vocab_override, _pick_smiles_column, _read_smiles_csv, _scan_ligand_vocab
from mobo.pretrain_targets import PRETRAIN_PROPERTY_NAMES
from train_gin_surrogate import TensorNetFingerprintPretrainModel

PROPERTY_NAMES = list(PRETRAIN_PROPERTY_NAMES)


INIT_ENCODER_PRETRAIN_DEFAULTS = {
    "epochs": 100,
    "batch_size": 128,
    "lr": 5.0e-4,
    "weight_decay": 1.0e-6,
    "fp_bits": 2048,
    "fp_radius": 2,
    "fp_weight": 0.5,
    "prop_weight": 1.0,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "min_lr": 5.0e-6,
    "early_stop_patience": 15,
    "early_stop_min_delta": 0.0,
    "device": "auto",
}

ROUND_ENCODER_REFRESH_DEFAULTS = {
    "epochs": 12,
    "batch_size": 24,
    "lr": 1.0e-4,
    "weight_decay": 1.0e-5,
    "fp_bits": 2048,
    "fp_radius": 2,
    "fp_weight": 0.25,
    "prop_weight": 2.0,
    "scheduler": "cosine",
    "warmup_epochs": 2,
    "min_lr": 1.0e-5,
    "early_stop_patience": 6,
    "early_stop_min_delta": 5.0e-4,
    "device": "auto",
}


def resolve_init_encoder_pretrain_config(cfg: dict) -> dict:
    section = dict(((cfg.get("surrogate") or {}).get("init_encoder_pretrain") or {}))
    merged = dict(INIT_ENCODER_PRETRAIN_DEFAULTS)
    merged.update(section)
    return merged


def resolve_round_encoder_refresh_config(cfg: dict) -> dict:
    section = dict(((cfg.get("surrogate") or {}).get("round_encoder_refresh") or {}))
    merged = dict(ROUND_ENCODER_REFRESH_DEFAULTS)
    merged.update(section)
    return merged


def _build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    warmup_epochs = max(int(warmup_epochs), 0)
    total_epochs = max(int(total_epochs), 1)
    min_scale = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step_idx: int) -> float:
        epoch_idx = int(step_idx) + 1
        if warmup_epochs > 0 and epoch_idx <= warmup_epochs:
            return max(epoch_idx / warmup_epochs, min_scale)
        if total_epochs <= warmup_epochs:
            return min_scale
        progress = (epoch_idx - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_torchmd_cfg(cfg: dict, hidden_dim: int, dropout: float) -> dict:
    torchmd = dict((cfg.get("model") or {}).get("torchmd") or {})
    return {
        "embedding_dim": int(torchmd.get("embedding_dim", hidden_dim)),
        "num_layers": int(torchmd.get("num_layers", 2)),
        "num_rbf": int(torchmd.get("num_rbf", 32)),
        "rbf_type": str(torchmd.get("rbf_type", "expnorm")),
        "trainable_rbf": bool(torchmd.get("trainable_rbf", False)),
        "activation": str(torchmd.get("activation", "silu")),
        "cutoff_lower": float(torchmd.get("cutoff_lower", 0.0)),
        "cutoff_upper": float(torchmd.get("cutoff_upper", 4.5)),
        "max_num_neighbors": int(torchmd.get("max_num_neighbors", 64)),
        "equivariance_invariance_group": str(
            torchmd.get("equivariance_invariance_group", torchmd.get("equivariance_group", "O(3)"))
        ),
        "static_shapes": bool(torchmd.get("static_shapes", True)),
        "check_errors": bool(torchmd.get("check_errors", True)),
        "dropout": float(torchmd.get("dropout", dropout)),
        "reduce_op": str(torchmd.get("reduce_op", torchmd.get("reduce", "sum"))),
        "head_hidden_dim": torchmd.get("head_hidden_dim"),
    }


def _resolve_ligand_vocab(root: Path, ligand_vocab_override: list[str] | None) -> list[str]:
    if ligand_vocab_override:
        return [str(x) for x in ligand_vocab_override]
    vocab = _load_ligand_vocab_override(root)
    if vocab:
        return list(vocab)
    smiles_df = _read_smiles_csv(str(root / "smiles.csv"))
    smiles_col = _pick_smiles_column(smiles_df.columns)
    if smiles_col is None:
        raise RuntimeError("smiles.csv missing smiles column for TensorNet vocab inference.")
    return _scan_ligand_vocab(smiles_df[smiles_col].astype(str).tolist())


def _normalize_fp_target(batch) -> torch.Tensor:
    if not hasattr(batch, "fp"):
        raise RuntimeError("Fingerprint target 'fp' is required for pretraining.")
    target = batch.fp.to(torch.float32)
    if target.dim() == 3 and target.size(1) == 1:
        target = target.squeeze(1)
    if target.dim() != 2:
        raise RuntimeError(f"Expected batched fingerprint target with 2 dims, got {tuple(target.shape)}")
    return target


def _normalize_prop_target(batch) -> torch.Tensor:
    if not hasattr(batch, "pretrain_props"):
        raise RuntimeError("Property target 'pretrain_props' is required for encoder pretraining.")
    target = batch.pretrain_props.to(torch.float32)
    if target.dim() == 3 and target.size(1) == 1:
        target = target.squeeze(1)
    if target.dim() != 2:
        raise RuntimeError(f"Expected batched property target with 2 dims, got {tuple(target.shape)}")
    if target.size(-1) != len(PROPERTY_NAMES):
        raise RuntimeError(
            f"Expected {len(PROPERTY_NAMES)} common property targets, got last dim {target.size(-1)}"
        )
    return target


def _fit_property_stats(loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for batch in loader:
        batch = batch.to(device)
        values.append(_normalize_prop_target(batch))
    if not values:
        raise RuntimeError("Failed to collect common molecular property targets for encoder pretraining.")
    all_values = torch.cat(values, dim=0)
    mean = all_values.mean(dim=0)
    std = all_values.std(dim=0, unbiased=False)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean.detach(), std.detach()


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prop_mean: torch.Tensor,
    prop_std: torch.Tensor,
    fp_weight: float,
    prop_weight: float,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_fp_acc = 0.0
    total_prop_mse_z = 0.0
    total_samples = 0
    fp_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    prop_criterion = torch.nn.MSELoss(reduction="mean")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            fp_target = _normalize_fp_target(batch)
            prop_target = _normalize_prop_target(batch)
            prop_target_std = (prop_target - prop_mean) / prop_std
            fp_logits, prop_pred = model(batch)
            fp_loss = fp_criterion(fp_logits, fp_target)
            prop_loss = prop_criterion(prop_pred, prop_target_std)
            loss = float(fp_weight) * fp_loss + float(prop_weight) * prop_loss
            fp_pred = (torch.sigmoid(fp_logits) >= 0.5).to(fp_target.dtype)
            fp_acc = (fp_pred == fp_target).to(torch.float32).mean()
            prop_mse_z = torch.mean((prop_pred - prop_target_std) ** 2)
            batch_size = int(fp_target.size(0))
            total_loss += float(loss.item()) * batch_size
            total_fp_acc += float(fp_acc.item()) * batch_size
            total_prop_mse_z += float(prop_mse_z.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("Validation loader produced no samples.")
    return (
        total_loss / total_samples,
        total_fp_acc / total_samples,
        math.sqrt(max(total_prop_mse_z / total_samples, 0.0)),
        total_samples,
    )


def run_tensornet_encoder_pretrain(
    *,
    config_path: str,
    dataset_root: str | Path,
    save_path: str | Path,
    init_encoder_ckpt: str | Path | None = None,
    epochs: int,
    batch_size: int | None,
    lr: float,
    weight_decay: float,
    fp_bits: int,
    fp_radius: int,
    device_arg: str,
    ligand3d_cache_dir: str | None,
    ligand_vocab_override: list[str] | None = None,
    ligand3d_num_workers: int = 1,
    ligand3d_mp_chunksize: int = 16,
    confgen_seed: int = 0,
    fp_weight: float = 0.5,
    prop_weight: float = 1.0,
    scheduler_name: str = "cosine",
    warmup_epochs: int = 5,
    min_lr: float = 5.0e-6,
    early_stop_patience: int = 15,
    early_stop_min_delta: float = 0.0,
) -> Path:
    cfg = load_config(config_path)
    surrogate_cfg = dict(cfg.get("surrogate") or {})
    general_cfg = dict(cfg.get("general") or {})
    candidate_3d_cfg = dict(((cfg.get("candidate") or {}).get("candidate_3d") or {}))
    torchmd_cfg = _build_torchmd_cfg(
        cfg,
        hidden_dim=int(surrogate_cfg.get("retrain_hidden_dim", 128)),
        dropout=float(surrogate_cfg.get("retrain_dropout", 0.1)),
    )

    dataset_root = Path(dataset_root or general_cfg.get("dataset_root") or "")
    if not dataset_root:
        raise ValueError("dataset_root is required for TensorNet encoder pretraining")
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if int(fp_bits) <= 0:
        raise ValueError("fp_bits must be > 0")

    batch_size = int(batch_size or surrogate_cfg.get("retrain_batch_size", 32))
    device = _resolve_device(device_arg)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_ligand_vocab_override = cfg.get("ligand_vocab_override")
    if ligand_vocab_override is not None:
        shared_ligand_vocab = [str(x) for x in ligand_vocab_override]
    else:
        if cfg_ligand_vocab_override is not None:
            cfg_ligand_vocab_override = [str(x) for x in cfg_ligand_vocab_override]
        shared_ligand_vocab = _resolve_ligand_vocab(dataset_root, cfg_ligand_vocab_override)

    confgen_cfg = {
        "max_attempts": int(candidate_3d_cfg.get("max_attempts", 3)),
        "seed": int(confgen_seed),
        "num_confs": int(candidate_3d_cfg.get("num_confs", 4)),
        "max_opt_iters": int(candidate_3d_cfg.get("max_opt_iters", 100)),
        "optimize": bool(candidate_3d_cfg.get("optimize", True)),
        "prefer_mmff": bool(candidate_3d_cfg.get("prefer_mmff", False)),
    }

    store = LigandOnly3DStore(
        root=dataset_root,
        ligand_vocab_override=shared_ligand_vocab,
        cache_dir=ligand3d_cache_dir or surrogate_cfg.get("ligand3d_cache_dir"),
        fp_dim=int(fp_bits),
        fp_radius=int(fp_radius),
        confgen_max_attempts=int(confgen_cfg["max_attempts"]),
        confgen_seed=int(confgen_cfg["seed"]),
        confgen_num_confs=int(confgen_cfg["num_confs"]),
        confgen_max_opt_iters=int(confgen_cfg["max_opt_iters"]),
        confgen_optimize=bool(confgen_cfg["optimize"]),
        confgen_prefer_mmff=bool(confgen_cfg["prefer_mmff"]),
        build_num_workers=int(ligand3d_num_workers),
        build_mp_chunksize=int(ligand3d_mp_chunksize),
    )
    train_ds = store.get_split_dataset("train")
    val_ds = store.get_split_dataset("valid")
    if len(train_ds) == 0:
        raise RuntimeError("No training samples available for TensorNet encoder pretraining.")
    if len(val_ds) == 0:
        val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = TensorNetFingerprintPretrainModel(
        fp_bits=int(fp_bits),
        embedding_dim=int(torchmd_cfg["embedding_dim"]),
        num_layers=int(torchmd_cfg["num_layers"]),
        num_rbf=int(torchmd_cfg["num_rbf"]),
        rbf_type=str(torchmd_cfg["rbf_type"]),
        trainable_rbf=bool(torchmd_cfg["trainable_rbf"]),
        activation=str(torchmd_cfg["activation"]),
        cutoff_lower=float(torchmd_cfg["cutoff_lower"]),
        cutoff_upper=float(torchmd_cfg["cutoff_upper"]),
        node_feat_dim=int(getattr(store, "num_node_classes_", 0)),
        max_num_neighbors=int(torchmd_cfg["max_num_neighbors"]),
        equivariance_invariance_group=str(torchmd_cfg["equivariance_invariance_group"]),
        static_shapes=bool(torchmd_cfg["static_shapes"]),
        check_errors=bool(torchmd_cfg["check_errors"]),
        dropout=float(torchmd_cfg["dropout"]),
        reduce_op=str(torchmd_cfg["reduce_op"]),
        head_hidden_dim=None if torchmd_cfg["head_hidden_dim"] in (None, "", "null") else int(torchmd_cfg["head_hidden_dim"]),
        property_dim=len(PROPERTY_NAMES),
    ).to(device)
    if init_encoder_ckpt is not None:
        init_ckpt = torch.load(str(init_encoder_ckpt), map_location=device)
        encoder_state = init_ckpt.get("encoder_state")
        if encoder_state is None:
            raise RuntimeError("Initial encoder checkpoint missing encoder_state.")
        model.encoder.load_state_dict(encoder_state, strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler_name = str(scheduler_name).strip().lower()
    if scheduler_name != "cosine":
        raise RuntimeError(f"Unsupported pretrain scheduler: {scheduler_name}")
    scheduler = _build_cosine_warmup_scheduler(
        optimizer,
        base_lr=float(lr),
        min_lr=float(min_lr),
        warmup_epochs=int(warmup_epochs),
        total_epochs=int(epochs),
    )
    fp_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    prop_criterion = torch.nn.MSELoss(reduction="mean")
    prop_mean, prop_std = _fit_property_stats(train_loader, device)

    log_path = save_path.with_suffix(save_path.suffix + ".train.log")
    history_path = save_path.with_suffix(save_path.suffix + ".history.csv")
    best_val_loss = None
    best_state = None
    stale_epochs = 0
    history_rows = []
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("TensorNet encoder pretraining log\n")
        log_file.write(f"dataset_root={dataset_root.resolve()}\n")
        log_file.write(f"save_path={save_path.resolve()}\n")
        log_file.write(f"property_names={PROPERTY_NAMES}\n")
        for epoch in range(1, int(epochs) + 1):
            model.train()
            train_loss_sum = 0.0
            train_fp_acc_sum = 0.0
            train_prop_mse_z_sum = 0.0
            train_samples = 0
            for batch in train_loader:
                batch = batch.to(device)
                fp_target = _normalize_fp_target(batch)
                prop_target = _normalize_prop_target(batch)
                prop_target_std = (prop_target - prop_mean) / prop_std
                fp_logits, prop_pred = model(batch)
                fp_loss = fp_criterion(fp_logits, fp_target)
                prop_loss = prop_criterion(prop_pred, prop_target_std)
                loss = float(fp_weight) * fp_loss + float(prop_weight) * prop_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                fp_pred = (torch.sigmoid(fp_logits) >= 0.5).to(fp_target.dtype)
                fp_acc = (fp_pred == fp_target).to(torch.float32).mean()
                prop_mse_z = torch.mean((prop_pred - prop_target_std) ** 2)
                batch_size_actual = int(fp_target.size(0))
                train_loss_sum += float(loss.item()) * batch_size_actual
                train_fp_acc_sum += float(fp_acc.item()) * batch_size_actual
                train_prop_mse_z_sum += float(prop_mse_z.item()) * batch_size_actual
                train_samples += batch_size_actual

            if train_samples == 0:
                raise RuntimeError("TensorNet encoder pretraining produced no training batches.")
            train_loss = train_loss_sum / train_samples
            train_fp_acc = train_fp_acc_sum / train_samples
            train_prop_rmse_z = math.sqrt(max(train_prop_mse_z_sum / train_samples, 0.0))
            val_loss, val_fp_acc, val_prop_rmse_z, _ = _evaluate(
                model,
                val_loader,
                device,
                prop_mean,
                prop_std,
                fp_weight=float(fp_weight),
                prop_weight=float(prop_weight),
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            history_row = {
                "epoch": int(epoch),
                "lr": current_lr,
                "train_loss": float(train_loss),
                "train_fp_acc": float(train_fp_acc),
                "train_prop_rmse_z": float(train_prop_rmse_z),
                "val_loss": float(val_loss),
                "val_fp_acc": float(val_fp_acc),
                "val_prop_rmse_z": float(val_prop_rmse_z),
            }
            history_rows.append(history_row)
            log_file.write(
                f"epoch={epoch:03d} lr={current_lr:.6g} train_loss={train_loss:.4f} train_fp_acc={train_fp_acc:.4f} "
                f"train_prop_rmse_z={train_prop_rmse_z:.4f} val_loss={val_loss:.4f} "
                f"val_fp_acc={val_fp_acc:.4f} val_prop_rmse_z={val_prop_rmse_z:.4f}\n"
            )
            log_file.flush()
            improved = best_val_loss is None or val_loss < (best_val_loss - float(early_stop_min_delta))
            if improved:
                best_val_loss = float(val_loss)
                stale_epochs = 0
                best_state = {
                    "encoder_state": model.encoder.state_dict(),
                    "model_state": model.state_dict(),
                    "best_val_loss": float(val_loss),
                    "best_val_fp_acc": float(val_fp_acc),
                    "best_val_prop_rmse_z": float(val_prop_rmse_z),
                    "best_epoch": int(epoch),
                    "ligand_vocab": list(shared_ligand_vocab),
                    "config": {
                        "dataset_root": str(dataset_root.resolve()),
                        "fp_bits": int(fp_bits),
                        "fp_radius": int(fp_radius),
                        "batch_size": int(batch_size),
                        "lr": float(lr),
                        "weight_decay": float(weight_decay),
                        "epochs": int(epochs),
                        "scheduler": scheduler_name,
                        "warmup_epochs": int(warmup_epochs),
                        "min_lr": float(min_lr),
                        "early_stop_patience": int(early_stop_patience),
                        "early_stop_min_delta": float(early_stop_min_delta),
                        "node_feat_dim": int(getattr(store, "num_node_classes_", 0)),
                        "train_size": int(len(train_ds)),
                        "val_size": int(len(val_ds)),
                        "property_names": list(PROPERTY_NAMES),
                        "ligand3d_num_workers": int(ligand3d_num_workers),
                        "ligand3d_mp_chunksize": int(ligand3d_mp_chunksize),
                        "property_mean": prop_mean.detach().cpu().tolist(),
                        "property_std": prop_std.detach().cpu().tolist(),
                        "fp_weight": float(fp_weight),
                        "prop_weight": float(prop_weight),
                        "torchmd": torchmd_cfg,
                        "confgen": confgen_cfg,
                    },
                }
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= int(early_stop_patience):
                log_file.write(
                    f"early_stop epoch={epoch:03d} best_val_loss={best_val_loss:.4f} patience={int(early_stop_patience)}\n"
                )
                log_file.flush()
                break
    if best_state is None:
        raise RuntimeError("Failed to produce a pretrained TensorNet encoder checkpoint.")
    torch.save(best_state, save_path)
    metrics_path = save_path.with_suffix(save_path.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_loss": float(best_state["best_val_loss"]),
                "best_val_fp_acc": float(best_state["best_val_fp_acc"]),
                "best_val_prop_rmse_z": float(best_state["best_val_prop_rmse_z"]),
                "best_epoch": int(best_state["best_epoch"]),
                "save_path": str(save_path),
                "log_path": str(log_path),
                "history_path": str(history_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with history_path.open("w", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch", "lr", "train_loss", "train_fp_acc", "train_prop_rmse_z",
                "val_loss", "val_fp_acc", "val_prop_rmse_z",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)
    print(
        f"saved pretrained encoder: {save_path} best_epoch={best_state['best_epoch']} "
        f"best_val_loss={best_state['best_val_loss']:.4f} best_val_fp_acc={best_state['best_val_fp_acc']:.4f} "
        f"best_val_prop_rmse_z={best_state['best_val_prop_rmse_z']:.4f} log={log_path.name}"
    )
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a TensorNet encoder to reconstruct Morgan fingerprints and common molecular properties.")
    parser.add_argument("--config", type=str, default="config/surrogate/config.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--fp-bits", type=int, default=None)
    parser.add_argument("--fp-radius", type=int, default=None)
    parser.add_argument("--fp-weight", type=float, default=None)
    parser.add_argument("--prop-weight", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ligand3d-cache-dir", type=str, default=None)
    parser.add_argument("--ligand3d-workers", type=int, default=1)
    parser.add_argument("--ligand3d-chunksize", type=int, default=16)
    parser.add_argument("--confgen-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(str(args.config))
    pretrain_cfg = resolve_init_encoder_pretrain_config(cfg)
    run_tensornet_encoder_pretrain(
        config_path=str(args.config),
        dataset_root=str(args.dataset_root),
        save_path=str(args.save_path),
        epochs=int(pretrain_cfg["epochs"] if args.epochs is None else args.epochs),
        batch_size=(int(pretrain_cfg["batch_size"]) if args.batch_size is None else int(args.batch_size)),
        lr=float(pretrain_cfg["lr"] if args.lr is None else args.lr),
        weight_decay=float(pretrain_cfg["weight_decay"] if args.weight_decay is None else args.weight_decay),
        fp_bits=int(pretrain_cfg["fp_bits"] if args.fp_bits is None else args.fp_bits),
        fp_radius=int(pretrain_cfg["fp_radius"] if args.fp_radius is None else args.fp_radius),
        device_arg=str(pretrain_cfg["device"] if args.device is None else args.device),
        ligand3d_cache_dir=args.ligand3d_cache_dir,
        ligand3d_num_workers=int(args.ligand3d_workers),
        ligand3d_mp_chunksize=int(args.ligand3d_chunksize),
        confgen_seed=int(args.confgen_seed),
        fp_weight=float(pretrain_cfg["fp_weight"] if args.fp_weight is None else args.fp_weight),
        prop_weight=float(pretrain_cfg["prop_weight"] if args.prop_weight is None else args.prop_weight),
        scheduler_name=str(pretrain_cfg["scheduler"] if args.scheduler is None else args.scheduler),
        warmup_epochs=int(pretrain_cfg["warmup_epochs"] if args.warmup_epochs is None else args.warmup_epochs),
        min_lr=float(pretrain_cfg["min_lr"] if args.min_lr is None else args.min_lr),
        early_stop_patience=int(pretrain_cfg["early_stop_patience"] if args.early_stop_patience is None else args.early_stop_patience),
        early_stop_min_delta=float(pretrain_cfg["early_stop_min_delta"] if args.early_stop_min_delta is None else args.early_stop_min_delta),
    )


if __name__ == "__main__":
    main()
