#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.ligand_only_3d_dataset import LigandOnly3DStore
from mobo.config_utils import load_config
from mobo.constants import ATOM_EXTRA_DIM
from mobo.io_utils import _scan_ligand_vocab
from mobo.smiles_utils import compute_qed, compute_sa
from train_gin_surrogate import TensorNetEncoder

PROPERTY_NAMES = [
    "qed",
    "sa",
    "mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rot_bonds",
    "fsp3",
    "ring_count",
]


class PackedLigandDataset(Dataset):
    def __init__(self, data_list: Sequence[object]):
        self.data_list = list(data_list)
        if not self.data_list:
            raise RuntimeError("PackedLigandDataset is empty.")
        self.num_node_classes_ = int(getattr(self.data_list[0], "num_node_classes_", 0))

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx]


def _section(cfg: dict, key: str) -> dict:
    out = cfg.get(key, {}) if isinstance(cfg, dict) else {}
    return out if isinstance(out, dict) else {}


def _device_from_arg(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(name))


def _pick_smiles_column(columns: Sequence[str], preferred: str) -> str:
    lookup = {str(c).lower(): str(c) for c in columns}
    preferred_key = str(preferred).lower()
    if preferred_key in lookup:
        return lookup[preferred_key]
    for key in ["smiles_canonical", "canonical_smiles", "smiles", "smile"]:
        if key in lookup:
            return lookup[key]
    raise RuntimeError(f"Failed to resolve smiles column. preferred={preferred}")


def _sha1_path(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _canonicalize_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        raise RuntimeError(f"Invalid SMILES encountered: {smi}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _prepare_dataset_root(
    input_csv: Path,
    smiles_col: str,
    dataset_root: Path,
    valid_frac: float,
    seed: int,
    max_rows: int | None,
) -> tuple[Path, str, list[str]]:
    df = pd.read_csv(input_csv)
    if df.empty:
        raise RuntimeError("Input CSV is empty.")
    resolved_smiles_col = _pick_smiles_column(df.columns, smiles_col)
    raw_smiles = df[resolved_smiles_col].tolist()
    if max_rows is not None and int(max_rows) > 0:
        raw_smiles = raw_smiles[: int(max_rows)]
        df = df.iloc[: int(max_rows)].reset_index(drop=True)
    canonical_smiles = [_canonicalize_smiles(str(s).strip()) for s in raw_smiles]
    prepared = pd.DataFrame({"smiles_canonical": canonical_smiles})
    prepared = prepared.drop_duplicates(subset=["smiles_canonical"]).reset_index(drop=True)
    if prepared.empty:
        raise RuntimeError("No valid canonical SMILES found in input CSV.")
    prepared.insert(0, "ligand_id", [f"PRE_{i + 1:07d}" for i in range(prepared.shape[0])])

    if "split" in df.columns:
        raw_split = df["split"].astype(str).str.lower().tolist()
        split_map = {}
        for smi, split in zip(canonical_smiles, raw_split):
            if split in {"train", "valid"} and smi not in split_map:
                split_map[smi] = split
        if split_map and len(split_map) == prepared.shape[0]:
            prepared["split"] = [split_map[smi] for smi in prepared["smiles_canonical"].tolist()]
        else:
            rng = np.random.default_rng(int(seed))
            order = rng.permutation(prepared.shape[0])
            n_valid = int(round(prepared.shape[0] * float(valid_frac)))
            if prepared.shape[0] > 1:
                n_valid = max(1, min(n_valid, prepared.shape[0] - 1))
            else:
                n_valid = 0
            valid_idx = set(int(i) for i in order[:n_valid])
            prepared["split"] = ["valid" if i in valid_idx else "train" for i in range(prepared.shape[0])]
    else:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(prepared.shape[0])
        n_valid = int(round(prepared.shape[0] * float(valid_frac)))
        if prepared.shape[0] > 1:
            n_valid = max(1, min(n_valid, prepared.shape[0] - 1))
        else:
            n_valid = 0
        valid_idx = set(int(i) for i in order[:n_valid])
        prepared["split"] = ["valid" if i in valid_idx else "train" for i in range(prepared.shape[0])]

    dataset_root.mkdir(parents=True, exist_ok=True)
    smiles_path = dataset_root / "smiles.csv"
    prepared.to_csv(smiles_path, index=False)
    return dataset_root, _sha1_path(smiles_path), prepared["smiles_canonical"].astype(str).tolist()


def _compute_property_vector(smiles: str) -> torch.Tensor:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise RuntimeError(f"Invalid SMILES for property computation: {smiles}")
    values = [
        float(compute_qed(smiles)),
        float(compute_sa(smiles)),
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(Lipinski.NumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
    ]
    return torch.tensor(values, dtype=torch.float32)


def _attach_targets(data_list: Sequence[object]) -> None:
    for data in data_list:
        fp = getattr(data, "fp", None)
        if fp is None:
            raise RuntimeError("Packed pretraining data is missing fingerprint targets.")
        if fp.dim() != 2:
            raise RuntimeError(f"Expected fp tensor with 2 dims, got {tuple(fp.shape)}")
        smiles = getattr(data, "smiles", None)
        if smiles is None:
            raise RuntimeError("Packed pretraining data is missing smiles.")
        props = _compute_property_vector(str(smiles))
        data.prop_targets = props.unsqueeze(0)
        data.num_node_classes_ = int(getattr(data, "x").size(-1))


def _pack_matches(found: dict, expected: dict) -> bool:
    keys = [
        "dataset_signature",
        "split",
        "fp_bits",
        "fp_radius",
        "property_names",
        "ligand_vocab",
        "atom_feat_dim",
        "confgen",
    ]
    return all(found.get(key) == expected.get(key) for key in keys)


def _load_packed_dataset(pack_path: Path, expected_meta: dict) -> PackedLigandDataset | None:
    if not pack_path.exists():
        return None
    payload = torch.load(pack_path, map_location="cpu", weights_only=False)
    meta = payload.get("metadata")
    data_list = payload.get("data_list")
    if not isinstance(meta, dict) or not isinstance(data_list, list):
        return None
    if not _pack_matches(meta, expected_meta):
        return None
    return PackedLigandDataset(data_list)


def _finalize_split_data_list(
    data_list: Sequence[object],
    smiles_list: Sequence[str],
    ligand_id_map: dict[str, str],
    split: str,
    failed_csv: Path,
    allow_skip_failed_3d: bool,
) -> list[object]:
    by_smiles: dict[str, object] = {}
    for data in data_list:
        key = str(getattr(data, "smiles", "")).strip()
        if not key:
            raise RuntimeError(f"Encountered graph without smiles in split='{split}'.")
        if key in by_smiles:
            raise RuntimeError(f"Duplicate SMILES encountered while finalizing split='{split}': {key}")
        data.smiles = key
        data.ligand_id = str(ligand_id_map[key])
        data.num_node_classes_ = int(getattr(data, "x").size(-1))
        by_smiles[key] = data
    missing = [str(smi) for smi in smiles_list if str(smi) not in by_smiles]
    if missing:
        failed_rows = pd.DataFrame(
            {
                "ligand_id": [ligand_id_map[smi] for smi in missing],
                "smiles_canonical": missing,
                "split": [str(split)] * len(missing),
            }
        )
        failed_csv.parent.mkdir(parents=True, exist_ok=True)
        failed_rows.to_csv(failed_csv, index=False)
        if not allow_skip_failed_3d:
            raise RuntimeError(
                f"3D graph build failed for {len(missing)} molecules in split='{split}'. "
                f"Inspect {failed_csv} or rerun with --skip-failed-3d."
            )
        print(f"[skip_3d] split={split} skipped={len(missing)} wrote={failed_csv}")
    ordered = [by_smiles[str(smi)] for smi in smiles_list if str(smi) in by_smiles]
    if not ordered:
        raise RuntimeError(f"No 3D graphs available after filtering split='{split}'.")
    return ordered


def _build_or_load_split_dataset(
    dataset_root: Path,
    split: str,
    pack_path: Path,
    dataset_signature: str,
    ligand_vocab: Sequence[str],
    fp_bits: int,
    fp_radius: int,
    cache_dir: Path,
    candidate_3d_cfg: dict,
    confgen_seed: int,
    num_workers: int,
    mp_chunksize: int,
    allow_skip_failed_3d: bool,
) -> PackedLigandDataset:
    expected_meta = {
        "dataset_signature": dataset_signature,
        "split": str(split),
        "fp_bits": int(fp_bits),
        "fp_radius": int(fp_radius),
        "property_names": list(PROPERTY_NAMES),
        "ligand_vocab": list(ligand_vocab),
        "atom_feat_dim": int(len(ligand_vocab) + ATOM_EXTRA_DIM),
        "confgen": {
            "max_attempts": int(candidate_3d_cfg.get("max_attempts", 3)),
            "seed": int(confgen_seed),
            "num_confs": int(candidate_3d_cfg.get("num_confs", 4)),
            "max_opt_iters": int(candidate_3d_cfg.get("max_opt_iters", 100)),
            "optimize": bool(candidate_3d_cfg.get("optimize", True)),
            "prefer_mmff": bool(candidate_3d_cfg.get("prefer_mmff", False)),
        },
    }
    loaded = _load_packed_dataset(pack_path, expected_meta)
    if loaded is not None:
        print(f"[packed] loaded {split} dataset: {pack_path}")
        return loaded

    split_df = pd.read_csv(dataset_root / "smiles.csv")
    split_df = split_df[split_df["split"].astype(str).str.lower() == str(split).lower()].reset_index(drop=True)
    if split_df.empty:
        raise RuntimeError(f"No rows available for split='{split}'.")
    smiles_list = split_df["smiles_canonical"].astype(str).tolist()
    ligand_id_map = dict(zip(split_df["smiles_canonical"].astype(str).tolist(), split_df["ligand_id"].astype(str).tolist()))
    failed_csv = pack_path.parent / f"{split}_failed_3d.csv"

    store = LigandOnly3DStore(
        root=dataset_root,
        rows=split_df.to_dict("records"),
        ligand_vocab_override=list(ligand_vocab),
        cache_dir=cache_dir,
        fp_dim=int(fp_bits),
        fp_radius=int(fp_radius),
        confgen_max_attempts=int(candidate_3d_cfg.get("max_attempts", 3)),
        confgen_seed=int(confgen_seed),
        confgen_num_confs=int(candidate_3d_cfg.get("num_confs", 4)),
        confgen_max_opt_iters=int(candidate_3d_cfg.get("max_opt_iters", 100)),
        confgen_optimize=bool(candidate_3d_cfg.get("optimize", True)),
        confgen_prefer_mmff=bool(candidate_3d_cfg.get("prefer_mmff", False)),
        build_num_workers=int(num_workers),
        build_mp_chunksize=int(mp_chunksize),
    )
    if not store.data_list:
        raise RuntimeError(f"No samples available for split='{split}'.")
    data_list = list(store.data_list)
    data_list = _finalize_split_data_list(
        data_list=data_list,
        smiles_list=smiles_list,
        ligand_id_map=ligand_id_map,
        split=split,
        failed_csv=failed_csv,
        allow_skip_failed_3d=bool(allow_skip_failed_3d),
    )
    _attach_targets(data_list)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": expected_meta, "data_list": data_list}, pack_path)
    print(f"[packed] wrote {split} dataset: {pack_path}")
    return PackedLigandDataset(data_list)


def _property_stats(dataset: PackedLigandDataset) -> tuple[torch.Tensor, torch.Tensor]:
    props = []
    for data in dataset.data_list:
        target = getattr(data, "prop_targets", None)
        if target is None:
            raise RuntimeError("Dataset missing property targets.")
        props.append(target.view(-1))
    stack = torch.stack(props, dim=0)
    mean = stack.mean(dim=0)
    std = stack.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return mean, std


class TensorNetMultiTaskPretrainModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_rbf: int,
        rbf_type: str,
        trainable_rbf: bool,
        activation: str,
        cutoff_lower: float,
        cutoff_upper: float,
        node_feat_dim: int,
        max_num_neighbors: int,
        equivariance_invariance_group: str,
        static_shapes: bool,
        check_errors: bool,
        dropout: float,
        reduce_op: str,
        fp_bits: int,
        prop_dim: int,
        head_hidden_dim: int,
    ):
        super().__init__()
        self.encoder = TensorNetEncoder(
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
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dropout=dropout,
            reduce_op=reduce_op,
            fp_dim=0,
        )
        feat_dim = int(self.encoder.output_dim)
        self.fp_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, int(fp_bits)),
        )
        self.prop_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, int(prop_dim)),
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(data)
        return self.fp_head(feat), self.prop_head(feat)


def _evaluate(
    model: TensorNetMultiTaskPretrainModel,
    loader: DataLoader,
    device: torch.device,
    prop_mean: torch.Tensor,
    prop_std: torch.Tensor,
    fp_loss_weight: float,
    prop_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    fp_criterion = nn.BCEWithLogitsLoss(reduction="mean")
    prop_criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_fp_loss = 0.0
    total_prop_loss = 0.0
    total_fp_acc = 0.0
    total_prop_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target_fp = batch.fp.to(torch.float32)
            target_prop = batch.prop_targets.to(torch.float32)
            if target_fp.dim() != 2 or target_prop.dim() != 2:
                raise RuntimeError("Expected batched fp and prop targets with 2 dims.")
            pred_fp, pred_prop = model(batch)
            norm_prop = (target_prop - prop_mean) / prop_std
            fp_loss = fp_criterion(pred_fp, target_fp)
            prop_loss = prop_criterion(pred_prop, norm_prop)
            loss = float(fp_loss_weight) * fp_loss + float(prop_loss_weight) * prop_loss
            fp_acc = ((torch.sigmoid(pred_fp) >= 0.5) == target_fp).to(torch.float32).mean()
            pred_prop_raw = pred_prop * prop_std + prop_mean
            prop_mae = torch.abs(pred_prop_raw - target_prop).mean()
            batch_size = int(target_fp.size(0))
            total_loss += float(loss.item()) * batch_size
            total_fp_loss += float(fp_loss.item()) * batch_size
            total_prop_loss += float(prop_loss.item()) * batch_size
            total_fp_acc += float(fp_acc.item()) * batch_size
            total_prop_mae += float(prop_mae.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    return {
        "loss": total_loss / total_samples,
        "fp_loss": total_fp_loss / total_samples,
        "prop_loss": total_prop_loss / total_samples,
        "fp_acc": total_fp_acc / total_samples,
        "prop_mae": total_prop_mae / total_samples,
    }


def _build_torchmd_cfg(args: argparse.Namespace, cfg: dict) -> dict:
    surrogate_cfg = _section(cfg, "surrogate")
    torchmd = dict((_section(cfg, "model")).get("torchmd") or {})
    embedding_dim = int(args.embedding_dim or torchmd.get("embedding_dim", surrogate_cfg.get("retrain_hidden_dim", 128)))
    num_layers = int(args.num_layers or torchmd.get("num_layers", 3))
    dropout = float(args.dropout if args.dropout is not None else torchmd.get("dropout", surrogate_cfg.get("retrain_dropout", 0.1)))
    head_hidden_dim = int(args.head_hidden_dim or torchmd.get("head_hidden_dim", embedding_dim))
    return {
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "num_rbf": int(torchmd.get("num_rbf", 32)),
        "rbf_type": str(torchmd.get("rbf_type", "expnorm")),
        "trainable_rbf": bool(torchmd.get("trainable_rbf", False)),
        "activation": str(torchmd.get("activation", "silu")),
        "cutoff_lower": float(torchmd.get("cutoff_lower", 0.0)),
        "cutoff_upper": float(torchmd.get("cutoff_upper", 4.5)),
        "max_num_neighbors": int(torchmd.get("max_num_neighbors", 64)),
        "equivariance_invariance_group": str(torchmd.get("equivariance_invariance_group", torchmd.get("equivariance_group", "O(3)"))),
        "static_shapes": bool(torchmd.get("static_shapes", True)),
        "check_errors": bool(torchmd.get("check_errors", True)),
        "dropout": dropout,
        "reduce_op": str(torchmd.get("reduce_op", torchmd.get("reduce", "sum"))),
        "head_hidden_dim": head_hidden_dim,
    }


def _resolve_candidate_3d_cfg(args: argparse.Namespace, cfg: dict) -> dict:
    candidate_3d_cfg = dict(_section(_section(cfg, "candidate"), "candidate_3d"))
    if args.confgen_max_attempts is not None:
        candidate_3d_cfg["max_attempts"] = int(args.confgen_max_attempts)
    if args.confgen_num_confs is not None:
        candidate_3d_cfg["num_confs"] = int(args.confgen_num_confs)
    if args.confgen_max_opt_iters is not None:
        candidate_3d_cfg["max_opt_iters"] = int(args.confgen_max_opt_iters)
    if args.confgen_optimize is not None:
        candidate_3d_cfg["optimize"] = bool(args.confgen_optimize)
    if args.confgen_prefer_mmff is not None:
        candidate_3d_cfg["prefer_mmff"] = bool(args.confgen_prefer_mmff)
    return candidate_3d_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a TensorNet encoder with fingerprint reconstruction and common-property prediction.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--smiles-col", default="canonical_smiles")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/surrogate/config.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--ligand-vocab-file", default=None)
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-loss-weight", type=float, default=1.0)
    parser.add_argument("--prop-loss-weight", type=float, default=1.0)
    parser.add_argument("--ligand3d-cache-dir", default=None)
    parser.add_argument("--packed-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--mp-chunksize", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-failed-3d", action="store_true")
    parser.add_argument("--confgen-seed", type=int, default=0)
    parser.add_argument("--confgen-max-attempts", type=int, default=None)
    parser.add_argument("--confgen-num-confs", type=int, default=None)
    parser.add_argument("--confgen-max-opt-iters", type=int, default=None)
    parser.add_argument("--confgen-optimize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--confgen-prefer-mmff", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    surrogate_cfg = _section(cfg, "surrogate")
    candidate_3d_cfg = _resolve_candidate_3d_cfg(args, cfg)
    torchmd_cfg = _build_torchmd_cfg(args, cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root) if args.dataset_root else (output_dir / "dataset")
    packed_dir = Path(args.packed_dir) if args.packed_dir else (output_dir / "packed")
    cache_dir = Path(args.ligand3d_cache_dir) if args.ligand3d_cache_dir else (output_dir / "ligand3d_cache")
    save_path = Path(args.save_path) if args.save_path else (output_dir / "encoder_pretrain.pt")
    device = _device_from_arg(args.device)

    dataset_root, dataset_signature, all_smiles = _prepare_dataset_root(
        input_csv=Path(args.input_csv),
        smiles_col=str(args.smiles_col),
        dataset_root=dataset_root,
        valid_frac=float(args.valid_frac),
        seed=int(args.seed),
        max_rows=args.max_rows,
    )

    if args.ligand_vocab_file:
        ligand_vocab = json.loads(Path(args.ligand_vocab_file).read_text(encoding="utf-8"))
        if not isinstance(ligand_vocab, list) or not ligand_vocab:
            raise RuntimeError("--ligand-vocab-file must contain a non-empty JSON list.")
        ligand_vocab = [str(x) for x in ligand_vocab]
    else:
        ligand_vocab = _scan_ligand_vocab(list(all_smiles))
    if not ligand_vocab:
        raise RuntimeError("Resolved ligand vocabulary is empty.")
    (output_dir / "ligand_vocab.json").write_text(json.dumps(ligand_vocab, indent=2), encoding="utf-8")

    train_ds = _build_or_load_split_dataset(
        dataset_root=dataset_root,
        split="train",
        pack_path=packed_dir / "train_dataset.pt",
        dataset_signature=dataset_signature,
        ligand_vocab=ligand_vocab,
        fp_bits=int(args.fp_bits),
        fp_radius=int(args.fp_radius),
        cache_dir=cache_dir,
        candidate_3d_cfg=candidate_3d_cfg,
        confgen_seed=int(args.confgen_seed),
        num_workers=int(args.num_workers),
        mp_chunksize=int(args.mp_chunksize),
        allow_skip_failed_3d=bool(args.skip_failed_3d),
    )
    valid_ds = _build_or_load_split_dataset(
        dataset_root=dataset_root,
        split="valid",
        pack_path=packed_dir / "valid_dataset.pt",
        dataset_signature=dataset_signature,
        ligand_vocab=ligand_vocab,
        fp_bits=int(args.fp_bits),
        fp_radius=int(args.fp_radius),
        cache_dir=cache_dir,
        candidate_3d_cfg=candidate_3d_cfg,
        confgen_seed=int(args.confgen_seed),
        num_workers=int(args.num_workers),
        mp_chunksize=int(args.mp_chunksize),
        allow_skip_failed_3d=bool(args.skip_failed_3d),
    )

    print(
        "[confgen] "
        f"max_attempts={candidate_3d_cfg.get('max_attempts', 3)} "
        f"num_confs={candidate_3d_cfg.get('num_confs', 4)} "
        f"max_opt_iters={candidate_3d_cfg.get('max_opt_iters', 100)} "
        f"optimize={bool(candidate_3d_cfg.get('optimize', True))} "
        f"prefer_mmff={bool(candidate_3d_cfg.get('prefer_mmff', False))} "
        f"workers={int(args.num_workers)} chunksize={int(args.mp_chunksize)} "
        f"skip_failed_3d={bool(args.skip_failed_3d)}"
    )
    print(
        "[dataset] "
        f"n_total={len(all_smiles)} train={len(train_ds)} valid={len(valid_ds)} "
        f"dataset_root={dataset_root}"
    )
    if args.prepare_only:
        (output_dir / "dataset_prep_summary.json").write_text(
            json.dumps(
                {
                    "dataset_root": str(dataset_root.resolve()),
                    "packed_dir": str(packed_dir.resolve()),
                    "cache_dir": str(cache_dir.resolve()),
                    "n_total": int(len(all_smiles)),
                    "n_train": int(len(train_ds)),
                    "n_valid": int(len(valid_ds)),
                    "skip_failed_3d": bool(args.skip_failed_3d),
                    "confgen": {
                        "max_attempts": int(candidate_3d_cfg.get("max_attempts", 3)),
                        "num_confs": int(candidate_3d_cfg.get("num_confs", 4)),
                        "max_opt_iters": int(candidate_3d_cfg.get("max_opt_iters", 100)),
                        "optimize": bool(candidate_3d_cfg.get("optimize", True)),
                        "prefer_mmff": bool(candidate_3d_cfg.get("prefer_mmff", False)),
                        "num_workers": int(args.num_workers),
                        "mp_chunksize": int(args.mp_chunksize),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("prepared packed datasets only; skipping training")
        return

    batch_size = int(args.batch_size or surrogate_cfg.get("retrain_batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    prop_mean, prop_std = _property_stats(train_ds)
    prop_mean = prop_mean.to(device)
    prop_std = prop_std.to(device)

    node_feat_dim = int(getattr(train_ds.data_list[0], "x").size(-1))
    model = TensorNetMultiTaskPretrainModel(
        embedding_dim=int(torchmd_cfg["embedding_dim"]),
        num_layers=int(torchmd_cfg["num_layers"]),
        num_rbf=int(torchmd_cfg["num_rbf"]),
        rbf_type=str(torchmd_cfg["rbf_type"]),
        trainable_rbf=bool(torchmd_cfg["trainable_rbf"]),
        activation=str(torchmd_cfg["activation"]),
        cutoff_lower=float(torchmd_cfg["cutoff_lower"]),
        cutoff_upper=float(torchmd_cfg["cutoff_upper"]),
        node_feat_dim=node_feat_dim,
        max_num_neighbors=int(torchmd_cfg["max_num_neighbors"]),
        equivariance_invariance_group=str(torchmd_cfg["equivariance_invariance_group"]),
        static_shapes=bool(torchmd_cfg["static_shapes"]),
        check_errors=bool(torchmd_cfg["check_errors"]),
        dropout=float(torchmd_cfg["dropout"]),
        reduce_op=str(torchmd_cfg["reduce_op"]),
        fp_bits=int(args.fp_bits),
        prop_dim=len(PROPERTY_NAMES),
        head_hidden_dim=int(torchmd_cfg["head_hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    fp_criterion = nn.BCEWithLogitsLoss(reduction="mean")
    prop_criterion = nn.MSELoss(reduction="mean")

    best_state = None
    best_val = None
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_fp_loss_sum = 0.0
        train_prop_loss_sum = 0.0
        train_fp_acc_sum = 0.0
        train_prop_mae_sum = 0.0
        train_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            target_fp = batch.fp.to(torch.float32)
            target_prop = batch.prop_targets.to(torch.float32)
            pred_fp, pred_prop = model(batch)
            norm_prop = (target_prop - prop_mean) / prop_std
            fp_loss = fp_criterion(pred_fp, target_fp)
            prop_loss = prop_criterion(pred_prop, norm_prop)
            loss = float(args.fp_loss_weight) * fp_loss + float(args.prop_loss_weight) * prop_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            fp_acc = ((torch.sigmoid(pred_fp) >= 0.5) == target_fp).to(torch.float32).mean()
            pred_prop_raw = pred_prop * prop_std + prop_mean
            prop_mae = torch.abs(pred_prop_raw - target_prop).mean()
            batch_size_actual = int(target_fp.size(0))
            train_loss_sum += float(loss.item()) * batch_size_actual
            train_fp_loss_sum += float(fp_loss.item()) * batch_size_actual
            train_prop_loss_sum += float(prop_loss.item()) * batch_size_actual
            train_fp_acc_sum += float(fp_acc.item()) * batch_size_actual
            train_prop_mae_sum += float(prop_mae.item()) * batch_size_actual
            train_samples += batch_size_actual
        if train_samples == 0:
            raise RuntimeError("Training loader produced no samples.")

        train_metrics = {
            "loss": train_loss_sum / train_samples,
            "fp_loss": train_fp_loss_sum / train_samples,
            "prop_loss": train_prop_loss_sum / train_samples,
            "fp_acc": train_fp_acc_sum / train_samples,
            "prop_mae": train_prop_mae_sum / train_samples,
        }
        valid_metrics = _evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            prop_mean=prop_mean,
            prop_std=prop_std,
            fp_loss_weight=float(args.fp_loss_weight),
            prop_loss_weight=float(args.prop_loss_weight),
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_fp_loss": float(train_metrics["fp_loss"]),
            "train_prop_loss": float(train_metrics["prop_loss"]),
            "train_fp_acc": float(train_metrics["fp_acc"]),
            "train_prop_mae": float(train_metrics["prop_mae"]),
            "val_loss": float(valid_metrics["loss"]),
            "val_fp_loss": float(valid_metrics["fp_loss"]),
            "val_prop_loss": float(valid_metrics["prop_loss"]),
            "val_fp_acc": float(valid_metrics["fp_acc"]),
            "val_prop_mae": float(valid_metrics["prop_mae"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_fp_acc={row['val_fp_acc']:.4f} val_prop_mae={row['val_prop_mae']:.4f}"
        )
        if best_val is None or row["val_loss"] < best_val:
            best_val = float(row["val_loss"])
            best_state = {
                "encoder_state": model.encoder.state_dict(),
                "model_state": model.state_dict(),
                "best_val_loss": float(row["val_loss"]),
                "best_val_fp_acc": float(row["val_fp_acc"]),
                "best_val_prop_mae": float(row["val_prop_mae"]),
                "config": {
                    "input_csv": str(Path(args.input_csv).resolve()),
                    "dataset_root": str(dataset_root.resolve()),
                    "packed_dir": str(packed_dir.resolve()),
                    "cache_dir": str(cache_dir.resolve()),
                    "property_names": list(PROPERTY_NAMES),
                    "ligand_vocab": list(ligand_vocab),
                    "node_feat_dim": int(node_feat_dim),
                    "fp_bits": int(args.fp_bits),
                    "fp_radius": int(args.fp_radius),
                    "batch_size": int(batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "epochs": int(args.epochs),
                    "torchmd": dict(torchmd_cfg),
                    "dataset_signature": dataset_signature,
                    "train_size": int(len(train_ds)),
                    "valid_size": int(len(valid_ds)),
                    "prop_mean": prop_mean.detach().cpu().tolist(),
                    "prop_std": prop_std.detach().cpu().tolist(),
                    "confgen": {
                        "max_attempts": int(candidate_3d_cfg.get("max_attempts", 3)),
                        "seed": int(args.confgen_seed),
                        "num_confs": int(candidate_3d_cfg.get("num_confs", 4)),
                        "max_opt_iters": int(candidate_3d_cfg.get("max_opt_iters", 100)),
                        "optimize": bool(candidate_3d_cfg.get("optimize", True)),
                        "prefer_mmff": bool(candidate_3d_cfg.get("prefer_mmff", False)),
                        "num_workers": int(args.num_workers),
                        "mp_chunksize": int(args.mp_chunksize),
                    },
                },
            }
    if best_state is None:
        raise RuntimeError("Failed to produce a pretrained encoder checkpoint.")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "save_path": str(save_path.resolve()),
                "train_seconds": float(time.perf_counter() - t0),
                "best_val_loss": float(best_state["best_val_loss"]),
                "best_val_fp_acc": float(best_state["best_val_fp_acc"]),
                "best_val_prop_mae": float(best_state["best_val_prop_mae"]),
                "property_names": list(PROPERTY_NAMES),
                "node_feat_dim": int(node_feat_dim),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved pretrained encoder: {save_path}")


if __name__ == "__main__":
    main()


