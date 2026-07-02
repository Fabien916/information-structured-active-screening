#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

import numpy as np
import torch
import yaml
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import BondType as BT
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from models_gvt import GraphVQVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = int(sum(p.numel() for p in module.parameters()))
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return total, trainable


def _split_indices(
    n: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    n = int(n)
    if n < 3:
        raise ValueError(f"dataset must contain at least 3 samples, got {n}")

    train_split = float(train_split)
    val_split = float(val_split)
    test_split = float(test_split)
    total = train_split + val_split + test_split
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(
            f"train/val/test splits must sum to 1.0, got {train_split} + {val_split} + {test_split} = {total}"
        )

    sizes = [
        int(round(train_split * n)),
        int(round(val_split * n)),
        int(round(test_split * n)),
    ]
    sizes[0] += n - sum(sizes)
    if min(sizes) <= 0:
        raise ValueError(f"invalid split sizes for n={n}: {sizes}")

    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)

    train_end = sizes[0]
    val_end = sizes[0] + sizes[1]
    return idx[:train_end].tolist(), idx[train_end:val_end].tolist(), idx[val_end:].tolist()


@dataclass
class PriorConfig:
    dataset_path: str
    ae_config_path: str
    ae_ckpt_path: str
    ae_codes_out_dir: str
    auto_build_vqdataset: bool
    save_dir: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    hidden_dim: int
    emb_dim: int
    use_codebook_input: bool
    freeze_codebook_embedding: bool
    num_layers: int
    dropout: float
    max_seq_len: int
    train_split: float
    val_split: float
    test_split: float
    seed: int
    num_workers: int
    clip_grad: float
    scheduler: str
    warmup_ratio: float
    min_lr_ratio: float
    early_stop_patience: int
    early_stop_min_delta: float
    early_stop_min_epochs: int
    sample_every: int
    sample_batch: int
    sample_num_samples: int
    sample_max_new_tokens: int
    sample_temperature: float
    sample_top_k: int
    sample_eval_vun: bool
    sample_eval_fcd: bool
    sample_fcd_ref_max: int
    sample_fcd_batch_size: int
    sample_fcd_device: str
    sample_fcd_pref_path: str
    sample_decode_graph_batch: int
    device: str
    wandb: dict


def default_config() -> PriorConfig:
    return PriorConfig(
        dataset_path="checkpoints/vqdataset_moses_random_250k_smallcfg/vqdataset_valid_reduced.pt",
        ae_config_path="config_gvt_ae_moses_random_250k.yaml",
        ae_ckpt_path="checkpoints/gvt_ae_moses_random_250k_smallcfg.pt",
        ae_codes_out_dir="checkpoints/vqdataset_moses_random_250k_smallcfg",
        auto_build_vqdataset=True,
        save_dir="checkpoints/gru_prior_moses_random_250k",
        batch_size=256,
        epochs=200,
        lr=3.0e-4,
        weight_decay=1.0e-5,
        hidden_dim=256,
        emb_dim=128,
        use_codebook_input=True,
        freeze_codebook_embedding=True,
        num_layers=2,
        dropout=0.15,
        max_seq_len=0,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=42,
        num_workers=0,
        clip_grad=1.0,
        scheduler="warmup_cosine",
        warmup_ratio=0.03,
        min_lr_ratio=0.1,
        early_stop_patience=30,
        early_stop_min_delta=5.0e-4,
        early_stop_min_epochs=40,
        sample_every=10,
        sample_batch=64,
        sample_num_samples=2000,
        sample_max_new_tokens=64,
        sample_temperature=1.0,
        sample_top_k=50,
        sample_eval_vun=True,
        sample_eval_fcd=False,
        sample_fcd_ref_max=10000,
        sample_fcd_batch_size=256,
        sample_fcd_device="cpu",
        sample_fcd_pref_path="",
        sample_decode_graph_batch=64,
        device="auto",
        wandb={
            "enabled": False,
            "project": "gvt_prior",
            "entity": None,
            "name": None,
            "mode": "online",
            "tags": [],
        },
    )


def load_config(path: str) -> PriorConfig:
    cfg = asdict(default_config())
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    for k, v in raw.items():
        if k in cfg:
            cfg[k] = v
    return PriorConfig(**cfg)


def _init_wandb(cfg: PriorConfig, extra_config: dict):
    wb = cfg.wandb if isinstance(cfg.wandb, dict) else {}
    if not bool(wb.get("enabled", False)):
        return None

    run_root = _resolve_from_cwd(cfg.save_dir)
    tmp_root = (run_root / '.wandb_tmp').resolve()
    cache_root = (run_root / '.wandb_cache').resolve()
    cfg_root = (run_root / '.wandb_config').resolve()
    for p in (run_root, tmp_root, cache_root, cfg_root):
        p.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_DIR'] = str(run_root)
    os.environ['WANDB_CACHE_DIR'] = str(cache_root)
    os.environ['WANDB_CONFIG_DIR'] = str(cfg_root)
    os.environ['TMPDIR'] = str(tmp_root)
    os.environ['TMP'] = str(tmp_root)
    os.environ['TEMP'] = str(tmp_root)

    import wandb

    run_cfg = asdict(cfg)
    settings = wandb.Settings(
        start_method='thread',
        root_dir=str(run_root),
        x_service_wait=60.0,
        console='off',
    )
    return wandb.init(
        project=str(wb.get("project", "gvt_prior")),
        entity=wb.get("entity", None),
        name=wb.get("name", None),
        mode=str(wb.get("mode", "online")),
        tags=wb.get("tags", None),
        dir=str(run_root),
        settings=settings,
        config={**run_cfg, **extra_config},
    )


def _resolve_from_cwd(path_like: str) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _ensure_vq_dataset(cfg: PriorConfig) -> str:
    dataset_path = _resolve_from_cwd(cfg.dataset_path)
    if dataset_path.exists():
        return str(dataset_path)

    if not bool(cfg.auto_build_vqdataset):
        raise FileNotFoundError(f"vq dataset not found: {dataset_path}")

    ae_cfg_path = _resolve_from_cwd(cfg.ae_config_path)
    ae_ckpt_path = _resolve_from_cwd(cfg.ae_ckpt_path)
    if not ae_cfg_path.exists():
        raise FileNotFoundError(f"ae config not found: {ae_cfg_path}")
    if not ae_ckpt_path.exists():
        raise FileNotFoundError(f"ae checkpoint not found: {ae_ckpt_path}")

    raw = yaml.safe_load(ae_cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid ae config format: {ae_cfg_path}")
    raw.setdefault("train", {})
    if not isinstance(raw["train"], dict):
        raise ValueError(f"invalid ae config train section: {ae_cfg_path}")

    out_dir = _resolve_from_cwd(cfg.ae_codes_out_dir or str(dataset_path.parent))
    raw["train"]["epochs"] = 0
    raw["train"]["save_path"] = str(ae_ckpt_path)
    raw["train"]["codes_out_dir"] = str(out_dir)
    raw["train"]["wandb"] = {"enabled": False}

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="tmp_export_vq_",
        delete=False,
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=False)
        tmp_cfg_path = Path(f.name)

    try:
        cmd = [sys.executable, str((Path.cwd() / "train_gvt_ae.py").resolve()), "--config", str(tmp_cfg_path)]
        print(f"[vqdataset] missing: {dataset_path}")
        print(f"[vqdataset] generating from ae ckpt: {ae_ckpt_path}")
        subprocess.run(cmd, check=True, cwd=str(Path.cwd()))
    finally:
        tmp_cfg_path.unlink(missing_ok=True)

    if dataset_path.exists():
        return str(dataset_path)

    candidates = [
        out_dir / "vqdataset_valid_reduced.pt",
        out_dir / "vqdataset_valid.pt",
        out_dir / "vqdataset_reduced.pt",
        out_dir / "vqdataset.pt",
    ]
    for cand in candidates:
        if cand.exists():
            print(f"[vqdataset] requested path missing, fallback to: {cand}")
            return str(cand)

    raise FileNotFoundError(
        f"vq dataset generation finished but no exported file found in {out_dir}"
    )


class VQSequenceDataset(Dataset):
    """
    Build BOS/.../EOS token sequences from exported vqdataset payload:
    - embed_ind: concatenated code indices
    - slices: number of nodes per graph
    - codebook: [K, D]
    """

    def __init__(self, dataset_path: str, max_seq_len: int = 0):
        payload = torch.load(dataset_path, map_location="cpu")
        self.embed_ind: torch.Tensor = payload["embed_ind"].long().cpu()
        self.slices: torch.Tensor = payload["slices"].long().cpu()
        self.codebook: torch.Tensor = payload["codebook"].cpu()
        self.generation_config = payload.get("generation_config", {})
        self.is_reduced = bool(payload.get("is_reduced", False))

        self.codebook_size = int(self.codebook.size(0))
        self.bos_token_id = int(self.codebook_size)
        self.eos_token_id = int(self.codebook_size + 1)
        self.pad_token_id = int(self.codebook_size + 2)
        self.vocab_size = int(self.codebook_size + 3)

        self.max_seq_len = int(max_seq_len)
        self.cum_slices = torch.cat(
            [torch.tensor([0], dtype=torch.long), self.slices.cumsum(dim=0)], dim=0
        )

    def __len__(self) -> int:
        return int(self.slices.numel())

    def get_code_sequence(self, idx: int) -> torch.Tensor:
        start = int(self.cum_slices[idx].item())
        end = start + int(self.slices[idx].item())
        return self.embed_ind[start:end].long()

    def __getitem__(self, idx: int) -> torch.Tensor:
        core = self.get_code_sequence(idx)
        seq = torch.cat(
            [
                torch.tensor([self.bos_token_id], dtype=torch.long),
                core.long(),
                torch.tensor([self.eos_token_id], dtype=torch.long),
            ],
            dim=0,
        )
        if self.max_seq_len > 0 and seq.numel() > self.max_seq_len:
            seq = torch.cat(
                [seq[: self.max_seq_len - 1], torch.tensor([self.eos_token_id], dtype=torch.long)],
                dim=0,
            )
        return seq




@contextmanager
def _rdkit_log_block():
    blocker = rdBase.BlockLogs()
    try:
        yield
    finally:
        del blocker


def _canonicalize_smiles(smiles: str) -> str | None:
    if not smiles:
        return None
    with _rdkit_log_block():
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None


def _load_reference_smiles_from_ae_cfg(ae_config_path: str) -> set[str]:
    ae_cfg = yaml.safe_load(Path(ae_config_path).read_text(encoding="utf-8")) or {}
    data_cfg = ae_cfg.get("data", {})
    csv_path = _resolve_from_cwd(str(data_cfg.get("input_csv", "")))
    smiles_col = str(data_cfg.get("smiles_column", "smiles"))
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=[smiles_col])
    out: set[str] = set()
    for smi in df[smiles_col].astype(str).tolist():
        c = _canonicalize_smiles(smi.strip())
        if c is not None:
            out.add(c)
    return out



def _load_reference_smiles_list_from_ae_cfg(ae_config_path: str) -> list[str]:
    ae_cfg = yaml.safe_load(Path(ae_config_path).read_text(encoding="utf-8")) or {}
    data_cfg = ae_cfg.get("data", {})
    csv_path = _resolve_from_cwd(str(data_cfg.get("input_csv", "")))
    smiles_col = str(data_cfg.get("smiles_column", "smiles"))
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, usecols=[smiles_col])
    out: list[str] = []
    for smi in df[smiles_col].astype(str).tolist():
        c = _canonicalize_smiles(smi.strip())
        if c is not None:
            out.append(c)
    return out


def _resolve_fcd_pref_path(save_dir: Path, cfg_path: str) -> Path:
    raw = str(cfg_path).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p
    return (save_dir / "sample_fcd_pref.npz").resolve()


def _load_fcd_pref_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
        mu = arr["mu"]
        sigma = arr["sigma"]
        if mu.ndim != 1 or sigma.ndim != 2:
            return None
        return {"mu": mu, "sigma": sigma}
    except Exception:
        return None


def _save_fcd_pref_cache(path: Path, pref: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=np.asarray(pref["mu"]), sigma=np.asarray(pref["sigma"]))


def _compute_fcd_with_pref(
    smiles_list: Sequence[str | None],
    fcd_metric,
    pref: dict,
    n_ref: int,
) -> dict:
    gen: list[str] = []
    for s in smiles_list:
        c = _canonicalize_smiles(s) if s is not None else None
        if c is not None:
            gen.append(c)

    if len(gen) < 2 or int(n_ref) < 2:
        return {"fcd": float("nan"), "n_gen": int(len(gen)), "n_ref": int(n_ref), "ok": False}

    try:
        score = float(fcd_metric(pref=pref, gen=gen))
        return {"fcd": score, "n_gen": int(len(gen)), "n_ref": int(n_ref), "ok": True}
    except Exception:
        return {"fcd": float("nan"), "n_gen": int(len(gen)), "n_ref": int(n_ref), "ok": False}

def _bond_type_from_name(name: str) -> BT | None:
    key = str(name).upper().split(".")[-1]
    mapping = {
        "SINGLE": BT.SINGLE,
        "DOUBLE": BT.DOUBLE,
        "TRIPLE": BT.TRIPLE,
        "AROMATIC": BT.AROMATIC,
    }
    return mapping.get(key, None)


def _graph_to_smiles(
    node_labels: torch.Tensor,
    edge_labels: torch.Tensor,
    atom_types: Sequence[str],
    bond_types: Sequence[str],
) -> str | None:
    n = int(node_labels.numel())
    if n <= 0:
        return None
    with _rdkit_log_block():
        mol = Chem.RWMol()
        try:
            for i in range(n):
                atom_idx = int(node_labels[i].item())
                if atom_idx < 0 or atom_idx >= len(atom_types):
                    return None
                mol.AddAtom(Chem.Atom(str(atom_types[atom_idx])))

            no_bond_class = len(bond_types)
            for i in range(n):
                for j in range(i + 1, n):
                    b = int(edge_labels[i, j].item())
                    if b < 0 or b >= no_bond_class:
                        continue
                    bt = _bond_type_from_name(str(bond_types[b]))
                    if bt is None:
                        continue
                    if mol.GetBondBetweenAtoms(i, j) is None:
                        mol.AddBond(i, j, bt)

            m = mol.GetMol()
            Chem.SanitizeMol(m)
            smi = Chem.MolToSmiles(m, canonical=True)
            return smi if smi else None
        except Exception:
            return None


def _tokens_to_code_sequence(
    row: Sequence[int],
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    codebook_size: int,
) -> list[int]:
    toks = list(int(x) for x in row)
    if toks and toks[0] == int(bos_token_id):
        toks = toks[1:]
    out: list[int] = []
    for t in toks:
        if t == int(eos_token_id) or t == int(pad_token_id):
            break
        if 0 <= t < int(codebook_size):
            out.append(int(t))
    return out


def _load_ae_for_sampling(cfg: PriorConfig, device: torch.device):
    ae_cfg_path = _resolve_from_cwd(cfg.ae_config_path)
    ae_ckpt_path = _resolve_from_cwd(cfg.ae_ckpt_path)
    ae_cfg = yaml.safe_load(ae_cfg_path.read_text(encoding="utf-8")) or {}
    model_cfg = ae_cfg.get("model", {})

    ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=False)
    atom_types = list(ckpt.get("atom_types", []))
    bond_types = list(ckpt.get("bond_types", []))
    if not atom_types or not bond_types:
        raise RuntimeError("AE checkpoint missing atom_types/bond_types for decode")

    ae = GraphVQVAE(
        num_layers=int(model_cfg["num_layers"]),
        input_dim=len(atom_types),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        edge_dim=len(bond_types),
        codebook_size=int(model_cfg["codebook_size"]),
        lamb_edge=float(model_cfg.get("lamb_edge", 1.0)),
        lamb_node=float(model_cfg.get("lamb_node", 1.0)),
        pe_dim=int(model_cfg.get("pe_dim", 6)),
        vq_commitment_weight=float(model_cfg.get("vq_commitment_weight", 0.25)),
        vq_decay=float(model_cfg.get("vq_decay", 0.8)),
        vq_dead_code_threshold=float(model_cfg.get("vq_dead_code_threshold", 2.0)),
        vq_use_cosine_sim=bool(model_cfg.get("vq_use_cosine_sim", True)),
        vq_kmeans_init=bool(model_cfg.get("vq_kmeans_init", False)),
        vq_kmeans_iters=int(model_cfg.get("vq_kmeans_iters", 10)),
        vq_sample_codebook_temp=float(model_cfg.get("vq_sample_codebook_temp", 0.0)),
        vq_orthogonal_reg_weight=float(model_cfg.get("vq_orthogonal_reg_weight", 0.0)),
        vq_codebook_dim=model_cfg.get("vq_codebook_dim", None),
        gvt_heads=int(model_cfg.get("gvt_heads", 8)),
        gvt_dropout=float(model_cfg.get("gvt_dropout", 0.1)),
        encoder_backbone=str(model_cfg.get("encoder_backbone", "seq_lm")),
        lm_causal=bool(model_cfg.get("lm_causal", True)),
        lm_dropout=float(model_cfg.get("lm_dropout", 0.1)),
        lm_heads=int(model_cfg.get("lm_heads", 8)),
        decoder_dropout=float(model_cfg.get("decoder_dropout", 0.1)),
        decoder_heads=int(model_cfg.get("decoder_heads", 8)),
        decoder_rope_theta_base=float(model_cfg.get("decoder_rope_theta_base", 10000.0)),
        decoder_edge_mlp_hidden_dim=model_cfg.get("decoder_edge_mlp_hidden_dim", None),
        decoder_edge_mlp_out_dim=int(model_cfg.get("decoder_edge_mlp_out_dim", 16)),
        decoder_edge_recon_hidden_dim=int(model_cfg.get("decoder_edge_recon_hidden_dim", 64)),
    )
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    ae.load_state_dict(state, strict=True)
    ae.to(device)
    ae.eval()
    return ae, atom_types, bond_types


@torch.no_grad()
def _decode_sequences_to_smiles(
    code_seqs: Sequence[Sequence[int]],
    codebook: torch.Tensor,
    ae_model: GraphVQVAE,
    atom_types: Sequence[str],
    bond_types: Sequence[str],
    device: torch.device,
    decode_batch_graphs: int,
) -> list[str | None]:
    out: list[str | None] = []
    cb = codebook.to(device=device, dtype=torch.float32)

    for start in range(0, len(code_seqs), max(1, int(decode_batch_graphs))):
        chunk = code_seqs[start : start + max(1, int(decode_batch_graphs))]
        lengths = [int(len(x)) for x in chunk]
        if sum(lengths) == 0:
            out.extend([None] * len(chunk))
            continue

        flat_tokens: list[int] = []
        batch_vec: list[int] = []
        for gidx, seq in enumerate(chunk):
            for tok in seq:
                flat_tokens.append(int(tok))
                batch_vec.append(int(gidx))

        tok_t = torch.tensor(flat_tokens, dtype=torch.long, device=device)
        quantized = cb[tok_t]
        batch_t = torch.tensor(batch_vec, dtype=torch.long, device=device)
        pred_node, pred_adj, _mask_nodes, _dec_aux = ae_model.decoder(
            quantized,
            batch=batch_t,
            use_internal_reorder=False,
        )
        node_labels = pred_node.argmax(dim=-1).detach().cpu()
        edge_labels = pred_adj.argmax(dim=-1).detach().cpu()

        cursor = 0
        for gidx, n in enumerate(lengths):
            if n <= 0:
                out.append(None)
                continue
            node_part = node_labels[cursor : cursor + n]
            edge_part = edge_labels[gidx, :n, :n]
            cursor += n
            out.append(_graph_to_smiles(node_part, edge_part, atom_types=atom_types, bond_types=bond_types))

    return out


def _compute_vun(smiles_list: Sequence[str | None], ref_smiles_set: set[str]) -> dict:
    total = int(len(smiles_list))
    valid_canon = []
    for s in smiles_list:
        if s is None:
            continue
        c = _canonicalize_smiles(s)
        if c is not None:
            valid_canon.append(c)

    valid_n = int(len(valid_canon))
    unique_valid = set(valid_canon)
    unique_n = int(len(unique_valid))
    novel_n = int(sum(1 for s in unique_valid if s not in ref_smiles_set))

    validity = float(valid_n) / float(total) if total > 0 else 0.0
    uniqueness = float(unique_n) / float(valid_n) if valid_n > 0 else 0.0
    novelty = float(novel_n) / float(unique_n) if unique_n > 0 else 0.0
    vun = validity * uniqueness * novelty

    return {
        "total": total,
        "valid": valid_n,
        "unique": unique_n,
        "novel": novel_n,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "vun": vun,
    }

def _resolve_split_fcd_pref_path(save_dir: Path, cfg_path: str, split_name: str) -> Path:
    raw = str(cfg_path).strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p.with_name(f"{p.stem}_{split_name}{p.suffix}")
    return (save_dir / f"sample_fcd_pref_{split_name}.npz").resolve()


@torch.no_grad()
def _decode_index_split_to_canonical_smiles(
    dataset: VQSequenceDataset,
    indices: Sequence[int],
    codebook: torch.Tensor,
    ae_model: GraphVQVAE,
    atom_types: Sequence[str],
    bond_types: Sequence[str],
    device: torch.device,
    decode_batch_graphs: int,
    split_name: str,
) -> list[str]:
    chunk_size = max(1, int(decode_batch_graphs))
    raw_smiles: list[str | None] = []
    chunk: list[list[int]] = []
    for idx in indices:
        chunk.append(dataset.get_code_sequence(int(idx)).tolist())
        if len(chunk) >= chunk_size:
            raw_smiles.extend(
                _decode_sequences_to_smiles(
                    code_seqs=chunk,
                    codebook=codebook,
                    ae_model=ae_model,
                    atom_types=atom_types,
                    bond_types=bond_types,
                    device=device,
                    decode_batch_graphs=chunk_size,
                )
            )
            chunk = []
    if chunk:
        raw_smiles.extend(
            _decode_sequences_to_smiles(
                code_seqs=chunk,
                codebook=codebook,
                ae_model=ae_model,
                atom_types=atom_types,
                bond_types=bond_types,
                device=device,
                decode_batch_graphs=chunk_size,
            )
        )

    canon: list[str] = []
    for offset, smi in enumerate(raw_smiles):
        c = _canonicalize_smiles(smi) if smi is not None else None
        if c is None:
            raise ValueError(f"failed to canonicalize decoded reference molecule in {split_name} split at offset {offset}")
        canon.append(c)
    if len(canon) != len(indices):
        raise ValueError(f"reference split size mismatch for {split_name}: expected {len(indices)}, got {len(canon)}")
    return canon


def _load_or_build_reference_splits(
    dataset: VQSequenceDataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    codebook: torch.Tensor,
    ae_model: GraphVQVAE,
    atom_types: Sequence[str],
    bond_types: Sequence[str],
    device: torch.device,
    decode_batch_graphs: int,
    save_dir: Path,
) -> dict[str, list[str]]:
    ref_path = (save_dir / "reference_smiles_split.json").resolve()
    if ref_path.exists():
        payload = json.loads(ref_path.read_text(encoding="utf-8"))
        out = {
            "train": [str(x) for x in payload["train"]],
            "val": [str(x) for x in payload["val"]],
            "test": [str(x) for x in payload["test"]],
        }
        if len(out["train"]) != len(train_idx) or len(out["val"]) != len(val_idx) or len(out["test"]) != len(test_idx):
            raise ValueError(f"reference split cache size mismatch: {ref_path}")
        return out

    out = {
        "train": _decode_index_split_to_canonical_smiles(dataset, train_idx, codebook, ae_model, atom_types, bond_types, device, decode_batch_graphs, "train"),
        "val": _decode_index_split_to_canonical_smiles(dataset, val_idx, codebook, ae_model, atom_types, bond_types, device, decode_batch_graphs, "val"),
        "test": _decode_index_split_to_canonical_smiles(dataset, test_idx, codebook, ae_model, atom_types, bond_types, device, decode_batch_graphs, "test"),
    }
    ref_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def _init_fcd_metric(cfg: PriorConfig):
    from fcd_torch import FCD

    return FCD(
        device=str(cfg.sample_fcd_device),
        n_jobs=1,
        batch_size=int(cfg.sample_fcd_batch_size),
        canonize=False,
    )


def _prepare_fcd_pref(
    fcd_metric,
    save_dir: Path,
    cfg_path: str,
    split_name: str,
    ref_smiles_list: Sequence[str],
) -> dict | None:
    pref_path = _resolve_split_fcd_pref_path(save_dir=save_dir, cfg_path=cfg_path, split_name=split_name)
    cached_pref = _load_fcd_pref_cache(pref_path)
    if cached_pref is not None:
        print(f"  sample_fcd_pref_{split_name}: loaded {pref_path}")
        return cached_pref
    if len(ref_smiles_list) < 2:
        return None
    pref = fcd_metric.precalc(list(ref_smiles_list))
    if not (isinstance(pref, dict) and ("mu" in pref) and ("sigma" in pref)):
        raise ValueError(f"invalid FCD pref payload for {split_name} split")
    _save_fcd_pref_cache(pref_path, pref)
    print(f"  sample_fcd_pref_{split_name}: built and saved {pref_path}")
    return pref


@torch.no_grad()
def _sample_code_sequences(
    model,
    dataset: VQSequenceDataset,
    batch_size: int,
    n_total: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> tuple[list[list[int]], list[int]]:
    micro_batch = max(1, int(batch_size))
    target_n = max(1, int(n_total))
    code_seqs: list[list[int]] = []
    lengths: list[int] = []
    generated = 0
    while generated < target_n:
        b = min(micro_batch, target_n - generated)
        sampled = model.generate(
            bos_token_id=dataset.bos_token_id,
            eos_token_id=dataset.eos_token_id,
            batch_size=b,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        for row in sampled:
            core = _tokens_to_code_sequence(
                row=row.tolist(),
                bos_token_id=dataset.bos_token_id,
                eos_token_id=dataset.eos_token_id,
                pad_token_id=dataset.pad_token_id,
                codebook_size=dataset.codebook_size,
            )
            code_seqs.append(core)
            lengths.append(len(core))
        generated += int(b)
    return code_seqs, lengths


@torch.no_grad()
def _run_sampling_eval(
    *,
    model,
    dataset: VQSequenceDataset,
    cfg: PriorConfig,
    device: torch.device,
    tag: str,
    ae_model: GraphVQVAE,
    atom_types: Sequence[str],
    bond_types: Sequence[str],
    novelty_train_ref_set: set[str],
    fcd_metric,
    fcd_pref: dict | None,
    fcd_ref_n: int,
) -> tuple[dict, str]:
    code_seqs, lengths = _sample_code_sequences(
        model=model,
        dataset=dataset,
        batch_size=int(cfg.sample_batch),
        n_total=int(cfg.sample_num_samples),
        max_new_tokens=int(cfg.sample_max_new_tokens),
        temperature=float(cfg.sample_temperature),
        top_k=int(cfg.sample_top_k),
        device=device,
    )
    sample_log = {
        f"{tag}/len_mean": float(np.mean(lengths)),
        f"{tag}/len_std": float(np.std(lengths)),
        f"{tag}/n": int(len(lengths)),
    }
    msg = f"[{tag}] n={len(lengths)} len_mean={float(np.mean(lengths)):.2f} len_std={float(np.std(lengths)):.2f}"

    decoded_smiles = _decode_sequences_to_smiles(
        code_seqs=code_seqs,
        codebook=dataset.codebook,
        ae_model=ae_model,
        atom_types=atom_types,
        bond_types=bond_types,
        device=device,
        decode_batch_graphs=int(cfg.sample_decode_graph_batch),
    )

    if bool(cfg.sample_eval_vun):
        vun_metrics = _compute_vun(decoded_smiles, novelty_train_ref_set)
        sample_log.update({
            f"{tag}/vun_train_ref": float(vun_metrics["vun"]),
            f"{tag}/validity": float(vun_metrics["validity"]),
            f"{tag}/uniqueness": float(vun_metrics["uniqueness"]),
            f"{tag}/novelty_train_ref": float(vun_metrics["novelty"]),
        })
        msg += (
            f" vun_train_ref={vun_metrics['vun']:.4f} validity={vun_metrics['validity']:.4f}"
            f" uniqueness={vun_metrics['uniqueness']:.4f} novelty_train_ref={vun_metrics['novelty']:.4f}"
        )

    if bool(cfg.sample_eval_fcd):
        if (fcd_metric is not None) and (fcd_pref is not None):
            fcd_metrics = _compute_fcd_with_pref(
                decoded_smiles,
                fcd_metric=fcd_metric,
                pref=fcd_pref,
                n_ref=int(fcd_ref_n),
            )
        else:
            fcd_metrics = {"fcd": float("nan"), "n_gen": 0, "n_ref": int(fcd_ref_n), "ok": False}
        if bool(fcd_metrics.get("ok", False)):
            sample_log[f"{tag}/fcd"] = float(fcd_metrics["fcd"])
            msg += f" fcd={float(fcd_metrics['fcd']):.4f}"
        else:
            msg += f" fcd=nan(gen={int(fcd_metrics['n_gen'])},ref={int(fcd_metrics['n_ref'])})"

    return sample_log, msg



def make_collate_fn(pad_token_id: int):
    def _collate(batch: Sequence[torch.Tensor]) -> dict:
        padded = pad_sequence(batch, batch_first=True, padding_value=int(pad_token_id))
        input_ids = padded[:, :-1]
        labels = padded[:, 1:].clone()
        labels[labels == int(pad_token_id)] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "lengths": torch.tensor([int(x.numel()) for x in batch], dtype=torch.long),
        }

    return _collate


class GRUPrior(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        codebook_size: int = 0,
        codebook: torch.Tensor | None = None,
        use_codebook_input: bool = False,
        freeze_codebook_embedding: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.codebook_size = int(codebook_size)
        self.use_codebook_input = bool(use_codebook_input)

        self.token_emb: nn.Embedding | None = None
        self.special_emb: nn.Embedding | None = None
        self.input_proj: nn.Module = nn.Identity()

        if self.use_codebook_input:
            if codebook is None:
                raise ValueError("codebook tensor is required when use_codebook_input=True")
            cb = codebook.detach().float().cpu()
            if cb.dim() != 2:
                raise ValueError(f"codebook must be rank-2, got shape={tuple(cb.shape)}")
            if int(cb.size(0)) != self.codebook_size:
                raise ValueError(
                    f"codebook size mismatch: codebook_size={self.codebook_size}, codebook_rows={int(cb.size(0))}"
                )
            self.register_buffer("codebook_embed", cb, persistent=False)
            self.special_emb = nn.Embedding(3, int(cb.size(1)))
            nn.init.normal_(self.special_emb.weight, mean=float(cb.mean().item()), std=float(cb.std().item() + 1.0e-6))
            if bool(freeze_codebook_embedding):
                self.special_emb.weight.requires_grad = False
            if int(cb.size(1)) != int(emb_dim):
                self.input_proj = nn.Linear(int(cb.size(1)), int(emb_dim))
        else:
            self.token_emb = nn.Embedding(int(vocab_size), int(emb_dim), padding_idx=int(pad_token_id))

        self.rnn = nn.GRU(
            input_size=int(emb_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.lm_head = nn.Linear(int(hidden_dim), int(vocab_size))

    def _embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_codebook_input:
            assert self.token_emb is not None
            return self.token_emb(input_ids)

        assert hasattr(self, "codebook_embed")
        assert self.special_emb is not None
        codebook_embed = self.codebook_embed.to(device=input_ids.device)
        d = int(codebook_embed.size(1))
        x = torch.zeros((*input_ids.shape, d), device=input_ids.device, dtype=codebook_embed.dtype)

        code_mask = input_ids < int(self.codebook_size)
        if bool(code_mask.any().item()):
            x[code_mask] = codebook_embed[input_ids[code_mask]]

        special_mask = ~code_mask
        if bool(special_mask.any().item()):
            sid = (input_ids[special_mask] - int(self.codebook_size)).clamp(min=0, max=2)
            x[special_mask] = self.special_emb(sid)

        return self.input_proj(x)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self._embed_input(input_ids)
        h, _ = self.rnn(x)
        h = self.dropout(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        bos_token_id: int,
        eos_token_id: int,
        batch_size: int = 32,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        dev = torch.device(device)
        seq = torch.full((int(batch_size), 1), int(bos_token_id), dtype=torch.long, device=dev)
        finished = torch.zeros((int(batch_size),), dtype=torch.bool, device=dev)

        for _ in range(int(max_new_tokens)):
            logits, _ = self(seq, labels=None)
            step_logits = logits[:, -1, :] / max(float(temperature), 1.0e-6)
            if int(top_k) > 0:
                v, _ = torch.topk(step_logits, k=min(int(top_k), step_logits.size(-1)))
                step_logits[step_logits < v[:, [-1]]] = -float("inf")
            probs = nn.functional.softmax(step_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            nxt[finished.unsqueeze(-1)] = int(eos_token_id)
            seq = torch.cat([seq, nxt], dim=1)
            finished |= nxt.squeeze(-1).eq(int(eos_token_id))
            if bool(torch.all(finished).item()):
                break
        return seq

def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: PriorConfig, steps_per_epoch: int):
    if str(cfg.scheduler).lower() in ("none", "off", "disabled"):
        return None

    total_steps = max(1, int(cfg.epochs) * max(1, int(steps_per_epoch)))
    warmup_steps = int(round(float(cfg.warmup_ratio) * total_steps))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    min_lr_ratio = max(0.0, min(float(cfg.min_lr_ratio), 1.0))

    def _lr_lambda(step: int) -> float:
        s = max(0, int(step))
        if warmup_steps > 0 and s < warmup_steps:
            return max(1.0e-8, float(s + 1) / float(warmup_steps))
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(s - warmup_steps) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


@torch.no_grad()
def evaluate(model: GRUPrior, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_total = 0.0
    tok_correct = 0
    tok_count = 0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, loss = model(input_ids, labels=labels)
        loss_total += float(loss.item())
        n += 1

        pred = logits.argmax(dim=-1)
        valid = labels.ne(-100)
        tok_correct += int(pred[valid].eq(labels[valid]).sum().item())
        tok_count += int(valid.sum().item())

    n = max(n, 1)
    tok_count = max(tok_count, 1)
    return {"loss": loss_total / n, "token_acc": float(tok_correct) / float(tok_count)}


def parse_args() -> PriorConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/generative/config_gru_prior_moses_random_250k.yaml")
    args = ap.parse_args()
    return load_config(str(args.config))


def main() -> int:
    cfg = parse_args()
    set_seed(cfg.seed)

    if str(cfg.device).lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(cfg.device))

    dataset_path = _ensure_vq_dataset(cfg)
    dataset = VQSequenceDataset(dataset_path, max_seq_len=cfg.max_seq_len)
    tr_idx, va_idx, te_idx = _split_indices(
        len(dataset),
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        seed=cfg.seed,
    )
    collate_fn = make_collate_fn(dataset.pad_token_id)

    tr_loader = DataLoader(
        Subset(dataset, tr_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    va_loader = DataLoader(
        Subset(dataset, va_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    te_loader = DataLoader(
        Subset(dataset, te_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = GRUPrior(
        vocab_size=dataset.vocab_size,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_token_id=dataset.pad_token_id,
        codebook_size=dataset.codebook_size,
        codebook=dataset.codebook,
        use_codebook_input=bool(cfg.use_codebook_input),
        freeze_codebook_embedding=bool(cfg.freeze_codebook_embedding),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg=cfg, steps_per_epoch=len(tr_loader))

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "gru_prior_best.pt"
    log_path = save_dir / "gru_prior_train_log.jsonl"
    cfg_path = save_dir / "gru_prior_config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print("[gru_prior]")
    print(f"  dataset: {dataset_path}")
    print(f"  samples: {len(dataset)} train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    print(f"  codebook_size: {dataset.codebook_size} vocab_size: {dataset.vocab_size}")
    print(f"  tokens: bos={dataset.bos_token_id} eos={dataset.eos_token_id} pad={dataset.pad_token_id}")
    print(f"  model: emb={cfg.emb_dim} hidden={cfg.hidden_dim} layers={cfg.num_layers} dropout={cfg.dropout}")
    print(f"  input_mode: {'codebook_frozen' if cfg.use_codebook_input and cfg.freeze_codebook_embedding else ('codebook' if cfg.use_codebook_input else 'token_embedding')}")
    total_params, trainable_params = count_parameters(model)
    emb_mod = model.token_emb if model.token_emb is not None else model.special_emb
    emb_params, _ = count_parameters(emb_mod) if emb_mod is not None else (0, 0)
    rnn_params, _ = count_parameters(model.rnn)
    head_params, _ = count_parameters(model.lm_head)
    print(f"  params_total: {total_params}")
    print(f"  params_trainable: {trainable_params}")
    proj_params, _ = count_parameters(model.input_proj) if not isinstance(model.input_proj, nn.Identity) else (0, 0)
    print(f"  params_breakdown: emb={emb_params} proj={proj_params} rnn={rnn_params} head={head_params}")
    print(f"  device: {device}")
    wandb_run = _init_wandb(
        cfg,
        extra_config={
            "dataset_path": dataset_path,
            "train_size": len(tr_idx),
            "val_size": len(va_idx),
            "test_size": len(te_idx),
            "codebook_size": dataset.codebook_size,
            "vocab_size": dataset.vocab_size,
            "device": str(device),
            "model_type": "gru_prior",
        },
    )

    sample_ae = None
    sample_atom_types: list[str] = []
    sample_bond_types: list[str] = []
    train_ref_set: set[str] = set()
    val_ref_list: list[str] = []
    test_ref_list: list[str] = []
    sample_fcd_metric = None
    val_fcd_pref: dict | None = None
    test_fcd_pref: dict | None = None
    sample_need_decode = bool(cfg.sample_eval_vun) or bool(cfg.sample_eval_fcd)
    if sample_need_decode:
        sample_ae, sample_atom_types, sample_bond_types = _load_ae_for_sampling(cfg, device=device)
        ref_splits = _load_or_build_reference_splits(
            dataset=dataset,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=te_idx,
            codebook=dataset.codebook,
            ae_model=sample_ae,
            atom_types=sample_atom_types,
            bond_types=sample_bond_types,
            device=device,
            decode_batch_graphs=int(cfg.sample_decode_graph_batch),
            save_dir=save_dir,
        )
        train_ref_set = set(ref_splits["train"])
        val_ref_list = list(ref_splits["val"])
        test_ref_list = list(ref_splits["test"])
        if bool(cfg.sample_eval_fcd):
            sample_fcd_metric = _init_fcd_metric(cfg)
            val_fcd_pref = _prepare_fcd_pref(
                fcd_metric=sample_fcd_metric,
                save_dir=save_dir,
                cfg_path=str(cfg.sample_fcd_pref_path),
                split_name="val",
                ref_smiles_list=val_ref_list[: int(cfg.sample_fcd_ref_max)] if int(cfg.sample_fcd_ref_max) > 0 else val_ref_list,
            )
            test_fcd_pref = _prepare_fcd_pref(
                fcd_metric=sample_fcd_metric,
                save_dir=save_dir,
                cfg_path=str(cfg.sample_fcd_pref_path),
                split_name="test",
                ref_smiles_list=test_ref_list[: int(cfg.sample_fcd_ref_max)] if int(cfg.sample_fcd_ref_max) > 0 else test_ref_list,
            )
        print(
            f"  sample_eval: n={cfg.sample_num_samples} micro_batch={cfg.sample_batch} "
            f"decode_batch_graphs={cfg.sample_decode_graph_batch} "
            f"ref_train={len(train_ref_set)} ref_val={len(val_ref_list)} ref_test={len(test_ref_list)} fcd={bool(cfg.sample_eval_fcd)}"
        )

    best_val = None
    best_epoch = 0
    bad_epochs = 0
    last_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        last_epoch = epoch
        model.train()
        tr_loss = 0.0
        tr_tok_correct = 0
        tr_tok_count = 0
        steps = 0

        for batch in tr_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, labels=labels)
            loss.backward()
            if cfg.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            tr_loss += float(loss.item())
            pred = logits.argmax(dim=-1)
            valid = labels.ne(-100)
            tr_tok_correct += int(pred[valid].eq(labels[valid]).sum().item())
            tr_tok_count += int(valid.sum().item())
            steps += 1

        tr_metrics = {
            "loss": tr_loss / max(steps, 1),
            "token_acc": float(tr_tok_correct) / float(max(tr_tok_count, 1)),
        }
        va_metrics = evaluate(model, va_loader, device=device)

        line = {
            "epoch": epoch,
            "train": tr_metrics,
            "val": va_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(
            f"epoch {epoch:04d} | "
            f"tr(loss={tr_metrics['loss']:.4f},acc={tr_metrics['token_acc']:.4f}) | "
            f"va(loss={va_metrics['loss']:.4f},acc={va_metrics['token_acc']:.4f})"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": int(epoch),
                    "train/loss": float(tr_metrics["loss"]),
                    "train/token_acc": float(tr_metrics["token_acc"]),
                    "val/loss": float(va_metrics["loss"]),
                    "val/token_acc": float(va_metrics["token_acc"]),
                    "optim/lr": float(optimizer.param_groups[0]["lr"]),
                    "early_stop/bad_epochs": int(bad_epochs),
                    "best/epoch": int(best_epoch),
                    "best/val_loss": float(best_val) if best_val is not None else None,
                },
                step=int(epoch),
            )

        cur = float(va_metrics["loss"])
        if best_val is None or (best_val - cur) > cfg.early_stop_min_delta:
            best_val = cur
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                    "epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    "best_val": float(best_val),
                    "config": asdict(cfg),
                    "dataset_meta": {
                        "codebook_size": dataset.codebook_size,
                        "vocab_size": dataset.vocab_size,
                        "bos_token_id": dataset.bos_token_id,
                        "eos_token_id": dataset.eos_token_id,
                        "pad_token_id": dataset.pad_token_id,
                    },
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        if cfg.sample_every > 0 and (epoch % cfg.sample_every == 0):
            model.eval()
            sample_log, msg = _run_sampling_eval(
                model=model,
                dataset=dataset,
                cfg=cfg,
                device=device,
                tag="sample",
                ae_model=sample_ae,
                atom_types=sample_atom_types,
                bond_types=sample_bond_types,
                novelty_train_ref_set=train_ref_set,
                fcd_metric=sample_fcd_metric,
                fcd_pref=val_fcd_pref,
                fcd_ref_n=len(val_ref_list[: int(cfg.sample_fcd_ref_max)]) if int(cfg.sample_fcd_ref_max) > 0 else len(val_ref_list),
            )
            print(f"{msg} ref=novelty_train+fcd_val")
            if wandb_run is not None:
                wandb_run.log(sample_log, step=int(epoch))

        if epoch >= cfg.early_stop_min_epochs and bad_epochs >= cfg.early_stop_patience:
            print(f"[early_stop] stop at epoch {epoch}; best_epoch={best_epoch} best_val={best_val:.4f}")
            break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    test_metrics = evaluate(model, te_loader, device=device)
    print(f"[test] loss={test_metrics['loss']:.4f} acc={test_metrics['token_acc']:.4f}")

    test_sample_log: dict[str, float | int] = {}
    if sample_need_decode:
        test_sample_log, test_msg = _run_sampling_eval(
            model=model,
            dataset=dataset,
            cfg=cfg,
            device=device,
            tag="test_sample",
            ae_model=sample_ae,
            atom_types=sample_atom_types,
            bond_types=sample_bond_types,
            novelty_train_ref_set=train_ref_set,
            fcd_metric=sample_fcd_metric,
            fcd_pref=test_fcd_pref,
            fcd_ref_n=len(test_ref_list[: int(cfg.sample_fcd_ref_max)]) if int(cfg.sample_fcd_ref_max) > 0 else len(test_ref_list),
        )
        print(f"{test_msg} ref=novelty_train+fcd_test")

    test_summary = {
        "best_epoch": int(ckpt["best_epoch"]),
        "best_val": float(ckpt["best_val"]),
        "last_epoch": int(last_epoch),
        "test": test_metrics,
        "test_sample": test_sample_log,
    }
    test_summary_path = save_dir / "gru_prior_test_summary.json"
    test_summary_path.write_text(json.dumps(test_summary, indent=2), encoding="utf-8")

    if wandb_run is not None:
        wandb_run.summary.update({
            "best/epoch": int(ckpt["best_epoch"]),
            "best/val_loss": float(ckpt["best_val"]),
            "test/loss": float(test_metrics["loss"]),
            "test/token_acc": float(test_metrics["token_acc"]),
            **{k: v for k, v in test_sample_log.items()},
        })
        wandb_run.finish()
    print(f"[ckpt] saved: {ckpt_path}")
    print(f"[log] saved: {log_path}")
    print(f"[config] saved: {cfg_path}")
    print(f"[test_summary] saved: {test_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



