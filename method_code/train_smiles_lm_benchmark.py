#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import BRICS
from torch import nn
from torch.utils.data import DataLoader, Dataset

RDLogger.DisableLog("rdApp.*")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


@dataclass
class LMConfig:
    input_csv: str = "smiles_clean.csv"
    smiles_column: str = "SMILES_clean"
    out_dir: str = "runs/smiles_lm_benchmark"
    models: str = "gru,transformer,jtvae"
    seed: int = 42
    train_split: float = 0.9
    val_split: float = 0.1
    max_rows: int = 0
    max_len: int = 140
    min_len: int = 2

    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 80
    gru_epochs: int = 0
    transformer_batch_size: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-5
    clip_grad: float = 1.0
    early_stop_patience: int = 12
    early_stop_min_delta: float = 5e-4

    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    tf_num_heads: int = 8
    tf_ff_mult: int = 4

    sample_num: int = 2000
    sample_max_new_tokens: int = 140
    sample_temperature: float = 1.0
    sample_top_k: int = 100

    jtvae_cmd: str = ""


def load_config(path: str) -> LMConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = LMConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def canonicalize_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def load_smiles(cfg: LMConfig) -> List[str]:
    df = pd.read_csv(cfg.input_csv)
    if cfg.smiles_column not in df.columns:
        raise ValueError(f"column {cfg.smiles_column} not found in {cfg.input_csv}")
    col = df[cfg.smiles_column]
    col = col[col.notna()].astype(str).str.strip()
    if int(cfg.max_rows) > 0:
        col = col.iloc[: int(cfg.max_rows)]

    out: List[str] = []
    for s in col.tolist():
        cs = canonicalize_smiles(s)
        if cs is None:
            continue
        if len(cs) < int(cfg.min_len) or len(cs) > int(cfg.max_len):
            continue
        out.append(cs)
    if not out:
        raise RuntimeError("no valid smiles after filtering")
    return out


class CharTokenizer:
    def __init__(self, seqs: Sequence[str]):
        chars = sorted(set("".join(seqs)))
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.itos = [self.pad_token, self.bos_token, self.eos_token, self.unk_token] + chars
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return int(len(self.itos))

    def encode(self, s: str, max_len: int) -> List[int]:
        core = [self.stoi.get(ch, self.unk_id) for ch in s]
        core = core[: max(0, int(max_len) - 2)]
        return [self.bos_id] + core + [self.eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        chars: List[str] = []
        for i in ids:
            if int(i) == self.eos_id:
                break
            if int(i) in {self.pad_id, self.bos_id}:
                continue
            tok = self.itos[int(i)] if 0 <= int(i) < len(self.itos) else self.unk_token
            if tok.startswith("<") and tok.endswith(">"):
                continue
            chars.append(tok)
        return "".join(chars)


class SmilesDataset(Dataset):
    def __init__(self, smiles: Sequence[str], tok: CharTokenizer, max_len: int):
        self.smiles = list(smiles)
        self.tok = tok
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.tok.encode(self.smiles[idx], self.max_len)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    x = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
    y = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        x[i, :n] = item["input_ids"]
        y[i, :n] = item["labels"]
    return {"input_ids": x, "labels": y}


class GRULM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float, pad_id: int):
        super().__init__()
        self.pad_id = int(pad_id)
        self.emb = nn.Embedding(int(vocab_size), int(emb_dim), padding_idx=int(pad_id))
        self.rnn = nn.GRU(
            input_size=int(emb_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.lm_head = nn.Linear(int(hidden_dim), int(vocab_size))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.emb(input_ids)
        h, _ = self.rnn(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return logits, loss


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
        num_heads: int,
        ff_mult: int,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.hidden_dim = int(hidden_dim)
        self.emb = nn.Embedding(int(vocab_size), int(emb_dim), padding_idx=int(pad_id))
        self.input_proj = nn.Identity() if int(emb_dim) == int(hidden_dim) else nn.Linear(int(emb_dim), int(hidden_dim))
        nhead = int(num_heads)
        while nhead > 1 and int(hidden_dim) % nhead != 0:
            nhead -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=max(1, int(nhead)),
            dim_feedforward=int(hidden_dim) * int(ff_mult),
            dropout=float(dropout),
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.dropout = nn.Dropout(float(dropout))
        self.ln = nn.LayerNorm(int(hidden_dim))
        self.lm_head = nn.Linear(int(hidden_dim), int(vocab_size))

    def _pos(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, self.hidden_dim))
        )
        pe = torch.zeros((seq_len, self.hidden_dim), device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe.to(dtype=dtype)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        bsz, seq_len = input_ids.shape
        h = self.emb(input_ids)
        h = self.input_proj(h)
        h = self.dropout(h + self._pos(seq_len, h.device, h.dtype).unsqueeze(0))
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=h.device, dtype=torch.bool), diagonal=1)
        pad_mask = input_ids.eq(self.pad_id)
        h = self.backbone(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return logits, loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    tok_corr = 0
    tok_n = 0
    n = 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits, loss = model(x, labels=y)
        loss_sum += float(loss.item())
        pred = logits.argmax(dim=-1)
        valid = y.ne(-100)
        tok_corr += int(pred[valid].eq(y[valid]).sum().item())
        tok_n += int(valid.sum().item())
        n += 1
    avg_loss = loss_sum / max(1, n)
    ppl = math.exp(min(20.0, avg_loss))
    return {"loss": avg_loss, "acc": float(tok_corr) / float(max(1, tok_n)), "ppl": float(ppl)}


@torch.no_grad()
def sample_sequences(
    model: nn.Module,
    tok: CharTokenizer,
    device: torch.device,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    micro_batch: int = 128,
) -> List[str]:
    model.eval()
    out: List[str] = []
    while len(out) < int(n):
        b = min(int(micro_batch), int(n) - len(out))
        seq = torch.full((b, 1), int(tok.bos_id), dtype=torch.long, device=device)
        finished = torch.zeros((b,), dtype=torch.bool, device=device)
        for _ in range(int(max_new_tokens)):
            logits, _ = model(seq, labels=None)
            step = logits[:, -1, :] / max(1e-6, float(temperature))
            if int(top_k) > 0:
                vals, _ = torch.topk(step, k=min(int(top_k), step.size(-1)))
                step[step < vals[:, [-1]]] = -float("inf")
            probs = F.softmax(step, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            nxt[finished.unsqueeze(-1)] = int(tok.eos_id)
            seq = torch.cat([seq, nxt], dim=1)
            finished |= nxt.squeeze(-1).eq(int(tok.eos_id))
            if bool(torch.all(finished).item()):
                break
        for row in seq.tolist():
            out.append(tok.decode(row))
    return out


def generation_metrics(gen: Sequence[str], train_ref: Sequence[str]) -> Dict[str, float]:
    valid = [canonicalize_smiles(s) for s in gen]
    valid = [s for s in valid if s is not None]
    if len(gen) == 0:
        return {"validity": 0.0, "uniqueness": 0.0, "novelty": 0.0, "n_gen": 0, "n_valid": 0}
    uniq = len(set(valid)) / max(1, len(valid)) if valid else 0.0
    train_set = set(train_ref)
    novel = sum(1 for s in set(valid) if s not in train_set) / max(1, len(set(valid))) if valid else 0.0
    return {
        "validity": float(len(valid)) / float(len(gen)),
        "uniqueness": float(uniq),
        "novelty": float(novel),
        "n_gen": int(len(gen)),
        "n_valid": int(len(valid)),
    }


def _compute_vun(smiles_list: Sequence[str | None], ref_smiles_set: set[str]) -> dict:
    canon: list[str] = []
    for s in smiles_list:
        c = canonicalize_smiles(s) if s is not None else None
        if c is not None:
            canon.append(c)

    total_n = int(len(smiles_list))
    valid_n = int(len(canon))
    unique_set = set(canon)
    unique_n = int(len(unique_set))

    validity = float(valid_n) / float(total_n) if total_n > 0 else 0.0
    uniqueness = float(unique_n) / float(valid_n) if valid_n > 0 else 0.0
    novel_n = sum(1 for s in unique_set if s not in ref_smiles_set)
    novelty = float(novel_n) / float(unique_n) if unique_n > 0 else 0.0
    vun = validity * uniqueness * novelty

    return {
        "total": total_n,
        "valid": valid_n,
        "unique": unique_n,
        "novel": int(novel_n),
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "vun": vun,
    }


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
        c = canonicalize_smiles(s) if s is not None else None
        if c is not None:
            gen.append(c)

    if len(gen) < 2 or int(n_ref) < 2:
        return {
            "fcd": float("nan"),
            "n_gen": int(len(gen)),
            "n_ref": int(n_ref),
            "ok": False,
        }

    try:
        score = float(fcd_metric(pref=pref, gen=gen))
        return {
            "fcd": score,
            "n_gen": int(len(gen)),
            "n_ref": int(n_ref),
            "ok": True,
        }
    except Exception:
        return {
            "fcd": float("nan"),
            "n_gen": int(len(gen)),
            "n_ref": int(n_ref),
            "ok": False,
        }


def _model_batch_size(cfg: LMConfig, model_name: str) -> int:
    if model_name == "transformer" and int(cfg.transformer_batch_size) > 0:
        return int(cfg.transformer_batch_size)
    return int(cfg.batch_size)


def _model_epochs(cfg: LMConfig, model_name: str) -> int:
    if model_name == "gru" and int(cfg.gru_epochs) > 0:
        return int(cfg.gru_epochs)
    return int(cfg.epochs)


def run_lm(model_name: str, cfg: LMConfig, train_smiles: List[str], val_smiles: List[str], test_smiles: List[str], out_dir: Path, device: torch.device) -> Dict[str, object]:
    tok = CharTokenizer(train_smiles)
    train_ds = SmilesDataset(train_smiles, tok=tok, max_len=int(cfg.max_len))
    val_ds = SmilesDataset(val_smiles, tok=tok, max_len=int(cfg.max_len))
    test_ds = SmilesDataset(test_smiles, tok=tok, max_len=int(cfg.max_len))

    batch_size = _model_batch_size(cfg, model_name)
    epochs = _model_epochs(cfg, model_name)
    collate = lambda b: collate_batch(b, pad_id=tok.pad_id)
    tr_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(cfg.num_workers), collate_fn=collate)
    va_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(cfg.num_workers), collate_fn=collate)
    te_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(cfg.num_workers), collate_fn=collate)

    if model_name == "gru":
        model = GRULM(
            vocab_size=tok.vocab_size,
            emb_dim=int(cfg.emb_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
            pad_id=tok.pad_id,
        )
    elif model_name == "transformer":
        model = TransformerLM(
            vocab_size=tok.vocab_size,
            emb_dim=int(cfg.emb_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
            pad_id=tok.pad_id,
            num_heads=int(cfg.tf_num_heads),
            ff_mult=int(cfg.tf_ff_mult),
        )
    else:
        raise ValueError(model_name)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    sample_eval_every = 10
    sample_eval_n = 2000
    train_ref_set = set(train_smiles)
    val_ref_list = list(val_smiles)
    test_ref_list = list(test_smiles)
    sample_fcd_metric = None
    val_fcd_pref = None
    test_fcd_pref = None
    val_pref_path = out_dir / f"{model_name}_fcd_ref_pref_val.npz"
    test_pref_path = out_dir / f"{model_name}_fcd_ref_pref_test.npz"
    try:
        from fcd_torch import FCD

        sample_fcd_metric = FCD(device=str(device), n_jobs=0)
        cached_val = _load_fcd_pref_cache(val_pref_path)
        if cached_val is not None:
            val_fcd_pref = cached_val
            print(f"  sample_fcd_pref_val: loaded {val_pref_path}")
        else:
            pref = sample_fcd_metric.precalc(val_ref_list)
            if not (isinstance(pref, dict) and ("mu" in pref) and ("sigma" in pref)):
                raise ValueError("invalid val FCD pref payload")
            val_fcd_pref = pref
            _save_fcd_pref_cache(val_pref_path, val_fcd_pref)
            print(f"  sample_fcd_pref_val: built and saved {val_pref_path}")

        cached_test = _load_fcd_pref_cache(test_pref_path)
        if cached_test is not None:
            test_fcd_pref = cached_test
            print(f"  sample_fcd_pref_test: loaded {test_pref_path}")
        else:
            pref = sample_fcd_metric.precalc(test_ref_list)
            if not (isinstance(pref, dict) and ("mu" in pref) and ("sigma" in pref)):
                raise ValueError("invalid test FCD pref payload")
            test_fcd_pref = pref
            _save_fcd_pref_cache(test_pref_path, test_fcd_pref)
            print(f"  sample_fcd_pref_test: built and saved {test_pref_path}")
    except Exception:
        sample_fcd_metric = None
        val_fcd_pref = None
        test_fcd_pref = None

    best_val = None
    best_epoch = 0
    bad = 0
    ckpt = out_dir / f"{model_name}_best.pt"
    logs: List[Dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        tr_loss = 0.0
        tr_corr = 0
        tr_tok = 0
        n_steps = 0
        for batch in tr_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad()
            logits, loss = model(x, labels=y)
            loss.backward()
            if float(cfg.clip_grad) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_grad))
            opt.step()

            tr_loss += float(loss.item())
            pred = logits.argmax(dim=-1)
            valid = y.ne(-100)
            tr_corr += int(pred[valid].eq(y[valid]).sum().item())
            tr_tok += int(valid.sum().item())
            n_steps += 1

        tr_metrics = {
            "loss": tr_loss / max(1, n_steps),
            "acc": float(tr_corr) / float(max(1, tr_tok)),
        }
        va_metrics = evaluate(model, va_loader, device)
        logs.append({"epoch": epoch, "tr_loss": tr_metrics["loss"], "tr_acc": tr_metrics["acc"], "va_loss": va_metrics["loss"], "va_acc": va_metrics["acc"], "va_ppl": va_metrics["ppl"]})
        print(
            f"epoch {epoch:04d} | tr(loss={tr_metrics['loss']:.4f},acc={tr_metrics['acc']:.4f}) | "
            f"va(loss={va_metrics['loss']:.4f},acc={va_metrics['acc']:.4f},ppl={va_metrics['ppl']:.2f})",
            flush=True,
        )

        if epoch % sample_eval_every == 0:
            sampled = sample_sequences(
                model=model,
                tok=tok,
                device=device,
                n=sample_eval_n,
                max_new_tokens=int(cfg.sample_max_new_tokens),
                temperature=float(cfg.sample_temperature),
                top_k=int(cfg.sample_top_k),
                micro_batch=min(int(batch_size), 256),
            )
            vun_metrics = _compute_vun(sampled, train_ref_set)
            fcd_val = float("nan")
            if (sample_fcd_metric is not None) and (val_fcd_pref is not None):
                fcd_metrics = _compute_fcd_with_pref(
                    smiles_list=sampled,
                    fcd_metric=sample_fcd_metric,
                    pref=val_fcd_pref,
                    n_ref=len(val_ref_list),
                )
                fcd_val = float(fcd_metrics.get("fcd", float("nan")))
            fcd_n = int(fcd_metrics.get("n_gen", 0)) if (sample_fcd_metric is not None and val_fcd_pref is not None) else 0
            print(
                f"[sample] epoch={epoch} n={sample_eval_n} "
                f"vun={vun_metrics['vun']:.4f} validity={vun_metrics['validity']:.4f} "
                f"uniqueness={vun_metrics['uniqueness']:.4f} novelty={vun_metrics['novelty']:.4f} "
                f"fcd_val={fcd_val:.4f} fcd_n={fcd_n}",
                flush=True,
            )

        cur = float(va_metrics["loss"])
        if best_val is None or (best_val - cur) > float(cfg.early_stop_min_delta):
            best_val = cur
            best_epoch = int(epoch)
            bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": int(epoch), "tokenizer": tok.__dict__, "config": cfg.__dict__}, ckpt)
        else:
            bad += 1
            if bad >= int(cfg.early_stop_patience):
                print(
                    f"[early_stop] {model_name} stop at epoch {epoch}; best_epoch={best_epoch} val_loss={best_val:.4f}",
                    flush=True,
                )
                break

    best = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate(model, te_loader, device)
    gen = sample_sequences(
        model=model,
        tok=tok,
        device=device,
        n=int(cfg.sample_num),
        max_new_tokens=int(cfg.sample_max_new_tokens),
        temperature=float(cfg.sample_temperature),
        top_k=int(cfg.sample_top_k),
        micro_batch=min(int(batch_size), 256),
    )
    g_metrics = generation_metrics(gen, train_ref=train_smiles)
    test_sample = dict(g_metrics)
    if (sample_fcd_metric is not None) and (test_fcd_pref is not None):
        fcd_metrics = _compute_fcd_with_pref(
            smiles_list=gen,
            fcd_metric=sample_fcd_metric,
            pref=test_fcd_pref,
            n_ref=len(test_ref_list),
        )
        test_sample["fcd"] = float(fcd_metrics.get("fcd", float("nan")))
        test_sample["fcd_n"] = int(fcd_metrics.get("n_gen", 0))
    else:
        test_sample["fcd"] = float("nan")
        test_sample["fcd_n"] = 0

    with (out_dir / f"{model_name}_train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "model": model_name,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "params": count_parameters(model),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val if best_val is not None else float("nan")),
        "test": test_metrics,
        "gen": g_metrics,
        "test_sample": test_sample,
        "n_train": len(train_smiles),
        "n_val": len(val_smiles),
        "n_test": len(test_smiles),
    }
    with (out_dir / f"{model_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_jtvae_external(cfg: LMConfig, out_dir: Path) -> Dict[str, object]:
    cmd = str(cfg.jtvae_cmd).strip()
    if not cmd:
        return {"model": "jtvae", "ok": False, "error": "jtvae_cmd is empty; skipped"}
    proc = subprocess.run(cmd, shell=True, cwd=str(Path.cwd()), capture_output=True, text=True)
    rec = {
        "model": "jtvae",
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
    with (out_dir / "jtvae_external_run.json").open("w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec




def _brics_fragments(smi: str) -> List[str]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    try:
        frags = sorted(BRICS.BRICSDecompose(mol, keepNonLeafNodes=False, singlePass=True))
    except Exception:
        frags = []
    if not frags:
        c = canonicalize_smiles(smi)
        return [c] if c is not None else []
    return frags


class FragTokenizer:
    def __init__(self, seqs: Sequence[Sequence[str]]):
        toks = sorted(set(tok for seq in seqs for tok in seq))
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.itos = [self.pad_token, self.bos_token, self.eos_token, self.unk_token] + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return int(len(self.itos))

    def encode(self, frags: Sequence[str], max_len: int) -> List[int]:
        core = [self.stoi.get(f, self.unk_id) for f in frags]
        core = core[: max(0, int(max_len) - 2)]
        return [self.bos_id] + core + [self.eos_id]

    def decode(self, ids: Sequence[int]) -> List[str]:
        out: List[str] = []
        for i in ids:
            j = int(i)
            if j == self.eos_id:
                break
            if j in {self.pad_id, self.bos_id}:
                continue
            tok = self.itos[j] if 0 <= j < len(self.itos) else self.unk_token
            if tok.startswith("<") and tok.endswith(">"):
                continue
            out.append(tok)
        return out


class FragSeqDataset(Dataset):
    def __init__(self, smiles: Sequence[str], tok: FragTokenizer, max_len: int):
        self.frag_seqs = [_brics_fragments(s) for s in smiles]
        self.tok = tok
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.frag_seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tok.encode(self.frag_seqs[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long)


def collate_frag_batch(batch: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(x.size(0) for x in batch)
    out = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : x.size(0)] = x
    return out


class JTFragVAE(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, dropout: float, pad_id: int):
        super().__init__()
        self.pad_id = int(pad_id)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.emb = nn.Embedding(int(vocab_size), int(emb_dim), padding_idx=int(pad_id))
        self.enc = nn.GRU(
            input_size=int(emb_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.mu = nn.Linear(int(hidden_dim), int(latent_dim))
        self.logvar = nn.Linear(int(hidden_dim), int(latent_dim))

        self.z2h = nn.Linear(int(latent_dim), int(hidden_dim) * int(num_layers))
        self.dec = nn.GRU(
            input_size=int(emb_dim) + int(latent_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(int(hidden_dim), int(vocab_size))

    def encode(self, ids: torch.Tensor):
        emb = self.emb(ids)
        out, _ = self.enc(emb)
        h = out[:, -1, :]
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_teacher(self, x_ids: torch.Tensor, z: torch.Tensor):
        emb = self.emb(x_ids)
        zrep = z.unsqueeze(1).expand(-1, x_ids.size(1), -1)
        dec_in = torch.cat([emb, zrep], dim=-1)
        h0 = self.z2h(z).view(self.num_layers, x_ids.size(0), self.hidden_dim).contiguous()
        out, _ = self.dec(dec_in, h0)
        logits = self.head(out)
        return logits

    def forward(self, ids: torch.Tensor, beta: float = 0.05):
        x = ids[:, :-1]
        y = ids[:, 1:]
        mu, logvar = self.encode(ids)
        z = self.reparam(mu, logvar)
        logits = self.decode_teacher(x, z)
        recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=self.pad_id)
        kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        loss = recon + float(beta) * kl
        return logits, recon, kl, loss

    @torch.no_grad()
    def sample_ids(self, bos_id: int, eos_id: int, n: int, max_len: int, temperature: float, top_k: int, device: torch.device) -> List[List[int]]:
        self.eval()
        z = torch.randn((int(n), self.latent_dim), device=device)
        h = self.z2h(z).view(self.num_layers, int(n), self.hidden_dim).contiguous()
        cur = torch.full((int(n), 1), int(bos_id), dtype=torch.long, device=device)
        done = torch.zeros((int(n),), dtype=torch.bool, device=device)
        out_ids: List[List[int]] = [[int(bos_id)] for _ in range(int(n))]

        for _ in range(int(max_len)):
            emb = self.emb(cur)
            zin = z.unsqueeze(1)
            din = torch.cat([emb, zin], dim=-1)
            o, h = self.dec(din, h)
            step = self.head(o[:, -1, :]) / max(1e-6, float(temperature))
            if int(top_k) > 0:
                vals, _ = torch.topk(step, k=min(int(top_k), step.size(-1)))
                step[step < vals[:, [-1]]] = -float("inf")
            probs = F.softmax(step, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            nxt[done.unsqueeze(-1)] = int(eos_id)
            for i in range(int(n)):
                out_ids[i].append(int(nxt[i, 0].item()))
            done |= nxt.squeeze(-1).eq(int(eos_id))
            cur = nxt
            if bool(torch.all(done).item()):
                break
        return out_ids


def _frags_to_smiles(frags: Sequence[str]) -> str | None:
    if not frags:
        return None
    mols = [Chem.MolFromSmiles(f) for f in frags]
    mols = [m for m in mols if m is not None]
    if not mols:
        return None

    try:
        built = BRICS.BRICSBuild(mols, onlyCompleteMols=True, scrambleReagents=False, maxDepth=max(1, len(mols)))
        for m in built:
            s = Chem.MolToSmiles(m, canonical=True)
            c = canonicalize_smiles(s)
            if c is not None:
                return c
    except Exception:
        pass

    for f in frags:
        c = canonicalize_smiles(f)
        if c is not None:
            return c
    return None


def evaluate_jtvae(model: JTFragVAE, loader: DataLoader, device: torch.device, beta: float = 0.05) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    tok_corr = 0
    tok_n = 0
    n = 0
    with torch.no_grad():
        for ids in loader:
            ids = ids.to(device)
            logits, recon, kl, loss = model(ids, beta=beta)
            y = ids[:, 1:]
            pred = logits.argmax(dim=-1)
            valid = y.ne(model.pad_id)
            tok_corr += int(pred[valid].eq(y[valid]).sum().item())
            tok_n += int(valid.sum().item())
            loss_sum += float(loss.item())
            recon_sum += float(recon.item())
            kl_sum += float(kl.item())
            n += 1
    avg_loss = loss_sum / max(1, n)
    ppl = math.exp(min(20.0, avg_loss))
    return {
        "loss": avg_loss,
        "recon": recon_sum / max(1, n),
        "kl": kl_sum / max(1, n),
        "acc": float(tok_corr) / float(max(1, tok_n)),
        "ppl": float(ppl),
    }


def run_jtvae_internal(cfg: LMConfig, train_smiles: List[str], val_smiles: List[str], test_smiles: List[str], out_dir: Path, device: torch.device) -> Dict[str, object]:
    print("[jtvae_internal] using BRICS fragment-junction approximation")

    train_frag = [_brics_fragments(s) for s in train_smiles]
    tok = FragTokenizer(train_frag)

    max_frag_len = 0
    for seq in train_frag:
        max_frag_len = max(max_frag_len, len(seq))
    max_frag_len = max(8, min(32, int(max_frag_len) + 2))

    tr_ds = FragSeqDataset(train_smiles, tok=tok, max_len=max_frag_len)
    va_ds = FragSeqDataset(val_smiles, tok=tok, max_len=max_frag_len)
    te_ds = FragSeqDataset(test_smiles, tok=tok, max_len=max_frag_len)

    collate = lambda b: collate_frag_batch(b, pad_id=tok.pad_id)
    tr_loader = DataLoader(tr_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers), collate_fn=collate)
    va_loader = DataLoader(va_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers), collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers), collate_fn=collate)

    latent_dim = max(32, int(cfg.hidden_dim) // 2)
    model = JTFragVAE(
        vocab_size=tok.vocab_size,
        emb_dim=int(cfg.emb_dim),
        hidden_dim=int(cfg.hidden_dim),
        latent_dim=int(latent_dim),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        pad_id=tok.pad_id,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    sample_eval_every = 10
    sample_eval_n = 2000
    train_ref_set = set(train_smiles)
    val_ref_list = list(val_smiles)
    test_ref_list = list(test_smiles)
    sample_fcd_metric = None
    val_fcd_pref = None
    test_fcd_pref = None
    val_pref_path = out_dir / "jtvae_fcd_ref_pref_val.npz"
    test_pref_path = out_dir / "jtvae_fcd_ref_pref_test.npz"
    try:
        from fcd_torch import FCD

        sample_fcd_metric = FCD(device=str(device), n_jobs=0)
        cached_val = _load_fcd_pref_cache(val_pref_path)
        if cached_val is not None:
            val_fcd_pref = cached_val
            print(f"  sample_fcd_pref_val: loaded {val_pref_path}")
        else:
            pref = sample_fcd_metric.precalc(val_ref_list)
            if not (isinstance(pref, dict) and ("mu" in pref) and ("sigma" in pref)):
                raise ValueError("invalid val FCD pref payload")
            val_fcd_pref = pref
            _save_fcd_pref_cache(val_pref_path, val_fcd_pref)
            print(f"  sample_fcd_pref_val: built and saved {val_pref_path}")

        cached_test = _load_fcd_pref_cache(test_pref_path)
        if cached_test is not None:
            test_fcd_pref = cached_test
            print(f"  sample_fcd_pref_test: loaded {test_pref_path}")
        else:
            pref = sample_fcd_metric.precalc(test_ref_list)
            if not (isinstance(pref, dict) and ("mu" in pref) and ("sigma" in pref)):
                raise ValueError("invalid test FCD pref payload")
            test_fcd_pref = pref
            _save_fcd_pref_cache(test_pref_path, test_fcd_pref)
            print(f"  sample_fcd_pref_test: built and saved {test_pref_path}")
    except Exception:
        sample_fcd_metric = None
        val_fcd_pref = None
        test_fcd_pref = None

    best_val = None
    best_epoch = 0
    bad = 0
    beta = 0.05
    ckpt = out_dir / "jtvae_best.pt"
    logs: List[Dict[str, float]] = []

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        tr_loss = 0.0
        tr_recon = 0.0
        tr_kl = 0.0
        tr_corr = 0
        tr_tok = 0
        n_steps = 0
        for ids in tr_loader:
            ids = ids.to(device)
            opt.zero_grad()
            logits, recon, kl, loss = model(ids, beta=beta)
            loss.backward()
            if float(cfg.clip_grad) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_grad))
            opt.step()

            y = ids[:, 1:]
            pred = logits.argmax(dim=-1)
            valid = y.ne(model.pad_id)
            tr_corr += int(pred[valid].eq(y[valid]).sum().item())
            tr_tok += int(valid.sum().item())
            tr_loss += float(loss.item())
            tr_recon += float(recon.item())
            tr_kl += float(kl.item())
            n_steps += 1

        tr_metrics = {
            "loss": tr_loss / max(1, n_steps),
            "recon": tr_recon / max(1, n_steps),
            "kl": tr_kl / max(1, n_steps),
            "acc": float(tr_corr) / float(max(1, tr_tok)),
        }
        va_metrics = evaluate_jtvae(model, va_loader, device=device, beta=beta)
        logs.append({
            "epoch": epoch,
            "tr_loss": tr_metrics["loss"],
            "tr_recon": tr_metrics["recon"],
            "tr_kl": tr_metrics["kl"],
            "tr_acc": tr_metrics["acc"],
            "va_loss": va_metrics["loss"],
            "va_acc": va_metrics["acc"],
            "va_ppl": va_metrics["ppl"],
        })

        print(
            f"epoch {epoch:04d} | tr(loss={tr_metrics['loss']:.4f},recon={tr_metrics['recon']:.4f},kl={tr_metrics['kl']:.4f},acc={tr_metrics['acc']:.4f}) | "
            f"va(loss={va_metrics['loss']:.4f},acc={va_metrics['acc']:.4f},ppl={va_metrics['ppl']:.2f})",
            flush=True,
        )

        if epoch % sample_eval_every == 0:
            sampled_ids = model.sample_ids(
                bos_id=tok.bos_id,
                eos_id=tok.eos_id,
                n=sample_eval_n,
                max_len=max_frag_len,
                temperature=float(cfg.sample_temperature),
                top_k=int(cfg.sample_top_k),
                device=device,
            )
            sampled_smiles = [_frags_to_smiles(tok.decode(ids)) for ids in sampled_ids]
            vun_metrics = _compute_vun(sampled_smiles, train_ref_set)
            fcd_val = float("nan")
            if (sample_fcd_metric is not None) and (val_fcd_pref is not None):
                fcd_metrics = _compute_fcd_with_pref(
                    smiles_list=sampled_smiles,
                    fcd_metric=sample_fcd_metric,
                    pref=val_fcd_pref,
                    n_ref=len(val_ref_list),
                )
                fcd_val = float(fcd_metrics.get("fcd", float("nan")))
            fcd_n = int(fcd_metrics.get("n_gen", 0)) if (sample_fcd_metric is not None and val_fcd_pref is not None) else 0
            print(
                f"[sample] epoch={epoch} n={sample_eval_n} "
                f"vun={vun_metrics['vun']:.4f} validity={vun_metrics['validity']:.4f} "
                f"uniqueness={vun_metrics['uniqueness']:.4f} novelty={vun_metrics['novelty']:.4f} "
                f"fcd_val={fcd_val:.4f} fcd_n={fcd_n}",
                flush=True,
            )

        cur = float(va_metrics["loss"])
        if best_val is None or (best_val - cur) > float(cfg.early_stop_min_delta):
            best_val = cur
            best_epoch = int(epoch)
            bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": int(epoch), "config": cfg.__dict__, "tokenizer": tok.__dict__, "max_frag_len": int(max_frag_len)}, ckpt)
        else:
            bad += 1
            if bad >= int(cfg.early_stop_patience):
                print(f"[early_stop] jtvae stop at epoch {epoch}; best_epoch={best_epoch} val_loss={best_val:.4f}", flush=True)
                break

    best = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    te_metrics = evaluate_jtvae(model, te_loader, device=device, beta=beta)

    final_ids = model.sample_ids(
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        n=int(cfg.sample_num),
        max_len=max_frag_len,
        temperature=float(cfg.sample_temperature),
        top_k=int(cfg.sample_top_k),
        device=device,
    )
    final_smiles = [_frags_to_smiles(tok.decode(ids)) for ids in final_ids]
    g_metrics = generation_metrics([s if s is not None else "" for s in final_smiles], train_ref=train_smiles)
    test_sample = dict(g_metrics)
    if (sample_fcd_metric is not None) and (test_fcd_pref is not None):
        fcd_metrics = _compute_fcd_with_pref(
            smiles_list=final_smiles,
            fcd_metric=sample_fcd_metric,
            pref=test_fcd_pref,
            n_ref=len(test_ref_list),
        )
        test_sample["fcd"] = float(fcd_metrics.get("fcd", float("nan")))
        test_sample["fcd_n"] = int(fcd_metrics.get("n_gen", 0))
    else:
        test_sample["fcd"] = float("nan")
        test_sample["fcd_n"] = 0

    with (out_dir / "jtvae_train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "model": "jtvae",
        "impl": "internal_brics_fragment",
        "params": count_parameters(model),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val if best_val is not None else float("nan")),
        "test": te_metrics,
        "gen": g_metrics,
        "test_sample": test_sample,
        "n_train": len(train_smiles),
        "n_val": len(val_smiles),
        "n_test": len(test_smiles),
    }
    with (out_dir / "jtvae_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

def split_data(smiles: List[str], train_ratio: float, val_ratio: float, seed: int):
    idx = np.arange(len(smiles))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    n = len(smiles)
    n_train = int(round(n * float(train_ratio)))
    n_val = int(round(n * float(val_ratio)))
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]
    return [smiles[i] for i in tr], [smiles[i] for i in va], [smiles[i] for i in te]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/generative/config_smiles_lm_benchmark.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.seed))
    out_dir = Path(str(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles = load_smiles(cfg)
    train_smiles, val_smiles, test_smiles = split_data(
        smiles,
        train_ratio=float(cfg.train_split),
        val_ratio=float(cfg.val_split),
        seed=int(cfg.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.strip().lower() for m in str(cfg.models).split(",") if m.strip()]

    print("[smiles_lm_benchmark]")
    print(f"  csv: {cfg.input_csv} col={cfg.smiles_column}")
    print(f"  samples: total={len(smiles)} train={len(train_smiles)} val={len(val_smiles)} test={len(test_smiles)}")
    print(f"  device: {device}")
    print(f"  models: {models}")

    all_results: List[Dict[str, object]] = []
    for m in models:
        if m in {"gru", "transformer"}:
            res = run_lm(m, cfg, train_smiles, val_smiles, test_smiles, out_dir=out_dir, device=device)
        elif m in {"jtvae", "jt-vae"}:
            if str(cfg.jtvae_cmd).strip():
                res = run_jtvae_external(cfg, out_dir=out_dir)
            else:
                res = run_jtvae_internal(cfg, train_smiles, val_smiles, test_smiles, out_dir=out_dir, device=device)
        else:
            res = {"model": m, "ok": False, "error": "unsupported model name"}
        all_results.append(res)
        print(f"[summary:{m}] {json.dumps(res, ensure_ascii=False)}")

    with (out_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[done] summary saved: {out_dir / 'benchmark_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
