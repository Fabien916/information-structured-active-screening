#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from train_gru_prior import (
    PriorConfig,
    VQSequenceDataset,
    _build_scheduler,
    _ensure_vq_dataset,
    _init_fcd_metric,
    _init_wandb,
    _load_ae_for_sampling,
    _load_or_build_reference_splits,
    _prepare_fcd_pref,
    _run_sampling_eval,
    _split_indices,
    count_parameters,
    evaluate,
    load_config,
    make_collate_fn,
    set_seed,
)


class GPTBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(int(hidden_dim))
        self.ln_2 = nn.LayerNorm(int(hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(nhead),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(float(dropout))
        self.mlp = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) * 4),
            nn.GELU(),
            nn.Linear(int(hidden_dim) * 4, int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.ln_1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTPrior(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        codebook_size: int = 0,
        codebook: torch.Tensor | None = None,
        use_codebook_input: bool = False,
        freeze_codebook_embedding: bool = True,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.codebook_size = int(codebook_size)
        self.use_codebook_input = bool(use_codebook_input)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)

        self.token_emb: nn.Embedding | None = None
        self.special_emb: nn.Embedding | None = None
        self.input_proj: nn.Module = nn.Identity()

        if self.use_codebook_input:
            if codebook is None:
                raise ValueError('codebook tensor is required when use_codebook_input=True')
            cb = codebook.detach().float().cpu()
            if cb.dim() != 2:
                raise ValueError(f'codebook must be rank-2, got shape={tuple(cb.shape)}')
            if int(cb.size(0)) != self.codebook_size:
                raise ValueError(
                    f'codebook size mismatch: codebook_size={self.codebook_size}, codebook_rows={int(cb.size(0))}'
                )
            self.register_buffer('codebook_embed', cb, persistent=False)
            self.special_emb = nn.Embedding(3, int(cb.size(1)))
            nn.init.normal_(
                self.special_emb.weight,
                mean=float(cb.mean().item()),
                std=float(cb.std().item() + 1.0e-6),
            )
            if bool(freeze_codebook_embedding):
                self.special_emb.weight.requires_grad = False
            if int(cb.size(1)) != int(emb_dim):
                self.input_proj = nn.Linear(int(cb.size(1)), int(emb_dim))
        else:
            self.token_emb = nn.Embedding(int(vocab_size), int(emb_dim), padding_idx=int(pad_token_id))

        self.model_in_proj: nn.Module = nn.Identity()
        if int(emb_dim) != int(hidden_dim):
            self.model_in_proj = nn.Linear(int(emb_dim), int(hidden_dim))

        if int(hidden_dim) % 8 == 0:
            nhead = 8
        elif int(hidden_dim) % 4 == 0:
            nhead = 4
        elif int(hidden_dim) % 2 == 0:
            nhead = 2
        else:
            raise ValueError(f'hidden_dim must be divisible by 2 for GPTPrior, got {hidden_dim}')

        self.pos_emb = nn.Embedding(int(max_seq_len), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(
            [GPTBlock(hidden_dim=int(hidden_dim), nhead=int(nhead), dropout=float(dropout)) for _ in range(int(num_layers))]
        )
        self.ln_f = nn.LayerNorm(int(hidden_dim))
        self.lm_head = nn.Linear(int(hidden_dim), int(vocab_size), bias=False)

    def _embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_codebook_input:
            assert self.token_emb is not None
            x = self.token_emb(input_ids)
            return self.input_proj(x)

        assert hasattr(self, 'codebook_embed')
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
        _bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f'seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}')

        x = self._embed_input(input_ids)
        x = self.model_in_proj(x)
        pos_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        x = self.dropout(x + self.pos_emb(pos_ids).unsqueeze(0))

        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1
        )
        pad_mask = input_ids.eq(int(self.pad_token_id))
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, pad_mask=pad_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

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
        device: str | torch.device = 'cpu',
    ) -> torch.Tensor:
        dev = torch.device(device)
        seq = torch.full((int(batch_size), 1), int(bos_token_id), dtype=torch.long, device=dev)
        finished = torch.zeros((int(batch_size),), dtype=torch.bool, device=dev)

        for _ in range(int(max_new_tokens)):
            if seq.size(1) > self.max_seq_len:
                seq = seq[:, -self.max_seq_len :]
            logits, _ = self(seq, labels=None)
            step_logits = logits[:, -1, :] / max(float(temperature), 1.0e-6)
            if int(top_k) > 0:
                v, _ = torch.topk(step_logits, k=min(int(top_k), step_logits.size(-1)))
                step_logits[step_logits < v[:, [-1]]] = -float('inf')
            probs = nn.functional.softmax(step_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            nxt[finished.unsqueeze(-1)] = int(eos_token_id)
            seq = torch.cat([seq, nxt], dim=1)
            finished |= nxt.squeeze(-1).eq(int(eos_token_id))
            if bool(torch.all(finished).item()):
                break
        return seq

def parse_args() -> PriorConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/generative/config_gpt_prior_tcm_merged_local.yaml")
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

    model = GPTPrior(
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
        max_seq_len=max(256, int(cfg.max_seq_len) if int(cfg.max_seq_len) > 0 else 512),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _build_scheduler(optimizer, cfg=cfg, steps_per_epoch=len(tr_loader))

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "gpt_prior_best.pt"
    log_path = save_dir / "gpt_prior_train_log.jsonl"
    cfg_path = save_dir / "gpt_prior_config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print("[gpt_prior]")
    print(f"  dataset: {dataset_path}")
    print(f"  samples: {len(dataset)} train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    print(f"  codebook_size: {dataset.codebook_size} vocab_size: {dataset.vocab_size}")
    print(f"  tokens: bos={dataset.bos_token_id} eos={dataset.eos_token_id} pad={dataset.pad_token_id}")
    print(f"  model: emb={cfg.emb_dim} hidden={cfg.hidden_dim} layers={cfg.num_layers} dropout={cfg.dropout}")
    print(
        f"  input_mode: {'codebook_frozen' if cfg.use_codebook_input and cfg.freeze_codebook_embedding else ('codebook' if cfg.use_codebook_input else 'token_embedding')}"
    )
    total_params, trainable_params = count_parameters(model)
    emb_mod = model.token_emb if model.token_emb is not None else model.special_emb
    emb_params, _ = count_parameters(emb_mod) if emb_mod is not None else (0, 0)
    block_params = sum(count_parameters(block)[0] for block in model.blocks)
    pos_params, _ = count_parameters(model.pos_emb)
    ln_f_params, _ = count_parameters(model.ln_f)
    core_params = block_params + pos_params + ln_f_params
    head_params, _ = count_parameters(model.lm_head)
    print(f"  params_total: {total_params}")
    print(f"  params_trainable: {trainable_params}")
    proj_params, _ = count_parameters(model.input_proj) if not isinstance(model.input_proj, nn.Identity) else (0, 0)
    model_in_proj_params, _ = count_parameters(model.model_in_proj) if not isinstance(model.model_in_proj, nn.Identity) else (0, 0)
    print(
        f"  params_breakdown: emb={emb_params} input_proj={proj_params} model_in_proj={model_in_proj_params} "
        f"gpt_core={core_params} head={head_params}"
    )
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
            "model_type": "gpt_prior",
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
            print(f"{msg} ref=fcd_val")
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
        print(f"{test_msg} ref=fcd_test")

    test_summary = {
        "best_epoch": int(ckpt["best_epoch"]),
        "best_val": float(ckpt["best_val"]),
        "last_epoch": int(last_epoch),
        "test": test_metrics,
        "test_sample": test_sample_log,
    }
    test_summary_path = save_dir / "gpt_prior_test_summary.json"
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


