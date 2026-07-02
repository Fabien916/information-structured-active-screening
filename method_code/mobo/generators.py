
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from train_smiles_lm_benchmark import CharTokenizer, GRULM, TransformerLM


class CandidateGenerator(Protocol):
    def sample_smiles(self, n: int, *, temperature: float, top_k: int) -> list[str]:
        ...

    @property
    def info(self) -> dict:
        ...


class _TokenizerView(CharTokenizer):
    def __init__(self, payload: dict):
        self.pad_token = payload['pad_token']
        self.bos_token = payload['bos_token']
        self.eos_token = payload['eos_token']
        self.unk_token = payload['unk_token']
        self.itos = list(payload['itos'])
        self.stoi = dict(payload['stoi'])
        self.pad_id = int(payload['pad_id'])
        self.bos_id = int(payload['bos_id'])
        self.eos_id = int(payload['eos_id'])
        self.unk_id = int(payload['unk_id'])


@torch.no_grad()
def _sample_lm_sequences(
    model: torch.nn.Module,
    tok: CharTokenizer,
    device: torch.device,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    micro_batch: int,
) -> list[str]:
    model.eval()
    out: list[str] = []
    while len(out) < int(n):
        b = min(int(micro_batch), int(n) - len(out))
        seq = torch.full((b, 1), int(tok.bos_id), dtype=torch.long, device=device)
        finished = torch.zeros((b,), dtype=torch.bool, device=device)
        for _ in range(int(max_new_tokens)):
            logits, _ = model(seq, labels=None)
            step = logits[:, -1, :] / max(1.0e-6, float(temperature))
            if int(top_k) > 0:
                vals, _ = torch.topk(step, k=min(int(top_k), step.size(-1)))
                step[step < vals[:, [-1]]] = -float('inf')
            probs = F.softmax(step, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            nxt[finished.unsqueeze(-1)] = int(tok.eos_id)
            seq = torch.cat([seq, nxt], dim=1)
            finished |= nxt.squeeze(-1).eq(int(tok.eos_id))
            if bool(torch.all(finished).item()):
                break
        out.extend(tok.decode(row) for row in seq.tolist())
    return out


@dataclass
class VAEGenerator:
    model: torch.nn.Module
    vocab: object
    token_type: str
    latent_dim: int
    max_len: int
    device: torch.device

    def sample_smiles(self, n: int, *, temperature: float, top_k: int) -> list[str]:
        from smiles_vae_utils import sample_sequences as sample_vae_sequences
        samples_out = sample_vae_sequences(
            self.model,
            self.vocab,
            int(self.latent_dim),
            int(n),
            int(self.max_len),
            float(temperature),
            int(top_k),
            self.device,
            str(self.token_type),
        )
        return [str(smi) for _, smi in samples_out]

    @property
    def info(self) -> dict:
        return {
            'generator_kind': 'vae',
            'token_type': str(self.token_type),
            'max_len': int(self.max_len),
            'latent_dim': int(self.latent_dim),
        }


@dataclass
class LMGenerator:
    model: torch.nn.Module
    tokenizer: CharTokenizer
    max_len: int
    device: torch.device
    kind: str
    ckpt_path: str
    generator_batch_size: int = 128

    def sample_smiles(self, n: int, *, temperature: float, top_k: int) -> list[str]:
        return _sample_lm_sequences(
            self.model,
            self.tokenizer,
            self.device,
            n=int(n),
            max_new_tokens=int(self.max_len),
            temperature=float(temperature),
            top_k=int(top_k),
            micro_batch=int(self.generator_batch_size),
        )

    @property
    def info(self) -> dict:
        return {
            'generator_kind': str(self.kind),
            'ckpt_path': str(self.ckpt_path),
            'max_len': int(self.max_len),
            'generator_batch_size': int(self.generator_batch_size),
            'sample_batch': int(self.generator_batch_size),
            'vocab_size': int(self.tokenizer.vocab_size),
        }


def _load_lm_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
    model_kind: str,
    generator_batch_size: int = 128,
    max_len_cap: int | None = None,
) -> LMGenerator:
    ckpt_path = Path(ckpt_path).resolve()
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(payload.get('config') or {})
    tok_payload = dict(payload.get('tokenizer') or {})
    if not tok_payload:
        raise RuntimeError(f'LM checkpoint missing tokenizer payload: {ckpt_path}')
    tok = _TokenizerView(tok_payload)
    vocab_size = int(len(tok.itos))
    emb_dim = int(cfg.get('emb_dim', 128))
    hidden_dim = int(cfg.get('hidden_dim', 256))
    num_layers = int(cfg.get('num_layers', 2))
    dropout = float(cfg.get('dropout', 0.2))

    kind = str(model_kind).strip().lower()
    if kind == 'gru_lm':
        model = GRULM(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=int(tok.pad_id),
        )
    elif kind == 'transformer_lm':
        model = TransformerLM(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=int(tok.pad_id),
            num_heads=int(cfg.get('tf_num_heads', 8)),
            ff_mult=int(cfg.get('tf_ff_mult', 4)),
        )
    else:
        raise ValueError(f'Unsupported LM generator kind: {model_kind}')

    model.load_state_dict(payload['model_state'], strict=True)
    model.to(device)
    model.eval()
    base_max_len = int(cfg.get('sample_max_new_tokens', cfg.get('max_len', 140)))
    if max_len_cap is not None:
        max_len_cap = int(max_len_cap)
        if max_len_cap <= 0:
            raise ValueError(f'candidate.max_len must be positive when provided, got {max_len_cap}')
        max_len = min(base_max_len, max_len_cap)
    else:
        max_len = base_max_len
    return LMGenerator(
        model=model,
        tokenizer=tok,
        max_len=max_len,
        device=device,
        kind=kind,
        ckpt_path=str(ckpt_path),
        generator_batch_size=int(generator_batch_size),
    )


def load_candidate_generator(cfg: dict, device: torch.device):
    general_cfg = dict(cfg.get('general') or {})
    candidate_cfg = dict(cfg.get('candidate') or {})
    generator_cfg = dict(general_cfg.get('generator') or {})
    kind = str(generator_cfg.get('kind', general_cfg.get('generator_kind', 'vae'))).strip().lower()
    sample_batch = int(candidate_cfg.get('sample_batch', generator_cfg.get('sample_batch', 128)))
    generator_batch_raw = candidate_cfg.get('generator_batch_size', generator_cfg.get('generator_batch_size'))
    if generator_batch_raw in (None, '', 'null'):
        generator_batch_size = min(int(sample_batch), 128)
    else:
        generator_batch_size = int(generator_batch_raw)
    if generator_batch_size <= 0:
        raise ValueError(f'candidate.generator_batch_size must be positive, got {generator_batch_size}')
    max_len_cap_raw = candidate_cfg.get('max_len', generator_cfg.get('sample_max_new_tokens_cap'))
    max_len_cap = None if max_len_cap_raw in (None, '', 'null') else int(max_len_cap_raw)

    if kind == 'vae':
        from mobo_qpmhi import load_vae
        vae_ckpt = str(generator_cfg.get('ckpt', general_cfg.get('vae_ckpt', 'checkpoints/selfie_vae.pt')))
        model, vocab, vae_cfg = load_vae(vae_ckpt, device)
        data_cfg = dict(vae_cfg.get('data') or {})
        model_cfg = dict(vae_cfg.get('model') or {})
        base_max_len = int(data_cfg.get('max_len', 140))
        effective_max_len = min(base_max_len, max_len_cap) if max_len_cap is not None else base_max_len
        generator = VAEGenerator(
            model=model,
            vocab=vocab,
            token_type=str(data_cfg.get('token_type', 'selfies')),
            latent_dim=int(model_cfg.get('latent_dim', 256)),
            max_len=int(effective_max_len),
            device=device,
        )
        info = dict(generator.info)
        info['ckpt_path'] = str(Path(vae_ckpt).resolve())
        if max_len_cap is not None:
            info['max_len_cap'] = int(max_len_cap)
        return generator, info

    if kind in {'gru_lm', 'transformer_lm'}:
        ckpt = generator_cfg.get('ckpt')
        if not ckpt:
            raise RuntimeError(f'general.generator.ckpt is required for generator kind={kind}')
        generator = _load_lm_checkpoint(
            ckpt,
            device=device,
            model_kind=kind,
            generator_batch_size=generator_batch_size,
            max_len_cap=max_len_cap,
        )
        info = dict(generator.info)
        if max_len_cap is not None:
            info['max_len_cap'] = int(max_len_cap)
        return generator, info

    raise RuntimeError(f'Unsupported generator kind: {kind}')
