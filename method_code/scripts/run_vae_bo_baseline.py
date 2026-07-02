#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data.ligand_only_3d_dataset import GLOBAL_LIGAND3D_CACHE_DIR
from mobo.config_utils import load_config
from mobo.init_selection import load_latent_library_csv
from mobo.metrics import evaluate_oracle_accuracy_from_csv
from mobo.oracle import run_oracle_docking
from mobo.retrospective import build_objective_tensor, compute_hypervolume
from mobo.smiles_utils import canonicalize_smiles_noh
from mobo_qpmhi import _fill_extra_metrics_in_csv, append_candidates_to_smiles_csv, load_vae
from run_mobo_main_experiment import (
    _assign_new_rows_random_predictor_split,
    _batch_hv_gain,
    _build_or_load_downstream_protocol,
    _build_oracle_call_kwargs,
    _default_init_library_candidates,
    _dynamic_ref_point_from_labeled,
    _export_selected_batch_with_oracle,
    _load_observed_df,
    _log,
    _normalize_ratio_config,
    _prepare_init_dataset,
    _require_dict,
    _resolve_cli_or_config,
    _resolve_append_global_vocab,
    _resolve_init_pretrain_ligand_vocab,
    _resolve_or_train_init_encoder_ckpt,
    _resolve_oracle_asset_root,
    _section,
    _validate_mobo_config,
)
from torch.nn.utils.rnn import pad_sequence
from selfies_data import TokenVocab, to_selfies, tokenize_selfies, tokenize_smiles
from smiles_vae_utils import decode_latent_batch


SPECIAL_ATOM_EXTRA_DIM = 3


def _load_vae_bundle(ckpt_path: str, device: torch.device):
    model, vocab, cfg = load_vae(ckpt_path, device)
    return model, vocab, cfg


def _tokenize_for_vae(smiles: str, token_type: str) -> list[str]:
    src = str(smiles).strip()
    if not src:
        raise RuntimeError("Encountered empty SMILES while encoding VAE latents.")
    if token_type == "smiles":
        tokens = tokenize_smiles(src)
    elif token_type == "selfies":
        selfies_str = to_selfies(src)
        tokens = tokenize_selfies(selfies_str) if selfies_str else []
    else:
        raise RuntimeError(f"Unsupported VAE token_type: {token_type}")
    if not tokens:
        raise RuntimeError(f"Failed to tokenize SMILES for VAE encoding: {smiles}")
    return tokens



def _encode_smiles_to_latent_mu(
    model: torch.nn.Module,
    vocab: TokenVocab,
    smiles_list: list[str],
    token_type: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    seqs = []
    for smi in smiles_list:
        tokens = _tokenize_for_vae(str(smi), token_type=token_type)
        ids = [int(vocab.start_id)] + vocab.encode(tokens) + [int(vocab.end_id)]
        seqs.append(torch.tensor(ids, dtype=torch.long))
    mus: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(seqs), int(batch_size)):
            seq_batch = pad_sequence(seqs[start : start + int(batch_size)], batch_first=True, padding_value=int(vocab.pad_id)).to(device)
            mu, _logvar = model.encoder(seq_batch)
            mus.append(mu.detach().cpu().numpy())
    if not mus:
        raise RuntimeError("Failed to encode any labeled molecules into VAE latent space.")
    return np.concatenate(mus, axis=0)



def _decode_latents(
    model: torch.nn.Module,
    vocab: TokenVocab,
    z: torch.Tensor,
    *,
    max_len: int,
    temperature: float,
    top_k: int,
    token_type: str,
) -> list[tuple[str, str]]:
    return decode_latent_batch(
        model=model,
        vocab=vocab,
        z=z,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        token_type=token_type,
    )


def _decode_one_latent_with_retries(
    model: torch.nn.Module,
    vocab: TokenVocab,
    z: torch.Tensor,
    *,
    max_len: int,
    temperature: float,
    top_k: int,
    token_type: str,
    accepted: set[str],
    generator: torch.Generator,
    decode_attempts: int,
    decode_noise_std: float,
) -> tuple[str, str, str, torch.Tensor, int] | None:
    base = z.view(1, -1)
    for attempt_idx in range(max(1, int(decode_attempts))):
        if attempt_idx == 0:
            z_try = base
        else:
            noise = torch.randn(
                base.shape,
                generator=generator,
                device=base.device,
                dtype=base.dtype,
            ) * float(decode_noise_std)
            z_try = base + noise
        seq_text, raw_smiles = _decode_latents(
            model,
            vocab,
            z_try,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            token_type=token_type,
        )[0]
        try:
            canon = canonicalize_smiles_noh(raw_smiles)
        except ValueError:
            continue
        if canon and canon not in accepted:
            return seq_text, raw_smiles, canon, z_try[0].detach().clone(), attempt_idx + 1
    return None


def _objective_tensor_from_df(
    df: pd.DataFrame,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
) -> np.ndarray:
    return build_objective_tensor(
        df,
        dock_sign=dock_sign,
        qed_sign=qed_sign,
        sa_sign=sa_sign,
    ).detach().cpu().numpy().astype(np.float64)



def _select_gp_training_subset_indices(train_y_full: torch.Tensor, baseline_cap: int) -> torch.Tensor:
    total = int(train_y_full.shape[0])
    cap = int(baseline_cap)
    if cap <= 0 or total <= cap:
        return torch.arange(total, device=train_y_full.device, dtype=torch.long)
    try:
        from botorch.utils.multi_objective.pareto import is_non_dominated
    except Exception as exc:
        raise RuntimeError("VAE+MOBO baseline requires botorch to compute a Pareto-preserving GP subset.") from exc

    values = train_y_full.detach().cpu().numpy()
    n_obj = int(values.shape[1])
    remaining = torch.arange(total, device=train_y_full.device, dtype=torch.long)
    selected: list[int] = []

    while int(remaining.numel()) > 0 and len(selected) < cap:
        front_mask = is_non_dominated(train_y_full.index_select(0, remaining))
        front_indices = [int(idx) for idx in remaining[front_mask].detach().cpu().tolist()]
        front_indices = sorted(
            front_indices,
            key=lambda idx: tuple([-float(values[idx, obj_idx]) for obj_idx in range(n_obj)] + [int(idx)]),
        )
        take = min(cap - len(selected), len(front_indices))
        selected.extend(front_indices[:take])
        remaining = remaining[~front_mask]

    if len(selected) != cap:
        raise RuntimeError(f"Failed to build deterministic GP subset: selected={len(selected)} cap={cap}")
    return torch.as_tensor(selected, dtype=torch.long, device=train_y_full.device)



def _fit_gp_and_score_pool(
    x_train: np.ndarray,
    y_train: np.ndarray,
    z_pool: np.ndarray,
    ref_point: list[float],
    device: torch.device,
    score_batch_size: int,
    baseline_cap: int,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.sampling import SobolQMCNormalSampler
        from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
        from botorch.utils.transforms import normalize
        from gpytorch.mlls import SumMarginalLogLikelihood
    except Exception as exc:
        raise RuntimeError("VAE+MOBO baseline requires botorch and gpytorch to be installed.") from exc

    train_x_full = torch.as_tensor(x_train, dtype=torch.float64, device=device)
    train_y_full = torch.as_tensor(y_train, dtype=torch.float64, device=device)
    pool_x = torch.as_tensor(z_pool, dtype=torch.float64, device=device)
    if int(baseline_cap) > 0 and int(train_x_full.shape[0]) > int(baseline_cap):
        keep = _select_gp_training_subset_indices(train_y_full, int(baseline_cap))
        train_x = train_x_full.index_select(0, keep)
        train_y = train_y_full.index_select(0, keep)
    else:
        train_x = train_x_full
        train_y = train_y_full

    bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])
    span = (bounds[1] - bounds[0]).clamp_min(1e-6)
    x_norm = normalize(train_x, torch.stack([bounds[0], bounds[0] + span]))
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_norm = (train_y - y_mean) / y_std
    pool_x_norm = normalize(pool_x, torch.stack([bounds[0], bounds[0] + span]))

    ref_point_raw = torch.as_tensor(ref_point, dtype=train_y.dtype, device=device).view(1, -1)
    if int(ref_point_raw.shape[1]) != int(train_y.shape[1]):
        raise RuntimeError(
            f"ref_point dimension mismatch for VAE+MOBO GP fitting: ref_point={int(ref_point_raw.shape[1])} objectives={int(train_y.shape[1])}"
        )
    ref_point_t = ((ref_point_raw - y_mean) / y_std).view(-1)

    gps = [SingleTaskGP(x_norm, y_norm[:, j : j + 1]) for j in range(int(y_norm.shape[1]))]
    model = ModelListGP(*gps)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    partitioning = FastNondominatedPartitioning(ref_point=ref_point_t, Y=y_norm)
    sampler = SobolQMCNormalSampler(torch.Size([int(mc_samples)]))
    acq = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_t.tolist(),
        partitioning=partitioning,
        sampler=sampler,
    )

    acq_batches: list[torch.Tensor] = []
    mean_batches: list[torch.Tensor] = []
    std_batches: list[torch.Tensor] = []
    chunk_size = max(1, int(score_batch_size))
    with torch.no_grad():
        for start in range(0, int(pool_x_norm.shape[0]), chunk_size):
            pool_chunk = pool_x_norm[start : start + chunk_size]
            acq_batches.append(acq(pool_chunk.unsqueeze(1)).detach().cpu())
            posterior = model.posterior(pool_chunk)
            mean_batches.append((posterior.mean.detach().cpu() * y_std.cpu()) + y_mean.cpu())
            std_batches.append(posterior.variance.clamp_min(0.0).sqrt().detach().cpu() * y_std.cpu())
    acq_vals = torch.cat(acq_batches, dim=0).numpy().reshape(int(pool_x_norm.shape[0]))
    mean = torch.cat(mean_batches, dim=0).numpy().reshape(int(pool_x_norm.shape[0]), -1)
    std = torch.cat(std_batches, dim=0).numpy().reshape(int(pool_x_norm.shape[0]), -1)

    del acq, sampler, partitioning, mll, model, gps, pool_x_norm, pool_x, x_norm, y_norm, train_x, train_y, train_x_full, train_y_full
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return acq_vals, mean, std


def _select_from_latent_pool(
    model: torch.nn.Module,
    vocab: TokenVocab,
    token_type: str,
    max_len: int,
    latent_dim: int,
    *,
    latent_pool_size: int,
    batch_size: int,
    oversample_factor: int,
    resample_rounds: int,
    seed: int,
    iter_idx: int,
    device: torch.device,
    acq_fn,
    temperature: float,
    top_k: int,
    decode_attempts: int,
    decode_noise_std: float,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    existing_smiles: set[str],
) -> pd.DataFrame:
    selected_rows: list[dict] = []
    accepted = set(str(x) for x in existing_smiles)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 3001 * int(iter_idx))

    for round_idx in range(1, int(resample_rounds) + 1):
        z_pool = torch.randn((int(latent_pool_size), int(latent_dim)), generator=generator, device=device, dtype=torch.float64)
        acq_vals_np, mean_np, std_np = acq_fn(z_pool.cpu().numpy())
        order = np.argsort(-acq_vals_np)
        candidate_n = min(int(len(order)), int(batch_size) * max(int(oversample_factor), 1))
        top_idx = order[:candidate_n]
        for rank_pos, pool_idx in enumerate(top_idx.tolist()):
            z_candidate = z_pool[pool_idx].to(dtype=torch.float32)
            decoded = _decode_one_latent_with_retries(
                model,
                vocab,
                z_candidate,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                token_type=token_type,
                accepted=accepted,
                generator=generator,
                decode_attempts=decode_attempts,
                decode_noise_std=decode_noise_std,
            )
            if decoded is None:
                continue
            seq_text, raw_smiles, canon, z_used, decode_attempt = decoded
            accepted.add(canon)
            pred_obj = mean_np[pool_idx]
            pred_std = std_np[pool_idx]
            selected_rows.append({
                "latent_round": int(round_idx),
                "latent_rank": int(len(selected_rows) + 1),
                "pool_rank": int(rank_pos + 1),
                "decode_attempt": int(decode_attempt),
                "acq_qlognehvi": float(acq_vals_np[pool_idx]),
                "pred_dock_score_mean": float(pred_obj[0] / float(dock_sign)),
                "pred_qed_mean": float(pred_obj[1] / float(qed_sign)),
                "pred_sa_score_mean": float(pred_obj[2] / float(sa_sign)),
                "pred_dock_score_std": float(pred_std[0] / abs(float(dock_sign))),
                "pred_qed_std": float(pred_std[1] / abs(float(qed_sign))),
                "pred_sa_score_std": float(pred_std[2] / abs(float(sa_sign))),
                "smiles_raw": str(raw_smiles),
                "smiles_canonical": str(canon),
                "decoded_sequence": str(seq_text),
                **{f"z_{dim}": float(z_used[dim].item()) for dim in range(int(latent_dim))},
            })
            if len(selected_rows) >= int(batch_size):
                break
        if len(selected_rows) >= int(batch_size):
            break

    if len(selected_rows) < int(batch_size):
        raise RuntimeError(
            f"VAE+BO failed to decode enough unique molecules: selected={len(selected_rows)} required={batch_size}"
        )
    return pd.DataFrame(selected_rows[: int(batch_size)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the VAE+MOBO baseline under the main 8UN4 protocol.")
    ap.add_argument("--config", default="config/surrogate/config.yaml")
    ap.add_argument("--mobo-config", default="config/mobo/vae_bo_baseline.yaml")
    ap.add_argument("--vae-config", default="config/generative/config_vae.yaml")
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--reuse-init-from", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--latent-pool-size", type=int, default=None)
    ap.add_argument("--decode-oversample", type=int, default=None)
    ap.add_argument("--resample-rounds", type=int, default=None)
    ap.add_argument("--sample-temperature", type=float, default=None)
    ap.add_argument("--sample-top-k", type=int, default=None)
    ap.add_argument("--decode-attempts", type=int, default=None)
    ap.add_argument("--decode-noise-std", type=float, default=None)
    ap.add_argument("--latent-encode-batch-size", type=int, default=None)
    ap.add_argument("--init-library-csv", default=None)
    ap.add_argument("--init-library-smiles-col", default=None)
    ap.add_argument("--init-encoder-ckpt", default=None)
    ap.add_argument("--init-training-dataset-root", default=None)
    ap.add_argument("--init-ligand-vocab-file", default=None)
    ap.add_argument("--init-library-max-rows", type=int, default=None)
    ap.add_argument("--butina-cutoff", type=float, default=None)
    ap.add_argument("--fp-radius", type=int, default=None)
    ap.add_argument("--fp-bits", type=int, default=None)
    ap.add_argument("--init-train-valid-total", type=int, default=None)
    ap.add_argument("--holdout-test-total", type=int, default=None)
    ap.add_argument("--selection-seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    vae_cfg = load_config(args.vae_config)
    mobo_cfg = load_config(args.mobo_config)
    mobo_validate_cfg = dict(mobo_cfg)
    mobo_validate_cfg.pop("vae_bo", None)
    _validate_mobo_config(mobo_validate_cfg)
    mobo_paths_cfg = _require_dict(mobo_cfg.get("paths"), "mobo.paths")
    mobo_init_cfg = _require_dict(mobo_cfg.get("init_selection"), "mobo.init_selection")
    mobo_experiment_cfg = _require_dict(mobo_cfg.get("experiment"), "mobo.experiment")
    vae_bo_cfg = _require_dict(mobo_cfg.get("vae_bo"), "mobo.vae_bo")

    resolved = argparse.Namespace(
        init_library_csv=args.init_library_csv if args.init_library_csv is not None else mobo_paths_cfg.get("init_library_csv"),
        init_library_smiles_col=_resolve_cli_or_config(args.init_library_smiles_col, mobo_paths_cfg.get("init_library_smiles_col"), "paths.init_library_smiles_col"),
        init_encoder_ckpt=args.init_encoder_ckpt if args.init_encoder_ckpt is not None else mobo_paths_cfg.get("init_encoder_ckpt"),
        init_training_dataset_root=_resolve_cli_or_config(args.init_training_dataset_root, mobo_paths_cfg.get("init_training_dataset_root"), "paths.init_training_dataset_root"),
        init_ligand_vocab_file=args.init_ligand_vocab_file if args.init_ligand_vocab_file is not None else mobo_paths_cfg.get("init_ligand_vocab_file"),
        output_dir=_resolve_cli_or_config(args.output_dir, mobo_paths_cfg.get("output_dir"), "paths.output_dir"),
        init_library_max_rows=int(_resolve_cli_or_config(args.init_library_max_rows, mobo_init_cfg.get("library_max_rows", 0), "init_selection.library_max_rows")),
        butina_cutoff=float(_resolve_cli_or_config(args.butina_cutoff, mobo_init_cfg.get("butina_cutoff", 0.6), "init_selection.butina_cutoff")),
        fp_radius=int(_resolve_cli_or_config(args.fp_radius, mobo_init_cfg.get("fp_radius", 2), "init_selection.fp_radius")),
        fp_bits=int(_resolve_cli_or_config(args.fp_bits, mobo_init_cfg.get("fp_bits", 2048), "init_selection.fp_bits")),
        init_train_valid_total=int(_resolve_cli_or_config(args.init_train_valid_total, mobo_init_cfg.get("init_train_valid_total", 480), "init_selection.init_train_valid_total")),
        holdout_test_total=int(_resolve_cli_or_config(args.holdout_test_total, mobo_init_cfg.get("holdout_test_total", 120), "init_selection.holdout_test_total")),
        selection_seed=int(_resolve_cli_or_config(args.selection_seed, mobo_init_cfg.get("selection_seed", mobo_experiment_cfg.get("seed")), "init_selection.selection_seed")),
        rounds=int(_resolve_cli_or_config(args.rounds, mobo_experiment_cfg.get("rounds"), "experiment.rounds")),
        seed=int(_resolve_cli_or_config(args.seed, mobo_experiment_cfg.get("seed"), "experiment.seed")),
    )

    general_cfg = _section(cfg, "general")
    dataset_cfg = _section(cfg, "dataset")
    selection_cfg = _section(cfg, "selection")
    objective_cfg = _section(cfg, "objective")
    oracle_cfg = _section(cfg, "oracle")
    run_cfg = _section(cfg, "run")
    candidate_cfg = _section(cfg, "candidate")
    candidate_3d_cfg = dict(_section(candidate_cfg, "candidate_3d"))
    candidate_3d_cfg["seed"] = int(resolved.seed)

    weights = [float(x) for x in selection_cfg.get("weights", [0.7, 0.1, 0.2])]
    if len(weights) != 3:
        raise RuntimeError(f"selection.weights must have length 3 for dock/QED/SA, got {weights}")

    latent_pool_size = int(args.latent_pool_size if args.latent_pool_size is not None else vae_bo_cfg.get("latent_pool_size", 3000))
    decode_oversample = int(args.decode_oversample if args.decode_oversample is not None else vae_bo_cfg.get("decode_oversample", 6))
    resample_rounds = int(args.resample_rounds if args.resample_rounds is not None else vae_bo_cfg.get("resample_rounds", 8))
    sample_temperature = float(args.sample_temperature if args.sample_temperature is not None else vae_bo_cfg.get("sample_temperature", 1.0))
    sample_top_k = int(args.sample_top_k if args.sample_top_k is not None else vae_bo_cfg.get("sample_top_k", 0))
    decode_attempts = int(args.decode_attempts if args.decode_attempts is not None else vae_bo_cfg.get("decode_attempts", 6))
    decode_noise_std = float(args.decode_noise_std if args.decode_noise_std is not None else vae_bo_cfg.get("decode_noise_std", 0.05))
    latent_encode_batch_size = int(args.latent_encode_batch_size if args.latent_encode_batch_size is not None else vae_bo_cfg.get("encode_batch_size", 256))
    latent_score_batch_size = int(vae_bo_cfg.get("score_batch_size", 128))
    gp_baseline_cap = int(vae_bo_cfg.get("baseline_cap", 768))
    gp_mc_samples = int(vae_bo_cfg.get("mc_samples", 64))

    if not bool(objective_cfg.get("use_sa", True)):
        raise RuntimeError("VAE+MOBO baseline requires dock/QED/SA objectives.")

    device = torch.device("cuda" if str(general_cfg.get("device", "auto")).lower() != "cpu" and torch.cuda.is_available() else "cpu")
    gpu_total_gib = None
    gpu_free_gib = None
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu_total_gib = round(float(total_b) / float(1024 ** 3), 3)
        gpu_free_gib = round(float(free_b) / float(1024 ** 3), 3)
    out_root = Path(str(resolved.output_dir)) / "vae_bo"
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_root = out_root / "dataset"
    batch_size = int(selection_cfg.get("batch_size", 20))
    pretrain_split_ratio = _normalize_ratio_config(
        dataset_cfg.get("pretrain_split_ratio", dataset_cfg.get("split_ratio", [0.8, 0.1, 0.1])),
        default=(0.8, 0.1, 0.1),
        expected_len=3,
    )
    pretrain_split_seed = int(dataset_cfg.get("pretrain_split_seed", dataset_cfg.get("split_seed", resolved.seed)))
    pretrain_use_existing_split = bool(dataset_cfg.get("pretrain_use_existing_split", False))
    downstream_pool_ratio = _normalize_ratio_config(
        dataset_cfg.get("downstream_pool_split_ratio", [0.8, 0.2]),
        default=(0.8, 0.2),
        expected_len=2,
    )
    downstream_pool_seed = int(dataset_cfg.get("downstream_pool_split_seed", resolved.seed))
    scaffold_include_chirality = bool(dataset_cfg.get("scaffold_include_chirality", False))
    predictor_valid_ratio = float(_section(cfg, "surrogate").get("predictor_valid_ratio", dataset_cfg.get("predictor_valid_ratio", 0.2)))
    predictor_valid_seed = int(_section(cfg, "surrogate").get("predictor_valid_seed", dataset_cfg.get("split_seed", resolved.seed)))
    dock_sign = float(objective_cfg.get("dock_sign", -1.0))
    qed_sign = float(objective_cfg.get("qed_sign", 1.0))
    sa_sign = float(objective_cfg.get("sa_sign", -1.0))
    fixed_ref_point = [float(x) for x in objective_cfg.get("ref_point", [0.0, 0.0, -20.0])]
    dock_valid_max = objective_cfg.get("dock_valid_max", 0.0)
    dock_valid_max = None if dock_valid_max is None else float(dock_valid_max)

    vae_ckpt = str(args.vae_ckpt or vae_bo_cfg.get("vae_ckpt") or _section(vae_cfg, "train").get("save_path", "checkpoints/selfie_vae.pt"))
    if not Path(vae_ckpt).exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")
    vae_model, vae_vocab, vae_train_cfg = _load_vae_bundle(vae_ckpt, device)
    token_type = str(_section(vae_train_cfg, "data").get("token_type", "selfies"))
    max_len = int(_section(vae_train_cfg, "data").get("max_len", 120))
    latent_dim = int(_section(vae_train_cfg, "model").get("latent_dim", 64))

    shared_ligand3d_cache = GLOBAL_LIGAND3D_CACHE_DIR
    shared_ligand3d_cache.mkdir(parents=True, exist_ok=True)
    candidate_3d_cfg["cache_dir"] = str(shared_ligand3d_cache)
    oracle_asset_root = _resolve_oracle_asset_root(oracle_cfg)
    online_oracle_kwargs = _build_oracle_call_kwargs(
        run_cfg=run_cfg,
        oracle_cfg=oracle_cfg,
        evaluate_reference=False,
        dock_valid_max=dock_valid_max,
    )
    _log(
        "vae_bo oracle config | "
        f"backend={online_oracle_kwargs['docking_backend']} "
        f"exhaustiveness={oracle_cfg.get('oracle_exhaustiveness', 32)} "
        f"pocket_radius={oracle_cfg.get('oracle_pocket_radius', 10.0)} "
        f"vina_cuda_thread={online_oracle_kwargs['vina_cuda_thread']} "
        f"vina_cuda_search_depth={online_oracle_kwargs['vina_cuda_search_depth']} "
        f"unidock_scoring={online_oracle_kwargs['unidock_scoring']} "
        f"unidock_search_mode={online_oracle_kwargs['unidock_search_mode']} "
        f"unidock_num_modes={online_oracle_kwargs['unidock_num_modes']} "
        f"unidock_timeout_sec={online_oracle_kwargs['unidock_timeout_sec']}"
    )

    if args.reuse_init_from:
        init_root = Path(str(args.reuse_init_from)).resolve()
        init_set_path = init_root / "init_set.csv"
        if not init_set_path.exists():
            raise FileNotFoundError(f"reuse-init-from missing init_set.csv: {init_set_path}")
        init_df = pd.read_csv(init_set_path)
        if init_df.empty:
            raise RuntimeError(f"reuse-init-from init_set is empty: {init_set_path}")
        meta_path = init_root / "experiment_meta.json"
        existing_methods: list[str] = []
        if meta_path.exists():
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(existing_meta, dict):
                raise RuntimeError(f"experiment_meta.json must contain an object: {meta_path}")
            existing_methods = [str(x).strip() for x in existing_meta.get("methods", []) if str(x).strip()]
        global_vocab = _resolve_append_global_vocab(
            out_root=init_root,
            existing_methods=existing_methods,
            init_df=init_df,
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
        )
        init_encoder_ckpt = None
        _log(f"vae_bo reuse_init | from={init_root} init_n={init_df.shape[0]}")
    else:
        default_init_library_candidates = _default_init_library_candidates(resolved.init_training_dataset_root, oracle_asset_root)
        if resolved.init_library_csv:
            init_library_csv = Path(str(resolved.init_library_csv)).resolve()
        else:
            init_library_csv = None
            for candidate in default_init_library_candidates:
                if candidate.exists():
                    init_library_csv = candidate.resolve()
                    break
            if init_library_csv is None:
                raise RuntimeError("Failed to resolve init library for VAE+BO baseline.")
        init_library_raw_df, init_library_smiles_col = load_latent_library_csv(
            init_library_csv,
            smiles_col=None if not resolved.init_library_smiles_col else str(resolved.init_library_smiles_col),
        )
        if int(resolved.init_library_max_rows) > 0:
            init_library_raw_df = init_library_raw_df.head(int(resolved.init_library_max_rows)).copy().reset_index(drop=True)
        init_encoder_ckpt = _resolve_or_train_init_encoder_ckpt(
            args=resolved,
            cfg_path=args.config,
            out_root=out_root,
            init_library_df=init_library_raw_df,
            shared_ligand3d_cache=shared_ligand3d_cache,
            candidate_3d_cfg=candidate_3d_cfg,
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
            source_csv_path=init_library_csv,
            pretrain_split_ratio=pretrain_split_ratio,
            pretrain_split_seed=pretrain_split_seed,
            pretrain_use_existing_split=pretrain_use_existing_split,
        )
        protocol = _build_or_load_downstream_protocol(
            out_root=out_root,
            downstream_raw_df=init_library_raw_df,
            smiles_col=init_library_smiles_col,
            source_csv_path=init_library_csv,
            downstream_pool_ratio=downstream_pool_ratio,
            downstream_pool_seed=downstream_pool_seed,
            scaffold_include_chirality=scaffold_include_chirality,
            butina_cutoff=float(resolved.butina_cutoff),
            fp_radius=int(resolved.fp_radius),
            fp_bits=int(resolved.fp_bits),
            selection_seed=int(resolved.selection_seed),
            init_train_valid_total=int(resolved.init_train_valid_total),
            holdout_test_total=int(resolved.holdout_test_total),
            predictor_valid_ratio=float(predictor_valid_ratio),
            predictor_valid_seed=int(predictor_valid_seed),
            append_mode=False,
        )
        init_df = protocol["protocol_init_df"].copy().reset_index(drop=True)
        global_vocab = _resolve_init_pretrain_ligand_vocab(
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
            init_library_df=init_library_raw_df,
        )
        _log(
            "vae_bo init protocol | "
            f"init_train_valid_total={int((init_df['split_role'].astype(str) == 'init_pool').sum())} "
            f"holdout_test_total={int((init_df['split_role'].astype(str) == 'holdout_test').sum())} "
            "holdout_excluded=1"
        )

    init_stats = _prepare_init_dataset(
        oracle_asset_root,
        dataset_root,
        schema_df=init_df,
        init_df=init_df,
        oracle_cfg=oracle_cfg,
        objective_cfg=objective_cfg,
        run_cfg=run_cfg,
        global_vocab=global_vocab,
    )

    meta = {
        "method": "vae_bo",
        "rounds": int(resolved.rounds),
        "batch_size": int(batch_size),
        "latent_pool_size": int(latent_pool_size),
        "decode_oversample": int(decode_oversample),
        "resample_rounds": int(resample_rounds),
        "decode_attempts": int(decode_attempts),
        "decode_noise_std": float(decode_noise_std),
        "sample_temperature": float(sample_temperature),
        "sample_top_k": int(sample_top_k),
        "latent_encode_batch_size": int(latent_encode_batch_size),
        "latent_score_batch_size": int(latent_score_batch_size),
        "gp_baseline_cap": int(gp_baseline_cap),
        "pretrain_split_ratio": [float(x) for x in pretrain_split_ratio],
        "pretrain_split_seed": int(pretrain_split_seed),
        "downstream_pool_split_ratio": [float(x) for x in downstream_pool_ratio],
        "downstream_pool_split_seed": int(downstream_pool_seed),
        "scaffold_include_chirality": bool(scaffold_include_chirality),
        "butina_cutoff": float(resolved.butina_cutoff),
        "fp_radius": int(resolved.fp_radius),
        "fp_bits": int(resolved.fp_bits),
        "init_train_valid_total": int(resolved.init_train_valid_total),
        "holdout_test_total": int(resolved.holdout_test_total),
        "selection_seed": int(resolved.selection_seed),
        "predictor_valid_ratio": float(predictor_valid_ratio),
        "predictor_valid_seed": int(predictor_valid_seed),
        "gp_mc_samples": int(gp_mc_samples),
        "gp_device": str(device),
        "gpu_total_gib": gpu_total_gib,
        "gpu_free_gib": gpu_free_gib,
        "acquisition": "qLogEHVI",
        "vae_ckpt": str(Path(vae_ckpt).resolve()),
        "init_encoder_ckpt": None if init_encoder_ckpt is None else str(Path(init_encoder_ckpt).resolve()),
        "reuse_init_from": str(Path(args.reuse_init_from).resolve()) if args.reuse_init_from else None,
        "oracle_config": {
            "docking_backend": str(online_oracle_kwargs["docking_backend"]),
            "oracle_exhaustiveness": int(oracle_cfg.get("oracle_exhaustiveness", 32)),
            "oracle_pocket_radius": float(oracle_cfg.get("oracle_pocket_radius", 10.0)),
            "vina_cuda_thread": int(online_oracle_kwargs["vina_cuda_thread"]),
            "vina_cuda_search_depth": int(online_oracle_kwargs["vina_cuda_search_depth"]),
            "vina_cuda_rilc_bfgs": int(online_oracle_kwargs["vina_cuda_rilc_bfgs"]),
            "unidock_service_url": online_oracle_kwargs["unidock_service_url"],
            "unidock_scoring": str(online_oracle_kwargs["unidock_scoring"]),
            "unidock_search_mode": str(online_oracle_kwargs["unidock_search_mode"]),
            "unidock_num_modes": int(online_oracle_kwargs["unidock_num_modes"]),
            "unidock_timeout_sec": int(online_oracle_kwargs["unidock_timeout_sec"]),
        },
    }
    (out_root / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"vae_bo start | out_root={out_root} rounds={resolved.rounds} batch_size={batch_size} latent_pool_size={latent_pool_size} sample_top_k={sample_top_k} decode_attempts={decode_attempts} gp_device={device} score_batch_size={latent_score_batch_size} baseline_cap={gp_baseline_cap} mc_samples={gp_mc_samples} gpu_free_gib={gpu_free_gib} gpu_total_gib={gpu_total_gib}")

    iter_rows: list[dict] = []
    for iter_idx in range(int(resolved.rounds) + 1):
        iter_dir = out_root / f"iter{iter_idx:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        labeled_df = _load_observed_df(dataset_root, dock_valid_max=dock_valid_max)
        hv = compute_hypervolume(
            build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign),
            ref_point=fixed_ref_point,
        )
        acq_ref_point = _dynamic_ref_point_from_labeled(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
        iter_row = {
            "iter": int(iter_idx),
            "labeled_n": int(labeled_df.shape[0]),
            "best_observed_dock": float(labeled_df["dock_score"].min()),
            "hv": float(hv),
            "acq_ref_point_dock": float(acq_ref_point[0]),
            "acq_ref_point_qed": float(acq_ref_point[1]),
            "acq_ref_point_sa": float(acq_ref_point[2]),
        }
        if int(iter_idx) == 0:
            iter_row.update({
                "init_n": int(init_stats.get("init_n", labeled_df.shape[0])),
                "init_valid_dock_n": int(init_stats.get("valid_init_dock_n", labeled_df.shape[0])),
                "init_best_dock": float(init_stats.get("init_best_dock", labeled_df["dock_score"].min())),
                "init_mean_dock": float(init_stats.get("init_mean_dock", pd.to_numeric(labeled_df["dock_score"], errors="coerce").mean())),
            })

        if iter_idx < int(resolved.rounds):
            latent_x = _encode_smiles_to_latent_mu(
                model=vae_model,
                vocab=vae_vocab,
                smiles_list=labeled_df["smiles_canonical"].astype(str).tolist(),
                token_type=token_type,
                device=device,
                batch_size=int(latent_encode_batch_size),
            )
            objective_y = _objective_tensor_from_df(
                labeled_df,
                dock_sign=dock_sign,
                qed_sign=qed_sign,
                sa_sign=sa_sign,
            )

            def _score_latent_pool(z_pool_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                return _fit_gp_and_score_pool(
                    latent_x,
                    objective_y,
                    z_pool_np,
                    ref_point=acq_ref_point,
                    device=device,
                    score_batch_size=int(latent_score_batch_size),
                    baseline_cap=int(gp_baseline_cap),
                    mc_samples=int(gp_mc_samples),
                )

            selected = _select_from_latent_pool(
                vae_model,
                vae_vocab,
                token_type,
                max_len,
                latent_dim,
                latent_pool_size=int(latent_pool_size),
                batch_size=int(batch_size),
                oversample_factor=int(decode_oversample),
                resample_rounds=int(resample_rounds),
                seed=int(resolved.seed),
                iter_idx=int(iter_idx),
                device=device,
                acq_fn=_score_latent_pool,
                temperature=float(sample_temperature),
                top_k=int(sample_top_k),
                decode_attempts=int(decode_attempts),
                decode_noise_std=float(decode_noise_std),
                dock_sign=dock_sign,
                qed_sign=qed_sign,
                sa_sign=sa_sign,
                existing_smiles=set(labeled_df["smiles_canonical"].astype(str).tolist()),
            )
            selected["selected_iter"] = int(iter_idx) + 1
            selected.to_csv(iter_dir / "selected_batch.csv", index=False)

            selected_smiles = selected["smiles_canonical"].astype(str).tolist()
            selected_pred_added = selected["pred_dock_score_mean"].astype(float).tolist()
            added, new_ids, kept_indices = append_candidates_to_smiles_csv(
                str(dataset_root / "smiles.csv"),
                selected_smiles,
                id_prefix=str(oracle_cfg.get("oracle_id_prefix", "GEN")),
                sa_clamp_min=float(objective_cfg.get("sa_clamp_min", -10.0)),
                sa_clamp_max=float(objective_cfg.get("sa_clamp_max", 20.0)),
                split_ratio=None,
                added_iter=iter_idx + 1,
                molecule_origin="generated",
            )
            if kept_indices != list(range(selected.shape[0])):
                selected = selected.iloc[kept_indices].copy().reset_index(drop=True)
                selected_pred_added = [selected_pred_added[i] for i in kept_indices]
            if added != len(new_ids) or added != len(kept_indices):
                raise RuntimeError(f"Append bookkeeping mismatch at iter {iter_idx}: added={added} new_ids={len(new_ids)} kept={len(kept_indices)}")

            oracle_kwargs = _build_oracle_call_kwargs(run_cfg=run_cfg, oracle_cfg=oracle_cfg, evaluate_reference=False, dock_valid_max=dock_valid_max)
            dock_stats = run_oracle_docking(str(dataset_root), new_ids, **oracle_kwargs)
            _fill_extra_metrics_in_csv(dataset_root, objective_cfg, prior_state=None, force=False)
            new_train_n, new_valid_n = _assign_new_rows_random_predictor_split(
                dataset_root,
                new_ids,
                valid_ratio=float(predictor_valid_ratio),
                seed=int(predictor_valid_seed + iter_idx + 1),
            )
            _log(
                f"vae_bo iter {iter_idx}/{resolved.rounds} predictor_split_new_rows | "
                f"train={new_train_n} valid={new_valid_n} holdout_excluded=1"
            )
            _export_selected_batch_with_oracle(dataset_root, selected, new_ids, iter_dir / "selected_batch_with_oracle.csv")

            smiles_all = pd.read_csv(dataset_root / "smiles.csv")
            smiles_all["ligand_id"] = smiles_all["ligand_id"].astype(str)
            new_df = smiles_all.loc[smiles_all["ligand_id"].isin([str(x) for x in new_ids])].copy().reset_index(drop=True)
            valid_mask = []
            for val in new_df.get("dock_score", pd.Series([], dtype=float)).tolist():
                try:
                    num = float(val)
                except Exception:
                    num = None
                valid_mask.append(num is not None and (dock_valid_max is None or num <= dock_valid_max))
            valid_new_df = new_df.loc[np.asarray(valid_mask, dtype=bool)].copy().reset_index(drop=True) if len(valid_mask) else new_df.head(0).copy()

            oracle_eval = evaluate_oracle_accuracy_from_csv(
                str(dataset_root / "smiles.csv"),
                new_ids,
                selected_pred_added,
                dock_valid_max=dock_valid_max,
            )
            if valid_new_df.empty:
                selected_true_dock_mean = np.nan
                selected_true_dock_best = np.nan
                selected_oracle_hv_gain = 0.0
            else:
                dock_vals = pd.to_numeric(valid_new_df["dock_score"], errors="coerce")
                selected_true_dock_mean = float(dock_vals.mean())
                selected_true_dock_best = float(dock_vals.min())
                selected_oracle_hv_gain = float(_batch_hv_gain(labeled_df, valid_new_df, dock_sign, qed_sign, sa_sign, fixed_ref_point))

            iter_row.update({
                "selected_n": int(selected.shape[0]),
                "candidate_pool_n": int(latent_pool_size) * int(resample_rounds),
                "selected_true_dock_mean": selected_true_dock_mean,
                "selected_true_dock_best": selected_true_dock_best,
                "selected_oracle_hv_gain": selected_oracle_hv_gain,
                "oracle_attempted": int(dock_stats.get("attempted", 0)),
                "oracle_docked": int(dock_stats.get("docked", 0)),
                "oracle_failed": int(dock_stats.get("failed", 0)),
                "selected_pred_n": int(oracle_eval.get("matched", 0)),
                "selected_pred_skipped": int(oracle_eval.get("skipped", 0)),
                "selected_pred_invalid_dock": int(oracle_eval.get("invalid_dock", 0)),
                "selected_pred_invalid_pred": int(oracle_eval.get("invalid_pred", 0)),
                "selected_pred_rmse": np.nan,
                "selected_pred_mae": np.nan,
                "selected_pred_spearman": np.nan,
                "selected_pred_kendall": np.nan,
                "utility_weight_dock": float(weights[0]),
                "utility_weight_qed": float(weights[1]),
                "utility_weight_sa": float(weights[2]),
            })

        iter_rows.append(iter_row)
        pd.DataFrame(iter_rows).to_csv(out_root / "iter_metrics.csv", index=False)
        _log(f"vae_bo iter {iter_idx}/{resolved.rounds} done | hv={iter_row['hv']:.4f} best_observed_dock={iter_row['best_observed_dock']:.4f}")


if __name__ == "__main__":
    main()

