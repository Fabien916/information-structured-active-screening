from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated as bt_is_non_dominated

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobo.analytic_hvi_fast import (
    _compute_slopes_batch,
    _piecewise_hvi_eval_rows,
    _precompute_shared_segments,
    _standardize_analytic_inputs_3d,
    nehvi_gaussian_analytic_3d as nehvi_gaussian_analytic_3d_fast,
    qphv_prob_gaussian_analytic_3d as qphv_prob_gaussian_analytic_3d_fast,
)
from mobo.metrics import _PIECEWISE_NUMERIC_TOL, _apply_objective_weights_and_mask, _build_piecewise_hvi_3d, qnehvi_scores_from_samples_nd
from mobo.qpmhi_analytic import (
    _pack_piecewise_family,
    _piecewise_hvi_gaussian_expectation,
    _score_active_family,
    nehvi_gaussian_analytic_3d,
    qphv_prob_gaussian_analytic_3d,
)

try:
    from scipy.stats import kendalltau as scipy_kendalltau
except ImportError:  # pragma: no cover
    scipy_kendalltau = None


DEFAULT_DOCK_SIGN = -1.0
DEFAULT_QED_SIGN = 1.0
DEFAULT_SA_SIGN = -1.0
DEFAULT_WEIGHTS = [1.0, 1.0, 1.0]
TOPK_DIAGNOSTIC_KS = [50, 100, 200, 500, 1000]


def _format_stat(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{float(value):.6g}"


def _subset_benchmark_data(data: dict[str, np.ndarray], candidate_subset: int | None) -> dict[str, np.ndarray]:
    if candidate_subset is None:
        return data
    limit = int(candidate_subset)
    if limit <= 0:
        raise ValueError("candidate_subset must be positive.")
    return {
        "dock_mu": np.asarray(data["dock_mu"], dtype=np.float64)[:limit].copy(),
        "dock_sigma": np.asarray(data["dock_sigma"], dtype=np.float64)[:limit].copy(),
        "exact_obj": np.asarray(data["exact_obj"], dtype=np.float64)[:limit].copy(),
        "y_train": np.asarray(data["y_train"], dtype=np.float64).copy(),
        "ref_point": np.asarray(data["ref_point"], dtype=np.float64).copy(),
    }


def make_synthetic_benchmark_data(N: int, P_size: int, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dock_mu = rng.uniform(-12.0, -6.0, size=N).astype(np.float64)
    dock_sigma = rng.uniform(0.3, 1.5, size=N).astype(np.float64)
    exact_obj = np.column_stack(
        [
            rng.uniform(0.2, 0.9, size=N),
            rng.uniform(1.5, 5.0, size=N),
        ]
    ).astype(np.float64)

    t = np.linspace(0.0, 1.0, num=P_size, dtype=np.float64)
    y_train = np.column_stack(
        [
            -12.0 + 6.0 * t,
            0.25 + 0.6 * np.power(t, 0.9),
            5.0 - 3.2 * np.power(t, 1.1),
        ]
    ).astype(np.float64)
    y_train += rng.normal(loc=0.0, scale=np.asarray([0.05, 0.01, 0.05], dtype=np.float64), size=y_train.shape)
    y_train[:, 0] = np.clip(y_train[:, 0], -12.5, -5.5)
    y_train[:, 1] = np.clip(y_train[:, 1], 0.2, 0.95)
    y_train[:, 2] = np.clip(y_train[:, 2], 1.5, 5.1)

    ref_point = np.asarray(
        [
            float(np.max(y_train[:, 0]) + 0.5),
            float(np.min(y_train[:, 1]) - 0.05),
            float(np.max(y_train[:, 2]) + 0.5),
        ],
        dtype=np.float64,
    )
    return {
        "dock_mu": dock_mu,
        "dock_sigma": dock_sigma,
        "exact_obj": exact_obj,
        "y_train": y_train,
        "ref_point": ref_point,
    }


def load_benchmark_data(path: str) -> dict[str, np.ndarray]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"benchmark input file not found: {file_path}")
    if file_path.suffix.lower() == ".npz":
        raw = np.load(file_path)
        data = {key: raw[key] for key in raw.files}
    elif file_path.suffix.lower() in {".pt", ".pth"}:
        raw = torch.load(file_path, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("torch input must be a dict with benchmark arrays.")
        data = {}
        for key, value in raw.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.detach().cpu().numpy()
            else:
                data[key] = np.asarray(value)
    else:
        raise ValueError("input file must be .npz, .pt, or .pth")

    required = {"dock_mu", "dock_sigma", "exact_obj", "y_train", "ref_point"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"benchmark input missing required keys: {missing}")
    dock_sigma = np.asarray(data["dock_sigma"], dtype=np.float64)
    if not np.isfinite(dock_sigma).all():
        raise ValueError("dock_sigma contains non-finite values.")
    if np.any(dock_sigma < 0.0):
        raise ValueError(f"dock_sigma must be non-negative, got min={float(np.min(dock_sigma))}.")
    return {
        "dock_mu": np.asarray(data["dock_mu"], dtype=np.float64),
        "dock_sigma": dock_sigma,
        "exact_obj": np.asarray(data["exact_obj"], dtype=np.float64),
        "y_train": np.asarray(data["y_train"], dtype=np.float64),
        "ref_point": np.asarray(data["ref_point"], dtype=np.float64),
    }


def convert_to_signed_tensors(
    data: dict[str, np.ndarray],
    dock_sign: float = DEFAULT_DOCK_SIGN,
    qed_sign: float = DEFAULT_QED_SIGN,
    sa_sign: float = DEFAULT_SA_SIGN,
) -> dict[str, Any]:
    dock_mu_raw = torch.as_tensor(data["dock_mu"], dtype=torch.double).view(-1)
    dock_sigma = torch.as_tensor(data["dock_sigma"], dtype=torch.double).view(-1)
    exact_raw = torch.as_tensor(data["exact_obj"], dtype=torch.double)
    y_train_raw = torch.as_tensor(data["y_train"], dtype=torch.double)
    ref_raw = torch.as_tensor(data["ref_point"], dtype=torch.double).view(3)

    signed = {
        "dock_mu": dock_sign * dock_mu_raw,
        "dock_sigma": dock_sigma,
        "exact_obj": torch.stack(
            [
                qed_sign * exact_raw[:, 0],
                sa_sign * exact_raw[:, 1],
            ],
            dim=-1,
        ),
        "y_train": torch.stack(
            [
                dock_sign * y_train_raw[:, 0],
                qed_sign * y_train_raw[:, 1],
                sa_sign * y_train_raw[:, 2],
            ],
            dim=-1,
        ),
        "ref_point": [
            float(dock_sign * ref_raw[0].item()),
            float(qed_sign * ref_raw[1].item()),
            float(sa_sign * ref_raw[2].item()),
        ],
    }
    return signed


def build_mc_samples(
    data: dict[str, np.ndarray],
    sample_count: int,
    seed: int,
    dock_sign: float = DEFAULT_DOCK_SIGN,
    qed_sign: float = DEFAULT_QED_SIGN,
    sa_sign: float = DEFAULT_SA_SIGN,
) -> torch.Tensor:
    dock_mu = torch.as_tensor(data["dock_mu"], dtype=torch.double).view(1, -1)
    dock_sigma = torch.as_tensor(data["dock_sigma"], dtype=torch.double).view(1, -1)
    exact_obj = torch.as_tensor(data["exact_obj"], dtype=torch.double)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    dock_samples = dock_mu + dock_sigma * torch.randn((int(sample_count), dock_mu.size(1)), generator=generator, dtype=torch.double)
    dock_signed = dock_sign * dock_samples
    qed_expanded = (qed_sign * exact_obj[:, 0]).view(1, -1).expand(int(sample_count), -1)
    sa_expanded = (sa_sign * exact_obj[:, 1]).view(1, -1).expand(int(sample_count), -1)
    return torch.stack([dock_signed, qed_expanded, sa_expanded], dim=-1)


def qpmhi_mc_prefix_scores_from_samples(
    samples: torch.Tensor,
    sample_sizes: list[int],
    y_train: torch.Tensor | None,
    weights: list[float] | None,
    ref_point: list[float] | None,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")
    requested_sizes = sorted({int(size) for size in sample_sizes})
    if not requested_sizes:
        raise ValueError("sample_sizes must be non-empty.")
    if requested_sizes[0] <= 0:
        raise ValueError("sample_sizes must be positive.")
    if requested_sizes[-1] > int(samples.size(0)):
        raise ValueError("sample_sizes cannot exceed available MC samples.")

    start_time = time.perf_counter()
    samples, y_train, ref_point = _apply_objective_weights_and_mask(
        samples=samples,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    )
    y_base = samples.mean(0) if y_train is None else y_train
    mean = y_base.mean(0)
    std = y_base.std(0).clamp_min(1e-8)
    y_base_std = (y_base - mean) / std
    s_std = (samples - mean) / std
    if ref_point is not None:
        ref_raw = torch.tensor(ref_point, device=samples.device, dtype=samples.dtype)
        if ref_raw.numel() != y_base_std.size(1):
            raise ValueError("ref_point length must match objective dimension.")
        ref = (ref_raw - mean) / std
    else:
        ref = y_base_std.min(0).values - 1.0

    if y_base_std.numel() == 0:
        front = y_base_std
    else:
        front = y_base_std[bt_is_non_dominated(y_base_std)]
    if front.numel() == 0:
        front = y_base_std

    n_count = int(s_std.size(1))
    counts = torch.zeros(n_count, device=samples.device, dtype=torch.double)
    prefix_scores: dict[int, torch.Tensor] = {}
    prefix_elapsed_s: dict[int, float] = {}
    next_target_idx = 0

    ref_np = ref.detach().cpu().numpy().astype(np.float64)
    front_np = front.detach().cpu().numpy().astype(np.float64) if front.numel() > 0 else np.empty((0, s_std.size(2)), dtype=np.float64)
    dock_samples_np = s_std[:, :, 0].detach().cpu().numpy().astype(np.float64)
    exact_np = s_std[0, :, 1:].detach().cpu().numpy().astype(np.float64)
    rect_mask = (exact_np[:, 0] > float(ref_np[1])) & (exact_np[:, 1] > float(ref_np[2]))
    active_indices = np.flatnonzero(rect_mask)
    kept_indices = np.empty((0,), dtype=np.int64)
    slopes_kept = np.empty((0, 0), dtype=np.float64)
    values_start_kept = np.empty((0, 0), dtype=np.float64)
    starts = np.asarray([float(ref_np[0])], dtype=np.float64)

    if active_indices.size > 0:
        shared = _precompute_shared_segments(front=front_np, ref_point=ref_np)
        starts = np.asarray(shared["starts"], dtype=np.float64)
        slopes_active, _intercepts_active, values_start_active = _compute_slopes_batch(
            front=front_np,
            shared_starts=starts,
            cand_yz=exact_np[active_indices],
            ref_point=ref_np,
            shared_segments=shared["segments"],
        )
        support_mask = np.any(slopes_active > _PIECEWISE_NUMERIC_TOL, axis=1)
        kept_indices = active_indices[support_mask]
        slopes_kept = slopes_active[support_mask]
        values_start_kept = values_start_active[support_mask]

    for sample_idx in range(int(s_std.size(0))):
        best_idx = 0
        if kept_indices.size > 0:
            gains = _piecewise_hvi_eval_rows(
                starts=starts,
                ref_x=float(ref_np[0]),
                slopes=slopes_kept,
                values_start=values_start_kept,
                dock_values=dock_samples_np[sample_idx, kept_indices],
            )
            local_best = int(np.argmax(gains))
            if float(gains[local_best]) > 0.0:
                best_idx = int(kept_indices[local_best])
        counts[best_idx] += 1.0

        completed = sample_idx + 1
        while next_target_idx < len(requested_sizes) and completed == requested_sizes[next_target_idx]:
            target = requested_sizes[next_target_idx]
            prefix_scores[target] = counts.clone() / float(target)
            prefix_elapsed_s[target] = float(time.perf_counter() - start_time)
            next_target_idx += 1

    return prefix_scores, prefix_elapsed_s


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if scipy_kendalltau is not None:
        stat = scipy_kendalltau(a_arr, b_arr, nan_policy="omit").statistic
        return float(0.0 if stat is None or np.isnan(stat) else stat)
    if a_arr.size > 3000:
        raise RuntimeError("scipy is unavailable and fallback Kendall tau is too expensive for N > 3000.")
    concordant = 0
    discordant = 0
    for i in range(a_arr.size - 1):
        da = a_arr[i + 1:] - a_arr[i]
        db = b_arr[i + 1:] - b_arr[i]
        prod = da * db
        concordant += int(np.sum(prod > 0.0))
        discordant += int(np.sum(prod < 0.0))
    denom = concordant + discordant
    return 0.0 if denom == 0 else float((concordant - discordant) / denom)


def topk_overlap(reference_scores: np.ndarray, other_scores: np.ndarray, k: int = 50) -> tuple[int, int]:
    ref = np.asarray(reference_scores, dtype=np.float64)
    other = np.asarray(other_scores, dtype=np.float64)
    if ref.shape[0] == 0:
        return 0, 0
    top_k = min(int(k), int(ref.shape[0]))
    ref_idx = np.argsort(-ref, kind="mergesort")[:top_k]
    other_idx = np.argsort(-other, kind="mergesort")[:top_k]
    overlap = len(set(ref_idx.tolist()).intersection(other_idx.tolist()))
    return overlap, top_k


def score_distribution(scores: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    positive = finite[finite > 0.0]
    out: dict[str, float | int | None] = {
        "n_total": int(arr.shape[0]),
        "n_finite": int(finite.shape[0]),
        "n_positive": int(positive.shape[0]),
        "n_zero": int(finite.shape[0] - positive.shape[0]),
        "p50_positive": None,
        "p90_positive": None,
        "p95_positive": None,
        "p99_positive": None,
        "max": None,
    }
    if positive.size > 0:
        out["p50_positive"] = float(np.percentile(positive, 50))
        out["p90_positive"] = float(np.percentile(positive, 90))
        out["p95_positive"] = float(np.percentile(positive, 95))
        out["p99_positive"] = float(np.percentile(positive, 99))
        out["max"] = float(np.max(positive))
    elif finite.size > 0:
        out["max"] = float(np.max(finite))
    return out


def topk_union_diagnostics(reference_scores: np.ndarray, other_scores: np.ndarray, ks: list[int] | None = None) -> dict[str, dict[str, float | int]]:
    ref = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
    other = np.asarray(other_scores, dtype=np.float64).reshape(-1)
    topk_values = TOPK_DIAGNOSTIC_KS if ks is None else [int(k) for k in ks]
    out: dict[str, dict[str, float | int]] = {}
    for raw_k in topk_values:
        if ref.shape[0] == 0:
            out[str(raw_k)] = {"requested_k": int(raw_k), "k": 0, "overlap": 0, "union_size": 0, "tau_on_union": 0.0}
            continue
        k = min(int(raw_k), int(ref.shape[0]))
        top_ref = np.argsort(-ref, kind="mergesort")[:k]
        top_other = np.argsort(-other, kind="mergesort")[:k]
        union_idx = np.union1d(top_ref, top_other)
        overlap = int(np.intersect1d(top_ref, top_other).shape[0])
        tau_union = 0.0 if union_idx.size <= 1 else kendall_tau(ref[union_idx], other[union_idx])
        out[str(raw_k)] = {
            "requested_k": int(raw_k),
            "k": int(k),
            "overlap": overlap,
            "union_size": int(union_idx.size),
            "tau_on_union": float(tau_union),
        }
    return out


def dataset_summary(data: dict[str, np.ndarray]) -> dict[str, int]:
    signed = convert_to_signed_tensors(data)
    y_train = signed["y_train"]
    pf_size = int(bt_is_non_dominated(y_train).sum().item()) if y_train.numel() > 0 else 0
    return {
        "N": int(np.asarray(data["dock_mu"]).shape[0]),
        "y_train_size": int(np.asarray(data["y_train"]).shape[0]),
        "pf_size": pf_size,
    }


def build_result_row(
    method: str,
    category: str,
    reference_scores: np.ndarray,
    scores: np.ndarray,
    prefilter_n: str,
    prefilter_time_s: float | None,
    score_time_s: float | None,
    total_time_s: float | None,
) -> dict[str, Any]:
    ref = np.asarray(reference_scores, dtype=np.float64)
    cur = np.asarray(scores, dtype=np.float64)
    tau_all = float(kendall_tau(ref, cur))
    overlap_50, top50_k = topk_overlap(ref, cur, k=50)
    dist = score_distribution(cur)
    topk_diag = topk_union_diagnostics(ref, cur)
    return {
        "method": method,
        "category": category,
        "tau_vs_reference": tau_all,
        "tau_all": tau_all,
        "tau_top50": float(topk_diag["50"]["tau_on_union"]),
        "tau_top200": float(topk_diag["200"]["tau_on_union"]),
        "top50_overlap": f"{overlap_50}/{top50_k}",
        "prefilter_n": prefilter_n,
        "prefilter_time_s": prefilter_time_s,
        "score_time_s": score_time_s,
        "total_time_s": total_time_s,
        "score_distribution": dist,
        "topk_diagnostics": topk_diag,
        "score_p50_positive": dist["p50_positive"],
        "score_p99_positive": dist["p99_positive"],
        "score_max": dist["max"],
    }


def _profile_original_nehvi(
    dock_mu: torch.Tensor,
    dock_sigma: torch.Tensor,
    exact_obj: torch.Tensor,
    y_train: torch.Tensor,
    weights: list[float],
    ref_point: list[float],
) -> tuple[torch.Tensor, dict[str, float | int]]:
    prepared = _standardize_analytic_inputs_3d(
        dock_mu=dock_mu,
        dock_sigma=dock_sigma,
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    )
    ref_np = prepared["ref_np"]
    front_np = prepared["front_np"]
    mu_np = prepared["mu_np"]
    sigma_np = prepared["sigma_np"]
    exact_np = prepared["exact_np"]
    scores = np.zeros(mu_np.shape[0], dtype=np.float64)
    metadata: dict[str, float | int] = {
        "input_n": int(mu_np.shape[0]),
        "prefilter_ref_rect_zero_n": 0,
        "prefilter_no_support_n": 0,
        "scored_n": 0,
        "prefilter_time_s": 0.0,
        "score_time_s": 0.0,
        "total_time_s": 0.0,
    }

    total_start = time.perf_counter()
    rect_mask = (exact_np[:, 0] > float(ref_np[1])) & (exact_np[:, 1] > float(ref_np[2]))
    metadata["prefilter_ref_rect_zero_n"] = int((~rect_mask).sum())
    pieces: dict[int, dict[str, np.ndarray | float]] = {}
    active_indices: list[int] = []
    prefilter_start = time.perf_counter()
    for idx in np.flatnonzero(rect_mask):
        piece = _build_piecewise_hvi_3d(front_np, cand_y=float(exact_np[idx, 0]), cand_z=float(exact_np[idx, 1]), ref_point=ref_np)
        pieces[int(idx)] = piece
        slopes = np.asarray(piece["slopes"], dtype=np.float64)
        if np.any(slopes > _PIECEWISE_NUMERIC_TOL):
            active_indices.append(int(idx))
    prefilter_end = time.perf_counter()
    metadata["prefilter_no_support_n"] = int(rect_mask.sum() - len(active_indices))
    metadata["scored_n"] = int(len(active_indices))
    metadata["prefilter_time_s"] = float(prefilter_end - prefilter_start)

    score_start = time.perf_counter()
    for idx in active_indices:
        scores[idx] = _piecewise_hvi_gaussian_expectation(
            piece=pieces[idx],
            mu=float(mu_np[idx]),
            sigma=float(sigma_np[idx]),
        )
    score_end = time.perf_counter()
    metadata["score_time_s"] = float(score_end - score_start)
    metadata["total_time_s"] = float(score_end - total_start)
    return torch.as_tensor(scores, dtype=torch.double), metadata


def _profile_original_qpmhi(
    dock_mu: torch.Tensor,
    dock_sigma: torch.Tensor,
    exact_obj: torch.Tensor,
    y_train: torch.Tensor,
    weights: list[float],
    ref_point: list[float],
    quadrature_points: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    prepared = _standardize_analytic_inputs_3d(
        dock_mu=dock_mu,
        dock_sigma=dock_sigma,
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    )
    ref_np = prepared["ref_np"]
    front_np = prepared["front_np"]
    mu_np = prepared["mu_np"]
    sigma_np = prepared["sigma_np"]
    exact_np = prepared["exact_np"]
    n_count = int(mu_np.shape[0])
    scores = np.zeros(n_count, dtype=np.float64)
    metadata: dict[str, float | int] = {
        "input_n": n_count,
        "prefilter_ref_rect_zero_n": 0,
        "prefilter_no_support_n": 0,
        "prefilter_zero_quadrature_gain_n": 0,
        "scored_n": 0,
        "parallel_workers": 1,
        "parallel_chunks": 0,
        "parallel_thread_fallback": 0,
        "prefilter_time_s": 0.0,
        "score_time_s": 0.0,
        "total_time_s": 0.0,
    }

    total_start = time.perf_counter()
    rect_mask = (exact_np[:, 0] > float(ref_np[1])) & (exact_np[:, 1] > float(ref_np[2]))
    metadata["prefilter_ref_rect_zero_n"] = int((~rect_mask).sum())
    pieces: list[dict[str, np.ndarray | float] | None] = [None] * n_count
    active_indices: list[int] = []
    prefilter_start = time.perf_counter()
    for idx in np.flatnonzero(rect_mask):
        piece = _build_piecewise_hvi_3d(front_np, cand_y=float(exact_np[idx, 0]), cand_z=float(exact_np[idx, 1]), ref_point=ref_np)
        pieces[int(idx)] = piece
        if np.any(np.asarray(piece["slopes"], dtype=np.float64) > _PIECEWISE_NUMERIC_TOL):
            active_indices.append(int(idx))
    prefilter_end = time.perf_counter()
    metadata["prefilter_no_support_n"] = int(rect_mask.sum() - len(active_indices))
    metadata["scored_n"] = int(len(active_indices))
    metadata["prefilter_time_s"] = float(prefilter_end - prefilter_start)

    score_start = time.perf_counter()
    if active_indices:
        active_idx_arr = np.asarray(active_indices, dtype=np.int64)
        active_mu = mu_np[active_idx_arr]
        active_sigma = sigma_np[active_idx_arr]
        active_pieces = [pieces[int(idx)] for idx in active_idx_arr.tolist()]
        packed = _pack_piecewise_family(active_pieces, mu_values=active_mu)
        nodes, gh_weights = np.polynomial.hermite.hermgauss(int(quadrature_points))
        nodes = nodes.astype(np.float64)
        gh_weights = gh_weights.astype(np.float64) / math.sqrt(math.pi)
        local_scores, zero_gain_count, workers_used, thread_fallback = _score_active_family(
            active_pieces=active_pieces,
            packed=packed,
            active_mu=active_mu,
            active_sigma=active_sigma,
            nodes=nodes,
            gh_weights=gh_weights,
            tol=1e-12,
            parallel_workers=1,
        )
        metadata["prefilter_zero_quadrature_gain_n"] = int(zero_gain_count)
        metadata["parallel_workers"] = int(workers_used)
        metadata["parallel_chunks"] = 1
        metadata["parallel_thread_fallback"] = int(thread_fallback)
        scores[active_idx_arr] = local_scores
        scores = np.clip(scores, 0.0, None)
        total = float(scores.sum())
        if total > 0.0:
            scores /= total
    score_end = time.perf_counter()
    metadata["score_time_s"] = float(score_end - score_start)
    metadata["total_time_s"] = float(score_end - total_start)
    return torch.as_tensor(scores, dtype=torch.double), metadata


def _bench_nehvi_methods(
    data: dict[str, np.ndarray],
    mc_sample_sizes: list[int],
    seed: int,
    validate_fast: bool,
) -> list[dict[str, Any]]:
    signed = convert_to_signed_tensors(data)
    ref_scores, ref_meta = _profile_original_nehvi(
        dock_mu=signed["dock_mu"],
        dock_sigma=signed["dock_sigma"],
        exact_obj=signed["exact_obj"],
        y_train=signed["y_train"],
        weights=list(DEFAULT_WEIGHTS),
        ref_point=list(signed["ref_point"]),
    )
    ref_np = ref_scores.detach().cpu().numpy()
    results: list[dict[str, Any]] = [
        build_result_row(
            method="analytic_nehvi",
            category="nehvi",
            reference_scores=ref_np,
            scores=ref_np,
            prefilter_n=f"{ref_meta['input_n']} -> {ref_meta['scored_n']}",
            prefilter_time_s=float(ref_meta["prefilter_time_s"]),
            score_time_s=float(ref_meta["score_time_s"]),
            total_time_s=float(ref_meta["total_time_s"]),
        )
    ]

    start_fast = time.perf_counter()
    fast_scores, fast_meta = nehvi_gaussian_analytic_3d_fast(
        dock_mu=signed["dock_mu"],
        dock_sigma=signed["dock_sigma"],
        exact_obj=signed["exact_obj"],
        y_train=signed["y_train"],
        weights=list(DEFAULT_WEIGHTS),
        ref_point=list(signed["ref_point"]),
        return_metadata=True,
        validate=validate_fast,
    )
    fast_total = time.perf_counter() - start_fast
    fast_np = fast_scores.detach().cpu().numpy()
    results.append(
        build_result_row(
            method="analytic_nehvi_fast",
            category="nehvi",
            reference_scores=ref_np,
            scores=fast_np,
            prefilter_n=f"{fast_meta['input_n']} -> {fast_meta['scored_n']}",
            prefilter_time_s=float(fast_meta["prefilter_time_s"]),
            score_time_s=float(fast_meta["score_time_s"]),
            total_time_s=float(fast_total),
        )
    )

    full_samples = build_mc_samples(data=data, sample_count=max(mc_sample_sizes), seed=seed)
    for sample_count in mc_sample_sizes:
        start_mc = time.perf_counter()
        mc_scores = qnehvi_scores_from_samples_nd(
            samples=full_samples[:sample_count],
            y_train=signed["y_train"],
            weights=list(DEFAULT_WEIGHTS),
            ref_point=list(signed["ref_point"]),
            show_progress=False,
        )
        total_mc = time.perf_counter() - start_mc
        mc_np = mc_scores.detach().cpu().numpy()
        results.append(
            build_result_row(
                method=f"mc_nehvi_S{sample_count}",
                category="nehvi",
                reference_scores=ref_np,
                scores=mc_np,
                prefilter_n="N/A",
                prefilter_time_s=None,
                score_time_s=float(total_mc),
                total_time_s=float(total_mc),
            )
        )
    return results


def _bench_qpmhi_methods(
    data: dict[str, np.ndarray],
    mc_sample_sizes: list[int],
    seed: int,
    quadrature_points: int,
    validate_fast: bool,
) -> list[dict[str, Any]]:
    signed = convert_to_signed_tensors(data)
    ref_scores, ref_meta = _profile_original_qpmhi(
        dock_mu=signed["dock_mu"],
        dock_sigma=signed["dock_sigma"],
        exact_obj=signed["exact_obj"],
        y_train=signed["y_train"],
        weights=list(DEFAULT_WEIGHTS),
        ref_point=list(signed["ref_point"]),
        quadrature_points=quadrature_points,
    )
    ref_np = ref_scores.detach().cpu().numpy()
    results: list[dict[str, Any]] = [
        build_result_row(
            method="analytic_qpmhi",
            category="qpmhi",
            reference_scores=ref_np,
            scores=ref_np,
            prefilter_n=f"{ref_meta['input_n']} -> {ref_meta['scored_n']}",
            prefilter_time_s=float(ref_meta["prefilter_time_s"]),
            score_time_s=float(ref_meta["score_time_s"]),
            total_time_s=float(ref_meta["total_time_s"]),
        )
    ]

    start_fast = time.perf_counter()
    fast_scores, fast_meta = qphv_prob_gaussian_analytic_3d_fast(
        dock_mu=signed["dock_mu"],
        dock_sigma=signed["dock_sigma"],
        exact_obj=signed["exact_obj"],
        y_train=signed["y_train"],
        weights=list(DEFAULT_WEIGHTS),
        ref_point=list(signed["ref_point"]),
        quadrature_points=quadrature_points,
        return_metadata=True,
        validate=validate_fast,
    )
    fast_total = time.perf_counter() - start_fast
    fast_np = fast_scores.detach().cpu().numpy()
    results.append(
        build_result_row(
            method="analytic_qpmhi_fast",
            category="qpmhi",
            reference_scores=ref_np,
            scores=fast_np,
            prefilter_n=f"{fast_meta['input_n']} -> {fast_meta['scored_n']}",
            prefilter_time_s=float(fast_meta["prefilter_time_s"]),
            score_time_s=float(fast_meta["score_time_s"]),
            total_time_s=float(fast_total),
        )
    )

    full_samples = build_mc_samples(data=data, sample_count=max(mc_sample_sizes), seed=seed + 1000)
    prefix_scores, prefix_elapsed_s = qpmhi_mc_prefix_scores_from_samples(
        samples=full_samples,
        sample_sizes=mc_sample_sizes,
        y_train=signed["y_train"],
        weights=list(DEFAULT_WEIGHTS),
        ref_point=list(signed["ref_point"]),
    )
    mc_sorted_sizes = sorted(int(size) for size in mc_sample_sizes)
    for sample_count in mc_sorted_sizes:
        mc_scores = prefix_scores[int(sample_count)]
        cumulative_elapsed = float(prefix_elapsed_s[int(sample_count)])
        mc_np = mc_scores.detach().cpu().numpy()
        results.append(
            build_result_row(
                method=f"mc_qpmhi_S{sample_count}",
                category="qpmhi",
                reference_scores=ref_np,
                scores=mc_np,
                prefilter_n="N/A",
                prefilter_time_s=None,
                score_time_s=float(cumulative_elapsed),
                total_time_s=float(cumulative_elapsed),
            )
        )
    return results


def print_results_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    print(
        f"{'Method':<22} {'Prefilter(s)':>12} {'Score(s)':>10} {'Total(s)':>10} "
        f"{'tau_all':>8} {'tau_top50':>10} {'tau_top200':>11} {'Top50':>8} "
        f"{'Score_p50':>11} {'Score_p99':>11} {'Score_max':>11}"
    )
    for row in rows:
        prefilter = "N/A" if row["prefilter_time_s"] is None else f"{row['prefilter_time_s']:.3f}"
        score = "N/A" if row["score_time_s"] is None else f"{row['score_time_s']:.3f}"
        total = "N/A" if row["total_time_s"] is None else f"{row['total_time_s']:.3f}"
        tau_all = "N/A" if row["tau_all"] is None else f"{row['tau_all']:.3f}"
        tau_top50 = "N/A" if row["tau_top50"] is None else f"{row['tau_top50']:.3f}"
        tau_top200 = "N/A" if row["tau_top200"] is None else f"{row['tau_top200']:.3f}"
        print(
            f"{row['method']:<22} {prefilter:>12} {score:>10} {total:>10} "
            f"{tau_all:>8} {tau_top50:>10} {tau_top200:>11} {row['top50_overlap']:>8} "
            f"{_format_stat(row['score_p50_positive']):>11} {_format_stat(row['score_p99_positive']):>11} {_format_stat(row['score_max']):>11}"
        )
    print("")


def print_score_distribution_diagnostics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        dist = row["score_distribution"]
        print(f"=== Score distribution ({row['method']}) ===")
        print(
            f"N_zero: {dist['n_zero']}  N_positive: {dist['n_positive']}  "
            f"N_total: {dist['n_total']}"
        )
        print(
            "Percentiles of positive scores: "
            f"p50={_format_stat(dist['p50_positive'])} "
            f"p90={_format_stat(dist['p90_positive'])} "
            f"p95={_format_stat(dist['p95_positive'])} "
            f"p99={_format_stat(dist['p99_positive'])} "
            f"max={_format_stat(dist['max'])}"
        )
        print("")


def print_rank_diagnostics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(f"=== Rank diagnostics ({row['method']}) ===")
        for raw_k in TOPK_DIAGNOSTIC_KS:
            diag = row["topk_diagnostics"][str(raw_k)]
            label = f"Top-{diag['requested_k']}"
            if int(diag["requested_k"]) != int(diag["k"]):
                label = f"{label} (effective {diag['k']})"
            print(
                f"{label}: overlap={diag['overlap']}/{diag['k']}, "
                f"tau_on_union={float(diag['tau_on_union']):.3f}, union={diag['union_size']}"
            )
        print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark analytic HVI against Monte Carlo baselines.")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic benchmark mode.")
    parser.add_argument("--quick", action="store_true", help="Run a single lightweight configuration.")
    parser.add_argument("--input-npz", type=str, default=None, help="Load benchmark arrays from a saved .npz/.pt/.pth file.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="analysis/analytic_hvi_benchmark_results.json",
        help="Where to save the benchmark JSON summary.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--include-qpmhi", action="store_true", help="Also benchmark qPMHI analytic and MC variants.")
    parser.add_argument("--qpmhi-only", action="store_true", help="Only benchmark qPMHI methods and skip NEHVI.")
    parser.add_argument("--validate-fast", action="store_true", help="Validate fast analytic outputs against the original implementations.")
    parser.add_argument("--quadrature-points", type=int, default=16, help="Gauss-Hermite points for qPMHI.")
    parser.add_argument("--mc-sizes", type=int, nargs="+", default=None, help="Override MC sample sizes, e.g. --mc-sizes 64 128 256.")
    parser.add_argument("--n-values", "--N", dest="n_values", type=int, nargs="+", default=None, help="Override synthetic candidate counts.")
    parser.add_argument("--p-values", "--P-sizes", dest="p_values", type=int, nargs="+", default=None, help="Override synthetic Pareto front sizes.")
    parser.add_argument("--candidate-subset", type=int, default=None, help="Only benchmark the first N candidates from the provided candidate pool.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic and args.input_npz is not None:
        raise ValueError("Use either --synthetic or --input-npz, not both.")
    run_nehvi = not args.qpmhi_only
    run_qpmhi = bool(args.include_qpmhi or args.qpmhi_only)
    if args.quick:
        n_values = [1000]
        p_values = [30]
        mc_sample_sizes = [64, 128]
    else:
        n_values = [500, 1000, 2000, 5000]
        p_values = [10, 30, 50, 100]
        mc_sample_sizes = [64, 128, 256, 512, 1024]
    if args.n_values:
        n_values = [int(x) for x in args.n_values]
    if args.p_values:
        p_values = [int(x) for x in args.p_values]
    if args.mc_sizes:
        mc_sample_sizes = [int(x) for x in args.mc_sizes]

    all_results: list[dict[str, Any]] = []
    if args.input_npz is not None and not args.synthetic:
        data = _subset_benchmark_data(load_benchmark_data(args.input_npz), candidate_subset=args.candidate_subset)
        summary = dataset_summary(data)
        cfg_title = (
            f"=== Loaded benchmark data: N={summary['N']}, "
            f"|y_train|={summary['y_train_size']}, |PF|={summary['pf_size']} ==="
        )
        rows: list[dict[str, Any]] = []
        if run_nehvi:
            nehvi_rows = _bench_nehvi_methods(data=data, mc_sample_sizes=mc_sample_sizes, seed=args.seed, validate_fast=args.validate_fast)
            print_results_table(cfg_title, nehvi_rows)
            print_score_distribution_diagnostics(nehvi_rows)
            print_rank_diagnostics(nehvi_rows)
            rows.extend(nehvi_rows)
        if run_qpmhi:
            qpmhi_rows = _bench_qpmhi_methods(
                data=data,
                mc_sample_sizes=mc_sample_sizes,
                seed=args.seed,
                quadrature_points=args.quadrature_points,
                validate_fast=args.validate_fast,
            )
            print_results_table(cfg_title if not run_nehvi else "=== qPMHI ===", qpmhi_rows)
            print_score_distribution_diagnostics(qpmhi_rows)
            print_rank_diagnostics(qpmhi_rows)
            rows.extend(qpmhi_rows)
        all_results.append(
            {
                "config": {
                    "source": str(args.input_npz),
                    "N": summary["N"],
                    "y_train_size": summary["y_train_size"],
                    "pf_size": summary["pf_size"],
                    "mc_sample_sizes": list(mc_sample_sizes),
                    "candidate_subset": None if args.candidate_subset is None else int(args.candidate_subset),
                    "run_nehvi": bool(run_nehvi),
                    "run_qpmhi": bool(run_qpmhi),
                },
                "rows": rows,
            }
        )
    else:
        cfg_index = 0
        for n_value in n_values:
            for p_value in p_values:
                cfg_seed = int(args.seed + cfg_index * 17)
                data = _subset_benchmark_data(
                    make_synthetic_benchmark_data(N=n_value, P_size=p_value, seed=cfg_seed),
                    candidate_subset=args.candidate_subset,
                )
                summary = dataset_summary(data)
                title = (
                    f"=== Synthetic benchmark: N={summary['N']}, "
                    f"|y_train|={summary['y_train_size']}, |PF|={summary['pf_size']} ==="
                )
                rows: list[dict[str, Any]] = []
                if run_nehvi:
                    nehvi_rows = _bench_nehvi_methods(data=data, mc_sample_sizes=mc_sample_sizes, seed=cfg_seed, validate_fast=args.validate_fast)
                    print_results_table(title, nehvi_rows)
                    print_score_distribution_diagnostics(nehvi_rows)
                    print_rank_diagnostics(nehvi_rows)
                    rows.extend(nehvi_rows)
                config_entry: dict[str, Any] = {
                    "config": {
                        "source": "synthetic",
                        "seed": cfg_seed,
                        "requested_N": int(n_value),
                        "requested_P_size": int(p_value),
                        "N": summary["N"],
                        "y_train_size": summary["y_train_size"],
                        "pf_size": summary["pf_size"],
                        "mc_sample_sizes": list(mc_sample_sizes),
                        "candidate_subset": None if args.candidate_subset is None else int(args.candidate_subset),
                        "run_nehvi": bool(run_nehvi),
                        "run_qpmhi": bool(run_qpmhi),
                    },
                    "rows": rows,
                }
                if run_qpmhi:
                    qpmhi_rows = _bench_qpmhi_methods(
                        data=data,
                        mc_sample_sizes=mc_sample_sizes,
                        seed=cfg_seed,
                        quadrature_points=args.quadrature_points,
                        validate_fast=args.validate_fast,
                    )
                    print_results_table(title if not run_nehvi else "=== qPMHI ===", qpmhi_rows)
                    print_score_distribution_diagnostics(qpmhi_rows)
                    print_rank_diagnostics(qpmhi_rows)
                    config_entry["rows"].extend(qpmhi_rows)
                all_results.append(config_entry)
                cfg_index += 1

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
    print(f"Saved benchmark JSON to {out_path}")


if __name__ == "__main__":
    main()
