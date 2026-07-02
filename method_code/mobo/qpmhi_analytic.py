from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated as bt_is_non_dominated

from mobo.metrics import _PIECEWISE_NUMERIC_TOL, _build_piecewise_hvi_3d, _piecewise_hvi_eval


_QPMHI_WORKER_STATE: dict[str, object] = {}


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return np.exp(-0.5 * np.square(x)) / math.sqrt(2.0 * math.pi)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=torch.double)
    out = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return out.detach().cpu().numpy()


def _piecewise_hvi_gaussian_expectation(piece: dict[str, np.ndarray | float], mu: float, sigma: float) -> float:
    mu_val = float(mu)
    sigma_val = float(sigma)
    if sigma_val < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma_val}.")
    if sigma_val <= 1e-12:
        return float(_piecewise_hvi_eval(piece, np.asarray([mu_val], dtype=np.float64))[0])

    starts = np.asarray(piece["starts"], dtype=np.float64)
    ends = np.asarray(piece["ends"], dtype=np.float64)
    slopes = np.asarray(piece["slopes"], dtype=np.float64)
    values_start = np.asarray(piece["values_start"], dtype=np.float64)
    total = 0.0
    for idx in range(int(starts.size)):
        slope = float(slopes[idx])
        start = float(starts[idx])
        end = float(ends[idx])
        value_start = float(values_start[idx])
        intercept = value_start - slope * start
        alpha = (start - mu_val) / sigma_val
        if math.isinf(end):
            prob = 1.0 - float(_normal_cdf(np.asarray([alpha], dtype=np.float64))[0])
            x_interval_mean = mu_val * prob + sigma_val * float(_normal_pdf(np.asarray([alpha], dtype=np.float64))[0])
        else:
            beta = (end - mu_val) / sigma_val
            cdf_vals = _normal_cdf(np.asarray([alpha, beta], dtype=np.float64))
            pdf_vals = _normal_pdf(np.asarray([alpha, beta], dtype=np.float64))
            prob = float(cdf_vals[1] - cdf_vals[0])
            x_interval_mean = mu_val * prob + sigma_val * float(pdf_vals[0] - pdf_vals[1])
        if prob <= 0.0:
            continue
        total += slope * x_interval_mean + intercept * prob
    return max(total, 0.0)


def _pack_piecewise_family(
    pieces: Sequence[dict[str, np.ndarray | float]],
    mu_values: np.ndarray,
) -> dict[str, np.ndarray]:
    mu_arr = np.asarray(mu_values, dtype=np.float64)
    if len(pieces) != int(mu_arr.shape[0]):
        raise ValueError("pieces and mu_values must have the same length.")
    if not pieces:
        raise ValueError("pieces must not be empty.")
    n_count = len(pieces)
    max_segments = max(int(np.asarray(piece["starts"], dtype=np.float64).size) for piece in pieces)
    starts = np.zeros((n_count, max_segments), dtype=np.float64)
    slopes = np.zeros((n_count, max_segments), dtype=np.float64)
    values_start = np.zeros((n_count, max_segments), dtype=np.float64)
    values_end = np.full((n_count, max_segments), -np.inf, dtype=np.float64)
    mu_gain = np.zeros(n_count, dtype=np.float64)
    for idx, piece in enumerate(pieces):
        piece_starts = np.asarray(piece["starts"], dtype=np.float64)
        length = int(piece_starts.size)
        starts[idx, :length] = piece_starts
        slopes[idx, :length] = np.asarray(piece["slopes"], dtype=np.float64)
        values_start[idx, :length] = np.asarray(piece["values_start"], dtype=np.float64)
        values_end[idx, :length] = np.asarray(piece["values_end"], dtype=np.float64)
        mu_gain[idx] = float(_piecewise_hvi_eval(piece, np.asarray([float(mu_arr[idx])], dtype=np.float64))[0])
    return {
        "starts": starts,
        "slopes": slopes,
        "values_start": values_start,
        "values_end": values_end,
        "mu_gain": mu_gain,
    }


def _piecewise_gain_cdf_block(
    packed: dict[str, np.ndarray],
    gains_block: np.ndarray,
    mu_block: np.ndarray,
    sigma_block: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    sigma_arr = np.asarray(sigma_block, dtype=np.float64)
    if np.any(sigma_arr < 0.0):
        min_sigma = float(np.min(sigma_arr))
        raise ValueError(f"sigma_block must be non-negative, got min={min_sigma}.")
    out = np.empty_like(gains_block, dtype=np.float64)
    det_mask = sigma_arr <= 1e-12
    row_slice = slice(int(start_idx), int(start_idx) + gains_block.shape[0])
    if np.any(det_mask):
        det_vals = packed["mu_gain"][row_slice][det_mask][:, None]
        out[det_mask] = (gains_block[det_mask] >= det_vals).astype(np.float64)
    if np.any(~det_mask):
        starts = packed["starts"][row_slice][~det_mask]
        slopes = packed["slopes"][row_slice][~det_mask]
        values_start = packed["values_start"][row_slice][~det_mask]
        values_end = packed["values_end"][row_slice][~det_mask]
        gains_pos = gains_block[~det_mask]
        inv = np.full(gains_pos.shape, -np.inf, dtype=np.float64)
        unresolved = gains_pos >= 0.0
        for seg_idx in range(slopes.shape[1]):
            slope_col = slopes[:, seg_idx][:, None]
            active = slope_col > _PIECEWISE_NUMERIC_TOL
            if not np.any(active):
                continue
            upper = values_end[:, seg_idx][:, None]
            mask = unresolved & active & (gains_pos < upper)
            if np.any(mask):
                safe_slope = np.where(active, slope_col, 1.0)
                candidate = starts[:, seg_idx][:, None] + (gains_pos - values_start[:, seg_idx][:, None]) / safe_slope
                inv = np.where(mask, candidate, inv)
                unresolved &= ~mask
        inv[unresolved] = np.inf
        inv_t = torch.as_tensor(inv, dtype=torch.double)
        mu_t = torch.as_tensor(np.asarray(mu_block, dtype=np.float64)[~det_mask][:, None], dtype=torch.double)
        sigma_t = torch.as_tensor(sigma_arr[~det_mask][:, None], dtype=torch.double)
        cdf = 0.5 * (1.0 + torch.erf((inv_t - mu_t) / (sigma_t * math.sqrt(2.0))))
        out[~det_mask] = cdf.detach().cpu().numpy()
    return out


def _piecewise_gain_product_packed(
    packed: dict[str, np.ndarray],
    gains: np.ndarray,
    mu_values: np.ndarray,
    sigma_values: np.ndarray,
    self_index: int,
    tol: float,
    block_size: int = 512,
) -> np.ndarray:
    del tol
    base_gains = np.asarray(gains, dtype=np.float64).reshape(1, -1)
    mu_arr = np.asarray(mu_values, dtype=np.float64)
    sigma_arr = np.asarray(sigma_values, dtype=np.float64)
    n_count = int(mu_arr.shape[0])
    integrand = np.ones(base_gains.shape[1], dtype=np.float64)
    for row_start in range(0, n_count, int(block_size)):
        row_end = min(row_start + int(block_size), n_count)
        gains_block = np.broadcast_to(base_gains, (row_end - row_start, base_gains.shape[1])).copy()
        cdf_block = _piecewise_gain_cdf_block(
            packed=packed,
            gains_block=gains_block,
            mu_block=mu_arr[row_start:row_end],
            sigma_block=sigma_arr[row_start:row_end],
            start_idx=row_start,
        )
        if row_start <= self_index < row_end:
            cdf_block[self_index - row_start, :] = 1.0
        integrand *= np.prod(cdf_block, axis=0, dtype=np.float64)
        if not np.any(integrand > 0.0):
            break
    return integrand


def _resolve_parallel_workers(parallel_workers: int | None, active_n: int) -> int:
    if active_n <= 1:
        return 1
    if parallel_workers is None:
        env_workers = os.environ.get("QPMHI_PARALLEL_WORKERS", "").strip()
        if env_workers:
            parallel_workers = int(env_workers)
        else:
            cpu_count = os.cpu_count() or 1
            parallel_workers = min(8, max(1, cpu_count // 2))
    workers = int(parallel_workers)
    if workers < 1:
        raise ValueError("parallel_workers must be positive.")
    env_min_active = os.environ.get("QPMHI_PARALLEL_MIN_ACTIVE", "").strip()
    min_active = int(env_min_active) if env_min_active else 512
    if active_n < int(min_active):
        return 1
    return min(workers, active_n)


def _chunk_ranges(n_count: int, workers: int) -> list[tuple[int, int]]:
    if n_count <= 0:
        return []
    chunk_count = min(n_count, max(workers, workers * 4))
    chunk_size = int(math.ceil(n_count / float(chunk_count)))
    out: list[tuple[int, int]] = []
    for start in range(0, n_count, chunk_size):
        out.append((start, min(start + chunk_size, n_count)))
    return out


def _score_active_chunk_local(
    active_pieces: Sequence[dict[str, np.ndarray | float]],
    packed: dict[str, np.ndarray],
    active_mu: np.ndarray,
    active_sigma: np.ndarray,
    nodes: np.ndarray,
    gh_weights: np.ndarray,
    tol: float,
    chunk_start: int,
    chunk_end: int,
) -> tuple[int, np.ndarray, int]:
    chunk_scores = np.zeros(int(chunk_end - chunk_start), dtype=np.float64)
    zero_gain_count = 0
    for local_idx in range(int(chunk_start), int(chunk_end)):
        piece = active_pieces[local_idx]
        sigma_i = float(active_sigma[local_idx])
        if sigma_i <= 1e-12:
            dock_vals = np.asarray([float(active_mu[local_idx])], dtype=np.float64)
        else:
            dock_vals = float(active_mu[local_idx]) + math.sqrt(2.0) * sigma_i * nodes
        gains = _piecewise_hvi_eval(piece, dock_vals)
        if not np.any(gains > tol):
            zero_gain_count += 1
            continue
        integrand = _piecewise_gain_product_packed(
            packed=packed,
            gains=gains,
            mu_values=active_mu,
            sigma_values=active_sigma,
            self_index=local_idx,
            tol=tol,
        )
        integrand = np.where(gains > tol, integrand, 0.0)
        if sigma_i <= 1e-12:
            chunk_scores[local_idx - chunk_start] = float(integrand[0])
        else:
            chunk_scores[local_idx - chunk_start] = float(np.dot(gh_weights, integrand))
    return int(chunk_start), chunk_scores, int(zero_gain_count)


def _init_qpmhi_worker(
    active_pieces: Sequence[dict[str, np.ndarray | float]],
    packed: dict[str, np.ndarray],
    active_mu: np.ndarray,
    active_sigma: np.ndarray,
    nodes: np.ndarray,
    gh_weights: np.ndarray,
    tol: float,
) -> None:
    global _QPMHI_WORKER_STATE
    _QPMHI_WORKER_STATE = {
        "active_pieces": active_pieces,
        "packed": packed,
        "active_mu": active_mu,
        "active_sigma": active_sigma,
        "nodes": nodes,
        "gh_weights": gh_weights,
        "tol": float(tol),
    }


def _score_active_chunk_worker(chunk: tuple[int, int]) -> tuple[int, np.ndarray, int]:
    state = _QPMHI_WORKER_STATE
    if not state:
        raise RuntimeError("qPMHI worker state is empty.")
    return _score_active_chunk_local(
        active_pieces=state["active_pieces"],
        packed=state["packed"],
        active_mu=state["active_mu"],
        active_sigma=state["active_sigma"],
        nodes=state["nodes"],
        gh_weights=state["gh_weights"],
        tol=float(state["tol"]),
        chunk_start=int(chunk[0]),
        chunk_end=int(chunk[1]),
    )


def _score_active_family(
    active_pieces: Sequence[dict[str, np.ndarray | float]],
    packed: dict[str, np.ndarray],
    active_mu: np.ndarray,
    active_sigma: np.ndarray,
    nodes: np.ndarray,
    gh_weights: np.ndarray,
    tol: float,
    parallel_workers: int | None,
) -> tuple[np.ndarray, int, int, int]:
    active_n = int(active_mu.shape[0])
    workers = _resolve_parallel_workers(parallel_workers=parallel_workers, active_n=active_n)
    chunks = _chunk_ranges(active_n, workers)
    local_scores = np.zeros(active_n, dtype=np.float64)
    zero_gain_count = 0
    if workers <= 1 or len(chunks) <= 1:
        for chunk_start, chunk_end in chunks:
            out_start, chunk_scores, chunk_zero_gain = _score_active_chunk_local(
                active_pieces=active_pieces,
                packed=packed,
                active_mu=active_mu,
                active_sigma=active_sigma,
                nodes=nodes,
                gh_weights=gh_weights,
                tol=tol,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
            local_scores[out_start:out_start + chunk_scores.shape[0]] = chunk_scores
            zero_gain_count += int(chunk_zero_gain)
        return local_scores, int(zero_gain_count), int(workers), 0

    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_qpmhi_worker,
        initargs=(active_pieces, packed, active_mu, active_sigma, nodes, gh_weights, float(tol)),
    ) as executor:
        for out_start, chunk_scores, chunk_zero_gain in executor.map(_score_active_chunk_worker, chunks):
            local_scores[out_start:out_start + chunk_scores.shape[0]] = chunk_scores
            zero_gain_count += int(chunk_zero_gain)
    return local_scores, int(zero_gain_count), int(workers), 0



def nehvi_gaussian_analytic_3d(
    dock_mu: torch.Tensor,
    dock_sigma: torch.Tensor,
    exact_obj: torch.Tensor,
    y_train: torch.Tensor | None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
    return_metadata: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, int]]:
    if dock_mu.dim() != 1:
        raise ValueError(f"dock_mu must be (N,), got {tuple(dock_mu.shape)}")
    if dock_sigma.shape != dock_mu.shape:
        raise ValueError("dock_sigma shape must match dock_mu.")
    if exact_obj.dim() != 2 or exact_obj.size(0) != dock_mu.numel() or exact_obj.size(1) != 2:
        raise ValueError(f"exact_obj must be (N,2), got {tuple(exact_obj.shape)}")

    out_device = dock_mu.device
    cand = torch.cat([dock_mu.view(-1, 1), exact_obj], dim=1).detach().cpu().to(dtype=torch.double)
    sigma_raw = dock_sigma.detach().cpu().to(dtype=torch.double)
    if not torch.isfinite(sigma_raw).all():
        raise ValueError("dock_sigma contains non-finite values.")
    if (sigma_raw < 0).any():
        raise ValueError("dock_sigma must be non-negative.")

    if weights is None:
        w_all = torch.ones(3, dtype=torch.double)
    else:
        if len(weights) != 3:
            raise ValueError("analytic NEHVI currently requires exactly three objectives.")
        w_all = torch.tensor(weights, dtype=torch.double)
    active = w_all.abs() > 0
    if int(active.sum().item()) != 3:
        raise ValueError("analytic NEHVI currently requires three active objectives: docking, QED, and -SA.")
    if not bool(active[0].item()):
        raise ValueError("analytic NEHVI requires docking to be the first active objective.")

    cand_weighted = cand * w_all.view(1, -1)
    if y_train is None:
        y_base = cand_weighted
    else:
        if y_train.dim() != 2 or y_train.size(1) != 3:
            raise ValueError(f"y_train must be (M,3), got {tuple(y_train.shape)}")
        y_base = y_train.detach().cpu().to(dtype=torch.double) * w_all.view(1, -1)

    mean = y_base.mean(0)
    std = y_base.std(0).clamp_min(1e-8)
    cand_std = (cand_weighted - mean) / std
    sigma_std = sigma_raw * float(abs(w_all[0].item())) / float(std[0].item())
    n_count = int(cand_std.size(0))
    scores = np.zeros(n_count, dtype=np.float64)
    metadata = {
        "input_n": n_count,
        "prefilter_ref_rect_zero_n": 0,
        "prefilter_no_support_n": 0,
        "scored_n": 0,
    }
    if n_count == 0:
        out = torch.zeros((0,), device=out_device, dtype=torch.double)
        return (out, metadata) if return_metadata else out

    if ref_point is not None:
        ref_raw = torch.tensor(ref_point, dtype=torch.double)
        if ref_raw.numel() != 3:
            raise ValueError("ref_point length must match objective dimension.")
        ref = (ref_raw * w_all - mean) / std
    else:
        ref = ((y_base - mean) / std).min(0).values - 1.0

    y_base_std = (y_base - mean) / std
    front = y_base_std[bt_is_non_dominated(y_base_std)] if y_base_std.numel() > 0 else y_base_std
    if front.numel() == 0:
        front = y_base_std

    ref_np = ref.detach().cpu().numpy().astype(np.float64)
    front_np = front.detach().cpu().numpy().astype(np.float64) if front.numel() > 0 else np.empty((0, 3), dtype=np.float64)
    mu_np = cand_std[:, 0].detach().cpu().numpy().astype(np.float64)
    sigma_np = sigma_std.detach().cpu().numpy().astype(np.float64)
    exact_np = cand_std[:, 1:].detach().cpu().numpy().astype(np.float64)

    rect_mask = (exact_np[:, 0] > float(ref_np[1])) & (exact_np[:, 1] > float(ref_np[2]))
    metadata["prefilter_ref_rect_zero_n"] = int((~rect_mask).sum())

    active_indices_list: list[int] = []
    for idx in np.flatnonzero(rect_mask):
        piece = _build_piecewise_hvi_3d(
            front_np,
            cand_y=float(exact_np[idx, 0]),
            cand_z=float(exact_np[idx, 1]),
            ref_point=ref_np,
        )
        slopes = np.asarray(piece["slopes"], dtype=np.float64)
        if not np.any(slopes > _PIECEWISE_NUMERIC_TOL):
            continue
        active_indices_list.append(int(idx))
        scores[int(idx)] = _piecewise_hvi_gaussian_expectation(
            piece=piece,
            mu=float(mu_np[idx]),
            sigma=float(sigma_np[idx]),
        )
    active_indices = np.asarray(active_indices_list, dtype=np.int64)
    metadata["prefilter_no_support_n"] = int(rect_mask.sum() - active_indices.size)
    metadata["scored_n"] = int(active_indices.size)

    out = torch.tensor(scores, device=out_device, dtype=torch.double)
    return (out, metadata) if return_metadata else out


def qphv_prob_gaussian_analytic_3d(
    dock_mu: torch.Tensor,
    dock_sigma: torch.Tensor,
    exact_obj: torch.Tensor,
    y_train: torch.Tensor | None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
    quadrature_points: int = 16,
    return_metadata: bool = False,
    parallel_workers: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, int]]:
    if dock_mu.dim() != 1:
        raise ValueError(f"dock_mu must be (N,), got {tuple(dock_mu.shape)}")
    if dock_sigma.shape != dock_mu.shape:
        raise ValueError("dock_sigma shape must match dock_mu.")
    if exact_obj.dim() != 2 or exact_obj.size(0) != dock_mu.numel() or exact_obj.size(1) != 2:
        raise ValueError(f"exact_obj must be (N,2), got {tuple(exact_obj.shape)}")
    if quadrature_points <= 0:
        raise ValueError("quadrature_points must be positive.")

    out_device = dock_mu.device
    cand = torch.cat([dock_mu.view(-1, 1), exact_obj], dim=1).detach().cpu().to(dtype=torch.double)
    sigma_raw = dock_sigma.detach().cpu().to(dtype=torch.double)
    if not torch.isfinite(sigma_raw).all():
        raise ValueError("dock_sigma contains non-finite values.")
    if (sigma_raw < 0).any():
        raise ValueError("dock_sigma must be non-negative.")

    if weights is None:
        w_all = torch.ones(3, dtype=torch.double)
    else:
        if len(weights) != 3:
            raise ValueError("analytic qPMHI currently requires exactly three objectives.")
        w_all = torch.tensor(weights, dtype=torch.double)
    active = w_all.abs() > 0
    if int(active.sum().item()) != 3:
        raise ValueError("analytic qPMHI currently requires three active objectives: docking, QED, and -SA.")
    if not bool(active[0].item()):
        raise ValueError("analytic qPMHI requires docking to be the first active objective.")

    cand_weighted = cand * w_all.view(1, -1)
    if y_train is None:
        y_base = cand_weighted
    else:
        if y_train.dim() != 2 or y_train.size(1) != 3:
            raise ValueError(f"y_train must be (M,3), got {tuple(y_train.shape)}")
        y_base = y_train.detach().cpu().to(dtype=torch.double) * w_all.view(1, -1)

    mean = y_base.mean(0)
    std = y_base.std(0).clamp_min(1e-8)
    cand_std = (cand_weighted - mean) / std
    sigma_std = sigma_raw * float(abs(w_all[0].item())) / float(std[0].item())
    n_count = int(cand_std.size(0))
    scores = np.zeros(n_count, dtype=np.float64)
    metadata = {
        "input_n": n_count,
        "prefilter_ref_rect_zero_n": 0,
        "prefilter_no_support_n": 0,
        "prefilter_zero_quadrature_gain_n": 0,
        "scored_n": 0,
        "parallel_workers": 1,
        "parallel_chunks": 0,
        "parallel_thread_fallback": 0,
    }
    if n_count == 0:
        out = torch.zeros((0,), device=out_device, dtype=torch.double)
        return (out, metadata) if return_metadata else out

    if ref_point is not None:
        ref_raw = torch.tensor(ref_point, dtype=torch.double)
        if ref_raw.numel() != 3:
            raise ValueError("ref_point length must match objective dimension.")
        ref = (ref_raw * w_all - mean) / std
    else:
        ref = ((y_base - mean) / std).min(0).values - 1.0

    y_base_std = (y_base - mean) / std
    front = y_base_std[bt_is_non_dominated(y_base_std)] if y_base_std.numel() > 0 else y_base_std
    if front.numel() == 0:
        front = y_base_std

    ref_np = ref.detach().cpu().numpy().astype(np.float64)
    front_np = front.detach().cpu().numpy().astype(np.float64) if front.numel() > 0 else np.empty((0, 3), dtype=np.float64)
    mu_np = cand_std[:, 0].detach().cpu().numpy().astype(np.float64)
    sigma_np = sigma_std.detach().cpu().numpy().astype(np.float64)
    exact_np = cand_std[:, 1:].detach().cpu().numpy().astype(np.float64)

    rect_mask = (exact_np[:, 0] > float(ref_np[1])) & (exact_np[:, 1] > float(ref_np[2]))
    metadata["prefilter_ref_rect_zero_n"] = int((~rect_mask).sum())

    pieces: list[dict[str, np.ndarray | float] | None] = [None] * n_count
    active_indices_list: list[int] = []
    for idx in np.flatnonzero(rect_mask):
        piece = _build_piecewise_hvi_3d(
            front_np,
            cand_y=float(exact_np[idx, 0]),
            cand_z=float(exact_np[idx, 1]),
            ref_point=ref_np,
        )
        pieces[int(idx)] = piece
        slopes = np.asarray(piece["slopes"], dtype=np.float64)
        if np.any(slopes > _PIECEWISE_NUMERIC_TOL):
            active_indices_list.append(int(idx))
    active_indices = np.asarray(active_indices_list, dtype=np.int64)
    metadata["prefilter_no_support_n"] = int(rect_mask.sum() - active_indices.size)
    metadata["scored_n"] = int(active_indices.size)
    if active_indices.size == 0:
        out = torch.tensor(scores, device=out_device, dtype=torch.double)
        return (out, metadata) if return_metadata else out

    active_mu = mu_np[active_indices]
    active_sigma = sigma_np[active_indices]
    active_pieces = [pieces[int(idx)] for idx in active_indices.tolist()]
    if any(piece is None for piece in active_pieces):
        raise RuntimeError("Active qPMHI piece is unexpectedly missing.")
    packed = _pack_piecewise_family(active_pieces, mu_values=active_mu)

    nodes, gh_weights = np.polynomial.hermite.hermgauss(int(quadrature_points))
    nodes = nodes.astype(np.float64)
    gh_weights = gh_weights.astype(np.float64) / math.sqrt(math.pi)
    tol = 1e-12

    local_scores, zero_gain_count, workers_used, thread_fallback = _score_active_family(
        active_pieces=active_pieces,
        packed=packed,
        active_mu=active_mu,
        active_sigma=active_sigma,
        nodes=nodes,
        gh_weights=gh_weights,
        tol=tol,
        parallel_workers=parallel_workers,
    )
    metadata["prefilter_zero_quadrature_gain_n"] = int(zero_gain_count)
    metadata["parallel_workers"] = int(workers_used)
    metadata["parallel_chunks"] = int(len(_chunk_ranges(int(active_indices.size), int(workers_used))))
    metadata["parallel_thread_fallback"] = int(thread_fallback)
    scores[active_indices] = local_scores

    scores = np.clip(scores, 0.0, None)
    total = float(scores.sum())
    if total > 0.0:
        scores /= total
    out = torch.tensor(scores, device=out_device, dtype=torch.double)
    return (out, metadata) if return_metadata else out
