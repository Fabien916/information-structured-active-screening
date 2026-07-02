from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated as bt_is_non_dominated

from mobo.metrics import _PIECEWISE_NUMERIC_TOL


def _normal_cdf_torch(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def _normal_pdf_torch(values: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * torch.square(values)) / math.sqrt(2.0 * math.pi)


def _as_torch_double(values: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(dtype=torch.double)
    return torch.as_tensor(values, dtype=torch.double)


def _standardize_analytic_inputs_3d(
    dock_mu: torch.Tensor | np.ndarray,
    dock_sigma: torch.Tensor | np.ndarray,
    exact_obj: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray | None,
    weights: Sequence[float] | None,
    ref_point: Sequence[float] | np.ndarray | torch.Tensor | None,
) -> dict[str, Any]:
    dock_mu_t = _as_torch_double(dock_mu).view(-1)
    dock_sigma_t = _as_torch_double(dock_sigma).view(-1)
    exact_obj_t = _as_torch_double(exact_obj)
    if exact_obj_t.dim() != 2 or exact_obj_t.size(0) != dock_mu_t.numel() or exact_obj_t.size(1) != 2:
        raise ValueError(f"exact_obj must be (N,2), got {tuple(exact_obj_t.shape)}")
    if dock_sigma_t.shape != dock_mu_t.shape:
        raise ValueError("dock_sigma shape must match dock_mu.")
    if not torch.isfinite(dock_sigma_t).all():
        raise ValueError("dock_sigma contains non-finite values.")
    if (dock_sigma_t < 0).any():
        raise ValueError("dock_sigma must be non-negative.")

    out_device = dock_mu.device if isinstance(dock_mu, torch.Tensor) else torch.device("cpu")
    cand = torch.cat([dock_mu_t.view(-1, 1), exact_obj_t], dim=1)

    if weights is None:
        w_all = torch.ones(3, dtype=torch.double)
    else:
        if len(weights) != 3:
            raise ValueError("analytic 3D HVI currently requires exactly three objectives.")
        w_all = torch.as_tensor(weights, dtype=torch.double)
    active = w_all.abs() > 0
    if int(active.sum().item()) != 3:
        raise ValueError("analytic 3D HVI currently requires three active objectives.")
    if not bool(active[0].item()):
        raise ValueError("analytic 3D HVI requires docking to be the first active objective.")

    cand_weighted = cand * w_all.view(1, -1)
    if y_train is None:
        y_base = cand_weighted
    else:
        y_train_t = _as_torch_double(y_train)
        if y_train_t.dim() != 2 or y_train_t.size(1) != 3:
            raise ValueError(f"y_train must be (M,3), got {tuple(y_train_t.shape)}")
        y_base = y_train_t * w_all.view(1, -1)

    mean = y_base.mean(0)
    std = y_base.std(0).clamp_min(1e-8)
    cand_std = (cand_weighted - mean) / std
    sigma_std = dock_sigma_t * float(abs(w_all[0].item())) / float(std[0].item())

    if ref_point is not None:
        ref_raw = _as_torch_double(ref_point).view(-1)
        if ref_raw.numel() != 3:
            raise ValueError("ref_point length must match objective dimension.")
        ref = (ref_raw * w_all - mean) / std
    else:
        ref = ((y_base - mean) / std).min(0).values - 1.0

    y_base_std = (y_base - mean) / std
    front = y_base_std[bt_is_non_dominated(y_base_std)] if y_base_std.numel() > 0 else y_base_std
    if front.numel() == 0:
        front = y_base_std

    return {
        "out_device": out_device,
        "cand_std": cand_std,
        "sigma_std": sigma_std,
        "front_np": front.detach().cpu().numpy().astype(np.float64) if front.numel() > 0 else np.empty((0, 3), dtype=np.float64),
        "ref_np": ref.detach().cpu().numpy().astype(np.float64),
        "mu_np": cand_std[:, 0].detach().cpu().numpy().astype(np.float64),
        "sigma_np": sigma_std.detach().cpu().numpy().astype(np.float64),
        "exact_np": cand_std[:, 1:].detach().cpu().numpy().astype(np.float64),
    }


def _prepare_union_area_segment(points: np.ndarray, ref_y: float, ref_z: float) -> dict[str, np.ndarray]:
    if points.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {
            "y_prev": empty,
            "widths": empty,
            "suffix_max_z": empty,
        }
    pts = np.asarray(points, dtype=np.float64)
    mask = (pts[:, 0] > ref_y) & (pts[:, 1] > ref_z)
    pts = pts[mask]
    if pts.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {
            "y_prev": empty,
            "widths": empty,
            "suffix_max_z": empty,
        }

    order = np.argsort(pts[:, 0], kind="mergesort")
    pts = pts[order]
    unique_y: list[float] = []
    max_z_at_y: list[float] = []
    for y_val, z_val in pts.tolist():
        y_float = float(y_val)
        z_float = float(z_val)
        if unique_y and y_float == unique_y[-1]:
            if z_float > max_z_at_y[-1]:
                max_z_at_y[-1] = z_float
        else:
            unique_y.append(y_float)
            max_z_at_y.append(z_float)

    y_arr = np.asarray(unique_y, dtype=np.float64)
    z_arr = np.asarray(max_z_at_y, dtype=np.float64)
    suffix_max_z = np.maximum.accumulate(z_arr[::-1])[::-1]
    y_prev = np.empty_like(y_arr)
    y_prev[0] = float(ref_y)
    if y_arr.size > 1:
        y_prev[1:] = y_arr[:-1]
    widths = np.maximum(y_arr - y_prev, 0.0)
    return {
        "y_prev": y_prev,
        "widths": widths,
        "suffix_max_z": suffix_max_z,
    }


def _precompute_shared_segments(front: np.ndarray, ref_point: np.ndarray) -> dict[str, Any]:
    front_arr = np.asarray(front, dtype=np.float64)
    if front_arr.size == 0:
        front_arr = np.empty((0, 3), dtype=np.float64)
    if front_arr.ndim != 2 or front_arr.shape[1] != 3:
        raise ValueError(f"front must be (N,3), got {tuple(front_arr.shape)}")

    ref = np.asarray(ref_point, dtype=np.float64)
    if ref.shape != (3,):
        raise ValueError(f"ref_point must be (3,), got {tuple(ref.shape)}")

    ref_x, ref_y, ref_z = float(ref[0]), float(ref[1]), float(ref[2])
    starts_list = [ref_x]
    if front_arr.size > 0:
        starts_list.extend(float(x) for x in front_arr[:, 0].tolist() if float(x) > ref_x)
    starts = np.asarray(sorted(set(starts_list)), dtype=np.float64)
    ends = np.empty_like(starts)
    if starts.size > 1:
        ends[:-1] = starts[1:]
    ends[-1] = np.inf

    segments: list[dict[str, np.ndarray]] = []
    for start in starts.tolist():
        active = front_arr[front_arr[:, 0] > float(start)]
        segments.append(_prepare_union_area_segment(active[:, 1:3], ref_y=ref_y, ref_z=ref_z))

    return {
        "ref_x": ref_x,
        "ref_y": ref_y,
        "ref_z": ref_z,
        "starts": starts,
        "ends": ends,
        "segments": segments,
    }


def _covered_area_from_segment_batch(
    segment: dict[str, np.ndarray],
    cand_y: np.ndarray,
    cand_z: np.ndarray,
    ref_z: float,
) -> np.ndarray:
    if cand_y.size == 0:
        return np.empty((0,), dtype=np.float64)
    widths = segment["widths"]
    if widths.size == 0:
        return np.zeros_like(cand_y, dtype=np.float64)
    y_prev = segment["y_prev"]
    suffix_max_z = segment["suffix_max_z"]
    width_matrix = np.clip(cand_y[:, None] - y_prev[None, :], 0.0, widths[None, :])
    height_matrix = np.maximum(np.minimum(cand_z[:, None], suffix_max_z[None, :]) - float(ref_z), 0.0)
    return np.sum(width_matrix * height_matrix, axis=1, dtype=np.float64)


def _compute_slopes_batch(
    front: np.ndarray,
    shared_starts: np.ndarray,
    cand_yz: np.ndarray,
    ref_point: np.ndarray,
    shared_segments: list[dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del front
    ref = np.asarray(ref_point, dtype=np.float64)
    starts = np.asarray(shared_starts, dtype=np.float64)
    ends = np.empty_like(starts)
    if starts.size > 1:
        ends[:-1] = starts[1:]
    ends[-1] = np.inf
    yz = np.asarray(cand_yz, dtype=np.float64)
    if yz.size == 0:
        empty = np.empty((0, starts.size), dtype=np.float64)
        return empty, empty, empty
    if yz.ndim != 2 or yz.shape[1] != 2:
        raise ValueError(f"cand_yz must be (N,2), got {tuple(yz.shape)}")
    if shared_segments is None:
        shared_segments = _precompute_shared_segments(front=np.empty((0, 3), dtype=np.float64), ref_point=ref)["segments"]

    ref_y, ref_z = float(ref[1]), float(ref[2])
    rect_area = np.maximum(yz[:, 0] - ref_y, 0.0) * np.maximum(yz[:, 1] - ref_z, 0.0)
    slope_tol = _PIECEWISE_NUMERIC_TOL * np.maximum(1.0, np.abs(rect_area))
    slopes = np.zeros((yz.shape[0], starts.size), dtype=np.float64)
    for seg_idx, segment in enumerate(shared_segments):
        covered = _covered_area_from_segment_batch(segment=segment, cand_y=yz[:, 0], cand_z=yz[:, 1], ref_z=ref_z)
        slope = np.maximum(rect_area - covered, 0.0)
        slopes[:, seg_idx] = np.where(slope <= slope_tol, 0.0, slope)

    interval_widths = np.where(np.isfinite(ends), ends - starts, 0.0)
    values_start = np.zeros_like(slopes)
    if starts.size > 1:
        cumulative = np.cumsum(slopes[:, :-1] * interval_widths[:-1][None, :], axis=1, dtype=np.float64)
        values_start[:, 1:] = cumulative
    intercepts = values_start - slopes * starts[None, :]
    values_start[np.abs(values_start) <= slope_tol[:, None]] = 0.0
    return slopes, intercepts, values_start


def _piecewise_hvi_eval_rows(
    starts: np.ndarray,
    ref_x: float,
    slopes: np.ndarray,
    values_start: np.ndarray,
    dock_values: np.ndarray,
) -> np.ndarray:
    xs = np.asarray(dock_values, dtype=np.float64).reshape(-1)
    out = np.zeros_like(xs, dtype=np.float64)
    if xs.size == 0:
        return out
    mask = xs > float(ref_x)
    if not np.any(mask):
        return out
    idx = np.searchsorted(starts[1:], xs[mask], side="right")
    rows = np.nonzero(mask)[0]
    out[mask] = values_start[rows, idx] + slopes[rows, idx] * (xs[mask] - starts[idx])
    return out


def _piecewise_hvi_eval_single(
    starts: np.ndarray,
    ref_x: float,
    slopes_row: np.ndarray,
    values_start_row: np.ndarray,
    dock_values: np.ndarray,
) -> np.ndarray:
    xs = np.asarray(dock_values, dtype=np.float64)
    out = np.zeros_like(xs, dtype=np.float64)
    mask = xs > float(ref_x)
    if not np.any(mask):
        return out
    idx = np.searchsorted(starts[1:], xs[mask], side="right")
    out[mask] = values_start_row[idx] + slopes_row[idx] * (xs[mask] - starts[idx])
    return out


def _build_values_end(starts: np.ndarray, ends: np.ndarray, slopes: np.ndarray, values_start: np.ndarray) -> np.ndarray:
    values_end = values_start.copy()
    finite_mask = np.isfinite(ends)
    if np.any(finite_mask):
        widths = (ends[finite_mask] - starts[finite_mask])[None, :]
        values_end[:, finite_mask] = values_start[:, finite_mask] + slopes[:, finite_mask] * widths
    if np.isinf(float(ends[-1])):
        values_end[:, -1] = np.where(slopes[:, -1] > 0.0, np.inf, values_start[:, -1])
    return values_end


def _analytic_ehvi_batch(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    slopes: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    values_start: torch.Tensor,
    ref_x: float,
) -> torch.Tensor:
    if mu.dim() != 1 or sigma.dim() != 1:
        raise ValueError("mu and sigma must be 1-D.")
    if slopes.dim() != 2 or values_start.dim() != 2:
        raise ValueError("slopes and values_start must be 2-D.")
    if slopes.shape != values_start.shape:
        raise ValueError("slopes and values_start shapes must match.")
    if slopes.size(1) != starts.numel() or starts.shape != ends.shape:
        raise ValueError("segment shapes are inconsistent.")
    if torch.any(sigma < 0.0):
        min_sigma = float(torch.min(sigma).item())
        raise ValueError(f"sigma must be non-negative, got min={min_sigma}.")

    out = torch.zeros_like(mu, dtype=torch.double)
    det_mask = sigma <= 1e-12
    if torch.any(det_mask):
        mu_det = mu[det_mask]
        out_det = torch.zeros_like(mu_det, dtype=torch.double)
        eval_mask = mu_det > float(ref_x)
        if torch.any(eval_mask):
            idx = torch.searchsorted(starts[1:], mu_det[eval_mask], right=True)
            row_idx = torch.nonzero(det_mask, as_tuple=False).view(-1)[eval_mask]
            out_det[eval_mask] = values_start[row_idx, idx] + slopes[row_idx, idx] * (mu_det[eval_mask] - starts[idx])
        out[det_mask] = out_det

    if torch.any(~det_mask):
        mu_pos = mu[~det_mask]
        sigma_pos = sigma[~det_mask]
        slopes_pos = slopes[~det_mask]
        values_start_pos = values_start[~det_mask]
        starts_row = starts.view(1, -1)
        ends_row = ends.view(1, -1)
        alpha = (starts_row - mu_pos[:, None]) / sigma_pos[:, None]
        cdf_alpha = _normal_cdf_torch(alpha)
        pdf_alpha = _normal_pdf_torch(alpha)
        finite_mask = torch.isfinite(ends_row)
        beta = torch.where(finite_mask, (ends_row - mu_pos[:, None]) / sigma_pos[:, None], torch.zeros_like(alpha))
        cdf_beta = torch.where(finite_mask, _normal_cdf_torch(beta), torch.ones_like(alpha))
        pdf_beta = torch.where(finite_mask, _normal_pdf_torch(beta), torch.zeros_like(alpha))
        prob = torch.where(finite_mask, cdf_beta - cdf_alpha, 1.0 - cdf_alpha)
        x_interval_mean = mu_pos[:, None] * prob + sigma_pos[:, None] * torch.where(
            finite_mask,
            pdf_alpha - pdf_beta,
            pdf_alpha,
        )
        intercepts = values_start_pos - slopes_pos * starts_row
        contrib = torch.where(prob > 0.0, slopes_pos * x_interval_mean + intercepts * prob, torch.zeros_like(prob))
        out[~det_mask] = torch.clamp(contrib.sum(dim=1), min=0.0)
    return out


def _pack_piecewise_family_from_batch(
    starts: np.ndarray,
    ends: np.ndarray,
    slopes: np.ndarray,
    values_start: np.ndarray,
    mu_values: np.ndarray,
    ref_x: float,
) -> dict[str, np.ndarray]:
    starts_arr = np.broadcast_to(starts.reshape(1, -1), slopes.shape).copy()
    values_end = _build_values_end(starts=starts, ends=ends, slopes=slopes, values_start=values_start)
    mu_gain = _piecewise_hvi_eval_rows(
        starts=starts,
        ref_x=ref_x,
        slopes=slopes,
        values_start=values_start,
        dock_values=np.asarray(mu_values, dtype=np.float64),
    )
    return {
        "starts": starts_arr,
        "slopes": np.asarray(slopes, dtype=np.float64),
        "values_start": np.asarray(values_start, dtype=np.float64),
        "values_end": values_end,
        "mu_gain": mu_gain,
    }


def _piecewise_gain_cdf_block_fast(
    packed: dict[str, np.ndarray],
    gains_block: np.ndarray,
    mu_block: np.ndarray,
    sigma_block: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    gains_arr = np.asarray(gains_block, dtype=np.float64)
    out = np.empty_like(gains_arr, dtype=np.float64)
    sigma_arr = np.asarray(sigma_block, dtype=np.float64)
    if np.any(sigma_arr < 0.0):
        min_sigma = float(np.min(sigma_arr))
        raise ValueError(f"sigma_block must be non-negative, got min={min_sigma}.")
    det_mask = sigma_arr <= 1e-12
    row_slice = slice(int(start_idx), int(start_idx) + gains_arr.shape[0])
    if np.any(det_mask):
        det_vals = packed["mu_gain"][row_slice][det_mask][:, None]
        out[det_mask] = (gains_arr[det_mask] >= det_vals).astype(np.float64)
    if np.any(~det_mask):
        starts = packed["starts"][row_slice][~det_mask]
        slopes = packed["slopes"][row_slice][~det_mask]
        values_start = packed["values_start"][row_slice][~det_mask]
        values_end = packed["values_end"][row_slice][~det_mask]
        gains_pos = gains_arr[~det_mask]
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
                if not np.any(unresolved):
                    break
        inv[unresolved] = np.inf
        inv_t = torch.as_tensor(inv, dtype=torch.double)
        mu_t = torch.as_tensor(np.asarray(mu_block, dtype=np.float64)[~det_mask][:, None], dtype=torch.double)
        sigma_t = torch.as_tensor(sigma_arr[~det_mask][:, None], dtype=torch.double)
        cdf = _normal_cdf_torch((inv_t - mu_t) / sigma_t)
        out[~det_mask] = cdf.detach().cpu().numpy()
    return out


def _piecewise_gain_product_packed_fast(
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
    integrand = np.ones(base_gains.shape[1], dtype=np.float64)
    n_count = int(mu_arr.shape[0])
    for row_start in range(0, n_count, int(block_size)):
        row_end = min(row_start + int(block_size), n_count)
        gains_block = np.broadcast_to(base_gains, (row_end - row_start, base_gains.shape[1])).copy()
        cdf_block = _piecewise_gain_cdf_block_fast(
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


def _piecewise_gain_product_packed_fast_many(
    packed: dict[str, np.ndarray],
    gains: np.ndarray,
    mu_values: np.ndarray,
    sigma_values: np.ndarray,
    self_indices: np.ndarray,
    tol: float,
    block_size: int = 512,
) -> np.ndarray:
    del tol
    gains_arr = np.asarray(gains, dtype=np.float64)
    if gains_arr.ndim != 2:
        raise ValueError(f"gains must be (C,Q), got {tuple(gains_arr.shape)}")
    self_arr = np.asarray(self_indices, dtype=np.int64).reshape(-1)
    if self_arr.size != gains_arr.shape[0]:
        raise ValueError("self_indices length must match gains rows.")
    mu_arr = np.asarray(mu_values, dtype=np.float64)
    sigma_arr = np.asarray(sigma_values, dtype=np.float64)
    if np.any(self_arr < 0) or np.any(self_arr >= mu_arr.shape[0]):
        raise ValueError("self_indices out of range.")

    chunk_n, node_n = gains_arr.shape
    integrand = np.ones((chunk_n, node_n), dtype=np.float64)
    flat_base = gains_arr.reshape(1, chunk_n * node_n)
    n_count = int(mu_arr.shape[0])
    for row_start in range(0, n_count, int(block_size)):
        row_end = min(row_start + int(block_size), n_count)
        gains_block = np.broadcast_to(flat_base, (row_end - row_start, flat_base.shape[1])).copy()
        cdf_block = _piecewise_gain_cdf_block_fast(
            packed=packed,
            gains_block=gains_block,
            mu_block=mu_arr[row_start:row_end],
            sigma_block=sigma_arr[row_start:row_end],
            start_idx=row_start,
        ).reshape(row_end - row_start, chunk_n, node_n)

        self_offsets = self_arr - row_start
        in_block = (self_offsets >= 0) & (self_offsets < (row_end - row_start))
        if np.any(in_block):
            chunk_positions = np.nonzero(in_block)[0]
            cdf_block[self_offsets[in_block], chunk_positions, :] = 1.0

        integrand *= np.prod(cdf_block, axis=0, dtype=np.float64)
        if not np.any(integrand > 0.0):
            break
    return integrand


def nehvi_gaussian_analytic_3d(
    dock_mu: torch.Tensor | np.ndarray,
    dock_sigma: torch.Tensor | np.ndarray,
    exact_obj: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray | None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
    return_metadata: bool = False,
    validate: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    from mobo.qpmhi_analytic import nehvi_gaussian_analytic_3d as _reference_nehvi

    total_start = time.perf_counter()
    prepared = _standardize_analytic_inputs_3d(
        dock_mu=dock_mu,
        dock_sigma=dock_sigma,
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    )
    out_device = prepared["out_device"]
    ref_np = prepared["ref_np"]
    mu_np = prepared["mu_np"]
    sigma_np = prepared["sigma_np"]
    exact_np = prepared["exact_np"]
    n_count = int(mu_np.shape[0])
    scores = np.zeros(n_count, dtype=np.float64)
    metadata: dict[str, float | int] = {
        "input_n": n_count,
        "prefilter_ref_rect_zero_n": 0,
        "prefilter_no_support_n": 0,
        "scored_n": 0,
        "prefilter_time_s": 0.0,
        "score_time_s": 0.0,
        "total_time_s": 0.0,
    }
    if n_count == 0:
        out = torch.zeros((0,), device=out_device, dtype=torch.double)
        return (out, metadata) if return_metadata else out

    prefilter_start = time.perf_counter()
    rect_mask = (exact_np[:, 0] > float(ref_np[1])) & (exact_np[:, 1] > float(ref_np[2]))
    metadata["prefilter_ref_rect_zero_n"] = int((~rect_mask).sum())
    active_indices = np.flatnonzero(rect_mask)
    shared = _precompute_shared_segments(front=prepared["front_np"], ref_point=ref_np)
    if active_indices.size > 0:
        slopes_active, _intercepts_active, values_start_active = _compute_slopes_batch(
            front=prepared["front_np"],
            shared_starts=shared["starts"],
            cand_yz=exact_np[active_indices],
            ref_point=ref_np,
            shared_segments=shared["segments"],
        )
        support_mask = np.any(slopes_active > _PIECEWISE_NUMERIC_TOL, axis=1)
        kept_indices = active_indices[support_mask]
    else:
        slopes_active = np.empty((0, shared["starts"].size), dtype=np.float64)
        values_start_active = np.empty((0, shared["starts"].size), dtype=np.float64)
        support_mask = np.zeros((0,), dtype=bool)
        kept_indices = np.empty((0,), dtype=np.int64)
    metadata["prefilter_no_support_n"] = int(active_indices.size - kept_indices.size)
    metadata["scored_n"] = int(kept_indices.size)
    prefilter_end = time.perf_counter()
    metadata["prefilter_time_s"] = float(prefilter_end - prefilter_start)

    score_start = time.perf_counter()
    if kept_indices.size > 0:
        slopes_kept = torch.as_tensor(slopes_active[support_mask], dtype=torch.double, device=out_device)
        values_start_kept = torch.as_tensor(values_start_active[support_mask], dtype=torch.double, device=out_device)
        mu_kept = torch.as_tensor(mu_np[kept_indices], dtype=torch.double, device=out_device)
        sigma_kept = torch.as_tensor(sigma_np[kept_indices], dtype=torch.double, device=out_device)
        starts_t = torch.as_tensor(shared["starts"], dtype=torch.double, device=out_device)
        ends_t = torch.as_tensor(shared["ends"], dtype=torch.double, device=out_device)
        batch_scores = _analytic_ehvi_batch(
            mu=mu_kept,
            sigma=sigma_kept,
            slopes=slopes_kept,
            starts=starts_t,
            ends=ends_t,
            values_start=values_start_kept,
            ref_x=float(shared["ref_x"]),
        )
        scores[kept_indices] = batch_scores.detach().cpu().numpy()
    score_end = time.perf_counter()
    metadata["score_time_s"] = float(score_end - score_start)
    metadata["total_time_s"] = float(score_end - total_start)

    out = torch.as_tensor(scores, dtype=torch.double, device=out_device)
    if validate:
        ref_out = _reference_nehvi(
            dock_mu=_as_torch_double(dock_mu),
            dock_sigma=_as_torch_double(dock_sigma),
            exact_obj=_as_torch_double(exact_obj),
            y_train=None if y_train is None else _as_torch_double(y_train),
            weights=weights,
            ref_point=ref_point,
            return_metadata=False,
        )
        if not torch.allclose(out.detach().cpu(), ref_out.detach().cpu(), rtol=1e-6, atol=1e-8):
            max_diff = torch.max(torch.abs(out.detach().cpu() - ref_out.detach().cpu())).item()
            raise AssertionError(f"Fast NEHVI mismatch against reference implementation: max_diff={max_diff:.3e}")
    return (out, metadata) if return_metadata else out


def qphv_prob_gaussian_analytic_3d(
    dock_mu: torch.Tensor | np.ndarray,
    dock_sigma: torch.Tensor | np.ndarray,
    exact_obj: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray | None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
    quadrature_points: int = 16,
    return_metadata: bool = False,
    parallel_workers: int | None = None,
    validate: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
    from mobo.qpmhi_analytic import qphv_prob_gaussian_analytic_3d as _reference_qpmhi

    del parallel_workers
    if quadrature_points <= 0:
        raise ValueError("quadrature_points must be positive.")

    total_start = time.perf_counter()
    prepared = _standardize_analytic_inputs_3d(
        dock_mu=dock_mu,
        dock_sigma=dock_sigma,
        exact_obj=exact_obj,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    )
    out_device = prepared["out_device"]
    ref_np = prepared["ref_np"]
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
    if n_count == 0:
        out = torch.zeros((0,), device=out_device, dtype=torch.double)
        return (out, metadata) if return_metadata else out

    prefilter_start = time.perf_counter()
    rect_mask = (exact_np[:, 0] > float(ref_np[1])) & (exact_np[:, 1] > float(ref_np[2]))
    metadata["prefilter_ref_rect_zero_n"] = int((~rect_mask).sum())
    active_indices = np.flatnonzero(rect_mask)
    shared = _precompute_shared_segments(front=prepared["front_np"], ref_point=ref_np)
    if active_indices.size > 0:
        slopes_active, _intercepts_active, values_start_active = _compute_slopes_batch(
            front=prepared["front_np"],
            shared_starts=shared["starts"],
            cand_yz=exact_np[active_indices],
            ref_point=ref_np,
            shared_segments=shared["segments"],
        )
        support_mask = np.any(slopes_active > _PIECEWISE_NUMERIC_TOL, axis=1)
        kept_indices = active_indices[support_mask]
        slopes_kept = slopes_active[support_mask]
        values_start_kept = values_start_active[support_mask]
    else:
        support_mask = np.zeros((0,), dtype=bool)
        kept_indices = np.empty((0,), dtype=np.int64)
        slopes_kept = np.empty((0, shared["starts"].size), dtype=np.float64)
        values_start_kept = np.empty((0, shared["starts"].size), dtype=np.float64)
    metadata["prefilter_no_support_n"] = int(active_indices.size - kept_indices.size)
    metadata["scored_n"] = int(kept_indices.size)
    prefilter_end = time.perf_counter()
    metadata["prefilter_time_s"] = float(prefilter_end - prefilter_start)

    score_start = time.perf_counter()
    if kept_indices.size > 0:
        starts = np.asarray(shared["starts"], dtype=np.float64)
        ends = np.asarray(shared["ends"], dtype=np.float64)
        active_mu = mu_np[kept_indices]
        active_sigma = sigma_np[kept_indices]
        packed = _pack_piecewise_family_from_batch(
            starts=starts,
            ends=ends,
            slopes=slopes_kept,
            values_start=values_start_kept,
            mu_values=active_mu,
            ref_x=float(shared["ref_x"]),
        )
        nodes, gh_weights = np.polynomial.hermite.hermgauss(int(quadrature_points))
        nodes = nodes.astype(np.float64)
        gh_weights = gh_weights.astype(np.float64) / math.sqrt(math.pi)
        tol = 1e-12
        zero_gain_count = 0
        active_scores = np.zeros(kept_indices.size, dtype=np.float64)
        candidate_block_size = 4
        node_scale = math.sqrt(2.0) * nodes.reshape(1, -1)
        for chunk_start in range(0, int(kept_indices.size), candidate_block_size):
            chunk_end = min(chunk_start + candidate_block_size, int(kept_indices.size))
            chunk_slice = slice(chunk_start, chunk_end)
            chunk_mu = active_mu[chunk_slice]
            chunk_sigma = active_sigma[chunk_slice]
            dock_vals = chunk_mu[:, None] + chunk_sigma[:, None] * node_scale
            gains = np.empty_like(dock_vals, dtype=np.float64)
            for node_idx in range(int(nodes.size)):
                gains[:, node_idx] = _piecewise_hvi_eval_rows(
                    starts=starts,
                    ref_x=float(shared["ref_x"]),
                    slopes=slopes_kept[chunk_slice],
                    values_start=values_start_kept[chunk_slice],
                    dock_values=dock_vals[:, node_idx],
                )
            nonzero_mask = np.any(gains > tol, axis=1)
            zero_gain_count += int((~nonzero_mask).sum())
            if not np.any(nonzero_mask):
                continue
            local_indices = np.arange(chunk_start, chunk_end, dtype=np.int64)[nonzero_mask]
            integrand = _piecewise_gain_product_packed_fast_many(
                packed=packed,
                gains=gains[nonzero_mask],
                mu_values=active_mu,
                sigma_values=active_sigma,
                self_indices=local_indices,
                tol=tol,
            )
            integrand = np.where(gains[nonzero_mask] > tol, integrand, 0.0)
            active_scores[local_indices] = np.dot(integrand, gh_weights)
        metadata["prefilter_zero_quadrature_gain_n"] = int(zero_gain_count)
        scores[kept_indices] = active_scores

    scores = np.clip(scores, 0.0, None)
    total = float(scores.sum())
    if total > 0.0:
        scores /= total
    score_end = time.perf_counter()
    metadata["score_time_s"] = float(score_end - score_start)
    metadata["total_time_s"] = float(score_end - total_start)
    metadata["parallel_chunks"] = int(math.ceil(max(int(kept_indices.size), 1) / 4.0)) if kept_indices.size > 0 else 0

    out = torch.as_tensor(scores, dtype=torch.double, device=out_device)
    if validate:
        ref_out = _reference_qpmhi(
            dock_mu=_as_torch_double(dock_mu),
            dock_sigma=_as_torch_double(dock_sigma),
            exact_obj=_as_torch_double(exact_obj),
            y_train=None if y_train is None else _as_torch_double(y_train),
            weights=weights,
            ref_point=ref_point,
            quadrature_points=quadrature_points,
            return_metadata=False,
        )
        if not torch.allclose(out.detach().cpu(), ref_out.detach().cpu(), rtol=1e-6, atol=1e-8):
            max_diff = torch.max(torch.abs(out.detach().cpu() - ref_out.detach().cpu())).item()
            raise AssertionError(f"Fast qPMHI mismatch against reference implementation: max_diff={max_diff:.3e}")
    return (out, metadata) if return_metadata else out


nehvi_gaussian_analytic_3d_fast = nehvi_gaussian_analytic_3d
qphv_prob_gaussian_analytic_3d_fast = qphv_prob_gaussian_analytic_3d
