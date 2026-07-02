from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.utils.multi_objective.pareto import is_non_dominated as bt_is_non_dominated
from botorch.utils.multi_objective.hypervolume import DominatedPartitioning
from tqdm.auto import tqdm

from mobo.io_utils import _read_smiles_csv, _is_valid_dock_score, _pick_smiles_column
from mobo.smiles_utils import compute_qed, compute_sa


def _apply_objective_weights_and_mask(
    samples: torch.Tensor,
    y_train: torch.Tensor | None,
    weights: Sequence[float] | None,
    ref_point: Sequence[float] | None,
) -> tuple[torch.Tensor, torch.Tensor | None, Sequence[float] | None]:
    """
    Apply objective weights and REMOVE zero-weight dimensions.
    Keeping zero-weight dimensions would make all points/ref equal on those axes,
    which can collapse hypervolume gains to exactly zero.
    """
    if weights is None:
        return samples, y_train, ref_point
    if len(weights) != samples.size(-1):
        raise ValueError("weights length must match objective dimension.")

    w_all = torch.tensor(weights, device=samples.device, dtype=samples.dtype)
    active = w_all.abs() > 0
    if int(active.sum().item()) <= 0:
        raise ValueError("At least one objective must have non-zero weight.")

    w = w_all[active]
    samples = samples[..., active] * w.view(1, 1, -1)
    if y_train is not None:
        y_train = y_train[..., active] * w.view(1, -1)

    if ref_point is not None:
        ref_raw = torch.tensor(ref_point, device=samples.device, dtype=samples.dtype)
        if ref_raw.numel() == w_all.numel():
            ref_raw = ref_raw[active]
        elif ref_raw.numel() != w.numel():
            raise ValueError("ref_point length must match objective dimension (full or active).")
        ref_point = (ref_raw * w).tolist()

    return samples, y_train, ref_point


def build_ref_point_from_objective(
    dock_sign: float,
    qed_sign: float,
    sa_sign: float | None = None,
    use_sa: bool = False,
    dock_valid_max: float | None = 0.0,
    sa_clamp_max: float | None = None,
    qed_min: float = 0.0,
    sa_worst_default: float = 10.0,
) -> list[float]:
    dock_worst = float(dock_valid_max if dock_valid_max is not None else 0.0)
    qed_worst = float(qed_min)
    if use_sa:
        sa_worst = float(sa_clamp_max if sa_clamp_max is not None else sa_worst_default)
        sa_mult = float(sa_sign) if sa_sign is not None else 1.0
        return [float(dock_sign) * dock_worst, float(qed_sign) * qed_worst, sa_mult * sa_worst]
    return [float(dock_sign) * dock_worst, float(qed_sign) * qed_worst]


def _is_non_dominated_nd(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=points.device)
    if points.dim() != 2:
        raise ValueError(f"points must be (N,D), got {tuple(points.shape)}")
    p = points
    ge = p[:, None, :] >= p[None, :, :]
    gt = p[:, None, :] > p[None, :, :]
    dominates = ge.all(-1) & gt.any(-1)
    dominated = dominates.any(0)
    return ~dominated


def pareto_prob_from_samples_nd(samples: torch.Tensor) -> torch.Tensor:
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")
    s_count, n_count = samples.shape[:2]
    counts = torch.zeros(n_count, device=samples.device, dtype=torch.double)
    for s in range(s_count):
        nd_mask = _is_non_dominated_nd(samples[s])
        counts += nd_mask.to(dtype=torch.double)
    return counts / max(s_count, 1)


def qphv_prob_scaled_nd_botorch(
    samples: torch.Tensor,
    y_train: torch.Tensor | None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
) -> torch.Tensor:
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")
    samples, y_train, ref_point = _apply_objective_weights_and_mask(
        samples=samples,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    )

    if y_train is None:
        y_base = samples.mean(0)
    else:
        y_base = y_train

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

    hv_base = DominatedPartitioning(ref, front).compute_hypervolume().item() if front.numel() > 0 else 0.0

    s_count, n_count = s_std.shape[:2]
    counts = torch.zeros(n_count, device=samples.device, dtype=torch.double)

    for s in range(s_count):
        sample = s_std[s]
        if front.numel() == 0:
            dom = torch.zeros(n_count, dtype=torch.bool, device=samples.device)
        else:
            ge = (front[:, None, :] >= sample[None, :, :]).all(-1)
            gt = (front[:, None, :] > sample[None, :, :]).any(-1)
            dom = (ge & gt).any(0)
        best_idx = 0
        best_gain = -float("inf")
        for i in range(n_count):
            if dom[i]:
                gain = 0.0
            else:
                pts = torch.cat([front, sample[i].view(1, -1)], dim=0) if front.numel() > 0 else sample[i].view(1, -1)
                hv_new = DominatedPartitioning(ref, pts).compute_hypervolume().item()
                gain = hv_new - hv_base
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        counts[best_idx] += 1.0
    return counts / max(s_count, 1)


def qphv_topk_prob_nd_botorch(
    samples: torch.Tensor,
    y_train: torch.Tensor | None,
    top_k: int,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
) -> torch.Tensor:
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    samples, y_train, ref_point = _apply_objective_weights_and_mask(
        samples=samples,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    )

    if y_train is None:
        y_base = samples.mean(0)
    else:
        y_base = y_train

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

    s_count, n_count = s_std.shape[:2]
    k = min(int(top_k), int(n_count))
    counts = torch.zeros(n_count, device=samples.device, dtype=torch.double)

    if int(s_std.size(-1)) != 3:
        raise ValueError(f"qPMHI fast kernel currently requires three active objectives, got {int(s_std.size(-1))}.")

    ref_np = ref.detach().cpu().numpy().astype(np.float64)
    front_np = front.detach().cpu().numpy().astype(np.float64) if front.numel() > 0 else np.empty((0, 3), dtype=np.float64)
    exact_np = s_std[0, :, 1:].detach().cpu().numpy().astype(np.float64)
    shared = _precompute_shared_segments_3d(front=front_np, ref_point=ref_np)
    slopes, values_start = _compute_slopes_batch_3d(
        shared_starts=np.asarray(shared["starts"], dtype=np.float64),
        cand_yz=exact_np,
        ref_point=ref_np,
        shared_segments=shared["segments"],
    )

    for sample_idx in range(s_count):
        dock_vals = s_std[sample_idx, :, 0].detach().cpu().numpy().astype(np.float64)
        gains_np = _piecewise_hvi_eval_rows(
            starts=np.asarray(shared["starts"], dtype=np.float64),
            ref_x=float(shared["ref_x"]),
            slopes=slopes,
            values_start=values_start,
            dock_values=dock_vals,
        )
        gains = torch.as_tensor(gains_np, device=samples.device, dtype=torch.double)
        top_idx = torch.topk(gains, k=k).indices
        counts[top_idx] += 1.0
    return counts / max(s_count, 1)


def _union_area_from_ref_2d(points: np.ndarray, ref_y: float, ref_z: float) -> float:
    if points.size == 0:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N,2), got {tuple(pts.shape)}")
    mask = (pts[:, 0] > ref_y) & (pts[:, 1] > ref_z)
    pts = pts[mask]
    if pts.size == 0:
        return 0.0

    # Collapse duplicate y breakpoints before building the suffix frontier.
    order = np.argsort(pts[:, 0], kind="mergesort")
    pts = pts[order]
    unique_y: list[float] = []
    max_z_at_y: list[float] = []
    for y, z in pts.tolist():
        y_val = float(y)
        z_val = float(z)
        if unique_y and y_val == unique_y[-1]:
            if z_val > max_z_at_y[-1]:
                max_z_at_y[-1] = z_val
        else:
            unique_y.append(y_val)
            max_z_at_y.append(z_val)

    y_arr = np.asarray(unique_y, dtype=np.float64)
    z_arr = np.asarray(max_z_at_y, dtype=np.float64)
    suffix_max_z = np.maximum.accumulate(z_arr[::-1])[::-1]

    hv = 0.0
    prev_y = float(ref_y)
    for y_val, z_val in zip(y_arr.tolist(), suffix_max_z.tolist()):
        width = max(float(y_val) - prev_y, 0.0)
        height = max(float(z_val) - float(ref_z), 0.0)
        hv += width * height
        prev_y = float(y_val)
    return hv


def _prepare_union_area_segment_batch(points: np.ndarray, ref_y: float, ref_z: float) -> dict[str, np.ndarray]:
    if points.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {"y_prev": empty, "widths": empty, "suffix_max_z": empty}
    pts = np.asarray(points, dtype=np.float64)
    mask = (pts[:, 0] > ref_y) & (pts[:, 1] > ref_z)
    pts = pts[mask]
    if pts.size == 0:
        empty = np.empty((0,), dtype=np.float64)
        return {"y_prev": empty, "widths": empty, "suffix_max_z": empty}
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
    return {"y_prev": y_prev, "widths": widths, "suffix_max_z": suffix_max_z}


def _precompute_shared_segments_3d(front: np.ndarray, ref_point: np.ndarray) -> dict[str, np.ndarray | float | list[dict[str, np.ndarray]]]:
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
        segments.append(_prepare_union_area_segment_batch(active[:, 1:3], ref_y=ref_y, ref_z=ref_z))
    return {"ref_x": ref_x, "ref_y": ref_y, "ref_z": ref_z, "starts": starts, "ends": ends, "segments": segments}


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


def _compute_slopes_batch_3d(
    shared_starts: np.ndarray,
    cand_yz: np.ndarray,
    ref_point: np.ndarray,
    shared_segments: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(shared_starts, dtype=np.float64)
    ends = np.empty_like(starts)
    if starts.size > 1:
        ends[:-1] = starts[1:]
    ends[-1] = np.inf
    yz = np.asarray(cand_yz, dtype=np.float64)
    if yz.size == 0:
        empty = np.empty((0, starts.size), dtype=np.float64)
        return empty, empty
    if yz.ndim != 2 or yz.shape[1] != 2:
        raise ValueError(f"cand_yz must be (N,2), got {tuple(yz.shape)}")
    ref = np.asarray(ref_point, dtype=np.float64)
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
    values_start[np.abs(values_start) <= slope_tol[:, None]] = 0.0
    return slopes, values_start


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


def _build_piecewise_hvi_3d(front: np.ndarray, cand_y: float, cand_z: float, ref_point: np.ndarray) -> dict[str, np.ndarray | float]:
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
    values_start = np.zeros_like(starts)
    slopes = np.zeros_like(starts)
    rect_area = max(float(cand_y) - ref_y, 0.0) * max(float(cand_z) - ref_z, 0.0)
    slope_tol = _PIECEWISE_NUMERIC_TOL * max(1.0, abs(rect_area))
    cumulative = 0.0
    for idx, start in enumerate(starts.tolist()):
        values_start[idx] = cumulative
        if rect_area <= 0.0:
            slopes[idx] = 0.0
        else:
            active = front_arr[front_arr[:, 0] > float(start)]
            if active.size == 0:
                covered = 0.0
            else:
                clipped = np.column_stack([
                    np.minimum(active[:, 1], float(cand_y)),
                    np.minimum(active[:, 2], float(cand_z)),
                ])
                covered = _union_area_from_ref_2d(clipped, ref_y=ref_y, ref_z=ref_z)
            slope_val = max(rect_area - covered, 0.0)
            slopes[idx] = 0.0 if slope_val <= slope_tol else slope_val
        if np.isfinite(ends[idx]):
            cumulative += slopes[idx] * max(float(ends[idx] - starts[idx]), 0.0)
    values_end = values_start.copy()
    finite_mask = np.isfinite(ends)
    values_end[finite_mask] = values_start[finite_mask] + slopes[finite_mask] * (ends[finite_mask] - starts[finite_mask])
    if np.isinf(ends[-1]):
        values_end[-1] = np.inf if slopes[-1] > 0.0 else values_start[-1]
    values_start[np.abs(values_start) <= slope_tol] = 0.0
    finite_small = finite_mask & (np.abs(values_end) <= slope_tol)
    values_end[finite_small] = 0.0
    return {
        "ref_x": ref_x,
        "starts": starts,
        "ends": ends,
        "slopes": slopes,
        "values_start": values_start,
        "values_end": values_end,
    }


def _piecewise_hvi_eval(piece: dict[str, np.ndarray | float], dock_values: np.ndarray) -> np.ndarray:
    xs = np.asarray(dock_values, dtype=np.float64)
    out = np.zeros_like(xs, dtype=np.float64)
    mask = xs > float(piece["ref_x"])
    if not np.any(mask):
        return out
    starts = np.asarray(piece["starts"], dtype=np.float64)
    values_start = np.asarray(piece["values_start"], dtype=np.float64)
    slopes = np.asarray(piece["slopes"], dtype=np.float64)
    idx = np.searchsorted(starts[1:], xs[mask], side="right")
    out[mask] = values_start[idx] + slopes[idx] * (xs[mask] - starts[idx])
    return out


def _piecewise_gain_cdf(piece: dict[str, np.ndarray | float], gains: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    g = np.asarray(gains, dtype=np.float64)
    sigma_val = float(sigma)
    if sigma_val < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma_val}.")
    if sigma_val <= 1e-12:
        value = float(_piecewise_hvi_eval(piece, np.asarray([mu], dtype=np.float64))[0])
        return (g >= value).astype(np.float64)
    starts = np.asarray(piece["starts"], dtype=np.float64)
    slopes = np.asarray(piece["slopes"], dtype=np.float64)
    values_start = np.asarray(piece["values_start"], dtype=np.float64)
    values_end = np.asarray(piece["values_end"], dtype=np.float64)
    inv = np.full(g.shape, -np.inf, dtype=np.float64)
    unresolved = g >= 0.0
    for idx in range(starts.size):
        if not np.any(unresolved):
            break
        slope = float(slopes[idx])
        if slope <= _PIECEWISE_NUMERIC_TOL:
            continue
        upper = float(values_end[idx])
        mask = unresolved & (g < upper)
        if np.any(mask):
            inv[mask] = starts[idx] + (g[mask] - values_start[idx]) / slope
            unresolved[mask] = False
    inv[unresolved] = np.inf
    inv_t = torch.as_tensor(inv, dtype=torch.double)
    cdf = 0.5 * (1.0 + torch.erf((inv_t - float(mu)) / (sigma_val * math.sqrt(2.0))))
    return cdf.detach().cpu().numpy()


def qphv_prob_gaussian_analytic_3d(
    dock_mu: torch.Tensor,
    dock_sigma: torch.Tensor,
    exact_obj: torch.Tensor,
    y_train: torch.Tensor | None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
    quadrature_points: int = 16,
) -> torch.Tensor:
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
    pieces = [
        _build_piecewise_hvi_3d(front_np, cand_y=float(exact_np[i, 0]), cand_z=float(exact_np[i, 1]), ref_point=ref_np)
        for i in range(cand_std.size(0))
    ]

    nodes, gh_weights = np.polynomial.hermite.hermgauss(int(quadrature_points))
    gh_weights = gh_weights.astype(np.float64) / math.sqrt(math.pi)
    tol = 1e-12
    scores = np.zeros(cand_std.size(0), dtype=np.float64)

    for i in range(cand_std.size(0)):
        sigma_i = float(sigma_np[i])
        if sigma_i <= 1e-12:
            dock_vals = np.asarray([float(mu_np[i])], dtype=np.float64)
            gains = _piecewise_hvi_eval(pieces[i], dock_vals)
            if not bool(gains[0] > tol):
                continue
            integrand = np.ones(1, dtype=np.float64)
            for j in range(cand_std.size(0)):
                if j == i:
                    continue
                integrand *= _piecewise_gain_cdf(pieces[j], gains, mu=float(mu_np[j]), sigma=float(sigma_np[j]))
            scores[i] = float(integrand[0])
            continue
        dock_vals = float(mu_np[i]) + math.sqrt(2.0) * sigma_i * nodes
        gains = _piecewise_hvi_eval(pieces[i], dock_vals)
        integrand = np.ones_like(gains, dtype=np.float64)
        for j in range(cand_std.size(0)):
            if j == i:
                continue
            integrand *= _piecewise_gain_cdf(pieces[j], gains, mu=float(mu_np[j]), sigma=float(sigma_np[j]))
            if not np.any(integrand > 0.0):
                break
        integrand = np.where(gains > tol, integrand, 0.0)
        scores[i] = float(np.dot(gh_weights, integrand))

    scores = np.clip(scores, 0.0, None)
    total = float(scores.sum())
    if total > 0.0:
        scores /= total
    return torch.tensor(scores, device=out_device, dtype=torch.double)

def qnehvi_scores_from_samples_nd(
    samples: torch.Tensor,
    y_train: torch.Tensor | None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")

    samples, y_train, ref_point = _apply_objective_weights_and_mask(
        samples=samples,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    )

    y_base = y_train if y_train is not None else samples.mean(0)
    if y_base is None:
        return torch.zeros(samples.size(1), device=samples.device, dtype=torch.double)

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

    hv_base = DominatedPartitioning(ref, front).compute_hypervolume().item() if front.numel() > 0 else 0.0

    s_count, n_count = s_std.shape[:2]
    scores = torch.zeros(n_count, device=samples.device, dtype=torch.double)

    pbar = None
    if show_progress:
        pbar = tqdm(
            total=int(s_count * n_count),
            desc="qnehvi: hv gains",
            leave=False,
            dynamic_ncols=True,
        )
    for s in range(s_count):
        sample = s_std[s]
        if front.numel() == 0:
            dom = torch.zeros(n_count, dtype=torch.bool, device=samples.device)
        else:
            ge = (front[:, None, :] >= sample[None, :, :]).all(-1)
            gt = (front[:, None, :] > sample[None, :, :]).any(-1)
            dom = (ge & gt).any(0)
        for i in range(n_count):
            if dom[i]:
                gain = 0.0
            else:
                pts = (
                    torch.cat([front, sample[i].view(1, -1)], dim=0)
                    if front.numel() > 0
                    else sample[i].view(1, -1)
                )
                hv_new = DominatedPartitioning(ref, pts).compute_hypervolume().item()
                gain = hv_new - hv_base
            if gain > 0.0:
                scores[i] += gain
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.set_postfix_str(f"sample {s + 1}/{s_count}", refresh=False)
    if pbar is not None:
        pbar.close()
    scores /= max(s_count, 1)
    return scores


class _DiscreteScoreAcq(torch.nn.Module):
    """
    A discrete acquisition wrapper over precomputed per-candidate scores.
    Candidate tensor encodes indices as a single float column.
    """

    def __init__(self, scores: torch.Tensor):
        super().__init__()
        if scores.dim() != 1:
            raise ValueError(f"scores must be 1-D, got {tuple(scores.shape)}")
        self.register_buffer("scores", scores)
        self.X_pending = None

    def set_X_pending(self, X_pending: torch.Tensor | None = None) -> None:
        self.X_pending = X_pending

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Expected from optimize_acqf_discrete: (..., q, d) with d=1; handle (..., d) as q=1.
        if X.dim() == 2:
            X = X.unsqueeze(-2)
        if X.dim() != 3 or X.size(-1) != 1:
            raise ValueError(f"Expected X shape (batch,q,1), got {tuple(X.shape)}")
        idx = torch.round(X[..., 0]).long().clamp_(0, int(self.scores.numel()) - 1)
        vals = self.scores[idx]
        return vals.sum(dim=-1)


def select_qnehvi_indices_discrete_from_samples_nd(
    samples: torch.Tensor,
    batch_size: int,
    y_train: torch.Tensor | None = None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
) -> list[int]:
    """
    Select q candidates on a discrete pool via optimize_acqf_discrete.
    Uses qNEHVI per-candidate scores as the acquisition value.
    """
    if batch_size <= 0:
        return []
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")
    n_count = samples.size(1)
    if n_count == 0:
        return []
    q = min(int(batch_size), int(n_count))
    scores = qnehvi_scores_from_samples_nd(
        samples=samples,
        y_train=y_train,
        weights=weights,
        ref_point=ref_point,
    ).to(dtype=torch.double)
    acq = _DiscreteScoreAcq(scores=scores)
    choices = torch.arange(n_count, device=samples.device, dtype=torch.double).view(-1, 1)
    Xopt, _ = optimize_acqf_discrete(
        acq_function=acq,
        q=q,
        choices=choices,
        unique=True,
    )
    idx_raw = torch.round(Xopt.reshape(-1)).long().tolist()
    out: list[int] = []
    seen: set[int] = set()
    for i in idx_raw:
        ii = int(max(0, min(n_count - 1, i)))
        if ii in seen:
            continue
        seen.add(ii)
        out.append(ii)
        if len(out) >= q:
            break
    return out


def select_qphv_indices_from_samples_nd(
    samples: torch.Tensor,
    batch_size: int,
    y_train: torch.Tensor | None = None,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
    eps: float = 1e-12,
) -> list[int]:
    if batch_size <= 0:
        return []
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")
    n_count = samples.size(1)
    if n_count == 0:
        return []
    probs = qphv_prob_scaled_nd_botorch(samples, y_train=y_train, weights=weights, ref_point=ref_point)
    idx = torch.nonzero(probs > eps).squeeze(-1)
    if idx.numel() > 0:
        idx = idx[torch.argsort(probs[idx], descending=True)]
    selected = idx[:batch_size].tolist()

    if len(selected) < batch_size:
        mask = torch.ones(n_count, dtype=torch.bool, device=probs.device)
        if selected:
            mask[torch.tensor(selected, device=probs.device, dtype=torch.long)] = False
        fallback = pareto_prob_from_samples_nd(samples[:, mask])
        extra = torch.topk(fallback, min(batch_size - len(selected), fallback.numel())).indices
        pool_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        selected += pool_idx[extra].tolist()
    return selected


def select_pareto_mc_indices_from_samples(
    samples: torch.Tensor,
    batch_size: int,
    weights: Sequence[float] | None = None,
    eps: float = 1e-12,
) -> list[int]:
    if batch_size <= 0:
        return []
    if samples.dim() != 3:
        raise ValueError(f"samples must be (S,N,D), got {tuple(samples.shape)}")
    n_count = samples.size(1)
    if n_count == 0:
        return []
    if weights is not None and len(weights) == samples.size(-1):
        w = torch.tensor(weights, device=samples.device, dtype=samples.dtype)
        samples = samples * w.view(1, 1, -1)

    probs = pareto_prob_from_samples_nd(samples)
    idx = torch.nonzero(probs > eps).squeeze(-1)
    if idx.numel() > 0:
        idx = idx[torch.argsort(probs[idx], descending=True)]
    selected = idx[:batch_size].tolist()

    if len(selected) < batch_size:
        mask = torch.ones(n_count, dtype=torch.bool, device=probs.device)
        if selected:
            mask[torch.tensor(selected, device=probs.device, dtype=torch.long)] = False
        fallback = samples.mean(0).sum(-1)
        extra = torch.topk(fallback[mask], min(batch_size - len(selected), mask.sum().item())).indices
        pool_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        selected += pool_idx[extra].tolist()
    return selected


def _as_float(val):
    try:
        num = float(val)
    except Exception:
        raise ValueError(f"Failed to parse float value: {val!r}")
    if not math.isfinite(num):
        raise ValueError(f"Non-finite float value: {val!r}")
    return num


def _is_non_dominated_2d(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=points.device)
    if points.dim() != 2 or points.size(-1) != 2:
        raise ValueError(f"points must be (N,2), got {tuple(points.shape)}")
    order = torch.argsort(points[:, 0], descending=True)
    mask = torch.zeros(points.size(0), dtype=torch.bool, device=points.device)
    max_y = None
    for idx in order.tolist():
        y = points[idx, 1].item()
        if max_y is None or y > max_y:
            mask[idx] = True
            max_y = y
    return mask


def _hypervolume_2d_max(points: torch.Tensor, ref: torch.Tensor) -> float:
    if points.numel() == 0:
        return 0.0
    mask = _is_non_dominated_2d(points)
    front = points[mask]
    if front.numel() == 0:
        return 0.0
    order = torch.argsort(front[:, 0])
    front = front[order]
    ref_x, ref_y = float(ref[0].item()), float(ref[1].item())
    hv = 0.0
    prev_x = ref_x
    for x, y in front.tolist():
        width = max(x - prev_x, 0.0)
        height = max(y - ref_y, 0.0)
        hv += width * height
        prev_x = x
    return float(hv)


_PIECEWISE_NUMERIC_TOL = 1e-12


def _pareto_layers_2d(points: torch.Tensor) -> list[torch.Tensor]:
    layers: list[torch.Tensor] = []
    if points.numel() == 0:
        return layers
    remaining = torch.arange(points.size(0), device=points.device)
    pts = points
    while pts.numel() > 0:
        mask = _is_non_dominated_2d(pts)
        layers.append(remaining[mask])
        keep = ~mask
        remaining = remaining[keep]
        pts = pts[keep]
    return layers


def _pareto_layers_nd(points: torch.Tensor) -> list[torch.Tensor]:
    layers: list[torch.Tensor] = []
    if points.numel() == 0:
        return layers
    remaining = torch.arange(points.size(0), device=points.device)
    pts = points
    while pts.numel() > 0:
        mask = _is_non_dominated_nd(pts)
        layers.append(remaining[mask])
        keep = ~mask
        remaining = remaining[keep]
        pts = pts[keep]
    return layers


def _compute_rank_and_distance(points: torch.Tensor, new_mask: torch.Tensor) -> dict | None:
    if points.numel() == 0 or new_mask.sum() == 0:
        return None
    if points.size(1) == 2:
        layers = _pareto_layers_2d(points)
        front_mask = _is_non_dominated_2d(points)
    else:
        layers = _pareto_layers_nd(points)
        front_mask = _is_non_dominated_nd(points)
    rank = torch.zeros(points.size(0), dtype=torch.long, device=points.device)
    for i, idx in enumerate(layers, start=1):
        rank[idx] = i
    new_rank = rank[new_mask]
    front = points[front_mask]
    if front.numel() == 0:
        return None
    mins = points.min(0).values
    maxs = points.max(0).values
    denom = (maxs - mins).clamp_min(1e-8)
    pts_norm = (points - mins) / denom
    front_norm = (front - mins) / denom
    new_pts_norm = pts_norm[new_mask]
    dists = torch.cdist(new_pts_norm, front_norm, p=2)
    min_dist = dists.min(dim=1).values
    return {
        "rank_min": int(new_rank.min().item()),
        "rank_median": float(new_rank.median().item()),
        "rank_mean": float(new_rank.float().mean().item()),
        "dist_min": float(min_dist.min().item()),
        "dist_median": float(min_dist.median().item()),
        "dist_mean": float(min_dist.mean().item()),
    }


def compute_hvi_from_csv(
    csv_path: str,
    new_ids: Sequence[str],
    dock_sign: float,
    qed_sign: float,
    sa_sign: float | None = None,
    use_sa: bool = False,
    sa_clamp_min: float | None = None,
    sa_clamp_max: float | None = None,
    dock_valid_max: float | None = 0.0,
    ref_point: Sequence[float] | None = None,
    extra_objectives: Sequence[str] | None = None,
    enabled_objectives: Sequence[str] | None = None,
) -> dict | None:
    df = _read_smiles_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"compute_hvi_from_csv: empty CSV: {csv_path}")
    if "ligand_id" not in df.columns:
        raise RuntimeError("compute_hvi_from_csv: missing required column ligand_id.")
    if "dock_score" not in df.columns:
        raise RuntimeError("compute_hvi_from_csv: missing required column dock_score.")
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        raise RuntimeError("compute_hvi_from_csv: missing smiles column.")
    qed_col = None
    for cand in ("qed", "QED"):
        if cand in df.columns:
            qed_col = cand
            break

    supported_order = ["dock", "qed", "sa", "sim_prior", "win", "hb", "fsp3"]
    supported_extra = ["sim_prior", "win", "hb", "fsp3"]
    extra_objectives = [str(x).strip().lower() for x in (extra_objectives or []) if str(x).strip()]
    extra_objectives = [x for x in supported_extra if x in set(extra_objectives)]

    default_objectives = ["dock", "qed"] + (["sa"] if use_sa else []) + list(extra_objectives)
    if enabled_objectives is None:
        active_objectives = list(default_objectives)
    else:
        enabled_set = {str(x).strip().lower() for x in enabled_objectives if str(x).strip()}
        active_objectives = [name for name in supported_order if name in enabled_set and name in set(default_objectives)]
    if not active_objectives:
        raise RuntimeError("compute_hvi_from_csv: no active objectives after weight mask.")

    new_set = {str(x) for x in new_ids}
    points = []
    is_new = []
    total_rows = len(df)
    total_with_dock = 0
    qed_series = df[qed_col] if qed_col is not None else None
    sa_series = df["sa_score"] if "sa_score" in df.columns else None
    sim_series = df["sim_prior"] if ("sim_prior" in df.columns and "sim_prior" in extra_objectives) else None
    win_series = df["win"] if ("win" in df.columns and "win" in extra_objectives) else None
    hb_series = df["hb"] if ("hb" in df.columns and "hb" in extra_objectives) else None
    fsp3_series = df["fsp3"] if ("fsp3" in df.columns and "fsp3" in extra_objectives) else None

    for lig_id, dock_score, smi, qed_raw, sa_raw, sim_raw, win_raw, hb_raw, fsp3_raw in zip(
        df["ligand_id"].astype(str),
        df["dock_score"],
        df[smiles_col],
        qed_series if qed_series is not None else [None] * len(df),
        sa_series if sa_series is not None else [None] * len(df),
        sim_series if sim_series is not None else [None] * len(df),
        win_series if win_series is not None else [None] * len(df),
        hb_series if hb_series is not None else [None] * len(df),
        fsp3_series if fsp3_series is not None else [None] * len(df),
    ):
        dock = _as_float(dock_score)
        has_valid_dock = _is_valid_dock_score(dock, dock_valid_max=dock_valid_max)
        if has_valid_dock:
            total_with_dock += 1

        # Build only active objectives (masked by non-zero weights in caller).
        row_obj = []
        if "dock" in active_objectives:
            if not has_valid_dock:
                continue
            row_obj.append(dock_sign * dock)

        if "qed" in active_objectives:
            qed_val = _as_float(qed_raw)
            if qed_val is None or not np.isfinite(qed_val):
                qed_val = compute_qed(str(smi))
            if not np.isfinite(qed_val):
                continue
            row_obj.append(qed_sign * float(qed_val))

        if "sa" in active_objectives:
            sa_val = _as_float(sa_raw)
            if sa_val is None or not np.isfinite(sa_val):
                sa_val = compute_sa(str(smi), clamp_min=sa_clamp_min, clamp_max=sa_clamp_max)
            if not np.isfinite(sa_val):
                continue
            sa_mult = float(sa_sign) if sa_sign is not None else 1.0
            row_obj.append(sa_mult * float(sa_val))

        # Extra objectives are expected in [0,1] and already aligned with maximization.
        if "sim_prior" in active_objectives:
            sim_val = _as_float(sim_raw)
            if sim_val is None:
                continue
            row_obj.append(float(sim_val))
        if "win" in active_objectives:
            win_val = _as_float(win_raw)
            if win_val is None:
                continue
            row_obj.append(float(win_val))
        if "hb" in active_objectives:
            hb_val = _as_float(hb_raw)
            if hb_val is None:
                continue
            row_obj.append(float(hb_val))
        if "fsp3" in active_objectives:
            fsp3_val = _as_float(fsp3_raw)
            if fsp3_val is None:
                continue
            row_obj.append(float(fsp3_val))

        points.append(row_obj)
        is_new.append(lig_id in new_set)
    if not points:
        raise RuntimeError("compute_hvi_from_csv: no valid objective points constructed.")
    pts = torch.tensor(points, dtype=torch.float64)
    new_mask = torch.tensor(is_new, dtype=torch.bool)
    old_pts = pts[~new_mask]
    # Normalize reference point length if the caller provided legacy base dims.
    if ref_point is not None and len(ref_point) != pts.size(1):
        base_dim = 2 + (1 if use_sa else 0)
        if len(ref_point) == base_dim and pts.size(1) > base_dim:
            ref_point = list(ref_point) + [0.0] * (pts.size(1) - base_dim)

    use_nd = pts.size(1) >= 3
    if use_nd:
        if old_pts.numel() == 0:
            hv_old = 0.0
            ref = old_pts.new_zeros((pts.size(1),))
        else:
            if ref_point is None:
                ref = old_pts.min(0).values - 1.0
            else:
                ref = torch.tensor(ref_point, dtype=torch.float64)
                if ref.numel() != pts.size(1):
                    raise ValueError("ref_point length must match objective dimension.")
            old_front = old_pts[bt_is_non_dominated(old_pts)] if old_pts.numel() > 0 else old_pts
            hv_old = DominatedPartitioning(ref, old_front).compute_hypervolume().item() if old_front.numel() > 0 else 0.0
        new_front = pts[bt_is_non_dominated(pts)] if pts.numel() > 0 else pts
        hv_new = DominatedPartitioning(ref, new_front).compute_hypervolume().item() if new_front.numel() > 0 else 0.0
        pareto_count = int(_is_non_dominated_nd(pts).sum().item())
    else:
        # 2D-only path (legacy): maximize both objectives.
        if old_pts.numel() == 0:
            hv_old = 0.0
            ref = old_pts.new_zeros((2,))
        else:
            if ref_point is None:
                ref = old_pts.min(0).values - 1.0
            else:
                ref = torch.tensor(ref_point, dtype=torch.float64)
                if ref.numel() != 2:
                    raise ValueError("ref_point length must match objective dimension.")
            hv_old = _hypervolume_2d_max(old_pts, ref)
        hv_new = _hypervolume_2d_max(pts, ref)
        pareto_count = int(_is_non_dominated_2d(pts).sum().item())
    hvi = hv_new - hv_old
    rank_stats = _compute_rank_and_distance(pts, new_mask)
    return {
        "hv_old": hv_old,
        "hv_new": hv_new,
        "hvi": hvi,
        "new_count": int(new_mask.sum().item()),
        "total_count": int(pts.size(0)),
        "total_rows": int(total_rows),
        "total_with_dock": int(total_with_dock),
        "pareto_count": pareto_count,
        "rank_stats": rank_stats,
    }


def collect_oracle_points(
    csv_path: str,
    ligand_ids: Sequence[str],
    dock_sign: float,
    qed_sign: float,
    sa_sign: float | None = None,
    use_sa: bool = False,
    sa_clamp_min: float | None = None,
    sa_clamp_max: float | None = None,
    dock_valid_max: float | None = 0.0,
    extra_objectives: Sequence[str] | None = None,
) -> tuple[list[str], list[str], torch.Tensor, list[float], list[float], list[float] | None]:
    df = _read_smiles_csv(csv_path)
    if df.empty or "ligand_id" not in df.columns or "dock_score" not in df.columns:
        raise RuntimeError("collect_oracle_points: missing required columns or empty CSV.")
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        raise RuntimeError("collect_oracle_points: missing smiles column.")
    qed_col = None
    for cand in ("qed", "QED"):
        if cand in df.columns:
            qed_col = cand
            break
    df_idx = df.copy()
    df_idx["ligand_id"] = df_idx["ligand_id"].astype(str)
    df_idx = df_idx.set_index("ligand_id", drop=False)
    extra_objectives = [str(x).strip().lower() for x in (extra_objectives or []) if str(x).strip()]
    supported_extra = ["sim_prior", "win", "hb", "fsp3"]
    extra_objectives = [x for x in supported_extra if x in set(extra_objectives)]

    smiles = []
    lig_ids = []
    dock_raw_list = []
    qed_raw_list = []
    sa_raw_list: list[float] = []
    obj_vals = []
    for lig_id in ligand_ids:
        lig_id = str(lig_id)
        if lig_id not in df_idx.index:
            raise KeyError(f"collect_oracle_points: ligand_id '{lig_id}' not found in CSV.")
        row = df_idx.loc[lig_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        dock = _as_float(row.get("dock_score"))
        if not _is_valid_dock_score(dock, dock_valid_max=dock_valid_max):
            raise RuntimeError(f"collect_oracle_points: invalid dock score for ligand_id '{lig_id}': {row.get('dock_score')}")
        smi = row.get(smiles_col, "")
        if smi is None:
            raise RuntimeError(f"collect_oracle_points: missing SMILES for ligand_id '{lig_id}'")
        qed_val = _as_float(row.get(qed_col)) if qed_col is not None else None
        if qed_val is None:
            qed_val = compute_qed(str(smi))
        if not np.isfinite(qed_val):
            raise RuntimeError(f"collect_oracle_points: non-finite QED for ligand_id '{lig_id}'")
        sa_val = None
        if use_sa:
            sa_val = compute_sa(str(smi), clamp_min=sa_clamp_min, clamp_max=sa_clamp_max)
            if sa_val is None or not np.isfinite(sa_val):
                sa_val = _as_float(row.get("sa_score"))
            if sa_val is None or not np.isfinite(sa_val):
                raise RuntimeError(f"collect_oracle_points: non-finite SA for ligand_id '{lig_id}'")
        smiles.append(str(smi))
        lig_ids.append(lig_id)
        dock_raw_list.append(float(dock))
        qed_raw_list.append(float(qed_val))
        if use_sa:
            sign = sa_sign if sa_sign is not None else -1.0
            sa_raw_list.append(float(sa_val))
            row_obj = [dock_sign * float(dock), qed_sign * float(qed_val), sign * float(sa_val)]
        else:
            row_obj = [dock_sign * float(dock), qed_sign * float(qed_val)]

        # Extra objectives: read from CSV (already aligned with maximization in [0,1]).
        # If any requested objective is missing for this row, skip to keep objective dimensionality consistent.
        if "sim_prior" in extra_objectives:
            v = _as_float(row.get("sim_prior"))
            if v is None:
                raise RuntimeError(f"collect_oracle_points: missing sim_prior for ligand_id '{lig_id}'")
            row_obj.append(float(v))
        if "win" in extra_objectives:
            v = _as_float(row.get("win"))
            if v is None:
                raise RuntimeError(f"collect_oracle_points: missing win for ligand_id '{lig_id}'")
            row_obj.append(float(v))
        if "hb" in extra_objectives:
            v = _as_float(row.get("hb"))
            if v is None:
                raise RuntimeError(f"collect_oracle_points: missing hb for ligand_id '{lig_id}'")
            row_obj.append(float(v))
        if "fsp3" in extra_objectives:
            v = _as_float(row.get("fsp3"))
            if v is None:
                raise RuntimeError(f"collect_oracle_points: missing fsp3 for ligand_id '{lig_id}'")
            row_obj.append(float(v))

        obj_vals.append(row_obj)
    if not obj_vals:
        raise RuntimeError("collect_oracle_points: no objective values assembled.")
    return lig_ids, smiles, torch.tensor(obj_vals, dtype=torch.float32), dock_raw_list, qed_raw_list, (sa_raw_list if use_sa else None)


def _pareto_front_indices(
    selected: Sequence[int],
    dock_vals: Sequence[float],
    qed_vals: Sequence[float],
) -> list[int]:
    front: list[int] = []
    for i in selected:
        d_i = dock_vals[i]
        q_i = qed_vals[i]
        dominated = False
        for j in selected:
            if i == j:
                continue
            d_j = dock_vals[j]
            q_j = qed_vals[j]
            if (d_j <= d_i and q_j >= q_i) and (d_j < d_i or q_j > q_i):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _topk_selected(
    selected: Sequence[int],
    smiles: Sequence[str],
    mu_obj: torch.Tensor,
    dock_sign: float,
    qed_sign: float,
    top_k: int = 5,
) -> list[tuple[int, str, float, float, float]]:
    if not selected:
        return []
    scores = (mu_obj[:, 0] + mu_obj[:, 1]).cpu().numpy().tolist()
    ranked = sorted(selected, key=lambda i: scores[i], reverse=True)
    rows = []
    for rank, idx in enumerate(ranked[:top_k], start=1):
        dock_raw = float(mu_obj[idx, 0]) * dock_sign
        qed_raw = float(mu_obj[idx, 1]) * qed_sign
        score = float(mu_obj[idx, 0] + mu_obj[idx, 1])
        rows.append((rank, smiles[idx], dock_raw, qed_raw, score))
    return rows


def _topk_by_dock(
    selected: Sequence[int],
    smiles: Sequence[str],
    mu_obj: torch.Tensor,
    dock_sign: float,
    qed_sign: float,
    top_k: int = 5,
) -> list[tuple[int, str, float, float]]:
    if not selected:
        return []
    dock_raw = (mu_obj[:, 0].cpu().numpy() * dock_sign).tolist()
    qed_raw = (mu_obj[:, 1].cpu().numpy() * qed_sign).tolist()
    ranked = sorted(selected, key=lambda i: dock_raw[i])[:top_k]
    rows = []
    for rank, idx in enumerate(ranked, start=1):
        rows.append((rank, smiles[idx], float(dock_raw[idx]), float(qed_raw[idx])))
    return rows


def _topk_pareto(
    selected: Sequence[int],
    smiles: Sequence[str],
    mu_obj: torch.Tensor,
    dock_sign: float,
    qed_sign: float,
    top_k: int = 5,
) -> list[tuple[int, str, float, float]]:
    if not selected:
        return []
    dock_raw = (mu_obj[:, 0].cpu().numpy() * dock_sign).tolist()
    qed_raw = (mu_obj[:, 1].cpu().numpy() * qed_sign).tolist()
    front = _pareto_front_indices(selected, dock_raw, qed_raw)
    front_sorted = sorted(front, key=lambda i: (dock_raw[i], -qed_raw[i]))
    rows = []
    for rank, idx in enumerate(front_sorted[:top_k], start=1):
        rows.append((rank, smiles[idx], float(dock_raw[idx]), float(qed_raw[idx])))
    return rows


def _topk_oracle_rows(
    ids: Sequence[str],
    smiles: Sequence[str],
    mu_obj: torch.Tensor,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float | None,
    sa_vals: Sequence[float] | None,
    pred_dock_vals: Sequence[float] | None,
    weights: Sequence[float] | None,
    top_k: int = 5,
) -> list[list[object]]:
    if mu_obj.numel() == 0:
        return []
    if weights is None or len(weights) != mu_obj.size(-1):
        w = torch.ones(mu_obj.size(-1), device=mu_obj.device, dtype=mu_obj.dtype)
    else:
        w = torch.tensor(weights, device=mu_obj.device, dtype=mu_obj.dtype)
    scores = (mu_obj * w.view(1, -1)).sum(-1)
    ranked = torch.argsort(scores, descending=True)[:top_k].tolist()
    rows = []
    for rank, idx in enumerate(ranked, start=1):
        dock_raw = float(mu_obj[idx, 0]) * dock_sign
        pred_raw = ""
        if pred_dock_vals is not None and idx < len(pred_dock_vals):
            pred_val = pred_dock_vals[idx]
            if pred_val is not None and np.isfinite(pred_val):
                pred_raw = f"{float(pred_val):.4f}"
        qed_raw = float(mu_obj[idx, 1]) * qed_sign
        sa_raw = ""
        if mu_obj.size(-1) >= 3:
            if sa_vals is not None and idx < len(sa_vals):
                sa_raw = f"{float(sa_vals[idx]):.4f}"
            else:
                sa_raw = f"{float(mu_obj[idx, 2]) * (sa_sign if sa_sign is not None else -1.0):.4f}"
        rows.append(
            [rank, ids[idx], f"{dock_raw:.4f}", pred_raw, f"{qed_raw:.4f}", sa_raw, f"{float(scores[idx]):.4f}", smiles[idx]]
        )
    return rows


def _topk_oracle_rows_weighted_pct(
    dataset_csv_path: str,
    ids: Sequence[str],
    smiles: Sequence[str],
    mu_obj: torch.Tensor,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float | None,
    use_sa: bool,
    dock_valid_max: float | None,
    pred_dock_vals: Sequence[float] | None,
    weights: Sequence[float] | None,
    extra_objectives: Sequence[str] | None = None,
    sim_prior_sign: float = 1.0,
    win_sign: float = 1.0,
    hb_sign: float = 1.0,
    fsp3_sign: float = 1.0,
    top_k: int = 5,
) -> list[list[object]]:
    """
    Rank the newly-oracled points (ids) by a percentile-based weighted score:
      score = sum_i w_i * pct_i / sum_i w_i
    where pct_i is the empirical percentile (fraction <= x) of the *signed* objective i
    among all valid points in dataset_csv_path.

    This is more interpretable than summing raw objectives with different scales.
    """
    if mu_obj.numel() == 0 or not ids:
        return []
    extra_objectives = list(extra_objectives or [])
    base_dim = 2 + (1 if use_sa else 0)
    expected_dim = base_dim + len(extra_objectives)
    if mu_obj.size(-1) != expected_dim:
        # Fallback: keep working even if dims drift (e.g., legacy runs).
        expected_dim = int(mu_obj.size(-1))
    if weights is None or len(weights) != expected_dim:
        w = np.ones(expected_dim, dtype=np.float64)
    else:
        w = np.asarray([float(x) for x in weights], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    w_sum = float(w.sum()) if float(w.sum()) > 0 else 1.0

    df = _read_smiles_csv(dataset_csv_path)
    if df.empty or "dock_score" not in df.columns:
        return []

    # Build signed objective arrays for percentile reference.
    # For dock, restrict to "valid" docks (e.g., <= 0.0) to avoid failure artifacts skewing ranks.
    dock_raw = pd.to_numeric(df.get("dock_score"), errors="coerce").to_numpy(dtype=np.float64)
    mask_valid = np.isfinite(dock_raw)
    if dock_valid_max is not None:
        mask_valid &= dock_raw <= float(dock_valid_max)
    # Require QED/SA only if those dims are active in scoring (w>0).
    qed_raw_all = pd.to_numeric(df.get("qed", df.get("QED")), errors="coerce").to_numpy(dtype=np.float64)
    if w.size >= 2 and w[1] > 0:
        mask_valid &= np.isfinite(qed_raw_all)
    sa_raw_all = None
    if use_sa and base_dim >= 3:
        sa_raw_all = pd.to_numeric(df.get("sa_score"), errors="coerce").to_numpy(dtype=np.float64)
        if w[2] > 0:
            mask_valid &= np.isfinite(sa_raw_all)

    signed_cols: list[np.ndarray] = []
    signed_cols.append(dock_raw * float(dock_sign))
    signed_cols.append(qed_raw_all * float(qed_sign))
    if use_sa and base_dim >= 3:
        signed_cols.append(sa_raw_all * float(sa_sign if sa_sign is not None else -1.0))

    extra_sign_map = {
        "sim_prior": float(sim_prior_sign),
        "win": float(win_sign),
        "hb": float(hb_sign),
        "fsp3": float(fsp3_sign),
    }
    for name in extra_objectives[: max(0, expected_dim - base_dim)]:
        col = pd.to_numeric(df.get(name), errors="coerce").to_numpy(dtype=np.float64)
        if w[len(signed_cols)] > 0:
            mask_valid &= np.isfinite(col)
        signed_cols.append(col * extra_sign_map.get(name, 1.0))

    # Percentile reference (sorted) for each dim.
    sorted_ref: list[np.ndarray] = []
    for d in range(expected_dim):
        if w[d] <= 0:
            sorted_ref.append(np.asarray([], dtype=np.float64))
            continue
        vals = signed_cols[d][mask_valid]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            sorted_ref.append(np.asarray([], dtype=np.float64))
        else:
            sorted_ref.append(np.sort(vals))

    # Dataset score distribution (for rank_ds).
    score_all = None
    if mask_valid.any():
        score_parts = []
        for d in range(expected_dim):
            if w[d] <= 0:
                continue
            ref = sorted_ref[d]
            if ref.size == 0:
                part = np.zeros(mask_valid.sum(), dtype=np.float64)
            else:
                x = signed_cols[d][mask_valid]
                part = np.searchsorted(ref, x, side="right") / float(ref.size)
            score_parts.append(w[d] * part)
        if score_parts:
            score_all = np.sum(np.stack(score_parts, axis=0), axis=0) / w_sum

    # Compute per-id score from mu_obj (already signed), then rank.
    per_id: list[tuple[int, float, int]] = []  # (idx_in_batch, score, rank_ds)
    for i in range(min(len(ids), mu_obj.size(0))):
        parts = []
        for d in range(expected_dim):
            if w[d] <= 0:
                continue
            ref = sorted_ref[d]
            if ref.size == 0:
                pct = 0.0
            else:
                v = float(mu_obj[i, d].item())
                pct = float(np.searchsorted(ref, v, side="right") / float(ref.size))
            parts.append(w[d] * pct)
        score = float(sum(parts) / w_sum) if parts else 0.0
        rank_ds = 0
        if score_all is not None and score_all.size > 0:
            rank_ds = int(np.sum(score_all > score)) + 1
        per_id.append((i, score, rank_ds))

    per_id.sort(key=lambda t: t[1], reverse=True)
    per_id = per_id[:top_k]

    rows: list[list[object]] = []
    for rank, (i, score, rank_ds) in enumerate(per_id, start=1):
        dock_raw_i = float(mu_obj[i, 0]) * dock_sign
        pred_raw = ""
        if pred_dock_vals is not None and i < len(pred_dock_vals):
            pred_val = pred_dock_vals[i]
            if pred_val is not None and np.isfinite(pred_val):
                pred_raw = f"{float(pred_val):.4f}"
        qed_raw_i = float(mu_obj[i, 1]) * qed_sign
        out = [
            rank,
            ids[i],
            f"{score:.4f}",
            rank_ds if rank_ds else "",
            f"{dock_raw_i:.4f}",
            pred_raw,
            f"{qed_raw_i:.4f}",
        ]
        # SA (raw, not signed)
        if use_sa and expected_dim >= 3:
            sa_raw = float(mu_obj[i, 2]) * float(sa_sign if sa_sign is not None else -1.0)
            out.append(f"{sa_raw:.4f}")
        # Extra objectives (raw).
        extra_offset = base_dim
        for j, name in enumerate(extra_objectives[: max(0, expected_dim - base_dim)]):
            sign = extra_sign_map.get(name, 1.0)
            raw = float(mu_obj[i, extra_offset + j]) * float(sign)
            out.append(f"{raw:.4f}")
        out.append(smiles[i])
        rows.append(out)
    return rows


def _topk_oracle_by_dock_rows(
    ids: Sequence[str],
    smiles: Sequence[str],
    mu_obj: torch.Tensor,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float | None,
    sa_vals: Sequence[float] | None,
    pred_dock_vals: Sequence[float] | None,
    top_k: int = 5,
) -> list[list[object]]:
    if mu_obj.numel() == 0:
        return []
    dock_raw = (mu_obj[:, 0] * dock_sign).cpu().numpy().tolist()
    qed_raw = (mu_obj[:, 1] * qed_sign).cpu().numpy().tolist()
    ranked = sorted(range(len(dock_raw)), key=lambda i: dock_raw[i])[:top_k]
    rows = []
    for rank, idx in enumerate(ranked, start=1):
        pred_raw = ""
        if pred_dock_vals is not None and idx < len(pred_dock_vals):
            pred_val = pred_dock_vals[idx]
            if pred_val is not None and np.isfinite(pred_val):
                pred_raw = f"{float(pred_val):.4f}"
        sa_raw = ""
        if mu_obj.size(-1) >= 3:
            if sa_vals is not None and idx < len(sa_vals):
                sa_raw = f"{float(sa_vals[idx]):.4f}"
            else:
                sa_raw = f"{float(mu_obj[idx, 2]) * (sa_sign if sa_sign is not None else -1.0):.4f}"
        rows.append([rank, ids[idx], f"{dock_raw[idx]:.4f}", pred_raw, f"{qed_raw[idx]:.4f}", sa_raw, smiles[idx]])
    return rows


def _topk_oracle_pareto_rows(
    ids: Sequence[str],
    smiles: Sequence[str],
    mu_obj: torch.Tensor,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float | None,
    sa_vals: Sequence[float] | None,
    pred_dock_vals: Sequence[float] | None,
    top_k: int = 5,
) -> list[list[object]]:
    if mu_obj.numel() == 0:
        return []
    dock_raw = (mu_obj[:, 0] * dock_sign).cpu().numpy().tolist()
    qed_raw = (mu_obj[:, 1] * qed_sign).cpu().numpy().tolist()
    selected = list(range(len(dock_raw)))
    front = _pareto_front_indices(selected, dock_raw, qed_raw)
    front_sorted = sorted(front, key=lambda i: (dock_raw[i], -qed_raw[i]))
    rows = []
    for rank, idx in enumerate(front_sorted[:top_k], start=1):
        pred_raw = ""
        if pred_dock_vals is not None and idx < len(pred_dock_vals):
            pred_val = pred_dock_vals[idx]
            if pred_val is not None and np.isfinite(pred_val):
                pred_raw = f"{float(pred_val):.4f}"
        sa_raw = ""
        if mu_obj.size(-1) >= 3:
            if sa_vals is not None and idx < len(sa_vals):
                sa_raw = f"{float(sa_vals[idx]):.4f}"
            else:
                sa_raw = f"{float(mu_obj[idx, 2]) * (sa_sign if sa_sign is not None else -1.0):.4f}"
        rows.append([rank, ids[idx], f"{dock_raw[idx]:.4f}", pred_raw, f"{qed_raw[idx]:.4f}", sa_raw, smiles[idx]])
    return rows


def evaluate_oracle_accuracy(
    ids: Sequence[str],
    pred_vals: Sequence[float],
    true_vals: Sequence[float],
) -> dict:
    if not ids or not pred_vals or not true_vals:
        return {
            "matched": 0,
            "missing": len(ids),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
            "kendall": float("nan"),
        }
    n = min(len(pred_vals), len(true_vals))
    preds = np.asarray(pred_vals[:n], dtype=np.float64)
    trues = np.asarray(true_vals[:n], dtype=np.float64)
    if preds.size == 0:
        return {
            "matched": 0,
            "missing": len(ids),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
            "kendall": float("nan"),
        }
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    mae = float(np.mean(np.abs(preds - trues)))
    denom = np.sum((trues - trues.mean()) ** 2)
    r2 = float(1.0 - np.sum((preds - trues) ** 2) / denom) if denom > 0 else float("nan")
    rank_stats = _rank_metrics(trues, preds)
    out = {
        "matched": int(n),
        "missing": int(max(len(ids) - n, 0)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    out.update(rank_stats)
    return out


def _rankdata_avg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    num = np.sum((rx - rx_mean) * (ry - ry_mean))
    den = np.sqrt(np.sum((rx - rx_mean) ** 2) * np.sum((ry - ry_mean) ** 2))
    if den == 0:
        return float("nan")
    return float(num / den)


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)

def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n - 1):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _ndcg_at_k(trues: np.ndarray, preds: np.ndarray, k: int) -> float:
    trues = np.asarray(trues, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    n = trues.size
    if n == 0:
        return float("nan")
    k = max(1, min(int(k), n))
    rel = -trues
    rel = rel - np.min(rel)
    order_pred = np.argsort(preds)
    order_true = np.argsort(trues)
    def dcg(order):
        score = 0.0
        for i, idx in enumerate(order[:k]):
            denom = np.log2(i + 2.0)
            score += float(rel[idx]) / denom
        return score
    dcg_pred = dcg(order_pred)
    dcg_true = dcg(order_true)
    if dcg_true <= 0:
        return float("nan")
    return float(dcg_pred / dcg_true)


def _hit_at_k(trues: np.ndarray, preds: np.ndarray, k: int) -> float:
    trues = np.asarray(trues, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    n = trues.size
    if n == 0:
        return float("nan")
    k = max(1, min(int(k), n))
    top_true = set(np.argsort(trues)[:k].tolist())
    top_pred = set(np.argsort(preds)[:k].tolist())
    return float(len(top_true & top_pred)) / float(k)


def _rank_metrics(trues: np.ndarray, preds: np.ndarray, ks: Sequence[int] = (10, 50, 100)) -> dict:
    out = {
        "spearman": _spearmanr(trues, preds),
        "pearson": _pearsonr(trues, preds),
        "kendall": _kendall_tau_b(trues, preds),
    }
    for k in ks:
        out[f"hit@{k}"] = _hit_at_k(trues, preds, k)
        out[f"ndcg@{k}"] = _ndcg_at_k(trues, preds, k)
    return out


def evaluate_oracle_accuracy_from_csv(
    csv_path: str,
    ids: Sequence[str],
    pred_vals: Sequence[float],
    dock_valid_max: float | None = 0.0,
) -> dict:
    df = _read_smiles_csv(csv_path)
    if df.empty or "ligand_id" not in df.columns or "dock_score" not in df.columns:
        raise RuntimeError("evaluate_oracle_accuracy_from_csv: missing required columns or empty CSV.")

    df_idx = df.copy()
    df_idx["ligand_id"] = df_idx["ligand_id"].astype(str)
    df_idx = df_idx.set_index("ligand_id", drop=False)

    matched_ids: list[str] = []
    true_vals: list[float] = []
    matched_pred: list[float] = []

    missing_ids: list[str] = []
    invalid_dock_ids: list[str] = []
    invalid_pred_ids: list[str] = []

    for lig_id, pred in zip(ids, pred_vals):
        lig_id = str(lig_id)

        if lig_id not in df_idx.index:
            missing_ids.append(lig_id)
            continue

        row = df_idx.loc[lig_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        raw_dock = row.get("dock_score")
        try:
            dock = float(raw_dock)
        except Exception:
            invalid_dock_ids.append(lig_id)
            continue

        if not np.isfinite(dock):
            invalid_dock_ids.append(lig_id)
            continue

        if not _is_valid_dock_score(dock, dock_valid_max=dock_valid_max):
            invalid_dock_ids.append(lig_id)
            continue

        if pred is None or not np.isfinite(pred):
            invalid_pred_ids.append(lig_id)
            continue

        matched_ids.append(lig_id)
        true_vals.append(float(dock))
        matched_pred.append(float(pred))

    stats = evaluate_oracle_accuracy(matched_ids, matched_pred, true_vals)
    stats["total"] = int(len(ids))
    stats["matched"] = int(len(matched_ids))
    stats["missing_csv"] = int(len(missing_ids))
    stats["invalid_dock"] = int(len(invalid_dock_ids))
    stats["invalid_pred"] = int(len(invalid_pred_ids))
    stats["skipped"] = int(len(missing_ids) + len(invalid_dock_ids) + len(invalid_pred_ids))
    return stats
