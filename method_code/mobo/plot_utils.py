from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from mobo.io_utils import _read_smiles_csv


def _as_float(val):
    try:
        num = float(val)
    except Exception:
        raise ValueError(f"Failed to parse float value: {val!r}")
    if not math.isfinite(num):
        raise ValueError(f"Non-finite float value: {val!r}")
    return num


def plot_run_artifacts(
    run_dir: Path,
    dataset_root: Path,
    metrics_path: Path,
    max_points: int = 5000,
    dock_abs_max: float | None = 100.0,
    dpi: int = 150,
    seed: int = 42,
) -> None:
    import matplotlib.pyplot as plt

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        with metrics_path.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            iters = [int(float(r.get("iteration", i + 1))) for i, r in enumerate(rows)]
            hv_after = [_as_float(r.get("hv_after")) for r in rows]
            hvi = [_as_float(r.get("hvi")) for r in rows]
            pareto = [_as_float(r.get("pareto_count")) for r in rows]
            missing = [_as_float(r.get("oracle_missing")) for r in rows]
            rmse = [_as_float(r.get("rmse")) for r in rows]
            mae = [_as_float(r.get("mae")) for r in rows]
            r2 = [_as_float(r.get("r2")) for r in rows]

            fig, axes = plt.subplots(3, 2, figsize=(12, 12))
            axes = axes.flatten()
            axes[0].plot(iters, [v if v is not None else float("nan") for v in hv_after], marker="o")
            axes[0].set_title("HV After vs Iter")
            axes[0].set_xlabel("Iter")
            axes[0].set_ylabel("HV After")

            axes[1].plot(iters, [v if v is not None else float("nan") for v in hvi], marker="o")
            axes[1].set_title("HVI vs Iter")
            axes[1].set_xlabel("Iter")
            axes[1].set_ylabel("HVI")

            axes[2].plot(iters, [v if v is not None else float("nan") for v in pareto], marker="o")
            axes[2].set_title("Pareto Count vs Iter")
            axes[2].set_xlabel("Iter")
            axes[2].set_ylabel("Pareto Count")

            axes[3].plot(iters, [v if v is not None else float("nan") for v in missing], marker="o")
            axes[3].set_title("Oracle Missing vs Iter")
            axes[3].set_xlabel("Iter")
            axes[3].set_ylabel("Missing Dock")

            axes[4].plot(iters, [v if v is not None else float("nan") for v in rmse], marker="o", label="RMSE")
            axes[4].plot(iters, [v if v is not None else float("nan") for v in mae], marker="o", label="MAE")
            axes[4].legend()
            axes[4].set_title("RMSE/MAE vs Iter")
            axes[4].set_xlabel("Iter")
            axes[4].set_ylabel("Error")

            axes[5].plot(iters, [v if v is not None else float("nan") for v in r2], marker="o")
            axes[5].set_title("R2 vs Iter")
            axes[5].set_xlabel("Iter")
            axes[5].set_ylabel("R2")

            fig.tight_layout()
            fig.savefig(plots_dir / "metrics_overview.png", dpi=dpi)
            plt.close(fig)

    smiles_path = dataset_root / "smiles.csv"
    if not smiles_path.exists():
        raise FileNotFoundError(f"Missing smiles.csv for plotting: {smiles_path}")
    df = _read_smiles_csv(str(smiles_path))
    if df.empty:
        raise RuntimeError(f"smiles.csv is empty: {smiles_path}")
    cols = df.columns
    dock_col = "dock_score" if "dock_score" in cols else ("vina_score" if "vina_score" in cols else None)
    qed_col = "qed" if "qed" in cols else ("QED" if "QED" in cols else None)
    sa_col = "sa_score" if "sa_score" in cols else None
    lig_col = "ligand_id" if "ligand_id" in cols else None
    if dock_col is None or qed_col is None:
        raise RuntimeError(f"smiles.csv missing dock/qed columns: {smiles_path}")

    dock, qed, sa, label = [], [], [], []
    for _, row in df.iterrows():
        d = _as_float(row.get(dock_col))
        q = _as_float(row.get(qed_col))
        if d is None or q is None:
            raise RuntimeError("Encountered non-finite dock/qed while plotting run artifacts.")
        if dock_abs_max is not None and abs(d) > dock_abs_max:
            continue
        dock.append(float(d))
        qed.append(float(q))
        sa.append(_as_float(row.get(sa_col)) if sa_col else None)
        lig = str(row.get(lig_col, "")) if lig_col else ""
        label.append("GEN" if lig.startswith("GEN_") else "LIG")

    n = len(dock)
    if n == 0:
        raise RuntimeError("No valid points available for oracle_scatter plotting.")
    if n > max_points:
        random.seed(seed)
        idx = random.sample(range(n), max_points)
        dock = [dock[i] for i in idx]
        qed = [qed[i] for i in idx]
        sa = [sa[i] for i in idx]
        label = [label[i] for i in idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ["#1f77b4" if l == "GEN" else "#ff7f0e" for l in label]
    ax = axes[0, 0]
    ax.scatter(dock, qed, s=8, c=colors, alpha=0.6)
    ax.set_title("Dock vs QED")
    ax.set_xlabel("Dock Score")
    ax.set_ylabel("QED")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", label="GEN", markerfacecolor="#1f77b4", markersize=6),
            plt.Line2D([0], [0], marker="o", color="w", label="LIG", markerfacecolor="#ff7f0e", markersize=6),
        ],
        loc="best",
        frameon=False,
    )

    ax = axes[0, 1]
    sa_vals = [v for v in sa if v is not None and math.isfinite(v)]
    if sa_vals:
        ax.hist(sa_vals, bins=40, color="#2ca02c", alpha=0.7)
        ax.set_title("SA Distribution")
        ax.set_xlabel("SA Score")
        ax.set_ylabel("Count")
    else:
        ax.set_title("SA Distribution (no data)")

    ax = axes[1, 0]
    if sa_vals:
        ax.scatter(dock, sa, s=8, c=colors, alpha=0.6)
        ax.set_title("Dock vs SA")
        ax.set_xlabel("Dock Score")
        ax.set_ylabel("SA Score")
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w", label="GEN", markerfacecolor="#1f77b4", markersize=6),
                plt.Line2D([0], [0], marker="o", color="w", label="LIG", markerfacecolor="#ff7f0e", markersize=6),
            ],
            loc="best",
            frameon=False,
        )
    else:
        ax.set_title("Dock vs SA (no data)")

    ax = axes[1, 1]
    if sa_vals:
        ax.scatter(qed, sa, s=8, c=colors, alpha=0.6)
        ax.set_title("QED vs SA")
        ax.set_xlabel("QED")
        ax.set_ylabel("SA Score")
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w", label="GEN", markerfacecolor="#1f77b4", markersize=6),
                plt.Line2D([0], [0], marker="o", color="w", label="LIG", markerfacecolor="#ff7f0e", markersize=6),
            ],
            loc="best",
            frameon=False,
        )
    else:
        ax.set_title("QED vs SA (no data)")

    fig.tight_layout()
    fig.savefig(plots_dir / "oracle_scatter.png", dpi=dpi)
    plt.close(fig)
