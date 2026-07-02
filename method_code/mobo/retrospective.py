from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from mobo.metrics import DominatedPartitioning, bt_is_non_dominated
from mobo.smiles_utils import canonicalize_smiles_noh, compute_qed, compute_sa


POOL_COLUMNS = [
    "smiles",
    "smiles_canonical",
    "dock_score",
    "dock_score_median",
    "dock_score_mean",
    "dock_score_min",
    "dock_score_max",
    "dock_obs",
    "qed",
    "sa_score",
    "ingredient_name",
    "mol_id",
    "source_runs",
    "source_count",
    "is_reference",
]


def build_archived_oracle_pool(
    csv_paths: Sequence[Path],
    out_csv: Path | None = None,
    out_meta: Path | None = None,
) -> pd.DataFrame:
    if not csv_paths:
        raise RuntimeError("No archived smiles.csv paths provided.")
    records = []
    bad_rows = 0
    for csv_path in csv_paths:
        run_name = csv_path.parents[1].name
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        for row in df.to_dict(orient="records"):
            smiles = str(row.get("smiles") or row.get("smiles_canonical") or "").strip()
            smiles_canonical = str(row.get("smiles_canonical") or "").strip()
            if not smiles and not smiles_canonical:
                continue
            dock_raw = row.get("dock_score")
            try:
                dock_score = float(dock_raw)
            except Exception:
                bad_rows += 1
                continue
            if not math.isfinite(dock_score):
                bad_rows += 1
                continue
            if not smiles_canonical:
                smiles_canonical = canonicalize_smiles_noh(smiles)
            if not smiles:
                smiles = smiles_canonical
            qed = row.get("qed")
            sa = row.get("sa_score")
            try:
                qed_val = float(qed)
            except Exception:
                qed_val = float(compute_qed(smiles_canonical))
            try:
                sa_val = float(sa)
            except Exception:
                sa_val = float(compute_sa(smiles_canonical))
            records.append({
                "smiles": str(row.get("smiles") or smiles_canonical),
                "smiles_canonical": smiles_canonical,
                "dock_score": dock_score,
                "qed": qed_val,
                "sa_score": sa_val,
                "ingredient_name": str(row.get("ingredient_name", "")).strip(),
                "mol_id": str(row.get("mol_id", "")).strip(),
                "run_name": run_name,
                "is_reference": int(str(row.get("is_reference", "0")).strip() in {"1", "1.0", "true", "True"}),
            })
    if not records:
        raise RuntimeError("No valid archived docking rows found.")
    raw = pd.DataFrame(records)
    grouped_rows = []
    for smiles_canonical, sub in raw.groupby("smiles_canonical", sort=False):
        docks = sub["dock_score"].to_numpy(dtype=np.float64)
        ingredient_name = next((x for x in sub["ingredient_name"].tolist() if x), "")
        mol_id = next((x for x in sub["mol_id"].tolist() if x), "")
        source_runs = sorted(set(sub["run_name"].tolist()))
        grouped_rows.append({
            "smiles": str(sub.iloc[0]["smiles"]),
            "smiles_canonical": smiles_canonical,
            "dock_score": float(np.median(docks)),
            "dock_score_median": float(np.median(docks)),
            "dock_score_mean": float(np.mean(docks)),
            "dock_score_min": float(np.min(docks)),
            "dock_score_max": float(np.max(docks)),
            "dock_obs": int(docks.size),
            "qed": float(np.mean(sub["qed"].to_numpy(dtype=np.float64))),
            "sa_score": float(np.mean(sub["sa_score"].to_numpy(dtype=np.float64))),
            "ingredient_name": ingredient_name,
            "mol_id": mol_id,
            "source_runs": ";".join(source_runs),
            "source_count": int(len(source_runs)),
            "is_reference": int(sub["is_reference"].max()),
        })
    pool = pd.DataFrame(grouped_rows)
    pool = pool.loc[pool["is_reference"].astype(int) == 0].copy().reset_index(drop=True)
    pool = pool.sort_values(["dock_score", "smiles_canonical"], ascending=[True, True]).reset_index(drop=True)
    for col in POOL_COLUMNS:
        if col not in pool.columns:
            pool[col] = ""
    pool = pool[POOL_COLUMNS].copy()
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(out_csv, index=False)
    if out_meta is not None:
        meta = {
            "n_input_csv": int(len(csv_paths)),
            "n_raw_rows": int(len(records)),
            "n_unique_molecules": int(pool.shape[0]),
            "bad_rows": int(bad_rows),
            "source_runs": [str(p.parents[1].name) for p in csv_paths],
        }
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return pool


def stratified_holdout_split(
    df: pd.DataFrame,
    holdout_frac: float,
    seed: int,
    stratify_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < float(holdout_frac) < 1.0:
        raise ValueError(f"holdout_frac must be in (0,1), got {holdout_frac}")
    rng = random.Random(int(seed))
    work = df.copy().reset_index(drop=True)
    ranks = pd.qcut(work["dock_score"], q=min(int(stratify_bins), max(2, work.shape[0] // 20)), duplicates="drop")
    work["_bin"] = ranks.astype(str)
    holdout_idx = []
    active_idx = []
    for _, sub in work.groupby("_bin", sort=False):
        idxs = sub.index.tolist()
        rng.shuffle(idxs)
        n_holdout = max(1, int(round(len(idxs) * float(holdout_frac))))
        holdout_idx.extend(idxs[:n_holdout])
        active_idx.extend(idxs[n_holdout:])
    holdout = work.loc[sorted(holdout_idx)].drop(columns=["_bin"]).reset_index(drop=True)
    active = work.loc[sorted(active_idx)].drop(columns=["_bin"]).reset_index(drop=True)
    if holdout.empty or active.empty:
        raise RuntimeError("Holdout split produced an empty partition.")
    return active, holdout


def write_round_dataset(
    labeled_df: pd.DataFrame,
    out_dir: Path,
    valid_frac: float,
    seed: int,
    smiles_col: str = "smiles_canonical",
) -> Path:
    if labeled_df.empty:
        raise RuntimeError("Cannot write retrospective dataset with no labeled rows.")
    if smiles_col not in labeled_df.columns:
        raise ValueError(f"Missing smiles column: {smiles_col}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    rows = labeled_df.copy().reset_index(drop=True)
    idxs = list(range(rows.shape[0]))
    rng.shuffle(idxs)
    n_valid = max(1, int(round(rows.shape[0] * float(valid_frac))))
    valid_idx = set(idxs[:n_valid])
    export_rows = []
    for i, row in rows.iterrows():
        export_rows.append({
            "smiles": str(row.get("smiles") or row.get(smiles_col)),
            "smiles_canonical": str(row.get("smiles_canonical") or row.get(smiles_col)),
            "qed": float(row["qed"]),
            "sa_score": float(row["sa_score"]),
            "ligand_id": str(row.get("ligand_id") or f"POOL_{i + 1:05d}"),
            "is_reference": 0,
            "dock_score": float(row["dock_score"]),
            "split": "valid" if i in valid_idx else "train",
            "added_iter": int(row.get("added_iter", 0)),
        })
    out_csv = out_dir / "smiles.csv"
    pd.DataFrame(export_rows).to_csv(out_csv, index=False)
    return out_dir


def build_objective_tensor(df: pd.DataFrame, dock_sign: float, qed_sign: float, sa_sign: float) -> torch.Tensor:
    vals = np.stack([
        dock_sign * df["dock_score"].to_numpy(dtype=np.float32),
        qed_sign * df["qed"].to_numpy(dtype=np.float32),
        sa_sign * df["sa_score"].to_numpy(dtype=np.float32),
    ], axis=1)
    return torch.tensor(vals, dtype=torch.float32)


def compute_hypervolume(points: torch.Tensor, ref_point: Sequence[float]) -> float:
    if points.numel() == 0:
        return 0.0
    if points.dim() != 2:
        raise ValueError(f"points must be (N,D), got {tuple(points.shape)}")
    front = points[bt_is_non_dominated(points)]
    if front.numel() == 0:
        return 0.0
    ref = torch.tensor(ref_point, dtype=front.dtype, device=front.device)
    return float(DominatedPartitioning(ref, front).compute_hypervolume().item())


def _rankdata_avg(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float64).cpu()
    n = int(values.numel())
    if n == 0:
        return values
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(n, dtype=torch.float64)
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


def spearmanr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    num = float(((rx - rx_mean) * (ry - ry_mean)).sum().item())
    den = float(torch.sqrt(((rx - rx_mean) ** 2).sum() * ((ry - ry_mean) ** 2).sum()).item())
    if den == 0.0:
        return float("nan")
    return num / den


def kendall_tau_b_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x_np = x.detach().to(torch.float64).cpu().numpy()
    y_np = y.detach().to(torch.float64).cpu().numpy()
    if x_np.size < 2 or y_np.size < 2:
        return float("nan")
    tau = kendalltau(x_np, y_np, variant="b", nan_policy="omit").statistic
    if tau is None or not math.isfinite(float(tau)):
        return float("nan")
    return float(tau)


def ndcg_at_k(trues: torch.Tensor, preds: torch.Tensor, k: int) -> float:
    trues = trues.to(torch.float64).cpu()
    preds = preds.to(torch.float64).cpu()
    n = int(trues.numel())
    if n == 0:
        return float("nan")
    k = max(1, min(int(k), n))
    rel = -trues
    rel = rel - rel.min()
    order_pred = torch.argsort(preds)
    order_true = torch.argsort(trues)
    def dcg(order):
        score = 0.0
        for i, idx in enumerate(order[:k]):
            score += float(rel[idx]) / math.log2(i + 2.0)
        return score
    dcg_pred = dcg(order_pred)
    dcg_true = dcg(order_true)
    if dcg_true <= 0.0:
        return float("nan")
    return float(dcg_pred / dcg_true)


def hit_at_k(trues: torch.Tensor, preds: torch.Tensor, k: int) -> float:
    trues = trues.to(torch.float64).cpu()
    preds = preds.to(torch.float64).cpu()
    n = int(trues.numel())
    if n == 0:
        return float("nan")
    k = max(1, min(int(k), n))
    top_true = set(torch.argsort(trues)[:k].tolist())
    top_pred = set(torch.argsort(preds)[:k].tolist())
    return float(len(top_true & top_pred)) / float(k)


def rank_metrics(trues: torch.Tensor, preds: torch.Tensor, ks: Sequence[int] = (10, 50, 100)) -> dict[str, float]:
    out = {
        "spearman": spearmanr_torch(trues, preds),
        "kendall": kendall_tau_b_torch(trues, preds),
    }
    for k in ks:
        out[f"hit@{int(k)}"] = hit_at_k(trues, preds, int(k))
        out[f"ndcg@{int(k)}"] = ndcg_at_k(trues, preds, int(k))
    return out


def build_train_distance_scores(
    train_smiles: Sequence[str],
    candidate_smiles: Sequence[str],
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(fp_radius), fpSize=int(fp_bits))
    train_fps = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid train SMILES for distance score: {smi}")
        train_fps.append(gen.GetFingerprint(mol))
    if not train_fps:
        raise RuntimeError("No train fingerprints available for distance scoring.")
    scores = np.zeros(len(candidate_smiles), dtype=np.float32)
    for i, smi in enumerate(candidate_smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid candidate SMILES for distance score: {smi}")
        fp = gen.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        scores[i] = float(1.0 - max(sims)) if sims else 0.0
    return scores
