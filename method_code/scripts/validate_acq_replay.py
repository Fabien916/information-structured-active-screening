"""Replay acquisition rules on frozen per-round posteriors.

For each frozen (target, round, source-trajectory) state, each acquisition
rule scores the same posterior, pool and incumbent set. The script then
selects a top-K batch and evaluates the realized hypervolume gain.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_dockstring_experiments import (
    mean_hvi_gain, true_single_candidate_hvi_gain, mc_objective_samples,
    qnehvi_scores_from_samples_fast_3d, qnparego_scores_from_samples_nd, eval_batch,
)
from mobo.analytic_hvi_fast import nehvi_gaussian_analytic_3d, qphv_prob_gaussian_analytic_3d
from mobo.retrospective import build_objective_tensor

DEFAULT_REPLAY_ROOT = REPO / "runs" / "dockstring_benchmark" / "virtual_loop_scratch_gaussianens4_sharedpool_v4_alltargets_seed42"
TARGETS = ["ESR2", "F2", "JAK2", "KIT", "PARP1"]
SRC = ["analytic_ehvi", "analytic_pomhi"]
DOCK_SIGN, QED_SIGN, SA_SIGN = -1.0, 1.0, -1.0
WEIGHTS = [1.0, 1.0, 1.0]
REF = [0.0, 0.0, -20.0]
TOPK = 100
MC_SAMPLES = 128
NPAREGO = 8


def sequential_mean_hvi_scores(pool: pd.DataFrame, inc: pd.DataFrame, batch_size: int) -> np.ndarray:
    """Batch-aware deterministic exploitation using posterior-mean objectives.

    The selected candidate is appended to a temporary front with its predicted
    docking mean as the docking coordinate, then remaining candidates are
    rescored. The returned array is a ranking score used only to recover the
    sequentially selected batch through the common top-k path below.
    """
    remaining = np.arange(len(pool), dtype=np.int64)
    scores = np.full(len(pool), -np.inf, dtype=np.float64)
    inc_aug = inc.copy()
    n_select = min(int(batch_size), int(len(pool)))
    for rank in range(n_select):
        gains = mean_hvi_gain(
            pool.iloc[remaining],
            inc_aug,
            DOCK_SIGN,
            QED_SIGN,
            SA_SIGN,
            WEIGHTS,
            REF,
        )
        local_pos = int(np.argmax(gains))
        chosen = int(remaining[local_pos])
        scores[chosen] = float(n_select - rank)

        added = pool.iloc[[chosen]].copy()
        added["dock_score"] = added["pred_dock_mean"]
        inc_aug = pd.concat([inc_aug, added], ignore_index=True)
        remaining = np.delete(remaining, local_pos)
    return scores

def score_rules(pool: pd.DataFrame, inc: pd.DataFrame, seed: int, include_sequential_mean_hvi: bool) -> dict[str, np.ndarray]:
    y_train = build_objective_tensor(inc, dock_sign=DOCK_SIGN, qed_sign=QED_SIGN, sa_sign=SA_SIGN)
    exact = torch.stack([QED_SIGN*torch.tensor(pool["qed"].to_numpy(np.float32)),
                         SA_SIGN*torch.tensor(pool["sa_score"].to_numpy(np.float32))], dim=-1)
    dmu = DOCK_SIGN*torch.tensor(pool["pred_dock_mean"].to_numpy(np.float32))
    dsig = torch.tensor(pool["pred_dock_std"].to_numpy(np.float32))
    out = {}
    out["greedy_mean"] = mean_hvi_gain(pool, inc, DOCK_SIGN, QED_SIGN, SA_SIGN, WEIGHTS, REF)
    if include_sequential_mean_hvi:
        out["sequential_mean_hvi"] = sequential_mean_hvi_scores(pool, inc, TOPK)
    out["analytic_ehvi"] = nehvi_gaussian_analytic_3d(dock_mu=dmu, dock_sigma=dsig, exact_obj=exact, y_train=y_train, weights=WEIGHTS, ref_point=REF, return_metadata=True)[0].cpu().numpy()
    out["analytic_pomhi"] = qphv_prob_gaussian_analytic_3d(dock_mu=dmu, dock_sigma=dsig, exact_obj=exact, y_train=y_train, weights=WEIGHTS, ref_point=REF, return_metadata=True)[0].cpu().numpy()
    s_nehvi = mc_objective_samples(pool, MC_SAMPLES, DOCK_SIGN, QED_SIGN, SA_SIGN, seed+11)
    out["qnehvi_mc"] = qnehvi_scores_from_samples_fast_3d(s_nehvi, y_train=y_train, weights=WEIGHTS, ref_point=REF).cpu().numpy()
    s_par = mc_objective_samples(pool, MC_SAMPLES, DOCK_SIGN, QED_SIGN, SA_SIGN, seed+37)
    out["qnparego_mc"] = qnparego_scores_from_samples_nd(s_par, y_train=y_train, weights=WEIGHTS, scalarizations=NPAREGO, seed=seed+41).cpu().numpy()
    rng = np.random.default_rng(seed); out["random"] = rng.random(len(pool))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-sequential-mean-hvi",
        action="store_true",
        help="Add the deterministic batch-aware sequential mean-HVI baseline. This is much slower because it rescores after every selected molecule.",
    )
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=DEFAULT_REPLAY_ROOT,
        help="Frozen Dockstring replay root containing TARGET/RULE/roundXX_candidate_predictions.csv tables.",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Limit the number of frozen target/source/round states. Useful for timing or smoke checks.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "paper_data/dockstring_virtual_loop/acq_replay_validation.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay_root = args.replay_root if args.replay_root.is_absolute() else REPO / args.replay_root
    states = []
    for tgt in TARGETS:
        for src in SRC:
            for r in range(1, 11):
                states.append((tgt, src, r))
    if args.max_states is not None:
        if args.max_states <= 0:
            raise RuntimeError("--max-states must be positive when provided.")
        states = states[: int(args.max_states)]

    rows = []
    for state_idx, (tgt, src, r) in enumerate(states, start=1):
        base = replay_root / tgt / src
        pf = base / f"round{r:02d}_candidate_predictions.csv"
        inf = base / f"round{r:02d}_dataset" / "smiles.csv"
        if not pf.exists() or not inf.exists():
            print(f"skip {tgt}/{src}/r{r}: missing")
            continue
        print(f"state {state_idx}/{len(states)} {tgt}/{src}/r{r}")
        pool = pd.read_csv(pf); inc = pd.read_csv(inf)
        # ground-truth single-candidate HVI (exact descriptors + TRUE dock)
        true_g = true_single_candidate_hvi_gain(pool, inc, DOCK_SIGN, QED_SIGN, SA_SIGN, WEIGHTS, REF)
        k_true = np.argsort(-true_g)[:TOPK]
        scores = score_rules(
            pool,
            inc,
            seed=1000+r,
            include_sequential_mean_hvi=bool(args.include_sequential_mean_hvi),
        )
        for rule, sc in scores.items():
            pick = np.argsort(-sc)[:TOPK]
            batch = pool.iloc[pick]
            ev = eval_batch(inc, batch, DOCK_SIGN, QED_SIGN, SA_SIGN, REF, -10.0, 0.5, 3.0)
            recall = len(set(pick.tolist()) & set(k_true.tolist())) / TOPK
            rows.append(dict(target=tgt, src=src, round=r, n_labeled=len(inc), rule=rule,
                             true_hv_gain=ev["hv_gained"], topk_true_hvi_recall=recall,
                             sum_true_hvi_selected=float(true_g[pick].sum())))
    df = pd.DataFrame(rows)
    outp = args.out if args.out.is_absolute() else (REPO / args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False); print("wrote", outp, "rows", len(df))

    def band(d):
        return d.groupby("rule").agg(hv=("true_hv_gain","mean"), recall=("topk_true_hvi_recall","mean")).round(3)
    print("\n=== ALL rounds: mean true HV gain & top-100 recall per rule ===")
    print(band(df).sort_values("hv", ascending=False))
    print("\n=== EARLY rounds 1-3 (sparse Pareto; where uncertainty should matter most) ===")
    print(band(df[df["round"]<=3]).sort_values("hv", ascending=False))
    print("\n=== LATE rounds 8-10 (near ceiling) ===")
    print(band(df[df["round"]>=8]).sort_values("hv", ascending=False))
    # headline deltas vs greedy, paired by state
    piv = df.pivot_table(index=["target","src","round"], columns="rule", values="true_hv_gain")
    print("\n=== paired delta (rule - greedy_mean) true HV gain, mean [boot 95% CI] ===")
    comparison_rules = ["analytic_ehvi","analytic_pomhi","qnehvi_mc","qnparego_mc"]
    if args.include_sequential_mean_hvi:
        comparison_rules.insert(0, "sequential_mean_hvi")
    for rule in comparison_rules:
        d = (piv[rule]-piv["greedy_mean"]).dropna().to_numpy()
        bs = np.array([np.mean(np.random.default_rng(s).choice(d, len(d))) for s in range(2000)])
        win = float((d>0).mean())
        print(f"  {rule:16s} mean={d.mean():+.3f}  CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  win_rate={win:.0%}  (n={len(d)})")
    print("\nINTERPRETATION: paired positive deltas indicate higher realized batch hypervolume gain under the same frozen posterior states.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
