"""Summarize Dockstring closed-loop sample-efficiency metrics.

The input is the NIG matched-control trajectory table used by the NC manuscript.
For each method, this script reports terminal HV, mean round-wise HV gain over
greedy-mean, and the number of labelled evaluations needed to reach the matched
greedy-mean terminal HV within each target-seed block.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "paper_data"
    / "dockstring_virtual_loop"
    / "nig_placeholder_3targets_3seeds"
    / "trajectories.csv"
)
OUTPUT = ROOT / "paper_data" / "dockstring_virtual_loop" / "dockstring_budget_efficiency.csv"


def main() -> None:
    df = pd.read_csv(INPUT)
    required = {"seed", "target", "method", "round", "hv", "labeled_n"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT}: {sorted(missing)}")

    key_round = ["seed", "target", "round"]
    greedy_round = (
        df.loc[df["method"] == "greedy_mean", key_round + ["hv"]]
        .rename(columns={"hv": "greedy_round_hv"})
        .copy()
    )
    with_round = df.merge(greedy_round, on=key_round, how="left", validate="many_to_one")
    with_round["delta_vs_greedy_round_hv"] = (
        with_round["hv"] - with_round["greedy_round_hv"]
    )

    terminal = df.loc[df["round"] == 10].copy()
    greedy_terminal = (
        terminal.loc[terminal["method"] == "greedy_mean", ["seed", "target", "hv"]]
        .rename(columns={"hv": "greedy_final_hv"})
        .copy()
    )
    terminal = terminal.merge(
        greedy_terminal, on=["seed", "target"], how="left", validate="many_to_one"
    )
    terminal["delta_vs_greedy_final_hv"] = terminal["hv"] - terminal["greedy_final_hv"]
    terminal["beats_greedy_final_hv"] = terminal["delta_vs_greedy_final_hv"] > 0

    hit_rows = []
    with_terminal = df.merge(
        greedy_terminal, on=["seed", "target"], how="left", validate="many_to_one"
    )
    for (seed, target, method), sub in with_terminal.groupby(["seed", "target", "method"]):
        sub = sub.sort_values("round")
        hit = sub.loc[sub["hv"] >= sub["greedy_final_hv"]]
        if hit.empty:
            hit_round = None
            hit_evals = None
        else:
            first = hit.iloc[0]
            hit_round = int(first["round"])
            hit_evals = int(first["labeled_n"])
        hit_rows.append(
            {
                "seed": seed,
                "target": target,
                "method": method,
                "reaches_greedy_final": hit_round is not None,
                "first_round_at_greedy_final": hit_round,
                "first_labeled_n_at_greedy_final": hit_evals,
            }
        )
    hits = pd.DataFrame(hit_rows)

    final_summary = terminal.groupby("method").agg(
        final_hv_mean=("hv", "mean"),
        final_hv_sd=("hv", "std"),
        final_delta_vs_greedy_mean=("delta_vs_greedy_final_hv", "mean"),
        final_delta_vs_greedy_sd=("delta_vs_greedy_final_hv", "std"),
        wins_vs_greedy=("beats_greedy_final_hv", "sum"),
        blocks=("beats_greedy_final_hv", "size"),
    )
    round_summary = with_round.groupby("method").agg(
        mean_round_delta_vs_greedy=("delta_vs_greedy_round_hv", "mean"),
        sd_round_delta_vs_greedy=("delta_vs_greedy_round_hv", "std"),
    )
    hit_summary = hits.groupby("method").agg(
        reaches_greedy_final_n=("reaches_greedy_final", "sum"),
        median_round_to_greedy_final=("first_round_at_greedy_final", "median"),
        mean_round_to_greedy_final=("first_round_at_greedy_final", "mean"),
        median_labeled_n_to_greedy_final=("first_labeled_n_at_greedy_final", "median"),
        mean_labeled_n_to_greedy_final=("first_labeled_n_at_greedy_final", "mean"),
    )

    out = (
        final_summary.join(round_summary)
        .join(hit_summary)
        .reset_index()
        .sort_values(
            ["final_delta_vs_greedy_mean", "mean_round_delta_vs_greedy"],
            ascending=[False, False],
        )
    )
    out.to_csv(OUTPUT, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
