#!/usr/bin/env python3
"""Plot constrained vs unconstrained candidate set size ratio from evaluate_molecules CSV.

Reads results/molecule_evaluation_test.csv (or --csv path).
Filters to paired rows where both solves completed without timeout.

Usage
-----
    python plot_set_size_ratio.py
    python plot_set_size_ratio.py --csv results/my_run/molecule_evaluation_test.csv --output results/ratio.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import pandas as pd

DEFAULT_CSV = "eval_results/synthetic-ei/mlp/molecule_eval/molecule_evaluation_test.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    # Paired rows: constrained succeeded, unconstrained completed with candidates
    paired = df[
        df["status"].isin(["ok", "empty_network_error"]) &
        ~df["unc_timed_out"].astype(bool) &
        (df["unc_num_candidates"] > 0)
    ].copy()

    print(f"Total rows: {len(df)}  |  Paired (both solved): {len(paired)}")

    ratio = paired["num_candidates"] / paired["unc_num_candidates"]
    print(f"Ratio con/unc  —  mean: {ratio.mean():.3f}  median: {ratio.median():.3f}  "
          f"min: {ratio.min():.3f}  max: {ratio.max():.3f}")

    exact = paired["exact_match"].astype(bool)
    ratio_exact = ratio[exact]
    ratio_no_exact = ratio[~exact]

    print(f"Exact match present : {exact.sum()}  |  GT excluded : {(~exact).sum()}")

    unc = paired["unc_num_candidates"].values
    con = paired["num_candidates"].values

    # Scatter: one point per sample, x=unc size, y=con size, log+1 scale
    # Color: GT present vs excluded
    fig, ax = plt.subplots(figsize=(7, 6))
    fig2, ax_ratio = plt.subplots(figsize=(7, 5))

    lim = max(unc.max(), con.max()) + 5

    # y=x diagonal (no reduction)
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1.0,
            zorder=0, label="y = x (no reduction)")

    ax.scatter(
        unc[exact.values], con[exact.values],
        s=30, alpha=0.7, color="steelblue", zorder=2,
        label=f"GT in constrained set (n={exact.sum()})",
    )
    ax.scatter(
        unc[~exact.values], con[~exact.values],
        s=30, alpha=0.8, color="crimson", zorder=3, marker="x",
        label=f"GT excluded from constrained set (n={(~exact).sum()})",
    )

    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.set_xlabel("Unconstrained candidate set size")
    ax.set_ylabel("Constrained candidate set size")
    #ax.set_title(
        #f"Constrained (EI-MLP) vs unconstrained candidate set size\n"
        #f"n={len(paired)} paired samples (symlog scale)"
    #)
    ax.legend()
    fig.tight_layout()

    # Second figure: ratio histogram broken down by exact match
    bins_ratio = np.linspace(0, 1, 31)
    ax_ratio.hist(ratio_exact, bins=bins_ratio, alpha=0.7, color="steelblue",
                  edgecolor="white", linewidth=0.5, label=f"GT in constrained set (n={exact.sum()})")
    ax_ratio.hist(ratio_no_exact, bins=bins_ratio, alpha=0.7, color="crimson",
                  edgecolor="white", linewidth=0.5, label=f"GT excluded (n={(~exact).sum()})")
    ax_ratio.axvline(ratio.mean(), color="black", linestyle="--", linewidth=1.0,
                     label=f"Overall mean = {ratio.mean():.3f}")
    ax_ratio.set_xlabel("Constrained / unconstrained set size")
    ax_ratio.set_ylabel("Count")
    #ax_ratio.set_title("Candidate set size ratio (constrained vs unconstrained; EI-MLP)")
    ax_ratio.legend()
    fig2.tight_layout()

    if args.output:
        out = Path(args.output)
        scatter_path = out.with_stem(out.stem + "_scatter")
        ratio_path = out.with_stem(out.stem + "_ratio")
        fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
        fig2.savefig(ratio_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {scatter_path} and {ratio_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
