#!/usr/bin/env python3
"""Plot key metrics vs alpha from the alpha-experiment results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path("eval_results/synthetic-ei/mlp/molecule_eval/alpha-experiment")
OUT_PATH = Path("results/alpha_experiment.pdf")

REFERENCE_ALPHA = 0.05


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for alpha_dir in sorted(results_dir.glob("alpha_*")):
        summary_path = alpha_dir / "molecule_evaluation_summary.json"
        if not summary_path.exists():
            continue
        alpha = float(alpha_dir.name.removeprefix("alpha_"))
        data = json.loads(summary_path.read_text())
        rows.append({"alpha": alpha, **data})
    rows.sort(key=lambda r: r["alpha"])
    return rows


def main() -> None:
    rows = load_results(RESULTS_DIR)
    if not rows:
        raise FileNotFoundError(f"No results found in {RESULTS_DIR}")

    alphas = np.array([r["alpha"] for r in rows])
    n_total = rows[0]["n_evaluated"]

    con = [r["constrained"] for r in rows]
    unc = [r["unconstrained"] for r in rows]

    exact_match_pct   = np.array([c["pct_exact_match"] for c in con])
    mean_tanimoto     = np.array([c["mean_max_tanimoto"] for c in con])
    mean_spt_con      = np.array([c["mean_size_penalised_tanimoto"] for c in con])
    mean_spt_unc      = np.array([u["mean_size_penalised_tanimoto"] for u in unc])
    mean_cands_con    = np.array([c["mean_num_candidates"] for c in con])
    mean_cands_unc    = np.array([u["mean_num_candidates"] for u in unc])
    median_cands_con  = np.array([c["median_num_candidates"] for c in con])
    mean_mrr          = np.array([c["mean_reciprocal_rank"] for c in con])
    n_network_error   = np.array([r["n_empty_network_error"] for r in rows])
    pct_network_error = 100.0 * n_network_error / n_total

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    fig.suptitle("Conformal prediction: metrics vs α (synthetic-EI MLP, joint calibration)", fontsize=12)

    def _vline(ax):
        ax.axvline(REFERENCE_ALPHA, color="grey", linestyle="--", linewidth=0.9, alpha=0.7, label=f"α={REFERENCE_ALPHA}")

    def _fmt(ax, ylabel, ylim=None):
        ax.set_xlabel("α (miscoverage level)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(alphas)
        ax.xaxis.set_tick_params(labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.5)
        if ylim is not None:
            ax.set_ylim(ylim)

    # ── Panel 1: Exact match % ───────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(alphas, exact_match_pct, "o-", color="steelblue", linewidth=1.5)
    _vline(ax)
    _fmt(ax, "Exact match (%)")
    ax.set_title("Exact match rate")

    # ── Panel 2: Mean SPT (constrained vs unconstrained) ────────────────────
    ax = axes[0, 1]
    ax.plot(alphas, mean_spt_con, "o-", color="steelblue", linewidth=1.5, label="Constrained")
    ax.plot(alphas, mean_spt_unc, "s--", color="coral", linewidth=1.5, label="Unconstrained")
    _vline(ax)
    ax.legend(fontsize=8)
    _fmt(ax, "Mean SPT")
    ax.set_title("Size-penalised Tanimoto")

    # ── Panel 3: Candidate set size ──────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(alphas, mean_cands_con, "o-", color="steelblue", linewidth=1.5, label="Mean (con.)")
    ax.plot(alphas, median_cands_con, "^--", color="steelblue", linewidth=1.2, alpha=0.7, label="Median (con.)")
    ax.axhline(mean_cands_unc[0], color="coral", linestyle="--", linewidth=1.4, label="Mean (unc.)")
    _vline(ax)
    ax.legend(fontsize=8)
    _fmt(ax, "# candidates")
    ax.set_title("Candidate set size")

    # ── Panel 4: Coverage failures ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.bar(alphas, pct_network_error, width=0.025, color="salmon", alpha=0.85)
    _vline(ax)
    _fmt(ax, "Empty sets — model failure (%)")
    ax.set_title("Coverage failures (model failure %)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")

    # also save PNG for quick viewing
    png_path = OUT_PATH.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {png_path}")


if __name__ == "__main__":
    main()
