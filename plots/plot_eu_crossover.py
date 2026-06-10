#!/usr/bin/env python3
"""Plot the EU crossover threshold λ*(α) = ΔT̄ / Δ|M| over alpha values."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("eval_results/synthetic-ei/mlp/molecule_eval/alpha-experiment")
OUT_PATH = Path("results/eu_crossover_threshold.pdf")
REFERENCE_ALPHA = 0.05
VALID_STATUSES = {"ok", "empty_network_error"}


def load_per_sample(results_dir: Path) -> list[tuple[float, list[dict]]]:
    out = []
    for alpha_dir in sorted(results_dir.glob("alpha_*")):
        csv_path = alpha_dir / "molecule_evaluation_test.csv"
        if not csv_path.exists():
            continue
        alpha = float(alpha_dir.name.removeprefix("alpha_"))
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        out.append((alpha, rows))
    out.sort(key=lambda x: x[0])
    return out


def crossover_threshold(rows: list[dict]) -> float | None:
    paired = [
        r for r in rows
        if r["status"] in VALID_STATUSES and r["unc_timed_out"] == "False"
    ]
    if not paired:
        return None
    delta_t = np.mean([float(r["unc_max_tanimoto"]) for r in paired]) \
             - np.mean([float(r["max_tanimoto"]) for r in paired])
    delta_m = np.mean([int(r["unc_num_candidates"]) for r in paired]) \
             - np.mean([int(r["num_candidates"]) for r in paired])
    if delta_m == 0:
        return None
    return float(delta_t / delta_m)


def main() -> None:
    data = load_per_sample(RESULTS_DIR)
    if not data:
        raise FileNotFoundError(f"No results in {RESULTS_DIR}")

    alphas, thresholds = [], []
    for alpha, rows in data:
        lam_star = crossover_threshold(rows)
        if lam_star is not None:
            alphas.append(alpha)
            thresholds.append(lam_star)

    alphas = np.array(alphas)
    thresholds = np.array(thresholds)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(alphas, thresholds, "o-", color="steelblue", linewidth=1.8, zorder=2)

    # Annotate k = 1/λ* on right axis
    ax2 = ax.twinx()
    ax2.set_ylabel("$k = 1/\\lambda^*$  (break-even review budget)", fontsize=9)
    # Avoid division by zero; map thresholds → k
    k_vals = np.where(thresholds > 0, 1.0 / thresholds, np.nan)
    ax2.plot(alphas, k_vals, alpha=0)  # invisible, just sets scale
    ax2.set_ylim(
        1.0 / max(thresholds) if max(thresholds) > 0 else 0,
        1.0 / min(thresholds[thresholds > 0]) if any(thresholds > 0) else 1,
    )

    ax.axvline(REFERENCE_ALPHA, color="grey", linestyle="--", linewidth=0.9,
               alpha=0.7, label=f"α={REFERENCE_ALPHA}")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("α (miscoverage level)")
    ax.set_ylabel("$\\lambda^* = \\Delta\\bar{T} / \\Delta|M|$  (crossover threshold)",
                  fontsize=9)
    ax.set_title(
        "EU crossover threshold $\\lambda^*(\\alpha)$\n"
        "Constraints improve EU when $\\lambda > \\lambda^*$  ·  "
        "synthetic-EI MLP, joint calibration"
    )
    ax.set_xticks(alphas)
    ax.xaxis.set_tick_params(labelsize=7.5)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
