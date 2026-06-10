#!/usr/bin/env python3
"""Plot mean SPT_β vs alpha for multiple beta values."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("eval_results/synthetic-ei/mlp/molecule_eval/alpha-experiment")
OUT_PATH = Path("results/spt_beta_delta_curves.pdf")

BETAS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
REFERENCE_ALPHA = 0.05


def spt_beta(max_tan: float, n_cands: int, beta: float) -> float:
    if n_cands == 0:
        return 0.0
    return max_tan / n_cands ** beta


def load_per_sample(results_dir: Path) -> list[tuple[float, list[dict]]]:
    """Return [(alpha, rows)] sorted by alpha."""
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


def mean_spt(rows: list[dict], beta: float, unc: bool = False) -> float:
    tan_col = "unc_max_tanimoto" if unc else "max_tanimoto"
    cands_col = "unc_num_candidates" if unc else "num_candidates"
    status_col = "status"
    valid = [
        r for r in rows
        if r[status_col] in ("ok", "empty_network_error")
    ]
    if not valid:
        return 0.0
    return float(np.mean([
        spt_beta(float(r[tan_col]), int(r[cands_col]), beta)
        for r in valid
    ]))


def main() -> None:
    data = load_per_sample(RESULTS_DIR)
    if not data:
        raise FileNotFoundError(f"No results in {RESULTS_DIR}")

    alphas = np.array([alpha for alpha, _ in data])

    cmap = plt.cm.plasma
    colours = [cmap(i / (len(BETAS) - 1)) for i in range(len(BETAS))]

    fig, ax = plt.subplots(figsize=(9, 5))

    for beta, colour in zip(BETAS, colours):
        con_spts = np.array([mean_spt(rows, beta, unc=False) for _, rows in data])
        unc_spts = np.array([mean_spt(rows, beta, unc=True) for _, rows in data])
        delta = con_spts - unc_spts
        ax.plot(alphas, delta, "o-", color=colour, linewidth=1.5, label=f"β={beta}")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axvline(REFERENCE_ALPHA, color="grey", linestyle="--", linewidth=0.9, alpha=0.7,
               label=f"α={REFERENCE_ALPHA}")

    ax.set_xlabel("α (miscoverage level)")
    ax.set_ylabel("ΔSPT_β  (constrained − unconstrained)")
    ax.set_title("SPT_β delta vs α for varying β\n(synthetic-EI MLP, joint calibration)")
    ax.set_xticks(alphas)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
