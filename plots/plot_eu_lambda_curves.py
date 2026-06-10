#!/usr/bin/env python3
"""Plot mean EU_λ delta (constrained − unconstrained) vs alpha for multiple lambda values."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RESULTS_DIR = Path("eval_results/synthetic-ei/mlp/molecule_eval/alpha-experiment")
OUT_PATH = Path("results/eu_lambda_curves.pdf")

LAMBDAS = [0.001, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
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


def eu(max_tan: float, n: int, lam: float) -> float:
    return max_tan - lam * n


def mean_eu_delta(rows: list[dict], lam: float) -> float:
    paired = [r for r in rows
              if r["status"] in VALID_STATUSES and r["unc_timed_out"] == "False"]
    if not paired:
        return 0.0
    deltas = [
        eu(float(r["max_tanimoto"]), int(r["num_candidates"]), lam)
        - eu(float(r["unc_max_tanimoto"]), int(r["unc_num_candidates"]), lam)
        for r in paired
    ]
    return float(np.mean(deltas))


def mean_eu_constrained(rows: list[dict], lam: float) -> float:
    valid = [r for r in rows if r["status"] in VALID_STATUSES]
    if not valid:
        return 0.0
    return float(np.mean([
        eu(float(r["max_tanimoto"]), int(r["num_candidates"]), lam)
        for r in valid
    ]))


def lam_star(rows: list[dict]) -> float | None:
    paired = [r for r in rows
              if r["status"] in VALID_STATUSES and r["unc_timed_out"] == "False"]
    if not paired:
        return None
    delta_t = np.mean([float(r["unc_max_tanimoto"]) for r in paired]) \
            - np.mean([float(r["max_tanimoto"])     for r in paired])
    delta_m = np.mean([int(r["unc_num_candidates"]) for r in paired]) \
            - np.mean([int(r["num_candidates"])     for r in paired])
    return float(delta_t / delta_m) if delta_m != 0 else None


def crossover_alpha(alphas, deltas) -> float | None:
    for i in range(len(deltas) - 1):
        if deltas[i] <= 0 and deltas[i + 1] > 0:
            t = -deltas[i] / (deltas[i + 1] - deltas[i])
            return alphas[i] + t * (alphas[i + 1] - alphas[i])
        if deltas[i] >= 0 and deltas[i + 1] < 0:
            t = deltas[i] / (deltas[i] - deltas[i + 1])
            return alphas[i] + t * (alphas[i + 1] - alphas[i])
    return None


def _fmt(ax, ylabel, alphas):
    ax.set_xlabel("α (miscoverage level)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(alphas)
    ax.xaxis.set_tick_params(labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)


def _vline(ax):
    ax.axvline(REFERENCE_ALPHA, color="grey", linestyle="--", linewidth=0.9,
               alpha=0.7, label=f"α={REFERENCE_ALPHA}")


def _plot_delta(data, alphas, out_path: Path) -> None:
    norm = mcolors.Normalize(vmin=LAMBDAS[0], vmax=LAMBDAS[-1])
    cmap = plt.cm.coolwarm

    fig, ax = plt.subplots(figsize=(9, 5))

    all_deltas = []
    for lam in LAMBDAS:
        deltas = [mean_eu_delta(rows, lam) for _, rows in data]
        colour = cmap(norm(lam))
        ax.plot(alphas, deltas, "o-", color=colour, linewidth=1.5, zorder=2,
                label=f"λ={lam:.3f}")
        all_deltas.append(deltas)

        xover = crossover_alpha(alphas.tolist(), deltas)
        if xover is not None:
            ax.plot(xover, 0, "*", color=colour, markersize=10, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5)

    y_top = max(max(d) for d in all_deltas)
    y_bot = min(min(d) for d in all_deltas)
    pad = (y_top - y_bot) * 0.08
    ax.set_ylim(y_bot - pad, y_top + pad)

    ax.axhspan(0, y_top + pad, color="steelblue", alpha=0.06, zorder=0)
    ax.axhspan(y_bot - pad, 0, color="salmon", alpha=0.06, zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.8, zorder=1)
    _vline(ax)

    ax.text(alphas[-1] + 0.005, y_top * 0.35, "constrained\nwins",
            fontsize=9, color="steelblue", va="center", ha="left")
    ax.text(alphas[-1] + 0.005, y_bot * 0.35, "unconstrained\nwins",
            fontsize=9, color="salmon", va="center", ha="left")

    # Twin axis: λ*(α) curve scaled to lambda range so it tracks zero-crossings visually
    lam_stars = [lam_star(rows) for _, rows in data]
    valid_mask = [v is not None for v in lam_stars]
    ls_alphas = alphas[valid_mask]
    ls_vals = np.array([v for v in lam_stars if v is not None])

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ls_alphas, ls_vals, "s-", color="black", linewidth=2.2, zorder=4,
             markersize=6, markerfacecolor="white", markeredgecolor="black",
             markeredgewidth=1.4, label="$\\lambda^*(\\alpha)$")
    ax2.set_ylim(LAMBDAS[0], LAMBDAS[-1])
    ax2.set_ylabel("$\\lambda^*$  (crossover threshold)", fontsize=11)
    ax2.yaxis.set_tick_params(labelsize=10)

    ax.set_xlim(alphas[0] - 0.01, alphas[-1] + 0.06)

    # Merge legends from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              fontsize=9, loc="upper left", ncol=2, title="λ (cost/candidate)",
              title_fontsize=9)
    _fmt(ax, "$\\Delta\\mathrm{EU}_\\lambda$  (constrained − unconstrained)", alphas)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


def _plot_constrained(data, alphas, out_path: Path) -> None:
    cmap = plt.cm.viridis
    colours = [cmap(i / (len(LAMBDAS) - 1)) for i in range(len(LAMBDAS))]

    fig, ax = plt.subplots(figsize=(9, 5))

    for lam, colour in zip(LAMBDAS, colours):
        eus = [mean_eu_constrained(rows, lam) for _, rows in data]
        ax.plot(alphas, eus, "o-", color=colour, linewidth=1.5, label=f"λ={lam:.3f}")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    _vline(ax)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        "Constrained expected utility vs α  ·  $U_\\lambda = \\mathrm{maxT} - \\lambda|M|$\n"
        "synthetic-EI MLP, joint calibration"
    )
    _fmt(ax, "$\\mathrm{EU}_\\lambda$  (constrained)", alphas)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


def main() -> None:
    data = load_per_sample(RESULTS_DIR)
    if not data:
        raise FileNotFoundError(f"No results in {RESULTS_DIR}")

    alphas = np.array([a for a, _ in data])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    _plot_delta(data, alphas, OUT_PATH)
    _plot_constrained(data, alphas, Path("results/eu_lambda_constrained.pdf"))


if __name__ == "__main__":
    main()
