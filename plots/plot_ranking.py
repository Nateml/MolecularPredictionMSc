#!/usr/bin/env python3
"""Plot per-rank Tanimoto similarity for a single test instance.

Two curves are shown:
  - Scored   : candidates sorted by model log-likelihood (descending)
  - Randomised: same candidates in a random order

--sample-idx indexes into data/splits/synthetic-ei/test_solvable_5s.pkl,
which is a list of dataset indices into the full dataset (config.data.dataset_path).

Usage
-----
    python plot_ranking.py --sample-idx 0
    python plot_ranking.py --sample-idx 3 --output ranking.png
    python plot_ranking.py --sample-idx 3 \
        --solvable-pkl data/splits/synthetic-ei/test_solvable_5s.pkl \
        --checkpoint models/final/multitask_mlp_mixed_best.pt
"""
from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import config
from data.datasets import dataset_from_schema
from model.loader import load_model, load_thresholds
from model.predict import predict_single

from evaluate_molecules import (
    canonical,
    canonical_no_stereo,
    _morgan_fp,
    tanimoto,
    score_molecule,
    run_asp_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SOLVABLE_PKL = "data/splits/synthetic-ei/test_solvable_5s.pkl"


def tanimoto_curve(gt_smiles: str, candidates: list[str]) -> list[float]:
    gt_canon = canonical(gt_smiles)
    gt_canon_ns = canonical_no_stereo(gt_smiles)
    gt_fp = _morgan_fp(gt_canon)
    sims = []
    for cand in candidates:
        cand_canon = canonical(cand)
        if cand_canon is None:
            sims.append(0.0)
        elif canonical_no_stereo(cand) == gt_canon_ns:
            sims.append(1.0)
        else:
            sims.append(tanimoto(gt_canon, cand_canon, gt_fp=gt_fp))
    return sims


def load_record(solvable_pkl: Path, sample_idx: int, schema: str | None) -> dict:
    with open(solvable_pkl, "rb") as f:
        solvable_indices: list[int] = pickle.load(f)

    if sample_idx >= len(solvable_indices):
        raise IndexError(
            f"--sample-idx {sample_idx} out of range "
            f"(solvable list has {len(solvable_indices)} entries)"
        )
    dataset_idx = solvable_indices[sample_idx]

    schema_path = Path(schema) if schema else config.data.resolved_schema_path()
    dataset_path = config.data.resolved_dataset_path()
    ds = dataset_from_schema(dataset_path, schema_path, spectrum_len=config.model.spectrum_length)
    return ds._load_record(dataset_idx)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot candidate Tanimoto ranking for one solvable test instance.")
    p.add_argument(
        "--solvable-pkl", default=DEFAULT_SOLVABLE_PKL,
        help=f"Path to solvable-indices pkl (default: {DEFAULT_SOLVABLE_PKL}).",
    )
    p.add_argument(
        "--sample-idx", type=int, default=0,
        help="Index into the solvable pkl list (default: 0).",
    )
    p.add_argument("--schema", default=None)
    p.add_argument(
        "--checkpoint",
        default="models/final/synthetic-EI/multitask_mlp.best.pt",
        help="Path to model checkpoint (default: models/final/synthetic-EI/multitask_mlp.best.pt).",
    )
    p.add_argument("--asp-timeout", type=int, default=config.solver.timeout)
    p.add_argument("--max-combos", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--thresholds-type", default="joint", choices=["joint", "marginal"])
    p.add_argument("--output", default=None, help="Save plot to file instead of displaying.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for randomised curve (default: 42).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alpha = args.alpha or config.conformal.alpha

    # ── model ─────────────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"No checkpoint found at {ckpt_path}.")

    log.info(f"Loading model from {ckpt_path} ...")
    model, model_dataset, device = load_model(ckpt_path)

    use_joint = args.thresholds_type == "joint"
    thresholds = load_thresholds(ckpt_path, alpha=alpha, joint=use_joint)

    # ── record ────────────────────────────────────────────────────────────────────
    record = load_record(Path(args.solvable_pkl), args.sample_idx, args.schema)
    gt_smiles = record.get("smiles", "")
    spectrum_str = record.get("spectrum", "")
    if not gt_smiles or not spectrum_str:
        sys.exit("Record missing smiles or spectrum.")
    log.info(f"GT SMILES: {gt_smiles}")

    # ── inference ─────────────────────────────────────────────────────────────────
    x = model_dataset._spectrum_to_dense(spectrum_str)
    pred = predict_single(x, model, model_dataset, threshold=args.threshold, thresholds=thresholds)

    # ── ASP (GT atom counts) ──────────────────────────────────────────────────────
    sf = record.get("structural_features", {})
    count_ranges = {
        "num_carbons":   (sf["element_C"], sf["element_C"]),
        "num_hydrogens": (sf["element_H"], sf["element_H"]),
        "num_oxygens":   (sf["element_O"], sf["element_O"]),
        "num_nitrogens": (sf["element_N"], sf["element_N"]),
        "num_phosphorus":(sf["element_P"], sf["element_P"]),
        "num_sulfurs":   (sf["element_S"], sf["element_S"]),
    }
    log.info(f"GT formula: C{sf['element_C']}H{sf['element_H']}O{sf['element_O']}N{sf['element_N']}P{sf['element_P']}S{sf['element_S']}")
    log.info("Running ASP ...")
    candidates, timed_out = run_asp_pipeline(
        pred, count_ranges,
        asp_timeout=args.asp_timeout,
        max_combos=args.max_combos,
    )
    log.info(f"Got {len(candidates)} candidates (timed_out={timed_out}).")

    if not candidates:
        sys.exit("No candidates — try a different --sample-idx.")

    # ── sort by model score ───────────────────────────────────────────────────────
    scores: list[float] = []
    if pred.raw is not None:
        scores = [
            score_molecule(smi, pred.raw) if canonical(smi) else float("-inf")
            for smi in candidates
        ]
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        candidates = [candidates[i] for i in order]
        scores = [scores[i] for i in order]

    # ── randomised copy ───────────────────────────────────────────────────────────
    rand_candidates = candidates[:]
    rng = random.Random(args.seed)
    rng.shuffle(rand_candidates)

    # ── tanimoto curves ───────────────────────────────────────────────────────────
    scored_sims = tanimoto_curve(gt_smiles, candidates)
    rand_sims = tanimoto_curve(gt_smiles, rand_candidates)
    ranks = np.arange(1, len(candidates) + 1)

    gt_canon_ns = canonical_no_stereo(gt_smiles)
    exact_rank = next(
        (r for r, cand in enumerate(candidates, 1) if canonical_no_stereo(cand) == gt_canon_ns),
        None,
    )

    # ── plot ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranks, scored_sims, label="Tanimoto (scored)", linewidth=1.2, color="steelblue")
    ax.plot(ranks, rand_sims, label="Tanimoto (randomised)", linewidth=1.0, color="coral", alpha=0.7)

    if exact_rank is not None:
        ax.axvline(exact_rank, color="green", linestyle="--", linewidth=1.0,
                   label=f"Exact match (rank {exact_rank} / {len(candidates)})")

    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("Tanimoto similarity (Morgan r=2)")

    if scores:
        ax2 = ax.twinx()
        ax2.plot(ranks, scores, label="Log-likelihood (model)", linewidth=1.0,
                 color="mediumpurple", alpha=0.6, linestyle=":")
        ax2.set_ylabel("Model log-likelihood")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.legend()

    #ax.set_title(
        #f"Per-rank Tanimoto — solvable sample {args.sample_idx}\n"
        #f"GT: {gt_smiles[:80]}{'...' if len(gt_smiles) > 80 else ''}  |  N={len(candidates)}"
    #)
    ax.set_title("")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
