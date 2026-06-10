#!/usr/bin/env python3
"""Scatter plot: model log-likelihood score vs Tanimoto similarity for all candidates
across the solvable test subset.

Each point is one candidate from one sample. GT (exact match) candidates are
highlighted separately.

Usage
-----
    python plot_score_tanimoto.py
    python plot_score_tanimoto.py --output results/score_tanimoto.png
    python plot_score_tanimoto.py --solvable-pkl data/splits/synthetic-ei/test_solvable_5s.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
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
from tqdm import tqdm

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


def tanimoto_scores(gt_smiles: str, candidates: list[str]) -> list[float]:
    gt_canon = canonical(gt_smiles)
    gt_canon_ns = canonical_no_stereo(gt_smiles)
    gt_fp = _morgan_fp(gt_canon)
    out = []
    for cand in candidates:
        cand_canon = canonical(cand)
        if cand_canon is None:
            out.append(0.0)
        elif canonical_no_stereo(cand) == gt_canon_ns:
            out.append(1.0)
        else:
            out.append(tanimoto(gt_canon, cand_canon, gt_fp=gt_fp))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score vs Tanimoto scatter over all solvable test samples.")
    p.add_argument("--solvable-pkl", default=DEFAULT_SOLVABLE_PKL)
    p.add_argument("--schema", default=None)
    p.add_argument(
        "--checkpoint",
        default="models/final/synthetic-EI/multitask_mlp.best.pt",
        help="Path to model checkpoint.",
    )
    p.add_argument("--asp-timeout", type=int, default=config.solver.timeout)
    p.add_argument("--max-combos", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--thresholds-type", default="joint", choices=["joint", "marginal"])
    p.add_argument("--output", default=None, help="Save plot to file instead of displaying.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alpha = args.alpha or config.conformal.alpha

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"No checkpoint found at {ckpt_path}.")

    log.info(f"Loading model from {ckpt_path} ...")
    model, model_dataset, device = load_model(ckpt_path)

    use_joint = args.thresholds_type == "joint"
    thresholds = load_thresholds(ckpt_path, alpha=alpha, joint=use_joint)

    schema_path = Path(args.schema) if args.schema else config.data.resolved_schema_path()
    dataset_path = config.data.resolved_dataset_path()
    ds = dataset_from_schema(dataset_path, schema_path, spectrum_len=config.model.spectrum_length)

    with open(args.solvable_pkl, "rb") as f:
        solvable_indices: list[int] = pickle.load(f)

    log.info(f"Running over {len(solvable_indices)} solvable samples ...")

    # Collect (log_likelihood, tanimoto, is_exact_match) per candidate
    all_scores: list[float] = []
    all_tanimoto: list[float] = []
    all_exact: list[bool] = []

    for dataset_idx in tqdm(solvable_indices):
        record = ds._load_record(dataset_idx)
        gt_smiles = record.get("smiles", "")
        spectrum_str = record.get("spectrum", "")
        sf = record.get("structural_features", {})

        if not gt_smiles or not spectrum_str or not sf:
            continue

        try:
            x = model_dataset._spectrum_to_dense(spectrum_str)
            pred = predict_single(x, model, model_dataset, threshold=args.threshold, thresholds=thresholds)
        except Exception as exc:
            log.warning(f"Inference failed for idx {dataset_idx}: {exc}")
            continue

        count_ranges = {
            "num_carbons":    (sf["element_C"], sf["element_C"]),
            "num_hydrogens":  (sf["element_H"], sf["element_H"]),
            "num_oxygens":    (sf["element_O"], sf["element_O"]),
            "num_nitrogens":  (sf["element_N"], sf["element_N"]),
            "num_phosphorus": (sf["element_P"], sf["element_P"]),
            "num_sulfurs":    (sf["element_S"], sf["element_S"]),
        }

        try:
            candidates, _ = run_asp_pipeline(
                pred, count_ranges,
                asp_timeout=args.asp_timeout,
                max_combos=args.max_combos,
            )
        except Exception as exc:
            log.warning(f"ASP failed for idx {dataset_idx}: {exc}")
            continue

        if not candidates or pred.raw is None:
            continue

        gt_canon_ns = canonical_no_stereo(gt_smiles)
        tans = tanimoto_scores(gt_smiles, candidates)

        for cand, tan in zip(candidates, tans):
            if canonical(cand) is None:
                continue
            sc = score_molecule(cand, pred.raw)
            is_exact = (canonical_no_stereo(cand) == gt_canon_ns)
            all_scores.append(sc)
            all_tanimoto.append(tan)
            all_exact.append(is_exact)

    log.info(f"Collected {len(all_scores)} candidate points ({sum(all_exact)} exact matches).")

    scores_arr = np.array(all_scores)
    tan_arr = np.array(all_tanimoto)
    exact_arr = np.array(all_exact, dtype=bool)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.scatter(
        scores_arr[~exact_arr], tan_arr[~exact_arr],
        s=3, alpha=0.25, color="steelblue", label="Non-exact",
        rasterized=True,
    )
    ax.scatter(
        scores_arr[exact_arr], tan_arr[exact_arr],
        s=12, alpha=0.85, color="coral", label="Exact match (GT)",
        zorder=3,
    )

    ax.set_xlabel("Log-likelihood score")
    ax.set_ylabel("Tanimoto similarity to ground truth")
    #ax.set_title(
        #f"Score vs. Tanimoto  "
        #f"($n={{{len(all_scores)}}}$ candidates, $m={{{len(solvable_indices)}}}$ molecules)"
    #)
    ax.set_title("")
    ax.legend(loc="best")
    fig.tight_layout(pad=0.5)

    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
