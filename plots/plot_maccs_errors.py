#!/usr/bin/env python3
"""Plot MACCS key prediction errors across the solvable test subset.

For each MACCS key, counts false positives (predicted 1, GT 0) and false
negatives (predicted 0, GT 1) across all solvable test samples, then plots
the top-N most frequently mis-predicted keys as a horizontal stacked bar chart.

Usage
-----
    python plot_maccs_errors.py
    python plot_maccs_errors.py --top-n 20 --output results/maccs_errors.png
    python plot_maccs_errors.py --solvable-pkl data/splits/synthetic-ei/test_solvable_5s.pkl
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
import torch
from tqdm import tqdm

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
from predicates.definitions import MACCS_DEFINITIONS

try:
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdMolDescriptors import CalcMACCSKeys as GenMACCSKeys
    _HAS_RDKIT = True
except ImportError:
    try:
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem import MACCSkeys
        def GenMACCSKeys(mol):
            return MACCSkeys.GenMACCSKeys(mol)
        _HAS_RDKIT = True
    except ImportError:
        _HAS_RDKIT = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SOLVABLE_PKL = "data/splits/synthetic-ei/test_solvable_5s.pkl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MACCS key error frequency plot over solvable test subset.")
    p.add_argument("--solvable-pkl", default=DEFAULT_SOLVABLE_PKL)
    p.add_argument("--schema", default=None)
    p.add_argument(
        "--checkpoint",
        default="models/final/synthetic-EI/multitask_mlp.best.pt",
        help="Path to model checkpoint.",
    )
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid threshold for binary prediction (default: 0.5).")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--thresholds-type", default="joint", choices=["joint", "marginal"])
    p.add_argument("--top-n", type=int, default=20, help="Show top-N most-errored keys.")
    p.add_argument("--output", default=None, help="Save plot to file instead of displaying.")
    return p.parse_args()


def main() -> None:
    if not _HAS_RDKIT:
        sys.exit("RDKit required for GT MACCS key computation.")

    args = parse_args()
    alpha = args.alpha or config.conformal.alpha

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"No checkpoint found at {ckpt_path}.")

    log.info(f"Loading model from {ckpt_path} ...")
    model, model_dataset, device = load_model(ckpt_path)
    maccs_length: int = getattr(model, "maccs_length", 167)

    use_joint = args.thresholds_type == "joint"
    thresholds = load_thresholds(ckpt_path, alpha=alpha, joint=use_joint)

    schema_path = Path(args.schema) if args.schema else config.data.resolved_schema_path()
    dataset_path = config.data.resolved_dataset_path()
    ds = dataset_from_schema(dataset_path, schema_path, spectrum_len=config.model.spectrum_length)

    with open(args.solvable_pkl, "rb") as f:
        solvable_indices: list[int] = pickle.load(f)

    log.info(f"Evaluating MACCS predictions over {len(solvable_indices)} samples ...")

    fp_counts = np.zeros(maccs_length, dtype=int)  # predicted 1, GT 0
    fn_counts = np.zeros(maccs_length, dtype=int)  # predicted 0, GT 1
    n_samples = 0

    for dataset_idx in tqdm(solvable_indices):
        record = ds._load_record(dataset_idx)
        gt_smiles = record.get("smiles", "")
        spectrum_str = record.get("spectrum", "")
        if not gt_smiles or not spectrum_str:
            continue

        gt_mol = MolFromSmiles(gt_smiles)
        if gt_mol is None:
            continue
        gt_maccs = GenMACCSKeys(gt_mol)

        try:
            x = model_dataset._spectrum_to_dense(spectrum_str)
            pred = predict_single(x, model, model_dataset, threshold=args.threshold, thresholds=thresholds)
        except Exception as exc:
            log.warning(f"Inference failed for idx {dataset_idx}: {exc}")
            continue

        if pred.raw is None or not pred.raw.binary:
            continue

        logits = torch.tensor(pred.raw.binary)
        probs = torch.sigmoid(logits).numpy()
        pred_binary = (probs > args.threshold).astype(int)

        for key_id in range(min(maccs_length, len(pred_binary))):
            gt_val = int(gt_maccs[key_id])
            pv = pred_binary[key_id]
            if pv == 1 and gt_val == 0:
                fp_counts[key_id] += 1
            elif pv == 0 and gt_val == 1:
                fn_counts[key_id] += 1

        n_samples += 1

    log.info(f"Processed {n_samples} samples.")

    # Build label map from MACCS_DEFINITIONS
    label_map: dict[int, str] = {
        k: v["predicate"] for k, v in MACCS_DEFINITIONS.items()
    }

    total_errors = fp_counts + fn_counts
    # Only consider keys that have at least one error and are in MACCS_DEFINITIONS
    defined_keys = sorted(MACCS_DEFINITIONS.keys())
    defined_keys = [k for k in defined_keys if k < maccs_length]

    key_errors = [(k, fp_counts[k], fn_counts[k], total_errors[k]) for k in defined_keys]
    key_errors.sort(key=lambda x: x[3], reverse=True)
    top = key_errors[:args.top_n]

    if not top:
        sys.exit("No MACCS key errors found.")

    labels = [f"MACCS {k}\n{label_map.get(k, '')}" for k, *_ in top]
    fps = np.array([e[1] for e in top])
    fns = np.array([e[2] for e in top])
    y = np.arange(len(top))

    fig_height = max(4.0, len(top) * 0.38)
    fig, ax = plt.subplots(figsize=(6.0, fig_height))

    ax.barh(y, fps, color="#D6604D", label="False positive (pred=1, GT=0)", height=0.6)
    ax.barh(y, fns, left=fps, color="#4393C3", label="False negative (pred=0, GT=1)", height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Error count  ($n={n_samples}$ samples)")
    ax.set_ylabel("MACCS key")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.6)

    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
