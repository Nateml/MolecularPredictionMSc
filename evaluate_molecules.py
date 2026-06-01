#!/usr/bin/env python3
"""
End-to-end evaluation of the full neural + conformal + ASP pipeline. For each test spectrum, 
it runs the trained MLP, optionally applies conformal prediction sets, converts predictions into ASP constraints, 
and calls the clingo solver to enumerate candidate SMILES. Candidates are scored against ground-truth SMILES 
using Tanimoto similarity (Morgan fingerprints, radius=2), size-penalised Tanimoto, and exact match. 

Metrics
-------
max_tanimoto
    max( Tanimoto(cand, gt) ) over all ASP candidates.
    Uses Morgan fingerprints (radius=2, 2048 bits).

size_penalised_tanimoto (SPT)
    SPT(M) = max_tanimoto / log(1 + |M|)
    where |M| is the number of candidates. Zero when |M| == 0.

exact_match
    True if the canonical ground-truth SMILES appears in the candidate set.

Outputs
-------
<output>/molecule_evaluation_test.csv   — per-sample results
<output>/molecule_evaluation_summary.json — aggregate metrics

Usage
-----
    python evaluate_molecules.py
    python evaluate_molecules.py --from-splits --split-name test
    python evaluate_molecules.py --test-set data/my_test.json
    python evaluate_molecules.py --checkpoint models/run1/multitask_mlp.best.pt \\
        --asp-timeout 30 --limit 200 --output results/run1/
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import config
from data.datasets import dataset_from_schema, RawFeatureDataset
from model.loader import load_model, load_thresholds
from model.conformal import sets_from_outputs
from predicates.interface import PredictionInterface
from predicates.compiler import compile_asp_constraint
from asp.bridge import solve

# Re-use inference helpers from infer.py — avoids duplicating logic.
from infer import (
    _conformal_structural_entries,
    _conformal_atom_count_ranges,
    _structural_entries,
    _atom_count_ranges_from_quantiles,
    _atom_counts_from_categorical,
    _continuous_constraints_from_intervals,
    _continuous_constraints_from_median,
)

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── SMILES / fingerprint utilities ────────────────────────────────────────────

def canonical(smi: str) -> str | None:
    """Return RDKit canonical SMILES, or None if the input is unparseable."""
    if not _HAS_RDKIT:
        return smi
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def _morgan_fp(smi: str):
    """Morgan fingerprint (radius=2, 2048 bits) for a canonical SMILES, or None."""
    if not _HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol else None


def tanimoto(gt_canon: str, cand_canon: str, gt_fp=None) -> float:
    """Tanimoto similarity between two canonical SMILES strings."""
    if not _HAS_RDKIT:
        return 1.0 if gt_canon == cand_canon else 0.0
    fp_a = gt_fp if gt_fp is not None else _morgan_fp(gt_canon)
    fp_b = _morgan_fp(cand_canon)
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def size_penalised_tanimoto(max_tan: float, n_candidates: int) -> float:
    """SPT(M) = max_tanimoto / log(1 + |M|).  Returns 0.0 when |M| == 0."""
    if n_candidates == 0:
        return 0.0
    return max_tan / math.log(1.0 + n_candidates)


# ── dataset / record loading ──────────────────────────────────────────────────

def _load_test_records(args: argparse.Namespace) -> list[dict]:
    """Return a list of raw record dicts from the requested test source.

    Two modes:
      --test-set FILE : load from a JSON dataset file directly.
      (default)       : load from saved split pickles (--from-splits).

    Each returned dict contains at least 'spectrum' and 'smiles' keys
    """
    if args.test_set:
        data_path = Path(args.test_set)
        schema_path = (Path(args.schema) if args.schema
                       else config.data.resolved_schema_path())
        ds = dataset_from_schema(
            data_path, schema_path,
            spectrum_len=config.model.spectrum_length,
        )
        n = args.limit if args.limit else len(ds)
        return [ds._load_record(i) for i in range(min(n, len(ds)))]

    # --- from saved splits (default) ----------------------------------------
    from data.split import splits_dir

    data_path = (Path(args.data_path) if args.data_path
                 else config.data.resolved_dataset_path())
    schema_path = (Path(args.schema) if args.schema
                   else config.data.resolved_schema_path())

    fractions = {"train": 0.75, "val": 0.10, "calibration": 0.05, "test": 0.10}
    split_dir = splits_dir(data_path, schema_path, fractions, seed=42)

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            "Run src/data/split.py to generate splits first."
        )

    with open(split_dir / "dataset.pkl", "rb") as f:
        full_dataset = pickle.load(f)
    with open(split_dir / f"{args.split_name}.pkl", "rb") as f:
        indices: list[int] = pickle.load(f)

    if args.limit:
        indices = indices[: args.limit]
    log.info(f"Loaded {len(indices)} indices from split '{args.split_name}' at {split_dir.name}.")
    return [full_dataset._load_record(idx) for idx in indices]


# ── inference pipeline ────────────────────────────────────────────────────────

def run_inference(
    spectrum_str: str,
    model,
    dataset: RawFeatureDataset,
    device: torch.device,
    thresholds: dict | None,
    threshold: float,
) -> tuple[dict, dict[str, tuple[int, int]], list[str]]:
    """Neural prediction → (structural_entries, count_ranges, extra_asp_rules).

    Mirrors infer.py main() exactly, including conformal and fallback
    threshold-based prediction.  Always returns count_ranges as a
    dict {key: (lo, hi)} so the caller can iterate over combinations.

    Uses `dataset` (model_dataset from load_model) for spectrum preprocessing
    and for all schema / normalisation attributes.
    """
    x = dataset._spectrum_to_dense(spectrum_str).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)

    is_raw = isinstance(dataset, RawFeatureDataset)
    num_quantiles = getattr(model, "num_quantiles", config.model.num_quantiles)

    if thresholds is not None:
        q_low_idx = 0
        q_high_idx = num_quantiles - 1
        conf_sets = sets_from_outputs(outputs, thresholds, q_low_idx, q_high_idx)
        structural = _conformal_structural_entries(conf_sets, dataset)

        if is_raw and conf_sets.get("quantile") is not None:
            count_ranges = _conformal_atom_count_ranges(
                conf_sets, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
            extra = _continuous_constraints_from_intervals(
                conf_sets["quantile"].squeeze(0),
                dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
        elif is_raw:
            # Conformal thresholds loaded but quantile sets unavailable; fall
            # back to raw quantile output ranges.
            q_logits = [t.squeeze(0) for t in outputs["quantile"]]
            count_ranges = _atom_count_ranges_from_quantiles(
                q_logits, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
            extra = _continuous_constraints_from_median(
                q_logits, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
        else:
            # PackedDataset + conformal: counts from categorical argmax.
            cat_logits = [t.squeeze(0) for t in outputs["categorical"]]
            single = _atom_counts_from_categorical(cat_logits, dataset.onehot_feature_order)
            count_ranges = {k: (v, v) for k, v in single.items()}
            extra = []

    else:
        # No conformal thresholds, manual threshold on raw outputs.
        structural = _structural_entries(outputs, dataset, threshold=threshold)
        if is_raw:
            q_logits = [t.squeeze(0) for t in outputs["quantile"]]
            count_ranges = _atom_count_ranges_from_quantiles(
                q_logits, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
            extra = _continuous_constraints_from_median(
                q_logits, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
        else:
            cat_logits = [t.squeeze(0) for t in outputs["categorical"]]
            single = _atom_counts_from_categorical(cat_logits, dataset.onehot_feature_order)
            count_ranges = {k: (v, v) for k, v in single.items()}
            extra = []

    return structural, count_ranges, extra


def run_asp_pipeline(
    structural: dict,
    count_ranges: dict[str, tuple[int, int]],
    extra_constraints: list[str],
    asp_timeout: int,
    max_combos: int | None,
) -> tuple[list[str], bool]:
    """Enumerate atom-count formula combinations and collect unique candidate SMILES.

    Returns (unique_smiles, any_solve_timed_out).
    """
    count_keys = list(count_ranges.keys())
    ranges_list = [range(lo, hi + 1) for lo, hi in count_ranges.values()]

    n_combos = 1
    for r in ranges_list:
        n_combos *= len(r)
    if max_combos is not None and n_combos > max_combos:
        log.warning(
            f"Formula combinations ({n_combos}) exceed --max-combos={max_combos}; "
            "truncating iteration — some candidate SMILES may be missed."
        )

    combo_iter = (
        dict(zip(count_keys, combo))
        for combo in itertools.islice(
            itertools.product(*ranges_list),
            max_combos,
        )
    )

    seen: set[str] = set()
    candidates: list[str] = []
    any_timed_out = False

    for count_dict in combo_iter:
        try:
            interface = PredictionInterface.model_validate({**structural, **count_dict})
        except Exception as exc:
            log.debug(f"Invalid interface for counts {count_dict}: {exc}")
            continue

        asp_constraints, atom_counts = compile_asp_constraint(interface)
        asp_constraints = asp_constraints + extra_constraints

        try:
            cands, _, timed_out = solve(
                asp_constraints, atom_counts,
                output_list_of_smiles=True,
                timeout=asp_timeout,
            )
        except Exception as exc:
            log.debug(f"ASP solve error for {count_dict}: {exc}")
            continue

        if timed_out:
            any_timed_out = True
        for smi in cands:
            if smi not in seen:
                seen.add(smi)
                candidates.append(smi)

    return candidates, any_timed_out


# ── per-sample metrics ────────────────────────────────────────────────────────

def evaluate_candidates(
    gt_smiles: str,
    candidate_smiles: list[str],
) -> tuple[float, float, bool]:
    """Compute (max_tanimoto, spt, exact_match) for one test sample.

    Ground-truth SMILES is used only here, strictly for metric computation, 
    not to influence inference or candidate generation.
    """
    if not candidate_smiles:
        return 0.0, 0.0, False

    gt_canon = canonical(gt_smiles)
    if gt_canon is None:
        return 0.0, 0.0, False

    gt_fp = _morgan_fp(gt_canon)
    max_tan = 0.0
    exact_match = False

    for cand in candidate_smiles:
        cand_canon = canonical(cand)
        if cand_canon is None:
            continue
        if cand_canon == gt_canon:
            exact_match = True
            max_tan = 1.0
            break
        tan = tanimoto(gt_canon, cand_canon, gt_fp=gt_fp)
        if tan > max_tan:
            max_tan = tan

    spt = size_penalised_tanimoto(max_tan, len(candidate_smiles))
    return max_tan, spt, exact_match


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Per-sample CSV saved to {path}")


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info(f"Summary JSON saved to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Molecule-level evaluation: full neural + conformal + ASP pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--test-set", default=None,
        help="Path to a JSON dataset file to use directly as the test set.",
    )
    src.add_argument(
        "--from-splits", action="store_true",
        help="Load test records from saved split pickles (default behaviour).",
    )

    p.add_argument(
        "--split-name", default="test",
        help="Split to evaluate: 'train', 'val', 'calibration', or 'test' "
             "(default: test). Only used with --from-splits.",
    )
    p.add_argument(
        "--data-path", default=None,
        help="Path to the full dataset JSON. Defaults to config value. "
             "Only used with --from-splits.",
    )
    p.add_argument(
        "--schema", default=None,
        help="Path to schema JSON. Defaults to config value.",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (.pt). "
             "Defaults to models/multitask_mlp.best.pt.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold applied to raw logits when no conformal "
             "thresholds are found (default: 0.5).",
    )
    p.add_argument(
        "--asp-timeout", type=int, default=config.solver.timeout,
        help=f"Per-solve ASP wall-clock timeout in seconds "
             f"(0 = no limit, default: {config.solver.timeout}).",
    )
    p.add_argument(
        "--max-combos", type=int, default=None,
        help="Maximum atom-count formula combinations per sample "
             "(default: unlimited). Use to cap runtime when predicted "
             "ranges are very wide.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N samples (for debugging).",
    )
    p.add_argument(
        "--output", default="results",
        help="Output directory for CSV and JSON results (default: results/).",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device override, e.g. 'cpu' or 'cuda:0' (default: auto-detect).",
    )
    p.add_argument(
        "--max-candidates-saved", type=int, default=50,
        help="Maximum candidate SMILES stored per row in the output CSV "
             "(default: 50).",
    )
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)

    # ── model ────────────────────────────────────────────────────────────────
    ckpt_path = (
        Path(args.checkpoint) if args.checkpoint
        else config.checkpointing.resolved_save_dir() / "multitask_mlp.best.pt"
    )
    if not ckpt_path.exists():
        ckpt_path = config.checkpointing.resolved_save_dir() / "multitask_mlp.latest.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint found at {ckpt_path}. Train the model first.")
        sys.exit(1)

    log.info(f"Loading model from {ckpt_path} ...")
    model, model_dataset, device = load_model(ckpt_path)

    if args.device:
        device = torch.device(args.device)
        model = model.to(device)

    thresholds = load_thresholds(ckpt_path)
    if thresholds is not None:
        log.info(f"Conformal thresholds loaded (alpha={config.conformal.alpha}).")
    else:
        log.info(
            f"No conformal thresholds found; "
            f"using manual threshold={args.threshold}."
        )

    # ── test records ─────────────────────────────────────────────────────────
    log.info("Loading test records ...")
    records = _load_test_records(args)
    if args.limit:
        records = records[: args.limit]
    log.info(f"Evaluating {len(records)} samples.")

    # ── per-sample loop ───────────────────────────────────────────────────────
    per_sample_rows: list[dict] = []
    n_failed = 0
    n_timeout = 0
    n_empty = 0
    n_exact = 0

    for sample_idx, record in enumerate(records):
        gt_smiles_raw: str | None = record.get("smiles")
        spectrum_str: str = record.get("spectrum", "")

        row: dict = {
            "sample_idx": sample_idx,
            "true_smiles": gt_smiles_raw or "",
            "num_candidates": 0,
            "max_tanimoto": 0.0,
            "size_penalised_tanimoto": 0.0,
            "exact_match": False,
            "timed_out": False,
            "status": "ok",
            "candidates": "",
        }

        # Validate ground-truth SMILES (needed for metrics, not inference).
        if not gt_smiles_raw:
            row["status"] = "no_gt_smiles"
            n_failed += 1
            per_sample_rows.append(row)
            log.warning(f"[{sample_idx}] No SMILES in record — skipping.")
            continue

        if canonical(str(gt_smiles_raw)) is None:
            row["status"] = "invalid_gt_smiles"
            n_failed += 1
            per_sample_rows.append(row)
            log.warning(f"[{sample_idx}] Unparseable GT SMILES: {gt_smiles_raw!r}")
            continue

        if not spectrum_str:
            row["status"] = "no_spectrum"
            n_failed += 1
            per_sample_rows.append(row)
            log.warning(f"[{sample_idx}] No spectrum data — skipping.")
            continue

        # ── neural inference ────────────────────────────
        try:
            structural, count_ranges, extra_constraints = run_inference(
                spectrum_str, model, model_dataset, device, thresholds, args.threshold
            )
        except Exception as exc:
            row["status"] = "inference_error"
            n_failed += 1
            per_sample_rows.append(row)
            log.warning(f"[{sample_idx}] Inference error: {exc}")
            continue

        # ── ASP solving ─────────────────────────────────
        try:
            t0 = time.monotonic()
            candidates, any_timed_out = run_asp_pipeline(
                structural, count_ranges, extra_constraints,
                asp_timeout=args.asp_timeout,
                max_combos=args.max_combos,
            )
            elapsed = time.monotonic() - t0
        except Exception as exc:
            row["status"] = "asp_error"
            n_failed += 1
            per_sample_rows.append(row)
            log.warning(f"[{sample_idx}] ASP pipeline error: {exc}")
            continue

        n_cands = len(candidates)
        row["num_candidates"] = n_cands

        if any_timed_out:
            row["timed_out"] = True
            row["status"] = "asp_timeout"
            n_timeout += 1
        if n_cands == 0:
            if not any_timed_out:
                row["status"] = "empty_candidates"
            n_empty += 1

        # ── metric computation (GT SMILES used here) ────────
        max_tan, spt, exact = evaluate_candidates(gt_smiles_raw, candidates)
        if exact:
            n_exact += 1

        row.update({
            "max_tanimoto": round(max_tan, 6),
            "size_penalised_tanimoto": round(spt, 6),
            "exact_match": exact,
            "candidates": " | ".join(candidates[: args.max_candidates_saved]),
        })
        # Preserve timeout/empty status; only upgrade to "ok" on clean success.
        if row["status"] not in ("asp_timeout", "empty_candidates") and n_cands > 0:
            row["status"] = "ok"

        per_sample_rows.append(row)

        tag = (
            "TIMEO" if any_timed_out else
            "EMPTY" if n_cands == 0 else
            "EXACT" if exact else
            "OK   "
        )
        log.info(
            f"[{sample_idx:5d}] {tag} | cands={n_cands:5d} | "
            f"tan={max_tan:.3f} | spt={spt:.4f} | "
            f"t={elapsed:.1f}s | {gt_smiles_raw[:55]}"
        )

    # ── aggregate metrics ─────────────────────────────────────────────────────
    n_eval = len(per_sample_rows)
    n_with_cands = sum(1 for r in per_sample_rows if r["num_candidates"] > 0)

    max_tans = np.array([r["max_tanimoto"] for r in per_sample_rows], dtype=float)
    spts = np.array([r["size_penalised_tanimoto"] for r in per_sample_rows], dtype=float)
    n_cands_arr = np.array([r["num_candidates"] for r in per_sample_rows], dtype=int)

    summary = {
        "n_evaluated": n_eval,
        "n_failed": n_failed,
        "n_asp_timeout": n_timeout,
        "n_empty_candidates": n_empty,
        "n_with_candidates": n_with_cands,
        "n_exact_match": n_exact,
        "pct_with_candidates": round(100.0 * n_with_cands / n_eval, 2) if n_eval else 0.0,
        "pct_exact_match": round(100.0 * n_exact / n_eval, 2) if n_eval else 0.0,
        "mean_max_tanimoto": round(float(np.mean(max_tans)), 6) if n_eval else 0.0,
        "mean_size_penalised_tanimoto": round(float(np.mean(spts)), 6) if n_eval else 0.0,
        "mean_num_candidates": round(float(np.mean(n_cands_arr)), 2) if n_eval else 0.0,
        "median_num_candidates": int(np.median(n_cands_arr)) if n_eval else 0,
        "checkpoint": str(ckpt_path),
        "split_or_file": (
            args.test_set if args.test_set else args.split_name
        ),
        "asp_timeout_s": args.asp_timeout,
        "manual_threshold": args.threshold,
        "conformal": thresholds is not None,
    }

    print("\n=== Molecule-Level Evaluation Summary ===")
    print(f"  Samples evaluated     : {n_eval}")
    print(f"  Failed / skipped      : {n_failed}")
    print(f"  ASP timeouts          : {n_timeout}")
    print(f"  Empty candidate sets  : {n_empty}")
    print(f"  With ≥1 candidate     : {n_with_cands}  ({summary['pct_with_candidates']:.1f}%)")
    print(f"  Exact match           : {n_exact}  ({summary['pct_exact_match']:.1f}%)")
    print(f"  Mean max-Tanimoto     : {summary['mean_max_tanimoto']:.4f}")
    print(f"  Mean SPT              : {summary['mean_size_penalised_tanimoto']:.4f}")
    print(f"  Mean num candidates   : {summary['mean_num_candidates']:.1f}")
    print(f"  Median num candidates : {summary['median_num_candidates']}")

    _save_csv(per_sample_rows, out_dir / "molecule_evaluation_test.csv")
    _save_json(summary, out_dir / "molecule_evaluation_summary.json")


if __name__ == "__main__":
    main()
