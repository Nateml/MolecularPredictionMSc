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
    python evaluate_molecules.py --split-pkl data/splits/<hash>/test.pkl
    python evaluate_molecules.py --split-pkl data/splits/<hash>/test.pkl \\
        --full-dataset data/<dataset_name>.json
    python evaluate_molecules.py --test-set data/my_test.json
    python evaluate_molecules.py --split-pkl data/splits/<hash>/test.pkl \\
        --checkpoint models/run1/multitask_mlp.best.pt \\
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
from tqdm import tqdm

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import config
from data.datasets import dataset_from_schema, RawFeatureDataset
from model.loader import load_model, load_thresholds, thresholds_path
from model.predict import predict_single, RecordPrediction
from predicates.compiler import compile_asp_constraint
from predicates.interface import PredictionInterface
from asp.bridge import solve
from filters import dou_filter, heavy_atom_filter, small_mol_filter, element_type_filter, formula_complexity_filter, formula_complexity_score

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
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


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

def _load_test_records(args: argparse.Namespace, filter="medium") -> list[dict]:
    """Return a list of raw record dicts from the requested test source.

    Two modes:
      --test-set FILE   : load from a JSON dataset file directly.
      --split-pkl FILE  : load indices from a split pickle; full dataset
                         loaded from --full-dataset or the sibling dataset.pkl.

    Each returned dict contains at least 'spectrum' and 'smiles' keys.
    """
    if args.test_set:
        data_path = Path(args.test_set)
        schema_path = (Path(args.schema) if args.schema
                       else config.data.resolved_schema_path())
        ds = dataset_from_schema(
            data_path, schema_path,
            spectrum_len=config.model.spectrum_length,
        )
        out = []
        out_len = 0
        n_skipped = 0
        for i in range(len(ds)):
            rec = ds._load_record(i)
            smiles = rec.get("smiles")
            # Filter
            if not formula_complexity_filter(smiles, max_score=13):
                n_skipped += 1
                continue

            out.append(rec)
            out_len += 1

            if args.limit and out_len >= args.limit:
                break 

        return out



    # --- from split pkl -------------------------------------------------------
    split_pkl = Path(args.split_pkl)
    if not split_pkl.exists():
        raise FileNotFoundError(f"Split pkl not found: {split_pkl}")

    full_dataset = (
        Path(args.full_dataset) if args.full_dataset
        else split_pkl.parent / "dataset.pkl"
    )
    if not full_dataset.exists():
        raise FileNotFoundError(
            f"Full dataset pkl not found: {full_dataset}\n"
            "Pass --full-dataset explicitly or place dataset.pkl in the same directory as the split."
        )

    if full_dataset.suffix == ".json":
        schema_path = (Path(args.schema) if args.schema
                       else config.data.resolved_schema_path())
        full_dataset = dataset_from_schema(
            full_dataset, schema_path,
            spectrum_len=config.model.spectrum_length,
        )
    else:
        with open(full_dataset, "rb") as f:
            full_dataset = pickle.load(f)

    with open(split_pkl, "rb") as f:
        indices: list[int] = pickle.load(f)

    log.info(f"Loaded {len(indices)} indices from {split_pkl.name}.")

    out = []
    out_len = 0
    n_skipped = 0
    for i in indices:
        rec = full_dataset._load_record(i)
        smiles = rec.get("smiles")
        # Filter
        if not formula_complexity_filter(smiles, max_score=13):
            n_skipped += 1
            continue

        out.append(rec)
        out_len += 1

        if args.limit and out_len >= args.limit:
            break 

    return out, n_skipped


# ── inference pipeline ────────────────────────────────────────────────────────

def _combo_iter(count_ranges: dict[str, tuple[int, int]], max_combos: int | None):
    """Yield count dicts for all formula combinations, capped at max_combos."""
    ranges_list = [range(lo, hi + 1) for lo, hi in count_ranges.values()]
    n_combos = 1
    for r in ranges_list:
        n_combos *= len(r)
    if max_combos is not None and n_combos > max_combos:
        log.warning(
            f"Formula combinations ({n_combos}) exceed --max-combos={max_combos}; "
            "truncating iteration — some candidate SMILES may be missed."
        )
    for combo in itertools.islice(itertools.product(*ranges_list), max_combos):
        yield dict(zip(count_ranges, combo))


def run_asp_pipeline(
    pred: RecordPrediction,
    count_ranges: dict[str, tuple[int, int]],
    asp_timeout: int,
    max_combos: int | None,
) -> tuple[list[str], bool]:
    """Enumerate atom-count formula combinations and collect unique candidate SMILES.

    Returns (unique_smiles, any_solve_timed_out).
    """
    seen: set[str] = set()
    candidates: list[str] = []
    any_timed_out = False

    for count_dict in _combo_iter(count_ranges, max_combos):
        try:
            interface = pred.to_interface(counts=count_dict)
        except Exception as exc:
            log.debug(f"Invalid interface for counts {count_dict}: {exc}")
            continue

        asp_constraints, atom_counts = compile_asp_constraint(interface)

        log.info(f"Solving for molecule C{atom_counts['num_carbons']}H{atom_counts['num_hydrogens']}O{atom_counts['num_oxygens']}N{atom_counts['num_nitrogens']}P{atom_counts['num_phosphorus']}S{atom_counts['num_sulfurs']}")

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


def run_asp_pipeline_unconstrained(
    count_ranges: dict[str, tuple[int, int]],
    asp_timeout: int,
    max_combos: int | None,
) -> tuple[list[str], bool]:
    """Same formula iteration as run_asp_pipeline but with no structural constraints.

    Builds a PredictionInterface containing only atom counts (no binary/categorical
    entries, no continuous ranges), giving the baseline candidate set for a given
    formula space.

    Returns (unique_smiles, any_solve_timed_out).
    """
    seen: set[str] = set()
    candidates: list[str] = []
    any_timed_out = False

    for count_dict in _combo_iter(count_ranges, max_combos):
        try:
            interface = PredictionInterface.model_validate(count_dict)
        except Exception as exc:
            log.debug(f"Invalid unconstrained interface for counts {count_dict}: {exc}")
            continue

        asp_constraints, atom_counts = compile_asp_constraint(interface)

        try:
            cands, _, timed_out = solve(
                asp_constraints, atom_counts,
                output_list_of_smiles=True,
                timeout=asp_timeout,
            )
        except Exception as exc:
            log.debug(f"ASP unconstrained solve error for {count_dict}: {exc}")
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
        "--split-pkl", default=None,
        help="Path to a split indices .pkl file (e.g. data/splits/<hash>/test.pkl).",
    )

    p.add_argument(
        "--full-dataset", default=None,
        help="Path to the full dataset(.pkl/.json) file. "
             "Defaults to the dataset defined in config.yaml."
    )
    p.add_argument(
        "--schema", default=None,
        help="Path to schema JSON. Defaults to config value. Only used with --test-set.",
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
    p.add_argument(
        "--use-median-counts", action="store_true",
        help="Use the predicted medians of atom counts instead of ranges, reducing ASP "
        "calls significantly at the very high probability of not finding the true "
        "molecule."
    )
    p.add_argument(
        "--use-gt-counts", action="store_true",
        help="Use the ground truth molecular formulae for atom counts instead of the "
        "network's predictions. This is useful for evaluating the rest of the pipeline "
        "and circumventing the combinatorial explosion issue with predicted formulae."
    )
    p.add_argument(
        "--filter", default="medium",
        help="The filter preset to use when filtering out molecules. Ultralight, light, medium, or heavy."
    )
    p.add_argument(
        "--thresholds-type", default="joint", choices=["joint", "marginal"],
        help="Which conformal thresholds to load: 'joint' (default) uses the jointly-calibrated "
             "thresholds (.joint.thresholds.pt); 'marginal' uses the per-feature thresholds "
             "(.thresholds.pt).",
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

    use_joint = args.thresholds_type == "joint"
    thresholds = load_thresholds(ckpt_path, alpha=config.conformal.alpha, joint=use_joint)
    if thresholds is not None:
        log.info(f"Conformal thresholds loaded (alpha={config.conformal.alpha}, type={args.thresholds_type}).")
    else:
        log.info(
            f"No conformal thresholds found; "
            f"using manual threshold={args.threshold}."
        )

    # ── test records ─────────────────────────────────────────────────────────
    log.info("Loading test records ...")
    records, n_skipped = _load_test_records(args, filter=args.filter)
    log.info(f"Evaluating {len(records)} samples.")

    # ── per-sample loop ───────────────────────────────────────────────────────
    per_sample_rows: list[dict] = []
    n_failed = 0
    n_timeout = 0
    n_empty = 0

    for sample_idx, record in tqdm(enumerate(records)):
        gt_smiles_raw: str | None = record.get("smiles")
        spectrum_str: str = record.get("spectrum", "")
        smiles = record.get("smiles")

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
            "unc_num_candidates": 0,
            "unc_max_tanimoto": 0.0,
            "unc_size_penalised_tanimoto": 0.0,
            "unc_exact_match": False,
            "unc_timed_out": False,
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
            x = model_dataset._spectrum_to_dense(spectrum_str)
            pred = predict_single(x, model, model_dataset, threshold=args.threshold, thresholds=thresholds)
        except Exception as exc:
            row["status"] = "inference_error"
            n_failed += 1
            per_sample_rows.append(row)
            log.warning(f"[{sample_idx}] Inference error: {exc}")
            continue

        count_ranges = pred.count_ranges
        if args.use_median_counts:
            count_ranges = {k: ((lo + hi) // 2, (lo + hi) // 2) for k, (lo, hi) in count_ranges.items()}

        if args.use_gt_counts:
            count_ranges["num_carbons"] = tuple([record.get("structural_features")["element_C"]] * 2)
            count_ranges["num_hydrogens"] = tuple([record.get("structural_features")["element_H"]] * 2)
            count_ranges["num_oxygens"] = tuple([record.get("structural_features")["element_O"]] * 2)
            count_ranges["num_nitrogens"] = tuple([record.get("structural_features")["element_N"]] * 2)
            count_ranges["num_phosphorus"] = tuple([record.get("structural_features")["element_P"]] * 2)
            count_ranges["num_sulfurs"] = tuple([record.get("structural_features")["element_S"]] * 2)

        # ── ASP solving ─────────────────────────────────
        try:
            t0 = time.monotonic()
            candidates, any_timed_out = run_asp_pipeline(
                pred, count_ranges,
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

        row.update({
            "max_tanimoto": round(max_tan, 6),
            "size_penalised_tanimoto": round(spt, 6),
            "exact_match": exact,
            "candidates": " | ".join(candidates[: args.max_candidates_saved]),
        })
        # Preserve timeout/empty status; only upgrade to "ok" on clean success.
        if row["status"] not in ("asp_timeout", "empty_candidates") and n_cands > 0:
            row["status"] = "ok"

        # ── unconstrained ASP solve (same formula space, no structural constraints) ──
        try:
            unc_candidates, unc_timed_out = run_asp_pipeline_unconstrained(
                count_ranges,
                asp_timeout=args.asp_timeout,
                max_combos=args.max_combos,
            )
        except Exception as exc:
            log.debug(f"[{sample_idx}] Unconstrained ASP error: {exc}")
            unc_candidates, unc_timed_out = [], False

        unc_n_cands = len(unc_candidates)
        unc_max_tan, unc_spt, unc_exact = evaluate_candidates(gt_smiles_raw, unc_candidates)

        row.update({
            "unc_num_candidates": unc_n_cands,
            "unc_max_tanimoto": round(unc_max_tan, 6),
            "unc_size_penalised_tanimoto": round(unc_spt, 6),
            "unc_exact_match": unc_exact,
            "unc_timed_out": unc_timed_out,
        })

        per_sample_rows.append(row)

        tag = (
            "TIMEO" if any_timed_out else
            "EMPTY" if n_cands == 0 else
            "EXACT" if exact else
            "OK   "
        )
        log.info(
            f"[{sample_idx:5d}] {tag} | cands={n_cands:5d} (unc={unc_n_cands:5d}) | "
            f"tan={max_tan:.3f} | spt={spt:.4f} | "
            f"unc_tan={unc_max_tan:.3f} | unc_spt={unc_spt:.4f} | "
            f"t={elapsed:.1f}s | {gt_smiles_raw[:55]} | complexity={formula_complexity_score(smiles)}"
        )

    # ── aggregate metrics ─────────────────────────────────────────────────────
    n_eval = len(per_sample_rows)
    metric_rows = [r for r in per_sample_rows if not r["timed_out"]]
    n_metric = len(metric_rows)
    n_with_cands = sum(1 for r in metric_rows if r["num_candidates"] > 0)
    n_exact = sum(1 for r in metric_rows if r["exact_match"])

    max_tans = np.array([r["max_tanimoto"] for r in metric_rows], dtype=float)
    spts = np.array([r["size_penalised_tanimoto"] for r in metric_rows], dtype=float)
    n_cands_arr = np.array([r["num_candidates"] for r in metric_rows], dtype=int)

    # Unconstrained aggregate metrics (exclude rows where unc solve timed out)
    unc_metric_rows = [r for r in per_sample_rows if not r["unc_timed_out"] and not r["timed_out"]]
    n_unc_metric = len(unc_metric_rows)
    unc_n_with_cands = sum(1 for r in unc_metric_rows if r["unc_num_candidates"] > 0)
    unc_n_exact = sum(1 for r in unc_metric_rows if r["unc_exact_match"])
    unc_max_tans = np.array([r["unc_max_tanimoto"] for r in unc_metric_rows], dtype=float)
    unc_spts = np.array([r["unc_size_penalised_tanimoto"] for r in unc_metric_rows], dtype=float)
    unc_n_cands_arr = np.array([r["unc_num_candidates"] for r in unc_metric_rows], dtype=int)

    summary = {
        "n_evaluated": n_eval,
        "n_metric": n_metric,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "n_asp_timeout": n_timeout,
        "n_empty_candidates": n_empty,
        "n_with_candidates": n_with_cands,
        "n_exact_match": n_exact,
        "pct_with_candidates": round(100.0 * n_with_cands / n_metric, 2) if n_metric else 0.0,
        "pct_exact_match": round(100.0 * n_exact / n_metric, 2) if n_metric else 0.0,
        "mean_max_tanimoto": round(float(np.mean(max_tans)), 6) if n_metric else 0.0,
        "mean_size_penalised_tanimoto": round(float(np.mean(spts)), 6) if n_metric else 0.0,
        "mean_num_candidates": round(float(np.mean(n_cands_arr)), 2) if n_metric else 0.0,
        "median_num_candidates": int(np.median(n_cands_arr)) if n_metric else 0,
        "unconstrained": {
            "n_metric": n_unc_metric,
            "n_with_candidates": unc_n_with_cands,
            "n_exact_match": unc_n_exact,
            "pct_with_candidates": round(100.0 * unc_n_with_cands / n_unc_metric, 2) if n_unc_metric else 0.0,
            "pct_exact_match": round(100.0 * unc_n_exact / n_unc_metric, 2) if n_unc_metric else 0.0,
            "mean_max_tanimoto": round(float(np.mean(unc_max_tans)), 6) if n_unc_metric else 0.0,
            "mean_size_penalised_tanimoto": round(float(np.mean(unc_spts)), 6) if n_unc_metric else 0.0,
            "mean_num_candidates": round(float(np.mean(unc_n_cands_arr)), 2) if n_unc_metric else 0.0,
            "median_num_candidates": int(np.median(unc_n_cands_arr)) if n_unc_metric else 0,
        },
        "checkpoint": str(ckpt_path),
        "split_or_file": (
            args.test_set if args.test_set else args.split_pkl
        ),
        "asp_timeout_s": args.asp_timeout,
        "manual_threshold": args.threshold,
        "conformal": thresholds is not None,
        "conformal_type": args.thresholds_type if thresholds is not None else None,
        "conformal_thresholds_path": (
            str(thresholds_path(ckpt_path, alpha=config.conformal.alpha, joint=use_joint))
            if thresholds is not None else None
        ),
    }

    unc = summary["unconstrained"]
    print("\n=== Molecule-Level Evaluation Summary ===")
    print(f"  Checkpoint            : {ckpt_path}")
    if thresholds is not None:
        print(f"  Conformal type        : {summary['conformal_type']}")
        print(f"  Conformal thresholds  : {summary['conformal_thresholds_path']}")
    else:
        print(f"  Manual threshold      : {args.threshold}  (no conformal thresholds)")
    print(f"  Samples evaluated     : {n_eval}")
    print(f"  Included in metrics   : {n_metric}  (excludes timeouts)")
    print(f"  Failed                : {n_failed}")
    print(f"  Skipped (filtered)    : {n_skipped}")
    print(f"  ASP timeouts          : {n_timeout}")
    print(f"  Empty candidate sets  : {n_empty}")
    print(f"  With ≥1 candidate     : {n_with_cands}  ({summary['pct_with_candidates']:.1f}%)")
    print(f"  Exact match           : {n_exact}  ({summary['pct_exact_match']:.1f}%)")
    print(f"  Mean max-Tanimoto     : {summary['mean_max_tanimoto']:.4f}")
    print(f"  Mean SPT              : {summary['mean_size_penalised_tanimoto']:.4f}")
    print(f"  Mean num candidates   : {summary['mean_num_candidates']:.1f}")
    print(f"  Median num candidates : {summary['median_num_candidates']}")
    print(f"\n--- Unconstrained baseline (same formula space, no structural constraints) ---")
    print(f"  With ≥1 candidate     : {unc_n_with_cands}  ({unc['pct_with_candidates']:.1f}%)")
    print(f"  Exact match           : {unc_n_exact}  ({unc['pct_exact_match']:.1f}%)")
    print(f"  Mean max-Tanimoto     : {unc['mean_max_tanimoto']:.4f}")
    print(f"  Mean SPT              : {unc['mean_size_penalised_tanimoto']:.4f}")
    print(f"  Mean num candidates   : {unc['mean_num_candidates']:.1f}")
    print(f"  Median num candidates : {unc['median_num_candidates']}")
    if n_metric and n_unc_metric:
        spt_delta = summary['mean_size_penalised_tanimoto'] - unc['mean_size_penalised_tanimoto']
        cand_ratio = (summary['mean_num_candidates'] / unc['mean_num_candidates']
                      if unc['mean_num_candidates'] > 0 else float('nan'))
        print(f"\n  SPT delta (constrained - unconstrained): {spt_delta:+.4f}")
        print(f"  Candidate count ratio (con/unc):         {cand_ratio:.3f}")

    _save_csv(per_sample_rows, out_dir / "molecule_evaluation_test.csv")
    _save_json(summary, out_dir / "molecule_evaluation_summary.json")


if __name__ == "__main__":
    main()
