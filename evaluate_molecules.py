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
import bisect
import csv
import itertools
import json
import logging
import math
import pickle
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import config
from data.datasets import dataset_from_schema, RawFeatureDataset
from model.loader import load_model, load_thresholds, thresholds_path
from model.predict import predict_single, RecordPrediction, RawModelOutputs
from predicates.compiler import compile_asp_constraint
from predicates.interface import PredictionInterface
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS
from asp.bridge import solve
from filters import dou_filter, heavy_atom_filter, small_mol_filter, element_type_filter, formula_complexity_filter, formula_complexity_score

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys
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


def canonical_no_stereo(smi: str) -> str | None:
    """Canonical SMILES with all stereochemistry stripped."""
    if not _HAS_RDKIT:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def diagnose_empty_candidates(
    record: dict,
    pred: RecordPrediction,
    model_dataset,
) -> tuple[str, list[str]]:
    """Determine why constrained ASP returned no candidates.

    Checks whether every GT structural feature is inside the corresponding
    prediction set or range.  If all are covered, the network's constraints are
    consistent with the truth and the empty set is an ASP implementation bug.
    If any are missed, the network predicted wrong constraints.

    Checks three constraint types:
      - MACCS key binary predictions (pred_set must contain GT bit value)
      - Feature-vector categorical predictions (GT bin must match predicted bin)
      - Continuous range predictions (GT value must lie within [lo, hi])

    Returns:
        ("asp_bug", [])                  — all GT features covered
        ("network_error", [descriptions]) — one or more GT features missed
    """
    if not _HAS_RDKIT:
        return "unknown", []

    gt_smiles = record.get("smiles", "")
    gt_structural = record.get("structural_features", {})

    base_ds = model_dataset.dataset if hasattr(model_dataset, "dataset") else model_dataset
    has_edges = hasattr(base_ds, "onehot_feature_edges")

    gt_mol = Chem.MolFromSmiles(gt_smiles) if gt_smiles else None
    gt_maccs = MACCSkeys.GenMACCSKeys(gt_mol) if gt_mol is not None else None

    missed: list[str] = []

    for key_id, entry in pred.structural.items():
        if key_id in MACCS_DEFINITIONS:
            if gt_maccs is None:
                continue
            gt_val = int(gt_maccs[key_id])
            if gt_val not in entry.pred_set:
                missed.append(
                    f"maccs_{key_id}:{MACCS_DEFINITIONS[key_id]['predicate']}"
                    f"(gt={gt_val},pred={entry.pred_set})"
                )

        elif key_id in FEATURE_VECTOR_DEFINITIONS and has_edges:
            predicate = FEATURE_VECTOR_DEFINITIONS[key_id]["predicate"]
            try:
                feat_name, bin_str = predicate.rsplit("_bin_", 1)
                pred_bin = int(bin_str)
            except ValueError:
                continue
            raw = gt_structural.get(feat_name)
            if raw is None:
                continue
            edges = base_ds.onehot_feature_edges.get(feat_name)
            if edges is None:
                continue
            gt_bin = bisect.bisect_left(edges, float(raw))
            if gt_bin != pred_bin:
                missed.append(
                    f"{feat_name}(gt_bin={gt_bin},pred_bin={pred_bin},raw={raw})"
                )

    for feat_name, (lo, hi) in pred.continuous_ranges.items():
        raw = gt_structural.get(feat_name)
        if raw is None:
            continue
        val = float(raw)
        if val < lo or val > hi:
            missed.append(
                f"{feat_name}(gt={val:.4f},range=[{lo:.4f},{hi:.4f}])"
            )

    return ("network_error", missed) if missed else ("asp_bug", [])


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


def _spt_log(max_tan: float, n_candidates: int) -> float:
    """Archived log-based SPT: max_tanimoto / log(1 + |M|). Returns 0.0 when |M| == 0."""
    if n_candidates == 0:
        return 0.0
    return max_tan / math.log(1.0 + n_candidates)


def size_penalised_tanimoto(max_tan: float, n_candidates: int, beta: float = 0.5) -> float:
    """SPT_β(M) = max_tanimoto / |M|^β.

    β=0 → pure max_tanimoto (no size penalty).
    β=1 → linear penalty.
    β=0.5 (default) → square-root penalty.
    |M|=1 → no penalty regardless of β (singleton is the best possible outcome).
    Returns 0.0 when |M| == 0 (max_tanimoto is already 0 in that case).
    """
    if n_candidates == 0:
        return 0.0
    return max_tan / n_candidates ** beta

def expected_utility(max_tan: float, n_candidates: int, lam: float) -> float:
    """U_λ(M) = max_tanimoto(M) - λ × |M|.

    Reward is the maximum Tanimoto similarity to the ground truth over all candidates.
    λ is the cost per candidate relative to the value of a perfect identification.
    λ=0 → pure max_tanimoto; larger λ → stronger penalty for large sets.
    """
    return max_tan - lam * n_candidates


def score_molecule(smiles: str, raw: RawModelOutputs) -> float:
    mol = Chem.MolFromSmiles(smiles)
    maccs = MACCSkeys.GenMACCSKeys(mol)

    logits = torch.tensor(raw.binary)
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)

    score = 0.0
    for j in MACCS_DEFINITIONS:
        if j < len(raw.binary):
            score += log_p[j].item() if maccs[j] else log_1mp[j].item()

    return score


def _best_candidate_rank(gt_smiles: str, sorted_candidates: list[str]) -> tuple[int, int]:
    """Return (rank, N) of the highest-tanimoto candidate in sorted_candidates.

    rank is 1-indexed; 0 means no valid candidate found. N = len(sorted_candidates).
    """
    N = len(sorted_candidates)
    if N == 0:
        return 0, 0
    gt_canon = canonical(gt_smiles)
    if gt_canon is None:
        return 0, N
    gt_canon_ns = canonical_no_stereo(gt_smiles)
    gt_fp = _morgan_fp(gt_canon)

    best_tan = -1.0
    best_rank = 0
    for rank, cand in enumerate(sorted_candidates, start=1):
        cand_canon = canonical(cand)
        if cand_canon is None:
            continue
        if canonical_no_stereo(cand) == gt_canon_ns:
            tan = 1.0
        else:
            tan = tanimoto(gt_canon, cand_canon, gt_fp=gt_fp)
        if tan > best_tan:
            best_tan = tan
            best_rank = rank

    return best_rank, N


def reciprocal_rank(gt_smiles: str, sorted_candidates: list[str]) -> float:
    """1 / rank of the highest-tanimoto candidate. Returns 0.0 when empty."""
    best_rank, _ = _best_candidate_rank(gt_smiles, sorted_candidates)
    return 1.0 / best_rank if best_rank > 0 else 0.0


def _exact_match_rank(gt_smiles: str, candidates: list[str]) -> tuple[int, int]:
    """Return (rank, N) of the exact-match candidate (canonical no-stereo) in candidates.

    rank is 1-indexed; 0 means exact match not in list. N = len(candidates).
    """
    N = len(candidates)
    if N == 0:
        return 0, 0
    gt_canon_ns = canonical_no_stereo(gt_smiles)
    if gt_canon_ns is None:
        return 0, N

    match = 0, N
    for rank, cand in enumerate(candidates, start=1):
        if canonical_no_stereo(cand) == gt_canon_ns:
            if match[0] != 0:
                print("Multiple exact matches found!")
            match = (rank, N)
    return match


def rank_percentile(gt_smiles: str, candidates: list[str]) -> float:
    """1 - (rank - 1) / N of the best candidate (exact match if present, else highest Tanimoto).

    1.0 = best candidate at top, ~0.5 = random baseline, 0.0 = best candidate at bottom.
    Returns 0.0 when the candidate list is empty or no valid candidate found.
    """
    N = len(candidates)
    if N == 0:
        return 0.0
    rank, N = _exact_match_rank(gt_smiles, candidates)
    if rank == 0:
        rank, N = _best_candidate_rank(gt_smiles, candidates)
    if rank == 0:
        return 0.0
    return 1.0 - (rank - 0.5) / N

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
            #if not formula_complexity_filter(smiles, max_score=24):
                #n_skipped += 1
                #continue

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
        #if not formula_complexity_filter(smiles, max_score=24):
            #n_skipped += 1
            #continue

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
                canonicalize=True,
                dedup=True
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
                canonicalize=True,
                dedup=True
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
    spt_beta: float = 0.5,
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

    gt_canon_ns = canonical_no_stereo(gt_smiles)
    gt_fp = _morgan_fp(gt_canon)
    max_tan = 0.0
    exact_match = False

    for cand in candidate_smiles:
        cand_canon = canonical(cand)
        if cand_canon is None:
            continue
        if canonical_no_stereo(cand) == gt_canon_ns:
            exact_match = True
            max_tan = 1.0
            break
        tan = tanimoto(gt_canon, cand_canon, gt_fp=gt_fp)
        if tan > max_tan:
            max_tan = tan

    spt = size_penalised_tanimoto(max_tan, len(candidate_smiles), beta=spt_beta)

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
    p.add_argument(
        "--alpha", type=float, default=None,
        help="The value of alpha (significance level) to try and load conformal thresholds for. Defaults to whatever is set in config.yaml."
    )
    p.add_argument(
        "--eu-lambda", type=float, default=0.05,
        help="Cost per candidate λ for expected utility: EU_λ = I[true ∈ M] - λ×|M|. "
             "Constrained beats unconstrained when λ > ΔCoverage/ΔSetSize (default: 0.01).",
    )
    p.add_argument(
        "--spt-beta", type=float, default=0.5,
        help="Exponent β for size-penalised Tanimoto: SPT_β = max_tan / (1 + |M|)^β. "
             "β=0 → pure max_tanimoto, β=1 → linear penalty (default: 0.5).",
    )
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)

    alpha = args.alpha if args.alpha else config.conformal.alpha

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
    thresholds = load_thresholds(ckpt_path, alpha=alpha, joint=use_joint)
    if thresholds is not None:
        log.info(f"Conformal thresholds loaded (alpha={alpha}, type={args.thresholds_type}).")
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
    n_asp_unsupported = 0
    n_empty_asp_bug = 0
    n_empty_network_error = 0

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
            "empty_reason": "",
            "missed_features": "",
            "candidates": "",
            "reciprocal_rank": 0.0,
            "reciprocal_rank_unsorted": 0.0,
            "rank_percentile": 0.0,
            "rank_percentile_unsorted": 0.0,
            "rank_percentile_randomised": 0.0,
            "expected_utility": 0.0,
            "unc_num_candidates": 0,
            "unc_max_tanimoto": 0.0,
            "unc_size_penalised_tanimoto": 0.0,
            "unc_exact_match": False,
            "unc_timed_out": False,
            "unc_reciprocal_rank": 0.0,
            "unc_reciprocal_rank_unsorted": 0.0,
            "unc_rank_percentile": 0.0,
            "unc_rank_percentile_unsorted": 0.0,
            "unc_rank_percentile_randomised": 0.0,
            "unc_expected_utility": 0.0,
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
        max_tan, spt, exact = evaluate_candidates(gt_smiles_raw, candidates, spt_beta=args.spt_beta)
        rr_unsorted = reciprocal_rank(gt_smiles_raw, candidates)
        rp_unsorted = rank_percentile(gt_smiles_raw, candidates)

        # Sort candidates by model score descending, then compute scored metrics.
        if pred.raw is not None and candidates:
            def _score_key(smi):
                try:
                    return score_molecule(smi, pred.raw)
                except Exception:
                    return float('-inf')
            candidates.sort(key=_score_key, reverse=True)
        rr = reciprocal_rank(gt_smiles_raw, candidates)
        rp = rank_percentile(gt_smiles_raw, candidates)

        rand_candidates = candidates[:]
        random.shuffle(rand_candidates)
        rp_rand = rank_percentile(gt_smiles_raw, rand_candidates)

        eu = expected_utility(max_tan, n_cands, args.eu_lambda)
        row.update({
            "max_tanimoto": round(max_tan, 6),
            "size_penalised_tanimoto": round(spt, 6),
            "exact_match": exact,
            "expected_utility": round(eu, 6),
            "reciprocal_rank": round(rr, 6),
            "reciprocal_rank_unsorted": round(rr_unsorted, 6),
            "rank_percentile": round(rp, 6),
            "rank_percentile_unsorted": round(rp_unsorted, 6),
            "rank_percentile_randomised": round(rp_rand, 6),
            "candidates": " | ".join(candidates[: args.max_candidates_saved]),
        })
        # Preserve timeout/empty status; only upgrade to "ok" on clean success.
        if row["status"] not in ("asp_timeout", "empty_candidates") and n_cands > 0:
            row["status"] = "ok"

        # ── unconstrained ASP solve (same formula space, no structural constraints) ──
        if any_timed_out:
            unc_candidates, unc_timed_out = [], True
        else:
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

        unc_max_tan, unc_spt, unc_exact = evaluate_candidates(gt_smiles_raw, unc_candidates, spt_beta=args.spt_beta)
        unc_rr_unsorted = reciprocal_rank(gt_smiles_raw, unc_candidates)
        unc_rp_unsorted = rank_percentile(gt_smiles_raw, unc_candidates)

        if pred.raw is not None and unc_candidates:
            def _unc_score_key(smi):
                try:
                    return score_molecule(smi, pred.raw)
                except Exception:
                    return float('-inf')
            unc_candidates.sort(key=_unc_score_key, reverse=True)
        unc_rr = reciprocal_rank(gt_smiles_raw, unc_candidates)
        unc_rp = rank_percentile(gt_smiles_raw, unc_candidates)

        unc_rand_candidates = unc_candidates[:]
        random.shuffle(unc_rand_candidates)
        unc_rp_rand = rank_percentile(gt_smiles_raw, unc_rand_candidates)

        unc_eu = expected_utility(unc_max_tan, unc_n_cands, args.eu_lambda)
        row.update({
            "unc_num_candidates": unc_n_cands,
            "unc_max_tanimoto": round(unc_max_tan, 6),
            "unc_size_penalised_tanimoto": round(unc_spt, 6),
            "unc_exact_match": unc_exact,
            "unc_timed_out": unc_timed_out,
            "unc_reciprocal_rank": round(unc_rr, 6),
            "unc_reciprocal_rank_unsorted": round(unc_rr_unsorted, 6),
            "unc_rank_percentile": round(unc_rp, 6),
            "unc_rank_percentile_unsorted": round(unc_rp_unsorted, 6),
            "unc_rank_percentile_randomised": round(unc_rp_rand, 6),
            "unc_expected_utility": round(unc_eu, 6),
        })

        # Reclassify: if constrained AND unconstrained both empty (and unc didn't
        # time out), the formula itself is unsupported by ASP — not a pipeline failure.
        if row["status"] == "empty_candidates" and unc_n_cands == 0 and not unc_timed_out:
            row["status"] = "asp_unsupported"
            n_asp_unsupported += 1

        # Diagnose remaining empty constrained sets: check whether GT structural
        # features are all covered by the prediction sets.
        # If yes → ASP constraint bug (network was right, ASP implementation wrong).
        # If no  → network error (wrong constraints excluded the true molecule).
        elif row["status"] == "empty_candidates":
            reason, missed = diagnose_empty_candidates(record, pred, model_dataset)
            row["empty_reason"] = reason
            row["missed_features"] = "; ".join(missed)
            if reason == "asp_bug":
                row["status"] = "empty_asp_bug"
                n_empty_asp_bug += 1
            else:
                row["status"] = "empty_network_error"
                n_empty_network_error += 1

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

    # Set 1: constrained attempts where the network is responsible for the outcome.
    # Includes empty sets caused by wrong network predictions (empty_network_error).
    # Excludes ASP-unsupported formulas and empty sets caused by ASP bugs.
    con_ok_rows = [r for r in per_sample_rows if r["status"] in ("ok", "empty_network_error")]
    n_con_ok = len(con_ok_rows)
    con_exact = sum(1 for r in con_ok_rows if r["exact_match"])
    con_eu_arr = np.array([r["expected_utility"] for r in con_ok_rows], dtype=float)
    con_max_tans = np.array([r["max_tanimoto"] for r in con_ok_rows], dtype=float)
    con_spts = np.array([r["size_penalised_tanimoto"] for r in con_ok_rows], dtype=float)
    con_n_cands_arr = np.array([r["num_candidates"] for r in con_ok_rows], dtype=int)
    con_rr_arr = np.array([r["reciprocal_rank"] for r in con_ok_rows], dtype=float)
    con_rr_unsorted_arr = np.array([r["reciprocal_rank_unsorted"] for r in con_ok_rows], dtype=float)
    con_rp_arr = np.array([r["rank_percentile"] for r in con_ok_rows], dtype=float)
    con_rp_unsorted_arr = np.array([r["rank_percentile_unsorted"] for r in con_ok_rows], dtype=float)
    con_rp_rand_arr = np.array([r["rank_percentile_randomised"] for r in con_ok_rows], dtype=float)
    con_exact_rows = [r for r in con_ok_rows if r["exact_match"]]
    con_rp_rand_exact_arr = np.array([r["rank_percentile_randomised"] for r in con_exact_rows], dtype=float)

    # Set 2: paired rows where both constrained and unconstrained succeeded
    # (subset of set 1; used for fair head-to-head comparison).
    paired_rows = [r for r in con_ok_rows if not r["unc_timed_out"]]
    n_paired = len(paired_rows)
    paired_exact = sum(1 for r in paired_rows if r["exact_match"])
    paired_max_tans = np.array([r["max_tanimoto"] for r in paired_rows], dtype=float)
    paired_spts = np.array([r["size_penalised_tanimoto"] for r in paired_rows], dtype=float)
    paired_n_cands_arr = np.array([r["num_candidates"] for r in paired_rows], dtype=int)
    paired_rr_arr = np.array([r["reciprocal_rank"] for r in paired_rows], dtype=float)
    paired_rr_unsorted_arr = np.array([r["reciprocal_rank_unsorted"] for r in paired_rows], dtype=float)
    paired_rp_arr = np.array([r["rank_percentile"] for r in paired_rows], dtype=float)
    paired_rp_unsorted_arr = np.array([r["rank_percentile_unsorted"] for r in paired_rows], dtype=float)
    paired_rp_rand_arr = np.array([r["rank_percentile_randomised"] for r in paired_rows], dtype=float)
    paired_exact_rows = [r for r in paired_rows if r["exact_match"]]
    paired_rp_rand_exact_arr = np.array([r["rank_percentile_randomised"] for r in paired_exact_rows], dtype=float)
    paired_eu_arr = np.array([r["expected_utility"] for r in paired_rows], dtype=float)
    unc_n_with_cands = sum(1 for r in paired_rows if r["unc_num_candidates"] > 0)
    unc_exact = sum(1 for r in paired_rows if r["unc_exact_match"])
    unc_eu_arr = np.array([r["unc_expected_utility"] for r in paired_rows], dtype=float)
    unc_max_tans = np.array([r["unc_max_tanimoto"] for r in paired_rows], dtype=float)
    unc_spts = np.array([r["unc_size_penalised_tanimoto"] for r in paired_rows], dtype=float)
    unc_n_cands_arr = np.array([r["unc_num_candidates"] for r in paired_rows], dtype=int)
    unc_rr_arr = np.array([r["unc_reciprocal_rank"] for r in paired_rows], dtype=float)
    unc_rr_unsorted_arr = np.array([r["unc_reciprocal_rank_unsorted"] for r in paired_rows], dtype=float)
    unc_rp_arr = np.array([r["unc_rank_percentile"] for r in paired_rows], dtype=float)
    unc_rp_unsorted_arr = np.array([r["unc_rank_percentile_unsorted"] for r in paired_rows], dtype=float)
    unc_rp_rand_arr = np.array([r["unc_rank_percentile_randomised"] for r in paired_rows], dtype=float)
    unc_exact_rows = [r for r in paired_rows if r["unc_exact_match"]]
    unc_rp_rand_exact_arr = np.array([r["unc_rank_percentile_randomised"] for r in unc_exact_rows], dtype=float)

    # Cases where constrained solved but unconstrained timed out.
    n_con_ok_unc_timeout = sum(1 for r in con_ok_rows if r["unc_timed_out"])
    pct_con_ok_unc_timeout = round(100.0 * n_con_ok_unc_timeout / n_con_ok, 2) if n_con_ok else 0.0

    def _con_metrics(rows, max_t, spts_, n_cands_, n_exact_, eu_arr=None, rr_arr=None, rr_unsorted_arr=None, rp_arr=None, rp_unsorted_arr=None, rp_rand_arr=None, rp_rand_exact_arr=None) -> dict:
        n = len(rows)
        n_exact_for_rp = len(rp_rand_exact_arr) if rp_rand_exact_arr is not None else 0
        out = {
            "n_metric": n,
            "n_exact_match": n_exact_,
            "pct_exact_match": round(100.0 * n_exact_ / n, 2) if n else 0.0,
            "mean_expected_utility": round(float(np.mean(eu_arr)), 6) if (n and eu_arr is not None) else 0.0,
            "mean_max_tanimoto": round(float(np.mean(max_t)), 6) if n else 0.0,
            "mean_size_penalised_tanimoto": round(float(np.mean(spts_)), 6) if n else 0.0,
            "mean_reciprocal_rank": round(float(np.mean(rr_arr)), 6) if (n and rr_arr is not None) else 0.0,
            "mean_reciprocal_rank_unsorted": round(float(np.mean(rr_unsorted_arr)), 6) if (n and rr_unsorted_arr is not None) else 0.0,
            "mean_rank_percentile": round(float(np.mean(rp_arr)), 6) if (n and rp_arr is not None) else 0.0,
            "mean_rank_percentile_unsorted": round(float(np.mean(rp_unsorted_arr)), 6) if (n and rp_unsorted_arr is not None) else 0.0,
            "mean_rank_percentile_randomised": round(float(np.mean(rp_rand_arr)), 6) if (n and rp_rand_arr is not None) else 0.0,
            "mean_rank_percentile_randomised_exact": round(float(np.mean(rp_rand_exact_arr)), 6) if n_exact_for_rp else 0.0,
            "mean_num_candidates": round(float(np.mean(n_cands_)), 2) if n else 0.0,
            "mean_num_candidates_exact_match": round(float(np.sum(n_cands_) / (n_exact_)), 2) if n_exact_ else 0.0,
            "median_num_candidates": int(np.median(n_cands_)) if n else 0,
        }
        return out

    summary = {
        "n_evaluated": n_eval,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "n_asp_timeout": n_timeout,
        "n_empty_candidates": n_empty,
        "n_asp_unsupported": n_asp_unsupported,
        "n_empty_asp_bug": n_empty_asp_bug,
        "n_empty_network_error": n_empty_network_error,
        "n_con_ok_unc_timeout": n_con_ok_unc_timeout,
        "pct_con_ok_unc_timeout": pct_con_ok_unc_timeout,
        # All constrained successes (max coverage).
        "constrained": _con_metrics(con_ok_rows, con_max_tans, con_spts, con_n_cands_arr, con_exact, con_eu_arr, con_rr_arr, con_rr_unsorted_arr, con_rp_arr, con_rp_unsorted_arr, con_rp_rand_arr, con_rp_rand_exact_arr),
        # Paired subset: both solves succeeded (for head-to-head comparison).
        "constrained_paired": _con_metrics(paired_rows, paired_max_tans, paired_spts, paired_n_cands_arr, paired_exact, paired_eu_arr, paired_rr_arr, paired_rr_unsorted_arr, paired_rp_arr, paired_rp_unsorted_arr, paired_rp_rand_arr, paired_rp_rand_exact_arr),
        "unconstrained": {
            "n_metric": n_paired,
            "n_with_candidates": unc_n_with_cands,
            "n_exact_match": unc_exact,
            "pct_with_candidates": round(100.0 * unc_n_with_cands / n_paired, 2) if n_paired else 0.0,
            "pct_exact_match": round(100.0 * unc_exact / n_paired, 2) if n_paired else 0.0,
            "mean_expected_utility": round(float(np.mean(unc_eu_arr)), 6) if n_paired else 0.0,
            "mean_max_tanimoto": round(float(np.mean(unc_max_tans)), 6) if n_paired else 0.0,
            "mean_size_penalised_tanimoto": round(float(np.mean(unc_spts)), 6) if n_paired else 0.0,
            "mean_reciprocal_rank": round(float(np.mean(unc_rr_arr)), 6) if n_paired else 0.0,
            "mean_reciprocal_rank_unsorted": round(float(np.mean(unc_rr_unsorted_arr)), 6) if n_paired else 0.0,
            "mean_rank_percentile": round(float(np.mean(unc_rp_arr)), 6) if n_paired else 0.0,
            "mean_rank_percentile_unsorted": round(float(np.mean(unc_rp_unsorted_arr)), 6) if n_paired else 0.0,
            "mean_rank_percentile_randomised": round(float(np.mean(unc_rp_rand_arr)), 6) if n_paired else 0.0,
            "mean_rank_percentile_randomised_exact": round(float(np.mean(unc_rp_rand_exact_arr)), 6) if len(unc_exact_rows) else 0.0,
            "mean_num_candidates": round(float(np.mean(unc_n_cands_arr)), 2) if n_paired else 0.0,
            "median_num_candidates": int(np.median(unc_n_cands_arr)) if n_paired else 0,
        },
        "checkpoint": str(ckpt_path),
        "split_or_file": (
            args.test_set if args.test_set else args.split_pkl
        ),
        "asp_timeout_s": args.asp_timeout,
        "eu_lambda": args.eu_lambda,
        "spt_beta": args.spt_beta,
        "manual_threshold": args.threshold,
        "conformal": thresholds is not None,
        "conformal_type": args.thresholds_type if thresholds is not None else None,
        "conformal_thresholds_path": (
            str(thresholds_path(ckpt_path, alpha=config.conformal.alpha, joint=use_joint))
            if thresholds is not None else None
        ),
    }

    con = summary["constrained"]
    conp = summary["constrained_paired"]
    unc = summary["unconstrained"]
    print("\n=== Molecule-Level Evaluation Summary ===")
    print(f"  Checkpoint            : {ckpt_path}")
    if thresholds is not None:
        print(f"  Conformal type        : {summary['conformal_type']}")
        print(f"  Conformal thresholds  : {summary['conformal_thresholds_path']}")
    else:
        print(f"  Manual threshold       : {args.threshold}  (no conformal thresholds)")
    print(f"  Samples evaluated           : {n_eval}")
    print(f"  Failed                      : {n_failed}")
    print(f"  Skipped (filtered)          : {n_skipped}")
    print(f"  ASP timeouts                : {n_timeout}")
    print(f"  Empty candidate sets        : {n_empty}")
    print(f"  ASP-unsupported             : {n_asp_unsupported}  (formula unsupported → excluded)")
    print(f"  Empty — ASP constraint bug  : {n_empty_asp_bug}  (GT covered by pred sets, ASP failed → excluded)")
    print(f"  Empty — network error       : {n_empty_network_error}  (GT not covered by pred sets → scored 0)")
    print(f"  Con. solved, unc. timeout   : {n_con_ok_unc_timeout} / {n_con_ok}  ({pct_con_ok_unc_timeout:.1f}%)")

    print(f"\n--- Constrained (all successes, n={n_con_ok}) ---")
    print(f"  Exact match                 : {con_exact}  ({con['pct_exact_match']:.1f}%)")
    print(f"  Mean max-Tanimoto           : {con['mean_max_tanimoto']:.4f}")
    print(f"  Mean expected utility (λ={args.eu_lambda}) : {con['mean_expected_utility']:.4f}")
    print(f"  Mean SPT                    : {con['mean_size_penalised_tanimoto']:.4f}")
    print(f"  Mean reciprocal rank        : {con['mean_reciprocal_rank']:.4f}  (unsorted: {con['mean_reciprocal_rank_unsorted']:.4f})")
    print(f"  Mean rank percentile        : {con['mean_rank_percentile']:.4f}  (unsorted: {con['mean_rank_percentile_unsorted']:.4f}, randomised: {con['mean_rank_percentile_randomised']:.4f})  [exact match or best Tanimoto; 1=top, ~0.5=random]")
    print(f"  Mean rank percentile (rand) : {con['mean_rank_percentile_randomised_exact']:.4f}  [exact-match rows only; ~0.5=random, n={len(con_exact_rows)}]")
    print(f"  Mean num candidates         : {con['mean_num_candidates']:.1f}")
    print(f"  Mean num candidates (exact) : {con['mean_num_candidates_exact_match']:.1f}")
    print(f"  Median num candidates       : {con['median_num_candidates']}")

    print(f"\n--- Constrained vs Unconstrained (paired, n={n_paired}) ---")
    print(f"  {'':30s} {'Constrained':>12}  {'Unconstrained':>13}")
    print(f"  {'Exact match':30s} {paired_exact:>10}  ({conp['pct_exact_match']:4.1f}%)   {unc_exact:>10}  ({unc['pct_exact_match']:4.1f}%)")
    print(f"  {'Mean max-Tanimoto':30s} {conp['mean_max_tanimoto']:>12.4f}  {unc['mean_max_tanimoto']:>13.4f}")
    print(f"  {'Mean EU (λ={:.3f})'.format(args.eu_lambda):30s} {conp['mean_expected_utility']:>12.4f}  {unc['mean_expected_utility']:>13.4f}")
    print(f"  {'Mean SPT':30s} {conp['mean_size_penalised_tanimoto']:>12.4f}  {unc['mean_size_penalised_tanimoto']:>13.4f}")
    print(f"  {'MRR (scored)':30s} {conp['mean_reciprocal_rank']:>12.4f}  {unc['mean_reciprocal_rank']:>13.4f}")
    print(f"  {'MRR (unsorted)':30s} {conp['mean_reciprocal_rank_unsorted']:>12.4f}  {unc['mean_reciprocal_rank_unsorted']:>13.4f}")
    print(f"  {'Rank % (scored)':30s} {conp['mean_rank_percentile']:>12.4f}  {unc['mean_rank_percentile']:>13.4f}  [exact match rank; 1=top]")
    print(f"  {'Rank % (unsorted)':30s} {conp['mean_rank_percentile_unsorted']:>12.4f}  {unc['mean_rank_percentile_unsorted']:>13.4f}  [~0.5=random]")
    print(f"  {'Rank % (randomised)':30s} {conp['mean_rank_percentile_randomised']:>12.4f}  {unc['mean_rank_percentile_randomised']:>13.4f}  [exact or best Tan.; ~0.5=random]")
    print(f"  {'Rank % (rand, exact only)':30s} {conp['mean_rank_percentile_randomised_exact']:>12.4f}  {unc['mean_rank_percentile_randomised_exact']:>13.4f}  [exact-match rows only; ~0.5=random]")
    print(f"  {'Mean num candidates':30s} {conp['mean_num_candidates']:>12.1f}  {unc['mean_num_candidates']:>13.1f}")
    print(f"  {'Mean num candidates (exact)':30s} {conp['mean_num_candidates_exact_match']:>12.1f}  {unc['mean_num_candidates']:>13.1f}")
    print(f"  {'Median num candidates':30s} {conp['median_num_candidates']:>12d}  {unc['median_num_candidates']:>13d}")
    if n_paired:
        spt_delta = conp['mean_size_penalised_tanimoto'] - unc['mean_size_penalised_tanimoto']
        cand_ratio = (conp['mean_num_candidates'] / unc['mean_num_candidates']
                      if unc['mean_num_candidates'] > 0 else float('nan'))
        delta_t = unc['mean_max_tanimoto'] - conp['mean_max_tanimoto']
        delta_m = unc['mean_num_candidates'] - conp['mean_num_candidates']
        lam_star = delta_t / delta_m if delta_m != 0 else None
        k_star = 1.0 / lam_star if (lam_star is not None and lam_star > 0) else None
        summary["eu_lambda_star"] = round(lam_star, 6) if lam_star is not None else None
        summary["eu_k_star"] = round(k_star, 1) if k_star is not None else None
        summary["eu_delta_t"] = round(delta_t, 6)
        summary["eu_delta_m"] = round(delta_m, 2)
        print(f"\n  SPT delta (constrained - unconstrained): {spt_delta:+.4f}")
        print(f"  Candidate count ratio (con/unc):         {cand_ratio:.3f}")
        print(f"  ΔT̄ (unc - con max-T):                   {delta_t:+.4f}")
        print(f"  Δ|M| (unc - con set size):               {delta_m:+.1f}")
        if lam_star is not None:
            k_str = f"  →  k* = {k_star:.1f}" if k_star is not None else "  →  k* undefined (λ* ≤ 0)"
            print(f"  λ* = ΔT̄/Δ|M| (EU crossover):           {lam_star:.4f}{k_str}")
        else:
            print(f"  λ* = ΔT̄/Δ|M| (EU crossover):           undefined (Δ|M| = 0)")

    _save_csv(per_sample_rows, out_dir / "molecule_evaluation_test.csv")
    _save_json(summary, out_dir / "molecule_evaluation_summary.json")


if __name__ == "__main__":
    main()
