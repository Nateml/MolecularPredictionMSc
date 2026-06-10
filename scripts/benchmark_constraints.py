#!/usr/bin/env python3
"""
Compare ASP solve time: no constraints vs. ground-truth constraints.

For each test molecule, runs solve with:
  (A) unconstrained  — atom counts only
  (B) maccs_only     — atom counts + all 166 ground-truth MACCS keys
  (C) continuous_only — atom counts + all 89 structural feature exact ranges
  (D) all            — atom counts + MACCS + structural features

Dataset: data/MoNA-filtered-curated-with-maccs-and-structure-deduped.json
Schema:  data/test_maccs_structural_schema.json

Usage:
    python benchmark_constraints.py
    python benchmark_constraints.py --timeout 60 --runs 2
    python benchmark_constraints.py --formulas medium
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from asp.bridge import solve
from predicates.compiler import compile_asp_constraint
from predicates.interface import PredictionInterface, PredictionEntry, ContinuousRange
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS_CONTINUOUS

DATASET_PATH = "data/MoNA-filtered-curated-with-maccs-and-structure-deduped.json"
SCHEMA_PATH = "data/test_maccs_structural_schema.json"

# ---------------------------------------------------------------------------
# Target formulas grouped by difficulty
# ---------------------------------------------------------------------------

FORMULA_ATOM_COUNTS = {
    "C3H6O":  dict(num_carbons=3, num_hydrogens=6,  num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0),
    "C5H10O": dict(num_carbons=5, num_hydrogens=10, num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0),
    "C7H14O": dict(num_carbons=7, num_hydrogens=14, num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0),
    "C6H7N":  dict(num_carbons=6, num_hydrogens=7,  num_oxygens=0, num_nitrogens=1, num_sulfurs=0, num_phosphorus=0),
    "C7H8":   dict(num_carbons=7, num_hydrogens=8,  num_oxygens=0, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0),
    "C7H8O":  dict(num_carbons=7, num_hydrogens=8,  num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0),
    "C8H10":  dict(num_carbons=8, num_hydrogens=10, num_oxygens=0, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0),
}

FORMULA_SETS = {
    "small":  ["C3H6O"],
    "medium": ["C3H6O", "C5H10O", "C7H14O"],
    "large":  ["C6H7N", "C7H8", "C7H8O", "C8H10"],
    "all":    list(FORMULA_ATOM_COUNTS.keys()),
}

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def maccs_entries(maccs_list: list) -> dict:
    entries = {}
    for key_id in MACCS_DEFINITIONS:
        if key_id < len(maccs_list):
            entries[key_id] = PredictionEntry(pred_set=[int(maccs_list[key_id])])
    return entries


def continuous_entries(structural_features: dict) -> dict:
    entries = {}
    for feat_name in FEATURE_VECTOR_DEFINITIONS_CONTINUOUS:
        if feat_name in structural_features:
            val = float(structural_features[feat_name])
            entries[feat_name] = ContinuousRange(lo=val, hi=val)
    return entries


# ---------------------------------------------------------------------------
# Solve wrapper
# ---------------------------------------------------------------------------

def timed_solve(atom_counts: dict, pred_entries: dict, timeout: int) -> tuple[float, int]:
    entries = {**pred_entries, **atom_counts}
    interface = PredictionInterface.model_validate(entries)
    asp_code, _ = compile_asp_constraint(interface)
    t0 = time.monotonic()
    _, count, _ = solve(asp_code, atom_counts, output_list_of_smiles=True, timeout=timeout)
    return time.monotonic() - t0, count


# ---------------------------------------------------------------------------
# Dataset lookup
# ---------------------------------------------------------------------------

def find_molecules(target_formulas: list[str]) -> list[tuple[str, str, dict, dict]]:
    """Return (formula, smiles, atom_counts, mol_data) for first match per formula."""
    found: dict[str, tuple] = {}
    with open(DATASET_PATH) as f:
        for line in f:
            if len(found) == len(target_formulas):
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            mol = Chem.MolFromSmiles(ex["smiles"])
            if mol is None:
                continue
            formula = CalcMolFormula(mol)
            if formula in target_formulas and formula not in found:
                found[formula] = (ex["smiles"], FORMULA_ATOM_COUNTS[formula], ex)

    return [
        (formula, *found[formula])
        for formula in target_formulas
        if formula in found
    ]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_molecule(mol_data: dict, atom_counts: dict, n_runs: int, timeout: int) -> dict:
    maccs_e = maccs_entries(mol_data["maccs"])
    cont_e = continuous_entries(mol_data["structural_features"])

    conditions = {
        "unconstrained":   {},
        "maccs_only":      maccs_e,
        "continuous_only": cont_e,
        "all":             {**maccs_e, **cont_e},
    }

    results = {}
    for cond, pred_entries in conditions.items():
        times, n_cands, err = [], 0, None
        for _ in range(n_runs):
            try:
                elapsed, n_cands = timed_solve(atom_counts, pred_entries, timeout)
                times.append(elapsed)
            except Exception as e:
                err = str(e)
                break
        results[cond] = {
            "mean_s": statistics.mean(times) if times else float("nan"),
            "n_candidates": n_cands,
            "error": err,
            "runs": len(times),
        }
    return results


def print_results(all_results: list[tuple[str, str, dict]]) -> None:
    col = max(len(f) for f, _, _ in all_results)
    conds = ["unconstrained", "maccs_only", "continuous_only", "all"]

    print(f"\n{'Formula':<{col}}  {'condition':<18}  {'mean(s)':>8}  {'cands':>6}  {'speedup':>8}")
    print("-" * (col + 52))

    for formula, smiles, results in all_results:
        baseline = results.get("unconstrained", {}).get("mean_s", float("nan"))
        first = True
        for cond in conds:
            r = results.get(cond, {})
            label = formula if first else ""
            first = False
            if r.get("error") and r.get("runs", 0) == 0:
                print(f"{label:<{col}}  {cond:<18}  ERROR: {r['error']}")
                continue
            mean_s = r.get("mean_s", float("nan"))
            n_cands = r.get("n_candidates", 0)
            if mean_s == mean_s and mean_s > 0 and baseline == baseline and baseline > 0:
                speedup = f"{baseline / mean_s:.2f}x"
            else:
                speedup = "n/a"
            print(f"{label:<{col}}  {cond:<18}  {mean_s:>8.3f}  {n_cands:>6}  {speedup:>8}")
        print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--formulas", choices=list(FORMULA_SETS.keys()), default="all")
    args = p.parse_args()

    target_formulas = FORMULA_SETS[args.formulas]
    molecules = find_molecules(target_formulas)

    print(f"ASP constraint benchmark")
    print(f"  Dataset : {DATASET_PATH}")
    print(f"  Schema  : {SCHEMA_PATH}")
    print(f"  Formulas: {args.formulas} ({len(molecules)} found)")
    print(f"  Runs    : {args.runs}  Timeout: {args.timeout}s")

    all_results = []
    for formula, smiles, atom_counts, mol_data in molecules:
        print(f"\n  Solving {formula}  ({smiles[:50]}) ...", flush=True)
        results = run_molecule(mol_data, atom_counts, args.runs, args.timeout)
        all_results.append((formula, smiles, results))
        for cond, r in results.items():
            err = f" ERROR={r['error']}" if r.get("error") else ""
            print(f"    {cond:<18}  {r['mean_s']:.3f}s  {r['n_candidates']} cands{err}")

    print_results(all_results)


if __name__ == "__main__":
    main()
