#!/usr/bin/env python3
"""
Oracle evaluation: replace neural net predictions with ground truth features,
then check whether the ASP solver recovers the true molecule.

Tests ASP bridge soundness — given perfect feature constraints, the true
molecule should appear in the candidate set.

Usage:
    python oracle_eval.py
    python oracle_eval.py --limit 50 --timeout 30
    python oracle_eval.py --show-constraints --limit 5
    python oracle_eval.py --diagnose --limit 10
    python oracle_eval.py --output results.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config
from data.datasets import dataset_from_schema, RawFeatureDataset
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS
from predicates.interface import PredictionInterface, PredictionEntry
from predicates.compiler import compile_asp_constraint
from asp_bridge import solve, is_asp_supported

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    print("Warning: rdkit not available; SMILES comparison will be string-exact.", file=sys.stderr)

_ELEMENT_TO_COUNT_KEY: dict[str, str] = {
    "element_C": "num_carbons",
    "element_H": "num_hydrogens",
    "element_N": "num_nitrogens",
    "element_O": "num_oxygens",
    "element_P": "num_phosphorus",
    "element_S": "num_sulfurs",
}

_PREDICATE_TO_FVD_KEY: dict[str, int] = {
    v["predicate"]: k for k, v in FEATURE_VECTOR_DEFINITIONS.items()
}


def canonical(smi: str) -> str | None:
    if not _HAS_RDKIT:
        return smi
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def gt_entries_from_record(record: dict, dataset: RawFeatureDataset) -> dict:
    """Build PredictionInterface entries from ground truth record data."""
    entries: dict = {}

    # Binary: include all MACCS keys that have ASP definitions, exact GT value
    maccs_bits: list[int] = record["maccs"]
    for i, bit in enumerate(maccs_bits):
        if i in MACCS_DEFINITIONS:
            entries[i] = PredictionEntry(pred_set=[bit], proba=[1.0])

    # Categorical: bin GT raw value using schema edges
    features: dict = record.get("structural_features", {})
    for feat in dataset.onehot_feature_order:
        raw = features.get(feat)
        if raw is None:
            continue
        class_idx = bisect.bisect_left(dataset.onehot_feature_edges[feat], float(raw))
        predicate_name = f"{feat}_bin_{class_idx}"
        key = _PREDICATE_TO_FVD_KEY.get(predicate_name)
        if key is not None:
            entries[key] = PredictionEntry(pred_set=[1], proba=[1.0])

    # Atom counts: exact GT values
    for elem_feat, count_key in _ELEMENT_TO_COUNT_KEY.items():
        raw = features.get(elem_feat)
        entries[count_key] = max(0, int(round(float(raw)))) if raw is not None else 0

    return entries


def _entry_label(key: int | str) -> str:
    """Human-readable name for an entries dict key."""
    if isinstance(key, str):
        return key
    if key in MACCS_DEFINITIONS:
        return f"MACCS[{key}] {MACCS_DEFINITIONS[key]['predicate']}"
    if key in FEATURE_VECTOR_DEFINITIONS:
        return f"FVD[{key}] {FEATURE_VECTOR_DEFINITIONS[key]['predicate']}"
    return str(key)


def diagnose_miss(
    entries: dict,
    gt_canon: str,
    timeout: int,
    known_buggy_keys: set,
) -> list[tuple[str, int]]:
    """Greedy sequential constraint elimination to find culprit constraints.

    Iterates predicate constraints one at a time. When removing a constraint
    unblocks the true molecule, it is permanently removed from the working set
    and recorded as a culprit. Subsequent constraints are tested against this
    already-reduced set, so interacting bad constraints are peeled away one by
    one and each appears as a culprit in sequence.

    Atom count entries (int values) are skipped — removing them forces the
    count to 0, which is a different (unrelated) problem.
    """
    from tqdm import tqdm
    # strip already-confirmed buggy constraints — no point re-testing them
    working_entries = {k: v for k, v in entries.items() if k not in known_buggy_keys}
    predicate_keys = [k for k, v in working_entries.items() if isinstance(v, PredictionEntry)]
    random.shuffle(predicate_keys)

    culprits: list[tuple[str, int]] = []

    for key in tqdm(predicate_keys, desc="    diagnosing", unit="constraint", leave=False):
        del working_entries[key]  # strip unconditionally — may need multiple removals
        try:
            interface = PredictionInterface.model_validate(working_entries)
            asp_constraints, atom_counts = compile_asp_constraint(interface)
            candidates, _ = solve(
                asp_constraints, atom_counts,
                output_list_of_smiles=True,
                timeout=timeout,
            )
            cand_canons = {canonical(s) for s in candidates} - {None}
            if gt_canon in cand_canons:
                label = _entry_label(key)
                value = entries[key].pred_set[0]
                culprits.append((label, value))
                known_buggy_keys.add(key)
                print(f"    CULPRIT: {label} = {value}")
                break
        except Exception as e:
            print(f"    ERROR testing {_entry_label(key)}: {e}", file=sys.stderr)

    if not culprits:
        print("    No culprits found (constraints correct but molecule unreachable — ASP coverage gap?)")

    return culprits


def _solve_record(
    dataset_idx: int,
    gt_smi_raw: str,
    gt_canon: str | None,
    asp_constraints: list[str],
    atom_counts: dict,
    timeout: int,
) -> dict:
    """Worker: solve one record. Module-level so ProcessPoolExecutor can pickle it."""
    t0 = time.monotonic()
    candidates, _ = solve(asp_constraints, atom_counts, output_list_of_smiles=True, timeout=timeout)
    elapsed = time.monotonic() - t0
    cand_canons = {canonical(s) for s in candidates} - {None}
    found = gt_canon in cand_canons if gt_canon else False
    return {
        "idx": dataset_idx,
        "smiles": gt_smi_raw,
        "found": found,
        "n_candidates": len(candidates),
        "elapsed_s": round(elapsed, 2),
        "culprits": [],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle evaluation of ASP bridge using ground truth features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", default=None, help="Path to dataset JSON (default: from config)")
    p.add_argument("--schema", default=None, help="Path to schema JSON (default: from config)")
    p.add_argument("--limit", type=int, default=None, help="Max records to evaluate")
    p.add_argument("--timeout", type=int, default=config.solver.timeout,
                   help=f"Per-solve wall-clock timeout in seconds (0=no limit, default: {config.solver.timeout})")
    p.add_argument("--diagnose", action="store_true",
                   help="On MISS, remove constraints one at a time to find culprit(s)")
    p.add_argument("--diagnose-timeout", type=int, default=None, metavar="SECS",
                   help="Separate (usually shorter) timeout for diagnosis solves (default: same as --timeout)")
    p.add_argument("--show-constraints", action="store_true",
                   help="Print compiled ASP constraints for each record")
    p.add_argument("--show-formula", action="store_true",
                   help="Print the molecular formula for each record")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Save per-record results to a JSON file")
    p.add_argument("--report", default=None, metavar="FILE",
                   help="Save a human-readable markdown report to FILE")
    p.add_argument("--mol-size-limit", type=int, default=None,
                   help="Skip molecules with more than this many total atoms")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2),
                   help="Parallel worker processes for solving (default: cpu_count//2, ignored with --diagnose)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    diagnose_timeout = args.diagnose_timeout if args.diagnose_timeout is not None else args.timeout

    data_path = Path(args.dataset) if args.dataset else config.data.resolved_dataset_path()
    schema_path = Path(args.schema) if args.schema else config.data.resolved_schema_path()

    dataset = dataset_from_schema(data_path, schema_path, spectrum_len=config.model.spectrum_length)
    if not isinstance(dataset, RawFeatureDataset):
        print("Error: oracle_eval requires RawFeatureDataset (needs raw feature schema).", file=sys.stderr)
        sys.exit(1)

    n_total = len(dataset)
    limit = min(args.limit, n_total) if args.limit else n_total
    print(f"Dataset: {data_path.name} ({n_total} records, evaluating {limit})")
    print(f"Timeout: {args.timeout}s per solve (0 = no limit)" if args.timeout else "Timeout: none")
    print()

    # --- Collection phase (sequential): load, filter, compile constraints ---
    WorkItem = tuple  # (dataset_idx, gt_smi_raw, gt_canon, asp_constraints, atom_counts, entries)
    work_items: list[WorkItem] = []
    raw_idx = n_skip = 0

    while len(work_items) < limit:
        record = dataset._load_record(raw_idx)
        dataset_idx = raw_idx
        raw_idx += 1

        gt_smi_raw = record.get("smiles")
        if not gt_smi_raw:
            n_skip += 1
            continue
        gt_canon = canonical(str(gt_smi_raw))
        supported, _ = is_asp_supported(str(gt_smi_raw))
        if gt_canon is None or not supported:
            n_skip += 1
            continue

        entries = gt_entries_from_record(record, dataset)
        interface = PredictionInterface.model_validate(entries)
        asp_constraints, atom_counts = compile_asp_constraint(interface)

        mol_size = sum(atom_counts.values())
        if args.mol_size_limit and mol_size > args.mol_size_limit:
            n_skip += 1
            continue

        if args.show_formula:
            formula = (f"C{atom_counts.get('num_carbons',0)}"
                       f"H{atom_counts.get('num_hydrogens',0)}"
                       f"N{atom_counts.get('num_nitrogens',0)}"
                       f"O{atom_counts.get('num_oxygens',0)}"
                       f"P{atom_counts.get('num_phosphorus',0)}"
                       f"S{atom_counts.get('num_sulfurs',0)}")
            print(f"\n[{dataset_idx}] {gt_smi_raw[:70]}  formula={formula}  {mol_size} atoms")
        if args.show_constraints:
            for c in asp_constraints:
                print(f"  {c}")

        work_items.append((dataset_idx, gt_smi_raw, gt_canon, asp_constraints, atom_counts, entries))

    # --- Evaluation phase ---
    results = []
    n_found = 0

    if args.diagnose:
        # Sequential — diagnose_miss mutates known_buggy_keys across records
        known_buggy_keys: set = set()
        for dataset_idx, gt_smi_raw, gt_canon, asp_constraints, atom_counts, entries in work_items:
            culprits: list[tuple[str, int]] = []
            found = False
            candidates = []
            elapsed = 0.0

            while True:
                filtered_entries = {k: v for k, v in entries.items() if k not in known_buggy_keys}
                interface = PredictionInterface.model_validate(filtered_entries)
                asp_constraints, atom_counts = compile_asp_constraint(interface)

                t0 = time.monotonic()
                candidates, _ = solve(asp_constraints, atom_counts, output_list_of_smiles=True, timeout=args.timeout)
                elapsed = time.monotonic() - t0

                cand_canons = {canonical(s) for s in candidates} - {None}
                found = gt_canon in cand_canons

                tag = "FOUND" if found else "MISS "
                print(f"[{dataset_idx:4d}] {tag} | {len(candidates):4d} candidates | {elapsed:5.1f}s | {gt_smi_raw[:60]}")

                if found:
                    n_found += 1
                    break
                prev_count = len(known_buggy_keys)
                culprits += diagnose_miss(entries, gt_canon, timeout=diagnose_timeout, known_buggy_keys=known_buggy_keys)
                if len(known_buggy_keys) == prev_count:
                    print(f"    No new buggy constraint found — stopping diagnosis for this molecule.")
                    break

            results.append({
                "idx": dataset_idx,
                "smiles": gt_smi_raw,
                "found": found,
                "n_candidates": len(candidates),
                "elapsed_s": round(elapsed, 2),
                "culprits": [{"constraint": label, "value": val} for label, val in culprits],
            })

    else:
        print(f"Solving {len(work_items)} records with {args.workers} worker(s)...")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_solve_record, dataset_idx, gt_smi_raw, gt_canon, asp_constraints, atom_counts, args.timeout): dataset_idx
                for dataset_idx, gt_smi_raw, gt_canon, asp_constraints, atom_counts, _ in work_items
            }
            for future in as_completed(futures):
                result = future.result()
                tag = "FOUND" if result["found"] else "MISS "
                print(f"[{result['idx']:4d}] {tag} | {result['n_candidates']:4d} candidates | {result['elapsed_s']:5.1f}s | {result['smiles'][:60]}")
                if result["found"]:
                    n_found += 1
                results.append(result)

        results.sort(key=lambda r: r["idx"])

    n_eval = len(results)
    recall = 100 * n_found / n_eval if n_eval else 0.0
    print(f"\n=== Summary ===")
    print(f"Evaluated : {n_eval}")
    print(f"Found     : {n_found} / {n_eval}  ({recall:.1f}% recall)")
    if n_skip:
        print(f"Skipped   : {n_skip}  (unsupported, unparseable, or too large)")

    if args.diagnose:
        culprit_counts: dict[str, int] = {}
        for r in results:
            for c in r["culprits"]:
                culprit_counts[c["constraint"]] = culprit_counts.get(c["constraint"], 0) + 1
        if culprit_counts:
            print(f"\n=== Buggy constraints (confirmed across {n_eval} records) ===")
            for label in sorted(culprit_counts):
                print(f"  {label}")
        else:
            print("\nNo buggy constraints found.")

    if args.output:
        out = {
            "summary": {
                "n_eval": n_eval,
                "n_found": n_found,
                "recall_pct": round(recall, 2),
                "n_skipped": n_skip,
            },
            "records": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")

    if args.report:
        import datetime
        lines = [
            "# Oracle Evaluation Report",
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Dataset:** {data_path.name}",
            f"**Molecule size limit:** {args.mol_size_limit or 'none'}",
            f"**Solve timeout:** {args.timeout}s",
            "",
            "## Summary",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Records evaluated | {n_eval} |",
            f"| Found | {n_found} |",
            f"| Missed | {n_eval - n_found} |",
            f"| Recall | {recall:.1f}% |",
            f"| Skipped | {n_skip} |",
            "",
        ]

        if args.diagnose and culprit_counts:
            # Build constraint → list of SMILES mapping
            constraint_to_smiles: dict[str, list[str]] = {}
            for r in results:
                for c in r["culprits"]:
                    constraint_to_smiles.setdefault(c["constraint"], []).append(r["smiles"])

            lines += [
                "## Buggy Constraints",
                "Constraints confirmed as buggy across all missed records.",
                "All listed constraints are definitively wrong.",
                "",
                "| Constraint | Molecules |",
                "|------------|-----------|",
            ]
            for label in sorted(constraint_to_smiles):
                smiles_list = "; ".join(
                    f"`{s[:50]}`" for s in constraint_to_smiles[label]
                )
                lines.append(f"| `{label}` | {smiles_list} |")
            lines.append("")

        misses = [r for r in results if not r["found"]]
        if misses:
            lines += [
                "## Missed Records",
                "| # | SMILES | Candidates | Time (s) | Culprit |",
                "|---|--------|------------|----------|---------|",
            ]
            for r in misses:
                culprit_str = ", ".join(
                    f"`{c['constraint']}={c['value']}`" for c in r["culprits"]
                ) or "—"
                lines.append(
                    f"| {r['idx']} | `{r['smiles'][:60]}` "
                    f"| {r['n_candidates']} | {r['elapsed_s']} | {culprit_str} |"
                )
            lines.append("")

        Path(args.report).write_text("\n".join(lines), encoding="utf-8")
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
