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
    python oracle_eval.py --output results.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config
from data.datasets import dataset_from_schema, RawFeatureDataset
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS
from predicates.interface import PredictionInterface, PredictionEntry
from predicates.compiler import compile_asp_constraint
from asp_bridge import solve

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
    p.add_argument("--show-constraints", action="store_true",
                   help="Print compiled ASP constraints for each record")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Save per-record results to a JSON file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

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

    results = []
    n_found = n_skip = 0

    for idx in range(limit):
        record = dataset._load_record(idx)
        gt_smi_raw = record.get("smiles")
        if not gt_smi_raw:
            n_skip += 1
            continue
        gt_canon = canonical(str(gt_smi_raw))
        if gt_canon is None:
            n_skip += 1
            continue

        entries = gt_entries_from_record(record, dataset)
        interface = PredictionInterface.model_validate(entries)
        asp_constraints, atom_counts = compile_asp_constraint(interface)

        if args.show_constraints:
            formula = (f"C{atom_counts.get('num_carbons',0)}"
                       f"H{atom_counts.get('num_hydrogens',0)}"
                       f"N{atom_counts.get('num_nitrogens',0)}"
                       f"O{atom_counts.get('num_oxygens',0)}")
            print(f"\n[{idx}] {gt_smi_raw[:70]}  formula={formula}")
            for c in asp_constraints:
                print(f"  {c}")

        t0 = time.monotonic()
        candidates, model_count = solve(
            asp_constraints, atom_counts,
            output_list_of_smiles=True,
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - t0

        cand_canons = {canonical(s) for s in candidates} - {None}
        found = gt_canon in cand_canons
        if found:
            n_found += 1

        tag = "FOUND" if found else "MISS "
        print(f"[{idx:4d}] {tag} | {len(candidates):4d} candidates | {elapsed:5.1f}s | {gt_smi_raw[:60]}")

        results.append({
            "idx": idx,
            "smiles": gt_smi_raw,
            "found": found,
            "n_candidates": len(candidates),
            "elapsed_s": round(elapsed, 2),
        })

    n_eval = len(results)
    recall = 100 * n_found / n_eval if n_eval else 0.0
    print(f"\n=== Summary ===")
    print(f"Evaluated : {n_eval}")
    print(f"Found     : {n_found} / {n_eval}  ({recall:.1f}% recall)")
    if n_skip:
        print(f"Skipped   : {n_skip}  (missing or unparseable SMILES)")

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


if __name__ == "__main__":
    main()
