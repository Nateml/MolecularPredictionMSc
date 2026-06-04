"""
Convert structural features to continuous format.

This script takes records with structural features and creates a schema where
all features are stored as continuous (non-discretized) values.

Input records are expected to include:
    {
        "smiles": "...",
        "maccs": [...],
        "structural_features": {
            "ring_count": 2,
            "rotatable_bond_count": 5,
            ...
        },
        "structural_vector": [...]
    }

Output fields added per record:
    - continuous_vector: [maccs_bits] + [structural_features_in_order]

A companion schema file is also written to document feature names and types.

Usage:
    python continuous_structural_features.py
    python continuous_structural_features.py --input "C:/path/to/in.json" --output "C:/path/to/out.json" --schema-output "C:/path/to/schema.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def to_fp_tuple(maccs: object) -> tuple[int, ...] | None:
    """Convert MACCS field to tuple of ints."""
    if not isinstance(maccs, list):
        return None
    try:
        return tuple(int(x) for x in maccs)
    except (TypeError, ValueError):
        return None


def extract_maccs(record: dict, maccs_field: str) -> list[int] | None:
    """Read the MACCS vector from a record."""
    candidates = [maccs_field, "maccs", "MACCS", "fingerprint", "fp"]
    for field in candidates:
        if field in record:
            fp = to_fp_tuple(record.get(field))
            if fp is not None:
                return list(fp)
    return None


def get_structural_vector_order_from_schema(
    records: list[dict], source_field: str = "structural_features"
) -> list[str]:
    """Determine the canonical feature order from structural_features."""
    feature_keys: set[str] = set()
    for record in records:
        feats = record.get(source_field)
        if isinstance(feats, dict):
            feature_keys.update(feats.keys())
    return sorted(feature_keys)


def build_continuous_schema(
    records: list[dict],
    maccs_length: int,
    source_field: str = "structural_features",
) -> dict:
    """Build a schema for continuous features."""
    feature_order = get_structural_vector_order_from_schema(records, source_field)

    feature_schema: dict[str, dict] = {}
    for feat_name in feature_order:
        feature_schema[feat_name] = {
            "type": "continuous",
        }

    schema = {
        "source_field": source_field,
        "maccs_field": "maccs",
        "maccs_length": maccs_length,
        "continuous_feature_order": feature_order,
        "n_continuous": len(feature_order),
        "feature_vector_continuous_length": maccs_length + len(feature_order),
        "feature_schema": feature_schema,
    }

    return schema


def build_continuous_records(
    input_path: Path,
    output_path: Path,
    schema_output_path: Path,
    maccs_field: str,
    source_field: str = "structural_features",
) -> None:
    """Process records and build continuous feature vectors."""
    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit("Input JSON must be a list of records.")

    total = len(data)
    kept = 0
    dropped_no_maccs = 0
    dropped_no_features = 0

    # First pass: determine feature order
    print("Scanning records to determine feature order...")
    feature_order = get_structural_vector_order_from_schema(data, source_field)

    maccs_length = 0
    for record in data:
        maccs = extract_maccs(record, maccs_field)
        if maccs is not None:
            maccs_length = len(maccs)
            break

    if maccs_length == 0:
        sys.exit("Could not determine MACCS length from records.")

    print(f"Feature order: {len(feature_order)} continuous features")
    print(f"MACCS length: {maccs_length}")

    # Build schema
    schema = build_continuous_schema(data, maccs_length, source_field)

    print(f"Processing {total:,} records ...")

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        first = True

        for idx, record in enumerate(data, start=1):
            maccs = extract_maccs(record, maccs_field)
            if maccs is None:
                dropped_no_maccs += 1
                continue

            feats = record.get(source_field)
            if not isinstance(feats, dict):
                dropped_no_features += 1
                continue

            # Build continuous vector: [maccs] + [structural_features_in_order]
            continuous_vector = list(maccs)
            for feat_name in feature_order:
                value = float(feats.get(feat_name, 0.0))
                continuous_vector.append(value)

            out = dict(record)
            out["continuous_vector"] = continuous_vector

            if not first:
                f_out.write(",\n")
            json.dump(out, f_out)
            first = False
            kept += 1

            #also add continuous_feature_names to schema for later reference

            if idx % 10_000 == 0:
                print(f"  {idx:,} / {total:,} processed, kept {kept:,} ...", flush=True)

            

        f_out.write("\n]")

    # Write schema
    with open(schema_output_path, "w", encoding="utf-8") as f_schema:
        json.dump(schema, f_schema, indent=2)

    print("\nFinished.")
    print(f"  Kept:                         {kept:,} / {total:,}")
    print(f"  Dropped - no MACCS:           {dropped_no_maccs:,}")
    print(f"  Dropped - no features:        {dropped_no_features:,}")
    print(f"Output written to {output_path}")
    print(f"Schema written to {schema_output_path}")
    print(f"Continuous feature vector length: {maccs_length} (MACCS) + {len(feature_order)} (features) = {maccs_length + len(feature_order)}")


if __name__ == "__main__":
    inputpath = Path(r"C:\Users\Tom\Desktop\Nieuwe map\MolecularPredictionMSc\output\EI_Dataset_MACCS_structural.json")
    outputpath = Path(r"C:\Users\Tom\Desktop\Nieuwe map\MolecularPredictionMSc\output\EI_Dataset_MACCS_Structural_Continuous.json")
    schema_output_path = Path(r"C:\Users\Tom\Desktop\Nieuwe map\MolecularPredictionMSc\output\EI_Dataset_MACCS_Structural_Schema.json")
    if not inputpath.is_file():
        sys.exit(f"Input file not found: {inputpath}")
    build_continuous_records(inputpath, outputpath, schema_output_path, maccs_field="maccs")