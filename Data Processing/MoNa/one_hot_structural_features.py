"""
Convert structural features into categorical one-hot features.

This is useful when downstream prediction requires categorical outcomes (for
example conformal prediction over classes instead of raw counts).

Input records are expected to include:
    {
        "smiles": "...",
        "maccs": [...],
        "structural_features": {
            "ring_count": 2,
            "rotatable_bond_count": 5,
            ...
        }
    }

Output fields added per record:
    - structural_vector_onehot_packed_b64 (default): packed bits, base64 encoded
    - feature_vector_onehot_packed_b64 (default): packed bits, base64 encoded
    - *_nbits fields for exact unpacking length
    - combined_packed_b64 / combined_nbits aliases for compact pipelines

Optional (disabled by default to keep files small):
    - structural_features_onehot dictionary
    - dense structural_vector_onehot and feature_vector_onehot

Compact-only mode:
        - keeps only `spectrum`, `combined_packed_b64`, and `combined_nbits`
            (plus `smiles` by default, configurable)

A companion schema file is also written to map one-hot indices to names.

Usage:
    python one_hot_structural_features.py
    python one_hot_structural_features.py --input "C:/path/to/in.json" --output "C:/path/to/out.json" --schema-output "C:/path/to/schema.json"
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def is_int_like(v: object) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return v.is_integer()
    return False


def quantile_edges(values: list[float], n_bins: int) -> list[float]:
    """Return sorted quantile cut points (excluding min and max)."""
    if n_bins <= 1 or len(values) < 2:
        return []

    sorted_vals = sorted(values)
    edges: list[float] = []
    n = len(sorted_vals)
    for k in range(1, n_bins):
        pos = int(round(k * (n - 1) / n_bins))
        edge = sorted_vals[pos]
        edges.append(edge)

    # Deduplicate to avoid empty bins when many repeated values exist.
    deduped = []
    for e in edges:
        if not deduped or e > deduped[-1]:
            deduped.append(e)
    return deduped


def make_bin_labels(edges: list[float]) -> list[str]:
    labels: list[str] = []
    if not edges:
        labels.append("all")
        return labels

    labels.append(f"<= {edges[0]:.6g}")
    for i in range(1, len(edges)):
        labels.append(f"> {edges[i - 1]:.6g} and <= {edges[i]:.6g}")
    labels.append(f"> {edges[-1]:.6g}")
    return labels


def assign_bin_index(value: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def sanitize_value_for_name(value: object) -> str:
    text = str(value)
    text = text.replace(" ", "_")
    text = text.replace("=", "eq")
    text = text.replace("<", "lt")
    text = text.replace(">", "gt")
    text = text.replace(".", "p")
    return text


def build_feature_schema(
    records: list[dict],
    source_field: str,
    discrete_max_values: int,
    continuous_bins: int,
) -> tuple[list[str], dict[str, dict]]:
    """Build one-hot feature names and encoding metadata for each source feature."""
    per_feature_values: dict[str, list[object]] = {}

    for record in records:
        feats = record.get(source_field)
        if not isinstance(feats, dict):
            continue
        for name, value in feats.items():
            per_feature_values.setdefault(name, []).append(value)

    if not per_feature_values:
        return [], {}

    onehot_names: list[str] = []
    schema: dict[str, dict] = {}

    for feat_name in sorted(per_feature_values.keys()):
        values = per_feature_values[feat_name]
        unique_vals = sorted(set(values), key=lambda x: (str(type(x)), x))

        all_int_like = all(is_int_like(v) for v in unique_vals)
        use_exact = all_int_like and len(unique_vals) <= discrete_max_values

        if use_exact:
            categories = [int(v) if is_int_like(v) else v for v in unique_vals]
            cat_names = []
            for cat in categories:
                name = f"{feat_name}={sanitize_value_for_name(cat)}"
                cat_names.append(name)
                onehot_names.append(name)

            schema[feat_name] = {
                "encoding": "exact",
                "categories": categories,
                "onehot_names": cat_names,
            }
            continue

        # Continuous/high-cardinality path: bin to keep classes manageable.
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (TypeError, ValueError):
                pass

        if not numeric_values:
            # Fallback: exact string categories for non-numeric values.
            categories = sorted(set(str(v) for v in values))
            cat_names = []
            for cat in categories:
                name = f"{feat_name}={sanitize_value_for_name(cat)}"
                cat_names.append(name)
                onehot_names.append(name)

            schema[feat_name] = {
                "encoding": "exact-string",
                "categories": categories,
                "onehot_names": cat_names,
            }
            continue

        bins = max(2, continuous_bins)
        edges = quantile_edges(numeric_values, bins)
        labels = make_bin_labels(edges)
        cat_names = []
        for i, label in enumerate(labels):
            name = f"{feat_name}_bin_{i}:{label}"
            cat_names.append(name)
            onehot_names.append(name)

        schema[feat_name] = {
            "encoding": "binned",
            "edges": edges,
            "labels": labels,
            "onehot_names": cat_names,
        }

    return onehot_names, schema


def iter_json_array(path: Path):
    """Yield objects from a large JSON file containing a top-level array.

    This uses incremental buffering and json.JSONDecoder.raw_decode so the
    whole file is never loaded into memory.
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        # Skip whitespace and initial '['
        chunk = f.read(1)
        while chunk and chunk.isspace():
            chunk = f.read(1)
        if not chunk:
            return
        if chunk != "[":
            # Not an array: try to decode single object
            f.seek(0)
            buf = f.read()
            obj, _ = decoder.raw_decode(buf)
            yield obj
            return

        buffer = ""
        EOF = False
        while True:
            chunk = f.read(65536)
            if not chunk:
                EOF = True
            buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                # Stop if array end
                if buffer[0] == "]":
                    return
                # Skip commas between elements
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except ValueError:
                    # Need more data
                    break
                yield obj
                buffer = buffer[idx:]
            if EOF:
                break


def build_feature_schema_stream(
    input_path: Path,
    source_field: str,
    discrete_max_values: int,
    continuous_bins: int,
    sample_per_feature: int = 10000,
) -> tuple[list[str], dict[str, dict], int]:
    """Build schema by streaming through the input JSON array.

    To avoid unbounded memory growth, this function:
    - collects up to `sample_per_feature` numeric samples per feature for binning
    - collects unique non-numeric categories up to `discrete_max_values + 1`
    Returns (onehot_names, schema, total_records_seen).
    """
    per_feature_numeric: dict[str, list[float]] = {}
    per_feature_uniques: dict[str, set] = {}
    per_feature_has_numeric: dict[str, bool] = {}
    total = 0

    for record in iter_json_array(input_path):
        total += 1
        feats = record.get(source_field)
        if not isinstance(feats, dict):
            continue
        for name, value in feats.items():
            # Track whether we have numeric values for this feature
            if name not in per_feature_has_numeric:
                per_feature_has_numeric[name] = True

            # Try numeric
            try:
                v = float(value)
            except Exception:
                # Non-numeric: track unique string categories (bounded)
                per_feature_has_numeric[name] = per_feature_has_numeric.get(name, False) or False
                s = per_feature_uniques.setdefault(name, set())
                if len(s) <= discrete_max_values + 1:
                    s.add(str(value))
                continue

            # Numeric path: add to numeric sample up to limit
            lst = per_feature_numeric.setdefault(name, [])
            if len(lst) < sample_per_feature:
                lst.append(v)

    # Build schema using sampled/collected values
    # Reuse existing build logic by synthesizing small records list where needed.
    synthesized_records: list[dict] = []
    for feat_name in set(list(per_feature_numeric.keys()) + list(per_feature_uniques.keys())):
        rec = {feat_name: None}
        synthesized_records.append({source_field: rec})

    onehot_names: list[str] = []
    schema: dict[str, dict] = {}

    # For each feature, decide encoding like original function but using samples
    for feat_name in sorted(set(list(per_feature_numeric.keys()) + list(per_feature_uniques.keys()))):
        numeric_samples = per_feature_numeric.get(feat_name, [])
        string_uniques = per_feature_uniques.get(feat_name, set())

        if numeric_samples:
            # numeric: decide bins
            bins = max(2, continuous_bins)
            edges = quantile_edges(numeric_samples, bins)
            labels = make_bin_labels(edges)
            cat_names = []
            for i, label in enumerate(labels):
                name = f"{feat_name}_bin_{i}:{label}"
                cat_names.append(name)
                onehot_names.append(name)
            schema[feat_name] = {
                "encoding": "binned",
                "edges": edges,
                "labels": labels,
                "onehot_names": cat_names,
            }
        else:
            # treat as string categories limited by collected uniques
            categories = sorted(string_uniques)
            cat_names = []
            for cat in categories:
                name = f"{feat_name}={sanitize_value_for_name(cat)}"
                cat_names.append(name)
                onehot_names.append(name)
            schema[feat_name] = {
                "encoding": "exact-string",
                "categories": categories,
                "onehot_names": cat_names,
            }

    return onehot_names, schema, total


def encode_one_record(
    feats: dict,
    onehot_names: list[str],
    schema: dict[str, dict],
) -> tuple[dict[str, int], list[int]]:
    onehot = {name: 0 for name in onehot_names}

    for feat_name, spec in schema.items():
        if feat_name not in feats:
            continue

        value = feats[feat_name]
        encoding = spec["encoding"]

        if encoding == "exact":
            try:
                v = int(value) if is_int_like(value) else value
            except (TypeError, ValueError):
                continue

            categories = spec["categories"]
            if v in categories:
                idx = categories.index(v)
                onehot_name = spec["onehot_names"][idx]
                onehot[onehot_name] = 1

        elif encoding == "exact-string":
            v = str(value)
            categories = spec["categories"]
            if v in categories:
                idx = categories.index(v)
                onehot_name = spec["onehot_names"][idx]
                onehot[onehot_name] = 1

        elif encoding == "binned":
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            edges = spec["edges"]
            idx = assign_bin_index(numeric_value, edges)
            onehot_name = spec["onehot_names"][idx]
            onehot[onehot_name] = 1

    vector = [onehot[name] for name in onehot_names]
    return onehot, vector


def to_int_list(values: object) -> list[int] | None:
    if not isinstance(values, list):
        return None
    try:
        return [int(x) for x in values]
    except (TypeError, ValueError):
        return None


def pack_bits(values: list[int]) -> bytes:
    """Pack a list of 0/1 ints into bytes (MSB first per byte)."""
    n_bits = len(values)
    out = bytearray((n_bits + 7) // 8)
    for i, bit in enumerate(values):
        if bit:
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


def run(
    input_path: Path,
    output_path: Path,
    schema_output: Path,
    source_field: str,
    maccs_field: str,
    discrete_max_values: int,
    continuous_bins: int,
    store_packed_targets: bool,
    keep_dense_vectors: bool,
    keep_onehot_dict: bool,
    compact_only: bool,
    keep_smiles_in_compact: bool,
) -> None:
    print(f"Scanning {input_path} to build schema (streaming)...")
    onehot_names, schema, total_records = build_feature_schema_stream(
        input_path=input_path,
        source_field=source_field,
        discrete_max_values=discrete_max_values,
        continuous_bins=continuous_bins,
    )

    if not onehot_names:
        sys.exit(f"No features found under '{source_field}'.")

    print(f"Built one-hot schema with {len(onehot_names):,} categorical columns.")

    kept = 0
    dropped_no_source = 0
    dropped_no_maccs = 0

    print(f"Processing and writing output to {output_path} (streaming)...")
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        first = True

        for i, record in enumerate(iter_json_array(input_path), start=1):
            feats = record.get(source_field)
            if not isinstance(feats, dict):
                dropped_no_source += 1
                continue

            maccs = to_int_list(record.get(maccs_field))
            if maccs is None:
                dropped_no_maccs += 1
                continue

            onehot_dict, onehot_vec = encode_one_record(feats, onehot_names, schema)
            feature_vec = maccs + onehot_vec

            if compact_only:
                out = {}
                if "spectrum" in record:
                    out["spectrum"] = record["spectrum"]
                if keep_smiles_in_compact and "smiles" in record:
                    out["smiles"] = record["smiles"]
            else:
                out = dict(record)

            if not compact_only and keep_onehot_dict:
                out["structural_features_onehot"] = onehot_dict
            if not compact_only and keep_dense_vectors:
                out["structural_vector_onehot"] = onehot_vec
                out["feature_vector_onehot"] = feature_vec

            if store_packed_targets:
                structural_packed = pack_bits(onehot_vec)
                feature_packed = pack_bits(feature_vec)
                out["structural_vector_onehot_nbits"] = len(onehot_vec)
                out["feature_vector_onehot_nbits"] = len(feature_vec)
                out["structural_vector_onehot_packed_b64"] = base64.b64encode(structural_packed).decode("ascii")
                out["feature_vector_onehot_packed_b64"] = base64.b64encode(feature_packed).decode("ascii")
                # Compact pipeline aliases.
                out["combined_nbits"] = len(feature_vec)
                out["combined_packed_b64"] = out["feature_vector_onehot_packed_b64"]

            if not first:
                f_out.write(",\n")
            json.dump(out, f_out)
            first = False
            kept += 1

            if i % 10000 == 0:
                print(f"  {i:,} processed, kept {kept:,} ...", flush=True)

        f_out.write("\n]")

    schema_payload = {
        "source_field": source_field,
        "maccs_field": maccs_field,
        "maccs_length": 167,
        "onehot_length": len(onehot_names),
        "feature_vector_onehot_length": 167 + len(onehot_names),
        "packed_bit_order": "msb_first",
        "compact_combined_field_b64": "combined_packed_b64",
        "compact_combined_field_nbits": "combined_nbits",
        "onehot_names": onehot_names,
        "feature_schema": schema,
        "feature_vector_index_map": [
            *[f"maccs_{i}" for i in range(167)],
            *onehot_names,
        ],
    }
    with open(schema_output, "w", encoding="utf-8") as f_schema:
        json.dump(schema_payload, f_schema, indent=2)

    print("\nFinished.")
    print(f"  Kept records:             {kept:,} / {total_records:,}")
    print(f"  Dropped (no source):      {dropped_no_source:,}")
    print(f"  Dropped (no valid maccs): {dropped_no_maccs:,}")
    print(f"  One-hot structural dims:  {len(onehot_names):,}")
    print(f"  Total feature dims:       {167 + len(onehot_names):,}")
    print(f"  Packed targets enabled:   {store_packed_targets}")
    print(f"  Dense vectors kept:       {keep_dense_vectors}")
    print(f"  One-hot dict kept:        {keep_onehot_dict}")
    print(f"  Compact-only mode:        {compact_only}")
    print(f"  Keep smiles in compact:   {keep_smiles_in_compact}")
    print(f"  Output data:              {output_path}")
    print(f"  Output schema:            {schema_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert structural feature counts/values to one-hot categorical features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-curated-with-maccs-and-structure.json"
        ),
        help="Input JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-curated-with-maccs-and-structure-onehot.json"
        ),
        help="Output JSON path with one-hot features added.",
    )
    parser.add_argument(
        "--schema-output",
        type=Path,
        default=Path(
            "C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\feature_vector_onehot_schema.json"
        ),
        help="Output JSON schema/index map path.",
    )
    parser.add_argument(
        "--source-field",
        type=str,
        default="structural_features",
        help="Field containing source structural features.",
    )
    parser.add_argument(
        "--maccs-field",
        type=str,
        default="maccs",
        help="Field containing MACCS vector.",
    )
    parser.add_argument(
        "--discrete-max-values",
        type=int,
        default=32,
        help="Exact one-hot is used for int-like features with <= this many unique values.",
    )
    parser.add_argument(
        "--continuous-bins",
        type=int,
        default=6,
        help="Quantile bins used for non-discrete/high-cardinality numeric features.",
    )
    parser.add_argument(
        "--store-packed-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store packed bit vectors (base64) for one-hot targets.",
    )
    parser.add_argument(
        "--keep-dense-vectors",
        action="store_true",
        help="Keep dense one-hot vectors in JSON (larger files).",
    )
    parser.add_argument(
        "--keep-onehot-dict",
        action="store_true",
        help="Keep structural_features_onehot dictionary in JSON (largest output).",
    )
    parser.add_argument(
        "--compact-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only compact fields (spectrum + combined packed target).",
    )
    parser.add_argument(
        "--keep-smiles-in-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In compact-only mode, keep the smiles field (default: true).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    if args.discrete_max_values < 1:
        sys.exit("--discrete-max-values must be >= 1")
    if args.continuous_bins < 2:
        sys.exit("--continuous-bins must be >= 2")

    run(
        input_path=args.input,
        output_path=args.output,
        schema_output=args.schema_output,
        source_field=args.source_field,
        maccs_field=args.maccs_field,
        discrete_max_values=args.discrete_max_values,
        continuous_bins=args.continuous_bins,
        store_packed_targets=args.store_packed_targets,
        keep_dense_vectors=args.keep_dense_vectors,
        keep_onehot_dict=args.keep_onehot_dict,
        compact_only=args.compact_only,
        keep_smiles_in_compact=args.keep_smiles_in_compact,
    )
