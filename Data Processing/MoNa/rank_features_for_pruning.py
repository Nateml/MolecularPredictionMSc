"""
Rank structural features by prune priority (worst features first).

The top-ranked features are likely candidates to remove because they are:
  - near-constant (always on/off or close to it),
  - overly specific (many singleton values),
  - low-information (low entropy),
  - weak at splitting molecular ambiguity.

Input records are expected to contain at least:
  - smiles
  - maccs
  - structural_features (dictionary)

Usage:
  python rank_features_for_pruning.py
  python rank_features_for_pruning.py --input "C:/path/to/MoNA-filtered-curated-with-maccs-and-structure.json" --top 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required: pip install rdkit")


def to_int_tuple(values: object) -> tuple[int, ...] | None:
    if not isinstance(values, list):
        return None
    try:
        return tuple(int(x) for x in values)
    except (TypeError, ValueError):
        return None


def element_composition_signature(mol: Chem.Mol) -> tuple[tuple[str, int], ...]:
    expanded = Chem.AddHs(mol)
    counts: Counter[str] = Counter()
    for atom in expanded.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return tuple(sorted(counts.items()))


def normalized_entropy(counter: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    distinct = len(counter)
    if distinct <= 1:
        return 0.0

    entropy = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)

    return entropy / math.log2(distinct)


def split_reduction_pct(
    feature_name: str,
    ambiguous_groups: list[set[str]],
    molecule_to_features: dict[str, dict[str, int | float]],
) -> float:
    """Expected candidate reduction for a single feature within ambiguous groups."""
    before = 0.0
    after = 0.0

    for mol_set in ambiguous_groups:
        partitions: dict[object, int] = defaultdict(int)

        # Skip groups with missing values for this feature.
        complete = True
        for canon in mol_set:
            feats = molecule_to_features.get(canon)
            if feats is None or feature_name not in feats:
                complete = False
                break
            partitions[feats[feature_name]] += 1
        if not complete:
            continue

        n = len(mol_set)
        expected_after = sum(size * size for size in partitions.values()) / n
        before += n * n
        after += n * expected_after

    if before == 0:
        return 0.0
    return 100.0 * (1.0 - after / before)


def is_binary_feature(values: list[object]) -> bool:
    s = set(values)
    return s.issubset({0, 1}) or s.issubset({0.0, 1.0})


def analyse(input_path: Path, top_n: int, output_json: Path | None) -> None:
    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print("Input JSON is empty or not a list of records.")
        return

    baseline_to_molecules: dict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], set[str]] = defaultdict(set)
    molecule_to_features: dict[str, dict[str, int | float]] = {}

    invalid_smiles = 0
    missing_maccs = 0
    invalid_maccs = 0
    missing_structural = 0

    for record in data:
        smiles = str(record.get("smiles", ""))
        maccs = record.get("maccs")
        structural = record.get("structural_features")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue
        canon = Chem.MolToSmiles(mol)

        if not isinstance(structural, dict):
            missing_structural += 1
            continue

        if canon not in molecule_to_features:
            molecule_to_features[canon] = structural

        # Baseline groups are optional for score stability; if maccs is unusable,
        # we still rank by distributional quality.
        if maccs is None:
            missing_maccs += 1
            continue
        maccs_key = to_int_tuple(maccs)
        if maccs_key is None:
            invalid_maccs += 1
            continue

        composition = element_composition_signature(mol)
        baseline_to_molecules[(maccs_key, composition)].add(canon)

    if not molecule_to_features:
        print("No usable structural_features found.")
        return

    ambiguous_groups = [
        mol_set for mol_set in baseline_to_molecules.values() if len(mol_set) > 1
    ]

    feature_names = sorted({k for d in molecule_to_features.values() for k in d.keys()})
    total_molecules = len(molecule_to_features)

    rows = []

    for feature_name in feature_names:
        values = []
        for feats in molecule_to_features.values():
            if feature_name in feats:
                values.append(feats[feature_name])

        n = len(values)
        if n == 0:
            continue

        counter = Counter(values)
        mode_value, mode_count = counter.most_common(1)[0]
        mode_fraction = mode_count / n
        miss_rate = 1.0 - (n / total_molecules)
        distinct = len(counter)
        distinct_ratio = distinct / n
        norm_ent = normalized_entropy(counter, n)
        low_entropy_component = 1.0 - norm_ent

        singleton_sample_fraction = sum(c for c in counter.values() if c == 1) / n

        binary = is_binary_feature(values)
        on_rate = None
        if binary:
            on_count = counter.get(1, 0) + counter.get(1.0, 0)
            on_rate = on_count / n

        split_red = split_reduction_pct(feature_name, ambiguous_groups, molecule_to_features)
        non_split_component = 1.0 - (split_red / 100.0)

        # Prune priority score (higher = stronger candidate to remove).
        # Emphasize near-constant and low-information behavior, while including
        # a penalty for weak ambiguity splitting.
        score = (
            0.35 * mode_fraction
            + 0.20 * low_entropy_component
            + 0.20 * singleton_sample_fraction
            + 0.20 * non_split_component
            + 0.05 * miss_rate
        )

        # Extra penalty for binary features that are mostly always-off/on.
        if binary and on_rate is not None and (on_rate <= 0.05 or on_rate >= 0.95):
            score += 0.10

        if score > 1.0:
            score = 1.0

        reasons = []
        if mode_fraction >= 0.98:
            reasons.append("near-constant")
        if binary and on_rate is not None and (on_rate <= 0.02 or on_rate >= 0.98):
            reasons.append("always-on/off-like")
        if singleton_sample_fraction >= 0.40:
            reasons.append("over-specific")
        if split_red <= 5.0:
            reasons.append("weak-split")
        if miss_rate >= 0.10:
            reasons.append("missing-often")

        rows.append(
            {
                "feature": feature_name,
                "prune_score": score,
                "mode_value": mode_value,
                "mode_fraction_pct": 100.0 * mode_fraction,
                "normalized_entropy": norm_ent,
                "singleton_sample_fraction_pct": 100.0 * singleton_sample_fraction,
                "distinct_ratio_pct": 100.0 * distinct_ratio,
                "split_reduction_pct": split_red,
                "missing_rate_pct": 100.0 * miss_rate,
                "is_binary": binary,
                "on_rate_pct": (100.0 * on_rate) if on_rate is not None else None,
                "n_molecules_with_value": n,
                "flags": reasons,
            }
        )

    rows.sort(key=lambda r: r["prune_score"], reverse=True)

    print("=" * 110)
    print("FEATURE PRUNE RANKING (TOP = REMOVE FIRST)")
    print("=" * 110)
    print("\n--- Data summary ---")
    print(f"  Total records:                              {len(data):,}")
    print(f"  Unique canonical molecules used:            {total_molecules:,}")
    print(f"  Invalid smiles:                             {invalid_smiles:,}")
    print(f"  Missing structural_features:                {missing_structural:,}")
    print(f"  Baseline ambiguous groups (if maccs usable):{len(ambiguous_groups):,}")
    print(f"  Missing maccs:                              {missing_maccs:,}")
    print(f"  Invalid maccs:                              {invalid_maccs:,}")

    print(f"\n--- Top {top_n} prune candidates ---")
    print(
        "  "
        + f"{'feature':<34}"
        + f" {'score':>7}"
        + f" {'mode%':>7}"
        + f" {'ent':>7}"
        + f" {'single%':>9}"
        + f" {'split%':>8}"
        + f" {'on%':>7}"
        + f" {'flags':<28}"
    )
    print("  " + "-" * 106)

    for row in rows[:top_n]:
        on_pct_str = "-"
        if row["on_rate_pct"] is not None:
            on_pct_str = f"{row['on_rate_pct']:.1f}"

        print(
            "  "
            + f"{row['feature']:<34}"
            + f" {row['prune_score']:>7.3f}"
            + f" {row['mode_fraction_pct']:>7.1f}"
            + f" {row['normalized_entropy']:>7.3f}"
            + f" {row['singleton_sample_fraction_pct']:>9.1f}"
            + f" {row['split_reduction_pct']:>8.2f}"
            + f" {on_pct_str:>7}"
            + f" {','.join(row['flags']):<28}"
        )

    if output_json is not None:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nFull ranking saved to {output_json}")

    print("\nInterpretation:")
    print("  - Top rows are strongest prune candidates.")
    print("  - mode% high + entropy low + split% low usually means little practical value.")
    print("  - For binary features, on% close to 0 or 100 indicates always-off/on behavior.")
    print("=" * 110)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank structural features by prune priority (near-constant, over-specific, low-split)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "C:/Users/Tom/Documents/Planning/MolecularPredictionMSc/Data Processing/MoNa/MoNA-filtered-curated-with-maccs-and-structure_max-600Da.json"
        ),
        help="Path to JSON file with smiles, maccs, structural_features.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="How many top prune candidates to print.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full ranking as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    if args.top < 1:
        sys.exit("--top must be >= 1")
    analyse(args.input, args.top, args.output_json)
