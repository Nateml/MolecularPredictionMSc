"""
Test which structural traits best reduce MACCS+composition collisions.

Analyzes the collision-breaking power of various structural features.

Usage:
    pip install rdkit
    python structural_traits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required: pip install rdkit")


def to_fp_tuple(maccs: object) -> tuple[int, ...] | None:
    if not isinstance(maccs, list):
        return None
    try:
        return tuple(int(x) for x in maccs)
    except (TypeError, ValueError):
        return None


def element_composition_signature(mol: Chem.Mol) -> tuple[tuple[str, int], ...]:
    expanded = Chem.AddHs(mol)
    counts: Counter[str] = Counter()
    for atom in expanded.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return tuple(sorted(counts.items()))


def compute_structural_features(mol: Chem.Mol) -> dict[str, int | float]:
    """Compute various structural descriptors."""
    features = {}
    
    # Ring information
    ri = mol.GetRingInfo()
    features["aromatic_ring_count"] = len(ri.AtomRings())
    features["num_aromatic_atoms"] = len([a for a in mol.GetAtoms() if a.GetIsAromatic()])
    
    # Bond unsaturation
    double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
    triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.TRIPLE)
    features["double_bond_count"] = double_bonds
    features["triple_bond_count"] = triple_bonds
    
    # Degree of unsaturation (DBE)
    features["degree_of_unsaturation"] = rdMolDescriptors.CalcDegreeOfUnsaturation(mol)
    
    # Rotatable bonds
    features["rotatable_bond_count"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    
    # H-bond donors/acceptors
    features["h_bond_donors"] = rdMolDescriptors.CalcNumHBD(mol)
    features["h_bond_acceptors"] = rdMolDescriptors.CalcNumHBA(mol)
    
    # Functional groups (heuristic)
    features["num_heteroatoms"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
    
    return features


def test_trait_combination(
    combo_record_count: Counter,
    combo_to_canonical: dict,
    all_features: dict,
    traits: list[str],
    used_records: int,
    total_unique_molecules: int,
) -> tuple[int, int, int]:
    """Test a specific combination of traits and return (unique_keys, ambiguous_groups, ambiguous_molecules)."""
    key_to_canon: dict = defaultdict(set)
    
    for combo_key, canon_set in combo_to_canonical.items():
        fp, composition = combo_key
        smiles = next(iter(canon_set), None)  # Get one example canonicalized SMILES
        
        if smiles not in all_features:
            continue
        
        features = all_features[smiles]
        trait_values = tuple(features.get(t, 0) for t in traits)
        
        combined_key = (fp, composition, trait_values)
        key_to_canon[combined_key].update(canon_set)
    
    unique_keys = len(key_to_canon)
    ambiguous_groups = sum(1 for canon_set in key_to_canon.values() if len(canon_set) > 1)
    ambiguous_molecules = sum(len(canon_set) for canon_set in key_to_canon.values() if len(canon_set) > 1)
    
    return unique_keys, ambiguous_groups, ambiguous_molecules


def analyse(input_path: Path) -> None:
    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print("Input JSON is empty or not a list of records.")
        return

    print("Building MACCS + composition keys and computing structural features...")
    
    combo_record_count: Counter[tuple[tuple[int, ...], tuple[tuple[str, int], ...]]] = Counter()
    combo_to_canonical: dict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], set[str]] = defaultdict(set)
    all_features: dict[str, dict[str, int | float]] = {}
    
    missing_maccs = 0
    invalid_maccs = 0
    invalid_smiles = 0

    for i, record in enumerate(data):
        smiles = str(record.get("smiles", ""))
        maccs = record.get("maccs")

        if maccs is None:
            missing_maccs += 1
            continue

        fp = to_fp_tuple(maccs)
        if fp is None:
            invalid_maccs += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue

        canon = Chem.MolToSmiles(mol)
        composition = element_composition_signature(mol)
        combo_key = (fp, composition)

        combo_record_count[combo_key] += 1
        combo_to_canonical[combo_key].add(canon)
        
        if canon not in all_features:
            all_features[canon] = compute_structural_features(mol)

        if (i + 1) % 100_000 == 0:
            print(f"  {i+1:,} processed...", flush=True)

    total_records = len(data)
    used_records = sum(combo_record_count.values())
    
    if used_records == 0:
        print("No valid records found.")
        return

    print(f"\nFeature computation complete. {len(all_features):,} unique molecules with features.\n")

    # Baseline: MACCS + composition
    ambiguous_groups_base = sum(1 for canon_set in combo_to_canonical.values() if len(canon_set) > 1)
    ambiguous_molecules_base = sum(len(canon_set) for canon_set in combo_to_canonical.values() if len(canon_set) > 1)

    print("=" * 90)
    print("STRUCTURAL TRAIT COLLISION-BREAKING ANALYSIS")
    print("=" * 90)

    print(f"\nBaseline (MACCS + elemental composition):")
    print(f"  Ambiguous groups: {ambiguous_groups_base:,}")
    print(f"  Ambiguous molecules: {ambiguous_molecules_base:,} ({100.0 * ambiguous_molecules_base / len(all_features):.2f}%)")

    # Test individual traits
    available_traits = [
        "aromatic_ring_count",
        "num_aromatic_atoms",
        "double_bond_count",
        "triple_bond_count",
        "degree_of_unsaturation",
        "rotatable_bond_count",
        "h_bond_donors",
        "h_bond_acceptors",
        "num_heteroatoms",
    ]

    print("\n--- Individual traits (MACCS + composition + trait) ---")
    trait_results = []
    for trait in available_traits:
        unique_keys, ambiguous_groups, ambiguous_molecules = test_trait_combination(
            combo_record_count,
            combo_to_canonical,
            all_features,
            [trait],
            used_records,
            len(all_features),
        )
        
        molecules_resolved = ambiguous_molecules_base - ambiguous_molecules
        pct_reduction = (100.0 * molecules_resolved / ambiguous_molecules_base) if ambiguous_molecules_base else 0
        trait_results.append((trait, molecules_resolved, pct_reduction, ambiguous_molecules))
        
        print(f"  + {trait:<30} → {ambiguous_molecules:>7,} ambiguous moles ({pct_reduction:>6.2f}% reduction, {molecules_resolved:>6,} resolved)")

    # Sort by improvement
    trait_results.sort(key=lambda x: x[1], reverse=True)
    
    # Test top trait combinations
    print("\n--- Top trait combinations ---")
    if len(trait_results) >= 1:
        top_trait = trait_results[0][0]
        print(f"\nBest single trait: {top_trait}")
        
        # Try combining top traits
        if len(trait_results) >= 2:
            second_trait = trait_results[1][0]
            unique_keys, ambiguous_groups, ambiguous_molecules = test_trait_combination(
                combo_record_count,
                combo_to_canonical,
                all_features,
                [top_trait, second_trait],
                used_records,
                len(all_features),
            )
            molecules_resolved = ambiguous_molecules_base - ambiguous_molecules
            pct_reduction = (100.0 * molecules_resolved / ambiguous_molecules_base) if ambiguous_molecules_base else 0
            print(f"+ {top_trait} + {second_trait}")
            print(f"  → {ambiguous_molecules:,} ambiguous molecules ({pct_reduction:.2f}% reduction, {molecules_resolved:,} resolved)")

        if len(trait_results) >= 3:
            third_trait = trait_results[2][0]
            unique_keys, ambiguous_groups, ambiguous_molecules = test_trait_combination(
                combo_record_count,
                combo_to_canonical,
                all_features,
                [top_trait, second_trait, third_trait],
                used_records,
                len(all_features),
            )
            molecules_resolved = ambiguous_molecules_base - ambiguous_molecules
            pct_reduction = (100.0 * molecules_resolved / ambiguous_molecules_base) if ambiguous_molecules_base else 0
            print(f"+ {top_trait} + {second_trait} + {third_trait}")
            print(f"  → {ambiguous_molecules:,} ambiguous molecules ({pct_reduction:.2f}% reduction, {molecules_resolved:,} resolved)")

    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test which structural traits best reduce molecule collisions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "C:/Users/Tom/Documents/Planning/MolecularPredictionMSc/Data Processing/MoNa/MoNA-filtered-curated-with-maccs.json"
        ),
        help="Path to JSON file containing smiles and maccs fields.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    analyse(args.input)
