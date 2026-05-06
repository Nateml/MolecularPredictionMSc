"""
Automatically identify which structural descriptors are most discriminative
for breaking MACCS+composition collisions by analyzing variance within groups.

Usage:
    pip install rdkit
    python identify_traits.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
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


def compute_all_descriptors(mol: Chem.Mol) -> dict[str, float | int]:
    """Compute only STRUCTURAL descriptors (purely topological/connectivity-based, no physicochemical properties)."""
    descriptors = {}
    
    # Basic atom/bond counts (structural)
    descriptors["num_atoms"] = mol.GetNumAtoms()
    descriptors["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
    descriptors["num_bonds"] = mol.GetNumBonds()
    
    # Ring information (structural)
    ri = mol.GetRingInfo()
    descriptors["num_aromatic_rings"] = len([r for r in ri.AtomRings() if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)])
    descriptors["num_aliphatic_rings"] = len([r for r in ri.AtomRings() if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)])
    
    # Bond types (structural)
    descriptors["num_double_bonds"] = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    descriptors["num_triple_bonds"] = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE)
    
    # Structural connectivity/topology (version-safe unsaturation proxy)
    descriptors["unsaturation_index"] = (
        descriptors["num_double_bonds"]
        + 2 * descriptors["num_triple_bonds"]
        + rdMolDescriptors.CalcNumRings(mol)
    )
    descriptors["num_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    descriptors["num_heteroatoms"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
    
    # H-bond pattern (structural recognition)
    descriptors["num_h_bond_donors"] = rdMolDescriptors.CalcNumHBD(mol)
    descriptors["num_h_bond_acceptors"] = rdMolDescriptors.CalcNumHBA(mol)
    
    # Aromaticity (structural pattern)
    descriptors["num_aromatic_atoms"] = len([a for a in mol.GetAtoms() if a.GetIsAromatic()])
    descriptors["frac_aromatic_atoms"] = descriptors["num_aromatic_atoms"] / descriptors["num_heavy_atoms"] if descriptors["num_heavy_atoms"] > 0 else 0
    
    # Excluded: LogP (physicochemical), TPSA (surface property), MolWt (composition-derived), PEOE_VSA (electronic property)
    
    return descriptors


def compute_descriptor_variance_in_collisions(combo_to_canonical: dict, all_descriptors: dict) -> dict[str, float]:
    """
    For each collision group, compute variance of each descriptor.
    Return average variance across collision groups for each descriptor.
    """
    descriptor_variances: dict[str, list[float]] = defaultdict(list)
    
    for combo_key, canon_set in combo_to_canonical.items():
        if len(canon_set) <= 1:
            continue  # Skip non-collision groups
        
        # Get descriptors for all molecules in this collision group
        collision_descrs = []
        for canon in canon_set:
            if canon in all_descriptors:
                collision_descrs.append(all_descriptors[canon])
        
        if not collision_descrs or len(collision_descrs) < 2:
            continue
        
        # For each descriptor, compute variance within this collision group
        all_desc_names = collision_descrs[0].keys()
        for desc_name in all_desc_names:
            values = [d[desc_name] for d in collision_descrs]
            
            # Compute coefficient of variation (std / mean) to normalize
            mean_val = sum(values) / len(values)
            if mean_val != 0:
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                std_dev = variance ** 0.5
                coeff_var = std_dev / abs(mean_val) if mean_val != 0 else 0
                descriptor_variances[desc_name].append(coeff_var)
    
    # Compute average coefficient of variation for each descriptor
    avg_variances = {}
    for desc_name, variances in descriptor_variances.items():
        if variances:
            avg_variances[desc_name] = sum(variances) / len(variances)
    
    return avg_variances


def analyse(input_path: Path, top_n: int) -> None:
    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print("Input JSON is empty or not a list of records.")
        return

    print("Building MACCS + composition keys and computing descriptors...")
    
    combo_to_canonical: dict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], set[str]] = defaultdict(set)
    all_descriptors: dict[str, dict[str, float | int]] = {}
    
    missing_maccs = 0
    invalid_maccs = 0
    invalid_smiles = 0

    for i, record in enumerate(data):
        smiles = str(record.get("smiles", ""))
        maccs = record.get("maccs")

        if maccs is None:
            missing_maccs += 1
            continue

        fp = tuple(int(x) for x in maccs) if isinstance(maccs, list) else None
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

        combo_to_canonical[combo_key].add(canon)
        
        if canon not in all_descriptors:
            all_descriptors[canon] = compute_all_descriptors(mol)

        if (i + 1) % 100_000 == 0:
            print(f"  {i+1:,} processed...", flush=True)

    print(f"\nDescriptor computation complete. {len(all_descriptors):,} unique molecules.\n")

    # Analyze variance within collision groups
    print("Analyzing descriptor variance within collision groups...")
    avg_variances = compute_descriptor_variance_in_collisions(combo_to_canonical, all_descriptors)
    
    # Compute baseline collision stats
    total_collision_groups = sum(1 for canon_set in combo_to_canonical.values() if len(canon_set) > 1)
    total_colliding_molecules = sum(len(canon_set) for canon_set in combo_to_canonical.values() if len(canon_set) > 1)

    print("=" * 100)
    print("AUTOMATICALLY IDENTIFIED DISCRIMINATIVE DESCRIPTORS")
    print("=" * 100)

    print(f"\nCollision baseline (MACCS + elemental composition):")
    print(f"  Collision groups: {total_collision_groups:,}")
    print(f"  Molecules in collisions: {total_colliding_molecules:,}")

    print(f"\n--- Top {top_n} most discriminative descriptors (by variance within collision groups) ---")
    print("(High variance = this trait differs greatly among molecules in the same collision group)")
    print()
    
    sorted_descrs = sorted(avg_variances.items(), key=lambda x: x[1], reverse=True)
    
    for idx, (desc_name, avg_var) in enumerate(sorted_descrs[:top_n], start=1):
        print(f"  {idx:>2}. {desc_name:<40} (avg coeff. variation: {avg_var:>8.4f})")

    print("\n" + "=" * 100)
    print("INTERPRETATION:")
    print("=" * 100)
    print("""
Descriptors with HIGH variance (top of list) are excellent for breaking collisions
because they differ significantly among molecules already assigned to the same group.

Descriptors with LOW variance are poor discriminators—they're similar across 
colliding molecules, so they wouldn't help separate them.

RECOMMENDATION: Use descriptors from the top 5-10 for your combined key to 
maximize collision-breaking power.
    """)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify which structural descriptors are most discriminative for collision resolution."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "C:/Users/Tom/Documents/Planning/MolecularPredictionMSc/Data Processing/MoNa/MoNA-filtered-curated-with-maccs.json"
        ),
        help="Path to JSON file containing smiles and maccs fields.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many top descriptors to display.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    if args.top < 1:
        sys.exit("--top must be >= 1")
    analyse(args.input, args.top)
