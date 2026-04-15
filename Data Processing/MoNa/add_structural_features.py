"""
Add structural, graph-derived features to a MoNA JSON file and combine them
with an existing MACCS vector.

The script expects records shaped roughly like:

    {
        "smiles": "CCO",
        "spectrum": "43.02:100 45.03:12 ...",
        "maccs": [0, 1, 0, 0, ...]
    }

It computes a structural feature vector from the SMILES and writes back:

    {
        "smiles": "CCO",
        "spectrum": "...",
        "maccs": [...],
        "structural_features": {...},
        "structural_vector": [...],
        "feature_vector": [...]
    }

The combined feature vector is simply:

    feature_vector = maccs + structural_vector

The structural features are chosen to be graph-based or subgraph-based so they
can be translated into constraints in a downstream logic / ASP model.

Usage:
    pip install rdkit
    python add_structural_features.py
    python add_structural_features.py --input "C:/path/to/MoNA-filtered-curated-with-maccs.json" --output "C:/path/to/MoNA-with-structure.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("RDKit is required: pip install rdkit")


COMMON_ELEMENTS = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si", "B"]

# SMARTS patterns for common structural motifs that can be treated as logical
# substructure constraints later on.
SUBSTRUCTURE_PATTERNS: dict[str, str] = {
    "carboxylic_acid": "C(=O)[OX2H1]",
    "ester": "C(=O)O[C;!$(C=O)]",
    "amide": "C(=O)N",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "alcohol": "[CX4][OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "nitro": "[NX3](=O)=O",
    "phosphate": "P(=O)(O)(O)O",
    "sulfate": "S(=O)(=O)(O)O",
    "halogenated": "[F,Cl,Br,I]",
    "aromatic_ring": "a1aaaaa1",
    "nitrile": "[CX2]#N",
    "sulfonamide": "S(=O)(=O)N",
    "sulfoxide": "[#16X3](=O)",
    "phosphonate": "P(=O)(O)(O)[#6]",
    "imide": "[NX3]([CX3](=O))[CX3](=O)",
    "lactam": "[NX3;R][CX3;R](=O)",
    "lactone": "[OX2;R][CX3;R](=O)",
    "phenol": "c[OX2H]",
    "aniline": "c[NX3;H2,H1;!$(NC=O)]",
}

COMPILED_SUBSTRUCTURE_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in SUBSTRUCTURE_PATTERNS.items()
}


STRUCTURAL_VECTOR_ORDER = [
    "exact_mass_da",
    "molecular_weight_da",
    "heavy_atom_count",
    "total_atom_count_with_h",
    "connected_component_count",
    "heteroatom_count",
    "ring_count",
    "ring_size_3_count",
    "ring_size_4_count",
    "ring_size_5_count",
    "ring_size_6_count",
    "ring_size_7plus_count",
    "fused_ring_pair_count",
    "ring_system_count",
    "fused_ring_system_count",
    "aromatic_ring_count",
    "aliphatic_ring_count",
    "aromatic_atom_count",
    "aromatic_heteroatom_count",
    "aromatic_n_count",
    "aromatic_o_count",
    "aromatic_s_count",
    "aromatic_p_count",
    "heteroatom_in_ring_count",
    "heteroatom_exocyclic_count",
    "double_bond_count",
    "triple_bond_count",
    "stereo_double_bond_E_count",
    "stereo_double_bond_Z_count",
    "stereo_double_bond_unspecified_count",
    "chiral_center_count",
    "assigned_chiral_center_count",
    "unassigned_chiral_center_count",
    "rotatable_bond_count",
    "degree_of_unsaturation",
    "branch_point_count",
    "degree_1_atom_count",
    "degree_2_atom_count",
    "degree_3_atom_count",
    "degree_4plus_atom_count",
    "n_valence_1_count",
    "n_valence_2_count",
    "n_valence_3_count",
    "n_valence_4plus_count",
    "o_valence_1_count",
    "o_valence_2_count",
    "o_valence_3plus_count",
    "s_valence_2_count",
    "s_valence_4_count",
    "s_valence_6plus_count",
    "p_valence_3_count",
    "p_valence_5plus_count",
    "spiro_atom_count",
    "bridgehead_atom_count",
    "formal_charge",
    *[f"element_{elem}" for elem in COMMON_ELEMENTS],
    *SUBSTRUCTURE_PATTERNS.keys(),
]


def to_fp_tuple(maccs: object) -> tuple[int, ...] | None:
    if not isinstance(maccs, list):
        return None
    try:
        return tuple(int(x) for x in maccs)
    except (TypeError, ValueError):
        return None


def parse_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles.strip())


def element_composition_signature(mol: Chem.Mol) -> dict[str, int]:
    """Count elements after adding implicit hydrogens.

    This is used to derive formula-like features that are still graph-based.
    """
    expanded = Chem.AddHs(mol)
    counts: Counter[str] = Counter()
    for atom in expanded.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return dict(counts)


def count_substructure_matches(mol: Chem.Mol, smarts: str) -> int:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return 0
    return len(mol.GetSubstructMatches(pattern, uniquify=True))


def count_substructure_matches_compiled(mol: Chem.Mol, pattern: Chem.Mol | None) -> int:
    if pattern is None:
        return 0
    return len(mol.GetSubstructMatches(pattern, uniquify=True))


def ring_topology_features(mol: Chem.Mol) -> dict[str, int]:
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    ring_size_3 = 0
    ring_size_4 = 0
    ring_size_5 = 0
    ring_size_6 = 0
    ring_size_7plus = 0
    aromatic_ring_count = 0
    aliphatic_ring_count = 0

    ring_sets = [set(r) for r in atom_rings]
    fused_ring_pair_count = 0

    for ring in atom_rings:
        size = len(ring)
        if size == 3:
            ring_size_3 += 1
        elif size == 4:
            ring_size_4 += 1
        elif size == 5:
            ring_size_5 += 1
        elif size == 6:
            ring_size_6 += 1
        elif size >= 7:
            ring_size_7plus += 1

        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_ring_count += 1
        else:
            aliphatic_ring_count += 1

    # Fused rings share at least two atoms.
    ring_graph: dict[int, set[int]] = {i: set() for i in range(len(ring_sets))}
    for i in range(len(ring_sets)):
        for j in range(i + 1, len(ring_sets)):
            shared = len(ring_sets[i] & ring_sets[j])
            if shared >= 2:
                fused_ring_pair_count += 1
            if shared >= 1:
                ring_graph[i].add(j)
                ring_graph[j].add(i)

    ring_system_count = 0
    fused_ring_system_count = 0
    seen = set()
    for start in ring_graph:
        if start in seen:
            continue
        ring_system_count += 1
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(ring_graph[node] - seen)
        if len(component) > 1:
            fused_ring_system_count += 1

    return {
        "ring_size_3_count": ring_size_3,
        "ring_size_4_count": ring_size_4,
        "ring_size_5_count": ring_size_5,
        "ring_size_6_count": ring_size_6,
        "ring_size_7plus_count": ring_size_7plus,
        "aromatic_ring_count": aromatic_ring_count,
        "aliphatic_ring_count": aliphatic_ring_count,
        "fused_ring_pair_count": fused_ring_pair_count,
        "ring_system_count": ring_system_count,
        "fused_ring_system_count": fused_ring_system_count,
    }


def compute_structural_features(mol: Chem.Mol) -> dict[str, int | float]:
    """Compute features that are directly derived from molecular structure."""
    features: dict[str, int | float] = {}

    expanded = Chem.AddHs(mol)
    composition = element_composition_signature(mol)
    ring_features = ring_topology_features(mol)

    features["exact_mass_da"] = float(Descriptors.ExactMolWt(mol))
    features["molecular_weight_da"] = float(Descriptors.MolWt(mol))
    features["heavy_atom_count"] = mol.GetNumHeavyAtoms()
    features["total_atom_count_with_h"] = expanded.GetNumAtoms()
    features["connected_component_count"] = len(Chem.GetMolFrags(mol))
    features["heteroatom_count"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
    features["ring_count"] = rdMolDescriptors.CalcNumRings(mol)

    features.update(ring_features)
    features["aromatic_atom_count"] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    features["aromatic_heteroatom_count"] = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic() and atom.GetAtomicNum() not in (1, 6)
    )
    features["aromatic_n_count"] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic() and atom.GetSymbol() == "N")
    features["aromatic_o_count"] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic() and atom.GetSymbol() == "O")
    features["aromatic_s_count"] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic() and atom.GetSymbol() == "S")
    features["aromatic_p_count"] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic() and atom.GetSymbol() == "P")
    features["heteroatom_in_ring_count"] = sum(
        1 for atom in mol.GetAtoms() if atom.IsInRing() and atom.GetAtomicNum() not in (1, 6)
    )
    features["heteroatom_exocyclic_count"] = sum(
        1 for atom in mol.GetAtoms() if not atom.IsInRing() and atom.GetAtomicNum() not in (1, 6)
    )
    features["double_bond_count"] = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
    features["triple_bond_count"] = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.TRIPLE)
    features["stereo_double_bond_E_count"] = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() == Chem.BondStereo.STEREOE
    )
    features["stereo_double_bond_Z_count"] = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() == Chem.BondStereo.STEREOZ
    )
    features["stereo_double_bond_unspecified_count"] = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
    )

    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    features["chiral_center_count"] = len(chiral_centers)
    features["assigned_chiral_center_count"] = sum(1 for _, label in chiral_centers if label != "?")
    features["unassigned_chiral_center_count"] = features["chiral_center_count"] - features["assigned_chiral_center_count"]

    features["rotatable_bond_count"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    features["degree_of_unsaturation"] = (
        features["double_bond_count"]
        + 2 * features["triple_bond_count"]
        + features["ring_count"]
    )

    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    features["branch_point_count"] = sum(1 for atom in heavy_atoms if atom.GetDegree() >= 3)
    features["degree_1_atom_count"] = sum(1 for atom in heavy_atoms if atom.GetDegree() == 1)
    features["degree_2_atom_count"] = sum(1 for atom in heavy_atoms if atom.GetDegree() == 2)
    features["degree_3_atom_count"] = sum(1 for atom in heavy_atoms if atom.GetDegree() == 3)
    features["degree_4plus_atom_count"] = sum(1 for atom in heavy_atoms if atom.GetDegree() >= 4)

    n_atoms = [atom for atom in heavy_atoms if atom.GetSymbol() == "N"]
    o_atoms = [atom for atom in heavy_atoms if atom.GetSymbol() == "O"]
    s_atoms = [atom for atom in heavy_atoms if atom.GetSymbol() == "S"]
    p_atoms = [atom for atom in heavy_atoms if atom.GetSymbol() == "P"]

    features["n_valence_1_count"] = sum(1 for atom in n_atoms if atom.GetTotalValence() == 1)
    features["n_valence_2_count"] = sum(1 for atom in n_atoms if atom.GetTotalValence() == 2)
    features["n_valence_3_count"] = sum(1 for atom in n_atoms if atom.GetTotalValence() == 3)
    features["n_valence_4plus_count"] = sum(1 for atom in n_atoms if atom.GetTotalValence() >= 4)

    features["o_valence_1_count"] = sum(1 for atom in o_atoms if atom.GetTotalValence() == 1)
    features["o_valence_2_count"] = sum(1 for atom in o_atoms if atom.GetTotalValence() == 2)
    features["o_valence_3plus_count"] = sum(1 for atom in o_atoms if atom.GetTotalValence() >= 3)

    features["s_valence_2_count"] = sum(1 for atom in s_atoms if atom.GetTotalValence() == 2)
    features["s_valence_4_count"] = sum(1 for atom in s_atoms if atom.GetTotalValence() == 4)
    features["s_valence_6plus_count"] = sum(1 for atom in s_atoms if atom.GetTotalValence() >= 6)

    features["p_valence_3_count"] = sum(1 for atom in p_atoms if atom.GetTotalValence() == 3)
    features["p_valence_5plus_count"] = sum(1 for atom in p_atoms if atom.GetTotalValence() >= 5)

    features["spiro_atom_count"] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    features["bridgehead_atom_count"] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    features["formal_charge"] = Chem.GetFormalCharge(mol)

    for elem in COMMON_ELEMENTS:
        features[f"element_{elem}"] = int(composition.get(elem, 0))

    for name, pattern in COMPILED_SUBSTRUCTURE_PATTERNS.items():
        features[name] = count_substructure_matches_compiled(mol, pattern)

    return features


def structural_feature_vector(features: dict[str, int | float]) -> list[int | float]:
    return [features[name] for name in STRUCTURAL_VECTOR_ORDER]


def extract_maccs(record: dict, maccs_field: str) -> list[int] | None:
    """Read the MACCS vector from a record.

    The field name defaults to 'maccs', but the loader will also accept a few
    common aliases if the requested field is missing.
    """
    candidates = [maccs_field, "maccs", "MACCS", "fingerprint", "fp"]
    for field in candidates:
        if field in record:
            fp = to_fp_tuple(record.get(field))
            if fp is not None:
                return list(fp)
    return None


def build_feature_records(input_path: Path, output_path: Path, maccs_field: str) -> None:
    print(f"Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit("Input JSON must be a list of records.")

    total = len(data)
    kept = 0
    dropped_no_smiles = 0
    dropped_no_maccs = 0
    dropped_invalid_smiles = 0

    print(f"Processing {total:,} records ...")

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        first = True

        for idx, record in enumerate(data, start=1):
            smiles = str(record.get("smiles", "")).strip()
            if not smiles:
                dropped_no_smiles += 1
                continue

            maccs = extract_maccs(record, maccs_field)
            if maccs is None:
                dropped_no_maccs += 1
                continue

            mol = parse_smiles(smiles)
            if mol is None:
                dropped_invalid_smiles += 1
                continue

            features = compute_structural_features(mol)
            vector = structural_feature_vector(features)
            combined_vector = list(maccs) + vector

            out = dict(record)
            out["smiles"] = smiles
            out["maccs"] = maccs
            out["structural_features"] = features
            out["structural_vector"] = vector
            out["feature_vector"] = combined_vector

            if not first:
                f_out.write(",\n")
            json.dump(out, f_out)
            first = False
            kept += 1

            if idx % 10_000 == 0:
                print(f"  {idx:,} / {total:,} processed, kept {kept:,} ...", flush=True)

        f_out.write("\n]")

    print("\nFinished.")
    print(f"  Kept:                  {kept:,} / {total:,}")
    print(f"  Dropped - no SMILES:    {dropped_no_smiles:,}")
    print(f"  Dropped - no MACCS:     {dropped_no_maccs:,}")
    print(f"  Dropped - invalid SMILES:{dropped_invalid_smiles:,}")
    print(f"Output written to {output_path}")
    print(f"Structural vector length: {len(STRUCTURAL_VECTOR_ORDER)}")
    print(f"Combined vector length:    MACCS length + {len(STRUCTURAL_VECTOR_ORDER)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add graph-derived structural features to MoNA JSON records and merge them with MACCS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-curated-with-maccs.json"
        ),
        help="Path to the input JSON file containing smiles, spectrum, and MACCS vectors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "C:\\Users\\Tom\\Desktop\\Nieuwe map\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-filtered-curated-with-maccs-and-structure.json"
        ),
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--maccs-field",
        type=str,
        default="maccs",
        help="Name of the field that contains the MACCS vector.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    build_feature_records(args.input, args.output, args.maccs_field)