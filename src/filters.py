from rdkit import Chem
from typing import Union, Literal

# ---------------------------------------------------------------------------
# Pre-solve formula complexity filter (works on atom_counts dicts, not SMILES)
# ---------------------------------------------------------------------------

def _dou_from_counts(atom_counts: dict) -> float:
    numerator = (
        2 * (atom_counts.get("num_carbons", 0) + atom_counts.get("num_sulfurs", 0))
        + 2
        + atom_counts.get("num_nitrogens", 0)
        + atom_counts.get("num_phosphorus", 0)
        - atom_counts.get("num_hydrogens", 0)
    )
    return numerator / 2


def formula_complexity_score(mol: Union[str, dict]) -> float:
    """Estimate ASP solve complexity as non_H_atoms * (DoU + 1).

    Accepts a SMILES string or an atom_counts dict (keys: num_carbons, etc.).

    Calibrated against benchmark_asp.py timings:
      score ~8   → < 0.1s (e.g. C3H6O)
      score ~16  → 0.2-3s  (e.g. C7H14O)
      score ~35  → ~3s     (e.g. C7H8)
      score ~40  → 10-30s  (e.g. C8H10, C7H8O)
      score ~50+ → minutes (e.g. C5H8O1N4)
    """
    if isinstance(mol, str):
        rdmol = Chem.MolFromSmiles(mol)
        non_h = rdmol.GetNumAtoms()
        dou = max(0.0, _get_dou_from_structure(rdmol))
    else:
        non_h = sum(v for k, v in mol.items() if k != "num_hydrogens")
        dou = max(0.0, _dou_from_counts(mol))
    return non_h * (dou + 1)


def formula_complexity_filter(mol: Union[str, dict], max_score: float = 40.0) -> bool:
    """Return True if molecule is tractable for the ASP solver.

    Accepts a SMILES string or an atom_counts dict (keys: num_carbons, etc.).
    Returns False (skip) when score >= max_score. Default threshold of 40
    rejects 10-30s+ formulas while allowing most sub-10s cases through.
    """
    return formula_complexity_score(mol) < max_score

PRESETS = {
        "ultralight": {"heavy_atoms": 6,  "dou": 0},
        "light":      {"heavy_atoms": 10, "dou": 4},
        "medium":     {"heavy_atoms": 18, "dou": 7},
        "heavy":      {"heavy_atoms": 26, "dou": 12}
}

def _get_dou_from_structure(mol: Chem.Mol) -> int:
    # 1. Start with the total number of rings
    rings = mol.GetRingInfo().NumRings()
    
    # 2. Add the number of pi bonds
    pi_bonds = 0
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.AROMATIC:
            # Aromatic bonds contribute 0.5 pi bonds per bond in a Kekulé sense
            pi_bonds += 0.5
            
    return int(rings + pi_bonds)

def element_type_filter(smiles: str, exclude_elements: list = []):
    """
    Filters out molecules with the presence of any of the specified elements.
    """
    mol = Chem.MolFromSmiles(smiles)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]

    return set(atoms) - set(exclude_elements) == set(atoms)

def small_mol_filter(smiles: str, min_heavy_atoms: int = 4):
    """
    Filters out tiny molecules with less non-hydrogen atoms than `min_heavy_atoms`
    """
    mol = Chem.MolFromSmiles(smiles)
    heavy_atom_num = mol.GetNumAtoms()

    return heavy_atom_num >= min_heavy_atoms


def heavy_atom_filter(smiles: str, max_heavy_atoms: int = None, preset: Union[Literal["ultralight", "light", "medium", "heavy"], None] = None) -> bool:
    """
    Filters out molecules which have more non-hydrogen atoms than the specified value. Setting the `preset` arg overrides `max_heavy_atoms`.

    Returns `True` if the molecule passes the filter (i.e. has less (or equal) heavy atoms than max_heavy_atoms), `False` otherwise.

    Presets:
    ```
        if preset == "ultralight":
            max_heavy_atoms = 6
        elif preset == "light":
            max_heavy_atoms = 10
        elif preset == "medium":
            max_heavy_atoms = 18
        elif preset == "heavy":
            max_heavy_atoms = 26
    ```
    """
    mol = Chem.MolFromSmiles(smiles)
    heavy_atom_num = mol.GetNumAtoms()

    assert max_heavy_atoms is not None or preset, "One of `max_heavy_atoms` and `preset` must not be None."

    if preset == "ultralight":
        max_heavy_atoms = PRESETS["ultralight"]["heavy_atoms"]
    elif preset == "light":
        max_heavy_atoms = PRESETS["light"]["heavy_atoms"]
    elif preset == "medium":
        max_heavy_atoms = PRESETS["medium"]["heavy_atoms"]
    elif preset == "heavy":
        max_heavy_atoms = PRESETS["heavy"]["heavy_atoms"]
    elif preset:
        raise ValueError(f"Invalid value for arg 'preset'. Only the following presets are supported: f{PRESETS.keys()}")

    # If max_heavy_atoms is set
    return heavy_atom_num <= max_heavy_atoms

def dou_filter(smiles: str, max_dou: int = None, preset: Union[Literal["ultralight", "light", "medium", "heavy"], None] = None):
    mol = Chem.MolFromSmiles(smiles)
    dou = _get_dou_from_structure(mol)

    assert max_dou is not None or preset, "One of `max_dou` and `preset` must not be None."

    if preset == "ultralight":
        max_dou = PRESETS["ultralight"]["dou"]
    elif preset == "light":
        max_dou = PRESETS["light"]["dou"]
    elif preset == "medium":
        max_dou = PRESETS["medium"]["dou"]
    elif preset == "heavy":
        max_dou = PRESETS["heavy"]["dou"]
    elif preset:
        raise ValueError(f"Invalid value for arg 'preset'. Only the following presets are supported: f{PRESETS.keys()}")

    # If max_heavy_atoms is set
    return dou <= max_dou





