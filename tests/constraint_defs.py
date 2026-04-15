from dataclasses import dataclass
from typing import Callable
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import MACCSkeys

from predicates.definitions import MACCS_DEFINITIONS


@dataclass
class ConstraintTest:
    key: int              # MACCS index (e.g. 154)
    name: str             # predicate name (e.g. "has_carbonyl")
    rdkit_check: Callable[[Mol], bool]


def make_maccs_check(key: int):
    def check(mol: Mol) -> bool:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return bool(fp[key])
    return check


constraints = [
    ConstraintTest(
        key=k,
        name=definition["predicate"],
        rdkit_check=make_maccs_check(k),
    )
    for k, definition in MACCS_DEFINITIONS.items()
]


# You can add custom constraints here if you want to test beyond the MACCS keys.
# Define CUSTOM_DEFINITIONS in predicates/definitions.py and add corresponding checks here.

# custom_checks = {
#     "has_c_o_single_bond": lambda mol: any(
#         bond.GetBondType() == Chem.rdchem.BondType.SINGLE and
#         (bond.GetBeginAtom().GetSymbol() == "C" and bond.GetEndAtom().GetSymbol() == "O" or
#          bond.GetBeginAtom().GetSymbol() == "O" and bond.GetEndAtom().GetSymbol() == "C")
#         for bond in mol.GetBonds()
#     ),
# }

# constraints.extend([
#     ConstraintTest(
#         key=k,
#         name=definition["predicate"],
#         rdkit_check=custom_checks[k],
#     )
#     for k, definition in CUSTOM_DEFINITIONS.items()
#     if k in custom_checks
# ])