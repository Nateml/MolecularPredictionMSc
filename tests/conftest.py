import pytest
import re
from rdkit import Chem

from predicates.interface import PredictionEntry, PredictionInterface
from predicates.compiler import compile_asp_constraint
from asp_bridge import solve

# --------------------------
# Cache
# --------------------------

_PIPELINE_CACHE = {}

def make_hashable(predictions):
    def convert_value(v):
        if isinstance(v, int):
            return v

        # Handle PredictionEntry
        if hasattr(v, "pred_set"):
            return (
                "PredictionEntry",
                tuple(sorted(v.pred_set))
            )

        raise TypeError(f"Unsupported type in predictions: {type(v)}")

    return tuple(
        sorted(
            (str(k), convert_value(v))
            for k, v in predictions.items()
        )
    )


@pytest.fixture(scope="function")
def cached_run_pipeline(run_pipeline):
    def wrapper(predictions):
        key = make_hashable(predictions)

        if key not in _PIPELINE_CACHE:
            _PIPELINE_CACHE[key] = list(run_pipeline(predictions))

        return _PIPELINE_CACHE[key]

    return wrapper

# -------------------------
# Core helpers
# -------------------------

@pytest.fixture(scope="session")
def smiles_to_mol():
    def _fn(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smi}")
        return mol
    return _fn


@pytest.fixture(scope="session")
def canonicalize(smiles_to_mol):
    def _fn(smiles_list):
        out = set()
        for smi in smiles_list:
            mol = smiles_to_mol(smi)
            out.add(Chem.MolToSmiles(mol, canonical=True))
        return sorted(out)
    return _fn


@pytest.fixture(scope="session")
def run_pipeline(canonicalize):
    def _fn(predictions: dict):
        interface = PredictionInterface.model_validate(predictions)

        asp_constraints, atom_counts = compile_asp_constraint(interface)

        output, _ = solve(
            asp_constraints,
            atom_counts,
            output_list_of_smiles=True
        )

        return canonicalize(output)
    return _fn


# -------------------------
# Formula parsing
# -------------------------

@pytest.fixture
def parse_formula():
    def _fn(formula: str):
        counts = {"C": 0, "H": 0, "O": 0, "N": 0, "S": 0, "P": 0}

        matches = re.findall(r"([A-Z])(\d*)", formula)

        for element, count in matches:
            counts[element] = int(count) if count else 1

        return counts
    return _fn


# -------------------------
# Test data
# -------------------------
@pytest.fixture
def formulas():
    return [
        "C2H6",     # saturated hydrocarbon (no heteroatoms, no multiple bonds)
        "C2H4",     # allows C=C
        "C2H2",     # allows C≡C
        "C2H4O",    # allows C-O single + carbonyl
        "C2H3N",    # allows C=N and C≡N
        "CH3NO",    # allows N-O and C-N and C-O combinations
    ]


# -------------------------
# Prediction builder
# -------------------------

@pytest.fixture
def build_predictions(parse_formula):
    def _fn(formula: str, constraint=None):
        counts = parse_formula(formula)

        pred = {
            "num_carbons": counts["C"],
            "num_hydrogens": counts["H"],
            "num_oxygens": counts["O"],
            "num_nitrogens": counts["N"],
            "num_sulfurs": counts["S"],
            "num_phosphorus": counts["P"],
        }

        if constraint is not None:
            pred[constraint.key] = PredictionEntry(pred_set=[1])

        return pred

    return _fn