import pytest
import threading
import re
from rdkit import Chem

from predicates.interface import PredictionEntry, PredictionInterface
from predicates.compiler import compile_asp_constraint
from asp.bridge import solve

class PipelineTimeoutError(Exception):
    def __init__(self, predictions: dict, timeout: float | None):
        self.predictions = predictions
        self.timeout = timeout
        msg = (
            f"Pipeline previously timed out for predictions:\n{predictions}"
            if timeout is None
            else f"Pipeline timed out after {timeout}s for predictions:\n{predictions}"
        )
        super().__init__(msg)

# --------------------------
# Cache
# --------------------------

_PIPELINE_CACHE = {}
_PIPELINE_TIMEOUTS = set()

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

@pytest.fixture(scope="session")
def cached_run_pipeline(run_pipeline):
    def wrapper(predictions):
        key = make_hashable(predictions)

        if key in _PIPELINE_TIMEOUTS:
            raise PipelineTimeoutError(predictions, timeout=None)

        if key not in _PIPELINE_CACHE:
            try:
                _PIPELINE_CACHE[key] = list(run_pipeline(predictions))
            except PipelineTimeoutError:
                _PIPELINE_TIMEOUTS.add(key)
                raise

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
    def _fn(predictions: dict, timeout: float = 20.0):
        result = [None]
        exception = [None]

        def _target():
            try:
                interface = PredictionInterface.model_validate(predictions)
                asp_constraints, atom_counts = compile_asp_constraint(interface)
                output, _, _ = solve(
                    asp_constraints,
                    atom_counts,
                    output_list_of_smiles=True,
                )
                result[0] = canonicalize(output)
            except Exception as e:
                exception[0] = e

        # Thread join with slack covers grounding hangs (not covered by solve's timeout).
        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout + 2.0)

        if thread.is_alive():
            raise PipelineTimeoutError(predictions, timeout)
        if exception[0] is not None:
            raise exception[0]

        return result[0]

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
        "C2H6",     # saturated hydrocarbon
        "C2H4",     # C=C
        "C2H2",     # C≡C
        "C2H4O",    # C-O + carbonyl
        "C2H3N",    # C=N and C≡N
        "CH3NO",    # N-O, C-N, C-O combinations
        "H2SO2",    # S=O and S-O
        "N2H4",     # N-N and N-H
        "CH2O2",
        "C4H4N2",
        "C4H4N2",
        "PH3",
        "H2S2",
        "H4N2O2S",
        "CH5NS",
        "CH4O2S",
        "CH3NS",
        "HNO2",
        "C2H7NO",
        "C4H8",
        "H2SO3",    # S bonded to 3 O's: has_os_oo, has_q_ooo
        "CH5NO2",   # N bonded to 2 O + C: has_on_oc; Q-CH2-Q: has_phosphorus
        "C7H14",    # 7-membered ring: has_7m_ring, has_ch3_aaa_ch2_a
        "C4H10",    # 3+ CH3 groups: has_ch3_gt2
        "C2H4O4",   # 4 oxygens: has_oxygen_gt3
        "C4H4S",    # S exo to ring: has_cq_cc_a, has_s_exo_aromatic, has_ring_atom_to_s
        "C3H4N2O",  # O exo to aromatic ring: has_o_exo_aromatic
        "N10H8",    # bicyclic all-N: has_8m_or_larger_ring, has_6m_ring_gt1
        "CH5N3",    # guanidine-like: has_nc_nn (C bonded to 3 N's)
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
