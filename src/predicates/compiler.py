from typing import Tuple

from src.predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS
from src.predicates.interface import PredictionInterface, PredictionEntry
from src.predicates.definitions import HELPER_PREDICATES

def compile_asp_constraint(interface: PredictionInterface, HELPERS = HELPER_PREDICATES) -> Tuple[str, dict[str, int]]:
    """
    Compiles the prediction interface into a list of ASP rules and constraints.

    Args:
        interface (PredictionInterface): The prediction interface containing the prediction entries and atom counts.

    """
    rules = set()
    constraints = []
    atom_counts = {}

    # Compile helper predicates
    for helper in HELPERS.values():
        rules.add(helper)

    for k, entry in interface.root.items():

        # Check if the entry is an integer (for atom counts) or a PredictionEntry (for feature predictions)
        if isinstance(entry, int):
            # encode as constant
            atom_counts[k] = entry 
            continue

        assert isinstance(entry, PredictionEntry)

        pred_set = entry.pred_set

        definition = MACCS_DEFINITIONS.get(k) or FEATURE_VECTOR_DEFINITIONS.get(k)
        if not definition:
            raise ValueError(f"No predicate definition found for key {k}. Ensure that the key is defined in definitions.py.")

        # Add predicate definition
        rules.add(definition['definition'])

        pred = definition['predicate']
        domain = definition['domain']

        # If the prediction set is equal to the domain, then we don't need to add any constraints
        if len(pred_set) == len(domain):
            continue

        # BINARY CASE
        if domain == {0, 1}:
            if set(pred_set) == {1}:  # pred_set is actually a list so that it is ordered
                constraints.append(f":- not {pred}.")
            elif set(pred_set) == {0}:
                constraints.append(f":- {pred}.")
            else:
                raise ValueError(f"Invalid prediction set {set(pred_set)} for binary predicate {pred}.")
            continue

        # MULTI-CLASS CASE
        for v in pred_set:
            constraints.append(f"allowed_{pred}({v}).")

        # Enforce membership
        constraints.append(
            f":- {pred}(V), not allowed_{pred}(V)."
        )

    return list(rules) + constraints, atom_counts
