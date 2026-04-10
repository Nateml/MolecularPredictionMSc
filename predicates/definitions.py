from typing import Dict, TypedDict

class ASPPredicate(TypedDict):
    """
    A constraint to be applied in the ASP program, represented as a string.
    """
    predicate: str  # The predicate name in the ASP program, e.g. has_carbonyl
    definition: str  # The actual predicate definition, e.g. "has_carbonyl :- ..."
    domain: set  # The set of possible values

# Define the MACCS keys and their corresponding ASP constraints
MACCS_DEFINITIONS: Dict[int, ASPPredicate] = {
    154: {
        "predicate": "has_carbonyl",
        "definition": """
            has_carbonyl :-
                symbol(I, "C"),
                symbol(J, "O"),
                parent(I, _, J),
                multi_bond(J, 2).
            """,
        "domain": {0, 1}
    },
}

COUNT_KEYS = ['num_carbons', 'num_hydrogens', 'num_nitrogens', 'num_oxygens']

def KEYS():
    maccs_keys = set(MACCS_DEFINITIONS.keys())
    count_keys = set(COUNT_KEYS)

    assert maccs_keys.isdisjoint(count_keys), "MACCS keys and count keys must be disjoint"

    return maccs_keys.union(count_keys)