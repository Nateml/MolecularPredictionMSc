from typing import Dict, TypedDict

class ASPPredicate(TypedDict):
    """
    A constraint to be applied in the ASP program, represented as a string.
    """
    predicate: str  # The predicate name in the ASP program, e.g. has_carbonyl
    definition: str  # The actual predicate definition, e.g. "has_carbonyl :- ..."
    domain: set  # The set of possible values

# Definitions for helper predicates that are used by predicates defined in MACCS_DEFINITIONS and other predicates we may want to define. These are compiled into the ASP program by compiler.compile_asp_constraint.
HELPER_PREDICATES: Dict[str, str] = {
    "is_multi_bond": """
        is_multi_bond(J) :- multi_bond(J, _).
    """,

    "adjacent_single": """
        adjacent_single(I, J) :-
            parent(I, _, J),
            not is_multi_bond(J).
    """,

    "single_bond_between": """
        single_bond_between(I, J) :-
            adjacent_single(I, J).

        single_bond_between(I, J) :-
            adjacent_single(J, I).
    """,

    "double_bond_between": """
        double_bond_between(I, J) :-
            parent(I, _, J),
            multi_bond(J, 2).
        double_bond_between(I, J) :-
            parent(J, _, I),
            multi_bond(I, 2).
    """,

    "triple_bond_between": """
        triple_bond_between(I, J) :-
            parent(I, _, J),
            multi_bond(J, 3).
        triple_bond_between(I, J) :-
            parent(J, _, I),
            multi_bond(I, 3).
    """,

    "any_bond_between": """
        any_bond_between(I, J) :-
            parent(I, _, J).
        any_bond_between(I, J) :-
            parent(J, _, I).
    """,
}


# Define the MACCS keys and their corresponding ASP constraints
MACCS_DEFINITIONS: Dict[int, ASPPredicate] = {
    ########################
    ##### SINGLE BONDS #####
    ########################
    157: {
        "predicate": "has_c_o_single_bond",
        "definition": """
            has_c_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "O").
        """,
        "domain": {0, 1}
    },
    158: {
        "predicate": "has_c_n_single_bond",
        "definition": """
            has_c_n_single_bond :-
                single_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "N").
        """,
        "domain": {0, 1}
    },
    24: {
        "predicate": "has_n_o_single_bond",
        "definition": """
            has_n_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "N"),
                symbol(J, "O").
        """,
        "domain": {0, 1}
    },
    ########################
    ##### DOUBLE BONDS #####
    ########################
    99: {
        "predicate": "has_c_c_double_bond",
        "definition": """
            has_c_c_double_bond :-
                double_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "C").
        """,
        "domain": {0, 1}
    },
    78: {
        "predicate": "has_c_n_double_bond",
        "definition": """
            has_c_n_double_bond :-
                double_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "N").
        """,
        "domain": {0, 1}
    },
    63: {
        "predicate": "has_n_o_double_bond",
        "definition": """
            has_n_o_double_bond :-
                double_bond_between(I, J),
                symbol(I, "N"),
                symbol(J, "O").
        """,
        "domain": {0, 1}
    },
    154: {
        "predicate": "has_carbonyl",
        "definition": """
            has_carbonyl :-
                symbol(I, "C"),
                symbol(J, "O"),
                double_bond_between(I, J).
            """,
        "domain": {0, 1}
    },
    ########################
    ##### TRIPLE BONDS #####
    ########################
    41: {
        "predicate": "has_c_n_triple_bond",
        "definition": """
            has_c_n_triple_bond :-
                triple_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "N").
        """,
        "domain": {0, 1}
    },
    17: {
        "predicate": "has_c_c_triple_bond",
        "definition": """
            has_c_c_triple_bond :-
                triple_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "C").
        """,
        "domain": {0, 1}
    },
    #########################
    ####### ANY BONDS #######
    #########################
    71: {
        "predicate": "has_n_o_bond",
        "definition": """
            has_n_o_bond :-
                any_bond_between(I, J),
                symbol(I, "N"),
                symbol(J, "O").
        """,
        "domain": {0, 1}
    }
}

COUNT_KEYS = ['num_carbons', 'num_hydrogens', 'num_nitrogens', 'num_oxygens']

def KEYS():
    maccs_keys = set(MACCS_DEFINITIONS.keys())
    count_keys = set(COUNT_KEYS)

    assert maccs_keys.isdisjoint(count_keys), "MACCS keys and count keys must be disjoint"

    return maccs_keys.union(count_keys)