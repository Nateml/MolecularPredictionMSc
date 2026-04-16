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

    "alcohol_atom": """
        alcohol_atom(O) :-
            single_bond_between(C, O),
            symbol(C, "C"),
            symbol(O, "O"),
            free_bonds(O, 1).
    """,

    "alcohol_count": """
        alcohol_count(N) :-
            N = #count { O : alcohol_atom(O) }.
    """,

    "aldehyde_atom": """
        aldehyde_atom(C) :-
            symbol(C, "C"),
            symbol(O, "O"),
            symbol(C2, "C"),
            double_bond_between(C, O),
            single_bond_between(C, C2),
            free_bonds(C, 1).
    """,

    "aldehyde_count": """
        aldehyde_count(N) :-
            N = #count { C : aldehyde_atom(C) }.
    """,

    "aliphatic_ring_count": """
        aliphatic_ring_count(N) :-
            N = #count { CYCLE : cycle_start(_, CYCLE) }.
    """,

    "amide_atom": """
        amide_atom(N) :-
            symbol(N, "N"),
            symbol(C, "C"),
            symbol(O, "O"),
            single_bond_between(C, N),
            double_bond_between(C, O).
    """,

    "amide_count": """
        amide_count(N) :-
            N = #count { I : amide_atom(I) }.
    """,

    "amine_atom": """
        amine_atom(N) :-
            symbol(N, "N"),
            symbol(C, "C"),
            single_bond_between(N, C),
            not amide_atom(N).
    """,

    "amine_count": """
        amine_count(N) :-
            N = #count { I : amine_atom(I) }.
    """,

    "aniline_atom": """
        aniline_atom(N) :-
            amine_atom(N),
            symbol(C, "C"),
            single_bond_between(N, C),
            cycle_start(C, _).
        aniline_atom(N) :-
            amine_atom(N),
            symbol(C, "C"),
            single_bond_between(N, C),
            cycle_end(C, _).
    """,

    "aniline_count": """
        aniline_count(N) :-
            N = #count { I : aniline_atom(I) }.
    """,

    "aromatic_atom": """
        aromatic_atom(I) :-
            cycle_start(I, _).
        aromatic_atom(I) :-
            cycle_end(I, _).
        aromatic_atom(I) :-
            cycle_start(J, _),
            any_bond_between(J, I).
        aromatic_atom(I) :-
            cycle_end(J, _),
            any_bond_between(J, I).
    """,

    "aromatic_atom_count": """
        aromatic_atom_count(N) :-
            N = #count { I : aromatic_atom(I) }.
    """,

    "aromatic_heteroatom_count": """
        aromatic_heteroatom_count(N) :-
            N = #count { I : aromatic_atom(I), symbol(I, E), E != "C", E != "H" }.
    """,

    "aromatic_n_count": """
        aromatic_n_count(N) :-
            N = #count { I : aromatic_atom(I), symbol(I, "N") }.
    """,

    "aromatic_o_count": """
        aromatic_o_count(N) :-
            N = #count { I : aromatic_atom(I), symbol(I, "O") }.
    """,

    "aromatic_p_count": """
        aromatic_p_count(N) :-
            N = #count { I : aromatic_atom(I), symbol(I, "P") }.
    """,

    "aromatic_ring_count": """
        aromatic_ring_count(N) :-
            N = #count { CYCLE : cycle_start(_, CYCLE) }.
    """,

    "aromatic_ring": """
        aromatic_ring :-
            aromatic_ring_count(N),
            N > 0.
    """,

    "aromatic_s_count": """
        aromatic_s_count(N) :-
            N = #count { I : aromatic_atom(I), symbol(I, "S") }.
    """,

    "assigned_chiral_center_count": """
        assigned_chiral_center_count(0).
    """,

    "branch_point_atom": """
        branch_point_atom(I) :-
            branching(1, B),
            B >= 3,
            I = 1.
        branch_point_atom(I) :-
            branching(I, B),
            I > 1,
            B >= 2.
    """,

    "branch_point_count": """
        branch_point_count(N) :-
            N = #count { I : branch_point_atom(I) }.
    """,

    "bridgehead_atom_count": """
        bridgehead_atom_count(0).
    """,

    "carboxylic_acid_atom": """
        carboxylic_acid_atom(C) :-
            symbol(C, "C"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            O1 != O2,
            double_bond_between(C, O1),
            single_bond_between(C, O2),
            free_bonds(O2, 1).
    """,

    "carboxylic_acid_count": """
        carboxylic_acid_count(N) :-
            N = #count { C : carboxylic_acid_atom(C) }.
    """,

    "carbonyl_carbon": """
        carbonyl_carbon(C) :-
            symbol(C, "C"),
            symbol(O, "O"),
            double_bond_between(C, O).
    """,

    "ester_atom": """
        ester_atom(C) :-
            symbol(C, "C"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            O1 != O2,
            double_bond_between(C, O1),
            single_bond_between(C, O2),
            single_bond_between(O2, C2),
            symbol(C2, "C"),
            free_bonds(O2, 0).
    """,

    "ester_count": """
        ester_count(N) :-
            N = #count { C : ester_atom(C) }.
    """,

    "ether_atom": """
        ether_atom(O) :-
            symbol(O, "O"),
            single_bond_between(O, C1),
            single_bond_between(O, C2),
            C1 != C2,
            symbol(C1, "C"),
            symbol(C2, "C"),
            free_bonds(O, 0),
            not carbonyl_carbon(C1),
            not carbonyl_carbon(C2).
    """,

    "ether_count": """
        ether_count(N) :-
            N = #count { O : ether_atom(O) }.
    """,

    "chiral_center_count": """
        chiral_center_count(0).
    """,

    "connected_component_count": """
        connected_component_count(1).
    """,

    "heavy_neighbor": """
        heavy_neighbor(I, J) :-
            any_bond_between(I, J),
            I != J.
        heavy_neighbor(I, J) :-
            cycle_start(I, CYCLE),
            cycle_end(J, CYCLE),
            I != J.
        heavy_neighbor(I, J) :-
            cycle_end(I, CYCLE),
            cycle_start(J, CYCLE),
            I != J.
    """,

    "atom_degree": """
        atom_degree(I, D) :-
            symbol(I, _),
            D = #count { J : heavy_neighbor(I, J) }.
    """,

    "degree_1_atom_count": """
        degree_1_atom_count(N) :-
            N = #count { I : atom_degree(I, 1) }.
    """,

    "degree_2_atom_count": """
        degree_2_atom_count(N) :-
            N = #count { I : atom_degree(I, 2) }.
    """,

    "degree_3_atom_count": """
        degree_3_atom_count(N) :-
            N = #count { I : atom_degree(I, 3) }.
    """,

    "degree_4plus_atom_count": """
        degree_4plus_atom_count(N) :-
            N = #count { I : atom_degree(I, D), D >= 4 }.
    """,

    "double_bond_count": """
        double_bond_count(N) :-
            N = #count { J : multi_bond(J, 2) }.
    """,

    "triple_bond_count": """
        triple_bond_count(N) :-
            N = #count { J : multi_bond(J, 3) }.
    """,

    "ring_count": """
        ring_count(N) :-
            N = #count { CYCLE : cycle_start(_, CYCLE) }.
    """,

    "degree_of_unsaturation": """
        degree_of_unsaturation(N) :-
            double_bond_count(DB),
            triple_bond_count(TB),
            ring_count(R),
            N = DB + 2 * TB + R.
    """,

    "element_B_count": """
        element_B_count(0).
    """,

    "element_Br_count": """
        element_Br_count(0).
    """,

    "element_C_count": """
        element_C_count(N) :-
            N = #count { I : symbol(I, "C") }.
    """,

    "element_Cl_count": """
        element_Cl_count(0).
    """,

    "element_F_count": """
        element_F_count(0).
    """,

    "element_H_count": """
        element_H_count(N) :-
            molecular_formula("H", N).
        element_H_count(0) :-
            not molecular_formula("H", _).
    """,

    "element_I_count": """
        element_I_count(0).
    """,

    "element_N_count": """
        element_N_count(N) :-
            N = #count { I : symbol(I, "N") }.
    """,

    "element_O_count": """
        element_O_count(N) :-
            N = #count { I : symbol(I, "O") }.
    """,

    "element_P_count": """
        element_P_count(0).
    """,

    "element_S_count": """
        element_S_count(0).
    """,

    "element_Si_count": """
        element_Si_count(0).
    """,

##TODO: future elements need adding
    "exact_mass_da_scaled": """
        exact_mass_da_scaled(M) :-
            M = 12000000 * c + 1007825 * h + 1403074 * n + 1599491 * o.
    """,


    "heavy_atom_count": """
        heavy_atom_count(0).
    """,

    "heteroatom_count": """
        heteroatom_count(0).
    """,

    "heteroatom_exocyclic_count": """
        heteroatom_exocyclic_count(0).
    """,

    "heteroatom_in_ring_count": """
        heteroatom_in_ring_count(0).
    """,

    "formal_charge": """
        formal_charge(0).
    """,

    "fused_ring_pair_count": """
        fused_ring_pair_count(0).
    """,

    "fused_ring_system_count": """
        fused_ring_system_count(0).
    """,

    "imide": """
        imide(0).
    """,

    "ketone": """
        ketone(0).
    """,

    "lactam": """
        lactam(0).
    """,

    "lactone": """
        lactone(0).
    """,

    "halogenated": """
        halogenated :-
            element_F_count(N), N > 0.
        halogenated :-
            element_Cl_count(N), N > 0.
        halogenated :-
            element_Br_count(N), N > 0.
        halogenated :-
            element_I_count(N), N > 0.
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
    60: {
        "predicate": "has_s_o_double_bond",
        "definition": """
            has_s_o_double_bond :-
                double_bond_between(I, J),
                smiles_symbol(I, "S"),
                symbol(J, "O").
        """,
        "domain": {0, 1}
    },
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

# Feature-vector translations grounded in predicates emitted by
# eval-2024-asp-molecules/smiles.lp.
# Indices start at 167 (after MACCS keys 0-166)
FEATURE_VECTOR_DEFINITIONS: Dict[int, ASPPredicate] = {
    167: {
        "predicate": "alcohol_bin_0",
        "definition": """
            alcohol_bin_0 :-
                alcohol_count(0).
        """,
        "domain": {0, 1}
    },
    168: {
        "predicate": "alcohol_bin_1",
        "definition": """
            alcohol_bin_1 :-
                alcohol_count(1).
        """,
        "domain": {0, 1}
    },
    169: {
        "predicate": "alcohol_bin_2",
        "definition": """
            alcohol_bin_2 :-
                alcohol_count(N),
                N > 1,
                N <= 4.
        """,
        "domain": {0, 1}
    },
    170: {
        "predicate": "alcohol_bin_3",
        "definition": """
            alcohol_bin_3 :-
                alcohol_count(N),
                N > 4.
        """,
        "domain": {0, 1}
    },
    171: {
        "predicate": "aldehyde_bin_0",
        "definition": """
            aldehyde_bin_0 :-
                aldehyde_count(0).
        """,
        "domain": {0, 1}
    },
    172: {
        "predicate": "aldehyde_bin_1",
        "definition": """
            aldehyde_bin_1 :-
                aldehyde_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    173: {
        "predicate": "aliphatic_ring_count_bin_0",
        "definition": """
            aliphatic_ring_count_bin_0 :-
                aliphatic_ring_count(0).
        """,
        "domain": {0, 1}
    },
    174: {
        "predicate": "aliphatic_ring_count_bin_1",
        "definition": """
            aliphatic_ring_count_bin_1 :-
                aliphatic_ring_count(1).
        """,
        "domain": {0, 1}
    },
    175: {
        "predicate": "aliphatic_ring_count_bin_2",
        "definition": """
            aliphatic_ring_count_bin_2 :-
                aliphatic_ring_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    176: {
        "predicate": "amide_bin_0",
        "definition": """
            amide_bin_0 :-
                amide_count(0).
        """,
        "domain": {0, 1}
    },
    177: {
        "predicate": "amide_bin_1",
        "definition": """
            amide_bin_1 :-
                amide_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    178: {
        "predicate": "amine_bin_0",
        "definition": """
            amine_bin_0 :-
                amine_count(0).
        """,
        "domain": {0, 1}
    },
    179: {
        "predicate": "amine_bin_1",
        "definition": """
            amine_bin_1 :-
                amine_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    180: {
        "predicate": "aniline_bin_0",
        "definition": """
            aniline_bin_0 :-
                aniline_count(0).
        """,
        "domain": {0, 1}
    },
    181: {
        "predicate": "aniline_bin_1",
        "definition": """
            aniline_bin_1 :-
                aniline_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    182: {
        "predicate": "aromatic_atom_count_bin_0",
        "definition": """
            aromatic_atom_count_bin_0 :-
                aromatic_atom_count(0).
        """,
        "domain": {0, 1}
    },
    183: {
        "predicate": "aromatic_atom_count_bin_1",
        "definition": """
            aromatic_atom_count_bin_1 :-
                aromatic_atom_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    184: {
        "predicate": "aromatic_heteroatom_count_bin_0",
        "definition": """
            aromatic_heteroatom_count_bin_0 :-
                aromatic_heteroatom_count(0).
        """,
        "domain": {0, 1}
    },
    185: {
        "predicate": "aromatic_heteroatom_count_bin_1",
        "definition": """
            aromatic_heteroatom_count_bin_1 :-
                aromatic_heteroatom_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    186: {
        "predicate": "aromatic_n_count_bin_0",
        "definition": """
            aromatic_n_count_bin_0 :-
                aromatic_n_count(0).
        """,
        "domain": {0, 1}
    },
    187: {
        "predicate": "aromatic_n_count_bin_1",
        "definition": """
            aromatic_n_count_bin_1 :-
                aromatic_n_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    188: {
        "predicate": "aromatic_o_count_bin_0",
        "definition": """
            aromatic_o_count_bin_0 :-
                aromatic_o_count(0).
        """,
        "domain": {0, 1}
    },
    189: {
        "predicate": "aromatic_o_count_bin_1",
        "definition": """
            aromatic_o_count_bin_1 :-
                aromatic_o_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    190: {
        "predicate": "aromatic_p_count_bin_0",
        "definition": """
            aromatic_p_count_bin_0 :-
                aromatic_p_count(0).
        """,
        "domain": {0, 1}
    },
    191: {
        "predicate": "aromatic_p_count_bin_1",
        "definition": """
            aromatic_p_count_bin_1 :-
                aromatic_p_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    192: {
        "predicate": "aromatic_ring_bin_0",
        "definition": """
            aromatic_ring_bin_0 :-
                not aromatic_ring.
        """,
        "domain": {0, 1}
    },
    193: {
        "predicate": "aromatic_ring_bin_1",
        "definition": """
            aromatic_ring_bin_1 :-
                aromatic_ring.
        """,
        "domain": {0, 1}
    },
    194: {
        "predicate": "aromatic_ring_count_bin_0",
        "definition": """
            aromatic_ring_count_bin_0 :-
                aromatic_ring_count(0).
        """,
        "domain": {0, 1}
    },
    195: {
        "predicate": "aromatic_ring_count_bin_1",
        "definition": """
            aromatic_ring_count_bin_1 :-
                aromatic_ring_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    196: {
        "predicate": "aromatic_s_count_bin_0",
        "definition": """
            aromatic_s_count_bin_0 :-
                aromatic_s_count(0).
        """,
        "domain": {0, 1}
    },
    197: {
        "predicate": "aromatic_s_count_bin_1",
        "definition": """
            aromatic_s_count_bin_1 :-
                aromatic_s_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    198: {
        "predicate": "assigned_chiral_center_count_bin_0",
        "definition": """
            assigned_chiral_center_count_bin_0 :-
                assigned_chiral_center_count(0).
        """,
        "domain": {0, 1}
    },
    199: {
        "predicate": "assigned_chiral_center_count_bin_1",
        "definition": """
            assigned_chiral_center_count_bin_1 :-
                assigned_chiral_center_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    200: {
        "predicate": "branch_point_count_bin_0",
        "definition": """
            branch_point_count_bin_0 :-
                branch_point_count(N),
                N <= 4.
        """,
        "domain": {0, 1}
    },
    201: {
        "predicate": "branch_point_count_bin_1",
        "definition": """
            branch_point_count_bin_1 :-
                branch_point_count(N),
                N > 4,
                N <= 5.
        """,
        "domain": {0, 1}
    },
    202: {
        "predicate": "branch_point_count_bin_2",
        "definition": """
            branch_point_count_bin_2 :-
                branch_point_count(N),
                N > 5,
                N <= 6.
        """,
        "domain": {0, 1}
    },
    203: {
        "predicate": "branch_point_count_bin_3",
        "definition": """
            branch_point_count_bin_3 :-
                branch_point_count(N),
                N > 6,
                N <= 8.
        """,
        "domain": {0, 1}
    },
    204: {
        "predicate": "branch_point_count_bin_4",
        "definition": """
            branch_point_count_bin_4 :-
                branch_point_count(N),
                N > 8.
        """,
        "domain": {0, 1}
    },
    205: {
        "predicate": "bridgehead_atom_count_bin_0",
        "definition": """
            bridgehead_atom_count_bin_0 :-
                bridgehead_atom_count(0).
        """,
        "domain": {0, 1}
    },
    206: {
        "predicate": "bridgehead_atom_count_bin_1",
        "definition": """
            bridgehead_atom_count_bin_1 :-
                bridgehead_atom_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    207: {
        "predicate": "carboxylic_acid_bin_0",
        "definition": """
            carboxylic_acid_bin_0 :-
                carboxylic_acid_count(0).
        """,
        "domain": {0, 1}
    },
    208: {
        "predicate": "carboxylic_acid_bin_1",
        "definition": """
            carboxylic_acid_bin_1 :-
                carboxylic_acid_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    209: {
        "predicate": "chiral_center_count_bin_0",
        "definition": """
            chiral_center_count_bin_0 :-
                chiral_center_count(N),
                N <= 2.
        """,
        "domain": {0, 1}
    },
    210: {
        "predicate": "chiral_center_count_bin_1",
        "definition": """
            chiral_center_count_bin_1 :-
                chiral_center_count(N),
                N > 2,
                N <= 3.
        """,
        "domain": {0, 1}
    },
    211: {
        "predicate": "chiral_center_count_bin_2",
        "definition": """
            chiral_center_count_bin_2 :-
                chiral_center_count(N),
                N > 3,
                N <= 6.
        """,
        "domain": {0, 1}
    },
    212: {
        "predicate": "chiral_center_count_bin_3",
        "definition": """
            chiral_center_count_bin_3 :-
                chiral_center_count(N),
                N > 6.
        """,
        "domain": {0, 1}
    },
    213: {
        "predicate": "connected_component_count_bin_0",
        "definition": """
            connected_component_count_bin_0 :-
                connected_component_count(N),
                N <= 1.
        """,
        "domain": {0, 1}
    },
    214: {
        "predicate": "connected_component_count_bin_1",
        "definition": """
            connected_component_count_bin_1 :-
                connected_component_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    215: {
        "predicate": "degree_1_atom_count_bin_0",
        "definition": """
            degree_1_atom_count_bin_0 :-
                degree_1_atom_count(N),
                N <= 7.
        """,
        "domain": {0, 1}
    },
    216: {
        "predicate": "degree_1_atom_count_bin_1",
        "definition": """
            degree_1_atom_count_bin_1 :-
                degree_1_atom_count(N),
                N > 7,
                N <= 8.
        """,
        "domain": {0, 1}
    },
    217: {
        "predicate": "degree_1_atom_count_bin_2",
        "definition": """
            degree_1_atom_count_bin_2 :-
                degree_1_atom_count(N),
                N > 8,
                N <= 9.
        """,
        "domain": {0, 1}
    },
    218: {
        "predicate": "degree_1_atom_count_bin_3",
        "definition": """
            degree_1_atom_count_bin_3 :-
                degree_1_atom_count(N),
                N > 9.
        """,
        "domain": {0, 1}
    },
    219: {
        "predicate": "degree_2_atom_count_bin_0",
        "definition": """
            degree_2_atom_count_bin_0 :-
                degree_2_atom_count(N),
                N <= 31.
        """,
        "domain": {0, 1}
    },
    220: {
        "predicate": "degree_2_atom_count_bin_1",
        "definition": """
            degree_2_atom_count_bin_1 :-
                degree_2_atom_count(N),
                N > 31,
                N <= 39.
        """,
        "domain": {0, 1}
    },
    221: {
        "predicate": "degree_2_atom_count_bin_2",
        "definition": """
            degree_2_atom_count_bin_2 :-
                degree_2_atom_count(N),
                N > 39,
                N <= 45.
        """,
        "domain": {0, 1}
    },
    222: {
        "predicate": "degree_2_atom_count_bin_3",
        "definition": """
            degree_2_atom_count_bin_3 :-
                degree_2_atom_count(N),
                N > 45,
                N <= 55.
        """,
        "domain": {0, 1}
    },
    223: {
        "predicate": "degree_2_atom_count_bin_4",
        "definition": """
            degree_2_atom_count_bin_4 :-
                degree_2_atom_count(N),
                N > 55,
                N <= 63.
        """,
        "domain": {0, 1}
    },
    224: {
        "predicate": "degree_2_atom_count_bin_5",
        "definition": """
            degree_2_atom_count_bin_5 :-
                degree_2_atom_count(N),
                N > 63.
        """,
        "domain": {0, 1}
    },
    225: {
        "predicate": "degree_3_atom_count_bin_0",
        "definition": """
            degree_3_atom_count_bin_0 :-
                degree_3_atom_count(N),
                N <= 3.
        """,
        "domain": {0, 1}
    },
    226: {
        "predicate": "degree_3_atom_count_bin_1",
        "definition": """
            degree_3_atom_count_bin_1 :-
                degree_3_atom_count(N),
                N > 3,
                N <= 5.
        """,
        "domain": {0, 1}
    },
    227: {
        "predicate": "degree_3_atom_count_bin_2",
        "definition": """
            degree_3_atom_count_bin_2 :-
                degree_3_atom_count(N),
                N > 5,
                N <= 8.
        """,
        "domain": {0, 1}
    },
    228: {
        "predicate": "degree_3_atom_count_bin_3",
        "definition": """
            degree_3_atom_count_bin_3 :-
                degree_3_atom_count(N),
                N > 8.
        """,
        "domain": {0, 1}
    },
    229: {
        "predicate": "degree_4plus_atom_count_bin_0",
        "definition": """
            degree_4plus_atom_count_bin_0 :-
                degree_4plus_atom_count(0).
        """,
        "domain": {0, 1}
    },
    230: {
        "predicate": "degree_4plus_atom_count_bin_1",
        "definition": """
            degree_4plus_atom_count_bin_1 :-
                degree_4plus_atom_count(1).
        """,
        "domain": {0, 1}
    },
    231: {
        "predicate": "degree_4plus_atom_count_bin_2",
        "definition": """
            degree_4plus_atom_count_bin_2 :-
                degree_4plus_atom_count(2).
        """,
        "domain": {0, 1}
    },
    232: {
        "predicate": "degree_4plus_atom_count_bin_3",
        "definition": """
            degree_4plus_atom_count_bin_3 :-
                degree_4plus_atom_count(N),
                N > 2.
        """,
        "domain": {0, 1}
    },
    233: {
        "predicate": "degree_of_unsaturation_bin_0",
        "definition": """
            degree_of_unsaturation_bin_0 :-
                degree_of_unsaturation(N),
                N <= 4.
        """,
        "domain": {0, 1}
    },
    234: {
        "predicate": "degree_of_unsaturation_bin_1",
        "definition": """
            degree_of_unsaturation_bin_1 :-
                degree_of_unsaturation(N),
                N > 4,
                N <= 5.
        """,
        "domain": {0, 1}
    },
    235: {
        "predicate": "degree_of_unsaturation_bin_2",
        "definition": """
            degree_of_unsaturation_bin_2 :-
                degree_of_unsaturation(N),
                N > 5,
                N <= 7.
        """,
        "domain": {0, 1}
    },
    236: {
        "predicate": "degree_of_unsaturation_bin_3",
        "definition": """
            degree_of_unsaturation_bin_3 :-
                degree_of_unsaturation(N),
                N > 7,
                N <= 9.
        """,
        "domain": {0, 1}
    },
    237: {
        "predicate": "degree_of_unsaturation_bin_4",
        "definition": """
            degree_of_unsaturation_bin_4 :-
                degree_of_unsaturation(N),
                N > 9,
                N <= 13.
        """,
        "domain": {0, 1}
    },
    238: {
        "predicate": "degree_of_unsaturation_bin_5",
        "definition": """
            degree_of_unsaturation_bin_5 :-
                degree_of_unsaturation(N),
                N > 13.
        """,
        "domain": {0, 1}
    },
    239: {
        "predicate": "element_B_bin_0",
        "definition": """
            element_B_bin_0 :-
                element_B_count(0).
        """,
        "domain": {0, 1}
    },
    240: {
        "predicate": "element_B_bin_1",
        "definition": """
            element_B_bin_1 :-
                element_B_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    241: {
        "predicate": "element_Br_bin_0",
        "definition": """
            element_Br_bin_0 :-
                element_Br_count(0).
        """,
        "domain": {0, 1}
    },
    242: {
        "predicate": "element_Br_bin_1",
        "definition": """
            element_Br_bin_1 :-
                element_Br_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    243: {
        "predicate": "element_C_bin_0",
        "definition": """
            element_C_bin_0 :-
                element_C_count(N),
                N <= 35.
        """,
        "domain": {0, 1}
    },
    244: {
        "predicate": "element_C_bin_1",
        "definition": """
            element_C_bin_1 :-
                element_C_count(N),
                N > 35,
                N <= 42.
        """,
        "domain": {0, 1}
    },
    245: {
        "predicate": "element_C_bin_2",
        "definition": """
            element_C_bin_2 :-
                element_C_count(N),
                N > 42,
                N <= 49.
        """,
        "domain": {0, 1}
    },
    246: {
        "predicate": "element_C_bin_3",
        "definition": """
            element_C_bin_3 :-
                element_C_count(N),
                N > 49,
                N <= 60.
        """,
        "domain": {0, 1}
    },
    247: {
        "predicate": "element_C_bin_4",
        "definition": """
            element_C_bin_4 :-
                element_C_count(N),
                N > 60,
                N <= 66.
        """,
        "domain": {0, 1}
    },
    248: {
        "predicate": "element_C_bin_5",
        "definition": """
            element_C_bin_5 :-
                element_C_count(N),
                N > 66.
        """,
        "domain": {0, 1}
    },
    249: {
        "predicate": "element_Cl_bin_0",
        "definition": """
            element_Cl_bin_0 :-
                element_Cl_count(0).
        """,
        "domain": {0, 1}
    },
    250: {
        "predicate": "element_Cl_bin_1",
        "definition": """
            element_Cl_bin_1 :-
                element_Cl_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    251: {
        "predicate": "element_F_bin_0",
        "definition": """
            element_F_bin_0 :-
                element_F_count(0).
        """,
        "domain": {0, 1}
    },
    252: {
        "predicate": "element_F_bin_1",
        "definition": """
            element_F_bin_1 :-
                element_F_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    253: {
        "predicate": "element_H_bin_0",
        "definition": """
            element_H_bin_0 :-
                element_H_count(N),
                N <= 64.
        """,
        "domain": {0, 1}
    },
    254: {
        "predicate": "element_H_bin_1",
        "definition": """
            element_H_bin_1 :-
                element_H_count(N),
                N > 64,
                N <= 77.
        """,
        "domain": {0, 1}
    },
    255: {
        "predicate": "element_H_bin_2",
        "definition": """
            element_H_bin_2 :-
                element_H_count(N),
                N > 77,
                N <= 88.
        """,
        "domain": {0, 1}
    },
    256: {
        "predicate": "element_H_bin_3",
        "definition": """
            element_H_bin_3 :-
                element_H_count(N),
                N > 88,
                N <= 102.
        """,
        "domain": {0, 1}
    },
    257: {
        "predicate": "element_H_bin_4",
        "definition": """
            element_H_bin_4 :-
                element_H_count(N),
                N > 102,
                N <= 113.
        """,
        "domain": {0, 1}
    },
    258: {
        "predicate": "element_H_bin_5",
        "definition": """
            element_H_bin_5 :-
                element_H_count(N),
                N > 113.
        """,
        "domain": {0, 1}
    },
    259: {
        "predicate": "element_I_bin_0",
        "definition": """
            element_I_bin_0 :-
                element_I_count(0).
        """,
        "domain": {0, 1}
    },
    260: {
        "predicate": "element_I_bin_1",
        "definition": """
            element_I_bin_1 :-
                element_I_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    261: {
        "predicate": "element_N_bin_0",
        "definition": """
            element_N_bin_0 :-
                element_N_count(0).
        """,
        "domain": {0, 1}
    },
    262: {
        "predicate": "element_N_bin_1",
        "definition": """
            element_N_bin_1 :-
                element_N_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    263: {
        "predicate": "element_N_bin_2",
        "definition": """
            element_N_bin_2 :-
                element_N_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    264: {
        "predicate": "element_O_bin_0",
        "definition": """
            element_O_bin_0 :-
                element_O_count(N),
                N <= 8.
        """,
        "domain": {0, 1}
    },
    265: {
        "predicate": "element_O_bin_1",
        "definition": """
            element_O_bin_1 :-
                element_O_count(N),
                N > 8,
                N <= 10.
        """,
        "domain": {0, 1}
    },
    266: {
        "predicate": "element_O_bin_2",
        "definition": """
            element_O_bin_2 :-
                element_O_count(N),
                N > 10,
                N <= 11.
        """,
        "domain": {0, 1}
    },
    267: {
        "predicate": "element_O_bin_3",
        "definition": """
            element_O_bin_3 :-
                element_O_count(N),
                N > 11.
        """,
        "domain": {0, 1}
    },
    268: {
        "predicate": "element_P_bin_0",
        "definition": """
            element_P_bin_0 :-
                element_P_count(0).
        """,
        "domain": {0, 1}
    },
    269: {
        "predicate": "element_P_bin_1",
        "definition": """
            element_P_bin_1 :-
                element_P_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    270: {
        "predicate": "element_P_bin_2",
        "definition": """
            element_P_bin_2 :-
                element_P_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    271: {
        "predicate": "element_S_bin_0",
        "definition": """
            element_S_bin_0 :-
                element_S_count(0).
        """,
        "domain": {0, 1}
    },
    272: {
        "predicate": "element_S_bin_1",
        "definition": """
            element_S_bin_1 :-
                element_S_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    273: {
        "predicate": "element_Si_bin_0",
        "definition": """
            element_Si_bin_0 :-
                element_Si_count(0).
        """,
        "domain": {0, 1}
    },
    274: {
        "predicate": "element_Si_bin_1",
        "definition": """
            element_Si_bin_1 :-
                element_Si_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    275: {
        "predicate": "ester_bin_0",
        "definition": """
            ester_bin_0 :-
                ester_count(N),
                N <= 2.
        """,
        "domain": {0, 1}
    },
    276: {
        "predicate": "ester_bin_1",
        "definition": """
            ester_bin_1 :-
                ester_count(N),
                N > 2,
                N <= 3.
        """,
        "domain": {0, 1}
    },
    277: {
        "predicate": "ester_bin_2",
        "definition": """
            ester_bin_2 :-
                ester_count(N),
                N > 3.
        """,
        "domain": {0, 1}
    },
    278: {
        "predicate": "ether_bin_0",
        "definition": """
            ether_bin_0 :-
                ether_count(N),
                N <= 2.
        """,
        "domain": {0, 1}
    },
    279: {
        "predicate": "ether_bin_1",
        "definition": """
            ether_bin_1 :-
                ether_count(N),
                N > 2,
                N <= 3.
        """,
        "domain": {0, 1}
    },
    280: {
        "predicate": "ether_bin_2",
        "definition": """
            ether_bin_2 :-
                ether_count(N),
                N > 3,
                N <= 4.
        """,
        "domain": {0, 1}
    },
    281: {
        "predicate": "ether_bin_3",
        "definition": """
            ether_bin_3 :-
                ether_count(N),
                N > 4.
        """,
        "domain": {0, 1}
    },
    282: {
        "predicate": "exact_mass_da_bin_0",
        "definition": """
            exact_mass_da_bin_0 :-
                exact_mass_da_scaled(M),
                M <= 651580000.
        """,
        "domain": {0, 1}
    },
    283: {
        "predicate": "exact_mass_da_bin_1",
        "definition": """
            exact_mass_da_bin_1 :-
                exact_mass_da_scaled(M),
                M > 651580000,
                M <= 751515000.
        """,
        "domain": {0, 1}
    },
    284: {
        "predicate": "exact_mass_da_bin_2",
        "definition": """
            exact_mass_da_bin_2 :-
                exact_mass_da_scaled(M),
                M > 751515000,
                M <= 834614000.
        """,
        "domain": {0, 1}
    },
    285: {
        "predicate": "exact_mass_da_bin_3",
        "definition": """
            exact_mass_da_bin_3 :-
                exact_mass_da_scaled(M),
                M > 834614000,
                M <= 1006720000.
        """,
        "domain": {0, 1}
    },
    286: {
        "predicate": "exact_mass_da_bin_4",
        "definition": """
            exact_mass_da_bin_4 :-
                exact_mass_da_scaled(M),
                M > 1006720000,
                M <= 1112800000.
        """,
        "domain": {0, 1}
    },
    287: {
        "predicate": "exact_mass_da_bin_5",
        "definition": """
            exact_mass_da_bin_5 :-
                exact_mass_da_scaled(M),
                M > 1112800000.
        """,
        "domain": {0, 1}
    },
    288: {
        "predicate": "formal_charge_bin_0",
        "definition": """
            formal_charge_bin_0 :-
                formal_charge(0).
        """,
        "domain": {0, 1}
    },
    289: {
        "predicate": "formal_charge_bin_1",
        "definition": """
            formal_charge_bin_1 :-
                formal_charge(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    290: {
        "predicate": "fused_ring_pair_count_bin_0",
        "definition": """
            fused_ring_pair_count_bin_0 :-
                fused_ring_pair_count(0).
        """,
        "domain": {0, 1}
    },
    291: {
        "predicate": "fused_ring_pair_count_bin_1",
        "definition": """
            fused_ring_pair_count_bin_1 :-
                fused_ring_pair_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    292: {
        "predicate": "fused_ring_system_count_bin_0",
        "definition": """
            fused_ring_system_count_bin_0 :-
                fused_ring_system_count(0).
        """,
        "domain": {0, 1}
    },
    293: {
        "predicate": "fused_ring_system_count_bin_1",
        "definition": """
            fused_ring_system_count_bin_1 :-
                fused_ring_system_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    294: {
        "predicate": "halogenated_bin_0",
        "definition": """
            halogenated_bin_0 :-
                not halogenated.
        """,
        "domain": {0, 1}
    },
    295: {
        "predicate": "halogenated_bin_1",
        "definition": """
            halogenated_bin_1 :-
                halogenated.
        """,
        "domain": {0, 1}
    },

    ##TODO: currently use placeholder helper method to define heavy_atom_count
    296: {
        "predicate": "heavy_atom_count_bin_0",
        "definition": """
            heavy_atom_count_bin_0 :-
                heavy_atom_count(N),
                N <= 45.
        """,
        "domain": {0, 1}
    },
    297: {
        "predicate": "heavy_atom_count_bin_1",
        "definition": """
            heavy_atom_count_bin_1 :-
                heavy_atom_count(N),
                N > 45,
                N <= 52.
        """,
        "domain": {0, 1}
    },
    298: {
        "predicate": "heavy_atom_count_bin_2",
        "definition": """
            heavy_atom_count_bin_2 :-
                heavy_atom_count(N),
                N > 52,
                N <= 58.
        """,
        "domain": {0, 1}
    },
    299: {
        "predicate": "heavy_atom_count_bin_3",
        "definition": """
            heavy_atom_count_bin_3 :-
                heavy_atom_count(N),
                N > 58,
                N <= 70.
        """,
        "domain": {0, 1}
    },
    300: {
        "predicate": "heavy_atom_count_bin_4",
        "definition": """
            heavy_atom_count_bin_4 :-
                heavy_atom_count(N),
                N > 70,
                N <= 78.
        """,
        "domain": {0, 1}
    },
    301: {
        "predicate": "heavy_atom_count_bin_5",
        "definition": """
            heavy_atom_count_bin_5 :-
                heavy_atom_count(N),
                N > 78.
        """,
        "domain": {0, 1}
    },
    302: {
        "predicate": "heteroatom_count_bin_0",
        "definition": """
            heteroatom_count_bin_0 :-
                heteroatom_count(N),
                N <= 9.
        """,
        "domain": {0, 1}
    },
    303: {
        "predicate": "heteroatom_count_bin_1",
        "definition": """
            heteroatom_count_bin_1 :-
                heteroatom_count(N),
                N > 9,
                N <= 10.
        """,
        "domain": {0, 1}
    },
    304: {
        "predicate": "heteroatom_count_bin_2",
        "definition": """
            heteroatom_count_bin_2 :-
                heteroatom_count(N),
                N > 10,
                N <= 12.
        """,
        "domain": {0, 1}
    },
    305: {
        "predicate": "heteroatom_count_bin_3",
        "definition": """
            heteroatom_count_bin_3 :-
                heteroatom_count(N),
                N > 12.
        """,
        "domain": {0, 1}
    },
    306: {
        "predicate": "heteroatom_exocyclic_count_bin_0",
        "definition": """
            heteroatom_exocyclic_count_bin_0 :-
                heteroatom_exocyclic_count(N),
                N <= 9.
        """,
        "domain": {0, 1}
    },
    307: {
        "predicate": "heteroatom_exocyclic_count_bin_1",
        "definition": """
            heteroatom_exocyclic_count_bin_1 :-
                heteroatom_exocyclic_count(N),
                N > 9,
                N <= 10.
        """,
        "domain": {0, 1}
    },
    308: {
        "predicate": "heteroatom_exocyclic_count_bin_2",
        "definition": """
            heteroatom_exocyclic_count_bin_2 :-
                heteroatom_exocyclic_count(N),
                N > 10,
                N <= 12.
        """,
        "domain": {0, 1}
    },
    309: {
        "predicate": "heteroatom_exocyclic_count_bin_3",
        "definition": """
            heteroatom_exocyclic_count_bin_3 :-
                heteroatom_exocyclic_count(N),
                N > 12.
        """,
        "domain": {0, 1}
    },
    310: {
        "predicate": "heteroatom_in_ring_count_bin_0",
        "definition": """
            heteroatom_in_ring_count_bin_0 :-
                heteroatom_in_ring_count(0).
        """,
        "domain": {0, 1}
    },
    311: {
        "predicate": "heteroatom_in_ring_count_bin_1",
        "definition": """
            heteroatom_in_ring_count_bin_1 :-
                heteroatom_in_ring_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    312: {
        "predicate": "heteroatom_in_ring_count_bin_2",
        "definition": """
            heteroatom_in_ring_count_bin_2 :-
                heteroatom_in_ring_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    313: {
        "predicate": "imide_bin_0",
        "definition": """
            imide_bin_0 :-
                imide(0).
        """,
        "domain": {0, 1}
    },
    314: {
        "predicate": "imide_bin_1",
        "definition": """
            imide_bin_1 :-
                imide(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    315: {
        "predicate": "ketone_bin_0",
        "definition": """
            ketone_bin_0 :-
                ketone(0).
        """,
        "domain": {0, 1}
    },
    316: {
        "predicate": "ketone_bin_1",
        "definition": """
            ketone_bin_1 :-
                ketone(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    317: {
        "predicate": "lactam_bin_0",
        "definition": """
            lactam_bin_0 :-
                lactam(0).
        """,
        "domain": {0, 1}
    },
    318: {
        "predicate": "lactam_bin_1",
        "definition": """
            lactam_bin_1 :-
                lactam(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    319: {
        "predicate": "lactone_bin_0",
        "definition": """
            lactone_bin_0 :-
                lactone(0).
        """,
        "domain": {0, 1}
    },
    320: {
        "predicate": "lactone_bin_1",
        "definition": """
            lactone_bin_1 :-
                lactone(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    
}

COUNT_KEYS = ['num_carbons', 'num_hydrogens', 'num_nitrogens', 'num_oxygens', 'num_sulfurs', 'num_phosphorus']

def KEYS():
    maccs_keys = set(MACCS_DEFINITIONS.keys())
    feature_vector_keys = set(FEATURE_VECTOR_DEFINITIONS.keys())
    count_keys = set(COUNT_KEYS)

    assert maccs_keys.isdisjoint(feature_vector_keys), "MACCS keys and feature vector keys must be disjoint"
    assert maccs_keys.isdisjoint(count_keys), "MACCS keys and count keys must be disjoint"
    assert feature_vector_keys.isdisjoint(count_keys), "Feature vector keys and count keys must be disjoint"

    return maccs_keys.union(feature_vector_keys).union(count_keys)