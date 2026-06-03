from typing import Dict, TypedDict, Set
from rdkit import Chem


class ASPPredicate(TypedDict):
    """
    A constraint to be applied in the ASP program, represented as a string.
    """
    predicate: str  # The predicate name in the ASP program, e.g. has_carbonyl
    definition: str  # The actual predicate definition, e.g. "has_carbonyl :- ..."
    domain: Set[int]  # The set of possible values


class ASPContinuousPredicate(TypedDict):
    predicate: str  # helper predicate name, e.g. "ring_count" or "alcohol_count"
    definition: str  # the helper predicate definition (ASP rule)

# Definitions for helper predicates that are used by predicates defined in MACCS_DEFINITIONS and other predicates we may want to define. These are compiled into the ASP program by compiler.compile_asp_constraint.
HELPER_PREDICATES: Dict[str, str] = {
    "is_multi_bond": """
        is_multi_bond(J) :- multi_bond(J, _).
    """,

    "ring_atom_with_dbl": """
        % Two parallel ring closures between the same atom pair encode a double bond.
        ring_closure_dbl(S, E) :-
            cycle_start(S, CY),
            cycle_end(E, CY),
            cycle_start(S, CY2),
            cycle_end(E, CY2),
            CY != CY2.

        % True when atom I is in any ring that contains at least one double bond,
        % whether from a backbone multi_bond or a ring-closure double bond.
        % Checks any ring containing I so that fused-ring aromaticity is captured
        % even when the double bond is in an adjacent ring.
        ring_atom_with_dbl(I) :-
            atom_in_ring(I, CY),
            atom_in_ring(K, CY),
            multi_bond(K, _).
        ring_atom_with_dbl(I) :-
            atom_in_ring(I, CY),
            atom_in_ring(S, CY),
            ring_closure_dbl(S, _).
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

        % Ring closure bonds not encoded as double (parallel pair) are single.
        % ring_closure_dbl is defined in the ring_atom_with_dbl helper (always loaded).
        single_bond_between(I, J) :-
            cycle_start(I, CY),
            cycle_end(J, CY),
            not ring_closure_dbl(I, J).

        single_bond_between(J, I) :-
            cycle_start(I, CY),
            cycle_end(J, CY),
            not ring_closure_dbl(I, J).
    """,

    "double_bond_between": """
        double_bond_between(I, J) :-
            parent(I, _, J),
            multi_bond(J, 2).
        double_bond_between(I, J) :-
            parent(J, _, I),
            multi_bond(I, 2).
        % Ring closure double bonds (encoded as two parallel ring labels).
        % ring_closure_dbl is defined in the ring_atom_with_dbl helper (always loaded).
        double_bond_between(I, J) :-
            ring_closure_dbl(I, J).
        double_bond_between(J, I) :-
            ring_closure_dbl(I, J).
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
        % ring-closing bond (cycle_end → cycle_start) is not in parent relation
        any_bond_between(I, J) :-
            cycle_start(I, CYCLE),
            cycle_end(J, CYCLE).
        any_bond_between(I, J) :-
            cycle_end(I, CYCLE),
            cycle_start(J, CYCLE).
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

    # Count in-ring double bonds for cycle C.
    # multi_bond(I, 2) encodes "bond from atom I-1 to I is double" on the main chain.
    # The ring-closing bond (cycle_end → cycle_start) is not in multi_bond; we skip it
    # by only counting I in (S, E] (S itself would be outside the ring).
    "ring_double_bond_count": """
        ring_double_bond_count(C, N) :-
            cycle_start(_, C),
            N = #count { I : atom_in_ring(I, C), multi_bond(I, 2) }.
    """,

    # True when ring C has two backbone-adjacent atoms both with multi_bond (cumulene).
    # Aromatic rings have alternating single/double bonds; cumulenes have consecutive ones.
    "consecutive_ring_doubles": """
        consecutive_ring_doubles(C) :-
            cycle_start(S, C),
            cycle_end(E, C),
            multi_bond(I, _),
            multi_bond(J, _),
            parent(I, _, J),
            S < J, J <= E,
            S < I, I <= E.
    """,

    # A ring cycle is aromatic in Kekulé representation if:
    #   6-membered ring with 3 alternating double bonds (benzene-type), or
    #   5-membered ring with 2 double bonds + one heteroatom (furan/pyrrole/thiophene).
    # Consecutive double bonds (cumulenes) are excluded — they are never aromatic.
    "aromatic_ring_cycle": """
        aromatic_ring_cycle(C) :-
            cycle_start(S, C),
            cycle_end(E, C),
            E - S = 5,
            ring_double_bond_count(C, 3),
            not consecutive_ring_doubles(C).
        aromatic_ring_cycle(C) :-
            cycle_start(S, C),
            cycle_end(E, C),
            E - S = 4,
            ring_double_bond_count(C, 2),
            ring_atom(I), symbol(I, EL), EL != "C", EL != "H",
            S <= I, I <= E,
            not consecutive_ring_doubles(C).
    """,

    "aromatic_atom": """
        aromatic_atom(I) :-
            atom_in_ring(I, C),
            aromatic_ring_cycle(C).
    """,

    "aromatic_atom_count": """
        aromatic_atom_count(N) :-
            N = #count { I : aromatic_atom(I) }.
    """,

    "aromatic_heteroatom_count": """
        aromatic_heteroatom_count(N) :-
            N = #count { I : aromatic_atom(I), smiles_symbol(I, E), E != "C", E != "H" }.
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
            N = #count { I : aromatic_atom(I), smiles_symbol(I, "P") }.
    """,

    "aromatic_ring_count": """
        aromatic_ring_count(N) :-
            N = #count { C : aromatic_ring_cycle(C) }.
    """,

    "aromatic_ring": """
        aromatic_ring :-
            aromatic_ring_count(N),
            N > 0.
    """,

    "aromatic_s_count": """
        aromatic_s_count(N) :-
            N = #count { I : aromatic_atom(I), smiles_symbol(I, "S") }.
    """,

    # smiles.lp never outputs @ / @@ stereo notation, so no generated molecule
    # ever has an assigned chiral center — assigned is always 0.
    "assigned_chiral_center_count": """
        assigned_chiral_center_count(0).
    """,

    # Topological potential chiral centers without explicit stereo = unassigned.
    "unassigned_chiral_center_count": """
        unassigned_chiral_center_count(N) :-
            chiral_center_count(N).
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
        cycle_marker(I, CYCLE) :-
            cycle_start(I, CYCLE).
        cycle_marker(I, CYCLE) :-
            cycle_end(I, CYCLE).

        bridgehead_atom(I) :-
            cycle_marker(I, CYCLE1),
            cycle_marker(I, CYCLE2),
            CYCLE1 < CYCLE2.

        bridgehead_atom_count(N) :-
            N = #count { I : bridgehead_atom(I) }.
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

    ##TODO: might fuckup, this is overly generous apperntly
    "chiral_center_count": """
        potential_tetrahedral_chiral_center(I) :-
            symbol(I, "C"),
            atom_degree(I, 4),
            free_bonds(I, 0),
            not double_bond_between(I, _),
            not triple_bond_between(I, _).

        potential_tetrahedral_chiral_center(I) :-
            symbol(I, "C"),
            atom_degree(I, 3),
            free_bonds(I, 1),
            not double_bond_between(I, _),
            not triple_bond_between(I, _).

        chiral_center_count(N) :-
            N = #count { I : potential_tetrahedral_chiral_center(I) }.
    """,


    #TODO probably a useless feature, we are only testing single component molecules
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

    "stereo_double_bond_E_count": """
        % E/Z stereo is currently not represented in smiles.lp.
        stereo_double_bond_E_count(0).
    """,

    "stereo_double_bond_Z_count": """
        % E/Z stereo is currently not represented in smiles.lp.
        stereo_double_bond_Z_count(0).
    """,

    "stereo_double_bond_unspecified_count": """
        % Without E/Z stereo tags, all modeled double bonds are unspecified.
        stereo_double_bond_unspecified_count(N) :-
            double_bond_count(N).
    """,

    "ring_count": """
        ring_count(N) :-
            N = #count { CYCLE : cycle_start(_, CYCLE) }.
    """,

    "ring_atom": """
        ring_atom(I) :- cycle_start(I, _).
        ring_atom(I) :- cycle_end(I, _).
        % All atoms on the true ring backbone path are ring atoms.
        ring_atom(I) :- atom_in_ring(I).
    """,

    "atom_in_ring": """
        % Path-based ring membership: only atoms on the true DFS backbone
        % path from cycle_start to cycle_end are ring members.
        atom_in_ring(I, CY) :- ring_path_atom(CY, I).
        % The atoms that open/close the ring are always ring members.
        % ring_path_atom misses them when cycle_start and cycle_end are DFS siblings.
        atom_in_ring(I, CY) :- cycle_start(I, CY).
        atom_in_ring(I, CY) :- cycle_end(I, CY).
        % Extended rule 1: when cycle_end is adjacent to cycle_start (E = S+1),
        % the ring path runs backward through the root via atoms 1..S-1.
        atom_in_ring(I, CY) :-
            cycle_start(S, CY),
            cycle_end(E, CY),
            E = S + 1,
            symbol(I, _),
            1 <= I, I < S.
        % Extended rule 2: atom I is on the ring path when it branches toward
        % cycle_start S (via child C1 with C1 <= S) and toward cycle_end E
        % (via a different child C2 with C1 < C2 <= E).
        % Handles backward-through-root rings of any size.
        atom_in_ring(I, CY) :-
            cycle_start(S, CY),
            cycle_end(E, CY),
            parent(I, _, C1),
            parent(I, _, C2),
            C1 != C2,
            C1 <= S,
            C1 < C2,
            C2 <= E.
        % Propagation: if atom I is in ring CY and has a child J strictly before
        % cycle_start S, J is also on the ring path (backward path through root).
        % In DFS-ordered trees all children have higher indices than parents, so
        % J < S implies I < S < J is impossible -- J is only added when I < S,
        % i.e. I was itself added by an extended rule, making this safe.
        atom_in_ring(J, CY) :-
            atom_in_ring(I, CY),
            cycle_start(S, CY),
            parent(I, _, J),
            J < S.
        % Forward propagation from branching atom toward cycle_end.
        % For backward-through-root rings the branching atom has one child going
        % toward the cycle_start (already covered by cycle_start/ring_path_atom)
        % and another child going toward the cycle_end.  Atoms on the latter path
        % are missed by ring_path_atom because they are not descendants of S.
        % We propagate forward (I < J <= E) while excluding S's own subtree.
        atom_in_ring(J, CY) :-
            atom_in_ring(I, CY),
            cycle_end(E, CY),
            cycle_start(S, CY),
            parent(I, _, J),
            I < J, J <= E,
            not is_descendant(S, J).
        atom_in_ring(I) :-
            atom_in_ring(I, _).
    """,

    "ring_span_size": """
        ring_span_size(CY, N) :-
            cycle_start(S, CY),
            cycle_end(E, CY),
            N = E - S + 1.
    """,

    "is_descendant": """
        % is_descendant(A, B): B is A or a descendant of A in the parent (DFS) tree.
        is_descendant(A, A) :- symbol(A, _).
        is_descendant(A, C) :- parent(A, _, B), is_descendant(B, C).
    """,

    "ring_path_atom": """
        % ring_path_atom(CY, I): I lies on the backbone DFS path from
        % cycle_start(CY) to cycle_end(CY).  Replaces the old span heuristic
        % I2 - I1 + 1 which overcounts branch atoms.
        ring_path_atom(CY, I) :-
            cycle_start(S, CY),
            cycle_end(E, CY),
            is_descendant(S, I),
            is_descendant(I, E).
    """,

    "ring_size": """
        % True ring size: count of atoms on the backbone path for ring CY.
        ring_size(CY, N) :-
            cycle_start(_, CY),
            N = #count { I : atom_in_ring(I, CY) }.
    """,

    "same_ring": """
        same_ring(I, J) :-
            atom_in_ring(I, CY),
            atom_in_ring(J, CY),
            I != J.
    """,

    "approx_ring_bond": """
        approx_ring_bond(I, J) :-
            any_bond_between(I, J),
            same_ring(I, J).
    """,

    "approx_nonring_bond": """
        approx_nonring_bond(I, J) :-
            any_bond_between(I, J),
            not approx_ring_bond(I, J).
    """,

    "ring_bond_degree": """
        ring_bond_degree(I, N) :-
            atom_in_ring(I),
            N = #count { J : approx_ring_bond(I, J) }.
    """,

    "degree_of_unsaturation": """
        degree_of_unsaturation(N) :-
            double_bond_count(DB),
            triple_bond_count(TB),
            ring_count(R),
            N = DB + 2 * TB + R.
    """,

    "rotatable_bond_count": """
        amide_link(I, J) :-
            amide_atom(N),
            symbol(C, "C"),
            single_bond_between(C, N),
            I = C,
            J = N.
        amide_link(I, J) :-
            amide_link(J, I).

        rotatable_bond(I, J) :-
            single_bond_between(I, J),
            I < J,
            symbol(I, E1), E1 != "H",
            symbol(J, E2), E2 != "H",
            not ring_atom(I),
            not ring_atom(J),
            atom_degree(I, D1), D1 > 1,
            atom_degree(J, D2), D2 > 1,
            not amide_link(I, J).

        rotatable_bond_count(N) :-
            N = #count { I, J : rotatable_bond(I, J) }.
    """,

    "element_B_count": """
        element_B_count(N) :-
            N = #count { I : smiles_symbol(I, "B") }.
    """,

    "element_Br_count": """
        element_Br_count(N) :-
            N = #count { I : smiles_symbol(I, "Br") }.
    """,

    "element_C_count": """
        element_C_count(N) :-
            N = #count { I : symbol(I, "C") }.
    """,

    "element_Cl_count": """
        element_Cl_count(N) :-
            N = #count { I : smiles_symbol(I, "Cl") }.
    """,

    "element_F_count": """
        element_F_count(N) :-
            N = #count { I : smiles_symbol(I, "F") }.
    """,

    "element_H_count": """
        element_H_count(N) :-
            molecular_formula("H", N).
        element_H_count(0) :-
            not molecular_formula("H", _).
    """,

    "element_I_count": """
        element_I_count(N) :-
            N = #count { I : smiles_symbol(I, "I") }.
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
        element_P_count(N) :-
            N = #count{ I : smiles_symbol(I, "P") }.
    """,

    "element_S_count": """
        element_S_count(N) :-
            N = #count{ I : smiles_symbol(I, "S") }.
    """,

    "element_Si_count": """
        element_Si_count(N) :-
            N = #count { I : smiles_symbol(I, "Si") }.
    """,

    "ch2_atom": """
        ch2_atom(I) :-
            symbol(I, "C"),
            free_bonds(I, 2).
    """,

    "ch3_atom": """
            ch3_atom(I) :-
                symbol(I, "C"),
                free_bonds(I, H),
                H >= 3.
        """,

    "qh_atom": """
        qh_atom(I) :-
            heteroatom(I),
            free_bonds(I, H),
            H >= 1.
    """,

    "ch2_or_ch3_atom": """
    ch2_or_ch3_atom(I) :-
        symbol(I, "C"),
        free_bonds(I, H),
        H >= 2.
    """,

##TODO: future elements need adding
    "exact_mass_da": """
        exact_mass_da(M) :-
            M = 12 * c + 1 * h + 14 * n + 16 * o.
    """,

    ##TODO: currently only C/H/N/O are exposed; add P/S/halogens/etc when available.
    "molecular_weight_da": """
        molecular_weight_da(M) :-
            M = 12 * c + 1 * h + 14 * n + 16 * o.
    """,

    # smiles.lp N is always neutral valence-3; every N atom contributes total
    # valence 3 (explicit bonds + implicit H = 3) regardless of hybridization.
    # free_bonds = implicit-H count, NOT a valence indicator.
    "n_valence_1_count": """
        n_valence_1_count(0).
    """,

    "n_valence_2_count": """
        n_valence_2_count(0).
    """,

    "n_valence_3_count": """
        n_valence_3_count(N) :-
            N = #count { I : symbol(I, "N") }.
    """,

    "n_valence_4plus_count": """
        n_valence_4plus_count(0).
    """,

    "nitrile_atom": """
        nitrile_atom(C) :-
            symbol(C, "C"),
            symbol(N, "N"),
            triple_bond_between(C, N).
    """,

    "nitrile_count_value": """
        nitrile_count_value(N) :-
            N = #count { C : nitrile_atom(C) }.
    """,

    "nitrile_count": """
        nitrile_count(N) :- nitrile_count_value(N).
    """,

    "nitro_atom": """
        nitro_atom(N) :-
            symbol(N, "N"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            O1 != O2,
            double_bond_between(N, O1),
            single_bond_between(N, O2).
    """,

    "nitro_count_value": """
        nitro_count_value(N) :-
            N = #count { I : nitro_atom(I) }.
    """,

    "nitro_count": """
        nitro_count(N) :- nitro_count_value(N).
    """,

    # smiles.lp O is always neutral valence-2; every O atom contributes total
    # valence 2 (explicit bonds + implicit H = 2) regardless of bonding.
    "o_valence_1_count": """
        o_valence_1_count(0).
    """,

    "o_valence_2_count": """
        o_valence_2_count(N) :-
            N = #count { I : symbol(I, "O") }.
    """,

    "o_valence_3plus_count": """
        o_valence_3plus_count(0).
    """,

    # smiles.lp splits P into P3 (valence 3) and P5 (valence 5); use the
    # symbol type directly instead of free_bonds (which is implicit-H count).
    "p_valence_3_count": """
        p_valence_3_count(N) :-
            N = #count { I : symbol(I, "P3") }.
    """,

    "p_valence_5plus_count": """
        p_valence_5plus_count(N) :-
            N = #count { I : symbol(I, "P5") }.
    """,

    "s_valence_2_count": """
        s_valence_2_count(N) :-
            N = #count { I : symbol(I, "S2") }.
    """,

    "s_valence_4_count": """
        s_valence_4_count(N) :-
            N = #count { I : symbol(I, "S4") }.
    """,

    "s_valence_6plus_count": """
        s_valence_6plus_count(N) :-
            N = #count { I : symbol(I, "S6") }.
    """,

    # TODO: Approximate spiro counting from cycle interval boundaries.
    # Replace with exact ring-membership intersection logic if/when smiles.lp
    # exposes per-cycle atom membership facts.
    "spiro_atom_count": """
        % Two cycles are treated as spiro-connected when their cycle intervals
        % touch at exactly one atom index.
        spiro_pair_atom(I, C1, C2) :-
            cycle_start(S1, C1),
            cycle_end(E1, C1),
            cycle_start(S2, C2),
            cycle_end(E2, C2),
            C1 < C2,
            E1 = S2,
            I = S2.

        spiro_pair_atom(I, C1, C2) :-
            cycle_start(S1, C1),
            cycle_end(E1, C1),
            cycle_start(S2, C2),
            cycle_end(E2, C2),
            C1 < C2,
            E2 = S1,
            I = S1.

        spiro_atom_count(N) :-
            N = #count { I : spiro_pair_atom(I, _, _) }.
    """,

    "phenol_atom": """
        phenol_atom(O) :-
            symbol(O, "O"),
            symbol(C, "C"),
            single_bond_between(O, C),
            free_bonds(O, 1),
            aromatic_atom(C).
    """,

    "phenol_count_value": """
        phenol_count_value(N) :-
            N = #count { O : phenol_atom(O) }.
    """,

    "phenol_count": """
        phenol_count(N) :- phenol_count_value(N).
    """,

    "phosphate_atom": """
        phosphate_atom(P) :-
            smiles_symbol(P, "P"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            double_bond_between(P, O1),
            single_bond_between(P, O2).
    """,

    "phosphate_count_value": """
        phosphate_count_value(N) :-
            N = #count { P : phosphate_atom(P) }.
    """,

    "phosphate_count": """
        phosphate_count(N) :- phosphate_count_value(N).
    """,

    "phosphonate_atom": """
        phosphonate_atom(P) :-
            smiles_symbol(P, "P"),
            symbol(C, "C"),
            symbol(O, "O"),
            single_bond_between(P, C),
            single_bond_between(P, O).
    """,

    "phosphonate_count_value": """
        phosphonate_count_value(N) :-
            N = #count { P : phosphonate_atom(P) }.
    """,

    "phosphonate_count": """
        phosphonate_count(N) :- phosphonate_count_value(N).
    """,

    "sulfate_atom": """
        sulfate_atom(S) :-
            smiles_symbol(S, "S"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            symbol(O3, "O"),
            symbol(O4, "O"),
            O1 < O2,
            O3 < O4,
            O1 != O3,
            O1 != O4,
            O2 != O3,
            O2 != O4,
            double_bond_between(S, O1),
            double_bond_between(S, O2),
            single_bond_between(S, O3),
            single_bond_between(S, O4).
    """,

    "sulfate_count_value": """
        sulfate_count_value(N) :-
            N = #count { S : sulfate_atom(S) }.
    """,

    "sulfate_count": """
        sulfate_count(N) :- sulfate_count_value(N).
    """,

    "sulfonamide_atom": """
        sulfonamide_atom(S) :-
            smiles_symbol(S, "S"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            symbol(N, "N"),
            O1 < O2,
            N != O1,
            N != O2,
            double_bond_between(S, O1),
            double_bond_between(S, O2),
            single_bond_between(S, N).
    """,

    "sulfonamide_count_value": """
        sulfonamide_count_value(N) :-
            N = #count { S : sulfonamide_atom(S) }.
    """,

    "sulfonamide_count": """
        sulfonamide_count(N) :- sulfonamide_count_value(N).
    """,

    "sulfoxide_atom": """
        sulfoxide_atom(S) :-
            smiles_symbol(S, "S"),
            atom_degree(S, 3),
            1 = #count { O : symbol(O, "O"), double_bond_between(S, O) }.
    """,

    "sulfoxide_count_value": """
        sulfoxide_count_value(N) :-
            N = #count { S : sulfoxide_atom(S) }.
    """,

    "sulfoxide_count": """
        sulfoxide_count(N) :- sulfoxide_count_value(N).
    """,

    "total_atom_count_with_h": """
        total_atom_count_with_h(N) :-
            heavy_atom_count(HA),
            element_H_count(H),
            N = HA + H.
    """,

    "ring_size_3_count": """
        ring_size_3_count(N) :-
            N = #count { CYCLE : ring_size(CYCLE, 3) }.
    """,

    "ring_size_4_count": """
        ring_size_4_count(N) :-
            N = #count { CYCLE : ring_size(CYCLE, 4) }.
    """,

    "ring_size_5_count": """
        ring_size_5_count(N) :-
            N = #count { CYCLE : ring_size(CYCLE, 5) }.
    """,

    "ring_size_6_count": """
        ring_size_6_count(N) :-
            N = #count { CYCLE : ring_size(CYCLE, 6) }.
    """,

    "ring_size_7plus_count": """
        ring_size_7plus_count(N) :-
            N = #count { CYCLE : ring_size(CYCLE, S), S >= 7 }.
    """,

    "ring_size_7_count": """
        ring_size_7_count(N) :-
            N = #count { CYCLE : ring_size(CYCLE, 7) }.
    """,

    "ring_size_8plus_count": """
        ring_size_8plus_count(N) :-
            N = #count { CYCLE : ring_size(CYCLE, S), S >= 8 }.
    """,
    
    "nonring_o_a_a_atom": """
        nonring_o_a_a_atom(I) :-
            symbol(I, "O"),
            atom_degree(I, 2),
            not ring_atom(I),
            any_bond_between(I, J),
            any_bond_between(I, K),
            J != K,
            J != I, K != I,
            not ring_atom(J),
            not ring_atom(K).
    """,

    "nonring_ch2_a_a_atom": """
        nonring_ch2_a_a_atom(I) :-
            ch2_atom(I),
            atom_degree(I, 2),
            not ring_atom(I),
            any_bond_between(I, J),
            any_bond_between(I, K),
            J != K,
            J != I, K != I,
            not ring_atom(J),
            not ring_atom(K).
    """,

    "heavy_atom_count": """
        heavy_atom_count(N) :-
            N = #count { I : symbol(I, E), E != "H" }.
    """,

    "heteroatom_count": """
        heteroatom_count(N) :-
            N = #count { I : symbol(I, E), E != "C", E != "H" }.
    """,

    "heteroatom_exocyclic_count": """
        heteroatom_exocyclic_count(N) :-
            N = #count { I : heteroatom(I), not ring_atom(I) }.
    """,

    "heteroatom_in_ring_count": """
        heteroatom_in_ring_count(N) :-
            N = #count { I : heteroatom(I), ring_atom(I) }.
    """,

    #TODO formal charge
    "formal_charge": """
        formal_charge(0).
    """,

    "fused_ring_pair_count": """
        fused_ring_pair(C1, C2) :-
            cycle_start(I, C1),
            cycle_start(I, C2),
            C1 < C2.
        fused_ring_pair(C1, C2) :-
            cycle_end(I, C1),
            cycle_end(I, C2),
            C1 < C2.
        fused_ring_pair(C1, C2) :-
            cycle_start(I, C1),
            cycle_end(I, C2),
            C1 < C2.

        fused_ring_pair_count(N) :-
            N = #count { C1, C2 : fused_ring_pair(C1, C2) }.
    """,

    "ring_system_count": """
        ring_cycle_interval(CYCLE, I1, I2) :-
            cycle_start(I1, CYCLE),
            cycle_end(I2, CYCLE).

        ring_system_edge(C1, C2) :-
            ring_cycle_interval(C1, S1, E1),
            ring_cycle_interval(C2, S2, E2),
            C1 < C2,
            S1 <= E2,
            S2 <= E1.

        ring_system_connected(C1, C2) :-
            ring_system_edge(C1, C2).
        ring_system_connected(C1, C2) :-
            ring_system_edge(C2, C1).

        ring_system_reach(C, C) :-
            ring_cycle_interval(C, _, _).
        ring_system_reach(C1, C3) :-
            ring_system_reach(C1, C2),
            ring_system_connected(C2, C3).

        ring_system_rep(C, R) :-
            ring_system_reach(C, R),
            not ring_system_reach(C, R2) : ring_cycle_interval(R2, _, _), R2 < R.

        ring_system_count(N) :-
            N = #count { R : ring_system_rep(_, R) }.
    """,

    "fused_ring_system_count": """
        fused_ring_system_count(1) :-
            fused_ring_pair_count(N),
            N > 0.
        fused_ring_system_count(0) :-
            fused_ring_pair_count(0).
    """,

    "imide": """
        imide(N) :-
            N = #count {
                I :
                    symbol(I, "N"),
                    symbol(C1, "C"),
                    symbol(C2, "C"),
                    C1 != C2,
                    symbol(O1, "O"),
                    symbol(O2, "O"),
                    double_bond_between(C1, O1),
                    double_bond_between(C2, O2),
                    single_bond_between(I, C1),
                    single_bond_between(I, C2)
            }.
    """,

    "ketone": """
        ketone(N) :-
            N = #count {
                C :
                    symbol(C, "C"),
                    symbol(O, "O"),
                    double_bond_between(C, O),
                    symbol(C1, "C"),
                    symbol(C2, "C"),
                    C1 != C2,
                    single_bond_between(C, C1),
                    single_bond_between(C, C2)
            }.
    """,

    "lactam": """
        lactam(N) :-
            N = #count {
                C :
                    symbol(C, "C"),
                    symbol(O, "O"),
                    double_bond_between(C, O),
                    ring_atom(C),
                    symbol(N1, "N"),
                    ring_atom(N1),
                    single_bond_between(C, N1)
            }.
    """,

    "lactone": """
        lactone(N) :-
            N = #count {
                C :
                    symbol(C, "C"),
                    symbol(O1, "O"),
                    double_bond_between(C, O1),
                    ring_atom(C),
                    symbol(O2, "O"),
                    ring_atom(O2),
                    O1 != O2,
                    single_bond_between(C, O2)
            }.
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

    "halogenated_count": """
        halogenated_count(N) :-
            element_F_count(F),
            element_Cl_count(Cl),
            element_Br_count(Br),
            element_I_count(I),
            N = F + Cl + Br + I.
    """,

    "heteroatom": """
        heteroatom(I) :-
            smiles_symbol(I, E),
            E != "C",
            E != "H".
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
            % Non-ring C-O single bond (alcohol, ether, ester, etc.)
            has_c_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "O"),
                not approx_ring_bond(I, J).

            % Ring C-O single bond only when neither atom is in an aromatic ring.
            % MACCS 157 uses [#6]-[#8] (explicit single) which does not match aromatic C-O bonds.
            has_c_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "O"),
                approx_ring_bond(I, J),
                not aromatic_atom(I),
                not aromatic_atom(J).

            % 3- and 4-membered rings are never aromatic, so a ring C-O single bond
            % in a small ring is always a genuine single bond regardless of whether
            % the ring also contains a double bond (e.g. oxirene-like C=C-O rings).
            has_c_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "O"),
                atom_in_ring(I, CY),
                atom_in_ring(J, CY),
                ring_size(CY, N),
                N <= 4.
        """,
        "domain": {0, 1}
    },
    158: {
        "predicate": "has_c_n_single_bond",
        "definition": """
            % Non-ring C-N single bond
            has_c_n_single_bond :-
                single_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "N"),
                not approx_ring_bond(I, J).

            % Ring C-N single bond only when neither atom is in an aromatic ring.
            % MACCS 158 uses [#6]-[#7] which does not match aromatic C-N bonds.
            has_c_n_single_bond :-
                single_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "N"),
                approx_ring_bond(I, J),
                not aromatic_atom(I),
                not aromatic_atom(J).

            % 3- and 4-membered rings are never aromatic; ring C-N single bond in
            % a small ring is always genuine regardless of other double bonds in the ring.
            has_c_n_single_bond :-
                single_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "N"),
                atom_in_ring(I, CY),
                atom_in_ring(J, CY),
                ring_size(CY, N),
                N <= 4.
        """,
        "domain": {0, 1}
    },
    24: {
        "predicate": "has_n_o_single_bond",
        "definition": """
            % Non-ring-bond N-O single bond (including exocyclic bonds where one
            % atom is in a ring). Excludes only bonds where BOTH atoms share a ring.
            has_n_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "N"),
                symbol(J, "O"),
                not approx_ring_bond(I, J).

            % Ring N-O single bond only when neither atom is in an aromatic ring.
            has_n_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "N"),
                symbol(J, "O"),
                approx_ring_bond(I, J),
                not aromatic_atom(I),
                not aromatic_atom(J).

            % 3- and 4-membered rings are never aromatic; ring N-O single bond is genuine.
            has_n_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "N"),
                symbol(J, "O"),
                atom_in_ring(I, CY),
                atom_in_ring(J, CY),
                ring_size(CY, SZ),
                SZ <= 4.
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
            % Non-aromatic C=C: excludes bonds where both atoms are in aromatic rings.
            % RDKit's MACCS 99 ([#6]=[#6]) does not match aromatic C:C bonds.
            has_c_c_double_bond :-
                double_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "C"),
                not aromatic_atom(I),
                not aromatic_atom(J).
        """,
        "domain": {0, 1}
    },
    78: {
        "predicate": "has_c_n_double_bond",
        "definition": """
            % Non-aromatic C=N only: same aromatic Kekulé mismatch as has_c_c_double_bond.
            has_c_n_double_bond :-
                double_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "N"),
                not aromatic_atom(I),
                not aromatic_atom(J).
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
    },
    52: {
        "predicate": "has_n_n_bond",
        "definition": """
            has_n_n_bond :-
                any_bond_between(I, J),
                symbol(I, "N"),
                symbol(J, "N"),
                I < J.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### ELEMENT PRESENCE   #####
    ##############################
    161: {
        "predicate": "has_nitrogen",
        "definition": """
            has_nitrogen :- symbol(_, "N").
        """,
        "domain": {0, 1}
    },
    164: {
        "predicate": "has_oxygen",
        "definition": """
            has_oxygen :- symbol(_, "O").
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### ELEMENT COUNT      #####
    ##############################
    159: {
        "predicate": "has_oxygen_gt1",
        "definition": """
            has_oxygen_gt1 :- #count{ I : symbol(I, "O") } >= 2.
        """,
        "domain": {0, 1}
    },
    142: {
        "predicate": "has_nitrogen_gt1",
        "definition": """
            has_nitrogen_gt1 :- #count{ I : symbol(I, "N") } >= 2.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### RING PRESENCE      #####
    ##############################
    165: {
    "predicate": "has_ring",
    "definition": """
        has_ring :- atom_in_ring(_).
    """,
    "domain": {0, 1}
    },
    ##############################
    ##### 3-ATOM CHAINS      #####
    ##############################
    110: {
        "predicate": "has_n_c_o",
        "definition": """
            has_n_c_o :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "N"),
                symbol(J, "C"),
                symbol(K, "O"),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    123: {
        "predicate": "has_o_c_o",
        "definition": """
            has_o_c_o :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "O"),
                symbol(J, "C"),
                symbol(K, "O"),
                I != J, J != K, I < K.
        """,
        "domain": {0, 1}
    },
    77: {
        "predicate": "has_n_a_n",
        "definition": """
            has_n_a_n :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "N"),
                symbol(K, "N"),
                I != J, J != K, I < K.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### 4-ATOM CHAINS      #####
    ##############################
    72: {
        "predicate": "has_o_aa_o",
        "definition": """
            has_o_aa_o :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                symbol(I, "O"),
                symbol(L, "O"),
                I != J, J != K, K != L, I != K, J != L, I < L.
        """,
        "domain": {0, 1}
    },
    79: {
        "predicate": "has_n_aa_n",
        "definition": """
            has_n_aa_n :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                symbol(I, "N"),
                symbol(L, "N"),
                I != J, J != K, K != L, I != K, J != L, I < L.
        """,
        "domain": {0, 1}
    },
    95: {
        "predicate": "has_n_aa_o",
        "definition": """
            has_n_aa_o :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                symbol(I, "N"),
                symbol(L, "O"),
                I != J, J != K, K != L, I != K, J != L, I != L.
        """,
        "domain": {0, 1}
    },
    ##############################
    #####       HYDROGEN      ####
    ##############################
    84: {
        "predicate": "has_nh2",
        "definition": """
            has_nh2 :- symbol(I, "N"), free_bonds(I, 2).
        """,
        "domain": {0, 1}
    },
    151: {
        "predicate": "has_nh",
        "definition": """
            has_nh :- symbol(I, "N"), free_bonds(I, H), H >= 1.
        """,
        "domain": {0, 1}
    },
    139: {
        "predicate": "has_oh",
        "definition": """
            has_oh :- symbol(I, "O"), free_bonds(I, H), H >= 1.
        """,
        "domain": {0, 1}
    },
    160: {
        "predicate": "has_ch3",
        "definition": """
            has_ch3 :- symbol(I, "C"), free_bonds(I, H), H >= 3.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### SULFUR PRESENCE    #####
    ##############################
    88: {
        "predicate": "has_sulfur",
        "definition": """
            has_sulfur :- smiles_symbol(_, "S").
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### PHOSPHORUS PRESENCE ####
    ##############################
    29: {
        "predicate": "has_phosphorus",
        "definition": """
            has_phosphorus :- smiles_symbol(_, "P").
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### S DIRECT BONDS     #####
    ##############################
    14: {
        "predicate": "has_s_s_bond",
        "definition": """
            has_s_s_bond :-
                single_bond_between(I, J),
                smiles_symbol(I, "S"),
                smiles_symbol(J, "S"),
                I < J.
        """,
        "domain": {0, 1}
    },
    40: {
        "predicate": "has_s_o_single_bond",
        "definition": """
            has_s_o_single_bond :-
                single_bond_between(I, J),
                smiles_symbol(I, "S"),
                symbol(J, "O").
        """,
        "domain": {0, 1}
    },
    33: {
        "predicate": "has_n_s_bond",
        "definition": """
            has_n_s_bond :-
                any_bond_between(I, J),
                symbol(I, "N"),
                smiles_symbol(J, "S").
        """,
        "domain": {0, 1}
    },
    73: {
        "predicate": "has_s_double_bond",
        "definition": """
            has_s_double_bond :-
                double_bond_between(I, J),
                smiles_symbol(I, "S").
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### S 3-ATOM CHAINS    #####
    ##############################
    32: {
        "predicate": "has_c_s_n",
        "definition": """
            has_c_s_n :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "C"),
                smiles_symbol(J, "S"),
                symbol(K, "N"),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    51: {
        "predicate": "has_c_s_o",
        "definition": """
            has_c_s_o :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "C"),
                smiles_symbol(J, "S"),
                symbol(K, "O"),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    55: {
        "predicate": "has_o_s_o",
        "definition": """
            has_o_s_o :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "O"),
                smiles_symbol(J, "S"),
                symbol(K, "O"),
                I != J, J != K, I < K.
        """,
        "domain": {0, 1}
    },
    47: {
        "predicate": "has_s_a_n",
        "definition": """
            has_s_a_n :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                smiles_symbol(I, "S"),
                symbol(K, "N"),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### HETEROATOM BONDS   #####
    ##############################
    67: {
        "predicate": "has_q_s",
        "definition": """
            has_q_s :-
                any_bond_between(I, J),
                heteroatom(I),
                smiles_symbol(J, "S").
        """,
        "domain": {0, 1}
    },
    94: {
        "predicate": "has_q_n",
        "definition": """
            has_q_n :-
                any_bond_between(I, J),
                heteroatom(I),
                symbol(J, "N").
        """,
        "domain": {0, 1}
    },
    102: {
        "predicate": "has_q_o",
        "definition": """
            has_q_o :-
                any_bond_between(I, J),
                heteroatom(I),
                symbol(J, "O").
        """,
        "domain": {0, 1}
    },
    124: {
        "predicate": "has_qq",
        "definition": """
            has_qq :-
                any_bond_between(I, J),
                heteroatom(I),
                heteroatom(J),
                I < J.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### HETEROATOM CHAINS  #####
    ##############################
    58: {
        "predicate": "has_q_s_q",
        "definition": """
            has_q_s_q :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                heteroatom(I),
                smiles_symbol(J, "S"),
                heteroatom(K),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    70: {
        "predicate": "has_q_n_q",
        "definition": """
            has_q_n_q :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                heteroatom(I),
                symbol(J, "N"),
                heteroatom(K),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    117: {
        "predicate": "has_nao",
        "definition": """
            has_nao :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "N"),
                symbol(K, "O"),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    13: {
        "predicate": "has_on_cc",
        "definition": """
            has_on_cc :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "O"),
                symbol(J, "N"),
                symbol(K, "C"),
                symbol(L, "C"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### ALKENE PATTERNS    #####
    ##############################
    45: {
        "predicate": "has_c_eq_c_n",
        "definition": """
            % Non-aromatic C=C with adjacent N.
            has_c_eq_c_n :-
                double_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "C"),
                symbol(J, "C"),
                symbol(K, "N"),
                not aromatic_atom(I),
                not aromatic_atom(J),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    50: {
        "predicate": "has_c_eq_c_cc",
        "definition": """
            % Non-aromatic C=C with two C neighbors.
            has_c_eq_c_cc :-
                double_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "C"),
                symbol(J, "C"),
                symbol(K, "C"),
                symbol(L, "C"),
                not aromatic_atom(I),
                not aromatic_atom(J),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### N=A                #####
    ##############################
    119: {
        "predicate": "has_n_eq_a",
        "definition": """
            % Non-aromatic N= only; aromatic ring N= excluded (Kekulé mismatch).
            has_n_eq_a :-
                double_bond_between(I, J),
                symbol(I, "N"),
                not aromatic_atom(I),
                I != J.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### BRANCHED CENTERS   #####
    ##############################
    23: {
        "predicate": "has_nc_oo",
        "definition": """
            has_nc_oo :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "N"),
                symbol(J, "C"),
                symbol(K, "O"),
                symbol(L, "O"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    37: {
        "predicate": "has_nc_on",
        "definition": """
            has_nc_on :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "N"),
                symbol(J, "C"),
                symbol(K, "O"),
                symbol(L, "N"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    38: {
        "predicate": "has_nc_cn",
        "definition": """
            has_nc_cn :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "N"),
                symbol(J, "C"),
                symbol(K, "C"),
                symbol(L, "N"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    85: {
        "predicate": "has_cn_cc",
        "definition": """
            has_cn_cc :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "C"),
                symbol(J, "N"),
                symbol(K, "C"),
                symbol(L, "C"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    92: {
        "predicate": "has_oc_nc",
        "definition": """
            has_oc_nc :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "O"),
                symbol(J, "C"),
                symbol(K, "N"),
                symbol(L, "C"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    122: {
        "predicate": "has_an_aa",
        "definition": """
            has_an_aa :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(J, "N"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    152: {
        "predicate": "has_oc_cc",
        "definition": """
            has_oc_cc :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "O"),
                symbol(J, "C"),
                symbol(K, "C"),
                symbol(L, "C"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### 5-ATOM CHAINS      #####
    ##############################
    80: {
        "predicate": "has_naaan",
        "definition": """
            has_naaan :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                any_bond_between(L, M),
                symbol(I, "N"),
                symbol(M, "N"),
                I != J, J != K, K != L, L != M,
                I != K, I != L, I != M,
                J != L, J != M,
                K != M,
                I < M.
        """,
        "domain": {0, 1}
    },
    89: {
        "predicate": "has_oaaao",
        "definition": """
            has_oaaao :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                any_bond_between(L, M),
                symbol(I, "O"),
                symbol(M, "O"),
                I != J, J != K, K != L, L != M,
                I != K, I != L, I != M,
                J != L, J != M,
                K != M,
                I < M.
        """,
        "domain": {0, 1}
    },
    97: {
        "predicate": "has_naaao",
        "definition": """
            has_naaao :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                any_bond_between(L, M),
                symbol(I, "N"),
                symbol(M, "O"),
                I != J, J != K, K != L, L != M,
                I != K, I != L, I != M,
                J != L, J != M,
                K != M.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### CH2 / CH3 PATTERNS #####
    ##############################
    28: {
        "predicate": "has_q_ch2_q",
        "definition": """
            has_q_ch2_q :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                heteroatom(I),
                symbol(J, "C"),
                free_bonds(J, 2),
                heteroatom(K),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    93: {
        "predicate": "has_q_ch3",
        "definition": """
            has_q_ch3 :-
                any_bond_between(I, J),
                heteroatom(I),
                symbol(J, "C"),
                free_bonds(J, 3).
        """,
        "domain": {0, 1}
    },
    100: {
        "predicate": "has_ach2n",
        "definition": """
            has_ach2n :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(J, "C"),
                free_bonds(J, 2),
                symbol(K, "N"),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    109: {
        "predicate": "has_ach2o",
        "definition": """
            has_ach2o :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(J, "C"),
                free_bonds(J, 2),
                symbol(K, "O"),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    153: {
        "predicate": "has_q_ch2a",
        "definition": """
            has_q_ch2a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                heteroatom(I),
                symbol(J, "C"),
                free_bonds(J, 2),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### ELEMENT COUNT > N  #####
    ##############################
    146: {
        "predicate": "has_oxygen_gt2",
        "definition": """
            has_oxygen_gt2 :- #count{ I : symbol(I, "O") } >= 3.
        """,
        "domain": {0, 1}
    },
    149: {
        "predicate": "has_ch3_gt1",
        "definition": """
            has_ch3_gt1 :- #count{ I : symbol(I, "C"), free_bonds(I, F), F >= 3 } >= 2.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### BRANCHED CENTERS 2 #####
    ##############################
    15: {
        "predicate": "has_oc_oo",
        "definition": """
            has_oc_oo :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "O"),
                symbol(J, "C"),
                symbol(K, "O"),
                symbol(L, "O"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    25: {
        "predicate": "has_nc_nn",
        "definition": """
            has_nc_nn :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "N"),
                symbol(J, "C"),
                symbol(K, "N"),
                symbol(L, "N"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    39: {
        "predicate": "has_os_oo",
        "definition": """
            has_os_oo :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "O"),
                smiles_symbol(J, "S"),
                symbol(K, "O"),
                symbol(L, "O"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    48: {
        "predicate": "has_q_ooo",
        "definition": """
            has_q_ooo :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "O"),
                heteroatom(J),
                symbol(K, "O"),
                symbol(L, "O"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    56: {
        "predicate": "has_on_oc",
        "definition": """
            has_on_oc :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "O"),
                symbol(J, "N"),
                symbol(K, "O"),
                symbol(L, "C"),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    61: {
        "predicate": "has_s_3bonds",
        "definition": """
            has_s_3bonds :-
                smiles_symbol(J, "S"),
                atom_degree(J, D),
                D >= 3.
        """,
        "domain": {0, 1}
    },
    66: {
        "predicate": "has_cc_cca",
        "definition": """
            has_cc_cca :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                any_bond_between(J, M),
                symbol(I, "C"),
                symbol(J, "C"),
                symbol(K, "C"),
                symbol(L, "C"),
                I != J, J != K, J != L, J != M,
                I != K, I != L, I != M,
                K != L, K != M, L != M.
        """,
        "domain": {0, 1}
    },
    148: {
        "predicate": "has_aq_aa",
        "definition": """
            has_aq_aa :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                heteroatom(J),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### H-BEARING CHAINS   #####
    ##############################
    43: {
        "predicate": "has_qh_a_qh",
        "definition": """
            has_qh_a_qh :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                heteroatom(I),
                heteroatom(K),
                free_bonds(I, FI), FI >= 1,
                free_bonds(K, FK), FK >= 1,
                I != J, J != K, I < K.
        """,
        "domain": {0, 1}
    },
    54: {
        "predicate": "has_qh_aa_qh",
        "definition": """
            has_qh_aa_qh :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                heteroatom(I),
                heteroatom(L),
                free_bonds(I, FI), FI >= 1,
                free_bonds(L, FL), FL >= 1,
                I != J, J != K, K != L, I != K, J != L, I < L.
        """,
        "domain": {0, 1}
    },
    68: {
        "predicate": "has_qh_qh",
        "definition": """
            has_qh_qh :-
                any_bond_between(I, J),
                heteroatom(I),
                heteroatom(J),
                free_bonds(I, FI), FI >= 1,
                free_bonds(J, FJ), FJ >= 1,
                I < J.
        """,
        "domain": {0, 1}
    },
    69: {
        "predicate": "has_qq_h",
        "definition": """
            has_qq_h :-
                any_bond_between(I, J),
                heteroatom(I),
                heteroatom(J),
                free_bonds(J, FJ), FJ >= 1.
        """,
        "domain": {0, 1}
    },
    82: {
        "predicate": "has_ach2_qh",
        "definition": """
            has_ach2_qh :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(J, "C"),
                free_bonds(J, 2),
                heteroatom(K),
                free_bonds(K, FK), FK >= 1,
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### CH3 / C=C PATTERNS #####
    ##############################
    21: {
        "predicate": "has_c_eq_c_qq",
        "definition": """
            % Non-aromatic C=C with two heteroatom neighbors.
            has_c_eq_c_qq :-
                double_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "C"),
                symbol(J, "C"),
                heteroatom(K),
                heteroatom(L),
                not aromatic_atom(I),
                not aromatic_atom(J),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    34: {
        "predicate": "has_ch2_eq_a",
        "definition": """
            has_ch2_eq_a :-
                double_bond_between(I, J),
                symbol(I, "C"),
                free_bonds(I, 2),
                I != J.
        """,
        "domain": {0, 1}
    },
    74: {
        "predicate": "has_ch3_a_ch3",
        "definition": """
            has_ch3_a_ch3 :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                symbol(I, "C"), free_bonds(I, 3),
                symbol(K, "C"), free_bonds(K, 3),
                I != J, J != K, I < K.
        """,
        "domain": {0, 1}
    },
    76: {
        "predicate": "has_c_eq_c_aa",
        "definition": """
            % Non-aromatic C=C with two neighbors (any type).
            has_c_eq_c_aa :-
                double_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "C"),
                symbol(J, "C"),
                not aromatic_atom(I),
                not aromatic_atom(J),
                I != J, J != K, J != L, I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },
    ##############################
    ##### COUNT > N (cont.)  #####
    ##############################
    130: {
        "predicate": "has_qq_gt1",
        "definition": """
            has_qq_gt1 :- #count{ I, J : any_bond_between(I, J), heteroatom(I), heteroatom(J), I < J } >= 2.
        """,
        "domain": {0, 1}
    },
    131: {
        "predicate": "has_qh_gt1",
        "definition": """
            has_qh_gt1 :- #count{ I : heteroatom(I), free_bonds(I, F), F >= 1 } >= 2.
        """,
        "domain": {0, 1}
    },
    136: {
        "predicate": "has_o_eq_a_gt1",
        "definition": """
            has_o_eq_a_gt1 :- #count{ I : symbol(I, "O"), double_bond_between(I, J) } >= 2.
        """,
        "domain": {0, 1}
    },
    140: {
        "predicate": "has_oxygen_gt3",
        "definition": """
            has_oxygen_gt3 :- #count{ I : symbol(I, "O") } >= 4.
        """,
        "domain": {0, 1}
    },
    141: {
        "predicate": "has_ch3_gt2",
        "definition": """
            has_ch3_gt2 :- #count{ I : symbol(I, "C"), free_bonds(I, 3) } >= 3.
        """,
        "domain": {0, 1}
    },
    ##############################
    30: {
    "predicate": "has_cq_cc_a",
    "definition": """
        has_cq_cc_a :-
            any_bond_between(I, J),
            any_bond_between(J, K),
            any_bond_between(J, L),
            any_bond_between(J, M),
            symbol(I, "C"),
            heteroatom(J),
            symbol(K, "C"),
            symbol(L, "C"),
            J != I, J != K, J != L, J != M,
            I != K, I != L, I != M,
            K != L, K != M, L != M.
    """,
    "domain": {0, 1}
    },

    53: {
        "predicate": "has_qh_aaa_qh",
        "definition": """
            has_qh_aaa_qh :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                any_bond_between(L, M),
                qh_atom(I),
                qh_atom(M),
                I != J, J != K, K != L, L != M,
                I != K, I != L, I != M,
                J != L, J != M,
                K != M,
                I < M.
        """,
        "domain": {0, 1}
    },

    81: {
        "predicate": "has_s_a_aa",
        "definition": """
            has_s_a_aa :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                smiles_symbol(I, "S"),
                I != J, J != K, J != L,
                I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },

    86: {
        "predicate": "has_ch23_q_ch23",
        "definition": """
            has_ch23_q_ch23 :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                ch2_or_ch3_atom(I),
                heteroatom(J),
                ch2_or_ch3_atom(K),
                I != J, J != K,
                I < K.
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    104: {
        "predicate": "has_qh_a_ch2_a",
        "definition": """
            has_qh_a_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                qh_atom(I),
                ch2_atom(K),
                I != J, J != K, K != L,
                I != K, I != L, J != L.
        """,
        "domain": {0, 1}
    },

    106: {
        "predicate": "has_qa_q_q",
        "definition": """
            has_qa_q_q :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                heteroatom(I),
                heteroatom(K),
                heteroatom(L),
                I != J, J != K, J != L,
                I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },

    111: {
        "predicate": "has_n_a_ch2_a",
        "definition": """
            has_n_a_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                symbol(I, "N"),
                ch2_atom(K),
                I != J, J != K, K != L,
                I != K, I != L, J != L.
        """,
        "domain": {0, 1}
    },

    112: {
        "predicate": "has_aa_aa_a",
        "definition": """
            has_aa_aa_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                any_bond_between(J, M),
                I != J, J != K, J != L, J != M,
                I != K, I != L, I != M,
                K != L, K != M, L != M.
        """,
        "domain": {0, 1}
    },

    114: {
        "predicate": "has_ch3_ch2_a",
        "definition": """
            has_ch3_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                ch3_atom(I),
                ch2_atom(J),
                I != J, J != K, I != K.
        """,
        "domain": {0, 1}
    },

    115: {
        "predicate": "has_ch3_a_ch2_a",
        "definition": """
            has_ch3_a_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                ch3_atom(I),
                ch2_atom(K),
                I != J, J != K, K != L,
                I != K, I != L, J != L.
        """,
        "domain": {0, 1}
    },

    132: {
        "predicate": "has_o_a_ch2_a",
        "definition": """
            has_o_a_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                symbol(I, "O"),
                ch2_atom(K),
                I != J, J != K, K != L,
                I != K, I != L, J != L.
        """,
        "domain": {0, 1}
    },

    156: {
        "predicate": "has_n_a_aa",
        "definition": """
            has_n_a_aa :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(J, L),
                symbol(I, "N"),
                I != J, J != K, J != L,
                I != K, I != L, K != L.
        """,
        "domain": {0, 1}
    },

    108: {
        "predicate": "has_ch3_aaa_ch2_a",
        "definition": """
            has_ch3_aaa_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                any_bond_between(L, M),
                any_bond_between(M, N),
                ch3_atom(I),
                ch2_atom(M),
                I != J, J != K, K != L, L != M, M != N,
                I != K, I != L, I != M, I != N,
                J != L, J != M, J != N,
                K != M, K != N,
                L != N.
        """,
        "domain": {0, 1}
    },

    116: {
        "predicate": "has_ch3_aa_ch2_a",
        "definition": """
            has_ch3_aa_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                any_bond_between(L, M),
                ch3_atom(I),
                ch2_atom(L),
                I != J, J != K, K != L, L != M,
                I != K, I != L, I != M,
                J != L, J != M,
                K != M.
        """,
        "domain": {0, 1}
    },

    128: {
        "predicate": "has_a_ch2_aaa_ch2_a",
        "definition": """
            has_a_ch2_aaa_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                any_bond_between(L, M),
                any_bond_between(M, N),
                any_bond_between(N, O),
                ch2_atom(J),
                ch2_atom(N),
                I != J, J != K, K != L, L != M, M != N, N != O,
                I != K, I != L, I != M, I != N, I != O,
                J != L, J != M, J != N, J != O,
                K != M, K != N, K != O,
                L != N, L != O,
                M != O.
        """,
        "domain": {0, 1}
    },

    129: {
        "predicate": "has_a_ch2_aa_ch2_a",
        "definition": """
            has_a_ch2_aa_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                any_bond_between(L, M),
                any_bond_between(M, N),
                ch2_atom(J),
                ch2_atom(M),
                I != J, J != K, K != L, L != M, M != N,
                I != K, I != L, I != M, I != N,
                J != L, J != M, J != N,
                K != M, K != N,
                L != N.
        """,
        "domain": {0, 1}
    },

    147: {
        "predicate": "has_a_ch2_ch2_a",
        "definition": """
            has_a_ch2_ch2_a :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                any_bond_between(K, L),
                ch2_atom(J),
                ch2_atom(K),
                I != J, J != K, K != L,
                I != K, I != L, J != L.
        """,
        "domain": {0, 1}
    },

    11: {
        "predicate": "has_4m_ring",
        "definition": """
            has_4m_ring :-
                ring_size_4_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },

    19: {
        "predicate": "has_7m_ring",
        "definition": """
            has_7m_ring :-
                ring_size_7_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },

    22: {
        "predicate": "has_3m_ring",
        "definition": """
            has_3m_ring :-
                ring_size_3_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },

    96: {
        "predicate": "has_5m_ring",
        "definition": """
            has_5m_ring :-
                ring_size_5_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },

    101: {
        "predicate": "has_8m_or_larger_ring",
        "definition": """
            has_8m_or_larger_ring :-
                ring_size_8plus_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },

    126: {
        "predicate": "has_a_nonring_o_a",
        "definition": """
            % *!@[#8]!@*: O with two non-ring bonds
            has_a_nonring_o_a :-
                symbol(O, "O"),
                approx_nonring_bond(A, O),
                approx_nonring_bond(O, B),
                A != B.
        """,
        "domain": {0, 1}
    },

    155: {
        "predicate": "has_a_nonring_ch2_a",
        "definition": """
            % *!@[CH2]!@*: CH2 with two non-ring bonds
            has_a_nonring_ch2_a :-
                ch2_atom(C),
                approx_nonring_bond(A, C),
                approx_nonring_bond(C, B),
                A != B.
        """,
        "domain": {0, 1}
    },

    145: {
        "predicate": "has_6m_ring_gt1",
        "definition": """
            has_6m_ring_gt1 :-
                ring_size_6_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },

    163: {
        "predicate": "has_6m_ring",
        "definition": """
            has_6m_ring :-
                ring_size_6_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },

    2: {
        "predicate": "has_element_104",
        "definition": """
            has_element_104 :-
                smiles_symbol(_, "Rf").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    3: {
        "predicate": "has_group_iva_va_via_rows4to6",
        "definition": """
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "Ge").
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "As").
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "Se").
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "Sn").
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "Sb").
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "Te").
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "Pb").
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "Bi").
            has_group_iva_va_via_rows4to6 :-
                smiles_symbol(_, "Po").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    4: {
        "predicate": "has_actinide",
        "definition": """
            has_actinide :- smiles_symbol(_, "Ac").
            has_actinide :- smiles_symbol(_, "Th").
            has_actinide :- smiles_symbol(_, "Pa").
            has_actinide :- smiles_symbol(_, "U").
            has_actinide :- smiles_symbol(_, "Np").
            has_actinide :- smiles_symbol(_, "Pu").
            has_actinide :- smiles_symbol(_, "Am").
            has_actinide :- smiles_symbol(_, "Cm").
            has_actinide :- smiles_symbol(_, "Bk").
            has_actinide :- smiles_symbol(_, "Cf").
            has_actinide :- smiles_symbol(_, "Es").
            has_actinide :- smiles_symbol(_, "Fm").
            has_actinide :- smiles_symbol(_, "Md").
            has_actinide :- smiles_symbol(_, "No").
            has_actinide :- smiles_symbol(_, "Lr").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    5: {
        "predicate": "has_group_iiib_ivb",
        "definition": """
            has_group_iiib_ivb :- smiles_symbol(_, "Sc").
            has_group_iiib_ivb :- smiles_symbol(_, "Ti").
            has_group_iiib_ivb :- smiles_symbol(_, "Y").
            has_group_iiib_ivb :- smiles_symbol(_, "Zr").
            has_group_iiib_ivb :- smiles_symbol(_, "Hf").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    6: {
        "predicate": "has_lanthanide",
        "definition": """
            has_lanthanide :- smiles_symbol(_, "La").
            has_lanthanide :- smiles_symbol(_, "Ce").
            has_lanthanide :- smiles_symbol(_, "Pr").
            has_lanthanide :- smiles_symbol(_, "Nd").
            has_lanthanide :- smiles_symbol(_, "Pm").
            has_lanthanide :- smiles_symbol(_, "Sm").
            has_lanthanide :- smiles_symbol(_, "Eu").
            has_lanthanide :- smiles_symbol(_, "Gd").
            has_lanthanide :- smiles_symbol(_, "Tb").
            has_lanthanide :- smiles_symbol(_, "Dy").
            has_lanthanide :- smiles_symbol(_, "Ho").
            has_lanthanide :- smiles_symbol(_, "Er").
            has_lanthanide :- smiles_symbol(_, "Tm").
            has_lanthanide :- smiles_symbol(_, "Yb").
            has_lanthanide :- smiles_symbol(_, "Lu").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    7: {
        "predicate": "has_group_vb_vib_viib",
        "definition": """
            has_group_vb_vib_viib :- smiles_symbol(_, "V").
            has_group_vb_vib_viib :- smiles_symbol(_, "Cr").
            has_group_vb_vib_viib :- smiles_symbol(_, "Mn").
            has_group_vb_vib_viib :- smiles_symbol(_, "Nb").
            has_group_vb_vib_viib :- smiles_symbol(_, "Mo").
            has_group_vb_vib_viib :- smiles_symbol(_, "Tc").
            has_group_vb_vib_viib :- smiles_symbol(_, "Ta").
            has_group_vb_vib_viib :- smiles_symbol(_, "W").
            has_group_vb_vib_viib :- smiles_symbol(_, "Re").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    9: {
        "predicate": "has_group_viii",
        "definition": """
            has_group_viii :- smiles_symbol(_, "Fe").
            has_group_viii :- smiles_symbol(_, "Co").
            has_group_viii :- smiles_symbol(_, "Ni").
            has_group_viii :- smiles_symbol(_, "Ru").
            has_group_viii :- smiles_symbol(_, "Rh").
            has_group_viii :- smiles_symbol(_, "Pd").
            has_group_viii :- smiles_symbol(_, "Os").
            has_group_viii :- smiles_symbol(_, "Ir").
            has_group_viii :- smiles_symbol(_, "Pt").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    12: {
        "predicate": "has_group_ib_iib",
        "definition": """
            has_group_ib_iib :- smiles_symbol(_, "Cu").
            has_group_ib_iib :- smiles_symbol(_, "Zn").
            has_group_ib_iib :- smiles_symbol(_, "Ag").
            has_group_ib_iib :- smiles_symbol(_, "Cd").
            has_group_ib_iib :- smiles_symbol(_, "Au").
            has_group_ib_iib :- smiles_symbol(_, "Hg").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    18: {
        "predicate": "has_group_iiia",
        "definition": """
            has_group_iiia :- smiles_symbol(_, "B").
            has_group_iiia :- smiles_symbol(_, "Al").
            has_group_iiia :- smiles_symbol(_, "Ga").
            has_group_iiia :- smiles_symbol(_, "In").
            has_group_iiia :- smiles_symbol(_, "Tl").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    20: {
        "predicate": "has_silicon",
        "definition": """
            has_silicon :- smiles_symbol(_, "Si").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    27: {
        "predicate": "has_iodine",
        "definition": """
            has_iodine :- smiles_symbol(_, "I").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    31: {
        "predicate": "has_q_x",
        "definition": """
            has_q_x :-
                any_bond_between(I, J),
                heteroatom(I),
                smiles_symbol(J, "F").
            has_q_x :-
                any_bond_between(I, J),
                heteroatom(I),
                smiles_symbol(J, "Cl").
            has_q_x :-
                any_bond_between(I, J),
                heteroatom(I),
                smiles_symbol(J, "Br").
            has_q_x :-
                any_bond_between(I, J),
                heteroatom(I),
                smiles_symbol(J, "I").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    35: {
        "predicate": "has_alkali_metal",
        "definition": """
            has_alkali_metal :- smiles_symbol(_, "Li").
            has_alkali_metal :- smiles_symbol(_, "Na").
            has_alkali_metal :- smiles_symbol(_, "K").
            has_alkali_metal :- smiles_symbol(_, "Rb").
            has_alkali_metal :- smiles_symbol(_, "Cs").
            has_alkali_metal :- smiles_symbol(_, "Fr").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    42: {
        "predicate": "has_fluorine",
        "definition": """
            has_fluorine :- smiles_symbol(_, "F").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    44: {
        "predicate": "has_other_element",
        "definition": """
            has_other_element :-
                smiles_symbol(_, E),
                E != "H",
                E != "C",
                E != "N",
                E != "O",
                E != "F",
                E != "Si",
                E != "P",
                E != "S",
                E != "Cl",
                E != "Br",
                E != "I".
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    46: {
        "predicate": "has_bromine",
        "definition": """
            has_bromine :- smiles_symbol(_, "Br").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    134: {
        "predicate": "has_halogen",
        "definition": """
            has_halogen :- smiles_symbol(_, "F").
            has_halogen :- smiles_symbol(_, "Cl").
            has_halogen :- smiles_symbol(_, "Br").
            has_halogen :- smiles_symbol(_, "I").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    90: {
        "predicate": "has_qh_a_a_ch2_a",
        "definition": """
            has_qh_a_a_ch2_a :-
                heavy_neighbor(I, J),
                heavy_neighbor(J, K),
                heavy_neighbor(K, L),
                heavy_neighbor(L, M),
                qh_atom(I),
                ch2_atom(L),
                I != J, J != K, K != L, L != M,
                I != K, I != L, I != M,
                J != L, J != M,
                K != M.
        """,
        "domain": {0, 1}
    },

    91: {
        "predicate": "has_qh_a_a_a_ch2_a",
        "definition": """
            has_qh_a_a_a_ch2_a :-
                heavy_neighbor(I, J),
                heavy_neighbor(J, K),
                heavy_neighbor(K, L),
                heavy_neighbor(L, M),
                heavy_neighbor(M, N),
                qh_atom(I),
                ch2_atom(M),
                I != J, J != K, K != L, L != M, M != N,
                I != K, I != L, I != M, I != N,
                J != L, J != M, J != N,
                K != M, K != N,
                L != N.
        """,
        "domain": {0, 1}
    },

    103: {
        "predicate": "has_chlorine",
        "definition": """
            has_chlorine :-
                smiles_symbol(_, "Cl").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    107: {
        "predicate": "has_x_a_aa",
        "definition": """
            halogen_atom(I) :-
                smiles_symbol(I, "F").
            halogen_atom(I) :-
                smiles_symbol(I, "Cl").
            halogen_atom(I) :-
                smiles_symbol(I, "Br").
            halogen_atom(I) :-
                smiles_symbol(I, "I").

            has_x_a_aa :-
                heavy_neighbor(I, J),
                heavy_neighbor(J, K),
                heavy_neighbor(J, L),
                halogen_atom(I),
                I != J, J != K, J != L,
                I != K, I != L, K != L.
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    118: {
        "predicate": "has_a_ch2_ch2_a_gt1",
        "definition": """
            % SMARTS $(*~[CH2]~[CH2]~*) counts each outer atom I adjacent to a CH2-CH2
            % pair — two distinct outer atoms per pair, not one tuple per pair.
            a_ch2_ch2_a_outer(I) :-
                heavy_neighbor(I, J),
                heavy_neighbor(J, K),
                heavy_neighbor(K, L),
                ch2_atom(J),
                ch2_atom(K),
                I != J, J != K, K != L,
                I != K, I != L, J != L.

            has_a_ch2_ch2_a_gt1 :-
                2 <= #count { I : a_ch2_ch2_a_outer(I) }.
        """,
        "domain": {0, 1}
    },

    138: {
        "predicate": "has_q_ch2_a_gt1",
        "definition": """
            q_ch2_a_match(I, J, K) :-
                any_bond_between(I, J),
                any_bond_between(J, K),
                heteroatom(I),
                symbol(J, "C"),
                free_bonds(J, 2),
                I != J, J != K, I != K.

            has_q_ch2_a_gt1 :-
                2 <= #count { I, J, K : q_ch2_a_match(I, J, K) }.
        """,
        "domain": {0, 1}
    },

    ##############################
    ##### PADDING / TRIVIAL  #####
    ##############################
    1: {
        "predicate": "has_isotope",
        "definition": """
            % MACCS key 1: ISOTOPE - smiles.lp does not model isotope labels; always 0.
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    49: {
        "predicate": "has_charge",
        "definition": """
            % MACCS key 49: CHARGE - smiles.lp only models neutral molecules; always 0.
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    166: {
        "predicate": "has_fragments",
        "definition": """
            % MACCS key 166: FRAGMENTS > 1 - smiles.lp models single connected molecules; always 0.
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    ##############################
    ##### ALKALINE / METALS  #####
    ##############################
    10: {
        "predicate": "has_alkaline_earth",
        "definition": """
            has_alkaline_earth :- smiles_symbol(_, "Be").
            has_alkaline_earth :- smiles_symbol(_, "Mg").
            has_alkaline_earth :- smiles_symbol(_, "Ca").
            has_alkaline_earth :- smiles_symbol(_, "Sr").
            has_alkaline_earth :- smiles_symbol(_, "Ba").
            has_alkaline_earth :- smiles_symbol(_, "Ra").
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    ##############################
    ##### HETEROATOM IN RING #####
    ##############################
    16: {
        "predicate": "has_q_in_3m_ring",
        "definition": """
            has_q_in_3m_ring :-
                heteroatom(I),
                atom_in_ring(I, CY),
                ring_size(CY, 3).
        """,
        "domain": {0, 1}
    },

    8: {
        "predicate": "has_q_in_4m_ring",
        "definition": """
            has_q_in_4m_ring :-
                heteroatom(I),
                atom_in_ring(I, CY),
                ring_size(CY, 4).
        """,
        "domain": {0, 1}
    },

    83: {
        "predicate": "has_q_in_5m_ring",
        "definition": """
            has_q_in_5m_ring :-
                heteroatom(I),
                atom_in_ring(I, CY),
                ring_size(CY, 5).
        """,
        "domain": {0, 1}
    },

    98: {
        "predicate": "has_q_in_6m_ring",
        "definition": """
            has_q_in_6m_ring :-
                heteroatom(I),
                atom_in_ring(I, CY),
                ring_size(CY, 6).
        """,
        "domain": {0, 1}
    },

    ##############################
    ##### S / O / N IN RING  #####
    ##############################
    36: {
        "predicate": "has_s_in_ring",
        "definition": """
            has_s_in_ring :-
                smiles_symbol(I, "S"),
                atom_in_ring(I).
        """,
        "domain": {0, 1}
    },

    57: {
        "predicate": "has_o_in_ring",
        "definition": """
            has_o_in_ring :-
                symbol(I, "O"),
                atom_in_ring(I).
        """,
        "domain": {0, 1}
    },

    121: {
        "predicate": "has_n_in_ring",
        "definition": """
            has_n_in_ring :-
                symbol(I, "N"),
                atom_in_ring(I).
        """,
        "domain": {0, 1}
    },

    120: {
        "predicate": "has_heteroatom_in_ring_gt1",
        "definition": """
            has_heteroatom_in_ring_gt1 :-
                2 <= #count { I : heteroatom(I), atom_in_ring(I) }.
        """,
        "domain": {0, 1}
    },

    137: {
        "predicate": "has_heterocycle",
        "definition": """
            has_heterocycle :-
                heteroatom(I),
                atom_in_ring(I).
        """,
        "domain": {0, 1}
    },

    ##############################
    ##### C=C IN RING        #####
    ##############################
    26: {
        "predicate": "has_c_eq_c_ring",
        "definition": """
            % C$=C: C=C double bond where both carbons share a ring
            has_c_eq_c_ring :-
                double_bond_between(I, J),
                symbol(I, "C"),
                symbol(J, "C"),
                same_ring(I, J).
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    ##############################
    ##### AROMATIC PATTERNS  #####
    ##############################
    162: {
        "predicate": "has_aromatic_atom",
        "definition": """
            % Aromatic: approximated as any atom in a ring span
            has_aromatic_atom :- atom_in_ring(_).
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    125: {
        "predicate": "has_aromatic_ring_gt1",
        "definition": """
            % Aromatic ring > 1
            has_aromatic_ring_gt1 :-
                aromatic_ring_count(N),
                N > 1.
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    65: {
        "predicate": "has_aromatic_c_n",
        "definition": """
            % c:n - aromatic C bonded to aromatic N (approximated as ring bond C-N)
            has_aromatic_c_n :-
                symbol(I, "C"),
                symbol(J, "N"),
                approx_ring_bond(I, J).
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    ##############################
    ##### EXOCYCLIC BONDS    #####
    ##############################
    59: {
        "predicate": "has_s_exo_aromatic",
        "definition": """
            % Snot%A%A: S (non-ring) with non-ring bond to ring atom J, J has ring bond to K
            has_s_exo_aromatic :-
                smiles_symbol(I, "S"),
                not atom_in_ring(I),
                approx_nonring_bond(I, J),
                approx_ring_bond(J, K),
                I != K.
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    113: {
        "predicate": "has_o_exo_aromatic",
        "definition": """
            % Onot%A%A: O (non-ring) with non-ring bond to a ring atom
            has_o_exo_aromatic :-
                symbol(I, "O"),
                not atom_in_ring(I),
                approx_nonring_bond(I, J),
                atom_in_ring(J).
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    135: {
        "predicate": "has_n_exo_aromatic",
        "definition": """
            % Nnot%A%A: N (non-ring) with non-ring bond to a ring atom
            has_n_exo_aromatic :-
                symbol(I, "N"),
                not atom_in_ring(I),
                approx_nonring_bond(I, J),
                atom_in_ring(J).
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    87: {
        "predicate": "has_halogen_to_ring",
        "definition": """
            % X!A$A: halogen with non-ring bond to a ring atom
            has_halogen_to_ring :-
                smiles_symbol(I, "F"),
                approx_nonring_bond(I, J),
                atom_in_ring(J).
            has_halogen_to_ring :-
                smiles_symbol(I, "Cl"),
                approx_nonring_bond(I, J),
                atom_in_ring(J).
            has_halogen_to_ring :-
                smiles_symbol(I, "Br"),
                approx_nonring_bond(I, J),
                atom_in_ring(J).
            has_halogen_to_ring :-
                smiles_symbol(I, "I"),
                approx_nonring_bond(I, J),
                atom_in_ring(J).
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

    64: {
        "predicate": "has_ring_atom_to_s",
        "definition": """
            % A$A!S: ring atom with non-ring bond to S
            has_ring_atom_to_s :-
                atom_in_ring(I),
                approx_nonring_bond(I, J),
                smiles_symbol(J, "S").
        """,
        "domain": {0, 1}
    },

    143: {
        "predicate": "has_ring_atom_to_o",
        "definition": """
            % A$A!O: ring atom with non-ring bond to O
            has_ring_atom_to_o :-
                atom_in_ring(I),
                approx_nonring_bond(I, J),
                symbol(J, "O").
        """,
        "domain": {0, 1}
    },

    127: {
        "predicate": "has_ring_atom_to_o_gt1",
        "definition": """
            % A$A!O > 1: more than 1 ring atom with non-ring bond to O
            ring_to_o_match(I, J) :-
                atom_in_ring(I),
                approx_nonring_bond(I, J),
                symbol(J, "O").

            has_ring_atom_to_o_gt1 :-
                2 <= #count { I, J : ring_to_o_match(I, J) }.
        """,
        "domain": {0, 1}
    },

    133: {
        "predicate": "has_ring_atom_to_n",
        "definition": """
            % A$A!N: ring atom with non-ring bond to N
            has_ring_atom_to_n :-
                atom_in_ring(I),
                approx_nonring_bond(I, J),
                symbol(J, "N").
        """,
        "domain": {0, 1}
    },

    75: {
        "predicate": "has_nonring_bond_to_n_in_ring",
        "definition": """
            % A!N$A: non-ring bond to N that is in a ring
            has_nonring_bond_to_n_in_ring :-
                symbol(J, "N"),
                atom_in_ring(J),
                approx_nonring_bond(I, J).
        """,
        "domain": {0, 1}
    },

    ##############################
    ##### RING BOND PATTERNS #####
    ##############################
    62: {
        "predicate": "has_ring_inter_nonring_ring",
        "definition": """
            % *@*!@*@*: ring-bond, non-ring-bond, ring-bond (e.g. biphenyl)
            has_ring_inter_nonring_ring :-
                approx_ring_bond(A, B),
                approx_nonring_bond(B, C),
                approx_ring_bond(C, D),
                A != C, B != D.
        """,
        "domain": {0, 1}
    },

    105: {
        "predicate": "has_ring_junction_atom",
        "definition": """
            % *@*(@*)@*: atom with 3 or more ring bonds
            has_ring_junction_atom :-
                ring_bond_degree(I, N),
                N >= 3.
        """,
        "domain": {0, 1}
    },

    150: {
        "predicate": "has_nonring_ring_nonring",
        "definition": """
            % *!@*@*!@*: non-ring bond, ring bond, non-ring bond
            has_nonring_ring_nonring :-
                approx_nonring_bond(A, B),
                approx_ring_bond(B, C),
                approx_nonring_bond(C, D),
                A != C, B != D, A != D.
        """,
        "domain": {0, 1}
    },

    144: {
        "predicate": "has_nonaro_aro_aro_nonaro",
        "definition": """
            % Anot%A%Anot%A: non-aromatic-aromatic-aromatic-non-aromatic
            % Approximated as non-ring bond, ring bond, non-ring bond (same as 150)
            has_nonaro_aro_aro_nonaro :-
                approx_nonring_bond(A, B),
                approx_ring_bond(B, C),
                approx_nonring_bond(C, D),
                A != C, B != D, A != D.
        """,
        "domain": {0, 1},
        "unsupported": True,
    },

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
        "predicate": "double_bond_count_bin_0",
        "definition": """
            double_bond_count_bin_0 :-
                double_bond_count(N),
                N <= 3.
        """,
        "domain": {0, 1}
    },
    240: {
        "predicate": "double_bond_count_bin_1",
        "definition": """
            double_bond_count_bin_1 :-
                double_bond_count(N),
                N > 3,
                N <= 5.
        """,
        "domain": {0, 1}
    },
    241: {
        "predicate": "double_bond_count_bin_2",
        "definition": """
            double_bond_count_bin_2 :-
                double_bond_count(N),
                N > 5,
                N <= 7.
        """,
        "domain": {0, 1}
    },
    242: {
        "predicate": "double_bond_count_bin_3",
        "definition": """
            double_bond_count_bin_3 :-
                double_bond_count(N),
                N > 7,
                N <= 9.
        """,
        "domain": {0, 1}
    },
    243: {
        "predicate": "double_bond_count_bin_4",
        "definition": """
            double_bond_count_bin_4 :-
                double_bond_count(N),
                N > 9,
                N <= 13.
        """,
        "domain": {0, 1}
    },
    244: {
        "predicate": "double_bond_count_bin_5",
        "definition": """
            double_bond_count_bin_5 :-
                double_bond_count(N),
                N > 13.
        """,
        "domain": {0, 1}
    },
    245: {
        "predicate": "element_B_bin_0",
        "definition": """
            element_B_bin_0 :-
                element_B_count(0).
        """,
        "domain": {0, 1}
    },
    246: {
        "predicate": "element_B_bin_1",
        "definition": """
            element_B_bin_1 :-
                element_B_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    247: {
        "predicate": "element_Br_bin_0",
        "definition": """
            element_Br_bin_0 :-
                element_Br_count(0).
        """,
        "domain": {0, 1}
    },
    248: {
        "predicate": "element_Br_bin_1",
        "definition": """
            element_Br_bin_1 :-
                element_Br_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    249: {
        "predicate": "element_C_bin_0",
        "definition": """
            element_C_bin_0 :-
                element_C_count(N),
                N <= 35.
        """,
        "domain": {0, 1}
    },
    250: {
        "predicate": "element_C_bin_1",
        "definition": """
            element_C_bin_1 :-
                element_C_count(N),
                N > 35,
                N <= 42.
        """,
        "domain": {0, 1}
    },
    251: {
        "predicate": "element_C_bin_2",
        "definition": """
            element_C_bin_2 :-
                element_C_count(N),
                N > 42,
                N <= 49.
        """,
        "domain": {0, 1}
    },
    252: {
        "predicate": "element_C_bin_3",
        "definition": """
            element_C_bin_3 :-
                element_C_count(N),
                N > 49,
                N <= 60.
        """,
        "domain": {0, 1}
    },
    253: {
        "predicate": "element_C_bin_4",
        "definition": """
            element_C_bin_4 :-
                element_C_count(N),
                N > 60,
                N <= 66.
        """,
        "domain": {0, 1}
    },
    254: {
        "predicate": "element_C_bin_5",
        "definition": """
            element_C_bin_5 :-
                element_C_count(N),
                N > 66.
        """,
        "domain": {0, 1}
    },
    255: {
        "predicate": "element_Cl_bin_0",
        "definition": """
            element_Cl_bin_0 :-
                element_Cl_count(0).
        """,
        "domain": {0, 1}
    },
    256: {
        "predicate": "element_Cl_bin_1",
        "definition": """
            element_Cl_bin_1 :-
                element_Cl_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    257: {
        "predicate": "element_F_bin_0",
        "definition": """
            element_F_bin_0 :-
                element_F_count(0).
        """,
        "domain": {0, 1}
    },
    258: {
        "predicate": "element_F_bin_1",
        "definition": """
            element_F_bin_1 :-
                element_F_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    259: {
        "predicate": "element_H_bin_0",
        "definition": """
            element_H_bin_0 :-
                element_H_count(N),
                N <= 64.
        """,
        "domain": {0, 1}
    },
    260: {
        "predicate": "element_H_bin_1",
        "definition": """
            element_H_bin_1 :-
                element_H_count(N),
                N > 64,
                N <= 77.
        """,
        "domain": {0, 1}
    },
    261: {
        "predicate": "element_H_bin_2",
        "definition": """
            element_H_bin_2 :-
                element_H_count(N),
                N > 77,
                N <= 88.
        """,
        "domain": {0, 1}
    },
    262: {
        "predicate": "element_H_bin_3",
        "definition": """
            element_H_bin_3 :-
                element_H_count(N),
                N > 88,
                N <= 102.
        """,
        "domain": {0, 1}
    },
    263: {
        "predicate": "element_H_bin_4",
        "definition": """
            element_H_bin_4 :-
                element_H_count(N),
                N > 102,
                N <= 113.
        """,
        "domain": {0, 1}
    },
    264: {
        "predicate": "element_H_bin_5",
        "definition": """
            element_H_bin_5 :-
                element_H_count(N),
                N > 113.
        """,
        "domain": {0, 1}
    },
    265: {
        "predicate": "element_I_bin_0",
        "definition": """
            element_I_bin_0 :-
                element_I_count(0).
        """,
        "domain": {0, 1}
    },
    266: {
        "predicate": "element_I_bin_1",
        "definition": """
            element_I_bin_1 :-
                element_I_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    267: {
        "predicate": "element_N_bin_0",
        "definition": """
            element_N_bin_0 :-
                element_N_count(0).
        """,
        "domain": {0, 1}
    },
    268: {
        "predicate": "element_N_bin_1",
        "definition": """
            element_N_bin_1 :-
                element_N_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    269: {
        "predicate": "element_N_bin_2",
        "definition": """
            element_N_bin_2 :-
                element_N_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    270: {
        "predicate": "element_O_bin_0",
        "definition": """
            element_O_bin_0 :-
                element_O_count(N),
                N <= 8.
        """,
        "domain": {0, 1}
    },
    271: {
        "predicate": "element_O_bin_1",
        "definition": """
            element_O_bin_1 :-
                element_O_count(N),
                N > 8,
                N <= 10.
        """,
        "domain": {0, 1}
    },
    272: {
        "predicate": "element_O_bin_2",
        "definition": """
            element_O_bin_2 :-
                element_O_count(N),
                N > 10,
                N <= 11.
        """,
        "domain": {0, 1}
    },
    273: {
        "predicate": "element_O_bin_3",
        "definition": """
            element_O_bin_3 :-
                element_O_count(N),
                N > 11.
        """,
        "domain": {0, 1}
    },
    274: {
        "predicate": "element_P_bin_0",
        "definition": """
            element_P_bin_0 :-
                element_P_count(0).
        """,
        "domain": {0, 1}
    },
    275: {
        "predicate": "element_P_bin_1",
        "definition": """
            element_P_bin_1 :-
                element_P_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    276: {
        "predicate": "element_P_bin_2",
        "definition": """
            element_P_bin_2 :-
                element_P_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    277: {
        "predicate": "element_S_bin_0",
        "definition": """
            element_S_bin_0 :-
                element_S_count(0).
        """,
        "domain": {0, 1}
    },
    278: {
        "predicate": "element_S_bin_1",
        "definition": """
            element_S_bin_1 :-
                element_S_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    279: {
        "predicate": "element_Si_bin_0",
        "definition": """
            element_Si_bin_0 :-
                element_Si_count(0).
        """,
        "domain": {0, 1}
    },
    280: {
        "predicate": "element_Si_bin_1",
        "definition": """
            element_Si_bin_1 :-
                element_Si_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    281: {
        "predicate": "ester_bin_0",
        "definition": """
            ester_bin_0 :-
                ester_count(N),
                N <= 2.
        """,
        "domain": {0, 1}
    },
    282: {
        "predicate": "ester_bin_1",
        "definition": """
            ester_bin_1 :-
                ester_count(N),
                N > 2,
                N <= 3.
        """,
        "domain": {0, 1}
    },
    283: {
        "predicate": "ester_bin_2",
        "definition": """
            ester_bin_2 :-
                ester_count(N),
                N > 3.
        """,
        "domain": {0, 1}
    },
    284: {
        "predicate": "ether_bin_0",
        "definition": """
            ether_bin_0 :-
                ether_count(N),
                N <= 2.
        """,
        "domain": {0, 1}
    },
    285: {
        "predicate": "ether_bin_1",
        "definition": """
            ether_bin_1 :-
                ether_count(N),
                N > 2,
                N <= 3.
        """,
        "domain": {0, 1}
    },
    286: {
        "predicate": "ether_bin_2",
        "definition": """
            ether_bin_2 :-
                ether_count(N),
                N > 3,
                N <= 4.
        """,
        "domain": {0, 1}
    },
    287: {
        "predicate": "ether_bin_3",
        "definition": """
            ether_bin_3 :-
                ether_count(N),
                N > 4.
        """,
        "domain": {0, 1}
    },
    288: {
        "predicate": "exact_mass_da_bin_0",
        "definition": """
            exact_mass_da_bin_0 :-
                exact_mass_da(M),
                M <= 651.58.
        """,
        "domain": {0, 1}
    },
    289: {
        "predicate": "exact_mass_da_bin_1",
        "definition": """
            exact_mass_da_bin_1 :-
                exact_mass_da(M),
                M > 651.58,
                M <= 751.515.
        """,
        "domain": {0, 1}
    },
    290: {
        "predicate": "exact_mass_da_bin_2",
        "definition": """
            exact_mass_da_bin_2 :-
                exact_mass_da(M),
                M > 751.515,
                M <= 834.614.
        """,
        "domain": {0, 1}
    },
    291: {
        "predicate": "exact_mass_da_bin_3",
        "definition": """
            exact_mass_da_bin_3 :-
                exact_mass_da(M),
                M > 834.614,
                M <= 1006.72.
        """,
        "domain": {0, 1}
    },
    292: {
        "predicate": "exact_mass_da_bin_4",
        "definition": """
            exact_mass_da_bin_4 :-
                exact_mass_da(M),
                M > 1006.72,
                M <= 1112.8.
        """,
        "domain": {0, 1}
    },
    293: {
        "predicate": "exact_mass_da_bin_5",
        "definition": """
            exact_mass_da_bin_5 :-
                exact_mass_da(M),
                M > 1112.8.
        """,
        "domain": {0, 1}
    },
    294: {
        "predicate": "formal_charge_bin_0",
        "definition": """
            formal_charge_bin_0 :-
                formal_charge(0).
        """,
        "domain": {0, 1}
    },
    295: {
        "predicate": "formal_charge_bin_1",
        "definition": """
            formal_charge_bin_1 :-
                formal_charge(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    296: {
        "predicate": "fused_ring_pair_count_bin_0",
        "definition": """
            fused_ring_pair_count_bin_0 :-
                fused_ring_pair_count(0).
        """,
        "domain": {0, 1}
    },
    297: {
        "predicate": "fused_ring_pair_count_bin_1",
        "definition": """
            fused_ring_pair_count_bin_1 :-
                fused_ring_pair_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    298: {
        "predicate": "fused_ring_system_count_bin_0",
        "definition": """
            fused_ring_system_count_bin_0 :-
                fused_ring_system_count(0).
        """,
        "domain": {0, 1}
    },
    299: {
        "predicate": "fused_ring_system_count_bin_1",
        "definition": """
            fused_ring_system_count_bin_1 :-
                fused_ring_system_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    300: {
        "predicate": "halogenated_bin_0",
        "definition": """
            halogenated_bin_0 :-
                not halogenated.
        """,
        "domain": {0, 1}
    },
    301: {
        "predicate": "halogenated_bin_1",
        "definition": """
            halogenated_bin_1 :-
                halogenated.
        """,
        "domain": {0, 1}
    },

    ##TODO: currently use placeholder helper method to define heavy_atom_count
    302: {
        "predicate": "heavy_atom_count_bin_0",
        "definition": """
            heavy_atom_count_bin_0 :-
                heavy_atom_count(N),
                N <= 45.
        """,
        "domain": {0, 1}
    },
    303: {
        "predicate": "heavy_atom_count_bin_1",
        "definition": """
            heavy_atom_count_bin_1 :-
                heavy_atom_count(N),
                N > 45,
                N <= 52.
        """,
        "domain": {0, 1}
    },
    304: {
        "predicate": "heavy_atom_count_bin_2",
        "definition": """
            heavy_atom_count_bin_2 :-
                heavy_atom_count(N),
                N > 52,
                N <= 58.
        """,
        "domain": {0, 1}
    },
    305: {
        "predicate": "heavy_atom_count_bin_3",
        "definition": """
            heavy_atom_count_bin_3 :-
                heavy_atom_count(N),
                N > 58,
                N <= 70.
        """,
        "domain": {0, 1}
    },
    306: {
        "predicate": "heavy_atom_count_bin_4",
        "definition": """
            heavy_atom_count_bin_4 :-
                heavy_atom_count(N),
                N > 70,
                N <= 78.
        """,
        "domain": {0, 1}
    },
    307: {
        "predicate": "heavy_atom_count_bin_5",
        "definition": """
            heavy_atom_count_bin_5 :-
                heavy_atom_count(N),
                N > 78.
        """,
        "domain": {0, 1}
    },
    308: {
        "predicate": "heteroatom_count_bin_0",
        "definition": """
            heteroatom_count_bin_0 :-
                heteroatom_count(N),
                N <= 9.
        """,
        "domain": {0, 1}
    },
    309: {
        "predicate": "heteroatom_count_bin_1",
        "definition": """
            heteroatom_count_bin_1 :-
                heteroatom_count(N),
                N > 9,
                N <= 10.
        """,
        "domain": {0, 1}
    },
    310: {
        "predicate": "heteroatom_count_bin_2",
        "definition": """
            heteroatom_count_bin_2 :-
                heteroatom_count(N),
                N > 10,
                N <= 12.
        """,
        "domain": {0, 1}
    },
    311: {
        "predicate": "heteroatom_count_bin_3",
        "definition": """
            heteroatom_count_bin_3 :-
                heteroatom_count(N),
                N > 12.
        """,
        "domain": {0, 1}
    },
    312: {
        "predicate": "heteroatom_exocyclic_count_bin_0",
        "definition": """
            heteroatom_exocyclic_count_bin_0 :-
                heteroatom_exocyclic_count(N),
                N <= 9.
        """,
        "domain": {0, 1}
    },
    313: {
        "predicate": "heteroatom_exocyclic_count_bin_1",
        "definition": """
            heteroatom_exocyclic_count_bin_1 :-
                heteroatom_exocyclic_count(N),
                N > 9,
                N <= 10.
        """,
        "domain": {0, 1}
    },
    314: {
        "predicate": "heteroatom_exocyclic_count_bin_2",
        "definition": """
            heteroatom_exocyclic_count_bin_2 :-
                heteroatom_exocyclic_count(N),
                N > 10,
                N <= 12.
        """,
        "domain": {0, 1}
    },
    315: {
        "predicate": "heteroatom_exocyclic_count_bin_3",
        "definition": """
            heteroatom_exocyclic_count_bin_3 :-
                heteroatom_exocyclic_count(N),
                N > 12.
        """,
        "domain": {0, 1}
    },
    316: {
        "predicate": "heteroatom_in_ring_count_bin_0",
        "definition": """
            heteroatom_in_ring_count_bin_0 :-
                heteroatom_in_ring_count(0).
        """,
        "domain": {0, 1}
    },
    317: {
        "predicate": "heteroatom_in_ring_count_bin_1",
        "definition": """
            heteroatom_in_ring_count_bin_1 :-
                heteroatom_in_ring_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    318: {
        "predicate": "heteroatom_in_ring_count_bin_2",
        "definition": """
            heteroatom_in_ring_count_bin_2 :-
                heteroatom_in_ring_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    319: {
        "predicate": "imide_bin_0",
        "definition": """
            imide_bin_0 :-
                imide(0).
        """,
        "domain": {0, 1}
    },
    320: {
        "predicate": "imide_bin_1",
        "definition": """
            imide_bin_1 :-
                imide(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    321: {
        "predicate": "ketone_bin_0",
        "definition": """
            ketone_bin_0 :-
                ketone(0).
        """,
        "domain": {0, 1}
    },
    322: {
        "predicate": "ketone_bin_1",
        "definition": """
            ketone_bin_1 :-
                ketone(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    323: {
        "predicate": "lactam_bin_0",
        "definition": """
            lactam_bin_0 :-
                lactam(0).
        """,
        "domain": {0, 1}
    },
    324: {
        "predicate": "lactam_bin_1",
        "definition": """
            lactam_bin_1 :-
                lactam(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    325: {
        "predicate": "lactone_bin_0",
        "definition": """
            lactone_bin_0 :-
                lactone(0).
        """,
        "domain": {0, 1}
    },
    326: {
        "predicate": "lactone_bin_1",
        "definition": """
            lactone_bin_1 :-
                lactone(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    327: {
        "predicate": "molecular_weight_da_bin_0",
        "definition": """
            molecular_weight_da_bin_0 :-
                molecular_weight_da(M),
                M <= 652.058.
        """,
        "domain": {0, 1}
    },
    328: {
        "predicate": "molecular_weight_da_bin_1",
        "definition": """
            molecular_weight_da_bin_1 :-
                molecular_weight_da(M),
                M > 652.058,
                M <= 752.027.
        """,
        "domain": {0, 1}
    },
    329: {
        "predicate": "molecular_weight_da_bin_2",
        "definition": """
            molecular_weight_da_bin_2 :-
                molecular_weight_da(M),
                M > 752.027,
                M <= 835.201.
        """,
        "domain": {0, 1}
    },
    330: {
        "predicate": "molecular_weight_da_bin_3",
        "definition": """
            molecular_weight_da_bin_3 :-
                molecular_weight_da(M),
                M > 835.201,
                M <= 1007.42.
        """,
        "domain": {0, 1}
    },
    331: {
        "predicate": "molecular_weight_da_bin_4",
        "definition": """
            molecular_weight_da_bin_4 :-
                molecular_weight_da(M),
                M > 1007.42,
                M <= 1113.59.
        """,
        "domain": {0, 1}
    },
    332: {
        "predicate": "molecular_weight_da_bin_5",
        "definition": """
            molecular_weight_da_bin_5 :-
                molecular_weight_da(M),
                M > 1113.59.
        """,
        "domain": {0, 1}
    },
    333: {
        "predicate": "n_valence_1_count_bin_0",
        "definition": """
            n_valence_1_count_bin_0 :-
                n_valence_1_count(0).
        """,
        "domain": {0, 1}
    },
    334: {
        "predicate": "n_valence_1_count_bin_1",
        "definition": """
            n_valence_1_count_bin_1 :-
                n_valence_1_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    335: {
        "predicate": "n_valence_2_count_bin_0",
        "definition": """
            n_valence_2_count_bin_0 :-
                n_valence_2_count(0).
        """,
        "domain": {0, 1}
    },
    336: {
        "predicate": "n_valence_2_count_bin_1",
        "definition": """
            n_valence_2_count_bin_1 :-
                n_valence_2_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    337: {
        "predicate": "n_valence_3_count_bin_0",
        "definition": """
            n_valence_3_count_bin_0 :-
                n_valence_3_count(0).
        """,
        "domain": {0, 1}
    },
    338: {
        "predicate": "n_valence_3_count_bin_1",
        "definition": """
            n_valence_3_count_bin_1 :-
                n_valence_3_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    339: {
        "predicate": "n_valence_4plus_count_bin_0",
        "definition": """
            n_valence_4plus_count_bin_0 :-
                n_valence_4plus_count(0).
        """,
        "domain": {0, 1}
    },
    340: {
        "predicate": "n_valence_4plus_count_bin_1",
        "definition": """
            n_valence_4plus_count_bin_1 :-
                n_valence_4plus_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    341: {
        "predicate": "n_valence_4plus_count_bin_2",
        "definition": """
            n_valence_4plus_count_bin_2 :-
                n_valence_4plus_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    342: {
        "predicate": "nitrile_bin_0",
        "definition": """
            nitrile_bin_0 :-
                nitrile_count(0).
        """,
        "domain": {0, 1}
    },
    343: {
        "predicate": "nitrile_bin_1",
        "definition": """
            nitrile_bin_1 :-
                nitrile_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    344: {
        "predicate": "nitro_bin_0",
        "definition": """
            nitro_bin_0 :-
                nitro_count(0).
        """,
        "domain": {0, 1}
    },
    345: {
        "predicate": "nitro_bin_1",
        "definition": """
            nitro_bin_1 :-
                nitro_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    346: {
        "predicate": "o_valence_1_count_bin_0",
        "definition": """
            o_valence_1_count_bin_0 :-
                o_valence_1_count(0).
        """,
        "domain": {0, 1}
    },
    347: {
        "predicate": "o_valence_1_count_bin_1",
        "definition": """
            o_valence_1_count_bin_1 :-
                o_valence_1_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    348: {
        "predicate": "o_valence_1_count_bin_2",
        "definition": """
            o_valence_1_count_bin_2 :-
                o_valence_1_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    349: {
        "predicate": "o_valence_2_count_bin_0",
        "definition": """
            o_valence_2_count_bin_0 :-
                o_valence_2_count(N),
                N <= 7.
        """,
        "domain": {0, 1}
    },
    350: {
        "predicate": "o_valence_2_count_bin_1",
        "definition": """
            o_valence_2_count_bin_1 :-
                o_valence_2_count(N),
                N > 7,
                N <= 8.
        """,
        "domain": {0, 1}
    },
    351: {
        "predicate": "o_valence_2_count_bin_2",
        "definition": """
            o_valence_2_count_bin_2 :-
                o_valence_2_count(N),
                N > 8,
                N <= 10.
        """,
        "domain": {0, 1}
    },
    352: {
        "predicate": "o_valence_2_count_bin_3",
        "definition": """
            o_valence_2_count_bin_3 :-
                o_valence_2_count(N),
                N > 10,
                N <= 11.
        """,
        "domain": {0, 1}
    },
    353: {
        "predicate": "o_valence_2_count_bin_4",
        "definition": """
            o_valence_2_count_bin_4 :-
                o_valence_2_count(N),
                N > 11.
        """,
        "domain": {0, 1}
    },
    354: {
        "predicate": "o_valence_3plus_count_bin_0",
        "definition": """
            o_valence_3plus_count_bin_0 :-
                o_valence_3plus_count(0).
        """,
        "domain": {0, 1}
    },
    355: {
        "predicate": "o_valence_3plus_count_bin_1",
        "definition": """
            o_valence_3plus_count_bin_1 :-
                o_valence_3plus_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    356: {
        "predicate": "p_valence_3_count_bin_0",
        "definition": """
            p_valence_3_count_bin_0 :-
                p_valence_3_count(0).
        """,
        "domain": {0, 1}
    },
    357: {
        "predicate": "p_valence_3_count_bin_1",
        "definition": """
            p_valence_3_count_bin_1 :-
                p_valence_3_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    358: {
        "predicate": "p_valence_5plus_count_bin_0",
        "definition": """
            p_valence_5plus_count_bin_0 :-
                p_valence_5plus_count(0).
        """,
        "domain": {0, 1}
    },
    359: {
        "predicate": "p_valence_5plus_count_bin_1",
        "definition": """
            p_valence_5plus_count_bin_1 :-
                p_valence_5plus_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    360: {
        "predicate": "p_valence_5plus_count_bin_2",
        "definition": """
            p_valence_5plus_count_bin_2 :-
                p_valence_5plus_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    361: {
        "predicate": "phenol_bin_0",
        "definition": """
            phenol_bin_0 :-
                phenol_count(0).
        """,
        "domain": {0, 1}
    },
    362: {
        "predicate": "phenol_bin_1",
        "definition": """
            phenol_bin_1 :-
                phenol_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    363: {
        "predicate": "phosphate_bin_0",
        "definition": """
            phosphate_bin_0 :-
                phosphate_count(0).
        """,
        "domain": {0, 1}
    },
    364: {
        "predicate": "phosphate_bin_1",
        "definition": """
            phosphate_bin_1 :-
                phosphate_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    365: {
        "predicate": "phosphate_bin_2",
        "definition": """
            phosphate_bin_2 :-
                phosphate_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    366: {
        "predicate": "phosphonate_bin_0",
        "definition": """
            phosphonate_bin_0 :-
                phosphonate_count(0).
        """,
        "domain": {0, 1}
    },
    367: {
        "predicate": "phosphonate_bin_1",
        "definition": """
            phosphonate_bin_1 :-
                phosphonate_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    368: {
        "predicate": "ring_count_bin_0",
        "definition": """
            ring_count_bin_0 :-
                ring_count(0).
        """,
        "domain": {0, 1}
    },
    369: {
        "predicate": "ring_count_bin_1",
        "definition": """
            ring_count_bin_1 :-
                ring_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    370: {
        "predicate": "ring_count_bin_2",
        "definition": """
            ring_count_bin_2 :-
                ring_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    371: {
        "predicate": "ring_size_3_count_bin_0",
        "definition": """
            ring_size_3_count_bin_0 :-
                ring_size_3_count(0).
        """,
        "domain": {0, 1}
    },
    372: {
        "predicate": "ring_size_3_count_bin_1",
        "definition": """
            ring_size_3_count_bin_1 :-
                ring_size_3_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    373: {
        "predicate": "ring_size_4_count_bin_0",
        "definition": """
            ring_size_4_count_bin_0 :-
                ring_size_4_count(0).
        """,
        "domain": {0, 1}
    },
    374: {
        "predicate": "ring_size_4_count_bin_1",
        "definition": """
            ring_size_4_count_bin_1 :-
                ring_size_4_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    375: {
        "predicate": "ring_size_5_count_bin_0",
        "definition": """
            ring_size_5_count_bin_0 :-
                ring_size_5_count(0).
        """,
        "domain": {0, 1}
    },
    376: {
        "predicate": "ring_size_5_count_bin_1",
        "definition": """
            ring_size_5_count_bin_1 :-
                ring_size_5_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    377: {
        "predicate": "ring_size_6_count_bin_0",
        "definition": """
            ring_size_6_count_bin_0 :-
                ring_size_6_count(0).
        """,
        "domain": {0, 1}
    },
    378: {
        "predicate": "ring_size_6_count_bin_1",
        "definition": """
            ring_size_6_count_bin_1 :-
                ring_size_6_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    379: {
        "predicate": "ring_size_6_count_bin_2",
        "definition": """
            ring_size_6_count_bin_2 :-
                ring_size_6_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    380: {
        "predicate": "ring_size_7plus_count_bin_0",
        "definition": """
            ring_size_7plus_count_bin_0 :-
                ring_size_7plus_count(0).
        """,
        "domain": {0, 1}
    },
    381: {
        "predicate": "ring_size_7plus_count_bin_1",
        "definition": """
            ring_size_7plus_count_bin_1 :-
                ring_size_7plus_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    382: {
        "predicate": "ring_system_count_bin_0",
        "definition": """
            ring_system_count_bin_0 :-
                ring_system_count(0).
        """,
        "domain": {0, 1}
    },
    383: {
        "predicate": "ring_system_count_bin_1",
        "definition": """
            ring_system_count_bin_1 :-
                ring_system_count(N),
                N > 0,
                N <= 1.
        """,
        "domain": {0, 1}
    },
    384: {
        "predicate": "ring_system_count_bin_2",
        "definition": """
            ring_system_count_bin_2 :-
                ring_system_count(N),
                N > 1.
        """,
        "domain": {0, 1}
    },
    385: {
        "predicate": "rotatable_bond_count_bin_0",
        "definition": """
            rotatable_bond_count_bin_0 :-
                rotatable_bond_count(N),
                N <= 30.
        """,
        "domain": {0, 1}
    },
    386: {
        "predicate": "rotatable_bond_count_bin_1",
        "definition": """
            rotatable_bond_count_bin_1 :-
                rotatable_bond_count(N),
                N > 30,
                N <= 37.
        """,
        "domain": {0, 1}
    },
    387: {
        "predicate": "rotatable_bond_count_bin_2",
        "definition": """
            rotatable_bond_count_bin_2 :-
                rotatable_bond_count(N),
                N > 37,
                N <= 42.
        """,
        "domain": {0, 1}
    },
    388: {
        "predicate": "rotatable_bond_count_bin_3",
        "definition": """
            rotatable_bond_count_bin_3 :-
                rotatable_bond_count(N),
                N > 42,
                N <= 50.
        """,
        "domain": {0, 1}
    },
    389: {
        "predicate": "rotatable_bond_count_bin_4",
        "definition": """
            rotatable_bond_count_bin_4 :-
                rotatable_bond_count(N),
                N > 50,
                N <= 57.
        """,
        "domain": {0, 1}
    },
    390: {
        "predicate": "rotatable_bond_count_bin_5",
        "definition": """
            rotatable_bond_count_bin_5 :-
                rotatable_bond_count(N),
                N > 57.
        """,
        "domain": {0, 1}
    },
    391: {
        "predicate": "s_valence_2_count_bin_0",
        "definition": """
            s_valence_2_count_bin_0 :-
                s_valence_2_count(0).
        """,
        "domain": {0, 1}
    },
    392: {
        "predicate": "s_valence_2_count_bin_1",
        "definition": """
            s_valence_2_count_bin_1 :-
                s_valence_2_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    393: {
        "predicate": "s_valence_4_count_bin_0",
        "definition": """
            s_valence_4_count_bin_0 :-
                s_valence_4_count(0).
        """,
        "domain": {0, 1}
    },
    394: {
        "predicate": "s_valence_4_count_bin_1",
        "definition": """
            s_valence_4_count_bin_1 :-
                s_valence_4_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    395: {
        "predicate": "s_valence_6plus_count_bin_0",
        "definition": """
            s_valence_6plus_count_bin_0 :-
                s_valence_6plus_count(0).
        """,
        "domain": {0, 1}
    },
    396: {
        "predicate": "s_valence_6plus_count_bin_1",
        "definition": """
            s_valence_6plus_count_bin_1 :-
                s_valence_6plus_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    397: {
        "predicate": "spiro_atom_count_bin_0",
        "definition": """
            spiro_atom_count_bin_0 :-
                spiro_atom_count(0).
        """,
        "domain": {0, 1}
    },
    398: {
        "predicate": "spiro_atom_count_bin_1",
        "definition": """
            spiro_atom_count_bin_1 :-
                spiro_atom_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    # TODO: Stereo bins currently reflect placeholder stereo counts because
    # smiles.lp does not provide E/Z stereo annotations yet.
    399: {
        "predicate": "stereo_double_bond_E_count_bin_0",
        "definition": """
            stereo_double_bond_E_count_bin_0 :-
                stereo_double_bond_E_count(0).
        """,
        "domain": {0, 1}
    },
    400: {
        "predicate": "stereo_double_bond_E_count_bin_1",
        "definition": """
            stereo_double_bond_E_count_bin_1 :-
                stereo_double_bond_E_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    401: {
        "predicate": "stereo_double_bond_Z_count_bin_0",
        "definition": """
            stereo_double_bond_Z_count_bin_0 :-
                stereo_double_bond_Z_count(0).
        """,
        "domain": {0, 1}
    },
    402: {
        "predicate": "stereo_double_bond_Z_count_bin_1",
        "definition": """
            stereo_double_bond_Z_count_bin_1 :-
                stereo_double_bond_Z_count(N),
                N > 0,
                N <= 2.
        """,
        "domain": {0, 1}
    },
    403: {
        "predicate": "stereo_double_bond_Z_count_bin_2",
        "definition": """
            stereo_double_bond_Z_count_bin_2 :-
                stereo_double_bond_Z_count(N),
                N > 2,
                N <= 4.
        """,
        "domain": {0, 1}
    },
    404: {
        "predicate": "stereo_double_bond_Z_count_bin_3",
        "definition": """
            stereo_double_bond_Z_count_bin_3 :-
                stereo_double_bond_Z_count(N),
                N > 4,
                N <= 6.
        """,
        "domain": {0, 1}
    },
    405: {
        "predicate": "stereo_double_bond_Z_count_bin_4",
        "definition": """
            stereo_double_bond_Z_count_bin_4 :-
                stereo_double_bond_Z_count(N),
                N > 6,
                N <= 9.
        """,
        "domain": {0, 1}
    },
    406: {
        "predicate": "stereo_double_bond_Z_count_bin_5",
        "definition": """
            stereo_double_bond_Z_count_bin_5 :-
                stereo_double_bond_Z_count(N),
                N > 9.
        """,
        "domain": {0, 1}
    },
    407: {
        "predicate": "stereo_double_bond_unspecified_count_bin_0",
        "definition": """
            stereo_double_bond_unspecified_count_bin_0 :-
                stereo_double_bond_unspecified_count(N),
                N <= 2.
        """,
        "domain": {0, 1}
    },
    408: {
        "predicate": "stereo_double_bond_unspecified_count_bin_1",
        "definition": """
            stereo_double_bond_unspecified_count_bin_1 :-
                stereo_double_bond_unspecified_count(N),
                N > 2,
                N <= 3.
        """,
        "domain": {0, 1}
    },
    409: {
        "predicate": "stereo_double_bond_unspecified_count_bin_2",
        "definition": """
            stereo_double_bond_unspecified_count_bin_2 :-
                stereo_double_bond_unspecified_count(N),
                N > 3,
                N <= 4.
        """,
        "domain": {0, 1}
    },
    410: {
        "predicate": "stereo_double_bond_unspecified_count_bin_3",
        "definition": """
            stereo_double_bond_unspecified_count_bin_3 :-
                stereo_double_bond_unspecified_count(N),
                N > 4.
        """,
        "domain": {0, 1}
    },
    411: {
        "predicate": "sulfate_bin_0",
        "definition": """
            sulfate_bin_0 :-
                sulfate_count(0).
        """,
        "domain": {0, 1}
    },
    412: {
        "predicate": "sulfate_bin_1",
        "definition": """
            sulfate_bin_1 :-
                sulfate_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    413: {
        "predicate": "sulfonamide_bin_0",
        "definition": """
            sulfonamide_bin_0 :-
                sulfonamide_count(0).
        """,
        "domain": {0, 1}
    },
    414: {
        "predicate": "sulfonamide_bin_1",
        "definition": """
            sulfonamide_bin_1 :-
                sulfonamide_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    415: {
        "predicate": "sulfoxide_bin_0",
        "definition": """
            sulfoxide_bin_0 :-
                sulfoxide_count(0).
        """,
        "domain": {0, 1}
    },
    416: {
        "predicate": "sulfoxide_bin_1",
        "definition": """
            sulfoxide_bin_1 :-
                sulfoxide_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    417: {
        "predicate": "total_atom_count_with_h_bin_0",
        "definition": """
            total_atom_count_with_h_bin_0 :-
                total_atom_count_with_h(N),
                N <= 110.
        """,
        "domain": {0, 1}
    },
    418: {
        "predicate": "total_atom_count_with_h_bin_1",
        "definition": """
            total_atom_count_with_h_bin_1 :-
                total_atom_count_with_h(N),
                N > 110,
                N <= 129.
        """,
        "domain": {0, 1}
    },
    419: {
        "predicate": "total_atom_count_with_h_bin_2",
        "definition": """
            total_atom_count_with_h_bin_2 :-
                total_atom_count_with_h(N),
                N > 129,
                N <= 147.
        """,
        "domain": {0, 1}
    },
    420: {
        "predicate": "total_atom_count_with_h_bin_3",
        "definition": """
            total_atom_count_with_h_bin_3 :-
                total_atom_count_with_h(N),
                N > 147,
                N <= 173.
        """,
        "domain": {0, 1}
    },
    421: {
        "predicate": "total_atom_count_with_h_bin_4",
        "definition": """
            total_atom_count_with_h_bin_4 :-
                total_atom_count_with_h(N),
                N > 173,
                N <= 191.
        """,
        "domain": {0, 1}
    },
    422: {
        "predicate": "total_atom_count_with_h_bin_5",
        "definition": """
            total_atom_count_with_h_bin_5 :-
                total_atom_count_with_h(N),
                N > 191.
        """,
        "domain": {0, 1}
    },
    423: {
        "predicate": "triple_bond_count_bin_0",
        "definition": """
            triple_bond_count_bin_0 :-
                triple_bond_count(0).
        """,
        "domain": {0, 1}
    },
    424: {
        "predicate": "triple_bond_count_bin_1",
        "definition": """
            triple_bond_count_bin_1 :-
                triple_bond_count(N),
                N > 0.
        """,
        "domain": {0, 1}
    },
    425: {
        "predicate": "unassigned_chiral_center_count_bin_0",
        "definition": """
            unassigned_chiral_center_count_bin_0 :-
                unassigned_chiral_center_count(N),
                N <= 2.
        """,
        "domain": {0, 1}
    },
    426: {
        "predicate": "unassigned_chiral_center_count_bin_1",
        "definition": """
            unassigned_chiral_center_count_bin_1 :-
                unassigned_chiral_center_count(N),
                N > 2,
                N <= 3.
        """,
        "domain": {0, 1}
    },
    427: {
        "predicate": "unassigned_chiral_center_count_bin_2",
        "definition": """
            unassigned_chiral_center_count_bin_2 :-
                unassigned_chiral_center_count(N),
                N > 3,
                N <= 6.
        """,
        "domain": {0, 1}
    },
    428: {
        "predicate": "unassigned_chiral_center_count_bin_3",
        "definition": """
            unassigned_chiral_center_count_bin_3 :-
                unassigned_chiral_center_count(N),
                N > 6.
        """,
        "domain": {0, 1}
    },
    
}

FEATURE_VECTOR_DEFINITIONS_CONTINUOUS: Dict[str, ASPContinuousPredicate] = {
    "alcohol":                              {"predicate": "alcohol_count",                          "definition": HELPER_PREDICATES["alcohol_count"]},
    "aliphatic_ring_count":                 {"predicate": "aliphatic_ring_count",                   "definition": HELPER_PREDICATES["aliphatic_ring_count"]},
    "branch_point_count":                   {"predicate": "branch_point_count",                     "definition": HELPER_PREDICATES["branch_point_count"]},
    "chiral_center_count":                  {"predicate": "chiral_center_count",                    "definition": HELPER_PREDICATES["chiral_center_count"]},
    "degree_1_atom_count":                  {"predicate": "degree_1_atom_count",                    "definition": HELPER_PREDICATES["degree_1_atom_count"]},
    "degree_2_atom_count":                  {"predicate": "degree_2_atom_count",                    "definition": HELPER_PREDICATES["degree_2_atom_count"]},
    "degree_3_atom_count":                  {"predicate": "degree_3_atom_count",                    "definition": HELPER_PREDICATES["degree_3_atom_count"]},
    "degree_4plus_atom_count":              {"predicate": "degree_4plus_atom_count",                "definition": HELPER_PREDICATES["degree_4plus_atom_count"]},
    "degree_of_unsaturation":               {"predicate": "degree_of_unsaturation",                 "definition": HELPER_PREDICATES["degree_of_unsaturation"]},
    "double_bond_count":                    {"predicate": "double_bond_count",                      "definition": HELPER_PREDICATES["double_bond_count"]},
    "ester":                                {"predicate": "ester_count",                            "definition": HELPER_PREDICATES["ester_count"]},
    "ether":                                {"predicate": "ether_count",                            "definition": HELPER_PREDICATES["ether_count"]},
    "exact_mass_da":                        {"predicate": "exact_mass_da",                          "definition": HELPER_PREDICATES["exact_mass_da"]},
    "heavy_atom_count":                     {"predicate": "heavy_atom_count",                       "definition": HELPER_PREDICATES["heavy_atom_count"]},
    "heteroatom_count":                     {"predicate": "heteroatom_count",                       "definition": HELPER_PREDICATES["heteroatom_count"]},
    "heteroatom_exocyclic_count":           {"predicate": "heteroatom_exocyclic_count",             "definition": HELPER_PREDICATES["heteroatom_exocyclic_count"]},
    "heteroatom_in_ring_count":             {"predicate": "heteroatom_in_ring_count",               "definition": HELPER_PREDICATES["heteroatom_in_ring_count"]},
    "molecular_weight_da":                  {"predicate": "molecular_weight_da",                    "definition": HELPER_PREDICATES["molecular_weight_da"]},
    "n_valence_4plus_count":                {"predicate": "n_valence_4plus_count",                  "definition": HELPER_PREDICATES["n_valence_4plus_count"]},
    "o_valence_1_count":                    {"predicate": "o_valence_1_count",                      "definition": HELPER_PREDICATES["o_valence_1_count"]},
    "o_valence_2_count":                    {"predicate": "o_valence_2_count",                      "definition": HELPER_PREDICATES["o_valence_2_count"]},
    "p_valence_5plus_count":                {"predicate": "p_valence_5plus_count",                  "definition": HELPER_PREDICATES["p_valence_5plus_count"]},
    "phosphate":                            {"predicate": "phosphate_count",                        "definition": HELPER_PREDICATES["phosphate_count"]},
    "ring_count":                           {"predicate": "ring_count",                             "definition": HELPER_PREDICATES["ring_count"]},
    "ring_size_6_count":                    {"predicate": "ring_size_6_count",                      "definition": HELPER_PREDICATES["ring_size_6_count"]},
    "ring_system_count":                    {"predicate": "ring_system_count",                      "definition": HELPER_PREDICATES["ring_system_count"]},
    "rotatable_bond_count":                 {"predicate": "rotatable_bond_count",                   "definition": HELPER_PREDICATES["rotatable_bond_count"]},
    "stereo_double_bond_Z_count":           {"predicate": "stereo_double_bond_Z_count",             "definition": HELPER_PREDICATES["stereo_double_bond_Z_count"]},
    "stereo_double_bond_unspecified_count": {"predicate": "stereo_double_bond_unspecified_count",   "definition": HELPER_PREDICATES["stereo_double_bond_unspecified_count"]},
    "total_atom_count_with_h":              {"predicate": "total_atom_count_with_h",                "definition": HELPER_PREDICATES["total_atom_count_with_h"]},
    "unassigned_chiral_center_count":       {"predicate": "unassigned_chiral_center_count",         "definition": HELPER_PREDICATES["unassigned_chiral_center_count"]},
    "aldehyde":                             {"predicate": "aldehyde_count",                         "definition": HELPER_PREDICATES["aldehyde_count"]},
    "amide":                                {"predicate": "amide_count",                             "definition": HELPER_PREDICATES["amide_count"]},
    "amine":                                {"predicate": "amine_count",                             "definition": HELPER_PREDICATES["amine_count"]},
    "aniline":                              {"predicate": "aniline_count",                           "definition": HELPER_PREDICATES["aniline_count"]},
    "aromatic_atom_count":                  {"predicate": "aromatic_atom_count",                     "definition": HELPER_PREDICATES["aromatic_atom_count"]},
    "aromatic_heteroatom_count":            {"predicate": "aromatic_heteroatom_count",               "definition": HELPER_PREDICATES["aromatic_heteroatom_count"]},
    "aromatic_n_count":                     {"predicate": "aromatic_n_count",                        "definition": HELPER_PREDICATES["aromatic_n_count"]},
    "aromatic_o_count":                     {"predicate": "aromatic_o_count",                        "definition": HELPER_PREDICATES["aromatic_o_count"]},
    "aromatic_p_count":                     {"predicate": "aromatic_p_count",                        "definition": HELPER_PREDICATES["aromatic_p_count"]},
    "aromatic_ring":                        {"predicate": "aromatic_ring_count",                     "definition": HELPER_PREDICATES["aromatic_ring_count"]},
    "aromatic_ring_count":                  {"predicate": "aromatic_ring_count",                     "definition": HELPER_PREDICATES["aromatic_ring_count"]},
    "aromatic_s_count":                     {"predicate": "aromatic_s_count",                        "definition": HELPER_PREDICATES["aromatic_s_count"]},
    "assigned_chiral_center_count":         {"predicate": "assigned_chiral_center_count",            "definition": HELPER_PREDICATES["assigned_chiral_center_count"]},
    "bridgehead_atom_count":                {"predicate": "bridgehead_atom_count",                   "definition": HELPER_PREDICATES["bridgehead_atom_count"]},
    "carboxylic_acid":                      {"predicate": "carboxylic_acid_count",                   "definition": HELPER_PREDICATES["carboxylic_acid_count"]},
    "connected_component_count":            {"predicate": "connected_component_count",               "definition": HELPER_PREDICATES["connected_component_count"]},
    "element_B":                            {"predicate": "element_B_count",                         "definition": HELPER_PREDICATES["element_B_count"]},
    "element_Br":                           {"predicate": "element_Br_count",                        "definition": HELPER_PREDICATES["element_Br_count"]},
    "element_C":                            {"predicate": "element_C_count",                         "definition": HELPER_PREDICATES["element_C_count"]},
    "element_Cl":                           {"predicate": "element_Cl_count",                        "definition": HELPER_PREDICATES["element_Cl_count"]},
    "element_F":                            {"predicate": "element_F_count",                         "definition": HELPER_PREDICATES["element_F_count"]},
    "element_H":                            {"predicate": "element_H_count",                         "definition": HELPER_PREDICATES["element_H_count"]},
    "element_I":                            {"predicate": "element_I_count",                         "definition": HELPER_PREDICATES["element_I_count"]},
    "element_N":                            {"predicate": "element_N_count",                         "definition": HELPER_PREDICATES["element_N_count"]},
    "element_O":                            {"predicate": "element_O_count",                         "definition": HELPER_PREDICATES["element_O_count"]},
    "element_P":                            {"predicate": "element_P_count",                         "definition": HELPER_PREDICATES["element_P_count"]},
    "element_S":                            {"predicate": "element_S_count",                         "definition": HELPER_PREDICATES["element_S_count"]},
    "element_Si":                           {"predicate": "element_Si_count",                        "definition": HELPER_PREDICATES["element_Si_count"]},
    "formal_charge":                        {"predicate": "formal_charge",                           "definition": HELPER_PREDICATES["formal_charge"]},
    "fused_ring_pair_count":                {"predicate": "fused_ring_pair_count",                   "definition": HELPER_PREDICATES["fused_ring_pair_count"]},
    "fused_ring_system_count":              {"predicate": "fused_ring_system_count",                 "definition": HELPER_PREDICATES["fused_ring_system_count"]},
    "halogenated":                          {"predicate": "halogenated_count",                       "definition": HELPER_PREDICATES["halogenated_count"]},
    "imide":                                {"predicate": "imide",                                   "definition": HELPER_PREDICATES["imide"]},
    "ketone":                               {"predicate": "ketone",                                  "definition": HELPER_PREDICATES["ketone"]},
    "lactam":                               {"predicate": "lactam",                                  "definition": HELPER_PREDICATES["lactam"]},
    "lactone":                              {"predicate": "lactone",                                 "definition": HELPER_PREDICATES["lactone"]},
    "n_valence_1_count":                    {"predicate": "n_valence_1_count",                       "definition": HELPER_PREDICATES["n_valence_1_count"]},
    "n_valence_2_count":                    {"predicate": "n_valence_2_count",                       "definition": HELPER_PREDICATES["n_valence_2_count"]},
    "n_valence_3_count":                    {"predicate": "n_valence_3_count",                       "definition": HELPER_PREDICATES["n_valence_3_count"]},
    "nitrile":                              {"predicate": "nitrile_count",                           "definition": HELPER_PREDICATES["nitrile_count"]},
    "nitro":                                {"predicate": "nitro_count",                             "definition": HELPER_PREDICATES["nitro_count"]},
    "o_valence_3plus_count":                {"predicate": "o_valence_3plus_count",                   "definition": HELPER_PREDICATES["o_valence_3plus_count"]},
    "p_valence_3_count":                    {"predicate": "p_valence_3_count",                       "definition": HELPER_PREDICATES["p_valence_3_count"]},
    "phenol":                               {"predicate": "phenol_count",                            "definition": HELPER_PREDICATES["phenol_count"]},
    "phosphonate":                          {"predicate": "phosphonate_count",                       "definition": HELPER_PREDICATES["phosphonate_count"]},
    "ring_size_3_count":                    {"predicate": "ring_size_3_count",                       "definition": HELPER_PREDICATES["ring_size_3_count"]},
    "ring_size_4_count":                    {"predicate": "ring_size_4_count",                       "definition": HELPER_PREDICATES["ring_size_4_count"]},
    "ring_size_5_count":                    {"predicate": "ring_size_5_count",                       "definition": HELPER_PREDICATES["ring_size_5_count"]},
    "ring_size_7plus_count":                {"predicate": "ring_size_7plus_count",                   "definition": HELPER_PREDICATES["ring_size_7plus_count"]},
    "s_valence_2_count":                    {"predicate": "s_valence_2_count",                       "definition": HELPER_PREDICATES["s_valence_2_count"]},
    "s_valence_4_count":                    {"predicate": "s_valence_4_count",                       "definition": HELPER_PREDICATES["s_valence_4_count"]},
    "s_valence_6plus_count":                {"predicate": "s_valence_6plus_count",                   "definition": HELPER_PREDICATES["s_valence_6plus_count"]},
    "spiro_atom_count":                     {"predicate": "spiro_atom_count",                        "definition": HELPER_PREDICATES["spiro_atom_count"]},
    "stereo_double_bond_E_count":           {"predicate": "stereo_double_bond_E_count",              "definition": HELPER_PREDICATES["stereo_double_bond_E_count"]},
    "sulfate":                              {"predicate": "sulfate_count",                           "definition": HELPER_PREDICATES["sulfate_count"]},
    "sulfonamide":                          {"predicate": "sulfonamide_count",                       "definition": HELPER_PREDICATES["sulfonamide_count"]},
    "sulfoxide":                            {"predicate": "sulfoxide_count",                         "definition": HELPER_PREDICATES["sulfoxide_count"]},
    "triple_bond_count":                    {"predicate": "triple_bond_count",                       "definition": HELPER_PREDICATES["triple_bond_count"]},
}

COUNT_KEYS = ['num_carbons', 'num_hydrogens', 'num_nitrogens', 'num_oxygens', 'num_sulfurs', 'num_phosphorus']

def KEYS():
    maccs_keys = set(MACCS_DEFINITIONS.keys())
    feature_vector_keys = set(FEATURE_VECTOR_DEFINITIONS.keys())
    count_keys = set(COUNT_KEYS)
    continuous_keys = set(FEATURE_VECTOR_DEFINITIONS_CONTINUOUS.keys())

    assert maccs_keys.isdisjoint(feature_vector_keys), "MACCS keys and feature vector keys must be disjoint"
    assert maccs_keys.isdisjoint(count_keys), "MACCS keys and count keys must be disjoint"
    assert feature_vector_keys.isdisjoint(count_keys), "Feature vector keys and count keys must be disjoint"

    return maccs_keys.union(feature_vector_keys).union(count_keys).union(continuous_keys)
