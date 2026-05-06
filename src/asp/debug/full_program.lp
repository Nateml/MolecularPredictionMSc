min_main_chain_len(3).

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
    

        adjacent_single(I, J) :-
            parent(I, _, J),
            not is_multi_bond(J).
    

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
    

        aromatic_s_count(N) :-
            N = #count { I : aromatic_atom(I), smiles_symbol(I, "S") }.
    

        alcohol_atom(O) :-
            single_bond_between(C, O),
            symbol(C, "C"),
            symbol(O, "O"),
            free_bonds(O, 1).
    

        aromatic_o_count(N) :-
            N = #count { I : aromatic_atom(I), symbol(I, "O") }.
    

        carboxylic_acid_count(N) :-
            N = #count { C : carboxylic_acid_atom(C) }.
    

        s_valence_4_count(N) :-
            N = #count { I : symbol(I, "S4") }.
    

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
    

        % E/Z stereo is currently not represented in smiles.lp.
        stereo_double_bond_Z_count(0).
    

        aliphatic_ring_count(N) :-
            N = #count { CYCLE : cycle_start(_, CYCLE) }.
    

        ester_count(N) :-
            N = #count { C : ester_atom(C) }.
    

        phenol_atom(O) :-
            symbol(O, "O"),
            symbol(C, "C"),
            single_bond_between(O, C),
            free_bonds(O, 1),
            aromatic_atom(C).
    

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
    

        sulfate_count(N) :- sulfate_count_value(N).
    

        ring_size_5_count(N) :-
            N = #count {
                CYCLE :
                    cycle_start(I1, CYCLE),
                    cycle_end(I2, CYCLE),
                    I2 - I1 + 1 = 5
            }.
    

        heteroatom_count(N) :-
            N = #count { I : symbol(I, E), E != "C", E != "H" }.
    

        element_Si_count(N) :-
            N = #count { I : smiles_symbol(I, "Si") }.
    

        is_multi_bond(J) :- multi_bond(J, _).
    

        branch_point_atom(I) :-
            branching(1, B),
            B >= 3,
            I = 1.
        branch_point_atom(I) :-
            branching(I, B),
            I > 1,
            B >= 2.
    

        atom_degree(I, D) :-
            symbol(I, _),
            D = #count { J : heavy_neighbor(I, J) }.
    

        nitro_atom(N) :-
            symbol(N, "N"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            O1 != O2,
            double_bond_between(N, O1),
            single_bond_between(N, O2).
    

        unassigned_chiral_center_count(0).
    

        element_B_count(N) :-
            N = #count { I : smiles_symbol(I, "B") }.
    

        sulfoxide_count_value(N) :-
            N = #count { S : sulfoxide_atom(S) }.
    

        n_valence_1_count(N) :-
            N = #count { I : symbol(I, "N"), free_bonds(I, 2) }.
    

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
    

        triple_bond_count(N) :-
            N = #count { J : multi_bond(J, 3) }.
    

        degree_of_unsaturation(N) :-
            double_bond_count(DB),
            triple_bond_count(TB),
            ring_count(R),
            N = DB + 2 * TB + R.
    

        s_valence_2_count(N) :-
            N = #count { I : symbol(I, "S2") }.
    

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
    

        aromatic_ring_count(N) :-
            N = #count { CYCLE : cycle_start(_, CYCLE) }.
    

        assigned_chiral_center_count(N) :-
            chiral_center_count(N).
    

        phosphate_count_value(N) :-
            N = #count { P : phosphate_atom(P) }.
    

        exact_mass_da(M) :-
            M = 12 * c + 1 * h + 14 * n + 16 * o.
    

        n_valence_2_count(N) :-
            N = #count { I : symbol(I, "N"), free_bonds(I, 1) }.
    

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
    

        total_atom_count_with_h(N) :-
            heavy_atom_count(HA),
            element_H_count(H),
            N = HA + H.
    

        nitro_count_value(N) :-
            N = #count { I : nitro_atom(I) }.
    

        aniline_count(N) :-
            N = #count { I : aniline_atom(I) }.
    

        n_valence_3_count(N) :-
            N = #count { I : symbol(I, "N"), free_bonds(I, 0) }.
    

        ring_size_3_count(N) :-
            N = #count {
                CYCLE :
                    cycle_start(I1, CYCLE),
                    cycle_end(I2, CYCLE),
                    I2 - I1 + 1 = 3
            }.
    

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
    

        p_valence_3_count(N) :-
            N = #count { I : symbol(I, "P"), free_bonds(I, 0) }.
    

        sulfonamide_count_value(N) :-
            N = #count { S : sulfonamide_atom(S) }.
    

        double_bond_between(I, J) :-
            parent(I, _, J),
            multi_bond(J, 2).
        double_bond_between(I, J) :-
            parent(J, _, I),
            multi_bond(I, 2).
    

        alcohol_count(N) :-
            N = #count { O : alcohol_atom(O) }.
    

        o_valence_1_count(N) :-
            N = #count { I : symbol(I, "O"), free_bonds(I, 2) }.
    

        phenol_count_value(N) :-
            N = #count { O : phenol_atom(O) }.
    

        amine_count(N) :-
            N = #count { I : amine_atom(I) }.
    

        ring_size_4_count(N) :-
            N = #count {
                CYCLE :
                    cycle_start(I1, CYCLE),
                    cycle_end(I2, CYCLE),
                    I2 - I1 + 1 = 4
            }.
    

        aromatic_n_count(N) :-
            N = #count { I : aromatic_atom(I), symbol(I, "N") }.
    

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
    

        amine_atom(N) :-
            symbol(N, "N"),
            symbol(C, "C"),
            single_bond_between(N, C),
            not amide_atom(N).
    

        sulfonamide_count(N) :- sulfonamide_count_value(N).
    

        aromatic_atom_count(N) :-
            N = #count { I : aromatic_atom(I) }.
    

        ether_count(N) :-
            N = #count { O : ether_atom(O) }.
    

        aromatic_p_count(N) :-
            N = #count { I : aromatic_atom(I), smiles_symbol(I, "P") }.
    

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
    

        connected_component_count(1).
    

        phosphonate_atom(P) :-
            smiles_symbol(P, "P"),
            symbol(C, "C"),
            symbol(O, "O"),
            single_bond_between(P, C),
            single_bond_between(P, O).
    

        heteroatom_exocyclic_count(N) :-
            N = #count { I : heteroatom(I), not ring_atom(I) }.
    

        double_bond_count(N) :-
            N = #count { J : multi_bond(J, 2) }.
    

        heteroatom_in_ring_count(N) :-
            N = #count { I : heteroatom(I), ring_atom(I) }.
    

        sulfoxide_count(N) :- sulfoxide_count_value(N).
    

        phosphate_atom(P) :-
            smiles_symbol(P, "P"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            double_bond_between(P, O1),
            single_bond_between(P, O2).
    

        nitrile_count(N) :- nitrile_count_value(N).
    

        element_F_count(N) :-
            N = #count { I : smiles_symbol(I, "F") }.
    

        element_C_count(N) :-
            N = #count { I : symbol(I, "C") }.
    

        phosphate_count(N) :- phosphate_count_value(N).
    

        % Without E/Z stereo tags, all modeled double bonds are unspecified.
        stereo_double_bond_unspecified_count(N) :-
            double_bond_count(N).
    

        element_Cl_count(N) :-
            N = #count { I : smiles_symbol(I, "Cl") }.
    

        element_I_count(N) :-
            N = #count { I : smiles_symbol(I, "I") }.
    

        sulfate_count_value(N) :-
            N = #count { S : sulfate_atom(S) }.
    

        branch_point_count(N) :-
            N = #count { I : branch_point_atom(I) }.
    

        amide_count(N) :-
            N = #count { I : amide_atom(I) }.
    

        sulfoxide_atom(S) :-
            smiles_symbol(S, "S"),
            atom_degree(S, 3),
            1 = #count { O : symbol(O, "O"), double_bond_between(S, O) }.
    

        aldehyde_atom(C) :-
            symbol(C, "C"),
            symbol(O, "O"),
            symbol(C2, "C"),
            double_bond_between(C, O),
            single_bond_between(C, C2),
            free_bonds(C, 1).
    

        element_N_count(N) :-
            N = #count { I : symbol(I, "N") }.
    

        degree_3_atom_count(N) :-
            N = #count { I : atom_degree(I, 3) }.
    

        element_S_count(N) :-
            N = #count{ I : smiles_symbol(I, "S") }.
    

        phosphonate_count_value(N) :-
            N = #count { P : phosphonate_atom(P) }.
    

        molecular_weight_da(M) :-
            M = 12 * c + 1 * h + 14 * n + 16 * o.
    

        ring_count(N) :-
            N = #count { CYCLE : cycle_start(_, CYCLE) }.
    

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
    

        element_Br_count(N) :-
            N = #count { I : smiles_symbol(I, "Br") }.
    

        nitro_count(N) :- nitro_count_value(N).
    

        % E/Z stereo is currently not represented in smiles.lp.
        stereo_double_bond_E_count(0).
    

        degree_4plus_atom_count(N) :-
            N = #count { I : atom_degree(I, D), D >= 4 }.
    

        p_valence_5plus_count(N) :-
            N = #count { I : symbol(I, "P"), free_bonds(I, F), F < 0 }.
    

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
    

        aldehyde_count(N) :-
            N = #count { C : aldehyde_atom(C) }.
    

        aromatic_ring :-
            aromatic_ring_count(N),
            N > 0.
    

        phosphonate_count(N) :- phosphonate_count_value(N).
    

        single_bond_between(I, J) :-
            adjacent_single(I, J).

        single_bond_between(I, J) :-
            adjacent_single(J, I).
    

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
    

        o_valence_2_count(N) :-
            N = #count { I : symbol(I, "O"), free_bonds(I, 1) }.
    

        carboxylic_acid_atom(C) :-
            symbol(C, "C"),
            symbol(O1, "O"),
            symbol(O2, "O"),
            O1 != O2,
            double_bond_between(C, O1),
            single_bond_between(C, O2),
            free_bonds(O2, 1).
    

        amide_atom(N) :-
            symbol(N, "N"),
            symbol(C, "C"),
            symbol(O, "O"),
            single_bond_between(C, N),
            double_bond_between(C, O).
    

        ring_size_7plus_count(N) :-
            N = #count {
                CYCLE :
                    cycle_start(I1, CYCLE),
                    cycle_end(I2, CYCLE),
                    I2 - I1 + 1 >= 7
            }.
    

        any_bond_between(I, J) :-
            parent(I, _, J).
        any_bond_between(I, J) :-
            parent(J, _, I).
    

        o_valence_3plus_count(N) :-
            N = #count { I : symbol(I, "O"), free_bonds(I, F), F <= 0 }.
    

        fused_ring_system_count(1) :-
            fused_ring_pair_count(N),
            N > 0.
        fused_ring_system_count(0) :-
            fused_ring_pair_count(0).
    

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
    

        aromatic_heteroatom_count(N) :-
            N = #count { I : aromatic_atom(I), smiles_symbol(I, E), E != "C", E != "H" }.
    

        heteroatom(I) :-
            smiles_symbol(I, E),
            E != "C",
            E != "H".
    

        element_O_count(N) :-
            N = #count { I : symbol(I, "O") }.
    

        n_valence_4plus_count(N) :-
            N = #count { I : symbol(I, "N"), free_bonds(I, F), F < 0 }.
    

        carbonyl_carbon(C) :-
            symbol(C, "C"),
            symbol(O, "O"),
            double_bond_between(C, O).
    

        ring_size_6_count(N) :-
            N = #count {
                CYCLE :
                    cycle_start(I1, CYCLE),
                    cycle_end(I2, CYCLE),
                    I2 - I1 + 1 = 6
            }.
    

        nitrile_atom(C) :-
            symbol(C, "C"),
            symbol(N, "N"),
            triple_bond_between(C, N).
    

        degree_1_atom_count(N) :-
            N = #count { I : atom_degree(I, 1) }.
    

            has_n_o_single_bond :-
                single_bond_between(I, J),
                symbol(I, "N"),
                symbol(J, "O").
        

        s_valence_6plus_count(N) :-
            N = #count { I : symbol(I, "S6") }.
    

        nitrile_count_value(N) :-
            N = #count { C : nitrile_atom(C) }.
    

        element_H_count(N) :-
            molecular_formula("H", N).
        element_H_count(0) :-
            not molecular_formula("H", _).
    

        halogenated :-
            element_F_count(N), N > 0.
        halogenated :-
            element_Cl_count(N), N > 0.
        halogenated :-
            element_Br_count(N), N > 0.
        halogenated :-
            element_I_count(N), N > 0.
    

        degree_2_atom_count(N) :-
            N = #count { I : atom_degree(I, 2) }.
    

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
    

        element_P_count(N) :-
            N = #count{ I : smiles_symbol(I, "P") }.
    

        heavy_atom_count(N) :-
            N = #count { I : symbol(I, E), E != "H" }.
    

        formal_charge(0).
    

        ring_atom(I) :- cycle_start(I, _).
        ring_atom(I) :- cycle_end(I, _).
    

        phenol_count(N) :- phenol_count_value(N).
    

        triple_bond_between(I, J) :-
            parent(I, _, J),
            multi_bond(J, 3).
        triple_bond_between(I, J) :-
            parent(J, _, I),
            multi_bond(I, 3).
    
:- has_n_o_single_bond.