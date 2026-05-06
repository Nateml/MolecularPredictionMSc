# STRUCTURAL_VECTOR_SCHEMA

This document records the order and grouping of entries in `STRUCTURAL_VECTOR_ORDER` (see `Data Processing/MoNa/add_structural_features.py`). It maps each index (0-based) in the `structural_vector` to a feature name, indicates whether the value is continuous or an integer-count, and groups related categorical/count features together.

## Summary
- Structural vector length: 89 (indices 0..88)
- Continuous (float): indices 0-1
- Integer/count features (small-range; treated as categorical/counts in the dataset): indices 2-88

## Index mapping (0-based)

0: exact_mass_da (continuous)
1: molecular_weight_da (continuous)

2: heavy_atom_count
3: total_atom_count_with_h
4: connected_component_count
5: heteroatom_count

6: ring_count
7: ring_size_3_count
8: ring_size_4_count
9: ring_size_5_count
10: ring_size_6_count
11: ring_size_7plus_count
12: fused_ring_pair_count
13: ring_system_count
14: fused_ring_system_count
15: aromatic_ring_count
16: aliphatic_ring_count

17: aromatic_atom_count
18: aromatic_heteroatom_count
19: aromatic_n_count
20: aromatic_o_count
21: aromatic_s_count
22: aromatic_p_count

23: heteroatom_in_ring_count
24: heteroatom_exocyclic_count

25: double_bond_count
26: triple_bond_count
27: stereo_double_bond_E_count
28: stereo_double_bond_Z_count
29: stereo_double_bond_unspecified_count

30: chiral_center_count
31: assigned_chiral_center_count
32: unassigned_chiral_center_count

33: rotatable_bond_count
34: degree_of_unsaturation
35: branch_point_count

36: degree_1_atom_count
37: degree_2_atom_count
38: degree_3_atom_count
39: degree_4plus_atom_count

40: n_valence_1_count
41: n_valence_2_count
42: n_valence_3_count
43: n_valence_4plus_count

44: o_valence_1_count
45: o_valence_2_count
46: o_valence_3plus_count

47: s_valence_2_count
48: s_valence_4_count
49: s_valence_6plus_count

50: p_valence_3_count
51: p_valence_5plus_count

52: spiro_atom_count
53: bridgehead_atom_count
54: formal_charge

55: element_C
56: element_H
57: element_N
58: element_O
59: element_P
60: element_S
61: element_F
62: element_Cl
63: element_Br
64: element_I
65: element_Si
66: element_B

Substructure match counts (each typically 0/1 or small integer):
67: carboxylic_acid
68: ester
69: amide
70: amine
71: alcohol
72: ether
73: aldehyde
74: ketone
75: nitro
76: phosphate
77: sulfate
78: halogenated
79: aromatic_ring
80: nitrile
81: sulfonamide
82: sulfoxide
83: phosphonate
84: imide
85: lactam
86: lactone
87: phenol
88: aniline

---

## Groupings (features that belong together semantically)
- Continuous measurements:
  - indices 0–1: `exact_mass_da`, `molecular_weight_da` (float)

- Atom / count group:
  - indices 2–5: general atom counts and components

- Ring topology group:
  - indices 6–16: ring counts and ring topology features

- Aromatic / aromatic-element counts:
  - indices 17–22: aromatic atom and heteroatom counts (N/O/S/P breakdown)

- Heteroatom location:
  - indices 23–24

- Bond / stereochemistry / chiral:
  - indices 25–33

- Degree / branching:
  - indices 34–39

- Valence-by-element groups:
  - N valence counts: indices 40–43
  - O valence counts: indices 44–46
  - S valence counts: indices 47–49
  - P valence counts: indices 50–51

- Special topological counts:
  - indices 52–54: spiro / bridgehead / formal charge

- Element composition (counts):
  - indices 55–66: element_C, element_H, ..., element_B

- Substructure match counts (logical motifs):
  - indices 67–88: each SMARTS-based motif from `SUBSTRUCTURE_PATTERNS`

## Notes / usage
- In the ML pipeline these integer/count features are usually treated as categorical (one-hot or integer classes) when they have a small finite range. Use `categorical_sizes_from_df()` on your dataset to derive exact class counts per feature.
- The substructure matches (indices 67–88) are often binary (0/1) but are stored as counts in case multiple matches occur; they can be binarised if desired.
- The element counts (indices 55–66) and many of the `*_count` features are natural candidates for categorical encoding if their observed range is small (e.g., 0–10). When creating `categorical_sizes`, ensure the DataFrame columns are integer-coded starting at 0 so `categorical_sizes_from_df()` computes `max+1` correctly.

If you want, I can:
- add a machine-readable JSON mapping (index → name → group), or
- change this file location or format (e.g., put in `neural_net/`), or
- generate the `categorical_sizes` automatically for the current dataset file you have.
