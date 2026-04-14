# Feature Vector Index Map

This file documents index positions for vectors produced by `add_structural_features.py`.

## Vector layout

- `maccs` length: 167
- `structural_vector` length: 89
- `feature_vector` length: 256
- Construction rule: `feature_vector = maccs + structural_vector`

## MACCS index range

- `feature_vector[0..166]` -> `maccs[0..166]`
- Label convention: `maccs_0`, `maccs_1`, ..., `maccs_166`

## Structural feature index map

Columns:
- `struct_idx`: index in `structural_vector`
- `feature_idx`: index in `feature_vector`
- `name`: feature name from `STRUCTURAL_VECTOR_ORDER`
- `description`: short meaning

| struct_idx | feature_idx | name | description |
|---:|---:|---|---|
| 0 | 167 | exact_mass_da | exact molecular mass (Da) |
| 1 | 168 | molecular_weight_da | average molecular weight (Da) |
| 2 | 169 | heavy_atom_count | number of non-hydrogen atoms |
| 3 | 170 | total_atom_count_with_h | total atoms after adding implicit hydrogens |
| 4 | 171 | connected_component_count | disconnected fragments/components in molecule graph |
| 5 | 172 | heteroatom_count | count of heteroatoms |
| 6 | 173 | ring_count | total ring count |
| 7 | 174 | ring_size_3_count | number of 3-membered rings |
| 8 | 175 | ring_size_4_count | number of 4-membered rings |
| 9 | 176 | ring_size_5_count | number of 5-membered rings |
| 10 | 177 | ring_size_6_count | number of 6-membered rings |
| 11 | 178 | ring_size_7plus_count | number of rings size >= 7 |
| 12 | 179 | fused_ring_pair_count | count of ring pairs sharing >= 2 atoms |
| 13 | 180 | ring_system_count | number of ring systems |
| 14 | 181 | fused_ring_system_count | ring systems containing >1 ring |
| 15 | 182 | aromatic_ring_count | aromatic ring count |
| 16 | 183 | aliphatic_ring_count | non-aromatic ring count |
| 17 | 184 | aromatic_atom_count | number of aromatic atoms |
| 18 | 185 | aromatic_heteroatom_count | aromatic atoms that are heteroatoms |
| 19 | 186 | aromatic_n_count | aromatic nitrogen count |
| 20 | 187 | aromatic_o_count | aromatic oxygen count |
| 21 | 188 | aromatic_s_count | aromatic sulfur count |
| 22 | 189 | aromatic_p_count | aromatic phosphorus count |
| 23 | 190 | heteroatom_in_ring_count | heteroatoms located in rings |
| 24 | 191 | heteroatom_exocyclic_count | heteroatoms outside rings |
| 25 | 192 | double_bond_count | count of double bonds |
| 26 | 193 | triple_bond_count | count of triple bonds |
| 27 | 194 | stereo_double_bond_E_count | E-configured double bonds |
| 28 | 195 | stereo_double_bond_Z_count | Z-configured double bonds |
| 29 | 196 | stereo_double_bond_unspecified_count | double bonds with unspecified/any stereo |
| 30 | 197 | chiral_center_count | total chiral centers (assigned + unassigned) |
| 31 | 198 | assigned_chiral_center_count | chiral centers with assigned R/S |
| 32 | 199 | unassigned_chiral_center_count | chiral centers marked unassigned |
| 33 | 200 | rotatable_bond_count | number of rotatable bonds |
| 34 | 201 | degree_of_unsaturation | double + 2*triple + rings |
| 35 | 202 | branch_point_count | heavy atoms with degree >= 3 |
| 36 | 203 | degree_1_atom_count | heavy atoms with degree 1 |
| 37 | 204 | degree_2_atom_count | heavy atoms with degree 2 |
| 38 | 205 | degree_3_atom_count | heavy atoms with degree 3 |
| 39 | 206 | degree_4plus_atom_count | heavy atoms with degree >= 4 |
| 40 | 207 | n_valence_1_count | nitrogen atoms with valence 1 |
| 41 | 208 | n_valence_2_count | nitrogen atoms with valence 2 |
| 42 | 209 | n_valence_3_count | nitrogen atoms with valence 3 |
| 43 | 210 | n_valence_4plus_count | nitrogen atoms with valence >= 4 |
| 44 | 211 | o_valence_1_count | oxygen atoms with valence 1 |
| 45 | 212 | o_valence_2_count | oxygen atoms with valence 2 |
| 46 | 213 | o_valence_3plus_count | oxygen atoms with valence >= 3 |
| 47 | 214 | s_valence_2_count | sulfur atoms with valence 2 |
| 48 | 215 | s_valence_4_count | sulfur atoms with valence 4 |
| 49 | 216 | s_valence_6plus_count | sulfur atoms with valence >= 6 |
| 50 | 217 | p_valence_3_count | phosphorus atoms with valence 3 |
| 51 | 218 | p_valence_5plus_count | phosphorus atoms with valence >= 5 |
| 52 | 219 | spiro_atom_count | number of spiro atoms |
| 53 | 220 | bridgehead_atom_count | number of bridgehead atoms |
| 54 | 221 | formal_charge | formal molecular charge |
| 55 | 222 | element_C | atom count of C (with implicit H expansion) |
| 56 | 223 | element_H | atom count of H (with implicit H expansion) |
| 57 | 224 | element_N | atom count of N (with implicit H expansion) |
| 58 | 225 | element_O | atom count of O (with implicit H expansion) |
| 59 | 226 | element_P | atom count of P (with implicit H expansion) |
| 60 | 227 | element_S | atom count of S (with implicit H expansion) |
| 61 | 228 | element_F | atom count of F (with implicit H expansion) |
| 62 | 229 | element_Cl | atom count of Cl (with implicit H expansion) |
| 63 | 230 | element_Br | atom count of Br (with implicit H expansion) |
| 64 | 231 | element_I | atom count of I (with implicit H expansion) |
| 65 | 232 | element_Si | atom count of Si (with implicit H expansion) |
| 66 | 233 | element_B | atom count of B (with implicit H expansion) |
| 67 | 234 | carboxylic_acid | SMARTS match count for carboxylic acid |
| 68 | 235 | ester | SMARTS match count for ester |
| 69 | 236 | amide | SMARTS match count for amide |
| 70 | 237 | amine | SMARTS match count for amine |
| 71 | 238 | alcohol | SMARTS match count for alcohol |
| 72 | 239 | ether | SMARTS match count for ether |
| 73 | 240 | aldehyde | SMARTS match count for aldehyde |
| 74 | 241 | ketone | SMARTS match count for ketone |
| 75 | 242 | nitro | SMARTS match count for nitro |
| 76 | 243 | phosphate | SMARTS match count for phosphate |
| 77 | 244 | sulfate | SMARTS match count for sulfate |
| 78 | 245 | halogenated | SMARTS match count for halogenated pattern |
| 79 | 246 | aromatic_ring | SMARTS match count for aromatic six-member ring motif |
| 80 | 247 | nitrile | SMARTS match count for nitrile |
| 81 | 248 | sulfonamide | SMARTS match count for sulfonamide |
| 82 | 249 | sulfoxide | SMARTS match count for sulfoxide |
| 83 | 250 | phosphonate | SMARTS match count for phosphonate |
| 84 | 251 | imide | SMARTS match count for imide |
| 85 | 252 | lactam | SMARTS match count for lactam |
| 86 | 253 | lactone | SMARTS match count for lactone |
| 87 | 254 | phenol | SMARTS match count for phenol |
| 88 | 255 | aniline | SMARTS match count for aniline |

## Programmatic naming for feature_vector

If you need machine-readable names for all 256 indices:

- `0..166`: `maccs_0` ... `maccs_166`
- `167..255`: names in the table above
