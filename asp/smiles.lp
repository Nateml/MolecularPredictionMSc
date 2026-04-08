% Source: eval-2024-asp-molecules (commit: 8fe473e) (modified by removing embedded python code)
% URL: https://github.com/knowsys/eval-2024-asp-molecules
% File: smiles.lp
% INPUT: C_{c}H_{h},O_{o}

#const c=0.
#const h=0.
#const n=0.
#const o=0.

molecular_formula("C", c) :- c != 0.
molecular_formula("H", h) :- h != 0.
molecular_formula("N", n) :- n != 0.
molecular_formula("O", o) :- o != 0.

element("C", 6, 4) :- molecular_formula("C", C), C > 0.
element("H", 1, 1) :- molecular_formula("H", H), H > 0.
element("N", 7, 3) :- molecular_formula("N", N), N > 0.
element("O", 8, 2) :- molecular_formula("O", O), O > 0.

% To allow alternative valence values for an element, add e.g. `element_valence("C",2)`.

% A version of this file without comments and without the INPUT section and the embedded Python code (will require setting `main_chain_len` programmatically) was produced with the following command:
% $> sed '1,18d; 60,72d; s/[^%].*@min_main_chain_len.*/1{ main_chain_len(MIN_LEN..ATOM_COUNT) }1 :- non_hydrogen_atom_count(ATOM_COUNT), min_main_chain_len(MIN_LEN)./; /^[ \t]*%/d; s/[ \t]*%.*$//; /^[ \t]*# /d' smiles.lp | cat -s > smiles_min.lp

% A version of this file without comments and the embedded Python code, but with the INPUT section and additional `min_main_chain_len(m)` INPUT was produced with the following command:
% $> sed '60,72d; s/[^%].*@min_main_chain_len.*/1{ main_chain_len(MIN_LEN..ATOM_COUNT) }1 :- non_hydrogen_atom_count(ATOM_COUNT), min_main_chain_len(MIN_LEN)./; /^[ \t]*%/d; s/[ \t]*%.*$//; /^[ \t]*# /d; 7i #const m=0. min_main_chain_len(m) :- m != 0.' smiles.lp | cat -s > smiles_eval.lp

% ---------------------------------------------

% Determine the number of atoms in the sum formula, not counting hydrogens.
non_hydrogen_atom_count(ATOM_COUNT) :- ATOM_COUNT = #sum{ COUNT, ELEMENT : molecular_formula(ELEMENT, COUNT), ELEMENT != "H" }.

% - Calculate the minimum length of the main chain.
% - We have the shortest main chain with a 'snowflake-like' shape,
%   or alternatively with a fixed-depth tree with branching dedree of 3.
% - This can be developped as a geometric series:
%   --> odd main chain length (snowflake)
%       e.g. NUM_ATOMS = 17 --> min_main_chain_len(17) = 5
%                  +
%               + +++ +                         +
%            +--+--+--+--+      ==>       +   +   +   +
%               + +++ +                  +++ +++ +++ +++
%                  +
%         \begin{align*}
%           a_1^{odd} &= 4\\
%           a_i^{odd} &= a_{i-1}^{odd} \cdot 3^{i-1}\\
%           s_n^{odd} &= 1 + a_1^{odd} \cdot \sum\limits_{i=0}^n a_i^{odd}\\
%           s_n^{odd} &= 1 + 4 \cdot \frac{3^{n-1}-1}{3-1} = 1 + 2 \cdot (3^{n-1}-1)\\
%           \underbrace{2 \cdot \lceil n-1 \rceil + 1}_{\text{odd main chain length}} &= 2 \cdot \left\lceil \log_3\left(\frac{s_n^{odd} - 1}{2}+1\right)\right\rceil + 1
%         \end{align*}
%   --> even main chain length (left and right half-snowflake)
%       e.g. NUM_ATOMS = 26 --> min_main_chain_len(26) = 6
%                +   +
%             + +++ +++ +                       +                  +
%          +--+--+---+--+--+      ==>       +   +   +          +   +   +
%             + +++ +++ +                  +++ +++ +++        +++ +++ +++
%                +   +
%         \begin{align*}
%           a_1^{even} &= 1\\
%           a_i^{even} &= a_{i-1}^{even} \cdot 3^{i-1}\\
%           s_n^{even} &= 2 \cdot a_1^{even} \cdot \sum\limits_{i=0}^n a_i^{even}\\
%           s_n^{even} &= 2 \cdot 1 \cdot \frac{3^{n-1}-1}{3-1} = 3^{n-1}-1\\
%           \underbrace{2 \cdot \lceil n-1 \rceil}_{\text{even main chain length}} &= 2 \cdot \left\lceil \log_3\left(s_n^{even} + 1\right)\right\rceil
%         \end{align*}
% #script (python)
% import clingo
% N = clingo.Number
% from math import ceil, log
% 
% def min_main_chain_len(NUM_ATOMS):
%     return N(min(2*ceil(log((NUM_ATOMS.number-1)/2+1,3))+1, 2*ceil(log(NUM_ATOMS.number+1,3))))
%     #            ^^ odd main chain length                   ^^ even main chain length
% #end.

% Generate the atoms.
atom(1..ATOM_COUNT) :- non_hydrogen_atom_count(ATOM_COUNT).

% Nondeterministically choose the length of the main chain.
% 1{ main_chain_len(min_main_chain_len(ATOM_COUNT)..ATOM_COUNT) }1 :- non_hydrogen_atom_count(ATOM_COUNT).
1{ main_chain_len(MIN_LEN..ATOM_COUNT) }1 :- non_hydrogen_atom_count(ATOM_COUNT), min_main_chain_len(MIN_LEN).


% Compute, how many atoms have which valnce value.
valence(1;3;4).
valence_count(VALENCE, X) :- X = #sum{ COUNT, ELEMENT : molecular_formula(ELEMENT, COUNT), element(ELEMENT, _, VALENCE) }, valence(VALENCE), not element_valence(_, VALENCE).
element_valence(ELEMENT, VALENCE) :- element(ELEMENT, _, VALENCE), element_valence(ELEMENT, _).
1{ element_valence_count(ELEMENT, VALENCE, 0..COUNT) }1 :- element_valence(ELEMENT, VALENCE), molecular_formula(ELEMENT, COUNT).
:- molecular_formula(ELEMENT, COUNT), COUNT != #sum{ C, VALENCE : element_valence_count(ELEMENT, VALENCE, C) }, element_valence(ELEMENT, _).
valence_count(VALENCE, X) :- X = #sum{ COUNT, ELEMENT : element_valence_count(ELEMENT, VALENCE, COUNT) }, element_valence(_, VALENCE).
% Compute, how many cycles / multi-bonds there should be, based on the hydrogen count in the sum formula.
num_cycles_or_multi_bonds((2*C+2 + N - H) / 2) :- valence_count(4, C), valence_count(3, N), valence_count(1, H).
% SENIOR rules (adaptation)
:- valence_count(4, C), valence_count(3, N), valence_count(1, H), 2*C+2 + N < H. % --> cannot have too many atoms with valence 1
:- valence_count(4, C), valence_count(3, N), valence_count(1, H), (2*C+2 + N - H) \ 2 != 0. % --> cannot have "half a cycle"
% Nondeterministically decide how many multi-bonds and cycles there should be.
1{ num_multi_bonds(0..NUM_CYCLES_OR_MULTI_BONDS) }1 :- num_cycles_or_multi_bonds(NUM_CYCLES_OR_MULTI_BONDS), NUM_CYCLES_OR_MULTI_BONDS > 0.
num_multi_bonds(0) :- num_cycles_or_multi_bonds(0).
num_cycles(NUM_CYCLES_OR_MULTI_BONDS - NUM_MULTI_BONDS) :- num_multi_bonds(NUM_MULTI_BONDS), num_cycles_or_multi_bonds(NUM_CYCLES_OR_MULTI_BONDS).

% Nonteterministically distribute the element symbols on the atoms.
1{ symbol(1, ELEMENT) : element(ELEMENT, _, _), ELEMENT != "H" }1 :- atom(1).
element_count(1, ELEMENT, COUNT) :- molecular_formula(ELEMENT, COUNT), not symbol(1, ELEMENT), ELEMENT != "H".
element_count(1, ELEMENT, COUNT-1) :- molecular_formula(ELEMENT, COUNT), symbol(1, ELEMENT), COUNT > 1.

1{ symbol(I, ELEMENT) : element_count(I-1, ELEMENT, COUNT) }1 :- atom(I), I>1.
element_count(I, ELEMENT, COUNT) :- element_count(I-1, ELEMENT, COUNT), not symbol(I, ELEMENT), atom(I), I>1.
element_count(I, ELEMENT, COUNT-1) :- element_count(I-1, ELEMENT, COUNT), symbol(I, ELEMENT), COUNT > 1.

% Nondeterministically distribute the multi-bonds.
multi_bond_count(1, MULTI_BONDS) :- num_multi_bonds(MULTI_BONDS), MULTI_BONDS > 0.
{ multi_bond(I, 2..M) }1 :- atom(I), I>1, multi_bond_count(I-1, MULTI_BONDS), M = #min{ 3; M' : M'=MULTI_BONDS+1 }.
multi_bond_count(I, MULTI_BONDS) :- multi_bond_count(I-1, MULTI_BONDS), not multi_bond(I, _), atom(I), I>1.
multi_bond_count(I, MULTI_BONDS-MULTIPLICITY+1) :- multi_bond_count(I-1, MULTI_BONDS), multi_bond(I, MULTIPLICITY), MULTI_BONDS > MULTIPLICITY-1.
:- multi_bond_count(ATOM_COUNT, _), non_hydrogen_atom_count(ATOM_COUNT).

% Nondeterministically distribute the cycle markers.
{ cycle_start_num(1, 1..M) }1 :- atom(1), num_cycles(NUM_CYCLES), NUM_CYCLES > 0, M = #min{ 3; N : N=NUM_CYCLES }.
cycle_start_count(1, NUM_CYCLES) :- num_cycles(NUM_CYCLES), not cycle_start_num(1, _), NUM_CYCLES > 0.
cycle_start_count(1, NUM_CYCLES - CYCLE_START_NUM) :- num_cycles(NUM_CYCLES), cycle_start_num(1, CYCLE_START_NUM), NUM_CYCLES > CYCLE_START_NUM.
{ cycle_start_num(I, 1..M) }1 :- atom(I), I>1, cycle_start_count(I-1, CYCLE_START_COUNT), M = #min{ 3; N : N=CYCLE_START_COUNT }.
cycle_start_count(I, CYCLE_START_COUNT) :- cycle_start_count(I-1, CYCLE_START_COUNT), not cycle_start_num(I, _), CYCLE_START_COUNT > 0, atom(I), I>1.
cycle_start_count(I, CYCLE_START_COUNT - CYCLE_START_NUM) :- cycle_start_count(I-1, CYCLE_START_COUNT), cycle_start_num(I, CYCLE_START_NUM), CYCLE_START_COUNT > CYCLE_START_NUM.
:- cycle_start_count(ATOM_COUNT, _), non_hydrogen_atom_count(ATOM_COUNT).

cycle_start(1, 1..CYCLE_START_NUM) :- cycle_start_num(1, CYCLE_START_NUM).
cycle_start(I, NUM_CYCLES-CYCLE_START_COUNT+(1..CYCLE_START_NUM)) :- cycle_start_count(I-1, CYCLE_START_COUNT), num_cycles(NUM_CYCLES), cycle_start_num(I, CYCLE_START_NUM).

cycle_end_count(1, NUM_CYCLES) :- num_cycles(NUM_CYCLES), NUM_CYCLES > 0.
{ cycle_end_num(I, 1..M) }1 :- atom(I), I>1, cycle_start_count(I-1, CYCLE_START_COUNT), cycle_end_count(I-1, CYCLE_END_COUNT), num_cycles(NUM_CYCLES), M = #min{ 3; S : S=NUM_CYCLES-CYCLE_START_COUNT; E : E=CYCLE_END_COUNT }.
{ cycle_end_num(I, 1..M) }1 :- atom(I), I>1, not cycle_start_count(I-1, _), cycle_end_count(I-1, CYCLE_END_COUNT), num_cycles(NUM_CYCLES), M = #min{ 3; S : S=NUM_CYCLES; E : E=CYCLE_END_COUNT }.
cycle_end_count(I, CYCLE_END_COUNT) :- cycle_end_count(I-1, CYCLE_END_COUNT), not cycle_end_num(I, _), CYCLE_END_COUNT > 0, atom(I), I>1.
cycle_end_count(I, CYCLE_END_COUNT - CYCLE_END_NUM) :- cycle_end_count(I-1, CYCLE_END_COUNT), cycle_end_num(I, CYCLE_END_NUM), CYCLE_END_COUNT > CYCLE_END_NUM.
:- cycle_end_count(ATOM_COUNT, _), non_hydrogen_atom_count(ATOM_COUNT).

CYCLE_END_NUM{ cycle_end(I, 1..NUM_CYCLES) }CYCLE_END_NUM :- cycle_end_num(I, CYCLE_END_NUM), num_cycles(NUM_CYCLES).
% Only pick a single cycle_end per cycle.
:- cycle_end(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
% Cycles should involve at least three atoms, as they are otherwise represented via multi-bonds.
:- cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), parent(I1, _, I2).
% Cycles startig at the same point should be numbered by cycle-end order.
:- cycle_start(I, CYCLE_1), cycle_start(I, CYCLE_2), cycle_end(I1, CYCLE_1), cycle_end(I2, CYCLE_2), CYCLE_1 < CYCLE_2, I1 > I2.
% Cycle start marker should preceed cycle end marker.
:- cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 >= I2.

%% Prohibit interleaving cycles, as they can always be written as outer and inner cycle.
%:- cycle_start(I1, CYCLE_1), cycle_start(I2, CYCLE_2), cycle_end(I3, CYCLE_1), cycle_end(I4, CYCLE_2), I1<I2, I2<I3, I3<I4, CYCLE_1 < CYCLE_2.
% NOTE: This constraint unfortunately is incompatible with the avoidance of shortening cycles.
%       E.g. for Cubane (with sumformula C8H8):
%            -- C₅25C₄4C₃3C₂2C₁1C₆5C₇4C₈13 has interleaving, but no shortening cycles (3 isomorphic models)
%            -- C₄34C₃2C₂(C₅35)C₁1C₆5C₇4C₈12 has no interleaving, but shortening cycles (2 isomorphic models)
%       --> For compounds with many cycles, prohibiting interleaving cycles is more important.
%       E.g. for C10H20 (single cycle)
%            -- 951 models, disallowing shortening cycles
%            -- 1833 models, allowing shortening cycles
%       --> For compounds with few cycles, prohibiting shortening cycles is more important.

% Sum up the per-atom bonds due to cyclicity.
preset_bonds(I, MULTIPLICITY) :- not cycle_start_num(I, _), not cycle_end_num(I, _), multi_bond(I, MULTIPLICITY).
preset_bonds(I, MULTIPLICITY + CYCLE_END_NUM) :- not cycle_start_num(I, _), cycle_end_num(I, CYCLE_END_NUM), multi_bond(I, MULTIPLICITY).
preset_bonds(I, MULTIPLICITY + CYCLE_START_NUM) :- cycle_start_num(I, CYCLE_START_NUM), not cycle_end_num(I, _), multi_bond(I, MULTIPLICITY).
preset_bonds(I, MULTIPLICITY + CYCLE_START_NUM + CYCLE_END_NUM) :- cycle_start_num(I, CYCLE_START_NUM), cycle_end_num(I, CYCLE_END_NUM), multi_bond(I, MULTIPLICITY).

preset_bonds(I, M) :- atom(I), not cycle_start_num(I, _), not cycle_end_num(I, _), not multi_bond(I, _), M = #min{ 1; 0: I==1 }.
preset_bonds(I, M + CYCLE_END_NUM) :- not cycle_start_num(I, _), cycle_end_num(I, CYCLE_END_NUM), not multi_bond(I, _), M = #min{ 1; 0: I==1 }.
preset_bonds(I, M + CYCLE_START_NUM) :- cycle_start_num(I, CYCLE_START_NUM), not cycle_end_num(I, _), not multi_bond(I, _), M = #min{ 1; 0: I==1 }.
preset_bonds(I, M + CYCLE_START_NUM + CYCLE_END_NUM) :- cycle_start_num(I, CYCLE_START_NUM), cycle_end_num(I, CYCLE_END_NUM), not multi_bond(I, _), M = #min{ 1; 0: I==1 }.

:- symbol(I, ELEMENT), element(ELEMENT, _, VALENCE), preset_bonds(I, BONDS), BONDS > VALENCE.

% Nondeterministically decide for a branching at each atom.
branching(1, 1) :- non_hydrogen_atom_count(2).
1{ branching(1, 2..MAX_BRANCHING) }1 :- non_hydrogen_atom_count(ATOM_COUNT), ATOM_COUNT > 2,
                                        MAX_BRANCHING = #min{ N-1 : N=ATOM_COUNT;
                                                              VALENCE-BONDS : symbol(1, ELEMENT), element(ELEMENT, _, VALENCE), preset_bonds(1, BONDS) }.

1{ branching(I, 1..MAX_BRANCHING) }1 :- symbol(I, ELEMENT), I>1,
                                        MAX_BRANCHING = #min{ SIZE'-DEPTH'+1 : SIZE'=SIZE, DEPTH'=DEPTH;
                                                              VALENCE-BONDS : element(ELEMENT, _, VALENCE), preset_bonds(I, BONDS) },
                                        size(I, SIZE), SIZE >= DEPTH,
                                        depth(I, DEPTH), DEPTH > 1.

postset_bonds(I, M1) :- branching(I, 1), parent(I, 0, I1), multi_bond(I1, M1).
postset_bonds(I, M1+1) :- branching(I, 2), parent(I, 0, I1), parent(I, 1, I2), I1<I2, multi_bond(I1, M1), not multi_bond(I2, _).
postset_bonds(I, 1+M2) :- branching(I, 2), parent(I, 0, I1), parent(I, 1, I2), I1<I2, not multi_bond(I1, _), multi_bond(I2, M2).
postset_bonds(I, M1+M2) :- branching(I, 2), parent(I, 0, I1), parent(I, 1, I2), I1<I2, multi_bond(I1, M1), multi_bond(I2, M2).
postset_bonds(I, M1+1+1) :- branching(I, 3), parent(I, 0, I1), parent(I, 1, I2), parent(I, 2, I3), I1<I2, I2<I3, multi_bond(I1, M1), not multi_bond(I2, _), not multi_bond(I3, _).
postset_bonds(I, 1+M2+1) :- branching(I, 3), parent(I, 0, I1), parent(I, 1, I2), parent(I, 2, I3), I1<I2, I2<I3, not multi_bond(I1, _), multi_bond(I2, M2), not multi_bond(I3, _).
postset_bonds(I, 1+1+M3) :- branching(I, 3), parent(I, 0, I1), parent(I, 1, I2), parent(I, 2, I3), I1<I2, I2<I3, not multi_bond(I1, _), not multi_bond(I2, _), multi_bond(I3, M3).
postset_bonds(I, M1+M2+1) :- branching(I, 3), parent(I, 0, I1), parent(I, 1, I2), parent(I, 2, I3), I1<I2, I2<I3, multi_bond(I1, M1), multi_bond(I2, M2), not multi_bond(I3, _).
postset_bonds(I, M1+1+M3) :- branching(I, 3), parent(I, 0, I1), parent(I, 1, I2), parent(I, 2, I3), I1<I2, I2<I3, multi_bond(I1, M1), not multi_bond(I2, _), multi_bond(I3, M3).
postset_bonds(I, 1+M2+M3) :- branching(I, 3), parent(I, 0, I1), parent(I, 1, I2), parent(I, 2, I3), I1<I2, I2<I3, not multi_bond(I1, _), multi_bond(I2, M2), multi_bond(I3, M3).
postset_bonds(I, M1+M2+M3) :- branching(I, 3), parent(I, 0, I1), parent(I, 1, I2), parent(I, 2, I3), I1<I2, I2<I3, multi_bond(I1, M1), multi_bond(I2, M2), multi_bond(I3, M3).

free_bonds(I, FREE_BONDS) :- preset_bonds(I, BONDS_PRE),
                             not postset_bonds(I, _),
                             not branching(I, _),
                             symbol(I, ELEMENT), element(ELEMENT, _, VALENCE),
                             FREE_BONDS = VALENCE - BONDS_PRE,
                             FREE_BONDS >= 0.
free_bonds(I, FREE_BONDS) :- preset_bonds(I, BONDS_PRE),
                             not postset_bonds(I, _),
                             branching(I, BRANCHING),
                             symbol(I, ELEMENT), element(ELEMENT, _, VALENCE),
                             FREE_BONDS = VALENCE - BONDS_PRE - BRANCHING,
                             FREE_BONDS >= 0.
free_bonds(I, FREE_BONDS) :- preset_bonds(I, BONDS_PRE),
                             postset_bonds(I, BONDS_POST),
                             symbol(I, ELEMENT), element(ELEMENT, _, VALENCE),
                             FREE_BONDS = VALENCE - BONDS_PRE - BONDS_POST,
                             FREE_BONDS >= 0.

:- atom(I), not free_bonds(I, _).

% Not neccessary, but improves performance...
:- branching(1, 4), parent(1, _, I), multi_bond(I, _).
% Prohibit three double bonds at root atom.
:- multi_bond(I1, 2), multi_bond(I2, 2), multi_bond(I3, 2),
   parent(1, CHAIN_1, I1), parent(1, CHAIN_2, I2), parent(1, CHAIN_3, I3),
   I1 < I2, I2 < I3, CHAIN_1 < CHAIN_2, CHAIN_2 < CHAIN_3.
% Prohibit two double bond at non-root atom.
:- multi_bond(I1, 2), multi_bond(I2, 2),
   parent(I, CHAIN_1, I1), parent(I, CHAIN_2, I2),
   I1 < I2, CHAIN_1 < CHAIN_2, I > 1.
% Prohibit multiple triple bonds at children of root atom.
:- multi_bond(I1, 3), multi_bond(I2, 3),
   parent(1, CHAIN_1, I1), parent(1, CHAIN_2, I2),
   I1 < I2, CHAIN_1 < CHAIN_2.
% Prohibit side chain start at triple bond.
:- multi_bond(I1, 3), parent(I, _, I1), not branching(I, 1), I > 1.
% Prohibit further multi-bond adjacent to triple bond.
:- multi_bond(I1, 3), parent(I, _, I1), multi_bond(I, _), I > 1.

% Nondeterministically decide for chain depths, making sure that right chains are at most as long as left ones.
% Nondeterministically decide for number of atoms in chain, making sure that there are enough to reach their depth.
depth(2, ((MAIN_CHAIN_LEN-1)+(MAIN_CHAIN_LEN-1)\2)/2) :- main_chain_len(MAIN_CHAIN_LEN), MAIN_CHAIN_LEN > 1.
depth(I+1, DEPTH-1) :- depth(I, DEPTH), atom(I+1), branching(I, _), DEPTH > 1.

size(1, ATOM_COUNT) :- non_hydrogen_atom_count(ATOM_COUNT).

1{ parent(I, 0, I+1) }1 :- branching(I, _), I < ATOM_COUNT, non_hydrogen_atom_count(ATOM_COUNT).
1{ parent(I, CHAIN, SUM+DEPTH..MAX_CHILD) }1 :- parent(I, CHAIN-1, SUM),
                                                depth(SUM, DEPTH),
                                                size(I, PARENT_SIZE),
                                                branching(I, BRANCHING), BRANCHING > CHAIN,
                                                MAX_CHILD = #min{ ATOM_COUNT : non_hydrogen_atom_count(ATOM_COUNT);
                                                                  T : T=I+PARENT_SIZE-BRANCHING+CHAIN },
                                                MAX_CHILD >= SUM+DEPTH.

size(I, POS-I) :- parent(I-1, 1, POS), POS < ATOM_COUNT, non_hydrogen_atom_count(ATOM_COUNT).
size(POS_1, POS_2-POS_1) :- parent(I, CHAIN, POS_1), parent(I, CHAIN+1, POS_2), POS_2 > POS_1, POS_2-POS_1 <= ATOM_COUNT, non_hydrogen_atom_count(ATOM_COUNT).
size(POS, I+PARENT_SIZE-POS) :- parent(I, CHAIN, POS), branching(I, CHAIN+1), size(I, PARENT_SIZE), POS <= I+PARENT_SIZE.

depth(2+LEFT_SIZE, ((MAIN_CHAIN_LEN-1)-(MAIN_CHAIN_LEN-1)\2)/2) :- main_chain_len(MAIN_CHAIN_LEN), MAIN_CHAIN_LEN > 2,
                                                                   size(2, LEFT_SIZE).
1{ depth(POS_2, 1..PREV_DEPTH) }1 :- branching(I, BRANCHING), BRANCHING > CHAIN,
                                     depth(POS_1, PREV_DEPTH),
                                     parent(I, CHAIN, POS_1),
                                     parent(I, CHAIN+1, POS_2).

% Ensure to reach the pre-selected depth.
:- not depth(I, 1), not branching(I, _), atom(I), I > 1.

% ---------------------------------------------

% Non functional constraints.
% --> Not neccessary, but significantly improves solving performance (at a slight cost to ground program size)
% --> In experiments, the uncommented combination is best...
:- depth(I, D1), depth(I, D2), D1 < D2.
:- size(I, S1), size(I, S2), S1 < S2.
%:- branching(I, BRANCHING), not parent(I, BRANCHING-1, _).
%:- depth(I, 1), size(I, SIZE), SIZE > 1.

% ---------------------------------------------

% Symmetry-breaking

% Lexicographic comparison of:
%   1) depth
%   2) size
%   3) branching
%   4) symbol
%   5) multi_bond
%   6) cycle_start
%   7) cycle_end
%   ... children

part_eq(I1, I2, 1) :- I1 < I2, atom(I1), atom(I2), (I1,I2) != (1,2),
                      depth(I1, DEPTH), depth(I2, DEPTH).
part_eq(I1, I2, 2) :- part_eq(I1, I2, 1),
                      size(I1, SIZE), size(I2, SIZE).
part_eq(I1, I2, 3) :- part_eq(I1, I2, 2),
                      not branching(I1, _), not branching(I2, _).
part_eq(I1, I2, 3) :- part_eq(I1, I2, 2),
                      branching(I1, BRANCHING), branching(I2, BRANCHING).
part_eq(I1, I2, 4) :- part_eq(I1, I2, 3),
                      symbol(I1, ELEMENT), symbol(I2, ELEMENT).
part_eq(I1, I2, 5) :- part_eq(I1, I2, 4), (I1,I2) != (1,2),
                      not multi_bond(I1, _), not multi_bond(I2, _).
part_eq(I1, I2, 5) :- part_eq(I1, I2, 4), (I1,I2) != (1,2),
                      multi_bond(I1, MULTIPLICITY), multi_bond(I2, MULTIPLICITY).
part_eq(I1, I2, 6) :- part_eq(I1, I2, 5),
                      not cycle_start_num(I1, _), not cycle_start_num(I2, _).
part_eq(I1, I2, 6) :- part_eq(I1, I2, 5),
                      cycle_start_num(I1, CYCLE_START_NUM), cycle_start_num(I2, CYCLE_START_NUM).
part_eq(I1, I2, 7) :- part_eq(I1, I2, 6),
                      not cycle_end_num(I1, _), not cycle_end_num(I2, _).
part_eq(I1, I2, 7) :- part_eq(I1, I2, 6),
                      cycle_end_num(I1, CYCLE_END_NUM), cycle_end_num(I2, CYCLE_END_NUM).
part_eq(I1, I2, 8+CHAIN) :- part_eq(I1, I2, 7+CHAIN), (I1,I2) != (1,2),
                            parent(I1, CHAIN, I1'), parent(I2, CHAIN, I2'),
                            eq(I1', I2').

eq(I1, I2) :- not branching(I1, _), part_eq(I1, I2, 7).
eq(I1, I2) :- branching(I1, BRANCHING), part_eq(I1, I2, 7+BRANCHING).

lt(I1, I2) :- I1 < I2, atom(I1), atom(I2), (I1,I2) != (1,2),
              depth(I1, DEPTH_1), depth(I2, DEPTH_2), DEPTH_1 < DEPTH_2.
lt(I1, I2) :- part_eq(I1, I2, 1),
              size(I1, SIZE_1), size(I2, SIZE_2), SIZE_1 < SIZE_2.
lt(I1, I2) :- part_eq(I1, I2, 2),
              not branching(I1, _), branching(I2, _).
lt(I1, I2) :- part_eq(I1, I2, 2),
              branching(I1, BRANCHING_1), branching(I2, BRANCHING_2), BRANCHING_1 < BRANCHING_2.
lt(I1, I2) :- part_eq(I1, I2, 3),
              symbol(I1, ELEMENT_1), symbol(I2, ELEMENT_2),
              element(ELEMENT_1, ATOMIC_NR_1, _), element(ELEMENT_2, ATOMIC_NR_2, _), ATOMIC_NR_1 < ATOMIC_NR_2.
lt(I1, I2) :- part_eq(I1, I2, 4), (I1,I2) != (1,2),
              not multi_bond(I1, _), multi_bond(I2, _).
lt(I1, I2) :- part_eq(I1, I2, 4), (I1,I2) != (1,2),
              multi_bond(I1, MULTIPLICITY_1), multi_bond(I2, MULTIPLICITY_2), MULTIPLICITY_1 < MULTIPLICITY_2.
lt(I1, I2) :- part_eq(I1, I2, 5),
              not cycle_start_num(I1, _), cycle_start_num(I2, _).
lt(I1, I2) :- part_eq(I1, I2, 5),
              cycle_start_num(I1, CYCLE_START_NUM_1), cycle_start_num(I2, CYCLE_START_NUM_2), CYCLE_START_NUM_1 < CYCLE_START_NUM_2.
lt(I1, I2) :- part_eq(I1, I2, 6),
              not cycle_end_num(I1, _), cycle_end_num(I2, _).
lt(I1, I2) :- part_eq(I1, I2, 6),
              cycle_end_num(I1, CYCLE_END_NUM_1), cycle_end_num(I2, CYCLE_END_NUM_2), CYCLE_END_NUM_1 < CYCLE_END_NUM_2.
lt(I1, I2) :- part_eq(I1, I2, 7+CHAIN), (I1,I2) != (1,2),
              parent(I1, CHAIN, I1'), parent(I2, CHAIN, I2'),
              lt(I1', I2').

part_eq(1, 2, 3) :- branching(1, BRANCHING_1), branching(2, BRANCHING_2), BRANCHING_1-1 == BRANCHING_2.
part_eq(1, 2, 5) :- part_eq(1, 2, 4).
part_eq(1, 2, 8+CHAIN) :- part_eq(1, 2, 7+CHAIN),
                          parent(1, CHAIN+1, I1'), parent(2, CHAIN, I2'),
                          eq(I2', I1').
lt(1, 2) :- branching(1, BRANCHING_1), branching(2, BRANCHING_2), BRANCHING_1 <= BRANCHING_2.
lt(1, 2) :- part_eq(1, 2, 7+CHAIN),
            parent(1, CHAIN+1, I1'), parent(2, CHAIN, I2'),
            lt(I2', I1').

:- parent(I, CHAIN, I1), parent(I, CHAIN+1, I2), lt(I1, I2).
:- main_chain_len(MAIN_CHAIN_LEN), MAIN_CHAIN_LEN\2 = 0, lt(1, 2).

% Symmetry breaking for cycles ...

max_len((ATOM_COUNT-ATOM_COUNT\2)/2) :- non_hydrogen_atom_count(ATOM_COUNT).

% Compute transitive chain relation.
chain(I, I+1) :- branching(I, _),
                 not num_cycles(0).
chain(I1, I2+1) :- chain(I1, I2), branching(I2, _), I1 < I2,
                   max_len(MAX_LEN), I2-I1 < MAX_LEN,
                   not num_cycles(0).

% Compute off-chain paths with lengths.
path(I1, I2, 1) :- parent(I1, CHAIN, I2), CHAIN > 0, (I1,CHAIN) != (1,1), I1+1 < I2.
path(I1, I2, 1+I2-I) :- parent(I1, CHAIN, I), CHAIN > 0, (I1,CHAIN) != (1,1), I1+1 < I,
                        chain(I, I2), I < I2,
                        max_len(MAX_LEN), I2-I < MAX_LEN,
                        not num_cycles(0).
path(I1, I2, 1+LEN) :- parent(I1, CHAIN, I), CHAIN > 0, (I1,CHAIN) != (1,1), I1+1 < I,
                       path(I, I2, LEN), I < I2, LEN >= I2-I,
                       max_len(MAX_LEN), LEN < MAX_LEN,
                       not num_cycles(0).

%    +-----------------------------+
%    |              *              |
%    |             *   _           |
%    |    _       I    |           |
%    |    |      * *   | L2        |
%    |  L |     *   *  |_          |
%    |    |    *    I2 ‾|          |
%    |    ‾   I'   /  * | DEPTH_2  |
%    |       * *  /    *‾          |
%    |      *   I1                 |
%    |     *     *                 |
%    |    *       *                |
%    +-----------------------------+

% Determine, which cycles are between two side chains of an ancestor chain (side cycle).
cycle(CYCLE, I+1, I', L1, L2) :- chain(I, I'), I < I', I'-I <= MAX_LEN, max_len(MAX_LEN),
                                 path(I', I1, L1), I'+1 < I1, L1 <= MAX_LEN,
                                 path(I, I2, L2), I+1 < I2, L2 <= MAX_LEN,
                                 cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
% Determine, which cycles are between a chain and its side chain, branching off below (branch cycle).
cycle(CYCLE, I1+1, I', L2, 0) :- chain(I1, I'), I1 < I', I'-I1 <= MAX_LEN, max_len(MAX_LEN),
                                 path(I', I2, L2), I'+1 < I2, L2 <= MAX_LEN,
                                 cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
% Determine, which cycles are between a chain and its side chain, branching off above (branch cycle).
cycle(CYCLE, I+1, I1, 0, L2) :- chain(I, I1), I < I1, I1-I <= MAX_LEN, max_len(MAX_LEN),
                                path(I, I2, L2), I+1 < I2, L2 <= MAX_LEN,
                                cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
% Determine, which cycles are within a chain (chain cycle).
cycle(CYCLE, I1+1, I2, 0, 0) :- chain(I1, I2), I1 < I2, I2-I1 <= MAX_LEN, max_len(MAX_LEN),
                                cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
% Determine, which side cycles originate in the root atom.
cycle(CYCLE, I', I', L1, L2) :- parent(1, 1, I'), 1 < I', max_len(MAX_LEN),
                                path(I', I1, L1), I'+1 < I1, L1 <= MAX_LEN,
                                path(1, I2, L2), 2 < I2, L2 <= MAX_LEN,
                                cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
cycle(CYCLE, I'', I', L1, L2) :- parent(1, 1, I''), 1 < I'', chain(I'', I'), I'' < I', I'-I'' <= MAX_LEN, max_len(MAX_LEN),
                                 path(I', I1, L1), I'+1 < I1, L1 <= MAX_LEN,
                                 path(1, I2, L2), 2 < I2, L2 <= MAX_LEN,
                                 cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
% Determine, which branch cycles originate in the root atom.
cycle(CYCLE, I1, I1, 0, L2) :- parent(1, 1, I1), 1 < I1, max_len(MAX_LEN),
                               path(1, I2, L2), 2 < I2, L2 <= MAX_LEN,
                               cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
cycle(CYCLE, I'', I1, 0, L2) :- parent(1, 1, I''), 1 < I'', chain(I'', I1), I'' < I1, I1-I'' <= MAX_LEN, max_len(MAX_LEN),
                                path(1, I2, L2), 2 < I2, L2 <= MAX_LEN,
                                cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.
cycle(CYCLE, I', I', L2, 0) :- parent(1, 1, I'), 1 < I', max_len(MAX_LEN),
                               path(I', I2, L2), 2 < I2, L2 <= MAX_LEN,
                               cycle_start(1, CYCLE), cycle_end(I2, CYCLE), 1 < I2.
cycle(CYCLE, I'', I', L2, 0) :- parent(1, 1, I''), 1 < I'', chain(I'', I'), I'' < I', I'-I'' <= MAX_LEN, max_len(MAX_LEN),
                                path(I', I2, L2), 2 < I2, L2 <= MAX_LEN,
                                cycle_start(1, CYCLE), cycle_end(I2, CYCLE), 1 < I2.
% Determine, which chain cycles originate in the root atom.
cycle(CYCLE, I'', I2, 0, 0) :- parent(1, 1, I''), 1 < I'', chain(I'', I2), I'' < I2, I2-I'' <= MAX_LEN, max_len(MAX_LEN),
                               cycle_start(1, CYCLE), cycle_end(I2, CYCLE), 1 < I2.

%    +-------------------------------------+
%    |          _               _          |
%    |      *   | DEPTH_1   *   | DEPTH_2  |
%    |      I1 _‾           I2 _‾          |
%    |      *  | L1         *  | L2        |
%    |      *  |            *  |           |
%    |  * * I *‾* * * * * * I'*‾*          |
%    |            ^ ^ ^                    |
%    |            2 1 I''                  |
%    |      |---------------|              |
%    |        L = I+I'-I''                 |
%    +-------------------------------------+

% Determine, which side cycles span between left and right half of the main chain.
cross_cycle(CYCLE, I, I', L1, L2) :- chain(1, I), 1 < I, I-1 <= MAX_LEN, max_len(MAX_LEN),
                                     path(I, I1, L1), I+1 < I1, L1 <= MAX_LEN,
                                     parent(1, 1, I''), I1 < I'', chain(I'', I'), I'' < I', I'-I'' < MAX_LEN, 1+I'-I''+L2 < I'', I+L1 < I'',
                                     path(I', I2, L2), I'+1 < I2, L2 <= MAX_LEN,
                                     cycle_start(I1, CYCLE), cycle_end(I2, CYCLE).
cross_cycle(CYCLE, I, I', L1, L2) :- chain(1, I), 1 < I, I-1 <= MAX_LEN, max_len(MAX_LEN),
                                     path(I, I1, L1), I+1 < I1, L1 <= MAX_LEN,
                                     parent(1, 1, I'), I1 < I', 1+L2 < I', I+L1 < I',
                                     path(I', I2, L2), I'+1 < I2, L2 <= MAX_LEN,
                                     cycle_start(I1, CYCLE), cycle_end(I2, CYCLE).
% Determine, which branch cycles span between left and right half of the main chain.
cross_cycle(CYCLE, I1, I', 0, L2) :- chain(1, I1), 1 < I1, I1-1 <= MAX_LEN, max_len(MAX_LEN),
                                     parent(1, 1, I''), I1 < I'', chain(I'', I'), I'' < I', I'-I'' < MAX_LEN, 1+I'-I''+L2 < I'',
                                     path(I', I2, L2), I'+1 < I2, L2 <= MAX_LEN,
                                     cycle_start(I1, CYCLE), cycle_end(I2, CYCLE).
cross_cycle(CYCLE, I1, I', 0, L2) :- chain(1, I1), 1 < I1, I1-1 <= MAX_LEN, max_len(MAX_LEN),
                                     parent(1, 1, I'), I1 < I', 1+L2 < I',
                                     path(I', I2, L2), I'+1 < I2, L2 <= MAX_LEN,
                                     cycle_start(I1, CYCLE), cycle_end(I2, CYCLE).
cross_cycle(CYCLE, I, I2, L1, 0) :- chain(1, I), 1 < I, I-1 <= MAX_LEN, max_len(MAX_LEN),
                                    path(I, I1, L1), I+1 < I1, L1 <= MAX_LEN,
                                    parent(1, 1, I''), I1 < I'', chain(I'', I2), I'' < I2, I2-I'' < MAX_LEN, 1+I2-I'' < I'', I+L1 < I'',
                                    cycle_start(I1, CYCLE), cycle_end(I2, CYCLE).
cross_cycle(CYCLE, I, I2, L1, 0) :- chain(1, I), 1 < I, I-1 <= MAX_LEN, max_len(MAX_LEN),
                                    path(I, I1, L1), I+1 < I1, L1 <= MAX_LEN,
                                    parent(1, 1, I2), I1 < I2, 1 < I2, I+L1 < I2,
                                    cycle_start(I1, CYCLE), cycle_end(I2, CYCLE).
% Determine, which chain cycles span between left and right half of the main chain.
cross_cycle(CYCLE, I1, I2, 0, 0) :- chain(1, I1), 1 < I1, I1-1 <= MAX_LEN, max_len(MAX_LEN),
                                    parent(1, 1, I''), I1 < I'', chain(I'', I2), I'' < I2, I2-I'' < MAX_LEN, 1+I2-I'' < I'', I1 < I'',
                                    cycle_start(I1, CYCLE), cycle_end(I2, CYCLE).
cross_cycle(CYCLE, I1, I2, 0, 0) :- chain(1, I1), 1 < I1, I1-1 <= MAX_LEN, max_len(MAX_LEN),
                                    parent(1, 1, I2), I1 < I2, 1 < I2,
                                    cycle_start(I1, CYCLE), cycle_end(I2, CYCLE).

% Prohibit cycle between base of and atom in side chain.
:- path(I1, I2, L2), L2 <= MAX_LEN, max_len(MAX_LEN),
   cycle_start(I1, CYCLE), I1 != 1, cycle_end(I2, CYCLE), I1 < I2.

% Prohibit cycle between two side chains originating at the same atom.
:- path(I, I1, L1), I < I1, L1 <= MAX_LEN, max_len(MAX_LEN),
   path(I, I2, L2), I < I2, L2 <= MAX_LEN,
   not chain(I1, I2),
   cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2.

% Compute side chain depths.
side_chain(I_BASE, SIDE_DEPTH) :- parent(I_BASE, CHAIN, I_SIDE), CHAIN > 0, (I_BASE,CHAIN) != (1,1), I_BASE+1 < I_SIDE,
                                  depth(I_SIDE, SIDE_DEPTH),
                                  max_len(MAX_LEN), SIDE_DEPTH < MAX_LEN,
                                  not num_cycles(0).

% Cycles need to have longer path on main chain than on side chain.
% ---> sacrifice distance between the branch-off points to gain both cycle heights
%            1
%           #  ...
%          + #      <==>  (I' -I) < L1 + L2
%         +   #     <==>  (5-3) < 1+2
%        #   /      <==>  2 < 3
%       # # ‾
%      #   .
%      e.g. C₄C₃1C₂(C₅1)C₁C₆C₇C₈               cycle(1,I=3,I'=3,L=0,L1=0,L2=1)
%           C₄C₃C₂1C₁(C₇1)(C₈)C₅C₆             cycle(1,I=2,I'=2,L=0,L1=0,L2=1)
%           C₄C₃C₂(C₅1)C₁(C₉1)C₆C₇C₈           cycle(1,I=2,I'=2,L=0,L1=1,L2=1)
:- cycle(CYCLE, I, I', L1, L2), I'-I < L1+L2.
% ---> sacrifice depth beneth first cycle branch-off point to gain both cycle heights
%           1
%          #  ...  <==>  DEPTH' -1 < L1 + L2
%         # #      <==>  2-1 < 1+1
%        # /       <==>  1 < 2
%       # /
%      + #
%      e.g. C₇C₆C₅(C₈1)C₄C₃C₂C₁(C₁₄C₁₅1)C₉C₁₀C₁₁C₁₂C₁₃   cycle(1,I=2,I'=5,L=3,L1=1,L2=2)
%      (NOTE: covered by next rule for smaller atom counts, where mostly L2 <= 1)
:- cycle(CYCLE, I, I', L1, L2), I'-I >= L1+L2, depth(I', DEPTH'), L1 < DEPTH', L2 < DEPTH'+(I'-I+1), DEPTH'-1 < L1+L2, L1 != 0.
%                               ‾‾‾‾‾‾‾‾‾‾‾‾‾‾                    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
% ---> sacrifice depth beneth first cycle branch-off point to gain first cycle height and depth beneth second cycle marker
%             1
%            #  ...
%           # +      <==>  DEPTH' -1 < L1 + DEPTH_2
%          #   #     <==>  3-1 < 1+2
%         #   / #    <==>  2 < 3
%        #   /
%       + # ‾
%      +
%      e.g. C₄C₃1C₂C₁(C₈1C₉)C₅C₆C₇             cycle(1,I=2,I'=3,L=1,L1=0,L2=1)
%           C₅1C₄C₃C₂C₁(C₉1)C₆C₇C₈             cycle(1,I=2,I'=5,L=3,L1=0,L2=1)
%           C₄C₃C₂C₁(C₇C₈)(C₉1)C₅C₆1           cycle(1,I=5,I'=6,L=1,L1=0,L2=1)
:- cycle(CYCLE, I, I', L1, L2), I'-I >= L1+L2, depth(I', DEPTH'), L1 < DEPTH', L2+DEPTH_2-1 < DEPTH'+(I'-I+1), cycle_end(I2, CYCLE), I'+L1 < I2, depth(I2, DEPTH_2), DEPTH'-1 < L1+DEPTH_2, L2 != 0.
%                               ‾‾‾‾‾‾‾‾‾‾‾‾‾‾                    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                       ‾‾‾‾‾‾‾‾‾‾‾

% Cycles may not have side chains which are longer than their head or tail.
% ---> sacrifice depth beneth first cycle branch-off and distance between side-chain branch-off and second cycle branch-off to gain both cycle heights and side chain
%               1
%              #  ...   <==>  (I_BASE - I) + DEPTH' -1 < SIDE_DEPTH + L1 + L2
%             + #       <==>  (5-3) + 4-1 < 2+2+2
%            +   #      <==>  2 + 3 < 2+2+2
%       # # #   / .     <==>  5 < 6
%          #   /
%         #   /
%        + # /
%       +   #
%      +     .
%      e.g. C₄C₃1C₂(C₅)C₁(C₉1)C₆C₇C₈           cycle(1,I=2,I'=3,L=1,L1=0,L2=1)
%           C₄1C₃C₂(C₅)C₁1C₆C₇(C₉)C₈           cycle(1,I=2,I'=4,L=2,L1=0,L2=0)
%           C₄C₃1C₂(C₅C₆)C₁1C₇C₈C₉             cycle(1,I=2,I'=3,L=1,L1=0,L2=0)
:- cycle(CYCLE, I, I', L1, L2), I'-I >= L1+L2, side_chain(I_BASE, SIDE_DEPTH), I <= I_BASE, I_BASE < I', depth(I', DEPTH'), L1 < DEPTH', SIDE_DEPTH < (I'-I_BASE)+DEPTH', L2 < DEPTH'+(I'-I+1), (I_BASE-I)+DEPTH'-1 < SIDE_DEPTH+L1+L2.
%                               ‾‾‾‾‾‾‾‾‾‾‾‾‾‾                                                                              ‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

% Cross cycles need to have longer path on main chain than on side chain.
% ---> sacrifice lenth between cycle branch-off to gain both cycle heights
%      # # # + 1 # # #  <==>  I + I'-I'' - 1 < L1 + L2
%          #     #      <==>  3 + 8-8 -1 < 2+2
%          # - - #      <==>  2 < 4
%      e.g. C₄C₃C₂(C₅1)C₁C₆(C₉1)C₇C₈           cross_cycle(1,I=2,I'=6,L1=1,L2=1)
%           C₄C₃C₂(C₅1)(C₆)C₁C₇(C₁₀1)C₈C₉      cross_cycle(1,I=2,I'=7,L1=1,L2=1)
%           C₄C₃(C₅)C₂(C₆C₇1)C₁C₈1C₉C₁₀        cross_cycle(1,I=2,I'=8,L1=2,L2=0)
:- cross_cycle(CYCLE, I, I', L1, L2), parent(1, 1, I''), I+L1 < I'', I+I'-I''-1 < L1+L2.
%                                                        ‾‾‾‾‾‾‾‾‾‾‾
% ---> sacrifice left side to gain first cycle height and depth beneth second cycle marker
%      + + # # # # # # # #  <==>  DEPTH - 1 < L1 + DEPTH_2
%          # -     +        <==>  3 -1 < 1+2
%          +   ‾ - #        <==>  2 < 3
%                  #
%      e.g. C₅C₄(C₆)C₃(C₇C₈1)C₂C₁C₉C₁₀(C₁₃1)C₁₁C₁₂         cross_cycle(1,I=3,I'=10,L1=2,L2=1)
%           C₅C₄C₃(C₆C₇)(C₈1)C₂C₁C₉(C₁₃1C₁₄)C₁₀(C₁₂)C₁₁    cross_cycle(1,I=3,I'=9,L1=1,L2=1)
%           C₅C₄(C₆)C₃(C₇1C₈)C₂C₁C₉(C₁₃1C₁₄)C₁₀C₁₁C₁₂      cross_cycle(1,I=3,I'=9,L1=1,L2=1)
%      (NOTE: inactive for smaller atom counts, where mostly L1 == 1)
:- cross_cycle(CYCLE, I, I', L1, L2), depth(I, DEPTH), L1 < DEPTH, cycle_end(I2, CYCLE), I'+L2 < I2, depth(I2, DEPTH_2), L2+DEPTH_2 < (I-1)+DEPTH, DEPTH-1 < L1+DEPTH_2, L2 != 0.
%                                                      ‾‾‾‾‾‾‾‾‾‾‾                       ‾‾‾‾‾‾‾‾‾‾‾                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
% ---> sacrifice right side to gain depth beneth first cycle marker and second cycle height
%      # # # # # # # # + +  <==>  DEPTH' - 1 < L2 + DEPTH_1
%            +     - #      <==>  3 -1 < 1+2
%            # - ‾   +      <==>  2 < 3
%            #
%      e.g. C₅C₄C₃(C₆1)C₂C₁C₇C₈C₉1                 cross_cycle(1,I=3,I'=9,L1=1,L2=0)
%           C₄C₃C₂(C₅1)(C₆)C₁C₇C₈C₉1               cross_cycle(1,I=2,I'=9,L1=1,L2=0)
%           C₅C₄(C₆)C₃(C₇1C₈)C₂C₁C₉(C₁₂1)C₁₀C₁₁    cross_cycle(1,I=3,I'=9,L1=1,L2=1)
:- cross_cycle(CYCLE, I, I', L1, L2), depth(I', DEPTH'), L2 < DEPTH', cycle_start(I1, CYCLE), I+L1 < I1, depth(I1, DEPTH_1), I+L1+DEPTH_1-1 <= MAX_LEN, max_len(MAX_LEN), DEPTH'-1 < L2+DEPTH_1, L1 != 0.
%                                                        ‾‾‾‾‾‾‾‾‾‾‾‾                         ‾‾‾‾‾‾‾‾‾‾                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

% Cross cycles may not have side chains which are longer than their head or tail; outer cycle prohibits all side chains.
% ---> sacrifice both sides to gain side chain and both cycle heights
%                  #
%      + + # # # # # # +  <==>  DEPTH + DEPTH' - 2 < SIDE_DEPTH + L1 + L2
%          #     _ - #    <==>  3+2-2 < 1+2+1
%          # - ‾          <==>  3 < 4
%      --> side chain on left side of main chain (1st branch of root atom) or on the root atom
%          e.g. C₄1C₃(C₅)(C₆)C₂C₁C₇C₈C₉1           cross_cycle(1,I=4,I'=9,L1=0,L2=0)
%               C₅1C₄(C₆)C₃C₂C₁C₇C₈C₉1             cross_cycle(1,I=5,I'=9,L1=0,L2=0)
%               C₄1C₃C₂C₁(C₇C₈)(C₉)C₅1C₆           cross_cycle(1,I=4,I'=5,L1=0,L2=0)
:- cross_cycle(CYCLE, I, I', L1, L2), side_chain(I_BASE, SIDE_DEPTH), 1 <= I_BASE, I_BASE < I, depth(I, DEPTH), L1 < DEPTH, SIDE_DEPTH < (I-I_BASE)+DEPTH, depth(I', DEPTH'), L2 < DEPTH', I+DEPTH <= I'+DEPTH', DEPTH+DEPTH'-2 < SIDE_DEPTH+L1+L2.
%                                                                                                               ‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                    ‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
%      --> side chain on right side of main chain (2nd branch of root atom)
%          e.g. C₅C₄1C₃C₂C₁(C₁₁)C₆(C₉C₁₀)C₇C₈1           cross_cycle(1,I=4,I'=8,L1=0,L2=0)
%               C₅C₄1C₃C₂C₁(C₁₂)C₆(C₁₀C₁₁)C₇(C₉)C₈1      cross_cycle(1,I=4,I'=8,L1=0,L2=0)
%               C₅C₄1C₃C₂(C₆)(C₇)C₁C₈C₉(C₁₂C₁₃)C₁₀C₁₁1   cross_cycle(1,I=4,I'=11,L1=0,L2=0)
%      (NOTE: inactive for smaller atom counts, side chains on left side of main chain prevent longer side chains on right side of main chain)
:- cross_cycle(CYCLE, I, I', L1, L2), I+I'-I''-1 >= L1+L2, parent(1, 1, I''), side_chain(I_BASE, SIDE_DEPTH), I'' <= I_BASE, I_BASE < I', depth(I, DEPTH), L1 < DEPTH, I+DEPTH-1+L1 < I'', depth(I', DEPTH'), L2 < DEPTH', I+DEPTH <= I'+DEPTH', SIDE_DEPTH < (I'-I_BASE)+DEPTH', DEPTH+DEPTH'-2 < SIDE_DEPTH+L1+L2.
%                                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                                                                                                 ‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                    ‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
% ---> sacrifice right side and left spanning distance to gain side chain and both cycle heights
%                #
%      # # # # + # # # # + +  <==>  (I - I_BASE - 1) + (DEPTH' - 1) < SIDE_DEPTH + L1 + L2
%            #     _ - #      <==>  (3-1-1)+3-1 < 1+2+1
%            # - ‾     .      <==>  3 < 4
%      --> side chain on left side of main chain (1st branch of root atom) or on the root atom
%          e.g. C₄C₃1(C₅)C₂(C₆)C₁(C₉)C₇C₈1         cross_cycle(1,I=3,I'=8,L1=0,L2=0)
%               C₄C₃(C₅)C₂(C₆1)C₁(C₉)C₇1C₈         cross_cycle(1,I=2,I'=7,L1=1,L2=0)
%               C₄C₃C₂1C₁(C₈C₉)C₅C₆1C₇             cross_cycle(1,I=2,I'=6,L1=0,L2=0)
:- cross_cycle(CYCLE, I, I', L1, L2), side_chain(I_BASE, SIDE_DEPTH), 1 <= I_BASE, I_BASE < I, depth(I', DEPTH'), L2 < DEPTH', (I-I_BASE-1)+DEPTH'-1 < SIDE_DEPTH+L1+L2.
%                                                                                                                 ‾‾‾‾‾‾‾‾‾‾‾‾
%      --> side chain on right side of main chain (2nd branch of root atom)
%          e.g. C₄C₃C₂1(C₅)C₁C₆(C₉)C₇C₈1           cross_cycle(1,I=2,I'=8,L1=0,L2=0)
%               C₄C₃(C₅)C₂(C₆1)C₁C₇(C₁₀)C₈1C₉      cross_cycle(1,I=2,I'=8,L1=1,L2=0)
%               C₄C₃C₂(C₅1)(C₆)C₁C₇(C₁₀)C₈1C₉      cross_cycle(1,I=2,I'=8,L1=1,L2=0)
:- cross_cycle(CYCLE, I, I', L1, L2), I+I'-I''-1 >= L1+L2, parent(1, 1, I''), I+L1 < I'', side_chain(I_BASE, SIDE_DEPTH), I'' <= I_BASE, I_BASE < I', depth(I', DEPTH'), L2 < DEPTH', (I-1+I_BASE-1)+DEPTH'-1 < SIDE_DEPTH+L1+L2.
%                                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                    ‾‾‾‾‾‾‾‾‾‾‾                                                                                ‾‾‾‾‾‾‾‾‾‾‾‾
% ---> sacrifice left side and right spanning distance to gain side chain and both cycle heights
%              # +
%      + # # # # + # # # #  <==>  (I' - I_BASE - 1) + (DEPTH - 1) < SIDE_DEPTH + L1 + L2
%        # - _     #        <==>  (3-1-1)+2-1 < 1+1+2
%              ‾ - #        <==>  2 < 4
%      --> side chain on left side of main chain (1st branch of root atom) or on the root atom
%          e.g. C₄C₃1C₂(C₅)C₁C₆(C₉1)C₇C₈           cross_cycle(1,I=3,I'=6,L1=0,L2=1)
%               C₄C₃1C₂(C₅C₆)C₁C₇1C₈C₉             cross_cycle(1,I=3,I'=7,L1=0,L2=0)
%               C₅1C₄(C₆)C₃C₂C₁C₇1C₈C₉             cross_cycle(1,I=5,I'=7,L1=0,L2=0)
:- cross_cycle(CYCLE, I, I', L1, L2), side_chain(I_BASE, SIDE_DEPTH), 1 <= I_BASE, I_BASE < I, depth(I, DEPTH), L1 < DEPTH, SIDE_DEPTH < (I-I_BASE)+DEPTH, parent(1, 1, I''), I+DEPTH-1+L1 < I'', I'' <= I', (I_BASE-1+I'-I'')+DEPTH-1 < SIDE_DEPTH+L1+L2.
%                                                                                                               ‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾
%      --> side chain on right side of main chain (2nd branch of root atom)
%          e.g. C₄1C₃C₂(C₅)(C₆)C₁C₇(C₁₀)C₈1C₉      cross_cycle(1,I=4,I'=8,L1=0,L2=0)
%               C₆C₅1C₄C₃C₂C₁(C₁₃)C₇(C₁₂)C₈(C₁₁1)C₉C₁₀cross_cycle(1,I=5,I'=8,L1=0,L2=1)
%               C₆C₅1C₄C₃C₂C₁(C₁₃)C₇(C₁₁C₁₂)C₈1C₉C₁₀cross_cycle(1,I=5,I'=8,L1=0,L2=0)
:- cross_cycle(CYCLE, I, I', L1, L2), I+I'-I''-1 >= L1+L2, parent(1, 1, I''), side_chain(I_BASE, SIDE_DEPTH), I'' <= I_BASE, I_BASE < I', depth(I, DEPTH), L1 < DEPTH, I+DEPTH-1+L1 < I'', (I'-I_BASE-1)+DEPTH-1 < SIDE_DEPTH+L1+L2.
%                                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                                                                                                 ‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

% ---> move up the lower cycle height
%      --> have side cycle with chain beneth I2 --> need to keep 2 branching for DEPTH_1 > 1 and 1 branching for DEPTH_1 == 1 at I2
%               1                      1
%              +  ...                 +  ...
%             + +                    + +
%            +   +                  +   + +
%           +   / +     ==>        +   + +
%          +   /                  +    __ +
%         +   /                  + --‾‾
%        + + /                  +
%       +   +                  +
%      +     +                +
%      e.g. C₇C₆C₅C₄(C₈1)C₃C₂C₁(C₁₄1C₁₅)C₉C₁₀C₁₁C₁₂C₁₃     cycle(1,I=2,I'=4,L=2,L1=1,L2=1)
%      (NOTE: inactive for smaller atom counts, where either L1 or L2 are usually zero)
:- cycle(CYCLE, I, I', L1, L2), L1 > 0, L2 != 0, I'-I >= L1+L2, cycle_start(I1, CYCLE), depth(I1, DEPTH_1), cycle_end(I2, CYCLE), I1 < I2, I'+L1 < I1, I'+L1+L2 < I2, branching(I2, BRANCHING), (BRANCHING,DEPTH_1) < (2,2), BRANCHING < 3.
%                                                ‾‾‾‾‾‾‾‾‾‾‾‾‾‾                                                                   ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾
%      --> have side cycle without chain beneth I2
%               1                      1
%              +  ...                 +  ...
%             + +                    + +
%            +   +                  +   + +
%           +   /       ==>        +   + +
%          +   /                  +    __ +
%         +   /                  + --‾‾
%        + + /                  +
%       +   + +                +
%      +     +                +
%      e.g. C₆C₅C₄(C₇1)C₃C₂C₁(C₁₂1)C₈C₉C₁₀C₁₁  cycle(1,I=2,I'=4,L=2,L1=1,L2=1)
%           C₆C₅C₄(C₇1)C₃C₂C₁(C₁₃1)C₈C₉C₁₀C₁₁C₁₂cycle(1,I=2,I'=4,L=2,L1=1,L2=1)
%           C₆C₅(C₇)C₄(C₈1)C₃C₂C₁(C₁₃1)C₉C₁₀C₁₁C₁₂cycle(1,I=2,I'=4,L=2,L1=1,L2=1)
:- cycle(CYCLE, I, I', L1, L2), L1 > 0, L2 != 0, I'-I >= L1+L2, cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I1 < I2, I'+L1 < I1, I'+L1+L2 < I2, not branching(I2, _).
%                                                ‾‾‾‾‾‾‾‾‾‾‾‾‾‾                                               ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾‾‾‾‾‾
%      --> have branch down cycle from non-root atom --> need to keep 2 branching for DEPTH_2 > 1 and 1 branching for DEPTH_2 == 1 at I1
%               1                      1
%              +  ...                 +  ...
%             +                      + +
%            + \                    +   + +
%           +  |       ==>         +   + +
%          +   /                  +    __/
%         +   /                  + --‾‾
%        + + /                  +
%       +   + +                +
%      +     +                +
%      e.g. C₆C₅C₄(C₇1)C₃C₂1C₁C₈C₉C₁₀C₁₁C₁₂    cycle(1,I=3,I'=4,L=1,L1=1,L2=0)
%           C₆C₅C₄(C₇1)C₃C₂1C₁(C₁₂)C₈C₉C₁₀C₁₁  cycle(1,I=3,I'=4,L=1,L1=1,L2=0)
%           C₆C₅C₄C₃C₂(C₇)C₁C₈1C₉C₁₀(C₁₃1)C₁₁C₁₂cycle(1,I=9,I'=10,L=1,L1=1,L2=0)
:- cycle(CYCLE, I, I', L1, L2), L1 > 0, L2 == 0, I'-I >= L1+L2, cycle_end(I2, CYCLE), I'+L1 < I2, depth(I2, DEPTH_2), cycle_start(I1, CYCLE), I1 < I, I1 != 1, branching(I1, BRANCHING), (BRANCHING,DEPTH_2) < (2,2), BRANCHING < 3.
%                                                ‾‾‾‾‾‾‾‾‾‾‾‾‾‾                       ‾‾‾‾‾‾‾‾‾‾‾                                             ‾‾‾‾‾‾‾
%      --> have branch down cycle from root atom
%             1                      1
%            + \ ...                +  ... +
%           +  |       ==>         +        + +
%          +   /                  +        + +
%         +   /                  + __________/
%        + + /                  +
%       +   + +                +
%      +     +                +
%      e.g. C₅C₄C₃(C₆1)C₂C₁1C₇C₈C₉             cycle(1,I=2,I'=3,L=1,L1=1,L2=0)
%           C₅C₄C₃(C₆1)(C₇)C₂C₁1C₈C₉C₁₀        cycle(1,I=2,I'=3,L=1,L1=1,L2=0)
%           C₅C₄C₃(C₆1)C₂C₁1(C₁₀)C₇C₈C₉        cycle(1,I=2,I'=3,L=1,L1=1,L2=0)
:- cycle(CYCLE, I, I', L1, L2), L1 > 0, L2 == 0, I'-I >= L1+L2, cycle_end(I2, CYCLE), I'+L1 < I2, depth(I2, DEPTH_2), cycle_start(1, CYCLE), branching(1, BRANCHING), (BRANCHING,DEPTH_2) < (3,2), BRANCHING < 4.
%                                                ‾‾‾‾‾‾‾‾‾‾‾‾‾‾                       ‾‾‾‾‾‾‾‾‾‾‾

% ---> move up part of chain cycle to branch up cycle
%                 1                      1
%                +--_ ...               + + ...
%               +   /        ==>       +  _+
%      + + + + +   /                  +_-‾  +
%             +   /                  +       +
%            +--‾‾                  +
%           +                      +
%          +                      +
%      e.g. C₄1C₃(C₅)C₂C₁1C₆C₇                 cycle(1,I=2,I'=4,L=2,L1=0,L2=0)
%           C₅C₄C₃C₂C₁1C₆C₇(C₉)C₈1             cycle(1,I=6,I'=8,L=2,L1=0,L2=0)
%           C₅C₄1C₃(C₆C₇)C₂C₁1C₈C₉C₁₀C₁₁       cycle(1,I=2,I'=4,L=2,L1=0,L2=0)
:- cycle(CYCLE, I, I', 0, 0), depth(I', DEPTH'), side_chain(I_BASE, SIDE_DEPTH), I <= I_BASE, I_BASE < I', SIDE_DEPTH == DEPTH'-1 + I'-I_BASE, cycle_start(I1, CYCLE), I1 < I, branching(I1, BRANCHING), (BRANCHING,I1) < (3,2), BRANCHING < 4.
%                                                                                                                                                                      ‾‾‾‾‾‾‾

% ---> move the right cycle height to the left
%      + + + + + + + + + + + + +         + + + + + + + + + + + + +
%              + - _     +         ==>           +      _-‾
%              +     ‾ - +                       +  _-‾‾
%                        +                     + + +
%      e.g. C₅C₄C₃1C₂C₁(C₁₀)C₆(C₉1)C₇C₈        cross_cycle(1,I=3,I'=6,L1=0,L2=1)
%           C₅C₄C₃1(C₆)C₂C₁C₇(C₁₀1)C₈C₉        cross_cycle(1,I=3,I'=7,L1=0,L2=1)
%           C₅C₄C₃1C₂C₁(C₁₂)C₆(C₁₀1C₁₁)C₇(C₉)C₈cross_cycle(1,I=3,I'=6,L1=0,L2=1)
:- cross_cycle(CYCLE, I, I', L1, L2), L2 != 0, cycle_start(I1, CYCLE), branching(I1, BRANCHING), (BRANCHING,DEPTH_2) < (2,2), BRANCHING < 3, depth(I, DEPTH), max_len(MAX_LEN), I+DEPTH <=MAX_LEN+1, cycle_end(I2, CYCLE), I <= I1, I1 < I', I' <= I2, I1 < I2, depth(I2, DEPTH_2), DEPTH_2 < DEPTH-L1.
%                                                                                                                                                             ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾                       ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾‾ ‾‾‾‾‾‾‾‾

%:- not multi_bond(_, _).
%:- multi_bond(_, _).
%:- cycle(_, _, _, _, _).
%:- cross_cycle(_, _, _, _, _).
%:- not side_chain(_,_).
%:- side_chain(1,_).
%:- not cond(0).

%#show cycle/5. % (uncomment, to view cycle classification with smiles-vis.py script)

#show branching/2.
#show symbol/2.
#show multi_bond/2.
#show cycle_start/2.
#show cycle_end/2.
