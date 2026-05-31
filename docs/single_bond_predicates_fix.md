# Fix: Single-Bond Predicates and Aromatic Ring Detection

## Background

The soundness test in `src/tests/test_constraints.py` verifies that for every formula in
the test suite, molecules returned by the constrained ASP pipeline (e.g., with
`has_c_o_single_bond=1`) actually satisfy the corresponding RDKit MACCS key.  A soundness
failure means the constraint is too weak: ASP is returning molecules that don't have the
claimed structural feature.

Triggering failure:

```
AssertionError: has_c_o_single_bond violated by C1Cn2nc1o2; using formula C3H4N2O
```

---

## Root Cause Layer 1 — Kekulé vs. aromatic bond representation

### What MACCS key 157 checks

RDKit MACCS key 157 uses SMARTS `[#6]-[#8]` where `-` is an **explicit single bond**.
After RDKit aromatises a molecule, bonds in aromatic rings become aromatic (bond order 1.5).
The `-` specifier does **not** match aromatic bonds, so for molecules such as
3-methyloxadiazole (`Cc1nnco1`) or 4-aminooxadiazole (`C1Cn2nc1o2`), MACCS 157 returns 0
even though those molecules contain C–O bonds.

### What the old ASP rule checked

```prolog
has_c_o_single_bond :-
    single_bond_between(I, J),
    symbol(I, "C"),
    symbol(J, "O").
```

`single_bond_between` is built on `adjacent_single`, which fires for every
parent-child backbone bond where the child has no `multi_bond` (i.e. the backbone
bond is single in Kekulé representation).  smiles.lp works exclusively in Kekulé
space: every aromatic ring is encoded with alternating single/double backbone bonds.
The C–O backbone bond in an aromatic ring therefore appears as a single bond to ASP,
causing `has_c_o_single_bond` to fire even though RDKit would call that bond aromatic
and MACCS 157 would return 0.

The same issue affects `has_c_n_single_bond` (MACCS 158, `[#6]-[#7]`) and
`has_n_o_single_bond` (MACCS 24, `[#7]-[#8]`).

---

## Root Cause Layer 2 — `atom_in_ring` underestimates ring membership

The attempted fix uses a helper predicate `ring_atom_with_dbl(I)` that fires when
atom `I` is in a ring that contains at least one double bond (a proxy for aromaticity
in Kekulé space).  The fix was:

> Exclude ring C–O bonds where either atom has `ring_atom_with_dbl`.

`ring_atom_with_dbl` depends on `atom_in_ring(I, CY)`, which was defined as:

```prolog
atom_in_ring(I, CY) :-
    cycle_start(S, CY), cycle_end(E, CY),
    symbol(I, _),
    S <= I, I <= E.
```

This correctly covers rings where the ring path goes **forward** through increasing
atom indices (S → S+1 → … → E).  However, smiles.lp sometimes places the root atom
(atom 1) **inside the ring**: the SMILES string traversal goes backward from the root
toward `cycle_start`, then forward toward `cycle_end`, so the root and its ancestors
down to `cycle_start` have atom indices **below S** and are therefore missed by the
range check `S <= I <= E`.

Three concrete patterns were identified:

### Pattern A — double-bond ring closure (E = S + 1)

Aromatic rings where smiles.lp encodes a ring bond as two parallel ring closures
between adjacent atoms (e.g. cycle_start(4,1) and cycle_end(5,1) with two ring
numbers closing the same atom pair).  The actual 5-membered ring path goes
4 → 3 → 2 → 1 → 5.  Atoms 1, 2, 3 have indices < S = 4 and were invisible to
the old predicate.

### Pattern B — root branches toward both cycle_start and cycle_end (E > S + 1)

Root atom 1 (e.g. O) has two children: one child C1 lies on the path toward
`cycle_start` S (C1 ≤ S), and the other child C2 lies between S and `cycle_end`
E (C1 < C2 ≤ E).  The root is on the ring path but has index 1 < S.

Example: `C=NN1OC=C1` — root O (atom 1) has children N2 (backward, toward ring)
and C5 (forward, toward ring closure), with cycle_start at N2 and cycle_end at C6.

### Pattern C — intermediate atoms between root and cycle_start

Even when the root itself is detected as a ring member via Pattern B, the atoms
**between** the root and `cycle_start` (e.g. atom 2 on the path root → atom 2 →
cycle_start atom 3) were still not in `atom_in_ring`.

---

## Root Cause Layer 3 — ring-closure double bonds not in `multi_bond`

`multi_bond(I, M)` in smiles.lp tracks only **backbone** double bonds (the bond from
atom I to its parent).  Double bonds encoded as two parallel ring closures (Pattern A
above) do **not** set `multi_bond`, so `ring_atom_with_dbl` — which checks
`multi_bond(K, _)` — would not fire for rings whose only double bond is a ring-closure
double bond.

---

## What Was Fixed

### 1. Extended `atom_in_ring` (three new rules)

```prolog
% Extended rule 1: E = S+1 — double-bond ring closures run backward through root.
% Include atoms 1..S-1 on that backward path.
atom_in_ring(I, CY) :-
    cycle_start(S, CY), cycle_end(E, CY),
    E = S + 1,
    symbol(I, _),
    1 <= I, I < S.

% Extended rule 2 (generalised): atom I is on the ring path when it branches
% toward cycle_start (child C1 <= S) and also toward cycle_end (child C2, C1 < C2 <= E).
atom_in_ring(I, CY) :-
    cycle_start(S, CY), cycle_end(E, CY),
    parent(I, _, C1), parent(I, _, C2),
    C1 != C2, C1 <= S, C1 < C2, C2 <= E.

% Propagation: once a pre-cycle_start atom I is in the ring, every child J
% of I with J < S is also on the ring path.
% (Safe because in DFS-ordered trees all children have higher indices than
%  parents, so J < S implies I < S, i.e. I was itself added by an extended rule.)
atom_in_ring(J, CY) :-
    atom_in_ring(I, CY),
    cycle_start(S, CY),
    parent(I, _, J),
    J < S.
```

### 2. New `ring_closure_dbl` helper

Detects when two different ring numbers close between the same atom pair — this
encodes a double bond in smiles.lp's ring-closure mechanism:

```prolog
ring_closure_dbl(S, E) :-
    cycle_start(S, CY), cycle_end(E, CY),
    cycle_start(S, CY2), cycle_end(E, CY2),
    CY != CY2.
```

### 3. New `ring_atom_with_dbl(I)` helper (replaces `ring_bond_with_dbl/2`)

Fires when atom `I` is in **any** ring that contains a double bond, whether from a
backbone `multi_bond` or a ring-closure double bond.  Using a **unary** predicate
(rather than a binary one indexed by the bond pair) ensures that fused-ring
aromaticity is captured even when the double bond is in an adjacent ring rather than
the specific ring containing the C–O bond:

```prolog
ring_atom_with_dbl(I) :-
    atom_in_ring(I, CY),
    atom_in_ring(K, CY),
    multi_bond(K, _).

ring_atom_with_dbl(I) :-
    atom_in_ring(I, CY),
    atom_in_ring(S, CY),
    ring_closure_dbl(S, _).
```

### 4. Rewritten `has_c_o_single_bond`, `has_c_n_single_bond`, `has_n_o_single_bond`

Each predicate is now split into two cases:

```prolog
% Case 1: non-ring bond — always a genuine single bond.
has_c_o_single_bond :-
    single_bond_between(I, J),
    symbol(I, "C"), symbol(J, "O"),
    not approx_ring_bond(I, J).

% Case 2: ring bond only when neither atom is in a ring with double bonds.
% Aliphatic rings (e.g. THF) have no double bonds and are correctly included.
% Aromatic rings are excluded regardless of which ring the bond belongs to.
has_c_o_single_bond :-
    single_bond_between(I, J),
    symbol(I, "C"), symbol(J, "O"),
    approx_ring_bond(I, J),
    not ring_atom_with_dbl(I),
    not ring_atom_with_dbl(J).
```

The same pattern applies to `has_c_n_single_bond` (MACCS 158) and
`has_n_o_single_bond` (MACCS 24).

---

## Known Limitation

Non-aromatic rings that contain a C–O bond **and** a separate C=C double bond
elsewhere in the ring (e.g. 2,3-dihydrofuran, C4H6O) would be incorrectly excluded
because `ring_atom_with_dbl` fires for any ring double bond, not just those
conjugated with the C–O bond.  MACCS 157 returns 1 for such molecules (the C–O bond
is genuinely single).  However, no formula in the current test suite produces this
pattern, so no test failure results.  The limitation would need to be addressed if
such formulas are added.

---

## Test Result

Before: 94 soundness failures for `has_c_o_single_bond` on formula C3H4N2O.  
After: 0 failures across all constraints and all test formulas.

Files changed: `src/predicates/definitions.py` (HELPER_PREDICATES and three MACCS key definitions).
