# ASP Completeness Bug Findings

Bugs found and fixed while debugging `test_constraint_completeness` failures.
All fixes are in `src/predicates/definitions.py` unless noted otherwise.

---

## 1. Wrong completeness assertion (`src/tests/test_constraints.py`)

**Bug:** `constrained_smiles[0:len(expected)] == expected` is a prefix check, not a subset check. Any extra element in `constrained_smiles` that sorts before an expected element causes a false failure.

**Fix:** Changed to:
```python
missing = sorted(set(expected) - set(constrained_smiles))
assert not missing, ...
```

---

## 2. `single_bond_between` missing ring closure bonds

**Bug:** `single_bond_between` is built from `adjacent_single` which uses `parent`. Ring closure bonds (`cycle_start`/`cycle_end`) are not in `parent`, so ring closure single bonds are invisible. Molecules where the *only* C-N (or similar) bond is a ring closure were never detected.

**Example:** `C#CC1ON1N` — the C-N bond is the ring closure; `single_bond_between(C, N)` was False.

**Fix:** Added to `single_bond_between` helper:
```asp
single_bond_between(I, J) :-
    cycle_start(I, CY), cycle_end(J, CY), not ring_closure_dbl(I, J).
single_bond_between(J, I) :-
    cycle_start(I, CY), cycle_end(J, CY), not ring_closure_dbl(I, J).
```

---

## 3. `double_bond_between` missing ring closure double bonds

**Bug:** `double_bond_between` only checks backbone `multi_bond(J, 2)`. Ring closure double bonds are encoded as two parallel ring labels (`ring_closure_dbl`) and were invisible.

**Example:** `C#CC1=C(N)N1` traversed from N — the ring C=C becomes a ring-closure double bond with no backbone `multi_bond`.

**Fix:** Added to `double_bond_between` helper:
```asp
double_bond_between(I, J) :- ring_closure_dbl(I, J).
double_bond_between(J, I) :- ring_closure_dbl(I, J).
```

---

## 4. `atom_in_ring` missing cycle_start/cycle_end atoms in sibling rings

**Bug:** When `cycle_start` and `cycle_end` are DFS siblings (both children of the same parent, not in ancestor-descendant relation), `ring_path_atom` finds no path between them, so neither is added to `atom_in_ring(I, CY)`.

**Example:** `C1=C(C2NO2)N1` — C(cycle_start) and N(cycle_end) are siblings; `atom_in_ring` only contained the parent C.

**Fix:** Added to `atom_in_ring` helper:
```asp
atom_in_ring(I, CY) :- cycle_start(I, CY).
atom_in_ring(I, CY) :- cycle_end(I, CY).
```

---

## 5. `atom_in_ring` missing backward-through-root ring members

**Bug:** For rings where `cycle_start` and `cycle_end` are in different subtrees of the DFS root (a "backward-through-root" ring), the ring path goes up to the root and back down another branch. The extended rules (rule 2) add the *branching atom* but not atoms *between the branching atom and cycle_end*.

**Example:** `N[SH]1NOO1` — DFS root N(1) branches to S(2)=cycle_start and O(4). O(4) is between N(1) and O(5)=cycle_end. O(4) was missing from `atom_in_ring`, giving `ring_size=3` instead of 4.

**Diagnosis:** Inspect `atom_in_ring` and `ring_size` in a clingo debug run. If `ring_size` is wrong for a ring you expect to be size N, look for missing atoms between the branching node and `cycle_end`.

**Fix:** Added forward-propagation rule to `atom_in_ring`:
```asp
% Forward propagation toward cycle_end, avoiding cycle_start's subtree.
atom_in_ring(J, CY) :-
    atom_in_ring(I, CY),
    cycle_end(E, CY),
    cycle_start(S, CY),
    parent(I, _, J),
    I < J, J <= E,
    not is_descendant(S, J).
```

---

## 6. `ring_double_bond_count` using span instead of ring membership

**Bug:** `ring_double_bond_count` used `S < I <= E` (index span) to find ring atoms. In bicyclic compounds this span includes atoms from *other* rings, producing an inflated double-bond count. This caused non-aromatic rings to be misclassified as aromatic.

**Example:** `C1=CC12N=CN2` — ring 2 (4-membered, `cycle_start=2, cycle_end=6`) has span 2..6 which includes atom 4 (in ring 1). Counting `multi_bond(4,2)` gave DC=2 for ring 2, making it look aromatic when it isn't.

**Fix:** Changed to use ring membership:
```asp
ring_double_bond_count(C, N) :-
    cycle_start(_, C),
    N = #count { I : atom_in_ring(I, C), multi_bond(I, 2) }.
```

---

## 7. `aromatic_atom` using span instead of ring membership

**Bug:** `aromatic_atom` used `S <= I <= E` span check. Same problem as #6: atoms from other rings within the span were falsely marked aromatic, blocking the C=C double bond predicate from firing.

**Fix:**
```asp
aromatic_atom(I) :-
    atom_in_ring(I, C),
    aromatic_ring_cycle(C).
```

---

## 8. `aromatic_ring_cycle` not excluding cumulenes

**Bug:** A 5-membered ring with 2 *consecutive* double bonds (cumulene, e.g., N=C=C) satisfies the DC=2 criterion for aromaticity, but is not aromatic. This caused cumulene ring C-O bonds to be incorrectly excluded.

**Fix:** Added `consecutive_ring_doubles` helper and guard:
```asp
consecutive_ring_doubles(C) :-
    cycle_start(S, C), cycle_end(E, C),
    multi_bond(I, _), multi_bond(J, _),
    parent(I, _, J), S < J, J <= E, S < I, I <= E.

% Added to both aromatic_ring_cycle rules:
not consecutive_ring_doubles(C).
```

---

## 9. Ring bond predicates using `ring_atom_with_dbl` / `not ring_atom` as aromatic guard

**Bug:** `has_c_o_single_bond`, `has_c_n_single_bond` used `not ring_atom_with_dbl(I)` (any ring with any double bond) as a proxy for "not aromatic." This blocked all ring bonds in rings with double bonds — including non-aromatic rings like cyclopentadiene or cumulene rings.

Similarly, `has_c_c_double_bond`, `has_c_n_double_bond`, and alkene pattern predicates (`has_c_eq_c_n`, `has_c_eq_c_cc`, `has_c_eq_c_qq`, `has_c_eq_c_aa`, `has_n_eq_a`) used `not ring_atom(I)` (completely excludes all ring atoms) when they should only exclude *aromatic* ring atoms.

**Fix:** Replaced with `not aromatic_atom(I), not aromatic_atom(J)` (after fixing `aromatic_atom` in #7).

---

## 10. `has_n_o_single_bond` first rule too restrictive

**Bug:** The first rule used `not ring_atom(I), not ring_atom(J)` — this blocks exocyclic N-O bonds where N is in a ring but O is pendant.

**Example:** `ON1N[SH]1O` — N is a ring atom; the O-N bond is non-ring (exocyclic). `not ring_atom(N)` blocked it.

**Fix:** Changed to `not approx_ring_bond(I, J)` (like `has_c_o_single_bond` already did), which only excludes bonds where *both* atoms are in the same ring.

---

## 11. DoU check in `bridge.py` ignoring S4 sulfur

**Bug:** `_dou_numerator` in `src/asp/bridge.py` did not include sulfur. Valence-4 sulfur (S4) contributes to DoU like carbon (adds 2 per atom). For formulas like H2SO2 with S4 (DoU=1), the check computed DoU=0 and skipped the solve entirely.

**Fix:** Added `2 * num_sulfurs` to treat all S as potentially S4 (maximum DoU estimate):
```python
2 * (atom_counts.get("num_carbons", 0) + atom_counts.get("num_sulfurs", 0))
```

---

## General debugging approach

When a molecule appears in `base_smiles` but not `constrained_smiles`:

1. Check if the predicate is even *derivable* by removing the hard constraint (`re.sub`) and adding `#show predicate/0.` — if it's False in the unconstrained run, the constraint is correctly eliminating it (but your rules are incomplete).

2. Use `#show double_bond_between/2`, `#show symbol/2`, `#show multi_bond/2`, `#show ring_closure_dbl/2`, `#show atom_in_ring/2`, `#show ring_size/2` to inspect the internal state for a specific molecule.

3. Key question: is the bond you expect a **backbone bond** (in `parent`) or a **ring closure bond** (in `cycle_start`/`cycle_end`)? Backbone bonds are in `single_bond_between`/`double_bond_between` by default; ring closure bonds require the ring_closure_dbl rules.

4. For ring membership issues: check `atom_in_ring` and `ring_size`. If `ring_size` is wrong, look for missing atoms using `ring_path_atom` and check if the ring is a backward-through-root ring.

5. For aromatic misclassification: check `aromatic_ring_cycle` and `ring_double_bond_count`. In bicyclic molecules, verify DC is computed per-ring (not spanning multiple rings).
