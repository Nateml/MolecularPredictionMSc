# ASP Predicate Soundness: Root Causes and Fixes

## Background

The soundness test (`test_constraint_soundness`) forces each ASP predicate to be True,
runs the solver, and checks that every generated molecule satisfies the corresponding
RDKit MACCS key. Failures mean the ASP predicate accepts molecules that don't actually
have the structural feature.

35 soundness failures were identified and resolved in two categories.

---

## Category A: Ring Membership / Ring Size Bug

### Root Cause

All ring-size and ring-membership predicates used a **SMILES span heuristic**:

```asp
ring_size_3_count(N) :-
    N = #count { CYCLE : cycle_start(I1, CYCLE), cycle_end(I2, CYCLE), I2 - I1 + 1 = 3 }.
```

The span `I2 - I1 + 1` counts all atoms whose index falls between the ring-open atom
and the ring-close atom. This overcounts when **branch atoms appear between the ring
opener and closer in atom-index space**.

In ASP's SMILES model, `parent(I, 0, I+1)` forces branch-0 (the first child) to always
be atom `I+1`. Side branches (B=1, B=2, ...) have higher indices than the entire B=0
subtree. When a ring closes via a B=1+ child, the B=0 subtree atoms (which are NOT on
the ring path) fall between the ring opener and ring closer in index space and are
incorrectly counted.

### Concrete Example

`C#CC1=C(N)N1` (formula C4H4N2, ring should be 3-membered):

- atom 3: C (ring opener, `cycle_start(3, 1)`)
- atom 4: C (B=0 child of 3, ring backbone)
- atom 5: N (B=0 child of 4 — branch N, **not in ring**)
- atom 6: N (B=1 child of 4 — ring closer, `cycle_end(6, 1)`)

Old span: `6 - 3 + 1 = 4` → `ring_size_4_count = 1` (wrong)  
With fix: ring path `{3, 4, 6}` = 3 atoms → `ring_size_3_count = 1` (correct)

Atom 5 (the exocyclic N) was counted because `3 ≤ 5 ≤ 6`.

### Fix

Added an `is_descendant`-based ring path computation:

```asp
is_descendant(A, A) :- symbol(A, _).
is_descendant(A, C) :- parent(A, _, B), is_descendant(B, C).

ring_path_atom(CY, I) :-
    cycle_start(S, CY), cycle_end(E, CY),
    is_descendant(S, I), is_descendant(I, E).

ring_size(CY, N) :-
    cycle_start(_, CY),
    N = #count { I : atom_in_ring(I, CY) }.
```

`is_descendant(S, I)` and `is_descendant(I, E)` together select only atoms that lie on
the unique tree path from the ring opener to the ring closer. Branch atoms that don't
lead toward the ring closer are excluded.

The `atom_in_ring` basic rule was updated from the span rule to:

```asp
atom_in_ring(I, CY) :- ring_path_atom(CY, I).
```

`ring_atom`, `ring_size_N_count`, and `has_q_in_Xm_ring` predicates were all updated to
use the corrected computation.

### Predicates Fixed

- `has_3m_ring`, `has_4m_ring`, `has_5m_ring`, `has_6m_ring`, `has_6m_ring_gt1`,
  `has_7m_ring`, `has_8m_or_larger_ring`
- `has_s_in_ring`, `has_o_in_ring`, `has_n_in_ring`
- `has_heteroatom_in_ring_gt1`, `has_heterocycle`
- `has_q_in_3m_ring`, `has_q_in_4m_ring`, `has_q_in_5m_ring`, `has_q_in_6m_ring`
- `has_ring_junction_atom`, `has_q_ch2_a_gt1`

### Known Remaining Limitation

For **bicyclic** molecules, the backbone ring path can still overestimate the SSSR ring
size when a second ring closure provides a shortcut within the first ring's path. Example:

- Ring 1 backbone path: S → A → B → E (4 atoms)
- But ring closure 2 connects A directly to E (3-membered shortcut)
- RDKit SSSR sees a 3-membered ring; ASP computes size 4

This is harder to fix without full SSSR computation in ASP and occurs only in polycyclic
molecules. The fix covers the much more common monocyclic and branch-atom cases.

---

## Category B: Aromatic Bond Mismatch (Kekulé vs. RDKit)

### Root Cause

ASP generates SMILES in **Kekulé form** with explicit alternating double bonds
(`multi_bond(I, 2)`). The `smiles-vis.py` script feeds this to RDKit, which sanitizes
and **aromatizes** the molecule. The resulting canonical SMILES uses lowercase atoms
and aromatic bonds (e.g., `c1ccccc1`).

This creates a fundamental bidirectional mismatch for any molecule that RDKit will
aromatize:

**Direction 1 — Soundness failure (constraint too loose):**  
ASP fires `has_c_c_double_bond` because `multi_bond(I, 2)` exists for a ring carbon in
Kekulé form. The solver generates the molecule. After RDKit aromatization, the C-C bond
becomes an aromatic bond (`:`) not a double bond (`=`). MACCS key 99 (`[C]=[C]`) matches
only explicit double bonds → returns 0 → soundness violation.

**Direction 2 — Completeness failure (constraint too strict):**  
ASP applies `not has_c_c_double_bond`. This blocks all molecules with `multi_bond(I, 2)`
between carbons. But benzene in Kekulé form *requires* `multi_bond` → benzene and all
aromatic C-C-containing molecules are excluded from the candidate set.

### Why It Can't Be Fully Fixed in ASP

Correctly predicting which rings RDKit will aromatize requires implementing RDKit's
aromaticity perception: ring size + π electron count + Hückel's rule (4n+2) + atom
hybridization checks + fused ring handling. ASP has no notion of orbital hybridization
or electron counting. Any heuristic approximation (e.g., alternating double bonds in a
6-membered ring) fails for heteroaromatic rings, fused systems, and edge cases like
4-membered rings that some aromaticity models accept.

The existing `aromatic_ring_cycle` helper uses span-size + double-bond-count but is
unreliable for the same reasons.

### Partial Fix: Restrict Double Bond Predicates to Non-Ring Bonds

The mismatch only applies to **ring** double bonds — non-ring double bonds (alkenes,
imines, carbonyl-adjacent bonds, etc.) are never aromatized and are always sound.

For the bond-type predicates, all definitions were updated to exclude ring atoms:

```asp
% Example: has_c_c_double_bond before (unsound for aromatic molecules)
has_c_c_double_bond :- double_bond_between(I, J), symbol(I, "C"), symbol(J, "C").

% After fix (non-ring only — always sound)
has_c_c_double_bond :-
    double_bond_between(I, J),
    symbol(I, "C"), symbol(J, "C"),
    not ring_atom(I), not ring_atom(J).
```

Same pattern applied to `has_c_n_double_bond`, `has_n_o_single_bond`, `has_n_eq_a`,
and the compound predicates `has_c_eq_c_n`, `has_c_eq_c_cc`, `has_c_eq_c_qq`,
`has_c_eq_c_aa`.

#### Soundness and completeness analysis

**Soundness — holds unconditionally.**
If ASP fires the constraint, it found a double bond where both endpoints satisfy
`not ring_atom`. Non-ring atoms are never part of an aromatic ring, so RDKit's
sanitization never converts this bond to an aromatic bond. The explicit double bond
survives into the RDKit molecule, and the corresponding MACCS key matches. No false
positives are possible.

**Completeness — holds only for non-ring bonds.**
If MACCS key = 1, the molecule has the relevant double bond. Two sub-cases:

| Case | Example | `not ring_atom` holds? | Constraint fires? |
|---|---|---|---|
| Non-ring double bond | propene (C=C), imine (C=N) | ✓ | ✓ complete |
| Ring double bond, non-aromatic | cyclohexene | ✗ | ✗ missed |
| Ring double bond, aromatic | benzene | n/a | MACCS=0, not required |

Cyclohexene is a **completeness gap**: MACCS 99 = 1 (ring C=C, non-aromatic), but
`ring_atom` is True for the ring carbons so the constraint doesn't fire. When the neural
net predicts `has_c_c_double_bond=1` and the true molecule has a ring alkene, the
constraint is silently skipped — no wrong candidates are admitted, but the right
candidate isn't positively constrained for. This is a conservative approximation.

In practice, aromatic molecules dominate typical MS datasets, so most MACCS 99 = 0
predictions correspond to aromatic systems (where the old predicate was catastrophically
wrong). The ring alkene gap is the smaller failure mode and is accepted as a tradeoff.

These 8 predicates are now **re-activated** (not unsupported) with the restricted
definition.

> **Open question — consider marking unsupported instead.**
> The completeness failure for ring alkenes (when predicting 1) may outweigh the pruning
> benefit, since completeness failures (true molecule excluded) are worse than soundness
> failures (extra wrong candidates admitted). The original predicate also had completeness
> failures — for aromatic molecules when predicting 0 — but that failure mode is arguably
> more common in MS datasets. If ring-alkene coverage in the evaluation set is
> significant, marking these 8 predicates `unsupported` (no constraint applied, no
> failures) is the safer choice. The current implementation keeps them active on the basis
> that the non-ring pruning benefit outweighs the rare ring-alkene miss.

### Remaining Unsupported — Marked `unsupported: True`

**Bond-type predicates where ring exclusion doesn't help:**
- `has_c_eq_c_ring` (MACCS 26) — *requires* the C=C to be in a ring; can't restrict to non-ring
- `has_ch23_q_ch23` (MACCS 86) — different failure mode (exotic strained ring structures)

**Aromatic-detection predicates ("in any ring" ≠ "aromatic"):**
- `has_aromatic_atom` (MACCS 162) — fires for oxirane, cyclopropane, etc.
- `has_aromatic_ring_gt1` (MACCS 125)
- `has_aromatic_c_n` (MACCS 65)
- `has_s_exo_aromatic` (MACCS 59)
- `has_o_exo_aromatic` (MACCS 113)
- `has_n_exo_aromatic` (MACCS 135)
- `has_nonaro_aro_aro_nonaro` (MACCS 144)

Marking as `unsupported` excludes them from the compiler (predictions for these keys are
ignored) and from soundness/completeness tests.

---

## Summary

| Category | Root Cause | Fix |
|---|---|---|
| Ring size / membership | Span heuristic includes branch atoms | Path-based `is_descendant` computation |
| Double bond predicates (non-ring) | Kekulé ring double bonds aromatized by RDKit; MACCS doesn't match | Restrict to `not ring_atom` — re-activated |
| Double bond predicates (ring-specific) / aromatic detection | Require aromaticity perception not possible in ASP | Marked `unsupported` |
