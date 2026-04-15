import pytest
from constraint_defs import constraints

# -------------------------
# Coverage test
# -------------------------

@pytest.mark.parametrize("constraint", constraints)
def test_formula_coverage_verbose(
    constraint, formulas, cached_run_pipeline, smiles_to_mol, build_predictions
):
    coverage = {"pos": set(), "neg": set()}

    for formula in formulas:
        predictions = build_predictions(formula)

        for smi in cached_run_pipeline(predictions):
            mol = smiles_to_mol(smi)

            if constraint.rdkit_check(mol):
                coverage["pos"].add(formula)
            else:
                coverage["neg"].add(formula)

    # Assert coverage
    assert coverage["pos"], \
        f"[COVERAGE] {constraint.name}: no positive examples in formulas {formulas}"

    assert coverage["neg"], \
        f"[COVERAGE] {constraint.name}: no negative examples in formulas {formulas}"


# -------------------------
# Soundness
# -------------------------

@pytest.mark.parametrize("constraint", constraints)
def test_constraint_soundness(
    constraint, formulas, cached_run_pipeline, smiles_to_mol, build_predictions
):
    for formula in formulas:
        predictions = build_predictions(formula, constraint)

        smiles_list = cached_run_pipeline(predictions)

        for smi in smiles_list:
            mol = smiles_to_mol(smi)
            assert constraint.rdkit_check(mol), \
                f"{constraint.name} violated by {smi}"


# -------------------------
# Completeness
# -------------------------

@pytest.mark.parametrize("constraint", constraints)
def test_constraint_completeness(
    constraint, formulas, cached_run_pipeline, smiles_to_mol, build_predictions
):
    for formula in formulas:
        # Base
        base_pred = build_predictions(formula, constraint=None)
        base_smiles = cached_run_pipeline(base_pred)

        # Constrained
        constrained_pred = build_predictions(formula, constraint)
        constrained_smiles = cached_run_pipeline(constrained_pred)

        # Expected subset
        expected = []
        for smi in base_smiles:
            mol = smiles_to_mol(smi)
            if constraint.rdkit_check(mol):
                expected.append(smi)

        expected = sorted(expected)
        constrained_smiles = sorted(constrained_smiles)

        assert constrained_smiles == expected, (
            f"{constraint.name} does not correctly filter for {formula}.\n"
            f"Expected: {expected}\n"
            f"Got: {constrained_smiles}"
        )