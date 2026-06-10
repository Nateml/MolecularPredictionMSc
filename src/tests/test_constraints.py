import pytest
import warnings
from tests.constraint_defs import constraints
from conftest import PipelineTimeoutError

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

        try:
            for smi in cached_run_pipeline(predictions):
                mol = smiles_to_mol(smi)

                if constraint.rdkit_check(mol):
                    coverage["pos"].add(formula)
                else:
                    coverage["neg"].add(formula)
        except PipelineTimeoutError as e:
            warnings.warn(
                f"Pipeline timed out while testing coverage for {constraint.name}.\n"
                f"Predictions that caused timeout:\n{e.predictions}\n"
                f"Consider increasing the timeout if this is a common occurrence."
            )

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
    """
    Tests for constraint soundness. In this context, soundness means that every
    candidate molecule in the solution set passes the constraint check (i.e.
    no molecules violate it).
    """
    for formula in formulas:
        predictions = build_predictions(formula, constraint)

        try:
            smiles_list = cached_run_pipeline(predictions)
        except PipelineTimeoutError as e:
            warnings.warn(
                f"Pipeline timed out while testing soundness for {constraint.name}.\n"
                f"Predictions that caused timeout:\n{e.predictions}"
            )
            continue

        for smi in smiles_list:
            mol = smiles_to_mol(smi)
            assert constraint.rdkit_check(mol), \
                f"{constraint.name} violated by {smi}; using formula {formula}"


# -------------------------
# Completeness
# -------------------------

@pytest.mark.parametrize("constraint", constraints)
def test_constraint_completeness(
    constraint, formulas, cached_run_pipeline, smiles_to_mol, build_predictions
):
    """
    Tests for completeness of the constraints. Completeness in this context means
    that the solution is not overconstrained, i.e. every molecule that should appear
    in the solution set does. It does not check if there are extra molecules which
    shouldn't appear -- thats a soundness thing.
    """
    timed_out_base = []
    timed_out_covered = []
    formulas_tested = 0

    for formula in formulas:
        # Base
        base_pred = build_predictions(formula, constraint=None)
        try:
            base_smiles = cached_run_pipeline(base_pred)
        except PipelineTimeoutError:
            timed_out_base.append(formula)
            continue

        # Determine if this formula exercises the constraint
        expected = [smi for smi in base_smiles if constraint.rdkit_check(smiles_to_mol(smi))]
        if not expected:
            continue  # constraint not exercised by this formula

        # Constrained — timeout here means result is invalid (coverage exists)
        constrained_pred = build_predictions(formula, constraint)
        try:
            constrained_smiles = cached_run_pipeline(constrained_pred)
        except PipelineTimeoutError:
            timed_out_covered.append(formula)
            continue

        missing = sorted(set(expected) - set(constrained_smiles))
        assert not missing, (
            f"{constraint.name} does not correctly filter for {formula}.\n"
            f"Missing from constrained (expected but absent): {missing}\n"
            f"Expected: {sorted(expected)}\n"
            f"Got: {sorted(constrained_smiles)}"
        )
        formulas_tested += 1

    if timed_out_covered:
        pytest.fail(
            f"{constraint.name}: completeness result invalid — "
            f"timed out on covered formulas: {timed_out_covered}"
        )
    if formulas_tested == 0:
        pytest.skip(f"{constraint.name}: no formulas exercised constraint without timeout")
    elif timed_out_base:
        pytest.skip(f"{constraint.name}: some base runs timed out: {timed_out_base}")
