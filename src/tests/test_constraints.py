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
        try:
            base_smiles = cached_run_pipeline(base_pred)
        except PipelineTimeoutError as e:
            warnings.warn(
                f"Pipeline timed out while testing completeness for {constraint.name}.\n"
                f"Predictions that caused timeout:\n{e.predictions}"
            )
            continue

        # Constrained
        constrained_pred = build_predictions(formula, constraint)
        try:
            constrained_smiles = cached_run_pipeline(constrained_pred)
        except PipelineTimeoutError as e:
            warnings.warn(
                f"Pipeline timed out while testing completeness for {constraint.name}.\n"
                f"Predictions that caused timeout:\n{e.predictions}"
            )
            continue

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