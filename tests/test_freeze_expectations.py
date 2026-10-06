"""Semantic guard: the frozen function (x, theta) -> <Z> must not drift.

This is the test that makes refactoring safe. Any change to how the circuit is
built, dispatched, or executed must leave these 20 numbers unchanged to 1e-10
(UPGRADE_PLAN.md A2).
"""
import numpy as np
import pytest

from QCNN import freeze


@pytest.mark.slow
def test_expectations_match_archived_reference(headline_model, archived_params, regression_inputs):
    reference = np.load(freeze.EXPECTATION_FIXTURE)

    np.testing.assert_array_equal(
        reference["inputs"], regression_inputs,
        err_msg="The committed regression inputs changed; the fixture is invalid.",
    )

    actual = freeze.headline_expectations(headline_model, archived_params, regression_inputs)

    np.testing.assert_allclose(
        actual, reference["expectations"], atol=freeze.REGRESSION_TOL, rtol=0.0,
        err_msg=(
            "The frozen model's outputs drifted beyond 1e-10. Under UPGRADE_PLAN.md "
            "A2 the refactor that caused this is not behaviour-preserving."
        ),
    )


@pytest.mark.slow
def test_expectations_are_deterministic(headline_model, archived_params, regression_inputs):
    first = freeze.headline_expectations(headline_model, archived_params, regression_inputs[:3])
    second = freeze.headline_expectations(headline_model, archived_params, regression_inputs[:3])
    np.testing.assert_array_equal(first, second)


def test_expectations_are_valid_pauli_z_values(headline_model, archived_params):
    reference = np.load(freeze.EXPECTATION_FIXTURE)
    assert reference["expectations"].shape == (freeze.N_REGRESSION_INPUTS,)
    assert np.all(np.abs(reference["expectations"]) <= 1.0 + 1e-12)
