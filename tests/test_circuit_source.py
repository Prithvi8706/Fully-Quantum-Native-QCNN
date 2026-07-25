"""One circuit definition, consumed by every path (UPGRADE_PLAN.md 0.4, F7).

noise_sim.py previously re-implemented the forward pass by hand, so the noise
study could silently drift away from the model it claimed to measure.
"""
import inspect
import json

import numpy as np
import pytest

from QCNN import circuits, freeze


def test_model_delegates_to_the_shared_builder():
    from QCNN.models.QCNNModel import PureQuantumNativeCNN
    source = inspect.getsource(PureQuantumNativeCNN._pure_quantum_forward)
    assert "circuits.build_circuit" in source or "build_circuit(" in source


def test_noise_sim_has_no_duplicate_topology():
    with open("noise_sim.py", encoding="utf-8") as fh:
        source = fh.read()
    assert "build_circuit" in source, "noise_sim.py must consume QCNN/circuits.py"
    for copied in ("get_conv_windows", "make_pairing", "quantum_unitary_pooling"):
        assert copied not in source, (
            "noise_sim.py still rebuilds the topology by hand ({})".format(copied))


def test_hooks_are_off_by_default():
    hooks = circuits.CircuitHooks()
    assert hooks.after_encoding is None
    assert hooks.after_conv_window is None
    assert hooks.after_pool is None
    assert hooks.after_classifier is None
    assert hooks.before_readout is None


def test_signature_survives_the_consolidation(headline_model):
    with open(freeze.SIGNATURE_FIXTURE) as fh:
        reference = json.load(fh)
    assert freeze.signature_hash(freeze.circuit_signature(headline_model)) == reference["hash"]


@pytest.mark.slow
def test_expectations_survive_the_consolidation(headline_model, archived_params, regression_inputs):
    reference = np.load(freeze.EXPECTATION_FIXTURE)
    actual = freeze.headline_expectations(headline_model, archived_params, regression_inputs)
    np.testing.assert_allclose(
        actual, reference["expectations"], atol=freeze.REGRESSION_TOL, rtol=0.0)
