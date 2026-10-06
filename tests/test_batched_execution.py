"""Batched execution must be behaviourally identical to the sequential path
(UPGRADE_PLAN.md 1.1, roadmap M1.1).

The per-sample loop is the correctness oracle. The batched path changes only the
execution strategy -- device and differentiation method -- never the circuit,
which both paths obtain from ``QCNN/circuits.py``. Roadmap M1.1 requires
agreement on raw outputs, loss, *every* parameter gradient, and one optimizer
update before the batched path may carry a single result.
"""
import json

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest

from QCNN import freeze
from QCNN.training.Qtrainer import QuantumNativeTrainer

# Eight samples exercises broadcasting without making the sequential oracle
# (~1.3 s/sample) dominate the suite's runtime.
BATCH = 8

# Sequential runs adjoint differentiation on lightning.qubit; batched runs
# backprop on default.qubit. Both are exact methods in float64, so they may
# differ only by accumulated round-off.
GRADIENT_TOL = 1e-10


@pytest.fixture(scope="module")
def batch_inputs():
    return freeze.fixed_regression_inputs()[:BATCH]


@pytest.fixture(scope="module")
def batch_labels():
    return np.array([1.0, -1.0] * (BATCH // 2))


@pytest.fixture(scope="module")
def trainer():
    t = QuantumNativeTrainer()
    t._pos_weight = 1.0
    t._neg_weight = 1.0
    return t


def _sequential_preds(model, X, params):
    """The frozen per-sample path, exactly as Qtrainer drives it today."""
    return pnp.array([model.quantum_circuit(pnp.array(x), params) for x in X])


def _batched_preds(model, X, params):
    return model.batched_circuit(pnp.array(X), params)


@pytest.mark.slow
def test_batched_outputs_match_the_sequential_oracle(
        headline_model, archived_params, batch_inputs):
    sequential = np.array(
        [float(headline_model.quantum_circuit(np.asarray(x), archived_params))
         for x in batch_inputs])
    batched = np.asarray(
        headline_model.batched_circuit(np.asarray(batch_inputs), archived_params),
        dtype=float)

    assert batched.shape == sequential.shape
    np.testing.assert_allclose(
        batched, sequential, atol=freeze.REGRESSION_TOL, rtol=0.0)


@pytest.mark.slow
def test_batched_loss_matches_the_sequential_oracle(
        headline_model, archived_params, batch_inputs, batch_labels, trainer):
    sequential = trainer._bce_loss(
        _sequential_preds(headline_model, batch_inputs, archived_params), batch_labels)
    batched = trainer._bce_loss(
        _batched_preds(headline_model, batch_inputs, archived_params), batch_labels)

    np.testing.assert_allclose(
        float(batched), float(sequential), atol=freeze.REGRESSION_TOL, rtol=0.0)


@pytest.mark.slow
def test_every_parameter_gradient_matches_the_sequential_oracle(
        headline_model, archived_params, batch_inputs, batch_labels, trainer):
    """Every one of the 269 allocated slots, not just the 74 effective ones."""
    def sequential_cost(p):
        return trainer._bce_loss(
            _sequential_preds(headline_model, batch_inputs, p), batch_labels)

    def batched_cost(p):
        return trainer._bce_loss(
            _batched_preds(headline_model, batch_inputs, p), batch_labels)

    g_sequential = np.asarray(qml.grad(sequential_cost)(archived_params), dtype=float)
    g_batched = np.asarray(qml.grad(batched_cost)(archived_params), dtype=float)

    assert g_sequential.shape == (freeze.HEADLINE_N_PARAM_SLOTS,)
    assert g_batched.shape == (freeze.HEADLINE_N_PARAM_SLOTS,)
    np.testing.assert_allclose(g_batched, g_sequential, atol=GRADIENT_TOL, rtol=0.0)


@pytest.mark.slow
def test_one_optimizer_update_matches_the_sequential_oracle(
        headline_model, archived_params, batch_inputs, batch_labels, trainer):
    """Agreement must survive the optimizer, not just the gradient."""
    def sequential_cost(p):
        return trainer._bce_loss(
            _sequential_preds(headline_model, batch_inputs, p), batch_labels)

    def batched_cost(p):
        return trainer._bce_loss(
            _batched_preds(headline_model, batch_inputs, p), batch_labels)

    stepped_sequential = qml.AdamOptimizer(stepsize=0.02).step(
        sequential_cost, archived_params.copy())
    stepped_batched = qml.AdamOptimizer(stepsize=0.02).step(
        batched_cost, archived_params.copy())

    np.testing.assert_allclose(
        np.asarray(stepped_batched, dtype=float),
        np.asarray(stepped_sequential, dtype=float),
        atol=GRADIENT_TOL, rtol=0.0)


def test_batched_path_emits_the_frozen_topology(headline_model):
    """The batched path must be the same circuit, not a re-implementation.

    Guards the M0.6 single-source-of-truth property across the new execution
    path: same committed signature hash, so a batched-only structural edit
    fails here rather than silently producing different physics.
    """
    with open(freeze.SIGNATURE_FIXTURE) as fh:
        reference = json.load(fh)

    n_slots = len(headline_model._flatten_params(headline_model.quantum_params))
    marker = pnp.array(np.arange(1, n_slots + 1, dtype=float), requires_grad=True)
    tape = freeze.qnode_tape(
        headline_model.batched_circuit, freeze.fixed_regression_inputs()[0], marker)

    assert freeze.signature_hash(freeze.tape_signature(tape, n_slots)) == reference["hash"]


def test_batched_path_stays_unitary(headline_model):
    """A3 holds on the new path too: no mid-circuit measurement, no channel."""
    n_slots = len(headline_model._flatten_params(headline_model.quantum_params))
    marker = pnp.array(np.arange(1, n_slots + 1, dtype=float), requires_grad=True)
    tape = freeze.qnode_tape(
        headline_model.batched_circuit, freeze.fixed_regression_inputs()[0], marker)

    assert freeze.unitarity_violations(tape) == []


@pytest.mark.slow
def test_final_test_evaluation_matches_the_sequential_oracle(
        headline_model, archived_params, batch_inputs):
    """The single final test evaluation is now batched (UPGRADE_PLAN.md 1.1).

    It is off the training hot path, but at n=10 the per-sample path costs
    ~1.3 s/sample -- about 68 minutes on a full test split. It may only be
    ported if it changes no number.
    """
    from QCNN.utils.metrics import predict_raw_outputs

    headline_model.quantum_params = headline_model._unflatten_params(archived_params)
    sequential = np.array(
        [float(headline_model.quantum_circuit(np.asarray(x), archived_params))
         for x in batch_inputs])
    batched = predict_raw_outputs(headline_model, np.asarray(batch_inputs),
                                  already_preprocessed=True)

    assert batched.shape == sequential.shape
    np.testing.assert_allclose(batched, sequential, atol=freeze.REGRESSION_TOL, rtol=0.0)
