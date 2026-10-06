"""Precomputed amplitude inputs must not change what the circuit sees
(UPGRADE_PLAN.md 1.2, roadmap M1.2).

Amplitude vectors are parameter-independent, so padding and L2 normalisation are
hoisted out of the training loop. The cache is only legitimate if the circuit
cannot tell the difference, and if the hoisting actually happens once per split
rather than once per batch.
"""
import numpy as np
import pytest

from QCNN import freeze
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.encoding import PureQuantumEncoder
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training import Qtrainer as qtrainer_module
from QCNN.training.Qtrainer import QuantumNativeTrainer

N_QUBITS = freeze.HEADLINE_N_QUBITS
ENCODED_LEN = 2 ** N_QUBITS


@pytest.fixture(scope="module")
def raw_inputs():
    """Unnormalised, unpadded inputs in the archived dataset's feature length."""
    return np.random.default_rng(20260725).random((6, 784))


def test_precompute_pads_and_normalises(raw_inputs):
    encoded = PureQuantumEncoder.precompute_amplitudes(raw_inputs, N_QUBITS)

    assert encoded.shape == (len(raw_inputs), ENCODED_LEN)
    np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(encoded[:, 784:], 0.0, atol=0.0)


def test_precompute_is_idempotent(raw_inputs):
    """Re-encoding an encoded array is a no-op: no redundant normalisation."""
    once = PureQuantumEncoder.precompute_amplitudes(raw_inputs, N_QUBITS)
    twice = PureQuantumEncoder.precompute_amplitudes(once, N_QUBITS)

    np.testing.assert_allclose(twice, once, atol=1e-15, rtol=0.0)


def test_precompute_matches_single_and_batched(raw_inputs):
    batched = PureQuantumEncoder.precompute_amplitudes(raw_inputs, N_QUBITS)
    for i, row in enumerate(raw_inputs):
        single = PureQuantumEncoder.precompute_amplitudes(row, N_QUBITS)
        assert single.shape == (ENCODED_LEN,)
        np.testing.assert_allclose(single, batched[i], atol=1e-15, rtol=0.0)


def test_zero_vector_falls_back_to_a_basis_state():
    """A zero row is not a valid state; it must become |0...0>, not NaN."""
    x = np.zeros((2, 784))
    x[1, 3] = 2.0
    encoded = PureQuantumEncoder.precompute_amplitudes(x, N_QUBITS)

    assert not np.isnan(encoded).any()
    assert encoded[0, 0] == 1.0
    np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), 1.0, atol=1e-15)


@pytest.mark.slow
def test_cached_and_uncached_outputs_agree(headline_model, archived_params, raw_inputs):
    """The M1.2 exit check: the circuit cannot tell the cache from raw input."""
    uncached = np.asarray(
        headline_model.batched_circuit(
            np.asarray(headline_model._preprocess_input(raw_inputs)), archived_params),
        dtype=float)
    cached = np.asarray(
        headline_model.batched_circuit(
            PureQuantumEncoder.precompute_amplitudes(raw_inputs, N_QUBITS), archived_params),
        dtype=float)

    np.testing.assert_allclose(cached, uncached, atol=freeze.REGRESSION_TOL, rtol=0.0)


def test_identity_is_stable_and_content_sensitive(raw_inputs):
    encoded = PureQuantumEncoder.precompute_amplitudes(raw_inputs, N_QUBITS)
    identity = PureQuantumEncoder.amplitude_identity(encoded)

    assert identity == PureQuantumEncoder.amplitude_identity(encoded)
    assert identity['shape'] == [len(raw_inputs), ENCODED_LEN]

    perturbed = encoded.copy()
    perturbed[0, 0] += 1e-12
    assert PureQuantumEncoder.amplitude_identity(perturbed)['sha256'] != identity['sha256']


@pytest.mark.slow
def test_training_encodes_once_per_split_not_once_per_batch(monkeypatch):
    """Repeated epochs and batches must not re-normalise the inputs."""
    cfg = QuantumNativeConfig.from_image_size(8, 'amplitude')
    cfg.seed = 0
    cfg.n_epochs = 2
    cfg.batch_size = 4
    model = PureQuantumNativeCNN(cfg)

    rng = np.random.default_rng(0)
    X_train, y_train = rng.random((8, 64)), np.array([1.0, -1.0] * 4)
    X_val, y_val = rng.random((4, 64)), np.array([1.0, -1.0] * 2)

    calls = []
    original = PureQuantumEncoder.precompute_amplitudes

    def counting(data, n_qubits):
        calls.append(np.shape(data))
        return original(data, n_qubits)

    monkeypatch.setattr(
        qtrainer_module.PureQuantumEncoder, 'precompute_amplitudes', staticmethod(counting))

    trainer = QuantumNativeTrainer()
    trainer.train_pure_quantum_cnn(
        model, X_train, y_train, X_val, y_val,
        log_filepath=None, summary_filepath=None, weights_path=None)

    # Exactly two: one per split. Four batches x two epochs must add nothing.
    assert calls == [(8, 64), (4, 64)], calls
    assert set(trainer.encoded_cache_identity) == {'train', 'val'}
    assert trainer.encoded_cache_identity['train']['shape'] == [8, 2 ** cfg.n_qubits]
