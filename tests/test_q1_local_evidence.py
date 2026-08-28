"""Pure-Python contracts for the bounded local-evidence adapter.

Backend execution is deliberately exercised with the pinned ``.venv-qiskit``
command, not during the Python 3.9 training-environment suite.  These tests
cover the invariants that must hold before that bounded command is launched.
"""

import hashlib

import numpy as np
import pytest

from experiments import q1_local_evidence as evidence


def test_local_evidence_scope_and_resource_axes_are_frozen():
    assert evidence.RESOURCE_QUBITS == (4, 6, 8, 10)
    assert evidence.POOLING_ARMS == ("none", "measurement", "unitary", "coherent", "su4")
    assert evidence.DEFAULT_SEED_TRANSPILER == 42
    assert evidence.DEFAULT_OPTIMIZATION_LEVEL == 1
    assert "ecr" in evidence.TARGET_BASIS
    assert "QPU" in evidence.build_rehearsal.__doc__


@pytest.mark.parametrize("task,expected", [
    ("mnist:3,5", ("mnist", (3, 5))),
    ("fashion_mnist:0,6", ("fashion_mnist", (0, 6))),
    ("kmnist:2,3", ("kmnist", (2, 3))),
])
def test_task_parser_is_strict_and_source_identifying(task, expected):
    assert evidence._parse_task(task) == expected


@pytest.mark.parametrize("task", ["mnist", "mnist:0", "mnist:0,0", "unknown:0,1"])
def test_task_parser_rejects_ambiguous_or_invalid_tasks(task):
    with pytest.raises(ValueError):
        evidence._parse_task(task)


def test_amplitude_preprocessing_is_padded_normalized_and_deterministic():
    images = np.zeros((2, 28 * 28), dtype=np.uint8)
    images[0, 0] = 255
    images[1, 27] = 128
    amplitudes = evidence._amplitudes_from_pixels(images, n_qubits=10)
    assert amplitudes.shape == (2, 1024)
    np.testing.assert_allclose(np.linalg.norm(amplitudes, axis=1), 1.0)
    # Bit reversal can move a source pixel into the padded tail; support, not
    # the post-padding slice, is the invariant that matters here.
    assert np.count_nonzero(amplitudes[0]) == 1
    assert np.count_nonzero(amplitudes[1]) == 1
    np.testing.assert_array_equal(amplitudes, evidence._amplitudes_from_pixels(images, n_qubits=10))


def test_amplitude_preprocessing_uses_qiskit_bit_reversal_for_canonical_wires():
    images = np.zeros((1, 28 * 28), dtype=np.uint8)
    images[0, 1] = 255
    amplitudes = evidence._amplitudes_from_pixels(images, n_qubits=10)
    # Source index 1 (binary ...0001) becomes Qiskit index 512 (...1000000000).
    assert amplitudes[0, 512] == 1.0
    assert np.count_nonzero(amplitudes) == 1


def test_hash_is_canonical_and_does_not_allow_nonfinite_values():
    payload = {"b": [2, 1], "a": {"value": 3}}
    expected = hashlib.sha256(b'{"a":{"value":3},"b":[2,1]}').hexdigest()
    assert evidence._json_hash(payload) == expected
    with pytest.raises(ValueError):
        evidence._json_hash({"value": float("nan")})


def test_qiskit_circuit_schedule_separates_state_and_model_components():
    state = evidence._build_state_preparation(4)
    body = evidence._build_model_body(4, "unitary", seed=0)
    assert state.count_ops().get("initialize") == 1
    assert body.count_ops().get("initialize", 0) == 0
    assert body.count_ops().get("cry") == 3
    assert body.count_ops().get("crz") == 3
    assert body.count_ops().get("cx") > 0


def test_measurement_style_schedule_is_explicitly_dynamic():
    body = evidence._build_model_body(4, "measurement", seed=0)
    counts = {str(name): int(value) for name, value in body.count_ops().items()}
    assert counts.get("measure") == 3
    assert counts.get("if_else") == 3
    assert "if_else" not in {"rx", "ry", "rz", "cx"}


def test_noise_ladder_requires_zero_noise_anchor():
    with pytest.raises(ValueError, match="begin"):
        evidence.build_noise_validation(
            ["mnist:3,5"], [0], samples=1, levels=(0.01,), data_root="missing"
        )
