"""Pure-Python contracts for the bounded local-evidence adapter.

Backend execution is deliberately exercised with the pinned ``.venv-qiskit``
command, not during the Python 3.9 training-environment suite.  These tests
cover the invariants that must hold before that bounded command is launched.
"""

import hashlib
import json

import numpy as np
import pytest

from experiments import q1_local_evidence as evidence

try:
    import qiskit  # noqa: F401
except ImportError:
    HAS_QISKIT = False
else:
    HAS_QISKIT = True


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


@pytest.mark.skipif(
    not HAS_QISKIT,
    reason="Qiskit schedule contracts run in the isolated .venv-qiskit environment",
)
def test_qiskit_circuit_schedule_separates_state_and_model_components():
    state = evidence._build_state_preparation(4)
    body = evidence._build_model_body(4, "unitary", seed=0)
    assert state.count_ops().get("initialize") == 1
    assert body.count_ops().get("initialize", 0) == 0
    assert body.count_ops().get("cry") == 3
    assert body.count_ops().get("crz") == 3
    assert body.count_ops().get("cx") > 0


@pytest.mark.skipif(
    not HAS_QISKIT,
    reason="Qiskit schedule contracts run in the isolated .venv-qiskit environment",
)
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


def _checkpoint_status(split_id="split-1", seed=0, **config_overrides):
    config = {
        "evidence_role": "scientific",
        "image_size": 28,
        "n_qubits": 10,
        "encoding_type": "amplitude",
        "pooling_mode": "unitary",
        "ablation": {"name": "proposed"},
    }
    config.update(config_overrides)
    return {"state": "complete", "seed": seed, "split_id": split_id, "config": config}


def test_noise_checkpoint_validation_rejects_missing_or_fixture_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="requires a complete proposed checkpoint"):
        evidence._validate_checkpoint_identity(
            None, "mnist", (3, 5), 0, "split-1"
        )

    checkpoint = tmp_path / "weights.npz"
    np.savez(checkpoint, weights=np.zeros(1))
    with pytest.raises(ValueError, match="provenance is missing"):
        evidence._validate_checkpoint_identity(
            checkpoint, "mnist", (3, 5), 0, "split-1"
        )


@pytest.mark.parametrize(
    "status_kwargs,match",
    [
        ({"split_id": "other"}, "split does not match"),
        ({"pooling_mode": "none"}, "configuration does not match"),
    ],
)
def test_noise_checkpoint_validation_rejects_identity_mismatch(
        tmp_path, status_kwargs, match):
    checkpoint = tmp_path / "weights.npz"
    np.savez(checkpoint, weights=np.zeros(1))
    status = _checkpoint_status(**status_kwargs)
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        evidence._validate_checkpoint_identity(
            checkpoint, "mnist", (3, 5), 0, "split-1"
        )
