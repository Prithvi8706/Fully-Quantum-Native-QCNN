"""The E3 pooling arms (UPGRADE_PLAN.md 2.2, roadmap M2.1/M2.4).

Five labelled arms bracket the frozen block: `none` is the floor, `measurement`
is predicted to tie exactly (E1), `unitary` is the headline, and `coherent` and
`su4` sit in Proposition 2's strict extension. None of them is the headline
model; `coherent` in particular is an `[ARCH -- opt-in]` arm that ran only after
an explicit sign-off (roadmap 18.2, given 2026-07-25).

The load-bearing checks here are that `coherent` reduces exactly to the frozen
block when its extra angle is zero, and that both extension arms are detectably
*outside* the measurement-simulable class -- which is Proposition 3's other half,
the part E2 could not measure on the frozen block.
"""
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import pytest

from QCNN import freeze
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.layers import QuantumNativePooling
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from experiments.pooling_analysis import (
    E1_TOLERANCE,
    _hooks_dephasing,
    _readouts_with_hooks,
    readouts,
)

SMALL_IMAGE = 8          # -> 6 qubits


def _cfg(mode):
    cfg = QuantumNativeConfig.from_image_size(SMALL_IMAGE, 'amplitude')
    cfg.seed = freeze.HEADLINE_SEED
    cfg.pooling_mode = mode
    return cfg


def _inputs(n_qubits, n=2):
    rng = np.random.default_rng(freeze.REGRESSION_INPUT_SEED)
    x = rng.random((n, 2 ** n_qubits))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _params(mode):
    model = PureQuantumNativeCNN(_cfg(mode))
    return model._unflatten_params(model._flatten_params(model.quantum_params))


def _widen_to_coherent(unitary_params, n_pairs):
    """Frozen params re-laid-out for the coherent arm, with delta = 0.

    Each pair's three frozen angles are copied and a fourth, zero, is appended,
    so the extra CRY becomes the identity.
    """
    widened = {}
    for key, block in unitary_params.items():
        if not key.startswith('quantum_pooling_'):
            widened[key] = block
            continue
        flat = np.asarray(block).reshape(-1)
        out = np.zeros(4 * n_pairs)
        for i in range(n_pairs):
            out[4 * i:4 * i + 3] = flat[3 * i:3 * i + 3]
        widened[key] = pnp.array(out, requires_grad=True)
    return widened


def test_angles_per_pair_matches_the_documented_arms():
    assert QuantumNativePooling.angles_per_pair('none') == 0
    assert QuantumNativePooling.angles_per_pair('unitary') == 3
    assert QuantumNativePooling.angles_per_pair('measurement') == 3
    assert QuantumNativePooling.angles_per_pair('coherent') == 4
    assert QuantumNativePooling.angles_per_pair('su4') == 15


def test_an_unknown_arm_is_rejected_at_allocation():
    with pytest.raises(ValueError, match='Unknown pooling_mode'):
        QuantumNativePooling.angles_per_pair('bogus')


def test_the_frozen_arm_keeps_its_269_slots():
    """Arm-aware allocation must not move the headline's parameter layout."""
    cfg = QuantumNativeConfig.from_image_size(
        freeze.HEADLINE_IMAGE_SIZE, freeze.HEADLINE_ENCODING)
    cfg.seed = freeze.HEADLINE_SEED
    model = PureQuantumNativeCNN(cfg)
    assert len(model._flatten_params(model.quantum_params)) == freeze.HEADLINE_N_PARAM_SLOTS


@pytest.mark.parametrize("mode,expected", [
    ('none', 224), ('unitary', 269), ('measurement', 269),
    ('coherent', 284), ('su4', 449)])
def test_each_arm_allocates_the_slots_it_uses(mode, expected):
    cfg = QuantumNativeConfig.from_image_size(
        freeze.HEADLINE_IMAGE_SIZE, freeze.HEADLINE_ENCODING)
    cfg.seed = freeze.HEADLINE_SEED
    cfg.pooling_mode = mode
    model = PureQuantumNativeCNN(cfg)
    assert len(model._flatten_params(model.quantum_params)) == expected


@pytest.mark.slow
def test_coherent_with_zero_extra_angle_is_the_frozen_block():
    """The arm must be a strict superset, not a different block.

    If this fails, any accuracy difference E3 reports for `coherent` could be
    an artefact of the arm being built differently rather than of the extra gate.
    """
    cfg_u, cfg_c = _cfg('unitary'), _cfg('coherent')
    n_pairs = cfg_u.n_qubits // 2
    unitary_params = _params('unitary')
    coherent_params = _widen_to_coherent(unitary_params, n_pairs)
    x = _inputs(cfg_u.n_qubits)

    frozen = readouts(cfg_u, unitary_params, x)
    reduced = readouts(cfg_c, coherent_params, x)

    np.testing.assert_allclose(reduced, frozen, atol=E1_TOLERANCE, rtol=0.0)


@pytest.mark.slow
def test_coherent_with_a_nonzero_extra_angle_departs_from_the_frozen_block():
    cfg_u, cfg_c = _cfg('unitary'), _cfg('coherent')
    n_pairs = cfg_u.n_qubits // 2
    unitary_params = _params('unitary')
    coherent_params = _widen_to_coherent(unitary_params, n_pairs)
    for key in coherent_params:
        if key.startswith('quantum_pooling_'):
            block = np.asarray(coherent_params[key]).copy()
            block[3::4] = 0.9
            coherent_params[key] = pnp.array(block, requires_grad=True)
    x = _inputs(cfg_u.n_qubits)

    assert np.abs(readouts(cfg_c, coherent_params, x)
                  - readouts(cfg_u, unitary_params, x)).max() > 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("mode", ['coherent', 'su4'])
def test_the_extension_arms_are_outside_the_measurement_simulable_class(mode):
    """Proposition 3's other half.

    E2 showed Delta_coh == 0 for the frozen block. Proposition 3 also predicts
    Delta_coh > 0 for non-diagonal V. These arms are non-diagonal in the
    compressed qubit's basis, so dephasing that qubit before pooling *must* be
    detectable -- otherwise "strict extension" would be a claim with no
    observable content.
    """
    cfg = _cfg(mode)
    params = _params(mode)
    x = _inputs(cfg.n_qubits)

    clean = _readouts_with_hooks(cfg, params, x, None)
    dephased = _readouts_with_hooks(cfg, params, x, _hooks_dephasing('discard'))

    assert np.abs(clean - dephased).max() > 1e-6, (
        '{} is indistinguishable from its dephased form, so it does not '
        'actually leave the measurement-simulable class'.format(mode))


@pytest.mark.slow
def test_su4_emits_one_general_two_qubit_unitary_per_pair():
    cfg = _cfg('su4')
    model = PureQuantumNativeCNN(cfg)
    flat = model._flatten_params(model.quantum_params)
    tape = freeze.qnode_tape(model.quantum_circuit, _inputs(cfg.n_qubits)[0], flat)

    arbitrary = [op for op in tape.operations if op.name == 'ArbitraryUnitary']
    assert arbitrary, 'su4 arm emitted no ArbitraryUnitary'
    for op in arbitrary:
        assert len(op.wires) == 2
        assert np.asarray(op.data[0]).size == 15


@pytest.mark.slow
def test_pool_none_emits_no_pooling_gate():
    cfg = _cfg('none')
    model = PureQuantumNativeCNN(cfg)
    flat = model._flatten_params(model.quantum_params)
    tape = freeze.qnode_tape(model.quantum_circuit, _inputs(cfg.n_qubits)[0], flat)

    assert not any(op.name in ('CRY', 'CRZ', 'ArbitraryUnitary') for op in tape.operations)
