"""E1: unitary pooling exactly reproduces measure-and-condition pooling.

Theorem 1 (UPGRADE_PLAN.md 2.1) predicts an exact tie, so this is not a
measurement that could come out either way -- a miss is an implementation or
theorem-mapping defect (roadmap M2.2), never a result to report.

These tests are the standing version of that check. They run at n=6 so the
density-matrix simulation stays cheap; `experiments/pooling_analysis.py` runs it
at the headline n=10 with the archived weights and writes the F-A evidence.
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
    _trace_distance,
    readouts,
    run_e4,
)

SMALL_IMAGE = 8          # -> 6 qubits, a 64x64 density matrix
ANGLES = pnp.array([0.31, -0.72, 1.14, 0.05, 0.88, -1.31])


def _small(pooling_mode):
    cfg = QuantumNativeConfig.from_image_size(SMALL_IMAGE, 'amplitude')
    cfg.seed = freeze.HEADLINE_SEED
    cfg.pooling_mode = pooling_mode
    return cfg


@pytest.fixture(scope='module')
def small_setup():
    cfg = _small('unitary')
    model = PureQuantumNativeCNN(cfg)
    params = model._unflatten_params(model._flatten_params(model.quantum_params))
    rng = np.random.default_rng(freeze.REGRESSION_INPUT_SEED)
    x = rng.random((2, 2 ** cfg.n_qubits))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return params, x


def test_kraus_operators_are_a_valid_channel():
    """sum_m K_m^dag K_m = I, or the 'channel' is not a channel."""
    for idx in range(2):
        kraus = QuantumNativePooling.measurement_channel_kraus(ANGLES, idx)
        total = sum(k.conj().T @ k for k in kraus)
        np.testing.assert_allclose(total, np.eye(4), atol=1e-14)


def test_kraus_branches_are_the_theorem_s_two_unitaries():
    """K_0 carries U_0 = RY(gamma); K_1 carries U_1 = RY(gamma) RZ(beta) RY(alpha)."""
    alpha, beta, gamma = (float(v) for v in QuantumNativePooling.pair_angles(ANGLES, 0))
    k0, k1 = QuantumNativePooling.measurement_channel_kraus(ANGLES, 0)

    ry = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)],
                             [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)
    rz = lambda t: np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)

    u0 = ry(gamma)
    u1 = ry(gamma) @ rz(beta) @ ry(alpha)
    np.testing.assert_allclose(k0, np.kron(u0, np.diag([1, 0])), atol=1e-14)
    np.testing.assert_allclose(k1, np.kron(u1, np.diag([0, 1])), atol=1e-14)


def test_every_pooling_angle_reaches_the_measurement_arm():
    """Direct guard on B1.

    The pre-2026-07-25 arm applied RY(alpha) on outcome 1 and nothing on outcome
    0, so beta and gamma were dead. If either stops mattering again, the channel
    stops being the one Theorem 1 maps onto and E1 silently becomes wrong.
    """
    base = QuantumNativePooling.measurement_channel_kraus(ANGLES, 0)
    for slot in range(3):
        perturbed = pnp.array(ANGLES).copy()
        perturbed[slot] = perturbed[slot] + 0.4
        moved = QuantumNativePooling.measurement_channel_kraus(perturbed, 0)
        delta = max(np.abs(a - b).max() for a, b in zip(base, moved))
        assert delta > 1e-3, 'angle slot {} does not affect the channel'.format(slot)


@pytest.mark.slow
def test_e1_unitary_and_measurement_pooling_agree_exactly(small_setup):
    """The E1 claim itself, at reduced size."""
    params, inputs = small_setup

    unitary = readouts(_small('unitary'), params, inputs)
    channel = readouts(_small('measurement_channel'), params, inputs)

    np.testing.assert_allclose(channel, unitary, atol=E1_TOLERANCE, rtol=0.0)


@pytest.mark.slow
def test_the_mid_circuit_arm_matches_the_channel_arm(small_setup):
    """The trainable arm and the E1 arm must implement the same map.

    'measurement' uses qml.measure/qml.cond and runs on pure-state devices for
    training; 'measurement_channel' is the CPTP form used for E1. If they drift
    apart, E3 would train a different arm than E1 validated.
    """
    params, inputs = small_setup

    channel = readouts(_small('measurement_channel'), params, inputs)
    mid_circuit = readouts(_small('measurement'), params, inputs)

    np.testing.assert_allclose(mid_circuit, channel, atol=E1_TOLERANCE, rtol=0.0)


@pytest.mark.slow
def test_pool_none_actually_differs():
    """A sanity control: the tie is a property of the map, not of the harness.

    If every arm agreed, the E1 tie would prove nothing about Theorem 1 -- it
    would just mean the readout is insensitive to pooling.
    """
    cfg = _small('unitary')
    model = PureQuantumNativeCNN(cfg)
    params = model._unflatten_params(model._flatten_params(model.quantum_params))
    rng = np.random.default_rng(freeze.REGRESSION_INPUT_SEED)
    x = rng.random((2, 2 ** cfg.n_qubits))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    unitary = readouts(_small('unitary'), params, x)
    none = readouts(_small('none'), params, x)

    assert np.abs(unitary - none).max() > 1e-6


def test_dephasing_really_dephases():
    """PhaseFlip(0.5) must send rho -> diag(rho), or E2 measures nothing."""
    dev = qml.device('default.mixed', wires=1)

    @qml.qnode(dev)
    def circuit(dephase):
        qml.Hadamard(wires=0)          # |+>, maximal off-diagonal coherence
        if dephase:
            qml.PhaseFlip(0.5, wires=0)
        return qml.density_matrix(wires=0)

    coherent = np.asarray(circuit(False))
    dephased = np.asarray(circuit(True))

    assert abs(coherent[0, 1]) == pytest.approx(0.5, abs=1e-12)
    assert abs(dephased[0, 1]) < 1e-14
    np.testing.assert_allclose(np.diag(dephased), np.diag(coherent), atol=1e-14)


@pytest.mark.slow
def test_e2_dephasing_discarded_wires_changes_nothing(small_setup):
    """Proposition 3 at reduced size: Delta_coh == 0 for the frozen block."""
    params, inputs = small_setup
    cfg = _small('unitary')

    clean = _readouts_with_hooks(cfg, params, inputs, None)
    dephased = _readouts_with_hooks(cfg, params, inputs, _hooks_dephasing('discard'))

    np.testing.assert_allclose(dephased, clean, atol=E1_TOLERANCE, rtol=0.0)
    assert np.all(np.sign(clean) == np.sign(dephased))


@pytest.mark.slow
def test_e2_control_dephasing_kept_wires_does_change_things(small_setup):
    """Without this the E2 null result could just mean the channel never fired."""
    params, inputs = small_setup
    cfg = _small('unitary')

    clean = _readouts_with_hooks(cfg, params, inputs, None)
    control = _readouts_with_hooks(cfg, params, inputs, _hooks_dephasing('keep'))

    assert np.abs(clean - control).max() > 1e-6


def test_swap_witnesses_strict_containment():
    """Proposition 2: unitary pooling strictly contains measure-and-condition.

    Witness V = SWAP. It carries rho_b onto the retained register *including*
    off-diagonal coherences. No measure-and-condition map can do that -- such a
    map sees rho_b only through its diagonal. So dephasing b before the block is
    detectable for SWAP and undetectable for the frozen block, which is exactly
    the contrast E1/E2 measured.

    This is what makes the frozen block's tie a statement about *where it sits*
    in the family, not a limitation of the family.
    """
    dev = qml.device('default.mixed', wires=2)

    @qml.qnode(dev)
    def swap_pool(dephase_b):
        qml.Hadamard(wires=1)              # b = |+>: coherence, no population bias
        if dephase_b:
            qml.PhaseFlip(0.5, wires=1)
        qml.SWAP(wires=[0, 1])
        return qml.density_matrix(wires=0)  # the retained register

    coherent = np.asarray(swap_pool(False))
    dephased = np.asarray(swap_pool(True))

    assert abs(coherent[0, 1]) == pytest.approx(0.5, abs=1e-12)
    assert abs(dephased[0, 1]) < 1e-14
    assert _trace_distance(coherent, dephased) > 0.4


@pytest.mark.slow
def test_e4_reports_physically_admissible_quantities():
    """E4's numbers must obey the bounds their definitions impose."""
    result = run_e4(image_size=SMALL_IMAGE, n_inputs=2, use_archived_weights=False)

    for tag in result['stage_order']:
        m = result['summary'][tag]
        d = 2 ** m['n_kept']
        assert 0.0 <= m['von_neumann_entropy']['mean'] <= m['n_kept'] + 1e-9, tag
        assert 1.0 / d - 1e-9 <= m['purity']['mean'] <= 1.0 + 1e-9, tag
        assert m['l1_coherence']['mean'] >= -1e-12, tag
        if 'mutual_information' in m:
            assert m['mutual_information']['mean'] >= -1e-9, tag


@pytest.mark.slow
def test_e4_confirms_the_global_state_is_pure_after_encoding():
    """A3's purity invariant, measured rather than asserted."""
    result = run_e4(image_size=SMALL_IMAGE, n_inputs=2, use_archived_weights=False)
    encoded = result['summary']['encoded']

    assert encoded['n_kept'] == result['n_qubits']
    assert encoded['purity']['mean'] == pytest.approx(1.0, abs=1e-10)
    assert abs(encoded['von_neumann_entropy']['mean']) < 1e-9


@pytest.mark.slow
def test_e4_classifier_cannot_change_the_readout_spectrum():
    """Internal consistency: with one active wire the head is a local unitary.

    Local unitaries leave the spectrum of a reduced state alone, so purity and
    entropy must be unchanged across the classifier while coherence -- which is
    basis dependent -- may move. If this ever fails, the stage instrumentation
    is mislabelling states.
    """
    result = run_e4(image_size=SMALL_IMAGE, n_inputs=2, use_archived_weights=False)
    last_pool = [t for t in result['stage_order'] if t.startswith('after_pool_')][-1]
    before, after = result['summary'][last_pool], result['summary']['after_classifier']

    if after['n_kept'] != 1:
        pytest.skip('classifier acts on more than one wire at this size')
    assert after['purity']['mean'] == pytest.approx(before['purity']['mean'], abs=1e-10)
    assert after['von_neumann_entropy']['mean'] == pytest.approx(
        before['von_neumann_entropy']['mean'], abs=1e-10)


def test_unknown_pooling_mode_is_rejected():
    with pytest.raises(ValueError, match='Unknown pooling_mode'):
        QuantumNativePooling.apply_pooling('bogus', ANGLES, [0], [1])
