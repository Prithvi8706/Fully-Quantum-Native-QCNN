"""Phase 3 runner checks (roadmap M3).

The measures themselves are pinned against analytically known cases in
``test_capacity.py`` and ``test_state_metrics.py``. What is checked here is the
part those cannot see: that the runner feeds them the right thing.

Everything runs at image_size 4 (n=4), so the file stays cheap.
"""
import numpy as np
import pennylane as qml
import pytest

from QCNN import freeze
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from experiments import model_analysis

SMALL = 4  # image_size 4 -> n = 4 qubits


@pytest.fixture(scope='module')
def small_model():
    cfg = model_analysis._config(SMALL)
    return cfg, PureQuantumNativeCNN(cfg)


def test_scaling_family_covers_the_intended_qubit_counts():
    """The sweep must span n = 4..14, which is what F-D and F-E are defined over."""
    counts = [model_analysis._config(s).n_qubits for s in model_analysis.SCALING_IMAGE_SIZES]
    assert counts == [4, 6, 8, 10, 12, 14]


def test_headline_size_sits_in_the_family(small_model):
    """n=10 must be one of the swept points, or the sweep misses the frozen model."""
    assert freeze.HEADLINE_IMAGE_SIZE in model_analysis.SCALING_IMAGE_SIZES
    assert model_analysis._config(freeze.HEADLINE_IMAGE_SIZE).n_qubits == freeze.HEADLINE_N_QUBITS


def test_the_designated_slot_is_not_a_dead_parameter(small_model):
    """The trap this guards: conv1-3 contribute zero effective parameters (M0).

    Tracking a structurally dead slot would show a flat, tiny gradient variance
    at every n and read as "no barren plateau" while measuring nothing at all.
    The designated slot must demonstrably move the readout.
    """
    cfg, model = small_model
    slot = model_analysis._designated_slot(model)

    rng = np.random.default_rng(0)
    n_slots = len(model._flatten_params(model.quantum_params))
    inputs = model_analysis._random_inputs(rng, cfg, 2)
    flat = qml.numpy.array(rng.uniform(0, 2 * np.pi, n_slots), requires_grad=True)

    jacobian = np.atleast_2d(np.asarray(
        qml.jacobian(lambda p: model.batched_circuit(inputs, p))(flat), dtype=float))

    assert np.abs(jacobian[:, slot]).max() > freeze.EFFECTIVE_GRADIENT_TOL, (
        'designated slot %d has no gradient; the variance series would be vacuous' % slot)


def test_designated_slot_falls_inside_the_first_conv_group(small_model):
    _, model = small_model
    slot = model_analysis._designated_slot(model)
    ranges = freeze.slot_ranges(model)
    name = next(iter(ranges))          # groups are emitted in circuit order
    start, stop = ranges[name]
    assert start <= slot < stop


def test_gradient_sweep_produces_a_well_formed_record():
    """End-to-end at two qubit counts, enough to exercise the fit."""
    result = model_analysis.run_gradient_variance(
        image_sizes=(4, 8, 16), n_inits=2, batch=1)

    assert [r['n_qubits'] for r in result['per_n']] == [4, 6, 8]
    for record in result['per_n']:
        assert record['n_live_slots'] > 0
        assert record['n_draws'] == 2
        assert record['median_live_slot_variance'] > 0.0
    assert result['fit_median_live']['preferred_law'] in ('exponential', 'power_law')
    assert result['artifact'] == 'F-E'
    assert result['caveats']


def test_the_terminal_hook_returns_a_normalised_state(small_model):
    """If the hook did not capture a real statevector, every 3.4 number is noise."""
    cfg, model = small_model
    circuit = model_analysis._state_qnode(cfg, model)

    rng = np.random.default_rng(1)
    n_slots = len(model._flatten_params(model.quantum_params))
    x = model_analysis._random_inputs(rng, cfg, 1)[0]
    state = np.asarray(circuit(x, rng.uniform(0, 2 * np.pi, n_slots))).reshape(-1)

    assert state.size == 2 ** cfg.n_qubits
    assert float(np.vdot(state, state).real) == pytest.approx(1.0, abs=1e-10)


def test_identical_parameters_give_fidelity_one(small_model):
    """The known case for the sampling machinery: no parameter change, no change.

    This is what makes a *low* sampled fidelity meaningful -- it proves the
    fidelity is responding to the parameters rather than to numerical noise.
    """
    cfg, model = small_model
    circuit = model_analysis._state_qnode(cfg, model)

    rng = np.random.default_rng(2)
    n_slots = len(model._flatten_params(model.quantum_params))
    x = model_analysis._random_inputs(rng, cfg, 1)[0]
    flat = rng.uniform(0, 2 * np.pi, n_slots)

    a = np.asarray(circuit(x, flat)).reshape(-1)
    b = np.asarray(circuit(x, flat)).reshape(-1)
    assert float(np.abs(np.vdot(a, b)) ** 2) == pytest.approx(1.0, abs=1e-12)

    different = np.asarray(circuit(x, rng.uniform(0, 2 * np.pi, n_slots))).reshape(-1)
    assert float(np.abs(np.vdot(a, different)) ** 2) < 0.999


def test_expressibility_run_is_well_formed():
    result = model_analysis.run_expressibility(image_size=SMALL, n_pairs=12, n_bins=25)

    assert result['n_qubits'] == 4
    assert result['expressibility']['n_samples'] == 12
    assert 0.0 <= result['meyer_wallach']['mean'] <= 1.0 + 1e-9
    assert result['expressibility_kl_by_bin_count']
    assert result['artifact'] == 'T6'


def test_haar_reference_sampler_matches_the_known_mean():
    """The Haar fidelity law has mean 1/N exactly; the inverse-CDF sampler must hit it.

    This sampler is the reference the ansatz's KL is judged against, so if it
    were wrong the whole 3.4 conclusion would be measured off a bad baseline.
    """
    rng = np.random.default_rng(7)
    for n in (4, 6, 8):
        sample = model_analysis.haar_reference_fidelities(rng, n, 200000)
        assert sample.mean() == pytest.approx(1.0 / 2 ** n, rel=0.05)
        assert sample.min() >= 0.0
        assert sample.max() <= 1.0


def test_haar_reference_scores_as_haar_like():
    """Scoring the Haar law against itself must land near zero, not far from it."""
    from QCNN.utils.capacity import expressibility_kl

    rng = np.random.default_rng(8)
    sample = model_analysis.haar_reference_fidelities(rng, 6, 5000)
    assert expressibility_kl(sample, 6, n_bins=75)['kl_divergence'] < 0.05


def test_expressibility_controls_separate_gross_deviations():
    """A null is only meaningful if the measure can still fail. n=4 keeps it cheap."""
    result = model_analysis.run_expressibility(image_size=SMALL, n_pairs=400, n_bins=75)
    c = result['controls']

    assert c['degenerate_ensemble_kl'] > c['haar_reference_kl'] + 1.0
    assert c['inflated_fidelity_10x_kl'] > c['haar_reference_kl']
    assert 0.0 <= c['haar_mass_in_first_bin'] <= 1.0
