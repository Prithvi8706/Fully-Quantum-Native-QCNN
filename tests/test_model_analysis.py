"""Phase 3 runner checks (roadmap M3).

The measures themselves are pinned against analytically known cases in
``test_capacity.py`` and ``test_state_metrics.py``. What is checked here is the
part those cannot see: that the runner feeds them the right thing.

Everything runs at image_size 4 (n=4), so the file stays cheap.
"""
import json

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


# --- 3.1 dynamical Lie algebra ------------------------------------------------
#
# The closure itself is pinned against algebras of known dimension in
# test_lie_algebra.py. What is checked here is what those tests cannot see: that
# the generators handed to the closure are the frozen circuit's, that state
# preparation is kept out of them, and -- the load-bearing one -- that the
# propagated set really does reconstruct the circuit it claims to describe.


def test_generator_extraction_covers_every_trainable_gate(small_model):
    """A missed parameterised gate understates the DLA, which is the direction
    that manufactures a favourable trainability result."""
    cfg, model = small_model
    n_slots = len(model._flatten_params(model.quantum_params))
    audit = model_analysis.trainable_gate_counts(model, range(n_slots))
    sets = model_analysis.dla_generator_sets(cfg, model)

    assert sets['n_parameterized_gates'] == audit['n_trainable_gates']
    assert len(sets['parameterized']) == len(sets['propagated'])
    # Every operation is accounted for exactly once, bar the state preparation.
    assert (sets['n_parameterized_gates'] + sets['n_fixed_gates']
            == audit['n_operations_total'] - 1)


def test_state_preparation_is_excluded_from_the_generators(small_model):
    """AmplitudeEmbedding carries the input. Folding its ~2,026-CNOT Mottonen
    decomposition into the ansatz would describe a different object entirely."""
    cfg, model = small_model
    gates = model_analysis._tape_gates(cfg, model)
    names = {operation.name for operation, _ in gates}

    assert not names.intersection(freeze._STATE_PREP_OPS)
    assert names == {'RX', 'RY', 'RZ', 'CRY', 'CRZ', 'CNOT'}


def test_propagated_generators_reconstruct_the_circuit(small_model):
    """The propagation claim, checked against the circuit itself.

    ``U = F . prod_k exp(-i theta_k H~_k / 2)`` with ``F`` the fixed gates in
    order and the product ordered later-gate-first. If the conjugation order
    were wrong this identity would fail, and the DLA would be the closure of a
    generator set belonging to no circuit.
    """
    import scipy.linalg as sla
    from QCNN.utils import lie_algebra as la

    cfg, model = small_model
    n = cfg.n_qubits
    wire_order = list(range(n))
    dim = 2 ** n

    gates = model_analysis._tape_gates(cfg, model)
    rng = np.random.default_rng(11)
    angles = rng.uniform(0.0, 2.0 * np.pi, len(gates))

    circuit = np.eye(dim, dtype=complex)
    fixed_product = np.eye(dim, dtype=complex)
    parameterized = np.eye(dim, dtype=complex)
    tables, prefix = {}, []

    for (operation, is_parameterized), theta in zip(gates, angles):
        wires = [int(w) for w in operation.wires]
        if is_parameterized:
            generator = la.gate_generator(operation.name, wires, n)
            carried = generator
            for key, fixed_wires in reversed(prefix):
                carried = la.conjugate(carried, tables[key], fixed_wires, n)
            rotation = sla.expm(-0.5j * theta * la.sentence_matrix(carried, n))
            parameterized = rotation @ parameterized
            gate = qml.matrix(getattr(qml, operation.name)(theta, wires=wires),
                              wire_order=wire_order)
        else:
            gate = qml.matrix(operation, wire_order=wire_order)
            key = model_analysis._fixed_gate_key(operation)
            if key not in tables:
                tables[key] = la.conjugation_table(
                    qml.matrix(operation, wire_order=list(operation.wires)), len(wires))
            prefix.append((key, wires))
            fixed_product = gate @ fixed_product
        circuit = gate @ circuit

    assert np.allclose(circuit, fixed_product @ parameterized, atol=1e-9)


def test_the_three_generator_sets_are_nested(small_model):
    """g_param <= g_prop <= g_full by construction. If that ordering broke, the
    three numbers would not bracket anything."""
    from QCNN.utils import lie_algebra as la

    cfg, model = small_model
    n = cfg.n_qubits
    sets = model_analysis.dla_generator_sets(cfg, model)
    dimensions = {name: la.lie_closure(sets[name], keep_basis=True)
                  for name in ('parameterized', 'propagated', 'full')}

    assert (dimensions['parameterized']['dimension']
            <= dimensions['propagated']['dimension']
            <= dimensions['full']['dimension'])
    for generator in sets['parameterized']:
        assert la.contains(dimensions['propagated']['span'], generator)
        assert la.contains(dimensions['full']['span'], generator)


def test_the_closure_cap_is_unreachable_in_the_exact_regime():
    """Below the exact threshold the cap must not be able to bind, or a closure
    that happens to fill su(2^n) would be mislabelled as truncated."""
    for n in range(2, model_analysis.EXACT_CLOSURE_MAX_QUBITS + 1):
        assert model_analysis._closure_cap(n) > 4 ** n - 1
    assert model_analysis._closure_cap(12, truncation_cap=1000) == 1000


def test_dla_run_is_well_formed_and_serialisable():
    """Runs at n=4 only, so the file stays cheap."""
    result = model_analysis.run_dla(image_sizes=(SMALL,), variant_image_size=SMALL)

    assert result['artifact'] == 'T6 / F-E'
    assert result['headline'] is None          # n=10 not in this sweep
    record = result['per_n'][0]
    assert record['n_qubits'] == 4
    for name in ('parameterized', 'propagated', 'full'):
        algebra = record['algebras'][name]
        assert algebra['closed']
        assert 0 < algebra['dimension'] <= algebra['su_ceiling']
        assert algebra['su_ceiling'] == 4 ** 4 - 1
    # No echelon objects or numpy scalars may leak into the evidence file.
    json.dumps(result)


def test_the_dla_controls_reproduce_their_known_dimensions():
    """3.4's habit: a dimension is unreadable without a reference point whose
    answer is known independently. The Ising arm is the one that shows the
    method returns a polynomial dimension when the ansatz has one."""
    result = model_analysis.run_dla(image_sizes=(SMALL,), variant_image_size=SMALL)
    controls = result['per_n'][0]['controls']

    assert controls['local_rotations_only']['dimension'] == 3 * 4
    assert controls['transverse_field_ising']['dimension'] == 2 * 4 * 4 - 4
    for control in controls.values():
        assert control['agrees_with_expected']


def test_removing_pooling_removes_the_parameterised_entanglement():
    """Attribution check: with the pooling block gone, no *trainable* gate is
    entangling, so the parameterised algebra collapses to local su(2)."""
    result = model_analysis.run_dla(image_sizes=(SMALL,), variant_image_size=SMALL)
    variants = result['circuit_variants']

    assert variants['pool_none']['algebras']['parameterized']['dimension'] == 3 * 4
    assert (variants['frozen']['algebras']['parameterized']['dimension']
            > variants['pool_none']['algebras']['parameterized']['dimension'])


# --- 3.5 generalization bound -------------------------------------------------
#
# The bound arithmetic is pinned in test_capacity.py. What matters here is that
# T is a gate count taken off the tape, because the whole point of 3.5 is that
# the parameter counts the manuscript quotes are the wrong input to the theorem.


def test_gate_count_is_not_the_slot_count(small_model):
    """T counts gates. Slots and gates disagree, and the runner must not conflate them."""
    cfg, model = small_model
    n_slots = len(model._flatten_params(model.quantum_params))
    audit = model_analysis.trainable_gate_counts(model, range(n_slots))

    assert audit['n_trainable_gates'] > 0
    assert audit['n_slots_on_tape'] <= n_slots
    assert audit['n_trainable_gates'] <= audit['n_operations_total']


def test_reused_slots_make_gates_outnumber_the_slots_that_drive_them(small_model):
    """Modular indexing means one slot can drive several gates; that must show up."""
    cfg, model = small_model
    n_slots = len(model._flatten_params(model.quantum_params))
    audit = model_analysis.trainable_gate_counts(model, range(n_slots))

    assert audit['max_gates_from_one_slot'] >= 1
    if audit['n_slots_driving_multiple_gates'] > 0:
        assert audit['n_trainable_gates'] > audit['n_slots_on_tape']


def test_headline_tape_audit_agrees_with_the_effective_params_fixture():
    """Two independent code paths must agree on how many slots reach the tape.

    The fixture counts syntactically-used slots via the M0 audit; this walks the
    frozen tape signature. Disagreement would mean one of them is describing a
    circuit the other is not.
    """
    with open(freeze.EFFECTIVE_PARAMS_FIXTURE) as fh:
        fixture = json.load(fh)
    audit = model_analysis.trainable_gate_counts(freeze.build_headline_model(),
                                                 fixture['effective_slots'])

    assert audit['n_slots_on_tape'] == fixture['n_syntactically_used']
    assert audit['n_trainable_gates_from_effective_slots'] <= audit['n_trainable_gates']


def test_generalization_run_is_well_formed():
    result = model_analysis.run_generalization_bound()

    assert result['artifact'] == 'T6'
    assert result['n_train'] > 0
    assert result['preferred_reading'] in result['bounds']
    assert set(result['T_readings']) == set(result['bounds'])
    for name, bound in result['bounds'].items():
        assert bound['n_trainable_gates'] == result['T_readings'][name]
        assert bound['n_train'] == result['n_train']


def test_the_bound_grows_with_T():
    """Monotonicity in T is what makes the four readings comparable at all."""
    result = model_analysis.run_generalization_bound()
    by_t = sorted(result['bounds'].values(), key=lambda b: b['n_trainable_gates'])
    values = [b['sqrt_T_logT_over_N'] for b in by_t]

    assert values == sorted(values)


def test_n_train_comes_from_the_manifest(tmp_path):
    """N must be read from the split, not hardcoded, or the bound can silently drift."""
    manifest = tmp_path / 'fake_split.json'
    manifest.write_text(json.dumps({'train_idx': list(range(500)), 'id': 'deadbeef'}))

    result = model_analysis.run_generalization_bound(manifest_path=str(manifest))

    assert result['n_train'] == 500
    assert result['split_id'] == 'deadbeef'
