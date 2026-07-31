#!/usr/bin/env python3
"""Phase 3 -- model analysis: trainability and capacity of the frozen ansatz.

Roadmap M3. Each item is a ``run_*`` function writing JSON evidence into
``Results/evidence/``, following ``experiments/pooling_analysis.py``.

Implemented here:

  3.1 dynamical Lie algebra -- the dimension of the Lie closure of the ansatz's
      gate generators, by iterated commutators. The same number certifies
      trainability (Ragone et al. 2024) and classical simulability (g-sim), in
      opposite directions, so both halves are reported. Three generator sets are
      closed because "the ansatz's generators" is ambiguous once fixed gates are
      interleaved with trainable ones -- see the section header below.

  3.2 gradient variance  -- how Var[d<Z>/dtheta] scales with qubit count. A barren
      plateau is exponential decay in n; a trainable ansatz decays polynomially.
      Both laws are fitted and the data chooses (``capacity.variance_decay_fit``).
      This is the F-E source data.

  3.4 expressibility + Meyer-Wallach -- how much of state space the ansatz
      reaches, as the KL divergence of its output-fidelity distribution from the
      Haar law (Sim et al. 2019), plus the mean entangling capability Q. The
      *input is fixed and the parameters vary*, so what is measured is the
      ansatz's expressibility rather than the data's spread.

  3.5 generalization bound -- Caro et al. (2022) ``sqrt(T log T / N)``. T is
      counted off the frozen tape rather than taken from the parameter-slot
      total, because the two differ in both directions (see
      ``trainable_gate_counts``).

Every measure used here is validated against analytically known cases in
``tests/test_capacity.py`` and ``tests/test_state_metrics.py`` before any circuit
output is believed -- the convention that caught a sign defect in E4's entropy.

Usage:
  python -m experiments.model_analysis --experiment dla
  python -m experiments.model_analysis --experiment gradient_variance
  python -m experiments.model_analysis --experiment expressibility
  python -m experiments.model_analysis --experiment generalization_bound
  python -m experiments.model_analysis --experiment all

``both`` is kept as the pre-3.5 pair (gradient variance + expressibility).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pennylane as qml

from QCNN import circuits, freeze
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.encoding import PureQuantumEncoder
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.utils import capacity, lie_algebra, state_metrics

# image_size -> n_qubits under amplitude encoding: 4, 6, 8, 10, 12, 14.
# 28 is the frozen headline (freeze.HEADLINE_IMAGE_SIZE).
SCALING_IMAGE_SIZES = (4, 8, 16, 28, 64, 128)

# Barren-plateau sampling draws parameters from the full periodic range of the
# rotation angles, which is the convention in McClean et al. (2018) and what
# makes the variance a statement about the ansatz rather than about one
# initialiser's width.
PARAMETER_RANGE = (0.0, 2.0 * np.pi)

# Seeds are fixed so the sweep is reproducible; distinct from the frozen
# regression seed so this study cannot be confused with the freeze fixtures.
GRADIENT_SEED = 20260726
EXPRESSIBILITY_SEED = 20260727

GRADIENT_EVIDENCE = os.path.join('Results', 'evidence', 'f_e_gradient_variance.json')
EXPRESSIBILITY_EVIDENCE = os.path.join('Results', 'evidence', 't6_expressibility.json')


def _config(image_size: int) -> QuantumNativeConfig:
    """The frozen headline configuration, resized. Nothing else varies."""
    cfg = QuantumNativeConfig.from_image_size(image_size, freeze.HEADLINE_ENCODING)
    cfg.seed = freeze.HEADLINE_SEED
    return cfg


def _random_inputs(rng, cfg, n_inputs: int) -> np.ndarray:
    """Normalised amplitude vectors. Gradient variance needs no labelled data."""
    return PureQuantumEncoder.precompute_amplitudes(
        rng.random((n_inputs, 2 ** cfg.n_qubits)), cfg.n_qubits)


# ---------------------------------------------------------------------------
# 3.2 -- gradient variance (F-E)
#
# For each qubit count, draw parameter vectors uniformly from the rotation
# range, differentiate <Z_readout> with respect to every slot, and take the
# variance of each slot's gradient across draws. Reported per slot, then reduced
# to one series per qubit count so the decay law can be fitted.
# ---------------------------------------------------------------------------

def _designated_slot(model) -> int:
    """A fixed slot to track across qubit counts: the first conv0 angle.

    conv0 is the only convolution group with non-zero effective parameters at
    the headline size (M0: conv1-3 contribute 0 of 48 each), so tracking a slot
    there measures a parameter that demonstrably moves the readout rather than
    one that is structurally dead and trivially flat.
    """
    ranges = freeze.slot_ranges(model)
    for name in ('conv0', 'conv_0', 'conv'):
        if name in ranges:
            return int(ranges[name][0])
    return int(min(start for start, _ in ranges.values()))


def _gradient_samples(model, cfg, rng, n_inits: int, batch: int) -> np.ndarray:
    """``(n_inits * batch, n_slots)`` gradients of <Z> w.r.t. every slot.

    Uses the batched backprop QNode -- proven equivalent to the sequential
    adjoint oracle to 1e-10 in tests/test_batched_execution.py -- because it is
    ~87x faster and this sweep is thousands of gradient evaluations.
    """
    n_slots = len(model._flatten_params(model.quantum_params))
    low, high = PARAMETER_RANGE
    rows = []

    for _ in range(n_inits):
        flat = qml.numpy.array(rng.uniform(low, high, n_slots), requires_grad=True)
        inputs = _random_inputs(rng, cfg, batch)
        jacobian = qml.jacobian(lambda p: model.batched_circuit(inputs, p))(flat)
        rows.append(np.atleast_2d(np.asarray(jacobian, dtype=float)))

    return np.vstack(rows)


def run_gradient_variance(image_sizes=SCALING_IMAGE_SIZES, n_inits: int = 200,
                          batch: int = 4) -> dict:
    """Gradient-variance sweep over the scaling family (roadmap 3.2)."""
    rng = np.random.default_rng(GRADIENT_SEED)
    per_n, series = [], []

    for image_size in image_sizes:
        cfg = _config(image_size)
        model = PureQuantumNativeCNN(cfg)
        slot = _designated_slot(model)

        started = time.time()
        gradients = _gradient_samples(model, cfg, rng, n_inits, batch)
        elapsed = time.time() - started

        variances = gradients.var(axis=0, ddof=1)
        # "Live" slots are those the readout actually depends on; dead slots have
        # variance ~0 by construction and would dilute any average over slots.
        live = variances > freeze.EFFECTIVE_GRADIENT_TOL

        record = {
            'image_size': int(image_size),
            'n_qubits': int(cfg.n_qubits),
            'n_slots': int(gradients.shape[1]),
            'n_live_slots': int(live.sum()),
            # Recorded so the live set can be compared against the M0 effective-
            # parameter fixture, not merely its cardinality: at n=10 both give 74,
            # from different criteria and different weights.
            'live_slots': [int(i) for i in np.flatnonzero(live)],
            'n_draws': int(gradients.shape[0]),
            'designated_slot': slot,
            'designated_slot_variance': float(variances[slot]),
            'designated_slot_mean_gradient': float(gradients[:, slot].mean()),
            'max_slot_variance': float(variances.max()),
            'median_live_slot_variance': float(np.median(variances[live])) if live.any() else 0.0,
            'mean_live_slot_variance': float(variances[live].mean()) if live.any() else 0.0,
            'mean_abs_gradient_live': float(np.abs(gradients[:, live]).mean()) if live.any() else 0.0,
            'seconds': float(elapsed),
        }
        per_n.append(record)
        series.append(record['median_live_slot_variance'])
        print('  n={:>2d} (image {:>3d})  live {:>3d}/{:<3d}  median Var {:.6e}  '
              '[{:.1f}s]'.format(record['n_qubits'], image_size, record['n_live_slots'],
                                 record['n_slots'], series[-1], elapsed))

    n_values = [r['n_qubits'] for r in per_n]
    fit = capacity.variance_decay_fit(n_values, series)
    designated_fit = capacity.variance_decay_fit(
        n_values, [r['designated_slot_variance'] for r in per_n])

    return {
        'experiment': '3.2 gradient variance',
        'artifact': 'F-E',
        'claim': 'how gradient variance scales with qubit count for the frozen ansatz',
        'device': 'default.qubit (backprop)',
        'parameter_distribution': 'uniform [0, 2pi) per slot',
        'variance_over': 'random parameter initialisations x random inputs',
        'n_inits': int(n_inits),
        'batch_per_init': int(batch),
        'seed': GRADIENT_SEED,
        'gradient_tolerance_for_live_slots': freeze.EFFECTIVE_GRADIENT_TOL,
        'reduction_for_fit': 'median variance over live slots',
        'per_n': per_n,
        'fit_median_live': fit,
        'fit_designated_slot': designated_fit,
        'caveats': [
            'The slot count grows with n (188 at n=4 to 278 at n=12), so the '
            'median-over-live-slots series is taken over a set whose size and '
            'group composition change across the sweep. The designated conv0 '
            'slot is the controlled comparison: same index, same role, every n.',
            'Variance is taken over random parameter initialisations AND random '
            'inputs, which is what a training run experiences; the textbook '
            'barren-plateau statement fixes the input distribution.',
            'Inputs are uniform random amplitude vectors, not MNIST. Gradient '
            'variance here is a property of the ansatz, not of the dataset.',
            'Six points span n=4..14, so the decay-law preference rests on a '
            'short series; R-squared is reported for both laws rather than one '
            'being assumed.',
        ],
    }


def _report_gradient_variance(result: dict) -> None:
    print('\n3.2 -- gradient variance across the scaling family')
    print('  parameters         : {}'.format(result['parameter_distribution']))
    print('  draws per n        : {} inits x {} inputs'.format(
        result['n_inits'], result['batch_per_init']))
    print()
    print('  {:>3s}  {:>9s}  {:>14s}  {:>14s}  {:>14s}'.format(
        'n', 'live/all', 'median Var', 'designated Var', 'mean |grad|'))
    for r in result['per_n']:
        print('  {:>3d}  {:>4d}/{:<4d}  {:>14.6e}  {:>14.6e}  {:>14.6e}'.format(
            r['n_qubits'], r['n_live_slots'], r['n_slots'],
            r['median_live_slot_variance'], r['designated_slot_variance'],
            r['mean_abs_gradient_live']))

    for label, fit in (('median over live slots', result['fit_median_live']),
                       ('designated conv0 slot', result['fit_designated_slot'])):
        print('\n  fit -- {}'.format(label))
        print('    exponential  Var ~ {:.4g} exp(-{:.4f} n)   R2 = {:.4f}'.format(
            fit['exponential_prefactor'], fit['exponential_rate'],
            fit['exponential_r_squared']))
        print('    power law    Var ~ {:.4g} n^-{:.4f}        R2 = {:.4f}'.format(
            fit['power_law_prefactor'], fit['power_law_exponent'],
            fit['power_law_r_squared']))
        print('    preferred    : {}'.format(fit['preferred_law']))
        print('    variance ratio per added qubit : {:.4f}'.format(
            fit['variance_ratio_per_qubit']))


# ---------------------------------------------------------------------------
# 3.4 -- expressibility and entangling capability (T6)
#
# Sim et al. (2019): sample pairs of parameter vectors, take the fidelity
# between the two output states, and compare that distribution against the Haar
# law. The input is held fixed throughout, so the spread measured is the
# ansatz's, not the data's.
# ---------------------------------------------------------------------------

def haar_reference_fidelities(rng, n_qubits: int, size: int) -> np.ndarray:
    """Exact samples from the Haar fidelity law, by inverse CDF.

    ``P(F) = (N-1)(1-F)^(N-2)`` has CDF ``1-(1-F)^(N-1)``, so
    ``F = 1 - u^(1/(N-1))`` for uniform ``u``. Used as a same-sample-size
    reference: a finite sample of the Haar law itself does not score KL = 0, so
    the ansatz's score is only interpretable against it.
    """
    dimension = 2 ** n_qubits
    u = rng.random(size)
    return 1.0 - u ** (1.0 / (dimension - 1))


def _state_qnode(cfg, model):
    """QNode returning the full statevector after the frozen circuit."""
    dev = qml.device('default.qubit', wires=cfg.n_qubits)
    hooks = circuits.CircuitHooks(terminal=lambda readout, active: qml.state())

    @qml.qnode(dev)
    def circuit(x, flat):
        return circuits.build_circuit(x, model._unflatten_params(flat), cfg, hooks=hooks)

    return circuit


def run_expressibility(image_size: int = freeze.HEADLINE_IMAGE_SIZE,
                       n_pairs: int = 1000, n_bins: int = 75) -> dict:
    """Expressibility KL and Meyer-Wallach Q for the frozen ansatz (roadmap 3.4)."""
    cfg = _config(image_size)
    model = PureQuantumNativeCNN(cfg)
    circuit = _state_qnode(cfg, model)

    rng = np.random.default_rng(EXPRESSIBILITY_SEED)
    n_slots = len(model._flatten_params(model.quantum_params))
    low, high = PARAMETER_RANGE

    # One fixed input for the whole ensemble: this is the ansatz's expressibility.
    fixed_input = _random_inputs(rng, cfg, 1)[0]

    started = time.time()
    fidelities, q_values = [], []
    for _ in range(n_pairs):
        flat_a = rng.uniform(low, high, n_slots)
        flat_b = rng.uniform(low, high, n_slots)
        state_a = np.asarray(circuit(fixed_input, flat_a)).reshape(-1)
        state_b = np.asarray(circuit(fixed_input, flat_b)).reshape(-1)

        fidelities.append(float(np.abs(np.vdot(state_a, state_b)) ** 2))
        q_values.append(state_metrics.meyer_wallach(state_a, cfg.n_qubits))
    elapsed = time.time() - started

    expressibility = capacity.expressibility_kl(fidelities, cfg.n_qubits, n_bins=n_bins)
    q = np.asarray(q_values, dtype=float)

    # The KL depends on the binning -- a known property of this measure, and at
    # n=10 the Haar law puts >99% of its mass in the first bin, so the value is
    # especially sensitive there. Reporting a spread rather than one number
    # keeps that visible instead of implying a precision the measure lacks.
    bin_sensitivity = {
        str(b): capacity.expressibility_kl(fidelities, cfg.n_qubits, n_bins=b)['kl_divergence']
        for b in (25, 50, 75, 100, 150)
    }

    # Controls, without which a KL near zero cannot be read.
    #
    # (a) A finite sample of the Haar law itself does not score exactly zero, so
    #     it is the reference the ansatz must be compared against -- not 0.
    # (b) At n=10 the Haar law already puts 0.999999 of its mass in the first of
    #     75 linear bins. That is a resolution limit, so it is reported; but the
    #     deliberately-inexpressible arms below show the measure still separates
    #     gross deviations, which is what makes a null here meaningful rather
    #     than vacuous.
    control_rng = np.random.default_rng(EXPRESSIBILITY_SEED + 1)
    haar_reference = haar_reference_fidelities(control_rng, cfg.n_qubits, n_pairs)
    inflated = np.clip(haar_reference * 10.0, 0.0, 1.0)
    degenerate = np.clip(control_rng.normal(0.9, 0.02, n_pairs), 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    controls = {
        'haar_reference_kl': capacity.expressibility_kl(
            haar_reference, cfg.n_qubits, n_bins=n_bins)['kl_divergence'],
        'haar_reference_mean_fidelity': float(haar_reference.mean()),
        'inflated_fidelity_10x_kl': capacity.expressibility_kl(
            inflated, cfg.n_qubits, n_bins=n_bins)['kl_divergence'],
        'degenerate_ensemble_kl': capacity.expressibility_kl(
            degenerate, cfg.n_qubits, n_bins=n_bins)['kl_divergence'],
        'haar_mass_in_first_bin': float(capacity.haar_bin_probabilities(
            edges, cfg.n_qubits)[0]),
        'note': ('a KL near the Haar-reference value means indistinguishable '
                 'from Haar at this sample size; the inflated and degenerate '
                 'arms show the measure still separates gross deviations at '
                 'this n, so the null is not an artefact of the binning'),
    }

    return {
        'experiment': '3.4 expressibility and entangling capability',
        'artifact': 'T6',
        'claim': "the frozen ansatz's coverage of state space and mean entanglement",
        'n_qubits': int(cfg.n_qubits),
        'image_size': int(image_size),
        'device': 'default.qubit',
        'weights': 'random parameters, uniform [0, 2pi)',
        'input': 'one fixed amplitude-encoded input (ansatz expressibility, not data spread)',
        'seed': EXPRESSIBILITY_SEED,
        'n_pairs': int(n_pairs),
        'n_slots': int(n_slots),
        'expressibility': expressibility,
        'expressibility_kl_by_bin_count': bin_sensitivity,
        'controls': controls,
        'meyer_wallach': {
            'mean': float(q.mean()),
            'std': float(q.std(ddof=0)),
            'min': float(q.min()),
            'max': float(q.max()),
            'n_states': int(q.size),
        },
        'seconds': float(elapsed),
    }


def _report_expressibility(result: dict) -> None:
    e = result['expressibility']
    q = result['meyer_wallach']
    print('\n3.4 -- expressibility and entangling capability')
    print('  n_qubits           : {}'.format(result['n_qubits']))
    print('  parameter pairs    : {}'.format(result['n_pairs']))
    print('  input              : fixed')
    print()
    print('  expressibility KL  : {:.6f}   (0 = Haar-like; {} bins)'.format(
        e['kl_divergence'], e['n_bins']))
    print('  mean fidelity      : {:.6e}   (Haar: {:.6e})'.format(
        e['mean_fidelity'], e['haar_mean_fidelity']))
    bins = result['expressibility_kl_by_bin_count']
    print('  KL by bin count    : {}'.format(
        '  '.join('{}:{:.4f}'.format(b, bins[b]) for b in sorted(bins, key=int))))
    c = result['controls']
    print()
    print('  controls (same sample size, same binning):')
    print('    Haar reference KL     : {:.6f}   <- the value "Haar-like" actually means'.format(
        c['haar_reference_kl']))
    print('    10x fidelity KL       : {:.4f}'.format(c['inflated_fidelity_10x_kl']))
    print('    degenerate KL         : {:.4f}'.format(c['degenerate_ensemble_kl']))
    print('    Haar mass in bin 0    : {:.7f}   (resolution limit)'.format(
        c['haar_mass_in_first_bin']))
    print()
    print('  Meyer-Wallach Q    : {:.4f} +/- {:.4f}   (range {:.4f} to {:.4f})'.format(
        q['mean'], q['std'], q['min'], q['max']))
    print('  elapsed            : {:.1f}s'.format(result['seconds']))


# ---------------------------------------------------------------------------
# 3.5 -- generalization bound (T6)
#
# Caro et al. (2022) bound the generalization gap by ~sqrt(T log T / N), for T
# trainable gates and N training samples. The quantity that needs care here is
# T, not the arithmetic: the manuscript's "269 parameters" is not a gate count,
# and neither is M0's "74 effective". Both are counted, and the tape is counted
# too, so the reader can see how far apart they are.
# ---------------------------------------------------------------------------

HEADLINE_SPLIT_MANIFEST = os.path.join('Results', 'manifests',
                                       'idx_0v1_n12665_seed42.json')
GENERALIZATION_EVIDENCE = os.path.join('Results', 'evidence',
                                       't6_generalization_bound.json')


def _slots_driven_by(operation: dict) -> list:
    """Slot indices an operation reads, from a ``freeze.tape_signature`` entry."""
    return [int(p[len('slot'):]) for p in operation['params'] if p.startswith('slot')]


def trainable_gate_counts(model, effective_slots) -> dict:
    """Count trainable *gates* on the frozen tape. This is Caro's T.

    Counting parameter slots instead would be wrong in two directions at once:

    - most allocated slots never reach the tape (the classifier allocates 32 and
      indexes them modularly, the dead conv groups allocate 48 each and are
      never read), so the allocated total overstates the circuit; and
    - several slots that do reach the tape drive more than one gate, through
      that same modular indexing, so the effective slot count understates it.

    Both directions are reported rather than reconciled, because the manuscript
    currently quotes a parameter count where the theorem wants a gate count.
    """
    signature = freeze.circuit_signature(model)
    effective = {int(s) for s in effective_slots}

    n_trainable = n_from_effective = 0
    gates_per_slot = {}
    for op in signature['operations']:
        slots = _slots_driven_by(op)
        if not slots:
            continue
        n_trainable += 1
        if effective.intersection(slots):
            n_from_effective += 1
        for slot in slots:
            gates_per_slot[slot] = gates_per_slot.get(slot, 0) + 1

    return {
        'n_operations_total': len(signature['operations']),
        'n_trainable_gates': n_trainable,
        'n_trainable_gates_from_effective_slots': n_from_effective,
        'n_slots_on_tape': len(gates_per_slot),
        'n_slots_driving_multiple_gates': sum(1 for c in gates_per_slot.values() if c > 1),
        'max_gates_from_one_slot': max(gates_per_slot.values()) if gates_per_slot else 0,
    }


def _headline_train_size(manifest_path: str) -> tuple:
    """``(n_train, split_id)`` read from the clean 60/15/25 manifest.

    Read rather than hardcoded so N cannot drift out of agreement with the split
    the headline was actually trained on.
    """
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    return len(manifest['train_idx']), manifest['id']


def run_generalization_bound(manifest_path: str = HEADLINE_SPLIT_MANIFEST,
                             fixture_path: str = None) -> dict:
    """Caro generalization scaling at the frozen headline (roadmap 3.5)."""
    fixture_path = fixture_path or freeze.EFFECTIVE_PARAMS_FIXTURE
    with open(fixture_path) as fh:
        fixture = json.load(fh)

    n_train, split_id = _headline_train_size(manifest_path)
    started = time.time()
    gates = trainable_gate_counts(freeze.build_headline_model(),
                                  fixture['effective_slots'])
    elapsed = time.time() - started

    # Four readings of T. The first two are gate counts and are what the theorem
    # asks for; the last two are the parameter counts the manuscript currently
    # quotes, carried so the difference is visible rather than silently resolved.
    readings = {
        'trainable_gates': gates['n_trainable_gates'],
        'trainable_gates_from_effective_slots':
            gates['n_trainable_gates_from_effective_slots'],
        'effective_parameter_slots': int(fixture['n_effective']),
        'allocated_parameter_slots': int(fixture['n_allocated']),
    }
    bounds = {name: capacity.caro_generalization_bound(t, n_train)
              for name, t in readings.items()}

    return {
        'experiment': '3.5 generalization bound',
        'artifact': 'T6',
        'claim': 'Caro et al. (2022) sqrt(T log T / N) at the frozen headline',
        'n_qubits': freeze.HEADLINE_N_QUBITS,
        'image_size': freeze.HEADLINE_IMAGE_SIZE,
        'n_train': n_train,
        'split_manifest': os.path.basename(manifest_path),
        'split_id': split_id,
        'gate_audit': gates,
        'T_readings': readings,
        'bounds': bounds,
        'preferred_reading': 'trainable_gates',
        'seconds': float(elapsed),
        'caveats': [
            'The theorem counts trainable gates, not parameters. T is therefore '
            'read off the frozen tape. The parameter-count readings are carried '
            'only so the gap is visible: the allocated total (269) overstates '
            'the circuit because most allocated slots never reach the tape, and '
            'the effective total (74) understates it because slots are reused '
            'across gates by the classifier block\'s modular indexing.',
            'Caro et al. state a big-O result. What is reported is the scaling '
            'term with no constant attached, so "non-vacuous" means below 1 up '
            'to that unquantified constant and must be stated that way.',
            'The effective-slot set comes from the M0 audit criterion (max '
            '|gradient| over 20 fixed inputs at archived weights, tolerance '
            '1e-12). Run 3.2 reproduced the same count of 74 at n=10 from an '
            'independent criterion, so the split is structural, not an artifact '
            'of that threshold.',
            'N is the clean 60/15/25 training split, which is 14% smaller than '
            'the historical 70/30 training set. The bound is computed against '
            'the protocol the headline was actually trained under.',
        ],
    }


# ---------------------------------------------------------------------------
# 3.1 -- dynamical Lie algebra (T6 / F-E)
#
# The DLA is the Lie closure of the ansatz's gate generators. Its dimension is
# what two separate results are stated over -- Ragone et al. (2024) / Fontana et
# al. (2024) for gradient variance, and the Lie-algebraic ("g-sim") simulation
# results of Somma et al. (2006) / Goh et al. (2023) for classical simulability.
# A polynomially-sized DLA gives a trainability certificate *and* an efficient
# classical simulation of the family; an exponentially-sized one gives neither.
# Both halves come from the same number, so 3.1 reports both.
#
# Three generator sets are computed, because "the ansatz's generators" is not
# one thing once fixed gates are interleaved with trainable ones:
#
#   parameterized -- generators of the trainable gates exactly as they appear.
#       The literal reading of UPGRADE_PLAN.md 3.1. It ignores the 96 fixed
#       CNOTs and the 8 fixed RY(0.02)s, so it is a lower bound on anything the
#       circuit can do, not a description of it.
#
#   propagated -- the same generators conjugated through the fixed gates that
#       precede them, giving H~_k = A_k^dag H_k A_k with A_k the fixed prefix.
#       The circuit then really is F . prod_k exp(-i theta_k H~_k), which is the
#       form the variance expressions assume. **This is the algebra that governs
#       trainability**, and using `parameterized` in its place would be quoting
#       a theorem outside its hypotheses.
#
#   full -- parameterized generators plus the generators of the fixed gates.
#       Every gate is then exp of an element, so exp(g_full) provably contains
#       the whole circuit unitary. **This is the algebra the simulability
#       argument needs**, since g-sim requires the circuit to lie in exp(g).
#
# By construction g_param <= g_prop <= g_full, so the three bracket the answer.
# ---------------------------------------------------------------------------

DLA_EVIDENCE = os.path.join('Results', 'evidence', 't6_dynamical_lie_algebra.json')
DLA_SEED = 20260730

# 4^10 - 1 = 1,048,575 basis elements is the largest exact enumeration this
# machine completes in minutes rather than hours, so the headline n=10 closes
# exactly and n=12/14 do not. Above this the closure is capped and its dimension
# is reported as a lower bound -- a truncated closure is not a dimension.
EXACT_CLOSURE_MAX_QUBITS = 10
TRUNCATION_CAP = 120000


def _closure_cap(n_qubits: int, truncation_cap: int = TRUNCATION_CAP) -> int:
    """Dimension cap for the closure at this qubit count.

    In the exact regime the cap is ``4^n``, one above the su(2^n) ceiling and so
    unreachable -- the closure stops when it closes, never because of the cap.
    """
    if n_qubits <= EXACT_CLOSURE_MAX_QUBITS:
        return 4 ** n_qubits
    return min(4 ** n_qubits, truncation_cap)


def _fixed_gate_key(operation) -> tuple:
    """Identity of a fixed gate, for caching its conjugation table."""
    return (operation.name, tuple(round(float(v), 12) for v in operation.data))


def _tape_gates(cfg, model) -> list:
    """``(operation, is_parameterized)`` for every gate after state preparation.

    State preparation is excluded on purpose: ``AmplitudeEmbedding`` carries the
    *input*, so it belongs to the state the ansatz acts on, not to the ansatz.
    Its ~2,026-CNOT Mottonen decomposition would otherwise dominate the algebra
    with data-dependent angles that no theorem here quantifies over.
    """
    n_slots = len(model._flatten_params(model.quantum_params))
    marker = qml.numpy.array(np.arange(1, n_slots + 1, dtype=float), requires_grad=True)
    encoded = _random_inputs(np.random.default_rng(DLA_SEED), cfg, 1)[0]
    tape = freeze.headline_tape(model, encoded, marker)
    signature = freeze.tape_signature(tape, n_slots)

    gates = []
    for operation, entry in zip(tape.operations, signature['operations']):
        if operation.name in freeze._STATE_PREP_OPS:
            continue
        parameterized = any(p.startswith('slot') for p in entry['params'])
        gates.append((operation, parameterized))
    return gates


def dla_generator_sets(cfg, model) -> dict:
    """The three generator sets described above, plus the gate census."""
    n_qubits = cfg.n_qubits
    parameterized, propagated, fixed = [], [], []
    tables, prefix = {}, []

    for operation, is_parameterized in _tape_gates(cfg, model):
        wires = [int(w) for w in operation.wires]
        if is_parameterized:
            generator = lie_algebra.gate_generator(operation.name, wires, n_qubits)
            parameterized.append(generator)
            # A_k^dag H_k A_k with A_k = f_m ... f_1 the fixed prefix, so the
            # conjugations apply latest-fixed-gate-first.
            carried = generator
            for key, fixed_wires in reversed(prefix):
                carried = lie_algebra.conjugate(carried, tables[key], fixed_wires, n_qubits)
            propagated.append(carried)
        else:
            matrix = qml.matrix(operation, wire_order=list(operation.wires))
            key = _fixed_gate_key(operation)
            if key not in tables:
                tables[key] = lie_algebra.conjugation_table(matrix, len(wires))
            fixed.append(lie_algebra.unitary_generator(matrix, wires, n_qubits))
            prefix.append((key, wires))

    sets = {
        'parameterized': parameterized,
        'propagated': propagated,
        'full': parameterized + fixed,
        'n_parameterized_gates': len(parameterized),
        'n_fixed_gates': len(fixed),
    }
    sets['conditioning'] = {name: _generator_conditioning(sets[name])
                            for name in ('parameterized', 'propagated', 'full')}
    return sets


def _generator_conditioning(generators) -> dict:
    """How far the generator coefficients sit above the dependence tolerance.

    The one numerical hazard in the propagated set would be conjugation through
    the fixed ``RY(0.02)`` gates driving coefficients toward ``sin(0.02)^k``: a
    pivot normalised by a near-tolerance coefficient amplifies round-off. It does
    not happen here -- CNOT conjugation is Clifford, so it maps a Pauli string to
    a single Pauli string, and every ``RY(0.02)`` acts on a discarded wire that
    no later parameterised gate touches, so it is inert under propagation exactly
    as it is inert in the circuit (decision 1). This records the margin instead
    of asserting it.
    """
    coefficients = [abs(c) for g in generators for c in g.values()]
    return {
        'min_abs_coefficient': min(coefficients) if coefficients else 0.0,
        'max_terms_per_generator': max((len(g) for g in generators), default=0),
        'dependence_tolerance': lie_algebra.TOL,
        'margin_over_tolerance': (min(coefficients) / lie_algebra.TOL
                                  if coefficients else 0.0),
    }


def _summarise_closure(closure: dict, n_qubits: int, seconds: float) -> dict:
    """Dimension plus the structure that makes it readable rather than an integer."""
    ceiling = 4 ** n_qubits - 1
    components = lie_algebra.support_components(closure['basis'], n_qubits)
    block_sum = lie_algebra.direct_sum_of_full_blocks(components)
    certificate = lie_algebra.full_su_certificate(
        closure['span'], n_qubits, components=components)

    return {
        'dimension': closure['dimension'],
        'closed': closure['closed'],
        'is_lower_bound_only': not closure['closed'],
        'commutator_rounds': closure['commutator_rounds'],
        'n_generators': closure['n_generators'],
        'n_independent_generators': closure['n_independent_generators'],
        'su_ceiling': ceiling,
        'fraction_of_su_ceiling': closure['dimension'] / ceiling,
        'is_full_su': closure['closed'] and closure['dimension'] == ceiling,
        'support_components': [list(c) for c in components],
        'component_sizes': [len(c) for c in components],
        'direct_sum_of_full_blocks': block_sum,
        'is_direct_sum_of_full_blocks': closure['closed'] and closure['dimension'] == block_sum,
        'full_su_certificate': certificate,
        'seconds': float(seconds),
    }


def _close(generators, n_qubits: int, truncation_cap: int, cap: int = None) -> dict:
    """Close a generator set. ``cap`` overrides the qubit-count rule."""
    started = time.time()
    closure = lie_algebra.lie_closure(
        generators,
        max_dimension=cap if cap is not None else _closure_cap(n_qubits, truncation_cap))
    return _summarise_closure(closure, n_qubits, time.time() - started)


# Reference ansatze with independently known DLA dimensions, closed at the same
# qubit count as the measurement. 3.4's lesson: a dimension means nothing on its
# own. The Ising arm is the load-bearing control -- it is the one that shows the
# method returns a *polynomial* dimension at n=10 when the ansatz has one, so an
# exponential answer for the FQCNN is a property of the circuit and not of the
# closure code.
def _control_generators(name: str, n_qubits: int) -> list:
    single = lie_algebra.single
    compose = lie_algebra.compose
    if name == 'local_rotations_only':
        return [{single(n_qubits, q, letter): 1.0}
                for q in range(n_qubits) for letter in 'XYZ']
    if name == 'transverse_field_ising':
        generators = [{single(n_qubits, q, 'X'): 1.0} for q in range(n_qubits)]
        return generators + [
            {compose(single(n_qubits, q, 'Z'), single(n_qubits, q + 1, 'Z')): 1.0}
            for q in range(n_qubits - 1)]
    raise ValueError('unknown control {!r}'.format(name))


def _control_expectation(name: str, n_qubits: int) -> int:
    if name == 'local_rotations_only':
        return 3 * n_qubits                      # su(2)^(+)n
    if name == 'transverse_field_ising':
        return 2 * n_qubits * n_qubits - n_qubits  # Wiersema et al. (2024)
    raise ValueError('unknown control {!r}'.format(name))


CONTROL_ANSATZE = ('local_rotations_only', 'transverse_field_ising')

# Ablation arms already defined in the config, used here to attribute the
# algebra to circuit blocks rather than reporting one number for the whole tape.
CIRCUIT_VARIANTS = {
    'frozen': {},
    'pool_none': {'pooling_mode': 'none'},
    'ent_none': {'conv_entanglement': 'none'},
}


def _variant_model(image_size: int, overrides: dict):
    cfg = _config(image_size)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg, PureQuantumNativeCNN(cfg)


def run_dla(image_sizes=SCALING_IMAGE_SIZES,
            truncation_cap: int = TRUNCATION_CAP,
            variant_image_size: int = freeze.HEADLINE_IMAGE_SIZE) -> dict:
    """Dynamical Lie algebra of the frozen ansatz (roadmap 3.1)."""
    per_n = []
    for image_size in image_sizes:
        cfg, model = _variant_model(image_size, {})
        n_qubits = cfg.n_qubits
        sets = dla_generator_sets(cfg, model)

        algebras = {}
        for name in ('parameterized', 'propagated', 'full'):
            algebras[name] = _close(sets[name], n_qubits, truncation_cap)
            summary = algebras[name]
            print('  n={:>2d} {:<14s} dim {:>9d}{}  ({}/{} of su ceiling)  '
                  'blocks {}  [{:.1f}s]'.format(
                      n_qubits, name, summary['dimension'],
                      ' (lower bound)' if summary['is_lower_bound_only'] else '',
                      summary['dimension'], summary['su_ceiling'],
                      summary['component_sizes'], summary['seconds']))

        controls = {}
        for control in CONTROL_ANSATZE:
            closure = _close(_control_generators(control, n_qubits),
                             n_qubits, truncation_cap)
            expected = _control_expectation(control, n_qubits)
            closure['expected_dimension'] = expected
            closure['agrees_with_expected'] = (closure['closed']
                                               and closure['dimension'] == expected)
            controls[control] = closure

        per_n.append({
            'image_size': int(image_size),
            'n_qubits': int(n_qubits),
            'n_slots': int(len(model._flatten_params(model.quantum_params))),
            'n_parameterized_gates': sets['n_parameterized_gates'],
            'n_fixed_gates': sets['n_fixed_gates'],
            'generator_conditioning': sets['conditioning'],
            'closure_cap': _closure_cap(n_qubits, truncation_cap),
            'algebras': algebras,
            'controls': controls,
        })

    # Block attribution at one qubit count: which part of the tape supplies the
    # entanglement the algebra is built from. Two economies here, both stated
    # rather than hidden:
    #
    #   - the frozen arm is reused from the sweep when it is already there,
    #     because re-closing su(2^10) exactly costs ~20 minutes; and
    #   - the variant closures are capped at ``truncation_cap`` even where the
    #     sweep would enumerate exactly. This is an attribution study, not the
    #     headline measurement: what it needs is which arms are exponential and
    #     which are not, and where an arm hits the cap the su(2^n) certificate
    #     supplies the exact dimension from the universality result. The headline
    #     numbers stay exact enumerations.
    already_run = {r['image_size']: r for r in per_n}
    variants = {}
    for name, overrides in CIRCUIT_VARIANTS.items():
        cfg, model = _variant_model(variant_image_size, overrides)
        sets = dla_generator_sets(cfg, model)
        reusable = already_run.get(variant_image_size) if not overrides else None
        print('  variant {:<10s} (n={})'.format(name, cfg.n_qubits))
        variants[name] = {
            'overrides': overrides,
            'n_qubits': int(cfg.n_qubits),
            'n_parameterized_gates': sets['n_parameterized_gates'],
            'n_fixed_gates': sets['n_fixed_gates'],
            'reused_from_sweep': reusable is not None,
            'closure_cap': None if reusable is not None else int(truncation_cap),
            'algebras': reusable['algebras'] if reusable is not None else {
                key: _close(sets[key], cfg.n_qubits, truncation_cap,
                            cap=truncation_cap)
                for key in ('parameterized', 'propagated', 'full')},
        }

    headline = next((r for r in per_n
                     if r['n_qubits'] == freeze.HEADLINE_N_QUBITS), None)
    return {
        'experiment': '3.1 dynamical Lie algebra',
        'artifact': 'T6 / F-E',
        'claim': ('dimension of the Lie closure of the frozen ansatz generators, '
                  'and what it certifies about trainability and simulability'),
        'method': 'iterated commutators to closure, exact in the Pauli basis',
        'seed': DLA_SEED,
        'state_preparation': 'excluded -- AmplitudeEmbedding carries the input, '
                             'not the ansatz',
        'exact_closure_max_qubits': EXACT_CLOSURE_MAX_QUBITS,
        'truncation_cap': int(truncation_cap),
        'generator_conventions': {
            'parameterized': 'generators of the trainable gates as they appear; '
                             'the literal 3.1 spec, and a lower bound only',
            'propagated': 'trainable generators conjugated through the fixed '
                          'prefix, so the circuit is F . prod exp(-i theta H~); '
                          'the algebra the Ragone/Fontana variance expressions '
                          'are stated over',
            'full': 'trainable plus fixed-gate generators, so exp(g) provably '
                    'contains the circuit; the algebra g-sim simulability needs',
        },
        'per_n': per_n,
        'circuit_variants': variants,
        'headline': None if headline is None else {
            'n_qubits': headline['n_qubits'],
            'parameterized_dimension': headline['algebras']['parameterized']['dimension'],
            'propagated_dimension': headline['algebras']['propagated']['dimension'],
            'full_dimension': headline['algebras']['full']['dimension'],
            'su_ceiling': headline['algebras']['full']['su_ceiling'],
        },
        'caveats': [
            'The three generator sets are not interchangeable. Quoting the '
            'parameterized-only dimension as "the DLA" would understate the '
            'circuit by ignoring 96 fixed CNOTs, and it is the understatement '
            'that manufactures a favourable trainability result.',
            'A polynomially-sized DLA cuts both ways: it would certify no '
            'barren plateau (Ragone et al. 2024) AND imply the family is '
            'efficiently classically simulable at arbitrary n (g-sim). Neither '
            'half may be reported without the other.',
            'Dimensions marked as lower bounds hit the truncation cap and are '
            'not closures. Where the su(2^n) certificate holds, the exact value '
            'follows from the universality result rather than from enumeration, '
            'and is labelled as such.',
            'The DLA is a property of the ansatz, not of the trained weights or '
            'of the data. It says nothing on its own about the loss landscape '
            'at the archived parameters.',
        ],
    }


def _report_dla(result: dict) -> None:
    print('\n3.1 -- dynamical Lie algebra of the frozen ansatz')
    print('  method             : {}'.format(result['method']))
    print('  state prep         : {}'.format(result['state_preparation']))
    print()
    print('  {:>3s}  {:>10s}  {:>12s}  {:>12s}  {:>12s}'.format(
        'n', 'su ceiling', 'parameterized', 'propagated', 'full'))
    for record in result['per_n']:
        cells = []
        for name in ('parameterized', 'propagated', 'full'):
            algebra = record['algebras'][name]
            cells.append('{}{}'.format(
                '>=' if algebra['is_lower_bound_only'] else '', algebra['dimension']))
        print('  {:>3d}  {:>10d}  {:>12s}  {:>12s}  {:>12s}'.format(
            record['n_qubits'], record['algebras']['full']['su_ceiling'], *cells))

    print('\n  block structure of the parameterized algebra '
          '(pooling connectivity):')
    for record in result['per_n']:
        algebra = record['algebras']['parameterized']
        print('    n={:>2d}  components {:<16s}  sum(4^|C| - 1) = {:<8d}  '
              'matches dimension: {}'.format(
                  record['n_qubits'], str(algebra['component_sizes']),
                  algebra['direct_sum_of_full_blocks'],
                  algebra['is_direct_sum_of_full_blocks']))

    print('\n  controls (same n, dimensions known independently):')
    for record in result['per_n']:
        row = []
        for name in CONTROL_ANSATZE:
            control = record['controls'][name]
            row.append('{} {}/{} {}'.format(
                name, control['dimension'], control['expected_dimension'],
                'ok' if control['agrees_with_expected'] else 'MISMATCH'))
        print('    n={:>2d}  {}'.format(record['n_qubits'], '   '.join(row)))

    print('\n  circuit variants at n={}:'.format(
        result['circuit_variants']['frozen']['n_qubits']))
    for name, variant in result['circuit_variants'].items():
        cells = []
        for key in ('parameterized', 'propagated', 'full'):
            algebra = variant['algebras'][key]
            if algebra['is_lower_bound_only']:
                certificate = algebra['full_su_certificate']
                cells.append('>={} ({})'.format(
                    algebra['dimension'],
                    'su(2^n) by certificate = {}'.format(certificate['implied_dimension'])
                    if certificate['certified_full_su'] else 'no certificate'))
            else:
                cells.append(str(algebra['dimension']))
        print('    {:<10s}  parameterized {}   propagated {}   full {}'.format(
            name, *cells))


def _report_generalization_bound(result: dict) -> None:
    g = result['gate_audit']
    print('\n3.5 -- generalization bound (Caro et al. 2022)')
    print('  n_qubits           : {}'.format(result['n_qubits']))
    print('  N (train)          : {}   [{}]'.format(
        result['n_train'], result['split_manifest']))
    print()
    print('  tape audit:')
    print('    operations total          : {}'.format(g['n_operations_total']))
    print('    trainable gates           : {}'.format(g['n_trainable_gates']))
    print('    slots reaching the tape   : {} of {}'.format(
        g['n_slots_on_tape'], result['T_readings']['allocated_parameter_slots']))
    print('    slots driving >1 gate     : {}   (max {} gates from one slot)'.format(
        g['n_slots_driving_multiple_gates'], g['max_gates_from_one_slot']))
    print()
    print('  {:<38s} {:>6s}  {:>12s}  {:>11s}'.format(
        'reading of T', 'T', 'sqrt(TlogT/N)', 'non-vacuous'))
    for name, bound in result['bounds'].items():
        marker = ' <-' if name == result['preferred_reading'] else ''
        print('  {:<38s} {:>6d}  {:>12.4f}  {:>11s}{}'.format(
            name.replace('_', ' '), bound['n_trainable_gates'],
            bound['sqrt_T_logT_over_N'],
            'yes' if bound['non_vacuous'] else 'no', marker))
    print('\n  elapsed            : {:.1f}s'.format(result['seconds']))


def _write(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print('\n  evidence -> {}'.format(path))


def main():
    ap = argparse.ArgumentParser(description='Phase 3 model analysis (M3)')
    ap.add_argument('--experiment',
                    choices=['dla', 'gradient_variance', 'expressibility',
                             'generalization_bound', 'both', 'all'],
                    default='all')
    ap.add_argument('--image-size', type=int, default=freeze.HEADLINE_IMAGE_SIZE,
                    help='expressibility only; the gradient sweep spans the family')
    ap.add_argument('--image-sizes', type=int, nargs='+', default=list(SCALING_IMAGE_SIZES),
                    help='gradient sweep: image sizes giving n = 4, 6, 8, 10, 12, 14')
    ap.add_argument('--n-inits', type=int, default=200)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--n-pairs', type=int, default=1000)
    ap.add_argument('--n-bins', type=int, default=75)
    ap.add_argument('--split-manifest', default=HEADLINE_SPLIT_MANIFEST,
                    help='generalization bound only; supplies N')
    ap.add_argument('--variant-image-size', type=int,
                    default=freeze.HEADLINE_IMAGE_SIZE,
                    help='DLA only; qubit count for the pool_none / ent_none '
                         'block attribution')
    ap.add_argument('--truncation-cap', type=int, default=TRUNCATION_CAP,
                    help='DLA only; dimension cap above n={}, where exact '
                         'enumeration stops being affordable'.format(
                             EXACT_CLOSURE_MAX_QUBITS))
    args = ap.parse_args()

    if args.experiment in ('dla', 'all'):
        print('3.1 -- dynamical Lie algebra (closure by iterated commutators)')
        result = run_dla(image_sizes=tuple(args.image_sizes),
                         truncation_cap=args.truncation_cap,
                         variant_image_size=args.variant_image_size)
        _report_dla(result)
        _write(result, DLA_EVIDENCE)

    if args.experiment in ('gradient_variance', 'both', 'all'):
        print('3.2 -- gradient variance sweep ({} inits per qubit count)'.format(args.n_inits))
        result = run_gradient_variance(image_sizes=tuple(args.image_sizes),
                                       n_inits=args.n_inits, batch=args.batch)
        _report_gradient_variance(result)
        _write(result, GRADIENT_EVIDENCE)

    if args.experiment in ('expressibility', 'both', 'all'):
        print('\n3.4 -- expressibility ({} parameter pairs)'.format(args.n_pairs))
        result = run_expressibility(image_size=args.image_size, n_pairs=args.n_pairs,
                                    n_bins=args.n_bins)
        _report_expressibility(result)
        _write(result, EXPRESSIBILITY_EVIDENCE)

    if args.experiment in ('generalization_bound', 'all'):
        print('\n3.5 -- generalization bound')
        result = run_generalization_bound(manifest_path=args.split_manifest)
        _report_generalization_bound(result)
        _write(result, GENERALIZATION_EVIDENCE)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
