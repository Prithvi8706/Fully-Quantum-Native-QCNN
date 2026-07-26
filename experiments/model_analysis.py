#!/usr/bin/env python3
"""Phase 3 -- model analysis: trainability and capacity of the frozen ansatz.

Roadmap M3. Each item is a ``run_*`` function writing JSON evidence into
``Results/evidence/``, following ``experiments/pooling_analysis.py``.

Implemented here:

  3.2 gradient variance  -- how Var[d<Z>/dtheta] scales with qubit count. A barren
      plateau is exponential decay in n; a trainable ansatz decays polynomially.
      Both laws are fitted and the data chooses (``capacity.variance_decay_fit``).
      This is the F-E source data.

  3.4 expressibility + Meyer-Wallach -- how much of state space the ansatz
      reaches, as the KL divergence of its output-fidelity distribution from the
      Haar law (Sim et al. 2019), plus the mean entangling capability Q. The
      *input is fixed and the parameters vary*, so what is measured is the
      ansatz's expressibility rather than the data's spread.

Every measure used here is validated against analytically known cases in
``tests/test_capacity.py`` and ``tests/test_state_metrics.py`` before any circuit
output is believed -- the convention that caught a sign defect in E4's entropy.

Usage:
  python -m experiments.model_analysis --experiment gradient_variance
  python -m experiments.model_analysis --experiment expressibility
  python -m experiments.model_analysis --experiment both
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
from QCNN.utils import capacity, state_metrics

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


def _write(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print('\n  evidence -> {}'.format(path))


def main():
    ap = argparse.ArgumentParser(description='Phase 3 model analysis (M3)')
    ap.add_argument('--experiment',
                    choices=['gradient_variance', 'expressibility', 'both'],
                    default='both')
    ap.add_argument('--image-size', type=int, default=freeze.HEADLINE_IMAGE_SIZE,
                    help='expressibility only; the gradient sweep spans the family')
    ap.add_argument('--image-sizes', type=int, nargs='+', default=list(SCALING_IMAGE_SIZES),
                    help='gradient sweep: image sizes giving n = 4, 6, 8, 10, 12, 14')
    ap.add_argument('--n-inits', type=int, default=200)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--n-pairs', type=int, default=1000)
    ap.add_argument('--n-bins', type=int, default=75)
    args = ap.parse_args()

    if args.experiment in ('gradient_variance', 'both'):
        print('3.2 -- gradient variance sweep ({} inits per qubit count)'.format(args.n_inits))
        result = run_gradient_variance(image_sizes=tuple(args.image_sizes),
                                       n_inits=args.n_inits, batch=args.batch)
        _report_gradient_variance(result)
        _write(result, GRADIENT_EVIDENCE)

    if args.experiment in ('expressibility', 'both'):
        print('\n3.4 -- expressibility ({} parameter pairs)'.format(args.n_pairs))
        result = run_expressibility(image_size=args.image_size, n_pairs=args.n_pairs,
                                    n_bins=args.n_bins)
        _report_expressibility(result)
        _write(result, EXPRESSIBILITY_EVIDENCE)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
