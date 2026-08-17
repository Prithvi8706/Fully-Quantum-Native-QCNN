#!/usr/bin/env python3
"""Project the experiment grid's wall-clock and gate it (UPGRADE_PLAN.md 1.4).

Every number here is measured on this machine, not estimated: each config is
calibrated by timing real gradient and forward passes through its own circuit,
then projected over the requested (dataset x config x seed) grid.

The gate is roadmap M1.4: the grid must fit in seven unattended nights. If it
does not, the mandated reduction order is seeds, then datasets, then non-pooling
ablations -- and this script reports the first cut that fits rather than
applying one. E1/E2 are Phase 2 fixed-parameter experiments, not grid cells, so
they are never candidates for reduction.

Usage:
  python -m experiments.estimate_cost --jobs 0
  python -m experiments.estimate_cost --seeds 0 1 2 3 4 --samples 400 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from QCNN.encoding import PureQuantumEncoder
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training.Qtrainer import QuantumNativeTrainer
from experiments import run_experiments
from experiments.run_experiments import (
    ABLATION_CONFIGS,
    DEFAULT_MNIST_DIR,
    build_config,
    prepare_split,
)

# An "unattended night" is not defined in the plan; ten hours is this repo's
# working definition and is overridable. The gate is seven of them.
HOURS_PER_NIGHT = 10.0
NIGHT_BUDGET = 7

# Trainer constants the projection depends on (QCNN/training/Qtrainer.py).
TRAIN_ACC_PROBE = 200

# Pooling arms carry the paper's central claim; they are cut last and only after
# every other axis (roadmap M1.4, UPGRADE_PLAN.md risk register).
POOLING_ARMS = ('proposed', 'pool_none', 'pool_measurement')
SEED_FLOOR = 5          # M5.3: never fewer than five seeds per cell
DATASET_FLOOR = 3       # M4 gate: at least three dataset domains


def _synthetic_inputs(cfg, n: int, seed: int = 20260725) -> np.ndarray:
    """Inputs of the exact width this config's circuit consumes."""
    rng = np.random.default_rng(seed)
    if cfg.encoding_type in ('amplitude', 'patch'):
        return PureQuantumEncoder.precompute_amplitudes(
            rng.random((n, 2 ** cfg.n_qubits)), cfg.n_qubits)
    return rng.random((n, cfg.n_qubits))


def calibrate(config_name: str, seed: int, epochs, reps: int, forward_probe: int) -> dict:
    """Time this config's real gradient and forward passes."""
    cfg = build_config(ABLATION_CONFIGS[config_name], seed)
    if epochs is not None:
        cfg.n_epochs = epochs

    model = PureQuantumNativeCNN(cfg)
    params = model._flatten_params(model.quantum_params)
    trainer = QuantumNativeTrainer(learning_rate=cfg.learning_rate)
    trainer._pos_weight = trainer._neg_weight = 1.0

    batch = _synthetic_inputs(cfg, cfg.batch_size)
    labels = np.resize(np.array([1.0, -1.0]), cfg.batch_size)

    def cost(p):
        return trainer._bce_loss(
            pnp.atleast_1d(model.batch_expectations(pnp.array(batch), p)), labels)

    qml.grad(cost)(params)                      # warm up tracing
    start = time.perf_counter()
    for _ in range(reps):
        qml.grad(cost)(params)
    grad_s = (time.perf_counter() - start) / reps

    probe = _synthetic_inputs(cfg, forward_probe, seed=1)
    model.batch_expectations(pnp.array(probe), params)
    start = time.perf_counter()
    model.batch_expectations(pnp.array(probe), params)
    batched_fwd_s = (time.perf_counter() - start) / forward_probe

    # The single final test evaluation still goes through the per-sample path
    # in QCNN/utils/metrics.py, so it is priced at the sequential rate.
    single = probe[0]
    model.quantum_circuit(single, params)
    start = time.perf_counter()
    for i in range(min(3, forward_probe)):
        model.quantum_circuit(probe[i], params)
    seq_fwd_s = (time.perf_counter() - start) / min(3, forward_probe)

    return {
        'config': config_name,
        'n_qubits': cfg.n_qubits,
        'batch_size': cfg.batch_size,
        'epochs': cfg.n_epochs,
        'grad_s_per_batch': grad_s,
        'batched_forward_s_per_sample': batched_fwd_s,
        'sequential_forward_s_per_sample': seq_fwd_s,
        'batched': model.supports_batched(cfg.batch_size),
    }


def cell_seconds(cal: dict, n_train: int, n_val: int, n_test: int) -> dict:
    """Projected wall-clock for one (dataset, config, seed) cell."""
    n_batches = math.ceil(n_train / cal['batch_size'])
    train_s = n_batches * cal['grad_s_per_batch']
    monitor_s = (min(TRAIN_ACC_PROBE, n_train) + n_val) * cal['batched_forward_s_per_sample']
    epoch_s = train_s + monitor_s
    final_eval_s = n_test * cal['sequential_forward_s_per_sample']
    return {
        'epoch_s': epoch_s,
        'train_s_per_epoch': train_s,
        'monitor_s_per_epoch': monitor_s,
        'final_eval_s': final_eval_s,
        'total_s': cal['epochs'] * epoch_s + final_eval_s,
    }


def calibrate_baselines(seed: int, epochs, n_train: int, n_test: int) -> float:
    """Seconds of baseline work attached to each 'proposed' cell.

    Baselines run once per (dataset, seed) inside the proposed cell, so their
    cost is charged there rather than spread across the grid.
    """
    from baselines.classical_cnn import run_classical_baselines
    from baselines.quantum_baselines import _ARCHITECTURES, _train_architecture

    cfg = build_config(ABLATION_CONFIGS['proposed'], seed)
    if epochs is not None:
        cfg.n_epochs = epochs

    X_tr = _synthetic_inputs(cfg, min(n_train, 64))
    y_tr = np.resize(np.array([1.0, -1.0]), len(X_tr))
    X_val = _synthetic_inputs(cfg, min(n_test, 32), seed=2)
    y_val = np.resize(np.array([1.0, -1.0]), len(X_val))
    X_te = _synthetic_inputs(cfg, min(n_test, 32), seed=3)
    y_te = np.resize(np.array([1.0, -1.0]), len(X_te))
    split = dict(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val,
                 X_test=X_te, y_test=y_te)

    start = time.perf_counter()
    run_classical_baselines(**split, seed=seed, target_params=269)
    classical_s = time.perf_counter() - start

    quantum_s = 0.0
    for name in _ARCHITECTURES:
        start = time.perf_counter()
        _train_architecture(arch_name=name, **split, seed=seed,
                            n_qubits=cfg.n_qubits, n_epochs=1,
                            learning_rate=cfg.learning_rate, use_bce=True)
        one_epoch = time.perf_counter() - start
        quantum_s += one_epoch * cfg.n_epochs

    # Measured on a reduced probe; scale the training part to the real split.
    scale = max(1.0, n_train / max(1, len(X_tr)))
    return classical_s * scale + quantum_s * scale


def project(configs, pairs, seeds, calibrations, sizes, baseline_s, jobs, with_baselines):
    """Serial seconds and wall-clock at `jobs` workers, plus a per-cell table."""
    rows, serial_s = [], 0.0
    for ds_name in pairs:
        n_train, n_val, n_test = sizes[ds_name]
        for config_name in configs:
            cost = cell_seconds(calibrations[config_name], n_train, n_val, n_test)
            per_cell = cost['total_s']
            if with_baselines and config_name == 'proposed':
                per_cell += baseline_s
            rows.append((ds_name, config_name, cost, per_cell))
            serial_s += per_cell * len(seeds)
    return rows, serial_s, serial_s / max(1, jobs)


def propose_reduction(configs, pairs, seeds, calibrations, sizes, baseline_s, jobs,
                      with_baselines, budget_hours):
    """First cut that fits, following the mandated order (roadmap M1.4)."""
    steps = []

    # 1. Seeds, down to the M5.3 floor of five.
    for n_seeds in range(len(seeds), SEED_FLOOR - 1, -1):
        trial = seeds[:n_seeds]
        _, _, hours = _hours(configs, pairs, trial, calibrations, sizes, baseline_s,
                             jobs, with_baselines)
        if hours <= budget_hours:
            return steps + [f'seeds {len(seeds)} -> {n_seeds}'], hours
    steps.append(f'seeds {len(seeds)} -> {SEED_FLOOR} (floor)')
    seeds = seeds[:SEED_FLOOR]

    # 2. Datasets, down to the M4 floor of three.
    for n_pairs in range(len(pairs), DATASET_FLOOR - 1, -1):
        trial = pairs[:n_pairs]
        _, _, hours = _hours(configs, trial, seeds, calibrations, sizes, baseline_s,
                             jobs, with_baselines)
        if hours <= budget_hours:
            return steps + [f'datasets {len(pairs)} -> {n_pairs}'], hours
    steps.append(f'datasets {len(pairs)} -> {DATASET_FLOOR} (floor)')
    pairs = pairs[:DATASET_FLOOR]

    # 3. Non-pooling ablations. Pooling arms are never cut.
    kept = [c for c in configs if c in POOLING_ARMS]
    _, _, hours = _hours(kept, pairs, seeds, calibrations, sizes, baseline_s,
                         jobs, with_baselines)
    steps.append('drop non-pooling ablations: ' +
                 ', '.join(c for c in configs if c not in POOLING_ARMS))
    return steps, hours


def _hours(configs, pairs, seeds, calibrations, sizes, baseline_s, jobs, with_baselines):
    rows, serial_s, wall_s = project(configs, pairs, seeds, calibrations, sizes,
                                     baseline_s, jobs, with_baselines)
    return rows, serial_s / 3600.0, wall_s / 3600.0


def _write_json(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', dir=directory, delete=False,
                prefix='.campaign-cost-', suffix='.tmp') as fh:
            temporary = fh.name
            json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


def _nonfinite_field(values: dict):
    for name, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                return name
    return None


def main():
    ap = argparse.ArgumentParser(description='Project and gate the experiment grid')
    ap.add_argument('--datasets', nargs='+', default=['0,1', '3,5', '4,9', '5,8'])
    ap.add_argument('--configs', nargs='+', default=list(ABLATION_CONFIGS.keys()))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--samples', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--jobs', type=int, default=1,
                    help='Worker processes the grid will use. 0 = cores - 2.')
    ap.add_argument('--mnist-dir', default=DEFAULT_MNIST_DIR)
    ap.add_argument('--no-baselines', action='store_true')
    ap.add_argument('--reps', type=int, default=3, help='Timed gradient calls per config')
    ap.add_argument('--forward-probe', type=int, default=16)
    ap.add_argument('--hours-per-night', type=float, default=HOURS_PER_NIGHT)
    ap.add_argument('--nights', type=int, default=NIGHT_BUDGET)
    ap.add_argument('--output-json', help='Write the complete approval record as JSON')
    args = ap.parse_args()

    unknown = [c for c in args.configs if c not in ABLATION_CONFIGS]
    if unknown:
        ap.error('unknown config(s) {}; choose from {}'.format(
            ', '.join(unknown), ', '.join(sorted(ABLATION_CONFIGS))))

    budget_h = args.nights * args.hours_per_night
    if args.nights <= 0 or args.hours_per_night <= 0:
        with_baselines = not args.no_baselines
        requested_config_cells = len(args.configs) * len(args.datasets) * len(args.seeds)
        requested_baseline_cells = (len(args.datasets) * len(args.seeds)
                                    if with_baselines else 0)
        failure = {
            'kind': 'request', 'name': 'budget',
            'error': 'nights and hours-per-night must both be positive',
        }
        payload = {
            'schema': {'name': 'fqcnn_campaign_cost_estimate', 'version': 1},
            'status': 'failed',
            'approval': {
                'approved': False,
                'reasons': ['invalid non-positive budget'],
                'budget_hours': budget_h,
            },
            'request': {
                'datasets': args.datasets, 'configs': args.configs,
                'seeds': args.seeds, 'samples': args.samples,
                'epochs': args.epochs, 'with_baselines': with_baselines,
            },
            'counts': {
                'requested_configs': len(args.configs),
                'measurable_configs': 0,
                'requested_cells': requested_config_cells + requested_baseline_cells,
                'measurable_cells': 0,
                'failed_calibrations': 0,
                'total_failures': 1,
            },
            'calibrations': {
                'configs': {}, 'baseline_seconds_per_proposed_cell': None},
            'failures': [failure],
            'projection': None,
            'reduction': None,
        }
        _write_json(args.output_json, payload)
        print('VERDICT: REJECTED (invalid non-positive budget)')
        return 1

    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 3) - 2)
    with_baselines = not args.no_baselines
    pairs = [tuple(int(c) for c in d.split(',')) for d in args.datasets]

    # Exact split sizes, from the same code path the runner uses.
    print('Measuring split sizes...', flush=True)
    sizes, pair_names = {}, []
    # Estimating must not mint evidence: prepare_split persists a manifest, so
    # send those copies to scratch rather than Results/manifests/.
    real_root = run_experiments.MANIFEST_ROOT
    with tempfile.TemporaryDirectory() as scratch:
        run_experiments.MANIFEST_ROOT = scratch
        try:
            for pair in pairs:
                cfg = build_config(ABLATION_CONFIGS['proposed'], args.seeds[0])
                split, _ = prepare_split(cfg, pair, args.mnist_dir, args.samples)
                name = '{}v{}'.format(*pair)
                sizes[name] = (len(split[0]), len(split[2]), len(split[4]))
                pair_names.append(name)
                print('  {:6s} train={:5d} val={:5d} test={:5d}'.format(name, *sizes[name]))
        finally:
            run_experiments.MANIFEST_ROOT = real_root

    print('\nCalibrating configs (measured on this machine)...', flush=True)
    calibrations, failures = {}, []
    for config_name in args.configs:
        try:
            cal = calibrate(config_name, args.seeds[0], args.epochs,
                            args.reps, args.forward_probe)
        except Exception as exc:
            failures.append({'kind': 'config', 'name': config_name, 'error': repr(exc)})
            print('  {:18s} NOT MEASURABLE: {}'.format(config_name, exc), flush=True)
            continue
        nonfinite = _nonfinite_field(cal)
        if nonfinite is not None:
            error = 'non-finite calibration field: {}'.format(nonfinite)
            failures.append({'kind': 'config', 'name': config_name, 'error': error})
            print('  {:18s} NOT MEASURABLE: {}'.format(config_name, error), flush=True)
            continue
        calibrations[config_name] = cal
        print('  {:18s} n={:2d}  grad/batch {:8.3f}s  fwd/sample {:7.4f}s  '
              'seq-fwd/sample {:7.4f}s  {}'.format(
                  cal['config'], cal['n_qubits'], cal['grad_s_per_batch'],
                  cal['batched_forward_s_per_sample'],
                  cal['sequential_forward_s_per_sample'],
                  'batched' if cal['batched'] else 'SEQUENTIAL (memory cap)'), flush=True)

    baseline_s = 0.0
    baseline_measurable = not with_baselines
    if with_baselines:
        print('\nCalibrating baselines...', flush=True)
        first = pair_names[0]
        try:
            baseline_s = calibrate_baselines(
                args.seeds[0], args.epochs, sizes[first][0], sizes[first][2])
            if not math.isfinite(baseline_s):
                raise ValueError('non-finite baseline calibration')
            baseline_measurable = True
            print('  {:.1f}s attached to each proposed cell'.format(baseline_s))
        except Exception as exc:
            failures.append({'kind': 'baseline', 'name': 'all', 'error': repr(exc)})
            print('  baselines NOT MEASURABLE: {}'.format(exc), flush=True)

    requested_config_cells = len(args.configs) * len(pair_names) * len(args.seeds)
    requested_baseline_cells = (len(pair_names) * len(args.seeds)
                                if with_baselines else 0)
    measurable_config_cells = len(calibrations) * len(pair_names) * len(args.seeds)
    measurable_baseline_cells = (requested_baseline_cells if baseline_measurable else 0)
    budget_h = args.nights * args.hours_per_night
    payload = {
        'schema': {'name': 'fqcnn_campaign_cost_estimate', 'version': 1},
        'status': 'failed' if failures else 'pending',
        'approval': {
            'approved': False,
            'reasons': ['unmeasurable requested cells'] if failures else [],
            'budget_hours': budget_h,
        },
        'request': {
            'datasets': args.datasets,
            'configs': args.configs,
            'seeds': args.seeds,
            'samples': args.samples,
            'epochs': args.epochs,
            'with_baselines': with_baselines,
        },
        'counts': {
            'requested_configs': len(args.configs),
            'measurable_configs': len(calibrations),
            'requested_cells': requested_config_cells + requested_baseline_cells,
            'measurable_cells': measurable_config_cells + measurable_baseline_cells,
            'failed_calibrations': len(failures),
            'total_failures': len(failures),
        },
        'calibrations': {'configs': calibrations,
                         'baseline_seconds_per_proposed_cell': (
                             baseline_s if baseline_measurable else None)},
        'failures': failures,
        'projection': None,
        'reduction': None,
    }
    if failures:
        _write_json(args.output_json, payload)
        print('\nVERDICT: REJECTED ({} requested calibration failure(s))'.format(
            len(failures)))
        return 1

    rows, serial_s, wall_s = project(args.configs, pair_names, args.seeds,
                                     calibrations, sizes, baseline_s, jobs, with_baselines)

    print('\n{:<8s} {:<18s} {:>10s} {:>12s} {:>12s}'.format(
        'dataset', 'config', 'epoch(s)', 'cell(s)', 'x{} seeds(h)'.format(len(args.seeds))))
    print('-' * 64)
    for ds_name, config_name, cost, per_cell in rows:
        print('{:<8s} {:<18s} {:>10.1f} {:>12.1f} {:>12.2f}'.format(
            ds_name, config_name, cost['epoch_s'], per_cell,
            per_cell * len(args.seeds) / 3600.0))

    serial_h, wall_h = serial_s / 3600.0, wall_s / 3600.0
    if not all(math.isfinite(value) for value in (serial_h, wall_h, budget_h)):
        payload['status'] = 'failed'
        payload['approval']['reasons'] = ['non-finite cost projection']
        payload['failures'].append({
            'kind': 'projection', 'name': 'campaign',
            'error': 'projection produced a non-finite value'})
        payload['counts']['total_failures'] += 1
        _write_json(args.output_json, payload)
        print('\nVERDICT: REJECTED (non-finite projection)')
        return 1

    payload['projection'] = {
        'serial_hours': serial_h,
        'wall_hours': wall_h,
        'workers': jobs,
        'budget_hours': budget_h,
        'budget_fraction': wall_h / budget_h,
    }
    print('\n{} cells | serial {:.1f} h | {} worker(s) -> {:.1f} h wall-clock'.format(
        payload['counts']['requested_cells'], serial_h, jobs, wall_h))
    print('budget: {} nights x {:.0f} h = {:.0f} h'.format(
        args.nights, args.hours_per_night, budget_h))

    if wall_h <= budget_h:
        payload['status'] = 'approved'
        payload['approval']['approved'] = True
        payload['approval']['reasons'] = ['complete measurable request fits budget']
        _write_json(args.output_json, payload)
        print('\nVERDICT: FITS ({:.1f} h of {:.0f} h, {:.0f}% of budget, '
              '{:.1f} nights)'.format(
                  wall_h, budget_h, 100.0 * wall_h / budget_h, wall_h / args.hours_per_night))
        return 0

    print('\nVERDICT: OVER BUDGET by {:.1f} h'.format(wall_h - budget_h))
    steps, reduced_h = propose_reduction(
        args.configs, pair_names, args.seeds, calibrations, sizes, baseline_s,
        jobs, with_baselines, budget_h)
    payload['status'] = 'over_budget'
    payload['approval']['reasons'] = ['complete measurable request exceeds budget']
    payload['reduction'] = {
        'steps': steps,
        'projected_wall_hours': reduced_h,
        'fits_budget': reduced_h <= budget_h,
    }
    _write_json(args.output_json, payload)
    print('mandated reduction order (seeds -> datasets -> non-pooling ablations):')
    for step in steps:
        print('  - {}'.format(step))
    print('  => {:.1f} h ({})'.format(
        reduced_h, 'fits' if reduced_h <= budget_h else 'still over; escalate to §18.4 JAX/GPU'))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
