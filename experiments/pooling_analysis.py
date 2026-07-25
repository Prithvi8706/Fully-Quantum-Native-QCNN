#!/usr/bin/env python3
"""E1 -- exact equivalence of unitary and measurement-based pooling.

Theorem 1 (UPGRADE_PLAN.md 2.1): the frozen pooling block's controls are
diagonal in the discarded qubit's basis, so

    V = |0><0|_b (x) U_0 + |1><1|_b (x) U_1,
    U_0 = RY(gamma),  U_1 = RY(gamma) . RZ(beta) . RY(alpha)

and by deferred measurement Tr_b[V rho V^dag] = sum_m U_m <m|rho|m> U_m^dag. So
the retained register -- and every observable on it, including the terminal
<Z_readout> -- is *identical* under coherent unitary pooling and under
measure-and-condition pooling. The predicted difference is exactly zero.

E1 confirms that prediction numerically, at fixed parameters, on default.mixed.
The comparison arm is the explicit Kraus channel rather than mid-circuit
measurement, because a simulator may legitimately implement mid-circuit
measurement *by* deferring it back into the very controlled gates under test --
which would make the tie a tautology about the simulator instead of a result
about the circuit. The channel is genuinely non-unitary, so the tie is real.

Usage:
  python -m experiments.pooling_analysis            # headline n=10
  python -m experiments.pooling_analysis --n-inputs 5 --image-size 16
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pennylane as qml

from QCNN import circuits, freeze
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.encoding import PureQuantumEncoder
from QCNN.models.QCNNModel import PureQuantumNativeCNN

# Theorem 1 predicts exact equality; this is the numerical-noise allowance for a
# density-matrix simulation, and the threshold UPGRADE_PLAN.md 2.3 sets for E1.
E1_TOLERANCE = 1e-12

EVIDENCE_PATH = os.path.join('Results', 'evidence', 'e1_pooling_equivalence.json')


def _config(image_size: int, pooling_mode: str) -> QuantumNativeConfig:
    cfg = QuantumNativeConfig.from_image_size(image_size, freeze.HEADLINE_ENCODING)
    cfg.seed = freeze.HEADLINE_SEED
    cfg.pooling_mode = pooling_mode
    return cfg


def readouts(cfg, params, inputs, device: str = 'default.mixed') -> np.ndarray:
    """``<Z_readout>`` for each input, on a mixed-state device."""
    dev = qml.device(device, wires=cfg.n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        return circuits.build_circuit(x, params, cfg)

    return np.array([float(circuit(x)) for x in inputs])


def run_e1(image_size: int = freeze.HEADLINE_IMAGE_SIZE, n_inputs: int = 8,
           use_archived_weights: bool = True) -> dict:
    """Compare the frozen unitary block against measure-and-condition pooling."""
    cfg_unitary = _config(image_size, 'unitary')
    cfg_channel = _config(image_size, 'measurement_channel')

    model = PureQuantumNativeCNN(cfg_unitary)
    if use_archived_weights and cfg_unitary.n_qubits == freeze.HEADLINE_N_QUBITS:
        flat = freeze.load_archived_params(model)
        weights = 'archived headline weights'
    else:
        flat = model._flatten_params(model.quantum_params)
        weights = 'seeded initial weights'
    params = model._unflatten_params(flat)

    rng = np.random.default_rng(freeze.REGRESSION_INPUT_SEED)
    inputs = PureQuantumEncoder.precompute_amplitudes(
        rng.random((n_inputs, 2 ** cfg_unitary.n_qubits)), cfg_unitary.n_qubits)

    unitary = readouts(cfg_unitary, params, inputs)
    channel = readouts(cfg_channel, params, inputs)
    diff = np.abs(unitary - channel)

    return {
        'experiment': 'E1',
        'claim': 'unitary pooling exactly reproduces measure-and-condition pooling',
        'n_qubits': int(cfg_unitary.n_qubits),
        'image_size': int(image_size),
        'weights': weights,
        'device': 'default.mixed',
        'n_inputs': int(n_inputs),
        'tolerance': E1_TOLERANCE,
        'max_abs_difference': float(diff.max()),
        'mean_abs_difference': float(diff.mean()),
        'passes': bool(diff.max() <= E1_TOLERANCE),
        'unitary_readouts': [float(v) for v in unitary],
        'measurement_readouts': [float(v) for v in channel],
        'abs_differences': [float(v) for v in diff],
    }


# ---------------------------------------------------------------------------
# E2 -- fixed-parameter dephasing (Proposition 3)
#
# Delta_coh(block) = || rho_a^V - rho_a^{deph_b -> V} ||_1, the trace-distance
# effect of fully dephasing the discarded qubit immediately before pooling, at
# identical parameters. Theorem 1 implies Delta_coh == 0 for the frozen block:
# its controls read only b's populations, so erasing b's coherences changes
# nothing on the retained register. That is an a-priori prediction, made before
# the measurement, which is what makes confirming it worth reporting.
# ---------------------------------------------------------------------------

def _dephase(wires):
    """Full computational-basis dephasing: rho -> (rho + Z rho Z)/2 = diag(rho)."""
    for wire in wires:
        qml.PhaseFlip(0.5, wires=wire)


def _hooks_dephasing(target: str) -> circuits.CircuitHooks:
    """Dephase the discarded wires ('discard') or, as a control, the kept ones."""
    if target == 'discard':
        return circuits.CircuitHooks(before_pool=lambda keep, discard: _dephase(discard))
    if target == 'keep':
        return circuits.CircuitHooks(before_pool=lambda keep, discard: _dephase(keep))
    raise ValueError("target must be 'discard' or 'keep'")


def _readouts_with_hooks(cfg, params, inputs, hooks, device='default.mixed'):
    dev = qml.device(device, wires=cfg.n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        return circuits.build_circuit(x, params, cfg, hooks=hooks)

    return np.array([float(circuit(x)) for x in inputs])


def _readout_states(cfg, params, inputs, hooks, device='default.mixed'):
    """Reduced density matrix of the readout qubit, for the Prop 3 functional."""
    base = hooks or circuits.CircuitHooks()
    probe = circuits.CircuitHooks(
        before_pool=base.before_pool,
        terminal=lambda readout, active: qml.density_matrix(wires=readout))
    dev = qml.device(device, wires=cfg.n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        return circuits.build_circuit(x, params, cfg, hooks=probe)

    return [np.asarray(circuit(x)) for x in inputs]


def _trace_distance(rho, sigma) -> float:
    """(1/2) * sum of singular values of (rho - sigma)."""
    return float(0.5 * np.abs(np.linalg.eigvalsh(rho - sigma)).sum())


def run_e2(image_size: int = freeze.HEADLINE_IMAGE_SIZE, n_inputs: int = 8,
           use_archived_weights: bool = True) -> dict:
    """Dephase discarded wires before pooling; the frozen block must not notice."""
    cfg = _config(image_size, 'unitary')

    model = PureQuantumNativeCNN(cfg)
    if use_archived_weights and cfg.n_qubits == freeze.HEADLINE_N_QUBITS:
        flat = freeze.load_archived_params(model)
        weights = 'archived headline weights'
    else:
        flat = model._flatten_params(model.quantum_params)
        weights = 'seeded initial weights'
    params = model._unflatten_params(flat)

    rng = np.random.default_rng(freeze.REGRESSION_INPUT_SEED)
    inputs = PureQuantumEncoder.precompute_amplitudes(
        rng.random((n_inputs, 2 ** cfg.n_qubits)), cfg.n_qubits)

    clean = _readouts_with_hooks(cfg, params, inputs, None)
    dephased = _readouts_with_hooks(cfg, params, inputs, _hooks_dephasing('discard'))
    control = _readouts_with_hooks(cfg, params, inputs, _hooks_dephasing('keep'))

    delta = np.abs(clean - dephased)
    control_delta = np.abs(clean - control)

    rho_clean = _readout_states(cfg, params, inputs, None)
    rho_dephased = _readout_states(cfg, params, inputs, _hooks_dephasing('discard'))
    coh = [_trace_distance(a, b) for a, b in zip(rho_clean, rho_dephased)]

    # The operational statement: the decision for every input is unchanged, so
    # accuracy is unchanged against any labelling whatsoever.
    same_decision = bool(np.all(np.sign(clean) == np.sign(dephased)))

    return {
        'experiment': 'E2',
        'claim': 'dephasing the discarded wires before pooling has no operational effect',
        'proposition': 'Prop 3: Delta_coh == 0 for the frozen block',
        'n_qubits': int(cfg.n_qubits),
        'image_size': int(image_size),
        'weights': weights,
        'device': 'default.mixed',
        'n_inputs': int(n_inputs),
        'tolerance': E1_TOLERANCE,
        'max_abs_readout_delta': float(delta.max()),
        'max_trace_distance': float(max(coh)),
        'decisions_unchanged': same_decision,
        'delta_accuracy': 0.0 if same_decision else None,
        'control_dephase_kept_max_delta': float(control_delta.max()),
        'passes': bool(delta.max() <= E1_TOLERANCE
                       and max(coh) <= E1_TOLERANCE
                       and same_decision),
        'control_is_live': bool(control_delta.max() > 1e-6),
        'clean_readouts': [float(v) for v in clean],
        'dephased_readouts': [float(v) for v in dephased],
        'abs_readout_deltas': [float(v) for v in delta],
        'trace_distances': [float(v) for v in coh],
    }


def _report_e2(result: dict) -> None:
    print('E2 -- dephasing the discarded wires before pooling (Prop 3)')
    print('  n_qubits           : {}'.format(result['n_qubits']))
    print('  weights            : {}'.format(result['weights']))
    print('  inputs             : {}'.format(result['n_inputs']))
    print()
    print('  {:>3s}  {:>22s}  {:>22s}  {:>12s}  {:>12s}'.format(
        'i', 'clean <Z>', 'dephased <Z>', '|delta|', 'trace dist'))
    for i, (c, d, delta, td) in enumerate(zip(result['clean_readouts'],
                                              result['dephased_readouts'],
                                              result['abs_readout_deltas'],
                                              result['trace_distances'])):
        print('  {:>3d}  {:>22.16f}  {:>22.16f}  {:>12.3e}  {:>12.3e}'.format(
            i, c, d, delta, td))
    print()
    print('  max |readout delta|: {:.3e}'.format(result['max_abs_readout_delta']))
    print('  max trace distance : {:.3e}   (Prop 3 predicts 0)'.format(
        result['max_trace_distance']))
    print('  decisions unchanged: {}  -> delta accuracy = {}'.format(
        result['decisions_unchanged'], result['delta_accuracy']))
    print('  control (dephase kept wires instead): {:.3e}  {}'.format(
        result['control_dephase_kept_max_delta'],
        'channel is live' if result['control_is_live'] else 'CONTROL FAILED'))
    print('  tolerance          : {:.0e}'.format(result['tolerance']))
    print('  VERDICT            : {}'.format('PASS' if result['passes'] else 'FAIL'))


def main():
    ap = argparse.ArgumentParser(description='E1/E2 pooling validation')
    ap.add_argument('--experiment', choices=['e1', 'e2', 'both'], default='both')
    ap.add_argument('--image-size', type=int, default=freeze.HEADLINE_IMAGE_SIZE)
    ap.add_argument('--n-inputs', type=int, default=8)
    ap.add_argument('--out', default=EVIDENCE_PATH)
    args = ap.parse_args()

    if args.experiment == 'e2':
        result = run_e2(image_size=args.image_size, n_inputs=args.n_inputs)
        _report_e2(result)
        out = args.out.replace('e1_pooling_equivalence', 'e2_dephasing')
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print('\n  evidence -> {}'.format(out))
        return 0 if result['passes'] else 1

    result = run_e1(image_size=args.image_size, n_inputs=args.n_inputs)

    print('E1 -- unitary vs measure-and-condition pooling')
    print('  n_qubits           : {}'.format(result['n_qubits']))
    print('  weights            : {}'.format(result['weights']))
    print('  device             : {}'.format(result['device']))
    print('  inputs             : {}'.format(result['n_inputs']))
    print()
    print('  {:>3s}  {:>22s}  {:>22s}  {:>12s}'.format(
        'i', 'unitary <Z>', 'measurement <Z>', '|diff|'))
    for i, (u, m, d) in enumerate(zip(result['unitary_readouts'],
                                      result['measurement_readouts'],
                                      result['abs_differences'])):
        print('  {:>3d}  {:>22.16f}  {:>22.16f}  {:>12.3e}'.format(i, u, m, d))
    print()
    print('  max |difference|   : {:.3e}'.format(result['max_abs_difference']))
    print('  tolerance          : {:.0e}'.format(result['tolerance']))
    print('  VERDICT            : {}'.format('PASS' if result['passes'] else 'FAIL'))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print('\n  evidence -> {}'.format(args.out))
    ok = result['passes']

    if args.experiment == 'both':
        print()
        e2 = run_e2(image_size=args.image_size, n_inputs=args.n_inputs)
        _report_e2(e2)
        out = args.out.replace('e1_pooling_equivalence', 'e2_dephasing')
        with open(out, 'w') as fh:
            json.dump(e2, fh, indent=2, sort_keys=True)
        print('\n  evidence -> {}'.format(out))
        ok = ok and e2['passes']

    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
