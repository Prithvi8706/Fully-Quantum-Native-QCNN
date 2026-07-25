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


def main():
    ap = argparse.ArgumentParser(description='E1 pooling-equivalence validation')
    ap.add_argument('--image-size', type=int, default=freeze.HEADLINE_IMAGE_SIZE)
    ap.add_argument('--n-inputs', type=int, default=8)
    ap.add_argument('--out', default=EVIDENCE_PATH)
    args = ap.parse_args()

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

    return 0 if result['passes'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
