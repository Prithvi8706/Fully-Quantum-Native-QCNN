"""Freeze machinery for the FQCNN headline model (UPGRADE_PLAN.md §A2).

The protected object is the function ``(x, theta) -> <Z_readout>``, not the
incidental layout of the source. Everything here exists to pin that function
down so structural or semantic drift becomes a test failure instead of a silent
change in the paper's numbers.
"""
import hashlib
import json
import os

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN

# Identity of the archived headline run, mirrored from Results/metadata.json.
HEADLINE_IMAGE_SIZE = 28
HEADLINE_ENCODING = 'amplitude'
HEADLINE_SEED = 42
HEADLINE_N_QUBITS = 10
HEADLINE_N_PARAM_SLOTS = 269
HEADLINE_WEIGHTS = os.path.join('Results', 'Weights', 'quantum_model_params.npz')

FIXTURE_DIR = os.path.join('tests', 'fixtures')

# Regression inputs are drawn once, from a fixed seed, and never re-drawn.
N_REGRESSION_INPUTS = 20
REGRESSION_INPUT_SEED = 20260725
REGRESSION_TOL = 1e-10


def build_headline_model() -> PureQuantumNativeCNN:
    """The frozen headline model with freshly seeded (untrained) parameters."""
    cfg = QuantumNativeConfig.from_image_size(HEADLINE_IMAGE_SIZE, HEADLINE_ENCODING)
    cfg.seed = HEADLINE_SEED
    return PureQuantumNativeCNN(cfg)


def load_archived_params(model: PureQuantumNativeCNN) -> pnp.ndarray:
    """Archived trained weights, flattened in the model's own slot order."""
    data = np.load(HEADLINE_WEIGHTS)
    ordered = [pnp.array(data[key], requires_grad=True) for key in model.quantum_params]
    return pnp.concatenate([p.flatten() for p in ordered])


def fixed_regression_inputs() -> np.ndarray:
    """The committed regression inputs: 20 fixed vectors in the encoded length."""
    rng = np.random.default_rng(REGRESSION_INPUT_SEED)
    return rng.random((N_REGRESSION_INPUTS, 2 ** HEADLINE_N_QUBITS))


def slot_ranges(model: PureQuantumNativeCNN) -> dict:
    """Named ``[start, stop)`` ranges of each parameter group in the flat vector."""
    ranges = {}
    cursor = 0
    for name, block in model.quantum_params.items():
        size = int(np.prod(block.shape))
        ranges[name] = [cursor, cursor + size]
        cursor += size
    return ranges


SIGNATURE_FIXTURE = os.path.join(FIXTURE_DIR, 'headline_signature.json')

# State-preparation operations carry data, not trainable slots; their parameters
# are described by length so the signature stays a topology statement.
_STATE_PREP_OPS = ('AmplitudeEmbedding', 'StatePrep', 'MottonenStatePreparation')


def headline_tape(model, x, flat_params):
    """Construct and return the QNode's tape for one (input, parameter) pair."""
    model.quantum_circuit.construct((x, flat_params), {})
    return model.quantum_circuit.tape


def _marker_vector(n_slots: int) -> pnp.ndarray:
    """Parameter vector whose slot ``i`` carries the unique tag ``i + 1``."""
    return pnp.array(np.arange(1, n_slots + 1, dtype=float), requires_grad=True)


def _describe_param(value, lookup) -> str:
    slot = lookup.get(round(float(value), 9))
    if slot is not None:
        return 'slot{}'.format(slot)
    return 'const{:.12g}'.format(float(value))


def circuit_signature(model) -> dict:
    """Serialise the frozen topology: gate name, wires, and parameter slot."""
    n_slots = len(model._flatten_params(model.quantum_params))
    marker = _marker_vector(n_slots)
    lookup = {round(float(i + 1), 9): i for i in range(n_slots)}
    tape = headline_tape(model, fixed_regression_inputs()[0], marker)

    operations = []
    for op in tape.operations:
        if op.name in _STATE_PREP_OPS:
            params = ['data{}'.format(int(np.shape(op.data[0])[0]))]
        else:
            params = [_describe_param(p, lookup) for p in op.data]
        operations.append({
            'name': op.name,
            'wires': [int(w) for w in op.wires],
            'params': params,
        })

    return {
        'operations': operations,
        'measurements': [str(m) for m in tape.measurements],
    }


def signature_hash(signature: dict) -> str:
    blob = json.dumps(signature, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def unitarity_violations(tape) -> list:
    """Operations on the main path that break the A3 unitarity invariant.

    An empty result means the tape is pure state preparation plus unitaries up
    to a single terminal measurement: no mid-circuit measurement, no classical
    feed-forward, no non-unitary channel.
    """
    violations = []
    for op in tape.operations:
        if isinstance(op, qml.measurements.MidMeasureMP):
            violations.append('{}: mid-circuit measurement'.format(op.name))
        elif isinstance(op, qml.ops.op_math.Conditional):
            violations.append('{}: classical feed-forward'.format(op.name))
        elif isinstance(op, qml.operation.Channel):
            violations.append('{}: non-unitary channel'.format(op.name))
    return violations
