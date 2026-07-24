"""Freeze machinery for the FQCNN headline model (UPGRADE_PLAN.md §A2).

The protected object is the function ``(x, theta) -> <Z_readout>``, not the
incidental layout of the source. Everything here exists to pin that function
down so structural or semantic drift becomes a test failure instead of a silent
change in the paper's numbers.
"""
import os

import numpy as np
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
