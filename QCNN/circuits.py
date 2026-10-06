"""The canonical frozen FQCNN circuit (UPGRADE_PLAN.md 0.4).

This is the single topology consumed by training, inference, noise simulation,
hardware transpilation, drawing, and resource counting. It is frozen under A1:
no gate may be added, removed, reordered, or reparameterised here.

``CircuitHooks`` are evaluation-layer injection points used by noise and probe
studies. With ``hooks=None`` -- always the case for the headline model -- the
emitted operations are exactly the frozen circuit, which is what the committed
expectation fixtures verify.
"""
import math

import pennylane as qml

from QCNN.encoding import PureQuantumEncoder
from QCNN.layers import QuantumNativeConvolution
from QCNN.layers import QuantumNativePooling


class CircuitHooks:
    """Optional evaluation-only callbacks. Never part of the model definition.

    ``before_pool`` and ``terminal`` are the dephasing and state-observation
    hooks roadmap M2.1 requires: E2 needs to dephase discarded wires immediately
    before pooling, and Proposition 3 is a statement about the retained
    register's reduced state rather than about one observable of it. Both are
    ``None`` on the headline path, so the frozen circuit is unaffected (A3
    permits channels in evaluation-time studies only).
    """

    def __init__(self, after_encoding=None, after_conv_window=None, after_pool=None,
                 after_classifier=None, cnot=None, before_readout=None,
                 before_pool=None, terminal=None):
        self.after_encoding = after_encoding
        self.after_conv_window = after_conv_window
        self.before_pool = before_pool
        self.after_pool = after_pool
        self.after_classifier = after_classifier
        self.cnot = cnot
        self.before_readout = before_readout
        self.terminal = terminal


_NO_HOOKS = CircuitHooks()


def build_circuit(x, params, cfg, hooks=None):
    """Emit the frozen circuit and return its terminal expectation value.

    Args:
        x: encoded input (amplitude vector, or raw features for feature maps).
        params: unflattened parameter dict keyed as in ``QCNNModel.quantum_params``.
        cfg: a ``QuantumNativeConfig``.
        hooks: optional ``CircuitHooks``; ``None`` gives the frozen path.
    """
    hooks = hooks or _NO_HOOKS
    emit_cnot = hooks.cnot or (lambda a, b: qml.CNOT(wires=[a, b]))

    all_qubits = list(range(cfg.n_qubits))

    if cfg.encoding_type in ('amplitude', 'patch'):
        PureQuantumEncoder.amplitude_encoding(x, all_qubits)
    else:
        PureQuantumEncoder.quantum_feature_map(x, all_qubits)
    if hooks.after_encoding:
        hooks.after_encoding(all_qubits)

    active_qubits = all_qubits.copy()
    current_image_size = cfg.image_size

    for layer in range(cfg.n_conv_layers):
        n_current = len(active_qubits)
        if n_current >= 4:
            width = int(math.sqrt(n_current))
            while n_current % width != 0:
                width -= 1
            height = n_current // width
            w, h = max(width, height), min(width, height)
            base_windows = QuantumNativeConvolution.get_conv_windows(w, h)
            kernel = params[f'quantum_conv_kernel_{layer}']
            rotations = getattr(cfg, 'kernel_rotations', 'su2')
            entanglement = getattr(cfg, 'conv_entanglement', 'full')
            for rel_window in base_windows:
                if max(rel_window) < n_current:
                    window_qubits = [active_qubits[i] for i in rel_window]
                    QuantumNativeConvolution.quantum_conv2d_kernel(
                        kernel, window_qubits,
                        rotations=rotations, entanglement=entanglement)
                    if hooks.after_conv_window:
                        hooks.after_conv_window(window_qubits)

        if layer < cfg.n_conv_layers - 1:
            n_qubits_current = len(active_qubits)
            if n_qubits_current < 2:
                break
            pairs = QuantumNativePooling.make_pairing(active_qubits)
            if len(pairs) == 0:
                break
            keep = [k for (k, _) in pairs]
            discard = [d for (_, d) in pairs]
            pool_key = f'quantum_pooling_{layer}'
            if hooks.before_pool:
                hooks.before_pool(keep, discard)
            QuantumNativePooling.apply_pooling(
                getattr(cfg, 'pooling_mode', 'unitary'),
                params[pool_key],
                input_qubits=keep,
                output_qubits=discard
            )
            if hooks.after_pool:
                hooks.after_pool(keep, discard)
            active_qubits = keep
            if current_image_size > 2:
                current_image_size = max(2, current_image_size // 2)

    classifier_params = params['quantum_classifier']
    n_active = len(active_qubits)
    readout = active_qubits[0]

    for i, q in enumerate(active_qubits[:min(n_active, 4)]):
        qml.RX(classifier_params[i * 2 % 32], wires=q)
        qml.RY(classifier_params[(i * 2 + 1) % 32], wires=q)
        qml.RZ(classifier_params[(i * 2 + 8) % 32], wires=q)

    for i in range(n_active - 1):
        emit_cnot(active_qubits[i], active_qubits[i + 1])
    if n_active >= 2:
        emit_cnot(active_qubits[n_active - 1], active_qubits[0])

    for i, q in enumerate(active_qubits[:min(n_active, 4)]):
        qml.RX(classifier_params[(i * 2 + 16) % 32], wires=q)
        qml.RY(classifier_params[(i * 2 + 17) % 32], wires=q)

    if n_active >= 2:
        emit_cnot(active_qubits[0], active_qubits[min(n_active - 1, 1)])
    qml.RZ(classifier_params[31], wires=readout)

    if hooks.after_classifier:
        hooks.after_classifier(active_qubits)
    if hooks.before_readout:
        hooks.before_readout(readout)

    if hooks.terminal:
        return hooks.terminal(readout, active_qubits)
    return qml.expval(qml.PauliZ(readout))
