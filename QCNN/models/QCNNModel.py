import math
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from QCNN.config import QuantumNativeConfig
from QCNN.layers import QuantumNativeConvolution
from QCNN.layers import QuantumNativePooling
from QCNN.layers import QuanvolutionalLayer
from QCNN.encoding import PureQuantumEncoder


class PureQuantumNativeCNN:
    """
    Fully quantum-native convolutional neural network.
    Implements encoding, convolution, pooling, and classification
    entirely using unitary quantum operations with shared kernels
    and differentiable PennyLane QNodes.
    """

    #constructor
    def __init__(self, config: QuantumNativeConfig):
        self.config = config
        # Fast path: default.qubit + backprop natively broadcasts a whole batch through a
        # single statevector pass. Fallback: the configured device (lightning.qubit) with
        # adjoint, evaluated one sample at a time.
        # Batched backprop OOMs at high qubit counts (memory ~ batch × 2^n_qubits × depth),
        # so gate the fast path by qubit count; larger circuits use the memory-light
        # lightning+adjoint per-sample path instead.
        _want_fast = getattr(config, 'fast_backprop', False)
        _max_q = getattr(config, 'fast_backprop_max_qubits', 12)
        self.fast_backprop = _want_fast and config.n_qubits <= _max_q
        if _want_fast and not self.fast_backprop:
            print(f"[QCNNModel] {config.n_qubits} qubits > {_max_q}: using memory-safe "
                  f"lightning+adjoint per-sample path (batched backprop would OOM).")
        device_name = 'default.qubit' if self.fast_backprop else config.device
        self.device = qml.device(device_name, wires=config.n_qubits)
        self.num_qubits = config.n_qubits

        self.quanv_layer = None
        if config.encoding_type == 'patch':
            self.quanv_layer = QuanvolutionalLayer(
                patch_size=config.patch_size,
                n_filters=config.n_quanv_filters,
                stride=config.patch_stride,
                device_name=config.device,
                random_params=True
            )

        self.quantum_params = self._initialize_quantum_parameters()

        # default.qubit supports backprop + input broadcasting, so a batched x (shape
        # (batch, features)) returns one <Z> per sample in a single pass. lightning.qubit
        # uses adjoint ('best') and is driven one sample at a time by the callers.
        diff_method = 'backprop' if self.fast_backprop else 'best'
        @qml.qnode(self.device, interface='autograd', diff_method=diff_method)
        def quantum_circuit(x, flat_params):
            params = self._unflatten_params(flat_params)
            return self._pure_quantum_forward(x, params)

        self.quantum_circuit = quantum_circuit
        self.training_history = {'loss': [], 'accuracy': [], 'epoch_times': []}

    def _initialize_quantum_parameters(self) -> dict[str, pnp.ndarray]:
        seed = getattr(self.config, 'seed', 42)
        np.random.seed(seed)
        pnp.random.seed(seed)
        params = {}
        init_range = np.pi / 4
        conv_depth = 4
        rpq = QuantumNativeConvolution.rotations_per_qubit(
            getattr(self.config, 'kernel_rotations', 'su2'))
        kernel_shape = (4, conv_depth, rpq)

        for layer in range(self.config.n_conv_layers):
            kernel_tensor = pnp.array(
                np.random.uniform(-init_range, init_range, kernel_shape),
                requires_grad=True
            )
            params[f'quantum_conv_kernel_{layer}'] = kernel_tensor

        max_pairs = self.num_qubits // 2
        pool_angles_per_layer = 3 * max_pairs
        n_pool_layers = max(1, self.config.n_conv_layers - 1)
        for pl in range(n_pool_layers):
            params[f'quantum_pooling_{pl}'] = pnp.array(
                np.random.uniform(-init_range, init_range, pool_angles_per_layer),
                requires_grad=True
            )

        params['quantum_classifier'] = pnp.array(
            np.random.uniform(-init_range, init_range, 32),
            requires_grad=True
        )
        return params

    def _flatten_params(self, params: dict[str, pnp.ndarray]) -> pnp.ndarray:
        return pnp.concatenate([p.flatten() for p in params.values()])

    def _unflatten_params(self, flat_params: pnp.ndarray) -> dict[str, pnp.ndarray]:
        params = {}
        idx = 0
        conv_depth = 4
        rpq = QuantumNativeConvolution.rotations_per_qubit(
            getattr(self.config, 'kernel_rotations', 'su2'))
        kernel_size = 4 * conv_depth * rpq
        for layer in range(self.config.n_conv_layers):
            slice_flat = flat_params[idx:idx + kernel_size]
            params[f'quantum_conv_kernel_{layer}'] = slice_flat.reshape(4, conv_depth, rpq)
            idx += kernel_size

        max_pairs = self.num_qubits // 2
        pool_angles_per_layer = 3 * max_pairs
        n_pool_layers = max(1, self.config.n_conv_layers - 1)
        for pl in range(n_pool_layers):
            params[f'quantum_pooling_{pl}'] = flat_params[idx:idx + pool_angles_per_layer]
            idx += pool_angles_per_layer

        params['quantum_classifier'] = flat_params[idx:idx + 32]
        idx += 32
        return params

    def _pure_quantum_forward(self, x: np.ndarray, params: dict) -> float:
        all_qubits = list(range(self.config.n_qubits))

        if self.config.encoding_type in ('amplitude', 'patch'):
            PureQuantumEncoder.amplitude_encoding(x, all_qubits)
        else:
            PureQuantumEncoder.quantum_feature_map(x, all_qubits)

        active_qubits = all_qubits.copy()
        current_image_size = self.config.image_size

        for layer in range(self.config.n_conv_layers):
            n_current = len(active_qubits)
            if n_current >= 4:
                width = int(math.sqrt(n_current))
                while n_current % width != 0:
                    width -= 1
                height = n_current // width
                w, h = max(width, height), min(width, height)
                base_windows = QuantumNativeConvolution.get_conv_windows(w, h)
                kernel = params[f'quantum_conv_kernel_{layer}']
                rotations = getattr(self.config, 'kernel_rotations', 'su2')
                entanglement = getattr(self.config, 'conv_entanglement', 'full')
                for rel_window in base_windows:
                    if max(rel_window) < n_current:
                        window_qubits = [active_qubits[i] for i in rel_window]
                        QuantumNativeConvolution.quantum_conv2d_kernel(
                            kernel, window_qubits,
                            rotations=rotations, entanglement=entanglement)

            if layer < self.config.n_conv_layers - 1:
                n_qubits_current = len(active_qubits)
                if n_qubits_current < 2:
                    break
                pairs = QuantumNativePooling.make_pairing(active_qubits)
                if len(pairs) == 0:
                    break
                keep = [k for (k, _) in pairs]
                discard = [d for (_, d) in pairs]
                pool_key = f'quantum_pooling_{layer}'
                QuantumNativePooling.apply_pooling(
                    getattr(self.config, 'pooling_mode', 'unitary'),
                    params[pool_key],
                    input_qubits=keep,
                    output_qubits=discard
                )
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
            qml.CNOT(wires=[active_qubits[i], active_qubits[i+1]])
        if n_active >= 2:
            qml.CNOT(wires=[active_qubits[n_active-1], active_qubits[0]])

        for i, q in enumerate(active_qubits[:min(n_active, 4)]):
            qml.RX(classifier_params[(i * 2 + 16) % 32], wires=q)
            qml.RY(classifier_params[(i * 2 + 17) % 32], wires=q)

        if n_active >= 2:
            qml.CNOT(wires=[active_qubits[0], active_qubits[min(n_active-1, 1)]])
        qml.RZ(classifier_params[31], wires=readout)

        return qml.expval(qml.PauliZ(readout))

    def _preprocess_input(self, x: np.ndarray) -> np.ndarray:
        if self.config.encoding_type == 'patch' and self.quanv_layer is not None:
            is_batched = False
            if x.ndim == 3:
                is_batched = True
            elif x.ndim == 2:
                h, w = x.shape
                if h != self.config.image_size or w != self.config.image_size:
                    is_batched = True
            if is_batched:
                return self.quanv_layer.process_batch(x, image_size=self.config.image_size)
            return self.quanv_layer.process_image(x)

        elif self.config.encoding_type == 'amplitude':
            if x.ndim > 2:
                x = x.reshape(x.shape[0], -1)
            target_len = 2 ** self.num_qubits
            is_batched = x.ndim == 2
            feat_len = x.shape[1] if is_batched else len(x)
            if feat_len < target_len:
                if is_batched:
                    padded = np.zeros((x.shape[0], target_len))
                    padded[:, :feat_len] = x
                    return padded
                else:
                    padded = np.zeros(target_len)
                    padded[:feat_len] = x
                    return padded
            return x[:, :target_len] if is_batched else x[:target_len]
        else:
            return x

    def quantum_predict_single(self, x: np.ndarray) -> float:
        x_processed = self._preprocess_input(np.array([x]))[0]
        flat_params = self._flatten_params(self.quantum_params)
        return float(self.quantum_circuit(x_processed, flat_params))

    def quantum_predict_batch(self, X: np.ndarray) -> np.ndarray:
        X_processed = self._preprocess_input(X)
        flat_params = self._flatten_params(self.quantum_params)
        if self.fast_backprop:
            # Single broadcasted pass → vector of <Z>, one per sample.
            outputs = np.asarray(self.quantum_circuit(pnp.array(X_processed), flat_params))
        else:
            outputs = np.array([
                float(self.quantum_circuit(X_processed[i], flat_params))
                for i in range(len(X_processed))
            ])
        return np.where(outputs > 0, 1, -1)

    def quantum_loss_function(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        X_processed = self._preprocess_input(X_batch)
        flat_params = self._flatten_params(self.quantum_params)
        if self.fast_backprop:
            preds = self.quantum_circuit(pnp.array(X_processed), flat_params)
        else:
            preds = pnp.array([
                self.quantum_circuit(pnp.array(X_processed[i]), flat_params)
                for i in range(len(X_processed))
            ])
        y_batch = pnp.array(y_batch)
        return pnp.mean((preds - y_batch) ** 2)