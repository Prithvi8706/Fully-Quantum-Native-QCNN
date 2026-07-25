import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from QCNN import circuits
from QCNN.config import QuantumNativeConfig
from QCNN.layers import QuantumNativeConvolution
from QCNN.layers import QuanvolutionalLayer


# Device for the batched training path (UPGRADE_PLAN.md 1.1). default.qubit
# broadcasts AmplitudeEmbedding over a leading batch axis and supports backprop,
# so one statevector pass differentiates a whole batch. lightning.qubit does
# neither, which is why the sequential path below remains the correctness oracle
# (tests/test_batched_execution.py pins the two together).
BATCHED_DEVICE = 'default.qubit'


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
        self.device = qml.device(config.device, wires=config.n_qubits)
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

        # FIX: removed @qml.transforms.broadcast_expand — not compatible with
        # AmplitudeEmbedding / MottonenStatePreparation in PennyLane 0.38
        @qml.qnode(self.device, interface='autograd', diff_method='best')
        def quantum_circuit(x, flat_params):
            params = self._unflatten_params(flat_params)
            return self._pure_quantum_forward(x, params)

        self.quantum_circuit = quantum_circuit

        # Same circuit, batched execution strategy. Accepts a (batch, features)
        # input and returns one <Z> per sample; also accepts a single 1D input.
        self.batched_device = qml.device(BATCHED_DEVICE, wires=config.n_qubits)

        @qml.qnode(self.batched_device, interface='autograd', diff_method='backprop')
        def batched_circuit(x, flat_params):
            params = self._unflatten_params(flat_params)
            return self._pure_quantum_forward(x, params)

        self.batched_circuit = batched_circuit
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
        return circuits.build_circuit(x, params, self.config)

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
        outputs = np.asarray(
            self.batched_circuit(np.asarray(X_processed), flat_params),
            dtype=float).reshape(-1)
        return np.where(outputs > 0, 1, -1)