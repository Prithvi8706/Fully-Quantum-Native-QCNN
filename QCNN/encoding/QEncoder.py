import hashlib

import pennylane as qml
import pennylane.numpy as pnp
import numpy as np


class PureQuantumEncoder:
    """
    Quantum-native encoder implementing amplitude encoding and
    rotation-based quantum feature maps. Provides purely unitary
    data embedding for QCNN pipelines without classical preprocessing.
    """

    @staticmethod
    def amplitude_encoding(data: np.ndarray, wires: list[int]) -> None:
        """
        Pure quantum amplitude encoding with batch support.
        Maps classical vectors into the amplitudes of an n-qubit
        quantum state, padding or truncating as required. 
        
        Args:
            data: 1D array (features,) or 2D array (batch_size, features)
            wires: list of qubit indices for the embedding
        """
        data = pnp.asarray(data)
        
        # Flatten spatial dimensions if they exist (e.g., [batch, h, w] -> [batch, h*w])
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        elif data.ndim == 2:
            # Check if it's a batch of flattened vectors or a single image
            # In QCNN context, we often deal with square images.
            # If it's a batch, it should be (batch_size, features).
            # If it's a single image, it's (h, w).
            # However, AmplitudeEmbedding expects 1D or 2D.
            # If it's 2D and we want it to be a single sample, we must flatten it to 1D.
            # But wait, if it's 2D and it's a batch, we leave it.
            # The calling code should have already decided if it's a batch.
            # To be safe, if the caller didn't flatten the single image (h, w), we should.
            pass
            
        is_batched = data.ndim == 2
        max_size = 2 ** len(wires)
        
        # Determine current feature size
        feat_size = data.shape[1] if is_batched else data.shape[0]
        
        if feat_size > max_size:
            if is_batched:
                data = data[:, :max_size]
            else:
                data = data[:max_size]
        elif feat_size < max_size:
            if is_batched:
                padded = pnp.zeros((data.shape[0], max_size))
                padded[:, :feat_size] = data
                data = padded
            else:
                padded = pnp.zeros(max_size)
                padded[:feat_size] = data
                data = padded
        
        # Normalize for quantum amplitudes (must sum to 1)
        eps = 1e-15
        if is_batched:
            norms = pnp.linalg.norm(data, axis=1, keepdims=True)
            # Find indices of all-zero vectors in batch
            zero_masks = (norms < eps).flatten()
            if any(zero_masks):
                # Replace zero vectors with a valid basis state (first element = 1)
                # to prevent NaN gradients and valid state errors
                data[zero_masks, 0] = 1.0
                norms = pnp.linalg.norm(data, axis=1, keepdims=True)
            data_norm = data / norms
        else:
            norm = pnp.linalg.norm(data)
            if norm < eps:
                # Replace with first basis state
                data = pnp.zeros_like(data)
                data[0] = 1.0
                norm = 1.0
            data_norm = data / norm
        
        # Pure quantum encoding - natively supports batched input
        qml.AmplitudeEmbedding(features=data_norm, wires=wires, normalize=True)
    
    @staticmethod
    def precompute_amplitudes(data: np.ndarray, n_qubits: int) -> np.ndarray:
        """Pad/truncate and L2-normalise once, outside the circuit (UPGRADE_PLAN.md 1.2).

        Amplitude vectors do not depend on the parameters, so this work does not
        belong in the per-batch training loop. The result is exactly the array
        ``amplitude_encoding`` would hand to ``AmplitudeEmbedding``, so feeding it
        back through the encoder is numerically an identity and the emitted
        state-preparation op -- hence the frozen fingerprint -- is unchanged.

        Args:
            data: 1D (features,), 2D (batch, features), or higher (flattened).
            n_qubits: encoded length is ``2 ** n_qubits``.

        Returns:
            float64 array, L2-normalised along the feature axis. 1D in, 1D out.
        """
        x = np.asarray(data, dtype=float)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        single = x.ndim == 1
        if single:
            x = x[None, :]

        max_size = 2 ** n_qubits
        feat_size = x.shape[1]
        if feat_size > max_size:
            x = x[:, :max_size].copy()
        elif feat_size < max_size:
            padded = np.zeros((x.shape[0], max_size))
            padded[:, :feat_size] = x
            x = padded
        else:
            x = x.copy()

        # Same all-zero fallback as amplitude_encoding: a zero vector is not a
        # valid state, so it becomes |0...0> rather than a NaN gradient.
        eps = 1e-15
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        degenerate = (norms < eps).ravel()
        if degenerate.any():
            x[degenerate] = 0.0
            x[degenerate, 0] = 1.0
            norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms

        return x[0] if single else x

    @staticmethod
    def amplitude_identity(encoded: np.ndarray) -> dict:
        """Content hash of a precomputed amplitude cache, for run metadata.

        UPGRADE_PLAN.md 1.2 requires the cache identity to be recorded, so a run
        can be traced to the exact vectors that entered the circuit.
        """
        arr = np.ascontiguousarray(np.asarray(encoded, dtype=float))
        return {
            'sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
            'shape': [int(d) for d in arr.shape],
            'dtype': str(arr.dtype),
        }

    @staticmethod
    def quantum_feature_map(data: np.ndarray, wires: list[int]) -> None:
        """
        Rotation-based entangling quantum feature map.
        Supports batched inputs.
        
        Args:
            data: 1D array (features,) or 2D array (batch_size, features)
            wires: list of qubit indices used for the feature map
        """
        data = pnp.asarray(data, dtype=float)
        is_batched = data.ndim == 2
        feat_size = data.shape[1] if is_batched else data.shape[0]
        
        L = min(feat_size, len(wires))
        
        if is_batched:
            x = data[:, :L]
        else:
            x = data[:L]
            
        if x.size > 0:
            x = pnp.clip(x, 0.0, 1.0)  # assume [0, 1] normalized input

        angles_z = pnp.pi * x           # RZ(π x) ∈ [0, π]
        angles_y = pnp.pi * x           # RY(π x) ∈ [0, π]

        # Layer 1: Individual qubit rotations
        for i in range(L):
            qml.Hadamard(wires=wires[i])
            angle_z_i = angles_z[:, i] if is_batched else angles_z[i]
            qml.RZ(angle_z_i, wires=wires[i])

        # Layer 2: Quantum entangling interactions
        for i in range(L - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            if is_batched:
                corr_angle = (x[:, i] * x[:, i + 1]) * (pnp.pi / 4.0)
            else:
                corr_angle = (x[i] * x[i + 1]) * (pnp.pi / 4.0)
            qml.RZ(corr_angle, wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])

        # Optional 2D grid-style local entanglement for square layouts (e.g., 4x4)
        if len(wires) >= 4:
            width = int(pnp.sqrt(len(wires)))
            if width * width == len(wires):
                # Horizontal neighbors
                for r in range(width):
                    for c in range(width - 1):
                        a = r * width + c
                        b = r * width + (c + 1)
                        if a < L and b < L:
                            qml.CNOT(wires=[wires[a], wires[b]])
                            if is_batched:
                                qml.RZ((x[:, a] * x[:, b]) * (pnp.pi / 8.0), wires=wires[b])
                            else:
                                qml.RZ((x[a] * x[b]) * (pnp.pi / 8.0), wires=wires[b])
                            qml.CNOT(wires=[wires[a], wires[b]])
                # Vertical neighbors
                for c in range(width):
                    for r in range(width - 1):
                        a = r * width + c
                        b = (r + 1) * width + c
                        if a < L and b < L:
                            qml.CNOT(wires=[wires[a], wires[b]])
                            if is_batched:
                                qml.RZ((x[:, a] * x[:, b]) * (pnp.pi / 8.0), wires=wires[b])
                            else:
                                qml.RZ((x[a] * x[b]) * (pnp.pi / 8.0), wires=wires[b])
                            qml.CNOT(wires=[wires[a], wires[b]])

        # Layer 3: Second-order rotations
        for i in range(L):
            angle_y_i = angles_y[:, i] if is_batched else angles_y[i]
            qml.RY(angle_y_i, wires=wires[i])
