import pennylane as qml
import pennylane.numpy as pnp
import numpy as np


class QuantumNativePooling:
    """
    Quantum-native pooling operations for QCNN architectures.
    Provides both measurement-based pooling and fully unitary
    pooling mechanisms for coherent downsampling of qubit grids.
    """

    @staticmethod
    def quantum_measurement_pooling(measure_qubits: list[int]) -> list[qml.measurements.ExpectationMP]:
        """
        Quantum pooling via partial measurement.
        Measures the selected qubits with expval(Z), enabling
        a measurement-based reduction of quantum information.

        Args:
            measure_qubits: list of qubit indices to measure

        Returns:
            List of expectation value measurement objects.
        """
        measurements = []
        for qubit in measure_qubits:
            measurements.append(qml.expval(qml.PauliZ(qubit)))
        return measurements
    
    @staticmethod
    def pair_angles(angles, idx: int):
        """The three angles pair ``idx`` consumes: ``(cry, crz, post_ry_keep)``.

        Shared by every pooling arm. If the arms disagreed about which angle is
        which, E1's exact-tie prediction would fail for a bookkeeping reason and
        look like a physics result (UPGRADE_PLAN.md 2.3).
        """
        size = angles.size
        return (angles[(3 * idx) % size],
                angles[(3 * idx + 1) % size],
                angles[(3 * idx + 2) % size])

    @staticmethod
    def quantum_unitary_pooling(params: np.ndarray, input_qubits: list[int],
                                output_qubits: list[int]) -> None:
        """
        Quantum unitary pooling – coherently compresses local quantum information.
        
        Updated semantics:
        - input_qubits are the KEEP wires (targets).
        - output_qubits are the DISCARD wires (controls).
        - Uses CRY/CRZ controlled rotations from discard → keep.
        - Applies a small disentangling RY on discard.
        - Applies consolidation RY on keep.
        
        This implements a fully unitary downsampling layer
        consistent with quantum-native QCNN designs.

        Args:
            params: flat or vector-like array of trainable pooling angles
            input_qubits: list of qubits to keep after pooling
            output_qubits: list of qubits to discard after pooling
        """
        # Flatten while preserving autograd types (avoid dtype=float casts)
        angles = pnp.asarray(params).reshape(-1)
        if angles.size == 0:
            return

        # We consume 3 angles per pair: [cry, crz, post_ry_keep]. The mapping is
        # shared with the measurement arms so E1 compares like with like.
        def triple(idx):
            return QuantumNativePooling.pair_angles(angles, idx)

        n_pairs = min(len(input_qubits), len(output_qubits))
        for i in range(n_pairs):
            keep = input_qubits[i]
            discard = output_qubits[i]

            # Skip accidental self-pairing
            if keep == discard:
                continue

            a, b, c = triple(i)

            # Controlled rotations from discard -> keep
            qml.CRY(a, wires=[discard, keep])
            qml.CRZ(b, wires=[discard, keep])

            # Light disentangle touch on discard (not used after pooling)
            qml.RY(0.02, wires=discard)

            # Consolidate on keep
            qml.RY(c, wires=keep)

        # After this, the model should drop 'output_qubits' from the active set to downsample by half.

    @staticmethod
    def apply_pooling(mode: str, params: np.ndarray, input_qubits: list[int],
                      output_qubits: list[int]) -> None:
        """
        Dispatch to a pooling implementation based on the ablation ``mode``.

        Modes:
        - 'unitary'             : the frozen headline block (CRY/CRZ from discard
                                  to keep). Fully unitary; the proposed model.
        - 'none'                : no information-transfer gate; the wire is simply
                                  retired from the active set. Ablation floor.
        - 'measurement'         : measure-and-condition via mid-circuit measurement.
                                  Trainable arm; runs on pure-state devices.
        - 'measurement_channel' : the same map as an explicit CPTP channel.
                                  Needs a mixed-state device; used by E1.

        Theorem 1 predicts 'unitary' and the two measurement arms agree exactly on
        every observable of the retained register, because the frozen block's
        controls are diagonal in the discarded qubit's basis. E1 tests that.

        Args:
            mode: pooling mode string.
            params: trainable pooling angles (unused by 'none').
            input_qubits: qubits to keep.
            output_qubits: qubits to discard.
        """
        if mode == 'unitary':
            QuantumNativePooling.quantum_unitary_pooling(params, input_qubits, output_qubits)
        elif mode == 'none':
            return
        elif mode == 'measurement':
            QuantumNativePooling.quantum_conditional_pooling(params, input_qubits, output_qubits)
        elif mode == 'measurement_channel':
            QuantumNativePooling.quantum_measurement_channel_pooling(
                params, input_qubits, output_qubits)
        else:
            raise ValueError(f"Unknown pooling_mode '{mode}'. Use 'unitary', 'none', "
                             "'measurement', or 'measurement_channel'.")

    @staticmethod
    def quantum_conditional_pooling(params: np.ndarray, input_qubits: list[int],
                                    output_qubits: list[int]) -> None:
        """Measure-and-condition pooling: the exact channel Theorem 1 maps onto.

        The frozen block's controls are diagonal in the discarded qubit's basis,
        so it factorises as ``V = |0><0|_b (x) U_0 + |1><1|_b (x) U_1`` with

            U_0 = RY(gamma)
            U_1 = RY(gamma) . RZ(beta) . RY(alpha)

        Deferred measurement then gives ``Tr_b[V rho V^dag] = sum_m U_m <m|rho|m> U_m^dag``.
        This arm must therefore apply *exactly* U_0 on outcome 0 and U_1 on
        outcome 1, which is what makes E1's exact tie a prediction rather than a
        coincidence (UPGRADE_PLAN.md 2.3, Theorem 1).

        Before 2026-07-25 this applied the identity on outcome 0 and RY(alpha) on
        outcome 1, consuming one of the three angles per pair. That is a strictly
        weaker correction than the theorem requires, so E1 could not tie -- and
        its failure would have read as evidence for the very information-loss
        claim the upgrade exists to delete (F6).

        The frozen arm's ``RY(0.02)`` on the discarded wire has no counterpart
        here and needs none: it acts only on ``b``, and the partial trace over
        ``b`` is invariant under a unitary on ``b`` alone.
        """
        angles = pnp.asarray(params).reshape(-1)
        if angles.size == 0:
            return
        n_pairs = min(len(input_qubits), len(output_qubits))
        for i in range(n_pairs):
            keep = input_qubits[i]
            discard = output_qubits[i]
            if keep == discard:
                continue
            a, b, c = QuantumNativePooling.pair_angles(angles, i)
            m = qml.measure(discard)
            qml.cond(m, qml.RY)(a, wires=keep)
            qml.cond(m, qml.RZ)(b, wires=keep)
            qml.RY(c, wires=keep)          # the factor U_0 and U_1 share

    @staticmethod
    def measurement_channel_kraus(angles, idx: int):
        """Kraus operators for one pair's measure-and-condition channel.

        ``K_m = U_m (x) |m><m|`` on wires ``[keep, discard]``. Completeness:
        ``sum_m K_m^dag K_m = I (x) |0><0| + I (x) |1><1| = I``.

        This is the density-matrix form of the same map ``quantum_conditional_pooling``
        realises with mid-circuit measurement. E1 compares the frozen unitary
        block against *this*, because it is a genuine non-unitary channel rather
        than something a simulator might quietly defer back into the very
        controlled gates under test.
        """
        alpha, beta, gamma = QuantumNativePooling.pair_angles(angles, idx)
        alpha, beta, gamma = float(alpha), float(beta), float(gamma)

        def ry(t):
            return np.array([[np.cos(t / 2), -np.sin(t / 2)],
                             [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)

        def rz(t):
            return np.array([[np.exp(-0.5j * t), 0.0],
                             [0.0, np.exp(0.5j * t)]], dtype=complex)

        u0 = ry(gamma)
        u1 = ry(gamma) @ rz(beta) @ ry(alpha)
        p0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        p1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
        return [np.kron(u0, p0), np.kron(u1, p1)]

    @staticmethod
    def quantum_measurement_channel_pooling(params: np.ndarray, input_qubits: list[int],
                                            output_qubits: list[int]) -> None:
        """Measure-and-condition pooling as an explicit CPTP channel.

        Requires a mixed-state device. Used by E1; the trainable arm is
        ``quantum_conditional_pooling``, which implements the same map.
        """
        angles = pnp.asarray(params).reshape(-1)
        if angles.size == 0:
            return
        n_pairs = min(len(input_qubits), len(output_qubits))
        for i in range(n_pairs):
            keep = input_qubits[i]
            discard = output_qubits[i]
            if keep == discard:
                continue
            qml.QubitChannel(
                QuantumNativePooling.measurement_channel_kraus(angles, i),
                wires=[keep, discard])

    @staticmethod
    def make_pairing(active_wires: list[int]) -> list[tuple[int, int]]:
        """
        Default pairing strategy for pooling.
        Groups consecutive active wires into (keep, discard) pairs.
        
        Args:
            active_wires: list of active qubit indices

        Returns:
            List of (keep, discard) tuples.
        """
        pairs = []
        for i in range(0, len(active_wires) - 1, 2):
            keep = active_wires[i]
            discard = active_wires[i + 1]
            pairs.append((keep, discard))
        return pairs
