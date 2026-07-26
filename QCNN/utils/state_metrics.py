"""Quantum-information measures on circuit states (roadmap M2.5, feeds M3.6).

E4 reports coherence, purity, entropy and kept/discarded mutual information at
each stage of the frozen circuit. Roadmap M2.5 requires these functions to be
validated on small known states before any of their outputs are believed, which
is why they live here as a separately testable unit rather than inline in the
experiment: see ``tests/test_state_metrics.py``.

Conventions: entropies are in bits (log base 2); wire indices are absolute
positions in the full register, matching ``QCNN/circuits.py``.
"""
import numpy as np

# Eigenvalues of a numerically-obtained density matrix can sit a few ulp below
# zero; below this magnitude they are treated as exact zeros rather than as
# NaN-producing logarithms.
_EIGENVALUE_FLOOR = 1e-12


def reduced_density_matrix(state, n_qubits: int, keep) -> np.ndarray:
    """Partial trace of a pure state down to ``keep`` (absolute wire indices)."""
    keep = sorted(int(w) for w in keep)
    if not keep:
        raise ValueError('keep must name at least one wire')
    traced = [w for w in range(n_qubits) if w not in keep]

    tensor = np.asarray(state, dtype=complex).reshape([2] * n_qubits)
    tensor = np.transpose(tensor, keep + traced)
    matrix = tensor.reshape(2 ** len(keep), 2 ** len(traced))
    return matrix @ matrix.conj().T


def l1_coherence(rho: np.ndarray) -> float:
    """Baumgratz-Cramer-Plenio l1 measure: the off-diagonal absolute mass."""
    rho = np.asarray(rho)
    return float(np.abs(rho).sum() - np.abs(np.diag(rho)).sum())


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2). 1 for a pure state, 1/d for the maximally mixed state."""
    rho = np.asarray(rho)
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """-Tr(rho log2 rho), in bits.

    Clamped at zero: the quantity is non-negative by definition, but for a pure
    state the surviving eigenvalue is 1 + O(eps) and ``-p log2 p`` then returns a
    value around -1e-16. Reporting a negative entropy would be a defect in the
    output, not a property of the state.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(rho))
    eigenvalues = eigenvalues[eigenvalues > _EIGENVALUE_FLOOR]
    if eigenvalues.size == 0:
        return 0.0
    return float(max(0.0, -np.sum(eigenvalues * np.log2(eigenvalues))))


def mutual_information(state, n_qubits: int, part_a, part_b) -> float:
    """I(A:B) = S(A) + S(B) - S(AB), in bits, for disjoint wire sets."""
    part_a = sorted(int(w) for w in part_a)
    part_b = sorted(int(w) for w in part_b)
    if set(part_a) & set(part_b):
        raise ValueError('subsystems must be disjoint')
    if not part_a or not part_b:
        return 0.0

    s_a = von_neumann_entropy(reduced_density_matrix(state, n_qubits, part_a))
    s_b = von_neumann_entropy(reduced_density_matrix(state, n_qubits, part_b))
    s_ab = von_neumann_entropy(
        reduced_density_matrix(state, n_qubits, part_a + part_b))
    return float(s_a + s_b - s_ab)


def meyer_wallach(state, n_qubits: int) -> float:
    """Meyer-Wallach entangling capability Q (Phase 3.4).

    ``Q = 2 (1 - (1/n) sum_k Tr[rho_k^2])`` over single-qubit reductions. Zero
    for a product state, one for a maximally entangled state such as GHZ, and
    the standard measure reported alongside expressibility in Sim et al. (2019).
    """
    purities = [purity(reduced_density_matrix(state, n_qubits, [k]))
                for k in range(n_qubits)]
    return float(2.0 * (1.0 - sum(purities) / n_qubits))


def describe(state, n_qubits: int, keep, discard=None) -> dict:
    """Every E4 quantity for one stage, on the retained register."""
    rho = reduced_density_matrix(state, n_qubits, keep)
    record = {
        'n_kept': len(list(keep)),
        'l1_coherence': l1_coherence(rho),
        'purity': purity(rho),
        'von_neumann_entropy': von_neumann_entropy(rho),
    }
    if discard:
        record['mutual_information'] = mutual_information(
            state, n_qubits, keep, discard)
    return record
