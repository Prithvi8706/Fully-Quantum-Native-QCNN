"""Capacity, expressibility and generalization measures (roadmap M3 / Phase 3).

These are the learning-theory quantities T6 reports. Like ``state_metrics`` they
live apart from the experiment so they can be pinned against analytically known
cases before any circuit output is believed -- see ``tests/test_capacity.py``.

References implemented here:
  Sim, Johnson and Aspuru-Guzik, Adv. Quantum Technol. 2, 1900070 (2019)
      -- expressibility as KL divergence from the Haar fidelity distribution.
  Caro et al., Nat. Commun. 13, 4919 (2022)
      -- generalization scaling in the number of trainable gates.
"""
import numpy as np

# Probabilities below this are treated as empty bins when forming the KL sum,
# rather than producing an infinite term from a finite sample.
_PROBABILITY_FLOOR = 1e-12


def haar_fidelity_pdf(fidelity, n_qubits: int):
    """Haar-random fidelity density ``P(F) = (N-1)(1-F)^(N-2)``, ``N = 2**n``."""
    dimension = 2 ** n_qubits
    f = np.asarray(fidelity, dtype=float)
    return (dimension - 1) * np.power(np.clip(1.0 - f, 0.0, 1.0), dimension - 2)


def haar_bin_probabilities(edges, n_qubits: int) -> np.ndarray:
    """Exact Haar probability mass per histogram bin.

    Integrating the density analytically -- ``(1-F)^(N-1)`` is its antiderivative
    up to sign -- avoids the bias that midpoint sampling introduces in the first
    bin, where the density is steepest and most of the mass sits.
    """
    edges = np.asarray(edges, dtype=float)
    dimension = 2 ** n_qubits
    cdf = 1.0 - np.power(np.clip(1.0 - edges, 0.0, 1.0), dimension - 1)
    return np.diff(cdf)


def expressibility_kl(fidelities, n_qubits: int, n_bins: int = 75) -> dict:
    """KL divergence of the sampled fidelity distribution from Haar.

    Smaller is more expressible: zero means the ansatz covers state space like a
    Haar-random unitary. Reported with the bin count because the value depends
    on it, which is a known property of this measure rather than a defect.
    """
    fidelities = np.asarray(fidelities, dtype=float)
    if fidelities.size == 0:
        raise ValueError('expressibility needs at least one fidelity sample')

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(np.clip(fidelities, 0.0, 1.0), bins=edges)
    p_ansatz = counts / counts.sum()
    p_haar = haar_bin_probabilities(edges, n_qubits)

    mask = p_ansatz > _PROBABILITY_FLOOR
    kl = float(np.sum(p_ansatz[mask] * np.log(p_ansatz[mask] / np.maximum(
        p_haar[mask], _PROBABILITY_FLOOR))))
    return {
        'kl_divergence': kl,
        'n_bins': int(n_bins),
        'n_samples': int(fidelities.size),
        'mean_fidelity': float(fidelities.mean()),
        'haar_mean_fidelity': float(1.0 / (2 ** n_qubits)),
    }


def caro_generalization_bound(n_trainable_gates: int, n_train: int) -> dict:
    """Caro et al. (2022) generalization scaling, ``sqrt(T log T / N)``.

    Returned without an assumed constant: the theorem is a big-O statement, so
    the honest quantity is the scaling term itself. A value below 1 is
    non-vacuous only up to that constant, which the manuscript must say.
    """
    if n_trainable_gates < 1 or n_train < 1:
        raise ValueError('gate count and sample count must be positive')
    t = float(n_trainable_gates)
    scaling = float(np.sqrt(t * np.log(t) / n_train)) if t > 1 else 0.0
    return {
        'n_trainable_gates': int(n_trainable_gates),
        'n_train': int(n_train),
        'sqrt_T_logT_over_N': scaling,
        'non_vacuous': bool(scaling < 1.0),
    }


def effective_dimension(fisher_eigenvalues, n_train: int, n_parameters: int = None,
                        gamma: float = 1.0) -> dict:
    """Abbas et al. (2021) effective dimension from empirical Fisher eigenvalues.

    ``d_eff = 2 log( (1/V) integral sqrt(det(I + kappa F)) ) / log kappa`` reduces,
    for the normalised Fisher spectrum, to a sum over eigenvalues. The volume
    term cancels in the normalisation used here, so what is reported is the
    spectrum-driven part -- documented as such rather than presented as the full
    constant-exact quantity.
    """
    eigenvalues = np.asarray(fisher_eigenvalues, dtype=float)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    d = int(n_parameters if n_parameters is not None else eigenvalues.size)
    if d == 0:
        raise ValueError('need at least one parameter')

    trace = eigenvalues.sum()
    if trace <= 0:
        return {'effective_dimension': 0.0, 'n_parameters': d, 'n_train': int(n_train),
                'normalised': True}

    normalised = d * eigenvalues / trace          # so that the spectrum sums to d
    kappa = gamma * n_train / (2.0 * np.pi * np.log(n_train))
    numerator = float(np.sum(np.log1p(kappa * normalised)))
    d_eff = numerator / np.log(kappa) if kappa > 1.0 else 0.0
    return {
        'effective_dimension': float(d_eff),
        'n_parameters': d,
        'n_train': int(n_train),
        'kappa': float(kappa),
        'normalised': True,
    }
