"""Validate the Phase 3 capacity measures against known cases (roadmap M3).

Same rule as the E4 metrics: every quantity T6 will report is pinned here
against a case whose answer is known independently of the code, before any
circuit output is believed.
"""
import numpy as np
import pytest

from QCNN.utils.capacity import (
    caro_generalization_bound,
    effective_dimension,
    expressibility_kl,
    haar_bin_probabilities,
    haar_fidelity_pdf,
    variance_decay_fit,
)
from QCNN.utils.state_metrics import meyer_wallach

INV_SQRT2 = 1.0 / np.sqrt(2.0)


# --- Meyer-Wallach entangling capability ----------------------------------

def test_meyer_wallach_is_zero_for_a_product_state():
    psi = np.kron(np.array([1, 0], dtype=complex), np.array([INV_SQRT2, INV_SQRT2]))
    assert meyer_wallach(psi, 2) == pytest.approx(0.0, abs=1e-14)


def test_meyer_wallach_is_one_for_a_bell_state():
    bell = np.array([INV_SQRT2, 0, 0, INV_SQRT2], dtype=complex)
    assert meyer_wallach(bell, 2) == pytest.approx(1.0, abs=1e-14)


def test_meyer_wallach_is_one_for_ghz_at_three_qubits():
    ghz = np.zeros(8, dtype=complex)
    ghz[0] = ghz[7] = INV_SQRT2
    assert meyer_wallach(ghz, 3) == pytest.approx(1.0, abs=1e-14)


def test_meyer_wallach_of_a_fully_unentangled_register_is_zero():
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    assert meyer_wallach(psi, 4) == pytest.approx(0.0, abs=1e-14)


# --- Haar fidelity distribution -------------------------------------------

def test_haar_density_integrates_to_one():
    for n in (2, 4, 8):
        grid = np.linspace(0.0, 1.0, 200001)
        assert np.trapz(haar_fidelity_pdf(grid, n), grid) == pytest.approx(1.0, abs=1e-4)


def test_haar_bin_probabilities_sum_to_one():
    for n in (2, 6, 10):
        edges = np.linspace(0.0, 1.0, 76)
        assert haar_bin_probabilities(edges, n).sum() == pytest.approx(1.0, abs=1e-12)


def test_haar_bin_probabilities_are_non_negative():
    edges = np.linspace(0.0, 1.0, 76)
    assert (haar_bin_probabilities(edges, 10) >= 0).all()


def test_haar_mass_concentrates_near_zero_fidelity_as_dimension_grows():
    """Two random high-dimensional states are almost always nearly orthogonal."""
    edges = np.linspace(0.0, 1.0, 101)
    small = haar_bin_probabilities(edges, 2)[0]
    large = haar_bin_probabilities(edges, 10)[0]
    assert large > small
    assert large > 0.99


# --- Expressibility --------------------------------------------------------

def test_expressibility_of_haar_distributed_samples_is_near_zero():
    """Sampling the Haar law itself must score as maximally expressible."""
    n = 6
    dimension = 2 ** n
    rng = np.random.default_rng(0)
    # F = 1 - u**(1/(N-1)) is the inverse-CDF sample of the Haar fidelity law.
    u = rng.random(200000)
    fidelities = 1.0 - u ** (1.0 / (dimension - 1))

    result = expressibility_kl(fidelities, n)
    assert result['kl_divergence'] < 0.01
    assert result['mean_fidelity'] == pytest.approx(1.0 / dimension, rel=0.1)


def test_expressibility_of_a_degenerate_ansatz_is_large():
    """An ansatz that always produces the same state is maximally inexpressible."""
    result = expressibility_kl(np.ones(5000), n_qubits=6)
    assert result['kl_divergence'] > 5.0


def test_expressibility_rejects_an_empty_sample():
    with pytest.raises(ValueError, match='at least one'):
        expressibility_kl([], n_qubits=4)


def test_expressibility_reports_its_bin_count():
    """The value depends on binning, so the binning must travel with it."""
    result = expressibility_kl(np.linspace(0, 1, 1000), n_qubits=4, n_bins=50)
    assert result['n_bins'] == 50
    assert result['n_samples'] == 1000


# --- Caro generalization bound --------------------------------------------

def test_caro_bound_matches_the_closed_form():
    result = caro_generalization_bound(74, 7599)
    expected = np.sqrt(74 * np.log(74) / 7599)
    assert result['sqrt_T_logT_over_N'] == pytest.approx(expected)
    assert result['non_vacuous'] is True


def test_caro_bound_grows_with_gate_count():
    """Parameter frugality is the whole argument, so this must be monotone."""
    few = caro_generalization_bound(74, 7599)['sqrt_T_logT_over_N']
    many = caro_generalization_bound(269, 7599)['sqrt_T_logT_over_N']
    assert many > few


def test_caro_bound_shrinks_with_more_data():
    small = caro_generalization_bound(74, 500)['sqrt_T_logT_over_N']
    large = caro_generalization_bound(74, 50000)['sqrt_T_logT_over_N']
    assert large < small


def test_caro_bound_flags_a_vacuous_result():
    assert caro_generalization_bound(5000, 100)['non_vacuous'] is False


def test_caro_bound_rejects_nonsense_inputs():
    with pytest.raises(ValueError, match='positive'):
        caro_generalization_bound(0, 100)
    with pytest.raises(ValueError, match='positive'):
        caro_generalization_bound(10, 0)


# --- Effective dimension ---------------------------------------------------

def test_effective_dimension_is_bounded_by_the_parameter_count():
    rng = np.random.default_rng(1)
    eigenvalues = rng.random(40)
    result = effective_dimension(eigenvalues, n_train=5000)
    assert 0.0 <= result['effective_dimension'] <= result['n_parameters'] + 1e-9


def test_a_flat_fisher_spectrum_uses_most_of_the_parameter_space():
    """All directions equally informative -> effective dimension near the full count."""
    result = effective_dimension(np.ones(30), n_train=10000)
    assert result['effective_dimension'] > 0.7 * 30


def test_a_rank_one_fisher_spectrum_collapses_the_dimension():
    spectrum = np.zeros(30)
    spectrum[0] = 1.0
    flat = effective_dimension(np.ones(30), n_train=10000)['effective_dimension']
    peaked = effective_dimension(spectrum, n_train=10000)['effective_dimension']
    assert peaked < flat


def test_effective_dimension_of_a_dead_spectrum_is_zero():
    assert effective_dimension(np.zeros(10), n_train=1000)['effective_dimension'] == 0.0


def test_effective_dimension_rejects_an_empty_spectrum():
    with pytest.raises(ValueError, match='at least one'):
        effective_dimension([], n_train=100, n_parameters=0)


# ---------------------------------------------------------------------------
# variance_decay_fit (Phase 3.2 / F-E): the barren-plateau certificate.
#
# Pinned against series whose decay law is exact by construction, so a wrong
# rate or a wrong law preference is a defect in the fit, not in the circuit.
# ---------------------------------------------------------------------------

def test_recovers_an_exact_exponential_rate():
    """Var = 3 exp(-0.7 n) must return gamma = 0.7 and a perfect fit."""
    n = np.array([4, 6, 8, 10, 12, 14])
    variances = 3.0 * np.exp(-0.7 * n)

    result = variance_decay_fit(n, variances)

    assert result['exponential_rate'] == pytest.approx(0.7, abs=1e-10)
    assert result['exponential_prefactor'] == pytest.approx(3.0, rel=1e-9)
    assert result['exponential_r_squared'] == pytest.approx(1.0, abs=1e-12)
    assert result['preferred_law'] == 'exponential'


def test_recovers_an_exact_power_law_exponent():
    """Var = 5 n^-2 must return p = 2 and prefer the power law."""
    n = np.array([4, 6, 8, 10, 12, 14])
    variances = 5.0 * n.astype(float) ** -2.0

    result = variance_decay_fit(n, variances)

    assert result['power_law_exponent'] == pytest.approx(2.0, abs=1e-10)
    assert result['power_law_prefactor'] == pytest.approx(5.0, rel=1e-9)
    assert result['power_law_r_squared'] == pytest.approx(1.0, abs=1e-12)
    assert result['preferred_law'] == 'power_law'


def test_distinguishes_a_barren_plateau_from_polynomial_decay():
    """The two laws must not be confusable on data generated by either one."""
    n = np.array([4, 6, 8, 10, 12, 14])
    exponential = variance_decay_fit(n, 2.0 ** (-n.astype(float)))
    polynomial = variance_decay_fit(n, 1.0 / n.astype(float))

    assert exponential['preferred_law'] == 'exponential'
    assert polynomial['preferred_law'] == 'power_law'
    assert exponential['exponential_r_squared'] > exponential['power_law_r_squared']
    assert polynomial['power_law_r_squared'] > polynomial['exponential_r_squared']


def test_halving_per_qubit_is_reported_as_a_ratio():
    """Var = 2^-n halves per added qubit: rate = ln 2, ratio = 1/2."""
    n = np.array([4, 6, 8, 10, 12])
    result = variance_decay_fit(n, 2.0 ** (-n.astype(float)))

    assert result['exponential_rate'] == pytest.approx(np.log(2.0), abs=1e-10)
    assert result['variance_ratio_per_qubit'] == pytest.approx(0.5, rel=1e-9)


def test_a_flat_series_is_a_perfect_fit_with_zero_rate():
    """Constant variance: no decay, and the R-squared guard must not divide by zero."""
    result = variance_decay_fit([4, 6, 8, 10], [0.25] * 4)

    assert result['exponential_rate'] == pytest.approx(0.0, abs=1e-12)
    assert result['exponential_r_squared'] == 1.0
    assert result['power_law_r_squared'] == 1.0


def test_variance_decay_fit_rejects_unfittable_input():
    with pytest.raises(ValueError, match='equal length'):
        variance_decay_fit([4, 6, 8], [1.0, 2.0])
    with pytest.raises(ValueError, match='at least 3'):
        variance_decay_fit([4, 6], [1.0, 0.5])
    with pytest.raises(ValueError, match='positive'):
        variance_decay_fit([4, 6, 8], [1.0, 0.0, 0.5])
