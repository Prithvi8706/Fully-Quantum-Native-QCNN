"""Validate the E4 analysis functions on small known states (roadmap M2.5).

Every number E4 reports comes out of these five functions. If they are wrong,
the mechanism figure is wrong in a way no amount of circuit correctness would
catch -- so each is pinned against states whose values are known analytically
before any circuit output is believed.
"""
import numpy as np
import pytest

from QCNN.utils.state_metrics import (
    describe,
    l1_coherence,
    mutual_information,
    purity,
    reduced_density_matrix,
    von_neumann_entropy,
)

INV_SQRT2 = 1.0 / np.sqrt(2.0)

KET_0 = np.array([1, 0], dtype=complex)
KET_PLUS = np.array([INV_SQRT2, INV_SQRT2], dtype=complex)
BELL = np.array([INV_SQRT2, 0, 0, INV_SQRT2], dtype=complex)
MAXIMALLY_MIXED = np.eye(2, dtype=complex) / 2


def test_pure_computational_basis_state():
    rho = np.outer(KET_0, KET_0.conj())
    assert l1_coherence(rho) == pytest.approx(0.0, abs=1e-15)
    assert purity(rho) == pytest.approx(1.0, abs=1e-15)
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-15)


def test_plus_state_is_pure_but_maximally_coherent():
    rho = np.outer(KET_PLUS, KET_PLUS.conj())
    assert l1_coherence(rho) == pytest.approx(1.0, abs=1e-15)
    assert purity(rho) == pytest.approx(1.0, abs=1e-15)
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-14)


def test_maximally_mixed_qubit():
    assert l1_coherence(MAXIMALLY_MIXED) == pytest.approx(0.0, abs=1e-15)
    assert purity(MAXIMALLY_MIXED) == pytest.approx(0.5, abs=1e-15)
    assert von_neumann_entropy(MAXIMALLY_MIXED) == pytest.approx(1.0, abs=1e-14)


def test_entropy_of_a_maximally_mixed_two_qubit_state_is_two_bits():
    assert von_neumann_entropy(np.eye(4, dtype=complex) / 4) == pytest.approx(2.0, abs=1e-14)


def test_partial_trace_of_a_bell_state_is_maximally_mixed():
    rho = reduced_density_matrix(BELL, 2, keep=[0])
    np.testing.assert_allclose(rho, MAXIMALLY_MIXED, atol=1e-15)


def test_partial_trace_of_a_product_state_returns_the_factor():
    psi = np.kron(KET_0, KET_PLUS)
    np.testing.assert_allclose(
        reduced_density_matrix(psi, 2, keep=[0]), np.outer(KET_0, KET_0.conj()), atol=1e-15)
    np.testing.assert_allclose(
        reduced_density_matrix(psi, 2, keep=[1]), np.outer(KET_PLUS, KET_PLUS.conj()),
        atol=1e-15)


def test_partial_trace_respects_wire_order():
    """|0>|+> traced onto wire 1 must give |+>, not |0>."""
    psi = np.kron(KET_0, KET_PLUS)
    assert l1_coherence(reduced_density_matrix(psi, 2, keep=[0])) == pytest.approx(0.0, abs=1e-15)
    assert l1_coherence(reduced_density_matrix(psi, 2, keep=[1])) == pytest.approx(1.0, abs=1e-15)


def test_bell_state_has_two_bits_of_mutual_information():
    """Maximal for two qubits: S(A)=S(B)=1, S(AB)=0."""
    assert mutual_information(BELL, 2, [0], [1]) == pytest.approx(2.0, abs=1e-14)


def test_product_state_has_no_mutual_information():
    psi = np.kron(KET_PLUS, KET_PLUS)
    assert mutual_information(psi, 2, [0], [1]) == pytest.approx(0.0, abs=1e-14)


def test_mutual_information_rejects_overlapping_subsystems():
    with pytest.raises(ValueError, match='disjoint'):
        mutual_information(BELL, 2, [0], [0])


def test_mutual_information_of_an_empty_subsystem_is_zero():
    assert mutual_information(BELL, 2, [0], []) == 0.0


def test_ghz_state_across_an_uneven_cut():
    """|000> + |111>: one bit of entropy on either side of any 1-vs-2 cut."""
    ghz = np.zeros(8, dtype=complex)
    ghz[0] = ghz[7] = INV_SQRT2

    assert von_neumann_entropy(
        reduced_density_matrix(ghz, 3, keep=[0])) == pytest.approx(1.0, abs=1e-14)
    assert von_neumann_entropy(
        reduced_density_matrix(ghz, 3, keep=[1, 2])) == pytest.approx(1.0, abs=1e-14)
    assert mutual_information(ghz, 3, [0], [1, 2]) == pytest.approx(2.0, abs=1e-14)


def test_describe_reports_the_full_record():
    record = describe(BELL, 2, keep=[0], discard=[1])
    assert record['n_kept'] == 1
    assert record['purity'] == pytest.approx(0.5, abs=1e-14)
    assert record['von_neumann_entropy'] == pytest.approx(1.0, abs=1e-14)
    assert record['mutual_information'] == pytest.approx(2.0, abs=1e-14)


def test_describe_omits_mutual_information_without_a_discard_set():
    assert 'mutual_information' not in describe(BELL, 2, keep=[0])


def test_empty_keep_is_rejected():
    with pytest.raises(ValueError, match='at least one wire'):
        reduced_density_matrix(BELL, 2, keep=[])


def test_entropy_never_returns_a_negative_value():
    """A pure state's surviving eigenvalue is 1 + O(eps), so -p log2 p can go
    slightly negative. Entropy is non-negative by definition; a negative would be
    a defect in the reported number."""
    for state in (KET_0, KET_PLUS, BELL):
        rho = np.outer(state, state.conj())
        assert von_neumann_entropy(rho) >= 0.0
        assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-12)
