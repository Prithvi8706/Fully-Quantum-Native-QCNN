"""Pin the DLA primitives against Lie algebras of known dimension (roadmap M3.1).

Same rule as the E4 metrics and the Phase 3 capacity measures: every quantity
3.1 will report is checked here against a case whose answer is known
independently of this code, before any circuit output is believed. That
convention has already caught two real defects in this program.

The known cases used:

  su(2)                       dim 3        {X, Y, Z} on one qubit
  su(2)^(+)n                  dim 3n       local rotations, no coupling
  abelian                     dim n        {Z_i} alone
  su(2^n)                     dim 4^n - 1  local rotations + a connected chain
                                           of couplings (standard universality)
  transverse-field Ising      dim 2n^2 - n {X_i} + {Z_i Z_i+1}, open chain
                                           (Wiersema et al. 2024 classification)

The Ising case is the load-bearing one: it is the only pin whose answer is a
non-trivial polynomial, so it is what would catch a closure that silently stops
early or double-counts. It is checked against the formula *and* against
PennyLane's independent ``qml.pauli.lie_closure``.
"""
import numpy as np
import pytest

from pennylane.pauli import PauliSentence, PauliWord, lie_closure as pl_lie_closure

from QCNN.utils import lie_algebra as la


# --- helpers ---------------------------------------------------------------

def local(n, wire, letter):
    return {la.single(n, wire, letter): 1.0}


def coupling(n, a, b, letter_a, letter_b):
    return {la.compose(la.single(n, a, letter_a), la.single(n, b, letter_b)): 1.0}


def all_local_rotations(n):
    return [local(n, q, letter) for q in range(n) for letter in 'XYZ']


def to_pennylane(sentence, n_qubits):
    """Convert one of our sentences to a PennyLane ``PauliSentence``."""
    words = {}
    for key, coeff in sentence.items():
        word = la.to_string(key, n_qubits)
        mapping = {q: letter for q, letter in enumerate(word) if letter != 'I'}
        words[PauliWord(mapping)] = coeff
    return PauliSentence(words)


def pennylane_dimension(generators, n_qubits):
    return len(pl_lie_closure([to_pennylane(g, n_qubits) for g in generators]))


# --- Pauli arithmetic ------------------------------------------------------

def test_pauli_matrix_reproduces_the_single_qubit_paulis():
    assert np.allclose(la.pauli_matrix(la.from_string('X'), 1),
                       [[0, 1], [1, 0]])
    assert np.allclose(la.pauli_matrix(la.from_string('Y'), 1),
                       [[0, -1j], [1j, 0]])
    assert np.allclose(la.pauli_matrix(la.from_string('Z'), 1),
                       [[1, 0], [0, -1]])


def test_pauli_strings_round_trip_through_their_labels():
    for word in ('IXYZ', 'ZZZZ', 'IIII', 'YXZI'):
        assert la.to_string(la.from_string(word), 4) == word


@pytest.mark.parametrize('a,b', [('X', 'Y'), ('Y', 'Z'), ('Z', 'X'),
                                 ('XY', 'YX'), ('ZZ', 'XX'), ('IX', 'ZY'),
                                 ('XYZ', 'ZYX'), ('YYI', 'IZZ')])
def test_product_phase_matches_dense_multiplication(a, b):
    n = len(a)
    ka, kb = la.from_string(a), la.from_string(b)
    phase = (1j) ** la.product_phase(ka, kb)
    expected = la.pauli_matrix(ka, n) @ la.pauli_matrix(kb, n)
    assert np.allclose(expected, phase * la.pauli_matrix(la.compose(ka, kb), n))


@pytest.mark.parametrize('a,b', [('X', 'Y'), ('X', 'X'), ('XY', 'YX'),
                                 ('ZI', 'XX'), ('XYZ', 'ZYX')])
def test_commute_agrees_with_dense_matrices(a, b):
    n = len(a)
    ma, mb = la.pauli_matrix(la.from_string(a), n), la.pauli_matrix(la.from_string(b), n)
    dense_commutes = np.allclose(ma @ mb, mb @ ma)
    assert la.commute(la.from_string(a), la.from_string(b)) == dense_commutes


def test_bracket_matches_i_times_the_dense_commutator():
    n = 3
    a = {la.from_string('XYI'): 0.7, la.from_string('IZZ'): -1.3}
    b = {la.from_string('ZIX'): 0.4, la.from_string('YYY'): 2.0}
    ma, mb = la.sentence_matrix(a, n), la.sentence_matrix(b, n)
    expected = 1j * (ma @ mb - mb @ ma)
    assert np.allclose(la.sentence_matrix(la.bracket(a, b), n), expected)


def test_bracket_of_a_hermitian_pair_is_hermitian_and_real_coefficiented():
    n = 3
    a = {la.from_string('XYI'): 0.7, la.from_string('IZZ'): -1.3}
    b = {la.from_string('ZIX'): 0.4}
    result = la.bracket(a, b)
    assert result
    assert all(isinstance(c, float) for c in result.values())
    matrix = la.sentence_matrix(result, n)
    assert np.allclose(matrix, matrix.conj().T)


def test_bracket_of_commuting_elements_is_empty():
    n = 2
    assert la.bracket(local(n, 0, 'Z'), local(n, 1, 'X')) == {}
    assert la.bracket(local(n, 0, 'Z'), coupling(n, 0, 1, 'Z', 'Z')) == {}


# --- rank and reduction ----------------------------------------------------

def test_span_dimension_counts_independent_sentences_only():
    n = 2
    x0, z0 = local(n, 0, 'X'), local(n, 0, 'Z')
    combination = {la.single(n, 0, 'X'): 2.0, la.single(n, 0, 'Z'): -3.0}
    assert la.span_dimension([x0, z0]) == 2
    assert la.span_dimension([x0, z0, combination]) == 2
    assert la.span_dimension([x0, {la.single(n, 0, 'X'): 5.0}]) == 1


def test_identity_terms_are_dropped_as_central():
    n = 1
    assert la.span_dimension([{la.IDENTITY: 1.0}]) == 0
    assert la.span_dimension([{la.IDENTITY: 1.0, la.single(n, 0, 'X'): 1.0}]) == 1


# --- known Lie algebras ----------------------------------------------------

def test_su2_from_the_three_paulis():
    assert la.lie_closure(all_local_rotations(1))['dimension'] == 3


def test_local_rotations_alone_give_a_direct_sum_of_su2():
    for n in (1, 2, 3, 4):
        assert la.lie_closure(all_local_rotations(n))['dimension'] == 3 * n


def test_commuting_generators_close_immediately():
    n = 4
    closure = la.lie_closure([local(n, q, 'Z') for q in range(n)])
    assert closure['dimension'] == n
    assert closure['commutator_rounds'] == 1
    assert closure['closed']


@pytest.mark.parametrize('n', [2, 3, 4])
def test_local_rotations_plus_a_connected_chain_give_full_su(n):
    """Standard universality: single-qubit rotations everywhere plus a connected
    two-qubit coupling generate all of su(2^n)."""
    generators = all_local_rotations(n)
    generators += [coupling(n, q, q + 1, 'Z', 'Z') for q in range(n - 1)]
    closure = la.lie_closure(generators)
    assert closure['closed']
    assert closure['dimension'] == 4 ** n - 1


@pytest.mark.parametrize('n,expected', [(2, 6), (3, 15), (4, 28)])
def test_transverse_field_ising_matches_the_known_polynomial(n, expected):
    """Open-chain TFIM has DLA dimension ``2n^2 - n`` (Wiersema et al. 2024)."""
    generators = [local(n, q, 'X') for q in range(n)]
    generators += [coupling(n, q, q + 1, 'Z', 'Z') for q in range(n - 1)]
    closure = la.lie_closure(generators)
    assert closure['closed']
    assert closure['dimension'] == expected == 2 * n * n - n


@pytest.mark.parametrize('n', [2, 3, 4])
def test_closure_agrees_with_pennylanes_independent_implementation(n):
    """Cross-check against ``qml.pauli.lie_closure``.

    A second implementation of the same object, written by someone else, is the
    control that a formula pin cannot give: it catches an error shared between
    this code and my reading of the reference.
    """
    cases = [
        [local(n, q, 'X') for q in range(n)]
        + [coupling(n, q, q + 1, 'Z', 'Z') for q in range(n - 1)],
        all_local_rotations(n),
        all_local_rotations(n) + [coupling(n, 0, n - 1, 'Z', 'Z')],
    ]
    for generators in cases:
        assert (la.lie_closure(generators)['dimension']
                == pennylane_dimension(generators, n))


def test_a_dla_never_exceeds_the_su_ceiling():
    n = 3
    generators = all_local_rotations(n) + [
        coupling(n, 0, 1, 'X', 'Y'), coupling(n, 1, 2, 'Z', 'Z')]
    assert la.lie_closure(generators)['dimension'] <= 4 ** n - 1


# --- block structure -------------------------------------------------------

def test_disconnected_couplings_give_a_direct_sum_of_blocks():
    n = 4
    generators = all_local_rotations(n) + [coupling(n, 0, 1, 'Z', 'Z'),
                                           coupling(n, 2, 3, 'Z', 'Z')]
    closure = la.lie_closure(generators)
    components = la.support_components(closure['basis'], n)
    assert components == [[0, 1], [2, 3]]
    assert closure['dimension'] == la.direct_sum_of_full_blocks(components)
    assert closure['dimension'] == 2 * (4 ** 2 - 1)


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_the_full_su_certificate_agrees_with_exact_closure(n):
    """The certificate is only usable at n where enumeration is not, so it has to
    be pinned where both are available."""
    generators = all_local_rotations(n) + [coupling(n, q, q + 1, 'Z', 'Z')
                                           for q in range(n - 1)]
    closure = la.lie_closure(generators)
    certificate = la.full_su_certificate(closure['basis'], n)
    assert certificate['certified_full_su']
    assert certificate['implied_dimension'] == closure['dimension'] == 4 ** n - 1


def test_the_certificate_refuses_a_disconnected_algebra():
    n = 4
    closure = la.lie_closure(all_local_rotations(n) + [coupling(n, 0, 1, 'Z', 'Z'),
                                                       coupling(n, 2, 3, 'Z', 'Z')])
    certificate = la.full_su_certificate(closure['basis'], n)
    assert certificate['has_local_su2_on_every_wire']
    assert not certificate['support_is_connected']
    assert not certificate['certified_full_su']
    assert certificate['implied_dimension'] is None


def test_the_certificate_refuses_an_algebra_missing_local_control():
    """TFIM is connected but has no ``Z_i`` alone, and is far from full su."""
    n = 3
    generators = [local(n, q, 'X') for q in range(n)]
    generators += [coupling(n, q, q + 1, 'Z', 'Z') for q in range(n - 1)]
    closure = la.lie_closure(generators)
    certificate = la.full_su_certificate(closure['basis'], n)
    assert not certificate['certified_full_su']
    assert certificate['missing_local_generators']
    assert closure['dimension'] == 15 < 4 ** n - 1


def test_the_certificate_accepts_a_prebuilt_span():
    """The run reuses the closure's own echelon; rebuilding it from a million
    rows for each of the 3n membership tests would dominate the run."""
    n = 3
    closure = la.lie_closure(all_local_rotations(n)
                             + [coupling(n, q, q + 1, 'Z', 'Z') for q in range(n - 1)])
    components = la.support_components(closure['basis'], n)
    from_span = la.full_su_certificate(closure['span'], n, components=components)
    from_basis = la.full_su_certificate(closure['basis'], n)
    assert from_span == from_basis
    with pytest.raises(ValueError, match='components explicitly'):
        la.full_su_certificate(closure['span'], n)


def test_contains_recognises_membership_and_non_membership():
    n = 2
    basis = la.lie_closure(all_local_rotations(n))['basis']
    assert la.contains(basis, local(n, 0, 'X'))
    assert la.contains(basis, {la.single(n, 0, 'X'): 2.0, la.single(n, 1, 'Y'): 1.0})
    assert not la.contains(basis, coupling(n, 0, 1, 'Z', 'Z'))


def test_support_components_ignores_untouched_wires():
    n = 4
    closure = la.lie_closure([local(n, 1, 'X'), local(n, 1, 'Z')])
    assert la.support_components(closure['basis'], n) == [[1]]


# --- truncation ------------------------------------------------------------

def test_hitting_the_cap_is_reported_as_not_closed():
    n = 3
    generators = all_local_rotations(n) + [coupling(n, q, q + 1, 'Z', 'Z')
                                           for q in range(n - 1)]
    truncated = la.lie_closure(generators, max_dimension=20)
    assert not truncated['closed']
    assert truncated['dimension'] == 20
    assert la.lie_closure(generators)['dimension'] == 63


def test_the_certificate_still_fires_on_a_truncated_closure():
    """The case the variant attribution relies on: a capped enumeration is only a
    lower bound, but if the *partial* basis already satisfies the universality
    premises then the full closure is su(2^n) and the exact dimension follows."""
    n = 4
    generators = all_local_rotations(n) + [coupling(n, q, q + 1, 'Z', 'Z')
                                           for q in range(n - 1)]
    truncated = la.lie_closure(generators, max_dimension=60)
    certificate = la.full_su_certificate(truncated['basis'], n)

    assert not truncated['closed']
    assert truncated['dimension'] < 4 ** n - 1
    assert certificate['certified_full_su']
    assert certificate['implied_dimension'] == la.lie_closure(generators)['dimension']


def test_an_uncapped_closure_reports_itself_closed():
    assert la.lie_closure(all_local_rotations(2))['closed']


# --- gate generators -------------------------------------------------------

@pytest.mark.parametrize('name,letter', [('RX', 'X'), ('RY', 'Y'), ('RZ', 'Z')])
def test_rotation_generators_are_the_matching_pauli(name, letter):
    assert la.gate_generator(name, [1], 3) == {la.single(3, 1, letter): 1.0}


@pytest.mark.parametrize('name,letter', [('CRX', 'X'), ('CRY', 'Y'), ('CRZ', 'Z')])
def test_controlled_rotation_generators_match_the_projector_form(name, letter):
    """``CR(theta)`` is ``exp(-i theta |1><1|_c (x) H_t / 2)``, so its generator
    is ``(I - Z_c)/2 (x) H_t`` -- two Pauli terms, not one."""
    n = 2
    generated = la.sentence_matrix(la.gate_generator(name, [0, 1], n), n)
    projector = np.array([[0, 0], [0, 1]], dtype=complex)
    target = la.pauli_matrix(la.single(1, 0, letter), 1)
    assert np.allclose(generated, np.kron(projector, target))


def test_an_unknown_parameterised_gate_raises_rather_than_being_skipped():
    with pytest.raises(ValueError, match='no Pauli generator known'):
        la.gate_generator('IsingXX', [0, 1], 2)


def test_unitary_generator_reproduces_the_gate_it_came_from():
    import scipy.linalg as sla
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex)
    generator = la.unitary_generator(cnot, [0, 1], 2)
    reconstructed = sla.expm(-1j * la.sentence_matrix(generator, 2))
    # Fixed only up to the global phase the dropped identity term carries.
    ratio = reconstructed[0, 0] / cnot[0, 0]
    assert np.allclose(reconstructed, ratio * cnot)


def test_the_cnot_generator_is_the_expected_three_pauli_combination():
    n = 2
    generator = la.unitary_generator(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                 dtype=complex), [0, 1], n)
    assert set(generator) == {la.single(n, 0, 'Z'), la.single(n, 1, 'X'),
                              la.compose(la.single(n, 0, 'Z'), la.single(n, 1, 'X'))}


# --- conjugation -----------------------------------------------------------

def test_conjugation_by_cnot_matches_the_dense_computation():
    n = 3
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex)
    table = la.conjugation_table(cnot, 2)
    sentence = {la.from_string('XIZ'): 1.0, la.from_string('IYY'): -0.5}
    rotated = la.conjugate(sentence, table, [0, 2], n)

    dense_gate = np.eye(2 ** n, dtype=complex)
    # Build the same two-wire gate on wires (0, 2) densely, via a permutation of
    # the wire order, so the check does not reuse the code under test.
    import pennylane as qml
    dense_gate = qml.matrix(qml.CNOT(wires=[0, 2]), wire_order=[0, 1, 2])
    expected = dense_gate.conj().T @ la.sentence_matrix(sentence, n) @ dense_gate
    assert np.allclose(la.sentence_matrix(rotated, n), expected)


def test_conjugation_by_a_non_clifford_rotation_splits_a_term_in_two():
    """``RY(t)`` conjugation is what makes the propagated generators non-monomial;
    a Clifford would keep one term."""
    import pennylane as qml
    n = 2
    matrix = qml.matrix(qml.RY(0.02, wires=0), wire_order=[0])
    table = la.conjugation_table(matrix, 1)
    rotated = la.conjugate({la.single(n, 1, 'X'): 1.0}, table, [1], n)
    assert len(rotated) == 2
    expected = (qml.matrix(qml.RY(0.02, wires=1), wire_order=[0, 1]).conj().T
                @ la.sentence_matrix({la.single(n, 1, 'X'): 1.0}, n)
                @ qml.matrix(qml.RY(0.02, wires=1), wire_order=[0, 1]))
    assert np.allclose(la.sentence_matrix(rotated, n), expected)


def test_conjugation_preserves_the_closure_dimension():
    """Conjugating every generator by one unitary is an algebra isomorphism, so
    the dimension is invariant. This is the guard on the propagation step."""
    import pennylane as qml
    n = 3
    generators = all_local_rotations(n) + [coupling(n, 0, 1, 'Z', 'Z')]
    table = la.conjugation_table(qml.matrix(qml.CNOT(wires=[0, 1]),
                                            wire_order=[0, 1]), 2)
    rotated = [la.conjugate(g, table, [1, 2], n) for g in generators]
    assert (la.lie_closure(generators)['dimension']
            == la.lie_closure(rotated)['dimension'])
