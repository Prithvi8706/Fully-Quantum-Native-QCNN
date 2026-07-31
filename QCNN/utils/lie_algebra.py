"""Dynamical Lie algebra primitives (roadmap M3.1).

The dynamical Lie algebra (DLA) of an ansatz is the Lie closure of its gate
generators: the smallest subalgebra of su(2^n) containing them and closed under
commutation. Its dimension is the quantity 3.1 reports, because two separate
results are stated in terms of it:

  Ragone et al. (2024) / Fontana et al. (2024) -- exact variance expressions for
      loss gradients in terms of the DLA. A polynomially-sized DLA rules out a
      barren plateau; an exponentially-sized one predicts one.
  Somma et al. (2006) / Goh et al. (2023) -- Lie-algebraic ("g-sim") classical
      simulation. A polynomially-sized DLA whose group contains the circuit makes
      the family efficiently classically simulable at arbitrary n.

Both directions follow from the same number, which is why 3.1 must report both.

Representation
--------------
Elements are Hermitian operators written in the Pauli basis. A Pauli string is a
key ``(x, z)`` of two bitmasks -- bit ``q`` of each is qubit ``q`` -- standing for

    P(x, z) = (x) i^(x_q z_q) X^(x_q) Z^(z_q)      (per qubit)

so ``(0,0)=I``, ``(1,0)=X``, ``(0,1)=Z``, ``(1,1)=Y`` (since ``iXZ = Y``). A
"sentence" is a ``dict`` mapping such keys to **real** coefficients, i.e. a
Hermitian operator. The algebra element is ``i * H``; the bracket used
throughout is therefore ``<A, B> = i[A, B]``, which maps Hermitian pairs to
Hermitian results and matches the su(2^n) structure up to that factor.

Identity terms are dropped. The identity is central, so it changes no commutator,
and dropping it is what makes the algebra traceless -- ``exp`` of the difference
is a global phase.
"""
from __future__ import annotations

import numpy as np

# --- Pauli strings ---------------------------------------------------------

IDENTITY = (0, 0)

_LETTER_TO_KEY = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
_KEY_TO_LETTER = {v: k for k, v in _LETTER_TO_KEY.items()}

_SINGLE_MATRICES = {
    (0, 0): np.eye(2, dtype=complex),
    (1, 0): np.array([[0, 1], [1, 0]], dtype=complex),
    (0, 1): np.array([[1, 0], [0, -1]], dtype=complex),
    (1, 1): np.array([[0, -1j], [1j, 0]], dtype=complex),
}

# Default linear-dependence threshold. Coefficients here are O(1) -- every
# generator is normalised by its leading term during reduction -- so this sits
# far below any structurally meaningful value and far above float64 round-off.
TOL = 1e-9


def _popcount(value: int) -> int:
    return bin(value).count('1')


def single(n_qubits: int, wire: int, letter: str) -> tuple:
    """The Pauli string with ``letter`` on ``wire`` and identity elsewhere."""
    if not 0 <= wire < n_qubits:
        raise ValueError('wire {} outside 0..{}'.format(wire, n_qubits - 1))
    x, z = _LETTER_TO_KEY[letter.upper()]
    return (x << wire, z << wire)


def from_string(word: str) -> tuple:
    """``'XIZY' -> key``. Character ``q`` is qubit ``q`` (left-to-right)."""
    x = z = 0
    for wire, letter in enumerate(word):
        bx, bz = _LETTER_TO_KEY[letter.upper()]
        x |= bx << wire
        z |= bz << wire
    return (x, z)


def to_string(key: tuple, n_qubits: int) -> str:
    """Inverse of ``from_string``; the readable form used in evidence files."""
    x, z = key
    return ''.join(_KEY_TO_LETTER[((x >> q) & 1, (z >> q) & 1)]
                   for q in range(n_qubits))


def compose(*terms) -> tuple:
    """Product key of several Pauli strings, ignoring phase.

    Used to name a basis element (``compose(single(n,0,'Z'), single(n,1,'Y'))``),
    never inside arithmetic -- ``product_phase`` carries the phase there.
    """
    x = z = 0
    for tx, tz in terms:
        x ^= tx
        z ^= tz
    return (x, z)


def support(key: tuple) -> int:
    """Bitmask of the wires the string acts on non-trivially."""
    return key[0] | key[1]


def commute(a: tuple, b: tuple) -> bool:
    """Whether two Pauli strings commute (symplectic form even)."""
    return (_popcount(a[0] & b[1]) + _popcount(b[0] & a[1])) % 2 == 0


def product_phase(a: tuple, b: tuple) -> int:
    """``k`` with ``P_a P_b = i^k P_(a xor b)``.

    Per qubit ``P = i^(xz) X^x Z^z``, so moving ``X^x2`` left past ``Z^z1``
    contributes ``(-1)^(z1 x2)`` and the normalisation contributes the rest.
    """
    (x1, z1), (x2, z2) = a, b
    k = (_popcount(x1 & z1) + _popcount(x2 & z2)
         - _popcount((x1 ^ x2) & (z1 ^ z2)) + 2 * _popcount(z1 & x2))
    return k % 4


def pauli_matrix(key: tuple, n_qubits: int) -> np.ndarray:
    """Dense matrix of a Pauli string, qubit 0 most significant (PennyLane order)."""
    x, z = key
    out = np.array([[1.0 + 0.0j]])
    for q in range(n_qubits):
        out = np.kron(out, _SINGLE_MATRICES[((x >> q) & 1, (z >> q) & 1)])
    return out


def sentence_matrix(sentence: dict, n_qubits: int) -> np.ndarray:
    """Dense matrix of a Hermitian sentence. Tests only -- 4^n work."""
    dim = 2 ** n_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for key, coeff in sentence.items():
        out += coeff * pauli_matrix(key, n_qubits)
    return out


# --- the bracket -----------------------------------------------------------

def bracket(a: dict, b: dict, tol: float = TOL) -> dict:
    """``i[A, B]`` for Hermitian sentences ``A``, ``B``.

    Commuting Pauli pairs contribute nothing. An anticommuting pair gives
    ``PQ - QP = 2 PQ = 2 i^k R``, so the term is ``2 i^(k+1) R`` -- and ``k`` is
    odd exactly when the pair anticommutes, which is what keeps the coefficient
    real and the result Hermitian.
    """
    out = {}
    for pa, ca in a.items():
        for pb, cb in b.items():
            if commute(pa, pb):
                continue
            k = product_phase(pa, pb)
            # i^(k+1) with k odd is +-1: k=1 -> -1, k=3 -> +1.
            sign = -1.0 if k == 1 else 1.0
            key = (pa[0] ^ pb[0], pa[1] ^ pb[1])
            out[key] = out.get(key, 0.0) + 2.0 * sign * ca * cb
    return {k: v for k, v in out.items() if abs(v) > tol}


# --- linear algebra over the Pauli basis -----------------------------------

def _leading(vector: dict, tol: float):
    """Smallest key carrying a non-negligible coefficient, or ``None``."""
    lead = None
    for key, coeff in vector.items():
        if abs(coeff) > tol and (lead is None or key < lead):
            lead = key
    return lead


class _Echelon:
    """Incremental row reduction over the Pauli basis.

    Pivot rows are normalised and stored by leading key. Because every stored
    row's leading key is the smallest key it carries, eliminating a leading key
    can only introduce strictly larger ones, so reduction terminates.
    """

    def __init__(self, tol: float = TOL):
        self.tol = tol
        self.pivots = {}

    def reduce(self, vector: dict):
        """Reduced, normalised row, or ``None`` if already in the span."""
        work = {k: c for k, c in vector.items()
                if k != IDENTITY and abs(c) > self.tol}
        while True:
            lead = _leading(work, self.tol)
            if lead is None:
                return None
            pivot = self.pivots.get(lead)
            if pivot is None:
                scale = 1.0 / work[lead]
                return {k: c * scale for k, c in work.items()
                        if abs(c * scale) > self.tol}
            factor = work[lead]
            for key, coeff in pivot.items():
                updated = work.get(key, 0.0) - factor * coeff
                if abs(updated) <= self.tol:
                    work.pop(key, None)
                else:
                    work[key] = updated
            work.pop(lead, None)

    def add(self, vector: dict):
        """Reduce and store. Returns the stored row, or ``None`` if dependent."""
        reduced = self.reduce(vector)
        if reduced is None:
            return None
        self.pivots[_leading(reduced, self.tol)] = reduced
        return reduced


def span(vectors, tol: float = TOL) -> _Echelon:
    """Row-echelon form of a set of sentences, for repeated membership tests."""
    echelon = _Echelon(tol)
    for vector in vectors:
        echelon.add(vector)
    return echelon


def _as_span(basis_or_span, tol: float = TOL) -> _Echelon:
    return basis_or_span if isinstance(basis_or_span, _Echelon) else span(basis_or_span, tol)


def span_dimension(vectors, tol: float = TOL) -> int:
    """Rank of a set of sentences over the Pauli basis."""
    return len(span(vectors, tol).pivots)


# --- the closure -----------------------------------------------------------

def lie_closure(generators, max_dimension: int = None, tol: float = TOL,
                keep_basis: bool = True) -> dict:
    """Lie closure of ``generators`` under ``i[.,.]``, by iterated commutators.

    Iterates ``B <- B + [S, B]`` where ``S`` is a basis of the generator span.
    That suffices: the Jacobi identity rewrites any nested commutator as a
    combination of left-nested ones over the generating set, so bracketing new
    elements against the generators alone reaches the whole closure.

    ``max_dimension`` bounds the work. Hitting it returns ``closed=False`` and
    the dimension must then be quoted as a lower bound -- a truncated closure is
    not a dimension.
    """
    echelon = _Echelon(tol)
    seeds = []
    for generator in generators:
        stored = echelon.add(generator)
        if stored is not None:
            seeds.append(stored)

    basis = list(seeds)
    frontier = list(seeds)
    rounds = 0
    truncated = False

    while frontier and not truncated:
        rounds += 1
        discovered = []
        for element in frontier:
            for seed in seeds:
                child = bracket(element, seed, tol)
                if not child:
                    continue
                stored = echelon.add(child)
                if stored is None:
                    continue
                basis.append(stored)
                discovered.append(stored)
                if max_dimension is not None and len(basis) >= max_dimension:
                    truncated = True
                    break
            if truncated:
                break
        frontier = discovered

    result = {
        'dimension': len(basis),
        'closed': not truncated,
        'commutator_rounds': rounds,
        'n_generators': len(generators),
        'n_independent_generators': len(seeds),
    }
    if keep_basis:
        result['basis'] = basis
        # The echelon is already built; handing it back keeps the membership
        # tests in ``full_su_certificate`` from re-reducing a million rows.
        result['span'] = echelon
    return result


def support_components(basis, n_qubits: int) -> list:
    """Wire sets the algebra couples, as sorted lists.

    Two wires land in the same component when some basis element acts on both.
    A DLA that is a direct sum of full ``su(2^|C|)`` blocks -- which is what a
    connected entangling graph plus local rotations produces -- then has
    dimension ``sum(4^|C| - 1)``, so this decomposition is the cheap structural
    check on a dimension that would otherwise be an unexplained integer.
    """
    parent = list(range(n_qubits))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    touched = set()
    for element in basis:
        wires = set()
        for key in element:
            mask = support(key)
            wires.update(q for q in range(n_qubits) if (mask >> q) & 1)
        touched.update(wires)
        wires = sorted(wires)
        for wire in wires[1:]:
            union(wires[0], wire)

    groups = {}
    for wire in sorted(touched):
        groups.setdefault(find(wire), []).append(wire)
    return sorted(groups.values())


def direct_sum_of_full_blocks(components) -> int:
    """``sum(4^|C| - 1)``: the dimension if each component were full su(2^|C|)."""
    return sum(4 ** len(component) - 1 for component in components)


def contains(basis_or_span, vector: dict, tol: float = TOL) -> bool:
    """Whether ``vector`` lies in the span. Pass a ``span`` to reuse the echelon."""
    return _as_span(basis_or_span, tol).reduce(vector) is None


def full_su_certificate(basis_or_span, n_qubits: int, tol: float = TOL,
                        components=None) -> dict:
    """Sufficient structural condition for the closure to be all of su(2^n).

    If an algebra contains su(2) on every wire and its support graph is
    connected across all wires, it is the whole of su(2^n) (the standard
    universality result: local control plus one entangling coupling on a
    connected graph is universal). This is a *sufficient* condition, not a
    definition, and it exists for one purpose: at qubit counts where enumerating
    ~4^n basis elements is out of reach, it names the exact dimension that the
    capped enumeration can only bound from below. Where both are available --
    n <= 10 here -- they are reported together and must agree.

    Validated against exact closure for n <= 5 in ``tests/test_lie_algebra.py``.
    """
    echelon = _as_span(basis_or_span, tol)
    missing = []
    for wire in range(n_qubits):
        for letter in 'XYZ':
            if not contains(echelon, {single(n_qubits, wire, letter): 1.0}, tol):
                missing.append('{}{}'.format(letter, wire))

    if components is None:
        if isinstance(basis_or_span, _Echelon):
            raise ValueError('pass components explicitly when given a span')
        components = support_components(basis_or_span, n_qubits)
    connected = len(components) == 1 and len(components[0]) == n_qubits
    certified = connected and not missing
    return {
        'has_local_su2_on_every_wire': not missing,
        'missing_local_generators': missing,
        'support_is_connected': connected,
        'certified_full_su': certified,
        'implied_dimension': (4 ** n_qubits - 1) if certified else None,
    }


# --- gate generators -------------------------------------------------------

# Rotation gates as exp(-i theta H / 2). Only H matters here: the closure is a
# property of the span, so the factor of 1/2 is dropped.
_ROTATION_LETTER = {'RX': 'X', 'RY': 'Y', 'RZ': 'Z'}

# Controlled rotations: exp(-i theta |1><1|_c (x) H_t / 2), and
# |1><1| = (I - Z)/2, so the generator is (H_t - Z_c H_t) / 2.
_CONTROLLED_LETTER = {'CRX': 'X', 'CRY': 'Y', 'CRZ': 'Z'}


def gate_generator(name: str, wires, n_qubits: int) -> dict:
    """Hermitian generator of a parameterised gate, in the Pauli basis.

    Raises on anything unrecognised. Silently skipping an unknown parameterised
    gate would understate the algebra, which is the direction that manufactures
    a favourable trainability result.
    """
    if name in _ROTATION_LETTER:
        if len(wires) != 1:
            raise ValueError('{} expects one wire, got {}'.format(name, wires))
        return {single(n_qubits, int(wires[0]), _ROTATION_LETTER[name]): 1.0}

    if name in _CONTROLLED_LETTER:
        if len(wires) != 2:
            raise ValueError('{} expects two wires, got {}'.format(name, wires))
        control, target = int(wires[0]), int(wires[1])
        letter = _CONTROLLED_LETTER[name]
        target_key = single(n_qubits, target, letter)
        control_key = single(n_qubits, control, 'Z')
        return {target_key: 0.5, compose(control_key, target_key): -0.5}

    raise ValueError(
        "no Pauli generator known for gate '{}'; add it rather than skipping "
        "it -- an omitted generator understates the DLA".format(name))


def local_pauli_expansion(matrix: np.ndarray, n_local: int,
                          tol: float = 1e-12) -> dict:
    """Expand a ``2^k x 2^k`` matrix in the Pauli basis of its ``k`` wires."""
    dim = 2 ** n_local
    if matrix.shape != (dim, dim):
        raise ValueError('matrix shape {} is not 2^{}'.format(matrix.shape, n_local))
    out = {}
    for x in range(dim):
        for z in range(dim):
            key = (x, z)
            coeff = np.trace(pauli_matrix(key, n_local).conj().T @ matrix) / dim
            if abs(coeff) > tol:
                if abs(coeff.imag) > 1e-8:
                    raise ValueError('non-Hermitian matrix: coefficient {} on {}'
                                     .format(coeff, to_string(key, n_local)))
                out[key] = float(coeff.real)
    return out


def _embed(local_key: tuple, wires, n_qubits: int) -> tuple:
    """Lift a key over ``len(wires)`` local wires onto ``n_qubits`` wires."""
    lx, lz = local_key
    x = z = 0
    for position, wire in enumerate(wires):
        x |= ((lx >> position) & 1) << wire
        z |= ((lz >> position) & 1) << wire
    return (x, z)


def _restrict(key: tuple, wires, n_qubits: int):
    """Split a key into (local key over ``wires``, remainder key)."""
    x, z = key
    lx = lz = 0
    rx, rz = x, z
    for position, wire in enumerate(wires):
        lx |= ((x >> wire) & 1) << position
        lz |= ((z >> wire) & 1) << position
        rx &= ~(1 << wire)
        rz &= ~(1 << wire)
    return (lx, lz), (rx, rz)


def conjugation_table(unitary: np.ndarray, n_local: int) -> dict:
    """``{P: U^dag P U}`` for every Pauli string on the gate's own wires.

    Built once per distinct fixed gate. Conjugation of a full string then costs
    one table lookup, because ``U`` acts only on those wires and the per-qubit
    phase convention makes the tensor split exact.
    """
    dim = 2 ** n_local
    table = {}
    for x in range(dim):
        for z in range(dim):
            key = (x, z)
            rotated = unitary.conj().T @ pauli_matrix(key, n_local) @ unitary
            table[key] = local_pauli_expansion(rotated, n_local)
    return table


def conjugate(sentence: dict, table: dict, wires, n_qubits: int,
              tol: float = TOL) -> dict:
    """``U^dag H U`` for the fixed gate whose ``conjugation_table`` is given."""
    out = {}
    for key, coeff in sentence.items():
        local, remainder = _restrict(key, wires, n_qubits)
        for local_image, image_coeff in table[local].items():
            new_key = compose(remainder, _embed(local_image, wires, n_qubits))
            out[new_key] = out.get(new_key, 0.0) + coeff * image_coeff
    return {k: v for k, v in out.items() if abs(v) > tol}


def unitary_generator(unitary: np.ndarray, wires, n_qubits: int,
                      tol: float = 1e-12) -> dict:
    """A Hermitian ``H`` on ``wires`` with ``exp(-iH) = U``, in the Pauli basis.

    Used for the *fixed* gates of the frozen tape (the CNOTs and the constant
    ``RY(0.02)``). Which branch of the logarithm is taken does not matter: any
    ``H`` satisfying the relation puts the gate inside ``exp(g)``, which is the
    only property the simulability argument uses.
    """
    values, vectors = np.linalg.eig(np.asarray(unitary, dtype=complex))
    logarithm = vectors @ np.diag(1j * np.log(values)) @ np.linalg.inv(vectors)
    hermitian = 0.5 * (logarithm + logarithm.conj().T)
    if not np.allclose(logarithm, hermitian, atol=1e-8):
        raise ValueError('i log(U) is not Hermitian; U may not be unitary')

    n_local = len(wires)
    local = local_pauli_expansion(hermitian, n_local, tol=tol)
    out = {}
    for local_key, coeff in local.items():
        if local_key == IDENTITY:
            continue
        out[_embed(local_key, wires, n_qubits)] = coeff
    return out
