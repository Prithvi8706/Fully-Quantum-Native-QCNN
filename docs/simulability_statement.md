# Classical simulability of the FQCNN at the studied scale

**Phase 3 item 3.7.** Drafted 2026-07-29 on branch `plan/fqcnn-q1-upgrade`. Prose only, no
compute. This is the source text for the scope statement that M10.3 installs in the
manuscript; it is **not** in `fqcnn.tex` yet, and must not be until M10.3.

**Updated 2026-07-31:** §4's open item on the dynamical Lie algebra is closed by 3.1. The
result does not change the statement — see that section for what it rules out and what it
does not.

Anchors: `STATUS.md` §3 (parameter audit), §12 (3.2, 3.4, 3.5), `UPGRADE_PLAN.md` 3.7.

---

## 1. The statement

The frozen headline model runs on **10 qubits**. Its state space is 2¹⁰ = 1,024 complex
amplitudes — 16 KB in `complex128`. It is therefore **exactly classically simulable, cheaply,
on a laptop**, and every result in this paper was produced that way. We say so at the top of
the scope discussion rather than leaving a reviewer to derive it.

Three properties make the simulation not merely possible but *easy*, and two of them are
architectural rather than incidental:

1. **Scale.** Ten qubits is small. No circuit on ten qubits is hard to simulate. This alone
   settles the question for the headline model and is independent of anything the
   architecture does.
2. **The main path is unitary end to end.** There is no mid-circuit measurement, no reset and
   no feed-forward (invariant A3; Theorem 1 is what permits the pooling block to be written
   this way). A pure-statevector simulator therefore suffices — no density matrix, no
   trajectory sampling. This is a property of the design, and it is the same property that
   makes the model differentiable by backpropagation.
3. **The circuit is shallow.** The frozen tape carries **327 operations total**, of which
   **222 are trainable gates** (`Results/evidence/t6_generalization_bound.json`).

## 2. Consequently: no quantum-advantage claim is made

This work **does not claim, and its evidence cannot support, a quantum computational
advantage.** Nothing here is faster, cheaper or more accurate than a classical method by
virtue of being quantum. The dataset makes this doubly clear: MNIST 0-vs-1 is ~99.8% linearly
separable, so a logistic-regression baseline is expected to be competitive with or better than
the model's 98.29% (`STATUS.md` §3e). Phase 5's baseline table will report that comparison
rather than avoid it.

What the work does claim is narrower and, we argue, worth a Q1 venue on its own terms:

- **An architecture family is validated at a scale where it can be checked exactly.** The
  central theoretical result (Theorem 1: unitary pooling exactly simulates the measure-and-
  condition channel) is verified to **2.2e-16** at the headline configuration. That
  verification is *possible only because the model is simulable* — the residual is read off a
  full state vector computed two ways. Simulability is the precondition for the evidence, not
  a weakness of it.
- **The theory is scale-independent even though the experiment is not.** Theorem 1 and
  Propositions 2–3 are statements about the pooling block's structure, proved for arbitrary
  pair count. Their empirical confirmations are at n ∈ {6, 8, 10}.
- **Trainability is measured, not assumed, over n = 4…14** (item 3.2), with the honest reading
  recorded in `STATUS.md` §12: **no barren plateau is observed to n = 14**, and the six-point
  data cannot identify the decay law or license any asymptotic claim in either direction.

## 3. Provenance: every number in this paper is a simulation

| Path | Device | Used for |
|---|---|---|
| Batched training and gradients | `default.qubit`, backprop | headline training, Phase 3 analyses |
| Sequential correctness oracle | `lightning.qubit`, adjoint | frozen reference path, M1.1 equivalence check |
| Channel / noise studies | `default.mixed` | E1 Kraus channel, E2 dephasing, `noise_sim.py` |

**There is no hardware result.** Phase 7's real-QPU point (M7.3) has not been run; the IBM Open
Plan allocation stands at 0 minutes consumed (`STATUS.md` §5). `experiments/hardware_run.py`
exists and targets `qiskit.remote`, but it has produced no evidence artifact. Any sentence in
the manuscript that could be read as describing device behaviour must be qualified as
simulation until M7 lands, and the noise ladder (F-C) is likewise ideal-simulator-only today.

One resource caveat travels with this: state preparation is not free. `AmplitudeEmbedding` at
n=10 decomposes into a ~2,026-CNOT Möttönen sequence (`STATUS.md` §7), which dominates the
circuit by every gate-count measure. On real hardware the state-prep cost, not the model, would
be the binding constraint. Quantifying that split is M8.1's job and it has not run — this note
records the observation, not a number.

## 4. What is deliberately *not* claimed here — scaling

`UPGRADE_PLAN.md` 3.7 pairs the simulability admission with "and characterises its scaling
(Phase 8)". **Phase 8 has not run.** The scaling sweep (M8.2, n ∈ {4, 6, 8, 10, 12, 14}) and the
resource table (T2/M8.1) are both at zero cells (`STATUS.md` §4). Until they exist, the
statement stops at the admission. Writing the second half now would install a claim with no
artifact behind it.

Two further open items bear directly on how strong the simulability admission ought to be:

- **3.1, the dynamical Lie algebra, has now run (2026-07-31), and it does not change the
  statement.** This bullet previously recorded the open risk that a polynomially-scaling DLA
  would imply the *family* is efficiently simulable at arbitrary n via the Lie-algebraic
  ("g-sim") results — a substantive finding that would have had to be reported rather than
  buried. It does not arise. The DLA of the frozen ansatz at the headline configuration is the
  **whole of su(2¹⁰), dimension 1,048,575**, by exact closure — not a truncated enumeration
  (`Results/evidence/t6_dynamical_lie_algebra.json`; `STATUS.md` §12).

  What follows, and what does not:

  - **The g-sim route is unavailable.** It requires the circuit to lie in `exp(g)` for a
    polynomially-sized `g`, and here `g` is the full special unitary algebra. So this paper
    makes **no** claim that the family is efficiently simulable at arbitrary n.
  - **That is not a hardness result, and must not be written as one.** An exponential DLA
    closes *one* efficient-simulation argument. It says nothing about the others, and the
    shallow fixed-depth tape (327 operations, three pooling stages) is exactly the regime where
    a tensor-network or low-entanglement simulation may well succeed. The honest position is
    that simulability at arbitrary n is **open**, with one specific route now ruled out.
  - **§1's admission is unaffected.** At ten qubits the model is simulable for the trivial
    reason given there — 1,024 amplitudes — and that reason never depended on the DLA.
- **3.4 found the ansatz Haar-indistinguishable at n=10** to within a measure that separates
  ~10× fidelity deviations. That is a statement about the output ensemble, not about
  simulation hardness, and it should not be recruited as evidence of "quantumness".

## 5. Parameter counts, stated precisely

3.7's spec says "~76-parameter". That figure is stale and the ambiguity behind it is itself a
finding of 3.5, so the statement uses the audited numbers and always names which quantity it
means:

| Quantity | Value | Note |
|---|---|---|
| Allocated parameter slots | **269** | overstates the circuit — 191 never reach the tape |
| Slots appearing on the frozen tape | **78** | syntactic use |
| **Effective slots** (\|grad\| > 1e-12) | **74** | M0 audit, reproduced independently by 3.2 |
| **Trainable gates on the tape** | **222** | the count Caro's bound takes as *T* |
| Gates driven by effective slots | 218 | 48 slots drive up to 4 gates each |
| Total operations on the tape | 327 | — |

The two families disagree in **both** directions — 269 overstates because most allocated slots
are dead, 74 understates because live slots are reused by the classifier's modular indexing —
so "the model has ~76 parameters" is not a safe shorthand anywhere in the manuscript. Where the
paper makes the parameter-frugality argument it must be explicit that it is a *parameter-count*
claim (269 → 74 effective) and must not attach it to the generalization bound, where the gate
reading collapses the gap to 222 vs 218, about 1% (`STATUS.md` §12, 3.5).

## 6. Draft manuscript paragraph

Suggested text for M10.3 to install in the scope/limitations discussion:

> All results reported here are obtained by exact classical simulation of the quantum circuit
> (`default.qubit` and `lightning.qubit` state-vector simulation; `default.mixed` for the
> channel and noise studies). At the studied scale of ten qubits the model's 1,024-amplitude
> state space is trivially tractable, and we make no claim of quantum computational advantage:
> a classical baseline on this task is expected to be competitive, and our evidence would not
> support such a claim if it were made. The contribution is a theoretical and empirical
> characterisation of an architecture family — in particular the exact equivalence of unitary
> and measurement-based pooling (Theorem 1), confirmed numerically to 2.2 × 10⁻¹⁶ — at a scale
> where every claim can be checked against an exactly computed state. We regard that
> checkability as a precondition for the evidence rather than a limitation of it. No hardware
> execution is reported; the resource and scaling characterisation of the family at larger
> qubit counts is left to future work.

The final clause is written as future work on purpose. If M8 lands before submission it should
be rewritten to point at the resource table and the scaling family; if M7 lands, the "no
hardware execution" sentence must be revised rather than deleted.
