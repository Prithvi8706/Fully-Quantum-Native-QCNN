# FQCNN — Q1 Upgrade Plan (v2)

**Status:** planning document only. Nothing here has been executed.
**Target:** raise `fqcnn.tex` + this repo to Q1-journal level (IEEE TQE, Quantum Machine
Intelligence, EPJ Quantum Technology, Phys. Rev. Applied class).
**Written:** 2026-07-22 (supersedes v1).
**Constraints:** gaming laptop (multi-core CPU), IBM Quantum free Open Plan (~10 min QPU/month).

**v2 changes vs v1:** the plan now operates under an **architecture freeze** — the proposed
FQCNN circuit is immutable and every stage of the main pipeline stays strictly unitary. The
pooling-block redesign that v1 recommended is demoted to an *opt-in ablation arm* that never
replaces the headline model. A new model-analysis phase (Lie-algebra trainability, effective
dimension, generalization bounds, expressibility) is added, the coherence theory is reframed so
the equivalence result becomes a *feature* of the paper instead of a threat, and the
implementation guidance is sharpened (freeze-enforcement tests, caching, Aer noise path,
environment-lock repair).

---

## §A — GROUND RULES FOR THE EXECUTING AGENT (read before every phase)

These rules bind every task in this plan. They are not suggestions.

**A1. ARCHITECTURE FREEZE.** The proposed FQCNN — encoding → convolution → unitary pooling →
variational classifier, exactly as executed by `QCNNModel._pure_quantum_forward` today — is the
paper's model and **must not be changed**. No gates added, removed, reordered, or reparameterised
in the headline model. Do not "fix" the layer schedule, do not restructure pooling, do not touch
the classifier head. Where the code and the paper disagree, **the paper moves to match the code**,
never the reverse.

**A2. Freeze = observational equivalence, enforced by tests.** The protected object is the
function `(x, θ) ↦ ⟨Z_readout⟩`, not incidental bytes. Before any refactor, create and commit two
guard tests; every subsequent change must pass both:
  - **Circuit fingerprint:** build the tape at fixed dummy parameters, serialise
    `(gate name, wires, parameter-slot index)` for every operation, hash it, and assert the hash
    matches the committed reference. Catches accidental structural edits.
  - **Expectation regression:** 20 fixed random inputs × the archived trained weights
    (`Results/Weights/quantum_model_params.npz`) → assert `⟨Z⟩` values match committed references
    to 1e-10. Catches semantic drift through any refactor (e.g. the single-source-of-truth
    consolidation in Phase 0.4).
  The *only* permitted circuit edit is removal of a gate **proven inert** (see F4, the `RY(0.02)`
  on a discarded wire): allowed only with a written proof in the commit message *and* a passing
  expectation-regression test at 1e-12.

**A3. UNITARITY INVARIANT.** Every operation in the main pipeline remains unitary end-to-end:
no mid-circuit measurement, no reset, no classical feed-forward, no non-unitary channel ever
enters the proposed model. The global 10-qubit state stays pure from state preparation to the
single terminal measurement. Measurement-based pooling exists in this repo **only** as a
clearly-labelled ablation baseline (`pool_measurement`), never as part of the proposed model.
Noise channels appear **only** in evaluation-time robustness studies, never in the model
definition.

**A4. Variants are arms, not replacements.** Any circuit variant this plan introduces
(`pool_coherent`, encoding ablations, reduced-qubit hardware instance, scaling instances) is an
*experiment arm* living in `experiments/` config space, clearly named, and reported as an
ablation/extension. The headline row of every table is the frozen circuit.

**A5. Anything that would touch the architecture requires explicit user sign-off first.** Known
items in this category are flagged `[ARCH — opt-in]` below (multi-class head, promotion of
`pool_coherent`). Default is: do not do them.

**A6. Honesty invariant.** Never report a number produced under test-set leakage, never count a
parameter that receives no gradient as "trainable" without disclosure, never present another
paper's accuracy on a different split as a same-table comparison. When a fixed protocol lowers
the headline accuracy, report the lower number.

---

## §B — Findings that drive the plan

From reading the code against `fqcnn.tex`. Each is reviewer-visible. Under the freeze, each is
resolved by *protocol, disclosure, or paper edits* — not by circuit surgery.

### F1 — 72% of the "269 trainable parameters" receive zero gradient
`QCNNModel.py:118-135` skips convolution below 4 active qubits, and `get_conv_windows` returns
nothing for a 5×1 grid. Traced schedule at n=10: layer 0 convolves (5×2 grid, 4 windows), layers
1–3 never apply their kernels. `quantum_conv_kernel_{1,2,3}` = 144 dead parameters; pool layers
1–2 use 6 and 3 of 15 angles; the classifier uses 4 of 32. **Effective trainable count ≈ 76, and
there is one convolution stage, not a 4-level hierarchy.**
*Freeze-compliant resolution:* the paper describes the executed circuit — one convolution stage,
two pooling stages, the true parameter count — and the "hierarchical" language is scoped to the
pooling cascade. Keep the parameter *vector* at 269 slots (removing slots would shift the seeded
RNG stream and break reproduction of the archived run) but report **269 allocated / ~76
effective**, verified by a gradient-audit test (Phase 0.2). The deeper hierarchy becomes a
*scaling-family instance* at n=16 (Phase 8), reported as scaling evidence, not as the headline
model.

### F2 — One qubit is discarded with no pooling unitary
`QPool.make_pairing` drops the last wire of an odd active set: at 5 active qubits, qubit 8 leaves
the active set with no information-transfer gate. *Resolution:* disclose in the paper as part of
the pooling definition (an unpaired wire is retired by exclusion from subsequent operations; on
the still-pure global state this is deferred bookkeeping, not an extra operation). The
architecture is what it is; the description must match it.

### F3 — Model selection and early stopping run on the test set  **(most urgent)**
`Qtrainer.py:168-196`: `test_accuracy` drives checkpointing, LR plateau, and early stopping. The
98.86% is a best-of-24-epochs *test* score. The paper's "Validation accuracy 98.7%" has no
validation split behind it anywhere in the pipeline. This alone sinks the paper at a Q1 venue.
*Resolution:* Phase 0.3 — protocol change only, zero circuit change.

### F4 — The paper describes a circuit the code does not run
- Sec. III-B / VI-D two-stage encoding (`Ry Rz H` + ring `CNOT·Rz·CNOT`): **never executed** on
  the amplitude path — `AmplitudeEmbedding` only. Listed as contribution #4.
- Sec. III-E classifier (`n_r+1` params): code runs a 32-slot RX/RY/RZ block with modular reuse.
- Sec. III-D pooling `CRY·CRZ·RY` vs Sec. VI-B `CRZ·CRY·CRZ` vs code `CRY, CRZ, RY(0.02) on
  discard, RY on keep` — three different definitions of the paper's own contribution.
- Sec. IV-C parameter-shift: simulation actually uses adjoint differentiation
  (`diff_method='best'` on `lightning.qubit`).
- Sec. V dataset numbers (11,430 / 8,001 / 3,429) vs `training_log.txt` (12,665 / 8,865 / 3,800).
*Resolution:* Phase 0.4 reconciliation table; the paper is rewritten to the executed circuit.
Contribution #4 is deleted or re-scoped to what runs (an *encoding-augmentation arm* may test the
two-stage map as an ablation — new gates in an arm, not in the frozen model).

### F5 — "Spatial locality" is not spatial under amplitude encoding
Under global amplitude encoding a qubit indexes a bit-plane of the flattened pixel address, not a
pixel neighbourhood; "2×2 window" and "translational equivariance over the image" are unsupported
as written. *Freeze-compliant resolution:* rewrite the claim as locality **in register/index
space** (bounded entanglement spread, constant per-window depth, weight sharing over windows —
all true and all that Pesah et al.'s trainability argument needs), and *empirically test* the
image-equivariance question (Phase 3.6: shift the input, measure output drift) so the paper
reports what the inductive bias actually is instead of asserting one it doesn't have.

### F6 — The coherence claim, as stated, is vacuous for this pooling block — but the fix is a theorem, not a redesign
The pooling block is controlled rotations **diagonal in the discarded qubit's basis**, and local
unitaries on wires that are never used again. By deferred measurement, the retained register's
state — and hence every downstream observable — is *identical* to measure-and-condition pooling:
`Tr_b[VρV†] = Σ_m U_m ⟨m|ρ|m⟩ U_m†`. So "measurement pooling irrevocably loses information that
we preserve" (Abstract, Secs. I, VI-A) is false *for observables of this circuit*, and the
existing `pool_measurement` ablation is predicted to tie exactly, not to lose.
**Under the freeze this becomes the paper's best asset** (Phase 2): state it as a *simulation
theorem* — unitary pooling exactly reproduces every statistic of measurement-based pooling while
requiring **no mid-circuit measurement, no reset, no classical feed-forward** — which is precisely
the "theoretical justification vs measurement-based pooling" the faculty asked for. What is
genuinely, provably true of the frozen model and must become the claim: (i) the global state
stays pure through the entire pipeline (full unitarity — the property to be maintained), (ii)
hardware-minimal control requirements, (iii) end-to-end adjoint differentiability. What must be
deleted: "information loss avoided."

### F7 — Noise study omits the dominant cost and duplicates the circuit
`noise_sim.py` applies one round of 1-qubit noise after `AmplitudeEmbedding`, but Möttönen state
preparation at n=10 costs **2^{n+1}−2n−2 = 2,026 CNOTs** — ~40× the rest of the circuit. The file
also re-implements the forward pass by hand (drift risk). *Resolution:* Phase 7; evaluation-layer
fix only.

### F8 — The experimental harness has never actually been run
`Results/experiments/summary.csv` is `--quick` smoke output (2 seeds, 60 samples, 2 epochs;
`proposed` at 48% — below chance). The harness is well built; the study doesn't exist yet. The
classical CNN baseline (`build_baseline_cnn`) exists but is never called.

### F9 — The lock file lies about the environment  **(verified)**
Installed and used: **Python 3.9.13, PennyLane 0.38.0, numpy 1.26.4, scikit-learn 1.6.1.**
`requirements-lock.txt` claims: Python 3.14.5, PennyLane 0.44.0, numpy 2.4.2, scikit-learn 1.8.0.
A reviewer following the lock file gets a different simulator than produced the results.
*Resolution:* regenerate the lock from the real environment (`pip freeze`), record the true
Python version, and state simulator/version in the paper's reproducibility paragraph.

---

## §C — Phase overview

| Phase | What | Blocking? | Est. effort |
|---|---|---|---|
| 0 | Freeze guards + protocol integrity (no circuit changes) | **yes** | 3–4 d |
| 1 | Simulator throughput (10–50×) | **yes** | 2–3 d |
| 2 | Pooling theory: simulation theorem + strict-extension headroom | yes for the story | 4–6 d |
| 3 | Model-analysis suite (trainability, capacity, generalization) | no | 4–6 d |
| 4 | Harder datasets | no | 3–4 d + compute |
| 5 | Baselines + statistical rigor | no | 4–5 d + compute |
| 6 | Ablation grid | no | 2 d + compute |
| 7 | Noise realism + real hardware | no | 4–6 d |
| 8 | Resources + scaling family | no | 3–4 d |
| 9 | References | no | 1–2 d |
| 10 | Paper rewrite + journal targeting | last | 5–7 d |

Critical path: 0 → 1 → 2. Phases 3–8 are queueable compute. Total ≈ 6–8 focused weeks.

---

## Phase 0 — Freeze guards and protocol integrity (no circuit changes)

**0.1 Commit the freeze guards first** (§A2): circuit-fingerprint test + expectation-regression
test against the archived weights. Nothing else in this plan may start before these pass in CI
(or a local `pytest` gate wired into `reproduce.sh`).

**0.2 Gradient audit as a permanent test.** Compute `qml.grad` at 20 random inputs; record which
parameter slots are identically zero across all of them; assert the *effective* set matches the
committed reference (≈76 slots). This converts F1 from a hidden defect into a disclosed,
regression-guarded property. The paper reports both numbers with one sentence of explanation.

**0.3 Introduce a real validation split.** `train/val/test = 60/15/25`, stratified, seeded.
Checkpointing, LR plateau, and early stopping read **val only**; test is evaluated exactly once
per run at the end. Touch points: `Qtrainer.train_pure_quantum_cnn` signature,
`experiments/run_experiments.py:prepare_split`, `main.py`.
→ *Verify:* grep the trainer for `X_test`/`y_test` — must appear only in the final evaluation
block. Then retrain the headline configuration once under the clean protocol; this number
replaces 98.86% everywhere. Expect it to be somewhat lower; report it anyway (§A6).

**0.4 Single source of truth for the circuit.** Extract `_pure_quantum_forward` into
`QCNN/circuits.py::build_circuit(params, cfg, noise_hooks=None)` consumed by `QCNNModel`,
`noise_sim.py`, and `experiments/hardware_run.py`; delete the hand-copied topology in
`noise_sim.py:245-328`. The expectation-regression test (0.1) is the proof the refactor is
behavior-preserving. `noise_hooks` is an evaluation-layer injection point (per-gate channels) so
noise studies stop needing a parallel circuit definition — the model definition itself never
gains a channel (§A3).

**0.5 Paper↔code reconciliation table.** One row per equation in Secs. III–IV: paper says / code
does / paper edit required. All resolutions edit the paper (A1). Specific decisions:
  - Encoding: delete the two-stage map from the main-model description; optionally keep it as
    ablation arm `enc_augmented` (Phase 6) to test whether it *would* help.
  - Classifier: describe the 32-slot block and its modular indexing honestly (with effective
    counts from 0.2).
  - Pooling: one canonical definition matching the code; the inert `RY(0.02)` is either proven
    inert and removed under §A2's exception, or kept and documented as a no-op on discarded
    wires. Recommend removal-with-proof — a magic constant in the paper's core contribution
    invites questions.
  - Gradients: simulation uses adjoint differentiation; parameter-shift is presented as the
    hardware-execution path with its `2|Θ_eff|` cost stated (Phase 8).
  - Dataset numbers: regenerate from the actual pipeline under the new split protocol.

**0.6 Reproducibility repairs.** Regenerate `requirements-lock.txt` from the live environment
(F9); pin the seed policy in one place; delete only *non-circuit* dead code
(`quantum_loss_function` MSE duplicate, `quantum_model_params(old).npz`, `output.txt`);
make `reproduce.sh` run the freeze guards first.

---

## Phase 1 — Make the experiments affordable

The archived run: ≈78,682 s (~22 h) for 24 epochs, one config. A Q1 section needs 150–300 runs.
The bottleneck is a per-sample Python loop, not physics.

**1.1 Batched backprop (the big one).** `Qtrainer.py:128-131` loops sample-by-sample, with a
comment claiming `AmplitudeEmbedding` can't broadcast. But `baselines/quantum_baselines.py:193-208`
*in this same repo* already differentiates an entire batch in one statevector pass through the
same `PureQuantumEncoder.amplitude_encoding` (`default.qubit`, `diff_method="backprop"`). Port
that execution path to the trainer. The circuit is untouched (§A2 tests prove it); only the
execution strategy changes. Expected 10–50× on training throughput.
→ *Verify:* batched vs per-sample outputs agree to 1e-10 on identical parameters; then benchmark.

**1.2 Cache what never changes.** Amplitude vectors are parameter-independent: precompute the
padded, L2-normalised 1024-dim vectors once per dataset and feed `qml.StatePrep`-ready arrays;
stop re-normalising every epoch. Cache per-epoch evaluation on a fixed val subset instead of the
full test sweep the trainer currently does each epoch.

**1.3 Parallelism + resume.** Runs are embarrassingly parallel over (dataset × config × seed).
Add `--jobs N` (physical cores − 2, `OMP_NUM_THREADS=1` per worker) and make the runner skip any
`(config, seed)` whose `seed_<s>.json` already exists, so overnight sweeps survive interruption.

**1.4 Budget before running.** `experiments/estimate_cost.py`: time one epoch per config, print
projected wall-clock for the full grid. Gate: **the grid must fit in ≤ 7 nights unattended.**
Cut seeds before datasets, datasets before the pooling-theory arms.

**1.5 Optional accelerators (only if 1.1–1.3 fall short):** JAX interface + `jax.jit`/`vmap` on
`default.qubit`; `lightning.gpu` only matters at n ≥ 14 (Phase 8 sweep).

---

## Phase 2 — Pooling theory: the defining contribution, architecture untouched

Faculty: *"strengthen its theoretical justification and experimentally compare it against
measurement-based pooling — this could become the paper's defining contribution."* Under the
freeze, the contribution is a theorem pair plus the decisive experiments — no gate of the
headline model changes.

### 2.1 New paper section: "Unitary vs measurement-based pooling" (~1.5 pages)

**Theorem 1 (Exact simulation — the justification faculty asked for).** Every measurement-based
pooling step "measure discarded qubit *b* in the computational basis; apply U_m to kept qubit
*a* on outcome m" is *exactly* reproduced — in every observable statistic on the retained
register — by the unitary block `V = Σ_m |m⟩⟨m|_b ⊗ U_m` with *b* simply left idle afterwards
(deferred-measurement principle; cite Nielsen & Chuang). Consequences, all favorable and all
true of the frozen model:
  1. Coherent unitary pooling **loses nothing** relative to measurement pooling — it simulates
     the entire class at zero overhead.
  2. It does so while eliminating mid-circuit measurement, qubit reset, and classical
     feed-forward — capabilities that are limited, slow (µs-scale vs ns-scale), or absent on
     current devices. The advantage is *architectural and hardware-real*, and now *proved* rather
     than asserted.
  3. The global state remains pure through the entire pipeline — the literal, defensible meaning
     of "fully coherent," and the property this plan is required to maintain.

**Proposition 2 (Strict containment / headroom).** The family of unitary pooling channels
`ρ_a ↦ Tr_b[VρV†]`, V ∈ SU(4), *strictly contains* the measure-and-condition class. Witness:
`V = SWAP` maps ρ_a ← ρ_b including off-diagonal coherences, which no measure-and-condition map
can produce (its output depends on ρ_b only through diagonal elements). So unitary pooling ⊋
measurement pooling as a *family*; the frozen FQCNN instantiation sits in the equivalence
subclass (its controls are diagonal in b's basis), and the headroom of the strict extension is
an empirical question the ablation answers (2.3).

**Proposition 3 (Coherence transfer functional).** Define
`Δ_coh(block) = ‖ρ_a^V − ρ_a^{deph_b→V}‖₁` — the trace-distance effect of fully dephasing the
discarded qubit before pooling, at identical parameters. Theorem 1 ⇒ `Δ_coh ≡ 0` for the frozen
block: an *a priori prediction the experiments then confirm*, which is exactly how a Q1 paper
earns trust. For non-diagonal V, `Δ_coh > 0`.

Framing note for the writer: no prior QCNN paper in the citation list states Theorem 1 about its
own pooling; being the paper that *clarifies a field-wide ambiguity* — and then measures the
headroom — is a genuine contribution. The claim to delete everywhere: "measurement pooling
irrevocably loses information that we preserve." The claim to install: "unitary pooling
provably subsumes measurement pooling at zero cost and strictly extends it; FQCNN realises the
subsuming class with minimal hardware requirements."

### 2.2 Experiment arms (frozen model + labelled variants)

| Arm | Block | Params/pair | Role |
|---|---|---|---|
| `pool_measurement` | measure b, conditioned RY on a | 1–3 | baseline; predicted to **tie exactly** (Thm 1) |
| `pool_unitary` | **frozen FQCNN block** | 3 | headline model |
| `pool_coherent` `[ARCH — opt-in]` | + one `CRY_{a→b}` (non-diagonal in b) | 4 | strict-extension headroom probe; **ablation arm only**, still fully unitary; never promoted to headline without user sign-off |
| `pool_su4` | general two-qubit SU(4) | 15 | expressivity ceiling |
| `pool_none` | wire retired, no gate | 0 | floor |

This also answers "is the pooling layer theoretically optimal?" with a bracket instead of a yes:
Theorem 1 gives optimality *within* the measurement-simulable class at minimum gate cost; the
SU(4) row measures the distance to the unconstrained ceiling at 5× the parameters.

### 2.3 Decisive experiments

- **E1 — Equivalence validation (run first; nearly free).** Same trained parameters,
  `pool_unitary` vs `pool_measurement` on `default.mixed`: outputs must agree to ~1e-12. This
  simultaneously validates Theorem 1, the harness, and the mid-circuit-measurement code path.
  Report it — a theory-predicted exact tie, confirmed to machine precision, is a credibility
  anchor.
- **E2 — Fixed-parameter dephasing ablation (the money figure).** Insert a full dephasing
  channel on each discarded wire immediately before pooling, parameters unchanged. Predicted:
  Δacc = 0 for the frozen block (Prop 3), Δacc > 0 for `pool_coherent`/`pool_su4`. Isolates the
  operational value of coherence with zero optimisation confound.
- **E3 — Retrained head-to-head.** All five arms from scratch, ≥5 seeds, ≥3 datasets, matched
  budgets. Mean ± std, paired tests (Phase 5.3). Answers "does coherence preservation measurably
  improve accuracy?" with numbers for the *family*, while the headline model's virtue rests on
  Theorem 1 + hardware minimality — a two-legged story that stands whichever way E3 falls.
- **E4 — Information dynamics.** Per stage: ℓ1-coherence, purity, von Neumann entropy of the
  retained register; mutual information between kept and discarded sub-registers. Mechanism
  figure to accompany the outcome figures.
- **E5 — Hardware-practicality quantification.** Transpile each arm for a heavy-hex IBM backend:
  mid-circuit-measurement/feed-forward support, two-qubit depth, and estimated per-shot latency
  including measurement+reset (µs) vs gates (ns). Substantiates Theorem 1's consequence 2 with
  device numbers.

---

## Phase 3 — Model-analysis suite (new in v2; all read-only analyses of the frozen circuit)

Cheap on compute, high on Q1 signal — this is the "quantitative evidence" layer faculty asked
for, and none of it touches a gate.

**3.1 Dynamical Lie algebra (DLA) dimension.** Compute the dimension of the Lie closure of the
frozen ansatz's generators (iterated commutators to numerical closure; feasible at these sizes).
Polynomially-sized DLA ⇒ no barren plateau by the Ragone et al. (2024) / Fontana et al. exact
variance expressions — upgrading the paper's citation of Pesah et al. from analogy to a
*computed certificate for this exact circuit*. Very few QML papers do this; it reads as serious.

**3.2 Empirical gradient variance.** `Var[∂L/∂θ]` over ~200 random initialisations at each
n ∈ {4,6,8,10,12,14} (scaling instances, Phase 8). Log-scale plot vs n next to the DLA result:
theory + experiment on trainability in one figure pair.

**3.3 Effective dimension** (Abbas et al., Nat. Comput. Sci. 2021) of the frozen model vs the
parameter-matched MLP and CNN baselines, from the empirical Fisher information. Directly
addresses "quantitative evidence supporting the claimed advantages."

**3.4 Expressibility & entangling capability** (Sim et al. 2019 — already `ref50`): KL
divergence from Haar for the ansatz's state ensemble, and Meyer–Wallach Q. Report alongside
`cong`/`hur`/`ttn` baselines: positions FQCNN on the standard expressivity map with the metrics
the field already accepts.

**3.5 Generalization bound** (Caro et al., Nat. Commun. 2022): bound scales ~√(T/N) with T
trainable gates. With T_eff ≈ 76 and N ≈ 8k, the bound is *non-vacuous* — a rare, quotable
theoretical guarantee: "the architecture's parameter frugality yields a provable generalization
bound of ~X%." Half a page, one table row, big reviewer impact.

**3.6 Inductive-bias verification (resolves F5 honestly).** Empirically test image-translation
sensitivity: shift test digits by 1–2 px, measure output drift for FQCNN vs CNN (equivariant by
construction) vs MLP (not). Whatever the result, the paper then *reports* its inductive bias
instead of asserting one. Pair with an entanglement-entropy-vs-cut profile showing bounded
entanglement spread — the property that actually supports trainability.

**3.7 Classical-simulability statement.** A 10-qubit, ~76-parameter unitary circuit is
classically simulable — preempt the "so what's quantum here?" reviewer by saying so: this work
validates an architecture family at simulable scale and characterises its scaling (Phase 8);
no quantum-advantage claim is made. Q1 venues reward this candour; hiding it invites rejection.

---

## Phase 4 — Harder datasets

Answers "Why only MNIST 0 vs 1?" (0v1 is ~99.8% linearly separable — no claim can rest on it).

| Tier | Dataset | Why | Cost |
|---|---|---|---|
| 4.1 | MNIST hard pairs 3v5, 4v9, 5v8, 7v9 | already supported via `--datasets` | low |
| 4.2 | Fashion-MNIST pairs (pullover/coat, shirt/T-shirt) | same IDX format → **drop-in**; genuinely harder | low |
| 4.3 | **MedMNIST v2** (PneumoniaMNIST, BreastMNIST) | 28×28 grayscale, peer-reviewed benchmark (Sci. Data 2023), real-world framing — strongest single addition | low–med |
| 4.4 | KMNIST pair | drop-in third domain if 4.3 stalls | low |
| 4.5 | Multi-class MNIST `[ARCH — opt-in]` | needs a multi-outcome readout head → **architecture change; default OFF.** The binary-only question is instead answered by breadth (3 domains × hard pairs) plus an explicit scope statement | med |

Also fix `DATASET_README.md` and the paper's dataset table to the actual executed splits (F4).

---

## Phase 5 — Baselines and statistical rigor

Table `tab:comparison` currently mixes other papers' numbers from other splits — reviewers
reject that on sight. Every comparison row is either **reproduced on your split** or explicitly
labelled "as reported in [ref], different setup" and moved out of the main table.

**5.1 Classical arms** (identical splits/seeds): logistic (exists); parameter-matched MLP
(exists) + unconstrained MLP; **classical CNN** — wire the orphaned `build_baseline_cnn` into
`run_classical_baselines` (or port to PyTorch to drop the TF dependency), in two sizes:
(a) parameter-matched ≈76–270 params, (b) standard small CNN (~50k) as the practical ceiling.
Add a **downsampled-input control** (logistic on the same 16×16 input) so encoding effects are
separated from model effects.

**5.2 Quantum arms:** `cong`/`hur`/`ttn` (exist). Add the two controls reviewers most often
demand, both nearly free:
  - **Random-frozen-circuit control:** frozen FQCNN with random untrained parameters + trained
    logistic readout on ⟨Z⟩. If it matches the trained model, training isn't doing the work.
  - **Encoding-only control:** logistic regression directly on the normalised amplitude vector.

**5.3 Statistics.** ≥5 seeds (10 preferred) per cell; mean ± std **and** 95% bootstrap CIs;
paired tests — **McNemar on per-example predictions** within a split, Wilcoxon signed-rank
across seeds, Holm–Bonferroni across arms. Extend `aggregate_metrics` accordingly. No
single-run number appears anywhere in the paper again.

**5.4 Calibration.** The paper claims the model is "well-calibrated" with zero evidence: add
reliability diagrams + expected calibration error for FQCNN and baselines. Cheap; directly
supports an existing claim.

**5.5 Learning curves.** Accuracy vs training-set size N ∈ {50, 100, 200, 400, 800, 1600} for
FQCNN vs matched MLP/CNN. Ties into the Caro bound (3.5); sample-efficiency is the most
plausible practical advantage of a 76-parameter model and deserves its own figure.

**5.6 Cost columns.** Every comparison table carries params (effective), circuit depth, 2-qubit
gates, shots/inference, and training wall-clock. If the quantum model is 1000× slower and 0.3%
better, the table says so (§A6); the defensible claim is parameter/sample efficiency.

---

## Phase 6 — Ablation grid

`ABLATION_CONFIGS` exists; it has just never been run for real. Final grid (one factor at a time
from `proposed`, ≥5 seeds, ≥3 datasets):

| Axis | Arms |
|---|---|
| Pooling | `none`, `measurement`, `unitary`(frozen), `coherent`[opt-in], `su4` |
| Conv entanglement | `none`, `one_diagonal`, `full` |
| Kernel rotations | `ry`, `su2` |
| Encoding | amplitude (frozen), `enc_augmented` (+two-stage map, arm only), angle/compact encoding |
| EMA / grad-clip | on/off (training-recipe ablation — reviewers ask) |

Deliverable: one table of Δaccuracy ± CI per component vs the frozen model, plus a forest-style
plot. Every "novelty" claim in Sec. VI must map to a row here; claims without rows are deleted
in Phase 10.

---

## Phase 7 — Noise realism and real hardware

**7.1 Model state preparation honestly (credibility-blocking).** Decompose amplitude encoding
into its actual gate sequence before applying noise, so the ~2,026 state-prep CNOTs carry error
like every other gate. Recommended implementation: transpile the *full* circuit (prep + model)
with Qiskit and simulate with **AerSimulator + device noise models** — much faster than
`default.mixed` with thousands of channel insertions, and it reuses the transpilation needed for
Phase 8 anyway. Keep the existing lightweight sweep as a secondary "model-only noise" figure
with its scope stated. Expect the realistic curve to collapse; report it and pivot to 7.4.

**7.2 Calibration-accurate backends.** Replace the hand-written `IBM_BASE_NOISE` dict with
`qiskit_ibm_runtime` fake backends (FakeSherbrooke/FakeTorino: real T1/T2, gate and readout
errors, heavy-hex coupling, routing). Free, reproducible, reviewer-proof.

**7.3 Real QPU within the free Open Plan (~10 min/month).** Scope tightly:
  - a **6-qubit scaling-family instance** (state prep ≈ 114 CNOTs, not 2,026) — an instance of
    the same frozen architecture at smaller n (Phase 8 family), *not* a modified model;
  - 50–100 test samples × 1,024 shots, one backend, one seed;
  - with and without readout mitigation (M3); ZNE only if budget remains;
  - report the three-point ladder **ideal sim → fake backend → real QPU** — exactly what "what
    happens on real hardware?" is asking for.
  `experiments/hardware_run.py` is a usable scaffold but targets a deprecated API surface
  (`channel="ibm_quantum"`); verify against current `qiskit-ibm-runtime` primitives (V2) before
  spending QPU minutes, and rehearse the exact job on the fake backend first.

**7.4 Fidelity-threshold analysis.** From measured depth × 2-qubit error, derive the gate
fidelity at which the full 10-qubit model (including state prep) becomes viable, and compare to
current and near-term hardware. Converts the expected 7.1 collapse into a quantitative roadmap —
a result, not an embarrassment.

---

## Phase 8 — Resources and the scaling family

Answers "how many gates per inference?", "can this scale beyond 10 qubits?", "is amplitude
encoding practical?"

**8.1 Resource table (must-have).** Transpile to a hardware basis (`{RZ, SX, X, ECR}`); report
1q/2q gates, depth, params — **state preparation and model separately**:

| Component | 2q gates | Depth | Params | Scaling |
|---|---|---|---|---|
| Amplitude prep (Möttönen), n=10 | ~2,026 | ~O(2^n) | 0 | 2^{n+1}−2n−2 |
| Convolution stage | (measure) | O(1)/window | 48 (12 eff./slice) | O(windows) |
| Pooling stages | (measure) | O(1)/pair | ~9 eff. | O(n) |
| Classifier | (measure) | O(1) | ~4 eff. | O(n_r) |

Headline honesty: **state preparation is ~97% of the circuit.** Then address it: the compact-
encoding ablation arm (Phase 6) shows the model body is encoding-agnostic; QRAM/approximate
low-rank prep cited as future work; per-inference shot cost `O(1/ε²)` and per-gradient-step
`2|Θ_eff|` circuits under parameter-shift stated with concrete numbers.

**8.2 Scaling-family sweep.** The frozen architecture is a *family* parameterised by n: run
n ∈ {4, 6, 8, 10, 12, 14} on a fixed task (16 with GPU if available). Report accuracy, effective
params, transpiled gate counts, depth, wall-clock. n=16 also exhibits the deeper pooling cascade
(16→8→4→2→1), giving the hierarchy evidence F1 removed from the headline — as scaling data, with
the n=10 model still the paper's model.

**8.3 Trainability at scale** = Phase 3.1 + 3.2 outputs plotted over this sweep.

---

## Phase 9 — References

Current list: 50 entries, ~22 arXiv-only, and **`ref12`–`ref23`, `ref27`–`ref36` have no author
lists** — a desk-reject signal.
1. Full authors, venue, volume, pages, DOI for every entry; no exceptions.
2. Replace arXiv preprints with published versions wherever they exist (Quantum, PRA/PRX
   Quantum, npj QI, QMI, Nat. Commun.); keep genuinely unpublished 2024–25 preprints under ~20%
   of the list, labelled consistently.
3. Add what the new content needs: Nielsen & Chuang (deferred measurement); Möttönen et al. 2005
   + Shende–Bullock–Markov 2006 (state-prep cost); Baumgratz/Plenio (coherence resource theory,
   Prop 3); Larocca et al. + Ragone et al. 2024 (DLA/barren plateaus); Abbas et al. 2021
   (effective dimension); Caro et al. 2022 (generalization); Sim et al. 2019 (already present);
   MedMNIST (Yang et al. 2023); Fashion-MNIST (Xiao et al. 2017); M3 readout mitigation; the
   specific fake-backend/runtime tooling used.
4. Audit that each citation supports the sentence it decorates (e.g. `ref44` currently props up
   the image-locality claim it does not make).

---

## Phase 10 — Paper rewrite and venue

### Evidence artifacts the rewrite is built from

| # | Artifact | Phase |
|---|---|---|
| T1 | Dataset table, all datasets, train/val/test protocol | 0.3, 4 |
| T2 | Resource table, prep vs model, transpiled | 8.1 |
| T3 | Baseline table, all rows on your split, CIs + paired tests + cost columns | 5 |
| T4 | Ablation table, Δacc ± CI per component | 6 |
| T5 | Pooling-arm table incl. SU(4) ceiling + measurement tie | 2 |
| T6 | Capacity/trainability table: DLA dim, effective dim, expressibility, generalization bound | 3 |
| F-A | E1 exact-tie + E2 dephasing ablation (theory-predicted, confirmed) | 2.3 |
| F-B | Coherence/purity/entropy/MI per stage | E4 |
| F-C | Noise ladder: ideal → fake backend → QPU; fidelity threshold | 7 |
| F-D | Scaling family: accuracy/gates/wall-clock vs n | 8.2 |
| F-E | Gradient variance vs n + DLA certificate | 3.1–3.2 |
| F-F | Learning curves + calibration diagrams | 5.4–5.5 |

### Structural edits
- **Abstract/Intro:** replace "measurement pooling collapses the state and loses information"
  with Theorem 1's true claim: unitary pooling provably simulates measurement pooling at zero
  overhead, eliminates mid-circuit measurement/reset/feed-forward, and keeps the global state
  pure end-to-end. The fully-unitary identity of the model is *strengthened*, not diluted.
- **New Sec. III-E:** Theorem 1, Props 2–3, with proofs (deferred-measurement argument is short).
- **Secs. III/VI:** every equation matches the executed circuit (0.5); every novelty claim maps
  to an ablation row or is deleted; effective-parameter disclosure; locality claim rewritten to
  register-space + empirical inductive-bias result (3.6).
- **Sec. V:** rebuilt from T1–T6/F-A–F-F; "approx 98%" excised everywhere in favor of
  mean ± std under the clean protocol.
- **Limitations:** state-prep dominance, simulable scale (3.7), free-tier hardware scope, no
  quantum-advantage claim. Frank limitations buy credibility.
- **Reproducibility paragraph:** exact environment (F9 fixed), seeds, `reproduce.sh`, archived
  weights, and a Zenodo DOI for code+data at submission.

### Venue targeting (pick before writing style-sensitive sections)
- **IEEE TQE** — best format fit (IEEEtran), values engineering rigor + hardware results.
- **Quantum Machine Intelligence** — best topical fit; expects exactly T3/T4-style evidence.
- **EPJ Quantum Technology / Phys. Rev. Applied** — if the hardware ladder (F-C) is strong.
Note: `fqcnn.tex` is in IEEE *conference* format; journal submission needs the journal template —
plan a formatting pass at the end, not a reflow mid-writing.

---

## Reviewer question → evidence map

| Expected question | Answered by | Artifact |
|---|---|---|
| Why only MNIST 0 vs 1? | Phase 4 breadth + scope statement | T1, T3 |
| Why not compare with classical CNNs? | 5.1 (matched + full CNN, same split) | T3 |
| Is the pooling layer theoretically optimal? | Thm 1 (optimal within simulable class at min gates) + SU(4) bracket | Sec. III-E, T5 |
| Does coherence preservation measurably improve accuracy? | E1 exact tie (predicted) + E2 dephasing + family headroom E3 | F-A, T5 |
| Can this architecture scale beyond 10 qubits? | 8.2 family sweep + 3.1/3.2 trainability certificate | F-D, F-E |
| How many gates does one inference require? | 8.1 transpiled counts, prep vs model split | T2 |
| What happens on real quantum hardware? | 7.2–7.3 ladder + mitigation | F-C |
| Is amplitude encoding practical incl. state prep? | 8.1 honesty + compact-encoding arm + 7.4 threshold | T2, T4, F-C |
| Statistically meaningful? | 5.3 (seeds, CIs, McNemar/Wilcoxon) | all tables |
| Clean protocol? | 0.3 val/test separation + freeze guards | T1 |
| What's quantum about a simulable model? | 3.7 candour + scaling story | Limitations |

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Clean-protocol accuracy drops below 98% | **high** | Expected; §A6. A reproducible 96–97% ± CI beats an unreproducible 98.9%. Budget one re-tuning pass (val-driven only). |
| E3 shows `pool_coherent`/`su4` ≫ frozen block | medium | Fine under freeze: reported as family headroom; promotion is a user decision (§A5), not a plan default. Theorem 1 + hardware minimality carry the headline model regardless. |
| Realistic noise (7.1) collapses accuracy | **high** | Planned-for: fidelity-threshold roadmap (7.4) + 6-qubit instance that *is* viable. |
| Free-tier QPU minutes exhausted | medium | Rehearse on fake backend; one pre-sized job; QPU result framed as demonstration, not benchmark. |
| Grid exceeds compute budget | medium | 1.4 estimator gates the grid; cut order: seeds → datasets → non-pooling ablations. Pooling arms and E1/E2 are never cut. |
| DLA computation stalls at n=10 | low–med | Compute at n≤8 exactly + extrapolate with the empirical variance sweep (3.2); both together still beat citation-by-analogy. |
| PennyLane 0.38 lacks a needed feature (e.g. clean mid-circuit measure on mixed device) | med | Pin the working env (F9 fix) and isolate any Qiskit-side work (7.x, 8.1) in a separate venv; do not upgrade the training env mid-study. |

---

## Definition of done

- [ ] Freeze guards (fingerprint + expectation regression) committed and passing; every later
      commit runs them.
- [ ] Zero circuit changes to the headline model (inert-gate removal, if done, carries proof +
      1e-12 regression).
- [ ] Main pipeline contains no non-unitary operation; measurement pooling exists only as a
      labelled baseline arm.
- [ ] No test-set information reaches model selection (automated check).
- [ ] Effective-parameter audit committed; paper reports allocated vs effective counts.
- [ ] Every equation in the paper matches the executed circuit (reconciliation table complete).
- [ ] Theorem 1 / Props 2–3 written with proofs; E1 tie confirmed to ~1e-12; E2 run.
- [ ] ≥3 datasets × ≥5 seeds; all numbers mean ± std with CIs and paired tests.
- [ ] Every comparison row reproduced on your split or explicitly out-of-table.
- [ ] Resource table with state-prep/model split; scaling family n=4…14.
- [ ] Trainability certificate (DLA + variance sweep) and generalization bound in the paper.
- [ ] Noise ladder incl. one real-QPU point, scoped and caveated.
- [ ] `requirements-lock.txt` matches the real environment; `reproduce.sh` regenerates every
      table and figure from a clean checkout.
- [ ] Bibliography: full metadata, DOIs, peer-reviewed versions, no author-less entries.

---

## Suggested order of attack

1. **Week 1:** Phase 0 in full (guards first), Phase 1.1–1.2. Run E1 the moment the guards pass —
   it is nearly free and locks in the paper's story.
2. **Week 2:** Finish Phase 1; clean-protocol headline retrain (0.3); E2. Write Theorem 1 section
   in parallel (it depends on no compute).
3. **Week 3:** Launch Phases 4–6 as unattended overnight grid (gated by 1.4). Phase 3 analyses on
   the archived + retrained weights during the day.
4. **Week 4:** Phase 7 fake-backend ladder; rehearse then execute the QPU job. Phase 8 resource
   table + scaling sweep queued at night.
5. **Week 5:** Finish Phases 3/8; Phase 9 references.
6. **Weeks 6–7:** Phase 10 rewrite, venue formatting, reproducibility package (Zenodo), internal
   red-team pass against the reviewer map.
