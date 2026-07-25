# FQCNN Q1 Upgrade — Program Status

Roadmap §20 program dashboard. Update at every gate. One row per milestone.

**Governing spec:** `UPGRADE_PLAN.md` (v2)
**Roadmap:** `docs/superpowers/plans/2026-07-23-fqcnn-q1-upgrade-roadmap.md`
**Design:** `docs/superpowers/specs/2026-07-23-fqcnn-q1-upgrade-design.md`
**Month plan:** `docs/superpowers/plans/2026-07-25-fqcnn-remaining-work-month-plan.md`

**Last updated:** 2026-07-25 · **Branch:** `plan/fqcnn-q1-upgrade` · **Tests:** 111 passing

---

## 1. Milestone state

| Milestone | State | Gate | Evidence |
|---|---|---|---|
| **M0 — Phase 0: freeze + protocol** | **PASSED 2026-07-25** | tag `phase-0-gate` | 63 tests; `docs/superpowers/plans/2026-07-25-fqcnn-phase-0-freeze-and-protocol.md` |
| **M1 — Phase 1: affordable execution** | **IN PROGRESS** · 1.1–1.4 done, 1.5 not required | **grid fits ≤7 nights: PASSED** (2.2 h of 70 h) | §§3a–3d; 110 tests |
| M2 — Phase 2: pooling theory (E1–E5) | **E1, E2, 2.1, 2.7 done**; E3–E5 not started | **E1 agrees to ~1e-12: PASSED** (2.2e-16 at headline n=10) | `Results/evidence/`; §§7, 7a |
| M3 — Phase 3: model analysis | not started | — | — |
| M4 — Phase 4: harder datasets | not started | — | — |
| M5 — Phase 5: baselines + statistics | not started | — | — |
| M6 — Phase 6: ablation grid | not started | — | — |
| M7 — Phase 7: noise + real QPU | not started | one real-QPU point | — |
| M8 — Phase 8: resources + scaling | not started | — | — |
| M9 — Phase 9: references | not started | — | — |
| M10 — Phase 10: manuscript + venue | not started | — | — |
| M11 — final reproduction + red team | not started | — | — |

## 2. Active gate and blockers

**Active gate:** M1 — the seven-night budget gate **passed** 2026-07-25 (§3d). The remaining M1
item is the clean headline retrain, which replaces 98.86% everywhere.

| # | Blocker | Severity | Owner milestone | Status |
|---|---|---|---|---|
| B1 | `pool_measurement` did not implement the theorem's channel — E1 could not pass | **High** | M2.1 | **Closed 2026-07-25** — fixed; **E1 PASSES** (§7) |
| B2 | Batched path must be proven equivalent before any grid runs | High | M1.1 | **Closed 2026-07-25** (§3a) |
| B3 | Ablation grid runs n=8 while the frozen headline is n=10, so T4 would describe a different model | **High** | M5/M6 | Open, documented (§3d) |

## 3. M0 outcome (2026-07-25)

All eight sub-milestones pass. `python -m pytest tests/ -q` → 63 passed. `reproduce.sh` runs
the gate as step 0 before producing any result.

### Authoritative measurements

| Quantity | Value | Source |
|---|---|---|
| Allocated parameter slots | **269** | `tests/fixtures/effective_params.json` |
| Syntactically read slots | **78** | same |
| **Effective slots (\|grad\| > 1e-12)** | **74** | same |
| Naive nonzero-gradient count | 77 (3 are ~3e-17 round-off) | same |
| Zero-gradient slots | 195 | same |
| Gradient tolerance | 1e-12 | same |
| Audit inputs / seed | 20 inputs, seed 20260725 | same |
| Pooling pairs at n=10 | 8 (5 + 2 + 1) | `tests/fixtures/headline_signature.json` |
| Active-wire schedule | 10 → 5 → 2 → 1, wire 8 retired unpaired | same |
| Signature hash | `d29cbf3e…f673ffe3` | same |

**Per-group effective:** conv0 48/48 · conv1–3 **0**/48 each · pool0 12 · pool1 6 · pool2 3 ·
classifier 5/32.

### Verified environment (matches `requirements-lock.txt`)

Python 3.9.13 · PennyLane 0.38.0 · NumPy 1.26.4 · scikit-learn 1.6.1

## 3a. M1 progress

| Sub-milestone | State | Evidence |
|---|---|---|
| **1.1 Batched backpropagation** | **DONE 2026-07-25** | `tests/test_batched_execution.py` (6 tests); benchmark in §5 |
| **1.2 Input and validation caching** | **DONE 2026-07-25** (scope reduced on evidence, §3b) | `tests/test_encoded_cache.py` (7 tests) |
| **1.3 Safe parallelism and resume** | **DONE 2026-07-25** | `tests/test_resume_and_parallelism.py` (15 tests) |
| **1.4 Cost estimator and grid approval** | **DONE 2026-07-25 — GATE PASSES** | `experiments/estimate_cost.py`; `tests/test_cost_estimator.py` (9 tests); §3d |
| 1.5 Conditional accelerators | **not required** | 1.4 uses 3% of budget; §18.4 stays off |
| Clean headline retrain | not started | gated on 1.1–1.4 |

### M1.1 exit check (roadmap: outputs, loss, *every* gradient, one optimizer update)

`python -m pytest tests/ -q` → **69 passed** (63 from M0, 6 new). Batched vs sequential:

| Compared quantity | Tolerance | Result |
|---|---|---|
| Raw `<Z>` outputs | 1e-10 | pass |
| Batch BCE loss | 1e-10 | pass |
| All **269** parameter gradients | 1e-10 | pass; max observed diff **4.96e-16** |
| One Adam update (269 slots) | 1e-10 | pass |
| Circuit signature hash | exact | matches committed `d29cbf3e…` |
| A3 unitarity on the batched tape | exact | no violations |

**What changed.** `QCNN/models/QCNNModel.py` gains a second QNode on `default.qubit` with
`diff_method="backprop"` that calls the *same* `circuits.build_circuit`; `quantum_predict_batch`
and `Qtrainer.quantum_cost` now use it. The sequential `lightning.qubit` QNode is retained
unchanged as the correctness oracle. No circuit change — the signature test proves it.

`QCNN/freeze.py` splits `tape_signature(tape, n_slots)` and `qnode_tape(qnode, …)` out of
`circuit_signature` so the committed hash can be checked against any execution path.

**Not yet ported:** `QCNN/utils/metrics.py:55` still loops per-sample. It runs once per run
(final test evaluation), not per epoch, so it is not on the hot path.

## 3b. M1.2 — and why its speed premise is now dead

### Measured epoch breakdown (headline n=10, 60/15/25 on 12,665 samples)

Profiled 2026-07-25 *after* M1.1, at N_train=7,599 / N_val=1,900 / batch=32.

| Component | Per epoch | Share |
|---|---|---|
| Gradient steps (237 batches × 0.580 s) | **137.5 s** | **94.0%** |
| Full validation forward (1,900 samples) | 7.9 s | 5.4% |
| Train-accuracy probe (200 samples) | 0.85 s | 0.6% |
| `_preprocess_input` (both call sites, 237 batches) | **0.04 s** | **0.03%** |
| **Total** | **~146.3 s** | |

`UPGRADE_PLAN.md` 1.2 was written against the pre-M1.1 sequential path, where an epoch cost
~2.7 h and input handling looked material. After 87.6×, **input caching saves 0.03% of an
epoch.** The milestone's speed rationale no longer holds.

### What was implemented

The cache, for **provenance** rather than speed — roadmap M1.2 also requires cache identity in
run metadata, and that requirement survives the measurement:

- `PureQuantumEncoder.precompute_amplitudes(X, n_qubits)` — pad/truncate + L2-normalise once
  per split, outside the training loop. Produces exactly the array `amplitude_encoding` hands
  to `AmplitudeEmbedding`, so the state-prep op and the frozen fingerprint are unchanged.
- `PureQuantumEncoder.amplitude_identity(encoded)` — sha256 + shape + dtype.
- `Qtrainer` encodes train and val once before the epoch loop and exposes
  `trainer.encoded_cache_identity`.

**Exit check met.** Cached and uncached circuit outputs agree at 1e-10; a 2-epoch × 2-batch
training run calls the encoder exactly twice — once per split — proving repeated epochs add no
re-normalisation.

### What was declined, and why

**Validation subsetting is NOT implemented.** Roadmap M1.2 asks for a fixed validation subset
for per-epoch monitoring. Measured: full val costs 7.9 s/epoch, a 500-sample subset 2.1 s — a
**4% epoch saving in exchange for noisier model selection**, since val accuracy drives
checkpointing, LR plateau, and early stopping (M0.4). That is a bad trade at the current
throughput and it degrades a protocol property M0 was built to secure. Reopen only if M1.4
projects past the 7-night gate; the mandated cut order (seeds → datasets → non-pooling
ablations) should be exhausted first.

**Open item for M8:** validation is evaluated in one batched call. At n=10 that is a 31 MB
statevector block, which is fine; the n=14 scaling instances will need chunking to avoid
running out of memory.

## 3c. M1.3 — parallelism and resume

`python -m experiments.run_experiments --jobs N [--force]`

The sweep now enumerates every `(dataset, config, seed)` cell up front, splits it into
reusable and pending, and runs the pending set across worker processes. Resume is the
default; `--force` re-runs everything.

| Requirement | Implementation | Verified |
|---|---|---|
| Process-level jobs over independent cells | `ProcessPoolExecutor`, `--jobs N`; `0` = cores − 2 | 4 cells / 2 workers completed out of order, exit 0 |
| Worker thread limits | `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`, set in the parent *before* the pool so spawned children inherit at import | BLAS reads these at import, so an initializer would be too late |
| Skip only complete, schema-valid, identity-matching runs | `run_artifacts.is_reusable()` | 11 rejection-path tests |
| Failure manifest + nonzero exit | `Results/experiments/failures.json`, `sys.exit(1)` | forced failure → `EXIT=1`, manifest records dataset/config/seed/traceback |

**Exit check met.** A repeated smoke command reports `0 cells to run, 4 reused` and writes a
byte-identical `summary.csv`. Aggregation reads every cell's metrics back from disk in
schedule order, so a fresh sweep and a resumed one aggregate from identical input regardless
of completion order.

`is_reusable` rejects: unfinished or failed state, missing `weights.npz` or
`predictions.npz`, empty metrics, a status file predating the current schema, a different
seed, and any config difference — including an ablation switch, the case that would
otherwise put unitary-pooling numbers in a `pool_none` row. `split_id` is exempt because the
runner stamps it during the run, so a pre-run preview cannot carry it.

**Also fixed:** `--configs bogus` now exits 2 with the valid list instead of raising a bare
`KeyError` during scheduling.

**Pre-existing footgun, not fixed:** `--quick` overwrites `--epochs`, `--samples`,
`--seeds`, `--datasets` and `--configs` *after* parsing, so `--quick --epochs 30` silently
runs 2 epochs. Left alone as out of scope; worth a guard before the real grid.

## 3d. M1.4 — the seven-night gate: **PASSES**

`python -m experiments.estimate_cost --jobs 0`

Default grid: 4 MNIST pairs × 7 configs × 5 seeds = **140 cells**, 30 epochs, 400 samples.

| | |
|---|---|
| Serial | **43.5 h** |
| Wall-clock at 20 workers (22 cores − 2) | **2.2 h** |
| Budget (7 nights × 10 h) | 70 h |
| **Verdict** | **FITS — 3% of budget, 0.2 nights** |

An "unattended night" is not defined in the plan; this repo uses **10 h**, overridable with
`--hours-per-night`.

### Measured calibration (this machine)

| Config | n | grad/batch | fwd/sample | seq-fwd/sample | path |
|---|---|---|---|---|---|
| proposed | 8 | 0.247 s | 0.0037 s | 0.132 s | batched |
| pool_none | 8 | 0.202 s | 0.0056 s | 0.136 s | batched |
| pool_measurement | 8 | 0.236 s | 0.0034 s | 0.183 s | batched |
| ent_one_diagonal | 8 | 0.223 s | 0.0036 s | 0.132 s | batched |
| ent_none | 8 | 0.195 s | 0.0027 s | 0.127 s | batched |
| kernel_ry | 8 | 0.143 s | 0.0021 s | 0.129 s | batched |
| **enc_feature_map** | **16** | **12.538 s** | 0.334 s | 0.333 s | **sequential (memory cap)** |

Baselines add **76 s** to each `proposed` cell (classical + cong/hur/ttn).

### Consequences

- **No cuts.** The mandated reduction order is not triggered.
- **Ten seeds are affordable** — 4.4 h wall-clock, still 6% of budget. M5.3 prefers 10 over 5.
- **Decision §18.4 (JAX/GPU) resolves to OFF.** The verified default path clears the budget by
  more than an order of magnitude; `UPGRADE_PLAN.md` 1.5 activates only if it misses.

### Three findings this surfaced

1. **A regression I introduced in M1.1.** `enc_feature_map` is a 16-qubit config; batched
   backprop needs 32 × 2¹⁶ complex128 per retained intermediate and exhausts memory. The old
   per-sample path handled it. Fixed by a memory cap (`MAX_BATCHED_AMPLITUDES = 2¹⁸`):
   `batch_expectations()` falls back to the sequential path above it, and
   `quantum_predict_batch` chunks. Chunking is *not* used for gradients — under backprop every
   chunk's tape is retained until the backward pass, so it would not lower peak memory.

2. **`enc_feature_map` is 90% of the grid** (39.2 h of 43.5 h serial), because 16 qubits on the
   sequential path costs 12.5 s per batch against 0.25 s for the rest. It still fits, so it
   stays; but it is the first thing to cut if anything else grows.

3. **The ablation grid does not run the headline architecture.** Every `ABLATION_CONFIGS` entry
   uses `image_size=16` → **n=8 qubits**, while the frozen headline is `image_size=28` → **n=10**
   (`freeze.HEADLINE_N_QUBITS`). So T4's ablation rows would describe a different model than the
   headline row. This is an M5/M6 correctness issue, not a cost issue — flagged here because it
   also means the projection above understates the real grid. At n=10 the non-`enc` configs cost
   roughly 2.3× more (0.580 s vs 0.247 s per batch, measured), which still fits comfortably.

**Also worth porting now:** sequential forward is 0.132 s/sample against 0.0037 s batched — a
**35× gap**. The single final test evaluation still runs per-sample through
`QCNN/utils/metrics.py:55`, costing 18.7 s of each 133 s cell (14%). Deferred in M1.1 as
off-the-hot-path; the measurement now says otherwise.

## 4. Run cells: required vs completed

| Workstream | Required cells | Complete | Milestone |
|---|---|---|---|
| Clean headline retrain | 1 | 0 | M1 |
| E1 / E2 (fixed-parameter, no training) | 2 | 0 | M2 |
| E3 pooling arms (arms × ≥5 seeds × ≥3 datasets) | ≥75 | 0 | M2 |
| Datasets (hard pairs + 3 domains) | TBD at M4 | 0 | M4 |
| Classical + quantum baselines | TBD at M5 | 0 | M5 |
| Ablation grid | TBD at M6 | 0 | M6 |
| Scaling sweep n={4,6,8,10,12,14} | 6 | 0 | M8 |
| Real-QPU job | 1 | 0 | M7 |

Cell counts marked TBD are fixed by `estimate_cost.py` at the M1.4 gate.

**Failure manifest:** empty (no grid has run).

## 5. Compute budget

| Quantity | Value |
|---|---|
| Archived headline run (historical, pre-freeze) | ~78,682 s (~22 h), 24 epochs, 1 config |
| Sequential gradient cost | **1.256 s/sample** (lightning.qubit, adjoint; linear in batch) |
| Batched gradient cost @ batch 32 | **0.535 s/batch** = 0.017 s/sample (default.qubit, backprop) |
| **Measured speedup @ batch 32** | **87.6×** — replaces the unverified 10–50× estimate |
| **Measured epoch cost, headline** | **~146 s** (7,599 train / 1,900 val) — see §3b |
| **Projected grid wall-clock** | **2.2 h at 20 workers** (43.5 h serial) -- see 3d |
| **Budget verdict** | **FITS: 3% of 7 nights** |
| Nights consumed | 0 of 7 |

### M1.1 benchmark (2026-07-25, headline n=10 config)

Cost of one gradient of the batch BCE loss w.r.t. all 269 slots.

| Batch | Sequential (s) | Batched (s) | Speedup |
|---|---|---|---|
| 1 | 1.210 | 0.128 | 9.5× |
| 4 | 4.960 | 0.156 | 31.9× |
| 8 | 10.051 | 0.254 | 39.6× |
| 16 | 20.10 *(extrap.)* | 0.300 | 67.0× |
| **32** (trainer default) | **46.838** | **0.535** | **87.6×** |
| 64 | 80.41 *(extrap.)* | 0.919 | 87.5× |

Sequential linearity confirmed: 1.210 / 1.240 / 1.256 s/sample at n=1/4/8, so the two
extrapolated rows are sound. Batch 32 is measured end to end, not extrapolated.

**QPU budget:** IBM Open Plan, ~10 min/month. Consumed: 0. Reserved: one 6-qubit job in Month 3
(M7.3), pending rehearsal and §18.6 sign-off.

## 6. Evidence artifacts

| ID | Artifact | Milestone | Status |
|---|---|---|---|
| T1 | Datasets + clean split protocol | M0.4, M4 | protocol done; table pending M4 |
| T2 | Resource table, prep vs model, transpiled | M8.1 | not started |
| T3 | Baselines, CIs, paired tests, cost columns | M5 | not started |
| T4 | Ablation Δacc ± CI | M6 | not started |
| T5 | Pooling arms + SU(4) ceiling + measurement tie | M2 | measurement-tie row available |
| T6 | DLA, effective dim, expressibility, gen. bound | M3 | not started |
| F-A | E1 exact tie + E2 dephasing | M2.2–2.3 | **both generated** |
| F-B | Coherence / purity / entropy / MI per stage | M2.5 | not started |
| F-C | Noise ladder ideal → fake → real + threshold | M7 | not started |
| F-D | Scaling family | M8.2 | not started |
| F-E | Gradient variance + DLA certificate | M3.1–3.2 | not started |
| F-F | Learning curves + calibration | M5.4–5.5 | not started |

## 7. B1 — the E1 blocker (recorded 2026-07-25)

**E1 cannot pass against the current `pool_measurement`, and its failure would look like a
positive result.** Recorded here so it is not mistaken for a physics finding later.

The frozen block (`QCNN/layers/QPool.py:76-83`), per pair (keep `a`, discard `b`):

```
CRY(α, [b,a]) ; CRZ(β, [b,a]) ; RY(0.02, b) ; RY(γ, a)
```

Both controlled gates are controlled on `b`, so `V = |0⟩⟨0|_b ⊗ U_0 + |1⟩⟨1|_b ⊗ U_1` with

- `U_0 = RY(γ)`
- `U_1 = RY(γ)·RZ(β)·RY(α)`

Theorem 1 therefore requires the measure-and-condition arm to apply **`U_0` on outcome 0 and
`U_1` on outcome 1**. What `quantum_conditional_pooling` (`QPool.py:127-138`) actually applies:

- outcome 0 → identity (there is no else-branch); should be `RY(γ)`
- outcome 1 → `RY(α)`; should be `RY(γ)·RZ(β)·RY(α)`
- consumes 1 of the 3 angles per pair; `β` and `γ` are never read

So the arm mismatches by an O(1) amount, not 1e-12. Its docstring states the intent plainly —
*"deliberately collapses the quantum state (no coherence preservation)"* — it was written as a
strawman under the pre-upgrade narrative that measurement pooling must lose information. That
narrative is F6, the claim the upgrade exists to delete.

**The trap:** running E1 as-is yields a large gap, which reads as confirmation of the paper's
current (false) claim. Roadmap M2.2 is explicit that an E1 miss is an implementation or
theorem-mapping defect, never an experimental result to average away.

Note the same argument proves `RY(0.02)` inert: it acts only on `b`, and partial trace over `b`
is invariant under a unitary on `b` alone. That is why M0 measured 8.33e-16 for its removal.

### Resolution and E1 result (2026-07-25)

`quantum_conditional_pooling` now applies `U_0` on outcome 0 and `U_1` on outcome 1, reading all
three angles through a `pair_angles` mapping shared with the frozen arm so the arms cannot drift.

A second arm, `measurement_channel`, implements the same map as an explicit CPTP channel with
Kraus operators `K_m = U_m ⊗ |m⟩⟨m|`. **E1 compares against this one deliberately.** A simulator
may legitimately implement mid-circuit measurement *by deferring it back into the very controlled
gates under test, which would make the tie a tautology about PennyLane rather than a result about
the circuit.* The Kraus channel is genuinely non-unitary, so the agreement is real. (This also
sidesteps the `UPGRADE_PLAN.md` risk-register concern about `qml.measure`/`qml.cond` on
`default.mixed` under PennyLane 0.38.)

**E1 PASSES.** `python -m experiments.pooling_analysis`, on `default.mixed`, fixed parameters:

| n | weights | inputs | max \|difference\| | tolerance | verdict |
|---|---|---|---|---|---|
| **10 (headline)** | **archived** | 8 | **2.220e-16** | 1e-12 | **PASS** |
| 8 | seeded | 4 | 1.665e-16 | 1e-12 | PASS |
| 6 | seeded | 2 | machine precision | 1e-12 | PASS (`tests/test_pooling_equivalence.py`) |

The headline run took 4 h 45 min: on `default.mixed`, `AmplitudeEmbedding` decomposes into the
full ~2,026-CNOT Mottonen sequence against a 16 MB density matrix, which dominates. One of the
eight inputs agreed to **0.000e+00** — bit-identical.

Four orders of magnitude inside tolerance — the difference is float64 round-off, not physics.
Theorem 1's a-priori prediction of an *exact* tie is confirmed.

Three controls make the result meaningful rather than vacuous:

- **`pool_none` does differ** (>1e-6), so the readout is not simply insensitive to pooling.
- **The mid-circuit arm matches the channel arm**, so the trainable arm E3 will use is the same
  map E1 validated.
- **Every angle slot demonstrably moves the channel**, a direct regression guard on B1.

Evidence: `Results/evidence/e1_pooling_equivalence.json`. This is the F-A exact-tie data.

### E2 -- fixed-parameter dephasing (Proposition 3): **PASSES**

`python -m experiments.pooling_analysis --experiment e2`. Full computational-basis dephasing
(`PhaseFlip(0.5)`) is injected on every discarded wire *immediately before* pooling, at
unchanged parameters -- no retraining, so there is no optimisation confound.

| Quantity | Predicted | Measured | Tolerance |
|---|---|---|---|
| max \|delta readout\| | 0 | **1.293e-14** | 1e-12 |
| max trace distance (Prop 3's Delta_coh) | 0 | **6.978e-14** | 1e-12 |
| delta accuracy | 0 | **0.0 exactly** (every decision unchanged) | -- |
| **control:** dephase *kept* wires instead | > 0 | **1.129e-01** | -- |

The control is what makes this a result rather than a null. Dephasing the **kept** wires moves
the readout by 0.113; dephasing the **discarded** wires moves it by 1e-14. Twelve orders of
magnitude apart, with the same channel, in the same circuit. The channel is unambiguously live
-- the frozen block simply does not read the discarded wire's coherences, exactly as Theorem 1
says it cannot.

Delta accuracy is 0 *exactly*, not approximately: the sign of the readout is unchanged for every
input, so the predicted class is identical against any labelling whatsoever.

E2's residuals (1e-14) sit two orders above E1's (1e-16) because `default.mixed` accumulates more
round-off once explicit channels are in the tape. Both are far inside tolerance.

Two evaluation-only hooks were added to `CircuitHooks` for this: `before_pool` (dephasing
injection) and `terminal` (state observation, for the reduced density matrix Prop 3 is defined
on). Both default to `None`, so the headline path is unchanged -- A3 permits channels in
evaluation-time studies only.

Evidence: `Results/evidence/e2_dephasing.json`.

## 7a. M2.7 — the tested theory section (2026-07-25)

`fqcnn.tex` now carries **Sec. III-E, "Unitary Versus Measurement-Based Pooling"**: Theorem 1
(exact simulation), Proposition 2 (strict containment via the SWAP witness), Proposition 3
(coherence-transfer functional), each with a proof and each backed by a test.

| Statement | Test | Result |
|---|---|---|
| Theorem 1 | `test_e1_unitary_and_measurement_pooling_agree_exactly` | E1: **2.2e-16** at headline n=10 |
| Proposition 2 | `test_swap_witnesses_strict_containment` | trace distance **1/2** |
| Proposition 3 | `test_e2_dephasing_discarded_wires_changes_nothing` | E2: **1.3e-14**, ΔAcc **0.0 exactly** |

### The information-loss claim is deleted

It appeared in **six** places. All replaced with Theorem 1's defensible claims — exact
simulation at zero overhead, no mid-circuit measurement/reset/feed-forward, a globally pure
state, end-to-end adjoint differentiability:

Abstract · Sec. I (×2) · Sec. II (×2) · Sec. V analysis · Sec. V comparison · Sec. VI-A

Verified absent: no occurrence of *irrevocable*, *irreversible loss*, *information loss*,
*measurement-induced*, or *destructive projective* remains in the manuscript.

### A defect found while writing it

**Fig. 4's caption said the controlled rotations were "controlled on $q_a$"** — the *retained*
qubit. The code controls on $q_b$, the **compressed** qubit. That is not cosmetic: Theorem 1
holds precisely *because* the controls are diagonal in the compressed qubit's basis. Had the
caption been right, the theorem would not apply to the circuit it describes. Corrected, and the
control assignment is now explicit in `eq:poolblock` with a sentence saying why it is
load-bearing.

Also resolved: **Sec. VI-B carried a third, conflicting pooling equation** (`CRZ·CRY·CRZ`),
part of F4's "three different definitions of the paper's own contribution." Deleted; it now
cross-references the single canonical `eq:poolblock`.

### Build status

`pdflatex` × 3 passes → **11 pages, zero errors, zero undefined references or citations.**
Nielsen & Chuang added as `ref51` for the deferred-measurement principle (a Phase 9 item,
pulled forward because Theorem 1 needs it).

**Not done here:** the `approx 98\%` accuracy claims remain in the abstract, intro, and
conclusion. They cannot be replaced until the clean headline retrain exists (M1). Tracked as
reconciliation row 10.

## 8. Architecture sign-off decisions (roadmap §18)

| # | Decision | State | Outcome |
|---|---|---|---|
| 1 | Inert `RY(0.02)` removal | **Closed 2026-07-25** | **Retained.** Proven inert (9.44e-16 angle change, 8.33e-16 removal). Removal-with-proof branch was available and **declined** in favour of zero architecture change; documented as a no-op. See `docs/paper_code_reconciliation.md` row 4. |
| 2 | `pool_coherent` implementation | Open | Due Month 1, before M2.1. Default: do not run. |
| 3 | Multi-class head | Open | Due Month 2, before M4.5. Default: off; binary breadth instead. |
| 4 | JAX / GPU path | **Closed 2026-07-25** | **Off.** M1.4 projects 2.2 h against a 70 h budget; `UPGRADE_PLAN.md` 1.5 activates only on a miss. |
| 5 | Venue | Open | Due end of Month 3. Default: IEEE TQE. |
| 6 | Real-QPU submission | Open | Due Month 3, after fake-backend rehearsal. |

## 9. Definition-of-done checklist (`UPGRADE_PLAN.md`)

- [x] Freeze guards (fingerprint + expectation regression) committed and passing
- [x] Zero circuit changes to the headline model
- [x] Main pipeline contains no non-unitary operation
- [x] No test-set information reaches model selection (automated check)
- [x] Effective-parameter audit committed (269 allocated / 74 effective)
- [x] Every equation in the paper matches the executed circuit (reconciliation table complete)
- [x] `requirements-lock.txt` matches the real environment
- [ ] Theorem 1 / Props 2–3 written with proofs; E1 tie confirmed to ~1e-12; E2 run
- [ ] ≥3 datasets × ≥5 seeds; all numbers mean ± std with CIs and paired tests
- [ ] Every comparison row reproduced on your split or explicitly out-of-table
- [ ] Resource table with state-prep/model split; scaling family n=4…14
- [ ] Trainability certificate (DLA + variance sweep) and generalization bound
- [ ] Noise ladder incl. one real-QPU point, scoped and caveated
- [ ] `reproduce.sh` regenerates every table and figure from a clean checkout
- [ ] Bibliography: full metadata, DOIs, peer-reviewed versions, no author-less entries

**7 of 15 complete.**

## 10. Deferred items with in-code markers

| Item | Owner phase | Marker | Month |
|---|---|---|---|
| Classical baselines still select on their test argument | M5.1 | comment in `experiments/run_experiments.py:run_single` | 2 |
| `noise_sim` replays the historical 70/30 split | M7 | warning in `noise_sim.py` | 3 |
| Clean-protocol headline retrain | M1 | — | 1 |

## 11. Weekly report (roadmap §20)

**Week of 2026-07-21**

1. *What evidence became trustworthy?* The freeze guards, the 269/74 parameter audit, the
   60/15/25 split manifests, the single circuit source, and the environment lock. M0 gate passed.
2. *Which gate passed or remains blocked?* M0 passed (`phase-0-gate`). M1's ≤7-night gate is now
   active.
3. *What compute is queued next?* Nothing yet. The clean headline retrain is unblocked on
   throughput (M1.1 done, 87.6×) but still waits on 1.2–1.4.
4. *Which risks changed?* New: B1 — E1 will fail against the current `pool_measurement` for
   implementation reasons (§7). Identified before any compute was spent on it. Closed: B2 —
   batched execution is proven equivalent to the sequential oracle. "Batched speedup is
   insufficient" (roadmap §19) is retired: 87.6× measured against a 10–50× estimate.
5. *Which manuscript claims are now supported, rejected, or still untested?* The 98.86% headline
   is **rejected** (test-set leakage, F3) and has no replacement yet. The "269 trainable
   parameters" claim is **corrected** to 269 allocated / 74 effective. The information-loss claim
   (F6) is **rejected** pending Theorem 1's replacement text.
