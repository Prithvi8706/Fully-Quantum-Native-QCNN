# FQCNN Q1 Upgrade — Program Status

Roadmap §20 program dashboard. Update at every gate. One row per milestone.

**Governing architecture/protocol spec:** `UPGRADE_PLAN.md` (v2)
**Roadmap:** `docs/superpowers/plans/2026-07-23-fqcnn-q1-upgrade-roadmap.md`
**Design:** `docs/superpowers/specs/2026-07-23-fqcnn-q1-upgrade-design.md`
**Month plan:** `docs/superpowers/plans/2026-07-25-fqcnn-remaining-work-month-plan.md`
**Track A (grid, M4→M5→M6):** `docs/superpowers/plans/2026-07-27-track-a-grid-handoff.md`
**Track B (theory+hardware, M3/E5/M8→M7):** `docs/superpowers/plans/2026-07-27-track-b-theory-hardware-handoff.md`

**Active fast-track plan:** `docs/superpowers/plans/2026-08-27-q1-journal-fast-track.md`

**Planning amendment (2026-08-27):** The maximal Tasks 8–19 program has been compressed into a seven-focused-day Q1-journal fast track with two bounded unattended compute windows. Task 7 was integrated from `origin/dev` on 2026-08-27. The Q1 target is retained; evidence breadth is reduced only through explicit minimum bars and honest claim limits.

**Last updated:** 2026-08-30 · **Branch:** `dev` · **Verification:** Q1 comparison, transfer, clean reproduction, and package-integrity gates passed; archive is ready for upload after author approval

**State labels:** `pending` · `running` · `failed` · `complete`

---

## 0. Q1 fast-track daily execution

| Day | State | Required completion | Gate evidence |
|---|---|---|---|
| **1 — integrate + inventory** | `complete` | Integrate Task 7; preserve local files; repair resume if required; pass Task 7, architecture, protocol, and full suites; inventory evidence and manuscript claims | Task 7 synchronized at `0a0f60f`; disabled-layer resume regression fixed; final suite 570 passed / 2 documented Windows symlink skips; `Results/evidence/q1_fast_track_inventory.json`: 19 artifacts, 79 valid run cells, 0 invalid cells, 72 manuscript findings |
| **2 — multi-domain path** | `complete` | Add deterministic MNIST, Fashion-MNIST, and KMNIST sources; freeze four tasks; pass provenance/preprocessing tests and three smoke cells | `q1_dataset_provenance.json` (`A730615C…A1594B4D`): 3 families / 12 verified source files; split IDs hash source-stable IDs; isolated three-family two-epoch smoke completed and resumed 3/3 with 0 failures; final suite 585 passed / 2 documented Windows symlink skips |
| **3 — comparison campaign** | `complete` | Complete proposed/logistic/MLP/TTN matrix at the declared seed floor; complete Fashion-MNIST pooling transfer; aggregate paired statistics | `q1_comparison.json`: 4 tasks × 4 arms × 5 seeds, 80/80 complete, 0 failures; `q1_pooling_transfer.json`: 5 arms × 3 seeds, 15/15 complete, 0 failures; all manifests exactly 666/400/100/166; two-unit MLP capacity (1,573 parameters) is disclosed rather than called parameter-matched |
| **4 — local practicality evidence** | `complete` | Freeze statistics/tables; complete n=4,6,8,10 resources, pooling practicality, clean two-task noise ladder, and fake-backend rehearsal | `q1_resources.json`, `q1_pooling_practicality.json`, `q1_noise_validation.json` (6 records, zero-noise agreement pass), and `q1_fake_backend_rehearsal.json` all validate |
| **5 — core manuscript rewrite** | `complete` | Rewrite positioning, methods, theory, experiments, results, and limitations from canonical evidence | `fqcnn.tex` now uses the corrected protocol, cautious scope, exact Q1 table, and generated `figs_final/q1_comparison.png` |
| **6 — citations + final manuscript** | `complete` | Finish introduction/discussion/conclusion, sentence-level citation audit, complete claim ledger, and final PDF inspection | `q1_claim_ledger.json` (`ready_for_release`, no blockers); zero missing/uncited citation keys; three-pass IEEE journal LaTeX build: 9-page PDF, zero LaTeX errors/undefined references |
| **7 — reproduce + package** | `complete` | Run bounded clean-environment reproduction/health checks, reviewer-risk correction pass, and scoped release package | `q1_reproduction.json`: 632 tests, 628 passed, 4 documented skips, 0 failures; clean training and isolated Qiskit checks pass; verified 279-file archive is built from the clean release commit with SHA-256 recorded at handoff |

---

## 1. Milestone state

| Milestone | State | Gate | Evidence |
|---|---|---|---|
| **M0 — Phase 0: freeze + protocol** | `complete` | passed 2026-07-25; tag `phase-0-gate` | 63 tests; `docs/superpowers/plans/2026-07-25-fqcnn-phase-0-freeze-and-protocol.md` |
| **M1 — Phase 1: affordable execution** | `complete` | completed 2026-07-26; grid ≤7 nights passed; headline retrained clean at **98.29%** | §§3a–3e |
| M2 — Phase 2: pooling theory (E1–E5) | `running` | E1, E2, E3, and E4 complete; E5 blocked on isolated Qiskit tooling; E1 exact tie; E3 complete at 75 cells; coherent extension has no significant benefit; SU(4) remains headroom | `Results/evidence/e1_pooling_equivalence.json`, `e2_dephasing.json`, `e4_information_dynamics.json`, `t5_pooling_arms.json`; §§7, 7a–7c |
| M3 — Phase 3: model analysis | `running` | DLA, gradient variance, expressibility, generalization bound, and simulability complete; effective dimension/inductive bias blocked on M5.1; full su(2^10), dimension 1,048,575: **no polynomial-DLA certificate**; gradients support only “no barren plateau observed through n=14” | `Results/evidence/t6_dynamical_lie_algebra.json`, `f_e_gradient_variance.json`, `t6_expressibility.json`, `t6_generalization_bound.json`, `docs/simulability_statement.md`; §12 |
| M4 — Phase 4: harder datasets | `pending` | — | — |
| M5 — Phase 5: baselines + statistics | `pending` | — | — |
| M6 — Phase 6: ablation grid | `pending` | — | — |
| M7 — Phase 7: noise + real QPU | `pending` | one real-QPU point | — |
| M8 — Phase 8: resources + scaling | `pending` | — | — |
| M9 — Phase 9: references | `complete` | completed 2026-07-26; 0 author-less, 0 uncited, every published entry has a DOI; preprint ratio and sentence audit remain | `docs/paper_code_reconciliation.md`; §9 |
| M10 — Phase 10: manuscript + venue | `pending` | — | — |
| M11 — final reproduction + red team | `pending` | — | — |

### Parallel quality workstreams

- **Pooling transfer tooling:** `experiments/q1_pooling_transfer.py` and `tests/test_q1_pooling_transfer.py`; the 15-cell Fashion-MNIST `0,6` transfer run and aggregate are complete with common source-stable test identities.
- **Local practicality tooling:** `experiments/q1_local_evidence.py` and `tests/test_q1_local_evidence.py`; resource, pooling-practicality, six-record noise, and fake-backend rehearsal artifacts validate in the isolated Qiskit environment.
- **Manuscript audit:** `Results/evidence/q1_claim_ledger.json`; 23 claim records with the active manuscript bound to hashed evidence. `q1_claim_ledger_draft.json` remains diagnostic only and is excluded from release staging.

## 2. Active queue and blockers

**Active queue:** repair baseline leakage, align the experiment registry to the headline n=10 geometry, and isolate Qiskit tooling for E5/hardware work. M1 is complete; dataset, baseline, ablation, resource, noise, hardware, and manuscript claims remain pending unless backed by canonical artifacts.

| # | Blocker | Severity | Owner milestone | Status |
|---|---|---|---|---|
| B1 | `pool_measurement` did not implement the theorem's channel — fixed 2026-07-25; E1 passes (§7) | **High** | M2.1 | `complete` |
| B2 | Batched path equivalence proven 2026-07-25 before grid runs (§3a) | High | M1.1 | `complete` |
| B3 | Ablation grid runs n=8 while the frozen headline is n=10, so T4 would describe a different model; open and documented (§3d) | **High** | M5/M6 | `pending` |

## 3. M0 outcome (2026-07-25)

All eight sub-milestones pass. `python -m pytest tests/ -q` → 63 passed. `reproduce.sh` runs
the gate as step 0 before producing any result.

### Authoritative measurements

| Quantity | Value | Source |
|---|---|---|
| Input geometry | **28×28 = 784 flattened pixels; zero-pad to 1,024 amplitudes; n=10** | `Results/headline_manifest.json` |
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
| **Clean headline retrain** | **DONE 2026-07-26 -- test accuracy 0.98294** | §3e; `Results/metrics.json` |

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

## 3e. Clean headline retrain (2026-07-26) -- **98.29%**

The number that replaces 98.86%. One run of the frozen headline configuration under the
60/15/25 manifest protocol, split id `adbb1486`, 7,599 train / 1,900 val / 3,166 test,
disjoint verified. Selection on validation only; test read exactly once under
`TestEvaluationGuard`.

| Quantity | Value |
|---|---|
| **Test accuracy** | **0.98294** |
| Precision | 0.98340 |
| Recall | 0.98457 |
| F1 | 0.98399 |
| ROC-AUC | 0.99901 |
| Confusion | TP 1659 - TN 1453 - FP 28 - FN 26 |
| Best validation (selection) | 0.9821 at **epoch 4** |
| Stopping | early stop after **25** epochs, patience 3 |
| Wall-clock | 4,469.7 s (25 epochs) |

Evidence: `Results/metrics.json`, `Results/headline_retrain_summary.txt`,
`Results/manifests/idx_0v1_n12665_seed42.json`, weights `Results/Weights/run_seed42.npz`.

### What it cost to become defensible

The risk register rated "clean accuracy below 98%" as **high** likelihood, against **two**
headwinds: the leak removed, *and* 14% less training data (7,599 vs 8,865) because validation
had to come from somewhere. Both were absorbed for **0.57 points**.

The trajectory shows why the old number was inflated: validation peaked at 98.21% early, then
sat at 97.3--97.4% for a dozen epochs. Under the old protocol that peak was read off the **test**
set and reported. Now it only selected a checkpoint, and test -- consulted once, afterwards --
independently landed at 98.29%.

### NOT yet written into the manuscript, deliberately

`UPGRADE_PLAN.md` 5.3 is explicit: *"No single-run number appears anywhere in the paper again."*
This is **one seed**. Writing 98.29% into the abstract as a bare figure would swap one
single-run claim for another and violate 5.3 the day after M9 removed a different integrity
problem.

So 98.29% is the **protocol-clean reference number** -- it retires 98.86% for all internal
purposes and for reconciliation row 10 -- but the manuscript must quote **mean +/- std with a
95% CI across >= 5 seeds**, produced by Phase 5. The three `approx 98\%` claims in `fqcnn.tex`
stay untouched until that distribution exists.

**Caveat that travels with the number:** MNIST 0v1 is ~99.8% linearly separable, so 98.29% is
defensible but not impressive -- a logistic baseline should beat it. That comparison lands in
T3, and Phase 4's harder pairs are where the model has to earn its place.

## 3f. Manuscript pass on the clean run (2026-07-26)

Sec. IV/V still described the **leaked** protocol. `approx 98\%` was never the real problem;
the specifics around it were. All fixed against `Results/metrics.json` and the split manifest.

| Site | Was (leaked run) | Now (clean run) |
|---|---|---|
| Dataset prose + table | 11,430 balanced; 70/30; 8,001 / 3,429; no val split | 12,665 (5,923 / 6,742); 60/15/25; **7,599 / 1,900 / 3,166**, manifest `adbb1486` |
| Sec. V prose, Table III | precision 99.2%, recall 98.5%; 14 FP / 25 FN over 3,429 | precision **98.3%**, recall 98.5%; **28 FP / 26 FN** over **3,166** |
| Train / val rows | 99.0% / 98.7% | **98.0% / 98.2%** |
| Fig. 6 caption | 1,701 TN, 1,689 TP, 14 FP, 25 FN | **1,453 / 1,659 / 28 / 26** |
| Fig. 9 caption | precision 0.992, F1 0.989, bias 0.059, var 0.099 | **0.983 / 0.984 / 0.052 / 0.101** |
| Fig. 10 caption | 24 epochs, 78,682 s | **25 epochs, 4,470 s** |
| Epoch count | "early-stopped ≈ 24" | **25** ("Early Stopping triggered after 25 epochs") |
| Figures 5–10 | archived leaked run (dated Jul 21) | regenerated clean run |

The three `approx 98\%` claims are **deliberately unchanged** — `UPGRADE_PLAN.md` 5.3 forbids a
single-run figure in the paper, and Phase 5 owes mean ± std ± CI over ≥ 5 seeds. Note the
consequence: Sec. V is now *internally consistent single-run reporting*, which satisfies A6 but
not yet 5.3. Phase 5 must replace the whole block, not just the accuracy claims.

### Three defects found while doing it

1. **`fqcnn.tex` was corrupted and the build could not see it.** Five commands had lost their
   backslash to the heredoc bug: `\times` → TAB+`imes` in the **abstract**, and `\ref` →
   CR+`ef{` at four sites (Sec. I, Sec. V analysis, Sec. V comparison, Sec. VI-A). The PDF
   rendered literal `ef{thm:exact}` where "Theorem 1" belonged. **`pdflatex` reported zero
   undefined references throughout — because with the backslash gone there is no `\ref`
   command to be undefined.** M2.7's and M9's "0 undefined" build checks were therefore
   vacuous for these sites. Now: "Theorem 1" renders 9×, verified in the extracted PDF text,
   not just the log.

2. **The recorded label mapping was inverted.** `encode_labels` assigns
   `sorted(unique(y))[0] → -1`, so digit 0 → **−1**; `main.py` recorded the opposite in every
   manifest, and the paper stated the opposite in two places. Accuracy is symmetric and hid it;
   precision, recall and F1 were being attributed to the wrong digit. Confirmed arithmetically:
   the test split holds 1,481 zeros and 1,685 ones, and `tp + fn = 1,685`. Fixed via
   `splits.class_mapping_for()` (shared by `main.py` and `run_experiments.py`), pinned by a test
   that checks the mapping against `encode_labels` for six class orderings. The committed
   manifest's field was corrected in place; `class_mapping` is not in `manifest_id`'s payload,
   so split identity `adbb14862dfec2f7` is provably unchanged.

3. **`main.py` mislabelled the per-epoch plot "Test Accuracy"** while plotting
   `training_history['accuracy']`, which `Qtrainer` fills with **validation** accuracy. Cosmetic
   in effect but it reads exactly like a per-epoch test-set leak. Title corrected. **Fig. 5's PNG
   still carries the old title** and is marked `% FIGURE AUDIT` in the tex: re-plotting needs the
   run's per-epoch history, which is not persisted, so it lands on the next headline run.

Also corrected: STATUS previously said best validation was "0.9821 at epoch 24". It was
**epoch 4** — (991+875)/1900 = 0.98211; epoch 24 sat at 0.973. The run trained 21 further epochs
without improving on the checkpoint it had at epoch 4.

Dropped while editing: the "well-calibrated" clause in Sec. V's comparison paragraph, which
reconciliation row 13 already marks as unsupported. Noted rather than smuggled.

## 4. Run cells: required vs completed

| Workstream | Required cells | Complete | Milestone |
|---|---|---|---|
| Clean headline retrain | 1 | **1** | M1 |
| E1 / E2 (fixed-parameter, no training) | 2 | **2** | M2 |
| E3 pooling arms (arms × 5 seeds × 3 datasets) | 75 | **75** | M2; `Results/evidence/t5_pooling_arms.json` |
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
| T0 | Repository preservation | Task 1 | **recorded** at pushed SHA `43abb754b2f18177a2c2f920c142afcbb641fcf9`; `Results/evidence/repository_preservation.json` |
| T1 | Datasets + clean split protocol | M0.4, M4 | protocol done; table pending M4 |
| T2 | Resource table, prep vs model, transpiled | M8.1 | not started |
| T3 | Baselines, CIs, paired tests, cost columns | M5 | not started |
| T4 | Ablation Δacc ± CI | M6 | not started |
| T5 | Pooling arms + SU(4) ceiling + measurement tie | M2 | **generated** (§7c) |
| T6 | DLA, effective dim, expressibility, gen. bound | M3 | **partial** — DLA, expressibility + Q, and bound generated (§12); effective dim remains (blocked on M5.1) |
| F-A | E1 exact tie + E2 dephasing | M2.2–2.3 | **both generated** |
| F-B | Coherence / purity / entropy / MI per stage | M2.5 | **generated** (§7b) |
| F-C | Noise ladder ideal → fake → real + threshold | M7 | not started |
| F-D | Scaling family | M8.2 | not started |
| F-E | Gradient variance + DLA certificate | M3.1–3.2 | **both generated** (§12) — but there is **no certificate**: the DLA is the full su(2ⁿ) |
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
| Proposition 3 | `test_e2_dephasing_discarded_wires_changes_nothing` | E2: **5.86e-14**, ΔAcc **0.0 exactly** |

### The information-loss claim is deleted

It appeared in **six** places. All replaced with Theorem 1's defensible claims — exact
simulation in the frozen construction, no mid-circuit measurement/reset/feed-forward, a globally
pure state, end-to-end adjoint differentiability:

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

## 7b. E4 — information dynamics (2026-07-25)

`python -m experiments.pooling_analysis --experiment e4`. Headline n=10, archived weights,
8 inputs, mean ± std, entropies in bits. Snapshots at each stage boundary on `default.qubit`;
the main path is unitary so a statevector device suffices and the retained register's entropy
*is* its entanglement entropy with the rest.

| Stage | kept | ℓ1-coherence | purity | entropy | I(keep:disc) |
|---|---|---|---|---|---|
| encoded | 10 | 767.26 ± 7.23 | **1.0000 ± 0.0000** | **0.0000 ± 0.0000** | — |
| before_pool_0 | 5 | 5.906 ± 0.206 | 0.0919 ± 0.0037 | 3.9070 ± 0.0321 | **7.814 ± 0.064** |
| after_pool_0 | 5 | 6.567 ± 0.170 | 0.0942 ± 0.0030 | 3.8901 ± 0.0309 | 7.780 ± 0.062 |
| before_pool_1 | 2 | 0.819 ± 0.051 | 0.3356 ± 0.0051 | 1.7565 ± 0.0120 | 0.204 ± 0.009 |
| after_pool_1 | 2 | 0.697 ± 0.038 | 0.3134 ± 0.0044 | 1.8214 ± 0.0111 | 0.358 ± 0.013 |
| before_pool_2 | 1 | 0.403 ± 0.017 | 0.5923 ± 0.0072 | 0.8624 ± 0.0111 | **0.005 ± 0.002** |
| after_pool_2 | 1 | 0.332 ± 0.022 | 0.5879 ± 0.0074 | 0.8691 ± 0.0114 | 0.016 ± 0.004 |
| after_classifier | 1 | 0.253 ± 0.016 | 0.5879 ± 0.0074 | 0.8691 ± 0.0114 | — |

The measures are validated against analytically known states before any circuit output is
believed — `tests/test_state_metrics.py`, 16 tests over $|0\rangle$, $|+\rangle$, $I/2$, Bell
and GHZ.

### What the data says — descriptive, not causal

1. **A3's purity invariant is now measured, not asserted.** After encoding, the 10-wire state
   has purity 1.0000 and entropy 0.0000. The global state is pure, exactly as the revised
   claims require.

2. **But the readout qubit is nearly maximally mixed** — purity 0.588 against 0.5 for a
   maximally mixed qubit, entropy 0.869 of a possible 1 bit. "Fully coherent" is true
   *globally* and false *locally*. This supports the M2.7 rewrite, which claims global purity,
   and would have contradicted a naive reading of the old wording.

3. **Pooling barely moves the retained register's entropy** — by ≤ 0.07 bits at every stage,
   and at pools 1 and 2 it *increases* slightly. There is no entropic evidence that pooling
   concentrates information into the kept wire. The paper's "transfers information from $q_b$
   into $q_a$" should stay a description of the gate pattern, not an information-theoretic
   claim.

4. **The pooling cascade runs out of correlation to act on.** I(keep:discard) falls
   **7.81 → 0.20 → 0.005 bits** across the three stages. At pool 2 the two subsystems are
   essentially uncorrelated, so the last pooling stage has almost nothing left to transfer.
   Read alongside F1 — convolution layers 1–3 contribute zero effective parameters — this says
   the deep end of the "hierarchy" does very little work at n=10. Relevant to M8's scaling
   argument and to how Sec. III-C describes the architecture.

5. **Instrumentation self-check.** Across the classifier, purity and entropy are unchanged to
   1e-10 while coherence moves (0.332 → 0.253). With one active wire the head is a local
   unitary, which cannot change a reduced state's spectrum but can change a basis-dependent
   coherence measure. This is asserted as a test, so mislabelled stage states would fail.

**Defect fixed while writing this:** `von_neumann_entropy` returned −8.0e-16 for a pure state,
because the surviving eigenvalue is $1+O(\epsilon)$ and $-p\log_2 p$ then goes slightly
negative. Entropy is non-negative by definition, so a negative is a defect in the reported
number, not a property of the state. Clamped, with a regression test.

Evidence: `Results/evidence/e4_information_dynamics.json` (per-stage aggregates plus per-input
detail). This is the F-B source data.

## 7c. E3 -- retrained head-to-head, T5 (2026-07-26)

75 cells: 5 arms x 5 seeds x 3 datasets (0v1, 3v5, 4v9), headline n=10 geometry, 400 samples,
30 epochs, matched budgets. 2,130 pooled test items. Reference arm is the frozen block.

| Arm | Slots | Accuracy | 95% CI | delta | McNemar p | Wilcoxon p |
|---|---|---|---|---|---|---|
| `pool_none` | 224 | 0.8507 +/- 0.0811 | [0.811, 0.892] | **-0.0272** | **0.0006 \*** | **0.0024 \*** |
| `pool_measurement` | 269 | 0.8779 +/- 0.0723 | [0.843, 0.915] | **+0.0000** | 1.000 | 1.000 |
| **`pool_unitary`** (frozen) | **269** | **0.8779 +/- 0.0723** | [0.843, 0.915] | (ref) | -- | -- |
| `pool_coherent` | 284 | 0.8798 +/- 0.0695 | [0.847, 0.916] | +0.0019 | 1.000 | 1.000 |
| `pool_su4` | 449 | 0.9019 +/- 0.0659 | [0.869, 0.935] | +0.0239 | **0.0020 \*** | 0.500 |

Holm-Bonferroni adjusted across the four compared arms; \* significant at alpha = 0.05.

Per dataset: 0v1 / 3v5 / 4v9 -- none 0.951/0.762/0.839 - frozen 0.966/0.807/0.861 -
coherent 0.963/0.810/0.866 - su4 0.972/**0.863**/0.870.

### Readings

1. **The measurement arm ties exactly** -- **zero discordant pairs out of 2,130**, not merely a
   non-significant difference. Theorem 1 holds in a *trained* setting, not only at fixed
   parameters. **But this row is a consistency check, not independent evidence:** on a
   statevector device PennyLane realises mid-circuit measurement by deferring it back into
   controlled gates, so the two arms execute the same thing. E1's Kraus channel on
   `default.mixed` remains the independent confirmation.
2. **Pooling earns its place.** `pool_none` is 2.7pp worse, significant on both tests
   (143 vs 85 discordant favouring the frozen block).
3. **Proposition 2's headroom is empty at one gate.** `pool_coherent` gains +0.19pp with
   near-symmetric discordance (107 vs 111) and p = 1.0 on both tests. It leaves the
   measurement-simulable class mathematically and gains nothing operationally.
4. **The frozen block is NOT the best arm.** `pool_su4` is +2.4pp overall and +5.6pp on 3v5,
   the hardest pair. Anticipated by the risk register: report family headroom, retain the
   frozen headline. The paper therefore cannot claim the frozen block is optimal. The Q1
   manuscript reports it as the frozen reference construction and binds resource statements
   to the local counts generated for each declared pooling arm.
5. **The two tests disagree about SU(4).** McNemar p = 0.002, Wilcoxon p = 0.50. McNemar treats
   2,130 items as observations; Wilcoxon has 15 seed-level scores and far less power, but is
   arguably the more appropriate unit for a generalisation claim. Honest wording: a consistent
   per-example advantage that seed-level variation does not confirm.

### Caveats attached to T5

- **~0.88 is not the headline accuracy.** These are 343-sample, 30-epoch matched-budget runs;
  T5 is an arm comparison. The clean headline retrain uses ~7,600 training samples.
- **Watch 0v1.** The frozen arm reaches 0.966 there, while logistic regression on 0v1 is
  ~99.9% and would likely clear 99% even at this budget. T3 may show the quantum model losing
  to a linear baseline on the easy pair -- which is exactly why Phase 4 moves to hard pairs.

### Method note

McNemar is pooled across the 15 (dataset, seed) splits by summing discordant counts. The splits
are disjoint, so each discordant pair is an independent Bernoulli trial under the null; per-split
counts are retained in the artifact so the pooling can be checked. `experiments/statistics.py`
uses the **exact** binomial test rather than the chi-square approximation, because discordance
is often zero here -- Theorem 1 guarantees it for one arm -- and chi-square is unreliable there.

Evidence: `Results/evidence/t5_pooling_arms.json`.

## 12. Phase 3 — model analysis: 3.1, 3.2, 3.4, 3.5, 3.7

`python -m experiments.model_analysis --experiment {dla,gradient_variance,expressibility}`

Runner follows `pooling_analysis.py`: a `run_*` per item, `--experiment` CLI, JSON into
`Results/evidence/`. Both measures were pinned against analytically known cases first —
`variance_decay_fit` is new and has 6 tests recovering exact exponential and power-law series.

### 3.1 — dynamical Lie algebra (**T6 / F-E**, 2026-07-31)

`python -m experiments.model_analysis --experiment dla`. Exact closure by iterated commutators
in the Pauli basis, 1,658 s total, no overnight compute. Evidence:
`Results/evidence/t6_dynamical_lie_algebra.json`.

**Exact closure was reached at the headline n=10 for all three generator sets** — 1,048,575
basis elements enumerated, not a truncation and not an extrapolation. n=12 and n=14 are capped
and are reported as lower bounds.

#### "The ansatz's generators" is ambiguous, and the ambiguity is the finding

The same trap 3.5 hit with Caro's *T*. `UPGRADE_PLAN.md` 3.1 says "the Lie closure of the frozen
ansatz's generators". The frozen tape carries **222 trainable gates and 104 fixed ones** (96
CNOTs in the convolution kernel plus the 8 inert `RY(0.02)`s), so that phrase names three
different algebras and they do not agree:

| Generator set | What it is | dim at n=10 |
|---|---|---|
| `parameterized` | trainable-gate generators as they appear; the literal spec | **65,550** |
| `propagated` | the same, conjugated through the fixed prefix, so the circuit really is `F · ∏ exp(−iθₖH̃ₖ)` — **the algebra the Ragone/Fontana variance expressions are stated over** | **1,048,575** |
| `full` | trainable plus fixed-gate generators, so `exp(g)` provably contains the circuit — **the algebra g-sim simulability needs** | **1,048,575** |

su(2¹⁰) has dimension 1,048,575. So the two algebras that any theorem here quantifies over are
**the entire special unitary algebra**, and the literal reading understates the circuit by a
factor of 16. Reporting 65,550 as "the DLA" would be quoting Ragone et al. outside their
hypotheses, because the 96 CNOTs it ignores are not optional parts of the circuit.

The propagation is checked against the circuit itself, not asserted:
`test_propagated_generators_reconstruct_the_circuit` rebuilds the frozen unitary at n=4 as
`F · ∏ exp(−iθₖH̃ₖ/2)` and matches to 1e-9. A wrong conjugation order would otherwise yield a
generator set belonging to no circuit.

#### The sweep

| n | su(2ⁿ) ceiling | `parameterized` | `propagated` | `full` |
|---|---|---|---|---|
| 4 | 255 | 255 | 255 | 255 |
| 6 | 4,095 | 270 | 4,095 | 4,095 |
| 8 | 65,535 | 65,535 | 65,535 | 65,535 |
| **10** | **1,048,575** | **65,550** | **1,048,575** | **1,048,575** |
| 12 | 16,777,215 | 65,790 | ≥120,000 | ≥120,000 |
| 14 | 268,435,455 | 65,805 | ≥120,000 | ≥120,000 |

#### 1. There is no trainability certificate, and this is the opposite of what 3.1 was for

`UPGRADE_PLAN.md` 3.1's premise was "polynomially-sized DLA ⇒ no barren plateau, upgrading the
paper's Pesah et al. citation from analogy to a computed certificate for this exact circuit."
**The premise fails.** dim g = 4ⁿ − 1 is exponential, so the Ragone et al. (2024) / Fontana et
al. route certifies nothing; applied under its own 2-design hypothesis its exact variance
expressions would predict a plateau, not rule one out.

So the definition-of-done item *"Trainability certificate (DLA + variance sweep)"* **cannot be
closed by its DLA half** and stays unticked (§9). The Pesah et al. citation remains an analogy.
What the manuscript may now say is stronger than silence and weaker than a certificate: the DLA
was computed exactly for this circuit, it is the full su(2ⁿ), and therefore the
poly-DLA argument is unavailable in either direction.

#### 2. The simulability half, reported because it is the other face of the same number

A polynomially-sized DLA would have implied the *family* is efficiently classically simulable
at arbitrary n via the Lie-algebraic (g-sim) results — the consequence `docs/simulability_statement.md`
§4 recorded as open and load-bearing. It does not arise: g-sim needs the circuit inside
`exp(g)` for a small `g`, and here `g` is everything.

**This is not a hardness result and must not be written as one.** It closes one
efficient-simulation route. The tape is shallow (327 operations, three pooling stages), which is
exactly the regime where tensor-network simulation may still succeed. Simulability at arbitrary
n is **open, with one route ruled out**. `docs/simulability_statement.md` §4 is updated
accordingly; §1's admission is untouched, because at ten qubits simulability follows from 1,024
amplitudes and never depended on the DLA.

#### 3. It discriminates the 3.2/3.4 tension: (a) is refuted

The recorded reconciliations of "Haar-indistinguishable at n=10, yet gradients survive to n=14":

- **(a) "only 74 of 269 slots are live, so the *effective* ansatz is far smaller than the
  nominal one" — refuted.** The live parameterisation generates the **entire** su(2¹⁰). A sparse
  live-slot count does not shrink the reachable algebra, so the effective-parameter audit cannot
  be the mechanism that saves trainability. This is the one reconciliation the measurement
  settles, and it settles it against the reading that was most convenient.
- **(b) "n ≤ 14 is short of asymptotic" — survives, and is sharpened into something different.**
  The circuit has **222 trainable gates against a DLA of dimension 1,048,575**, and the gate
  count grows roughly linearly in n while dim g grows as 4ⁿ. The gap widens with n rather than
  closing, so this family does not approach the asymptotic regime *at any n* — it is not that
  n=14 is too small.
- **(c) "the expressibility measure saturates at n=10" — survives** on 3.4's own controls
  (Haar holds 0.9999989 of its mass in the first of 75 bins).

The logic that ties this together, stated as modus tollens so it does not overreach: an
exponential DLA means that **if** the ansatz formed a 2-design over `exp(g)`, the variance would
be exponentially suppressed. 3.2 measures that it is not, to n=14. Therefore the 2-design
premise fails for this circuit at these depths. Both Holmes et al.'s expressibility argument and
the DLA variance expressions are statements about ansätze that randomise over their group, and
this one does not. **The manuscript may now say why the plateau predictions do not bite here,
rather than choosing silently among (a)/(b)/(c).**

#### 4. Where the algebra actually comes from — an architectural finding

The `parameterized` algebra is exactly a direct sum of full su blocks over the **pooling**
connectivity graph, at every n (`sum(4^|C| − 1)` matches the enumerated dimension in all six
cases):

| n | components | dimension |
|---|---|---|
| 8 | [8] | 65,535 |
| **10** | **[0–7], [8,9]** | **65,535 + 15 = 65,550** |
| 12 | [8], [4] | 65,535 + 255 = 65,790 |
| 14 | [8], [4], [2] | 65,535 + 255 + 15 = 65,805 |

The largest block is **pinned at 8 wires from n=8 onward**, because there are exactly three
pooling stages and the merge tree therefore cannot join more than 2³ wires. So the
parameterized-only algebra grows merely *linearly* in n past that point — "polynomial", with a
constant of 65,535, which is why polynomial scaling is worthless as a claim without its constant.

The two ablation arms attribute this (capped at 120,000 with the su(2ⁿ) certificate supplying the
exact value; the frozen row is the exact enumeration above):

| Arm at n=10 | `parameterized` | `propagated` | `full` |
|---|---|---|---|
| frozen | 65,550 | 1,048,575 | 1,048,575 |
| `pool_none` | **30** (= 3n, local su(2) only) | 1,048,575 (certificate) | 1,048,575 (certificate) |
| `ent_none` | 65,550 | **65,550** | **65,550** |

Two readings, both about the frozen architecture rather than about Lie theory:

1. **Pooling is the model's only trainable entangler.** Remove it and no parameterised gate is
   entangling at all — the algebra collapses to local su(2) on ten wires. The convolution layer
   contributes trainable *rotations* and *fixed* entanglement, nothing trainable and entangling.
   Read next to E3, where `pool_none` is 2.7pp worse and significant on both tests, this says
   what the pooling block is doing structurally.
2. **The exponential size is bought entirely by non-trainable gates.** Delete the 96 fixed CNOTs
   (`ent_none`) and all three algebras drop from 1,048,575 to 65,550 and stay block-diagonal —
   a factor of 16. The circuit's reach over state space comes from gates that have no parameters
   and were never presented as a design choice.

#### 5. Controls — what makes a dimension readable

Carrying 3.4's habit. Two reference ansätze with independently known dimensions were closed by
the same routine at **the same n**, in the same run:

| Control | Known dimension | At n=10 | Agrees |
|---|---|---|---|
| local rotations only | 3n (su(2)^⊕ⁿ) | 30 | yes |
| open-chain transverse-field Ising | 2n² − n (Wiersema et al. 2024) | 190 | yes |

Both agree at every n from 4 to 14. The Ising arm is the load-bearing one: it is a **polynomial**
answer from the same code that returns 1,048,575 for the FQCNN, so the exponential result is a
property of the circuit and not of the closure routine. In `tests/test_lie_algebra.py` the
primitive is additionally pinned against su(2) = 3, su(2)^⊕ⁿ = 3n, the universality result
4ⁿ − 1, TFIM at n = 2,3,4, and — the control a formula pin cannot give — **an independent
cross-check against PennyLane's `qml.pauli.lie_closure`**, which catches an error shared between
this code and my reading of the references.

#### 6. Numerical hygiene, checked rather than assumed

The one hazard was the propagated set: conjugation through the fixed `RY(0.02)`s could drive
coefficients toward `sin(0.02)^k`, and a pivot normalised by a near-tolerance coefficient
amplifies round-off. **It does not occur.** Propagated generators carry at most **2** Pauli terms
with minimum |coefficient| **0.5** — 5×10⁸ above the 1e-9 dependence tolerance — because CNOT
conjugation is Clifford (one Pauli string to one Pauli string), and every `RY(0.02)` sits on a
discarded wire that no later trainable gate touches. It is inert under propagation for the same
reason decision 1 found it inert in the circuit. Recorded in the artifact as
`generator_conditioning` rather than asserted.

#### Caveats that travel with the number

- **n=12 and n=14 are lower bounds (≥120,000), not closures.** 4¹² − 1 = 16.7M elements is out
  of reach here. Exactness stops at n=10, which is the headline, and the sweep says so.
- **The DLA is a property of the ansatz**, not of the trained weights and not of the data. It
  says nothing on its own about the loss landscape at the archived parameters.
- `AmplitudeEmbedding` is **excluded** from the generators: it carries the input, so it belongs
  to the state the ansatz acts on. Folding its ~2,026-CNOT Möttönen decomposition in would
  describe a different object with data-dependent angles no theorem here quantifies over.

### 3.2 — gradient variance (**F-E generated**)

200 parameter initialisations × 4 random inputs = 800 gradient draws per qubit count,
parameters uniform [0, 2π), backprop on `default.qubit`. 1,367 s serial (n=14 alone is 894 s).

| n | slots | live | median Var (live) | **designated conv0 Var** | mean \|grad\| |
|---|---|---|---|---|---|
| 4 | 188 | 62 | 2.816e-2 | **1.335e-2** | 1.287e-1 |
| 6 | 194 | 62 | 1.413e-2 | **9.270e-3** | 8.788e-2 |
| 8 | 260 | 122 | 2.910e-3 | **4.416e-3** | 4.653e-2 |
| 10 | 269 | **74** | 3.768e-3 | **1.889e-3** | 4.092e-2 |
| 12 | 278 | 128 | 3.334e-4 | **7.304e-4** | 1.619e-2 |
| 14 | 287 | 74 | 2.951e-3 | **1.037e-3** | 3.352e-2 |

Controlled-slot fits: exponential `Var ≈ 0.0466 exp(−0.3035 n)`, **R² = 0.9229**; power law
`Var ≈ 0.5162 n^−2.4296`, **R² = 0.9152**.

1. **The data cannot identify the decay law.** ΔR² = 0.008 across two qualitatively different
   laws on six points is noise, and the two reductions actively disagree — the median series
   prefers the power law (R² = 0.66), the controlled slot prefers the exponential (R² = 0.92).
   **F-E may not claim a barren plateau, nor claim its absence asymptotically.** What it can
   claim: no plateau is *observed* to n=14 — gradients stay O(10⁻²) and variance falls about
   one order of magnitude per five qubits, which is trainable in the tested range.
2. **The median-over-live-slots reduction is confounded** and should not be quoted. The live
   count is non-monotone — 62, 62, 122, 74, 128, 74 — so the median is taken over a slot
   population that changes size and group composition with n; n=8 and n=12 carry ~2× the live
   slots of their neighbours and correspondingly ~10× lower medians. This was written into the
   evidence as a caveat *before* the run, and it is the operative one. The designated conv0
   slot (same index, same role at every n) is the series to trust.
3. **n=14 breaks monotonicity** even on the controlled slot (1.037e-3 against 7.304e-4 at n=12),
   so the tail is not clean and the fit rests on a series that turns at its last point.
4. **Independent confirmation of the 269/74 audit.** At n=10 exactly **74** slots are live,
   reproducing M0's effective-parameter count from a *different criterion* (variance over 800
   random-parameter draws vs max \|gradient\| over 20 inputs) at *different weights* (random vs
   archived). The effective-parameter count is therefore a structural property of the circuit,
   not an artifact of the trained model — which strengthens the F1 disclosure considerably.
   Only the counts were compared; the runner now records `live_slots` so the *sets* can be
   compared on the next run.

### 3.4 — expressibility and entangling capability (**T6 partial**)

3,000 parameter pairs at headline n=10, input held fixed (the ansatz's expressibility, not the
data's), state captured through the `terminal` hook. 343 s.

| Quantity | Value |
|---|---|
| Expressibility KL (75 bins) | **1e-6** |
| **Haar reference KL**, same sample size and binning | **1e-6** |
| Mean output fidelity | 9.7423e-4 |
| Haar mean fidelity | 9.7656e-4 |
| **Meyer-Wallach Q** | **0.9745 ± 0.0135** (range 0.895–0.995) |

**The controls are what make this readable.** A KL of 1e-6 means nothing on its own, because a
finite sample of the Haar law itself does not score zero. Scored identically:

| Control | KL | Reading |
|---|---|---|
| Haar law sampled by inverse CDF | **1e-6** | the ansatz matches this exactly |
| Fidelities inflated 10× | 3.82 | measure still has power at n=10 |
| Degenerate ensemble (F ≈ 0.9) | 25.81 | gross failure is detected |
| Haar mass in first of 75 bins | 0.9999989 | the resolution limit |

So the ansatz's output ensemble is **indistinguishable from Haar-random at this measure's
resolution**, and that resolution is roughly an order-of-magnitude deviation in mean fidelity —
not finer. Stating it as "maximally expressible" would overclaim; the honest form is
"Haar-indistinguishable at n=10 to within a measure that separates ~10× deviations".

**This sits in tension with 3.2, and the tension is the interesting part.** Holmes et al. (2022)
tie high expressibility to barren plateaus, so a Haar-indistinguishable ansatz at n=10 is
precisely where a plateau is expected — yet 3.2 sees gradients surviving to n=14. Three
candidate reconciliations, in order of how much support they currently have:
(a) only 74 of 269 slots are live, so the *effective* ansatz is far smaller than the nominal
one, and the expressibility of the full parameterisation is not what governs its gradients;
(b) n ≤ 14 is short of asymptotic;
(c) the expressibility measure saturates at n=10, so "Haar-like" is a weaker statement than it
sounds. These are hypotheses, recorded so the manuscript does not silently pick one. The DLA
(3.1) is the item that would discriminate.

### 3.5 — generalization bound (**T6**, 2026-07-27)

Caro et al. (2022) `sqrt(T log T / N)` at the frozen headline. N = 7,599, read from the clean
split manifest `idx_0v1_n12665_seed42.json` rather than hardcoded. Runs in under a second.

| Reading of T | T | `sqrt(T logT / N)` |
|---|---|---|
| **Trainable gates on the tape** | **222** | **0.3973** |
| Trainable gates driven by effective slots | 218 | 0.3930 |
| Effective parameter slots | 74 | 0.2047 |
| Allocated parameter slots | 269 | 0.4450 |

**The planned inputs were wrong, and this is the finding.** The prior note here said "plug
T=74/269 → ~0.20 vs ~0.45, which is the parameter-frugality argument in one line". That
argument does not survive, because **Caro's T counts trainable gates, not parameters**, and
the two disagree in both directions at once:

- **191 of the 269 allocated slots never reach the tape.** Only **78** appear in the frozen
  circuit signature at all — the dead conv groups allocate 48 each and are never read, and the
  classifier allocates 32 it indexes modularly. So 269 overstates the circuit badly.
- **48 slots each drive more than one gate** (up to 4), through that same modular indexing. So
  the 74 effective slots actually drive **218** gates, and 74 understates the circuit.

In gate terms the effective-vs-allocated gap is **218 vs 222** — 0.3930 vs 0.3973, a difference
of about 1%. **The dramatic 269→74 reduction nearly vanishes once T is counted correctly**,
because the slots it removes were never on the circuit and the ones it keeps are reused. The
parameter-frugality claim can still be made as a *parameter-count* statement, but it must not
be attached to this bound.

The tape audit is cross-checked against the M0 effective-parameter fixture from an independent
code path: both give **78** syntactically-used slots
(`test_headline_tape_audit_agrees_with_the_effective_params_fixture`).

All four readings are non-vacuous (< 1), but only up to Caro's unquantified constant — the
theorem is big-O, and the manuscript must say so rather than reporting 0.3973 as a gap.

Evidence: `Results/evidence/t6_generalization_bound.json`. Suite: **210 passing**.

### 3.7 — classical-simulability statement (**DONE 2026-07-29**)

Prose only, no compute. Drafted into `docs/simulability_statement.md`; **not** installed in
`fqcnn.tex` — that is M10.3, and the Track B contract keeps both tracks out of the manuscript
until then. Suite unchanged at **210 passing**.

The admission: the headline model is 10 qubits — a 1,024-amplitude state, 16 KB in
`complex128` — so it is exactly and cheaply classically simulable, and **every number in this
paper is a classical simulation** (`default.qubit` backprop for training and Phase 3,
`lightning.qubit` adjoint as the sequential oracle, `default.mixed` for E1/E2 and `noise_sim`).
**No quantum-advantage claim is made**, and MNIST 0v1 being ~99.8% linearly separable means the
evidence could not support one. Two of the three reasons simulation is *easy* are architectural
rather than incidental — the main path is unitary end to end (A3, permitted by Theorem 1) so a
state-vector simulator suffices, and the tape is shallow (327 operations, 222 trainable gates).

The framing that earns its place: simulability is the **precondition** for the evidence, not a
weakness of it. Theorem 1's 2.2e-16 tie is readable only because the state can be computed
exactly two ways.

**Two corrections to the `UPGRADE_PLAN.md` 3.7 spec, applied rather than copied:**

1. The spec's "~76-parameter" is stale, and 3.5 showed the ambiguity behind it is itself a
   finding. The statement uses the audited figures and always names the quantity: 269 allocated
   slots · 78 on the tape · **74 effective** · **222 trainable gates** · 218 gates from effective
   slots · 327 total operations. The two families disagree in both directions, so "~76
   parameters" is not a safe shorthand anywhere in the manuscript.
2. The spec pairs the admission with "and characterises its scaling (Phase 8)". **M8 has not
   run** — the scaling sweep and resource table are both at zero cells (§4) — so the scaling
   half is written as pending, not asserted. Likewise **there is no hardware result**: M7.3's
   real-QPU point is unrun and the QPU allocation stands at 0 minutes consumed (§5).

Recorded as open in the draft, because they bear on how strong the admission should be: **3.1
(DLA) is the item that could change it qualitatively** — 10 qubits is simulable for trivial
reasons, but a polynomially-scaling DLA would imply the *family* is efficiently simulable at
arbitrary n via the Lie-algebraic simulation results, which is a finding to report rather than
bury. And 3.4's Haar-indistinguishability is a statement about the output ensemble, not about
simulation hardness, so it must not be recruited as evidence of "quantumness".

One resource caveat travels with the statement: `AmplitudeEmbedding` at n=10 decomposes into a
~2,026-CNOT Möttönen sequence (§7), so on real hardware state prep, not the model, would bind.
Quantifying that split is M8.1's and is recorded as an observation, not a number.

**Not yet done in Phase 3:** 3.3 effective dimension and 3.6 inductive bias, both of which need
the empirical Fisher and matched MLP/CNN controls and stay **gated on M5.1** (see the Track A
ordering). 3.1 landed 2026-07-31 and answered this section's open question in the negative —
see 3.1 above.

## 9. M9 bibliography audit (2026-07-26)

| Metric | Before | After |
|---|---|---|
| Entries with **no author list** | **22 of 51 (43%)** | **1** (ref27, flagged) |
| arXiv-only, no DOI | 21 (41%) | 17 (33%) |
| Missing DOI | 51 (100%) | 37 (73%) |
| Preprints upgraded to published versions | -- | **5** |

21 entries rewritten with metadata verified against the publisher record or the arXiv listing.
Nothing was filled in that could not be checked. `pdflatex` x3: 12 pages, 0 undefined.

### The audit found errors, not just gaps

**ref1 cites the wrong paper entirely.** `arXiv:2006.12763` is *"Method of fundamental solutions
for the problem of doubly-periodic potential flow"* by H. Ogata -- computational fluid dynamics,
unrelated to the "tutorial on quantum machine learning" it is cited as. Every other arXiv
identifier (ref2--ref13, ref23, ref28--ref36) was fetched and confirmed against its cited title
and first author; ref1 is the only bad one.

Three entries had the **wrong venue**, one also the wrong year:

| Entry | Cited as | Actually |
|---|---|---|
| ref15 | Nat. Commun. 2022 | npj Quantum Information 8, 90 (2022) |
| ref16 | Nat. Commun. 2025 | npj Quantum Information 11, 8 (2025) |
| ref20 | IEEE TNNLS 2022 | **IEEE Access 8, 188853--188860, 2020** |

Four had truncated or altered titles (ref15, ref16, ref17, ref31).

### Blocking author decisions

1. **ref1** -- wrong identifier, never cited.
2. **ref27** -- "Quantum reservoir computing implementation on superconducting circuits" matches
   no findable paper. Closest by topic/venue/year is Dudas et al., npj Quantum Inf. 9, 64
   (2023), but substituting would be guessing at intent, so it was **not** done.

Both carry `% M9 AUDIT` comments in `fqcnn.tex` so they cannot ship unnoticed.

### Uncited entries

**12 of 51 (24%) are never cited:** ref1, ref5, ref9, ref12, ref24, ref27, ref28, ref30, ref32,
ref33, ref34, ref35. IEEE style does not permit uncited references, so each must be cited or
removed -- an editorial call for the authors. Both problem entries above fall in this set,
which is consistent with them being padding added without verification.

### Remaining

DOIs for 37 entries; the two decisions above; the 12 uncited entries; and the sentence-level
citation audit (Phase 9 item 6), including `ref44` propping up the image-locality claim that
reconciliation row 11 already flags.

### Second pass (2026-07-26): uncited entries removed, DOIs completed

On author instruction the 12 uncited entries were deleted rather than cited. That disposed of
both unresolved items without any guesswork -- ref1 (wrong identifier) and ref27 (unfindable
title) were themselves uncited -- so the Dudas et al. substitution was never needed.

| Metric | Original | Final |
|---|---|---|
| Entries | 51 | **39** |
| No author list | 22 (43%) | **0** |
| Uncited entries | 12 (24%) | **0** |
| Undefined citations | 0 | 0 |
| Published entries missing a DOI | all | **0** |
| Entries missing any DOI | 51 (100%) | 11 (28%) -- all genuine preprints |
| Preprints upgraded to published versions | -- | 5 |

Deleted: ref1, ref5, ref9, ref12, ref24, ref27, ref28, ref30, ref32, ref33, ref34, ref35.

18 DOIs added, each from a Crossref record or publisher page and cross-checked against the
entry's existing title and venue. One near-miss worth recording: a fuzzy Crossref query for
ref44 returned *"Entanglement-Induced Barren Plateaus"* -- a different paper -- so that hit was
discarded and the DOI resolved separately. Fuzzy bibliographic matching is not verification.

`pdflatex` x3: 12 pages, zero undefined references or citations.

**Remaining Phase 9 item:** preprints are 28% of the list against the ~20% target. All 11 are
2024--25 submissions with no journal version yet, so closing the gap means citing fewer
preprints -- an editorial choice, not a metadata fix. The sentence-level citation audit
(Phase 9 item 6) also remains, including `ref44` propping up the image-locality claim flagged
in reconciliation row 11.

## 8. Architecture sign-off decisions (roadmap §18)

| # | Decision | State | Outcome |
|---|---|---|---|
| 1 | Inert `RY(0.02)` removal | **Closed 2026-07-25** | **Retained.** Proven inert (9.44e-16 angle change, 8.33e-16 removal). Removal-with-proof branch was available and **declined** in favour of zero architecture change; documented as a no-op. See `docs/paper_code_reconciliation.md` row 4. |
| 2 | `pool_coherent` implementation | **Closed 2026-07-26** | **Approved as an ablation arm.** Ran in E3; gains +0.19pp, not significant. Never headline; promotion would need a second sign-off (A5). |
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
- [x] Theorem 1 / Props 2–3 written with proofs; E1 tie confirmed to ~1e-12; E2 complete
- [ ] ≥3 datasets × ≥5 seeds; all numbers mean ± std with CIs and paired tests
- [ ] Every comparison row reproduced on your split or explicitly out-of-table
- [ ] Resource table with state-prep/model split; scaling family n=4…14
- [ ] Trainability certificate (DLA + variance sweep) and generalization bound — **the DLA half
  cannot close this.** 3.1 computed it exactly and it is the full su(2ⁿ) (§12), so the
  poly-DLA no-plateau argument does not apply. The variance sweep and the bound are done; what
  remains achievable here is a *measurement* ("no plateau observed to n=14") and not a
  certificate. Reword or drop at M10, do not tick.
- [ ] Noise ladder incl. one real-QPU point, scoped and caveated
- [ ] `reproduce.sh` regenerates every table and figure from a clean checkout
- [ ] Bibliography: full metadata, DOIs, peer-reviewed versions, no author-less entries

**8 of 15 complete.**

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
