# FQCNN Q1 Upgrade — Program Status

Roadmap §20 program dashboard. Update at every gate. One row per milestone.

**Governing spec:** `UPGRADE_PLAN.md` (v2)
**Roadmap:** `docs/superpowers/plans/2026-07-23-fqcnn-q1-upgrade-roadmap.md`
**Design:** `docs/superpowers/specs/2026-07-23-fqcnn-q1-upgrade-design.md`
**Month plan:** `docs/superpowers/plans/2026-07-25-fqcnn-remaining-work-month-plan.md`

**Last updated:** 2026-07-25 · **Branch:** `plan/fqcnn-q1-upgrade` · **HEAD:** `3cdb292`

---

## 1. Milestone state

| Milestone | State | Gate | Evidence |
|---|---|---|---|
| **M0 — Phase 0: freeze + protocol** | **PASSED 2026-07-25** | tag `phase-0-gate` | 63 tests; `docs/superpowers/plans/2026-07-25-fqcnn-phase-0-freeze-and-protocol.md` |
| **M1 — Phase 1: affordable execution** | **IN PROGRESS** (started 2026-07-25) · 1.1 done | grid fits ≤7 unattended nights | §3a below; 69 tests |
| M2 — Phase 2: pooling theory (E1–E5) | not started | E1 agrees to ~1e-12 | — |
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

**Active gate:** M1 — the grid must fit ≤7 unattended nights (`experiments/estimate_cost.py`).

| # | Blocker | Severity | Owner milestone | Status |
|---|---|---|---|---|
| B1 | `pool_measurement` does not implement the theorem's channel — E1 **cannot pass** as written (see §7) | **High** | M2.1 | Open, documented |
| B2 | Batched path must be proven equivalent before any grid runs | High | M1.1 | **Closed 2026-07-25** (§3a) |

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
| 1.3 Safe parallelism and resume | not started | — |
| 1.4 Cost estimator and grid approval | not started | — |
| 1.5 Conditional accelerators | not started | gated on 1.4 |
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
| Projected grid wall-clock | not yet estimated (M1.4) |
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
| T5 | Pooling arms + SU(4) ceiling + measurement tie | M2 | not started |
| T6 | DLA, effective dim, expressibility, gen. bound | M3 | not started |
| F-A | E1 exact tie + E2 dephasing | M2.2–2.3 | **blocked by B1** |
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

**Resolution:** M2.1 rewrites `pool_measurement` to the exact channel above, *then* E1 runs.
Secondary risk to check at that point: `qml.measure` / `qml.cond` behaviour on `default.mixed`
under PennyLane 0.38 (flagged in the `UPGRADE_PLAN.md` risk register).

Note the same argument proves `RY(0.02)` inert: it acts only on `b`, and partial trace over `b`
is invariant under a unitary on `b` alone. That is why M0 measured 8.33e-16 for its removal.

## 8. Architecture sign-off decisions (roadmap §18)

| # | Decision | State | Outcome |
|---|---|---|---|
| 1 | Inert `RY(0.02)` removal | **Closed 2026-07-25** | **Retained.** Proven inert (9.44e-16 angle change, 8.33e-16 removal). Removal-with-proof branch was available and **declined** in favour of zero architecture change; documented as a no-op. See `docs/paper_code_reconciliation.md` row 4. |
| 2 | `pool_coherent` implementation | Open | Due Month 1, before M2.1. Default: do not run. |
| 3 | Multi-class head | Open | Due Month 2, before M4.5. Default: off; binary breadth instead. |
| 4 | JAX / GPU path | Open | Due at the M1.4 cost gate. Default: off unless ≤7 nights is missed. |
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
