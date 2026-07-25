# FQCNN Q1 Upgrade — Month-Wise Plan for Remaining Work

**Written:** 2026-07-25, immediately after the M0 (Phase 0) gate passed.
**Scope:** M1–M11 only. M0 is closed and is not re-planned here.
**Derived from:** `UPGRADE_PLAN.md` (governing spec) and
`docs/superpowers/plans/2026-07-23-fqcnn-q1-upgrade-roadmap.md` §§4–19.

This document exists because neither governing doc carries a month-level view. The
roadmap has an 8-week table (§4) and `UPGRADE_PLAN.md` has a 7-week "suggested order of
attack"; both are written from a cold start where week 1 is Phase 0, so both are stale
by one milestone as of today. Nothing here changes scope, ordering constraints, or any
gate — it only redistributes the *remaining* work onto calendar months.

---

## 1. Assumptions (change these and the month boundaries move)

| Assumption | Value | Effect if wrong |
|---|---|---|
| Working pace | **~4 focused days/week (~17 focused days/month)** | See the compression table in §6 |
| Calendar anchor | Month 1 starts the week of **2026-07-28** | Shifts every month boundary |
| Overnight compute | One machine, unattended nights available | The Month 2 grid is the only place this binds |
| QPU access | IBM Open Plan, ~10 min/month | Month 3 gets one job, not a benchmark campaign |
| No hard external deadline | — | A fixed submission date forces cuts per §7 |

Effort figures below are the governing docs' own estimates, not new ones.

---

## 2. Remaining effort ledger

| Milestone | Phase | Focused days | Extra compute | Blocking? |
|---|---|---|---|---|
| M1 | 1 — affordable execution | 2–3 | 1 night (headline retrain) | **Yes**, for the grid |
| M2 | 2 — pooling theory, E1–E5 | 4–6 | queued (E3) | **Yes**, for the story |
| M3 | 3 — model analysis | 4–6 | yes (grad variance, DLA) | No |
| M4 | 4 — harder datasets | 3–4 | yes | No |
| M5 | 5 — baselines + statistics | 4–5 | yes | No |
| M6 | 6 — ablation grid | 2 | yes | No |
| M7 | 7 — noise + real QPU | 4–6 | + QPU scheduling latency | No |
| M8 | 8 — resources + scaling | 3–4 | yes | No |
| M9 | 9 — references | 1–2 | none | No |
| M10 | 10 — manuscript + venue | 5–7 | none | Last |
| M11 | final reproduction + red team | 2–4 | re-runs only | Last |
| | **Total** | **34–49** | | |

At ~17 focused days/month that is **2–3 months of pure effort**. The plan below spends
**4 months**, because QPU scheduling latency (M7) and the post-red-team correction loop
(M11) are calendar risks that cannot be compressed by working harder.

---

## 3. Dependency constraints that fix the ordering

From roadmap §2.1 and §17 — these are not negotiable by rescheduling:

```text
M1 ──► M2 (E1/E2 gate) ──┬──► M3, M4, M5, M6, M7, M8   (overlap permitted)
                         └──► M9 (independent, writing-only)
                                        └──► M10 ──► M11
```

- **M1 before everything** — no grid may run before the ≤7-night cost gate approves it.
- **M2's E1/E2 before the broad grid** — if E1 misses 1e-12 the theorem mapping is wrong
  and Phase 2 stops (roadmap §19). Running Phases 4–6 first would burn nights on a story
  that may need rebuilding.
- **M8.1 transpilation feeds M7** — do the resource table before the noise ladder.
- **M4 before M5 and M6** — datasets are the axis those grids run over.
- **M5.3 statistics before M6** — the ablation table needs the paired-test machinery.
- **M10 after all evidence** — roadmap §17: do not write result claims before artifacts
  exist.

---

## 4. Month-wise distribution

### Month 1 — 2026-07-28 → 2026-08-28 · *Critical path: make it affordable, then make the theory true*

**Focused days: 13–17. Theme: close both blocking milestones.**

| Work | Milestone | Days |
|---|---|---|
| Batched backprop, sequential path as oracle | M1.1 | 1–1.5 |
| Amplitude/validation caching | M1.2 | 0.5 |
| `--jobs N`, resume, failure manifest | M1.3 | 0.5–1 |
| `estimate_cost.py` + grid approval | M1.4 | 0.5 |
| **Clean headline retrain** (replaces 98.86% everywhere) | M1 debt | 1 night |
| Pooling arms: `pool_measurement` fix, `pool_none`, `pool_su4` | M2.1 | 1–2 |
| **E1 exact equivalence** (run the moment M1 lands — nearly free) | M2.2 | 0.5 |
| **E2 fixed-parameter dephasing** | M2.3 | 0.5–1 |
| E4 information dynamics | M2.5 | 1–1.5 |
| Theorem 1 / Props 2–3 write-up (no compute dependency) | M2.7 | 1–2 |
| Resource table: transpile to `{RZ,SX,X,ECR}`, prep vs model split | M8.1 | 2–3 |
| Dataset adapters: Fashion-MNIST, MedMNIST, KMNIST | M4 setup | 1–2 |

**Nights used:** ~1 (headline retrain). E1/E2 are daytime work.

**Decisions needed this month:**
- **`pool_coherent` sign-off** (§18.2) — before M2.1 implementation. Default: not run.
- **JAX/GPU activation** (§18.4) — only if M1.4 projects past 7 nights.

**Exit gates:** batched == sequential on outputs, loss, *every* gradient, one optimizer
step; grid fits ≤7 unattended nights; **E1 agrees to ~1e-12**; clean headline number
exists and is recorded.

**Deferred M0 debt cleared:** clean-protocol headline retrain.

**Risk:** E1 failing is the single worst outcome in the program. It stops Phase 2 and
means the theorem-to-code mapping is wrong, not that the experiment is noisy. Budget the
whole of Month 1's slack against it.

---

### Month 2 — 2026-08-31 → 2026-09-30 · *The grid month*

**Focused days: 14–17. Theme: daytime analysis, nighttime grid. This is the compute crunch.**

| Work | Milestone | Days |
|---|---|---|
| Finish dataset integration; T1 from executed manifests | M4 | 1–2 |
| Classical arms; **remove test-as-validation from every classical path** | M5.1 | 1.5–2 |
| Quantum controls: cong/hur/ttn, random-frozen, encoding-only | M5.2 | 1 |
| Bootstrap CIs, McNemar, Wilcoxon, Holm–Bonferroni | M5.3 | 1–1.5 |
| Calibration + learning curves | M5.4 | 1 |
| Ablation grid, one factor at a time | M6 | 2 |
| DLA closure (exact; fallback to smaller n) | M3.1 | 1.5–2 |
| Gradient variance, ~200 inits × n∈{4..14} | M3.2 | 0.5 + compute |
| Effective dimension, expressibility, Meyer–Wallach | M3.3–3.4 | 1.5 |
| Generalization bound; inductive-bias translation test; simulability statement | M3.5–3.7 | 1.5–2 |

**Nights used:** the full ≤7-night budget. Queued: **E3 retrained head-to-head** (5 arms
× ≥5 seeds × ≥3 datasets), the dataset grid, baselines, ablations, learning curves, and
the gradient-variance sweep. Roadmap §17 concurrency applies — daytime CPU analysis,
overnight queue A (datasets/baselines/ablations), overnight queue B (scaling/variance).

**Decisions needed:** **multi-class head** (§18.3) — only if M4 order reaches 4.5.
Default: stays off, binary breadth answers the reviewer question.

**Exit gates:** T3, T4, T5, T6 populated from executed runs; no single-run number remains
anywhere; failure manifest empty; every novelty claim maps to an ablation row or is
marked for deletion.

**Deferred M0 debt cleared:** classical baselines selecting on their test argument.

**Risk:** this is the month that overruns. If the M1.4 estimate was optimistic, apply the
mandated cut order — seeds, then datasets, then non-pooling ablations. **Never cut E1/E2.**

---

### Month 3 — 2026-10-01 → 2026-10-30 · *Hardware, scale, and citations*

**Focused days: 8–13, deliberately light. Theme: absorb QPU scheduling latency.**

| Work | Milestone | Days |
|---|---|---|
| Full-circuit noise: decompose state prep, transpile prep+body, Aer | M7.1 | 2–2.5 |
| Fake-backend selection + provenance snapshot | M7.2 | 0.5–1 |
| **Rehearse** the exact QPU job on a fake backend | M7.3a | 1 |
| **Execute** the real-QPU run (6-qubit instance, 50–100 samples × 1024 shots) | M7.3b | 0.5 + window |
| Fidelity-threshold derivation | M7.4 | 1 |
| Scaling sweep n∈{4,6,8,10,12,14} | M8.2 | 1 + nights |
| Join scaling with DLA/variance evidence | M8.3 | 1 |
| References: authors, DOIs, published versions, citation audit | M9 | 1–2 |

**Nights used:** scaling sweep at n=12/14; noise ladder simulations.

**Decisions needed:**
- **Real-QPU submission** (§18.6) — approve the rehearsed, pre-sized job before spending
  the monthly allocation. Do not skip the rehearsal.
- **Venue selection** (§18.5) — decide at end of month, before any style-sensitive
  rewriting starts. Default: IEEE TQE.

**Exit gates:** one real-QPU point exists; F-C ladder (ideal → fake → real) complete; T2
matches transpiled circuits including state prep; no author-less bibliography entry.

**Deferred M0 debt cleared:** `noise_sim` replaying the historical 70/30 split.

**Risk:** QPU availability (roadmap §19). If no window opens, Phase 7 stays open and the
rest of the month's work still completes — this month is sized with that slack on purpose.

---

### Month 4 — 2026-11-02 → 2026-11-27 · *Manuscript, reproduction, red team*

**Focused days: 7–11 plus a correction loop. Theme: nothing new is measured.**

| Work | Milestone | Days |
|---|---|---|
| Apply journal template (venue chosen in Month 3) | M10.1 | 0.5 |
| Build T1–T6 and F-A–F-F into the manuscript from artifacts | M10.2 | 2–3 |
| Rewrite claims: delete information-loss language, install Theorem 1, register-space locality, allocated-vs-effective params, excise all 98% claims, add limitations | M10.3 | 2–3 |
| Submission package: source, figures, supplementary, lock, manifests, code archive | M10.4 | 0.5–1 |
| Clean-checkout reproduction, 7 steps end to end | M11 | 1–2 |
| Definition-of-done audit + reviewer-question red team | M11 | 1–2 |
| **Correction loop from red-team findings** | M11 | unbounded — reserve the last week |

**Nights used:** none required; re-runs only if the audit finds a gap.

**Exit gate (final):** clean reproduction, definition-of-done checklist, reviewer map, and
submission-package checks all pass. Zenodo DOI created at submission readiness, not before.

**Risk:** the red team finds a claim with no traceable artifact. Roadmap §19 is explicit —
regenerate or rewrite the claim; never hand-edit numbers into agreement. Reserve the final
week for exactly this.

---

## 5. Decision-gate calendar (roadmap §18)

| # | Decision | When | Default |
|---|---|---|---|
| 1 | Inert `RY(0.02)` removal | **Closed 2026-07-25** | Retained; proven inert |
| 2 | `pool_coherent` implementation | Month 1, before M2.1 | Do not run |
| 3 | Multi-class head | Month 2, before M4.5 | Off; binary breadth instead |
| 4 | JAX / GPU path | Month 1, at the M1.4 cost gate | Off unless budget is missed |
| 5 | Venue | End of Month 3 | IEEE TQE |
| 6 | Real-QPU job submission | Month 3, after rehearsal | Requires explicit approval |

---

## 6. Compression / expansion by pace

Content and ordering are pace-invariant; only the calendar moves.

| Pace | Focused days/month | Duration | Submission-ready |
|---|---|---|---|
| Full-time (5 d/wk) | ~21 | ~2.5 months | early Oct 2026 |
| **Assumed (4 d/wk)** | **~17** | **4 months** | **late Nov 2026** |
| Half-time (2–3 d/wk) | ~10 | ~6 months | Jan–Feb 2027 |

QPU scheduling latency and the M11 correction loop do not shrink with pace, so the
full-time row is ~2.5 months, not 2.

---

## 7. If a hard deadline forces cuts

Apply in this order, and only this order:

1. **Seeds** — 10 → 5 per cell (M5.3 floor is 5; do not go below).
2. **Datasets** — drop to the 3-domain minimum (M4 gate floor).
3. **Non-pooling ablations** — trim M6 axes, keeping every axis that a paper claim maps to.
4. **M3 depth** — DLA at smaller n with the documented fallback instead of exact n=10.
5. **M7.3 real QPU** — leave Phase 7 incomplete with the fake-backend ladder reported.

**Never cut:** E1, E2, the freeze guards, the clean split protocol, or the ≥5-seed floor
on any number that appears in the manuscript.

---

## 8. Housekeeping note

Roadmap §20 requires a maintained program dashboard. It does not currently exist as a
tracked file — `git ls-files "*.md"` returns only the four planning documents. Committing
it (milestone state, active gate, run-cell counts, compute consumed, QPU budget, T1–T6 and
F-A–F-F artifact status, definition-of-done checklist, sign-off decisions) is a Month 1
housekeeping item.
