# Track B handoff — theory and hardware (M3 + E5 + M8 → M7)

Written 2026-07-27 when the program was split into two parallel windows. Read this
plus `STATUS.md` before doing anything. `STATUS.md` is the authoritative dashboard;
this file defines **what this window owns and what it must not touch**.

**Paired window:** Track A (`2026-07-27-track-a-grid-handoff.md`) — datasets, baselines,
ablation grid. Do not do Track A's work here even if it looks quick.

---

## 1. Scope

You own model analysis, the hardware environment, and the resource/noise chain:

| Milestone | Work | Focused days |
|---|---|---|
| **M3** | Remaining Phase 3: 3.1 DLA, 3.3 effective dimension, 3.5 Caro, 3.6 inductive bias, 3.7 simulability | 4–6 |
| **E5** | Last open M2 item — blocked on Qiskit ≥1.0 in an isolated venv | 0.5–1 |
| **M8** | Resource table (transpile, prep vs model split), scaling sweep n∈{4…14}, join with DLA/variance | 3–4 |
| **M7** | Full-circuit noise, fake-backend provenance, QPU rehearsal + real run, fidelity threshold | 4–6 |

Ordering fixed by month plan §3: **M8.1 transpilation feeds M7** — build the resource
table before the noise ladder. M3.1's DLA feeds M8.3's scaling join, which is why these
sit in one window.

**You do not own:** M4, M5, M6 — those are Track A. M10/M11 are single-window and strictly
last; neither track starts them.

---

## 2. Start here — two near-free items

Both close a definition-of-done checkbox in minutes and need nothing from Track A.

### 2.1 M3.5 — Caro generalization bound — **DONE 2026-07-27**

Run via `python -m experiments.model_analysis --experiment generalization_bound`.
Evidence: `Results/evidence/t6_generalization_bound.json`. Full write-up in `STATUS.md` §12.

**Read that write-up before quoting the number.** The plan for this item — "plug T=74 and
T=269, report ~0.20 vs ~0.45 as the parameter-frugality argument" — was wrong. Caro's T
counts **trainable gates**, not parameters. Only 78 of the 269 allocated slots reach the
tape, and 48 of those drive multiple gates, so the honest T is **222** (0.3973) and the
effective-vs-allocated gap collapses to 218 vs 222. The frugality claim survives as a
parameter-count statement but must not be hung on this bound.

### 2.2 M3.7 — simulability statement

Pure prose, trivial, not written. No compute, no primitive.

---

## 3. Then M3.1 — the DLA, which is the interesting one

3.2 and 3.4 left a genuine tension recorded in `STATUS.md` §12: the ansatz is
**Haar-indistinguishable at n=10** (KL 1e-6 against a Haar reference that itself scores
1e-6), yet **gradients survive to n=14** (variance 1.34e-2 → 1.04e-3 over n=4…14). Holmes
et al. tie high expressibility to barren plateaus, so a plateau is exactly what was
expected and did not appear.

Three candidate reconciliations are on record, deliberately unadopted:

- **(a)** only 74 of 269 slots are live, so the *effective* ansatz is far smaller than the
  nominal one and full-parameterisation expressibility does not govern its gradients;
- **(b)** n ≤ 14 is short of asymptotic;
- **(c)** the expressibility measure saturates at n=10 — Haar already holds 0.9999989 of
  its mass in the first of 75 bins.

**The DLA is the item that discriminates.** Exact closure; fall back to smaller n if it
does not close. Do not let the manuscript silently pick one of (a)/(b)/(c) — that choice
must be earned by 3.1.

Two guardrails from 3.2 that still bind: the **median-over-live-slots reduction is
confounded** and must never be quoted (live-slot count is non-monotone: 62, 62, 122, 74,
128, 74); the designated conv0 slot is the controlled series. And 3.2's exponential-vs-
power-law fit (R² 0.9229 vs 0.9152) is **noise on six points** — it does not license a
claim about barren plateaus either way, only "none observed to n=14".

---

## 4. M3.3 and M3.6 are gated on Track A

Both need matched MLP/CNN controls from `baselines/classical_cnn.py`
(`MLPClassifier` at line 69, Keras `Sequential` at line 107).

Those baselines currently **select on their test argument** — the deferred M5.1 defect
marked at `experiments/run_experiments.py:232`. Track A is fixing it as their first task.

**Do not measure 3.3 or 3.6 until Track A posts that fix in `STATUS.md`.** If you do, your
effective-dimension and inductive-bias numbers describe models trained under a protocol
that no longer exists, and they will need re-running. Do §2 and §3 while you wait — that
is roughly two to three days of work with no dependency.

When unblocked: 3.3 needs the empirical Fisher plus the matched controls
(`effective_dimension` primitive already exists at `QCNN/utils/capacity.py:142`, 5 tests).
3.6 shifts test digits 1–2 px and measures output drift for FQCNN vs CNN vs MLP, plus
entanglement-entropy-vs-cut.

---

## 5. E5, then M8 → M7

**E5** is the last open M2 item. `qiskit_ibm_runtime` fails to import: installed Qiskit is
**0.45.3**, needs **≥1.0**. The risk register forbids upgrading the training environment
mid-study, so this needs an **isolated venv** — see the environment clause in §6. Unblock
it early even though it is off the critical path: M7 needs the same upgraded Qiskit, and
if E5 slides it collides with Phase 7 in Month 3.

**M8.1** transpiles to `{RZ, SX, X, ECR}` with a state-prep vs model split, and **feeds
M7** — do it first. **M8.2** is the scaling sweep n∈{4,6,8,10,12,14} (queue B nights).
**M8.3** joins scaling with the DLA and variance evidence.

**M7** carries a known debt: `noise_sim.py` **replays the historical 70/30 split** and
prints `[PROTOCOL] ... this run is not clean-protocol evidence`. That must be moved to the
clean 60/15/25 manifest before any noise number counts as evidence. Note a depolarizing
sweep was interrupted by a machine restart on 2026-07-27 at ~03:08 (appending to
`Results/headline_retrain_log.txt`); it produced only `p=0.00 acc=0.9950` on 200 samples
and wrote no evidence JSON, so there is nothing partial to clean up.

**Do not skip the QPU rehearsal.** M7.3a rehearses the exact job on a fake backend before
M7.3b spends the monthly allocation. Real-QPU submission is an explicit decision gate
(roadmap §18.6) and needs sign-off on the rehearsed, pre-sized job.

**Exit gates:** one real-QPU point exists; the F-C ladder (ideal → fake → real) is
complete; T2 matches transpiled circuits including state prep.

---

## 6. Shared-resource contract

> **This section is identical in both track handoffs. Change it in both files or neither.**

**Overnight compute is one serialized resource.** Both tracks draw on the same ≤7-night
budget on one machine. Track A owns **queue A** (datasets, baselines, ablations); Track B
owns **queue B** (scaling, variance). **Never run both concurrently.** `estimate_cost.py`
already under-predicts wall-clock because it ignores memory-bandwidth contention — two
concurrent grids make both slower *and* invalidate the cost calibration that passed the
M1.4 gate. Post the launch in `STATUS.md` before starting a night; check the other track
has not posted one.

**The training environment is frozen.** `requirements-lock.txt` is the study environment
and the risk register forbids upgrading it mid-study. Track B creates a **separate** venv
for Qiskit ≥1.0 (E5/M7) and **never upgrades the base env in place** — Track A is running
grid nights out of it, and an accidental in-place upgrade poisons headline reproducibility
silently. You would find out at M11. Record the new venv's lock file as its own artifact.

**Branches.** Work on `track-b/*` off `plan/fqcnn-q1-upgrade`; merge at gates, not
continuously. The parent branch is currently **32 commits ahead of origin** — push it.

**`STATUS.md` section ownership.** Track B writes the M3/M7/M8/E5 sections; Track A writes
M4/M5/M6. In the §1 milestone table, **update only your own rows** — never rewrite the
table wholesale, or the merge will silently drop the other track's state.

**`fqcnn.tex` is off-limits to both tracks.** Roadmap §17: no result claims before the
artifacts exist. Evidence goes to `Results/evidence/*.json`; the manuscript is M10.

---

## 7. Conventions that are easy to lose

- **Analysis primitives get their own module and tests before being run.** This caught a
  real defect in E4 (`von_neumann_entropy` returning −8e-16 for a pure state) and a wrong
  Crossref match in M9. Phase 3's runners follow `experiments/pooling_analysis.py`: a
  `run_*` per item, an `--experiment` CLI, JSON into `Results/evidence/`. Extend the
  `choices=[...]` list at `experiments/model_analysis.py:400` as you add items.
- **Controls are what make a measurement readable.** 3.4's KL of 1e-6 means nothing
  without the Haar-reference, 10×-inflation and degenerate controls that established the
  measure still has power. Carry that habit into 3.1 and 3.3.
- **Never use heredocs in this shell.** `python - <<'EOF'` silently strips a level of
  backslash escaping; it corrupted `fqcnn.tex` once and bit three times. Write scripts to
  a file with the Write tool instead.
- **Treat `estimate_cost.py` projections as lower bounds.**
- Suite is **204 tests passing** as of 2026-07-27. Keep it green; it is the gate.

## 8. Decisions already taken — do not re-litigate

1. `RY(0.02)` retained, proven inert.
2. `pool_coherent` approved as an ablation arm; ran in E3, +0.19pp, not significant.
3. JAX/GPU off — M1.4 projects 2.2 h against a 70 h budget.
4. Validation subsetting declined.
5. Still open and **not yours to decide alone**: venue (default IEEE TQE, decided end of
   Month 3), real-QPU job approval, multi-class head (default off).
