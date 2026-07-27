# Track A handoff — the grid (M4 → M5 → M6)

Written 2026-07-27 when the program was split into two parallel windows. Read this
plus `STATUS.md` before doing anything. `STATUS.md` is the authoritative dashboard;
this file defines **what this window owns and what it must not touch**.

**Paired window:** Track B (`2026-07-27-track-b-theory-hardware-handoff.md`) — model
analysis, hardware, resources. Do not do Track B's work here even if it looks quick.

---

## 1. Scope

You own the empirical grid chain. Three milestones, strictly ordered:

| Milestone | Work | Focused days |
|---|---|---|
| **M4** | Harder datasets: Fashion-MNIST, MedMNIST, KMNIST adapters; T1 from executed manifests | 3–4 |
| **M5** | Classical + quantum baselines, bootstrap CIs, paired tests, calibration + learning curves | 4–5 |
| **M6** | Ablation grid, one factor at a time | 2 |

The ordering is fixed by month plan §3 and is **not negotiable by rescheduling**:
M4 before M5 and M6 (datasets are the axis those grids run over); M5.3's paired-test
machinery before M6 (the ablation table needs it).

**You do not own:** M3, E5, M7, M8 — those are Track B. M10/M11 are single-window and
strictly last; neither track starts them.

---

## 2. Do these two things first, before anything else

Both are cheap code fixes that get expensive if deferred. The first one **also unblocks
Track B**, so it is your highest priority in the whole track.

### 2.1 M5.1 — baselines must stop selecting on their test argument

`experiments/run_experiments.py:232` carries the marker:

```
# NOTE (Phase 5 / M5.1): these baselines still select on the data passed
# as their test set. They must be given X_val before any baseline number
# enters the manuscript.
```

The call two lines down passes `X_test, y_test` into `run_classical_baselines`. It must
receive `X_val, y_val` for model selection, with the test set touched exactly once at the
end. Same audit applies to `run_quantum_baselines` immediately below it.

**Why this is the first thing you do:** Track B's M3.3 (effective dimension) and M3.6
(inductive bias) both need matched MLP/CNN controls, which come from
`baselines/classical_cnn.py`. If Track B measures against the current defective protocol
and you fix it afterwards, their numbers describe superseded models and must be re-run.
**Announce in `STATUS.md` the moment this lands** — Track B is waiting on it.

### 2.2 B3 — ablation configs run n=8 while the headline is n=10

`ABLATION_CONFIGS` at `experiments/run_experiments.py:75` sets `image_size=16` (→ n=8)
for every entry. The frozen headline is `image_size=28` (→ n=10). As it stands T4 would
describe a different model than the paper is about. The comment at line 87 already records
this; E3 fixed it **for the pooling arms only**, via the `e3_pool_*` configs at line 95.

The config change is trivial. Re-running is not — which is exactly why it must happen
**before** you spend a single night on the grid. `enc_feature_map` at line 83 is the one
legitimate exception (`image_size=4`, feature_map needs few qubits) — leave it and
document why in the T4 caption.

---

## 3. Then the chain

**M4 — datasets.** `datasets/` currently holds MNIST only. Add adapters through
`QCNN/utils/dataset_loader.py`, following the existing IDX path. Each new dataset needs a
clean 60/15/25 split manifest into `Results/manifests/`, same service as
`idx_0v1_n12665_seed42.json` (split id `adbb14862dfec2f7`, 7,599 / 1,900 / 3,166) — the
disjointness check is not optional. T1 comes from executed manifests, never from a
config file.

**M5 — baselines and statistics.** After 2.1, the remaining pieces are the quantum
controls (cong/hur/ttn, random-frozen, encoding-only), then M5.3's bootstrap CIs,
McNemar, Wilcoxon and Holm–Bonferroni, then calibration and learning curves. **M5.3's
statistics go in their own module with their own tests before any of them is run against
real results** — see §5.

**M6 — ablation grid.** One factor at a time, on the B3-corrected configs, using M5.3's
paired-test machinery for the table.

**Exit gates (month plan, Month 2):** T1, T3, T4, T5, T6 populated from executed runs;
no single-run number remains anywhere; failure manifest empty; every novelty claim maps to
an ablation row or is marked for deletion.

**Cut order if you overrun** (mandated, do not improvise): seeds, then datasets, then
non-pooling ablations. Never cut E1/E2 — they are already done and must not be disturbed.

---

## 4. Shared-resource contract

> **This section is identical in both track handoffs. Change it in both files or neither.**

**Overnight compute is one serialized resource.** Both tracks draw on the same ≤7-night
budget on one machine. Track A owns **queue A** (datasets, baselines, ablations); Track B
owns **queue B** (scaling, variance). **Never run both concurrently.** `estimate_cost.py`
already under-predicts wall-clock because it ignores memory-bandwidth contention — two
concurrent grids make both slower *and* invalidate the cost calibration that passed the
M1.4 gate. Post the launch in `STATUS.md` before starting a night; check the other track
has not posted one.

**The training environment is frozen.** `requirements-lock.txt` is the study environment
and the risk register forbids upgrading it mid-study. Track B will create a **separate**
venv for Qiskit ≥1.0 (E5/M7). Neither track ever upgrades the base env in place. If you
see Qiskit change under you, stop and reconcile before running anything.

**Branches.** Work on `track-a/*` off `plan/fqcnn-q1-upgrade`; merge at gates, not
continuously. The parent branch is currently **32 commits ahead of origin** — push it.

**`STATUS.md` section ownership.** Track A writes the M4/M5/M6 sections; Track B writes
M3/M7/M8/E5. In the §1 milestone table, **update only your own rows** — never rewrite the
table wholesale, or the merge will silently drop the other track's state.

**`fqcnn.tex` is off-limits to both tracks.** Roadmap §17: no result claims before the
artifacts exist. Evidence goes to `Results/evidence/*.json`; the manuscript is M10.

---

## 5. Conventions that are easy to lose

- **Analysis primitives get their own module and tests before being run.** This caught a
  real defect in E4 (`von_neumann_entropy` returning −8e-16 for a pure state) and a wrong
  Crossref match in M9. It applies directly to M5.3's statistics.
- **Never use heredocs in this shell.** `python - <<'EOF'` silently strips a level of
  backslash escaping; it corrupted `fqcnn.tex` once and bit three times. Write scripts to
  a file with the Write tool instead.
- **Treat `estimate_cost.py` projections as lower bounds.** The E3 grid took noticeably
  longer than its 0.5 h projection.
- Suite is **204 tests passing** as of 2026-07-27. Keep it green; it is the gate.

## 6. Decisions already taken — do not re-litigate

1. `RY(0.02)` retained, proven inert.
2. `pool_coherent` approved as an ablation arm; ran in E3, +0.19pp, not significant.
   Never headline without a second sign-off.
3. JAX/GPU off — M1.4 projects 2.2 h against a 70 h budget.
4. Validation subsetting declined — 4% epoch saving for noisier model selection.
5. Multi-class head stays off by default; only revisit if M4 order reaches 4.5.
