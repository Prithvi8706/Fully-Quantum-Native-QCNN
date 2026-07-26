# Session handoff — 2026-07-26

Written when the working session was cleared. Read this plus `STATUS.md` before
resuming. `STATUS.md` is the authoritative dashboard; this file only covers what
is *in flight* and what to do next.

---

## 1. There is a job running RIGHT NOW

**The clean headline retrain (final M1 item).** Launched 2026-07-26 13:48 as a
detached `nohup` process, so it survives a cleared session.

```
python main.py --dataset idx --path datasets/MNIST --encoding amplitude \
  --image-size 28 --classes 0 1 --learning-rate 0.005 --epochs 50 \
  --batch-size 32 --seed 42 \
  --log-file Results/headline_retrain_log.txt \
  --summary-log Results/headline_retrain_summary.txt \
  > Results/headline_retrain_stdout.txt 2>&1 &
```

Config matches the archived headline exactly (`Results/headline_manifest.json`).
Split is the clean 60/15/25 manifest `idx_0v1_n12665_seed42.json`:
**7,599 train / 1,900 val / 3,166 test**, disjoint verified, split id `adbb14862dfec2f7`.

### Check on it

```bash
tail -5 Results/headline_retrain_summary.txt        # per-epoch trajectory
grep -E "Final Quantum Accuracy|Final Best Validation" Results/headline_retrain_stdout.txt
```

`stdout` is block-buffered and stays empty until the process exits — that is
normal, **not** a crash. Use the summary file for progress; use PowerShell
`Get-Process python` to confirm the process is alive.

Trajectory when the session ended: epoch 4/50, **val 98.2%** (95.6 → 97.5 → 98.2 → 98.2).
~170 s/epoch. Early stopping is patience-3 with an LR-plateau step, so it will
likely stop well before epoch 50.

### When it finishes — this is the whole point of the run

The final test accuracy **replaces 98.86% everywhere**. To collect it:

1. Read `Final Quantum Accuracy` from `Results/headline_retrain_stdout.txt`.
2. Record it in `STATUS.md` §3a (M1 row) and close the "Clean headline retrain" line.
3. Update `docs/paper_code_reconciliation.md` **row 10**, which is the open
   disposition for this number.
4. Replace the three `approx 98\%` claims in `fqcnn.tex` — abstract,
   introduction, conclusion. They are the last place the dead number survives.
5. Commit the split manifest `Results/manifests/idx_0v1_n12665_seed42.json` as
   evidence, and the summary log. `Results/headline_retrain_stdout.txt` is
   chatty; consider gitignoring it.

**Expect it lower than 98.86% and report it anyway (§A6).** Two headwinds, not
one: the leak is gone *and* the clean protocol trains on 14% less data
(7,599 vs 8,865) because validation had to come from somewhere. Those two
effects cannot be separated. Val is tracking ~98.2%, so the drop may be small.

Weights land in `Results/Weights/run_seed42.npz` (gitignored). The archived
headline weights are read-only and untouched.

---

## 2. Where Phase 3 got to

Phase 3 was *started*, not finished. What is committed and green:

| Piece | Location | State |
|---|---|---|
| Meyer-Wallach Q (3.4) | `QCNN/utils/state_metrics.py` | done, 4 tests |
| Haar fidelity law + expressibility KL (3.4) | `QCNN/utils/capacity.py` | done, 8 tests |
| Caro generalization bound (3.5) | `QCNN/utils/capacity.py` | done, 5 tests |
| Effective dimension (3.3) | `QCNN/utils/capacity.py` | done, 5 tests |

These are **primitives only** — validated against known cases, but **not yet run
against the circuit**. Nothing in `Results/evidence/` yet for Phase 3.

### Measured feasibility (probe run 2026-07-26)

Gradient cost per initialisation, batch of 4, backprop:

| image size | n | slots | grad |
|---|---|---|---|
| 4 | 4 | 188 | 0.053 s |
| 8 | 6 | 194 | 0.137 s |
| 16 | 8 | 260 | 0.204 s |
| 28 | 10 | 269 | 0.466 s |
| 64 | 12 | 278 | 1.362 s |
| 128 | 14 | — | ~4 s (extrapolated) |

So **3.2's full 200-init x 6-qubit-count sweep is ~21 minutes serial** and
parallelises. It is affordable; just run it.

Those image sizes are how you get the scaling family: `from_image_size(s, 'amplitude')`
gives n = 4, 6, 8, 10, 12, 14 for s = 4, 8, 16, 28, 64, 128. Random inputs are
fine for gradient variance — no dataset needed.

### What Phase 3 still needs

- **3.1 DLA** — Lie closure of the frozen ansatz generators. Roadmap allows an
  exact-small-n fallback with a documented limitation if n=10 stalls. Not attempted.
- **3.2 gradient variance** — the sweep above. Feeds **F-E**.
- **3.3 effective dimension** — primitive exists; still needs the empirical
  Fisher computed for FQCNN *and* matched MLP/CNN controls.
- **3.4 expressibility + Q** — primitives exist; still needs sampling of the
  circuit's state ensemble. Use the `terminal` hook with `qml.state()`, fixing
  the input and varying parameters (the ansatz's expressibility, not the data's).
- **3.5 Caro bound** — primitive exists. Plug in T=74 effective and N from the
  retrain. Report alongside T=269 allocated: ~0.20 vs ~0.45, which is the
  parameter-frugality argument in one line.
- **3.6 inductive bias** — not started. Shift test digits 1–2 px, measure output
  drift for FQCNN vs CNN vs MLP; plus entanglement-entropy-vs-cut.
- **3.7 simulability statement** — pure prose, trivial, not written.

Suggested home for the runner: `experiments/model_analysis.py` (roadmap §3),
following the pattern of `experiments/pooling_analysis.py` — a `run_*` function
per item, a `--experiment` CLI, JSON into `Results/evidence/`.

---

## 3. Useful context that is easy to lose

- **`analysis` primitives get their own module and tests before being run.**
  That convention caught a real defect in E4 (`von_neumann_entropy` returning
  −8e-16 for a pure state) and a wrong Crossref match in M9. Keep it.
- **Heredocs mangle backslashes in this shell.** `python - <<'EOF'` silently
  strips one level, so `"\\bibitem"` became a literal backspace and corrupted
  `fqcnn.tex` once. Write scripts to a file with the Write tool instead. This
  bit three times.
- **The estimator under-predicts wall-clock.** `estimate_cost.py` calibrates
  single-process speed and divides by worker count, ignoring memory-bandwidth
  contention. The E3 grid took noticeably longer than its 0.5 h projection.
  Treat projections as lower bounds.
- **Fuzzy bibliographic matching is not verification.** A Crossref query for
  ref44 confidently returned a different paper.

---

## 4. Open blockers

| # | Blocker | Owner | Note |
|---|---|---|---|
| **B3** | Ablation configs run n=8 while the headline is n=10, so T4 would describe a different model | M5/M6 | Config fix is trivial; re-running is not. E3 fixed this for the pooling arms only, via the `e3_pool_*` configs at `image_size=28`. |
| **E5 / Phase 7** | `qiskit_ibm_runtime` fails to import — needs Qiskit ≥1.0, installed is 0.45.3 | M7 | Risk register forbids upgrading the training env mid-study. Needs an isolated venv. |

## 5. Decisions already taken (do not re-litigate)

1. `RY(0.02)` **retained**, proven inert.
2. `pool_coherent` **approved** as an ablation arm; ran in E3, gains +0.19pp, not
   significant. Never headline without a second sign-off.
3. JAX/GPU **off** — M1.4 projects 2.2 h against a 70 h budget.
4. Uncited references **deleted** (12 of them), which disposed of ref1 and ref27
   without guesswork.
5. Validation subsetting **declined** — 4% epoch saving for noisier model selection.

Still open: multi-class head (default off), venue (default IEEE TQE), real-QPU
job approval.
