# FQCNN Q1 Upgrade Roadmap

> **For future agentic workers:** Before implementing a milestone, create a milestone-specific plan and use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. This document is the program roadmap; it intentionally does not start implementation.

**Goal:** Complete every non-optional requirement in `UPGRADE_PLAN.md` and produce a reproducible, evidence-backed Q1-journal submission package without changing the frozen headline FQCNN architecture.

**Architecture:** Execute the program through evidence gates. First establish model identity, clean evaluation protocol, shared circuit construction, and affordable execution; then validate the pooling theory; then run the broader analysis, dataset, baseline, ablation, noise, scaling, reference, and manuscript workstreams. All manuscript claims are generated from traceable evidence artifacts.

**Tech Stack:** Python 3.9.13, PennyLane 0.38.0, NumPy 1.26.4, scikit-learn 1.6.1, pytest, Qiskit/Aer and `qiskit-ibm-runtime` for hardware phases, LaTeX/IEEE journal tooling, shell reproduction scripts.

## Global constraints

- `UPGRADE_PLAN.md` is the governing specification.
- The current headline circuit is immutable except for the strictly governed inert-gate exception in §A2.
- The main pipeline remains unitary until its terminal measurement.
- The paper moves to match executed code; the headline code does not move to match the paper.
- Variants are labelled experiment arms, never silent replacements.
- `[ARCH — opt-in]` items require explicit user sign-off at their execution gate.
- Validation controls checkpointing, learning-rate changes, early stopping, and tuning; test is evaluated once per run.
- Negative results are retained and reported.
- No phase is considered complete without its evidence, tests, provenance, and paper-facing artifact.
- No implementation begins as part of this roadmap-writing session.

---

## 1. Program outcome

The finished program must deliver:

1. A regression-guarded frozen FQCNN implementation.
2. A leakage-free, reproducible training and experiment protocol.
3. A single circuit source used by training, noise, hardware, resource, drawing, and analysis paths.
4. An affordable, resumable experiment harness.
5. Tested theory and experiments for unitary versus measurement-based pooling.
6. Model-analysis evidence covering trainability, capacity, expressibility, generalization, and inductive bias.
7. Results across difficult datasets and matched classical/quantum baselines.
8. Statistically rigorous ablations, calibration, and learning curves.
9. Realistic state-preparation-aware noise results and one bounded real-QPU demonstration.
10. Resource and scaling evidence.
11. A complete, validated bibliography.
12. A journal-formatted manuscript and clean-checkout reproduction package.

The program ends only after the definition-of-done checklist in `UPGRADE_PLAN.md` and the reviewer-question evidence map both pass a final red-team review.

## 2. Delivery strategy

### 2.1 Critical path

```text
Phase 0: trust and protocol
  → Phase 1: affordable execution
    → Phase 2: pooling theory and decisive experiments
      → Phases 3–8: evidence production
        → Phase 9: citation completion
          → Phase 10: manuscript and submission package
            → final reproduction and red-team gate
```

Phase 0, Phase 1, and the core of Phase 2 are sequential because every later result depends on their integrity. After Phase 2's E1/E2 gate, Phases 3–8 can overlap subject to compute and artifact dependencies.

### 2.2 Planning depth

This roadmap is intentionally program-level. Before implementing each milestone, produce a smaller milestone plan with exact file changes, test cases, commands, and review checkpoints. Suggested plans are:

1. `phase-0-freeze-and-protocol-plan.md`
2. `phase-1-throughput-and-runner-plan.md`
3. `phase-2-pooling-theory-plan.md`
4. `phase-3-model-analysis-plan.md`
5. `phases-4-to-6-experimental-grid-plan.md`
6. `phases-7-to-8-hardware-and-resources-plan.md`
7. `phases-9-to-10-paper-and-release-plan.md`

This decomposition reduces speculative detail without reducing the upgrade's scope.

## 3. Intended repository boundaries

These are the expected ownership boundaries to use when writing milestone plans. Exact names may be adjusted only if the existing code makes a smaller surgical change clearer.

### Existing files expected to change

- `QCNN/models/QCNNModel.py` — consume the shared frozen circuit and remove topology duplication from the model class.
- `QCNN/training/Qtrainer.py` — train/validation-only control flow, batched execution, isolated checkpoints, and final run artifacts.
- `QCNN/config/Qconfig.py` — explicit headline, smoke, ablation, and scaling configuration fields.
- `QCNN/layers/QPool.py` — tested experiment-arm pooling definitions while preserving the frozen unitary arm.
- `QCNN/encoding/QEncoder.py` — cached amplitude-input contract without changing headline semantics.
- `QCNN/utils/data_preprocessing.py` and `QCNN/utils/dataset_loader.py` — common dataset identity and split inputs.
- `QCNN/utils/metrics.py` — per-example persistence, confidence intervals, paired tests, calibration, and aggregation.
- `experiments/run_experiments.py` — registry-driven runs, resume, parallelism, and failure manifests.
- `experiments/hardware_run.py` — replace the current sketch with shared-circuit fake/real backend execution.
- `baselines/classical_cnn.py` — matched and practical CNN baselines without test-as-validation leakage.
- `baselines/quantum_baselines.py` — compatible registered quantum controls and split consumption.
- `noise_sim.py` — remove hand-copied topology and distinguish model-only from full transpiled noise.
- `main.py` — central split use and exactly one final test evaluation.
- `reproduce.sh` — run integrity gates first and regenerate all final evidence.
- `requirements-lock.txt`, `requirements.txt`, `setup_env.ps1`, `setup_env.bat` — truthful environment setup.
- `DATASET_README.md`, `README.md`, and `fqcnn.tex` — executed protocols, results, limitations, and journal presentation.

### Likely new focused modules

- `QCNN/circuits.py` — canonical frozen circuit builder and evaluation-hook boundary.
- `QCNN/utils/splits.py` — deterministic split manifests and validation.
- `QCNN/utils/run_artifacts.py` — per-run paths, state, schema, resume, and failure recording.
- `experiments/configs.py` — explicit headline, smoke, baseline, ablation, and scaling registry.
- `experiments/estimate_cost.py` — projected experiment wall-clock and budget gate.
- `experiments/statistics.py` — bootstrap, Wilcoxon, McNemar, and Holm–Bonferroni analysis.
- `experiments/pooling_analysis.py` — E1–E4 pooling probes and evidence output.
- `experiments/model_analysis.py` or a focused `analysis/` package — Phase 3 analyses.
- `experiments/resource_analysis.py` — logical/transpiled resource counts and scaling schedules.
- `experiments/datasets.py` — Fashion-MNIST, KMNIST, and MedMNIST adapters.
- `tests/` — architecture, protocol, execution, statistics, analysis, and reproduction gates.
- `Results/manifests/` — immutable dataset and split identities.
- `Results/evidence/` — generated T1–T6 and F-A–F-F source data.

## 4. Schedule at a glance

The base schedule follows the upgrade plan's six-to-eight focused weeks. Keep a one-to-two-week contingency for QPU availability, long grids, or research-analysis fallback work.

| Week | Primary objective | Main exit gate |
|---|---|---|
| **1** | Complete Phase 0 foundations and begin Phase 1 batching | Frozen identity, clean split protocol, and shared circuit are trustworthy |
| **2** | Complete Phase 1; clean headline retrain; run E1/E2; draft theorem section | Grid is affordable and pooling equivalence is validated |
| **3** | Run Phase 2 E3–E5; launch Phases 4–6 grid; begin Phase 3 | Pooling evidence is complete and broader experiment queue is healthy |
| **4** | Continue Phases 3–6; build fake-backend noise ladder and Phase 8 resources | Baseline/ablation evidence and fake-hardware path are validated |
| **5** | Execute rehearsed QPU job; finish scaling/model analysis; complete references | Phases 3–9 have validated evidence artifacts |
| **6** | Rewrite manuscript around generated evidence; choose/apply journal format | Complete evidence-backed draft |
| **7** | Reproduction package, internal review, claim audit, and corrections | Clean reproduction and reviewer map pass |
| **8 / contingency** | Resolve incomplete runs, QPU scheduling, or red-team findings | Every required definition-of-done item passes |

The schedule is a dependency guide, not permission to skip unfinished work at a week boundary.

---

## 5. Milestone M0 — Phase 0: trusted foundation

**Target duration:** 3–4 focused days
**Blocking:** Yes; no later implementation or experiment may bypass it.

### M0.1 Version and identify the governing artifacts

**Outcome:** The plan, paper source, headline metadata, archived weights, and fixed fixtures have reviewable identities.

- Keep `UPGRADE_PLAN.md` as the governing specification.
- Put `fqcnn.tex` and the evidence sources required for reconciliation under version control before paper editing.
- Record which archived metadata and weights define the headline model.
- Distinguish historical outputs from new clean-protocol outputs.

**Exit check:** A reviewer can identify the exact circuit configuration and archived parameter file protected by the freeze.

### M0.2 Install freeze and unitarity guards

**Outcome:** Accidental architecture or semantic drift becomes an immediate test failure.

- Add the operation/parameter-slot fingerprint.
- Add 20-input archived expectation fixtures.
- Add a main-path operation audit confirming no mid-circuit measurement, reset, channel, or classical feed-forward.
- Run the guards before every future experiment entry point.

**Exit check:** Refactoring the circuit source without preserving behavior fails before an experiment runs.

### M0.3 Commit the effective-parameter audit

**Outcome:** The paper and resource calculations use measured effective parameters.

- Audit all parameter slots over fixed random inputs.
- Persist both the effective count and named slot ranges.
- Resolve the preliminary 74-versus-approximately-76 difference through the committed test and documented tolerance.
- Identify allocated, syntactically used, gradient-effective, and readout-effective parameters where they differ.

**Exit check:** Later parameter-matched baselines, gradient costs, and bounds consume one authoritative audit artifact.

### M0.4 Establish the clean split protocol

**Outcome:** No test-set information reaches model selection.

- Centralize seeded stratified 60/15/25 splitting.
- Persist selected sample IDs and exact train/validation/test indices.
- Update all training, experiment, noise, baseline, custom-dataset, and hardware paths to consume manifests.
- Use validation for checkpoints, schedules, stopping, and tuning.
- Evaluate test once after the selected model is fixed.

**Exit check:** Automated tests prove split disjointness and exactly one final test evaluation.

### M0.5 Isolate run outputs and provenance

**Outcome:** Every result is traceable and parallel-safe.

- Give each run isolated weights, predictions, metrics, configuration, log, timing, environment, and status artifacts.
- Persist per-example IDs and predictions for paired statistics.
- Ensure failed or partial outputs cannot be mistaken for completed cells.
- Preserve archived headline weights as read-only input.

**Exit check:** Two simultaneous smoke runs cannot overwrite each other's outputs or the archived headline weights.

### M0.6 Consolidate the circuit source

**Outcome:** Training, noise, hardware, drawing, resource, and analysis paths use one topology.

- Extract the frozen circuit into a canonical builder.
- Define evaluation hooks that cannot alter the headline path by default.
- Remove the hand-copied topology from `noise_sim.py` and prevent new copies.
- Rebuild hardware and resource tooling around the shared builder.

**Exit check:** Freeze expectations agree before and after consolidation to the specified tolerance.

### M0.7 Complete paper↔code reconciliation

**Outcome:** Every equation and architecture statement has a recorded disposition.

The reconciliation must cover at least:

- amplitude encoding versus the paper's unexecuted two-stage map;
- one effective convolution stage at n=10;
- odd-wire retirement;
- canonical pooling controls and the inert `RY(0.02)` decision;
- classifier topology, modular indexing, and effective parameters;
- adjoint differentiation in simulation versus parameter-shift on hardware;
- dataset counts, class-label mapping, and split protocol;
- register-space rather than pixel-space locality;
- removal of unsupported information-preservation and calibration claims.

**Exit check:** Each mismatch is resolved through a paper edit, disclosure, tested arm, or explicitly governed inert-gate exception.

### M0.8 Repair reproducibility inputs

**Outcome:** A clean environment reproduces the actual software stack.

- Regenerate the environment lock from the real training environment.
- Record Python and simulator versions in run metadata and the paper.
- Centralize seed policy.
- Align setup scripts with the lock.
- Remove only the non-circuit dead artifacts permitted by `UPGRADE_PLAN.md`.
- Make `reproduce.sh` run integrity gates first.

**M0 gate:** All Phase 0 tests pass; only then may Phase 1 or any new result-generating work begin.

---

## 6. Milestone M1 — Phase 1: affordable and resumable execution

**Target duration:** 2–3 focused days
**Blocking:** Yes for the full experiment grid.

### M1.1 Batched backpropagation

- Use `default.qubit` with backpropagation for the primary batched training path in the pinned environment.
- Retain the sequential frozen path as the correctness oracle.
- Compare raw outputs, loss, every parameter gradient, and one optimizer update.
- Benchmark representative batch sizes on the 10-qubit headline configuration.

**Exit check:** Batched execution is behaviorally equivalent and materially faster; the measured gain replaces the unverified 10–50× estimate.

### M1.2 Input and validation caching

- Precompute padded, normalized amplitude inputs once per dataset/split.
- Preserve the fingerprint's state-preparation semantics.
- Cache fixed validation input subsets, not predictions.
- Record cache identity in run metadata.

**Exit check:** Cached and uncached outputs agree and repeated epochs avoid redundant input normalization.

### M1.3 Safe parallelism and resume

- Add process-level jobs over independent run cells.
- Set worker thread limits to avoid nested CPU oversubscription.
- Skip only complete, schema-valid, identity-matching run artifacts.
- Emit a failure manifest and nonzero exit when required cells fail.

**Exit check:** A repeated smoke command resumes without rerunning completed cells and yields the same summary.

### M1.4 Cost estimator and grid approval

- Time representative epochs and evaluations for each model family.
- Project wall-clock by dataset, arm, seed count, worker count, and validation frequency.
- Include classical baselines and analysis overhead where meaningful.
- Approve the grid only if it fits seven unattended nights.
- If it does not fit, use the mandated reduction order: seeds, then datasets, then non-pooling ablations. Never cut E1/E2.

### M1.5 Conditional accelerators

- Evaluate JAX/JIT/vectorization only if the verified default path misses the budget.
- Evaluate `lightning.gpu` only for larger scaling instances where benchmarking justifies it.
- Do not change the training environment mid-study unless a separate, locked analysis environment is created.

**M1 gate:** The clean headline retrain and full approved study are computationally feasible, deterministic, and resumable.

---

## 7. Milestone M2 — Phase 2: pooling theory and decisive experiments

**Target duration:** 4–6 focused days plus queued runs
**Blocking:** Yes for the paper's central story.

### M2.1 Finalize the theorem-facing implementations

- Preserve `pool_unitary` as the frozen headline block.
- Correct `pool_measurement` so it represents the theorem's exact measure-and-condition channel.
- Implement and label `pool_none` and `pool_su4`.
- Present an explicit sign-off decision before implementing/running `pool_coherent`.
- Add evaluation-only dephasing and state-observation hooks.

### M2.2 E1 — exact equivalence validation

- Compare fixed-parameter unitary and measurement implementations on `default.mixed`.
- Require agreement near 1e-12 before using E1 in the paper.
- Treat failure as an implementation/theorem-mapping defect, not an experimental result to average away.

**Artifact:** F-A exact-tie data and a test-backed equivalence statement.

### M2.3 E2 — fixed-parameter dephasing

- Dephase discarded wires immediately before pooling without retraining.
- Confirm zero operational effect for the frozen block.
- Measure nonzero headroom only for approved non-diagonal/SU(4) arms where applicable.

**Artifact:** F-A dephasing figure and Proposition 3 evidence.

### M2.4 E3 — retrained head-to-head

- Train all approved pooling arms from scratch.
- Use at least five seeds and at least three datasets.
- Match splits, optimization budgets, stopping rules, and reporting.
- Retain outcomes even if the frozen block is not the highest-accuracy arm.

**Artifact:** T5 pooling-arm table with uncertainty and paired statistics.

### M2.5 E4 — information dynamics

- Measure per-stage coherence, purity, entropy, and kept/discarded mutual information.
- Validate analysis functions on small known states.
- Connect mechanism plots to E1–E3 without overclaiming causality.

**Artifact:** F-B mechanism figure and machine-readable state-analysis data.

### M2.6 E5 — hardware practicality

- Transpile each pooling arm for a selected heavy-hex fake backend.
- Report two-qubit gates, depth, required dynamic-circuit capabilities, and estimated latency.
- Separate measurement/reset/feed-forward latency from unitary gate latency.

**Artifact:** Hardware-practicality section in T5 or T2 and supporting data.

### M2.7 Write the tested theory section

- State Theorem 1 using the implemented channel mapping.
- State strict containment through the SWAP witness.
- State and test the coherence-transfer functional.
- Delete the unsupported claim that the frozen block preserves information lost by measurement pooling.
- Install the defensible claims: exact simulation, no mid-circuit measurement/reset/feed-forward, pure global execution, and adjoint differentiability.

**M2 gate:** The theory, E1/E2 predictions, implementations, and artifacts agree. Only then is the central narrative considered stable.

---

## 8. Milestone M3 — Phase 3: model-analysis suite

**Target duration:** 4–6 focused days plus compute
**Dependencies:** M0 shared circuit/effective slots; M1 throughput; M8 scaling configurations.

### Work packages

1. **Dynamical Lie algebra:** attempt the exact closure for the frozen/scaling ansatz; if n=10 stalls, complete exact smaller-n results and document the fallback.
2. **Gradient variance:** run approximately 200 initializations for n={4,6,8,10,12,14} within the approved budget.
3. **Effective dimension:** compute empirical-Fisher-based effective dimension for FQCNN and matched MLP/CNN controls.
4. **Expressibility and entangling capability:** calculate Haar KL divergence and Meyer–Wallach Q for FQCNN and quantum baselines.
5. **Generalization bound:** derive the bound using the audited effective trainable-gate count and actual N; report it honestly if vacuous.
6. **Inductive bias:** compare translation sensitivity for FQCNN, CNN, and MLP and generate entanglement-entropy-versus-cut evidence.
7. **Classical simulability:** add the explicit scale and no-quantum-advantage statement.

**Artifacts:** T6, F-E, inductive-bias evidence, and manuscript theory/limitations text.

**M3 gate:** Every Phase 3 claim has a derivation, experiment, fallback disclosure, and reproducible artifact.

---

## 9. Milestone M4 — Phase 4: harder datasets

**Target duration:** 3–4 focused days plus compute
**Dependencies:** M0 split manifests; M1 runner and cost gate.

### Dataset order

1. MNIST hard pairs: 3v5, 4v9, 5v8, 7v9.
2. Fashion-MNIST: pullover/coat and shirt/T-shirt.
3. MedMNIST v2: PneumoniaMNIST and BreastMNIST.
4. KMNIST pair if MedMNIST integration or results stall.
5. Multi-class MNIST only after explicit architecture sign-off.

### Required controls

- Use the same split-manifest contract across all datasets.
- Record class mapping, source version, sample IDs, preprocessing, and counts.
- Keep binary breadth as the default answer to the multi-class reviewer question.
- Update dataset documentation and T1 from executed manifests.

**M4 gate:** At least three dataset domains and the required hard pairs have clean, reproducible results or documented governed fallback outcomes.

---

## 10. Milestone M5 — Phase 5: baselines and statistical rigor

**Target duration:** 4–5 focused days plus compute
**Dependencies:** M0 provenance; M1 runner; M4 datasets.

### M5.1 Classical comparisons

- Logistic regression.
- Effective-parameter-aware small MLP and an unconstrained MLP.
- Parameter-matched small CNN and a practical approximately-50k-parameter CNN.
- Same-input downsampled logistic control.
- Remove test-as-validation behavior from every classical path.

### M5.2 Quantum controls

- Cong, Hur, and TTN baselines on compatible registered instances.
- Random frozen FQCNN plus trained logistic readout.
- Encoding-only logistic regression on normalized amplitude vectors.
- Clearly disclose when a baseline cannot use the 10-qubit shape and use a justified compatible comparison rather than silently changing the headline model.

### M5.3 Statistics

- At least five seeds per cell; target ten where budget permits.
- Mean, standard deviation, and 95% bootstrap confidence intervals.
- McNemar tests using persisted per-example predictions.
- Wilcoxon signed-rank tests across seeds.
- Holm–Bonferroni correction across arms.

### M5.4 Calibration and learning curves

- Define defensible probability/calibration outputs rather than test-set min-max normalized scores.
- Generate reliability diagrams and expected calibration error.
- Run N={50,100,200,400,800,1600} learning curves for FQCNN and matched MLP/CNN.

### M5.5 Cost disclosure

Every main comparison row includes effective parameters, depth, two-qubit gates, shots/inference where relevant, and training wall-clock.

**Artifacts:** T3, F-F, prediction files, statistical reports, and cost disclosure.

**M5 gate:** No single-run result remains in the manuscript and no main-table comparison mixes incompatible external splits.

---

## 11. Milestone M6 — Phase 6: ablation grid

**Target duration:** 2 focused days plus queued compute
**Dependencies:** M1 budgeted runner; M2 pooling arms; M4 datasets; M5 statistics.

### Required axes

- Pooling: none, measurement, frozen unitary, approved coherent arm, SU(4).
- Convolution entanglement: none, one diagonal, full.
- Kernel rotations: RY, SU(2).
- Encoding: frozen amplitude, augmented arm, angle/compact arm.
- EMA and gradient clipping: on/off.

### Rules

- Change one factor at a time from the frozen model.
- Do not confound encoding with a different image size/qubit count without explicit reporting.
- Use at least five seeds and at least three datasets.
- Map every novelty claim to an ablation row.
- Remove claims that lack evidence instead of leaving incomplete rows.

**Artifacts:** T4 and the forest-style ablation figure.

**M6 gate:** Every contribution claim has an executed ablation or has been deleted from the paper.

---

## 12. Milestone M7 — Phase 7: realistic noise and real hardware

**Target duration:** 4–6 focused days plus QPU scheduling
**Dependencies:** M0 shared circuit; M1 runner; M2 pooling resources; M8 transpilation code.

### M7.1 Full-circuit noise

- Decompose amplitude state preparation into its actual gate sequence.
- Transpile state preparation plus model body.
- Use AerSimulator and calibration-derived fake-backend noise.
- Apply errors at their actual one- and two-qubit gate sites.
- Retain the current lightweight model-only sweep as a separately labelled secondary result.

### M7.2 Backend selection and provenance

- Select reproducible heavy-hex fake backends.
- Record backend snapshot identity, basis gates, coupling map, T1/T2, gate errors, and readout errors.
- Keep the training environment pinned; isolate Qiskit tooling if needed.

### M7.3 Real-QPU demonstration

- Use the six-qubit scaling-family instance.
- Rehearse the exact circuit, transpilation, sample count, shots, and mitigation path on a fake backend.
- Run 50–100 test samples with 1,024 shots on one backend and one seed.
- Compare without and with M3 readout mitigation; use ZNE only if budget remains.
- Record the ideal → fake → real ladder.

### M7.4 Fidelity threshold

- Derive the gate-fidelity threshold for viable 10-qubit full-circuit execution.
- Compare the threshold to the selected fake/real backend and stated near-term targets.
- Use this roadmap result if realistic full-state-preparation noise collapses accuracy.

**Artifacts:** F-C, noise tables, backend manifests, QPU job records, and fidelity-threshold derivation.

**M7 gate:** One real-QPU point exists and all noise claims include state-preparation cost and calibration provenance.

---

## 13. Milestone M8 — Phase 8: resources and scaling family

**Target duration:** 3–4 focused days plus compute
**Dependencies:** M0 shared circuit; M1 throughput; feeds M3 and M7.

### M8.1 Resource table

- Transpile to the selected `{RZ, SX, X, ECR}` basis.
- Separate state-preparation and model-body gates/depth.
- Report allocated and effective parameters.
- State parameter-shift circuit cost and per-inference shot scaling with concrete values.
- Quantify state preparation's percentage of the full circuit.

**Artifact:** T2.

### M8.2 Scaling sweep

- Run n={4,6,8,10,12,14}; evaluate n=16 only if the approved accelerator/budget allows.
- Record the actual active-wire and convolution/pooling schedule generated by each instance.
- Report accuracy, effective parameters, logical/transpiled gates, depth, memory, and wall-clock.
- Do not claim a 16→8→4→2→1 schedule unless the frozen family actually executes it.

**Artifact:** F-D and inputs to F-E.

### M8.3 Trainability integration

- Join scaling outputs with Phase 3 DLA and gradient-variance evidence.
- Explain discontinuities caused by grid factorization and odd-wire retirement.

**M8 gate:** Resource and scaling claims match transpiled and executed circuits, including state preparation.

---

## 14. Milestone M9 — Phase 9: references

**Target duration:** 1–2 focused days, with citation checks continuing through manuscript review.

### Work packages

1. Add full authors for every entry.
2. Add venue, volume, issue, pages, year, and DOI where applicable.
3. Replace arXiv entries with published versions when available.
4. Keep genuinely unpublished preprints below the target proportion and format them consistently.
5. Add references required by deferred measurement, state preparation, coherence, DLA/barren plateaus, effective dimension, generalization, datasets, mitigation, fake backends, and runtime tooling.
6. Audit every sentence/citation pair, including the unsupported image-locality citation.
7. Prefer a validated `.bib` workflow if it reduces inline bibliography errors without causing unnecessary manuscript restructuring.

**M9 gate:** No authorless entry remains and each citation supports its associated statement.

---

## 15. Milestone M10 — Phase 10: paper rewrite and venue package

**Target duration:** 5–7 focused days
**Dependencies:** Evidence artifacts from M0–M9.

### M10.1 Choose venue before style-sensitive edits

- IEEE Transactions on Quantum Engineering is the default format fit.
- Reassess Quantum Machine Intelligence, EPJ Quantum Technology, and Physical Review Applied using the strength of the hardware ladder and evidence package.
- Apply the journal template after evidence-driven content stabilizes.

### M10.2 Build the manuscript from evidence artifacts

Required tables:

- **T1:** datasets and clean split protocol.
- **T2:** state-preparation/model resource split.
- **T3:** reproduced baselines, uncertainty, paired tests, and cost.
- **T4:** ablation effects.
- **T5:** pooling arms and theory predictions.
- **T6:** capacity/trainability/generalization analysis.

Required figures:

- **F-A:** E1 exact tie and E2 dephasing.
- **F-B:** coherence/purity/entropy/mutual-information dynamics.
- **F-C:** ideal/fake/real noise ladder and threshold.
- **F-D:** scaling family.
- **F-E:** gradient variance and DLA evidence.
- **F-F:** learning curves and calibration.

### M10.3 Rewrite claims and structure

- Replace information-loss language with exact simulation and hardware-control advantages.
- Add the tested theorem/proposition section.
- Make every architecture equation match code.
- Report allocated versus effective parameters.
- Reframe locality in register/index space and report the translation experiment.
- Replace all single or leaked 98% claims with clean-protocol distributions.
- Add honest limitations: state-preparation dominance, simulable scale, binary scope, hardware budget, and no quantum-advantage claim.
- Document exact environment, seeds, manifests, weights, and reproduction path.

### M10.4 Submission package

- Prepare final source, figures, tables, supplementary derivations, environment lock, data manifests, and code archive.
- Create the Zenodo archive/DOI at submission readiness, not before artifacts stabilize.

**M10 gate:** Every manuscript claim maps to an evidence artifact, theorem, or explicit limitation.

---

## 16. Milestone M11 — final reproduction and red-team review

**Target duration:** 2–4 focused days plus fixes.

### Clean-checkout reproduction

The final reproduction path must:

1. verify the pinned environment;
2. run freeze and protocol tests first;
3. verify/download datasets and manifests;
4. run or resume all required cells;
5. regenerate T1–T6 and F-A–F-F;
6. validate manuscript artifact references;
7. emit a machine-readable completion report.

### Definition-of-done audit

Check every item in `UPGRADE_PLAN.md`, including:

- architecture freeze and unitarity;
- no test leakage;
- effective-parameter disclosure;
- paper/code reconciliation;
- theorem and E1/E2;
- dataset/seed/statistics requirements;
- reproduced comparisons;
- state-preparation-aware resources and noise;
- scaling/trainability evidence;
- one real-QPU point;
- truthful environment and full reproduction;
- complete bibliography.

### Reviewer-question red team

For every reviewer question in the plan, verify that:

- the promised artifact exists;
- the artifact is generated from traceable data;
- the manuscript cites it accurately;
- limitations and negative outcomes are visible;
- no claim survives solely because it appeared in an earlier draft.

**Final gate:** The program is complete only when clean reproduction, definition of done, reviewer map, and submission-package checks all pass.

---

## 17. Parallel work after the critical path

Once M2's E1/E2 gate passes, use the following safe concurrency:

- **Daytime CPU analysis:** DLA/effective dimension/expressibility development and validation.
- **Overnight queue A:** datasets, baselines, and ablations.
- **Overnight queue B:** scaling and gradient-variance runs when resources permit.
- **Hardware track:** transpilation, fake-backend rehearsal, and QPU scheduling.
- **Writing track:** theorem proofs, reconciliation-backed methods, references, and limitations; do not write final result claims before artifacts exist.

Do not run multiple workstreams against mutable shared outputs. Every track consumes immutable manifests and writes isolated artifacts.

## 18. Decision gates requiring explicit user input

The roadmap can proceed without repeated architectural choices except at these points:

1. **Inert `RY(0.02)` removal:** retain by default unless proof and 1e-12 regression satisfy §A2 and removal is approved.
2. **`pool_coherent`:** explicit sign-off before implementation/run; never promote to headline without another sign-off.
3. **Multi-class head:** explicit sign-off; default remains binary breadth across domains.
4. **Optional JAX/GPU path:** activate only if Phase 1 benchmarks miss the seven-night budget.
5. **Venue selection:** choose before final style-sensitive rewriting.
6. **Real-QPU submission:** approve the rehearsed, pre-sized job before spending the monthly allocation.

All other choices should follow the defaults in `UPGRADE_PLAN.md` and this roadmap.

## 19. Risk and contingency map

| Risk | Trigger | Roadmap response |
|---|---|---|
| Clean accuracy falls below 98% | Leakage-free retrain result | Report it; allow one validation-driven tuning pass; never tune on test |
| Effective count differs from the plan estimate | Committed gradient audit | Use the measured mapping everywhere and explain why |
| Batched speedup is insufficient | Cost estimate exceeds seven nights | Tune batch/validation cadence, then evaluate JAX/GPU, then apply mandated grid cuts |
| Measurement equivalence fails | E1 exceeds tolerance | Stop Phase 2; correct the measurement-channel mapping or theorem implementation |
| Coherent/SU(4) arms outperform frozen pooling | E3 result | Report family headroom; retain frozen headline unless explicitly approved otherwise |
| DLA stalls | Closure computation exceeds resource budget | Exact smaller-n analysis plus empirical variance sweep and explicit limitation |
| Generalization bound is vacuous | Derived numerical bound | Report it as vacuous; remove guarantee language |
| Realistic noise collapses accuracy | Full prep-aware Aer result | Report collapse; emphasize fidelity threshold and six-qubit QPU demonstration |
| QPU availability is delayed | No suitable job window | Keep Phase 7 incomplete; continue fake-backend/resource/paper work; execute when available |
| Grid has failed cells | Failure manifest nonempty | Repair and resume; do not aggregate incomplete required comparisons |
| Paper and evidence drift | Artifact validation fails | Regenerate or rewrite the claim; never hand-edit numbers into agreement |

## 20. Progress reporting

Maintain one program dashboard or machine-readable status summary with:

- milestone state;
- active gate and blockers;
- required versus completed run cells;
- projected and consumed compute time;
- QPU budget status;
- evidence artifact status T1–T6 and F-A–F-F;
- definition-of-done checklist status;
- architecture sign-off decisions.

A weekly progress report should answer:

1. What evidence became trustworthy this week?
2. Which gate passed or remains blocked?
3. What compute is queued next?
4. Which risks changed?
5. Which manuscript claims are now supported, rejected, or still untested?

## 21. Immediate next session

No implementation starts today. At the next implementation session:

1. Re-read §A of `UPGRADE_PLAN.md`.
2. Confirm the working branch/worktree strategy.
3. Create the milestone-specific Phase 0 implementation plan.
4. Begin only with the freeze guards and their fixtures.
5. Do not refactor, retrain, or launch experiments until the Phase 0 freeze gate passes.
