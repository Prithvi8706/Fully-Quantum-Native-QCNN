# FQCNN Q1 Upgrade Execution Design

**Date:** 2026-07-23
**Status:** Approved design
**Governing specification:** `UPGRADE_PLAN.md` (v2)
**Target outcome:** Complete Phases 0–10 and every non-optional item in the upgrade plan's definition of done, producing a reproducible Q1-journal submission package.

## 1. Authority and scope

`UPGRADE_PLAN.md` is the governing specification for this program. This design defines how to execute it safely and verifiably; it does not reduce its scope.

The following rules are invariant throughout execution:

1. The headline FQCNN circuit is frozen as required by §A1–A3.
2. The protected behavior is `(x, θ) → ⟨Z_readout⟩`, not incidental source layout.
3. Main-pipeline execution remains unitary until the terminal measurement.
4. Variants remain labelled experimental arms and never silently replace the headline model.
5. Anything marked `[ARCH — opt-in]` receives explicit user sign-off at the point of implementation.
6. Test information never influences training, model selection, scheduling, checkpointing, or tuning.
7. Negative or disappointing results are retained and reported.
8. Completion means all Phases 0–10 and all non-optional definition-of-done items are finished; later research phases are not dropped after the critical path.

## 2. Grounded repository findings

Repository inspection identified execution details that the implementation must measure or correct while preserving the upgrade plan's intent:

- The current archived headline model is the 10-qubit, 28×28 amplitude-encoding model reconstructed from `Results/metadata.json` and `Results/Weights/quantum_model_params.npz`.
- A preliminary Jacobian audit found 74 output-effective parameter slots rather than the plan's approximate 76. Phase 0.2 must generate the authoritative committed mapping and tolerance before the manuscript uses a final count.
- `experiments/run_experiments.py` currently labels an 8-qubit, 16×16, 260-slot configuration as `proposed`; this must not be conflated with the frozen headline model.
- The current measurement-pooling implementation is not the theorem-equivalent measure-and-condition channel, so the existing arm cannot support E1 without correction.
- Batched amplitude execution works on the pinned PennyLane 0.38 `default.qubit` backpropagation path, but not on the current `lightning.qubit` path. The sequential path remains the regression oracle.
- Parallel execution is unsafe until weights and temporary outputs are isolated per run.
- Existing result files originate from incompatible splits, and exact split indices are not persisted.
- The current hardware runner is a CLI/preflight sketch rather than a valid shared-circuit QPU execution path.
- With the current four-layer loop, the n=16 active-wire schedule is `16→8→4→2`; scaling reports must use the executed schedule rather than assume `16→8→4→2→1`.
- Claims such as a non-vacuous generalization bound and feasible exact n=10 DLA computation are research hypotheses with explicit fallback paths, not guaranteed outcomes.

These findings refine implementation and acceptance criteria. They do not override architecture freeze, unitary execution, honesty, or full-program completion.

## 3. Chosen execution approach

The program uses an **evidence-first gated approach**.

Phase 0 establishes model identity, protocol integrity, reproducibility, and a single circuit source. Phase 1 makes the required studies affordable without changing model behavior. Phase 2 validates the paper's defining pooling theory and experimental implementation. Broader experiments and analyses begin only after those foundations pass.

This approach is preferred over a parallel research sprint because current split, circuit-duplication, output-path, and experiment-registration defects could invalidate several expensive workstreams at once. It is preferred over a paper-first rewrite because clean retraining and E1/E2 results will determine the final defensible claims.

## 4. System architecture

### 4.1 Frozen headline definition

The headline configuration is an explicit, immutable experiment registration containing:

- dataset preprocessing and 28×28 input geometry;
- 10-qubit amplitude encoding;
- the current convolution, unitary pooling, classifier, and readout schedule;
- the archived 269-slot parameter layout;
- archived trained weights and metadata;
- a canonical parameter-slot naming scheme;
- a canonical topology fingerprint;
- fixed input/output regression fixtures.

The freeze tests define model identity. Refactoring is permitted only when those tests prove observational equivalence. The inert-gate exception remains governed by §A2.

### 4.2 Canonical circuit builder

One circuit builder becomes the source of topology for:

- headline model training and inference;
- noise simulation;
- hardware execution and transpilation;
- circuit drawing;
- resource counting;
- pooling state probes;
- model-analysis tools;
- scaling-family instances.

Evaluation-only hooks may observe intermediate states or inject noise/dephasing in experimental paths. They must not add non-unitary operations to the headline model definition.

Quantum comparison baselines remain separate architectures.

### 4.3 Split and dataset service

A central dataset/split service:

- identifies the source dataset and class mapping;
- records selected sample IDs before splitting;
- creates seeded stratified 60/15/25 train/validation/test partitions;
- verifies disjointness and class balance;
- persists exact indices in a split manifest;
- allows all models and baselines to consume identical splits;
- supports MNIST, Fashion-MNIST/KMNIST IDX data, and a MedMNIST adapter.

A run never reconstructs a historical split by replaying an assumed random-number sequence.

### 4.4 Run artifact store

Each `(dataset, split, configuration, seed)` run owns an isolated directory containing:

- resolved configuration;
- split-manifest identity;
- code revision and environment identity;
- run state: `pending`, `running`, `complete`, or `failed`;
- weights and checkpoint history;
- per-example sample IDs, predictions, scores, and labels;
- epoch history and timings;
- aggregate metrics;
- logs and failure details.

A shared archived headline weight file is never used as a worker output destination.

### 4.5 Experiment registry

The registry distinguishes:

- the frozen 10-qubit headline model;
- quick smoke configurations;
- scaling-family instances;
- classical baselines;
- quantum baselines;
- pooling, encoding, convolution, training-recipe, and noise arms.

Each arm records whether it is headline, baseline, ablation, scaling instance, or opt-in architecture variant. Configuration names cannot imply equivalence to the headline model unless freeze identity is verified.

### 4.6 Evidence generation

Machine-readable run artifacts feed:

1. validated statistical aggregation;
2. generated tables and figures;
3. resource and hardware reports;
4. manuscript claim checks;
5. the final reproduction package.

No final manuscript number is copied manually from a terminal log.

## 5. Program gates

### Gate A — trusted model identity

Required before any refactor or new experiment:

- governing sources are versioned;
- circuit fingerprint passes;
- archived-weight expectation regression passes;
- effective-parameter mapping is committed;
- main-pipeline unitarity audit passes.

### Gate B — clean protocol and provenance

Required before retraining or baselines:

- split manifests are deterministic, stratified, and disjoint;
- validation alone controls model selection;
- test evaluation occurs once per run;
- seeds and environment metadata are accurate;
- per-run outputs are isolated.

### Gate C — behavior-preserving affordability

Required before the full grid:

- shared circuit builder matches the frozen oracle;
- batched and sequential outputs, losses, gradients, and one optimizer update agree;
- cached and uncached encoding agree;
- resume and parallel execution are deterministic and race-free;
- the cost estimator projects the approved grid within seven unattended nights or applies the upgrade plan's documented cut order.

### Gate D — pooling theory implementation

Required before broad theory-dependent claims:

- the corrected measurement arm represents the theorem's channel;
- E1 agrees with unitary pooling to the specified machine-precision tolerance;
- E2 dephasing hooks and measurements are verified;
- state/resource instrumentation is ready for E3–E5;
- opt-in architecture arms remain blocked until explicit sign-off.

## 6. Critical-path workflow

### 6.1 Phase 0 — freeze guards and protocol integrity

1. Add freeze, expectation, unitarity, and gradient-audit tests.
2. Establish canonical fixtures from archived metadata, weights, and fixed inputs.
3. Implement central split generation and persisted manifests.
4. Refactor the trainer to consume train and validation data only; perform final test evaluation outside model selection.
5. Isolate run outputs and persist per-example results.
6. Extract the shared circuit builder and prove regression equivalence.
7. Complete the equation-by-equation paper↔code reconciliation table.
8. Repair the environment lock, seed policy, setup scripts, and reproduction entry point.
9. Remove only the non-circuit dead artifacts explicitly permitted by the upgrade plan.

### 6.2 Phase 1 — experiment affordability

1. Implement batched `default.qubit` backpropagation.
2. Compare sequential and batched outputs, losses, gradients, and optimizer updates.
3. Cache padded normalized amplitude inputs and fixed validation subsets.
4. Add safe process parallelism and completed-run resume behavior.
5. Add a cost estimator covering training, validation, baselines, workers, and memory.
6. Enforce the seven-night grid gate.
7. Evaluate JAX/GPU acceleration only if required after the default path is benchmarked.

### 6.3 Phase 2 — pooling theory and experiments

1. Implement the exact theorem-equivalent measurement arm.
2. Add dephasing and intermediate-state probes outside the headline definition.
3. Implement `pool_none`, `pool_measurement`, `pool_unitary`, and `pool_su4` as labelled arms.
4. Present the §A5 sign-off gate before implementing or running `pool_coherent`.
5. Run E1 first; stop theory-dependent execution if equivalence fails.
6. Run E2 fixed-parameter dephasing.
7. Run E3 retrained head-to-head with matched data, seeds, and budgets.
8. Run E4 coherence, purity, entropy, and mutual-information analysis.
9. Run E5 transpiled hardware-practicality analysis.
10. Write and verify Theorem 1 and Propositions 2–3 against the tested implementations.

## 7. Remaining milestone sequence

All remaining phases are mandatory except items explicitly marked optional or opt-in by `UPGRADE_PLAN.md`.

### M3 — Phase 3: model analysis

Complete DLA analysis or its documented exact-small-n fallback, gradient-variance scaling, effective dimension, expressibility, entangling capability, generalization-bound analysis, translation-sensitivity testing, entanglement profile, and the classical-simulability statement.

### M4 — Phase 4: dataset breadth

Run MNIST hard pairs, Fashion-MNIST pairs, MedMNIST tasks, and the KMNIST fallback policy. Update dataset documentation and executed split tables. Multi-class MNIST remains behind its architecture sign-off gate.

### M5 — Phase 5: baselines and statistics

Run reproduced classical and quantum controls on identical splits; persist per-example outputs; calculate mean/std, bootstrap confidence intervals, McNemar tests, Wilcoxon signed-rank tests, and Holm–Bonferroni corrections; generate calibration analysis, learning curves, and cost columns.

### M6 — Phase 6: ablations

Execute the one-factor-at-a-time pooling, convolution-entanglement, kernel-rotation, encoding, EMA, and gradient-clipping grid. Every manuscript novelty claim must map to a completed ablation row or be removed.

### M7 — Phase 7: noise and real hardware

Model decomposed state preparation and model gates with realistic Aer/fake-backend noise; retain the lightweight model-only sweep with explicit scope; rehearse and execute the bounded real-QPU run; compare ideal, fake, and real results with mitigation; derive the fidelity-threshold roadmap.

### M8 — Phase 8: resources and scaling

Generate logical and hardware-basis resource tables separating state preparation from the model body. Execute the n=4…14 scaling family, report actual active-wire schedules, accuracy, effective parameters, gates, depth, and wall-clock, and connect the outputs to Phase 3 trainability analysis.

### M9 — Phase 9: references

Complete author, venue, volume, page, year, and DOI metadata; replace preprints with published versions where available; add the references required by new theory, datasets, mitigation, and tooling; verify that each citation supports its sentence.

### M10 — Phase 10: manuscript and venue

Generate T1–T6 and F-A–F-F from validated evidence, reconcile all equations and claims, rewrite the abstract/introduction/theory/experiments/limitations/reproducibility sections, select the venue before style-sensitive editing, apply the journal template at the end, and prepare the code/data archive package.

### M11 — final red-team and definition of done

Run a clean-checkout reproduction, walk every definition-of-done checkbox, verify the reviewer-question evidence map, confirm no hidden test leakage or architecture drift, and declare completion only when every required item passes.

## 8. Data flow

```text
dataset source
  → immutable source/sample manifest
  → seeded train/validation/test split manifest
  → cached encoded inputs
  → registered model/configuration/seed run
  → isolated weights, predictions, metrics, timings, environment, and logs
  → validated aggregate statistics and scientific analyses
  → generated tables, figures, resource reports, and theorem checks
  → manuscript claims, venue package, and reproducibility archive
```

Every reported result must be traceable backward through this chain.

## 9. Failure, resume, and budget policy

- Freeze-test failure blocks all later work.
- Split or provenance failure invalidates affected runs; results are not repaired by relabelling.
- Partial artifacts are never considered complete.
- Required grid failures are recorded in a failure manifest and cause a nonzero command exit.
- Resume skips only complete, schema-valid artifacts whose configuration and split identities match.
- Worker processes never write shared checkpoints or temporary files.
- The cost gate uses the upgrade plan's reduction order: seeds, then datasets, then non-pooling ablations. Required pooling E1/E2 work is not cut.
- If DLA computation stalls, exact smaller-n computation plus the empirical variance sweep is used and disclosed.
- If the generalization bound is vacuous, that result is reported rather than rewritten as a guarantee.
- If realistic noise collapses performance, the fidelity-threshold result and smaller hardware instance remain the planned contribution.
- If QPU access fails temporarily, the rehearsed job remains pending; Phase 7 and the final definition of done are not declared complete until the required real-hardware point is obtained.

## 10. Verification strategy

### 10.1 Architecture tests

- canonical fingerprint fixture;
- archived expectation fixture;
- unitary-operation audit;
- named effective-gradient map;
- inert-gate exception proof and stricter regression if invoked.

### 10.2 Protocol tests

- split reproducibility, stratification, and disjointness;
- identical split consumption across arms;
- validation-only model-selection controls;
- exactly one final test evaluation;
- correct seed and environment recording.

### 10.3 Execution tests

- shared builder versus frozen oracle;
- sequential versus batched outputs, losses, gradients, and optimizer update;
- cached versus uncached inputs;
- single-worker versus multi-worker equivalence;
- resume/idempotency and output isolation.

### 10.4 Scientific-analysis tests

- E1 and E2 numerical predictions;
- known small-state coherence, purity, entropy, and mutual-information fixtures;
- statistical routines against small known datasets;
- resource counts at logical and transpiled levels;
- scaling schedule extraction from executed tapes;
- table/figure provenance checks.

### 10.5 Reproduction tests

The final reproduction entry point must:

1. verify the environment;
2. run freeze and protocol tests first;
3. reconstruct or validate datasets and split manifests;
4. run or resume all required experiment cells;
5. regenerate every final table and figure;
6. validate manuscript artifact references;
7. emit a completion report matching the definition of done.

## 11. Completion criteria

The program is complete only when:

- every non-optional checkbox in `UPGRADE_PLAN.md` is satisfied;
- every `[ARCH — opt-in]` item has either explicit approval and completed evidence or an explicit recorded decision to remain off, as required by the governing plan;
- all final claims trace to generated evidence;
- the clean-checkout reproduction succeeds;
- the real-hardware point required by the definition of done exists;
- the journal-formatted manuscript, references, code, data manifests, environment lock, tables, figures, and archive package are ready for submission.

## 12. Implementation-planning boundary

The implementation plan derived from this design will provide file-level tasks, tests, commands, and checkpoints for Phases 0–2. It will provide complete deliverables, dependencies, acceptance criteria, and intended module boundaries for Phases 3–10. Before each later scientific milestone is implemented, its milestone task will be refined against outputs from the preceding gates. This just-in-time refinement prevents speculative algorithms while preserving full scope and the requirement to finish the entire upgrade.
