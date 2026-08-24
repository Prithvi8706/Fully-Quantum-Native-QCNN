# Journal Evidence Execution Design

## Goal

Finish roadmap Tasks 7–19 and produce a journal-ready paper around the existing FQCNN model. Improve the evidence, reproducibility, statistical support, figures, tables, citations, and manuscript package without changing the canonical FQCNN architecture.

A roadmap task is complete only when its focused tests, full-suite gate, canonical artifacts, and stated provenance checks pass. A queued campaign or submitted external job is reported as running or queued, never complete.

## Approved scope

The approved scope is **full evidence plus paper**:

- complete the remaining local datasets, baselines, statistics, ablations, model analyses, resource studies, noise studies, and hardware evidence;
- improve the manuscript, figures, tables, citations, traceability, and journal package;
- run clean-checkout reproduction and final reviewer-style red-team checks;
- preserve explicit human gates for compute launch, authenticated external access, real-QPU submission, venue selection, and publication.

## Non-negotiable architecture freeze

The canonical FQCNN model is frozen. This execution must not change:

- circuit topology or layer order;
- convolution or pooling semantics;
- feature-map definition;
- active-wire schedule;
- parameter allocation, sharing, or interpretation;
- readout observable or prediction rule;
- headline loss or training objective;
- the architecture fingerprint or frozen expectation regression.

Supporting infrastructure may change: campaign orchestration, dataset adapters, manifests, statistics, evidence schemas, exporters, resource counters, noise runners, hardware wrappers, figure generators, citation tooling, reproduction scripts, and manuscript source.

Any finding that appears to require a model change is a stop condition. Record the limitation and adjust the paper claim instead of modifying the FQCNN architecture. Ablation arms remain experiments around the frozen headline model and cannot silently replace it.

## Execution strategy

Use a hybrid gated pipeline:

1. Finish Task 7 before any canonical campaign or hardware execution.
2. Develop independent unit-level tooling and synthetic fixtures in parallel after shared schemas are frozen.
3. Serialize canonical training and Qiskit/hardware workloads through one exclusive queue.
4. Promote only complete, provenance-valid artifacts into downstream analyses or the manuscript.
5. Stop separately for authenticated access, real-QPU submission, venue selection, and publication.

Maximum parallelism is rejected because Tasks 7–15 share campaign schemas, run-artifact formats, dataset identities, and machine resources. Fully sequential development is also rejected because independent local tooling and tests can safely proceed concurrently.

## Cross-cutting contracts

### Provenance

Every canonical artifact must identify, as applicable:

- artifact schema and completion state;
- producer command and code commit;
- dirty-tree policy result;
- training or Qiskit environment-lock hash;
- input artifact paths and hashes;
- dataset, split, and ordered source sample identities;
- model/configuration identity;
- seeds and aggregation policy;
- external versus local status;
- caveats and negative results;
- generation timestamp.

Existing artifacts may use versioned adapters. No validator may silently assume older JSON already satisfies a new uniform schema.

### Fail-closed behavior

Missing, malformed, queued, historical, smoke, partial, failed, wrong-split, wrong-revision, or single-seed data cannot satisfy a final evidence requirement. Required commands must return nonzero on failure. Optional external work must use explicit statuses rather than shell failure masking.

### Compute queues

One local lease coordinates training, analysis workloads that materially contend for resources, Qiskit transpilation/noise workloads, and hardware submission preparation. Queue acquisition must be exclusive and rollback-safe. Stale-lease recovery must never delete a live owner's lease without explicit proof.

### Human approvals

Approval is separate from implementation and bound to exact scope. Changes to submission-affecting inputs invalidate approval. Human approval records cannot be manufactured by automated tests or treated as implied by passing code.

## Tranche A: Task 7, immutable campaign lifecycle

### Current state

The existing campaign artifacts are internally hash-consistent but launch-ineligible:

- they target commit `07ba7c5`, not current Task 7 HEAD;
- the manifest records a dirty repository;
- the full-suite record reports a failed clean-tree policy;
- launch approval is embedded as false rather than represented by a separate immutable attestation;
- the uncommitted second review adds regressions not implemented by current production code;
- there is no launch, runtime status, final failure record, or campaign output.

The campaign remains pending and unlaunched until every gate below passes.

### Lifecycle repairs

Implement the smallest changes needed to satisfy the second review:

- add a separate immutable approval artifact bound to campaign, final Git SHA, manifest hash, cost hash, and full-suite hash;
- keep planned manifests immutable and non-self-approved;
- propagate campaign experiment, run, split-manifest, and failure roots explicitly into Windows-spawned workers;
- add a rollback-safe shared exclusive-queue helper;
- establish exclusive campaign-directory ownership during planning;
- validate canonical campaign roots and all descendant artifact paths;
- validate strict request types and uniqueness;
- require positive `n_qubits` for baseline-enabled campaigns;
- distinguish scheduler cells, baseline side-effect cells, and total costed cells;
- recheck branch, upstream, ahead/behind counts, locks, worker resolution, cost, queue, and collisions at launch;
- validate launch records before status enters the runtime lifecycle;
- require final scheduler failure evidence before completion;
- treat malformed child evidence as failed;
- report running while the campaign owns the queue.

### Verification

Run, in order:

1. campaign-manifest regressions;
2. real Windows-spawn root-isolation coverage;
3. resume and parallelism tests;
4. cost and baseline contract tests;
5. the full suite;
6. `git diff --check` and an exact Task 7 diff review.

The full-suite evidence must be generated on the exact launch revision with a passing clean-tree policy.

### Evidence regeneration

Do not edit stale campaign evidence into validity. Regenerate the manifest, cost estimate, and full-suite evidence after code and tests are committed and the exact revision is known. Validate the unchanged approved request:

- campaign `baseline_mnist_n10_v1`;
- four MNIST class pairs;
- ten seeds;
- proposed model only;
- 400 samples;
- 30 epochs;
- no baselines;
- 40 scheduler cells;
- auto-resolved workers, currently expected to be 20;
- cost within the 70-hour budget.

The existing gated launch authorization applies only if this scope and estimate remain materially unchanged. Any changed dataset, model, seed count, worker policy, budget, or output scope requires a new decision.

### Launch sequence

1. Present final artifact hashes and gate results.
2. Create the separate immutable approval attestation.
3. Revalidate every gate immediately.
4. Acquire the shared queue.
5. Write the immutable launch record.
6. Start the campaign once.
7. Inspect the first completed cell for n=10 geometry, split identity, source sample IDs, one test evaluation, weights, predictions, and valid status.
8. Report running and allow unattended continuation only after first-cell validation.
9. Report complete only when all 40 cells validate, the queue is released, and the final scheduler failure manifest exists and is empty.

Any provenance drift, failed cell, malformed artifact, cost change, or queue conflict stops progress.

## Tranche B: Tasks 8–11, empirical evidence

### Task 8: deterministic dataset adapters

Freeze one shared adapter and provenance schema before parallel domain work. Support MNIST, Fashion-MNIST, KMNIST, and an approved MedMNIST task.

Required contract:

- dataset and classes are separate axes;
- source-stable string sample IDs;
- source version, checksums, license, and original split provenance;
- deterministic disjoint 60/15/25 project split;
- lower source label maps to `-1`, higher source label to `+1`;
- native 28x28 input, flatten to 784, zero-pad to 1024;
- deterministic uint8 scaling and per-example amplitude normalization;
- no dataset-wide fitted normalization;
- no implicit class balancing;
- smoke outputs cannot satisfy canonical evidence.

Completion requires at least three validated domains, adapter smoke cells, focused and full tests, `dataset_provenance.json`, and `t1_dataset_table.json`.

### Task 9: clean baseline and statistical campaign

Before launch, enrich run artifacts with structured selection records, selected weights, epoch histories, exact test-evaluation counts, ordered source IDs, split hashes, checkpoint hashes, and environment provenance.

Freeze the required model family and budgets before cost approval. At minimum, cover the proposed FQCNN, logistic regression, effective-parameter-matched MLP, Cong, Hur, TTN, and explicitly defined random-frozen and encoding-only controls. An optional CNN is included only if its role and environment are approved.

The prior Task 7 cost estimate excludes baselines and cannot approve Task 9. Measure a fresh cost gate after all baseline side effects are modeled.

Completion requires at least five valid seeds per approved domain, exact paired identities, validation-only selection, exactly one test evaluation, complete selected weights and histories, a final empty failure manifest, and validated T3 and F-F artifacts.

### Task 10: effective dimension and inductive bias

Consume stable Task 9 evidence without retraining missing controls internally. Compute:

- Fisher-based effective dimension with 74 effective slots as primary and 78/269 sensitivity arms;
- translation sensitivity under zero-fill shifts with circular-wrap control;
- register-space entanglement entropy versus cut with analytic control states.

Use validation-selected checkpoints and validation samples by default. Do not consume clean test data for analysis selection.

### Task 11: full one-factor ablation grid

Create a new immutable campaign. General amplitude arms use the frozen n=10 headline geometry. The 16-qubit feature-map arm remains a separately labelled geometry exception and is never presented as a strict one-factor n=10 comparison.

Require exact paired split and sample identities, declared one-factor semantics, at least the approved domain/seed floor, fresh cost approval, complete cells, a final empty failure manifest, paired confidence intervals, McNemar, Wilcoxon, and Holm correction.

Task 10 tooling and Task 11 aggregation may be developed in parallel after schemas stabilize. Their canonical compute remains serialized.

## Tranche C: Tasks 12–15, Qiskit and hardware

### One canonical exporter

Use one PennyLane-to-Qiskit exporter for pooling comparison, resource analysis, noise, fake rehearsal, and hardware payload generation. It must preserve operation and wire order, segment and hash state preparation/model body/readout, support dynamic operations, reject simulation-only channels as hardware payloads, prove small-circuit equivalence, and fail on unsupported operations.

The exporter is an evidence wrapper around the frozen model, not a new model implementation.

### Task 12: pooling practicality

Compare all approved pooling arms under identical local fake-backend target, layout, routing, optimization level, and transpiler seed. Record logical, decomposed, and transpiled metrics, dynamic-circuit properties, durations with units, and explicit equivalence limitations.

### Task 13: resource and scaling evidence

Separate state preparation, model body, terminal measurement, and total resources. Cover n=4,6,8,10,12,14 with actual stage schedules. Large-n timeout or memory caps produce explicit lower-bound, truncated, unsupported, or failed records, never fabricated exact values.

### Task 14: clean noise and fake-backend rehearsal

Replace historical RNG split reconstruction and hand-authored noise as canonical evidence. Use frozen clean manifests, weights, source IDs, preprocessing, canonical exported circuits, Aer noise, and one exact fake-backend rehearsal. Freeze an immutable payload and rehearsal hash. This task remains entirely local and must not authenticate or contact IBM services.

### Task 15: scoped real-QPU job

Use separate `rehearse`, `submit`, and `collect` lifecycles.

External gates:

1. explicit permission for authenticated account/backend preflight without submission;
2. user selection of a named backend and exact payload;
3. fresh one-use approval immediately before the network submission call;
4. durable immutable recording of the job ID;
5. separate collection of that exact job without retry or resubmission.

Any changed backend, service context, layout, circuits, samples, shots, mitigation, lock, Git revision, resource estimate, or payload hash invalidates approval and requires regeneration plus rehearsal.

## Tranche D: Tasks 16–17, citations and manuscript

### Task 16: sentence-level citation audit

Create an audit that distinguishes mechanical bibliography defects from editorial claim reviews. Every finding has a stable ID, exact manuscript location, claim text, citation keys, category, disposition, and resolution state.

Resolve unsupported trainability wording around `ref44`, retained preprints, comparison-table sources, hardware/noise claims, novelty language, state-preparation complexity, and any sentence that depends on incomplete upstream evidence.

The full DLA result must be reported honestly: no polynomial-DLA trainability certificate exists for this circuit. The paper may report finite-range gradient evidence and the negative DLA result but cannot convert them into a positive certificate.

### Task 17: manuscript and venue package

Generate T1–T6 and F-A–F-F only from canonical artifacts. Fail closed if any required input is incomplete. Maintain a claim registry mapping every numerical table cell, plotted coordinate, caption value, and prose claim to an artifact, pointer, hash, aggregation rule, identities, and caveats.

Remove or correct stale single-seed, historical-noise, unsupported hardware, architecture, trainability, and image-locality claims. Preserve accurate distinctions among allocated, tape-reaching, effective parameters, trainable gates, and total operations.

Human decisions remain required for venue, preprint dispositions, author/submission metadata, package scope, and claim strength if real-QPU evidence is absent. Package creation does not authorize upload, preprint posting, or journal submission.

## Tranche E: Tasks 18–19, reproduction and red team

### Task 18: clean-checkout reproduction

Run from a fresh isolated checkout at an approved exact commit. Create separate locked training and Qiskit environments. Run the full suite, bounded deterministic regeneration, complete evidence validation, offline QPU-record verification, manuscript asset generation, claim validation, three-pass LaTeX build, package-hash checks, and final clean-tree verification.

Do not resubmit a QPU job during reproduction. Validate immutable external records offline.

### Task 19: reviewer red team

Map anticipated reviewer questions to exact claims, artifacts, pointers, producer commands, split manifests, tests, and caveats. Audit test leakage, seed/domain floors, geometry, sample identity, paired statistics, generated assets, resource segmentation, parameter wording, DLA interpretation, finite-range trainability wording, simulability, pooling null results, noise/hardware scope, citation status, reproduction, and package integrity.

Completion requires zero open findings or named, reasoned, human-approved exceptions. Any correction returns to the owning tranche. Rerun Tasks 18 and 19 after the final correction.

The old definition-of-done item requiring a polynomial DLA trainability certificate must be explicitly revised or waived because the observed full `su(2^n)` DLA makes the planned certificate impossible. The paper must state the negative result rather than altering the architecture to obtain a preferred certificate.

## Parallel agent policy

After implementation-plan approval:

- assign agents to independent source/test domains only;
- freeze shared schemas before parallel implementation;
- isolate parallel edits when authorized;
- require each agent to return a narrow diff and focused verification;
- integrate centrally and rerun the combined full suite;
- never let agents create human approvals, launch unapproved compute, authenticate externally, select a venue, submit a QPU job, upload a package, or publish anything.

## Completion criteria

The remaining roadmap is complete only when:

- Tasks 7–19 have their required validated artifacts;
- the canonical FQCNN architecture and frozen regressions remain unchanged;
- every campaign has a final valid empty failure record;
- every numerical manuscript claim traces to canonical data;
- incomplete and negative results are labelled honestly;
- clean-checkout reproduction passes;
- the final red-team report has no unresolved findings;
- status, README, manuscript, evidence, and package agree;
- the final working tree is clean;
- external submission remains separately authorized.
