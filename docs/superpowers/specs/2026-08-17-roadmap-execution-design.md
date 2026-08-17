# FQCNN Roadmap Execution Design

## Goal

Execute the remaining project roadmap in dependency order, maximizing verified local progress on 2026-08-17 while launching long-running reproducible work where safe. A task is complete only when its tests or canonical evidence artifact pass. Queued experiments and external jobs are reported as launched, not complete.

## Scope

This design covers the 15 recommendations from the repository status audit:

1. Preserve the current branch commits.
2. Refresh project status documentation.
3. Fix baseline protocol leakage.
4. Correct the ablation circuit size.
5. Create an isolated Qiskit environment.
6. Add and run harder datasets.
7. Run the baseline and statistical study.
8. Complete effective-dimension and inductive-bias analyses.
9. Run the full ablation grid.
10. Run pooling hardware-practicality analysis.
11. Complete resource and scaling analysis.
12. Complete clean noise, fake-backend, and real-QPU work.
13. Finish the citation audit.
14. Integrate results into the final manuscript and venue package.
15. Run clean reproduction and red-team review.

The target for the first execution day is **unblock and launch**, not an unsupported claim that all computational and external work can finish in one day.

## Execution principles

- Preserve the frozen, trusted training environment.
- Use validation data for model selection and test data exactly once.
- Generate manuscript results only from clean manifests and canonical evidence artifacts.
- Keep source changes surgical and follow existing project patterns.
- Run focused tests before the full suite.
- Do not submit a real-QPU job without explicit approval immediately before submission.
- Do not present single-seed, queued, partial, historical-split, or failed runs as completed evidence.
- Allow parallel execution only after each job's protocol, inputs, and resource limits are verified.

## Phase A: Preserve work and establish an accurate dashboard

### Repository preservation

Inspect the 36 commits ahead of the upstream branch and confirm the intended remote and branch before pushing. Pushing is an outward-facing action and requires explicit approval at execution time.

Classify current untracked files into:

- intentional evidence,
- local logs,
- generated LaTeX/PDF output,
- and local Claude configuration.

Do not stage local configuration or generated output by default. Add ignore rules only when they match the repository's intended artifact policy.

### Documentation repair

Update `STATUS.md` so it accurately records:

- completed E1 through E4 work,
- five completed M3 analyses,
- the full-DLA negative result,
- actual remaining blockers,
- current run-cell and definition-of-done counts,
- and the next execution queue.

Update only misleading README statements that affect reproducibility or project status, including the environment version, headline-result qualification, baseline protocol state, and historical noise/hardware limitations. Do not install final manuscript claims prematurely.

### Verification

- Inspect the documentation diff against canonical evidence artifacts.
- Confirm no generated or local-only files are staged.
- Run relevant documentation or repository checks if available.

## Phase B: Remove local protocol blockers

### Baseline leakage fix

Refactor baseline entry points so their interfaces distinguish training, validation, and test inputs. Validation controls model selection. Test evaluation occurs once after selection.

Cover both the classical baseline runner and optional CNN path. Add regression tests that fail if test data are supplied as validation or if test evaluation occurs during tuning.

Success criteria:

- focused protocol tests pass,
- existing baseline tests pass,
- the full test suite passes,
- and no manuscript baseline result is generated through the contaminated path.

### Ablation circuit-size correction

Move general non-encoding ablations from the old n=8 setup to the n=10 headline circuit. Retain a smaller feature-map experiment only if its geometry is inherently different and the resulting table and caption identify that exception explicitly.

Success criteria:

- the registry resolves general ablations to the n=10 headline configuration,
- tests lock the intended configuration,
- and the full test suite remains green.

## Phase C: Isolate hardware tooling

Create a separate locked environment for Qiskit 1.x, IBM Runtime, Aer, and fake-backend tooling. Do not upgrade or mutate the frozen training environment.

Adapt hardware-facing runners to use:

- current circuit definitions,
- clean manifests,
- explicit transpilation settings,
- backend and package-version provenance,
- and a small rehearsed scaling-family circuit.

Verification proceeds in stages:

1. Import and version smoke test.
2. Local transpilation smoke test.
3. Fake-backend execution.
4. Exact payload rehearsal.
5. Real-QPU submission only after explicit approval.

## Phase D: Empirical campaign

### Dependency chain

1. Implement dataset adapters and deterministic manifests.
2. Run the clean multi-seed baseline grid.
3. Use trustworthy controls for effective dimension and inductive bias.
4. Run the full one-factor ablation grid.
5. Run pooling hardware-practicality comparisons.
6. Produce state-preparation/model-body resource decompositions and scaling sweeps.
7. Run clean full-circuit noise and fake-backend experiments.
8. Rehearse and, after approval, submit the real-QPU point.

### Controlled parallelism

After protocol gates pass, independent experiment cells may run concurrently through the existing resumable scheduler. Each job must have:

- a frozen manifest,
- a unique output location,
- a recorded seed,
- a resource estimate,
- resumability or a failure manifest,
- and an expected canonical evidence artifact.

Long-running work may continue beyond the first execution day. Its state must be reported as pending, running, failed, or complete rather than collapsed into a binary roadmap checkbox.

## Phase E: Integration and audit

Once the required evidence exists:

1. Finish the sentence-level citation audit and preprint editorial decision.
2. Select the venue and apply its template.
3. Rewrite claims around the actual evidence, including the full DLA, corrected Caro gate count, SU(4) headroom, absence of coherent-extension benefit, and information-dynamics result.
4. Replace single-run claims with distributions where the protocol requires them.
5. Generate final tables and figures from canonical evidence artifacts.
6. Reproduce from a clean checkout.
7. Run the definition-of-done and reviewer-question audit.
8. Correct every failed audit item before declaring submission readiness.

## Error handling and stopping rules

Stop the affected workstream when:

- a protocol test fails,
- input provenance cannot be established,
- a job would use test data for selection,
- the expected resource use exceeds the approved budget,
- the frozen training environment would need mutation,
- or an external action lacks approval.

Failures must produce enough information to diagnose or resume the job. Do not silently substitute a smaller circuit, fewer seeds, a historical split, or an easier dataset.

## Testing strategy

Each implementation unit follows this verification order:

1. Add or identify a test that captures the required protocol or behavior.
2. Confirm the test detects the pre-change defect when practical.
3. Make the smallest implementation change.
4. Run focused tests.
5. Run the full suite.
6. Run a smoke experiment when the change affects experiment execution.
7. Inspect the generated manifest or evidence artifact.

Experiment milestones additionally require schema validation and internal consistency checks before their outputs are used in documentation or the manuscript.

## First-day completion target

The realistic first-day target is:

- current commits safely preserved or ready for an explicitly approved push,
- untracked files classified,
- status documentation repaired,
- baseline leakage fixed and tested,
- n=10 ablation configuration fixed and tested,
- isolated Qiskit environment created and smoke-tested,
- and the first verified long-running empirical jobs launched if all prerequisite gates pass.

The entire 15-step roadmap is not considered complete until the empirical outputs, hardware work, manuscript integration, clean reproduction, and final audit have all passed their respective gates.
