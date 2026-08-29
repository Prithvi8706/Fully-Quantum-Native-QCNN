# FQCNN Q1-Journal Fast Track

**Approved:** 2026-08-27

**Goal:** Produce a Q1-targeted, technically defensible, reproducible journal submission without executing the former twelve remaining tasks as twelve separate research projects.

**Time box:** Seven focused workdays plus at most two unattended compute windows. No implementation workstream may grow beyond one focused day without a fresh scope decision.

**Meaning of Q1-targeted:** The package deliberately includes multi-domain evaluation, repeated seeds, relevant classical and quantum baselines, ablation evidence, local noise/resource analysis, conservative claims, and clean reproduction. Journal quartile and acceptance remain external editorial outcomes and cannot be guaranteed by a plan.

**Supersedes:** The execution sequence for Tasks 8–19 in `2026-08-24-journal-evidence-execution-design.md` and `2026-08-17-roadmap-execution.md`. Their scientific concerns are compressed into five evidence gates below rather than discarded. Task 7 is not reimplemented; its completed `origin/dev` work is integrated and verified once.

## 1. Q1 submission thesis

The paper is a fully unitary QCNN architecture, theory, and empirical-analysis contribution. It will support these claims:

1. the frozen FQCNN performs encoding, convolution, pooling, and classification without intermediate measurement;
2. the coherent pooling construction exactly represents the compared measurement-discard behavior on the retained subsystem while removing dynamic-circuit requirements;
3. repeated experiments across three image-dataset families show where the architecture works and where it does not;
4. comparisons with classical and quantum baselines quantify performance without claiming quantum advantage;
5. pooling ablations, effective-parameter analysis, finite-size gradients, expressibility, DLA, clean local noise sensitivity, and logical/transpiled resources characterize the model beyond accuracy;
6. every submitted numerical claim is traceable to versioned evidence and reproducible from the frozen environments.

The paper does **not** claim state-of-the-art image classification, proven absence of barren plateaus, broad medical generalization, hardware execution unless a real job is separately approved and completed, or quantum advantage.

## 2. Non-negotiable constraints

- Preserve the canonical FQCNN architecture and all frozen regression expectations.
- Reuse validated evidence before running new experiments.
- Retain Q1 evidence breadth, but implement only the thinnest dataset, analysis, and evidence paths needed for that breadth.
- Test data remain exactly-once evaluation data. Model selection uses validation data only.
- Report seed counts, distributions, uncertainty, and negative results; no single-run headline number returns.
- Historical, smoke, partial, malformed, wrong-split, or provenance-incomplete artifacts cannot become headline evidence.
- A failed optional analysis removes its claim. It must not silently lower a required multi-domain, baseline, statistical, or reproducibility gate.
- No authenticated service, QPU submission, upload, venue selection, or journal submission is authorized by this plan.

## 3. Compression of the old twelve tasks

| Old task | Q1 fast-track decision | Minimum retained evidence |
|---|---|---|
| 8 — harder-dataset adapters | **Keep, narrow** | Thin deterministic adapters for MNIST, Fashion-MNIST, and KMNIST; one predeclared binary task per family plus MNIST `0,1` calibration. |
| 9 — full baseline/statistical campaign | **Keep, narrow** | Five seeds; proposed FQCNN, logistic regression, compact two-unit MLP, and one hierarchical quantum baseline. |
| 10 — effective dimension and inductive bias | **Replace** | Use existing DLA, gradient, expressibility, generalization, and simulability evidence. Remove image-locality claims; no new effective-dimension framework. |
| 11 — full one-factor ablation grid | **Replace** | Use the completed pooling-arm grid and add one three-seed Fashion-MNIST transfer check. |
| 12 — pooling hardware practicality | **Keep, narrow** | One controlled local fake-backend comparison of the principal pooling arms. |
| 13 — n=4…14 resource scaling | **Keep, narrow** | Logical and transpiled resources for n=`4,6,8,10`; separate state preparation and model body. |
| 14 — clean noise/fake-backend campaign | **Keep, narrow** | A validation-set noise ladder on two representative tasks plus zero-noise agreement and one fake-backend rehearsal. |
| 15 — real-QPU job | **Conditional strengthening** | Prepare and rehearse the payload. Submit only with fresh user approval and only if access fits the schedule; otherwise state simulation-only scope. |
| 16 — citation audit | **Keep, simplify** | Manuscript-focused sentence audit with resolved findings; no general audit platform. |
| 17 — manuscript/package | **Keep** | Rewrite and generate tables/figures only from retained evidence. |
| 18 — clean reproduction | **Keep, bound** | Full tests, deterministic evidence regeneration/validation, and manuscript build from a clean worktree. |
| 19 — reviewer red team | **Keep, simplify** | One finite Q1 checklist and one correction pass; no general red-team framework. |

## 4. Frozen experimental design

Freeze this design before inspecting comparative test results:

| Dataset family | Binary task | Role |
|---|---|---|
| MNIST | `0,1` | continuity with the headline result; easy calibration task |
| MNIST | `3,5` | harder in-domain task |
| Fashion-MNIST | `0,6` | visually confusable cross-domain task |
| KMNIST | `2,3` | distinct character-domain task |

The Fashion-MNIST and KMNIST pairs are selected from label semantics before test inspection. If source metadata makes a pair invalid, record the reason and replace it before any test evaluation.

Common protocol:

- native 28×28 grayscale input, flattened to 784 and zero-padded to 1,024 amplitudes;
- source-stable sample IDs and recorded source/version/checksum;
- deterministic disjoint 60/15/25 project split;
- 400 training examples and 30 epochs unless the already approved protocol mandates a smaller availability-bound count;
- seeds `0..4` for the proposed model and primary baselines;
- identical ordered samples and split hashes for paired comparisons;
- validation-only checkpointing and exactly one final test evaluation per trained run.

Implementation note: because the persisted split uses integer rounding, the
400-example training budget is obtained from a 666-example pre-split pool
(`round(0.60 * 666) = 400`), yielding 100 validation and 166 test examples.
Every scientific manifest must expose these counts; the legacy 343-row result
from the former 70% presubset calculation is not admissible.

Primary comparators:

- logistic regression;
- compact two-unit dense MLP, with its larger parameter count reported explicitly;
- TTN as the hierarchical quantum comparator.

**Baseline accounting correction (2026-08-29):** “matched” means paired on the
same ordered samples, split identity, seed set, and selection protocol; it does
not mean equal dense parameter count. With 784 flattened input features, the
smallest supported dense MLP has two hidden units and 1,573 trainable parameters
(`784*2 + 2 + 2 + 1`), versus 785 logistic coefficients and 269 FQCNN allocated
slots (74 effective). The MLP is therefore reported as a compact two-unit
comparator, never as parameter-matched; this conservative capacity mismatch is
part of the result and is disclosed in the manuscript and aggregate protocol.

Cong and Hur may be included only when their already implemented paths finish inside the same compute cap. They are strengthening arms, not completion gates. CNN, new literature implementations, and additional datasets are out of scope.

## 5. Required evidence matrix

| Paper element | Q1 fast-track minimum | Preferred source |
|---|---|---|
| Architecture identity and unitarity | Frozen signature, expectation, and no-nonunitary-operation tests pass unchanged | Existing fixtures and architecture tests |
| Effective parameters | 269 allocated, 78 tape-reaching, 74 effective at the committed tolerance | `tests/fixtures/effective_params.json` |
| Pooling theory | Exact equivalence, dephasing, information-dynamics, and negative/null findings validate | Existing E1/E2/E4/T5 evidence |
| Multi-domain performance | Four predeclared tasks across three dataset families, five seeds, exact split/sample provenance | Bounded Q1 comparison campaign |
| Baselines | Logistic regression, compact two-unit MLP, and TTN on identical data; paired where identities permit | Existing baseline modules with selector support |
| Statistics | Per-task mean, standard deviation, 95% CI, paired deltas/tests, effect sizes, and Holm correction for declared primary comparisons | One bounded aggregation module |
| Ablation | Existing five-seed pooling grid plus three-seed Fashion-MNIST pooling transfer check | Existing arms and bounded fill run |
| Trainability/model analysis | Existing DLA, gradient-variance, expressibility, generalization, and simulability artifacts; negative DLA result retained | Existing Phase 3 evidence |
| Noise sensitivity | Clean validation-selected checkpoints, zero-noise equivalence, declared noise levels, two tasks, no hardware claim | Bounded local Qiskit/Aer evidence |
| Resources/practicality | n=`4,6,8,10` logical/decomposed/transpiled counts and one fake-backend pooling comparison | Existing isolated Qiskit environment |
| Reproducibility | Frozen locks, full tests, evidence validation, LaTeX build, exact commit and package hashes | Final clean-worktree gate |

If preferred evidence fails, the order is: repair metadata without altering results; rerun only missing bounded cells; narrow the associated claim. The multi-domain, primary baseline, statistics, and reproduction rows cannot be deleted while calling the package Q1-targeted.

## 6. Minimal file-change map

Only the following implementation surface is planned:

- create `QCNN/utils/dataset_registry.py` as a thin registry for the three frozen dataset sources, labels, versions, checksums, and sample IDs;
- minimally extend `QCNN/utils/dataset_loader.py` and `experiments/run_experiments.py` with repeated `--task dataset:low,high` inputs that become separate dataset and class fields in manifests, while preserving the current MNIST path;
- extend `baselines/quantum_baselines.py` and the runner with an explicit quantum-baseline selector so only TTN is required;
- create `experiments/q1_fast_track_analysis.py` for inventory, run validation, aggregation, statistics, and canonical table JSON;
- create `experiments/q1_local_evidence.py` for the bounded resource, pooling-practicality, clean-noise, and fake-backend commands using the existing isolated Qiskit environment;
- create `tests/test_q1_fast_track.py` for dataset identities, baseline selection, aggregation, resource segmentation, and zero-noise equivalence;
- modify `fqcnn.tex`, `README.md`, `STATUS.md`, and `docs/paper_code_reconciliation.md` to match the final evidence;
- create only the inventory, comparison, local-evidence, claim-ledger, reproduction, reviewer-checklist, and package-manifest artifacts named below.

Do not build a general dataset platform, effective-dimension framework, full ablation system, new circuit exporter, hardware client, citation platform, or red-team platform. Any additional production file requires a written reason that an existing module or honest claim correction cannot solve the problem.

## 7. Five-work-package implementation plan

### Work package A — integrate Task 7 and freeze the evidence inventory (half day)

**Outcome:** One trusted starting revision and a keep/rerun/remove decision for every current manuscript result.

1. Preserve current uncommitted user files and inspect their overlap with the twelve Task 7 commits on `origin/dev`.
2. Integrate Task 7 without recreating its queue, lifecycle, or validation code.
3. Run Task 7, frozen architecture, protocol, and full-suite tests.
4. Create `Results/evidence/q1_fast_track_inventory.json` with hashes, producer status, dataset, pair, seeds, split/sample identities, and disposition for every candidate artifact.
5. Inventory every numerical manuscript sentence, table, and figure. Flag single-seed values, historical noise, hardware implications, image-locality wording, DLA overclaims, and hand-entered comparison rows.

```powershell
python -m pytest tests/test_campaign_manifest.py tests/test_resume_and_parallelism.py tests/test_cost_estimator.py -q
python -m pytest tests/test_freeze_architecture.py tests/test_freeze_expectations.py tests/test_headline_identity.py tests/test_protocol.py -q
python -m pytest tests/ -q
python -m experiments.q1_fast_track_analysis inventory --manuscript fqcnn.tex --output Results/evidence/q1_fast_track_inventory.json
```

**Exit gate:** All protected tests pass, every retained claim has a candidate evidence source, and all new work fits the two compute windows.

**Stop rule:** Task 7 integration gets at most two focused hours beyond conflict resolution and tests. If it overruns, use the known passing remote revision as the clean base and reapply only intentional local edits.

### Work package B — add the thin multi-domain path (one day)

**Outcome:** Deterministic, tested access to the four frozen tasks without a general dataset framework.

1. Add source/version/checksum/label metadata and stable source sample IDs for MNIST, Fashion-MNIST, and KMNIST.
2. Preserve native source train/test provenance, then create the declared project split without overlap.
3. Reuse the current 28×28 amplitude preprocessing exactly.
4. Add fixture tests for label mapping, IDs, disjointness, deterministic splits, scaling, zero padding, and norm handling.
5. Run one two-epoch smoke cell per dataset family. Mark every output `smoke` and exclude it from manuscript aggregation.
6. Freeze `Results/evidence/q1_dataset_provenance.json` before launching scientific runs.

**Exit gate:** Three dataset families validate, smoke cells pass, and the full suite remains green.

**Stop rule:** Use direct, checksummed source files or a single existing library already compatible with the lock. Do not spend more than four focused hours on download abstractions, caching layers, or a registry API.

### Work package C — run the bounded multi-domain comparison and ablation transfer (one day plus compute window 1)

**Outcome:** The primary empirical evidence for the Q1-targeted paper.

1. Run proposed, logistic, compact two-unit MLP, and TTN on all four tasks with seeds `0..4`.
2. Reuse any existing proposed cells only when dataset, ordered IDs, split hash, protocol, revision, and artifact validation all match.
3. Validate the first new cell before unattended continuation.
4. Run the existing pooling arms on Fashion-MNIST `0,6` with seeds `0..2` as the cross-domain ablation check. Reuse the existing five-seed MNIST pooling grid.
5. Aggregate accuracy, balanced accuracy, F1, ROC-AUC where defined, mean, standard deviation, bootstrap 95% CI, paired seed deltas, effect sizes, Wilcoxon/McNemar where their assumptions fit, and Holm-adjusted primary comparisons.
6. Emit `Results/evidence/q1_comparison.json`, `q1_pooling_transfer.json`, and machine-readable table/figure sources. Never consume queued or partial outputs.

```powershell
python -m experiments.run_experiments --task mnist:0,1 --task mnist:3,5 --task fashion_mnist:0,6 --task kmnist:2,3 --configs proposed --seeds 0 1 2 3 4 --samples 400 --epochs 30 --jobs 0 --classical-baselines logistic mlp --quantum-baselines ttn
python -m experiments.q1_fast_track_analysis aggregate --runs-root Results/q1_comparison/runs --output Results/evidence/q1_comparison.json
```

The implementation may choose a less repetitive manifest syntax, but it must preserve the frozen matrix exactly.

**Compute cap:** Eighteen wall-clock hours for the comparison and six additional hours for the Fashion-MNIST pooling transfer.

**Cut order:** reuse validated completed cells; drop Cong/Hur if present; reduce TTN to one task per dataset family while keeping five seeds; reduce the cross-domain pooling transfer uniformly to two seeds and label it exploratory. Do not drop a dataset family, logistic/MLP baselines, uncertainty reporting, or proposed-model seed floor.

**Exit gate:** Four tasks across three families, five proposed/logistic/MLP seeds, the required TTN coverage, complete provenance, and no undisclosed failed cells.

### Work package D — produce bounded local resource and noise evidence (one day plus compute window 2)

**Outcome:** Practicality evidence strong enough for Q1 review without pretending to be hardware execution.

1. Reuse the existing PennyLane/Qiskit conversion and isolated Qiskit environment; do not introduce another circuit definition.
2. Record state-preparation, model-body, readout, and total logical/decomposed/transpiled counts for n=`4,6,8,10` under one declared local target, layout policy, optimization level, and transpiler seed.
3. Compare no-pooling, measurement-style, frozen unitary, coherent extension, and SU(4) arms on the same fake-backend settings. Report unsupported dynamic behavior honestly.
4. Use validation-selected checkpoints from MNIST `3,5` and Fashion-MNIST `0,6`. For three seeds and a fixed 25-example validation subset, evaluate zero noise and a small declared depolarizing/dephasing ladder without retraining.
5. Prove zero-noise agreement with the canonical statevector path. Record sample IDs, weights, noise model, shots if any, backend snapshot, and uncertainty.
6. Rehearse one exact hardware payload locally and hash it. Do not authenticate or submit.
7. Emit `Results/evidence/q1_resources.json`, `q1_pooling_practicality.json`, `q1_noise_validation.json`, and `q1_fake_backend_rehearsal.json`.

```powershell
.venv-qiskit\Scripts\python -m experiments.q1_local_evidence resources --qubits 4 6 8 10
.venv-qiskit\Scripts\python -m experiments.q1_local_evidence pooling
.venv-qiskit\Scripts\python -m experiments.q1_local_evidence noise --tasks mnist:3,5 fashion_mnist:0,6 --seeds 0 1 2 --samples 25
.venv-qiskit\Scripts\python -m experiments.q1_local_evidence rehearse
```

**Compute cap:** Twelve wall-clock hours total.

**Cut order:** reduce shots or validation examples uniformly; reduce the noise ladder to zero plus three nonzero points; report n=10 transpilation as capped/unsupported if it exceeds the declared timeout. Do not hand-author values, reconstruct old splits, or relabel historical `noise_sim.py` output as canonical.

**Conditional real-QPU lane:** After the local payload passes, present its exact backend requirements, circuit count, depth, shots, sample IDs, payload hash, and estimated quota. A real job requires fresh user approval immediately before submission. Failure to obtain access does not invalidate the Q1-targeted package, but the manuscript must remain explicitly simulation/local-transpilation only.

### Work package E — manuscript, reproduction, red team, and package (three and a half days)

**Outcome:** A coherent Q1-targeted submission package whose claims match its evidence.

1. Rewrite the title, abstract, introduction, contributions, results, limitations, and conclusion around §1.
2. Replace image-spatial-locality claims with precise register/index-space language.
3. Present the pooling theorem and null results without claiming information recovery or automatic accuracy advantage.
4. Report full `su(2^10)` honestly and state that it does not provide a polynomial-DLA trainability certificate.
5. Replace historical noise and unsupported hardware claims with the bounded local evidence and correct simulation-only scope.
6. Generate all numerical tables and figures from `q1_*` evidence. Create `Results/evidence/q1_claim_ledger.json` mapping every number to an artifact, pointer, hash, aggregation rule, and caveat.
7. Audit every citation used by the final manuscript. Resolve unsupported novelty wording, missing source metadata, comparison sources, retained preprints, and `ref44` trainability language. A bounded Markdown/JSON ledger is sufficient.
8. Add limitations covering binary tasks, three 28×28 grayscale families, small training subsets, finite seeds, classical simulability, local noise models, and absence of real-QPU evidence if applicable.
9. Build LaTeX three times with halt-on-error and inspect PDF text, tables, equations, and figures.
10. From a clean worktree, verify locks, run the full suite, regenerate bounded summaries, validate the claim ledger, and rebuild the manuscript.
11. Create `Results/evidence/q1_reproduction.json`, `docs/q1-reviewer-checklist.md`, and the submission package manifest. Allow one correction pass, then rerun affected checks.
12. Stop for separate venue selection and submission approval.

**Exit gate:** No unsupported headline claim, all Q1 minimum rows in §5 pass, full tests and clean reproduction pass, LaTeX builds, and no high-severity reviewer finding remains.

## 8. Finite Q1 reviewer checklist

1. Is the canonical architecture unchanged and demonstrably unitary?
2. Are three dataset families and all four predeclared tasks represented without test-driven pair selection?
3. Are split identities, ordered sample IDs, seed counts, uncertainty, and exactly-once test evaluation visible?
4. Are classical and hierarchical quantum baselines evaluated fairly on identical data?
5. Are statistical tests paired where appropriate, assumptions stated, and multiplicity controlled?
6. Does the pooling ablation separate theoretical equivalence, architectural benefit, and empirical accuracy?
7. Are 269 allocated, 78 tape-reaching, and 74 effective parameters distinguished?
8. Is the full DLA negative result reported without a false trainability certificate?
9. Are resource counts separated into state preparation, model body, readout, and total?
10. Does zero-noise output agree with the canonical path, and are local noise models not presented as device validation?
11. Are simulation, fake backend, local transpilation, and real hardware clearly distinguished?
12. Are quantum advantage, spatial locality, scalability, and noise-robustness claims no stronger than the evidence?
13. Does every numerical claim trace to committed evidence?
14. Can a clean checkout run tests, validate evidence, and build the manuscript?
15. Are limitations, negative results, and conditional hardware status explicit?

One high-severity failure blocks submission. Medium findings must be corrected or explicitly accepted with a claim reduction; they do not create an unlimited new workstream.

## 9. Seven-day schedule

| Day | Deliverable | Maximum active time |
|---|---|---:|
| 1 | Task 7 integration, full tests, evidence/manuscript inventory | 6 h |
| 2 | Thin three-family dataset path, tests, smoke cells; launch comparison | 7 h + compute window 1 |
| 3 | Validate/aggregate comparison, fill missing cells, launch local evidence | 6 h + compute window 2 |
| 4 | Statistics, resource/noise inspection, freeze tables and figures | 6 h |
| 5 | Rewrite methods, theory positioning, experiments, and results | 7 h |
| 6 | Rewrite introduction/discussion/limitations; citation audit; PDF build | 7 h |
| 7 | Clean reproduction, Q1 reviewer checklist, one correction pass, package | 7 h |

The schedule controls scope; it does not promise journal review or acceptance. If a task overruns, use its declared cut order rather than restoring the former open-ended twelve-task program.

### Execution record

- **Day 1 complete (2026-08-27):** Task 7 integrated; 19 evidence artifacts and 72 manuscript findings inventoried; 570 tests passed with two documented Windows symlink skips.
- **Day 2 complete (2026-08-27):** three checksummed dataset families and four tasks frozen; source IDs included in split identity; isolated MNIST/Fashion-MNIST/KMNIST smoke cells completed and resumed cleanly; 585 tests passed with the same two documented skips.
- **Days 3–4 complete (2026-08-29):** corrected comparison matrix (4 tasks × 4 arms × 5 seeds = 80/80 cells) and Fashion-MNIST pooling transfer (5 arms × 3 seeds = 15/15 cells) completed with exact 666/400/100/166 manifests, zero failures, and paired aggregate statistics. Resource, pooling-practicality, six-record noise, provenance, and fake-backend evidence all validate.
- **Days 5–6 complete (2026-08-29):** manuscript claims, tables, and the Q1 comparison figure were regenerated from canonical evidence; legacy single-seed plots were removed from the submission path; the bibliography now has 34 cited keys and 34 cited entries with no missing or uncited keys; the claim ledger was regenerated after the correction pass.
- **Day 7 complete (2026-08-29):** full suite passed (616 passed, 2 documented Windows symlink skips); clean training and isolated Qiskit environments pass dependency checks; static, archive, finite-value, checksum, secret, and scope audits pass. The final release commit is limited to task-owned evidence/manuscript files and excludes local credentials, editor state, scratch logs, and generated LaTeX intermediates.

## 10. Definition of done

The Q1 fast track is complete when:

- Task 7 is integrated once and all relevant tests pass;
- frozen architecture and protocol regressions pass unchanged;
- MNIST, Fashion-MNIST, and KMNIST provenance and smoke gates pass;
- four predeclared tasks have five-seed proposed, logistic, and matched-MLP results;
- TTN covers at least one task per dataset family with the same seed floor;
- statistics include uncertainty, paired effects where possible, and multiplicity control;
- existing MNIST pooling evidence and the Fashion-MNIST transfer check are reported with their seed counts;
- DLA, gradients, expressibility, generalization, and simulability evidence is used accurately or omitted;
- n=`4,6,8,10` resource evidence, fake-backend pooling evidence, and the clean two-task noise analysis validate;
- real-QPU status is either exact immutable evidence or an explicit simulation-only limitation;
- every numerical manuscript claim appears in the claim ledger;
- the final citation audit, full suite, clean reproduction, and LaTeX build pass;
- the Q1 reviewer checklist has no unresolved high-severity finding;
- the package manifest binds sources, PDF, figures, evidence, locks, and final commit.

Effective-dimension tooling, a full general ablation grid, n=`12,14` exact transpilation, extra datasets/baselines, and a real-QPU point remain optional strengthening work. Their absence does not remove the Q1 target because the retained gates preserve breadth, comparison quality, analysis depth, and reproducibility.
