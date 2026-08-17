# FQCNN Roadmap Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the remaining FQCNN roadmap in dependency order, first removing local protocol blockers and safely launching reproducible long-running work, then completing empirical, hardware, manuscript, and reproduction gates.

**Architecture:** Preserve the current branch and establish an accurate dashboard before changing experimental behavior. Repair baseline selection and n=10 ablation configuration under TDD, keep Qiskit hardware tooling in a separate environment, and require every campaign to use immutable manifests and canonical evidence artifacts. Later analyses consume only validated upstream artifacts.

**Tech Stack:** Python 3.9.13, PennyLane 0.38.0, NumPy 1.26.4, scikit-learn 1.6.1, pytest 7.4.4; separate Python 3.11 environment for Qiskit 1.x, Aer, IBM Runtime, fake backends, and optional PennyLane-Qiskit integration.

## Global Constraints

- Keep `requirements-lock.txt` unchanged as the frozen training environment.
- Validation performs model selection; test data are evaluated exactly once after selection.
- General non-encoding ablations use the n=10 headline geometry.
- `enc_feature_map` may retain smaller geometry only with an explicit machine-readable exception.
- Do not submit a real-QPU job without explicit approval immediately before submission.
- Do not call queued, running, partial, failed, historical-split, or single-seed work complete evidence.
- Do not run training queue A and hardware queue B concurrently on the same machine.
- Stop an affected workstream on failed protocol tests, unknown provenance, over-budget projection, required training-environment mutation, or unapproved external action.
- Manuscript evidence must originate from clean manifests and schema-validated canonical artifacts.
- Use TDD for behavioral changes: failing test, minimal implementation, focused tests, full suite, smoke run, artifact inspection.
- Commit only intentional files. Never stage `.claude/settings.local.json`, LaTeX intermediates, raw stdout logs, or unclassified PDFs.

## File and Responsibility Map

### Immediate tranche

- `scripts/audit_repository_state.py`: read-only repository state and untracked-file classification.
- `tests/test_repository_audit.py`: repository audit classification contract.
- `STATUS.md`: authoritative milestone and evidence dashboard.
- `README.md`: public reproducibility and environment guidance.
- `tests/test_documentation_contract.py`: guards reproducibility-sensitive documentation statements.
- `baselines/classical_cnn.py`: train/validation/test-safe classical model selection.
- `baselines/quantum_baselines.py`: validation-selected quantum baselines and n=10 active-wire schedules.
- `experiments/run_experiments.py`: split forwarding, corrected ablation registry, and campaign entry point.
- `QCNN/utils/run_artifacts.py`: baseline selection and exactly-once test provenance.
- `tests/test_baseline_protocol.py`: leakage and exactly-once regression tests.
- `tests/test_ablation_registry.py`: n=10 geometry and one-factor registry contract.
- `experiments/estimate_cost.py`: approval-grade JSON cost estimates that fail on unmeasurable cells.
- `requirements-qiskit.in`, `requirements-qiskit-lock.txt`: isolated hardware dependency specification and lock.
- `scripts/check_qiskit_environment.py`: isolated-environment provenance and transpilation smoke check.
- `tests/test_qiskit_environment_contract.py`: environment separation and artifact schema contract.
- `experiments/campaign.py`: immutable campaign plan, validation, launch, and status interface.
- `tests/test_campaign_manifest.py`: campaign manifest and launch-gate contract.

### Later empirical tranche

- `QCNN/utils/dataset_registry.py`: dataset identity, version, class mapping, and adapter lookup.
- `QCNN/utils/dataset_loader.py`: adapter implementations and deterministic split loading.
- `experiments/dataset_audit.py`: provenance and manifest validation.
- `experiments/baseline_analysis.py`: T3 aggregation, paired statistics, calibration, and learning curves.
- `QCNN/utils/effective_dimension.py`: Fisher/effective-dimension primitives.
- `QCNN/utils/inductive_bias.py`: translation sensitivity and entropy-versus-cut primitives.
- `experiments/model_analysis.py`: analysis orchestration and canonical artifacts.
- `experiments/ablation_analysis.py`: one-factor grid aggregation and paired effects.

### Hardware, manuscript, and reproduction tranche

- `QCNN/utils/qiskit_export.py`: canonical circuit conversion and equivalence checks.
- `QCNN/utils/resource_metrics.py`: logical, decomposed, and transpiled resource metrics.
- `experiments/resource_analysis.py`: n=4..14 resource and scaling campaign.
- `experiments/pooling_hardware_analysis.py`: hardware-cost comparison across pooling arms.
- `experiments/noise_analysis.py`: clean-manifest noise experiments.
- `experiments/fake_backend_run.py`: exact fake-backend payload rehearsal.
- `experiments/hardware_run.py`: explicit rehearse/submit/collect workflow.
- `scripts/audit_citations.py`: sentence-level citation audit.
- `scripts/build_manuscript_assets.py`: tables and figures from canonical evidence.
- `scripts/validate_manuscript_claims.py`: claim-to-evidence traceability.
- `scripts/validate_evidence.py`: schema and cross-artifact consistency validation.
- `scripts/reproduce_quick.py`: deterministic clean-checkout reproduction subset.
- `scripts/red_team_audit.py`: definition-of-done and reviewer-question audit.

---

## Tranche 1: Preserve, Repair, and Launch

### Task 1: Audit and preserve the current branch

**Recommendations:** 1

**Files:**
- Create: `scripts/audit_repository_state.py`
- Create: `tests/test_repository_audit.py`
- Create after approved push: `Results/evidence/repository_preservation.json`
- Modify after approved push: `STATUS.md`

**Interfaces:**
- Consumes: Git repository at the current working directory.
- Produces: `audit_repository(root: Path) -> dict` with branch, upstream, HEAD, remotes, divergence, tracked state, and classified untracked paths.

- [ ] **Step 1: Write classification tests**

```python
from pathlib import Path

from scripts.audit_repository_state import classify_untracked


def test_local_configuration_is_never_evidence():
    assert classify_untracked(Path(".claude/settings.local.json")) == "local_config"


def test_generated_latex_files_are_never_evidence():
    for name in ("fqcnn.aux", "fqcnn.log", "fqcnn.pdf"):
        assert classify_untracked(Path(name)) == "generated_output"


def test_raw_stdout_is_local_log():
    assert classify_untracked(Path("Results/phase3_expr_stdout.txt")) == "raw_log"
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
python -m pytest tests/test_repository_audit.py -q
```

Expected: FAIL because `scripts.audit_repository_state` does not exist.

- [ ] **Step 3: Implement the read-only audit module**

Implement exact categories:

```python
LOCAL_CONFIG = {Path(".claude/settings.local.json")}
GENERATED_SUFFIXES = {".aux", ".log", ".pdf"}
RAW_LOG_SUFFIXES = ("_stdout.txt", "_log.txt")


def classify_untracked(path: Path) -> str:
    normalized = Path(path.as_posix())
    if normalized in LOCAL_CONFIG:
        return "local_config"
    if normalized.suffix.lower() in GENERATED_SUFFIXES:
        return "generated_output"
    if normalized.name.endswith(RAW_LOG_SUFFIXES):
        return "raw_log"
    return "unclassified"
```

Use `subprocess.run(..., check=True, text=True, capture_output=True)` for read-only Git commands. The CLI writes JSON only when `--output` is passed.

- [ ] **Step 4: Run focused tests and generate the audit**

```bash
python -m pytest tests/test_repository_audit.py -q
python scripts/audit_repository_state.py --output Results/evidence/repository_preservation.pending.json
```

Expected: tests PASS; pending audit identifies 37 commits ahead at HEAD `05b449d` unless the branch changed during execution.

- [ ] **Step 5: Review the exact push target**

Run:

```bash
git status --short --branch
git remote -v
git branch -vv
git rev-list --count origin/plan/fqcnn-q1-upgrade..HEAD
git log --oneline origin/plan/fqcnn-q1-upgrade..HEAD
git diff --check origin/plan/fqcnn-q1-upgrade..HEAD
git diff --stat origin/plan/fqcnn-q1-upgrade..HEAD
```

Present the remote URL, source branch, destination branch, local SHA, remote SHA, divergence, and untracked classification. Stop for explicit push approval.

- [ ] **Step 6: Push only after approval**

```bash
git push origin plan/fqcnn-q1-upgrade
git fetch origin
git rev-list --left-right --count origin/plan/fqcnn-q1-upgrade...HEAD
git ls-remote origin refs/heads/plan/fqcnn-q1-upgrade
```

Expected divergence: `0 0`.

- [ ] **Step 7: Finalize preservation evidence and commit**

Regenerate `Results/evidence/repository_preservation.json` with the pushed SHA and remove the pending artifact.

```bash
git add scripts/audit_repository_state.py tests/test_repository_audit.py Results/evidence/repository_preservation.json STATUS.md
git commit -m "chore: record repository preservation state"
```

Do not stage any of the nine local/generated files.

### Task 2: Repair STATUS and README contracts

**Recommendations:** 2

**Files:**
- Modify: `STATUS.md`
- Modify: `README.md`
- Create: `tests/test_documentation_contract.py`

**Interfaces:**
- Consumes: canonical JSON artifacts under `Results/evidence/` and the clean split manifest.
- Produces: documentation whose numerical and protocol statements map to canonical evidence.

- [ ] **Step 1: Write failing documentation contract tests**

```python
from pathlib import Path

README = Path("README.md").read_text(encoding="utf-8")
STATUS = Path("STATUS.md").read_text(encoding="utf-8")


def test_readme_names_frozen_python_version():
    assert "Python 3.9.13" in README
    assert "Python 3.14" not in README


def test_readme_labels_historical_noise():
    assert "historical split" in README.lower()
    assert "not manuscript evidence" in README.lower()


def test_status_records_completed_pooling_campaign():
    assert "75" in STATUS
    assert "t5_pooling_arms.json" in STATUS


def test_status_no_longer_calls_m1_the_active_gate():
    assert "M1 active gate" not in STATUS
```

- [ ] **Step 2: Verify tests fail against stale documentation**

```bash
python -m pytest tests/test_documentation_contract.py -q
```

Expected: FAIL on Python version, historical noise qualification, or stale gate language.

- [ ] **Step 3: Update `STATUS.md` from canonical evidence**

Make the minimum edits required to record:

- E1, E2, E3, and E4 complete.
- E5 blocked on isolated Qiskit tooling.
- M3 complete for DLA, gradient variance, expressibility, generalization bound, and simulability.
- Full DLA as a negative result for the planned certificate.
- E3 as 75 complete cells.
- Baseline leakage, n=10 registry, and isolated Qiskit as the active queue.
- State labels `pending`, `running`, `failed`, `complete`.
- Current full-suite count only after Task 2 Step 5 runs.

- [ ] **Step 4: Update reproducibility-sensitive README statements**

Make the minimum edits required to:

- State Python 3.9.13 for `requirements-lock.txt`.
- Qualify old standalone accuracy claims and the 98.29% result as single-seed until distributions exist.
- State that current baseline numbers are blocked from manuscript use pending protocol repair.
- State that `noise_sim.py` uses a historical split and is not manuscript evidence.
- State that hardware dependencies belong in the separate Qiskit environment.
- Avoid claiming full reproduction until M11 passes.

- [ ] **Step 5: Run documentation tests and the full suite**

```bash
python -m pytest tests/test_documentation_contract.py -q
python -m pytest tests/ -q
git diff --check
git diff -- STATUS.md README.md tests/test_documentation_contract.py
```

Update the test-count statement in `STATUS.md` to the observed full-suite result, then rerun the documentation test.

- [ ] **Step 6: Commit documentation repair**

```bash
git add STATUS.md README.md tests/test_documentation_contract.py
git commit -m "docs: refresh project evidence status"
```

### Task 3: Remove baseline test leakage

**Recommendations:** 3

**Files:**
- Modify: `baselines/classical_cnn.py`
- Modify: `baselines/quantum_baselines.py`
- Modify: `experiments/run_experiments.py`
- Modify: `QCNN/utils/run_artifacts.py`
- Create: `tests/test_baseline_protocol.py`
- Modify: `tests/test_protocol.py`
- Modify: `tests/test_run_artifacts.py`
- Create: `Results/evidence/baseline_protocol_audit.json`

**Interfaces:**
- Produces:

```python
run_classical_baselines(
    *, X_train, y_train, X_val, y_val, X_test, y_test,
    seed=42, target_params=None,
) -> dict

run_quantum_baselines(
    *, X_train, y_train, X_val, y_val, X_test, y_test,
    seed=42, n_qubits=10, n_epochs=30,
    learning_rate=0.02, use_bce=True,
) -> dict
```

Each model result contains:

```python
{
    "metrics": {...},
    "selection": {
        "criterion": "validation_loss",
        "best_epoch": 7,
        "best_value": 0.123,
    },
    "test_evaluations": 1,
}
```

- [ ] **Step 1: Write signature and forwarding tests**

Test that positional four-split calls raise `TypeError`, `X_val/y_val` are required, and `experiments.run_experiments` forwards all three splits by keyword.

- [ ] **Step 2: Write selection-independence tests**

Use paired fixtures where only `y_test` changes. Assert the selected hyperparameter, best epoch, and saved parameters remain identical. Change `y_val` in a second fixture and assert selection can change.

- [ ] **Step 3: Write exactly-once test-evaluation tests**

Patch the final evaluator and assert its call count is one. For quantum training, patch validation evaluation separately and assert it runs once per epoch.

- [ ] **Step 4: Verify the red tests fail**

```bash
python -m pytest tests/test_baseline_protocol.py -q
```

Expected: FAIL because current runners omit explicit validation inputs and the CNN uses test data as validation.

- [ ] **Step 5: Refactor classical baselines minimally**

- Logistic regression: fit on training data; select `C` only on validation when a grid is requested; evaluate test once.
- MLP: use `warm_start=True`, train one epoch at a time, select an explicit validation checkpoint, restore it, evaluate test once.
- TensorFlow CNN: accept explicit validation/test arrays; pass `validation_data=(X_val, y_val)`; restore the best validation checkpoint; evaluate test once.

- [ ] **Step 6: Refactor quantum baselines minimally**

Train only on `X_train/y_train`, evaluate validation each epoch, copy the best parameter vector, restore it, and pass the test evaluation through `TestEvaluationGuard` once.

- [ ] **Step 7: Update runner and artifact persistence**

Forward all three splits by keyword. Persist `selection.json`, `metrics.json`, and `predictions.npz` under:

```text
Results/runs/<dataset>/baseline_<name>/seed_<seed>/
```

Include split ID, original test sample IDs, criterion, best epoch/hyperparameters, and test-evaluation count.

- [ ] **Step 8: Run focused and full tests**

```bash
python -m pytest tests/test_baseline_protocol.py -q
python -m pytest tests/test_protocol.py tests/test_run_artifacts.py -q
python -m pytest tests/ -q
```

- [ ] **Step 9: Generate the protocol audit artifact**

Write `Results/evidence/baseline_protocol_audit.json` with runner signatures, selection criterion, mutation-test outcome, test-evaluation count, split IDs, and artifact schema version.

- [ ] **Step 10: Commit baseline protocol repair**

```bash
git add baselines/classical_cnn.py baselines/quantum_baselines.py experiments/run_experiments.py QCNN/utils/run_artifacts.py tests/test_baseline_protocol.py tests/test_protocol.py tests/test_run_artifacts.py Results/evidence/baseline_protocol_audit.json
git commit -m "fix: isolate baseline validation and test evaluation"
```

### Task 4: Move general ablations to n=10

**Recommendations:** 4

**Files:**
- Modify: `experiments/run_experiments.py`
- Modify: `experiments/estimate_cost.py`
- Create: `tests/test_ablation_registry.py`
- Create: `Results/evidence/ablation_registry_audit.json`
- Create: `Results/evidence/campaign_cost_estimate.json`

**Interfaces:**
- Produces: `resolve_ablation(name: str) -> dict` with `image_size`, `n_qubits`, factor metadata, and optional `geometry_exception`.

- [ ] **Step 1: Write n=10 registry tests**

Assert `proposed`, `pool_none`, `pool_measurement`, `ent_one_diagonal`, `ent_none`, and `kernel_ry` resolve to image size 28 and `freeze.HEADLINE_N_QUBITS == 10`.

- [ ] **Step 2: Write one-factor and exception tests**

Assert each arm differs from `proposed` in exactly one intended factor. Assert `enc_feature_map` is the sole geometry exception and its exception text is emitted in metadata.

- [ ] **Step 3: Write cost-estimator failure tests**

Assert an unmeasurable configuration causes a nonzero approval verdict and JSON output records the failed configuration instead of silently excluding it.

- [ ] **Step 4: Verify red tests fail**

```bash
python -m pytest tests/test_ablation_registry.py tests/test_cost_estimator.py -q
```

- [ ] **Step 5: Implement the registry correction**

Set the six general amplitude configurations to `image_size=28`. Add:

```python
ABLATION_GEOMETRY_EXCEPTIONS = {
    "enc_feature_map": "feature-map simulation geometry is inherently different",
}
```

Include the exception in run metadata.

- [ ] **Step 6: Add JSON cost output and strict approval semantics**

Add `--output-json`. Return failure when any requested cell is unmeasurable. Record requested cells, measurable cells, failures, runtime projection, worker count, and budget verdict.

- [ ] **Step 7: Run focused and full tests**

```bash
python -m pytest tests/test_ablation_registry.py tests/test_headline_identity.py tests/test_freeze_architecture.py -q
python -m pytest tests/test_cost_estimator.py tests/test_resume_and_parallelism.py -q
python -m pytest tests/ -q
```

- [ ] **Step 8: Recompute the n=10 campaign cost**

```bash
python -m experiments.estimate_cost --datasets 0,1 3,5 4,9 5,8 --configs proposed pool_none pool_measurement ent_one_diagonal ent_none kernel_ry enc_feature_map --seeds 0 1 2 3 4 5 6 7 8 9 --samples 400 --epochs 30 --jobs 0 --output-json Results/evidence/campaign_cost_estimate.json
```

Stop if the estimate exceeds the approved budget or contains unmeasurable cells.

- [ ] **Step 9: Generate and commit registry evidence**

Write `ablation_registry_audit.json` from the resolved registry and one-factor checks.

```bash
git add experiments/run_experiments.py experiments/estimate_cost.py tests/test_ablation_registry.py tests/test_cost_estimator.py Results/evidence/ablation_registry_audit.json Results/evidence/campaign_cost_estimate.json
git commit -m "fix: align ablations with headline geometry"
```

### Task 5: Add a tested n=10 quantum-baseline schedule

**Recommendations:** prerequisite for 7

**Files:**
- Modify: `baselines/quantum_baselines.py`
- Create: `tests/test_quantum_baselines.py`

**Interfaces:**
- Produces: `active_wire_schedule(n_qubits: int) -> list[dict]`.
- n=10 schedule: `10 -> 5 -> 2 -> 1`; every stage records paired wires and the explicit treatment of an unpaired wire.

- [ ] **Step 1: Write schedule tests for n=4, 6, 8, and 10**

Assert each schedule terminates at one readout wire, consumes deterministic parameter counts, and records unpaired-wire handling.

- [ ] **Step 2: Write n=10 smoke and batch-equivalence tests**

Run one training epoch on a deterministic tiny fixture. Assert batched and sequential predictions agree and the protocol guard permits one test evaluation.

- [ ] **Step 3: Verify current power-of-two guard fails the tests**

```bash
python -m pytest tests/test_quantum_baselines.py -q
```

Expected: FAIL at `_check_pow2(10)`.

- [ ] **Step 4: Implement explicit active-wire reduction**

Replace the blanket power-of-two rejection with schedule construction. At `5 -> 2`, explicitly retire the unpaired fifth wire in metadata rather than silently dropping it. Do not change the proposed FQCNN schedule.

- [ ] **Step 5: Run focused and full tests**

```bash
python -m pytest tests/test_quantum_baselines.py tests/test_baseline_protocol.py -q
python -m pytest tests/ -q
```

- [ ] **Step 6: Commit n=10 baseline support**

```bash
git add baselines/quantum_baselines.py tests/test_quantum_baselines.py
git commit -m "feat: support headline-size quantum baselines"
```

If the schedule cannot be validated, stop baseline-enabled launch and use `--no-baselines`; do not revert general ablations to n=8.

### Task 6: Create the isolated Qiskit environment

**Recommendations:** 5

**Files:**
- Create: `requirements-qiskit.in`
- Create: `requirements-qiskit-lock.txt`
- Create: `scripts/check_qiskit_environment.py`
- Create: `tests/test_qiskit_environment_contract.py`
- Create: `Results/evidence/qiskit_environment.json`
- Create: `Results/evidence/qiskit_transpile_smoke.json`
- Modify: `README.md`
- Modify: `RUN_GUIDE.md`

**Interfaces:**
- Produces: environment JSON with Python/package versions, lock SHA-256, import status, fake-backend availability, and transpilation smoke metrics.

- [ ] **Step 1: Write environment-separation tests**

Assert `requirements-lock.txt` contains no Qiskit packages, the hardware lock is separate, and the environment artifact schema requires Python, Qiskit, Aer, Runtime, platform, lock hash, and fake-backend fields.

- [ ] **Step 2: Write transpilation contract tests**

Assert smoke output uses the canonical circuit source, records seed and optimization level, and contains only `rz`, `sx`, `x`, `ecr`, measurement, barrier, or reset operations after transpilation.

- [ ] **Step 3: Verify tests fail before files exist**

```bash
python -m pytest tests/test_qiskit_environment_contract.py -q
```

- [ ] **Step 4: Create the input dependency file**

Declare compatible major constraints, not guessed final pins:

```text
qiskit>=1,<2
qiskit-aer>=0.14
qiskit-ibm-runtime>=0.23
```

Include `pennylane-qiskit` only if the current hardware runner still requires the plugin after inspection.

- [ ] **Step 5: Create and resolve `.venv-qiskit`**

```powershell
py -3.11 -m venv .venv-qiskit
& ".venv-qiskit\Scripts\python.exe" -m pip install --upgrade pip
& ".venv-qiskit\Scripts\python.exe" -m pip install -r requirements-qiskit.in
& ".venv-qiskit\Scripts\python.exe" -m pip freeze | Out-File -Encoding utf8 requirements-qiskit-lock.txt
```

If resolution fails, adjust only `requirements-qiskit.in`; never touch `requirements-lock.txt`.

- [ ] **Step 6: Implement environment and transpilation smoke script**

Use a canonical n=4 or n=6 scaling-family circuit. Record versions, lock hash, basis counts, depth, width, circuit hash, seed transpiler, optimization level, and fake-backend class.

- [ ] **Step 7: Run isolated smoke and frozen-suite verification**

```powershell
& ".venv-qiskit\Scripts\python.exe" scripts/check_qiskit_environment.py
python -m pytest tests/test_qiskit_environment_contract.py -q
python -m pytest tests/ -q
```

- [ ] **Step 8: Update setup documentation and commit**

```bash
git add requirements-qiskit.in requirements-qiskit-lock.txt scripts/check_qiskit_environment.py tests/test_qiskit_environment_contract.py Results/evidence/qiskit_environment.json Results/evidence/qiskit_transpile_smoke.json README.md RUN_GUIDE.md
git commit -m "build: isolate qiskit hardware environment"
```

### Task 7: Add immutable campaign planning and launch the first safe run

**Recommendations:** advances 6 and 7

**Files:**
- Create: `experiments/campaign.py`
- Create: `tests/test_campaign_manifest.py`
- Create at runtime: `Results/campaigns/baseline_mnist_n10_v1/manifest.json`
- Create at runtime: `Results/campaigns/baseline_mnist_n10_v1/cost_estimate.json`
- Create at runtime: `Results/campaigns/baseline_mnist_n10_v1/launch.json`
- Create at runtime: `Results/campaigns/baseline_mnist_n10_v1/status.json`
- Create at runtime: `Results/campaigns/baseline_mnist_n10_v1/failures.json`

**Interfaces:**
- CLI: `plan`, `validate`, `launch`, `status`.
- Manifest fields: command, git SHA, dirty flag, lock hash, datasets, classes, configs, seeds, split policy, samples, epochs, jobs, expected cells, resource estimate, output roots, artifacts, and approval records.

- [ ] **Step 1: Write manifest schema tests**

Assert missing git SHA, lock hash, split policy, expected cell count, cost estimate, or dirty-state field fails validation.

- [ ] **Step 2: Write launch-gate tests**

Assert launch refuses a dirty tree, failed full-suite record, over-budget estimate, occupied queue, invalid manifest, or baseline-enabled n=10 run without a validated baseline schedule.

- [ ] **Step 3: Verify red tests fail**

```bash
python -m pytest tests/test_campaign_manifest.py -q
```

- [ ] **Step 4: Implement plan and validate commands**

`plan` writes an immutable manifest. `validate` checks repository, lock, cost, queue, output uniqueness, and expected artifact paths without starting work.

- [ ] **Step 5: Implement launch and status commands**

`launch` calls the existing resumable scheduler only after validation. `status` reports `pending`, `running`, `failed`, or `complete` and verifies expected cells.

- [ ] **Step 6: Run focused and full tests**

```bash
python -m pytest tests/test_campaign_manifest.py tests/test_resume_and_parallelism.py -q
python -m pytest tests/ -q
```

- [ ] **Step 7: Plan the first campaign**

```bash
python -m experiments.campaign plan --campaign baseline_mnist_n10_v1
python -m experiments.campaign validate --campaign baseline_mnist_n10_v1
```

Campaign command:

```bash
python -m experiments.run_experiments --datasets 0,1 3,5 4,9 5,8 --configs proposed --seeds 0 1 2 3 4 5 6 7 8 9 --samples 400 --epochs 30 --jobs 0 --no-baselines
```

Use `--no-baselines` for this first launch even if Task 5 passes; baseline evidence belongs to its separately reviewed campaign.

- [ ] **Step 8: Launch only after all gates pass**

```bash
python -m experiments.campaign launch --campaign baseline_mnist_n10_v1
python -m experiments.campaign status --campaign baseline_mnist_n10_v1
```

Report the result as launched/running. Inspect the first completed cell for n=10, matching split ID, expected sample IDs, one test evaluation, weights, predictions, status, and no hidden failure.

- [ ] **Step 9: Commit campaign tooling, not volatile run progress**

```bash
git add experiments/campaign.py tests/test_campaign_manifest.py Results/campaigns/baseline_mnist_n10_v1/manifest.json Results/campaigns/baseline_mnist_n10_v1/cost_estimate.json Results/campaigns/baseline_mnist_n10_v1/launch.json
git commit -m "feat: add validated experiment campaigns"
```

---

## Tranche 2: Datasets and Baseline Evidence

### Task 8: Add deterministic harder-dataset adapters

**Recommendations:** 6

**Files:**
- Create: `QCNN/utils/dataset_registry.py`
- Modify: `QCNN/utils/dataset_loader.py`
- Modify: `experiments/run_experiments.py`
- Create: `experiments/dataset_audit.py`
- Create: `tests/test_dataset_registry.py`
- Create: `tests/test_dataset_adapters.py`
- Modify: `tests/test_splits.py`
- Modify: `DATASET_README.md`
- Create: `Results/evidence/dataset_provenance.json`
- Create: `Results/evidence/t1_dataset_table.json`

**Interfaces:**
- `get_dataset_adapter(dataset_id: str) -> DatasetAdapter`.
- Adapter output includes arrays, original IDs, dataset version, class mapping, preprocessing, and deterministic 60/15/25 manifest.

- [ ] Write registry tests for MNIST, Fashion-MNIST, KMNIST, and the approved MedMNIST subdataset.
- [ ] Write adapter tests for deterministic IDs, disjoint splits, checksums/version, class mapping, and preprocessing metadata.
- [ ] Run tests and confirm adapters are missing.
- [ ] Implement the registry and adapters without overloading `--datasets`; add `--dataset` and `--classes` axes.
- [ ] Run one two-epoch, no-baseline smoke cell per adapter and label outputs `smoke`.
- [ ] Run focused and full tests:

```bash
python -m pytest tests/test_dataset_registry.py tests/test_dataset_adapters.py tests/test_splits.py -q
python -m pytest tests/ -q
```

- [ ] Generate provenance and T1 artifacts only after at least three domain manifests validate.
- [ ] Commit:

```bash
git add QCNN/utils/dataset_registry.py QCNN/utils/dataset_loader.py experiments/run_experiments.py experiments/dataset_audit.py tests/test_dataset_registry.py tests/test_dataset_adapters.py tests/test_splits.py DATASET_README.md Results/evidence/dataset_provenance.json Results/evidence/t1_dataset_table.json
git commit -m "feat: add deterministic benchmark datasets"
```

### Task 9: Run the clean baseline and statistical campaign

**Recommendations:** 7

**Files:**
- Create: `experiments/baseline_analysis.py`
- Create: `tests/test_campaign_aggregation.py`
- Modify: `experiments/statistics.py`
- Create: `Results/evidence/t3_baselines.json`
- Create: `Results/evidence/f_f_learning_curves.json`
- Create: `Results/evidence/f_f_calibration.json`

**Interfaces:**
- Aggregator consumes completed per-run predictions, selection records, split IDs, and failure manifests.
- Produces paired bootstrap CIs, exact McNemar, seed-level Wilcoxon, Holm-adjusted p-values, calibration, and learning curves.

- [ ] Write tests that reject failed cells, split mismatches, missing sample IDs, fewer than five seeds, and repeated test evaluations.
- [ ] Write deterministic statistic fixtures with known McNemar, Wilcoxon, bootstrap, and Holm outputs.
- [ ] Verify red tests fail because the aggregator is missing.
- [ ] Implement aggregation for proposed FQCNN, logistic regression, matched MLP, optional CNN, Cong, Hur, TTN, random-frozen, and encoding-only controls.
- [ ] Plan and validate the campaign with at least five seeds per approved domain.
- [ ] Run the corrected cost gate; stop if over budget.
- [ ] Launch through `experiments.campaign`; report running until all cells validate.
- [ ] Generate T3 and F-F artifacts only when `failures.json` is empty.
- [ ] Run:

```bash
python -m pytest tests/test_campaign_aggregation.py tests/test_statistics.py tests/test_baseline_protocol.py tests/test_quantum_baselines.py -q
python -m pytest tests/ -q
```

- [ ] Commit analysis code and completed canonical artifacts:

```bash
git add experiments/baseline_analysis.py experiments/statistics.py tests/test_campaign_aggregation.py Results/evidence/t3_baselines.json Results/evidence/f_f_learning_curves.json Results/evidence/f_f_calibration.json
git commit -m "exp: complete clean baseline study"
```

---

## Tranche 3: Model Analysis and Ablations

### Task 10: Complete effective dimension and inductive bias

**Recommendations:** 8

**Files:**
- Create: `QCNN/utils/effective_dimension.py`
- Create: `QCNN/utils/inductive_bias.py`
- Modify: `QCNN/utils/capacity.py`
- Modify: `experiments/model_analysis.py`
- Create: `tests/test_effective_dimension.py`
- Create: `tests/test_inductive_bias.py`
- Create: `Results/evidence/t6_effective_dimension.json`
- Create: `Results/evidence/t6_inductive_bias.json`

- [ ] Write analytic Fisher-rank and redundant-parameter tests.
- [ ] Write translation-invariant and deliberately translation-sensitive fixtures.
- [ ] Write entropy-versus-cut tests on product and Bell-state fixtures.
- [ ] Verify missing orchestration fails.
- [ ] Implement primitives and add `effective_dimension` and `inductive_bias` parser choices.
- [ ] Consume only validation-clean controls from T3.
- [ ] Run:

```bash
python -m pytest tests/test_effective_dimension.py tests/test_inductive_bias.py -q
python -m experiments.model_analysis --experiment effective_dimension
python -m experiments.model_analysis --experiment inductive_bias
python -m pytest tests/ -q
```

- [ ] Inspect artifacts for control sensitivity, split provenance, seed count, and caveats.
- [ ] Commit:

```bash
git add QCNN/utils/effective_dimension.py QCNN/utils/inductive_bias.py QCNN/utils/capacity.py experiments/model_analysis.py tests/test_effective_dimension.py tests/test_inductive_bias.py Results/evidence/t6_effective_dimension.json Results/evidence/t6_inductive_bias.json
git commit -m "exp: complete capacity and bias analyses"
```

### Task 11: Complete the full one-factor ablation grid

**Recommendations:** 9

**Files:**
- Create: `experiments/ablation_analysis.py`
- Create: `tests/test_ablation_analysis.py`
- Create: `Results/evidence/t4_ablation_grid.json`

- [ ] Write tests rejecting n=8 general rows, undeclared geometry substitutions, failed cells, split mismatches, missing prediction IDs, and insufficient seeds.
- [ ] Write paired-effect fixture tests for deltas, CIs, paired tests, and Holm adjustment.
- [ ] Verify red tests fail.
- [ ] Implement aggregation with `proposed` as reference and explicit `enc_feature_map` geometry metadata.
- [ ] Validate datasets, cost, queue availability, one-factor registry, and expected cell count.
- [ ] Launch the campaign and report running until every expected cell validates.
- [ ] Generate T4 only with empty failures and approved seed/domain counts.
- [ ] Run focused and full tests.
- [ ] Commit:

```bash
git add experiments/ablation_analysis.py tests/test_ablation_analysis.py Results/evidence/t4_ablation_grid.json
git commit -m "exp: complete one-factor ablation grid"
```

### Task 12: Measure pooling hardware practicality

**Recommendations:** 10

**Files:**
- Create: `experiments/pooling_hardware_analysis.py`
- Create: `tests/test_pooling_hardware_analysis.py`
- Create: `Results/evidence/t5_pooling_hardware.json`

- [ ] Write tests that require identical backend target, layout policy, optimization level, and transpiler seed across all pooling arms.
- [ ] Write metric-schema tests for pre/post-transpile one-qubit gates, ECR/two-qubit gates, depth, measurement, reset, conditions, width, and estimated duration.
- [ ] Implement comparison for no pooling, measurement, frozen unitary, coherent extension, and SU(4).
- [ ] Run in `.venv-qiskit`; inspect equivalence labels and cost outputs.
- [ ] Run full tests in the frozen environment.
- [ ] Commit code and artifact.

---

## Tranche 4: Resources, Noise, and Hardware

### Task 13: Build resource and scaling evidence

**Recommendations:** 11

**Files:**
- Create: `QCNN/utils/qiskit_export.py`
- Create: `QCNN/utils/resource_metrics.py`
- Create: `experiments/resource_analysis.py`
- Create: `tests/test_qiskit_export.py`
- Create: `tests/test_resource_metrics.py`
- Create: `tests/test_resource_analysis.py`
- Create: `Results/evidence/t2_resource_table.json`
- Create: `Results/evidence/f_d_scaling_family.json`
- Create: `Results/evidence/t6_scaling_join.json`

- [ ] Write small-circuit PennyLane/Qiskit output-equivalence tests.
- [ ] Write additive state-preparation/model-body/total counting tests.
- [ ] Write schema and n=12/14 lower-bound/truncation tests.
- [ ] Implement canonical export and resource metrics.
- [ ] Run:

```powershell
& ".venv-qiskit\Scripts\python.exe" -m experiments.resource_analysis --all
```

- [ ] Join new resource data with `t6_dynamical_lie_algebra.json` and `f_e_gradient_variance.json`.
- [ ] Run isolated focused tests and frozen full suite.
- [ ] Commit code and three canonical artifacts.

### Task 14: Replace historical noise with clean-manifest noise and fake-backend rehearsal

**Recommendations:** 12, excluding submission

**Files:**
- Create: `experiments/noise_analysis.py`
- Create: `experiments/fake_backend_run.py`
- Modify: `noise_sim.py`
- Modify: `experiments/hardware_run.py`
- Create: `tests/test_noise_protocol.py`
- Create: `tests/test_fake_backend_run.py`
- Create: `tests/test_hardware_payload.py`
- Create: `Results/evidence/f_c_noise_ideal.json`
- Create: `Results/evidence/f_c_noise_aer.json`
- Create: `Results/evidence/f_c_fake_backend.json`
- Create: `Results/evidence/f_c_fidelity_threshold.json`
- Create: `Results/hardware/payload.json`
- Create: `Results/hardware/rehearsal.json`

- [ ] Write tests rejecting RNG-reconstructed splits, mismatched weight/split IDs, missing original sample IDs, and absent preprocessing provenance.
- [ ] Write zero-noise agreement tests against the canonical PennyLane circuit.
- [ ] Write payload-hash and fake-backend provenance tests.
- [ ] Verify current `noise_sim.py` evidence path fails the tests.
- [ ] Implement manifest-first ideal, depolarizing, and realistic noise paths; keep historical mode explicitly non-evidence if retained.
- [ ] Implement exact fake-backend payload rehearsal and fidelity-threshold analysis.
- [ ] Run local and isolated focused tests plus frozen full suite.
- [ ] Inspect payload circuits, samples, shots, estimated use, backend snapshot, and hashes.
- [ ] Commit code and rehearsal artifacts. Do not submit hardware.

### Task 15: Submit and collect the scoped real-QPU job

**Recommendations:** final portion of 12

**Files:**
- Modify: `experiments/hardware_run.py`
- Create after submission: `Results/hardware/submission.json`
- Create after collection: `Results/hardware/job_<id>.json`
- Create after collection: `Results/evidence/f_c_real_qpu.json`
- Create after collection: `Results/evidence/f_c_noise_ladder.json`

**Interfaces:**

```bash
python -m experiments.hardware_run rehearse --payload Results/hardware/payload.json
python -m experiments.hardware_run submit --payload Results/hardware/payload.json --approved
python -m experiments.hardware_run collect --job-id <id>
```

- [ ] Write CLI tests requiring `--approved`, an exact rehearsal/payload hash match, named backend, clean provenance, valid token, and an approved resource estimate.
- [ ] Implement separate rehearse, submit, and collect commands.
- [ ] Run fake submission tests; verify no network submission occurs.
- [ ] Present backend, qubits, depth, ECR count, circuits, shots, samples, estimated use, payload hash, and rehearsal result to the user.
- [ ] Stop for explicit approval immediately before submission.
- [ ] Submit only after approval; record durable job ID and exact payload.
- [ ] Report queued/submitted, not complete.
- [ ] Collect when ready; validate result provenance and generate real-QPU and joined ladder artifacts.
- [ ] Commit code and immutable job/result records after collection.

---

## Tranche 5: Citation and Manuscript Package

### Task 16: Finish the sentence-level citation audit

**Recommendations:** 13

**Files:**
- Create: `scripts/audit_citations.py`
- Create: `tests/test_citation_audit.py`
- Modify: `fqcnn.tex`
- Modify: `docs/paper_code_reconciliation.md`
- Create: `Results/evidence/citation_audit.json`

- [ ] Write tests for undefined citations, uncited bibliography entries, missing DOI metadata, unresolved sentence-level findings, and explicit `ref44` resolution.
- [ ] Implement the audit and emit actionable finding locations.
- [ ] Run the audit, inspect every literature claim, and correct unsupported or imprecise citations.
- [ ] Record a reasoned editorial decision for each retained preprint; do not force a numeric ratio mechanically.
- [ ] Build LaTeX three times with `-halt-on-error` and inspect extracted PDF text for corruption.
- [ ] Require zero unresolved citation findings or explicitly approved exceptions.
- [ ] Commit citation code, manuscript corrections, reconciliation updates, and artifact.

### Task 17: Integrate evidence and build the venue package

**Recommendations:** 14

**Files:**
- Create: `scripts/build_manuscript_assets.py`
- Create: `scripts/validate_manuscript_claims.py`
- Create: `tests/test_manuscript_assets.py`
- Create: `tests/test_claim_traceability.py`
- Modify: `fqcnn.tex`
- Modify: `figs_final/*`
- Modify: `README.md`
- Modify: `STATUS.md`
- Create: `Results/evidence/manuscript_traceability.json`
- Create: `Results/evidence/submission_package_manifest.json`

- [ ] Write tests proving tables/figures derive from JSON/NPZ rather than hand-entered values.
- [ ] Write claim-traceability tests requiring evidence paths and caveats for every numerical manuscript claim.
- [ ] Stop for venue selection before applying a journal template.
- [ ] Implement asset generation and claim validation.
- [ ] Install the actual DLA, Caro gate-count, SU(4), coherent-extension, information-dynamics, simulability, and real-QPU findings without overstating them.
- [ ] Replace unqualified single-seed claims with distributions.
- [ ] Build and validate T1-T6 and F-A-F-F from canonical artifacts.
- [ ] Build the manuscript and source/supplement package.
- [ ] Hash the source, bibliography, figures, locks, manifests, evidence, and code commit into the package manifest.
- [ ] Run full tests and commit the complete package changes.

---

## Tranche 6: Reproduction and Red Team

### Task 18: Reproduce from a clean checkout

**Recommendations:** 15

**Files:**
- Modify: `reproduce.sh`
- Modify if maintained: `setup_env.ps1`
- Modify if maintained: `setup_env.bat`
- Create: `scripts/validate_evidence.py`
- Create: `scripts/reproduce_quick.py`
- Create: `tests/test_reproduction_contract.py`
- Create: `Results/evidence/reproduction_report.json`

- [ ] Write tests requiring clean Git state, frozen-lock hash, full-suite result, deterministic quick reproduction, evidence validation, manuscript build, and explicit external-artifact handling.
- [ ] Implement evidence schema/cross-reference validation and quick reproduction.
- [ ] Create an isolated worktree or fresh clone from the approved commit.
- [ ] Build the Python 3.9 training environment and run the full suite.
- [ ] Run quick reproduction, validate every canonical artifact, generate manuscript assets, and build LaTeX.
- [ ] Compare hashes or approved numerical tolerances.
- [ ] Mark QPU evidence as verified from immutable job records, not resubmitted.
- [ ] Generate a passing reproduction report and commit scripts plus report.

### Task 19: Run the definition-of-done and reviewer red team

**Recommendations:** final gate of 15

**Files:**
- Create: `docs/reviewer-question-map.md`
- Create: `scripts/red_team_audit.py`
- Create: `tests/test_red_team_audit.py`
- Create: `Results/evidence/red_team_report.json`
- Modify as findings require: `STATUS.md`, `fqcnn.tex`, evidence-producing code, or tests.

- [ ] Write audit tests covering leakage, seed counts, n=10 ablations, visible geometry exceptions, split identity, empty failure manifests, generated tables, state-preparation resource separation, DLA wording, real-QPU scope, clean reproduction, and historical/queued result labeling.
- [ ] Implement the reviewer map linking each question to manuscript claim, evidence artifact, command, split manifest, tests, caveats, and status.
- [ ] Run the audit and record every finding.
- [ ] Correct each failed item in its owning code/document/artifact and rerun relevant tests.
- [ ] Repeat until `red_team_report.json` has no open findings.
- [ ] Run the full suite and clean reproduction once more.
- [ ] Commit the reviewer map, audit tooling, final report, and corrections.
- [ ] Declare submission readiness only after every gate passes.

---

## Master Dependency Order

```text
Task 1 repository preservation
  -> Task 2 documentation truth
      -> Task 3 baseline protocol
      -> Task 4 n=10 ablations
          -> Task 5 n=10 quantum baselines
          -> Task 7 first safe campaign
      -> Task 6 isolated Qiskit

Task 3 + Task 4 + Task 7
  -> Task 8 harder datasets
      -> Task 9 baseline/statistical study
          -> Task 10 effective dimension and inductive bias
          -> Task 11 full ablation grid

Task 6
  -> Task 12 pooling practicality
  -> Task 13 resources/scaling
      -> Task 14 clean noise and fake backend
          -> Task 15 real-QPU approval, submission, collection

Tasks 8-15 complete
  -> Task 16 citation audit
      -> Task 17 manuscript/package
          -> Task 18 clean reproduction
              -> Task 19 red-team gate
```

## First-Day Completion Gate

The approved unblock-and-launch tranche is complete only when:

- The current local commits are preserved, or the exact reviewed push awaits explicit approval.
- All untracked files are classified and no local/generated files are staged.
- `STATUS.md` and `README.md` match canonical evidence.
- Baseline leakage tests fail before the fix and pass afterward.
- n=10 ablation tests fail before the fix and pass afterward.
- The frozen-environment full suite passes.
- The corrected n=10 cost estimate is reviewed and within budget.
- The isolated Qiskit environment imports and local transpilation pass.
- The first campaign has a valid manifest and is launched only after every prerequisite passes.
- A launched or running campaign is not described as complete.

## Canonical Completion Artifacts

| Recommendation | Completion artifact |
|---|---|
| 1 | `Results/evidence/repository_preservation.json` |
| 2 | Passing `tests/test_documentation_contract.py` and reviewed diff |
| 3 | `Results/evidence/baseline_protocol_audit.json` |
| 4 | `ablation_registry_audit.json`, `campaign_cost_estimate.json` |
| 5 | `qiskit_environment.json`, `qiskit_transpile_smoke.json` |
| 6 | `t1_dataset_table.json`, `dataset_provenance.json` |
| 7 | `t3_baselines.json`, learning curves, calibration |
| 8 | `t6_effective_dimension.json`, `t6_inductive_bias.json` |
| 9 | `t4_ablation_grid.json` |
| 10 | `t5_pooling_hardware.json` |
| 11 | `t2_resource_table.json`, `f_d_scaling_family.json`, `t6_scaling_join.json` |
| 12 | `f_c_noise_ladder.json` and immutable hardware records |
| 13 | `citation_audit.json` |
| 14 | `manuscript_traceability.json`, `submission_package_manifest.json` |
| 15 | `reproduction_report.json`, `red_team_report.json` |
