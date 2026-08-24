# Task 7 Immutable Campaign Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete roadmap Task 7 by making campaign planning, approval, launch, Windows-spawn execution, queue ownership, runtime status, and final completion evidence immutable and fail-closed without changing the canonical FQCNN architecture.

**Architecture:** Keep `experiments/campaign.py` as the lifecycle coordinator and `experiments/run_experiments.py` as the cell scheduler. Introduce one reusable exclusive-queue utility, pass campaign roots explicitly to every worker, validate separate immutable approval and launch records, and reject incomplete child evidence before completion.

**Tech Stack:** Python 3.9.13, pytest 7.4.4, `concurrent.futures.ProcessPoolExecutor`, Windows `spawn`, JSON/NPZ artifacts, existing PennyLane/NumPy training stack.

**Spec:** `docs/superpowers/specs/2026-08-24-journal-evidence-execution-design.md`

## Global Constraints

- Do not change FQCNN circuit topology, layer order, convolution or pooling semantics, feature map, active-wire schedule, parameter allocation/sharing, readout, objective, architecture fingerprint, or frozen expectation regression.
- Campaign manifests, cost estimates, approval attestations, and launch records are immutable.
- Production code and tests must not manufacture human approval.
- Missing, malformed, partial, queued, historical, failed, wrong-revision, or wrong-scope evidence fails closed.
- Planning, validation, and status inspection must not start training.
- A campaign may report `complete` only after every scheduler cell validates, the queue is no longer owned, and a valid final scheduler failure manifest exists and is empty.
- The approved first request remains `baseline_mnist_n10_v1`, datasets `0,1`, `3,5`, `4,9`, `5,8`, seeds `0..9`, config `proposed`, 400 samples, 30 epochs, no baselines, 40 scheduler cells, `--jobs 0`, and a maximum approved wall budget of 70 hours.
- Existing user-generated logs, PDFs, local settings, and stale campaign evidence must not be deleted or overwritten without a separate review decision.
- Do not launch the canonical campaign until every stop gate in this plan passes.

---

## File and Responsibility Map

### Production

- Create `QCNN/utils/exclusive_queue.py`
  - Atomically acquire, inspect, compare, and owner-release one compute lease.
- Modify `experiments/run_experiments.py`
  - Carry all campaign roots in each worker payload and apply them inside spawned workers.
- Modify `experiments/campaign.py`
  - Enforce manifest schema, canonical paths, planning ownership, approval validation, launch gates, launch-record validation, queue ownership, and runtime state.
- Modify `experiments/estimate_cost.py`
  - Separate scheduler cells, baseline side-effect cells, and total costed cells.
- Modify `QCNN/utils/run_artifacts.py`
  - Treat malformed status and NPZ artifacts as non-reusable instead of raising or passing.
- Modify `.gitignore`
  - Ignore new campaign runtime artifacts only after confirming the repository’s evidence policy.

### Tests

- Modify `tests/test_campaign_manifest.py`
  - Manifest, path, approval, queue, planning, launch, status, and cost contracts.
- Modify `tests/test_resume_and_parallelism.py`
  - Real Windows-spawn root propagation and malformed child artifacts.
- Modify `tests/test_cost_estimator.py`
  - Explicit cost-count semantics.
- Run existing `tests/test_baseline_protocol.py` and `tests/test_quantum_baselines.py`
  - Ensure Task 7 does not alter baseline behavior or FQCNN architecture.

### Existing runtime evidence

The current files under `Results/campaigns/baseline_mnist_n10_v1/` are stale and launch-ineligible. Preserve them until the pre-launch artifact-policy decision. Do not edit them into apparent validity.

---

### Task 1: Normalize the Second-Review Tests and Freeze Interfaces

**Files:**
- Modify: `tests/test_campaign_manifest.py:17-70`
- Modify: `tests/test_campaign_manifest.py:211-225`
- Modify: `tests/test_campaign_manifest.py:430-565`

**Interfaces:**
- Consumes: current campaign fixture helpers.
- Produces:

```python
def validate_manifest(manifest: dict) -> list:
    ...


def validate_cost(manifest: dict, cost: dict) -> list:
    ...


def approval_record_errors(manifest: dict, record: dict) -> list:
    ...


def launch_record_errors(manifest: dict, record: dict) -> list:
    ...


def launch_gate_errors(
    manifest: dict,
    current: dict = None,
    queue_path: Path = QUEUE_PATH,
) -> list:
    ...
```

- [ ] **Step 1: Change `_manifest()` into a valid non-approved fixture**

Use explicit count and approval paths:

```python
"request": {
    "datasets": ["0,1"],
    "class_pairs": [[0, 1]],
    "configs": ["proposed"],
    "seeds": [0],
    "split_policy": "deterministic stratified manifest per dataset and seed",
    "samples": 400,
    "epochs": 30,
    "jobs_requested": 1,
    "workers_resolved": 1,
    "with_baselines": False,
    "scheduler_cells": 1,
    "baseline_side_effect_cells": 0,
    "total_costed_cells": 1,
},
"artifacts": {
    "manifest": str(tmp_path / "manifest.json"),
    "cost_estimate": str(tmp_path / "cost_estimate.json"),
    "full_suite": str(tmp_path / "full_suite.json"),
    "approval": str(tmp_path / "approval.json"),
    "launch": str(tmp_path / "launch.json"),
    "status": str(tmp_path / "status.json"),
    "failures": str(tmp_path / "failures.json"),
    "scheduler_failures": str(tmp_path / "experiments" / "failures.json"),
},
"approval": {
    "required": True,
    "artifact": str(tmp_path / "approval.json"),
},
```

Do not include an authorizing boolean in the manifest.

- [ ] **Step 2: Add a test-only approval builder**

```python
def _approval_record(manifest, *, approver="human-reviewer"):
    return {
        "schema": {"name": "fqcnn_campaign_launch_approval", "version": 1},
        "approved": True,
        "campaign": manifest["campaign"],
        "git_sha": manifest["repository"]["git_sha"],
        "manifest_sha256": campaign.sha256_file(
            manifest["artifacts"]["manifest"]
        ),
        "cost_sha256": manifest["cost"]["sha256"],
        "full_suite_sha256": campaign.sha256_file(
            manifest["artifacts"]["full_suite"]
        ),
        "approver": approver,
        "approved_at_utc": "2026-08-24T00:00:00+00:00",
        "scope": {
            key: manifest["request"][key]
            for key in (
                "datasets", "configs", "seeds", "samples", "epochs",
                "jobs_requested", "workers_resolved", "with_baselines",
                "scheduler_cells", "baseline_side_effect_cells",
                "total_costed_cells",
            )
        },
    }
```

This creates test data only. Production receives no approval-minting API.

- [ ] **Step 3: Extend `_clean_current()` with full provenance**

```python
def _clean_current(manifest=None):
    workers = 1 if manifest is None else manifest["request"]["workers_resolved"]
    return {
        "git_sha": "abc",
        "branch": "dev",
        "upstream": "origin/dev",
        "ahead": 0,
        "behind": 0,
        "training_lock_sha256": "a" * 64,
        "qiskit_lock_sha256": "b" * 64,
        "workers_resolved": workers,
        "dirty": False,
        "dirty_policy_passed": True,
        "dirty_reasons": [],
    }
```

- [ ] **Step 4: Normalize review test expectations**

Make these exact changes:

- Rename the in-process root test to `test_worker_payload_applies_explicit_campaign_roots_in_process`.
- Replace production `valid_approval_record()` calls with `_approval_record()`.
- Add a test-only `_launch_record()` fixture rather than asking production to mint a valid approval.
- Replace `expected_cells` mutations with `scheduler_cells`.
- Require duplicate datasets/configs/seeds to fail explicitly.
- Require `type(n_qubits) is int and n_qubits > 0` when baselines are enabled.
- Assert a missing approval blocks launch while leaving the planned manifest schema-valid.

- [ ] **Step 5: Run the normalized red suite**

Run:

```powershell
python -m pytest tests/test_campaign_manifest.py -q
```

Expected: failures for missing production behavior, not fixture `KeyError`, contradictory approval defaults, or malformed test setup.

- [ ] **Step 6: Commit the red tests**

```powershell
git add tests/test_campaign_manifest.py
git commit -m "test: codify Task 7 immutable lifecycle review"
```

---

### Task 2: Add the Owner-Safe Exclusive Queue

**Files:**
- Create: `QCNN/utils/exclusive_queue.py`
- Modify: `tests/test_campaign_manifest.py`
- Modify: `experiments/campaign.py`

**Interfaces:**

```python
def acquire(path, metadata: dict) -> dict:
    """Atomically acquire and persist an owner record."""


def read(path) -> dict:
    """Read and validate a lease; raise ValueError if malformed."""


def is_owned(path, owner: dict) -> bool:
    """Return whether the current lease matches the supplied owner."""


def release(path, owner: dict) -> bool:
    """Remove only the exact owner's lease."""
```

- [ ] **Step 1: Add exclusive acquisition and foreign-release tests**

```python
def test_queue_acquisition_is_exclusive(tmp_path):
    path = tmp_path / "queue.lock"
    owner = exclusive_queue.acquire(path, {"campaign": "first"})
    with pytest.raises(FileExistsError):
        exclusive_queue.acquire(path, {"campaign": "second"})
    assert exclusive_queue.is_owned(path, owner)


def test_foreign_owner_cannot_release_live_lease(tmp_path):
    path = tmp_path / "queue.lock"
    owner = exclusive_queue.acquire(path, {"campaign": "first"})
    foreign = dict(owner, lease_id="not-owner")
    assert exclusive_queue.release(path, foreign) is False
    assert path.exists()
```

- [ ] **Step 2: Keep the metadata-write rollback regression**

```python
def test_queue_metadata_failure_rolls_back_lease(tmp_path, monkeypatch):
    path = tmp_path / "queue.lock"
    monkeypatch.setattr(
        exclusive_queue.json,
        "dump",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom")),
    )
    with pytest.raises(OSError):
        exclusive_queue.acquire(path, {"campaign": "unit"})
    assert not path.exists()
```

- [ ] **Step 3: Run queue tests and confirm they fail**

```powershell
python -m pytest tests/test_campaign_manifest.py -q -k "queue"
```

Expected: import or attribute failures because the shared queue helper is absent.

- [ ] **Step 4: Implement atomic acquisition with rollback**

```python
def acquire(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = dict(metadata)
    owner.update(
        lease_id=uuid.uuid4().hex,
        pid=os.getpid(),
        host=platform.node(),
        acquired_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(owner, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return owner
```

- [ ] **Step 5: Implement validated read and owner-only release**

Require an object with non-empty `lease_id`, integer PID, host, campaign/kind metadata, and timestamp. Re-read immediately before deletion and compare `lease_id`, campaign, host, and PID. Do not implement automatic stale-lease deletion.

- [ ] **Step 6: Replace campaign-local queue creation**

Use:

```python
lease = exclusive_queue.acquire(
    QUEUE_PATH,
    {"kind": "campaign", "campaign": manifest["campaign"]},
)
```

Release with:

```python
exclusive_queue.release(QUEUE_PATH, lease)
```

Never unconditionally unlink the queue path.

- [ ] **Step 7: Run focused queue and rollback tests**

```powershell
python -m pytest tests/test_campaign_manifest.py -q -k "queue or launch_record_failure"
```

Expected: pass.

- [ ] **Step 8: Commit**

```powershell
git add QCNN/utils/exclusive_queue.py experiments/campaign.py tests/test_campaign_manifest.py
git commit -m "feat: add owner-safe exclusive compute queue"
```

---

### Task 3: Propagate Campaign Roots Through Spawned Workers

**Files:**
- Modify: `experiments/run_experiments.py:61-64`
- Modify: `experiments/run_experiments.py:306-459`
- Modify: `experiments/campaign.py:527-558`
- Modify: `tests/test_campaign_manifest.py`
- Modify: `tests/test_resume_and_parallelism.py`

**Interfaces:**

```python
def _apply_output_roots(roots: dict) -> dict:
    ...


def _execute_cell(payload):
    # pair, config_name, seed, dataset_dir, samples, epochs,
    # use_bce, with_baselines, roots
    ...


def main(output_roots: dict = None):
    ...
```

Required keys:

```python
{
    "experiments": "...",
    "runs": "...",
    "manifests": "...",
    "failures": ".../experiments/failures.json",
}
```

- [ ] **Step 1: Extend the in-process worker payload test**

Assert all four effective roots after `_execute_cell()` receives an explicit roots dictionary.

- [ ] **Step 2: Add a real explicit-spawn test**

```python
def test_explicit_campaign_roots_survive_spawn(tmp_path):
    roots = {
        "experiments": str(tmp_path / "campaign" / "experiments"),
        "runs": str(tmp_path / "campaign" / "runs"),
        "manifests": str(tmp_path / "campaign" / "manifests"),
        "failures": str(
            tmp_path / "campaign" / "experiments" / "failures.json"
        ),
    }
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        observed = pool.submit(
            run_experiments._apply_output_roots, roots
        ).result()
    assert observed == roots
```

Do not skip this test on Windows.

- [ ] **Step 3: Run red root tests**

```powershell
python -m pytest tests/test_campaign_manifest.py -q -k "worker_payload"
python -m pytest tests/test_resume_and_parallelism.py -q -k "spawn"
```

Expected: failure because the payload does not contain or apply roots.

- [ ] **Step 4: Implement strict root application**

```python
def _apply_output_roots(roots):
    global EXP_ROOT, FAILURE_MANIFEST, MANIFEST_ROOT
    required = {"experiments", "runs", "manifests", "failures"}
    if not isinstance(roots, dict) or set(roots) != required:
        raise ValueError("worker output roots have an invalid shape")
    if any(not isinstance(value, str) or not value for value in roots.values()):
        raise ValueError("worker output roots must be non-empty strings")

    EXP_ROOT = roots["experiments"]
    FAILURE_MANIFEST = roots["failures"]
    MANIFEST_ROOT = roots["manifests"]
    run_artifacts.RUN_ROOT = roots["runs"]
    return {
        "experiments": EXP_ROOT,
        "runs": run_artifacts.RUN_ROOT,
        "manifests": MANIFEST_ROOT,
        "failures": FAILURE_MANIFEST,
    }
```

- [ ] **Step 5: Apply roots before any worker reuse or execution logic**

Unpack the ninth payload field and call `_apply_output_roots(roots)` first.

- [ ] **Step 6: Add roots to every pending cell**

Resolve defaults once in `main(output_roots=None)` and append the same normalized roots to every payload.

- [ ] **Step 7: Pass roots explicitly from campaign launch**

```python
roots = {
    "experiments": manifest["output_roots"]["experiments"],
    "runs": manifest["output_roots"]["runs"],
    "manifests": manifest["output_roots"]["manifests"],
    "failures": manifest["artifacts"]["scheduler_failures"],
}
run_experiments.main(output_roots=roots)
```

- [ ] **Step 8: Run focused tests**

```powershell
python -m pytest tests/test_resume_and_parallelism.py -q
python -m pytest tests/test_campaign_manifest.py -q -k "worker or root"
```

Expected: pass.

- [ ] **Step 9: Commit**

```powershell
git add experiments/run_experiments.py experiments/campaign.py tests/test_campaign_manifest.py tests/test_resume_and_parallelism.py
git commit -m "fix: isolate campaign roots in spawned workers"
```

---

### Task 4: Make Planning Ownership and Path Validation Fail Closed

**Files:**
- Modify: `experiments/campaign.py:146-248`
- Modify: `experiments/campaign.py:446-510`
- Modify: `tests/test_campaign_manifest.py`

**Interfaces:**

```python
def plan(campaign_name: str, repository_snapshot: dict = None) -> dict:
    ...
```

- [ ] **Step 1: Add canonical-root tests**

Reject each case independently:

- campaign root differs from `(CAMPAIGN_ROOT / campaign).resolve()`;
- a child path escapes with `..`;
- two output roots alias one another;
- approval, full-suite, or scheduler-failure paths leave the campaign root;
- artifact filenames differ from their exact canonical names;
- `manifest["cost"]["path"]` differs from `artifacts.cost_estimate`;
- a supported test symlink resolves outside the campaign root.

- [ ] **Step 2: Add strict request-shape tests**

Reject:

```python
datasets=[None]
datasets=["0,1", "0,1"]
configs=["proposed", "proposed"]
seeds=[0, 0]
samples=True
epochs=30.0
jobs_requested=True
workers_resolved=0
scheduler_cells=True
with_baselines=1
n_qubits=0
```

Also reject empty strings, malformed class pairs, negative seeds, duplicate class pairs, and mismatched counts.

- [ ] **Step 3: Add concurrent planning ownership tests**

Keep the existing loser-preserves-owner test and add a test that forces a failure after this invocation creates the directory. Assert cleanup removes only files created by that invocation.

- [ ] **Step 4: Run red planning tests**

```powershell
python -m pytest tests/test_campaign_manifest.py -q -k "canonical or path or duplicate or malformed or concurrent_plan"
```

Expected: fail under current check-then-create behavior.

- [ ] **Step 5: Enforce the exact canonical root**

```python
canonical = (CAMPAIGN_ROOT / manifest["campaign"]).resolve()
if Path(roots["campaign"]).resolve() != canonical:
    errors.append("campaign root is not canonical")
```

Require exact descendants for experiments, runs, manifests, manifest JSON, cost JSON, full-suite JSON, approval JSON, launch JSON, status JSON, campaign failure JSON, and scheduler failure JSON.

- [ ] **Step 6: Enforce strict types and unique axes**

Use `type(value) is int` where booleans must fail. Calculate:

```python
scheduler_cells = len(datasets) * len(configs) * len(seeds)
baseline_side_effect_cells = (
    len(datasets) * len(seeds) if with_baselines else 0
)
total_costed_cells = scheduler_cells + baseline_side_effect_cells
```

Require all recorded counts to match.

- [ ] **Step 7: Acquire campaign directory ownership atomically**

```python
created_directory = False
try:
    directory.mkdir(parents=True, exist_ok=False)
    created_directory = True
    # Generate owned artifacts.
except Exception:
    if created_directory:
        # Remove only artifacts created by this invocation.
        pass
    raise
```

Implement explicit owned-path tracking rather than recursive deletion. If `mkdir(..., exist_ok=False)` loses, do not remove anything.

- [ ] **Step 8: Capture exact repository state in the CLI path**

Return:

```python
{
    "git_sha": str,
    "branch": str,
    "upstream": str,
    "ahead": int,
    "behind": int,
    "training_lock_sha256": str,
    "qiskit_lock_sha256": str,
    "workers_resolved": int,
    "dirty": bool,
    "dirty_policy_passed": bool,
    "dirty_reasons": list,
}
```

Use read-only Git commands. `plan()` may accept an injected snapshot for unit tests. The CLI plan path must fail before writing evidence if the tree is dirty, detached unexpectedly, has no configured upstream, or violates the approved divergence policy.

- [ ] **Step 9: Run focused planning tests**

```powershell
python -m pytest tests/test_campaign_manifest.py -q -k "manifest or path or plan or repository"
```

Expected: pass.

- [ ] **Step 10: Commit**

```powershell
git add experiments/campaign.py tests/test_campaign_manifest.py
git commit -m "fix: make campaign planning ownership fail closed"
```

---

### Task 5: Separate Scheduler Cells From Baseline Cost Side Effects

**Files:**
- Modify: `experiments/estimate_cost.py:173-185`
- Modify: `experiments/estimate_cost.py:278-315`
- Modify: `experiments/estimate_cost.py:383-419`
- Modify: `experiments/campaign.py:251-286`
- Modify: `tests/test_cost_estimator.py`
- Modify: `tests/test_campaign_manifest.py`

**Interfaces:**

```python
"counts": {
    "requested_configs": int,
    "measurable_configs": int,
    "scheduler_cells": int,
    "baseline_side_effect_cells": int,
    "total_costed_cells": int,
    "measurable_scheduler_cells": int,
    "measurable_baseline_side_effect_cells": int,
    "measurable_total_costed_cells": int,
    "failed_calibrations": int,
    "total_failures": int,
}
```

- [ ] **Step 1: Add exact count tests**

For one dataset, one config, one seed:

- without baselines: scheduler `1`, baseline `0`, total `1`;
- with baselines: scheduler `1`, baseline `1`, total `2`;
- a failed baseline calibration leaves scheduler work measurable but total work incomplete;
- manifest scheduler count compares only to cost scheduler count;
- approval requires every total-costed cell to be measurable.

- [ ] **Step 2: Run red cost tests**

```powershell
python -m pytest tests/test_cost_estimator.py tests/test_campaign_manifest.py -q -k "cell or baseline or cost"
```

Expected: failure because current cost fields are ambiguous.

- [ ] **Step 3: Implement named counts in every cost-return path**

Update normal, invalid-budget, and failed-calibration outputs. Remove ambiguous `requested_cells` from the new schema rather than preserving it as an alias.

- [ ] **Step 4: Tighten `validate_cost()`**

Require:

- exact request identity;
- exact scheduler, baseline, and total counts;
- measurable counts equal requested counts;
- zero failures;
- finite nonnegative projections;
- positive finite budget;
- exact resolved worker count.

- [ ] **Step 5: Run baseline regression suites**

```powershell
python -m pytest tests/test_cost_estimator.py tests/test_baseline_protocol.py tests/test_quantum_baselines.py -q
```

Expected: pass without changing model implementations.

- [ ] **Step 6: Commit**

```powershell
git add experiments/estimate_cost.py experiments/campaign.py tests/test_cost_estimator.py tests/test_campaign_manifest.py
git commit -m "fix: distinguish scheduler and baseline cost cells"
```

---

### Task 6: Validate Separate Immutable Approval and Launch Records

**Files:**
- Modify: `experiments/campaign.py:289-401`
- Modify: `experiments/campaign.py:527-558`
- Modify: `tests/test_campaign_manifest.py`

**Interfaces:**

```python
def approval_record_errors(manifest: dict, record: dict) -> list:
    ...


def _approval_errors(manifest: dict) -> list:
    ...


def launch_record_errors(manifest: dict, record: dict) -> list:
    ...


def _load_valid_launch_record(manifest: dict):
    ...


def _build_launch_record(
    manifest: dict,
    approval_sha256: str,
    queue_owner: dict,
) -> dict:
    ...
```

- [ ] **Step 1: Add approval mutation tests**

Each mutation must block launch:

- wrong campaign;
- wrong Git SHA;
- wrong manifest, cost, or full-suite hash;
- `approved` not exactly `True`;
- missing approver or approval timestamp;
- changed datasets, configs, seeds, samples, epochs, workers, baseline mode, or counts;
- malformed JSON;
- approval path outside the canonical campaign directory.

- [ ] **Step 2: Add launch-record mutation tests**

Require the launch record to bind:

```python
{
    "schema": {"name": "fqcnn_campaign_launch", "version": 1},
    "campaign": manifest["campaign"],
    "git_sha": manifest["repository"]["git_sha"],
    "manifest_sha256": sha256_file(manifest["artifacts"]["manifest"]),
    "cost_sha256": manifest["cost"]["sha256"],
    "full_suite_sha256": sha256_file(manifest["artifacts"]["full_suite"]),
    "approval_sha256": sha256_file(manifest["artifacts"]["approval"]),
    "queue": str(QUEUE_PATH),
    "queue_lease_id": queue_owner["lease_id"],
    "owner_pid": queue_owner["pid"],
    "owner_host": queue_owner["host"],
    "launched_at_utc": "2026-08-24T00:00:00+00:00",
}
```

- [ ] **Step 3: Run red lifecycle tests**

```powershell
python -m pytest tests/test_campaign_manifest.py -q -k "approval or launch_record or runtime_lifecycle"
```

Expected: fail because separate validation does not exist.

- [ ] **Step 4: Implement immutable approval validation**

`_approval_errors()` must require the exact file, parse JSON fail-closed, validate schema and strict field types, recompute hashes, compare exact scope, and return errors without modifying files.

- [ ] **Step 5: Recheck full repository provenance and workers at launch**

Compare `git_sha`, branch, upstream, ahead, behind, both lock hashes, dirty policy, and `workers_resolved` with the manifest and cost evidence.

- [ ] **Step 6: Revalidate immediately before queue acquisition**

`launch_gate_errors()` remains side-effect free. `launch()` captures state again and reruns the gate immediately before acquiring the queue.

- [ ] **Step 7: Acquire the queue and then write the launch record**

Order:

```python
errors = launch_gate_errors(manifest)
if errors:
    return errors

lease = exclusive_queue.acquire(
    QUEUE_PATH,
    {"kind": "campaign", "campaign": manifest["campaign"]},
)
try:
    write_immutable_json(
        manifest["artifacts"]["launch"],
        _build_launch_record(
            manifest,
            approval_sha256=sha256_file(manifest["artifacts"]["approval"]),
            queue_owner=lease,
        ),
    )
    run_experiments.main(output_roots=roots)
finally:
    exclusive_queue.release(QUEUE_PATH, lease)
```

If launch-record writing fails, release only this lease and never start the scheduler.

- [ ] **Step 8: Prevent a second launch**

A valid existing launch record produces a specific error directing the operator to status/resume behavior. An incompatible record remains a hard conflict.

- [ ] **Step 9: Run launch lifecycle tests**

```powershell
python -m pytest tests/test_campaign_manifest.py -q -k "approval or launch or repository or worker"
```

Expected: pass.

- [ ] **Step 10: Commit**

```powershell
git add experiments/campaign.py tests/test_campaign_manifest.py
git commit -m "feat: bind campaign launch to immutable approval"
```

---

### Task 7: Fail Closed on Child Evidence and Completion

**Files:**
- Modify: `QCNN/utils/run_artifacts.py:43-118`
- Modify: `experiments/campaign.py:404-426`
- Modify: `experiments/campaign.py:561-585`
- Modify: `tests/test_resume_and_parallelism.py`
- Modify: `tests/test_campaign_manifest.py`

**Interfaces:**

```python
def _read_status(directory: str) -> dict:
    ...


def is_reusable(directory: str, config: dict = None, seed: int = None) -> bool:
    ...


def _scheduler_failure_evidence(manifest: dict) -> dict:
    ...


def campaign_state(
    manifest: dict,
    launch_record: dict = None,
    failure_evidence: dict = None,
    queue_owned: bool = False,
) -> str:
    ...
```

- [ ] **Step 1: Add malformed run-artifact tests**

Cover truncated `status.json`, a JSON list instead of an object, malformed config/seed, corrupt weights NPZ, corrupt predictions NPZ, missing prediction arrays, and unequal prediction lengths. Every case returns `False`, not an exception.

- [ ] **Step 2: Add the completion matrix**

| Valid launch | Queue owned | Cells valid | Final scheduler evidence | Expected |
|---|---:|---:|---|---|
| no | any | any | any | `pending` |
| yes | yes | all | valid empty | `running` |
| yes | no | incomplete | missing | `running` |
| yes | no | all | missing | `running` |
| yes | no | all | malformed | `failed` |
| yes | no | all | non-empty | `failed` |
| yes | no | all | valid empty | `complete` |

Also assert malformed launch evidence yields `pending` and does not create mutable status.

- [ ] **Step 3: Run red evidence tests**

```powershell
python -m pytest tests/test_resume_and_parallelism.py tests/test_campaign_manifest.py -q -k "malformed or corrupt or completion or status or queue_owned or failure_manifest"
```

Expected: failures or exceptions under current permissive behavior.

- [ ] **Step 4: Harden `_read_status()`**

```python
def _read_status(directory):
    path = os.path.join(directory, _STATUS)
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}
```

- [ ] **Step 5: Validate required NPZ contents**

Use `np.load(..., allow_pickle=False)` in a context manager. Require prediction arrays `sample_ids`, `y_true`, and `raw_outputs`, compatible one-dimensional lengths, and readable finite weights. Do not change parameter meaning or allocation.

- [ ] **Step 6: Return structured scheduler-failure evidence**

Missing evidence:

```python
{
    "available": False,
    "valid": False,
    "failures": [],
    "errors": ["missing final scheduler failure manifest"],
}
```

Malformed evidence:

```python
{
    "available": True,
    "valid": False,
    "failures": [{"error": "invalid scheduler failure manifest"}],
    "errors": ["invalid scheduler failure manifest"],
}
```

- [ ] **Step 7: Validate launch before runtime status**

An existing but invalid launch JSON does not count as launched.

- [ ] **Step 8: Make queue ownership authoritative**

If this campaign owns the queue, state remains `running`. A foreign or malformed queue becomes a status error, not evidence that this campaign is active.

- [ ] **Step 9: Write mutable status only after valid launch**

Include schema, campaign, state, checked time, valid/expected cells, queue ownership, scheduler-failure availability/validity/count, and errors.

- [ ] **Step 10: Run focused lifecycle tests**

```powershell
python -m pytest tests/test_resume_and_parallelism.py tests/test_campaign_manifest.py -q
```

Expected: pass.

- [ ] **Step 11: Commit**

```powershell
git add QCNN/utils/run_artifacts.py experiments/campaign.py tests/test_resume_and_parallelism.py tests/test_campaign_manifest.py
git commit -m "fix: fail closed on incomplete campaign evidence"
```

---

### Task 8: Resolve Runtime-Evidence Tracking Without Deleting User Files

**Files:**
- Potentially modify: `.gitignore`
- Review only: `Results/campaigns/baseline_mnist_n10_v1/*`
- Review only: current untracked logs, PDFs, and `.claude/settings.local.json`
- Modify: repository-coupled campaign tests if they read stale checked-in JSON directly

**Interfaces:**
- Consumes: the repository’s intended evidence-retention policy.
- Produces: a clean-tree policy that can coexist with exact-revision runtime evidence.

- [ ] **Step 1: Inventory every dirty path without modifying it**

Run:

```powershell
git status --short
git ls-files Results/campaigns/baseline_mnist_n10_v1
```

Classify each path as intentional source/evidence, historical stale evidence, local configuration, generated LaTeX/PDF output, or raw experiment log.

- [ ] **Step 2: Present an explicit artifact-policy decision**

For each existing path, offer only non-destructive choices:

- keep tracked as historical evidence and generate a new campaign revision;
- archive to an explicitly approved tracked historical location;
- ignore future generated copies while preserving current files;
- move a local-only copy outside the repository after explicit approval.

Do not delete, overwrite, or silently add broad ignore rules.

- [ ] **Step 3: Add a narrow campaign-runtime ignore rule only if approved**

```gitignore
# Campaign runtime evidence is generated against an exact code revision.
Results/campaigns/*/experiments/
Results/campaigns/*/runs/
Results/campaigns/*/manifests/
Results/campaigns/*/status.json
Results/campaigns/*/failures.json
```

Do not ignore immutable manifest, cost, full-suite, approval, or launch records unless the approved evidence policy explicitly places all campaign records outside Git.

- [ ] **Step 4: Remove repository-coupled tests**

Tests must generate manifests under `tmp_path`; they must not depend on stale `Results/campaigns/...` files.

- [ ] **Step 5: Run campaign and documentation contracts**

```powershell
python -m pytest tests/test_campaign_manifest.py tests/test_documentation_contract.py -q
```

Expected: pass.

- [ ] **Step 6: Commit only the approved policy and test changes**

```powershell
git add .gitignore tests/test_campaign_manifest.py tests/test_documentation_contract.py
git commit -m "chore: define campaign runtime artifact policy"
```

If `.gitignore` or documentation-contract tests do not require changes, omit them from the commit rather than creating speculative edits.

---

### Task 9: Run the Complete Task 7 Verification Gate

**Files:**
- Verify only: all Task 7 production and test files.

- [ ] **Step 1: Run campaign regressions**

```powershell
python -m pytest tests/test_campaign_manifest.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run real spawn coverage**

```powershell
python -m pytest tests/test_resume_and_parallelism.py -q -k "spawn or root"
```

Expected: zero failures and actual `multiprocessing.get_context("spawn")` coverage.

- [ ] **Step 3: Run the full resume/parallelism suite**

```powershell
python -m pytest tests/test_resume_and_parallelism.py -q
```

Expected: zero failures.

- [ ] **Step 4: Run cost and baseline contracts**

```powershell
python -m pytest tests/test_cost_estimator.py tests/test_baseline_protocol.py tests/test_quantum_baselines.py -q
```

Expected: zero failures.

- [ ] **Step 5: Run frozen architecture gates**

First locate the current architecture fingerprint tests without editing architecture code:

```powershell
python -m pytest tests/test_ablation_registry.py -q
```

Run the existing fingerprint and frozen-expectation test files identified in the repository. Expected: every frozen architecture test passes unchanged.

- [ ] **Step 6: Run the full suite**

```powershell
python -m pytest tests/ -q
```

Expected: exit code `0`, failed `0`, and no skip hiding required Windows-spawn coverage on the launch host.

- [ ] **Step 7: Review diff hygiene**

```powershell
git diff --check
git status --short
git diff -- .gitignore QCNN/utils/exclusive_queue.py QCNN/utils/run_artifacts.py experiments/campaign.py experiments/run_experiments.py experiments/estimate_cost.py tests/test_campaign_manifest.py tests/test_resume_and_parallelism.py tests/test_cost_estimator.py
```

Reject unrelated changes to circuits, layers, trainer behavior, manuscript, Qiskit tooling, or generated outputs.

- [ ] **Step 8: Commit integration-only fixes if needed**

```powershell
git add QCNN/utils/exclusive_queue.py QCNN/utils/run_artifacts.py experiments/campaign.py experiments/run_experiments.py experiments/estimate_cost.py tests/test_campaign_manifest.py tests/test_resume_and_parallelism.py tests/test_cost_estimator.py
git commit -m "fix: complete immutable campaign lifecycle"
```

Skip this commit if all work is already represented by prior verified commits.

---

### Task 10: Regenerate Exact-Revision Campaign Evidence Without Launching

**Files:**
- Generate: approved canonical campaign evidence paths under `Results/campaigns/baseline_mnist_n10_v1/` or a newly approved revisioned path.
- Do not generate: launch, status, failures, runs, experiment cells, or split manifests.

**Interfaces:**
- Consumes: clean exact Task 7 revision, approved artifact policy, free queue.
- Produces: manifest, cost estimate, and full-suite evidence bound to the exact revision.

- [ ] **Step 1: Verify regeneration preconditions**

```powershell
git status --short
git rev-parse HEAD
git branch --show-current
git rev-parse --abbrev-ref --symbolic-full-name "@{u}"
git rev-list --left-right --count "HEAD...@{u}"
```

Expected: clean tree and the exact reviewed branch/upstream/divergence policy. If local commits are intentionally ahead, stop for approval rather than pretending ahead/behind is zero.

- [ ] **Step 2: Confirm lock hashes and free queue**

```powershell
Get-FileHash requirements-lock.txt -Algorithm SHA256
Get-FileHash requirements-qiskit-lock.txt -Algorithm SHA256
Test-Path Results\.training-hardware-queue.lock
```

Expected: lock hashes match the reviewed values and queue path is absent.

- [ ] **Step 3: Plan the unchanged campaign**

```powershell
python -m experiments.campaign plan --campaign baseline_mnist_n10_v1
```

Expected: manifest and cost evidence only. No scheduler cells, queue lease, launch record, or runtime output.

- [ ] **Step 4: Inspect exact request and cost**

Verify:

- datasets are exactly `0,1`, `3,5`, `4,9`, `5,8`;
- config is exactly `proposed`;
- seeds are exactly `0..9`;
- samples `400`, epochs `30`;
- no baselines;
- scheduler `40`, baseline side effects `0`, total costed `40`;
- all cells measurable;
- worker count equals the current resolved value;
- projected wall time is at or below 70 hours.

A changed worker count or material estimate change is a stop requiring a new approval decision.

- [ ] **Step 5: Generate current-revision full-suite evidence**

Run:

```powershell
python -m pytest tests/ -q
```

Use the project’s reviewed evidence-recording path to write immutable `full_suite.json` with actual pass counts, exact Git SHA, training lock hash, UTC timestamp, and `dirty_policy.passed: true`.

- [ ] **Step 6: Validate before approval**

```powershell
python -m experiments.campaign validate --campaign baseline_mnist_n10_v1
```

Expected: exactly the missing-approval gate. Any additional manifest, cost, suite, repository, queue, worker, lock, or path error stops progress.

- [ ] **Step 7: Present exact evidence hashes**

```powershell
Get-FileHash Results\campaigns\baseline_mnist_n10_v1\manifest.json -Algorithm SHA256
Get-FileHash Results\campaigns\baseline_mnist_n10_v1\cost_estimate.json -Algorithm SHA256
Get-FileHash Results\campaigns\baseline_mnist_n10_v1\full_suite.json -Algorithm SHA256
git rev-parse HEAD
git status --short
```

Also report branch, upstream, divergence, lock hashes, resolved workers, count fields, serial hours, wall hours, budget fraction, and test totals.

---

### Task 11: Create the Approval Attestation at the Human Gate

**Files:**
- Create only after approval: `Results/campaigns/baseline_mnist_n10_v1/approval.json`

**Interfaces:**
- Consumes: exact hashes and scope from Task 10.
- Produces: one immutable human approval record.

- [ ] **Step 1: Stop and request approval for the exact final evidence**

Display the manifest, cost, and full-suite hashes; Git SHA; branch/upstream/divergence; lock hashes; workers; scope; projected usage; and full test results.

- [ ] **Step 2: Confirm the earlier gated authorization still applies**

It applies only if the campaign remains exactly four approved MNIST pairs, proposed model, ten seeds, 400 samples, 30 epochs, no baselines, 40 cells, and materially unchanged compute scope. Otherwise request new approval.

- [ ] **Step 3: Write the immutable approval only after explicit confirmation**

The record must contain:

```json
{
  "schema": {
    "name": "fqcnn_campaign_launch_approval",
    "version": 1
  },
  "approved": true,
  "campaign": "baseline_mnist_n10_v1",
  "git_sha": "the exact reviewed Git SHA",
  "manifest_sha256": "the exact reviewed manifest hash",
  "cost_sha256": "the exact reviewed cost hash",
  "full_suite_sha256": "the exact reviewed full-suite hash",
  "approver": "the approving human identity",
  "approved_at_utc": "the approval time in UTC ISO-8601",
  "scope": {
    "datasets": ["0,1", "3,5", "4,9", "5,8"],
    "configs": ["proposed"],
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "samples": 400,
    "epochs": 30,
    "jobs_requested": 0,
    "workers_resolved": 20,
    "with_baselines": false,
    "scheduler_cells": 40,
    "baseline_side_effect_cells": 0,
    "total_costed_cells": 40
  }
}
```

Use the actual resolved worker count rather than forcing `20` if the machine changed.

- [ ] **Step 4: Revalidate immediately**

```powershell
python -m experiments.campaign validate --campaign baseline_mnist_n10_v1
git status --short
git rev-parse HEAD
```

Expected: `VALID`, exit code `0`, unchanged Git revision, and no unapproved dirty path.

---

### Task 12: Launch Once and Inspect the First Cell

**Files:**
- Runtime-create: immutable launch record, mutable status, queue lease, campaign cell outputs.

**Interfaces:**
- Consumes: valid approval-bound campaign.
- Produces: one launched/running campaign and first-cell evidence.

- [ ] **Step 1: Recheck every launch gate**

Verify exact repository identity, clean policy, locks, request, counts, cost, full-suite evidence, approval bindings, free queue, canonical paths, no output collision, and no valid existing launch record.

- [ ] **Step 2: Launch exactly once**

```powershell
python -m experiments.campaign launch --campaign baseline_mnist_n10_v1
```

Expected: one exclusive queue lease, one immutable launch record, and scheduler startup. A second launch command must refuse.

- [ ] **Step 3: Query status**

```powershell
python -m experiments.campaign status --campaign baseline_mnist_n10_v1
```

Expected: `running`, not `complete`.

- [ ] **Step 4: Inspect the first completed cell**

Require:

- n=10 geometry;
- `proposed` identity;
- expected split ID and ordered source sample IDs;
- predictions readable and aligned with labels;
- exactly one test evaluation;
- readable weights;
- complete valid run status;
- all roots beneath the campaign;
- no files under default non-campaign experiment/run/manifest roots;
- no scheduler failure entry;
- queue still owned by this campaign.

- [ ] **Step 5: Stop on any first-cell defect**

Do not continue unattended if geometry, split, IDs, roots, weights, predictions, status, queue, or test-evaluation count is wrong.

- [ ] **Step 6: Report launched/running**

Do not call the campaign complete while cells remain or final scheduler evidence is absent.

---

### Task 13: Verify Final Completion and Close Task 7

**Files:**
- Runtime-read: campaign runs, scheduler failure evidence, launch record, status.
- Update later planning ledger/status only after validated completion.

- [ ] **Step 1: Wait for the campaign process to finish through its existing runtime mechanism**

Do not launch a duplicate process or silently substitute partial outputs.

- [ ] **Step 2: Validate all 40 cells**

Every run must pass hardened `run_artifacts.is_reusable()` with exact config, seed, readable status, readable predictions, readable weights, and consistent identities.

- [ ] **Step 3: Validate final scheduler evidence**

Require the final scheduler failure manifest to exist, parse, and report `n_failed == 0` with `failures == []`.

- [ ] **Step 4: Confirm the queue is released**

The campaign must no longer own the shared queue before completion is reported.

- [ ] **Step 5: Query final status**

```powershell
python -m experiments.campaign status --campaign baseline_mnist_n10_v1
```

Expected: `complete` only when all prior conditions are true.

- [ ] **Step 6: Run the post-campaign focused and full suites**

```powershell
python -m pytest tests/test_campaign_manifest.py tests/test_resume_and_parallelism.py -q
python -m pytest tests/ -q
```

Expected: zero failures.

- [ ] **Step 7: Record Task 7 completion**

Update the execution ledger and `STATUS.md` only from validated canonical evidence. Record actual test totals, campaign state, hashes, elapsed time, and any caveat. Do not alter FQCNN architecture claims.

---

## Launch Stop-Gate Checklist

Do not execute Task 12 unless every item is true:

- [ ] Exact Git SHA equals manifest, suite, and approval.
- [ ] Branch, upstream, ahead, and behind values match the approved provenance policy.
- [ ] Working tree satisfies the approved clean policy.
- [ ] Training and Qiskit lock hashes match.
- [ ] Frozen architecture tests pass unchanged.
- [ ] Campaign request is unchanged.
- [ ] Scheduler/baseline/total count fields equal `40/0/40`.
- [ ] Resolved worker count matches manifest, cost, and approval.
- [ ] Cost is fully measurable and at or below the approved budget.
- [ ] Full-suite evidence passes on the exact revision.
- [ ] Separate approval bindings all validate.
- [ ] Shared queue is absent before acquisition.
- [ ] No campaign runtime collision exists.
- [ ] Canonical root checks pass.
- [ ] No valid launch record already exists.
- [ ] Launch-record creation occurs after lease acquisition and before scheduler invocation.

## Completion Checklist

Task 7 is complete only when:

- [ ] All 40 expected run directories validate.
- [ ] Predictions and weights NPZ files are readable and structurally valid.
- [ ] Every cell matches exact config, seed, split, and sample identities.
- [ ] Final scheduler failure evidence exists and is valid.
- [ ] Final scheduler failures are empty.
- [ ] The campaign no longer owns the queue.
- [ ] The immutable launch record remains valid.
- [ ] Status reports `complete`.
- [ ] Focused and full tests pass after campaign completion.
- [ ] Status documentation matches canonical evidence.
- [ ] No canonical FQCNN architecture file or behavior changed.
