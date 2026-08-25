#!/usr/bin/env python3
"""Immutable provenance and fail-closed gates for experiment campaigns."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from QCNN.utils import exclusive_queue, run_artifacts
from experiments import run_experiments

CAMPAIGN_STATES = {"pending", "running", "failed", "complete"}
CAMPAIGN_ROOT = Path("Results") / "campaigns"
QUEUE_PATH = Path("Results") / ".training-hardware-queue.lock"
TRAINING_LOCK = Path("requirements-lock.txt")
QISKIT_LOCK = Path("requirements-qiskit-lock.txt")
FIRST_CAMPAIGN = "baseline_mnist_n10_v1"
FIRST_DATASETS = ["0,1", "3,5", "4,9", "5,8"]
FIRST_CONFIGS = ["proposed"]
FIRST_SEEDS = list(range(10))


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(stat_result):
    return stat_result.st_dev, stat_result.st_ino


def _restore_claimed_file(claimed, path):
    """Restore a non-owned claim without replacing a newer canonical file."""
    try:
        os.link(str(claimed), str(path), follow_symlinks=False)
    except OSError:
        return False
    try:
        claimed.unlink()
    except OSError:
        return False
    return True


def _unlink_if_owned(path, identity):
    """Atomically claim the pathname, then verify and remove only its inode."""
    path = Path(path)
    claim_directory = Path(tempfile.mkdtemp(
        prefix=".campaign-cleanup-", dir=str(path.parent)
    ))
    claimed = claim_directory / path.name
    try:
        os.rename(str(path), str(claimed))
    except OSError:
        with contextlib.suppress(OSError):
            claim_directory.rmdir()
        return False
    try:
        current = _file_identity(
            os.stat(str(claimed), follow_symlinks=False)
        )
    except OSError:
        _restore_claimed_file(claimed, path)
        with contextlib.suppress(OSError):
            claim_directory.rmdir()
        return False
    if current != identity:
        _restore_claimed_file(claimed, path)
        with contextlib.suppress(OSError):
            claim_directory.rmdir()
        return False
    try:
        claimed.unlink()
    except OSError:
        _restore_claimed_file(claimed, path)
        with contextlib.suppress(OSError):
            claim_directory.rmdir()
        return False
    with contextlib.suppress(OSError):
        claim_directory.rmdir()
    return True


def write_immutable_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    identity = _file_identity(os.fstat(fd))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
    except Exception:
        _unlink_if_owned(path, identity)
        raise
    return identity


def _atomic_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False,
                                         prefix=".campaign-", suffix=".tmp") as fh:
            temporary = fh.name
            json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
        os.replace(temporary, str(path))
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


def _git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def repository_state():
    """Capture exact linked-worktree-safe repository and planning state."""
    git_sha = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    if not branch:
        raise RuntimeError("detached HEAD is not approved for campaign planning")
    upstream = _git(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    behind_text, ahead_text = _git(
        "rev-list", "--left-right", "--count", upstream + "...HEAD"
    ).split()
    raw = [
        line
        for line in _git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    ]
    return {
        "git_sha": git_sha,
        "branch": branch,
        "upstream": upstream,
        "ahead": int(ahead_text),
        "behind": int(behind_text),
        "training_lock_sha256": sha256_file(TRAINING_LOCK),
        "qiskit_lock_sha256": sha256_file(QISKIT_LOCK),
        "workers_resolved": _resolved_workers(0),
        "dirty": bool(raw),
        "dirty_policy_passed": not raw,
        "dirty_reasons": raw,
    }


def current_launch_state():
    """Recapture the complete repository and worker state before launch."""
    return repository_state()


def expected_config(config_name, seed, epochs):
    return run_experiments._expected_config(config_name, seed, epochs)


def _expected_runner_argv(request):
    argv = ["python", "-m", "experiments.run_experiments", "--datasets",
            *request["datasets"], "--configs", *request["configs"], "--seeds",
            *[str(seed) for seed in request["seeds"]], "--samples", str(request["samples"]),
            "--epochs", str(request["epochs"]), "--jobs", str(request["jobs_requested"])]
    if not request["with_baselines"]:
        argv.append("--no-baselines")
    return argv


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_safe_path_component(value):
    return (
        isinstance(value, str) and bool(value) and value not in {".", ".."} and
        not Path(value).is_absolute() and len(Path(value).parts) == 1 and
        "/" not in value and "\\" not in value and
        all(character.isalnum() or character in "._-" for character in value)
    )


def validate_manifest(manifest):
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if manifest.get("schema") != {"name": "fqcnn_campaign_manifest", "version": 1}:
        errors.append("invalid manifest schema")
    if not _is_safe_path_component(manifest.get("campaign")):
        errors.append("invalid campaign name")
    if manifest.get("state") not in CAMPAIGN_STATES:
        errors.append("invalid campaign state")
    repo = manifest.get("repository")
    if not isinstance(repo, dict):
        errors.append("missing manifest field: repository")
    else:
        for key in ("git_sha", "branch", "upstream", "ahead", "behind", "dirty", "dirty_policy"):
            if key not in repo:
                errors.append("missing manifest field: repository." + key)
        if not isinstance(repo.get("git_sha"), str) or not repo.get("git_sha"):
            errors.append("invalid repository Git SHA")
        if not isinstance(repo.get("branch"), str) or not repo.get("branch"):
            errors.append("invalid repository branch")
        if not isinstance(repo.get("upstream"), str) or not repo.get("upstream"):
            errors.append("invalid repository upstream")
        if (type(repo.get("ahead")) is not int or repo.get("ahead", -1) < 0 or
                type(repo.get("behind")) is not int or repo.get("behind", -1) < 0):
            errors.append("invalid repository divergence")
        if not isinstance(repo.get("dirty"), bool):
            errors.append("invalid repository dirty state")
        policy = repo.get("dirty_policy")
        if (not isinstance(policy, dict) or
                set(policy) != {"passed", "reasons", "raw"} or
                policy.get("passed") is not True or
                policy.get("reasons") != [] or policy.get("raw") != []):
            errors.append("invalid dirty policy")
        if repo.get("dirty") is True:
            errors.append("working tree is dirty")
    locks = manifest.get("locks")
    if not isinstance(locks, dict):
        errors.append("missing manifest field: locks")
    else:
        for name in ("training", "qiskit"):
            lock = locks.get(name)
            if not isinstance(lock, dict) or not isinstance(lock.get("path"), str) or not _is_sha256(lock.get("sha256")):
                errors.append("invalid {} lock provenance".format(name))
    request = manifest.get("request")
    if not isinstance(request, dict):
        errors.append("missing manifest field: request")
        request = {}
    required_request = (
        "datasets", "class_pairs", "configs", "seeds", "split_policy",
        "samples", "epochs", "jobs_requested", "workers_resolved",
        "with_baselines", "scheduler_cells", "baseline_side_effect_cells",
        "total_costed_cells",
    )
    if any(key not in request for key in required_request):
        errors.append("incomplete campaign request")

    datasets = request.get("datasets")
    parsed_pairs = []
    datasets_valid = isinstance(datasets, list) and bool(datasets)
    if datasets_valid:
        for value in datasets:
            if not isinstance(value, str) or not value:
                datasets_valid = False
                break
            parts = value.split(",")
            if (len(parts) != 2 or any(not part.isdigit() for part in parts) or
                    parts[0] == parts[1]):
                datasets_valid = False
                break
            parsed_pairs.append([int(parts[0]), int(parts[1])])
    if not datasets_valid:
        errors.append("invalid datasets")
        datasets = []
    elif len(set(datasets)) != len(datasets):
        errors.append("duplicate datasets")

    pairs = request.get("class_pairs")
    pairs_valid = (
        isinstance(pairs, list) and
        all(
            isinstance(pair, list) and len(pair) == 2 and
            all(type(label) is int and label >= 0 for label in pair)
            for pair in pairs
        )
    )
    if not pairs_valid or pairs != parsed_pairs:
        errors.append("dataset/class-pair mismatch")
    if pairs_valid and len(set(map(tuple, pairs))) != len(pairs):
        errors.append("duplicate class_pairs")

    configs = request.get("configs")
    if (not isinstance(configs, list) or not configs or
            any(not isinstance(value, str) or not value for value in configs)):
        errors.append("invalid configs")
        configs = []
    elif len(set(configs)) != len(configs):
        errors.append("duplicate configs")

    seeds = request.get("seeds")
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int or seed < 0 for seed in seeds)):
        errors.append("invalid seeds")
        seeds = []
    elif len(set(seeds)) != len(seeds):
        errors.append("duplicate seeds")

    for key in ("samples", "epochs", "workers_resolved"):
        if type(request.get(key)) is not int or request.get(key, 0) <= 0:
            errors.append("invalid request " + key)
    if type(request.get("jobs_requested")) is not int or request.get("jobs_requested", -1) < 0:
        errors.append("invalid requested jobs")
    with_baselines = request.get("with_baselines")
    if type(with_baselines) is not bool:
        errors.append("invalid request with_baselines")
        with_baselines = False
    if not isinstance(request.get("split_policy"), str) or not request.get("split_policy"):
        errors.append("invalid request split_policy")
    if "n_qubits" in request and (
            type(request.get("n_qubits")) is not int or request.get("n_qubits", 0) <= 0):
        errors.append("invalid request n_qubits")
    elif with_baselines and "n_qubits" not in request:
        errors.append("invalid request n_qubits")

    scheduler_cells = len(datasets) * len(configs) * len(seeds)
    baseline_side_effect_cells = (
        len(datasets) * len(seeds)
        if with_baselines and "proposed" in configs else 0
    )
    total_costed_cells = scheduler_cells + baseline_side_effect_cells
    expected_counts = {
        "scheduler_cells": scheduler_cells,
        "baseline_side_effect_cells": baseline_side_effect_cells,
        "total_costed_cells": total_costed_cells,
    }
    for key, expected in expected_counts.items():
        if type(request.get(key)) is not int or request.get(key) != expected or expected < 0:
            errors.append("malformed campaign count: " + key)

    runner = manifest.get("runner")
    if isinstance(runner, dict):
        try:
            expected_argv = _expected_runner_argv(request)
        except (KeyError, TypeError):
            expected_argv = None
        if (expected_argv is None or runner.get("argv") != expected_argv or
                runner.get("command") != shlex.join(expected_argv)):
            errors.append("runner command/argv does not exactly match request")
    else:
        errors.append("missing manifest field: runner")
    cost = manifest.get("cost")
    if not isinstance(cost, dict) or not isinstance(cost.get("path"), str) or not _is_sha256(cost.get("sha256")):
        errors.append("invalid cost provenance")
    approval = manifest.get("approval")
    if (not isinstance(approval, dict) or approval.get("required") is not True or
            not isinstance(approval.get("artifact"), str) or
            set(approval) != {"required", "artifact"}):
        errors.append("invalid approval descriptor")
    errors.extend(_path_errors(manifest))
    return errors


def _path_errors(manifest):
    roots = manifest.get("output_roots")
    artifacts = manifest.get("artifacts")
    if not isinstance(roots, dict) or not isinstance(artifacts, dict):
        return ["invalid output/artifact paths"]
    errors = []
    try:
        campaign_name = manifest["campaign"]
        if not _is_safe_path_component(campaign_name):
            return ["campaign path is not a safe component"]
        configured_campaign = CAMPAIGN_ROOT / campaign_name
        resolved_base = CAMPAIGN_ROOT.resolve()
        resolved_campaign = resolved_base / campaign_name
        if resolved_campaign.parent != resolved_base:
            errors.append("campaign root must be beneath canonical campaign root")

        if roots["campaign"] != str(configured_campaign):
            errors.append("campaign root path string is not canonical")
        if Path(roots["campaign"]).resolve() != resolved_campaign:
            errors.append("campaign root is not canonical")

        configured_roots = {
            "experiments": configured_campaign / "experiments",
            "runs": configured_campaign / "runs",
            "manifests": configured_campaign / "manifests",
        }
        resolved_roots = {}
        for name, configured in configured_roots.items():
            if roots[name] != str(configured):
                errors.append("output root path string mismatch: " + name)
            resolved = Path(roots[name]).resolve()
            resolved_roots[name] = resolved
            if resolved != resolved_campaign / name:
                errors.append("output root path mismatch: " + name)
        if len(set(resolved_roots.values())) != len(resolved_roots):
            errors.append("output roots must not alias")

        configured_artifacts = {
            "manifest": configured_campaign / "manifest.json",
            "cost_estimate": configured_campaign / "cost_estimate.json",
            "full_suite": configured_campaign / "full_suite.json",
            "approval": configured_campaign / "approval.json",
            "launch": configured_campaign / "launch.json",
            "status": configured_campaign / "status.json",
            "failures": configured_campaign / "failures.json",
            "scheduler_failures": configured_campaign / "experiments" / "failures.json",
        }
        resolved_artifacts = {
            name: resolved_campaign / configured.relative_to(configured_campaign)
            for name, configured in configured_artifacts.items()
        }
        for name, configured in configured_artifacts.items():
            if artifacts[name] != str(configured):
                errors.append("artifact path string mismatch: " + name)
            if Path(artifacts[name]).resolve() != resolved_artifacts[name]:
                errors.append("artifact path mismatch: " + name)
        if manifest["cost"]["path"] != artifacts["cost_estimate"]:
            errors.append("cost path does not exactly match cost artifact path")
        if manifest["approval"]["artifact"] != artifacts["approval"]:
            errors.append("approval path does not exactly match approval artifact path")
    except (KeyError, OSError, TypeError, ValueError):
        errors.append("invalid output/artifact paths")
    return errors


def validate_cost(manifest, cost):
    errors = []
    if not isinstance(cost, dict):
        return ["cost estimate must be an object"]
    if cost.get("schema") != {"name": "fqcnn_campaign_cost_estimate", "version": 1}:
        errors.append("invalid cost estimate schema")
    if (cost.get("status") != "approved" or
            not isinstance(cost.get("approval"), dict) or
            cost["approval"].get("approved") is not True):
        errors.append("cost estimate is not approved")

    req = manifest.get("request", {})
    expected_request = {
        "datasets": req.get("datasets"),
        "configs": req.get("configs"),
        "seeds": req.get("seeds"),
        "samples": req.get("samples"),
        "epochs": req.get("epochs"),
        "jobs_requested": req.get("jobs_requested"),
        "with_baselines": req.get("with_baselines"),
    }
    if cost.get("request") != expected_request:
        errors.append("cost estimate request does not exactly match campaign")

    expected_counts = {
        "requested_configs": len(req.get("configs", [])),
        "measurable_configs": len(req.get("configs", [])),
        "scheduler_cells": req.get("scheduler_cells"),
        "baseline_side_effect_cells": req.get("baseline_side_effect_cells"),
        "total_costed_cells": req.get("total_costed_cells"),
        "measurable_scheduler_cells": req.get("scheduler_cells"),
        "measurable_baseline_side_effect_cells": req.get(
            "baseline_side_effect_cells"
        ),
        "measurable_total_costed_cells": req.get("total_costed_cells"),
        "failed_calibrations": 0,
        "total_failures": 0,
    }
    counts = cost.get("counts")
    if (not isinstance(counts, dict) or counts != expected_counts or
            cost.get("failures") != []):
        errors.append("cost estimate is incomplete")

    projection = cost.get("projection")
    if not isinstance(projection, dict):
        errors.append("cost estimate has no projection")
    else:
        for key in ("serial_hours", "wall_hours", "budget_fraction"):
            value = projection.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                errors.append("invalid or non-finite cost projection field: " + key)
        budget = projection.get("budget_hours")
        if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
            errors.append("invalid or non-finite cost projection field: budget_hours")
        workers = projection.get("workers")
        if type(workers) is not int or workers <= 0:
            errors.append("invalid cost projection field: workers")
        elif workers != req.get("workers_resolved"):
            errors.append("cost worker policy does not match campaign")
        wall = projection.get("wall_hours")
        if (type(wall) in (int, float) and not isinstance(wall, bool) and
                type(budget) in (int, float) and not isinstance(budget, bool) and
                math.isfinite(wall) and math.isfinite(budget) and wall > budget):
            errors.append("cost estimate is over budget")
    return errors


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _valid_aware_timestamp(value):
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _full_suite_errors(manifest):
    try:
        path = Path(manifest["artifacts"]["full_suite"])
    except (KeyError, TypeError):
        return ["missing full-suite evidence"]
    if not path.is_file():
        return ["missing full-suite evidence"]
    try:
        record = _load_json(path)
    except (OSError, ValueError):
        return ["invalid full-suite evidence"]
    if not isinstance(record, dict):
        return ["invalid full-suite evidence"]
    errors = []
    if record.get("schema") != {"name": "fqcnn_full_suite_evidence", "version": 1}:
        errors.append("invalid full-suite evidence schema")
    if record.get("command") != "python -m pytest tests/ -q":
        errors.append("full-suite evidence command mismatch")
    tests = record.get("tests", {})
    if (not isinstance(tests, dict) or
            type(tests.get("passed")) is not int or tests.get("passed", 0) <= 0 or
            tests.get("failed") != 0 or tests.get("total") != tests.get("passed")):
        errors.append("full-suite evidence totals are incomplete")
    if record.get("status") != "pass" or record.get("exit_code") != 0:
        errors.append("full-suite evidence did not pass")
    if record.get("git_sha") != manifest.get("repository", {}).get("git_sha"):
        errors.append("full-suite evidence Git SHA mismatch")
    if (record.get("training_lock_sha256") !=
            manifest.get("locks", {}).get("training", {}).get("sha256")):
        errors.append("full-suite evidence training-lock mismatch")
    if record.get("dirty_policy", {}).get("passed") is not True:
        errors.append("full-suite evidence dirty policy failed")
    return errors


def approval_record_errors(manifest, record):
    if not isinstance(record, dict):
        return ["approval record must be an object"]
    errors = []
    required_fields = {
        "schema", "approved", "campaign", "git_sha", "manifest_sha256",
        "cost_sha256", "full_suite_sha256", "approver", "approved_at_utc",
        "scope",
    }
    if set(record) != required_fields:
        errors.append("invalid approval record shape")
    if record.get("schema") != {
            "name": "fqcnn_campaign_launch_approval", "version": 1}:
        errors.append("invalid approval record schema")
    if record.get("approved") is not True:
        errors.append("approval is not exactly true")
    if record.get("campaign") != manifest.get("campaign"):
        errors.append("approval campaign mismatch")
    if record.get("git_sha") != manifest.get("repository", {}).get("git_sha"):
        errors.append("approval Git SHA mismatch")

    artifacts = manifest.get("artifacts", {})
    manifest_path = Path(artifacts.get("manifest", ""))
    if (not manifest_path.is_file() or
            record.get("manifest_sha256") != sha256_file(manifest_path)):
        errors.append("approval manifest hash mismatch")
    cost_path = Path(artifacts.get("cost_estimate", ""))
    if (not cost_path.is_file() or
            record.get("cost_sha256") != manifest.get("cost", {}).get("sha256") or
            record.get("cost_sha256") != sha256_file(cost_path)):
        errors.append("approval cost hash mismatch")
    suite_path = Path(artifacts.get("full_suite", ""))
    if (not suite_path.is_file() or
            record.get("full_suite_sha256") != sha256_file(suite_path)):
        errors.append("full-suite evidence hash mismatch")
    errors.extend(_full_suite_errors(manifest))

    if not isinstance(record.get("approver"), str) or not record["approver"].strip():
        errors.append("missing approval approver")
    if not _valid_aware_timestamp(record.get("approved_at_utc")):
        errors.append("invalid approval timestamp")
    request = manifest.get("request", {})
    scope_fields = (
        "datasets", "configs", "seeds", "samples", "epochs",
        "jobs_requested", "workers_resolved", "with_baselines",
        "scheduler_cells", "baseline_side_effect_cells", "total_costed_cells",
    )
    expected_scope = {key: request.get(key) for key in scope_fields}
    if record.get("scope") != expected_scope:
        errors.append("approval scope does not exactly match campaign")
    return errors


def _approval_errors(manifest):
    path_errors = _path_errors(manifest)
    if path_errors:
        return path_errors
    path = Path(manifest["artifacts"]["approval"])
    if not path.is_file():
        return ["missing launch approval record"]
    try:
        record = _load_json(path)
    except (OSError, ValueError):
        return ["invalid launch approval record"]
    return approval_record_errors(manifest, record)


def launch_record_errors(manifest, record):
    if not isinstance(record, dict):
        return ["launch record must be an object"]
    errors = []
    required_fields = {
        "schema", "campaign", "git_sha", "manifest_sha256", "cost_sha256",
        "full_suite_sha256", "approval_sha256", "queue", "queue_lease_id",
        "owner_pid", "owner_host", "launched_at_utc",
    }
    if set(record) != required_fields:
        errors.append("invalid launch record shape")
    if record.get("schema") != {"name": "fqcnn_campaign_launch", "version": 1}:
        errors.append("invalid launch record schema")
    if record.get("campaign") != manifest.get("campaign"):
        errors.append("launch campaign mismatch")
    if record.get("git_sha") != manifest.get("repository", {}).get("git_sha"):
        errors.append("launch Git SHA mismatch")

    artifacts = manifest.get("artifacts", {})
    identities = (
        ("manifest_sha256", artifacts.get("manifest"), None),
        ("cost_sha256", artifacts.get("cost_estimate"),
         manifest.get("cost", {}).get("sha256")),
        ("full_suite_sha256", artifacts.get("full_suite"), None),
        ("approval_sha256", artifacts.get("approval"), None),
    )
    for field, path_value, expected in identities:
        path = Path(path_value or "")
        if (not path.is_file() or not _is_sha256(record.get(field)) or
                record.get(field) != sha256_file(path) or
                (expected is not None and record.get(field) != expected)):
            errors.append("launch {} mismatch".format(field))
    if record.get("queue") != str(QUEUE_PATH):
        errors.append("launch queue path mismatch")
    if (not isinstance(record.get("queue_lease_id"), str) or
            not record["queue_lease_id"]):
        errors.append("invalid launch queue lease")
    if type(record.get("owner_pid")) is not int or record["owner_pid"] <= 0:
        errors.append("invalid launch owner PID")
    if not isinstance(record.get("owner_host"), str) or not record["owner_host"]:
        errors.append("invalid launch owner host")
    if not _valid_aware_timestamp(record.get("launched_at_utc")):
        errors.append("invalid launch timestamp")
    errors.extend(_approval_errors(manifest))
    return errors


def _load_valid_launch_record(manifest):
    try:
        path = Path(manifest["artifacts"]["launch"])
    except (KeyError, TypeError):
        return None
    if not path.is_file():
        return None
    try:
        record = _load_json(path)
    except (OSError, ValueError):
        return None
    return record if not launch_record_errors(manifest, record) else None


def _build_launch_record(manifest, approval_sha256, queue_owner):
    return {
        "schema": {"name": "fqcnn_campaign_launch", "version": 1},
        "campaign": manifest["campaign"],
        "git_sha": manifest["repository"]["git_sha"],
        "manifest_sha256": sha256_file(manifest["artifacts"]["manifest"]),
        "cost_sha256": manifest["cost"]["sha256"],
        "full_suite_sha256": sha256_file(manifest["artifacts"]["full_suite"]),
        "approval_sha256": approval_sha256,
        "queue": str(QUEUE_PATH),
        "queue_lease_id": queue_owner["lease_id"],
        "owner_pid": queue_owner["pid"],
        "owner_host": queue_owner["host"],
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _output_errors(manifest):
    artifacts = manifest["artifacts"]
    roots = manifest["output_roots"]
    launch_path = Path(artifacts["launch"])
    if launch_path.exists():
        if _load_valid_launch_record(manifest) is not None:
            return ["campaign already launched; use status/resume behavior"]
        return ["artifact collision: incompatible existing launch artifact"]
    errors = []
    for name in ("experiments", "runs", "manifests"):
        if Path(roots[name]).exists():
            errors.append("output collision: " + name)
    for name in ("launch", "status", "failures"):
        if Path(artifacts[name]).exists():
            errors.append("artifact collision: " + name)
    return errors


def launch_gate_errors(manifest, current=None, queue_path=None):
    manifest_errors = validate_manifest(manifest)
    errors = list(manifest_errors)
    queue_path = QUEUE_PATH if queue_path is None else queue_path
    if Path(queue_path).exists():
        errors.append("global training/hardware queue is occupied")
    if manifest_errors:
        return errors
    current = current_launch_state() if current is None else current
    if not isinstance(current, dict):
        return errors + ["invalid current launch state"]
    repo = manifest["repository"]
    locks = manifest["locks"]
    if current.get("git_sha") != repo["git_sha"]:
        errors.append("wrong Git SHA")
    provenance_fields = ("branch", "upstream", "ahead", "behind")
    if any(current.get(key) != repo.get(key) for key in provenance_fields):
        errors.append("repository provenance changed")
    if current.get("training_lock_sha256") != locks["training"]["sha256"]:
        errors.append("wrong training lock hash")
    if current.get("qiskit_lock_sha256") != locks["qiskit"]["sha256"]:
        errors.append("wrong Qiskit lock hash")
    if current.get("workers_resolved") != manifest["request"]["workers_resolved"]:
        errors.append("resolved worker count changed")
    if (current.get("dirty") is not False or
            current.get("dirty_policy_passed") is not True or
            current.get("dirty_reasons") != [] or repo.get("dirty")):
        errors.append("working tree is dirty")

    errors.extend(_full_suite_errors(manifest))
    cost_path = Path(manifest["cost"]["path"])
    if not cost_path.is_file():
        errors.append("missing cost estimate")
    else:
        if sha256_file(cost_path) != manifest["cost"]["sha256"]:
            errors.append("cost estimate hash mismatch")
        try:
            errors.extend(validate_cost(manifest, _load_json(cost_path)))
        except (OSError, ValueError):
            errors.append("invalid cost estimate")
    errors.extend(_approval_errors(manifest))
    errors.extend(_output_errors(manifest))
    req = manifest["request"]
    if req.get("with_baselines") and req.get("n_qubits") == 10:
        evidence = manifest.get("approvals", {}).get("baseline_schedule", {})
        if evidence.get("validated") is not True:
            errors.append("missing validated baseline schedule evidence")
    return errors


def validate_campaign(manifest, current=None):
    return launch_gate_errors(
        manifest,
        current=current if current is not None else current_launch_state(),
    )


def _cells(manifest):
    req = manifest["request"]
    for dataset in req["datasets"]:
        dataset_id = dataset.replace(",", "v")
        for config in req["configs"]:
            for seed in req["seeds"]:
                yield dataset_id, config, seed


def _valid_cell_count(manifest):
    valid = 0
    for dataset, config, seed in _cells(manifest):
        directory = run_artifacts.run_dir(
            dataset, config, seed,
            root=manifest["output_roots"]["runs"], create=False,
        )
        if run_artifacts.is_reusable(
                directory,
                config=expected_config(config, seed, manifest["request"]["epochs"]),
                seed=seed):
            valid += 1
    return valid


def campaign_state(manifest, launch_record=None, failure_evidence=None,
                   queue_owned=False):
    if launch_record is None:
        return "pending"
    if queue_owned:
        return "running"
    evidence = failure_evidence or {
        "available": False, "valid": False, "failures": [], "errors": []
    }
    if evidence.get("available") and not evidence.get("valid"):
        return "failed"
    if evidence.get("valid") and evidence.get("failures"):
        return "failed"
    expected = manifest["request"]["scheduler_cells"]
    if (_valid_cell_count(manifest) == expected and
            evidence.get("available") and evidence.get("valid") and
            evidence.get("failures") == []):
        return "complete"
    return "running"


def _runner_argv():
    return ["python", "-m", "experiments.run_experiments", "--datasets", *FIRST_DATASETS,
            "--configs", *FIRST_CONFIGS, "--seeds", *[str(s) for s in FIRST_SEEDS],
            "--samples", "400", "--epochs", "30", "--jobs", "0", "--no-baselines"]


def _resolved_workers(requested):
    return requested if requested > 0 else max(1, (os.cpu_count() or 3) - 2)


def _cost_command(path):
    return ["python", "-m", "experiments.estimate_cost", "--datasets", *FIRST_DATASETS,
            "--configs", *FIRST_CONFIGS, "--seeds", *[str(s) for s in FIRST_SEEDS],
            "--samples", "400", "--epochs", "30", "--jobs", "0", "--no-baselines",
            "--output-json", str(path)]


def _validate_planning_snapshot(snapshot, provenance_policy):
    required = {
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
    if not isinstance(snapshot, dict) or set(snapshot) != set(required):
        raise RuntimeError("repository snapshot has an invalid shape")
    for key, expected_type in required.items():
        if type(snapshot[key]) is not expected_type:
            raise RuntimeError("repository snapshot has an invalid field: " + key)
    if (not snapshot["git_sha"] or not snapshot["branch"] or
            not snapshot["upstream"] or
            not _is_sha256(snapshot["training_lock_sha256"]) or
            not _is_sha256(snapshot["qiskit_lock_sha256"]) or
            snapshot["workers_resolved"] <= 0 or snapshot["ahead"] < 0 or
            snapshot["behind"] < 0):
        raise RuntimeError("repository snapshot is incomplete")
    if snapshot["dirty"] or not snapshot["dirty_policy_passed"] or snapshot["dirty_reasons"]:
        raise RuntimeError("working tree is dirty")

    policy_types = {
        "git_sha": str,
        "branch": str,
        "upstream": str,
        "ahead": int,
        "behind": int,
    }
    if (not isinstance(provenance_policy, dict) or
            set(provenance_policy) != set(policy_types)):
        raise RuntimeError("reviewed provenance policy is required")
    for key, expected_type in policy_types.items():
        if type(provenance_policy[key]) is not expected_type:
            raise RuntimeError("reviewed provenance policy has an invalid field: " + key)
    if (not provenance_policy["git_sha"] or not provenance_policy["branch"] or
            not provenance_policy["upstream"] or provenance_policy["ahead"] < 0 or
            provenance_policy["behind"] < 0):
        raise RuntimeError("reviewed provenance policy is incomplete")
    if any(snapshot[key] != provenance_policy[key] for key in policy_types):
        raise RuntimeError("reviewed provenance policy does not match repository snapshot")


def plan(campaign_name: str, repository_snapshot: dict = None,
         provenance_policy: dict = None) -> dict:
    if campaign_name != FIRST_CAMPAIGN:
        raise ValueError("only the reviewed first campaign is supported")
    directory = CAMPAIGN_ROOT / campaign_name
    manifest_path = directory / "manifest.json"
    cost_path = directory / "cost_estimate.json"
    owned_paths = []
    created_directory = False
    old_argv = sys.argv[:]
    try:
        directory.mkdir(parents=True, exist_ok=False)
        created_directory = True
        snapshot = (
            repository_snapshot
            if repository_snapshot is not None
            else repository_state()
        )
        _validate_planning_snapshot(snapshot, provenance_policy)
        from experiments import estimate_cost
        cost_identity = None

        def publish_cost(path, payload):
            nonlocal cost_identity
            if Path(path) != cost_path:
                raise RuntimeError("cost publisher received a noncanonical path")
            if cost_identity is not None:
                raise RuntimeError("cost publisher was invoked more than once")
            cost_identity = write_immutable_json(cost_path, payload)
            owned_paths.append((cost_path, cost_identity))

        sys.argv = ["estimate_cost"] + _cost_command(cost_path)[3:]
        rc = estimate_cost.main(json_writer=publish_cost)
        if cost_identity is None:
            raise RuntimeError("cost estimate was not published exclusively")
        if rc != 0:
            raise RuntimeError("cost estimate was not approved")

        roots = {
            "campaign": str(directory),
            "experiments": str(directory / "experiments"),
            "runs": str(directory / "runs"),
            "manifests": str(directory / "manifests"),
        }
        artifacts = {
            "manifest": str(manifest_path),
            "cost_estimate": str(cost_path),
            "full_suite": str(directory / "full_suite.json"),
            "approval": str(directory / "approval.json"),
            "launch": str(directory / "launch.json"),
            "status": str(directory / "status.json"),
            "failures": str(directory / "failures.json"),
            "scheduler_failures": str(directory / "experiments" / "failures.json"),
        }
        argv = _runner_argv()
        scheduler_cells = len(FIRST_DATASETS) * len(FIRST_CONFIGS) * len(FIRST_SEEDS)
        manifest = {
            "schema": {"name": "fqcnn_campaign_manifest", "version": 1},
            "campaign": campaign_name,
            "state": "pending",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner": {"command": shlex.join(argv), "argv": argv},
            "repository": {
                "git_sha": snapshot["git_sha"],
                "branch": snapshot["branch"],
                "upstream": snapshot["upstream"],
                "ahead": snapshot["ahead"],
                "behind": snapshot["behind"],
                "dirty": snapshot["dirty"],
                "dirty_policy": {
                    "passed": snapshot["dirty_policy_passed"],
                    "reasons": snapshot["dirty_reasons"],
                    "raw": snapshot["dirty_reasons"],
                },
            },
            "locks": {
                "training": {
                    "path": str(TRAINING_LOCK),
                    "sha256": snapshot["training_lock_sha256"],
                },
                "qiskit": {
                    "path": str(QISKIT_LOCK),
                    "sha256": snapshot["qiskit_lock_sha256"],
                },
            },
            "request": {
                "datasets": FIRST_DATASETS,
                "class_pairs": [[0, 1], [3, 5], [4, 9], [5, 8]],
                "configs": FIRST_CONFIGS,
                "seeds": FIRST_SEEDS,
                "split_policy": "deterministic stratified manifest per dataset and seed",
                "samples": 400,
                "epochs": 30,
                "jobs_requested": 0,
                "workers_resolved": snapshot["workers_resolved"],
                "with_baselines": False,
                "scheduler_cells": scheduler_cells,
                "baseline_side_effect_cells": 0,
                "total_costed_cells": scheduler_cells,
            },
            "cost": {
                "path": str(cost_path),
                "sha256": sha256_file(cost_path),
                "projected_resource_use": _load_json(cost_path)["projection"],
            },
            "output_roots": roots,
            "artifacts": artifacts,
            "approval": {"required": True, "artifact": artifacts["approval"]},
        }
        manifest_errors = validate_manifest(manifest)
        if manifest_errors:
            raise RuntimeError(
                "planned manifest is invalid: " + "; ".join(manifest_errors)
            )
        manifest_identity = write_immutable_json(manifest_path, manifest)
        owned_paths.append((manifest_path, manifest_identity))
        return manifest
    except Exception:
        if created_directory:
            for path, identity in reversed(owned_paths):
                _unlink_if_owned(path, identity)
            with contextlib.suppress(OSError):
                directory.rmdir()
        raise
    finally:
        sys.argv = old_argv


def _load_campaign(name):
    return _load_json(CAMPAIGN_ROOT / name / "manifest.json")


def _acquire_queue(campaign_name):
    return exclusive_queue.acquire(
        QUEUE_PATH,
        {"kind": "campaign", "campaign": campaign_name},
    )


def launch(manifest):
    errors = launch_gate_errors(manifest)
    if errors:
        return errors
    # Mutable approval, repository, worker, output, and queue inputs are checked
    # again at the final boundary immediately before the atomic queue claim.
    errors = launch_gate_errors(manifest)
    if errors:
        return errors

    lease = _acquire_queue(manifest["campaign"])
    old_argv = sys.argv[:]
    old_roots = (run_experiments.EXP_ROOT, run_experiments.FAILURE_MANIFEST,
                 run_experiments.MANIFEST_ROOT, run_artifacts.RUN_ROOT)
    try:
        approval_sha256 = sha256_file(manifest["artifacts"]["approval"])
        write_immutable_json(
            manifest["artifacts"]["launch"],
            _build_launch_record(manifest, approval_sha256, lease),
        )
        roots = {
            "experiments": manifest["output_roots"]["experiments"],
            "runs": manifest["output_roots"]["runs"],
            "manifests": manifest["output_roots"]["manifests"],
            "failures": manifest["artifacts"]["scheduler_failures"],
        }
        sys.argv = manifest["runner"]["argv"][2:]
        run_experiments.main(output_roots=roots)
    finally:
        sys.argv = old_argv
        (run_experiments.EXP_ROOT, run_experiments.FAILURE_MANIFEST,
         run_experiments.MANIFEST_ROOT, run_artifacts.RUN_ROOT) = old_roots
        exclusive_queue.release(QUEUE_PATH, lease)
    return []


def _scheduler_failure_evidence(manifest):
    path = Path(manifest["artifacts"]["scheduler_failures"])
    if not path.is_file():
        return {
            "available": False,
            "valid": False,
            "failures": [],
            "errors": ["missing final scheduler failure manifest"],
        }
    invalid = {
        "available": True,
        "valid": False,
        "failures": [{"error": "invalid scheduler failure manifest"}],
        "errors": ["invalid scheduler failure manifest"],
    }
    try:
        payload = _load_json(path)
    except (OSError, ValueError):
        return invalid
    if not isinstance(payload, dict):
        return invalid
    failures = payload.get("failures")
    if (not isinstance(failures, list) or
            type(payload.get("n_failed")) is not int or
            payload["n_failed"] != len(failures) or
            set(payload) != {"n_failed", "failures"}):
        return invalid
    return {
        "available": True,
        "valid": True,
        "failures": failures,
        "errors": [],
    }


def _campaign_queue_status(manifest, launch_record):
    if not Path(QUEUE_PATH).exists():
        return False, []
    try:
        owner = exclusive_queue.read(QUEUE_PATH)
    except (OSError, TypeError, ValueError):
        return False, ["invalid global training/hardware queue evidence"]
    expected = {
        "campaign": manifest["campaign"],
        "lease_id": launch_record["queue_lease_id"],
        "pid": launch_record["owner_pid"],
        "host": launch_record["owner_host"],
    }
    if all(owner.get(key) == value for key, value in expected.items()):
        return True, []
    return False, ["global training/hardware queue is owned by another launch"]


def status_campaign(manifest):
    launch_record = _load_valid_launch_record(manifest)
    if launch_record is None:
        return {
            "schema": {"name": "fqcnn_campaign_status", "version": 1},
            "campaign": manifest["campaign"],
            "state": "pending",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "valid_cells": 0,
            "expected_cells": manifest["request"]["scheduler_cells"],
            "queue_owned": False,
            "scheduler_failure_available": False,
            "scheduler_failure_valid": False,
            "scheduler_failure_count": 0,
            "errors": [],
        }

    evidence = _scheduler_failure_evidence(manifest)
    queue_owned, queue_errors = _campaign_queue_status(manifest, launch_record)
    valid_cells = _valid_cell_count(manifest)
    state = campaign_state(
        manifest,
        launch_record=launch_record,
        failure_evidence=evidence,
        queue_owned=queue_owned,
    )
    errors = list(queue_errors) + list(evidence["errors"])
    if queue_errors:
        state = "failed"
    payload = {
        "schema": {"name": "fqcnn_campaign_status", "version": 1},
        "campaign": manifest["campaign"],
        "state": state,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid_cells": valid_cells,
        "expected_cells": manifest["request"]["scheduler_cells"],
        "queue_owned": queue_owned,
        "scheduler_failure_available": evidence["available"],
        "scheduler_failure_valid": evidence["valid"],
        "scheduler_failure_count": len(evidence["failures"]),
        "errors": errors,
    }
    _atomic_json(manifest["artifacts"]["status"], payload)
    _atomic_json(
        manifest["artifacts"]["failures"],
        {"n_failed": len(evidence["failures"]),
         "failures": evidence["failures"], "errors": errors},
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan_command = sub.add_parser("plan")
    plan_command.add_argument("--campaign", required=True)
    plan_command.add_argument("--provenance-policy", required=True)
    for action in ("validate", "launch", "status"):
        command = sub.add_parser(action)
        command.add_argument("--campaign", required=True)
    args = parser.parse_args()
    if args.action == "plan":
        manifest = plan(
            args.campaign,
            provenance_policy=_load_json(args.provenance_policy),
        )
        print("planned {} at {}".format(args.campaign, manifest["artifacts"]["manifest"]))
        return 0
    manifest = _load_campaign(args.campaign)
    if args.action == "validate":
        errors = validate_campaign(manifest)
        if errors:
            for error in errors:
                print("BLOCKED: " + error)
            return 1
        print("VALID")
        return 0
    if args.action == "launch":
        errors = launch(manifest)
        if errors:
            for error in errors:
                print("BLOCKED: " + error)
            return 1
        return 0
    state = status_campaign(manifest)["state"]
    print(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
