#!/usr/bin/env python3
"""Immutable provenance and fail-closed gates for experiment campaigns."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from QCNN.utils import run_artifacts
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


def write_immutable_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


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


def _read_git_ref(ref):
    git_dir = Path(".git")
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text().strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                sha, name = line.split(" ", 1)
                if name == ref:
                    return sha
    raise RuntimeError("cannot resolve Git ref " + ref)


def repository_state():
    """Capture non-mutating Git identity without starting a subprocess.

    A complete index/worktree comparison requires Git. Planning is forbidden from
    starting any subprocess, so cleanliness and divergence deliberately remain
    unproven and therefore fail closed until launch-time revalidation.
    """
    head = Path(".git/HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        branch = ref.rsplit("/", 1)[-1]
        sha = _read_git_ref(ref)
    else:
        branch, sha = None, head
    return {"git_sha": sha, "branch": branch, "upstream": None,
            "ahead": None, "behind": None, "dirty": True,
            "dirty_policy": {
                "passed": False,
                "reasons": ["planning cannot prove a clean tree without a forbidden Git subprocess"],
                "raw": [],
            }}


def expected_config(config_name, seed, epochs):
    return run_experiments._expected_config(config_name, seed, epochs)


def validate_manifest(manifest):
    errors = []
    required = [
        ("schema",), ("campaign",), ("state",), ("runner", "command"),
        ("runner", "argv"), ("repository", "git_sha"), ("repository", "dirty"),
        ("repository", "dirty_policy"), ("locks", "training"),
        ("request", "split_policy"), ("request", "expected_cells"),
        ("cost", "path"), ("cost", "sha256"), ("output_roots",),
        ("artifacts",), ("approvals", "full_suite"), ("approvals", "launch"),
    ]
    for keys in required:
        value = manifest
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                errors.append("missing manifest field: " + ".".join(keys))
                break
            value = value[key]
    if manifest.get("schema") != {"name": "fqcnn_campaign_manifest", "version": 1}:
        errors.append("invalid manifest schema")
    if manifest.get("state") not in CAMPAIGN_STATES:
        errors.append("invalid campaign state")
    request = manifest.get("request", {})
    calculated = (len(request.get("datasets", [])) * len(request.get("configs", [])) *
                  len(request.get("seeds", [])))
    if request.get("expected_cells") != calculated or calculated <= 0:
        errors.append("malformed expected-cell count")
    return errors


def validate_cost(manifest, cost):
    errors = []
    if cost.get("schema") != {"name": "fqcnn_campaign_cost_estimate", "version": 1}:
        errors.append("invalid cost estimate schema")
    if cost.get("status") != "approved" or cost.get("approval", {}).get("approved") is not True:
        errors.append("cost estimate is not approved")
    counts = cost.get("counts", {})
    expected = manifest.get("request", {}).get("expected_cells")
    if (counts.get("requested_cells") != expected or counts.get("measurable_cells") != expected or
            counts.get("total_failures") != 0 or cost.get("failures")):
        errors.append("cost estimate is incomplete")
    req = manifest.get("request", {})
    expected_request = {"datasets": req.get("datasets"), "configs": req.get("configs"),
                        "seeds": req.get("seeds"), "samples": req.get("samples"),
                        "epochs": req.get("epochs"), "with_baselines": req.get("with_baselines")}
    if cost.get("request") != expected_request:
        errors.append("cost estimate request does not exactly match campaign")
    projection = cost.get("projection")
    if not isinstance(projection, dict):
        errors.append("cost estimate has no projection")
    else:
        for key, value in projection.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                errors.append("cost estimate contains non-finite projection")
                break
        if projection.get("workers") != req.get("workers_resolved"):
            errors.append("cost worker policy does not match campaign")
        if projection.get("wall_hours", math.inf) > projection.get("budget_hours", -math.inf):
            errors.append("cost estimate is over budget")
    return errors


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _full_suite_errors(manifest):
    path = Path(manifest.get("approvals", {}).get("full_suite", {}).get("path", ""))
    if not path.is_file():
        return ["missing full-suite evidence"]
    try:
        record = _load_json(path)
    except (OSError, ValueError):
        return ["invalid full-suite evidence"]
    errors = []
    if record.get("schema") != {"name": "fqcnn_full_suite_evidence", "version": 1}:
        errors.append("invalid full-suite evidence schema")
    if record.get("status") != "pass" or record.get("exit_code") != 0:
        errors.append("full-suite evidence did not pass")
    if record.get("git_sha") != manifest.get("repository", {}).get("git_sha"):
        errors.append("full-suite evidence Git SHA mismatch")
    if record.get("training_lock_sha256") != manifest.get("locks", {}).get("training", {}).get("sha256"):
        errors.append("full-suite evidence training-lock mismatch")
    if record.get("dirty_policy", {}).get("passed") is not True:
        errors.append("full-suite evidence dirty policy failed")
    return errors


def launch_gate_errors(manifest, current=None, queue_path=QUEUE_PATH):
    errors = validate_manifest(manifest)
    if errors:
        return errors
    current = current or {
        "git_sha": _git("rev-parse", "HEAD"),
        "training_lock_sha256": sha256_file(TRAINING_LOCK),
        "qiskit_lock_sha256": sha256_file(QISKIT_LOCK),
        "dirty": repository_state()["dirty"],
        "dirty_policy_passed": repository_state()["dirty_policy"]["passed"],
    }
    repo = manifest["repository"]
    locks = manifest["locks"]
    if current["git_sha"] != repo["git_sha"]:
        errors.append("wrong Git SHA")
    if current["training_lock_sha256"] != locks["training"]["sha256"]:
        errors.append("wrong training lock hash")
    if current["qiskit_lock_sha256"] != locks["qiskit"]["sha256"]:
        errors.append("wrong Qiskit lock hash")
    if current.get("dirty") or not current.get("dirty_policy_passed") or repo.get("dirty"):
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
        except ValueError:
            errors.append("invalid cost estimate")
    if Path(queue_path).exists():
        errors.append("global training/hardware queue is occupied")
    for name in ("experiments", "runs", "manifests"):
        root = Path(manifest["output_roots"][name])
        if root.exists():
            errors.append("output collision: " + name)
    req = manifest["request"]
    if req.get("with_baselines") and req.get("n_qubits") == 10:
        evidence = manifest.get("approvals", {}).get("baseline_schedule", {})
        if evidence.get("validated") is not True:
            errors.append("missing validated baseline schedule evidence")
    if manifest.get("approvals", {}).get("launch", {}).get("approved") is not True:
        errors.append("missing launch approval record")
    return errors


def validate_campaign(manifest, current=None):
    if current is None:
        current = {
            "git_sha": manifest.get("repository", {}).get("git_sha"),
            "training_lock_sha256": sha256_file(TRAINING_LOCK),
            "qiskit_lock_sha256": sha256_file(QISKIT_LOCK),
            "dirty": manifest.get("repository", {}).get("dirty", True),
            "dirty_policy_passed": False,
        }
    return launch_gate_errors(manifest, current=current)


def _cells(manifest):
    req = manifest["request"]
    for dataset in req["datasets"]:
        dataset_id = dataset.replace(",", "v")
        for config in req["configs"]:
            for seed in req["seeds"]:
                yield dataset_id, config, seed


def campaign_state(manifest, launched, child_failures):
    if not launched:
        return "pending"
    if child_failures:
        return "failed"
    valid = 0
    for dataset, config, seed in _cells(manifest):
        directory = run_artifacts.run_dir(
            dataset, config, seed, root=manifest["output_roots"]["runs"], create=False)
        if run_artifacts.is_reusable(
                directory, config=expected_config(config, seed, manifest["request"]["epochs"]),
                seed=seed):
            valid += 1
    return "complete" if valid == manifest["request"]["expected_cells"] else "running"


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


def plan(campaign_name):
    if campaign_name != FIRST_CAMPAIGN:
        raise ValueError("only the reviewed first campaign is supported")
    directory = CAMPAIGN_ROOT / campaign_name
    manifest_path = directory / "manifest.json"
    cost_path = directory / "cost_estimate.json"
    if manifest_path.exists() or cost_path.exists():
        raise FileExistsError("campaign planning artifacts already exist")
    directory.mkdir(parents=True, exist_ok=True)
    # Cost measurement is explicitly approved planning evidence. It runs in this
    # process and never invokes the scheduler or starts a subprocess.
    from experiments import estimate_cost
    old_argv = sys.argv[:]
    try:
        sys.argv = ["estimate_cost"] + _cost_command(cost_path)[3:]
        rc = estimate_cost.main()
    finally:
        sys.argv = old_argv
    if rc != 0:
        raise RuntimeError("cost estimate was not approved")
    repo = repository_state()
    workers = _resolved_workers(0)
    roots = {"campaign": str(directory),
             "experiments": str(directory / "experiments"),
             "runs": str(directory / "runs"),
             "manifests": str(directory / "manifests")}
    argv = _runner_argv()
    manifest = {
        "schema": {"name": "fqcnn_campaign_manifest", "version": 1},
        "campaign": campaign_name, "state": "pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": {"command": shlex.join(argv), "argv": argv},
        "repository": repo,
        "locks": {
            "training": {"path": str(TRAINING_LOCK), "sha256": sha256_file(TRAINING_LOCK)},
            "qiskit": {"path": str(QISKIT_LOCK), "sha256": sha256_file(QISKIT_LOCK)},
        },
        "request": {"datasets": FIRST_DATASETS, "class_pairs": [[0, 1], [3, 5], [4, 9], [5, 8]],
                    "configs": FIRST_CONFIGS, "seeds": FIRST_SEEDS,
                    "split_policy": "deterministic stratified manifest per dataset and seed",
                    "samples": 400, "epochs": 30, "jobs_requested": 0,
                    "workers_resolved": workers, "with_baselines": False,
                    "expected_cells": 40},
        "cost": {"path": str(cost_path), "sha256": sha256_file(cost_path),
                 "projected_resource_use": _load_json(cost_path)["projection"]},
        "output_roots": roots,
        "artifacts": {"manifest": str(manifest_path), "cost_estimate": str(cost_path),
                      "launch": str(directory / "launch.json"),
                      "status": str(directory / "status.json"),
                      "failures": str(directory / "failures.json")},
        "approvals": {
            "full_suite": {"path": str(directory / "full_suite.json")},
            "cost": {"approved": True},
            "launch": {"approved": True, "scope": "tooling implemented; execution still gated"},
        },
    }
    write_immutable_json(manifest_path, manifest)
    return manifest


def _load_campaign(name):
    return _load_json(CAMPAIGN_ROOT / name / "manifest.json")


def _acquire_queue(campaign_name):
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(QUEUE_PATH), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as fh:
        json.dump({"campaign": campaign_name, "pid": os.getpid(), "host": platform.node(),
                   "acquired_at_utc": datetime.now(timezone.utc).isoformat()}, fh,
                  indent=2, sort_keys=True)
        fh.write("\n")


def launch(manifest):
    errors = launch_gate_errors(manifest)
    if errors:
        return errors
    _acquire_queue(manifest["campaign"])
    write_immutable_json(manifest["artifacts"]["launch"], {
        "schema": {"name": "fqcnn_campaign_launch", "version": 1},
        "campaign": manifest["campaign"], "git_sha": manifest["repository"]["git_sha"],
        "manifest_sha256": sha256_file(manifest["artifacts"]["manifest"]),
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue": str(QUEUE_PATH), "owner_pid": os.getpid(), "owner_host": platform.node(),
    })
    old_argv = sys.argv[:]
    old_roots = (run_experiments.EXP_ROOT, run_experiments.FAILURE_MANIFEST,
                 run_experiments.MANIFEST_ROOT, run_artifacts.RUN_ROOT)
    try:
        roots = manifest["output_roots"]
        run_experiments.EXP_ROOT = roots["experiments"]
        run_experiments.FAILURE_MANIFEST = str(Path(roots["experiments"]) / "failures.json")
        run_experiments.MANIFEST_ROOT = roots["manifests"]
        run_artifacts.RUN_ROOT = roots["runs"]
        sys.argv = manifest["runner"]["argv"][2:]
        run_experiments.main()
    finally:
        sys.argv = old_argv
        (run_experiments.EXP_ROOT, run_experiments.FAILURE_MANIFEST,
         run_experiments.MANIFEST_ROOT, run_artifacts.RUN_ROOT) = old_roots
        with contextlib.suppress(OSError):
            QUEUE_PATH.unlink()
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "validate", "launch", "status"):
        command = sub.add_parser(action)
        command.add_argument("--campaign", required=True)
    args = parser.parse_args()
    if args.action == "plan":
        manifest = plan(args.campaign)
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
    launch_path = Path(manifest["artifacts"]["launch"])
    failures_path = Path(manifest["artifacts"]["failures"])
    failures = _load_json(failures_path).get("failures", []) if failures_path.exists() else []
    state = campaign_state(manifest, launch_path.exists(), failures)
    print(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
