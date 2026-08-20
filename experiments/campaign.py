#!/usr/bin/env python3
"""Immutable provenance and fail-closed gates for experiment campaigns."""
from __future__ import annotations

import argparse
import configparser
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
    """Capture non-mutating Git identity without starting a subprocess."""
    head = Path(".git/HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        branch = ref.rsplit("/", 1)[-1]
        sha = _read_git_ref(ref)
    else:
        branch, sha = "detached", head
    config = configparser.ConfigParser()
    config.read(".git/config")
    section = 'branch "{}"'.format(branch)
    remote = config.get(section, "remote", fallback=None)
    merge = config.get(section, "merge", fallback=None)
    upstream = (remote + "/" + merge.rsplit("/", 1)[-1]) if remote and merge else "unconfigured"
    return {"git_sha": sha, "branch": branch, "upstream": upstream,
            "ahead": "unverified", "behind": "unverified", "dirty": True,
            "dirty_policy": {
                "passed": False,
                "reasons": ["planning cannot prove a clean tree without a forbidden Git subprocess"],
                "raw": [],
            }}


def current_launch_state():
    """Recheck exact revision and cleanliness immediately before launch."""
    raw = [line for line in _git("status", "--porcelain=v1").splitlines() if line]
    return {"git_sha": _git("rev-parse", "HEAD"),
            "training_lock_sha256": sha256_file(TRAINING_LOCK),
            "qiskit_lock_sha256": sha256_file(QISKIT_LOCK),
            "dirty": bool(raw), "dirty_policy_passed": not raw,
            "dirty_reasons": raw}


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


def validate_manifest(manifest):
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if manifest.get("schema") != {"name": "fqcnn_campaign_manifest", "version": 1}:
        errors.append("invalid manifest schema")
    if not isinstance(manifest.get("campaign"), str) or not manifest.get("campaign"):
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
        if repo.get("ahead") is None or repo.get("behind") is None:
            errors.append("invalid repository divergence")
        if not isinstance(repo.get("dirty"), bool):
            errors.append("invalid repository dirty state")
        policy = repo.get("dirty_policy")
        if not isinstance(policy, dict) or not isinstance(policy.get("passed"), bool):
            errors.append("invalid dirty policy")
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
    required_request = ("datasets", "class_pairs", "configs", "seeds", "split_policy",
                        "samples", "epochs", "jobs_requested", "workers_resolved",
                        "with_baselines", "expected_cells")
    if any(key not in request for key in required_request):
        errors.append("incomplete campaign request")
    datasets = request.get("datasets", [])
    pairs = request.get("class_pairs", [])
    if not isinstance(datasets, list) or not datasets or pairs != [list(map(int, d.split(","))) for d in datasets if isinstance(d, str) and "," in d]:
        errors.append("dataset/class-pair mismatch")
    if not isinstance(request.get("configs"), list) or not request.get("configs"):
        errors.append("invalid configs")
    if not isinstance(request.get("seeds"), list) or not request.get("seeds") or any(type(s) is not int for s in request.get("seeds", [])):
        errors.append("invalid seeds")
    for key in ("samples", "epochs", "workers_resolved"):
        if type(request.get(key)) is not int or request.get(key, 0) <= 0:
            errors.append("invalid request " + key)
    if type(request.get("jobs_requested")) is not int or request.get("jobs_requested", -1) < 0:
        errors.append("invalid requested jobs")
    if not isinstance(request.get("with_baselines"), bool) or not isinstance(request.get("split_policy"), str) or not request.get("split_policy"):
        errors.append("invalid request policy")
    calculated = len(datasets) * len(request.get("configs", [])) * len(request.get("seeds", []))
    if request.get("expected_cells") != calculated or calculated <= 0:
        errors.append("malformed expected-cell count")
    runner = manifest.get("runner")
    if isinstance(runner, dict) and not errors:
        expected_argv = _expected_runner_argv(request)
        if runner.get("argv") != expected_argv or runner.get("command") != shlex.join(expected_argv):
            errors.append("runner command/argv does not exactly match request")
    elif not isinstance(runner, dict):
        errors.append("missing manifest field: runner")
    cost = manifest.get("cost")
    if not isinstance(cost, dict) or not isinstance(cost.get("path"), str) or not _is_sha256(cost.get("sha256")):
        errors.append("invalid cost provenance")
    approvals = manifest.get("approvals")
    if not isinstance(approvals, dict) or not isinstance(approvals.get("launch"), dict) or type(approvals.get("launch", {}).get("approved")) is not bool:
        errors.append("invalid launch approval shape")
    errors.extend(_path_errors(manifest))
    return errors


def _path_errors(manifest):
    errors = []
    roots = manifest.get("output_roots")
    artifacts = manifest.get("artifacts")
    if not isinstance(roots, dict) or not isinstance(artifacts, dict):
        return ["invalid output/artifact paths"]
    try:
        campaign_root = Path(roots["campaign"]).resolve()
        child_roots = [Path(roots[name]).resolve() for name in ("experiments", "runs", "manifests")]
        if len(set(child_roots)) != 3 or any(root == campaign_root or campaign_root not in root.parents for root in child_roots):
            errors.append("output roots must be distinct children of campaign root")
        expected = {"manifest": campaign_root / "manifest.json",
                    "cost_estimate": campaign_root / "cost_estimate.json",
                    "launch": campaign_root / "launch.json", "status": campaign_root / "status.json",
                    "failures": campaign_root / "failures.json"}
        for name, path in expected.items():
            if Path(artifacts.get(name, "")).resolve() != path:
                errors.append("artifact path mismatch: " + name)
    except (KeyError, OSError, TypeError):
        errors.append("invalid output/artifact paths")
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
                        "epochs": req.get("epochs"), "jobs_requested": req.get("jobs_requested"),
                        "with_baselines": req.get("with_baselines")}
    if cost.get("request") != expected_request:
        errors.append("cost estimate request does not exactly match campaign")
    projection = cost.get("projection")
    if not isinstance(projection, dict):
        errors.append("cost estimate has no projection")
    else:
        for key in ("serial_hours", "wall_hours", "budget_hours", "budget_fraction"):
            value = projection.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                errors.append("invalid or non-finite cost projection field: " + key)
        workers = projection.get("workers")
        if type(workers) is not int or workers <= 0:
            errors.append("invalid cost projection field: workers")
        elif workers != req.get("workers_resolved"):
            errors.append("cost worker policy does not match campaign")
        wall = projection.get("wall_hours")
        budget = projection.get("budget_hours")
        if type(wall) in (int, float) and type(budget) in (int, float) and not isinstance(wall, bool) and not isinstance(budget, bool) and wall > budget:
            errors.append("cost estimate is over budget")
    return errors


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _full_suite_errors(manifest):
    evidence = manifest.get("approvals", {}).get("full_suite", {})
    path = Path(evidence.get("path", ""))
    if not path.is_file():
        return ["missing full-suite evidence"]
    if not _is_sha256(evidence.get("sha256")) or sha256_file(path) != evidence.get("sha256"):
        return ["full-suite evidence hash mismatch"]
    try:
        record = _load_json(path)
    except (OSError, ValueError):
        return ["invalid full-suite evidence"]
    errors = []
    if record.get("schema") != {"name": "fqcnn_full_suite_evidence", "version": 1}:
        errors.append("invalid full-suite evidence schema")
    if record.get("command") != "python -m pytest tests/ -q":
        errors.append("full-suite evidence command mismatch")
    tests = record.get("tests", {})
    if (type(tests.get("passed")) is not int or tests.get("passed", 0) <= 0 or
            tests.get("failed") != 0 or tests.get("total") != tests.get("passed")):
        errors.append("full-suite evidence totals are incomplete")
    if record.get("status") != "pass" or record.get("exit_code") != 0:
        errors.append("full-suite evidence did not pass")
    if record.get("git_sha") != manifest.get("repository", {}).get("git_sha"):
        errors.append("full-suite evidence Git SHA mismatch")
    if record.get("training_lock_sha256") != manifest.get("locks", {}).get("training", {}).get("sha256"):
        errors.append("full-suite evidence training-lock mismatch")
    if record.get("dirty_policy", {}).get("passed") is not True:
        errors.append("full-suite evidence dirty policy failed")
    return errors


def _output_errors(manifest):
    artifacts = manifest["artifacts"]
    roots = manifest["output_roots"]
    launch_path = Path(artifacts["launch"])
    if launch_path.exists():
        try:
            record = _load_json(launch_path)
            manifest_path = Path(artifacts["manifest"])
            if (record.get("schema") != {"name": "fqcnn_campaign_launch", "version": 1} or
                    record.get("campaign") != manifest["campaign"] or
                    record.get("git_sha") != manifest["repository"]["git_sha"] or
                    not manifest_path.is_file() or
                    record.get("manifest_sha256") != sha256_file(manifest_path)):
                return ["artifact collision: incompatible existing launch artifact"]
        except (OSError, ValueError):
            return ["artifact collision: incompatible existing launch artifact"]
        return []
    errors = []
    for name in ("experiments", "runs", "manifests"):
        if Path(roots[name]).exists():
            errors.append("output collision: " + name)
    for name in ("launch", "status", "failures"):
        if Path(artifacts[name]).exists():
            errors.append("artifact collision: " + name)
    return errors


def launch_gate_errors(manifest, current=None, queue_path=QUEUE_PATH):
    errors = validate_manifest(manifest)
    if errors:
        return errors
    current = current or current_launch_state()
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
    errors.extend(_output_errors(manifest))
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
        repository = repository_state()
        current = {
            "git_sha": repository["git_sha"],
            "training_lock_sha256": sha256_file(TRAINING_LOCK),
            "qiskit_lock_sha256": sha256_file(QISKIT_LOCK),
            "dirty": repository["dirty"],
            "dirty_policy_passed": repository["dirty_policy"]["passed"],
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
    from experiments import estimate_cost
    old_argv = sys.argv[:]
    try:
        sys.argv = ["estimate_cost"] + _cost_command(cost_path)[3:]
        rc = estimate_cost.main()
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
            "request": {"datasets": FIRST_DATASETS,
                        "class_pairs": [[0, 1], [3, 5], [4, 9], [5, 8]],
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
                "full_suite": {"path": str(directory / "full_suite.json"), "sha256": "0" * 64},
                "cost": {"approved": True},
                "launch": {"approved": False, "scope": "human approval required"},
            },
        }
        write_immutable_json(manifest_path, manifest)
        return manifest
    except Exception:
        with contextlib.suppress(OSError):
            cost_path.unlink()
        with contextlib.suppress(OSError):
            manifest_path.unlink()
        with contextlib.suppress(OSError):
            directory.rmdir()
        raise
    finally:
        sys.argv = old_argv


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
    old_argv = sys.argv[:]
    old_roots = (run_experiments.EXP_ROOT, run_experiments.FAILURE_MANIFEST,
                 run_experiments.MANIFEST_ROOT, run_artifacts.RUN_ROOT)
    try:
        launch_path = Path(manifest["artifacts"]["launch"])
        if not launch_path.exists():
            write_immutable_json(launch_path, {
                "schema": {"name": "fqcnn_campaign_launch", "version": 1},
                "campaign": manifest["campaign"], "git_sha": manifest["repository"]["git_sha"],
                "manifest_sha256": sha256_file(manifest["artifacts"]["manifest"]),
                "launched_at_utc": datetime.now(timezone.utc).isoformat(),
                "queue": str(QUEUE_PATH), "owner_pid": os.getpid(), "owner_host": platform.node(),
            })
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


def _child_failures(manifest):
    scheduler_path = Path(manifest["output_roots"]["experiments"]) / "failures.json"
    if not scheduler_path.is_file():
        return []
    payload = _load_json(scheduler_path)
    failures = payload.get("failures", [])
    if not isinstance(failures, list) or payload.get("n_failed") != len(failures):
        return [{"error": "invalid scheduler failure manifest"}]
    return failures


def status_campaign(manifest, launched=None):
    if launched is None:
        launched = Path(manifest["artifacts"]["launch"]).exists()
    failures = _child_failures(manifest) if launched else []
    state = campaign_state(manifest, launched, failures)
    payload = {"schema": {"name": "fqcnn_campaign_status", "version": 1},
               "campaign": manifest["campaign"], "state": state,
               "checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "child_failures": len(failures)}
    if launched:
        _atomic_json(manifest["artifacts"]["status"], payload)
        _atomic_json(manifest["artifacts"]["failures"],
                     {"n_failed": len(failures), "failures": failures})
    return payload


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
    state = status_campaign(manifest)["state"]
    print(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
