import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from QCNN.utils import run_artifacts
from experiments import campaign


REQUIRED_STATES = {"pending", "running", "failed", "complete"}


def _cost():
    return {
        "schema": {"name": "fqcnn_campaign_cost_estimate", "version": 1},
        "status": "approved",
        "approval": {"approved": True, "reasons": ["fits"], "budget_hours": 70.0},
        "request": {
            "datasets": ["0,1"], "configs": ["proposed"], "seeds": [0],
            "samples": 400, "epochs": 30, "with_baselines": False,
        },
        "counts": {"requested_configs": 1, "measurable_configs": 1,
                   "requested_cells": 1, "measurable_cells": 1,
                   "failed_calibrations": 0, "total_failures": 0},
        "calibrations": {"configs": {"proposed": {}},
                         "baseline_seconds_per_proposed_cell": None},
        "failures": [],
        "projection": {"serial_hours": 1.0, "wall_hours": 1.0, "workers": 1,
                       "budget_hours": 70.0, "budget_fraction": 1 / 70},
        "reduction": None,
    }


def _manifest(tmp_path):
    cost_path = tmp_path / "cost_estimate.json"
    cost_path.write_text(json.dumps(_cost()))
    return {
        "schema": {"name": "fqcnn_campaign_manifest", "version": 1},
        "campaign": "unit",
        "state": "pending",
        "runner": {
            "command": "python -m experiments.run_experiments --datasets 0,1 --configs proposed --seeds 0 --samples 400 --epochs 30 --jobs 1 --no-baselines",
            "argv": ["python", "-m", "experiments.run_experiments", "--datasets", "0,1", "--configs", "proposed", "--seeds", "0", "--samples", "400", "--epochs", "30", "--jobs", "1", "--no-baselines"],
        },
        "repository": {"git_sha": "abc", "branch": "dev", "upstream": None,
                       "ahead": 0, "behind": 0, "dirty": False,
                       "dirty_policy": {"passed": True, "reasons": [], "raw": []}},
        "locks": {"training": {"path": "requirements-lock.txt", "sha256": "train"},
                  "qiskit": {"path": "requirements-qiskit-lock.txt", "sha256": "qiskit"}},
        "request": {"datasets": ["0,1"], "class_pairs": [[0, 1]],
                    "configs": ["proposed"], "seeds": [0],
                    "split_policy": "deterministic stratified manifest per dataset and seed",
                    "samples": 400, "epochs": 30, "jobs_requested": 1,
                    "workers_resolved": 1, "with_baselines": False,
                    "expected_cells": 1},
        "cost": {"path": str(cost_path), "sha256": campaign.sha256_file(cost_path),
                 "projected_resource_use": _cost()["projection"]},
        "output_roots": {"campaign": str(tmp_path), "experiments": str(tmp_path / "experiments"),
                         "runs": str(tmp_path / "runs"), "manifests": str(tmp_path / "manifests")},
        "artifacts": {"manifest": str(tmp_path / "manifest.json"),
                      "cost_estimate": str(cost_path), "launch": str(tmp_path / "launch.json"),
                      "status": str(tmp_path / "status.json"), "failures": str(tmp_path / "failures.json")},
        "approvals": {"full_suite": {"path": str(tmp_path / "full-suite.json")},
                      "cost": {"approved": True}, "launch": {"approved": True}},
    }


@pytest.mark.parametrize("path", [
    ("repository", "git_sha"), ("locks", "training"), ("request", "split_policy"),
    ("request", "expected_cells"), ("cost", "path"), ("repository", "dirty"),
])
def test_manifest_requires_provenance_fields(tmp_path, path):
    manifest = _manifest(tmp_path)
    del manifest[path[0]][path[1]]
    assert campaign.validate_manifest(manifest)


def test_manifest_rejects_unknown_state(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["state"] = "planned"
    assert campaign.validate_manifest(manifest)
    assert campaign.CAMPAIGN_STATES == REQUIRED_STATES


def test_manifest_creation_is_exclusive(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    campaign.write_immutable_json(path, manifest)
    with pytest.raises(FileExistsError):
        campaign.write_immutable_json(path, manifest)


@pytest.mark.parametrize("mutation, expected", [
    (lambda c: c["approval"].update(approved=False), "cost estimate is not approved"),
    (lambda c: c["counts"].update(measurable_cells=0), "cost estimate is incomplete"),
    (lambda c: c["projection"].update(wall_hours=float("inf")), "non-finite"),
])
def test_cost_gate_rejects_unapproved_incomplete_or_nonfinite(tmp_path, mutation, expected):
    manifest = _manifest(tmp_path)
    cost = _cost()
    mutation(cost)
    assert any(expected in reason for reason in campaign.validate_cost(manifest, cost))


def _suite(path, *, status="pass", sha="abc", dirty_passed=True):
    path.write_text(json.dumps({
        "schema": {"name": "fqcnn_full_suite_evidence", "version": 1},
        "status": status, "command": "python -m pytest tests/ -q", "exit_code": 0,
        "tests": {"passed": 329, "failed": 0}, "git_sha": sha,
        "training_lock_sha256": "train", "timestamp_utc": "2026-08-18T00:00:00Z",
        "dirty_policy": {"passed": dirty_passed, "reasons": []},
    }))


def test_launch_refuses_dirty_tree(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["repository"]["dirty"] = True
    assert "working tree is dirty" in campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "train", "qiskit_lock_sha256": "qiskit", "dirty": True, "dirty_policy_passed": False})


@pytest.mark.parametrize("present,status", [(False, "pass"), (True, "fail")])
def test_launch_refuses_missing_or_failed_full_suite(tmp_path, present, status):
    manifest = _manifest(tmp_path)
    suite = Path(manifest["approvals"]["full_suite"]["path"])
    if present:
        _suite(suite, status=status)
    errors = campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "train", "qiskit_lock_sha256": "qiskit", "dirty": False, "dirty_policy_passed": True})
    assert any("full-suite" in e for e in errors)


def test_launch_refuses_occupied_queue(tmp_path):
    manifest = _manifest(tmp_path)
    _suite(Path(manifest["approvals"]["full_suite"]["path"]))
    queue = tmp_path / "queue.lock"
    queue.write_text("occupied")
    assert "global training/hardware queue is occupied" in campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "train", "qiskit_lock_sha256": "qiskit", "dirty": False, "dirty_policy_passed": True}, queue_path=queue)


def test_launch_refuses_output_collision(tmp_path):
    manifest = _manifest(tmp_path)
    _suite(Path(manifest["approvals"]["full_suite"]["path"]))
    Path(manifest["output_roots"]["experiments"]).mkdir()
    errors = campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "train", "qiskit_lock_sha256": "qiskit", "dirty": False, "dirty_policy_passed": True})
    assert any("output collision" in e for e in errors)


def test_baseline_n10_requires_schedule_evidence(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["request"]["with_baselines"] = True
    manifest["request"]["n_qubits"] = 10
    _suite(Path(manifest["approvals"]["full_suite"]["path"]))
    errors = campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "train", "qiskit_lock_sha256": "qiskit", "dirty": False, "dirty_policy_passed": True})
    assert any("baseline schedule evidence" in e for e in errors)


def _complete_cell(root, dataset="0v1", config="proposed", seed=0):
    directory = run_artifacts.run_dir(dataset, config, seed, root=str(root))
    expected = campaign.expected_config(config, seed, 30)
    run_artifacts.start_run(directory, expected, "split", seed, {"python": "3.9.13"})
    run_artifacts.save_predictions(directory, [1], [1], [0.5])
    np.savez(run_artifacts.weights_path(directory), w=np.zeros(1))
    run_artifacts.complete_run(directory, {"accuracy": 1.0})


@pytest.mark.parametrize("mode", ["missing", "partial", "failed", "invalid", "insufficient"])
def test_completion_requires_every_valid_cell_and_zero_failures(tmp_path, mode):
    manifest = _manifest(tmp_path)
    root = Path(manifest["output_roots"]["runs"])
    if mode != "missing":
        _complete_cell(root)
    if mode == "partial":
        os.remove(root / "0v1" / "proposed" / "seed_0" / "weights.npz")
    elif mode == "failed":
        run_artifacts.fail_run(str(root / "0v1" / "proposed" / "seed_0"), "boom")
    elif mode == "invalid":
        status = root / "0v1" / "proposed" / "seed_0" / "status.json"
        data = json.loads(status.read_text()); data["config"]["n_epochs"] = 2; status.write_text(json.dumps(data))
    elif mode == "insufficient":
        manifest["request"]["expected_cells"] = 2
    failures = [] if mode != "failed" else [{"error": "boom"}]
    assert campaign.campaign_state(manifest, launched=True, child_failures=failures) != "complete"


def test_complete_requires_unlaunched_flag_to_be_false(tmp_path):
    manifest = _manifest(tmp_path)
    _complete_cell(Path(manifest["output_roots"]["runs"]))
    assert campaign.campaign_state(manifest, launched=False, child_failures=[]) == "pending"
    assert campaign.campaign_state(manifest, launched=True, child_failures=[]) == "complete"


def test_planning_repository_snapshot_never_invokes_subprocess(monkeypatch):
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("subprocess called"))
    snapshot = campaign.repository_state()
    assert snapshot["git_sha"]
    assert snapshot["dirty_policy"]["passed"] is False


def test_validate_never_invokes_scheduler(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("subprocess called"))
    campaign.validate_campaign(manifest, current={"git_sha": "wrong", "training_lock_sha256": "train", "qiskit_lock_sha256": "qiskit", "dirty": False, "dirty_policy_passed": True})
