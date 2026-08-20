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
            "samples": 400, "epochs": 30, "jobs_requested": 1, "with_baselines": False,
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
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        "repository": {"git_sha": "abc", "branch": "dev", "upstream": "origin/dev",
                       "ahead": 0, "behind": 0, "dirty": False,
                       "dirty_policy": {"passed": True, "reasons": [], "raw": []}},
        "locks": {"training": {"path": "requirements-lock.txt", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
                  "qiskit": {"path": "requirements-qiskit-lock.txt", "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}},
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
        "tests": {"passed": 329, "failed": 0, "total": 329}, "git_sha": sha,
        "training_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "timestamp_utc": "2026-08-18T00:00:00Z",
        "dirty_policy": {"passed": dirty_passed, "reasons": []},
    }))


def test_launch_refuses_dirty_tree(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["repository"]["dirty"] = True
    assert "working tree is dirty" in campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "qiskit_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": True, "dirty_policy_passed": False})


@pytest.mark.parametrize("present,status", [(False, "pass"), (True, "fail")])
def test_launch_refuses_missing_or_failed_full_suite(tmp_path, present, status):
    manifest = _manifest(tmp_path)
    suite = Path(manifest["approvals"]["full_suite"]["path"])
    if present:
        _suite(suite, status=status)
    errors = campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "qiskit_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": False, "dirty_policy_passed": True})
    assert any("full-suite" in e for e in errors)


def test_launch_refuses_occupied_queue(tmp_path):
    manifest = _manifest(tmp_path)
    _suite(Path(manifest["approvals"]["full_suite"]["path"]))
    queue = tmp_path / "queue.lock"
    queue.write_text("occupied")
    assert "global training/hardware queue is occupied" in campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "qiskit_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": False, "dirty_policy_passed": True}, queue_path=queue)


def test_launch_refuses_output_collision(tmp_path):
    manifest = _manifest(tmp_path)
    _suite(Path(manifest["approvals"]["full_suite"]["path"]))
    Path(manifest["output_roots"]["experiments"]).mkdir()
    errors = campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "qiskit_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": False, "dirty_policy_passed": True})
    assert any("output collision" in e for e in errors)


def test_baseline_n10_requires_schedule_evidence(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["request"]["with_baselines"] = True
    manifest["request"]["n_qubits"] = 10
    manifest["runner"]["argv"] = campaign._expected_runner_argv(manifest["request"])
    manifest["runner"]["command"] = campaign.shlex.join(manifest["runner"]["argv"])
    _suite(Path(manifest["approvals"]["full_suite"]["path"]))
    errors = campaign.launch_gate_errors(manifest, current={"git_sha": "abc", "training_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "qiskit_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": False, "dirty_policy_passed": True})
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
    campaign.validate_campaign(manifest, current={"git_sha": "wrong", "training_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "qiskit_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": False, "dirty_policy_passed": True})


# Review round 1 regressions.
def _valid_suite(manifest):
    path = Path(manifest["approvals"]["full_suite"]["path"])
    _suite(path)
    record = json.loads(path.read_text())
    record["tests"]["total"] = record["tests"]["passed"]
    path.write_text(json.dumps(record))
    manifest["approvals"]["full_suite"]["sha256"] = campaign.sha256_file(path)
    return path


def _clean_current():
    return {"git_sha": "abc", "training_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "qiskit_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": False,
            "dirty_policy_passed": True}


def test_real_clean_launch_state_is_not_forced_dirty(monkeypatch):
    monkeypatch.setattr(campaign, "_git", lambda *args: {
        ("rev-parse", "HEAD"): "abc", ("status", "--porcelain=v1"): ""
    }[args])
    assert campaign.current_launch_state()["dirty"] is False
    assert campaign.current_launch_state()["dirty_policy_passed"] is True


def test_first_campaign_plan_has_no_self_approval():
    manifest = json.loads(Path("Results/campaigns/baseline_mnist_n10_v1/manifest.json").read_text())
    assert manifest["approvals"]["launch"]["approved"] is False


def test_status_reads_scheduler_failure_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    _complete_cell(Path(manifest["output_roots"]["runs"]))
    failure_path = Path(manifest["output_roots"]["experiments"]) / "failures.json"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text(json.dumps({"n_failed": 1, "failures": [{"error": "boom"}]}))
    assert campaign.status_campaign(manifest, launched=True)["state"] == "failed"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(command="python -m pytest tests/test_campaign_manifest.py -q"),
    lambda r: r["tests"].update(failed=1),
    lambda r: r["tests"].update(total=r["tests"]["passed"] + 1),
])
def test_full_suite_requires_exact_complete_command_and_totals(tmp_path, mutation):
    manifest = _manifest(tmp_path)
    path = _valid_suite(manifest)
    record = json.loads(path.read_text())
    mutation(record)
    path.write_text(json.dumps(record))
    manifest["approvals"]["full_suite"]["sha256"] = campaign.sha256_file(path)
    assert campaign._full_suite_errors(manifest)


def test_full_suite_hash_is_immutable_gate(tmp_path):
    manifest = _manifest(tmp_path)
    path = _valid_suite(manifest)
    path.write_text(path.read_text() + " ")
    assert "full-suite evidence hash mismatch" in campaign._full_suite_errors(manifest)


def test_cost_request_requires_requested_jobs(tmp_path):
    manifest = _manifest(tmp_path)
    cost = _cost()
    cost["request"]["jobs_requested"] = 0
    assert "cost estimate request does not exactly match campaign" in campaign.validate_cost(manifest, cost)


@pytest.mark.parametrize("mutation", [
    lambda m, p: m["output_roots"].update(runs=str(p / "outside")),
    lambda m, p: m["output_roots"].update(runs=m["output_roots"]["experiments"]),
    lambda m, p: m["artifacts"].update(failures=str(p / "wrong" / "failures.json")),
])
def test_manifest_rejects_unsafe_or_inconsistent_paths(tmp_path, mutation):
    manifest = _manifest(tmp_path / "campaign")
    mutation(manifest, tmp_path)
    assert any("path" in error or "root" in error for error in campaign.validate_manifest(manifest))


@pytest.mark.parametrize("artifact", ["launch", "status", "failures"])
def test_launch_preflights_runtime_artifact_collisions(tmp_path, artifact):
    manifest = _manifest(tmp_path)
    _valid_suite(manifest)
    Path(manifest["artifacts"][artifact]).write_text("occupied")
    errors = campaign.launch_gate_errors(manifest, current=_clean_current(), queue_path=tmp_path / "queue")
    assert any("artifact collision" in error for error in errors)


def test_launch_record_failure_releases_queue(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    _valid_suite(manifest)
    queue = tmp_path / "queue"
    monkeypatch.setattr(campaign, "QUEUE_PATH", queue)
    monkeypatch.setattr(campaign, "launch_gate_errors", lambda manifest: [])
    monkeypatch.setattr(campaign, "write_immutable_json", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        campaign.launch(manifest)
    assert not queue.exists()


def test_status_before_launch_does_not_create_runtime_state(tmp_path):
    manifest = _manifest(tmp_path)
    payload = campaign.status_campaign(manifest, launched=False)
    assert payload["state"] == "pending"
    assert not Path(manifest["artifacts"]["status"]).exists()


def test_status_after_launch_writes_mutable_status_json(tmp_path):
    manifest = _manifest(tmp_path)
    payload = campaign.status_campaign(manifest, launched=True)
    assert payload["state"] == "running"
    assert json.loads(Path(manifest["artifacts"]["status"]).read_text())["state"] == "running"


def test_plan_rolls_back_cost_when_manifest_build_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "CAMPAIGN_ROOT", tmp_path)
    monkeypatch.setattr(campaign, "repository_state", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    from experiments import estimate_cost
    def fake_cost():
        output = Path(os.sys.argv[os.sys.argv.index("--output-json") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_cost()))
        return 0
    monkeypatch.setattr(estimate_cost, "main", fake_cost)
    with pytest.raises(RuntimeError):
        campaign.plan(campaign.FIRST_CAMPAIGN)
    assert not (tmp_path / campaign.FIRST_CAMPAIGN / "cost_estimate.json").exists()


@pytest.mark.parametrize("mutation", [
    lambda m: m["runner"].update(command="not the argv"),
    lambda m: m["request"].update(class_pairs=[[9, 9]]),
    lambda m: m["locks"]["training"].update(sha256=""),
    lambda m: m["approvals"].update(launch={"approved": "yes"}),
])
def test_manifest_rejects_malformed_identity_shapes(tmp_path, mutation):
    manifest = _manifest(tmp_path)
    mutation(manifest)
    assert campaign.validate_manifest(manifest)


def test_failed_gate_has_no_launch_side_effects(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    called = []
    monkeypatch.setattr(campaign, "launch_gate_errors", lambda manifest: ["blocked"])
    monkeypatch.setattr(campaign, "_acquire_queue", lambda *a: called.append("queue"))
    monkeypatch.setattr(campaign.run_experiments, "main", lambda: called.append("scheduler"))
    assert campaign.launch(manifest) == ["blocked"]
    assert called == []
    assert not Path(manifest["artifacts"]["launch"]).exists()


def test_approved_clean_launch_uses_scheduler_and_cleans_queue(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    Path(manifest["artifacts"]["manifest"]).write_text(json.dumps(manifest))
    queue = tmp_path / "queue"
    monkeypatch.setattr(campaign, "QUEUE_PATH", queue)
    monkeypatch.setattr(campaign, "launch_gate_errors", lambda manifest: [])
    called = []
    monkeypatch.setattr(campaign.run_experiments, "main", lambda: called.append("scheduler"))
    assert campaign.launch(manifest) == []
    assert called == ["scheduler"]
    assert Path(manifest["artifacts"]["launch"]).is_file()
    assert not queue.exists()


def test_runner_command_and_argv_must_exactly_encode_request(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["runner"]["argv"][-2] = "31"
    manifest["runner"]["command"] = " ".join(manifest["runner"]["argv"])
    assert any("runner" in error for error in campaign.validate_manifest(manifest))


@pytest.mark.parametrize("field,value", [
    ("wall_hours", True), ("serial_hours", -1.0), ("workers", 0),
    ("budget_hours", "70"), ("budget_fraction", float("nan")),
])
def test_cost_projection_rejects_malformed_numeric_values(tmp_path, field, value):
    manifest = _manifest(tmp_path)
    cost = _cost()
    cost["request"]["jobs_requested"] = 1
    cost["projection"][field] = value
    assert campaign.validate_cost(manifest, cost)


@pytest.mark.parametrize("lock_name", ["training", "qiskit"])
def test_manifest_requires_complete_nested_lock_shape(tmp_path, lock_name):
    manifest = _manifest(tmp_path)
    del manifest["locks"][lock_name]["sha256"]
    assert campaign.validate_manifest(manifest)


@pytest.mark.parametrize("field", ["branch", "upstream", "ahead", "behind"])
def test_manifest_requires_populated_repository_provenance(tmp_path, field):
    manifest = _manifest(tmp_path)
    manifest["repository"][field] = None
    assert campaign.validate_manifest(manifest)


def test_compatible_same_campaign_resume_outputs_are_allowed(tmp_path):
    manifest = _manifest(tmp_path)
    Path(manifest["artifacts"]["manifest"]).write_text(json.dumps(manifest))
    launch_record = {
        "schema": {"name": "fqcnn_campaign_launch", "version": 1},
        "campaign": manifest["campaign"],
        "git_sha": manifest["repository"]["git_sha"],
        "manifest_sha256": campaign.sha256_file(manifest["artifacts"]["manifest"]),
    }
    Path(manifest["artifacts"]["launch"]).write_text(json.dumps(launch_record))
    Path(manifest["output_roots"]["runs"]).mkdir()
    assert not campaign._output_errors(manifest)


def test_incompatible_existing_launch_record_is_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    Path(manifest["artifacts"]["launch"]).write_text(json.dumps({"campaign": "other"}))
    assert campaign._output_errors(manifest)
