"""Run isolation and provenance (UPGRADE_PLAN.md M0.5).

Two runs must not be able to overwrite each other's outputs or the archived
headline weights, partial output must never look complete, and the test set must
be evaluated exactly once per run.
"""
import json
import os

import numpy as np
import pytest

from QCNN.utils import run_artifacts


def test_runs_get_distinct_directories(tmp_path):
    root = str(tmp_path)
    a = run_artifacts.run_dir("0v1", "headline", 0, root=root)
    b = run_artifacts.run_dir("0v1", "headline", 1, root=root)
    c = run_artifacts.run_dir("3v5", "headline", 0, root=root)
    assert len({a, b, c}) == 3


def test_run_weights_never_target_the_archived_headline_file(tmp_path):
    directory = run_artifacts.run_dir("0v1", "headline", 0, root=str(tmp_path))
    archived = os.path.normpath(os.path.join("Results", "Weights", "quantum_model_params.npz"))
    assert os.path.normpath(run_artifacts.weights_path(directory)) != archived


def test_status_lifecycle_pending_running_complete(tmp_path):
    directory = run_artifacts.run_dir("0v1", "headline", 0, root=str(tmp_path))

    assert run_artifacts.is_complete(directory) is False

    run_artifacts.start_run(directory, config={"n_qubits": 10}, split_id="abc",
                            seed=0, environment={"python": "3.9.13"})
    assert run_artifacts.is_complete(directory) is False

    run_artifacts.complete_run(directory, metrics={"accuracy": 0.9})
    assert run_artifacts.is_complete(directory) is True

    with open(os.path.join(directory, "status.json")) as fh:
        status = json.load(fh)
    assert status["state"] == "complete"
    assert status["split_id"] == "abc"


def test_failed_run_is_not_complete(tmp_path):
    directory = run_artifacts.run_dir("0v1", "headline", 0, root=str(tmp_path))
    run_artifacts.start_run(directory, config={}, split_id="abc", seed=0, environment={})
    run_artifacts.fail_run(directory, error="simulator crashed")

    assert run_artifacts.is_complete(directory) is False
    with open(os.path.join(directory, "status.json")) as fh:
        assert json.load(fh)["state"] == "failed"


def test_per_example_predictions_are_persisted_for_paired_statistics(tmp_path):
    directory = run_artifacts.run_dir("0v1", "headline", 0, root=str(tmp_path))
    run_artifacts.start_run(directory, config={}, split_id="abc", seed=0, environment={})

    path = run_artifacts.save_predictions(
        directory,
        sample_ids=[11, 22, 33],
        y_true=np.array([1, -1, 1]),
        raw_outputs=np.array([0.4, -0.7, 0.1]),
    )
    stored = np.load(path)
    np.testing.assert_array_equal(stored["sample_ids"], [11, 22, 33])
    np.testing.assert_array_equal(stored["y_true"], [1, -1, 1])
    np.testing.assert_allclose(stored["raw_outputs"], [0.4, -0.7, 0.1])


def test_baseline_result_persists_protocol_and_predictions_before_completion(tmp_path):
    directory = run_artifacts.run_dir("0v1", "baseline_logistic", 4,
                                      root=str(tmp_path))
    result = {
        "metrics": {"accuracy": 0.5},
        "selection": {
            "criterion": "validation_loss", "best_epoch": None,
            "best_value": 0.2, "hyperparameters": {"C": 1.0},
            "n_validation_evaluations": 2,
        },
        "test_evaluations": 1,
        "raw_outputs": np.array([0.3, -0.2]),
        "selected_parameters": np.array([0.1, 0.2]),
    }
    run_artifacts.save_baseline_result(
        directory=directory, result=result, split_id="split-abc",
        sample_ids=[19, 23], y_test=[1, -1], seed=4,
        environment={"python": "3.9.13"})

    with open(os.path.join(directory, "status.json")) as fh:
        status = json.load(fh)
    with open(os.path.join(directory, "selection.json")) as fh:
        selection = json.load(fh)
    with open(os.path.join(directory, "metrics.json")) as fh:
        metrics = json.load(fh)
    predictions = np.load(os.path.join(directory, "predictions.npz"))

    assert status["state"] == "complete"
    assert status["split_id"] == "split-abc"
    assert selection["criterion"] == "validation_loss"
    assert selection["test_evaluations"] == 1
    assert metrics == {"accuracy": 0.5}
    np.testing.assert_array_equal(predictions["sample_ids"], [19, 23])
    assert run_artifacts.is_reusable(
        directory, config={"artifact_schema_version": 1}, seed=4)


def test_test_evaluation_guard_allows_exactly_one_evaluation():
    guard = run_artifacts.TestEvaluationGuard()
    assert guard.evaluate(lambda: "metrics") == "metrics"
    assert guard.count == 1

    with pytest.raises(RuntimeError, match="exactly once"):
        guard.evaluate(lambda: "metrics again")
