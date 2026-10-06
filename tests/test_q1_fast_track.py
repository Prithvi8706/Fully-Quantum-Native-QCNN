import json
from pathlib import Path

import numpy as np
import pytest

from experiments import q1_fast_track_analysis as analysis
from experiments import run_experiments
from QCNN.utils import run_artifacts
from QCNN.utils.metrics import compute_classification_metrics


def test_train_sample_budget_matches_frozen_split_fraction():
    total = run_experiments._total_samples_for_train_budget(400)
    assert total == 666
    assert round(0.60 * total) == 400
    assert round(0.60 * (total - 1)) < 400


def test_malformed_evidence_is_never_a_keep_candidate(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{")
    record = analysis.inspect_json_artifact(path, tmp_path)
    assert record["parse_state"] == "malformed"
    assert record["disposition"] == "remove_claim"


def test_inventory_does_not_hash_itself(tmp_path):
    evidence = tmp_path / "Results" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "q1_fast_track_inventory.json").write_text("{}")
    (evidence / "candidate.json").write_text("{}")
    records = analysis.evidence_artifacts(tmp_path)
    assert [record["path"] for record in records] == ["Results/evidence/candidate.json"]


def test_dirty_unapproved_campaign_requires_bounded_rerun(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "repository": {"dirty": True},
        "approvals": {"launch": {"approved": False}},
        "request": {"seeds": [0, 1]},
    }))
    record = analysis.inspect_json_artifact(path, tmp_path)
    assert record["disposition"] == "bounded_rerun"
    assert record["seeds"] == ["0", "1"]


def test_run_groups_validate_complete_cells(tmp_path):
    root = tmp_path
    directory = run_artifacts.run_dir(
        "0v1", "proposed", 0, root=str(root / "Results" / "runs")
    )
    config = {"n_qubits": 10}
    run_artifacts.start_run(
        directory, config=config, split_id="split-a", seed=0,
        environment={"python": "3.9.13"},
    )
    np.savez(run_artifacts.weights_path(directory), weights=np.zeros(3))
    run_artifacts.save_predictions(directory, [1], [1], [0.3])
    run_artifacts.complete_run(directory, {"accuracy": 1.0})

    groups = analysis.run_groups(root)
    assert groups == [{
        "dataset_pair": "0v1",
        "config": "proposed",
        "valid_seeds": [0],
        "invalid_runs": [],
        "split_ids": ["split-a"],
        "disposition": "keep_candidate",
    }]


def test_manuscript_findings_locate_scope_sensitive_claims(tmp_path):
    manuscript = tmp_path / "paper.tex"
    manuscript.write_text(
        "Accuracy is 98\\% on hardware.\n"
        "The spatial model is robust to depolarizing noise.\n"
        "\\begin{thebibliography}{1}\n"
        "Noise in a cited title should not be audited.\n"
    )
    findings = analysis.manuscript_findings(manuscript, tmp_path)
    assert [item["line"] for item in findings] == [1, 2]
    assert {category for item in findings for category in item["categories"]} >= {
        "numerical_result", "hardware_scope", "image_locality", "noise_or_robustness"
    }


def test_balanced_accuracy_is_reported_for_imbalanced_test_data():
    metrics = compute_classification_metrics(
        np.array([1, 1, -1, -1, -1, -1]),
        np.array([0.8, -0.2, -0.4, -0.5, 0.1, -0.7]),
    )
    # Positive recall=1/2, negative recall=3/4.
    assert metrics["balanced_accuracy"] == 0.625


def _comparison_cell(root, task, arm, seed, raw, split_id="split-unit"):
    config_name = "proposed" if arm == "proposed" else f"baseline_{arm}"
    directory = run_artifacts.run_dir(task, config_name, seed, root=str(root))
    y_true = np.array([-1, 1, -1, 1])
    metrics = compute_classification_metrics(y_true, np.asarray(raw, dtype=float))
    if arm == "proposed":
        run_artifacts.start_run(
            directory, config={"arm": arm}, split_id=split_id, seed=seed,
            environment={})
        np.savez(run_artifacts.weights_path(directory), parameters=np.array([0.1]))
        run_artifacts.save_predictions(
            directory, ["a", "b", "c", "d"], y_true, raw)
        run_artifacts.complete_run(directory, {"accuracy": metrics["accuracy"]})
    else:
        run_artifacts.save_baseline_result(
            directory=directory,
            result={
                "metrics": metrics,
                "selection": {
                    "criterion": "validation_loss", "best_epoch": 1,
                    "best_value": 0.2, "hyperparameters": {},
                    "n_validation_evaluations": 1,
                },
                "test_evaluations": 1,
                "raw_outputs": np.asarray(raw, dtype=float),
                "selected_parameters": np.array([0.2]),
            },
            split_id=split_id, sample_ids=["a", "b", "c", "d"],
            y_test=y_true, seed=seed, environment={},
        )
    return Path(directory)


def test_comparison_aggregation_validates_and_pairs_the_frozen_matrix(tmp_path):
    runs = tmp_path / "runs"
    task = "mnist_0v1"
    for seed in (0, 1):
        _comparison_cell(runs, task, "proposed", seed, [-0.8, 0.8, -0.6, 0.7])
        _comparison_cell(runs, task, "logistic", seed, [-0.7, 0.6, -0.4, 0.2])
        _comparison_cell(runs, task, "mlp", seed, [-0.5, -0.1, -0.2, 0.3])
        _comparison_cell(runs, task, "ttn", seed, [0.1, 0.4, -0.3, 0.2])

    payload = analysis.build_comparison(
        runs, seeds=(0, 1),
        tasks=(("mnist", (0, 1), "unit task"),),
        allow_noncanonical=True,
    )
    result = payload["tasks"][task]
    assert result["arms"]["proposed"]["metrics"]["accuracy"]["mean"] == 1.0
    assert set(result["proposed_vs"]) == {"logistic", "mlp", "ttn"}
    assert payload["protocol"]["multiplicity"]["n_tests"] == 3
    source_paths = {
        item["path"] for item in payload["provenance"]["source_files"]
    }
    assert "experiments/q1_fast_track_analysis.py" in source_paths
    assert "QCNN/circuits.py" in source_paths
    assert payload["provenance"]["environment"]["lock_files"][0]["path"] == (
        "requirements-lock.txt"
    )


def test_comparison_aggregation_rejects_a_split_mismatch(tmp_path):
    runs = tmp_path / "runs"
    task = "mnist_0v1"
    for arm in analysis.COMPARISON_ARMS:
        split_id = "wrong" if arm == "ttn" else "right"
        _comparison_cell(
            runs, task, arm, 0, [-0.8, 0.8, -0.6, 0.7], split_id=split_id)
    with pytest.raises(ValueError, match="split mismatch"):
        analysis.build_comparison(
            runs, seeds=(0,), tasks=(("mnist", (0, 1), "unit task"),),
            allow_noncanonical=True,
        )


def test_scientific_comparison_rejects_custom_task_matrix(tmp_path):
    with pytest.raises(ValueError, match="frozen Q1 task registry"):
        analysis.build_comparison(
            tmp_path / "runs",
            seeds=tuple(range(5)),
            tasks=(("mnist", (0, 1), "post-hoc task"),),
        )


def test_scientific_comparison_requires_canonical_manifests(tmp_path):
    with pytest.raises(ValueError, match="canonical comparison manifest"):
        analysis.build_comparison(
            tmp_path / "runs",
            manifests_root=tmp_path / "manifests",
        )
