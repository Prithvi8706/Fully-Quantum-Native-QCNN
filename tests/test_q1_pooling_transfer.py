import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from experiments import q1_pooling_transfer as transfer
from QCNN.utils import run_artifacts


def _cell(
    root: Path,
    arm: str,
    seed: int,
    *,
    split_id: Optional[str] = None,
    role: str = "scientific",
    sample_ids=None,
    y_true=None,
    raw_outputs=None,
):
    directory = Path(run_artifacts.run_dir(
        transfer.TASK_NAME, arm, seed, root=str(root), create=True
    ))
    y_true = np.asarray([-1, 1, -1, 1] if y_true is None else y_true)
    if sample_ids is None:
        sample_ids = [f"fashion_mnist:train:{index:05d}" for index in range(4)]
    if raw_outputs is None:
        raw_outputs = np.asarray([-0.8, 0.8, -0.4, 0.4])
    config = {
        "arm": arm,
        "pooling_mode": transfer.POOLING_MODES[arm],
        "evidence_role": role,
    }
    metrics = transfer.compute_classification_metrics(y_true, raw_outputs)
    run_artifacts.start_run(
        str(directory), config=config, split_id=split_id or f"split-{seed}",
        seed=seed, environment={}
    )
    np.savez(run_artifacts.weights_path(str(directory)), parameters=np.asarray([0.1]))
    run_artifacts.save_predictions(str(directory), sample_ids, y_true, raw_outputs)
    run_artifacts.complete_run(str(directory), {"accuracy": metrics["accuracy"]})
    return directory


def _matrix(root: Path, *, split_ids=None, role_overrides=None):
    split_ids = split_ids or {}
    role_overrides = role_overrides or {}
    for seed in transfer.SEEDS:
        for arm in transfer.POOLING_ARMS:
            _cell(
                root, arm, seed,
                split_id=split_ids.get((seed, arm)),
                role=role_overrides.get((seed, arm), "scientific"),
            )


def _canonical_matrix(root: Path, *, split_ids=None, role_overrides=None,
                      raw_outputs_by_arm=None):
    """Build a complete fixture using the frozen-size manifests."""
    split_ids = split_ids or {}
    role_overrides = role_overrides or {}
    raw_outputs_by_arm = raw_outputs_by_arm or {}
    manifests = root / "manifests"
    runs = root / "runs"
    all_ids = [f"fashion_mnist:train:{index:05d}" for index in range(666)]
    labels = np.asarray([-1] * 333 + [1] * 333)
    from QCNN.utils import splits

    for seed in transfer.SEEDS:
        manifest = splits.make_split_manifest(
            labels, seed=seed, dataset_id=f"{transfer.TASK_NAME}_n666",
            class_mapping={"0": -1, "6": 1}, sample_ids=all_ids,
        )
        manifests.mkdir(parents=True, exist_ok=True)
        (manifests / f"{transfer.TASK_NAME}_n666_seed{seed}.json").write_text(
            json.dumps(manifest)
        )
        test_idx = np.asarray(manifest["test_idx"], dtype=int)
        test_ids = [manifest["sample_ids"][index] for index in manifest["test_idx"]]
        y_true = labels[test_idx]
        for arm in transfer.POOLING_ARMS:
            if arm in raw_outputs_by_arm:
                raw_outputs = raw_outputs_by_arm[arm](y_true)
            else:
                raw_outputs = np.where(y_true > 0, 0.7, -0.7)
            _cell(
                runs, arm, seed,
                split_id=split_ids.get((seed, arm), manifest["id"]),
                role=role_overrides.get((seed, arm), "scientific"),
                sample_ids=test_ids, y_true=y_true, raw_outputs=raw_outputs,
            )
    return runs, manifests


def test_plan_freezes_fifteen_scientific_cells(tmp_path):
    plan = transfer.plan_cells(transfer_root=tmp_path)
    assert plan["dataset"] == "fashion_mnist"
    assert plan["classes"] == [0, 6]
    assert plan["seeds"] == [0, 1, 2]
    assert plan["arms"] == list(transfer.POOLING_ARMS)
    assert plan["n_cells"] == 15
    assert all(cell["evidence_role"] == "scientific" for cell in plan["cells"])
    assert all("smoke" not in cell["run_directory"] for cell in plan["cells"])


def test_transfer_aggregate_validates_and_summarizes_five_arms(tmp_path):
    runs, manifests = _canonical_matrix(tmp_path)
    payload = transfer.build_transfer(runs, manifests_root=manifests)

    assert payload["schema"] == transfer.SCHEMA
    assert payload["protocol"]["task"] == transfer.TASK_NAME
    assert payload["protocol"]["seeds"] == [0, 1, 2]
    assert payload["protocol"]["arms"] == list(transfer.POOLING_ARMS)
    assert set(payload["arms"]) == set(transfer.POOLING_ARMS)
    assert set(payload["comparisons"]) == set(transfer.POOLING_ARMS) - {transfer.REFERENCE_ARM}
    assert payload["arms"][transfer.REFERENCE_ARM]["metrics"]["accuracy"]["n"] == 3
    assert all(
        len(payload["arms"][arm]["cells"]) == 3
        for arm in transfer.POOLING_ARMS
    )
    source_paths = {
        item["path"] for item in payload["provenance"]["source_files"]
    }
    assert "experiments/q1_pooling_transfer.py" in source_paths
    assert "QCNN/circuits.py" in source_paths
    assert payload["provenance"]["environment"]["lock_files"][0]["path"] == (
        "requirements-lock.txt"
    )


def test_transfer_does_not_use_repeated_seed_rows_as_mcnemar_trials(tmp_path):
    """Cross-seed discordances remain descriptive when sample IDs repeat."""
    runs, manifests = _canonical_matrix(
        tmp_path,
        raw_outputs_by_arm={
            "e3_pool_none": lambda labels: -np.where(labels > 0, 0.8, -0.8),
            "e3_pool_unitary": lambda labels: np.where(labels > 0, 0.8, -0.8),
        },
    )
    payload = transfer.build_transfer(runs, manifests_root=manifests)
    comparison = payload["comparisons"]["e3_pool_none"]
    pooled = comparison["mcnemar_pooled"]

    # Canonical seed manifests overlap in source IDs.  Counts can be retained
    # for descriptive reporting, but repeated rows cannot be treated as
    # independent McNemar observations.
    test_ids = []
    for path in sorted(manifests.glob(f"{transfer.TASK_NAME}_n666_seed*.json")):
        test_ids.extend(
            json.loads(path.read_text())["sample_ids"][index]
            for index in json.loads(path.read_text())["test_idx"]
        )
    assert pooled["n_observations"] == 3 * 166
    assert pooled["n_unique_sample_ids"] == len(set(test_ids))
    assert pooled["n_unique_sample_ids"] < pooled["n_observations"]
    assert pooled["b01"] == 3 * 166
    assert pooled["b10"] == 0
    assert pooled["descriptive_only"] is True
    assert pooled["p_value"] is None
    assert pooled["p_holm"] is None
    assert pooled["significant"] is False
    assert payload["protocol"]["multiplicity"]["mcnemar"]["n_tests"] == 0
    assert comparison["wilcoxon"]["p_value"] is not None


def test_transfer_rejects_smoke_role_even_when_artifact_is_complete(tmp_path):
    runs, manifests = _canonical_matrix(
        tmp_path, role_overrides={(0, transfer.POOLING_ARMS[0]): "smoke"})
    with pytest.raises(ValueError, match="smoke/partial"):
        transfer.build_transfer(runs, manifests_root=manifests)


def test_transfer_rejects_cross_arm_split_mismatch(tmp_path):
    runs, manifests = _canonical_matrix(
        tmp_path, split_ids={(1, "e3_pool_su4"): "different-split"})
    with pytest.raises(ValueError, match="split ID disagrees"):
        transfer.build_transfer(runs, manifests_root=manifests)


def test_transfer_requires_canonical_manifests_for_scientific_aggregation(tmp_path):
    _matrix(tmp_path)
    with pytest.raises(ValueError, match="manifests_root is required"):
        transfer.build_transfer(tmp_path)


def test_transfer_rejects_smoke_namespace(tmp_path):
    smoke_root = tmp_path / "smoke" / "runs"
    with pytest.raises(ValueError, match="smoke namespace"):
        transfer.build_transfer(smoke_root)


def test_transfer_run_constructs_bounded_existing_runner_command(tmp_path, monkeypatch):
    captured = {}

    def fake_main(output_roots=None):
        captured["argv"] = list(__import__("sys").argv)
        captured["roots"] = output_roots

    monkeypatch.setattr("experiments.run_experiments.main", fake_main)
    result = transfer.run_transfer(
        data_root=tmp_path / "datasets",
        transfer_root=tmp_path / "transfer",
        jobs=2,
        output=None,
    )
    assert result is None
    assert captured["argv"][0] == "run_experiments"
    assert captured["argv"][1:3] == ["--task", "fashion_mnist:0,6"]
    assert captured["argv"][captured["argv"].index("--configs") + 1:
                       captured["argv"].index("--seeds")] == list(transfer.POOLING_ARMS)
    assert captured["argv"][captured["argv"].index("--seeds") + 1:
                       captured["argv"].index("--samples")] == ["0", "1", "2"]
    assert "--no-baselines" in captured["argv"]
    assert captured["roots"]["runs"].endswith("transfer\\runs") or captured["roots"]["runs"].endswith("transfer/runs")


def test_manifest_gate_matches_test_ids_when_requested(tmp_path):
    # Construct one canonical manifest per frozen seed and mirror its test IDs
    # into every arm.  This checks that the evidence command consumes the
    # persisted split rather than merely trusting status.json.
    manifests = tmp_path / "manifests"
    runs = tmp_path / "runs"
    for seed in transfer.SEEDS:
        all_ids = [f"fashion_mnist:train:{index:05d}" for index in range(666)]
        labels = np.asarray([-1] * 333 + [1] * 333)
        from QCNN.utils import splits

        manifest = splits.make_split_manifest(
            labels, seed=seed, dataset_id=f"{transfer.TASK_NAME}_n666",
            class_mapping={"0": -1, "6": 1}, sample_ids=all_ids,
        )
        manifests.mkdir(parents=True, exist_ok=True)
        (manifests / f"{transfer.TASK_NAME}_n666_seed{seed}.json").write_text(
            json.dumps(manifest)
        )
        test_ids = [manifest["sample_ids"][index] for index in manifest["test_idx"]]
        y_true = labels[np.asarray(manifest["test_idx"], dtype=int)]
        for arm in transfer.POOLING_ARMS:
            _cell(
                runs, arm, seed, split_id=manifest["id"], sample_ids=test_ids,
                y_true=y_true,
                raw_outputs=np.where(y_true > 0, 0.7, -0.7),
            )

    payload = transfer.build_transfer(runs, manifests_root=manifests)
    assert payload["manifests"]["0"]["id"]
