import gzip
import os
import struct

import numpy as np
import pytest

from QCNN.utils import dataset_registry
from experiments import run_experiments


def _write_idx(root, dataset, images, labels, *, partition="train", image_magic=2051):
    spec = dataset_registry.get_spec(dataset)
    directory = root / spec.directory
    directory.mkdir(parents=True)
    image_source = next(
        source for source in spec.files
        if source.partition == partition and source.kind == "images"
    )
    label_source = next(
        source for source in spec.files
        if source.partition == partition and source.kind == "labels"
    )
    opener = gzip.open if image_source.compressed else open
    with opener(directory / image_source.filename, "wb") as handle:
        handle.write(struct.pack(">IIII", image_magic, len(images), 28, 28))
        handle.write(np.asarray(images, dtype=np.uint8).reshape(len(images), -1).tobytes())
    opener = gzip.open if label_source.compressed else open
    with opener(directory / label_source.filename, "wb") as handle:
        handle.write(struct.pack(">II", 2049, len(labels)))
        handle.write(np.asarray(labels, dtype=np.uint8).tobytes())


def test_q1_tasks_are_frozen_before_test_inspection():
    assert dataset_registry.Q1_TASKS == (
        ("mnist", (0, 1), "headline continuity / easy calibration"),
        ("mnist", (3, 5), "harder in-domain task"),
        ("fashion_mnist", (0, 6), "T-shirt/top versus Shirt"),
        ("kmnist", (2, 3), "distinct Kuzushiji character classes"),
    )


def test_all_sources_pin_sha256_and_remote_sources_require_tls():
    for dataset in dataset_registry.DATASETS:
        for source in dataset_registry.get_spec(dataset).files:
            assert source.sha256 is not None
            assert len(source.sha256) == 64
    for dataset in ("fashion_mnist", "kmnist"):
        for source in dataset_registry.get_spec(dataset).files:
            assert source.url.startswith("https://")


def test_mnist_is_a_checksum_pinned_local_snapshot():
    spec = dataset_registry.get_spec("mnist")
    assert "checksum-pinned local release" in spec.version
    assert all(source.url is None for source in spec.files)
    assert all(source.md5 is not None and source.sha256 is not None for source in spec.files)


@pytest.mark.parametrize(
    "dataset,classes", [("mnist", (0, 1)), ("fashion_mnist", (0, 6)), ("kmnist", (2, 3))]
)
def test_registered_binary_loader_preserves_source_ids(tmp_path, dataset, classes):
    images = np.arange(6 * 28 * 28, dtype=np.uint8).reshape(6, 28, 28)
    labels = np.asarray([classes[0], 9, classes[1], classes[0], 8, classes[1]])
    _write_idx(tmp_path, dataset, images, labels)
    selected, selected_labels, sample_ids = dataset_registry.load_binary(
        dataset, classes, data_root=tmp_path
    )
    assert selected.shape == (4, 784)
    np.testing.assert_array_equal(selected_labels, [classes[0], classes[1], classes[0], classes[1]])
    assert sample_ids == [
        f"{dataset}:train:00000", f"{dataset}:train:00002",
        f"{dataset}:train:00003", f"{dataset}:train:00005",
    ]


def test_quantum_loader_reuses_frozen_28_by_28_preprocessing(tmp_path):
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    images[1, 0, 0] = 255
    images[3, 1, 1] = 128
    _write_idx(tmp_path, "fashion_mnist", images, [0, 6, 0, 6])
    features, labels, sample_ids = dataset_registry.load_binary_quantum(
        "fashion_mnist", (0, 6), data_root=tmp_path
    )
    assert features.shape == (4, 784)
    assert features.min() == 0.0 and features.max() == 1.0
    np.testing.assert_array_equal(labels, [-1, 1, -1, 1])
    assert len(sample_ids) == 4


def test_invalid_idx_magic_fails_closed(tmp_path):
    _write_idx(tmp_path, "kmnist", np.zeros((2, 28, 28)), [2, 3], image_magic=0)
    with pytest.raises(ValueError, match="invalid IDX image header"):
        dataset_registry.load_raw("kmnist", data_root=tmp_path)


def test_registered_runner_task_parser_and_smoke_namespace():
    assert run_experiments._parse_task("fashion_mnist:0,6") == (
        "fashion_mnist", (0, 6))
    assert run_experiments._fmt_task("fashion_mnist", (0, 6)) == "fashion_mnist_0v6"
    assert run_experiments._fmt_task(
        "fashion_mnist", (0, 6), "smoke") == "smoke__fashion_mnist_0v6"
    with pytest.raises(Exception, match="dataset:low,high"):
        run_experiments._parse_task("fashion_mnist")


def test_registered_runner_manifest_preserves_source_identity(tmp_path, monkeypatch):
    features = np.arange(200 * 16, dtype=float).reshape(200, 16)
    labels = np.asarray([-1] * 100 + [1] * 100)
    source_ids = [f"kmnist:train:{index:05d}" for index in range(200)]

    def fake_loader(*args, **kwargs):
        return features, labels, source_ids

    monkeypatch.setattr(dataset_registry, "load_binary_quantum", fake_loader)
    monkeypatch.setattr(run_experiments, "MANIFEST_ROOT", str(tmp_path / "manifests"))
    cfg = run_experiments.build_config(
        run_experiments.ABLATION_CONFIGS["proposed"], seed=4)
    split, manifest = run_experiments.prepare_split(
        cfg, (2, 3), tmp_path, None, dataset_key="kmnist")

    assert manifest["dataset_id"] == "kmnist_2v3_n200"
    assert manifest["sample_ids"] == source_ids
    assert cfg.split_id == manifest["id"]
    assert sum(len(part) for part in split[::2]) == 200


def test_smoke_campaign_isolates_every_output_root(tmp_path, monkeypatch):
    captured = []

    def fake_execute(payload):
        captured.append(payload)
        pair, config_name, seed = payload[:3]
        return payload[9], pair, config_name, seed, payload[10], None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_experiments, "EXP_ROOT", os.path.join("Results", "experiments"))
    monkeypatch.setattr(
        run_experiments, "FAILURE_MANIFEST",
        os.path.join("Results", "experiments", "failures.json"))
    monkeypatch.setattr(
        run_experiments, "MANIFEST_ROOT", os.path.join("Results", "manifests"))
    monkeypatch.setattr(
        run_experiments.run_artifacts, "RUN_ROOT", os.path.join("Results", "runs"))
    monkeypatch.setattr(run_experiments, "_execute_cell", fake_execute)
    monkeypatch.setattr(run_experiments, "_is_reusable_cell", lambda *args: False)
    monkeypatch.setattr("sys.argv", ["run_experiments", "--smoke"])
    run_experiments.main()

    assert len(captured) == 3
    roots = captured[0][8]
    smoke_root = os.path.join("Results", "smoke", "q1_datasets")
    assert roots == {
        "experiments": os.path.join(smoke_root, "experiments"),
        "runs": os.path.join(smoke_root, "runs"),
        "manifests": os.path.join(smoke_root, "manifests"),
        "failures": os.path.join(smoke_root, "experiments", "failures.json"),
    }
    assert not (tmp_path / "Results" / "experiments" / "summary.csv").exists()


def test_registered_scientific_campaign_isolates_every_output_root(tmp_path, monkeypatch):
    captured = []

    def fake_execute(payload):
        captured.append(payload)
        pair, config_name, seed = payload[:3]
        return payload[9], pair, config_name, seed, payload[10], None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_experiments, "EXP_ROOT", os.path.join("Results", "experiments"))
    monkeypatch.setattr(
        run_experiments, "FAILURE_MANIFEST",
        os.path.join("Results", "experiments", "failures.json"))
    monkeypatch.setattr(
        run_experiments, "MANIFEST_ROOT", os.path.join("Results", "manifests"))
    monkeypatch.setattr(
        run_experiments.run_artifacts, "RUN_ROOT", os.path.join("Results", "runs"))
    monkeypatch.setattr(run_experiments, "_execute_cell", fake_execute)
    monkeypatch.setattr(run_experiments, "_is_reusable_cell", lambda *args: False)
    monkeypatch.setattr("sys.argv", [
        "run_experiments", "--task", "mnist:0,1", "--configs", "proposed",
        "--seeds", "0", "--no-baselines",
    ])
    run_experiments.main()

    assert len(captured) == 1
    roots = captured[0][8]
    comparison_root = os.path.join("Results", "q1_comparison")
    assert roots == {
        "experiments": os.path.join(comparison_root, "experiments"),
        "runs": os.path.join(comparison_root, "runs"),
        "manifests": os.path.join(comparison_root, "manifests"),
        "failures": os.path.join(comparison_root, "experiments", "failures.json"),
    }
    assert not (tmp_path / "Results" / "experiments" / "summary.csv").exists()
