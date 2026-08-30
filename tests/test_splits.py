"""Split protocol tests (UPGRADE_PLAN.md 0.3, F3).

Model selection must never see the test set, so the split has to be
deterministic, stratified, disjoint, and recorded.
"""
import os
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import numpy as np
import pytest

from QCNN.utils import splits


@pytest.fixture
def labels():
    # Deliberately imbalanced so stratification is observable.
    return np.array([1] * 120 + [-1] * 80)


def test_fractions_are_the_mandated_60_15_25():
    assert splits.SPLIT_FRACTIONS == (0.60, 0.15, 0.25)


def test_split_sizes_follow_the_fractions(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={"1": 1, "-1": -1})
    assert len(manifest["train_idx"]) == 120
    assert len(manifest["val_idx"]) == 30
    assert len(manifest["test_idx"]) == 50


@pytest.mark.parametrize("n_total", [200, 400, 800, 1000, 3800, 12665])
def test_split_sizes_are_exact_at_realistic_dataset_sizes(n_total):
    """Guards a floating-point trap: 0.15/(0.15+0.25) is 0.37499999999999994.

    Deriving the validation size from that fraction floors one sample short,
    which would silently mis-size every split in the study.
    """
    y = np.array([1] * (n_total // 2) + [-1] * (n_total - n_total // 2))
    manifest = splits.make_split_manifest(y, seed=0, dataset_id="unit", class_mapping={})

    assert len(manifest["train_idx"]) == round(0.60 * n_total)
    assert len(manifest["val_idx"]) == round(0.15 * n_total)
    assert (
        len(manifest["train_idx"]) + len(manifest["val_idx"]) + len(manifest["test_idx"])
        == n_total
    )


def test_splits_are_disjoint_and_exhaustive(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    train, val, test = (set(manifest[k]) for k in ("train_idx", "val_idx", "test_idx"))
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(range(len(labels)))


def test_splits_are_stratified(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    overall = np.mean(labels == 1)
    for key in ("train_idx", "val_idx", "test_idx"):
        part = labels[np.array(manifest[key])]
        assert abs(np.mean(part == 1) - overall) < 0.05


def test_same_seed_reproduces_the_same_split(labels):
    a = splits.make_split_manifest(labels, seed=7, dataset_id="unit", class_mapping={})
    b = splits.make_split_manifest(labels, seed=7, dataset_id="unit", class_mapping={})
    assert a["train_idx"] == b["train_idx"]
    assert a["val_idx"] == b["val_idx"]
    assert a["test_idx"] == b["test_idx"]
    assert a["id"] == b["id"]


def test_different_seeds_give_different_splits(labels):
    a = splits.make_split_manifest(labels, seed=1, dataset_id="unit", class_mapping={})
    b = splits.make_split_manifest(labels, seed=2, dataset_id="unit", class_mapping={})
    assert a["id"] != b["id"]


def test_manifest_round_trips_through_disk(labels, tmp_path):
    manifest = splits.make_split_manifest(labels, seed=3, dataset_id="unit", class_mapping={"a": 1})
    path = os.path.join(str(tmp_path), "split.json")
    splits.save_manifest(manifest, path)
    assert splits.load_manifest(path) == manifest


def test_concurrent_manifest_writers_never_expose_partial_json(labels, tmp_path):
    """Parallel experiment workers must leave an all-or-nothing manifest."""
    path = str(tmp_path / "split.json")
    manifests = [
        splits.make_split_manifest(
            labels, seed=seed, dataset_id="unit", class_mapping={})
        for seed in (0, 1)
    ]
    errors = []
    stop = threading.Event()

    def writer(manifest):
        for _ in range(40):
            splits.save_manifest(manifest, path)

    def reader():
        while not stop.is_set():
            try:
                loaded = splits.load_manifest(path)
                assert loaded in manifests
            except (AssertionError, OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(exc)

    splits.save_manifest(manifests[0], path)
    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(writer, manifests))
    stop.set()
    reader_thread.join()
    assert not errors
    assert splits.load_manifest(path) in manifests


def test_apply_manifest_returns_matching_features_and_labels(labels):
    X = np.arange(len(labels) * 4, dtype=float).reshape(len(labels), 4)
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits.apply_manifest(manifest, X, labels)

    assert len(X_tr) == len(y_tr) == 120
    assert len(X_va) == len(y_va) == 30
    assert len(X_te) == len(y_te) == 50
    # Row identity is preserved: column 0 equals 4 * original index.
    np.testing.assert_array_equal(X_tr[:, 0], 4.0 * np.array(manifest["train_idx"]))


def test_source_stable_string_sample_ids_round_trip(labels, tmp_path):
    sample_ids = [f"fashion_mnist:train:{index:05d}" for index in range(len(labels))]
    manifest = splits.make_split_manifest(
        labels, seed=0, dataset_id="fashion-unit", class_mapping={},
        sample_ids=sample_ids,
    )
    path = tmp_path / "split.json"
    splits.save_manifest(manifest, str(path))
    loaded = splits.load_manifest(str(path))
    assert loaded["sample_ids"] == sample_ids
    assert loaded["id"] == manifest["id"]


def test_source_sample_ids_are_part_of_manifest_identity(labels):
    first = [f"mnist:train:{index:05d}" for index in range(len(labels))]
    second = [f"fashion_mnist:train:{index:05d}" for index in range(len(labels))]
    a = splits.make_split_manifest(
        labels, seed=0, dataset_id="binary-unit", class_mapping={}, sample_ids=first,
    )
    b = splits.make_split_manifest(
        labels, seed=0, dataset_id="binary-unit", class_mapping={}, sample_ids=second,
    )
    assert a["train_idx"] == b["train_idx"]
    assert a["id"] != b["id"]


@pytest.mark.parametrize("sample_ids", [["same"] * 200, [""] * 200, [True] * 200])
def test_invalid_source_sample_ids_are_rejected(labels, sample_ids):
    with pytest.raises(ValueError, match="sample IDs"):
        splits.make_split_manifest(
            labels, seed=0, dataset_id="unit", class_mapping={},
            sample_ids=sample_ids,
        )


def test_verify_manifest_rejects_an_overlapping_split(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    manifest["val_idx"] = manifest["val_idx"] + [manifest["train_idx"][0]]
    with pytest.raises(ValueError, match="disjoint"):
        splits.verify_manifest(manifest)


def test_verify_manifest_rejects_corrupt_source_ids(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    manifest["sample_ids"][1] = manifest["sample_ids"][0]
    with pytest.raises(ValueError, match="sample IDs"):
        splits.verify_manifest(manifest)


def test_apply_manifest_rejects_a_size_mismatch(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    with pytest.raises(ValueError, match="n_total"):
        splits.apply_manifest(manifest, np.zeros((10, 4)), labels[:10])


def test_class_mapping_matches_the_encoder_that_actually_labels_the_data():
    """The manifest must name the same positive class the encoder produces.

    ``encode_labels`` maps sorted(unique(y))[0] -> -1 and [1] -> +1. A manifest
    built from CLI argument order records the opposite whenever the arguments
    are not already sorted, which misattributes precision/recall/F1 to the wrong
    digit while leaving accuracy (symmetric) unchanged and therefore silent.
    """
    from QCNN.utils.data_preprocessing import encode_labels

    for classes in [(0, 1), (1, 0), (3, 5), (5, 3), (4, 9), (9, 4)]:
        raw = np.array([classes[0]] * 5 + [classes[1]] * 5)
        encoded = encode_labels(raw, encoding="binary")
        mapping = splits.class_mapping_for(classes)

        for digit, sign in zip(raw, encoded):
            assert mapping[str(digit)] == sign, (
                "manifest says %s -> %d but encode_labels produced %d"
                % (digit, mapping[str(digit)], sign)
            )


def test_class_mapping_is_independent_of_argument_order():
    assert splits.class_mapping_for((0, 1)) == splits.class_mapping_for((1, 0))
    assert splits.class_mapping_for((0, 1)) == {"0": -1, "1": 1}
