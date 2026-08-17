"""Behavioral contract for leakage-free classical and quantum baselines."""
import inspect

import numpy as np
import pytest

from baselines import classical_cnn, quantum_baselines
from experiments import run_experiments


_SPLIT = {
    "X_train": np.array([[-2.0], [-1.0], [1.0], [2.0]]),
    "y_train": np.array([-1, -1, 1, 1]),
    "X_val": np.array([[-1.5], [1.5]]),
    "y_val": np.array([-1, 1]),
    "X_test": np.array([[-0.5], [0.5]]),
    "y_test": np.array([-1, 1]),
}


@pytest.mark.parametrize("runner", [
    classical_cnn.run_classical_baselines,
    quantum_baselines.run_quantum_baselines,
])
def test_public_baseline_runners_require_keyword_only_six_split_api(runner):
    params = inspect.signature(runner).parameters
    for name in _SPLIT:
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        runner(*_SPLIT.values())


def test_experiment_runner_forwards_all_splits_by_exact_keyword(monkeypatch, tmp_path):
    split = tuple(_SPLIT[name] for name in _SPLIT)
    manifest = {"id": "split-1", "sample_ids": list(range(6)), "test_idx": [4, 5]}
    seen = []

    monkeypatch.setattr(run_experiments, "prepare_split", lambda *args: (split, manifest))
    monkeypatch.setattr(run_experiments, "run_qcnn", lambda *args: {"n_params": 7})
    monkeypatch.setattr(run_experiments, "save_metrics_json", lambda *args: None)
    monkeypatch.setattr(run_experiments.run_artifacts, "run_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(run_experiments.run_artifacts, "save_baseline_result", lambda **kwargs: None)

    def capture(**kwargs):
        seen.append(kwargs)
        return {}

    monkeypatch.setattr(run_experiments, "run_classical_baselines", capture)
    monkeypatch.setattr(run_experiments, "run_quantum_baselines", capture)
    run_experiments.run_single("proposed", (0, 1), 3, "unused", 4, 1, True,
                               str(tmp_path), True)

    assert len(seen) == 2
    for forwarded in seen:
        for name, expected in _SPLIT.items():
            assert forwarded[name] is expected


def test_logistic_selection_ignores_test_labels_but_responds_to_validation_labels():
    kwargs = dict(_SPLIT)
    first = classical_cnn.run_logistic_baseline(**kwargs, c_grid=(0.01, 100.0))
    mutated_test = classical_cnn.run_logistic_baseline(
        **{**kwargs, "y_test": -kwargs["y_test"]}, c_grid=(0.01, 100.0))
    mutated_val = classical_cnn.run_logistic_baseline(
        **{**kwargs, "y_val": -kwargs["y_val"]}, c_grid=(0.01, 100.0))

    assert first["selection"] == mutated_test["selection"]
    np.testing.assert_allclose(first["selected_parameters"],
                               mutated_test["selected_parameters"])
    assert first["selection"]["hyperparameters"] != mutated_val["selection"]["hyperparameters"]
    assert first["test_evaluations"] == 1


def test_quantum_checkpoint_uses_validation_each_epoch_and_test_once(monkeypatch):
    validation_calls = []
    test_calls = []
    checkpoints = [np.array([3.0]), np.array([2.0]), np.array([1.0])]

    monkeypatch.setattr(quantum_baselines, "_train_epoch",
                        lambda **kwargs: checkpoints.pop(0).copy())
    monkeypatch.setattr(quantum_baselines, "_validation_loss",
                        lambda **kwargs: validation_calls.append(kwargs["params"].copy()) or
                        float(kwargs["params"][0]))
    monkeypatch.setattr(quantum_baselines, "_raw_outputs",
                        lambda **kwargs: test_calls.append(kwargs["params"].copy()) or
                        np.array([0.2, -0.2]))
    monkeypatch.setattr(quantum_baselines, "_initial_params", lambda **kwargs: np.array([4.0]))

    result = quantum_baselines._train_architecture(
        arch_name="cong", **_SPLIT, seed=1, n_qubits=8, n_epochs=3,
        learning_rate=0.02, use_bce=True)

    assert len(validation_calls) == 3
    assert len(test_calls) == 1
    np.testing.assert_array_equal(test_calls[0], [1.0])
    assert result["selection"]["best_epoch"] == 3
    assert result["test_evaluations"] == 1
