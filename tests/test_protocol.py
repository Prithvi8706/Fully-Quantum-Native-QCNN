"""No test-set information may reach model selection (UPGRADE_PLAN.md 0.3, F3).

The trainer previously drove checkpointing, LR plateau, and early stopping from
test accuracy, which is what made the archived 98.86% unusable at a Q1 venue.
"""
import inspect
import re

import numpy as np
import pytest

from QCNN.training.Qtrainer import QuantumNativeTrainer


def _trainer_source():
    return inspect.getsource(QuantumNativeTrainer)


def test_trainer_signature_takes_validation_not_test():
    params = list(inspect.signature(
        QuantumNativeTrainer.train_pure_quantum_cnn).parameters)
    assert "X_val" in params and "y_val" in params
    assert "X_test" not in params and "y_test" not in params


def test_trainer_source_never_mentions_the_test_set():
    """The grep check mandated by UPGRADE_PLAN.md 0.3."""
    offenders = re.findall(r"\b(?:X_test|y_test|test_accuracy|test_preds)\b", _trainer_source())
    assert offenders == [], "test-set identifiers still present in the trainer: {}".format(offenders)


def test_trainer_selects_on_validation_accuracy():
    source = _trainer_source()
    assert "val_accuracy" in source
    assert "best_val_accuracy" in source


def test_trainer_does_not_write_the_archived_headline_weights():
    source = _trainer_source()
    assert "quantum_model_params.npz" not in source, (
        "the trainer must write to a caller-supplied per-run path, never the "
        "archived headline weights (UPGRADE_PLAN.md M0.5)"
    )


def test_weight_saving_is_conditional_on_a_caller_supplied_path():
    """With weights_path=None the trainer must not write any .npz file."""
    params = inspect.signature(QuantumNativeTrainer.train_pure_quantum_cnn).parameters
    assert params["weights_path"].default is None

    source = inspect.getsource(QuantumNativeTrainer.train_pure_quantum_cnn)
    assert "if weights_path" in source


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_main_uses_split_manifests_not_ad_hoc_train_test_split():
    source = _read("main.py")
    assert "train_test_split(" not in source, (
        "main.py must consume a split manifest (UPGRADE_PLAN.md 0.3)")
    assert "make_split_manifest" in source
    assert "apply_manifest" in source


def test_runner_uses_split_manifests_not_ad_hoc_train_test_split():
    source = _read("experiments/run_experiments.py")
    assert "train_test_split(" not in source
    assert "apply_manifest" in source


def test_main_evaluates_the_test_set_through_the_guard():
    source = _read("main.py")
    assert "TestEvaluationGuard" in source
    assert source.count("guard.evaluate(") == 1


def test_runner_evaluates_the_test_set_through_the_guard():
    source = _read("experiments/run_experiments.py")
    assert "TestEvaluationGuard" in source
    assert source.count("guard.evaluate(") == 1


def test_seed_policy_lives_in_one_module():
    for path in ("main.py", "experiments/run_experiments.py", "noise_sim.py"):
        source = _read(path)
        assert "def _seed_everything" not in source, (
            "{} still defines its own seed policy; import "
            "QCNN.utils.seeding.seed_everything instead".format(path))


def test_runner_persists_per_example_predictions():
    source = _read("experiments/run_experiments.py")
    assert "save_predictions" in source, (
        "per-example predictions are required for the paired tests in Phase 5 "
        "(UPGRADE_PLAN.md M0.5)")
    assert "run_artifacts.run_dir" in source or "run_dir(" in source
