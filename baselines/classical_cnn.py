"""Leakage-free classical baselines for comparison with the QCNN."""
from __future__ import annotations

import copy
import warnings

import numpy as np

from QCNN.utils.metrics import compute_classification_metrics
from QCNN.utils.run_artifacts import TestEvaluationGuard


def _flatten(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    return X.reshape(X.shape[0], -1)


def _raw_from_proba(proba_pos: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(proba_pos, dtype=float) - 1.0


def _positive_proba(clf, X):
    positive = int(np.flatnonzero(np.asarray(clf.classes_) == 1)[0])
    return clf.predict_proba(X)[:, positive]


def _result(y_test, raw, selection, selected_parameters, extra_metrics=None, test_count=1):
    metrics = compute_classification_metrics(np.asarray(y_test), raw)
    if extra_metrics:
        metrics.update(extra_metrics)
    return {
        "metrics": metrics,
        "selection": selection,
        "test_evaluations": int(test_count),
        "raw_outputs": np.asarray(raw),
        "selected_parameters": np.asarray(selected_parameters, dtype=float),
    }


def run_logistic_baseline(*, X_train, y_train, X_val, y_val, X_test, y_test,
                          seed: int = 42, c_grid=None) -> dict:
    """Fit on training data, select C on validation loss, and test once."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    Xtr, Xv, Xte = _flatten(X_train), _flatten(X_val), _flatten(X_test)
    ytr = np.where(np.asarray(y_train) == 1, 1, 0)
    yv = np.where(np.asarray(y_val) == 1, 1, 0)
    candidates = tuple(c_grid) if c_grid is not None else (1.0,)
    best = None
    for c in candidates:
        clf = LogisticRegression(C=float(c), max_iter=2000, random_state=seed)
        clf.fit(Xtr, ytr)
        value = float(log_loss(yv, _positive_proba(clf, Xv), labels=[0, 1]))
        if best is None or value < best[0]:
            best = (value, float(c), clf)
    value, selected_c, clf = best
    guard = TestEvaluationGuard()
    raw = guard.evaluate(lambda: _raw_from_proba(_positive_proba(clf, Xte)))
    params = np.concatenate([clf.coef_.reshape(-1), clf.intercept_.reshape(-1)])
    return _result(y_test, raw, {
        "criterion": "validation_loss", "best_epoch": None, "best_value": value,
        "hyperparameters": {"C": selected_c},
        "n_validation_evaluations": len(candidates),
    }, params, test_count=guard.count)


def _hidden_for_param_budget(n_features: int, target_params: int | None) -> tuple:
    if not target_params or target_params <= 0:
        return (16,)
    h = max(2, int(round((target_params - 1) / (n_features + 2))))
    return (min(h, 256),)


def run_mlp_baseline(*, X_train, y_train, X_val, y_val, X_test, y_test,
                     seed: int = 42, target_params: int | None = None,
                     n_epochs: int = 30) -> dict:
    """Train one epoch at a time and restore the best validation checkpoint."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import log_loss
    from sklearn.neural_network import MLPClassifier

    Xtr, Xv, Xte = _flatten(X_train), _flatten(X_val), _flatten(X_test)
    ytr = np.where(np.asarray(y_train) == 1, 1, 0)
    yv = np.where(np.asarray(y_val) == 1, 1, 0)
    hidden = _hidden_for_param_budget(Xtr.shape[1], target_params)
    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=1, warm_start=True,
                        early_stopping=False, random_state=seed)
    best = None
    for epoch in range(1, max(1, n_epochs) + 1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(Xtr, ytr)
        value = float(log_loss(yv, _positive_proba(clf, Xv), labels=[0, 1]))
        if best is None or value < best[0]:
            best = (value, epoch, copy.deepcopy(clf.coefs_), copy.deepcopy(clf.intercepts_))
    value, best_epoch, coefs, intercepts = best
    clf.coefs_ = copy.deepcopy(coefs)
    clf.intercepts_ = copy.deepcopy(intercepts)
    guard = TestEvaluationGuard()
    raw = guard.evaluate(lambda: _raw_from_proba(_positive_proba(clf, Xte)))
    params = np.concatenate([p.reshape(-1) for p in coefs + intercepts])
    return _result(y_test, raw, {
        "criterion": "validation_loss", "best_epoch": best_epoch, "best_value": value,
        "hyperparameters": {"hidden_layer_sizes": list(hidden)},
        "n_validation_evaluations": max(1, n_epochs),
    }, params, {"hidden_layer_sizes": list(hidden)}, guard.count)


def run_classical_baselines(*, X_train, y_train, X_val, y_val, X_test, y_test,
                            seed: int = 42, target_params: int | None = None) -> dict:
    common = dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                  X_test=X_test, y_test=y_test, seed=seed)
    return {
        "logistic": run_logistic_baseline(**common),
        "mlp": run_mlp_baseline(**common, target_params=target_params),
    }


def build_baseline_cnn(input_shape=(4, 4, 1)):
    try:
        from tensorflow.keras import layers, models
    except ImportError as exc:
        raise ImportError("TensorFlow not installed. Install via: pip install tensorflow") from exc
    model = models.Sequential([
        layers.Input(shape=input_shape), layers.Conv2D(4, 2, activation="relu"),
        layers.Flatten(), layers.Dense(4, activation="relu"), layers.Dense(1, activation="tanh")])
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return model


def train_baseline_cnn(*, X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=40, image_size=None, seed=42):
    """Train with validation only, restore best weights, and predict test once."""
    import tensorflow as tf
    tf.random.set_seed(seed)
    arrays = [np.asarray(x) for x in (X_train, X_val, X_test)]
    if image_size is None:
        image_size = int(round(arrays[0].reshape(arrays[0].shape[0], -1).shape[1] ** 0.5))
    Xtr, Xv, Xte = [x.reshape(-1, image_size, image_size, 1) for x in arrays]
    model = build_baseline_cnn((image_size, image_size, 1))
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=epochs, restore_best_weights=True)
    history = model.fit(Xtr, y_train, validation_data=(Xv, y_val), epochs=epochs,
                        batch_size=16, verbose=0, callbacks=[callback])
    losses = history.history["val_loss"]
    best_epoch = int(np.argmin(losses)) + 1
    guard = TestEvaluationGuard()
    raw = guard.evaluate(lambda: np.asarray(model.predict(Xte, verbose=0)).reshape(-1))
    return _result(y_test, raw, {
        "criterion": "validation_loss", "best_epoch": best_epoch,
        "best_value": float(losses[best_epoch - 1]), "hyperparameters": {},
        "n_validation_evaluations": len(losses),
    }, np.concatenate([w.reshape(-1) for w in model.get_weights()]), test_count=guard.count)
