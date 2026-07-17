"""
Reusable classification-metric utilities for the FQCNN pipeline.

Separates two concerns that were previously inlined in ``main.py``:

1. ``predict_raw_outputs`` — run the quantum circuit and collect the continuous
   PauliZ expectation values (in roughly [-1, 1]) for a set of samples.
2. ``compute_classification_metrics`` — turn (y_true, raw_outputs) into the full
   suite of metrics the reviewers asked for: accuracy, precision, recall, F1,
   ROC-AUC, PR-AUC and the confusion-matrix counts.

Keeping metric computation free of any circuit/PennyLane dependency makes it
trivial to unit-test and lets both ``main.py`` and the multi-run experiment
runner share one implementation (suggestions #3, #5, #10).
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def predict_raw_outputs(model, X_eval: np.ndarray, already_preprocessed: bool = False) -> np.ndarray:
    """
    Collect continuous circuit outputs (⟨Z⟩ expectation values) for each sample.

    Args:
        model: a ``PureQuantumNativeCNN`` instance.
        X_eval: evaluation inputs. Either raw inputs (``already_preprocessed=False``)
            or inputs already mapped to the model's encoder representation
            (``already_preprocessed=True``) — e.g. cached quanvolutional features,
            which must NOT be re-preprocessed.
        already_preprocessed: skip ``model._preprocess_input`` when True.

    Returns:
        1D array of continuous outputs, one per sample.
    """
    flat_params = model._flatten_params(model.quantum_params)

    if already_preprocessed:
        X_processed = np.asarray(X_eval)
    else:
        X_processed = model._preprocess_input(np.asarray(X_eval))

    outputs = np.array([
        float(np.squeeze(model.quantum_circuit(X_processed[i], flat_params)))
        for i in range(len(X_processed))
    ])
    return outputs


def compute_classification_metrics(y_true: np.ndarray, raw_outputs: np.ndarray,
                                   threshold: float = 0.0) -> dict:
    """
    Compute the full metric suite from continuous circuit outputs.

    Args:
        y_true: ground-truth labels in {-1, +1}.
        raw_outputs: continuous circuit outputs (≈ [-1, 1]).
        threshold: decision boundary on the raw output (default 0.0).

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc, pr_auc,
        confusion-matrix counts (tp/tn/fp/fn), prediction bias/variance,
        and the continuous ``prob_scores`` (min-max normalised to [0, 1]).
    """
    y_true = np.asarray(y_true)
    raw_outputs = np.asarray(raw_outputs, dtype=float)

    predictions = np.where(raw_outputs > threshold, 1, -1)

    # Min-max normalise continuous outputs to [0, 1] for threshold-free curves.
    span = raw_outputs.max() - raw_outputs.min()
    prob_scores = (raw_outputs - raw_outputs.min()) / (span + 1e-8)

    # Map {-1,+1} -> {0,1} for sklearn.
    y_bin = np.where(y_true == 1, 1, 0)
    pred_bin = np.where(predictions == 1, 1, 0)

    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    tn = int(np.sum((predictions == -1) & (y_true == -1)))
    fp = int(np.sum((predictions == 1) & (y_true == -1)))
    fn = int(np.sum((predictions == -1) & (y_true == 1)))

    accuracy = float(np.mean(predictions == y_true))
    precision = float(precision_score(y_bin, pred_bin, zero_division=0))
    recall = float(recall_score(y_bin, pred_bin, zero_division=0))
    f1 = float(f1_score(y_bin, pred_bin, zero_division=0))

    # ROC-AUC / PR-AUC need both classes present; guard degenerate test sets.
    roc_auc = pr_auc = float("nan")
    if len(np.unique(y_bin)) == 2:
        fpr, tpr, _ = roc_curve(y_bin, prob_scores)
        roc_auc = float(auc(fpr, tpr))
        prec_vals, rec_vals, _ = precision_recall_curve(y_bin, prob_scores)
        pr_auc = float(auc(rec_vals, prec_vals))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "prediction_bias": float(np.mean(prob_scores) - np.mean(y_bin)),
        "prediction_variance": float(np.var(prob_scores)),
        "prob_scores": prob_scores,
        "predictions": predictions,
    }


# Keys worth aggregating across seeds (scalars only — drop arrays/counts handled separately).
_SCALAR_METRIC_KEYS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "prediction_bias", "prediction_variance",
)


def aggregate_metrics(metric_dicts: list[dict]) -> dict:
    """
    Aggregate a list of per-run metric dicts into mean ± std per scalar metric.

    Args:
        metric_dicts: list of dicts as returned by ``compute_classification_metrics``.

    Returns:
        dict mapping ``<metric>`` -> {"mean", "std", "n", "values", "ci_low", "ci_high"}.
        ``std`` is the sample standard deviation (ddof=1). ``ci_low``/``ci_high`` are the
        95% confidence interval for the mean via the Student-t interval
        (mean ± t_{0.975, n-1} * s/sqrt(n)). NaNs (e.g. degenerate ROC-AUC) are ignored.
    """
    from scipy import stats  # scipy is already a project dependency

    out = {"n_runs": len(metric_dicts)}
    for key in _SCALAR_METRIC_KEYS:
        vals = np.array([d[key] for d in metric_dicts if key in d], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            mean = float(np.mean(finite))
            n = int(finite.size)
            # sample std (ddof=1) for an unbiased spread; 0 when n==1
            std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
            if n > 1 and std > 0:
                half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
            else:
                half = 0.0
            out[key] = {
                "mean": mean,
                "std": std,
                "n": n,
                "values": finite.tolist(),
                "ci_low": mean - half,
                "ci_high": mean + half,
            }
        else:
            out[key] = {"mean": float("nan"), "std": float("nan"), "n": 0,
                        "values": [], "ci_low": float("nan"), "ci_high": float("nan")}
    return out


def _json_safe(metrics: dict) -> dict:
    """Drop / convert non-JSON-serialisable entries (numpy arrays) from a metrics dict."""
    safe = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            continue  # prob_scores / predictions — too large and not needed in JSON
        elif isinstance(v, (np.integer,)):
            safe[k] = int(v)
        elif isinstance(v, (np.floating,)):
            safe[k] = float(v)
        else:
            safe[k] = v
    return safe


def save_metrics_json(metrics: dict, path: str) -> None:
    """Write a metrics dict to JSON, stripping arrays and coercing numpy scalars."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
