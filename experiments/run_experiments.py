#!/usr/bin/env python3
"""
Multi-run experiment harness for the FQCNN reviewer response.

One harness produces the evidence for several reviewer suggestions at once, all on
*identical* data splits and seeds so every comparison is fair (#1, #4, #5, #10):

  - #1/#10 fair comparison vs classical baselines (logistic, MLP) on the same split
  - #2      harder datasets (MNIST hard digit pairs via --datasets)
  - #4      ablation studies (pooling / entanglement / kernel-rotation toggles)
  - #5      multiple seeds → mean ± std for every metric

Outputs:
  Results/experiments/<dataset>/<config>/seed_<s>.json   per-run metrics
  Results/experiments/<dataset>/<config>/aggregate.json  mean ± std
  Results/experiments/summary.csv                         one row per (dataset, config)

Usage examples:
  # fast smoke test (tiny, one pair, one ablation, 2 seeds)
  python -m experiments.run_experiments --quick

  # full study
  python -m experiments.run_experiments \
      --datasets 0,1 3,5 4,9 5,8 --seeds 0 1 2 3 4 --samples 400 --epochs 30
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import platform

import numpy as np
import pennylane as qml

from QCNN.utils import run_artifacts
from QCNN.utils import dataset_registry
from QCNN.utils import splits as split_service
from QCNN.utils.run_artifacts import TestEvaluationGuard
from QCNN.utils.seeding import seed_everything
from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN
from QCNN.training.Qtrainer import QuantumNativeTrainer
from QCNN.utils.dataset_loader import load_dataset
from QCNN.utils.metrics import (
    predict_raw_outputs,
    compute_classification_metrics,
    aggregate_metrics,
    save_metrics_json,
)
from baselines.classical_cnn import run_classical_baselines
from baselines.quantum_baselines import run_quantum_baselines

DEFAULT_MNIST_DIR = os.path.join("datasets", "MNIST")
EXP_ROOT = os.path.join("Results", "experiments")
FAILURE_MANIFEST = os.path.join(EXP_ROOT, "failures.json")
MANIFEST_ROOT = os.path.join("Results", "manifests")

# Each worker gets one CPU thread. Set in the parent before the pool is created
# so spawned children inherit it at interpreter start -- BLAS reads these at
# import time, so an initializer would run too late to have any effect.
_THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Ablation configurations. Each toggles ONE component off/variant relative to the
# proposed architecture so the contribution of each piece is measurable (#4).
# image_size/encoding are per-config because feature_map needs few qubits.
ABLATION_CONFIGS = {
    "proposed":        dict(image_size=28, encoding="amplitude"),  # su2 / full / unitary
    "pool_none":       dict(image_size=28, encoding="amplitude", pooling_mode="none"),
    "pool_measurement":dict(image_size=28, encoding="amplitude", pooling_mode="measurement"),
    "ent_one_diagonal":dict(image_size=28, encoding="amplitude", conv_entanglement="one_diagonal"),
    "ent_none":        dict(image_size=28, encoding="amplitude", conv_entanglement="none"),
    "kernel_ry":       dict(image_size=28, encoding="amplitude", kernel_rotations="ry"),
    # Encoding ablation uses a small image so feature_map stays simulable.
    "enc_feature_map": dict(image_size=4, encoding="feature_map"),
}

# E3 pooling arms (UPGRADE_PLAN.md 2.2, roadmap M2.4). These run at the HEADLINE
# geometry -- image_size=28 -> n=10 -- so every row of T5 describes the model the
# paper is about. The general amplitude arms above now share that geometry.
#
# 'coherent' is [ARCH -- opt-in] and ran only after the roadmap 18.2 sign-off
# (given 2026-07-25). It is an ablation arm; promotion to headline would need a
# second sign-off (A5).
E3_POOLING_ARMS = {
    "e3_pool_none":        dict(image_size=28, encoding="amplitude", pooling_mode="none"),
    "e3_pool_measurement": dict(image_size=28, encoding="amplitude", pooling_mode="measurement"),
    "e3_pool_unitary":     dict(image_size=28, encoding="amplitude", pooling_mode="unitary"),
    "e3_pool_coherent":    dict(image_size=28, encoding="amplitude", pooling_mode="coherent"),
    "e3_pool_su4":         dict(image_size=28, encoding="amplitude", pooling_mode="su4"),
}
ABLATION_CONFIGS.update(E3_POOLING_ARMS)

ABLATION_GEOMETRY_EXCEPTIONS = {
    "enc_feature_map": {
        "code": "encoding_requires_distinct_geometry",
        "factor": "encoding_type",
        "reference_image_size": 28,
        "reference_n_qubits": 10,
        "image_size": 4,
        "n_qubits": 16,
        "reason": "Feature-map simulation uses one qubit per pixel and is not feasible at headline geometry.",
        "comparable_as_one_factor": False,
    },
}

_ABLATION_FACTORS = {
    "pool_none": "pooling_mode",
    "pool_measurement": "pooling_mode",
    "ent_one_diagonal": "conv_entanglement",
    "ent_none": "conv_entanglement",
    "kernel_ry": "kernel_rotations",
    "enc_feature_map": "encoding_type",
}


def resolve_ablation(name: str) -> dict:
    """Resolve an ablation to executable geometry and comparison metadata."""
    cfg = build_config(ABLATION_CONFIGS[name], seed=0)
    resolved = {
        "name": name,
        "image_size": cfg.image_size,
        "n_qubits": cfg.n_qubits,
        "encoding_type": cfg.encoding_type,
        "pooling_mode": cfg.pooling_mode,
        "conv_entanglement": cfg.conv_entanglement,
        "kernel_rotations": cfg.kernel_rotations,
        "factor": _ABLATION_FACTORS.get(name),
    }
    if name in ABLATION_GEOMETRY_EXCEPTIONS:
        resolved["geometry_exception"] = ABLATION_GEOMETRY_EXCEPTIONS[name]
    return resolved


def build_config(overrides: dict, seed: int) -> QuantumNativeConfig:
    """Build a config from per-config overrides + seed."""
    image_size = overrides.get("image_size", 16)
    encoding = overrides.get("encoding", "amplitude")
    cfg = QuantumNativeConfig.from_image_size(image_size, encoding)
    cfg.seed = seed
    for k, v in overrides.items():
        if k in ("image_size", "encoding"):
            continue
        setattr(cfg, k, v)
    return cfg


def _config_metadata(cfg: QuantumNativeConfig, config_name: str) -> dict:
    metadata = {k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, type(None)))}
    resolved = resolve_ablation(config_name)
    metadata["ablation"] = resolved
    return metadata


def _total_samples_for_train_budget(train_sample_size: int) -> int:
    """Return the smallest pool whose frozen split contains the train budget."""
    target = int(train_sample_size)
    if target < 1:
        raise ValueError("train_sample_size must be positive")
    train_fraction = float(split_service.SPLIT_FRACTIONS[0])
    # ``make_split_manifest`` uses integer rounding for the partition sizes.
    # Account for that half-row margin, then verify both sides so this remains
    # correct if the frozen fraction is ever changed.
    total_needed = max(1, int(np.floor((target - 0.5) / train_fraction)))
    while int(round(train_fraction * total_needed)) < target:
        total_needed += 1
    while total_needed > 1 and int(round(train_fraction * (total_needed - 1))) >= target:
        total_needed -= 1
    return total_needed


def prepare_split(cfg: QuantumNativeConfig, classes, dataset_dir, train_sample_size,
                  dataset_key=None):
    """
    Load + preprocess the dataset and produce a deterministic train/test split.
    Mirrors main.py's pipeline so QCNN and baselines see the same representation.
    """
    seed_everything(cfg.seed)
    if dataset_key is None:
        X, y = load_dataset(
            source=dataset_dir,
            dataset_type="idx",
            n_qubits=cfg.n_qubits,
            image_size=cfg.image_size,
            normalization=cfg.preprocessing_mode,
            encoding_type=cfg.encoding_type,
            classes=classes,
        )
        sample_ids = list(range(len(y)))
        dataset_id = "idx_{}v{}".format(classes[0], classes[1])
    else:
        X, y, sample_ids = dataset_registry.load_binary_quantum(
            dataset_key,
            classes,
            data_root=dataset_dir,
            n_qubits=cfg.n_qubits,
            image_size=cfg.image_size,
            normalization=cfg.preprocessing_mode,
            encoding_type=cfg.encoding_type,
        )
        dataset_id = "{}_{}v{}".format(dataset_key, classes[0], classes[1])

    # Optional stratified subsample to keep quantum simulation tractable.  The
    # public ``--samples`` argument is a training-example budget, so choose the
    # smallest pre-split pool whose frozen 60% training partition can contain
    # that many rows.  The previous implementation divided by 0.70 (a legacy
    # assumption) and silently produced 343 training rows for ``--samples 400``.
    if train_sample_size is not None:
        total_needed = _total_samples_for_train_budget(train_sample_size)
        if total_needed < len(X):
            idx_pos = np.where(y == 1)[0]
            idx_neg = np.where(y == -1)[0]
            n_pos = min(len(idx_pos), total_needed // 2)
            n_neg = min(len(idx_neg), total_needed - n_pos)
            sel = np.concatenate([
                np.random.choice(idx_pos, n_pos, replace=False),
                np.random.choice(idx_neg, n_neg, replace=False),
            ])
            np.random.shuffle(sel)
            X, y = X[sel], y[sel]
            sample_ids = [sample_ids[int(index)] for index in sel]

    manifest = split_service.make_split_manifest(
        y,
        seed=cfg.seed,
        dataset_id="{}_n{}".format(dataset_id, len(y)),
        class_mapping=split_service.class_mapping_for(classes),
        sample_ids=sample_ids,
    )
    split_service.save_manifest(manifest, os.path.join(
        MANIFEST_ROOT, "{}_seed{}.json".format(manifest["dataset_id"], cfg.seed)))

    cfg.split_id = manifest["id"]
    X_train, y_train, X_val, y_val, X_test, y_test = split_service.apply_manifest(manifest, X, y)
    if train_sample_size is not None and train_sample_size < len(X_train):
        X_train, y_train = X_train[:train_sample_size], y_train[:train_sample_size]
    return (X_train, y_train, X_val, y_val, X_test, y_test), manifest


def run_qcnn(cfg: QuantumNativeConfig, split, use_bce: bool, log_path: str,
             directory: str, test_sample_ids, config_name: str) -> dict:
    """Train the QCNN on a prepared split and return its metric dict."""
    X_train, y_train, X_val, y_val, X_test, y_test = split
    model = PureQuantumNativeCNN(cfg)
    n_params = int(sum(np.prod(p.shape) for p in model.quantum_params.values()))
    trainer = QuantumNativeTrainer(learning_rate=cfg.learning_rate, use_bce=use_bce)

    run_artifacts.start_run(
        directory,
        config=_config_metadata(cfg, config_name),
        split_id=cfg.split_id,
        seed=cfg.seed,
        environment={"python": platform.python_version(), "pennylane": qml.version()},
    )

    # Trainer is very chatty; capture its output to a per-run log file.
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    guard = TestEvaluationGuard()
    try:
        with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
            trained = trainer.train_pure_quantum_cnn(
                model, X_train, y_train, X_val, y_val,
                log_filepath=log_path, summary_filepath=None,
                weights_path=run_artifacts.weights_path(directory),
            )
            raw = guard.evaluate(predict_raw_outputs, trained, X_test,
                                 already_preprocessed=False)
    except Exception as exc:
        run_artifacts.fail_run(directory, error=repr(exc))
        raise

    run_artifacts.save_predictions(directory, test_sample_ids, y_test, raw)
    metrics = compute_classification_metrics(y_test, raw)
    metrics["n_params"] = n_params
    run_artifacts.complete_run(directory, metrics={
        k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics


def run_single(config_name: str, classes, seed: int, dataset_dir: str,
               train_sample_size: int, epochs: int, use_bce: bool,
               out_dir: str, with_baselines: bool, dataset_key=None,
               evidence_role="scientific", classical_baselines=None,
               quantum_baselines=None) -> dict:
    """Run one (config, dataset, seed). Returns the QCNN metric dict."""
    overrides = ABLATION_CONFIGS[config_name]
    cfg = build_config(overrides, seed)
    if epochs is not None:
        cfg.n_epochs = epochs
    cfg.evidence_role = evidence_role

    if dataset_key is None:
        split, manifest = prepare_split(cfg, classes, dataset_dir, train_sample_size)
    else:
        split, manifest = prepare_split(
            cfg, classes, dataset_dir, train_sample_size, dataset_key=dataset_key)

    run_dir = os.path.join(out_dir, config_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"seed_{seed}.log")

    # Clean-protocol artifacts live under Results/runs/, kept separate from the
    # historical Results/experiments/ tree produced under the leaked protocol.
    task_name = _fmt_task(dataset_key, classes, evidence_role)
    directory = run_artifacts.run_dir(task_name, config_name, seed)
    test_ids = [manifest["sample_ids"][i] for i in manifest["test_idx"]]

    metrics = run_qcnn(cfg, split, use_bce, log_path, directory, test_ids, config_name)
    save_metrics_json(metrics, os.path.join(run_dir, f"seed_{seed}.json"))

    # Baselines share the EXACT split → fair comparison. Only needed once per
    # (dataset, seed); the 'proposed' config is the natural place to run them.
    if with_baselines and config_name == "proposed":
        X_train, y_train, X_val, y_val, X_test, y_test = split
        baseline_split = dict(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test)
        base = run_classical_baselines(
            **baseline_split, seed=seed, target_params=metrics.get("n_params"),
            baselines=classical_baselines)
        # Task 5 owns arbitrary-n baseline scheduling; this path remains on the
        # explicitly compatible power-of-two geometry used by "proposed".
        base.update(run_quantum_baselines(
            **baseline_split, seed=seed, n_qubits=cfg.n_qubits,
            n_epochs=cfg.n_epochs, learning_rate=cfg.learning_rate, use_bce=use_bce,
            baselines=quantum_baselines))
        for name, result in base.items():
            bdir = os.path.join(out_dir, f"baseline_{name}")
            os.makedirs(bdir, exist_ok=True)
            save_metrics_json(result["metrics"], os.path.join(bdir, f"seed_{seed}.json"))
            artifact_dir = run_artifacts.run_dir(
                task_name, f"baseline_{name}", seed)
            run_artifacts.save_baseline_result(
                directory=artifact_dir, result=result, split_id=manifest["id"],
                sample_ids=test_ids, y_test=y_test, seed=seed,
                environment={"python": platform.python_version(),
                             "pennylane": qml.version()})
    return metrics


def _fmt_pair(pair) -> str:
    return f"{pair[0]}v{pair[1]}"


def _fmt_task(dataset_key, pair, evidence_role="scientific") -> str:
    """Filesystem-safe task identity; smoke cells cannot mix with evidence."""
    name = _fmt_pair(pair) if dataset_key is None else f"{dataset_key}_{_fmt_pair(pair)}"
    return f"smoke__{name}" if evidence_role == "smoke" else name


def _parse_task(value: str):
    """Parse ``dataset:low,high`` into a registered binary task."""
    try:
        dataset_key, pair_text = value.split(":", 1)
        pair = tuple(int(item) for item in pair_text.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "task must use dataset:low,high (for example fashion_mnist:0,6)"
        ) from exc
    if dataset_key not in dataset_registry.DATASETS:
        raise argparse.ArgumentTypeError(
            "unknown dataset {}; choose from {}".format(
                dataset_key, ", ".join(sorted(dataset_registry.DATASETS))))
    if len(pair) != 2 or pair[0] == pair[1] or any(value < 0 or value > 9 for value in pair):
        raise argparse.ArgumentTypeError("task classes must be two distinct digits from 0 to 9")
    return dataset_key, pair


# ---------------------------------------------------------------------------
# Cells: one (dataset, config, seed) unit of work. Independent by construction,
# which is what makes both the parallelism and the resume safe (M1.3).
# ---------------------------------------------------------------------------
def _apply_output_roots(roots: dict) -> dict:
    global EXP_ROOT, FAILURE_MANIFEST, MANIFEST_ROOT
    required = {"experiments", "runs", "manifests", "failures"}
    if not isinstance(roots, dict) or set(roots) != required:
        raise ValueError("worker output roots have an invalid shape")
    if any(not isinstance(value, str) or not value for value in roots.values()):
        raise ValueError("worker output roots must be non-empty strings")

    EXP_ROOT = roots["experiments"]
    FAILURE_MANIFEST = roots["failures"]
    MANIFEST_ROOT = roots["manifests"]
    run_artifacts.RUN_ROOT = roots["runs"]
    return {
        "experiments": EXP_ROOT,
        "runs": run_artifacts.RUN_ROOT,
        "manifests": MANIFEST_ROOT,
        "failures": FAILURE_MANIFEST,
    }


def _metrics_path(pair, config_name: str, seed: int, dataset_key=None,
                  evidence_role="scientific") -> str:
    return os.path.join(
        EXP_ROOT, _fmt_task(dataset_key, pair, evidence_role), config_name,
        f"seed_{seed}.json")


def _expected_config(config_name: str, seed: int, epochs,
                     evidence_role="scientific") -> dict:
    """The config dict run_qcnn would record, computed without loading data."""
    cfg = build_config(ABLATION_CONFIGS[config_name], seed)
    if epochs is not None:
        cfg.n_epochs = epochs
    cfg.evidence_role = evidence_role
    return _config_metadata(cfg, config_name)


def _is_reusable_cell(pair, config_name: str, seed: int, epochs,
                      dataset_key=None, evidence_role="scientific",
                      required_baselines=()) -> bool:
    """Skip only a cell this invocation would otherwise reproduce exactly."""
    if not os.path.exists(_metrics_path(
            pair, config_name, seed, dataset_key, evidence_role)):
        return False
    directory = run_artifacts.run_dir(
        _fmt_task(dataset_key, pair, evidence_role), config_name, seed, create=False)
    if not run_artifacts.is_reusable(directory, config=_expected_config(
            config_name, seed, epochs, evidence_role), seed=seed):
        return False
    task_name = _fmt_task(dataset_key, pair, evidence_role)
    return all(run_artifacts.is_reusable(
        run_artifacts.run_dir(
            task_name, f"baseline_{name}", seed, create=False),
        config={"artifact_schema_version": 1}, seed=seed)
        for name in required_baselines)


def _execute_cell(payload):
    """Run one cell in this process. Returns (pair, config, seed, error|None).

    Metrics are not returned: every cell's numbers are read back from its saved
    JSON so a fresh run and a resumed run aggregate from byte-identical input.
    """
    if len(payload) == 9:
        pair, config_name, seed, dataset_dir, samples, epochs, use_bce, with_baselines, roots = payload
        dataset_key, evidence_role = None, "scientific"
        classical_baselines, quantum_baselines = None, None
    elif len(payload) == 11:
        (pair, config_name, seed, dataset_dir, samples, epochs, use_bce,
         with_baselines, roots, dataset_key, evidence_role) = payload
        classical_baselines, quantum_baselines = None, None
    else:
        (pair, config_name, seed, dataset_dir, samples, epochs, use_bce,
         with_baselines, roots, dataset_key, evidence_role, classical_baselines,
         quantum_baselines) = payload
    _apply_output_roots(roots)
    task_name = _fmt_task(dataset_key, pair, evidence_role)
    try:
        run_single(config_name, pair, seed, dataset_dir, samples, epochs,
                   use_bce=use_bce, out_dir=os.path.join(EXP_ROOT, task_name),
                   with_baselines=with_baselines, dataset_key=dataset_key,
                   evidence_role=evidence_role,
                   classical_baselines=classical_baselines,
                   quantum_baselines=quantum_baselines)
        return dataset_key, pair, config_name, seed, evidence_role, None
    except Exception:
        return dataset_key, pair, config_name, seed, evidence_role, traceback.format_exc()


def _write_failure_manifest(failures) -> None:
    """Always written, empty list included, so a stale manifest cannot mislead."""
    os.makedirs(EXP_ROOT, exist_ok=True)
    with open(FAILURE_MANIFEST, "w") as fh:
        json.dump({"n_failed": len(failures), "failures": failures}, fh, indent=2)


def main(output_roots: dict = None):
    using_default_roots = output_roots is None
    if output_roots is None:
        output_roots = {
            "experiments": EXP_ROOT,
            "runs": run_artifacts.RUN_ROOT,
            "manifests": MANIFEST_ROOT,
            "failures": FAILURE_MANIFEST,
        }
    ap = argparse.ArgumentParser(description="FQCNN ablation / multi-seed study")
    ap.add_argument("--datasets", nargs="+", default=["0,1", "3,5", "4,9", "5,8"],
                    help="Class pairs as 'a,b' (default: hard MNIST pairs)")
    ap.add_argument("--task", action="append", type=_parse_task,
                    help="Registered task as dataset:low,high; repeat for multiple tasks")
    ap.add_argument("--configs", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                    help="Ablation configs to run (default: all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--samples", type=int, default=400, help="Train sample size")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--mnist-dir", default=DEFAULT_MNIST_DIR)
    ap.add_argument("--data-root", default="datasets",
                    help="Root containing registered dataset directories")
    ap.add_argument("--use-mse", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--classical-baselines", nargs="+", choices=("logistic", "mlp"),
                    default=["logistic", "mlp"])
    ap.add_argument("--quantum-baselines", nargs="+", choices=("cong", "hur", "ttn"),
                    default=["ttn"])
    ap.add_argument("--quick", action="store_true",
                    help="Tiny smoke run: 1 pair, proposed+pool_none, 2 seeds, 60 samples")
    ap.add_argument("--smoke", action="store_true",
                    help="Three-family validation only: proposed, seed 0, 60 samples, 2 epochs")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes over (dataset, config, seed) cells. "
                         "0 = physical cores - 2. Each worker is pinned to one CPU thread.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run every cell, including complete ones (default: resume)")
    args = ap.parse_args()

    if args.quick and args.smoke:
        ap.error("--quick and --smoke are mutually exclusive")

    if args.quick:
        args.datasets = ["0,1"]
        args.configs = ["proposed", "pool_none"]
        args.seeds = [0, 1]
        args.samples = 60
        args.epochs = 2

    if args.smoke:
        if args.task is None:
            args.task = [
                ("mnist", (0, 1)),
                ("fashion_mnist", (0, 6)),
                ("kmnist", (2, 3)),
            ]
        args.configs = ["proposed"]
        args.seeds = [0]
        args.samples = 60
        args.epochs = 2
        args.no_baselines = True

        if using_default_roots:
            smoke_root = os.path.join("Results", "smoke", "q1_datasets")
            output_roots = {
                "experiments": os.path.join(smoke_root, "experiments"),
                "runs": os.path.join(smoke_root, "runs"),
                "manifests": os.path.join(smoke_root, "manifests"),
                "failures": os.path.join(smoke_root, "experiments", "failures.json"),
            }

    if args.task is not None and not args.smoke and using_default_roots:
        comparison_root = os.path.join("Results", "q1_comparison")
        output_roots = {
            "experiments": os.path.join(comparison_root, "experiments"),
            "runs": os.path.join(comparison_root, "runs"),
            "manifests": os.path.join(comparison_root, "manifests"),
            "failures": os.path.join(
                comparison_root, "experiments", "failures.json"),
        }

    roots = _apply_output_roots(output_roots)

    unknown = [c for c in args.configs if c not in ABLATION_CONFIGS]
    if unknown:
        ap.error("unknown config(s) {}; choose from {}".format(
            ", ".join(unknown), ", ".join(sorted(ABLATION_CONFIGS))))

    evidence_role = "smoke" if args.smoke else "scientific"
    if args.task is not None:
        tasks = [
            {
                "dataset_key": dataset_key,
                "pair": pair,
                "dataset_dir": args.data_root,
                "evidence_role": evidence_role,
            }
            for dataset_key, pair in args.task
        ]
    else:
        tasks = [
            {
                "dataset_key": None,
                "pair": tuple(int(c) for c in value.split(",")),
                "dataset_dir": args.mnist_dir,
                "evidence_role": evidence_role,
            }
            for value in args.datasets
        ]
    os.makedirs(EXP_ROOT, exist_ok=True)

    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 3) - 2)

    # Schedule: enumerate every cell, then split into reusable and pending.
    pending, n_reused = [], 0
    for task in tasks:
        pair = task["pair"]
        dataset_key = task["dataset_key"]
        role = task["evidence_role"]
        task_name = _fmt_task(dataset_key, pair, role)
        required_baselines = (
            tuple(args.classical_baselines) + tuple(args.quantum_baselines)
            if not args.no_baselines else ())
        for config_name in args.configs:
            for seed in args.seeds:
                if not args.force and _is_reusable_cell(
                        pair, config_name, seed, args.epochs, dataset_key, role,
                        required_baselines if config_name == "proposed" else ()):
                    n_reused += 1
                    print(f"  [{task_name}] {config_name} seed={seed} "
                          f"-> reusing complete run", flush=True)
                    continue
                pending.append((pair, config_name, seed, task["dataset_dir"], args.samples,
                                args.epochs, not args.use_mse, not args.no_baselines,
                                roots, dataset_key, role, args.classical_baselines,
                                args.quantum_baselines))

    print(f"\n{len(pending)} cells to run, {n_reused} reused, {jobs} worker(s)\n")

    failures = []

    def record(dataset_key, pair, config_name, seed, role, error):
        task_name = _fmt_task(dataset_key, pair, role)
        label = f"[{task_name}] {config_name} seed={seed}"
        if error is None:
            print(f"  {label} -> done", flush=True)
            return
        print(f"  {label} -> FAILED\n{error}", flush=True)
        failures.append({"dataset": task_name, "config": config_name,
                         "seed": int(seed), "error": error})

    if jobs > 1 and pending:
        for var in _THREAD_LIMIT_VARS:
            os.environ[var] = "1"
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_execute_cell, payload) for payload in pending]
            for future in as_completed(futures):
                record(*future.result())
    else:
        for payload in pending:
            record(*_execute_cell(payload))

    # Aggregate in schedule order, from disk, so a resumed sweep and a fresh one
    # produce the same summary regardless of completion order (M1.3 exit check).
    summary_rows = []
    for task in tasks:
        pair = task["pair"]
        dataset_key = task["dataset_key"]
        role = task["evidence_role"]
        ds_name = _fmt_task(dataset_key, pair, role)
        for config_name in args.configs:
            per_seed = []
            for seed in args.seeds:
                path = _metrics_path(pair, config_name, seed, dataset_key, role)
                if os.path.exists(path):
                    with open(path) as fh:
                        per_seed.append(json.load(fh))
            if per_seed:
                agg = aggregate_metrics(per_seed)
                cfg_dir = os.path.join(EXP_ROOT, ds_name, config_name)
                os.makedirs(cfg_dir, exist_ok=True)
                with open(os.path.join(cfg_dir, "aggregate.json"), "w") as f:
                    json.dump(agg, f, indent=2)
                summary_rows.append((ds_name, config_name, agg))

    # Aggregate baselines (collected under baseline_* dirs) into the summary too.
    _append_baseline_rows([_fmt_task(
        task["dataset_key"], task["pair"], task["evidence_role"])
        for task in tasks], args.seeds, summary_rows)

    _write_summary_csv(summary_rows, os.path.join(EXP_ROOT, "summary.csv"))
    print(f"\nWrote summary to {os.path.join(EXP_ROOT, 'summary.csv')}")

    _write_failure_manifest(failures)
    if failures:
        print(f"\n{len(failures)} cell(s) failed; see {FAILURE_MANIFEST}", flush=True)
        sys.exit(1)


def _append_baseline_rows(tasks, seeds, summary_rows):
    """Aggregate the per-seed baseline JSONs that run_single saved."""
    for task in tasks:
        ds_name = task if isinstance(task, str) else _fmt_pair(task)
        ds_dir = os.path.join(EXP_ROOT, ds_name)
        if not os.path.isdir(ds_dir):
            continue
        for entry in sorted(os.listdir(ds_dir)):
            if not entry.startswith("baseline_"):
                continue
            bdir = os.path.join(ds_dir, entry)
            dicts = []
            for seed in seeds:
                p = os.path.join(bdir, f"seed_{seed}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        dicts.append(json.load(f))
            if dicts:
                summary_rows.append((ds_name, entry, aggregate_metrics(dicts)))


_SUMMARY_METRICS = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def _write_summary_csv(rows, path):
    header = ["dataset", "config", "n_runs"]
    for m in _SUMMARY_METRICS:
        header += [f"{m}_mean", f"{m}_std"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for ds_name, cfg_name, agg in rows:
            row = [ds_name, cfg_name, agg.get("n_runs", 0)]
            for m in _SUMMARY_METRICS:
                stat = agg.get(m, {})
                row += [f"{stat.get('mean', float('nan')):.4f}",
                        f"{stat.get('std', float('nan')):.4f}"]
            w.writerow(row)


if __name__ == "__main__":
    main()
