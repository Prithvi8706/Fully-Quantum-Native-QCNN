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
import io
import json
import os
import random
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import platform

import numpy as np
import pennylane as qml

from QCNN.utils import run_artifacts
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
    "proposed":        dict(image_size=16, encoding="amplitude"),  # su2 / full / unitary
    "pool_none":       dict(image_size=16, encoding="amplitude", pooling_mode="none"),
    "pool_measurement":dict(image_size=16, encoding="amplitude", pooling_mode="measurement"),
    "ent_one_diagonal":dict(image_size=16, encoding="amplitude", conv_entanglement="one_diagonal"),
    "ent_none":        dict(image_size=16, encoding="amplitude", conv_entanglement="none"),
    "kernel_ry":       dict(image_size=16, encoding="amplitude", kernel_rotations="ry"),
    # Encoding ablation uses a small image so feature_map stays simulable.
    "enc_feature_map": dict(image_size=4, encoding="feature_map"),
}

# E3 pooling arms (UPGRADE_PLAN.md 2.2, roadmap M2.4). These run at the HEADLINE
# geometry -- image_size=28 -> n=10 -- so every row of T5 describes the model the
# paper is about. The entries above use image_size=16 -> n=8 and are blocker B3;
# they are left alone here because widening them is M6's call, not E3's.
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


def prepare_split(cfg: QuantumNativeConfig, classes, dataset_dir, train_sample_size):
    """
    Load + preprocess the dataset and produce a deterministic train/test split.
    Mirrors main.py's pipeline so QCNN and baselines see the same representation.
    """
    seed_everything(cfg.seed)
    X, y = load_dataset(
        source=dataset_dir,
        dataset_type="idx",
        n_qubits=cfg.n_qubits,
        image_size=cfg.image_size,
        normalization=cfg.preprocessing_mode,
        encoding_type=cfg.encoding_type,
        classes=classes,
    )

    # Optional stratified subsample to keep quantum simulation tractable.
    if train_sample_size is not None:
        total_needed = int(train_sample_size / 0.7)
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

    manifest = split_service.make_split_manifest(
        y,
        seed=cfg.seed,
        dataset_id="idx_{}v{}_n{}".format(classes[0], classes[1], len(y)),
        class_mapping={str(classes[0]): 1, str(classes[1]): -1},
    )
    split_service.save_manifest(manifest, os.path.join(
        MANIFEST_ROOT, "{}_seed{}.json".format(manifest["dataset_id"], cfg.seed)))

    cfg.split_id = manifest["id"]
    X_train, y_train, X_val, y_val, X_test, y_test = split_service.apply_manifest(manifest, X, y)
    if train_sample_size is not None and train_sample_size < len(X_train):
        X_train, y_train = X_train[:train_sample_size], y_train[:train_sample_size]
    return (X_train, y_train, X_val, y_val, X_test, y_test), manifest


def run_qcnn(cfg: QuantumNativeConfig, split, use_bce: bool, log_path: str,
             directory: str, test_sample_ids) -> dict:
    """Train the QCNN on a prepared split and return its metric dict."""
    X_train, y_train, X_val, y_val, X_test, y_test = split
    model = PureQuantumNativeCNN(cfg)
    n_params = int(sum(np.prod(p.shape) for p in model.quantum_params.values()))
    trainer = QuantumNativeTrainer(learning_rate=cfg.learning_rate, use_bce=use_bce)

    run_artifacts.start_run(
        directory,
        config={k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, type(None)))},
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
               out_dir: str, with_baselines: bool) -> dict:
    """Run one (config, dataset, seed). Returns the QCNN metric dict."""
    overrides = ABLATION_CONFIGS[config_name]
    cfg = build_config(overrides, seed)
    if epochs is not None:
        cfg.n_epochs = epochs

    split, manifest = prepare_split(cfg, classes, dataset_dir, train_sample_size)

    run_dir = os.path.join(out_dir, config_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"seed_{seed}.log")

    # Clean-protocol artifacts live under Results/runs/, kept separate from the
    # historical Results/experiments/ tree produced under the leaked protocol.
    directory = run_artifacts.run_dir(_fmt_pair(classes), config_name, seed)
    test_ids = [manifest["sample_ids"][i] for i in manifest["test_idx"]]

    metrics = run_qcnn(cfg, split, use_bce, log_path, directory, test_ids)
    save_metrics_json(metrics, os.path.join(run_dir, f"seed_{seed}.json"))

    # Baselines share the EXACT split → fair comparison. Only needed once per
    # (dataset, seed); the 'proposed' config is the natural place to run them.
    if with_baselines and config_name == "proposed":
        X_train, y_train, X_val, y_val, X_test, y_test = split
        # NOTE (Phase 5 / M5.1): these baselines still select on the data passed
        # as their test set. They must be given X_val before any baseline number
        # enters the manuscript.
        base = run_classical_baselines(X_train, y_train, X_test, y_test, seed=seed,
                                       target_params=metrics.get("n_params"))
        # Quantum-architecture baselines (#1): published QCNN/TTN models on the
        # IDENTICAL split / qubit count / optimiser budget as the proposed model.
        base.update(run_quantum_baselines(
            X_train, y_train, X_test, y_test, seed=seed, n_qubits=cfg.n_qubits,
            n_epochs=cfg.n_epochs, learning_rate=cfg.learning_rate, use_bce=use_bce))
        for name, bm in base.items():
            bdir = os.path.join(out_dir, f"baseline_{name}")
            os.makedirs(bdir, exist_ok=True)
            save_metrics_json(bm, os.path.join(bdir, f"seed_{seed}.json"))
    return metrics


def _fmt_pair(pair) -> str:
    return f"{pair[0]}v{pair[1]}"


# ---------------------------------------------------------------------------
# Cells: one (dataset, config, seed) unit of work. Independent by construction,
# which is what makes both the parallelism and the resume safe (M1.3).
# ---------------------------------------------------------------------------
def _metrics_path(pair, config_name: str, seed: int) -> str:
    return os.path.join(EXP_ROOT, _fmt_pair(pair), config_name, f"seed_{seed}.json")


def _expected_config(config_name: str, seed: int, epochs) -> dict:
    """The config dict run_qcnn would record, computed without loading data."""
    cfg = build_config(ABLATION_CONFIGS[config_name], seed)
    if epochs is not None:
        cfg.n_epochs = epochs
    return {k: v for k, v in vars(cfg).items()
            if isinstance(v, (int, float, str, bool, type(None)))}


def _is_reusable_cell(pair, config_name: str, seed: int, epochs) -> bool:
    """Skip only a cell this invocation would otherwise reproduce exactly."""
    if not os.path.exists(_metrics_path(pair, config_name, seed)):
        return False
    directory = run_artifacts.run_dir(
        _fmt_pair(pair), config_name, seed, create=False)
    return run_artifacts.is_reusable(
        directory, config=_expected_config(config_name, seed, epochs), seed=seed)


def _execute_cell(payload):
    """Run one cell in this process. Returns (pair, config, seed, error|None).

    Metrics are not returned: every cell's numbers are read back from its saved
    JSON so a fresh run and a resumed run aggregate from byte-identical input.
    """
    pair, config_name, seed, dataset_dir, samples, epochs, use_bce, with_baselines = payload
    try:
        run_single(config_name, pair, seed, dataset_dir, samples, epochs,
                   use_bce=use_bce, out_dir=os.path.join(EXP_ROOT, _fmt_pair(pair)),
                   with_baselines=with_baselines)
        return pair, config_name, seed, None
    except Exception:
        return pair, config_name, seed, traceback.format_exc()


def _write_failure_manifest(failures) -> None:
    """Always written, empty list included, so a stale manifest cannot mislead."""
    os.makedirs(EXP_ROOT, exist_ok=True)
    with open(FAILURE_MANIFEST, "w") as fh:
        json.dump({"n_failed": len(failures), "failures": failures}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description="FQCNN ablation / multi-seed study")
    ap.add_argument("--datasets", nargs="+", default=["0,1", "3,5", "4,9", "5,8"],
                    help="Class pairs as 'a,b' (default: hard MNIST pairs)")
    ap.add_argument("--configs", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                    help="Ablation configs to run (default: all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--samples", type=int, default=400, help="Train sample size")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--mnist-dir", default=DEFAULT_MNIST_DIR)
    ap.add_argument("--use-mse", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="Tiny smoke run: 1 pair, proposed+pool_none, 2 seeds, 60 samples")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes over (dataset, config, seed) cells. "
                         "0 = physical cores - 2. Each worker is pinned to one CPU thread.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run every cell, including complete ones (default: resume)")
    args = ap.parse_args()

    if args.quick:
        args.datasets = ["0,1"]
        args.configs = ["proposed", "pool_none"]
        args.seeds = [0, 1]
        args.samples = 60
        args.epochs = 2

    unknown = [c for c in args.configs if c not in ABLATION_CONFIGS]
    if unknown:
        ap.error("unknown config(s) {}; choose from {}".format(
            ", ".join(unknown), ", ".join(sorted(ABLATION_CONFIGS))))

    pairs = [tuple(int(c) for c in d.split(",")) for d in args.datasets]
    os.makedirs(EXP_ROOT, exist_ok=True)

    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 3) - 2)

    # Schedule: enumerate every cell, then split into reusable and pending.
    pending, n_reused = [], 0
    for pair in pairs:
        for config_name in args.configs:
            for seed in args.seeds:
                if not args.force and _is_reusable_cell(pair, config_name, seed, args.epochs):
                    n_reused += 1
                    print(f"  [{_fmt_pair(pair)}] {config_name} seed={seed} "
                          f"-> reusing complete run", flush=True)
                    continue
                pending.append((pair, config_name, seed, args.mnist_dir, args.samples,
                                args.epochs, not args.use_mse, not args.no_baselines))

    print(f"\n{len(pending)} cells to run, {n_reused} reused, {jobs} worker(s)\n")

    failures = []

    def record(pair, config_name, seed, error):
        label = f"[{_fmt_pair(pair)}] {config_name} seed={seed}"
        if error is None:
            print(f"  {label} -> done", flush=True)
            return
        print(f"  {label} -> FAILED\n{error}", flush=True)
        failures.append({"dataset": _fmt_pair(pair), "config": config_name,
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
    for pair in pairs:
        ds_name = _fmt_pair(pair)
        for config_name in args.configs:
            per_seed = []
            for seed in args.seeds:
                path = _metrics_path(pair, config_name, seed)
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
    _append_baseline_rows(pairs, args.seeds, summary_rows)

    _write_summary_csv(summary_rows, os.path.join(EXP_ROOT, "summary.csv"))
    print(f"\nWrote summary to {os.path.join(EXP_ROOT, 'summary.csv')}")

    _write_failure_manifest(failures)
    if failures:
        print(f"\n{len(failures)} cell(s) failed; see {FAILURE_MANIFEST}", flush=True)
        sys.exit(1)


def _append_baseline_rows(pairs, seeds, summary_rows):
    """Aggregate the per-seed baseline JSONs that run_single saved."""
    for pair in pairs:
        ds_name = _fmt_pair(pair)
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


_SUMMARY_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


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
