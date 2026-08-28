"""Bounded cross-domain pooling-ablation transfer for the Q1 study.

The existing five pooling arms were evaluated on MNIST before the fast-track
dataset registry was introduced.  This module provides the deliberately small
Fashion-MNIST transfer lane: one predeclared binary task (classes 0 and 6),
three seeds, and the same five arms.  It is an orchestration and validation
layer around :mod:`experiments.run_experiments`; it does not define a second
pooling implementation.

The scientific aggregate is intentionally fail-closed.  A cell is accepted
only when it is a complete run with the scientific evidence role, a canonical
manifest, and source-stable test identities.  All five arms for a seed must
then have the same split, ordered test IDs, and labels before any paired
statistics are calculated.  Smoke artifacts live in a different namespace and
are never eligible for this evidence file.

Typical use::

    python -m experiments.q1_pooling_transfer plan
    python -m experiments.q1_pooling_transfer run --jobs 1
    python -m experiments.q1_pooling_transfer aggregate

The ``run`` command is bounded to 15 cells (five arms x three seeds), 400
training examples, and 30 epochs.  It can be resumed safely by re-running the
same command; the underlying run harness skips only reusable cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from QCNN.utils import dataset_registry, run_artifacts
from QCNN.utils import splits as split_service
from QCNN.utils.metrics import compute_classification_metrics
from experiments import statistics as qstats


SCHEMA = {"name": "fqcnn_q1_pooling_transfer", "version": 1}
DATASET = "fashion_mnist"
CLASSES = (0, 6)
TASK_NAME = "fashion_mnist_0v6"
SEEDS = (0, 1, 2)
REFERENCE_ARM = "e3_pool_unitary"
POOLING_ARMS = (
    "e3_pool_none",
    "e3_pool_measurement",
    "e3_pool_unitary",
    "e3_pool_coherent",
    "e3_pool_su4",
)
POOLING_MODES = {
    "e3_pool_none": "none",
    "e3_pool_measurement": "measurement",
    "e3_pool_unitary": "unitary",
    "e3_pool_coherent": "coherent",
    "e3_pool_su4": "su4",
}
METRICS = ("accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc")
DEFAULT_TRANSFER_ROOT = Path("Results") / "q1_pooling_transfer"
DEFAULT_EVIDENCE = Path("Results") / "evidence" / "q1_pooling_transfer.json"
FROZEN_TRAIN_SAMPLES = 400
FROZEN_TOTAL_SAMPLES = 666
FROZEN_EPOCHS = 30


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _root_paths(root: Path) -> Dict[str, str]:
    """Return the four roots understood by ``run_experiments.main``."""
    root = Path(root)
    experiments = root / "experiments"
    return {
        "experiments": str(experiments),
        "runs": str(root / "runs"),
        "manifests": str(root / "manifests"),
        "failures": str(experiments / "failures.json"),
    }


def _contains_smoke_namespace(path: Path) -> bool:
    """Whether a path is explicitly under the smoke namespace.

    We inspect path components rather than substring matching so a temporary
    test directory such as ``smoke_runs`` does not get rejected accidentally.
    """
    return any(part.lower() == "smoke" or part.lower().startswith("smoke__")
               for part in path.parts)


def _validate_design(seeds: Sequence[int], arms: Sequence[str]) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    normalized_seeds = tuple(int(seed) for seed in seeds)
    normalized_arms = tuple(arms)
    if normalized_seeds != SEEDS:
        raise ValueError(
            "Fashion-MNIST pooling transfer is frozen to seeds 0, 1, 2; "
            f"received {normalized_seeds}"
        )
    if normalized_arms != POOLING_ARMS:
        raise ValueError(
            "pooling transfer arms are frozen to "
            f"{POOLING_ARMS}; received {normalized_arms}"
        )
    return normalized_seeds, normalized_arms


def _validate_budget(samples: int, epochs: int) -> None:
    if samples != FROZEN_TRAIN_SAMPLES or epochs != FROZEN_EPOCHS:
        raise ValueError(
            "Fashion-MNIST pooling transfer is frozen to 400 training samples "
            "and 30 epochs; use the separate smoke path for tiny runs"
        )


def plan_cells(
    *,
    transfer_root: Path = DEFAULT_TRANSFER_ROOT,
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[str] = POOLING_ARMS,
    samples: int = 400,
    epochs: int = 30,
) -> Dict[str, Any]:
    """Return the immutable cell schedule without touching the filesystem."""
    normalized_seeds, normalized_arms = _validate_design(seeds, arms)
    _validate_budget(samples, epochs)
    cells = [
        {
            "dataset": DATASET,
            "classes": list(CLASSES),
            "task": TASK_NAME,
            "arm": arm,
            "seed": seed,
            "evidence_role": "scientific",
            "run_directory": str(
                Path(transfer_root) / "runs" / TASK_NAME / arm / f"seed_{seed}"
            ),
        }
        for arm in normalized_arms
        for seed in normalized_seeds
    ]
    return {
        "schema": {"name": "fqcnn_q1_pooling_transfer_plan", "version": 1},
        "dataset": DATASET,
        "classes": list(CLASSES),
        "task": TASK_NAME,
        "arms": list(normalized_arms),
        "seeds": list(normalized_seeds),
        "n_cells": len(cells),
        "samples": int(samples),
        "epochs": int(epochs),
        "n_total": FROZEN_TOTAL_SAMPLES,
        "expected_split_sizes": {
            "train": FROZEN_TRAIN_SAMPLES,
            "validation": 100,
            "test": 166,
        },
        "evidence_role": "scientific",
        "smoke_policy": "smoke artifacts are ineligible for this schedule and aggregate",
        "cells": cells,
    }


def _manifest_for_seed(manifests_root: Optional[Path], seed: int) -> Optional[dict]:
    """Load the one canonical manifest for a transfer seed, if requested."""
    if manifests_root is None:
        return None
    manifests_root = Path(manifests_root)
    candidates = sorted(manifests_root.glob(f"{TASK_NAME}_n*_seed{seed}.json"))
    if not candidates:
        raise ValueError(
            f"missing canonical manifest for {TASK_NAME} seed {seed} in {manifests_root}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"ambiguous canonical manifests for {TASK_NAME} seed {seed}: "
            f"{', '.join(str(path) for path in candidates)}"
        )
    try:
        manifest = split_service.load_manifest(str(candidates[0]))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"invalid canonical manifest {candidates[0]}") from exc
    if manifest.get("dataset_id", "").split("_n", 1)[0] != TASK_NAME:
        raise ValueError(f"manifest has the wrong dataset task: {candidates[0]}")
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(f"manifest has the wrong seed: {candidates[0]}")
    if int(manifest.get("n_total", -1)) != FROZEN_TOTAL_SAMPLES:
        raise ValueError(
            f"canonical manifest has n_total={manifest.get('n_total')}; "
            f"expected {FROZEN_TOTAL_SAMPLES}: {candidates[0]}"
        )
    if len(manifest.get("train_idx", [])) != FROZEN_TRAIN_SAMPLES:
        raise ValueError(
            f"canonical manifest has {len(manifest.get('train_idx', []))} training rows; "
            f"expected {FROZEN_TRAIN_SAMPLES}: {candidates[0]}"
        )
    return manifest


def _expected_pooling_mode(arm: str) -> str:
    try:
        return POOLING_MODES[arm]
    except KeyError as exc:
        raise ValueError(f"unknown pooling arm {arm}") from exc


def _validate_arm_config(config: Any, arm: str, expected_role: str) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{arm} has no run configuration")
    role = config.get("evidence_role")
    if role != expected_role:
        raise ValueError(
            f"{arm} is marked {role!r}, expected evidence role {expected_role!r}; "
            "smoke/partial artifacts cannot enter scientific evidence"
        )
    expected_mode = _expected_pooling_mode(arm)
    observed_mode = config.get("pooling_mode")
    ablation = config.get("ablation")
    if observed_mode is None and isinstance(ablation, dict):
        observed_mode = ablation.get("pooling_mode")
    observed_name = config.get("arm")
    if observed_name is None and isinstance(ablation, dict):
        observed_name = ablation.get("name")
    if observed_name is not None and observed_name != arm:
        raise ValueError(f"run configuration identity mismatch: expected {arm}, got {observed_name}")
    if observed_mode is not None and observed_mode != expected_mode:
        raise ValueError(
            f"run configuration pooling mode mismatch for {arm}: "
            f"expected {expected_mode}, got {observed_mode}"
        )


def _load_cell(
    runs_root: Path,
    arm: str,
    seed: int,
    *,
    expected_role: str = "scientific",
    manifest: Optional[dict] = None,
) -> Dict[str, Any]:
    """Read and validate one complete transfer artifact."""
    directory = Path(run_artifacts.run_dir(
        TASK_NAME, arm, seed, root=str(runs_root), create=False
    ))
    if not run_artifacts.is_reusable(str(directory)):
        raise ValueError(f"comparison cell is incomplete or invalid: {directory}")
    try:
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or malformed status: {directory}") from exc
    _validate_arm_config(status.get("config"), arm, expected_role)
    split_id = status.get("split_id")
    if not isinstance(split_id, str) or not split_id:
        raise ValueError(f"missing canonical split identity: {directory}")
    if type(status.get("seed")) is not int or status["seed"] != int(seed):
        raise ValueError(f"seed identity mismatch: {directory}")

    try:
        with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
            required = ("sample_ids", "y_true", "raw_outputs")
            if any(name not in archive.files for name in required):
                raise ValueError(f"predictions are missing required arrays: {directory}")
            sample_ids = np.asarray(archive["sample_ids"])
            y_true = np.asarray(archive["y_true"]).reshape(-1)
            raw_outputs = np.asarray(archive["raw_outputs"], dtype=float).reshape(-1)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("predictions"):
            raise
        raise ValueError(f"invalid predictions artifact: {directory}") from exc

    if sample_ids.ndim != 1 or not (len(sample_ids) == len(y_true) == len(raw_outputs)):
        raise ValueError(f"prediction lengths are inconsistent: {directory}")
    if len(sample_ids) == 0:
        raise ValueError(f"empty test predictions are not evidence: {directory}")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f"test sample IDs are not unique: {directory}")
    if not np.all(np.isfinite(raw_outputs)):
        raise ValueError(f"non-finite predictions are not evidence: {directory}")
    if not np.all(np.isin(y_true, (-1, 1))):
        raise ValueError(f"binary labels must be -1/+1: {directory}")
    for sample_id in sample_ids.tolist():
        if not isinstance(sample_id, str) or not sample_id.startswith(f"{DATASET}:train:"):
            raise ValueError(
                f"test sample IDs must preserve {DATASET}:train source identity: {directory}"
            )

    if manifest is not None:
        expected_ids = [manifest["sample_ids"][index] for index in manifest["test_idx"]]
        if split_id != _manifest_id(manifest):
            raise ValueError(f"status split ID disagrees with canonical manifest: {directory}")
        if sample_ids.tolist() != expected_ids:
            raise ValueError(f"test sample IDs disagree with canonical manifest: {directory}")

    metrics = compute_classification_metrics(y_true, raw_outputs)
    recorded_accuracy = status.get("metrics", {}).get("accuracy")
    if recorded_accuracy is None or not np.isclose(
        float(recorded_accuracy), metrics["accuracy"], atol=1e-12, rtol=0.0
    ):
        raise ValueError(f"recorded accuracy disagrees with predictions: {directory}")
    files = {
        path.name: _sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    return {
        "directory": directory.as_posix(),
        "seed": int(seed),
        "split_id": split_id,
        "sample_ids": sample_ids,
        "sample_ids_sha256": _json_sha256(sample_ids.tolist()),
        "y_true": y_true,
        "raw_outputs": raw_outputs,
        "metrics": metrics,
        "files": files,
    }


def _manifest_id(manifest: Mapping[str, Any]) -> str:
    """Recompute a manifest ID without trusting the persisted ``id`` field."""
    candidate = dict(manifest)
    candidate.pop("id", None)
    return split_service.manifest_id(candidate)


def _validate_manifest_budget(manifest: Mapping[str, Any], seed: int) -> None:
    """Fail closed if a transfer manifest does not implement the frozen budget."""
    expected = (666, 400, 100, 166)
    observed = (
        int(manifest.get("n_total", -1)),
        len(manifest.get("train_idx", [])),
        len(manifest.get("val_idx", [])),
        len(manifest.get("test_idx", [])),
    )
    if observed != expected:
        raise ValueError(
            f"transfer manifest seed {seed} has counts {observed}; "
            f"expected n_total/train/val/test={expected}"
        )


def _paired_effect(reference: Sequence[float], comparator: Sequence[float]) -> dict:
    differences = np.asarray(reference, dtype=float) - np.asarray(comparator, dtype=float)
    std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    if std == 0.0:
        effect = 0.0 if np.allclose(differences, 0.0) else None
    else:
        effect = float(differences.mean() / std)
    return {
        "direction": "unitary_reference_minus_comparator",
        "values": differences.tolist(),
        "mean": float(differences.mean()),
        "cohens_dz": effect,
    }


def build_transfer(
    runs_root: Path,
    *,
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[str] = POOLING_ARMS,
    manifests_root: Optional[Path] = None,
    source_provenance: Optional[dict] = None,
    expected_role: str = "scientific",
) -> Dict[str, Any]:
    """Validate and aggregate the frozen Fashion-MNIST transfer matrix."""
    seeds, arms = _validate_design(seeds, arms)
    runs_root = Path(runs_root)
    if expected_role != "scientific":
        raise ValueError("only scientific transfer artifacts can be aggregated")
    if _contains_smoke_namespace(runs_root):
        raise ValueError(f"smoke namespace is ineligible for scientific evidence: {runs_root}")

    manifests = {
        seed: _manifest_for_seed(manifests_root, seed)
        for seed in seeds
    }
    for seed, manifest in manifests.items():
        if manifest is not None:
            _validate_manifest_budget(manifest, seed)
    cells: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for seed in seeds:
        cells[seed] = {
            arm: _load_cell(
                runs_root, arm, seed, expected_role=expected_role,
                manifest=manifests[seed]
            )
            for arm in arms
        }

        reference = cells[seed][REFERENCE_ARM]
        for arm in arms:
            cell = cells[seed][arm]
            if cell["split_id"] != reference["split_id"]:
                raise ValueError(f"split mismatch for seed {seed} arm {arm}")
            if not np.array_equal(cell["sample_ids"], reference["sample_ids"]):
                raise ValueError(f"sample identity mismatch for seed {seed} arm {arm}")
            if not np.array_equal(cell["y_true"], reference["y_true"]):
                raise ValueError(f"label mismatch for seed {seed} arm {arm}")

    arm_results: Dict[str, Any] = {}
    for arm in arms:
        per_seed = [cells[seed][arm] for seed in seeds]
        metric_summary: Dict[str, Any] = {}
        for metric in METRICS:
            values = [float(cell["metrics"][metric]) for cell in per_seed]
            metric_summary[metric] = {
                **qstats.mean_std(values),
                "values": values,
                "ci95": qstats.bootstrap_ci(values),
            }
        arm_results[arm] = {
            "metrics": metric_summary,
            "split_ids": [cell["split_id"] for cell in per_seed],
            "cells": [
                {
                    "seed": cell["seed"],
                    "split_id": cell["split_id"],
                    "n_test": int(len(cell["y_true"])),
                    "sample_ids_sha256": cell["sample_ids_sha256"],
                    "artifact_hashes": cell["files"],
                }
                for cell in per_seed
            ],
        }

    comparisons: Dict[str, Any] = {}
    reference_accuracy = [
        arm_results[REFERENCE_ARM]["metrics"]["accuracy"]["values"][index]
        for index in range(len(seeds))
    ]
    wilcoxon_raw: List[Optional[float]] = []
    mcnemar_raw: List[Optional[float]] = []
    for arm in arms:
        if arm == REFERENCE_ARM:
            continue
        comparator_accuracy = arm_results[arm]["metrics"]["accuracy"]["values"]
        wilcoxon = qstats.wilcoxon_across_seeds(reference_accuracy, comparator_accuracy)
        mcnemar_by_seed = {
            str(seed): qstats.mcnemar(
                cells[seed][REFERENCE_ARM]["raw_outputs"] > 0.0,
                cells[seed][arm]["raw_outputs"] > 0.0,
            )
            for seed in seeds
        }
        comparisons[arm] = {
            "paired_accuracy_effect": _paired_effect(
                reference_accuracy, comparator_accuracy
            ),
            "wilcoxon": wilcoxon,
            "mcnemar_by_seed": mcnemar_by_seed,
        }
        wilcoxon_raw.append(wilcoxon.get("p_value"))
        mcnemar_raw.append(
            qstats.mcnemar(
                np.concatenate([
                    cells[seed][REFERENCE_ARM]["raw_outputs"] > 0.0 for seed in seeds
                ]),
                np.concatenate([
                    cells[seed][arm]["raw_outputs"] > 0.0 for seed in seeds
                ]),
            ).get("p_value")
        )

    wilcoxon_correction = qstats.holm_bonferroni(wilcoxon_raw)
    mcnemar_correction = qstats.holm_bonferroni(mcnemar_raw)
    comparator_index = 0
    for arm in arms:
        if arm == REFERENCE_ARM:
            continue
        comparison = comparisons[arm]
        comparison["wilcoxon"]["p_holm"] = wilcoxon_correction["adjusted"][comparator_index]
        comparison["wilcoxon"]["significant"] = wilcoxon_correction["rejected"][comparator_index]
        comparison["mcnemar_pooled"] = qstats.mcnemar(
            np.concatenate([
                cells[seed][REFERENCE_ARM]["raw_outputs"] > 0.0 for seed in seeds
            ]),
            np.concatenate([
                cells[seed][arm]["raw_outputs"] > 0.0 for seed in seeds
            ]),
        )
        comparison["mcnemar_pooled"]["p_holm"] = mcnemar_correction["adjusted"][comparator_index]
        comparison["mcnemar_pooled"]["significant"] = mcnemar_correction["rejected"][comparator_index]
        comparator_index += 1

    manifests_summary = {
        str(seed): {
            "id": manifests[seed].get("id") if manifests[seed] is not None else None,
            "dataset_id": manifests[seed].get("dataset_id") if manifests[seed] is not None else None,
            "n_total": manifests[seed].get("n_total") if manifests[seed] is not None else None,
            "n_train": len(manifests[seed].get("train_idx", [])) if manifests[seed] is not None else None,
            "n_validation": len(manifests[seed].get("val_idx", [])) if manifests[seed] is not None else None,
            "n_test": len(manifests[seed].get("test_idx", [])) if manifests[seed] is not None else None,
        }
        for seed in seeds
    }
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": DATASET,
            "classes": list(CLASSES),
            "task": TASK_NAME,
            "evidence_role": "scientific",
            "seeds": list(seeds),
            "arms": list(arms),
            "reference_arm": REFERENCE_ARM,
            "train_sample_size": 400,
            "n_total": FROZEN_TOTAL_SAMPLES,
            "expected_split_sizes": {
                "train": FROZEN_TRAIN_SAMPLES,
                "validation": 100,
                "test": 166,
            },
            "epochs": 30,
            "canonical_split_required": True,
            "source_stable_test_identity_required": True,
            "metrics": list(METRICS),
            "multiplicity": {
                "wilcoxon": {
                    "method": "Holm-Bonferroni",
                    "alpha": wilcoxon_correction["alpha"],
                    "n_tests": wilcoxon_correction["n_tests"],
                },
                "mcnemar": {
                    "method": "Holm-Bonferroni",
                    "alpha": mcnemar_correction["alpha"],
                    "n_tests": mcnemar_correction["n_tests"],
                },
            },
        },
        "source": source_provenance,
        "manifests": manifests_summary,
        "arms": arm_results,
        "comparisons": comparisons,
    }


def write_evidence(payload: dict, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_transfer(
    *,
    data_root: Path = Path("datasets"),
    transfer_root: Path = DEFAULT_TRANSFER_ROOT,
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[str] = POOLING_ARMS,
    samples: int = 400,
    epochs: int = 30,
    jobs: int = 1,
    force: bool = False,
    output: Optional[Path] = DEFAULT_EVIDENCE,
) -> Optional[dict]:
    """Run the frozen 15-cell matrix through the existing experiment harness."""
    _validate_design(seeds, arms)
    _validate_budget(samples, epochs)
    if jobs < 1:
        raise ValueError("jobs must be at least one")
    transfer_root = Path(transfer_root)
    if _contains_smoke_namespace(transfer_root):
        raise ValueError("scientific transfer runs cannot be written under a smoke namespace")

    # ``run_experiments.main`` is an argparse entry point.  Supplying explicit
    # output roots keeps this lane isolated from both the historical tree and
    # the main comparison campaign, while this short argv context avoids a
    # second runner implementation.
    from experiments import run_experiments

    argv = [
        "run_experiments",
        "--task", f"{DATASET}:{CLASSES[0]},{CLASSES[1]}",
        "--configs", *arms,
        "--seeds", *(str(seed) for seed in seeds),
        "--samples", str(samples),
        "--epochs", str(epochs),
        "--data-root", str(data_root),
        "--no-baselines",
        "--jobs", str(jobs),
    ]
    if force:
        argv.append("--force")
    previous_argv = sys.argv
    try:
        sys.argv = argv
        run_experiments.main(output_roots=_root_paths(transfer_root))
    finally:
        sys.argv = previous_argv

    if output is None:
        return None
    provenance = dataset_registry.provenance(DATASET, Path(data_root))
    payload = build_transfer(
        transfer_root / "runs",
        seeds=seeds,
        arms=arms,
        manifests_root=transfer_root / "manifests",
        source_provenance=provenance,
    )
    write_evidence(payload, Path(output))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="print the frozen 15-cell schedule")
    plan.add_argument("--transfer-root", default=str(DEFAULT_TRANSFER_ROOT))
    plan.add_argument("--samples", type=int, default=400)
    plan.add_argument("--epochs", type=int, default=30)

    run = subparsers.add_parser("run", help="run/resume the five-arm transfer matrix")
    run.add_argument("--data-root", default="datasets")
    run.add_argument("--transfer-root", default=str(DEFAULT_TRANSFER_ROOT))
    run.add_argument("--samples", type=int, default=400)
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--force", action="store_true")
    run.add_argument("--output", default=str(DEFAULT_EVIDENCE))

    aggregate = subparsers.add_parser("aggregate", help="validate and aggregate completed cells")
    aggregate.add_argument("--runs-root", default=str(DEFAULT_TRANSFER_ROOT / "runs"))
    aggregate.add_argument("--manifests-root", default=str(DEFAULT_TRANSFER_ROOT / "manifests"))
    aggregate.add_argument("--data-root", default="datasets")
    aggregate.add_argument("--output", default=str(DEFAULT_EVIDENCE))
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan_cells(
            transfer_root=Path(args.transfer_root),
            samples=args.samples,
            epochs=args.epochs,
        ), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        payload = run_transfer(
            data_root=Path(args.data_root),
            transfer_root=Path(args.transfer_root),
            samples=args.samples,
            epochs=args.epochs,
            jobs=args.jobs,
            force=args.force,
            output=Path(args.output),
        )
        if payload is not None:
            print(json.dumps({
                "output": str(Path(args.output)),
                "cells": len(payload["protocol"]["seeds"]) * len(payload["protocol"]["arms"]),
            }, sort_keys=True))
        return 0

    provenance = dataset_registry.provenance(DATASET, Path(args.data_root))
    payload = build_transfer(
        Path(args.runs_root),
        manifests_root=Path(args.manifests_root),
        source_provenance=provenance,
    )
    write_evidence(payload, Path(args.output))
    print(json.dumps({
        "output": str(Path(args.output)),
        "cells": len(payload["protocol"]["seeds"]) * len(payload["protocol"]["arms"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
