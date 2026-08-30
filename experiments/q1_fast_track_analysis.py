"""Bounded inventory and comparison aggregation for the Q1 journal fast track.

This module implements only the frozen inventory and four-arm comparison
contracts; it is not a general research workflow framework.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from QCNN.utils import run_artifacts
from QCNN.utils import splits as split_service
from QCNN.utils.dataset_registry import Q1_TASKS
from QCNN.utils.metrics import compute_classification_metrics
from experiments import evidence_provenance
from experiments import statistics as qstats


SCHEMA = {"name": "fqcnn_q1_fast_track_inventory", "version": 1}
COMPARISON_SCHEMA = {"name": "fqcnn_q1_comparison", "version": 1}
COMPARISON_ARMS = ("proposed", "logistic", "mlp", "ttn")
COMPARISON_METRICS = (
    "accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc")
CLASSICAL_CAPACITY = {
    "input_features": 784,
    "logistic": {
        "trainable_parameters": 785,
        "accounting": "784 coefficients plus one intercept",
    },
    "mlp": {
        "hidden_layer_sizes": [2],
        "trainable_parameters": 1573,
        "accounting": "784*2 + 2 hidden biases + 2 output weights + 1 output bias",
        "matching_status": "not_parameter_matched_to_fqcnn",
        "selection_note": (
            "target_params=269 selects the smallest supported dense hidden layer; "
            "the resulting capacity is reported explicitly"
        ),
    },
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PROVENANCE_SOURCES = (
    "experiments/evidence_provenance.py",
    "experiments/q1_fast_track_analysis.py",
    "experiments/run_experiments.py",
    "experiments/statistics.py",
    "baselines/classical_cnn.py",
    "baselines/quantum_baselines.py",
    "QCNN/circuits.py",
    "QCNN/config/Qconfig.py",
    "QCNN/layers/QPool.py",
    "QCNN/models/QCNNModel.py",
    "QCNN/utils/dataset_registry.py",
    "QCNN/utils/metrics.py",
    "QCNN/utils/run_artifacts.py",
    "QCNN/utils/splits.py",
)
_UNFINISHED_STATES = {"failed", "partial", "pending", "queued", "running"}
_FROZEN_COMPARISON_SEEDS = tuple(range(5))
_FROZEN_TOTAL_SAMPLES = 666
_FROZEN_TRAIN_SAMPLES = 400
_FROZEN_VALIDATION_SAMPLES = 100
_FROZEN_TEST_SAMPLES = 166
_CLAIM_PATTERNS = {
    "numerical_result": re.compile(r"(?:\\approx|\b\d+(?:\.\d+)?\\?%|\baccuracy\b)", re.I),
    "noise_or_robustness": re.compile(r"\b(?:noise|noisy|depolari[sz]|dephas|robust)\w*", re.I),
    "hardware_scope": re.compile(r"\b(?:hardware|NISQ|device|backend|feed-forward)\b", re.I),
    "image_locality": re.compile(r"\b(?:spatial|translation|image locality|locality-preserving)\b", re.I),
    "trainability_scope": re.compile(r"\b(?:trainab|barren plateau|Lie algebra|DLA)\w*", re.I),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(root), check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def repository_snapshot(root: Path) -> Dict[str, Any]:
    try:
        branch = _git(root, "branch", "--show-current")
        upstream = _git(root, "rev-parse", "--abbrev-ref", "@{upstream}")
        ahead, behind = (
            int(value)
            for value in _git(
                root, "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
            ).split()
        )
        dirty_paths = [
            line[3:] for line in _git(root, "status", "--porcelain").splitlines() if line
        ]
        return {
            "git_sha": _git(root, "rev-parse", "HEAD"),
            "branch": branch,
            "upstream": upstream,
            "ahead": ahead,
            "behind": behind,
            "dirty": bool(dirty_paths),
            "dirty_paths": dirty_paths,
        }
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc), "dirty": True, "dirty_paths": []}


def _collect_key_values(value: Any, wanted: Set[str]) -> Dict[str, Set[str]]:
    found = {key: set() for key in wanted}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in wanted:
                    values = child if isinstance(child, list) else [child]
                    for item in values:
                        if isinstance(item, (str, int)) and not isinstance(item, bool):
                            found[key].add(str(item))
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found


def inspect_json_artifact(path: Path, root: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        record.update(
            parse_state="malformed",
            disposition="remove_claim",
            reasons=[f"malformed JSON: {exc}"],
        )
        return record

    if not isinstance(payload, dict):
        record.update(
            parse_state="valid_non_object",
            disposition="remove_claim",
            reasons=["top-level JSON is not an object"],
        )
        return record

    state = payload.get("status", payload.get("state", payload.get("completion_state")))
    schema = payload.get("schema")
    values = _collect_key_values(
        payload,
        {"seed", "seeds", "dataset", "datasets", "class_pair", "class_pairs", "split_id"},
    )
    reasons: List[str] = []
    disposition = "keep_candidate"
    normalized_state = state.lower() if isinstance(state, str) else None
    if normalized_state in _UNFINISHED_STATES:
        disposition = "bounded_rerun"
        reasons.append(f"artifact state is {normalized_state}")

    repository = payload.get("repository")
    if isinstance(repository, dict):
        dirty_policy = repository.get("dirty_policy")
        if repository.get("dirty") is True or (
            isinstance(dirty_policy, dict) and dirty_policy.get("passed") is False
        ):
            disposition = "bounded_rerun"
            reasons.append("artifact is bound to a dirty repository state")

    approvals = payload.get("approvals")
    if isinstance(approvals, dict):
        launch = approvals.get("launch")
        if isinstance(launch, dict) and launch.get("approved") is False:
            disposition = "bounded_rerun"
            reasons.append("campaign launch is not approved/completed")

    if not reasons:
        reasons.append("candidate requires semantic validation at its owning evidence gate")

    record.update(
        parse_state="valid",
        schema=schema,
        recorded_state=state,
        disposition=disposition,
        reasons=reasons,
        seeds=sorted(values["seed"] | values["seeds"]),
        datasets=sorted(values["dataset"] | values["datasets"]),
        class_pairs=sorted(values["class_pair"] | values["class_pairs"]),
        split_ids=sorted(values["split_id"]),
    )
    return record


def evidence_artifacts(root: Path) -> List[Dict[str, Any]]:
    candidates: Set[Path] = set()
    for directory in (root / "Results" / "evidence", root / "Results" / "campaigns"):
        if directory.exists():
            candidates.update(directory.rglob("*.json"))
    for name in ("headline_manifest.json", "metrics.json"):
        path = root / "Results" / name
        if path.exists():
            candidates.add(path)
    # The inventory cannot hash itself, and it must not snapshot the derived
    # claim ledgers: the claim ledger hashes the inventory, so including it
    # here would create a circular, permanently stale previous-snapshot hash.
    derived_ledgers = {
        "q1_fast_track_inventory.json",
        "q1_claim_ledger.json",
        "q1_claim_ledger_draft.json",
    }
    candidates = {path for path in candidates if path.name not in derived_ledgers}
    return [inspect_json_artifact(path, root) for path in sorted(candidates)]


def run_groups(root: Path) -> List[Dict[str, Any]]:
    runs_root = root / "Results" / "runs"
    groups: Dict[str, Dict[str, Any]] = {}
    if not runs_root.exists():
        return []

    for status_path in sorted(runs_root.rglob("status.json")):
        run_dir = status_path.parent
        relative = run_dir.relative_to(runs_root)
        if len(relative.parts) < 3:
            continue
        dataset_pair, config = relative.parts[0], relative.parts[1]
        key = f"{dataset_pair}/{config}"
        group = groups.setdefault(
            key,
            {
                "dataset_pair": dataset_pair,
                "config": config,
                "valid_seeds": [],
                "invalid_runs": [],
                "split_ids": [],
            },
        )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            seed = status.get("seed")
            valid = run_artifacts.is_reusable(
                str(run_dir), config=status.get("config"), seed=seed
            )
            if valid:
                group["valid_seeds"].append(seed)
                if isinstance(status.get("split_id"), str):
                    group["split_ids"].append(status["split_id"])
            else:
                group["invalid_runs"].append(_relative(run_dir, root))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            group["invalid_runs"].append(_relative(run_dir, root))

    result = []
    for key in sorted(groups):
        group = groups[key]
        group["valid_seeds"] = sorted(set(group["valid_seeds"]))
        group["split_ids"] = sorted(set(group["split_ids"]))
        group["disposition"] = (
            "keep_candidate" if group["valid_seeds"] and not group["invalid_runs"]
            else "bounded_rerun"
        )
        result.append(group)
    return result


def manuscript_findings(path: Path, root: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    body = text.split("\\begin{thebibliography}", 1)[0]
    findings = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        categories = [name for name, pattern in _CLAIM_PATTERNS.items() if pattern.search(line)]
        if not categories:
            continue
        findings.append(
            {
                "line": line_number,
                "categories": categories,
                "text": line.strip()[:500],
                "disposition": "review",
            }
        )
    return findings


def _lock_record(path: Path, root: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return {"path": _relative(path, root), "sha256": sha256_file(path)}


def build_inventory(
    root: Path,
    manuscript: Path,
    full_suite_passed: Optional[int] = None,
    full_suite_skipped: Optional[int] = None,
) -> Dict[str, Any]:
    artifacts = evidence_artifacts(root)
    groups = run_groups(root)
    findings = manuscript_findings(manuscript, root)
    dispositions: Dict[str, int] = {}
    for artifact in artifacts:
        name = artifact["disposition"]
        dispositions[name] = dispositions.get(name, 0) + 1
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository_snapshot(root),
        "environment_locks": [
            record
            for record in (
                _lock_record(root / "requirements-lock.txt", root),
                _lock_record(root / "requirements-qiskit-lock.txt", root),
            )
            if record is not None
        ],
        "verification": {
            "command": "python -m pytest tests/ -q",
            "passed": full_suite_passed,
            "skipped": full_suite_skipped,
        },
        "manuscript": {
            "path": _relative(manuscript, root),
            "sha256": sha256_file(manuscript),
            "findings": findings,
        },
        "evidence_artifacts": artifacts,
        "run_groups": groups,
        "summary": {
            "evidence_artifacts": len(artifacts),
            "artifact_dispositions": dispositions,
            "run_groups": len(groups),
            "valid_run_cells": sum(len(group["valid_seeds"]) for group in groups),
            "invalid_run_cells": sum(len(group["invalid_runs"]) for group in groups),
            "manuscript_findings": len(findings),
        },
    }


def write_inventory(payload: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_name(dataset: str, classes: Sequence[int]) -> str:
    return f"{dataset}_{classes[0]}v{classes[1]}"


def _cell_directory(runs_root: Path, task_name: str, arm: str, seed: int) -> Path:
    config = "proposed" if arm == "proposed" else f"baseline_{arm}"
    return runs_root / task_name / config / f"seed_{seed}"


def _normalise_comparison_tasks(
    tasks: Sequence[Tuple[str, Sequence[int], str]],
) -> Tuple[Tuple[str, Tuple[int, int], str], ...]:
    """Normalise and validate the task tuple before it reaches the filesystem."""
    normalised = []
    for task in tasks:
        if not isinstance(task, (tuple, list)) or len(task) != 3:
            raise ValueError(
                "comparison tasks must be (dataset, (class_a, class_b), role) tuples"
            )
        dataset, classes, role = task
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("comparison task dataset names must be non-empty strings")
        if not isinstance(classes, (tuple, list)) or len(classes) != 2:
            raise ValueError("comparison task classes must contain two labels")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
               for value in classes):
            raise ValueError("comparison task classes must be integer labels")
        class_pair = (int(classes[0]), int(classes[1]))
        if class_pair[0] == class_pair[1]:
            raise ValueError("comparison task classes must be distinct")
        if not isinstance(role, str) or not role:
            raise ValueError("comparison task roles must be non-empty strings")
        normalised.append((dataset, class_pair, role))
    return tuple(normalised)


def _canonical_comparison_manifest(
    manifests_root: Path,
    dataset: str,
    classes: Sequence[int],
    seed: int,
) -> Dict[str, Any]:
    """Load and verify the frozen manifest for one Q1 task/seed."""
    task_name = _task_name(dataset, classes)
    manifests_root = Path(manifests_root)
    candidates = sorted(
        manifests_root.glob(f"{task_name}_n*_seed{seed}.json")
    )
    if len(candidates) != 1:
        state = "missing" if not candidates else "ambiguous"
        raise ValueError(
            f"{state} canonical comparison manifest for {task_name} seed {seed} "
            f"in {manifests_root}"
        )
    path = candidates[0]
    try:
        manifest = split_service.load_manifest(str(path))
        expected_id = split_service.manifest_id(manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid canonical comparison manifest: {path}") from exc

    expected_dataset_id = f"{task_name}_n{_FROZEN_TOTAL_SAMPLES}"
    if manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError(
            f"comparison manifest has dataset_id={manifest.get('dataset_id')!r}; "
            f"expected {expected_dataset_id!r}: {path}"
        )
    if type(manifest.get("seed")) is not int or manifest["seed"] != int(seed):
        raise ValueError(f"comparison manifest has the wrong seed: {path}")
    if manifest.get("id") != expected_id:
        raise ValueError(f"comparison manifest ID is not self-consistent: {path}")
    if manifest.get("class_mapping") != split_service.class_mapping_for(classes):
        raise ValueError(f"comparison manifest class mapping is wrong: {path}")
    observed_sizes = (
        int(manifest.get("n_total", -1)),
        len(manifest.get("train_idx", [])),
        len(manifest.get("val_idx", [])),
        len(manifest.get("test_idx", [])),
    )
    expected_sizes = (
        _FROZEN_TOTAL_SAMPLES,
        _FROZEN_TRAIN_SAMPLES,
        _FROZEN_VALIDATION_SAMPLES,
        _FROZEN_TEST_SAMPLES,
    )
    if observed_sizes != expected_sizes:
        raise ValueError(
            f"comparison manifest has sizes {observed_sizes}; expected {expected_sizes}: {path}"
        )
    expected_prefix = f"{dataset}:train:"
    if any(not isinstance(sample_id, str) or not sample_id.startswith(expected_prefix)
           for sample_id in manifest.get("sample_ids", [])):
        raise ValueError(
            f"comparison manifest sample IDs do not preserve {expected_prefix!r}: {path}"
        )
    return manifest


def _validate_comparison_run_config(status: Dict[str, Any], arm: str, seed: int) -> None:
    """Enforce the geometry and role of a scientific comparison cell."""
    config = status.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{arm} comparison cell has no run configuration")
    if type(status.get("seed")) is not int or status["seed"] != int(seed):
        raise ValueError(f"{arm} comparison cell has the wrong seed")
    if arm == "proposed":
        if config.get("evidence_role") != "scientific":
            raise ValueError(
                f"{arm} comparison cell is not marked scientific evidence"
            )
        required = {
            "image_size": 28,
            "n_qubits": 10,
            "encoding_type": "amplitude",
            "pooling_mode": "unitary",
            "conv_entanglement": "full",
            "kernel_rotations": "su2",
            "n_epochs": 30,
        }
        observed = {key: config.get(key) for key in required}
        if observed != required:
            raise ValueError(
                f"proposed comparison geometry/configuration is not frozen: {observed}"
            )
        ablation = config.get("ablation")
        if not isinstance(ablation, dict) or ablation.get("name") != "proposed":
            raise ValueError("proposed comparison cell has no proposed ablation identity")
    else:
        expected = {
            "artifact_schema_version": 1,
            "baseline": arm,
            "evidence_role": "scientific",
        }
        legacy = {"artifact_schema_version": 1}
        # Baselines produced before the role/identity metadata was introduced
        # remain admissible only inside this fully canonical matrix: the
        # proposed cell is scientific, all arms share its canonical split and
        # ordered test IDs, and the baseline path names the requested arm.
        if config not in (expected, legacy):
            raise ValueError(
                f"{arm} baseline configuration is not the frozen scientific identity"
            )


def _load_comparison_cell(
    directory: Path,
    arm: str,
    seed: int,
    *,
    manifest: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    status_path = directory / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or malformed comparison status: {status_path}") from exc
    if not run_artifacts.is_reusable(str(directory), seed=seed):
        raise ValueError(f"comparison cell is incomplete or invalid: {directory}")
    if strict:
        _validate_comparison_run_config(status, arm, seed)
        if manifest is None:
            raise ValueError(f"missing canonical manifest for comparison cell: {directory}")
        expected_ids = [manifest["sample_ids"][index] for index in manifest["test_idx"]]
        if status.get("split_id") != manifest.get("id"):
            raise ValueError(f"comparison split ID disagrees with canonical manifest: {directory}")
    if arm != "proposed":
        try:
            selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"baseline selection evidence is invalid: {directory}") from exc
        if selection.get("test_evaluations") != 1:
            raise ValueError(f"baseline did not record exactly one test evaluation: {directory}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        sample_ids = np.asarray(archive["sample_ids"])
        y_true = np.asarray(archive["y_true"])
        raw_outputs = np.asarray(archive["raw_outputs"], dtype=float)
    if not (len(sample_ids) == len(y_true) == len(raw_outputs)) or len(sample_ids) == 0:
        raise ValueError(f"comparison predictions have invalid lengths: {directory}")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f"comparison sample IDs are not unique: {directory}")
    if strict and sample_ids.tolist() != expected_ids:
        raise ValueError(f"comparison test IDs disagree with canonical manifest: {directory}")
    if strict and not np.all(np.isin(y_true, (-1, 1))):
        raise ValueError(f"comparison labels are not binary: {directory}")
    if strict and not np.all(np.isfinite(raw_outputs)):
        raise ValueError(f"comparison predictions are non-finite: {directory}")
    metrics = compute_classification_metrics(y_true, raw_outputs)
    recorded_accuracy = status.get("metrics", {}).get("accuracy")
    if recorded_accuracy is None or not np.isclose(
            float(recorded_accuracy), metrics["accuracy"], atol=1e-12, rtol=0.0):
        raise ValueError(f"recorded accuracy disagrees with predictions: {directory}")
    return {
        "directory": directory.as_posix(),
        "split_id": status["split_id"],
        "sample_ids": sample_ids,
        "y_true": y_true,
        "raw_outputs": raw_outputs,
        "correct": np.where(raw_outputs > 0.0, 1, -1) == y_true,
        "metrics": metrics,
        "files": {
            path.name: sha256_file(path)
            for path in sorted(directory.iterdir()) if path.is_file()
        },
    }


def _paired_effect(reference: Sequence[float], comparator: Sequence[float]) -> Dict[str, Any]:
    reference = np.asarray(reference, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    differences = reference - comparator
    std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    if std == 0.0:
        effect = 0.0 if np.allclose(differences, 0.0) else None
    else:
        effect = float(differences.mean() / std)
    return {
        "direction": "proposed_minus_comparator",
        "values": differences.tolist(),
        "mean": float(differences.mean()),
        "cohens_dz": effect,
    }


def build_comparison(
    runs_root: Path,
    *,
    seeds: Sequence[int] = tuple(range(5)),
    tasks: Optional[Sequence[Tuple[str, Sequence[int], str]]] = None,
    arms: Sequence[str] = COMPARISON_ARMS,
    manifests_root: Optional[Path] = None,
    provenance_root: Optional[Path] = None,
    allow_noncanonical: bool = False,
) -> Dict[str, Any]:
    """Validate and aggregate only the frozen Q1 comparison matrix.

    Scientific aggregation is deliberately fail-closed: callers must use the
    predeclared tasks/seeds and canonical split manifests.  Tiny or synthetic
    fixtures can still exercise the numerical aggregation code by opting into
    ``allow_noncanonical=True``; that mode is never used by the command-line
    scientific aggregate.
    """
    tasks = _normalise_comparison_tasks(Q1_TASKS if tasks is None else tasks)
    seeds = tuple(int(seed) for seed in seeds)
    arms = tuple(arms)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("comparison seeds must be a non-empty unique sequence")
    if arms != COMPARISON_ARMS:
        raise ValueError(f"comparison arms must be exactly {COMPARISON_ARMS}")
    if not allow_noncanonical:
        expected_tasks = _normalise_comparison_tasks(Q1_TASKS)
        if tasks != expected_tasks:
            raise ValueError(
                "scientific comparison tasks must exactly match the frozen Q1 task registry"
            )
        if seeds != _FROZEN_COMPARISON_SEEDS:
            raise ValueError(
                "scientific comparison is frozen to seeds 0, 1, 2, 3, 4"
            )
        if manifests_root is None:
            manifests_root = Path(runs_root).resolve().parent / "manifests"

    task_results: Dict[str, Any] = {}
    primary_tests = []
    for dataset, classes, role in tasks:
        task_name = _task_name(dataset, classes)
        manifest_by_seed = {
            seed: _canonical_comparison_manifest(
                manifests_root, dataset, classes, seed)
            for seed in seeds
        } if not allow_noncanonical else {seed: None for seed in seeds}
        cells = {
            arm: {
                seed: _load_comparison_cell(
                    _cell_directory(runs_root, task_name, arm, seed), arm, seed,
                    manifest=manifest_by_seed[seed],
                    strict=not allow_noncanonical,
                )
                for seed in seeds
            }
            for arm in arms
        }
        for seed in seeds:
            reference = cells["proposed"][seed]
            for arm in arms[1:]:
                candidate = cells[arm][seed]
                if candidate["split_id"] != reference["split_id"]:
                    raise ValueError(f"split mismatch for {task_name} seed {seed} arm {arm}")
                if not np.array_equal(candidate["sample_ids"], reference["sample_ids"]):
                    raise ValueError(f"sample identity mismatch for {task_name} seed {seed} arm {arm}")
                if not np.array_equal(candidate["y_true"], reference["y_true"]):
                    raise ValueError(f"label mismatch for {task_name} seed {seed} arm {arm}")

        arm_results = {}
        for arm in arms:
            metric_summary = {}
            for metric in COMPARISON_METRICS:
                values = [cells[arm][seed]["metrics"][metric] for seed in seeds]
                metric_summary[metric] = {
                    **qstats.mean_std(values),
                    "values": values,
                    "ci95": qstats.bootstrap_ci(values),
                }
            arm_results[arm] = {
                "metrics": metric_summary,
                "split_ids": [cells[arm][seed]["split_id"] for seed in seeds],
                "artifact_hashes": {
                    str(seed): cells[arm][seed]["files"] for seed in seeds
                },
            }

        comparisons = {}
        proposed_accuracy = arm_results["proposed"]["metrics"]["accuracy"]["values"]
        for arm in arms[1:]:
            arm_accuracy = arm_results[arm]["metrics"]["accuracy"]["values"]
            wilcoxon = qstats.wilcoxon_across_seeds(proposed_accuracy, arm_accuracy)
            mcnemar_by_seed = {
                str(seed): qstats.mcnemar(
                    cells["proposed"][seed]["correct"], cells[arm][seed]["correct"])
                for seed in seeds
            }
            comparisons[arm] = {
                "paired_accuracy_effect": _paired_effect(proposed_accuracy, arm_accuracy),
                "wilcoxon": wilcoxon,
                "mcnemar_by_seed": mcnemar_by_seed,
            }
            primary_tests.append((task_name, arm, wilcoxon.get("p_value")))

        task_results[task_name] = {
            "dataset": dataset,
            "classes": list(classes),
            "role": role,
            "seeds": list(seeds),
            "arms": arm_results,
            "proposed_vs": comparisons,
        }

    correction = qstats.holm_bonferroni([item[2] for item in primary_tests])
    for index, (task_name, arm, raw_p) in enumerate(primary_tests):
        test = task_results[task_name]["proposed_vs"][arm]["wilcoxon"]
        test["family"] = "all task-by-primary-comparator accuracy tests"
        test["p_holm"] = correction["adjusted"][index]
        test["significant"] = correction["rejected"][index]

    return {
        "schema": COMPARISON_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": evidence_provenance.build_binding(
            _REPOSITORY_ROOT if provenance_root is None else Path(provenance_root),
            source_paths=COMPARISON_PROVENANCE_SOURCES,
        ),
        "protocol": {
            "tasks_frozen_before_test_inspection": not allow_noncanonical,
            "seeds": list(seeds),
            "arms": list(arms),
            "metrics": list(COMPARISON_METRICS),
            "primary_reference": "proposed",
            "canonical_manifests_required": not allow_noncanonical,
            "classical_capacity": CLASSICAL_CAPACITY,
            "multiplicity": {
                "method": "Holm-Bonferroni",
                "alpha": correction["alpha"],
                "n_tests": correction["n_tests"],
            },
        },
        "tasks": task_results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory", help="inventory existing evidence and claims")
    inventory.add_argument("--root", default=".")
    inventory.add_argument("--manuscript", default="fqcnn.tex")
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--full-suite-passed", type=int)
    inventory.add_argument("--full-suite-skipped", type=int)
    aggregate = subparsers.add_parser(
        "aggregate", help="validate and aggregate the frozen Q1 comparison matrix")
    aggregate.add_argument("--root", default=".")
    aggregate.add_argument("--runs-root", default="Results/q1_comparison/runs")
    aggregate.add_argument("--manifests-root", default="Results/q1_comparison/manifests")
    aggregate.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    aggregate.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if args.command == "aggregate":
        payload = build_comparison(
            (root / args.runs_root).resolve(),
            manifests_root=(root / args.manifests_root).resolve(),
            seeds=args.seeds,
        )
        write_inventory(payload, output)
        print(json.dumps({"tasks": len(payload["tasks"]), "output": str(output)}, sort_keys=True))
        return 0
    manuscript = (root / args.manuscript).resolve()
    payload = build_inventory(root, manuscript,
                              full_suite_passed=args.full_suite_passed,
                              full_suite_skipped=args.full_suite_skipped)
    write_inventory(payload, output)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
