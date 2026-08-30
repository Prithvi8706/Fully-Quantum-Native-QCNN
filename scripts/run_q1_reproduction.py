"""Run the bounded Q1 release checks and write a machine-readable report.

The report is generated only after the checks complete. It records the exact Git
revision tested, command exit codes, pytest counts, and semantic evidence-gate
result; it does not claim that a journal or a real quantum device was used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    from scripts import build_q1_submission_package as package
except ImportError:  # direct ``python scripts/run_q1_reproduction.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import build_q1_submission_package as package


_ABSOLUTE_WINDOWS_PATH = re.compile(
    r"(?i)[A-Z]:[\\/]+(?:[^\\/\r\n\"']+[\\/]+)*[^\\/\r\n\"']+"
)


EVIDENCE_REQUIRED = (
    "Results/evidence/q1_claim_ledger.json",
    "Results/evidence/q1_comparison.json",
    "Results/evidence/q1_dataset_provenance.json",
    "Results/evidence/q1_fake_backend_rehearsal.json",
    "Results/evidence/q1_noise_validation.json",
    "Results/evidence/q1_pooling_controls.json",
    "Results/evidence/q1_pooling_practicality.json",
    "Results/evidence/q1_pooling_transfer.json",
    "Results/evidence/q1_resources.json",
    "Results/evidence/q1_reproduction.json",
)

# The claim ledger is deliberately checked against a fixed inventory rather
# than whatever paths happen to be present in the JSON.  Otherwise an edited
# ledger could simply remove the hash for the changed source/artifact and the
# release check would still report success.  These are the release inputs
# whose identity controls the Q1 numbers and their interpretation.  The
# ledger may contain additional historical artifacts, but it may not omit one
# of these bindings.
LEDGER_REQUIRED_SOURCE_PATHS = (
    "fqcnn.tex",
    "docs/paper_code_reconciliation.md",
    "Results/evidence/q1_fast_track_inventory.json",
    "experiments/evidence_provenance.py",
    "experiments/q1_fast_track_analysis.py",
    "experiments/q1_local_evidence.py",
    "experiments/q1_pooling_transfer.py",
    "experiments/run_experiments.py",
    "experiments/statistics.py",
    "baselines/classical_cnn.py",
    "baselines/quantum_baselines.py",
    "QCNN/circuits.py",
    "QCNN/config/Qconfig.py",
    "QCNN/layers/QPool.py",
    "QCNN/models/QCNNModel.py",
    "QCNN/training/Qtrainer.py",
    "QCNN/utils/dataset_registry.py",
    "QCNN/utils/exclusive_queue.py",
    "QCNN/utils/metrics.py",
    "QCNN/utils/run_artifacts.py",
    "QCNN/utils/splits.py",
    "requirements-lock.txt",
    "requirements-qiskit-lock.txt",
    "tests/fixtures/effective_params.json",
    "tests/fixtures/headline_signature.json",
    "tests/fixtures/headline_expectations.npz",
    "scripts/build_q1_submission_package.py",
    "scripts/refresh_q1_claim_ledger.py",
    "scripts/run_q1_reproduction.py",
)

RELEASE_LINT_PATHS = (
    "QCNN/utils/dataset_registry.py",
    "QCNN/utils/exclusive_queue.py",
    "experiments/evidence_provenance.py",
    "experiments/pooling_analysis.py",
    "experiments/q1_fast_track_analysis.py",
    "experiments/q1_pooling_transfer.py",
    "experiments/run_experiments.py",
    "experiments/statistics.py",
    "scripts/build_q1_submission_package.py",
    "scripts/refresh_q1_claim_ledger.py",
    "scripts/run_q1_reproduction.py",
    "tests/test_dataset_registry.py",
    "tests/test_evidence_provenance.py",
    "tests/test_pooling_equivalence.py",
    "tests/test_q1_local_evidence.py",
    "tests/test_q1_fast_track.py",
    "tests/test_q1_pooling_transfer.py",
    "tests/test_q1_reproduction.py",
    "tests/test_q1_submission_package.py",
)

# ``q1_reproduction.json`` is written after the checks complete, so it cannot
# be part of the source fingerprint captured before/while the report is
# built.  The package gate validates this fingerprint against the current
# checkout and therefore permits metadata-only commits after the expensive
# test run while still rejecting source drift.
REPRODUCTION_SOURCE_PATHS = tuple(
    dict.fromkeys((
        "fqcnn.tex",
        "requirements-lock.txt",
        "requirements-qiskit-lock.txt",
        *RELEASE_LINT_PATHS,
    ))
)

QISKIT_SCHEDULE_CHECK = (
    "from experiments import q1_local_evidence as e\n"
    "state = e._build_state_preparation(4)\n"
    "body = e._build_model_body(4, 'unitary', seed=0)\n"
    "assert state.count_ops().get('initialize') == 1\n"
    "assert body.count_ops().get('initialize', 0) == 0\n"
    "assert body.count_ops().get('cry') == 3\n"
    "assert body.count_ops().get('crz') == 3\n"
    "assert body.count_ops().get('cx', 0) > 0\n"
    "dynamic = e._build_model_body(4, 'measurement', seed=0)\n"
    "counts = {str(k): int(v) for k, v in dynamic.count_ops().items()}\n"
    "assert counts.get('measure') == 3\n"
    "assert counts.get('if_else') == 3\n"
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    )
    return result.stdout.rstrip("\r\n")


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite(child) for child in value)
    return True


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> Any:
    """Remove only volatile aggregate timestamps before comparing JSON values."""

    if isinstance(value, dict):
        return {
            key: _canonical_json(child)
            for key, child in value.items()
            if key != "generated_at_utc"
        }
    if isinstance(value, list):
        return [_canonical_json(child) for child in value]
    return value


def _first_difference(expected: Any, observed: Any, path: str = "$") -> Optional[str]:
    """Return a path for the first semantic JSON difference, without values."""

    if type(expected) is not type(observed):
        return path
    if isinstance(expected, dict):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            return f"{path} (keys)"
        for key in sorted(expected_keys):
            difference = _first_difference(
                expected[key], observed[key], f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path} (length)"
        for index, (left, right) in enumerate(zip(expected, observed)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if expected == observed else path


def _validate_aggregate_matches(
    label: str, stored: Any, recomputed: Any
) -> list[str]:
    """Check an aggregate's semantic content, not just its JSON shape."""

    difference = _first_difference(
        _canonical_json(recomputed), _canonical_json(stored)
    )
    if difference is None:
        return []
    return [f"{label} differs from canonical run-artifact recomputation at {difference}"]


def _safe_repo_file(root: Path, relative: str) -> Optional[Path]:
    """Resolve a ledger path while rejecting absolute paths and traversal."""

    if not isinstance(relative, str) or not relative:
        return None
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _validate_hash_bindings(
    root: Path, ledger: Any, *, check_hashes: bool = True
) -> list[str]:
    """Validate the claim ledger's complete, repository-relative hash bindings."""

    errors: list[str] = []
    if not isinstance(ledger, dict):
        return ["claim ledger is not a JSON object"]
    if ledger.get("schema") != {"name": "fqcnn_q1_claim_ledger", "version": 2}:
        errors.append("claim ledger schema is missing or unsupported")
    if ledger.get("status") != "ready_for_release":
        errors.append("claim ledger is not ready_for_release")
    for field in ("unresolved_items", "release_blockers"):
        value = ledger.get(field)
        if value:
            errors.append(f"claim ledger has non-empty {field}")

    source_hashes = ledger.get("source_hashes_sha256")
    if not isinstance(source_hashes, dict):
        errors.append("claim ledger source_hashes_sha256 is missing or invalid")
        source_hashes = {}
    for relative in LEDGER_REQUIRED_SOURCE_PATHS:
        if relative not in source_hashes:
            errors.append(f"claim ledger is missing source hash: {relative}")
    for relative, expected in source_hashes.items():
        path = _safe_repo_file(root, relative)
        if path is None:
            errors.append(f"claim ledger source path is unsafe: {relative}")
            continue
        if not _SHA256.fullmatch(str(expected)):
            errors.append(f"claim ledger source hash is invalid: {relative}")
            continue
        if not path.is_file():
            errors.append(f"claim ledger source file is missing: {relative}")
            continue
        if check_hashes and package.sha256_file(path) != expected:
            errors.append(f"claim ledger source hash mismatch: {relative}")

    inventory = ledger.get("artifact_inventory")
    if not isinstance(inventory, list):
        errors.append("claim ledger artifact_inventory is missing or invalid")
        inventory = []
    artifact_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(inventory):
        if not isinstance(record, dict):
            errors.append(f"claim ledger artifact record {index} is invalid")
            continue
        relative = record.get("path")
        path = _safe_repo_file(root, relative)
        if path is None:
            errors.append(f"claim ledger artifact path is unsafe: {relative}")
            continue
        if relative in artifact_records:
            errors.append(f"claim ledger has duplicate artifact hash: {relative}")
            continue
        artifact_records[relative] = record
        expected = record.get("sha256")
        if not _SHA256.fullmatch(str(expected)):
            errors.append(f"claim ledger artifact hash is invalid: {relative}")
            continue
        if not path.is_file():
            errors.append(f"claim ledger artifact is missing: {relative}")
            continue
        if check_hashes and package.sha256_file(path) != expected:
            errors.append(f"claim ledger artifact hash mismatch: {relative}")

    # The ledger itself cannot be listed here without creating a circular hash.
    # Every other required evidence file must have an independently recorded
    # digest, including the reproduction report and pooling controls.
    for relative in EVIDENCE_REQUIRED:
        if relative == "Results/evidence/q1_claim_ledger.json":
            continue
        if relative not in artifact_records:
            errors.append(f"claim ledger is missing artifact hash: {relative}")
    return errors


def _validate_reproduction_source(
    root: Path,
    source: Any,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
) -> list[str]:
    """Validate a report's tested revision and source fingerprint.

    The commit/tree identify the exact checkout used to start the run.  A
    later metadata-only commit is permitted when every source file hashed in
    the report still matches the current checkout; this is necessary because
    the report and claim ledger themselves are committed after the expensive
    checks complete.
    """

    if not isinstance(source, dict):
        return ["reproduction report source identity is missing"]
    errors: list[str] = []
    commit = source.get("commit")
    tree = source.get("tree")
    if not isinstance(commit, str) or not commit:
        errors.append("reproduction report source commit is missing")
    if not isinstance(tree, str) or not tree:
        errors.append("reproduction report source tree is missing")
    if source.get("tracked_tree_clean_before_run") is not True:
        errors.append("reproduction report was not generated from a clean tracked tree")

    # A source fingerprint is useful only if its claimed revision is a real
    # Git object whose tree matches the recorded tree.  Do this check when the
    # caller supplied the current revision (the strict release path); the
    # small helper unit tests can still exercise hash comparison in isolation.
    if expected_commit or expected_tree:
        if isinstance(commit, str) and isinstance(tree, str) and commit and tree:
            try:
                recorded_tree = _git(root, "rev-parse", f"{commit}^{{tree}}")
            except (OSError, subprocess.CalledProcessError):
                errors.append("reproduction report source commit is not a valid Git revision")
            else:
                if recorded_tree != tree:
                    errors.append("reproduction report source tree disagrees with its commit")

    source_files = source.get("source_files")
    records: dict[str, dict[str, Any]] = {}
    if not isinstance(source_files, list):
        source_files = []
        errors.append("reproduction report source_files fingerprint is missing")
    for index, record in enumerate(source_files):
        if not isinstance(record, dict):
            errors.append(f"reproduction source record {index} is invalid")
            continue
        relative = record.get("path")
        path = _safe_repo_file(root, relative)
        if path is None:
            errors.append(f"reproduction source path is unsafe: {relative}")
            continue
        if relative in records:
            errors.append(f"reproduction source path is duplicated: {relative}")
            continue
        records[relative] = record
        expected = record.get("sha256")
        if not _SHA256.fullmatch(str(expected)):
            errors.append(f"reproduction source hash is invalid: {relative}")
            continue
        if not path.is_file():
            errors.append(f"reproduction source file is missing: {relative}")
            continue
        if package.sha256_file(path) != expected:
            errors.append(f"reproduction source hash mismatch: {relative}")

    if source_files:
        for relative in REPRODUCTION_SOURCE_PATHS:
            if relative not in records:
                errors.append(f"reproduction source fingerprint is missing: {relative}")
    elif expected_commit and commit != expected_commit:
        # Compatibility behavior for old reports: without a source
        # fingerprint, only an exact current revision can be trusted.
        errors.append("reproduction report source commit does not match current HEAD")
    elif expected_tree and tree != expected_tree:
        errors.append("reproduction report source tree does not match current HEAD")
    return errors


def _recompute_canonical_aggregates(
    root: Path, stored: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild the two bounded aggregates from committed run artifacts.

    This does not train models or access a service.  It parses the existing
    predictions/status/manifests, recomputes metrics and statistics, and
    therefore remains cheap enough for every release/package check.
    """

    from experiments import q1_fast_track_analysis as comparison_analysis
    from experiments import q1_pooling_transfer as transfer_analysis

    comparison = comparison_analysis.build_comparison(
        root / "Results/q1_comparison/runs", provenance_root=root
    )
    stored_transfer = stored["Results/evidence/q1_pooling_transfer.json"]
    transfer = transfer_analysis.build_transfer(
        root / "Results/q1_pooling_transfer/runs",
        manifests_root=root / "Results/q1_pooling_transfer/manifests",
        source_provenance=stored_transfer.get("source"),
        provenance_root=root,
    )
    return {
        "Results/evidence/q1_comparison.json": comparison,
        "Results/evidence/q1_pooling_transfer.json": transfer,
    }


def _canonical_run_artifact_state(root: Path) -> tuple[str, str | None]:
    """Determine whether release-local run artifacts can be recomputed.

    Run directories are intentionally excluded from the submission archive.
    A fresh clone therefore has neither canonical run root and must rely on
    the committed evidence/ledger bindings.  A one-sided checkout is not a
    valid release state: silently skipping there would hide a partial or
    accidentally deleted campaign.
    """

    comparison_root = root / "Results/q1_comparison/runs"
    transfer_root = root / "Results/q1_pooling_transfer/runs"
    comparison_exists = comparison_root.exists()
    transfer_exists = transfer_root.exists()
    if comparison_exists != transfer_exists:
        return (
            "partial",
            "exactly one canonical run-artifact root is present; refusing to skip recomputation",
        )
    if not comparison_exists:
        return (
            "skipped",
            "canonical run artifacts are absent; relying on committed evidence and ledger hashes",
        )
    return "available", None


def validate_evidence(
    root: Path,
    *,
    verify_bindings: bool = True,
    require_current_report: bool = True,
) -> dict[str, Any]:
    """Validate required evidence shape, provenance, and canonical aggregates.

    ``verify_bindings=False`` is intended only for the report-generation
    preflight: that command is about to write a new reproduction report and
    refresh the ledger.  Package construction always uses the strict defaults.
    """

    root = Path(root).resolve()
    errors: list[str] = []
    loaded: dict[str, Any] = {}
    for relative in EVIDENCE_REQUIRED:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing required evidence: {relative}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON {relative}: {exc}")
            continue
        if not isinstance(payload, dict) or not _finite(payload):
            errors.append(f"non-object or non-finite JSON: {relative}")
            continue
        loaded[relative] = payload

    ledger = loaded.get("Results/evidence/q1_claim_ledger.json")
    provenance_errors = _validate_hash_bindings(
        root, ledger, check_hashes=verify_bindings
    )
    errors.extend(provenance_errors)

    comparison = loaded.get("Results/evidence/q1_comparison.json", {})
    protocol = comparison.get("protocol", {}) if isinstance(comparison, dict) else {}
    if not isinstance(protocol, dict):
        protocol = {}
        errors.append("comparison protocol is not a JSON object")
    if protocol.get("arms") != ["proposed", "logistic", "mlp", "ttn"]:
        errors.append("comparison protocol arms differ from the frozen four-arm matrix")
    if protocol.get("seeds") != [0, 1, 2, 3, 4]:
        errors.append("comparison protocol does not contain seeds 0..4")
    tasks = comparison.get("tasks", {}) if isinstance(comparison, dict) else {}
    if not isinstance(tasks, dict) or len(tasks) != 4:
        errors.append("comparison aggregate does not contain four frozen tasks")

    transfer = loaded.get("Results/evidence/q1_pooling_transfer.json", {})
    if not isinstance(transfer, dict):
        transfer = {}
        errors.append("pooling-transfer evidence is not a JSON object")
    if transfer.get("status") == "fail":
        errors.append("pooling-transfer evidence has status=fail")
    controls = loaded.get("Results/evidence/q1_pooling_controls.json", {})
    if not isinstance(controls, dict) or controls.get("status") != "pass":
        errors.append("pooling-controls evidence is not pass")

    report = loaded.get("Results/evidence/q1_reproduction.json", {})
    if not isinstance(report, dict) or report.get("status") != "pass":
        errors.append("reproduction report is not pass")
    if require_current_report:
        try:
            current_commit = _git(root, "rev-parse", "HEAD")
            current_tree = _git(root, "rev-parse", "HEAD^{tree}")
            errors.extend(
                _validate_reproduction_source(
                    root,
                    report.get("source"),
                    expected_commit=current_commit,
                    expected_tree=current_tree,
                )
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"cannot verify current Git revision for reproduction report: {exc}")

    aggregate_errors: list[str] = []
    aggregate_status, aggregate_reason = _canonical_run_artifact_state(root)
    recomputed: Optional[dict[str, Any]] = None
    if aggregate_status == "partial":
        aggregate_errors.append(aggregate_reason or "canonical run artifacts are partial")
    elif aggregate_status == "available":
        try:
            recomputed = _recompute_canonical_aggregates(root, loaded)
            for relative, payload in recomputed.items():
                if relative in loaded:
                    aggregate_errors.extend(
                        _validate_aggregate_matches(relative, loaded[relative], payload)
                    )
            # The transfer source is copied into the aggregate by the data
            # registry.  Bind it to the checked-in dataset provenance artifact so
            # a self-consistent but unrelated source block cannot pass.
            dataset_provenance = loaded.get(
                "Results/evidence/q1_dataset_provenance.json", {}
            )
            if not isinstance(dataset_provenance, dict):
                dataset_provenance = {}
            expected_transfer_source = next(
                (
                    item
                    for item in dataset_provenance.get("datasets", [])
                    if isinstance(item, dict) and item.get("dataset") == "fashion_mnist"
                ),
                None,
            )
            if expected_transfer_source is None or transfer.get("source") != expected_transfer_source:
                aggregate_errors.append(
                    "pooling-transfer source provenance disagrees with q1_dataset_provenance.json"
                )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
            aggregate_errors.append(f"canonical aggregate recomputation failed: {exc}")
    errors.extend(aggregate_errors)

    return {
        "status": "pass" if not errors else "fail",
        "required_files": len(EVIDENCE_REQUIRED),
        "errors": errors,
        "provenance": {
            "status": "pass" if not provenance_errors else "fail",
            "errors": provenance_errors,
        },
        "aggregate_recomputation": {
            "status": (
                aggregate_status
                if aggregate_status == "skipped"
                else "pass" if not aggregate_errors else "fail"
            ),
            "reason": aggregate_reason,
            "errors": aggregate_errors,
        },
    }


def _tail(value: str, limit: int = 4000) -> str:
    value = value or ""
    return value if len(value) <= limit else value[-limit:]


def _redact(value: str, root: Path) -> str:
    """Remove workstation-specific paths from the persisted release report."""

    text = str(value or "")
    replacements = {
        str(root.resolve()): "<repository>",
        str(root.resolve()).replace("\\", "/"): "<repository>",
        str(Path(tempfile.gettempdir())): "<temporary>",
        str(Path(tempfile.gettempdir())).replace("\\", "/"): "<temporary>",
        str(Path.home()): "<user-home>",
        str(Path.home()).replace("\\", "/"): "<user-home>",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = _ABSOLUTE_WINDOWS_PATH.sub("<absolute-path>", text)
    # MiKTeX may wrap an absolute path at the terminal width, splitting the
    # drive prefix from the remainder.  Remove the residual identity markers
    # as a final defense for persisted logs.
    if Path.home().name:
        text = text.replace(Path.home().name, "<user>")
    text = text.replace("C:", "<drive>").replace("c:", "<drive>")
    return text


def run_command(
    root: Path, label: str, command: Sequence[str], *, timeout: int = 3600
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "label": label,
        "command": [_redact(str(item), root) for item in command],
    }
    try:
        completed = subprocess.run(
            list(command),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        record.update(
            {
                "returncode": completed.returncode,
                "stdout_tail": _tail(_redact(completed.stdout, root)),
                "stderr_tail": _tail(_redact(completed.stderr, root)),
            }
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        record.update({"returncode": None, "error": _redact(str(exc), root)})
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record["status"] = "pass" if record.get("returncode") == 0 else "fail"
    return record


def pytest_counts(junit_path: Path) -> dict[str, Any]:
    if not junit_path.is_file():
        return {"status": "missing", "tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    elapsed = 0.0
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.attrib.get(key, "0"))
        elapsed += float(suite.attrib.get("time", "0"))
    counts["time_seconds"] = round(elapsed, 3)
    counts["status"] = (
        "pass" if counts["failures"] == 0 and counts["errors"] == 0 else "fail"
    )
    return counts


def scan_tracked_text_for_secrets(root: Path) -> dict[str, Any]:
    tracked = package.tracked_files(root)
    selected = [
        relative
        for relative in tracked
        if (root / relative).suffix.lower() in package.TEXT_SUFFIXES
    ]
    findings = package.scan_for_secrets(root, selected)
    return {"status": "pass" if not findings else "fail", "findings": findings}


def build_report(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    tracked_clean = not bool(_git(root, "status", "--porcelain=v1", "--untracked-files=no"))
    if not tracked_clean:
        raise RuntimeError("tracked files must be clean before the reproduction run")

    # The report is the output of this command, and the claim ledger is
    # refreshed after that output is committed.  Keep the preflight semantic
    # (run-artifact) checks, but defer persisted hash/current-report checks to
    # the strict package gate once this report exists.
    evidence = validate_evidence(
        root, verify_bindings=False, require_current_report=False
    )
    secret_scan = scan_tracked_text_for_secrets(root)
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fqcnn-q1-") as temporary:
        junit = Path(temporary) / "pytest.xml"
        checks.append(
            run_command(
                root,
                "full pytest suite",
                [
                    os.fspath(sys.executable),
                    "-m",
                    "pytest",
                    "tests",
                    "-q",
                    f"--junitxml={junit}",
                ],
                timeout=5400,
            )
        )
        pytest_result = pytest_counts(junit)

    ruff = shutil.which("ruff")
    if ruff:
        checks.append(
            run_command(
                root,
                "ruff release-source check",
                [ruff, "check", *RELEASE_LINT_PATHS],
                timeout=300,
            )
        )
    else:
        checks.append({"label": "ruff source check", "status": "skipped", "reason": "ruff not installed"})

    checks.append(
        run_command(
            root,
            "Python bytecode compilation",
            [
                os.fspath(sys.executable),
                "-m",
                "compileall",
                "-q",
                "QCNN",
                "baselines",
                "experiments",
                "scripts",
                "tests",
            ],
            timeout=300,
        )
    )

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        for pass_number in range(1, 4):
            checks.append(
                run_command(
                    root,
                    f"LaTeX build pass {pass_number}",
                    [
                        pdflatex,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-file-line-error",
                        "fqcnn.tex",
                    ],
                    timeout=300,
                )
            )
    else:
        checks.append({"label": "LaTeX build", "status": "skipped", "reason": "pdflatex not installed"})

    checks.append(
        run_command(
            root,
            "training environment dependency check",
            [os.fspath(sys.executable), "-m", "pip", "check"],
            timeout=300,
        )
    )

    qiskit_python = root / ".venv-qiskit" / "Scripts" / "python.exe"
    if qiskit_python.is_file():
        checks.append(
            run_command(
                root,
                "isolated Qiskit dependency check",
                [os.fspath(qiskit_python), "-m", "pip", "check"],
                timeout=300,
            )
        )
        checks.append(
            run_command(
                root,
                "isolated Qiskit schedule contract",
                [os.fspath(qiskit_python), "-c", QISKIT_SCHEDULE_CHECK],
                timeout=300,
            )
        )
    else:
        checks.append(
            {
                "label": "isolated Qiskit dependency check",
                "status": "skipped",
                "reason": "isolated Qiskit environment not present",
            }
        )
        checks.append(
            {
                "label": "isolated Qiskit schedule contract",
                "status": "skipped",
                "reason": "isolated Qiskit environment not present",
            }
        )

    checks_status = [item["status"] for item in checks]
    overall = (
        "pass"
        if evidence["status"] == "pass"
        and secret_scan["status"] == "pass"
        and all(status in {"pass", "skipped"} for status in checks_status)
        and pytest_result.get("status") == "pass"
        else "fail"
    )
    source_files = package.file_records(root, REPRODUCTION_SOURCE_PATHS)
    return {
        "schema": {"name": "fqcnn_q1_reproduction", "version": 1},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "source": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "branch": _git(root, "branch", "--show-current"),
            "tracked_tree_clean_before_run": tracked_clean,
            "source_files": source_files,
        },
        "scope": {
            "simulation_only": True,
            "real_qpu_job": False,
            "journal_submission": False,
            "dataset_downloads": "external upstream sources, checksummed in q1_dataset_provenance.json",
        },
        "evidence_validation": evidence,
        "secret_scan": secret_scan,
        "pytest": pytest_result,
        "checks": checks,
        "next_step": "Build the verified ZIP with scripts/build_q1_submission_package.py after committing this report.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("Results/evidence/q1_reproduction.json")
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_report(args.root, (args.root / args.output).resolve())
    output = (args.root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "pytest": report["pytest"]}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
