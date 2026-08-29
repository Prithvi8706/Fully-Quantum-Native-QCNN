"""Run the bounded Q1 release checks and write a machine-readable report.

The report is generated only after the checks complete. It records the exact Git
revision tested, command exit codes, pytest counts, and artifact-validation result;
it does not claim that a journal or a real quantum device was used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from scripts import build_q1_submission_package as package
except ImportError:  # direct ``python scripts/run_q1_reproduction.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import build_q1_submission_package as package


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
)

RELEASE_LINT_PATHS = (
    "QCNN/utils/dataset_registry.py",
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


def validate_evidence(root: Path) -> dict[str, Any]:
    """Validate required JSON shape and finite numeric values without rerunning data."""

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

    comparison = loaded.get("Results/evidence/q1_comparison.json", {})
    protocol = comparison.get("protocol", {})
    if protocol.get("arms") != ["proposed", "logistic", "mlp", "ttn"]:
        errors.append("comparison protocol arms differ from the frozen four-arm matrix")
    if protocol.get("seeds") != [0, 1, 2, 3, 4]:
        errors.append("comparison protocol does not contain seeds 0..4")
    if len(comparison.get("tasks", {})) != 4:
        errors.append("comparison aggregate does not contain four frozen tasks")

    transfer = loaded.get("Results/evidence/q1_pooling_transfer.json", {})
    if transfer.get("status") == "fail":
        errors.append("pooling-transfer evidence has status=fail")
    controls = loaded.get("Results/evidence/q1_pooling_controls.json", {})
    if controls.get("status") != "pass":
        errors.append("pooling-controls evidence is not pass")

    return {
        "status": "pass" if not errors else "fail",
        "required_files": len(EVIDENCE_REQUIRED),
        "errors": errors,
    }


def _tail(value: str, limit: int = 4000) -> str:
    value = value or ""
    return value if len(value) <= limit else value[-limit:]


def run_command(
    root: Path, label: str, command: Sequence[str], *, timeout: int = 3600
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "label": label,
        "command": [str(item) for item in command],
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
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            }
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        record.update({"returncode": None, "error": str(exc)})
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

    evidence = validate_evidence(root)
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
    return {
        "schema": {"name": "fqcnn_q1_reproduction", "version": 1},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "source": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "branch": _git(root, "branch", "--show-current"),
            "tracked_tree_clean_before_run": tracked_clean,
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
