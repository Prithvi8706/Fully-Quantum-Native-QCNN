import copy
from pathlib import Path

import pytest

from scripts import build_q1_submission_package as package
from scripts import run_q1_reproduction as reproduction


def test_finite_validator_rejects_nan():
    assert not reproduction._finite(float("nan"))
    assert not reproduction._finite({"nested": [1.0, float("inf")]})


def test_aggregate_validator_rejects_shape_preserving_metric_mutation():
    recomputed = {
        "generated_at_utc": "2026-01-01T00:00:00Z",
        "tasks": {"mnist_0v1": {"accuracy": {"mean": 0.75}}},
    }
    stored = copy.deepcopy(recomputed)
    # Preserve the JSON structure and all protocol keys while changing only a
    # reported number.  A shape-only validator must not accept this payload.
    stored["tasks"]["mnist_0v1"]["accuracy"]["mean"] = 0.95

    errors = reproduction._validate_aggregate_matches(
        "Results/evidence/q1_comparison.json", stored, recomputed
    )

    assert errors == [
        "Results/evidence/q1_comparison.json differs from canonical "
        "run-artifact recomputation at $.tasks.mnist_0v1.accuracy.mean"
    ]


def test_reproduction_source_fingerprint_rejects_provenance_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_path = tmp_path / "source.py"
    source_path.write_text("answer = 1\n", encoding="utf-8")
    record = package.file_records(tmp_path, ["source.py"])[0]
    monkeypatch.setattr(reproduction, "REPRODUCTION_SOURCE_PATHS", ("source.py",))

    source_path.write_text("answer = 2\n", encoding="utf-8")
    errors = reproduction._validate_reproduction_source(
        tmp_path,
        {
            "commit": "tested-commit",
            "tree": "tested-tree",
            "tracked_tree_clean_before_run": True,
            "source_files": [record],
        },
    )

    assert errors == ["reproduction source hash mismatch: source.py"]


def test_package_gate_blocks_invalid_semantic_evidence(monkeypatch, tmp_path: Path):
    def invalid_gate(*args, **kwargs):
        return {"status": "fail", "errors": ["aggregate metric mismatch"]}

    monkeypatch.setattr(reproduction, "validate_evidence", invalid_gate)

    with pytest.raises(RuntimeError, match="release evidence gate failed"):
        package.assert_release_evidence(tmp_path)


def test_missing_run_roots_are_explicitly_skipped_for_packaged_evidence(tmp_path: Path):
    state, reason = reproduction._canonical_run_artifact_state(tmp_path)

    assert state == "skipped"
    assert reason == (
        "canonical run artifacts are absent; relying on committed evidence and ledger hashes"
    )


def test_partial_run_roots_fail_closed_instead_of_skipping(tmp_path: Path):
    (tmp_path / "Results" / "q1_comparison" / "runs").mkdir(parents=True)

    state, reason = reproduction._canonical_run_artifact_state(tmp_path)

    assert state == "partial"
    assert reason == (
        "exactly one canonical run-artifact root is present; refusing to skip recomputation"
    )


def test_pytest_counts_reads_junit_summary(tmp_path: Path):
    report = tmp_path / "pytest.xml"
    report.write_text(
        '<testsuites><testsuite tests="7" failures="0" errors="0" skipped="2" time="1.25" /></testsuites>',
        encoding="utf-8",
    )
    counts = reproduction.pytest_counts(report)
    assert counts == {
        "tests": 7,
        "failures": 0,
        "errors": 0,
        "skipped": 2,
        "time_seconds": 1.25,
        "status": "pass",
    }


def test_report_redaction_removes_absolute_workstation_paths(tmp_path: Path):
    raw = r'C:\Users\author\AppData\Local\Temp\run\pytest.xml C:\Program Files\MiKTeX\bin'
    escaped_raw = raw.replace("\\", "\\\\")
    redacted = reproduction._redact(raw + " " + escaped_raw, tmp_path)
    assert "C:\\" not in redacted
    assert "prith" not in redacted
    assert "<temporary>" not in redacted  # arbitrary paths are redacted generically
    assert "<absolute-path>" in redacted
