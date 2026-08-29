from pathlib import Path

from scripts import run_q1_reproduction as reproduction


def test_finite_validator_rejects_nan():
    assert not reproduction._finite(float("nan"))
    assert not reproduction._finite({"nested": [1.0, float("inf")]})


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
    assert "<temporary>" not in redacted  # arbitrary paths are redacted generically
    assert "<absolute-path>" in redacted
