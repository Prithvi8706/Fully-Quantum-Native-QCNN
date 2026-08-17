from pathlib import Path

README = Path("README.md").read_text(encoding="utf-8")
STATUS = Path("STATUS.md").read_text(encoding="utf-8")


def test_readme_names_frozen_python_version():
    assert "Python 3.9.13" in README
    assert "Python 3.14" not in README


def test_readme_labels_historical_noise():
    assert "historical split" in README.lower()
    assert "not manuscript evidence" in README.lower()


def test_status_records_completed_pooling_campaign():
    assert "75" in STATUS
    assert "t5_pooling_arms.json" in STATUS


def test_status_no_longer_calls_m1_the_active_gate():
    assert "M1 active gate" not in STATUS


def test_status_tables_use_declared_state_labels():
    allowed = {"pending", "running", "failed", "complete"}
    milestone_table = STATUS.split("## 1. Milestone state", 1)[1].split(
        "## 2. Active queue and blockers", 1
    )[0]
    blocker_table = STATUS.split("| # | Blocker | Severity | Owner milestone | Status |", 1)[1].split(
        "## 3. M0 outcome", 1
    )[0]

    milestone_states = {
        row.split("|")[2].strip().strip("*`")
        for row in milestone_table.splitlines()
        if row.startswith("| ") and "Milestone" not in row
    }
    blocker_states = {
        row.split("|")[5].strip().strip("*`")
        for row in blocker_table.splitlines()
        if row.startswith("| B")
    }

    assert milestone_states <= allowed
    assert blocker_states <= allowed
