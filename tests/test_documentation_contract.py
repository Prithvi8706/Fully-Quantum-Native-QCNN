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
