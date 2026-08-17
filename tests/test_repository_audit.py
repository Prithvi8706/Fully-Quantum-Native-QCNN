from pathlib import Path

from scripts.audit_repository_state import classify_untracked


def test_local_configuration_is_never_evidence():
    assert classify_untracked(Path(".claude/settings.local.json")) == "local_config"


def test_generated_latex_files_are_never_evidence():
    for name in ("fqcnn.aux", "fqcnn.log", "fqcnn.pdf"):
        assert classify_untracked(Path(name)) == "generated_output"


def test_raw_stdout_is_local_log():
    assert classify_untracked(Path("Results/phase3_expr_stdout.txt")) == "raw_log"
