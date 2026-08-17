from pathlib import Path
from types import SimpleNamespace

from scripts.audit_repository_state import audit_repository, classify_untracked


def test_local_configuration_is_never_evidence():
    assert classify_untracked(Path(".claude/settings.local.json")) == "local_config"


def test_generated_latex_files_are_never_evidence():
    for name in ("fqcnn.aux", "fqcnn.log", "fqcnn.pdf"):
        assert classify_untracked(Path(name)) == "generated_output"


def test_raw_stdout_is_local_log():
    assert classify_untracked(Path("Results/phase3_expr_stdout.txt")) == "raw_log"


def test_audit_repository_parses_git_state(monkeypatch, tmp_path):
    outputs = {
        ("branch", "--show-current"): "plan/fqcnn-q1-upgrade\n",
        ("rev-parse", "HEAD"): "local-sha\n",
        (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ): "origin/plan/fqcnn-q1-upgrade\n",
        ("rev-parse", "origin/plan/fqcnn-q1-upgrade"): "remote-sha\n",
        (
            "rev-list",
            "--left-right",
            "--count",
            "origin/plan/fqcnn-q1-upgrade...HEAD",
        ): "2\t5\n",
        (
            "remote",
            "-v",
        ): "origin https://example.test/repository.git (fetch)\norigin https://example.test/repository.git (push)\n",
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ): " M STATUS.md\n?? .claude/settings.local.json\n?? Results/run_stdout.txt\n?? fqcnn.pdf\n?? notes.txt\n",
    }
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=outputs[tuple(command[1:])])

    monkeypatch.setattr("scripts.audit_repository_state.subprocess.run", fake_run)

    audit = audit_repository(tmp_path)

    assert audit == {
        "branch": "plan/fqcnn-q1-upgrade",
        "upstream": "origin/plan/fqcnn-q1-upgrade",
        "head": "local-sha",
        "remote_sha": "remote-sha",
        "remotes": [
            {
                "name": "origin",
                "url": "https://example.test/repository.git",
                "operation": "fetch",
            },
            {
                "name": "origin",
                "url": "https://example.test/repository.git",
                "operation": "push",
            },
        ],
        "divergence": {"behind": 2, "ahead": 5},
        "tracked_state": [{"path": "STATUS.md", "status": " M"}],
        "untracked_paths": [
            {
                "path": ".claude/settings.local.json",
                "classification": "local_config",
            },
            {"path": "Results/run_stdout.txt", "classification": "raw_log"},
            {"path": "fqcnn.pdf", "classification": "generated_output"},
            {"path": "notes.txt", "classification": "unclassified"},
        ],
    }
    assert all(command[0] == "git" for command, _ in calls)
    assert all(kwargs["cwd"] == tmp_path.resolve() for _, kwargs in calls)
    assert all(
        kwargs["check"] and kwargs["text"] and kwargs["capture_output"]
        for _, kwargs in calls
    )
