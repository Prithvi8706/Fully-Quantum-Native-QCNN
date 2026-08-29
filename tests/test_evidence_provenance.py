from pathlib import Path

import pytest

from experiments import evidence_provenance


def test_binding_records_only_relative_hashed_inputs(tmp_path, monkeypatch):
    (tmp_path / "source.py").write_text("answer = 42\n")
    (tmp_path / "lock.txt").write_text("package==1.0\n")
    monkeypatch.setattr(
        evidence_provenance.metadata, "version", lambda package: f"{package}-version"
    )

    binding = evidence_provenance.build_binding(
        tmp_path,
        source_paths=("source.py",),
        lock_paths=("lock.txt",),
        packages=("package",),
    )

    assert binding["source_files"][0]["path"] == "source.py"
    assert len(binding["source_files"][0]["sha256"]) == 64
    assert binding["environment"]["lock_files"][0]["path"] == "lock.txt"
    assert binding["environment"]["packages"] == {"package": "package-version"}
    assert str(tmp_path.resolve()) not in str(binding)


def test_binding_fails_closed_for_missing_or_escaping_inputs(tmp_path):
    (tmp_path / "lock.txt").write_text("package==1.0\n")
    with pytest.raises(FileNotFoundError, match="missing provenance input"):
        evidence_provenance.build_binding(
            tmp_path,
            source_paths=("missing.py",),
            lock_paths=("lock.txt",),
            packages=(),
        )
    with pytest.raises(ValueError, match="escapes repository root"):
        evidence_provenance.build_binding(
            tmp_path,
            source_paths=(str(Path("..") / "outside.py"),),
            lock_paths=("lock.txt",),
            packages=(),
        )
