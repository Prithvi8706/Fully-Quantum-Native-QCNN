"""Small, fail-closed provenance bindings for release evidence artifacts.

The scientific artifacts already hash every run input they consume.  This
module adds the other half of the release identity: the source files that
interpret those inputs and the immutable environment lock for that source.
Only repository-relative paths and SHA-256 digests are recorded; workstation
paths and credentials are deliberately excluded.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


SCHEMA = {"name": "fqcnn_evidence_provenance", "version": 1}
DEFAULT_PACKAGES = (
    "numpy",
    "pennylane",
    "pennylane-lightning",
    "scikit-learn",
    "scipy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(root: Path, relative_path: str) -> Dict[str, Any]:
    root = root.resolve()
    path = (root / relative_path).resolve()
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"provenance path escapes repository root: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"missing provenance input: {normalized}")
    return {
        "path": normalized,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _versions(packages: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required provenance package is not installed: {package}"
            ) from exc
    return versions


def build_binding(
    root: Path,
    *,
    source_paths: Sequence[str],
    lock_paths: Sequence[str] = ("requirements-lock.txt",),
    packages: Sequence[str] = DEFAULT_PACKAGES,
) -> Dict[str, Any]:
    """Bind an artifact to exact source, lock, and runtime package identities."""
    if not source_paths:
        raise ValueError("at least one source path is required for provenance")
    if not lock_paths:
        raise ValueError("at least one environment lock is required for provenance")
    return {
        "schema": SCHEMA,
        "hash_algorithm": "sha256",
        "source_files": [_record(root, path) for path in source_paths],
        "environment": {
            "python": platform.python_version(),
            "packages": _versions(packages),
            "lock_files": [_record(root, path) for path in lock_paths],
        },
    }
