"""Build and verify the FQCNN Q1 submission archive from a clean Git commit.

The archive intentionally contains the current manuscript and PDF, every
figure referenced by the manuscript, the bounded Q1 evidence, the scientific
source and tests, environment locks, and submission/reviewer documentation.
Historical draft PDFs and unrelated generated outputs are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


SCHEMA_NAME = "fqcnn_q1_submission_package"
SCHEMA_VERSION = 1
MANIFEST_NAME = "SUBMISSION_PACKAGE_MANIFEST.json"

ROOT_FILES = {
    ".gitignore",
    "CITATION.cff",
    "CONTEXT.md",
    "DATASET_README.md",
    "README.md",
    "RUN_GUIDE.md",
    "fqcnn.tex",
    "main.py",
    "noise_sim.py",
    "pytest.ini",
    "quick_test.py",
    "reproduce.sh",
    "requirements-lock.txt",
    "requirements-qiskit-lock.txt",
    "requirements-qiskit.in",
    "requirements.txt",
    "runApp.bat",
    "runApp.sh",
    "setup_env.bat",
    "setup_env.ps1",
}

INCLUDED_PREFIXES = (
    "QCNN/",
    "baselines/",
    "experiments/",
    "scripts/",
    "tests/",
    "Results/evidence/",
    "Results/q1_comparison/",
    "Results/q1_pooling_transfer/",
    "docs/submission/",
)

DOCUMENTATION_FILES = {
    "docs/paper_code_reconciliation.md",
    "docs/q1-journal-shortlist.md",
    "docs/q1-reviewer-checklist.md",
    "docs/simulability_statement.md",
}

REQUIRED_FILES = {
    "fqcnn.tex",
    "fqcnn.pdf",
    "requirements-lock.txt",
    "requirements-qiskit-lock.txt",
    "Results/evidence/q1_claim_ledger.json",
    "Results/evidence/q1_comparison.json",
    "Results/evidence/q1_dataset_provenance.json",
    "Results/evidence/q1_fake_backend_rehearsal.json",
    "Results/evidence/q1_noise_validation.json",
    "Results/evidence/q1_pooling_controls.json",
    "Results/evidence/q1_pooling_practicality.json",
    "Results/evidence/q1_pooling_transfer.json",
    "Results/evidence/q1_reproduction.json",
    "Results/evidence/q1_resources.json",
    "docs/q1-journal-shortlist.md",
    "docs/q1-reviewer-checklist.md",
    "docs/submission/README.md",
    "docs/submission/cover-letter.md",
}

GRAPHICS_PATTERN = re.compile(
    r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}"
)

TEXT_SUFFIXES = {
    ".bat",
    ".cff",
    ".csv",
    ".in",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".tex",
    ".txt",
}

SECRET_PATTERNS = {
    "private_key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"
    ),
    "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "github_token": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{50,255})"
    ),
    # Require a token boundary so ordinary words such as ``task-by`` do not
    # create a false ``sk-`` match.
    "openai_api_key": re.compile(
        rb"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    ),
    "slack_token": re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "google_api_key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "stripe_live_key": re.compile(rb"[rs]k_live_[0-9A-Za-z]{16,}"),
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.rstrip("\r\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_path(value: str) -> PurePosixPath:
    """Return a normalized, relative archive path or reject it."""

    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {value!r}")
    if path.parts[0].endswith(":"):
        raise ValueError(f"unsafe archive path: {value!r}")
    return path


def referenced_graphics(tex_source: str) -> list[str]:
    """Return normalized manuscript graphics, adding an extension if needed."""

    graphics = []
    for raw in GRAPHICS_PATTERN.findall(tex_source):
        normalized = safe_member_path(raw.strip()).as_posix()
        if not PurePosixPath(normalized).suffix:
            normalized += ".png"
        graphics.append(normalized)
    return sorted(set(graphics))


def tracked_files(root: Path) -> set[str]:
    return {
        safe_member_path(line).as_posix()
        for line in _git(root, "ls-files").splitlines()
        if line
    }


def selected_files(root: Path) -> list[str]:
    tracked = tracked_files(root)
    selected = {
        path
        for path in tracked
        if path in ROOT_FILES
        or path in DOCUMENTATION_FILES
        or path.startswith(INCLUDED_PREFIXES)
    }

    tex_path = root / "fqcnn.tex"
    selected.update(
        referenced_graphics(tex_path.read_text(encoding="utf-8"))
    )
    selected.add("fqcnn.pdf")

    missing = sorted(path for path in REQUIRED_FILES if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(
            "required submission files are missing: " + ", ".join(missing)
        )

    untracked_selected = sorted(
        path for path in selected if path != "fqcnn.pdf" and path not in tracked
    )
    if untracked_selected:
        raise RuntimeError(
            "submission inputs must be tracked at HEAD: "
            + ", ".join(untracked_selected)
        )

    missing_selected = sorted(path for path in selected if not (root / path).is_file())
    if missing_selected:
        raise FileNotFoundError(
            "selected package files are missing: " + ", ".join(missing_selected)
        )
    return sorted(selected)


def assert_clean_tracked_tree(root: Path) -> None:
    state = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if state:
        raise RuntimeError(
            "tracked files differ from HEAD; commit or restore them before packaging:\n"
            + state
        )


def assert_fresh_pdf(root: Path) -> None:
    tex = root / "fqcnn.tex"
    pdf = root / "fqcnn.pdf"
    if not pdf.is_file():
        raise FileNotFoundError("fqcnn.pdf is missing; build the manuscript first")
    if pdf.stat().st_mtime_ns < tex.stat().st_mtime_ns:
        raise RuntimeError("fqcnn.pdf is older than fqcnn.tex; rebuild the manuscript")
    if not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("fqcnn.pdf does not have a valid PDF header")


def file_records(root: Path, paths: Iterable[str]) -> list[dict]:
    records = []
    for relative in sorted(paths):
        path = root / relative
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def scan_for_secrets(root: Path, paths: Iterable[str]) -> list[dict]:
    """Scan package text inputs for common credential signatures.

    Only finding type and path are returned; matched secret material is never
    echoed to the terminal or written to the manifest.
    """

    findings = []
    for relative in sorted(paths):
        path = root / relative
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_bytes()
        for finding_type, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append({"path": relative, "type": finding_type})
    return findings


def assert_release_evidence(root: Path) -> dict:
    """Run the semantic evidence gate before packaging any release bytes.

    File hashes in the ZIP manifest prove only that the archive was assembled
    consistently.  The reproduction gate additionally verifies the claim
    ledger bindings, tested source fingerprint, and both Q1 aggregates against
    their canonical run artifacts.  Import lazily to keep the two command-line
    modules usable independently and to avoid an import cycle.
    """

    # When this file is launched as ``python scripts/build_q1_submission_package.py``,
    # Python puts ``scripts/`` (rather than the repository root) first on
    # ``sys.path``.  On Windows that can resolve an unrelated installed
    # ``scripts`` package and make the release gate unavailable.  Ensure the
    # repository namespace wins while retaining the normal package import used
    # by tests and ``python -m`` callers.
    import sys

    root_text = str(root)
    scripts_text = str(root / "scripts")
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    loaded_scripts = sys.modules.get("scripts")
    loaded_paths = getattr(loaded_scripts, "__path__", ())
    if loaded_scripts is not None and not any(
        str(Path(path).resolve()) == scripts_text
        for path in loaded_paths
    ):
        sys.modules.pop("scripts", None)
    from scripts import run_q1_reproduction

    result = run_q1_reproduction.validate_evidence(
        root, verify_bindings=True, require_current_report=True
    )
    if result.get("status") != "pass":
        errors = result.get("errors", [])
        detail = "; ".join(str(error) for error in errors[:8])
        if len(errors) > 8:
            detail += f"; ... ({len(errors)} total errors)"
        raise RuntimeError(f"release evidence gate failed: {detail}")
    return result


def build_manifest(root: Path, records: list[dict]) -> dict:
    commit = _git(root, "rev-parse", "HEAD")
    commit_time = _git(root, "show", "-s", "--format=%cI", "HEAD")
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        # Bind the manifest timestamp to the source commit so that rebuilding
        # the archive from the same commit and inputs is byte-reproducible.
        "generated_at_utc": commit_time,
        "source": {
            "commit": commit,
            "tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "commit_time": commit_time,
            "branch": _git(root, "branch", "--show-current"),
            "tracked_tree_clean": True,
        },
        "scope": {
            "article": "fqcnn.tex",
            "rendered_article": "fqcnn.pdf",
            "simulation_only": True,
            "real_qpu_submission": False,
            "journal_submission_performed": False,
            "contents": [
                "manuscript and referenced figures",
                "Q1 evidence and per-cell comparison manifests/metrics",
                "scientific implementation and tests",
                "environment locks and reproduction instructions",
                "reviewer, venue, and submission-preparation documents",
            ],
            "excluded": [
                "historical draft PDFs",
                "local credentials and editor settings",
                "regenerable transcripts and LaTeX intermediates",
                "downloaded datasets and run-weight archives",
            ],
        },
        "integrity": {
            "algorithm": "SHA-256",
            "file_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records),
            "files": records,
        },
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def write_archive(
    root: Path,
    output: Path,
    package_root: str,
    records: list[dict],
    manifest: dict,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = safe_member_path(package_root).as_posix()
    with zipfile.ZipFile(output, "w", compresslevel=9) as archive:
        for record in records:
            relative = safe_member_path(record["path"]).as_posix()
            archive.writestr(
                _zip_info(f"{prefix}/{relative}"),
                (root / relative).read_bytes(),
            )
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        archive.writestr(
            _zip_info(f"{prefix}/{MANIFEST_NAME}"), manifest_bytes
        )


def verify_archive(output: Path, package_root: str, manifest: dict) -> None:
    prefix = safe_member_path(package_root).as_posix()
    expected = {
        f"{prefix}/{record['path']}": record for record in manifest["integrity"]["files"]
    }
    manifest_member = f"{prefix}/{MANIFEST_NAME}"

    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        for name in names:
            safe_member_path(name)
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate member names")
        if set(names) != set(expected) | {manifest_member}:
            raise RuntimeError("archive members do not match the package manifest")
        archived_manifest = json.loads(archive.read(manifest_member))
        if archived_manifest != manifest:
            raise RuntimeError("archived manifest differs from the generated manifest")
        for member, record in expected.items():
            content = archive.read(member)
            digest = hashlib.sha256(content).hexdigest()
            if digest != record["sha256"] or len(content) != record["bytes"]:
                raise RuntimeError(f"archive integrity mismatch: {record['path']}")


def build(root: Path, output_dir: Path) -> dict:
    root = root.resolve()
    assert_clean_tracked_tree(root)
    evidence_validation = assert_release_evidence(root)
    assert_fresh_pdf(root)
    paths = selected_files(root)
    secret_findings = scan_for_secrets(root, paths)
    if secret_findings:
        locations = ", ".join(
            f"{item['path']} ({item['type']})" for item in secret_findings
        )
        raise RuntimeError(f"potential credentials found in package inputs: {locations}")
    records = file_records(root, paths)
    manifest = build_manifest(root, records)
    short_commit = manifest["source"]["commit"][:12]
    package_root = f"fqcnn-q1-submission-{short_commit}"
    output = output_dir.resolve() / f"{package_root}.zip"
    write_archive(root, output, package_root, records, manifest)
    verify_archive(output, package_root, manifest)
    return {
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "source_commit": manifest["source"]["commit"],
        "source_tree": manifest["source"]["tree"],
        "file_count": manifest["integrity"]["file_count"],
        "evidence_validation": evidence_validation,
        "verified": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("submission_dist"))
    args = parser.parse_args(argv)
    result = build(args.root, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
