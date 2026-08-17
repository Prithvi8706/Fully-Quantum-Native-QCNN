"""Record the Git repository state without modifying it."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence


LOCAL_CONFIG = {Path(".claude/settings.local.json")}
GENERATED_SUFFIXES = {".aux", ".log", ".pdf"}
RAW_LOG_SUFFIXES = ("_stdout.txt", "_log.txt")


def classify_untracked(path: Path) -> str:
    normalized = Path(path.as_posix())
    if normalized in LOCAL_CONFIG:
        return "local_config"
    if normalized.suffix.lower() in GENERATED_SUFFIXES:
        return "generated_output"
    if normalized.name.endswith(RAW_LOG_SUFFIXES):
        return "raw_log"
    return "unclassified"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def audit_repository(root: Path) -> dict:
    root = root.resolve()
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    upstream = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    remote_sha = _git(root, "rev-parse", upstream)
    behind, ahead = (
        int(value)
        for value in _git(root, "rev-list", "--left-right", "--count", f"{upstream}...HEAD").split()
    )

    remotes = []
    for line in _git(root, "remote", "-v").splitlines():
        name, url, operation = line.split()
        remotes.append(
            {"name": name, "url": url, "operation": operation.strip("()")}
        )

    tracked = []
    untracked = []
    for line in _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines():
        status = line[:2]
        path = Path(line[3:])
        if status == "??":
            untracked.append(
                {"path": path.as_posix(), "classification": classify_untracked(path)}
            )
        else:
            tracked.append({"path": path.as_posix(), "status": status})

    return {
        "branch": branch,
        "upstream": upstream,
        "head": head,
        "remote_sha": remote_sha,
        "remotes": remotes,
        "divergence": {"behind": behind, "ahead": ahead},
        "tracked_state": tracked,
        "untracked_paths": untracked,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    audit = audit_repository(Path.cwd())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
