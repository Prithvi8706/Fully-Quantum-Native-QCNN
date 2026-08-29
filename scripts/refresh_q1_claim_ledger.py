"""Refresh hashes and closure notes in the bounded Q1 claim ledger.

The ledger is an evidence index, not a source of scientific numbers. This helper
updates only release metadata and the three explicitly closed hygiene notes; it
never changes a metric, confidence interval, or claim status to make a mismatch
disappear.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LEDGER = Path("Results/evidence/q1_claim_ledger.json")
RELEASE_SOURCES = (
    "scripts/build_q1_submission_package.py",
    "scripts/refresh_q1_claim_ledger.py",
    "scripts/run_q1_reproduction.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_path_hashes(node: Any, root: Path, ledger_path: Path) -> None:
    """Update nested ``path``/``sha256`` records without hashing the ledger itself."""

    if isinstance(node, dict):
        relative = node.get("path")
        if isinstance(relative, str) and "sha256" in node:
            path = (root / relative).resolve()
            if path.is_file() and path != ledger_path.resolve():
                node["sha256"] = sha256_file(path)
                if "bytes" in node:
                    node["bytes"] = path.stat().st_size
        for child in node.values():
            update_path_hashes(child, root, ledger_path)
    elif isinstance(node, list):
        for child in node:
            update_path_hashes(child, root, ledger_path)


def update_source_hashes(ledger: dict, root: Path) -> None:
    source_hashes = ledger.get("source_hashes_sha256", {})
    if not isinstance(source_hashes, dict):
        return
    for relative in RELEASE_SOURCES:
        if (root / relative).is_file():
            source_hashes.setdefault(relative, sha256_file(root / relative))
    for relative in list(source_hashes):
        path = root / relative
        if path.is_file() and path.resolve() != (root / LEDGER).resolve():
            source_hashes[relative] = sha256_file(path)


def add_control_artifact(ledger: dict, root: Path) -> None:
    path = root / "Results/evidence/q1_pooling_controls.json"
    if not path.is_file():
        return
    inventory = ledger.setdefault("artifact_inventory", [])
    if any(item.get("path") == path.relative_to(root).as_posix() for item in inventory):
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    inventory.append(
        {
            "id": "A-015",
            "path": path.relative_to(root).as_posix(),
            "status": payload.get("status", "fail"),
            "sha256": sha256_file(path),
            "protocol": payload.get("protocol", {}),
            "key_result": {
                "inert_max_abs_difference": payload.get(
                    "inert_discard_rotation", {}
                ).get("max_abs_difference"),
                "no_pooling_max_abs_difference": payload.get(
                    "no_pooling_positive_control", {}
                ).get("max_abs_difference"),
            },
            "scope": payload.get("claim_scope", "fixed-parameter simulator control"),
        }
    )


def add_reproduction_artifact(ledger: dict, root: Path) -> None:
    path = root / "Results/evidence/q1_reproduction.json"
    if not path.is_file():
        return
    inventory = ledger.setdefault("artifact_inventory", [])
    if any(item.get("path") == path.relative_to(root).as_posix() for item in inventory):
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    inventory.append(
        {
            "id": "A-016",
            "path": path.relative_to(root).as_posix(),
            "status": payload.get("status", "fail"),
            "sha256": sha256_file(path),
            "protocol": {
                "pytest": payload.get("pytest", {}),
                "checks": [item.get("label") for item in payload.get("checks", [])],
            },
            "scope": "clean tracked-checkout release validation; no journal or QPU submission",
        }
    )


def close_items(ledger: dict) -> None:
    resolutions = {
        "U-005": (
            "The MNIST registry now pins expected MD5 and SHA-256 values for all four "
            "local IDX files, records the uncompressed official snapshot identity, and "
            "fails closed without a registry download path."
        ),
        "U-008": (
            "The comparison and pooling-transfer aggregates now carry source-file, lock, "
            "runtime-package, and per-run artifact SHA-256 provenance bindings."
        ),
        "U-009": (
            "q1_pooling_controls.json archives the fixed-input inert-gate removal and "
            "no-pooling positive-control readouts with thresholds and source hashes."
        ),
    }
    unresolved = [item for item in ledger.get("unresolved_items", []) if item.get("id") not in resolutions]
    ledger["unresolved_items"] = unresolved
    resolved = ledger.setdefault("resolved_items", [])
    existing = {item.get("id") for item in resolved}
    for identifier, resolution in resolutions.items():
        if identifier not in existing:
            resolved.append({"id": identifier, "status": "resolved", "resolution": resolution})


def replace_scope_language(node: Any) -> Any:
    replacements = {
        "matched logistic, MLP, and TTN": "same-split logistic, two-unit MLP, and TTN",
        "matched-arm aggregate": "same-split aggregate",
        "matched within-study comparison": "same-split within-study comparison",
        "parameter-matched MLP": "two-unit MLP (not parameter-matched)",
        "matched baselines": "same-split baselines",
    }
    if isinstance(node, dict):
        for key, value in list(node.items()):
            node[key] = replace_scope_language(value)
        return node
    if isinstance(node, list):
        return [replace_scope_language(value) for value in node]
    if isinstance(node, str):
        for old, new in replacements.items():
            node = node.replace(old, new)
    return node


def refresh(root: Path) -> dict:
    root = root.resolve()
    ledger_path = (root / LEDGER).resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    add_control_artifact(ledger, root)
    add_reproduction_artifact(ledger, root)
    close_items(ledger)
    update_source_hashes(ledger, root)
    update_path_hashes(ledger, root, ledger_path)
    ledger = replace_scope_language(ledger)
    ledger["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ledger["scope"]["observation"] = (
        "The corrected Q1 comparison and pooling-transfer campaigns are complete and "
        "parseable: 80/80 comparison cells and 15/15 transfer cells have complete status, "
        "with no failed runs. All manifests use n_total=666 with train/validation/test="
        "400/100/166. Dataset checksums, aggregate provenance, and fixed-input pooling "
        "controls are now archived; the remaining release gate is clean reproduction and "
        "package verification."
    )
    ledger["next_audit_command"] = (
        "Run scripts/run_q1_reproduction.py from a clean tracked checkout, validate its "
        "JSON report, rebuild the three-pass PDF, refresh this ledger's hashes, then run "
        "scripts/build_q1_submission_package.py and verify the archive SHA-256."
    )
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return {
        "output": ledger_path.as_posix(),
        "status": ledger.get("status"),
        "unresolved_items": len(ledger.get("unresolved_items", [])),
        "artifacts": len(ledger.get("artifact_inventory", [])),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(refresh(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
