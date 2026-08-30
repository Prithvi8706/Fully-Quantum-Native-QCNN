"""Owner-safe exclusive queue leases."""
from __future__ import annotations

import contextlib
import json
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


@contextlib.contextmanager
def _operation_guard(path, wait=False):
    guard_path = path.with_name(path.name + ".operation.lock")
    while True:
        try:
            fd = os.open(str(guard_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            if not wait:
                raise
            time.sleep(0.01)
        except PermissionError as exc:
            # Windows may report a sharing violation rather than
            # ``FileExistsError`` while another worker owns the guard.  Treat
            # an existing guard as contention; a missing guard still indicates
            # a genuine permission problem and must not be hidden.
            if not wait:
                if guard_path.exists():
                    raise FileExistsError(str(guard_path)) from exc
                raise
            if not guard_path.exists():
                raise
            time.sleep(0.01)
    try:
        yield
    finally:
        os.close(fd)
        guard_path.unlink()


def acquire(path, metadata: dict) -> dict:
    """Atomically acquire and persist an owner record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = dict(metadata)
    owner.update(
        lease_id=uuid.uuid4().hex,
        pid=os.getpid(),
        host=platform.node(),
        acquired_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    with _operation_guard(path):
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except PermissionError as exc:
            # The lease already exists but is open by another Windows worker;
            # expose the same fail-fast contention contract as O_EXCL on POSIX.
            if path.exists():
                raise FileExistsError(str(path)) from exc
            raise
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(owner, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
        except Exception:
            with contextlib.suppress(OSError):
                path.unlink()
            raise
    return owner


def read(path) -> dict:
    """Read and validate a lease; raise ValueError if malformed."""
    with open(path) as handle:
        owner = json.load(handle)
    if not isinstance(owner, dict):
        raise ValueError("queue lease must be an object")
    if not isinstance(owner.get("lease_id"), str) or not owner["lease_id"]:
        raise ValueError("queue lease has invalid lease_id")
    if type(owner.get("pid")) is not int or owner["pid"] <= 0:
        raise ValueError("queue lease has invalid pid")
    if not isinstance(owner.get("host"), str) or not owner["host"]:
        raise ValueError("queue lease has invalid host")
    metadata = [owner.get("campaign"), owner.get("kind")]
    if not any(isinstance(value, str) and value for value in metadata):
        raise ValueError("queue lease has invalid campaign/kind metadata")
    if any(value is not None and (not isinstance(value, str) or not value) for value in metadata):
        raise ValueError("queue lease has invalid campaign/kind metadata")
    timestamp = owner.get("acquired_at_utc")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("queue lease has invalid timestamp")
    try:
        acquired_at = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("queue lease has invalid timestamp") from exc
    if acquired_at.tzinfo is None:
        raise ValueError("queue lease has invalid timestamp")
    return owner


def is_owned(path, owner: dict) -> bool:
    """Return whether the current lease matches the supplied owner."""
    if not isinstance(owner, dict):
        return False
    try:
        current = read(path)
        return all(
            current.get(key) == owner.get(key)
            for key in ("lease_id", "campaign", "host", "pid")
        )
    except (OSError, TypeError, ValueError):
        return False


def release(path, owner: dict) -> bool:
    """Remove only the exact owner's lease."""
    path = Path(path)
    with _operation_guard(path, wait=True):
        if not is_owned(path, owner):
            return False
        try:
            path.unlink()
        except OSError:
            return False
        return True
