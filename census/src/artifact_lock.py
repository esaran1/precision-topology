"""Guard against two processes writing the same result artifact.

A concurrent write produced a mixed-schema CSV during the projection sweep.
That failure was loud -- the columns did not match and reading it raised -- but
the same race can silently interleave rows from two runs, which would be far
worse: the file would parse, and the numbers would be a blend of two
configurations with nothing to indicate it.

Writers take an exclusive lock naming the owning process.  A second writer
fails immediately with the identity of the first rather than corrupting the
output.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterator


class ArtifactLocked(RuntimeError):
    """Raised when another process already owns this artifact."""


@contextmanager
def artifact_lock(stem: Path, description: str = "") -> Iterator[Path]:
    """Hold an exclusive lock on ``stem`` for the duration of the block."""

    lock_path = stem.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "started": datetime.now(timezone.utc).isoformat(),
            "description": description,
        }
    )
    try:
        # O_EXCL makes creation atomic: exactly one process can succeed.
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _read_lock(lock_path)
        raise ArtifactLocked(
            f"{stem.name} is being written by pid {existing.get('pid', '?')} "
            f"since {existing.get('started', '?')}"
            f"{': ' + existing['description'] if existing.get('description') else ''}. "
            f"Remove {lock_path} if that process is gone."
        ) from None
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
        yield stem
    finally:
        lock_path.unlink(missing_ok=True)


def _read_lock(lock_path: Path) -> dict[str, object]:
    try:
        return json.loads(lock_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def stale_locks(directory: Path) -> list[Path]:
    """Lock files whose owning process is no longer running."""

    stale: list[Path] = []
    for lock_path in directory.glob("*.lock"):
        pid = _read_lock(lock_path).get("pid")
        if not isinstance(pid, int):
            stale.append(lock_path)
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            stale.append(lock_path)
        except PermissionError:
            pass
    return stale
