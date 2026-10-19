"""Unit-level resumability for the U2 sample jobs (Track U, U2).

Each unit's result is one JSON file written ATOMICALLY: written to a temporary file in the same directory, flushed and
fsync'ed, then os.replace'd onto its final name. A unit is complete iff its final file exists and parses. A restart
skips completed units, so a pause or kill loses at most the unit that was running. Stray temporary files are ignored.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


def _safe(key):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", key)


class UnitStore:
    def __init__(self, root, extra=None):
        """extra: fields merged into every stored record (e.g. the check status label)."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.extra = dict(extra or {})

    def path(self, key):
        return self.root / f"{_safe(key)}.json"

    def get(self, key):
        p = self.path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None                                  # an unreadable file counts as not done (it is rewritten)

    def done(self, key):
        return self.get(key) is not None

    def put(self, key, rec):
        p = self.path(key)
        if self.extra and isinstance(rec, dict):
            rec = {**rec, **self.extra}
        fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=self.root)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(rec, default=float))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return p

    def keys(self):
        return sorted(p.stem for p in self.root.glob("*.json") if not p.name.startswith(".tmp_"))


def run_units(store, units, log=None):
    """units: iterable of (key, fn) with fn() -> dict.  Completed keys are skipped (their stored result is returned);
    each new result is stored atomically before the next unit starts.  Returns {key: result}."""
    out = {}
    for key, fn in units:
        r = store.get(key)
        if r is None:
            r = fn()
            store.put(key, r)
            if log:
                log(key, r, False)
        elif log:
            log(key, r, True)
        out[key] = r
    return out
