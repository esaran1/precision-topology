"""Memory gate for the U2 sample launcher, enforced in code (Track U, U2).

The gate is its OWN step: `python -m src.u2_gate gate "<label>"` measures memory_pressure free %, swap free and disk free,
appends one JSON line to the gate log, and exits non-zero on FAIL.  A launcher calls `require_gate()` first; it refuses to
start (raises GateRefused) unless the newest gate entry
  * exists and is OK,
  * was written by a DIFFERENT process (entry pid != os.getpid()) that had finished before this process started
    (entry time < this process's start time),
  * is fresh (younger than max_age_s), and
  * is newer than the last launch recorded in the launch log (one gate entry licenses one launch).
On acceptance the launch is appended to the launch log, so a second launch needs a new gate entry.

Thresholds (defaults): memory free >= 25 %, swap free >= 500 MB, disk free >= 20 GB (pass --disk-gb 25 while the
author's U2 disk hold is in force).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "results" / "uniform_eps" / "u2_sample"
GATE_LOG = DEFAULT_DIR / "gate_entries.jsonl"
LAUNCH_LOG = DEFAULT_DIR / "launches.jsonl"
MAX_AGE_S = 600.0


class GateRefused(RuntimeError):
    pass


def measure():
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    free = float(re.search(r"free percentage:\s*([0-9.]+)%", out).group(1))
    sw = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    swap_free = float(re.search(r"free = ([0-9.]+)M", sw).group(1))
    st = os.statvfs("/")
    disk_gb = st.f_bavail * st.f_frsize / 2 ** 30
    return {"mem_free_pct": free, "swap_free_mb": swap_free, "disk_free_gb": disk_gb}


def evaluate(m, mem_pct=25.0, swap_mb=500.0, disk_gb=20.0):
    return m["mem_free_pct"] >= mem_pct and m["swap_free_mb"] >= swap_mb and m["disk_free_gb"] >= disk_gb


def write_entry(label, m, ok, log=GATE_LOG, now=None, pid=None):
    e = {"time": time.time() if now is None else now, "pid": os.getpid() if pid is None else pid, "label": label,
         "ok": bool(ok), **m}
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(json.dumps(e) + "\n")
    return e


def _last(log):
    if not Path(log).exists():
        return None
    lines = [l for l in Path(log).read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def process_start_time():
    """Wall-clock start time of this process (ps lstart), falling back to import time."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
        return time.mktime(time.strptime(out.strip(), "%a %b %d %H:%M:%S %Y"))
    except Exception:
        return _IMPORT_TIME


_IMPORT_TIME = time.time()


def require_gate(label, gate_log=GATE_LOG, launch_log=LAUNCH_LOG, max_age_s=MAX_AGE_S, now=None, pid=None,
                 start_time=None):
    """Raise GateRefused unless a separate, earlier, fresh OK gate entry newer than the last launch exists."""
    now = time.time() if now is None else now
    pid = os.getpid() if pid is None else pid
    start = process_start_time() if start_time is None else start_time
    e = _last(gate_log)
    if e is None:
        raise GateRefused("no gate entry")
    if not e.get("ok"):
        raise GateRefused(f"last gate entry is FAIL ({e.get('label')})")
    if e["pid"] == pid:
        raise GateRefused("gate entry written by this process")
    if e["time"] >= start:
        raise GateRefused("gate entry not written before this process started")
    if now - e["time"] > max_age_s:
        raise GateRefused(f"gate entry stale ({now - e['time']:.0f} s > {max_age_s:.0f} s)")
    last_launch = _last(launch_log)
    if last_launch is not None and last_launch["time"] >= e["time"]:
        raise GateRefused("gate entry already used by an earlier launch")
    rec = {"time": now, "pid": pid, "label": label, "gate_time": e["time"], "gate_pid": e["pid"]}
    Path(launch_log).parent.mkdir(parents=True, exist_ok=True)
    with open(launch_log, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def main(argv):
    if argv[0] != "gate":
        raise SystemExit("usage: python -m src.u2_gate gate <label> [--disk-gb X]")
    disk = float(argv[argv.index("--disk-gb") + 1]) if "--disk-gb" in argv else 20.0
    label = argv[1] if len(argv) > 1 else ""
    m = measure()
    ok = evaluate(m, disk_gb=disk)
    write_entry(label, m, ok)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} gate '{label}' free={m['mem_free_pct']:.0f}% "
          f"swap_free={m['swap_free_mb']:.0f}M disk_free={m['disk_free_gb']:.1f}G (need {disk:g}G) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
