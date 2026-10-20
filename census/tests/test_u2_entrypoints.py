"""Every U2 compute entry point outside the test suite goes through the gate: run as a module without a separate,
earlier gate entry it refuses (exit 3) before doing any work; with one it gets past the gate."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = ["src.u2_sample", "src.u2_checker_timing", "src.u2_soundness"]


def _run(mod, args, gate_dir):
    env = {**os.environ, "U2_GATE_DIR": str(gate_dir)}
    return subprocess.run([sys.executable, "-m", mod, *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=120)


@pytest.mark.parametrize("mod", ENTRY_POINTS)
def test_refuses_without_a_gate_entry(tmp_path, mod):
    r = _run(mod, ["probes", "nonexistent.json", "x"] if mod == "src.u2_sample" else [str(tmp_path / "o.json")],
             tmp_path)
    assert r.returncode == 3 and "REFUSED" in r.stdout


@pytest.mark.parametrize("mod", ENTRY_POINTS)
def test_refuses_a_fail_entry(tmp_path, mod):
    from src import u2_gate as g
    g.write_entry("t", {"mem_free_pct": 10.0, "swap_free_mb": 100.0, "disk_free_gb": 1.0}, False,
                  log=tmp_path / "gate_entries.jsonl", now=time.time() - 5, pid=1)
    r = _run(mod, ["probes", "nonexistent.json", "x"] if mod == "src.u2_sample" else [str(tmp_path / "o.json")],
             tmp_path)
    assert r.returncode == 3 and "FAIL" in r.stdout


def test_passes_the_gate_with_a_separate_earlier_entry(tmp_path):
    """With a fresh OK entry from another process, the entry point gets past the gate (and then fails on the missing
    spec file -- i.e. it did not refuse)."""
    from src import u2_gate as g
    g.write_entry("t", {"mem_free_pct": 50.0, "swap_free_mb": 1000.0, "disk_free_gb": 30.0}, True,
                  log=tmp_path / "gate_entries.jsonl", now=time.time() - 5, pid=1)
    r = _run("src.u2_sample", ["probes", str(tmp_path / "missing.json"), "x"], tmp_path)
    assert r.returncode not in (0, 3) and "REFUSED" not in r.stdout
    launches = [json.loads(l) for l in (tmp_path / "launches.jsonl").read_text().splitlines()]
    assert launches and launches[-1]["gate_pid"] == 1


def test_every_u2_main_calls_the_gate():
    """Static check: each U2 module with a __main__ block calls require_gate before dispatching."""
    for f in sorted((ROOT / "src").glob("u2_*.py")):
        src = f.read_text()
        if '__name__ == "__main__"' not in src or f.name == "u2_gate.py":
            continue
        main = src.split('__name__ == "__main__"', 1)[1]
        assert "require_gate(" in main or ("require_gate(" in src and "main(" in main), f.name
