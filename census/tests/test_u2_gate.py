"""U2 launcher gate (src/u2_gate.py): the launcher must refuse without a separate, earlier, fresh OK gate entry newer
than the last launch, and accept exactly that.  Also the time-capped ε-slab splitting of src/u2_sample on constructed
cases (fake units), and a real outer unit that must stop within its time cap."""

import json
import time

import pytest

from src import u2_gate as g


def _entry(log, ok=True, t=None, pid=11111):
    g.write_entry("test", {"mem_free_pct": 50.0, "swap_free_mb": 1000.0, "disk_free_gb": 30.0}, ok, log=log,
                  now=t if t is not None else time.time() - 5, pid=pid)


def test_refuses_without_entry(tmp_path):
    with pytest.raises(g.GateRefused, match="no gate entry"):
        g.require_gate("x", tmp_path / "gate.jsonl", tmp_path / "launch.jsonl", start_time=time.time())


def test_refuses_fail_entry(tmp_path):
    _entry(tmp_path / "gate.jsonl", ok=False)
    with pytest.raises(g.GateRefused, match="FAIL"):
        g.require_gate("x", tmp_path / "gate.jsonl", tmp_path / "launch.jsonl", start_time=time.time())


def test_refuses_stale_entry(tmp_path):
    _entry(tmp_path / "gate.jsonl", t=time.time() - 3600)
    with pytest.raises(g.GateRefused, match="stale"):
        g.require_gate("x", tmp_path / "gate.jsonl", tmp_path / "launch.jsonl", max_age_s=600, start_time=time.time())


def test_refuses_entry_written_by_itself(tmp_path):
    import os
    _entry(tmp_path / "gate.jsonl", pid=os.getpid())
    with pytest.raises(g.GateRefused, match="this process"):
        g.require_gate("x", tmp_path / "gate.jsonl", tmp_path / "launch.jsonl", start_time=time.time())


def test_refuses_entry_written_after_start(tmp_path):
    """A gate written by another process but only after this process started (the 'gate && run' race) is refused."""
    t = time.time()
    _entry(tmp_path / "gate.jsonl", t=t)
    with pytest.raises(g.GateRefused, match="before this process started"):
        g.require_gate("x", tmp_path / "gate.jsonl", tmp_path / "launch.jsonl", start_time=t - 1)


def test_accepts_fresh_separate_ok_entry_once(tmp_path):
    gl, ll = tmp_path / "gate.jsonl", tmp_path / "launch.jsonl"
    _entry(gl)
    rec = g.require_gate("x", gl, ll, start_time=time.time())
    assert rec["gate_pid"] == 11111
    assert json.loads(ll.read_text().splitlines()[-1])["label"] == "x"
    with pytest.raises(g.GateRefused, match="already used"):          # one gate entry licenses one launch
        g.require_gate("y", gl, ll, start_time=time.time())
    time.sleep(0.01)
    _entry(gl, t=time.time())                                         # a new entry, after that launch, licenses the next
    g.require_gate("z", gl, ll, start_time=time.time() + 1)


def test_threshold_rule():
    m = {"mem_free_pct": 30.0, "swap_free_mb": 600.0, "disk_free_gb": 21.0}
    assert g.evaluate(m)
    assert not g.evaluate(m, disk_gb=25.0)
    assert not g.evaluate({**m, "mem_free_pct": 24.0})
    assert not g.evaluate({**m, "swap_free_mb": 499.0})


# ------------------------------------------------------------------ ε-slab splitting (constructed fake units)
def test_split_certifies_by_halving():
    from src.u2_sample import certify_split
    calls = []

    def fake(lo, hi, cap):                    # certifies only slabs of width <= 0.026
        calls.append((lo, hi))
        return {"certified": hi - lo <= 0.026 + 1e-15}
    out = certify_split(0.0, 0.1, fake, max_seconds=1.0, min_width=0.01)
    assert all(c for _, _, c, _ in out)
    assert [round(hi - lo, 6) for lo, hi, _, _ in out] == [0.025] * 4
    assert abs(out[0][0]) < 1e-15 and abs(out[-1][1] - 0.1) < 1e-15
    assert all(abs(out[i][1] - out[i + 1][0]) < 1e-15 for i in range(len(out) - 1))
    assert len(calls) == 7


def test_split_reports_failure_at_min_width():
    from src.u2_sample import certify_split

    def fake(lo, hi, cap):                    # a slab containing 0.0333 never certifies
        return {"certified": not (lo <= 0.0333 <= hi)}
    out = certify_split(0.0, 0.1, fake, max_seconds=1.0, min_width=0.01)
    bad = [(lo, hi) for lo, hi, c, _ in out if not c]
    assert len(bad) == 1 and bad[0][0] <= 0.0333 <= bad[0][1]
    assert 0.01 <= bad[0][1] - bad[0][0] < 0.02


def test_split_timeout_is_a_failure():
    from src.u2_sample import certify_split

    def fake(lo, hi, cap):                    # wide slabs 'time out' (return uncertified after the cap)
        t0 = time.time()
        if hi - lo > 0.03:
            while time.time() - t0 < cap:
                pass
            return {"certified": False, "timed_out": True}
        return {"certified": True}
    t0 = time.time()
    out = certify_split(0.0, 0.1, fake, max_seconds=0.05, min_width=0.01)
    assert all(c for _, _, c, _ in out) and time.time() - t0 < 2.0


def test_real_outer_unit_respects_time_cap():
    """A too-wide slab (ε ∈ [0.0462, 0.1462], original-coordinate domain; it fails at the 0.1 width) must return
    uncertified within about its cap, not after hours."""
    import json as _j
    from src import u2_sample as m
    from src.limit_bnb import population
    x, y = population()
    cons = _j.loads((m.RESULTS / "uniform_eps" / "constants.json").read_text())
    vs, J = m.tube(0.0962, x, y)
    w = (J[1] - J[0]) / 40
    c = m.branch_point(J[0] + 20.5 * w, 0.0962, x, y, start=(vs[0], vs[1]))
    t0 = time.time()
    r, _ = m.outer_unit(0.0462, 0.1462, J[0] + 20 * w, J[0] + 21 * w, tuple(c), x, y, "O", cons, record=False,
                        max_seconds=3.0)
    assert not r["certified"]
    assert time.time() - t0 < 3.0 + 10.0          # cap + one chunk (2,000 cells) + setup
