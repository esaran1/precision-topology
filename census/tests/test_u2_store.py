"""Unit-level resumability of the U2 sample jobs (src/u2_store.py, src/u2_sample.probe / switch_split): atomic unit
files, a restart skips completed units, an interrupted unit leaves nothing behind, a pause loses at most one unit."""

import json
import os

import pytest

from src import u2_store as us


def test_put_get_roundtrip_and_keys(tmp_path):
    st = us.UnitStore(tmp_path)
    st.put("s000_k001", {"certified": True, "x": 1.5})
    assert st.get("s000_k001") == {"certified": True, "x": 1.5}
    assert st.done("s000_k001") and not st.done("s000_k002")
    assert st.keys() == ["s000_k001"]


def test_interrupted_write_leaves_no_unit_and_no_temp(tmp_path, monkeypatch):
    st = us.UnitStore(tmp_path)

    def boom(a, b):
        raise KeyboardInterrupt
    monkeypatch.setattr(us.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        st.put("k", {"a": 1})
    monkeypatch.undo()
    assert not st.done("k")
    assert [p.name for p in tmp_path.iterdir()] == []


def test_stray_temp_and_corrupt_files_do_not_count(tmp_path):
    st = us.UnitStore(tmp_path)
    (tmp_path / ".tmp_abc.json").write_text('{"half": ')
    (tmp_path / "bad.json").write_text('{"truncated": ')
    assert st.keys() == ["bad"] and not st.done("bad")
    st.put("bad", {"ok": True})                                  # rewritten
    assert st.get("bad") == {"ok": True}


def test_restart_skips_completed_units_and_loses_at_most_one(tmp_path):
    st = us.UnitStore(tmp_path)
    calls = []

    def mk(i, fail_at=None):
        def fn():
            calls.append(i)
            if i == fail_at:
                raise KeyboardInterrupt                         # a pause/kill while unit i runs
            return {"i": i}
        return fn
    with pytest.raises(KeyboardInterrupt):
        us.run_units(st, [(f"u{i}", mk(i, fail_at=3)) for i in range(6)])
    assert calls == [0, 1, 2, 3] and st.keys() == ["u0", "u1", "u2"]      # only the running unit (3) is lost
    calls.clear()
    out = us.run_units(st, [(f"u{i}", mk(i)) for i in range(6)])
    assert calls == [3, 4, 5]                                    # completed units are not recomputed
    assert [out[f"u{i}"]["i"] for i in range(6)] == list(range(6))


def test_probe_is_unit_resumable(tmp_path, monkeypatch):
    from src import u2_sample as m
    calls = []
    monkeypatch.setattr(m, "tube", lambda ec, x, y: ([1.6, 1.3, 0.69], (0.66, 0.71)))

    def fake_unit(k, *a):
        calls.append(k)
        if k == 20 and len(calls) == 2:
            raise KeyboardInterrupt
        return {"k": k, "certified": True}
    monkeypatch.setattr(m, "_probe_unit", fake_unit)
    st = us.UnitStore(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        m.probe(0.0, 0.003, "R", [0.0, 0.5, 0.99], 40, None, None, {}, store=st, prefix="s007")
    assert st.keys() == ["s007_k000"]
    rows = m.probe(0.0, 0.003, "R", [0.0, 0.5, 0.99], 40, None, None, {}, store=st, prefix="s007")
    assert calls == [0, 20, 20, 39] and [r["k"] for r in rows] == [0, 20, 39]


def test_switch_split_is_unit_resumable(tmp_path, monkeypatch):
    from src import u2_sample as m
    calls = []

    def fake_switch(lo, hi, *a):                                # certifies sub-slabs of width <= 0.25e-3
        calls.append((lo, hi))
        return {"unique_sign_change": hi - lo <= 0.25e-3 + 1e-15, "n_boxes": 1}
    monkeypatch.setattr(m, "switch_unit", fake_switch)
    st = us.UnitStore(tmp_path)
    r1 = m.switch_split(0.0, 1e-3, (0.66, 0.71), None, None, None, store=st, prefix="sw")
    n1 = len(calls)
    assert r1["unique_sign_change_all"] and r1["n_subslabs"] == 4
    r2 = m.switch_split(0.0, 1e-3, (0.66, 0.71), None, None, None, store=st, prefix="sw")
    assert len(calls) == n1 and r2["n_subslabs"] == 4             # nothing recomputed on the rerun
