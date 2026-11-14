"""POST HOC forecast-horizon analysis (src/posthoc_horizon.py): constructed pass / fail / unresolved cases, and the full
path from the data table to the numbers on realistic rows (the committed forecasts fed back as the 'recomputed' ones
must reproduce the committed verdicts; perturbed rows must change them)."""
import copy
import json

import numpy as np
import pytest

from src import posthoc_horizon as H
from src import trackA_causal as T
from src import phase1c as P1C
from src import phase1b as P1B
from src import u2_gate


def test_grid():
    assert H.FGRID == (0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98)
    assert all(H._fk(f) in {H._fk(g) for g in H.FGRID} for f in (0.9, 0.95))
    assert H._fk(0.9) == "0.9" and H._fk(0.95) == "0.95"


def test_summary_row_numbers():
    crit = {"C1": {"n_within": 8, "frac_within": 0.8, "tau": 10}, "C2": {"n_within": 6, "frac_within": 0.6, "tau": 15},
            "C3": {"median_ratio": 1.1}}
    r = H.summary_row("W1", 0.9, crit, 12, 11, 10, 2, 12, 0, [3, -4, 5, 20, 1, 0, 2, -2], [10] * 8, True, "FAIL C2",
                      0.95, 100.0)
    assert r["f_is_used"] and r["n_forecast"] == 8 and r["n_miss"] == 2
    assert r["abs_err_cross_q50"] == 2.5 and r["abs_err_cross_in_lags_q50"] == 0.25
    assert r["label"] == "width-1 sine, SGD, a = 1.58"
    assert not H.summary_row("G", 0.9, crit, 1, 1, 1, 0, 1, 0, [1], [1], True, "PASS", 1, 1)["f_is_used"]


def test_wilson():
    lo, hi = H.wilson(93, 98)
    assert 0.88 < lo < 0.949 < hi < 0.98
    assert all(np.isnan(H.wilson(0, 0)))


# ---------------------------------------------------------------- full path on realistic rows: Track A, a = 1.77
def _a177_mine(shift=0, nan_bad_seed=None):
    ob = {o["seed"]: o for o in T._rows(T.OUT / "observed.jsonl")}
    mine = {}
    for r in T._rows(T.OUT / "runs.jsonl"):
        if r.get("status") != "ok":
            mine[r["seed"]] = {"status": r.get("status")}
            continue
        fc = copy.deepcopy(r["fc"])
        p = fc["0.9"].get("primary")
        if p is not None:
            if shift and p.get("t_fc") is not None:
                p["t_fc"] += shift
            if r["seed"] == nan_bad_seed:
                p["nan_recompute_identical"] = False
        t_sw = {k: (ob.get(r["seed"], {}).get("by_f") or {}).get(k, {}).get("t_sw") for k in fc}
        mine[r["seed"]] = {"status": "ok", "fc": fc, "t_sw_own_f": t_sw}
    return mine


def test_a177_full_path_reproduces_the_registered_pass():
    row, res = H.score_a177(_a177_mine(), 0.9)
    assert res["outcome"] == "PASS" and row["C1_n_within"] == 93 and row["n_scored"] == 98
    assert row["n_miss"] == 3 and row["nan_recompute_differs"] == 0
    assert all(H.registered_check("A177", res).values())


def test_a177_full_path_fail_and_unresolved():
    row, res = H.score_a177(_a177_mine(shift=25), 0.9)   # +25 > tau 10 + the largest |error| inside it
    assert res["C1"]["verdict"] == "FAIL" and not H.registered_check("A177", res)["C1"]
    seed = next(r["seed"] for r in T._rows(T.OUT / "runs.jsonl") if (r.get("fc") or {}).get("0.9", {}).get("primary"))
    row, res = H.score_a177(_a177_mine(nan_bad_seed=seed), 0.9)
    assert res["outcome"].startswith("UNRESOLVED") and row["nan_recompute_differs"] == 1


# ---------------------------------------------------------------- full path on realistic rows: 1C W1
def _w1_mine(shift=0):
    mine = {}
    for r in P1C._rows(P1C._runs_file("W1")):
        if r.get("status") != "ok" or not r.get("path"):
            continue
        fc = {"0.9": copy.deepcopy(r["forecast"]), "0.95": copy.deepcopy(r["forecast_other_f"])}
        if shift and fc["0.9"].get("t_fc") is not None:
            fc["0.9"]["t_fc"] += shift
        mine[r["seed"]] = {"status": "ok", "fc": fc}
    return mine


def test_w1_full_path_reproduces_and_fails_when_perturbed():
    row, res = H.score_1c("W1", _w1_mine(), 0.9)
    assert res["outcome"] == "PASS" and all(H.registered_check("W1", res).values())
    assert row["n_scored"] == 84 and row["C1_n_within"] == 84
    row, res = H.score_1c("W1", _w1_mine(shift=25), 0.9)   # +11 left most errors (median -4) inside tau 10
    assert res["C1"]["verdict"] == "FAIL" and not H.registered_check("W1", res)["C1"]


def test_w1_missing_row_refuses():
    m = _w1_mine()
    m.pop(next(iter(m)))
    with pytest.raises(AssertionError, match="missing horizon row"):
        H.score_1c("W1", m, 0.9)


# ---------------------------------------------------------------- full path on realistic rows: a = 1.65 (Phase 1B)
def _a165_mine(key, shift=0):
    arm_id = H.P1B_ARM[key]
    mine = {}
    for r in P1B._rows(P1B.OUT / "runs_track_a.jsonl"):
        if r["arm_id"] != arm_id:
            continue
        fc = copy.deepcopy(r["fc"])
        if shift and fc["0.9"].get("t_fc") is not None:
            fc["0.9"]["t_fc"] += shift
        mine[r["seed"]] = {"status": "ok", "fc": fc, "s_run_cutoff_branch": r["s_run_cutoff_branch"],
                           "t_sw_cutoff_branch": r["t_sw_cutoff_branch"]}
    return mine


@pytest.mark.parametrize("key,outcome", [("A165_sgd", "FAIL C3"), ("A165_adam", "FAIL C1+C2")])
def test_a165_full_path_reproduces_phase1b(key, outcome):
    row, res = H.score_a165(key, _a165_mine(key), 0.9)
    assert res["outcome_1c_rule"] == outcome and all(H.registered_check(key, res).values())


def test_a165_perturbed_differs():
    row, res = H.score_a165("A165_sgd", _a165_mine("A165_sgd", shift=50), 0.9)
    assert not H.registered_check("A165_sgd", res)["C1"]


# ---------------------------------------------------------------- refusals
def test_a177_retrain_hash_mismatch_is_refused(monkeypatch):
    seed = next(r["seed"] for r in T._rows(T.OUT / "runs.jsonl") if r.get("status") == "ok")
    z = np.zeros((5, 4))
    monkeypatch.setattr(T, "train", lambda s, *a, **k: (z, z, z))
    r = H.compute_a177(seed)
    assert r["status"].startswith("REFUSED") and "fc" not in r


def test_run_refuses_without_a_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "OUT", tmp_path)
    with pytest.raises(u2_gate.GateRefused):
        H.run()


def test_same_ignores_secs_only():
    a = {"t_fc": 10, "secs": 1.0, "x": [1.0, float("nan")]}
    assert H._same(a, {**a, "secs": 9.0})
    assert not H._same(a, {**a, "t_fc": 11})
