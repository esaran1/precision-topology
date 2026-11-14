"""Rebuttal figures 1-4 (src/figures_rebuttal.py): the data-extraction functions on constructed data, and the full path
from the committed tables to the plotted arrays, which must reproduce the ledger-pinned numbers."""
import json

import numpy as np
import pandas as pd
import pytest

from src import figures_rebuttal as FR


# ------------------------------------------------------------------------------------------ constructed data
def test_mech_crossings_streams_and_filters(tmp_path):
    p = tmp_path / "ck.csv"
    pd.DataFrame({"a": [1.3, 1.3, 1.3, 1.5, 1.5, 1.30000001],
                  "budget": [32_000, 32_000, 2_000, 32_000, 32_000, 32_000],
                  "w1": [-0.8, 0.9, 0.7, -1.0, 1.1, 0.85],
                  "w2": [-5.0, 4.0, 3.0, 2.6, -2.7, 5.1],
                  "crossing": [True, False, True, True, "", True],
                  "other": range(6)}).to_csv(p, index=False)
    d = FR.mech_crossings(p, chunksize=2)
    assert len(d) == 3                                       # budget 32,000 and crossing True only
    assert sorted(d.a.tolist()) == [1.3, 1.3, 1.5]
    assert (d.w1 > 0).all() and (d.w2 > 0).all()
    assert sorted(d.w2.tolist()) == [2.6, 5.0, 5.1]


def test_law_formula():
    assert FR.law(0.25, 0.5, 0.0) == pytest.approx(0.5 * 8.0)
    assert FR.law(0.25, 0.5, 2.0) == pytest.approx(0.5 * 8.0 * 1.5)
    assert np.all(np.diff(FR.law(np.array([0.1, 0.2, 0.4]), 0.68, 0.66)) < 0)


def test_scope_standard_constructed(tmp_path, monkeypatch):
    keys = ["width-1 sine | Adam", "width-1 sine | SGD", "width-1 GELU | Adam", "band task in R^d | Adam",
            "width 2 | Adam"]
    pooled = {k: {"summary": {"n": 10 * (i + 1), "n_abs_le_0.1": i + 1, "frac_abs_le_0.1": (i + 1) / (10 * (i + 1))}}
              for i, k in enumerate(keys)}
    (tmp_path / "phase2c").mkdir()
    (tmp_path / "phase2c" / "kappa_chi_map.json").write_text(json.dumps({"pooled": pooled}))
    monkeypatch.setattr(FR, "RESULTS", tmp_path)
    s = FR.scope_standard()
    assert [r["n"] for r in s] == [10, 20, 30, 40, 50]
    assert [r["k"] for r in s] == [1, 2, 3, 4, 5]
    assert all(r["frac"] == pytest.approx(0.1) for r in s)


def test_scope_forecasts_constructed(tmp_path, monkeypatch):
    def arm(k, n, f, k90=None):
        o = {"f": f, "C1": {"n_within": k, "n": n}}
        if k90 is not None:
            o["DESCRIPTIVE"] = {"other_f": {"f": 0.90, "criteria_stats": {"C1": {"n_within": k90, "n": n}}}}
        return o
    for d, j in (("phase1c", {"arms": {"G": arm(7, 8, 0.95, 1), "T": arm(5, 5, 0.95, 0), "Tp": arm(4, 6, 0.95, 2)}}),
                 ("phase3", {"arms": {"Q": arm(9, 10, 0.95), "S": arm(3, 3, 0.95)}}),
                 ("phase2a", {"criteria": {"C1": {"n_within": 2, "n": 2}}})):
        (tmp_path / d).mkdir()
        (tmp_path / d / "scores.json").write_text(json.dumps(j))
    monkeypatch.setattr(FR, "RESULTS", tmp_path)
    r = FR.scope_forecasts()
    assert [(x["k"], x["n"]) for x in r] == [(7, 8), (5, 5), (4, 6), (9, 10), (3, 3), (2, 2)]
    assert [x["k90"] for x in r] == [1, 0, 2, None, None, None]


def test_captions_have_no_internal_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "OUT", tmp_path)
    lag = {"n_scored": 1, "n_miss": 0, "median_ratio": 1.0}
    st = {"mechanism": {1.30: {"n_cross": 38, "n_path": 40}, 1.50: {"n_cross": 38, "n_path": 40}},
          "switch_law": {"n_finite": 6, "n_small": 5, "A_star": 0.685, "c_s": 0.662, "A_lo": 0.68125, "A_hi": 0.6875,
                         "max_abs_rel_dev_from_law": 0.01},
          "lag_causal": {k: lag for k in ("sgd_158", "sgd_165", "adam_177", "adam_165")},
          "scope": {"forecasts": [{"n": 79}], "standard": [{"n": 75}], "mlp": []}}
    FR.captions(st)
    text = " ".join(p.read_text() for p in tmp_path.glob("fig_*_caption.md"))
    assert len(list(tmp_path.glob("fig_*_caption.md"))) == 4
    for bad in ("Track", "Phase", "1C", "PASS", "FAIL", "W2-A", "GELU-T", "arm T"):
        assert bad not in text, bad
    for need in ("Message.", "Population and n.", "Uncertainty.", "post hoc", "0.95 was chosen after the pilot"):
        assert need in text, need


# ------------------------------------------------------------------------------------------ full path (committed data)
def test_switch_law_data_committed():
    d, k = FR.switch_law_data()
    fin = d[d.source == "finite-a certificate"]
    assert fin.a.tolist() == [1.30, 1.35, 1.40, 1.45, 1.50, 1.60]
    assert len(d[d.source == "small-eps certificate"]) == 5
    assert k["A_lo"] == pytest.approx(0.68125, abs=1e-9)                      # ledger "A* bracket lower" (tol 1e-9)
    assert k["A_hi"] == pytest.approx(0.6875, abs=1e-9)                       # ledger "A* bracket upper"
    assert k["c_s"] == pytest.approx(0.6621547, abs=1e-7)                     # ledger "A'/A* lo"
    assert k["A_lo"] <= k["A_star"] <= k["A_hi"]
    mid = 0.5 * (d.s_lo + d.s_hi)
    assert np.all(np.abs(mid / FR.law(d.eps, k["A_star"], k["c_s"]) - 1) < 0.03)  # within 3% (submitted claim)


def test_lag_registered_a158_committed():
    r = FR.lag_w1_158()
    assert (r["n_scored"], r["n_miss"], r["c1_within"], len(r["r_obs"])) == (84, 0, 84, 84)
    assert r["median_ratio"] == pytest.approx(1.169, abs=6e-4)
    assert r["f"] == 0.90
    assert np.median(r["r_obs"] / r["r_fc"]) == pytest.approx(r["median_ratio"], abs=1e-12)


def test_lag_registered_a177_committed():
    r = FR.lag_adam_177()
    assert (r["n_scored"], r["n_miss"], r["c1_within"], len(r["r_obs"])) == (98, 3, 93, 95)
    assert r["median_ratio"] == pytest.approx(0.940, abs=6e-4)
    assert r["f"] == 0.90


@pytest.mark.parametrize("opt,n,miss,c1,med", [("sgd", 63, 0, 63, 1.216), ("adam", 74, 8, 39, 1.158)])
def test_lag_posthoc_a165_committed(opt, n, miss, c1, med):
    r = FR.lag_165(opt)
    assert (r["n_scored"], r["n_miss"], r["c1_within"], len(r["r_obs"])) == (n, miss, c1, n - miss)
    assert r["median_ratio"] == pytest.approx(med, abs=6e-4)
    assert r["f"] == 0.90


def test_scope_committed():
    fa = {r["label"]: r for r in FR.scope_forecasts()}
    assert [(r["k"], r["n"], r["k90"]) for r in fa.values()] == [
        (74, 79, 9), (101, 101, 0), (120, 120, 0), (110, 112, None), (119, 119, None), (22, 22, None)]
    sb = {r["label"]: r for r in FR.scope_standard()}
    assert (sb["width-2 sine, Adam"]["n"], round(sb["width-2 sine, Adam"]["frac"], 2)) == (75, 0.12)    
    assert (sb["width-1 sine, Adam"]["n"], round(sb["width-1 sine, Adam"]["frac"], 2)) == (408, 0.73)
    assert (sb["width-1 sine, SGD"]["n"], round(sb["width-1 sine, SGD"]["frac"], 2)) == (207, 0.89)
    ml = [r for r in FR.scope_mlp() if r["arm"] == "output LR ÷16"]
    assert [round(r["lo"], 4) for r in ml] == [-0.0015, -0.0051, -0.0099]                               
    assert all(r["n"] == 40 for r in ml)
