"""Phase 2A-PS2 (src/phase2a_ps2.py): the approved page's settings; 2A-PS's module imported UNCHANGED (its hash equals
2A-PS's registration manifest); (a) the activity test (a 1e−93 share inactive, 0.129 active, the boundary inclusive,
â built from the active units only and its effect on 2,975,052's first step); (b) t* (constructed integrals, stall,
not computable), budgets and caps (⌈k·t*⌉ + 3000, ⌈192/ρ⌉, over cap counted, the clean edge between the two rates),
the per-run step cap and the RSS abort (counted, not scored); the status order with every new status; counts with the
untraceable count first, the gate, the author's over-cap check, the plan with budgets; constructed PASS / FAIL /
UNRESOLVED for every criterion, the gate and validity through score_tables (aborted runs excluded and counted); the
wider search DESCRIPTIVE (never a verdict); the ENFORCEMENT tests (a forecaster reading s at or after its cutoff fails
this file; a differing NaN recomputation aborts) and the NaN recomputation on a real path; REPRODUCTION with the repairs
switched off of 2A-PS's committed evaluations (all 400 seeds), freeze (2,975,052), runs and observations (2,975,052 at
both rates, 2,975,015 at 2⁻¹⁴); and the exploration's class, t* and wider search of seed 2,930,143."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import causal_forecast as CF
from src import causal_forecast_fold as CFF
from src import phase2a as P2A
from src import phase2a_ps as PS
from src import phase2a_ps2 as PS2

ROOT = Path(__file__).resolve().parents[1]
PS_PARTS = ROOT / "results/phase2a_ps/frozen_parts.jsonl"
PS_RUNS = ROOT / "results/phase2a_ps/runs.jsonl"
PS_OBS = ROOT / "results/phase2a_ps/observed.jsonl"
EXPLORE = ROOT / "results/designs/phase2a_ps2_explore/p2_explore.jsonl"
FROZEN = ROOT / "results/phase2a_ps2/frozen.json"
PILOT = ROOT / "results/phase2a_ps2/pilot.json"
OMEGA0 = CFF.OMEGA0


def _ps_rows():
    return [json.loads(ln) for ln in PS_PARTS.read_text().splitlines()]


def _ps_row(seed):
    return next(r for r in _ps_rows() if r["seed"] == seed)


def _ps_fs(seed):
    return PS.seed_record([_ps_row(seed)], seed)


def _committed(path, seed, l2):
    return next(r for r in (json.loads(ln) for ln in path.read_text().splitlines())
                if r["seed"] == seed and r["log2rho"] == l2)


# ================================================================================================ settings
def test_registered_settings_match_the_approved_page():
    assert PS2.SEEDS == tuple(range(2_987_000, 2_987_600)) and len(PS2.SEEDS) == 600
    assert PS2.PILOT_SEEDS == tuple(range(2_988_000, 2_988_060))
    assert not (set(PS2.SEEDS) | set(PS2.PILOT_SEEDS)) & (set(PS.SEEDS) | set(PS.PILOT_SEEDS) | set(PS2.EXPLORATION_SEEDS))
    assert PS2.THETA == 1e-8 and PS2.N_ACTIVE == 3
    assert PS2.K_BUDGET == {"clean": 1.5, "none": 2.5, "mixed": 2.5} and PS2.BUDGET_ADD == 3000
    assert PS2.step_cap(2.0 ** -14) == 3_145_728 and PS2.step_cap(2.0 ** -16) == 12_582_912
    assert PS2.RSS_RUN_CAP == 1024 ** 3 and PS2.OVERCAP_STOP_MAX == 5
    assert PS2.CAPS == {"clean": 40, "none": 30, "mixed": 20} and PS2.GATE_MIN_CLEAN == 24 and PS2.H_MIN_N == 10
    assert PS2.RATES == {"clean": (-14.0, -16.0), "none": (-14.0,), "mixed": (-14.0,)}
    # unchanged from 2A-PS: the class rule and every criterion constant
    assert PS.CLEAN_MIN == 27 and PS.PERTURB_N == 30 and PS.CLASS_SCALE == 1.01 and PS.F_CUT == 0.95
    assert PS.OBS_END == 1.25 and PS.H_MIN_FRAC == 0.80 and PS.F_MIN_FRAC == 0.90 and PS.C3_MIN_FRAC == 0.90


def test_phase2a_ps_is_imported_unchanged():
    """src/phase2a_ps.py (and the posthoc helpers) are byte for byte the files 2A-PS's registration and the posthoc
    diagnosis committed."""
    man = dict(reversed(ln.split()) for ln in (ROOT / "results/phase2a_ps/registration.sha256").read_text().splitlines())
    for rel in ("src/phase2a_ps.py", "src/causal_forecast_fold.py", "src/causal_forecast.py", "src/phase2a.py",
                "src/sb_fold.py"):
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == man[rel], rel


def test_status_order():
    o = PS2.STATUS_ORDER
    assert o[o.index("release Newton failed") + 1] == PS2.INACTIVE
    assert o[-5:] == ("ineligible", PS2.TSTAR_FAILED, PS2.STALL, PS2.OVER_CAP, "scoreable")
    assert [s for s in o if s not in (PS2.INACTIVE, PS2.TSTAR_FAILED, PS2.STALL, PS2.OVER_CAP)] == list(PS.STATUS_ORDER)


def test_scan_patterns_cover_exactly_the_two_ranges():
    import re
    rx = {k: re.compile(v) for k, v in PS2.SCAN_PATTERNS.items()}
    for s in (2_987_000, 2_987_599, 2_987_321):
        for txt in (str(s), f"{s:_}", f"{s:,}", f"x={s};"):
            assert rx["registered"].search(txt), txt
    for s in (2_988_000, 2_988_059):
        for txt in (str(s), f"{s:_}", f"{s:,}"):
            assert rx["pilot"].search(txt), txt
    for s in (2_986_999, 2_987_600, 2_987_9990, 29_870_001, 2_988_060, 2_975_052):
        for txt in (str(s), f"{s:_}", f"{s:,}"):
            assert not any(r.search(txt) for r in rx.values()), txt
    assert not rx["registered"].search("0.2987123") and not rx["registered"].search("12,987,123")


# ================================================================================================ (a) the activity test
def test_activity_test_constructed():
    s_tot = 3.0
    v = np.array([1.2, 1.8 - 1e-93 * s_tot, 1e-93 * s_tot, 0.0])
    assert PS2.active_units(v) == [0, 1]                                      # the 1e−93 share is inactive
    a = PS2.activity(np.r_[np.zeros(12), v, 0.0])
    assert a["active_registered"] == [0, 1, 2] and a["inactive"] == [2] and a["min_share_registered"] < 1e-90
    v = np.array([0.129, 0.5, 0.371, 0.0])                                    # the smallest live share 0.129: active
    assert PS2.active_units(v) == [0, 1, 2]
    v = np.array([1e-8, 1 - 1e-8, 0.0, 0.0])                                  # |v| = θ·s exactly: active (inclusive)
    assert PS2.active_units(v) == [0, 1]
    v = np.array([np.nextafter(1e-8, 0), 1 - 1e-8, 0.0, 0.0])
    assert PS2.active_units(v) == [1]
    assert PS2.active_units(np.zeros(4)) == []


def test_scale_direction_uses_the_active_units_only():
    v = np.array([0.4, -0.6, 1e-90, 0.0])
    a = PS2.scale_direction(v)
    assert a[2] == 0.0 and a[3] == 0.0 and np.allclose(a[:2], [1 / math.sqrt(2), -1 / math.sqrt(2)])


def test_activity_effect_on_2975052_first_step():
    """2,975,052 (2A-PS): v₂ = 1.2e−93 is counted active by v ≠ 0, so â leaks into the full-η part: Δs at step 1 is
    0.1028 at 2⁻¹⁴; with â on the active units only it is ≈ 9.4e−6 (the posthoc numbers)."""
    r = _ps_row(2_975_052)
    th = np.array(r["release_theta"], float)
    a = PS2.activity(th)
    assert a["inactive"] == [2] and a["active_registered"] == [0, 1, 2] and a["min_share_registered"] < 1e-90
    X, Y = PS.sample(2_975_052)
    rho = 2.0 ** -14
    ds_reg = PS.scale(PS.make_step(rho, a["active_registered"], X, Y)(th)) - PS.scale(th)
    ds_act = PS.scale(PS.make_step(rho, a["active_theta"], X, Y)(th)) - PS.scale(th)
    assert ds_reg == pytest.approx(0.1028, abs=5e-4)
    assert abs(ds_act) < 1e-4 and ds_act == pytest.approx(9.408e-6, rel=1e-2)


def test_activity_test_on_every_committed_2A_PS_release():
    """Over 2A-PS's 258 committed releases: the test drops a unit on exactly the five dead-unit releases (shares
    ≤ 2.1e−89 · s… 3.1e−88), all of them scoreable in 2A-PS; every other release keeps its three units."""
    drop = {}
    for r in _ps_rows():
        if r.get("release_theta") is None:
            continue
        a = PS2.activity(r["release_theta"])
        assert len(a["active_registered"]) == 3
        if a["inactive"]:
            drop[r["seed"]] = (r["evaluation"]["status"], a["min_share_registered"])
        else:
            assert a["min_share_registered"] > 0.04
    assert sorted(drop) == [2_975_052, 2_975_127, 2_975_226, 2_975_260, 2_975_345]
    assert all(st == "scoreable" and sh < 1e-87 for st, sh in drop.values())


# ================================================================================================ (b) t*, budgets, caps
def test_rho_t_star_constructed_integral_stall_and_not_computable():
    ss = np.linspace(2.0, 0.9999 * 5.0, 121)
    d = np.full(121, -0.01)
    rt, st = PS2.rho_t_star_of(ss, d, 2.0, 5.0)
    assert st == "ok" and rt == pytest.approx((0.9999 * 5.0 - 2.0) / (3 * 0.01), rel=1e-12)
    d2 = -0.01 * (1 + ss)                                                       # trapezoid of 1/(3·0.01(1+s))
    rt2, _ = PS2.rho_t_star_of(ss, d2, 2.0, 5.0)
    assert rt2 == pytest.approx(math.log((1 + 0.9999 * 5) / 3) / 0.03, rel=1e-4)
    d3 = d.copy()
    d3[60] = 0.0                                                                # dL_M/ds ≥ 0 at one point: stall
    assert PS2.rho_t_star_of(ss, d3, 2.0, 5.0) == (None, "stall")
    d4 = d.copy()
    d4[120] = 1e-9
    assert PS2.rho_t_star_of(ss, d4, 2.0, 5.0) == (None, "stall")
    d5 = list(d)
    d5[7] = None                                                                # Newton failed at a grid point
    assert PS2.rho_t_star_of(ss, d5, 2.0, 5.0) == (None, "not computable")
    d6 = d.copy()
    d6[3] = np.nan
    assert PS2.rho_t_star_of(ss, d6, 2.0, 5.0) == (None, "not computable")


def test_budget_formula_and_cap_boundary():
    rho = 2.0 ** -14
    b = PS2.budget_steps(20.0, 1.5, rho)
    assert b["t_star"] == 20.0 * 2 ** 14 and b["B_uncapped"] == math.ceil(1.5 * 20 * 2 ** 14) + 3000
    assert b["B"] == b["B_uncapped"] and not b["over_cap"] and b["cap"] == 3_145_728
    t_eq = (3_145_728 - 3000) / 1.5                                             # ⌈1.5 t*⌉ + 3000 = cap exactly
    b = PS2.budget_steps(t_eq * rho, 1.5, rho)
    assert b["B_uncapped"] == b["cap"] and b["B"] == b["cap"] and not b["over_cap"]
    b = PS2.budget_steps((t_eq + 1) * rho, 1.5, rho)
    assert b["B_uncapped"] == b["cap"] + 2 and b["B"] == b["cap"] and b["over_cap"]
    b = PS2.budget_steps(1.3, 1.5, rho)                                         # the + 3000
    assert b["B"] == math.ceil(1.5 * 1.3 * 2 ** 14) + 3000


def test_budgets_per_class_and_rate():
    bc = PS2.budgets("clean", 30.0)
    assert set(bc) == {"-14", "-16"} and bc["-16"]["B"] == math.ceil(45 * 2 ** 16) + 3000
    assert bc["-14"]["B"] == math.ceil(45 * 2 ** 14) + 3000
    bn = PS2.budgets("none", 30.0)
    assert set(bn) == {"-14"} and bn["-14"]["B"] == math.ceil(75 * 2 ** 14) + 3000
    assert PS2.budgets("mixed", 30.0) == bn
    assert PS2.budgets("none", 76.7)["-14"]["over_cap"] is False                # 2.5 × 76.7 = 191.75 (+0.18) < 192
    assert PS2.budgets("none", 76.8)["-14"]["over_cap"] is True
    assert PS2.budgets("clean", 127.8)["-14"]["over_cap"] is False and PS2.budgets("clean", 128.0)["-14"]["over_cap"]


# ================================================================================================ evaluation
def _fake_eval(monkeypatch, status, cls="clean"):
    monkeypatch.setattr(PS, "evaluate", lambda rec: {"status": status, "class": cls, "checks": {}, "Lambda_F": 1e-3,
                                                     "abs_mc": 1e-3, "eligible": status != "ineligible",
                                                     "s_F": 5.0, "s_star": 4.0, "s0": 2.0})


def _rec(inactive=False, tstatus="ok", rt=20.0, wide="mixed"):
    act = {"active_registered": [0, 1, 2], "active_theta": [0, 1] if inactive else [0, 1, 2],
           "inactive": [2] if inactive else [], "shares": [0.3, 0.3, 1e-93 if inactive else 0.4, 0.0],
           "min_share_registered": 1e-93 if inactive else 0.3}
    ts = {"rho_t_star": rt if tstatus == "ok" else None, "status": tstatus}
    return {"seed": 1, "ps2": {"ps_status": "scoreable", "activity": act, "tstar": ts,
                               "wide": {"class": {"pooled": wide}}}}


@pytest.mark.parametrize("ps_status,kw,cls,expected", [
    ("scoreable", {}, "clean", "scoreable"),
    ("scoreable", {"inactive": True}, "clean", PS2.INACTIVE),
    ("ineligible", {"inactive": True}, "none", PS2.INACTIVE),          # inactive comes BEFORE ineligible
    ("ineligible", {}, "none", "ineligible"),
    ("scoreable", {"tstatus": "not computable"}, "none", PS2.TSTAR_FAILED),
    ("scoreable", {"tstatus": "stall"}, "clean", PS2.STALL),
    ("scoreable", {"rt": 128.0}, "clean", PS2.OVER_CAP),
    ("scoreable", {"rt": 80.0}, "none", PS2.OVER_CAP),
    ("scoreable", {"rt": 80.0}, "clean", "scoreable"),
    ("scoreable", {"rt": 80.0}, "mixed", PS2.OVER_CAP),
    ("no s*", {"inactive": True}, "none", "no s*"),                      # earlier statuses are never changed
    ("release Newton failed", {}, "clean", "release Newton failed"),
    ("untraceable: M homotopy failed", {}, None, "untraceable: M homotopy failed"),
])
def test_evaluate_status_order(monkeypatch, ps_status, kw, cls, expected):
    _fake_eval(monkeypatch, ps_status, cls)
    e = PS2.evaluate(_rec(**kw))
    assert e["status"] == expected and e["class"] == cls                        # the class is never changed
    if expected == "scoreable":
        assert e["budgets"] == PS2.budgets(cls, kw.get("rt", 20.0)) and e["over_cap"] is False
    assert e["wide_class_DESCRIPTIVE"] == "mixed"


def test_clean_over_cap_at_either_rate(monkeypatch):
    """A clean seed is over cap if it is over cap at 2⁻¹⁴ OR 2⁻¹⁶ (the + 3000 weighs more at 2⁻¹⁴)."""
    _fake_eval(monkeypatch, "scoreable", "clean")
    rt = (192 - 3000 * 2.0 ** -16 - 0.05) / 1.5                                 # within the cap at 2⁻¹⁶ only
    b = PS2.budgets("clean", rt)
    assert b["-14"]["over_cap"] and not b["-16"]["over_cap"]
    assert PS2.evaluate(_rec(rt=rt))["status"] == PS2.OVER_CAP


def test_evaluate_repairs_off_is_2A_PS_on_every_committed_seed():
    """Repairs switched off: PS2.evaluate equals 2A-PS's committed evaluation on all 400 registered seeds."""
    rows = _ps_rows()
    assert len(rows) == 400
    for r in rows:
        raw = {k: v for k, v in r.items() if k != "evaluation"}
        assert P2A._jsonable(PS2.evaluate(raw, repairs=False)) == r["evaluation"], r["seed"]


# ================================================================================================ counts, gate, plan
def _row(seed, status, cls, rt=20.0):
    b = PS2.budgets(cls, rt) if status in ("scoreable", PS2.OVER_CAP) else None
    return {"seed": seed, "evaluation": {"status": status, "class": cls, "budgets": b, "rho_t_star": rt,
                                         "wide_class_DESCRIPTIVE": "mixed"}}


def test_counts_untraceable_first_and_gate():
    rows = ([_row(i, "untraceable: M homotopy failed", None) for i in range(5)]
            + [_row(10 + i, "scoreable", "clean") for i in range(24)] + [_row(50, PS2.INACTIVE, "clean"),
                                                                          _row(51, PS2.STALL, "none"),
                                                                          _row(52, PS2.OVER_CAP, "none", 80.0)])
    c = PS2.counts(rows)
    assert list(c)[:3] == ["n_seeds", "n_untraceable", "untraceable_frac"] and c["n_untraceable"] == 5
    assert c["n_inactive_unit"] == 1 and c["n_stall"] == 1 and c["n_over_cap"] == 1 and c["n_scoreable"] == 24
    assert c["class_by_status"][PS2.INACTIVE]["clean"] == 1
    assert PS2.gate(rows)["pass"] and not PS2.gate(rows[:-4])["pass"]          # 24 PASS, 23 FAIL
    with pytest.raises(AssertionError):
        PS2.counts(rows + [_row(99, "bogus", None)])


@pytest.mark.parametrize("n_clean,n_none,ok", [(5, 5, True), (6, 0, False), (0, 6, False), (5, 0, True)])
def test_author_over_cap_check(n_clean, n_none, ok):
    rows = ([_row(i, PS2.OVER_CAP, "clean", 130.0) for i in range(n_clean)]
            + [_row(100 + i, PS2.OVER_CAP, "none", 80.0) for i in range(n_none)]
            + [_row(200 + i, PS2.OVER_CAP, "mixed", 80.0) for i in range(9)])        # mixed never counts
    oc = PS2.over_cap_check(rows)
    assert oc["pass"] is ok and oc["n_over_cap"] == {"clean": n_clean, "none": n_none, "mixed": 9}


def test_plan_caps_in_seed_order_with_budgets():
    rows = ([_row(1000 + i, "scoreable", "clean", 10.0 + i) for i in range(45)]
            + [_row(2000 + i, "scoreable", "none", 20.0) for i in range(33)]
            + [_row(3000 + i, "scoreable", "mixed", 20.0) for i in range(21)]
            + [_row(999, PS2.OVER_CAP, "clean", 130.0), _row(998, PS2.STALL, "clean")])
    pl, surplus = PS2.plan(rows[::-1])
    clean = [p for p in pl if p["class"] == "clean"]
    assert [p["seed"] for p in clean[::2]] == list(range(1000, 1040))
    assert [p["log2rho"] for p in clean[:2]] == [-14.0, -16.0]
    assert clean[1]["budget"] == math.ceil(1.5 * 10.0 * 2 ** 16) + 3000
    assert [p["seed"] for p in pl if p["class"] == "none"] == list(range(2000, 2030))
    assert [p["budget"] for p in pl if p["class"] == "none"] == [math.ceil(50 * 2 ** 14) + 3000] * 30
    assert len([p for p in pl if p["class"] == "mixed"]) == 20
    assert surplus == {"clean": list(range(1040, 1045)), "none": list(range(2030, 2033)), "mixed": [3020]}
    assert 999 not in {p["seed"] for p in pl} and 998 not in {p["seed"] for p in pl}


# ================================================================================================ criteria via score_tables
def _table(n_clean=40, n_none=30, n_mixed=5):
    rows = []
    for i in range(n_clean):
        for l2, eps in ((-14.0, 1.6e-3), (-16.0, 4e-4)):
            r_fc = OMEGA0 * eps ** (2 / 3)
            r_obs = 1.2 * r_fc
            s_F = 5.0
            rows.append({"seed": i, "class": "clean", "log2rho": l2, "s_F": s_F, "s_star": 4.0, "Lambda_F": 1e-3,
                         "t_c": 1000, "has_fc": True, "status": "ok", "t_fc": 2000 - 10, "t_F_fc": 1800,
                         "delay_fc": 190, "eps_fc": eps, "r_fc": r_fc, "s_c_fc": s_F * (1 + 0.95 * r_obs),
                         "s_c_fixed": s_F * (1 + 0.5 * r_obs), "has_fc_fixed": True, "t_fc_fixed": 1900,
                         "t_obs": 2000, "s_obs": s_F * (1 + r_obs), "t_F": 1800, "r_obs": r_obs,
                         "reached_obs_end": False, "follow_in_M": True, "idle_ok": True, "sign_ok": True,
                         "aborted": False, "wide_class_DESCRIPTIVE": "clean"})
    for k, n in (("none", n_none), ("mixed", n_mixed)):
        for i in range(n):
            rows.append({"seed": 1000 * (k == "none") + 2000 * (k == "mixed") + i, "class": k, "log2rho": -14.0,
                         "s_F": 5.0, "s_star": 4.0, "t_c": 1000, "has_fc": True, "t_fc": 1990, "t_F_fc": 1800,
                         "delay_fc": 190, "eps_fc": 1.6e-3, "r_fc": 0.03, "s_c_fc": 5.1, "s_c_fixed": 5.05,
                         "t_obs": None, "s_obs": None, "t_F": 1800, "r_obs": None, "reached_obs_end": True,
                         "follow_in_M": True, "idle_ok": k == "none", "sign_ok": False, "aborted": False,
                         "wide_class_DESCRIPTIVE": "mixed"})
    return rows


def _mod(T, cond, **kw):
    for r in T:
        if cond(r):
            r.update(kw)
    return T


C16 = lambda r: r["class"] == "clean" and r["log2rho"] == -16.0                 # noqa: E731


def test_score_tables_constructed_pass():
    S = PS2.score_tables(_table())
    assert S["outcome"] == "PASS" and S["n_aborted_not_scored"] == 0
    assert set(S["verdicts"]) == {"F", "H", "E_seed", "C3", "C4", "P"}


@pytest.mark.parametrize("name,mut,outcome", [
    ("F 90%", lambda T: _mod(T, lambda r: C16(r) and r["seed"] < 5, s_obs=7.0), "FAIL F"),
    ("F falsifier", lambda T: _mod(T, lambda r: C16(r) and r["seed"] == 0, s_obs=4.9), "FAIL F"),
    ("H 80%", lambda T: _mod(T, lambda r: r["class"] == "none" and r["seed"] < 1007, t_obs=900, s_obs=5.5), "FAIL H"),
    ("H budget end", lambda T: _mod(T, lambda r: r["class"] == "none" and r["seed"] < 1007, reached_obs_end=False),
     "FAIL H"),
    ("E_seed", lambda T: _mod(T, lambda r: r["class"] == "clean" and r["log2rho"] == -14.0, eps_fc=8e-4), "FAIL"),
    ("C3", lambda T: _mod(T, lambda r: C16(r) and r["seed"] < 5, r_fc=1.0), "FAIL"),
    ("C4", lambda T: _mod(T, C16, t_fc=2500), "FAIL"),
    ("P", lambda T: _mod(T, C16, s_c_fc=10.0), "FAIL"),
])
def test_score_tables_each_failure(name, mut, outcome):
    S = PS2.score_tables(mut(_table()))
    assert S["outcome"].startswith(outcome), (name, S["verdicts"], S["outcome"])
    assert "FAIL" in S["outcome"]


def test_score_tables_unresolved_cases():
    assert PS2.score_tables(_table(n_none=9))["verdicts"]["H"] == "UNRESOLVED"
    assert PS2.score_tables(_table(n_none=10))["verdicts"]["H"] == "PASS"
    assert PS2.score_tables(_table(n_none=9))["outcome"] == "UNRESOLVED"
    assert PS2.score_tables(_table(), gate_ok=False)["outcome"] == "UNRESOLVED (gate)"
    assert PS2.score_tables(_table(n_clean=0))["verdicts"]["F"] == "UNRESOLVED"
    assert PS2.score_tables(_table(n_clean=2))["verdicts"]["E_seed"] == "UNRESOLVED"


@pytest.mark.parametrize("k_bad,ok", [(4, True), (5, False)])
def test_validity_follow_and_cutoff_before_90pct(k_bad, ok):
    S = PS2.score_tables(_mod(_table(), lambda r: C16(r) and r["seed"] < k_bad, follow_in_M=False))
    assert S["validity"]["ok"] is ok and (S["outcome"] == "UNRESOLVED (validity)") is (not ok)
    S = PS2.score_tables(_mod(_table(), lambda r: C16(r) and r["seed"] < k_bad, t_c=3000))
    assert S["validity"]["cutoff_before_ok"] is ok


def test_validity_idle_and_signs_every_clean_run():
    S = PS2.score_tables(_mod(_table(), lambda r: r["class"] == "clean" and r["log2rho"] == -14.0
                              and r["seed"] == 3, sign_ok=False))
    assert S["outcome"] == "UNRESOLVED (validity)"
    S = PS2.score_tables(_mod(_table(), lambda r: C16(r) and r["seed"] == 3, idle_ok=False))
    assert S["outcome"] == "UNRESOLVED (validity)"


def test_aborted_runs_are_counted_not_scored():
    T = _mod(_table(), lambda r: C16(r) and r["seed"] < 5, s_obs=7.0, follow_in_M=False, aborted=True)
    S = PS2.score_tables(T)
    assert S["outcome"] == "PASS" and S["n_aborted_not_scored"] == 5 and S["n_clean_scored"] == 35
    T = _mod(_table(), lambda r: r["class"] == "none" and r["seed"] < 1021, aborted=True)
    assert PS2.score_tables(T)["verdicts"]["H"] == "UNRESOLVED"                 # 9 trained none seeds left
    T = _mod(_table(), lambda r: r["class"] == "clean" and r["log2rho"] == -14.0 and r["seed"] < 3, aborted=True)
    S = PS2.score_tables(T)
    assert S["criteria"]["E_seed"]["n_excluded"] == 3 and S["outcome"] == "PASS"


def test_wider_search_is_descriptive_only():
    T = _table()
    S = PS2.score_tables(T)
    for w in ("clean", "none", "mixed", None):
        T2 = _mod(_table(), lambda r: True, wide_class_DESCRIPTIVE=w)
        S2 = PS2.score_tables(T2)
        assert S2["verdicts"] == S["verdicts"] and S2["outcome"] == S["outcome"]
    D = PS2.wide_crossing_rates(_mod(_table(), lambda r: r["class"] == "none" and r["seed"] < 1003, t_obs=900))
    assert "verdict" not in json.dumps(D) and D["by"]["none -14 wide-mixed"]["n_crossed"] == 3
    assert D["by"]["none -14 wide-mixed"]["n"] == 30 and D["label"].startswith("DESCRIPTIVE")


# ================================================================================================ ENFORCEMENT
@pytest.fixture(scope="module")
def fs015():
    import torch
    torch.set_num_threads(1)
    return _ps_fs(2_975_015)


def test_enforcement_leaky_forecaster_fails_run_one(monkeypatch, fs015):
    real = CFF.forecast_fold

    def leaky(s_view, t_c, *a, **k):
        _ = s_view[t_c]
        return real(s_view, t_c, *a, **k)
    monkeypatch.setattr(CFF, "forecast_fold", leaky)
    with pytest.raises(CF.CausalityViolation):
        PS2.run_one(fs015, -8.0, 40_000)

    def leaky_slice(s_view, t_c, *a, **k):
        _ = s_view[: t_c + 1]
        return real(s_view, t_c, *a, **k)
    monkeypatch.setattr(CFF, "forecast_fold", leaky_slice)
    with pytest.raises(CF.CausalityViolation):
        PS2.run_one(fs015, -8.0, 40_000)


def test_enforcement_differing_recomputation_aborts(monkeypatch, fs015):
    monkeypatch.setattr(CFF, "run_forecast", lambda *a, **k: ({"status": "ok"}, False))
    with pytest.raises(CF.CausalityViolation):
        PS2.run_one(fs015, -8.0, 40_000)


def test_forecast_budget_is_B_minus_t_c_and_nan_recomputation_on_a_real_path(monkeypatch, fs015):
    seen = []
    real = CFF.run_forecast

    def spy(S, t_c, s_F, L, eta, budget_left, f):
        seen.append((t_c, s_F, L, budget_left, f, S.copy()))
        return real(S, t_c, s_F, L, eta, budget_left, f)
    monkeypatch.setattr(CFF, "run_forecast", spy)
    r = PS2.run_one(fs015, -8.0, 40_000)
    assert r["t_c"] is not None and r["budget"] == 40_000 and r["nan_recompute_identical"]
    assert [x[:5] for x in seen] == [(r["t_c"], fs015["s_F"], fs015["Lambda_F"], 40_000 - r["t_c"], 0.95),
                                     (r["t_c"], PS.S_F_FIXED, PS.LAMF_FIXED, 40_000 - r["t_c"], 0.95)]
    assert r["forecast"]["max_index_read"] == r["t_c"] - 1
    S, t_c = seen[0][5], r["t_c"]
    rec = CFF.forecast_fold(S, t_c, fs015["s_F"], fs015["Lambda_F"], 1.0, 40_000 - t_c, 0.95)
    for poison in (np.nan, 1e9, -3.0):
        S2 = S.copy()
        S2[t_c:] = poison
        assert CFF.same_forecast(rec, CFF.forecast_fold(S2, t_c, fs015["s_F"], fs015["Lambda_F"], 1.0,
                                                        40_000 - t_c, 0.95))
    assert not (PS2.FORBIDDEN & P2A._keys(r))


def test_step_cap_and_rss_abort_counted(fs015):
    with pytest.raises(AssertionError):
        PS2.run_one(fs015, -8.0, PS2.step_cap(2.0 ** -8) + 1)
    big = lambda: 2 * 1024 ** 3                                                 # noqa: E731
    r = PS2.run_one(fs015, -14.0, 25_000, rss_fn=big)
    assert r["aborted"] and r["t_c"] is None and r["run_status"] == PS2.ABORTED
    assert r["forecast"]["status"] == PS2.ABORTED and not CFF.has_forecast(r["forecast"])
    ob, S = PS2.observe_one(fs015, -14.0, {"t_c": None}, 25_000, rss_fn=big)
    assert ob["aborted"] and ob["steps_observed"] == PS2.RSS_EVERY
    row = PS2.run_row({**fs015, "wide_class": "clean"}, r, ob)
    assert row["aborted"] and row["wide_class_DESCRIPTIVE"] == "clean"
    S, th, t_c, st = PS2.train_to_cutoff(np.array(fs015["release_theta"]), 2.0 ** -14, fs015["s_F"],
                                         *PS.sample(2_975_015), 1000)
    assert t_c is None and st == "no cutoff within the budget" and len(S) == 1001


# ================================================================================================ reproduction, repairs off
@pytest.mark.parametrize("l2", [-14.0, -16.0])
def test_reproduces_2A_PS_run_and_observation_of_2975052_with_repairs_off(l2):
    """2A-PS's committed run and observation of the dead-unit seed 2,975,052, reproduced bit for bit by PS2's run_one
    and observe_one at 2A-PS's budget ⌈64/ρ⌉ (repairs off: the v ≠ 0 active set; no per-seed budget)."""
    fs = _ps_fs(2_975_052)
    cr, co = _committed(PS_RUNS, 2_975_052, l2), _committed(PS_OBS, 2_975_052, l2)
    nb = PS.budget(PS.rho_of(l2))
    r = PS2.run_one(fs, l2, nb)
    assert r["t_c"] == cr["t_c"] == 73 and r["state_tc_sha256"] == cr["state_tc_sha256"]
    assert P2A._jsonable(r["forecast"]) == cr["forecast"] and P2A._jsonable(r["forecast_fixed"]) == cr["forecast_fixed"]
    ob, _ = PS2.observe_one(fs, l2, r, nb)
    assert P2A._jsonable({k: v for k, v in ob.items() if k not in ("budget", "aborted")}) == \
        {k: v for k, v in co.items() if k != "secs"}
    assert ob["sign_first_violation"] == 1 and not ob["aborted"]


def test_reproduces_2A_PS_run_and_observation_of_2975015_with_repairs_off(fs015):
    cr, co = _committed(PS_RUNS, 2_975_015, -14.0), _committed(PS_OBS, 2_975_015, -14.0)
    nb = PS.budget(2.0 ** -14)
    r = PS2.run_one(fs015, -14.0, nb)
    assert r["t_c"] == cr["t_c"] == 186_968 and r["state_tc_sha256"] == cr["state_tc_sha256"]
    assert P2A._jsonable(r["forecast"]) == cr["forecast"] and P2A._jsonable(r["forecast_fixed"]) == cr["forecast_fixed"]
    ob, _ = PS2.observe_one(fs015, -14.0, r, nb)
    assert P2A._jsonable({k: v for k, v in ob.items() if k not in ("budget", "aborted")}) == \
        {k: v for k, v in co.items() if k != "secs"}


@pytest.fixture(scope="module")
def frozen_052():
    import torch
    torch.set_num_threads(1)
    return PS2.freeze_seed(2_975_052, timing=False)


def test_reproduces_2A_PS_freeze_of_2975052_and_marks_it_inactive(frozen_052):
    r, c = frozen_052, _ps_row(2_975_052)
    raw_c = {k: v for k, v in c.items() if k not in ("evaluation", "secs", "peak_rss_gb")}
    raw_r = P2A._jsonable({k: v for k, v in r.items() if k not in ("evaluation", "ps2")})
    assert raw_r == raw_c
    assert P2A._jsonable(PS2.evaluate(r, repairs=False)) == c["evaluation"]
    e = r["evaluation"]
    assert e["status"] == PS2.INACTIVE and e["class"] == c["evaluation"]["class"] == "clean"
    assert r["ps2"]["activity"]["inactive"] == [2] and r["ps2"]["ps_status"] == "scoreable"
    assert r["ps2"]["tstar"]["status"] in ("ok", "stall", "not computable") and "wide" in r["ps2"]
    assert PS2.evaluate(json.loads(json.dumps(P2A._jsonable(r)))) == P2A._jsonable(e)


def _explore(seed):
    return next(json.loads(ln) for ln in EXPLORE.read_text().splitlines() if json.loads(ln)["seed"] == seed)


@pytest.fixture(scope="module")
def frozen_2930143():
    import torch
    torch.set_num_threads(1)
    return PS2.freeze_seed(2_930_143, timing=False)


def test_reproduces_the_exploration_class_tstar_and_wider_search(frozen_2930143):
    """Exploration seed 2,930,143 (p2_explore): the registered 30-minimum class (none, 0 of 30) exactly, ρ·t*
    (n = 3) and dL_M/ds at both ends exactly, and the wider search's counts (none → wide mixed) exactly."""
    r, h = frozen_2930143, _explore(2_930_143)
    e = r["evaluation"]
    assert e["status"] == h["status"] == "scoreable" and e["class"] == h["class_registered"] == "none"
    assert e["class_n_rho2_ge_q"] == h["n_ge_q_101"] == 0
    assert r["ps2"]["activity"]["inactive"] == [] and h["n_theta_active"] == 3
    assert e["rho_t_star"] == h["rho_t_star"]
    assert [r["ps2"]["tstar"]["dLds"][0], r["ps2"]["tstar"]["dLds"][-1]] == h["dLds_at_s0_and_sF"]
    w, hw = r["ps2"]["wide"]["class"], h["wide"]
    assert w == hw and w["pooled"] == e["wide_class_DESCRIPTIVE"] == "mixed"
    assert e["budgets"]["-14"]["B"] == math.ceil(2.5 * h["rho_t_star"] * 2 ** 14) + 3000


def test_code_closure_and_manifest():
    cc = PS2.code_closure()
    for m in ("src/phase2a_ps2.py", "src/phase2a_ps.py", "src/phase2a_ps_posthoc.py", "src/causal_forecast_fold.py",
              "src/causal_forecast.py", "src/sb_fold.py"):
        assert m in cc
    mf = PS2.manifest_files()
    for f in ("results/designs/phase2a_ps2_design.md", "results/phase2a_ps2_registration.md",
              "results/phase2a_ps/registration.sha256", "tests/test_phase2a_ps2.py"):
        assert f in mf


# ================================================================================================ frozen artifacts
@pytest.mark.skipif(not FROZEN.exists(), reason="not yet frozen")
def test_frozen_summary_consistent():
    fr = json.loads(FROZEN.read_text())
    parts = ROOT / "results/phase2a_ps2/frozen_parts.jsonl"
    assert fr["frozen_parts_sha256"] == hashlib.sha256(parts.read_bytes()).hexdigest()
    rows = [json.loads(ln) for ln in parts.read_text().splitlines()]
    assert [r["seed"] for r in rows] == list(PS2.SEEDS)
    assert hashlib.sha256(PS2.class_listing(rows).encode()).hexdigest() == fr["class_listing_sha256"]
    assert fr["counts"] == PS2.counts(rows) and fr["gate"] == PS2.gate(rows)
    assert fr["over_cap_check"] == P2A._jsonable(PS2.over_cap_check(rows))
    pl, _ = PS2.plan(rows)
    assert fr["plan"] == P2A._jsonable(pl)


@pytest.mark.skipif(not PILOT.exists(), reason="pilot not yet run")
def test_pilot_is_pilot_seeds_only():
    P = json.loads(PILOT.read_text())
    seeds = {int(s) for s in P["statuses"]}
    assert seeds <= set(PS2.PILOT_SEEDS)
    for key, run in P["runs"].items():
        assert int(key.split()[1]) in PS2.PILOT_SEEDS and not run["aborted"]
        assert run["observed"]["state_tc_hash_ok"] in (True, None) and run["nan_recompute_identical"] in (True, None)
