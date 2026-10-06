"""Phase 3 (width 4 on the asymmetric windows): tests of every registered rule (src/phase3_w4.py).

Constructed PASS / FAIL / UNRESOLVED cases for every criterion (C1-C4, S), gate and validity condition (V1-V7, the
follow check, the 90% cutoff rule, the NaN recomputation); the activity test (the 2A-PS 1.2e−93 unit); budgets and caps
(over-cap seeds counted, not trained); the causality enforcement (guard, last row read, NaN recomputation, a leaky and a
guard-bypassing forecaster); the width-2 reproduction of W2-A's frozen numbers; numerics against finite differences.
Population and constructed inputs only: no registered or pilot seed is drawn here."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import causal_forecast as C
from src import phase3_w4 as P

ROOT = Path(__file__).resolve().parents[1]
W2_LAND = ROOT / "results" / "width2_asym" / "landscape.json"
P3_LAND = ROOT / "results" / "phase3" / "landscape.json"


def _to4(zw):
    """W2-A's layout (α₁, β₁, α₂, β₂, b) -> this module's (α₁, α₂, β₁, β₂, b)."""
    return np.array([zw[0], zw[2], zw[1], zw[3], zw[4]], float)


# ================================================================================================ the page
def test_registered_settings_match_the_approved_page():
    assert P.SEEDS[P.Q] == tuple(range(7_430_000, 7_430_120)) and len(P.SEEDS[P.Q]) == 120
    assert P.SEEDS[P.S] == tuple(range(7_431_000, 7_431_120)) and len(P.SEEDS[P.S]) == 120
    assert P.SEEDS[P.T4] == tuple(range(7_432_000, 7_432_710)) and len(P.SEEDS[P.T4]) == 710
    assert P.PILOT_SEEDS == tuple(range(7_439_000, 7_439_010))
    assert P.SHARES == (0.1, 0.2, 0.3, 0.4) and P.ETA == 0.03 and P.F_CUT == 0.95
    assert abs(P.s0_value() - P.S0_PAGE) < 5e-8
    assert P.THETA_ACTIVE == 1e-8
    assert (P.BUDGET_MULT, P.BUDGET_ADD, P.STEP_CAP, P.RSS_RUN_CAP) == (1.5, 3000, 10 ** 6, 1024 ** 3)
    assert P.GATE_MIN == {"Q": 96, "S": 96, "T4": 60}
    assert P.GATE_HOLD_CONDITION == {"Q": True, "S": True, "T4": False}
    assert P.RHO_START == {"Q": 2.0 ** -13, "S": 2.0 ** -12, "T4": 2.0 ** -10}
    assert P.C3_BAND == (0.9, 1.1) and P.MIN_ABS_LAG == 6 and P.WITHIN_MIN_FRAC == 0.8 and P.S_MIN_FRAC == 0.8
    assert P.CRITERIA == {"Q": ("C1", "C2", "C3", "C4"), "S": ("C1", "C2", "C3", "C4", "S"),
                          "T4": ("C1", "C2", "C3", "C4")}
    assert (P.FAMILY, P.WINDOW_FRAC, P.MIN_WINDOW, P.HORIZON) == ("quad", 0.05, 50, 1.6)
    assert P.START == {"Q": "branch point", "S": "branch point", "T4": "random hold"}
    assert P.CUTOFF_BEFORE_MIN_FRAC == 0.9 and P.MIN_SCORED == 60 and P.BUDGET_CONDITION_MAX_OVER == 24


def test_seed_ranges_disjoint_and_not_exploration():
    allr = [set(P.SEEDS_Q), set(P.SEEDS_S), set(P.SEEDS_T4), set(P.PILOT_SEEDS), set(P.PILOT_SEEDS_T4)]
    for i in range(len(allr)):
        for j in range(i + 1, len(allr)):
            assert not (allr[i] & allr[j])
    assert not (P.all_phase3_seeds() & set(P.EXPLORATION_SEEDS))


def test_winding_table_has_129_entries():
    assert len(P.WINDING_TABLE) == 129
    assert P.in_winding_table((0, 0, 0, 0)) and P.in_winding_table((1, -1, 0, 1)) and P.in_winding_table((0, 0, -3, 0))
    assert not P.in_winding_table((2, 0, 0, -2)) and not P.in_winding_table(None)


# ================================================================================================ numerics
def test_gradient_and_hessian_against_finite_differences():
    x, y = P.population()
    rng = np.random.default_rng(1)
    z, v = rng.uniform(-1, 1, 9), rng.uniform(0.1, 1, 4)
    gz_, gv_ = P.grad(z, v, x, y)
    H = P.hess(z, v, x, y)
    q = np.concatenate([z, v])
    e = 1e-6
    L = lambda q: P.loss(q[:9], q[9:], x, y)                                    # noqa: E731
    G = lambda q: np.concatenate(P.grad(q[:9], q[9:], x, y))                    # noqa: E731
    gfd = np.array([(L(q + e * u) - L(q - e * u)) / (2 * e) for u in np.eye(13)])
    Hfd = np.array([(G(q + e * u) - G(q - e * u)) / (2 * e) for u in np.eye(13)])
    assert np.abs(np.concatenate([gz_, gv_]) - gfd).max() < 1e-8
    assert np.abs(H - Hfd).max() < 1e-7


def test_k2_enclosure_equals_width2_asym():
    from src import width2_asym as W2
    rng = np.random.default_rng(3)
    for _ in range(5):
        zw = rng.uniform(-2, 2, 5)
        v = rng.uniform(0.01, 0.5, 2)
        a, b = P.gap_enclosure(_to4(zw), v), W2.gap_enclosure(zw, v)
        assert abs(a[0] - b[0]) <= 1e-12 and abs(a[1] - b[1]) <= 1e-12
        assert abs(P.gap_dense(_to4(zw), v) - W2.gap_dense(zw, v)) <= 1e-12


def test_width2_reproduction_of_w2a_frozen_numbers():
    """At K = 2 the width-4 code reproduces W2-A's frozen population landscape (T, T′): λ_min(s₀), the validated switch
    (equal to the last digit), κ₀ and the winding coefficients (to 1e−9), from W2-A's frozen points."""
    land = json.loads(W2_LAND.read_text())
    x, y = P.population()
    v = np.array(land["v_T"])
    for c in ("T", "Tp"):
        ref = land["copies"][c]
        z, gm, lam, ok = P.newton(_to4(ref["z_s0"]), v, x, y)
        assert ok and abs(lam - ref["lam_min_s0"]) < 1e-12
        sw = P.switch_of_copy(z, v, x, y, 30.0)
        assert sw["valid"]
        assert abs(sw["s_switch"] / ref["switch"]["s_switch"] - 1) < 1e-12
        assert abs(sw["kappa0"] - ref["switch"]["kappa0"]) < 1e-9
        assert np.allclose(sw["kappa_winding_coef"], ref["switch"]["kappa_winding_coef"], atol=1e-9, rtol=0)
        assert abs(sw["lam_min_switch"] - ref["switch"]["lam_min_switch"]) < 1e-12


def test_walk_equals_w2a_own_path_switch_and_r4_at_width_2():
    """The one-pass branch walk (own-path switch + R4) and the incremental OwnSwitch equal W2-A's registered
    OwnBranch / own_path_switch / r4_recursion on a K = 2 path."""
    from src import width2_asym as W2
    land = json.loads(W2_LAND.read_text())
    x, y = P.population()
    v = np.array(land["v_T"])
    zw = np.array(land["copies"]["T"]["z_s0"])
    zw_rel = zw + 1e-4 * np.array([1.0, -1.0, 1.0, -1.0, 0.5])
    Pp = W2.train_path(W2.qof(zw_rel, v), 0.03, 2.0 ** -7, 2500, x, y)
    V = Pp[:, W2.VI]
    br = W2.OwnBranch(V, zw, x, y, False)
    t_sw, s_sw = W2.own_path_switch(br, len(V) - 1)
    t_hit, st, _, _ = W2.r4_recursion(br, zw_rel, 0.03, len(V) - 1)
    assert t_sw is not None and t_hit is not None
    w = P.walk(V, _to4(zw), x, y, _to4(zw_rel), 0.03)
    assert w["t_sw"] == t_sw and abs(w["s_switch"] - s_sw) < 1e-10
    assert w["t_hit"] == t_hit and w["r4_status"] == st
    o = P.OwnSwitch(_to4(zw), V[0], x, y)
    for row in V[1:]:
        o.feed(row)
        if o.done:
            break
    assert o.t_sw == t_sw and abs(o.s_sw - s_sw) < 1e-10
    assert len(o.br.lam) == t_sw + 1 and abs(o.br.lam[5] - br.lam[5]) < 1e-12


def test_population_landscape_validated():
    if not P3_LAND.exists():
        pytest.skip("landscape.json not present")
    land = json.loads(P3_LAND.read_text())
    assert land["validated"] and abs(land["s0"] - P.S0_PAGE) < 5e-8
    for c in ("Q", "S", "T4"):
        r = land["copies"][c]
        assert r["valid"] and r["type"] == P.COPY_TYPE[c] and r["matches_page"]
        assert tuple(tuple(g) for g in r["partition"]) == P.COPY_PARTITION[c]
        assert abs(r["s_pop_branch"] / P.POP_SWITCH_PAGE[c] - 1) <= P.POP_SWITCH_REL
        assert np.sign(r["switch"]["kappa0"]) == P.POP_KAPPA_SIGN_PAGE[c]


# ================================================================================================ symmetries
def test_groups_partition_and_type():
    T4 = np.array(P.POP_START["T4"])
    assert P.partition(T4) == ((0,), (1, 2, 3))
    v0 = P.v_held(P.S0_PAGE)
    assert P.btype(T4, v0) == "31:+-"
    Sp = np.array(P.POP_START["S"])
    assert P.partition(Sp) == ((0, 2), (1,), (3,)) and P.btype(Sp, v0) == "121:+-+"
    Qp = np.array(P.POP_START["Q"])
    assert P.partition(Qp) == ((0,), (1,), (2,), (3,)) and P.btype(Qp, v0) == "1111:++--"
    z = T4.copy()
    z[5] += 2 * math.pi                                                       # a winding keeps the coincidence
    assert P.partition(z) == ((0,), (1, 2, 3))
    z[1] += 1e-5                                                              # a split beyond 1e−6 breaks it
    assert P.partition(z) == ((0,), (1,), (2, 3))


def test_shift_is_a_symmetry_and_windings_to_recovers_k():
    x, y = P.population()
    rng = np.random.default_rng(5)
    z, v = rng.uniform(-1, 1, 9), rng.uniform(0.05, 0.5, 4)
    for k in ((1, 0, 0, 0), (0, -1, 2, 0), (1, 1, -1, 0)):
        zk = P.shift(z, v, k)
        assert abs(P.loss(zk, v, x, y) - P.loss(z, v, x, y)) < 1e-12
        assert P.windings_to(zk, z) == k


def test_kappa_at_a_winding_is_affine_and_exact():
    """κ_k = κ₀ + a·k equals κ computed directly at the shifted switch point (population T4 switch)."""
    if not P3_LAND.exists():
        pytest.skip("landscape.json not present")
    land = json.loads(P3_LAND.read_text())
    sw = land["copies"]["T4"]["switch"]
    x, y = P.population()
    z, v = np.array(sw["z_switch"]), np.array(sw["v_switch"])
    k0 = P.kappa_at(z, v, sw["vprime"], x, y)
    for k in ((1, 0, 0, 0), (0, 1, -1, 0), (0, 0, 0, 2)):
        kk = P.kappa_at(P.shift(z, v, k), v, sw["vprime"], x, y)
        assert abs(kk["kappa0"] - P.kappa_winding(k0["kappa0"], k0["kappa_winding_coef"], k)) < 1e-6 * max(
            1.0, abs(kk["kappa0"]))


def test_reduced_lambda_excludes_the_split_directions():
    if not P3_LAND.exists():
        pytest.skip("landscape.json not present")
    land = json.loads(P3_LAND.read_text())
    x, y = P.population()
    v0 = np.array(land["v0"])
    for c in ("S", "T4"):
        z = np.array(land["copies"][c]["z_s0"])
        H = P.blocks(z, v0, x, y)[0]
        lr, ls = P.reduced_lam(H, z, v0)
        assert ls is not None and lr > 0 and ls > 0
        assert P.split_basis(z, v0).shape[1] == 2 * sum(len(g) - 1 for g in P.groups(z))
    z = np.array(land["copies"]["Q"]["z_s0"])
    assert P.reduced_lam(P.blocks(z, v0, x, y)[0], z, v0)[1] is None


# ================================================================================================ the activity test
V52 = np.array([1.499465234754688, 1.446173704391896, 1.1793476661290464e-93, 0.0])      # 2A-PS seed 2,975,052
G52 = np.array([-0.07707232379345327, -0.07707232379283815, 3.1581182483505337e-78, 0.0])


def test_activity_dead_and_zero_units_inactive():
    assert P.active_units(V52) == [0, 1]
    assert P.active_units(np.array([1.0, 1e-93, 2.0, 0.5])) == [0, 2, 3]
    assert P.active_units(np.array([0.3, 0.0, 0.2, 0.1])) == [0, 2, 3]
    assert not P.all_units_active(V52)


def test_activity_genuine_small_units_active():
    assert P.active_units(P.S0_PAGE * np.array([0.1, 0.2, 0.3, 0.4])) == [0, 1, 2, 3]
    assert P.active_units(0.6264 * np.array([0.03593, 0.32934, 0.31737, 0.31737])) == [0, 1, 2, 3]
    assert P.active_units(np.array([1.0, 1e-6, 1.0, 1.0])) == [0, 1, 2, 3]
    assert P.all_units_active(P.v_held())


def test_activity_threshold_inclusive_and_scale_free():
    th = 2.0 ** -20
    v = np.array([0.5, 0.25, 0.25 - 2.0 ** -20, 2.0 ** -20])
    v2 = np.array([0.5, 0.25, 0.25 - 2.0 ** -21, 2.0 ** -21])
    assert P.active_units(v, theta=th) == [0, 1, 2, 3]
    assert P.active_units(v2, theta=th) == [0, 1, 2]
    assert P.active_units(2.0 ** 30 * v, theta=th) == [0, 1, 2, 3]
    assert P.active_units(2.0 ** 30 * v2, theta=th) == [0, 1, 2]


def test_scale_direction_uses_active_units_only():
    a = P.scale_direction(V52)
    assert np.allclose(a, [2 ** -0.5, 2 ** -0.5, 0, 0])
    assert not np.allclose(a, np.array([1, 1, 1, 0]) / np.sqrt(3))


@pytest.mark.parametrize("rho, ds_new", [(2.0 ** -14, 9.408e-06), (2.0 ** -16, 2.352e-06)])
def test_activity_effect_on_the_first_scale_step(rho, ds_new):
    old = P.scale_only_step(V52, G52, 1.0, rho, np.array([1, 1, 1, 0]) / np.sqrt(3))
    new = P.scale_only_step(V52, G52, 1.0, rho, P.scale_direction(V52))
    assert np.abs(old).sum() - np.abs(V52).sum() == pytest.approx(0.10277, abs=2e-5)
    assert np.abs(new).sum() - np.abs(V52).sum() == pytest.approx(ds_new, rel=1e-3)


# ================================================================================================ release classification
@pytest.fixture(scope="module")
def t4_point():
    v0 = P.v_held()
    zc = np.array(P.POP_START["T4"])
    return zc, v0


def _cls(z_state, zn, ok, zc, v, part=P.COPY_PARTITION["T4"]):
    x, y = P.population()
    return P.classify_release(z_state, v, {"T4": (zc, part)}, x, y, newton_result=(zn, 0.0, 0.01, ok))


def test_release_on_copy_and_winding_fixed(t4_point):
    zc, v0 = t4_point
    c, k, rec = _cls(zc + 5e-4, zc, True, zc, v0)
    assert c == "T4" and k == (0, 0, 0, 0) and rec["all_units_active"]
    kk = (1, 0, -1, 0)
    zk = P.shift(zc, v0, kk)
    c, k, _ = _cls(zk + 5e-4, zk, True, zc, v0)
    assert c == "T4" and k == kk


def test_release_neither_cases(t4_point):
    zc, v0 = t4_point
    assert _cls(zc, zc, False, zc, v0)[0] is None                              # Newton not accepted
    assert _cls(zc + 2e-3, zc, True, zc, v0)[0] is None                         # state > 1e−3 from the point
    assert _cls(zc, zc + 2e-6, True, zc, v0)[0] is None                         # Newton point > 1e−6 from the copy
    z2 = zc.copy()
    z2[1] += 5e-7                                                               # within 1e−6 of the copy …
    c, _, _ = _cls(z2, z2, True, zc, v0, part=((0,), (1,), (2, 3)))             # … but another partition
    assert c is None


def test_release_with_an_inactive_unit_is_neither(t4_point):
    """THE ACTIVITY TEST: a release on the copy with a numerically dead unit (2A-PS's 1.2e−93) is 'neither'."""
    zc, v0 = t4_point
    vd = v0.copy()
    vd[2] = 1.1793476661290464e-93
    c, k, rec = _cls(zc, zc, True, zc, vd)
    assert c is None and k is None and rec["note"] == "neither (inactive unit)" and rec["active_units"] == [0, 1, 3]


# ================================================================================================ budgets and caps
def test_budget_rule_and_the_step_cap():
    assert P.drive_steps(0.25, 2.0 ** -13) == pytest.approx(0.25 / (2.0 ** -13 * 0.03))
    assert P.budget_of(1000.0) == 4500 and not P.over_cap(1000.0)
    t_edge = (P.STEP_CAP - P.BUDGET_ADD) / 1.5                                 # ⌈1.5 t*⌉ + 3000 = 10⁶ exactly
    assert P.budget_uncapped(t_edge) == P.STEP_CAP and not P.over_cap(t_edge) and P.budget_of(t_edge) == P.STEP_CAP
    assert P.over_cap(t_edge + 1.0) and P.budget_of(t_edge + 1.0) == P.STEP_CAP
    assert P.budget_of(1000.2) == 4501                                          # ⌈1500.3⌉ + 3000


def _fs(arm="Q", on=True, valid=True, k=(0, 0, 0, 0), over=False, B=5000):
    return {"arm": arm, "seed": 1, "release": {"on_copy": on, "windings": list(k), "copy_at_release": arm if on else None,
                                               "hold_G_positive": False},
            "copies": {P.COPY_OF[arm]: {"valid": valid}}, "budget": {"over_cap": over, "B": None if over else B}}


def test_eligibility_and_over_cap_seeds_are_counted_not_trained(monkeypatch):
    assert P.eligibility(_fs()) == (True, "eligible")
    assert not P.eligibility(_fs(on=False))[0]
    assert not P.eligibility(_fs(valid=False))[0]
    assert not P.eligibility(_fs(k=(2, 2, 0, 0)))[0]
    ok, why = P.eligibility(_fs(over=True))
    assert not ok and "over the 10⁶-step cap" in why

    def boom(*a, **k):
        raise AssertionError("an over-cap seed must not be trained")
    monkeypatch.setattr(P, "train_to_cutoff", boom)
    r = P.run_one(_fs(over=True), 2.0 ** -13, {"v0": list(P.v_held())})
    assert r["eligible"] is False and "t_c" not in r and "forecast" not in r


def test_train_to_cutoff_caps(monkeypatch):
    x, y = P.population()
    z0, v0 = np.array(P.POP_START["Q"]), P.v_held()
    with pytest.raises(AssertionError):
        P.train_to_cutoff(z0, v0, 1e-3, 10.0, P.STEP_CAP + 1, x, y)
    monkeypatch.setattr(P, "RSS_EVERY", 7)
    monkeypatch.setattr(P, "current_rss", lambda: 2 * 1024 ** 3)
    Z, V, t_c, st = P.train_to_cutoff(z0, v0, 1e-3, 10.0, 100, x, y)
    assert t_c is None and st.startswith("aborted: per-run RSS cap") and len(V) == 8
    monkeypatch.setattr(P, "current_rss", lambda: 0)
    Z, V, t_c, st = P.train_to_cutoff(z0, v0, 1e-3, 10.0, 30, x, y)
    assert t_c is None and st == "no cutoff within the budget" and len(V) == 31


def test_train_to_cutoff_is_a_stopping_time():
    x, y = P.population()
    z0, v0 = np.array(P.POP_START["Q"]), P.v_held()
    lvl = 1.0005 * float(np.abs(v0).sum())
    Z, V, t_c, st = P.train_to_cutoff(z0, v0, 2.0 ** -6, lvl, 5000, x, y)
    s = np.abs(V).sum(axis=1)
    assert st == "ok" and len(V) == t_c + 1 and s[t_c] >= lvl and np.all(s[1:t_c] < lvl)


def test_budget_condition_stops_above_24_over_cap():
    def rows(n_over, n=120):
        return [{"budget": {"over_cap": i < n_over, "B": P.STEP_CAP if i < n_over else 5000,
                            "B_uncapped": 2_000_000 if i < n_over else 5000, "t_star": 1.0}} for i in range(n)]
    assert P.budget_condition({"Q": rows(24), "S": rows(0)})["pass"]
    bc = P.budget_condition({"Q": rows(25), "S": rows(0)})
    assert not bc["pass"] and bc["Q"]["n_over_cap"] == 25
    assert not P.budget_condition({"Q": rows(0), "S": rows(30)})["pass"]


# ================================================================================================ pilot rules
def test_pilot_rule_keeps_halves_and_stops():
    seq = {2.0 ** -13: [0.05] * 10}
    r = P.pilot_rule(lambda rho: seq[rho], 2.0 ** -13)
    assert r["status"] == "ok" and r["rho"] == 2.0 ** -13
    vals = {2.0 ** -13: [0.2] * 10, 2.0 ** -14: [0.09] * 10}
    r = P.pilot_rule(lambda rho: vals[rho], 2.0 ** -13)
    assert r["status"] == "ok" and r["rho"] == 2.0 ** -14 and len(r["history"]) == 2
    r = P.pilot_rule(lambda rho: [0.5] * 10, 2.0 ** -10)
    assert r["status"] == "STOP" and len(r["history"]) == 4 and r["history"][-1]["rho"] == 2.0 ** -13
    assert P.pilot_rule(lambda rho: [], 2.0 ** -10)["status"] == "STOP"


def test_tolerance_rule_is_phase_1a_s():
    from src.phase1a_pilot import _ceil_to
    t = P.tolerances([4.0] * 9 + [3.0], [0.5] * 10)
    assert t["tau_cross"] == _ceil_to(1.5 * 4.0, 5) == 10 and t["tau_lag"] == 5
    t = P.tolerances(list(range(-20, 21)), [7, -7, 9, None, float("nan")])
    q = float(np.percentile(np.abs(np.arange(-20, 21)), 90))
    assert t["tau_cross"] == _ceil_to(1.5 * q, 5) and t["n_lag"] == 3
    assert t["tau_lag"] == _ceil_to(1.5 * float(np.percentile([7, 7, 9], 90)), 5)
    assert P.tolerances([], [])["tau_cross"] is None


# ================================================================================================ criteria
def test_c1_c2_within_and_misses():
    hf = np.ones(10, bool)
    assert P.criterion_within([1] * 8 + [50] * 2, 10, hf)["verdict"] == "PASS"
    assert P.criterion_within([1] * 7 + [50] * 3, 10, hf)["verdict"] == "FAIL"
    assert P.criterion_within([10] * 10, 10, hf)["verdict"] == "PASS"                     # inclusive
    hf2 = hf.copy()
    hf2[:3] = False
    assert P.criterion_within([1] * 10, 10, hf2)["verdict"] == "FAIL"                     # misses not within
    assert P.criterion_within([], 10, [])["verdict"] == "UNRESOLVED"


def test_c3_band_and_the_lag_floor():
    hf = np.ones(5, bool)
    ro = np.full(5, 0.02)
    assert P.criterion_c3(ro, ro / 1.05, [10] * 5, hf)["verdict"] == "PASS"
    assert P.criterion_c3(ro, ro / 1.2, [10] * 5, hf)["verdict"] == "FAIL"
    assert P.criterion_c3(ro, ro / 1.1, [10] * 5, hf)["verdict"] == "PASS"                # closed band
    r = P.criterion_c3(ro, ro / np.array([1.0, 1.0, 1.0, 2.0, 2.0]), [10, 10, 10, 5, -5], hf)
    assert r["n"] == 3 and r["verdict"] == "PASS"                                           # |lag_fc| < 6 excluded
    assert P.criterion_c3(ro, ro, [5] * 5, hf)["verdict"] == "UNRESOLVED"
    assert P.criterion_c3(ro, ro, [10] * 5, np.zeros(5, bool))["verdict"] == "UNRESOLVED"


def test_c4_pass_fail_unresolved():
    n = 20
    t_obs = np.full(n, 1100.0)
    r = P.criterion_c4(t_obs + 2, t_obs - 100, t_obs, np.ones(n, bool))
    assert r["verdict"] == "PASS" and r["ci95"][1] < 0
    r = P.criterion_c4(t_obs + 2, t_obs + 1, t_obs, np.ones(n, bool))                     # entirely above 0
    assert r["verdict"] == "FAIL" and r["ci95"][0] > 0
    d = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    r = P.criterion_c4(t_obs + 2, t_obs + 2 - d, t_obs, np.ones(n, bool))                 # containing 0
    assert r["verdict"] == "FAIL" and r["ci95"][0] < 0 < r["ci95"][1]
    assert P.criterion_c4([1.0], [0.0], [1.0], [True])["verdict"] == "UNRESOLVED"
    hf = np.ones(n, bool)
    hf[:5] = False
    assert P.criterion_c4(t_obs + 2, t_obs - 100, t_obs, hf)["n"] == 15                    # misses excluded


def test_bootstrap_and_criteria_equal_phase1c_s_procedures():
    from src import phase1c as P1
    rng = np.random.default_rng(0)
    D = rng.normal(-3, 5, 40)
    assert P.bootstrap_mean_ci(D, seed=P1.BOOT_SEED) == P1.bootstrap_mean_ci(D)
    t_obs = rng.integers(1000, 2000, 40).astype(float)
    t_fc = t_obs + rng.integers(-8, 9, 40)
    t_sw_fc = t_fc - rng.integers(-60, 60, 40)
    hf = rng.random(40) > 0.1
    a, b = P.criterion_within(t_fc - t_obs, 5, hf), P1.criterion_within(t_fc - t_obs, 5, hf)
    assert a["verdict"] == b["verdict"] and a["frac_within"] == b["frac_within"]
    s1, s2 = P.criterion_sign(t_fc - t_sw_fc, t_obs - t_sw_fc, hf), P1.criterion_sign(t_fc - t_sw_fc, t_obs - t_sw_fc, hf)
    assert (s1["verdict"], s1["frac"], s1["n_eligible"]) == (s2["verdict"], s2["frac"], s2["n_eligible"])


def test_sign_criterion():
    hf = np.ones(10, bool)
    assert P.criterion_sign([-10] * 10, [-12] * 8 + [3] * 2, hf)["verdict"] == "PASS"
    assert P.criterion_sign([-10] * 10, [-12] * 7 + [3] * 3, hf)["verdict"] == "FAIL"
    r = P.criterion_sign([-10] * 8 + [-3] * 2, [-12] * 8 + [3] * 2, hf)                    # |lag_fc| < 6 excluded
    assert r["n_eligible"] == 8 and r["verdict"] == "PASS"
    hf2 = hf.copy()
    hf2[:3] = False                                                                       # misses count as failures
    r = P.criterion_sign([-10] * 10, [-12] * 10, hf2)
    assert r["n_eligible"] == 10 and r["frac"] == 0.7 and r["verdict"] == "FAIL"
    assert P.criterion_sign([-3] * 4, [-3] * 4, np.ones(4, bool))["verdict"] == "UNRESOLVED"


def test_outcome_rule():
    assert P.outcome({"C1": "PASS", "C2": "PASS"}) == "PASS"
    assert P.outcome({"C1": "PASS", "C2": "FAIL", "C4": "FAIL"}) == "FAIL C2+C4"
    assert P.outcome({"C1": "UNRESOLVED", "C2": "FAIL"}) == "UNRESOLVED"


# ================================================================================================ gates
@pytest.mark.parametrize("arm, n, n_on, hold, ok", [
    ("Q", 120, 96, False, True), ("Q", 120, 95, False, False), ("Q", 120, 120, True, False),
    ("S", 120, 96, False, True), ("S", 120, 100, True, False),
    ("T4", 710, 60, False, True), ("T4", 710, 59, False, False), ("T4", 710, 60, True, True)])
def test_gates(arm, n, n_on, hold, ok):
    on = np.zeros(n, bool)
    on[:n_on] = True
    hp = np.zeros(n, bool)
    hp[0] = hold
    assert P.gate(arm, on, hp)["pass"] is ok


# ================================================================================================ the verdict per arm
def _table(arm="Q", n=120):
    """A constructed arm that PASSES every criterion and validity condition."""
    neg = arm == "S"
    t_sw = np.full(n, 1000.0)
    t_obs = t_sw + (-50.0 if neg else 100.0)
    s_sw = np.full(n, 0.48)
    s_obs = s_sw * (1 + (-0.01 if neg else 0.02))
    lag_fc = -50.0 if neg else 100.0
    t_sw_fc = t_sw + 1
    t_fc = t_sw_fc + lag_fc
    r_obs = s_obs / s_sw - 1
    return {"on_copy": np.ones(n, bool), "eligible": np.ones(n, bool), "over_cap": np.zeros(n, bool),
            "aborted": np.zeros(n, bool), "hold_positive": np.zeros(n, bool), "crossed": np.ones(n, bool),
            "t_obs": t_obs, "s_obs": s_obs, "t_sw": t_sw, "s_sw": s_sw, "t_c": t_sw - 300, "t_fc": t_fc,
            "t_sw_fc": t_sw_fc, "s_fc": s_obs, "r_fc": r_obs * 1.02, "nan_identical": np.ones(n, bool),
            "follows": np.ones(n, bool), "r_cf": np.full(n, 0.02), "kappa": np.full(n, -0.012 if neg else 0.017),
            "eta_lam": np.full(n, 0.00015), "lag_steps": np.full(n, -60.0 if neg else 110.0),
            "chi_tsw": np.full(n, 0.05), "chi_win_max": np.full(n, 0.07), "s_pop": np.full(n, 0.48)}


def _score(R, arm="Q"):
    return P.score_arm(arm, R, pilot_chi=0.05, tau_cross=10, tau_lag=5)


@pytest.mark.parametrize("arm, n", [("Q", 120), ("S", 120), ("T4", 710)])
def test_constructed_arm_passes(arm, n):
    r = _score(_table(arm, n), arm)
    assert r["outcome"] == "PASS" and r["valid"] and r["n_scored"] == n
    assert all(r[k]["verdict"] == "PASS" for k in P.CRITERIA[arm])
    if arm != "S":
        assert "S_descriptive" in r["DESCRIPTIVE"]
    if arm == "T4":
        assert "sensitivity_excluding_hold_G_positive" in r["DESCRIPTIVE"]


def test_gate_failure_is_unresolved_gate():
    R = _table()
    R["on_copy"][:25] = False
    r = _score(R)
    assert r["outcome"] == "UNRESOLVED (gate)" and all(r[k]["verdict"] == "UNRESOLVED" for k in P.CRITERIA["Q"])
    R = _table()
    R["hold_positive"][3] = True
    assert _score(R)["outcome"] == "UNRESOLVED (gate)"


def _mut(key, idx, val, arm="Q"):
    R = _table(arm)
    R[key] = np.array(R[key], copy=True)
    R[key][idx] = val
    return R


@pytest.mark.parametrize("name, R", [
    ("V1_min_scored_60", _mut("crossed", slice(0, 61), False)),
    ("V1_min_scored_60", _mut("follows", slice(0, 61), False)),                      # the follow check (partition)
    ("V2_tsw_before_crossing_90pct_kappa_pos", _mut("t_sw", slice(0, 13), 1200.0)),
    ("V3_regime_eta_lam_le_0p5_in_80pct", _mut("eta_lam", slice(0, 25), 0.6)),
    ("V4_median_abs_predicted_lag_ge_10_steps", _mut("lag_steps", slice(None), 5.0)),
    ("V5_q90_abs_kappa_chi_le_0p1", _mut("r_cf", slice(0, 13), 0.2)),
    ("V6_median_chi_within_30pct_of_pilot", _mut("chi_tsw", slice(None), 0.07)),
    ("V7_q90_max_chi_window_le_0p25", _mut("chi_win_max", slice(0, 13), 0.3)),
    ("V7_q90_max_chi_window_le_0p25", _mut("chi_win_max", 4, np.nan)),
    ("cutoff_before_crossing_90pct", _mut("t_c", slice(0, 13), 1200.0)),
    ("nan_recomputation_identical", _mut("nan_identical", 7, False))])
def test_each_validity_failure_is_unresolved_validity(name, R):
    r = _score(R)
    assert r["validity"][name] is False and r["outcome"] == "UNRESOLVED (validity)"
    assert all(r[k]["verdict"] == "UNRESOLVED" for k in P.CRITERIA["Q"])


def test_v4_and_v5_use_absolute_values_for_negative_lags():
    r = _score(_table("S"), "S")
    assert r["validity"]["V4_median_abs_predicted_lag_ge_10_steps"] and r["validity"]["V5_q90_abs_kappa_chi_le_0p1"]


def test_validity_equals_w2a_registered_function():
    """V1-V7 equal width2_asym.score_arm's on the same scored set (arm T, gate passing)."""
    from src import width2_asym as W2
    rng = np.random.default_rng(7)
    n = 200
    R = _table("Q", n)
    R["eta_lam"] = rng.uniform(0.0001, 0.7, n)
    R["lag_steps"] = rng.normal(15, 20, n)
    R["r_cf"] = rng.normal(0, 0.06, n)
    R["chi_tsw"] = rng.uniform(0.03, 0.07, n)
    R["chi_win_max"] = rng.uniform(0.05, 0.3, n)
    R["t_sw"] = R["t_obs"] - rng.integers(-20, 200, n)
    R["kappa"] = rng.normal(0.01, 0.01, n)
    R["follows"] = rng.random(n) > 0.1
    scored = R["follows"] & R["crossed"]
    V, _ = P.validity_v(scored, R["on_copy"], R["crossed"], R["t_obs"], R["t_sw"], R["kappa"], R["eta_lam"],
                        R["lag_steps"], R["r_cf"], R["chi_tsw"], R["chi_win_max"], 0.05)
    w = W2.score_arm("T", R["on_copy"], R["hold_positive"], R["crossed"], R["t_obs"], R["s_obs"], R["s_sw"],
                     np.where(scored, R["s_obs"], np.nan), R["t_obs"], np.where(scored, 1.0, np.nan), R["r_cf"],
                     R["kappa"], R["t_sw"], R["eta_lam"], R["lag_steps"], R["chi_tsw"], R["chi_win_max"], 0.05,
                     np.full(n, 0.48), follows=R["follows"])
    assert w["validity"] == V


@pytest.mark.parametrize("arm, mut, crit", [
    ("Q", lambda R: (R["t_fc"].__setitem__(slice(0, 30), R["t_fc"][:30] + 20),
                     R["t_sw_fc"].__setitem__(slice(0, 30), R["t_sw_fc"][:30] + 20)), "C1"),
    ("Q", lambda R: R["t_sw_fc"].__setitem__(slice(0, 30), R["t_sw_fc"][:30] + 20), "C2"),
    ("Q", lambda R: R.__setitem__("r_fc", R["r_fc"] / 1.3), "C3"),
    ("S", lambda R: R.__setitem__("t_obs", R["t_obs"] + 60), "S")])
def test_each_criterion_fails_on_its_own(arm, mut, crit):
    R = _table(arm)
    R = {k: np.array(v, copy=True) for k, v in R.items()}
    mut(R)
    r = _score(R, arm)
    assert r["valid"] and r[crit]["verdict"] == "FAIL" and crit in r["outcome"]


def test_c4_fails_when_the_forecast_is_no_better_than_no_lag():
    R = _table()
    R["t_fc"] = R["t_obs"] + 2
    R["t_sw_fc"] = R["t_obs"] + 1                                                # |lag_fc| = 1: C3 has no eligible run
    r = _score(R)
    assert r["C4"]["verdict"] == "FAIL" and r["C3"]["verdict"] == "UNRESOLVED" and r["outcome"] == "UNRESOLVED"


def test_misses_are_scored_and_count_against_c1_and_late_cutoffs_are_unscored():
    R = _table()
    R["t_fc"][:30] = np.nan
    r = _score(R)
    assert r["n_scored"] == 120 and r["n_scored_miss"] == 30 and r["C1"]["verdict"] == "FAIL"
    R = _table()
    R["t_c"][:5] = R["t_obs"][:5]                                                 # not strictly before
    r = _score(R)
    assert r["n_scored"] == 115 and r["n_crossed_cutoff_not_before"] == 5 and r["valid"]


def test_over_cap_and_aborted_runs_are_counted_and_unscored():
    R = _table()
    R["over_cap"][:4] = True
    R["eligible"][:4] = False
    R["crossed"][:4] = False
    R["aborted"][4] = True
    R["eligible"][4] = False
    r = _score(R)
    assert r["n_over_cap_not_trained"] == 4 and r["n_aborted_rss"] == 1 and r["n_scored"] == 115
    assert r["gate"]["n_on_copy"] == 120


def test_follow_decision_keeps_the_partition():
    zc = np.array(P.POP_START["T4"])
    part = P.COPY_PARTITION["T4"]
    assert P.follow_decision(zc + 5e-7, True, zc, True, part)
    assert not P.follow_decision(zc + 2e-6, True, zc, True, part)
    assert not P.follow_decision(zc, False, zc, True, part) and not P.follow_decision(zc, True, zc, False, part)
    z2 = zc.copy()
    z2[1] += 5e-7                                                                 # partition still kept (≤ 1e−6)
    assert P.follow_decision(z2, True, zc, True, part)
    assert not P.follow_decision(zc, True, zc, True, ((0,), (1,), (2, 3)))        # another partition


def test_collapse_table_counts():
    rel = [{"newton_ok": True, "newton_type": "31:+-", "newton_placed": False, "n_groups": 2, "on_copy": True,
            "genuine_four_unit": False, "landing_on_other_copies": {"Q": {"on": False}, "S": {"on": False}}},
           {"newton_ok": True, "newton_type": "1111:++--", "newton_placed": False, "n_groups": 4, "on_copy": False,
            "genuine_four_unit": True, "landing_on_other_copies": {"Q": {"on": True}, "S": {"on": False}}},
           {"newton_ok": True, "newton_type": "22:+-", "newton_placed": True, "n_groups": 2, "on_copy": False,
            "genuine_four_unit": False, "landing_on_other_copies": {}},
           {"newton_ok": False, "newton_type": "not accepted", "genuine_four_unit": False}]
    t = P.collapse_table(rel)
    assert (t["n_random_holds"], t["n_newton_accepted"], t["n_genuine_four_unit_unplaced"], t["n_on_T4"],
            t["n_on_Q_own_point"], t["n_coinciding_unit_points"], t["n_placed"]) == (4, 3, 1, 1, 1, 2, 1)
    assert t["by_type"]["22:+- placed"] == 1 and t["by_type"]["not accepted"] == 1


def test_table_builds_arrays_over_every_seed(monkeypatch):
    monkeypatch.setitem(P.SEEDS, "Q", (1, 2, 3))
    fr = {s: {"release": {"on_copy": s != 3, "hold_G_positive": False}, "budget": {"over_cap": s == 2}}
          for s in (1, 2, 3)}
    runs = [{"seed": 1, "eligible": True, "t_c": 90, "forecast": {"t_fc": 120, "t_sw_fc": 101, "s_fc": 0.5, "r_fc": 0.01,
                                                                  "nan_recompute_identical": True}},
            {"seed": 2, "eligible": False}, {"seed": 3, "eligible": False}]
    obs = [{"seed": 1, "crossed": True, "t_obs": 118, "s_obs": 0.5, "t_sw": 100, "s_sw": 0.49, "follows_branch": True,
            "r_cf": 0.01, "kappa": 0.01, "eta_lam": 1e-4, "lag_steps_pred": 20, "chi_tsw": 0.05, "chi_window_max": 0.06,
            "s_pop": 0.48}]
    R = P.table("Q", fr, runs, obs)
    assert R["on_copy"] == [True, True, False] and R["over_cap"] == [False, True, False]
    assert R["t_fc"][0] == 120 and math.isnan(R["t_fc"][1]) and R["crossed"] == [True, False, False]


# ================================================================================================ causality (width 4)
@pytest.fixture(scope="module")
def pop_case():
    """Population, copy Q, released at its population point and trained at ρ = 2⁻⁷ to t_c (+ 60 rows past it, for the
    poisoning and bypass tests)."""
    if not P3_LAND.exists():
        pytest.skip("landscape.json not present")
    land = json.loads(P3_LAND.read_text())
    x, y = P.population()
    v0 = np.array(land["v0"])
    cf = land["copies"]["Q"]
    copy_row = {"z_s0": cf["z_s0"], "s_switch": cf["switch"]["s_switch"]}
    z_rel = np.array(cf["z_s0"]) + 1e-5
    rho = 2.0 ** -7
    Z, V, t_c, st = P.train_to_cutoff(z_rel, v0, rho, P.F_CUT * copy_row["s_switch"], 20_000, x, y)
    assert st == "ok"
    z, v = Z[-1].copy(), V[-1].copy()
    Zx, Vx = [Z], [V]
    for _ in range(60):
        g_z, g_v = P.grad(z, v, x, y)
        z, v = z - P.ETA * g_z, v - rho * P.ETA * g_v
        Zx.append(z[None]), Vx.append(v[None])
    return np.vstack(Vx), np.vstack(Zx), t_c, copy_row, x, y


def _fc(case, V=None, Z=None):
    Vc, Zc, t_c, cr, x, y = case
    return P.forecast_one(Vc if V is None else V, Zc if Z is None else Z, t_c, 20_000, cr, (0, 0, 0, 0), x, y)


def test_forecast_reads_only_rows_before_its_cutoff(pop_case):
    V, Z, t_c = pop_case[:3]
    rec = _fc(pop_case)
    assert rec["t_c"] == t_c and rec["t_fc"] is not None and rec["t_sw_fc"] is not None, rec["status"]
    assert max(rec["max_index_read"].values()) < t_c and rec["max_index_read"]["z"] == 0
    assert rec["nan_recompute_identical"] is True
    for how in ("nan", "garbage"):
        Vp, Zp = np.array(V, copy=True), np.array(Z, copy=True)
        Vp[t_c:] = np.nan if how == "nan" else 7.7
        Zp[1:] = np.nan if how == "nan" else -3.3                                    # every hidden row but the release
        rec2 = _fc(pop_case, Vp, Zp)
        assert P.same_forecast({k: v for k, v in rec.items() if k != "secs"},
                               {k: v for k, v in rec2.items() if k != "secs"})


def test_forecast_touching_the_cutoff_row_fails(pop_case, monkeypatch):
    real = P.ADAPTERS["w4"]

    def leaky(inp, cfg):
        inp.out[inp.t_c]
        return real(inp, cfg)
    monkeypatch.setitem(P.ADAPTERS, "w4", leaky)
    with pytest.raises(C.CausalityViolation):
        _fc(pop_case)

    def leaky_hidden(inp, cfg):
        inp.hid[1]                                                                    # a hidden row other than the release
        return real(inp, cfg)
    monkeypatch.setitem(P.ADAPTERS, "w4", leaky_hidden)
    with pytest.raises(C.CausalityViolation):
        _fc(pop_case)


def test_forecast_bypassing_the_guard_is_caught_and_invalidates_the_arm(pop_case, monkeypatch):
    real = P.ADAPTERS["w4"]

    def bypass(inp, cfg):
        res = real(inp, cfg)
        o = inp.out
        data = o._GuardedArray__data if isinstance(o, C.GuardedArray) else np.asarray(o)
        val = np.asarray(data[min(inp.t_c + 5, len(data) - 1)], float).sum()
        return {**res, "t_fc": (res["t_fc"] or 0) + (0 if np.isfinite(val) else 1)}
    monkeypatch.setitem(P.ADAPTERS, "w4", bypass)
    rec = _fc(pop_case)
    assert rec["nan_recompute_identical"] is False
    R = _table()
    R["nan_identical"][0] = rec["nan_recompute_identical"]
    assert _score(R)["outcome"] == "UNRESOLVED (validity)"


def test_share_extrapolation_per_unit():
    """Each unit's share is extrapolated by the frozen family (anchored), clipped at 0 and renormalised."""
    cfg = P.config()
    t = np.arange(400, dtype=float)
    s = 0.2 * np.exp(1e-3 * t)
    U = np.column_stack([0.1 + 1e-5 * t, 0.2 - 2e-5 * t, 0.3 + 0.5e-5 * t, 0.4 + 0.5e-5 * t])
    U = U / U.sum(axis=1, keepdims=True)
    V = s[:, None] * U
    Vh, st, info = P.extrapolate_v4(V, cfg, 0.3, 10_000)
    assert st == "ok" and Vh.shape[1] == 4 and np.all(Vh > 0)
    sh = np.abs(Vh).sum(axis=1)
    assert np.allclose((np.abs(Vh) / sh[:, None]).sum(axis=1), 1.0)
    assert sh[-1] >= P.HORIZON * 0.3 and sh[-2] < P.HORIZON * 0.3
    Vbad = V.copy()
    Vbad[-3, 1] = -Vbad[-3, 1]
    assert P.extrapolate_v4(Vbad, cfg, 0.3, 10_000)[1].startswith("no forecast: an output weight changes sign")


# ================================================================================================ end to end (population)
def test_end_to_end_on_the_population(monkeypatch):
    """freeze_seed -> run_one -> observe_one on the POPULATION (own_sample replaced; seed 0, not a Phase 3 seed), copy
    Q at ρ = 2⁻⁷: the run row holds no observed key, the t_c state is re-asserted bit for bit at observation, the
    release is on Q at winding 0 with no G > 0 in the hold."""
    if not P3_LAND.exists():
        pytest.skip("landscape.json not present")
    land = json.loads(P3_LAND.read_text())
    monkeypatch.setattr(P, "own_sample", lambda seed: P.population())
    rho = 2.0 ** -7
    fs = P.freeze_seed("Q", 0, land, rho)
    assert fs["release"]["on_copy"] and fs["release"]["windings"] == [0, 0, 0, 0]
    assert not fs["release"]["hold_G_positive"] and fs["copies"]["Q"]["valid"]
    b = fs["budget"]
    assert b["B"] == P.budget_of(fs["copies"]["Q"]["drive_integral"] / (rho * P.ETA)) and not b["over_cap"]
    rr = P.run_one(fs, rho, land)
    assert rr["eligible"] and rr["t_c"] is not None and not (P.FORBIDDEN & P._keys([rr]))
    assert rr["forecast"]["nan_recompute_identical"] is True
    ob = P.observe_one(fs, rho, rr, land)
    assert ob["state_tc_hash_ok"] is True and ob["crossed"] and ob["t_sw"] is not None and ob["follows_branch"]
    fs_bad = json.loads(json.dumps(fs))
    fs_bad["z_release"][0] += 1e-12
    with pytest.raises(AssertionError):
        P.run_one(fs_bad, rho, land)


# ================================================================================================ frozen artifacts
def test_pilot_json_follows_its_rules():
    p = ROOT / "results" / "phase3" / "pilot.json"
    if not p.exists():
        pytest.skip("pilot.json not present")
    d = json.loads(p.read_text())["arms"]
    for arm in P.ARMS:
        a = d[arm]
        h = a["rule"]["history"]
        assert h[0]["rho"] == P.RHO_START[arm] and a["rho"] == h[-1]["rho"] and h[-1]["q90"] <= 0.1
        assert all(x["q90"] > 0.1 for x in h[:-1])
        fr = a["forecast_runs"]
        t = P.tolerances([r["err_cross"] for r in fr], [r["err_lag"] for r in fr])
        assert (t["tau_cross"], t["tau_lag"]) == (a["tau"]["tau_cross"], a["tau"]["tau_lag"])


def test_frozen_summary_is_consistent():
    p = ROOT / "results" / "phase3" / "frozen.json"
    if not p.exists():
        pytest.skip("frozen.json not present")
    fr = json.loads(p.read_text())
    assert fr["frozen_parts_sha256"] == P._sha(ROOT / "results" / "phase3" / "frozen_parts.jsonl")
    rows = P._frozen_rows()
    by = {a: [r for r in rows if r["arm"] == a] for a in P.ARMS}
    for a in P.ARMS:
        assert sorted(r["seed"] for r in by[a]) == list(P.SEEDS[a])
        g = P.gate(a, [r["release"]["on_copy"] for r in by[a]], [r["release"]["hold_G_positive"] for r in by[a]])
        assert g == fr["arms"][a]["gate_at_freeze"]
        for r in by[a]:
            if r["budget"].get("B") is not None:
                assert r["budget"]["B"] == P.budget_of(r["budget"]["t_star"])
                assert r["budget"]["over_cap"] == P.over_cap(r["budget"]["t_star"])
    assert P.budget_condition(by)["pass"] == fr["budget_condition"]["pass"]
