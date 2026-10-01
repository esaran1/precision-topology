"""Phase 1C (src/phase1c.py): constructed pass / fail / unresolved cases for every gate, criterion and validity rule
(C1-C3, C4 as the paired bootstrap incl. an interval above zero, S, the 90% cutoff rule, the miss rule, the per-run NaN
recomputation, the horizons), the registered settings, and the ENFORCEMENT test: a 1C prediction that touches any row
at or after its cutoff makes this file fail (guard: CausalityViolation; a guard bypass: the NaN recomputation differs and
the arm is not valid)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src import causal_forecast as C
from src import phase1c as P

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "results" / "phase1a" / "fixtures"


# ================================================================================================ registered settings
def test_registered_settings_match_the_approved_page():
    assert P.F == {"W1": 0.90, "G": 0.95, "T": 0.95, "Tp": 0.95}
    assert P.F_OTHER == {"W1": 0.95, "G": 0.90, "T": 0.90, "Tp": 0.90}
    assert P.TAU_CROSS == {"W1": 10, "G": 10, "T": 15, "Tp": 30}
    assert P.TAU_LAG == {"W1": 15, "G": 10, "T": 5, "Tp": 5}
    assert P.BAND == {"W1": 0.20, "G": 0.15, "T": 0.10, "Tp": 0.10}
    assert {a: len(s) for a, s in P.SEEDS.items()} == {"W1": 100, "G": 80, "T": 200, "Tp": 120}
    assert (P.WITHIN_MIN_FRAC, P.S_MIN_FRAC, P.S_MIN_ABS_LAG, P.CUTOFF_BEFORE_MIN_FRAC) == (0.8, 0.8, 6, 0.9)
    assert (P.BOOT_N, P.BOOT_PCT) == (10_000, (2.5, 97.5))
    for arm in P.ARMS:
        cfg = P.config(arm)
        assert (cfg.f, cfg.family, cfg.window_frac, cfg.min_window, cfg.horizon_factor) == (P.F[arm], "quad", 0.05, 50,
                                                                                            1.6)
    from src import phase1a_pilot as PP
    assert PP.A_W1 == 1.58 and PP.W1_LR == 0.3 and PP.W1_BUDGET == 32_000


def test_w1_tolerances_follow_the_stated_rule_from_the_est_seeds():
    """change 1: τ = 1.5 × est-seed q90 rounded up to the next multiple of 5 steps; b = 0.20 (est median ratio 1.17)."""
    from src import phase1a_pilot as PP
    summ = json.loads((ROOT / "results" / "phase1a" / "summary.json").read_text())
    T = PP._tolerances(summ, "w1_sgd_run", 0.9)
    assert T["step_tolerance_cross"] == P.TAU_CROSS["W1"] == 10 and T["step_tolerance_lag"] == P.TAU_LAG["W1"] == 15
    assert T["ratio_band_halfwidth"] == P.BAND["W1"] == 0.20
    assert abs(T["pilot_median_ratio"] - 1.1716) < 1e-3


def test_seed_ranges_disjoint_and_not_registered():
    allseeds = [s for ss in P.SEEDS.values() for s in ss]
    assert len(set(allseeds)) == len(allseeds)
    assert not any(P.registered_overlap().values())


# ================================================================================================ validity: the cutoff
def test_cutoff_before_is_strict_and_no_cutoff_is_not_before():
    b = P.cutoff_before([5, 10, None, 9], [10, 10, 10, None])
    assert b.tolist() == [True, False, False, False]


@pytest.mark.parametrize("k,ok", [(10, True), (9, True), (8, False)])
def test_cutoff_validity_90pct(k, ok):
    base = np.ones(10, bool)
    before = np.arange(10) < k
    cv = P.cutoff_validity(before, base)
    assert cv["ok"] is ok and cv["n_cutoff_not_before_unscored"] == 10 - k


def test_cutoff_validity_only_counts_base_runs_and_fails_with_none():
    base = np.array([True] * 9 + [False])
    before = np.array([True] * 9 + [False])
    assert P.cutoff_validity(before, base)["ok"]
    assert not P.cutoff_validity(np.zeros(3, bool), np.zeros(3, bool))["ok"]


# ================================================================================================ the miss rule
def test_miss_rule():
    assert P.is_forecast([1, None, 3], [0, 0, None]).tolist() == [True, False, False]
    c = P.criterion_within([0, np.nan, 0, 0, 0], 5, [True, False, True, True, True])
    assert c["n"] == 5 and c["n_within"] == 4 and c["n_miss"] == 1 and c["verdict"] == "PASS"
    c = P.criterion_within([0, np.nan, np.nan, 0, 0], 5, [True, False, False, True, True])
    assert c["frac_within"] == 0.6 and c["verdict"] == "FAIL"           # misses count against C1 / C2


# ================================================================================================ C1, C2
@pytest.mark.parametrize("n_in,verdict", [(8, "PASS"), (10, "PASS"), (7, "FAIL")])
def test_c1_c2_within_tau(n_in, verdict):
    err = np.where(np.arange(10) < n_in, 10.0, 10.5)                   # |err| ≤ τ is inclusive
    assert P.criterion_within(err, 10, np.ones(10, bool))["verdict"] == verdict
    assert P.criterion_within(-err, 10, np.ones(10, bool))["verdict"] == verdict      # signed errors, |·|


def test_c1_c2_unresolved_and_undefined_error():
    assert P.criterion_within([], 10, [])["verdict"] == "UNRESOLVED"
    c = P.criterion_within([0.0, np.nan], 10, [True, True])
    assert c["n_err_undefined"] == 1 and c["n_within"] == 1


# ================================================================================================ C3
def test_c3_ratio_band():
    r_fc = np.full(5, 0.1)
    assert P.criterion_ratio(r_fc * 1.2, r_fc, np.ones(5, bool), 0.2)["verdict"] == "PASS"     # closed band
    assert P.criterion_ratio(r_fc * 0.8, r_fc, np.ones(5, bool), 0.2)["verdict"] == "PASS"
    assert P.criterion_ratio(r_fc * 1.21, r_fc, np.ones(5, bool), 0.2)["verdict"] == "FAIL"
    assert P.criterion_ratio(-r_fc, r_fc, np.ones(5, bool), 0.2)["verdict"] == "FAIL"           # sign wrong
    assert P.criterion_ratio(r_fc, r_fc, np.zeros(5, bool), 0.2)["verdict"] == "UNRESOLVED"      # all misses
    c = P.criterion_ratio([0.1, 0.1, 5.0], [0.1, 0.0, 0.1], [True, True, False], 0.1)
    assert c["n"] == 1 and c["n_excluded"] == 2 and c["verdict"] == "PASS"                     # r_fc = 0, miss excluded


# ================================================================================================ C4 (change 4)
def test_c4_pass_when_forecast_beats_no_lag():
    t_obs = np.arange(100, 130, dtype=float)
    c = P.criterion_c4(t_obs + 2, t_obs - 80, t_obs, np.ones(30, bool))
    assert c["verdict"] == "PASS" and c["ci95"][1] < 0 and c["DESCRIPTIVE_n_forecast_closer"] == 30


def test_c4_fail_when_interval_entirely_above_zero():
    t_obs = np.arange(100, 130, dtype=float)
    c = P.criterion_c4(t_obs + 50, t_obs - 1, t_obs, np.ones(30, bool))
    assert c["ci95"][0] > 0 and c["verdict"] == "FAIL"


def test_c4_fail_when_interval_contains_zero():
    t_obs = np.zeros(40)
    rng = np.random.default_rng(1)
    d = rng.choice([-3.0, 3.0], 40)                                     # mean D ≈ 0
    t_fc = np.where(d < 0, 0.0, 6.0)
    t_nolag = np.full(40, 3.0)
    c = P.criterion_c4(t_fc, t_nolag, t_obs, np.ones(40, bool))
    assert c["ci95"][0] < 0 < c["ci95"][1] and c["verdict"] == "FAIL"


def test_c4_unresolved_and_misses_excluded_and_deterministic():
    assert P.criterion_c4([1.0], [5.0], [0.0], [True])["verdict"] == "UNRESOLVED"
    assert P.criterion_c4([1.0, 2.0], [5.0, 5.0], [0.0, 0.0], [True, False])["verdict"] == "UNRESOLVED"
    t_obs = np.arange(20.0)
    a = P.criterion_c4(t_obs + 1, t_obs - 5 + (t_obs % 3), t_obs, np.ones(20, bool))
    b = P.criterion_c4(t_obs + 1, t_obs - 5 + (t_obs % 3), t_obs, np.ones(20, bool))
    assert a["ci95"] == b["ci95"] and a["boot_seed"] == P.BOOT_SEED


def test_c4_bootstrap_matches_registered_procedure():
    from src import width2_asym as W
    D = np.random.default_rng(3).normal(-1, 2, 50)
    assert P.bootstrap_mean_ci(D, seed=884_000) == W.bootstrap_mean_ci(D, seed=884_000)


# ================================================================================================ S (T′)
def test_sign_criterion():
    lf = np.array([-10, -10, -10, -10, -10, 3, -2])
    lo = np.array([-8, -8, -8, -8, 2, -8, 5])
    c = P.criterion_sign(lf, lo, np.ones(7, bool))
    assert c["n_eligible"] == 5 and c["n_both_negative"] == 4 and c["verdict"] == "PASS"    # |lag_fc| < 6 ignored
    c = P.criterion_sign(lf, np.array([-8, -8, -8, 2, 2, -8, 5]), np.ones(7, bool))
    assert c["verdict"] == "FAIL"
    assert P.criterion_sign([3, -5], [-1, -1], [True, True])["verdict"] == "UNRESOLVED"     # no eligible run


def test_sign_criterion_misses_count_as_failures():
    """Author's decision 2026-10-01: a miss enters S's denominator as not both negative."""
    lf = np.array([-10.0] * 8 + [np.nan] * 2)
    lo = np.full(10, -8.0)
    has = np.array([True] * 8 + [False] * 2)
    c = P.criterion_sign(lf, lo, has)
    assert c["n_eligible"] == 10 and c["n_eligible_miss"] == 2 and c["frac"] == 0.8 and c["verdict"] == "PASS"
    has3 = np.array([True] * 7 + [False] * 3)
    lf3 = np.where(has3, -10.0, np.nan)
    c3 = P.criterion_sign(lf3, lo, has3)
    assert c3["frac"] == 0.7 and c3["verdict"] == "FAIL"                 # a miss lowers the fraction
    c0 = P.criterion_sign([np.nan, np.nan], [-1.0, -1.0], [False, False])
    assert c0["n_eligible"] == 2 and c0["verdict"] == "FAIL"             # only misses: eligible, failures
    assert P.criterion_sign([], [], [])["verdict"] == "UNRESOLVED"       # UNRESOLVED only with no eligible run


# ================================================================================================ horizons, outcome
def test_horizons_in_steps_and_lags():
    h = P.horizons([100, 200], [300, 260], [220, 250], [80, 10])
    assert h["cross_steps"]["min"] == 60 and h["cross_steps"]["max"] == 200
    assert h["switch_steps"]["median"] == 85
    assert h["cross_lags"]["min"] == 200 / 80 and h["cross_lags"]["max"] == 6.0
    assert P.horizons([100], [101], [101], [0])["cross_lags"]["max"] == 1.0                     # lag 0: ÷ 1


def test_outcome_rule():
    assert P.outcome({"C1": "PASS", "C2": "PASS"}) == "PASS"
    assert P.outcome({"C1": "FAIL", "C2": "UNRESOLVED"}) == "UNRESOLVED"
    assert P.outcome({"C1": "FAIL", "C2": "PASS", "C4": "FAIL"}) == "FAIL C1+C4"


def test_phase2_gate():
    s = lambda w, g, t: {"W1": {"outcome": w}, "G": {"outcome": g}, "T": {"outcome": t}, "Tp": {"outcome": "FAIL S"}}  # noqa
    assert P.phase2_gate(s("PASS", "PASS", "FAIL C1"))["pass"]
    assert P.phase2_gate(s("PASS", "UNRESOLVED", "PASS"))["pass"]
    assert not P.phase2_gate(s("PASS", "FAIL C4", "UNRESOLVED (gate)"))["pass"]
    assert not P.phase2_gate(s("UNRESOLVED (validity)", "PASS", "PASS"))["pass"]


def test_same_forecast():
    a = {"t_fc": 5, "r_fc": float("nan"), "x": [1.0, 2.0], "max_index_read": {"w2": 3}}
    assert P.same_forecast(a, {**a, "max_index_read": {"w2": 9}})
    assert not P.same_forecast(a, {**a, "t_fc": 6})
    assert not P.same_forecast(a, {**a, "x": [1.0, 2.5]})
    assert not P.same_forecast(a, {k: v for k, v in a.items() if k != "x"})


# ================================================================================================ score_arm_1c: W1
def _w1_table(n=80, n_cross=70, miss=0, late=0, nan_bad=0, err=2, lag_err=3):
    t_obs = np.full(n, 1000.0)
    t_obs[n_cross:] = np.nan
    t_sw = t_obs - 80
    t_c = t_obs - 180
    t_c[:late] = t_obs[:late] + 1
    t_fc = t_obs + err
    t_sw_fc = t_fc - 80 - lag_err
    t_fc[late:late + miss] = np.nan
    s_sw = np.full(n, 2.0)
    s_obs = np.where(np.isfinite(t_obs), 2.2, np.nan)
    r_fc = np.full(n, 0.1 / 1.1)
    nan_ok = np.ones(n, bool)
    nan_ok[:nan_bad] = False
    return {"crossed": np.isfinite(t_obs), "t_obs": t_obs, "s_obs": s_obs, "t_sw": t_sw, "s_sw": s_sw, "t_c": t_c,
            "t_fc": t_fc, "t_sw_fc": t_sw_fc, "s_fc": np.full(n, 2.2), "r_fc": r_fc, "nan_identical": nan_ok,
            "hold_positive": np.zeros(n, bool), "frozen_ok": np.ones(n, bool), "r_cf": np.full(n, 0.1)}


def test_w1_pass():
    o = P.score_arm_1c("W1", _w1_table())
    assert o["outcome"] == "PASS" and o["n_scored"] == 70 and o["C3"]["median_ratio"] == pytest.approx(1.1)


def test_w1_fail_c1_and_c2():
    o = P.score_arm_1c("W1", _w1_table(err=11, lag_err=16))
    assert o["outcome"] == "FAIL C1+C2"


def test_w1_unresolved_fewer_than_60_scored():
    o = P.score_arm_1c("W1", _w1_table(n_cross=59))
    assert o["outcome"] == "UNRESOLVED (validity)" and o["C1"]["verdict"] == "UNRESOLVED"
    assert o["C1"]["verdict_if_valid"] == "PASS"


def test_w1_misses_are_scored_and_count_against_c1():
    o = P.score_arm_1c("W1", _w1_table(miss=15))
    assert o["n_scored"] == 70 and o["n_scored_miss"] == 15 and o["C1"]["verdict"] == "FAIL"
    assert o["C3"]["n"] == 55 and o["C4"]["n"] == 55                     # misses not in C3 / C4


def test_w1_cutoff_rule_90pct():
    assert P.score_arm_1c("W1", _w1_table(late=7))["outcome"] == "PASS"                 # 63/70 = 0.9
    o = P.score_arm_1c("W1", _w1_table(late=8))
    assert o["outcome"] == "UNRESOLVED (validity)" and o["n_crossed_cutoff_not_before"] == 8


def test_w1_nan_recomputation_differs_is_not_valid():
    o = P.score_arm_1c("W1", _w1_table(nan_bad=1))
    assert not o["validity"]["nan_recomputation_identical"] and o["outcome"] == "UNRESOLVED (validity)"


def test_w1_c4_fails_when_no_better_than_no_lag():
    T = _w1_table()
    T["t_sw_fc"] = T["t_fc"] - 1                                         # forecast lag 1 step: t_nolag ≈ t_fc
    T["t_fc"] = T["t_obs"] + 5
    o = P.score_arm_1c("W1", T)
    assert o["C4"]["verdict"] == "FAIL" and "C4" in o["outcome"]


# ================================================================================================ score_arm_1c: G, T, T′
def _reg_table(arm, n, n_on, n_cross, miss=0, late=0, hold=0, follows_bad=0, lag=-30.0, chi=None):
    from src import gelu_transfer as G
    from src import width2_asym as W
    pc = (json.loads((G.OUT / "pilot.json").read_text())["pilot_median_chi_tsw"]["random"] if arm == "G" else
          json.loads((W.OUT / "pilot.json").read_text())["arms"][arm]["pilot_median_chi_tsw"])
    chi = pc if chi is None else chi
    on = np.arange(n) < n_on
    t_obs = np.where(np.arange(n) < n_cross, 5000.0, np.nan)
    crossed = np.isfinite(t_obs)
    t_sw = t_obs - lag if arm == "Tp" else t_obs - 20
    t_c = t_obs - 900
    t_c[:late] = t_obs[:late]
    t_fc = t_obs + 1
    t_sw_fc = t_sw + 1
    t_fc[late:late + miss] = np.nan
    s_sw = np.full(n, 1.0)
    rf = -0.002 if arm == "Tp" else 0.002
    kap = -0.1 if arm == "Tp" else 0.1
    R = {"crossed": crossed, "t_obs": t_obs, "s_obs": np.where(crossed, 1 + rf, np.nan), "t_sw": t_sw, "s_sw": s_sw,
         "t_c": t_c, "t_fc": t_fc, "t_sw_fc": t_sw_fc, "s_fc": np.full(n, 1 + rf), "r_fc": np.full(n, rf),
         "nan_identical": np.ones(n, bool), "hold_positive": np.arange(n) < hold, "on_branch": on,
         "follows": np.arange(n) >= follows_bad, "r_cf": np.full(n, kap * chi), "kappa": np.full(n, kap),
         "eta_lam": np.full(n, 0.1), "lag_steps": np.full(n, kap / 0.1 * 100), "chi_tsw": np.full(n, chi),
         "chi_win_max": np.full(n, 0.05), "s_pop": np.full(n, 0.5)}
    return R, pc, (6.64 if arm == "G" else R["s_pop"])


@pytest.mark.parametrize("arm,n,n_on", [("G", 80, 80), ("T", 200, 100), ("Tp", 120, 120)])
def test_registered_arms_pass(arm, n, n_on):
    R, pc, sp = _reg_table(arm, n, n_on, n_on)
    o = P.score_arm_1c(arm, R, pilot_chi=pc, s_pop=sp)
    assert o["gate"]["pass"] and all(o["validity"].values()), o["validity"]
    assert o["outcome"] == "PASS" and o["n_scored"] == n_on
    if arm == "Tp":
        assert o["S"]["verdict"] == "PASS"


@pytest.mark.parametrize("arm,n,n_on,hold", [("G", 80, 63, 0), ("T", 200, 59, 0), ("Tp", 120, 107, 0),
                                             ("Tp", 120, 120, 1)])
def test_registered_gates_unchanged(arm, n, n_on, hold):
    R, pc, sp = _reg_table(arm, n, n_on, n_on, hold=hold)
    o = P.score_arm_1c(arm, R, pilot_chi=pc, s_pop=sp)
    assert not o["gate"]["pass"] and o["outcome"] == "UNRESOLVED (gate)" and o["C1"]["verdict"] == "UNRESOLVED"


def test_g_hold_positive_does_not_fail_the_random_gate():
    R, pc, sp = _reg_table("G", 80, 80, 80, hold=5)
    o = P.score_arm_1c("G", R, pilot_chi=pc, s_pop=sp)
    assert o["gate"]["pass"] and o["outcome"] == "PASS"
    assert o["DESCRIPTIVE"]["sensitivity_excluding_hold_G_positive"]["n_scored"] == 75


def test_v1_counts_misses_and_follow_failures_are_not_scored():
    R, pc, sp = _reg_table("T", 200, 70, 70, miss=8, follows_bad=5)
    o = P.score_arm_1c("T", R, pilot_chi=pc, s_pop=sp)
    assert o["n_scored"] == 65 and o["n_scored_miss"] == 3 and o["validity"]["V1_min_scored_60"]
    R, pc, sp = _reg_table("T", 200, 64, 64, follows_bad=5)
    o = P.score_arm_1c("T", R, pilot_chi=pc, s_pop=sp)
    assert not o["validity"]["V1_min_scored_60"] and o["outcome"] == "UNRESOLVED (validity)"


def test_registered_v6_unchanged_and_late_cutoffs_unscored():
    R, pc, sp = _reg_table("G", 80, 80, 80, chi=None)
    R["chi_tsw"] = R["chi_tsw"] * 1.31
    o = P.score_arm_1c("G", R, pilot_chi=pc, s_pop=sp)
    assert not o["validity"]["V6_median_chi_within_30pct_of_pilot"] and o["outcome"] == "UNRESOLVED (validity)"
    R, pc, sp = _reg_table("G", 80, 80, 80, late=9)
    o = P.score_arm_1c("G", R, pilot_chi=pc, s_pop=sp)
    assert o["n_scored"] == 71 and not o["validity"]["cutoff_before_crossing_90pct"]


def test_validity_with_placeholders_equals_registered_validity():
    """The registered V1-V7 depend on the scored set only, not on the prediction values: the 1C placeholder call gives
    the registered function's own validity for the same runs."""
    from src import width2_asym as W
    R, pc, sp = _reg_table("T", 200, 100, 100)
    o = P.score_arm_1c("T", R, pilot_chi=pc, s_pop=sp)
    rnd = np.random.default_rng(0).normal(0.002, 0.001, 200)
    reg = W.score_arm("T", R["on_branch"], R["hold_positive"], R["crossed"], R["t_obs"], R["s_obs"], R["s_sw"],
                      R["s_obs"], R["t_obs"], rnd, R["r_cf"], R["kappa"], R["t_sw"], R["eta_lam"], R["lag_steps"],
                      R["chi_tsw"], R["chi_win_max"], pc, sp, follows=R["follows"])
    assert {k: v for k, v in o["validity"].items() if k.startswith("V")} == reg["validity"]


def test_tp_sign_fail():
    R, pc, sp = _reg_table("Tp", 120, 120, 120)
    R["t_sw"] = R["t_obs"] - 30                                          # observed lag positive
    R["t_sw_fc"] = R["t_sw"] + 1
    R["t_fc"] = R["t_sw_fc"] - 30                                        # forecast lag negative
    o = P.score_arm_1c("Tp", R, pilot_chi=pc, s_pop=sp)
    assert o["S"]["verdict"] == "FAIL" and "S" in o["outcome"]


def test_runs_rows_never_hold_observed_quantities():
    assert P._keys([{"forecast": {"t_fc": 1, "nested": [{"t_obs": 3}]}}]) & P.FORBIDDEN == {"t_obs"}


# ================================================================================================ ENFORCEMENT
def _fixture(name):
    p = FIX / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    d = np.load(p, allow_pickle=False)
    return d, json.loads(str(d["meta"]))


def _fixture_case(arm):
    name = {"G": "gelu_random", "T": "w2a_T"}[arm]
    d, meta = _fixture(name)
    row = meta["row"]
    out, hid_rows, hid = np.array(d["out"]), np.array(d["hid_rows"]), np.array(d["hid"])
    return row, out, hid_rows, hid, P.context(arm, row)


@pytest.fixture(scope="module")
def w1_case():
    from src import phase1a_pilot as PP
    seed = 9_310_002                                                      # a Phase 1A dev seed (not a 1C seed)
    fr = PP.w1_frozen(seed)
    W, _, _ = PP.w1_train(seed, budget=2500)
    s = np.abs(W[:, 2])
    rr = PP.w1_rule_rows(s, fr["s_frozen"])
    row = {"seed": seed, "s_ref": fr["s_frozen"], "budget": PP.W1_BUDGET}
    return row, W[:, 2].copy(), rr, W[rr][:, [0, 1, 3]], P.context("W1", row)


def _case(arm, w1_case):
    return w1_case if arm == "W1" else _fixture_case(arm)


def _poison(a, t_c, how):
    b = np.array(a, float, copy=True)
    b[t_c:] = np.nan if how == "nan" else 1e3 * (1 + np.arange(len(b) - t_c)).reshape(-1, *([1] * (b.ndim - 1)))
    return b


@pytest.mark.parametrize("arm", ["W1", "G", "T"])
def test_1c_prediction_reads_only_rows_before_its_cutoff(arm, w1_case):
    row, out, hid_rows, hid, ctx = _case(arm, w1_case)
    rec = P.forecast_one(arm, row, out, hid_rows, hid, ctx, P.F[arm])
    assert rec["t_c"] is not None and rec["status"] == "ok" and rec["t_fc"] is not None
    assert max(rec["max_index_read"].values()) < rec["t_c"]
    assert rec["nan_recompute_identical"] is True
    for how in ("nan", "garbage"):          # rows > t_c poisoned (the harness reads s at t_c to stop): the same prediction
        hp = np.array(hid, float, copy=True)
        hp[np.asarray(hid_rows) > rec["t_c"]] = np.nan if how == "nan" else 7.7
        rec2 = P.forecast_one(arm, row, _poison(out, rec["t_c"] + 1, how), hid_rows, hp, ctx, P.F[arm])
        assert P.same_forecast({k: v for k, v in rec.items() if k != "secs"},
                               {k: v for k, v in rec2.items() if k != "secs"})


@pytest.mark.parametrize("arm", ["W1", "G", "T"])
def test_1c_prediction_touching_the_cutoff_row_fails(arm, w1_case, monkeypatch):
    row, out, hid_rows, hid, ctx = _case(arm, w1_case)
    kind = P.KIND[arm]
    real = C.ADAPTERS[kind]

    def leaky(inp, cfg):
        inp.out[inp.t_c]                                                  # the row at the cutoff
        return real(inp, cfg)
    monkeypatch.setitem(C.ADAPTERS, kind, leaky)
    with pytest.raises(C.CausalityViolation):
        P.forecast_one(arm, row, out, hid_rows, hid, ctx, P.F[arm])


@pytest.mark.parametrize("arm", ["W1", "G", "T"])
def test_1c_prediction_bypassing_the_guard_is_caught_and_invalidates_the_arm(arm, w1_case, monkeypatch):
    """A forecaster that reads the private array behind the guard (a row after the cutoff) is not caught by the guard,
    but its NaN recomputation differs; score_arm_1c then makes the arm not valid."""
    row, out, hid_rows, hid, ctx = _case(arm, w1_case)
    kind = P.KIND[arm]
    real = C.ADAPTERS[kind]

    def bypass(inp, cfg):
        res = real(inp, cfg)
        o = inp.out
        data = o._GuardedArray__data if isinstance(o, C.GuardedArray) else np.asarray(o)
        v = np.asarray(data[min(inp.t_c + 5, len(data) - 1)], float).sum()
        return {**res, "t_fc": (res["t_fc"] or 0) + (0 if np.isfinite(v) else 1)}
    monkeypatch.setitem(C.ADAPTERS, kind, bypass)
    rec = P.forecast_one(arm, row, out, hid_rows, hid, ctx, P.F[arm])
    assert rec["nan_recompute_identical"] is False
    T = _w1_table()
    T["nan_identical"][0] = rec["nan_recompute_identical"]
    assert P.score_arm_1c("W1", T)["outcome"] == "UNRESOLVED (validity)"


def test_harness_cutoff_is_a_stopping_time(w1_case):
    row, out, hid_rows, hid, ctx = w1_case
    t_c, ex = P.harness_cutoff("W1", row, out, hid_rows, hid, ctx, P.F["W1"])
    assert t_c is not None and ex["cutoff_rule_point"] < t_c
    hp = np.array(hid, float, copy=True)
    hp[np.asarray(hid_rows) > t_c] = np.nan
    assert P.harness_cutoff("W1", row, _poison(out, t_c + 1, "nan"), hid_rows, hp, ctx, P.F["W1"]) == (t_c, ex)


def test_pilot_check_reproduces_the_page_table():
    p = ROOT / "results" / "phase1c" / "pilot_check.json"
    if not p.exists():
        pytest.skip("pilot_check.json not present")
    d = json.loads(p.read_text())["arms"]
    exp = {"W1": (32, 32, 1.172), "G": (19, 19, 0.942), "T": (26, 26, 0.975), "Tp": (20, 20, 1.019)}
    for arm, (c1, c2, m) in exp.items():
        c = d[arm][str(P.F[arm])]["criteria"]
        assert c["C1"]["n_within"] == c1 and c["C2"]["n_within"] == c2 and round(c["C3"]["median_ratio"], 3) == m
        assert all(v["verdict"] == "PASS" for v in c.values())
