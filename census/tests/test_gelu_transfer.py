"""GELU-T (registered lag-law test at GELU under free SGD, two arms): every decision rule on constructed pass, fail and
unresolved cases: hold length, copy assignment, release classification (target, mirror, neither), gates, t_sw, the
closed form and χ path, the pilot rule, the bootstrap rule, every validity condition, L1-L5 and the outcome."""

import math

import numpy as np
import pytest

from src import gelu_transfer as X


# ------------------------------------------------------------------------------------------ registered constants
def test_registered_constants():
    assert X.SEEDS == tuple(range(876_000, 876_080)) and X.PILOT_SEEDS == tuple(range(876_900, 876_910))
    assert not set(X.SEEDS) & set(X.PILOT_SEEDS)
    assert (X.ETA, X.BUDGET, X.HOLD_LR, X.W_MIN, X.W_RELAX, X.S0_FRAC) == (0.03, 40_000, 0.3, 4000, 25.0, 0.5)
    assert (X.NEWTON_GTOL, X.ON_TOL, X.STATE_TOL) == (1e-8, 1e-6, 1e-3)
    assert X.GATE_FRAC == {"random": 0.80, "branch": 0.90} and X.ARMS == ("random", "branch")
    assert (X.L1_BAND, X.L2_BAND, X.L3_MIN) == ((0.90, 1.10), (0.80, 1.20), 0.5)
    assert (X.BOOT_N, X.BOOT_PCT) == (10_000, (2.5, 97.5))
    assert (X.MIN_CROSS, X.TSW_MIN_FRAC, X.REGIME_MAX, X.REGIME_MIN_FRAC, X.LAG_MIN_STEPS, X.KC_Q90_MAX,
            X.CHI_REL_TOL, X.CHI_PATH_Q90_MAX) == (60, 0.90, 0.5, 0.80, 10.0, 0.1, 0.30, 0.25)
    assert (X.PILOT_KC_MAX, X.PILOT_MAX_HALVINGS) == (0.1, 3)
    assert X.FOLLOW_FRAC == 0.8 and X.FOLLOW_STATE_TOL is None              # the Newton condition only (§13)


def test_s_values_come_from_the_registered_3a_file():
    s_pop, s_glob, s0 = X.s_values()
    assert s_pop == pytest.approx(6.645633, abs=5e-7) and s_glob == pytest.approx(6.641492, abs=5e-7)
    assert s0 == 0.5 * s_pop


# ------------------------------------------------------------------------------------------ hold, copies, init
def test_w_hold():
    assert X.w_hold(0.035) == 4000                                  # 25/(0.3·0.035) = 2381 < 4000
    assert X.w_hold(25 / (0.3 * 5000)) == 5000                      # exactly 5000 steps
    assert X.w_hold(0.01) == math.ceil(25 / 0.003)
    assert X.w_hold(0.0) is None and X.w_hold(-1.0) is None and X.w_hold(float("nan")) is None


def test_assigned_copy_by_parity_and_arm_hold_steps():
    assert X.assigned_copy(876_000) == 1 and X.assigned_copy(876_001) == -1
    W = {1: 4000, -1: 6000}
    assert X.arm_hold_steps("branch", 876_000, W) == 4000 and X.arm_hold_steps("branch", 876_001, W) == 6000
    assert X.arm_hold_steps("random", 876_000, W) == 6000                  # the larger of the two copies
    assert X.arm_hold_steps("random", 876_000, {1: 4000, -1: None}) is None


def test_random_init_is_3a_draw_without_touching_global_state():
    import torch
    before = torch.random.get_rng_state().clone()
    th = X.init_state("random", 876_123, [1.0, -1.0, 0.0], 3.3)
    assert torch.equal(before, torch.random.get_rng_state())            # no global RNG change
    with torch.random.fork_rng():
        torch.manual_seed(876_123)
        ref = torch.empty(4).uniform_(-1.0, 1.0).double().numpy()        # Track 3A's draw
    assert th[0] == ref[0] and th[1] == ref[1] and th[3] == ref[3] and th[2] == 3.3
    assert th.dtype == np.float64


def test_branch_init_parity_mirror():
    z = [1.2, -0.9, -0.1]
    assert list(X.init_state("branch", 876_010, z, 3.3)) == [1.2, -0.9, 3.3, -0.1]
    assert list(X.init_state("branch", 876_011, z, 3.3)) == [-1.2, -0.9, 3.3, -0.1]


# ------------------------------------------------------------------------------------------ release classification
COP = {1: np.array([1.5, -1.0, -0.2]), -1: np.array([-1.4, -1.0, -0.2])}


def test_classify_target_mirror_neither():
    z = COP[1]
    assert X.classify_release(z + 5e-4, z + 5e-7, True, COP) == 1
    assert X.classify_release(COP[-1] - 5e-4, COP[-1], True, COP) == -1
    assert X.classify_release(z, z, False, COP) == 0                          # Newton not accepted
    assert X.classify_release(z, z + 2e-6, True, COP) == 0                    # Newton point off the frozen point
    assert X.classify_release(z + 2e-3, z, True, COP) == 0                    # release state too far (> 1e-3)
    assert X.classify_release([0.0, -1.0, -0.2], [0.0, -1.0, -0.2], True, COP) == 0   # a third point (w₁ = 0)
    assert X.classify_release(z, z, True, {1: None, -1: COP[-1]}) == 0      # copy without a frozen point


def test_on_branch_rule_per_arm():
    assert X.is_on_branch("random", 876_000, 1) and X.is_on_branch("random", 876_000, -1)
    assert not X.is_on_branch("random", 876_000, 0)
    assert X.is_on_branch("branch", 876_000, 1) and not X.is_on_branch("branch", 876_000, -1)
    assert X.is_on_branch("branch", 876_001, -1) and not X.is_on_branch("branch", 876_001, 1)
    assert not X.is_on_branch("branch", 876_001, 0)


# ------------------------------------------------------------------------------------------ gates
def test_gate_primary_edges():
    on = np.zeros(80, bool); on[:64] = True                                 # 64/80 = 0.80
    g = X.gate("random", on, np.zeros(80, bool))
    assert g["pass"] and g["frac_on_branch"] == 0.8
    on[63] = False
    assert not X.gate("random", on, np.zeros(80, bool))["pass"]
    on[63] = True
    hp = np.zeros(80, bool); hp[:30] = True                                 # G > 0 in the hold: no effect on the primary
    g = X.gate("random", on, hp)
    assert g["pass"] and g["n_hold_G_positive"] == 30 and not g["hold_condition_applies"]


def test_gate_control_edges():
    on = np.zeros(80, bool); on[:72] = True                                 # 72/80 = 0.90
    assert X.gate("branch", on, np.zeros(80, bool))["pass"]
    hp = np.zeros(80, bool); hp[75] = True                                  # the control keeps the hold condition
    g = X.gate("branch", on, hp)
    assert not g["pass"] and g["frac_ok"] and not g["hold_ok"] and g["hold_condition_applies"]
    on[71] = False
    assert not X.gate("branch", on, np.zeros(80, bool))["pass"]
    assert not X.gate("random", np.zeros(0, bool), np.zeros(0, bool))["pass"]


# ------------------------------------------------------------------------------------------ t_sw, closed form, χ path
def test_t_switch():
    s = np.array([3.3, 3.5, 6.0, 6.7, 6.6, 7.0])
    assert X.t_switch(s, 6.7) == 3 and X.t_switch(s, 6.65) == 3 and X.t_switch(s, 3.3) == 0
    assert X.t_switch(s, 8.0) is None and X.t_switch(s, None) is None and X.t_switch(s, float("nan")) is None


def test_closed_form_matches_hand_computation():
    s = 3.0 + 0.001 * np.arange(5000)
    t_sw = X.t_switch(s, 6.0)
    assert t_sw == 3000
    r, chi, sdot = X.closed_form(s, t_sw, 6.0, 0.13, 0.1)
    assert sdot == pytest.approx(0.001)
    assert chi == pytest.approx((0.001 / 6.0) / (0.03 * 0.1))
    assert r == pytest.approx(0.13 * chi)
    r, chi, sdot = X.closed_form(s, 40, 3.04, 0.13, 0.1)                   # window min(100, t_sw) = 40
    assert sdot == pytest.approx(0.001)
    assert all(math.isnan(v) for v in X.closed_form(s, None, 6.0, 0.13, 0.1))
    assert all(math.isnan(v) for v in X.closed_form(s, 0, 6.0, 0.13, 0.1))


def test_chi_path():
    s = np.array([1.0, 1.1, 1.3, 1.4])
    cp = X.chi_path(s, 3, lambda v: 2.0)
    assert np.allclose(cp, [0.1 / 1.0 / 0.06, 0.2 / 1.1 / 0.06, 0.1 / 1.3 / 0.06])
    assert len(X.chi_path(s, None, lambda v: 1.0)) == 0


# ------------------------------------------------------------------------------------------ pilot rule
def test_pilot_rho_formula():
    assert X.pilot_rho(0.003) == 1.0 and X.pilot_rho(0.1) == 1.0 and X.pilot_rho(0.0) == 1.0
    assert X.pilot_rho(0.11) == 0.5                                        # 0.1/0.11 = 0.909 → 2^-1
    assert X.pilot_rho(0.3) == 0.25 and X.pilot_rho(0.2) == 0.5 and X.pilot_rho(0.41) == 0.125


def test_pilot_rule_inactive_runs_once():
    calls = []
    r = X.pilot_rule(lambda rho: calls.append(rho) or [0.002, -0.003, 0.004])
    assert r["status"] == "ok" and r["rho"] == 1.0 and calls == [1.0]


def test_pilot_rule_uses_signed_kappa_chi():
    r = X.pilot_rule(lambda rho: [0.01] * 8 + [-0.3 * rho] * 2)            # signed: negative values do not raise q90
    assert r["status"] == "ok" and r["rho"] == 1.0 and r["history"][0]["q90_kc"] == pytest.approx(0.01)
    r = X.pilot_rule(lambda rho: [0.01] * 8 + [0.3 * rho] * 2)
    assert r["status"] == "ok" and r["rho"] == 0.25


def test_pilot_rule_active_keeps_rho_after_rerun():
    calls = []
    # κχ ∝ ρ: at ρ = 1 the q90 is 0.3 → ρ = 0.25; the rerun gives 0.075 ≤ 0.1: kept
    r = X.pilot_rule(lambda rho: calls.append(rho) or list(0.3 * rho * np.ones(10)))
    assert r["status"] == "ok" and r["rho"] == 0.25 and calls == [1.0, 0.25]


def test_pilot_rule_halves_then_stops():
    calls = []
    r = X.pilot_rule(lambda rho: calls.append(rho) or [0.5] * 10)           # never ≤ 0.1
    assert r["status"] == "STOP" and r["rho"] is None
    assert calls == [1.0, 0.125, 0.0625, 0.03125, 0.015625]                  # rerun at ρ, then three halvings
    calls = []
    seq = iter([[0.5] * 10, [0.2] * 10, [0.15] * 10, [0.09] * 10])         # passes after two halvings
    r = X.pilot_rule(lambda rho: calls.append(rho) or next(seq))
    assert r["status"] == "ok" and r["rho"] == 0.125 / 4 and len(calls) == 4


def test_pilot_rule_stops_without_values():
    r = X.pilot_rule(lambda rho: [])
    assert r["status"] == "STOP"


# ------------------------------------------------------------------------------------------ bootstrap rule
def test_bootstrap_is_deterministic_and_brackets_the_mean():
    rng = np.random.default_rng(3)
    D = rng.normal(-0.01, 0.005, 70)
    a, b = X.bootstrap_mean_ci(D), X.bootstrap_mean_ci(D)
    assert a == b and a[1] < a[0] < a[2] < 0
    m, lo, hi = X.bootstrap_mean_ci(np.full(10, -0.2))
    assert m == lo == hi == pytest.approx(-0.2)
    m, lo, hi = X.bootstrap_mean_ci(rng.normal(0.0, 0.01, 70))
    assert lo < 0 < hi or hi >= 0


# ------------------------------------------------------------------------------------------ scoring
S_GLOB = 6.641492


def _case(n=80, seed=0, ratio=1.02, noise=0.02, n_on=80, n_cross=80, lag=0.003):
    """Constructed runs: own switches spread around s_glob; lag r_pred = κχ ≈ lag; obs = pred·ratio·(1 + noise)."""
    rng = np.random.default_rng(seed)
    s_sw = S_GLOB * rng.uniform(0.8, 1.3, n)
    r_traj = lag * rng.uniform(0.7, 1.3, n)
    r_cf = r_traj * 1.01
    r_obs = r_traj * ratio * (1 + noise * rng.uniform(-1, 1, n))
    on = np.zeros(n, bool); on[:n_on] = True
    crossed = np.zeros(n, bool); crossed[:n_cross] = True
    return dict(arm="random", on_branch=on, hold_positive=np.zeros(n, bool), crossed=crossed,
                step_obs=np.where(crossed, 6000.0, np.nan), s_obs=np.where(crossed, s_sw * (1 + r_obs), np.nan),
                s_sw=s_sw, s_traj=s_sw * (1 + r_traj), r_traj=r_traj, r_cf=r_cf, kappa=np.full(n, 0.13),
                t_sw=np.full(n, 5950.0), eta_lam=np.full(n, 0.003), lag_steps=np.full(n, 40.0),
                chi_tsw=np.full(n, 0.022), chi_path_max=np.full(n, 0.05), pilot_chi_median=0.022, s_glob=S_GLOB)


def _v(s):
    return tuple(s[k]["verdict"] for k in ("L1", "L2", "L3", "L4", "L5"))


def test_all_pass():
    s = X.score_arm(**_case())
    assert s["gate"]["pass"] and s["valid"] and _v(s) == ("PASS",) * 5 and s["outcome"] == "PASS"
    assert s["n_scored"] == 80 and s["L4"]["ci95"][1] < 0 and s["L5"]["ci95"][1] < 0


def test_gate_fail_makes_everything_unresolved_primary_and_control():
    c = _case(n_on=63)
    s = X.score_arm(**c)
    assert s["outcome"] == "UNRESOLVED (gate)" and _v(s) == ("UNRESOLVED",) * 5 and s["scored_index"] == []
    c = _case(); c["hold_positive"] = c["hold_positive"].copy(); c["hold_positive"][5] = True
    s = X.score_arm(**c)                                                    # primary: flagged, scored, gate passes
    assert s["outcome"] == "PASS" and s["n_scored"] == 80 and s["n_on_branch_hold_G_positive"] == 1
    c["arm"] = "branch"
    s = X.score_arm(**c)
    assert s["outcome"] == "UNRESOLVED (gate)" and not s["gate"]["hold_ok"]
    c = _case(n_on=71); c["arm"] = "branch"                                 # 71/80 < 0.90 for the control
    assert X.score_arm(**c)["outcome"] == "UNRESOLVED (gate)"
    c = _case(n_on=72); c["arm"] = "branch"
    s = X.score_arm(**c)
    assert s["gate"]["pass"] and s["n_scored"] == 72 and s["role"] == "mechanism control"


def test_neither_runs_are_counted_not_scored():
    c = _case(n_on=66)
    c["s_obs"] = c["s_obs"].copy(); c["s_obs"][66:] = 100.0                 # would fail everything if scored
    s = X.score_arm(**c)
    assert s["n_scored"] == 66 and s["n_neither_or_off_target"] == 14 and s["outcome"] == "PASS"


def test_L1_fails_alone():
    c = _case(ratio=1.12)                                                   # L1 median ~1.12 out of [0.9, 1.1]
    c["r_cf"] = c["r_traj"] * 1.12 / 1.05                                    # L2 median ~1.05: inside [0.8, 1.2]
    s = X.score_arm(**c)
    assert s["L1"]["verdict"] == "FAIL" and s["L2"]["verdict"] == "PASS" and s["L3"]["verdict"] == "PASS"
    assert "L1" in s["outcome"] and s["outcome"].startswith("FAIL")


def test_L1_band_edges():
    for ratio, v in ((0.9005, "PASS"), (1.0995, "PASS"), (0.8995, "FAIL"), (1.1005, "FAIL")):
        c = _case(ratio=ratio, noise=0.0)
        c["r_cf"] = c["r_traj"] * ratio                                      # keep L2 at exactly 1
        assert X.score_arm(**c)["L1"]["verdict"] == v, ratio


def test_L2_fails_alone():
    c = _case()
    c["r_cf"] = c["r_traj"] * 0.8                                            # r_obs/r_cf ≈ 1.275
    s = X.score_arm(**c)
    assert s["L2"]["verdict"] == "FAIL" and s["L1"]["verdict"] == "PASS" and s["outcome"] == "FAIL L2"


def test_L3_fails_when_ordering_is_unrelated():
    c = _case(noise=0.0)
    rng = np.random.default_rng(9)
    perm = rng.permutation(80)
    c["r_obs_perm"] = None
    c.pop("r_obs_perm")
    ro = c["r_traj"][perm]
    c["s_obs"] = c["s_sw"] * (1 + ro)
    s = X.score_arm(**c)
    assert s["L3"]["spearman"] < 0.5 and s["L3"]["verdict"] == "FAIL"


def test_L4_fails_when_the_prediction_is_no_closer_than_s_glob():
    c = _case()
    c["s_sw"] = np.full(80, S_GLOB); c["s_traj"] = c["s_sw"] * (1 + 0.05)    # predicted far above; obs at s_glob
    c["r_traj"] = np.full(80, 0.05); c["r_cf"] = np.full(80, 0.05)
    c["s_obs"] = S_GLOB * (1 + 1e-4 * np.linspace(-1, 1, 80))
    s = X.score_arm(**c)
    assert s["L4"]["verdict"] == "FAIL" and s["L4"]["mean_D"] > 0


def test_L5_fails_when_the_lag_is_not_resolved():
    c = _case(noise=0.0, lag=0.005)                                          # a predicted lag of ~0.5%
    c["s_obs"] = c["s_sw"] * (1 + 1e-5 * np.linspace(0.5, 1.5, 80))          # the data cross at the switch itself
    s = X.score_arm(**c)
    assert s["valid"] and s["L5"]["verdict"] == "FAIL" and s["L5"]["mean_D"] > 0 and "L5" in s["outcome"]
    assert s["L1"]["verdict"] == "FAIL" and s["L4"]["verdict"] == "PASS"     # L4 can pass while L5 fails


def test_L5_passes_only_with_a_resolved_lag():
    c = _case(noise=0.01, lag=0.003)
    s = X.score_arm(**c)
    assert s["L5"]["verdict"] == "PASS" and s["L5"]["ci95"][1] < 0


def test_V1_fewer_than_60_crossings_with_prediction():
    s = X.score_arm(**_case(n_cross=59))
    assert not s["valid"] and not s["validity"]["V1_min_crossing_with_prediction"]
    assert s["outcome"] == "UNRESOLVED (validity)" and _v(s) == ("UNRESOLVED",) * 5
    c = _case(n_cross=80); c["r_traj"] = c["r_traj"].copy(); c["r_traj"][:21] = np.nan   # 59 with a prediction
    assert X.score_arm(**c)["outcome"] == "UNRESOLVED (validity)"
    assert X.score_arm(**_case(n_cross=60))["outcome"] == "PASS"


def test_V1_primary_counts_out_of_80_with_neither_runs():
    s = X.score_arm(**_case(n_on=64, n_cross=80))                           # 64 on-branch cross: valid
    assert s["valid"] and s["n_scored"] == 64
    c = _case(n_on=64, n_cross=80); c["r_traj"] = c["r_traj"].copy(); c["r_traj"][:5] = np.nan
    assert X.score_arm(**c)["outcome"] == "UNRESOLVED (validity)"          # 59 < 60


def test_V2_tsw_before_crossing_for_positive_kappa_only():
    c = _case(); c["t_sw"] = c["t_sw"].copy(); c["t_sw"][:8] = 6000.0       # 72/80 = 0.9 (strict <)
    s = X.score_arm(**c)
    assert s["frac_tsw_before_crossing_kappa_pos"] == 0.9 and s["valid"]
    c["t_sw"][8] = 7000.0
    s = X.score_arm(**c)
    assert not s["validity"]["V2_tsw_before_crossing_90pct_kappa_pos"] and s["outcome"] == "UNRESOLVED (validity)"
    c = _case(); c["kappa"] = c["kappa"].copy(); c["kappa"][:20] = -0.19   # negative κ: predicted early, still scored
    c["t_sw"] = c["t_sw"].copy(); c["t_sw"][:20] = 6100.0
    c["r_traj"] = c["r_traj"].copy(); c["r_traj"][:20] *= -1; c["r_cf"] = c["r_traj"] * 1.01
    c["s_traj"] = c["s_sw"] * (1 + c["r_traj"])
    c["s_obs"] = c["s_sw"] * (1 + c["r_traj"] * 1.02)
    s = X.score_arm(**c)
    assert s["n_kappa_pos_crossing"] == 60 and s["frac_tsw_before_crossing_kappa_pos"] == 1.0
    assert s["valid"] and s["n_scored"] == 80
    c["t_sw"][30] = np.nan                                                  # undefined t_sw counts as not preceding
    c["t_sw"][31:40] = 7000.0
    assert not X.score_arm(**c)["validity"]["V2_tsw_before_crossing_90pct_kappa_pos"]


def test_V3_regime():
    c = _case(); c["eta_lam"] = c["eta_lam"].copy(); c["eta_lam"][:16] = 0.6   # 64/80 = 0.8
    assert X.score_arm(**c)["valid"]
    c["eta_lam"][16] = 0.6
    assert not X.score_arm(**c)["validity"]["V3_regime_eta_lam_le_0p5_in_80pct"]


def test_V4_resolution():
    c = _case(); c["lag_steps"] = np.full(80, 10.0)
    assert X.score_arm(**c)["valid"]
    c["lag_steps"] = np.full(80, 9.9)
    s = X.score_arm(**c)
    assert not s["validity"]["V4_median_predicted_lag_ge_10_steps"] and s["outcome"] == "UNRESOLVED (validity)"


def test_V5_q90_kappa_chi():
    c = _case(); c["r_cf"] = np.full(80, 0.1); c["r_traj"] = np.full(80, 0.1 / 1.01)
    c["s_traj"] = c["s_sw"] * (1 + c["r_traj"]); c["s_obs"] = c["s_sw"] * (1 + 0.1)
    assert X.score_arm(**c)["validity"]["V5_q90_kappa_chi_le_0p1"]
    c["r_cf"] = np.full(80, 0.1001)
    assert not X.score_arm(**c)["validity"]["V5_q90_kappa_chi_le_0p1"]
    c = _case(); c["r_cf"] = c["r_cf"].copy(); c["r_cf"][:10] = -0.2         # signed: negative κχ does not raise q90
    assert X.score_arm(**c)["validity"]["V5_q90_kappa_chi_le_0p1"]


def test_V6_chi_against_pilot():
    c = _case(); c["pilot_chi_median"] = 0.022 / 1.299                       # median χ 29.9% above the pilot's
    assert X.score_arm(**c)["validity"]["V6_median_chi_within_30pct_of_pilot"]
    c["pilot_chi_median"] = 0.022 / 0.701
    assert X.score_arm(**c)["validity"]["V6_median_chi_within_30pct_of_pilot"]
    c["pilot_chi_median"] = 0.022 / 1.301
    assert not X.score_arm(**c)["validity"]["V6_median_chi_within_30pct_of_pilot"]
    c["pilot_chi_median"] = 0.022 / 0.69
    assert not X.score_arm(**c)["validity"]["V6_median_chi_within_30pct_of_pilot"]
    c["pilot_chi_median"] = float("nan")
    assert X.score_arm(**c)["outcome"] == "UNRESOLVED (validity)"


def test_V7_follows_branch():
    c = _case(); c["chi_path_max"] = np.where(np.arange(80) < 70, 0.1, 0.3)
    s = X.score_arm(**c)
    assert s["q90_max_chi_window"] > 0.25 and not s["validity"]["V7_q90_max_chi_window_le_0p25"]
    c["chi_path_max"] = np.full(80, 0.25)
    assert X.score_arm(**c)["validity"]["V7_q90_max_chi_window_le_0p25"]
    c["chi_path_max"] = np.full(80, 0.1); c["chi_path_max"][3] = np.nan     # an undefined path max fails V7
    assert not X.score_arm(**c)["validity"]["V7_q90_max_chi_window_le_0p25"]


def test_invalid_even_if_statistics_would_fail():
    c = _case(n_cross=50, ratio=1.5)
    s = X.score_arm(**c)
    assert _v(s) == ("UNRESOLVED",) * 5 and s["L1"]["median_ratio"] == pytest.approx(1.5, rel=0.03)


def test_non_crossing_runs_are_not_scored_and_multiple_failures_are_named():
    c = _case(n_cross=70, ratio=1.3)
    c["r_cf"] = c["r_traj"] * 0.5
    s = X.score_arm(**c)
    assert s["n_scored"] == 70 and s["outcome"].startswith("FAIL") and "L1" in s["outcome"] and "L2" in s["outcome"]


def test_hold_keeps_w2_fixed_and_release_applies_rho_to_w2_only():
    import torch
    from torch.nn import functional as F
    from src.act_general import GAct
    from src.fold1d import make_data
    u = GAct("gelu").torch_u
    xt, yt = make_data(200, 876_950)                                         # a seed outside every registered range
    Xt, Yt = xt.to(torch.float64), yt.to(torch.float64)
    th0 = np.array([0.3, -0.2, 3.3, 0.1])
    th, rec = X.hold(th0, 5, Xt, Yt, u)
    assert th[2] == 3.3 and rec["W_hold"] == 5 and not np.allclose(th[[0, 1, 3]], th0[[0, 1, 3]])
    t = torch.tensor(th, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(F.binary_cross_entropy_with_logits(t[2] * u(t[0] * Xt + t[1]) + t[3], Yt), t)[0].numpy()
    for rho in (1.0, 0.5):
        Wp = X.release_train(th, Xt, Yt, u, rho, 1)
        assert np.allclose(Wp[1, [0, 1, 3]], th[[0, 1, 3]] - 0.03 * g[[0, 1, 3]], rtol=0, atol=1e-15)
        assert Wp[1, 2] == pytest.approx(th[2] - rho * 0.03 * g[2], abs=1e-15)
    Wp = X.release_train(th, Xt, Yt, u, 1.0, 50, stop_s=th[2] + 1e-9)       # pilot: stop at t_sw
    assert len(Wp) < 51 and abs(Wp[-1, 2]) >= th[2] + 1e-9


def test_L3_with_all_predictions_tied_is_unresolved():
    c = _case(noise=0.0)
    c["r_traj"] = np.full(80, 0.003); c["r_cf"] = np.full(80, 0.003); c["s_traj"] = c["s_sw"] * 1.003
    c["s_obs"] = c["s_sw"] * (1 + 0.003 * np.linspace(0.98, 1.02, 80))
    with np.errstate(invalid="ignore"):
        s = X.score_arm(**c)
    assert s["valid"] and s["L3"]["verdict"] == "UNRESOLVED" and s["outcome"] == "UNRESOLVED"
    assert s["L1"]["verdict"] == "PASS"


def test_newton_at_rejects_a_singular_hessian_without_raising(monkeypatch):
    from src import act_general
    P = X.own_problem(876_950)                                               # a seed outside every registered range
    monkeypatch.setattr(act_general, "branch_point", lambda z0, s, x, y, act: (np.asarray(z0, float), 0.0))
    monkeypatch.setattr(X, "hessian_z", lambda z, s, P: np.zeros((3, 3)))    # singular H at the Newton point
    z, g, lam, ok = X.newton_at([0.0, -1.0, 0.1], 3.3, P)
    assert not ok and lam == 0.0
    monkeypatch.setattr(X, "hessian_z", lambda z, s, P: np.full((3, 3), np.nan))
    assert not X.newton_at([0.0, -1.0, 0.1], 3.3, P)[3]
    assert X.classify_release([0.0, -1.0, 0.1], z, ok, COP) == 0             # counted as on neither copy


def test_hessian_z_matches_act_general():
    from src.act_general import hessian_and_tangent
    P = X.own_problem(876_950)
    z = np.array([1.3, -1.0, -0.3])
    H, _ = hessian_and_tangent(z, 3.3, P.x, P.y, P.act)
    assert np.allclose(X.hessian_z(z, 3.3, P), H, rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------------------------------------ author's decisions 2026-09-29
def test_chi_window_max_uses_only_s_at_or_above_0p8_switch():
    s = np.array([3.0, 3.5, 4.0, 4.79, 4.81, 5.5, 6.0, 6.1])
    chi = np.array([0.9, 0.8, 0.7, 0.6, 0.05, 0.04, 0.03])                 # χ_t for t = 0 … t_sw − 1 (t_sw = 7)
    assert X.chi_window_max(s, 7, chi, 6.0) == 0.05                         # s_t ≥ 4.8 from t = 4 (χ_0…χ_3 excluded)
    assert X.chi_window_max(s, 7, chi, 100.0) != X.chi_window_max(s, 7, chi, 6.0)
    assert math.isnan(X.chi_window_max(s, 7, chi, 100.0))                  # empty window
    chi2 = chi.copy(); chi2[5] = np.nan
    assert math.isnan(X.chi_window_max(s, 7, chi2, 6.0))
    assert math.isnan(X.chi_window_max(s, None, chi, 6.0))
    assert X.first_at_fraction(s, 6.0) == 4 and X.first_at_fraction(s, 100.0) is None


def test_follows_branch_rule():
    assert X.follows_branch(1, 1) and X.follows_branch(-1, -1)
    assert not X.follows_branch(1, -1) and not X.follows_branch(-1, 1)
    assert not X.follows_branch(1, 0) and not X.follows_branch(0, 0)


def test_follow_classification_newton_identity_and_state_tolerance():
    zt = COP[1]
    lagging = zt + 0.02                                                     # a state lagging its branch by 2e-2
    assert X.classify_release(lagging, zt, True, COP, state_tol=None) == 1
    assert X.classify_release(lagging, zt, True, COP, state_tol=1e-3) == 0
    assert X.classify_release(lagging, COP[-1], True, COP, state_tol=None) == -1   # Newton lands on the other copy
    assert X.classify_release(lagging, zt + 2e-6, True, COP, state_tol=None) == 0
    assert X.classify_release(lagging, zt, False, COP, state_tol=None) == 0


def test_runs_not_following_their_branch_are_not_scored():
    c = _case()
    fol = np.ones(80, bool); fol[:10] = False
    c["s_obs"] = c["s_obs"].copy(); c["s_obs"][:10] = 100.0                 # would fail everything if scored
    s = X.score_arm(**c, follows=fol)
    assert s["n_scored"] == 70 and s["n_on_branch_not_following"] == 10 and s["outcome"] == "PASS"
    fol[:21] = False
    assert X.score_arm(**c, follows=fol)["outcome"] == "UNRESOLVED (validity)"      # 59 scored < 60


def test_sensitivity_exclusion_is_separate_from_the_registered_scoring():
    c = _case()
    hp = np.zeros(80, bool); hp[:25] = True
    c["hold_positive"] = hp
    s = X.score_arm(**c)
    assert s["n_scored"] == 80 and s["outcome"] == "PASS"
    sens = X.score_arm(**c, exclude=hp)
    assert sens["n_scored"] == 55 and sens["outcome"] == "UNRESOLVED (validity)"   # fewer than 60 once excluded
    assert sens["gate"]["pass"] and sens["n_excluded_sensitivity"] == 25


def test_kappa_invariant_under_mirror_coordinates_G_sign_and_s_direction():
    rng = np.random.default_rng(1)
    for _ in range(50):
        A = rng.normal(size=(3, 3)); H = A @ A.T + 0.1 * np.eye(3)
        tan, dG = rng.normal(size=3), np.r_[rng.normal(size=2), 0.0]
        k = X.kappa_transforms(H, tan, dG)
        for key in ("mirror_coords", "G_sign", "s_direction"):
            assert k[key] == pytest.approx(k["kappa"], rel=1e-9, abs=1e-12)
    # a negative κ stays negative: the sign is not a convention
    H = np.diag([1.0, 10.0, 1.0]); tan = np.array([1.0, 1.0, 0.0]); dG = np.array([-1.0, 1.2, 0.0])
    k = X.kappa_transforms(H, tan, dG)
    assert k["kappa"] < 0 and k["mirror_coords"] < 0 and k["G_sign"] < 0 and k["s_direction"] < 0


def test_kappa_recomputed_on_the_mirrored_sample_equals_the_original():
    from src.act_general import hessian_and_tangent
    from src.lag_law import kappa
    seed, s = 876_950, 5.0                                                  # outside every registered range
    P = X.own_problem(seed)
    z, g, lam, ok = X.newton_at([1.3, -1.0, -0.3], s, P)
    assert ok
    H, tan = hessian_and_tangent(z, s, P.x, P.y, P.act)
    dG, _ = X.grad_gap_checked(z[0], z[1], P.act)
    k0 = kappa(H, tan, dG, np.ones(3))[0]
    m = X.kappa_mirror_recomputed(seed, z, s)
    assert m["newton_ok"] and m["dist_from_mirror_image"] < 1e-9
    assert m["kappa"] == pytest.approx(k0, rel=1e-6)


def _neg_case(n_neg=20, ratio=1.02):
    """20 predicted-early runs (κ < 0): signed lags negative, crossing before t_sw."""
    c = _case(noise=0.0, ratio=ratio)
    k = c["kappa"].copy(); k[:n_neg] = -0.19; c["kappa"] = k
    rt = c["r_traj"].copy(); rt[:n_neg] *= -1; c["r_traj"] = rt; c["r_cf"] = rt * 1.01
    c["s_traj"] = c["s_sw"] * (1 + rt)
    c["s_obs"] = c["s_sw"] * (1 + rt * ratio)
    ts = c["t_sw"].copy(); ts[:n_neg] = 6100.0; c["t_sw"] = ts                # t_sw after the (early) crossing
    return c


def test_negative_kappa_runs_are_scored_with_signed_ratios():
    s = X.score_arm(**_neg_case())
    assert s["valid"] and s["n_scored"] == 80 and s["n_kappa_nonpos_scored"] == 20 and s["outcome"] == "PASS"
    assert s["n_kappa_pos_crossing"] == 60 and s["frac_tsw_before_crossing_kappa_pos"] == 1.0


def test_negative_kappa_runs_with_the_wrong_sign_fail():
    c = _neg_case()
    c["s_obs"] = c["s_obs"].copy()
    c["s_obs"][:20] = c["s_sw"][:20] * (1 - c["r_traj"][:20])              # observed LATE where predicted early
    s = X.score_arm(**c)
    ratios = (c["s_obs"] / c["s_sw"] - 1) / c["r_traj"]
    assert (ratios[:20] < 0).all()                                          # signed ratios: −1 for those runs
    assert s["valid"] and s["L3"]["spearman"] < 0.9 and s["L5"]["verdict"] in ("PASS", "FAIL")
    c["s_obs"][:45] = c["s_sw"][:45] * (1 - c["r_traj"][:45])               # 45 of 80 of the wrong sign: L1 fails
    assert X.score_arm(**c)["L1"]["verdict"] == "FAIL"


def test_V4_uses_signed_predicted_lag():
    c = _neg_case(n_neg=41)
    c["lag_steps"] = np.where(c["kappa"] < 0, -40.0, 40.0)                 # median of signed lags < 10
    s = X.score_arm(**c)
    assert s["median_predicted_lag_steps"] < 10 and not s["validity"]["V4_median_predicted_lag_ge_10_steps"]
