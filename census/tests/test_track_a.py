"""Track A (registered lag-law test at a = 1.65): every rule and validity check on constructed pass/fail cases."""

import math

import numpy as np
import pytest

from src import track_a as T


def test_winding_rule_principal_copy():
    for b1 in (-2.3, 3.83, -2.45 - 2 * math.pi, 10.1, math.pi, -math.pi + 1e-9):
        th, k = T.principal_copy([1.0, b1, 5.0], 2.0)
        assert -math.pi < th[1] <= math.pi
        assert th[1] == pytest.approx(b1 + 2 * math.pi * k)
        assert th[2] == pytest.approx(5.0 - 2 * math.pi * k * 2.0)       # logits unchanged: b₂ − 2πk·s
    assert T.principal_copy([1.0, 3.83, 0.0], 1.0)[1] == -1              # 1.30's retained copy → k = −1, as observed
    assert T.principal_copy([1.0, -2.29, 0.0], 1.0)[1] == 0


def test_rule_step_is_the_last_upward_passage_of_half_the_frozen_switch():
    s = np.array([0.1, 0.4, 0.9, 0.95, 1.0, 1.2, 2.1])
    assert T.rule_step(s, 2.0) == 4                                       # s = 1.0 ≥ 0.5·2.0 from step 4 to the top
    s = np.array([0.95, 1.2, 0.3, 0.2, 0.8, 1.1, 1.5, 2.2])              # initial transient above 0.9, then a dip
    assert T.rule_step(s, 1.8) == 5
    assert T.rule_step(s, 10.0) is None                                   # never reaches s_frozen
    s = np.array([0.95, 1.0, 1.4, 1.9])                                   # never below 0.5·s_frozen after init
    assert T.rule_step(s, 1.8) == 1


def test_closed_form_matches_the_1A_kappa_and_scales():
    from src import lag_law
    H = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])
    tan = np.array([0.4, -0.7, 1.3]); dG = np.array([1.0, -1.0, 0.0]); p = np.array([3.0, 1.5, 2.0])
    r, kap, chi = T.closed_form_r(H, tan, dG, p, sdot=1e-3, s_star=2.0, lr=0.01)
    assert kap == pytest.approx(lag_law.kappa(H, tan, dG, p)[0])
    r2, kap2, chi2 = T.closed_form_r(H, tan, dG, 7 * p, sdot=1e-3, s_star=2.0, lr=0.01)
    assert kap2 == pytest.approx(kap) and chi2 == pytest.approx(chi / 7)  # κ invariant to P's scale, χ is not
    # r = ∇G·(ηPH)⁻¹θ*′ ṡ / (s*·∇G·θ*′)
    direct = float(dG @ np.linalg.solve(0.01 * p[:, None] * H, tan)) * 1e-3 / (2.0 * float(dG @ tan))
    assert r == pytest.approx(direct)


def test_first_placed():
    assert T.first_placed([-1, -0.5, 0.0, 0.1, -1]) == 3
    assert T.first_placed([-1, -1]) is None


def _case(n=80, n_cross=70, ratio_traj=1.0, ratio_cf=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.05, 0.12, n)
    obs = pred * ratio_traj * (1 + noise * rng.normal(size=n))
    crossed = np.zeros(n, bool); crossed[:n_cross] = True
    obs = np.where(crossed, obs, np.nan)
    cf = obs / ratio_cf
    return obs, pred, cf, crossed, np.ones(n, bool)


def test_all_pass():
    s = T.score_opt(*_case(ratio_traj=1.05, ratio_cf=1.15, noise=0.01))
    assert s["valid"] and s["L1"]["verdict"] == "PASS" and s["L2"]["verdict"] == "PASS" and s["L3"]["verdict"] == "PASS"


def test_band_edges_and_fails():
    assert T.score_opt(*_case(ratio_traj=0.89))["L1"]["verdict"] == "FAIL"
    assert T.score_opt(*_case(ratio_traj=1.11))["L1"]["verdict"] == "FAIL"
    assert T.score_opt(*_case(ratio_traj=0.90))["L1"]["verdict"] == "PASS"
    assert T.score_opt(*_case(ratio_traj=1.10))["L1"]["verdict"] == "PASS"
    assert T.score_opt(*_case(ratio_cf=1.21))["L2"]["verdict"] == "FAIL"
    assert T.score_opt(*_case(ratio_cf=0.79))["L2"]["verdict"] == "FAIL"
    assert T.score_opt(*_case(ratio_cf=0.81))["L2"]["verdict"] == "PASS"


def test_spearman_fail_when_uncorrelated():
    obs, pred, cf, crossed, rb = _case()
    rng = np.random.default_rng(3)
    obs = np.where(crossed, rng.permutation(pred) , np.nan)            # same distribution, no rank relation
    s = T.score_opt(obs, pred, cf, crossed, rb)
    assert s["L3"]["spearman"] < 0.5 and s["L3"]["verdict"] == "FAIL"


def test_validity_fewer_than_60_crossings_is_unresolved():
    s = T.score_opt(*_case(n_cross=59))
    assert not s["valid"]
    assert s["L1"]["verdict"] == s["L2"]["verdict"] == s["L3"]["verdict"] == "UNRESOLVED"
    s = T.score_opt(*_case(n_cross=60))
    assert s["valid"] and s["L1"]["verdict"] == "PASS"


def test_missing_predictions_and_rule_after_crossing_are_excluded_and_counted():
    obs, pred, cf, crossed, rb = _case(n_cross=70)
    pred = pred.copy(); pred[:15] = np.nan                                 # 55 crossing runs with a prediction
    s = T.score_opt(obs, pred, cf, crossed, rb)
    assert s["valid"] and s["L1"]["verdict"] == "UNRESOLVED" and s["L3"]["verdict"] == "UNRESOLVED"
    assert s["L2"]["verdict"] == "PASS"
    obs, pred, cf, crossed, rb = _case(n_cross=70)
    rb = rb.copy(); rb[:12] = False
    s = T.score_opt(obs, pred, cf, crossed, rb)
    assert s["n_rule_not_before_crossing"] == 12 and s["L1"]["n"] == 58 and s["L1"]["verdict"] == "UNRESOLVED"
    obs, pred, cf, crossed, rb = _case(n_cross=80)
    rb = rb.copy(); rb[:12] = False
    obs2 = obs.copy(); obs2[:12] = 100.0                                   # would fail L1 if not excluded
    s = T.score_opt(obs2, pred, cf, crossed, rb)
    assert s["n_rule_not_before_crossing"] == 12 and s["L1"]["n"] == 68 and s["L1"]["verdict"] == "PASS"


def test_spearman_helper():
    assert T.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert T.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_diag_metrics_on_constructed_cases():
    from src import track_a_diag as D
    pred = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    m = D.metrics(pred * 1.05, pred)
    assert m["spearman"] == pytest.approx(1.0) and m["frac_within_10pct"] == 1.0
    assert m["median_obs_over_pred"] == pytest.approx(1.05)
    m = D.metrics(pred[::-1] * 1.0, pred)
    assert m["spearman"] == pytest.approx(-1.0)
    m = D.metrics(np.array([0.1, np.nan, 0.2, 0.3]), np.array([0.1, 0.2, np.nan, 0.5]))
    assert m["n"] == 2
    m = D.metrics(pred * np.array([1.0, 1.2, 0.8, 1.0, 1.09, 0.95]), pred)
    assert m["frac_within_10pct"] == pytest.approx(4 / 6)
