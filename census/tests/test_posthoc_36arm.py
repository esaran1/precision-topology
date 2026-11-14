"""POST HOC 36-arm causal re-score (rebuttal): the full path from run rows (the format `run` writes) through table() and
arm_stats() to the per-arm numbers, on constructed pass, fail and unresolved cases; and the raw recorder leaves the
replay unchanged."""
import numpy as np
import pytest

from src import posthoc_36arm as M


def _fc(t_fc, t_sw_fc, r_fc, status="ok", nan_ok=True):
    return {"status": status, "t_fc": t_fc, "t_sw_fc": t_sw_fc, "r_fc": r_fc, "nan_recompute_identical": nan_ok}


def _row(seed, *, t_obs=1100, s_obs=1.10, s_run=1.0, t_sw=1000, t_c=900, t_fc=1100, t_sw_fc=1000, r_fc=0.10,
         opt="adam", status="ok", nan_ok=True, crossed=True, t_star=None, t_rule=500, arm="0.5", st="lag1", a=1.3,
         fc_status="ok"):
    sw = {"t_c": t_c, "s_run": s_run, "t_rule": t_rule, "t_sw": t_sw}
    fcd = {}
    for f in M.FS:
        if opt == "adam":
            fcd[str(f)] = {"rule_point": _fc(t_fc, t_sw_fc, r_fc, fc_status, nan_ok),
                           "cutoff": _fc(t_fc + 3, t_sw_fc, r_fc * 1.5, fc_status, nan_ok), "_switch": sw}
        else:
            fcd[str(f)] = {"None": _fc(t_fc, t_sw_fc, r_fc, fc_status, nan_ok), "_switch": sw}
    r = {"set": st, "a": a, "seed": seed, "arm": arm, "status": status, "opt": opt, "t_star": t_star, "fc": fcd,
         "replay_n_steps_ok": True, "replay_s_max_ok": True, "crossed": crossed}
    if crossed:
        r.update(t_obs=t_obs, s_obs=s_obs, obs_index_ok=True)
    return r


def test_exact_forecasts_pass_every_yardstick():
    rows = [_row(i) for i in range(40)]
    A = M.arm_stats(M.table(rows, 0.90))
    a = A.iloc[0]
    assert a.n_scored == 40 and a.n_forecast == 40 and a.n_miss == 0 and a.n_nan_differs == 0
    assert a.ratio_of_medians == pytest.approx(1.0) and a.arm_within_5pct
    assert a.frac_runs_within_10pct == 1.0 and a.C1_frac_cross_within_10 == 1.0 and a.C2_frac_lag_within_15 == 1.0
    assert a.C3_median_in_band


def test_biased_forecasts_fail_the_5pct_and_10pct_yardsticks():
    # predicted lag 0.10, observed 0.13: ratio 1.3 in every run
    rows = [_row(i, s_obs=1.13, t_obs=1130) for i in range(40)]
    a = M.arm_stats(M.table(rows, 0.90)).iloc[0]
    assert a.ratio_of_medians == pytest.approx(1.3) and not a.arm_within_5pct
    assert a.frac_runs_within_10pct == 0.0 and not a.C3_median_in_band
    assert a.C1_frac_cross_within_10 == 0.0                    # |1100 − 1130| = 30 > 10


def test_misses_count_against_fractions_and_unresolved_has_no_ratio():
    rows = [_row(i) for i in range(30)] + [_row(100 + i, fc_status="no forecast: x") for i in range(10)]
    a = M.arm_stats(M.table(rows, 0.90)).iloc[0]
    assert a.n_scored == 40 and a.n_forecast == 30 and a.n_miss == 10
    assert a.frac_runs_within_10pct == pytest.approx(0.75)
    rows = [_row(i, fc_status="no forecast: x") for i in range(5)]
    a = M.arm_stats(M.table(rows, 0.90)).iloc[0]
    assert a.n_forecast == 0 and np.isnan(a.ratio_of_medians) and not a.arm_within_5pct


def test_cutoff_not_before_crossing_and_nan_differs_and_uncrossed():
    rows = [_row(0, t_c=1100), _row(1, t_c=1200), _row(2, nan_ok=False), _row(3, crossed=False), _row(4)]
    T = M.table(rows, 0.90)
    assert list(T.scored) == [False, False, True, False, True]
    a = M.arm_stats(T).iloc[0]
    assert a.n_runs == 5 and a.n_crossed == 4 and a.n_cutoff_before == 2 and a.n_nan_differs == 1


def test_adam_p_rule_and_sgd_key():
    rows = [_row(i) for i in range(10)]
    t_rp = M.table(rows, 0.95, "rule_point"); t_cut = M.table(rows, 0.95, "cutoff")
    assert (t_rp.r_fc == 0.10).all() and np.allclose(t_cut.r_fc, 0.15)
    rows = [_row(i, opt="sgd", st="SGD", arm="sgd") for i in range(10)]
    assert (M.table(rows, 0.90, "cutoff").r_fc == 0.10).all()     # SGD ignores the P rule


def test_lag_error_and_interventions_on_realistic_rows():
    # realistic magnitudes: switch 2.53, observed lag 9%, forecast lag 8.5%, crossing 6 steps early
    rows = [_row(i, s_run=2.527, s_obs=2.527 * 1.09, r_fc=0.085, t_sw=1500 + i, t_obs=1660 + i, t_fc=1654 + i,
                 t_sw_fc=1498 + i, t_c=1450 + i, t_rule=900, t_star=1200 if i % 2 else 800, st="4b", arm="teleport",
                 a=1.5) for i in range(48)]
    T = M.table(rows, 0.90)
    assert np.allclose(T.lag_err, (1654 - 1498) - (1660 - 1500))          # −4 steps
    a = M.arm_stats(T).iloc[0]
    assert a.ratio_of_medians == pytest.approx(0.09 / 0.085)
    assert not a.arm_within_5pct and a.frac_runs_within_10pct == 1.0
    assert a.n_intervention_after_rule == 24
    o = M.overall(M.arm_stats(T), T)
    assert o["n_arms_defined"] == 1 and o["n_arms_within_5pct"] == 0 and o["n_miss"] == 0


def test_replay_failure_is_counted():
    rows = [_row(i) for i in range(4)]
    rows[0]["replay_s_max_ok"] = False
    rows[1]["obs_index_ok"] = False
    assert M.arm_stats(M.table(rows, 0.90)).iloc[0].n_replay_not_ok == 2


def test_budget_rules():
    assert M.budget_of("lag1", "0.25", None) == 128_000
    assert M.budget_of("lag2-prim", "2.0", 1001) == 1001 + int(np.ceil((32_000 - 1001) / 2.0))
    assert M.budget_of("4b", "reset", 900) == 32_000 and M.budget_of("SGD", "sgd", None) == 32_000


def test_raw_recorder_leaves_the_replay_unchanged():
    """One short SGD and one short TaskB (Adam) replay: the recorded s and canonical θ equal linear_response's own
    replay bit for bit; V̂ is Adam's bias-corrected second moment."""
    from src import linear_response as LR
    for kind, arm in (("SGD", "sgd"), ("TaskB", "adam")):
        p0, x0, y0 = LR.replay_path(kind, 1.5, 3, arm, 1.5)
        p1, x1, y1 = M.replay(kind, 1.5, 3, arm, 1.5)
        assert np.array_equal(p0.s, p1.s) and np.array_equal(p0.th, p1.th) and np.array_equal(x0, x1)
        assert np.array_equal(np.abs(p1.W[:, 2]), p1.s)
        if kind == "TaskB":
            assert np.array_equal(p0.PP[1:], p1.PP[1:])
            assert np.allclose(1.0 / (np.sqrt(p1.V[1:, [0, 1, 3]]) + LR.EPS), p1.PP[1:], rtol=1e-12, atol=0)
            assert np.isnan(p1.V[0]).all()
    assert LR.Path_ is not M.RawPath and LR._record_state is M._ORIG_RECORD
