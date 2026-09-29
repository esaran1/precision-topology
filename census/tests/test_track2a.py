"""Test 2A (registered; Adam per-run ordering at a = 1.85): every decision rule on constructed pass, fail and
unresolved cases, t_sw detection, and the reuse of the unchanged Track A pipeline."""

import numpy as np
import pytest

from src import track2a as X
from src import track_a as T


# ------------------------------------------------------------------------------------------ pipeline reuse
def test_pipeline_is_track_a_rebound_without_touching_the_imported_module():
    m = X.pipeline()
    assert m.A == 1.85 and m.SEEDS == X.SEEDS and m.OPTS == ("adam",) and m.OUT == X.OUT and m.PATHS == X.PATHS
    assert m is not T and T.A == 1.65 and T.OUT.name == "track_a" and T.SEEDS[0] == 1_650_000
    assert m.BUDGET == T.BUDGET == 32_000 and m.LR_ADAM == T.LR_ADAM == 0.01 and m.RULE_FRAC == T.RULE_FRAC == 0.5
    assert X._sha(X.ROOT / "src" / "track_a.py") == X.TRACK_A_SHA256
    assert X.SEEDS == tuple(range(1_850_000, 1_850_080)) and len(X.SEEDS) == 80 and X.A == 1.85


def test_registered_constants():
    assert (X.A1_MIN, X.A2_MIN, X.A2_TOL, X.A3_BAND, X.MIN_CROSS, X.TSW_MIN_FRAC) == (0.5, 0.75, 0.10, (0.9, 1.1), 60, 0.90)


# ------------------------------------------------------------------------------------------ t_sw detection
def test_t_switch_is_the_first_step_reaching_s_run():
    s = np.array([0.9, 0.3, 0.8, 1.29, 1.3, 1.2, 1.4])
    assert X.t_switch(s, 1.3) == 4                    # ≥, not >
    assert X.t_switch(s, 1.35) == 6                   # later dips do not matter: the FIRST step
    assert X.t_switch(s, 0.85) == 0                   # the initial state counts if it is already there
    assert X.t_switch(s, 1.5) is None                 # never reaches s*_run
    assert X.t_switch(s, None) is None and X.t_switch(s, float("nan")) is None


def test_v_at_switch_replaces_only_the_rule_point_row():
    V = np.arange(40, dtype=float).reshape(10, 4)
    Vm = X.v_at_switch(V, 3, 7)
    assert np.array_equal(Vm[3], V[7])
    mask = np.ones(10, bool); mask[3] = False
    assert np.array_equal(Vm[mask], V[mask]) and V[3, 0] == 12.0      # the input is not modified


# ------------------------------------------------------------------------------------------ scoring
def _case(n=80, n_cross=76, ratio=1.03, noise=0.02, seed=0):
    """Constructed runs: rule point < t_sw < crossing for every crossing run; obs = pred·ratio·(1 + noise)."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(0.05, 0.12, n)
    obs = pred * ratio * (1 + noise * rng.uniform(-1, 1, n))
    crossed = np.zeros(n, bool); crossed[:n_cross] = True
    step = np.where(crossed, 3000.0, np.nan)
    t_rule = np.full(n, 1000.0); t_sw = np.full(n, 2800.0)
    obs = np.where(crossed, obs, np.nan)
    return dict(r_obs=obs, r_pred=pred, crossed=crossed, step_obs=step, t_rule=t_rule, t_sw=t_sw)


def _v(s):
    return s["A1"]["verdict"], s["A2"]["verdict"], s["A3"]["verdict"]


def test_all_pass():
    s = X.score_2a(**_case())
    assert s["valid"] and _v(s) == ("PASS", "PASS", "PASS") and s["outcome"] == "PASS"
    assert s["n_crossed"] == 76 and s["n_scored"] == 76 and s["frac_tsw_before_crossing"] == 1.0


def test_A1_fails_when_ranks_are_unrelated():
    c = _case(noise=0.0)
    rng = np.random.default_rng(5)
    c["r_obs"] = np.where(c["crossed"], rng.permutation(c["r_pred"]), np.nan)
    s = X.score_2a(**c)
    assert s["valid"] and s["A1"]["spearman"] < 0.5 and s["A1"]["verdict"] == "FAIL"


def test_A1_fail_alone_with_A2_A3_passing():
    # ratios all within 10% but ordered against the predictions: A2, A3 pass, A1 fails
    pred = np.linspace(0.080, 0.090, 80)
    obs = pred[::-1].copy()                                         # obs/pred ∈ [0.889, 1.125]: tighten the spread
    obs = 0.085 + (obs - 0.085) * 0.5
    c = _case(); c.update(r_pred=pred, r_obs=obs)
    s = X.score_2a(**c)
    assert s["A1"]["spearman"] < 0 and _v(s) == ("FAIL", "PASS", "PASS") and s["outcome"] == "FAIL A1"


def test_A2_threshold_edges():
    c = _case(n=80, n_cross=80, noise=0.0, ratio=1.0)
    pred = c["r_pred"]
    k = 60                                                          # 60/80 = 0.75 exactly within 10%
    r = np.where(np.arange(80) < k, 1.05, 1.3)
    r[70:] = 0.7                                                    # keep the median inside [0.9, 1.1]
    c["r_obs"] = pred * r
    s = X.score_2a(**c)
    assert s["A2"]["frac_within_10pct"] == 0.75 and s["A2"]["verdict"] == "PASS"
    r[59] = 1.3                                                     # 59/80 < 0.75
    c["r_obs"] = pred * r
    s = X.score_2a(**c)
    assert s["A2"]["frac_within_10pct"] == pytest.approx(59 / 80) and s["A2"]["verdict"] == "FAIL"
    assert s["A3"]["verdict"] == "PASS" and s["outcome"].startswith("FAIL") and "A2" in s["outcome"]


def test_A2_uses_absolute_relative_error_both_sides():
    c = _case(n=80, n_cross=80, noise=0.0)
    r = np.where(np.arange(80) % 2 == 0, 0.95, 1.08)
    c["r_obs"] = c["r_pred"] * r
    assert X.score_2a(**c)["A2"]["frac_within_10pct"] == 1.0
    c["r_obs"] = c["r_pred"] * np.where(np.arange(80) % 2 == 0, 0.85, 1.15)
    assert X.score_2a(**c)["A2"]["frac_within_10pct"] == 0.0


def test_A3_band_edges_and_fails():
    for ratio, v in ((0.9, "PASS"), (1.1, "PASS"), (0.899, "FAIL"), (1.101, "FAIL"), (1.0, "PASS")):
        c = _case(noise=0.0, ratio=ratio)
        s = X.score_2a(**c)
        assert s["A3"]["verdict"] == v, ratio
        assert s["A3"]["median_ratio"] == pytest.approx(ratio)
    s = X.score_2a(**_case(noise=0.0, ratio=1.2))                   # A3 and A2 fail, A1 passes
    assert _v(s) == ("PASS", "FAIL", "FAIL") and s["outcome"] == "FAIL A2+A3"


def test_validity_V1_fewer_than_60_crossings_is_unresolved():
    s = X.score_2a(**_case(n_cross=59))
    assert not s["valid"] and not s["validity"]["min_crossings"]
    assert _v(s) == ("UNRESOLVED",) * 3 and s["outcome"] == "UNRESOLVED"
    s = X.score_2a(**_case(n_cross=60))
    assert s["valid"] and _v(s) == ("PASS", "PASS", "PASS")


def test_validity_V1_unresolved_even_if_the_statistics_would_fail():
    c = _case(n_cross=50, ratio=1.5)
    s = X.score_2a(**c)
    assert _v(s) == ("UNRESOLVED",) * 3 and s["A3"]["median_ratio"] == pytest.approx(1.5, rel=0.03)


def test_validity_V2_tsw_before_crossing_in_90pct():
    c = _case(n_cross=80)
    c["t_sw"] = c["t_sw"].copy(); c["t_sw"][:8] = 3000.0          # t_sw == crossing step: NOT before (strict)
    s = X.score_2a(**c)
    assert s["frac_tsw_before_crossing"] == 0.9 and s["valid"] and s["n_scored"] == 72
    assert s["n_crossed_tsw_not_before"] == 8 and _v(s) == ("PASS", "PASS", "PASS")
    c["t_sw"][8] = 3500.0                                           # 71/80 < 0.9
    s = X.score_2a(**c)
    assert not s["valid"] and not s["validity"]["tsw_before_90pct"] and _v(s) == ("UNRESOLVED",) * 3


def test_validity_V2_undefined_tsw_counts_as_not_preceding():
    c = _case(n_cross=80)
    c["t_sw"] = c["t_sw"].copy(); c["t_sw"][:9] = np.nan          # branch rule failed / never reached s*_run
    c["r_pred"] = c["r_pred"].copy(); c["r_pred"][:9] = np.nan
    s = X.score_2a(**c)
    assert s["n_tsw_before_crossing"] == 71 and not s["valid"] and s["outcome"] == "UNRESOLVED"


def test_unscored_runs_do_not_enter_the_statistics():
    c = _case(n_cross=80, noise=0.0, ratio=1.0)
    c["t_sw"] = c["t_sw"].copy(); c["t_sw"][:8] = 3100.0          # after the crossing: unscored
    c["r_obs"] = c["r_obs"].copy(); c["r_obs"][:8] = 100.0          # would fail A2/A3 if scored
    s = X.score_2a(**c)
    assert s["valid"] and s["n_scored"] == 72 and _v(s) == ("PASS", "PASS", "PASS")


def test_runs_without_prediction_are_counted_not_scored():
    c = _case(n_cross=80)
    c["r_pred"] = c["r_pred"].copy(); c["r_pred"][:15] = np.nan
    s = X.score_2a(**c)
    assert s["valid"] and s["n_scored"] == 65 and s["n_crossed_tsw_before_no_prediction"] == 15
    assert _v(s) == ("PASS", "PASS", "PASS")


def test_validity_V3_rule_point_not_before_crossing_in_a_scored_run():
    c = _case(n_cross=80)
    c["t_rule"] = c["t_rule"].copy(); c["t_rule"][0] = 3000.0      # a scored run (t_sw < crossing) with t_R = crossing
    c["t_sw"] = c["t_sw"].copy(); c["t_sw"][0] = 2500.0
    s = X.score_2a(**c)
    assert s["n_scored_rule_not_before_crossing"] == 1 and not s["validity"]["rule_before_all_scored"]
    assert not s["valid"] and _v(s) == ("UNRESOLVED",) * 3
    c["t_sw"][0] = 3000.0                                           # the same run unscored (t_sw not before): valid
    s = X.score_2a(**c)
    assert s["valid"] and s["n_scored_rule_not_before_crossing"] == 0 and s["n_scored"] == 79


def test_degenerate_statistics_are_unresolved():
    # V1-V3 hold but predictions are missing, leaving < 3 scored runs: A1 needs 3, A2 and A3 need 1
    c = _case(n_cross=80)
    c["r_pred"] = c["r_pred"].copy(); c["r_pred"][2:] = np.nan
    s = X.score_2a(**c)
    assert s["valid"] and s["n_scored"] == 2
    assert s["A1"]["verdict"] == "UNRESOLVED" and s["A2"]["verdict"] == "PASS" and s["outcome"] == "UNRESOLVED"
    c["r_pred"][:] = np.nan
    s = X.score_2a(**c)
    assert _v(s) == ("UNRESOLVED",) * 3


def test_secondary_closed_form_is_reported_not_scored():
    c = _case()
    s = X.score_2a(**c, r_secondary=np.full(80, 10.0))
    assert s["outcome"] == "PASS" and s["secondary_closed_form"]["frac_within_10pct"] == 0.0
    assert s["secondary_closed_form"]["note"] == "reported, not scored"


def test_non_crossing_runs_are_ignored():
    c = _case(n_cross=64)
    s = X.score_2a(**c)
    assert s["n_crossed"] == 64 and s["n_scored"] == 64 and s["valid"]
