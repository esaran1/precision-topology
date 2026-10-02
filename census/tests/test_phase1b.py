"""Phase 1B (src/phase1b.py, POST HOC causal re-scoring): the decisions fixed before any outcome, the frozen forecaster's
hash, constructed cases for every new function (raw_errors, rescore incl. misses / late cutoffs / the 90% rule / the NaN
rule / D without a tolerance, same_core), the registered scored sets and the registered verdicts reproduced, and the
ENFORCEMENT test for every family (width 1 SGD and Adam with both P rules, GELU-T, W2-A T and D): the forecaster reads
only rows < t_c (every guard's last row < t_c; poisoning every row after the cutoff leaves the record identical; the
NaN recomputation is identical; a forecaster touching the cutoff row raises CausalityViolation)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src import causal_forecast as C
from src import phase1b as B
from src import phase1c as P1C

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "results" / "phase1a" / "fixtures"


# ================================================================================================ decisions
def test_decisions_fixed_before_outcomes():
    assert B.ARMS == ("track_a/sgd", "track_a/adam", "track2a/adam", "gelu/random", "gelu/branch", "w2a/T", "w2a/D",
                      "w2a/Tp")
    for a in ("track_a/sgd", "track_a/adam", "track2a/adam"):
        assert B.F[a] == 0.90 == P1C.F["W1"] and B.F_OTHER[a] == 0.95 and B.TOL_KEY[a] == "W1"
    for a in ("gelu/random", "gelu/branch"):
        assert B.F[a] == 0.95 == P1C.F["G"] and B.F_OTHER[a] == 0.90 and B.TOL_KEY[a] == "G"
    assert (B.TOL_KEY["w2a/T"], B.TOL_KEY["w2a/Tp"], B.TOL_KEY["w2a/D"]) == ("T", "Tp", None)
    assert all(B.F[a] == 0.95 and B.F_OTHER[a] == 0.90 for a in ("w2a/T", "w2a/D", "w2a/Tp"))
    assert B.ADAM_P == {"track_a/adam": "rule_point", "track2a/adam": "cutoff"}
    assert B.ADAM_P_OTHER == {"track_a/adam": "cutoff", "track2a/adam": "rule_point"}


def test_config_is_phase1c_config():
    for a in B.ARMS:
        for f in (B.F[a], B.F_OTHER[a]):
            cfg = B.config(a, f, B.ADAM_P.get(a))
            ref = P1C.config("W1", f)
            assert (cfg.f, cfg.family, cfg.window_frac, cfg.min_window, cfg.horizon_factor) == \
                (ref.f, "quad", 0.05, 50, 1.6) == (f, ref.family, ref.window_frac, ref.min_window, ref.horizon_factor)
    assert B.config("track2a/adam", 0.9, "cutoff").adam_P == "cutoff"


def test_forecaster_is_the_frozen_one():
    assert B.assert_forecaster() == B.FORECASTER_SHA256


# ================================================================================================ constructed cases
def test_raw_errors_constructed():
    t_obs = np.array([100., 200., 300., 400.])
    t_sw = np.array([90., 190., 290., 390.])
    t_fc = np.array([102., 197., 300., np.nan])
    t_sw_fc = np.array([91., 190., 285., np.nan])
    r_obs = np.array([0.1, 0.2, 0.3, 0.4])
    r_fc = np.array([0.1, 0.1, 0.6, np.nan])
    has = P1C.is_forecast(t_fc, t_sw_fc)
    r = B.raw_errors(t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs, has)
    assert r["n"] == 4 and r["n_miss"] == 1
    assert r["abs_err_cross"]["median"] == 2.0 and r["abs_err_cross"]["max"] == 3.0
    assert r["abs_err_cross"]["signed_median"] == 0.0
    # lag errors: (11 - 10), (7 - 10), (15 - 10) -> 1, -3, 5
    assert r["abs_err_lag"]["median"] == 3.0 and r["abs_err_lag"]["signed_median"] == 1.0
    assert r["ratio_r_obs_over_r_fc"]["median"] == 1.0 and r["ratio_r_obs_over_r_fc"]["n"] == 3
    # D = |t_fc - t_obs| - |t_sw_fc - t_obs| = 2 - 9, 3 - 10, 0 - 15
    assert r["D_vs_no_lag"]["n"] == 3 and r["D_vs_no_lag"]["n_D_negative"] == 3
    assert r["D_vs_no_lag"]["mean"] == pytest.approx(-29 / 3)
    lo, hi = r["D_vs_no_lag"]["ci95"]
    assert lo <= -29 / 3 <= hi < 0


def _w1_like(n=80, err=3.0):
    t_obs = 1000.0 + np.arange(n)
    t_sw = t_obs - 50
    t_c = t_obs - 200
    t_fc = t_obs + err
    t_sw_fc = t_sw + err
    r_obs = np.full(n, 0.1)
    r_fc = np.full(n, 0.1)
    return dict(base=np.ones(n, bool), t_c=t_c, t_fc=t_fc, t_sw_fc=t_sw_fc, r_fc=r_fc, t_obs=t_obs, t_sw=t_sw,
                r_obs=r_obs, nan_identical=np.ones(n, bool))


def _rs(key, w1, **k):
    return B.rescore(key, k["base"], k["t_c"], k["t_fc"], k["t_sw_fc"], k["r_fc"], k["t_obs"], k["t_sw"], k["r_obs"],
                     k["nan_identical"], w1)


def test_rescore_pass_and_counts():
    r = _rs("W1", True, **_w1_like())
    assert r["outcome_criteria"] == "PASS" == r["outcome_1c_rule"] and r["valid_1c"]
    assert r["n_scored"] == 80 and r["n_scored_miss"] == 0
    assert r["criteria"]["C1"]["n_within"] == 80 and r["criteria"]["C3"]["median_ratio"] == 1.0
    assert r["horizon"]["cross_steps"]["median"] == 200.0


def test_rescore_fail_c1_outside_tolerance():
    r = _rs("W1", True, **_w1_like(err=11.0))
    assert r["criteria"]["C1"]["verdict"] == "FAIL" and r["outcome_criteria"].startswith("FAIL C1")


def test_rescore_misses_count_against_c1_and_late_cutoffs_are_unscored():
    k = _w1_like()
    k["t_fc"][:20] = np.nan                     # 20 misses (scored, no forecast)
    k["t_c"][-5:] = k["t_obs"][-5:]             # 5 cutoffs not strictly before the crossing: unscored
    r = _rs("W1", True, **k)
    assert r["n_scored"] == 75 and r["n_scored_miss"] == 20 and r["n_base_cutoff_not_before"] == 5
    assert r["criteria"]["C1"]["n_within"] == 55 and r["criteria"]["C1"]["verdict"] == "FAIL"   # 55/75 < 0.8
    assert r["cutoff_validity"]["n_cutoff_before_crossing"] == 75


def test_rescore_90pct_rule_and_nan_rule_make_the_1c_outcome_unresolved():
    k = _w1_like()
    k["t_c"][:9] = np.nan                        # 71/80 < 90%
    r = _rs("W1", True, **k)
    assert not r["validity_1c"]["cutoff_before_crossing_90pct"] and r["outcome_1c_rule"] == "UNRESOLVED (validity)"
    assert r["outcome_criteria"] == "PASS" and r["n_base_no_cutoff"] == 9
    k = _w1_like()
    k["nan_identical"][3] = False
    r = _rs("W1", True, **k)
    assert not r["validity_1c"]["nan_recomputation_identical"] and r["outcome_1c_rule"] == "UNRESOLVED (validity)"
    assert r["n_nan_recompute_differs"] == 1


def test_rescore_width1_needs_60_scored_runs():
    k = {a: v[:59] for a, v in _w1_like().items()}
    r = _rs("W1", True, **k)
    assert not r["validity_1c"]["W1_min_60_scored"] and r["outcome_1c_rule"] == "UNRESOLVED (validity)"
    assert "W1_min_60_scored" not in _rs("G", False, **k)["validity_1c"]


def test_rescore_without_tolerance_reports_raw_errors_only():
    r = _rs(None, False, **_w1_like())
    assert r["criteria"] is None and r["outcome_criteria"] == "no tolerance set: raw errors only"
    assert r["raw"]["abs_err_cross"]["median"] == 3.0


def test_rescore_sign_criterion_for_tp():
    k = _w1_like(n=20)
    k["t_sw"] = k["t_obs"] + 40                  # observed lag −40
    k["t_sw_fc"] = k["t_fc"] + 40                # forecast lag −40
    r = _rs("Tp", False, **k)
    assert r["criteria"]["S"]["verdict"] == "PASS" and r["criteria"]["S"]["n_both_negative"] == 20


def test_same_core():
    a = {"verdict": "PASS", "n": 80, "median_ratio": 1.0000000000000087, "ci95": [-0.1, -0.05]}
    assert B.same_core(a, {**a, "median_ratio": 1.000000000000008})
    assert not B.same_core(a, {**a, "verdict": "FAIL"})
    assert not B.same_core(a, {**a, "n": 79})
    assert not B.same_core(a, {**a, "median_ratio": 1.001})
    assert not B.same_core(a, {**a, "ci95": [-0.1, -0.06]})


# ================================================================================================ registered sets
@pytest.mark.parametrize("arm,n_runs,n_scored", [("track_a/sgd", 80, 63), ("track_a/adam", 80, 74),
                                                 ("track2a/adam", 80, 75), ("gelu/random", 80, 80),
                                                 ("gelu/branch", 80, 80), ("w2a/T", 200, 98), ("w2a/D", 120, 120),
                                                 ("w2a/Tp", 120, 120)])
def test_registered_scored_sets_and_verdicts_reproduced(arm, n_runs, n_scored):
    d = B.registered_table(arm)
    assert len(d) == n_runs and int(d.reg_scored.sum()) == n_scored
    sc = B.registered_scores(arm)
    k = B.REG_CRITERIA[B.test_of(arm)][0]
    assert sc[k]["n"] == n_scored if B.test_of(arm) != "w2a" else sc["n_scored"] == n_scored
    assert B.registered_reproduced(arm, d)


# ================================================================================================ ENFORCEMENT
SEED_W1 = 9_310_002                     # a Phase 1A dev seed (not a registered seed, not a 1C seed)


@pytest.fixture(scope="module")
def w1_runs():
    """Short width-1 runs at Track A's a = 1.65 (Track A's protocol), and s*_frozen by Track A's R0 rule."""
    from src import linear_response as LR
    from src import track_a as TA
    from src.phase1a_pilot import w1_train
    d, th_pop, *_ = TA._land()
    s_pop = d["s_star_pop"]
    x, y = TA._sample(SEED_W1)
    th, res, _ = LR.newton(th_pop, s_pop, TA.A, x, y)
    s_fr = LR.Branch(s_pop, th, TA.A, x, y, 0.3 * s_pop, 1.7 * s_pop, 0.002 * s_pop).switch(s_pop)
    return {opt: (s_fr,) + tuple(w1_train(SEED_W1, opt=opt, budget=2000, a=TA.A)) for opt in ("sgd", "adam")}


def _poison(a, t, how):
    b = np.array(a, float, copy=True)
    b[t:] = np.nan if how == "nan" else 1e3 * (1 + np.arange(len(b) - t)).reshape(-1, *([1] * (b.ndim - 1)))
    return b


def _w1(w1_runs, opt, poison=None):
    from src import track_a as TA
    s_fr, W, M, V = w1_runs[opt]
    case = B.w1_case(TA.A, SEED_W1, s_fr, opt, W if poison is None else poison(W))
    if opt == "adam":
        case.M, case.V = (M, V) if poison is None else (poison(M), poison(V))
    return case


def _fixture(name):
    p = FIX / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    d = np.load(p, allow_pickle=False)
    return d, json.loads(str(d["meta"]))["row"]


def _hid_full(n, hid0):
    H = np.full((n, len(hid0)), np.nan)
    H[0] = hid0
    return H


def _gelu(poison=None):
    d, row = _fixture("gelu_random")
    out = np.array(d["out"])
    hid = _hid_full(len(out), np.array(d["hid"])[0])
    cf = row["frozen"]["copies"][str(row["release"]["copy_at_release"])]
    p = poison or (lambda a: a)
    return B.gelu_case(row["seed"], cf, p(out), p(hid), row["budget"])


def _w2a_T(poison=None):
    d, row = _fixture("w2a_T")
    out = np.array(d["out"])
    hid = _hid_full(len(out), np.array(d["hid"])[0])
    copy = row["release"]["copy_at_release"]
    p = poison or (lambda a: a)
    return B.w2a_case("T", row["seed"], row["frozen"]["copies"][copy], row["release"]["windings"], copy, p(out),
                      p(hid), row["budget"])


def _w2a_D(poison=None):
    """W2-A D (no fixture exists: D was not piloted): a registered D path, used only if present (untracked)."""
    from src import width2_asym as W
    d = B.registered_table("w2a/D")
    r = next(d[d.reg_scored].sort_values("step_obs").itertuples())
    if not W._path_file("D", int(r.seed)).exists():
        pytest.skip("W2-A D paths not present")
    X, _, _ = B.load_verified("w2a/D", r)
    X = X[: int(r.step_obs) + 2000]
    p = poison or (lambda a: a)
    cf = W._frozen()[int(r.seed)]["copies"][r.copy_at_release]
    return B.w2a_case("D", int(r.seed), cf, json.loads(r.windings), r.copy_at_release, p(X[:, W.VI]), p(X[:, W.ZI]),
                      int(r.budget))


FAMILIES = [("w1_sgd", "track_a/sgd", None), ("w1_adam", "track_a/adam", "rule_point"),
            ("w1_adam", "track2a/adam", "cutoff"), ("gelu", "gelu/random", None), ("w2a_T", "w2a/T", None),
            ("w2a_D", "w2a/D", None)]


def _make(fam, w1_runs, poison=None):
    if fam == "w1_sgd":
        return _w1(w1_runs, "sgd", poison)
    if fam == "w1_adam":
        return _w1(w1_runs, "adam", poison)
    return {"gelu": _gelu, "w2a_T": _w2a_T, "w2a_D": _w2a_D}[fam](poison)


def _strip(rec):
    return {k: v for k, v in rec.items() if k != "secs"}


@pytest.mark.parametrize("fam,arm,adam_P", FAMILIES)
def test_forecaster_sees_only_rows_before_the_cutoff(fam, arm, adam_P, w1_runs):
    cfg = B.config(arm, B.F[arm], adam_P)
    case = _make(fam, w1_runs)
    rec = B.forecast_one(case, cfg)
    t_c = rec["t_c"]
    assert t_c is not None and rec["status"] == "ok" and rec["t_fc"] is not None
    assert max(rec["max_index_read"].values()) < t_c
    assert rec["nan_recompute_identical"] is True
    if fam == "w1_adam":
        assert rec["max_index_read"]["M"] == rec["t_rule"] < t_c
        assert rec["max_index_read"]["Vhat"] == (rec["t_rule"] if adam_P == "rule_point" else t_c - 1)
    if fam.startswith("w1"):
        assert rec["max_index_read"]["hidden"] == rec["t_rule"] < t_c
    else:
        assert rec["max_index_read"]["hidden"] == 0
    for how in ("nan", "garbage"):
        # every row after the cutoff poisoned (the harness reads s at t_c to stop; the forecaster reads rows < t_c)
        rec2 = B.forecast_one(_make(fam, w1_runs, lambda a, how=how: _poison(a, t_c + 1, how)), cfg)
        assert P1C.same_forecast(_strip(rec), _strip(rec2))


@pytest.mark.parametrize("fam,arm,adam_P", FAMILIES)
def test_forecaster_touching_the_cutoff_row_is_caught(fam, arm, adam_P, w1_runs, monkeypatch):
    case = _make(fam, w1_runs)
    real = C.ADAPTERS[case.kind]

    def leaky(inp, cfg):
        inp.out[inp.t_c]                                                  # the row at the cutoff
        return real(inp, cfg)
    monkeypatch.setitem(C.ADAPTERS, case.kind, leaky)
    with pytest.raises(C.CausalityViolation):
        B.forecast_one(case, B.config(arm, B.F[arm], adam_P))


def test_adam_moments_after_the_cutoff_are_never_read(w1_runs, monkeypatch):
    case = _make("w1_adam", w1_runs)
    real = C.ADAPTERS["w1"]

    def leaky(inp, cfg):
        inp.Vhat[inp.t_c]                                                 # v̂ at the cutoff
        return real(inp, cfg)
    monkeypatch.setitem(C.ADAPTERS, "w1", leaky)
    with pytest.raises(C.CausalityViolation):
        B.forecast_one(case, B.config("track2a/adam", 0.9, "cutoff"))


def test_forecaster_bypassing_the_guard_is_caught_by_the_nan_recomputation(w1_runs, monkeypatch):
    case = _make("gelu", w1_runs)
    real = C.ADAPTERS["gelu"]

    def bypass(inp, cfg):
        res = real(inp, cfg)
        o = inp.out
        data = o._GuardedArray__data if isinstance(o, C.GuardedArray) else np.asarray(o)
        v = float(np.asarray(data[min(inp.t_c + 5, len(data) - 1)]).sum())
        return {**res, "t_fc": (res["t_fc"] or 0) + (0 if np.isfinite(v) else 1)}
    monkeypatch.setitem(C.ADAPTERS, "gelu", bypass)
    rec = B.forecast_one(case, B.config("gelu/random", 0.95))
    assert rec["nan_recompute_identical"] is False


def test_w1_cutoff_is_phase1c_nested_stopping_time(w1_runs):
    from src import track_a as TA
    case = _make("w1_sgd", w1_runs)
    t_c, ex = case.cutoff(0.9)
    tR = C.causal_rule_step(case.s[:t_c], case.s_frozen)
    assert ex["cutoff_rule_point"] == tR < t_c and case.s[t_c] >= 0.9 * ex["cutoff_s_run"]
    assert case.a == TA.A and case.lr == TA.LR_SGD and case.budget == TA.BUDGET
