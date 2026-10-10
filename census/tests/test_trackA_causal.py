"""Track A causal Adam (src/causal_adam.py, src/trackA_causal.py): the frozen rules slin_20 and coupled_sw on synthetic
paths, their source hash, constructed PASS / FAIL / UNRESOLVED cases for C1-C4 and every validity condition, the pilot
tolerance rule, the guards (CausalityViolation on v̂ / M / w₂ at row t_c and on a second hidden row; a bypass fails the
NaN recomputation), the OpenTimestamps guard on `run`, and an EXACT reproduction of one exploration run
(results/designs/trackA_explore/, seed 9,771,000 at a = 1.85)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import causal_adam as CA
from src import causal_forecast as C
from src import linear_response as LR
from src import trackA_causal as T

ROOT = Path(__file__).resolve().parents[1]
EXPL = ROOT / "results" / "designs" / "trackA_explore"


# ================================================================================================ registered settings
def test_registered_settings_match_the_approved_page():
    assert T.A == 1.77 and T.BUDGET == 32_000 and T.LR_ADAM == 0.01
    assert (T.F, T.F_DESC) == (0.90, 0.95)
    assert T.SEEDS == tuple(range(9_776_000, 9_776_100)) and len(T.SEEDS) == 100
    assert T.PILOT_SEEDS == tuple(range(9_774_000, 9_774_040)) and len(T.PILOT_SEEDS) == 40
    assert not set(T.SEEDS) & set(T.PILOT_SEEDS)
    assert (T.EXT, T.PRULE) == ("slin_20", "coupled_sw")
    assert (T.WITHIN_MIN_FRAC, T.CUTOFF_BEFORE_MIN_FRAC, T.MIN_CROSS) == (0.8, 0.9, 60)
    assert (T.TOL_FACTOR, T.TOL_STEP, T.TOL_BAND) == (1.5, 5, 0.05)
    assert (CA.SLIN_K, CA.SLIN_MIN_POINTS, CA.HORIZON) == (20, 3, 1.6)
    assert (CA.B1, CA.B2, CA.EPS) == (0.9, 0.999, 1e-8)


def test_frozen_rule_hash_recomputes():
    """The committed hash of the frozen rules (slin_20, coupled_sw) equals the hash of their current source."""
    d = json.loads((T.OUT / "frozen_rules.json").read_text())
    assert d["sha256"] == CA.rule_hashes()
    assert d["rules"] == {"extrapolation": "slin_20", "preconditioner": "coupled_sw"}
    T.assert_rules_frozen()


def test_rule_hash_covers_the_constants(monkeypatch):
    h0 = CA.rule_hashes()["all_with_constants"]
    monkeypatch.setattr(CA, "SLIN_K", 21)
    assert CA.rule_hashes()["all_with_constants"] != h0


# ================================================================================================ slin_20 (synthetic)
def test_slin_exact_line_anchored_and_stops_at_horizon():
    s = 0.5 + 0.01 * np.arange(300)                   # exactly linear
    sh, st, info = CA.slin_extrapolate(s, 200, 3.0, 10_000)
    assert st == "ok" and info["n_window"] == 20 and abs(info["slope"] - 0.01) < 1e-12
    k = np.arange(1, len(sh) + 1)
    assert np.allclose(sh, s[-1] + 0.01 * k, atol=1e-12)
    assert sh[-1] >= 4.8 and sh[-2] < 4.8               # stops at the first ŝ ≥ 1.6·s*_frozen (inclusive)


def test_slin_window_is_confined_to_the_rule_point_and_at_least_3():
    s = np.concatenate([np.full(100, 9.0), 0.5 + 0.01 * np.arange(10)])   # a burst before t_R = 100
    sh, st, info = CA.slin_extrapolate(s, 100, 1.0, 10_000)
    assert info["n_window"] == 10 and abs(info["slope"] - 0.01) < 1e-12
    _, _, info = CA.slin_extrapolate(s, 108, 1.0, 10_000)
    assert info["n_window"] == 3                         # min(20, t_c − t_R) = 2 -> at least 3 points


def test_slin_anchored_at_the_last_point_with_noise():
    rng = np.random.default_rng(0)
    s = 0.5 + 0.01 * np.arange(50) + 0.002 * rng.standard_normal(50)
    sh, st, info = CA.slin_extrapolate(s, 0, 1.0, 10_000)
    assert st == "ok" and abs((sh[0] - s[-1]) - info["slope"]) < 1e-12


@pytest.mark.parametrize("slope", [0.0, -0.01])
def test_slin_no_forecast_when_not_growing(slope):
    s = 1.0 + slope * np.arange(50)
    sh, st, _ = CA.slin_extrapolate(s, 0, 1.0, 10_000)
    assert len(sh) == 0 and st == "no forecast: not growing at the cutoff"


def test_slin_respects_the_budget():
    s = 0.5 + 1e-6 * np.arange(50)
    sh, st, _ = CA.slin_extrapolate(s, 0, 1.0, 7)
    assert st == "ok" and len(sh) == 7 and sh[-1] < 1.6


# ================================================================================================ coupled_sw (synthetic)
class _FakeBranch:
    def __init__(self, H, th=(0.0, 0.0, 0.0)):
        self.H, self.th = np.asarray(H, float), np.asarray(th, float)

    def contains(self, s):
        return True

    def theta(self, s):
        return self.th.copy()

    def hess(self, s):
        return self.H


def test_coupled_carry_first_step_is_the_beta2_recursion(monkeypatch):
    monkeypatch.setattr(CA.LR, "gap_exact", lambda w1, b1, a: -1.0)          # never crosses
    H = np.array([[2.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 0.5]])
    d0, m0, v0 = np.array([1e-3, -2e-3, 5e-4]), np.array([1e-4, 0.0, -1e-4]), np.array([4e-6, 1e-6, 2e-6])
    t_c = 300
    s_ext = np.linspace(1.0, 1.1, 400)
    t_hit, st, vh = CA.coupled_vhat(s_ext, t_c - 1, _FakeBranch(H), 1.77, d0, 0.01, m0, {}, v0, t_c)
    g = H @ d0
    v = CA.B2 * v0 + (1 - CA.B2) * g ** 2
    assert np.array_equal(vh[t_c], v / (1 - CA.B2 ** t_c))
    assert t_hit is None and st == "no forecast: no predicted crossing on the extrapolated path"
    assert min(vh) == t_c and max(vh) == len(s_ext) - 1


def test_coupled_carry_with_zero_gradient_is_the_decay(monkeypatch):
    monkeypatch.setattr(CA.LR, "gap_exact", lambda w1, b1, a: -1.0)
    t_c, t0 = 250, 200
    P_vis = {t: np.ones(3) for t in range(t0 + 1, t_c)}
    v0 = np.array([4e-6, 1e-6, 2e-6])
    _, _, vh = CA.coupled_vhat(np.linspace(1, 1.1, 300), t0, _FakeBranch(np.zeros((3, 3))), 1.77, np.full(3, 1e-3),
                               0.01, np.zeros(3), P_vis, v0, t_c)
    for t in (t_c, t_c + 7, 299):
        assert np.allclose(vh[t], v0 * CA.B2 ** (t - t_c + 1) / (1 - CA.B2 ** t), rtol=1e-12)
    assert min(vh) == t_c                                 # visible P before the cutoff: nothing carried there


def test_coupled_sw_vhat_three_sources():
    V = np.abs(np.random.default_rng(1).standard_normal((60, 4))) * 1e-6
    t_c = 40
    Vg = C.guard(V, t_c, name="Vhat")
    vh = {45: np.array([1.0, 2.0, 3.0])}
    v, src = CA.coupled_sw_vhat(45, vh, Vg, t_c)
    assert src == "coupled" and v[CA.HID].tolist() == [1.0, 2.0, 3.0]
    v, src = CA.coupled_sw_vhat(30, vh, Vg, t_c)
    assert src == "visible" and np.array_equal(v, V[30])
    v, src = CA.coupled_sw_vhat(50, vh, Vg, t_c)
    assert src == "decay"
    assert np.allclose(v, V[39] * (1 - CA.B2 ** 39) * CA.B2 ** 11 / (1 - CA.B2 ** 50), rtol=1e-14)
    assert Vg.max_index_read == 39
    v, src = CA.coupled_sw_vhat(t_c, {}, Vg, t_c)          # t_sw,fc = t_c with no carry: decay from row t_c − 1
    assert src == "decay" and Vg.max_index_read == t_c - 1
    with pytest.raises(C.CausalityViolation):
        Vg[t_c]


# ================================================================================================ criteria (constructed)
def _R(n=100, err=1, lag=30, ratio=1.0, miss=0, crossed=None, before_fail=0, tsw_undef=0, nan_bad=0, c4_bad=False):
    t_sw = np.full(n, 1000.0)
    t_obs = t_sw + lag
    t_fc = t_obs + err
    t_sw_fc = t_sw.copy()
    if c4_bad:
        t_sw_fc = t_obs.copy()                             # the no-lag forecast is exact: D > 0
    s_sw = np.full(n, 1.0)
    r_obs = 0.1
    s_obs = s_sw * (1 + r_obs)
    r_fc = np.full(n, r_obs / ratio)
    t_c = t_sw - 50
    t_fc[:miss] = np.nan
    t_c[:before_fail] = t_obs[:before_fail] + 1
    t_sw[n - tsw_undef:] = np.nan
    nan_ok = np.ones(n, bool)
    nan_ok[:nan_bad] = False
    return {"frozen_ok": np.ones(n, bool), "crossed": np.ones(n, bool) if crossed is None else crossed,
            "t_obs": t_obs, "s_obs": s_obs, "t_sw": t_sw, "s_sw": s_sw, "t_c": t_c, "t_fc": t_fc, "t_sw_fc": t_sw_fc,
            "r_fc": r_fc, "nan_identical": nan_ok}


TOL = {"tau_cross": 10, "tau_lag": 10, "band": 0.20}


def test_pass():
    r = T.score_track_a(_R(), TOL)
    assert r["valid"] and r["outcome"] == "PASS"
    assert [r[k]["verdict"] for k in ("C1", "C2", "C3", "C4")] == ["PASS"] * 4


def test_c1_c2_fail_and_boundary():
    r = T.score_track_a(_R(err=11), TOL)
    assert r["C1"]["verdict"] == "FAIL" and r["C2"]["verdict"] == "FAIL" and r["outcome"].startswith("FAIL")
    assert T.score_track_a(_R(err=10), TOL)["C1"]["verdict"] == "PASS"        # |err| ≤ τ (closed)


@pytest.mark.parametrize("miss,verdict", [(20, "PASS"), (21, "FAIL")])
def test_a_miss_counts_as_not_within(miss, verdict):
    r = T.score_track_a(_R(miss=miss), TOL)
    assert r["C1"]["verdict"] == verdict and r["C1"]["n"] == 100 and r["C1"]["n_miss"] == miss
    assert r["C3"]["n"] == 100 - miss                      # misses excluded from C3 and C4
    assert r["C4"]["n"] == 100 - miss
    assert r["n_scored_miss"] == miss


@pytest.mark.parametrize("ratio,verdict", [(1.19, "PASS"), (1.21, "FAIL"), (0.81, "PASS"), (0.79, "FAIL")])
def test_c3_band(ratio, verdict):
    assert T.score_track_a(_R(ratio=ratio), TOL)["C3"]["verdict"] == verdict


def test_c4_fail_when_no_better_than_no_lag():
    r = T.score_track_a(_R(c4_bad=True), TOL)
    assert r["C4"]["verdict"] == "FAIL" and r["C4"]["ci95"][0] > 0 and r["outcome"] == "FAIL C2+C4"
    assert r["falsifier_C4_fails"]


def test_c4_bootstrap_is_1c_procedure_with_own_seed():
    from src import phase1c as P1
    D = np.random.default_rng(3).standard_normal(50)
    assert T.BOOT_SEED == 9_776_000
    assert P1.bootstrap_mean_ci(D, P1.BOOT_N, T.BOOT_SEED) == P1.bootstrap_mean_ci(D, seed=T.BOOT_SEED)


@pytest.mark.parametrize("kw,flag", [
    ({"crossed": np.r_[np.ones(59, bool), np.zeros(41, bool)]}, "min_60_crossings"),
    ({"before_fail": 41}, "min_60_scored"),
    ({"before_fail": 11}, "cutoff_before_crossing_90pct"),
    ({"tsw_undef": 1}, "actual_switch_defined"),
    ({"nan_bad": 1}, "nan_recomputation_identical")])
def test_validity_unresolved(kw, flag):
    r = T.score_track_a(_R(**kw), TOL)
    assert not r["validity"][flag] and not r["valid"] and r["outcome"] == "UNRESOLVED (validity)"
    assert all(r[k]["verdict"] == "UNRESOLVED" for k in ("C1", "C2", "C3", "C4"))


def test_validity_boundaries_pass():
    assert T.score_track_a(_R(before_fail=10), TOL)["valid"]                  # 90/100 before: ok
    r = T.score_track_a(_R(crossed=np.r_[np.ones(60, bool), np.zeros(40, bool)]), TOL)
    assert r["valid"] and r["n_scored"] == 60


def test_unstable_run_is_a_miss():
    R = _R()
    R["t_fc"][0] = np.nan                                 # R4-unstable: no t_fc
    r = T.score_track_a(R, TOL)
    assert r["n_scored_miss"] == 1 and r["C1"]["n_within"] == 99


# ================================================================================================ pilot rule
def _err(q90c, q90l, med, q25, q75, n=38):
    return {"abs_err_cross": {"q90": q90c}, "abs_err_lag": {"q90": q90l},
            "ratio_r_obs_over_r_fc": {"median": med, "q25": q25, "q75": q75, "n": n, "sd": 0.2}}


def test_pilot_rule():
    t = T.pilot_tolerances(_err(6, 4, 0.94, 0.85, 1.05))
    se = 1.2533 * (0.20 / 1.349) / math.sqrt(38)
    assert t["tau_cross"] == 10 and t["tau_lag"] == 10                       # 9 -> 10, 6 -> 10
    assert abs(t["se_median"] - se) < 1e-15
    assert t["band"] == T._ceil_to(0.06 + 2 * se, 0.05) == 0.15
    t = T.pilot_tolerances(_err(10, 2, 1.0, 1.0, 1.0))
    assert t["tau_cross"] == 15 and t["tau_lag"] == 5 and t["band"] == 0.0   # exact multiples stay; no floor


def test_pilot_errors_use_the_scored_definition_and_forecasts_only():
    R = _R(n=10, err=3, miss=2, before_fail=1)
    R = {k: list(v) for k, v in R.items()}
    R.update(seed=list(range(10)), status=["s"] * 10, P_relerr=[[0, 0, 0]] * 10)
    e = T.pilot_errors(R)
    assert e["n_pilot_set"] == 9 and e["n_with_forecast"] == 8 and e["n_miss"] == 1
    assert e["abs_err_cross"]["q90"] == 3.0


# ================================================================================================ scan patterns
def test_a_pattern():
    import re
    rx = re.compile(T.A_PATTERN)
    for s in ("a = 1.77", "1.77,", "1.7700", "x 1.769999 y", "1.7699999999,"):
        assert rx.search(s), s
    for s in ("1.775", "11.77", "1.7", "1.78", "1.177", "131.769999755892", "1.7699"):
        assert not rx.search(s), s


# ================================================================================================ OTS guard
def test_run_refuses_without_the_ots_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "OUT", tmp_path)
    monkeypatch.setattr(T, "train", lambda *a, **k: pytest.fail("trained without the OTS proof"))
    for fn in (T.run, T.observe):
        with pytest.raises(SystemExit, match="REFUSED"):
            fn()
    (tmp_path / "registration_stamp.txt.ots").write_bytes(b"x")       # proof present, no manifest: still refuses
    with pytest.raises((AssertionError, FileNotFoundError)):
        T.run()


def test_real_out_has_no_ots_proof_yet_and_run_refuses():
    if (T.OUT / "registration_stamp.txt.ots").exists():
        pytest.skip("registered: the proof exists")
    with pytest.raises(SystemExit, match="REFUSED"):
        T.run()


# ================================================================================================ real run (exploration seed)
@pytest.fixture(scope="module")
def run_1p85():
    """Exploration seed 9,771,000 at a = 1.85 (2A's landscape), as explore.py ran it."""
    land = json.loads((ROOT / "results" / "track2a" / "landscape.json").read_text())
    fr, x, y = T.s_frozen_of(9_771_000, land, 1.85)
    W, M, V = T.train(9_771_000, 1.85, 32_000)
    return fr["s_frozen"], x, y, W, M, V


def _expl(fname, seed=9_771_000):
    for ln in (EXPL / fname).read_text().splitlines():
        r = json.loads(ln)
        if r["seed"] == seed:
            return r


def test_reproduces_the_exploration_run_exactly(run_1p85):
    s_fr, x, y, W, M, V = run_1p85
    e1, e2 = _expl("explore_runs.jsonl"), _expl("explore2_runs.jsonl")
    assert s_fr == e1["s_frozen"]
    from src.phase1a_pilot import w1_observe
    assert w1_observe(W, a=1.85) == e1["t_obs"] and abs(W[e1["t_obs"], 2]) == e1["s_obs"]
    for f in ("0.9", "0.95"):
        t_c, tR, s_run = T.harness_cutoff(W, float(f), s_fr, x, y, 1.85)
        assert t_c == e1["fc"][f]["t_c"] and s_run == e1["fc"][f]["s_run_cut"]
        for ext, rec in (("slin_20", e2["fc"][f]["slin_20"]), ("ext_1c", e1["fc"][f]["ext"])):
            for prule, key in (("coupled_sw", "coupled_sw"), ("cut", "cut")):
                fc = T.forecast_with_nan_check(W, M, V, x, y, s_fr, t_c, ext, prule, 1.85, 32_000)
                ref = rec["P"][key]
                assert fc["t_rule"] == rec["t_rule"] and fc["s_run"] == rec["s_run"]
                assert fc["t_sw_fc"] == rec["t_sw_fc"]
                assert fc["t_fc"] == ref["t_fc"] and fc["s_fc"] == ref["s_fc"] and fc["r_fc"] == ref["r_fc"], (f, ext)
                assert fc["nan_recompute_identical"]
                assert max(fc["max_index_read"].values()) < t_c


def test_guards_raise_at_row_t_c(run_1p85):
    s_fr, x, y, W, M, V = run_1p85
    t_c = 387
    inp = CA.guarded_inputs(1.85, x, y, s_fr, t_c, 32_000, W, M, V)
    for g in (inp.Vhat, inp.M, inp.out):
        with pytest.raises(C.CausalityViolation):
            g[t_c]
        with pytest.raises(C.CausalityViolation):
            g[t_c - 5:t_c + 1]
    inp.hid[127]
    with pytest.raises(C.CausalityViolation):            # a second hidden row
        inp.hid[128]


def test_a_leaky_forecaster_is_caught(run_1p85, monkeypatch):
    s_fr, x, y, W, M, V = run_1p85
    monkeypatch.setattr(CA, "coupled_sw_vhat", lambda t_sw, vh, Vg, t_c: (C._row(Vg, t_c), "leak"))
    with pytest.raises(C.CausalityViolation):
        T.forecast_with_nan_check(W, M, V, x, y, s_fr, 387, a=1.85)


def test_a_guard_bypass_fails_the_nan_recomputation(run_1p85, monkeypatch):
    s_fr, x, y, W, M, V = run_1p85

    def bypass(t_sw, vh, Vg, t_c):
        data = Vg._GuardedArray__data if isinstance(Vg, C.GuardedArray) else np.asarray(Vg)
        return np.array(data[t_c]), "bypass"
    monkeypatch.setattr(CA, "coupled_sw_vhat", bypass)
    fc = T.forecast_with_nan_check(W, M, V, x, y, s_fr, 387, a=1.85)
    assert fc["nan_recompute_identical"] is False
    R = _R()
    R["nan_identical"][5] = False
    assert T.score_track_a(R, TOL)["outcome"] == "UNRESOLVED (validity)"


def test_forecast_rows_hold_no_observed_quantity(run_1p85):
    s_fr, x, y, W, M, V = run_1p85
    fcs = T.forecasts(W, M, V, x, y, s_fr, a=1.85)
    assert not (T.FORBIDDEN & T.P1._keys(fcs))
    p = fcs["0.9"]["primary"]
    assert p["P_source"] in ("coupled", "visible", "decay") and p["ext"] == "slin_20" and p["P_rule"] == "coupled_sw"
    assert set(fcs["0.9"]) >= {"primary", "P_cut", "ext_1c"} and "P_cut" not in fcs["0.95"]


def test_pilot_tolerances_recompute_from_the_committed_pilot_runs():
    """pilot.json's tolerances = the pilot rule on pilot_runs.jsonl (pilot seeds only), with the frozen rule hashes."""
    d = json.loads((T.OUT / "pilot.json").read_text())
    rows = T._rows(T.OUT / "pilot_runs.jsonl")
    assert sorted(r["seed"] for r in rows) == list(T.PILOT_SEEDS)
    assert not set(r["seed"] for r in rows) & set(T.SEEDS)
    obs = [r["observed_PILOT_ONLY"] for r in rows if r.get("observed_PILOT_ONLY")]
    err = T.pilot_errors(T.table(rows, obs, T.F, "primary"))
    t = T.pilot_tolerances(err)
    assert {k: t[k] for k in ("tau_cross", "tau_lag", "band")} == {k: d["tolerances"][k] for k in ("tau_cross", "tau_lag",
                                                                                                   "band")}
    assert d["rule_sha256"] == CA.rule_hashes()


def test_registered_tolerances_are_the_approved_values():
    """Author 2026-10-09: τ_cross 10, τ_lag 10, b 0.10 (thin b margin a known risk); equal to pilot.json's rule output."""
    assert T.TOLERANCES == {"tau_cross": 10, "tau_lag": 10, "band": 0.10}
    assert T.tolerances() == T.TOLERANCES


def test_observed_lag_measured_as_1c_width1():
    """D8 (author's condition): t_sw = the first step with s ≥ the cutoff's s*_run, s_sw = s*_run, r_obs = s_obs/s_sw − 1,
    lag_obs = t_obs − t_sw, as src/phase1c.py observe_one (W1) and score_arm_1c."""
    import inspect
    src = inspect.getsource(T.observe_one)
    assert 'rec.get("cutoff_s_run")' in src and "LR._first_ge(s, s_run)" in src and '"s_sw": s_run' in src
    assert 'R["s_obs"] / R["s_sw"] - 1' in inspect.getsource(T.score_track_a)
    assert "(t_obs - t_sw)" in inspect.getsource(T.criteria)
