"""Phase 2A (src/phase2a.py, src/causal_forecast_fold.py): the registered settings; constructed pass / fail / unresolved
cases for every gate, criterion and validity rule (F incl. the at-or-below-1.25·s* falsifier, C1, C2, C3-asymptotic
incl. a non-monotone FAIL and a ratio ≤ 1 FAIL, C4 bootstrap, E band, the no-forecast rule, the 90% validity rules, the
pilot STOP rule and the τ rule); the ENFORCEMENT test (a forecaster that touches s at or after its cutoff makes this file
fail: the guard raises CausalityViolation, a guard bypass is caught by the NaN recomputation); the NaN recomputation on
a real path; the model pieces (v3 init bit for bit from a local generator, ρ₂ = v2.rho2_batch, the scale-only rule, the
idle unit exactly 0) and a real fast-rate run checked against the committed exploration."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import causal_forecast as CF
from src import causal_forecast_fold as CFF
from src import phase2a as P

ROOT = Path(__file__).resolve().parents[1]
LAMF_EXPLORE = 0.0015099537806674871          # p2a_explore_a, ρ = 1 row (the exploratory Λ_F)
S_F = 4.767689442106793
S_STAR = 3.5913755424683727
Q = 0.3914103370353161


# ================================================================================================ registered settings
def test_registered_settings_match_the_approved_page():
    assert len(P.LADDER_LOG2) == 27 == P.N_RATES and len(set(P.LADDER_LOG2)) == 27
    assert P.LADDER_LOG2[:3] == (-13.0, -14.0, -15.0)
    assert P.LADDER_LOG2[3:] == tuple(-(15 + k / 8) for k in range(1, 25))
    assert P.LADDER_LOG2[3] == -15.125 and P.LADDER_LOG2[-1] == -18.0
    assert list(P.LADDER_LOG2) == sorted(P.LADDER_LOG2, reverse=True)            # fastest first
    # author 2026-10-04, S1 (b): the middle pilot rate moved from 2^-16.5 (= ladder k 12) to 2^-16.5625 (k 12.5)
    assert P.PILOT_LOG2 == (-15.0625, -16.5625, -17.9375) and not set(P.PILOT_LOG2) & set(P.LADDER_LOG2)
    assert [(-l2 - 15) * 8 for l2 in P.PILOT_LOG2] == [0.5, 12.5, 23.5]
    # author 2026-10-04: verdicts over the 22 rates with no prior outcome; the 5 explored rates are run, descriptive
    assert P.PREOBSERVED_LOG2 == (-13.0, -14.0, -15.0, -16.0, -17.0) and set(P.PREOBSERVED_LOG2) <= set(P.LADDER_LOG2)
    assert P.N_SCORED == 22 and P.SCORED_LOG2 == tuple(l2 for l2 in P.LADDER_LOG2 if l2 not in P.PREOBSERVED_LOG2)
    assert P.GATE_MIN_M == P.N_RATES == 27                                # all 27 still run, one per M seed
    assert (P.SEEDS[0], P.SEEDS[-1], len(P.SEEDS)) == (2_937_000, 2_937_119, 120)
    assert (P.PILOT_SEEDS[0], P.PILOT_SEEDS[-1], len(P.PILOT_SEEDS)) == (2_937_900, 2_937_959, 60)
    assert not set(P.SEEDS) & set(P.PILOT_SEEDS)
    assert not (set(P.SEEDS) | set(P.PILOT_SEEDS)) & set(range(2_930_000, 2_930_200))   # exploration seeds
    assert (P.S0, P.ETA, P.LAM, P.F_CUT, P.OBS_END, P.FOLLOW_FRAC, P.LABEL_TOL) == (1.7957, 1.0, 1e-4, 0.95, 1.25, 0.8,
                                                                                     1e-3)
    assert (P.GATE_MIN_M, P.F_MIN_FRAC, P.F_HI, P.F_FALSIFIER, P.WITHIN_MIN_FRAC, P.C3_MIN_SPEARMAN, P.E_BAND,
            P.VALID_MIN_FRAC) == (27, 0.9, 1.25, 1.25, 0.8, 0.8, (0.55, 0.80), 0.9)
    assert (P.BOOT_N, P.BOOT_PCT) == (10_000, (2.5, 97.5))
    assert (P.PILOT_EPS_MAX, P.PILOT_CHI_MAX, P.TAU_FACTOR, P.TAU_ROUND) == (0.01, 0.1, 1.5, 0.05)
    cfg = CFF.config(P.F_CUT)
    assert (cfg.f, cfg.family, cfg.window_frac, cfg.min_window, cfg.horizon_factor) == (0.95, "quad", 0.05, 50, 1.6)
    assert abs(CFF.OMEGA0 - 2.338107410459767) < 1e-15


def test_tightening_cutoff_calibration_is_descriptive():
    assert P.f_tight(2.0 ** -15) == pytest.approx(0.95, abs=1e-15)
    assert P.f_tight(2.0 ** -13) == pytest.approx(1 - 0.05 * 2 ** (4 / 3))
    assert P.f_tight(2.0 ** -18) == pytest.approx(0.9875)
    for l2 in P.LADDER_LOG2:                                        # 1 − f ∝ ρ^{2/3}
        assert (1 - P.f_tight(2.0 ** l2)) / (2.0 ** l2) ** (2 / 3) == pytest.approx(0.05 / 2 ** (-10))


def test_seed_ranges_not_registered_elsewhere():
    from src import phase1c as P1C
    mine = set(P.SEEDS) | set(P.PILOT_SEEDS)
    for mod in ("track_a", "track2a", "track2b", "track2c", "gelu_transfer", "width2_asym", "phase1a_pilot", "phase1c"):
        import importlib
        m = importlib.import_module(f"src.{mod}")
        for name in dir(m):
            if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
                assert not mine & set(P1C._flat_ints(getattr(m, name))), f"{mod}.{name}"


def test_budget_and_frozen_forecaster_untouched():
    assert P.budget(2.0 ** -15) == math.ceil(64 * 2 ** 15)
    import inspect
    src = inspect.getsource(CFF.forecast_fold)
    assert "CF.extrapolate_scale" in src and "CF._prefix(s_view, t_c)" in src


# ================================================================================================ model pieces
def test_init_is_v3_init_bit_for_bit_without_global_rng():
    import torch
    before = torch.random.get_rng_state().clone()
    rows = [P.init_row(s) for s in (2_930_000, 2_930_001, 7)]
    assert torch.equal(before, torch.random.get_rng_state())          # the global RNG is untouched
    with torch.random.fork_rng():
        for s, r in zip((2_930_000, 2_930_001, 7), rows):
            torch.manual_seed(s)
            hid = torch.nn.Linear(2, 4).double(); out = torch.nn.Linear(4, 1).double()
            ref = np.concatenate([p.detach().numpy().ravel() for p in (hid.weight, hid.bias, out.weight, out.bias)])
            assert np.array_equal(ref, r)
    r = P.rescaled_init(7)
    assert abs(np.abs(r[12:16]).sum() - P.S0) < 1e-15 and np.array_equal(r[:12], rows[2][:12])


def test_rho2_equals_v2_rho2_batch():
    import torch
    from src import simplicity_bias_v2 as v2
    th = P.branch_theta("M", P.S0)
    rng = np.random.default_rng(0)
    rows = np.array([th] + [th + 0.3 * rng.normal(size=17) for _ in range(20)])
    ref = v2.rho2_batch(rows, P.X, torch)
    fast = np.array([P.rho2(r) for r in rows])
    assert np.abs(ref - fast).max() < 1e-13


def test_scale_only_rule_and_idle_unit_exactly_zero():
    th = P.branch_theta("M", P.S0)
    active = [k for k in range(4) if th[12 + k] != 0]
    idle = [k for k in range(4) if k not in active]
    assert len(active) == 3 and len(idle) == 1                     # a three-unit test
    rho = 2.0 ** -9
    step = P.make_step(rho, active)
    new = step(th)
    _, g = P.loss_grad(th)
    a = np.zeros(4); a[active] = np.sign(th[12:16][active]); a /= np.linalg.norm(a)
    gv = g[12:16]
    exp_v = th[12:16] - ((np.eye(4) - np.outer(a, a)) @ gv + rho * a * (a @ gv))
    assert np.allclose(new[12:16], exp_v, atol=1e-15, rtol=0)
    assert np.array_equal(new[:12], th[:12] - g[:12]) and new[16] == th[16] - g[16]
    # the scale moves at rate ρ: ds = −ρ·√3·(a·∇_v L) to first order
    ds = P.scale(new) - P.scale(th)
    assert ds == pytest.approx(-rho * math.sqrt(3) * (a @ gv), rel=1e-6)
    x = th.copy()
    k = idle[0]
    for _ in range(200):
        x = step(x)
    assert x[2 * k] == 0.0 and x[2 * k + 1] == 0.0 and x[8 + k] == 0.0 and x[12 + k] == 0.0


def test_release_is_on_M_and_labels():
    th = P.branch_theta("M", P.S0)
    assert abs(P.scale(th) - P.S0) < 1e-12
    _, g = P.loss_grad(th)
    act = [k for k in range(4) if th[12 + k] != 0]
    assert np.abs(g[[i for k in act for i in (2 * k, 2 * k + 1, 8 + k)] + [16]]).max() < 1e-9
    bp = {k: P.branch_P(k, P.S0) for k in ("L0", "M", "S", "S2")}
    lab, d = P.label_at(P.to_landscape(th)[0], bp)
    assert lab == "M" and d["M"] < 1e-9
    lab, _ = P.label_at(P.to_landscape(th)[0] * 0 + 0.3, bp)
    assert lab == "other"


def test_plan_rule():
    labels = ["L0", "M", "other", "M", "M"]
    seeds = (10, 11, 12, 13, 14)
    pl, surplus = P.plan(labels, seeds, (-1.0, -2.0))
    assert pl == [(11, -1.0), (13, -2.0)] and surplus == [14]
    pl, surplus = P.plan(["M"] * 30, tuple(range(30)))
    assert [l2 for _, l2 in pl] == list(P.LADDER_LOG2) and surplus == [27, 28, 29]


# ================================================================================================ the forecaster
def _exp_path(t_c, s0=1.0, alpha=1e-3, extra=10):
    t = np.arange(t_c + extra, dtype=float)
    return s0 * np.exp(alpha * t)


def test_forecast_constructed_exponential_path():
    """log s exactly linear: the quad extrapolation is exact, so t̂_F, ṡ̂_F, ε̂_F, ŝ_c and t_fc follow by hand."""
    alpha, s_F, LamF = 1e-3, 2.0, 1.0
    t_c = 300
    s = _exp_path(t_c, alpha=alpha)
    rec, same = CFF.run_forecast(s, t_c, s_F, LamF, 1.0, 10 ** 6, 0.95)
    assert same and rec["status"] == "ok" and rec["max_index_read"] == t_c - 1
    tF = int(math.ceil(math.log(s_F) / alpha - 1e-9))
    assert abs(rec["t_F_fc"] - tF) <= 1
    t = rec["t_F_fc"]
    sdot = math.exp(alpha * t) - math.exp(alpha * (t - 1))
    eps = (sdot / s_F) / LamF
    assert rec["eps_fc"] == pytest.approx(eps, rel=1e-6)
    s_c = s_F * (1 + CFF.OMEGA0 * eps ** (2 / 3))
    assert rec["s_c_fc"] == pytest.approx(s_c, rel=1e-9)
    assert abs(rec["t_fc"] - math.ceil(math.log(s_c) / alpha)) <= 1
    assert rec["delay_fc"] == rec["t_fc"] - rec["t_F_fc"] > 0


def test_no_forecast_cases():
    t_c = 300
    flat = np.full(t_c + 5, 1.0)                                    # not growing at the cutoff
    rec, _ = CFF.run_forecast(flat, t_c, 2.0, 1e-3, 1.0, 10 ** 6, 0.95)
    assert rec["status"].startswith("no forecast") and not CFF.has_forecast(rec)
    t = np.arange(t_c + 5, dtype=float)                            # log s concave: peaks below s_F
    sat = np.exp(1e-3 * t - 2e-6 * t ** 2)
    rec, _ = CFF.run_forecast(sat, t_c, 2.0, 1e-3, 1.0, 10 ** 6, 0.95)
    assert rec["status"].startswith("no forecast") and rec["t_fc"] is None
    s = _exp_path(t_c, alpha=1e-3)                                  # ŝ_c beyond 1.6·s_F (huge ε: tiny Λ_F)
    rec, _ = CFF.run_forecast(s, t_c, 2.0, 1e-12, 1.0, 10 ** 6, 0.95)
    assert rec["status"] == "no forecast: no extrapolated s reaches s_c within 1.6·s_F"
    assert rec["t_F_fc"] is not None and rec["t_fc"] is None and not CFF.has_forecast(rec)


# ================================================================================================ ENFORCEMENT
def test_enforcement_leaky_forecaster_reading_the_cutoff_row_is_caught(monkeypatch):
    s = _exp_path(300)
    real = CFF.forecast_fold

    def leaky(s_view, t_c, *a, **k):
        _ = s_view[t_c]                                             # reads the cutoff row
        return real(s_view, t_c, *a, **k)
    monkeypatch.setattr(CFF, "forecast_fold", leaky)
    with pytest.raises(CF.CausalityViolation):
        CFF.run_forecast(s, 300, 2.0, 1e-3, 1.0, 10 ** 6, 0.95)

    def leaky_slice(s_view, t_c, *a, **k):
        _ = s_view[: t_c + 1]
        return real(s_view, t_c, *a, **k)
    monkeypatch.setattr(CFF, "forecast_fold", leaky_slice)
    with pytest.raises(CF.CausalityViolation):
        CFF.run_forecast(s, 300, 2.0, 1e-3, 1.0, 10 ** 6, 0.95)


def test_enforcement_guard_bypass_is_caught_by_the_nan_recomputation(monkeypatch):
    s = _exp_path(300)
    real = CFF.forecast_fold
    private = {"a": s}

    def bypass(s_view, t_c, *a, **k):
        out = real(s_view, t_c, *a, **k)
        out["peek"] = float(private["a"][t_c]) if isinstance(s_view, CF.GuardedArray) else float(np.asarray(s_view)[t_c])
        return out
    monkeypatch.setattr(CFF, "forecast_fold", bypass)
    rec, same = CFF.run_forecast(s, 300, 2.0, 1e-3, 1.0, 10 ** 6, 0.95)
    assert not same and rec["nan_recompute_identical"] is False


def test_enforcement_run_rate_aborts_on_a_differing_recomputation(monkeypatch):
    fr = _fr()
    monkeypatch.setattr(CFF, "run_forecast", lambda *a, **k: ({"status": "ok"}, False))
    with pytest.raises(CF.CausalityViolation):
        P.run_rate(-9.0, fr)


def test_real_forecaster_reads_only_rows_before_the_cutoff_and_nan_garbage_identical():
    fr = _fr()
    S, th, t_c = P.train_to_cutoff(np.array(fr["release_theta"]), 2.0 ** -10, S_F, 0.95)
    ext = np.concatenate([S, S[-1] * np.linspace(1, 1.3, 500)])
    g = CF.GuardedArray(ext, t_c, name="s")
    rec = CFF.forecast_fold(g, t_c, S_F, LAMF_EXPLORE, 1.0, P.budget(2.0 ** -10) - t_c + 1, 0.95)
    assert g.max_index_read == t_c - 1 and not g.violations
    for poison in (np.nan, 1e9, -3.0):
        e2 = ext.copy(); e2[t_c:] = poison
        rec2 = CFF.forecast_fold(e2, t_c, S_F, LAMF_EXPLORE, 1.0, P.budget(2.0 ** -10) - t_c + 1, 0.95)
        assert CFF.same_forecast(rec, rec2)


# ================================================================================================ a real fast run
def _fr(table=True):
    th = P.branch_theta("M", P.S0)
    fr = {"s_F": S_F, "Lambda_F": LAMF_EXPLORE, "release_theta": [float(x) for x in th], "s_star": S_STAR, "q": Q}
    if table:
        ss = np.linspace(3.5, 4.7, 13)
        fr["M_table"] = {"s": list(ss), "lam_min": [P.tangent_lmin(P.branch_theta("M", s)) for s in ss]}
    return fr


def test_real_run_matches_the_committed_exploration_at_2_to_minus_10():
    """p2a_explore_g, population run at 2⁻¹⁰ (stride-20 ρ₂ refined to the step): t_obs 26,660, t_F 17,797; at f = 0.95
    t_c 15,563, t̂_F 17,819, t_fc 27,025, ε̂_F 0.013893.  The registered code (ρ₂ every step) reproduces them."""
    fr = _fr()
    rr = P.run_rate(-10.0, fr)
    fc = rr["forecast"]
    assert rr["t_c"] == 15_563 and fc["t_F_fc"] == 17_819 and fc["t_fc"] == 27_025
    assert fc["eps_fc"] == pytest.approx(0.013893249980971245, rel=1e-12)
    assert rr["nan_recompute_identical"] and fc["max_index_read"] == rr["t_c"] - 1
    assert rr["chi_max_sstar_tc"] is not None and rr["chi_max_sstar_tc"] < 0.5
    ob, S = P.observe_rate(-10.0, fr, rr)
    assert ob["t_obs"] == 26_660 and ob["t_F"] == 17_797 and ob["state_tc_hash_ok"]
    assert ob["idle_zero_to_crossing"] and ob["signs_fixed_to_crossing"]
    assert ob["rho2_at_obs"] >= Q and ob["s_obs"] == pytest.approx(1.1675098032563402 * S_F, rel=1e-12)
    e1, e2 = P.pilot_errors(fc["t_fc"], fc["t_F_fc"], ob["t_obs"], ob["t_F"])
    assert e1 == pytest.approx(365 / 9206) and e2 == pytest.approx(343 / 9206)
    tight = P.tight_forecast(-10.0, fr, S)
    assert tight["f"] == pytest.approx(P.f_tight(2.0 ** -10)) and tight["nan_recompute_identical"]


def test_observe_detects_a_tampered_state():
    fr = _fr(table=False)
    S, th, t_c = P.train_to_cutoff(np.array(fr["release_theta"]), 2.0 ** -8, S_F, 0.95)
    bad = th.copy(); bad[0] = np.nextafter(bad[0], np.inf)
    rr = {"t_c": t_c, "state_tc": [float(x) for x in bad], "state_tc_sha256": P.state_sha(bad)}
    with pytest.raises(AssertionError):
        P.observe_rate(-8.0, fr, rr, n_budget=t_c + 10)


def test_cutoff_is_a_stopping_time_on_s():
    fr = _fr(table=False)
    S, th, t_c = P.train_to_cutoff(np.array(fr["release_theta"]), 2.0 ** -8, S_F, 0.95)
    assert S[t_c] >= 0.95 * S_F and (S[1:t_c] < 0.95 * S_F).all() and len(S) == t_c + 1
    assert CF.cutoff_step(S, 0.95 * S_F) == t_c
    S2, th2, t2 = P.train_to_cutoff(np.array(fr["release_theta"]), 2.0 ** -8, S_F, 0.95)
    assert t2 == t_c and np.array_equal(th, th2)                     # deterministic, bit for bit
    _, _, none = P.train_to_cutoff(np.array(fr["release_theta"]), 2.0 ** -8, S_F, 0.95, n_budget=10)
    assert none is None


# ================================================================================================ F
def _F(s_obs):
    return P.criterion_F(s_obs, S_F, S_STAR, 27)


def test_F_pass_fail_falsifier_unresolved():
    ok = [1.01 * S_F] * 27
    assert _F(ok)["verdict"] == "PASS"
    assert _F([1.01 * S_F] * 25 + [1.3 * S_F, None])["verdict"] == "PASS"          # 25/27 = 0.926
    r = _F([1.01 * S_F] * 24 + [1.3 * S_F, None, 0.99 * S_F])                      # 24/27 = 0.889
    assert r["verdict"] == "FAIL" and r["n_no_crossing"] == 1 and not r["falsified"]
    assert _F([S_F] * 26 + [1.25 * S_F])["verdict"] == "PASS"                       # both ends inclusive
    r = _F([1.01 * S_F] * 26 + [1.25 * S_STAR])                                     # at 1.25·s*: the falsifier
    assert r["verdict"] == "FAIL" and r["falsified"] and r["n_falsifier"] == 1
    assert _F([1.01 * S_F] * 26 + [1.2 * S_STAR])["falsified"]
    assert not _F([1.01 * S_F] * 26 + [1.2501 * S_STAR])["falsified"]
    assert P.criterion_F([], S_F, S_STAR, 0)["verdict"] == "UNRESOLVED"


# ================================================================================================ C1, C2 and the no-forecast rule
def test_C1_C2_within_rule():
    d = np.full(27, 1000.0)
    err = np.zeros(27); err[:5] = 500                               # 22 within at τ = 0.2
    has = np.ones(27, bool)
    assert P.criterion_within(err, d, 0.2, has, 27)["verdict"] == "PASS"          # 22/27 = 0.815
    err[5] = 500
    assert P.criterion_within(err, d, 0.2, has, 27)["verdict"] == "FAIL"          # 21/27 = 0.778
    err = np.full(27, 200.0)
    assert P.criterion_within(err, d, 0.2, has, 27)["verdict"] == "PASS"          # inclusive
    assert P.criterion_within(-err, d, 0.2, has, 27)["verdict"] == "PASS"         # absolute value
    assert P.criterion_within(err * 1.0001, d, 0.2, has, 27)["verdict"] == "FAIL"
    assert P.criterion_within([], [], 0.2, [], 0)["verdict"] == "UNRESOLVED"


def test_no_forecast_fails_C1_C2_C4_and_no_crossing_is_not_within():
    T = [_row(l2) for l2 in P.LADDER_LOG2]
    base = P.criteria(T, S_F, S_STAR, 0.2, 0.25, 27)
    assert all(base[k]["verdict"] == "PASS" for k in ("F", "C1", "C2", "C3", "C4", "E")), base
    T2 = [dict(r) for r in T]
    for r in T2[:6]:
        r.update(has_fc=False, t_fc=None, t_F_fc=None, delay_fc=None, eps_fc=None, r_fc=None)
    c = P.criteria(T2, S_F, S_STAR, 0.2, 0.25, 27)
    assert c["C1"]["verdict"] == c["C2"]["verdict"] == "FAIL" and c["C1"]["n_no_forecast"] == 6
    assert c["C4"]["verdict"] == "FAIL" and c["C4"]["n_no_forecast"] == 6
    T3 = [dict(r) for r in T]
    T3[0].update(has_fc=False, t_fc=None, t_F_fc=None, delay_fc=None, eps_fc=None, r_fc=None)
    assert P.criteria(T3, S_F, S_STAR, 0.2, 0.25, 27)["C4"]["verdict"] == "FAIL"    # one rate without a forecast
    T4 = [dict(r) for r in T]
    for r in T4[:6]:
        r.update(t_obs=None, s_obs=None)
    c = P.criteria(T4, S_F, S_STAR, 0.2, 0.25, 27)
    assert c["C1"]["verdict"] == "FAIL" and c["C1"]["n_err_undefined"] == 6 and c["F"]["verdict"] == "FAIL"


def _row(l2, ratio=None):
    """A synthetic rate with the fold law: ε ∝ ρ, r_fc = Ω₀ε^{2/3}, r_obs = ratio·r_fc (ratio rising with ρ),
    t_obs a little before t_fc."""
    rho = 2.0 ** l2
    eps = 3.0 * rho
    r_fc = CFF.OMEGA0 * eps ** (2 / 3)
    ratio = 1.1 + 0.02 * (l2 + 18) if ratio is None else ratio
    r_obs = ratio * r_fc
    tF = int(9.4 / rho); delay = int(0.35 * tF * eps ** (2 / 3)) + 50
    return {"rho": rho, "log2rho": l2, "has_fc": True, "t_F_fc": tF, "t_fc": tF + delay, "delay_fc": delay,
            "eps_fc": eps, "r_fc": r_fc, "t_obs": tF + delay - delay // 20, "t_F": tF - delay // 40,
            "s_obs": S_F * (1 + r_obs)}


# ================================================================================================ C3
def test_C3_pass_nonmonotone_fail_ratio_le_1_fail_unresolved():
    rho = 2.0 ** np.array(P.LADDER_LOG2)
    r_fc = np.full(27, 0.01)
    rising = 1.05 + 0.2 * np.log2(rho / rho.min()) / 5                 # rises with ρ, all > 1
    has = np.ones(27, bool)
    assert P.criterion_C3(rho, rising * r_fc, r_fc, has)["verdict"] == "PASS"
    shuffled = np.random.default_rng(1).permutation(rising)          # non-monotone: Spearman ≪ 0.8
    r = P.criterion_C3(rho, shuffled * r_fc, r_fc, has)
    assert r["verdict"] == "FAIL" and r["spearman"] < 0.8 and r["all_ratios_gt_1"]
    falling = rising[::-1]                                            # ratio falls with ρ: Spearman −1
    assert P.criterion_C3(rho, falling * r_fc, r_fc, has)["verdict"] == "FAIL"
    one = rising.copy(); one[-1] = 1.0                               # ratio exactly 1 at the slowest rate
    r = P.criterion_C3(rho, one * r_fc, r_fc, has)
    assert r["verdict"] == "FAIL" and not r["all_ratios_gt_1"]
    below = rising.copy(); below[-1] = 0.99                          # still perfectly monotone, but ≤ 1
    r = P.criterion_C3(rho, below * r_fc, r_fc, has)
    assert r["spearman"] == pytest.approx(1.0) and r["verdict"] == "FAIL"
    h2 = np.zeros(27, bool); h2[:2] = True                           # fewer than 3 ratios
    assert P.criterion_C3(rho, rising * r_fc, r_fc, h2)["verdict"] == "UNRESOLVED"
    h3 = has.copy(); h3[5] = False                                    # a rate without a forecast is excluded
    r = P.criterion_C3(rho, rising * r_fc, r_fc, h3)
    assert r["n"] == 26 and r["verdict"] == "PASS"


def test_spearman_average_ranks():
    assert P.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert P.spearman([1, 2, 3, 4], [1, 1, 2, 2]) == pytest.approx(np.corrcoef([1, 2, 3, 4], [1.5, 1.5, 3.5, 3.5])[0, 1])
    assert math.isnan(P.spearman([1, 2, 3], [5, 5, 5]))


# ================================================================================================ C4
def test_C4_bootstrap_pass_fail_unresolved():
    n = 27
    t_obs = np.full(n, 1000.0); has = np.ones(n, bool)
    t_fc = t_obs + 10; t_F = t_obs - 500                              # forecast much closer: D < 0
    r = P.criterion_C4(t_fc, t_F, t_obs, has)
    assert r["verdict"] == "PASS" and r["ci95"][1] < 0 and r["boot_seed"] == P.BOOT_SEED
    r = P.criterion_C4(t_obs + 600, t_obs - 10, t_obs, has)            # interval entirely above 0
    assert r["verdict"] == "FAIL" and r["ci95"][0] > 0
    D = np.r_[np.full(14, -100.0), np.full(13, 100.0)]                 # interval containing 0
    r = P.criterion_C4(t_obs + 200 + D, t_obs - 200, t_obs, has)
    assert r["ci95"][0] < 0 < r["ci95"][1] and r["verdict"] == "FAIL"
    h = np.zeros(n, bool); h[0] = True                                 # one rate with a forecast, others none
    assert P.criterion_C4(t_fc, t_F, t_obs, h)["verdict"] == "FAIL"   # the no-forecast rule wins
    t1 = t_obs.copy(); t1[1:] = np.nan                                # forecasts everywhere, one crossing
    assert P.criterion_C4(t_fc, t_F, t1, has)["verdict"] == "UNRESOLVED"
    a = P.criterion_C4(t_fc, t_F, t_obs, has); b = P.criterion_C4(t_fc, t_F, t_obs, has)
    assert a["ci95"] == b["ci95"]                                     # local generator: reproducible


# ================================================================================================ E
def test_E_band_pass_fail_unresolved():
    eps = 2.0 ** np.linspace(-13, -18, 27) * 15
    has = np.ones(27, bool)
    r = P.criterion_E(eps, 1.8 * eps ** (2 / 3), has)
    assert r["verdict"] == "PASS" and r["slope"] == pytest.approx(2 / 3) and r["ci95"][0] == pytest.approx(2 / 3)
    assert P.criterion_E(eps, 5 * eps, has)["verdict"] == "FAIL"                  # a linear lag: slope 1
    assert P.criterion_E(eps, np.full(27, 0.02), has)["verdict"] == "FAIL"        # no rate dependence: slope 0
    noisy = 1.8 * eps ** 0.7 * np.exp(np.random.default_rng(3).normal(0, 0.4, 27))
    r = P.criterion_E(eps, noisy, has)                                              # slope inside, CI too wide
    assert r["ci95"][0] < 0.55 or r["ci95"][1] > 0.80
    assert r["verdict"] == "FAIL"
    assert P.criterion_E(eps, 1.8 * eps ** 0.551, has)["verdict"] == "PASS"       # just inside the band
    assert P.criterion_E(eps, 1.8 * eps ** 0.549, has)["verdict"] == "FAIL"       # just below it
    assert P.criterion_E(eps, 1.8 * eps ** 0.799, has)["verdict"] == "PASS"
    assert P.criterion_E(eps, 1.8 * eps ** 0.801, has)["verdict"] == "FAIL"
    h = np.zeros(27, bool); h[:2] = True
    assert P.criterion_E(eps, 1.8 * eps ** (2 / 3), h)["verdict"] == "UNRESOLVED"
    r_obs = 1.8 * eps ** (2 / 3); r_obs[:3] = -0.01                                # crossings below s_F excluded
    assert P.criterion_E(eps, r_obs, has)["n"] == 24


# ================================================================================================ gate, validity, outcome
def test_gate():
    assert P.gate(["M"] * 27 + ["L0"] * 93)["pass"]
    g = P.gate(["M"] * 26 + ["other"] * 94)
    assert not g["pass"] and g["n_M"] == 26 and g["counts"] == {"M": 26, "other": 94}


def test_validity_rules():
    n = 27
    ones = np.ones(n, bool)
    t_c = np.full(n, 100.0); t_obs = np.full(n, 200.0)
    assert P.validity(ones, t_c, t_obs, ones, ones, n)["ok"]
    f = ones.copy(); f[:2] = False                                    # follow 25/27 ≥ 0.9
    assert P.validity(f, t_c, t_obs, ones, ones, n)["follow_ok"]
    f[2] = False                                                      # 24/27 < 0.9
    v = P.validity(f, t_c, t_obs, ones, ones, n)
    assert not v["follow_ok"] and not v["ok"]
    tc = t_c.copy(); tc[:2] = 200                                    # t_c == t_obs is not before (strict)
    assert P.validity(ones, tc, t_obs, ones, ones, n)["cutoff_before_ok"]
    tc[2] = 300
    v = P.validity(ones, tc, t_obs, ones, ones, n)
    assert not v["cutoff_before_ok"] and v["n_cutoff_not_before"] == 3 and not v["ok"]
    to = t_obs.copy(); to[:5] = np.nan                                # a cutoff and no crossing counts as before
    assert P.validity(ones, t_c, to, ones, ones, n)["cutoff_before_ok"]
    tc = t_c.copy(); tc[:3] = np.nan                                  # no cutoff: not before
    assert not P.validity(ones, tc, t_obs, ones, ones, n)["cutoff_before_ok"]
    idle = ones.copy(); idle[7] = False                               # every rate: idle unit exactly 0
    v = P.validity(ones, t_c, t_obs, idle, ones, n)
    assert not v["idle_zero_all"] and not v["ok"]
    sg = ones.copy(); sg[0] = False                                   # every rate: signs fixed
    assert not P.validity(ones, t_c, t_obs, ones, sg, n)["ok"]


def test_outcome_rule():
    allp = {k: "PASS" for k in ("F", "C1", "C2", "C3", "C4", "E")}
    assert P.outcome(allp) == "PASS"
    assert P.outcome(allp, gate_ok=False) == "UNRESOLVED (gate)"
    assert P.outcome(allp, valid_ok=False) == "UNRESOLVED (validity)"
    assert P.outcome({**allp, "E": "UNRESOLVED", "C1": "FAIL"}) == "UNRESOLVED"
    assert P.outcome({**allp, "C1": "FAIL", "C4": "FAIL"}) == "FAIL C1+C4"


# ================================================================================================ pilot: STOP rule and τ
def test_tau_rule():
    assert P.tau_rule([0.10, 0.05, 0.003]) == 0.15                   # 1.5 × 0.10 = 0.15 exactly
    assert P.tau_rule([0.10196, 0.055, 0.003]) == 0.2
    assert P.tau_rule([0.1386, 0.1146, 0.0922]) == 0.25
    assert P.tau_rule([0.01, 0.0, 0.0]) == 0.05
    assert P.tau_rule([0.1, None, 0.1]) is None and P.tau_rule([]) is None


def test_pilot_never_runs_a_ladder_rate():
    assert all(P.pilot_rate_allowed(l2) for l2 in P.PILOT_LOG2)
    assert not P.pilot_rate_allowed(-16.5) and not P.pilot_rate_allowed(-15.0)


def test_pilot_errors():
    assert P.pilot_errors(27_025, 17_819, 26_660, 17_797) == pytest.approx((365 / 9206, 343 / 9206))
    assert P.pilot_errors(None, 1, 2, 3) == (None, None) and P.pilot_errors(10, 5, None, 3) == (None, None)


@pytest.mark.parametrize("eps,chi,nM,t1,t2,stop", [
    (0.0004, 0.004, 3, 0.2, 0.25, False),
    (0.01, 0.1, 3, 0.2, 0.25, False),                                 # at the thresholds: no STOP
    (0.0101, 0.004, 3, 0.2, 0.25, True),
    (0.0004, 0.1001, 3, 0.2, 0.25, True),
    (None, 0.004, 3, 0.2, 0.25, True),
    (0.0004, 0.004, 2, 0.2, 0.25, True),
    (0.0004, 0.004, 3, None, 0.25, True),
])
def test_pilot_stop_rule(eps, chi, nM, t1, t2, stop):
    assert P.pilot_stop(eps, chi, nM, t1, t2)["stop"] is stop


def test_horizon_descriptive():
    assert P.horizon(100, 400, 300) == {"steps": 300, "delays": 3.0}
    assert P.horizon(100, None, 300) == {"steps": None, "delays": None}


def test_chi_path_constructed():
    S = np.exp(1e-4 * np.arange(500)) * 3.6
    table = {"s": [3.0, 5.0], "lam_min": [2e-4, 2e-4]}
    chi = P.chi_path(S, 200, 300, table)
    assert len(chi) == 100 and np.allclose(chi, 1e-4 / 2e-4)


# ================================================================================================ frozen inputs (once frozen)
FROZEN = ROOT / "results" / "phase2a" / "frozen.json"


@pytest.mark.skipif(not FROZEN.exists(), reason="not yet frozen")
def test_frozen_inputs():
    fr = json.loads(FROZEN.read_text())
    from src import sb_fold as SBF
    assert fr["s_F"] == SBF.fold_of("M") == pytest.approx(S_F, abs=1e-12)
    assert fr["q"] == Q and fr["s_star"] == S_STAR
    assert fr["Lambda_F"] == pytest.approx(LAMF_EXPLORE, rel=1e-6)
    assert fr["Lambda_F"] == pytest.approx(math.sqrt(fr["abs_mc"] * fr["s_F"]), rel=1e-14)
    rel = np.array(fr["release_theta"])
    assert np.array_equal(rel, P.branch_theta("M", P.S0)) and fr["release_sha256"] == P.state_sha(rel)
    assert fr["release_active_units"] == [k for k in range(4) if rel[12 + k] != 0] and len(fr["release_active_units"]) == 3
    assert fr["ladder_log2"] == list(P.LADDER_LOG2) and fr["pilot_log2"] == list(P.PILOT_LOG2)
    ts = np.array(fr["M_table"]["s"])
    assert ts[0] <= P.S0 and ts[-1] < fr["s_F"] and (np.array(fr["M_table"]["lam_min"]) > 0).all()
    assert fr["branch_points_s0"]["M"] is not None


# ================================================================================================ the 22 scored rates
def _table27(**over):
    """All 27 ladder rates, each passing every rule (the synthetic fold law of _row), with validity fields."""
    T = []
    for l2 in P.LADDER_LOG2:
        r = _row(l2)
        r.update(t_c=r["t_F_fc"] // 2, follow_in_M=True, idle_ok=True, sign_ok=True)
        T.append(r)
    return T


def _break(r):
    """Make a rate fail F, C1, C2, C3 (ratio < 1), validity and give it no forecast."""
    r.update(s_obs=1.2 * S_STAR, t_obs=r["t_c"] - 5, has_fc=False, t_fc=None, t_F_fc=None, delay_fc=None,
             eps_fc=None, r_fc=None, follow_in_M=False, idle_ok=False, sign_ok=False)


def test_split_scored_and_gate_with_all_27_run():
    T = _table27()
    sc, pre = P.split_scored(T)
    assert len(sc) == 22 and len(pre) == 5 and {r["log2rho"] for r in pre} == set(P.PREOBSERVED_LOG2)
    assert P.gate(["M"] * 27 + ["L0"] * 93)["pass"] and not P.gate(["M"] * 26 + ["L0"] * 94)["pass"]
    pl, _ = P.plan(["M"] * 27, tuple(range(27)))
    assert [l2 for _, l2 in pl] == list(P.LADDER_LOG2)                 # every one of the 27 rates is run
    ST = P.score_tables(T, S_F, S_STAR, 0.2, 0.25)
    assert ST["outcome"] == "PASS" and ST["n_scored_rows"] == 22 and ST["n_preobserved_rows"] == 5
    for k in ("F", "C1", "C2"):
        assert ST["criteria"][k]["n"] == 22
    assert ST["criteria"]["C3"]["n"] == 22 and ST["criteria"]["C4"]["n"] == 22 and ST["criteria"]["E"]["n"] == 22
    assert ST["validity"]["n"] == 22 and ST["DESCRIPTIVE_preobserved"]["criteria"]["F"]["n"] == 5
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25, gate_ok=False)["outcome"] == "UNRESOLVED (gate)"


def test_preobserved_rates_never_enter_a_verdict():
    T = _table27()
    for r in T:
        if r["log2rho"] in P.PREOBSERVED_LOG2:
            _break(r)                                                     # even the falsifier at a pre-observed rate
    ST = P.score_tables(T, S_F, S_STAR, 0.2, 0.25)
    assert ST["outcome"] == "PASS" and not ST["criteria"]["F"]["falsified"]
    d = ST["DESCRIPTIVE_preobserved"]
    assert d["criteria"]["F"]["falsified"] and d["criteria"]["F"]["verdict"] == "FAIL" and not d["validity"]["ok"]


def _scored_idx(T):
    return [i for i, r in enumerate(T) if r["log2rho"] not in P.PREOBSERVED_LOG2]


@pytest.mark.parametrize("k_bad,verdict", [(2, "PASS"), (3, "FAIL")])     # 20/22 = 0.909 ≥ 0.9; 19/22 = 0.864
def test_F_over_22(k_bad, verdict):
    T = _table27()
    for i in _scored_idx(T)[:k_bad]:
        T[i].update(s_obs=1.3 * S_F)                                      # outside the window, not a falsifier
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]["F"]["verdict"] == verdict
    T[_scored_idx(T)[-1]].update(s_obs=1.25 * S_STAR)                     # one scored falsifier: FAIL
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]["F"]["falsified"]


@pytest.mark.parametrize("k_bad,verdict", [(4, "PASS"), (5, "FAIL")])     # 18/22 = 0.818 ≥ 0.8; 17/22 = 0.773
def test_C1_C2_over_22(k_bad, verdict):
    for crit, field in (("C1", "t_fc"), ("C2", "t_F")):
        T = _table27()
        for i in _scored_idx(T)[:k_bad]:
            T[i][field] = T[i][field] + (T[i]["delay_fc"] if field == "t_fc" else -T[i]["delay_fc"])
        ST = P.score_tables(T, S_F, S_STAR, 0.2, 0.25)
        assert ST["criteria"][crit]["n"] == 22 and ST["criteria"][crit]["verdict"] == verdict, crit


def test_no_forecast_at_a_scored_rate_fails_C1_C2_C4_but_not_at_a_preobserved_one():
    T = _table27()
    i = _scored_idx(T)[0]
    T[i].update(has_fc=False, t_fc=None, t_F_fc=None, delay_fc=None, eps_fc=None, r_fc=None)
    c = P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]
    assert c["C4"]["verdict"] == "FAIL" and c["C1"]["n_no_forecast"] == 1 and c["C2"]["n_no_forecast"] == 1
    assert c["C3"]["n"] == 21 and c["E"]["n"] == 21
    T = _table27()
    T[0].update(has_fc=False, t_fc=None, t_F_fc=None, delay_fc=None, eps_fc=None, r_fc=None)    # 2^-13: pre-observed
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]["C4"]["verdict"] == "PASS"


def test_C3_over_scored_finite_ratios():
    T = _table27()
    sc = _scored_idx(T)
    for i in sc[:19]:                                                     # only 3 scored rates keep a forecast
        T[i].update(has_fc=False, t_fc=None, t_F_fc=None, delay_fc=None, eps_fc=None, r_fc=None)
    c = P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]["C3"]
    assert c["n"] == 3 and c["verdict"] == "PASS"
    T[sc[19]].update(has_fc=False, t_fc=None, t_F_fc=None, delay_fc=None, eps_fc=None, r_fc=None)
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]["C3"]["verdict"] == "UNRESOLVED"
    T = _table27()
    j = sc[-1]
    T[j]["s_obs"] = S_F * (1 + 0.99 * T[j]["r_fc"])                       # a scored ratio below 1: FAIL
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]["C3"]["verdict"] == "FAIL"


def test_C4_and_E_bootstraps_over_the_22():
    T = _table27()
    for r in T:
        if r["log2rho"] in P.PREOBSERVED_LOG2:                            # huge D at pre-observed rates: ignored
            r["t_fc"] = r["t_obs"] + 10 ** 7
            r["s_obs"] = S_F * (1 + 50 * r["r_fc"])
    c = P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]
    assert c["C4"]["n"] == 22 and c["C4"]["verdict"] == "PASS" and c["C4"]["ci95"][1] < 0
    assert c["E"]["n"] == 22 and c["E"]["verdict"] == "PASS"
    sc = _scored_idx(T)
    for i in sc:                                                          # now the scored D are positive: FAIL
        T[i]["t_F_fc"] = T[i]["t_obs"]
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["criteria"]["C4"]["verdict"] == "FAIL"


@pytest.mark.parametrize("k_bad,ok", [(2, True), (3, False)])             # 20/22 ≥ 0.9; 19/22 < 0.9
def test_validity_90pct_over_22(k_bad, ok):
    for field in ("follow", "cutoff"):
        T = _table27()
        for i in _scored_idx(T)[:k_bad]:
            if field == "follow":
                T[i]["follow_in_M"] = False
            else:
                T[i]["t_c"] = T[i]["t_obs"]                               # not strictly before
        v = P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["validity"]
        assert v["n"] == 22 and v[f"{field}_ok" if field == "follow" else "cutoff_before_ok"] is ok
        assert (P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["outcome"] == "PASS") is ok


def test_idle_and_signs_every_scored_rate():
    T = _table27()
    T[_scored_idx(T)[3]]["idle_ok"] = False
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["outcome"] == "UNRESOLVED (validity)"
    T = _table27()
    T[_scored_idx(T)[3]]["sign_ok"] = False
    assert P.score_tables(T, S_F, S_STAR, 0.2, 0.25)["outcome"] == "UNRESOLVED (validity)"


PILOT = ROOT / "results" / "phase2a" / "pilot.json"


@pytest.mark.skipif(not PILOT.exists(), reason="pilot not yet run")
def test_pilot_sets_tau_by_the_rule_and_did_not_stop():
    Pj = json.loads(PILOT.read_text())
    assert [a["log2rho"] for a in Pj["assignment"]] == list(P.PILOT_LOG2)
    assert all(a["seed"] in P.PILOT_SEEDS for a in Pj["assignment"])
    assert sorted(float(k) for k in Pj["runs"]) == sorted(P.PILOT_LOG2)
    assert Pj["tau1"] == P.tau_rule(Pj["errors_C1"]) == 0.15 and Pj["tau2"] == P.tau_rule(Pj["errors_C2"]) == 0.25
    assert not Pj["stop_rule"]["stop"]
    first = Pj["runs"]["-15.0625"]
    assert first["forecast"]["eps_fc"] <= P.PILOT_EPS_MAX and first["chi_max_sstar_tc"] <= P.PILOT_CHI_MAX
