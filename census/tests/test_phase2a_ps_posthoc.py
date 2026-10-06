"""Phase 2A-PS POST HOC (src/phase2a_ps_posthoc.py): constructed cases for every new function — the forecaster's
extrapolation unpacked (quad_profile agrees with the registered forecaster on constructed paths), the quasi-static ρ·t
on M, the stall scale, the PIPELINE / LANDSCAPE failure reading, the dead-unit test, the step decomposition (equal to
the registered make_step), the block and wider-search class rules, the branch match, the first ρ₂ ≥ q scale and the
rescoring with seeds excluded (the registered score_tables on the remaining rows)."""

import math

import numpy as np
import pytest

from src import causal_forecast_fold as CFF
from src import phase2a_ps as PS
from src import phase2a_ps_posthoc as H

from tests.test_phase2a_ps import _table


# ------------------------------------------------------------------------------------------ quad_profile
@pytest.mark.parametrize("a2,budget,reach", [(2e-11, 100_000, True), (2e-11, 500, False), (0.0, 100_000, True),
                                             (-2e-10, 100_000, False)])
def test_quad_profile_agrees_with_the_registered_forecaster(a2, budget, reach):
    t = np.arange(20_000.0)
    s = np.exp(np.log(5.0) + 1e-5 * t + a2 * t ** 2)
    s_F = float(s[-1] * 1.03)
    q = H.quad_profile(s, s_F, budget)
    rec = CFF.forecast_fold(s, len(s), s_F, 1e-3, 1.0, budget, PS.F_CUT)
    assert q["k_reach_sF_within_used_path"] is reach
    assert (rec["t_F_fc"] is not None) is reach
    if reach:
        assert rec["t_F_fc"] - len(s) + 1 == q["k_reach_sF_uncapped"]   # ŝ index k ↔ step t_c + k (k from 0)
    assert q["growth_rate_at_cutoff"] == pytest.approx(rec["growth_rate_at_cutoff"], rel=1e-9)
    assert q["n_window"] == rec["n_window"]


def test_quad_profile_concave_vertex_below_sF_never_reaches():
    t = np.arange(10_000.0)
    s = np.exp(np.log(5.0) + 1e-5 * t - 4e-10 * t ** 2)       # vertex at t = 12,500, log-gain 0.0625 after t_c
    q = H.quad_profile(s, 5.0 * 1.2, 10 ** 7)
    assert q["concave"] and q["k_reach_sF_uncapped"] is None
    assert q["k_vertex"] == pytest.approx(12_500 - 9_999, rel=1e-3)
    assert q["s_hat_max_uncapped"] < 6.0


def test_quad_profile_budget_cap_flag():
    t = np.arange(20_000.0)
    s = np.exp(np.log(5.0) + 1e-5 * t)
    q = H.quad_profile(s, float(s[-1] * 1.03), 1_000)
    assert q["budget_capped"] and q["n_ahead_used"] == 1_000
    assert q["k_reach_sF_uncapped"] > 1_000 and not q["k_reach_sF_within_used_path"]


# ------------------------------------------------------------------------------------------ M's slope
def test_quasi_static_rho_time_constant_slope_and_stall():
    sg = np.linspace(4.0, 5.0, 11)
    assert H.quasi_static_rho_time(sg, np.full(11, -0.01), 4.0, 5.0) == pytest.approx(1.0 / (3 * 0.01))
    assert H.quasi_static_rho_time(sg, np.full(11, -0.01), 4.25, 4.75, n_active=2) == pytest.approx(0.5 / 0.02)
    d = np.full(11, -0.01); d[8] = 0.0
    assert H.quasi_static_rho_time(sg, d, 4.0, 5.0) == math.inf
    assert H.quasi_static_rho_time(sg, d, 4.0, 4.5) < math.inf
    assert H.quasi_static_rho_time(sg, d, 5.0, 4.0) == 0.0


def test_stall_scale():
    sg = np.array([1.0, 2.0, 3.0, 4.0])
    assert H.stall_scale(sg, np.array([-2.0, -1.0, 1.0, 2.0])) == pytest.approx(2.5)
    assert H.stall_scale(sg, np.array([-2.0, -1.0, -0.5, -0.1])) is None
    assert H.stall_scale(sg, np.array([0.0, -1.0, -1.0, -1.0])) == 1.0


def test_quasi_static_matches_the_scale_rule_on_a_constructed_state():
    """ds per step = −ρ·Σ sign(v_k)∂L/∂v_k when ∇_v L is parallel to â (on M): the identity behind quasi_static."""
    v = np.array([1.0, -0.5, 0.7, 0.0])
    mu = -0.03
    g_v = mu * np.sign(v)
    d = H.step_decomposition(g_v, v, [0, 1, 2], 2.0 ** -14)
    assert d["ds"] == pytest.approx(-(2.0 ** -14) * 3 * mu, rel=1e-12)
    assert d["dv_orthogonal"] == pytest.approx([0, 0, 0, 0], abs=1e-15)


# ------------------------------------------------------------------------------------------ failure reading
def test_classify_failure_every_branch():
    nf = "no forecast: no extrapolated s reaches s_F within 1.6·s_F"
    assert H.classify_failure(nf, True, 10.0, "s_F", None, 100)[0] == "PIPELINE"
    lab, why = H.classify_failure(nf, False, math.inf, "s_F", None, 100)
    assert lab == "LANDSCAPE" and why.startswith("stall")
    lab, why = H.classify_failure(nf, False, 80.0, "s_F", None, 100)
    assert lab == "LANDSCAPE" and why.startswith("slow growth")
    lab, why = H.classify_failure(nf, False, 50.0, "s_F", 500, 100)
    assert lab == "PIPELINE" and "budget cap" in why
    lab, why = H.classify_failure(nf, False, 50.0, "s_F", None, 100)
    assert lab == "PIPELINE" and "turns over" in why
    lab, why = H.classify_failure("no forecast: no cutoff within the budget", False, 50.0, "0.95·s_F", None, 0)
    assert lab == "PIPELINE" and why.startswith("budget")
    assert H.classify_failure("no forecast: no cutoff within the budget", False, None, "0.95·s_F", None, 0)[0] == \
        "LANDSCAPE"


# ------------------------------------------------------------------------------------------ the sign seed
def test_effective_dead_units():
    th = np.zeros(17)
    th[12:16] = [1.5, 1.4, 1.2e-93, 0.0]
    assert H.effective_dead_units(th, [0, 1, 2]) == [2]
    th[14] = 1e-6
    assert H.effective_dead_units(th, [0, 1, 2]) == []


def test_step_decomposition_equals_the_registered_step_and_shows_the_leak():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (40, 2)); Y = (X[:, 1] > 0).astype(float)
    th = rng.normal(0, 1, 17); th[[6, 7, 11, 15]] = 0.0
    th[14] = 1e-90; th[[4, 5, 10]] = 1e-80                    # unit 2 numerically dead, counted active (v ≠ 0.0)
    act = [0, 1, 2]
    _, g = PS.loss_grad(th, X, Y)
    for rho in (2.0 ** -14, 2.0 ** -16):
        d = H.step_decomposition(g[12:16], th[12:16], act, rho)
        new = PS.make_step(rho, act, X, Y)(th)
        assert np.allclose(new[12:16], d["v_new"], rtol=0, atol=1e-15)
        assert 2 in d["sign_flips"] or abs(d["v_new"][2]) > 1e3 * abs(th[14])
        assert "ds_live_units_only" in d
    # on M with a dead unit (the 2,975,052 configuration): ∇_v L = μ·sign(v) on the live units, 0 on the dead one
    v = np.array([1.5, 1.4, 1e-93, 0.0]); g_v = np.array([-0.077, -0.077, 0.0, 0.0])
    d1 = H.step_decomposition(g_v, v, act, 2.0 ** -14)
    d2 = H.step_decomposition(g_v, v, act, 2.0 ** -16)
    assert d1["ds"] == pytest.approx(d2["ds"], rel=1e-3)       # the leaked part does not scale with ρ
    assert d1["ds"] > 1e3 * d1["ds_live_units_only"]
    assert d1["ds_live_units_only"] == pytest.approx(2.0 ** -14 * 2 * 0.077, rel=1e-12)
    assert d1["sign_flips"] == [2] and d1["v_new"][2] == pytest.approx(-(2 / 3) * 0.077 * (1 - 2.0 ** -14))


# ------------------------------------------------------------------------------------------ classes
def test_block_class():
    assert H.block_class(27) == "clean" and H.block_class(26) == "mixed" and H.block_class(0) == "none"
    assert H.block_class(81, 90) == "clean" and H.block_class(80, 90) == "mixed" and H.block_class(1, 90) == "mixed"


def test_wider_class():
    b = [{"scale": f, "amp": a, "n": 30, "n_ge_q": 0} for f in H.WIDE_SCALES for a in H.WIDE_AMPS]
    w = H.wider_class(b)
    assert w["pooled"] == "none" and w["first_scale_with_ge_q"] is None and w["n"] == 360
    for x in b:
        if x["scale"] == 1.10:
            x["n_ge_q"] = 30
    w = H.wider_class(b)
    assert w["pooled"] == "mixed" and w["per_scale"]["1.1"] == "clean" and w["per_scale"]["1.01"] == "none"
    assert w["first_scale_with_ge_q"] == "1.1"
    for x in b:
        x["n_ge_q"] = 28
    assert H.wider_class(b)["pooled"] == "clean"


def test_match_branch():
    assert H.match_branch({"S": 2e-4, "S2": 5e-4, "L0": None}) == "S"
    assert H.match_branch({"S": 2e-3, "S2": 5e-4, "L0": None}) == "S2"
    assert H.match_branch({"S": 2e-3, "S2": None, "L0": math.inf}) == "other"


def test_first_crossing_scale():
    assert H.first_crossing_scale([1.0, 1.1, 1.2], [0.3, 0.38, 0.40], q=0.39) == pytest.approx(1.15)
    assert H.first_crossing_scale([1.0, 1.1], [0.3, 0.38], q=0.39) is None
    assert H.first_crossing_scale([1.0, 1.1], [0.5, 0.6], q=0.39) == 1.0
    assert H.first_crossing_scale([1.0, 1.1, 1.2], [np.nan, 0.5, 0.6], q=0.39) == 1.1


# ------------------------------------------------------------------------------------------ rescoring
def test_score_excluding_removes_the_seed_and_uses_the_registered_rules():
    T = _table()
    for r in T:
        if r["class"] == "clean" and r["seed"] == 3:
            r["sign_ok"] = False                                  # one clean seed breaks validity at both rates
    assert PS.score_tables(T)["outcome"] == "UNRESOLVED (validity)"
    S = H.score_excluding(T, [3])
    assert S["validity"]["ok"] and S["outcome"] == "PASS" and S["n_clean_scored"] == 39
    assert S == PS.score_tables([r for r in T if r["seed"] != 3])
    assert H.score_excluding(T, [])["outcome"] == "UNRESOLVED (validity)"
