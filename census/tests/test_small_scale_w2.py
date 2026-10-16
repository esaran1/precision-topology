"""§22 (Track K follow-up): the ramp limit, the mirror fold, the fast-oscillation bound, and the ball-arithmetic and
exact-rational constants of Theorems K11 (symmetric windows, width 2) and K12 (asymmetric windows), on constructed pass
and fail cases, and the committed numbers."""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import arb

from src import small_scale_k as sk
from src import small_scale_w2 as w2

ROOT = Path(__file__).resolve().parents[1]
A = 1.3
f = lambda t, a=A: t + a * np.sin(t)


def _json():
    return json.loads((ROOT / "results" / "small_scale_w2.json").read_text())


def test_mirror_fold_identity():
    """Units (alpha, beta) and (-alpha, pi - beta) have the same sine part: weights p, q (p + q = 1) give a full-amplitude
    sine plus any slope in [-|alpha|, |alpha|].  A wrong mirror (pi + beta) does not."""
    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2.4, 2001)
    for _ in range(20):
        al, be, p, sg = rng.uniform(-5, 5), rng.uniform(0, 2 * np.pi), rng.uniform(0, 1), rng.choice([-1, 1])
        phi = sg * p * f(al * x + be) + sg * (1 - p) * f(-al * x + np.pi - be)
        target = sg * (2 * p - 1) * al * x + sg * A * np.sin(al * x + be)
        d = phi - target
        assert np.ptp(d) < 1e-12                                  # equal up to a constant
        wrong = sg * p * f(al * x + be) + sg * (1 - p) * f(-al * x + np.pi + be)
        if abs(al) > 0.5 and 0.2 < p < 0.8:
            assert np.ptp(wrong - target) > 1e-3


def test_ramp_limit_is_uniform_with_error_a_eps():
    """eps f_a(alpha x + beta) with eps alpha = gamma: minus (gamma x + const) it is a eps sin(.), so the sup error is
    a*eps (attained up to the grid) and tends to 0; with eps fixed and alpha -> infinity it does not."""
    x = np.linspace(-2, 2.4, 200001)
    gamma, beta = 0.7, 1.1
    for eps in (1e-1, 1e-2, 1e-3):
        al = gamma / eps
        d = eps * f(al * x + beta) - gamma * x - eps * beta
        assert np.max(np.abs(d)) <= A * eps * (1 + 1e-12)
        assert np.max(np.abs(d)) >= 0.99 * A * eps
    eps = 0.1
    assert np.max(np.abs(eps * f(1e4 * x + beta) - gamma * x)) > 100


def test_fast_oscillation_bound_and_a_fail_case():
    """|int_J G sin(alpha x + beta) dx| <= (2 sup|G| + TV(G))/|alpha|.  For G = sigma(kx + b) the bound holds; a G of large
    variation (a square wave locked to the sine) violates the bound without its TV term."""
    gx, gw = np.polynomial.legendre.leggauss(4000)
    lo, hi = 1.2, 2.4
    xx = (hi - lo) / 2 * gx + (hi + lo) / 2; ww = gw * (hi - lo) / 2
    for k, b in ((0.3, -0.1), (3.0, 0.5), (-2.0, 1.0)):
        G = 1 / (1 + np.exp(-(k * xx + b)))
        tv = abs(G[-1] - G[0])                                    # monotone
        for al in (5.0, 37.0, 400.0):
            I = abs(np.sum(G * np.sin(al * xx + 0.3) * ww))
            assert I <= (2 + tv) / al + 1e-12
    xs = np.linspace(lo, hi, 2_000_001)
    al = 400.0
    G = 0.5 + 0.5 * np.sign(np.sin(al * xs + 0.3))                # |G| <= 1, TV ~ 2 * 1.2 * 400 / pi
    I = abs(np.trapezoid(G * np.sin(al * xs + 0.3), xs))
    assert I > 2 / al                                              # without TV the bound fails
    tv = np.sum(np.abs(np.diff(G)))
    assert I <= (2 + tv) / al


def test_lower_localisation_pass_and_fail():
    D_lo = sk.D_lower_at(sk.ALPHA0)
    res = w2.lower_localisation(D_lo - arb("0.05"))
    assert res["ok"]
    assert 1.58 < res["alpha_L"] < 1.60 and res["kept_span_hi"] > float(sk.ALPHA0)
    bad = w2.lower_localisation(arb("1.58"))                     # above max |D_c| = 1.5732...: nothing kept
    assert not bad["ok"] and bad["kept_cells"] == 0
    tiny = w2.lower_localisation(arb("0.4"))                     # (32/15)/4 = 8/15 > 0.4: small-alpha clearance fails
    assert not tiny["ok"]


def _pc_inputs():
    D_lo = sk.D_lower_at(sk.ALPHA0)
    Vc = w2.var_cos_upper_on_box()
    thr = D_lo - w2._a(w2.ETA_BAR)
    aL = w2.lower_localisation(thr)["alpha_L_exact"]
    return D_lo, Vc, thr, aL


def test_pc_constants_pass_at_s2_fail_above():
    D_lo, Vc, thr, aL = _pc_inputs()
    J = {r["a"]: r for r in _json()["continuous_symmetric"]["rows"]}
    for a in (1.30, 1.60):
        s2 = Fraction(J[a]["s2_exact"])
        ok, det = w2.pc_constants(a, s2, D_lo, Vc, aL, thr)
        assert ok and abs(det["W2"] - J[a]["W2"]) < 1e-9
        ok_small, det_small = w2.pc_constants(a, s2 / 10, D_lo, Vc, aL, thr)
        assert ok_small and det_small["W1"] <= det["W1"]          # monotone: valid on all of (0, s2]
        ok_big, det_big = w2.pc_constants(a, 3 * s2, D_lo, Vc, aL, thr)
        assert not ok_big and not det_big["conds"][3]             # Case 2's quartic remainder is what fails
    ok_bad, _ = w2.pc_constants(1.30, Fraction(1, 1000), D_lo, Vc, Fraction(1, 10), thr)   # alpha_L too small
    assert not ok_bad


def test_pair_variance_box_matches_float():
    Vc = float(w2.var_cos_upper_on_box())
    gx, gw = np.polynomial.legendre.leggauss(400)
    xi, xo = 0.8 * gx, 0.4 * gx + 1.6
    al = float(sk.ALPHA0)
    E = lambda h: (np.sum(h(xi) * gw) / 2 + np.sum(h(xo) * gw) / 2) / 2   # P_c is symmetric: O by its right half
    v = E(lambda t: np.cos(al * t) ** 2) - E(lambda t: np.cos(al * t)) ** 2
    assert v <= Vc <= v + 1e-3


def test_Dc_second_derivative_and_uniqueness():
    for al in (0.7, 1.79, 3.3):
        h = 1e-4
        fd = (sk.Dc(arb(al + h)) - 2 * sk.Dc(arb(al)) + sk.Dc(arb(al - h))) / h ** 2
        enc = w2.Dc_second(arb(al))
        assert abs(float(enc.mid()) - float(fd.mid())) < 1e-5
    uq = w2.uniqueness_on_box()
    assert uq["ok"] and uq["D2_lower"] > 2.4
    bad = w2.uniqueness_on_box(centre=Fraction(3), r=Fraction(1, 2))   # D_c'' changes sign on a wide box
    assert not bad["ok"]


def test_placement_box_pass_and_fail():
    ok = w2.placement_on_box(1.30)
    assert ok["ok"] and abs(ok["G_plus_lower"] - 0.887) < 2e-3   # registered width-2 value G = +0.890 at alpha*
    bad = w2.placement_on_box(1.30, centre=Fraction(3), r=Fraction(1, 100))
    assert not bad["ok"]


def test_moments_exact():
    mo = w2.moments()
    assert mo["m"] == Fraction(11, 25) and mo["mass0"] == mo["mass1"] == Fraction(1, 2)
    assert mo["Vx"] == Fraction(11917, 7500) and mo["Eabs"] == Fraction(53, 50)
    sym = ((Fraction(-4, 5), Fraction(4, 5), Fraction(5, 16), 0), (Fraction(-2), Fraction(-6, 5), Fraction(5, 16), 1),
           (Fraction(6, 5), Fraction(2), Fraction(5, 16), 1))
    ms = w2.moments(sym)
    assert ms["m"] == 0 and ms["Vx"] == sk.VAR_X_C


def test_max_interval_mass():
    assert w2.max_interval_mass(w2.PA, Fraction(2)) == Fraction(1, 2)
    assert w2.max_interval_mass(w2.PA, Fraction(4, 5)) == Fraction(1, 4)
    toy = ((Fraction(0), Fraction(1), Fraction(1, 2), 0), (Fraction(3), Fraction(4), Fraction(1, 2), 1))
    assert w2.max_interval_mass(toy, Fraction(1)) == Fraction(1, 2)
    assert w2.max_interval_mass(toy, Fraction(3)) == Fraction(1, 2)
    assert w2.max_interval_mass(toy, Fraction(7, 2)) == Fraction(3, 4)


def test_certify_logistic_pass_and_fail():
    k0, b0, _ = w2.logistic_float()
    good = w2.certify_logistic(k0, b0)
    assert good["ok"] and good["radius_upper"] < 1e-12
    bad = w2.certify_logistic(k0 + 1e-3, b0)                      # gradient too large for the ball
    assert not bad["ok"]


def test_gamma0_and_pa_constants_pass_fail():
    k0, b0, quad = w2.logistic_float()
    cert = w2.certify_logistic(k0, b0)
    al0, be0 = w2.residual_gap_float(k0, b0, quad)
    gam = w2.certify_gamma0(k0, b0, al0, be0, cert["_rc"])
    assert 0.37 < gam["gamma0_lower"] <= gam["gamma_at_point"] < 0.39
    g_wrong = w2.certify_gamma0(k0, b0, al0, be0 + math.pi, cert["_rc"])   # the opposite phase: negative
    assert g_wrong["gamma0_lower"] < 0
    J = {r["a"]: r for r in _json()["continuous_asymmetric"]["rows"]}
    for a in (1.30, 1.60):
        sa = Fraction(J[a]["s_a_exact"])
        ok, det = w2.pa_constants(a, sa, gam["_g0"])
        assert ok and abs(det["W_a"] - J[a]["W_a"]) < 1e-9
        assert w2.pa_constants(a, sa / 10, gam["_g0"])[0]
        assert not w2.pa_constants(a, 2 * sa, gam["_g0"])[0]
    assert not w2.pa_constants(1.30, Fraction(1, 1000), arb("0.0001"))[0]   # no residual gap: no theorem
    sym = ((Fraction(-4, 5), Fraction(4, 5), Fraction(5, 16), 0), (Fraction(-2), Fraction(-6, 5), Fraction(5, 16), 1),
           (Fraction(6, 5), Fraction(2), Fraction(5, 16), 1))
    assert not w2.pa_constants(1.30, Fraction(1, 1000), gam["_g0"], windows=sym, tau=Fraction(1, 10))[0]


def test_s_unplaced_formula():
    m, vx = Fraction(11, 25), Fraction(11917, 7500)
    for a in (1.02, 1.30, 1.60):
        su = w2.s_unplaced(a, m, vx)
        aq = Fraction(a).limit_denominator(10 ** 6)
        # at s_u the placed lower bound equals the pure-ramp upper bound exactly
        assert su / 4 * (aq * m * Fraction(5, 7) + 2 * aq) == m ** 2 / (8 * vx)
    assert w2.s_unplaced(1.30, m, vx) > w2.s_unplaced(1.60, m, vx)
    assert w2.s_unplaced(1.30, Fraction(0), vx) == 0              # symmetric data: no ramp reward, no s_u


def test_largest_scale_toy():
    assert w2.largest_scale(lambda s: s < Fraction(123, 10000)) == Fraction(12, 1000)
    assert w2.largest_scale(lambda s: False) is None
    assert w2.largest_scale(lambda s: True) == 1


def test_committed_numbers():
    d = _json()
    L = d["ledger"]
    c = d["continuous_symmetric"]
    assert c["localisation"]["ok"] and L["W2 localisation ok"] == 1 and c["uniqueness"]["ok"]
    assert all(r["conds"] == [True] * 5 and r["placement"]["ok"] for r in c["rows"])
    assert L["W2 bound max over a"] == max(r["W2"] for r in c["rows"]) and L["W2 bound max over a"] < 9
    assert L["W2 s2 min over a"] == min(r["s2"] for r in c["rows"]) >= 0.006
    assert all(r["W2"] >= r["two_A0"] for r in c["rows"])
    p = d["continuous_asymmetric"]
    assert p["logistic"]["ok"] and p["F_tau1"] == "1/2" and p["m"] == "11/25"
    assert all(r["conds"] == [True] * 3 and r["s_a"] < r["s_u"] for r in p["rows"])
    assert abs(L["W2 Pa kappa*"] - 0.28255) < 1e-5 and abs(L["W2 Pan m"] - 0.44) < 1e-12
    assert d["population_asymmetric"]["n_outer"] == d["population_asymmetric"]["n_inner"] == 400
