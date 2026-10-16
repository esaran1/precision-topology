"""Track K follow-up (math note §22): width-2 small-scale compactness and the ramp limit.  Theory only; no training.

Every constant §22 uses, in ball arithmetic (python-flint arb at 128 bits, rigorous integrals by acb.integral) or in
exact rationals.  Floats are used only to choose test points (the logistic-regression start and the residual-gap
frequency), never as a bound.

  P_c  (symmetric continuous windows, §21's object): the two-case exclusion of escaping units (Theorem K11):
       the localisation threshold alpha_L, and for each a the largest verified scale s2 and the bound W2.
       The limit's placement: D_c < 0 and G_+ >= a*g_lo > 0 on the maximiser box.
  P_a  (asymmetric continuous windows, Delta = 0.4: 1/2 U[-0.8, 0.8] + 1/2 U([-2, -1.2] u [1.2, 2.4])): exact moments,
       the spreading fraction at tau = 1, a certified enclosure of the logistic-regression optimum (kappa*, b*), a
       certified lower bound gamma0 on the residual-weighted sine gap, and for each a the non-attainment scale s_a and
       W_a (Theorem K12), and the unplaced scale s_u (Proposition K13).
  P_an (the 800-point asymmetric population of Track 2, asym_pilot.population(0.4)): exact m and Var(x) of its binary64
       points, and s_u there.

One process, nice 15, under a few seconds of CPU.  Writes results/small_scale_w2.json.

    python -m src.small_scale_w2
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path

from flint import acb, arb, ctx

from .small_scale_k import (ALPHA0, F_C, R_BOX, VAR_X_C, Dc, D_lower_at, _a, _cell, _lower_fraction,
                            _upper_fraction, var_cos_c)

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "small_scale_w2.json"
PREC = 128
A_ALL = (1.02, 1.05, 1.10, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
TAU_C = Fraction(2, 5)                     # §17's spreading half-length on P_c
ETA_BAR = Fraction(1, 20)                  # the localisation threshold is D_lo - ETA_BAR
SMALL = Fraction(1, 2)                     # |D_c| <= (32/15) alpha^2 on (0, SMALL]

# P_a: (lo, hi, density, label).  Class 0 has mass 1/2 on I (density 5/16); class 1 mass 1/2 on O (length 2, density 1/4).
PA = ((Fraction(-4, 5), Fraction(4, 5), Fraction(5, 16), 0),
      (Fraction(-2), Fraction(-6, 5), Fraction(1, 4), 1),
      (Fraction(6, 5), Fraction(12, 5), Fraction(1, 4), 1))
TAU_A = Fraction(1)                        # spreading half-length on P_a (closed intervals of length 2)


def _aa(a: float) -> arb:
    return _a(Fraction(a).limit_denominator(10 ** 6))


# ============================================================================================ symmetric windows P_c
def var_cos_upper_on_box(centre: Fraction = ALPHA0, r: Fraction = R_BOX, cells: int = 400) -> arb:
    """Upper end of Var(cos(alpha x)) under P_c over [centre - r, centre + r]: the pair's variance is a^2 times this."""
    ctx.prec = PREC
    best = None
    h = 2 * r / cells
    for i in range(cells):
        v = var_cos_c(_cell(centre - r + i * h, centre - r + (i + 1) * h)).upper()
        best = v if best is None or v > best else best
    return best


def lower_localisation(thr: arb, max_depth: int = 40, keep_width: Fraction = Fraction(1, 10 ** 4)):
    """Certify alpha_L with: every alpha > 0 with |D_c(alpha)| >= thr has alpha >= alpha_L.

    (0, SMALL] is cleared analytically when (32/15) SMALL^2 < thr; alpha > A = 3/(0.8 thr) is cleared by the decay
    bound.  [SMALL, A] is bisected: a cell is cleared when the enclosure of |D_c| has upper end < thr, kept when its
    width is <= keep_width, else split.  alpha_L = the smallest kept lower end.  ok requires no unresolved cell and at
    least one kept cell (a threshold above max |D_c| keeps nothing and is reported as not ok)."""
    ctx.prec = PREC
    small_ok = Fraction(32, 15) * SMALL ** 2 < _lower_fraction(thr)
    A_hi = _upper_fraction(3 / (_a(Fraction(4, 5)) * thr))
    stack = [(SMALL, A_hi, 0)]
    kept, failed, cleared = [], [], 0
    while stack:
        lo, hi, d = stack.pop()
        if abs(Dc(_cell(lo, hi))).upper() < thr:
            cleared += 1
            continue
        if hi - lo <= keep_width:
            kept.append((lo, hi))
            continue
        if d >= max_depth:
            failed.append((float(lo), float(hi)))
            continue
        mid = (lo + hi) / 2
        stack += [(lo, mid, d + 1), (mid, hi, d + 1)]
    ok = bool(small_ok and kept and not failed)
    return {"ok": ok, "alpha_L": float(min(c[0] for c in kept)) if kept else None,
            "alpha_L_exact": min(c[0] for c in kept) if kept else None,
            "kept_span_hi": float(max(c[1] for c in kept)) if kept else None,
            "kept_cells": len(kept), "cleared_cells": cleared, "failed_cells": failed[:5]}


def pc_constants(a: float, s: Fraction, D_lo: arb, Vc_hi: arb, alpha_L: Fraction, thr: arb):
    """Theorem K11's conditions at scale s (each side is monotone in s; see §22.2).  Returns (ok, details)."""
    ctx.prec = PREC
    aa, ss, F, tau, vx = _aa(a), _a(s), _a(F_C), _a(TAU_C), _a(VAR_X_C)
    sqvx = vx.sqrt()
    A0 = 3 / (_a(Fraction(4, 5)) * D_lo)
    Vp = aa ** 2 * Vc_hi                                     # >= Var(phi) of the cancelling pair at any maximiser
    psi2 = arb(1).cosh().log()
    c0 = ss < (8 * F * psi2 / Vp).sqrt()                     # K6(i) applies
    cbar = (aa + (_a(Fraction(6, 5)) * Vp / F).sqrt()) / tau # |c| <= cbar
    eta0 = ss * Vp / (2 * aa)
    epsb = ss * Vp / (aa * D_lo)                             # weight of a unit with |alpha| >= 2 A0
    c1 = 2 * epsb < 1                                        # at most one unit with |alpha| >= 2 A0
    eta1 = eta0 / (1 - epsb)
    c2 = (D_lo - eta1 > thr) and (eta1 < D_lo / 2)
    kc = 1 - (1 - eta1 / D_lo) ** 2                          # cos^2 beta_1 <= kc
    B = aa * kc.sqrt()
    base = (1 - epsb) * _a(alpha_L) * sqvx - B
    P_lo = base ** 2 if base > 0 else arb(0)
    M = (cbar * sqvx + aa) ** 2 + aa ** 2
    R = 4 * cbar + 2 * aa
    lhs2 = P_lo - aa ** 2 * (2 * epsb + kc) - 2 * epsb * aa * M.sqrt()
    rhs2 = ss ** 2 * R ** 4 / 24
    c3 = base > 0 and lhs2 > rhs2                            # Case 2 (and single units) strictly beaten by the pair
    K1 = _a(Fraction(15, 8)) + ss * A0 * (1 + aa) / 2
    den = aa * (D_lo - eta1) - ss * (2 * A0 * sqvx + aa) ** 2
    c4 = den > 0
    W1 = 4 * aa * K1 / den if c4 else None
    W2 = max(float((2 * A0).upper()), float(W1.upper())) if c4 else None
    ok = bool(c0 and c1 and c2 and c3 and c4)
    return ok, {"s": float(s), "cbar": float(cbar.upper()), "eps_bar": float(epsb.upper()), "eta1": float(eta1.upper()),
                "cos2_bound": float(kc.upper()), "case2_lhs": float(lhs2.lower()), "case2_rhs": float(rhs2.upper()),
                "R": float(R.upper()), "K1": float(K1.upper()), "W1": float(W1.upper()) if c4 else None, "W2": W2,
                "two_A0": float((2 * A0).upper()), "conds": [bool(c0), bool(c1), bool(c2), bool(c3), bool(c4)]}


def largest_scale(check, lo: Fraction = Fraction(1, 10 ** 8), hi: Fraction = Fraction(1), iters: int = 60):
    """Bisection (in log scale) for the largest s with check(s) True, then rounded DOWN to two significant digits and
    re-verified.  Returns the verified Fraction or None."""
    if not check(lo):
        return None
    if check(hi):
        return hi
    import math
    l, h = math.log(float(lo)), math.log(float(hi))
    for _ in range(iters):
        m = (l + h) / 2
        if check(Fraction(math.exp(m)).limit_denominator(10 ** 12)):
            l = m
        else:
            h = m
    s = math.exp(l)
    e = math.floor(math.log10(s)) - 1
    s2 = Fraction(math.floor(s / 10 ** e)) * Fraction(10) ** e
    return s2 if check(s2) else None


def placement_on_box(a: float, centre: Fraction = ALPHA0, r: Fraction = R_BOX, cells: int = 400):
    """On [centre - r, centre + r]: D_c < 0 (upper end), and g(alpha) = cos(0.8 alpha) - max(cos 1.2 alpha, cos 2 alpha)
    has a positive lower end.  Valid when 0.8 alpha < pi and 1.2 alpha <= pi <= 2 alpha < 2 pi on the box (checked):
    then min over I of cos(alpha x) is cos(0.8 alpha) and max over O is at |x| = 1.2 or 2, so the window gap of
    phi = -a cos(alpha x) is G_+ = a g(alpha)."""
    ctx.prec = PREC
    pi = arb.pi()
    lo_b, hi_b = _a(centre - r), _a(centre + r)
    geom = bool((_a(Fraction(4, 5)) * hi_b < pi) and (_a(Fraction(6, 5)) * hi_b < pi) and (2 * lo_b > pi)
                and (2 * hi_b < 2 * pi))
    h = 2 * r / cells
    g_lo, D_hi = None, None
    for i in range(cells):
        al = _cell(centre - r + i * h, centre - r + (i + 1) * h)
        g = ((_a(Fraction(4, 5)) * al).cos().lower()
             - max((_a(Fraction(6, 5)) * al).cos().upper(), (2 * al).cos().upper()))
        d = Dc(al).upper()
        g_lo = g if g_lo is None or g < g_lo else g_lo
        D_hi = d if D_hi is None or d > D_hi else D_hi
    ok = bool(geom and D_hi < 0 and g_lo > 0)
    return {"ok": ok, "geometry_ok": geom, "D_upper_on_box": float(D_hi), "g_lower": float(g_lo),
            "G_plus_lower": float((_aa(a) * g_lo).lower())}


def Dc_second(alpha: arb) -> arb:
    """D_c''(alpha) = E_I[x^2 cos(alpha x)] - E_O[x^2 cos(alpha x)] on P_c, in closed form with the antiderivative
    F(x) = x^2 sin(ax)/a + 2x cos(ax)/a^2 - 2 sin(ax)/a^3 (both classes have density 1/1.6 and are symmetric)."""
    F = lambda x: (x ** 2 * (alpha * x).sin() / alpha + 2 * x * (alpha * x).cos() / alpha ** 2
                   - 2 * (alpha * x).sin() / alpha ** 3)
    c = _a(Fraction(5, 4))                                    # 2 / 1.6
    return c * (F(_a(Fraction(4, 5)))) - c * (F(arb(2)) - F(_a(Fraction(6, 5))))


def uniqueness_on_box(centre: Fraction = ALPHA0, r: Fraction = R_BOX, cells: int = 400):
    """D_c'' > 0 on every cell of the box (lower end), so D_c is strictly convex there; with D_c < 0 on the box and every
    maximiser of |D_c| inside it (Lemma K3), |D_c| has exactly one maximiser alpha_hat > 0."""
    ctx.prec = PREC
    h = 2 * r / cells
    lo = None
    for i in range(cells):
        v = Dc_second(_cell(centre - r + i * h, centre - r + (i + 1) * h)).lower()
        lo = v if lo is None or v < lo else lo
    return {"ok": bool(lo > 0), "D2_lower": float(lo)}


# ============================================================================================ asymmetric windows P_a
def moments(windows=PA):
    """Exact E x, E x^2, Var x, E|x|, m = E_O x - E_I x and the class masses of a piecewise-uniform measure."""
    Ex = sum(r * (hi ** 2 - lo ** 2) / 2 for lo, hi, r, _ in windows)
    Ex2 = sum(r * (hi ** 3 - lo ** 3) / 3 for lo, hi, r, _ in windows)
    Eabs = sum(r * (abs(hi) * hi - abs(lo) * lo) / 2 for lo, hi, r, _ in windows)
    mass = {0: Fraction(0), 1: Fraction(0)}
    first = {0: Fraction(0), 1: Fraction(0)}
    for lo, hi, r, y in windows:
        mass[y] += r * (hi - lo)
        first[y] += r * (hi ** 2 - lo ** 2) / 2
    m = first[1] / mass[1] - first[0] / mass[0]
    return {"Ex": Ex, "Ex2": Ex2, "Vx": Ex2 - Ex ** 2, "Eabs": Eabs, "m": m, "mass0": mass[0], "mass1": mass[1]}


def max_interval_mass(windows, length: Fraction) -> Fraction:
    """Exact sup over t of the mass of [t, t + length]: piecewise linear in t, so the max is at a t where t or t + length
    is a window endpoint."""
    ends = sorted({e for lo, hi, _, _ in windows for e in (lo, hi)})
    cands = set(ends) | {e - length for e in ends}
    mass = lambda t: sum(r * max(Fraction(0), min(hi, t + length) - max(lo, t)) for lo, hi, r, _ in windows)
    return max(mass(t) for t in cands)


def _sig(z):
    return 1 / (1 + (-z).exp())


def _integral(fn, lo: Fraction, hi: Fraction) -> arb:
    return acb.integral(lambda x, analytic: fn(x), _a(lo), _a(hi)).real


def logistic_float(windows=PA, n: int = 200):
    """Float Newton for min over (kappa, b) of E l(kappa x + b, y) (Gauss-Legendre); a start point only."""
    import numpy as np
    xs, ws, ys = [], [], []
    g, w = np.polynomial.legendre.leggauss(n)
    for lo, hi, r, y in windows:
        lo, hi, r = float(lo), float(hi), float(r)
        xs.append((hi - lo) / 2 * g + (hi + lo) / 2); ws.append(w * (hi - lo) / 2 * r); ys.append(np.full(n, y))
    x, w, y = np.concatenate(xs), np.concatenate(ws), np.concatenate(ys)
    k, b = 0.0, 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(k * x + b)))
        rr, q = w * (p - y), w * p * (1 - p)
        G = np.array([np.sum(rr * x), np.sum(rr)])
        H = np.array([[np.sum(q * x * x), np.sum(q * x)], [np.sum(q * x), np.sum(q)]])
        dk, db = np.linalg.solve(H, G)
        k, b = k - dk, b - db
    return k, b, (x, w, y)


def certify_logistic(k0: float, b0: float, windows=PA, r: Fraction = Fraction(1, 10 ** 6)):
    """Certified enclosure of the logistic-regression optimum (kappa*, b*) on a piecewise-uniform measure.

    The objective f(kappa, b) = E l(kappa x + b, y) is lambda-strongly convex on the ball B((k0, b0), r) with
    lambda = sigma'(Z) * lambda_min(E[(x, 1)(x, 1)^T]), Z = (|k0| + r) X + |b0| + r, X = max |x|.  If ||grad f(k0, b0)|| <
    lambda r / 2, the minimiser lies in the ball and within ||grad|| / lambda of (k0, b0)."""
    ctx.prec = PREC
    kq, bq = Fraction(k0), Fraction(b0)
    K, Bb = _a(kq), _a(bq)
    gk = sum(_a(rho) * _integral(lambda x: (_sig(acb(K) * x + acb(Bb)) - y) * x, lo, hi) for lo, hi, rho, y in windows)
    gb = sum(_a(rho) * _integral(lambda x: _sig(acb(K) * x + acb(Bb)) - y, lo, hi) for lo, hi, rho, y in windows)
    gnorm = (gk ** 2 + gb ** 2).sqrt()
    mo = moments(windows)
    e1, e2 = _a(mo["Ex"]), _a(mo["Ex2"])
    lam_M = (e2 + 1 - ((e2 - 1) ** 2 + 4 * e1 ** 2).sqrt()) / 2
    X = max(abs(e) for lo, hi, _, _ in windows for e in (lo, hi))
    Z = (abs(K) + _a(r)) * _a(X) + abs(Bb) + _a(r)
    sp = _sig(Z) * (1 - _sig(Z))
    lam = sp * lam_M
    ok = bool(gnorm < lam * _a(r) / 2)
    rc = gnorm / lam
    return {"ok": ok, "kappa0": str(kq), "b0": str(bq), "kappa0_float": k0, "b0_float": b0,
            "grad_norm_upper": float(gnorm.upper()), "lambda_lower": float(lam.lower()),
            "radius_upper": float(rc.upper()), "_rc": rc}


def residual_gap_float(k0: float, b0: float, quad, alphas=None):
    """Float search for the frequency maximising gamma(alpha) = max_beta E[(y - sigma0) sin(alpha x + beta)]
    = sqrt(S^2 + C^2); returns (alpha0, beta0).  A test point only."""
    import numpy as np
    x, w, y = quad
    res = w * (y - 1 / (1 + np.exp(-(k0 * x + b0))))
    alphas = np.linspace(0.01, 8, 8000) if alphas is None else alphas
    S = np.array([np.sum(res * np.sin(al * x)) for al in alphas])
    C = np.array([np.sum(res * np.cos(al * x)) for al in alphas])
    i = int(np.argmax(np.hypot(S, C)))
    return float(alphas[i]), float(np.arctan2(C[i], S[i]))


def certify_gamma0(k0: float, b0: float, alpha0: float, beta0: float, rc: arb, windows=PA):
    """Lower bound of gamma* = E[(y - sigma(kappa* x + b*)) sin(alpha0 x + beta0)], with (kappa*, b*) within rc of
    (k0, b0) in the Euclidean norm: |d gamma| <= (1/4)(E|x| |d kappa| + |d b|) <= (1/4)(E|x| + 1) rc."""
    ctx.prec = PREC
    K, Bb = _a(Fraction(k0)), _a(Fraction(b0))
    al, be = _a(Fraction(alpha0)), _a(Fraction(beta0))
    g = sum(_a(rho) * _integral(lambda x: (y - _sig(acb(K) * x + acb(Bb))) * (acb(al) * x + acb(be)).sin(), lo, hi)
            for lo, hi, rho, y in windows)
    Eabs = _a(moments(windows)["Eabs"])
    lo_b = g - (Eabs + 1) * rc / 4
    return {"gamma_at_point": float(g.mid()), "gamma0_lower": float(lo_b.lower()), "_g0": lo_b.lower(),
            "alpha0": alpha0, "beta0": beta0}


def pa_constants(a: float, s: Fraction, g0: arb, windows=PA, tau: Fraction = TAU_A):
    """Theorem K12's conditions at scale s.  Returns (ok, details)."""
    ctx.prec = PREC
    mo = moments(windows)
    F = 1 - max_interval_mass(windows, 2 * tau)
    aa, ss, m, vx, t = _aa(a), _a(s), _a(mo["m"]), _a(mo["Vx"]), _a(tau)
    Fa = _a(F)
    coef = Fa * t / 2 - abs(m) / 4
    c0 = coef > 0
    kbar = None
    if c0:
        k1, k2 = (Fa * arb(2).log() + ss * aa * (1 + Fa) / 2) / coef, ss * aa / t
        kbar = k1 if k1.upper() >= k2.upper() else k2         # an upper bound of both
    rho2 = 2 * sum(_a(r) for _, _, r, _ in windows)
    K1 = rho2 + kbar / 4 if c0 else None
    den = g0 - 7 * ss * aa / 8
    c1 = bool(c0 and den > 0)
    Wa = 2 * K1 / den if c1 else None
    c2 = bool(c1 and ss / 4 * (Wa * abs(m) + 2 * aa) < m ** 2 / (8 * vx))
    WR = K1 / (g0 - ss * aa / 8) if c1 else None
    ok = bool(c0 and c1 and c2)
    return ok, {"s": float(s), "F_tau": str(F), "coef": float(coef.lower()),
                "kappa_bar": float(kbar.upper()) if c0 else None, "K1": float(K1.upper()) if c0 else None,
                "W_a": float(Wa.upper()) if c1 else None, "W_R": float(WR.upper()) if c1 else None,
                "conds": [bool(c0), c1, c2]}


def s_unplaced(a: float, m: Fraction, vx: Fraction) -> Fraction:
    """Every theta with |c| <= a/1.4 (in particular every placed one) has L* >= log 2 - (s/4)(a|m|/1.4 + 2a), while the
    pure ramp gives L* <= log 2 - m^2/(8 Var x) at every s.  So placed points are strictly worse for s < s_u =
    m^2 / (2 Var x (a|m|/1.4 + 2a)).  Exact rational (a as its rational)."""
    aq = Fraction(a).limit_denominator(10 ** 6)
    return m ** 2 / (2 * vx * (aq * abs(m) / Fraction(7, 5) + 2 * aq))


def an_moments():
    """Exact m and Var(x) of the 800 binary64 points of asym_pilot.population(0.4), read as exact rationals."""
    from .asym_pilot import population
    x, y = population(0.4)
    X = [Fraction(float(t)) for t in x]
    Y = [int(v) for v in y]
    n = len(X)
    o = [u for u, v in zip(X, Y) if v == 1]
    i = [u for u, v in zip(X, Y) if v == 0]
    m = sum(o) / len(o) - sum(i) / len(i)
    Ex = sum(X) / n
    vx = sum(u * u for u in X) / n - Ex ** 2
    return {"n": n, "n_outer": len(o), "n_inner": len(i), "m": m, "Vx": vx}


# ============================================================================================================ main
def main():
    os.nice(15)
    ctx.prec = PREC
    out = {}
    # ---- P_c
    D_lo = D_lower_at(ALPHA0)
    Vc_hi = var_cos_upper_on_box()
    thr = D_lo - _a(ETA_BAR)
    loc = lower_localisation(thr)
    aL = loc["alpha_L_exact"]
    crow = []
    for a in A_ALL:
        chk = lambda s, a=a: pc_constants(a, s, D_lo, Vc_hi, aL, thr)[0]
        s2 = largest_scale(chk)
        _, det = pc_constants(a, s2, D_lo, Vc_hi, aL, thr)
        pl = placement_on_box(a)
        crow.append({"a": a, "s2": float(s2), "s2_exact": str(s2), **{k: v for k, v in det.items() if k != "s"},
                     "placement": pl})
    uq = uniqueness_on_box()
    out["continuous_symmetric"] = {"D_lo": float(D_lo), "uniqueness": uq, "var_cos_upper_box": float(Vc_hi), "threshold": float(thr),
                                   "eta_bar": float(ETA_BAR),
                                   "localisation": {k: v for k, v in loc.items() if k != "alpha_L_exact"},
                                   "rows": crow}
    # ---- P_a
    mo = moments()
    F_a = 1 - max_interval_mass(PA, 2 * TAU_A)
    k0, b0, quad = logistic_float()
    cert = certify_logistic(k0, b0)
    al0, be0 = residual_gap_float(k0, b0, quad)
    gam = certify_gamma0(k0, b0, al0, be0, cert["_rc"])
    g0 = gam["_g0"]
    arow = []
    for a in A_ALL:
        chk = lambda s, a=a: pa_constants(a, s, g0)[0]
        sa = largest_scale(chk)
        _, det = pa_constants(a, sa, g0)
        su = s_unplaced(a, mo["m"], mo["Vx"])
        arow.append({"a": a, "s_a": float(sa), "s_a_exact": str(sa), **{k: v for k, v in det.items() if k != "s"},
                     "s_u": float(su)})
    out["continuous_asymmetric"] = {"m": str(mo["m"]), "Ex": str(mo["Ex"]), "Ex2": str(mo["Ex2"]),
                                    "Vx": str(mo["Vx"]), "Vx_float": float(mo["Vx"]), "Eabs": str(mo["Eabs"]),
                                    "masses": [str(mo["mass0"]), str(mo["mass1"])], "F_tau1": str(F_a),
                                    "L0_upper_quadratic": float((arb(2).log() - _a(mo["m"] ** 2 / (8 * mo["Vx"]))).upper()),
                                    "logistic": {k: v for k, v in cert.items() if not k.startswith("_")},
                                    "gamma": {k: v for k, v in gam.items() if not k.startswith("_")}, "rows": arow}
    # ---- P_an
    an = an_moments()
    out["population_asymmetric"] = {"n": an["n"], "n_outer": an["n_outer"], "n_inner": an["n_inner"],
                                    "m": float(an["m"]), "Vx": float(an["Vx"]),
                                    "rows": [{"a": a, "s_u": float(s_unplaced(a, an["m"], an["Vx"]))} for a in A_ALL]}
    cr = {r_["a"]: r_ for r_ in crow}
    ar = {r_["a"]: r_ for r_ in arow}
    pn = {r_["a"]: r_ for r_ in out["population_asymmetric"]["rows"]}
    out["ledger"] = {
        "W2 D_lo": float(D_lo), "W2 var cos upper box": float(Vc_hi), "W2 localisation ok": float(loc["ok"]),
        "W2 alpha_L": loc["alpha_L"],
        "W2 s2 a=1.30": cr[1.30]["s2"], "W2 bound a=1.30": cr[1.30]["W2"],
        "W2 s2 a=1.60": cr[1.60]["s2"], "W2 bound a=1.60": cr[1.60]["W2"],
        "W2 s2 min over a": min(r_["s2"] for r_ in crow), "W2 bound max over a": max(r_["W2"] for r_ in crow),
        "W2 two A0": cr[1.30]["two_A0"],
        "W2 placement g lower": min(r_["placement"]["g_lower"] for r_ in crow),
        "W2 uniqueness ok": float(uq["ok"]), "W2 D2 lower on box": uq["D2_lower"],
        "W2 placement ok all a": float(all(r_["placement"]["ok"] for r_ in crow)),
        "W2 G_plus lower a=1.30": cr[1.30]["placement"]["G_plus_lower"],
        "W2 Pa m": float(mo["m"]), "W2 Pa Var x": float(mo["Vx"]), "W2 Pa F tau=1": float(F_a),
        "W2 Pa kappa*": k0, "W2 Pa b*": b0, "W2 Pa logistic certified": float(cert["ok"]),
        "W2 Pa logistic radius": cert["radius_upper"],
        "W2 Pa alpha0": al0, "W2 Pa gamma0 lower": gam["gamma0_lower"],
        "W2 Pa s_a a=1.30": ar[1.30]["s_a"], "W2 Pa W_a a=1.30": ar[1.30]["W_a"],
        "W2 Pa s_a a=1.60": ar[1.60]["s_a"], "W2 Pa W_a a=1.60": ar[1.60]["W_a"],
        "W2 Pa s_u a=1.30": ar[1.30]["s_u"], "W2 Pa s_u a=1.60": ar[1.60]["s_u"],
        "W2 Pan m": float(an["m"]), "W2 Pan s_u a=1.30": pn[1.30]["s_u"]}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
