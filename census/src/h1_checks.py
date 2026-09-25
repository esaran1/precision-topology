"""Sanity checks for results/h1_proof_attempt.md (H1: localisation of the gap maximiser).

These are NUMERICAL CHECKS of an analytic proof, not part of it, except item (A), which is an exact rational
evaluation used in the proof (G0 at the rational point (p, q) = (1.6, 1.2)).

    nice -n 15 python -m src.h1_checks

(A) exact G0(1.6, 1.2) in rational arithmetic.
(B) x_a^2 <= 6 eps and beta_a^2 <= eps(2 + eps) on a grid of eps in (0, 0.6878].
(C) the explicit constants: Lambda(eps), the p and q bounds, the K(eps) lower bound, eps thresholds.
(D) random placements at several eps: every placement with G > 0 satisfies the necessary conditions of Lemma 1
    (1.4 w1 < x_a; |m_j| < beta for the positive orientation; the ellipse inequality; G <= 0.4 eps w1).
(E) grid argmax of the rescaled gap at a few eps: location inside C, and K(eps) against the proved lower bound.
"""

from __future__ import annotations

import math
from fractions import Fraction as Fr

import numpy as np


def brentq(f, lo, hi, it=200):
    """Plain bisection (no scipy in this venv); f(lo) and f(hi) must differ in sign."""
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)

I_WIN = (-0.8, 0.8)
O_WINS = ((-2.0, -1.2), (1.2, 2.0))
CPRIME = 1.96 / 6            # (1.4)^2 / 6
CS = 2 + 0.36 / CPRIME       # Cauchy-Schwarz constant for |mu + 0.6 p|


# ---------------------------------------------------------------- (A) exact G0 at (1.6, 1.2)
def h_fr(s):
    return -s + s ** 3 / 6


def g0_exact_rational(p, q):
    """Exact G0(p, q) for rational p > 0, q, provided no window's sigma-range contains +-sqrt(2) in a way that
    makes an interior critical point the extremum.  We handle it exactly: h' = 0 at +-sqrt(2); local max at -sqrt(2)
    (value 2 sqrt 2/3), local min at +sqrt(2) (value -2 sqrt 2/3).  Returns (G0 as float of the exact value,
    and the exact pieces) -- membership of +-sqrt(2) is decided by comparing squares of rationals with 2."""
    def ext(lo, hi):
        s0, s1 = p * lo + q, p * hi + q
        vals = [h_fr(s0), h_fr(s1)]
        crit = []
        # +sqrt2 in (s0, s1)  <=>  s0 < sqrt2 < s1 ;  -sqrt2 in (s0, s1)  <=>  s0 < -sqrt2 < s1  (exact, via squares)
        if (s0 < 0 or s0 * s0 < 2) and (s1 > 0 and s1 * s1 > 2):
            crit.append(1)
        if (s0 < 0 and s0 * s0 > 2) and (s1 >= 0 or s1 * s1 < 2):
            crit.append(-1)
        return vals, crit
    out = {}
    for name, (lo, hi) in {"I": (Fr(-4, 5), Fr(4, 5)), "Om": (Fr(-2), Fr(-6, 5)), "Op": (Fr(6, 5), Fr(2))}.items():
        out[name] = ext(lo, hi)
    return out


# ---------------------------------------------------------------- vectorised exact gap of f_a
def fa(t, a):
    return t + a * np.sin(t)


def extrema(a, w1, b1, lo, hi):
    t0, t1 = w1 * lo + b1, w1 * hi + b1          # w1 > 0 assumed
    v0, v1 = fa(t0, a), fa(t1, a)
    mn, mx = np.minimum(v0, v1), np.maximum(v0, v1)
    c = math.acos(-1.0 / a)
    kmin = int(np.floor((t0.min() - c) / (2 * math.pi))) - 1
    kmax = int(np.ceil((t1.max() + c) / (2 * math.pi))) + 1
    for k in range(kmin, kmax + 1):
        for sgn in (1.0, -1.0):
            tc = sgn * c + 2 * math.pi * k
            ins = (tc > t0) & (tc < t1)
            if ins.any():
                val = tc + a * math.sin(tc)
                mn = np.where(ins, np.minimum(mn, val), mn)
                mx = np.where(ins, np.maximum(mx, val), mx)
    return mn, mx


def gap(a, w1, b1):
    iml, imx = extrema(a, w1, b1, *I_WIN)
    o = [extrema(a, w1, b1, *w) for w in O_WINS]
    omn = np.minimum(o[0][0], o[1][0]); omx = np.maximum(o[0][1], o[1][1])
    g1, g2 = omn - imx, iml - omx
    return np.maximum(g1, g2), g1, g2


def x_a(a):
    return brentq(lambda x: math.sin(x) - x / a, 1e-9, math.pi)


def main():
    rows = []
    # (A)
    ex = g0_exact_rational(Fr(8, 5), Fr(6, 5))
    print("(A) window values at (p,q)=(1.6,1.2):")
    for k, (vals, crit) in ex.items():
        print(f"    {k}: endpoint h = {[str(v) for v in vals]} = {[float(v) for v in vals]}, interior crit signs {crit}")
    maxI = max(ex["I"][0])                      # interior crit of I is +sqrt2 (local min) only -> max at endpoints
    minO = min(min(ex["Om"][0]), min(ex["Op"][0]))  # Om interior crit is -sqrt2 (local max); Op has none
    assert ex["I"][1] == [1] and ex["Om"][1] == [-1] and ex["Op"][1] == []
    G0 = minO - maxI
    print(f"    G0(1.6,1.2) orientation O-over-I = {G0} = {float(G0):.10f}")
    # other orientation is negative: min_I - max_O <= h(sigma(0.8)) - h(sigma(2)) ...
    # other orientation: min_I h - max_O h <= h(sigma(-0.8)) - h(sigma(2)) (exact rationals); must be < 0
    other = h_fr(Fr(8, 5) * Fr(-4, 5) + Fr(6, 5)) - h_fr(Fr(8, 5) * 2 + Fr(6, 5))
    print(f"    other orientation <= {float(other):.6f} (< 0)"); assert other < 0
    G0f = float(G0)

    # (B)
    eps_grid = np.concatenate([np.logspace(-8, -1, 400), np.linspace(0.1, 0.6878, 400)])
    worst_x, worst_b = 0.0, 0.0
    for e in eps_grid:
        a = 1 + e
        worst_x = max(worst_x, x_a(a) ** 2 / (6 * e))
        worst_b = max(worst_b, math.acos(1 / a) ** 2 / (e * (2 + e)))
    print(f"(B) max x_a^2/(6 eps) = {worst_x:.6f} (<= 1 required); max beta^2/(eps(2+eps)) = {worst_b:.6f}")

    # (C) constants
    def Lam(e):
        k1 = 1 - e * (2 + e) / 12
        k2 = 1 - 0.3 * e
        k3 = k2 * (1 - e * (2 + e) / 2)
        return 1 / ((1 + e) * min(k1, k3))

    def Mloc(e):   # sup over |sigma| <= 4.4 of |phi_eps - h| / eps, valid for eps <= 0.2
        return 12 ** 1.5 / 15 + (1 + e) * e * 4.4 ** 7 / 5040

    def Klow(e):
        return G0f - 2 * e * Mloc(e)
    e_pos = brentq(lambda e: Klow(e), 1e-6, 0.2)
    e_04 = brentq(lambda e: Klow(e) - 0.4, 1e-6, 0.2)
    print(f"(C) sup|g| on [0,4.4] check: {max(abs(s**3/6 - 1.2*s**5/120) for s in np.linspace(0,4.4,100001)):.4f} (eps=0.2),"
          f" {max(abs(s**3/6 - s**5/120) for s in np.linspace(0,4.4,100001)):.4f} (eps=0); bound 12^1.5/15 = {12**1.5/15:.4f}")
    print(f"    K(eps) lower bound > 0 for eps < {e_pos:.5f}; > 0.4 for eps < {e_04:.5f}")
    for e in (0.001, 0.01, 0.02, 0.029, 0.05, 0.1, 0.2):
        L = Lam(e)
        print(f"    eps={e}: Lambda={L:.5f}  p < {math.sqrt(L / CPRIME):.4f}  |q| < {math.sqrt(CS * L):.4f}"
              f"  box: p < {math.sqrt(6)/1.4:.4f}, |q| < {math.sqrt(2+e) + 0.6*math.sqrt(6)/1.4:.4f}  Klow={Klow(e):.4f}")

    # (D) random placements
    rng = np.random.default_rng(1)
    print("(D) random placements (w1 > 0, b1 in [0, 2pi)):")
    for e in (0.5, 0.2, 0.05, 0.029, 0.01, 1e-3):
        a = 1 + e
        xa, beta = x_a(a), math.acos(1 / a)
        n = 400_000
        w1 = np.concatenate([rng.uniform(1e-6, 3.0, n), math.sqrt(e) * rng.uniform(0, 3, n)])
        b1 = np.concatenate([rng.uniform(0, 2 * math.pi, n), math.pi + math.sqrt(e) * rng.uniform(-4, 4, n)])
        G, g1, g2 = gap(a, w1, b1)
        pos = G > 0
        # necessary conditions
        c1 = np.all(1.4 * w1[pos] < xa)
        # midpoints of the positive orientation, reduced mod 2pi to (-pi, pi]
        m1 = np.angle(np.exp(1j * (b1 - 0.6 * w1 - math.pi)))
        m2 = np.angle(np.exp(1j * (b1 + 0.6 * w1 - math.pi)))
        ok1 = (g1 <= 0) | (np.abs(m1) < beta)
        ok2 = (g2 <= 0) | (np.abs(m2) < beta)
        c2 = np.all(ok1[pos] & ok2[pos])
        # ellipse (rescaled) for the positive orientation(s)
        p = w1 / math.sqrt(e)
        mu1, mu2 = m1 / math.sqrt(e), m2 / math.sqrt(e)
        L = Lam(e)
        el1 = (g1 <= 0) | (mu1 ** 2 / 2 + CPRIME * p ** 2 < L)
        el2 = (g2 <= 0) | (mu2 ** 2 / 2 + CPRIME * p ** 2 < L)
        c3 = np.all(el1 & el2)
        ratio = np.max(G[pos] / (0.4 * e * w1[pos])) if pos.any() else float("nan")
        dmax = 2 * (math.sqrt(a * a - 1) - beta)
        print(f"    eps={e}: n={2*n}, #G>0={pos.sum()}, 1.4w1<x_a: {c1}, |m|<beta: {c2}, ellipse: {c3}, "
              f"max G/(0.4 eps w1) = {ratio:.4f}, max G/dropmax = {G.max()/dmax:.4f}")
        rows.append((e, pos.sum(), c1, c2, c3, ratio))
        assert c1 and c2 and c3 and ratio <= 1 + 1e-9

    # cross-check the vectorised gap against src.exact_extrema on a few points
    try:
        from .exact_extrema import exact_gap
        a = 1.05
        w = rng.uniform(0.01, 0.6, 200); b = rng.uniform(0, 2 * math.pi, 200)
        G, _, _ = gap(a, w, b)
        d = max(abs(G[i] - exact_gap(a, float(w[i]), float(b[i]))[0]) for i in range(200))
        print(f"    cross-check vs src.exact_extrema.exact_gap (200 pts, a=1.05): max diff {d:.2e}")
    except Exception as exc:  # pragma: no cover
        print("    cross-check skipped:", exc)

    # (E) grid argmax in rescaled coordinates
    print("(E) grid argmax of the rescaled gap (p in (0, 1.8], q in [-2.6, 2.6]):")
    P, Q = np.meshgrid(np.linspace(1e-3, 1.8, 901), np.linspace(-2.6, 2.6, 1301), indexing="ij")
    P, Q = P.ravel(), Q.ravel()
    for e in (0.2, 0.05, 0.029, 0.01, 1e-3, 1e-4):
        a = 1 + e
        G, _, _ = gap(a, math.sqrt(e) * P, math.pi + math.sqrt(e) * Q)
        Gt = G / e ** 1.5
        i = int(np.argmax(Gt))
        print(f"    eps={e}: K_grid={Gt[i]:.5f} at (p,q)=({P[i]:.3f},{Q[i]:.3f}); "
              f"max over p<1: {Gt[P < 1].max():.4f}; max over |q|>=2: {Gt[np.abs(Q) >= 2].max():.4f}; "
              f"proved lower bound {Klow(e) if e <= 0.2 else float('nan'):.4f}")


if __name__ == "__main__":
    main()
