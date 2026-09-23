"""Block 2: certified analysis of the limit (ε -> 0) profiled conditional problem.

Limit problem (T58/T70).  Logit z = A·h(σ(x)) + b2, σ = p x + q, h(σ) = -σ + σ³/6, A > 0
(A = s ε^{3/2} at leading order), profiled over b2.  L0*(p, q; A) = min_{b2} mean ℓ(z, y).
Limit class gap (A > 0 orientation): G0(p, q) = min_O h(σ) - max_I h(σ); the finite-a gap is
ε^{3/2}(G0 + O(ε)).  R = K·A/2 at leading order, K = sup G0 = 0.579454926.

Localisation (H1', proved here given the computed bounds):
  h is increasing on (-inf, -√2] ∪ [√2, inf) and h(2√2) = h(-√2) = 2√2/3·... = 0.9428, h(-2√2) =
  h(√2) = -0.9428; hence h restricted to {|σ| >= 2√2} is increasing in σ.  For any (p, q), the
  data points with |σ(x)| < 2√2 lie in an x-window of length 4√2/|p|; on the rest the logit is a
  monotone function of x, so L0* >= (1/n)·(best monotone logistic fit of the remaining points),
  and the best monotone fit is the isotonic regression of y (the same minimiser for every Bregman
  loss, including log loss).  B(P) = min over window positions of that bound with window length
  4√2/P lower-bounds L0* for all |p| >= P.  If |q| > 2√2 + 2|p| no point is in the window and
  L0* >= B_full.  So the global minimiser lies in K(P) = {|p| <= P, |q| <= 2√2 + 2P} whenever the
  certified minimum is below min(B(P), B_full).

Certified bounds on a sup-norm cell (centre c, half-widths hp, hq), per point:
  gradient (envelope): ∂p L0* = mean((σ(z)-y) A h'(σ) x), ∂q L0* = mean((σ(z)-y) A h'(σ)).
  curvature: δᵀ∇²L0* δ >= -A·mean(|h''| (|x|hp + hq)²) with |h''(σ)| = |σ| <= |σ_c| + |x|hp + hq.
  gap step: |ΔG0| <= max|h'| over the cell · ((0.8 + 2) hp + 2 hq), max|h'| <= 1 + σmax²/2.

    python -m src.limit_bnb localisation          # B(P) and the box
    python -m src.limit_bnb switch                # certified A*, H2', and the solve switch
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .profiled_bnb import _sigmoid, _softplus

RESULTS = Path(__file__).resolve().parents[1] / "results"
K_LIMIT = 0.579454926
R2 = math.sqrt(2)


def h(s):
    return -s + s ** 3 / 6


def hp(s):
    return -1 + s ** 2 / 2


def population():
    from . import blockB_landscape as bb
    x, y = bb.population_data()
    return x.numpy(), y.numpy()


def profile(p, q, A, x, y):
    p = np.asarray(p, float)[:, None]
    q = np.asarray(q, float)[:, None]
    sg_ = p * x[None, :] + q
    phi = h(sg_)
    ybar = y.mean()
    lo = math.log(ybar / (1 - ybar)) - A * phi.max(axis=1)
    hi = math.log(ybar / (1 - ybar)) - A * phi.min(axis=1)
    b2 = 0.5 * (lo + hi)
    for _ in range(200):
        z = A * phi + b2[:, None]
        sg = _sigmoid(z)
        g = sg.mean(axis=1) - ybar
        lo = np.where(g < 0, b2, lo)
        hi = np.where(g > 0, b2, hi)
        hh = (sg * (1 - sg)).mean(axis=1)
        newton = b2 - g / np.maximum(hh, 1e-300)
        bad = (newton <= lo) | (newton >= hi) | ~np.isfinite(newton)
        b2n = np.where(bad, 0.5 * (lo + hi), newton)
        done = (np.abs(g) <= 1e-14) | (hi - lo <= 1e-15 * np.maximum(1, np.abs(b2)))
        b2 = np.where(done, b2, b2n)
        if done.all():
            break
    z = A * phi + b2[:, None]
    sg = _sigmoid(z)
    L = (_softplus(z) - y[None, :] * z).mean(axis=1)
    r = (sg - y[None, :]) * A * hp(sg_)
    return L, b2, (r * x[None, :]).mean(axis=1), r.mean(axis=1)


def _h_extrema(lo, hi):
    """Exact min/max of h over [lo, hi] (critical points ±√2)."""
    vals = [h(lo), h(hi)]
    mn = np.minimum(vals[0], vals[1]); mx = np.maximum(vals[0], vals[1])
    for c, is_max in ((-R2, True), (R2, False)):
        inside = (lo <= c) & (c <= hi)
        if is_max:
            mx = np.where(inside, np.maximum(mx, h(c)), mx)
        else:
            mn = np.where(inside, np.minimum(mn, h(c)), mn)
    return mn, mx


def gap0(p, q, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    p = np.asarray(p, float); q = np.asarray(q, float)

    def ext(lo, hi):
        a1, a2 = p * lo + q, p * hi + q
        return _h_extrema(np.minimum(a1, a2), np.maximum(a1, a2))
    _, imax = ext(*inner)
    o1, _ = ext(*outer)
    o2, _ = ext(-outer[1], -outer[0])
    return np.minimum(o1, o2) - imax


def isotonic_logloss(yv):
    """(sum of) log loss of the isotonic (increasing) regression of the 0/1 sequence yv (PAVA)."""
    blocks = []                                   # [sum, count]
    for v in yv:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += c2
    tot = 0.0
    for s_, c in blocks:
        m = s_ / c
        for k, pr in ((s_, m), (c - s_, 1 - m)):
            if k > 0:
                tot -= k * math.log(pr)
    return tot


def monotone_bound(x, y, window):
    """min over window positions of (1/n)·best monotone (either direction) log loss outside it."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(x)
    best = math.inf
    cand = np.unique(np.concatenate([xs - window, xs]))      # window [c, c+window) left edges
    for c in cand:
        keep = ~((xs >= c) & (xs < c + window))
        yk = ys[keep]
        v = min(isotonic_logloss(yk), isotonic_logloss(yk[::-1]))
        best = min(best, v / n)
    return best


def certify(A, x, y, region, P, tol=1e-7, h0=0.25, max_cells=4_000_000, half=True):
    """half=True searches p >= 0 only: valid when the data are x-symmetric, since then
    L0*(p, q) = L0*(-p, q) and G0(p, q) = G0(-p, q) (substitute x -> -x)."""
    Q = 2 * math.sqrt(2) + 2 * P
    p0 = 0.0 if half else -P
    npn = int(math.ceil((P - p0) / h0)); nq = int(math.ceil(2 * Q / h0))
    hpp, hq = (P - p0) / npn / 2, Q / nq
    pc = p0 + (np.arange(npn) + 0.5) * 2 * hpp
    qc = -Q + (np.arange(nq) + 0.5) * 2 * hq
    cp, cq = np.meshgrid(pc, qc, indexing="ij"); cp, cq = cp.ravel(), cq.ravel()
    upper = math.log(2) if region == "-" else math.inf
    arg = (0.0, 0.0) if region == "-" else None
    ax = np.abs(x)
    rounds = 0
    while True:
        rounds += 1
        n = len(cp)
        L = np.empty(n); gp = np.empty(n); gq = np.empty(n); G = np.empty(n)
        curv = np.empty(n); stepG = np.empty(n)
        for i in range(0, n, 2000):
            sl = slice(i, i + 2000)
            L[sl], _, gp[sl], gq[sl] = profile(cp[sl], cq[sl], A, x, y)
            G[sl] = gap0(cp[sl], cq[sl])
            d = ax[None, :] * hpp + hq
            smax = np.abs(cp[sl, None] * x[None, :] + cq[sl, None]) + d
            curv[sl] = 0.5 * A * (smax * d ** 2).mean(axis=1)
            sm = 2 * np.abs(cp[sl]) + np.abs(cq[sl]) + 2 * hpp + hq
            stepG[sl] = (1 + sm ** 2 / 2) * (2.8 * hpp + 2 * hq)
        lb = L - np.abs(gp) * hpp - np.abs(gq) * hq - curv
        inreg = (G <= 0) if region == "-" else (G > 0)
        if inreg.any():
            j = int(np.argmin(np.where(inreg, L, np.inf)))
            if L[j] < upper:
                upper, arg = float(L[j]), (float(cp[j]), float(cq[j]))
        may = (G - stepG <= 0) if region == "-" else (G + stepG > 0)
        keep = may & (lb < upper)
        lower = float(lb[keep].min()) if keep.any() else upper
        conv = upper - lower <= tol or not keep.any()
        if conv or (region == "+" and lower > math.log(2)) or 4 * keep.sum() > max_cells:
            encl = None
            if keep.any():
                kp, kq = cp[keep], cq[keep]
                encl = (float(kp.min() - hpp), float(kp.max() + hpp), float(kq.min() - hq), float(kq.max() + hq))
            return {"encl": encl, "region": region, "A": A, "lower": min(lower, upper), "upper": upper,
                    "arg_p": arg[0] if arg else np.nan, "arg_q": arg[1] if arg else np.nan,
                    "cells": int(keep.sum()), "rounds": rounds, "hp": hpp, "hq": hq,
                    "P": P, "Q": Q, "converged": bool(conv or lower > math.log(2))}
        cp, cq = cp[keep], cq[keep]
        hpp, hq = hpp / 2, hq / 2
        cp = np.concatenate([cp - hpp, cp - hpp, cp + hpp, cp + hpp])
        cq = np.concatenate([cq - hq, cq + hq, cq - hq, cq + hq])


def localisation():
    x, y = population()
    rows = []
    for P in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0):
        rows.append({"P": P, "window": 4 * R2 / P, "B_P": monotone_bound(x, y, 4 * R2 / P)})
        print(rows[-1], flush=True)
    full = min(isotonic_logloss(y[np.argsort(x)]), isotonic_logloss(y[np.argsort(x)][::-1])) / len(x)
    t = pd.DataFrame(rows)
    t["B_full"] = full
    t.to_csv(RESULTS / "limit_localisation.csv", index=False)
    print("B_full", full)


if __name__ == "__main__" and sys.argv[1] == "localisation":
    localisation()


# ------------------------------------------------------------------------- switch and H2'
def _status(A, x, y, P, half=True, tol=1e-7):
    rm = certify(A, x, y, "-", P, tol=tol, half=half)
    rp = certify(A, x, y, "+", P, tol=tol, half=half)
    if rp["lower"] > rm["upper"]:
        st = "minus"
    elif rm["lower"] > rp["upper"]:
        st = "plus"
    else:
        st = "unresolved"
    return st, rm, rp


def solve_margin0(p, q, A, x, y, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    L, b2, *_ = profile([p], [q], A, x, y)

    def ext(lo, hi):
        a1, a2 = p * lo + q, p * hi + q
        return _h_extrema(np.array(min(a1, a2)), np.array(max(a1, a2)))
    _, imax = ext(*inner)
    o1, _ = ext(*outer)
    o2, _ = ext(-outer[1], -outer[0])
    return float(min(-(A * imax + b2[0]), A * min(o1, o2) + b2[0]))


def hessian0(p, q, A, x, y, e=1e-4):
    f = lambda pp, qq: float(profile([pp], [qq], A, x, y)[0][0])
    fpp = (f(p + e, q) - 2 * f(p, q) + f(p - e, q)) / e ** 2
    fqq = (f(p, q + e) - 2 * f(p, q) + f(p, q - e)) / e ** 2
    fpq = (f(p + e, q + e) - f(p + e, q - e) - f(p - e, q + e) + f(p - e, q - e)) / (4 * e ** 2)
    return np.linalg.eigvalsh(np.array([[fpp, fpq], [fpq, fqq]]))


def competitor_gap(A, x, y, P, centre, radius, half=True, tol=1e-6, h0=0.25):
    """Certified min of L0* over K(P) minus the ball of `radius` around `centre` (and its mirror)."""
    Q = 2 * math.sqrt(2) + 2 * P
    p0 = 0.0 if half else -P
    npn = int(math.ceil((P - p0) / h0)); nq = int(math.ceil(2 * Q / h0))
    hpp, hq = (P - p0) / npn / 2, Q / nq
    pc = p0 + (np.arange(npn) + 0.5) * 2 * hpp
    qc = -Q + (np.arange(nq) + 0.5) * 2 * hq
    cp, cq = np.meshgrid(pc, qc, indexing="ij"); cp, cq = cp.ravel(), cq.ravel()
    cs = [centre, (-centre[0], centre[1])]
    ax = np.abs(x)
    upper, rounds = math.inf, 0
    while True:
        rounds += 1
        inside = np.zeros(len(cp), bool)
        for (c0, c1) in cs:     # cell entirely inside a ball (sup-norm corners) -> excluded
            far = np.hypot(np.abs(cp - c0) + hpp, np.abs(cq - c1) + hq)
            inside |= far < radius
        cp, cq = cp[~inside], cq[~inside]
        n = len(cp)
        L = np.empty(n); gp = np.empty(n); gq = np.empty(n); curv = np.empty(n)
        for i in range(0, n, 2000):
            sl = slice(i, i + 2000)
            L[sl], _, gp[sl], gq[sl] = profile(cp[sl], cq[sl], A, x, y)
            d = ax[None, :] * hpp + hq
            smax = np.abs(cp[sl, None] * x[None, :] + cq[sl, None]) + d
            curv[sl] = 0.5 * A * (smax * d ** 2).mean(axis=1)
        lb = L - np.abs(gp) * hpp - np.abs(gq) * hq - curv
        outside = np.ones(n, bool)
        for (c0, c1) in cs:
            outside &= np.hypot(cp - c0, cq - c1) >= radius
        if outside.any():
            upper = min(upper, float(L[outside].min()))
        keep = lb < upper
        lower = float(lb[keep].min()) if keep.any() else upper
        if upper - lower <= tol or not keep.any() or rounds > 30:
            return {"lower": lower, "upper": upper, "rounds": rounds}
        cp, cq = cp[keep], cq[keep]
        hpp, hq = hpp / 2, hq / 2
        cp = np.concatenate([cp - hpp, cp - hpp, cp + hpp, cp + hpp])
        cq = np.concatenate([cq - hq, cq + hq, cq - hq, cq + hq])


def switch(P=24.0, lo=0.60, hi=0.80, width=1e-4, delta=2e-3):
    x, y = population()
    rows = []
    sl, _, _ = _status(lo, x, y, P)
    sh, _, _ = _status(hi, x, y, P)
    assert sl == "minus" and sh == "plus", (sl, sh)
    while hi - lo > width:
        mid = 0.5 * (lo + hi)
        st, rm, rp = _status(mid, x, y, P)
        rows.append({"A": mid, "status": st, "m_minus_lo": rm["lower"], "m_minus_hi": rm["upper"],
                     "m_plus_lo": rp["lower"], "m_plus_hi": rp["upper"],
                     "minus_p": rm["arg_p"], "minus_q": rm["arg_q"], "plus_p": rp["arg_p"],
                     "plus_q": rp["arg_q"], "converged": rm["converged"] and rp["converged"]})
        if st == "plus":
            hi = mid
        elif st == "minus":
            lo = mid
        else:
            break
    A_star = 0.5 * (lo + hi)
    # H2': gap of the global argmin just below and above the switch
    out = {"A_lo": lo, "A_hi": hi, "A_star": A_star, "R_glob_inf_lo": K_LIMIT * lo / 2,
           "R_glob_inf_hi": K_LIMIT * hi / 2}
    pts = {}
    for tag, A in (("minus", A_star - delta), ("plus", A_star + delta)):
        st, rm, rp = _status(A, x, y, P)
        g = rp if st == "plus" else rm
        pts[tag] = (g["arg_p"], g["arg_q"], float(gap0([g["arg_p"]], [g["arg_q"]])[0]), g["upper"])
    out["G_below"], out["G_above"] = pts["minus"][2], pts["plus"][2]
    out["dG_dA"] = (pts["plus"][2] - pts["minus"][2]) / (2 * delta)
    out["argmin_shift"] = math.hypot(pts["plus"][0] - pts["minus"][0], pts["plus"][1] - pts["minus"][1])
    st, rm, rp = _status(A_star, x, y, P)
    g = rp if rp["upper"] <= rm["upper"] else rm
    out["branch_loss"] = min(rm["upper"], rp["upper"])
    out["argmin_p"], out["argmin_q"] = g["arg_p"], g["arg_q"]
    ev = hessian0(g["arg_p"], g["arg_q"], A_star, x, y)
    out["hess_min"], out["hess_max"] = float(ev[0]), float(ev[1])
    cg = competitor_gap(A_star, x, y, P, (g["arg_p"], g["arg_q"]), radius=0.5)
    out["competitor_lower"] = cg["lower"]
    out["competitor_margin"] = cg["lower"] - out["branch_loss"]
    out["B_P"] = monotone_bound(x, y, 4 * R2 / P)
    out["localised"] = out["branch_loss"] < min(out["B_P"], 0.47738562622110964)
    pd.DataFrame(rows).to_csv(RESULTS / "limit_switch_evaluations.csv", index=False)
    pd.DataFrame([out]).to_csv(RESULTS / "limit_switch.csv", index=False)
    print(pd.Series(out).to_string())


if __name__ == "__main__" and sys.argv[1] == "switch":
    switch()
