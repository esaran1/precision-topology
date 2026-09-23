"""Blocks 1b/1c: profiled conditional loss and a certified branch-and-bound over placements.

Registered in results/conditional_audit_prediction.md (6b17508).

Setting.  N(x) = s·f_a(w1 x + b1) + b2, s = w2 > 0 fixed (the symmetry (w1,b1,w2) -> (-w1,-b1,-w2)
covers w2 < 0), labels y in {0,1}, loss L = mean(softplus(z) - y z), z = N(x).

1b  Profiled bias.  ∂L/∂b2 = mean(σ(z) - y) is strictly increasing in b2 when both classes are
    present, so b2* is the unique root.  Bracket: b2_lo = logit(ȳ) - s·max φ, b2_hi = logit(ȳ) - s·min φ
    (at b2_lo every σ(z) ≤ ȳ, at b2_hi every σ(z) ≥ ȳ); safeguarded Newton, bisection fallback,
    to |mean(σ(z) - y)| ≤ 1e-14.  L*(w1, b1; s) = L at b2*.

1c  Bounds used by the branch and bound (derived in conditional_audit.md):
    gradient (envelope theorem): ∂L*/∂w1 = mean((σ-y) s f'(t) x), ∂L*/∂b1 = mean((σ-y) s f'(t)).
    curvature: for any direction δ = (δw, δb), δᵀ∇²L* δ ≥ -s·a·mean((|x||δw| + |δb|)²), because the
      Schur-complement term (∂²_{θb}L·δ)²/∂²_{bb}L is dominated by the σ'-weighted Gauss-Newton term
      (Cauchy-Schwarz) and the remaining term is mean((σ-y) s f''(t)(xδw+δb)²) with |f''| ≤ a.
    Hence on a sup-norm cell with centre c and half-widths (hw, hb):
      L* ≥ L*(c) - |∂w L*|hw - |∂b L*|hb - ½ s a mean((|x|hw + hb)²).
    class gap (w2 > 0): G = min_O φ - max_I φ, computed EXACTLY from interval extrema of f_a;
      |ΔG| ≤ (1+a)(2.8 hw + 2 hb) on the cell.
    domain: b1 ∈ [0, 2π) (L* and G are 2π-periodic in b1); |w1| ≤ W(s, a) where
      W = min over splits c of [2a + 2 log(2^{1/π_c} - 1)/s] / (1.2 + c), π_c = the smaller class
      fraction of the two opposing groups (left-outer vs inner x ≥ c for w1 > 0; mirror for w1 < 0);
      beyond W, L* > log 2 ≥ m₋.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2 * math.pi
TOL_NEWTON = 1e-14
CHUNK = 2000          # cells per vectorised batch: ~13 MB per (cells x points) array


def f_a(t, a):
    return t + a * np.sin(t)


def _softplus(z):
    return np.logaddexp(0.0, z)


def _sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def profile(w1, b1, s, a, x, y):
    """Vectorised over cells: returns L*, b2*, gradient (gw, gb), and max |mean(σ-y)| residual."""
    w1 = np.asarray(w1, float)[:, None]
    b1 = np.asarray(b1, float)[:, None]
    t = w1 * x[None, :] + b1
    phi = f_a(t, a)
    ybar = y.mean()
    lo = math.log(ybar / (1 - ybar)) - s * phi.max(axis=1)
    hi = math.log(ybar / (1 - ybar)) - s * phi.min(axis=1)
    b2 = 0.5 * (lo + hi)
    for _ in range(200):
        z = s * phi + b2[:, None]
        sg = _sigmoid(z)
        g = sg.mean(axis=1) - ybar
        lo = np.where(g < 0, b2, lo)
        hi = np.where(g > 0, b2, hi)
        h = (sg * (1 - sg)).mean(axis=1)
        newton = b2 - g / np.maximum(h, 1e-300)
        bad = (newton <= lo) | (newton >= hi) | ~np.isfinite(newton)
        b2_new = np.where(bad, 0.5 * (lo + hi), newton)
        done = (np.abs(g) <= TOL_NEWTON) | (hi - lo <= 1e-15 * np.maximum(1, np.abs(b2)))
        b2 = np.where(done, b2, b2_new)
        if done.all():
            break
    z = s * phi + b2[:, None]
    sg = _sigmoid(z)
    resid = np.abs(sg.mean(axis=1) - ybar).max()
    L = (_softplus(z) - y[None, :] * z).mean(axis=1)
    fp = 1 + a * np.cos(t)
    r = (sg - y[None, :]) * s * fp
    gw = (r * x[None, :]).mean(axis=1)
    gb = r.mean(axis=1)
    return L, b2, gw, gb, resid


def _interval_extrema(tlo, thi, a):
    """min and max of f_a over [tlo, thi], exactly (endpoints + interior critical points)."""
    c = math.acos(-1.0 / a) if a > 1 else None
    vlo, vhi = f_a(tlo, a), f_a(thi, a)
    mx = np.maximum(vlo, vhi)
    mn = np.minimum(vlo, vhi)
    if c is not None:
        k = np.floor((thi - c) / TWO_PI)                 # largest local max  t = c + 2πk <= thi
        tm = c + TWO_PI * k
        mx = np.where(tm >= tlo, np.maximum(mx, f_a(tm, a)), mx)
        k = np.ceil((tlo + c) / TWO_PI)                  # smallest local min t = -c + 2πk >= tlo
        tn = -c + TWO_PI * k
        mn = np.where(tn <= thi, np.minimum(mn, f_a(tn, a)), mn)
    return mn, mx


def gap(w1, b1, a, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    """Exact one-sided class gap for w2 > 0: min over O of φ minus max over I of φ."""
    w1 = np.asarray(w1, float)
    b1 = np.asarray(b1, float)

    def ext(lo, hi):
        t1, t2 = w1 * lo + b1, w1 * hi + b1
        return _interval_extrema(np.minimum(t1, t2), np.maximum(t1, t2), a)
    _, imax = ext(*inner)
    omin1, _ = ext(outer[0], outer[1])
    omin2, _ = ext(-outer[1], -outer[0])
    return np.minimum(omin1, omin2) - imax


def w_bound(s, a, x, y, win=(-0.8, 0.8, 1.2, 2.0)):
    """|w1| beyond which L* > log 2, from the linear-growth argument.

    w1 > 0: left-outer points (x <= -1.2, class 1) have logit <= s(-1.2 w1 + b1 + a) + b2 and inner
    points with x >= c (class 0) have logit >= s(c w1 + b1 - a) + b2, a separation of
    Δ = s((1.2 + c) w1 - 2a) in the wrong direction; for any b2 one of the two groups is wrong by at
    least Δ/2, so L* >= (m/n) softplus(Δ/2), m = the smaller group's count.  w1 < 0 mirrors this with
    right-outer points and inner x <= -c.  W = the larger (worse) side, each side minimised over c.
    """
    i_lo, i_hi, o_lo, o_hi = win
    n = len(x)
    out = []
    for sign in (+1, -1):
        # opposing groups: outer points on the far side (sign*x <= -o_lo) vs inner points with sign*x >= c
        n_outer = int(((sign * x <= -o_lo) & (y == 1)).sum())
        best = math.inf
        top = i_hi if sign > 0 else -i_lo
        for c in np.linspace(min(0.0, top - 1e-3), top - 1e-3, 80):
            n_inner = int(((sign * x >= c) & (y == 0)).sum())
            m = min(n_outer, n_inner)
            if m == 0 or o_lo + c <= 0:
                continue
            pi = m / n
            delta = 2 * math.log(2 ** (1 / pi) - 1) if 1 / pi < 1000 else 2 * (1 / pi) * math.log(2)
            best = min(best, (2 * a + delta / s) / (o_lo + c))
        out.append(best)
    return max(out)


def certify(s, a, x, y, region, tol=1e-7, h0=0.05, max_cells=4_000_000, W=None,
            win=(-0.8, 0.8, 1.2, 2.0)):
    """Certified min of L* over {G <= 0} (region='-') or {G > 0} (region='+').

    Returns dict with lower/upper bounds, argmin, and whether the bound is above log 2.
    """
    if W is None:
        W = w_bound(s, a, x, y, win)
    i_lo, i_hi, o_lo, o_hi = win
    xmax = max(abs(i_lo), abs(i_hi), o_hi)
    nw = int(math.ceil(2 * W / h0))
    nb = int(math.ceil(TWO_PI / h0))
    hw, hb = W / nw, TWO_PI / nb / 2
    hw = 2 * W / nw / 2
    wc = -W + (np.arange(nw) + 0.5) * 2 * hw
    bc = (np.arange(nb) + 0.5) * 2 * hb
    cw, cb = np.meshgrid(wc, bc, indexing="ij")
    cw, cb = cw.ravel(), cb.ravel()
    upper = math.log(2) if region == "-" else math.inf   # constant predictor: G = 0, loss log 2
    arg = (0.0, 0.0) if region == "-" else None
    ax = np.abs(x)
    rounds = 0
    while True:
        rounds += 1
        L = np.empty(len(cw)); gw = np.empty(len(cw)); gb = np.empty(len(cw)); G = np.empty(len(cw))
        for i in range(0, len(cw), CHUNK):
            sl = slice(i, i + CHUNK)
            L[sl], _, gw[sl], gb[sl], _ = profile(cw[sl], cb[sl], s, a, x, y)
            G[sl] = gap(cw[sl], cb[sl], a, inner=(i_lo, i_hi), outer=(o_lo, o_hi))
        stepG = (1 + a) * ((max(abs(i_lo), abs(i_hi)) + o_hi) * hw + 2 * hb)
        curv = 0.5 * s * a * np.mean((ax * hw + hb) ** 2)
        lb = L - np.abs(gw) * hw - np.abs(gb) * hb - curv
        inreg_c = (G <= 0) if region == "-" else (G > 0)
        if inreg_c.any():
            j = int(np.argmin(np.where(inreg_c, L, np.inf)))
            if L[j] < upper:
                upper, arg = float(L[j]), (float(cw[j]), float(cb[j]))
        may = (G - stepG <= 0) if region == "-" else (G + stepG > 0)
        keep = may & (lb < upper)
        lower = float(lb[keep].min()) if keep.any() else upper
        if upper - lower <= tol or not keep.any():
            return {"region": region, "s": s, "a": a, "lower": min(lower, upper), "upper": upper,
                    "arg_w1": arg[0] if arg else np.nan, "arg_b1": arg[1] if arg else np.nan,
                    "cells": int(keep.sum()), "rounds": rounds, "hw": hw, "hb": hb, "W": W,
                    "above_log2": lower > math.log(2), "converged": True}
        if lower > math.log(2) and region == "+":
            return {"region": region, "s": s, "a": a, "lower": lower, "upper": upper,
                    "arg_w1": arg[0] if arg else np.nan, "arg_b1": arg[1] if arg else np.nan,
                    "cells": int(keep.sum()), "rounds": rounds, "hw": hw, "hb": hb, "W": W,
                    "above_log2": True, "converged": True}
        cw, cb = cw[keep], cb[keep]
        if 4 * len(cw) > max_cells:
            return {"region": region, "s": s, "a": a, "lower": lower, "upper": upper,
                    "arg_w1": arg[0] if arg else np.nan, "arg_b1": arg[1] if arg else np.nan,
                    "cells": int(len(cw)), "rounds": rounds, "hw": hw, "hb": hb, "W": W,
                    "above_log2": lower > math.log(2), "converged": False}
        hw, hb = hw / 2, hb / 2
        cw = np.concatenate([cw - hw, cw - hw, cw + hw, cw + hw])
        cb = np.concatenate([cb - hb, cb + hb, cb - hb, cb + hb])
