"""Redesigned uniqueness certificates for the limit problem: math_note_v2 (c) and the solve bracket.

The first design (math_note_v2_checks.annulus / solve_competitor) could not close. Its branch upper bound carried
a slack h_A·mean|h| larger than the loss rise at the edge of the 0.05 box (7.2e-3 against 3.3e-4 for the annulus).
The chain here holds for each A-sub-interval [A_lo, A_hi]. A is carried as an interval and never sampled:

  (1) LOCALISATION: every global minimiser lies in K(24)
      (existing: math_note_v2 (a); here re-checked as branch_upper < B(24)).
  (2) OUTER: L0*(θ; A) > L0*(θ_c; A) for every θ ∈ K(24) outside the 0.15 box around θ_c and every A in the
      sub-interval, by a certified margin                          -> outer_exclusion.
  (3) RING: ∇_{p,q} L0*(·; A) ≠ 0 on the ring 0.05 <= |θ − θ_c|_∞ <= 0.15, for every A in the sub-interval.
      An interval enclosure of the gradient excludes 0 on every box that covers the ring  -> ring_no_critical.
  (4) PD: the (p, q, b) Hessian is positive definite on the 0.05 box × the A range
      (existing: mn2_neighbourhood.csv; for the solve bracket, pd_boxes_solve).
  => A global minimiser exists in K(24) (1). It lies in the 0.15 box (2) and is a critical point, so it is not in
     the ring (3). It is therefore in the 0.05 box, where the critical point is unique (4). So the global minimiser
     is unique up to the reflection p -> −p (data and windows are x-symmetric; half-plane p >= 0 as in limit_bnb).

Interval arithmetic uses numpy arrays of (lo, hi) with outward rounding (np.nextafter) after every operation.
exp, pow(·, 3) and the logistic are widened by >= 4 ulps (libm is accurate to < 1 ulp). The profiled bias is
enclosed by certified sign changes of monotone bounds of F(b) = mean σ(A h(σ) + b) − ȳ. The outer exclusion is
the float branch and bound of limit_bnb.certify (same cell bound, math_note_v2 §3(b)), plus the A-term and a
SAFETY margin for float rounding in the profiled evaluations.

Every run has a cap (MAX_CELLS, MAX_RING_BOXES, RING_MIN_SIDE): it stops and reports "not closed" instead of
growing. Each finished A-sub-interval is appended to a parts file and skipped on restart.

    python -m src.certificates_v2 annulus [workers]     # A ∈ [0.66, 0.71]: 40 sub-intervals, failures halved
    python -m src.certificates_v2 solve [workers]       # the solve bracket (1.05875, 1.06]
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
RHO_IN, RHO_OUT = 0.05, 0.15
P_BOX = 24.0
B24 = 0.38797358308678997649               # math_note_v2 (a), mn2_bounds.csv (exact PAVA, 50 digits)
MAX_CELLS = 1_500_000                       # outer branch and bound: live cells after a split
MAX_RING_BOXES = 400_000                    # ring: boxes evaluated in total
RING_MIN_SIDE = 1.0 / 2048
PAD = 1e-12                                 # tiling overlap: ring boxes, outer and PD regions overlap by >= PAD
SAFETY = 1e-9                               # float rounding in profile(): L, gradient, b-tolerance (<< margins)
MARGIN = 1e-4                               # outer: every cell's lower bound must exceed branch_upper + MARGIN
N_SUB = 40
MAX_HALVINGS = 3
INF = np.inf


# ------------------------------------------------------------------------------------------ interval arithmetic
def _dn(v):
    return np.nextafter(v, -INF)


def _up(v):
    return np.nextafter(v, INF)


def _widen(lo, hi, k):
    for _ in range(k):
        lo, hi = _dn(lo), _up(hi)
    return lo, hi


class Iv:
    """Vectorised closed intervals [lo, hi], outward rounded."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(lo if hi is None else hi, float)

    def __add__(self, o):
        o = o if isinstance(o, Iv) else Iv(o)
        return Iv(_dn(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Iv) else Iv(o)))

    def __rsub__(self, o):
        return Iv(o) - self

    def __mul__(self, o):
        o = o if isinstance(o, Iv) else Iv(o)
        c = np.stack(np.broadcast_arrays(self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi))
        return Iv(_dn(c.min(axis=0)), _up(c.max(axis=0)))

    __rmul__ = __mul__

    def sq(self):
        lo2, hi2 = self.lo * self.lo, self.hi * self.hi
        mn = np.where((self.lo <= 0) & (self.hi >= 0), 0.0, _dn(np.minimum(lo2, hi2)))
        return Iv(np.maximum(mn, 0.0), _up(np.maximum(lo2, hi2)))

    def cube(self):
        """t ↦ t³ is increasing; np.power is within 1 ulp, widened by 4."""
        return Iv(*_widen(np.power(self.lo, 3), np.power(self.hi, 3), 4))

    def div(self, c):
        """Division by an exact positive float c."""
        return Iv(_dn(self.lo / c), _up(self.hi / c))

    def contains_zero(self):
        return (self.lo <= 0) & (self.hi >= 0)


def sigmoid_iv(z):
    """σ is increasing; 1/(1 + e^{−z}) at the ends (<= 3 ulps of error), widened by 8 ulps, clipped to [0, 1]."""
    with np.errstate(over="ignore"):
        lo = 1.0 / (1.0 + np.exp(-z.lo))
        hi = 1.0 / (1.0 + np.exp(-z.hi))
    lo, hi = _widen(lo, hi, 8)
    return Iv(np.maximum(lo, 0.0), np.minimum(hi, 1.0))


def h_iv(s):
    return s.cube().div(6.0) - s


def hp_iv(s):
    return s.sq().div(2.0) - 1.0


def mean_iv(v, axis=-1):
    """Rigorous mean: recursive summation error <= (n − 1)·u·Σ|terms| (u = 2^-53), bounded by n·2.3e-16·Σ|terms|."""
    n = v.lo.shape[axis]
    mag = np.maximum(np.abs(v.lo), np.abs(v.hi)).sum(axis=axis)
    err = _up(n * 2.3e-16 * mag)
    lo = _dn(_dn(v.lo.sum(axis=axis) - err) / n)
    hi = _up(_up(v.hi.sum(axis=axis) + err) / n)
    return Iv(lo, hi)


def ybar_iv(y):
    k, n = int(round(float(np.sum(y)))), len(y)
    assert np.array_equal(y, y.astype(bool).astype(float)) and k == int(y.sum())
    return Iv(_dn(k / n), _up(k / n))


def sigma_iv(P, Q, x):
    """σ = p x + q over boxes P × Q (length m) at the points x (length n) -> (m, n)."""
    a, b = P.lo[:, None] * x[None, :], P.hi[:, None] * x[None, :]
    return Iv(_dn(np.minimum(a, b)), _up(np.maximum(a, b))) + Iv(Q.lo[:, None], Q.hi[:, None])


# ------------------------------------------------------------------------------------------ profiled bias enclosure
def b_enclosure(Z0, yb):
    """Z0: (m, n) enclosure of A·h(σ_i) over each box (b not included). Returns B (length m) containing every
    profiled b*(θ, A) for θ, A in the box. F_lo(b) = mean σ(Z0.lo + b) − ȳ <= F(b) <= F_hi(b) = mean σ(Z0.hi + b) − ȳ,
    all increasing in b. F_hi(β) < 0 (certified) gives b* > β; F_lo(γ) > 0 (certified) gives b* < γ."""
    y0 = 0.5 * (yb.lo + yb.hi)

    def root(z):                                          # float candidate only
        lo = np.full(len(z), -60.0); hi = np.full(len(z), 60.0)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            with np.errstate(over="ignore"):
                v = (1.0 / (1.0 + np.exp(-(z + mid[:, None])))).mean(axis=1) - y0
            lo, hi = np.where(v < 0, mid, lo), np.where(v < 0, hi, mid)
        return 0.5 * (lo + hi)

    def F(z, b):
        return mean_iv(sigmoid_iv(Iv(z) + Iv(b[:, None])), axis=1) - yb

    beta = root(Z0.hi) - 1e-9
    gamma = root(Z0.lo) + 1e-9
    ok = (F(Z0.hi, beta).hi < 0) & (F(Z0.lo, gamma).lo > 0)
    return Iv(beta, gamma), ok


# ------------------------------------------------------------------------------------------ (3) ring: no critical point
def grad_enclosure(P, Q, A, x, y):
    """Enclosures of ∂_p L0* and ∂_q L0* over boxes P × Q × A (Iv of length m), b profiled (envelope theorem):
    ∂_p = mean((σ(z) − y)·A h′(σ)·x), ∂_q = mean((σ(z) − y)·A h′(σ)), z = A h(σ) + b*."""
    S = sigma_iv(P, Q, x)
    Ab = Iv(A.lo[:, None], A.hi[:, None])
    Z0 = Ab * h_iv(S)
    B, ok = b_enclosure(Z0, ybar_iv(y))
    Rz = sigmoid_iv(Z0 + Iv(B.lo[:, None], B.hi[:, None])) - Iv(y[None, :])
    D = Rz * (Ab * hp_iv(S))
    gq = mean_iv(D, axis=1)
    gp = mean_iv(D * Iv(np.broadcast_to(x[None, :], D.lo.shape)), axis=1)
    return gp, gq, ok


def ring_no_critical(A_lo, A_hi, centre, x, y, start_side=0.0125, chunk=64, rho_in=RHO_IN, rho_out=RHO_OUT,
                     max_boxes=MAX_RING_BOXES, min_side=RING_MIN_SIDE):
    """Cover the ring rho_in <= |θ − centre|_∞ <= rho_out with boxes (each padded by PAD). A box is certified when
    the profiled bias is enclosed and 0 ∉ ∂_p or 0 ∉ ∂_q over it, for every A in [A_lo, A_hi]. Uncertified boxes are
    split in four. A box is dropped only if it lies inside the (rho_in − PAD) box, which the PD certificate covers.
    Stops and reports at max_boxes or min_side."""
    p0, q0 = centre
    n = int(round(2 * rho_out / start_side))
    c = -rho_out + (np.arange(n) + 0.5) * start_side
    cp, cq = np.meshgrid(c, c, indexing="ij"); cp, cq = cp.ravel(), cq.ravel()
    half = start_side / 2
    total, depth = 0, 0
    gmin = math.inf
    while True:
        inner = (np.abs(cp) + half <= rho_in - PAD) & (np.abs(cq) + half <= rho_in - PAD)
        cp, cq = cp[~inner], cq[~inner]
        fail = np.zeros(len(cp), bool)
        for i in range(0, len(cp), chunk):
            sl = slice(i, i + chunk)
            m = len(cp[sl])
            P = Iv(_dn(p0 + cp[sl] - half - PAD), _up(p0 + cp[sl] + half + PAD))
            Q = Iv(_dn(q0 + cq[sl] - half - PAD), _up(q0 + cq[sl] + half + PAD))
            A = Iv(np.full(m, float(A_lo)), np.full(m, float(A_hi)))
            gp, gq, ok = grad_enclosure(P, Q, A, x, y)
            f = ~ok | (gp.contains_zero() & gq.contains_zero())
            fail[sl] = f
            mag = np.maximum(np.where(gp.contains_zero(), 0, np.minimum(np.abs(gp.lo), np.abs(gp.hi))),
                             np.where(gq.contains_zero(), 0, np.minimum(np.abs(gq.lo), np.abs(gq.hi))))
            if (~f).any():
                gmin = min(gmin, float(mag[~f].min()))
        total += len(cp)
        cp, cq = cp[fail], cq[fail]
        res = {"ring_boxes": total, "ring_depth": depth, "ring_final_side": 2 * half, "ring_grad_lower": gmin}
        if not len(cp):
            return {"ring_certified": True, **res, "ring_left": 0}
        if half < min_side or total + 4 * len(cp) > max_boxes:
            return {"ring_certified": False, **res, "ring_left": int(len(cp)),
                    "ring_note": "min side reached" if half < min_side else f"box cap {max_boxes} reached"}
        half /= 2; depth += 1
        cp = np.concatenate([cp - half, cp - half, cp + half, cp + half])
        cq = np.concatenate([cq - half, cq + half, cq - half, cq + half])


# ------------------------------------------------------------------------------------------ (2) outer exclusion
def branch_upper(A_lo, A_hi, centre, x, y):
    """max_{A ∈ [A_lo, A_hi]} L0*(θ_c; A) = max of the endpoint values (L0*(θ_c; ·) is convex in A). profile() returns
    the loss at a float bias: an upper bound of the minimum over b, up to rounding (SAFETY)."""
    from .limit_bnb import profile
    return max(float(profile([centre[0]], [centre[1]], A_lo, x, y)[0][0]),
               float(profile([centre[0]], [centre[1]], A_hi, x, y)[0][0])) + SAFETY


def outer_exclusion(A_lo, A_hi, centre, x, y, P=P_BOX, h0=0.25, rho=RHO_OUT, max_cells=MAX_CELLS, tol=1e-7,
                    margin=MARGIN):
    """Lower bound of L0*(θ; A) over K(P) ∩ {p >= 0} minus the rho box around θ_c, uniformly for A ∈ [A_lo, A_hi].
    A-term: L0*(θ; ·) is convex, so L0*(θ; A) >= L0*(θ; A_c) − h_A·|∂_A L0*(θ; A_c)|. By the envelope theorem
    ∂_A L0* = mean((σ(z_i) − y_i)·h(σ_i)), and mean(σ(z_i) − y_i) = 0 at the profiled b*, so for ANY constant c
        |∂_A L0*| = |mean((σ(z_i) − y_i)(h(σ_i) − c))| <= mean|h(σ_i) − c|
                  <= mean(|h(σ_i(θ_cell)) − c| + (1 + σmax_i²/2)·d_i),   d_i = |x_i|h_p + h_q,
    with c = the median of h(σ_i) at the cell centre. (The cruder bound mean|h(σ)| is ~10^4 near p = 0, |q| ~ 36,
    where h(σ) is nearly constant across points and is absorbed by b; it stalls the search.)
    Cell bound at A_c (math_note_v2 §3(b), curvature at A_hi):
        L(c) − |∂_p|h_p − |∂_q|h_q − ½A_hi·mean(σmax·d²) − h_A·(A-term) − SAFETY.
    Certified when every cell's bound exceeds branch_upper + margin (so the certified margin is >= margin).
    Stops if a point outside the box comes within margin + tol of the branch value, or at max_cells."""
    from .limit_bnb import h, profile
    p0, q0 = centre
    Ac, hA = 0.5 * (A_lo + A_hi), 0.5 * (A_hi - A_lo)
    Qb = 2 * math.sqrt(2) + 2 * P
    npn, nq = int(math.ceil(P / h0)), int(math.ceil(2 * Qb / h0))
    hp, hq = P / npn / 2, Qb / nq
    cp, cq = np.meshgrid((np.arange(npn) + 0.5) * 2 * hp, -Qb + (np.arange(nq) + 0.5) * 2 * hq, indexing="ij")
    cp, cq = cp.ravel(), cq.ravel()
    ax = np.abs(x)
    bu = branch_upper(A_lo, A_hi, centre, x, y)
    attained, rounds, peak, disc = math.inf, 0, len(cp), math.inf        # disc: min bound over discarded cells
    base = {"branch_upper": bu, "localised": bool(bu < B24), "outer_P": P}
    while True:
        rounds += 1
        inbox = (np.abs(cp - p0) + hp <= rho - PAD) & (np.abs(cq - q0) + hq <= rho - PAD)
        cp, cq = cp[~inbox], cq[~inbox]
        n = len(cp)
        lb = np.empty(n); Lv = np.empty(n)
        for i in range(0, n, 2000):
            sl = slice(i, i + 2000)
            L, _, gp, gq = profile(cp[sl], cq[sl], Ac, x, y)
            d = ax[None, :] * hp + hq
            smax = np.abs(cp[sl, None] * x[None, :] + cq[sl, None]) + d
            curv = 0.5 * A_hi * (smax * d ** 2).mean(axis=1)
            hc = h(cp[sl, None] * x[None, :] + cq[sl, None])
            dev = (np.abs(hc - np.median(hc, axis=1, keepdims=True)) + (1 + smax ** 2 / 2) * d).mean(axis=1)
            lb[sl] = L - np.abs(gp) * hp - np.abs(gq) * hq - curv - hA * dev - SAFETY
            Lv[sl] = L
        outside = (np.abs(cp - p0) > rho) | (np.abs(cq - q0) > rho)
        if outside.any():
            attained = min(attained, float(Lv[outside].min()))
        keep = lb <= bu + margin
        if (~keep).any():
            disc = min(disc, float(lb[~keep].min()))
        info = {**base, "outer_lower": disc, "outer_attained": attained, "outer_rounds": rounds,
                "outer_peak_cells": peak, "outer_final_h": max(hp, hq)}
        if not keep.any():
            return {"outer_certified": True, "outer_margin": disc - bu, **info}
        if attained <= bu + margin + tol:
            return {"outer_certified": False, "outer_margin": np.nan, **info,
                    "outer_note": "a point outside the box is within margin + tol of the branch value"}
        cp, cq = cp[keep], cq[keep]
        if 4 * len(cp) > max_cells:
            return {"outer_certified": False, "outer_margin": np.nan, **info,
                    "outer_note": f"cell cap {max_cells} reached"}
        hp, hq = hp / 2, hq / 2
        cp = np.concatenate([cp - hp, cp - hp, cp + hp, cp + hp])
        cq = np.concatenate([cq - hq, cq + hq, cq - hq, cq + hq])
        peak = max(peak, len(cp))


# ------------------------------------------------------------------------------------------ drivers
def _job(args):
    from .limit_bnb import population
    tag, A_lo, A_hi, centre = args
    x, y = population()
    out = {"tag": tag, "A_lo": A_lo, "A_hi": A_hi, "centre_p": centre[0], "centre_q": centre[1]}
    out.update(ring_no_critical(A_lo, A_hi, centre, x, y))
    if out["ring_certified"]:                             # the outer search is the expensive half
        out.update(outer_exclusion(A_lo, A_hi, centre, x, y))
    else:
        out.update({"outer_certified": False, "outer_note": "skipped: ring not certified"})
    out["certified"] = bool(out["ring_certified"] and out["outer_certified"] and out.get("localised", False))
    return out


def _done(d, lo, hi):
    return (not d.empty) and bool(((d.A_lo == lo) & (d.A_hi == hi)).any())


def _leaves(d, lo, hi):
    """The partition leaves under [lo, hi]: a row whose halves were both evaluated is replaced by its halves."""
    mid = 0.5 * (lo + hi)
    if _done(d, lo, mid) and _done(d, mid, hi):
        return _leaves(d, lo, mid) + _leaves(d, mid, hi)
    return [(lo, hi)]


def _leaves_to_split(d, lo, hi):
    out = []
    for a, b in _leaves(d, lo, hi):
        r = d[(d.A_lo == a) & (d.A_hi == b)]
        if len(r) and not bool(r.certified.iloc[-1]):
            m = 0.5 * (a + b)
            out += [(a, m), (m, b)]
    return out


def _covered(d, intervals):
    """True iff every leaf of every top-level interval is certified (the leaves tile [lo, hi] exactly)."""
    for lo, hi in intervals:
        for a, b in _leaves(d, lo, hi):
            r = d[(d.A_lo == a) & (d.A_hi == b)]
            if not len(r) or not bool(r.certified.iloc[-1]):
                return False
    return True


def _run(tag, intervals, centre, workers, parts_name):
    """Checkpoint each finished A-sub-interval to the parts file (skipped on restart). An uncertified leaf is
    halved, up to MAX_HALVINGS times; coverage is judged on the leaves (_covered)."""
    parts = RESULTS / parts_name

    def read():
        return pd.read_csv(parts, float_precision="round_trip") if parts.exists() else pd.DataFrame()

    todo = [(lo, hi) for lo, hi in intervals if not _done(read(), lo, hi)]
    for level in range(MAX_HALVINGS + 1):
        if todo:
            with Pool(workers) as p:
                for r in p.imap_unordered(_job, [(tag, lo, hi, centre) for lo, hi in todo], chunksize=1):
                    pd.DataFrame([r]).to_csv(parts, mode="a", header=not parts.exists(), index=False)
                    print(json.dumps({k: r.get(k) for k in ("A_lo", "A_hi", "certified", "ring_certified",
                                                            "outer_certified", "outer_margin", "ring_grad_lower")}),
                          flush=True)
        if level == MAX_HALVINGS:
            break
        d = read()
        todo = [iv_ for lo, hi in intervals for iv_ in _leaves_to_split(d, lo, hi) if not _done(d, *iv_)]
    return read()


def _summary(d, intervals, extra, name):
    c = d[d.certified.astype(bool)]
    s = {"range": f"[{intervals[0][0]}, {intervals[-1][1]}]", "covered": _covered(d, intervals),
         "sub_intervals_evaluated": len(d), "sub_intervals_certified": len(c),
         "min_outer_margin": float(c.outer_margin.min()) if len(c) else np.nan,
         "min_ring_grad_lower": float(c.ring_grad_lower.min()) if len(c) else np.nan,
         "max_branch_upper": float(c.branch_upper.max()) if len(c) else np.nan, "B24": B24,
         "max_outer_peak_cells": int(d.outer_peak_cells.max()) if "outer_peak_cells" in d else 0,
         "max_ring_boxes": int(d.ring_boxes.max()), **extra}
    pd.DataFrame([s]).to_csv(RESULTS / name, index=False)
    print(pd.Series(s).to_string())


def annulus(workers=1):
    ls = pd.read_csv(RESULTS / "limit_switch.csv", float_precision="round_trip").iloc[0]
    nb = pd.read_csv(RESULTS / "mn2_neighbourhood.csv", float_precision="round_trip").iloc[0]
    centre = (float(ls.argmin_p), float(ls.argmin_q))
    assert (nb.p0, nb.q0, nb.rho) == (centre[0], centre[1], RHO_IN)        # the PD certificate's box
    edges = np.linspace(0.66, 0.71, N_SUB + 1)
    intervals = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    d = _run("annulus", intervals, centre, workers, "certv2_annulus_parts.csv")
    _summary(d, intervals, {"pd_source": "mn2_neighbourhood.csv", "pd_all": bool(nb.hess_pd_all_boxes),
                            "pd_b2_all": bool(nb.b2_validated_all),
                            "pd_lambda_min_lower": float(nb.hess_lambda_min_lower_bound)},
             "certv2_annulus_summary.csv")


def _solve_setup():
    sl = pd.read_csv(RESULTS / "mn2_solve_limit.csv", float_precision="round_trip").sort_values("A")
    e = ast.literal_eval(sl.encl.iloc[-1])
    return (0.5 * (e[0] + e[1]), 0.5 * (e[2] + e[3])), (float(sl.A_solve_lo.iloc[0]), float(sl.A_solve_hi.iloc[0]))


def pd_boxes_solve(workers=1):
    """(p, q, b) Hessian PD on the 0.05 box around the solve centre × the whole bracket (A as an interval)."""
    from .math_note_v2_checks import SPLIT, _box_hessian_pd
    (p0, q0), (A_lo, A_hi) = _solve_setup()
    out = RESULTS / "certv2_solve_pd.csv"
    if out.exists():
        return pd.read_csv(out, float_precision="round_trip").iloc[0].to_dict()
    pe = np.linspace(p0 - RHO_IN, p0 + RHO_IN, SPLIT[0] + 1)
    qe = np.linspace(q0 - RHO_IN, q0 + RHO_IN, SPLIT[1] + 1)
    boxes = [(pe[i], pe[i + 1], qe[j], qe[j + 1], A_lo, A_hi) for i in range(SPLIT[0]) for j in range(SPLIT[1])]
    with Pool(workers) as p:
        res = list(p.imap(_box_hessian_pd, boxes, chunksize=4))
    r = {"pd_boxes": len(boxes), "pd_all": all(x["pd"] for x in res), "pd_b2_all": all(x["b2_ok"] for x in res),
         "pd_lambda_min_lower": float(np.nanmin([x["lam_lo"] for x in res])),
         "pd_p_lo": pe[0], "pd_p_hi": pe[-1], "pd_q_lo": qe[0], "pd_q_hi": qe[-1]}
    pd.DataFrame([r]).to_csv(out, index=False)
    return r


def solve(workers=1):
    centre, (A_lo, A_hi) = _solve_setup()
    pdres = pd_boxes_solve(workers)
    d = _run("solve", [(A_lo, A_hi)], centre, workers, "certv2_solve_parts.csv")
    _summary(d, [(A_lo, A_hi)], {"pd_source": "certv2_solve_pd.csv", **pdres}, "certv2_solve_summary.csv")


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("CV2_WORKERS", "1"))
    {"annulus": lambda: annulus(w), "solve": lambda: solve(w)}[cmd]()
