"""§20 (Track U, step U1): the rescaled conditional loss at a = 1 + ε to second order, a rigorous O(ε²) remainder, the
certified limit margin near the switch, and an explicit ε₀ below which the branch is the global conditional minimiser at
its switch.  Theory + certified constants only (no training, no seeds).

Notation (math note §1, §16): u = √ε σ, φ_ε(σ) = (u − (1 + ε) sin u)/ε^{3/2} = h(σ) + ε r(σ) + ε² ρ_ε(σ), with
h = −σ + σ³/6, r = σ³/6 − σ⁵/120, |ρ_ε(σ)| <= m2(|σ|) := |σ|⁷/5040 + |σ|⁵/120 for every real σ and every ε > 0 (Lemma U1).
The profiled loss L*_ε(θ; A) = min_b mean ℓ(Aφ_ε(σ_i) + b, y_i) satisfies (Lemma U2)

    L*_ε = L*_0 + ε L_1 + R_2,   L_1(θ; A) = A·mean_i (s(z_i) − y_i)·r(σ_i)   (z_i = A h(σ_i) + b*_0, s = logistic),
    |R_2| <= ε² C(θ; A, ε),      C = A·mean_i m2_i + ¼A²·max_i(|r_i − c| + ε m2_i)·mean_i(|r_i − c| + ε m2_i),

for every constant c.  Everything below is either an analytic bound (proved in §20) or a computation with a stated
rounding allowance, labelled in the JSON.

Producers (one process, nice 15, one thread):
    python -m src.uniform_eps constants            # -> results/uniform_eps/constants.json   (seconds)
    python -m src.uniform_eps outer <eps0> <k>     # -> results/uniform_eps/outer_eps<eps0>_k<k>.json + leaves .npz
    python -m src.uniform_eps links <eps0>         # -> results/uniform_eps/links_eps<eps0>.json (ring, PD, switch)
    python -m src.uniform_eps summary              # -> results/uniform_eps/summary.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from mpmath import iv

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "uniform_eps"
R2 = math.sqrt(2)
SAFETY = 1e-9            # float rounding in profile() and in the float means (the producer standard, certificates_v2)
TAU = 1e-7               # target: a cell is excluded when its decision value exceeds TAU (on top of SAFETY)
RHO = 0.15
P_BOX = 24.0
A_SUB = 0.00125          # the 40 annulus sub-intervals of [0.66, 0.71]
EPS_G2_GRID = 1e-4       # resolution of the Lemma G2 range scan
iv.dps = 30


# ------------------------------------------------------------------ Lemma U1: pointwise expansion to second order
def h(s): return -s + s ** 3 / 6
def r(s): return s ** 3 / 6 - s ** 5 / 120
def rp(s): return s ** 2 / 2 - s ** 4 / 24
def m2(S): return S ** 7 / 5040 + S ** 5 / 120          # |phi_eps - h - eps r| <= eps^2 m2(|sigma|)
def rp_bound(S): return S ** 2 / 2 + S ** 4 / 24        # |r'(sigma)| <= rp_bound(S) on |sigma| <= S
def hp_bound(S): return 1 + S ** 2 / 2                  # |h'(sigma)| <= hp_bound(S)


def r_max(S):
    """max_{|σ| <= S} |r(σ)|, exact: |r| = |σ|³|20 − σ²|/120 has its interior maximum at σ² = 12 (value
    12^{3/2}·8/120 = 2.7713), vanishes at σ² = 20 and is increasing for σ² > 20."""
    S = abs(S)
    loc = 12 ** 1.5 * 8 / 120 if S >= math.sqrt(12) else S ** 3 * (20 - S * S) / 120
    return max(loc, S ** 3 * abs(S * S - 20) / 120)


def phi_eps_iv(sig, eps):
    """φ_ε(σ) exactly, in interval arithmetic (sig, eps: mpf or iv)."""
    sig, eps = iv.mpf(sig), iv.mpf(eps)
    u = iv.sqrt(eps) * sig
    return (u - (1 + eps) * iv.sin(u)) / (eps * iv.sqrt(eps))


def lemma_U1_check(n=4000, seed=0):
    """Certified spot check of Lemma U1 (interval arithmetic, 30 digits): |φ_ε − h − εr| <= ε² m2 at random (σ, ε),
    σ ∈ [−100, 100], ε ∈ [1e−8, 0.6].  Returns the largest ratio |remainder|/(ε² m2) (upper end of the interval)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    dps0, iv.dps = iv.dps, 80        # the remainder is ~ε²: 30 digits lose it to cancellation at ε ~ 1e−8, small σ
    for _ in range(n):
        eps = mp.mpf(10 ** rng.uniform(-8, math.log10(0.6)))
        sig = mp.mpf(rng.uniform(-100, 100))
        rem = phi_eps_iv(sig, eps) - (h(iv.mpf(sig)) + iv.mpf(eps) * r(iv.mpf(sig)))
        bound = iv.mpf(eps) ** 2 * m2(abs(iv.mpf(sig)))
        worst = max(worst, float(mp.mpf((abs(rem) / bound).b)))
    iv.dps = dps0
    return worst


# ------------------------------------------------------------------ Lemma U2: the profiled loss to second order
def _sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def point_terms(p, q, A, x, y, eps):
    """L*_0, L_1 and the remainder constant C(θ; A, eps) at single points θ = (p, q) (arrays), float with SAFETY."""
    from .limit_bnb import profile
    p = np.atleast_1d(np.asarray(p, float)); q = np.atleast_1d(np.asarray(q, float))
    L, b, _, _ = profile(p, q, A, x, y)
    S = p[:, None] * x[None, :] + q[:, None]
    z = A * h(S) + b[:, None]
    e = _sigmoid(z) - y[None, :]
    rc = r(S)
    L1 = A * (e * rc).mean(axis=1)
    c = np.median(rc, axis=1, keepdims=True)
    dv = np.abs(rc - c)
    mm = m2(np.abs(S))
    C = A * mm.mean(axis=1) + 0.25 * A * A * (dv + eps * mm).max(axis=1) * (dv + eps * mm).mean(axis=1)
    return L, L1, C


def profiled_exact(p, q, A, eps, x, y):
    """L*_ε(θ; A) with the exact φ_ε (float; for tests of Lemma U2 on constructed cases)."""
    u = math.sqrt(eps) * (p * x + q)
    ph = (u - (1 + eps) * np.sin(u)) / eps ** 1.5
    return _profile_phi(A * ph, y)


def _profile_phi(z0, y):
    """min_b mean ℓ(z0 + b, y) by bisection/Newton on the monotone first-order condition (float)."""
    ybar = y.mean()
    lo, hi = math.log(ybar / (1 - ybar)) - z0.max(), math.log(ybar / (1 - ybar)) - z0.min()
    b = 0.5 * (lo + hi)
    for _ in range(300):
        s = _sigmoid(z0 + b)
        g = s.mean() - ybar
        if abs(g) < 1e-15:
            break
        if g < 0: lo = b
        else: hi = b
        hh = (s * (1 - s)).mean()
        nb = b - g / max(hh, 1e-300)
        b = nb if lo < nb < hi else 0.5 * (lo + hi)
        if hi - lo < 1e-15 * max(1, abs(b)):
            break
    z = z0 + b
    return float((np.logaddexp(0, z) - y * z).mean())


# ------------------------------------------------------------------ certified inputs (committed limit certificates)
def limit_inputs():
    parts = pd.read_csv(RESULTS / "certv2_annulus_parts.csv")
    parts = parts[parts.tag == "annulus"].reset_index(drop=True)
    summ = pd.read_csv(RESULTS / "certv2_annulus_summary.csv").iloc[0]
    arb = json.loads((RESULTS / "certificate_checks" / "outer_glob_annulus.json").read_text())
    fo = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    lg = json.loads((RESULTS / "local_global.json").read_text())
    b = pd.read_csv(RESULTS / "mn2_bounds.csv").set_index("quantity").value
    assert parts.certified.all() and len(parts) == 40
    return {"parts": parts, "pd_lambda": float(summ.pd_lambda_min_lower), "arb": arb["per_subinterval"],
            "A_star": (float(fo.A_star_lo), float(fo.A_star_hi)), "Z_A": tuple(lg["switch_box"]["A_range"]),
            "centre": (float(parts.centre_p.iloc[0]), float(parts.centre_q.iloc[0])),
            "B24": float(b["B(24) (exact PAVA, 50 digits)"]), "B_full": float(b["B_full (exact PAVA, 50 digits)"]),
            "eps0_G": float(lg["EPS0"])}


def switch_subintervals(c):
    """Indices k of the annulus sub-intervals [0.66 + k·A_SUB, 0.66 + (k+1)·A_SUB] that meet §16's switch box A-range."""
    za = c["Z_A"]
    return [k for k in range(40) if 0.66 + (k + 1) * A_SUB > za[0] and 0.66 + k * A_SUB < za[1]]


def margin_near_switch(c):
    """Task 2: the certified outer-exclusion margin on the sub-intervals that cover the switch box, with its enclosure.
    Lower end: the certificate (producer float B&B with SAFETY; Arb re-check independently).  Upper end: a point outside
    the 0.15 box that the producer evaluated (outer_attained at A_c) minus the branch upper bound (+ SAFETY)."""
    P = c["parts"]
    rows = []
    for k in switch_subintervals(c):
        pr = P.iloc[k]
        rows.append({"k": k, "A": [float(pr.A_lo), float(pr.A_hi)], "margin_producer": float(pr.outer_margin),
                     "margin_arb_check": float(c["arb"][k]["margin"]),
                     "upper_end": float(pr.outer_attained - pr.branch_upper + SAFETY),
                     "branch_upper": float(pr.branch_upper)})
    m_lo = min(r_["margin_producer"] for r_ in rows)
    return {"subintervals": rows, "m_lower": m_lo, "m_upper": min(r_["upper_end"] for r_ in rows),
            "m_lower_arb": min(r_["margin_arb_check"] for r_ in rows),
            "note": "the lower end is the producer's stopping target MARGIN = 1e-4 plus slack; the true margin lies in "
                    "[m_lower, m_upper]"}


# ------------------------------------------------------------------ the margin-vs-perturbation decision
def decide(m_lo, m_hi, d1_lo, d1_hi, c2, eps):
    """The difference Δ(ε) = L*_ε(competitor) − L*_ε(branch) lies in
        [m_lo + ε·d1_lo − ε²·c2,  m_hi + ε·d1_hi + ε²·c2]
    when Δ(0) ∈ [m_lo, m_hi], the first-order shift of the difference ε(L_1' − L_1) lies in [ε d1_lo, ε d1_hi], and both
    remainders are bounded by ε²·(their C) with c2 the sum (Lemma U2).  'pass': the branch wins for this ε; 'fail': the
    competitor certainly wins; 'undecided' otherwise."""
    lo = m_lo + eps * d1_lo - eps * eps * c2
    hi = m_hi + eps * d1_hi + eps * eps * c2
    return "pass" if lo > 0 else "fail" if hi < 0 else "undecided"


# ------------------------------------------------------------------ Task 3, uniform version: margin vs sup perturbation
def uniform_eps0(c, x, y, margin, A_hi, safety=0.8):
    """ε₀ from ONE certified margin against the perturbation of the difference, bounded uniformly over K(24) ∖ box:
        D(ε) = m + ε (L1_lo(comp) − L1_hi(branch)) − ε² (C_K + C_branch),
        L1_lo(comp) >= −A·r_max(S_K) (|L_1| <= A mean|r_i − c|, mean(s(z) − y) = 0),
        C_K = A·m2(S_K) + ¼A²(2 r_max(S_K) + ε m2(S_K))(2 r_max(S_K) + ε m2(S_K))  (c = 0, |r − 0| <= r_max).
    L1_hi(branch), C_branch at θ_c and both ends of the A-interval.  Returns the root of D and ε₀ = safety·root."""
    pc, qc = c["centre"]
    SK = 2 * P_BOX + 2 * R2 + 2 * P_BOX
    A_lo = A_hi - A_SUB
    bt = [point_terms([pc], [qc], A, x, y, 1e-3) for A in (A_lo, A_hi)]
    L1b = max(float(t[1][0]) for t in bt) + SAFETY
    Cb = max(float(t[2][0]) for t in bt)
    D1 = A_hi * r_max(SK) + L1b

    def D(e):
        CK = A_hi * m2(SK) + 0.25 * A_hi ** 2 * (2 * r_max(SK) + e * m2(SK)) ** 2
        return margin - e * D1 - e * e * (CK + Cb)
    lo, hi = 0.0, margin / D1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if D(mid) > 0 else (lo, mid)
    e0 = float(f"{safety * lo:.2g}")
    CK0 = A_hi * m2(SK) + 0.25 * A_hi ** 2 * (2 * r_max(SK) + e0 * m2(SK)) ** 2
    verdict = decide(margin, math.inf, -A_hi * r_max(SK) - L1b, math.inf, CK0 + Cb, e0)
    return {"S_K": SK, "r_max_SK": r_max(SK), "L1_branch_hi": L1b, "C_branch": Cb, "first_order_coeff": D1,
            "decision_at_eps0": verdict, "C_K_at_eps0": CK0,
            "root": lo, "eps0": float(f"{safety * lo:.2g}"), "D_at_eps0": D(float(f"{safety * lo:.2g}")),
            "first_order_share_at_root": lo * D1 / margin}


# ------------------------------------------------------------------ ε-uniform exact (isotonic) exclusions, Lemma G2
def b_table(x, y, Ps):
    """B(P) (§3(a), exact PAVA block sums, logs at 50 digits) on a grid of P: a lower bound of L*_ε for |p| >= P that
    holds for every ε in Lemma G2's range (the dropped points have |σ| < 2√2 at every such ε)."""
    from .math_note_v2_checks import monotone_bound_exact
    return [(float(P), float(monotone_bound_exact(x, y, 4 * R2 / P))) for P in Ps]


def separation_bound(S, d, x, A_lo, eps0):
    """§5.0's separation lemma in rescaled form, for every ε in [0, eps0] (Lemma G1 inflation): if every point of a
    class-1 window has φ_ε <= M and every inner (class-0) point has φ_ε >= M + g with g > 0, then for every b the logit
    gap is >= A·g >= A_lo·g and L* >= π·softplus(A_lo·g/2), π = min(#window, #inner)/n.  Both outer windows are tried;
    a class-1 window ABOVE the inner one is the correct order and gives nothing.  S, d: centre σ and half-widths per
    point (rows = cells).  Returns the bound per cell (0 where no window is separated)."""
    from .limit_bnb import _h_extrema
    from .local_global import m0
    lo, hi = S - d, S + d
    hmin, hmax = _h_extrema(lo, hi)
    infl = eps0 * m0(np.maximum(np.abs(lo), np.abs(hi)))
    plo, phi_ = hmin - infl, hmax + infl
    inner = np.abs(x) <= 0.8 + 1e-12
    n = len(x)
    out = np.zeros(S.shape[0])
    for win in (x <= -1.2 + 1e-12, x >= 1.2 - 1e-12):
        g = plo[:, inner].min(axis=1) - phi_[:, win].max(axis=1)
        pi = min(int(win.sum()), int(inner.sum())) / n
        val = np.where(g > 0, pi * np.logaddexp(0, A_lo * np.maximum(g, 0) / 2), 0.0)
        out = np.maximum(out, val)
    return out * (1 - 1e-12)


def favourable_profile(S, d, y, A_lo, A_hi, eps0):
    """A lower bound of L*_ε(θ; A) for every θ in the cell, A ∈ [A_lo, A_hi], ε ∈ [0, eps0].  Each point's φ_ε lies in
    [min h − eps0·m0, max h + eps0·m0] over its σ-interval (Lemma G1), so its logit A·φ_ε + b is at most (class 1) /
    at least (class 0) the favourable corner z_fav + b, and ℓ(·, 1) decreases, ℓ(·, 0) increases.  Hence
    L*_ε >= min_b mean ℓ(z_fav + b), whose minimum over b is bracketed by bisection on the monotone F(b) and bounded
    below by the tangent at the bracket's left end (convexity)."""
    from .limit_bnb import _h_extrema
    from .local_global import m0
    lo, hi = S - d, S + d
    hmin, hmax = _h_extrema(lo, hi)
    infl = eps0 * m0(np.maximum(np.abs(lo), np.abs(hi)))
    plo, phi_ = hmin - infl, hmax + infl
    one = y[None, :] > 0.5
    zf = np.where(one, np.maximum(A_lo * phi_, A_hi * phi_), np.minimum(A_lo * plo, A_hi * plo))
    ybar = y.mean()
    blo = math.log(ybar / (1 - ybar)) - zf.max(axis=1) - 1.0
    bhi = math.log(ybar / (1 - ybar)) - zf.min(axis=1) + 1.0
    for _ in range(200):
        bm = 0.5 * (blo + bhi)
        F = _sigmoid(zf + bm[:, None]).mean(axis=1) - ybar
        blo, bhi = np.where(F < 0, bm, blo), np.where(F < 0, bhi, bm)
    z = zf + blo[:, None]
    g = (np.logaddexp(0, z) - y[None, :] * z).mean(axis=1)
    gp = (_sigmoid(z) - y[None, :]).mean(axis=1)                 # <= 0 at the left end
    val = g + np.minimum(gp, 0) * (bhi - blo)
    return val - SAFETY * np.maximum(1.0, np.abs(val))


def iso_retained(ys_kept, n):
    """(1/n) × the best monotone logistic fit (either direction) to the retained labels, in exact arithmetic (integer
    PAVA block sums, 50-digit logs; math_note_v2_checks).  A lower bound of L*_ε whenever the logit is monotone in x on
    the retained points (Lemma G2, step 3) and the other points' losses are dropped (they are >= 0)."""
    from .math_note_v2_checks import _logloss_exact, _pava_blocks
    if len(ys_kept) == 0:
        return 0.0
    v = min(_logloss_exact(_pava_blocks(ys_kept)), _logloss_exact(_pava_blocks(ys_kept[::-1])))
    return float(v / n) - 1e-15                      # the float conversion, rounded down


def g2_range(x, y, A_min=0.66, step=EPS_G2_GRID, top=0.06):
    """Largest grid ε with Lemma G2(a) (one fold window, u0 < 2√2√ε) for s >= A_min ε^{-3/2}.  The conditions are
    monotone in ε (W increases, 2π − β decreases), so they hold on the whole of (0, ε]."""
    from .local_global import fold_window_conditions
    last, e = None, step
    while e <= top + 1e-12:
        f = fold_window_conditions(e, x, y)
        if not (f["one_fold"] and f["u0_below_2sqrt2_sqrt_eps"]):
            break
        last = {"eps": e, **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in f.items()}}
        e = round(e + step, 12)
    return last


# ------------------------------------------------------------------ ε-uniform outer exclusion, cell by cell
def outer_eps(A_lo, A_hi, centre, x, y, eps0, btab, B_full, P=P_BOX, h0=0.25, rho=RHO, tau=TAU,
              max_cells=1_500_000, max_rounds=16, record=True):
    """Certify L*_ε(θ; A) > L*_ε(θ_c; A) for every θ ∈ K(P) ∩ {p >= 0} outside the rho box, every A ∈ [A_lo, A_hi]
    and every ε ∈ [0, eps0] (§20.4).  A cell is excluded by ANY of:
      (iso)   p_lo >= P_k with B(P_k) > BU(eps0) + tau            (exact, ε-uniform, Lemma G2)
      (full)  every point has |σ_i| >= 2√2 on the whole cell, and B_full > BU(eps0) + tau      (exact, ε-uniform)
      (pert)  D(0) > tau and D(eps0) > tau, D(ε) = lb0 − bu0 + ε(L1_lo − L1_bu) − ε²(C + C_bu)  (Lemma U2; D is
              concave in ε, so its minimum over [0, eps0] is at an end)
    where lb0 is the producer's limit cell bound (certificates_v2.outer_exclusion, same formula), L1_lo the larger of a
    centre-plus-Lipschitz bound and the crude −A·mean|r − c|, and BU(ε) = bu0 + ε·max(L1_bu, 0) + ε²C_bu."""
    from .limit_bnb import profile
    p0, q0 = centre
    Ac, hA = 0.5 * (A_lo + A_hi), 0.5 * (A_hi - A_lo)
    Qb = 2 * R2 + 2 * P
    npn, nq = int(math.ceil(P / h0)), int(math.ceil(2 * Qb / h0))
    hp, hq = P / npn / 2, Qb / nq
    cp, cq = np.meshgrid((np.arange(npn) + 0.5) * 2 * hp, -Qb + (np.arange(nq) + 0.5) * 2 * hq, indexing="ij")
    cp, cq = cp.ravel(), cq.ravel()
    ax = np.abs(x)
    # branch side at θ_c: L*_0, L_1 and C at both ends (L*_ε(θ_c; ·) is convex in A for every ε)
    bt = [point_terms([p0], [q0], A, x, y, eps0) for A in (A_lo, A_hi)]
    bu0 = max(float(t[0][0]) for t in bt) + SAFETY
    L1bu = max(float(t[1][0]) for t in bt) + SAFETY
    Cbu = max(float(t[2][0]) for t in bt)
    BU = bu0 + eps0 * max(L1bu, 0.0) + eps0 ** 2 * Cbu
    Ps = np.array([t[0] for t in btab]); Bs = np.array([t[1] for t in btab])
    iso_ok = Bs > BU + tau
    P_iso = float(Ps[iso_ok].min()) if iso_ok.any() else math.inf
    full_ok = B_full > BU + tau
    leaves = []
    stats = {"iso": 0, "full": 0, "pert": 0, "iso_cell": 0, "sep": 0, "fav": 0, "in_box": 0}
    order = np.argsort(x, kind="stable")
    ys = y[order].astype(int)
    cache = {}

    def cell_iso(drop_sorted):
        key = np.packbits(drop_sorted).tobytes()
        if key not in cache:
            cache[key] = iso_retained(ys[~drop_sorted], len(ys))
        return cache[key]

    min_D = math.inf
    rounds, peak = 0, len(cp)
    while True:
        rounds += 1
        inbox = (np.abs(cp - p0) + hp <= rho - 1e-12) & (np.abs(cq - q0) + hq <= rho - 1e-12)
        stats["in_box"] += int(inbox.sum())
        cp, cq = cp[~inbox], cq[~inbox]
        n = len(cp)
        reason = np.zeros(n, np.int8)
        Dlo = np.full(n, -np.inf)
        # exact exclusions first (cheap)
        reason[(cp - hp >= P_iso)] = 1
        for i in range(0, n, 2000):
            sl = slice(i, i + 2000)
            idx = np.arange(n)[sl]
            pc_, qc_ = cp[sl], cq[sl]
            S = pc_[:, None] * x[None, :] + qc_[:, None]
            d = ax[None, :] * hp + hq
            smin = np.abs(S) - d
            if full_ok:
                full = (smin >= 2 * R2).all(axis=1) & (reason[sl] == 0)
                reason[idx[full]] = 2
            todo = reason[sl] == 0
            if not todo.any():
                continue
            j = idx[todo]
            S = S[todo]
            L, b, gp, gq = profile(cp[j], cq[j], Ac, x, y)
            smax = np.abs(S) + d
            curv = 0.5 * A_hi * (smax * d ** 2).mean(axis=1)
            hc = h(S)
            medh = np.median(hc, axis=1, keepdims=True)
            hdev = np.abs(hc - medh) + hp_bound(smax) * d
            lb0 = L - np.abs(gp) * hp - np.abs(gq) * hq - curv - hA * hdev.mean(axis=1) - SAFETY
            # first-order term over the cell
            z = Ac * hc + b[:, None]
            e = _sigmoid(z) - y[None, :]
            rc = r(S)
            L1c = Ac * (e * rc).mean(axis=1)
            medr = np.median(rc, axis=1, keepdims=True)
            rdev = np.abs(rc - medr) + rp_bound(smax) * d
            Dr = rdev.mean(axis=1)
            Hp = (hp_bound(smax) * ax[None, :]).max(axis=1)
            Hq = hp_bound(smax).max(axis=1)
            rpb = rp_bound(smax)
            Lp = 0.5 * A_hi ** 2 * Hp * Dr + A_hi * (rpb * ax[None, :]).mean(axis=1)
            Lq = 0.5 * A_hi ** 2 * Hq * Dr + A_hi * rpb.mean(axis=1)
            LA = Dr * (1 + 0.5 * A_hi * 2 * hdev.max(axis=1))
            L1_lip = L1c - hp * Lp - hq * Lq - hA * LA - SAFETY
            L1_lo = np.maximum(L1_lip, -A_hi * Dr - SAFETY)
            mm = m2(smax)
            C = A_hi * mm.mean(axis=1) + 0.25 * A_hi ** 2 * (rdev + eps0 * mm).max(axis=1) * (rdev + eps0 * mm).mean(axis=1)
            D0 = lb0 - bu0
            De = D0 + eps0 * (L1_lo - L1bu) - eps0 ** 2 * (C + Cbu)
            Dm = np.minimum(D0, De)
            Dlo[j] = Dm
            ok = Dm > tau
            reason[j[ok]] = 3
            # (sep) a class-1 window entirely below the inner window on the logit scale, for every ε <= eps0
            if (~ok).any():
                t_ = np.nonzero(~ok)[0]
                sep = separation_bound(S[t_], np.broadcast_to(d, S[t_].shape), x, A_lo, eps0)
                hit = sep > BU + tau
                reason[j[t_[hit]]] = 5
                ok = ok.copy(); ok[t_[hit]] = True
            # (fav) the favourable-logit profile over the cell: exact in ε (Lemma G1), zeroth order in θ and A
            if (~ok).any():
                t_ = np.nonzero(~ok)[0]
                fv = favourable_profile(S[t_], np.broadcast_to(d, S[t_].shape), y, A_lo, A_hi, eps0)
                hit = fv > BU + tau
                reason[j[t_[hit]]] = 6
                ok = ok.copy(); ok[t_[hit]] = True
            # (iso-cell) the cell's own dropped set: points that can reach |σ| < 2√2 somewhere on the cell
            for t in np.nonzero(~ok)[0]:
                drop = (np.abs(S[t]) - d[0] < 2 * R2)
                if cell_iso(drop[order]) > BU + tau:
                    reason[j[t]] = 4
        done = reason > 0
        if done.any():
            if (reason == 3).any():
                min_D = min(min_D, float(Dlo[reason == 3].min()))
            for k_, nm in ((1, "iso"), (2, "full"), (3, "pert"), (4, "iso_cell"), (5, "sep"), (6, "fav")):
                stats[nm] += int((reason == k_).sum())
            if record:
                leaves.append(np.rec.fromarrays([cp[done], cq[done], np.full(done.sum(), hp), np.full(done.sum(), hq),
                                                 reason[done], np.where(reason[done] == 3, Dlo[done], np.nan)],
                                                names="p,q,hp,hq,reason,D"))
        keep = ~done
        info = {"A": [A_lo, A_hi], "eps0": eps0, "branch_upper0": bu0, "L1_branch": L1bu, "C_branch": Cbu,
                "BU_eps0": BU, "P_iso": P_iso, "full_ok": bool(full_ok), "rounds": rounds, "peak_cells": peak,
                "final_h": max(hp, hq), "stats": dict(stats), "min_D_pert": min_D, "tau": tau}
        if not keep.any():
            return {"certified": True, **info}, leaves
        if rounds >= max_rounds or 4 * keep.sum() > max_cells:
            worst = int(np.argmin(np.where(keep, Dlo, np.inf)))
            return {"certified": False, "live": int(keep.sum()),
                    "worst_live": {"p": float(cp[worst]), "q": float(cq[worst]), "D": float(Dlo[worst])}, **info}, leaves
        cp, cq = cp[keep], cq[keep]
        hp, hq = hp / 2, hq / 2
        cp = np.concatenate([cp - hp, cp - hp, cp + hp, cp + hp])
        cq = np.concatenate([cq - hq, cq + hq, cq - hq, cq + hq])
        peak = max(peak, len(cp))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fmt(e): return f"{e:g}".replace("e-0", "e-")


# ------------------------------------------------------------------ drivers
def run_constants():
    from .limit_bnb import population
    x, y = population()
    c = limit_inputs()
    t0 = time.time()
    mg = margin_near_switch(c)
    A_hi = max(r_["A"][1] for r_ in mg["subintervals"])
    pc, qc = c["centre"]
    radii = {"S_c": 2 * abs(pc) + abs(qc), "S_U": 2 * (abs(pc) + 0.05) + abs(qc) + 0.05,
             "S_R": 2 * (abs(pc) + RHO) + abs(qc) + RHO, "S_K": 4 * P_BOX + 2 * R2}
    sets = {k: {"S": v, "m2": m2(v), "r_max": r_max(v), "rp_bound": rp_bound(v)} for k, v in radii.items()}
    # C on the branch point and as a uniform bound on each set (c = 0: |r_i| <= r_max(S))
    for k, v in radii.items():
        sets[k]["C_uniform_eps1e-3"] = A_hi * m2(v) + 0.25 * A_hi ** 2 * (2 * r_max(v) + 1e-3 * m2(v)) ** 2
    bt = {f"A={A}": dict(zip(("L0", "L1", "C"), [float(t[0]) for t in point_terms([pc], [qc], A, x, y, 1e-3)]))
          for A in (c["Z_A"][0], c["Z_A"][1], mg["subintervals"][0]["A"][0], A_hi)}
    unif = uniform_eps0(c, x, y, mg["m_lower"], A_hi)
    unif_arb = uniform_eps0(c, x, y, mg["m_lower_arb"], A_hi)
    btab = b_table(x, y, [18.0, 18.25, 18.5, 18.75, 19.0, 19.5, 20.0, 21.0, 22.0, 24.0])
    g2 = g2_range(x, y)
    u1 = lemma_U1_check()
    res = {"lemma_U1_spot_check_max_ratio": u1, "radii": radii, "sets": sets, "branch_point_terms": bt,
           "margin": mg, "uniform_eps0_producer_margin": unif, "uniform_eps0_arb_margin": unif_arb,
           "eps0_section16": c["eps0_G"], "B_table": btab, "B_full": c["B_full"], "G2_range": g2,
           "switch_subintervals": switch_subintervals(c), "Z_A": c["Z_A"], "seconds": time.time() - t0,
           "status": {"lemma_U1": "proved (Taylor remainders of sin); spot-checked in interval arithmetic",
                      "lemma_U2": "proved", "margin": "certified (producer float B&B + SAFETY; Arb re-check)",
                      "uniform_eps0": "proved given the certified margin (float evaluation of L1, C at theta_c "
                                      "with SAFETY)", "B_table": "exact (integer PAVA, 50-digit logs)",
                      "G2_range": "float evaluation of W(s, a) (the §5.0 formula) on a grid"}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "constants.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("sets",)}, indent=1, default=float))


def run_outer(eps0, ks=None):
    """The ε-uniform outer exclusion on every annulus sub-interval (resumable: a finished k is skipped).  The leaf list
    is written for the sub-intervals that meet the switch box (the export sample for the U3 checker); for the others the
    SHA-256 of the in-memory leaf array is recorded so that a re-run can be compared bit for bit."""
    from .limit_bnb import population
    x, y = population()
    c = limit_inputs()
    cons = json.loads((OUT / "constants.json").read_text())
    ks = range(40) if ks is None else ks
    for k in ks:
        tag = f"outer_eps_{_fmt(eps0)}_k{k:02d}"
        if (OUT / f"{tag}.json").exists():
            continue
        pr = c["parts"].iloc[k]
        t0 = time.time()
        res, leaves = outer_eps(float(pr.A_lo), float(pr.A_hi), c["centre"], x, y, eps0, cons["B_table"], c["B_full"])
        res["seconds"] = time.time() - t0
        res["k"] = k
        res["eps_in_G2_range"] = bool(eps0 <= cons["G2_range"]["eps"])
        if leaves:
            lv = np.concatenate(leaves)
            arr = {f: np.ascontiguousarray(lv[f]) for f in ("p", "q", "hp", "hq", "reason", "D")}
            res["n_leaves"] = int(len(lv))
            res["leaves_array_sha256"] = hashlib.sha256(b"".join(arr[f].tobytes() for f in arr)).hexdigest()
            if k in cons["switch_subintervals"]:
                np.savez_compressed(OUT / f"{tag}_leaves.npz", **arr)
                res["leaves_file"] = f"{tag}_leaves.npz"
                res["leaves_sha256"] = _sha(OUT / f"{tag}_leaves.npz")
        res["inputs_sha256"] = {n: _sha(RESULTS / n) for n in ("certv2_annulus_parts.csv", "mn2_bounds.csv",
                                                                "uniform_eps/constants.json")}
        (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
        print(k, res["certified"], res["min_D_pert"], res["stats"], round(res["seconds"], 1), flush=True)


def run_outer_scan(eps_list, k):
    """EXPLORATORY: how far the outer link alone reaches at larger ε (leaves not recorded; not part of the theorem)."""
    from .limit_bnb import population
    x, y = population()
    c = limit_inputs()
    cons = json.loads((OUT / "constants.json").read_text())
    pr = c["parts"].iloc[k]
    rows = []
    for e in eps_list:
        t0 = time.time()
        res, _ = outer_eps(float(pr.A_lo), float(pr.A_hi), c["centre"], x, y, e, cons["B_table"], c["B_full"],
                           record=False, max_rounds=10)
        res["seconds"] = time.time() - t0
        rows.append(res)
        print(json.dumps(res, default=float), flush=True)
    (OUT / f"outer_scan_{k:02d}.json").write_text(json.dumps({"k": k, "rows": rows}, indent=1, default=float))


def ring_pd_limits(c, k_list, A_hi):
    """Ring and PD links (§16.2's first-order bounds, Lemma G1) with the near-switch margins: the largest ε with
    positive slack, by bisection (all bounds are increasing in ε)."""
    from . import local_global as lg
    P = c["parts"]
    ring_m = min(float(P.iloc[k].ring_grad_lower) for k in k_list)
    cc = {"outer_margin": 1.0, "ring_grad_lower": ring_m, "branch_upper": max(float(P.iloc[k].branch_upper) for k in k_list),
          "pd_lambda": c["pd_lambda"], "B24": c["B24"], "B_full": c["B_full"], "centre": c["centre"]}
    out = {"ring_margin": ring_m, "pd_margin": c["pd_lambda"]}
    for name in ("ring", "pd"):
        lo, hi = 0.0, 1e-2
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if lg.link_conditions(mid, cc, A=A_hi)[name]["slack"] > 0 else (lo, mid)
        out[f"{name}_eps_limit"] = lo
    return out, cc


def m2_1(S): return S ** 6 / 720 + S ** 4 / 24           # |φ_ε' − h' − ε r'|  <= ε² m2_1   (Lemma U1')
def m2_2(S): return S ** 5 / 120 + S ** 3 / 6            # |φ_ε'' − h'' − ε r''| <= ε² m2_2  (Lemma U1')


class Inflated2:
    """φ_ε, φ_ε', φ_ε'' enclosed for every ε ∈ [0, eps0] to SECOND order (Lemmas U1, U1'):
    h + [0, eps0]·r + [−1, 1]·eps0²·m2,  h' + [0, eps0]·r' + [−1, 1]·eps0²·m2_1,  h'' + [0, eps0]·r'' + [−1, 1]·eps0²·m2_2.
    The attribute e = eps0 keeps local_global's first-order extrema allowance (Lemma G1) in gap_bounds."""

    def __init__(self, eps0):
        self.e = eps0
        self.E = iv.mpf([0, mp.mpf(eps0)])

    def _mag(self, s):
        return float(max(abs(mp.mpf(s.a)), abs(mp.mpf(s.b))))

    def _w(self, m):
        w = mp.mpf(self.e) ** 2 * mp.mpf(m)
        return iv.mpf([-w, w])

    def phi(self, s):
        return -s + s ** 3 / 6 + self.E * (s ** 3 / 6 - s ** 5 / 120) + self._w(m2(self._mag(s)))

    def phi1(self, s):
        return -1 + s ** 2 / 2 + self.E * (s ** 2 / 2 - s ** 4 / 24) + self._w(m2_1(self._mag(s)))

    def phi2(self, s):
        return s + self.E * (s - s ** 3 / 6) + self._w(m2_2(self._mag(s)))


def switch_part(eps0, J, c, x, y, order2=True, rz=None):
    """§16.3 restricted to J: the switch box Z (4-variable Krawczyk) and 3-variable branch boxes over J, all with the
    ε-inflation of Lemma G1 at eps0 (local_global's machinery, imported read-only)."""
    from . import local_global as lg
    ph = Inflated2(eps0) if order2 else lg.Inflated(eps0)
    fo = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    Astar = 0.5 * (float(fo.A_star_lo) + float(fo.A_star_hi))
    zf = np.array([float(fo.p_star), float(fo.q_star), float(fo.b_star), Astar])
    funZ = lambda X, Jf=True: lg.system(X, None, x, y, ph, with_gap=True, want_J=Jf)
    okZ, KZ, XZ = lg.krawczyk(funZ, zf, rz or lg.RZ)
    p, q = XZ[0], XZ[1]
    I_act, O_act = ph.phi(q + 0.8 * p), ph.phi(q - 1.2 * p)
    others_I = [ph.phi(q - 0.8 * p)]
    others_O = [ph.phi(q - 2.0 * p), ph.phi(q + 1.2 * p), ph.phi(q + 2.0 * p)]
    sI0, sOm1, sOp0 = q - 0.8 * p, q - 1.2 * p, q + 1.2 * p
    # φ_ε's critical points are at ±arccos(1/a)/√ε = ±√2(1 + O(ε)); require the σ-ranges to clear them by 0.01 + 2√2·ε0
    dcrit = 0.01 + 2 * R2 * eps0
    crit_ok = (sI0.a > -R2 + dcrit) and (sOm1.b < R2 - dcrit) and (sOp0.a > R2 + dcrit)
    active_ok = all(I_act.a > o.b for o in others_I) and all(O_act.b < o.a for o in others_O) and crit_ok
    ZA = (lg._lo(XZ[3]), lg._hi(XZ[3]))
    Zth = [(lg._lo(XZ[i]), lg._hi(XZ[i])) for i in range(3)]
    pc, qc = c["centre"]
    v = zf[:3].copy()
    rows = []
    edges = [J[0]]
    while edges[-1] < ZA[0] - 1e-15:
        edges.append(min(ZA[0], edges[-1] + min(2e-3, max(2e-5, 0.25 * (ZA[0] - edges[-1])))))
    nin = max(1, int(math.ceil((ZA[1] - ZA[0]) / 2e-5)))
    edges += [ZA[0] + (ZA[1] - ZA[0]) * i / nin for i in range(1, nin + 1)]
    while edges[-1] < J[1] - 1e-15:
        edges.append(min(J[1], edges[-1] + min(2e-3, max(2e-5, 0.25 * (edges[-1] - ZA[1]) + 2e-5))))
    queue = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    while queue:
        a_lo, a_hi = queue.pop(0)
        Ac = 0.5 * (a_lo + a_hi)
        vn, Jm = lg.branch_newton(Ac, v, x, y)
        v = vn
        F0, _ = lg.system([iv.mpf(mp.mpf(t)) for t in vn], iv.mpf(mp.mpf(Ac)), x, y, lg.Inflated(0.0))
        Fp = lg.system([iv.mpf(mp.mpf(t)) for t in vn], iv.mpf(mp.mpf(Ac + 1e-6)), x, y, lg.Inflated(0.0), want_J=False)
        th1 = -np.linalg.solve(Jm, np.array([(lg._mid(Fp[i]) - lg._mid(F0[i])) / 1e-6 for i in range(3)]))
        Aint = iv.mpf([mp.mpf(a_lo), mp.mpf(a_hi)])
        fun = lambda Xb, Jf=True: lg.system(Xb, Aint, x, y, ph, want_J=Jf)
        rad = [abs(th1[i]) * (a_hi - a_lo) / 2 + 1e-7 + 50 * eps0 for i in range(3)]
        ok, Kb = False, None
        for _ in range(12):
            ok, Kb, _X = lg.krawczyk(fun, vn, rad)
            if ok:
                break
            rad = [max(rad[i], 1.5 * max(abs(lg._lo(Kb[i]) - vn[i]), abs(lg._hi(Kb[i]) - vn[i]))) for i in range(3)]
        inU = ok and all(lg._lo(Kb[i]) > (pc, qc)[i] - 0.05 and lg._hi(Kb[i]) < (pc, qc)[i] + 0.05 for i in range(2))
        glo, ghi = lg.gap_bounds(Kb[0], Kb[1], ph) if ok else (np.nan, np.nan)
        within = ZA[0] - 1e-15 <= a_lo and a_hi <= ZA[1] + 1e-15
        inZ = ok and all(lg._lo(Kb[i]) > Zth[i][0] and lg._hi(Kb[i]) < Zth[i][1] for i in range(3))
        sign = (-1 if ghi < 0 else 1 if glo > 0 else 0) if ok else None
        if ok and not within and sign == 0 and a_hi - a_lo > 1e-7:
            m_ = 0.5 * (a_lo + a_hi)
            queue[:0] = [(a_lo, m_), (m_, a_hi)]
            continue
        rows.append({"A_lo": a_lo, "A_hi": a_hi, "krawczyk_ok": ok, "in_U": inU, "G_lo": glo, "G_hi": ghi,
                     "sign": sign, "within_Z_A": within, "box_in_Z_theta": inZ})
    left = [r_ for r_ in rows if r_["A_hi"] <= ZA[0] + 1e-15]
    right = [r_ for r_ in rows if r_["A_lo"] >= ZA[1] - 1e-15]
    inside = [r_ for r_ in rows if r_ not in left and r_ not in right]
    cover = abs(rows[0]["A_lo"] - J[0]) < 1e-15 and abs(rows[-1]["A_hi"] - J[1]) < 1e-15 and \
        all(abs(rows[i]["A_hi"] - rows[i + 1]["A_lo"]) < 1e-15 for i in range(len(rows) - 1))
    unique = bool(okZ and active_ok and cover and all(r_["krawczyk_ok"] and r_["in_U"] for r_ in rows)
                  and all(r_["sign"] == -1 for r_ in left) and all(r_["sign"] == 1 for r_ in right)
                  and all(r_["box_in_Z_theta"] for r_ in inside))
    return {"switch_box_ok": bool(okZ), "active_pair_ok": bool(active_ok), "Z_A": ZA, "Z_theta": Zth, "J": list(J),
            "n_boxes": len(rows), "covered": bool(cover), "n_left": len(left), "n_inside": len(inside),
            "n_right": len(right), "max_G_hi_left": max((r_["G_hi"] for r_ in left), default=None),
            "min_G_lo_right": min((r_["G_lo"] for r_ in right), default=None),
            "unique_sign_change_on_J": unique, "rows": rows}


def run_links(eps0):
    """Ring and PD links at eps0 over [0.66, 0.71] (margins: min over the 40 sub-intervals), and the branch's gap over
    the whole range with the switch box (second-order inflation)."""
    from .limit_bnb import population
    from . import local_global as lg
    x, y = population()
    c = limit_inputs()
    ks = list(range(40))
    J = (0.66, 0.71)
    t0 = time.time()
    rp_, cc = ring_pd_limits(c, ks, J[1])
    lk = lg.link_conditions(eps0, cc, A=J[1])
    sw = switch_part(eps0, J, c, x, y)
    res = {"eps0": eps0, "J": J, "subintervals": ks, "ring_pd": rp_,
           "ring": lk["ring"], "pd": lk["pd"], "switch": sw, "seconds": time.time() - t0}
    (OUT / f"links_eps_{_fmt(eps0)}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "switch"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in sw.items() if k != "rows"}, indent=1, default=float))


def run_summary(eps0):
    """The U1 conclusion at eps0: every link at eps0 from its own artifact."""
    cons = json.loads((OUT / "constants.json").read_text())
    ks = list(range(40))
    outer = [json.loads((OUT / f"outer_eps_{_fmt(eps0)}_k{k:02d}.json").read_text()) for k in ks]
    links = json.loads((OUT / f"links_eps_{_fmt(eps0)}.json").read_text())
    loc_slack = min(cons["B_table"][-1][1], cons["B_full"], math.log(2)) - max(o["BU_eps0"] for o in outer)
    res = {"eps0": eps0, "subintervals": ks, "J": links["J"],
           "outer_certified": all(o["certified"] for o in outer), "outer_min_D": min(o["min_D_pert"] for o in outer),
           "outer_seconds": sum(o["seconds"] for o in outer),
           "outer_min_D_switch": min(outer[k]["min_D_pert"] for k in cons["switch_subintervals"]),
           "outer_leaves_total": sum(o["n_leaves"] for o in outer),
           "outer_max_rounds": max(o["rounds"] for o in outer), "outer_max_peak_cells": max(o["peak_cells"] for o in outer),
           "outer_stats_total": {kk: sum(o["stats"][kk] for o in outer) for kk in outer[0]["stats"]},
           "ring_slack": links["ring"]["slack"], "pd_slack": links["pd"]["slack"],
           "ring_eps_limit": links["ring_pd"]["ring_eps_limit"], "pd_eps_limit": links["ring_pd"]["pd_eps_limit"],
           "localisation_slack": loc_slack, "eps_in_G2_range": bool(eps0 <= cons["G2_range"]["eps"]),
           "switch_unique_on_J": links["switch"]["unique_sign_change_on_J"],
           "uniform_eps0": cons["uniform_eps0_producer_margin"]["eps0"], "eps0_section16": cons["eps0_section16"]}
    lims = {"ring": res["ring_eps_limit"], "pd": res["pd_eps_limit"]}
    res["binding_link"] = min(lims, key=lims.get)
    res["eps0_rule"] = "0.8 x min(ring, PD eps limits), 2 significant figures"
    res["eps0_matches_rule"] = bool(eps0 == float(f"{0.8 * min(lims.values()):.2g}"))
    res["theorem_U1_holds"] = bool(res["eps0_matches_rule"] and res["outer_certified"] and res["ring_slack"] > 0 and res["pd_slack"] > 0
                                   and loc_slack > 0 and res["eps_in_G2_range"] and res["switch_unique_on_J"])
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    os.nice(15)
    cmd = sys.argv[1]
    if cmd == "constants":
        run_constants()
    elif cmd == "outer":
        run_outer(float(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else None)
    elif cmd == "outer_scan":
        run_outer_scan([float(v) for v in sys.argv[3].split(",")], int(sys.argv[2]))
    elif cmd == "links":
        run_links(float(sys.argv[2]))
    elif cmd == "summary":
        run_summary(float(sys.argv[2]))
