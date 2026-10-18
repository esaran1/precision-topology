"""Track U, step U2 -- SAMPLE LEAF TIMING ONLY (exploratory prototype; NOT a certificate, nothing here is cited as proved).

Purpose: time a sample of U2 leaves (outer exclusion, ring, PD, switch Krawczyk) with ε as an interval dimension, so that
the full U2 run can be projected before it is committed to (results/uniform_eps/U2_design.md).  The bounds follow the
producer standard of §20 (float evaluation + a stated rounding allowance); the kernel envelopes (Lemma U4 in the design)
are computed on a grid here and still have to be proved before any of this is used as a certificate.

Coordinates (both regimes): the rescaled coordinates of §1, σ = p x + q, w₁ = √ε p, b₁ = π + √ε q, A = s ε^{3/2}, with the
EXACT feature (no expansion), written through entire kernels of u = √ε σ:

    φ   = −σ + (1+ε) σ³ Gt(u),            Gt = (u − sin u)/u³
    φ_σ = −1 + (1+ε) σ² J(u),             J  = (1 − cos u)/u²
    φ_σσ = (1+ε) σ sinc(u)
    φ_ε  = σ³ Gt(u) + (1+ε) σ⁵ K1(u),     K1 = Gt'(u)/(2u)
    φ_σε = σ² J(u) + (1+ε) σ⁴ JJ(u),      JJ = J'(u)/(2u)
    φ_εε = 2σ⁵ K1(u) + (1+ε) σ⁷ K2(u),    K2 = K1'(u)/(2u)

Regime R (ε_hi <= 0.0462): domain K(24) ∩ {p >= 0} (Lemma G2), exact isotonic exclusions (iso, full) as in §20.5.
Regime O (any ε): domain {0 <= w₁ <= W(s_min, a_max)} × one b₁-period (§5.0, valid for every a), mapped to (p, q).

    python -m src.u2_sample outer <eps_lo> <eps_hi> <A_lo> <A_hi> <regime> [export]
    python -m src.u2_sample links <eps_lo> <eps_hi> <A_lo> <A_hi>
    python -m src.u2_sample sweep <name>             # the timing sample (resumable; one JSON line per unit)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "uniform_eps" / "u2_sample"
R2 = math.sqrt(2)
SAFETY = 1e-9
TAU = 1e-7
RHO = 0.15
RHO_U = 0.05
P_BOX = 24.0
G2_EPS = 0.0462
CHUNK = 2000
BOX_CHUNK = int(os.environ.get("U2_BOX_CHUNK", "256"))     # 0 = uncapped (diagnosis only)


# ------------------------------------------------------------------ kernels (entire, even in u)
def _fact(n):
    return float(math.factorial(n))


_NT = 40
_C = {
    "Gt": [(-1) ** m / _fact(2 * m + 3) for m in range(_NT)],
    "J": [(-1) ** m / _fact(2 * m + 2) for m in range(_NT)],
    "sinc": [(-1) ** m / _fact(2 * m + 1) for m in range(_NT)],
    "K1": [(-1) ** (m + 1) * (m + 1) / _fact(2 * m + 5) for m in range(_NT)],
    "JJ": [(-1) ** (m + 1) * (m + 1) / _fact(2 * m + 4) for m in range(_NT)],
    "K2": [(-1) ** m * (m + 2) * (m + 1) / _fact(2 * m + 7) for m in range(_NT)],
    "SS": [(-1) ** (m + 1) * (m + 1) / _fact(2 * m + 3) for m in range(_NT)],
}
_C["K3"] = [(j + 1) * ((-1) ** (j + 1) * (j + 3) * (j + 2) / _fact(2 * j + 9)) for j in range(_NT)]
U_SERIES = 6.0


def _closed(name, u):
    s, c = np.sin(u), np.cos(u)
    if name == "Gt":
        return (u - s) / u ** 3
    if name == "J":
        return (1 - c) / u ** 2
    if name == "sinc":
        return s / u
    if name == "SS":
        return (c / u - s / u ** 2) / (2 * u)
    if name == "K3":                                  # K2'(u)/(2u), K2' by a central difference (u > 6 only)
        hd = 1e-4
        return (_closed("K2", u + hd) - _closed("K2", u - hd)) / (2 * hd) / (2 * u)
    Gt1 = (1 - c) / u ** 3 - 3 * (u - s) / u ** 4
    if name == "K1":
        return Gt1 / (2 * u)
    if name == "JJ":
        return (s / u ** 2 - 2 * (1 - c) / u ** 3) / (2 * u)
    if name == "K2":
        Gt2 = s / u ** 3 - 6 * (1 - c) / u ** 4 + 12 * (u - s) / u ** 5
        K1p = Gt2 / (2 * u) - Gt1 / (2 * u ** 2)
        return K1p / (2 * u)
    raise KeyError(name)


def kern(name, u):
    u = np.abs(np.asarray(u, float))
    out = np.empty_like(u)
    sm = u <= U_SERIES
    if sm.any():
        u2 = u[sm] ** 2
        acc = np.zeros_like(u2)
        for cf in reversed(_C[name]):
            acc = acc * u2 + cf
        out[sm] = acc
    if (~sm).any():
        out[~sm] = _closed(name, u[~sm])
    return out


# envelopes env_k(U) = sup_{|u| >= U} |k(u)| on a grid (step 1e-3, padded by a Lipschitz step), and global Lipschitz
_UG_STEP = 1e-3
_UG = np.arange(0.0, 60.0 + _UG_STEP, _UG_STEP)
_ENV, _LIP = {}, {}
for _nm in _C:
    _v = np.abs(kern(_nm, _UG))
    _d = np.abs(np.diff(kern(_nm, _UG))) / _UG_STEP
    _L = float(_d.max()) * 1.05 + 1e-12
    _LIP[_nm] = _L
    _ENV[_nm] = np.maximum.accumulate(_v[::-1])[::-1] + _L * _UG_STEP


def env(name, U):
    """sup_{|u| >= U} |k(u)| (grid envelope; U > 60 is not expected here and raises)."""
    U = np.asarray(U, float)
    if (U > _UG[-1]).any():
        raise ValueError("u beyond the envelope table")
    return _ENV[name][np.floor(U / _UG_STEP).astype(np.int64)]


# ------------------------------------------------------------------ φ_ε and derivatives (pointwise, float)
def feats(S, e, which=("phi", "phi_s", "phi_e")):
    u = math.sqrt(e) * S
    out = {}
    S2 = S * S
    S3 = S2 * S
    if "phi" in which:
        out["phi"] = -S + (1 + e) * S3 * kern("Gt", u)
    if "phi_s" in which:
        out["phi_s"] = -1 + (1 + e) * S2 * kern("J", u)
    if "phi_ss" in which:
        out["phi_ss"] = (1 + e) * S * kern("sinc", u)
    if "phi_e" in which:
        out["phi_e"] = S3 * kern("Gt", u) + (1 + e) * S3 * S2 * kern("K1", u)
    if "phi_se" in which:
        out["phi_se"] = S2 * kern("J", u) + (1 + e) * S2 * S2 * kern("JJ", u)
    if "phi_ee" in which:
        out["phi_ee"] = 2 * S3 * S2 * kern("K1", u) + (1 + e) * S3 * S3 * S * kern("K2", u)
    return out


def sup_derivs(Smin, Smax, e_lo, e_hi):
    """Per point sup over the cell (|σ| in [Smin, Smax], ε in [e_lo, e_hi]) of |φ_σ|, |φ_σσ|, |φ_ε|, |φ_σε|, |φ_εε|."""
    um = math.sqrt(e_lo) * Smin
    a = 1 + e_hi
    S2 = Smax * Smax
    eGt, eJ, eS, eK1, eJJ, eK2, eSS = (env(k, um) for k in ("Gt", "J", "sinc", "K1", "JJ", "K2", "SS"))
    return {"s": 1 + a * S2 * eJ, "ss": a * Smax * eS, "e": S2 * Smax * eGt + a * S2 * S2 * Smax * eK1,
            "se": S2 * eJ + a * S2 * S2 * eJJ, "ee": 2 * S2 * S2 * Smax * eK1 + a * S2 ** 3 * Smax * eK2,
            "sss": a + 0 * Smax, "sse": Smax * eS + a * S2 * Smax * eSS,
            "see": 10 * S2 * S2 * eK1 + (4 * e_hi + 7 * a) * S2 ** 3 * eK2 + 2 * a * e_hi * S2 ** 4 * env("K3", um)}


# ------------------------------------------------------------------ profiled loss for logits z0 + b (vectorised rows)
def _sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def profile_z0(z0, y):
    """Rows of z0: L* = min_b mean ℓ(z0 + b, y), b*, residual vector e = s(z) − y (float; profiled_bnb.profile's
    safeguarded Newton)."""
    ybar = y.mean()
    L0 = math.log(ybar / (1 - ybar))
    lo = L0 - z0.max(axis=1)
    hi = L0 - z0.min(axis=1)
    b = 0.5 * (lo + hi)
    for _ in range(200):
        sg = _sigmoid(z0 + b[:, None])
        g = sg.mean(axis=1) - ybar
        lo = np.where(g < 0, b, lo)
        hi = np.where(g > 0, b, hi)
        hh = (sg * (1 - sg)).mean(axis=1)
        nw = b - g / np.maximum(hh, 1e-300)
        bad = (nw <= lo) | (nw >= hi) | ~np.isfinite(nw)
        bn = np.where(bad, 0.5 * (lo + hi), nw)
        done = (np.abs(g) <= 1e-14) | (hi - lo <= 1e-15 * np.maximum(1, np.abs(b)))
        b = np.where(done, b, bn)
        if done.all():
            break
    z = z0 + b[:, None]
    L = (np.logaddexp(0, z) - y[None, :] * z).mean(axis=1)
    return L, b, _sigmoid(z) - y[None, :]


def point_eval(p, q, A, e, x, y):
    """L*, gradient (p, q, A, ε) at points (envelope theorem)."""
    p = np.atleast_1d(np.asarray(p, float)); q = np.atleast_1d(np.asarray(q, float))
    S = p[:, None] * x[None, :] + q[:, None]
    f = feats(S, e)
    L, b, r = profile_z0(A * f["phi"], y)
    gp = A * (r * f["phi_s"] * x[None, :]).mean(axis=1)
    gq = A * (r * f["phi_s"]).mean(axis=1)
    gA = (r * f["phi"]).mean(axis=1)
    ge = A * (r * f["phi_e"]).mean(axis=1)
    return L, b, gp, gq, gA, ge, S, f, r


def branch_point(A, e, x, y, start=(1.6686, 1.3697), iters=40):
    """Float Newton for the branch critical point of L*(·; A, ε) (finite differences of the envelope gradient)."""
    v = np.array(start, float)
    for _ in range(iters):
        _, _, gp, gq, *_ = point_eval([v[0]], [v[1]], A, e, x, y)
        g = np.array([gp[0], gq[0]])
        H = np.empty((2, 2))
        for j, dv in enumerate(((1e-6, 0), (0, 1e-6))):
            _, _, gp2, gq2, *_ = point_eval([v[0] + dv[0]], [v[1] + dv[1]], A, e, x, y)
            H[:, j] = (np.array([gp2[0], gq2[0]]) - g) / 1e-6
        step = np.linalg.solve(H, g)
        v = v - step
        if np.abs(step).max() < 1e-13:
            break
    return v


# ------------------------------------------------------------------ branch side over the (A, ε) box at fixed θ_c
def branch_side(pc, qc, A_lo, A_hi, e_lo, e_hi, x, y):
    """Upper bound of L*(θ_c; A, ε) as L_b + g_b·δ + curv_b over the box; returns centre values for the coupled test."""
    Ac, ec = 0.5 * (A_lo + A_hi), 0.5 * (e_lo + e_hi)
    hA, he = 0.5 * (A_hi - A_lo), 0.5 * (e_hi - e_lo)
    L, b, gp, gq, gA, ge, S, f, r = point_eval([pc], [qc], Ac, ec, x, y)
    aS = np.abs(S[0])
    sd = sup_derivs(aS, aS, e_lo, e_hi)
    # (A, ε) Hessian of L*(θ_c; ·), upper bound: ¼ mean (δ_A (φ − c1) + δ_ε A (φ_ε − c2))² + mean |2 δ_A δ_ε φ_ε + δ_ε² A φ_εε|
    ph, phe = f["phi"][0], f["phi_e"][0]
    dph = np.abs(ph - np.median(ph)) + he * sd["e"]
    dphe = np.abs(phe - np.median(phe)) + he * sd["ee"]
    gn = 0.25 * ((hA * dph + he * A_hi * dphe) ** 2).mean()
    nn = (2 * hA * he * (np.abs(phe) + he * sd["ee"]) + he * he * A_hi * sd["ee"]).mean()
    curv_b = 0.5 * (gn + nn)
    upper = float(L[0] + abs(gA[0]) * hA + abs(ge[0]) * he + curv_b + SAFETY)
    return {"Lb": float(L[0]), "gAb": float(gA[0]), "geb": float(ge[0]), "curv_b": float(curv_b), "BU": upper,
            "gp_b": float(gp[0]), "gq_b": float(gq[0])}


# ------------------------------------------------------------------ favourable-logit lower bound (direct, exact in ε)
def phi_range(lo, hi, e_lo, e_hi):
    """Per point [min, max] of φ_ε(σ) over σ ∈ [lo, hi] and ε ∈ [e_lo, e_hi]: exact extrema at ε_c (endpoints and the
    critical points u = ±β + 2πk, β = arccos(1/a)), widened by h_ε·sup|φ_ε|."""
    ec, he = 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)
    se = math.sqrt(ec)
    beta = math.acos(1 / (1 + ec))
    f_lo = feats(lo, ec, ("phi",))["phi"]; f_hi = feats(hi, ec, ("phi",))["phi"]
    mn, mx = np.minimum(f_lo, f_hi), np.maximum(f_lo, f_hi)
    ulo, uhi = lo * se, hi * se
    for c0 in (beta, -beta):
        k0 = np.ceil((ulo - c0) / (2 * math.pi)); k1 = np.floor((uhi - c0) / (2 * math.pi))
        kmax = int(np.nanmax(k1 - k0)) if k1.size else -1
        for j in range(kmax + 1):
            kk = k0 + j
            uc = c0 + 2 * math.pi * kk
            inside = (kk <= k1)
            sc = np.where(inside, uc / se, 0.0)
            fc = feats(sc, ec, ("phi",))["phi"]
            mn = np.where(inside, np.minimum(mn, fc), mn)
            mx = np.where(inside, np.maximum(mx, fc), mx)
    Smin = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    Smax = np.maximum(np.abs(lo), np.abs(hi))
    w = he * sup_derivs(Smin, Smax, e_lo, e_hi)["e"]
    tol = 1e-12 * (1 + np.abs(mn) + np.abs(mx))
    return mn - w - tol, mx + w + tol


def favourable(plo, phi_, y, A_lo, A_hi):
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
    gp = (_sigmoid(z) - y[None, :]).mean(axis=1)
    val = g + np.minimum(gp, 0) * (bhi - blo)
    return val - SAFETY * np.maximum(1.0, np.abs(val))


# ------------------------------------------------------------------ the outer-exclusion unit
def w_domain(A_lo, e_lo, e_hi, x, y):
    from .profiled_bnb import w_bound
    W = w_bound(A_lo / e_hi ** 1.5, 1 + e_hi, x, y)          # W increases as s falls and a rises
    return W, W / math.sqrt(e_lo), math.pi / math.sqrt(e_lo)


def outer_unit(e_lo, e_hi, A_lo, A_hi, centre, x, y, regime, cons, h0=0.25, rho=RHO, tau=TAU,
               max_cells=1_500_000, max_rounds=18, record=True, max_seconds=600.0):
    t_start = time.time()
    pc, qc = centre
    Ac, ec = 0.5 * (A_lo + A_hi), 0.5 * (e_lo + e_hi)
    hA, he = 0.5 * (A_hi - A_lo), 0.5 * (e_hi - e_lo)
    br = branch_side(pc, qc, A_lo, A_hi, e_lo, e_hi, x, y)
    BU = br["BU"]
    if regime == "R":
        assert e_hi <= G2_EPS
        Pmax, Qb = P_BOX, 2 * R2 + 2 * P_BOX
        Ps = np.array([t[0] for t in cons["B_table"]]); Bs = np.array([t[1] for t in cons["B_table"]])
        ok = Bs > BU + tau
        P_iso = float(Ps[ok].min()) if ok.any() else math.inf
        full_ok = cons["B_full"] > BU + tau
        loc_slack = min(cons["B_table"][-1][1], cons["B_full"], math.log(2)) - BU
        q_lo, q_hi = -Qb, Qb
    else:
        W, Pmax, Qh = w_domain(A_lo, e_lo, e_hi, x, y)
        P_iso, full_ok = math.inf, False
        loc_slack = math.log(2) - BU
        q_lo, q_hi = -Qh, Qh
    npn = int(math.ceil(Pmax / h0)); nq = int(math.ceil((q_hi - q_lo) / h0))
    hp, hq = Pmax / npn / 2, (q_hi - q_lo) / nq / 2
    cp, cq = np.meshgrid((np.arange(npn) + 0.5) * 2 * hp, q_lo + (np.arange(nq) + 0.5) * 2 * hq, indexing="ij")
    cp, cq = cp.ravel(), cq.ravel()
    ax = np.abs(x)
    stats = {"iso": 0, "full": 0, "pert": 0, "fav": 0, "in_box": 0}
    leaves = []
    min_D = math.inf
    rounds, peak, evals = 0, len(cp), 0
    while True:
        rounds += 1
        inbox = (np.abs(cp - pc) + hp <= rho - 1e-12) & (np.abs(cq - qc) + hq <= rho - 1e-12)
        stats["in_box"] += int(inbox.sum())
        cp, cq = cp[~inbox], cq[~inbox]
        n = len(cp)
        reason = np.zeros(n, np.int8)
        Dlo = np.full(n, -np.inf)
        reason[cp - hp >= P_iso] = 1
        timed_out = False
        for i in range(0, n, CHUNK):
            if time.time() - t_start > max_seconds:          # time cap inside a round, not only between rounds
                timed_out = True
                break
            sl = slice(i, min(n, i + CHUNK))
            idx = np.arange(sl.start, sl.stop)
            S = cp[sl, None] * x[None, :] + cq[sl, None]
            d = ax[None, :] * hp + hq
            Smin = np.maximum(np.abs(S) - d, 0.0)
            Smax = np.abs(S) + d
            if full_ok:
                full = (Smin >= 2 * R2).all(axis=1) & (reason[sl] == 0)
                reason[idx[full]] = 2
            todo = reason[sl] == 0
            if not todo.any():
                continue
            j = idx[todo]
            evals += len(j)
            St, Smin_t, Smax_t = S[todo], Smin[todo], Smax[todo]
            dd = np.broadcast_to(d, St.shape)
            f = feats(St, ec)
            L, b, r = profile_z0(Ac * f["phi"], y)
            gp = Ac * (r * f["phi_s"] * x[None, :]).mean(axis=1)
            gq = Ac * (r * f["phi_s"]).mean(axis=1)
            gA = (r * f["phi"]).mean(axis=1)
            ge = Ac * (r * f["phi_e"]).mean(axis=1)
            sd = sup_derivs(Smin_t, Smax_t, e_lo, e_hi)
            # per-cell shift c(ε) = φ_ε(q_c) (absorbed by the profiled b): the ε-terms need only |φ_ε(σ_i) − φ_ε(q_c)|
            qc_ = cq[j][:, None]
            dsig = np.abs(St - qc_) + dd
            hl, hh_ = np.minimum(St - dd, qc_), np.maximum(St + dd, qc_)          # hull of σ_i's range and q_c
            Hmin = np.where((hl <= 0) & (hh_ >= 0), 0.0, np.minimum(np.abs(hl), np.abs(hh_)))
            sh = sup_derivs(Hmin, np.maximum(np.abs(hl), np.abs(hh_)), e_lo, e_hi)
            ce = np.minimum(sd["e"], sh["se"] * dsig)
            cee = np.minimum(sd["ee"], sh["see"] * dsig)
            T = A_hi * sd["ss"] * dd ** 2 + 2 * sd["s"] * dd * hA + 2 * A_hi * sd["se"] * dd * he \
                + 2 * ce * hA * he + A_hi * cee * he * he
            curv = 0.5 * T.mean(axis=1)
            D = (L - br["Lb"]) - np.abs(gp) * hp - np.abs(gq) * hq - np.abs(gA - br["gAb"]) * hA \
                - np.abs(ge - br["geb"]) * he - curv - br["curv_b"] - 2 * SAFETY
            Dlo[j] = D
            okp = D > tau
            reason[j[okp]] = 3
            if (~okp).any():
                t_ = np.nonzero(~okp)[0]
                plo, phi_ = phi_range(St[t_] - dd[t_], St[t_] + dd[t_], e_lo, e_hi)
                fv = favourable(plo, phi_, y, A_lo, A_hi)
                hit = fv > BU + tau
                reason[j[t_[hit]]] = 6
        done = reason > 0
        if done.any():
            if (reason == 3).any():
                min_D = min(min_D, float(Dlo[reason == 3].min()))
            for k_, nm in ((1, "iso"), (2, "full"), (3, "pert"), (6, "fav")):
                stats[nm] += int((reason == k_).sum())
            if record:
                leaves.append(np.rec.fromarrays([cp[done], cq[done], np.full(done.sum(), hp), np.full(done.sum(), hq),
                                                 reason[done], np.where(reason[done] == 3, Dlo[done], np.nan)],
                                                names="p,q,hp,hq,reason,D"))
        keep = ~done
        info = {"eps": [e_lo, e_hi], "A": [A_lo, A_hi], "regime": regime, "centre": [pc, qc], "branch": br,
                "P_iso": P_iso, "full_ok": bool(full_ok), "localisation_slack": loc_slack, "domain_p_max": Pmax,
                "domain_q": [q_lo, q_hi], "rounds": rounds, "peak_cells": peak, "evaluations": evals,
                "final_h": max(hp, hq), "stats": dict(stats), "min_D_pert": min_D}
        if not keep.any() and not timed_out:
            return {"certified": True, **info}, leaves
        info["timed_out"] = bool(timed_out or time.time() - t_start > max_seconds)
        if timed_out or rounds >= max_rounds or 4 * keep.sum() > max_cells or time.time() - t_start > max_seconds:
            w = int(np.argmax(np.where(keep, Dlo, -np.inf)))
            wl = int(np.argmin(np.where(keep, np.where(np.isfinite(Dlo), Dlo, np.inf), np.inf)))
            return {"certified": False, "live": int(keep.sum()),
                    "worst_live": {"p": float(cp[wl]), "q": float(cq[wl]), "D": float(Dlo[wl])}, **info}, leaves
        cp, cq = cp[keep], cq[keep]
        hp, hq = hp / 2, hq / 2
        cp = np.concatenate([cp - hp, cp - hp, cp + hp, cp + hp])
        cq = np.concatenate([cq - hq, cq + hq, cq - hq, cq + hq])
        peak = max(peak, len(cp))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30          # macOS: bytes


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ interval helpers (numpy, outward by one ulp per op)
def _out(lo, hi):
    return np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)


def imul(alo, ahi, blo, bhi):
    c = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return _out(c.min(axis=0), c.max(axis=0))


def isq(lo, hi):
    a, b = lo * lo, hi * hi
    mn = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(a, b))
    return _out(mn, np.maximum(a, b))


def iscal(c, lo, hi):
    """float scalar or array c times an interval."""
    a, b = c * lo, c * hi
    return _out(np.minimum(a, b), np.maximum(a, b))


def isig(lo, hi):
    return _out(_sigmoid(lo), _sigmoid(hi))


def idsig(lo, hi):
    sl, sh = _sigmoid(lo), _sigmoid(hi)
    vl, vh = sl * (1 - sl), sh * (1 - sh)
    mx = np.where((lo <= 0) & (hi >= 0), 0.25, np.maximum(vl, vh))
    return _out(np.minimum(vl, vh), mx)


def feat_intervals(Sc, d, e_lo, e_hi):
    """Per point intervals of φ, φ_σ, φ_σσ over σ ∈ Sc ± d, ε ∈ [e_lo, e_hi] (mean-value forms around (Sc, ε_c))."""
    ec, he = 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)
    f = feats(Sc, ec, ("phi", "phi_s", "phi_ss"))
    Smin = np.maximum(np.abs(Sc) - d, 0.0); Smax = np.abs(Sc) + d
    sd = sup_derivs(Smin, Smax, e_lo, e_hi)
    r0 = sd["s"] * d + sd["e"] * he
    r1 = sd["ss"] * d + sd["se"] * he
    r2 = sd["sss"] * d + sd["sse"] * he
    tol = 1e-12
    return ((f["phi"] - r0 - tol * (1 + np.abs(f["phi"])), f["phi"] + r0 + tol * (1 + np.abs(f["phi"]))),
            (f["phi_s"] - r1 - tol * (1 + np.abs(f["phi_s"])), f["phi_s"] + r1 + tol * (1 + np.abs(f["phi_s"]))),
            (f["phi_ss"] - r2 - tol * (1 + np.abs(f["phi_ss"])), f["phi_ss"] + r2 + tol * (1 + np.abs(f["phi_ss"]))))


def _root_rows(z, y):
    """Float root of mean σ(z + b) = ȳ per row (bisection, 200 steps)."""
    ybar = y.mean()
    lo = math.log(ybar / (1 - ybar)) - z.max(axis=1) - 1.0
    hi = math.log(ybar / (1 - ybar)) - z.min(axis=1) + 1.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        F = _sigmoid(z + m[:, None]).mean(axis=1) - ybar
        lo, hi = np.where(F < 0, m, lo), np.where(F < 0, hi, m)
    return lo, hi


def b_bracket(z0lo, z0hi, y):
    """b* ∈ [root(z0hi), root(z0lo)] for every z0 in the per-point intervals (F(b) = mean σ(z0 + b) − ȳ is increasing in b
    and in each z0)."""
    blo, _ = _root_rows(z0hi, y)
    _, bhi = _root_rows(z0lo, y)
    return blo - 1e-12 * (1 + np.abs(blo)), bhi + 1e-12 * (1 + np.abs(bhi))


# ------------------------------------------------------------------ PD on the 0.05 box (interval Hessian of L(p, q, b))
def pd_unit(e_lo, e_hi, A_lo, A_hi, centre, x, y, n0=4, max_depth=5, rho_u=RHO_U, max_boxes=200_000):
    pc, qc = centre
    h = rho_u / n0
    g = (np.arange(n0) + 0.5) * 2 * h - rho_u
    P, Q = np.meshgrid(pc + g, qc + g, indexing="ij")
    boxes = [(P.ravel(), Q.ravel(), h)]
    ax = np.abs(x)
    nbox, worst, depth = 0, math.inf, 0
    Alo_, Ahi_ = np.full(1, A_lo), np.full(1, A_hi)
    while boxes:
        bp, bq, hh = boxes.pop()
        if BOX_CHUNK and len(bp) > BOX_CHUNK:            # bounded working set: at most BOX_CHUNK boxes x n points
            boxes.append((bp[BOX_CHUNK:], bq[BOX_CHUNK:], hh))
            bp, bq = bp[:BOX_CHUNK], bq[:BOX_CHUNK]
        nbox += len(bp)
        if nbox > max_boxes:
            return {"pd_ok": False, "boxes": nbox, "stopped": "max_boxes"}
        Sc = bp[:, None] * x[None, :] + bq[:, None]
        d = ax[None, :] * hh + hh
        (f0l, f0h), (f1l, f1h), (f2l, f2h) = feat_intervals(Sc, d, e_lo, e_hi)
        z0l, z0h = imul(f0l, f0h, A_lo, A_hi)
        bl, bh = b_bracket(z0l, z0h, y)
        zl, zh = z0l + bl[:, None], z0h + bh[:, None]
        sl, sh = isig(zl, zh)
        rl, rh = sl - y[None, :], sh - y[None, :]
        dl, dh = idsig(zl, zh)
        vq = imul(f1l, f1h, A_lo, A_hi)
        vp = imul(vq[0], vq[1], np.minimum(x, x)[None, :], np.maximum(x, x)[None, :])
        wq = imul(f2l, f2h, A_lo, A_hi)
        wp = iscal(x[None, :], *wq)
        wpp = iscal((x * x)[None, :], *wq)
        one = (np.ones_like(zl), np.ones_like(zl))
        V = [vp, vq, one]
        Wm = [[wpp, wp, None], [wp, wq, None], [None, None, None]]
        Hl = np.zeros((len(bp), 3, 3)); Hh = np.zeros((len(bp), 3, 3))
        for i in range(3):
            for j in range(i, 3):
                pr = isq(*V[i]) if i == j else imul(*V[i], *V[j])
                tl, th = imul(dl, dh, *pr)
                if Wm[i][j] is not None:
                    ul, uh = imul(rl, rh, *Wm[i][j])
                    tl, th = tl + ul, th + uh
                Hl[:, i, j] = Hl[:, j, i] = tl.mean(axis=1)
                Hh[:, i, j] = Hh[:, j, i] = th.mean(axis=1)
        M = 0.5 * (Hl + Hh); R = 0.5 * (Hh - Hl)
        lam = np.linalg.eigvalsh(M)[:, 0] - np.sqrt((R ** 2).sum(axis=(1, 2))) - 1e-12
        ok = lam > 0
        if ok.any():
            worst = min(worst, float(lam[ok].min()))
        if (~ok).any():
            if hh / 2 < rho_u / n0 / 2 ** max_depth:
                return {"pd_ok": False, "boxes": nbox, "fail_at": [float(bp[~ok][0]), float(bq[~ok][0])],
                        "lam_lower_failed": float(lam[~ok].min())}
            fp, fq = bp[~ok], bq[~ok]
            h2 = hh / 2
            boxes.append((np.concatenate([fp - h2, fp - h2, fp + h2, fp + h2]),
                          np.concatenate([fq - h2, fq + h2, fq - h2, fq + h2]), h2))
            depth = max(depth, int(round(math.log2(rho_u / n0 / h2))))
    return {"pd_ok": True, "boxes": nbox, "lam_lower": worst, "max_depth": depth}


# ------------------------------------------------------------------ ring: one gradient component nonzero on every box
def ring_unit(e_lo, e_hi, A_lo, A_hi, centre, x, y, side=0.0125, max_depth=5, rho_in=RHO_U, rho_out=RHO,
              max_boxes=2_000_000, record=None):
    """record: optional list; passing boxes are appended as (p, q, h, margin, component 0=p/1=q)."""
    pc, qc = centre
    n = int(round(2 * rho_out / side))
    g = -rho_out + (np.arange(n) + 0.5) * side
    P, Q = np.meshgrid(g, g, indexing="ij")
    m = np.maximum(np.abs(P), np.abs(Q)) + side / 2 > rho_in + 1e-12
    boxes = [(pc + P[m], qc + Q[m], side / 2)]
    Ac, ec = 0.5 * (A_lo + A_hi), 0.5 * (e_lo + e_hi)
    hA, he = 0.5 * (A_hi - A_lo), 0.5 * (e_hi - e_lo)
    ax = np.abs(x)
    nbox, worst, depth = 0, math.inf, 0
    while boxes:
        bp, bq, hh = boxes.pop()
        if BOX_CHUNK and len(bp) > BOX_CHUNK:            # bounded working set: at most BOX_CHUNK boxes x n points
            boxes.append((bp[BOX_CHUNK:], bq[BOX_CHUNK:], hh))
            bp, bq = bp[:BOX_CHUNK], bq[:BOX_CHUNK]
        nbox += len(bp)
        if nbox > max_boxes:
            return {"ring_ok": False, "boxes": nbox, "stopped": "max_boxes"}
        L, b, gp, gq, gA, ge, S, f, r = point_eval(bp, bq, Ac, ec, x, y)
        d = ax[None, :] * hh + hh
        Smin = np.maximum(np.abs(S) - d, 0.0); Smax = np.abs(S) + d
        sd = sup_derivs(Smin, Smax, e_lo, e_hi)
        fe = feats(S, ec, ("phi_e",))["phi_e"]
        vq_c = Ac * f["phi_s"]; vp_c = vq_c * x[None, :]; vA_c = f["phi"]; ve_c = Ac * fe
        var_q = hA * sd["s"] + A_hi * (sd["ss"] * d + sd["se"] * he)
        dv = {"p": np.abs(vp_c - np.median(vp_c, axis=1, keepdims=True)) + ax[None, :] * var_q,
              "q": np.abs(vq_c - np.median(vq_c, axis=1, keepdims=True)) + var_q,
              "A": np.abs(vA_c - np.median(vA_c, axis=1, keepdims=True)) + sd["s"] * d + sd["e"] * he,
              "e": np.abs(ve_c - np.median(ve_c, axis=1, keepdims=True)) + hA * sd["e"] + A_hi * (sd["se"] * d + sd["ee"] * he)}
        Qv = {k: (v ** 2).mean(axis=1) for k, v in dv.items()}
        zsec = {("p", "p"): A_hi * sd["ss"] * ax ** 2, ("p", "q"): A_hi * sd["ss"] * ax, ("p", "A"): sd["s"] * ax,
                ("p", "e"): A_hi * sd["se"] * ax, ("q", "q"): A_hi * sd["ss"], ("q", "A"): sd["s"], ("q", "e"): A_hi * sd["se"]}
        hs = {"p": hh, "q": hh, "A": hA, "e": he}
        okc = np.zeros(len(bp), bool)
        marg = np.full(len(bp), -np.inf)
        for k, gk in (("p", gp), ("q", gq)):
            var = np.zeros(len(bp))
            for a_ in ("p", "q", "A", "e"):
                key = (k, a_) if (k, a_) in zsec else (a_, k)
                Mka = 0.25 * np.sqrt(Qv[k] * Qv[a_]) + zsec[key].mean(axis=1)
                var += Mka * hs[a_]
            mk = np.abs(gk) - var - SAFETY
            marg = np.maximum(marg, mk)
            okc |= mk > 0
        if okc.any():
            worst = min(worst, float(marg[okc].min()))
            if record is not None:
                record.append(np.stack([bp[okc], bq[okc], np.full(okc.sum(), hh), marg[okc]], axis=1))
        if (~okc).any():
            if hh / 2 < side / 2 / 2 ** max_depth:
                return {"ring_ok": False, "boxes": nbox, "fail_at": [float(bp[~okc][0]), float(bq[~okc][0])]}
            fp, fq = bp[~okc], bq[~okc]
            h2 = hh / 2
            boxes.append((np.concatenate([fp - h2, fp - h2, fp + h2, fp + h2]),
                          np.concatenate([fq - h2, fq + h2, fq - h2, fq + h2]), h2))
            depth = max(depth, int(round(math.log2(side / 2 / h2))))
    return {"ring_ok": True, "boxes": nbox, "grad_margin_lower": worst, "max_depth": depth}


# ------------------------------------------------------------------ switch: parametric Krawczyk with ε as an interval
def _iv(lo, hi):
    return (np.atleast_1d(np.asarray(lo, float)), np.atleast_1d(np.asarray(hi, float)))


def _iadd(a, b):
    return _out(a[0] + b[0], a[1] + b[1])


def _isub(a, b):
    return _out(a[0] - b[1], a[1] - b[0])


def _imean(a):
    return _out(np.atleast_1d(a[0].mean(axis=-1)), np.atleast_1d(a[1].mean(axis=-1)))


def _sig_interval(P, Q, xs):
    """σ = p x + q over the box P × Q (intervals), per x in xs: centre and half-width."""
    lo, hi = imul(P[0], P[1], xs, xs)
    lo, hi = lo + Q[0], hi + Q[1]
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def system_iv(X, A_par, e_lo, e_hi, x, y, with_gap, want_J=True):
    P, Q, B = X[0], X[1], X[2]
    Ai = X[3] if with_gap else A_par
    Sc, d = _sig_interval(P, Q, x)
    (f0l, f0h), (f1l, f1h), (f2l, f2h) = feat_intervals(Sc, d, e_lo, e_hi)
    z = imul(f0l, f0h, Ai[0], Ai[1]); z = (z[0] + B[0], z[1] + B[1])
    s_ = isig(*z)
    res = (s_[0] - y, s_[1] - y)
    vq = imul(f1l, f1h, Ai[0], Ai[1])
    vp = iscal(x, *vq)
    vb = (np.ones_like(x), np.ones_like(x))
    V = [vp, vq, vb]
    F = [_imean(imul(*res, *v)) for v in V]
    k = 4 if with_gap else 3
    if with_gap:
        So_c, So_d = _sig_interval(P, Q, np.array([-1.2])); Si_c, Si_d = _sig_interval(P, Q, np.array([0.8]))
        (o0l, o0h), (o1l, o1h), _ = feat_intervals(So_c, So_d, e_lo, e_hi)
        (i0l, i0h), (i1l, i1h), _ = feat_intervals(Si_c, Si_d, e_lo, e_hi)
        F.append(_isub((o0l, o0h), (i0l, i0h)))
    if not want_J:
        return F
    ds = idsig(*z)
    wq = imul(f2l, f2h, Ai[0], Ai[1]); wp = iscal(x, *wq); wpp = iscal(x * x, *wq)
    Wm = [[wpp, wp, None], [wp, wq, None], [None, None, None]]
    J = [[None] * k for _ in range(3)]
    for i in range(3):
        for j in range(3):
            t = imul(*ds, *(isq(*V[i]) if i == j else imul(*V[i], *V[j])))
            if Wm[i][j] is not None:
                t = _iadd(t, imul(*res, *Wm[i][j]))
            J[i][j] = _imean(t)
        if with_gap:
            dA = [iscal(x, f1l, f1h), (f1l, f1h), (np.zeros_like(x), np.zeros_like(x))]
            t = imul(*ds, *imul(f0l, f0h, *V[i]))
            t = _iadd(t, imul(*res, *dA[i]))
            J[i][3] = _imean(t)
    if with_gap:
        g0 = _isub(iscal(-1.2, o1l, o1h), iscal(0.8, i1l, i1h))
        g1 = _isub((o1l, o1h), (i1l, i1h))
        z0 = (np.zeros(1), np.zeros(1))
        J.append([g0, g1, z0, z0])
    return F, J


def krawczyk_iv(fun, xt, rad):
    k = len(xt)
    X = [_iv(xt[i] - rad[i], xt[i] + rad[i]) for i in range(k)]
    F0 = fun([_iv(t, t) for t in xt], False)
    _, J = fun(X, True)
    Jm = np.array([[0.5 * float(J[i][j][0][0] + J[i][j][1][0]) for j in range(k)] for i in range(k)])
    if not np.isfinite(Jm).all() or abs(np.linalg.det(Jm)) < 1e-300:
        return False, [(-math.inf, math.inf)] * k
    try:
        Y = np.linalg.inv(Jm)
    except np.linalg.LinAlgError:
        return False, [(-math.inf, math.inf)] * k
    Kb = []
    for i in range(k):
        lo, hi = np.array([xt[i]]), np.array([xt[i]])
        for j in range(k):
            a, b = iscal(-float(Y[i, j]), F0[j][0], F0[j][1])
            lo, hi = lo + a, hi + b
        for j in range(k):
            cl, ch = np.array([1.0 if i == j else 0.0]), np.array([1.0 if i == j else 0.0])
            for l_ in range(k):
                a, b = iscal(-float(Y[i, l_]), J[l_][j][0], J[l_][j][1])
                cl, ch = cl + a, ch + b
            a, b = imul(cl, ch, X[j][0] - xt[j], X[j][1] - xt[j])
            lo, hi = lo + a, hi + b
        lo, hi = _out(lo, hi)
        Kb.append((float(lo[0]), float(hi[0])))
    ok = all(Kb[i][0] > X[i][0][0] and Kb[i][1] < X[i][1][0] for i in range(k))
    return ok, Kb


def _crit(e_lo, e_hi):
    """σ* = arccos(1/a)/√ε (first critical point of φ_ε) and σ** = (2π − arccos(1/a))/√ε over the slab."""
    # arccos(1/(1+ε))/√ε is analytic in ε with limit √2 at ε = 0 (series below 1e-6); (2π − β)/√ε → ∞
    f = lambda e: (math.acos(1 / (1 + e)) / math.sqrt(e)) if e > 1e-6 else R2 * (1 - 5 * e / 12)
    g = lambda e: ((2 * math.pi - math.acos(1 / (1 + e))) / math.sqrt(e)) if e > 0 else math.inf
    return min(f(e_lo), f(e_hi)), max(f(e_lo), f(e_hi)), min(g(e_lo), g(e_hi))


def gap_on_box(P, Q, e_lo, e_hi, delta=0.01):
    """(valid active pair, G interval) over the box: G = φ(q − 1.2p) − φ(q + 0.8p) when the window extrema are those."""
    smin, smax, s2 = _crit(e_lo, e_hi)
    def rng(x1, x2):
        c = [pp * xx + qq for pp in P for xx in (x1, x2) for qq in Q]
        return min(c), max(c)
    il, ih = rng(-0.8, 0.8); nl, nh = rng(-2.0, -1.2); ol, oh = rng(1.2, 2.0)
    geo = (il > -smin + delta) and (ih < s2 - delta) and (nh < smin - delta) and (nl > -s2 + delta) \
        and (ol > smax + delta) and (oh < s2 - delta)
    def phi_iv(xs):
        Sc, d = _sig_interval(_iv(*P), _iv(*Q), np.array([xs]))
        (l0, h0), _, _ = feat_intervals(Sc, d, e_lo, e_hi)
        return float(l0[0]), float(h0[0])
    I_act, I_oth = phi_iv(0.8), phi_iv(-0.8)
    O_act, O_m2, O_p1 = phi_iv(-1.2), phi_iv(-2.0), phi_iv(1.2)
    dom = (I_act[0] > I_oth[1]) and (O_act[1] < O_m2[0]) and (O_act[1] < O_p1[0])
    return bool(geo and dom), (O_act[0] - I_act[1], O_act[1] - I_act[0])


def switch_float(e, x, y, v0):
    """Float Newton for (p, q, A) with ∇_θ L* = 0 and G = 0 on the active pair, at ε."""
    v = np.array(v0, float)
    def Fv(v):
        _, b, gp, gq, *_ = point_eval([v[0]], [v[1]], v[2], e, x, y)
        f = feats(np.array([v[1] - 1.2 * v[0], v[1] + 0.8 * v[0]]), e, ("phi",))["phi"]
        return np.array([gp[0], gq[0], f[0] - f[1]]), b[0]
    for _ in range(40):
        F, b = Fv(v)
        Jm = np.empty((3, 3))
        for j in range(3):
            dv = np.zeros(3); dv[j] = 1e-7
            Jm[:, j] = (Fv(v + dv)[0] - F) / 1e-7
        st = np.linalg.solve(Jm, F)
        v = v - st
        if np.abs(st).max() < 1e-13:
            break
    return v, Fv(v)[1]


def switch_unit(e_lo, e_hi, J_tube, x, y, v0, rz=(6e-4, 6e-4, 2e-4, 5e-4), max_boxes=4000):
    ec = 0.5 * (e_lo + e_hi)
    vs, bs = switch_float(ec, x, y, v0)
    zf = [vs[0], vs[1], bs, vs[2]]
    funZ = lambda X, Jf=True: system_iv(X, None, e_lo, e_hi, x, y, True, Jf)
    rad = list(rz)
    okZ, KZ = False, None
    for _ in range(12):
        okZ, KZ = krawczyk_iv(funZ, zf, rad)
        if okZ:
            break
        rad = [max(rad[i], 1.5 * max(abs(KZ[i][0] - zf[i]), abs(KZ[i][1] - zf[i]))) for i in range(4)]
        if not all(np.isfinite(rad)) or max(r_ / r0 for r_, r0 in zip(rad, rz)) > 50:
            break
    if not okZ:
        return {"switch_box_ok": False, "rad": rad, "unique_sign_change": False}
    XZ = [(zf[i] - rad[i], zf[i] + rad[i]) for i in range(4)]
    actZ, _ = gap_on_box(XZ[0], XZ[1], e_lo, e_hi)
    ZA = XZ[3]
    edges = [J_tube[0]]
    while edges[-1] < ZA[0] - 1e-15:
        edges.append(min(ZA[0], edges[-1] + min(2e-3, max(2e-5, 0.25 * (ZA[0] - edges[-1])))))
    nin = max(1, int(math.ceil((ZA[1] - ZA[0]) / 2e-5)))
    edges += [ZA[0] + (ZA[1] - ZA[0]) * i / nin for i in range(1, nin + 1)]
    while edges[-1] < J_tube[1] - 1e-15:
        edges.append(min(J_tube[1], edges[-1] + min(2e-3, max(2e-5, 0.25 * (edges[-1] - ZA[1]) + 2e-5))))
    queue = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    rows = []
    v = np.array([vs[0], vs[1]])
    while queue and len(rows) < max_boxes:
        a_lo, a_hi = queue.pop(0)
        Ac = 0.5 * (a_lo + a_hi)
        v = branch_point(Ac, ec, x, y, start=v)
        _, bb, *_ = point_eval([v[0]], [v[1]], Ac, ec, x, y)
        xt = [v[0], v[1], bb[0]]
        Aint = _iv(a_lo, a_hi)
        fun = lambda Xb, Jf=True: system_iv(Xb, Aint, e_lo, e_hi, x, y, False, Jf)
        rd = [1e-6 + 0.5 * (a_hi - a_lo) + 2 * (e_hi - e_lo)] * 3
        ok, Kb = False, None
        for _ in range(12):
            ok, Kb = krawczyk_iv(fun, xt, rd)
            if ok:
                break
            rd = [max(rd[i], 1.5 * max(abs(Kb[i][0] - xt[i]), abs(Kb[i][1] - xt[i]))) for i in range(3)]
            if not all(np.isfinite(rd)) or max(rd) > 0.05:
                break
        within = ZA[0] - 1e-15 <= a_lo and a_hi <= ZA[1] + 1e-15
        if ok:
            act, (glo, ghi) = gap_on_box(Kb[0], Kb[1], e_lo, e_hi)
        else:
            act, glo, ghi = False, np.nan, np.nan
        sign = (-1 if ghi < 0 else 1 if glo > 0 else 0) if (ok and act) else None
        inZ = ok and all(Kb[i][0] > XZ[i][0] and Kb[i][1] < XZ[i][1] for i in range(3))
        if ok and not within and sign == 0 and a_hi - a_lo > 1e-7:
            m_ = 0.5 * (a_lo + a_hi)
            queue[:0] = [(a_lo, m_), (m_, a_hi)]
            continue
        rows.append({"A_lo": a_lo, "A_hi": a_hi, "ok": ok, "act": act, "G": [glo, ghi], "sign": sign,
                     "within_Z_A": within, "in_Z": inZ, "box": Kb})
    left = [r_ for r_ in rows if r_["A_hi"] <= ZA[0] + 1e-15]
    right = [r_ for r_ in rows if r_["A_lo"] >= ZA[1] - 1e-15]
    inside = [r_ for r_ in rows if r_ not in left and r_ not in right]
    unique = bool(okZ and actZ and not queue and all(r_["ok"] for r_ in rows) and all(r_["sign"] == -1 for r_ in left)
                  and all(r_["sign"] == 1 for r_ in right) and all(r_["in_Z"] for r_ in inside))
    return {"switch_box_ok": okZ, "active_pair_Z": actZ, "Z": XZ, "Z_A": list(ZA), "n_boxes": len(rows),
            "n_left": len(left), "n_inside": len(inside), "n_right": len(right), "unique_sign_change": unique,
            "seconds": None,
            "max_G_hi_left": max((r_["G"][1] for r_ in left), default=None),
            "min_G_lo_right": min((r_["G"][0] for r_ in right), default=None),
            "boxes_U_extent": [min(min(r_["box"][0][0] for r_ in rows if r_["ok"]), 1e9),
                               max(max(r_["box"][0][1] for r_ in rows if r_["ok"]), -1e9)] if rows else None,
            "float_switch": [float(vs[0]), float(vs[1]), float(vs[2]), float(bs)], "J_tube": list(J_tube),
            "rows_boxes": [{"A_lo": r_["A_lo"], "A_hi": r_["A_hi"], "box": r_["box"]} for r_ in rows if r_["ok"]]}


# ------------------------------------------------------------------ timing drivers
TUBE = (0.66 / 0.68544523757565, 0.71 / 0.68544523757565)     # U1's A-window [0.66, 0.71], relative to A*


def pop():
    p = os.environ.get("U2_POP")
    if p:
        d = np.load(p)
        return d["x"], d["y"]
    from .limit_bnb import population
    return population()


def tube(ec, x, y, v0=(1.668, 1.369, 0.6855)):
    vs, bs = switch_float(ec, x, y, v0)
    return vs, (vs[2] * TUBE[0], vs[2] * TUBE[1])


def probe(e_lo, e_hi, regime, fracs, n_sub, x, y, cons, export=None, do_links=False, max_rounds=8, max_cells=1_500_000,
          max_seconds=600.0, store=None, prefix="u"):
    """Outer units for a few A-subintervals of the slab's tube (fractions in [0, 1] of the tube; n_sub sub-intervals
    in the tube), with timing, peak RSS, leaf counts; optional leaf export (compressed npz + SHA-256).
    store: a u2_store.UnitStore; each A-sub-interval (outer + ring + PD + export) is one unit, stored atomically when it
    completes and skipped on a restart."""
    from .u2_store import UnitStore, run_units
    store = store if store is not None else UnitStore(OUT / "units" / "_scratch")
    ec = 0.5 * (e_lo + e_hi)
    vs, J = tube(ec, x, y)
    w = (J[1] - J[0]) / n_sub
    ks = sorted({min(n_sub - 1, int(fr * n_sub)) for fr in fracs})
    units = [(f"{prefix}_k{k:03d}", (lambda k=k: _probe_unit(k, e_lo, e_hi, regime, n_sub, J, w, vs, x, y, cons, export,
                                                             do_links, max_rounds, max_cells, max_seconds))) for k in ks]
    out = run_units(store, units)
    return [out[key] for key, _ in units]


def probe_wide(e_lo, e_hi, regime, A_rel, x, y, cons, store, prefix, max_seconds=300.0):
    """Outer-exclusion units at A = r·A_ε (r in A_rel, e.g. 0.5 … 2), each of the standard sub-interval width
    A_ε·(TUBE[1] − TUBE[0])/40, centred on the branch at that A (cost sample for a wider window; status not built)."""
    from .u2_store import run_units
    ec = 0.5 * (e_lo + e_hi)
    vs, J = tube(ec, x, y)
    w = (J[1] - J[0]) / 40

    def one(r):
        A_lo, A_hi = r * vs[2] - w / 2, r * vs[2] + w / 2
        c = branch_point(r * vs[2], ec, x, y, start=(vs[0], vs[1]))
        t0 = time.time()
        res, _ = outer_unit(e_lo, e_hi, A_lo, A_hi, tuple(c), x, y, regime, cons, record=False, max_rounds=8,
                            max_seconds=max_seconds)
        out = {k: res[k] for k in ("certified", "rounds", "peak_cells", "evaluations", "stats", "min_D_pert",
                                    "timed_out", "worst_live") if k in res}
        out.update({"A_rel": r, "A": [A_lo, A_hi], "centre": list(map(float, c)), "seconds": time.time() - t0,
                    "n_leaves": sum(res["stats"][s_] for s_ in ("iso", "full", "pert", "fav")), "rss_gb": rss_gb()})
        print("   wide", e_lo, e_hi, regime, r, out["certified"], out["n_leaves"], round(out["seconds"], 1), flush=True)
        return out
    res = run_units(store, [(f"{prefix}_r{r:.3f}", (lambda r=r: one(r))) for r in A_rel])
    return list(res.values())


def _probe_unit(k, e_lo, e_hi, regime, n_sub, J, w, vs, x, y, cons, export, do_links, max_rounds, max_cells, max_seconds):
    ec = 0.5 * (e_lo + e_hi)
    if True:
        A_lo, A_hi = J[0] + k * w, J[0] + (k + 1) * w
        c = branch_point(0.5 * (A_lo + A_hi), ec, x, y, start=(vs[0], vs[1]))
        t0 = time.time()
        res, lv = outer_unit(e_lo, e_hi, A_lo, A_hi, tuple(c), x, y, regime, cons, record=export is not None,
                             max_rounds=max_rounds, max_cells=max_cells, max_seconds=max_seconds)
        res["seconds"] = time.time() - t0
        res["k"], res["n_sub"], res["tube"], res["A_switch_float"] = k, n_sub, list(J), float(vs[2])
        res["n_leaves"] = sum(res["stats"][s] for s in ("iso", "full", "pert", "fav"))
        if export is not None and lv:
            arr = np.concatenate(lv)
            path = OUT / f"{export}_k{k:03d}.npz"
            np.savez_compressed(path, **{f: np.ascontiguousarray(arr[f]) for f in arr.dtype.names})
            (OUT / f"{export}_k{k:03d}.meta.json").write_text(json.dumps({"eps": [e_lo, e_hi], "A": [A_lo, A_hi],
                "centre": list(map(float, c)), "regime": regime, "fields": list(arr.dtype.names)}))
            res["export"] = {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha(path),
                             "leaves": int(len(arr)), "bytes_per_leaf": path.stat().st_size / max(1, len(arr))}
        if do_links:
            # ring and PD on closed ε-sub-slabs, halved on failure down to 1e-4 (each sub-slab with the unit's θ_c)
            def ring_run(lo, hi, cap):
                rec_ = [] if export is not None else None
                r_ = ring_unit(lo, hi, A_lo, A_hi, tuple(c), x, y, record=rec_)
                return {"certified": r_["ring_ok"], "rec": rec_, "sub": [lo, hi], **r_}
            def pd_run(lo, hi, cap):
                r_ = pd_unit(lo, hi, A_lo, A_hi, tuple(c), x, y)
                return {"certified": r_["pd_ok"], **r_}
            for nm, fn in (("ring", ring_run), ("pd", pd_run)):
                t1 = time.time()
                lv_ = certify_split(e_lo, e_hi, fn, 600.0, 1e-4)
                if nm == "ring":
                    rb = [np.c_[b_, np.full((len(b_), 2), r_["sub"])] for _, _, c_, r_ in lv_ if c_ and r_.get("rec")
                          for b_ in r_["rec"]]
                res[nm] = {"ok": all(c_ for _, _, c_, _ in lv_), "n_subslabs": len(lv_),
                           "subslab_widths": sorted({hi - lo for lo, hi, _, _ in lv_}),
                           "boxes": sum(r_.get("boxes", 0) for _, _, _, r_ in lv_),
                           "margin_lower": min((r_.get("grad_margin_lower", r_.get("lam_lower", math.inf))
                                                for _, _, c_, r_ in lv_ if c_), default=None),
                           "seconds": time.time() - t1}
            if rb:
                path = OUT / f"{export}_k{k:03d}_ring.npz"
                np.savez_compressed(path, boxes=np.concatenate(rb))
                res["ring"]["export"] = {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha(path),
                                         "boxes": int(sum(len(b_) for b_ in rb))}
        res["rss_gb"] = rss_gb()
        print("   unit", e_lo, e_hi, regime, k, n_sub, res["certified"], res["n_leaves"], round(res["seconds"], 1),
              res.get("worst_live"), flush=True)
        return res


def run_probes(spec_file, out_name):
    """spec_file: JSON list of {e_lo, e_hi, regime, fracs, n_sub, export?, links?, switch?}.  Resumable at the UNIT
    level (u2_store): every A-sub-interval unit and every switch sub-slab is stored atomically on completion under
    u2_sample/units/<out_name>/, and a restart skips them; the per-spec JSONL line is a summary written last."""
    from .u2_store import UnitStore
    x, y = pop()
    cons = json.loads((RESULTS / "uniform_eps" / "constants.json").read_text())
    specs = json.loads(Path(spec_file).read_text())
    out = OUT / f"{out_name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["spec_index"] for l in out.read_text().splitlines() if l.strip()}
    for i, sp in enumerate(specs):
        if i in done:
            continue
        t0 = time.time()
        rec = {"spec_index": i, "spec": sp}
        try:
            ust = UnitStore(OUT / "units" / out_name)
            if sp.get("A_rel"):                           # item 4: widening the window (outer exclusion only, sampled)
                rec["wide"] = probe_wide(sp["e_lo"], sp["e_hi"], sp["regime"], sp["A_rel"], x, y, cons, ust,
                                         f"s{i:03d}_wide", sp.get("max_seconds", 300.0))
            elif sp.get("outer", True):
                rec["outer"] = probe(sp["e_lo"], sp["e_hi"], sp["regime"], sp["fracs"], sp["n_sub"], x, y, cons,
                                     export=sp.get("export"), do_links=sp.get("links", False),
                                     max_rounds=sp.get("max_rounds", 8), max_seconds=sp.get("max_seconds", 300.0),
                                     store=ust, prefix=f"s{i:03d}")
            if sp.get("switch"):
                ec = 0.5 * (sp["e_lo"] + sp["e_hi"])
                vs, J = tube(ec, x, y)
                t1 = time.time()
                sw = switch_split(sp["e_lo"], sp["e_hi"], J, x, y, vs, keep_rows=bool(sp.get("export")),
                                  max_seconds=sp.get("switch_seconds", 1800.0), store=ust, prefix=f"s{i:03d}_sw")
                sw["seconds"] = time.time() - t1
                exl = sw.pop("export_leaves", [])
                if sp.get("export") and exl:
                    path = OUT / f"{sp['export']}_switch.json"
                    path.write_text(json.dumps(exl, default=float))
                    sw["export"] = {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
                rec["switch"] = sw
        except Exception as ex:                       # a probe that breaks is recorded, not fatal
            rec["error"] = repr(ex)
        rec["seconds"] = time.time() - t0
        rec["rss_gb"] = rss_gb()
        with out.open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        o = rec.get("outer", [])
        print(i, sp, "cert", [r_["certified"] for r_ in o], "leaves", [r_["n_leaves"] for r_ in o],
              "s", [round(r_["seconds"], 1) for r_ in o], "sw", rec.get("switch", {}).get("unique_sign_change_all"), rec.get("switch", {}).get("n_subslabs"),
              rec.get("error", ""), round(rec["seconds"], 1), "rss", round(rec["rss_gb"], 3), flush=True)


def _cur_rss_gb():
    """Current RSS (ps), GB; ru_maxrss is the process peak."""
    import subprocess
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 2 ** 20


def switch_split(e_lo, e_hi, J_tube, x, y, v0, min_width=1e-5, max_seconds=1800.0, keep_rows=False, store=None,
                 prefix="sw"):
    """switch_unit on closed ε-sub-slabs: a sub-slab whose switch part fails is halved (recursively, to min_width).
    store: each sub-slab's result is a unit (stored atomically, skipped on a restart)."""
    t0 = time.time()
    leaves = []

    def go(lo, hi):
        if time.time() - t0 > max_seconds:
            leaves.append({"eps": [lo, hi], "unique_sign_change": False, "stopped": "time cap"}); return
        key = f"{prefix}_{lo!r}_{hi!r}"
        r = store.get(key) if store is not None else None
        if r is None:
            t1 = time.time()
            r = switch_unit(lo, hi, J_tube, x, y, v0)
            r["seconds"] = time.time() - t1
            if not keep_rows:
                r = {k: v for k, v in r.items() if k != "rows_boxes"}
            r["eps"] = [lo, hi]
            if store is not None:
                store.put(key, r)
        if r.get("unique_sign_change") or (hi - lo) / 2 < min_width:
            leaves.append(r); return
        m_ = 0.5 * (lo + hi)
        go(lo, m_); go(m_, hi)
    go(e_lo, e_hi)
    ok = all(l_.get("unique_sign_change") for l_ in leaves)
    return {"unique_sign_change_all": ok, "n_subslabs": len(leaves), "subslab_widths": sorted({l_["eps"][1] - l_["eps"][0]
            for l_ in leaves}), "boxes_total": sum(l_.get("n_boxes", 0) for l_ in leaves),
            "seconds": time.time() - t0, "leaves": [{k: l_.get(k) for k in ("eps", "unique_sign_change", "n_boxes",
            "Z_A", "switch_box_ok", "active_pair_Z", "stopped", "seconds")} for l_ in leaves],
            "export_leaves": [{"eps": l_["eps"], "Z": l_.get("Z"), "J_tube": l_.get("J_tube"), "rows": l_.get("rows_boxes")}
                              for l_ in leaves if keep_rows and l_.get("unique_sign_change")]}


# ------------------------------------------------------------------ item 3 diagnosis: sheared and first-order forms in ε
# (producer side, float intervals with one-ulp outward steps; producer standard.)  Not used by any certificate yet.
def feat5(Sc, d, e_lo, e_hi):
    """Intervals of φ, φ_σ, φ_σσ, φ_ε, φ_σε over σ ∈ Sc ± d, ε ∈ [e_lo, e_hi] (mean-value radii around (Sc, ε_c))."""
    ec, he = 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)
    f = feats(Sc, ec, ("phi", "phi_s", "phi_ss", "phi_e", "phi_se"))
    Smin = np.maximum(np.abs(Sc) - d, 0.0); Smax = np.abs(Sc) + d
    sd = sup_derivs(Smin, Smax, max(e_lo, 0.0), e_hi)
    rad = {"phi": sd["s"] * d + sd["e"] * he, "phi_s": sd["ss"] * d + sd["se"] * he,
           "phi_ss": sd["sss"] * d + sd["sse"] * he, "phi_e": sd["se"] * d + sd["ee"] * he,
           "phi_se": sd["sse"] * d + sd["see"] * he}
    out = []
    for k in ("phi", "phi_s", "phi_ss", "phi_e", "phi_se"):
        v = f[k]; r = rad[k] + 1e-12 * (1 + np.abs(v))
        out.append(_out(v - r, v + r))
    return out


def system_iv2(X, A_iv, e_lo, e_hi, x, y, with_gap):
    """F = (∂_p L, ∂_q L, ∂_b L[, G]), J = ∂F/∂(p, q, b[, A]), dA = ∂F/∂A (3-var case), dE = ∂F/∂ε, all as intervals
    over the box X (intervals), A (X[3] if with_gap else A_iv) and ε ∈ [e_lo, e_hi]."""
    P, Q, B = X[0], X[1], X[2]
    Ai = X[3] if with_gap else A_iv
    Sc, d = _sig_interval(P, Q, x)
    f0, f1, f2, fe, fse = feat5(Sc, d, e_lo, e_hi)
    z = imul(*f0, Ai[0], Ai[1]); z = (z[0] + B[0], z[1] + B[1])
    s_ = isig(*z); res = (s_[0] - y, s_[1] - y); ds = idsig(*z)
    vq = imul(*f1, Ai[0], Ai[1]); vp = iscal(x, *vq); one = (np.ones_like(x), np.ones_like(x))
    V = [vp, vq, one]
    wq = imul(*f2, Ai[0], Ai[1]); wp = iscal(x, *wq); wpp = iscal(x * x, *wq)
    Wm = [[wpp, wp, None], [wp, wq, None], [None, None, None]]
    dAv = [iscal(x, *f1), f1, None]
    eq = imul(*fse, Ai[0], Ai[1]); dEv = [iscal(x, *eq), eq, None]
    Afe = imul(*fe, Ai[0], Ai[1])
    F, J, dA, dE = [], [], [], []
    for i in range(3):
        F.append(_imean(imul(*res, *V[i])))
        row = []
        for j in range(3):
            t = imul(*ds, *(isq(*V[i]) if i == j else imul(*V[i], *V[j])))
            if Wm[i][j] is not None:
                t = _iadd(t, imul(*res, *Wm[i][j]))
            row.append(_imean(t))
        t = imul(*ds, *imul(*f0, *V[i]))
        if dAv[i] is not None:
            t = _iadd(t, imul(*res, *dAv[i]))
        dA.append(_imean(t))
        if with_gap:
            row.append(dA[-1])
        J.append(row)
        t = imul(*ds, *imul(*Afe, *V[i]))
        if dEv[i] is not None:
            t = _iadd(t, imul(*res, *dEv[i]))
        dE.append(_imean(t))
    if with_gap:
        So = _sig_interval(P, Q, np.array([-1.2])); Si = _sig_interval(P, Q, np.array([0.8]))
        o = feat5(So[0], So[1], e_lo, e_hi); ii = feat5(Si[0], Si[1], e_lo, e_hi)
        F.append(_isub(o[0], ii[0]))
        J.append([_isub(iscal(-1.2, *o[1]), iscal(0.8, *ii[1])), _isub(o[1], ii[1]), (np.zeros(1), np.zeros(1)),
                  (np.zeros(1), np.zeros(1))])
        dE.append(_isub(o[3], ii[3]))
    return {"F": F, "J": J, "dA": dA, "dE": dE}


def _imid(t):
    return 0.5 * (float(t[0][0]) + float(t[1][0]))


def _matvec_iv(J, v):
    """interval matrix × float vector."""
    out = []
    for row in J:
        lo, hi = np.zeros(1), np.zeros(1)
        for Jij, vj in zip(row, v):
            a, b = iscal(float(vj), Jij[0], Jij[1])
            lo, hi = lo + a, hi + b
        out.append(_out(lo, hi))
    return out


def _iv_scale(t, h):
    """[−h, h] · t (interval)."""
    m = max(abs(float(t[0][0])), abs(float(t[1][0])))
    return (np.array([-h * m]), np.array([h * m]))


def _box(xc, r):
    return [_iv(xc[i] - r[i], xc[i] + r[i]) for i in range(len(xc))]


def krawczyk_modes(xc, rad, e_lo, e_hi, x, y, mode, A_iv=None, with_gap=True, A_shear=None):
    """Krawczyk on the switch (with_gap: unknowns p, q, b, A) or on the 3-var branch system (A ∈ A_iv), with ε ∈ [e_lo,
    e_hi] handled by
      'plain' : F evaluated at the point x̃ with ε as an interval (the current form);
      'mv'    : F(x̃, ε_c) + [−h, h]·∂_εF(x̃, E)  (first-order / mean-value form in ε on the same box);
      'shear' : the box follows x̃(ε) = x̃ + (ε − ε_c)·X1, X1 = −J⁻¹∂_εF at the centre (the predicted ε-dependence, e.g.
                A = A*(1 + c_s ε) at first order); only the offset is certified, for every ε in E at once:
                F̃(0; ε) ∈ F(x̃, ε_c) + [−h, h]·(J(Xs, E)·X1 + ∂_εF(Xs, E)), Xs the sheared segment.
    Returns (ok, radii, info)."""
    k = len(xc)
    ec, he = 0.5 * (e_lo + e_hi), 0.5 * (e_hi - e_lo)
    pt = system_iv2([_iv(t, t) for t in xc], A_iv, ec, ec, x, y, with_gap)
    Jm = np.array([[_imid(pt["J"][i][j]) for j in range(k)] for i in range(k)])
    dEm = np.array([_imid(t) for t in pt["dE"]])
    try:
        X1 = -np.linalg.solve(Jm, dEm) if mode == "shear" else np.zeros(k)
    except np.linalg.LinAlgError:
        return False, rad, {"note": "singular"}
    seg = [he * abs(X1[i]) for i in range(k)]
    r = list(rad)
    for it in range(12):
        if mode == "plain":
            F0 = system_iv2([_iv(t, t) for t in xc], A_iv, e_lo, e_hi, x, y, with_gap)["F"]
        else:
            F0 = []
            if mode == "mv":
                T = system_iv2([_iv(t, t) for t in xc], A_iv, e_lo, e_hi, x, y, with_gap)["dE"]
            else:
                g = system_iv2(_box(xc, seg), A_iv, e_lo, e_hi, x, y, with_gap)
                T = [_iadd(a, b) for a, b in zip(_matvec_iv(g["J"], X1), g["dE"])]
            for i in range(k):
                F0.append(_iadd(pt["F"][i], _iv_scale(T[i], he)))
        Xb = _box(xc, [r[i] + seg[i] for i in range(k)])
        Jb = system_iv2(Xb, A_iv, e_lo, e_hi, x, y, with_gap)["J"]
        Jmid = np.array([[_imid(Jb[i][j]) for j in range(k)] for i in range(k)])
        if not np.isfinite(Jmid).all():
            return False, r, {"note": "non-finite J"}
        Y = np.linalg.inv(Jmid)
        ok, K = True, []
        for i in range(k):
            lo, hi = np.zeros(1), np.zeros(1)
            for j in range(k):
                a, b = iscal(-float(Y[i, j]), F0[j][0], F0[j][1]); lo, hi = lo + a, hi + b
            for j in range(k):
                cl, ch = np.array([1.0 if i == j else 0.0]), np.array([1.0 if i == j else 0.0])
                for l_ in range(k):
                    a, b = iscal(-float(Y[i, l_]), Jb[l_][j][0], Jb[l_][j][1]); cl, ch = cl + a, ch + b
                a, b = imul(cl, ch, np.array([-r[j]]), np.array([r[j]])); lo, hi = lo + a, hi + b
            lo, hi = _out(lo, hi)
            K.append((float(lo[0]), float(hi[0])))
            ok = ok and (K[-1][0] > -r[i]) and (K[-1][1] < r[i])
        if ok:
            return True, r, {"iterations": it + 1, "X1": X1.tolist(), "seg": seg}
        r = [max(r[i], 1.5 * max(abs(K[i][0]), abs(K[i][1]))) for i in range(k)]
        if not all(np.isfinite(r)) or max(r[i] / rad[i] for i in range(k)) > 50:
            break
    return False, r, {"X1": X1.tolist(), "seg": seg}


def ring3d(centre3, A_lo, A_hi, e_lo, e_hi, x, y, mode, side=0.0125, rho_in=RHO_U, rho_out=RHO, max_depth=4,
           max_boxes=400_000):
    """No zero of F = ∇_{p,q,b} L in the ring rho_in <= |θ − θ_c(ε, A)|_∞ <= rho_out(+shear), any b, A, ε: boxes in
    (ψ_p, ψ_q, β) are excluded if 0 ∉ F̃_p, F̃_q or F̃_b, with
      F̃ ∈ F(θ_c + ψ, b_c + β; A_c, ε_c) + [−h_ε, h_ε]·T_ε + [−h_A, h_A]·T_A,
      'mv'   : T_ε = ∂_εF, T_A = ∂_AF over the hull (first-order form; θ_c fixed);
      'shear': T_ε = J·X1 + ∂_εF, T_A = J·XA + ∂_AF, X1 = −J⁻¹∂_εF, XA = −J⁻¹∂_AF at the centre (the box follows the
               branch's ε- and A-dependence; the ring is then centred on the moving branch).
    centre3 = (p_c, q_c, b_c) at (A_c, ε_c)."""
    Ac, ec = 0.5 * (A_lo + A_hi), 0.5 * (e_lo + e_hi)
    hA, he = 0.5 * (A_hi - A_lo), 0.5 * (e_hi - e_lo)
    c = np.array(centre3, float)
    pt = system_iv2([_iv(t, t) for t in c], _iv(Ac, Ac), ec, ec, x, y, False)
    Jm = np.array([[_imid(pt["J"][i][j]) for j in range(3)] for i in range(3)])
    X1 = -np.linalg.solve(Jm, np.array([_imid(t) for t in pt["dE"]])) if mode == "shear" else np.zeros(3)
    XA = -np.linalg.solve(Jm, np.array([_imid(t) for t in pt["dA"]])) if mode == "shear" else np.zeros(3)
    drift = np.abs(X1) * he + np.abs(XA) * hA
    ro = rho_out + max(drift[0], drift[1])
    # b-offset range: the whole bracket of b* over the hull, minus the centre
    n = int(np.ceil(2 * ro / side))
    g = -ro + (np.arange(n) + 0.5) * (2 * ro / n)
    hb0 = ro / n
    Pp, Qq = np.meshgrid(g, g, indexing="ij")
    m = np.maximum(np.abs(Pp), np.abs(Qq)) + hb0 > rho_in
    Sc_all, d_all = c[0] + Pp[m], c[1] + Qq[m]
    Sh = Sc_all[:, None] * x[None, :] + d_all[:, None]             # σ = p x + q at the tile centres
    half = np.abs(x)[None, :] * (hb0 + drift[0]) + hb0 + drift[1]
    (zl0, zh0), *_ = feat5(Sh, np.broadcast_to(half, Sh.shape), e_lo, e_hi)
    zl, zh = imul(zl0, zh0, A_lo, A_hi)
    blo, bhi = b_bracket(zl, zh, y)
    Bw = float(max(np.abs(blo - c[2]).max(), np.abs(bhi - c[2]).max())) + drift[2] + 1e-9
    nb = 8
    boxes = [(c[0] + Pp[m][i], c[1] + Qq[m][i], c[2] - Bw + (2 * j + 1) * Bw / nb, hb0, Bw / nb, 0)
             for i in range(int(m.sum())) for j in range(nb)]
    n_eval, left = 0, []
    while boxes:
        bp, bq, bb, hh, hbeta, dep = boxes.pop()
        n_eval += 1
        if n_eval > max_boxes:
            return {"certified": False, "stopped": "max_boxes", "boxes": n_eval}
        if max(abs(bp - c[0]), abs(bq - c[1])) + hh <= rho_in - 1e-12:
            continue
        Xc = [_iv(bp - hh, bp + hh), _iv(bq - hh, bq + hh), _iv(bb - hbeta, bb + hbeta)]
        F = system_iv2(Xc, _iv(Ac, Ac), ec, ec, x, y, False)["F"]
        hull = [_iv(Xc[i][0][0] - drift[i], Xc[i][1][0] + drift[i]) for i in range(3)]
        gH = system_iv2(hull, _iv(A_lo, A_hi), e_lo, e_hi, x, y, False)
        if mode == "shear":
            Te = [_iadd(a, b) for a, b in zip(_matvec_iv(gH["J"], X1), gH["dE"])]
            Ta = [_iadd(a, b) for a, b in zip(_matvec_iv(gH["J"], XA), gH["dA"])]
        else:
            Te, Ta = gH["dE"], gH["dA"]
        ex = False
        for i in range(3):
            t = _iadd(_iadd(F[i], _iv_scale(Te[i], he)), _iv_scale(Ta[i], hA))
            if t[0][0] > 0 or t[1][0] < 0:
                ex = True; break
        if ex:
            continue
        if dep >= max_depth:
            left.append((bp, bq, bb)); continue
        h2, hb2 = hh / 2, hbeta / 2
        boxes += [(bp + sp * h2, bq + sq * h2, bb + sb * hb2, h2, hb2, dep + 1)
                  for sp in (-1, 1) for sq in (-1, 1) for sb in (-1, 1)]
    return {"certified": not left, "boxes": n_eval, "left": len(left), "first_left": left[:3], "Bw": Bw,
            "drift": drift.tolist(), "rho_out_used": ro}


def run_diag(cases, out_name):
    """Item 3 diagnosis (resumable per case): cases = [{kind: 'switch' | 'ring', e_lo, e_hi, regime, frac, n_sub,
    mode}].  Switch: Z (4-var Krawczyk) at the slab's float switch point with the U1 radii; ring: ring3d on the unit."""
    from .u2_store import UnitStore
    x, y = pop()
    st = UnitStore(OUT / "units" / out_name)
    for cs in cases:
        key = f"{cs['kind']}_{cs['mode']}_{cs['e_lo']!r}_{cs['e_hi']!r}_{cs.get('frac', 0.5)}"
        if st.done(key):
            continue
        t0 = time.time()
        ec = 0.5 * (cs["e_lo"] + cs["e_hi"])
        vs, J = tube(ec, x, y)
        if cs["kind"] == "switch":
            vsf, bs = switch_float(ec, x, y, vs)
            xc = [vsf[0], vsf[1], bs, vsf[2]]
            ok, r, info = krawczyk_modes(xc, [6e-4, 6e-4, 2e-4, 5e-4], cs["e_lo"], cs["e_hi"], x, y, cs["mode"])
            rec = {"case": cs, "ok": bool(ok), "radii": r, "info": info}
        else:
            n_sub = cs.get("n_sub", 40); w = (J[1] - J[0]) / n_sub
            k = min(n_sub - 1, int(cs.get("frac", 0.5) * n_sub))
            A_lo, A_hi = J[0] + k * w, J[0] + (k + 1) * w
            c = branch_point(0.5 * (A_lo + A_hi), ec, x, y, start=(vs[0], vs[1]))
            _, b, *_ = point_eval([c[0]], [c[1]], 0.5 * (A_lo + A_hi), ec, x, y)
            if cs["mode"] == "lip":                       # the current Lipschitz-in-ε ring, for reference
                r_ = ring_unit(cs["e_lo"], cs["e_hi"], A_lo, A_hi, tuple(c), x, y)
                rec = {"case": cs, "k": k, "certified": r_["ring_ok"], **{kk: v for kk, v in r_.items() if kk != "ring_ok"}}
            else:
                rec = {"case": cs, "k": k, **ring3d((c[0], c[1], b[0]), A_lo, A_hi, cs["e_lo"], cs["e_hi"], x, y,
                                                    cs["mode"])}
        rec["seconds"] = time.time() - t0
        rec["rss_gb"] = rss_gb()
        st.put(key, rec)
        print(key, {k_: rec[k_] for k_ in rec if k_ in ("ok", "certified", "boxes", "left", "seconds")}, flush=True)


def run_stages(e_lo, e_hi, regime, frac, n_sub, stages=("outer", "ring", "pd", "switch"), max_seconds=300.0):
    """One unit, stage by stage, with the peak RSS reached after each stage (diagnosis of the 3 GB excursion)."""
    x, y = pop()
    cons = json.loads((RESULTS / "uniform_eps" / "constants.json").read_text())
    ec = 0.5 * (e_lo + e_hi)
    vs, J = tube(ec, x, y)
    w = (J[1] - J[0]) / n_sub
    k = min(n_sub - 1, int(frac * n_sub))
    A_lo, A_hi = J[0] + k * w, J[0] + (k + 1) * w
    c = branch_point(0.5 * (A_lo + A_hi), ec, x, y, start=(vs[0], vs[1]))
    rec = {"eps": [e_lo, e_hi], "regime": regime, "k": k, "n_sub": n_sub, "A": [A_lo, A_hi], "box_chunk": BOX_CHUNK,
           "rss_start_gb": rss_gb(), "stages": {}}
    from .u2_store import UnitStore
    store = UnitStore(OUT / "units" / f"stages_{regime}_{e_lo!r}_{e_hi!r}_k{k}_c{BOX_CHUNK}")
    for st in stages:
        prev = store.get(st)
        if prev is not None:                              # completed in an earlier run (its peak RSS is that run's)
            rec["stages"][st] = {**prev, "resumed": True}
            continue
        t0 = time.time()
        if st == "outer":
            r, _ = outer_unit(e_lo, e_hi, A_lo, A_hi, tuple(c), x, y, regime, cons, record=False, max_rounds=8,
                              max_seconds=max_seconds)
            r = {kk: r[kk] for kk in ("certified", "rounds", "peak_cells", "evaluations", "stats", "min_D_pert")}
        elif st == "ring":
            r = ring_unit(e_lo, e_hi, A_lo, A_hi, tuple(c), x, y)
        elif st == "pd":
            r = pd_unit(e_lo, e_hi, A_lo, A_hi, tuple(c), x, y)
        else:
            r = switch_split(e_lo, e_hi, J, x, y, vs, store=store, prefix="switch_sub")
        r["seconds"] = time.time() - t0
        r["peak_rss_gb_after"] = rss_gb()
        r["rss_now_gb"] = _cur_rss_gb()
        rec["stages"][st] = r
        store.put(st, r)
        print(st, json.dumps({kk: r[kk] for kk in r if kk in ("certified", "ring_ok", "pd_ok", "unique_sign_change",
              "boxes", "n_boxes", "seconds", "peak_rss_gb_after", "rss_now_gb", "peak_cells", "stopped",
              "unique_sign_change_all", "n_subslabs", "subslab_widths", "boxes_total", "fail_at")}, default=float),
              flush=True)
    return rec


# ------------------------------------------------------------------ time-capped units with ε-slab splitting
def certify_split(e_lo, e_hi, run_unit, max_seconds, min_width, depth=0, log=None):
    """run_unit(e_lo, e_hi, max_seconds) -> dict with 'certified' (and anything else).  If a unit is not certified within
    max_seconds (timeout or a failed cell), the ε-slab is halved and both halves are retried, recursively, down to
    min_width; a slab narrower than 2·min_width that still fails is reported as failed.  Returns the list of leaf slabs
    [(e_lo, e_hi, certified, result)] in ε order."""
    t0 = time.time()
    r = run_unit(e_lo, e_hi, max_seconds)
    r = dict(r); r["wall"] = time.time() - t0
    if log is not None:
        log(e_lo, e_hi, depth, r)
    if r.get("certified"):
        return [(e_lo, e_hi, True, r)]
    if (e_hi - e_lo) / 2 < min_width:
        return [(e_lo, e_hi, False, r)]
    m = 0.5 * (e_lo + e_hi)
    return certify_split(e_lo, m, run_unit, max_seconds, min_width, depth + 1, log) + \
        certify_split(m, e_hi, run_unit, max_seconds, min_width, depth + 1, log)


def outer_runner(A_frac, n_sub, regime, x, y, cons, max_rounds=8):
    """A run_unit for certify_split: one outer unit at the A-sub-interval of the slab's own tube."""
    def run(e_lo, e_hi, max_seconds):
        ec = 0.5 * (e_lo + e_hi)
        vs, J = tube(ec, x, y)
        w = (J[1] - J[0]) / n_sub
        k = min(n_sub - 1, int(A_frac * n_sub))
        A_lo, A_hi = J[0] + k * w, J[0] + (k + 1) * w
        c = branch_point(0.5 * (A_lo + A_hi), ec, x, y, start=(vs[0], vs[1]))
        r, _ = outer_unit(e_lo, e_hi, A_lo, A_hi, tuple(c), x, y, regime, cons, record=False, max_rounds=max_rounds,
                          max_seconds=max_seconds)
        return {"certified": r["certified"], "k": k, "A": [A_lo, A_hi], "worst_live": r.get("worst_live"),
                "rounds": r["rounds"], "evaluations": r["evaluations"], "timed_out": bool(r.get("timed_out"))}
    return run


def run_ascan(e_lo, e_hi, regime, n_sub, max_seconds, out_name):
    """Which A-sub-intervals of the tube fail at one (wide) ε-slab: one time-capped outer unit per sub-interval, no
    splitting, no export (one JSON line per sub-interval)."""
    x, y = pop()
    cons = json.loads((RESULTS / "uniform_eps" / "constants.json").read_text())
    from .u2_store import UnitStore
    store = UnitStore(OUT / "units" / out_name)
    ec = 0.5 * (e_lo + e_hi)
    vs, J = tube(ec, x, y)
    w = (J[1] - J[0]) / n_sub
    for k in range(n_sub):
        if store.done(f"k{k:03d}"):
            continue
        A_lo, A_hi = J[0] + k * w, J[0] + (k + 1) * w
        c = branch_point(0.5 * (A_lo + A_hi), ec, x, y, start=(vs[0], vs[1]))
        t0 = time.time()
        r, _ = outer_unit(e_lo, e_hi, A_lo, A_hi, tuple(c), x, y, regime, cons, record=False, max_rounds=8,
                          max_seconds=max_seconds)
        rec = {"k": k, "eps": [e_lo, e_hi], "regime": regime, "A": [A_lo, A_hi], "certified": r["certified"],
               "timed_out": bool(r.get("timed_out")),
               "worst_live": r.get("worst_live"), "rounds": r["rounds"], "seconds": time.time() - t0,
               "rss_gb": rss_gb()}
        store.put(f"k{k:03d}", rec)                       # atomic; a restart skips it
        print(k, rec["certified"], round(rec["seconds"], 1), rec["worst_live"], flush=True)


if __name__ == "__main__":
    try:
        os.nice(15) if os.getpriority(os.PRIO_PROCESS, 0) < 15 else None
    except PermissionError:
        pass
    from .u2_gate import GateRefused, require_gate
    try:
        require_gate(" ".join(sys.argv[1:]))
    except GateRefused as ex:
        print(f"REFUSED: {ex}", flush=True)
        sys.exit(3)
    if sys.argv[1] == "probes":
        run_probes(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "stages":
        rec = run_stages(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], float(sys.argv[5]), int(sys.argv[6]),
                         stages=tuple(sys.argv[7].split(",")) if len(sys.argv) > 7 else ("outer", "ring", "pd", "switch"))
        (OUT / f"stages_{sys.argv[4]}_{sys.argv[2]}_{sys.argv[3]}_c{BOX_CHUNK}.json").write_text(
            json.dumps(rec, indent=1, default=float))
    elif sys.argv[1] == "diag":
        run_diag(json.loads(Path(sys.argv[2]).read_text()), sys.argv[3])
    elif sys.argv[1] == "ascan":
        run_ascan(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), sys.argv[7])
