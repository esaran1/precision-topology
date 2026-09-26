"""Theorem 1 (the small-scale criterion, math note §10-§11) for the sine family f_a(t) = t + a sin t, width 1:
numerical checks of its hypotheses.  The PROVED / CHECKED status of each hypothesis is stated in
results/theorem1_hypotheses.md; this module produces the numbers (results/theorem1_checks.json).

    python -m src.theorem1_checks            # all checks (a = 1.30-1.60 and a = 1.02, 1.05, 1.10); ~1-2 min, one process

Checks
  H-A1  data-gap attainment: Gamma_n = sup G_n > 0 is a sup over |w1| < a/1.4 (Step 2), so it is attained; the grid +
        refinement maximiser is reported with its distance to the boundary |w1| = a/1.4, against the certified Ghat.
  H-A2  class-mean-gap attainment: sup_alpha |D(alpha)|.  Continuous windows: |D_c(alpha)| <= 3/(0.8 alpha), so the sup
        is over a compact interval and attained; a Lipschitz grid certifies alpha*_c is the global maximiser.  Data
        points (800-point population): every x is an integer multiple of q = 0.4/79401 (inner even, outer odd), so
        D is 2pi/q-periodic and |D(pi/q)| = 2 > |D*| = 1.571: the sup over alpha in R is attained at an ALIAS, not at
        alpha*.  Reported: the first alpha beyond alpha* where |D| reaches |D*| (Lipschitz scan), and the expansion
        crossover scale below which such a competitor beats theta* at second order.
  H-C   compactness: the localisation lemma W(s) (proved, §5.0) at the scales used, and where W(s) reaches the alias.
  H-R   uniform remainder: R = L* - [log 2 - (s/4) dmu + (s^2/8) Var] against s^4 m4 over sampled theta.
  H-Q   quadratic growth near alpha*: D'' certified of one sign on [alpha* - d, alpha* + d] (Lipschitz bound on D''),
        giving |D*| - |D(alpha)| >= c (alpha - alpha*)^2 there, plus the certified separation margin outside.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_MAIN = (1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
A_SMALL = (1.02, 1.05, 1.10)
Q = 0.4 / 79401                       # lattice unit of the 800-point population (x = n q, n integer)


def _data():
    from .conditional_certified import _population
    x, y = _population()
    return x, y


# ------------------------------------------------------------------ generic, testable pieces


def lipschitz_scan(f, lo, hi, h, L, level):
    """Rigorous-in-exact-arithmetic Lipschitz scan: returns (certified, first_bad).  certified iff f(g) + L h/2 < level
    at every cell centre g of [lo, hi] with cells of width h; first_bad = the first cell centre where this fails."""
    n = int(math.ceil((hi - lo) / h))
    for i0 in range(0, n, 2_000_000):
        g = lo + (np.arange(i0, min(n, i0 + 2_000_000)) + 0.5) * h
        v = f(g)
        bad = np.nonzero(v + L * h / 2 >= level)[0]
        if len(bad):
            return False, float(g[bad[0]])
    return True, None


def adaptive_scan(f, lo, hi, h, L, level, min_h=1e-6, chunk=2_000_000):
    """Lipschitz scan with refinement: cells failing f(g) + L h/2 < level are split 10x, down to min_h.  Returns
    (certified, first_exceed, first_unresolved): first_exceed = the smallest grid point found with f >= level
    (a counterexample), first_unresolved = the smallest cell left undecided at min_h."""
    n = int(math.ceil((hi - lo) / h))
    first_exceed, first_unres = None, None
    for i0 in range(0, n, chunk):
        cells = [(lo + (np.arange(i0, min(n, i0 + chunk)) + 0.5) * h, h)]
        while cells:
            g, hh = cells.pop()
            v = f(g)
            ex = np.nonzero(v >= level)[0]
            if len(ex):
                first_exceed = float(g[ex[0]]) if first_exceed is None else min(first_exceed, float(g[ex[0]]))
            bad = np.nonzero((v + L * hh / 2 >= level) & (v < level))[0]
            if not len(bad):
                continue
            if hh / 10 < min_h:
                first_unres = float(g[bad[0]]) if first_unres is None else min(first_unres, float(g[bad[0]]))
                continue
            sub = (g[bad][:, None] - hh / 2 + (np.arange(10) + 0.5)[None, :] * hh / 10).ravel()
            cells.append((sub, hh / 10))
        if first_exceed is not None or first_unres is not None:
            break                                   # the first failing chunk locates the first failure
    return (first_exceed is None and first_unres is None), first_exceed, first_unres


def quadratic_growth(f2, lo, hi, h, L3):
    """Sign-definiteness of a second derivative f2 on [lo, hi] (Lipschitz constant L3): returns (ok, c) with
    c = min |f2| - L3 h/2 > 0 iff f2 keeps one sign; then |f(alpha) - f(alpha0)| >= (c/2)(alpha - alpha0)^2 around a
    critical point alpha0 in the interval."""
    n = int(math.ceil((hi - lo) / h))
    g = lo + (np.arange(n) + 0.5) * h
    v = f2(g)
    same = bool(np.all(v > 0) or np.all(v < 0))
    c = float(np.min(np.abs(v)) - L3 * h / 2)
    return same and c > 0, c


def profiled_loss(phi, y, s, iters=60):
    """L*(s) = min_b mean softplus(z) - y z, z = s phi + b (float64 Newton on b; b* is unique)."""
    b = 0.0
    for _ in range(iters):
        z = s * phi + b
        sg = 1 / (1 + np.exp(-z))
        g = sg.mean() - y.mean()
        H = (sg * (1 - sg)).mean()
        step = g / H
        b -= step
        if abs(step) < 1e-17:
            break
    z = s * phi + b
    return float(np.mean(np.logaddexp(0, z) - y * z))


def remainder_ratio(phi, y, s, var_coef=1 / 8):
    """|R|/(s^4 m4) with R = L* - [log 2 - (s/4) dmu + (s^2/8) Var(phi)] (balanced classes)."""
    dmu = phi[y == 1].mean() - phi[y == 0].mean()
    var = phi.var()
    m4 = np.mean((phi - phi.mean()) ** 4)
    R = profiled_loss(phi, y, s) - (math.log(2) - s * dmu / 4 + var_coef * s * s * var)
    return abs(R) / (s ** 4 * m4), R, m4


# ------------------------------------------------------------------ the class-mean profile D(alpha)


def D_data_factory(x, y):
    xo, xi = x[y == 1], x[y == 0]

    def D(al):
        al = np.atleast_1d(al)
        out = np.empty(len(al))
        for i in range(0, len(al), 20_000):
            A = al[i:i + 20_000, None]
            out[i:i + 20_000] = np.cos(A * xo).mean(1) - np.cos(A * xi).mean(1)
        return out

    def D2(al):
        A = np.atleast_1d(al)[:, None]
        return -(xo ** 2 * np.cos(A * xo)).mean(1) + (xi ** 2 * np.cos(A * xi)).mean(1)
    L1 = float(np.abs(xo).mean() + np.abs(xi).mean())
    L3 = float((np.abs(xo) ** 3).mean() + (np.abs(xi) ** 3).mean())
    return D, D2, L1, L3


def D_lattice_factory(x, y):
    """D on the data lattice via Dirichlet sums (O(1) per alpha): x = (n0 + d k) q for each arithmetic run."""
    n = np.round(x / Q).astype(np.int64)
    assert np.max(np.abs(x - n * Q)) < 1e-12
    runs = []                                     # (sign, n0, d, count, weight)
    for cls, w in ((1, 1.0), (0, -1.0)):
        m = np.sort(n[y == cls])
        # split into arithmetic runs
        start = 0
        for j in range(1, len(m) + 1):
            if j == len(m) or (j - start >= 2 and m[j] - m[j - 1] != m[start + 1] - m[start]):
                seg = m[start:j]
                d = int(seg[1] - seg[0]) if len(seg) > 1 else 1
                runs.append((w / (y == cls).sum(), int(seg[0]), d, len(seg)))
                start = j
    assert sum(r[3] for r in runs) == len(x)

    def D(al):
        t = np.atleast_1d(al) * Q
        tot = np.zeros(len(t))
        for w, n0, d, cnt in runs:
            u = t * d
            den = np.sin(u / 2)
            small = np.abs(den) < 1e-6
            ratio = np.sin(cnt * u / 2) / np.where(small, 1.0, den)
            if small.any():                       # near a pole: direct sum for these alpha only
                ks = np.arange(cnt)
                tt = t[small][:, None]
                direct = np.cos(tt * (n0 + d * ks)).sum(1)
                val = np.cos(t[small] * n0 + (cnt - 1) * u[small] / 2)
                ratio[small] = np.where(np.abs(val) > 1e-300, direct / np.where(val == 0, 1, val), cnt)
            tot += w * np.cos(t * n0 + (cnt - 1) * u / 2) * ratio
        return tot
    return D, runs


def D_cont_factory(nodes=400):
    gx, gw = np.polynomial.legendre.leggauss(nodes)

    def mean_on(lo, hi, fn):
        xs = 0.5 * (hi - lo) * gx + 0.5 * (hi + lo)
        return lambda A: (fn(A, xs) * gw).sum(-1) / 2

    def D(al):
        A = np.atleast_1d(al)[:, None]
        cosf = lambda A, xs: np.cos(A * xs)
        return mean_on(1.2, 2.0, cosf)(A) - mean_on(-0.8, 0.8, cosf)(A)

    def D2(al):
        A = np.atleast_1d(al)[:, None]
        f = lambda A, xs: -xs ** 2 * np.cos(A * xs)
        return mean_on(1.2, 2.0, f)(A) - mean_on(-0.8, 0.8, f)(A)
    L1 = 1.6 + 0.4
    L3 = (2.0 ** 4 - 1.2 ** 4) / (4 * 0.8) + 0.8 ** 3 / 4
    return D, D2, L1, L3


# ------------------------------------------------------------------ checks


def check_attainment_data_gap(a, x, y, Ghat_lo):
    from .profiled_bnb import gap
    W = a / 1.4
    xo, xi = x[y == 1], x[y == 0]

    def Gn(w1, b1):
        w1 = np.asarray(w1)[:, None]; b1 = np.asarray(b1)[:, None]
        po = w1 * xo + b1 + a * np.sin(w1 * xo + b1)
        pi = w1 * xi + b1 + a * np.sin(w1 * xi + b1)
        return np.maximum(po.min(1) - pi.max(1), pi.min(1) - po.max(1))
    wg = np.linspace(-W, W, 281)
    bg = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    Wm, Bm = np.meshgrid(wg, bg, indexing="ij")
    se = math.sqrt(a - 1)                        # plus the rescaled window w1 = sqrt(eps) u, b1 = pi + sqrt(eps) v
    Wr, Br = np.meshgrid(se * np.linspace(-8, 8, 321), math.pi + se * np.linspace(-12, 12, 481), indexing="ij")
    cw, cb = np.concatenate([Wm.ravel(), Wr.ravel()]), np.concatenate([Bm.ravel(), Br.ravel()])
    v = Gn(cw, cb)
    best, w, b = -np.inf, None, None
    for j in np.argsort(-v)[:12]:                # pattern refinement from the 12 best grid points
        wj, bj, bj_v = cw[j], cb[j], v[j]
        hw, hb = (wg[1] - wg[0], bg[1] - bg[0]) if j < Wm.size else (se * 0.05, se * 0.05)
        for _ in range(90):
            CW, CB = np.meshgrid(np.clip(wj + np.linspace(-hw, hw, 21), -W, W), bj + np.linspace(-hb, hb, 21),
                                 indexing="ij")
            vv = Gn(CW.ravel(), CB.ravel())
            k = int(np.argmax(vv))
            if vv[k] >= bj_v:
                bj_v, wj, bj = vv[k], CW.ravel()[k], CB.ravel()[k]
            if vv[k] <= bj_v or _ % 2:
                hw, hb = hw / 1.5, hb / 1.5
        if bj_v > best:
            best, w, b = bj_v, wj, bj
    G_cont = float(gap([w], [b], a)[0])
    return {"a": a, "Gamma_n_found": float(best), "argmax_w1": float(w), "argmax_b1": float(b % (2 * math.pi)),
            "domain_bound_a_over_1.4": W, "argmax_interior_margin": float(W - abs(w)),
            "Ghat_certified_lo": Ghat_lo, "Gamma_n_ge_Ghat_lo": bool(best >= Ghat_lo - 1e-12),
            "G_cont_at_argmax": G_cont, "Gamma_n_positive": bool(best > 0)}


def _argmax_abs(f, lo, hi, iters=200):
    """Golden-section maximisation of |f| on [lo, hi] (unimodal bracket from a fine grid)."""
    r = (math.sqrt(5) - 1) / 2
    a_, b_ = lo, hi
    c_, d_ = b_ - r * (b_ - a_), a_ + r * (b_ - a_)
    for _ in range(iters):
        if abs(f(np.array([c_]))[0]) > abs(f(np.array([d_]))[0]):
            b_ = d_
        else:
            a_ = c_
        c_, d_ = b_ - r * (b_ - a_), a_ + r * (b_ - a_)
    return 0.5 * (a_ + b_)


def check_classmean_attainment(x, y):
    Dd, D2d, L1d, L3d = D_data_factory(x, y)
    Dl, runs = D_lattice_factory(x, y)
    Dc, D2c, L1c, L3c = D_cont_factory()
    out = {}
    # continuous windows: decay bound |D_c| <= 3/(0.8 alpha); exhaustive certified scan on [0, 3/(0.8 |D_c*|)]
    g = np.arange(0.0005, 50, 0.0005)
    vc = np.abs(Dc(g)); j = int(np.argmax(vc))
    ac = _argmax_abs(Dc, g[j] - 0.001, g[j] + 0.001)
    Dcs = float(abs(Dc(np.array([ac]))[0]))
    tail = 3 / (0.8 * Dcs)
    d = 0.05
    okq, cq = quadratic_growth(lambda t: D2c(t) * np.sign(Dc(np.array([ac]))[0]), ac - d, ac + d, 1e-4, L3c)
    f_out = lambda t: np.abs(Dc(t))
    ok1, bad1 = lipschitz_scan(f_out, 0.0, ac - d, 1e-4, L1c, Dcs)
    ok2, bad2 = lipschitz_scan(f_out, ac + d, tail, 1e-4, L1c, Dcs)
    out["continuous"] = {"alpha_star": ac, "D_star_abs": Dcs, "decay_bound_beyond": tail,
                         "quad_growth_interval": [ac - d, ac + d], "quad_growth_ok": okq,
                         "quad_growth_c_min_abs_D2_minus_slack": cq,
                         "separation_certified_outside": bool(ok1 and ok2),
                         "global_maximiser_certified": bool(okq and ok1 and ok2)}
    # data points
    g = np.arange(0.0005, 50, 0.0005)
    vd = np.abs(Dd(g)); j = int(np.argmax(vd))
    ad = _argmax_abs(Dd, g[j] - 0.001, g[j] + 0.001)
    Dds = float(abs(Dd(np.array([ad]))[0]))
    okqd, cqd = quadratic_growth(lambda t: D2d(t) * np.sign(Dd(np.array([ad]))[0]), ad - d, ad + d, 1e-4, L3d)
    # lattice closed form agrees with direct sum
    tst = np.concatenate([np.linspace(0.01, 50, 200), np.linspace(1000, 6e5, 200)])
    agree = float(np.max(np.abs(Dl(tst) - Dd(tst))))
    ok_lo, _ = lipschitz_scan(lambda t: np.abs(Dl(t)), 0.0, ad - d, 1e-4, L1d, Dds)
    period_half = math.pi / Q
    # first alpha beyond alpha* + d where |D| can reach |D*|: adaptive Lipschitz scan
    ok_hi, first_exceed, first_unres = adaptive_scan(lambda t: np.abs(Dl(t)), ad + d, period_half, 0.02, L1d, Dds)
    first_bad = first_unres
    alias = period_half
    out["data"] = {"alpha_star": ad, "D_star_abs": Dds, "quad_growth_ok": okqd, "quad_growth_c": cqd,
                   "lattice_unit_q": Q, "lattice_closed_form_max_abs_err": agree,
                   "separation_certified_below_alpha_star": ok_lo,
                   "alpha_star_global_on_(0, pi/q)": bool(ok_hi),
                   "first_alpha_Lipschitz_uncertified": first_bad, "first_alpha_with_absD_ge_Dstar": first_exceed,
                   "D_at_first_exceed": float(Dl(first_exceed)[0]) if first_exceed else None,
                   "alias_alpha_pi_over_q": alias, "D_at_alias": float(Dl(alias)[0]),
                   "D_at_alias_direct": float(Dd(alias)[0]), "sup_absD": 2.0}
    return out


def crossover(a, x, y, alpha_c, D_c, D_star, alpha_star):
    """Second-order crossover scale where a competitor (alpha_c, |D_c| > |D*|) beats theta* in the small-s expansion,
    and a direct L* comparison at 10x above and below it."""
    b_of = lambda D: -math.pi / 2 if D < 0 else math.pi / 2          # sign(w2) = +1, sin b1 D(alpha) = |D|
    phi = lambda al, b: al * x + b + a * np.sin(al * x + b)
    ps, pc = phi(alpha_star, b_of(D_star)), phi(alpha_c, b_of(D_c))
    dmu_s = ps[y == 1].mean() - ps[y == 0].mean(); dmu_c = pc[y == 1].mean() - pc[y == 0].mean()
    sc = 2 * (dmu_c - dmu_s) / (pc.var() - ps.var())
    import mpmath as mp
    mp.mp.dps = 40

    def Lmp(ph, s):
        ph = [mp.mpf(float(v)) for v in ph]; yy = [int(v) for v in y]
        b = mp.mpf(0)
        for _ in range(60):
            z = [s * p + b for p in ph]
            sg = [1 / (1 + mp.e ** (-zz)) for zz in z]
            g = sum(sg) / len(sg) - mp.mpf(1) / 2
            H = sum(v * (1 - v) for v in sg) / len(sg)
            b -= g / H
            if abs(g) < mp.mpf(10) ** -35:
                break
        z = [s * p + b for p in ph]
        return sum(mp.log(1 + mp.e ** zz) - yv * zz for zz, yv in zip(z, yy)) / len(z)
    res = {"a": a, "alpha_competitor": alpha_c, "dmu_star": float(dmu_s), "dmu_competitor": float(dmu_c),
           "var_star": float(ps.var()), "var_competitor": float(pc.var()), "s_cross_expansion": float(sc)}
    for lab, fac in (("below", 0.1), ("above", 10.0)):
        s = mp.mpf(sc * fac)
        res[f"L_competitor_minus_L_star_at_{lab}"] = float(Lmp(pc, s) - Lmp(ps, s))
    res["competitor_wins_below_and_loses_above"] = bool(res["L_competitor_minus_L_star_at_below"] < 0 <
                                                        res["L_competitor_minus_L_star_at_above"])
    return res


def check_remainder(a, x, y, rng, n=400):
    rows = []
    W = 10.0
    thetas = [(1.7913244, -math.pi / 2), (0.3, 3.3)] + [(rng.uniform(-W, W), rng.uniform(0, 2 * math.pi))
                                                        for _ in range(n)]
    for s in (0.3, 0.1, 0.03, 0.01):
        rs = []
        for w1, b1 in thetas:
            phi = w1 * x + b1 + a * np.sin(w1 * x + b1)
            r, R, m4 = remainder_ratio(phi, y, s)
            if s ** 4 * m4 > 1e-9:                # float64 resolution of R (L* to ~1e-16)
                rs.append(r)
        rows.append({"s": s, "n_theta": len(rs), "max_ratio": float(np.max(rs)), "median_ratio": float(np.median(rs))})
    return {"a": a, "box_abs_w1_le": W, "by_s": rows, "max_ratio_all": max(r["max_ratio"] for r in rows)}


def main():
    os.nice(15)
    t0 = time.time()
    x, y = _data()
    mech = pd.read_csv(RESULTS / "wp12_mechanism.csv", float_precision="round_trip").set_index("a")
    out = {"theorem": "math note §10.1 criterion (Lemma 1, Corollaries 1-2, Lemma 2) with §11 Step 1-2, width 1, f_a"}
    out["H-A1_data_gap_attainment"] = [check_attainment_data_gap(a, x, y, float(mech.loc[round(a, 2), "Ghat_rigorous"]))
                                       for a in A_MAIN + A_SMALL]
    cm = check_classmean_attainment(x, y)
    out["H-A2_classmean_attainment_and_H-Q_quadratic_growth"] = cm
    dd = cm["data"]
    comp = []
    for a in A_MAIN:
        c = {"first_exceed": crossover(a, x, y, dd["first_alpha_with_absD_ge_Dstar"], dd["D_at_first_exceed"],
                                       -dd["D_star_abs"], dd["alpha_star"])} if dd["first_alpha_with_absD_ge_Dstar"] else {}
        c["alias"] = crossover(a, x, y, dd["alias_alpha_pi_over_q"], dd["D_at_alias"], -dd["D_star_abs"], dd["alpha_star"])
        comp.append(c)
    out["H-A2_data_competitor_crossovers"] = comp
    from .profiled_bnb import w_bound
    out["H-C_compactness"] = {
        "large_s": "proved: for s > s1(a) every minimiser has G_n > 0, so |w1| < a/1.4 (§11 Proposition (ii) + Step 2)",
        "s1_data": {str(a): float(mech.loc[a, "s1_data"]) for a in A_MAIN},
        "W_of_s_at_1.30": {str(s): float(w_bound(s, 1.30, x, y)) for s in (1.0, 0.3, 0.1, 0.05, 0.01, 1e-3)},
        "W_times_s_limit_at_1.30": float(w_bound(1e-6, 1.30, x, y) * 1e-6),
        "s_where_W_reaches_alias_1.30": float(w_bound(1e-6, 1.30, x, y) * 1e-6 / (math.pi / Q))}
    rng = np.random.default_rng(0)
    out["H-R_uniform_remainder"] = [check_remainder(a, x, y, rng) for a in (1.30, 1.50, 1.60)]
    out["seconds"] = time.time() - t0
    (RESULTS / "theorem1_checks.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
