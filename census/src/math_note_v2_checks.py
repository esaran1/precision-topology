"""Rigour checks for math_note_v2.md (Block 2).

bounds     B_full in closed form and B(P) in exact arithmetic (integer PAVA block sums, logs at 50 digits).
uniform    branch-loss upper bound over A in a neighbourhood of the certified switch, against B(24) and B_full.
rounding   interval-arithmetic (mpmath.iv) re-evaluation of the attained values used by the certificates.
h2         H2': dG/dA along the branch at the switch, as an interval (implicit-function formula, interval
           evaluation over the certified argmin enclosure).
solve      the limit solve threshold: bracket, nondegeneracy, transversality of the solve margin.

    python -m src.math_note_v2_checks bounds | uniform | rounding | h2 | solve
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
mp.mp.dps = 50


def _pava_blocks(yv):
    blocks = []
    for v in yv:
        blocks.append([int(v), 1])
        while len(blocks) > 1 and blocks[-2][0] * blocks[-1][1] > blocks[-1][0] * blocks[-2][1]:
            s2, c2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += c2
    return blocks


def _logloss_exact(blocks):
    tot = mp.mpf(0)
    for k, n in blocks:
        for c, pr in ((k, mp.mpf(k) / n), (n - k, mp.mpf(n - k) / n)):
            if c > 0:
                tot -= c * mp.log(pr)
    return tot


def monotone_bound_exact(x, y, window):
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(int)
    n = len(x)
    best = None
    for c in np.unique(np.concatenate([xs - window, xs])):
        keep = ~((xs >= c) & (xs < c + window))
        yk = ys[keep]
        v = min(_logloss_exact(_pava_blocks(yk)), _logloss_exact(_pava_blocks(yk[::-1])))
        best = v if best is None or v < best else best
    return best / n


def bounds():
    from .limit_bnb import population
    x, y = population()
    closed = mp.log(3) / 4 + mp.log(mp.mpf(3) / 2) / 2
    ys = y[np.argsort(x)].astype(int)
    full = min(_logloss_exact(_pava_blocks(ys)), _logloss_exact(_pava_blocks(ys[::-1]))) / len(x)
    b24 = monotone_bound_exact(x, y, 4 * math.sqrt(2) / 24)
    rows = [{"quantity": "B_full (closed form: ln3/4 + ln(3/2)/2)", "value": mp.nstr(closed, 20)},
            {"quantity": "B_full (exact PAVA, 50 digits)", "value": mp.nstr(full, 20)},
            {"quantity": "B(24) (exact PAVA, 50 digits)", "value": mp.nstr(b24, 20)},
            {"quantity": "|B_full closed - PAVA|", "value": mp.nstr(abs(closed - full), 5)}]
    pd.DataFrame(rows).to_csv(RESULTS / "mn2_bounds.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "bounds":
    bounds()


# ----------------------------------------------------------------------------- uniformity over A
A_NEIGHBOURHOOD = (0.60, 0.76)
A_GRID_STEP = 0.02


def uniform(P=24.0):
    """Upper bound on the global minimum m(A) = min(m-, m+) of L0* over the neighbourhood.

    At grid points A_i: an attained value U_i = L0*(θ_i; A_i) at the certified argmin θ_i (rigorous upper
    bound once re-evaluated in interval arithmetic, see `rounding`).  Between grid points, the same θ_i
    gives L0*(θ_i; A) <= U_i + |A - A_i|·mean|h(σ_i(x))| (logistic loss is 1-Lipschitz in the logit; the
    profiled bias can only lower it).  Hence sup over [A_i, A_{i+1}] of m(A) is bounded by the smaller of
    the two bridges at the midpoint.  Compared with B(24) and B_full, which do not depend on A.
    """
    from .limit_bnb import certify, h, population
    x, y = population()
    grid = np.round(np.arange(A_NEIGHBOURHOOD[0], A_NEIGHBOURHOOD[1] + 1e-9, A_GRID_STEP), 6)
    rows = []
    for A in grid:
        rm = certify(A, x, y, "-", P)
        rp = certify(A, x, y, "+", P)
        g = rp if rp["upper"] < rm["upper"] else rm
        sig = g["arg_p"] * x + g["arg_q"]
        rows.append({"A": A, "U": g["upper"], "mean_abs_h": float(np.mean(np.abs(h(sig)))),
                     "p": g["arg_p"], "q": g["arg_q"], "region": g["region"]})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    bridge = []
    for i in range(len(d) - 1):
        mid = 0.5 * A_GRID_STEP
        bridge.append(min(d.U[i] + mid * d.mean_abs_h[i], d.U[i + 1] + mid * d.mean_abs_h[i + 1]))
    d["sup_bridge_to_next"] = bridge + [np.nan]
    b24 = float(mp.mpf("0.38797358308678997649"))
    bfull = float(mp.log(3) / 4 + mp.log(mp.mpf(3) / 2) / 2)
    sup_U = float(max(d.U.max(), np.nanmax(d.sup_bridge_to_next)))
    d["sup_over_neighbourhood"] = sup_U
    d["margin_B24"] = b24 - sup_U
    d["margin_Bfull"] = bfull - sup_U
    d.to_csv(RESULTS / "mn2_uniformity.csv", index=False)
    print(f"sup of the branch loss over A in {A_NEIGHBOURHOOD}: {sup_U:.6f}; "
          f"margin below B(24): {b24 - sup_U:.6f}; below B_full: {bfull - sup_U:.6f}")


if __name__ == "__main__" and sys.argv[1] == "uniform":
    uniform()


# ----------------------------------------------------------------------------- interval tools
iv = mp.iv
iv.dps = 30


def _mid(v):
    return float((mp.mpf(v.a) + mp.mpf(v.b)) / 2)


def _rad(v):
    return float((mp.mpf(v.b) - mp.mpf(v.a)) / 2)


def _ivx(v, r=0.0):
    return iv.mpf([mp.mpf(v) - r, mp.mpf(v) + r])


def _softplus_iv(z):
    # softplus(z) = log(1 + e^z), monotone increasing: evaluate at the interval ends
    lo = mp.log1p(mp.exp(z.a)) if z.a < 0 else z.a + mp.log1p(mp.exp(-z.a))
    hi = mp.log1p(mp.exp(z.b)) if z.b < 0 else z.b + mp.log1p(mp.exp(-z.b))
    return iv.mpf([lo, hi])


def loss_upper_iv(phi_fn, theta, b2, scale, x, y):
    """Rigorous upper bound on the profiled loss at a point: L(θ, b2) >= L*(θ) for any b2, evaluated in
    interval arithmetic; returns the interval's upper end."""
    tot = iv.mpf(0)
    for xi, yi in zip(x, y):
        z = scale * phi_fn(theta, iv.mpf(float(xi))) + iv.mpf(b2)
        tot += _softplus_iv(z) - (z if yi == 1 else 0)
    return (tot / len(x)).b


def rounding():
    """Interval re-evaluation of the attained values that the certificates use as upper bounds.

    The argmins and float upper bounds are read from the certificate runs' own evaluation files.
    """
    from .limit_bnb import population, profile as profile0
    x, y = population()
    rows = []
    h_iv = lambda th, xi: -(th[0] * xi + th[1]) + (th[0] * xi + th[1]) ** 3 / 6
    le = pd.read_csv(RESULTS / "limit_switch_evaluations.csv")
    for r in le.itertuples():
        for reg, pp, qq, fu in (("-", r.minus_p, r.minus_q, r.m_minus_hi), ("+", r.plus_p, r.plus_q, r.m_plus_hi)):
            if not np.isfinite(pp):
                continue
            _, b2, *_ = profile0([pp], [qq], r.A, x, y)
            up = loss_upper_iv(h_iv, (iv.mpf(pp), iv.mpf(qq)), float(b2[0]), iv.mpf(r.A), x, y)
            rows.append({"problem": "limit", "A_or_s": r.A, "region": reg, "float_upper": fu,
                         "interval_upper": float(up), "diff": float(up) - fu})
    from . import blockB_landscape as bb
    from .profiled_bnb import profile
    xb, yb = bb.population_data()
    xb, yb = xb.numpy(), yb.numpy()
    ev = pd.read_csv(RESULTS / "cond_certified_bracket_evaluations.csv")
    for r in ev.itertuples():
        a, s_ = float(r.a), float(r.s)
        fa = lambda th, xi, a=a: (th[0] * xi + th[1]) + a * iv.sin(th[0] * xi + th[1])
        for reg, w, bb1, fu in (("-", r.minus_arg_w1, r.minus_arg_b1, r.m_minus_hi),
                                ("+", r.plus_arg_w1, r.plus_arg_b1, r.m_plus_hi)):
            if not np.isfinite(w) or w == 0.0:
                continue
            _, b2, *_ = profile([w], [bb1], s_, a, xb, yb)
            up = loss_upper_iv(fa, (iv.mpf(w), iv.mpf(bb1)), float(b2[0]), iv.mpf(s_), xb, yb)
            rows.append({"problem": f"a={a:.2f}", "A_or_s": s_, "region": reg, "float_upper": fu,
                         "interval_upper": float(up), "diff": float(up) - fu})
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "mn2_rounding.csv", index=False)
    print(d.describe().to_string())
    print("rows:", len(d), " max |interval upper - float upper|:", d["diff"].abs().max())


if __name__ == "__main__" and sys.argv[1] == "rounding":
    rounding()


# ----------------------------------------------------------------------------- H2' and the solve switch
R2 = math.sqrt(2)


def _sig_iv(z):
    f = lambda t: 1 / (1 + mp.exp(-t))
    return iv.mpf([f(z.a), f(z.b)])


def _dsig_iv(s):
    lo, hi = s.a, s.b
    vals = [lo * (1 - lo), hi * (1 - hi)]
    mx = mp.mpf("0.25") if lo <= 0.5 <= hi else max(vals)
    return iv.mpf([min(vals), mx])


def _h(t):
    return -t + t ** 3 / 6


def _box(encl):
    return iv.mpf([encl[0], encl[1]]), iv.mpf([encl[2], encl[3]])


def b2_enclosure(P, Q, A, x, y, b_c, r=1e-3):
    """Interval Newton: an enclosure N of the profiled b2*(p, q) for all (p, q) in the box."""
    B = iv.mpf([b_c - r, b_c + r])
    ybar = mp.mpf(int(y.sum())) / len(y)
    g_c = iv.mpf(0); g_b = iv.mpf(0)
    for xi in x:
        sg = P * float(xi) + Q
        zc = A * _h(sg) + b_c
        g_c += _sig_iv(zc)
        g_b += _dsig_iv(_sig_iv(A * _h(sg) + B))
    g_c = g_c / len(x) - ybar
    g_b = g_b / len(x)
    N = b_c - g_c / g_b
    ok = (N.a >= B.a) and (N.b <= B.b)
    return N, ok


_IFT_CACHE = {}


def ift_limit(A, x, y, encl, b_c):
    """dθ/dA = -H3^{-1} ∂A∇L over the box (θ = (p, q, b2)), and ∇G; returns dG/dA as an interval."""
    P, Q = _box(encl)
    Aiv = iv.mpf(A)
    B, ok = b2_enclosure(P, Q, Aiv, x, y, b_c)
    if not ok:
        return {"b2_validated": False}
    n = len(x)
    H = [[iv.mpf(0)] * 3 for _ in range(3)]
    c = [iv.mpf(0)] * 3
    for xi, yi in zip(x, y):
        xi = float(xi)
        sg = P * xi + Q
        hp_ = -1 + sg ** 2 / 2
        z = Aiv * _h(sg) + B
        s_ = _sig_iv(z); ds = _dsig_iv(s_)
        r_ = s_ - int(yi)
        dz = [Aiv * hp_ * xi, Aiv * hp_, iv.mpf(1)]
        d2 = {(0, 0): Aiv * sg * xi * xi, (0, 1): Aiv * sg * xi, (1, 1): Aiv * sg}
        for i in range(3):
            for j in range(3):
                H[i][j] += ds * dz[i] * dz[j] + r_ * d2.get((min(i, j), max(i, j)), 0)
        dAdz = [hp_ * xi, hp_, iv.mpf(0)]
        for i in range(3):
            c[i] += ds * _h(sg) * dz[i] + r_ * dAdz[i]
    H = [[H[i][j] / n for j in range(3)] for i in range(3)]
    c = [c[i] / n for i in range(3)]
    Hm = np.array([[_mid(H[i][j]) for j in range(3)] for i in range(3)])
    cm = np.array([_mid(c[i]) for i in range(3)])
    x0 = -np.linalg.solve(Hm, cm)
    Rv = [sum(H[i][j] * float(x0[j]) for j in range(3)) + c[i] for i in range(3)]
    Rn = math.sqrt(sum(max(abs(float(mp.mpf(v.a))), abs(float(mp.mpf(v.b)))) ** 2 for v in Rv))
    rad = math.sqrt(sum(_rad(H[i][j]) ** 2 for i in range(3) for j in range(3)))
    lam_mid = float(np.linalg.eigvalsh(Hm)[0])
    lam_lo = lam_mid - rad
    if lam_lo <= 0:
        return {"b2_validated": True, "hessian_pd": False}
    err = Rn / lam_lo
    dtheta = [iv.mpf([x0[i] - err, x0[i] + err]) for i in range(3)]
    # ∇G at the active points (A > 0 orientation: G0 = min_O h - max_I h)
    def cands(lo, hi, crit):
        out = [("end", lo), ("end", hi)]
        for cval in crit:            # interior critical point sigma = cval, if inside the sigma-range
            out.append(("crit", cval))
        return out
    grads = {}
    for name, (lo, hi), crit, want in (("inner", (-0.8, 0.8), [-R2], "max"),
                                       ("outer+", (1.2, 2.0), [R2], "min"),
                                       ("outer-", (-2.0, -1.2), [R2], "min")):
        vals = []
        for kind, v in cands(lo, hi, crit):
            if kind == "end":
                sg = P * v + Q
                vals.append((kind, v, _h(sg), (-1 + sg ** 2 / 2) * v, (-1 + sg ** 2 / 2)))
            else:
                # interior point where sigma = v: inside the window iff x = (v - q)/p in (lo, hi) over the box
                xs = (v - Q) / P
                inside = (xs.a > lo) and (xs.b < hi)
                maybe = not ((xs.b <= lo) or (xs.a >= hi))
                if inside:
                    vals.append((kind, v, iv.mpf(_h(mp.mpf(v))), iv.mpf(0), iv.mpf(0)))
                elif maybe:
                    return {"b2_validated": True, "hessian_pd": True, "active_set_unique": False,
                            "why": f"critical point may cross {name} window edge"}
        grads[name] = (vals, want)
    # outer min over both outer windows
    outer = grads["outer+"][0] + grads["outer-"][0]
    inner = grads["inner"][0]
    def unique_ext(vals, want):
        best = None
        for v in vals:
            dominated = all((v[2].b < o[2].a) if want == "min" else (v[2].a > o[2].b)
                            for o in vals if o is not v)
            if dominated:
                best = v
        return best
    om = unique_ext(outer, "min"); im = unique_ext(inner, "max")
    if om is None or im is None:
        return {"b2_validated": True, "hessian_pd": True, "active_set_unique": False}
    gp = om[3] - im[3]; gq = om[4] - im[4]
    dG = gp * dtheta[0] + gq * dtheta[1]
    _IFT_CACHE["last"] = {"dtheta": dtheta, "om": om, "im": im}
    return {"b2_validated": True, "hessian_pd": True, "active_set_unique": True,
            "hess_lambda_min_lo": lam_lo, "dG_dA_lo": float(mp.mpf(dG.a)), "dG_dA_hi": float(mp.mpf(dG.b)),
            "outer_active": f"{om[0]} {om[1]}", "inner_active": f"{im[0]} {im[1]}",
            "G_lo": float(mp.mpf((om[2] - im[2]).a)), "G_hi": float(mp.mpf((om[2] - im[2]).b))}


def h2(P=24.0):
    from .limit_bnb import certify, population, profile
    x, y = population()
    rows = []
    for A in (0.68125, 0.684375, 0.6875):
        for reg in ("-", "+"):
            r = certify(A, x, y, reg, P, tol=1e-11, max_cells=16_000_000)
            if r["encl"] is None or not np.isfinite(r["arg_p"]):
                continue
            _, b2, *_ = profile([r["arg_p"]], [r["arg_q"]], A, x, y)
            out = ift_limit(A, x, y, r["encl"], float(b2[0]))
            rows.append({"A": A, "region": reg, "upper": r["upper"], "lower": r["lower"],
                         "encl": str(tuple(round(v, 9) for v in r["encl"])), **out})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "mn2_h2prime.csv", index=False)


if __name__ == "__main__" and sys.argv[1] == "h2":
    h2()


# ----------------------------------------------------------------------------- solve threshold, finite a
def _fa_iv(t, a):
    return t + a * iv.sin(t)


def _fa_extrema_iv(Wiv, Biv, lo, hi, a):
    """Interval enclosure of (min, max) of f_a(w x + b) over x in [lo, hi], for all (w, b) in the box (w > 0).

    Exact structure: the extrema over t in [w lo + b, w hi + b] are attained at the two end t's or at the
    critical points t = ±c + 2πk (c = arccos(-1/a)), whose values f_a(±c + 2πk) are exact constants.  A
    critical point counts for the lower end of the min (or upper end of the max) if it MAY lie inside the
    t-range for some box member, and for the other end only if it lies inside for EVERY box member.
    """
    assert Wiv.a > 0
    T0 = Wiv * lo + Biv
    T1 = Wiv * hi + Biv
    f0, f1 = _fa_iv(T0, a), _fa_iv(T1, a)
    E = lambda v: mp.mpf(v)
    mn_lo, mn_hi = min(E(f0.a), E(f1.a)), min(E(f0.b), E(f1.b))
    mx_lo, mx_hi = max(E(f0.a), E(f1.a)), max(E(f0.b), E(f1.b))
    c = mp.acos(-mp.mpf(1) / a)
    two_pi = 2 * mp.pi
    t0a, t0b, t1a, t1b = (mp.mpf(v) for v in (T0.a, T0.b, T1.a, T1.b))
    kmin = int(mp.floor((t0a - c) / two_pi)) - 1
    kmax = int(mp.ceil((t1b + c) / two_pi)) + 1
    for k in range(kmin, kmax + 1):
        for tc, is_max in ((c + two_pi * k, True), (-c + two_pi * k, False)):
            val = iv.mpf(tc + a * mp.sin(tc))                     # exact constant, enclosed
            may = (tc >= t0a) and (tc <= t1b)
            surely = (tc > t0b) and (tc < t1a)
            if is_max:
                if may:
                    mx_hi = max(mx_hi, mp.mpf(val.b))
                if surely:
                    mx_lo = max(mx_lo, mp.mpf(val.a))
            else:
                if may:
                    mn_lo = min(mn_lo, mp.mpf(val.a))
                if surely:
                    mn_hi = min(mn_hi, mp.mpf(val.b))
    return iv.mpf([mn_lo, mn_hi]), iv.mpf([mx_lo, mx_hi])


def solve_finite():
    from . import blockB_landscape as bb
    from .profiled_bnb import certify, profile
    xb, yb = bb.population_data()
    xb, yb = xb.numpy(), yb.numpy()
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    rows = []
    for rr in br[br.kind == "solve"].itertuples():
        a = float(rr.a)
        for s in (float(rr.w2_lo), float(rr.w2_hi)):
            rm = certify(s, a, xb, yb, "-", tol=1e-11, max_cells=16_000_000)
            rp = certify(s, a, xb, yb, "+", tol=1e-11, max_cells=16_000_000)
            g = rp if rp["upper"] < rm["upper"] else rm
            enc = g.get("encl_pos")
            if enc is None:
                rows.append({"a": a, "s": s, "note": "no enclosure"}); continue
            Wiv, Biv = iv.mpf([enc[0], enc[1]]), iv.mpf([enc[2], enc[3]])
            wc, bc = 0.5 * (enc[0] + enc[1]), 0.5 * (enc[2] + enc[3])
            _, b2c, *_ = profile([wc], [bc], s, a, xb, yb)
            # b2 enclosure by interval Newton over the box
            B = iv.mpf([float(b2c[0]) - 1e-3, float(b2c[0]) + 1e-3])
            ybar = mp.mpf(int(yb.sum())) / len(yb)
            g_c = iv.mpf(0); g_b = iv.mpf(0)
            for xi in xb:
                t = Wiv * float(xi) + Biv
                g_c += _sig_iv(s * _fa_iv(t, a) + float(b2c[0]))
                g_b += _dsig_iv(_sig_iv(s * _fa_iv(t, a) + B))
            N = float(b2c[0]) - (g_c / len(xb) - ybar) / (g_b / len(xb))
            ok = (N.a >= B.a) and (N.b <= B.b)
            _, imax = _fa_extrema_iv(Wiv, Biv, -0.8, 0.8, a)
            o1, _ = _fa_extrema_iv(Wiv, Biv, 1.2, 2.0, a)
            o2, _ = _fa_extrema_iv(Wiv, Biv, -2.0, -1.2, a)
            omin = iv.mpf([min(o1.a, o2.a), min(o1.b, o2.b)])
            m1 = -(s * imax + N)
            m2 = s * omin + N
            margin = iv.mpf([min(m1.a, m2.a), min(m1.b, m2.b)])
            sign = "solves" if margin.a > 0 else ("fails" if margin.b < 0 else "undetermined")
            rows.append({"a": a, "s": s, "global_region": g["region"], "b2_validated": ok,
                         "encl_w1_width": enc[1] - enc[0], "encl_b1_width": enc[3] - enc[2],
                         "margin_lo": float(mp.mpf(margin.a)), "margin_hi": float(mp.mpf(margin.b)),
                         "certified_sign": sign})
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "mn2_solve_finite.csv", index=False)
    print(d.to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "solve_finite":
    solve_finite()


# ----------------------------------------------------------------------------- solve threshold, limit
def _h_extrema_iv(P, Q, lo, hi):
    """Interval (min, max) of h(p x + q) over x in [lo, hi], all (p, q) in the box (p > 0)."""
    assert P.a > 0
    S0, S1 = P * lo + Q, P * hi + Q
    f0, f1 = _h(S0), _h(S1)
    E = lambda v: mp.mpf(v)
    mn_lo, mn_hi = min(E(f0.a), E(f1.a)), min(E(f0.b), E(f1.b))
    mx_lo, mx_hi = max(E(f0.a), E(f1.a)), max(E(f0.b), E(f1.b))
    s0a, s0b, s1a, s1b = E(S0.a), E(S0.b), E(S1.a), E(S1.b)
    for sc, is_max in ((-mp.sqrt(2), True), (mp.sqrt(2), False)):
        val = -sc + sc ** 3 / 6
        may = (sc >= s0a) and (sc <= s1b)
        surely = (sc > s0b) and (sc < s1a)
        if is_max:
            if may: mx_hi = max(mx_hi, val)
            if surely: mx_lo = max(mx_lo, val)
        else:
            if may: mn_lo = min(mn_lo, val)
            if surely: mn_hi = min(mn_hi, val)
    return iv.mpf([mn_lo, mn_hi]), iv.mpf([mx_lo, mx_hi])


def _margin_limit_iv(P, Q, B, A):
    _, imax = _h_extrema_iv(P, Q, -0.8, 0.8)
    o1, _ = _h_extrema_iv(P, Q, 1.2, 2.0)
    o2, _ = _h_extrema_iv(P, Q, -2.0, -1.2)
    omin = iv.mpf([min(mp.mpf(o1.a), mp.mpf(o2.a)), min(mp.mpf(o1.b), mp.mpf(o2.b))])
    m1 = -(A * imax + B); m2 = A * omin + B
    return iv.mpf([min(mp.mpf(m1.a), mp.mpf(m2.a)), min(mp.mpf(m1.b), mp.mpf(m2.b))]), m1, m2


def solve_limit(P=24.0):
    from .limit_bnb import certify, population, profile, solve_margin0
    x, y = population()
    rows = []
    # 1. locate: float margin of the global argmin on a grid, then bisection to width 0.002
    def glob(A, tol=1e-7):
        rm = certify(A, x, y, "-", P, tol=tol); rp = certify(A, x, y, "+", P, tol=tol)
        return rp if rp["upper"] < rm["upper"] else rm
    grid = np.round(np.arange(0.96, 1.16 + 1e-9, 0.02), 4)
    marg = []
    for A in grid:
        g = glob(A)
        marg.append(solve_margin0(g["arg_p"], g["arg_q"], A, x, y))
        print(A, marg[-1], flush=True)
    i = next(j for j in range(1, len(grid)) if marg[j - 1] < 0 <= marg[j])
    lo, hi = grid[i - 1], grid[i]
    while hi - lo > 0.002:
        mid = round(0.5 * (lo + hi), 6)
        g = glob(mid)
        if solve_margin0(g["arg_p"], g["arg_q"], mid, x, y) >= 0:
            hi = mid
        else:
            lo = mid
    # 2. certify the sign at both ends over tight enclosures; Hessian PD and d(margin)/dA by IFT
    for A in (lo, hi):
        g = glob(A, tol=1e-11)
        enc = g["encl"]
        Pv, Qv = _box(enc)
        _, b2c, *_ = profile([0.5 * (enc[0] + enc[1])], [0.5 * (enc[2] + enc[3])], A, x, y)
        B, ok = b2_enclosure(Pv, Qv, iv.mpf(A), x, y, float(b2c[0]))
        M, m1, m2 = _margin_limit_iv(Pv, Qv, B, iv.mpf(A))
        ift = ift_limit(A, x, y, enc, float(b2c[0]))
        sign = "solves" if M.a > 0 else ("fails" if M.b < 0 else "undetermined")
        dM = (np.nan, np.nan)
        if ift.get("active_set_unique"):
            c_ = _IFT_CACHE["last"]; dth = c_["dtheta"]; om, im = c_["om"], c_["im"]
            if mp.mpf(m2.b) < mp.mpf(m1.a):          # outer term active: A*min_O h + b2
                d_ = om[2] + iv.mpf(A) * (om[3] * dth[0] + om[4] * dth[1]) + dth[2]
            elif mp.mpf(m1.b) < mp.mpf(m2.a):        # inner term active: -(A*max_I h + b2)
                d_ = -(im[2] + iv.mpf(A) * (im[3] * dth[0] + im[4] * dth[1]) + dth[2])
            else:
                d_ = None
            if d_ is not None:
                dM = (float(d_.a), float(d_.b))
        rows.append({"A": A, "region": g["region"], "b2_validated": ok, "margin_lo": float(mp.mpf(M.a)),
                     "margin_hi": float(mp.mpf(M.b)), "certified_sign": sign,
                     "hess_lambda_min_lo": ift.get("hess_lambda_min_lo"),
                     "dmargin_dA_lo": dM[0], "dmargin_dA_hi": dM[1], "encl": str(enc)})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d["A_solve_lo"], d["A_solve_hi"] = lo, hi
    from .limit_bnb import K_LIMIT
    d["R_solve_inf_lo"], d["R_solve_inf_hi"] = K_LIMIT * lo / 2, K_LIMIT * hi / 2
    d.to_csv(RESULTS / "mn2_solve_limit.csv", index=False)
    print(d.to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "solve_limit":
    solve_limit()


# ----------------------------------------------------------------------------- neighbourhood U and constants
RHO = 0.05
SPLIT = (10, 10, 5)          # sub-boxes in p, q, A for the certified PD check
A_RANGE = (0.66, 0.71)
EPS_MAX = 0.05


def taylor_constants(S, eps):
    """Exact remainder bounds for φ_ε(σ) = [f_a(π+√εσ) − π]/ε^{3/2} against h on |σ| <= S (derivation in the note).

    φ_ε    = h + ε r + ε² e0,  r = σ³/6 − σ⁵/120,   |e0| <= S⁵/120 + (1+ε) S⁷/5040
    φ_ε'   = h' + ε r' + ε² e1, r' = σ²/2 − σ⁴/24,  |e1| <= S⁴/24 + (1+ε) S⁶/720
    φ_ε''  = h'' + ε r'' + ε² e2, r'' = σ − σ³/6,   |e2| <= S³/6 + (1+ε) S⁵/120
    φ_ε''' − h''' = (1+ε) cos(√εσ) − 1,              |·| <= ε + (1+ε) ε S²/2
    Returns M0..M3 with |φ_ε^{(k)} − h^{(k)}| <= ε M_k.
    """
    sig = np.linspace(-S, S, 200001)
    r = sig ** 3 / 6 - sig ** 5 / 120
    r1 = sig ** 2 / 2 - sig ** 4 / 24
    r2 = sig - sig ** 3 / 6
    M0 = np.abs(r).max() + eps * (S ** 5 / 120 + (1 + eps) * S ** 7 / 5040)
    M1 = np.abs(r1).max() + eps * (S ** 4 / 24 + (1 + eps) * S ** 6 / 720)
    M2 = np.abs(r2).max() + eps * (S ** 3 / 6 + (1 + eps) * S ** 5 / 120)
    M3 = 1 + (1 + eps) * S ** 2 / 2
    return {"S": S, "eps_max": eps, "M0": M0, "M1": M1, "M2": M2, "M3": M3}


def _box_hessian_pd(args):
    """Certified PD of the (p, q, b) Hessian over one sub-box, with b enclosed by interval Newton."""
    pl, ph, ql, qh, al, ah = args
    from .limit_bnb import population, profile
    x, y = population()
    iv.dps = 30
    P, Q, Aiv = iv.mpf([pl, ph]), iv.mpf([ql, qh]), iv.mpf([al, ah])
    _, b2c, *_ = profile([0.5 * (pl + ph)], [0.5 * (ql + qh)], 0.5 * (al + ah), x, y)
    ybar = mp.mpf(int(y.sum())) / len(y)
    ok, r, N = False, 0.01, None
    while not ok and r < 2:
        B = iv.mpf([float(b2c[0]) - r, float(b2c[0]) + r])
        gc = iv.mpf(0); gb = iv.mpf(0)
        for xi in x:
            sg = P * float(xi) + Q
            gc += _sig_iv(Aiv * _h(sg) + float(b2c[0]))
            gb += _dsig_iv(_sig_iv(Aiv * _h(sg) + B))
        N = float(b2c[0]) - (gc / len(x) - ybar) / (gb / len(x))
        ok = (mp.mpf(N.a) >= mp.mpf(B.a)) and (mp.mpf(N.b) <= mp.mpf(B.b))
        r *= 2
    if not ok:
        return {"box": args, "b2_ok": False, "pd": False, "lam_lo": np.nan}
    H = [[iv.mpf(0)] * 3 for _ in range(3)]
    for xi, yi in zip(x, y):
        xi = float(xi)
        sg = P * xi + Q
        hp_ = -1 + sg ** 2 / 2
        z = Aiv * _h(sg) + N
        s_ = _sig_iv(z); ds = _dsig_iv(s_)
        r_ = s_ - int(yi)
        dz = [Aiv * hp_ * xi, Aiv * hp_, iv.mpf(1)]
        d2 = {(0, 0): Aiv * sg * xi * xi, (0, 1): Aiv * sg * xi, (1, 1): Aiv * sg}
        for i in range(3):
            for j in range(3):
                H[i][j] += ds * dz[i] * dz[j] + r_ * d2.get((min(i, j), max(i, j)), 0)
    H = [[H[i][j] / len(x) for j in range(3)] for i in range(3)]
    Hm = np.array([[_mid(H[i][j]) for j in range(3)] for i in range(3)])
    rad = math.sqrt(sum(_rad(H[i][j]) ** 2 for i in range(3) for j in range(3)))
    lam = float(np.linalg.eigvalsh(Hm)[0])
    return {"box": args, "b2_ok": True, "pd": lam - rad > 0, "lam_lo": lam - rad}


def neighbourhood(workers=1):
    from multiprocessing import Pool
    ls = pd.read_csv(RESULTS / "limit_switch.csv").iloc[0]
    p0, q0 = float(ls.argmin_p), float(ls.argmin_q)
    S_U = 2 * (abs(p0) + RHO) + abs(q0) + RHO
    pe = np.linspace(p0 - RHO, p0 + RHO, SPLIT[0] + 1)
    qe = np.linspace(q0 - RHO, q0 + RHO, SPLIT[1] + 1)
    ae = np.linspace(A_RANGE[0], A_RANGE[1], SPLIT[2] + 1)
    boxes = [(pe[i], pe[i + 1], qe[j], qe[j + 1], ae[k], ae[k + 1])
             for i in range(SPLIT[0]) for j in range(SPLIT[1]) for k in range(SPLIT[2])]
    with Pool(workers) as pool:
        res = pool.map(_box_hessian_pd, boxes)
    lam_lo = [r["lam_lo"] for r in res]
    out = {"p0": p0, "q0": q0, "rho": RHO, "A_lo": A_RANGE[0], "A_hi": A_RANGE[1], "S_U": S_U,
           "sub_boxes": len(boxes), "b2_validated_all": all(r["b2_ok"] for r in res),
           "hess_pd_all_boxes": all(r["pd"] for r in res),
           "hess_lambda_min_lower_bound": float(np.nanmin(lam_lo)), **taylor_constants(S_U, EPS_MAX)}
    pd.DataFrame([out]).to_csv(RESULTS / "mn2_neighbourhood.csv", index=False)
    print(pd.Series(out).to_string())


if __name__ == "__main__" and sys.argv[1] == "neighbourhood":
    neighbourhood(int(sys.argv[2]) if len(sys.argv) > 2 else 1)


# ----------------------------------------------------------------------------- annulus / uniqueness over A
A_SUBINTERVALS = 8


def annulus_job(args):
    """Certified lower bound of L0*(p, q; A) over K(P) minus the ρ-box around the branch argmin, uniformly for
    A in [A_lo, A_hi], and a certified upper bound of the branch loss over the same A-interval.

    Uniformity in A: for fixed θ, L0*(θ; A) = min_b mean ℓ(A h(σ) + b) is convex in A (ℓ is convex and the
    logit is jointly linear in (A, b)), so L0*(θ; A) >= L0*(θ; A_c) - h_A |∂_A L0*(θ; A_c)| on the interval,
    and |∂_A L0*| = |mean((σ(z) - y) h(σ))| <= mean |h(σ)|, bounded per cell by |h| at the cell's max |σ|.
    Branch upper bound: m(A) <= L0*(θ_c; A_c) + h_A mean|h(σ_c)| (the loss is 1-Lipschitz in the logit and the
    profiled bias can only lower it).
    """
    A_lo, A_hi, P, p0, q0, rho, tol, h0 = args
    from .limit_bnb import population, profile
    x, y = population()
    Ac, hA = 0.5 * (A_lo + A_hi), 0.5 * (A_hi - A_lo)
    Q = 2 * math.sqrt(2) + 2 * P
    npn = int(math.ceil(P / h0)); nq = int(math.ceil(2 * Q / h0))
    hp, hq = P / npn / 2, Q / nq
    cp, cq = np.meshgrid((np.arange(npn) + 0.5) * 2 * hp, -Q + (np.arange(nq) + 0.5) * 2 * hq, indexing="ij")
    cp, cq = cp.ravel(), cq.ravel()
    ax = np.abs(x)
    # branch upper bound over the A-interval, from the argmin at A_c
    Lc, _, _, _ = profile([p0], [q0], Ac, x, y)
    sig0 = p0 * x + q0
    branch_upper = float(Lc[0]) + hA * float(np.mean(np.abs(-sig0 + sig0 ** 3 / 6)))
    upper, rounds = math.inf, 0
    while True:
        rounds += 1
        inbox = (np.abs(cp - p0) + hp <= rho) & (np.abs(cq - q0) + hq <= rho)     # cell entirely in the box
        cp, cq = cp[~inbox], cq[~inbox]
        n = len(cp)
        lb = np.empty(n); Lv = np.empty(n)
        for i in range(0, n, 2000):
            sl = slice(i, i + 2000)
            L, _, gp, gq = profile(cp[sl], cq[sl], Ac, x, y)
            d = ax[None, :] * hp + hq
            smax = np.abs(cp[sl, None] * x[None, :] + cq[sl, None]) + d
            curv = 0.5 * Ac * (smax * d ** 2).mean(axis=1)
            habs = (smax + smax ** 3 / 6).mean(axis=1)                         # >= mean |h(σ)| over the cell
            lb[sl] = L - np.abs(gp) * hp - np.abs(gq) * hq - curv - hA * habs
            Lv[sl] = L
        outside = ~((np.abs(cp - p0) < rho) & (np.abs(cq - q0) < rho))
        if outside.any():
            upper = min(upper, float(Lv[outside].min()))
        keep = lb < min(upper, branch_upper + 1.0)
        lower = float(lb[keep].min()) if keep.any() else upper
        done = (lower > branch_upper) or (upper - lower <= tol) or not keep.any() or rounds > 30
        if done:
            return {"A_lo": A_lo, "A_hi": A_hi, "branch_upper": branch_upper, "competitor_lower": lower,
                    "competitor_attained": upper, "margin": lower - branch_upper, "rounds": rounds,
                    "certified": lower > branch_upper}
        cp, cq = cp[keep], cq[keep]
        hp, hq = hp / 2, hq / 2
        cp = np.concatenate([cp - hp, cp - hp, cp + hp, cp + hp])
        cq = np.concatenate([cq - hq, cq + hq, cq - hq, cq + hq])


def annulus(workers=1, P=24.0):
    from multiprocessing import Pool
    ls = pd.read_csv(RESULTS / "limit_switch.csv").iloc[0]
    p0, q0 = float(ls.argmin_p), float(ls.argmin_q)
    edges = np.linspace(A_RANGE[0], A_RANGE[1], A_SUBINTERVALS + 1)
    jobs = [(edges[i], edges[i + 1], P, p0, q0, RHO, 1e-7, 0.25) for i in range(A_SUBINTERVALS)]
    with Pool(workers) as pool:
        rows = pool.map(annulus_job, jobs)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "mn2_annulus.csv", index=False)
    print(d.to_string(index=False))
    print("all A-subintervals certified:", bool(d.certified.all()), " min margin:", d.margin.min())


if __name__ == "__main__" and sys.argv[1] == "annulus":
    annulus(int(sys.argv[2]) if len(sys.argv) > 2 else 1)


def uniform_summary(A_lo=0.66, A_hi=0.72):
    """Restrict the uniformity bound to [A_lo, A_hi] (grid points and bridges inside it)."""
    d = pd.read_csv(RESULTS / "mn2_uniformity.csv")
    inside = d[(d.A >= A_lo - 1e-9) & (d.A <= A_hi + 1e-9)].reset_index(drop=True)
    bridges = inside.sup_bridge_to_next.iloc[:-1]          # bridges between consecutive grid points inside
    sup_ = float(max(inside.U.max(), bridges.max()))
    b24 = float(mp.mpf("0.38797358308678997649"))
    bfull = float(mp.log(3) / 4 + mp.log(mp.mpf(3) / 2) / 2)
    full_sup = float(d.sup_over_neighbourhood.iloc[0])
    out = pd.DataFrame([{"A_lo": A_lo, "A_hi": A_hi, "sup_branch_loss": sup_, "margin_B24": b24 - sup_,
                         "margin_Bfull": bfull - sup_, "certified": sup_ < b24,
                         "wider_range": "0.60-0.76", "wider_sup": full_sup, "wider_margin_B24": b24 - full_sup,
                         "wider_certified": full_sup < b24}])
    out.to_csv(RESULTS / "mn2_uniformity_summary.csv", index=False)
    print(out.T.to_string())


if __name__ == "__main__" and sys.argv[1] == "uniform_summary":
    uniform_summary()
