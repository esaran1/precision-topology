"""§16: local to global near the limit (Track 3 (1)).  The continued branch is the global conditional minimiser, and its
gap changes sign exactly once, for every A in [0.66, 0.71] and every 0 < eps <= EPS0, with EPS0 explicit.

Everything here is either an analytic bound (evaluated in floats with a stated safety factor, and all quantities
monotone in the sup |sigma| used) or a parametric Krawczyk certificate in mpmath interval arithmetic.  The limit
certificates it rests on are read from the committed files (certv2_annulus_summary.csv, certv2_annulus_parts.csv,
mn2_bounds.csv, first_order_c1.csv).

    python -m src.local_global          # -> results/local_global.json   (one process, a few minutes)
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from mpmath import iv

RESULTS = Path(__file__).resolve().parents[1] / "results"
iv.dps = 30
mp.mp.dps = 30
R2 = math.sqrt(2)
A_RANGE = (0.66, 0.71)
P_BOX = 24.0
RZ = [6e-4, 6e-4, 2e-4, 5e-4]         # switch-box radii (p, q, b, A); set from the float tangent below


# ------------------------------------------------------------------ pointwise remainder bounds (proved in §16.1)
def m0(S): return S ** 5 / 120 + S ** 3 / 6            # |phi_eps - h|   <= eps m0(|sigma|)
def m1(S): return S ** 4 / 24 + S ** 2 / 2             # |phi_eps' - h'| <= eps m1(|sigma|)
def m2(S): return S ** 3 / 6 + S                       # |phi_eps''- h''|<= eps m2(|sigma|)


def limit_inputs():
    summ = pd.read_csv(RESULTS / "certv2_annulus_summary.csv").iloc[0]
    parts = pd.read_csv(RESULTS / "certv2_annulus_parts.csv")
    parts = parts[parts.tag == "annulus"]
    assert bool(summ.covered) and int(summ.sub_intervals_certified) == int(summ.sub_intervals_evaluated) == 40
    assert parts.certified.all() and parts.localised.all()
    pc, qc = float(parts.centre_p.iloc[0]), float(parts.centre_q.iloc[0])
    assert (parts.centre_p == pc).all() and (parts.centre_q == qc).all()
    b = pd.read_csv(RESULTS / "mn2_bounds.csv").set_index("quantity").value
    return {"outer_margin": float(summ.min_outer_margin), "ring_grad_lower": float(summ.min_ring_grad_lower),
            "branch_upper": float(summ.max_branch_upper), "pd_lambda": float(summ.pd_lambda_min_lower),
            "B24": float(b["B(24) (exact PAVA, 50 digits)"]), "B_full": float(b["B_full (exact PAVA, 50 digits)"]),
            "centre": (pc, qc), "A_lo": float(parts.A_lo.min()), "A_hi": float(parts.A_hi.max())}


def radii(c):
    pc, qc = c["centre"]
    return {"S_c": 2 * abs(pc) + abs(qc),                                  # |sigma| at theta_c (x in [-2, 2])
            "S_U": 2 * (abs(pc) + 0.05) + abs(qc) + 0.05,                  # 0.05 box (link 4)
            "S_R": 2 * (abs(pc) + 0.15) + abs(qc) + 0.15,                  # 0.15 box (ring, link 3)
            "S_K": 2 * P_BOX + 2 * R2 + 2 * P_BOX}                           # K(24): |p| <= 24, |q| <= 2 sqrt2 + 48


def link_conditions(eps, c, A=0.71):
    """The perturbation of each limit link at eps, against its certified margin (§16.2).  Returns slack >= 0 iff the
    link transfers to every eps' in (0, eps] (all bounds are increasing in eps)."""
    r = radii(c)
    Sc, SU, SR, SK = r["S_c"], r["S_U"], r["S_R"], r["S_K"]
    out = {}
    # branch loss at finite eps (upper bound) against the localisation bounds
    U = c["branch_upper"] + A * eps * m0(Sc)
    out["localisation"] = {"branch_upper_eps": U, "B24": c["B24"], "B_full": c["B_full"], "log2": math.log(2),
                           "slack": min(c["B24"], c["B_full"], math.log(2)) - U}
    # link 2: outer exclusion on K(24) \ 0.15 box
    pert = A * eps * (m0(SK) + m0(Sc))
    out["outer"] = {"margin": c["outer_margin"], "perturbation": pert, "slack": c["outer_margin"] - pert}
    # link 3: ring, gradient components
    grad = 2 * A * eps * (m1(SR) + (SR ** 2 / 2 + 1) * (A / 2) * m0(SR))
    out["ring"] = {"margin": c["ring_grad_lower"], "perturbation": grad, "slack": c["ring_grad_lower"] - grad}
    # link 4: PD of the (p, q, b) Hessian on the 0.05 box, including the shift of b* by <= A eps m0
    d = max(1.0, 2 * A * (SU ** 2 / 2 + 1 + eps * m1(SU)))
    e_entry = (1 / (6 * math.sqrt(3))) * A * eps * m0(SU) * d * d + d * 2 * A * eps * m1(SU) \
        + A * A * eps * m0(SU) * (SU + eps * m2(SU)) + 4 * A * eps * m2(SU)
    Lb = (1 / (6 * math.sqrt(3))) * d * d + A * SU
    dH = 3 * e_entry + 3 * Lb * A * eps * m0(SU)
    out["pd"] = {"margin": c["pd_lambda"], "perturbation": dH, "slack": c["pd_lambda"] - dH}
    return out


def fold_window_conditions(eps, x, y):
    """Finite-eps localisation (§16.1): the positive zero u0 of u - a sin u satisfies u0 < 2 sqrt2 sqrt(eps), and the
    t-range of every |w1| <= W(s, a) contains at most one fold window: 4W + u0 < 2 pi - beta."""
    from .profiled_bnb import w_bound
    a = 1 + eps
    c = 2 * R2
    lower_g = -c + (1 + eps) * c ** 3 / 6 - (1 + eps) * eps * c ** 5 / 120      # g(c sqrt eps)/eps^{3/2} >= this
    beta = math.acos(1 / a)
    s_min = A_RANGE[0] / eps ** 1.5
    W = w_bound(s_min, a, x, y)
    u0_max = c * math.sqrt(eps)
    return {"g_at_2sqrt2_scaled_lower": lower_g, "u0_below_2sqrt2_sqrt_eps": lower_g > 0,
            "W_max": W, "beta": beta, "four_W_plus_u0": 4 * W + u0_max, "two_pi_minus_beta": 2 * math.pi - beta,
            "one_fold": 4 * W + u0_max < 2 * math.pi - beta}


# ------------------------------------------------------------------ interval arithmetic with eps inflation
class Inflated:
    """phi_eps and its first two derivatives enclosed for every eps in [0, eps0]: h + [-1, 1] eps0 m_k(|sigma|)."""

    def __init__(self, eps0):
        self.e = eps0

    def _mag(self, s):
        return max(abs(mp.mpf(s.a)), abs(mp.mpf(s.b)))

    def phi(self, s):
        e = self.e * m0(float(self._mag(s)))
        return -s + s ** 3 / 6 + iv.mpf([-e, e])

    def phi1(self, s):
        e = self.e * m1(float(self._mag(s)))
        return -1 + s ** 2 / 2 + iv.mpf([-e, e])

    def phi2(self, s):
        e = self.e * m2(float(self._mag(s)))
        return s + iv.mpf([-e, e])


def _sig(z):
    f = lambda t: 1 / (1 + mp.exp(-t))
    return iv.mpf([f(z.a), f(z.b)])


def _dsig(S):
    lo, hi = S.a, S.b
    vals = [lo * (1 - lo), hi * (1 - hi)]
    return iv.mpf([min(vals), mp.mpf("0.25") if lo <= 0.5 <= hi else max(vals)])


def _mid(v): return float((mp.mpf(v.a) + mp.mpf(v.b)) / 2)
def _lo(v): return float(mp.mpf(v.a))
def _hi(v): return float(mp.mpf(v.b))


def system(X, A, x, y, ph, with_gap=False, want_J=True):
    """Branch system F = grad_{p,q,b} L (3 eqs; A a parameter interval), or the switch system (4 eqs, A unknown,
    with G = phi(q - 1.2 p) - phi(q + 0.8 p) on the certified active pair)."""
    p, q, b = X[0], X[1], X[2]
    if with_gap:
        A = X[3]
    n = len(x)
    k = 4 if with_gap else 3
    F = [iv.mpf(0)] * 3
    J = [[iv.mpf(0)] * k for _ in range(3)]
    for xi, yi in zip(x, y):
        xi = float(xi)
        s = p * xi + q
        f0, f1, f2 = ph.phi(s), ph.phi1(s), ph.phi2(s)
        z = A * f0 + b
        S = _sig(z); res = S - int(yi)
        dz = [A * f1 * xi, A * f1, iv.mpf(1)]
        for i in range(3):
            F[i] += res * dz[i]
        if not want_J:
            continue
        w = _dsig(S)
        d2 = [[A * f2 * xi * xi, A * f2 * xi, 0], [A * f2 * xi, A * f2, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                J[i][j] += w * dz[i] * dz[j] + res * d2[i][j]
            if with_gap:
                dA = [f1 * xi, f1, iv.mpf(0)]
                J[i][3] += w * f0 * dz[i] + res * dA[i]
    F = [f / n for f in F]
    if with_gap:
        sO, sI = q - 1.2 * p, q + 0.8 * p
        F.append(ph.phi(sO) - ph.phi(sI))
    if not want_J:
        return F
    J = [[J[i][j] / n for j in range(k)] for i in range(3)]
    if with_gap:
        sO, sI = q - 1.2 * p, q + 0.8 * p
        J.append([-1.2 * ph.phi1(sO) - 0.8 * ph.phi1(sI), ph.phi1(sO) - ph.phi1(sI), iv.mpf(0), iv.mpf(0)])
    return F, J


def krawczyk(fun, xt, rad):
    k = len(xt)
    X = [iv.mpf([mp.mpf(xt[i]) - rad[i], mp.mpf(xt[i]) + rad[i]]) for i in range(k)]
    F0 = fun([iv.mpf(mp.mpf(t)) for t in xt], False)
    _, J = fun(X, True)
    Jm = np.array([[_mid(J[i][j]) for j in range(k)] for i in range(k)])
    Y = np.linalg.inv(Jm)
    Kb = []
    for i in range(k):
        acc = iv.mpf(mp.mpf(xt[i]))
        for j in range(k):
            acc -= float(Y[i, j]) * F0[j]
        for j in range(k):
            cij = (1.0 if i == j else 0.0) - sum(float(Y[i, l]) * J[l][j] for l in range(k))
            acc += cij * (X[j] - mp.mpf(xt[j]))
        Kb.append(acc)
    ok = all((Kb[i].a > X[i].a) and (Kb[i].b < X[i].b) for i in range(k))
    return ok, Kb, X


def _h_extrema(lo, hi, ph):
    """Enclosure of min and max of phi over [lo, hi] (lo, hi floats); critical points of h at +-sqrt2 (inflation covers
    the O(eps) shift of phi_eps's critical points: phi_eps' is within eps0 m1 of h', so on the interval the extremum
    of phi_eps is within eps0 m0 of that of h at the same points; we add the inflation to every candidate)."""
    cands = [lo, hi] + [c for c in (-R2, R2) if lo < c < hi]
    vals = [ph.phi(iv.mpf(mp.mpf(c))) for c in cands]
    e = ph.e * m0(max(abs(lo), abs(hi)))
    # min over the interval is >= min over candidates of h minus e (h's extrema are at candidates; phi within e of h)
    mn = min(_lo(v) for v in vals) - e
    mx = max(_hi(v) for v in vals) + e
    return mn, mx


def gap_bounds(P, Q, ph):
    """Certified [lower, upper] for G(p, q) = min_O phi - max_I phi over the box P x Q (A > 0 orientation)."""
    plo, phi_, qlo, qhi = _lo(P), _hi(P), _lo(Q), _hi(Q)

    def srange(x1, x2):
        c = [pp * xx + qq for pp in (plo, phi_) for xx in (x1, x2) for qq in (qlo, qhi)]
        return min(c), max(c)
    minO = min(_h_extrema(*srange(1.2, 2.0), ph)[0], _h_extrema(*srange(-2.0, -1.2), ph)[0])
    maxI = _h_extrema(*srange(-0.8, 0.8), ph)[1]
    lower = minO - maxI
    # upper: pick single points
    upO = min(_hi(ph.phi(P * xo + Q)) for xo in (-2.0, -1.2, 1.2, 2.0))
    loI = max(_lo(ph.phi(P * xi + Q)) for xi in (-0.8, 0.0, 0.8))
    upper = upO - loI
    return lower, upper


def branch_newton(A, v, x, y):
    ph = Inflated(0.0)
    for _ in range(30):
        F, J = system([iv.mpf(mp.mpf(t)) for t in v], iv.mpf(mp.mpf(A)), x, y, ph)
        Fm = np.array([_mid(f) for f in F]); Jm = np.array([[_mid(t) for t in r_] for r_ in J])
        v = v - np.linalg.solve(Jm, Fm)
    return v, Jm


def gap_part(eps0, c, x, y, log=print):
    """§16.3: the branch's gap along A in [0.66, 0.71] for all eps in [0, eps0].  Returns the certified structure."""
    ph = Inflated(eps0)
    pc, qc = c["centre"]
    fo = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    Astar = 0.5 * (float(fo.A_star_lo) + float(fo.A_star_hi))
    from .limit_bnb import profile
    _, b2, *_ = profile([pc], [qc], Astar, x, y)
    v = np.array([float(fo.p_star), float(fo.q_star), float(fo.b_star)])
    # switch box Z (4-var Krawczyk)
    rZ = RZ
    zf = np.array([float(fo.p_star), float(fo.q_star), float(fo.b_star), Astar])
    funZ = lambda X, J=True: system(X, None, x, y, ph, with_gap=True, want_J=J)
    okZ, KZ, XZ = krawczyk(funZ, zf, rZ)
    # active pair on XZ (inflated)
    p, q = XZ[0], XZ[1]
    I_act, O_act = ph.phi(q + 0.8 * p), ph.phi(q - 1.2 * p)
    others_I = [ph.phi(q - 0.8 * p)]
    others_O = [ph.phi(q - 2.0 * p), ph.phi(q + 1.2 * p), ph.phi(q + 2.0 * p)]
    sI0, sOm1, sOp0 = q - 0.8 * p, q - 1.2 * p, q + 1.2 * p
    dcrit = 0.01                                   # |beta~ - sqrt2| <= sqrt2 eps0 << 0.01 (§16.3)
    crit_ok = (sI0.a > -R2 + dcrit) and (sOm1.b < R2 - dcrit) and (sOp0.a > R2 + dcrit)
    active_ok = all(I_act.a > o.b for o in others_I) and all(O_act.b < o.a for o in others_O) and crit_ok
    ZA = (_lo(XZ[3]), _hi(XZ[3]))
    Ztheta = [(_lo(XZ[i]), _hi(XZ[i])) for i in range(3)]
    log({"switch_box_ok": okZ, "active_ok": active_ok, "ZA": ZA})
    # 3-var boxes over A sub-intervals covering [0.66, 0.71]
    def geometric(a_from, a_to, toward_hi):
        """edges from a_from to a_to with widths 0.25 x distance to the Z edge, clipped to [2e-5, 2e-3]."""
        out = [a_from]
        while abs(out[-1] - a_to) > 1e-15:
            d = abs(a_to - out[-1])
            w = min(2e-3, max(2e-5, 0.25 * d))
            nxt = out[-1] + w if toward_hi else out[-1] - w
            nxt = min(nxt, a_to) if toward_hi else max(nxt, a_to)
            out.append(nxt)
        return out
    left = geometric(A_RANGE[0], ZA[0], True)
    right = geometric(A_RANGE[1], ZA[1], False)[::-1]
    nin = max(1, int(math.ceil((ZA[1] - ZA[0]) / 2e-5)))
    inside = [ZA[0] + (ZA[1] - ZA[0]) * k / nin for k in range(nin + 1)]
    edges = left[:-1] + inside[:-1] + right
    rows = []
    cache = {}

    def one(a_lo, a_hi):
        Ac = 0.5 * (a_lo + a_hi)
        near = min(cache, key=lambda k: abs(k - Ac)) if cache else None
        v0 = cache[near] if near is not None else v
        vn, Jm = branch_newton(Ac, v0, x, y)
        cache[Ac] = vn
        F0, J0 = system([iv.mpf(mp.mpf(t)) for t in vn], iv.mpf(mp.mpf(Ac)), x, y, Inflated(0.0))
        dA = 1e-6
        Fp = system([iv.mpf(mp.mpf(t)) for t in vn], iv.mpf(mp.mpf(Ac + dA)), x, y, Inflated(0.0), want_J=False)
        dF = np.array([(_mid(Fp[k]) - _mid(F0[k])) / dA for k in range(3)])
        th1 = -np.linalg.solve(Jm, dF)
        wA = a_hi - a_lo
        Aint = iv.mpf([mp.mpf(a_lo), mp.mpf(a_hi)])
        fun = lambda Xb, J=True: system(Xb, Aint, x, y, ph, want_J=J)
        rad = [abs(th1[k]) * wA / 2 + 1e-7 for k in range(3)]
        ok, Kb = False, None
        for _ in range(12):                        # epsilon-inflation: grow the box to 1.5 x the Krawczyk image
            ok, Kb, _X = krawczyk(fun, vn, rad)
            if ok:
                break
            rad = [max(rad[k], 1.5 * max(abs(_lo(Kb[k]) - vn[k]), abs(_hi(Kb[k]) - vn[k]))) for k in range(3)]
        inU = ok and all(_lo(Kb[k]) > (pc, qc)[k] - 0.05 and _hi(Kb[k]) < (pc, qc)[k] + 0.05 for k in range(2))
        glo, ghi = gap_bounds(Kb[0], Kb[1], ph) if ok else (np.nan, np.nan)
        inZ = ok and all(_lo(Kb[k]) > Ztheta[k][0] and _hi(Kb[k]) < Ztheta[k][1] for k in range(3))
        return {"A_lo": a_lo, "A_hi": a_hi, "krawczyk_ok": ok, "in_U": inU, "G_lo": glo, "G_hi": ghi,
                "sign": (-1 if ghi < 0 else 1 if glo > 0 else 0) if ok else None,
                "within_Z_A": ZA[0] <= a_lo and a_hi <= ZA[1], "box_in_Z_theta": inZ,
                "rad_p": _hi(Kb[0]) - _lo(Kb[0]) if ok else None}

    # process from A* outward (continuation); a sub-interval outside Z whose gap sign is undecided is bisected
    queue = sorted([(edges[i], edges[i + 1]) for i in range(len(edges) - 1)],
                   key=lambda e: abs(0.5 * (e[0] + e[1]) - Astar))
    while queue:
        a_lo, a_hi = queue.pop(0)
        r_ = one(a_lo, a_hi)
        if (not r_["within_Z_A"]) and r_["krawczyk_ok"] and r_["sign"] == 0 and (a_hi - a_lo) > 1e-7:
            m_ = 0.5 * (a_lo + a_hi)
            queue[:0] = [(a_lo, m_), (m_, a_hi)]
            continue
        rows.append(r_)
    rows.sort(key=lambda r_: r_["A_lo"])
    return {"switch_box_ok": okZ, "active_pair_ok": active_ok, "ZA": ZA, "Ztheta": Ztheta, "rows": rows}


def assess(gp):
    rows = gp["rows"]
    ZA = gp["ZA"]
    all_k = all(r_["krawczyk_ok"] and r_["in_U"] for r_ in rows)
    left = [r_ for r_ in rows if r_["A_hi"] <= ZA[0] + 1e-15]
    right = [r_ for r_ in rows if r_["A_lo"] >= ZA[1] - 1e-15]
    inside = [r_ for r_ in rows if r_ not in left and r_ not in right]
    cover = abs(rows[0]["A_lo"] - A_RANGE[0]) < 1e-15 and abs(rows[-1]["A_hi"] - A_RANGE[1]) < 1e-15 and \
        all(abs(rows[i]["A_hi"] - rows[i + 1]["A_lo"]) < 1e-15 for i in range(len(rows) - 1))
    return {"n_subintervals": len(rows), "covered": cover, "all_krawczyk_in_U": all_k,
            "left_all_negative": all(r_["sign"] == -1 for r_ in left),
            "right_all_positive": all(r_["sign"] == 1 for r_ in right),
            "inside_boxes_in_Z": all(r_["box_in_Z_theta"] for r_ in inside), "n_inside": len(inside),
            "unique_sign_change": bool(cover and all_k and gp["switch_box_ok"] and gp["active_pair_ok"]
                                       and all(r_["sign"] == -1 for r_ in left) and all(r_["sign"] == 1 for r_ in right)
                                       and all(r_["box_in_Z_theta"] for r_ in inside))}


def main():
    os.nice(15)
    t0 = time.time()
    from .limit_bnb import population
    x, y = population()
    c = limit_inputs()
    # EPS0: the largest eps with every link slack > 0, from the binding (outer) link, with a 0.8 safety factor
    r = radii(c)
    eps_outer = c["outer_margin"] / (0.71 * (m0(r["S_K"]) + m0(r["S_c"])))
    EPS0 = float(f"{0.8 * eps_outer:.2g}")
    links = link_conditions(EPS0, c)
    fold = fold_window_conditions(EPS0, x, y)
    gp = gap_part(EPS0, c, x, y, log=lambda d: print(d, flush=True))
    res = {"EPS0": EPS0, "eps_outer_limit": eps_outer, "radii": r, "limit_inputs": c, "links": links,
           "fold_window": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in fold.items()},
           "gap": assess(gp), "gap_rows": gp["rows"], "switch_box": {"ok": gp["switch_box_ok"],
           "active_pair_ok": gp["active_pair_ok"], "A_range": gp["ZA"], "theta_ranges": gp["Ztheta"]},
           "all_links_slack_positive": all(v["slack"] > 0 for v in links.values()),
           "seconds": time.time() - t0}
    res["theorem_G_holds"] = bool(res["all_links_slack_positive"] and fold["u0_below_2sqrt2_sqrt_eps"]
                                  and fold["one_fold"] and res["gap"]["unique_sign_change"])
    (RESULTS / "local_global.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "gap_rows"}, indent=1, default=float))


if __name__ == "__main__":
    main()
