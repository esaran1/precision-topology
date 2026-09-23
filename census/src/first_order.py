"""First-order coefficient of R_glob(a) = R_glob^inf (1 + c1 ε + O(ε²)), as a certified interval (math_note_v2 §8).

φ_ε(σ) = [f_a(π + √ε σ) − π]/ε^{3/2} = h(σ) + ε r(σ) + O(ε²),  h = −σ + σ³/6,  r = σ³/6 − σ⁵/120.
R = sĜ(a)/2 = A_ε K(ε)/2 with A = sε^{3/2} and K(ε) = Ĝ(a)/ε^{3/2}, so c1 = A'(0)/A* + k1, k1 = K'(0)/K.

1. Switch point.  Φ(p, q, b, A) = (∂_p L0, ∂_q L0, ∂_b L0, G0) with G0 on its active pair (outer x = −1.2, inner
   x = 0.8; every other candidate strictly dominated over the box).  Krawczyk test on a box X around the float
   zero: K(X) ⊂ int X certifies a unique zero in X.  Quadrature objective (800 points), mpmath.iv.
2. A'(0) = −[J⁻¹ ∂_εΦ]_A (implicit-function theorem), with J = ∂_{(p,q,b,A)}Φ over X and
   ∂_εΦ = (mean[σ(z)(1−σ(z)) A r ∂z + (σ(z) − y) ∂(A r)], r(σ_O) − r(σ_I)); the linear system is enclosed by
   d̃ ± ‖Y(J d̃ + b)‖∞/(1 − ‖I − YJ‖∞), Y = mid(J)⁻¹.
3. k1.  The gap maximiser is the vertex where I(−0.8) = I(0.8) and O(−2.0) = O(−1.2) (both orientations, u >= 0,
   x-symmetry gives its mirror v -> −v).  Krawczyk in 2D; the vertex persists under ε, so
   K'(0) = ∂_ε(O − I) + ∇(O − I)·θ',  J_E θ' = −∂_ε E.  Strict local maximum: 0 lies in the interior of the convex
   hull of the four piece-gradient differences.  Global: the branch and bound for K (limit_windows.K_certified)
   keeps cells only near ±vertex.

    python -m src.first_order                 # certified c1 (theory)
    python -m src.first_order finite [workers]   # registered test: certified Ĝ(a) and R_glob(a), a = 1.01-1.04
    python -m src.first_order score           # registered scoring (first_order_prediction.md)
"""

from __future__ import annotations

import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from mpmath import iv

RESULTS = Path(__file__).resolve().parents[1] / "results"
iv.dps = 30
mp.mp.dps = 30
R2 = math.sqrt(2)


def h(s): return -s + s ** 3 / 6
def h1(s): return -1 + s ** 2 / 2
def h2(s): return s
def r(s): return s ** 3 / 6 - s ** 5 / 120
def r1(s): return s ** 2 / 2 - s ** 4 / 24


def _sig(z):
    f = lambda t: 1 / (1 + mp.exp(-t))
    return iv.mpf([f(z.a), f(z.b)])


def _dsig(S):
    lo, hi = S.a, S.b
    vals = [lo * (1 - lo), hi * (1 - hi)]
    return iv.mpf([min(vals), mp.mpf("0.25") if lo <= 0.5 <= hi else max(vals)])


def _mid(v): return float((mp.mpf(v.a) + mp.mpf(v.b)) / 2)
def _mag(v): return float(max(abs(mp.mpf(v.a)), abs(mp.mpf(v.b))))
def _lo(v): return float(mp.mpf(v.a))
def _hi(v): return float(mp.mpf(v.b))


def _data():
    from .limit_bnb import population
    return population()


def switch_system(X, x, y, want_J=True):
    """Φ and (optionally) its Jacobian and ∂_εΦ over the interval box X = (p, q, b, A)."""
    p, q, b, A = X
    n = len(x)
    F = [iv.mpf(0)] * 3
    J = [[iv.mpf(0)] * 4 for _ in range(3)]
    E = [iv.mpf(0)] * 3
    for xi, yi in zip(x, y):
        xi = float(xi)
        s = p * xi + q
        hs, h1s, h2s = h(s), h1(s), h2(s)
        z = A * hs + b
        S = _sig(z); res = S - int(yi)
        dz = [A * h1s * xi, A * h1s, iv.mpf(1)]
        for i in range(3):
            F[i] += res * dz[i]
        if not want_J:
            continue
        w = _dsig(S)
        d2 = [[A * h2s * xi * xi, A * h2s * xi, 0], [A * h2s * xi, A * h2s, 0], [0, 0, 0]]
        dA = [h1s * xi, h1s, iv.mpf(0)]
        rs, r1s = r(s), r1(s)
        de = [A * r1s * xi, A * r1s, iv.mpf(0)]
        for i in range(3):
            for j in range(3):
                J[i][j] += w * dz[i] * dz[j] + res * d2[i][j]
            J[i][3] += w * hs * dz[i] + res * dA[i]
            E[i] += w * A * rs * dz[i] + res * de[i]
    F = [f / n for f in F]
    sO, sI = q - 1.2 * p, q + 0.8 * p
    F.append(h(sO) - h(sI))
    if not want_J:
        return F
    J = [[J[i][j] / n for j in range(4)] for i in range(3)]
    J.append([-1.2 * h1(sO) - 0.8 * h1(sI), h1(sO) - h1(sI), iv.mpf(0), iv.mpf(0)])
    E = [e / n for e in E] + [r(sO) - r(sI)]
    return F, J, E


def _pt(v):
    return [iv.mpf(mp.mpf(t)) for t in v]


def krawczyk(fun, xt, rad):
    """fun(Xbox) -> (F, J) intervals; xt float point; box xt ± rad.  Returns (ok, K box, Y, J)."""
    k = len(xt)
    X = [iv.mpf([mp.mpf(xt[i]) - rad[i], mp.mpf(xt[i]) + rad[i]]) for i in range(k)]
    F0 = fun(_pt(xt), False)
    _, J = fun(X, True)[:2]
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
    return ok, Kb, Y, J


def lin_enclose(J, b):
    """Enclose {d : J d = −b} over interval J, b: d̃ ± e (∞-norm)."""
    k = len(b)
    Jm = np.array([[_mid(J[i][j]) for j in range(k)] for i in range(k)])
    bm = np.array([_mid(v) for v in b])
    Y = np.linalg.inv(Jm)
    dt = -Y @ bm
    Rv = [sum(J[i][j] * float(dt[j]) for j in range(k)) + b[i] for i in range(k)]
    YR = [sum(float(Y[i, l]) * Rv[l] for l in range(k)) for i in range(k)]
    E = [[(1.0 if i == j else 0.0) - sum(float(Y[i, l]) * J[l][j] for l in range(k)) for j in range(k)]
         for i in range(k)]
    nE = max(sum(_mag(E[i][j]) for j in range(k)) for i in range(k))
    assert nE < 1, nE
    e = max(_mag(v) for v in YR) / (1 - nE)
    return dt, e


def _newton_float(fun, v, iters=40):
    for _ in range(iters):
        F, J = fun(_pt(v), True)[:2]
        Fm = np.array([_mid(f) for f in F]); Jm = np.array([[_mid(t) for t in row] for row in J])
        v = v - np.linalg.solve(Jm, Fm)
    return v


def switch_part():
    from .limit_bnb import profile
    x, y = _data()
    ls = pd.read_csv(RESULTS / "limit_switch.csv").iloc[0]
    p0, q0, A0 = float(ls.argmin_p), float(ls.argmin_q), float(ls.A_star)
    _, b2, *_ = profile([p0], [q0], A0, x, y)
    fun = lambda X, J=True: switch_system(X, x, y, J)[:2] if J else switch_system(X, x, y, False)
    v = _newton_float(fun, np.array([p0, q0, float(b2[0]), A0]))
    rad = [1e-9] * 4
    ok, Kb, _, _ = krawczyk(fun, v, rad)
    X = [iv.mpf([mp.mpf(v[i]) - rad[i], mp.mpf(v[i]) + rad[i]]) for i in range(4)]
    # active set over X: inner max at x = 0.8, outer min at x = -1.2, all other candidates strictly dominated
    p, q = X[0], X[1]
    I_act, O_act = h(q + 0.8 * p), h(q - 1.2 * p)
    others_I = [h(q - 0.8 * p)]
    others_O = [h(q - 2.0 * p), h(q + 1.2 * p), h(q + 2.0 * p)]
    sI = [q - 0.8 * p, q + 0.8 * p]; sOm = [q - 2.0 * p, q - 1.2 * p]; sOp = [q + 1.2 * p, q + 2.0 * p]
    crit_ok = (sI[0].a > -R2) and (sOm[1].b < R2) and (sOp[0].a > R2)   # no interior local max in I, min in O
    active_ok = all(I_act.a > o.b for o in others_I) and all(O_act.b < o.a for o in others_O) and crit_ok
    _, J, Eps = switch_system(X, x, y, True)
    dt, e = lin_enclose(J, Eps)
    A_iv = X[3]
    A1 = iv.mpf([dt[3] - e, dt[3] + e])
    return {"switch_krawczyk_ok": ok, "active_set_unique": active_ok,
            "p_star": v[0], "q_star": v[1], "b_star": v[2], "A_star_lo": _lo(Kb[3]), "A_star_hi": _hi(Kb[3]),
            "dA_deps_lo": _lo(A1), "dA_deps_hi": _hi(A1),
            "A1_over_A_lo": _lo(A1 / A_iv), "A1_over_A_hi": _hi(A1 / A_iv),
            "dG_margin_inner": _lo(I_act) - max(_hi(o) for o in others_I),
            "dG_margin_outer": min(_lo(o) for o in others_O) - _hi(O_act)}, A1 / A_iv


def vertex_system(X, want_J=True):
    u, v = X
    s = {k: v + k * u for k in (-0.8, 0.8, -2.0, -1.2)}
    F = [h(s[-0.8]) - h(s[0.8]), h(s[-2.0]) - h(s[-1.2])]
    if not want_J:
        return F
    J = [[-0.8 * h1(s[-0.8]) - 0.8 * h1(s[0.8]), h1(s[-0.8]) - h1(s[0.8])],
         [-2.0 * h1(s[-2.0]) + 1.2 * h1(s[-1.2]), h1(s[-2.0]) - h1(s[-1.2])]]
    return F, J, s


def k1_part():
    from .limit_windows import K_certified
    fun = lambda X, J=True: vertex_system(X, True)[:2] if J else vertex_system(X, False)
    t = _newton_float(fun, np.array([1.605575, 1.204182]))
    rad = [1e-12, 1e-12]
    ok, Kb, _, _ = krawczyk(fun, t, rad)
    X = [iv.mpf([mp.mpf(t[i]) - rad[i], mp.mpf(t[i]) + rad[i]]) for i in range(2)]
    _, J, s = vertex_system(X, True)
    K = h(s[-1.2]) - h(s[0.8])
    Eps = [r(s[-0.8]) - r(s[0.8]), r(s[-2.0]) - r(s[-1.2])]
    dt, e = lin_enclose(J, Eps)
    th = [iv.mpf([dt[i] - e, dt[i] + e]) for i in range(2)]
    grad = [-1.2 * h1(s[-1.2]) - 0.8 * h1(s[0.8]), h1(s[-1.2]) - h1(s[0.8])]
    Kp = r(s[-1.2]) - r(s[0.8]) + grad[0] * th[0] + grad[1] * th[1]
    k1 = Kp / K
    # other candidates dominated at the vertex
    u, v = X
    crit_ok = ((v - 0.8 * u).a > -R2) and ((v - 1.2 * u).b < R2) and ((v + 1.2 * u).a > R2)
    dom_ok = crit_ok and (h(v + 1.2 * u).a > h(s[-1.2]).b)
    # strict local max: 0 in the interior of conv{∇O_i − ∇I_j}
    g = {k: np.array([k * _mid(h1(s[k])), _mid(h1(s[k]))]) for k in s}
    cvec = [g[o] - g[i] for o in (-2.0, -1.2) for i in (-0.8, 0.8)]
    ang = sorted(math.atan2(c[1], c[0]) for c in cvec)
    gaps = [ang[i + 1] - ang[i] for i in range(3)] + [2 * math.pi - (ang[-1] - ang[0])]
    strict = max(gaps) < math.pi - 1e-6
    Klo, Khi, arg, _ = K_certified((-0.8, 0.8), (1.2, 2.0), rel=1e-7)
    return {"vertex_krawczyk_ok": ok, "vertex_u": t[0], "vertex_v": t[1], "K_vertex_lo": _lo(K), "K_vertex_hi": _hi(K),
            "K_bnb_lo": Klo, "K_bnb_hi": Khi, "vertex_in_bnb_enclosure": (Klo <= _hi(K)) and (_lo(K) <= Khi),
            "bnb_argmax_dist_to_vertex": min(math.hypot(arg[0] - t[0], arg[1] - s_) for s_ in (t[1], -t[1])),
            "other_pieces_dominated": dom_ok, "strict_local_max": strict, "max_angular_gap": max(gaps),
            "k1_lo": _lo(k1), "k1_hi": _hi(k1)}, k1


def main():
    sw, a1 = switch_part()
    kk, k1 = k1_part()
    c1 = a1 + k1
    out = {**sw, **kk, "c1_lo": _lo(c1), "c1_hi": _hi(c1)}
    # sharp R_glob^inf: the K enclosure of the branch and bound (rel 1e-7) times the Krawczyk A*.  Conditional on
    # math_note_v2 (c): the global minimiser is the U-branch for A in [0.66, 0.71] (annulus).
    out["R_glob_inf_lo"] = out["K_bnb_lo"] * out["A_star_lo"] / 2
    out["R_glob_inf_hi"] = out["K_bnb_hi"] * out["A_star_hi"] / 2
    pd.DataFrame([out]).to_csv(RESULTS / "first_order_c1.csv", index=False)
    print(pd.Series(out).to_string())


# ------------------------------------------------------------------ registered finite-a test
A_TEST = (1.01, 1.02, 1.03, 1.04)
REL_WIDTH = 2e-4
GHAT_REL = 1e-6
C3_MAX = 1.0


def _finite_job(a):
    import sys as _s
    from .conditional_certified import _population, evaluate
    from .ghat_bnb import certify as ghat_certify
    eps = a - 1.0
    x, y = _population()
    g = ghat_certify(a, target_rel=GHAT_REL)
    A_star = float(pd.read_csv(RESULTS / "first_order_c1.csv").A_star_lo.iloc[0])
    rows = []

    def status(sv):
        r_ = evaluate(sv, a, x, y, True)
        rows.append(r_)
        print(a, sv, r_["status"], flush=True)
        return r_["status"]
    lo = hi = round(A_star / eps ** 1.5, 6)
    st = status(lo)
    if st == "plus":
        while st == "plus":
            hi = lo; lo = round(lo * 0.99, 6); st = status(lo)
    else:
        while st != "plus":
            if st == "unresolved":
                break
            lo = hi; hi = round(hi * 1.01, 6); st = status(hi)
    unresolved = st == "unresolved"
    while not unresolved and (hi - lo) / lo > REL_WIDTH:
        mid = round(0.5 * (lo + hi), 9)
        st = status(mid)
        if st == "plus":
            hi = mid
        elif st == "minus":
            lo = mid
        else:
            unresolved = True
    G_lo, G_hi = float(g["Ghat_lo"]), float(g["Ghat_hi"])
    return ({"a": a, "eps": eps, "s_lo": lo, "s_hi": hi, "s_rel_width": (hi - lo) / lo,
             "stopped_unresolved": unresolved, "Ghat_lo": G_lo, "Ghat_hi": G_hi, "Ghat_rel_width": (G_hi - G_lo) / G_lo,
             "R_lo": lo * G_lo / 2, "R_hi": hi * G_hi / 2,
             "A_lo": lo * eps ** 1.5, "A_hi": hi * eps ** 1.5,
             "K_lo": G_lo / eps ** 1.5, "K_hi": G_hi / eps ** 1.5,
             "all_converged": all(r_["converged"] for r_ in rows), "evaluations": len(rows)}, rows)


def finite(workers=2):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        out = pool.map(_finite_job, A_TEST, chunksize=1)
    t = pd.DataFrame([r_ for r_, _ in out])
    t.to_csv(RESULTS / "first_order_finite.csv", index=False)
    pd.DataFrame([e for _, ev in out for e in ev]).to_csv(RESULTS / "first_order_finite_evaluations.csv", index=False)
    print(t.to_string(index=False))


def _feasible_c1(eps, lo, hi, base_lo, base_hi, c3=C3_MAX, grid=11):
    """Range of c1 over all two-term laws base·(1 + c1 ε + c2 ε²) (base in its interval, c2 free) passing through
    every interval [lo_i, hi_i] widened by ±c3·ε_i³·base.  Exact 2D LP by vertex enumeration per base value."""
    c1s = []
    for B in np.linspace(base_lo, base_hi, grid):
        cons = []                                   # a1*c1 + a2*c2 <= b
        for e_, l_, h_ in zip(eps, lo, hi):
            w = c3 * e_ ** 3
            cons.append((e_, e_ ** 2, h_ / B - 1 + w))
            cons.append((-e_, -e_ ** 2, -(l_ / B - 1 - w)))
        for i in range(len(cons)):
            for j in range(i + 1, len(cons)):
                M = np.array([cons[i][:2], cons[j][:2]])
                if abs(np.linalg.det(M)) < 1e-18:
                    continue
                v = np.linalg.solve(M, np.array([cons[i][2], cons[j][2]]))
                if all(a1 * v[0] + a2 * v[1] <= b + 1e-12 for a1, a2, b in cons):
                    c1s.append(v[0])
    return (min(c1s), max(c1s)) if c1s else (np.nan, np.nan)


def score():
    t = pd.read_csv(RESULTS / "first_order_finite.csv").sort_values("a")
    c = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    eps = t.eps.values
    rows = []
    for name, lo, hi, blo, bhi, pred in (
            ("c1 (primary): R_glob", t.R_lo.values, t.R_hi.values, c.R_glob_inf_lo, c.R_glob_inf_hi, (c.c1_lo, c.c1_hi)),
            ("k1 (secondary): Ghat/eps^1.5", t.K_lo.values, t.K_hi.values, c.K_bnb_lo, c.K_bnb_hi, (c.k1_lo, c.k1_hi)),
            ("A'/A* (secondary): s eps^1.5", t.A_lo.values, t.A_hi.values, c.A_star_lo, c.A_star_hi,
             (c.A1_over_A_lo, c.A1_over_A_hi))):
        f_lo, f_hi = _feasible_c1(eps, lo, hi, blo, bhi)
        empty = not np.isfinite(f_lo)
        width = f_hi - f_lo if not empty else np.nan
        consistent = (not empty) and (pred[1] >= f_lo) and (pred[0] <= f_hi)
        verdict = ("FAIL (no two-term law fits)" if empty else
                   "INCONCLUSIVE (feasible width > 0.1)" if width > 0.1 else
                   "PASS" if consistent else "FAIL")
        rows.append({"quantity": name, "pred_lo": pred[0], "pred_hi": pred[1], "feasible_lo": f_lo,
                     "feasible_hi": f_hi, "feasible_width": width, "verdict": verdict,
                     "excludes_0.49": (not empty) and not (f_lo <= 0.49 <= f_hi),
                     "excludes_k1_only_-0.377": (not empty) and not (f_lo <= -0.377 <= f_hi),
                     "excludes_switch_only_0.662": (not empty) and not (f_lo <= 0.662 <= f_hi),
                     "excludes_0": (not empty) and not (f_lo <= 0 <= f_hi)})
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "first_order_scores.csv", index=False)
    print(t.to_string(index=False)); print(d.to_string(index=False))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "c1"
    if cmd == "finite":
        finite(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif cmd == "score":
        score()
    else:
        main()
