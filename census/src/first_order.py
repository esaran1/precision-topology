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
    python -m src.first_order corner          # the tied corner: active set and strict max certified for |ε| <= 0.05,
                                              # plus a free (active-set-agnostic) check at ε = 0.005, 1e-3, 1e-4
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


# ------------------------------------------------------------------ the tied corner, uniformly in ε
E_MAX = 0.05
E_PIECES = 400
N_SERIES = 14


def _tail(e1, S, deriv):
    """Bound on (1+e1)·Σ_{k>N} e1^{k-1} S^{2k+1-deriv}/(2k+1-deriv)!."""
    t = mp.mpf(0)
    for k in range(N_SERIES + 1, N_SERIES + 60):
        t += mp.mpf(e1) ** (k - 1) * mp.mpf(S) ** (2 * k + 1 - deriv) / mp.factorial(2 * k + 1 - deriv)
    return (1 + mp.mpf(e1)) * t


def phi_iv(sg, E, e1, S):
    """φ_ε(σ) = −σ + (1+ε) Σ_{k>=1} (−1)^{k+1} ε^{k-1} σ^{2k+1}/(2k+1)!  (entire in ε; any sign of ε)."""
    acc = iv.mpf(0)
    Ek = iv.mpf(1)
    for k in range(1, N_SERIES + 1):
        acc += (-1) ** (k + 1) * Ek * sg ** (2 * k + 1) / mp.factorial(2 * k + 1)
        Ek = Ek * E
    T = _tail(e1, S, 0)
    return -sg + (1 + E) * acc + iv.mpf([-T, T])


def dphi_iv(sg, E, e1, S):
    acc = iv.mpf(0)
    Ek = iv.mpf(1)
    for k in range(1, N_SERIES + 1):
        acc += (-1) ** (k + 1) * Ek * sg ** (2 * k) / mp.factorial(2 * k)
        Ek = Ek * E
    T = _tail(e1, S, 1)
    return -1 + (1 + E) * acc + iv.mpf([-T, T])


def _phi_f(sg, e):
    """Float φ_ε from the same series (stable for small |ε|, any sign)."""
    acc, Ek = 0.0, 1.0
    for k in range(1, 30):
        acc += (-1) ** (k + 1) * Ek * sg ** (2 * k + 1) / math.factorial(2 * k + 1)
        Ek *= e
    return -sg + (1 + e) * acc


def _dphi_f(sg, e):
    acc, Ek = 0.0, 1.0
    for k in range(1, 30):
        acc += (-1) ** (k + 1) * Ek * sg ** (2 * k) / math.factorial(2 * k)
        Ek *= e
    return -1 + (1 + e) * acc


def _vertex_float(e, t0):
    t = np.array(t0, float)
    for _ in range(50):
        u, v = t
        sg = {k: v + k * u for k in (-0.8, 0.8, -2.0, -1.2)}
        F = np.array([_phi_f(sg[-0.8], e) - _phi_f(sg[0.8], e), _phi_f(sg[-2.0], e) - _phi_f(sg[-1.2], e)])
        d = {k: _dphi_f(sg[k], e) for k in sg}
        J = np.array([[-0.8 * d[-0.8] - 0.8 * d[0.8], d[-0.8] - d[0.8]],
                      [-2.0 * d[-2.0] + 1.2 * d[-1.2], d[-2.0] - d[-1.2]]])
        t = t - np.linalg.solve(J, F)
    return t


def corner_piece(E_lo, E_hi, t0, rad=1e-3):
    E = iv.mpf([E_lo, E_hi])
    e1 = max(abs(E_lo), abs(E_hi))
    em = 0.5 * (E_lo + E_hi)
    tt = _vertex_float(em, t0)
    S = 2.0 * (abs(tt[0]) + rad) + abs(tt[1]) + rad
    P = lambda sg: phi_iv(sg, E, e1, S)
    D = lambda sg: dphi_iv(sg, E, e1, S)

    def fun(X, want_J=True):
        u, v = X
        sg = {k: v + k * u for k in (-0.8, 0.8, -2.0, -1.2)}
        F = [P(sg[-0.8]) - P(sg[0.8]), P(sg[-2.0]) - P(sg[-1.2])]
        if not want_J:
            return F
        d = {k: D(sg[k]) for k in sg}
        J = [[-0.8 * d[-0.8] - 0.8 * d[0.8], d[-0.8] - d[0.8]],
             [-2.0 * d[-2.0] + 1.2 * d[-1.2], d[-2.0] - d[-1.2]]]
        return F, J
    ok, Kb, _, _ = krawczyk(fun, tt, [rad, rad])
    X = Kb                      # contracted box: contains the unique zero for every ε in E (if ok)
    u, v = X
    sI_lo, sI_hi = v - 0.8 * u, v + 0.8 * u
    sOm_lo, sOm_hi = v - 2.0 * u, v - 1.2 * u
    sOp_lo, sOp_hi = v + 1.2 * u, v + 2.0 * u
    # concave for σ < 0, convex for σ > 0: φ'' = (1+ε) sin(√ε σ)/√ε (ε > 0; sinh form for ε < 0), and √ε|σ| <= √e1·S < π
    shape_ok = math.sqrt(e1) * S < math.pi
    neg_part = iv.mpf([min(float(mp.mpf(sI_lo.a)), 0.0), 0.0])
    inner_decreasing_neg = (sI_lo.a >= 0) or (D(neg_part).b < 0)
    om_negative = sOm_hi.b < 0
    op_increasing = (sOp_lo.a > 0) and (D(iv.mpf([sOp_lo.a, sOp_hi.b])).a > 0)
    op_dominated = P(sOp_lo).a > P(sOm_hi).b
    other_orient_neg = (P(sI_lo) - P(sOp_hi)).b < 0 or (P(sI_hi) - P(sOp_hi)).b < 0
    # strict (sharp) max: positive barycentric weights for three of the four c_ij = ∇O_i − ∇I_j
    sg = {k: v + k * u for k in (-0.8, 0.8, -2.0, -1.2)}
    g = {k: (k * D(sg[k]), D(sg[k])) for k in sg}
    c = {(o, i): (g[o][0] - g[i][0], g[o][1] - g[i][1]) for o in (-2.0, -1.2) for i in (-0.8, 0.8)}
    keys = list(c)
    strict, lam_min = False, -np.inf
    for drop in keys:
        tri = [k for k in keys if k != drop]
        M = [[c[tri[0]][0], c[tri[1]][0], c[tri[2]][0]], [c[tri[0]][1], c[tri[1]][1], c[tri[2]][1]],
             [iv.mpf(1), iv.mpf(1), iv.mpf(1)]]
        try:
            lt, le = lin_enclose(M, [iv.mpf(0), iv.mpf(0), iv.mpf(-1)])
        except (AssertionError, np.linalg.LinAlgError):
            continue
        lm = float(min(lt)) - le
        if lm > lam_min:
            lam_min = lm
        if lm > 0:
            strict = True
    return {"eps_lo": E_lo, "eps_hi": E_hi, "krawczyk_ok": ok, "u_mid": tt[0], "v_mid": tt[1],
            "box_width_u": _hi(Kb[0]) - _lo(Kb[0]), "box_width_v": _hi(Kb[1]) - _lo(Kb[1]),
            "shape_ok": shape_ok, "inner_decreasing_on_negative_part": inner_decreasing_neg,
            "outer_minus_window_negative_sigma": om_negative, "outer_plus_increasing": op_increasing,
            "outer_plus_dominated": op_dominated, "other_orientation_negative": other_orient_neg,
            "strict_max_barycentric_min": lam_min, "strict_max": strict,
            "all_ok": all([ok, shape_ok, inner_decreasing_neg, om_negative, op_increasing, op_dominated,
                           other_orient_neg, strict])}


def _phi_exact(sg, a, eps):
    se = math.sqrt(eps) * sg
    return (se - a * np.sin(se)) / eps ** 1.5


def _phi_extrema(lo, hi, a, eps):
    """Exact min/max of φ_ε over [lo, hi] (vectorised): endpoints and critical points s = ±arccos(1/a) + 2πk."""
    vals = [_phi_exact(lo, a, eps), _phi_exact(hi, a, eps)]
    mn, mx = np.minimum(*vals), np.maximum(*vals)
    base = math.acos(1 / a)
    for k in (-1, 0, 1):
        for sgn in (-1, 1):
            c = (sgn * base + 2 * math.pi * k) / math.sqrt(eps)
            inside = (lo <= c) & (c <= hi)
            val = _phi_exact(c, a, eps)
            mn = np.where(inside, np.minimum(mn, val), mn)
            mx = np.where(inside, np.maximum(mx, val), mx)
    return mn, mx


def K_free(eps, U=8.0, V=12.0, h0=0.05, rel=1e-7):
    """Branch and bound for sup G[φ_ε∘σ] over (u, v) ∈ [0, U] × [−V, V], both orientations, exact extrema; no
    active set assumed.  Step: |φ_ε'| <= 1 + aσ²/2."""
    a = 1 + eps
    nu, nv = int(U / h0), int(2 * V / h0)
    hu, hv = U / nu / 2, V / nv
    cu, cv = np.meshgrid((np.arange(nu) + 0.5) * 2 * hu, -V + (np.arange(nv) + 0.5) * 2 * hv, indexing="ij")
    cu, cv = cu.ravel(), cv.ravel()
    best, arg = -np.inf, None
    for _ in range(60):
        def ext(lo, hi):
            a1, a2 = cu * lo + cv, cu * hi + cv
            return _phi_extrema(np.minimum(a1, a2), np.maximum(a1, a2), a, eps)
        imn, imx = ext(-0.8, 0.8)
        p_mn, p_mx = ext(1.2, 2.0)
        n_mn, n_mx = ext(-2.0, -1.2)
        omn, omx = np.minimum(p_mn, n_mn), np.maximum(p_mx, n_mx)
        G = np.maximum(omn - imx, imn - omx)
        smax = 2.0 * (np.abs(cu) + hu) + np.abs(cv) + hv
        ub = G + (1 + a * smax ** 2 / 2) * (2.8 * hu + 2 * hv)
        j = int(np.argmax(G))
        if G[j] > best:
            best, arg = float(G[j]), (float(cu[j]), float(cv[j]))
        keep = ub > best
        hi = float(ub[keep].max()) if keep.any() else best
        if hi - best <= rel * best:
            spread = float(np.max(np.minimum(np.hypot(cu[keep] - arg[0], cv[keep] - arg[1]),
                                              np.hypot(cu[keep] - arg[0], cv[keep] + arg[1]))))
            return best, hi, arg, spread
        cu, cv = cu[keep], cv[keep]
        hu, hv = hu / 2, hv / 2
        cu = np.concatenate([cu - hu, cu - hu, cu + hu, cu + hu])
        cv = np.concatenate([cv - hv, cv + hv, cv - hv, cv + hv])
    raise RuntimeError("K_free not converged")


def corner():
    c = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    t0 = (float(c.vertex_u), float(c.vertex_v))
    edges = np.linspace(-E_MAX, E_MAX, E_PIECES + 1)
    rows = [corner_piece(float(edges[i]), float(edges[i + 1]), t0) for i in range(E_PIECES)]
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "first_order_corner.csv", index=False)
    print(d[["eps_lo", "eps_hi", "krawczyk_ok", "box_width_u", "strict_max_barycentric_min", "all_ok"]]
          .iloc[::40].to_string(index=False))
    print("pieces:", len(d), "all ok:", bool(d.all_ok.all()), "min barycentric weight:", d.strict_max_barycentric_min.min())
    K0 = 0.5 * (float(c.K_vertex_lo) + float(c.K_vertex_hi))
    free = []
    for e in (0.005, 1e-3, 1e-4):
        lo, hi, arg, spread = K_free(e)
        tt = _vertex_float(e, t0)
        u, v = tt
        K_fixed = _phi_exact(v - 1.2 * u, 1 + e, e) - _phi_exact(v + 0.8 * u, 1 + e, e)
        free.append({"eps": e, "K_free_lo": lo, "K_free_hi": hi, "free_argmax_u": arg[0], "free_argmax_v": arg[1],
                     "kept_cells_max_dist_to_argmax_or_mirror": spread,
                     "fixed_active_vertex_u": u, "fixed_active_vertex_v": v, "K_fixed_active": K_fixed,
                     "fixed_in_free_enclosure": lo - 1e-12 <= K_fixed <= hi + 1e-12,
                     "slope_free_lo": (lo / K0 - 1) / e, "slope_free_hi": (hi / K0 - 1) / e,
                     "slope_fixed": (K_fixed / K0 - 1) / e})
    f = pd.DataFrame(free)
    f.to_csv(RESULTS / "first_order_corner_free.csv", index=False)
    print(f.to_string(index=False))


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
             "Ghat_converged": bool(g["converged"]),
             "Ghat_argmax_u": g["w1"] / math.sqrt(eps), "Ghat_argmax_v": (g["b1"] - math.pi) / math.sqrt(eps),
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


# ------------------------------------------------------------------ SUPPLEMENTARY: switch located along the branch
def _fa(t, a): return t + a * iv.sin(t) if isinstance(t, type(iv.mpf(0))) else t + a * math.sin(t)
def _fa1(t, a): return 1 + a * iv.cos(t) if isinstance(t, type(iv.mpf(0))) else 1 + a * math.cos(t)
def _fa2(t, a): return -a * iv.sin(t) if isinstance(t, type(iv.mpf(0))) else -a * math.sin(t)


def branch_system(X, a, x, y, want_J=True):
    """Φ(w1, b1, b2, s) = (∇_{w1,b1,b2} L, G) on the quadrature objective at finite a, G on the active pair
    (outer x = −1.2, inner x = 0.8; w2 > 0 orientation).  Returns Φ, J = ∂Φ/∂(w1, b1, b2, s)."""
    w1, b1, b2, s = X
    n = len(x)
    F = [iv.mpf(0)] * 3
    J = [[iv.mpf(0)] * 4 for _ in range(3)]
    for xi, yi in zip(x, y):
        xi = float(xi)
        t = w1 * xi + b1
        f0, f1, f2 = _fa(t, a), _fa1(t, a), _fa2(t, a)
        z = s * f0 + b2
        S = _sig(z); res = S - int(yi)
        dz = [s * f1 * xi, s * f1, iv.mpf(1)]
        for i in range(3):
            F[i] += res * dz[i]
        if not want_J:
            continue
        w = _dsig(S)
        d2 = [[s * f2 * xi * xi, s * f2 * xi, 0], [s * f2 * xi, s * f2, 0], [0, 0, 0]]
        ds = [f1 * xi, f1, iv.mpf(0)]
        for i in range(3):
            for j in range(3):
                J[i][j] += w * dz[i] * dz[j] + res * d2[i][j]
            J[i][3] += w * f0 * dz[i] + res * ds[i]
    F = [f / n for f in F]
    tO, tI = w1 * -1.2 + b1, w1 * 0.8 + b1
    F.append(_fa(tO, a) - _fa(tI, a))
    if not want_J:
        return F
    J = [[J[i][j] / n for j in range(4)] for i in range(3)]
    J.append([-1.2 * _fa1(tO, a) - 0.8 * _fa1(tI, a), _fa1(tO, a) - _fa1(tI, a), iv.mpf(0), iv.mpf(0)])
    return F, J


def _active_ok(W1, B1, a):
    """Over the box: max over I of f_a is at x = 0.8 and min over O at x = −1.2, every other candidate (edges and
    interior critical points t = π ± arccos(1/a) + 2πk) strictly dominated."""
    d = math.acos(1 / a)
    inner = [(-0.8, 0.8)]; outer = [(-2.0, -1.2), (1.2, 2.0)]
    I_act, O_act = _fa(W1 * 0.8 + B1, a), _fa(W1 * -1.2 + B1, a)
    others_I, others_O = [_fa(W1 * -0.8 + B1, a)], [_fa(W1 * -2.0 + B1, a), _fa(W1 * 1.2 + B1, a), _fa(W1 * 2.0 + B1, a)]
    for (lo, hi), acc, base in ((inner[0], others_I, math.pi - d), (outer[0], others_O, math.pi + d),
                                (outer[1], others_O, math.pi + d)):
        t1, t2 = W1 * lo + B1, W1 * hi + B1
        tlo, thi = float(mp.mpf(min(t1.a, t2.a))), float(mp.mpf(max(t1.b, t2.b)))
        for k in range(math.floor((tlo - base) / (2 * math.pi)) - 1, math.ceil((thi - base) / (2 * math.pi)) + 2):
            c = base + 2 * math.pi * k
            if tlo - 1e-9 <= c <= thi + 1e-9:
                acc.append(iv.mpf(_fa(c, a)))
    return all(I_act.a > o.b for o in others_I) and all(O_act.b < o.a for o in others_O)


def supplementary_one(a, rad=(1e-9, 1e-9, 1e-7, 1e-6)):
    from .conditional_certified import _population
    from .own_threshold import global_min
    from .profiled_bnb import competitor_gap, profile
    eps = a - 1
    x, y = _population()
    c = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    s0 = float(c.A_star_lo) * (1 + float(c.A1_over_A_lo) * eps) / eps ** 1.5     # starting guess only
    _, w1, b1, _, _ = global_min(s0, a, x, y)
    if w1 < 0:
        w1 = -w1                                     # population is x-symmetric: use the w1 > 0 member of the pair
    _, b2, *_ = profile([w1], [b1], s0, a, x, y)
    fun = lambda X, J=True: branch_system(X, a, x, y, J) if J else branch_system(X, a, x, y, False)
    v = _newton_float(fun, np.array([w1, b1, float(b2[0]), s0]))
    ok, Kb, _, _ = krawczyk(fun, v, list(rad))
    X = [iv.mpf([mp.mpf(v[i]) - rad[i], mp.mpf(v[i]) + rad[i]]) for i in range(4)]
    _, J = branch_system(X, a, x, y, True)
    Hm = np.array([[_mid(J[i][j]) for j in range(3)] for i in range(3)])
    Hr = math.sqrt(sum(float(mp.mpf((J[i][j].b - J[i][j].a) / 2)) ** 2 for i in range(3) for j in range(3)))
    lam_lo = float(np.linalg.eigvalsh(Hm)[0]) - Hr
    act = _active_ok(X[0], X[1], a)
    s_star = float(v[3])
    L_branch = float(profile([v[0]], [v[1]], s_star, a, x, y)[0][0])
    rad_ball = 0.25 * math.sqrt(eps)
    cg = competitor_gap(s_star, a, x, y, [(v[0], v[1]), (-v[0], v[1])], rad_ball, tol=1e-8)
    return {"a": a, "eps": eps, "krawczyk_ok": ok, "s_lo": _lo(Kb[3]), "s_hi": _hi(Kb[3]), "w1": v[0], "b1": v[1],
            "hess_lambda_min_lo": lam_lo, "active_set_unique": act, "competitor_ball_radius": rad_ball,
            "competitor_lower": cg["lower"], "branch_loss": L_branch, "competitor_margin": cg["lower"] - L_branch,
            "competitor_converged": cg["converged"],
            "certified": bool(ok and lam_lo > 0 and act and cg["lower"] > L_branch)}


def _supp_job(a):
    return supplementary_one(a)


def supplementary(workers=2):
    """SUPPLEMENTARY to first_order_prediction.md (written before the registered result was scored): the switch located
    along the branch minimiser by a Krawczyk test on (∇L, G) = 0 in (w1, b1, b2, s), same objective and branch."""
    from multiprocessing import Pool
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_supp_job, A_TEST, chunksize=1))
    d.to_csv(RESULTS / "first_order_supplementary.csv", index=False)
    print(d.to_string(index=False))


def score_supplementary():
    """The c1 feasible set implied by the supplementary brackets, with the registered construction (Ĝ from the
    registered run; the same limit interval and O(ε³) allowance)."""
    sp = pd.read_csv(RESULTS / "first_order_supplementary.csv", float_precision="round_trip").sort_values("a")
    fin = pd.read_csv(RESULTS / "first_order_finite.csv", float_precision="round_trip").set_index("a")
    c = pd.read_csv(RESULTS / "first_order_c1.csv", float_precision="round_trip").iloc[0]
    G_lo = sp.a.map(fin.Ghat_lo).values; G_hi = sp.a.map(fin.Ghat_hi).values
    R_lo, R_hi = sp.s_lo.values * G_lo / 2, sp.s_hi.values * G_hi / 2
    f_lo, f_hi = _feasible_c1(sp.eps.values, R_lo, R_hi, c.R_glob_inf_lo, c.R_glob_inf_hi)
    out = pd.DataFrame([{"all_certified": bool(sp.certified.all()), "feasible_lo": f_lo, "feasible_hi": f_hi,
                         "feasible_width": f_hi - f_lo, "pred_lo": c.c1_lo, "pred_hi": c.c1_hi,
                         "pred_inside": bool(f_lo <= c.c1_hi and c.c1_lo <= f_hi),
                         "max_rel_bracket_width": float(np.max((sp.s_hi - sp.s_lo) / sp.s_lo)),
                         **{f"competing_{name}": ("inside" if f_lo <= v <= f_hi else "below" if v < f_lo else "above")
                            for name, v in (("0.49_earlier", 0.49), ("-0.377_K_only", -0.377),
                                            ("0.662_switch_only", 0.662), ("0_no_first_order", 0.0))}}])
    pts = sp.assign(R_lo=R_lo, R_hi=R_hi)
    pts.to_csv(RESULTS / "first_order_supplementary.csv", index=False)
    out.to_csv(RESULTS / "first_order_supplementary_scores.csv", index=False)
    print(pts.to_string(index=False)); print(out.T.to_string())


# ------------------------------------------------------------------ SUPPLEMENTARY: Ĝ(a) certified via the rescaled box
def _gap_both(w1, b1, a):
    """Vectorised exact oriented gap, both orientations: max(min_O f − max_I f, min_I f − max_O f)."""
    from .profiled_bnb import _interval_extrema
    w1 = np.asarray(w1, float); b1 = np.asarray(b1, float)

    def ext(lo, hi):
        t1, t2 = w1 * lo + b1, w1 * hi + b1
        return _interval_extrema(np.minimum(t1, t2), np.maximum(t1, t2), a)
    imn, imx = ext(-0.8, 0.8)
    p_mn, p_mx = ext(1.2, 2.0)
    n_mn, n_mx = ext(-2.0, -1.2)
    return np.maximum(np.minimum(p_mn, n_mn) - imx, imn - np.maximum(p_mx, n_mx))


def ghat_rescaled(a, rel=1e-6, U=8.0, V=12.0, h0=0.05, max_cells=2_000_000, max_rounds=60, chunk=200_000):
    """SUPPLEMENTARY to the registered Ĝ (ghat_bnb.certify): the global supremum of G over placements, certified as
    (i) the branch-and-bound supremum over the rescaled box w1 = √ε u, b1 = π + √ε v, (u, v) ∈ [0, U] × [−V, V]
    (K_free: exact extrema of φ_ε, step |φ_ε'| <= 1 + aσ²/2), and (ii) an exclusion branch and bound over the rest of
    the placement domain (w1 ∈ (0, a], b1 ∈ [0, 2π); ghat_bnb's domain and step (1 + a)(2.8 h_w + 2 h_b)) proving
    every point outside the box has G below the box's attained value.  Ĝ ∈ ε^{3/2}[K_lo, K_hi] iff (ii) closes."""
    eps = a - 1
    K_lo, K_hi, arg, spread = K_free(eps, U=U, V=V, rel=rel)
    tau = K_lo * eps ** 1.5
    se = math.sqrt(eps)
    box_w, box_b = (0.0, U * se), (math.pi - V * se, math.pi + V * se)
    nw, nb = max(1, math.ceil(a / h0)), math.ceil(2 * math.pi / h0)
    hw, hb = a / nw / 2, 2 * math.pi / nb / 2
    cw, cb = np.meshgrid((np.arange(nw) + 0.5) * 2 * hw, (np.arange(nb) + 0.5) * 2 * hb, indexing="ij")
    cw, cb = cw.ravel(), cb.ravel()
    closed, rounds, peak = False, 0, len(cw)
    for rounds in range(1, max_rounds + 1):
        inside = (cw - hw >= box_w[0]) & (cw + hw <= box_w[1]) & (cb - hb >= box_b[0]) & (cb + hb <= box_b[1])
        cw, cb = cw[~inside], cb[~inside]
        if not len(cw):
            closed = True; break
        ub = np.empty(len(cw))
        for i in range(0, len(cw), chunk):
            ub[i:i + chunk] = _gap_both(cw[i:i + chunk], cb[i:i + chunk], a) + (1 + a) * (2.8 * hw + 2 * hb)
        keep = ub >= tau
        if not keep.any():
            closed = True; break
        cw, cb = cw[keep], cb[keep]
        if 4 * len(cw) > max_cells:
            break
        hw, hb = hw / 2, hb / 2
        cw = np.concatenate([cw - hw, cw - hw, cw + hw, cw + hw])
        cb = np.concatenate([cb - hb, cb + hb, cb - hb, cb + hb])
        peak = max(peak, len(cw))
    return {"a": a, "eps": eps, "K_lo": K_lo, "K_hi": K_hi, "box_argmax_u": arg[0], "box_argmax_v": arg[1],
            "box_kept_spread": spread, "exclusion_closed": closed, "exclusion_rounds": rounds, "exclusion_peak_cells": peak,
            "Ghat_lo": K_lo * eps ** 1.5 if closed else np.nan, "Ghat_hi": K_hi * eps ** 1.5 if closed else np.nan}

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "c1"
    if cmd == "finite":
        finite(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif cmd == "score":
        score()
    elif cmd == "corner":
        corner()
    elif cmd == "supplementary":
        supplementary(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif cmd == "score_supplementary":
        score_supplementary()
    elif cmd == "ghat_rescaled":
        d = pd.DataFrame([ghat_rescaled(a) for a in A_TEST])
        d.to_csv(RESULTS / "first_order_ghat_rescaled.csv", index=False)
        print(d.to_string(index=False))
    else:
        main()
