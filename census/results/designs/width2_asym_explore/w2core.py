"""EXPLORATORY helpers for the width-2 asymmetric-window design (not registered; nothing here is frozen).

Width 2, f_a(t) = t + a·sin t with a = 1.30; asymmetric windows (Δ = 0.4), set via asym_register._setup().
Parameters in width2_train's layout q = (α₁, β₁, v₁, α₂, β₂, v₂, b).  Fast coordinates z = (α₁, β₁, α₂, β₂, b); slow
coordinates v = (v₁, v₂), s = ‖v‖₁.  Loss: mean BCE with logits (as width2_train).  Analytic gradient and Hessian (numpy).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/Evan/precision-topology/census")
A = 1.30
ZI = [0, 1, 3, 4, 6]            # z inside q
VI = [2, 5]                     # v inside q
TWO_PI = 2 * math.pi


def nice():
    try:
        os.nice(15)
    except OSError:
        pass


def setup():
    """Asymmetric windows (Δ = 0.4) for the exact-extrema gap; returns the population (x, y)."""
    from src.asym_register import _setup
    x, y = _setup()
    return np.asarray(x, float), np.asarray(y, float)


def own_sample(seed):
    from src.asym_register import training_set
    return training_set(seed)


def act():
    from src.width2_geometry import Act
    return Act("fa", A)


def _f(t):
    return t + A * np.sin(t), 1 + A * np.cos(t), -A * np.sin(t)


def _sig(n):
    return 0.5 * (1 + np.tanh(0.5 * n))


def loss(q, x, y):
    t1, t2 = q[0] * x + q[1], q[3] * x + q[4]
    n = q[2] * (t1 + A * np.sin(t1)) + q[5] * (t2 + A * np.sin(t2)) + q[6]
    return float(np.mean(np.logaddexp(0, n) - y * n))


def jac(q, x):
    """∂N/∂q (len(x) × 7) and the pieces for the second derivatives."""
    t1, t2 = q[0] * x + q[1], q[3] * x + q[4]
    f1, d1, s1 = _f(t1); f2, d2, s2 = _f(t2)
    J = np.stack([q[2] * d1 * x, q[2] * d1, f1, q[5] * d2 * x, q[5] * d2, f2, np.ones_like(x)], 1)
    n = q[2] * f1 + q[5] * f2 + q[6]
    return J, n, (d1, s1, d2, s2)


def grad(q, x, y):
    J, n, _ = jac(q, x)
    return (_sig(n) - y) @ J / len(x)


def hess(q, x, y):
    J, n, (d1, s1, d2, s2) = jac(q, x)
    p = _sig(n); r = p - y; w = p * (1 - p)
    H = (J * w[:, None]).T @ J
    m = len(x)
    for (ia, ib, iv, d, s) in ((0, 1, 2, d1, s1), (3, 4, 5, d2, s2)):
        v = q[iv]
        c = np.sum(r * v * s * x)
        H[ia, ia] += np.sum(r * v * s * x * x); H[ib, ib] += np.sum(r * v * s)
        H[ia, ib] += c; H[ib, ia] += c
        H[ia, iv] += np.sum(r * d * x); H[iv, ia] += np.sum(r * d * x)
        H[ib, iv] += np.sum(r * d); H[iv, ib] += np.sum(r * d)
    return H / m


def gz(z, v, x, y):
    return grad(qof(z, v), x, y)[ZI]


def Hz(z, v, x, y):
    return hess(qof(z, v), x, y)[np.ix_(ZI, ZI)]


def qof(z, v):
    q = np.empty(7); q[ZI] = z; q[VI] = v
    return q


def newton(z0, v, x, y, tol=1e-10, maxit=200):
    """Damped Newton on ∇_z L(·, v) = 0 (backtracking on |∇|).  Accepted iff max|∇| < tol and H_z PD."""
    z = np.asarray(z0, float).copy()
    g = gz(z, v, x, y)
    for _ in range(maxit):
        if np.abs(g).max() < tol:
            break
        H = Hz(z, v, x, y)
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            return z, False, float(np.abs(g).max())
        t = 1.0
        for _ in range(40):
            zn = z + t * d
            gn = gz(zn, v, x, y)
            if np.all(np.isfinite(gn)) and np.linalg.norm(gn) < (1 - 1e-4 * t) * np.linalg.norm(g):
                break
            t *= 0.5
        z, g = zn, gn
    ok = np.abs(g).max() < tol and np.linalg.eigvalsh(Hz(z, v, x, y)).min() > 0
    return z, bool(ok), float(np.abs(g).max())


def gap(z, v, ac):
    """Exact-extrema enclosure of G₊ of φ = v₁f(t₁) + v₂f(t₂) on the asymmetric windows."""
    from src.width2_geometry import gaps
    return gaps(np.array([z[0], z[1], z[2], z[3]]), np.asarray(v, float), ac)["G+"]


def gapm(z, v, ac):
    lo, hi = gap(z, v, ac)
    return 0.5 * (lo + hi)


def canon(z, v):
    """Canonical form with v ≥ 0 (per-unit sign flip (α, β, v) → (−α, −β, −v) is exact because f_a is odd), unit order by
    (α, β mod 2π), β reduced to (−π, π]; returns (kind, windings, canonical hidden)."""
    th = [[z[0], z[1], v[0]], [z[2], z[3], v[1]]]
    for u in th:
        if u[2] < 0:
            u[0], u[1], u[2] = -u[0], -u[1], -u[2]
    k, red = [], []
    for u in th:
        bb = (u[1] + math.pi) % TWO_PI - math.pi
        k.append(int(round((u[1] - bb) / TWO_PI))); red.append((u[0], bb, u[2]))
    dup = abs(red[0][0] - red[1][0]) <= 1e-6 * max(1, abs(red[0][0])) and abs(red[0][1] - red[1][1]) <= 1e-6
    return ("duplicate" if dup else "two-unit"), tuple(k), red


def dz_ds(z, v, dvds, x, y):
    """Branch tangent dz*/ds along v′ = dvds: −H_z⁻¹ (∂²L/∂z∂v · v′)."""
    H = hess(qof(z, v), x, y)
    return -np.linalg.solve(H[np.ix_(ZI, ZI)], H[np.ix_(ZI, VI)] @ dvds)


def grad_gap(z, v, ac, h=1e-6):
    g = np.zeros(5)
    for i in range(4):
        e = np.zeros(5); e[i] = h
        g[i] = (gapm(z + e, v, ac) - gapm(z - e, v, ac)) / (2 * h)
    return g                                                     # b does not enter G₊


def antisym_basis(z, v):
    """For a duplicate point (canonical v ≥ 0 frame, same orientation): the 2-D subspace δ(α₁, β₁) = v₂e, δ(α₂, β₂) = −v₁e
    (first-order function-preserving), orthonormalised in z coordinates."""
    B = np.zeros((5, 2))
    B[0, 0], B[2, 0] = v[1], -v[0]
    B[1, 1], B[3, 1] = v[1], -v[0]
    return np.linalg.qr(B)[0]


def kappa(z, v, dvds, x, y, ac, drop=None):
    """κ (P = I) = λ_min·[∇G·H⁻¹θ*′]/[∇G·θ*′] in z; `drop`: an orthonormal basis of directions excluded from the reduced
    Hessian (λ_min taken on its orthogonal complement)."""
    H = Hz(z, v, x, y)
    tp = dz_ds(z, v, dvds, x, y)
    dG = grad_gap(z, v, ac)
    if drop is not None:
        Q = np.linalg.qr(np.column_stack([drop, np.eye(5)]))[0][:, drop.shape[1]:]
        lam = np.linalg.eigvalsh(Q.T @ H @ Q).min()
    else:
        lam = np.linalg.eigvalsh(H).min()
    # width 2: G₊ also depends on the output direction, so the crossing condition's denominator is the TOTAL derivative of the
    # branch gap along the path, ∇_zG·θ*′ + ∂_vG·v′ (∂_vG·v′ = 0 for a duplicate branch at its switch: G is homogeneous in v).
    h = 1e-6
    dGv = (gapm(z, np.asarray(v) + h * dvds, ac) - gapm(z, np.asarray(v) - h * dvds, ac)) / (2 * h)
    return float(lam * (dG @ np.linalg.solve(H, tp)) / (dG @ tp + dGv)), float(lam), float(dG @ tp), float(dGv)


def continue_ray(z0, s0, d, x, y, ac, s_max, h=0.01, hmin=1e-5):
    """Natural-parameter continuation in log s along the ray v = s·d (d with ‖d‖₁ = 1), tangent predictor + Newton; step
    halved on Newton failure or a jump (> 0.2 sup); stops at the first G₊ sign change (bisected in log s to 1e−10).
    Returns (status, s_switch, z_switch, path)."""
    z, p = np.asarray(z0, float), math.log(s0)
    g0 = gapm(z, s0 * d, ac)
    path = [(s0, float(np.linalg.eigvalsh(Hz(z, s0 * d, x, y)).min()), g0)]
    while p < math.log(s_max):
        s = math.exp(p)
        t = dz_ds(z, s * d, d, x, y) * s                         # dz/dlog s
        pn = p + h
        zn, ok, _ = newton(z + h * t, math.exp(pn) * d, x, y)
        if not ok or np.abs(zn - z - h * t).max() > 0.2:
            h /= 2
            if h < hmin:
                return "failed", None, z, path
            continue
        gn = gapm(zn, math.exp(pn) * d, ac)
        if (gn > 0) != (g0 > 0):
            lo, hi, zl = p, pn, z
            while hi - lo > 1e-10:
                mid = 0.5 * (lo + hi)
                zm, okm, _ = newton(zl + (mid - lo) * dz_ds(zl, math.exp(lo) * d, d, x, y) * math.exp(lo), math.exp(mid) * d, x, y)
                if (gapm(zm, math.exp(mid) * d, ac) > 0) == (g0 > 0):
                    lo, zl = mid, zm
                else:
                    hi = mid
            return "switch", math.exp(0.5 * (lo + hi)), zl, path
        z, p = zn, pn
        path.append((math.exp(p), float(np.linalg.eigvalsh(Hz(z, math.exp(p) * d, x, y)).min()), gn))
        h = min(0.02, 1.5 * h)
    return "none", None, z, path


def lam_reduced(z, v, x, y):
    """(λ_min of the reduced Hessian, λ_min of the antisymmetric block or None, λ_max): for a duplicate point the 2-D
    antisymmetric (split) subspace is excluded, as the idle/symmetry directions are in width2_fold."""
    H = Hz(z, v, x, y)
    ev = np.linalg.eigvalsh(H)
    if canon(z, v)[0] != "duplicate":
        return float(ev[0]), None, float(ev[-1])
    B = antisym_basis(z, v)
    Q = np.linalg.qr(np.column_stack([B, np.eye(5)]))[0][:, 2:]
    return float(np.linalg.eigvalsh(Q.T @ H @ Q).min()), float(np.linalg.eigvalsh(B.T @ H @ B).min()), float(ev[-1])
