"""EXPLORATORY helpers for the Phase 3 width-4 asymmetric-window design (not registered; nothing here is frozen).

Width K (K = 4 here; K = 2 is used only to reproduce W2-A's frozen population numbers as a check of this code).
f_a(t) = t + a·sin t, a = 1.30; asymmetric windows I = [−0.8, 0.8], O = [−2.0, −1.2] ∪ [1.2, 2.4] (Δ = 0.4).
Layout (this file only): z = (α₁..α_K, β₁..β_K, b) (2K + 1 fast coordinates), v = (v₁..v_K) (slow), s = ‖v‖₁.
N(x) = Σ v_i f(α_i x + β_i) + b; loss = mean BCE with logits; analytic gradient and Hessian (numpy, float64).
G₊ = min_O φ − max_I φ (φ = N − b): exact-extrema enclosure (width2_geometry's algorithm generalised to K units),
screened by a dense grid (the dense value bounds G₊ from above).
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/Evan/precision-topology/census")
A = 1.30
TWO_PI = 2 * math.pi
INNER = (-0.8, 0.8)
OUTER = ((-2.0, -1.2), (1.2, 2.4))
XI_D = np.linspace(*INNER, 801)
XO_D = np.concatenate([np.linspace(*OUTER[0], 401), np.linspace(*OUTER[1], 401)])
DUP_TOL = 1e-6


def nice():
    try:
        os.nice(15)
    except OSError:
        pass


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3      # macOS: bytes


def rss_guard():
    if rss_gb() > 3.0:
        raise SystemExit("STOP: RSS above 3 GB")


def population():
    from src.asym_register import _setup
    x, y = _setup()
    return np.asarray(x, float), np.asarray(y, float)


def own_sample(seed):
    from src.asym_register import training_set
    x, y = training_set(seed)
    return np.asarray(x, float), np.asarray(y, float)


# ------------------------------------------------------------------------------------------ the network
def _f(t):
    st = np.sin(t)
    return t + A * st, 1 + A * np.cos(t), -A * st


def _sig(n):
    return 0.5 * (1 + np.tanh(0.5 * n))


def K_of(z):
    return (len(z) - 1) // 2


def loss(z, v, x, y):
    K = K_of(z)
    t = np.outer(x, z[:K]) + z[K:2 * K]
    n = (t + A * np.sin(t)) @ v + z[-1]
    return float(np.mean(np.logaddexp(0, n) - y * n))


def _parts(z, v, x):
    K = K_of(z)
    t = np.outer(x, z[:K]) + z[K:2 * K]
    f, d, s = _f(t)
    n = f @ v + z[-1]
    return K, f, d, s, n


def grad(z, v, x, y):
    """(∇_z L, ∇_v L)."""
    K, f, d, s, n = _parts(z, v, x)
    r = (_sig(n) - y) / len(x)
    vd = d * v
    gz = np.concatenate([(r * x) @ vd, r @ vd, [r.sum()]])
    return gz, r @ f


def gz(z, v, x, y):
    return grad(z, v, x, y)[0]


def hess(z, v, x, y):
    """Full Hessian in (z, v) order (3K + 1)."""
    K, f, d, s, n = _parts(z, v, x)
    m = len(x)
    p = _sig(n)
    r = p - y
    w = p * (1 - p)
    vd = d * v
    J = np.concatenate([vd * x[:, None], vd, np.ones((m, 1)), f], 1)       # ∂N/∂(α, β, b, v)
    H = (J * w[:, None]).T @ J
    vs = s * v
    for i in range(K):
        ia, ib, iv = i, K + i, 2 * K + 1 + i
        H[ia, ia] += np.sum(r * vs[:, i] * x * x)
        H[ib, ib] += np.sum(r * vs[:, i])
        c = np.sum(r * vs[:, i] * x)
        H[ia, ib] += c
        H[ib, ia] += c
        c = np.sum(r * d[:, i] * x)
        H[ia, iv] += c
        H[iv, ia] += c
        c = np.sum(r * d[:, i])
        H[ib, iv] += c
        H[iv, ib] += c
    return H / m


def blocks(z, v, x, y):
    H = hess(z, v, x, y)
    nz = len(z)
    return H[:nz, :nz], H[:nz, nz:]


def newton(z0, v, x, y, tol=1e-10, maxit=100):
    """Damped Newton on ∇_z L(·, v) = 0.  Accepted iff max|∇| < tol and H_z positive definite."""
    z = np.asarray(z0, float).copy()
    g = gz(z, v, x, y)
    for _ in range(maxit):
        if np.abs(g).max() < tol:
            break
        H = blocks(z, v, x, y)[0]
        try:
            dz = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            return z, False
        t = 1.0
        for _ in range(40):
            zn = z + t * dz
            gn = gz(zn, v, x, y)
            if np.all(np.isfinite(gn)) and np.linalg.norm(gn) < (1 - 1e-4 * t) * np.linalg.norm(g):
                break
            t *= 0.5
        z, g = zn, gn
    if not np.abs(g).max() < tol:
        return z, False
    return z, bool(np.linalg.eigvalsh(blocks(z, v, x, y)[0]).min() > 0)


def dz_dv(z, v, x, y):
    Hzz, Hzv = blocks(z, v, x, y)
    return -np.linalg.solve(Hzz, Hzv)


# ------------------------------------------------------------------------------------------ the gap
_ACT = None


def act():
    global _ACT
    if _ACT is None:
        from src.width2_geometry import Act
        _ACT = Act("fa", A)
    return _ACT


def phi(xs, z, v):
    K = K_of(z)
    t = np.outer(np.asarray(xs, float), z[:K]) + z[K:2 * K]
    return (t + A * np.sin(t)) @ np.asarray(v, float)


def dphi(xs, z, v):
    K = K_of(z)
    t = np.outer(np.asarray(xs, float), z[:K]) + z[K:2 * K]
    return (1 + A * np.cos(t)) @ (np.asarray(v, float) * z[:K])


def _m2(m, h, z, v):
    K = K_of(z)
    ac = act()
    out = 0.0
    for i in range(K):
        al, be = z[i], z[K + i]
        out = out + abs(v[i]) * al ** 2 * ac.d2u_bound(al * (m - h) + be, al * (m + h) + be)
    return out


def extrema(z, v, lo, hi, tol=1e-9, vtol=1e-13, n0=64, max_cells=400_000):
    """width2_geometry.extrema with K units: certified enclosures (min_lo, min_hi, max_lo, max_hi) of φ on [lo, hi]."""
    ends = phi(np.array([lo, hi]), z, v)
    mn_lo = mn_hi = float(ends.min())
    mx_lo = mx_hi = float(ends.max())
    edges = np.linspace(lo, hi, n0 + 1)
    m = 0.5 * (edges[:-1] + edges[1:])
    h = np.full(n0, 0.5 * (hi - lo) / n0)
    total = 0
    while len(m):
        total += len(m)
        if total > max_cells:
            raise RuntimeError("extrema: cell cap")
        d = dphi(m, z, v)
        M2 = _m2(m, h, z, v)
        mono = np.abs(d) > M2 * h
        if mono.any():
            e = phi(np.concatenate([m[mono] - h[mono], m[mono] + h[mono]]), z, v)
            mn_lo = min(mn_lo, float(e.min())); mn_hi = min(mn_hi, float(e.min()))
            mx_lo = max(mx_lo, float(e.max())); mx_hi = max(mx_hi, float(e.max()))
        m, h, d, M2 = m[~mono], h[~mono], d[~mono], M2[~mono]
        r_all = np.abs(d) * h + 0.5 * M2 * h ** 2
        small = (h < tol) | (r_all < vtol)
        if small.any():
            fv = phi(m[small], z, v)
            r = r_all[small]
            mn_lo = min(mn_lo, float((fv - r).min())); mn_hi = min(mn_hi, float(fv.min()))
            mx_lo = max(mx_lo, float(fv.max())); mx_hi = max(mx_hi, float((fv + r).max()))
        m, h = m[~small], h[~small]
        h = h / 2
        m = np.concatenate([m - h, m + h])
        h = np.concatenate([h, h])
    return mn_lo, mn_hi, mx_lo, mx_hi


def gap_enc(z, v):
    _, _, ixl, ixh = extrema(z, v, *INNER)
    oL, oR = extrema(z, v, *OUTER[0]), extrema(z, v, *OUTER[1])
    return float(min(oL[0], oR[0]) - ixh), float(min(oL[1], oR[1]) - ixl)


def gap_mid(z, v):
    lo, hi = gap_enc(z, v)
    return 0.5 * (lo + hi)


def gap_dense(z, v):
    return float(phi(XO_D, z, v).min() - phi(XI_D, z, v).max())


def placed_state(z, v):
    """Observation rule: enclosure lower end > 0 (dense screen)."""
    if gap_dense(z, v) <= 0:
        return False
    return gap_enc(z, v)[0] > 0


def branch_placed(z, v):
    """Branch / prediction rule: enclosure midpoint > 0 (dense screen)."""
    return gap_dense(z, v) > 0 and gap_mid(z, v) > 0


# ------------------------------------------------------------------------------------------ symmetries, types
def shift(z, v, k):
    """2π windings: β_i + 2πk_i, b − 2π Σ k_i v_i."""
    z = np.asarray(z, float).copy()
    K = K_of(z)
    k = np.asarray(k, float)
    z[K:2 * K] += TWO_PI * k
    z[-1] -= TWO_PI * float(k @ np.asarray(v, float))
    return z


def windings(z):
    """(k, reduced z): β_i = β̃_i + 2πk_i with β̃ ∈ (−π, π]; b adjusted is NOT done here (classification only)."""
    K = K_of(z)
    b = z[K:2 * K]
    red = (b + math.pi) % TWO_PI - math.pi
    k = np.rint((b - red) / TWO_PI).astype(int)
    return tuple(int(i) for i in k), red


def groups(z, tol=DUP_TOL):
    """Partition of the units into coinciding groups (α equal, β equal mod 2π), as tuples of unit indices."""
    K = K_of(z)
    _, red = windings(z)
    left, out = list(range(K)), []
    while left:
        i = left.pop(0)
        g = [i]
        for j in list(left):
            db = red[i] - red[j]
            db -= TWO_PI * round(db / TWO_PI)
            if abs(z[i] - z[j]) <= tol * max(1.0, abs(z[i])) and abs(db) <= tol:
                g.append(j)
                left.remove(j)
        out.append(tuple(g))
    return tuple(out)


def btype(z, v):
    """Branch type label: the coincidence partition (group sizes, sorted) and, per group in order of total share, the
    sign of α ('+'/'-'; '0' if |α| < 1e-6).  E.g. '1111:+-+-' (four distinct units), '4:+' (all coincide)."""
    gs = groups(z)
    K = K_of(z)
    sh = [sum(v[i] for i in g) for g in gs]
    order = np.argsort(-np.asarray(sh), kind="stable")
    sizes = "".join(str(len(gs[o])) for o in order)
    sg = "".join("0" if abs(z[gs[o][0]]) < 1e-6 else ("+" if z[gs[o][0]] > 0 else "-") for o in order)
    return f"{sizes}:{sg}"


def split_basis(z, v):
    """Orthonormal basis of the split subspace: for each coinciding group, the 2(m − 1) first-order function-preserving
    directions δ(α_i, β_i) = v_j e, δ(α_j, β_j) = −v_i e (pairs (g₀, j), j in the group)."""
    K = K_of(z)
    cols = []
    for g in groups(z):
        i = g[0]
        for j in g[1:]:
            for off in (0, K):
                c = np.zeros(2 * K + 1)
                c[off + i], c[off + j] = v[j], -v[i]
                cols.append(c)
    if not cols:
        return None
    return np.linalg.qr(np.column_stack(cols))[0]


def reduced_lam(H, z, v):
    H = 0.5 * (H + H.T)
    B = split_basis(z, v)
    if B is None:
        return float(np.linalg.eigvalsh(H).min()), None
    n, m = H.shape[0], B.shape[1]
    Q = np.linalg.qr(np.column_stack([B, np.eye(n)]))[0][:, m:]
    return float(np.linalg.eigvalsh(Q.T @ H @ Q).min()), float(np.linalg.eigvalsh(B.T @ H @ B).min())


# ------------------------------------------------------------------------------------------ κ (total derivative)
def grad_gap(z, v, vp, h=1e-6):
    K = K_of(z)
    g0 = gap_mid(z, v)
    out, dis = np.zeros(2 * K + 1), []
    for i in range(2 * K):
        e = np.zeros(2 * K + 1)
        e[i] = h
        gp, gm = gap_mid(z + e, v), gap_mid(z - e, v)
        out[i] = (gp - gm) / (2 * h)
        dis.append(abs((gp - g0) - (g0 - gm)) / h)
    gp, gm = gap_mid(z, v + h * vp), gap_mid(z, v - h * vp)
    dv = (gp - gm) / (2 * h)
    dis.append(abs((gp - g0) - (g0 - gm)) / h)
    sc = max(float(np.abs(out).max()), abs(dv), 1e-300)
    return out, float(dv), float(max(dis) / sc)


def kappa_at(z, v, vp, x, y):
    """κ₀ = λ_min·[∇_zG·H⁻¹θ*′]/[∇_zG·θ*′ + ∂_vG·v′] (reduced λ_min: split directions excluded), winding coefficients
    a_i = −2π·λ·(H⁻¹∇_zG)_b·v′_i/[dG/ds] (κ_k = κ₀ + a·k)."""
    Hzz, Hzv = blocks(z, v, x, y)
    vp = np.asarray(vp, float)
    tan = -np.linalg.solve(Hzz, Hzv @ vp)
    lam, lam_split = reduced_lam(Hzz, z, v)
    dGz, dGv, dis = grad_gap(z, v, vp)
    u = np.linalg.solve(Hzz, dGz)
    den = float(dGz @ tan + dGv)
    kap0 = float(lam * (u @ tan) / den)
    a = [float(-TWO_PI * lam * u[-1] * vp[i] / den) for i in range(len(vp))]
    return {"kappa0": kap0, "a": a, "lam": lam, "lam_split": lam_split, "dGz_tan": float(dGz @ tan), "dGv": dGv,
            "dG_ds": den, "gradG_rel_dis": dis}


# ------------------------------------------------------------------------------------------ continuation
def flow_rhs(z, v, x, y):
    g = -grad(z, v, x, y)[1]
    ds = float(np.sign(v) @ g)
    return (g / ds) if ds > 0 else None


def rk4_step(z, v, hs, x, y):
    def stage(zb, vb, vn):
        return newton(zb + dz_dv(zb, vb, x, y) @ (vn - vb), vn, x, y)
    k1 = flow_rhs(z, v, x, y)
    if k1 is None:
        return z, v, False
    v2 = v + 0.5 * hs * k1
    z2, ok = stage(z, v, v2)
    k2 = flow_rhs(z2, v2, x, y) if ok else None
    if k2 is None:
        return z, v, False
    v3 = v + 0.5 * hs * k2
    z3, ok = stage(z, v, v3)
    k3 = flow_rhs(z3, v3, x, y) if ok else None
    if k3 is None:
        return z, v, False
    v4 = v + hs * k3
    z4, ok = stage(z, v, v4)
    k4 = flow_rhs(z4, v4, x, y) if ok else None
    if k4 is None:
        return z, v, False
    vn = v + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    zn, ok = stage(z, v, vn)
    return zn, vn, ok


def continue_adiabatic(z0, v0, x, y, s_max, hs_rel=0.005):
    """Adiabatic reference (reduced gradient flow of v, RK4 in s): first branch-gap sign change, bisected."""
    z, v = np.asarray(z0, float), np.asarray(v0, float)
    g0 = gap_mid(z, v)
    min_lam, min_split, n = float("inf"), float("inf"), 0
    while True:
        s = float(np.abs(v).sum())
        if s >= s_max:
            return {"status": "none", "s_last": s, "min_split": min_split}
        hs = hs_rel * s
        zn, vn, ok = rk4_step(z, v, hs, x, y)
        if not ok:
            return {"status": "failed", "s_last": s, "z": z, "v": v, "min_split": min_split,
                    "groups_last": [list(t) for t in groups(z)]}
        lam, lsp = reduced_lam(blocks(zn, vn, x, y)[0], zn, vn)
        min_lam, n = min(min_lam, lam), n + 1
        if lsp is not None:
            min_split = min(min_split, lsp)
        gd = gap_dense(zn, vn)
        gn = gap_mid(zn, vn) if gd > -0.05 * s else gd
        if (gn > 0) != (g0 > 0):
            lo, hi = 0.0, hs
            while hi - lo > 1e-12 * s:
                mid = 0.5 * (lo + hi)
                zm, vm, okm = rk4_step(z, v, mid, x, y)
                if okm and (gap_mid(zm, vm) > 0) == (g0 > 0):
                    lo = mid
                else:
                    hi = mid
            zs, vs, oks = rk4_step(z, v, 0.5 * (lo + hi), x, y)
            return {"status": "switch", "s_switch": float(np.abs(vs).sum()), "z": zs, "v": vs,
                    "vprime": flow_rhs(zs, vs, x, y), "ok": oks, "min_lam": min_lam, "min_split": min_split, "n": n,
                    "G0": g0, "groups_switch": [list(t) for t in groups(zs)]}
        z, v = zn, vn


def continue_ray(z0, v0, x, y, s_max, h=0.02):
    """Continuation along the ray v = (s/s₀)·v₀ (fixed shares), log-s steps h; first sign change bisected."""
    z, d = np.asarray(z0, float), np.asarray(v0, float) / np.abs(v0).sum()
    p = math.log(np.abs(v0).sum())
    g0 = gap_mid(z, math.exp(p) * d)
    min_lam = float("inf")
    while p < math.log(s_max):
        J = dz_dv(z, math.exp(p) * d, x, y) @ d * math.exp(p)
        zn, ok = newton(z + h * J, math.exp(p + h) * d, x, y)
        if not ok or np.abs(zn - z - h * J).max() > 0.2:
            h /= 2
            if h < 1e-6:
                return {"status": "failed", "s_last": math.exp(p)}
            continue
        vn = math.exp(p + h) * d
        min_lam = min(min_lam, reduced_lam(blocks(zn, vn, x, y)[0], zn, vn)[0])
        gn = gap_mid(zn, vn)
        if (gn > 0) != (g0 > 0):
            lo, hi, zl = p, p + h, z
            while hi - lo > 1e-11:
                mid = 0.5 * (lo + hi)
                Jl = dz_dv(zl, math.exp(lo) * d, x, y) @ d * math.exp(lo)
                zm, okm = newton(zl + (mid - lo) * Jl, math.exp(mid) * d, x, y)
                if okm and (gap_mid(zm, math.exp(mid) * d) > 0) == (g0 > 0):
                    lo, zl = mid, zm
                else:
                    hi = mid
            return {"status": "switch", "s_switch": math.exp(0.5 * (lo + hi)), "z": zl, "v": math.exp(lo) * d,
                    "vprime": d, "min_lam": min_lam}
        z, p = zn, p + h
        h = min(0.02, 1.5 * h)
    return {"status": "none"}


# ------------------------------------------------------------------------------------------ training, own path, R4
def hold(z, v0, W, x, y, lr=1.0, check_every=1):
    z = np.asarray(z, float).copy()
    npos = int(placed_state(z, v0))
    for k in range(1, int(W) + 1):
        z = z - lr * gz(z, v0, x, y)
        if k % check_every == 0 and gap_dense(z, v0) > 0 and placed_state(z, v0):
            npos += 1
    return z, npos


class OwnBranch:
    def __init__(self, V, z_anchor, x, y, jump=0.2):
        self.V, self.x, self.y, self.jump = np.asarray(V, float), x, y, jump
        self.Z, self.H, self.lam, self.lost_at, self._hzv = [], [], [], None, None
        z, ok = newton(z_anchor, self.V[0], x, y)
        if ok:
            self._app(z, 0)
        else:
            self.lost_at = 0

    def _app(self, z, t):
        Hzz, Hzv = blocks(z, self.V[t], self.x, self.y)
        self.Z.append(z); self.H.append(Hzz); self._hzv = Hzv
        self.lam.append(reduced_lam(Hzz, z, self.V[t])[0])

    def ensure(self, t):
        while len(self.Z) <= t:
            k = len(self.Z)
            if self.lost_at is not None or k >= len(self.V):
                return False
            pred = self.Z[-1] + (-np.linalg.solve(self.H[-1], self._hzv)) @ (self.V[k] - self.V[k - 1])
            zn, ok = newton(pred, self.V[k], self.x, self.y)
            if not ok or np.abs(zn - pred).max() > self.jump:
                self.lost_at = k
                return False
            self._app(zn, k)
        return True


def own_path_switch(br, t_max, t_from=1):
    for t in range(max(1, t_from), int(t_max) + 1):
        if not br.ensure(t):
            return None, float("nan")
        if branch_placed(br.Z[t], br.V[t]):
            lo, hi, zl = 0.0, 1.0, br.Z[t - 1]
            v0, v1 = br.V[t - 1], br.V[t]
            while hi - lo > 1e-12:
                mid = 0.5 * (lo + hi)
                vm = v0 + mid * (v1 - v0)
                zm, okm = newton(zl, vm, br.x, br.y)
                if okm and gap_mid(zm, vm) <= 0:
                    lo, zl = mid, zm
                else:
                    hi = mid
            return int(t), float(np.abs(v0 + 0.5 * (lo + hi) * (v1 - v0)).sum())
    return None, float("nan")


def r4(br, z_rel, eta, t_max):
    if not br.ensure(0):
        return None, "lost"
    d = np.asarray(z_rel, float) - br.Z[0]
    for t in range(0, int(t_max) + 1):
        if not br.ensure(t):
            return None, "lost"
        if t % 10 == 0 and np.abs(1 - eta * np.linalg.eigvalsh(0.5 * (br.H[t] + br.H[t].T))).max() > 1:
            return None, "unstable"
        if t >= 1 and branch_placed(br.Z[t] + d, br.V[t]):
            return t, "ok"
        if t == t_max or not br.ensure(t + 1):
            break
        d = d - eta * (br.H[t] @ d) - (br.Z[t + 1] - br.Z[t])
        if np.abs(d).max() > 1:
            return None, "diverged"
    return None, "none"


def train(z0, v0, eta, rho, n_max, x, y, stop_after_cross=0, t_sw_stop=None):
    """Full-batch SGD (η on z, ρη on v).  Returns (Z, V, t_obs) with t_obs the first t ≥ 1 placed (lower end > 0); runs
    stop_after_cross more steps after the crossing (or to n_max)."""
    z, v = np.asarray(z0, float).copy(), np.asarray(v0, float).copy()
    Z, V = [z.copy()], [v.copy()]
    t_obs = None
    for t in range(1, int(n_max) + 1):
        g_z, g_v = grad(z, v, x, y)
        z = z - eta * g_z
        v = v - rho * eta * g_v
        Z.append(z.copy()); V.append(v.copy())
        if t_obs is None and gap_dense(z, v) > 0 and placed_state(z, v):
            t_obs = t
        if t_obs is not None and t >= t_obs + stop_after_cross:
            break
    return np.array(Z), np.array(V), t_obs


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return round(time.time() - self.t0, 1)
