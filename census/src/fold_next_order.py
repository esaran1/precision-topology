"""Track C (theory): the NEXT-ORDER term of the fold-passage delay, from the normal form at the fold, with no fitted
constant (math note §18).

Result (§18; formal matched asymptotics, checked against direct integration of synthetic normal forms in
tests/test_fold_next_order.py).  For the slow-fast system, in steps,

    x_{n+1} = x_n + F(x_n, σ_n),   σ_{n+1} = σ_n + ρ·G(x_n, σ_n),
    F = p σ + q x² + f₃ x³ + f₁₁ x σ + …,   G = g₀ + g₁ x + …     (p, q, g₀ > 0; escape toward x → +∞),

the slow coordinate σ at the first step where the threshold is passed is

    σ_q = Ω₀ μ + K ρ ln(1/ρ) + D ρ + O(ρ^{4/3} ln²ρ),   μ = (g₀ρ)^{2/3}(pq)^{-1/3},
    K = −A g₀/(3q),
    D = (g₀/q)·(A ln λ̂ − A c_A − B/2) + J + g₀/2,          λ̂ = (g₀ p)^{1/3} q^{-2/3},
    A = (f₃ − q²)/q − g₁/g₀,   B = (f₁₁ − pq)/p − g₁/g₀    (the −q², −pq and g₀/2 are the discrete-step terms;
                                                          drop them for a flow),
    c_A = lim_{τ→0} [∫_{a₁+τ}^∞ Ai′³/Ai dz + k² ln τ]/k² = −1.00879…,  c_B = 1/2,  k = Ai′(a₁),
    J  = lim_{x_m→0} [Σ_{steps from x_m to the threshold} G − g₀/(q x_m) − (g₀A/q) ln x_m]
         (the regularised escape sum of the frozen-σ map along the unstable branch to the threshold).

Phase 2A's landscape (src/phase2a.py): x is the coordinate along the fold's null vector in the 12-dimensional fixed-s
tangent of the active units (metric P = I, η = 1), σ = s − s_F, G = ṡ/ρ = −3η ∂L/∂s, threshold ρ₂ ≥ q.  All inputs are
FROZEN landscape quantities (results/phase2a/frozen.json; results/sb_fold/ continuation): the fold point is refined
from them, the coefficients are derivatives of the loss at it, J is one deterministic frozen-s gradient-descent run
from the fold.  Nothing is trained; `predict` reads no run, observation or score file.

    python -m src.fold_next_order predict     # -> results/fold_next_order.json (the prediction; committed first)
"""

from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FROZEN = RESULTS / "phase2a" / "frozen.json"
OUT_JSON = RESULTS / "fold_next_order.json"
GATE_LOG = RESULTS / "fold_next_order_memory_gate.log"
RSS_LIMIT = 1024 ** 3                     # stop above 1 GB
ETA = 1.0
X0_ESCAPE = 0.005                         # start of the frozen-s escape run (CM point at x = 0.005)
XM_GRID = (0.4, 0.2, 0.1, 0.05, 0.025, 0.018, 0.0125, 0.009)
XM_RICH = (0.018, 0.009)                  # two-point Richardson (error O(x_m)) -> J
XM_RICH_CHECK = (0.025, 0.0125)


# ============================================================================================ universal constants
def airy_constants(dps=30):
    """Ω₀ = −a₁, k = Ai′(a₁), c_A (regularised ∫ Ai′³/Ai), c_B = −∫_{a₁}^∞ z Ai Ai′ dz / k² (= 1/2 exactly)."""
    import mpmath as mp
    mp.mp.dps = dps
    a1 = mp.airyaizero(1)
    k = mp.airyai(a1, derivative=1)
    f = lambda z: mp.airyai(z, derivative=1) ** 3 / mp.airyai(z)                    # noqa: E731
    i1 = mp.quad(lambda z: f(z) - k ** 2 / (z - a1), [a1, a1 + 0.5, a1 + 1])
    i2 = mp.quad(f, [a1 + 1, 0, 2, 5, 10, mp.inf])
    cB = mp.quad(lambda z: -z * mp.airyai(z) * mp.airyai(z, derivative=1), [a1, 0, 5, mp.inf]) / k ** 2
    return {"Omega0": float(-a1), "k": float(k), "c_A": float((i1 + i2) / k ** 2), "c_B": float(cB)}


OMEGA0 = 2.338107410459767
C_A = -1.0087907567050538                 # airy_constants()["c_A"] (asserted in the tests and the ledger)
C_B = 0.5                                 # exact: ∫ z Ai Ai′ dz = Ai′²/2


# ============================================================================================ the expansion
def inner_AB(p, q, f3, f11, g0, g1, discrete=True):
    """A, B of the inner problem dX/dT = T + X² + δ(A X³ + B X T), δ = λ.  discrete: the map's modified field
    F − ½F_xF adds −q² to f₃ and −pq to f₁₁."""
    if discrete:
        f3, f11 = f3 - q * q, f11 - p * q
    return f3 / q - g1 / g0, f11 / p - g1 / g0


def coefficients(p, q, g0, A, B, J, discrete=True):
    """K, D of σ_q = Ω₀μ + Kρ ln(1/ρ) + Dρ, and the pieces of D."""
    lam_hat = (g0 * p) ** (1 / 3) * q ** (-2 / 3)
    c1 = A * C_A + B * C_B
    inner = (g0 / q) * (A * math.log(lam_hat) - c1)
    half = 0.5 * g0 if discrete else 0.0
    return {"K": -A * g0 / (3 * q), "D": inner + J + half, "D_inner": inner, "D_J": J, "D_half_step": half,
            "lambda_hat": lam_hat, "c1": c1}


def expansion(rho, p, q, g0, K, D):
    """σ_q and its local exponent d ln σ_q / d ln ρ (= d ln r / d ln ε, since ε ∝ ρ exactly)."""
    mu = (g0 * rho) ** (2 / 3) * (p * q) ** (-1 / 3)
    lead = OMEGA0 * mu
    lg = K * rho * math.log(1 / rho)
    lin = D * rho
    sig = lead + lg + lin
    dsig = (2 / 3) * lead + K * rho * (math.log(1 / rho) - 1) + lin
    return {"mu": mu, "lead": lead, "log_term": lg, "lin_term": lin, "sigma": sig, "slope": dsig / sig}


# ============================================================================================ synthetic normal forms
def _rk45(f, z, h):
    a = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
         [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
         [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
         [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
    b5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    b4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
    K = []
    for i in range(7):
        K.append(f(z + h * sum(a[i][j] * K[j] for j in range(i)) if i else z))
    K = np.array(K)
    z5 = z + h * (b5 @ K)
    return z5, float(np.abs(z5 - (z + h * (b4 @ K))).max())


def _start_on_branch(Fx, s0, p, q):
    lo, hi = -3 * math.sqrt(-p * s0 / q), -1e-15
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if Fx(mid, s0) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def simulate_flow(eps, p, q, f3, f11, g0, g1, xq, T0=-10.0, tol=1e-13):
    """σ when x first reaches xq for ẋ = F, σ̇ = εG (polynomial normal form), started on the attracting branch at
    σ₀ = T0·μ (exponentially small memory of the start).  Adaptive Dormand–Prince; the crossing is resolved by halving."""
    F = lambda x, s: p * s + q * x * x + f3 * x ** 3 + f11 * x * s                     # noqa: E731
    mu = (g0 * eps) ** (2 / 3) * (p * q) ** (-1 / 3)
    s0 = T0 * mu
    z = np.array([_start_on_branch(F, s0, p, q), s0])
    f = lambda z: np.array([F(z[0], z[1]), eps * (g0 + g1 * z[0])])                    # noqa: E731
    h = 1e-3
    while True:
        zn, err = _rk45(f, z, h)
        if not np.isfinite(err) or err > tol * max(1.0, np.abs(z).max()):
            h *= max(0.2, 0.9 * (tol / err) ** 0.2) if np.isfinite(err) else 0.2
            continue
        if zn[0] >= xq:
            if h < 1e-12:
                return float(zn[1])
            h *= 0.5
            continue
        z = zn
        h *= min(2.0, 0.9 * (tol / max(err, 1e-300)) ** 0.2)


def simulate_map(eps, p, q, f3, f11, g0, g1, xq, T0=-12.0):
    """σ at the threshold for the map x ← x + F, σ ← σ + εG (coefficients per step), linear interpolation within the
    crossing step (removes the O(ε·G) integer-step jitter so the O(ε) term can be checked exactly)."""
    F = lambda x, s: p * s + q * x * x + f3 * x ** 3 + f11 * x * s                     # noqa: E731
    mu = (g0 * eps) ** (2 / 3) * (p * q) ** (-1 / 3)
    s = T0 * mu
    x = _start_on_branch(F, s, p, q)
    while True:
        xn, sn = x + F(x, s), s + eps * (g0 + g1 * x)
        if xn >= xq:
            return s + (xq - x) / (xn - x) * (sn - s)
        x, s = xn, sn


def J_flow(q, f3, g0, g1, A, xq):
    """J for the flow normal form: ∫₀^{xq} [G/F − g₀/(qx²) + (g₀A/q)/x] dx − g₀/(q xq) − (g₀A/q) ln xq (F, G at σ = 0;
    mpmath at 40 digits: the integrand is a difference of 1/x² terms)."""
    import mpmath as mp
    mp.mp.dps = 40
    R = lambda x: (g0 + g1 * x) / (q * x * x + f3 * x ** 3) - g0 / (q * x * x) + (g0 * A / q) / x   # noqa: E731
    return float(mp.quad(R, [0, xq / 2, xq])) - g0 / (q * xq) - (g0 * A / q) * math.log(xq)


def J_map(q, f3, g0, g1, A, xq, x0=1e-4, xms=(0.005, 0.0025)):
    """J for the map normal form (σ frozen at 0): the regularised escape sum with the same interpolated crossing as
    simulate_map, two-point Richardson in x_m."""
    F = lambda x: q * x * x + f3 * x ** 3                                              # noqa: E731
    x, xs, Gs = x0, [], []
    while x < xq:
        xs.append(x); Gs.append(g0 + g1 * x); x = x + F(x)
    xs, Gs = np.array(xs), np.array(Gs)
    frac = (xq - xs[-1]) / (x - xs[-1])
    tail = np.cumsum(Gs[::-1])[::-1] - (1 - frac) * Gs[-1]
    Js = []
    for xm in xms:
        k = int(np.argmax(xs >= xm))
        Js.append(tail[k] - g0 / (q * xs[k]) - (g0 * A / q) * math.log(xs[k]))
    return 2 * Js[1] - Js[0]


# ============================================================================================ machine rules
def memory_gate(tag):
    """free ≥ 25 % and swap free ≥ 500 MB (act_fold.check_memory; waits, re-checking every 60 s), logged."""
    from .act_fold import check_memory
    while True:
        try:
            ok, f, w = check_memory()
        except Exception:                                                                # noqa: BLE001
            ok, f, w = False, float("nan"), float("nan")
        with open(GATE_LOG, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB {'OK' if ok else 'WAIT'}\n")
        if ok:
            return
        time.sleep(60)


def rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss                             # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 1 GB")
    return rss


def _setup():
    cur = os.getpriority(os.PRIO_PROCESS, 0)
    if cur < 15:
        os.nice(15 - cur)
    import torch
    torch.set_num_threads(1)
    return torch


# ============================================================================================ Phase 2A's landscape
class Landscape:
    """The 2A loss in fast coordinates y (orthonormal basis of the fixed-s tangent of the active units) and s."""

    def __init__(self, th_ref):
        torch = _setup()
        from . import phase2a as P2
        self.torch, self.P2 = torch, P2
        self.X = torch.as_tensor(P2.X, dtype=torch.float64)
        self.Y = torch.as_tensor(P2.Y, dtype=torch.float64)
        th_ref = np.asarray(th_ref, float)
        act = [k for k in range(4) if th_ref[12 + k] != 0.0]
        self.active = act
        idx = [2 * k for k in act] + [2 * k + 1 for k in act] + [8 + k for k in act] + [12 + k for k in act] + [16]
        ahat = np.zeros(17); ahat[[12 + k for k in act]] = np.sign(th_ref[[12 + k for k in act]]) / math.sqrt(len(act))
        m = len(idx)
        U, _, _ = np.linalg.svd(np.eye(m) - np.outer(ahat[idx], ahat[idx]))
        E = np.zeros((17, m - 1)); E[idx, :] = U[:, :m - 1]
        self.ahat, self.E, self.n = ahat, E, m - 1
        self.At, self.Et = torch.tensor(ahat), torch.tensor(E)
        self.sqrt_n = math.sqrt(len(act))

    def loss(self, th):
        torch = self.torch
        W = th[:8].reshape(4, 2); c = th[8:12]; v = th[12:16]; b = th[16]
        z = torch.tanh(self.X @ W.T + c) @ v + b
        return (torch.nn.functional.binary_cross_entropy_with_logits(z, self.Y)
                + 0.5 * self.P2.LAM * ((W ** 2).sum() + (c ** 2).sum()))

    def theta(self, base, y, ds):
        """θ = base + E y + â·ds/√n_active (s = Σ|v| = √n â·v on the active units)."""
        return base + self.Et @ y + self.At * (ds / self.sqrt_n)


def fold_point(fr):
    """Moore–Spence Newton for (y, s, r): ∇_y L = 0, H r = 0, r₀·r = 1, started from the frozen M branch (sb_fold) at
    s_F − 2e-3; returns θ_F (training coordinates), s_F (refined) and the unit null vector r (17-vector)."""
    from . import phase2a as P2
    th_b = P2.branch_theta("M", fr["s_F"] - 2e-3)
    Ls = Landscape(th_b)
    torch = Ls.torch
    base = torch.tensor(th_b)
    s_b = float(np.abs(th_b[12:16]).sum())

    def phi(y, s):
        return Ls.loss(Ls.theta(base, y, s - s_b))

    def res(z):
        y, s, r = z[:Ls.n], z[Ls.n], z[Ls.n + 1:]
        g = torch.autograd.grad(phi(y, s), y, create_graph=True)[0] if y.requires_grad else None
        if g is None:
            yy = y.clone().requires_grad_(True)
            g = torch.autograd.grad(phi(yy, s), yy, create_graph=True)[0]
        H = torch.autograd.functional.hessian(lambda yy: phi(yy, s), y, create_graph=True)
        return torch.cat([g, H @ r, (r0 @ r - 1).reshape(1)])

    y0 = torch.zeros(Ls.n, dtype=torch.float64)
    H0 = torch.autograd.functional.hessian(lambda yy: phi(yy, torch.tensor(s_b, dtype=torch.float64)), y0).numpy()
    r0 = torch.tensor(np.linalg.eigh(H0)[1][:, 0])
    z = torch.cat([y0, torch.tensor([s_b], dtype=torch.float64), r0])
    hist = []
    for _ in range(30):
        F = res(z).detach()
        J = torch.autograd.functional.jacobian(res, z)
        dz = torch.linalg.solve(J, -F)
        z = (z + dz).detach()
        hist.append([float(F.abs().max()), float(dz.abs().max())])
        if hist[-1][0] < 1e-14 and hist[-1][1] < 1e-12:
            break
    y, s = z[:Ls.n], float(z[Ls.n])
    r = z[Ls.n + 1:] / z[Ls.n + 1:].norm()
    thF = Ls.theta(base, y, torch.tensor(s - s_b, dtype=torch.float64)).detach().numpy()
    return thF, s, (Ls.Et @ r).numpy(), hist


def fold_coefficients(thF, r17, eta=ETA):
    """Normal-form coefficients at the fold (fast coordinates y about θ_F, s offset σ):
    c = ½T[r,r,r], m = rᵀ∂_s∇L, κ₃ = Q₄/6 − ½T_rrᵀH⁺T_rr, κ₁₁ = ∂_sH[r,r] − T_rrᵀH⁺∂_s∇L (centre-manifold reduced
    gradient ∂_xL_red = mσ + cx² + κ₃x³ + κ₁₁xσ), g₀ = −3η∂_sL, g₁ = e·∇_y G; escape orientation e = ±r with
    ½T[e,e,e] < 0; then p = −η e·∂_s∇L, q = −η½T[e,e,e], f₃ = −ηκ₃, f₁₁ = −ηκ₁₁."""
    Ls = Landscape(thF)
    torch = Ls.torch
    base = torch.tensor(thF)
    phi = lambda y, s: Ls.loss(Ls.theta(base, y, s))                                    # noqa: E731
    y0 = torch.zeros(Ls.n, dtype=torch.float64)
    s0 = torch.zeros((), dtype=torch.float64)
    r = Ls.Et.T @ torch.tensor(r17); r = r / r.norm(); rn = r.numpy()
    H = torch.autograd.functional.hessian(lambda y: phi(y, s0), y0).numpy()
    ev = np.linalg.eigvalsh(H)

    def nder(f, n):
        t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        out, ders = f(t), []
        for _ in range(n):
            out = torch.autograd.grad(out, t, create_graph=True)[0]
            ders.append(float(out.detach()))
        return ders

    d = nder(lambda t: phi(t * r, s0), 4)
    c_r, Q4 = 0.5 * d[2], d[3]

    def rHr(y, s):
        t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        g1 = torch.autograd.grad(phi(y + t * r, s), t, create_graph=True)[0]
        return torch.autograd.grad(g1, t, create_graph=True)[0]

    yv = y0.clone().requires_grad_(True)
    Trr = torch.autograd.grad(rHr(yv, s0), yv)[0].numpy()
    sv = s0.clone().requires_grad_(True)
    dsHrr = float(torch.autograd.grad(rHr(y0, sv), sv)[0])
    yv = y0.clone().requires_grad_(True); sv = s0.clone().requires_grad_(True)
    g = torch.autograd.grad(phi(yv, sv), yv, create_graph=True)[0]
    b = np.array([float(torch.autograd.grad(g[i], sv, retain_graph=True)[0]) for i in range(Ls.n)])
    sv = s0.clone().requires_grad_(True)
    dLds = float(torch.autograd.grad(phi(y0, sv), sv)[0])

    def Gfun(y):                                         # ṡ/ρ = −3η ∂L/∂s = −√n η â·∇_θL (n = 3 active units)
        sv_ = torch.zeros((), dtype=torch.float64, requires_grad=True)
        return -Ls.sqrt_n ** 2 * eta * torch.autograd.grad(phi(y, sv_), sv_, create_graph=True)[0]

    yv = y0.clone().requires_grad_(True)
    gradG = torch.autograd.grad(Gfun(yv), yv)[0].numpy()
    Pp = np.eye(Ls.n) - np.outer(rn, rn)
    Hp = np.linalg.pinv(Pp @ H @ Pp, rcond=1e-10)
    kap3 = Q4 / 6 - 0.5 * Trr @ Hp @ Trr
    kap11 = dsHrr - Trr @ Hp @ b
    sg = -1.0 if c_r > 0 else 1.0
    e = sg * rn
    m_e, c_e = sg * float(b @ rn), sg * c_r
    p, q = -eta * m_e, -eta * c_e
    g0 = -Ls.sqrt_n ** 2 * eta * dLds
    g1 = float(gradG @ e)
    out = {"p": p, "q": q, "f3": -eta * kap3, "f11": -eta * kap11, "g0": g0, "g1": g1,
           "abs_mc": abs(m_e * c_e), "c_half_T_rrr": c_r, "Q4": Q4, "Q4_over_6": Q4 / 6,
           "TrrHpTrr_half": float(0.5 * Trr @ Hp @ Trr), "dsH_rr": dsHrr, "TrrHp_b": float(Trr @ Hp @ b),
           "kappa3": float(kap3), "kappa11": float(kap11), "dL_ds": dLds,
           "hessian_eigs": [float(x) for x in ev], "lambda2": float(ev[1]),
           "g1_over_3p": g1 / (3 * p)}
    e17 = Ls.E @ e
    h17 = Ls.E @ (-0.5 * (Hp @ Trr))                     # centre manifold w = x²·h (σ = 0), h = −½H⁺T_rr
    return out, e17, h17, Ls


def escape_sum(thF, e17, h17, coef, A, q_thr, Ls, x0=X0_ESCAPE):
    """Frozen-s map (θ ← θ − η(I − ââᵀ)∇L, the scale-only rule at ρ = 0) from the centre-manifold point at x0 to the
    first step with ρ₂ ≥ q.  Returns the regularised sums J(x_m) on XM_GRID, the Richardson J, N, x at the crossing,
    and the quartic diagnostic (Δx − qx² − f₃x³)/x⁴ along the run."""
    P2 = Ls.P2
    ahat = Ls.ahat
    th = thF + x0 * e17 + x0 ** 2 * h17
    xs, Gs = [], []
    t0 = time.time()
    while True:
        _, g = P2.loss_grad(th)
        xs.append(float((th - thF) @ e17))
        if P2.rho2(th) >= q_thr:
            break
        Gs.append(-Ls.sqrt_n * ETA * float(ahat @ g))
        th = th - ETA * (g - ahat * (ahat @ g))
        if len(Gs) % 100_000 == 0:
            rss_guard()
    xs, Gs = np.array(xs), np.array(Gs)
    tail = np.cumsum(Gs[::-1])[::-1]
    g0, q = coef["g0"], coef["q"]

    def J_at(xm):
        k = int(np.argmax(xs >= xm))
        xk = xs[k]
        return {"x_m": xm, "x_k": float(xk), "step": k, "tail_sum": float(tail[k]),
                "J": float(tail[k] - g0 / (q * xk) - (g0 * A / q) * math.log(xk))}

    table = [J_at(xm) for xm in XM_GRID]
    Jd = {round(t["x_m"], 6): t["J"] for t in table}
    rich = lambda a, b: (b * Jd[a] - a * Jd[b]) / (b - a)                                # noqa: E731
    J0 = rich(*XM_RICH)
    J0c = rich(*XM_RICH_CHECK)
    dx = np.diff(xs)
    quart = []
    for xm in (0.01, 0.015, 0.02, 0.03):
        k = int(np.argmax(xs >= xm))
        quart.append({"x": float(xs[k]),
                      "f4_est": float((dx[k] - q * xs[k] ** 2 - coef["f3"] * xs[k] ** 3) / xs[k] ** 4),
                      "f3_est": float((dx[k] - q * xs[k] ** 2) / xs[k] ** 3),
                      "g1_est": float((Gs[k] - g0) / xs[k])})
    return {"x0": x0, "N_steps": int(len(Gs)), "x_at_crossing": float(xs[-1]), "G_at_crossing": float(Gs[-1]),
            "G_max": float(Gs.max()), "J_table": table, "J": float(J0), "J_richardson_pair": list(XM_RICH),
            "J_check": float(J0c), "J_check_pair": list(XM_RICH_CHECK),
            "J_numerical_uncertainty": float(abs(J0 - J0c)), "secs": round(time.time() - t0, 1),
            "quartic_diagnostic": quart}


def predict():
    """The prediction (no run / observation / score file is read)."""
    _setup()
    memory_gate("fold_next_order predict")
    t0 = time.time()
    fr = json.loads(FROZEN.read_text())
    s_F, q_thr = float(fr["s_F"]), float(fr["q"])
    thF, s_ref, r17, hist = fold_point(fr)
    coef, e17, h17, Ls = fold_coefficients(thF, r17)
    rss_guard()
    p, q, g0 = coef["p"], coef["q"], coef["g0"]
    A, B = inner_AB(p, q, coef["f3"], coef["f11"], g0, coef["g1"], discrete=True)
    Af, Bf = inner_AB(p, q, coef["f3"], coef["f11"], g0, coef["g1"], discrete=False)
    esc = escape_sum(thF, e17, h17, coef, A, q_thr, Ls)
    rss = rss_guard()
    KD = coefficients(p, q, g0, A, B, esc["J"], discrete=True)
    LamF = math.sqrt(coef["abs_mc"] * s_F)
    f4 = esc["quartic_diagnostic"][0]["f4_est"]
    rows = []
    for l2 in fr["ladder_log2"]:
        rho = 2.0 ** l2
        ex = expansion(rho, p, q, g0, KD["K"], KD["D"])
        lam = (g0 * rho * p) ** (1 / 3) * q ** (-2 / 3)
        rows.append({"log2rho": l2, "rho": rho,
                     "status": "pre-observed" if l2 in fr["preobserved_log2_descriptive"] else "scored",
                     "eps_F_exact": (rho * g0 / s_F) / (ETA * LamF),
                     "r_lead": ex["lead"] / s_F, "r_log_term": ex["log_term"] / s_F, "r_lin_term": ex["lin_term"] / s_F,
                     "r_pred": ex["sigma"] / s_F, "s_pred": s_F + ex["sigma"],
                     "slope_pred": ex["slope"], "slope_lead_only": 2 / 3,
                     "diag_delta1_lambda_A": lam * abs(A), "diag_delta2_f4": abs(f4) * lam ** 2 / q,
                     "diag_inner_rate_over_lambda2": q * lam / coef["lambda2"], "lambda": lam})
    pairs = []
    for a_, b_ in zip(rows[:-1], rows[1:]):
        pairs.append({"pair": [a_["log2rho"], b_["log2rho"]],
                      "two_point_slope_pred": math.log(a_["r_pred"] / b_["r_pred"]) / math.log(a_["rho"] / b_["rho"])})
    kappa = s_F * ETA * LamF / g0                       # ρ = κ ε
    out = {
        "label": "Track C: next-order fold-passage delay (math note §18). PREDICTION from frozen landscape quantities "
                 "only; no run, observation or score file read.",
        "airy": airy_constants(),
        "frozen_inputs": {"s_F": s_F, "q": q_thr, "abs_mc_frozen": fr["abs_mc"], "Lambda_F_frozen": fr["Lambda_F"],
                          "eta": fr["eta"], "frozen_json": "results/phase2a/frozen.json"},
        "fold": {"s_F_refined": s_ref, "s_F_minus_frozen": s_ref - s_F, "newton_history": hist,
                 "rho2_at_fold": float(Ls.P2.rho2(thF)), "theta_F": [float(x) for x in thF]},
        "normal_form": coef,
        "inner": {"A": A, "B": B, "A_flow": Af, "B_flow": Bf,
                  "A_parts": {"f3_over_q": coef["f3"] / q, "minus_q_discrete": -q, "minus_g1_over_g0": -coef["g1"] / g0},
                  "B_parts": {"f11_over_p": coef["f11"] / p, "minus_q_discrete": -q, "minus_g1_over_g0": -coef["g1"] / g0}},
        "escape": esc,
        "expansion_sigma": KD,
        "expansion_r_eps": {"Lambda_F_exact": LamF, "rho_over_eps": kappa,
                            "K_r": KD["K"] * kappa / s_F,
                            "D_r": (KD["D"] - KD["K"] * math.log(kappa)) * kappa / s_F,
                            "note": "r = Ω₀ε^{2/3} + K_r ε ln(1/ε) + D_r ε, ε = (ρg₀/s_F)/(ηΛ_F), Λ_F exact"},
        "neglected_order": "O(ρ^{4/3} ln²ρ) in σ: second-order inner terms (quartic f₄, x²σ, σ², g₂, products of "
                           "first-order terms), the lag of the slow hyperbolic modes (inner rate/λ₂) and the σ-dependence "
                           "of the escape; diagnostics per rate in rows (δ₁ = λ|A|, δ₂ = |f₄|λ²/q, qλ/λ₂)",
        "threshold_jitter_bound_sigma_per_rho": esc["G_max"],
        "rows": rows, "two_point": pairs,
        "secs": round(time.time() - t0, 1), "peak_rss_gb": rss / 1e9,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("inner", "expansion_sigma", "expansion_r_eps")}, indent=1))
    for r in rows:
        print(f"{r['log2rho']:8.3f} r_pred {r['r_pred']:.6f} slope {r['slope_pred']:.4f}")
    return out


# ============================================================================================ comparison (later commit)
OBSERVED = RESULTS / "phase2a" / "observed.jsonl"
OUT_COMPARE = RESULTS / "fold_next_order_compare.json"


def compare_build():
    """DERIVED AFTER THE DATA WERE SEEN: the committed prediction (fold_next_order.json, d31cedc) against 2A's
    committed observations (s_obs per rate).  Since ε ∝ ρ exactly, observed slopes are Δ ln r_obs / Δ ln ρ."""
    P = json.loads(OUT_JSON.read_text())
    s_F = P["frozen_inputs"]["s_F"]
    obs = {json.loads(l)["log2rho"]: json.loads(l) for l in OBSERVED.read_text().splitlines() if l.strip()}
    rows = []
    for r in P["rows"]:
        o = obs[r["log2rho"]]
        r_obs = o["s_obs"] / s_F - 1
        sig_obs, sig_pred = r_obs * s_F, r["r_pred"] * s_F
        rows.append({"log2rho": r["log2rho"], "status": r["status"], "r_obs": r_obs, "r_pred": r["r_pred"],
                     "r_lead": r["r_lead"], "ratio_obs_pred": r_obs / r["r_pred"], "ratio_obs_lead": r_obs / r["r_lead"],
                     "excess_steps_obs": (sig_obs - r["r_lead"] * s_F) / (r["rho"] * P["normal_form"]["g0"]),
                     "excess_steps_pred": (sig_pred - r["r_lead"] * s_F) / (r["rho"] * P["normal_form"]["g0"]),
                     "resid_sigma_over_rho43": (sig_obs - sig_pred) / r["rho"] ** (4 / 3),
                     "slope_pred_local": r["slope_pred"]})
    pairs = []
    for a, b, tp in zip(rows[:-1], rows[1:], P["two_point"]):
        so = math.log(a["r_obs"] / b["r_obs"]) / ((a["log2rho"] - b["log2rho"]) * math.log(2))
        pairs.append({"pair": [a["log2rho"], b["log2rho"]], "slope_obs": so, "slope_pred": tp["two_point_slope_pred"],
                      "diff_pred_minus_obs": tp["two_point_slope_pred"] - so})
    sc = [x for x in rows if x["status"] == "scored"]
    ratios = np.array([x["ratio_obs_pred"] for x in sc])
    lead = np.array([x["ratio_obs_lead"] for x in sc])
    q43 = np.array([x["resid_sigma_over_rho43"] for x in rows])
    summ = {"ratio_obs_pred_scored_range": [float(ratios.min()), float(ratios.max())],
            "ratio_obs_lead_scored_range": [float(lead.min()), float(lead.max())],
            "ratio_obs_pred_at_2m18": rows[-1]["ratio_obs_pred"], "ratio_obs_pred_at_2m13": rows[0]["ratio_obs_pred"],
            "slope_obs_last_pair": pairs[-1]["slope_obs"], "slope_pred_last_pair": pairs[-1]["slope_pred"],
            "slope_diff_range": [float(min(p["diff_pred_minus_obs"] for p in pairs)),
                                 float(max(p["diff_pred_minus_obs"] for p in pairs))],
            "resid_over_rho43_range": [float(q43.min()), float(q43.max())],
            "resid_over_rho43_scored_range": [float(min(x["resid_sigma_over_rho43"] for x in sc)),
                                              float(max(x["resid_sigma_over_rho43"] for x in sc))]}
    return {"label": "DERIVED AFTER THE DATA WERE SEEN: committed prediction (d31cedc) vs Phase 2A's committed "
                     "observations; no fit", "prediction_commit": "d31cedc", "rows": rows, "two_point": pairs,
            "summary": summ}


def compare():
    out = compare_build()
    OUT_COMPARE.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "predict"
    if cmd == "predict":
        predict()
    elif cmd == "compare":
        compare()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
