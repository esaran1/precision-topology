"""Numerical checks for the theory sections of the math note (§14 onward).  None of these is part of a proof; each
checks a derived formula on a system where the answer is known independently.

    python -m src.theory_checks fold        # §14: dynamic fold (slow passage through a saddle-node) -> theory_checks_fold.json

§14 checks
  F0  Omega0 = -a1, the first zero of Ai, to 30 digits (mpmath).
  F1  the Riccati normal form X' = T + X^2 started on the stable branch at T0 blows up at Omega0 (RK4, fine step).
  F2  scalar preconditioned GD map y <- y + eta*p*(m (s_F - s) - c y^2), s ramped linearly: escape scale s_esc against
      s_F + Omega0 * sdot^(2/3) * (eta^2 m' c')^(-1/3), with m', c' in the normalisation r^T P^-1 r = 1.
  F3  a 2-D gradient system with a coupled hyperbolic direction and a non-diagonal preconditioner: the same law with
      m' = r^T d_s grad L, c' = (1/2) D^3L[r, r, r] computed at the fold from their definitions.
  F4  free-training corollary: sdot = eta * v (s itself trained); the delay in s is eta-invariant at leading order.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def omega0(dps=30):
    import mpmath as mp
    mp.mp.dps = dps
    return -mp.airyaizero(1)


OMEGA0 = 2.338107410459767


def riccati_blowup(T0=-8.0, h=1e-4, cap=1e4):
    """RK4 for X' = T + X^2 from X(T0) = -sqrt(-T0); returns the T at which X first exceeds cap."""
    X, T = -math.sqrt(-T0), T0
    f = lambda T, X: T + X * X
    while X < cap:
        hh = min(h, 1e-3 / max(1.0, abs(X)))
        k1 = f(T, X); k2 = f(T + hh / 2, X + hh / 2 * k1); k3 = f(T + hh / 2, X + hh / 2 * k2); k4 = f(T + hh, X + hh * k3)
        X += hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4); T += hh
    return T + 1.0 / X                      # X ~ 1/(T_b - T) near blow-up


def fold_delay_pred(sdot, eta, mprime, cprime):
    """Predicted leading-order delay in s past the fold: Omega0 * sdot^(2/3) * (eta^2 |m'| |c'|)^(-1/3)."""
    return OMEGA0 * sdot ** (2 / 3) * (eta ** 2 * abs(mprime) * abs(cprime)) ** (-1 / 3)


def gd_escape(grad, theta0, P, eta, s0, sdot, escaped, max_steps=50_000_000):
    """theta <- theta - eta P grad(theta, s), s = s0 + sdot t; returns s at the first step with escaped(theta)."""
    th = np.array(theta0, float)
    s = s0
    for t in range(max_steps):
        th = th - eta * P @ grad(th, s)
        s = s0 + sdot * (t + 1)
        if escaped(th):
            return s
    return float("nan")


def scalar_case(eta, m, c, p, sF, sdot, start_back=None):
    """1-D: L = c y^3/3 - m (sF - s) y, P = [p].  Normalisation r^T P^-1 r = 1 -> r = sqrt(p); m' = -m r, c' = c r^3."""
    r = math.sqrt(p)
    mprime, cprime = -m * r, c * r ** 3
    pred = fold_delay_pred(sdot, eta, mprime, cprime)
    back = start_back if start_back is not None else max(40 * pred, 0.05)
    s0 = sF - back
    y0 = math.sqrt(m * (sF - s0) / c)          # on the stable branch
    grad = lambda th, s: np.array([c * th[0] ** 2 - m * (sF - s)])
    s_esc = _fast_scalar(eta * p, m, c, sF, s0, sdot, y0)
    return {"eta": eta, "m": m, "c": c, "p": p, "sdot": sdot, "delay_obs": s_esc - sF, "delay_pred": pred,
            "ratio": (s_esc - sF) / pred}


def _fast_scalar(ep, m, c, sF, s0, sdot, y0, esc=-1.0):
    """Vectorisation-free but tight loop for the scalar map (the hot path of F2/F4)."""
    y, s, t = y0, s0, 0
    while y > esc:
        y = y + ep * (m * (sF - s) - c * y * y)
        t += 1
        s = s0 + sdot * t
        if t > 200_000_000:
            return float("nan")
    return s


def two_d_case(eta, sdot, sF=1.0, cy=1.0, m=1.0, k=3.0, g=0.7, e=0.4):
    """L(y, w; s) = cy y^3/3 - m (sF - s) y + (k/2)(w - g y)^2 + e y (w - g y)^2  (fold at y = w = 0, s = sF).
    Preconditioner P non-diagonal.  m', c' from the definitions at the fold, r normalised by r^T P^-1 r = 1."""
    P = np.array([[1.0, 0.3], [0.3, 0.6]])

    def grad(th, s):
        y, w = th
        u = w - g * y
        return np.array([cy * y * y - m * (sF - s) - k * g * u + e * u * u - 2 * e * g * y * u,
                         k * u + 2 * e * y * u])
    H = np.array([[k * g * g, -k * g], [-k * g, k]])              # Hessian at the fold (y = w = 0)
    r = np.array([1.0, g])                                          # H r = 0
    r = r / math.sqrt(r @ np.linalg.solve(P, r))
    hstep = 1e-4                                                    # m' = r . d_s grad ; c' = (1/2) D^3 L[r, r, r]
    mprime = float(r @ (grad(np.zeros(2), sF + hstep) - grad(np.zeros(2), sF - hstep)) / (2 * hstep))
    d2 = lambda z: (grad(z + hstep * r, sF) - 2 * grad(z, sF) + grad(z - hstep * r, sF)) / hstep ** 2
    cprime = float(0.5 * r @ d2(np.zeros(2)))
    pred = fold_delay_pred(sdot, eta, mprime, cprime)
    back = max(40 * pred, 0.05)
    s0 = sF - back
    y0 = math.sqrt(m * (sF - s0) / cy)
    # relax onto the branch at fixed s0 first
    th = np.array([y0, g * y0])
    for _ in range(20000):
        th = th - eta * P @ grad(th, s0)
    s_esc = gd_escape(grad, th, P, eta, s0, sdot, lambda th: th[0] < -1.0)
    return {"eta": eta, "sdot": sdot, "H_null_check": float(np.abs(H @ (r / np.linalg.norm(r))).max()),
            "m_prime": mprime, "c_prime": cprime, "delay_obs": s_esc - sF, "delay_pred": pred,
            "ratio": (s_esc - sF) / pred}


def lambda_sq_slope(sF=1.0, cy=1.0, m=1.0, k=3.0, g=0.7, e=0.4):
    """F5: along the branch of the 2-D system, lambda_min(P^1/2 H P^1/2)^2 against s near the fold; the slope
    -d(lambda^2)/ds at the fold should equal 4 |m' c'| (the measurable form of the fold constant)."""
    P = np.array([[1.0, 0.3], [0.3, 0.6]])
    w_, V = np.linalg.eigh(P); Ph = V @ np.diag(np.sqrt(w_)) @ V.T

    def grad(th, s):
        y, w = th
        u = w - g * y
        return np.array([cy * y * y - m * (sF - s) - k * g * u + e * u * u - 2 * e * g * y * u, k * u + 2 * e * y * u])

    def hess(th, s, h=1e-6):
        return np.array([(grad(th + h * ei, s) - grad(th - h * ei, s)) / (2 * h) for ei in np.eye(2)]).T
    rows = []
    for d in (1e-2, 5e-3, 2.5e-3, 1.25e-3):
        s = sF - d
        th = np.array([math.sqrt(m * d / cy), g * math.sqrt(m * d / cy)])
        for _ in range(100):                                   # Newton onto the branch
            th = th - np.linalg.solve(hess(th, s), grad(th, s))
        Hs = hess(th, s); Hs = 0.5 * (Hs + Hs.T)
        lam = float(np.linalg.eigvalsh(Ph @ Hs @ Ph)[0])
        rows.append({"sF_minus_s": d, "lambda_min": lam, "lambda_sq_over_d": lam ** 2 / d})
    ref = two_d_case(0.1, 1e-5)
    return {"rows": rows, "four_abs_mc": 4 * abs(ref["m_prime"] * ref["c_prime"])}


def fold_checks():
    out = {"Omega0_mpmath": str(omega0()), "Omega0_used": OMEGA0}
    out["F1_riccati_blowup"] = {str(T0): riccati_blowup(T0) for T0 in (-4.0, -6.0, -8.0)}
    out["F2_scalar"] = [scalar_case(0.3, 1.0, 1.0, 1.0, 1.0, sd) for sd in (1e-3, 1e-4, 1e-5, 1e-6)] + \
                       [scalar_case(0.1, 2.0, 0.5, 0.4, 3.0, sd) for sd in (1e-4, 1e-5, 1e-6)]
    out["F3_two_d"] = [two_d_case(0.1, sd) for sd in (1e-4, 1e-5, 1e-6)]
    # F4: free training, sdot = eta * v; delay in s should not depend on eta
    v = 1e-4
    out["F4_free_training"] = [dict(scalar_case(eta, 1.0, 1.0, 1.0, 1.0, eta * v), v=v) for eta in (0.2, 0.1, 0.05)]
    out["F5_lambda_sq_slope"] = lambda_sq_slope()
    return out


if __name__ == "__main__":
    os.nice(15)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fold"
    if cmd == "fold":
        res = fold_checks()
        (RESULTS / "theory_checks_fold.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res, indent=1, default=float))
