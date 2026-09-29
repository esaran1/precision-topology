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


def bessel_form_at_omega0(dps=30):
    """F0b: [KS01] Remark 2.4 defines Omega0 as the smallest positive zero of J_{-1/3}(2z^{3/2}/3) + J_{1/3}(2z^{3/2}/3);
    returns its value at z = -a1 and the smallest positive zero found by root-finding."""
    import mpmath as mp
    mp.mp.dps = dps
    f = lambda z: mp.besselj(-mp.mpf(1) / 3, 2 * z ** 1.5 / 3) + mp.besselj(mp.mpf(1) / 3, 2 * z ** 1.5 / 3)
    z0 = omega0(dps)
    grid = [mp.mpf(k) / 100 for k in range(1, 400)]
    first = next(g for g, h in zip(grid, grid[1:]) if f(g) * f(h) < 0)
    root = mp.findroot(f, (first, first + mp.mpf(1) / 100), solver="bisect")
    return {"value_at_minus_a1": str(f(z0)), "smallest_positive_zero": str(root)}


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
    out["F0b_bessel_form"] = bessel_form_at_omega0()
    out["F1_riccati_blowup"] = {str(T0): riccati_blowup(T0) for T0 in (-4.0, -6.0, -8.0)}
    out["F2_scalar"] = [scalar_case(0.3, 1.0, 1.0, 1.0, 1.0, sd) for sd in (1e-3, 1e-4, 1e-5, 1e-6)] + \
                       [scalar_case(0.1, 2.0, 0.5, 0.4, 3.0, sd) for sd in (1e-4, 1e-5, 1e-6)]
    out["F3_two_d"] = [two_d_case(0.1, sd) for sd in (1e-4, 1e-5, 1e-6)]
    # F4: free training, sdot = eta * v; delay in s should not depend on eta
    v = 1e-4
    out["F4_free_training"] = [dict(scalar_case(eta, 1.0, 1.0, 1.0, 1.0, eta * v), v=v) for eta in (0.2, 0.1, 0.05)]
    out["F5_lambda_sq_slope"] = lambda_sq_slope()
    return out


# ------------------------------------------------------------------ §15: the lag law as a theorem (Theorem L)

def lag_toy(k3=0.3, rho=0.4):
    """Toy branch in z-coordinates (P = I after z = P^{-1/2} theta): L = 1/2 d^T A(s) d + (k3/3) d1^3, d = z - z*(s),
    z*(s) = (0.5 sin s, 0.3 s), A(s) = diag(1 + 0.2 s, 3).  Placement gap G(z) = -z1 + z2 + 0.1 z2^2 - g0 with the
    branch switch at s* = 2.  Returns the objects and the Theorem L constants on the tube |d| <= rho, s in [1, 3]."""
    zs = lambda s: np.array([0.5 * math.sin(s), 0.3 * s])
    zs1 = lambda s: np.array([0.5 * math.cos(s), 0.3])
    A = lambda s: np.diag([1 + 0.2 * s, 3.0])
    grad = lambda z, s: A(s) @ (z - zs(s)) + np.array([k3 * (z[0] - zs(s)[0]) ** 2, 0.0])
    sstar = 2.0
    g0 = -0.5 * math.sin(sstar) + 0.3 * sstar + 0.1 * (0.3 * sstar) ** 2
    G = lambda z: -z[0] + z[1] + 0.1 * z[1] ** 2 - g0
    dG = lambda z: np.array([-1.0, 1 + 0.2 * z[1]])
    J = (1.0, 3.0)
    const = {
        "lam": min(1 + 0.2 * J[0] - 2 * k3 * rho, 3.0),          # lower eigenvalue bound on the tube
        "Lam": max(1 + 0.2 * J[1] + 2 * k3 * rho, 3.0),          # upper
        "M": 2 * k3,                                               # Lipschitz constant of the Hessian in z
        "D1": math.sqrt(0.25 + 0.09), "D2": 0.5,                  # sup |z*'|, sup |z*''|
        "H1": 0.2,                                                 # sup |d/ds H(z*(s), s)|
        "g1": math.sqrt(1 + (1 + 0.2 * (0.3 * J[1] + rho)) ** 2),  # sup |grad G| on the tube
        "g2": 0.2,                                                 # sup |Hess G|
        "rho": rho, "J": J, "sstar": sstar}
    gp = lambda s: -0.5 * math.cos(s) + 0.3 + 0.2 * 0.3 * (0.3 * s)       # g'(s)
    const["gamma"] = min(gp(s) for s in np.linspace(1.5, 2.5, 1001))       # g' >= gamma on the crossing window
    const["g2pp"] = const["g2"] * const["D1"] ** 2 + const["g1"] * const["D2"]
    return {"zs": zs, "zs1": zs1, "A": A, "grad": grad, "G": G, "dG": dG, "gp": gp, "const": const}


def theorem_L_bounds(c, sdot, eta, Ks=0.0, w=0.5):
    """Explicit constants of Theorems L1-L2 (§15.2-15.3) for step eta, maximal rate sdot (per step) and rate-variation
    constant Ks (|sdot_{t+1} - sdot_t| <= Ks sdot_t^2).  w = half-width of the crossing window J' = [s* - w, s* + w]."""
    lam, M, D1, D2, H1 = c["lam"], c["M"], c["D1"], c["D2"], c["H1"]
    q = sdot / (eta * lam)
    D1b = D1 + D2 * sdot / 2
    S = q * D1b
    Ce = 2 * M * D1b ** 2 / lam + H1 * D1b / lam + D2 + D1 * Ks
    E = Ce * q ** 2
    ok = (eta * c["Lam"] <= 1) and (E <= S) and (2 * S <= c["rho"])
    g1, g2, gam, g2pp = c["g1"], c["g2"], c["gamma"], c["g2pp"]
    S1 = q * (D2 + H1 * D1b / lam)
    hprime = g2 * D1 * S + g1 * S1
    ok = ok and hprime <= gam / 2
    Cl = (2 * g2pp * g1 ** 2 * S ** 2 / gam ** 2 + hprime * 2 * g1 * S / gam) / gam
    dv = Ks * sdot * (2 * w + sdot)
    B = g1 * E + g2 * (S + E) ** 2 / 2 + g1 * D1 * dv / (eta * lam)
    ok = ok and (w > 2 * g1 * S / gam + 2 * B / gam + sdot)
    return {"q": q, "S": S, "E": E, "C_ell": Cl, "B": B, "lag_error_bound": Cl + 2 * B / gam + sdot,
            "conditions_hold": bool(ok)}


def lag_toy_run(sdot, eta, k3=0.3, ramp="linear"):
    T = lag_toy(k3)
    c = T["const"]
    s0 = 1.2
    s, z = s0, T["zs"](s0)
    # start on the branch; ramp; record max |delta - delta_sl| after the transient and the crossing
    worst, t, prevG = 0.0, 0, T["G"](z)
    Ks = 0.0
    while s < 2.8:
        sd = sdot if ramp == "linear" else sdot * s / c["sstar"]
        if ramp != "linear":
            Ks = 1 / 1.2
        z = z - eta * T["grad"](z, s)
        s_new = s + sd
        t += 1
        # slaved displacement at the new s: -(eta A)^{-1} (z*(s + sdot) - z*(s))
        Delta = T["zs"](s_new + sd) - T["zs"](s_new)
        dsl = -np.linalg.solve(eta * T["A"](s_new), Delta)
        s = s_new
        d = z - T["zs"](s)
        if t > 50 / (eta * c["lam"]):
            worst = max(worst, float(np.linalg.norm(d - dsl)))
        Gz = T["G"](z)
        if prevG < 0 <= Gz:
            s_c = s
            s_ci = s - sd * Gz / (Gz - prevG)          # linear interpolation of G between the two steps
            break
        prevG = Gz
    sstar = c["sstar"]
    H = T["A"](sstar); z1 = T["zs1"](sstar); gG = T["dG"](T["zs"](sstar))
    lam_min = float(np.linalg.eigvalsh(H)[0])
    kappa = lam_min * float(gG @ np.linalg.solve(H, z1)) / float(gG @ z1)
    sdot_c = sdot if ramp == "linear" else sdot * sstar / sstar
    chi = sdot_c / (sstar * eta * lam_min)
    Delta = T["zs"](sstar + sdot_c) - T["zs"](sstar)
    ell = float(gG @ np.linalg.solve(eta * H, Delta)) / T["gp"](sstar)
    bnd = theorem_L_bounds(c, max(sdot, sdot * 2.8 / sstar) if ramp != "linear" else sdot, eta, Ks)
    return {"sdot": sdot, "eta": eta, "ramp": ramp, "chi": chi, "kappa": kappa, "r_obs": (s_c - sstar) / sstar,
            "kappa_chi": kappa * chi, "ell_over_sstar": ell / sstar,
            "lag_minus_ell": s_c - sstar - ell, "r_interp": (s_ci - sstar) / sstar,
            "interp_lag_minus_ell": s_ci - sstar - ell, "lag_error_bound": bnd["lag_error_bound"],
            "bound_holds": bool(abs(s_c - sstar - ell) <= bnd["lag_error_bound"]),
            "max_dev_from_slaved": worst, "E_bound": bnd["E"], "dev_bound_holds": bool(worst <= bnd["E"] * (1 + 1e-9)),
            "conditions_hold": bnd["conditions_hold"]}


def lag_checks():
    out = {"toy_constants": {k: v for k, v in lag_toy()["const"].items()}}
    rows = []
    for eta in (0.25, 0.1):
        for sdot in (1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5):
            rows.append(lag_toy_run(sdot, eta))
    for sdot in (5e-4, 1.25e-4):
        rows.append(lag_toy_run(sdot, 0.1, ramp="exponential"))
    out["L1_toy_ramps"] = rows
    # L2: second-order scaling of (r - kappa chi) at fixed eta
    lin = [r for r in rows if r["ramp"] == "linear" and r["eta"] == 0.1]
    out["L2_residual_over_chi2"] = [{"chi": r["chi"], "(r-ell/s*)/chi^2": (r["r_obs"] - r["ell_over_sstar"]) / r["chi"] ** 2,
                                     "(r_interp-ell/s*)/chi^2": (r["r_interp"] - r["ell_over_sstar"]) / r["chi"] ** 2}
                                    for r in lin]
    # L3: free-training corollary: s trained, sdot = eta * v(s) with v the negative s-gradient of a driving term
    out["L3_free_training"] = [free_training_toy(eta) for eta in (0.1, 0.05, 0.025)]
    return out


def free_training_toy(eta, v0=2e-3):
    """s itself trained: s <- s + eta * v0 * (1 + 0.1 (s - 2)) (a driving force independent of eta); chi at the crossing
    and r_obs should be eta-invariant at leading order (Corollary L)."""
    T = lag_toy()
    c = T["const"]
    s = 1.2
    z = T["zs"](s)
    prevG = T["G"](z)
    while s < 2.8:
        v = v0 * (1 + 0.1 * (s - 2))
        z = z - eta * T["grad"](z, s)
        s = s + eta * v
        Gz = T["G"](z)
        if prevG < 0 <= Gz:
            break
        prevG = Gz
    sstar = c["sstar"]
    lam_min = float(np.linalg.eigvalsh(T["A"](sstar))[0])
    chi = (eta * v0) / (sstar * eta * lam_min)
    return {"eta": eta, "chi_at_switch": chi, "r_obs": (s - sstar) / sstar}


if __name__ == "__main__":
    os.nice(15)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fold"
    if cmd == "fold":
        res = fold_checks()
        (RESULTS / "theory_checks_fold.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res, indent=1, default=float))
    elif cmd == "lag":
        res = lag_checks()
        (RESULTS / "theory_checks_lag.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res, indent=1, default=float))
