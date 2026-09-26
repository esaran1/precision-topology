"""Track 2B (POST HOC, existing runs; nothing here is registered and no verdict changes): does the width-1 lag law
r = κ·χ (src/lag_law.py, Track 1A) hold at width 2 for the asymmetric-window runs (a = 1.30, Δ = 0.4)?

Arms: T2-3 (full speed, φ₂ = 1; χ only is asked for, the comparison is descriptive), and the slowed arms T2-3b
(φ₂ = 0.009306), T2-3c and T2-3d (φ₂ = 0.01778).  T2-3 stays FAIL; T2-3b and T2-3c stay UNRESOLVED; T2-3d stays FAIL.

Per crossing run (definitions fixed before any comparison was computed):
  replay     the run is replayed deterministically to its recorded crossing step (its ‖v‖₁ there must reproduce the recorded
             s_cross to 1e−12).  Recorded: the crossing state q_c, Adam's bias-corrected v̂ at the crossing, ṡ = (s_c − s_{c−w})/w
             with w = min(100, step − 1), and the first-passage state at each scale of GRID (for Track 2C's calibration).
  branch     the run's width-2 analogue of the width-1 hidden coordinates: z = (α₁, β₁, α₂, β₂, b), all trained at lr η = 0.01;
             the output weights v = (v₁, v₂) are the slow variables (in the slowed arms both move at φ₂·η).  The tracked
             branch is z*(v) = the stationary point of the run's training loss in z at fixed v (damped Newton from the
             crossing state, H PD required), followed along the ray v = s·ṽ_c (ṽ_c = v_c/‖v_c‖₁, the crossing direction) by
             Newton continuation in log s (steps ≤ 2%, halved on a jump or a lost minimum).  Its placement gap
             g(s) = G₊(φ at z*(s), ṽ_c) (exact extrema, asymmetric windows) is followed until it changes sign; the root s*
             is bisected to relative 1e−10.  Search range s ∈ [s_c/10, 10·s_c].
  κ₂         as in 1A at (z*(s*), s*): H = ∇²_z L, θ*′ = −H⁻¹∂ₛ∇_z L (along the ray), ∇G by central differences (h = 1e−6),
             P = diag(1/(√v̂ + 1e−8)) from the run's own v̂ at the crossing (z coordinates), κ₂ = λ_min·[∇G·(PH)⁻¹θ*′]/[∇G·θ*′],
             λ_min = λ_min(P^{1/2}HP^{1/2}).  The run's actual coordinates are used (no canonicalisation), so its 2π winding
             enters θ*′ through b exactly as κ_k does at width 1.
  χ          (ṡ/s*)/(η·λ_min).     r_obs = s_c/s* − 1,  r_pred = κ₂·χ (no fit).
Scoring (per arm, as 1A's compare): median r_obs vs median r_pred, obs/pred, |median obs − median pred| ≤ max(0.01,
0.25|median pred|), and per run the fraction with |r_obs − r_pred| ≤ max(0.01, 0.25|r_pred|).  Also split by χ against the
largest width-1 χ at which the law held (lag_law predictions: per-run maximum over the 36 arms that held).
Diagnostics: direction drift of v at the crossing, relative to its scale growth; unit shares; ‖z_c − z*(v_c)‖∞ (dist_c:
how far the crossing state is from the branch it is assigned to); λ_min at the crossing branch and the number of relaxation
times elapsed before the crossing, step·η·λ_min,c; χ_c = (ṡ/s_c)/(ηλ_min,c) (the existing 'ratio' definition, s at crossing).
PRIMARY branch = along the run's own v-path (own_switch_path).  Added after the ray version (own_switch) had been run on
three T2-3c runs, one of which showed direction drift 0.88 of the scale growth; the ray version is kept as secondary.

    python -m src.width2_lag run [arm ...]      # one process, nice 15; resumable (parts CSV per arm)
    python -m src.width2_lag score
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "width2_lag"
STATES = OUT / "states"                                     # first-passage states (pickles; inputs to Track 2C)
LR, EPS, BETA2 = 1e-2, 1e-8, 0.999
GRID = tuple(10 ** (-2.5 + k / 8) for k in range(26))       # 0.00316 .. 5.62 (2C's candidate rule scales)
STEP_MAX, STEP_MIN = 0.02, 1e-4
RANGE = 10.0
GTOL = 1e-9
IDX = [0, 1, 3, 4, 6]                                       # z = (α₁, β₁, α₂, β₂, b) in width2_train's layout


def arms():
    k = json.loads((RESULTS / "asym_frozen.json").read_text())["k"]
    return {"T2-3": (RESULTS / "asym_parts" / "train.csv", 1.0, k),
            "T2-3b": (RESULTS / "asym_t23b" / "train.csv", json.loads((RESULTS / "asym_t23b_frozen.json").read_text())["phi2"], k),
            "T2-3c": (RESULTS / "asym_t23c" / "train.csv", json.loads((RESULTS / "asym_t23c_frozen.json").read_text())["phi2"], k),
            "T2-3d": (RESULTS / "asym_t23d" / "train.csv", 0.01778279410038923, k)}


# ------------------------------------------------------------------------------------------ replay
def replay(seed, phi, k, step, grid=GRID):
    """The arm's training loop (asym_register.train_one for φ₂ = 1; asym_t23c._step otherwise), to `step`.  Returns the
    crossing state, v̂, ṡ and first-passage states at the grid scales (a scale is passed at the first step t ≥ 1 with s
    between s(t−1) and s(t) inclusive, or s(0) = that scale)."""
    import torch
    from .asym_register import _act, _setup, training_set
    from .width2_train import init_params, logits
    torch.set_num_threads(1)
    _setup(); act = _act()
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); q0[2] *= k; q0[5] *= k
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    s_prev = float(q0[2].abs() + q0[5].abs())
    s_hist = {0: s_prev}; v_hist = {0: q0[[2, 5]].numpy().copy()}
    V = [q0[[2, 5]].numpy().copy()]
    passage, s_min, s_init = {}, s_prev, s_prev
    for g in grid:
        if g == s_prev:
            passage[g] = {"t": 0, "q": q0.numpy().copy(), "adam": None}
    for t in range(1, int(step) + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        vb = q.detach()[[2, 5]].clone()
        opt.step()
        if phi != 1.0:
            with torch.no_grad():
                q[[2, 5]] = vb + phi * (q[[2, 5]] - vb)
        qd = q.detach()
        s = float(qd[2].abs() + qd[5].abs())
        s_hist[t] = s; v_hist[t] = qd[[2, 5]].numpy().copy(); V.append(v_hist[t])
        s_hist.pop(t - 101, None); v_hist.pop(t - 101, None)
        if t < step:
            s_min = min(s_min, s)
        lo, hi = min(s_prev, s), max(s_prev, s)
        for g in grid:
            if g not in passage and lo <= g <= hi and t < step:
                passage[g] = {"t": t, "q": qd.numpy().copy(),
                              "adam": {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in opt.state[q].items()}}
        s_prev = s
    st = opt.state[q]
    vhat = st["exp_avg_sq"].numpy() / (1 - BETA2 ** int(st["step"]))
    w = min(100, int(step) - 1)
    sdot = (s_hist[int(step)] - s_hist[int(step) - w]) / w if w >= 1 else float("nan")
    vdot = (v_hist[int(step)] - v_hist[int(step) - w]) / w if w >= 1 else np.full(2, np.nan)
    return {"q": q.detach().numpy().copy(), "vhat": vhat, "sdot": sdot, "vdot": vdot, "s_init": s_init, "s_min_pre": s_min,
            "passage": passage, "x": x, "y": y, "V": np.array(V)}


# ------------------------------------------------------------------------------------------ the branch at fixed v
def _u(act, t):
    import torch
    return t + act.a * torch.sin(t) if act.name == "fa" else torch.tanh(t)


def fast_loss(z, v, X, Y, act):
    import torch
    return torch.nn.functional.binary_cross_entropy_with_logits(
        v[0] * _u(act, z[0] * X + z[1]) + v[1] * _u(act, z[2] * X + z[3]) + z[4], Y)


def branch_point(z0, v, x, y, act, iters=200, gtol=GTOL):
    """Damped Newton (Levenberg–Marquardt) on the training loss in z at fixed v.  Returns (z, max|∇|, H, converged)."""
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    vt = torch.tensor(np.asarray(v, float), dtype=torch.float64)
    f = lambda z: fast_loss(z, vt, X, Y, act)
    z = torch.tensor(np.asarray(z0, float), dtype=torch.float64)
    mu, f0 = 1e-3, float(f(z))
    for _ in range(iters):
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(f(zz), zz)[0]
        if float(g.abs().max()) < 1e-13:
            break
        H = torch.autograd.functional.hessian(f, z)
        sc = max(1.0, float(H.diag().abs().max()))
        ok = False
        for _ in range(30):
            d = torch.linalg.solve(H + mu * sc * torch.eye(5, dtype=torch.float64), -g)
            f1 = float(f(z + d))
            if f1 <= f0:
                z, f0, ok, mu = z + d, f1, True, max(mu / 3, 1e-14)
                break
            mu *= 10
        if not ok:
            break
    zz = z.clone().requires_grad_(True)
    gm = float(torch.autograd.grad(f(zz), zz)[0].abs().max())
    H = torch.autograd.functional.hessian(f, z).numpy()
    return z.numpy(), gm, H, gm <= gtol


def gap(z, vt, act):
    """Midpoint of the exact G₊ enclosure of φ = ṽ₁u(α₁x+β₁) + ṽ₂u(α₂x+β₂) on the current windows."""
    from .width2_geometry import gaps
    g = gaps(np.array([z[0], z[1], z[2], z[3]]), np.asarray(vt, float), act)["G+"]
    return 0.5 * (g[0] + g[1])


def grad_gap(z, vt, act, h=1e-6):
    """∇_z G by central differences (b does not enter G), and the largest relative one-sided disagreement."""
    out, asym = np.zeros(5), 0.0
    g0 = gap(z, vt, act)
    for i in range(4):
        e = np.zeros(5); e[i] = h
        gp, gm = gap(z + e, vt, act), gap(z - e, vt, act)
        out[i] = (gp - gm) / (2 * h)
        fw, bw = (gp - g0) / h, (g0 - gm) / h
        asym = max(asym, abs(fw - bw) / max(abs(out[i]), 1e-12)) if abs(out[i]) > 1e-8 else asym
    return out, asym


def _is_min(H):
    return bool(np.linalg.eigvalsh(0.5 * (H + H.T)).min() > 0)


def _dz_ds(z, s, vt, x, y, act, H):
    """Branch tangent dz*/ds = −H⁻¹∂ₛ∇_z L at v = s·ṽ (H given)."""
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    vtt = torch.tensor(np.asarray(vt, float), dtype=torch.float64)
    st = torch.tensor(s, dtype=torch.float64, requires_grad=True)
    zz = torch.tensor(np.asarray(z, float), dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(fast_loss(zz, st * vtt, X, Y, act), zz, create_graph=True)[0]
    dgds = np.array([float(torch.autograd.grad(g[i], st, retain_graph=True)[0]) for i in range(5)])
    return -np.linalg.solve(H, dgds)


def corrector_ok(z, z_pred, z_prev):
    """Predictor–corrector acceptance: the Newton correction is at most half the predictor step (or 1e−3).  On a
    smooth branch the correction is O(h²) against an O(h) step; a jump to another branch fails at every step size."""
    return float(np.abs(z - z_pred).max()) <= max(1e-3, 0.5 * float(np.abs(z_pred - z_prev).max()))


def own_switch(z_c, v_c, x, y, act, step_max=STEP_MAX, rng=RANGE):
    """Newton continuation of the fixed-v branch along v = s·ṽ_c from the crossing until its gap changes sign.
    Returns a dict with status in {'ok', 'no_branch_at_crossing', 'branch_lost', 'jump', 'no_switch_in_range'}."""
    s_c = float(np.abs(v_c).sum()); vt = np.asarray(v_c, float) / s_c
    z, gm, H, conv = branch_point(z_c, v_c, x, y, act)
    base = {"z_branch_c": z, "dist_c": float(np.abs(z - np.asarray(z_c)).max()), "branch_grad_c": gm,
            "branch_pd_c": _is_min(H) if conv else False}
    if not (conv and _is_min(H)):
        return {**base, "status": "no_branch_at_crossing"}
    g0 = gap(z, vt, act)
    base["g_branch_c"] = g0
    sgn = -1.0 if g0 > 0 else 1.0
    s_prev, z_prev, g_prev, H_prev, h, n = s_c, z, g0, H, step_max, 0
    while True:
        s = s_prev * math.exp(sgn * h)
        if s < s_c / rng or s > s_c * rng:
            return {**base, "status": "no_switch_in_range", "n_steps": n, "s_end": s_prev}
        z_pred = z_prev + _dz_ds(z_prev, s_prev, vt, x, y, act, H_prev) * (s - s_prev)
        z, gm, H, conv = branch_point(z_pred, s * vt, x, y, act)
        good = conv and _is_min(H) and corrector_ok(z, z_pred, z_prev)
        if not good:
            if h / 2 < STEP_MIN:
                why = "branch_lost" if (not conv or not _is_min(H)) else "jump"
                return {**base, "status": why, "n_steps": n, "s_end": s_prev}
            h /= 2
            continue
        n += 1
        g = gap(z, vt, act)
        if (g > 0) != (g_prev > 0):
            lo, hi, zl, zh = (s, s_prev, z, z_prev) if sgn < 0 else (s_prev, s, z_prev, z)
            gl = g if sgn < 0 else g_prev
            for _ in range(80):
                if hi / lo - 1 < 1e-10:
                    break
                mid = math.sqrt(lo * hi)
                zm, gmm, Hm, cm = branch_point(zl + (zh - zl) * (mid - lo) / (hi - lo), mid * vt, x, y, act)
                gmid = gap(zm, vt, act)
                if (gmid > 0) == (gl > 0):
                    lo, zl, gl = mid, zm, gmid
                else:
                    hi, zh = mid, zm
            s_star = math.sqrt(lo * hi)
            zs, gms, Hs, cs = branch_point(zl, s_star * vt, x, y, act)
            return {**base, "status": "ok", "n_steps": n, "s_star": s_star, "z_star": zs, "grad_star": gms,
                    "pd_star": _is_min(Hs), "H_star": Hs, "vt": vt}
        s_prev, z_prev, g_prev, H_prev = s, z, g, H
        h = min(step_max, h * 2)


def v_at(V, t, t_c, vdot):
    """The run's output weights at (real) time t: linear interpolation of the recorded path for t ≤ t_c, linear
    extrapolation with the crossing velocity beyond it."""
    if t >= t_c:
        return V[t_c] + vdot * (t - t_c)
    t = max(t, 0.0)
    i = int(math.floor(t)); f = t - i
    return V[i] * (1 - f) + V[min(i + 1, t_c)] * f


def jac_v(z, v, x, y, act):
    """J = ∂_v∇_z L (5 × 2)."""
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    vv = torch.tensor(np.asarray(v, float), dtype=torch.float64, requires_grad=True)
    zz = torch.tensor(np.asarray(z, float), dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(fast_loss(zz, vv, X, Y, act), zz, create_graph=True)[0]
    return np.array([torch.autograd.grad(g[i], vv, retain_graph=True)[0].numpy() for i in range(5)])


def own_switch_path(z_c, V, t_c, vdot, x, y, act, h_max=STEP_MAX, rng=RANGE):
    """PRIMARY branch definition: Newton continuation of z*(v(t)) along the run's own recorded output-weight path v(t)
    (backward from the crossing if the branch is placed there; forward along the crossing velocity if it is not), with
    the gap G(z*(t), ṽ(t)) followed until it changes sign.  Steps: ‖Δv‖₁ ≤ h·‖v‖₁ (h ≤ 2%), halved on a failed
    predictor–corrector check.  Statuses as own_switch, plus 'placed_back_to_init' (placed along the whole path)."""
    V = np.asarray(V, float); t_c = int(t_c)
    v_c = V[t_c]; s_c = float(np.abs(v_c).sum())
    z, gm, H, conv = branch_point(z_c, v_c, x, y, act)
    base = {"dist_c": float(np.abs(z - np.asarray(z_c)).max()), "branch_grad_c": gm,
            "branch_pd_c": _is_min(H) if conv else False, "H_c": H}
    if not (conv and _is_min(H)):
        return {**base, "status": "no_branch_at_crossing"}
    vn = lambda v: np.asarray(v, float) / np.abs(v).sum()
    g0 = gap(z, vn(v_c), act)
    base["g_branch_c"] = g0
    sgn = -1.0 if g0 > 0 else 1.0
    speed_f = float(np.abs(vdot).sum())
    t_end = 0.0 if sgn < 0 else t_c + max(10.0 * t_c, 1000.0)
    t_prev, z_prev, g_prev, h, n = float(t_c), z, g0, h_max, 0
    while True:
        v_prev = v_at(V, t_prev, t_c, vdot); s_prev = float(np.abs(v_prev).sum())
        if sgn < 0:
            if t_prev <= 0:
                return {**base, "status": "placed_back_to_init", "n_steps": n}
            j = max(int(math.ceil(t_prev)) - 1, 0)
            sp = max(float(np.abs(V[min(j + 1, t_c)] - V[j]).sum()), 1e-300)
        else:
            sp = max(speed_f, 1e-300)
        dt = min(max(h * s_prev / sp, 1e-6), 1e9)
        t = max(t_prev - dt, 0.0) if sgn < 0 else t_prev + dt
        v = v_at(V, t, t_c, vdot); s = float(np.abs(v).sum())
        if sgn > 0 and (t > t_end or s > rng * s_c or s < s_c / rng):
            return {**base, "status": "no_switch_in_range", "n_steps": n}
        z_pred = z_prev - np.linalg.solve(H, jac_v(z_prev, v_prev, x, y, act) @ (v - v_prev))
        zn, gm, Hn, conv = branch_point(z_pred, v, x, y, act)
        if not (conv and _is_min(Hn) and corrector_ok(zn, z_pred, z_prev)):
            if h / 2 < STEP_MIN:
                why = "branch_lost" if (not conv or not _is_min(Hn)) else "jump"
                return {**base, "status": why, "n_steps": n, "t_end": t_prev}
            h /= 2
            continue
        n += 1
        g = gap(zn, vn(v), act)
        if (g > 0) != (g_prev > 0):
            ta, tb, za, gA = t_prev, t, z_prev, g_prev          # ta on the crossing side, tb beyond the switch
            for _ in range(100):
                if abs(tb - ta) < 1e-7:
                    break
                tm = 0.5 * (ta + tb)
                zm, _, _, _ = branch_point(za, v_at(V, tm, t_c, vdot), x, y, act)
                gmid = gap(zm, vn(v_at(V, tm, t_c, vdot)), act)
                if (gmid > 0) == (gA > 0):
                    ta, za, gA = tm, zm, gmid
                else:
                    tb = tm
            t_star = 0.5 * (ta + tb)
            v_star = v_at(V, t_star, t_c, vdot)
            zs, gms, Hs, _ = branch_point(za, v_star, x, y, act)
            return {**base, "status": "ok", "n_steps": n, "t_star": t_star, "v_star": v_star,
                    "s_star": float(np.abs(v_star).sum()), "z_star": zs, "grad_star": gms, "pd_star": _is_min(Hs), "H_star": Hs}
        t_prev, z_prev, g_prev, H = t, zn, g, Hn
        h = min(h_max, h * 2)


def path_kappa(zs, v_star, vdot, sdot, p, x, y, act, eps=1e-6):
    """κ₂ along the path: θ̇* = −H⁻¹J·v̇, ġ = ∇_zG·θ̇* + D_ṽG[ṽ̇], ṽ̇ = (v̇ − ṽṡ)/s*;  κ₂ = λ_min[∇_zG·(PH)⁻¹θ̇*]/ġ.
    With ṽ̇ = 0 and v̇ ∥ ṽ this is exactly 1A's κ (θ̇* = θ*′ṡ)."""
    v_star = np.asarray(v_star, float); s_star = float(np.abs(v_star).sum()); vt = v_star / s_star
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    H = torch.autograd.functional.hessian(lambda q: fast_loss(q, torch.tensor(v_star), X, Y, act),
                                          torch.tensor(np.asarray(zs, float))).numpy()
    th_dot = -np.linalg.solve(H, jac_v(zs, v_star, x, y, act) @ np.asarray(vdot, float))
    vt_dot = (np.asarray(vdot, float) - vt * sdot) / s_star
    dG, asym = grad_gap(zs, vt, act)
    nv = float(np.abs(vt_dot).sum())
    dv = 0.0 if nv == 0 else (gap(zs, vt + eps * vt_dot / nv, act) - gap(zs, vt - eps * vt_dot / nv, act)) / (2 * eps) * nv
    gdot = float(dG @ th_dot) + dv
    Ph = np.sqrt(p)
    lam = float(np.linalg.eigvalsh((Ph[:, None] * H) * Ph[None, :]).min())
    kap = lam * float(dG @ np.linalg.solve(p[:, None] * H, th_dot)) / gdot
    lam1 = float(np.linalg.eigvalsh(H).min())
    kap_sgd = lam1 * float(dG @ np.linalg.solve(H, th_dot)) / gdot
    return {"kappa2": kap, "kappa2_sgd": kap_sgd, "lambda_min": lam, "gdot": gdot, "gdot_dir_part": dv,
            "gradG_asym": asym}


def tangent(z, s, vt, x, y, act):
    """H = ∇²_z L and θ*′ = −H⁻¹ ∂ₛ∇_z L at v = s·ṽ."""
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    zt = torch.tensor(np.asarray(z, float), dtype=torch.float64)
    vtt = torch.tensor(np.asarray(vt, float), dtype=torch.float64)
    H = torch.autograd.functional.hessian(lambda q: fast_loss(q, s * vtt, X, Y, act), zt).numpy()
    st = torch.tensor(s, dtype=torch.float64, requires_grad=True)
    zz = zt.clone().requires_grad_(True)
    g = torch.autograd.grad(fast_loss(zz, st * vtt, X, Y, act), zz, create_graph=True)[0]
    dgds = np.array([float(torch.autograd.grad(g[i], st, retain_graph=True)[0]) for i in range(5)])
    return H, -np.linalg.solve(H, dgds)


def kappa(H, tan, dG, p):
    """κ = λ_min(P^{1/2}HP^{1/2})·[∇G·(PH)⁻¹θ*′]/[∇G·θ*′], P = diag(p) (as lag_law.kappa, any dimension)."""
    H, tan, dG, p = (np.asarray(v, float) for v in (H, tan, dG, p))
    Ph = np.sqrt(p)
    lam = float(np.linalg.eigvalsh((Ph[:, None] * H) * Ph[None, :]).min())
    num = float(dG @ np.linalg.solve(p[:, None] * H, tan))
    den = float(dG @ tan)
    return lam * num / den, lam, num, den


def analyse(arm, seed, phi, k, step, s_cross, save_states=True):
    """One run: replay; PRIMARY: branch along the run's own v-path, switch, κ₂ (path), χ; SECONDARY: the ray version
    (v = s·ṽ_c, exactly 1A's formula).  Returns a flat row."""
    from .asym_register import _act, _setup
    _setup(); act = _act()
    r = replay(seed, phi, k, step)
    q = r["q"]; x, y = r["x"], r["y"]
    s_c = float(abs(q[2]) + abs(q[5]))
    if save_states:
        STATES.mkdir(parents=True, exist_ok=True)
        with open(STATES / f"{arm}_{seed}.pkl", "wb") as fh:
            pickle.dump({"passage": r["passage"], "q_cross": q, "s_init": r["s_init"], "s_min_pre": r["s_min_pre"],
                         "V": r["V"], "vdot": r["vdot"], "sdot": r["sdot"], "vhat": r["vhat"]}, fh)
    v_c = np.array([q[2], q[5]]); z_c = q[IDX]
    share = np.abs(v_c) / s_c
    vt_dot = (r["vdot"] - v_c / s_c * r["sdot"]) / s_c                 # d(ṽ)/dt: direction drift
    p = 1.0 / (np.sqrt(r["vhat"][IDX]) + EPS)
    row = {"arm": arm, "seed": seed, "phi2": phi, "step": int(step), "s_cross": s_cross, "s_replay": s_c,
           "reproduced": abs(s_c - s_cross) <= 1e-12, "s_init": r["s_init"], "s_min_pre": r["s_min_pre"],
           "sdot": r["sdot"], "min_share": float(share.min()),
           "dir_drift_rel": float(np.abs(vt_dot).sum() / (abs(r["sdot"]) / s_c)) if r["sdot"] else float("nan")}
    # primary: the run's own path
    sw = own_switch_path(z_c, r["V"], int(step), r["vdot"], x, y, act)
    row.update({"status": sw["status"], "dist_c": sw["dist_c"], "branch_grad_c": sw["branch_grad_c"],
                "branch_pd_c": sw["branch_pd_c"], "g_branch_c": sw.get("g_branch_c", float("nan")),
                "n_cont_steps": sw.get("n_steps", 0)})
    Hc = sw["H_c"]; Ph = np.sqrt(p)
    lam_c = float(np.linalg.eigvalsh((Ph[:, None] * Hc) * Ph[None, :]).min())
    row.update({"lambda_min_c": lam_c, "relax_times_c": int(step) * LR * lam_c,
                "chi_c": (r["sdot"] / s_c) / (LR * lam_c) if lam_c > 0 else float("nan")})
    if sw["status"] == "ok":
        pk = path_kappa(sw["z_star"], sw["v_star"], r["vdot"], r["sdot"], p, x, y, act)
        chi = (r["sdot"] / sw["s_star"]) / (LR * pk["lambda_min"])
        row.update({"t_star": sw["t_star"], "s_star": sw["s_star"], "grad_star": sw["grad_star"], "pd_star": sw["pd_star"],
                    **pk, "chi": chi, "r_obs": s_cross / sw["s_star"] - 1, "r_pred": pk["kappa2"] * chi,
                    "z_star": json.dumps([float(v) for v in sw["z_star"]])})
    # secondary: the ray through the crossing direction (1A's formula literally)
    sr = own_switch(z_c, v_c, x, y, act)
    row["ray_status"] = sr["status"]
    if sr["status"] == "ok":
        H, tan = tangent(sr["z_star"], sr["s_star"], sr["vt"], x, y, act)
        dG, _ = grad_gap(sr["z_star"], sr["vt"], act)
        kap, lam, _, _ = kappa(H, tan, dG, p)
        chi_r = (r["sdot"] / sr["s_star"]) / (LR * lam)
        row.update({"ray_s_star": sr["s_star"], "ray_kappa2": kap, "ray_chi": chi_r,
                    "ray_r_obs": s_cross / sr["s_star"] - 1, "ray_r_pred": kap * chi_r})
    row["p"] = json.dumps([float(v) for v in p])
    return row


# ------------------------------------------------------------------------------------------ driver
def _job(args):
    try:
        os.nice(15)
    except OSError:
        pass
    return analyse(*args)


COLUMNS = ["arm", "seed", "phi2", "step", "s_cross", "s_replay", "reproduced", "s_init", "s_min_pre", "sdot", "min_share",
           "dir_drift_rel", "status", "dist_c", "branch_grad_c", "branch_pd_c", "g_branch_c", "n_cont_steps", "lambda_min_c",
           "relax_times_c", "chi_c", "t_star", "s_star", "grad_star", "pd_star", "kappa2", "kappa2_sgd", "lambda_min", "gdot",
           "gdot_dir_part", "gradG_asym", "chi", "r_obs", "r_pred", "z_star", "ray_status", "ray_s_star", "ray_kappa2",
           "ray_chi", "ray_r_obs", "ray_r_pred", "p"]


def run(which=None):
    """Sequential, one process (the caller's), resumable."""
    try:
        os.nice(15)
    except OSError:
        pass
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    A = arms()
    for arm in (which or ["T2-3b", "T2-3c", "T2-3d", "T2-3"]):
        f, phi, k = A[arm]
        d = pd.read_csv(f, float_precision="round_trip")
        d = d[d.crossed & ~d.placed_at_init].sort_values("seed")
        part = OUT / f"parts_{arm}.csv"
        done = set() if not part.exists() else set(pd.read_csv(part).seed)
        for r in d.itertuples():
            if int(r.seed) in done:
                continue
            row = analyse(arm, int(r.seed), phi, k, int(r.step), float(r.s_cross))
            pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(part, mode="a", header=not part.exists(), index=False)
            print(json.dumps({"arm": arm, "seed": int(r.seed), "status": row["status"]}), flush=True)


# ------------------------------------------------------------------------------------------ scoring
def within(obs, pred):
    return abs(obs - pred) <= max(0.01, 0.25 * abs(pred))


def width1_chi():
    """Width-1 χ where the lag law held (lag_law compare: every arm within tolerance).  Returns the largest per-run χ
    and the largest arm-median χ among the arms that held, and the per-run χ range."""
    a = pd.read_csv(RESULTS / "lag_law" / "compare_arms.csv")
    p = pd.read_csv(RESULTS / "lag_law" / "predictions.csv")
    held = a[a.within]
    key = lambda d: d.set.astype(str) + "|" + d.a.round(2).astype(str) + "|" + d.arm.astype(str).map(
        lambda v: str(float(v)) if v.replace(".", "", 1).isdigit() else v)
    p = p[key(p).isin(set(key(held)))]
    ok = p[np.isfinite(p.chi) & (p.chi > 0)]
    return {"arms_held": int(len(held)), "arms_total": int(len(a)), "max_arm_median_chi": float(held.median_chi.max()),
            "max_run_chi": float(ok.chi.max()), "min_run_chi": float(ok.chi.min()), "p95_run_chi": float(ok.chi.quantile(0.95))}


def arm_summary(g, label):
    ok = g[g.status == "ok"]
    ok = ok[np.isfinite(ok.chi)]
    if not len(ok):
        return {"subset": label, "n_runs": int(len(g)), "n_scored": 0}
    mo, mp = float(ok.r_obs.median()), float(ok.r_pred.median())
    per = [within(o, p) for o, p in zip(ok.r_obs, ok.r_pred)]
    return {"subset": label, "n_runs": int(len(g)), "n_scored": int(len(ok)),
            "median_chi": float(ok.chi.median()), "chi_min": float(ok.chi.min()), "chi_max": float(ok.chi.max()),
            "median_kappa2": float(ok.kappa2.median()), "median_obs": mo, "median_pred": mp,
            "obs_over_pred": mo / mp if mp else float("nan"), "tol": max(0.01, 0.25 * abs(mp)),
            "median_within": within(mo, mp), "frac_runs_within": float(np.mean(per)),
            "median_abs_log_err_s_star": float(np.median(np.abs(np.log(ok.s_cross / ok.s_star)))),
            "spearman_obs_pred": float(np.corrcoef(ok.r_obs.rank(), ok.r_pred.rank())[0, 1]) if len(ok) > 2 else float("nan")}


ON_BRANCH = 0.05          # dist_c threshold for 'on its branch at the crossing' (POST HOC; see summary: the distribution is bimodal)


def score():
    rows, parts = [], []
    w1 = width1_chi()
    cmax = w1["max_run_chi"]
    for arm in arms():
        f = OUT / f"parts_{arm}.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f, float_precision="round_trip")
        parts.append(g)
        counts = g.status.value_counts().to_dict()
        ok = g[g.status == "ok"]
        subsets = (("all crossers", g),
                   ("chi <= width-1 max", ok[ok.chi <= cmax]),
                   ("chi > width-1 max", ok[ok.chi > cmax]),
                   ("chi <= width-1 max, on branch", ok[(ok.chi <= cmax) & (ok.dist_c <= ON_BRANCH)]),
                   ("chi <= width-1 max, off branch", ok[(ok.chi <= cmax) & (ok.dist_c > ON_BRANCH)]),
                   ("on branch, s_cross < 1", ok[(ok.dist_c <= ON_BRANCH) & (ok.s_cross < 1)]),
                   ("on branch, s_cross >= 1", ok[(ok.dist_c <= ON_BRANCH) & (ok.s_cross >= 1)]))
        for label, sub in subsets:
            rows.append({"arm": arm, **arm_summary(sub, label),
                         "median_dist_c": float(sub.dist_c.median()) if len(sub) else float("nan"),
                         "status_counts": json.dumps(counts), "replays_reproduce": bool(g.reproduced.all())})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary.csv", index=False)
    allp = pd.concat(parts, ignore_index=True)
    t23 = allp[allp.arm == "T2-3"]
    t23ok = t23[t23.status == "ok"]
    sl = allp[allp.arm != "T2-3"]
    slok = sl[sl.status == "ok"]
    q = lambda v: [float(np.nanmin(v)), float(np.nanmedian(v)), float(np.nanmax(v))] if len(v) else None
    out = {"label": "POST HOC (Track 2B); no verdict changes: T2-3 FAIL, T2-3b/T2-3c UNRESOLVED, T2-3d FAIL",
           "width1": w1, "on_branch_threshold": ON_BRANCH,
           "T2-3": {"n": int(len(t23)), "status_counts": t23.status.value_counts().to_dict(),
                    "chi_c_min_median_max": q(t23.chi_c.dropna()),
                    "chi_min_median_max_(branch_defined)": q(t23ok.chi),
                    "dist_c_min_median_max": q(t23.dist_c.dropna()),
                    "n_on_branch": int((t23.dist_c <= ON_BRANCH).sum()),
                    "relax_times_c_median": float(t23.relax_times_c.median())},
           "slowed": {"n": int(len(sl)), "status_counts": sl.status.value_counts().to_dict(),
                      "dist_c_on_branch_max": float(slok[slok.dist_c <= ON_BRANCH].dist_c.max()),
                      "dist_c_off_branch_min": float(slok[slok.dist_c > ON_BRANCH].dist_c.min()) if (slok.dist_c > ON_BRANCH).any() else None,
                      "chi_on_branch_max": float(slok[slok.dist_c <= ON_BRANCH].chi.max()),
                      "chi_c_all_min_median_max": q(sl.chi_c.dropna()),
                      "relax_times_c_on_branch_min": float(slok[slok.dist_c <= ON_BRANCH].relax_times_c.min()),
                      "relax_times_c_off_branch_max": float(slok[slok.dist_c > ON_BRANCH].relax_times_c.max()) if (slok.dist_c > ON_BRANCH).any() else None,
                      "relax_times_c_not_ok_median": float(sl[sl.status != "ok"].relax_times_c.median())}}
    # χ bins (all arms, branch-defined runs), the validity condition read from the data
    edges = [0, 1e-4, 1e-2, cmax, 0.3, 3.0, np.inf]
    ok_all = allp[allp.status == "ok"].copy()
    ok_all["bin"] = pd.cut(ok_all.chi, edges, right=True)
    ok_all["within"] = [within(o, p_) for o, p_ in zip(ok_all.r_obs, ok_all.r_pred)]
    Bn = ok_all.groupby("bin", observed=True).agg(n=("seed", "size"), arms=("arm", lambda a: ",".join(sorted(set(a)))),
                                                  frac_within=("within", "mean"), median_obs=("r_obs", "median"),
                                                  median_pred=("r_pred", "median"), median_kappa2=("kappa2", "median"),
                                                  median_dist_c=("dist_c", "median"),
                                                  median_relax_times_c=("relax_times_c", "median")).reset_index()
    Bn["bin"] = Bn["bin"].astype(str)
    Bn.to_csv(OUT / "chi_bins.csv", index=False)
    nd = allp[allp.status != "ok"]
    out["branch_undefined"] = {"n": int(len(nd)), "by_arm": nd.arm.value_counts().to_dict(),
                               "s_cross_max": float(nd.s_cross.max()), "step_median": float(nd.step.median()),
                               "relax_times_c_median": float(nd.relax_times_c.median()),
                               "chi_c_median": float(nd.chi_c.median())}
    print(Bn.round(5).to_string(index=False))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(S.drop(columns=["status_counts"]).round(5).to_string(index=False))
    print(json.dumps(out, indent=1, default=float))
    return S, out


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2:] or None), "score": score}[sys.argv[1]]()
