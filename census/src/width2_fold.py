"""POST HOC (author's request, 2026-09-28; existing data only): for the full-speed width-2 runs T2-3 (a = 1.30, Δ = 0.4, crossing at a
median 3.3 × the population threshold), is there an OCCUPIED branch whose fold or own switch matches the crossings?
T2-3's registered FAIL stands; nothing here re-scores it.

Hypothesis tested (from Track 1A): a run sits on the branch it occupies and leaves it where the branch ends, at its fold
(smallest eigenvalue of the reduced Hessian → 0, a turning point of the branch in s), or crosses where the branch's own gap turns
positive (its switch).

Definitions (fixed before any T2-3 branch was computed):
  fixed-scale conditional loss  L(w; s), w = (α₁, β₁, α₂, β₂, u, b), output weights v = s·(σ₁u, σ₂(1 − u)), u ∈ (0, 1), the signs
       σ fixed by the state (one orthant of the ℓ₁ sphere); the run's own training set; asymmetric windows.  Branch = {∇_w L = 0,
       Hessian PD} as a curve in (w, p = log s).
  reduced coordinates  'two-unit': all of w.  'duplicate' (units copies up to orientation and the transfer direction exactly flat:
       σ₁ = orientation·σ₂): u dropped (it is a symmetry).  'single' (one unit's share < 1e−3): u set to 0 or 1 and the idle unit's
       (α, β) dropped.  The reduced Hessian is the Hessian in the active coordinates.
  occupied branch at step t  the run is replayed; its state at step t (params + Adam moments) is relaxed at its fixed scale
       s(t) = ‖v(t)‖₁ (width2_train replay operations: Adam lr 0.01 with preserved moments, ℓ₁ projection, RELAX = 8,000 steps),
       then Newton-polished (Levenberg–Marquardt) in the active coordinates to max|∇| ≤ 1e−10 with the reduced Hessian PD.
       Times: t = round(f·t_c), f ∈ {0, 0.1, …, 0.9}, and t_c − 1 (t_c = the crossing step; duplicates removed; t = 0 is the
       initial state).  PRIMARY occupied branch = the one at the LATEST of these times whose relaxed basin is unplaced.
  continuation  pseudo-arclength in (w_active, p) upward in s from s(t) (tangent = null vector of [H | ∂ₚ∇L], oriented by the
       previous tangent; predictor–corrector with the arclength constraint; steps ≤ H_MAX, halved when the corrector fails, needs
       more than 8 iterations, or turns the tangent by more than 30°).  Stops at the first event:
         fold    the s-component of the tangent changes sign (located by bisection in the step to |τₚ| ≤ 1e−6·|τ|);
         switch  the exact gap G₊ of the branch's φ changes sign (bisection to 1e−9 in p);
         boundary u leaves (0, 1) (a two-unit branch becomes single-unit);  none  s > S_MAX_FACTOR·s_c;  failed.
  validation of a fold  (a) step halving: the fold recomputed with H_MAX/2 agrees to 1e−4 relative in s;  (b) eigenvalue path:
       λ_min/λ_max of the reduced Hessian at the fold ≤ 1e−3 and positive at every accepted point before it;  (c) independent search:
       at s_f·(1 + 0.01), 20 Levenberg–Marquardt minimisations from the fold point perturbed by N(0, 0.05²) find no PD minimum within
       max-distance 0.25 of the fold point, while at s_f·(1 − 0.01) the branch point exists.  Validated iff (a), (b) and (c).
  comparison  per run with a primary branch whose exit (fold or switch) was found: s_cross / s_exit; reported: the median, the
       fraction within 15% (|ratio − 1| ≤ 0.15), by event type; beside s_cross / s_pop (s_pop = 0.44508).
  ADDENDUM (added after timing runs 600,000 and 600,002, disclosed): whether the basin just before crossing (t_c − 1) is placed,
       and the first step at which the run's fixed-scale basin is placed (bisection over steps inside the grid bracket), with its
       scale s_bp; compared with s_pop and s_cross.  (In the two timed runs: 600,000's basin was placed from s ≈ 0.37 on although
       it crossed at 2.29; 600,002 stayed on an unplaced duplicate branch to its switch, 4.65, and crossed at 4.81.)

    python -m src.width2_fold run        # one process, nice 15, resumable; memory watchdog (stops cleanly)
    python -m src.width2_fold summary
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "width2_fold"
LR, RELAX, GTOL = 1e-2, 8000, 1e-10
H_MAX, H_MIN, MAX_STEPS, S_MAX_FACTOR = 0.05, 1e-6, 4000, 10.0
FRACS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
S_POP = 0.44507940623559955
SINGLE_SHARE, DUP_TOL = 1e-3, 1e-6


def _nice():
    try:
        os.nice(15)
    except OSError:
        pass


def memory_ok(min_free_frac=0.25, min_swap_free_mb=500):
    """Watchdog: macOS's 'System-wide memory free percentage' (memory_pressure -Q) ≥ 25% and swap free ≥ 500 MB."""
    try:
        mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
        free = float(mp.split("free percentage:")[1].split("%")[0]) / 100
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        swap_free = float(sw.split("free = ")[1].split("M")[0])
        return free >= min_free_frac and swap_free >= min_swap_free_mb, {"free_frac": free, "swap_free_mb": swap_free}
    except Exception as e:                                        # noqa: BLE001
        return False, {"error": str(e)}


# ------------------------------------------------------------------------------------------ generic continuation
def null_tangent(H, g_p, prev=None, direction=1.0):
    """Unit null vector of A = [H | g_p] (n × (n+1)), oriented along prev (or with τₚ·direction > 0)."""
    A = np.column_stack([H, g_p])
    _, _, Vt = np.linalg.svd(A)
    t = Vt[-1]
    if prev is not None:
        t = t if t @ prev >= 0 else -t
    elif t[-1] * direction < 0:
        t = -t
    return t / np.linalg.norm(t)


def corrector(prob, x_pred, tau, tol=1e-11, maxit=12):
    """Newton on [∇L(y, p); τ·(x − x_pred)] = 0.  Returns (x, converged, iterations)."""
    x = x_pred.copy()
    for it in range(1, maxit + 1):
        g = prob.grad(x[:-1], x[-1])
        r = np.r_[g, tau @ (x - x_pred)]
        if np.abs(g).max() <= tol:
            return x, True, it
        J = np.vstack([np.column_stack([prob.hess(x[:-1], x[-1]), prob.gp(x[:-1], x[-1])]), tau])
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return x, False, it
        x = x + dx
        if not np.all(np.isfinite(x)):
            return x, False, it
    g = prob.grad(x[:-1], x[-1])
    return x, bool(np.abs(g).max() <= tol * 100), maxit


def lam_min_rel(H):
    e = np.linalg.eigvalsh(0.5 * (H + H.T))
    return float(e[0]), float(e[0] / max(np.abs(e).max(), 1e-300))


def continue_branch(prob, y0, p0, p_max, h_max=H_MAX, h_min=H_MIN, max_steps=MAX_STEPS, event=None, boundary=None):
    """Pseudo-arclength continuation from (y0, p0) with τₚ > 0 initially.  `event(y, p)` → a float whose sign change is a
    switch; `boundary(y)` → True when the branch leaves its chart.  Returns a dict with status in
    {'fold', 'switch', 'boundary', 'none', 'failed'} and the path."""
    x = np.r_[np.asarray(y0, float), p0]
    tau = null_tangent(prob.hess(x[:-1], x[-1]), prob.gp(x[:-1], x[-1]), direction=1.0)
    h = h_max
    ev = event(x[:-1], x[-1]) if event else None
    l0 = lam_min_rel(prob.hess(x[:-1], x[-1]))
    path = [{"p": float(x[-1]), "lam": l0[0], "lam_rel": l0[1], "tau_p": float(tau[-1]), "event": ev}]
    for _ in range(max_steps):
        if x[-1] >= p_max:
            return {"status": "none", "path": path, "x": x}
        xp = x + h * tau
        xn, ok, it = corrector(prob, xp, tau)
        if ok:
            tn = null_tangent(prob.hess(xn[:-1], xn[-1]), prob.gp(xn[:-1], xn[-1]), prev=tau)
            ok = it <= 8 and float(tn @ tau) >= math.cos(math.radians(30))
        if not ok:
            if h / 2 < h_min:
                return {"status": "failed", "path": path, "x": x}
            h /= 2
            continue
        if boundary is not None and boundary(xn[:-1]):
            return {"status": "boundary", "path": path, "x": x}
        evn = event(xn[:-1], xn[-1]) if event else None
        lam, lrel = lam_min_rel(prob.hess(xn[:-1], xn[-1]))
        if tn[-1] <= 0 < tau[-1]:                                   # turning point in p between x and xn
            xf = _refine(prob, x, tau, 0.0, h, lambda xx, tt: tt[-1])
            return {"status": "fold", "path": path, "x": xf, "p_event": float(xf[-1]),
                    "lam_rel_event": lam_min_rel(prob.hess(xf[:-1], xf[-1]))[1]}
        if event and (evn > 0) != (ev > 0):
            xs = _refine(prob, x, tau, 0.0, h, lambda xx, tt: event(xx[:-1], xx[-1]))
            return {"status": "switch", "path": path, "x": xs, "p_event": float(xs[-1]),
                    "lam_rel_event": lam_min_rel(prob.hess(xs[:-1], xs[-1]))[1]}
        x, tau, ev = xn, tn, evn
        path.append({"p": float(x[-1]), "lam": lam, "lam_rel": lrel, "tau_p": float(tau[-1]), "event": ev})
        h = min(h_max, 1.5 * h)
    return {"status": "failed", "path": path, "x": x}


def _refine(prob, x, tau, a, b, fun, iters=60):
    """Bisection in the step length h ∈ [a, b] from x along τ for a sign change of fun(x(h), τ(h)) from ≤ 0 … (fun at h = a
    has the sign of 'before').  Returns the point at the located step."""
    def point(h):
        xn, ok, _ = corrector(prob, x + h * tau, tau)
        tn = null_tangent(prob.hess(xn[:-1], xn[-1]), prob.gp(xn[:-1], xn[-1]), prev=tau)
        return xn, tn
    fa = fun(x, tau)
    lo, hi = a, b
    xm = x
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        xm, tm = point(mid)
        fm = fun(xm, tm)
        if (fm > 0) == (fa > 0):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return point(0.5 * (lo + hi))[0]


def lm_minimise(prob, y0, p, gtol=GTOL, iters=300):
    """Levenberg–Marquardt on L(·; p) (prob.loss, grad, hess).  Returns (y, max|∇|, converged)."""
    y = np.asarray(y0, float).copy(); mu = 1e-3; f0 = prob.loss(y, p)
    for _ in range(iters):
        g = prob.grad(y, p)
        if np.abs(g).max() <= gtol:
            return y, float(np.abs(g).max()), True
        H = prob.hess(y, p); sc = max(1.0, float(np.abs(np.diag(H)).max()))
        ok = False
        for _ in range(30):
            try:
                d = np.linalg.solve(H + mu * sc * np.eye(len(y)), -g)
            except np.linalg.LinAlgError:
                mu *= 10; continue
            f1 = prob.loss(y + d, p)
            if f1 <= f0:
                y, f0, ok, mu = y + d, f1, True, max(mu / 3, 1e-14)
                break
            mu *= 10
        if not ok:
            break
    g = prob.grad(y, p)
    return y, float(np.abs(g).max()), bool(np.abs(g).max() <= gtol)


def dlam2_ds(path, s_f, rel=0.05, n_min=3):
    """d(λ_min²)/ds at a fold, from a least-squares line of λ_min² against s over the accepted path points with λ_min > 0 within
    `rel` below s_f (math note §14: |m′c′| = ¼·d(λ_min²)/ds).  NaN if fewer than n_min points."""
    pts = [(math.exp(q["p"]), q["lam"]) for q in path if q.get("lam") is not None and q["lam"] > 0
           and s_f * (1 - rel) <= math.exp(q["p"]) <= s_f]
    if len(pts) < n_min:
        return float("nan")
    sv, lv = np.array(pts).T
    return float(np.polyfit(sv, lv ** 2, 1)[0])


def validate_fold(prob, y_f, p_f, rerun_p, rng, n_search=20, sigma=0.05, radius=0.25, eps=0.01):
    """(a) step halving (rerun_p: the fold p recomputed with half the maximum step), (b) λ_min at the fold, (c) independent search."""
    s_f = math.exp(p_f)
    a_ok = rerun_p is not None and abs(math.exp(rerun_p) / s_f - 1) <= 1e-4
    lam, lrel = lam_min_rel(prob.hess(y_f, p_f))
    b_ok = abs(lrel) <= 1e-3
    p_beyond, p_before = math.log(s_f * (1 + eps)), math.log(s_f * (1 - eps))
    found_beyond = 0
    for _ in range(n_search):
        y, gm, conv = lm_minimise(prob, y_f + sigma * rng.standard_normal(len(y_f)), p_beyond)
        if conv and np.abs(y - y_f).max() <= radius and lam_min_rel(prob.hess(y, p_beyond))[0] > 0:
            found_beyond += 1
    yb, gb, cb = lm_minimise(prob, y_f, p_before)
    before_ok = cb and np.abs(yb - y_f).max() <= radius and lam_min_rel(prob.hess(yb, p_before))[0] > 0
    c_ok = found_beyond == 0 and before_ok
    return {"halving_ok": bool(a_ok), "lam_rel_at_fold": lrel, "eig_ok": bool(b_ok), "found_beyond": found_beyond,
            "exists_before": bool(before_ok), "search_ok": bool(c_ok), "validated": bool(a_ok and b_ok and c_ok)}


# ------------------------------------------------------------------------------------------ the width-2 problem
class Width2Problem:
    """L(w; p) on the run's training set, active coordinates only (the rest held at `w_full`)."""

    def __init__(self, x, y, act, w_full, signs, active):
        import torch
        self.X = torch.tensor(x, dtype=torch.float64); self.Y = torch.tensor(y, dtype=torch.float64)
        self.act, self.sig = act, np.asarray(signs, float)
        self.w_full = np.asarray(w_full, float).copy(); self.active = list(active)

    def _full(self, ya):
        import torch
        w = torch.tensor(self.w_full, dtype=torch.float64)
        return w.index_put((torch.tensor(self.active),), ya)

    def _L(self, ya, p):
        import torch
        w = self._full(ya)
        s = torch.exp(p)
        f = lambda t: t + self.act.a * torch.sin(t)
        v1, v2 = s * self.sig[0] * w[4], s * self.sig[1] * (1 - w[4])
        z = v1 * f(w[0] * self.X + w[1]) + v2 * f(w[2] * self.X + w[3]) + w[5]
        return torch.nn.functional.binary_cross_entropy_with_logits(z, self.Y)

    def _t(self, v):
        import torch
        return torch.tensor(np.asarray(v, float), dtype=torch.float64)

    def loss(self, ya, p):
        return float(self._L(self._t(ya), self._t(p)))

    def grad(self, ya, p):
        import torch
        yt = self._t(ya).requires_grad_(True)
        return torch.autograd.grad(self._L(yt, self._t(p)), yt)[0].numpy()

    def hess(self, ya, p):
        import torch
        pt = self._t(p)
        return torch.autograd.functional.hessian(lambda q: self._L(q, pt), self._t(ya)).numpy()

    def gp(self, ya, p):
        import torch
        yt = self._t(ya).requires_grad_(True); pt = self._t(p).requires_grad_(True)
        g = torch.autograd.grad(self._L(yt, pt), yt, create_graph=True)[0]
        return np.array([float(torch.autograd.grad(g[i], pt, retain_graph=True)[0]) for i in range(len(self.active))])

    def full(self, ya):
        w = self.w_full.copy(); w[self.active] = ya
        return w

    def gap(self, ya, p=None):
        from .width2_geometry import gaps
        w = self.full(ya)
        v = np.array([self.sig[0] * w[4], self.sig[1] * (1 - w[4])])
        g = gaps(np.array([w[0], w[1], w[2], w[3]]), v, self.act)["G+"]
        return 0.5 * (g[0] + g[1])


def to_w(q):
    """(α₁, β₁, v₁, α₂, β₂, v₂, b) → (w, signs, s)."""
    s = abs(q[2]) + abs(q[5])
    return np.array([q[0], q[1], q[3], q[4], abs(q[2]) / s, q[6]]), np.array([np.sign(q[2]) or 1.0, np.sign(q[5]) or 1.0]), s


def chart(w, signs):
    """Reduced coordinates for the configuration: ('single', …), ('duplicate', …) or ('two-unit', …)."""
    u = w[4]
    if u <= SINGLE_SHARE:
        return "single", [2, 3, 5], 0.0
    if 1 - u <= SINGLE_SHARE:
        return "single", [0, 1, 5], 1.0
    for o in (1.0, -1.0):
        if (abs(w[2] - o * w[0]) <= DUP_TOL * max(1, abs(w[0])) and abs(w[3] - o * w[1]) <= DUP_TOL * max(1, abs(w[1]))
                and signs[0] == o * signs[1]):
            return "duplicate", [0, 1, 2, 3, 5], None
    return "two-unit", [0, 1, 2, 3, 4, 5], None


# ------------------------------------------------------------------------------------------ per run
def replay_states(seed, k, step, fracs=FRACS, keep_all=False):
    """T2-3's training loop (asym_register.train_one, φ₂ = 1) to the crossing, saving (q, Adam state) at the grid steps."""
    import torch
    from .asym_register import _act, _setup, training_set
    from .width2_train import init_params, logits
    torch.set_num_threads(1)
    _setup(); act = _act()
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); q0[2] *= k; q0[5] *= k
    want = sorted({int(round(f * step)) for f in fracs} | {int(step) - 1})
    want = [t for t in want if 0 <= t < step]
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    st = {}
    if 0 in want:
        st[0] = (q0.numpy().copy(), None)
    for t in range(1, int(step) + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        opt.step()
        if t in want or (keep_all and t < step):
            st[t] = (q.detach().numpy().copy(), {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in opt.state[q].items()})
    s_end = float(q.detach()[2].abs() + q.detach()[5].abs())
    return st, x, y, act, s_end


def relax(q, adam, x, y, act, steps=RELAX):
    """Fixed-scale relaxation (width2_train replay operations: Adam, preserved moments, ℓ₁ projection)."""
    import torch
    from .width2_train import _project, logits
    radius = float(abs(q[2]) + abs(q[5]))
    qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([qt], lr=LR)
    if adam:
        opt.state[qt] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in adam.items()}
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(qt, X, act), Y).backward()
        opt.step()
        _project(qt, radius)
    return qt.detach().numpy().copy()


def occupied_basin(q, adam, x, y, act):
    """Relax at fixed scale, choose the chart, LM-polish in it (re-choosing the chart once if the polish lands on a duplicate)."""
    qr = relax(q, adam, x, y, act)
    w, sg, s = to_w(qr)
    p = math.log(s)
    for _ in range(2):
        kind, active, ufix = chart(w, sg)
        if ufix is not None:
            w = w.copy(); w[4] = ufix
        prob = Width2Problem(x, y, act, w, sg, active)
        ya, gm, conv = lm_minimise(prob, w[active], p)
        w = prob.full(ya)
        if chart(w, sg)[0] == kind:
            break
    lam, lrel = lam_min_rel(prob.hess(ya, p))
    return {"prob": prob, "y": ya, "p": p, "s": s, "kind": kind, "grad": gm, "converged": conv, "pd": lam > 0,
            "lam_rel": lrel, "gap": prob.gap(ya)}


def analyse_run(seed, k, step, s_cross, rng_seed=0):
    st_all, x, y, act, s_end = replay_states(seed, k, step, keep_all=True)
    want = sorted({int(round(f * step)) for f in FRACS} | {int(step) - 1})
    st = {t: st_all[t] for t in want if t in st_all}
    rows, basins = [], {}
    for t in sorted(st):
        b = occupied_basin(*st[t], x, y, act)
        basins[t] = b
        rows.append({"t": t, "s_t": b["s"], "kind": b["kind"], "converged": b["converged"], "pd": b["pd"],
                     "lam_rel": b["lam_rel"], "gap": b["gap"], "placed": b["gap"] > 0})
    ok = [t for t in sorted(basins) if basins[t]["converged"] and basins[t]["pd"] and basins[t]["gap"] <= 0]
    out = {"seed": seed, "cross_step": int(step), "s_cross": s_cross, "reproduced": abs(s_end - s_cross) <= 1e-12,
           "basins": json.dumps(rows), "n_times": len(rows), "n_unplaced_basins": len(ok),
           "n_placed_basins": int(sum(r["placed"] for r in rows))}
    # ADDENDUM (added after timing two runs, disclosed): the basin the run is in just before crossing, and the first step at
    # which the run's fixed-scale basin is placed (bisection over steps between the last unplaced-basin grid time before the
    # first placed-basin grid time and that time).
    ts = sorted(basins)
    last = basins[ts[-1]]
    out.update({"basin_before_cross_placed": bool(last["gap"] > 0), "basin_before_cross_kind": last["kind"]})
    first_pl = next((t for t in ts if basins[t]["gap"] > 0), None)
    if first_pl is not None:
        prev = [t for t in ts if t < first_pl]
        if prev:
            lo, hi = prev[-1], first_pl
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if occupied_basin(*st_all[mid], x, y, act)["gap"] > 0:
                    hi = mid
                else:
                    lo = mid
            s_of = lambda t: float(abs(st_all[t][0][2]) + abs(st_all[t][0][5]))
            out.update({"basin_placed_step": hi, "s_basin_placed": s_of(hi), "s_basin_last_unplaced": s_of(lo),
                        "ratio_cross_basin_placed": s_cross / s_of(hi)})
        else:
            out.update({"basin_placed_step": 0, "s_basin_placed": float(abs(st_all[0][0][2]) + abs(st_all[0][0][5]))})
    if not ok:
        return {**out, "status": "no unplaced occupied basin"}
    t0 = ok[-1]; b = basins[t0]
    prob = b["prob"]
    bnd = (lambda ya: not (0 < prob.full(ya)[4] < 1)) if b["kind"] == "two-unit" else None
    p_max = math.log(S_MAX_FACTOR * s_cross)
    res = continue_branch(prob, b["y"], b["p"], p_max, event=prob.gap, boundary=bnd)
    out.update({"t_primary": t0, "frac_primary": t0 / step, "s_primary": b["s"], "kind": b["kind"], "status": res["status"],
                "n_path": len(res["path"]), "min_lam_rel_path": float(min(r["lam_rel"] for r in res["path"]))})
    tail = res["path"][-30:]
    out["path_tail"] = json.dumps([{"s": math.exp(q["p"]), "lam_min": q["lam"], "lam_rel": q["lam_rel"], "tau_p": q["tau_p"],
                                    "gap": q["event"]} for q in tail])
    if res["status"] in ("fold", "switch"):
        s_ev = math.exp(res["p_event"])
        out["lam_min_event"] = lam_min_rel(prob.hess(res["x"][:-1], res["x"][-1]))[0]
        out.update({"s_exit": s_ev, "ratio_cross_exit": s_cross / s_ev, "lam_rel_event": res["lam_rel_event"]})
        if res["status"] == "fold":
            re = continue_branch(prob, b["y"], b["p"], p_max, h_max=H_MAX / 2, event=prob.gap, boundary=bnd)
            rerun = re["p_event"] if re["status"] == "fold" else None
            v = validate_fold(prob, res["x"][:-1], res["p_event"], rerun, np.random.default_rng(rng_seed))
            out.update({f"fold_{kk}": vv for kk, vv in v.items()})
            out["dlam2_ds_at_fold"] = dlam2_ds(res["path"], s_ev)
        else:
            re = continue_branch(prob, b["y"], b["p"], p_max, h_max=H_MAX / 2, event=prob.gap, boundary=bnd)
            out["switch_halving_ok"] = bool(re["status"] == "switch" and abs(math.exp(re["p_event"]) / s_ev - 1) <= 1e-4)
    return out


COLUMNS = ["seed", "cross_step", "s_cross", "reproduced", "status", "t_primary", "frac_primary", "s_primary", "kind", "s_exit",
           "ratio_cross_exit", "lam_rel_event", "n_path", "min_lam_rel_path", "fold_halving_ok", "fold_lam_rel_at_fold",
           "fold_eig_ok", "fold_found_beyond", "fold_exists_before", "fold_search_ok", "fold_validated", "switch_halving_ok",
           "lam_min_event", "dlam2_ds_at_fold", "path_tail",
           "n_times", "n_unplaced_basins", "n_placed_basins", "basin_before_cross_placed", "basin_before_cross_kind",
           "basin_placed_step", "s_basin_placed", "s_basin_last_unplaced", "ratio_cross_basin_placed", "basins"]


def run():
    _nice()
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    k = json.loads((RESULTS / "asym_frozen.json").read_text())["k"]
    tr = pd.read_csv(RESULTS / "asym_parts" / "train.csv", float_precision="round_trip")
    tr = tr[tr.crossed & ~tr.placed_at_init].sort_values("seed")
    f = OUT / "runs.csv"
    done = set() if not f.exists() else set(pd.read_csv(f).seed)
    for r in tr.itertuples():
        if int(r.seed) in done:
            continue
        ok, m = memory_ok()
        if not ok:
            print(json.dumps({"STOP": "memory watchdog", **m}), flush=True)
            return
        row = analyse_run(int(r.seed), k, int(r.step), float(r.s_cross), rng_seed=int(r.seed))
        pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(json.dumps({"seed": int(r.seed), "status": row["status"]}), flush=True)


def summary():
    d = pd.read_csv(OUT / "runs.csv", float_precision="round_trip")
    ex = d[d.status.isin(["fold", "switch"])]
    within = lambda r: float((np.abs(r - 1) <= 0.15).mean()) if len(r) else float("nan")
    by = {}
    for ev, g in ex.groupby("status"):
        val = g[g.fold_validated.astype(bool)] if ev == "fold" else g
        by[ev] = {"n": int(len(g)), "n_validated": int(len(val)) if ev == "fold" else int(g.switch_halving_ok.astype(bool).sum()),
                  "median_ratio": float(g.ratio_cross_exit.median()), "frac_within_15pct": within(g.ratio_cross_exit),
                  "s_exit_min_median_max": [float(g.s_exit.min()), float(g.s_exit.median()), float(g.s_exit.max())]}
    fv = ex[(ex.status == "switch") | ((ex.status == "fold") & ex.fold_validated.astype(bool))]
    out = {"label": "POST HOC (author's request 2026-09-28); T2-3's registered FAIL stands",
           "n_runs": int(len(d)), "replays_reproduce": bool(d.reproduced.all()),
           "status_counts": d.status.value_counts().to_dict(), "kind_counts": d.kind.value_counts().to_dict(),
           "exit_found": {"n": int(len(ex)), "median_ratio_cross_exit": float(ex.ratio_cross_exit.median()) if len(ex) else None,
                          "frac_within_15pct": within(ex.ratio_cross_exit)},
           "exit_validated": {"n": int(len(fv)), "median_ratio_cross_exit": float(fv.ratio_cross_exit.median()) if len(fv) else None,
                              "frac_within_15pct": within(fv.ratio_cross_exit)},
           "by_event": by,
           "all_runs_vs_population": {"median_ratio": float((d.s_cross / S_POP).median()),
                                      "frac_within_15pct": within(d.s_cross / S_POP)},
           "primary_time_frac_median": float(d.frac_primary.median()) if "frac_primary" in d else None,
           "addendum_basin": {"n_basin_before_cross_placed": int(d.basin_before_cross_placed.astype(bool).sum()),
                              "n_basin_placed_found": int(d.s_basin_placed.notna().sum()),
                              "s_basin_placed_min_median_max": [float(d.s_basin_placed.min()), float(d.s_basin_placed.median()),
                                                                float(d.s_basin_placed.max())],
                              "median_s_basin_placed_over_pop": float((d.s_basin_placed / S_POP).median()),
                              "frac_s_basin_placed_within_15pct_of_pop": within((d.s_basin_placed / S_POP).dropna()),
                              "median_ratio_cross_basin_placed": float(d.ratio_cross_basin_placed.median())}}
    own = RESULTS / "asym_posthoc" / "own_thresholds.csv"                 # width-2 own-sample thresholds (10 seeds, exploratory)
    if own.exists():
        o = pd.read_csv(own)[["seed", "s_own"]].merge(d[["seed", "s_basin_placed", "s_cross"]], on="seed").dropna()
        out["addendum_own_width2_threshold"] = {"n": int(len(o)), "seeds": [int(v) for v in o.seed],
                                                "median_s_basin_placed_over_own": float((o.s_basin_placed / o.s_own).median()),
                                                "frac_within_15pct": within(o.s_basin_placed / o.s_own),
                                                "median_s_cross_over_own": float((o.s_cross / o.s_own).median())}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    {"run": run, "summary": summary}[sys.argv[1]]()
