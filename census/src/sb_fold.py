"""Track 1A (POST HOC, existing data, no new training): the fold of the linear-dominant branch of the v2/v3
simplicity-bias landscape, and the v3 training runs' branch along their trajectories.  Report: results/sb_fold_report.md.

Landscape: the v2 weight-decayed fixed-scale objective (λ = 1e−4, same data), s* = 3.5914 (committed v2 s_q; the global
minimiser jumps there from the linear-dominant branch M, ρ₂ = 0.182, to the slab branch S, ρ₂ = 0.448).

Reduced coordinates (continuation, Hessian).  Only a branch's ACTIVE units are coordinates: u = (W_A, c_A, η_A∖g, b) with
v = η²/Σ η² over A and η_g ≡ 1 for the gauge unit g (removes the exactly flat η-radial direction), b trained (joint in
(u, b) rather than profiled: a stationary point of one is a stationary point of the other, and with ∂²L/∂b² > 0 the joint
Hessian is positive definite iff the profiled one is, Schur complement).  Idle units (v = 0, W = c = 0) are excluded:
in the η² parametrisation their Hessian block is diagonal, λ on (W, c) and 2(−Σ vₖ∂L/∂vₖ)/Σ η² on η, both positive,
decoupled from the active block (every mixed term carries vₖ = 0); they are exact spectators, not fold directions.
Permutations and sign flips of units are discrete symmetries (no zero eigenvalue unless two active units coincide,
which is checked).

Continuation: pseudo-arclength in x = (u, s) on F(u, s) = ∇_u L = 0: tangent = null vector of [F_u F_s] (oriented
continuously), Euler predictor of arclength h, Newton corrector on (F = 0, t·(x − x_pred) = 0).  A fold is where the
tangent's s-component changes sign; it is refined by bisection on the step length, and the smallest eigenvalue of
F_u (the reduced Hessian) must cross zero there.

    python -m src.sb_fold branches      # M, S (and the small-s linear branch L0) by continuation; fold; validation
    python -m src.sb_fold runs          # branch classification of the v3 runs (every step)
    python -m src.sb_fold summarise     # summary JSON + gate
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import simplicity_bias as sb
from . import simplicity_bias_v2 as v2

OUT = sb.RESULTS / "sb_fold"
LAM = 1e-4
S_STAR = 3.5913755424683727
V2_PARTS = sb.RESULTS / "simplicity_bias_v2" / "pilot_parts"
V3_RUNS = sb.RESULTS / "simplicity_bias_v3" / "runs"
ACTIVE_TOL = 1e-6
DIST_TOL = 1e-3          # function-space distance threshold for "on the branch" (unit-ℓ₁ φ, RMS over the 800 points)


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch                     # no global dtype change: every tensor below is created as float64 explicitly


# ------------------------------------------------------------------------------------------ generic continuation
def tangent(J, prev=None):
    """Unit null vector of J (m × (m+1)), oriented to agree with prev."""
    _, _, Vt = np.linalg.svd(J)
    t = Vt[-1]
    if prev is not None and t @ prev < 0:
        t = -t
    return t


def newton_fixed_s(F, JF, u0, s, tol=1e-11, maxit=50):
    """Newton on F(u, s) = 0 at fixed s.  Returns (u, converged, |F|)."""
    u = np.array(u0, float)
    for _ in range(maxit):
        f = F(u, s)
        if np.abs(f).max() <= tol:
            return u, True, float(np.abs(f).max())
        J = JF(u, s)[:, :-1]
        try:
            du = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            return u, False, float(np.abs(f).max())
        u = u + du
        if not np.all(np.isfinite(u)):
            return u, False, math.inf
    f = F(u, s)
    return u, bool(np.abs(f).max() <= tol), float(np.abs(f).max())


def corrector(F, JF, xp, t, tol=1e-11, maxit=30):
    x = xp.copy()
    for _ in range(maxit):
        f = F(x[:-1], x[-1])
        g = t @ (x - xp)
        if np.abs(f).max() <= tol and abs(g) <= tol:
            return x, True
        J = JF(x[:-1], x[-1])
        A = np.vstack([J, t[None]])
        try:
            dx = np.linalg.solve(A, -np.r_[f, g])
        except np.linalg.LinAlgError:
            return x, False
        x = x + dx
        if not np.all(np.isfinite(x)):
            return x, False
    f = F(x[:-1], x[-1])
    return x, bool(np.abs(f).max() <= tol)


def _record(F, JF, x, t):
    J = JF(x[:-1], x[-1]); H = 0.5 * (J[:, :-1] + J[:, :-1].T)
    ev = np.linalg.eigvalsh(H)
    return {"x": x.copy(), "t": t.copy(), "s": float(x[-1]), "ts": float(t[-1]), "eig_min": float(ev[0]),
            "eigs": ev[:3].tolist(), "res": float(np.abs(F(x[:-1], x[-1])).max())}


def step(F, JF, x, t, h, tol=1e-11):
    xn, ok = corrector(F, JF, x + h * t, t, tol=tol)
    if not ok:
        return None
    tn = tangent(JF(xn[:-1], xn[-1]), t)
    return xn, tn


def continuation(F, JF, x0, direction, h, n_max, s_bounds=(0.05, 60.0), stop_after_fold=True, tol=1e-11, h_min=1e-7):
    """Pseudo-arclength continuation from a solution x0 = (u0, s0) in the direction of increasing (+1) or decreasing
    (−1) s.  Returns the list of points and the refined folds (where the tangent's s-component changes sign)."""
    t = tangent(JF(x0[:-1], x0[-1]))
    if t[-1] * direction < 0:
        t = -t
    pts = [_record(F, JF, x0, t)]
    folds = []
    x = x0.copy()
    hh = h
    while len(pts) < n_max:
        r = step(F, JF, x, t, hh, tol)
        if r is None:
            hh /= 2
            if hh < h_min:
                break
            continue
        xn, tn = r
        if tn[-1] * t[-1] < 0:                                   # turning point in s between x and xn
            fx = refine_fold(F, JF, x, t, hh, tol)
            folds.append(fx)
            pts.append(_record(F, JF, xn, tn))
            x, t = xn, tn
            if stop_after_fold:
                break
            continue
        pts.append(_record(F, JF, xn, tn))
        x, t = xn, tn
        hh = min(h, hh * 1.5)
        if not (s_bounds[0] <= x[-1] <= s_bounds[1]):
            break
    return pts, folds


def refine_fold(F, JF, x, t, h, tol=1e-11, iters=60):
    """Bisect the step length in (0, h) for the point where the tangent's s-component is zero."""
    lo, hi = 0.0, h
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = step(F, JF, x, t, mid, tol)
        if r is None:
            hi = mid
            continue
        xm, tm = r
        best = (xm, tm)
        if tm[-1] * t[-1] > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    xm, tm = best
    rec = _record(F, JF, xm, tm)
    return {"s_fold": rec["s"], "ts": rec["ts"], "eig_min": rec["eig_min"], "eigs": rec["eigs"], "x": xm.tolist()}


# ------------------------------------------------------------------------------------------ the reduced landscape
class Reduced:
    """The weight-decayed loss restricted to the active units A (gauge unit g), joint in b."""

    def __init__(self, n_active, X, y, lam=LAM):
        self.torch = _torch()
        self.nA = n_active; self.lam = lam
        self.Xt = self.torch.as_tensor(X, dtype=self.torch.float64); self.Yt = self.torch.as_tensor(y, dtype=self.torch.float64)
        self.m = 2 * n_active + n_active + (n_active - 1) + 1

    def split(self, u):
        n = self.nA; T = self.torch
        W = u[:2 * n].reshape(n, 2); c = u[2 * n:3 * n]
        eta = T.cat([T.ones(1, dtype=u.dtype), u[3 * n:4 * n - 1]])
        b = u[4 * n - 1]
        return W, c, eta, b

    def loss_t(self, u, s):
        T = self.torch
        W, c, eta, b = self.split(u)
        v = eta ** 2 / (eta ** 2).sum()
        z = s * (T.tanh(self.Xt @ W.T + c) @ v) + b
        return T.nn.functional.binary_cross_entropy_with_logits(z, self.Yt) + 0.5 * self.lam * ((W ** 2).sum() + (c ** 2).sum())

    def F(self, u, s):
        T = self.torch
        ut = T.tensor(u, dtype=T.float64, requires_grad=True)
        return T.autograd.grad(self.loss_t(ut, T.tensor(float(s), dtype=T.float64)), ut)[0].numpy()

    def JF(self, u, s):
        T = self.torch

        def G(x):
            return T.autograd.grad(self.loss_t(x[:-1], x[-1]), x, create_graph=True)[0][:-1]
        x = T.tensor(np.r_[u, s], dtype=T.float64)
        return T.autograd.functional.jacobian(G, x).numpy()

    def loss(self, u, s):
        T = self.torch
        return float(self.loss_t(T.tensor(u, dtype=T.float64), T.tensor(float(s), dtype=T.float64)))

    def to_full(self, u):
        """16-vector of the v1/v2 landscape parametrisation (idle units zero) and b."""
        n = self.nA
        W = u[:2 * n].reshape(n, 2); c = u[2 * n:3 * n]; eta = np.r_[1.0, u[3 * n:4 * n - 1]]
        P = np.zeros(16)
        Wf = np.zeros((4, 2)); cf = np.zeros(4); ef = np.zeros(4)
        Wf[:n] = W; cf[:n] = c; ef[:n] = eta / np.linalg.norm(eta)
        P[:8] = Wf.ravel(); P[8:12] = cf; P[12:] = ef
        return P, float(u[-1])

    @staticmethod
    def from_full(p, s, X, y):
        """Reduced u from a 16-vector (active units = ṽ > ACTIVE_TOL; gauge = the largest), b profiled."""
        W, c, eta = sb.unpack(np.asarray(p, float)[None]); W, c = W[0], c[0]
        v = sb.vtilde(eta)[0]
        A = [k for k in np.argsort(-v) if v[k] > ACTIVE_TOL]
        g = A[0]
        etaA = np.sqrt(v[A] / v[g])
        b = float(v2.profile_b_batch(s * sb.phi(np.asarray(p, float)[None], X), y)[0])
        u = np.r_[W[A].ravel(), c[A], etaA[1:], b]
        return u, len(A)


def phi_unit(P, X):
    return sb.phi(np.atleast_2d(P), X)


def fdist(P, Q, X):
    """Function-space distance between unit-ℓ₁ network functions (invariant to unit permutations and sign flips)."""
    a = phi_unit(P, X); b = phi_unit(Q, X)
    return np.sqrt(((a - b) ** 2).mean(axis=1))


# ------------------------------------------------------------------------------------------ branches
def _v2_point(fam, s):
    return json.loads((V2_PARTS / f"{fam}_{s:.6f}.json").read_text())


BRANCH_STARTS = {"M": ("A", 3.58512), "S": ("A", 3.597642), "L0": ("A", 0.5), "S2": ("A", 9.094947)}


def run_branches(h=0.05, names=None):
    sb._init_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y = v2.data()
    f = OUT / "branches.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for name, (fam, s0) in BRANCH_STARTS.items():
        if names and name not in names:
            continue
        r = _v2_point(fam, s0)
        u0, nA = Reduced.from_full(np.array(r["p"]), s0, X, y)
        R = Reduced(nA, X, y)
        u0, ok, fres = newton_fixed_s(R.F, R.JF, u0, s0)
        x0 = np.r_[u0, s0]
        out = {"start": {"set": fam, "s": s0, "n_active": nA, "newton_ok": ok, "res": fres,
                         "loss": R.loss(u0, s0), "v2_loss": r["loss"]}}
        for d in (+1, -1):
            runs = {}
            for hh in (h, h / 2, h / 4):
                t0 = time.time()
                pts, folds = continuation(R.F, R.JF, x0, d, hh, n_max=int(40 / hh))
                runs[str(hh)] = {"folds": folds, "n_points": len(pts), "s_end": pts[-1]["s"],
                                 "seconds": time.time() - t0}
                if hh == h:
                    np.savez(OUT / f"branch_{name}_{'fwd' if d > 0 else 'bwd'}.npz",
                             x=np.array([p["x"] for p in pts]), t=np.array([p["t"] for p in pts]),
                             s=np.array([p["s"] for p in pts]), eig_min=np.array([p["eig_min"] for p in pts]),
                             eigs=np.array([p["eigs"] for p in pts]), n_active=nA)
                print(json.dumps({"branch": name, "dir": d, "h": hh, "n": len(pts), "s_end": pts[-1]["s"],
                                  "folds": [{k: f[k] for k in ("s_fold", "ts", "eig_min")} for f in folds]}), flush=True)
            out["fwd" if d > 0 else "bwd"] = runs
        res[name] = out
    (OUT / "branches.json").write_text(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------------------------------ branch tables
def load_branch(name):
    """Stored continuation points of a branch (backward reversed + forward), with its Reduced object."""
    X, y = v2.data()
    parts = []
    for d in ("bwd", "fwd"):
        z = np.load(OUT / f"branch_{name}_{d}.npz")
        xs, ev = z["x"], z["eig_min"]
        if d == "bwd":
            xs, ev = xs[::-1], ev[::-1]
        parts.append((xs, ev))
        nA = int(z["n_active"])
    xs = np.vstack([parts[0][0], parts[1][0][1:]]); ev = np.r_[parts[0][1], parts[1][1][1:]]
    return xs, ev, Reduced(nA, X, y)


def branch_point(name, s, cache={}):
    """Newton at fixed s from the nearest stored STABLE point of the branch; None if s is outside its stable range or
    Newton fails."""
    if name not in cache:
        xs, ev, R = load_branch(name)
        st = ev > 0
        cache[name] = (xs[st], R, xs[st][:, -1].min(), xs[st][:, -1].max())
    xs, R, lo, hi = cache[name]
    if not lo <= s <= hi:
        return None
    k = int(np.argmin(np.abs(xs[:, -1] - s)))
    u, ok, _ = newton_fixed_s(R.F, R.JF, xs[k, :-1], s)
    if not ok:
        return None
    return u, R


def stability_changes(s, ev):
    """Scales where the smallest reduced-Hessian eigenvalue changes sign along the stored points (linear interpolation)."""
    out = []
    for i in range(len(ev) - 1):
        if (ev[i] > 0) != (ev[i + 1] > 0):
            w = ev[i] / (ev[i] - ev[i + 1])
            out.append(float(s[i] + w * (s[i + 1] - s[i])))
    return out


def run_tables():
    import pandas as pd
    X, y = v2.data()
    rows = []
    summ = {}
    for name in BRANCH_STARTS:
        xs, ev, R = load_branch(name)
        for x, e in zip(xs, ev):
            P, b = R.to_full(x[:-1])
            fu = sb.feature_usage(P, X)
            rows.append({"branch": name, "s": x[-1], "loss": R.loss(x[:-1], x[-1]), "rho2": fu["rho2"], "eig_min": e,
                         "gplus": float(sb.gplus(P[None], X, y)[0]), "stable": bool(e > 0)})
        st = xs[ev > 0, -1]
        summ[name] = {"n_active": R.nA, "s_range": [float(xs[:, -1].min()), float(xs[:, -1].max())],
                      "stable_s_range": [float(st.min()), float(st.max())] if len(st) else None,
                      "eig_sign_changes": stability_changes(xs[:, -1], ev)}
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "branch_points.csv", index=False)
    # global comparison on a common grid (stable parts only)
    grid = np.round(np.arange(0.35, 12.0001, 0.01), 4)
    comp = []
    for s in grid:
        row = {"s": s}
        for name in BRANCH_STARTS:
            r = branch_point(name, s)
            row[name] = None if r is None else r[1].loss(r[0], s)
        comp.append(row)
    cf = pd.DataFrame(comp)
    cf.to_csv(OUT / "branch_losses.csv", index=False)
    L = cf[list(BRANCH_STARTS)].astype(float)
    best = L.idxmin(axis=1)
    switches = [{"s_between": [float(grid[i]), float(grid[i + 1])], "from": best[i], "to": best[i + 1]}
                for i in range(len(grid) - 1) if best[i] != best[i + 1]]
    summ["global_min_switches_on_grid"] = switches
    (OUT / "branch_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


# ------------------------------------------------------------------------------------------ validation near the fold
def local_min_batch(P0, s_rows, X, y, lam=LAM, gtol=1e-8, maxit=3000):
    """Full 16-coordinate local minimisation (v2 objective, b profiled) of each row at its own scale."""
    from .width2_conditional import bfgs_batch
    s_rows = np.asarray(s_rows, float)[:, None]
    P, L, G, it = bfgs_batch(lambda Q, rows: v2.loss_grad(Q, s_rows[rows], X, y, lam), np.asarray(P0, float),
                             gtol=gtol, maxit=maxit)
    return P, L, np.abs(G).max(axis=1)


def fold_of(name="M"):
    return json.loads((OUT / "branches.json").read_text())[name]["fwd"]["0.05"]["folds"][0]["s_fold"]


def run_validate(n_starts=30, seed=0):
    sb._init_worker()
    X, y = v2.data()
    rng = np.random.default_rng(seed)
    s_f = fold_of("M")
    res = {"s_fold": s_f}
    xs, ev, R = load_branch("M")
    k = int(np.argmax(xs[:, -1])); Pf, _ = R.to_full(xs[k, :-1])
    for tag, s in (("below_1pct", s_f * 0.99), ("below_0.1pct", s_f * 0.999), ("above_0.1pct", s_f * 1.001),
                   ("above_1pct", s_f * 1.01)):
        bp = branch_point("M", s)
        Pb = R.to_full(bp[0])[0] if bp is not None else Pf
        base = Pb if bp is not None else Pf
        noise = rng.normal(0, 1, (n_starts, 16)) * (0.05 * np.abs(base) + 0.02)
        P0 = base[None] + noise
        P0[:, 12:] = np.abs(P0[:, 12:])
        P, L, gn = local_min_batch(P0, np.full(n_starts, s), X, y)
        d_branch = fdist(P, (Pb if bp is not None else Pf)[None], X)
        rho = [sb.feature_usage(p, X)["rho2"] for p in P]
        newton = None
        if bp is None:
            u, ok, r = newton_fixed_s(R.F, R.JF, xs[k, :-1], s)
            newton = {"converged": ok, "residual": r}
        res[tag] = {"s": s, "branch_point_exists": bp is not None, "newton_from_fold_point": newton,
                    "n_on_branch": int((d_branch <= DIST_TOL).sum()), "n_starts": n_starts,
                    "fdist_min": float(d_branch.min()), "fdist_median": float(np.median(d_branch)),
                    "rho2_of_minima": [float(np.min(rho)), float(np.median(rho)), float(np.max(rho))],
                    "loss_min": float(L.min()), "gnorm_max": float(gn.max())}
        print(json.dumps({tag: res[tag]}), flush=True)
    # full-space Hessian (16 coordinates incl. idle units, η-radial projected) at M points
    hs = []
    for s in (1.2, 2.0, 3.0, 3.5914, 4.0, 4.5, 4.7, 4.76):
        bp = branch_point("M", s)
        P = R.to_full(bp[0])[0]
        hs.append({"s": s, "full_hess_min_eig": v2.hessian_min_eig(P, s, X, y, LAM),
                   "reduced_eig_min": float(np.linalg.eigvalsh(0.5 * (R.JF(bp[0], s)[:, :-1] + R.JF(bp[0], s)[:, :-1].T))[0])})
    res["hessian_path"] = hs
    print(json.dumps(hs), flush=True)
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------------------------------ the v3 runs' branches
GRID_DS = 0.001
S_MAX_RUNS = 11.5


def run_branch_grids():
    """Each branch on a grid of Δs = 0.001 over its stable range ∩ [0.3, 11.5] (Newton warm-started along the grid)."""
    X, y = v2.data()
    for name in BRANCH_STARTS:
        xs, ev, R = load_branch(name)
        st = xs[ev > 0]
        lo = max(0.3, st[:, -1].min()); hi = min(S_MAX_RUNS, st[:, -1].max())
        grid = np.round(np.arange(math.ceil(lo / GRID_DS) * GRID_DS, hi, GRID_DS), 6)
        k = int(np.argmin(np.abs(st[:, -1] - grid[0])))
        u = st[k, :-1]
        Ps, ok_all, ss, Us = [], [], [], []
        for s in grid:
            k = int(np.argmin(np.abs(st[:, -1] - s)))
            u_try, ok, _ = newton_fixed_s(R.F, R.JF, u, s)
            if not ok:
                u_try, ok, _ = newton_fixed_s(R.F, R.JF, st[k, :-1], s)
            if ok:
                u = u_try
                Ps.append(R.to_full(u)[0]); ss.append(s); Us.append(u.copy())
            ok_all.append(ok)
        Ps = np.array(Ps)
        step = fdist(Ps[1:], Ps[:-1], X) if len(Ps) > 1 else np.zeros(0)
        np.savez(OUT / f"grid_{name}.npz", s=np.array(ss), P=Ps, U=np.array(Us), n_active=R.nA)
        print(json.dumps({"branch": name, "n_neighbour_fdist_above_5e-4": int((step > 5e-4).sum()),
                          "s_where_above_5e-4": [float(np.array(ss)[1:][step > 5e-4].min()),
                                                 float(np.array(ss)[1:][step > 5e-4].max())] if (step > 5e-4).any() else None}))
        print(json.dumps({"branch": name, "n": len(ss), "newton_failures": int(len(ok_all) - sum(ok_all)),
                          "s_range": [ss[0], ss[-1]], "max_fdist_between_grid_neighbours": float(step.max())}), flush=True)


def run_to_full(params):
    """v3 run rows (W 8, c 4, v 4, b) → 16-vectors of the landscape parametrisation (signs absorbed, η = √(|v|/‖v‖₁))."""
    W = params[:, :8].reshape(-1, 4, 2).copy(); c = params[:, 8:12].copy(); v = params[:, 12:16]
    sg = np.where(v < 0, -1.0, 1.0)
    W *= sg[:, :, None]; c *= sg
    eta = np.sqrt(np.abs(v) / np.abs(v).sum(axis=1, keepdims=True))
    return np.concatenate([W.reshape(-1, 8), c, eta], axis=1)


def exact_branch_P(g, k, s):
    """Branch point at exactly s by Newton from grid point k (reduced coordinates); None if Newton fails."""
    gs, GP, GU, R = g
    u, ok, _ = newton_fixed_s(R.F, R.JF, GU[k], s)
    return R.to_full(u)[0] if ok else None


def classify_rows(P, s_rows, grids, X):
    """Labels per row: the branch whose point at the row's EXACT s lies within DIST_TOL (function space) of P; else
    'other'.  The nearest grid point (Δs = 0.001) is used first; rows within 10·DIST_TOL of it are re-measured
    against the branch point at the exact s (Newton from the grid point), which removes the grid error near folds."""
    lab = np.array(["other"] * len(P), dtype=object)
    dmin = np.full(len(P), np.inf)
    dists = {}
    for name, g in grids.items():
        gs, GP = g[0], g[1]
        k = np.clip(np.searchsorted(gs, s_rows), 1, len(gs) - 1)
        k = np.where(np.abs(gs[k - 1] - s_rows) < np.abs(gs[k] - s_rows), k - 1, k)
        inside = (s_rows >= gs[0] - GRID_DS / 2) & (s_rows <= gs[-1] + GRID_DS / 2)
        d = np.full(len(P), np.inf)
        if inside.any():
            d[inside] = fdist(P[inside], GP[k[inside]], X)
        for i in np.flatnonzero(d <= 10 * DIST_TOL):
            Pe = exact_branch_P(g, k[i], s_rows[i])
            d[i] = fdist(P[i:i + 1], Pe[None], X)[0] if Pe is not None else np.inf
        dists[name] = d
        better = (d <= DIST_TOL) & (d < dmin)
        lab[better] = name; dmin[better] = d[better]
    return lab, dists


def run_runs():
    import pandas as pd
    sb._init_worker()
    X, y = v2.data()
    grids = {}
    for name in BRANCH_STARTS:
        z = np.load(OUT / f"grid_{name}.npz")
        grids[name] = (z["s"], z["P"], z["U"], Reduced(int(z["n_active"]), X, y))
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for f in sorted(V3_RUNS.glob("run_*.npz")):
        seed = int(f.stem.split("_")[1])
        of = OUT / "runs" / f"steps_{seed}.csv"
        if of.exists():
            continue
        d = np.load(f)
        s = d["s"]; P0 = run_to_full(d["params"])
        t0 = time.time()
        Ploc, Lloc, gn = local_min_batch(P0, s, X, y)
        lab, dist = classify_rows(Ploc, s, grids, X)
        lab_raw, dist_raw = classify_rows(P0, s, grids, X)
        rho_loc = np.array([sb.feature_usage(p, X)["rho2"] for p in Ploc])
        df = pd.DataFrame({"step": np.arange(len(s)), "s": s, "basin": lab, "raw_on": lab_raw,
                           "loc_loss": Lloc, "loc_gnorm": gn, "loc_rho2": rho_loc,
                           **{f"d_basin_{k}": v for k, v in dist.items()}, **{f"d_raw_{k}": v for k, v in dist_raw.items()}})
        df.to_csv(of, index=False)
        print(json.dumps({"seed": seed, "steps": len(s), "seconds": time.time() - t0,
                          "basins": pd.Series(lab).value_counts().to_dict()}), flush=True)


# ------------------------------------------------------------------------------------------ fold constant (math note §14)
def fold_constant(name="M", s_fold=None, window=0.05):
    """λ_min of the reduced Hessian (P = I) on the Δs = 0.001 grid below the fold, and |m′c′| = ¼·|d(λ_min²)/ds| at the
    fold from a quadratic fit of λ_min² in (s_F − s) over the last `window` in s (and over half of it, as a check)."""
    X, y = v2.data()
    z = np.load(OUT / f"grid_{name}.npz")
    R = Reduced(int(z["n_active"]), X, y)
    s_F = fold_of(name) if s_fold is None else s_fold
    out = {"s_fold": s_F}
    sel = z["s"] >= s_F - window
    ss = z["s"][sel]; lam = []
    for u, s in zip(z["U"][sel], ss):
        J = R.JF(u, s)[:, :-1]
        lam.append(float(np.linalg.eigvalsh(0.5 * (J + J.T))[0]))
    lam = np.array(lam)
    for w in (window, window / 2):
        m = ss >= s_F - w
        d = s_F - ss[m]
        A = np.vstack([d, d ** 2]).T
        coef, *_ = np.linalg.lstsq(A, lam[m] ** 2, rcond=None)
        out[f"fit_window_{w}"] = {"d_lam2_ds_at_fold": float(-coef[0]), "abs_mc": float(abs(coef[0]) / 4),
                                  "n": int(m.sum())}
    out["lambda_min_path"] = [[float(a), float(b)] for a, b in zip(ss, lam)]
    return out


def run_fold_constants():
    res = {"M_upper": fold_constant("M"), "S_upper": fold_constant("S", fold_of("S"))}
    (OUT / "fold_constants.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "lambda_min_path"})


# ------------------------------------------------------------------------------------------ per-run comparison
def leave_scales(d):
    """From a run's per-step basin labels: s at the first step after the LAST step in the M basin (None if never in
    M); first and last s in M; and the first step in the S or S2 basin after that."""
    inM = np.flatnonzero(d.basin.values == "M")
    if not len(inM):
        return {"ever_M": False}
    last = inM[-1]
    nxt = last + 1 if last + 1 < len(d) else None
    return {"ever_M": True, "s_first_M": float(d.s.values[inM[0]]), "s_last_M": float(d.s.values[last]),
            "s_leave_M": None if nxt is None else float(d.s.values[nxt]),
            "basin_after_leave": None if nxt is None else str(d.basin.values[nxt]),
            "n_steps_M": int(len(inM))}


def run_compare():
    import pandas as pd
    torch = v2._torch()
    X, y = v2.data()
    s_f = fold_of("M")
    v3 = pd.read_csv(sb.RESULTS / "simplicity_bias_v3" / "runs_summary.csv").set_index("seed")
    rows = []
    for f in sorted((OUT / "runs").glob("steps_*.csv")):
        seed = int(f.stem.split("_")[1])
        d = pd.read_csv(f)
        z = np.load(V3_RUNS / f"run_{seed}.npz")
        rho = v2.rho2_batch(z["params"], X, torch)
        kf = int(np.argmax(z["s"] >= s_f)) if (z["s"] >= s_f).any() else None
        cs = v3.loc[seed, "s_cross"]
        grp = "non-crosser" if pd.isna(cs) else ("early" if cs < S_STAR else "late")
        dr = d[[c for c in d.columns if c.startswith("d_raw_")]].min(axis=1)
        L = leave_scales(d)
        rows.append({"seed": seed, "group": grp, "s_cross": cs, **L,
                     "leave_over_fold": None if L.get("s_leave_M") is None else L["s_leave_M"] / s_f,
                     "rho2_raw_at_fold": None if kf is None else float(rho[kf]),
                     "rho2_basin_min_at_fold": None if kf is None else float(d.loc_rho2.values[kf]),
                     "basin_at_fold": None if kf is None else str(d.basin.values[kf]),
                     "min_raw_dist_to_any_branch": float(dr.min()), "median_raw_dist_to_any_branch": float(dr.median()),
                     "frac_steps_raw_on_a_branch": float((d.raw_on != "other").mean()),
                     "basin_counts": json.dumps(d.basin.value_counts().to_dict())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_run.csv", index=False)
    summ = {"s_fold": s_f}
    for g in ("late", "early", "non-crosser", "all"):
        sub = df if g == "all" else df[df.group == g]
        lv = sub.leave_over_fold.dropna()
        summ[g] = {"n": int(len(sub)), "n_ever_M": int(sub.ever_M.sum()), "n_with_leave": int(len(lv)),
                   "median_leave_over_fold": float(lv.median()) if len(lv) else None,
                   "leave_over_fold_range": [float(lv.min()), float(lv.max())] if len(lv) else None,
                   "frac_leave_within_15pct_of_fold": float((abs(lv - 1) <= 0.15).mean()) if len(lv) else None,
                   "median_s_cross_over_fold": float((sub.s_cross / s_f).median()) if sub.s_cross.notna().any() else None,
                   "median_rho2_raw_at_fold": float(sub.rho2_raw_at_fold.median()),
                   "median_rho2_basin_min_at_fold": float(sub.rho2_basin_min_at_fold.median()),
                   "basin_at_fold_counts": sub.basin_at_fold.value_counts().to_dict(),
                   "median_min_raw_dist": float(sub.min_raw_dist_to_any_branch.median()),
                   "max_frac_steps_raw_on_a_branch": float(sub.frac_steps_raw_on_a_branch.max())}
    (OUT / "compare.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


# ------------------------------------------------------------------------------------------ Adam-preconditioned fold constant
def reconstruct_vhat(params, lr_betas=(0.9, 0.999)):
    """Adam's bias-corrected second moment for ALL 17 parameters at every recorded step, recomputed from the recorded
    parameters (full-batch gradients of the v3 training loss; exact up to float rounding).  Row t = after t updates."""
    torch = v2._torch()
    X, y = v2.data()
    Xt = torch.as_tensor(X, dtype=torch.float64); Yt = torch.as_tensor(y, dtype=torch.float64)
    b2 = lr_betas[1]
    m = np.zeros(17); out = [np.full(17, np.nan)]
    for t in range(1, len(params)):
        th = torch.tensor(params[t - 1], dtype=torch.float64, requires_grad=True)
        W = th[:8].reshape(4, 2); c = th[8:12]; v = th[12:16]; b = th[16]
        z = torch.tanh(Xt @ W.T + c) @ v + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Yt) + 0.5 * LAM * ((W ** 2).sum() + (c ** 2).sum())
        g = torch.autograd.grad(loss, th)[0].numpy()
        m = b2 * m + (1 - b2) * g ** 2
        out.append(m / (1 - b2 ** t))
    return np.array(out)


def run_precond_constant(window=0.05):
    import pandas as pd
    torch = v2._torch()
    X, y = v2.data()
    s_f = fold_of("M")
    Pblocks, sdots, checks = [], [], []
    for f in sorted(V3_RUNS.glob("run_*.npz")):
        z = np.load(f)
        if not (z["s"] >= s_f).any():
            continue
        k = int(np.argmax(z["s"] >= s_f))
        vh = reconstruct_vhat(z["params"][:k + 1])
        stored = z["vhat"][1:k + 1]; rec = np.concatenate([vh[1:, :12], vh[1:, 16:17]], axis=1)
        checks.append(float(np.max(np.abs(rec - stored) / np.abs(stored))))
        p = 1 / (np.sqrt(vh[k]) + 1e-8)
        Pblocks.append([np.median(p[:8]), np.median(p[8:12]), np.median(p[12:16]), p[16]])
        w = min(100, k)
        sdots.append((z["s"][k] - z["s"][k - w]) / w)
    Pb = np.median(np.array(Pblocks), axis=0)
    gz = np.load(OUT / "grid_M.npz"); nA = int(gz["n_active"])
    Xt = torch.as_tensor(X, dtype=torch.float64); Yt = torch.as_tensor(y, dtype=torch.float64)

    def H_train(u, s):
        n = nA
        W = u[:2 * n]; c = u[2 * n:3 * n]; eta = np.r_[1.0, u[3 * n:4 * n - 1]]; b = u[-1]
        v = s * eta ** 2 / (eta ** 2).sum()
        q0 = np.r_[W, c, v[:-1], b]

        def L(q):
            Wt = q[:2 * n].reshape(n, 2); ct = q[2 * n:3 * n]; vf = q[3 * n:4 * n - 1]
            vt = torch.cat([vf, (s - vf.sum()).reshape(1)]); bt = q[-1]
            zz = torch.tanh(Xt @ Wt.T + ct) @ vt + bt
            return torch.nn.functional.binary_cross_entropy_with_logits(zz, Yt) + 0.5 * LAM * ((Wt ** 2).sum() + (ct ** 2).sum())
        return torch.autograd.functional.hessian(L, torch.tensor(q0, dtype=torch.float64)).numpy()
    pdiag = np.r_[np.full(2 * nA, Pb[0]), np.full(nA, Pb[1]), np.full(nA - 1, Pb[2]), Pb[3]]
    sel = gz["s"] >= s_f - window
    ss = gz["s"][sel]; lamP, lamI = [], []
    for u, s in zip(gz["U"][sel], ss):
        H = H_train(u, s); ph = np.sqrt(pdiag)
        lamP.append(float(np.linalg.eigvalsh((ph[:, None] * H) * ph[None, :])[0]))
        lamI.append(float(np.linalg.eigvalsh(H)[0]))
    lamP, lamI = np.array(lamP), np.array(lamI)
    res = {"s_fold": s_f, "P_block_medians_W_c_v_b": Pb.tolist(), "n_runs": len(Pblocks),
           "vhat_reconstruction_max_rel_err": float(max(checks)), "median_sdot_at_fold": float(np.median(sdots)),
           "lr": 0.01}
    for tag, lam in (("adamP", lamP), ("identity_training_coords", lamI)):
        for w in (window, window / 2):
            m = ss >= s_f - w
            dd = s_f - ss[m]
            coef, *_ = np.linalg.lstsq(np.vstack([dd, dd ** 2]).T, lam[m] ** 2, rcond=None)
            res[f"{tag}_window_{w}"] = {"d_lam2_ds": float(-coef[0]), "abs_mc": float(abs(coef[0]) / 4)}
    mc = res[f"adamP_window_{window}"]["abs_mc"]
    Lam = math.sqrt(mc * s_f)
    eps = (res["median_sdot_at_fold"] / s_f) / (0.01 * Lam)
    res["fold_delay_law_post_hoc"] = {"Lambda_F": Lam, "eps_F": eps, "r_F": 2.338 * eps ** (2 / 3),
                                      "s_leave_pred": s_f * (1 + 2.338 * eps ** (2 / 3)),
                                      "note": "math note §14 constants read off POST HOC; eps_F >> 1 is outside the law's small-eps regime"
                                      if eps > 0.3 else "within small-eps regime"}
    res["lambda_min_path"] = {"s": ss.tolist(), "adamP": lamP.tolist(), "identity_training_coords": lamI.tolist()}
    (OUT / "fold_constant_adam.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "lambda_min_path"}, indent=1))


# ------------------------------------------------------------------------------------------ summary
def equal_loss(a, b, lo, hi, tol=1e-10):
    def diff(s):
        ra, rb = branch_point(a, s), branch_point(b, s)
        return ra[1].loss(ra[0], s) - rb[1].loss(rb[0], s)
    dl, dh = diff(lo), diff(hi)
    assert dl * dh < 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); dm = diff(mid)
        if dm * dl > 0:
            lo, dl = mid, dm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_summary():
    br = json.loads((OUT / "branches.json").read_text())
    bs = json.loads((OUT / "branch_summary.json").read_text())
    val = json.loads((OUT / "validation.json").read_text())
    cmp_ = json.loads((OUT / "compare.json").read_text())
    fc = json.loads((OUT / "fold_constants.json").read_text())
    fca = json.loads((OUT / "fold_constant_adam.json").read_text())
    folds = {}
    for name in BRANCH_STARTS:
        for d in ("fwd", "bwd"):
            fl = {h: (r["folds"][0]["s_fold"] if r["folds"] else None) for h, r in br[name][d].items()}
            folds[f"{name}_{d}"] = {"by_step": fl,
                                    "max_abs_change_under_halving": (max(fl.values()) - min(fl.values()))
                                    if all(v is not None for v in fl.values()) else None}
    s_f = fold_of("M")
    eq = {"L0_to_M": equal_loss("L0", "M", 1.48, 1.49), "M_to_S": equal_loss("M", "S", 3.58, 3.59),
          "S_to_S2": equal_loss("S", "S2", 8.61, 8.62)}
    late = cmp_["late"]
    gate_fold = bool(folds["M_fwd"]["max_abs_change_under_halving"] <= 1e-8
                     and val["below_1pct"]["n_on_branch"] > 0 and val["above_1pct"]["n_on_branch"] == 0
                     and not val["above_0.1pct"]["branch_point_exists"])
    match = abs(late["median_s_cross_over_fold"] - 1) <= 0.15
    summ = {"label": "POST HOC (Track 1A items 1-2; existing data, no new training)", "s_star_registered": S_STAR,
            "s_fold_M": s_f, "s_fold_over_s_star": s_f / S_STAR, "equal_loss_scales": eq,
            "s_fold_over_equal_loss_M_S": s_f / eq["M_to_S"], "folds": folds, "branch_summary": bs,
            "validation": {k: v for k, v in val.items() if k != "hessian_path"}, "hessian_path": val["hessian_path"],
            "fold_constant_identity": {k: v for k, v in fc["M_upper"].items() if k != "lambda_min_path"},
            "fold_constant_adam": {k: v for k, v in fca.items() if k != "lambda_min_path"},
            "runs": cmp_,
            "gate_1B": {"fold_exists_and_validated": gate_fold,
                        "late_median_s_cross_over_fold": late["median_s_cross_over_fold"],
                        "late_crossings_match_fold_within_15pct": bool(match),
                        "pass": bool(gate_fold and match)}}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: summ[k] for k in ("s_fold_M", "s_fold_over_s_star", "equal_loss_scales", "gate_1B")},
                     indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "branches":
        run_branches(names=sys.argv[2:] or None)
    elif cmd == "tables":
        run_tables()
    elif cmd == "validate":
        run_validate()
    elif cmd == "grids":
        run_branch_grids()
    elif cmd == "runs":
        run_runs()
    elif cmd == "foldconst":
        run_fold_constants()
    elif cmd == "compare":
        run_compare()
    elif cmd == "foldconst_adam":
        run_precond_constant()
    elif cmd == "summarise":
        run_summary()
