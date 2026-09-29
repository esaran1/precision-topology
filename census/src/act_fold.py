"""POST HOC (coordinator / author request, 2026-09-28): fold or switch of the occupied branch, Track 3A training runs.

Question (author): "Do the same post hoc check for GELU's non-early crossers: is there an occupied branch whose fold or
switch matches the crossings?"  Hypothesis under test (Track 1A): a run stays on the branch it occupies past the global
switch while that branch remains a local minimum, and leaves it at the branch's fold (λ_min of the reduced Hessian → 0,
a turning point of the branch in s).  GELU is the target; SiLU and Mish are reported descriptively.  No new training;
the registered Track 3A verdicts stand.  Nothing here is registered.

Definitions (fixed before any computation on the runs):
- Runs: the registered 200-seed arm (train_*.csv + train_ext_*.csv), crossing and not placed at init (as the scoring).
  **Non-early**: s_cross >= 0.5·s_glob (s_glob = the validated bracket's geometric midpoint).  The early ones (GELU's
  tail-placed crossings) are counted, not analysed.
- Replay: act_general.run_one's protocol (U(−1, 1)⁴ float32 → double, Adam lr 0.01, full batch, own sample
  fold1d.make_data(200, seed)); the replay must reproduce the recorded |w₂| at the crossing step bit for bit.
- Pre-crossing state: the last step before the crossing with |w₂| <= 0.95·s_cross (sensitivity: 0.85·s_cross).
- Occupied branch: damped Newton (act_general.branch_point) in z = (w₁, b₁, b₂) at w₂ = σ·s_pre from that state, on
  the run's own 400-point sample; accepted iff max|∇| < 1e−8 and the Hessian is positive definite.
- Continuation: pseudo-arclength in X = (z, s), F(X) = ∇_z L(z; w₂ = σs) = 0, tangent = null vector of [H | ∂_sF],
  Newton corrector on [F; t·(X − X_pred)], adaptive step (h0 = 0.01, hmax = 0.05).  Upward (s increasing) from s_pre to
  2·s_cross (or a fold); if the branch is placed at s_pre, also downward (to find its switch below).
- **Fold**: a sign change of the tangent's s-component (turning point), located by bisection in arclength to 1e−10;
  λ_min(H) at the fold is recorded (should be ≈ 0).  λ_min changing sign without a turning point is reported as a
  branch point.
- **Switch**: a sign change of G₊(σ·u(w₁x + b₁)) (exact-extrema enclosure midpoint on the continuous windows) along the
  stable (pre-fold) branch, nearest to s_pre (upward if unplaced at s_pre, downward if placed), bisection to 1e−10.
- Match: per run ratio s_cross/s_event; summary = median ratio and the fraction with |ratio − 1| <= 0.15.
- Validation: (i) step halving (h0, hmax halved) gives the same event types and locations within 1e−6 relative;
  (ii) at every fold, an independent search (BFGS from 40 perturbations of the fold point, σ = 0.05) finds a local
  minimum within 0.1 of the branch just below the fold (s_fold·(1 − 1e−3)) and none within 0.1 of the fold point just
  above it (s_fold·(1 + 1e−3)); (iii) the λ_min path is positive on the stable segment.

    python -m src.act_fold run [act ...]      # one process, nice 15, memory watchdog
    python -m src.act_fold summarise
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
SRC = RESULTS / "act_general"
OUT = RESULTS / "act_fold"
ACTS = ("gelu", "silu", "mish")
PRE, PRE_SENS = 0.95, 0.85
H0, HMAX, HMIN = 0.01, 0.05, 1e-9
MATCH = 0.15


# ------------------------------------------------------------------------------------------ memory watchdog
def parse_memory(pressure_text, swap_text):
    """(free %, swap free MB) from `memory_pressure` and `sysctl vm.swapusage` outputs."""
    import re
    m = re.search(r"free percentage:\s*(\d+)%", pressure_text)
    s = re.search(r"free\s*=\s*([\d.]+)M", swap_text)
    return (int(m.group(1)) if m else -1), (float(s.group(1)) if s else -1.0)


def memory_ok(free_pct, swap_free_mb, min_free=25, min_swap=500):
    return free_pct >= min_free and swap_free_mb >= min_swap


def check_memory():
    p = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    s = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    f, w = parse_memory(p, s)
    return memory_ok(f, w), f, w


# ------------------------------------------------------------------------------------------ generic continuation
def tangent(J, t_prev=None):
    """Unit null vector of J (n × (n+1)), oriented along t_prev."""
    _, _, Vt = np.linalg.svd(J)
    t = Vt[-1] / np.linalg.norm(Vt[-1])
    if t_prev is not None and t @ t_prev < 0:
        t = -t
    return t


def correct(Xp, t, F, J, tol=1e-11, maxit=15):
    X = Xp.copy()
    for i in range(maxit):
        f = F(X)
        r = np.r_[f, t @ (X - Xp)]
        if np.abs(f).max() < tol and abs(r[-1]) < tol:
            return X, True, i
        try:
            dX = np.linalg.solve(np.vstack([J(X), t]), -r)
        except np.linalg.LinAlgError:
            return X, False, i
        X = X + dX
        if not np.all(np.isfinite(X)):
            return X, False, i
    return X, bool(np.abs(F(X)).max() < tol), maxit


def lam_min(J, X):
    n = len(X) - 1
    Hz = J(X)[:, :n]
    return float(np.linalg.eigvalsh(0.5 * (Hz + Hz.T)).min())


def _refine(X, t, h, F, J, fn, tol=1e-10):
    """Bisection in arclength step hh ∈ (0, h) from X along t for a sign change of fn(Xhh) (fn(X) and fn(X(h)) differ)."""
    lo, hi = 0.0, h
    f_lo = fn(X)
    Xm = X
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        Xm, ok, _ = correct(X + m * t, t, F, J)
        if not ok:
            break
        fm = fn(Xm)
        if np.sign(fm) == np.sign(f_lo):
            lo, f_lo = m, fm
        else:
            hi = m
    Xm, _, _ = correct(X + 0.5 * (lo + hi) * t, t, F, J)
    return Xm


def continue_branch(F, J, X0, direction=1, s_stop=None, gapf=None, h0=H0, hmax=HMAX, hmin=HMIN, max_steps=5000,
                    zmax=1e3):
    """Pseudo-arclength continuation of F(z, s) = 0 from X0 = (z0, s0) with s initially moving in `direction`.
    Stops at the first fold (turning point in s), at s beyond s_stop, at |z| > zmax, or if lost.  Records the first
    gap sign change (switch) on the way, λ_min along the path, and λ_min sign changes without a turn (branch points)."""
    X = np.asarray(X0, float)
    t = tangent(J(X))
    if t[-1] * direction < 0:
        t = -t
    lam = lam_min(J, X)
    g = gapf(X) if gapf else float("nan")
    path = [{"s": X[-1], "lam": lam, "G": g, "z": X[:-1].tolist()}]
    ev = {"fold": None, "switch": None, "branch_point": None, "end": None, "min_lam_stable": lam}
    h = h0
    for _ in range(max_steps):
        Xn, ok, it = correct(X + h * t, t, F, J)
        if not ok or np.linalg.norm(Xn - X) > 3 * h:
            h /= 2
            if h < hmin:
                ev["end"] = "lost"
                break
            continue
        tn = tangent(J(Xn), t)
        if np.sign(tn[-1]) != np.sign(t[-1]) and abs(t[-1]) > 0:
            tsign = np.sign(t[-1])
            Xf = _refine(X, t, h, F, J, lambda Y: tsign * tangent(J(Y), t)[-1])
            ev["fold"] = {"s": float(Xf[-1]), "z": Xf[:-1].tolist(), "lam": lam_min(J, Xf),
                          "G": gapf(Xf) if gapf else float("nan")}
            if gapf and ev["switch"] is None:
                gf = ev["fold"]["G"]
                if np.isfinite(gf) and np.isfinite(g) and np.sign(gf) != np.sign(g) and (gf > 0) != (g > 0):
                    Xs = _refine(X, t, h, F, J, gapf)
                    ev["switch"] = {"s": float(Xs[-1]), "z": Xs[:-1].tolist(), "lam": lam_min(J, Xs),
                                    "to_placed": bool(gf > 0)}
            ev["end"] = "fold"
            break
        lamn = lam_min(J, Xn)
        if np.sign(lamn) != np.sign(lam) and ev["branch_point"] is None:
            ev["branch_point"] = {"s": float(Xn[-1]), "lam_before": lam, "lam_after": lamn}
        gn = gapf(Xn) if gapf else float("nan")
        if gapf and ev["switch"] is None and np.isfinite(gn) and np.isfinite(g) and (gn > 0) != (g > 0):
            Xs = _refine(X, t, h, F, J, gapf)
            ev["switch"] = {"s": float(Xs[-1]), "z": Xs[:-1].tolist(), "lam": lam_min(J, Xs), "to_placed": bool(gn > 0)}
        X, t, lam, g = Xn, tn, lamn, gn
        if lam > 0:
            ev["min_lam_stable"] = min(ev["min_lam_stable"], lam)
        path.append({"s": X[-1], "lam": lam, "G": g, "z": X[:-1].tolist()})
        if it <= 3:
            h = min(1.5 * h, hmax)
        if s_stop is not None and (X[-1] - s_stop) * direction >= 0:
            ev["end"] = "s_stop"
            break
        if np.abs(X[:-1]).max() > zmax:
            ev["end"] = "z_runaway"
            break
    else:
        ev["end"] = "max_steps"
    return ev, path


def independent_fold_check(grad_loss, z_fold, s_fold, z_below, n=40, sigma=0.05, radius=0.1, seed=0, rel=1e-3):
    """grad_loss(z, s) -> (loss, grad).  BFGS from n perturbations of the fold point at s just below and just above the
    fold.  Pass iff below: some converged local minimum (grad < 1e−7) lies within `radius` of z_below; above: none lies
    within `radius` of z_fold."""
    from .width2_conditional import bfgs
    rng = np.random.default_rng(seed)
    z_fold = np.asarray(z_fold, float)

    def minima(s):
        out = []
        for _ in range(n):
            z0 = z_fold + sigma * rng.standard_normal(len(z_fold))
            z, f, gr, _ = bfgs(lambda q: grad_loss(q, s), z0, gtol=1e-9, maxit=3000)
            if np.all(np.isfinite(z)) and np.abs(gr).max() < 1e-7:
                out.append(z)
        return out
    below = minima(s_fold * (1 - rel)); above = minima(s_fold * (1 + rel))
    nb = sum(np.linalg.norm(z - np.asarray(z_below)) < radius for z in below)
    na = sum(np.linalg.norm(z - z_fold) < radius for z in above)
    return {"n_min_below_near_branch": int(nb), "n_min_above_near_fold": int(na), "ok": bool(nb > 0 and na == 0),
            "n_converged_below": len(below), "n_converged_above": len(above)}


# ------------------------------------------------------------------------------------------ the network problem
class Problem:
    """Own-sample width-1 loss L(z; s) = BCE(σs·u(w₁x + b₁) + b₂, y), z = (w₁, b₁, b₂), in torch double."""

    def __init__(self, name, seed, sigma):
        import torch
        from .act_general import GAct
        from .fold1d import make_data
        torch.set_num_threads(1)
        self.act = GAct(name)
        self.sigma = float(sigma)
        xt, yt = make_data(200, int(seed))
        self.X, self.Y = xt.double(), yt.double()
        self.x, self.y = self.X.numpy().astype(float), self.Y.numpy().astype(float)

    def _L(self, q):
        import torch
        return torch.nn.functional.binary_cross_entropy_with_logits(
            self.sigma * q[3] * self.act.torch_u(q[0] * self.X + q[1]) + q[2], self.Y)

    def F(self, X):
        import torch
        q = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        return torch.autograd.grad(self._L(q), q)[0].numpy()[:3]

    def J(self, X):
        import torch
        q = torch.tensor(X, dtype=torch.float64)
        return torch.autograd.functional.hessian(self._L, q).numpy()[:3, :]

    def loss_grad(self, z, s):
        import torch
        q = torch.tensor(np.r_[z, s], dtype=torch.float64, requires_grad=True)
        L = self._L(q)
        return float(L), torch.autograd.grad(L, q)[0].numpy()[:3]

    def gap(self, X):
        from .act_general import gplus
        g = gplus(X[0], X[1], self.sigma, self.act)
        return 0.5 * (g[0] + g[1]) if np.isfinite(g[0]) and np.isfinite(g[1]) else float("nan")


def replay(name, seed, step):
    """act_general.run_one's protocol without detection: θ at every step up to `step` (float64)."""
    import torch
    from torch.nn import functional as F
    from .act_general import GAct
    from .fold1d import make_data
    torch.set_num_threads(1)
    u = GAct(name).torch_u
    xt, yt = make_data(200, int(seed))
    torch.manual_seed(int(seed))
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=1e-2)
    X, Y = xt.double(), yt.double()
    traj = np.empty((int(step) + 1, 4))
    traj[0] = th.detach().numpy()
    for k in range(1, int(step) + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * u(th[0] * X + th[1]) + th[3], Y).backward()
        opt.step()
        traj[k] = th.detach().numpy()
    return traj


def pre_state(traj, step, s_cross, frac):
    s = np.abs(traj[:int(step), 2])
    idx = np.flatnonzero(s <= frac * s_cross)
    if not len(idx):
        return None
    k = int(idx[-1])
    return k, traj[k]


def _continuations_only(name, seed, step, s_cross, s_glob, frac, h0, hmax, P, X0, G0):
    up, _ = continue_branch(P.F, P.J, X0, +1, s_stop=2 * s_cross, gapf=P.gap, h0=h0, hmax=hmax)
    r = {"end_up": up["end"], "fold_s": up["fold"]["s"] if up["fold"] else np.nan}
    if G0 > 0:
        dn, _ = continue_branch(P.F, P.J, X0, -1, s_stop=0.05 * s_cross, gapf=P.gap, h0=h0, hmax=hmax)
        sw = dn["switch"]
    else:
        sw = up["switch"]
    r["switch_s"] = sw["s"] if sw else np.nan
    return r


def analyse_run(name, seed, step, s_cross, s_glob, frac=PRE, h0=H0, hmax=HMAX, validate=True):
    from .act_general import branch_point
    traj = replay(name, seed, step)
    out = {"act": name, "seed": int(seed), "s_cross": s_cross, "step": int(step), "frac_pre": frac,
           "reproduced": bool(abs(traj[int(step), 2]) == s_cross)}
    ps = pre_state(traj, step, s_cross, frac)
    if ps is None:
        return {**out, "note": "no pre-crossing step at this fraction"}
    k, th = ps
    sg = 1.0 if th[2] > 0 else -1.0
    s_pre = abs(th[2])
    P = Problem(name, seed, sg)
    z, gr = branch_point([th[0], th[1], th[3]], sg * s_pre, P.x, P.y, P.act)
    lam0 = lam_min(P.J, np.r_[z, s_pre])
    out.update({"pre_step": k, "s_pre": s_pre, "sigma": sg, "branch_grad": gr, "branch_lam": lam0,
                "dist_state_branch": float(np.linalg.norm(z - np.array([th[0], th[1], th[3]]))),
                "z_pre": json.dumps(z.tolist())})
    if gr > 1e-8 or lam0 <= 0:
        return {**out, "note": "no stable branch point at the pre-crossing state"}
    X0 = np.r_[z, s_pre]
    G0 = P.gap(X0)
    out["G_pre"] = G0
    up, path_up = continue_branch(P.F, P.J, X0, +1, s_stop=2 * s_cross, gapf=P.gap, h0=h0, hmax=hmax)
    out.update({"end_up": up["end"], "min_lam_stable_up": up["min_lam_stable"],
                "fold_s": up["fold"]["s"] if up["fold"] else np.nan,
                "fold_lam": up["fold"]["lam"] if up["fold"] else np.nan,
                "fold_G": up["fold"]["G"] if up["fold"] else np.nan,
                "branch_point_s": up["branch_point"]["s"] if up["branch_point"] else np.nan,
                "n_path_up": len(path_up)})
    sw = None
    if G0 > 0:
        dn, path_dn = continue_branch(P.F, P.J, X0, -1, s_stop=0.05 * s_cross, gapf=P.gap, h0=h0, hmax=hmax)
        out.update({"end_down": dn["end"], "lower_fold_s": dn["fold"]["s"] if dn["fold"] else np.nan})
        sw = dn["switch"]
        out["switch_dir"] = "down"
    else:
        sw = up["switch"]
        out["switch_dir"] = "up"
    out.update({"switch_s": sw["s"] if sw else np.nan, "switch_lam": sw["lam"] if sw else np.nan,
                "switch_to_placed": sw["to_placed"] if sw else None,
                "up_switch_s": up["switch"]["s"] if up["switch"] else np.nan})
    # event order on the upward pass (unplaced at s_pre): fold before switch?
    f_s, u_s = out["fold_s"], out["up_switch_s"]
    out["fold_before_switch"] = bool(np.isfinite(f_s) and (not np.isfinite(u_s) or f_s < u_s)) if G0 <= 0 else None
    if not validate:
        return out
    # (i) step halving
    v = _continuations_only(name, seed, step, s_cross, s_glob, frac, h0 / 2, hmax / 2, P, X0, G0)
    same = lambda a, b: (np.isnan(a) and np.isnan(b)) or (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-6 * abs(a))
    out.update({"halving_fold_s": v["fold_s"], "halving_switch_s": v["switch_s"],
                "halving_ok": bool(same(out["fold_s"], v["fold_s"]) and same(out["switch_s"], v["switch_s"])
                                   and v["end_up"] == out["end_up"])})
    # (ii) independent search at a fold; (iii) λ_min positive on the stable segment
    if up["fold"]:
        below = [p for p in path_up if p["s"] < up["fold"]["s"] * (1 - 1e-3)]
        Xb, _, _ = correct(np.r_[below[-1]["z"], up["fold"]["s"] * (1 - 1e-3)], np.r_[0, 0, 0, 1.0], P.F, P.J) \
            if below else (None, False, 0)
        chk = independent_fold_check(P.loss_grad, up["fold"]["z"], up["fold"]["s"], Xb[:3] if Xb is not None else up["fold"]["z"],
                                     seed=int(seed))
        out.update({f"fold_check_{k}": v_ for k, v_ in chk.items()})
    out["lam_path_ok"] = bool(up["min_lam_stable"] > 0 and (up["branch_point"] is None))
    return out




# ------------------------------------------------------------------------------------------ driver
def runs(name):
    import pandas as pd
    d = pd.concat([pd.read_csv(SRC / f"train_{name}.csv", float_precision="round_trip"),
                   pd.read_csv(SRC / f"train_ext_{name}.csv", float_precision="round_trip")], ignore_index=True)
    c = d[(d.crossed == True) & (d.placed_at_init == False)].copy()
    s_glob = json.loads((SRC / f"kappa_{name}_frozen.json").read_text())["s_glob"]
    c["non_early"] = c.s_cross >= 0.5 * s_glob
    return c, s_glob


def run(acts=ACTS):
    import pandas as pd
    os.nice(15)
    OUT.mkdir(exist_ok=True)
    for name in acts:
        c, s_glob = runs(name)
        for frac, tag in ((PRE, "pre95"), (PRE_SENS, "pre85")):
            f = OUT / f"runs_{name}_{tag}.csv"
            done = set() if not f.exists() else set(pd.read_csv(f).seed)
            for r in c[c.non_early].itertuples():
                if int(r.seed) in done:
                    continue
                ok, fp, sw = check_memory()
                if not ok:
                    raise SystemExit(f"STOP (memory watchdog): free {fp}%, swap free {sw} MB")
                row = analyse_run(name, int(r.seed), int(r.step), float(r.s_cross), s_glob, frac=frac,
                                  validate=(tag == "pre95"))
                pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
                print(json.dumps({k: row.get(k) for k in ("act", "seed", "reproduced", "G_pre", "fold_s", "switch_s",
                                                          "s_cross", "end_up", "halving_ok")}, default=float), flush=True)


def summarise():
    import pandas as pd
    out = {"label": "POST HOC (coordinator request 2026-09-28); registered Track 3A verdicts stand", "acts": {}}
    for name in ACTS:
        f = OUT / f"runs_{name}_pre95.csv"
        if not f.exists():
            continue
        c, s_glob = runs(name)
        d = pd.read_csv(f)
        ok = d[d.reproduced & d.note.isna()] if "note" in d else d[d.reproduced]
        rf = ok.s_cross / ok.fold_s; rs = ok.s_cross / ok.switch_s
        within = lambda r: float((np.abs(r[np.isfinite(r)] - 1) <= MATCH).mean()) if np.isfinite(r).any() else float("nan")
        sens = OUT / f"runs_{name}_pre85.csv"
        agree = float("nan")
        if sens.exists():
            m = d.merge(pd.read_csv(sens), on="seed", suffixes=("", "_85"))
            same = lambda a, b: np.where(np.isnan(a) & np.isnan(b), True, np.abs(a - b) <= 1e-6 * np.abs(a))
            agree = float(np.mean(same(m.switch_s.values, m.switch_s_85.values) & same(m.fold_s.values, m.fold_s_85.values)))
        a = {"crossing_runs": int(len(c)), "non_early": int(c.non_early.sum()), "early": int((~c.non_early).sum()),
             "analysed": int(len(d)), "reproduced": int(d.reproduced.sum()), "usable": int(len(ok)),
             "notes": d.note.value_counts().to_dict() if "note" in d else {},
             "placed_at_pre": int((ok.G_pre > 0).sum()),
             "with_fold": int(np.isfinite(ok.fold_s).sum()),
             "fold_before_switch": int((ok.fold_before_switch == True).sum()),
             "with_switch": int(np.isfinite(ok.switch_s).sum()),
             "median_ratio_cross_over_switch": float(np.nanmedian(rs)) if np.isfinite(rs).any() else float("nan"),
             "frac_within_15pct_switch": within(rs),
             "frac_within_1pct_switch": float((np.abs(rs[np.isfinite(rs)] - 1) <= 0.01).mean()) if np.isfinite(rs).any() else float("nan"),
             "median_ratio_cross_over_fold": float(np.nanmedian(rf)) if np.isfinite(rf).any() else float("nan"),
             "frac_within_15pct_fold": within(rf),
             "fold_over_switch_median": float(np.nanmedian(ok.fold_s / ok.switch_s)) if np.isfinite(ok.fold_s).any() else float("nan"),
             "halving_ok": int(d.halving_ok.sum()) if "halving_ok" in d else None,
             "lam_path_ok": int(d.lam_path_ok.sum()) if "lam_path_ok" in d else None,
             "fold_checks_ok": int(d.fold_check_ok.sum()) if "fold_check_ok" in d else 0,
             "end_up": d.end_up.value_counts().to_dict() if "end_up" in d else {},
             "sensitivity_pre85_same_events": agree}
        out["acts"][name] = a
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    {"run": lambda: run(tuple(sys.argv[2:]) or ACTS), "summarise": summarise}[sys.argv[1]]()
