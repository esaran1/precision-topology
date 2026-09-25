"""Track 3 (POST HOC, EXPLORATORY; 2026-09-25): does the residual scale with a timescale ratio?

For every crossing run of the deconfounded lag test (lag_test2, primary rule, a ∈ {1.30, 1.50}, φ ∈ {0.25, 0.5, 1, 2},
n = 6,400 training sets), replay from its saved checkpoint to its recorded crossing step (the replay must reproduce the
recorded |w₂| at crossing bit for bit), and record at the crossing:
  growth   d log s/dt: log(|w₂|(t_c)/|w₂|(t_c − 100))/100, per step;
  relax    lr·λ_min(D^{-1/2} H D^{-1/2}): H the Hessian of the joint loss in (w₁, b₁, b₂) at w₂ fixed, evaluated at the
           BRANCH (the local minimiser of that loss reached from the crossing state by damped Newton), and
           D = diag(√v̂ + eps) from Adam's second moments at the crossing (the linearised Adam relaxation rate);
  ratio    growth / relax (dimensionless).
Then Spearman(residual, ratio) with residual = R_cross/R_own − 1, pooled over both a and all φ arms (run level), with a
bootstrap 95% interval; and at the level of the 8 cells (medians).  No mechanism is claimed.

    python -m src.residual_timescale run     # one worker, nice 15
    python -m src.residual_timescale score
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "residual_timescale_runs.csv"
WINDOW = 100
LR, EPS, BETA2 = 1e-2, 1e-8, 0.999
CLEAR = 0.6


def branch_hessian(a, x, y, w1, b1, w2, b2):
    """Damped Newton on the joint loss in z = (w₁, b₁, b₂) at w₂ fixed; returns (z*, H(z*), grad max, converged)."""
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)

    def L(z):
        t = z[0] * X + z[1]
        return torch.nn.functional.binary_cross_entropy_with_logits(w2 * (t + a * torch.sin(t)) + z[2], Y)
    z = torch.tensor([w1, b1, b2], dtype=torch.float64)
    mu, f0 = 1e-3, float(L(z))
    for _ in range(200):
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(L(zz), zz)[0]
        if float(g.abs().max()) < 1e-12:
            break
        H = torch.autograd.functional.hessian(L, z)
        ok = False
        for _ in range(30):
            d = torch.linalg.solve(H + mu * torch.eye(3, dtype=torch.float64) * max(1.0, float(H.diag().abs().max())), -g)
            f1 = float(L(z + d))
            if f1 < f0:
                z, f0, ok, mu = z + d, f1, True, max(mu / 3, 1e-12)
                break
            mu *= 10
        if not ok:
            break
    zz = z.clone().requires_grad_(True)
    g = torch.autograd.grad(L(zz), zz)[0]
    H = torch.autograd.functional.hessian(L, z)
    return z.numpy(), H.numpy(), float(g.abs().max()), float(g.abs().max()) < 1e-8


def relax_rate(H, v_hat, eps=EPS, lr=LR):
    d = 1.0 / np.sqrt(np.sqrt(v_hat) + eps)
    M = (d[:, None] * H) * d[None, :]
    return lr * float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())


def replay_one(args):
    """Replay a lag_test2 primary-rule continuation to its crossing step; Adam state and |w₂| path at the crossing."""
    import torch
    from . import lag_test2 as lt
    a, seed, factor, cross_step, w2_ref = args
    f, G, xt, yt, _, _ = lt._setup(a, seed)
    ck = torch.load(lt.STATES / f"primary_a{a:.2f}_s{seed}.pt", weights_only=False)
    theta = ck["theta"].clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)
    opt.load_state_dict(ck["opt"])
    t = ck["t_star"]
    path = {}
    for step in range(t + 1, int(cross_step) + 1):
        lt._step(theta, opt, f, xt, yt, factor)
        if step >= cross_step - WINDOW:
            path[step] = abs(float(theta.detach()[2]))
    st = opt.state[theta]
    n_steps = int(st["step"])
    v_hat = (st["exp_avg_sq"].numpy() / (1 - BETA2 ** n_steps)).copy()
    th = theta.detach().numpy().copy()
    return {"theta": th, "v_hat": v_hat, "path": path, "reproduced": abs(th[2]) == w2_ref,
            "x": xt.numpy(), "y": yt.numpy()}


def _job(args):
    os.nice(15)
    a, seed, factor, cross_step, w2_ref, R_cross, R_own = args
    r = replay_one((a, seed, factor, cross_step, w2_ref))
    th, v = r["theta"], r["v_hat"]
    s1 = r["path"].get(int(cross_step)); s0 = r["path"].get(int(cross_step) - WINDOW)
    growth = math.log(s1 / s0) / WINDOW if s0 and s1 else np.nan
    z, H, gmax, conv = branch_hessian(a, r["x"].astype(float), r["y"].astype(float), th[0], th[1], th[2], th[3])
    lam = relax_rate(H, v[[0, 1, 3]])
    row = {"a": a, "seed": seed, "factor": factor, "cross_step": cross_step, "reproduced": r["reproduced"],
           "residual": R_cross / R_own - 1, "growth": growth, "relax": lam,
           "ratio": growth / lam if lam > 0 else np.nan, "branch_converged": conv, "branch_grad": gmax,
           "branch_dist": float(np.abs(z - np.array([th[0], th[1], th[3]])).max()),
           "hess_min_eig": float(np.linalg.eigvalsh(H).min())}
    print(json.dumps({"a": a, "seed": seed, "factor": factor, "ok": bool(r["reproduced"])}), flush=True)
    return row


def run(workers=1):
    from multiprocessing import get_context
    from .lag_test2 import _own
    d = pd.read_csv(RESULTS / "lag_test2_runs.csv", float_precision="round_trip")
    d = d[(d.rule == "primary") & d.cross_step.notna()]
    own = _own()
    done = set() if not OUT.exists() else {(round(r.a, 2), r.seed, r.factor) for r in pd.read_csv(OUT).itertuples()}
    todo = [(r.a, int(r.seed), r.factor, int(r.cross_step), r.w2_abs, r.R_cross, float(own.loc[(r.a, r.seed)]))
            for r in d.itertuples() if (round(r.a, 2), int(r.seed), r.factor) not in done]
    todo.sort(key=lambda t: t[3])                                   # short replays first
    with get_context("spawn").Pool(workers) as pool:
        for row in pool.imap_unordered(_job, todo):
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_spearman(a, b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    k = rng.integers(0, len(a), (n, len(a)))
    v = np.array([spearman(a[i], b[i]) for i in k])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def score():
    d = pd.read_csv(OUT)
    if not d.reproduced.all():
        raise SystemExit(f"STOP: {int((~d.reproduced).sum())} replays did not reproduce their crossing")
    ok = d[np.isfinite(d.ratio) & (d.ratio > 0)]
    rho = spearman(ok.residual, ok.ratio)
    lo, hi = boot_spearman(ok.residual.values, ok.ratio.values)
    cells = ok.groupby(["a", "factor"]).agg(residual=("residual", "median"), ratio=("ratio", "median"),
                                             n=("seed", "size")).reset_index()
    rc = spearman(cells.residual, cells.ratio)
    within = {f"a={a:.2f}": spearman(g.residual, g.ratio) for a, g in ok.groupby("a")}
    out = {"n_runs": len(d), "n_used": len(ok), "excluded_nonpositive_relax": int(len(d) - len(ok)),
           "branch_not_converged": int((~d.branch_converged).sum()), "spearman_run_level": rho,
           "spearman_ci95": [lo, hi], "spearman_cell_level": rc, "spearman_within_a": within,
           "clear_relationship_for_track4": bool(rho >= CLEAR)}          # the author's rule: Spearman >= 0.6, pooled
    cells.to_csv(RESULTS / "residual_timescale_cells.csv", index=False)
    (RESULTS / "residual_timescale_summary.json").write_text(json.dumps(out, indent=1))
    print(cells.to_string(index=False)); print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    {"run": lambda: run(int(sys.argv[2]) if len(sys.argv) > 2 else 1), "score": score}[sys.argv[1]]()
