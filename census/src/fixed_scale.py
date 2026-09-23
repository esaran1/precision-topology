"""Blocks 4 and 5: fixed-scale replays from checkpoints with full optimiser state.

Training (shared): a = 1.30, fresh seeds (SEED_OFFSET + s), fold1d.make_data(200, seed), float64,
Adam lr 1e-2, initialisation torch.manual_seed(seed); U(-1,1)^4 (the phase 2b protocol).  Full state
(θ = (w1, b1, w2, b2), Adam exp_avg, exp_avg_sq, step) saved at log-spaced steps and at the first
placement (oriented dense-grid gap > 0, the phase 2b crossing definition, checked every step).

Replay: from a checkpoint, jointly rescale (w2, b2) by k > 0 — every decision sign(N(x)) is unchanged at
the moment of intervention — so that R/R_glob lands on a target level (R = |w2| Ĝ/2, certified Ĝ and
R_glob); then optimise (w1, b1, b2) with |w2| held fixed (w2 excluded from the optimiser) for a
registered horizon.  Optimiser state variants:
  "preserved": the checkpoint's Adam moments and step count for w1, b1, b2 (the b2 moments are kept as
               saved, not rescaled; w2's moments are dropped since w2 is frozen);
  "reset":     fresh Adam (zero moments, step 0).
Recorded: ΔG of the first update, G(t) every 25 steps, first step with G > 0 (Block 4) or with G <= 0
(Block 5), gradient norm of (w1, b1, b2) at the start and end, fraction of training points with
saturated logits (|σ(z) - y| < 1e-3) at the start, and the distance in (w1, b1) to the certified
conditional branch argmin at the held scale (Block 1c).

    python -m src.fixed_scale train N            # N fresh runs with full-state checkpoints
    python -m src.fixed_scale replay4            # Block 4 (before-placement checkpoints)
    python -m src.fixed_scale replay5            # Block 5 (just-after-placement checkpoints)
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
STATE_DIR = RESULTS / "fixed_scale_states"
A = 1.30
LR = 1e-2
SEED_OFFSET = 200_000
BUDGET = 12_000
LEVELS = (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
WORKERS = int(os.environ.get("FS_WORKERS", "4"))


def _ghat_and_glob():
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv")
    G = float(g[g.a.round(2) == A].Ghat_certified.iloc[0])
    t = pd.read_csv(RESULTS / "cond_certified_thresholds.csv")
    r = t[t.a.round(2) == A].iloc[0]
    w2_glob = 0.5 * (r.w2_glob_lo + r.w2_glob_hi)
    return G, w2_glob, w2_glob * G / 2


def _gap(f, w1, b1, w2):
    from .blockB_landscape import gap_of
    return gap_of(f, w1, b1, w2)


def train_one(seed):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, make_data
    torch.set_num_threads(1)
    f = activation("sin_family", A)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    marks = set(np.unique(np.logspace(0, np.log10(BUDGET), 60).astype(int)))
    states, placed_at = [], None
    for step in range(1, BUDGET + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
        w1, b1, w2, b2 = (float(v) for v in th.detach())
        G = _gap(f, w1, b1, w2)
        first = placed_at is None and G > 0
        if first:
            placed_at = step
        if step in marks or first:
            st = opt.state[th]
            states.append({"seed": seed, "step": step, "theta": th.detach().clone().numpy(),
                           "exp_avg": st["exp_avg"].clone().numpy(),
                           "exp_avg_sq": st["exp_avg_sq"].clone().numpy(),
                           "adam_step": int(st["step"]), "G": G, "first_placement": first,
                           "placed_before": placed_at is not None and not first})
        if placed_at is not None and step >= placed_at and step not in marks and not first:
            pass
    return states


def train(n):
    STATE_DIR.mkdir(exist_ok=True)
    seeds = [SEED_OFFSET + s for s in range(int(n))]
    with Pool(WORKERS) as p:
        out = p.map(train_one, seeds)
    allst = [s for ss in out for s in ss]
    with open(STATE_DIR / "checkpoints.pkl", "wb") as fh:
        pickle.dump(allst, fh)
    meta = pd.DataFrame([{k: v for k, v in s.items() if k not in ("theta", "exp_avg", "exp_avg_sq")}
                         | {"w1": s["theta"][0], "b1": s["theta"][1], "w2": s["theta"][2], "b2": s["theta"][3]}
                         for s in allst])
    meta.to_csv(RESULTS / "fixed_scale_checkpoints.csv", index=False)
    print(meta.groupby("seed").first_placement.any().value_counts().to_string())


def replay(ck, level, variant, horizon, stop_on, G_hat, w2_glob, branch=None, x=None, y=None):
    """Rescale to R/R_glob = level, hold |w2|, train (w1, b1, b2) for `horizon` steps."""
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, make_data
    f = activation("sin_family", A)
    if x is None:
        x, y = make_data(200, ck["seed"])
        x, y = x.double(), y.double()
    w1, b1, w2, b2 = (float(v) for v in ck["theta"])
    k = level * w2_glob / abs(w2)
    w2n, b2n = w2 * k, b2 * k
    p = torch.tensor([w1, b1, b2n], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p], lr=LR)
    if variant == "preserved":
        opt.state[p] = {"step": torch.tensor(float(ck["adam_step"])),
                        "exp_avg": torch.tensor(ck["exp_avg"][[0, 1, 3]], dtype=torch.float64),
                        "exp_avg_sq": torch.tensor(ck["exp_avg_sq"][[0, 1, 3]], dtype=torch.float64)}
    w2t = torch.tensor(w2n, dtype=torch.float64)

    def lossf():
        z = w2t * f(p[0] * x + p[1]) + p[2]
        return F.binary_cross_entropy_with_logits(z, y), z
    with torch.no_grad():
        _, z0 = lossf()
        sat0 = float(((torch.sigmoid(z0) - y).abs() < 1e-3).double().mean())
    G0 = _gap(f, w1, b1, w2n)
    traj, first_hit, g_start, dG1 = [], None, None, None
    for t in range(1, horizon + 1):
        opt.zero_grad(set_to_none=True)
        L, _ = lossf()
        L.backward()
        if t == 1:
            g_start = float(p.grad.norm())
        opt.step()
        if t == 1 or t % 25 == 0 or t == horizon:
            G = _gap(f, float(p[0]), float(p[1]), w2n)
            if t == 1:
                dG1 = G - G0
            traj.append((t, G))
            hit = (G > 0) if stop_on == "place" else (G <= 0)
            if first_hit is None and hit:
                first_hit = t
    opt.zero_grad(set_to_none=True)
    L, _ = lossf()
    L.backward()
    Gend = traj[-1][1]
    dist = (math.hypot(float(p[0]) - branch[0], float(p[1]) - branch[1])
            if branch is not None else np.nan)
    return {"seed": ck["seed"], "ck_step": ck["step"], "level": level, "variant": variant,
            "k": k, "G_start": G0, "dG_first": dG1, "G_end": Gend, "first_hit": first_hit,
            "placed_end": Gend > 0, "grad_start": g_start, "grad_end": float(p.grad.norm()),
            "sat_start": sat0, "dist_branch_end": dist,
            "traj_G": ";".join(f"{t}:{g:.6g}" for t, g in traj)}
