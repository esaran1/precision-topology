"""Is the jump / cold_high stall SATURATION or TRAPPING?

Registered in results/blockE_stall_prediction.md.  Three measurements, all fixed
before any was inspected:

  1. gradient norms at matched steps, jump / cold_high / control
  2. final loss against the conditional minimum at the same |w2|
  3. placement rate at 5x the extra budget (60,000 steps)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .blockB_landscape import best_conditional, population_data
from .blockE_intervene import A, HIGH, LR, _scale_to
from .fold1d import activation, make_data
from .fold1d_theorem import maximum_gap
from .r_variable import oriented_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"
GRAD_AT = (100, 500, 2_000, 8_000, 11_999)
LONG = 60_000
N_LONG = 15


def trace(a, seed, arm, R_glob, gstar, budget=12_000, grad_at=GRAD_AT):
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    if arm == "jump":
        _scale_to(th, 1.05 * R_glob, gstar)
    elif arm == "cold_high":
        _scale_to(th, HIGH * R_glob, gstar)
    hold = abs(float(th[2].detach())) if arm == "cold_high" else None
    opt = torch.optim.Adam([th], lr=LR)
    grads = {}
    placed_at = None
    for i in range(budget):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(
            th[2] * f(th[0] * x + th[1]) + th[3], y)
        loss.backward()
        if i in grad_at:
            g = th.grad.detach()
            # the trained coordinates: w1,b1,b2 always; w2 too unless held
            idx = [0, 1, 3] if hold is not None else [0, 1, 2, 3]
            grads[i] = float(torch.linalg.norm(g[idx]))
        opt.step()
        if hold is not None:
            with torch.no_grad():
                v = float(th[2])
                if v != 0:
                    th[2] = torch.tensor(np.sign(v) * hold, dtype=torch.float64)
        if i % 50 == 0 and placed_at is None:
            with torch.no_grad():
                w1, b1, _, _ = (float(v) for v in th)
            if oriented_gap(f, w1, b1) > 0:
                placed_at = i
    with torch.no_grad():
        w1, b1, w2, b2 = (float(v) for v in th)
        final_loss = float(F.binary_cross_entropy_with_logits(
            torch.tensor(w2, dtype=torch.float64) * f(
                torch.tensor(w1, dtype=torch.float64) * x + b1) + b2, y))
    return {"a": a, "seed": seed, "arm": arm, "budget": budget,
            "placed_at": placed_at, "placed": placed_at is not None,
            "gap_final": oriented_gap(f, w1, b1),
            "w2_final": abs(w2), "R_final": abs(w2) * gstar / 2,
            "loss_final": final_loss,
            **{f"grad_{k}": v for k, v in grads.items()}}


def main():
    gstar = maximum_gap(A, resolution=600)
    R_glob = float(pd.read_csv(RESULTS / "blockB_switches.csv")
                   .query(f"a=={A}").R_glob.iloc[0])
    print(f"a={A}  R_glob={R_glob:.5f}  Ghat={gstar:.6f}", flush=True)

    print("\n=== 1+2. gradient norms and final loss, 40 seeds, 12,000 steps ===",
          flush=True)
    rows = []
    for arm in ("control", "jump", "cold_high"):
        for s in range(40):
            rows.append(trace(A, s, arm, R_glob, gstar))
        d = pd.DataFrame([r for r in rows if r["arm"] == arm])
        gs = "  ".join(f"{k}:{d[f'grad_{k}'].median():.2e}" for k in GRAD_AT)
        print(f"  {arm:10s} placed {d.placed.mean():.3f}  grad medians  {gs}", flush=True)
    frame = pd.DataFrame(rows)

    print("\n  final loss against the conditional minimum at the same |w2|:",
          flush=True)
    xp, yp = population_data()
    extra = []
    for arm in ("jump", "cold_high"):
        sub = frame[(frame.arm == arm) & (~frame.placed)]
        for _, r in sub.head(12).iterrows():
            b = best_conditional(A, float(r.w2_final), xp, yp, restarts=24)
            extra.append({"arm": arm, "seed": int(r.seed),
                          "w2": float(r.w2_final), "R": float(r.R_final),
                          "loss_run": float(r.loss_final),
                          "loss_min": None if b is None else b["loss"],
                          "excess": None if b is None else float(r.loss_final) - b["loss"],
                          "gap_min": None if b is None else b["gap"]})
        e = pd.DataFrame([z for z in extra if z["arm"] == arm])
        ex = e.excess.dropna()
        print(f"    {arm:10s} n={len(ex)}  median excess = {ex.median():+.6f}  "
              f"(positive and > 0.005 => trapping)", flush=True)
    pd.DataFrame(extra).to_csv(RESULTS / "blockE_stall_loss.csv", index=False)

    print(f"\n=== 3. extended budget: {LONG} steps, {N_LONG} seeds ===", flush=True)
    long_rows = []
    for arm in ("jump", "cold_high"):
        for s in range(N_LONG):
            long_rows.append(trace(A, s, arm, R_glob, gstar, budget=LONG,
                                   grad_at=GRAD_AT))
        d = pd.DataFrame([r for r in long_rows if r["arm"] == arm])
        short = frame[(frame.arm == arm) & (frame.seed < N_LONG)]
        print(f"  {arm:10s} placed {int(d.placed.sum())}/{len(d)} at {LONG} steps  "
              f"vs {int(short.placed.sum())}/{len(short)} at 12,000", flush=True)
    lf = pd.DataFrame(long_rows)

    stem = RESULTS / "blockE_stall"
    with artifact_lock(stem, "blockE stall"):
        pd.concat([frame, lf]).to_csv(stem.with_suffix(".csv"), index=False)
    print("\nwritten blockE_stall.csv and blockE_stall_loss.csv")


if __name__ == "__main__":
    main()
