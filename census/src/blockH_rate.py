"""Block H: growth-rate dose-response.  Is output scale an annealing parameter?

Registered in results/blockH_rate_prediction.md.  float64.

From a matched early checkpoint (step 400, before any run has placed), impose

    |w2|(t) = |w2|(400) + m * R_NAT * (t - 400)

while (w1, b1, b2) train normally.  |w2| is projected onto the schedule after
every step; its sign is never changed; b2 is NOT rescaled, because the Block E
`jump` arm showed rescaling b2 is separately harmful and holding it fixed
isolates the rate.

R_NAT = 0.002223 is the median local d|w2|/dstep over steps 400-6,000 on the 40
unconstrained control seeds, measured before any imposed arm was run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import activation, make_data, solves
from .fold1d_theorem import maximum_gap
from .r_variable import oriented_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"
A = 1.30
LR = 1e-2
STEPS = 12_000
START = 400          # matched checkpoint: before Block A's earliest crossing (1,860)
R_NAT = 0.002223     # median control growth rate, measured first
MULTS = (0.5, 1.0, 2.0, 4.0)
N_SEEDS = 40
CHECK_EVERY = 50


def run(a, seed, mult, gstar, steps=STEPS):
    """mult=None is the unconstrained control."""

    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    w2_start = None
    placed_at = None
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(
            th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
        if mult is not None:
            if i == START:
                w2_start = abs(float(th[2].detach()))
            if w2_start is not None and i >= START:
                target = w2_start + mult * R_NAT * (i - START)
                with torch.no_grad():
                    v = float(th[2])
                    th[2] = torch.tensor(np.sign(v) * target if v != 0 else target,
                                         dtype=torch.float64)
        if i % CHECK_EVERY == 0 and placed_at is None:
            with torch.no_grad():
                w1, b1, _, _ = (float(v) for v in th)
            if oriented_gap(f, w1, b1) > 0:
                placed_at = i
    with torch.no_grad():
        w1, b1, w2, b2 = (float(v) for v in th)
    return {"a": a, "seed": seed, "mult": mult,
            "placed_at": placed_at, "placed": placed_at is not None,
            "solved": bool(solves(torch.tensor([w1, b1, w2, b2],
                                               dtype=torch.float64), f)),
            "gap_final": oriented_gap(f, w1, b1),
            "w2_final": abs(w2), "R_final": abs(w2) * gstar / 2,
            "w2_at_start": w2_start}


def main():
    gstar = maximum_gap(A, resolution=600)
    print(f"a={A}  Ghat={gstar:.6f}  R_NAT={R_NAT}  start step {START}", flush=True)
    rows = []
    for mult in (None,) + MULTS:
        for s in range(N_SEEDS):
            rows.append(run(A, s, mult, gstar))
        d = pd.DataFrame([r for r in rows
                          if (r["mult"] is None) == (mult is None)
                          and r["mult"] == mult])
        tag = "control" if mult is None else f"m={mult}"
        pa = d[d.placed_at.notna()].placed_at
        print(f"  {tag:9s} placed {d.placed.mean():.3f} ({int(d.placed.sum()):2d}/{len(d)})  "
              f"solved {d.solved.mean():.3f} ({int(d.solved.sum()):2d}/{len(d)})  "
              f"median placed_at {pa.median() if len(pa) else float('nan'):.0f}  "
              f"R_final {d.R_final.median():.4f}", flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "blockH_rate"
    with artifact_lock(stem, "blockH rate"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
    print("\nwritten results/blockH_rate.csv")


if __name__ == "__main__":
    main()
