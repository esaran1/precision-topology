"""Item #10: is |w2| growth epsilon-dependent?  Fit |w2| ~ B^alpha * eps^-nu.

Registered in results/nu_prediction.md.  float64.

budget_alpha.csv has a SINGLE a value (1.25), so no eps-dependence of alpha could
ever have been detected from it.  This runs the 2D grid.

Estimator matched to the committed alpha: median terminal |w2| per cell.
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

RESULTS = Path(__file__).resolve().parents[1] / "results"
BUDGETS = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
A_VALUES = (1.05, 1.10, 1.20, 1.30, 1.45, 1.60)
N_SEEDS = 30
LR = 1e-2


def run(a, seed, budget):
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
    with torch.no_grad():
        w1, b1, w2, b2 = (float(v) for v in th)
    return {"a": a, "eps": a - 1.0, "budget": budget, "seed": seed,
            "w2_abs": abs(w2),
            "solved": bool(solves(torch.tensor([w1, b1, w2, b2],
                                               dtype=torch.float64), f))}


def main():
    rows = []
    for a in A_VALUES:
        for B in BUDGETS:
            for s in range(N_SEEDS):
                rows.append(run(a, s, B))
            g = pd.DataFrame([r for r in rows if r["a"] == a and r["budget"] == B])
            print(f"  a={a:.2f} B={B:>7} median|w2|={g.w2_abs.median():9.4f} "
                  f"solved={g.solved.mean():.3f}", flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "nu_sweep"
    with artifact_lock(stem, "nu sweep"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
    print("\nwritten results/nu_sweep.csv")


if __name__ == "__main__":
    main()
