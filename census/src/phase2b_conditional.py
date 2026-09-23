"""Prediction of both thresholds, computed independently of unconstrained training trajectories,
from the slow-variable hypothesis.

Registered in results/phase2b_across_a_prediction.md BEFORE any crossing value
outside a = 1.30 was read.

If w2 is a slow variable and (w1, b1, b2) track the minimizer of the training
loss conditional on the current w2, then both thresholds are computable without
training: sweep |w2| on a grid, minimize the SAME loss on the SAME 400 points
over (w1, b1, b2) from many restarts, and find where the conditional minimizer
switches to G > 0 (R_cross) and where it first becomes sign-correct (R_solve).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, N_PER_CLASS,
                     activation, make_data, solves)
from .fold1d_theorem import maximum_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_DENSE = 4_001
A_VALUES = (1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
W2_GRID = np.arange(1.0, 12.01, 0.25)
RESTARTS = 24
INNER_STEPS = 600

_inner = torch.linspace(-INNER_MAX, INNER_MAX, N_DENSE, dtype=torch.float64)
_pos = torch.linspace(OUTER_MIN, OUTER_MAX, N_DENSE // 2, dtype=torch.float64)
_outer = torch.cat([_pos, -_pos])


def gap_at(f, w1: float, b1: float, w2: float) -> float:
    with torch.no_grad():
        vi = f(torch.tensor(w1, dtype=torch.float64) * _inner + b1)
        vo = f(torch.tensor(w1, dtype=torch.float64) * _outer + b1)
    if w2 > 0:
        return float(vo.min() - vi.max())
    return float(vi.min() - vo.max())


def conditional_min(a: float, w2: float, seed: int):
    """Minimise the training loss over (w1,b1,b2) with w2 HELD FIXED."""

    f = activation("sin_family", a)
    x, y = make_data(N_PER_CLASS, 0)                 # the same 400 points
    g = torch.Generator().manual_seed(seed)
    p = torch.empty(3, dtype=torch.float64)
    p[0] = torch.rand(1, generator=g).double() * 4 - 2      # w1
    p[1] = torch.rand(1, generator=g).double() * 8 - 4      # b1
    p[2] = torch.rand(1, generator=g).double() * 2 * abs(w2) - abs(w2)  # b2
    p = p.requires_grad_(True)
    opt = torch.optim.Adam([p], lr=5e-2)
    xd, yd = x.double(), y.double()
    w2t = torch.tensor(w2, dtype=torch.float64)
    for _ in range(INNER_STEPS):
        opt.zero_grad(set_to_none=True)
        out = w2t * f(p[0] * xd + p[1]) + p[2]
        F.binary_cross_entropy_with_logits(out, yd).backward()
        opt.step()
    q = p.detach()
    w1, b1, b2 = float(q[0]), float(q[1]), float(q[2])
    with torch.no_grad():
        out = w2t * f(q[0] * xd + q[1]) + q[2]
        loss = float(F.binary_cross_entropy_with_logits(out, yd))
    theta = torch.tensor([w1, b1, w2, b2], dtype=torch.float64)
    return {"w1": w1, "b1": b1, "b2": b2, "loss": loss,
            "gap": gap_at(f, w1, b1, w2),
            "solves": bool(solves(theta, f))}


def main() -> None:
    rows = []
    for a in A_VALUES:
        gs = maximum_gap(a, resolution=600)
        cross_w2 = solve_w2 = None
        for w2 in W2_GRID:
            cands = [conditional_min(a, float(w2), s) for s in range(RESTARTS)]
            best = min(cands, key=lambda c: c["loss"])       # dominant basin
            frac_g = float(np.mean([c["gap"] > 0 for c in cands]))
            if cross_w2 is None and best["gap"] > 0:
                cross_w2 = float(w2)
            if solve_w2 is None and best["solves"]:
                solve_w2 = float(w2)
            rows.append({"a": a, "w2": float(w2), "best_loss": best["loss"],
                         "best_gap": best["gap"], "best_solves": best["solves"],
                         "frac_restarts_gap_pos": frac_g, "gstar": gs})
            if solve_w2 is not None:
                break
        print(f"  a={a:<5} w2*_cross={cross_w2}  R_cross={None if cross_w2 is None else cross_w2*gs/2:.4f}   "
              f"w2*_solve={solve_w2}  R_solve={None if solve_w2 is None else solve_w2*gs/2:.4f}",
              flush=True)
        frame = pd.DataFrame(rows)
        stem = RESULTS / "phase2b_conditional"
        with artifact_lock(stem, "phase2b conditional"):
            tmp = stem.with_suffix(".csv.tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    print("done")


if __name__ == "__main__":
    main()
