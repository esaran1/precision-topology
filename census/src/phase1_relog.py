"""Phase 1: rerun family A with FULL parameter logging, in both precisions.

The committed sweeps store only |w1| and |w2|; b1 and b2 are absent and signs
are discarded, so placement and bias cannot be reconstructed (Phase 0, 0e).

Both precisions on the SAME seeds, because the pooled R dataset and every
headline number are float32 while new theory work is float64.  The
solve-outcome disagreement rate per cell is the quantity of interest: if it is
material, Phase 1 describes a different population from the headlines.

Success criterion throughout: solves(), the dense 4,001-point regional check.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, LR, N_PER_CLASS, STEPS,
                     activation, logits, make_data, solves)

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.02, 1.05, 1.10, 1.15, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 2.00, 3.00)
SEEDS = 200


def train(a: float, seed: int, budget: int = STEPS, double: bool = False):
    """One run; returns the full terminal parameter vector."""

    f = activation("sin_family", a)
    x, y = make_data(N_PER_CLASS, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4, dtype=torch.float64 if double else torch.float32)
    theta = theta.uniform_(-1.0, 1.0).requires_grad_(True)
    xx, yy = (x.double(), y.double()) if double else (x, y)
    optimizer = torch.optim.Adam([theta], lr=LR)
    for _ in range(budget):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, xx, f), yy).backward()
        optimizer.step()
    t = theta.detach()
    return t, bool(solves(t, f))


def main() -> None:
    rows = []
    for a in A_VALUES:
        for seed in range(SEEDS):
            for double in (False, True):
                t, ok = train(a, seed, double=double)
                w1, b1, w2, b2 = (float(v) for v in t)
                rows.append({"a": a, "seed": seed,
                             "precision": "float64" if double else "float32",
                             "w1": w1, "b1": b1, "w2": w2, "b2": b2,
                             "solved": ok})
        frame = pd.DataFrame(rows)
        cell = frame[frame.a == a]
        p32 = cell[cell.precision == "float32"].set_index("seed").solved
        p64 = cell[cell.precision == "float64"].set_index("seed").solved
        dis = int((p32 != p64).sum())
        print(f"  a={a:<5} 32-bit {int(p32.sum()):3d}/{len(p32)}  "
              f"64-bit {int(p64.sum()):3d}/{len(p64)}  disagree={dis}", flush=True)
        stem = RESULTS / "phase1_runs"
        with artifact_lock(stem, "phase1 runs"):
            tmp = stem.with_suffix(".csv.tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    print("done")


if __name__ == "__main__":
    main()
