"""Terminal knife-edge or trajectory divergence?

Registered in results/precision_prediction.md before this ran.

For each seed, run float32 and float64 side by side from identical
initialisation and record (i) the step at which the parameter vectors first
differ by more than 1% relative, and (ii) the relative distance between the
final parameters.  Split by whether the two precisions agreed on the outcome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import LR, N_PER_CLASS, STEPS, activation, logits, make_data, solves

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.40, 1.45, 1.50, 2.00)
SEEDS = 60
REL = 0.01


def paired(a: float, seed: int, budget: int = STEPS):
    """Train the same seed in both precisions, tracking divergence."""

    f = activation("sin_family", a)
    x, y = make_data(N_PER_CLASS, seed)
    torch.manual_seed(seed)
    init = torch.empty(4).uniform_(-1.0, 1.0)

    t32 = init.clone().requires_grad_(True)
    t64 = init.double().clone().requires_grad_(True)
    o32 = torch.optim.Adam([t32], lr=LR)
    o64 = torch.optim.Adam([t64], lr=LR)
    x64, y64 = x.double(), y.double()

    first = None
    for step in range(budget):
        o32.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(t32, x, f), y).backward()
        o32.step()
        o64.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(t64, x64, f), y64).backward()
        o64.step()
        if first is None:
            d = float((t32.detach().double() - t64.detach()).norm())
            n = float(t64.detach().norm())
            if n > 0 and d / n > REL:
                first = step
    a32, a64 = t32.detach(), t64.detach()
    dist = float((a32.double() - a64).norm())
    rel = dist / float(a64.norm()) if float(a64.norm()) > 0 else np.nan
    return {"a": a, "seed": seed,
            "solved32": bool(solves(a32, f)), "solved64": bool(solves(a64, f)),
            "first_divergence": first if first is not None else budget,
            "final_dist": dist, "final_rel": rel,
            "w2_32": float(a32[2]), "w2_64": float(a64[2])}


def main() -> None:
    rows = []
    for a in A_VALUES:
        for seed in range(SEEDS):
            rows.append(paired(a, seed))
        frame = pd.DataFrame(rows)
        c = frame[frame.a == a].copy()
        c["flipped"] = c.solved32 != c.solved64
        fl, ag = c[c.flipped], c[~c.flipped]
        print(f"  a={a:<5} flipped={len(fl):2d}/{len(c)}  "
              f"median rel dist flipped={fl.final_rel.median() if len(fl) else float('nan'):.4f} "
              f"agreeing={ag.final_rel.median() if len(ag) else float('nan'):.4f}  "
              f"median first-divergence step flipped="
              f"{fl.first_divergence.median() if len(fl) else float('nan'):.0f}", flush=True)
        stem = RESULTS / "precision_divergence"
        with artifact_lock(stem, "precision divergence"):
            tmp = stem.with_suffix(".csv.tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    print("done")


if __name__ == "__main__":
    main()
