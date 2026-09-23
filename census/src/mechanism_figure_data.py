"""Data for the main-text mechanism figure (Fig 10), all at a = 1.30, certified Ĝ.

branch       the conditional branch: at each fixed |w2| on a grid, the frozen Block B
             minimisation (blockB_landscape.best_conditional: 50 restarts, 600-step
             screen, 8 kept, 3,000 steps, degeneracy exclusion, 800-point population
             loss) -> the minimiser's class gap and whether it solves.
             phase2b_conditional.csv is NOT used: it comes from the under-converged
             600/900-step runs (paper_claims_delta.md R2).
trajectories training trajectories (gap and R per checkpoint) for the first 10 seeds that
             crossed at budget 32,000, from phase2b_checkpoints.csv.

    python -m src.mechanism_figure_data branch
    python -m src.mechanism_figure_data trajectories
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A = 1.30
W2_GRID = np.round(np.arange(2.0, 10.0 + 1e-9, 0.25), 2)
BRANCH = RESULTS / "mechanism_branch_a130.csv"
TRAJ = RESULTS / "mechanism_trajectories_a130.csv"


def _ghat():
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv")
    return float(g[g.a.round(2) == A].Ghat_certified.iloc[0])


def _point(w2):
    import torch
    from . import blockB_landscape as bb
    torch.set_num_threads(1)
    x, y = bb.population_data()
    b = bb.best_conditional(A, float(w2), x, y, restarts=bb.RESTARTS)
    if b is None:
        return {"w2": w2, "gap": np.nan, "solves": False, "loss": np.nan,
                "w1": np.nan, "b1": np.nan, "b2": np.nan}
    return {"w2": w2, "gap": b["gap"], "solves": b["solves"], "loss": b["loss"],
            "w1": b["w1"], "b1": b["b1"], "b2": b["b2"]}


def branch(workers=10):
    with Pool(workers) as p:
        rows = p.map(_point, list(W2_GRID))
    d = pd.DataFrame(rows).sort_values("w2")
    d["R_cert"] = d.w2 * _ghat() / 2
    d.to_csv(BRANCH, index=False)
    print(d.to_string(index=False))


def trajectories(n_seeds=10, budget=32_000):
    cols = ["a", "budget", "seed", "step", "w2", "gap", "crossing_step"]
    parts = []
    for ch in pd.read_csv(RESULTS / "phase2b_checkpoints.csv", usecols=cols, chunksize=2_000_000):
        parts.append(ch[(ch.a.round(2) == A) & (ch.budget == budget)])
    d = pd.concat(parts, ignore_index=True)
    crossed = sorted(d[d.crossing_step.notna()].seed.unique())[:n_seeds]
    d = d[d.seed.isin(crossed)].copy()
    d = d[(d.step % 25 == 0) | (d.step == d.crossing_step)].copy()   # thinned for the repository
    d["R_cert"] = d.w2.abs() * _ghat() / 2
    d[["seed", "step", "R_cert", "gap", "crossing_step"]].to_csv(TRAJ, index=False)
    print(f"{len(crossed)} seeds, {len(d)} checkpoints")


if __name__ == "__main__":
    {"branch": branch, "trajectories": trajectories}[sys.argv[1]]()
