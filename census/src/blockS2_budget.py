"""Does the budget law extend to the Ren-Lim S^2 u S^2 setting?

Registered in results/blockS2_prediction.md.  EXPLORATORY.

Construction from Ren & Lim's own repository
(github.com/7pocheR/low_dimensional_topology,
exp_4_higher_dim_r5_multicopy/generate_linked_spheres_dataset.py): the verified
tildeA / tildeB embeddings S^2 -> R^5, thickened by a uniform offset in the
normal space of radius rho.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .artifact_lock import artifact_lock

RESULTS = Path(__file__).resolve().parents[1] / "results"
SQRT2 = float(np.sqrt(2.0))
N_PER_CLASS = 1_000
RHO = 0.5
WIDTH, DEPTH = 5, 5
BUDGETS = (1_000, 4_000, 16_000, 64_000)
N_SEEDS = 20
LR = 1e-3


def tilde_a(u):
    u1, upr = u[:, 0], u[:, 1:]
    a = 1.0 - u1 / SQRT2
    return np.concatenate([upr / a[:, None], (-u1 / (SQRT2 * a))[:, None],
                           np.zeros_like(upr)], axis=1)


def tilde_b(v):
    v1, vpr = v[:, 0], v[:, 1:]
    b = 1.0 - v1 / SQRT2
    return np.concatenate([np.zeros_like(vpr), (v1 / (SQRT2 * b))[:, None],
                           vpr / b[:, None]], axis=1)


def _sphere(n, rng):
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def dataset(seed, n_per=N_PER_CLASS, rho=RHO):
    rng = np.random.default_rng(seed)
    A, B = tilde_a(_sphere(n_per, rng)), tilde_b(_sphere(n_per, rng))
    for M in (A, B):
        nz = rng.normal(size=M.shape)
        nz /= np.linalg.norm(nz, axis=1, keepdims=True)
        M += nz * (rho * rng.uniform(size=(len(M), 1)) ** (1 / 5))
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def net(act):
    layers = [nn.Linear(5, WIDTH), act()]
    for _ in range(DEPTH - 2):
        layers += [nn.Linear(WIDTH, WIDTH), act()]
    return nn.Sequential(*layers, nn.Linear(WIDTH, 1))


def run(act_name, seed, budget):
    act = nn.ReLU if act_name == "relu" else nn.GELU
    X, y = dataset(seed)
    torch.manual_seed(seed)
    m = net(act)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(m(X).squeeze(-1), y).backward()
        opt.step()
    with torch.no_grad():
        acc = float(((m(X).squeeze(-1) > 0).float() == y).float().mean())
    return {"activation": act_name, "monotonic": act_name == "relu",
            "seed": seed, "budget": budget, "accuracy": acc,
            "perfect": acc >= 1.0 - 1e-9}


def main():
    rows = []
    for budget in BUDGETS:
        for act_name in ("relu", "gelu"):
            for s in range(N_SEEDS):
                rows.append(run(act_name, s, budget))
            g = pd.DataFrame([r for r in rows
                              if r["budget"] == budget and r["activation"] == act_name])
            print(f"  B={budget:>6} {act_name:5s}: mean acc {g.accuracy.mean():.4f}  "
                  f"median {g.accuracy.median():.4f}  perfect {int(g.perfect.sum())}/{len(g)}",
                  flush=True)
        d = pd.DataFrame([r for r in rows if r["budget"] == budget])
        gap = (d[d.activation == "gelu"].accuracy.mean()
               - d[d.activation == "relu"].accuracy.mean())
        print(f"           gap (gelu - relu) = {gap:+.4f}", flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "blockS2_budget"
    with artifact_lock(stem, "blockS2 budget"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
    print("\nwritten results/blockS2_budget.csv")


if __name__ == "__main__":
    main()
