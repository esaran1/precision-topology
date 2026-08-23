"""Task B: the minimal fold task — sign(|x| - 1) with a width-1 network.

Registered in ``results/fold1d_prediction.md``.  Four parameters
(w1, b1, w2, b2), single logit, sign readout.  Monotonic activations
provably cannot solve it (a monotone logit has at most one sign change;
the task needs two), so the zero side is beyond dispute and everything
measured here is about reachability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock


INNER_MAX = 0.8
OUTER_MIN, OUTER_MAX = 1.2, 2.0
N_PER_CLASS = 200
STEPS = 2_000
LR = 1e-2
DENSE_N = 10_000


def make_data(n_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    inner = rng.uniform(-INNER_MAX, INNER_MAX, n_per_class)
    magnitude = rng.uniform(OUTER_MIN, OUTER_MAX, n_per_class)
    sign = rng.choice([-1.0, 1.0], n_per_class)
    outer = magnitude * sign
    x = np.concatenate([inner, outer]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.float32)
    return torch.tensor(x), torch.tensor(y)


def activation(name: str, parameter: float | None) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "sin_family":
        return lambda v: v + parameter * torch.sin(v)
    if name == "pwl_family":
        return lambda v: torch.where(v >= 0, v, parameter * v)
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    if name == "leaky_relu":
        return lambda v: F.leaky_relu(v, 0.01)
    if name == "gelu":
        return F.gelu
    raise ValueError(name)


def logits(theta: torch.Tensor, x: torch.Tensor, f) -> torch.Tensor:
    w1, b1, w2, b2 = theta[0], theta[1], theta[2], theta[3]
    return w2 * f(w1 * x + b1) + b2


def solves(theta: torch.Tensor, f, n_check: int = 4_001) -> bool:
    """Exact-region check: correct sign on dense grids of both class regions."""

    inner = torch.linspace(-INNER_MAX, INNER_MAX, n_check)
    outer_pos = torch.linspace(OUTER_MIN, OUTER_MAX, n_check // 2)
    outer = torch.cat([outer_pos, -outer_pos])
    with torch.no_grad():
        return bool(
            (logits(theta, inner, f) < 0).all() and (logits(theta, outer, f) > 0).all()
        )


def train_one(name: str, parameter: float | None, seed: int) -> dict:
    f = activation(name, parameter)
    x, y = make_data(N_PER_CLASS, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=LR)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
        optimizer.step()
    with torch.no_grad():
        xe, ye = make_data(N_PER_CLASS, 500_000 + seed)
        eval_errors = int(((logits(theta, xe, f) > 0).float() != ye).sum().item())
    solved = eval_errors == 0 and solves(theta.detach(), f)
    return {
        "activation": name, "parameter": parameter, "seed": seed,
        "eval_errors": eval_errors, "solved": solved,
        "w2_abs": float(theta[2].abs().item()),
        "w1_abs": float(theta[0].abs().item()),
    }


GRID = (
    [("sin_family", a) for a in (0.5, 0.9, 0.95, 1.0, 1.02, 1.05, 1.1, 1.15, 1.25, 1.5, 2.0, 3.0)]
    + [("pwl_family", al) for al in (0.5, 0.1, 0.0, -0.05, -0.1, -0.25, -0.5, -1.0)]
    + [("tanh", None), ("relu", None), ("leaky_relu", None), ("gelu", None)]
)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    rows = []
    for name, parameter in GRID:
        for seed in range(200):
            rows.append(train_one(name, parameter, seed))
        solved = sum(r["solved"] for r in rows if r["activation"] == name and r["parameter"] == parameter)
        print(f"{name}({parameter}): {solved}/200", flush=True)
        frame = pd.DataFrame(rows)
        stem = directory / "fold1d_sweep"
        with artifact_lock(stem, "1d fold sweep"):
            temp = stem.with_suffix(".csv.tmp")
            frame.to_csv(temp, index=False)
            temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
