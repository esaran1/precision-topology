"""Part 1a/1b: barrier heights in the 1D fold task, two estimators.

Energy is the training loss (BCE on the 400-point task sample).  Barrier
= max along path minus max(endpoint losses), the Goodfellow convention
already used in basin_profiles.csv.

Linear: straight line from initialization to solution, 101 points.
MEP: string method — discretize the path, repeatedly take a gradient step
on each interior image and re-space the images by arclength.  Reports the
converged path's barrier, which lower-bounds the linear estimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import activation, logits, make_data, solves

PATH_POINTS = 101
STRING_IMAGES = 41
STRING_STEPS = 400
STRING_LR = 5e-3
DATA_SEED = 0


def loss_of(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor, f) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits(theta, x, f), y)


def solution_at(a: float, seed: int, steps: int = 2_000) -> torch.Tensor | None:
    """Train one run; return theta if it solves."""

    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4, dtype=torch.float64).uniform_(-1.0, 1.0).requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss_of(theta, x.double(), y.double(), f).backward()
        optimizer.step()
    detached = theta.detach()
    return detached if solves(detached, f) else None


def initialization(seed: int, scale: float = 1.0) -> torch.Tensor:
    torch.manual_seed(100_000 + seed)
    return torch.empty(4, dtype=torch.float64).uniform_(-1.0, 1.0) * scale


def linear_barrier(start: torch.Tensor, end: torch.Tensor, a: float,
                   x: torch.Tensor, y: torch.Tensor) -> float:
    f = activation("sin_family", a)
    alphas = torch.linspace(0.0, 1.0, PATH_POINTS, dtype=torch.float64)
    losses = torch.stack([
        loss_of((1 - alpha) * start + alpha * end, x, y, f) for alpha in alphas
    ])
    return float(losses.max() - torch.maximum(losses[0], losses[-1]))


def string_barrier(start: torch.Tensor, end: torch.Tensor, a: float,
                   x: torch.Tensor, y: torch.Tensor) -> float:
    """Simplified string method: descend interior images, re-space by arclength."""

    f = activation("sin_family", a)
    alphas = torch.linspace(0.0, 1.0, STRING_IMAGES, dtype=torch.float64).unsqueeze(1)
    path = (1 - alphas) * start.unsqueeze(0) + alphas * end.unsqueeze(0)
    path = path.clone()
    for _ in range(STRING_STEPS):
        interior = path[1:-1].clone().requires_grad_(True)
        total = torch.stack([loss_of(p, x, y, f) for p in interior]).sum()
        (grad,) = torch.autograd.grad(total, interior)
        with torch.no_grad():
            path[1:-1] = interior - STRING_LR * grad
            # re-space by cumulative arclength
            deltas = (path[1:] - path[:-1]).norm(dim=1)
            arclength = torch.cat([torch.zeros(1, dtype=torch.float64), deltas.cumsum(0)])
            if float(arclength[-1]) <= 0:
                break
            arclength = arclength / arclength[-1]
            target = torch.linspace(0.0, 1.0, STRING_IMAGES, dtype=torch.float64)
            respaced = torch.empty_like(path)
            for dim in range(4):
                respaced[:, dim] = torch.from_numpy(
                    np.interp(target.numpy(), arclength.numpy(), path[:, dim].numpy()))
            path = respaced
    with torch.no_grad():
        losses = torch.stack([loss_of(p, x, y, f) for p in path])
    return float(losses.max() - torch.maximum(losses[0], losses[-1]))


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    x, y = make_data(200, DATA_SEED)
    x, y = x.double(), y.double()

    values = [1.02, 1.05, 1.10, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50,
              1.60, 1.75, 2.00, 2.50, 3.00]
    rows = []
    for a in values:
        # find a solution to serve as the path endpoint
        target = None
        for seed in range(60):
            target = solution_at(a, seed)
            if target is not None:
                break
        source = "found"
        if target is None:
            # below the onset: use the analytic construction as endpoint
            from .box_counterexample import counterexample
            spec = counterexample(a)
            target = torch.tensor([spec["w1"], spec["b1"], spec["w2"], spec["b2"]],
                                  dtype=torch.float64)
            source = "constructed"
        for init_seed in range(20):
            start = initialization(init_seed)
            lin = linear_barrier(start, target, a, x, y)
            mep = string_barrier(start, target, a, x, y) if init_seed < 5 else None
            rows.append({"a": a, "endpoint": source, "init_seed": init_seed,
                         "barrier_linear": lin, "barrier_mep": mep,
                         "w2_endpoint": float(target[2])})
        done = pd.DataFrame(rows)
        sub = done[done.a == a]
        print(f"a={a}: endpoint={source} |w2|={abs(float(target[2])):.2f} "
              f"linear median={sub.barrier_linear.median():.3f} "
              f"mep median={sub.barrier_mep.dropna().median():.3f}", flush=True)
        stem = directory / "barrier"
        with artifact_lock(stem, "barrier measurement"):
            temp = stem.with_suffix(".csv.tmp")
            done.to_csv(temp, index=False)
            temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
