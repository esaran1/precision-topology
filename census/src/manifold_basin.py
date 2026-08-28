"""Part 4: why do found solutions sit at |w2| ~ 5?

4a: basin volume as a function of |w2| along the solution manifold.
4b: the two candidate constraints, measured separately —
    (i) below the peak, b1-perturbations destroy the solution
        (measured as the b1 half-width of the solution sheet at that |w2|);
    (ii) above the peak, the target is reached from a shrinking set of
        initializations (measured as the fraction of standard inits whose
        trained |w2| reaches that value).
4c: does the peak move with initialization scale?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, activation, logits,
                     make_data, solves)

A = 1.5
PROBE = 4_001


def solution_at_w2(a: float, w2: float) -> torch.Tensor | None:
    """The centred solution on the manifold at this |w2|, if one exists."""

    import math

    f = activation("sin_family", a)
    c = math.acos(1.0 / a)
    b1 = math.pi + c
    inner = torch.linspace(-INNER_MAX, INNER_MAX, 4_001, dtype=torch.float64)
    positive = torch.linspace(OUTER_MIN, OUTER_MAX, 2_001, dtype=torch.float64)
    outer = torch.cat([positive, -positive])
    best = None
    for fraction in np.linspace(0.02, 0.99, 120):
        w1 = fraction * c
        gap = float(f(w1 * outer + b1).min() - f(w1 * inner + b1).max())
        if gap > 0 and (best is None or gap > best[1]):
            best = (w1, gap)
    if best is None:
        return None
    w1, _ = best
    values_inner = f(w1 * inner + b1)
    values_outer = f(w1 * outer + b1)
    b2 = -w2 * float(values_outer.min() + values_inner.max()) / 2.0
    theta = torch.tensor([w1, b1, w2, b2], dtype=torch.float64)
    return theta if solves(theta, f) else None


def half_width(theta: torch.Tensor, a: float, index: int) -> float:
    """Solution half-width in one coordinate through this point."""

    f = activation("sin_family", a)
    base = float(theta[index])
    span = max(abs(base), 1.0)
    offsets = torch.linspace(0.0, span, PROBE, dtype=torch.float64)
    for offset in offsets:
        probe = theta.clone()
        probe[index] = base + float(offset)
        if not solves(probe, f, n_check=801):
            return float(offset)
    return float(span)


def basin_fraction(theta: torch.Tensor, a: float, epsilon: float,
                   draws: int = 60, seed: int = 0) -> float:
    """Fraction of perturbed starts that retrain back to a solution."""

    f = activation("sin_family", a)
    x, y = make_data(200, 0)
    generator = torch.Generator().manual_seed(seed)
    recovered = 0
    for _ in range(draws):
        noise = torch.randn(4, generator=generator, dtype=torch.float64) * epsilon * theta.abs()
        start = (theta + noise).float().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([start], lr=1e-2)
        for _ in range(2_000):
            optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(start, x, f), y).backward()
            optimizer.step()
        recovered += solves(start.detach(), f)
    return recovered / draws


def reach_fraction(a: float, w2_target: float, scale: float = 1.0,
                   seeds: int = 100) -> float:
    """Fraction of trained runs whose |w2| reaches at least this value."""

    f = activation("sin_family", a)
    count = 0
    for seed in range(seeds):
        x, y = make_data(200, seed)
        torch.manual_seed(seed)
        theta = (torch.empty(4).uniform_(-1.0, 1.0) * scale).requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=1e-2)
        for _ in range(2_000):
            optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
            optimizer.step()
        count += abs(float(theta.detach()[2])) >= w2_target
    return count / seeds


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    rows = []
    for w2 in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0):
        theta = solution_at_w2(A, w2)
        if theta is None:
            print(f"|w2|={w2}: no solution", flush=True)
            continue
        row = {
            "a": A, "w2": w2,
            "halfwidth_b1": half_width(theta, A, 1),
            "halfwidth_w1": half_width(theta, A, 0),
            "halfwidth_b2": half_width(theta, A, 3),
            "basin_eps03": basin_fraction(theta, A, 0.3),
            "basin_eps10": basin_fraction(theta, A, 1.0),
            "reach_standard": reach_fraction(A, w2, 1.0),
            "reach_scaled3x": reach_fraction(A, w2, 3.0),
        }
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        frame = pd.DataFrame(rows)
        stem = directory / "manifold_basin"
        with artifact_lock(stem, "manifold basin"):
            temp = stem.with_suffix(".csv.tmp")
            frame.to_csv(temp, index=False)
            temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
