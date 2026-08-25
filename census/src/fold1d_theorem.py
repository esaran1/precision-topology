"""Verification of the Part 5 lower bound (results/fold1d_theorem.md).

Theorem: any solving width-1 network with logit margin m satisfies
|w2| >= 2m / G(w1, b1) >= 2m / G*(a), with G* = kappa(a) * D(a) and
kappa ~ 0.31 fixed by the task windows.

This module recomputes kappa(a), re-derives every recorded solver
deterministically from its seed, and checks the bound against each.
Solvers were found and recorded before the theorem was written.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, activation, logits,
                     make_data, solves)

INNER = torch.linspace(-INNER_MAX, INNER_MAX, 4_001, dtype=torch.float64)
_POSITIVE = torch.linspace(OUTER_MIN, OUTER_MAX, 2_001, dtype=torch.float64)
OUTER = torch.cat([_POSITIVE, -_POSITIVE])


def dip_depth(a: float) -> float:
    """D(a) = f(local max) - f(local min) of f_a."""

    c = math.acos(1.0 / a)
    f = lambda t: t + a * math.sin(t)  # noqa: E731
    return f(math.pi - c) - f(math.pi + c)


def class_gap(a: float, w1: float, b1: float) -> float:
    f = activation("sin_family", a)
    values_inner = f(torch.tensor(w1, dtype=torch.float64) * INNER + b1)
    values_outer = f(torch.tensor(w1, dtype=torch.float64) * OUTER + b1)
    return float(values_outer.min() - values_inner.max())


def maximum_gap(a: float, resolution: int = 400) -> float:
    """G*(a): the largest class gap any (w1, b1) can produce."""

    c = math.acos(1.0 / a)
    best = 0.0
    for fraction in np.linspace(0.6, 1.4, 33):
        b1 = math.pi + c * fraction
        for w1 in np.linspace(0.001, 3.0, resolution):
            best = max(best, class_gap(a, float(w1), float(b1)))
    return best


def rebuild_solver(a: float, seed: int) -> torch.Tensor | None:
    """Deterministically re-derive a recorded solver from its seed."""

    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4, dtype=torch.float64).uniform_(-1.0, 1.0).requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=1e-2)
    for _ in range(2_000):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(
            logits(theta, x.double(), f), y.double()).backward()
        optimizer.step()
    detached = theta.detach()
    return detached if solves(detached, f) else None


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    recorded = pd.concat([pd.read_csv(directory / "fold1d_sweep.csv"),
                          pd.read_csv(directory / "fold1d_refine.csv")])
    solved = recorded[(recorded.activation == "sin_family") & recorded.solved]

    print("kappa(a) = G*(a) / D(a):")
    kappa = {}
    for a in (1.02, 1.05, 1.10, 1.25, 1.35, 1.45, 1.50, 2.00, 3.00):
        gstar = maximum_gap(a)
        kappa[a] = gstar / dip_depth(a)
        print(f"  a={a:<5} G*={gstar:.5f}  D={dip_depth(a):.5f}  kappa={kappa[a]:.4f}")

    rows = []
    for a in sorted(solved.parameter.unique()):
        a = float(a)
        gstar = maximum_gap(a)
        for seed in solved[solved.parameter == a].seed.tolist()[:25]:
            theta = rebuild_solver(a, int(seed))
            if theta is None:
                continue
            w1, b1, w2, b2 = (float(v) for v in theta)
            f = activation("sin_family", a)
            values_inner = f(w1 * INNER + b1)
            values_outer = f(w1 * OUTER + b1)
            inner_logits = w2 * values_inner + b2
            outer_logits = w2 * values_outer + b2
            margin = min(float(-inner_logits.max()), float(outer_logits.min()))
            bound = 2 * margin / gstar
            rows.append({"a": a, "seed": int(seed), "w2_abs": abs(w2),
                         "margin": margin, "bound": bound,
                         "satisfies_bound": abs(w2) >= bound - 1e-9,
                         "slack": abs(w2) / bound if bound > 0 else float("nan"),
                         "w2_times_D": abs(w2) * dip_depth(a)})
    frame = pd.DataFrame(rows)
    violations = int((~frame.satisfies_bound).sum())
    print(f"\nsolvers checked: {len(frame)}   bound violations: {violations}")
    print(f"slack |w2|/bound — median {frame.slack.median():.3f}, min {frame.slack.min():.3f}")
    print("\n|w2|*D(a) is NOT constant across a (the pre-theorem reading):")
    print(frame.groupby("a").w2_times_D.min().round(3).to_string())
    stem = directory / "theorem_verification"
    with artifact_lock(stem, "theorem verification"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))
    if violations:
        raise SystemExit("BOUND VIOLATED — stop and investigate")


if __name__ == "__main__":
    main()
