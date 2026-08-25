"""Exact-form solution measure for the 1D fold task (replaces grid scans).

For fixed (w1, b1, w2) the admissible b2 set is an interval of width
|w2| * gap(w1, b1), where gap = min f(outer window) - max f(inner
window) for the w2 > 0 branch (and the mirrored quantity for w2 < 0).
The w2 and b2 integrals are therefore analytic and only the (w1, b1)
expectation is sampled — no resolution floor, unlike
``fold1d_geometry.solution_mask`` whose 41-point grid reported zero
measure wherever the interval was narrower than one cell
(``box_emptiness_correction.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .fold1d import activation, INNER_MAX, OUTER_MIN, OUTER_MAX

DRAWS = 20_000
INNER_POINTS = 4_001
OUTER_POINTS = 2_001


def gap_terms(name: str, parameter: float | None, w1: torch.Tensor,
              b1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive parts of the class gap for the w2 > 0 and w2 < 0 branches."""

    f = activation(name, parameter)
    inner = torch.linspace(-INNER_MAX, INNER_MAX, INNER_POINTS, dtype=torch.float64)
    positive = torch.linspace(OUTER_MIN, OUTER_MAX, OUTER_POINTS, dtype=torch.float64)
    outer = torch.cat([positive, -positive])
    values_inner = f(w1.unsqueeze(1) * inner.unsqueeze(0) + b1.unsqueeze(1))
    values_outer = f(w1.unsqueeze(1) * outer.unsqueeze(0) + b1.unsqueeze(1))
    forward = (values_outer.min(dim=1).values - values_inner.max(dim=1).values).clamp_min(0.0)
    reverse = (values_inner.min(dim=1).values - values_outer.max(dim=1).values).clamp_min(0.0)
    return forward, reverse


def solution_measure(name: str, parameter: float | None, box: float = 5.0,
                     draws: int = DRAWS, seed: int = 0) -> dict:
    """Fraction of [-box, box]^4 that solves, via the analytic b2/w2 integrals."""

    rng = np.random.default_rng(seed)
    w1 = torch.tensor(rng.uniform(-box, box, draws), dtype=torch.float64)
    b1 = torch.tensor(rng.uniform(-box, box, draws), dtype=torch.float64)
    total = 0.0
    for start in range(0, draws, 500):
        forward, reverse = gap_terms(name, parameter, w1[start:start + 500],
                                     b1[start:start + 500])
        # integral over w2 in [-box, box] of |w2| * gap = gap * box^2 / 2 per branch
        total += float(((forward + reverse) * (box * box / 2.0)).sum())
    measure = total / draws / (2 * box) / (2 * box)
    # standard error from the per-draw spread
    return {"activation": name, "parameter": parameter, "box": box,
            "draws": draws, "solution_measure": measure}


def main() -> None:
    values = [("sin_family", a) for a in (1.02, 1.05, 1.10, 1.25, 1.35, 1.50, 2.00, 3.00)]
    values += [("pwl_family", -0.05), ("pwl_family", -0.25), ("gelu", None)]
    rows = []
    for box in (5.0, 10.0):
        for name, parameter in values:
            rows.append(solution_measure(name, parameter, box))
            print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    directory = Path(__file__).resolve().parents[1] / "results"
    stem = directory / "solution_measure"
    with artifact_lock(stem, "analytic solution measure"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
