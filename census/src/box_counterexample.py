"""Counterexamples to T30's box-emptiness claim (correction 2026-08-25).

Solutions to sign(|x| - 1) with a width-1 sin-family network exist inside
[-5, 5]^4 at every tested a > 1, including a = 1.02, with |w2| = 1.  The
committed claim that the solution set is "empty within both boxes" at
a <= 1.10 was a grid-resolution artifact: the admissible b2 interval has
width |w2| * gap(a), and a 41-point grid (step 0.25) needs |w2| > 170 at
a = 1.05 for a node to land inside it — the origin of the reported
"required |w2| ~ 58".

Construction: place the data window inside the local MINIMUM of f_a at
m0 = pi + arccos(1/a) (both classes map into the dip; outer points, being
farther from the fold centre, land higher), then centre b2 in the gap.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX

VALUES = (1.02, 1.05, 1.10, 1.25, 1.5)
INNER_POINTS = 300_001
OUTER_POINTS = 100_001


def counterexample(a: float, w2: float = 1.0, fraction: float = 0.7) -> dict:
    """Best in-box solution at this a; verified in float64 on dense grids."""

    c = math.acos(1.0 / a)
    b1 = math.pi + c
    f = lambda v: v + a * torch.sin(v)  # noqa: E731
    inner = torch.linspace(-INNER_MAX, INNER_MAX, INNER_POINTS, dtype=torch.float64)
    positive = torch.linspace(OUTER_MIN, OUTER_MAX, OUTER_POINTS, dtype=torch.float64)
    outer = torch.cat([positive, -positive])

    best = None
    for step in range(1, 100):
        w1 = step / 100.0 * c
        values_inner = f(w1 * inner + b1)
        values_outer = f(w1 * outer + b1)
        gap = float(values_outer.min() - values_inner.max())
        if gap > 0 and (best is None or gap > best[1]):
            best = (w1, gap)
    if best is None:
        raise RuntimeError(f"no separating w1 found at a={a}")
    w1, gap = best
    values_inner = f(w1 * inner + b1)
    values_outer = f(w1 * outer + b1)
    b2 = -w2 * float(values_outer.min() + values_inner.max()) / 2.0
    logits_inner = w2 * values_inner + b2
    logits_outer = w2 * values_outer + b2
    solves = bool((logits_inner < 0).all() and (logits_outer > 0).all())
    theta = (w1, b1, w2, b2)
    return {"a": a, "w1": w1, "b1": b1, "w2": w2, "b2": b2, "class_gap": gap,
            "max_inner_logit": float(logits_inner.max()),
            "min_outer_logit": float(logits_outer.min()),
            "solves_float64": solves,
            "max_abs_parameter": max(abs(v) for v in theta),
            "inside_box_5": max(abs(v) for v in theta) <= 5.0,
            "grid41_box5_required_w2": 0.25 / gap}


def main() -> None:
    rows = [counterexample(a) for a in VALUES]
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    if not frame.solves_float64.all() or not frame.inside_box_5.all():
        raise SystemExit("a counterexample failed verification")
    directory = Path(__file__).resolve().parents[1] / "results"
    stem = directory / "box_counterexample"
    with artifact_lock(stem, "box counterexamples"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
