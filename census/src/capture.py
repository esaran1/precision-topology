"""Derived capture cross-section: does |w2|*gap(a) relative to the step predict rates?

From fold1d_theorem.md: for fixed (w1,b1,w2) the admissible b2 set is an
interval of width |w2|*G(w1,b1), G <= kappa*D(a), kappa ~ 0.31.  So the
solution set's thickness in b2 is known in closed form.  A capture
cross-section account says the chance of landing in it scales with that
width relative to the step the optimizer takes in b2.

No fitted parameters: G*(a) and the realized step are both measured
independently.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX, activation
from .fold1d_theorem import dip_depth, maximum_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"


def reachable_w2(a: float, lr: float, optimizer_name: str, seeds: int = 60) -> np.ndarray:
    """Final |w2| distribution -- what the optimizer actually reaches."""
    from .sharpness import train
    out = []
    for s in range(seeds):
        t, _, _ = train(a, s, lr, 2_000, optimizer_name)
        out.append(abs(float(t[2])))
    return np.array(out)


def capture_width(a: float, w2: float) -> float:
    """Width of the admissible b2 interval at this |w2|: |w2| * G*(a)."""
    return abs(w2) * maximum_gap(a)


def main() -> None:
    rows = []
    # observed rates from the committed sweeps
    observed = {1.02: 0/200, 1.10: 0/200, 1.25: 0/200, 1.35: 2/200,
                1.40: 15/200, 1.45: 50/200, 1.50: 81/200, 2.00: 143/200,
                3.00: 139/200}
    for a, rate in observed.items():
        gstar = maximum_gap(a)
        w2s = reachable_w2(a, 1e-2, "adam")
        # capture width at the |w2| the optimizer actually reaches
        width_med = capture_width(a, float(np.median(w2s)))
        # realized step in b2 near convergence
        from .sharpness import train
        steps = []
        for s in range(20):
            _, _, rz = train(a, s, 1e-2, 2_000, "adam")
            steps.append(rz[3])
        step_b2 = float(np.median(steps))
        rows.append({"a": a, "observed_rate": rate, "G_star": gstar,
                     "D": dip_depth(a), "w2_reached_median": float(np.median(w2s)),
                     "capture_width_b2": width_med, "step_b2": step_b2,
                     "ratio": width_med / step_b2})
        print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "capture_cross_section"
    with artifact_lock(stem, "capture cross section"):
        tmp = stem.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
