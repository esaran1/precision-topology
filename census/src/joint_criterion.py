"""Part 3 fixed: the JOINT condition over all four parameters.

The 1D |w2| criterion over-predicted (mean abs err 0.283, all errors
positive) because |w2| says nothing about whether (w1,b1) place the data in
the fold's usable window.  Here the criterion is evaluated on the terminal
parameter vector directly:

  joint(theta) = does the CENTRED version of theta classify correctly?

i.e. hold the terminal (w1, b1, w2) and choose the optimal b2 (the interval
midpoint).  That isolates "did training land (w1,b1,w2) in a configuration
where ANY b2 works" from "did it also get b2 right", so the criterion tests
the geometry rather than the full four-way coincidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, activation, make_data,
                     solves)
from .budget_law import run_full

RESULTS = Path(__file__).resolve().parents[1] / "results"
INNER = torch.linspace(-INNER_MAX, INNER_MAX, 4001, dtype=torch.float64)
_POS = torch.linspace(OUTER_MIN, OUTER_MAX, 2001, dtype=torch.float64)
OUTER = torch.cat([_POS, -_POS])


def gap_of(theta: torch.Tensor, a: float) -> float:
    """Class gap at the terminal (w1,b1,w2): positive => some b2 classifies."""
    f = activation("sin_family", a)
    w1, b1, w2 = float(theta[0]), float(theta[1]), float(theta[2])
    vi = f(w1 * INNER + b1) * w2
    vo = f(w1 * OUTER + b1) * w2
    return float(vo.min() - vi.max())


def centred_solves(theta: torch.Tensor, a: float) -> bool:
    """Would the terminal (w1,b1,w2) classify with the optimal b2?"""
    f = activation("sin_family", a)
    w1, b1, w2 = float(theta[0]), float(theta[1]), float(theta[2])
    vi = f(w1 * INNER + b1) * w2
    vo = f(w1 * OUTER + b1) * w2
    if float(vo.min() - vi.max()) <= 0:
        return False
    b2 = -float(vo.min() + vi.max()) / 2.0
    probe = torch.tensor([w1, b1, w2, b2], dtype=torch.float64)
    return bool(solves(probe, f))


def main() -> None:
    conditions = [("a_sweep", a, 1e-2, 1.0, 2_000) for a in
                  (1.02, 1.10, 1.25, 1.35, 1.45, 1.50, 2.00, 3.00)]
    conditions += [("lr_sweep", a, lr, 1.0, 2_000) for a in (1.25, 1.45)
                   for lr in (3e-2, 3e-3, 1e-3)]
    conditions += [("init_scale", a, 1e-2, sc, 2_000) for a in (1.25, 1.50)
                   for sc in (0.3, 3.0)]
    conditions += [("budget", 1.25, 1e-2, 1.0, B) for B in (4_000, 8_000, 16_000)]

    rows = []
    for kind, a, lr, scale, budget in conditions:
        n = 40
        obs = joint = onedim = 0
        from .fold1d_theorem import maximum_gap
        thresh = 2 * 0.1 / maximum_gap(a)
        for s in range(n):
            t, ok, loss, stalled = run_full(a, s, budget, lr, scale)
            obs += ok
            joint += centred_solves(t.double(), a)
            onedim += abs(float(t[2])) >= thresh
        rows.append({"kind": kind, "a": a, "lr": lr, "init_scale": scale,
                     "budget": budget, "observed": obs / n,
                     "joint_pred": joint / n, "onedim_pred": onedim / n, "n": n})
        print(f"{kind} a={a} lr={lr:g} sc={scale} B={budget}: obs={obs/n:.3f} "
              f"joint={joint/n:.3f} onedim={onedim/n:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    frame["joint_err"] = frame.joint_pred - frame.observed
    frame["onedim_err"] = frame.onedim_pred - frame.observed
    print(f"\njoint  : mean|err| {frame.joint_err.abs().mean():.4f} "
          f"max {frame.joint_err.abs().max():.4f}")
    print(f"onedim : mean|err| {frame.onedim_err.abs().mean():.4f} "
          f"max {frame.onedim_err.abs().max():.4f}")
    stem = RESULTS / "joint_criterion"
    with artifact_lock(stem, "joint criterion"):
        tmp = stem.with_suffix(".csv.tmp"); frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
