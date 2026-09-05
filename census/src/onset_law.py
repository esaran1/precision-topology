"""Measure the onset at each budget and test it against the registered
exponent -2*alpha/3 = -0.745 (band [-0.895, -0.595]).

Onset := the a at which the solve rate first crosses 50%, by bisection on a
fixed grid, 40 seeds per evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifact_lock import artifact_lock
from .budget_law import run_full

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = 40


def rate_at(a: float, budget: int) -> float:
    return sum(run_full(a, s, budget)[1] for s in range(SEEDS)) / SEEDS


def onset_for(budget: int, grid) -> tuple[float, list]:
    """First grid value whose rate >= 0.5, with the full curve."""
    curve = []
    onset = None
    for a in grid:
        r = rate_at(a, budget)
        curve.append((a, r))
        print(f"   B={budget} a={a:.4f}: rate={r:.3f}", flush=True)
        if r >= 0.5:
            onset = a          # keep descending; last >=50% value is the onset
        elif onset is not None:
            break              # crossed below 50%: bracketed
    return onset, curve


def main() -> None:
    # grid descends toward 1 so the first >=50% value is the onset
    # descend from well above the expected onset to well below it, so the
    # crossing is bracketed rather than merely bounded above
    grids = {
        2_000:   [1.60, 1.55, 1.50, 1.48, 1.46, 1.44, 1.42, 1.40, 1.35],
        8_000:   [1.30, 1.25, 1.22, 1.20, 1.18, 1.16, 1.14, 1.12, 1.10],
        32_000:  [1.16, 1.13, 1.11, 1.09, 1.08, 1.07, 1.06, 1.05, 1.04],
        128_000: [1.08, 1.06, 1.05, 1.045, 1.04, 1.035, 1.03, 1.025, 1.02],
    }
    rows, curves = [], []
    for budget, grid in grids.items():
        onset, curve = onset_for(budget, grid)
        rows.append({"budget": budget, "onset": onset})
        for a, r in curve:
            curves.append({"budget": budget, "a": a, "rate": r})
        print(f"B={budget}: onset={onset}", flush=True)
    frame = pd.DataFrame(rows).dropna()
    lb = np.log(frame.budget.values)
    lo = np.log(frame.onset.values - 1.0)
    slope, intercept = np.polyfit(lb, lo, 1)
    resid = lo - (slope * lb + intercept)
    r2 = 1 - resid.var() / lo.var()
    print(f"\nMEASURED ONSET EXPONENT = {slope:.4f}  (R^2 = {r2:.4f})")
    print(f"REGISTERED PREDICTION    = -0.7448, band [-0.895, -0.595]")
    print(f"VERDICT: {'WITHIN BAND' if -0.895 <= slope <= -0.595 else 'OUTSIDE BAND'}")
    for name, f in (("onset_law", frame), ("onset_curves", pd.DataFrame(curves))):
        stem = RESULTS / name
        with artifact_lock(stem, name):
            tmp = stem.with_suffix(".csv.tmp"); f.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    Path(RESULTS / "onset_law_fit.txt").write_text(
        f"measured_exponent={slope}\nr2={r2}\nregistered=-0.7448\n"
        f"band=[-0.895,-0.595]\nwithin={-0.895 <= slope <= -0.595}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
