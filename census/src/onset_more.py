"""2a: two more bracketed onsets (B = 4,000 and 64,000) for the exponent fit."""

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


def bracket(budget: int, grid) -> tuple[float | None, bool, list]:
    onset, curve, bracketed = None, [], False
    for a in grid:
        r = rate_at(a, budget)
        curve.append((budget, a, r))
        print(f"B={budget} a={a}: rate={r:.3f}", flush=True)
        if r >= 0.5:
            onset = a
        elif onset is not None:
            bracketed = True
            break
    return onset, bracketed, curve


def main() -> None:
    grids = {
        4_000:  [1.45, 1.40, 1.35, 1.32, 1.30, 1.28, 1.26, 1.24],
        64_000: [1.10, 1.07, 1.06, 1.05, 1.045, 1.04, 1.035, 1.03],
    }
    rows, curves = [], []
    for budget, grid in grids.items():
        onset, bracketed, curve = bracket(budget, grid)
        rows.append({"budget": budget, "onset": onset, "bracketed": bracketed})
        curves.extend(curve)
        print(f"B={budget}: onset={onset} bracketed={bracketed}", flush=True)

    prior = pd.read_csv(RESULTS / "onset_law.csv")
    combined = pd.concat([prior, pd.DataFrame(rows)]).sort_values("budget")
    fit = combined[combined.bracketed & combined.onset.notna()]
    lb, lo = np.log(fit.budget.values), np.log(fit.onset.values - 1.0)
    slope, intercept = np.polyfit(lb, lo, 1)
    resid = lo - (slope * lb + intercept)
    r2 = 1 - resid.var() / lo.var()
    print(f"\nBRACKETED CELLS: {len(fit)}")
    print(combined.to_string(index=False))
    print(f"SIX-POINT EXPONENT = {slope:.4f} (R^2 = {r2:.4f}, n = {len(fit)})")
    print(f"REGISTERED band [-0.895, -0.595]; point -0.7448")
    print(f"VERDICT: {'WITHIN BAND' if -0.895 <= slope <= -0.595 else 'OUTSIDE BAND'}")
    for name, f in (("onset_law_extended", combined),
                    ("onset_curves_extended", pd.DataFrame(curves, columns=["budget", "a", "rate"]))):
        stem = RESULTS / name
        with artifact_lock(stem, name):
            tmp = stem.with_suffix(".csv.tmp"); f.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    Path(RESULTS / "onset_law_extended_fit.txt").write_text(
        f"exponent={slope}\nr2={r2}\nn_bracketed={len(fit)}\n"
        f"within_band={-0.895 <= slope <= -0.595}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
