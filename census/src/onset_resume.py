"""Resume the 128k onset cell (the earlier run was killed by a stray pkill).

The 2k/8k/32k cells completed and are recovered from onset_law.log; only the
largest-budget cell is recomputed, continuing below a = 1.045 where it stopped.
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


def main() -> None:
    # recovered from the killed run's log
    curves = [(2_000, 1.60, 0.550), (2_000, 1.55, 0.450),
              (8_000, 1.30, 0.900), (8_000, 1.25, 0.875), (8_000, 1.22, 0.875),
              (8_000, 1.20, 0.875), (8_000, 1.18, 0.850), (8_000, 1.16, 0.475),
              (32_000, 1.16, 0.925), (32_000, 1.13, 0.925), (32_000, 1.11, 0.925),
              (32_000, 1.09, 0.925), (32_000, 1.08, 0.925), (32_000, 1.07, 0.925),
              (32_000, 1.06, 0.650), (32_000, 1.05, 0.000),
              (128_000, 1.08, 1.000), (128_000, 1.06, 1.000),
              (128_000, 1.05, 1.000), (128_000, 1.045, 1.000)]
    onsets = {2_000: 1.60, 8_000: 1.18, 32_000: 1.06}
    bracketed = {2_000: True, 8_000: True, 32_000: True}

    onset_128 = 1.045
    for a in (1.04, 1.035, 1.03, 1.025, 1.02):
        r = rate_at(a, 128_000)
        curves.append((128_000, a, r))
        print(f"B=128000 a={a}: rate={r:.3f}", flush=True)
        if r >= 0.5:
            onset_128 = a
        else:
            bracketed[128_000] = True
            break
    else:
        bracketed[128_000] = False
    onsets[128_000] = onset_128

    frame = pd.DataFrame([{"budget": b, "onset": o,
                           "bracketed": bracketed.get(b, False)}
                          for b, o in sorted(onsets.items())])
    print("\n" + frame.to_string(index=False))
    fit = frame[frame.bracketed]
    print(f"\nBRACKETED CELLS: {len(fit)} of {len(frame)}")
    lb, lo = np.log(fit.budget.values), np.log(fit.onset.values - 1.0)
    slope, intercept = np.polyfit(lb, lo, 1)
    resid = lo - (slope * lb + intercept)
    r2 = 1 - resid.var() / lo.var()
    print(f"MEASURED ONSET EXPONENT = {slope:.4f}  (R^2 = {r2:.4f}, n = {len(fit)})")
    print(f"REGISTERED = -0.7448, band [-0.895, -0.595]")
    print(f"VERDICT: {'WITHIN BAND' if -0.895 <= slope <= -0.595 else 'OUTSIDE BAND'}")

    for name, f in (("onset_law", frame),
                    ("onset_curves", pd.DataFrame(curves, columns=["budget", "a", "rate"]))):
        stem = RESULTS / name
        with artifact_lock(stem, name):
            tmp = stem.with_suffix(".csv.tmp"); f.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    Path(RESULTS / "onset_law_fit.txt").write_text(
        f"measured_exponent={slope}\nr2={r2}\nn_bracketed={len(fit)}\n"
        f"registered=-0.7448\nband=[-0.895,-0.595]\n"
        f"within={-0.895 <= slope <= -0.595}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
