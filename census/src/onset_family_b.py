"""2b: does the pwl family's onset also move with budget?

Family B (pwl_family: max(x, alpha*x), non-monotonic for alpha < 0) has no gap
at the standard budget -- it solves at ~57% immediately past alpha = 0.  The
test is whether ITS onset, wherever it sits, moves with budget in the same
direction as family A's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import activation, logits, make_data, solves

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = 40


def run_b(alpha: float, seed: int, budget: int) -> bool:
    f = activation("pwl_family", alpha)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
        opt.step()
    return bool(solves(theta.detach(), f))


def rate_at(alpha: float, budget: int) -> float:
    return sum(run_b(alpha, s, budget) for s in range(SEEDS)) / SEEDS


def main() -> None:
    # alpha -> 0- is family B's threshold; scan toward it from below
    grids = {
        2_000:  [-0.20, -0.10, -0.05, -0.03, -0.02, -0.015, -0.01, -0.005],
        16_000: [-0.05, -0.02, -0.01, -0.005, -0.003, -0.002, -0.001, -0.0005],
        64_000: [-0.02, -0.01, -0.005, -0.002, -0.001, -0.0005, -0.0002, -0.0001],
    }
    rows, curves = [], []
    for budget, grid in grids.items():
        onset, bracketed = None, False
        for alpha in grid:
            r = rate_at(alpha, budget)
            curves.append({"budget": budget, "alpha": alpha, "rate": r})
            print(f"B={budget} alpha={alpha}: rate={r:.3f}", flush=True)
            if r >= 0.5:
                onset = alpha          # closest-to-zero alpha still >=50%
            elif onset is not None:
                bracketed = True
                break
        rows.append({"budget": budget, "onset_alpha": onset,
                     "bracketed": bracketed})
        print(f"B={budget}: onset_alpha={onset} bracketed={bracketed}", flush=True)
    frame = pd.DataFrame(rows)
    print("\n" + frame.to_string(index=False))
    fit = frame[frame.bracketed & frame.onset_alpha.notna()]
    if len(fit) >= 2:
        lb = np.log(fit.budget.values); lo = np.log(np.abs(fit.onset_alpha.values))
        slope = np.polyfit(lb, lo, 1)[0]
        print(f"family B onset exponent (|alpha| ~ B^s) = {slope:.4f}, n = {len(fit)}")
        print(f"family A exponent for comparison = -0.7340")
    for name, f in (("onset_family_b", frame),
                    ("onset_family_b_curves", pd.DataFrame(curves))):
        stem = RESULTS / name
        with artifact_lock(stem, name):
            tmp = stem.with_suffix(".csv.tmp"); f.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
