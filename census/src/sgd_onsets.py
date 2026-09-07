"""2b: family A onsets under plain SGD, four budgets spanning 64x.

Tests whether the onset law is an Adam artifact. Registered prediction
-0.4792, band [-0.5973, -0.3611] (excludes Adam's -0.7340).
Same bracketing criteria as every prior onset measurement.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
from src.artifact_lock import artifact_lock
from src.fold1d import activation, logits, make_data, solves

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = 40
LR = 0.3          # the only SGD lr that solves at all (step_size_results.md)


def run(a: float, seed: int, budget: int) -> bool:
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = torch.optim.SGD([th], lr=LR)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        opt.step()
    return bool(solves(th.detach(), f))


def rate(a, B):
    return sum(run(a, s, B) for s in range(SEEDS)) / SEEDS


def main() -> None:
    grids = {
        2_000:   [2.20, 2.00, 1.80, 1.70, 1.60, 1.50, 1.45, 1.40],
        8_000:   [1.90, 1.70, 1.55, 1.45, 1.38, 1.32, 1.28, 1.24],
        32_000:  [1.60, 1.45, 1.35, 1.28, 1.22, 1.18, 1.15, 1.12],
        128_000: [1.40, 1.30, 1.22, 1.16, 1.12, 1.09, 1.07, 1.05],
    }
    rows, curves = [], []
    for B, grid in grids.items():
        onset, brk = None, False
        for a in grid:
            r = rate(a, B)
            curves.append({"budget": B, "a": a, "rate": r})
            print(f"  SGD B={B} a={a}: rate={r:.3f}", flush=True)
            if r >= 0.5:
                onset = a
            elif onset is not None:
                brk = True
                break
        rows.append({"optimizer": "sgd", "budget": B, "onset": onset,
                     "bracketed": brk})
        print(f"SGD B={B}: onset={onset} bracketed={brk}", flush=True)
        for nm, fr in (("sgd_onsets", pd.DataFrame(rows)),
                       ("sgd_onset_curves", pd.DataFrame(curves))):
            stem = RESULTS / nm
            with artifact_lock(stem, nm):
                tmp = stem.with_suffix(".csv.tmp"); fr.to_csv(tmp, index=False)
                tmp.replace(stem.with_suffix(".csv"))
    fit = pd.DataFrame(rows)
    fit = fit[fit.bracketed & fit.onset.notna()]
    print(f"\nBRACKETED CELLS: {len(fit)} of {len(rows)}")
    if len(fit) >= 2:
        s = np.polyfit(np.log(fit.budget.values), np.log(fit.onset.values - 1.0), 1)[0]
        lo, hi = -0.5973, -0.3611
        print(f"SGD ONSET EXPONENT = {s:.4f}")
        print(f"REGISTERED: -0.4792, band [{lo}, {hi}]")
        print(f"VERDICT: {'WITHIN BAND' if lo <= s <= hi else 'OUTSIDE BAND'}")
        print(f"Adam's value -0.7340: {'also consistent' if lo <= -0.734 <= hi else 'excluded by the band'}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
