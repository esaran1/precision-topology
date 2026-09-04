"""The budget power law: terminal |w2| ~ B^alpha, and the onset shift it implies.

Registered in results/arrhenius_prediction.md before onsets were measured.
Also: the joint (w1,b1,w2,b2) criterion for Part 3, and staller statistics.
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
LOG2 = float(np.log(2.0))


def run_full(a: float, seed: int, budget: int, lr: float = 1e-2,
             scale: float = 1.0):
    """Train once; return terminal theta, solved, final loss, stalled flag."""
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = (torch.empty(4).uniform_(-1.0, 1.0) * scale).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(logits(theta, x, f), y)
        loss.backward()
        opt.step()
    t = theta.detach()
    final = float(F.binary_cross_entropy_with_logits(logits(t, x, f), y))
    return t, bool(solves(t, f)), final, abs(final - LOG2) < 1e-4


def alpha_sweep(budgets, a: float = 1.25, seeds: int = 30) -> pd.DataFrame:
    rows = []
    for B in budgets:
        w2s, stalls, solved = [], 0, 0
        for s in range(seeds):
            t, ok, loss, stalled = run_full(a, s, B)
            w2s.append(abs(float(t[2]))); stalls += stalled; solved += ok
        rows.append({"a": a, "budget": B, "w2_med": float(np.median(w2s)),
                     "w2_p25": float(np.percentile(w2s, 25)),
                     "w2_p75": float(np.percentile(w2s, 75)),
                     "stall_frac": stalls / seeds, "rate": solved / seeds,
                     "n": seeds})
        print(f"alpha a={a} B={B}: w2_med={np.median(w2s):.2f} "
              f"stall={stalls/seeds:.3f} rate={solved/seeds:.3f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    budgets = (1_000, 2_000, 4_000, 8_000, 16_000, 40_000, 80_000, 160_000)
    frame = alpha_sweep(budgets)
    logB = np.log(frame.budget.values); logw = np.log(frame.w2_med.values)
    alpha, intercept = np.polyfit(logB, logw, 1)
    resid = logw - (alpha * logB + intercept)
    print(f"\nALPHA = {alpha:.4f}  (R^2 = {1 - resid.var()/logw.var():.5f})")
    print(f"predicted onset exponent -2*alpha/3 = {-2*alpha/3:.4f}")
    frame.attrs["alpha"] = alpha
    stem = RESULTS / "budget_alpha"
    with artifact_lock(stem, "budget alpha"):
        tmp = stem.with_suffix(".csv.tmp"); frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    Path(RESULTS / "budget_alpha_fit.txt").write_text(
        f"alpha={alpha}\nintercept={intercept}\n"
        f"predicted_onset_exponent={-2*alpha/3}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
