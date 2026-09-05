"""1a/1b: is alpha derivable from the loss geometry and Adam's update?

Measures, along real trajectories:
  - |dL/dw2| as a function of |w2|      (the gradient-flow exponent p)
  - the realized Adam step in w2 vs |w2| (what actually drives growth)
  - |w2|(t) directly, to confirm the power law within a single run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import activation, logits, make_data

RESULTS = Path(__file__).resolve().parents[1] / "results"


def trace(a: float, seed: int, budget: int = 40_000, lr: float = 1e-2,
          optimizer_name: str = "adam"):
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = (torch.optim.Adam([theta], lr=lr) if optimizer_name == "adam"
           else torch.optim.SGD([theta], lr=lr))
    rows = []
    prev = theta.detach().clone()
    for i in range(budget):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(logits(theta, x, f), y)
        loss.backward()
        g2 = float(theta.grad[2])
        opt.step()
        step2 = float(theta.detach()[2] - prev[2])
        prev = theta.detach().clone()
        if i % 50 == 0:
            rows.append({"seed": seed, "step": i + 1,
                         "w2": abs(float(theta.detach()[2])),
                         "grad_w2": abs(g2), "realized_step_w2": abs(step2),
                         "loss": float(loss)})
    return rows


def main() -> None:
    rows = []
    for seed in range(12):
        rows.extend(trace(1.25, seed))
        print(f"trace seed {seed} done", flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "alpha_trace"
    with artifact_lock(stem, "alpha trace"):
        tmp = stem.with_suffix(".csv.tmp"); frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))

    # only the growth phase: w2 rising, past the log-2 plateau
    g = frame[(frame.w2 > 1.0) & (frame.loss < 0.69)]
    lw = np.log(g.w2.values)
    p_grad = -np.polyfit(lw, np.log(np.maximum(g.grad_w2.values, 1e-300)), 1)[0]
    p_step = -np.polyfit(lw, np.log(np.maximum(g.realized_step_w2.values, 1e-300)), 1)[0]
    print(f"\ngradient exponent  p (|dL/dw2| ~ |w2|^-p) = {p_grad:.4f}"
          f"  -> alpha = 1/(p+1) = {1/(p_grad+1):.4f}")
    print(f"realized-step exponent p_step               = {p_step:.4f}"
          f"  -> alpha = {1/(p_step+1):.4f}")
    print(f"registered implication of measured alpha 1.1172: p = -0.105")
    # direct: |w2| vs t within traces
    per = []
    for s, sub in g.groupby("seed"):
        if len(sub) > 20:
            per.append(np.polyfit(np.log(sub.step.values), np.log(sub.w2.values), 1)[0])
    print(f"direct |w2| ~ t^alpha within traces: median {np.median(per):.4f} "
          f"(n={len(per)} traces, IQR {np.percentile(per,25):.3f}-{np.percentile(per,75):.3f})")
    Path(RESULTS / "alpha_derivation_fit.txt").write_text(
        f"p_grad={p_grad}\nalpha_from_grad={1/(p_grad+1)}\n"
        f"p_step={p_step}\nalpha_from_step={1/(p_step+1)}\n"
        f"alpha_direct_median={np.median(per)}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
