"""Part 2: where does loss minimization terminate, and why?

2a: the full trajectory in |w2| (growth / saturation / oscillation).
2b: dL/dw2 at termination -- approaching zero (attractor) or cut off (budget)?
2c: budget dependence at 2x/4x/10x.
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


def run(a: float, seed: int, steps: int, lr: float = 1e-2, record: bool = False):
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    trace = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(logits(theta, x, f), y)
        loss.backward()
        if record and (i % max(1, steps // 200) == 0 or i == steps - 1):
            trace.append((i, abs(float(theta[2].detach())),
                          float(theta.grad[2]), float(loss)))
        opt.step()
    t = theta.detach()
    # final dL/dw2
    tt = t.clone().requires_grad_(True)
    loss = F.binary_cross_entropy_with_logits(logits(tt, x, f), y)
    (g,) = torch.autograd.grad(loss, tt)
    return t, solves(t, f), float(g[2]), float(loss), trace


def main() -> None:
    rows, traces = [], []
    # 2a + 2b: trajectories at standard budget
    for a in (1.25, 1.35, 1.45, 1.50, 2.00):
        for seed in range(30):
            t, ok, dLdw2, loss, tr = run(a, seed, 2_000, record=(seed < 5))
            rows.append({"a": a, "seed": seed, "budget": 2_000,
                         "w2_final": abs(float(t[2])), "solved": bool(ok),
                         "dL_dw2_final": dLdw2, "loss_final": loss})
            for i, w2, g2, l in tr:
                traces.append({"a": a, "seed": seed, "step": i, "w2": w2,
                               "dL_dw2": g2, "loss": l})
        print(f"2a a={a} done", flush=True)

    # 2c: budget dependence
    for a in (1.25, 1.35, 1.45):
        for budget in (2_000, 4_000, 8_000, 20_000):
            solved = 0
            w2s = []
            for seed in range(40):
                t, ok, dLdw2, loss, _ = run(a, seed, budget)
                solved += bool(ok)
                w2s.append(abs(float(t[2])))
            rows.append({"a": a, "seed": -1, "budget": budget,
                         "w2_final": float(np.median(w2s)),
                         "solved": solved / 40, "dL_dw2_final": float("nan"),
                         "loss_final": float("nan")})
            print(f"2c a={a} budget={budget}: rate={solved/40:.3f} "
                  f"w2_med={np.median(w2s):.2f}", flush=True)

    for name, frame in (("termination", pd.DataFrame(rows)),
                        ("termination_traces", pd.DataFrame(traces))):
        stem = RESULTS / name
        with artifact_lock(stem, name):
            tmp = stem.with_suffix(".csv.tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
