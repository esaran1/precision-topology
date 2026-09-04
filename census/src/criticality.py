"""Part 1: are the zero-basin points critical points at all?

Reports ||grad L||, full Hessian spectrum (including lambda_min), and loss,
for zero-basin constructed solutions and for SGD-found solutions, plus
typical gradient norms along a training trajectory for scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .box_counterexample import counterexample
from .fold1d import activation, logits, make_data, solves
from .sharpness import hessian_eigs, loss_fn, train

RESULTS = Path(__file__).resolve().parents[1] / "results"


def grad_and_loss(theta: torch.Tensor, a: float, data_seed: int = 0):
    f = activation("sin_family", a)
    x, y = make_data(200, data_seed)
    x, y = x.double(), y.double()
    t = theta.double().clone().requires_grad_(True)
    loss = loss_fn(t, x, y, f)
    (g,) = torch.autograd.grad(loss, t)
    return float(g.norm()), float(loss), g.detach().numpy()


def trajectory_grad_norms(a: float, seed: int = 0, steps: int = 2_000):
    """Gradient norms along a standard Adam run, for scale."""
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)
    norms = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(logits(theta, x, f), y)
        loss.backward()
        norms.append(float(theta.grad.norm()))
        opt.step()
    return np.array(norms)


def main() -> None:
    rows = []
    for a in (1.02, 1.10, 1.25, 1.35, 1.45, 1.50, 2.00, 3.00):
        traj = trajectory_grad_norms(a)
        traj_start, traj_end = float(np.median(traj[:100])), float(np.median(traj[-100:]))

        spec = counterexample(a)
        tc = torch.tensor([spec["w1"], spec["b1"], spec["w2"], spec["b2"]],
                          dtype=torch.float64)
        gn, loss, _ = grad_and_loss(tc, a)
        e = hessian_eigs(tc, a)
        rows.append({"population": "zero_basin_constructed", "a": a, "seed": -1,
                     "w2": float(tc[2]), "grad_norm": gn, "loss": loss,
                     "lambda_min": float(e[0]), "lambda_max": float(e[-1]),
                     "traj_grad_start": traj_start, "traj_grad_end": traj_end})

        for seed in range(40):
            t, ok, _ = train(a, seed, 1e-2, 2_000, "adam")
            if ok:
                gn, loss, _ = grad_and_loss(t.double(), a, seed)
                e = hessian_eigs(t, a, seed)
                rows.append({"population": "found_adam", "a": a, "seed": seed,
                             "w2": float(t[2]), "grad_norm": gn, "loss": loss,
                             "lambda_min": float(e[0]), "lambda_max": float(e[-1]),
                             "traj_grad_start": traj_start, "traj_grad_end": traj_end})
                break
        print(f"a={a} done", flush=True)

    frame = pd.DataFrame(rows)
    stem = RESULTS / "criticality"
    with artifact_lock(stem, "criticality"):
        tmp = stem.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
