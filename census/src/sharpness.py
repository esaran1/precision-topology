"""Part 1: Hessian sharpness of 1D-task solutions vs the 2/eta threshold.

Registered in results/arrhenius_prediction.md.  Loss is full-batch BCE on the
400-point task sample, so the Hessian is exact and 4x4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, activation, logits,
                     make_data, solves)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def loss_fn(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor, f) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits(theta, x, f), y)


def hessian_eigs(theta: torch.Tensor, a: float, data_seed: int = 0) -> np.ndarray:
    f = activation("sin_family", a)
    x, y = make_data(200, data_seed)
    x, y = x.double(), y.double()
    t = theta.double().clone().requires_grad_(True)
    H = torch.autograd.functional.hessian(lambda p: loss_fn(p, x, y, f), t)
    return np.linalg.eigvalsh(H.detach().numpy())


def train(a: float, seed: int, lr: float = 1e-2, steps: int = 2_000,
          optimizer_name: str = "adam", scale: float = 1.0):
    """Return (theta, solved, realized_step_median_per_coord)."""

    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = (torch.empty(4).uniform_(-1.0, 1.0) * scale).requires_grad_(True)
    opt = (torch.optim.Adam([theta], lr=lr) if optimizer_name == "adam"
           else torch.optim.SGD([theta], lr=lr))
    prev = theta.detach().clone()
    deltas = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_fn(theta, x, y, f).backward()
        opt.step()
        if i >= steps - 300:          # realized step near convergence
            deltas.append((theta.detach() - prev).abs().numpy())
        prev = theta.detach().clone()
    t = theta.detach()
    realized = np.median(np.stack(deltas), axis=0) if deltas else np.zeros(4)
    return t, solves(t, f), realized


def manifold_point(a: float, w2: float) -> torch.Tensor | None:
    """Centred solution at this |w2| (same construction as manifold_basin)."""

    import math
    f = activation("sin_family", a)
    c = math.acos(1.0 / a)
    b1 = math.pi + c
    inner = torch.linspace(-INNER_MAX, INNER_MAX, 4001, dtype=torch.float64)
    pos = torch.linspace(OUTER_MIN, OUTER_MAX, 2001, dtype=torch.float64)
    outer = torch.cat([pos, -pos])
    best = None
    for frac in np.linspace(0.02, 0.99, 120):
        w1 = frac * c
        gap = float(f(w1 * outer + b1).min() - f(w1 * inner + b1).max())
        if gap > 0 and (best is None or gap > best[1]):
            best = (w1, gap)
    if best is None:
        return None
    w1, _ = best
    vi = f(w1 * inner + b1); vo = f(w1 * outer + b1)
    b2 = -w2 * float(vo.min() + vi.max()) / 2.0
    theta = torch.tensor([w1, b1, w2, b2], dtype=torch.float64)
    return theta if solves(theta, f) else None


def main() -> None:
    rows = []
    from .box_counterexample import counterexample

    # --- 1a/1b: found (Adam & SGD) vs constructed populations ---
    for a in (1.25, 1.35, 1.45, 1.50, 2.00, 3.00):
        for seed in range(40):
            t, ok, rz = train(a, seed, 1e-2, 2_000, "adam")
            if not ok:
                continue
            e = hessian_eigs(t, a, seed)
            rows.append({"population": "found_adam", "a": a, "seed": seed,
                         "w2": float(t[2]), "lambda_max": float(e[-1]),
                         "lambda_min": float(e[0]),
                         "realized_step": float(np.median(rz)),
                         "lr_nominal": 1e-2})
        spec = counterexample(a)
        tc = torch.tensor([spec["w1"], spec["b1"], spec["w2"], spec["b2"]])
        e = hessian_eigs(tc, a)
        rows.append({"population": "constructed_w2_1", "a": a, "seed": -1,
                     "w2": float(tc[2]), "lambda_max": float(e[-1]),
                     "lambda_min": float(e[0]),
                     "realized_step": float("nan"), "lr_nominal": float("nan")})
        print(f"a={a}: done", flush=True)

    # --- 1c: sharpness along the manifold ---
    for a in (1.5,):
        for w2 in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0):
            t = manifold_point(a, w2)
            if t is None:
                continue
            e = hessian_eigs(t, a)
            rows.append({"population": "manifold", "a": a, "seed": -1,
                         "w2": w2, "lambda_max": float(e[-1]),
                         "lambda_min": float(e[0]),
                         "realized_step": float("nan"), "lr_nominal": float("nan")})
    print("manifold done", flush=True)

    # --- 1d: sharpness of reachable solutions vs a ---
    for a in (1.02, 1.05, 1.10, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 2.00, 3.00):
        t = manifold_point(a, 5.0)
        if t is not None:
            e = hessian_eigs(t, a)
            rows.append({"population": "manifold_w2_5", "a": a, "seed": -1,
                         "w2": 5.0, "lambda_max": float(e[-1]),
                         "lambda_min": float(e[0]),
                         "realized_step": float("nan"), "lr_nominal": float("nan")})
        spec = counterexample(a)
        tc = torch.tensor([spec["w1"], spec["b1"], spec["w2"], spec["b2"]])
        e = hessian_eigs(tc, a)
        rows.append({"population": "constructed_vs_a", "a": a, "seed": -1,
                     "w2": float(tc[2]), "lambda_max": float(e[-1]),
                     "lambda_min": float(e[0]),
                     "realized_step": float("nan"), "lr_nominal": float("nan")})
    print("vs-a done", flush=True)

    frame = pd.DataFrame(rows)
    stem = RESULTS / "sharpness"
    with artifact_lock(stem, "sharpness"):
        tmp = stem.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
