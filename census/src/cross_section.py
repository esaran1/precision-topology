"""Successor hypothesis: capture limited by the solution sheet's cross-section.

P-step: smaller optimizer steps raise findability near the onset (a thin
sheet is overshot by a large step).  Opposite sign to the energetic
account, which is already falsified (barrier_results.md).

P-ratio: sheet thickness per direction / Adam's typical step in that
direction predicts which a are findable.

Also Part 4: basin volume along the solution manifold vs |w2|.
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

N_SEEDS = 200
STEPS = 2_000


def train_one(a: float, seed: int, lr: float, steps: int = STEPS,
              optimizer_name: str = "adam", measure_steps: bool = False):
    """Train one run; optionally also return the realized per-coordinate step.

    Adam normalizes by gradient magnitude, so nominal lr and realized step
    differ by up to ~8x over our sweep range (measured).  The hypothesis is
    about the realized step, so it is recorded alongside.  Plain SGD is run
    as a second arm because there the step is what you set it to.
    """

    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    optimizer = (torch.optim.Adam([theta], lr=lr) if optimizer_name == "adam"
                 else torch.optim.SGD([theta], lr=lr))
    deltas = []
    previous = theta.detach().clone() if measure_steps else None
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
        optimizer.step()
        if measure_steps and index < 300:
            deltas.append((theta.detach() - previous).abs().numpy())
            previous = theta.detach().clone()
    solved = solves(theta.detach(), f)
    if measure_steps:
        realized = np.median(np.stack(deltas), axis=0) if deltas else np.zeros(4)
        return solved, realized
    return solved


GRIDS = {
    # optimizer: (learning rates, step budget at lr=1e-2 equivalent)
    "adam": (3e-2, 1e-2, 3e-3, 1e-3, 3e-4),
    "sgd": (1.0, 0.3, 0.1, 0.03, 0.01),
}


def step_sweep() -> pd.DataFrame:
    """P-step: findability vs step size, Adam and plain SGD.

    Realized per-coordinate step is measured on the first 20 seeds of each
    cell, since Adam's realized step is not proportional to nominal lr.
    """

    rows = []
    for optimizer_name in ("adam", "sgd"):
        for a in (1.25, 1.35, 1.45, 1.50):
            for lr in GRIDS[optimizer_name]:
                reference = 1e-2 if optimizer_name == "adam" else 0.1
                steps = min(int(STEPS * (reference / lr)), 40_000) if lr < reference else STEPS
                solved = 0
                realized = []
                for seed in range(N_SEEDS):
                    if seed < 20:
                        ok, step = train_one(a, seed, lr, steps, optimizer_name, True)
                        realized.append(step)
                    else:
                        ok = train_one(a, seed, lr, steps, optimizer_name)
                    solved += bool(ok)
                median = np.median(np.stack(realized), axis=0)
                rows.append({"optimizer": optimizer_name, "a": a, "lr": lr,
                             "steps": steps, "solved": solved, "n": N_SEEDS,
                             **{f"realized_{n}": float(v) for n, v in
                                zip(("w1", "b1", "w2", "b2"), median)},
                             "realized_mean": float(median.mean())})
                print(f"{optimizer_name} a={a} lr={lr:g} steps={steps}: "
                      f"{solved}/{N_SEEDS}  realized_mean={median.mean():.2e}", flush=True)
    return pd.DataFrame(rows)


def sheet_thickness(a: float, n_probe: int = 20_001) -> dict:
    """Width of the solution set in each coordinate, through a solution.

    Uses the analytic construction as the reference point (it exists at
    every a > 1) and scans each coordinate holding the others fixed.
    """

    from .box_counterexample import counterexample

    spec = counterexample(a)
    theta = torch.tensor([spec["w1"], spec["b1"], spec["w2"], spec["b2"]],
                         dtype=torch.float64)
    f = activation("sin_family", a)
    out = {"a": a}
    for index, name in enumerate(("w1", "b1", "w2", "b2")):
        base = float(theta[index])
        span = max(abs(base), 1.0) * 2.0
        offsets = torch.linspace(-span, span, n_probe, dtype=torch.float64)
        ok = []
        for offset in offsets:
            probe = theta.clone()
            probe[index] = base + offset
            ok.append(solves(probe, f, n_check=801))
        ok = np.array(ok)
        # contiguous run containing the centre
        centre = n_probe // 2
        if not ok[centre]:
            out[f"thickness_{name}"] = 0.0
            continue
        left = centre
        while left > 0 and ok[left - 1]:
            left -= 1
        right = centre
        while right < n_probe - 1 and ok[right + 1]:
            right += 1
        step = float(offsets[1] - offsets[0])
        out[f"thickness_{name}"] = (right - left) * step
    return out


def adam_steps(a: float, seeds: range = range(20), probe_steps: int = 200) -> dict:
    """Typical per-coordinate Adam step size early in training."""

    f = activation("sin_family", a)
    magnitudes = []
    for seed in seeds:
        x, y = make_data(200, seed)
        torch.manual_seed(seed)
        theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=1e-2)
        previous = theta.detach().clone()
        deltas = []
        for _ in range(probe_steps):
            optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
            optimizer.step()
            deltas.append((theta.detach() - previous).abs().numpy())
            previous = theta.detach().clone()
        magnitudes.append(np.median(np.stack(deltas), axis=0))
    median = np.median(np.stack(magnitudes), axis=0)
    return {"a": a, **{f"step_{n}": float(v) for n, v in zip(("w1", "b1", "w2", "b2"), median)}}


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"

    print("=== P-ratio: sheet thickness vs Adam step ===", flush=True)
    rows = []
    for a in (1.02, 1.10, 1.25, 1.35, 1.45, 1.50, 2.00, 3.00):
        thick = sheet_thickness(a)
        steps = adam_steps(a)
        rows.append({**thick, **{k: v for k, v in steps.items() if k != "a"}})
        print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    stem = directory / "sheet_geometry"
    with artifact_lock(stem, "sheet geometry"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))

    print("\n=== P-step: findability vs step size ===", flush=True)
    sweep = step_sweep()
    stem = directory / "step_size_sweep"
    with artifact_lock(stem, "step size sweep"):
        temp = stem.with_suffix(".csv.tmp")
        sweep.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
