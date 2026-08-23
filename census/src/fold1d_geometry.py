"""Task B Parts 2a-2c: exact solution sets, basin volumes, estimator validation.

Registered volume prediction (before running, in-code registration mirrored
in results/fold1d_prediction.md's spirit and stated in the writeup):
basin volume grows sharply near the observed onset (a ~ 1.35-1.5) while
solution-set measure grows smoothly from the analytic threshold a = 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import activation, logits, make_data, solves


def solution_mask(name: str, parameter: float | None, box: float, n_grid: int) -> np.ndarray:
    """Boolean mask over the (w1, b1, w2, b2) grid: exact region solve."""

    f = activation(name, parameter)
    axis = torch.linspace(-box, box, n_grid)
    inner = torch.linspace(-0.8, 0.8, 401)
    outer_pos = torch.linspace(1.2, 2.0, 201)
    outer = torch.cat([outer_pos, -outer_pos])
    # broadcast: theta grid (n^4) x data points — evaluate in chunks over w1,b1
    mask = np.zeros((n_grid,) * 4, dtype=bool)
    for i, w1 in enumerate(axis):
        pre_inner = f(w1 * inner.unsqueeze(0) + axis.unsqueeze(1))    # (b1, x)
        pre_outer = f(w1 * outer.unsqueeze(0) + axis.unsqueeze(1))
        for k, w2 in enumerate(axis):
            si = w2 * pre_inner  # (b1, x)
            so = w2 * pre_outer
            # need max over inner of (si + b2) < 0 and min over outer > 0:
            # b2 < -max(si) and b2 > -min(so)
            hi = -si.max(dim=1).values   # (b1,)
            lo = -so.min(dim=1).values
            # count b2 grid points strictly inside (lo, hi)
            for j in range(len(axis)):
                if lo[j] < hi[j]:
                    inside = (axis > lo[j]) & (axis < hi[j])
                    mask[i, j, k, inside.numpy()] = True
    return mask


def basin_mask(name: str, parameter: float | None, box: float, n_grid: int,
               steps: int = 2_000) -> np.ndarray:
    """SGD from every grid initialization; True where training reaches a solve."""

    f = activation(name, parameter)
    x, y = make_data(200, 0)
    axis = np.linspace(-box, box, n_grid)
    mask = np.zeros((n_grid,) * 4, dtype=bool)
    coordinates = [(i, j, k, l) for i in range(n_grid) for j in range(n_grid)
                   for k in range(n_grid) for l in range(n_grid)]
    # batch all initializations as one parameter tensor for vectorized Adam
    thetas = torch.tensor(
        [[axis[i], axis[j], axis[k], axis[l]] for i, j, k, l in coordinates],
        dtype=torch.float32, requires_grad=True,
    )
    optimizer = torch.optim.Adam([thetas], lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pre = f(thetas[:, 0:1] * x.unsqueeze(0) + thetas[:, 1:2])
        out = thetas[:, 2:3] * pre + thetas[:, 3:4]
        loss = F.binary_cross_entropy_with_logits(
            out, y.unsqueeze(0).expand_as(out), reduction="mean"
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        for index, (i, j, k, l) in enumerate(coordinates):
            mask[i, j, k, l] = solves(thetas[index].detach(), f, n_check=801)
    return mask


def erosion_radius(mask: np.ndarray, box: float) -> float:
    """Inradius in grid units via repeated 4D 3^4 erosion, converted to units."""

    n_grid = mask.shape[0]
    cell = 2 * box / (n_grid - 1)
    current = mask.copy()
    rounds = 0
    while current.any():
        padded = np.pad(current, 1, constant_values=False)
        eroded = np.ones_like(current)
        for offset0 in (0, 1, 2):
            for offset1 in (0, 1, 2):
                for offset2 in (0, 1, 2):
                    for offset3 in (0, 1, 2):
                        eroded &= padded[
                            offset0:offset0 + n_grid,
                            offset1:offset1 + n_grid,
                            offset2:offset2 + n_grid,
                            offset3:offset3 + n_grid,
                        ]
        if not eroded.any():
            break
        current = eroded
        rounds += 1
    return rounds * cell


def connected_components(mask: np.ndarray) -> int:
    """4D flood fill, 8-neighbour (axis-aligned) connectivity."""

    remaining = mask.copy()
    count = 0
    shape = mask.shape
    while remaining.any():
        count += 1
        seed_index = tuple(int(v) for v in np.argwhere(remaining)[0])
        stack = [seed_index]
        remaining[seed_index] = False
        while stack:
            point = stack.pop()
            for axis_index in range(4):
                for delta in (-1, 1):
                    neighbour = list(point)
                    neighbour[axis_index] += delta
                    if 0 <= neighbour[axis_index] < shape[axis_index]:
                        neighbour = tuple(neighbour)
                        if remaining[neighbour]:
                            remaining[neighbour] = False
                            stack.append(neighbour)
    return count


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    rows = []
    values = [("sin_family", a) for a in (1.05, 1.1, 1.25, 1.35, 1.5, 2.0, 3.0)]
    values += [("pwl_family", -0.05), ("pwl_family", -0.25), ("gelu", None)]
    for box in (5.0, 10.0):
        n_sol = 41  # solution-set grid
        n_bas = 9   # basin grid (SGD from 6561 inits)
        for name, parameter in values:
            solution = solution_mask(name, parameter, box, n_sol)
            fraction = float(solution.mean())
            components = connected_components(solution) if fraction > 0 else 0
            inradius = erosion_radius(solution, box) if fraction > 0 else 0.0
            basin = basin_mask(name, parameter, box, n_bas)
            basin_fraction = float(basin.mean())
            rows.append({
                "activation": name, "parameter": parameter, "box": box,
                "solution_fraction": fraction, "components": components,
                "solution_inradius": inradius,
                "basin_fraction": basin_fraction,
                "ratio_basin_to_solution": basin_fraction / fraction if fraction else None,
            })
            print(f"box={box} {name}({parameter}): solution={fraction:.5f} "
                  f"components={components} inradius={inradius:.2f} "
                  f"basin={basin_fraction:.4f}", flush=True)
            frame = pd.DataFrame(rows)
            stem = directory / "fold1d_geometry"
            with artifact_lock(stem, "1d geometry"):
                temp = stem.with_suffix(".csv.tmp")
                frame.to_csv(temp, index=False)
                temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
