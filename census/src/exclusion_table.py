"""The exclusion table: four standard explanations, each ruled out by measurement.

For every zero-basin (constructed) solution and every found solution:
  barrier   -- minimum-energy-path barrier from typical init (string method)
  sharpness -- lambda_max * eta_eff against the 2/eta stability threshold
  distance  -- ||theta_solution - theta_init|| against the distance training travels
  margin    -- logit margin (the found solution sometimes has the SMALLER one)

Each column is a direct measurement that could have come out the other way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .barrier import initialization, linear_barrier, string_barrier
from .box_counterexample import counterexample
from .fold1d import activation, logits, make_data, solves
from .sharpness import hessian_eigs, train

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_INITS = 20


def margin_of(theta: torch.Tensor, a: float) -> float:
    """Minimum logit margin over both class regions."""
    from .fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX
    f = activation("sin_family", a)
    inner = torch.linspace(-INNER_MAX, INNER_MAX, 4001, dtype=torch.float64)
    pos = torch.linspace(OUTER_MIN, OUTER_MAX, 2001, dtype=torch.float64)
    outer = torch.cat([pos, -pos])
    t = theta.double()
    with torch.no_grad():
        li = logits(t, inner, f)
        lo = logits(t, outer, f)
    return min(float(-li.max()), float(lo.min()))


def row_for(theta: torch.Tensor, a: float, population: str, seed: int,
            realized_step: float, x, y) -> dict:
    eigs = hessian_eigs(theta, a)
    dists = [float((theta.double() - initialization(i)).norm()) for i in range(N_INITS)]
    barriers_lin = [linear_barrier(initialization(i), theta.double(), a, x, y)
                    for i in range(N_INITS)]
    barriers_mep = [string_barrier(initialization(i), theta.double(), a, x, y)
                    for i in range(5)]
    eta = realized_step if realized_step == realized_step else 1e-2
    return {
        "population": population, "a": a, "seed": seed,
        "w2": float(theta[2]),
        "barrier_mep_median": float(np.median(barriers_mep)),
        "barrier_mep_max": float(np.max(barriers_mep)),
        "barrier_linear_median": float(np.median(barriers_lin)),
        "lambda_max": float(eigs[-1]),
        "eta_eff": eta,
        "sharpness_product": float(eigs[-1]) * eta,
        "distance_median": float(np.median(dists)),
        "distance_min": float(np.min(dists)),
        "margin": margin_of(theta, a),
    }


def main() -> None:
    x, y = make_data(200, 0)
    x, y = x.double(), y.double()
    rows = []
    for a in (1.02, 1.10, 1.25, 1.35, 1.45, 1.50, 2.00, 3.00):
        spec = counterexample(a)
        tc = torch.tensor([spec["w1"], spec["b1"], spec["w2"], spec["b2"]],
                          dtype=torch.float64)
        rows.append(row_for(tc, a, "zero_basin_constructed", -1, float("nan"), x, y))
        print(rows[-1]["a"], "constructed done", flush=True)

        # a matched found solution, where one exists
        for seed in range(40):
            t, ok, rz = train(a, seed, 1e-2, 2_000, "adam")
            if ok:
                rows.append(row_for(t.double(), a, "found_adam", seed,
                                    float(np.median(rz)), x, y))
                print(rows[-1]["a"], "found done", flush=True)
                break
        frame = pd.DataFrame(rows)
        stem = RESULTS / "exclusion_table"
        with artifact_lock(stem, "exclusion table"):
            tmp = stem.with_suffix(".csv.tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
