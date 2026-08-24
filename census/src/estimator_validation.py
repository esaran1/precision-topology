"""Reconstructed Task B 2c harness (audit 2026-08-24, AUDIT.md finding 7).

``estimator_validation.csv`` and ``estimator_anisotropy.csv`` were
committed (66672be) without their generating source.  This module is the
committed replacement.  Two honesty notes:

- The **found** object is exactly reproducible: it is the fold1d solver
  at (sin_family, a = 1.5, seed = FOUND_SEED), rebuilt deterministically
  via ``fold1d.train_one``.
- The **constructed** object is re-derived from its recorded design
  constants (fold at the local max t* of f_1.5, shrink 0.1,
  amplification |w2| = 23), because the original theta was never
  persisted.  It is verified to solve exactly before use; rates for the
  constructed object are therefore comparable-by-procedure, not
  bit-identical to the original run.

Estimator under validation: perturb-retrain with **per-coordinate
|theta_i|-relative** Gaussian noise, Adam lr 1e-2, 2,000 retraining
steps on the found solver's own data seed, recovery judged by the exact
region check ``fold1d.solves``.  The noise convention was recovered by
reconciliation: whole-vector RMS-relative noise gives systematically
lower recovery (found eps=0.3: 56% vs committed 87%) while
per-coordinate relative noise reproduces every committed cell within
binomial noise (92/100 vs 87.3%, 76/100 vs 79%, and exact agreement at
smaller eps).  Per-coordinate is also the direct analogue of
``basin.py``'s per-tensor RMS rule when each scalar is its own
parameter tensor.  The RMS-convention run is preserved as
``estimator_*_rms_sensitivity.csv`` — the ~30-point spread between
conventions at eps=0.3 is itself a caveat on any radius-style number.
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import activation, logits, make_data, solves, train_one

A = 1.5
FOUND_SEED = 1          # first solved seed at a = 1.5 in fold1d_sweep.csv
SHRINK = 0.1
AMPLIFICATION = 23.0
EPSILONS = (0.003, 0.01, 0.03, 0.1, 0.3)
TRUTH_DRAWS = 300
ANISO_EPSILONS = (0.01, 0.1, 0.3, 1.0)
ANISO_DRAWS = 40
RETRAIN_STEPS = 2_000


def found_theta() -> torch.Tensor:
    f = activation("sin_family", A)
    x, y = make_data(200, FOUND_SEED)
    torch.manual_seed(FOUND_SEED)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=1e-2)
    for _ in range(2_000):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
        optimizer.step()
    theta = theta.detach()
    if not solves(theta, f):
        raise RuntimeError("found object failed to reproduce its solve")
    return theta


def constructed_theta() -> torch.Tensor:
    """Fold-at-t* construction from the recorded design constants."""

    f = activation("sin_family", A)
    t_star = math.pi - math.acos(1.0 / A)  # local max of f_a
    w1, b1 = SHRINK, t_star
    w2 = -AMPLIFICATION
    edge = 0.5 * (float(f(torch.tensor(w1 + b1))) + float(f(torch.tensor(-w1 + b1))))
    b2 = -w2 * edge
    theta = torch.tensor([w1, b1, w2, b2], dtype=torch.float32)
    if not solves(theta, f):
        raise RuntimeError("constructed object does not solve; design constants wrong")
    return theta


def perturb_retrain(theta: torch.Tensor, noise: torch.Tensor) -> bool:
    f = activation("sin_family", A)
    x, y = make_data(200, FOUND_SEED)
    start = (theta + noise).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([start], lr=1e-2)
    for _ in range(RETRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(start, x, f), y).backward()
        optimizer.step()
    return solves(start.detach(), f)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    objects = {"found": found_theta(), "constructed": constructed_theta()}
    for name, theta in objects.items():
        print(name, "theta:", [round(float(v), 4) for v in theta], flush=True)

    rows = []
    for name, theta in objects.items():
        for epsilon in EPSILONS:
            generator = torch.Generator().manual_seed(
                zlib.crc32(f"{name}|{epsilon}".encode()) % (2**31))
            recovered = 0
            for _ in range(TRUTH_DRAWS):
                noise = torch.randn(4, generator=generator) * epsilon * theta.abs()
                recovered += perturb_retrain(theta, noise)
            rows.append({"object": name, "epsilon": epsilon,
                         "n": TRUTH_DRAWS, "recovered": recovered})
            print(rows[-1], flush=True)
    stem = directory / "estimator_validation_recon"
    with artifact_lock(stem, "estimator validation reconstruction"):
        temp = stem.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))

    aniso = []
    for name, theta in objects.items():
        for coord in range(4):
            for epsilon in ANISO_EPSILONS:
                generator = torch.Generator().manual_seed(
                    zlib.crc32(f"{name}|{coord}|{epsilon}".encode()) % (2**31))
                recovered = 0
                for _ in range(ANISO_DRAWS):
                    noise = torch.zeros(4)
                    noise[coord] = float(torch.randn(1, generator=generator)) * epsilon * float(theta[coord].abs())
                    recovered += perturb_retrain(theta, noise)
                aniso.append({"object": name, "coord": coord,
                              "epsilon": epsilon, "recovered": recovered,
                              "n": ANISO_DRAWS})
                print(aniso[-1], flush=True)
    stem = directory / "estimator_anisotropy_recon"
    with artifact_lock(stem, "estimator anisotropy reconstruction"):
        temp = stem.with_suffix(".csv.tmp")
        pd.DataFrame(aniso).to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
