"""Design (not registered, not run): is the free-training residual adiabatic lag of w2?  results/lag_test_design.md.

Cells: a ∈ {1.30, 1.50} × w2 learning-rate factor φ ∈ {0.25, 0.5, 1, 2} at n = 6,400, on the size test's 50 seeds
(300,000-300,049; the same training sets, whose own thresholds sample_size_own.csv already holds).  Everything else is
the phase 2b protocol: float64, full batch, Adam lr 1e-2, torch.manual_seed(seed) then U(-1, 1)^4.  w2's learning rate is
1e-2·φ: after each standard Adam step, w2's update is rescaled by φ (identical to a per-parameter learning rate; at φ = 1
no operation, so φ = 1 is the reference runs' code path exactly).  Budget 32,000/φ.  Crossing: the first step with dense-grid
oriented gap > 0; R_cross = |w2| Ĝ_cert/2.

    python -m src.lag_test reproduce [workers]     # validity: φ = 1 against the size test's n = 6,400 runs, exactly
    python -m src.lag_test run [workers]
    python -m src.lag_test score
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.30, 1.50)
FACTORS = (0.25, 0.5, 1.0, 2.0)
N = 6400
SEED0, N_SEEDS = 300_000, 50
BASE_BUDGET = 32_000
MIN_CROSSINGS = 40


def scale_w2_step(theta, w2_before, factor):
    """w2's learning rate x factor: after the standard Adam step, w2's update is rescaled by factor (Adam's moment
    estimates are unaffected, as with a per-parameter learning rate).  factor = 1: no operation at all."""
    if factor != 1.0:
        import torch
        with torch.no_grad():
            theta[2] = w2_before + factor * (float(theta.detach()[2]) - w2_before)


def run_one(args):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation
    from .phase2b_ordering import state
    from .sample_size import _data, _pop
    a, seed, factor = args
    torch.set_num_threads(1)
    G = _pop(a)[0]
    f = activation("sin_family", a)
    xt, yt = (torch.tensor(v) for v in _data(N, seed))
    from .fold1d import logits
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)                          # the reference runs' optimiser, unchanged
    budget = int(round(BASE_BUDGET / factor))
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, xt, f), yt).backward()
        w2_before = float(theta.detach()[2])
        opt.step()
        scale_w2_step(theta, w2_before, factor)
        st = state(theta.detach(), f, a, G)
        if st["placement_ok"]:
            return {"a": a, "seed": seed, "factor": factor, "budget": budget, "cross_step": step,
                    "w2_abs": abs(st["w2"]), "R_cross": abs(st["w2"]) * G / 2}
    return {"a": a, "seed": seed, "factor": factor, "budget": budget, "cross_step": np.nan, "w2_abs": np.nan,
            "R_cross": np.nan}


def check_reproduce(results=None, name="lag_test_reproduce.csv"):
    """Validity: φ = 1 must reproduce the size test's n = 6,400 runs exactly -- crossing step and |w2| at crossing
    bit-identical (exact round-trip parsing on both sides), non-crossers on both sides."""
    R = results or RESULTS
    ref = pd.read_csv(R / "sample_size_free.csv", float_precision="round_trip")
    ref = ref[ref.n == N][["a", "seed", "cross_step", "w2_abs"]]
    new = pd.read_csv(R / name, float_precision="round_trip")
    new = new[new.factor == 1.0][["a", "seed", "cross_step", "w2_abs"]]
    m = ref.merge(new, on=["a", "seed"], suffixes=("_ref", "_new"))
    same = len(m) == len(new) and len(m) > 0 and all(
        np.array_equal(m[c + "_ref"].values, m[c + "_new"].values, equal_nan=True) for c in ("cross_step", "w2_abs"))
    if not same:
        raise SystemExit(f"STOP: phi = 1 does not reproduce the standard runs exactly ({len(m)} rows compared)")
    return len(m)


def reproduce(workers=3, seeds=None):
    seeds = seeds if seeds is not None else range(SEED0, SEED0 + N_SEEDS)
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(run_one, [(a, s, 1.0) for a in A_VALUES for s in seeds], chunksize=1))
    d.to_csv(RESULTS / "lag_test_reproduce.csv", index=False)
    print("rows reproduced exactly:", check_reproduce())


def run(workers=3):
    """All cells; φ = 1 included, and checked for exact reproduction before anything else is used."""
    jobs = [(a, s, fct) for fct in FACTORS for a in A_VALUES for s in range(SEED0, SEED0 + N_SEEDS)]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(run_one, jobs, chunksize=1))
    d.to_csv(RESULTS / "lag_test_runs.csv", index=False)
    print("phi = 1 reproduced exactly on", check_reproduce(name="lag_test_runs.csv"), "runs")


def _boot(v, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return np.median(rng.choice(v, (B, len(v))), axis=1)


def score(results=None):
    """Registered analysis (see the design): residual(φ) = median(R_cross/R_own) - 1 per (a, φ), R_own the own
    threshold (sample_size_own.csv at n = 6,400; branch-matched if registered so)."""
    R = results or RESULTS
    check_reproduce(R, "lag_test_runs.csv")
    runs = pd.read_csv(R / "lag_test_runs.csv", float_precision="round_trip")
    own = pd.read_csv(R / "sample_size_own.csv", float_precision="round_trip")
    own = own[own.n == N][["a", "seed", "w2_own"]]
    d = runs.merge(own, on=["a", "seed"])
    from .sample_size import _pop
    cells, tests = [], []
    for a in A_VALUES:
        G = _pop(a)[0]
        v, boots = {}, {}
        for fct in FACTORS:
            g = d[(d.a.round(2) == a) & (d.factor == fct) & d.R_cross.notna()]
            r = (g.R_cross / (g.w2_own * G / 2)).values
            v[fct], boots[fct] = float(np.median(r) - 1), _boot(r) - 1
            cells.append({"a": a, "factor": fct, "n_crossed": len(r), "n_runs": int(((d.a.round(2) == a) & (d.factor == fct)).sum()),
                          "residual": v[fct], "ci_lo": float(np.percentile(boots[fct], 2.5)),
                          "ci_hi": float(np.percentile(boots[fct], 97.5))})
        d1 = np.percentile(boots[1.0] - boots[0.25], [2.5, 97.5])
        d2 = np.percentile(boots[2.0] - boots[0.25], [2.5, 97.5])
        l1 = (v[0.25] < v[0.5] < v[1.0] < v[2.0]) and d1[0] > 0 and v[0.25] <= 0.5 * v[1.0]
        comp = d2[0] <= 0 <= d2[1]
        tests.append({"a": a, "res_1_minus_025_lo": d1[0], "res_1_minus_025_hi": d1[1], "res_2_minus_025_lo": d2[0],
                      "res_2_minus_025_hi": d2[1], "L1_pass": l1, "competing_no_dependence": comp})
    pd.DataFrame(cells).to_csv(R / "lag_test_cells.csv", index=False)
    pd.DataFrame(tests).to_csv(R / "lag_test_tests.csv", index=False)
    print(pd.DataFrame(cells).to_string(index=False)); print(pd.DataFrame(tests).T.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("LT_WORKERS", "3"))
    {"reproduce": lambda: reproduce(w), "run": lambda: run(w), "score": score}[cmd]()
