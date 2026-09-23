"""Exploratory (labelled): the Block 4 checkpoints placed at R/R_glob = 0.9 after 4,000 steps.

For each such run: its own per-seed certified R_glob (the certified procedure on the run's 400-point training
set, conditional_certified.scan), expressed relative to the population R_glob.  If a run's own threshold lies
below 0.9 x the population threshold, its placement at 0.9 is its own equilibrium (per-seed threshold), not
overshoot.  Long-horizon persistence and branch distances come from fixed_scale_horizons.csv.

    python -m src.diagnose_below
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _job(seed):
    from .conditional_certified import _seed_data, scan
    x, y = _seed_data(int(seed))
    grid = [round(v, 4) for v in np.arange(2.0, 9.0 + 1e-9, 0.5)]
    res, _ = scan(1.30, x, y, symmetric=False, grid=grid)
    res["seed"] = int(seed)
    return res


def main():
    d = pd.read_csv(RESULTS / "fixed_scale_block4.csv")
    seeds = sorted(d[(d.level == 0.9) & (d.variant == "preserved") & d.placed_end.astype(bool)].seed.unique())
    with Pool(int(os.environ.get("DB_WORKERS", "2"))) as p:
        out = p.map(_job, seeds)
    t = pd.DataFrame(out)
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    b = br[(br.a.round(2) == 1.30) & (br.kind == "glob")].iloc[0]
    w2_pop = 0.5 * (b.w2_lo + b.w2_hi)
    t["own_w2_glob_mid"] = 0.5 * (t.w2_glob_lo + t.w2_glob_hi)
    t["own_over_pop"] = t.own_w2_glob_mid / w2_pop
    t["own_threshold_below_0p9"] = t.own_over_pop < 0.9
    t.to_csv(RESULTS / "diagnose_below.csv", index=False)
    print(t[["seed", "w2_glob_lo", "w2_glob_hi", "own_over_pop", "own_threshold_below_0p9"]].to_string(index=False))


if __name__ == "__main__":
    main()
