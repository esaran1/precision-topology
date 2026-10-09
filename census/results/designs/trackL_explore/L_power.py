"""EXPLORATORY power (Track L, L2): P(criterion passes) at n seeds by resampling the exploration's per-seed paired
differences (arm - std, MNIST-randomised accuracy at the matched train-BCE levels; L_summary.paired).
  R : every lower end of the 95% percentile bootstrap interval of the median paired difference > 0 (all levels)
  N : every upper end < delta (all levels)
Local generators only (numpy default_rng(20261009)); 300 simulations, inner bootstrap 1,000 (2B's power method).
Usage (from census/): python results/designs/trackL_explore/L_power.py GROUP_PREFIX LEVELS DELTA FILE.jsonl ...
  e.g. "p=0.8 lr=0.01 w=256 sgd" 0.6,0.3,0.03 0.02 explore_gate.jsonl explore_lever.jsonl"""
from __future__ import annotations

import json
import sys

import numpy as np

import L_summary as S

NSIM, NBOOT = 300, 1000


def lower_upper(d, rng, nboot=NBOOT):
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    med = np.median(d[idx], axis=1)
    return np.quantile(med, 0.025), np.quantile(med, 0.975)


def power(diffs_by_level, n, crit, delta, rng):
    ok = 0
    for _ in range(NSIM):
        sel = rng.integers(0, len(diffs_by_level[0]), size=n)  # seeds resampled jointly across levels
        good = True
        for d in diffs_by_level:
            lo, hi = lower_upper(np.asarray(d)[sel], rng)
            if (crit == "R" and not lo > 0) or (crit == "N" and not hi < delta):
                good = False
                break
        ok += good
    return ok / NSIM


if __name__ == "__main__":
    group, levels, delta = sys.argv[1], [float(x) for x in sys.argv[2].split(",")], float(sys.argv[3])
    rows = S.load(sys.argv[4:])
    pr = S.paired(rows, levels, qty=("rand",))
    rng = np.random.default_rng(20261009)
    out = {}
    for key, cell in pr.items():
        if not key.startswith(group):
            continue
        diffs = [cell[f"{l}|rand"]["vals"] for l in levels]
        if len({len(d) for d in diffs}) != 1:
            print(key, "unequal reach", [len(d) for d in diffs])
            continue
        res = {"n_explore": len(diffs[0]), "median": [cell[f"{l}|rand"]["median"] for l in levels],
               "sd": [round(float(np.std(d, ddof=1)), 4) for d in diffs],
               "cost": [cell[f"{l}|cost"] for l in levels]}
        for n in (20, 40, 80):
            res[f"P(R) n={n}"] = power(diffs, n, "R", delta, rng)
            res[f"P(N) n={n}"] = power(diffs, n, "N", delta, rng)
        out[key] = res
        print(key, json.dumps(res))
