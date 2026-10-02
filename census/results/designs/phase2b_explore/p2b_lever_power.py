"""EXPLORATORY (NOT a registration): the proposed Phase 2B criteria (results/designs/phase2b_lever_design.md) applied to
the exploration seeds 2,953,000-2,953,019 (runs_lever/), and their power by resampling those seeds' paired differences.
Local generators only (np.random.default_rng); no global RNG state.

    python p2b_lever_power.py ARM2 ARM3        # e.g. out16b glob  ->  p2b_lever_power_ARM2_ARM3.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_lever"
SEEDS = tuple(range(2_953_000, 2_953_020))
S_STAR = 3.5913755424683727
POINTS = ("scale_1.25s*", "scale_2s*", "scale_3s*", "loss_0.1", "loss_0.03", "loss_0.01")
QTY = ("rho2", "acc_shuffled", "acc_reversed")
DELTA = {"rho2": 0.03, "acc_shuffled": 0.02, "acc_reversed": 0.03}      # margins for "no gain" (proposed)
ONSET_END, ONSET_MIN = "T40000", 0.75                                    # lasting to T_C; >= 75% of runs above s*
B_REG, SEED_REG = 10_000, 20261002                                       # registered-style bootstrap


def load(cond):
    return {s: json.loads((RUNS / f"{cond}_{s}.json").read_text()) for s in SEEDS}


def paired(std, arm):
    """(n_seeds, n_points * n_qty) paired differences arm - std (NaN where a point is not reached)."""
    D = np.full((len(SEEDS), len(POINTS) * len(QTY)), np.nan)
    for i, s in enumerate(SEEDS):
        for j, p in enumerate(POINTS):
            a, b = arm[s]["at"].get(p), std[s]["at"].get(p)
            if a and b:
                for k, q in enumerate(QTY):
                    D[i, j * len(QTY) + k] = a[q] - b[q]
    return D


def ci_median(D, B, rng):
    """95% percentile bootstrap interval of the median paired difference, per column (rows resampled jointly)."""
    idx = rng.integers(0, len(D), size=(B, len(D)))
    med = np.nanmedian(D[idx], axis=1)                 # (B, cols)
    return np.percentile(med, 2.5, axis=0), np.percentile(med, 97.5, axis=0)


def criteria(D2, D3, onset_ok, B, rng):
    lo2, _ = ci_median(D2, B, rng)
    _, hi3 = ci_median(D3, B, rng)
    marg = np.tile([DELTA[q] for q in QTY], len(POINTS))
    return {"R": bool((lo2 > 0).all()), "N": bool((hi3 < marg).all()), "O": bool(onset_ok.mean() >= ONSET_MIN),
            "lo2": lo2, "hi3": hi3}


def main(arm2, arm3):
    std, a2, a3 = load("std"), load(arm2), load(arm3)
    D2, D3 = paired(std, a2), paired(std, a3)
    on = np.array([bool(a2[s]["onset"].get(ONSET_END)) and a2[s]["onset"][ONSET_END]["s_over_s_star"] > 1 for s in SEEDS])
    rng = np.random.default_rng(SEED_REG)
    c = criteria(D2, D3, on, B_REG, rng)
    cols = [f"{p}|{q}" for p in POINTS for q in QTY]
    out = {"label": "EXPLORATORY (exploration seeds; never to be registered)", "arm2": arm2, "arm3": arm3,
           "on_exploration_seeds": {
               "R_pass": c["R"], "N_pass": c["N"], "O_pass": c["O"], "onset_frac_above_s*": float(on.mean()),
               "arm2_median_diff": dict(zip(cols, np.nanmedian(D2, 0).round(4).tolist())),
               "arm2_ci_low": dict(zip(cols, c["lo2"].round(4).tolist())),
               "arm2_n_up_down": {k: [int((D2[:, i] > 0).sum()), int((D2[:, i] < 0).sum())] for i, k in enumerate(cols)},
               "arm3_median_diff": dict(zip(cols, np.nanmedian(D3, 0).round(4).tolist())),
               "arm3_ci_high": dict(zip(cols, c["hi3"].round(4).tolist())),
               "arm3_n_up_down": {k: [int((D3[:, i] > 0).sum()), int((D3[:, i] < 0).sum())] for i, k in enumerate(cols)},
               "arm2_sd": dict(zip(cols, np.nanstd(D2, 0, ddof=1).round(4).tolist())),
               "arm3_sd": dict(zip(cols, np.nanstd(D3, 0, ddof=1).round(4).tolist()))}}
    # power: resample N seeds (rows jointly) from the 20 exploration paired differences; inner bootstrap B = 2,000
    sims, B = 300, 1000
    rng = np.random.default_rng(SEED_REG + 1)
    med2, med3 = np.nanmedian(D2, 0), np.nanmedian(D3, 0)
    marg = np.tile([DELTA[q] for q in QTY], len(POINTS))
    scen = {"as_explored": (D2, D3),
            "arm2_effect_halved_spread_doubled": (0.5 * med2 + 2 * (D2 - med2), D3),
            "arm3_spread_doubled": (D2, med3 + 2 * (D3 - med3)),
                        "arm3_shifted_to_margin (should fail ~>= 50%)": (D2, marg + (D3 - med3))}
    pw = {}
    for name, (A2, A3) in scen.items():
        pw[name] = {}
        for n in (30, 40, 60):
            hits = {"R": 0, "N": 0, "O": 0, "all": 0}
            for _ in range(sims):
                rows = rng.integers(0, len(SEEDS), n)
                cc = criteria(A2[rows], A3[rows], on[rows], B, rng)
                for k in ("R", "N", "O"):
                    hits[k] += cc[k]
                hits["all"] += cc["R"] and cc["N"] and cc["O"]
            pw[name][n] = {k: v / sims for k, v in hits.items()}
            print(name, n, pw[name][n], flush=True)
    # onset: exact binomial P(X >= ceil(0.75 n)) at the exploration rate and at its Wilson 95% lower bound
    from math import comb, ceil, sqrt
    k, n0 = int(on.sum()), len(on); z = 1.959964
    p_hat = k / n0
    wl = (p_hat + z * z / (2 * n0) - z * sqrt(p_hat * (1 - p_hat) / n0 + z * z / (4 * n0 * n0))) / (1 + z * z / n0)
    binom = lambda n, p: sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(ceil(ONSET_MIN * n), n + 1))  # noqa: E731
    out["onset_binomial"] = {"k_of_20": k, "wilson_low": wl,
                             **{f"P_pass_n{n}": {"at_p_hat": binom(n, p_hat), "at_wilson_low": binom(n, wl),
                                                 "at_0.85": binom(n, 0.85)} for n in (20, 30, 40, 60)}}
    out["power_resampled"] = pw
    out["power_settings"] = {"sims": sims, "inner_bootstrap": B, "rng": [SEED_REG, SEED_REG + 1], "margins": DELTA,
                             "onset_min": ONSET_MIN, "onset_end": ONSET_END, "points": POINTS}
    (HERE / f"p2b_lever_power_{arm2}_{arm3}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["on_exploration_seeds"], indent=1, default=float)); print(json.dumps(out["onset_binomial"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
