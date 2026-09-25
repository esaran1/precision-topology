"""Track 4: the conditional threshold under SGD (registration: results/sgd_own_registration.md).

The own-sample global thresholds for the phase-2b training sets (fold1d.make_data(200, seed), seeds 0-39, a = 1.30 and
1.50; results/own_threshold_crossing.csv) are properties of the loss and the data, not of the optimiser.  Here SGD
(lr 0.3, full batch, the only rate that solves; blockF_optimisers.SGD_LR) is trained on exactly those sets, from the
same initialisation as the Adam runs (phase2b_ordering.run), with every-step crossing detection (phase2b_ordering.state).

    python -m src.sgd_own pilot        # calibration seeds 700,000-700,019: |w2| growth only, no crossings
    python -m src.sgd_own train        # registered seeds 0-39 at both a
    python -m src.sgd_own score
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.30, 1.50)
SGD_LR = 0.3
CAL_SEEDS = tuple(range(700_000, 700_020))
SEEDS = tuple(range(40))
BUDGETS = (16_000, 32_000, 64_000, 128_000)
REACH = 1.5            # pilot rule: |w2| >= REACH x population |w2|_glob
REACH_FRAC = 0.95
MIN_CROSS = 30
N_BOOT = 10_000


def _setup(a, seed):
    import torch
    from .fold1d import N_PER_CLASS, activation, make_data
    torch.set_num_threads(1)
    f = activation("sin_family", a)
    x, y = make_data(N_PER_CLASS, seed)
    torch.manual_seed(seed)
    init = torch.empty(4).uniform_(-1.0, 1.0)                 # as phase2b_ordering.run: drawn once, then cast
    theta = init.double().clone().requires_grad_(True)
    return f, x.double(), y.double(), theta


def pilot_run(a, seed, budget=max(BUDGETS)):
    """|w2| at each candidate budget; no placement is evaluated."""
    import torch
    from torch.nn import functional as F
    from .fold1d import logits
    f, x, y, th = _setup(a, seed)
    opt = torch.optim.SGD([th], lr=SGD_LR)
    out = {}
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        opt.step()
        if step in BUDGETS:
            out[step] = abs(float(th[2]))
    return out


def choose_budget(w2_at, w2_pop):
    """Smallest budget at which >= 95% of calibration runs have |w2| >= 1.5 x w2_pop at every a; else the largest."""
    for B in BUDGETS:
        if all(np.mean([r[B] >= REACH * w2_pop[a] for r in w2_at[a]]) >= REACH_FRAC for a in w2_at):
            return B
    return BUDGETS[-1]


def train_run(a, seed, budget, gstar):
    """Every-step crossing detection with phase2b_ordering.state (dense 4,001-point windows, the Adam test's rule)."""
    import torch
    from torch.nn import functional as F
    from .fold1d import logits
    from .phase2b_ordering import state
    f, x, y, th = _setup(a, seed)
    opt = torch.optim.SGD([th], lr=SGD_LR)
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        opt.step()
        s = state(th.detach(), f, a, gstar)
        if s["placement_ok"]:
            return {"a": a, "seed": seed, "crossed": True, "step": step, "w2_cross": abs(s["w2"])}
    return {"a": a, "seed": seed, "crossed": False, "step": np.nan, "w2_cross": np.nan}


def _pilot_job(args):
    os.nice(15)
    a, seed = args
    return {"a": a, "seed": seed, **{f"w2_{B}": v for B, v in pilot_run(a, seed).items()}}


def _train_job(args):
    os.nice(15)
    a, seed, budget, gstar = args
    r = train_run(a, seed, budget, gstar)
    print(json.dumps({"a": a, "seed": seed, "done": True}), flush=True)
    return r


def _pop():
    d = pd.read_csv(RESULTS / "own_threshold_crossing.csv")
    return {round(a, 2): float(g.w2_pop.iloc[0]) for a, g in d.groupby("a")}


def pilot(workers=1):
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_pilot_job, [(a, s) for a in A_VALUES for s in CAL_SEEDS])
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "sgd_own_pilot.csv", index=False)
    w2_at = {a: [{B: r[f"w2_{B}"] for B in BUDGETS} for r in d[d.a == a].to_dict("records")] for a in A_VALUES}
    B = choose_budget(w2_at, _pop())
    (RESULTS / "sgd_own_budget.json").write_text(json.dumps({"budget": B, "rule": f">= {REACH_FRAC:.0%} of calibration "
                                                             f"runs at |w2| >= {REACH} x w2_pop at every a"}))
    print(d.groupby("a")[[f"w2_{b}" for b in BUDGETS]].median().to_string(), "\nbudget", B)


def train(workers=1):
    from multiprocessing import get_context
    from .fold1d_theorem import maximum_gap
    B = json.loads((RESULTS / "sgd_own_budget.json").read_text())["budget"]
    out = RESULTS / "sgd_own_runs.csv"
    done = set() if not out.exists() else {(round(r.a, 2), r.seed) for r in pd.read_csv(out).itertuples()}
    gs = {a: maximum_gap(a, resolution=600) for a in A_VALUES}          # as phase2b_ordering.main
    todo = [(a, s, B, gs[a]) for a in A_VALUES for s in SEEDS if (a, s) not in done]
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_train_job, todo):
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)


def paired_boot(d, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = rng.choice(d, (n, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def score_a(w2_cross, w2_own, w2_pop):
    """G1: mean(|log(cross/own)| − |log(cross/pop)|), paired run-level bootstrap 95% interval entirely below 0.
    G2: Spearman(cross, own) >= 0.6.  UNRESOLVED if fewer than MIN_CROSS crossings."""
    w2_cross, w2_own = np.asarray(w2_cross, float), np.asarray(w2_own, float)
    if len(w2_cross) < MIN_CROSS:
        return {"G1": "UNRESOLVED", "G2": "UNRESOLVED", "n": len(w2_cross)}
    diff = np.abs(np.log(w2_cross / w2_own)) - np.abs(np.log(w2_cross / w2_pop))
    m, lo, hi = paired_boot(diff)
    rho = spearman(w2_cross, w2_own)
    return {"n": len(w2_cross), "G1_mean_diff": m, "G1_lo": lo, "G1_hi": hi, "G1": "PASS" if hi < 0 else "FAIL",
            "G2_spearman": rho, "G2": "PASS" if rho >= 0.6 else "FAIL",
            "G3_median_resid_own": float(np.median(w2_cross / w2_own) - 1),
            "G3_median_resid_pop": float(np.median(w2_cross / w2_pop) - 1)}


def score():
    r = pd.read_csv(RESULTS / "sgd_own_runs.csv")
    own = pd.read_csv(RESULTS / "own_threshold_crossing.csv")
    own["a"] = own.a.round(2); r["a"] = r.a.round(2)
    m = r.merge(own[["a", "seed", "w2_own", "w2_pop"]], on=["a", "seed"])
    rows = []
    for a in A_VALUES:
        g = m[(m.a == a) & m.crossed]
        rows.append({"a": a, "runs": int((m.a == a).sum()), "crossed": len(g),
                     **score_a(g.w2_cross, g.w2_own, g.w2_pop.iloc[0] if len(g) else np.nan)})
    s = pd.DataFrame(rows)
    s.to_csv(RESULTS / "sgd_own_scores.csv", index=False)
    print(s.to_string(index=False))
    return s


if __name__ == "__main__":
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"pilot": lambda: pilot(w), "train": lambda: train(w), "score": score}[sys.argv[1]]()
