"""Block C: optimizer equivalence of the R50 threshold (TOST).

Registered in results/blockC_equivalence_prediction.md before any run.

  python -m src.blockC_equivalence pilot       # placement pilot, 8 seeds per budget
  python -m src.blockC_equivalence main        # targeted cells until >=500 in-band
  python -m src.blockC_equivalence analyze     # MLE R50, cell bootstrap, verdicts
"""

from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
PILOT = RESULTS / "blockC_pilot.csv"
MAIN = RESULTS / "blockC_runs.csv"
VERDICTS = RESULTS / "blockC_verdicts.csv"
A = 1.25
DELTA = 0.024
BAND = (0.30, 0.50)
TARGET_IN_BAND = 500
BLOCK = 60
MAX_CELLS = 12
N_BOOT = 4000
PILOT_SEEDS = range(900, 908)
CANDIDATES = {"adam": list(range(3_000, 12_001, 1_000)),
              "adamw": list(range(3_000, 12_001, 1_000)),
              "sgd": list(range(8_000, 48_001, 4_000))}
LRS = {"adam": 1e-2, "adamw": 1e-2, "sgd": 0.3}
WORKERS = int(os.environ.get("C_WORKERS", "1"))
FIELDS = ["optimizer", "budget", "seed", "w2", "R", "solved"]


def ghat():
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv").set_index("a")
    return float(g.loc[A, "Ghat_certified"])


def train(args):
    opt_name, budget, seed = args
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data, solves

    torch.set_num_threads(1)
    f = activation("sin_family", A)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    if opt_name == "adam":
        opt = torch.optim.Adam([theta], lr=LRS["adam"])
    elif opt_name == "adamw":
        opt = torch.optim.AdamW([theta], lr=LRS["adamw"], weight_decay=0.01)
    else:
        opt = torch.optim.SGD([theta], lr=LRS["sgd"])
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
        opt.step()
    t = theta.detach()
    w2 = abs(float(t[2]))
    return {"optimizer": opt_name, "budget": budget, "seed": seed, "w2": w2,
            "R": w2 * GHAT / 2, "solved": bool(solves(t, f))}


GHAT = ghat()


def _append(path, jobs):
    done = set()
    if path.exists():
        d = pd.read_csv(path)
        done = set(zip(d.optimizer, d.budget, d.seed))
    jobs = [j for j in jobs if j not in done]
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with Pool(WORKERS) as pool:
            for r in pool.imap_unordered(train, jobs, chunksize=4):
                w.writerow(r)
                f.flush()


def pilot():
    _append(PILOT, [(o, b, s) for o, bs in CANDIDATES.items() for b in bs
                    for s in PILOT_SEEDS])
    d = pd.read_csv(PILOT)
    print(d.groupby(["optimizer", "budget"]).R.median().unstack(0).round(3))


def selected_budgets():
    """Registered rule, step 2: pilot budgets whose median R lies in the band;
    if fewer than 6, add midpoints between qualifying and adjacent budgets until
    6 do.  Applied deterministically: halve the budget spacing, keep budgets whose
    pilot median R (linearly interpolated between piloted budgets) lies in the
    band, and repeat until at least 6 qualify.  Recorded in blockC_rule_application.md."""
    d = pd.read_csv(PILOT)
    med = d.groupby(["optimizer", "budget"]).R.median()
    out = {}
    for o in CANDIDATES:
        m = med[o].sort_index()
        bs = np.array(m.index, float)
        rs = m.values
        step = float(np.diff(bs).min())
        grid = bs
        while True:
            interp = np.interp(grid, bs, rs)
            sel = [int(b) for b, r in zip(grid, interp) if BAND[0] <= r <= BAND[1]]
            if len(sel) >= 6 or step < 100:
                break
            step /= 2
            grid = np.arange(bs.min(), bs.max() + 1e-9, step)
        if len(sel) > MAX_CELLS:
            idx = np.linspace(0, len(sel) - 1, MAX_CELLS).round().astype(int)
            sel = [sel[k] for k in idx]
        out[o] = sel
    return out


def main_runs():
    sel = selected_budgets()
    for o, bs in sel.items():
        print(f"  {o}: cells at budgets {bs}", flush=True)
    start = 100
    while True:
        jobs = [(o, b, s) for o, bs in sel.items() for b in bs
                for s in range(start, start + BLOCK)]
        _append(MAIN, jobs)
        d = pd.read_csv(MAIN)
        inb = d[(d.R >= BAND[0]) & (d.R <= BAND[1])].groupby("optimizer").size()
        print(f"  after seeds {start}-{start + BLOCK - 1}: in-band "
              + ", ".join(f"{o}={int(inb.get(o, 0))}" for o in sel), flush=True)
        todo = {o: bs for o, bs in sel.items() if inb.get(o, 0) < TARGET_IN_BAND}
        if not todo:
            break
        sel = todo
        start += BLOCK
        if start > 100 + 20 * BLOCK:
            print("  stopping: 20 blocks without reaching the target", flush=True)
            break


def mle_r50(R, y):
    X = np.c_[np.ones_like(R), R]
    b = np.array([-10.0, 30.0])
    for _ in range(200):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -40, 40)))
        W = p * (1 - p) + 1e-12
        step = np.linalg.solve((X * W[:, None]).T @ X + 1e-9 * np.eye(2), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-12:
            break
    return -b[0] / b[1]


def analyze(path=MAIN, label="new"):
    d = pd.read_csv(path)
    d["solved"] = d.solved.astype(bool)
    rng = np.random.default_rng(0)
    opts = sorted(d.optimizer.unique())
    cells = {o: [g for _, g in d[d.optimizer == o].groupby("budget")] for o in opts}
    point = {o: mle_r50(d[d.optimizer == o].R.values,
                        d[d.optimizer == o].solved.values.astype(float)) for o in opts}
    boots = {o: [] for o in opts}
    for _ in range(N_BOOT):
        for o in opts:
            cs = cells[o]
            s = pd.concat([cs[i] for i in rng.integers(0, len(cs), len(cs))])
            boots[o].append(mle_r50(s.R.values, s.solved.values.astype(float))
                            if s.solved.nunique() == 2 else np.nan)
    rows = []
    for i, o1 in enumerate(opts):
        for o2 in opts[i + 1:]:
            diff = np.array(boots[o1]) - np.array(boots[o2])
            diff = diff[np.isfinite(diff)]
            lo90, hi90 = np.percentile(diff, [5, 95])
            lo95, hi95 = np.percentile(diff, [2.5, 97.5])
            if -DELTA < lo90 and hi90 < DELTA:
                verdict = "equivalent"
            elif hi90 < -DELTA or lo90 > DELTA:
                verdict = "not equivalent"
            else:
                verdict = "inconclusive"
            rows.append({"data": label, "pair": f"{o1} - {o2}",
                         "diff": point[o1] - point[o2], "ci90_lo": lo90, "ci90_hi": hi90,
                         "ci95_lo": lo95, "ci95_hi": hi95, "delta": DELTA,
                         "verdict": verdict})
    inb = d[(d.R >= BAND[0]) & (d.R <= BAND[1])].groupby("optimizer").size()
    for o in opts:
        print(f"  {o:6s} n={int((d.optimizer == o).sum())} in-band={int(inb.get(o, 0))} "
              f"cells={len(cells[o])} R50={point[o]:.4f}")
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out, point


def _cli_analyze():
    new, _ = analyze()
    new.to_csv(VERDICTS, index=False)
    print(f"written {VERDICTS.name}")


def analyze_existing():
    """Context only (registered): the untargeted 600-run comparison, same estimator and test."""
    a = pd.read_csv(RESULTS / "alpha_composition.csv")
    w = pd.read_csv(RESULTS / "r_adamw.csv")
    d = pd.concat([a[["optimizer", "budget", "seed", "w2", "solved"]],
                   w[["optimizer", "budget", "seed", "w2", "solved"]]])
    d["R"] = d.w2 * GHAT / 2
    d["solved"] = d.solved.astype(str).str.lower() == "true"
    tmp = RESULTS / "_blockC_existing_input.csv"
    d.to_csv(tmp, index=False)
    out, _ = analyze(tmp, label="existing (untargeted, context only)")
    tmp.unlink()
    out.to_csv(RESULTS / "blockC_existing_context.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":
        pilot()
    elif cmd == "main":
        main_runs()
    elif cmd == "existing":
        analyze_existing()
    elif cmd == "analyze":
        _cli_analyze()
