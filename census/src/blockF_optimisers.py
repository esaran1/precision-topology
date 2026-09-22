"""Block F: crossing R across three optimisers, to score the registered lag account.

Registered in results/blockF_lag_prediction.md (F-1, F-2). float64.

No existing artifact has per-optimiser CROSSING R: r_adamw.csv, alpha_composition.csv
and r_pooled.csv all record TERMINAL |w2|, not the placement-crossing step.  This
module instruments the crossing the way Block A did, for Adam, AdamW and SGD at the
registered a = 1.25, so the Adam arm reproduces a known quantity as a control.

Also logs the local d|w2|/dstep at the crossing, which F-2 needs and which the
registration fixed in advance (Adam 0.00207, AdamW 0.00184, SGD 0.00075).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import activation, make_data, solves
from .fold1d_theorem import maximum_gap
from .r_variable import oriented_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"
A = 1.25
N_SEEDS = 60
BUDGET = 20_000          # SGD is ~2.8x slower, so it needs headroom to cross
CHECK_EVERY = 25
SGD_LR = 0.3             # the only SGD lr that solves at all (recorded in sgd_onsets.LR)
ADAM_LR = 1e-2
RATE_WINDOW = 400        # steps each side of the crossing for the local rate


def make_opt(name, params):
    if name == "adam":
        return torch.optim.Adam(params, lr=ADAM_LR)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=ADAM_LR)
    if name == "sgd":
        return torch.optim.SGD(params, lr=SGD_LR)
    raise ValueError(name)


def run(a, seed, opt_name, gstar, budget=BUDGET):
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = make_opt(opt_name, [th])
    trace = []
    cross = None
    for i in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
        if i % CHECK_EVERY == 0:
            with torch.no_grad():
                w1, b1, w2, b2 = (float(v) for v in th)
            gp = oriented_gap(f, w1, b1)
            trace.append((i, abs(w2), gp))
            if cross is None and gp > 0:
                cross = {"step": i, "w2": abs(w2), "R": abs(w2) * gstar / 2}
    tr = np.array(trace)
    # local d|w2|/dstep in a window around the crossing
    rate = np.nan
    if cross is not None:
        m = (tr[:, 0] >= cross["step"] - RATE_WINDOW) & (tr[:, 0] <= cross["step"] + RATE_WINDOW)
        if m.sum() >= 3:
            rate = float(np.polyfit(tr[m, 0], tr[m, 1], 1)[0])
    with torch.no_grad():
        w1, b1, w2, b2 = (float(v) for v in th)
    return {"a": a, "seed": seed, "optimiser": opt_name,
            "cross_step": None if cross is None else cross["step"],
            "cross_w2": None if cross is None else cross["w2"],
            "cross_R": None if cross is None else cross["R"],
            "local_rate": rate,
            "final_placed": oriented_gap(f, w1, b1) > 0,
            "final_solved": bool(solves(torch.tensor([w1, b1, w2, b2],
                                                     dtype=torch.float64), f)),
            "R_final": abs(w2) * gstar / 2}


def main():
    gstar = maximum_gap(A, resolution=600)
    print(f"a={A}  Ghat={gstar:.6f}  budget={BUDGET}  seeds={N_SEEDS}", flush=True)
    rows = []
    for name in ("adam", "adamw", "sgd"):
        for s in range(N_SEEDS):
            rows.append(run(A, s, name, gstar))
        d = pd.DataFrame([r for r in rows if r["optimiser"] == name])
        c = d[d.cross_R.notna()]
        print(f"  {name:6s} crossed {len(c)}/{len(d)}  solved {d.final_solved.mean():.3f}  "
              f"median cross R {c.cross_R.median() if len(c) else float('nan'):.4f}  "
              f"median cross step {c.cross_step.median() if len(c) else float('nan'):.0f}  "
              f"median local rate {c.local_rate.median() if len(c) else float('nan'):.6f}",
              flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "blockF_optimisers"
    with artifact_lock(stem, "blockF optimisers"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
    print("\nwritten results/blockF_optimisers.csv")


if __name__ == "__main__":
    main()
