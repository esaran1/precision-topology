"""Part 3: does the intersection of two independently-measured regions predict findability?

Region A (correctly-classifying, derived): from fold1d_theorem.md, a network
with margin m needs |w2| >= 2m/(kappa*D(a)) = 2m/G*(a).  Inverting: at a given
|w2| the best achievable margin is m(|w2|) = |w2|*G*(a)/2.

Region B (terminal, measured): the distribution of |w2| where Adam stops.

Overlap prediction: a run is correct iff it terminates at |w2| whose available
margin is large enough that the loss gradient does not push it away -- i.e.
iff terminal |w2| >= w2_min(a) for the margin training actually settles at.

The test: does the fraction of terminal |w2| above the threshold predict the
measured rate, across a, learning rate, and initialization scale?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .fold1d_theorem import maximum_gap
from .termination import run

RESULTS = Path(__file__).resolve().parents[1] / "results"


def terminal_w2(a: float, lr: float, scale: float, seeds: int, budget: int = 2_000):
    """Terminal |w2| distribution and observed solve rate."""
    from .fold1d import activation, logits, make_data, solves
    from torch.nn import functional as F
    f = activation("sin_family", a)
    w2s, solved = [], 0
    for s in range(seeds):
        x, y = make_data(200, s)
        torch.manual_seed(s)
        theta = (torch.empty(4).uniform_(-1.0, 1.0) * scale).requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(budget):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
            opt.step()
        t = theta.detach()
        w2s.append(abs(float(t[2])))
        solved += solves(t, f)
    return np.array(w2s), solved / seeds


def main() -> None:
    rows = []
    conditions = [("a_sweep", a, 1e-2, 1.0) for a in
                  (1.02, 1.10, 1.25, 1.35, 1.45, 1.50, 2.00, 3.00)]
    conditions += [("lr_sweep", a, lr, 1.0) for a in (1.25, 1.45)
                   for lr in (3e-2, 3e-3, 1e-3)]
    conditions += [("init_scale", a, 1e-2, sc) for a in (1.25, 1.50)
                   for sc in (0.3, 3.0)]

    for kind, a, lr, scale in conditions:
        w2s, rate = terminal_w2(a, lr, scale, 40)
        gstar = maximum_gap(a)
        # empirical margin threshold: the smallest |w2| among runs that solved,
        # is a measured quantity; the DERIVED threshold uses the theorem with
        # the margin training typically settles at.
        rows.append({"kind": kind, "a": a, "lr": lr, "init_scale": scale,
                     "rate": rate, "G_star": gstar,
                     "w2_med": float(np.median(w2s)),
                     "w2_p10": float(np.percentile(w2s, 10)),
                     "w2_p90": float(np.percentile(w2s, 90)),
                     "w2_all": ";".join(f"{v:.4f}" for v in w2s)})
        print(f"{kind} a={a} lr={lr:g} scale={scale}: rate={rate:.3f} "
              f"w2_med={np.median(w2s):.2f}", flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "overlap"
    with artifact_lock(stem, "overlap"):
        tmp = stem.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
