"""2c: onsets vs budget for constructed families with different beta.

Extreme pair (q=4, beta=1.25 and q=2/3, beta=2.5) runs first: they carry the
ordering test.  Same bracketing criteria as all prior onset work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .depth_families import make_family
from .fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX, make_data

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = 40
INNER = torch.linspace(-INNER_MAX, INNER_MAX, 2001)
_POS = torch.linspace(OUTER_MIN, OUTER_MAX, 1001)
OUTER = torch.cat([_POS, -_POS])


def solves_family(theta, f, a) -> bool:
    with torch.no_grad():
        w1, b1, w2, b2 = theta
        li = w2 * f(w1 * INNER + b1, a) + b2
        lo = w2 * f(w1 * OUTER + b1, a) + b2
        return bool((li < 0).all() and (lo > 0).all())


def run(q: float, a: float, seed: int, budget: int, lr: float = 1e-2) -> bool:
    f, _ = make_family(q)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        w1, b1, w2, b2 = theta[0], theta[1], theta[2], theta[3]
        out = w2 * f(w1 * x + b1, a) + b2
        F.binary_cross_entropy_with_logits(out, y).backward()
        opt.step()
    return solves_family(theta.detach(), f, a)


def rate(q, a, budget):
    return sum(run(q, a, s, budget) for s in range(SEEDS)) / SEEDS


def onset(q: float, budget: int, grid) -> tuple:
    found, bracketed, curve = None, False, []
    for eps in grid:
        r = rate(q, 1.0 + eps, budget)
        curve.append({"q": q, "budget": budget, "eps": eps, "rate": r})
        print(f"  q={q:.3f} B={budget} eps={eps:.5f}: rate={r:.3f}", flush=True)
        if r >= 0.5:
            found = eps
        elif found is not None:
            bracketed = True
            break
    return found, bracketed, curve


def main() -> None:
    order = [(4.0, "q4_beta1.25"), (2.0/3.0, "q0.667_beta2.5"),
             (2.0, "q2_beta1.5"), (1.0, "q1_beta2")]
    if len(sys.argv) > 1:
        order = [o for o in order if o[1] in sys.argv[1:]]
    budgets = (2_000, 8_000, 32_000, 128_000)
    grids = {
        4.0:      [0.60, 0.40, 0.25, 0.15, 0.10, 0.06, 0.04, 0.025, 0.015, 0.01, 0.006],
        2.0/3.0:  [0.60, 0.40, 0.25, 0.15, 0.10, 0.06, 0.04, 0.025, 0.015, 0.01, 0.006],
        2.0:      [0.60, 0.40, 0.25, 0.15, 0.10, 0.06, 0.04, 0.025, 0.015, 0.01, 0.006],
        1.0:      [0.60, 0.40, 0.25, 0.15, 0.10, 0.06, 0.04, 0.025, 0.015, 0.01, 0.006],
    }
    rows, curves = [], []
    for q, name in order:
        beta = 1.0 + 1.0 / q
        for B in budgets:
            o, br, curve = onset(q, B, grids[q])
            rows.append({"family": name, "q": q, "beta": beta, "budget": B,
                         "onset_eps": o, "bracketed": br})
            curves.extend(curve)
            print(f"q={q:.3f} B={B}: onset_eps={o} bracketed={br}", flush=True)
            # APPEND to any existing artifact rather than overwriting: a second
            # invocation for different families previously clobbered the first
            # run's rows and silently dropped two families from the figure.
            for nm, fr in (("family_onsets", pd.DataFrame(rows)),
                           ("family_onset_curves", pd.DataFrame(curves))):
                stem = RESULTS / nm
                with artifact_lock(stem, nm):
                    path = stem.with_suffix(".csv")
                    if path.exists():
                        prior = pd.read_csv(path)
                        key = (["family", "budget"] if nm == "family_onsets"
                               else ["q", "budget", "eps"])
                        fr = (pd.concat([prior, fr])
                                .drop_duplicates(subset=key, keep="last"))
                    tmp = stem.with_suffix(".csv.tmp"); fr.to_csv(tmp, index=False)
                    tmp.replace(path)
        sub = pd.DataFrame(rows)
        sub = sub[(sub.family == name) & sub.bracketed & sub.onset_eps.notna()]
        if len(sub) >= 2:
            s = np.polyfit(np.log(sub.budget.values), np.log(sub.onset_eps.values), 1)[0]
            print(f"*** {name}: beta={beta:.3f} measured exponent={s:.4f} "
                  f"predicted={-1.1172/beta:.4f} n={len(sub)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
