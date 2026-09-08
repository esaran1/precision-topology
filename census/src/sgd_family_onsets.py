"""Part 1a: family onsets under plain SGD, testing whether the GEOMETRIC part
of the law (exponent ~ 1/beta) transfers across optimizers.

alpha cancels in the ratio e(q4)/e(q0.667), predicted 2.0.
q0.667 uses a refined 1.25x eps grid (registered) since its resolution
dominates the ratio's uncertainty.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
from src.artifact_lock import artifact_lock
from src.depth_families import make_family
from src.fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX, make_data

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = 40
LR = 0.3
INNER = torch.linspace(-INNER_MAX, INNER_MAX, 2001)
_P = torch.linspace(OUTER_MIN, OUTER_MAX, 1001)
OUTER = torch.cat([_P, -_P])


def solves_family(theta, f, a) -> bool:
    with torch.no_grad():
        w1, b1, w2, b2 = theta
        li = w2 * f(w1 * INNER + b1, a) + b2
        lo = w2 * f(w1 * OUTER + b1, a) + b2
        return bool((li < 0).all() and (lo > 0).all())


def rate(q: float, eps: float, B: int) -> float:
    f, _ = make_family(q); a = 1.0 + eps; n = 0
    for s in range(SEEDS):
        x, y = make_data(200, s); torch.manual_seed(s)
        th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
        opt = torch.optim.SGD([th], lr=LR)
        for _ in range(B):
            opt.zero_grad(set_to_none=True)
            w1, b1, w2, b2 = th[0], th[1], th[2], th[3]
            F.binary_cross_entropy_with_logits(w2 * f(w1 * x + b1, a) + b2, y).backward()
            opt.step()
        n += solves_family(th.detach(), f, a)
    return n / SEEDS


def grid_for(q):
    if abs(q - 2/3) < 1e-6:                       # refined 1.25x grid
        return [round(0.6 / 1.25**k, 5) for k in range(13)]
    return [0.60, 0.40, 0.25, 0.15, 0.10, 0.06, 0.04, 0.025, 0.015]


def main() -> None:
    rows, curves = [], []
    # low budgets first so a partial result is interpretable
    for B in (2_000, 8_000, 32_000, 128_000):
        for q, name in ((4.0, "q4_beta1.25"), (2/3, "q0.667_beta2.5")):
            onset, brk = None, False
            for eps in grid_for(q):
                r = rate(q, eps, B)
                curves.append({"family": name, "budget": B, "eps": eps, "rate": r})
                print(f"  SGD {name} B={B} eps={eps}: rate={r:.3f}", flush=True)
                if r >= 0.5:
                    onset = eps
                elif onset is not None:
                    brk = True; break
            rows.append({"family": name, "q": q, "beta": 1 + 1/q, "budget": B,
                         "onset_eps": onset, "bracketed": brk})
            print(f"SGD {name} B={B}: onset={onset} bracketed={brk}", flush=True)
            for nm, fr in (("sgd_family_onsets", pd.DataFrame(rows)),
                           ("sgd_family_curves", pd.DataFrame(curves))):
                stem = RESULTS / nm
                with artifact_lock(stem, nm):
                    tmp = stem.with_suffix(".csv.tmp"); fr.to_csv(tmp, index=False)
                    tmp.replace(stem.with_suffix(".csv"))
        # report the ratio as soon as both families have >=3 bracketed cells
        d = pd.DataFrame(rows); d = d[d.bracketed & d.onset_eps.notna()]
        exps = {}
        for nm, sub in d.groupby("family"):
            if len(sub) >= 3:
                exps[nm] = np.polyfit(np.log(sub.budget.values),
                                      np.log(sub.onset_eps.values), 1)[0]
        if len(exps) == 2:
            r = exps["q4_beta1.25"] / exps["q0.667_beta2.5"]
            print(f"*** INTERIM RATIO after B={B}: {r:.4f} "
                  f"(predicted 2.0; Adam measured 1.661; null 1.0)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
