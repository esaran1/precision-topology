"""2a option 3: recompute onsets with separation defined at fixed margin m>0.

If the budget law and the four-family relationship survive under a
margin-based criterion, the theorem (stated at fixed m) and the experiments
can be connected by redefining the empirical criterion.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
from src.artifact_lock import artifact_lock
from src.fold1d import activation, logits, make_data, INNER_MAX, OUTER_MIN, OUTER_MAX

RESULTS = Path(__file__).resolve().parents[1] / "results"
INNER = torch.linspace(-INNER_MAX, INNER_MAX, 4001, dtype=torch.float64)
_P = torch.linspace(OUTER_MIN, OUTER_MAX, 2001, dtype=torch.float64)
OUTER = torch.cat([_P, -_P])
SEEDS = 40


def margin_of(theta, f) -> float:
    with torch.no_grad():
        li = logits(theta.double(), INNER, f); lo = logits(theta.double(), OUTER, f)
        return min(float(-li.max()), float(lo.min()))


def solves_at_margin(theta, f, m: float) -> bool:
    return margin_of(theta, f) >= m


def rate(a: float, budget: int, m: float) -> float:
    f = activation("sin_family", a); n = 0
    for s in range(SEEDS):
        x, y = make_data(200, s); torch.manual_seed(s)
        th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=1e-2)
        for _ in range(budget):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward(); opt.step()
        n += solves_at_margin(th.detach(), f, m)
    return n / SEEDS


def onset(budget: int, m: float, grid) -> tuple:
    found, brk = None, False
    for a in grid:
        r = rate(a, budget, m)
        print(f"    m={m} B={budget} a={a}: rate={r:.3f}", flush=True)
        if r >= 0.5: found = a
        elif found is not None: brk = True; break
    return found, brk


def main() -> None:
    grids = {2_000:[1.70,1.60,1.55,1.50,1.45,1.40,1.35],
             8_000:[1.40,1.30,1.25,1.20,1.18,1.16,1.14],
             32_000:[1.20,1.15,1.10,1.08,1.06,1.05,1.04]}
    rows = []
    for m in (0.0, 0.01, 0.05):
        for B, g in grids.items():
            o, brk = onset(B, m, g)
            rows.append({"margin": m, "budget": B, "onset": o, "bracketed": brk})
            print(f"  m={m} B={B}: onset={o} bracketed={brk}", flush=True)
        sub = pd.DataFrame(rows); sub = sub[(sub.margin == m) & sub.bracketed & sub.onset.notna()]
        if len(sub) >= 2:
            s = np.polyfit(np.log(sub.budget.values), np.log(sub.onset.values - 1.0), 1)[0]
            print(f"*** margin m={m}: budget-law exponent = {s:.4f} "
                  f"(sign-correctness value: -0.7340)", flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "margin_onsets"
    with artifact_lock(stem, "margin onsets"):
        tmp = stem.with_suffix(".csv.tmp"); frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
