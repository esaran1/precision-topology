"""1a verification: bisect on BUDGET at fixed a, the other axis.

Registered: log B vs log(a-1) should have slope -1.3624, band
[-1.6698, -1.1506]. This is a different measurement from the onset sweep
(bisection on budget, not on a), so agreement is a consistency check.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
from src.artifact_lock import artifact_lock
from src.fold1d import activation, logits, make_data, solves

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = 30


def rate(a: float, B: int) -> float:
    f = activation("sin_family", a); n = 0
    for s in range(SEEDS):
        x, y = make_data(200, s); torch.manual_seed(s)
        th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=1e-2)
        for _ in range(B):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
            opt.step()
        n += solves(th.detach(), f)
    return n / SEEDS


def budget_for(a: float, ladder) -> tuple:
    """Smallest budget on the ladder reaching >=50%, with bracketing."""
    prev = None
    for B in ladder:
        r = rate(a, B)
        print(f"  a={a} B={B}: rate={r:.3f}", flush=True)
        if r >= 0.5:
            return B, prev is not None, r
        prev = B
    return None, False, None


def main() -> None:
    ladder = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
    rows = []
    for a in (1.50, 1.35, 1.25, 1.15, 1.10):
        B, brk, r = budget_for(a, ladder)
        rows.append({"a": a, "eps": a - 1, "budget_50pct": B, "bracketed": brk})
        print(f"a={a}: B_50% = {B} bracketed={brk}", flush=True)
        stem = RESULTS / "cost_law"
        with artifact_lock(stem, "cost law"):
            tmp = stem.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(stem.with_suffix(".csv"))
    d = pd.DataFrame(rows); d = d[d.bracketed & d.budget_50pct.notna()]
    print(f"\nBRACKETED CELLS: {len(d)}")
    if len(d) >= 2:
        s = np.polyfit(np.log(d.eps.values), np.log(d.budget_50pct.values.astype(float)), 1)[0]
        print(f"COST EXPONENT (log B vs log eps) = {s:.4f}")
        print(f"REGISTERED: -1.3624, band [-1.6698, -1.1506]")
        print(f"VERDICT: {'WITHIN BAND' if -1.6698 <= s <= -1.1506 else 'OUTSIDE BAND'}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
