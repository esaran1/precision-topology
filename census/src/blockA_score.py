"""Block A: score the registered across-a discriminator.

O1 if CV(R) <= 0.15 and CV(R) < CV(|w2|)/2 across per-cell medians
O2 if the mirror holds; O3 otherwise.  Cluster bootstrap over cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def cv(v) -> float:
    v = np.asarray(v, float)
    return float(v.std(ddof=0) / v.mean()) if len(v) > 1 and v.mean() != 0 else float("nan")


def boot_cv(per_cell, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(per_cell, float)
    out = [cv(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return np.nanpercentile(out, [2.5, 97.5])


def main() -> None:
    d = pd.read_csv(RESULTS / "phase2b_checkpoints.csv")
    cr = d[d.crossing]
    print("=== Block A: crossing R and |w2| per cell ===\n")
    print(f"{'a':>6} {'B':>7} {'n':>4} {'med R':>8} {'CV(R) in cell':>14} "
          f"{'med |w2|':>9} {'CV in cell':>11}")
    rows = []
    for (a, B), g in cr.groupby(["a", "budget"]):
        w = g.w2.abs()
        rows.append({"a": a, "budget": B, "n": len(g),
                     "R": g.R.median(), "w2": w.median(),
                     "cvR": cv(g.R.values), "cvW": cv(w.values)})
        print(f"{a:>6} {B:>7} {len(g):>4} {g.R.median():>8.4f} {cv(g.R.values):>14.4f} "
              f"{w.median():>9.3f} {cv(w.values):>11.4f}")
    r = pd.DataFrame(rows)

    # per-a medians (pooling budgets), which is the registered unit
    per_a = r.groupby("a").agg(R=("R", "median"), w2=("w2", "median")).reset_index()
    print("\n=== per-a medians (the registered unit) ===")
    print(per_a.to_string(index=False))
    cR, cW = cv(per_a.R.values), cv(per_a.w2.values)
    loR, hiR = boot_cv(per_a.R.values)
    loW, hiW = boot_cv(per_a.w2.values)
    print(f"\n  CV(R)    = {cR:.4f}  95% CI [{loR:.4f}, {hiR:.4f}]")
    print(f"  CV(|w2|) = {cW:.4f}  95% CI [{loW:.4f}, {hiW:.4f}]")
    o1 = cR <= 0.15 and cR < cW / 2
    o2 = cW <= 0.15 and cW < cR / 2
    verdict = "O1 (geometry carries the threshold)" if o1 else \
              "O2 (|w2| is the quantity)" if o2 else "O3 (neither)"
    print(f"\n  registered O1: CV(R)<=0.15 and CV(R)<CV(|w2|)/2 -> {o1}")
    print(f"  registered O2: mirror -> {o2}")
    print(f"  VERDICT: {verdict}")
    r.to_csv(RESULTS / "blockA_crossings.csv", index=False)
    per_a.to_csv(RESULTS / "blockA_per_a.csv", index=False)


if __name__ == "__main__":
    main()
