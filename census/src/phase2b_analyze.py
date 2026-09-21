"""Score the ordering test against the registered predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def growth_ratio(g: pd.DataFrame, cross: int, window: int = 200):
    """d|w2|/dstep in matched windows before and after the crossing."""

    g = g.sort_values("step")
    pre = g[(g.step >= cross - window) & (g.step <= cross)]
    post = g[(g.step >= cross) & (g.step <= cross + window)]
    if len(pre) < 2 or len(post) < 2:
        return np.nan, np.nan
    def rate(h):
        dw = h.w2.abs().values
        ds = h.step.values
        return (dw[-1] - dw[0]) / max(ds[-1] - ds[0], 1)
    return rate(pre), rate(post)


def main() -> None:
    d = pd.read_csv(RESULTS / "phase2b_checkpoints.csv")
    cr = d[d.crossing]
    print("=== DECISIVE STATISTIC: R at the first good-placement step ===\n")
    print(f"{'a':>6} {'B':>7} {'n crossed':>10} {'median R@cross':>15} {'IQR':>20} "
          f"{'% below 0.30':>13}")
    rows = []
    for (a, B), g in cr.groupby(["a", "budget"]):
        med = g.R.median()
        rows.append({"a": a, "budget": B, "n": len(g), "median_R": med,
                     "frac_below_030": float((g.R < 0.30).mean())})
        print(f"{a:>6} {B:>7} {len(g):>10} {med:>15.4f} "
              f"[{g.R.quantile(.25):.4f}, {g.R.quantile(.75):.4f}]".rjust(20)
              + f"{100 * (g.R < 0.30).mean():>12.1f}%")
    overall = cr.R.median()
    print(f"\n  OVERALL median R at crossing: {overall:.4f}")
    print(f"  registered: H-reverse if < 0.20, H-forward if >= 0.30")
    print(f"  fraction of crossings below R=0.30: {100 * (cr.R < 0.30).mean():.1f}%")

    print("\n=== SECONDARY: |w2| growth rate, before vs after crossing ===")
    ratios = []
    for (a, B, s), g in d.groupby(["a", "budget", "seed"]):
        c = g[g.crossing]
        if not len(c):
            continue
        pre, post = growth_ratio(g, int(c.step.iloc[0]))
        if np.isfinite(pre) and np.isfinite(post) and pre > 0:
            ratios.append(post / pre)
    ratios = np.array(ratios)
    if len(ratios):
        print(f"  n={len(ratios)}  median post/pre ratio: {np.median(ratios):.3f}")
        print(f"  registered: H-reverse if > 1.5")
        print(f"  fraction with ratio > 1: {100 * (ratios > 1).mean():.1f}%")
    pd.DataFrame(rows).to_csv(RESULTS / "phase2b_crossing.csv", index=False)


if __name__ == "__main__":
    main()
