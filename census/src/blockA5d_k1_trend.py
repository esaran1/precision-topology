"""Block A5d at k = 1: budget trend of the perfect fraction. POST HOC, descriptive.

Not a registered test.  The registration (blockA5d_k1_prediction.md) defined the
onset as >= half perfect, and that level was never reached.  This tabulates, for
every a >= 1.35 and for GELU, the fraction perfect at each budget with an exact
(Clopper-Pearson) 95% interval, plus the smallest grid a with ANY perfect run at
each budget.  Perfect = the registered criterion (0 errors, uniform + stratified).

    python -m src.blockA5d_k1_trend
"""

from __future__ import annotations

from math import comb

import pandas as pd

from .blockA5d_k1 import RESULTS, RUNS

OUT = RESULTS / "blockA5d_k1_budget_trend.csv"
FIRST = RESULTS / "blockA5d_k1_first_perfect.csv"


def clopper_pearson(k, n, alpha=0.05):
    def tail_ge(p):
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))
    lo = hi = None
    if k == 0:
        lo = 0.0
    else:
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            a, b = (m, b) if tail_ge(m) < alpha / 2 else (a, m)
        lo = a
    if k == n:
        hi = 1.0
    else:
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            a, b = (m, b) if tail_le(m) > alpha / 2 else (a, m)
        hi = a
    return lo, hi


def main():
    d = pd.read_csv(RUNS, dtype={"act": str})
    d["perfect"] = d.perfect.astype(str).str.lower() == "true"
    rows = []
    for act, g in d.groupby("act"):
        keep = act == "gelu" or (act != "relu" and float(act) >= 1.35)
        if not keep:
            continue
        for B, gb in g.groupby("budget"):
            k, n = int(gb.perfect.sum()), len(gb)
            lo, hi = clopper_pearson(k, n)
            rows.append({"act": act, "budget": int(B), "k": k, "n": n, "frac": k / n,
                         "ci95_lo": lo, "ci95_hi": hi, "reaches_half": k / n >= 0.5})
    t = pd.DataFrame(rows)
    t["order"] = t.act.map(lambda a: 99.0 if a == "gelu" else float(a))
    t = t.sort_values(["order", "budget"]).drop(columns="order")
    t.to_csv(OUT, index=False)
    fa = d[~d.act.isin(["relu", "gelu"])].copy()
    fa["a"] = fa.act.astype(float)
    first = []
    for B, gb in fa.groupby("budget"):
        per = gb.groupby("a").perfect.sum()
        pos = per[per > 0]
        first.append({"budget": int(B), "smallest_a_any_perfect": float(pos.index.min()) if len(pos) else None,
                      "max_frac_any_a": float(gb.groupby("a").perfect.mean().max()),
                      "onset_half_reached": bool(gb.groupby("a").perfect.mean().max() >= 0.5)})
    f = pd.DataFrame(first)
    f.to_csv(FIRST, index=False)
    print(t.to_string(index=False))
    print()
    print(f.to_string(index=False))


if __name__ == "__main__":
    main()
