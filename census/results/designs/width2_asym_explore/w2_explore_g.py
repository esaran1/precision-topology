"""EXPLORATORY arithmetic (no training): pooled own-sample landing rates from w2_explore_e (arm D, 12 used seeds × 60
starts) and w2_explore_f (arm T, 6 used seeds × 40 starts), and the binomial chance that ≥ 60 of 120 runs (the proposed
gate) or ≥ 96 of 120 (GELU-T's 80%) land on the arm's branch type.  Ignores seed-to-seed heterogeneity."""
import json
from math import comb

e = json.load(open(__file__.replace("_g.py", "_e.json")))
f = json.load(open(__file__.replace("_g.py", "_f.json")))
pD = sum(r["landing"].get("duplicate unplaced", 0) for r in e["own"]) / (60 * len(e["own"]))
pT = sum(r["landing"].get("two-unit unplaced", 0) for r in f["unequal"]) / (40 * len(f["unequal"]))
sf = lambda k, n, p: sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
for arm, p in (("D", pD), ("T", pT)):
    print(arm, "pooled rate", round(p, 4), "P(>=60/120)", round(sf(60, 120, p), 4), "P(>=96/120)", f"{sf(96, 120, p):.1e}")
