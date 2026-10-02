"""EXPLORATORY: power of the proposed criteria at n = 80 (as explored), same settings as p2b_lever_power.py."""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import p2b_lever_power as P
std, a2, a3 = P.load("std"), P.load("out16"), P.load("glob")
D2, D3 = P.paired(std, a2), P.paired(std, a3)
on = np.array([bool(a2[s]["onset"].get(P.ONSET_END)) and a2[s]["onset"][P.ONSET_END]["s_over_s_star"] > 1 for s in P.SEEDS])
rng = np.random.default_rng(P.SEED_REG + 2)
hits = {"R": 0, "N": 0, "O": 0, "all": 0}
for _ in range(300):
    rows = rng.integers(0, 20, 80)
    c = P.criteria(D2[rows], D3[rows], on[rows], 1000, rng)
    for k in ("R", "N", "O"):
        hits[k] += c[k]
    hits["all"] += c["R"] and c["N"] and c["O"]
res = {"label": "EXPLORATORY", "n": 80, "sims": 300, "inner_bootstrap": 1000, "rng": P.SEED_REG + 2,
       "power": {k: v / 300 for k, v in hits.items()}}
(P.HERE / "p2b_lever_power80.json").write_text(json.dumps(res, indent=1)); print(res)
