"""EXPLORATORY (population).  For each accepted UNPLACED class of a landscape file (p3_landscape.py): the adiabatic
reference (reduced gradient flow of v, RK4 in s at 0.5% of s; w4core.continue_adiabatic) from the class point at v₀ to
the first branch-gap sign change (bisected); κ₀, the four winding coefficients a (κ_k = κ₀ + a·k, exact), the reduced
λ_min at the switch (split directions of every coinciding group excluded), the minimum split-block eigenvalue along the
path (a negative value means the coinciding units would separate before the switch), the partition at the switch, the
merged shares at the switch, κ/λ (the slaved lag scale, steps per unit η), and optionally the switch with the step
halved.  Usage: python p3_branches.py LANDSCAPE_JSON [min_count] [halve]"""
import json, sys
import numpy as np
import w4core as W
W.nice()
d = json.load(open(sys.argv[1]))
mc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
halve = len(sys.argv) > 3
x, y = W.population()
v0 = d["s0"] * np.array(d["shares"])
T = W.Timer()
import os
MULTI = os.environ.get("MULTI") == "1"
out = []
for j, c in sorted(enumerate(d["classes"]), key=lambda t: -t[1]["n"]):
    if c["placed"] or c["n"] < mc:
        continue
    if MULTI and len(c["type"].split(":")[0]) < 3:      # MULTI=1: only classes with >= 3 distinct functions
        continue
    z0 = np.array(c["zc"])
    # the canonical point may permute equal-share units; it is still a stationary point at v₀ (equal shares)
    z0, ok = W.newton(z0, v0, x, y)
    r = W.continue_adiabatic(z0, v0, x, y, 40.0)
    o = {"class": j, "n": c["n"], "type": c["type"], "status": r["status"], "groups_s0": [list(t) for t in W.groups(z0)],
         "lam_s0": W.reduced_lam(W.blocks(z0, v0, x, y)[0], z0, v0)}
    if r["status"] == "switch":
        k = W.kappa_at(r["z"], r["v"], r["vprime"], x, y)
        gs = W.groups(r["z"])
        o.update(s_switch=r["s_switch"], min_split=r["min_split"], groups_switch=r["groups_switch"],
                 merged_shares_switch=[float(sum(r["v"][i] for i in g) / r["s_switch"]) for g in gs],
                 kappa0=k["kappa0"], a=k["a"], lam_switch=k["lam"], lam_split_switch=k["lam_split"],
                 kappa_over_lam=k["kappa0"] / k["lam"], dGz_tan=k["dGz_tan"], dGv=k["dGv"], gradG_dis=k["gradG_rel_dis"],
                 vprime=list(map(float, r["vprime"])))
        if halve:
            r2 = W.continue_adiabatic(z0, v0, x, y, 40.0, hs_rel=0.0025)
            o["s_switch_halved"] = r2.get("s_switch")
    else:
        o.update({k_: (v_ if not isinstance(v_, np.ndarray) else v_.tolist()) for k_, v_ in r.items() if k_ not in ("z", "v")})
    out.append(o)
    def _r(k_, v_):
        try:
            return np.round(v_, 5).tolist() if isinstance(v_, (list, float)) and not k_.startswith("groups") else v_
        except ValueError:
            return v_
    print(T(), json.dumps({k_: _r(k_, v_) for k_, v_ in o.items()}, default=str), flush=True)
    W.rss_guard()
json.dump(out, open(sys.argv[1].replace(".json", "_branches.json"), "w"), indent=0)
print("peak RSS GB", round(W.rss_gb(), 3))
