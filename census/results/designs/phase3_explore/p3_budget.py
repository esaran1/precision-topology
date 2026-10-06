"""EXPLORATORY (population; exploration seeds 7,410,000-007 only).  The per-seed step-budget rule's input: the frozen
quasi-static DRIVE toward the switch.  Along the copy's frozen adiabatic reference (RK4 in s, 0.5% of s; w4core) the
SGD output update gives ds/step = ρη·D(s), D(s) = −sign(v)·∇_vL(z*(v), v).  The quasi-static step count to the switch
is t* = (1/ρη)·∫_{s₀}^{s_switch} ds/D(s) (trapezoid on the RK4 points).  Compared with the observed own-path switch
step t_sw of the runs in p3_runs.jsonl (same copy, sample, η, ρ): ratio t_sw/t*.
Usage: python p3_budget.py"""
import json, math
import numpy as np
import w4core as W
W.nice()
s0 = 0.2225397
v0 = s0 * np.array([0.1, 0.2, 0.3, 0.4])
b3 = json.load(open("p3_land_pop_box3.json"))["classes"]
b1 = json.load(open("p3_land_pop_0.1_0.2_0.3_0.4.json"))["classes"]
POP = {"Q": np.array(b3[49]["zc"]), "S": np.array(b3[39]["zc"]), "T4": np.array(b1[2]["zc"])}
R = [json.loads(l) for l in open("p3_runs.jsonl")]
T = W.Timer()


def integral(zp, x, y):
    """∫ ds/D along the adiabatic reference from v₀ to the first branch-gap sign change (the last step is cut at the
    bisected switch).  Returns (integral, s_switch)."""
    z, ok = W.newton(zp, v0, x, y)
    v = v0.copy()
    sw = W.continue_adiabatic(z, v0, x, y, 20.0)["s_switch"]
    D = lambda z, v: float(-np.sign(v) @ W.grad(z, v, x, y)[1])
    I, d0 = 0.0, D(z, v)
    while True:
        s = float(np.abs(v).sum())
        hs = min(0.005 * s, sw - s)
        zn, vn, ok = W.rk4_step(z, v, hs, x, y)
        d1 = D(zn, vn)
        I += 0.5 * hs * (1 / d0 + 1 / d1)
        z, v, d0 = zn, vn, d1
        if float(np.abs(v).sum()) >= sw * (1 - 1e-12) or hs <= 0:
            return I, sw


out = []
seen = set()
for r in R:
    key = (r["copy"], r["sample"])
    x, y = W.population() if r["sample"] == "pop" else W.own_sample(int(r["sample"]))
    if key not in seen:
        seen.add(key)
        I, sw = integral(POP[r["copy"]], x, y)
        cache = (I, sw)
    tstar = cache[0] / (r["rho"] * r["eta"])
    o = {"copy": r["copy"], "sample": r["sample"], "rho": r["rho"], "t_star": tstar, "t_sw": r["t_sw"],
         "t_obs": r["t_obs"], "ratio_tsw": r["t_sw"] / tstar, "ratio_tobs": r["t_obs"] / tstar}
    out.append(o)
    print(T(), json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
    W.rss_guard()
rat = np.array([o["ratio_tobs"] for o in out])
print(f"t_obs/t*: min {rat.min():.4f} median {np.median(rat):.4f} max {rat.max():.4f} (n {len(rat)})")
json.dump(out, open("p3_budget.json", "w"), indent=0)
print("peak RSS GB", round(W.rss_gb(), 3))
