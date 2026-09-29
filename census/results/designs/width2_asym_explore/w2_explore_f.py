"""EXPLORATORY.  ALREADY-USED T2-3 seeds 600,000-600,005 (their asymmetric training sets; landscape and holds only, no run
of any registered or candidate seed).
(i) Equal shares, seed 600,000 (24 of 60 starts not accepted after 4,000 hold steps in w2_explore_e): the same 60 starts
    (default_rng(seed + 7)) held for 40,000 steps at lr 1.0; counts by class.
(ii) Unequal shares v₀ = s₀·(0.1, 0.9): the own-sample copy of the population's main unplaced two-unit branch T1 (Newton
    from w2_explore_a_s0.2225_u0.1's class 0), its adiabatic switch (w2_explore_b's flow) and share there, and 40 random
    starts held 4,000 steps (default_rng(seed + 11)): counts by class.
Writes w2_explore_f.json."""
import json
import math
import os

import numpy as np

os.environ["W2A"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w2_explore_a_s0.2225_u0.1.json")
import w2core as W

W.nice()
xp, yp = W.setup(); ac = W.act()
HERE = os.path.dirname(os.path.abspath(__file__))
A1 = json.load(open(os.path.join(HERE, "w2_explore_a_s0.2225_u0.1.json")))
s0 = A1["s0"]; vq = np.array(A1["v0"]); veq = np.array([s0 / 2, s0 / 2])


def landing(x, y, v, n, rng, steps):
    cnt = {}
    for _ in range(n):
        z = rng.uniform(-1, 1, 5)
        for _ in range(steps):
            z = z - 1.0 * W.gz(z, v, x, y)
        zn, ok, _ = W.newton(z, v, x, y)
        key = "not accepted" if not ok else f"{W.canon(zn, v)[0]} {'placed' if W.gap(zn, v, ac)[0] > 0 else 'unplaced'}"
        cnt[key] = cnt.get(key, 0) + 1
    return cnt


def adiabatic(z, v, x, y, s_max=30.0, hs=0.005):
    g0 = W.gapm(z, v, ac)
    while v.sum() < s_max:
        def rhs(zz, vv):
            dv = -W.grad(W.qof(zz, vv), x, y)[W.VI]
            ds = np.sign(vv) @ dv
            return dv / ds if ds > 0 else None
        r1 = rhs(z, v)
        if r1 is None:
            return "s decreasing", None, None
        hstep = hs * np.abs(v).sum()
        vm = v + 0.5 * hstep * r1
        zm, ok, _ = W.newton(z + W.dz_ds(z, v, r1, x, y) * 0.5 * hstep, vm, x, y)
        r2 = rhs(zm, vm) if ok else None
        if r2 is None:
            return "failed", float(v.sum()), None
        vn = v + hstep * r2
        zn, ok, _ = W.newton(z + W.dz_ds(z, v, r2, x, y) * hstep, vn, x, y)
        if not ok:
            return "newton failed", float(vn.sum()), None
        if (W.gapm(zn, vn, ac) > 0) != (g0 > 0):
            return "switch", float(np.abs(vn).sum()), float(vn[0] / np.abs(vn).sum())
        z, v = zn, vn
    return "none", None, None


out = {}
x, y = W.own_sample(600000)
out["600000_equal_shares_40000_steps"] = landing(x, y, veq, 60, np.random.default_rng(600000 + 7), 40000)
print(out, flush=True)
r = next(r for r in A1["rows"] if r["ok"] and abs(r["G_lo"] - A1["classes"][0]["G_lo"]) < 1e-9)
zT = np.array(r["z"])
out["unequal"] = []
for seed in range(600000, 600006):
    x, y = W.own_sample(seed)
    z, ok, _ = W.newton(zT, vq, x, y)
    rec = {"seed": seed, "own_T1": bool(ok and W.canon(z, vq)[0] == "two-unit" and W.gap(z, vq, ac)[1] <= 0)}
    if rec["own_T1"]:
        rec["adiabatic"] = adiabatic(z.copy(), vq.copy(), x, y)
    rec["landing"] = landing(x, y, vq, 40, np.random.default_rng(seed + 11), 4000)
    out["unequal"].append(rec)
    print(json.dumps(rec), flush=True)
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1, default=float)
