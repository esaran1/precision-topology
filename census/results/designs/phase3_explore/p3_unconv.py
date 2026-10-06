"""EXPLORATORY (population).  The starts of p3_landscape.py whose Newton was not accepted after the W-step hold: the same
draws (default_rng(seed_draw), z ~ U(−1, 1)⁹ in order), held for W_long steps, then Newton.  Reports per start: accepted,
type, placed, max|∇_z|, λ_min of H_z at the held state, min |α|, the coincidence partition at 1e−3, and G at release.
Usage: python p3_unconv.py LANDSCAPE_JSON W_long"""
import json, sys
import numpy as np
import w4core as W
W.nice()
d = json.load(open(sys.argv[1])); WL = int(sys.argv[2])
x, y = W.population()
v0 = d["s0"] * np.array(d["shares"])
rng = np.random.default_rng(d["seed_draw"])
Z0 = [rng.uniform(-1, 1, 9) for _ in range(d["N"])]
T = W.Timer()
out = []
for r in d["rows"]:
    if r["ok"]:
        continue
    z, npos = W.hold(Z0[r["i"]], v0, WL, x, y, check_every=200)
    g = float(np.abs(W.gz(z, v0, x, y)).max())
    lam = float(np.linalg.eigvalsh(W.blocks(z, v0, x, y)[0]).min())
    zn, ok = W.newton(z, v0, x, y)
    grp = W.groups(z, tol=1e-3)
    G = W.gap_dense(z, v0)
    o = {"i": r["i"], "ok": ok, "type": W.btype(zn, v0) if ok else None, "placed": W.placed_state(zn, v0) if ok else None,
         "grad": g, "lam_state": lam, "min_abs_alpha": float(np.abs(z[:4]).min()), "groups_1e-3": [list(t) for t in grp],
         "G_dense_state": G, "dist": float(np.abs(z - zn).max()), "z": z.tolist()}
    out.append(o)
    print(T(), o["i"], ok, o["type"], o["placed"], f"grad {g:.1e} lam {lam:.2e} min|a| {o['min_abs_alpha']:.3f} groups {grp} G {G:.3f}", flush=True)
    W.rss_guard()
json.dump(out, open(sys.argv[1].replace(".json", f"_unconv{WL}.json"), "w"))
print("accepted after the long hold:", sum(o["ok"] for o in out), "of", len(out))
