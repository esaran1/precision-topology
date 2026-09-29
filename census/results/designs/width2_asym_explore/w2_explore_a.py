"""EXPLORATORY (population only; no candidate seed).  (0) analytic gradient/Hessian vs finite differences.
(1) Landscape at the held output v₀ = (s₀/2, s₀/2), s₀ = 0.5·s_pop2 (s_pop2 = 0.44508, T2-1's switch): 300 random hidden
starts z ~ U(−1, 1)⁵ (numpy default_rng(20260929)), each held by full-batch GD on z at lr 1.0 for W = 20,000 steps,
then damped Newton at v₀.  Classified: accepted (|∇| < 1e−10, H PD), kind (two-unit / duplicate mod 2π), windings, placed.
Distinct branch points listed with λ_min, λ_max, the release-to-Newton distance and G₊.
Writes w2_explore_a.json (default s0), or w2_explore_a_s<s0>.json with arguments: python w2_explore_a.py S0 LR W [U N]."""
import json
import math
import time

import numpy as np

import w2core as W

W.nice()
x, y = W.setup(); ac = W.act()
S_POP2 = 0.44507940623559955
import sys
s0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5 * S_POP2     # optional: s0, hold lr, W
U = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5             # optional 4th argument: share u, v0 = s0·(u, 1 − u)
N_ARG = int(sys.argv[5]) if len(sys.argv) > 5 else 300
v0 = np.array([s0 * U, s0 * (1 - U)])

# (0) derivative check
rng = np.random.default_rng(1)
q = rng.uniform(-1, 1, 7)
g = W.grad(q, x, y); H = W.hess(q, x, y)
eps = 1e-6
gfd = np.array([(W.loss(q + eps * e, x, y) - W.loss(q - eps * e, x, y)) / (2 * eps) for e in np.eye(7)])
Hfd = np.array([(W.grad(q + eps * e, x, y) - W.grad(q - eps * e, x, y)) / (2 * eps) for e in np.eye(7)])
print("grad err", np.abs(g - gfd).max(), "hess err", np.abs(H - Hfd).max(), flush=True)

rng = np.random.default_rng(20260929)
N, WSTEPS, LR = N_ARG, 20000, 1.0
if len(sys.argv) > 2:
    LR, WSTEPS = float(sys.argv[2]), int(sys.argv[3])
rows = []
t0 = time.time()
for i in range(N):
    z = rng.uniform(-1, 1, 5)
    z_init = z.copy()
    G_init = W.gap(z, v0, ac)
    for t in range(WSTEPS):
        z = z - LR * W.gz(z, v0, x, y)
    zr = z.copy()
    zn, ok, gm = W.newton(zr, v0, x, y)
    ev = np.linalg.eigvalsh(W.Hz(zn, v0, x, y))
    kind, k, red = W.canon(zn, v0)
    G = W.gap(zn, v0, ac)
    rows.append({"i": i, "ok": ok, "gmax": gm, "kind": kind, "k": list(k), "red": [list(map(float, r)) for r in red],
                 "lam_min": float(ev[0]), "lam_max": float(ev[-1]), "dist": float(np.abs(zr - zn).max()),
                 "grad_release": float(np.abs(W.gz(zr, v0, x, y)).max()), "G_lo": G[0], "G_hi": G[1],
                 "G_init_lo": G_init[0], "z": list(map(float, zn))})
    if i % 20 == 0:
        print(i, round(time.time() - t0), kind, k, ok, round(ev[0], 8), round(G[0], 4), flush=True)

acc = [r for r in rows if r["ok"]]
print("accepted", len(acc), "of", N)
cls = {}
for r in acc:
    key = (r["kind"], round(r["red"][0][0], 5), round(r["red"][0][1], 5), round(r["red"][1][0], 5), round(r["red"][1][1], 5))
    key2 = (r["kind"], round(r["red"][1][0], 5), round(r["red"][1][1], 5), round(r["red"][0][0], 5), round(r["red"][0][1], 5))
    kk = key if key in cls else (key2 if key2 in cls else key)
    cls.setdefault(kk, []).append(r)
summary = []
for kk, rs in sorted(cls.items(), key=lambda t: -len(t[1])):
    wind = {}
    for r in rs:
        wind[str(tuple(r["k"]))] = wind.get(str(tuple(r["k"])), 0) + 1
    d = {"class": [str(c) for c in kk], "n": len(rs), "placed": rs[0]["G_lo"] > 0, "G_lo": rs[0]["G_lo"],
         "lam_min": rs[0]["lam_min"], "lam_max": rs[0]["lam_max"], "windings": wind,
         "max_release_dist": max(r["dist"] for r in rs), "max_release_grad": max(r["grad_release"] for r in rs),
         "loss": W.loss(W.qof(np.array(rs[0]["z"]), v0), x, y)}
    summary.append(d)
    print(json.dumps(d))
print("placed at init:", sum(r["G_init_lo"] > 0 for r in rows))
json.dump({"s0": s0, "v0": list(v0), "N": N, "W": WSTEPS, "lr": LR, "rows": rows, "classes": summary},
          open(__file__.replace(".py", ".json" if len(sys.argv) == 1 else (f"_s{s0:.4f}.json" if U == 0.5 else f"_s{s0:.4f}_u{U}.json")), "w"), indent=1, default=float)
