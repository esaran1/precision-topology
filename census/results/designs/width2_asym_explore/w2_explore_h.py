"""EXPLORATORY (added after the author's approval with changes, 2026-09-29; population and ALREADY-USED T2-3 seeds
600,000-600,011 only; no 884,xxx seed).  How can a registered T′ arm get its runs?
(i) Is T′ an image of T?  T and T′ at v₀ = s₀·(0.1, 0.9) (w2_explore_a_s0.2225_u0.1 classes 0 and 2) in canonical form
    (v ≥ 0): the big-share unit's slope sign, and whether T′ equals T under unit swap (T at s₀·(0.9, 0.1)).
(ii) Landing scan on the population: v₀ = s₀·(u, 1 − u), u in U_SCAN, 100 random hidden starts each (default_rng(20261002)),
    GD hold lr 1.0, 4,000 steps, Newton; two-unit points typed by (sign α_big, sign α_small), placed or not.
    T-type = (+, −) unplaced, T′-type = (−, +) unplaced.
(iii) Branch-point start for T′ on the own samples of seeds 600,000-600,011: from the population T′ point at s₀·(0.1, 0.9),
    GD hold on the own sample (lr 1.0, W = max(4000, ⌈25/λ_min⌉)) and Newton: does the release land on the own T′ copy
    (unplaced two-unit, (−, +), within 1e−6 of Newton-from-population)?  The own T′ copy's adiabatic switch, the share
    there, and κ (P = I, total dG/ds) at it with the local flow direction.
Writes w2_explore_h.json."""
import json
import math
import os

import numpy as np

import w2core as W

W.nice()
xp, yp = W.setup(); ac = W.act()
HERE = os.path.dirname(os.path.abspath(__file__))
A1 = json.load(open(os.path.join(HERE, "w2_explore_a_s0.2225_u0.1.json")))
s0 = A1["s0"]; vq = np.array(A1["v0"])
U_SCAN = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)


def cls_point(ci, A, v):
    r = next(r for r in A["rows"] if r["ok"] and abs(r["G_lo"] - A["classes"][ci]["G_lo"]) < 1e-9)
    z = np.array(r["z"]); k = r["k"]
    z[1] -= 2 * math.pi * k[0]; z[3] -= 2 * math.pi * k[1]; z[4] += 2 * math.pi * (k[0] * v[0] + k[1] * v[1])
    return W.newton(z, v, xp, yp)[0]


def typ(z, v, x, y):
    kind, _, red = W.canon(z, v)
    placed = W.gap(z, v, ac)[0] > 0
    if kind == "duplicate":
        return "D " + ("placed" if placed else "unplaced")
    big, small = (red[0], red[1]) if red[0][2] >= red[1][2] else (red[1], red[0])
    sg = lambda a: "+" if a > 0 else "-"
    return f"two-unit ({sg(big[0])},{sg(small[0])}) " + ("placed" if placed else "unplaced")


def adiabatic_switch(z, v, x, y, s_max=30.0, hs=0.005):
    g0 = W.gapm(z, v, ac)
    while v.sum() < s_max:
        def rhs(zz, vv):
            dv = -W.grad(W.qof(zz, vv), x, y)[W.VI]
            ds = np.sign(vv) @ dv
            return dv / ds if ds > 0 else None
        r1 = rhs(z, v)
        if r1 is None:
            return None
        hstep = hs * np.abs(v).sum()
        vm = v + 0.5 * hstep * r1
        zm, ok, _ = W.newton(z + W.dz_ds(z, v, r1, x, y) * 0.5 * hstep, vm, x, y)
        r2 = rhs(zm, vm) if ok else None
        if r2 is None:
            return None
        vn = v + hstep * r2
        zn, ok, _ = W.newton(z + W.dz_ds(z, v, r2, x, y) * hstep, vn, x, y)
        if not ok:
            return None
        if (W.gapm(zn, vn, ac) > 0) != (g0 > 0):
            zs, _, _ = W.newton(zn, vn, x, y)
            kap = W.kappa(zs, vn, r2, x, y, ac)
            return {"s_sw": float(vn.sum()), "share": float(vn[0] / vn.sum()), "kappa": kap[0], "lam": kap[1],
                    "dGz_tp": kap[2], "dGv": kap[3]}
        z, v = zn, vn
    return None


out = {}
zT, zTp = cls_point(0, A1, vq), cls_point(2, A1, vq)
out["types_at_u0.1"] = {"T": typ(zT, vq, xp, yp), "T'": typ(zTp, vq, xp, yp)}
sw = np.array([zT[2], zT[3], zT[0], zT[1], zT[4]])                  # unit swap of T: T at s₀·(0.9, 0.1)
out["T'_equals_swapped_T"] = bool(np.abs(sw - zTp).max() < 1e-6)
out["T_swapped_is_stationary_at_(0.9,0.1)"] = bool(np.abs(W.gz(sw, vq[::-1], xp, yp)).max() < 1e-9)
print(json.dumps(out), flush=True)
out["scan"] = {}
for u in U_SCAN:
    v = s0 * np.array([u, 1 - u])
    rng = np.random.default_rng(20261002)
    cnt = {}
    for _ in range(100):
        z = rng.uniform(-1, 1, 5)
        for _ in range(4000):
            z = z - 1.0 * W.gz(z, v, xp, yp)
        zn, ok, _ = W.newton(z, v, xp, yp)
        key = typ(zn, v, xp, yp) if ok else "not accepted"
        cnt[key] = cnt.get(key, 0) + 1
    out["scan"][str(u)] = cnt
    print(u, cnt, flush=True)
out["own_Tprime"] = []
for seed in range(600000, 600012):
    x, y = W.own_sample(seed)
    zc, ok, _ = W.newton(zTp, vq, x, y)
    rec = {"seed": seed, "own_copy": bool(ok), "type": typ(zc, vq, x, y) if ok else None}
    if ok:
        lam0 = float(np.linalg.eigvalsh(W.Hz(zc, vq, x, y)).min())
        Wsteps = max(4000, math.ceil(25 / lam0))
        z = zTp.copy()
        for _ in range(Wsteps):
            z = z - 1.0 * W.gz(z, vq, x, y)
        zn, okn, _ = W.newton(z, vq, x, y)
        rec.update({"lam_s0": lam0, "W": Wsteps, "release_on_copy": bool(okn and np.abs(zn - zc).max() < 1e-6),
                    "release_dist": float(np.abs(z - zc).max()), "adiabatic": adiabatic_switch(zc.copy(), vq.copy(), x, y)})
    out["own_Tprime"].append(rec)
    print(json.dumps(rec, default=float), flush=True)
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1, default=float)
