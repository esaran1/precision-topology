"""EXPLORATORY.  Landscape at a held output v₀ = s₀·shares (width 4): N random hidden starts z ~ U(−1, 1)⁹
(numpy default_rng(seed_draw)), held by full-batch GD on z at lr 1.0 for W steps (v fixed), then damped Newton at v₀.
Classified: accepted (max|∇| < 1e−10, H PD), the release-to-Newton sup distance, branch type (coincidence partition and
α signs, w4core.btype), windings, placed (exact enclosure).  Classes = distinct Newton points modulo windings (β mod 2π,
b shifted back) and permutations of units with EQUAL shares (units sorted within each equal-share set).
Sample: 'pop' (population) or an own-sample seed (asym_register.training_set).
Usage: python p3_landscape.py SAMPLE s0 u1,u2,u3,u4 N W seed_draw [tag]"""
import json, math, sys
import numpy as np
import w4core as W
W.nice()
samp, s0 = sys.argv[1], float(sys.argv[2])
sh = np.array([float(u) for u in sys.argv[3].split(",")]); sh = sh / sh.sum()
N, WST, SD = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
tag = sys.argv[7] if len(sys.argv) > 7 else ""
x, y = W.population() if samp == "pop" else W.own_sample(int(samp))
v0 = s0 * sh
K = 4
T = W.Timer()
import os
BOX = float(os.environ.get("BOX", "1"))   # start box: z ~ U(−BOX, BOX)⁹ (default 1)


def canon_key(z):
    k, red = W.windings(z)
    b = z[-1] + W.TWO_PI * float(np.asarray(k) @ v0)
    units = [(z[i], red[i]) for i in range(K)]
    # permutations among equal shares
    order = list(range(K))
    for val in set(np.round(sh, 12)):
        idx = [i for i in range(K) if round(sh[i], 12) == val]
        srt = sorted(idx, key=lambda i: (round(units[i][0], 5), round(units[i][1], 5)))
        for a_, b_ in zip(idx, srt):
            order[a_] = b_
    zc = np.array([units[o][0] for o in order] + [units[o][1] for o in order] + [b])
    return zc, k, order


rng = np.random.default_rng(SD)
rows, classes = [], []
for i in range(N):
    z = rng.uniform(-BOX, BOX, 2 * K + 1)
    zr, npos = W.hold(z, v0, WST, x, y, check_every=50)
    zn, ok = W.newton(zr, v0, x, y)
    dist = float(np.abs(zr - zn).max())
    row = {"i": i, "ok": ok, "dist": dist, "hold_pos50": npos}
    if ok:
        zc, k, order = canon_key(zn)
        pl = W.placed_state(zn, v0)
        lam = W.reduced_lam(W.blocks(zn, v0, x, y)[0], zn, v0)[0]
        cid = None
        for j, c in enumerate(classes):
            if np.abs(np.array(c["zc"]) - zc).max() < 1e-5:
                cid = j; break
        if cid is None:
            classes.append({"zc": zc.tolist(), "type": W.btype(zc, v0), "placed": bool(pl), "lam": lam, "n": 0,
                            "n_near": 0, "windings": {}, "perm": {}, "z_example": zn.tolist()})
            cid = len(classes) - 1
        c = classes[cid]
        c["n"] += 1; c["n_near"] += int(dist <= 1e-3)
        c["windings"][str(k)] = c["windings"].get(str(k), 0) + 1
        c["perm"][str(order)] = c["perm"].get(str(order), 0) + 1
        row.update(cls=cid, type=c["type"], placed=c["placed"], k=k)
    rows.append(row)
    if i % 25 == 0:
        print(i, T(), row.get("type"), row.get("placed"), ok, f"{dist:.1e}", flush=True)
    W.rss_guard()
acc = sum(r["ok"] for r in rows)
print(f"sample {samp} s0 {s0} shares {sh.round(4).tolist()} N {N} W {WST}: accepted {acc}/{N}; "
      f"release within 1e-3 of Newton {sum(r['ok'] and r['dist'] <= 1e-3 for r in rows)}; time {T()} s")
for j, c in sorted(enumerate(classes), key=lambda t: -t[1]["n"]):
    print(f"class {j}: n {c['n']} (near {c['n_near']}) type {c['type']} {'PLACED' if c['placed'] else 'unplaced'} "
          f"lam {c['lam']:.3g} windings {c['windings']} perms {len(c['perm'])} z {np.round(c['zc'], 4).tolist()}")
out = {"sample": samp, "s0": s0, "shares": sh.tolist(), "N": N, "W": WST, "seed_draw": SD, "rows": rows,
       "classes": classes}
json.dump(out, open(f"p3_land_{samp}_{tag or sys.argv[3].replace(',', '_')}.json", "w"))
print("peak RSS GB", round(W.rss_gb(), 3))
