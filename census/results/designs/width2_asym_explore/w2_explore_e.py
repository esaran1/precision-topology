"""EXPLORATORY.  Own-sample landscape on ALREADY-USED T2-3 seeds 600,000-600,011 (their asymmetric training sets
asym_register.training_set(seed); no training of any run, landscape only), and unequal held shares on the population.
(i) per seed: Newton at v₀ = (s₀/2, s₀/2) from each population duplicate branch point (D1, D2 of w2_explore_b), the
    own-sample copy; continuation along the diagonal to its switch (and with h halved); κ (P = I, reduced Hessian) at the
    switch for k₁ + k₂ = 0 and λ_min there; λ_min at s₀; the minimum antisymmetric eigenvalue along the path (sampled);
    and 60 random hidden starts (default_rng(seed + 7)) held as in w2_explore_d (lr 1.0, 4,000 steps): counts by class.
(ii) population, v₀ = s₀·(u, 1 − u), u ∈ {0.1, 0.25, 0.4}: 100 random starts each, counts by class.
Writes w2_explore_e.json."""
import json
import math

import numpy as np

import w2core as W

W.nice()
xp, yp = W.setup(); ac = W.act()
A_ = json.load(open(__file__.replace("_e.py", "_a.json")))
B_ = json.load(open(__file__.replace("_e.py", "_b.json")))
s0 = A_["s0"]; v0 = np.array(A_["v0"]); d = np.array([0.5, 0.5])


def cls_point(ci):
    r = next(r for r in A_["rows"] if r["ok"] and abs(r["G_lo"] - A_["classes"][ci]["G_lo"]) < 1e-9)
    z = np.array(r["z"]); k = r["k"]
    z[1] -= 2 * math.pi * k[0]; z[3] -= 2 * math.pi * k[1]; z[4] += 2 * math.pi * (k[0] * v0[0] + k[1] * v0[1])
    return W.newton(z, v0, xp, yp)[0]


def landing(x, y, v, n, rng):
    cnt = {}
    for _ in range(n):
        z = rng.uniform(-1, 1, 5)
        for _ in range(4000):
            z = z - 1.0 * W.gz(z, v, x, y)
        zn, ok, _ = W.newton(z, v, x, y)
        key = "not accepted" if not ok else f"{W.canon(zn, v)[0]} {'placed' if W.gap(zn, v, ac)[0] > 0 else 'unplaced'}"
        cnt[key] = cnt.get(key, 0) + 1
    return cnt


pops = {ci: cls_point(ci) for ci, c in enumerate(A_["classes"]) if c["class"][0] == "duplicate"}
out = {"own": [], "shares": {}}
for seed in range(600000, 600012):
    x, y = W.own_sample(seed)
    rec = {"seed": seed}
    for ci, zp in pops.items():
        z, ok, _ = W.newton(zp, v0, x, y)
        tag = f"D{ci}"
        if not ok or W.canon(z, v0)[0] != "duplicate":
            rec[tag] = "no own copy"; continue
        lr0, la0, _ = W.lam_reduced(z, v0, x, y)
        st, ssw, zsw, path = W.continue_ray(z, s0, d, x, y, ac, 30.0)
        st2, ssw2, _, _ = W.continue_ray(z, s0, d, x, y, ac, 30.0, h=0.005)
        r = {"lam_s0": lr0, "status": st, "s_sw": ssw, "s_sw_half": ssw2}
        if st == "switch":
            vs = ssw * d
            zsw, _, _ = W.newton(zsw, vs, x, y)
            kap, lam, _, _dgv = W.kappa(zsw, vs, d, x, y, ac, drop=W.antisym_basis(zsw, vs))
            r.update({"kappa_k0": kap, "lam_sw": lam, "antisym_sw": W.lam_reduced(zsw, vs, x, y)[1],
                      "G_s0": W.gap(z, v0, ac)[1]})
        rec[tag] = r
    rec["landing"] = landing(x, y, v0, 60, np.random.default_rng(seed + 7))
    out["own"].append(rec)
    print(json.dumps(rec, default=float), flush=True)
for u in (0.1, 0.25, 0.4):
    v = s0 * np.array([u, 1 - u])
    out["shares"][str(u)] = landing(xp, yp, v, 100, np.random.default_rng(20261001))
    print(u, out["shares"][str(u)], flush=True)
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1, default=float)
