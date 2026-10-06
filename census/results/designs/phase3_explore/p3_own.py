"""EXPLORATORY (fresh exploration own samples 7,410,000 + i; verified unused, p3_seedscan.log).  Per seed, at
v₀ = s₀·(0.1, 0.2, 0.3, 0.4), s₀ = 0.2225397, for the three candidate copies
  Q  = the four-function branch (population class 49 of p3_land_pop_box3.json, type 1111:++--), headline candidate
  S  = a three-function negative-κ branch (population class 39 of p3_land_pop_box3.json, type 121:+-+), sign arm
  T4 = W2-A's T function embedded (population class 2 of p3_land_pop_0.1_0.2_0.3_0.4.json, type 31:+-), control
(i) own copy: damped Newton from the population point on the own sample; accepted and of the same type;
(ii) its adiabatic switch (RK4 0.5% of s) and κ₀, a, λ at the switch;
(iii) branch-point start (Q, S): the population point held W = max(4000, ⌈25/λ_own(s₀)⌉) steps at lr 1.0, Newton,
     on the own copy (within 1e−6 up to windings, state within 1e−3)?  hold G > 0 (checked every 50 steps)?
(iv) random hold (z ~ U(−1, 1)⁹, box 1, default_rng(seed)), NR starts per seed, W = 8000: landing on the own T4 / S / Q
     copy and on any T-type 31/22 copy; counts.
Usage: python p3_own.py first n NR"""
import json, sys
import numpy as np
import w4core as W
W.nice()
first, n, NR = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
s0 = 0.2225397
sh = np.array([0.1, 0.2, 0.3, 0.4]); v0 = s0 * sh
b3 = json.load(open("p3_land_pop_box3.json"))["classes"]
b1 = json.load(open("p3_land_pop_0.1_0.2_0.3_0.4.json"))["classes"]
POP = {"Q": np.array(b3[49]["zc"]), "S": np.array(b3[39]["zc"]), "T4": np.array(b1[2]["zc"])}
T = W.Timer()


def same(zn, zc):
    """On copy zc up to windings: β differences multiples of 2π, α and reduced β within 1e−6, b after the shift."""
    k = np.rint((zn[4:8] - zc[4:8]) / W.TWO_PI)
    zk = W.shift(zc, v0, k)
    return bool(np.abs(zn - zk).max() <= 1e-6), tuple(int(i) for i in k)


out = []
for seed in range(first, first + n):
    x, y = W.own_sample(seed)
    rec = {"seed": seed}
    own = {}
    for c, zp in POP.items():
        zo, ok = W.newton(zp, v0, x, y)
        typ = W.btype(zo, v0) if ok else None
        good = ok and typ == W.btype(zp, v0) and not W.placed_state(zo, v0)
        r = {"ok": ok, "type": typ, "valid": good}
        if good:
            own[c] = zo
            lam0 = W.reduced_lam(W.blocks(zo, v0, x, y)[0], zo, v0)[0]
            r["lam_s0"] = lam0
            a = W.continue_adiabatic(zo, v0, x, y, 20.0)
            r["status"] = a["status"]
            if a["status"] == "switch":
                k = W.kappa_at(a["z"], a["v"], a["vprime"], x, y)
                r.update(s_switch=a["s_switch"], kappa0=k["kappa0"], a=k["a"], lam_switch=k["lam"],
                         min_split=a["min_split"], kappa_over_lam=k["kappa0"] / k["lam"])
            if c in ("Q", "S"):
                Wh = max(4000, int(np.ceil(25 / lam0)))
                zr, npos = W.hold(zp, v0, Wh, x, y, check_every=50)
                zn, okn = W.newton(zr, v0, x, y)
                on, kk = same(zn, zo) if okn else (False, None)
                r.update(W_hold=Wh, bp_on=bool(on and np.abs(zr - zn).max() <= 1e-3), bp_k=kk, bp_hold_pos=npos)
        rec[c] = r
    rng = np.random.default_rng(seed)
    land = {"T4": 0, "S": 0, "Q": 0, "Ttype": 0, "unconv": 0, "placed": 0, "other": 0}
    for i in range(NR):
        zr, _ = W.hold(rng.uniform(-1, 1, 9), v0, 8000, x, y, check_every=10 ** 9)
        zn, ok = W.newton(zr, v0, x, y)
        if not ok:
            land["unconv"] += 1; continue
        hit = None
        for c, zo in own.items():
            if same(zn, zo)[0] and np.abs(zr - zn).max() <= 1e-3:
                hit = c
        if hit:
            land[hit] += 1
        if W.placed_state(zn, v0):
            land["placed"] += 1
        elif W.btype(zn, v0) in ("31:+-", "22:+-"):
            land["Ttype"] += 1
        elif not hit:
            land["other"] += 1
    rec["random_hold"] = land
    out.append(rec)
    print(T(), json.dumps(rec, default=lambda o: round(float(o), 6) if isinstance(o, (float, np.floating)) else str(o)), flush=True)
    W.rss_guard()
    json.dump(out, open(f"p3_own_{first}_{n}.json", "w"), default=float)
print("peak RSS GB", round(W.rss_gb(), 3))
