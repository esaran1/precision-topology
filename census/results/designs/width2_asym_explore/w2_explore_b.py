"""EXPLORATORY (population only; no candidate seed).  For each branch class found at v₀ by w2_explore_a (canonical winding
(0, 0)): (i) natural continuation along the diagonal ray v = (s/2, s/2) from s₀ up to 30 (h ≤ 0.02 in log s, and again with
h ≤ 0.01: step-halving agreement), λ_min along the path, the switch s_sw,ray; (ii) the ADIABATIC path: the reduced gradient
flow dv/dτ = −∇_v L(z*(v), v) from v₀ (envelope theorem; RK2 in steps of 0.5% of s, z* by Newton), its switch s_sw,ad and
the share v₁/s there; (iii) κ (P = I) at the ray switch for windings (k₁, k₂) with k₁ + k₂ = −3..3 (duplicates: λ_min with
the 2-D antisymmetric subspace excluded, and also with it included); dL*/ds at s₀.
Writes w2_explore_b.json."""
import json
import math

import numpy as np

import w2core as W

W.nice()
x, y = W.setup(); ac = W.act()
import os
A_ = json.load(open(os.environ.get("W2A", __file__.replace("_b.py", "_a.json"))))     # W2A: another w2_explore_a output
s0, v0 = A_["s0"], np.array(A_["v0"])
d = v0 / np.abs(v0).sum()                  # the held direction (the diagonal for equal shares)


def shift(z, v, k):
    z = z.copy(); z[1] += 2 * math.pi * k[0]; z[3] += 2 * math.pi * k[1]; z[4] -= 2 * math.pi * (k[0] * v[0] + k[1] * v[1])
    return z


def adiabatic(z, v, s_max=30.0, hs=0.005):
    g0 = W.gapm(z, v, ac)
    path = []
    while v.sum() < s_max:
        def rhs(zz, vv):
            gv = W.grad(W.qof(zz, vv), x, y)[W.VI]
            dv = -gv
            ds = np.sign(vv) @ dv
            return dv / ds if ds > 0 else None                       # dv per unit s
        r1 = rhs(z, v)
        if r1 is None:
            return {"status": "s decreasing on the flow", "s": float(np.abs(v).sum()), "path": path}
        hstep = hs * np.abs(v).sum()
        vm = v + 0.5 * hstep * r1
        zm, ok, _ = W.newton(z + W.dz_ds(z, v, r1, x, y) * 0.5 * hstep, vm, x, y)
        r2 = rhs(zm, vm) if ok else None
        if r2 is None:
            return {"status": "failed", "s": float(np.abs(v).sum()), "path": path}
        vn = v + hstep * r2
        zn, ok, _ = W.newton(z + W.dz_ds(z, v, r2, x, y) * hstep, vn, x, y)
        if not ok:
            return {"status": "newton failed", "s": float(np.abs(v).sum()), "path": path}
        gn = W.gapm(zn, vn, ac)
        path.append([float(np.abs(vn).sum()), float(vn[0] / np.abs(vn).sum()), float(gn)])
        if (gn > 0) != (g0 > 0):
            return {"status": "switch", "s": float(np.abs(vn).sum()), "share": float(vn[0] / np.abs(vn).sum()), "path": path[::20]}
        z, v = zn, vn
    return {"status": "none", "s": float(np.abs(v).sum()), "path": path[::20]}


out = []
for c in A_["classes"]:
    rows = [r for r in A_["rows"] if r["ok"] and r["kind"] == c["class"][0]
            and abs(r["G_lo"] - c["G_lo"]) < 1e-9]
    r = rows[0]
    z = shift(np.array(r["z"]), v0, [-k for k in r["k"]])
    z, ok, _ = W.newton(z, v0, x, y)
    kind = c["class"][0]
    rec = {"class": c["class"], "n_at_s0": c["n"], "placed_at_s0": c["placed"]}
    q = W.qof(z, v0)
    rec["dLds_s0"] = float(W.grad(q, x, y)[W.VI] @ d)
    st, ssw, zsw, path = W.continue_ray(z, s0, d, x, y, ac, 30.0)
    st2, ssw2, _, path2 = W.continue_ray(z, s0, d, x, y, ac, 30.0, h=0.005)
    rec.update({"ray_status": st, "s_sw_ray": ssw, "s_sw_ray_half": ssw2, "min_lam_path": min(p[1] for p in path),
                "n_path": len(path), "lam_min_s0": path[0][1]})
    if st == "switch":
        vs = ssw * d
        zsw, _, _ = W.newton(zsw, vs, x, y)
        rec["kind_at_switch"] = W.canon(zsw, vs)[0]
        drop = W.antisym_basis(zsw, vs) if rec["kind_at_switch"] == "duplicate" else None
        ev = np.linalg.eigvalsh(W.Hz(zsw, vs, x, y))
        rec["eig_switch"] = [float(e) for e in ev]
        if drop is not None:
            Hs = W.Hz(zsw, vs, x, y)
            rec["antisym_eigs"] = [float(e) for e in np.linalg.eigvalsh(drop.T @ Hs @ drop)]
        kap = {}
        for ksum in range(-3, 4):
            kk = (ksum, 0)
            zk = shift(zsw, vs, kk)
            k_red, lam_red, _, dgv = W.kappa(zk, vs, d, x, y, ac, drop=drop)
            k_all, lam_all, _, _ = W.kappa(zk, vs, d, x, y, ac)
            kap[str(ksum)] = {"kappa_reduced": k_red, "lam_reduced": lam_red, "kappa_full": k_all, "lam_full": lam_all, "dG_dv_term": dgv}
        rec["kappa_by_ksum"] = kap
        # check κ depends on k1 + k2 only
        k_a = W.kappa(shift(zsw, vs, (1, 0)), vs, d, x, y, ac, drop=drop)[0]  # noqa
        k_b = W.kappa(shift(zsw, vs, (0, 1)), vs, d, x, y, ac, drop=drop)[0]
        rec["kappa_(1,0)_vs_(0,1)"] = [k_a, k_b]
    rec["adiabatic"] = adiabatic(z.copy(), v0.copy())
    out.append(rec)
    print(json.dumps({kk: vv for kk, vv in rec.items() if kk not in ("adiabatic",)}, default=float), flush=True)
    print("  adiabatic:", rec["adiabatic"]["status"], rec["adiabatic"].get("s"), rec["adiabatic"].get("share"), flush=True)
json.dump(out, open(__file__.replace(".py", ".json" if "W2A" not in os.environ else "_" + os.path.basename(os.environ["W2A"]).split("_a_")[-1]), "w"), indent=1, default=float)
