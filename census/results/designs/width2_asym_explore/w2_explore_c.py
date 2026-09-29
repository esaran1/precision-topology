"""EXPLORATORY (population only; no candidate seed).  Free full-batch GD ("SGD" in the registered sense: torch SGD without
momentum on the full batch) released from each branch class's point at v₀ (w2_explore_b), hidden lr η, output lr ρη.
Usage: python w2_explore_c.py ETA RHO [BUDGET] [CLASS_INDEX ...]
Per run: the crossing (first step with G₊ > 0: dense screen, exact extrema to confirm), s_obs; the lag-free switch along
the run's OWN v path (z*(v_t) by warm-started Newton every 25 steps, then step bisection), s_sw,path; the ray switch from
w2_explore_b; r_obs vs both; the slaved prediction κχ at t_sw (χ = (ṡ/s_sw)/(η·λ_min), ṡ over min(100, t) steps, κ and
λ_min on the reduced Hessian at z*(v_{t_sw}) along the run's local v direction); max χ_t over s_t ≥ 0.8·s_sw (sampled every
25 steps); ηλ_max; the share v₁/s at the crossing; the release-to-crossing step count.  Prints one JSON line per run (run_all.sh collects them in w2_explore_c.log)."""
import json
import math
import sys
import time

import numpy as np

import w2core as W
from src.width2_geometry import phi

W.nice()
x, y = W.setup(); ac = W.act()
from src import width2_geometry as G
import os
A_ = json.load(open(os.environ.get("W2A", __file__.replace("_c.py", "_a.json"))))
B_ = json.load(open(os.environ.get("W2B", __file__.replace("_c.py", "_b.json"))))
s0, v0 = A_["s0"], np.array(A_["v0"])
ETA, RHO = float(sys.argv[1]), float(sys.argv[2])
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
IDX = [int(a) for a in sys.argv[4:]] or list(range(len(B_)))


def placed(z, v):
    th = np.array([z[0], z[1], z[2], z[3]])
    if float(phi(G._XO, th, v, ac).min() - phi(G._XI, th, v, ac).max()) <= 0:
        return False
    return W.gap(z, v, ac)[0] > 0


def zstar_placed(zg, v):
    zn, ok, _ = W.newton(zg, v, x, y)
    return zn, ok, (W.gapm(zn, v, ac) > 0) if ok else None


for ci in IDX:
    rec_b = B_[ci]
    r = next(r for r in A_["rows"] if r["ok"] and r["kind"] == rec_b["class"][0] and abs(r["G_lo"] - A_["classes"][ci]["G_lo"]) < 1e-9)
    # the class's canonical-winding point (as in w2_explore_b)
    zc = np.array(r["z"]); k = r["k"]
    zc[1] -= 2 * math.pi * k[0]; zc[3] -= 2 * math.pi * k[1]; zc[4] += 2 * math.pi * (k[0] * v0[0] + k[1] * v0[1])
    zc, ok, _ = W.newton(zc, v0, x, y)
    q = W.qof(zc, v0)
    t0 = time.time()
    V, S = [v0.copy()], [s0]
    cross = None
    lam_max = 0.0
    for t in range(1, BUDGET + 1):
        g = W.grad(q, x, y)
        q[W.ZI] -= ETA * g[W.ZI]; q[W.VI] -= RHO * ETA * g[W.VI]
        V.append(q[W.VI].copy()); S.append(float(np.abs(q[W.VI]).sum()))
        if placed(q[W.ZI], q[W.VI]):
            cross = t
            break
    out = {"class": ci, "kind": rec_b["class"][0], "eta": ETA, "rho": RHO, "budget": BUDGET, "crossed": cross is not None,
           "secs_train": round(time.time() - t0, 1)}
    if cross is None:
        print(json.dumps(out), flush=True); continue
    s_obs = S[cross]
    out.update({"cross_step": cross, "s_obs": s_obs, "share_at_cross": float(q[2] / S[cross]),
                "s_sw_ray": rec_b.get("s_sw_ray"), "s_sw_adiabatic": rec_b["adiabatic"].get("s")})
    # lag-free switch along the run's own v path
    zg, tprev, zprev, tsw = zc.copy(), 0, zc.copy(), None
    samples = []
    for t in list(range(25, len(V), 25)) + [len(V) - 1]:
        zn, okn, pl = zstar_placed(zg, V[t])
        if not okn:
            out["path_branch"] = f"lost at step {t}"; break
        lr_, la_, lx_ = W.lam_reduced(zn, V[t], x, y)
        dup = la_ is not None
        if t + 1 < len(S):
            chi_t = ((S[t + 1] - S[t]) / S[t]) / (ETA * lr_)
            samples.append((S[t], chi_t, lx_, la_))
        if pl:
            lo, hi, zl = tprev, t, zprev
            while hi - lo > 1:
                mid = (lo + hi) // 2
                zm, okm, plm = zstar_placed(zl, V[mid])
                if plm:
                    hi = mid
                else:
                    lo, zl = mid, zm
            tsw = hi
            break
        zg, tprev, zprev = zn, t, zn
    if tsw is not None:
        s_sw = S[tsw]
        zsw, _, _ = W.newton(zg, V[tsw], x, y)
        w = min(100, tsw)
        sdot = (S[tsw] - S[tsw - w]) / w
        dv = (V[tsw] - V[tsw - w]); dv = dv / (np.sign(V[tsw]) @ dv)
        dup = W.canon(zsw, V[tsw])[0] == "duplicate"
        drop = W.antisym_basis(zsw, V[tsw]) if dup else None
        kap, lam, _, _dgv = W.kappa(zsw, V[tsw], dv, x, y, ac, drop=drop)
        chi = (sdot / s_sw) / (ETA * lam)
        win = [c for (ss, c, _, _) in samples if ss >= 0.8 * s_sw]
        out.update({"t_sw_path": tsw, "s_sw_path": s_sw, "r_obs_path": s_obs / s_sw - 1, "kappa": kap, "lam_min_sw": lam,
                    "chi_sw": chi, "kchi": kap * chi, "obs_over_kchi": (s_obs / s_sw - 1) / (kap * chi),
                    "lag_steps_obs": cross - tsw, "lag_steps_slaved": kap / (ETA * lam),
                    "max_chi_window": max(win) if win else None, "max_chi_all": max((c for (_, c, _, _) in samples), default=None),
                    "eta_lam_max": ETA * max((lm for (_, _, lm, _) in samples), default=float("nan")),
                    "min_antisym_lam": min((a for (_, _, _, a) in samples if a is not None), default=None), "kind_at_switch": "duplicate" if dup else "two-unit",
                    "r_obs_ray": (s_obs / rec_b["s_sw_ray"] - 1) if rec_b.get("s_sw_ray") else None})
    print(json.dumps(out, default=float), flush=True)
