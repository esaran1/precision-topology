"""EXPLORATORY slow full-batch SGD released from the exact branch point M(s0) (population = training set; no seed is
involved: the release state is the Newton-refined branch point, idle unit exactly zero).  η = 1 on (W, c, b).
Output update, two forms:
  plain  v ← v − ρη ∇_v L                         (ρ on the whole output vector: the literal 'output learning rate')
  scale  v ← v − η[(I − ââᵀ) + ρ ââᵀ] ∇_v L,  â = sign(v)/√n over the active units   (ρ on the output SCALE only;
         the shares move at η; a fixed preconditioner while no active v changes sign)
Idle unit: exactly zero (stays zero: its gradients vanish identically), unless IDLE_EPS > 0 (then W, c, v of the idle
unit start at ±IDLE_EPS and its v gets lr η).
Records: s every step; parameters every REC steps; ρ₂ on the stored rows (src.simplicity_bias_v2.rho2_batch); the first
upward passage of q located to the step by re-running the bracketing REC steps; G₊ and the function-space distance to
M(s_t) (and S(s_t)) on a subset; the scale where the distance to M first exceeds 1e-3.

    python p2a_explore_c.py FORM LOG2RHO [S0] [IDLE_EPS] [ETA]
"""
import json, math, sys, time
import numpy as np
sys.path.insert(0, ".")
from results.designs.phase2a_explore import core as C
from src import simplicity_bias_v2 as v2
from src import sb_fold as F

REC = 20


def step_fn(form, rho, eta, idle):
    def step(th):
        L, g = C.loss_grad(th)
        new = th.copy()
        new[:12] -= eta * g[:12]; new[16] -= eta * g[16]
        gv = g[12:16].copy()
        act = [k for k in range(4) if k not in idle]
        if form == "plain":
            new[12:16] -= rho * eta * gv
        else:
            a = np.zeros(4); a[act] = np.sign(th[12:16][act]); a /= np.linalg.norm(a)
            gpar = a * (a @ gv)
            upd = (gv - gpar) + rho * gpar
            new[12:16] -= eta * upd
        return new, L
    return step


def main(form, log2rho, s0=1.7957, idle_eps=0.0, eta=1.0):
    rho = 2.0 ** log2rho
    th = C.branch_theta("M", s0)
    idle = [k for k in range(4) if th[12 + k] == 0.0]
    if idle_eps > 0:
        rng = np.random.default_rng(12345)
        for k in idle:
            th[[2 * k, 2 * k + 1, 8 + k, 12 + k]] = idle_eps * rng.choice([-1.0, 1.0], 4)
        idle = []
    step = step_fn(form, rho, eta, idle)
    s_stop = 7.0
    budget = int(60 / (rho * eta)) + 200_000
    S = [C.scale(th)]; rows = [th.copy()]; t = 0
    t0 = time.time()
    while t < budget and S[-1] < s_stop:
        th, _ = step(th); t += 1
        S.append(C.scale(th))
        if t % REC == 0:
            rows.append(th.copy())
    import torch
    torch.set_num_threads(1)
    rows = np.array(rows)
    rho2 = v2.rho2_batch(rows, C.X, torch)
    tc = v2.crossing_step(rho2, C.Q)
    res = {"label": "EXPLORATORY (population; release at the exact branch point M(s0); no seed)", "form": form,
           "log2rho": log2rho, "rho": rho, "eta": eta, "s0": s0, "idle_eps": idle_eps, "steps": t,
           "s_end": S[-1], "seconds_train": time.time() - t0}
    if tc is not None:
        k0 = (tc - 1) * REC
        th = rows[tc - 1].copy(); seg = [th.copy()]
        for _ in range(REC):
            th, _ = step(th); seg.append(th.copy())
        r2 = v2.rho2_batch(np.array(seg), C.X, torch)
        j = int(np.argmax(r2 >= C.Q))
        t_cross = k0 + j
        res.update({"t_cross": t_cross, "s_cross": S[t_cross], "s_cross_over_fold": S[t_cross] / C.S_FOLD,
                    "s_cross_over_switch": S[t_cross] / C.S_STAR})
    # tracking: function-space distance to M(s) and S(s) on a subset of stored rows
    idx = np.unique(np.r_[np.linspace(0, len(rows) - 1, 120).astype(int)])
    track = []
    for i in idx:
        s = C.scale(rows[i]); d = C.branch_dists(rows[i], ("M", "S"))
        track.append({"t": int(i * REC), "s": s, "rho2": float(rho2[i]), "gplus": C.gplus(rows[i]),
                      "dM": d["M"], "dS": d["S"], "idle_absmax": float(np.abs(rows[i][[k for k in range(17) if k in ()]]).max()) if False else None})
    leaveM = next((r["s"] for r in track if r["dM"] > 1e-3 and r["s"] > s0 * 1.05), None)
    res.update({"s_first_dM_gt_1e-3": leaveM,
                "max_dM_before_0.95_fold": max([r["dM"] for r in track if r["s"] < 0.95 * C.S_FOLD] or [math.nan]),
                "max_dM_before_switch": max([r["dM"] for r in track if r["s"] < C.S_STAR] or [math.nan]),
                "rho2_at_switch": float(np.interp(C.S_STAR, [r["s"] for r in track], [r["rho2"] for r in track])),
                "track": track})
    # observed rate at the fold and the §14 prediction with the scale-only P (Λ_F from p2a_explore_a, ρ = 1 row)
    s_arr = np.array(S)
    kF = int(np.argmax(s_arr >= C.S_FOLD)) if (s_arr >= C.S_FOLD).any() else None
    if kF:
        sdot = (s_arr[kF] - s_arr[kF - 100]) / 100
        res["t_fold"] = kF; res["sdot_at_fold"] = sdot
        A = json.load(open("results/designs/phase2a_explore/p2a_explore_a.json"))
        LamF = A["fold"]["1"]["Lambda_F"] if form == "scale" else A["fold"][f"{rho:.3g}"]["Lambda_F"] if f"{rho:.3g}" in A["fold"] else None
        if LamF:
            eps = (sdot / C.S_FOLD) / (eta * LamF)
            res["eps_F"] = eps; res["r_F_pred"] = C.OMEGA0 * eps ** (2 / 3)
            res["s_cross_pred_fold"] = C.S_FOLD * (1 + res["r_F_pred"])
    tag = f"{form}_r{-log2rho}_s{s0:.4f}" + (f"_idle{idle_eps:g}" if idle_eps else "") + (f"_eta{eta:g}" if eta != 1 else "")
    open(f"results/designs/phase2a_explore/p2a_explore_c_{tag}.json", "w").write(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "track"}, default=float), flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]), float(a[3]) if len(a) > 3 else 1.7957, float(a[4]) if len(a) > 4 else 0.0,
         float(a[5]) if len(a) > 5 else 1.0)
