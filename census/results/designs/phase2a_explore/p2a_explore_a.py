"""EXPLORATORY (population landscape only; no training, no seed).  Along the linear-dominant branch M in TRAINING
coordinates, with P = diag(1 on W, c, b; ρ on v):
  - the driving rate σ = ṡ/(ρη) = Σ_k sign(v_k)(−∂L/∂v_k) at the branch point;
  - λ_min(ρ) of P^{1/2}HP^{1/2} on the fixed-s tangent space (active units; Σ sign·δv = 0), and the v-share of its eigenvector;
  - χ(ρ) = (ṡ/s)/(ηλ_min) = ρσ/(sλ_min(ρ));
  - the idle unit's free-training block [[λ I, √ρ g],[√ρ gᵀ, 0]], g = ∂²L/∂W_idle∂v_idle (its negative eigenvalue);
  - λ_max of P^{1/2}HP^{1/2} (step-size bound);
  - the fold constant |m'c'|_ρ = ¼|d(λ_min²)/ds| at s_F from the last 0.05 (and 0.025) of the Δs = 0.001 grid,
    Λ_F = (|m'c'|s_F)^{1/2}, ε_F = (ṡ/s_F)/(ηΛ_F) = ρσ_F/(s_FΛ_F), predicted delay r_F = Ω₀ε_F^{2/3}.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from results.designs.phase2a_explore import core as C

RHOS = [1.0] + [2.0 ** -k for k in (4, 6, 8, 10, 12, 14)]
OUT = "results/designs/phase2a_explore/p2a_explore_a.json"


def tangent_basis(th):
    act = [k for k in range(4) if abs(th[12 + k]) > 1e-12]
    idx = [2 * k for k in act] + [2 * k + 1 for k in act] + [8 + k for k in act] + [12 + k for k in act] + [16]
    nv = len(act)
    sg = np.sign(th[[12 + k for k in act]])
    m = len(idx)
    B = np.eye(m)
    # constraint row: Σ sign·δv = 0 on the v block (last nv+1 .. -1)
    a = np.zeros(m); a[3 * nv:4 * nv] = sg
    a /= np.linalg.norm(a)
    Bq = B - np.outer(a, a)
    U, S, _ = np.linalg.svd(Bq)
    Qb = U[:, :m - 1]
    return idx, act, Qb, 3 * nv, 4 * nv


def at(th, rho):
    H = C.hessian(th)
    idx, act, Qb, v0, v1 = tangent_basis(th)
    Ha = H[np.ix_(idx, idx)]
    p = np.ones(len(idx)); p[v0:v1] = rho
    ph = np.sqrt(p)
    M = Qb.T @ ((ph[:, None] * Ha) * ph[None, :]) @ Qb
    w, V = np.linalg.eigh(M)
    vec = Qb @ V[:, 0]
    vshare = float((vec[v0:v1] ** 2).sum())
    Hfull = (np.sqrt(np.r_[np.ones(12), np.full(4, rho), 1.0])[:, None] * H) * np.sqrt(np.r_[np.ones(12), np.full(4, rho), 1.0])[None, :]
    lmax = float(np.linalg.eigvalsh(Hfull)[-1])
    return float(w[0]), vshare, lmax, H


def main():
    res = {"label": "EXPLORATORY: population landscape, branch M, training coordinates; no training, no seed", "rows": []}
    for s in (1.5, 1.7957, 2.5, 3.0, C.S_STAR, 4.0, 4.5, 4.7, 4.76):
        th = C.branch_theta("M", s)
        L, g = C.loss_grad(th)
        sg = np.sign(th[12:16])
        act = [k for k in range(4) if abs(th[12 + k]) > 1e-12]
        idle = [k for k in range(4) if k not in act]
        sigma = float(sum(sg[k] * -g[12 + k] for k in act))
        mu = [float(sg[k] * -g[12 + k]) for k in act]
        row = {"s": s, "loss": L, "sigma": sigma, "mu_per_unit": mu, "rho2": C.rho2(th), "gplus": C.gplus(th),
               "grad_hidden_max": float(np.abs(g[:12]).max()), "per_rho": {}}
        for rho in RHOS:
            lmin, vsh, lmax, H = at(th, rho)
            k = idle[0]
            gx = H[[2 * k, 2 * k + 1, 8 + k], 12 + k]
            blk = np.zeros((4, 4)); blk[:3, :3] = H[np.ix_([2 * k, 2 * k + 1, 8 + k], [2 * k, 2 * k + 1, 8 + k])]
            blk[:3, 3] = blk[3, :3] = np.sqrt(rho) * gx; blk[3, 3] = rho * H[12 + k, 12 + k]
            row["per_rho"][f"{rho:.3g}"] = {"lam_min": lmin, "slow_vec_v_share": vsh, "lam_max_P": lmax,
                                            "chi_over_eta_free": rho * sigma / (s * lmin),
                                            "idle_coupling_g": [float(x) for x in gx],
                                            "idle_block_min_eig": float(np.linalg.eigvalsh(blk)[0])}
        res["rows"].append(row)
        print(json.dumps({"s": s, "sigma": sigma, "rho2": row["rho2"], "gx": row["per_rho"]["1"]["idle_coupling_g"],
                          **{r: (round(v["lam_min"], 9), round(v["slow_vec_v_share"], 3), round(v["chi_over_eta_free"], 4), v["idle_block_min_eig"])
                             for r, v in row["per_rho"].items()}}), flush=True)
    # fold constants per ρ
    z = np.load("results/sb_fold/grid_M.npz")
    sel = z["s"] >= C.S_FOLD - 0.05
    fold = {}
    for rho in RHOS:
        lam = []
        for s in z["s"][sel]:
            th = C.branch_theta("M", float(s))
            lam.append(at(th, rho)[0])
        lam = np.array(lam); ss = z["s"][sel]
        out = {}
        for w in (0.05, 0.025):
            m = ss >= C.S_FOLD - w
            d = C.S_FOLD - ss[m]
            coef, *_ = np.linalg.lstsq(np.vstack([d, d ** 2]).T, lam[m] ** 2, rcond=None)
            out[f"abs_mc_w{w}"] = float(abs(coef[0]) / 4)
        thF = C.branch_theta("M", float(ss[-1]))
        _, gF = C.loss_grad(thF)
        sigF = float(sum(np.sign(thF[12 + k]) * -gF[12 + k] for k in range(4) if abs(thF[12 + k]) > 1e-12))
        LamF = (out["abs_mc_w0.05"] * C.S_FOLD) ** 0.5
        eps = rho * sigF / (C.S_FOLD * LamF)
        out.update({"sigma_F": sigF, "Lambda_F": LamF, "eps_F": eps, "r_F_pred": C.OMEGA0 * eps ** (2 / 3),
                    "s_cross_pred": C.S_FOLD * (1 + C.OMEGA0 * eps ** (2 / 3))})
        fold[f"{rho:.3g}"] = out
        print(json.dumps({"rho": rho, **out}), flush=True)
    res["fold"] = fold
    open(OUT, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
