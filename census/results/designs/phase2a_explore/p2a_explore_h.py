"""EXPLORATORY per-seed-sample survey for the per-seed fold design (NOT registered; 2026-10-04).
Design draft: results/designs/phase2a_perseed_design.md.  Exploration seeds 2,930,000-2,930,199 ONLY (never to be
registered); no registered or pilot seed drawn.  One process, nice 15, one thread, wall-clock capped.

Usage: python p2a_explore_h.py MODE SEED [SEED ...]     MODE = grid | iid

  sample   grid: p2a_explore_g.sample (the benchmark's product grid with i.i.d. levels, default_rng([seed, 7]));
           iid : 400 i.i.d. points per class from the generating distribution (default_rng([seed, 11])): class 1
                 x1 ~ 0.8 U[0.1, 1] + 0.2 U[-0.1, 0.1], x2 ~ +/-U[h + 0.2, 1]; class 0 the mirror in x1, x2 ~ U[-h, h];
                 rank-matched to the grid order (sort by class-oriented x1, 20 blocks of 20, each sorted by x2) so the
                 data homotopy pairs each point with a grid point.
  freeze   M by the data homotopy at fixed s ~ 4.49 (Newton at lambda = 1/n ... 1, n = 50, then n = 200 if 50 fails),
           identity record (Newton converged at every lambda, reduced lambda_min > 0 at every lambda, function-space
           distance of the final point to the population M point on the seed's X), continuation both ways (h 0.05,
           h 0.025 for the fold), fold validation (30 perturbed minima at 0.99/0.999/1.001/1.01 s_F: count on M, rho2,
           fraction with rho2 >= q), |m'c'| (P = I tangent, last 0.05 below s_F), S by the same homotopy, s* (M = S
           in loss), hold candidates: s0 = lo + 0.1776 (s_F - lo) [the fixed dataset's s0 = 1.7957 as a fraction of
           its M interval 1.154-4.7677] and s0 = s*/2 [the fixed dataset's s0/s* = 0.500]; landing of the seed's own
           init and of 10 further exploration inits (init_row(2,930,150 + j)) on the seed's M at the first rule's s0.
  run      scale-only slow SGD at rho = 2^-13 from the seed's exact M(s0) (first rule) to the crossing or
           1.25 s_F(seed); rho2 every 20 steps, refined to the step; the registered causal fold forecaster
           (causal_forecast_fold.run_forecast, f = 0.95) with the seed's own (s_F, Lambda_F), and the same forecaster
           at the same cutoff with the FIXED-dataset constants (s_F = 4.7677, Lambda_F = 1.50995e-3).
"""
import json, math, sys, time
import numpy as np
sys.path.insert(0, ".")
from results.designs.phase2a_explore import core as C
from results.designs.phase2a_explore import p2a_explore_g as G
from results.designs.phase2a_explore.p2a_explore_b import init_row
from src import simplicity_bias as sb
from src import simplicity_bias_v2 as v2
from src import sb_fold as F
from src import causal_forecast_fold as CFF

OUT = "results/designs/phase2a_explore/p2a_explore_h.jsonl"
CAP = 40 * 60
T0 = time.time()
import os
LOG2RHO = float(os.environ.get("P2AH_LOG2RHO", "-13"))   # the rate of the run part (default 2^-13)
SF_POP, LAMF_POP, LO_POP = C.S_FOLD, 0.001509953872657247, 1.154
FRAC0 = (1.7957 - LO_POP) / (SF_POP - LO_POP)


def emit(rec):
    rec = {"label": "EXPLORATORY", **rec}
    open(OUT, "a").write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)


def sample_iid(seed):
    rng = np.random.default_rng([seed, 11])
    p, n = sb.PILOT_P, sb.N1 * sb.N2
    w = (2 - 2 * sb.GAP) / 3; h = w / 2

    def x1():
        noise = rng.random(n) < p
        return np.where(noise, rng.uniform(-0.1, 0.1, n), rng.uniform(0.1, 1.0, n))
    a1 = x1(); b1 = rng.uniform(h + sb.GAP, 1.0, n) * np.where(rng.random(n) < 0.5, -1.0, 1.0)
    a0 = x1(); b0 = rng.uniform(-h, h, n)

    def order(a, b):                          # a = class-oriented x1 (ascending as the grid's l1), then x2 in blocks
        i = np.argsort(a, kind="stable"); a, b = a[i], b[i]
        out = []
        for k in range(sb.N1):
            j = np.argsort(b[k * sb.N2:(k + 1) * sb.N2], kind="stable") + k * sb.N2
            out.append(np.c_[a[j], b[j]])
        return np.vstack(out)
    X1 = order(a1, b1); X0 = order(a0, b0); X0[:, 0] = -X0[:, 0]
    return np.vstack([X0, X1]), np.r_[np.zeros(n), np.ones(n)]


def find_branch(name, X, Y, n_steps, h=0.05, s_start=4.5):
    xs, ev, Rp = F.load_branch(name)
    k = int(np.argmin(np.abs(xs[:, -1] - s_start)))
    s = xs[k, -1]; u0 = xs[k, :-1].copy(); min_eig = math.inf
    for lt in np.linspace(1.0 / n_steps, 1.0, n_steps):
        Rt = F.Reduced(Rp.nA, (1 - lt) * C.X + lt * X, Y)
        u1, ok, res = F.newton_fixed_s(Rt.F, Rt.JF, u0, s)
        if not ok:
            return None, {"newton_ok": False, "lambda_failed": float(lt), "n_steps": n_steps}
        J = Rt.JF(u1, s)[:, :-1]; min_eig = min(min_eig, float(np.linalg.eigvalsh(0.5 * (J + J.T))[0]))
        u0 = u1
    R = F.Reduced(Rp.nA, X, Y)
    Pseed = R.to_full(u0)[0]; Ppop = Rp.to_full(xs[k, :-1])[0]
    info = {"newton_ok": True, "n_steps": n_steps, "s_start": float(s), "homotopy_min_eig": min_eig,
            "fdist_to_pop_point": float(F.fdist(Pseed[None], Ppop[None], X)[0])}
    x0 = np.r_[u0, s]
    pf, ff = F.continuation(R.F, R.JF, x0, +1, h, n_max=int(40 / h))
    pb, fb = F.continuation(R.F, R.JF, x0, -1, h, n_max=int(40 / h))
    info["fold_up"] = ff[0]["s_fold"] if ff else None
    info["fold_down"] = fb[0]["s_fold"] if fb else None
    pts = pb[::-1] + pf[1:]
    if not any(p["eig_min"] > 0 for p in pts):
        info["no_stable_points"] = True
        return None, info
    return G.Branch(R, pts), info


def trace(name, X, Y, h=0.05):
    B, info = find_branch(name, X, Y, 50, h)
    if B is None:
        B, info2 = find_branch(name, X, Y, 200, h); info = {"first": info, **info2}
    return B, info


def rho2(th, X):
    return float(sb.feature_usage(C.to_landscape(th), X)["rho2"])


def train(th, rho, X, Y, s_F, LamF, s_star, q=C.Q, REC=20):
    import torch
    act = [k for k in range(4) if th[12 + k] != 0.0]
    S = [np.abs(th[12:16]).sum()]; rows = [th.copy()]; t = 0
    a = np.zeros(4); a[act] = np.sign(th[12:16][act]); a /= np.linalg.norm(a)

    def step(th):
        g = G.loss_grad(th, X, Y)
        new = th.copy(); new[:12] -= g[:12]; new[16] -= g[16]
        gv = g[12:16]; new[12:16] -= (gv - a * (a @ gv)) + rho * a * (a @ gv)
        return new
    t0 = time.time(); found = None; r_all = []
    while S[-1] < 1.25 * s_F and t < int(64 / rho):
        th = step(th); t += 1; S.append(np.abs(th[12:16]).sum())
        if t % REC == 0:
            rows.append(th.copy())
    S = np.array(S)
    r2 = v2.rho2_batch(np.array(rows), X, torch, chunk=100)
    tc = v2.crossing_step(r2, q)
    rec = {"steps": t, "sec": time.time() - t0, "rho2_end": float(r2[-1]), "signs_fixed": bool(np.all(np.sign(th[12:16][act]) == a[act] / abs(a[act]))) }
    t_obs = None
    if tc is not None:
        th2 = rows[tc - 1].copy(); seg = [th2.copy()]
        for _ in range(REC):
            th2 = step(th2); seg.append(th2.copy())
        j = int(np.argmax(v2.rho2_batch(np.array(seg), X, torch) >= q))
        t_obs = (tc - 1) * REC + j
        rec.update({"t_obs": t_obs, "s_obs": float(S[t_obs]), "s_obs_over_sF": float(S[t_obs] / s_F),
                    "in_F_window": bool(s_F <= S[t_obs] <= 1.25 * s_F),
                    "falsifier": bool(s_star is not None and S[t_obs] <= 1.25 * s_star)})
    else:
        rec["crossing"] = None
    t_c = int(np.argmax(S >= 0.95 * s_F)) if S.max() >= 0.95 * s_F else None
    rec["t_c"] = t_c
    if t_c is None or t_obs is None:
        return rec
    t_F = int(np.argmax(S >= s_F)); rec["t_F"] = t_F; rec["r_obs"] = float(S[t_obs] / s_F - 1)
    for tag, sF, L in (("own", s_F, LamF), ("pop", SF_POP, LAMF_POP)):
        fc, same = CFF.run_forecast(S, t_c, sF, L, 1.0, int(math.ceil(64 / rho)) - t_c, 0.95)
        out = {"status": fc["status"], "nan_identical": same, "s_c_fc": fc["s_c_fc"], "eps_fc": fc["eps_fc"],
               "r_fc": fc["r_fc"], "t_fc": fc["t_fc"], "t_F_fc": fc["t_F_fc"]}
        if fc["s_c_fc"] is not None:
            out["abs_log_scale_err"] = abs(math.log(fc["s_c_fc"] / S[t_obs]))
        if CFF.has_forecast(fc):
            out["abs_time_err"] = abs(fc["t_fc"] - t_obs); out["delay_fc"] = fc["delay_fc"]
            out["C1_rel"] = abs(fc["t_fc"] - t_obs) / fc["delay_fc"] if fc["delay_fc"] > 0 else None
            out["C2_rel"] = abs(fc["delay_fc"] - (t_obs - t_F)) / fc["delay_fc"] if fc["delay_fc"] > 0 else None
            out["C4_D"] = abs(fc["t_fc"] - t_obs) - abs(fc["t_F_fc"] - t_obs)
            out["ratio_r"] = rec["r_obs"] / fc["r_fc"]
        rec["fc_" + tag] = out
    rec["delay_obs"] = t_obs - t_F
    return rec


def part(mode, seed):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    X, Y = (G.sample if mode == "grid" else sample_iid)(seed)
    rec = {"mode": mode, "seed": seed}
    try:
        M, iM = trace("M", X, Y)
    except Exception as e:                                  # noqa: BLE001 (exploratory: record and go on)
        M, iM = None, {"error": repr(e)}
    rec["M"] = iM
    if M is None or iM.get("fold_up") is None:
        rec["traceable"] = False; rec["sec_freeze"] = time.time() - t0; emit(rec); return
    s_F = iM["fold_up"]; rec["traceable"] = True
    rec["M_stable"] = [float(M.lo), float(M.hi)]
    rec["rho2_M_max"] = max(rho2(M.theta(s), X) for s in np.linspace(M.lo + 1e-3, 0.9999 * s_F, 12))
    try:
        _, i2 = find_branch("M", X, Y, iM["n_steps"], h=0.025); rec["fold_h0.025"] = i2["fold_up"]
    except Exception as e:                                  # noqa: BLE001
        rec["fold_h0.025"] = repr(e)
    rng = np.random.default_rng(0); val = {}
    for tag, s in (("below_1pct", 0.99 * s_F), ("below_0.1pct", 0.999 * s_F), ("above_0.1pct", 1.001 * s_F),
                   ("above_1pct", 1.01 * s_F)):
        Pb = F.run_to_full(M.theta(min(s, 0.9999 * s_F))[None])[0]
        P0 = Pb[None] + rng.normal(0, 1, (30, 16)) * (0.05 * np.abs(Pb) + 0.02); P0[:, 12:] = np.abs(P0[:, 12:])
        P, L, gn = F.local_min_batch(P0, np.full(30, s), X, Y)
        d = F.fdist(P, Pb[None], X); r = np.array([sb.feature_usage(p, X)["rho2"] for p in P])
        val[tag] = {"n_on_M": int((d <= 1e-3).sum()), "rho2_med": float(np.median(r)), "n_rho2_ge_q": int((r >= C.Q).sum())}
    rec["validation"] = val
    ss = np.arange(s_F - 0.05, s_F, 0.001); ss = ss[ss < s_F - 1e-6]
    lam = np.array([G.tangent_lmin(M.theta(s), X, Y) for s in ss])
    d = s_F - ss; coef, *_ = np.linalg.lstsq(np.vstack([d, d ** 2]).T, lam ** 2, rcond=None)
    mc = float(abs(coef[0]) / 4); LamF = (mc * s_F) ** 0.5
    rec.update({"abs_mc": mc, "Lambda_F": LamF})
    s_star = None
    try:
        S_, iS = trace("S", X, Y)
    except Exception as e:                                  # noqa: BLE001
        S_, iS = None, {"error": repr(e)}
    rec["S"] = {k: iS.get(k) for k in ("newton_ok", "n_steps", "fold_up", "fold_down", "homotopy_min_eig")}
    if S_ is not None:
        lo, hi = max(M.lo, S_.lo) + 1e-3, min(M.hi, S_.hi) - 1e-3
        if lo < hi:
            dl = M.loss(lo) - S_.loss(lo); dh = M.loss(hi) - S_.loss(hi)
            if dl * dh < 0:
                while hi - lo > 1e-8:
                    mid = 0.5 * (lo + hi); dm = M.loss(mid) - S_.loss(mid)
                    if dm * dl > 0: lo, dl = mid, dm
                    else: hi = mid
                s_star = 0.5 * (lo + hi)
        if S_.lo <= 1.01 * s_F <= S_.hi:
            rec["rho2_S_at_1.01sF"] = rho2(S_.theta(1.01 * s_F), X)
    rec["s_star"] = s_star
    rec["sF_over_sstar"] = s_F / s_star if s_star else None
    s0a = M.lo + FRAC0 * (s_F - M.lo); s0b = s_star / 2 if s_star else None
    rec["s0_frac"] = s0a; rec["s0_half_sstar"] = s0b
    rec["s0_1.7957_on_M"] = bool(M.lo < 1.7957 < M.hi)
    rec["s0_half_sstar_on_M"] = bool(s0b is not None and M.lo < s0b < M.hi)
    thM = M.theta(s0a); PM = F.run_to_full(thM[None])
    rows = np.array([init_row(seed)] + [init_row(2930150 + j) for j in range(10)])
    rows[:, 12:16] *= s0a / np.abs(rows[:, 12:16]).sum(axis=1, keepdims=True)
    P, L, gn = F.local_min_batch(F.run_to_full(rows), np.full(len(rows), s0a), X, Y)
    dd = F.fdist(P, PM, X)
    rec["landing"] = {"own_init_on_M": bool(dd[0] <= 1e-3), "n_on_M_of_11": int((dd <= 1e-3).sum())}
    rec["sec_freeze"] = time.time() - t0
    emit(rec)
    if time.time() - T0 > CAP or os.environ.get("P2AH_NORUN"):
        emit({"mode": mode, "seed": seed, "run": "skipped: wall-clock cap or freeze only"}); return
    run = train(thM, 2.0 ** LOG2RHO, X, Y, s_F, LamF, s_star)
    emit({"mode": mode, "seed": seed, "part": "run", "log2rho": LOG2RHO, "s0": s0a, "s_F": s_F, **run})


if __name__ == "__main__":
    mode = sys.argv[1]
    for sd in sys.argv[2:]:
        sd = int(sd)
        assert 2930000 <= sd <= 2930199
        if time.time() - T0 > CAP:
            emit({"mode": mode, "seed": sd, "skipped": "wall-clock cap"}); continue
        part(mode, sd)
