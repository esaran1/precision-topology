"""EXPLORATORY timing and forecast-horizon check for the revised Phase 2A design (NOT registered; 2026-10-04).
Usage: `python p2a_explore_g.py pop L2 …` and `python p2a_explore_g.py SEED L2` (SEED in 2,930,000-2,930,199).
Uses only the population landscape and the NON-registered exploration seeds 2,930,000-2,930,199 (never to be
registered).  Two parts, one process, nice 15, one thread, wall-clock capped:

  seed SEED [LOG2RHO]   per-seed own-sample freeze + one training run (the cost of the per-seed-sample design):
      sample  = the benchmark generator with i.i.d. levels in place of quantile levels (per class a 20 × 20 product grid
                as in simplicity_bias.make_data; x₁ class-1 levels: exactly 4 from U[−0.1, 0.1] and 16 from U[0.1, 1]
                (the generator's mixture proportions p = 0.2, stratified), class 0 the mirror; x₂: 20 middle-slab levels
                U[−h, h], 10 outer levels U[h + 0.2, 1] mirrored to ±); drawn from numpy default_rng([SEED, 7]).
      freeze  = M and S found by a data homotopy at fixed s ≈ 4.49 (X(λ) = (1 − λ)X_pop + λX_seed, Newton at λ = 0.02 …
                1 from the population branch point), pseudo-arclength continuation (sb_fold) of M to its fold
                (h = 0.05 and 0.025, agreement), fold validation (30 perturbed local minima at 0.99/0.999/1.001/1.01·s_F),
                |m′c′| in training coordinates (P = I on the fixed-s tangent; last 0.05 of a Δs = 0.001 grid), s* (equal
                loss of M and S), ρ₂ of M at s_F and of S at 1.01·s_F, the hold of the seed's init (fixed-s BFGS) at
                s₀ = 1.7957 if that lies on the seed's M, else at 1.05 × M's lower end, and its distance to the seed's M.
      run     = scale-only slow SGD (η = 1) from the seed's exact M(s₀) (idle unit exactly 0) to 1.25·s_F(seed);
                ρ₂ every 20 steps (crossing refined to the step); causal fold forecast at f = 0.90, 0.95.
  (the first attempt, Newton directly at s = 3.0 / adaptive λ steps, failed or jumped branches: see the log)
  pop LOG2RHO ...       population runs as p2a_explore_f (same forecaster) at further rates.
"""
import json, math, sys, time
import numpy as np
sys.path.insert(0, ".")
from results.designs.phase2a_explore import core as C
from results.designs.phase2a_explore.p2a_explore_b import init_row
from src import simplicity_bias as sb
from src import simplicity_bias_v2 as v2
from src import sb_fold as F
from src import causal_forecast as CF

REC = 20
S0 = 1.7957
OUT = "results/designs/phase2a_explore/p2a_explore_g.jsonl"
T_START = time.time()
CAP = 27 * 60                                   # seconds of wall clock for the whole process


def emit(rec):
    rec = {"label": "EXPLORATORY", **rec}
    open(OUT, "a").write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, default=float), flush=True)


# ------------------------------------------------------------------------------------------ per-seed sample
def sample(seed):
    rng = np.random.default_rng([seed, 7])
    p, n1, n2 = sb.PILOT_P, sb.N1, sb.N2
    k_noise = int(round(p * n1))
    l1 = np.sort(np.r_[rng.uniform(-0.1, 0.1, k_noise), rng.uniform(0.1, 1.0, n1 - k_noise)])
    w = (2 - 2 * sb.GAP) / 3; h = w / 2
    mid = np.sort(rng.uniform(-h, h, n2))
    upper = np.sort(rng.uniform(h + sb.GAP, 1.0, n2 // 2))
    outer = np.concatenate([-upper[::-1], upper])
    X1 = np.array([(a, b) for a in l1 for b in outer]); X0 = np.array([(-a, b) for a in l1 for b in mid])
    X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    return X, y


def loss_grad(th, X, Y, lam=C.LAM):
    W, c, v, b = C.unpack(th)
    H = np.tanh(X @ W.T + c)
    z = H @ v + b
    r = (0.5 * (1 + np.tanh(0.5 * z)) - Y) / len(Y)
    D = (1 - H ** 2) * (r[:, None] * v[None, :])
    g = np.empty(17)
    g[:8] = (D.T @ X + lam * W).ravel(); g[8:12] = D.sum(0) + lam * c; g[12:16] = H.T @ r; g[16] = r.sum()
    return g


def hessian(th, X, Y, lam=C.LAM):
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float64); Yt = torch.as_tensor(Y, dtype=torch.float64)

    def Lf(q):
        W = q[:8].reshape(4, 2); c = q[8:12]; v = q[12:16]; b = q[16]
        z = torch.tanh(Xt @ W.T + c) @ v + b
        return torch.nn.functional.binary_cross_entropy_with_logits(z, Yt) + 0.5 * lam * ((W ** 2).sum() + (c ** 2).sum())
    return torch.autograd.functional.hessian(Lf, torch.tensor(th, dtype=torch.float64)).numpy()


class Branch:
    """A branch of the seed's own landscape: stored continuation points (stable part) and Newton at any s."""

    def __init__(self, R, pts):
        self.R = R
        xs = np.array([p["x"] for p in pts]); ev = np.array([p["eig_min"] for p in pts])
        self.xs = xs[ev > 0]
        self.lo, self.hi = self.xs[:, -1].min(), self.xs[:, -1].max()

    def u(self, s):
        k = int(np.argmin(np.abs(self.xs[:, -1] - s)))
        u, ok, _ = F.newton_fixed_s(self.R.F, self.R.JF, self.xs[k, :-1], s)
        return u if ok else None

    def theta(self, s):
        u = self.u(s); n = self.R.nA
        W = u[:2 * n].reshape(n, 2); c = u[2 * n:3 * n]; eta = np.r_[1.0, u[3 * n:4 * n - 1]]
        th = np.zeros(17); Wf = np.zeros((4, 2)); Wf[:n] = W
        th[:8] = Wf.ravel(); th[8:8 + n] = c; th[12:12 + n] = s * eta ** 2 / (eta ** 2).sum(); th[16] = u[-1]
        return th

    def loss(self, s):
        return self.R.loss(self.u(s), s)


def find_branch(name, s_start, X, Y, h=0.05):
    xs, ev, Rp = F.load_branch(name)
    k = int(np.argmin(np.abs(xs[:, -1] - s_start)))
    # data homotopy at fixed s: X(λ) = (1 − λ)·X_pop + λ·X_seed (the same product-grid ordering), Newton at
    # λ = 0.02, 0.04, …, 1 (fixed small steps; larger adaptive steps jumped branches), reduced Hessian checked > 0
    u0 = xs[k, :-1].copy(); n_newton = 0; min_eig = math.inf
    for lt in np.linspace(0.02, 1.0, 50):
        Rt = F.Reduced(Rp.nA, (1 - lt) * C.X + lt * X, Y)
        u1, ok, res = F.newton_fixed_s(Rt.F, Rt.JF, u0, xs[k, -1]); n_newton += 1
        if not ok:
            return None, {"newton_ok": False, "res": res, "lambda_failed": float(lt)}
        J = Rt.JF(u1, xs[k, -1])[:, :-1]; min_eig = min(min_eig, float(np.linalg.eigvalsh(0.5 * (J + J.T))[0]))
        u0 = u1
    R = F.Reduced(Rp.nA, X, Y)
    x0 = np.r_[u0, xs[k, -1]]
    out = {"newton_ok": True, "s_start": float(xs[k, -1]), "homotopy_newton_solves": n_newton, "homotopy_min_eig": min_eig}
    pts_f, folds_f = F.continuation(R.F, R.JF, x0, +1, h, n_max=int(40 / h))
    pts_b, folds_b = F.continuation(R.F, R.JF, x0, -1, h, n_max=int(40 / h))
    out["fold_up"] = folds_f[0]["s_fold"] if folds_f else None
    out["fold_down"] = folds_b[0]["s_fold"] if folds_b else None
    out["fold_up_eig"] = folds_f[0]["eig_min"] if folds_f else None
    pts = pts_b[::-1] + pts_f[1:]
    return Branch(R, pts), out


def tangent_lmin(th, X, Y):
    from results.designs.phase2a_explore.p2a_explore_a import tangent_basis
    H = hessian(th, X, Y)
    idx, act, Qb, v0, v1 = tangent_basis(th)
    return float(np.linalg.eigvalsh(Qb.T @ H[np.ix_(idx, idx)] @ Qb)[0])


def forecast(s_vis, s_F, LamF, eta=1.0):
    cfg = CF.ForecastConfig(family="quad", window_frac=0.05, min_window=50, horizon_factor=1.6)
    sh, st, info = CF.extrapolate_scale(s_vis, cfg, s_F, 10 ** 9)
    t_c = len(s_vis)
    kF = int(np.argmax(sh >= s_F))
    sdot = sh[kF] - (sh[kF - 1] if kF > 0 else s_vis[-1])
    eps = (sdot / s_F) / (eta * LamF)
    s_c = s_F * (1 + C.OMEGA0 * eps ** (2 / 3))
    kc = int(np.argmax(sh >= s_c))
    return {"t_F_fc": t_c + kF, "eps_fc": float(eps), "r_fc": float(s_c / s_F - 1), "t_fc": t_c + kc}


def train(th, rho, X, Y, s_F, LamF, q=C.Q):
    import torch
    idle = [k for k in range(4) if th[12 + k] == 0.0]
    act = [k for k in range(4) if k not in idle]
    t0 = time.time()
    cap = int(1.3 * 507720 * 2.0 ** -14 / rho) + 100_000
    S = np.empty(cap); S[0] = np.abs(th[12:16]).sum()
    rows = [th.copy()]; t = 0

    def step(th):
        g = loss_grad(th, X, Y)
        new = th.copy(); new[:12] -= g[:12]; new[16] -= g[16]
        gv = g[12:16]; a = np.zeros(4); a[act] = np.sign(th[12:16][act]); a /= np.linalg.norm(a)
        gp = a * (a @ gv)
        new[12:16] -= (gv - gp) + rho * gp
        return new
    while S[t] < 1.25 * s_F:
        th = step(th); t += 1
        if t >= len(S):
            S = np.resize(S, 2 * len(S))
        S[t] = np.abs(th[12:16]).sum()
        if t % REC == 0:
            rows.append(th.copy())
    S = S[:t + 1]; t_train = time.time() - t0
    t1 = time.time()
    r2 = v2.rho2_batch(np.array(rows), X, torch, chunk=100)
    t_rho = time.time() - t1
    tc = v2.crossing_step(r2, q)
    rec = {"log2rho": math.log2(rho), "steps": t, "sec_train": t_train, "sec_rho2": t_rho, "n_rows": len(rows),
           "us_per_step": 1e6 * t_train / t}
    if tc is None:
        rec["crossing"] = None
        return rec
    th = rows[tc - 1].copy(); seg = [th.copy()]
    for _ in range(REC):
        th = step(th); seg.append(th.copy())
    j = int(np.argmax(v2.rho2_batch(np.array(seg), X, torch) >= q))
    t_obs = (tc - 1) * REC + j; t_F = int(np.argmax(S >= s_F))
    rec.update({"t_obs": t_obs, "s_obs_over_sF": float(S[t_obs] / s_F), "r_obs": float(S[t_obs] / s_F - 1),
                "t_F": t_F, "delay_obs": t_obs - t_F})
    for f in (0.9, 0.95):
        t_c = CF.cutoff_step(S, f * s_F)
        fc = forecast(S[:t_c], s_F, LamF)
        fc.update({"t_c": t_c, "err_cross": fc["t_fc"] - t_obs, "err_tF": fc["t_F_fc"] - t_F,
                   "delay_fc": fc["t_fc"] - fc["t_F_fc"],
                   "err_delay": (fc["t_fc"] - fc["t_F_fc"]) - (t_obs - t_F), "ratio_r": rec["r_obs"] / fc["r_fc"],
                   "horizon_steps": t_obs - t_c, "horizon_delays": (t_obs - t_c) / max(t_obs - t_F, 1)})
        fc["C1_rel"] = abs(fc["err_cross"]) / fc["delay_fc"]; fc["C2_rel"] = abs(fc["err_delay"]) / fc["delay_fc"]
        fc["C4_diff"] = abs(fc["err_cross"]) - abs(fc["t_F_fc"] - t_obs)
        rec[f"f{f}"] = fc
    return rec


def part_seed(seed, log2rho):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    X, Y = sample(seed)
    rec = {"part": "seed", "seed": seed}
    M, iM = find_branch("M", 4.5, X, Y); rec["M"] = iM
    if M is None or iM["fold_up"] is None:
        emit({**rec, "sec_freeze": time.time() - t0}); return
    s_F = iM["fold_up"]
    _, iM2 = find_branch("M", 4.5, X, Y, h=0.025); rec["fold_h0.025"] = iM2["fold_up"]
    t_cont = time.time() - t0
    # fold validation (as sb_fold.run_validate, 30 starts per scale)
    t1 = time.time(); rng = np.random.default_rng(0); val = {}
    for tag, s in (("below_1pct", 0.99 * s_F), ("below_0.1pct", 0.999 * s_F), ("above_0.1pct", 1.001 * s_F), ("above_1pct", 1.01 * s_F)):
        sb_ = min(s, 0.9999 * s_F)
        Pb = F.run_to_full(M.theta(sb_)[None])[0]
        P0 = Pb[None] + rng.normal(0, 1, (30, 16)) * (0.05 * np.abs(Pb) + 0.02); P0[:, 12:] = np.abs(P0[:, 12:])
        P, L, gn = F.local_min_batch(P0, np.full(30, s), X, Y)
        d = F.fdist(P, Pb[None], X)
        val[tag] = {"n_within_1e-3_of_M": int((d <= 1e-3).sum()), "rho2_med": float(np.median([sb.feature_usage(p, X)["rho2"] for p in P]))}
    rec["validation"] = val; t_val = time.time() - t1
    # fold constant in training coordinates
    t1 = time.time()
    ss = np.arange(s_F - 0.05, s_F, 0.001); ss = ss[ss < s_F - 1e-6]
    lam = np.array([tangent_lmin(M.theta(s), X, Y) for s in ss])
    d = s_F - ss; coef, *_ = np.linalg.lstsq(np.vstack([d, d ** 2]).T, lam ** 2, rcond=None)
    mc = float(abs(coef[0]) / 4); LamF = (mc * s_F) ** 0.5
    rec.update({"abs_mc": mc, "Lambda_F": LamF}); t_mc = time.time() - t1
    # S branch, s*, rho2 separation
    t1 = time.time()
    S, iS = find_branch("S", 4.5, X, Y); rec["S"] = iS
    if S is not None:
        lo, hi = max(M.lo, S.lo) + 1e-3, min(M.hi, S.hi) - 1e-3
        dl = M.loss(lo) - S.loss(lo); dh = M.loss(hi) - S.loss(hi)
        if dl * dh < 0:
            while hi - lo > 1e-8:
                mid = 0.5 * (lo + hi); dm = M.loss(mid) - S.loss(mid)
                if dm * dl > 0: lo, dl = mid, dm
                else: hi = mid
            rec["s_star"] = 0.5 * (lo + hi)
        rec["rho2_M_at_fold"] = C_rho2(M.theta(0.9999 * s_F), X)
        rec["rho2_S_at_1.01sF"] = C_rho2(S.theta(1.01 * s_F), X) if S.lo <= 1.01 * s_F <= S.hi else None
        rec["rho2_M_max_on_branch"] = max(C_rho2(M.theta(s), X) for s in np.linspace(M.lo + 1e-3, 0.9999 * s_F, 12))
    rec["M_stable"] = [float(M.lo), float(M.hi)]
    s0 = S0 if M.lo < S0 < M.hi else 1.05 * M.lo
    rec["s0_used"] = s0; rec["s0_1.7957_on_M"] = bool(M.lo < S0 < M.hi)
    t_S = time.time() - t1
    # hold of this seed's init at s0 (fixed-s BFGS) and its distance to the seed's M(s0)
    t1 = time.time()
    row = init_row(seed); row[12:16] *= s0 / np.abs(row[12:16]).sum()
    P, L, gn = F.local_min_batch(F.run_to_full(row[None]), np.array([s0]), X, Y)
    rec["hold_dist_to_M"] = float(F.fdist(P, F.run_to_full(M.theta(s0)[None]), X)[0]); t_hold = time.time() - t1
    rec["sec"] = {"continuation": t_cont, "validation": t_val, "fold_constant": t_mc, "S_and_sstar": t_S, "hold": t_hold,
                  "freeze_total": time.time() - t0}
    emit(rec)
    run = train(M.theta(s0), 2.0 ** log2rho, X, Y, s_F, LamF)
    emit({"part": "seed_run", "seed": seed, "s_F": s_F, **run})


def C_rho2(th, X):
    return float(sb.feature_usage(C.to_landscape(th), X)["rho2"])


def part_pop(l2s):
    import torch
    torch.set_num_threads(1)
    import json as _j
    LamF = _j.load(open("results/designs/phase2a_explore/p2a_explore_a.json"))["fold"]["1"]["Lambda_F"]
    for l2 in l2s:
        est = 136 * 2.0 ** (-15 - l2) * 1.3
        if time.time() - T_START + est > CAP:
            emit({"part": "pop", "log2rho": l2, "skipped": "wall-clock cap", "elapsed": time.time() - T_START}); continue
        rec = train(C.branch_theta("M", S0), 2.0 ** l2, C.X, C.Y, C.S_FOLD, LamF)
        emit({"part": "pop", **rec})


if __name__ == "__main__":
    a = sys.argv
    # all SEED LOG2RHO_SEED [population LOG2RHO ...]  (one process, one wall-clock cap)
    if a[1] == "pop":
        part_pop([float(x) for x in a[2:]])
    else:
        part_seed(int(a[1]), float(a[2]))
