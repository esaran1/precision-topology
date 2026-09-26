"""Track 1A: the no-fit lag constant κ(a), r = κ(a)·χ  (derived after the fitted relationship was known).

Derivation (math note §13).  Hidden coordinates θ = (w₁, b₁, b₂) (b₂ is a trained coordinate in every protocol) follow
θ ← θ − ηP∇L(θ; s).  θ*(s) is the tracked conditional stationary point (joint Hessian H), with θ*′ = −H⁻¹∂ₛ∇L.  With
θ = θ* + δ and s moving at ṡ per step, the slaved (steady) solution is δ = −(ηPH)⁻¹θ*′ṡ; momentum (β₁ = 0.9) leaves the
steady lag unchanged (m = Hδ in steady state).  G depends on (w₁, b₁) only; g(s) = G(θ*(s)), g′(s*) = ∇G·θ*′.  The crossing
G(θ*(s_c) + δ) = 0 gives g′(s*)(s_c − s*) = ∇G·(ηPH)⁻¹θ*′ṡ, i.e.
    r = (s_c − s*)/s* = κ·χ,   χ = (ṡ/s*)/(ηλ_min),   κ = λ_min·[∇G·(PH)⁻¹θ*′]/[∇G·θ*′],   λ_min = λ_min(P^{1/2}HP^{1/2}).
κ is invariant to rescaling P, so only P's relative shape matters.  P = I for SGD; for Adam P = diag(1/(√v̂ + ε)) at the
crossing, the median (per coordinate, after normalising) over existing runs.

Landscape: width 1, the 800-point population objective, a ∈ {1.30, 1.45, 1.50, 1.60}; the switch s* is the root of
G(θ*(s)) on the branch that is the certified global minimiser (started from the conditional audit's retained minimiser
nearest the certified bracket), found inside the certified glob bracket.  ∇G by central differences of the exact-extrema
gap (step 1e-6), with the one-sided differences agreeing (active pair unique).

    python -m src.lag_law kappa      # κ(a), P from existing runs, committed predictions
    python -m src.lag_law compare    # only after the predictions are committed
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "lag_law"
A_VALUES = (1.30, 1.45, 1.50, 1.60)


def _pop():
    from .width2_conditional import population
    return population()


def _loss_t(z, s, a, X, Y):
    import torch
    t = z[0] * X + z[1]
    return torch.nn.functional.binary_cross_entropy_with_logits(s * (t + a * torch.sin(t)) + z[2], Y)


def branch_point(z0, s, a, x, y, iters=100):
    """Damped Newton on the joint loss in z = (w₁, b₁, b₂) at w₂ = s."""
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    f = lambda z: _loss_t(z, s, a, X, Y)
    z = torch.tensor(np.asarray(z0, float), dtype=torch.float64)
    mu, f0 = 1e-6, float(f(z))
    for _ in range(iters):
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(f(zz), zz)[0]
        if float(g.abs().max()) < 1e-13:
            break
        H = torch.autograd.functional.hessian(f, z)
        ok = False
        for _ in range(40):
            d = torch.linalg.solve(H + mu * torch.eye(3, dtype=torch.float64), -g)
            f1 = float(f(z + d))
            if f1 <= f0:
                z, f0, ok, mu = z + d, f1, True, max(mu / 10, 1e-14)
                break
            mu *= 10
        if not ok:
            break
    zz = z.clone().requires_grad_(True)
    g = torch.autograd.grad(f(zz), zz)[0]
    return z.numpy(), float(g.abs().max())


def gap(w1, b1, a):
    from .width2_geometry import Act, gaps
    g = gaps(np.array([w1, b1, 0.0, 0.0]), np.array([1.0, 0.0]), Act("fa", a))["G+"]
    return 0.5 * (g[0] + g[1])


def grad_gap(w1, b1, a, h=1e-6):
    gw = (gap(w1 + h, b1, a) - gap(w1 - h, b1, a)) / (2 * h)
    gb = (gap(w1, b1 + h, a) - gap(w1, b1 - h, a)) / (2 * h)
    fw = (gap(w1 + h, b1, a) - gap(w1, b1, a)) / h
    bw = (gap(w1, b1, a) - gap(w1 - h, b1, a)) / h
    return np.array([gw, gb, 0.0]), abs(fw - bw) / max(abs(gw), 1e-12)


def hessian_and_tangent(z, s, a, x, y):
    import torch
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    zt = torch.tensor(z, dtype=torch.float64)
    H = torch.autograd.functional.hessian(lambda q: _loss_t(q, s, a, X, Y), zt).numpy()
    st = torch.tensor(s, dtype=torch.float64, requires_grad=True)
    zz = zt.clone().requires_grad_(True)
    g = torch.autograd.grad(_loss_t(zz, st, a, X, Y), zz, create_graph=True)[0]
    dgds = np.array([float(torch.autograd.grad(g[i], st, retain_graph=True)[0]) for i in range(3)])
    tangent = -np.linalg.solve(H, dgds)
    return H, tangent


def switch(a):
    """The switch s* on the certified global branch, and the landscape quantities there."""
    x, y = _pop()
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv").query("kind == 'glob'")
    br = br[br.a.round(2) == round(a, 2)].iloc[0]
    au = pd.read_csv(RESULTS / "cond_audit_candidates.csv")
    r = au[(au.status == "RETAINED") & (au.a.round(2) == round(a, 2))].copy()
    r["d"] = (r.s - 0.5 * (br.w2_lo + br.w2_hi)).abs()
    r0 = r.sort_values("d").iloc[0]
    z = np.array([r0.w1, r0.b1, r0.b2])            # the retained minimiser as recorded (w₂ = +s orientation)
    lo, hi = float(br.w2_lo) * 0.995, float(br.w2_hi) * 1.005
    z_lo, _ = branch_point(z, lo, a, x, y); z_hi, _ = branch_point(z_lo, hi, a, x, y)
    g_lo, g_hi = gap(z_lo[0], z_lo[1], a), gap(z_hi[0], z_hi[1], a)
    if not (g_lo < 0 < g_hi):
        raise SystemExit(f"a={a}: gap does not change sign on the tracked branch ({g_lo}, {g_hi})")
    zm = z_lo
    for _ in range(60):                            # bisection on s for G(θ*(s)) = 0, continuation in z
        mid = 0.5 * (lo + hi)
        zm, gm = branch_point(zm, mid, a, x, y)
        if gap(zm[0], zm[1], a) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-11:
            break
    s_star = 0.5 * (lo + hi)
    z_star, gres = branch_point(zm, s_star, a, x, y)
    H, tan = hessian_and_tangent(z_star, s_star, a, x, y)
    dG, asym = grad_gap(z_star[0], z_star[1], a)
    return {"a": a, "s_star": s_star, "in_certified_bracket": bool(br.w2_lo <= s_star <= br.w2_hi),
            "z_star": z_star.tolist(), "grad_residual": gres, "H": H.tolist(), "tangent": tan.tolist(),
            "gradG": dG.tolist(), "gradG_onesided_rel_diff": asym, "H_pd": bool(np.linalg.eigvalsh(H).min() > 0)}


def kappa(H, tan, dG, p):
    """κ = λ_min(P^{1/2}HP^{1/2}) · [∇G·(PH)⁻¹θ*′] / [∇G·θ*′] for P = diag(p)."""
    H, tan, dG, p = (np.asarray(v, float) for v in (H, tan, dG, p))
    Ph = np.sqrt(p)
    lam = float(np.linalg.eigvalsh((Ph[:, None] * H) * Ph[None, :]).min())
    num = float(dG @ np.linalg.solve(p[:, None] * H, tan))
    den = float(dG @ tan)
    return lam * num / den, lam, num, den


# ------------------------------------------------------------------------------------------ Adam preconditioner at crossing
def _p_lag2(args):
    """v̂ at the crossing of a deconfounded-lag-test primary run at φ = 1 (replay from the saved state)."""
    os.nice(15)
    a, seed, cross_step, w2_ref = args
    from .timescale_consistency import replay_lag2
    th, op, hist, t0, x, y = replay_lag2(a, seed, 1.0, cross_step, "primary")
    st = op.state[th]
    vhat = st["exp_avg_sq"].numpy() / (1 - 0.999 ** int(st["step"]))
    return {"a": a, "seed": seed, "reproduced": abs(float(th.detach()[2])) == w2_ref,
            "sqrt_v_w1": math.sqrt(vhat[0]), "sqrt_v_b1": math.sqrt(vhat[1]), "sqrt_v_b2": math.sqrt(vhat[3])}


def _p_tsb(args):
    """v̂ at the crossing of a Task B run (standard protocol, deterministic replay of ts_test.run_one)."""
    os.nice(15)
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data
    from .phase2b_ordering import state
    from .fold1d_theorem import maximum_gap
    a, seed, cross_step, w2_ref = args
    torch.set_num_threads(1)
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=1e-2)
    x, y = x.double(), y.double()
    for _ in range(int(cross_step)):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        opt.step()
    st = opt.state[th]
    vhat = st["exp_avg_sq"].numpy() / (1 - 0.999 ** int(st["step"]))
    return {"a": a, "seed": seed, "reproduced": abs(float(th.detach()[2])) == w2_ref,
            "sqrt_v_w1": math.sqrt(vhat[0]), "sqrt_v_b1": math.sqrt(vhat[1]), "sqrt_v_b2": math.sqrt(vhat[3])}


def preconditioners(workers=3):
    from multiprocessing import get_context
    l2 = pd.read_csv(RESULTS / "lag_test2_runs.csv", float_precision="round_trip")
    l2 = l2[(l2.rule == "primary") & (l2.factor == 1.0) & l2.cross_step.notna()]
    tb = pd.read_csv(RESULTS / "ts_test" / "runs.csv", float_precision="round_trip"); tb = tb[tb.crossed]
    with get_context("spawn").Pool(workers) as pool:
        r1 = pool.map(_p_lag2, [(r.a, int(r.seed), int(r.cross_step), r.w2_abs) for r in l2.itertuples()])
        r2 = pool.map(_p_tsb, [(r.a, int(r.seed), int(r.step), r.w2_cross) for r in tb.itertuples()])
    d = pd.DataFrame(r1 + r2)
    d.to_csv(OUT / "preconditioner_runs.csv", index=False)
    return d


def run_kappa(workers=3):
    OUT.mkdir(exist_ok=True)
    pr = preconditioners(workers)
    if not pr.reproduced.all():
        raise SystemExit("STOP: a preconditioner replay did not reproduce its crossing")
    rows = []
    for a in A_VALUES:
        sw = switch(a)
        g = pr[pr.a.round(2) == round(a, 2)]
        eps = 1e-8
        P = np.column_stack([1 / (g.sqrt_v_w1 + eps), 1 / (g.sqrt_v_b1 + eps), 1 / (g.sqrt_v_b2 + eps)])
        Pn = P / P.sum(axis=1, keepdims=True)                         # κ is scale-invariant in P
        p_med = np.median(Pn, axis=0)
        k_adam, lam, num, den = kappa(sw["H"], sw["tangent"], sw["gradG"], p_med)
        k_runs = np.array([kappa(sw["H"], sw["tangent"], sw["gradG"], p)[0] for p in Pn])
        k_sgd = kappa(sw["H"], sw["tangent"], sw["gradG"], np.ones(3))[0]
        rows.append({**{k: v for k, v in sw.items() if k not in ("H", "tangent", "gradG", "z_star")},
                     "kappa_adam": k_adam, "kappa_adam_q25": float(np.percentile(k_runs, 25)),
                     "kappa_adam_q75": float(np.percentile(k_runs, 75)), "kappa_sgd": k_sgd,
                     "p_median": p_med.tolist(), "n_precond_runs": len(g), "lambda_min_adam_rel": lam,
                     "H": json.dumps(sw["H"]), "tangent": json.dumps(sw["tangent"]), "gradG": json.dumps(sw["gradG"]),
                     "z_star": json.dumps(sw["z_star"])})
        print(json.dumps({"a": a, "s_star": sw["s_star"], "kappa_adam": k_adam, "kappa_sgd": k_sgd,
                          "in_bracket": sw["in_certified_bracket"]}), flush=True)
    k = pd.DataFrame(rows)
    k.to_csv(OUT / "kappa.csv", index=False)
    predictions(k)


def predictions(k):
    """Committed BEFORE any comparison: per-run predicted residual r = κ(a)·χ for every run of the 32 intervention arms,
    the SGD test and Task B; stored WITHOUT the observed residuals."""
    ka = dict(zip(k.a.round(2), k.kappa_adam)); ks = dict(zip(k.a.round(2), k.kappa_sgd))
    rows = []
    tc = pd.read_csv(RESULTS / "timescale_consistency" / "runs.csv")
    p = pd.read_csv(RESULTS / "residual_timescale_runs.csv")
    for r in tc.itertuples():
        rows.append({"set": r.test, "a": round(r.a, 2), "seed": r.seed, "arm": str(r.arm), "chi": r.ratio,
                     "pred_r": ka[round(r.a, 2)] * r.ratio})
    for r in p.itertuples():
        rows.append({"set": "lag2-prim", "a": round(r.a, 2), "seed": r.seed, "arm": str(r.factor), "chi": r.ratio,
                     "pred_r": ka[round(r.a, 2)] * r.ratio})
    sg = pd.read_csv(RESULTS / "sgd_own_ratios.csv")
    for r in sg.itertuples():
        rows.append({"set": "SGD", "a": round(r.a, 2), "seed": r.seed, "arm": "sgd", "chi": r.ratio,
                     "pred_r": ks[round(r.a, 2)] * r.ratio})
    tb = pd.read_csv(RESULTS / "ts_test" / "runs.csv"); tb = tb[tb.crossed]
    for r in tb.itertuples():
        rows.append({"set": "TaskB", "a": round(r.a, 2), "seed": r.seed, "arm": "adam", "chi": r.ratio,
                     "pred_r": ka[round(r.a, 2)] * r.ratio})
    pd.DataFrame(rows).to_csv(OUT / "predictions.csv", index=False)




# ------------------------------------------------------------------------------------------ winding-dependent κ_k
TWO_PI = 2 * math.pi


def kappa_k(H, tan, dG, p, k):
    """κ for the run's winding k: b₁ → b₁ + 2πk leaves the loss, H, s* and ∇G unchanged but shifts the output bias's
    drift, θ*′ → θ*′ − 2πk·e_b₂ (b₂* = b₂* − 2πk·s).  Verified by controlled ramps (math note §13)."""
    t = np.asarray(tan, float) - TWO_PI * k * np.array([0.0, 0.0, 1.0])
    return kappa(H, t, dG, p)[0]


def canonical_b1(w1, b1, w2):
    return (b1 if w2 > 0 else -b1), (w1 if w2 > 0 else -w1)


def _theta_job(args):
    """(w₁, b₁, w₂) at the crossing, by deterministic replay, for sets whose records lack it."""
    os.nice(15)
    kind, a, seed, arm, step, w2_ref = args
    import torch
    if kind == "4b":
        from .timescale_consistency import replay_4b
        th, *_ = replay_4b(a, seed, arm, step)
        q = th.detach().numpy()
    elif kind == "SGD":
        from torch.nn import functional as F
        from .fold1d import logits
        from .sgd_own import SGD_LR, _setup
        f, x, y, th = _setup(a, seed)
        opt = torch.optim.SGD([th], lr=SGD_LR)
        for _ in range(int(step)):
            opt.zero_grad(set_to_none=True); F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward(); opt.step()
        q = th.detach().numpy()
    else:                                                      # Task B (standard protocol)
        from torch.nn import functional as F
        from .fold1d import activation, logits, make_data
        torch.set_num_threads(1)
        f = activation("sin_family", a); x, y = make_data(200, seed); torch.manual_seed(seed)
        th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
        opt = torch.optim.Adam([th], lr=1e-2); x, y = x.double(), y.double()
        for _ in range(int(step)):
            opt.zero_grad(set_to_none=True); F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward(); opt.step()
        q = th.detach().numpy()
    return {"set": kind, "a": round(a, 2), "seed": seed, "arm": str(arm), "w1": float(q[0]), "b1": float(q[1]),
            "w2": float(q[2]), "reproduced": abs(float(q[2])) == w2_ref}


def crossing_states(workers=3):
    from multiprocessing import get_context
    rows = []
    l1 = pd.read_csv(RESULTS / "lag_test_runs.csv"); l1 = l1[l1.cross_step.notna()]
    rows += [{"set": "lag1", "a": round(r.a, 2), "seed": r.seed, "arm": str(r.factor), "w1": r.w1, "b1": r.b1, "w2": r.w2,
              "reproduced": True} for r in l1.itertuples()]
    l2 = pd.read_csv(RESULTS / "lag_test2_runs.csv"); l2 = l2[l2.cross_step.notna()]
    rows += [{"set": "lag2-prim" if r.rule == "primary" else "lag2-tstar", "a": round(r.a, 2), "seed": r.seed,
              "arm": str(r.factor), "w1": r.w1, "b1": r.b1, "w2": r.w2, "reproduced": True} for r in l2.itertuples()]
    J = []
    b = pd.read_csv(RESULTS / "residual_mechanism_parts.csv", float_precision="round_trip"); b = b[b.cross_step.notna()]
    J += [("4b", r.a, int(r.seed), r.arm, int(r.cross_step), r.w2_abs) for r in b.itertuples()]
    s = pd.read_csv(RESULTS / "sgd_own_runs.csv", float_precision="round_trip"); s = s[s.crossed]
    J += [("SGD", r.a, int(r.seed), "sgd", int(r.step), r.w2_cross) for r in s.itertuples()]
    t = pd.read_csv(RESULTS / "ts_test" / "runs.csv", float_precision="round_trip"); t = t[t.crossed]
    J += [("TaskB", r.a, int(r.seed), "adam", int(r.step), r.w2_cross) for r in t.itertuples()]
    with get_context("spawn").Pool(workers) as pool:
        rows += pool.map(_theta_job, J)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "crossing_states.csv", index=False)
    return d


def commit_predictions(workers=3):
    """κ_k per run (its winding at crossing) and the predicted residual r = κ_k·χ, for Adam (median P) and SGD (P = I),
    written WITHOUT any observed residual.  Mirror-branch runs are counted."""
    k = pd.read_csv(OUT / "kappa.csv")
    cs = crossing_states(workers)
    if not cs.reproduced.all():
        raise SystemExit("STOP: a crossing-state replay did not reproduce")
    land = {round(r.a, 2): r for r in k.itertuples()}
    tc = pd.read_csv(RESULTS / "timescale_consistency" / "runs.csv")
    chis = [pd.DataFrame({"set": tc.test, "a": tc.a.round(2), "seed": tc.seed, "arm": tc.arm.astype(str), "chi": tc.ratio})]
    p = pd.read_csv(RESULTS / "residual_timescale_runs.csv")
    chis.append(pd.DataFrame({"set": "lag2-prim", "a": p.a.round(2), "seed": p.seed, "arm": p.factor.astype(str), "chi": p.ratio}))
    sg = pd.read_csv(RESULTS / "sgd_own_ratios.csv")
    chis.append(pd.DataFrame({"set": "SGD", "a": sg.a.round(2), "seed": sg.seed, "arm": "sgd", "chi": sg.ratio}))
    tb = pd.read_csv(RESULTS / "ts_test" / "runs.csv"); tb = tb[tb.crossed]
    chis.append(pd.DataFrame({"set": "TaskB", "a": tb.a.round(2), "seed": tb.seed, "arm": "adam", "chi": tb.ratio}))
    chi = pd.concat(chis, ignore_index=True)
    cs["arm"] = cs.arm.astype(str); chi["arm"] = chi.arm.astype(str)
    # lag-test arm labels: factors are floats in one file and strings in another; normalise
    norm = lambda v: str(float(v)) if v.replace(".", "", 1).isdigit() else v
    cs["arm"] = cs.arm.map(norm); chi["arm"] = chi.arm.map(norm)
    m = chi.merge(cs, on=["set", "a", "seed", "arm"], how="left")
    rows = []
    for r in m.itertuples():
        L_ = land[r.a]
        H = np.array(json.loads(L_.H)); tan = np.array(json.loads(L_.tangent)); dG = np.array(json.loads(L_.gradG))
        zs = json.loads(L_.z_star)
        b1c, w1c = canonical_b1(r.w1, r.b1, r.w2)
        # mirror branch (x → −x): (w₁, b₁) → (−w₁, b₁) leaves G, s* and κ unchanged on the symmetric population
        k_ = int(round((b1c - zs[1]) / TWO_PI))
        off = float(b1c - zs[1] - TWO_PI * k_)
        pvec = np.ones(3) if r.set == "SGD" else np.array(json.loads(L_.p_median))
        kap = kappa_k(H, tan, dG, pvec, k_)
        rows.append({"set": r.set, "a": r.a, "seed": r.seed, "arm": r.arm, "chi": r.chi, "winding_k": k_,
                     "b1_offset": off, "mirror": bool(np.sign(w1c) != np.sign(zs[0])), "kappa_k": kap,
                     "pred_r": kap * r.chi})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "predictions.csv", index=False)
    kk = [{"a": a, "k": kk_, "kappa_adam": kappa_k(np.array(json.loads(land[a].H)), json.loads(land[a].tangent),
                                                   json.loads(land[a].gradG), np.array(json.loads(land[a].p_median)), kk_),
           "kappa_sgd": kappa_k(np.array(json.loads(land[a].H)), json.loads(land[a].tangent), json.loads(land[a].gradG),
                                np.ones(3), kk_)} for a in land for kk_ in (-1, 0, 1)]
    pd.DataFrame(kk).to_csv(OUT / "kappa_by_winding.csv", index=False)
    print(d.groupby(["set", "a", "winding_k"]).size().to_string())
    print(pd.DataFrame(kk).to_string(index=False))
    print("mirror-branch runs:", int(d.mirror.sum()), "of", len(d), "; max |b1 offset|:", float(d.b1_offset.abs().max()))


def compare():
    """After the commit: per-arm predicted vs observed median residual (tolerance max(0.01, 0.25|pred|)), per-a
    observed-to-predicted slope ratio, and the post hoc through-origin fit per a beside it.  No refit of κ."""
    pr = pd.read_csv(OUT / "predictions.csv")
    tc = pd.read_csv(RESULTS / "timescale_consistency" / "runs.csv")
    obs = [pd.DataFrame({"set": tc.test, "a": tc.a.round(2), "seed": tc.seed, "arm": tc.arm.astype(str), "obs_r": tc.residual})]
    p = pd.read_csv(RESULTS / "residual_timescale_runs.csv")
    obs.append(pd.DataFrame({"set": "lag2-prim", "a": p.a.round(2), "seed": p.seed, "arm": p.factor.astype(str), "obs_r": p.residual}))
    s = pd.read_csv(RESULTS / "sgd_own_runs.csv"); s = s[s.crossed].copy(); s["a"] = s.a.round(2)
    own = pd.read_csv(RESULTS / "own_threshold_crossing.csv"); own["a"] = own.a.round(2)
    s = s.merge(own[["a", "seed", "w2_own"]], on=["a", "seed"])
    obs.append(pd.DataFrame({"set": "SGD", "a": s.a, "seed": s.seed, "arm": "sgd", "obs_r": s.w2_cross / s.w2_own - 1}))
    t = pd.read_csv(RESULTS / "ts_test" / "runs.csv"); t = t[t.crossed].copy(); t["a"] = t.a.round(2)
    fo = pd.read_csv(RESULTS / "ts_test_own_frozen.csv"); fo["a"] = fo.a.round(2)
    t = t.merge(fo[["a", "seed", "w2_own"]], on=["a", "seed"])
    obs.append(pd.DataFrame({"set": "TaskB", "a": t.a, "seed": t.seed, "arm": "adam", "obs_r": t.w2_cross / t.w2_own - 1}))
    o = pd.concat(obs, ignore_index=True)
    norm = lambda v: str(float(v)) if str(v).replace(".", "", 1).isdigit() else str(v)
    o["arm"] = o.arm.map(norm); pr["arm"] = pr.arm.astype(str).map(norm)
    m = pr.merge(o, on=["set", "a", "seed", "arm"], how="inner")
    assert len(m) == len(pr), (len(m), len(pr))
    arms = []
    for (st, a, arm), g in m.groupby(["set", "a", "arm"]):
        ok = g[np.isfinite(g.chi) & (g.chi > 0)]
        pred = float(ok.pred_r.median()); obsm = float(g.obs_r.median()); tol = max(0.01, 0.25 * abs(pred))
        arms.append({"set": st, "a": a, "arm": arm, "n": len(g), "median_chi": float(ok.chi.median()), "pred": pred,
                     "obs": obsm, "tol": tol, "within": abs(obsm - pred) <= tol, "obs_over_pred": obsm / pred})
    A = pd.DataFrame(arms)
    A.to_csv(OUT / "compare_arms.csv", index=False)
    per_a = []
    for a, g in m[np.isfinite(m.chi) & (m.chi > 0)].groupby("a"):
        kap = float(g.kappa_k.median())
        slope_obs = float((g.chi * g.obs_r).sum() / (g.chi ** 2).sum())          # post hoc through-origin fit
        per_a.append({"a": a, "n": len(g), "kappa_committed": kap, "slope_posthoc_through_origin": slope_obs,
                      "ratio_obs_to_pred": slope_obs / kap})
    P = pd.DataFrame(per_a); P.to_csv(OUT / "compare_per_a.csv", index=False)
    pd.set_option("display.width", 220)
    print(A.round(4).to_string(index=False)); print(P.round(4).to_string(index=False))
    print("arms within tolerance:", int(A.within.sum()), "of", len(A))


# ------------------------------------------------------------------------------------------ winding check (controlled ramps)
WINDING_CHECKS = ((1.30, 0, 0.003), (1.30, -1, 0.003), (1.30, 1, 0.003), (1.50, 0, 0.002), (1.50, 0, 0.005))


def winding_check():
    """Controlled population GD ramps (η = 0.3, P = I) on the winding copy k of the branch, from 0.9·s* at
    s = s₀·exp(γt) with γ = χ·η·λ_min(H): measured r = s_c/s* − 1 against κ_k·χ.  Run before any comparison of the
    committed predictions (math note §13); written to winding_check.csv."""
    import torch
    x, y = _pop(); X, Y = torch.tensor(x), torch.tensor(y)
    rows = []
    for a, k, chi in WINDING_CHECKS:
        sw = switch(a); s_star = sw["s_star"]; H = np.array(sw["H"]); tan = np.array(sw["tangent"]); g = np.array(sw["gradG"])
        lam = float(np.linalg.eigvalsh(H).min())
        kap = lam * float(g @ np.linalg.solve(H, tan - TWO_PI * k * np.array([0, 0, 1.0]))) / float(g @ tan)
        eta = 0.3; gamma = chi * eta * lam; s0 = 0.9 * s_star
        z0 = np.array(sw["z_star"]); z0 = np.array([z0[0], z0[1] + TWO_PI * k, z0[2] - TWO_PI * k * s_star])
        z, _ = branch_point(z0, s0, a, x, y); z = torch.tensor(z, requires_grad=True); t = 0
        while True:
            t += 1; s = s0 * math.exp(gamma * t)
            gg, = torch.autograd.grad(_loss_t(z, s, a, X, Y), z)
            with torch.no_grad():
                z -= eta * gg
            if gap(float(z[0].detach()), float(z[1].detach()), a) > 0 or s > 2 * s_star:
                break
        rows.append({"a": a, "k": k, "chi": chi, "b1_at_crossing": float(z[1].detach()), "kappa_k_sgd": kap,
                     "measured_r": s / s_star - 1, "predicted_r": kap * chi})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "winding_check.csv", index=False)


def relaxation_rates():
    """Math note §13.3(iii): the relaxation rate ηλ_min(P^{1/2}HP^{1/2}) per step with Adam's median ABSOLUTE crossing
    preconditioner (preconditioner_runs.csv) and η = 0.01, against the momentum time constant 1/(1 − β₁) = 10 steps."""
    k = pd.read_csv(OUT / "kappa.csv"); pr = pd.read_csv(OUT / "preconditioner_runs.csv")
    rows = []
    for r in k.itertuples():
        g = pr[pr.a.round(2) == round(r.a, 2)]
        if not len(g):
            continue
        p = np.array([np.median(1 / (g.sqrt_v_w1 + 1e-8)), np.median(1 / (g.sqrt_v_b1 + 1e-8)), np.median(1 / (g.sqrt_v_b2 + 1e-8))])
        H = np.array(json.loads(r.H))
        lam = float(np.linalg.eigvalsh((np.sqrt(p)[:, None] * H) * np.sqrt(p)[None, :]).min())
        rows.append({"a": round(r.a, 2), "n_runs": len(g), "eta_lambda_min": 0.01 * lam, "relaxation_steps": 1 / (0.01 * lam),
                     "momentum_steps": 10.0})
    d = pd.DataFrame(rows); d.to_csv(OUT / "relaxation.csv", index=False); print(d.to_string(index=False))


if __name__ == "__main__":
    {"kappa": lambda: run_kappa(int(sys.argv[2]) if len(sys.argv) > 2 else 3),
     "commit": lambda: commit_predictions(int(sys.argv[2]) if len(sys.argv) > 2 else 3), "compare": compare,
     "winding_check": winding_check, "relaxation": relaxation_rates}[sys.argv[1]]()
