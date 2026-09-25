"""Item 1 (POST HOC, existing data): is the frozen timescale relationship consistent with the within-a interventions?

Frozen relationship: residual_timescale_fit.json (committed in 1df48d1; SHA-256 16792946a149…), residual = α + β·ratio,
the file the SGD extension and Task B used.  For every crossing run of
  lag1        the first lag test (w₂'s learning-rate factor φ from step 1; lag_test.run_one), every φ arm
  lag2-prim   the deconfounded lag test, primary 0.7× rule (already replayed in Track 3: residual_timescale_runs.csv)
  lag2-tstar  the deconfounded lag test, t* rule, every φ arm (from its saved t* state; lag_test2.continue_one)
  4b          Block 4b, every arm (control / teleport / reset / teleport_reset; residual_mechanism.run_one)
the run is replayed deterministically to its recorded crossing step (the replay must reproduce the recorded |w₂| at
crossing exactly), and the ratio at crossing is measured as in Track 3 (growth d log|w₂|/dt over the last
min(100, steps since the replay started) steps; relaxation lr·λ_min(D^{-1/2} H D^{-1/2}) at the branch).
Residual = R_cross / R_own − 1 with the global own threshold (n = 6,400; lag_test2._own), as in the fit.

Per arm (test × a × φ or arm): predicted median = median(α + β·ratioᵢ), observed median residual, within
max(0.01, 0.25·|pred|) (the registered tests' tolerance).  Also: within-a vs pooled slopes, partial correlation of
residual and ratio controlling for a, and how far the φ arms move the ratio at crossing.

    python -m src.timescale_consistency run [workers] | summarise
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
OUT = RESULTS / "timescale_consistency"
WINDOW = 100


def _ratio(theta, opt, hist, step, t0, a, x, y):
    from .residual_timescale import branch_hessian, relax_rate
    win = min(WINDOW, step - t0)
    growth = math.log(hist[step] / hist[step - win]) / win if win >= 1 else float("nan")
    q = theta.detach().numpy()
    st = opt.state[theta]
    vhat = st["exp_avg_sq"].numpy() / (1 - 0.999 ** int(st["step"]))
    _, H, _, conv = branch_hessian(a, x, y, q[0], q[1], q[2], q[3])
    lam = relax_rate(H, vhat[[0, 1, 3]])
    return growth, lam, (growth / lam if lam > 0 else float("nan")), conv


def _drive(theta, opt, a, seed, factor, t0, cross_step, step_fn):
    """Step from t0 to cross_step; |w₂| history for the growth window; the state at the crossing."""
    hist = {t0: abs(float(theta.detach()[2]))}
    for t in range(t0 + 1, int(cross_step) + 1):
        step_fn(theta, opt, factor)
        hist[t] = abs(float(theta.detach()[2])); hist.pop(t - WINDOW - 1, None)
    return hist


def replay_lag1(a, seed, factor, cross_step, w2_ref):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits
    from .lag_test import N, scale_w2_step
    from .sample_size import _data
    torch.set_num_threads(1)
    f = activation("sin_family", a)
    xt, yt = (torch.tensor(v) for v in _data(N, seed))
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)

    def step_fn(th, op, fac):
        op.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, xt, f), yt).backward()
        w2b = float(th.detach()[2]); op.step(); scale_w2_step(th, w2b, fac)
    hist = _drive(theta, opt, a, seed, factor, 0, cross_step, step_fn)
    return theta, opt, hist, 0, xt.numpy().astype(float), yt.numpy().astype(float)


def replay_lag2(a, seed, factor, cross_step, rule):
    import torch
    from . import lag_test2 as lt
    f, G, xt, yt, _, _ = lt._setup(a, seed)
    ck = torch.load(lt.STATES / f"{rule}_a{a:.2f}_s{seed}.pt", weights_only=False)
    theta = ck["theta"].clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2); opt.load_state_dict(ck["opt"])
    t0 = ck["t_star"]
    hist = _drive(theta, opt, a, seed, factor, t0, cross_step, lambda th, op, fac: lt._step(th, op, f, xt, yt, fac))
    return theta, opt, hist, t0, xt.numpy().astype(float), yt.numpy().astype(float)


def replay_4b(a, seed, arm, cross_step):
    import torch
    from . import lag_test2 as lt
    from . import residual_mechanism as rm
    from .sample_size import _data
    ck = torch.load(lt.STATES / f"primary_a{a:.2f}_s{seed}.pt", weights_only=False)
    theta0 = tuple(float(v) for v in ck["theta"])
    x, y = _data(lt.N, seed)
    if arm.startswith("teleport"):
        th, _ = rm.apply_teleport(theta0, rm.teleport_target(a, seed, theta0, x, y), a, x, y)
    else:
        th = theta0
    opt_state = rm.reset_moments(ck["opt"]) if arm.endswith("reset") else ck["opt"]
    if arm == "teleport":
        # AS RUN: Block 4b ran the control arm first with the same optimiser-state dict, whose tensors the control
        # continuation mutated in place (load_state_dict shares them); the recorded teleport arm therefore started from
        # the control run's end-of-run Adam state.  Reproduce that (verified: the recorded crossing reproduces).
        rm.continue_from(a, seed, theta0, ck["opt"], ck["t_star"])
    f, G, xt, yt, _, _ = lt._setup(a, seed)
    theta = torch.tensor(th, dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2); opt.load_state_dict(opt_state)
    t0 = ck["t_star"]
    hist = _drive(theta, opt, a, seed, 1.0, t0, cross_step, lambda th_, op, fac: lt._step(th_, op, f, xt, yt, fac))
    return theta, opt, hist, t0, xt.numpy().astype(float), yt.numpy().astype(float)


def _job(args):
    os.nice(15)
    test, a, seed, arm, cross_step, w2_ref, R_cross, R_own = args
    if test == "lag1":
        th, op, hist, t0, x, y = replay_lag1(a, seed, arm, cross_step, w2_ref)
    elif test == "lag2-tstar":
        th, op, hist, t0, x, y = replay_lag2(a, seed, arm, cross_step, "tstar")
    else:
        th, op, hist, t0, x, y = replay_4b(a, seed, arm, cross_step)
    g, lam, r, conv = _ratio(th, op, hist, int(cross_step), t0, a, x, y)
    return {"test": test, "a": a, "seed": seed, "arm": str(arm), "cross_step": cross_step,
            "reproduced": abs(float(th.detach()[2])) == w2_ref, "growth": g, "relax": lam, "ratio": r,
            "branch_converged": conv, "residual": R_cross / R_own - 1}


def jobs():
    from .lag_test2 import _own
    own = _own()
    J = []
    l1 = pd.read_csv(RESULTS / "lag_test_runs.csv", float_precision="round_trip"); l1 = l1[l1.cross_step.notna()]
    J += [("lag1", r.a, int(r.seed), r.factor, int(r.cross_step), r.w2_abs, r.R_cross, float(own.loc[(r.a, r.seed)]))
          for r in l1.itertuples()]
    l2 = pd.read_csv(RESULTS / "lag_test2_runs.csv", float_precision="round_trip")
    l2 = l2[(l2.rule == "tstar") & l2.cross_step.notna()]
    J += [("lag2-tstar", r.a, int(r.seed), r.factor, int(r.cross_step), r.w2_abs, r.R_cross, float(own.loc[(r.a, r.seed)]))
          for r in l2.itertuples()]
    b = pd.read_csv(RESULTS / "residual_mechanism_parts.csv", float_precision="round_trip"); b = b[b.cross_step.notna()]
    J += [("4b", r.a, int(r.seed), r.arm, int(r.cross_step), r.w2_abs, r.R_cross, float(own.loc[(r.a, r.seed)]))
          for r in b.itertuples()]
    return J


def run(workers=2):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    f = OUT / "runs.csv"
    done = set() if not f.exists() else {(r.test, round(r.a, 2), r.seed, str(r.arm)) for r in pd.read_csv(f).itertuples()}
    todo = [j for j in jobs() if (j[0], round(j[1], 2), j[2], str(j[3])) not in done]
    todo.sort(key=lambda j: j[4])
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_job, todo):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)


def arm_table(d, alpha, beta):
    rows = []
    for (test, a, arm), g in d.groupby(["test", "a", "arm"]):
        ok = g[np.isfinite(g.ratio) & (g.ratio > 0)]
        pred = float(np.median(alpha + beta * ok.ratio)); obs = float(np.median(g.residual))
        tol = max(0.01, 0.25 * abs(pred))
        rows.append({"test": test, "a": a, "arm": arm, "n": len(g), "n_ratio": len(ok),
                     "median_ratio": float(ok.ratio.median()), "pred": pred, "obs": obs, "diff": obs - pred, "tol": tol,
                     "within_tol": abs(obs - pred) <= tol})
    return pd.DataFrame(rows)


def partial_spearman(d):
    """Spearman partial correlation of residual and ratio controlling for a: ranks demeaned within a."""
    r1 = d.groupby("a").residual.rank(); r2 = d.groupby("a").ratio.rank()
    r1 = r1 - r1.groupby(d.a).transform("mean"); r2 = r2 - r2.groupby(d.a).transform("mean")
    return float(np.corrcoef(r1, r2)[0, 1])


def summarise():
    fit = json.loads((RESULTS / "residual_timescale_fit.json").read_text())
    d = pd.read_csv(OUT / "runs.csv")
    if not d.reproduced.all():
        raise SystemExit(f"STOP: {int((~d.reproduced).sum())} replays did not reproduce")
    p = pd.read_csv(RESULTS / "residual_timescale_runs.csv")
    p = pd.DataFrame({"test": "lag2-prim", "a": p.a, "seed": p.seed, "arm": p.factor.astype(str), "cross_step": p.cross_step,
                      "reproduced": p.reproduced, "growth": p.growth, "relax": p.relax, "ratio": p.ratio,
                      "branch_converged": p.branch_converged, "residual": p.residual})
    d = pd.concat([d, p], ignore_index=True)
    d["a"] = d.a.round(2)
    tab = arm_table(d, fit["alpha"], fit["beta"])
    tab.to_csv(OUT / "arms.csv", index=False)
    ok = d[np.isfinite(d.ratio) & (d.ratio > 0)]
    slope = lambda g: float(np.polyfit(g.ratio, g.residual, 1)[0])
    out = {"label": "POST HOC; no registered verdict changes", "frozen_fit": {"alpha": fit["alpha"], "beta": fit["beta"]},
           "runs": len(d), "arms_within_tol": int(tab.within_tol.sum()), "arms": len(tab),
           "slope_pooled_all": slope(ok), "slope_within_a_all": {f"{a:.2f}": slope(g) for a, g in ok.groupby("a")},
           "slope_pooled_fitdata": slope(ok[ok.test == "lag2-prim"]),
           "slope_within_a_fitdata": {f"{a:.2f}": slope(g) for a, g in ok[ok.test == "lag2-prim"].groupby("a")},
           "partial_spearman_all": partial_spearman(ok), "partial_spearman_fitdata": partial_spearman(ok[ok.test == "lag2-prim"]),
           "ratio_by_arm": {f"{t}|{a:.2f}|{arm}": float(g.ratio.median()) for (t, a, arm), g in ok.groupby(["test", "a", "arm"])}}
    for test in ("lag1", "lag2-prim", "lag2-tstar", "4b"):
        g = tab[tab.test == test]
        for a in (1.3, 1.5):
            ga = g[g.a == a]
            if len(ga):
                out[f"ratio_range_across_arms|{test}|{a:.2f}"] = [float(ga.median_ratio.min()), float(ga.median_ratio.max())]
                out[f"obs_range_across_arms|{test}|{a:.2f}"] = [float(ga.obs.min()), float(ga.obs.max())]
                out[f"pred_range_across_arms|{test}|{a:.2f}"] = [float(ga.pred.min()), float(ga.pred.max())]
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False)); print(json.dumps({k: v for k, v in out.items() if k != "ratio_by_arm"}, indent=1))


if __name__ == "__main__":
    {"run": lambda: run(int(sys.argv[2]) if len(sys.argv) > 2 else 2), "summarise": summarise}[sys.argv[1]]()
