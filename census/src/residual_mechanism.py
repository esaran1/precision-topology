"""Block 4b (registered design: results/residual_mechanism_design.md, approved with the branch-preservation stop):
inherited displacement vs optimiser memory, by four interventions at the deconfounded switch point.

Arms, from each included run's saved primary-rule state (lag_test2_states/primary_a*_s*.pt), continued at φ = 1
with the lag test's own step (`lag_test2._step`) and crossing rule (`phase2b_ordering.state`):
  control          none
  teleport         (w₁, b₁) → the occupied branch's conditional minimiser at the current |w₂| (mirror_branches.half_min,
                   step 0.05, 16 refinements, the run's own 6,400 points), mapped back to the run's orientation with
                   the 2π-representative of b₁ nearest the current one; b₂ → its profiled optimum; Adam state kept
  reset            Adam's moments and step count zeroed; parameters unchanged
  teleport_reset   both

    python -m src.residual_mechanism run [workers]
    python -m src.residual_mechanism score
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "residual_mechanism_parts.csv"
ARMS = ("control", "teleport", "reset", "teleport_reset")
EXCLUDE_STOP_FRAC = 0.10
GRAD_TOL, B_TOL = 1e-6, 1e-10
MIN_CROSSINGS = 40


# ------------------------------------------------------------------------------------------ interventions (testable)
def canonical(w1, b1, w2):
    sg = 1.0 if w2 > 0 else -1.0
    return w1 * sg, (b1 * sg) % (2 * math.pi), sg


def teleport_target(a, seed, theta, x, y):
    """The occupied branch's conditional minimiser at the current scale, in canonical coordinates."""
    from .mirror_branches import half_min
    w1, b1, w2, _ = theta
    cw, cb, sg = canonical(w1, b1, w2)
    br = int(np.sign(cw))
    _, mw, mb, _ = half_min(abs(w2), a, x, y, br, step=0.05, n_refine=16)
    return mw, mb, br


def apply_teleport(theta, target, a, x, y):
    """New (w₁, b₁, w₂, b₂) with (w₁, b₁) at the target (canonical mw, mb) mapped back, b₂ profiled.  Returns the new
    parameters and the diagnostics the validity checks need."""
    from .profiled_bnb import gap, profile
    from .width2_conditional import profile_b
    w1, b1, w2, b2 = theta
    mw, mb, br = target
    cw, cb, sg = canonical(w1, b1, w2)
    w1n = sg * mw
    b1n = sg * mb
    b1n += 2 * math.pi * round((b1 - b1n) / (2 * math.pi))
    z0 = w2 * (lambda t: t + a * np.sin(t))(w1n * x + b1n)
    b2n = profile_b(z0, y)
    resid = abs(float((0.5 * (1 + np.tanh(0.5 * (z0 + b2n)))).mean() - y.mean()))
    _, _, gw, gb, _ = profile(np.array([mw]), np.array([mb]), abs(w2), a, x, y)
    G_after = float(gap(np.array([mw]), np.array([mb]), a)[0])
    cw_new, _, _ = canonical(w1n, b1n, w2)
    return (w1n, b1n, w2, b2n), {"G_after": G_after, "branch_before": int(np.sign(cw)),
                                 "branch_after": int(np.sign(cw_new)), "grad_norm": float(max(abs(gw[0]), abs(gb[0]))),
                                 "b2_residual": resid, "teleport_distance": float(math.hypot(cw - mw, (cb - mb + math.pi) % (2 * math.pi) - math.pi))}


def reset_moments(opt_state):
    """Zero Adam's moments and step count (a deep copy of the state dict)."""
    import copy
    import torch
    st = copy.deepcopy(opt_state)
    for k, v in st["state"].items():
        v["exp_avg"] = torch.zeros_like(v["exp_avg"]); v["exp_avg_sq"] = torch.zeros_like(v["exp_avg_sq"])
        v["step"] = torch.zeros_like(v["step"]) if hasattr(v["step"], "shape") else 0.0
    return st


def reset_ok(opt_state):
    return all(float(v["exp_avg"].abs().max()) == 0 and float(v["exp_avg_sq"].abs().max()) == 0 and float(v["step"]) == 0
               for v in opt_state["state"].values())


def checks(diag):
    """Per-run validity for the teleport arms: returns ('ok' | 'exclude' | 'stop', reason)."""
    if diag["branch_after"] != diag["branch_before"]:
        return "stop", "teleport changed the mirror branch"
    if diag["grad_norm"] > GRAD_TOL or diag["b2_residual"] > B_TOL:
        return "stop", "teleport did not land on the branch minimiser"
    if diag["G_after"] > 0:
        return "exclude", "teleport itself placed the run (G > 0)"
    return "ok", ""


# ------------------------------------------------------------------------------------------ continuation
def continue_from(a, seed, theta_new, opt_state, t0):
    import torch
    from . import lag_test2 as L2
    from .phase2b_ordering import state
    f, G, xt, yt, _, _ = L2._setup(a, seed)
    theta = torch.tensor(theta_new, dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)
    opt.load_state_dict(opt_state)
    for step in range(t0 + 1, L2.BASE_BUDGET + 1):
        L2._step(theta, opt, f, xt, yt, 1.0)
        st = state(theta.detach(), f, a, G)
        if st["placement_ok"]:
            return {"cross_step": step, "w2_abs": abs(st["w2"]), "R_cross": abs(st["w2"]) * G / 2}
    return {"cross_step": np.nan, "w2_abs": np.nan, "R_cross": np.nan}


def run_one(args):
    import torch
    from . import lag_test2 as L2
    from .sample_size import _data
    a, seed = args
    ck = torch.load(L2.STATES / f"primary_a{a:.2f}_s{seed}.pt", weights_only=False)
    theta = tuple(float(v) for v in ck["theta"])
    x, y = _data(L2.N, seed)
    target = teleport_target(a, seed, theta, x, y)
    th_tel, diag = apply_teleport(theta, target, a, x, y)
    status, why = checks(diag)
    rst = reset_moments(ck["opt"])
    out = []
    for arm in ARMS:
        th = th_tel if arm.startswith("teleport") else theta
        opt = rst if arm.endswith("reset") else ck["opt"]
        row = {"a": a, "seed": seed, "arm": arm, "t_switch": ck["t_star"], "check": status if arm.startswith("teleport") else "ok",
               "check_reason": why if arm.startswith("teleport") else "", "reset_ok": reset_ok(rst), **diag}
        if row["check"] == "stop":
            row.update(cross_step=np.nan, w2_abs=np.nan, R_cross=np.nan)
        else:
            row.update(continue_from(a, seed, th, opt, ck["t_star"]))
        out.append(row)
    return out


def run(workers=2):
    from multiprocessing import Pool
    ck = pd.read_csv(RESULTS / "lag_test2_checkpoints.csv")
    jobs = [(round(float(r.a), 2), int(r.seed)) for r in ck.itertuples() if r.status_primary == "ok"]
    done = set() if not PARTS.exists() else {(round(r.a, 2), int(r.seed)) for r in pd.read_csv(PARTS).itertuples()}
    jobs = [j for j in jobs if j not in done]
    with Pool(workers) as p:
        for rows in p.imap_unordered(run_one, jobs, chunksize=1):
            pd.DataFrame(rows).to_csv(PARTS, mode="a", header=not PARTS.exists(), index=False)
            if any(r["check"] == "stop" for r in rows):
                print("STOP:", rows[1]["check_reason"], rows[0]["a"], rows[0]["seed"], flush=True)
                raise SystemExit(2)


# ------------------------------------------------------------------------------------------ scoring
def control_reproduces(d, ref):
    """Validity: the control arm reproduces the lag test's φ = 1 primary continuation bit for bit."""
    c = d[d.arm == "control"][["a", "seed", "cross_step", "w2_abs"]]
    m = c.merge(ref[["a", "seed", "cross_step", "w2_abs"]], on=["a", "seed"], suffixes=("", "_ref"))
    return len(m) == len(c) and len(m) > 0 and all(np.array_equal(m[k].values, m[k + "_ref"].values, equal_nan=True)
                                                    for k in ("cross_step", "w2_abs"))


def _boot_diff(ctrl, arm, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ctrl), (n, len(ctrl)))
    diffs = np.median(ctrl[idx], axis=1) - np.median(arm[idx], axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def verdict(res):
    """res: {arm: (median residual, ci_lo, ci_hi of control − arm)} incl. 'control'."""
    ctrl = res["control"][0]
    removes = lambda arm: res[arm][0] <= 0.5 * ctrl and res[arm][1] > 0
    tel, rst = removes("teleport"), removes("reset")
    if tel and not rst:
        return "ID supported (inherited displacement)"
    if rst and not tel:
        return "OM supported (optimiser memory)"
    if tel and rst:
        return "both remove >= half; neither hypothesis alone supported"
    return "competing: neither removes half; mechanism unresolved"


def score():
    from .lag_test2 import _own
    d = pd.read_csv(PARTS, float_precision="round_trip")
    own = _own()
    d["R_own"] = [own.loc[(round(a, 2), int(s))] for a, s in zip(d.a, d.seed)]
    d["residual"] = d.R_cross / d.R_own - 1
    ref = pd.read_csv(RESULTS / "lag_test2_runs.csv", float_precision="round_trip")
    ref = ref[(ref.rule == "primary") & (ref.factor == 1.0)]
    rows = [{"check": "control reproduces the lag test's phi = 1 continuation bit for bit",
             "pass": control_reproduces(d, ref)}]
    out = []
    for a, g in d.groupby(d.a.round(2)):
        excl = g[(g.arm == "teleport") & (g.check == "exclude")].seed.unique()
        rows.append({"check": f"a={a}: excluded runs (teleport placed) {len(excl)} of {g.seed.nunique()}",
                     "pass": len(excl) <= EXCLUDE_STOP_FRAC * g.seed.nunique()})
        keep = g[~g.seed.isin(excl)]
        piv = keep.pivot_table(index="seed", columns="arm", values="residual")
        both = piv.dropna()
        res = {}
        for arm in ARMS:
            med = float(both[arm].median())
            ci = _boot_diff(both["control"].values, both[arm].values) if arm != "control" else (np.nan, np.nan)
            res[arm] = (med, ci[0], ci[1])
            out.append({"a": a, "arm": arm, "n_crossed": int(keep[keep.arm == arm].R_cross.notna().sum()),
                        "median_residual": med, "control_minus_arm_ci_lo": ci[0], "control_minus_arm_ci_hi": ci[1],
                        "sufficient": int(keep[keep.arm == arm].R_cross.notna().sum()) >= MIN_CROSSINGS})
        out.append({"a": a, "arm": "VERDICT", "verdict": verdict(res)})
    pd.DataFrame(rows).to_csv(RESULTS / "residual_mechanism_checks.csv", index=False)
    pd.DataFrame(out).to_csv(RESULTS / "residual_mechanism_scores.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    else:
        score()
