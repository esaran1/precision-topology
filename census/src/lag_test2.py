"""Deconfounded adiabatic-lag test (registration: results/lag_test2_prediction.md).

The w2 learning-rate factor φ is switched only **after** the run has left the constant-predictor plateau, so the
plateau-time confound of the first lag test (lag_test_prediction.md) cannot act.

Runs: the size test's n = 6,400 training sets, a ∈ {1.30, 1.50}, seeds 300,000-300,049, standard protocol (φ = 1),
which the lag test reproduced bit for bit.
  t*: the first step after the run's **last** plateau step before its crossing (plateau: |w1| < 0.05 or |w2| < 0.05);
      a run never on the plateau has t* = 1.  A run is excluded, and counted, if R(t*) > 0.8·R_own (it left the plateau
      too close to its own threshold) or t* >= its crossing step.
  Checkpoint: the full parameter and Adam state after step t* (φ = 1), saved to results/lag_test2_states/.
  Continuation: from the restored checkpoint, w2's learning rate x φ ∈ {0.25, 0.5, 1, 2} (lag_test.scale_w2_step),
      everything else unchanged, for ceil((32,000 − t*)/φ) further steps.

    python -m src.lag_test2 checkpoints [workers]
    python -m src.lag_test2 continue [workers]
    python -m src.lag_test2 diagnostics [workers]
    python -m src.lag_test2 score
"""

from __future__ import annotations

import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
STATES = RESULTS / "lag_test2_states"
A_VALUES = (1.30, 1.50)
FACTORS = (0.25, 0.5, 1.0, 2.0)
N = 6400
SEED0, N_SEEDS = 300_000, 50
BASE_BUDGET = 32_000
PLATEAU = 0.05
R_CAP = 0.8
MIN_CROSSINGS = 40
L2_TOL = 0.20


def on_plateau(w1, w2):
    return abs(w1) < PLATEAU or abs(w2) < PLATEAU


def choose_tstar(plateau_flags, R_path, cross_step, R_own):
    """plateau_flags[k], R_path[k]: after step k+1.  Returns (t*, status)."""
    upto = cross_step - 1                                   # steps strictly before the crossing
    idx = [k for k in range(upto) if plateau_flags[k]]
    t = 1 if not idx else idx[-1] + 2                        # first step after the last plateau step
    if t >= cross_step:
        return t, "excluded: leaves the plateau at or after crossing"
    if R_path[t - 1] > R_CAP * R_own:
        return t, "excluded: R(t*) > 0.8 R_own"
    return t, "ok"


def _setup(a, seed):
    import torch
    from .fold1d import activation
    from .sample_size import _data, _pop
    torch.set_num_threads(1)
    G = _pop(a)[0]
    f = activation("sin_family", a)
    xt, yt = (torch.tensor(v) for v in _data(N, seed))
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)
    return f, G, xt, yt, theta, opt


def _step(theta, opt, f, xt, yt, factor):
    from torch.nn import functional as F
    from .fold1d import logits
    from .lag_test import scale_w2_step
    opt.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(logits(theta, xt, f), yt).backward()
    w2_before = float(theta.detach()[2])
    opt.step()
    scale_w2_step(theta, w2_before, factor)


def checkpoint_one(args):
    """Pass 1: the φ = 1 trajectory to its crossing (plateau flags, R).  Pass 2: rerun to t* and save the state."""
    import copy
    import torch
    from .phase2b_ordering import state
    a, seed, R_own = args
    f, G, xt, yt, theta, opt = _setup(a, seed)
    flags, Rp, cross = [], [], None
    for step in range(1, BASE_BUDGET + 1):
        _step(theta, opt, f, xt, yt, 1.0)
        st = state(theta.detach(), f, a, G)
        flags.append(on_plateau(st["w1"], st["w2"])); Rp.append(abs(st["w2"]) * G / 2)
        if st["placement_ok"]:
            cross = step; break
    if cross is None:
        return {"a": a, "seed": seed, "status": "excluded: no crossing", "t_star": np.nan}
    t, status = choose_tstar(flags, Rp, cross, R_own)
    out = {"a": a, "seed": seed, "status": status, "t_star": t, "cross_step_phi1": cross, "R_t_star": Rp[t - 1],
           "R_own": R_own, "plateau_steps_before": int(sum(flags[:t]))}
    if status == "ok":
        f, G, xt, yt, theta, opt = _setup(a, seed)
        for _ in range(t):
            _step(theta, opt, f, xt, yt, 1.0)
        STATES.mkdir(exist_ok=True)
        torch.save({"theta": theta.detach().clone(), "opt": copy.deepcopy(opt.state_dict()), "t_star": t},
                   STATES / f"a{a:.2f}_s{seed}.pt")
    return out


def continue_one(args):
    import torch
    from .phase2b_ordering import state
    a, seed, factor = args
    f, G, xt, yt, _, _ = _setup(a, seed)
    ck = torch.load(STATES / f"a{a:.2f}_s{seed}.pt", weights_only=False)
    theta = ck["theta"].clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)
    opt.load_state_dict(ck["opt"])
    t = ck["t_star"]
    total = t + math.ceil((BASE_BUDGET - t) / factor)
    reentry = 0
    for step in range(t + 1, total + 1):
        _step(theta, opt, f, xt, yt, factor)
        st = state(theta.detach(), f, a, G)
        reentry += on_plateau(st["w1"], st["w2"])
        if st["placement_ok"]:
            return {"a": a, "seed": seed, "factor": factor, "t_star": t, "cross_step": step, "w2_abs": abs(st["w2"]),
                    "R_cross": abs(st["w2"]) * G / 2, "w1": st["w1"], "b1": st["b1"], "w2": st["w2"],
                    "plateau_reentry_steps": reentry}
    return {"a": a, "seed": seed, "factor": factor, "t_star": t, "cross_step": np.nan, "w2_abs": np.nan,
            "R_cross": np.nan, "w1": np.nan, "b1": np.nan, "w2": np.nan, "plateau_reentry_steps": reentry}


def check_continuation(results=None, name="lag_test2_runs.csv"):
    """Validity: φ = 1 continuations reproduce the original runs (sample_size_free.csv, n = 6,400) bit for bit --
    crossing step and |w2| at crossing, exact round-trip parsing on both sides, every continued run."""
    R = results or RESULTS
    ref = pd.read_csv(R / "sample_size_free.csv", float_precision="round_trip")
    ref = ref[ref.n == N][["a", "seed", "cross_step", "w2_abs"]]
    new = pd.read_csv(R / name, float_precision="round_trip")
    new = new[new.factor == 1.0][["a", "seed", "cross_step", "w2_abs"]]
    m = new.merge(ref, on=["a", "seed"], suffixes=("_new", "_ref"), how="left")
    same = len(m) > 0 and all(np.array_equal(m[c + "_new"].values, m[c + "_ref"].values, equal_nan=True)
                              for c in ("cross_step", "w2_abs"))
    if not same:
        raise SystemExit(f"STOP: phi = 1 continuations do not reproduce the original runs ({len(m)} rows)")
    return len(m)


def _own():
    from .sample_size import _pop
    o = pd.read_csv(RESULTS / "sample_size_own.csv", float_precision="round_trip")
    o = o[o.n == N].copy()
    o["R_own"] = [w * _pop(a)[0] / 2 for a, w in zip(o.a, o.w2_own)]
    return o.set_index(["a", "seed"]).R_own


def checkpoints(workers=3):
    own = _own()
    jobs = [(a, s, float(own.loc[(a, s)])) for a in A_VALUES for s in range(SEED0, SEED0 + N_SEEDS)]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(checkpoint_one, jobs, chunksize=1))
    d.to_csv(RESULTS / "lag_test2_checkpoints.csv", index=False)
    print(d.status.value_counts().to_string())


def cont(workers=3):
    ck = pd.read_csv(RESULTS / "lag_test2_checkpoints.csv")
    ok = ck[ck.status == "ok"]
    jobs = [(r.a, int(r.seed), fct) for fct in FACTORS for r in ok.itertuples()]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(continue_one, jobs, chunksize=1))
    d.to_csv(RESULTS / "lag_test2_runs.csv", index=False)
    print("phi = 1 continuations reproduced exactly:", check_continuation())


def diagnostics(workers=3):
    from .lag_test import _diag_job
    r = pd.read_csv(RESULTS / "lag_test2_runs.csv", float_precision="round_trip")
    r = r[r.R_cross.notna()]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_diag_job, [(x.a, int(x.seed), x.factor, x.w1, x.b1, x.w2) for x in r.itertuples()],
                               chunksize=2))
    d.to_csv(RESULTS / "lag_test2_diagnostics.csv", index=False)


def _boot(v, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return np.median(rng.choice(v, (B, len(v))), axis=1)


def score(results=None):
    R = results or RESULTS
    check_continuation(R)
    d = pd.read_csv(R / "lag_test2_runs.csv", float_precision="round_trip")
    own = pd.read_csv(R / "sample_size_own.csv", float_precision="round_trip")
    d = d.merge(own[own.n == N][["a", "seed", "w2_own"]], on=["a", "seed"])
    from .sample_size import _pop
    cells, tests, pairs, conf = [], [], [], []
    for a in A_VALUES:
        G = _pop(a)[0]
        v, boots = {}, {}
        for fct in FACTORS:
            g = d[(d.a.round(2) == a) & (d.factor == fct)]
            x = g[g.R_cross.notna()]
            r = (x.R_cross / (x.w2_own * G / 2)).values
            v[fct], boots[fct] = float(np.median(r) - 1), _boot(r) - 1
            cells.append({"a": a, "factor": fct, "n_runs": len(g), "n_crossed": len(x), "sufficient": len(x) >= MIN_CROSSINGS,
                          "residual": v[fct], "ci_lo": float(np.percentile(boots[fct], 2.5)),
                          "ci_hi": float(np.percentile(boots[fct], 97.5))})
            canon = np.sign(x.w1) * np.sign(x.w2)
            conf.append({"a": a, "factor": fct, "median_plateau_reentry": float(g.plateau_reentry_steps.median()),
                         "frac_with_reentry": float((g.plateau_reentry_steps > 0).mean()),
                         "frac_branch_plus_at_cross": float((canon > 0).mean())})
        d1 = np.percentile(boots[1.0] - boots[0.25], [2.5, 97.5])
        d2 = np.percentile(boots[2.0] - boots[0.25], [2.5, 97.5])
        l1 = (v[0.25] < v[0.5] < v[1.0] < v[2.0]) and d1[0] > 0 and v[0.25] <= 0.5 * v[1.0]
        r05, r025 = v[0.5] / v[1.0], v[0.25] / v[1.0]
        tests.append({"a": a, "res_1_minus_025_lo": d1[0], "res_1_minus_025_hi": d1[1], "res_2_minus_025_lo": d2[0],
                      "res_2_minus_025_hi": d2[1], "L1p_pass": l1, "competing_no_dependence": bool(d2[0] <= 0 <= d2[1]),
                      "ratio_05": r05, "ratio_025": r025,
                      "L2p_pass": abs(r05 - 0.5) <= L2_TOL and abs(r025 - 0.25) <= L2_TOL})
        for i in range(len(FACTORS) - 1):
            lo_f, hi_f = FACTORS[i], FACTORS[i + 1]
            ca, cb = np.percentile(boots[lo_f], [2.5, 97.5]), np.percentile(boots[hi_f], [2.5, 97.5])
            dd = np.percentile(boots[hi_f] - boots[lo_f], [2.5, 97.5])
            strict = v[lo_f] < v[hi_f]
            pairs.append({"a": a, "factor_small": lo_f, "factor_large": hi_f, "difference": v[hi_f] - v[lo_f],
                          "ci_lo": dd[0], "ci_hi": dd[1], "strictly_increasing": strict,
                          "note": "" if strict else ("registered ordering fails; the two values' intervals overlap: "
                                                     "indistinguishable" if (ca[0] <= cb[1] and cb[0] <= ca[1])
                                                     else "registered ordering fails")})
    cf = pd.DataFrame(conf)
    dg_f = R / "lag_test2_diagnostics.csv"
    if dg_f.exists():
        dg = pd.read_csv(dg_f, float_precision="round_trip")
        cf = cf.merge(dg.groupby([dg.a.round(2), "factor"]).dist_to_branch_min.median().rename("median_dist_to_branch_min")
                      .reset_index(), on=["a", "factor"], how="left")
    pd.DataFrame(cells).to_csv(R / "lag_test2_cells.csv", index=False)
    pd.DataFrame(tests).to_csv(R / "lag_test2_tests.csv", index=False)
    pd.DataFrame(pairs).to_csv(R / "lag_test2_pairs.csv", index=False)
    cf.to_csv(R / "lag_test2_confounds.csv", index=False)
    print(pd.DataFrame(cells).to_string(index=False)); print(pd.DataFrame(tests).T.to_string())
    print(pd.DataFrame(pairs).to_string(index=False)); print(cf.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("LT2_WORKERS", "3"))
    {"checkpoints": lambda: checkpoints(w), "continue": lambda: cont(w), "diagnostics": lambda: diagnostics(w),
     "score": score}[cmd]()
