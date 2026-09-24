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

Amendment 1 (primary rule): switch at the first step where R >= max(0.7·R_own, the branch-commitment level: 0.10 at
a = 1.30, 0.15 at 1.50), provided the run's last plateau visit before its crossing is earlier; otherwise excluded and
counted.  The t* rule above is kept as a secondary analysis.  Both are scored overall and by stratum (plateau visit or
not).

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
SWITCH_FRAC = 0.7                           # amendment 1: primary switch at R >= 0.7 R_own ...
COMMIT = {1.30: 0.10, 1.50: 0.15}           # ... but never below the branch-commitment level at that a
RULES = ("primary", "tstar")
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


def choose_switch(plateau_flags, R_path, cross_step, R_own, commit):
    """Amendment 1, the primary rule: the first step with R >= max(0.7 R_own, commit), provided the run's last plateau
    visit before its crossing comes earlier.  Returns (t, status, target, used_commit)."""
    target = max(SWITCH_FRAC * R_own, commit)
    used_commit = commit >= SWITCH_FRAC * R_own
    hits = [k for k in range(cross_step - 1) if R_path[k] >= target]
    if not hits:
        return None, "excluded: crosses before reaching the switch level", target, used_commit
    t = hits[0] + 1
    idx = [k for k in range(cross_step - 1) if plateau_flags[k]]
    if idx and idx[-1] + 1 >= t:
        return t, "excluded: last plateau visit at or after the switch point", target, used_commit
    return t, "ok", target, used_commit


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
        return {"a": a, "seed": seed, "status_tstar": "excluded: no crossing", "status_primary": "excluded: no crossing"}
    ts, st_s = choose_tstar(flags, Rp, cross, R_own)
    tp, st_p, target, used_commit = choose_switch(flags, Rp, cross, R_own, COMMIT[round(a, 2)])
    out = {"a": a, "seed": seed, "cross_step_phi1": cross, "R_own": R_own,
           "ever_plateau": bool(any(flags[:cross - 1])), "plateau_steps_total": int(sum(flags[:cross - 1])),
           "status_tstar": st_s, "t_tstar": ts, "R_at_tstar": Rp[ts - 1] if ts <= len(Rp) else np.nan,
           "status_primary": st_p, "t_primary": tp, "switch_target_R": target, "used_commit_level": used_commit,
           "R_at_primary": Rp[tp - 1] if tp is not None else np.nan}
    for rule, t, stt in (("tstar", ts, st_s), ("primary", tp, st_p)):
        if stt != "ok":
            continue
        f, G, xt, yt, theta, opt = _setup(a, seed)
        for _ in range(t):
            _step(theta, opt, f, xt, yt, 1.0)
        STATES.mkdir(exist_ok=True)
        torch.save({"theta": theta.detach().clone(), "opt": copy.deepcopy(opt.state_dict()), "t_star": t},
                   STATES / f"{rule}_a{a:.2f}_s{seed}.pt")
    return out


def continue_one(args):
    import torch
    from .phase2b_ordering import state
    a, seed, factor, rule = args
    f, G, xt, yt, _, _ = _setup(a, seed)
    ck = torch.load(STATES / f"{rule}_a{a:.2f}_s{seed}.pt", weights_only=False)
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
            return {"rule": rule, "a": a, "seed": seed, "factor": factor, "t_star": t, "cross_step": step, "w2_abs": abs(st["w2"]),
                    "R_cross": abs(st["w2"]) * G / 2, "w1": st["w1"], "b1": st["b1"], "w2": st["w2"],
                    "plateau_reentry_steps": reentry}
    return {"rule": rule, "a": a, "seed": seed, "factor": factor, "t_star": t, "cross_step": np.nan, "w2_abs": np.nan,
            "R_cross": np.nan, "w1": np.nan, "b1": np.nan, "w2": np.nan, "plateau_reentry_steps": reentry}


def check_continuation(results=None, name="lag_test2_runs.csv"):
    """Validity: φ = 1 continuations reproduce the original runs (sample_size_free.csv, n = 6,400) bit for bit --
    crossing step and |w2| at crossing, exact round-trip parsing on both sides, every continued run."""
    R = results or RESULTS
    ref = pd.read_csv(R / "sample_size_free.csv", float_precision="round_trip")
    ref = ref[ref.n == N][["a", "seed", "cross_step", "w2_abs"]]
    new = pd.read_csv(R / name, float_precision="round_trip")
    new = new[new.factor == 1.0][["a", "seed", "cross_step", "w2_abs"]]          # every rule's phi = 1 continuations
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


PARTS = RESULTS / "lag_test2_parts"


def _append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def checkpoints(workers=3):
    """Rows are appended as each run finishes; a restart skips finished runs (crash resilience only)."""
    PARTS.mkdir(exist_ok=True)
    pf = PARTS / "checkpoints.csv"
    done = set() if not pf.exists() else {(round(a, 2), int(s)) for a, s in zip(*[pd.read_csv(pf)[c] for c in ("a", "seed")])}
    own = _own()
    jobs = [(a, s, float(own.loc[(a, s)])) for a in A_VALUES for s in range(SEED0, SEED0 + N_SEEDS) if (a, s) not in done]
    with Pool(workers) as p:
        for r in p.imap_unordered(checkpoint_one, jobs, chunksize=1):
            _append(pf, r)
    d = pd.read_csv(pf).sort_values(["a", "seed"]).reset_index(drop=True)
    assert len(d) == len(A_VALUES) * N_SEEDS
    d.to_csv(RESULTS / "lag_test2_checkpoints.csv", index=False)
    print(d.status_primary.value_counts().to_string()); print(d.status_tstar.value_counts().to_string())
    print("runs using the commitment level instead of 0.7 R_own:", int(d.used_commit_level.fillna(False).sum()))


def cont(workers=3):
    ck = pd.read_csv(RESULTS / "lag_test2_checkpoints.csv")
    PARTS.mkdir(exist_ok=True)
    pf = PARTS / "runs.csv"
    done = set() if not pf.exists() else {(r_, round(a, 2), int(sd), f_) for r_, a, sd, f_ in
                                          zip(*[pd.read_csv(pf)[c] for c in ("rule", "a", "seed", "factor")])}
    jobs = [(r.a, int(r.seed), fct, rule) for rule in RULES for fct in FACTORS
            for r in ck[ck[f"status_{rule}"] == "ok"].itertuples() if (rule, round(r.a, 2), int(r.seed), fct) not in done]
    with Pool(workers) as p:
        for r in p.imap_unordered(continue_one, jobs, chunksize=1):
            _append(pf, r)
    d = pd.read_csv(pf, float_precision="round_trip").sort_values(["rule", "a", "factor", "seed"]).reset_index(drop=True)
    d.to_csv(RESULTS / "lag_test2_runs.csv", index=False)
    print("phi = 1 continuations reproduced exactly:", check_continuation())


def diagnostics(workers=3):
    from .lag_test import _diag_job
    r = pd.read_csv(RESULTS / "lag_test2_runs.csv", float_precision="round_trip")
    r = r[r.R_cross.notna()]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_diag_job, [(x.a, int(x.seed), x.factor, x.w1, x.b1, x.w2) for x in r.itertuples()],
                               chunksize=2))
    d["rule"] = r.rule.values
    d.to_csv(RESULTS / "lag_test2_diagnostics.csv", index=False)


def _boot(v, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return np.median(rng.choice(v, (B, len(v))), axis=1)


def _score_block(d, G_of):
    cells, tests, pairs, conf = [], [], [], []
    for a in A_VALUES:
        G = G_of(a)
        v, boots = {}, {}
        for fct in FACTORS:
            g = d[(d.a.round(2) == a) & (d.factor == fct)]
            x = g[g.R_cross.notna()]
            r = (x.R_cross / (x.w2_own * G / 2)).values
            if len(r) == 0:
                v[fct], boots[fct] = np.nan, np.full(10_000, np.nan)
            else:
                v[fct], boots[fct] = float(np.median(r) - 1), _boot(r) - 1
            cells.append({"a": a, "factor": fct, "n_runs": len(g), "n_crossed": len(x), "sufficient": len(x) >= MIN_CROSSINGS,
                          "residual": v[fct], "ci_lo": float(np.nanpercentile(boots[fct], 2.5)) if len(r) else np.nan,
                          "ci_hi": float(np.nanpercentile(boots[fct], 97.5)) if len(r) else np.nan})
            canon = np.sign(x.w1) * np.sign(x.w2)
            conf.append({"a": a, "factor": fct, "median_plateau_reentry": float(g.plateau_reentry_steps.median()) if len(g) else np.nan,
                         "frac_with_reentry": float((g.plateau_reentry_steps > 0).mean()) if len(g) else np.nan,
                         "frac_branch_plus_at_cross": float((canon > 0).mean()) if len(x) else np.nan})
        if any(np.isnan(v[f]) for f in FACTORS):
            tests.append({"a": a, "L1p_pass": np.nan, "L2p_pass": np.nan, "competing_no_dependence": np.nan, "note": "empty cell"})
            continue
        d1 = np.percentile(boots[1.0] - boots[0.25], [2.5, 97.5])
        d2 = np.percentile(boots[2.0] - boots[0.25], [2.5, 97.5])
        l1 = (v[0.25] < v[0.5] < v[1.0] < v[2.0]) and d1[0] > 0 and v[0.25] <= 0.5 * v[1.0]
        r05, r025 = v[0.5] / v[1.0], v[0.25] / v[1.0]
        tests.append({"a": a, "res_1_minus_025_lo": d1[0], "res_1_minus_025_hi": d1[1], "res_2_minus_025_lo": d2[0],
                      "res_2_minus_025_hi": d2[1], "L1p_pass": l1, "competing_no_dependence": bool(d2[0] <= 0 <= d2[1]),
                      "ratio_05": r05, "ratio_025": r025,
                      "L2p_pass": abs(r05 - 0.5) <= L2_TOL and abs(r025 - 0.25) <= L2_TOL,
                      "all_cells_sufficient": all(c["sufficient"] for c in cells if c["a"] == a)})
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
    return cells, tests, pairs, conf


def score(results=None):
    """Amendment 1: the primary switch rule is scored as the registered verdict; the t* rule is secondary.  Both are
    also reported by stratum (runs with a plateau visit before their crossing versus runs without)."""
    R = results or RESULTS
    check_continuation(R)
    d = pd.read_csv(R / "lag_test2_runs.csv", float_precision="round_trip")
    own = pd.read_csv(R / "sample_size_own.csv", float_precision="round_trip")
    d = d.merge(own[own.n == N][["a", "seed", "w2_own"]], on=["a", "seed"])
    ck_f = R / "lag_test2_checkpoints.csv"
    if ck_f.exists():
        ck = pd.read_csv(ck_f)
        d = d.merge(ck[["a", "seed", "ever_plateau"]], on=["a", "seed"], how="left")
    else:
        d["ever_plateau"] = np.nan
    from .sample_size import _pop
    G_of = lambda a: _pop(a)[0]
    outs = {k: [] for k in ("cells", "tests", "pairs", "conf")}
    for rule in RULES:
        for stratum, sel in (("all", lambda x: x), ("had plateau visit", lambda x: x[x.ever_plateau == True]),
                             ("never on plateau", lambda x: x[x.ever_plateau == False])):
            sub = sel(d[d.rule == rule])
            if not len(sub):
                continue
            c, t, pr, cf = _score_block(sub, G_of)
            tag = {"rule": rule, "role": "primary (verdict)" if (rule == "primary" and stratum == "all") else "reported",
                   "stratum": stratum}
            for key, rows in (("cells", c), ("tests", t), ("pairs", pr), ("conf", cf)):
                outs[key] += [{**tag, **r} for r in rows]
    cf = pd.DataFrame(outs["conf"])
    dg_f = R / "lag_test2_diagnostics.csv"
    if dg_f.exists() and len(cf):
        dg = pd.read_csv(dg_f, float_precision="round_trip")
        med = dg.groupby(["rule", dg.a.round(2), "factor"]).dist_to_branch_min.median().rename("median_dist_to_branch_min").reset_index()
        cf = cf.merge(med, on=["rule", "a", "factor"], how="left")
    pd.DataFrame(outs["cells"]).to_csv(R / "lag_test2_cells.csv", index=False)
    pd.DataFrame(outs["tests"]).to_csv(R / "lag_test2_tests.csv", index=False)
    pd.DataFrame(outs["pairs"]).to_csv(R / "lag_test2_pairs.csv", index=False)
    cf.to_csv(R / "lag_test2_confounds.csv", index=False)
    print(pd.DataFrame(outs["tests"]).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("LT2_WORKERS", "3"))
    {"checkpoints": lambda: checkpoints(w), "continue": lambda: cont(w), "diagnostics": lambda: diagnostics(w),
     "score": score}[cmd]()
