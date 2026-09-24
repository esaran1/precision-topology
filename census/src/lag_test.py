"""Design (not registered, not run): is the free-training residual adiabatic lag of w2?  results/lag_test_design.md.

Cells: a ∈ {1.30, 1.50} × w2 learning-rate factor φ ∈ {0.25, 0.5, 1, 2} at n = 6,400, on the size test's 50 seeds
(300,000-300,049; the same training sets, whose own thresholds sample_size_own.csv already holds).  Everything else is
the phase 2b protocol: float64, full batch, Adam lr 1e-2, torch.manual_seed(seed) then U(-1, 1)^4.  w2's learning rate is
1e-2·φ: after each standard Adam step, w2's update is rescaled by φ (identical to a per-parameter learning rate; at φ = 1
no operation, so φ = 1 is the reference runs' code path exactly).  Budget 32,000/φ.  Crossing: the first step with dense-grid
oriented gap > 0; R_cross = |w2| Ĝ_cert/2.

    python -m src.lag_test reproduce [workers]     # validity: φ = 1 against the size test's n = 6,400 runs, exactly
    python -m src.lag_test run [workers]
    python -m src.lag_test score
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.30, 1.50)
FACTORS = (0.25, 0.5, 1.0, 2.0)
N = 6400
SEED0, N_SEEDS = 300_000, 50
BASE_BUDGET = 32_000
MIN_CROSSINGS = 40
PLATEAU_W1 = 0.05                  # |w1| below this: the constant-predictor plateau
MATCH_RULE = 0.90                  # branch rule (registered before any mirror output was read)
L2_TOL = 0.20                      # L2: residual(φ)/residual(1) within φ ± 0.20


def branch_rule(results=None):
    """Registered rule: the initialisation-selected branch's own threshold is primary iff, in the mirror analysis, the
    initialisation-selected branch matches the branch at the free-training crossing in >= 90% of runs, pooled over its
    free-training groups (Block 4/5 first placements; phase 2b crossings at a = 1.30 and 1.50).  Otherwise the global
    own threshold is primary.  Both are reported either way."""
    occ = pd.read_csv((results or RESULTS) / "mirror_occupancy.csv")
    rate = float((occ.branch == occ.init_branch).mean())
    return ("init-selected branch" if rate >= MATCH_RULE else "global"), rate, len(occ)


def _bt_job(args):
    from .mirror_branches import branch_threshold
    from .sample_size import _data, _pop
    a, seed = args
    data = _data(N, seed)
    w2p = _pop(a)[1]
    return {"a": a, "seed": seed, "T_plus": branch_threshold(a, seed, +1, w2p, data, step=0.05, n_refine=16),
            "T_minus": branch_threshold(a, seed, -1, w2p, data, step=0.05, n_refine=16)}


def branch_thresholds(workers=3):
    """Both mirror branches' own thresholds for the 100 training sets (size-test settings: grid 0.05, 16 refinements)."""
    from .mirror_branches import init_sign
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_bt_job, [(a, s) for a in A_VALUES for s in range(SEED0, SEED0 + N_SEEDS)], chunksize=1))
    d["init_branch"] = [init_sign(s) for s in d.seed]
    d.to_csv(RESULTS / "lag_test_branch_thresholds.csv", index=False)


def _diag_job(args):
    """Distance, at the crossing, from (w1, b1) to the branch minimiser of the run's own objective at the current scale
    (canonical orientation w2 > 0, b1 mod 2π; same mirror branch as the run)."""
    import math
    from .mirror_branches import half_min
    from .sample_size import _data
    a, seed, factor, w1, b1, w2 = args
    x, y = _data(N, seed)
    sg = np.sign(w2)
    cw, cb = w1 * sg, (b1 * sg) % (2 * math.pi)
    br = int(np.sign(cw))
    _, mw, mb, _ = half_min(abs(w2), a, x, y, br, step=0.05, n_refine=16)
    db = abs(cb - mb) % (2 * math.pi)
    return {"a": a, "seed": seed, "factor": factor, "branch_at_cross": br,
            "dist_to_branch_min": math.hypot(cw - mw, min(db, 2 * math.pi - db))}


def diagnostics(workers=3):
    r = pd.read_csv(RESULTS / "lag_test_runs.csv", float_precision="round_trip")
    r = r[r.R_cross.notna()]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_diag_job, [(x.a, int(x.seed), x.factor, x.w1, x.b1, x.w2) for x in r.itertuples()],
                               chunksize=2))
    d.to_csv(RESULTS / "lag_test_diagnostics.csv", index=False)


def scale_w2_step(theta, w2_before, factor):
    """w2's learning rate x factor: after the standard Adam step, w2's update is rescaled by factor (Adam's moment
    estimates are unaffected, as with a per-parameter learning rate).  factor = 1: no operation at all."""
    if factor != 1.0:
        import torch
        with torch.no_grad():
            theta[2] = w2_before + factor * (float(theta.detach()[2]) - w2_before)


def run_one(args):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation
    from .phase2b_ordering import state
    from .sample_size import _data, _pop
    a, seed, factor = args
    torch.set_num_threads(1)
    G = _pop(a)[0]
    f = activation("sin_family", a)
    xt, yt = (torch.tensor(v) for v in _data(N, seed))
    from .fold1d import logits
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)                          # the reference runs' optimiser, unchanged
    budget = int(round(BASE_BUDGET / factor))
    plateau_steps = 0
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, xt, f), yt).backward()
        w2_before = float(theta.detach()[2])
        opt.step()
        scale_w2_step(theta, w2_before, factor)
        st = state(theta.detach(), f, a, G)
        if abs(st["w1"]) < PLATEAU_W1:
            plateau_steps += 1
        if st["placement_ok"]:
            return {"a": a, "seed": seed, "factor": factor, "budget": budget, "cross_step": step,
                    "w2_abs": abs(st["w2"]), "R_cross": abs(st["w2"]) * G / 2, "w1": st["w1"], "b1": st["b1"],
                    "w2": st["w2"], "plateau_steps": plateau_steps}
    return {"a": a, "seed": seed, "factor": factor, "budget": budget, "cross_step": np.nan, "w2_abs": np.nan,
            "R_cross": np.nan, "w1": np.nan, "b1": np.nan, "w2": np.nan, "plateau_steps": plateau_steps}


def check_reproduce(results=None, name="lag_test_reproduce.csv"):
    """Validity: φ = 1 must reproduce the size test's n = 6,400 runs exactly -- crossing step and |w2| at crossing
    bit-identical (exact round-trip parsing on both sides), non-crossers on both sides."""
    R = results or RESULTS
    ref = pd.read_csv(R / "sample_size_free.csv", float_precision="round_trip")
    ref = ref[ref.n == N][["a", "seed", "cross_step", "w2_abs"]]
    new = pd.read_csv(R / name, float_precision="round_trip")
    new = new[new.factor == 1.0][["a", "seed", "cross_step", "w2_abs"]]
    m = ref.merge(new, on=["a", "seed"], suffixes=("_ref", "_new"))
    same = len(m) == len(new) and len(m) > 0 and all(
        np.array_equal(m[c + "_ref"].values, m[c + "_new"].values, equal_nan=True) for c in ("cross_step", "w2_abs"))
    if not same:
        raise SystemExit(f"STOP: phi = 1 does not reproduce the standard runs exactly ({len(m)} rows compared)")
    return len(m)


def reproduce(workers=3, seeds=None):
    seeds = seeds if seeds is not None else range(SEED0, SEED0 + N_SEEDS)
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(run_one, [(a, s, 1.0) for a in A_VALUES for s in seeds], chunksize=1))
    d.to_csv(RESULTS / "lag_test_reproduce.csv", index=False)
    print("rows reproduced exactly:", check_reproduce())


def run(workers=3):
    """All cells; φ = 1 included, and checked for exact reproduction before anything else is used."""
    jobs = [(a, s, fct) for fct in FACTORS for a in A_VALUES for s in range(SEED0, SEED0 + N_SEEDS)]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(run_one, jobs, chunksize=1))
    d.to_csv(RESULTS / "lag_test_runs.csv", index=False)
    print("phi = 1 reproduced exactly on", check_reproduce(name="lag_test_runs.csv"), "runs")


def _boot(v, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return np.median(rng.choice(v, (B, len(v))), axis=1)


def score(results=None):
    """Registered analysis (see the design): residual(φ) = median(R_cross/R_own) - 1 per (a, φ), R_own the own
    threshold (sample_size_own.csv at n = 6,400; branch-matched if registered so)."""
    R = results or RESULTS
    check_reproduce(R, "lag_test_runs.csv")
    runs = pd.read_csv(R / "lag_test_runs.csv", float_precision="round_trip")
    own = pd.read_csv(R / "sample_size_own.csv", float_precision="round_trip")
    own = own[own.n == N][["a", "seed", "w2_own"]]
    d = runs.merge(own, on=["a", "seed"])
    bt_f = R / "lag_test_branch_thresholds.csv"
    if bt_f.exists():
        bt = pd.read_csv(bt_f, float_precision="round_trip")
        bt["w2_init_branch"] = np.where(bt.init_branch > 0, bt.T_plus, bt.T_minus)
        d = d.merge(bt[["a", "seed", "w2_init_branch"]], on=["a", "seed"], how="left")
        primary, rate, n_occ = branch_rule(R)
    else:
        d["w2_init_branch"] = np.nan
        primary, rate, n_occ = "global", np.nan, 0
    from .sample_size import _pop
    cells, tests, pairs = [], [], []
    for thr_label, col in (("global", "w2_own"), ("init-selected branch", "w2_init_branch")):
        if d[col].isna().all():
            continue
        for a in A_VALUES:
            G = _pop(a)[0]
            v, boots = {}, {}
            for fct in FACTORS:
                g = d[(d.a.round(2) == a) & (d.factor == fct) & d.R_cross.notna() & d[col].notna()]
                r = (g.R_cross / (g[col] * G / 2)).values
                v[fct], boots[fct] = float(np.median(r) - 1), _boot(r) - 1
                n_runs = int(((d.a.round(2) == a) & (d.factor == fct)).sum())
                cells.append({"threshold": thr_label, "primary": thr_label == primary, "a": a, "factor": fct,
                              "n_crossed": len(r), "n_runs": n_runs, "sufficient": len(r) >= MIN_CROSSINGS,
                              "residual": v[fct], "ci_lo": float(np.percentile(boots[fct], 2.5)),
                              "ci_hi": float(np.percentile(boots[fct], 97.5))})
            d1 = np.percentile(boots[1.0] - boots[0.25], [2.5, 97.5])
            d2 = np.percentile(boots[2.0] - boots[0.25], [2.5, 97.5])
            l1 = (v[0.25] < v[0.5] < v[1.0] < v[2.0]) and d1[0] > 0 and v[0.25] <= 0.5 * v[1.0]
            r05, r025 = v[0.5] / v[1.0], v[0.25] / v[1.0]
            l2 = abs(r05 - 0.5) <= L2_TOL and abs(r025 - 0.25) <= L2_TOL
            tests.append({"threshold": thr_label, "primary": thr_label == primary, "a": a,
                          "res_1_minus_025_lo": d1[0], "res_1_minus_025_hi": d1[1],
                          "res_2_minus_025_lo": d2[0], "res_2_minus_025_hi": d2[1],
                          "L1_pass": l1, "competing_no_dependence": bool(d2[0] <= 0 <= d2[1]),
                          "ratio_05": r05, "ratio_025": r025, "L2_pass": l2})
            order = list(FACTORS)
            for i in range(len(order) - 1):
                lo_f, hi_f = order[i], order[i + 1]
                strict = v[lo_f] < v[hi_f]
                ci_a = np.percentile(boots[lo_f], [2.5, 97.5]); ci_b = np.percentile(boots[hi_f], [2.5, 97.5])
                overlap = ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]
                dd = np.percentile(boots[hi_f] - boots[lo_f], [2.5, 97.5])
                pairs.append({"threshold": thr_label, "a": a, "factor_small": lo_f, "factor_large": hi_f,
                              "difference": v[hi_f] - v[lo_f], "ci_lo": dd[0], "ci_hi": dd[1], "strictly_increasing": strict,
                              "note": "" if strict else ("registered ordering fails; the two values' intervals overlap: "
                                                         "indistinguishable" if overlap else "registered ordering fails")})
    diag_f = R / "lag_test_diagnostics.csv"
    conf = []
    for (a, fct), g in d.groupby([d.a.round(2), "factor"]):
        x = g[g.R_cross.notna()]
        canon = np.sign(x.w1) * np.sign(x.w2)
        row = {"a": a, "factor": fct, "frac_branch_plus_at_cross": float((canon > 0).mean()) if len(x) else np.nan,
               "median_plateau_steps": float(g.plateau_steps.median()), "frac_with_plateau_steps": float((g.plateau_steps > 0).mean())}
        if diag_f.exists():
            dg = pd.read_csv(diag_f, float_precision="round_trip")
            dg = dg[(dg.a.round(2) == a) & (dg.factor == fct)]
            row["median_dist_to_branch_min"] = float(dg.dist_to_branch_min.median())
        conf.append(row)
    pd.DataFrame(cells).to_csv(R / "lag_test_cells.csv", index=False)
    pd.DataFrame(tests).assign(rule_match_rate=rate, rule_n=n_occ).to_csv(R / "lag_test_tests.csv", index=False)
    pd.DataFrame(pairs).to_csv(R / "lag_test_pairs.csv", index=False)
    pd.DataFrame(conf).to_csv(R / "lag_test_confounds.csv", index=False)
    print("primary threshold:", primary, "(match rate", rate, ")")
    print(pd.DataFrame(cells).to_string(index=False)); print(pd.DataFrame(tests).T.to_string())
    print(pd.DataFrame(pairs).to_string(index=False)); print(pd.DataFrame(conf).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("LT_WORKERS", "3"))
    {"reproduce": lambda: reproduce(w), "run": lambda: run(w), "score": score,
     "branch_thresholds": lambda: branch_thresholds(w), "diagnostics": lambda: diagnostics(w)}[cmd]()
