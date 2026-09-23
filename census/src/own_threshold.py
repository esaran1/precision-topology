"""Own-seed conditional thresholds (registered in results/own_threshold_prediction.md, 312829f).

For a training set (x, y) = fold1d.make_data(200, seed), at output scale s = |w2|: the global minimiser of
the profiled loss L*(w1, b1; s) (b2 profiled exactly, src/profiled_bnb.profile) is found by a dense
grid (w1 in [-W, W] with W = profiled_bnb.w_bound, b1 in [0, 2π), step 0.02), then local refinement
(BFGS with backtracking on the exact gradient) of the 8 best distinct grid cells.  Placed iff the refined global
minimiser has G > 0 (exact extrema, profiled_bnb.gap).  Own w2_glob is bracketed from the population
value in steps of 0.1 and bisected to width 0.01; R = |w2| Ĝ_cert(a)/2.

Validation: the certified branch and bound (conditional_certified.evaluate) at both bracket ends for 20
Block 4 seeds (numpy default_rng(0)), and against 1d's certified seeds at a = 1.30 where they overlap.

    python -m src.own_threshold block4        # a = 1.30, the 174 seeds behind Block 4's checkpoints
    python -m src.own_threshold crossing      # a = 1.30 and 1.50, phase 2b seeds 0-39
    python -m src.own_threshold validate      # certified checks at both bracket ends, 20 Block 4 seeds
    python -m src.own_threshold score         # S1-S3 (needs the long-horizon replays)
    python -m src.own_threshold block5_posthoc  # POST HOC: Block 5 retention midpoint vs median own/pop
"""

from __future__ import annotations

import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .profiled_bnb import gap, profile, w_bound

RESULTS = Path(__file__).resolve().parents[1] / "results"
STEP = 0.02
BRACKET_STEP = 0.1
WIDTH = 0.01
N_REFINE = 8
CHUNK = 2000
WORKERS = int(os.environ.get("OT_WORKERS", "3"))


def _data(seed):
    from .fold1d import make_data
    x, y = make_data(200, seed)
    return x.double().numpy(), y.double().numpy()


def _pop(a):
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv")
    G = float(g[g.a.round(2) == round(a, 2)].Ghat_certified.iloc[0])
    t = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    r = t[(t.a.round(2) == round(a, 2)) & (t.kind == "glob")].iloc[0]
    return G, 0.5 * (r.w2_lo + r.w2_hi)


def _bfgs(fg, x0, gtol=1e-10, maxit=500):
    xk = x0.astype(float)
    f, g = fg(xk)
    H = np.eye(2) * 0.1
    for _ in range(maxit):
        if np.abs(g).max() <= gtol:
            break
        d = -H @ g
        if d @ g >= 0:
            H = np.eye(2) * 0.1; d = -H @ g
        t = 1.0
        while True:
            fn, gn = fg(xk + t * d)
            if fn <= f + 1e-4 * t * (d @ g) or t < 1e-12:
                break
            t *= 0.5
        if t < 1e-12:
            break
        sv, yv = t * d, gn - g
        xk, f, g = xk + sv, fn, gn
        sy = sv @ yv
        if sy > 1e-300:
            r = 1.0 / sy
            I = np.eye(2)
            H = (I - r * np.outer(sv, yv)) @ H @ (I - r * np.outer(yv, sv)) + r * np.outer(sv, sv)
    return float(f), xk


def global_min(s, a, x, y):
    """Refined global minimiser of L*(.,.; s): (L, w1, b1, G)."""
    W = w_bound(s, a, x, y)
    w1g = np.arange(-W, W + STEP / 2, STEP)
    b1g = np.arange(0.0, 2 * math.pi, STEP)
    Wm, Bm = np.meshgrid(w1g, b1g, indexing="ij")
    wf, bf = Wm.ravel(), Bm.ravel()
    L = np.empty(wf.size)
    for i in range(0, wf.size, CHUNK):
        L[i:i + CHUNK] = profile(wf[i:i + CHUNK], bf[i:i + CHUNK], s, a, x, y)[0]
    order = np.argsort(L)
    starts = []
    for j in order:
        if all(abs(wf[j] - w) > 0.1 or min(abs(bf[j] - b), 2 * math.pi - abs(bf[j] - b)) > 0.1
               for w, b in starts):
            starts.append((wf[j], bf[j]))
        if len(starts) == N_REFINE:
            break

    def fg(p):
        Lv, _, gw, gb, _ = profile([p[0]], [p[1]], s, a, x, y)
        return float(Lv[0]), np.array([gw[0], gb[0]])
    best = None
    for w0, b0 in starts:
        f, xk = _bfgs(fg, np.array([w0, b0]))
        if best is None or f < best[0]:
            best = (f, float(xk[0]), float(xk[1]))
    Lb, w1, b1 = best
    return Lb, w1, b1, float(gap([w1], [b1], a)[0]), float(L[order[0]])


def own_threshold(a, seed, w2_start):
    x, y = _data(seed)
    placed = lambda s: global_min(s, a, x, y)[3] > 0
    n = 0
    lo = hi = round(w2_start, 6)
    p = placed(lo); n += 1
    if p:                                       # step down until not placed
        while p:
            hi = lo; lo = round(lo - BRACKET_STEP, 6); p = placed(lo); n += 1
            if lo <= 0.5:
                return {"a": a, "seed": seed, "w2_lo": np.nan, "w2_hi": hi, "evals": n, "note": "placed down to 0.5"}
    else:                                       # step up until placed
        while not p:
            lo = hi; hi = round(hi + BRACKET_STEP, 6); p = placed(hi); n += 1
            if hi >= 20:
                return {"a": a, "seed": seed, "w2_lo": lo, "w2_hi": np.nan, "evals": n, "note": "unplaced up to 20"}
    while hi - lo > WIDTH + 1e-12:
        mid = round(0.5 * (lo + hi), 6)
        if placed(mid):
            hi = mid
        else:
            lo = mid
        n += 1
    return {"a": a, "seed": seed, "w2_lo": lo, "w2_hi": hi, "evals": n, "note": ""}


def _job(args):
    return own_threshold(*args)


def _run(jobs, out):
    with Pool(WORKERS) as p:
        rows = p.map(_job, jobs, chunksize=1)
    d = pd.DataFrame(rows)
    G = {a: _pop(a)[0] for a in d.a.unique()}
    d["Ghat_cert"] = d.a.map(G)
    d["w2_own"] = 0.5 * (d.w2_lo + d.w2_hi)
    d["R_own"] = d.w2_own * d.Ghat_cert / 2
    d["w2_pop"] = d.a.map({a: _pop(a)[1] for a in d.a.unique()})
    d["own_over_pop"] = d.w2_own / d.w2_pop
    d.to_csv(RESULTS / out, index=False)
    print(d.describe().to_string())


def block4():
    """Seeds behind Block 4's checkpoints, plus Block 5's (for the post hoc Block 5 comparison)."""
    seeds = sorted(set(pd.read_csv(RESULTS / "fixed_scale_block4.csv").seed.unique())
                   | set(pd.read_csv(RESULTS / "fixed_scale_block5.csv").seed.unique()))
    _run([(1.30, int(s), _pop(1.30)[1]) for s in seeds], "own_threshold_block4.csv")


def crossing():
    _run([(a, s, _pop(a)[1]) for a in (1.30, 1.50) for s in range(40)], "own_threshold_crossing.csv")


def _val_job(args):
    from .conditional_certified import evaluate
    seed, lo, hi = args
    x, y = _data(seed)
    rl, rh = evaluate(lo, 1.30, x, y, False), evaluate(hi, 1.30, x, y, False)
    return {"seed": seed, "w2_lo": lo, "w2_hi": hi, "cert_status_lo": rl["status"], "cert_status_hi": rh["status"],
            "agrees": rl["status"] == "minus" and rh["status"] == "plus"}


def validate():
    d = pd.read_csv(RESULTS / "own_threshold_block4.csv").dropna(subset=["w2_lo", "w2_hi"])
    pick = np.random.default_rng(0).choice(d.seed.values, 20, replace=False)
    sub = d[d.seed.isin(pick)]
    with Pool(WORKERS) as p:
        rows = p.map(_val_job, [(int(r.seed), r.w2_lo, r.w2_hi) for r in sub.itertuples()], chunksize=1)
    v = pd.DataFrame(rows)
    v.to_csv(RESULTS / "own_threshold_validation.csv", index=False)
    print(v.to_string(index=False)); print("agreement:", v.agrees.mean())


def _spearman_perm(a, b, n_perm=100_000, seed=0):
    """Spearman rho and one-sided (positive) permutation p-value."""
    ra = pd.Series(a).rank().values; rb = pd.Series(b).rank().values
    ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std()
    rho = float((ra * rb).mean())
    rng = np.random.default_rng(seed)
    perm = np.array([(ra * rng.permutation(rb)).mean() for _ in range(n_perm)])
    return rho, float((1 + (perm >= rho - 1e-12).sum()) / (n_perm + 1))


def score():
    """Registered S1-S3 (own_threshold_prediction.md, 312829f)."""
    own = pd.read_csv(RESULTS / "own_threshold_block4.csv")
    hz = pd.read_csv(RESULTS / "fixed_scale_horizons.csv")
    hz = hz[hz.variant == "preserved"].merge(own[["seed", "w2_own", "w2_pop"]], on="seed", how="left")
    assert hz.w2_own.notna().all()
    placed = hz["placed_64000"].astype(bool)
    own_rule = hz.level * hz.w2_pop > hz.w2_own
    pop_rule = hz.level > 1.0
    ag_own, ag_pop = float((own_rule == placed).mean()), float((pop_rule == placed).mean())
    rows = [{"test": "S1", "n": len(hz), "own_rule_agreement": ag_own, "pop_rule_agreement": ag_pop,
             "pass": ag_own >= 0.95 and ag_own - ag_pop >= 0.05}]
    tst = pd.read_csv(RESULTS / "fixed_scale_horizons_tests.csv").set_index("variant")
    x50 = float(tst.loc["preserved", "x50_64k"])
    med = float(own[own.seed.isin(hz.seed.unique())].own_over_pop.median())
    rows.append({"test": "S2", "x50_64k": x50, "median_own_over_pop": med, "diff": x50 - med,
                 "pass": abs(x50 - med) <= 0.02})
    cr = pd.read_csv(RESULTS / "own_threshold_crossing.csv")
    runs = pd.read_csv(RESULTS / "wi_crossing_runs.csv")
    offsets = {1.30: 0.096, 1.50: 0.126}
    for a, off in offsets.items():
        o = cr[cr.a.round(2) == a]
        R_pop = float(o.w2_pop.iloc[0] * o.Ghat_cert.iloc[0] / 2)
        m = runs[(runs.a.round(2) == a) & (runs.budget == 32_000)].merge(o[["seed", "R_own"]], on="seed")
        excess = float(o.own_over_pop.median() - 1)
        rho, pval = _spearman_perm(m.R_cert.values, m.R_own.values)
        off_own = abs(float(np.median(np.log(m.R_cert / m.R_own))))
        off_pop = abs(float(np.median(np.log(m.R_cert / R_pop))))
        rows.append({"test": f"S3 a={a:.2f}", "n_crossing_runs": len(m), "median_own_over_pop_minus_1": excess,
                     "required_excess": off / 2, "S3a_pass": excess >= off / 2,
                     "spearman_rho": rho, "spearman_p_one_sided": pval, "S3b_pass": rho > 0 and pval < 0.05,
                     "offset_vs_own": off_own, "offset_vs_pop": off_pop, "S3c_pass": off_own < off_pop,
                     "competing_centred_within_1pct": abs(excess) <= 0.01,
                     "pass": excess >= off / 2 and rho > 0 and pval < 0.05 and off_own < off_pop})
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "own_threshold_scores.csv", index=False)
    print(d.T.to_string())


def block5_posthoc():
    """POST HOC (Block 5 was scored before this comparison was proposed): does Block 5's retention midpoint
    (x50 = 1.067) match median(own w2_glob)/population over Block 5's seeds, as S2 predicts for Block 4?"""
    own = pd.read_csv(RESULTS / "own_threshold_block4.csv")
    b5 = pd.read_csv(RESULTS / "fixed_scale_block5.csv")
    t5 = pd.read_csv(RESULTS / "fixed_scale_block5_tests.csv").set_index("variant")
    m = own[own.seed.isin(b5.seed.unique())]
    b5 = b5.merge(own[["seed", "w2_own", "w2_pop"]], on="seed")
    b5["own_rule"] = b5.level * b5.w2_pop > b5.w2_own                 # retained iff held R above own threshold
    b5["pop_rule"] = b5.level > 1.0
    b5["retained"] = b5.first_hit.isna()
    out = {"label": "post hoc", "n_seeds": m.seed.nunique(), "x50_preserved": float(t5.loc["preserved", "x50"]),
           "median_own_over_pop": float(m.own_over_pop.median()),
           "diff": float(t5.loc["preserved", "x50"] - m.own_over_pop.median()),
           "within_S2_tolerance_0.02": abs(float(t5.loc["preserved", "x50"] - m.own_over_pop.median())) <= 0.02,
           "own_rule_agreement": float((b5.own_rule == b5.retained).mean()),
           "pop_rule_agreement": float((b5.pop_rule == b5.retained).mean())}
    pd.DataFrame([out]).to_csv(RESULTS / "own_threshold_block5_posthoc.csv", index=False)
    print(pd.Series(out).to_string())


if __name__ == "__main__":
    {"block4": block4, "crossing": crossing, "validate": validate, "score": score,
     "block5_posthoc": block5_posthoc}[sys.argv[1]]()
