"""Registered sample-size test (results/sample_size_prediction.md): does the own-seed excess and the free-training
offset shrink as the training set grows?

Cells: a ∈ {1.30, 1.50} × n ∈ {400, 1,600, 6,400} total points (fold1d.make_data(n/2, seed)), seeds 300,000-300,049
(fresh; the same seed gives the same training set to both arms of a cell).

Population threshold (the same at every n): the certified quadrature bracket midpoint of cond_certified_brackets.csv
(kind glob), R_pop = w2_pop·Ĝ_cert(a)/2 with the certified Ĝ of ghat_certified_all.csv.

Own-seed arm: own_threshold's method at grid step 0.05 with 16 refinements (the same at every n), bracketed from w2_pop
in steps of 0.1 and bisected to 0.01.  Free-training arm: the phase 2b protocol (float64, Adam lr 1e-2, init
torch.manual_seed(seed) then U(-1, 1)^4, full batch, budget 32,000); crossing = the first step with dense-grid
oriented gap > 0; R_cross = |w2|·Ĝ_cert/2 at that step.  If fewer than 40 of 50 cross, 20 seeds at a time are added
(300,050 onward, free-training arm only), up to 110.

Validation: (i) at n = 400, a = 1.30, the method against 1d's certified per-seed brackets on the 50 1d seeds (0-49);
(ii) the certified branch and bound (conditional_certified.evaluate) at both bracket ends on 2 seeds per cell
(300,000 and 300,001).

    python -m src.sample_size validate400 [workers]
    python -m src.sample_size own [workers]
    python -m src.sample_size free [workers]
    python -m src.sample_size certify [workers]
    python -m src.sample_size score
"""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.30, 1.50)
N_VALUES = (400, 1600, 6400)
SEED0 = 300_000
N_SEEDS = 50
MIN_CROSSINGS = 40
MAX_SEEDS = 110
STEP, N_REFINE = 0.05, 16
BUDGET = 32_000


def _data(n, seed):
    from .fold1d import make_data
    x, y = make_data(n // 2, seed)
    return x.double().numpy(), y.double().numpy()


def _pop(a):
    from .own_threshold import _pop as p
    G, w2 = p(a)
    return G, w2, w2 * G / 2


def own_one(args):
    from .own_threshold import BRACKET_STEP, WIDTH, global_min
    a, n, seed = args
    x, y = _data(n, seed)
    placed = lambda s: global_min(s, a, x, y, step=STEP, n_refine=N_REFINE)[3] > 0
    w2 = round(_pop(a)[1], 6)
    lo = hi = w2
    p = placed(lo)
    if p:
        while p and lo > 0.5:
            hi = lo; lo = round(lo - BRACKET_STEP, 6); p = placed(lo)
    else:
        while not p and hi < 20:
            lo = hi; hi = round(hi + BRACKET_STEP, 6); p = placed(hi)
    while hi - lo > WIDTH + 1e-12:
        mid = round(0.5 * (lo + hi), 6)
        if placed(mid):
            hi = mid
        else:
            lo = mid
    return {"a": a, "n": n, "seed": seed, "w2_lo": lo, "w2_hi": hi, "w2_own": 0.5 * (lo + hi)}


def free_one(args):
    from .fold1d import activation, logits
    from .phase2b_ordering import state
    a, n, seed = args
    G, _, _ = _pop(a)
    f = activation("sin_family", a)
    xt, yt = (torch.tensor(v) for v in _data(n, seed))
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)
    for step in range(1, BUDGET + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, xt, f), yt).backward()
        opt.step()
        st = state(theta.detach(), f, a, G)
        if st["placement_ok"]:
            return {"a": a, "n": n, "seed": seed, "cross_step": step, "w2_abs": abs(st["w2"]),
                    "R_cross": abs(st["w2"]) * G / 2}
    return {"a": a, "n": n, "seed": seed, "cross_step": np.nan, "w2_abs": np.nan, "R_cross": np.nan}


def _pool(fn, jobs, workers):
    with Pool(workers) as p:
        return p.map(fn, jobs, chunksize=1)


def validate400(workers=3):
    rows = _pool(own_one, [(1.30, 400, s) for s in range(50)], workers)
    d = pd.DataFrame(rows)
    c = pd.read_csv(RESULTS / "cond_certified_seeds_a1.30.csv")[["seed", "w2_glob_lo", "w2_glob_hi"]]
    m = d.merge(c, on="seed")
    m["overlap"] = (m.w2_lo <= m.w2_glob_hi + 1e-9) & (m.w2_hi >= m.w2_glob_lo - 1e-9)
    m["mid_diff"] = m.w2_own - 0.5 * (m.w2_glob_lo + m.w2_glob_hi)
    m.to_csv(RESULTS / "sample_size_validate400.csv", index=False)
    print("brackets overlapping 1d's certified brackets:", int(m.overlap.sum()), "of", len(m),
          "  max |midpoint diff|:", float(m.mid_diff.abs().max()))


def own(workers=3):
    jobs = [(a, n, SEED0 + i) for n in N_VALUES for a in A_VALUES for i in range(N_SEEDS)]
    pd.DataFrame(_pool(own_one, jobs, workers)).to_csv(RESULTS / "sample_size_own.csv", index=False)


def free(workers=3):
    rows = _pool(free_one, [(a, n, SEED0 + i) for n in N_VALUES for a in A_VALUES for i in range(N_SEEDS)], workers)
    d = pd.DataFrame(rows)
    while True:
        more = []
        for (a, n), g in d.groupby(["a", "n"]):
            if g.cross_step.notna().sum() < MIN_CROSSINGS and len(g) < MAX_SEEDS:
                start = SEED0 + len(g)
                more += [(a, n, s) for s in range(start, start + 20)]
        if not more:
            break
        d = pd.concat([d, pd.DataFrame(_pool(free_one, more, workers))])
    d.to_csv(RESULTS / "sample_size_free.csv", index=False)


def _cert_job(args):
    from .conditional_certified import evaluate
    a, n, seed, lo, hi = args
    x, y = _data(n, seed)
    rl, rh = evaluate(lo, a, x, y, False), evaluate(hi, a, x, y, False)
    return {"a": a, "n": n, "seed": seed, "w2_lo": lo, "w2_hi": hi, "cert_lo": rl["status"], "cert_hi": rh["status"],
            "agrees": rl["status"] == "minus" and rh["status"] == "plus"}


def certify(workers=3):
    o = pd.read_csv(RESULTS / "sample_size_own.csv")
    sub = o[o.seed.isin([SEED0, SEED0 + 1])]
    rows = _pool(_cert_job, [(r.a, r.n, r.seed, r.w2_lo, r.w2_hi) for r in sub.itertuples()], workers)
    pd.DataFrame(rows).to_csv(RESULTS / "sample_size_certify.csv", index=False)


def _boot_median_diff(u, v, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    bu = np.median(rng.choice(u, (B, len(u))), axis=1)
    bv = np.median(rng.choice(v, (B, len(v))), axis=1)
    return np.percentile(bu - bv, [2.5, 97.5])


def score():
    o = pd.read_csv(RESULTS / "sample_size_own.csv")
    fr = pd.read_csv(RESULTS / "sample_size_free.csv")
    cells, tests = [], []
    for a in A_VALUES:
        _, w2p, Rp = _pop(a)
        ex, off, ex_v, off_v = {}, {}, {}, {}
        for n in N_VALUES:
            r_own = o[(o.a.round(2) == a) & (o.n == n)].w2_own.values / w2p
            r_fr = fr[(fr.a.round(2) == a) & (fr.n == n)].R_cross.dropna().values / Rp
            ex[n], off[n] = float(np.median(r_own) - 1), float(np.median(r_fr) - 1)
            ex_v[n], off_v[n] = r_own, r_fr
            cells.append({"a": a, "n": n, "n_own": len(r_own), "n_crossings": len(r_fr),
                          "own_excess": ex[n], "free_offset": off[n], "N3_pass": ex[n] >= off[n] / 2})
        d_ex = _boot_median_diff(ex_v[400], ex_v[6400])
        d_off = _boot_median_diff(off_v[400], off_v[6400])
        n1 = ex[400] > ex[1600] > ex[6400] and d_ex[0] > 0
        n2 = off[400] > off[1600] > off[6400] and d_off[0] > 0
        n3 = all(c["N3_pass"] for c in cells if c["a"] == a)
        comp = d_off[0] <= 0 <= d_off[1] and off[6400] >= off[400] / 2
        tests.append({"a": a, "excess_400_minus_6400_ci_lo": d_ex[0], "excess_400_minus_6400_ci_hi": d_ex[1],
                      "offset_400_minus_6400_ci_lo": d_off[0], "offset_400_minus_6400_ci_hi": d_off[1],
                      "N1_pass": n1, "N2_pass": n2, "N3_pass": n3, "competing_offset_not_finite_sample": comp})
    pd.DataFrame(cells).to_csv(RESULTS / "sample_size_cells.csv", index=False)
    pd.DataFrame(tests).to_csv(RESULTS / "sample_size_tests.csv", index=False)
    print(pd.DataFrame(cells).to_string(index=False)); print(pd.DataFrame(tests).T.to_string())
    detail(o, fr)


def _boot_median(u, B=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile(np.median(rng.choice(u, (B, len(u))), axis=1), [2.5, 97.5])


def detail(o, fr):
    """Reported beside the registered verdicts (it changes none of them): every cell's value with its bootstrap
    95% interval, every pairwise difference with its interval, and for each strict-ordering step that fails, whether
    both values are consistent with zero ("indistinguishable at those sizes") -- a different outcome from an
    offset that does not shrink."""
    rows, pairs = [], []
    for a in A_VALUES:
        _, w2p, Rp = _pop(a)
        for q, get in (("own_excess", lambda n: o[(o.a.round(2) == a) & (o.n == n)].w2_own.values / w2p),
                       ("free_offset", lambda n: fr[(fr.a.round(2) == a) & (fr.n == n)].R_cross.dropna().values / Rp)):
            v = {n: get(n) for n in N_VALUES}
            est = {n: float(np.median(v[n]) - 1) for n in N_VALUES}
            ci = {n: _boot_median(v[n]) - 1 for n in N_VALUES}
            for n in N_VALUES:
                rows.append({"a": a, "quantity": q, "n": n, "value": est[n], "ci_lo": ci[n][0], "ci_hi": ci[n][1],
                             "consistent_with_zero": ci[n][0] <= 0 <= ci[n][1]})
            for n1, n2 in ((400, 1600), (400, 6400), (1600, 6400)):
                d = _boot_median_diff(v[n1], v[n2])
                strict = est[n1] > est[n2]
                pairs.append({"a": a, "quantity": q, "n_small": n1, "n_large": n2, "difference": est[n1] - est[n2],
                              "ci_lo": d[0], "ci_hi": d[1], "strictly_decreasing": strict,
                              "note": "" if strict or (n1, n2) == (400, 6400) else
                              ("registered ordering fails; both values consistent with zero: indistinguishable at "
                               "these sizes (not the same as an offset that does not shrink)"
                               if (ci[n1][0] <= 0 <= ci[n1][1]) and (ci[n2][0] <= 0 <= ci[n2][1])
                               else "registered ordering fails")})
    pd.DataFrame(rows).to_csv(RESULTS / "sample_size_detail_cells.csv", index=False)
    pd.DataFrame(pairs).to_csv(RESULTS / "sample_size_detail_pairs.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("SS_WORKERS", "3"))
    {"validate400": lambda: validate400(w), "own": lambda: own(w), "free": lambda: free(w),
     "certify": lambda: certify(w), "score": score}[cmd]()
