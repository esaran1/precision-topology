"""EXPLORATORY (NOT registered; 2026-10-05): the |m'c'| two-window agreement on the traced exploration seeds, to propose
the windows of the approved page's rule 4 ("|m'c'| from two windows agreeing <= 5%"; the page does not name them).
Exploration seeds 2,930,000-2,930,199 ONLY (the grid seeds traced in p2a_explore_h); no registered or pilot seed drawn.
Per seed: the seed's M by src/phase2a_ps (homotopy, continuation, s_F), the training-coordinate tangent lambda_min on
the Delta s = 0.001 grid over the last 0.1 below s_F, and the quadratic fit lambda_min^2 = a d + b d^2 (|m'c'| = |a|/4)
over windows 0.1, 0.05, 0.025, 0.0125; plus the near-fold slope lambda_min^2/d at the last 5 points (descriptive).
Second pass (log "fold-form fits"): also the fold-form fit lambda_min^2 = a d + c d^{3/2} + b d^2 (lambda = sqrt(4|m'c'|d)
(1 + O(sqrt d)) near a fold) over the same windows, and the lambda_min grid itself.
One process, nice 15, one thread.     python p2a_explore_i.py SEED [SEED ...]"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
import torch
torch.set_num_threads(1)
from src import phase2a_ps as PS
from src import sb_fold as SBF

OUT = "results/designs/phase2a_explore/p2a_explore_i.jsonl"
WINDOWS = (0.1, 0.05, 0.025, 0.0125)


def fit(ss, lam, s_F, w, form="quad"):
    m = ss >= s_F - w
    d = s_F - ss[m]
    A = np.vstack([d, d ** 2]).T if form == "quad" else np.vstack([d, d ** 1.5, d ** 2]).T
    coef, *_ = np.linalg.lstsq(A, lam[m] ** 2, rcond=None)
    return float(abs(coef[0]) / 4), int(m.sum())


for sd in map(int, sys.argv[1:]):
    assert 2930000 <= sd <= 2930199
    t0 = time.time()
    X, Y = PS.sample(sd)
    u, nA, s_h, info = PS.homotopy_accepted("M", X, Y)
    if u is None:
        print(json.dumps({"seed": sd, "traceable": False})); continue
    R = SBF.Reduced(nA, X, Y)
    pf, ff, pb, fb = PS.continue_both(R, u, s_h, PS.H_CONT)
    if not ff:
        print(json.dumps({"seed": sd, "no_fold": True})); continue
    M = PS.SeedBranch(R, pb, pf)
    s_F = ff[0]["s_fold"]
    ss = np.arange(s_F - 0.1, s_F, 0.001); ss = ss[ss < s_F - 1e-6]
    ss = ss[ss >= M.lo]
    lam = np.array([PS.tangent_lmin(M.theta(s), X, Y) for s in ss])
    rec = {"label": "EXPLORATORY", "seed": sd, "s_F": s_F, "mc": {f"{w:g}": fit(ss, lam, s_F, w) for w in WINDOWS},
           "mc_foldform": {f"{w:g}": fit(ss, lam, s_F, w, "fold") for w in WINDOWS},
           "grid_s": [float(x) for x in ss], "grid_lam_min": [float(x) for x in lam],
           "near_fold_lam2_over_4d_last5": [float(x) for x in (lam[-5:] ** 2 / (4 * (s_F - ss[-5:])))],
           "secs": round(time.time() - t0, 1)}
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
