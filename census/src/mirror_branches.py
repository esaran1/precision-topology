"""EXPLORATORY, POST HOC: the two mirror branches of each run's own conditional loss.

Canonical orientation: w2 > 0 (a placement with w2 < 0 is (w1, b1) -> (-w1, -b1), since f_a is odd).  The two mirror
branches are then the half-planes w1 > 0 and w1 < 0 (separated by w1 = 0, where L* = log 2).  For each own training set
(fold1d.make_data(200, seed)):
  T+ and T-: the switch point of each half-plane's conditional minimiser (the own-threshold method restricted to the
  half-plane: grid 0.02, 8 refinements, bracket from the population |w2| in steps of 0.1, bisection to 0.01).
Occupancy, in canonical orientation:
  initial branch (torch.manual_seed(seed); U(-1, 1)^4, the protocol of phase 2b and Block 4/5 training);
  branch at the free-training crossing (phase 2b crossing rows at budget 32,000; Block 4/5 first-placement checkpoints);
  branch at each Block 4 replay's 64,000-step endpoint and at its starting checkpoint.
Then S1's per-replay agreement and S3's crossing correlation and residual are recomputed with the threshold of the branch
each run occupies (and with the branch its initialisation selects), beside the registered versions.  Registered verdicts
do not change.

    python -m src.mirror_branches thresholds [workers]
    python -m src.mirror_branches analyse
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


def _data(seed):
    from .fold1d import make_data
    x, y = make_data(200, seed)
    return x.double().numpy(), y.double().numpy()


def init_sign(seed):
    import torch
    torch.manual_seed(seed)
    t = torch.empty(4).uniform_(-1.0, 1.0)
    return int(np.sign(float(t[0])) * np.sign(float(t[2])))


def half_min(s, a, x, y, sign, step=0.02, n_refine=8):
    """Conditional minimiser of L*(.,.; s) restricted to sign·w1 > 0: (L, w1, b1, G)."""
    from .own_threshold import _bfgs
    from .profiled_bnb import gap, profile, w_bound
    W = w_bound(s, a, x, y)
    w1g = sign * np.arange(step / 2, W + step / 2, step)
    b1g = np.arange(0.0, 2 * math.pi, step)
    Wm, Bm = np.meshgrid(w1g, b1g, indexing="ij")
    wf, bf = Wm.ravel(), Bm.ravel()
    L = np.empty(wf.size)
    for i in range(0, wf.size, 2000):
        L[i:i + 2000] = profile(wf[i:i + 2000], bf[i:i + 2000], s, a, x, y)[0]
    order = np.argsort(L)
    starts = []
    for j in order:
        if all(abs(wf[j] - w) > 0.1 or min(abs(bf[j] - b), 2 * math.pi - abs(bf[j] - b)) > 0.1 for w, b in starts):
            starts.append((wf[j], bf[j]))
        if len(starts) == n_refine:
            break

    def fg(p):
        Lv, _, gw, gb, _ = profile([p[0]], [p[1]], s, a, x, y)
        return float(Lv[0]), np.array([gw[0], gb[0]])
    best = None
    for w0, b0 in starts:
        f, q = _bfgs(fg, np.array([w0, b0]))
        if np.sign(q[0]) != sign:                       # left the half-plane (cannot cross w1 = 0 downhill; guard)
            continue
        if best is None or f < best[0]:
            best = (f, float(q[0]), float(q[1]))
    if best is None:
        j = order[0]
        best = (float(L[j]), float(wf[j]), float(bf[j]))
    return best[0], best[1], best[2], float(gap([best[1]], [best[2]], a)[0])


def branch_threshold(a, seed, sign, w2_start):
    x, y = _data(seed)
    placed = lambda s: half_min(s, a, x, y, sign)[3] > 0
    lo = hi = round(w2_start, 6)
    p = placed(lo)
    if p:
        while p and lo > 0.3:
            hi = lo; lo = round(lo - 0.1, 6); p = placed(lo)
        if p:
            return np.nan
    else:
        while not p and hi < 20:
            lo = hi; hi = round(hi + 0.1, 6); p = placed(hi)
        if not p:
            return np.nan
    while hi - lo > 0.01 + 1e-12:
        mid = round(0.5 * (lo + hi), 6)
        if placed(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _job(args):
    group, a, seed, w2_start = args
    return {"group": group, "a": a, "seed": seed, "init_branch": init_sign(seed),
            "T_plus": branch_threshold(a, seed, +1, w2_start), "T_minus": branch_threshold(a, seed, -1, w2_start)}


def thresholds(workers=3):
    from .own_threshold import _pop
    b45 = sorted(pd.read_csv(RESULTS / "own_threshold_block4.csv").seed.unique())
    jobs = [("B45", 1.30, int(s), _pop(1.30)[1]) for s in b45]
    jobs += [("cross", a, s, _pop(a)[1]) for a in (1.30, 1.50) for s in range(40)]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_job, jobs, chunksize=1))
    d.to_csv(RESULTS / "mirror_branch_thresholds.csv", index=False)
    print(d.describe().to_string())


def _canon(w1, w2):
    return int(np.sign(w1) * np.sign(w2))


def _spear(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def analyse():
    from .own_threshold import _pop
    T = pd.read_csv(RESULTS / "mirror_branch_thresholds.csv", float_precision="round_trip")
    rows_occ, s1_rows, s3_rows = [], [], []
    # --- Block 4/5 runs: first free-training placement (budget 12,000) and replays
    ck = pd.read_csv(RESULTS / "fixed_scale_checkpoints.csv", float_precision="round_trip")
    G13, w2p13 = _pop(1.30)
    tb = T[T.group == "B45"].set_index("seed")
    fp = ck[ck.first_placement]
    for r in fp.itertuples():
        if r.seed in tb.index:
            rows_occ.append({"group": "B45 free crossing", "a": 1.30, "seed": r.seed, "init_branch": tb.loc[r.seed, "init_branch"],
                             "branch": _canon(r.w1, r.w2), "R_cross": abs(r.w2) * G13 / 2})
    hz = pd.read_csv(RESULTS / "fixed_scale_horizons.csv", float_precision="round_trip")
    hz = hz[hz.variant == "preserved"].copy()
    own = pd.read_csv(RESULTS / "own_threshold_block4.csv", float_precision="round_trip").set_index("seed")
    ckb = ck.set_index(["seed", "step"])
    hz["branch_end"] = [_canon(w, s) for w, s in zip(hz.w1_64000, hz.w2_sign)]
    hz["branch_start"] = [_canon(ckb.loc[(s, c), "w1"], ckb.loc[(s, c), "w2"]) for s, c in zip(hz.seed, hz.ck_step)]
    hz["init_branch"] = hz.seed.map(tb.init_branch)
    thr = lambda s, b: tb.loc[s, "T_plus"] if b > 0 else tb.loc[s, "T_minus"]
    placed = hz.placed_64000.astype(bool)
    held = hz.level * w2p13
    rules = {"registered: own global threshold": held > hz.seed.map(own.w2_own),
             "population rule (level > 1)": hz.level > 1.0,
             "branch occupied at the 64k endpoint (circular)": held > [thr(s, b) for s, b in zip(hz.seed, hz.branch_end)],
             "branch at the replay's starting checkpoint": held > [thr(s, b) for s, b in zip(hz.seed, hz.branch_start)],
             "branch selected by initialisation": held > [thr(s, b) for s, b in zip(hz.seed, hz.init_branch)]}
    for k, v in rules.items():
        s1_rows.append({"rule": k, "agreement": float((np.asarray(v) == placed.values).mean()), "n": len(hz)})
    s1_rows.append({"rule": "replays ending on the initialisation's branch", "agreement": float((hz.branch_end == hz.init_branch).mean()),
                    "n": len(hz)})
    s1_rows.append({"rule": "replays ending on their starting checkpoint's branch", "agreement": float((hz.branch_end == hz.branch_start).mean()),
                    "n": len(hz)})
    # --- crossing runs (phase 2b, budget 32,000)
    tc = T[T.group == "cross"]
    parts = []
    cols = ["a", "budget", "seed", "w1", "w2", "crossing"]
    for ch in pd.read_csv(RESULTS / "phase2b_checkpoints.csv", usecols=cols, chunksize=2_000_000):
        parts.append(ch[(ch.budget == 32_000) & ch.crossing.astype(bool)])
    cx = pd.concat(parts)
    runs = pd.read_csv(RESULTS / "wi_crossing_runs.csv")
    for a in (1.30, 1.50):
        G, w2p = _pop(a)
        t = tc[tc.a.round(2) == a].set_index("seed")
        c = cx[cx.a.round(2) == a].drop_duplicates("seed").set_index("seed")
        rr = runs[(runs.a.round(2) == a) & (runs.budget == 32_000)].set_index("seed")
        ot = pd.read_csv(RESULTS / "own_threshold_crossing.csv")
        ot = ot[ot.a.round(2) == a].set_index("seed")
        for s in rr.index:
            b = _canon(c.loc[s, "w1"], c.loc[s, "w2"])
            rows_occ.append({"group": f"phase 2b crossing a={a:.2f}", "a": a, "seed": s, "init_branch": t.loc[s, "init_branch"],
                             "branch": b, "R_cross": rr.loc[s, "R_cert"]})
        occ = pd.DataFrame([r for r in rows_occ if r["group"] == f"phase 2b crossing a={a:.2f}"]).set_index("seed")
        R_pop = w2p * G / 2
        for label, Tw in (("registered: own global threshold", ot.w2_own),
                          ("branch occupied at the crossing", pd.Series({s: (t.loc[s, "T_plus"] if occ.loc[s, "branch"] > 0 else t.loc[s, "T_minus"]) for s in occ.index})),
                          ("branch selected by initialisation", pd.Series({s: (t.loc[s, "T_plus"] if t.loc[s, "init_branch"] > 0 else t.loc[s, "T_minus"]) for s in occ.index}))):
            RT = Tw.reindex(occ.index) * G / 2
            ok = RT.notna()
            s3_rows.append({"a": a, "threshold": label, "n": int(ok.sum()),
                            "spearman_rho": _spear(occ.R_cross[ok], RT[ok]),
                            "residual_median_log": float(np.median(np.log(occ.R_cross[ok] / RT[ok]))),
                            "offset_vs_pop_median_log": float(np.median(np.log(occ.R_cross[ok] / R_pop))),
                            "median_threshold_over_pop": float(np.median(RT[ok] / R_pop))})
    occ = pd.DataFrame(rows_occ)
    occ["matches_init"] = occ.branch == occ.init_branch
    match = occ.groupby("group").agg(n=("seed", "size"), frac_matching_init=("matches_init", "mean"),
                                     frac_branch_plus=("branch", lambda v: float((v > 0).mean()))).reset_index()
    occ.to_csv(RESULTS / "mirror_occupancy.csv", index=False)
    match.to_csv(RESULTS / "mirror_init_match.csv", index=False)
    pd.DataFrame(s1_rows).to_csv(RESULTS / "mirror_s1.csv", index=False)
    pd.DataFrame(s3_rows).to_csv(RESULTS / "mirror_s3.csv", index=False)
    sp = T.assign(split=lambda d: (d.T_plus - d.T_minus).abs() / d[["T_plus", "T_minus"]].min(axis=1))
    sp.to_csv(RESULTS / "mirror_branch_thresholds_split.csv", index=False)
    print(match.to_string(index=False)); print(pd.DataFrame(s1_rows).to_string(index=False))
    print(pd.DataFrame(s3_rows).to_string(index=False))
    print(sp.groupby(["group", "a"]).split.describe().to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("MB_WORKERS", "3"))
    {"thresholds": lambda: thresholds(w), "analyse": analyse}[cmd]()
