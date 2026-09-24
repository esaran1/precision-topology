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


def half_min(s, a, x, y, sign, step=0.02, n_refine=8, win=(-0.8, 0.8, 1.2, 2.0)):
    """Conditional minimiser of L*(.,.; s) restricted to sign·w1 > 0: (L, w1, b1, G); win = (i_lo, i_hi, o_lo, o_hi)."""
    from .own_threshold import _bfgs
    from .profiled_bnb import gap, profile, w_bound
    W = w_bound(s, a, x, y, win)
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
    return best[0], best[1], best[2], float(gap([best[1]], [best[2]], a, win[:2], win[2:])[0])


def branch_threshold(a, seed, sign, w2_start, data=None, step=0.02, n_refine=8, win=(-0.8, 0.8, 1.2, 2.0)):
    x, y = data if data is not None else _data(seed)
    placed = lambda s: half_min(s, a, x, y, sign, step=step, n_refine=n_refine, win=win)[3] > 0
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



def _gb_job(args):
    from .own_threshold import global_min
    seed, level, s = args
    x, y = _data(seed)
    L, w1, b1, G, _ = global_min(s, 1.30, x, y)
    return {"seed": seed, "level": level, "held_s": s, "global_branch": int(np.sign(w1)), "global_G": G, "global_L": L}


def global_branch_levels(workers=1):
    """The globally preferred mirror branch of each run's own loss at each held scale of the horizon replays."""
    from .own_threshold import _pop
    w2p = _pop(1.30)[1]
    hz = pd.read_csv(RESULTS / "fixed_scale_horizons.csv", float_precision="round_trip")
    pairs = hz[hz.variant == "preserved"][["seed", "level"]].drop_duplicates()
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_gb_job, [(int(r.seed), float(r.level), float(r.level) * w2p) for r in pairs.itertuples()],
                               chunksize=4))
    d.to_csv(RESULTS / "mirror_global_branch.csv", index=False)
    print(d.describe().to_string())


def q2_s1_breakdown():
    """Post hoc: Q2 against own thresholds, and S1's disagreements by distance to the own threshold and by mirror
    occupancy (64k endpoint on the globally preferred branch at the held scale, or on its mirror)."""
    hz = pd.read_csv(RESULTS / "fixed_scale_horizons.csv", float_precision="round_trip")
    own = pd.read_csv(RESULTS / "own_threshold_block4.csv", float_precision="round_trip")
    gb = pd.read_csv(RESULTS / "mirror_global_branch.csv", float_precision="round_trip")
    p = hz[hz.variant == "preserved"].merge(own[["seed", "own_over_pop"]], on="seed")
    p = p.merge(gb[["seed", "level", "global_branch", "global_G"]], on=["seed", "level"])
    p["placed"] = p.placed_64000.astype(bool)
    p["own_rule"] = p.level > p.own_over_pop
    p["branch_end"] = [_canon(w, s) for w, s in zip(p.w1_64000, p.w2_sign)]
    p["on_mirror"] = p.branch_end != p.global_branch
    p["dist"] = (p.level / p.own_over_pop - 1).abs()
    p["dist_bin"] = pd.cut(p.dist, [0, 0.01, 0.03, 0.05, np.inf], labels=["<1%", "1-3%", "3-5%", ">5%"], right=False)
    seeds = own[own.seed.isin(p.seed.unique())]
    rows = [{"part": "Q2 vs own", "level": lv, "frac_seeds_own_below_level": float((seeds.own_over_pop < lv).mean()),
             "placed_frac_64k": float(p[p.level.round(2) == lv].placed.mean()),
             "placed_n": int(p[p.level.round(2) == lv].placed.sum()),
             "placed_above_own": int((p[p.level.round(2) == lv].placed & p[p.level.round(2) == lv].own_rule).sum()),
             "placed_below_own": int((p[p.level.round(2) == lv].placed & ~p[p.level.round(2) == lv].own_rule).sum()),
             "placed_below_own_on_mirror": int((p[p.level.round(2) == lv].placed & ~p[p.level.round(2) == lv].own_rule
                                                & p[p.level.round(2) == lv].on_mirror).sum())} for lv in (0.9, 0.95)]
    for key in ("dist_bin", "on_mirror"):
        for k, g in p.groupby(key, observed=False):
            rows.append({"part": f"S1 by {key}", "group": str(k), "n": len(g), "agreement": float((g.own_rule == g.placed).mean()),
                         "disagreements": int((g.own_rule != g.placed).sum()),
                         "placed_but_below_own": int((~g.own_rule & g.placed).sum()),
                         "unplaced_but_above_own": int((g.own_rule & ~g.placed).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "mirror_q2_s1_breakdown.csv", index=False)
    print(t.to_string(index=False))


def _s1_rows():
    from .own_threshold import _pop
    w2p = _pop(1.30)[1]
    hz = pd.read_csv(RESULTS / "fixed_scale_horizons.csv", float_precision="round_trip")
    own = pd.read_csv(RESULTS / "own_threshold_block4.csv", float_precision="round_trip")
    ck = pd.read_csv(RESULTS / "fixed_scale_checkpoints.csv", float_precision="round_trip").set_index(["seed", "step"])
    p = hz[hz.variant == "preserved"].merge(own[["seed", "own_over_pop", "w2_own"]], on="seed")
    p["held_s"] = p.level * w2p
    p["placed"] = p.placed_64000.astype(bool)
    p["own_rule"] = p.held_s > p.w2_own
    p["dist"] = (p.held_s / p.w2_own - 1).abs()
    p["branch_start"] = [_canon(ck.loc[(s_, c), "w1"], ck.loc[(s_, c), "w2"]) for s_, c in zip(p.seed, p.ck_step)]
    p["branch_end"] = [_canon(w, g) for w, g in zip(p.w1_64000, p.w2_sign)]
    return p


def mirror_gap():
    """Post hoc, item 1: |T+ - T-|/T_global per seed; for each S1 disagreement, whether the threshold of the branch the
    replay started on classifies it correctly (and whether that gap exceeds the replay's distance from its own threshold)."""
    T = pd.read_csv(RESULTS / "mirror_branch_thresholds.csv", float_precision="round_trip")
    T = T[T.group == "B45"].set_index("seed")
    own = pd.read_csv(RESULTS / "own_threshold_block4.csv", float_precision="round_trip").set_index("seed")
    T["gap_rel"] = (T.T_plus - T.T_minus).abs() / own.w2_own.reindex(T.index)
    p = _s1_rows()
    d = p[p.own_rule != p.placed].copy()
    d["T_start"] = [T.loc[s_, "T_plus"] if b > 0 else T.loc[s_, "T_minus"] for s_, b in zip(d.seed, d.branch_start)]
    d["T_end"] = [T.loc[s_, "T_plus"] if b > 0 else T.loc[s_, "T_minus"] for s_, b in zip(d.seed, d.branch_end)]
    d["start_rule_correct"] = (d.held_s > d.T_start) == d.placed
    d["end_rule_correct"] = (d.held_s > d.T_end) == d.placed
    d["seed_gap_rel"] = d.seed.map(T.gap_rel)
    d["gap_exceeds_distance"] = d.seed_gap_rel > d.dist
    d.to_csv(RESULTS / "mirror_gap_disagreements.csv", index=False)
    summ = pd.DataFrame([{"n_seeds": int(T.gap_rel.notna().sum()), "gap_rel_median": float(T.gap_rel.median()),
                          "gap_rel_min": float(T.gap_rel.min()), "gap_rel_max": float(T.gap_rel.max()),
                          "gap_rel_q25": float(T.gap_rel.quantile(0.25)), "gap_rel_q75": float(T.gap_rel.quantile(0.75)),
                          "n_disagreements": len(d), "gap_exceeds_distance": int(d.gap_exceeds_distance.sum()),
                          "start_branch_rule_correct": int(d.start_rule_correct.sum()),
                          "end_branch_rule_correct": int(d.end_rule_correct.sum())}])
    summ.to_csv(RESULTS / "mirror_gap_summary.csv", index=False)
    print(summ.T.to_string())


def _census_job(args):
    """One (seed, held scale): the two mirror-branch minima, and every disagreeing replay's endpoint basin."""
    from .own_threshold import _bfgs
    from .profiled_bnb import gap, profile
    seed, s, ends = args
    x, y = _data(seed)

    def fg(p_, s_=s):
        L, _, gw, gb, _ = profile([p_[0]], [p_[1]], s_, 1.30, x, y)
        return float(L[0]), np.array([gw[0], gb[0]])

    def hess(p_, h=1e-5):
        H = np.array([(fg(p_ + h * e)[1] - fg(p_ - h * e)[1]) / (2 * h) for e in np.eye(2)])
        return np.linalg.eigvalsh(0.5 * (H + H.T))
    br = {sg: half_min(s, 1.30, x, y, sg) for sg in (+1, -1)}
    L_glob = min(v[0] for v in br.values())

    def dist(p_, q_):
        db = abs(p_[1] - q_[1]) % (2 * math.pi)
        return math.hypot(p_[0] - q_[0], min(db, 2 * math.pi - db))
    out = []
    for e in ends:
        pe = np.array([e["w1c"], e["b1c"]])
        L0, g0 = fg(pe)
        qL, q = _bfgs(fg, pe.copy())
        q[1] %= 2 * math.pi
        which = next((sg for sg in (+1, -1) if dist(q, (br[sg][1], br[sg][2])) < 0.02), 0)
        row = {**e, "grad_norm_end": float(np.linalg.norm(g0)), "hess_min_end": float(hess(pe)[0]),
               "G_end": float(gap([pe[0]], [pe[1]], 1.30)[0]), "dist_end_to_loc_min": dist(pe, q),
               "loc_min_L": qL, "loc_min_w1": float(q[0]), "loc_min_b1": float(q[1]),
               "loc_min_G": float(gap([q[0]], [q[1]], 1.30)[0]), "loc_min_hess_min": float(hess(q)[0]),
               "basin": {1: "mirror +", -1: "mirror -", 0: "other"}[which], "L_minus_global": qL - L_glob,
               "branch_plus_L": br[1][0], "branch_minus_L": br[-1][0], "switch_s": np.nan}
        if which == 0:                                  # follow the basin in s to its own switch (G sign change)
            G0 = row["loc_min_G"]
            for direction in (+1, -1):
                qq, sprev, Gprev = q.copy(), s, G0
                for k in range(1, 80):
                    s_k = s + direction * 0.05 * k
                    if s_k <= 0.3:
                        break
                    _, qn = _bfgs(lambda p_: fg(p_, s_k), qq.copy())
                    if dist(qn, qq) > 0.2:              # basin lost
                        break
                    Gk = float(gap([qn[0]], [qn[1]], 1.30)[0])
                    if np.sign(Gk) != np.sign(Gprev):
                        row["switch_s"] = 0.5 * (s_k + sprev); break
                    qq, sprev, Gprev = qn, s_k, Gk
                if np.isfinite(row["switch_s"]):
                    break
        out.append(row)
    return out


def basin_census(workers=2, subset="all"):
    """Post hoc: disagreeing replays' 64k endpoints, classified as mirror + / mirror - / other basin on their own
    objective at the held scale (canonical orientation, b1 mod 2π).  subset="preferred": only the disagreements whose
    endpoint lies on the globally preferred branch at the held scale; each is further classified as a third basin,
    slow relaxation (endpoint not yet at its local minimum) or at the branch minimiser."""
    from .own_threshold import _pop
    w2p = _pop(1.30)[1]
    p = _s1_rows()
    d = p[p.own_rule != p.placed].copy()
    if subset == "preferred":
        gb = pd.read_csv(RESULTS / "mirror_global_branch.csv", float_precision="round_trip")
        d = d.merge(gb[["seed", "level", "global_branch"]], on=["seed", "level"])
        d = d[d.branch_end == d.global_branch].copy()
    d["w1c"] = d.w1_64000 * d.w2_sign
    d["b1c"] = (d.b1_64000 * d.w2_sign) % (2 * math.pi)
    jobs = []
    for (seed, s), g in d.groupby(["seed", "held_s"]):
        jobs.append((int(seed), float(s), g[["seed", "ck_step", "level", "held_s", "placed", "own_rule", "dist", "w1c", "b1c"]]
                     .to_dict("records")))
    with Pool(workers) as pool:
        rows = [r for part in pool.map(_census_job, jobs, chunksize=1) for r in part]
    c = pd.DataFrame(rows)
    c["switch_over_pop"] = c.switch_s / w2p
    c["kind"] = np.where(c.w1c.abs() < 0.05, "degenerate plateau (w1 ≈ 0, loss ≈ log 2; branch label arbitrary)",
                np.where(c.basin == "other", "third basin",
                         np.where((c.grad_norm_end > 1e-4) | (c.dist_end_to_loc_min > 1e-3), "slow relaxation near the switch",
                                  "at the branch minimiser")))
    c.to_csv(RESULTS / ("mirror_basin_census.csv" if subset == "all" else "mirror_basin_census_preferred.csv"), index=False)
    print(c[["seed", "level", "placed", "dist", "G_end", "loc_min_G", "grad_norm_end", "dist_end_to_loc_min",
             "loc_min_hess_min", "basin", "kind"]].to_string(index=False))
    print(c.basin.value_counts().to_string())
    oth = c[c.basin == "other"]
    if len(oth):
        print(oth[["seed", "level", "placed", "loc_min_G", "L_minus_global", "loc_min_hess_min", "switch_over_pop"]].to_string(index=False))



def _size_gap_job(args):
    from .sample_size import _data as ss_data
    a, n, seed, w2_start = args
    data = ss_data(n, seed)
    return {"a": a, "n": n, "seed": seed,
            "T_plus": branch_threshold(a, seed, +1, w2_start, data, step=0.05, n_refine=16),
            "T_minus": branch_threshold(a, seed, -1, w2_start, data, step=0.05, n_refine=16)}


def size_gap(workers=3, n_seeds=20):
    """Post hoc: the mirror-threshold gap |T+ - T-| / T_global for the size-test seeds (first 20 per cell) at n = 400,
    1,600 and 6,400 -- the size test's search settings (grid 0.05, 16 refinements); T_global from sample_size_own."""
    from .own_threshold import _pop
    from .sample_size import A_VALUES, N_VALUES, SEED0
    jobs = [(a, n, SEED0 + i, _pop(a)[1]) for a in A_VALUES for n in N_VALUES for i in range(n_seeds)]
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_size_gap_job, jobs, chunksize=1))
    o = pd.read_csv(RESULTS / "sample_size_own.csv", float_precision="round_trip")
    d = d.merge(o[["a", "n", "seed", "w2_own"]], on=["a", "n", "seed"])
    d["gap_rel"] = (d.T_plus - d.T_minus).abs() / d.w2_own
    d.to_csv(RESULTS / "mirror_size_gap.csv", index=False)
    t = d.groupby(["a", "n"]).gap_rel.agg(["size", "median", "min", "max"]).reset_index()
    t.to_csv(RESULTS / "mirror_size_gap_summary.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("MB_WORKERS", "3"))
    {"thresholds": lambda: thresholds(w), "analyse": analyse, "global_branch": lambda: global_branch_levels(w),
     "breakdown": q2_s1_breakdown, "gap": mirror_gap, "census": lambda: basin_census(w),
     "census_preferred": lambda: basin_census(w, "preferred"), "size_gap": lambda: size_gap(w)}[cmd]()
