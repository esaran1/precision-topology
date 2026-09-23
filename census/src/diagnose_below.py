"""Exploratory (labelled): the Block 4 checkpoints placed at R/R_glob = 0.9 after 4,000 steps.

For each such run: its own per-seed certified R_glob (the certified procedure on the run's 400-point training
set, conditional_certified.scan), expressed relative to the population R_glob.  If a run's own threshold lies
below 0.9 x the population threshold, its placement at 0.9 is its own equilibrium (per-seed threshold), not
overshoot.  Long-horizon persistence and branch distances come from fixed_scale_horizons.csv.

    python -m src.diagnose_below
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _job(seed):
    from .conditional_certified import _seed_data, scan
    x, y = _seed_data(int(seed))
    grid = [round(v, 4) for v in np.arange(2.0, 9.0 + 1e-9, 0.5)]
    res, _ = scan(1.30, x, y, symmetric=False, grid=grid)
    res["seed"] = int(seed)
    return res


def main():
    d = pd.read_csv(RESULTS / "fixed_scale_block4.csv")
    seeds = sorted(d[(d.level == 0.9) & (d.variant == "preserved") & d.placed_end.astype(bool)].seed.unique())
    with Pool(int(os.environ.get("DB_WORKERS", "2"))) as p:
        out = p.map(_job, seeds)
    t = pd.DataFrame(out)
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    b = br[(br.a.round(2) == 1.30) & (br.kind == "glob")].iloc[0]
    w2_pop = 0.5 * (b.w2_lo + b.w2_hi)
    t["own_w2_glob_mid"] = 0.5 * (t.w2_glob_lo + t.w2_glob_hi)
    t["own_over_pop"] = t.own_w2_glob_mid / w2_pop
    t["own_threshold_below_0p9"] = t.own_over_pop < 0.9
    t.to_csv(RESULTS / "diagnose_below.csv", index=False)
    print(t[["seed", "w2_glob_lo", "w2_glob_hi", "own_over_pop", "own_threshold_below_0p9"]].to_string(index=False))


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# ------------------------------------------------------------------------------------------ endpoint states (EXPLORATORY)
def _endpoint_job(args):
    """One placed 0.9x replay: its endpoint on the run's own training set at the held scale."""
    import math
    from .conditional_certified import _seed_data
    from .own_threshold import _bfgs
    from .profiled_bnb import certify, gap, profile
    row, s, a = args
    x, y = _seed_data(int(row["seed"]))
    out = {"seed": int(row["seed"]), "ck_step": int(row["ck_step"]), "held_s": s}
    sgn = row["w2_sign"]

    def fg(p):
        L, _, gw, gb, _ = profile([p[0]], [p[1]], s, a, x, y)
        return float(L[0]), np.array([gw[0], gb[0]])

    def hess(p, h=1e-5):
        H = np.array([(fg(p + h * e)[1] - fg(p - h * e)[1]) / (2 * h) for e in np.eye(2)])
        return np.linalg.eigvalsh(0.5 * (H + H.T))
    rm = certify(s, a, x, y, "-", tol=1e-9, max_cells=16_000_000)
    rp = certify(s, a, x, y, "+", tol=1e-9, max_cells=16_000_000)
    glob = rm if rm["upper"] <= rp["upper"] else rp
    out.update(m_minus_lo=rm["lower"], m_minus_hi=rm["upper"], m_plus_lo=rp["lower"], m_plus_hi=rp["upper"],
               global_region=glob["region"], global_w1=glob["arg_w1"], global_b1=glob["arg_b1"],
               global_G=float(gap([glob["arg_w1"]], [glob["arg_b1"]], a)[0]))
    for H in (4000, 16000, 64000):
        w1, b1 = row[f"w1_{H}"], row[f"b1_{H}"]
        if sgn < 0:                                   # w2 < 0: f odd, so (w1, b1) at -|w2| is (-w1, -b1) at +|w2|
            w1, b1 = -w1, -b1
        b1 = b1 % (2 * math.pi)
        p = np.array([w1, b1])
        L, g = fg(p)
        ev = hess(p)
        q_L, q = _bfgs(fg, p.copy())
        out.update({f"placed_{H}": bool(row[f"placed_{H}"]), f"G_end_{H}": float(gap([w1], [b1], a)[0]),
                    f"L_end_{H}": L, f"grad_norm_{H}": float(np.linalg.norm(g)),
                    f"hess_min_{H}": float(ev[0]), f"hess_max_{H}": float(ev[1]),
                    f"loc_min_L_{H}": q_L, f"loc_min_G_{H}": float(gap([q[0]], [q[1]], a)[0]),
                    f"loc_min_w1_{H}": float(q[0]), f"loc_min_b1_{H}": float(q[1] % (2 * math.pi)),
                    f"loc_min_hess_min_{H}": float(hess(q)[0]),
                    f"loss_gap_to_global_{H}": q_L - glob["upper"]})
    # barrier from the 64k local minimum to the global argmin: lowest sublevel set connecting them on a grid
    q = np.array([out["loc_min_w1_64000"], out["loc_min_b1_64000"]])
    g0 = np.array([glob["arg_w1"], glob["arg_b1"]])
    lo_, hi_ = np.minimum(q, g0) - 0.3, np.maximum(q, g0) + 0.3
    step = 0.004
    w_ax = np.arange(lo_[0], hi_[0] + step, step); b_ax = np.arange(lo_[1], hi_[1] + step, step)
    Wg, Bg = np.meshgrid(w_ax, b_ax, indexing="ij")
    Lg = np.empty(Wg.size)
    for i in range(0, Wg.size, 2000):
        Lg[i:i + 2000] = profile(Wg.ravel()[i:i + 2000], Bg.ravel()[i:i + 2000], s, a, x, y)[0]
    Lg = Lg.reshape(Wg.shape)
    nw, nb = Lg.shape
    parent = np.arange(Lg.size)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    idx = lambda p_: int(round((p_[0] - lo_[0]) / step)) * nb + int(round((p_[1] - lo_[1]) / step))
    ia, ib = idx(q), idx(g0)
    added = np.zeros(Lg.size, bool)
    barrier = np.nan
    for c in np.argsort(Lg, axis=None):
        added[c] = True
        i, j = divmod(c, nb)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < nw and 0 <= jj < nb and added[ii * nb + jj]:
                ra, rb = find(c), find(ii * nb + jj)
                if ra != rb:
                    parent[ra] = rb
        if added[ia] and added[ib] and find(ia) == find(ib):
            barrier = float(Lg.ravel()[c]); break
    out.update(barrier_level=barrier, barrier_above_local_min=barrier - out["loc_min_L_64000"],
               barrier_grid_step=step)
    return out


def endpoints():
    """EXPLORATORY: the seven Block 4 checkpoints placed at 0.9x (preserved variant), characterised at each horizon."""
    hz = pd.read_csv(RESULTS / "fixed_scale_horizons.csv")
    b4 = pd.read_csv(RESULTS / "fixed_scale_block4.csv")
    sel = b4[(b4.level == 0.9) & (b4.variant == "preserved") & b4.placed_end.astype(bool)][["seed", "ck_step"]]
    rows = hz[(hz.variant == "preserved") & (hz.level.round(2) == 0.9)].merge(sel, on=["seed", "ck_step"])
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    b = br[(br.a.round(2) == 1.30) & (br.kind == "glob")].iloc[0]
    s = 0.9 * 0.5 * (b.w2_lo + b.w2_hi)
    with Pool(int(os.environ.get("DB_WORKERS", "2"))) as p:
        out = p.map(_endpoint_job, [(r._asdict() if hasattr(r, "_asdict") else r, s, 1.30)
                                    for r in rows.to_dict("records")], chunksize=1)
    t = pd.DataFrame(out)
    t.to_csv(RESULTS / "diagnose_below_endpoints.csv", index=False)
    print(t.T.to_string())


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "endpoints":
    endpoints()
