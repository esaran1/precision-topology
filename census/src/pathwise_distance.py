"""Item #8 (optional): pathwise minimum distance to the nearest constructed solution.

basin_distances.csv records the distance from the INITIALISATION only -- a single
point.  This computes, for each Phase 2b run, the minimum over its whole recorded
trajectory, so the distance claim becomes pathwise.

The constructed |w2| = 1 solutions are the analytic family verified in
box_emptiness_correction.md and re-verified exactly in exact_separation.csv.  For
each a the solution is recovered by the recipe (fold the data window into the
local minimum m0 = pi + arccos(1/a), centre b2 in the resulting gap) and CHECKED
with exact extrema before use -- no unverified target is used as a reference.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .exact_extrema import separates
from .fold1d import INNER_MAX, OUTER_MAX, OUTER_MIN

RESULTS = Path(__file__).resolve().parents[1] / "results"


def constructed_family(a, n_w1=4000):
    """All |w2|=1 constructed solutions at this a, as a (k,4) array.

    Recipe: choose w1 so the outer window lands in the dip and the inner window
    outside it, then set b1 to centre the fold and b2 to split the achieved gap.
    Swept over w1 and kept where exact extrema confirm separation.
    """

    out = []
    m0 = math.pi + math.acos(1.0 / a)      # the local MINIMUM of f_a
    from .exact_extrema import extrema_on
    for w1 in np.linspace(0.02, 1.6, n_w1):
        # The INNER window goes in the dip: centre x = 0 on the local minimum, so
        # b1 = m0.  (Putting the OUTER window there is backwards -- verified
        # against the recorded a = 1.25 solution, whose b1 equals m0 exactly.)
        b1 = m0
        i_lo, i_hi = extrema_on(a, w1, b1, -INNER_MAX, INNER_MAX)
        p_lo, p_hi = extrema_on(a, w1, b1, OUTER_MIN, OUTER_MAX)
        n_lo, n_hi = extrema_on(a, w1, b1, -OUTER_MAX, -OUTER_MIN)
        o_lo, o_hi = min(p_lo, n_lo), max(p_hi, n_hi)
        for w2 in (1.0, -1.0):
            if w2 > 0:
                if not (o_lo > i_hi):
                    continue
                b2 = -w2 * (o_lo + i_hi) / 2.0
            else:
                if not (i_lo > o_hi):
                    continue
                b2 = -w2 * (i_lo + o_hi) / 2.0
            th = (float(w1), float(b1), w2, float(b2))
            ok, _, _ = separates(a, th)
            if ok:
                out.append(th)
    return np.array(out) if out else np.empty((0, 4))


def main(max_runs=400):
    ck = RESULTS / "phase2b_checkpoints.csv"
    if not ck.exists():
        print(f"{ck} not present (gitignored, 2.7 GB) -- regenerate with "
              f"src/phase2b_ordering.py")
        return
    print("=== item #8: pathwise minimum distance to a constructed solution ===\n",
          flush=True)
    d = pd.read_csv(ck)
    fams = {}
    for a in sorted(d.a.unique()):
        fams[a] = constructed_family(a)
        print(f"  a={a}: {len(fams[a])} verified constructed |w2|=1 solutions",
              flush=True)
    rows = []
    keys = list(d.groupby(["a", "budget", "seed"]).groups.keys())
    rng = np.random.default_rng(0)
    pick = [keys[i] for i in rng.choice(len(keys), min(max_runs, len(keys)),
                                        replace=False)]
    for (a, B, s) in pick:
        g = d[(d.a == a) & (d.budget == B) & (d.seed == s)].sort_values("step")
        fam = fams[a]
        if len(fam) == 0 or len(g) == 0:
            continue
        P = g[["w1", "b1", "w2", "b2"]].to_numpy()
        # pairwise distances, trajectory x family
        dist = np.linalg.norm(P[:, None, :] - fam[None, :, :], axis=2)
        per_step = dist.min(axis=1)
        j = int(per_step.argmin())
        rows.append({"a": a, "budget": B, "seed": s,
                     "n_checkpoints": len(g),
                     "init_distance": float(per_step[0]),
                     "min_distance": float(per_step.min()),
                     "min_at_step": int(g.step.iloc[j]),
                     "final_distance": float(per_step[-1]),
                     "final_solved": bool(g.final_solved.iloc[0])
                     if "final_solved" in g else None})
    f = pd.DataFrame(rows)
    stem = RESULTS / "pathwise_distance"
    with artifact_lock(stem, "pathwise distance"):
        f.to_csv(stem.with_suffix(".csv"), index=False)
    print(f"\n  runs: {len(f)}")
    print(f"  median distance at INIT      : {f.init_distance.median():.4f}")
    print(f"  median MINIMUM along the path: {f.min_distance.median():.4f}")
    print(f"  median distance at the END   : {f.final_distance.median():.4f}")
    print(f"  ratio min/init               : {(f.min_distance/f.init_distance).median():.4f}")
    print("\nwritten results/pathwise_distance.csv")


if __name__ == "__main__":
    main()
