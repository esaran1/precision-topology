"""Block C: does a run track the conditional minimiser of the branch it occupies?

Registered criterion: median relative distance below 0.2 counts as tracking.

Compares each checkpoint's (w1,b1,b2) with the conditional minimiser reached by
WARM-STARTING from that checkpoint -- i.e. the minimiser on the branch the run
currently occupies, not the global one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .blockB_landscape import minimise, population_data
from .fold1d import make_data

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(n_runs: int = 30) -> None:
    d = pd.read_csv(RESULTS / "phase2b_checkpoints.csv")
    d = d[d.a.isin([1.30, 1.40])]
    rows = []
    keys = list(d.groupby(["a", "budget", "seed"]).groups.keys())
    rng = np.random.default_rng(0)
    pick = [keys[i] for i in rng.choice(len(keys), min(n_runs, len(keys)), replace=False)]
    for (a, B, s) in pick:
        g = d[(d.a == a) & (d.budget == B) & (d.seed == s)].sort_values("step")
        cs = g[g.crossing]
        cross = int(cs.step.iloc[0]) if len(cs) else None
        x, y = make_data(200, int(s))
        x, y = x.double(), y.double()
        sub = g.iloc[::max(1, len(g) // 12)]
        for _, r in sub.iterrows():
            m = minimise(a, float(r.w2), x, y,
                         start=[float(r.w1), float(r.b1), float(r.b2)])
            v = np.array([r.w1, r.b1, r.b2], float)
            u = np.array([m["w1"], m["b1"], m["b2"]], float)
            nrm = np.linalg.norm(u)
            rows.append({"a": a, "budget": B, "seed": s, "step": int(r.step),
                         "before_crossing": cross is None or int(r.step) < cross,
                         "rel_dist": float(np.linalg.norm(v - u) / nrm) if nrm > 0 else np.nan,
                         "gap_run": float(r.gap), "gap_min": m["gap"]})
    frame = pd.DataFrame(rows)
    pre = frame[frame.before_crossing].rel_dist
    post = frame[~frame.before_crossing].rel_dist
    print(f"  checkpoints compared: {len(frame)} over {len(pick)} runs")
    print(f"  median relative distance  BEFORE crossing: {pre.median():.4f} (n={len(pre)})")
    print(f"  median relative distance  AFTER  crossing: {post.median():.4f} (n={len(post)})")
    print(f"  overall median: {frame.rel_dist.median():.4f}")
    print(f"  registered: tracking if < 0.2 -> "
          f"{'TRACKS' if frame.rel_dist.median() < 0.2 else 'DOES NOT TRACK'}")
    stem = RESULTS / "blockC_adiabatic"
    with artifact_lock(stem, "blockC adiabatic"):
        tmp = stem.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
