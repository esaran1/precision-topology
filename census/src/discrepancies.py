"""Producers for numbers whose earlier sources had no committed script (WRITER_INPUTS §15).

margins      Margin of found solutions against the constructed solution's margin.
             Population: the float32 phase-1 runs (Adam lr 1e-2, full batch, 2,000 steps,
             200 seeds per a; the same training as the exclusion table's "found" rows) that
             solve; margin = exclusion_table.margin_of (minimum signed logit over dense grids
             of both windows).  Reference: the constructed |w2| = 1 solution's margin from
             exclusion_table.csv.
pathwise     True initialisation distance for the 400 runs of pathwise_distance.csv.  The
             checkpoints start after one Adam step; the initialisation is regenerated
             (torch.manual_seed(seed); U(-1, 1)^4, as in phase2b_ordering.run) and checked by
             reproducing the first logged step.
mep          The MEP string settings and which a values the exclusion table covers.

    python -m src.discrepancies margins | pathwise | mep
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def margins():
    import torch
    from .exclusion_table import margin_of
    d = pd.read_csv(RESULTS / "phase1_decomposition.csv")
    d = d[(d.precision == "float32") & (d.solved.astype(str).str.lower() == "true")]
    ex = pd.read_csv(RESULTS / "exclusion_table.csv")
    zb = ex[ex.population == "zero_basin_constructed"]
    ref = pd.Series(zb["margin"].values, index=zb.a.round(2).values)
    rows, runs = [], []
    for a, g in d.groupby(d.a.round(2)):
        m = np.array([margin_of(torch.tensor([r.w1, r.b1, r.w2, r.b2], dtype=torch.float64), float(a))
                      for r in g.itertuples()])
        for s, v in zip(g.seed, m):
            runs.append({"a": a, "seed": int(s), "margin": v})
        rc = float(ref.loc[a]) if a in ref.index else np.nan
        rows.append({"a": a, "n_solved": len(m), "min_margin": float(m.min()),
                     "median_margin": float(np.median(m)), "constructed_margin": rc,
                     "n_below_constructed": int((m < rc).sum()) if np.isfinite(rc) else None})
    pd.DataFrame(runs).to_csv(RESULTS / "discrepancy_margin_runs.csv", index=False)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "discrepancy_margin.csv", index=False)
    print(t.to_string(index=False))


def pathwise():
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data
    from .pathwise_distance import constructed_family
    from .phase2b_ordering import LR, N_PER_CLASS
    pw = pd.read_csv(RESULTS / "pathwise_distance.csv")
    fams, rows = {}, []
    for r in pw.itertuples():
        a, seed = float(r.a), int(r.seed)
        if a not in fams:
            fams[a] = constructed_family(a)
        torch.manual_seed(seed)
        init = torch.empty(4).uniform_(-1.0, 1.0)
        theta = init.double().clone().requires_grad_(True)
        f = activation("sin_family", a)
        x, y = make_data(N_PER_CLASS, seed)
        opt = torch.optim.Adam([theta], lr=LR)
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, x.double(), f), y.double()).backward()
        opt.step()
        p0 = init.double().numpy()
        d0 = float(np.linalg.norm(fams[a] - p0[None, :], axis=1).min())
        d1 = float(np.linalg.norm(fams[a] - theta.detach().numpy()[None, :], axis=1).min())
        rows.append({"a": a, "budget": int(r.budget), "seed": seed, "init_distance_step0": d0,
                     "distance_step1_recomputed": d1, "logged_first_distance": float(r.init_distance),
                     "step1_reproduced": abs(d1 - float(r.init_distance)) < 1e-9,
                     "min_distance_logged": float(r.min_distance),
                     "min_including_step0": min(d0, float(r.min_distance)),
                     "min_at_step0": d0 < float(r.min_distance)})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "discrepancy_pathwise_init.csv", index=False)
    print(t[["init_distance_step0", "logged_first_distance", "min_including_step0"]].median())
    print("step 1 reproduced:", int(t.step1_reproduced.sum()), "of", len(t),
          "| minimum at step 0:", int(t.min_at_step0.sum()))


def mep():
    from . import barrier
    ex = pd.read_csv(RESULTS / "exclusion_table.csv")
    bc = pd.read_csv(RESULTS / "barrier.csv")
    t = pd.DataFrame([{"string_images": barrier.STRING_IMAGES, "string_steps": barrier.STRING_STEPS,
                       "string_lr": barrier.STRING_LR, "inits_per_row": 5, "rows": len(ex),
                       "distinct_a": ex.a.nunique(), "a_values": " ".join(f"{v:g}" for v in sorted(ex.a.unique())),
                       "paths": 5 * len(ex), "barrier_mep_max": float(ex.barrier_mep_max.max()),
                       "barrier_csv_rows": len(bc), "barrier_csv_distinct_a": bc.a.nunique(),
                       "barrier_csv_a_values": " ".join(f"{v:g}" for v in sorted(bc.a.unique())),
                       "barrier_csv_mep_max": float(bc.barrier_mep.max()),
                       "script_a_values_attempted": 15}])
    t.to_csv(RESULTS / "discrepancy_mep.csv", index=False)
    print(t.T.to_string())


if __name__ == "__main__":
    {"margins": margins, "pathwise": pathwise, "mep": mep}[sys.argv[1]]()
