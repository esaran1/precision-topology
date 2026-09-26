"""Track 0 (final round; POST HOC, existing data): reconcile WP-27's "stuck states are often lower in loss than any placed
state found" (own 400-point training sets) with the validated population result that the global conditional minimiser
is placed (width2_direct_check.csv).  Compare, AT THE SAME SCALE, each f_a stuck state's loss on the POPULATION objective
with the validated global (placed) population minimiser.

Every f_a stuck endpoint is a duplicate-unit configuration (WP-27, 2A): the network function is s·σ·f_a(α₁x + β₁) with
σ = ±1.  σ is identified by recomputing the profiled own-sample loss for both signs and matching the recorded endpoint
loss (width2_basins/endpoints.csv, loss_endpoint) to 1e−8; α₁, β₁ from the recorded replay (width2_nogating_parts/
replays.csv, step 16,000).  Losses: width2_unplaced.loss (profiled output bias), population = width2_conditional.population.
The validated population minimiser exists at R₂ = 0.001, 0.003, 0.01, 0.02 (direct check); endpoints are at R₂ = 0.003,
0.01, 0.03, 0.1, so the comparison covers R₂ = 0.003 and 0.01 and is reported as not available at 0.03 and 0.1.

    python -m src.width2_reconcile
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "width2_basins"


def main():
    from .width2_conditional import population, training_set
    from .width2_nogating import act_of
    from .width2_unplaced import loss
    e = pd.read_csv(OUT / "endpoints.csv", float_precision="round_trip")
    e = e[e.act.str.startswith("f") & e.duplicate]
    rp = pd.read_csv(RESULTS / "width2_nogating_parts" / "replays.csv", float_precision="round_trip")
    rp = rp[rp.step == 16_000]
    dc = pd.read_csv(RESULTS / "width2_direct_check.csv", float_precision="round_trip")
    xp, yp = population()
    rows = []
    for r in e.itertuples():
        q = rp[(rp.act == r.act) & (rp.seed == r.seed) & (rp.R2 == r.R2) & (rp.variant == r.variant)].iloc[0]
        act = act_of(r.act)
        x, y = training_set(int(r.seed))
        cand = {}
        for sg in (1.0, -1.0):
            z = np.array([q.alpha1, q.beta1, q.alpha1, q.beta1, 0.5 * sg, 0.5 * sg])
            cand[sg] = (z, loss(z, r.s, x, y, act))
        sg = min(cand, key=lambda k: abs(cand[k][1] - r.loss_endpoint))
        z, L_own = cand[sg]
        m = dc[(dc.act == r.act) & np.isclose(dc.R2, r.R2)]
        L_pop = loss(z, r.s, xp, yp, act)
        row = {"act": r.act, "seed": r.seed, "R2": r.R2, "variant": r.variant, "s": r.s, "sigma": sg,
               "own_loss_recomputed": L_own, "own_loss_recorded": r.loss_endpoint, "own_match": abs(L_own - r.loss_endpoint) <= 1e-8,
               "pop_loss_stuck": L_pop, "own_gap_to_placed_found": r.gap_to_placed}
        if len(m):
            m = m.iloc[0]
            row.update(pop_global_loss=float(m.retained_loss), pop_global_placed=bool(m.placed), s_direct=float(m.s),
                       same_s=bool(np.isclose(m.s, r.s, rtol=1e-12)), pop_gap=L_pop - float(m.retained_loss))
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "reconcile.csv", index=False)
    c = d[d.pop_gap.notna()]
    out = {"n_fa_endpoints": int(len(d)), "own_loss_matched": int(d.own_match.sum()),
           "n_compared": int(len(c)), "same_s_all": bool(c.same_s.all()), "direct_placed_all": bool(c.pop_global_placed.all()),
           "n_stuck_above_pop_global": int((c.pop_gap > 0).sum()),
           "pop_gap_min": float(c.pop_gap.min()), "pop_gap_median": float(c.pop_gap.median()), "pop_gap_max": float(c.pop_gap.max()),
           "by_R2": {f"{k:g}": {"n": int(len(g)), "above": int((g.pop_gap > 0).sum()), "gap_median": float(g.pop_gap.median()),
                               "gap_min": float(g.pop_gap.min())} for k, g in c.groupby("R2")},
           "n_not_comparable": int(d.pop_gap.isna().sum()),
           "own_lower_than_placed_found_in_compared": int((c.own_gap_to_placed_found < 0).sum()),
           "own_placed_found_in_compared": int(c.own_gap_to_placed_found.notna().sum())}
    (OUT / "reconcile_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
