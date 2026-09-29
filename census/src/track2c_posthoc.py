"""POST HOC for registered test 2C (labelled; written after the registered scores, results/track2c/scores.json).

The registered file src/track2c.py is left unchanged (its SHA-256 is in results/track2c/registration.sha256).

    python -m src.track2c_posthoc      # -> results/track2c/posthoc_unscored.csv, posthoc_summary.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .track2c import RESULTS, bootstrap_mean_ci

OUT = RESULTS / "track2c"


def run():
    """The crossing runs without a pre-crossing prediction (they crossed before |w₂| reached s_own,d).  Per run:
    s_obs/s_own,d; the mirror occupied at the rule point (from the path after the crossing, s path only) against the
    global minimiser's mirror; and s_obs against the frozen switch of the rule-point mirror.  Per cell, the registered
    statistics recomputed on all crossing runs with the static floor s_pred = s_own,d for runs without a pre-crossing
    prediction (NOT a registered rule)."""
    m = pd.read_csv(OUT / "observed_runs.csv")
    fz = pd.read_csv(OUT / "frozen_seeds.csv")
    for t in (m, fz):
        t["a"] = t.a.round(2)
    m = m.merge(fz[["seed", "d", "a", "s_mirror_+", "s_mirror_-"]], on=["seed", "d", "a"], how="left")
    c = m[m.crossed.astype(bool) & ~m.placed_at_init.astype(bool)].copy()
    c["s_rule_mirror"] = np.where(c.mirror_rule == "+", c["s_mirror_+"],
                                  np.where(c.mirror_rule == "-", c["s_mirror_-"], np.nan))
    c["obs_over_s_own_d"] = c.s_obs / c.s_own_d
    c["obs_over_rule_mirror_switch"] = c.s_obs / c.s_rule_mirror
    c["rule_mirror_is_global"] = c.mirror_rule == c.global_mirror_hi
    un = c[~c.scored.astype(bool)]
    un[["d", "a", "seed", "step_obs", "t_sw", "s_obs", "s_own_d", "obs_over_s_own_d", "mirror_rule", "global_mirror_hi",
        "rule_mirror_is_global", "s_rule_mirror", "obs_over_rule_mirror_switch", "status"]].assign(
        label="POST HOC").to_csv(OUT / "posthoc_unscored.csv", index=False)
    rows = []
    for (d, a), g in c.groupby(["d", "a"]):
        u = g[~g.scored.astype(bool)]
        sp = np.where(g.scored.astype(bool), g.s_pred, g.s_own_d)
        err = np.abs(np.log(g.s_obs / sp))
        D = err - np.abs(np.log(g.s_obs / g.s_own_x1))
        mD, lo, hi = bootstrap_mean_ci(D.to_numpy(float))
        rows.append({"label": "POST HOC", "d": int(d), "a": a, "n_crossing": len(g), "n_unscored": len(u),
                     "n_unscored_rule_mirror_not_global": int((~u.rule_mirror_is_global).sum()),
                     "n_unscored_within_1pct_of_rule_mirror_switch":
                         int(((u.obs_over_rule_mirror_switch - 1).abs() <= 0.01).sum()),
                     "unscored_obs_over_s_own_d_min": float(u.obs_over_s_own_d.min()) if len(u) else np.nan,
                     "unscored_obs_over_s_own_d_max": float(u.obs_over_s_own_d.max()) if len(u) else np.nan,
                     "all_crossing_floor_median_abs_log_err": float(np.median(err)),
                     "all_crossing_floor_C3_mean_D": mD, "all_crossing_floor_C3_ci95_hi": hi})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "posthoc_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.to_string(index=False))
    return S


if __name__ == "__main__":
    run()
