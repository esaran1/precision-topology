"""Track 4 (writer inputs): the two Phase 1 run populations, "2,400 float32" and "4,800 pooled", and whether the
float64 half re-runs the float32 runs (it does not: independent initialisations).  Producer for the numbers in WP-26.

    python -m src.track4_populations
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    d = pd.read_csv(RESULTS / "phase1_decomposition.csv")
    out = {}
    for prec, g in list(d.groupby("precision")) + [("pooled", d)]:
        out[prec] = {"runs": int(len(g)), "solved": int(g.solved.sum()),
                     "placement": int((g.failure == "placement").sum()), "bias": int((g.failure == "bias").sum())}
    f32 = d[d.precision == "float32"].set_index(["a", "seed"]).sort_index()
    f64 = d[d.precision == "float64"].set_index(["a", "seed"]).sort_index()
    assert f32.index.equals(f64.index)
    out["paired"] = {
        "pairs": int(len(f32)),
        "sign_w1_agree": float((np.sign(f32.w1) == np.sign(f64.w1)).mean()),
        "failure_class_agree": float((f32.failure.fillna("solved") == f64.failure.fillna("solved")).mean()),
        "solve_agree": float((f32.solved == f64.solved).mean()),
        "median_rel_diff_abs_w2": float(np.median(np.abs(f64.w2.abs() - f32.w2.abs()) / f32.w2.abs())),
    }
    # the float32 half is the sweep (sin_family, a > 1) plus the refinement, run for run
    sw = pd.read_csv(RESULTS / "fold1d_sweep.csv"); rf = pd.read_csv(RESULTS / "fold1d_refine.csv")
    src = pd.concat([sw[(sw.activation == "sin_family") & (sw.parameter > 1.0)], rf[rf.activation == "sin_family"]])
    src = src.assign(a=src.parameter.round(2)).set_index(["a", "seed"]).sort_index()
    f = f32.copy(); f.index = f.index.set_levels(f.index.levels[0].round(2), level=0)
    j = f.join(src[["solved", "w2_abs"]], rsuffix="_sweep", how="inner")
    out["float32_equals_sweep"] = {"matched": int(len(j)), "solved_agree": float((j.solved == j.solved_sweep).mean()),
                                   "max_abs_diff_abs_w2": float((j.w2.abs() - j.w2_abs).abs().max())}
    for prec, g in d[d.a.round(2) == 1.02].groupby("precision"):
        out[f"a1.02_{prec}"] = {"runs": int(len(g)), "solved": int(g.solved.sum()),
                                "median_abs_w2": float(g.w2.abs().median()), "max_abs_w2": float(g.w2.abs().max())}
    f = d[d.precision == "float32"]
    per_a = f.groupby(f.a.round(2)).agg(runs=("solved", "size"), solved=("solved", "sum"),
                                         placement=("failure", lambda v: int((v == "placement").sum())),
                                         bias=("failure", lambda v: int((v == "bias").sum()))).reset_index()
    per_a.to_csv(RESULTS / "track4_float32_decomposition.csv", index=False)
    out["float32_per_a"] = per_a.to_dict("records")
    (RESULTS / "track4_populations.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
