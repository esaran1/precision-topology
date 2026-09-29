"""POST HOC (after the registered 2B score; labelled; nothing here changes a verdict): where the realised stability
check (η_cell·λ_max(H_own) ≤ 1) was violated in test 2B, and how the realised λ_max compares with the population
branch's λ_max at the same scale.

Reads the committed predictions/observations and the saved paths (hashes asserted).  Per run with a violation:
  t_viol        the first step on its realised trajectory with η·λ_max > 1
  s_viol/s_br   the scale at that step over the run's frozen switch
  before_switch whether s_viol < s_branch (the violation came before the branch switch, i.e. in the tracking phase)
  lam_ratio     λ_max(H_own) at t_viol over the population-branch λ_max at s_viol (interpolated on the frozen grid)
  steps_to_cross crossing step − t_viol (crossing runs)
Also per cell: the fraction of the ramp's steps before the crossing that violate, and the median ratio of the valid
cells against κχ (descriptive).

    python -m src.track2b_posthoc      -> results/track2b/posthoc_stability.csv, results/track2b/posthoc_stability.json
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import track2b as X


def run():
    X._assert_registration()
    X._assert_predictions()
    o = X._read(X.OUT / "observed_runs.csv")
    br = X._read(X.OUT / "population_branch.csv")
    d = X._read(X.OUT / "design.csv")
    rows = []
    for r in o.itertuples():
        pf = X._path_file(r.a, r.kx_target, r.seed)
        assert hashlib.sha256(pf.read_bytes()).hexdigest() == r.path_sha256, pf
        lam = np.load(pf)["lam_max"]
        end = int(r.step_cross) if r.crossed else int(r.n_steps)
        e = r.eta_cell * lam[:end]
        v = np.nonzero(~(e <= 1.0))[0]
        s0 = X.S0_FRAC * X.R._s_star_pop(r.a)
        row = {"a": r.a, "kx_target": r.kx_target, "seed": r.seed, "crossed": bool(r.crossed),
               "violated": bool(len(v)), "n_steps_realised": end, "n_steps_violating": int(len(v))}
        if len(v):
            t = int(v[0]) + 1
            s = X.scale_at(t, s0, r.gamma)
            b = br[br.a.round(2) == round(r.a, 2)]
            lam_pop = float(np.interp(s, b.s, b.lambda_max))
            row.update(t_viol=t, t_viol_after_warmup=t - X.WARMUP, s_viol_over_s_branch=s / r.s_branch,
                       before_switch=bool(s < r.s_branch), lam_ratio=float(lam[t - 1] / lam_pop),
                       steps_to_cross=(int(r.step_cross) - t) if r.crossed else None)
        rows.append(row)
    P = pd.DataFrame(rows)
    P.to_csv(X.OUT / "posthoc_stability.csv", index=False)
    cells = []
    for c in d.itertuples():
        g = P[(P.a.round(2) == round(c.a, 2)) & np.isclose(P.kx_target, c.kx_target)]
        gv = g[g.violated]
        cells.append({"a": c.a, "kx_target": c.kx_target, "n_violated": int(len(gv)),
                      "frac_realised_steps_violating": float(g.n_steps_violating.sum() / g.n_steps_realised.sum()),
                      "median_t_viol_after_warmup": float(gv.t_viol_after_warmup.median()) if len(gv) else None,
                      "median_s_viol_over_s_branch": float(gv.s_viol_over_s_branch.median()) if len(gv) else None,
                      "n_viol_before_switch": int(gv.before_switch.sum()) if len(gv) else 0,
                      "median_lam_ratio_at_viol": float(gv.lam_ratio.median()) if len(gv) else None,
                      "median_steps_viol_to_cross": float(gv.steps_to_cross.dropna().median())
                      if len(gv) and gv.steps_to_cross.notna().any() else None})
    out = {"label": "POST HOC (after the registered score); descriptive; no verdict changes", "cells": cells}
    (X.OUT / "posthoc_stability.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(cells).to_string(index=False))
    return out


if __name__ == "__main__":
    run()
