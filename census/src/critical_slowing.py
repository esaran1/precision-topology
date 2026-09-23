"""Exploratory (POST HOC): critical slowing at fixed scale, and whether it reproduces the free-training offset.

Labelled post hoc throughout: the free-training crossings (phase 2b) were seen before this analysis.

1. Relaxation law.  From Block 4's fixed-scale replays ("preserved", a = 1.30): time to first placement
   (first 25-step check with G > 0) at each level above the threshold, fitted as
       log τ = c − ν log(R/R_glob − 1)
   on the per-level medians (levels 1.1, 1.25, 1.5, 2.0; level 1.0 sits at the threshold).
2. Integration along free training.  For each phase 2b trajectory (a = 1.30, budget 32,000, 40 seeds;
   R = |w2|·Ĝ_cert/2 per checkpoint), accumulate progress = ∫ dt / τ(R(t)/R_glob) over steps with
   R > R_glob, linear interpolation between checkpoints.  Predicted crossing = the R at which progress
   first reaches 1.  Compared with each run's observed crossing R (the certified per-a table).
   a = 1.50 has no fixed-scale replays; applying the a = 1.30 law there is a labelled transfer.

    python -m src.critical_slowing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def relaxation_law():
    d = pd.read_csv(RESULTS / "fixed_scale_block4.csv")
    d = d[(d.variant == "preserved") & d.first_hit.notna() & (d.level > 1.0)]
    med = d.groupby("level").first_hit.agg(["median", "size"]).reset_index()
    x = np.log(med.level - 1.0); y = np.log(med["median"])
    slope, icept = np.polyfit(x, y, 1)
    return med, -slope, icept


def predicted_crossings(a, nu, c, R_glob):
    cols = ["a", "budget", "seed", "step", "w2"]
    parts = []
    for ch in pd.read_csv(RESULTS / "phase2b_checkpoints.csv", usecols=cols, chunksize=2_000_000):
        parts.append(ch[(ch.a.round(2) == a) & (ch.budget == 32_000)])
    d = pd.concat(parts)
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv")
    G = float(g[g.a.round(2) == a].Ghat_certified.iloc[0])
    out = []
    for seed, t in d.groupby("seed"):
        t = t.sort_values("step")
        steps = t.step.values.astype(float)
        R = t.w2.abs().values * G / 2
        grid = np.arange(steps[0], steps[-1] + 1, 5.0)
        Rg = np.interp(grid, steps, R)
        u = Rg / R_glob - 1.0
        rate = np.where(u > 0, np.exp(-c) * np.power(np.maximum(u, 1e-12), nu), 0.0)
        prog = np.cumsum(rate) * 5.0
        hit = np.argmax(prog >= 1.0) if (prog >= 1.0).any() else None
        out.append({"a": a, "seed": seed, "pred_cross_R": Rg[hit] if hit is not None else np.nan,
                    "pred_cross_step": grid[hit] if hit is not None else np.nan})
    return pd.DataFrame(out)


def main():
    med, nu, c = relaxation_law()
    print("relaxation law (a = 1.30, preserved): per-level median time to placement")
    print(med.to_string(index=False))
    print(f"fit: tau = exp({c:.3f}) * (R/R_glob - 1)^(-{nu:.3f})")
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    obs = pd.read_csv(RESULTS / "wi_crossing_runs.csv")
    rows = []
    for a in (1.30, 1.50):
        b = br[(br.a.round(2) == a) & (br.kind == "glob")].iloc[0]
        R_glob = 0.5 * (b.R_lo + b.R_hi)
        p = predicted_crossings(a, nu, c, R_glob)
        o = obs[(obs.a.round(2) == a) & (obs.budget == 32_000)][["seed", "R_cert"]]
        m = p.merge(o, on="seed", how="inner")
        rows.append({"a": a, "label": "fitted at a = 1.30 (post hoc)" if a == 1.30 else "transfer of the a = 1.30 law (post hoc)",
                     "R_glob_cert": R_glob, "n": len(m), "pred_median_R": m.pred_cross_R.median(),
                     "obs_median_R": m.R_cert.median(),
                     "pred_offset_pct": (m.pred_cross_R.median() / R_glob - 1) * 100,
                     "obs_offset_pct": (m.R_cert.median() / R_glob - 1) * 100,
                     "median_abs_log_err_per_run": float(np.median(np.abs(np.log(m.pred_cross_R / m.R_cert))))})
    t = pd.DataFrame(rows)
    t["nu"], t["c"] = nu, c
    t.to_csv(RESULTS / "critical_slowing_posthoc.csv", index=False)
    med.to_csv(RESULTS / "critical_slowing_relaxation.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
