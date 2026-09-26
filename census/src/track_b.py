"""Track B (final round; statistics on existing data; POST HOC where labelled).

  1. run-level bootstrap 95% intervals (10,000 resamples, seed 20260926) on every arm median lag in the lag figure
     (lag from each run's tracked-branch switch, linear_response/compare_runs.csv r_obs) → track_b/arm_ci.csv;
  2. for all 36 arms, the fraction of runs within ±10% and ±20% relative (no absolute floor), closed form (the committed
     κ_kχ of lag_law/predictions.csv) and trajectory-integrated (the full model from 0.7 s*), against the tracked-branch
     lag; also the registered-reference closed form (obs_r_1A) → track_b/within.csv, within_summary.json;
  3. collapse data: observed/predicted lag against χ for every setting with data (width-1 arms; forced ramp cells; GELU,
     SiLU, Mish; the R^d band task; width-2 χ bins), marking which points are branch-conditioned post hoc; points whose
     median predicted lag is below 0.005 in magnitude are excluded as below resolution (stated) → track_b/collapse.csv;
  4. sample-size decomposition (sample_size_free.csv, sample_size_own.csv): per run log(s_cross/s_pop) =
     log(s_own/s_pop) [static finite-sample threshold shift] + log(s_cross/s_own) [dynamic lag]; per (a, n) medians with
     bootstrap intervals → track_b/sample_size_decomposition.csv;
  5. the inputs to κ, by origin → track_b/kappa_inputs.csv.

    python -m src.track_b
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "track_b"
SEED, B = 20260926, 10_000


def boot_median(v, rng, B=B):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(v)


def _norm(v):
    return str(float(v)) if str(v).replace(".", "", 1).isdigit() else str(v)


def runs():
    c = pd.read_csv(RESULTS / "linear_response" / "compare_runs.csv"); c["a"] = c.a.round(2); c["arm"] = c.arm.map(_norm)
    p = pd.read_csv(RESULTS / "lag_law" / "predictions.csv"); p["a"] = p.a.round(2); p["arm"] = p.arm.astype(str).map(_norm)
    m = c.merge(p[["set", "a", "seed", "arm", "pred_r"]], on=["set", "a", "seed", "arm"], how="left")
    assert m.pred_r.notna().all() and len(m) == 1750
    return m


def arm_ci(m):
    rng = np.random.default_rng(SEED)
    rows = []
    for (st, a, arm), g in m.groupby(["set", "a", "arm"], sort=False):
        med, lo, hi, n = boot_median(g.r_obs, rng)
        rows.append({"set": st, "a": a, "arm": arm, "n": n, "median_r_obs": med, "ci_lo": lo, "ci_hi": hi,
                     "median_chi_1A": float(g.chi_1A.median())})
    d = pd.DataFrame(rows); d.to_csv(OUT / "arm_ci.csv", index=False); return d


def within(m):
    rows = []
    specs = {"closed form (committed κχ), tracked-branch lag": ("r_obs", "pred_r"),
             "closed form (committed κχ), registered reference (global own threshold)": ("obs_r_1A", "pred_r"),
             "trajectory-integrated (full model, 0.7 s*), tracked-branch lag": ("r_obs", "full_7_r_pred")}
    for name, (o, pcol) in specs.items():
        for (st, a, arm), g in m.groupby(["set", "a", "arm"], sort=False):
            ok = g[np.isfinite(g[o]) & np.isfinite(g[pcol]) & (g[pcol] != 0)]
            q = ok[o] / ok[pcol]
            rows.append({"model": name, "set": st, "a": a, "arm": arm, "n": len(ok),
                         "frac_within_10": float((np.abs(q - 1) <= 0.10).mean()), "frac_within_20": float((np.abs(q - 1) <= 0.20).mean()),
                         "arm_median_ratio": float(ok[o].median() / ok[pcol].median())})
    d = pd.DataFrame(rows); d.to_csv(OUT / "within.csv", index=False)
    S = {}
    for name, g in d.groupby("model", sort=False):
        S[name] = {"arms": int(len(g)), "runs": int(g.n.sum()),
                   "pooled_frac_within_10": float((g.frac_within_10 * g.n).sum() / g.n.sum()),
                   "pooled_frac_within_20": float((g.frac_within_20 * g.n).sum() / g.n.sum()),
                   "arms_median_within_10": int((np.abs(g.arm_median_ratio - 1) <= 0.10).sum()),
                   "arms_median_within_20": int((np.abs(g.arm_median_ratio - 1) <= 0.20).sum()),
                   "per_arm_frac10_min": float(g.frac_within_10.min()), "per_arm_frac10_max": float(g.frac_within_10.max()),
                   "per_arm_frac20_min": float(g.frac_within_20.min()), "per_arm_frac20_max": float(g.frac_within_20.max())}
    (OUT / "within_summary.json").write_text(json.dumps(S, indent=1))
    return d, S


def collapse(m):
    rows = []
    for (st, a, arm), g in m.groupby(["set", "a", "arm"], sort=False):
        rows.append({"setting": "width 1, free training", "label": f"{st} a={a:.2f} {arm}", "chi": float(g.chi_1A.median()),
                     "obs": float(g.r_obs.median()), "pred": float(g.pred_r.median()), "branch_conditioned_post_hoc": True,
                     "reference": "tracked-branch switch (post hoc)"})
        rows.append({"setting": "width 1, free training (registered reference)", "label": f"{st} a={a:.2f} {arm}",
                     "chi": float(g.chi_1A.median()), "obs": float(g.obs_r_1A.median()), "pred": float(g.pred_r.median()),
                     "branch_conditioned_post_hoc": False, "reference": "global own threshold (committed comparison)"})
    rp = pd.read_csv(RESULTS / "ramp" / "posthoc_scored_runs.csv")
    for (a, k, opt, cell), g in rp.groupby(["a", "winding", "opt", "cell"]):
        rows.append({"setting": f"forced ramp ({opt})", "label": f"a={a:.2f} k={k} cell {cell}", "chi": float(g.chi.median()),
                     "obs": float(g.obs_r_branch.median()), "pred": float(g.pred_r.median()), "branch_conditioned_post_hoc": True,
                     "reference": "tracked-branch switch (post hoc)"})
    for act in ("gelu", "silu", "mish"):
        g = pd.read_csv(RESULTS / "act_general" / f"posthoc2_runs_{act}.csv")
        rows.append({"setting": "GELU/SiLU/Mish", "label": act, "chi": float(g.chi.median()), "obs": float(g.r_b.median()),
                     "pred": float(g.pred.median()), "branch_conditioned_post_hoc": True, "reference": "tracked-branch switch (post hoc)"})
    bs = pd.read_csv(RESULTS / "band_rd" / "posthoc_summary.csv"); br = pd.read_csv(RESULTS / "band_rd" / "posthoc_scored_runs.csv")
    for r in bs[bs.d > 1].itertuples():
        g = br[(br.arm == r.arm) & (br.d == r.d) & (br.a.round(2) == round(r.a, 2)) & br.crossed]
        rows.append({"setting": "R^d band task", "label": f"{r.arm} d={r.d} a={r.a:.2f}", "chi": float(g.chi.median()),
                     "obs": float(r.median_r_branch), "pred": float(r.median_pred_r), "branch_conditioned_post_hoc": True,
                     "reference": "own R^d branch switch (post hoc)"})
    for arm in ("T2-3", "T2-3b", "T2-3c", "T2-3d"):
        w = pd.read_csv(RESULTS / "width2_lag" / f"parts_{arm}.csv")
        w = w[(w.status == "ok") & np.isfinite(w.chi) & np.isfinite(w.r_obs) & np.isfinite(w.r_pred)]
        for lo, hi in ((0, 1e-4), (1e-4, 0.01), (0.01, 0.0637), (0.0637, 0.3), (0.3, 3.0), (3.0, np.inf)):
            g = w[(w.chi > lo) & (w.chi <= hi)]
            if len(g) >= 3:
                rows.append({"setting": "width 2", "label": f"{arm} chi in ({lo:g}, {hi:g}]", "chi": float(g.chi.median()),
                             "obs": float(g.r_obs.median()), "pred": float(g.r_pred.median()),
                             "branch_conditioned_post_hoc": True, "reference": "tracked-branch switch along the path (post hoc)"})
    bd = RESULTS / "ramp_boundary" / "scored_runs.csv"
    if bd.exists():                              # registered boundary test (final night): valid-crossing cells only
        V = json.loads((RESULTS / "ramp_boundary" / "verdicts.json").read_text())
        ok = {(round(c["a"], 2), c["chi_target"]) for c in V["cells"] if c["n_cross"] >= 30}
        b = pd.read_csv(bd)
        for (a, c), g in b.groupby([b.a.round(2), "chi_target"]):
            if (a, c) not in ok:
                continue
            rows.append({"setting": "forced ramp, boundary test (sgd)", "label": f"a={a:.2f} chi_target={c}",
                         "chi": float(g.chi_own.median()), "obs": float(g.obs_r.median()), "pred": float(g.pred_r.median()),
                         "branch_conditioned_post_hoc": False, "reference": "forced-branch switch frozen before any run (registered)"})
    d = pd.DataFrame(rows)
    d["ratio"] = d.obs / d.pred
    d["resolved"] = d.pred.abs() >= 0.005
    d.to_csv(OUT / "collapse.csv", index=False)
    return d


def sample_size():
    f = pd.read_csv(RESULTS / "sample_size_free.csv"); o = pd.read_csv(RESULTS / "sample_size_own.csv")
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv").query("kind == 'glob'")
    pop = {round(r.a, 2): 0.5 * (r.w2_lo + r.w2_hi) for r in br.itertuples()}
    f["a"] = f.a.round(2); o["a"] = o.a.round(2)
    m = f.merge(o[["a", "n", "seed", "w2_own"]], on=["a", "n", "seed"]).dropna(subset=["w2_abs", "w2_own"])
    m["s_pop"] = m.a.map(pop)
    m["total"] = np.log(m.w2_abs / m.s_pop); m["static"] = np.log(m.w2_own / m.s_pop); m["dynamic"] = np.log(m.w2_abs / m.w2_own)
    rng = np.random.default_rng(SEED)
    rows = []
    for (a, n), g in m.groupby(["a", "n"]):
        row = {"a": a, "n": int(n), "runs": len(g)}
        for k in ("total", "static", "dynamic"):
            med, lo, hi, _ = boot_median(g[k], rng)
            row.update({f"{k}_median": med, f"{k}_lo": lo, f"{k}_hi": hi})
        row["static_share_of_total_median"] = row["static_median"] / row["total_median"]
        rows.append(row)
    d = pd.DataFrame(rows); d.to_csv(OUT / "sample_size_decomposition.csv", index=False)
    return d


def kappa_inputs():
    rows = [
        ("H (Hessian in w₁, b₁, b₂ at the switch)", "landscape", "population objective, branch point θ*(s*) (lag_law.switch)"),
        ("θ*′ (branch tangent)", "landscape", "−H⁻¹∂ₛ∇L at s*"),
        ("∇G (gap gradient on the active pair)", "landscape", "central differences of the exact-extrema gap at θ*(s*)"),
        ("s* (switch)", "landscape", "root of G(θ*(s)) on the certified branch, inside the certified bracket"),
        ("winding k", "measured runs (1A)", "each run's b₁ at its crossing, relative to b₁*; every run was k = −1 at a = 1.30, 0 elsewhere"),
        ("P shape for Adam", "measured runs (1A)", "per-coordinate median of 1/(√v̂+ε) at crossing over existing runs (48–80 per a)"),
        ("P for SGD", "fixed", "P = I"),
        ("χ (per run)", "measured runs", "growth rate of s and the run's own v̂ at crossing (residual_timescale); not part of κ"),
    ]
    d = pd.DataFrame(rows, columns=["input", "origin", "detail"]); d.to_csv(OUT / "kappa_inputs.csv", index=False)
    return d


def kappa_chi_range(m):
    """The predicted lag κχ (committed κ_k times each run's χ) across the 36 free-training arms: arm medians and runs."""
    arm = m.groupby(["set", "a", "arm"]).pred_r.median()
    out = {"arm_median_min": float(arm.min()), "arm_median_max": float(arm.max()),
           "run_q05": float(m.pred_r.quantile(0.05)), "run_q95": float(m.pred_r.quantile(0.95)), "run_max": float(m.pred_r.max()),
           "arms": int(len(arm)), "runs": int(len(m)), "arm_of_max": " ".join(map(str, arm.idxmax())), "arm_of_min": " ".join(map(str, arm.idxmin()))}
    (OUT / "kappa_chi_range.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    m = runs()
    ci = arm_ci(m)
    w, S = within(m)
    c = collapse(m)
    ss = sample_size()
    kappa_inputs()
    print("kappa*chi range:", kappa_chi_range(m))
    summ = {"bootstrap": {"B": B, "seed": SEED}, "within": S,
            "collapse": {"points": int(len(c)), "resolved": int(c.resolved.sum()),
                         "resolved_ratio_range_chi_le_0.06": [float(c[c.resolved & (c.chi <= 0.06) & c.branch_conditioned_post_hoc].ratio.min()),
                                                               float(c[c.resolved & (c.chi <= 0.06) & c.branch_conditioned_post_hoc].ratio.max())]},
            "sample_size": ss.to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps(S, indent=1)); print(ss.round(4).to_string(index=False))
    print(c[c.resolved].groupby("setting").agg(n=("ratio", "size"), rmin=("ratio", "min"), rmax=("ratio", "max"), chi_med=("chi", "median")).round(3))


if __name__ == "__main__":
    main()
