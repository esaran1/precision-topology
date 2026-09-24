"""Check-interval sensitivity for the 50-step experiments (POST HOC, labelled; approved by the author 2026-09-24; no
registered verdict changes).  Input: every crossing run of the Block G base window (λ's data), Block 3 and the
prospective own-seed test replayed to its stored crossing with G logged at each 50-step check
(results/crossing_audit_full_runs.csv, `python -m src.crossing_audit full`; every stored crossing reproduced bit for
bit).  The crossing |w₂| is interpolated linearly in G between the last negative and the first positive check.

Each registered number is recomputed twice by the same code path: from the check-based crossings (which must reproduce
the committed value exactly -- the validation) and from the interpolated crossings (the sensitivity).

    python -m src.cadence_sensitivity
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .crossing_audit import FULL_PARTS, RESULTS

RUNS = RESULTS / "crossing_audit_full_runs.csv"
A_VALUES = (1.30, 1.50)


def runs():
    if FULL_PARTS.exists():
        d = pd.read_csv(FULL_PARTS, float_precision="round_trip")
        d.sort_values(["family", "window", "a", "seed"]).to_csv(RUNS, index=False)
    return pd.read_csv(RUNS, float_precision="round_trip")


def _ghat_windows():
    from .prospective import _cert_ghat
    return _cert_ghat()


def lam(d, col):
    """λ(a) = median crossing R / certified R_glob(base, a), as prospective.calibrate (base window, crossers)."""
    gh = _ghat_windows()
    rg = pd.read_csv(RESULTS / "prospective_rglob_all.csv")
    rg = {(r.window, round(r.a, 2)): r.R_glob for r in rg.itertuples()}
    b = d[d.family.str.startswith("Block G base")]
    out = {}
    for a in A_VALUES:
        g = b[b.a.round(2) == a]
        R = g[col] * gh[("base", a)] / 2
        out[a] = {"lambda": float(R.median() / rg[("base", a)]), "median_cross_w2": float(g[col].median()),
                  "n": len(g)}
    return out


def b2(d, col):
    """B2 = pooled median crossing R over every Block G crosser (five windows, both a), as prospective.calibrate."""
    gh = _ghat_windows()
    g = d[d.family.str.startswith("Block G")]
    return float(pd.Series([w2 * gh[(w, round(a, 2))] / 2 for w, a, w2 in zip(g.window, g.a, g[col])]).median())


def block3(d, col, lam_used=None, b1_w2=None, b2_val=None):
    """prospective.score's primary comparison (median crossing R per setting, |log error| of U, C, B1, B2, window...
    setting-level bootstrap of C − other), re-implemented; `lam_used` / `b1_w2` override C's λ and B1's base median
    |w₂| (cadence-matched variant)."""
    from .prospective import _boot_diff
    gh = _ghat_windows()
    r = d[d.family == "Block 3"].copy()
    r["R"] = [w2 * gh[(w, round(a, 2))] / 2 for w, a, w2 in zip(r.window, r.a, r[col])]
    obs = r.groupby(["window", r.a.round(2)]).agg(n_crossed=("R", "size"), obs_median_cross_R=("R", "median")).reset_index()
    pred = pd.read_csv(RESULTS / "prospective_predictions.csv")
    p1 = pred[pred.target == "median_cross_R"].pivot_table(index=["window", "a"], columns="model",
                                                          values="prediction").reset_index()
    p1["a"] = p1.a.round(2)
    m = obs.merge(p1, on=["window", "a"])
    if lam_used is not None:
        m["C"] = [lam_used[a] * u for a, u in zip(m.a, m.U)]
    if b1_w2 is not None:
        m["B1"] = [b1_w2[a] * gh[(w, a)] / 2 for w, a in zip(m.window, m.a)]
    if b2_val is not None:
        m["B2"] = b2_val
    for model in ("U", "C", "B1", "B2"):
        m[f"abslogerr_{model}"] = np.abs(np.log(m[model] / m.obs_median_cross_R))
    rows = []
    for other in ("B1", "B2", "U"):
        dd, lo, hi = _boot_diff(m["abslogerr_C"].values, m[f"abslogerr_{other}"].values)
        rows.append({"comparison": f"C - {other}", "mean_diff": dd, "ci95_lo": lo, "ci95_hi": hi,
                     "C_better_interval_excludes_zero": hi < 0})
    return m, pd.DataFrame(rows)


def own_seed(d, col, lam_used=None):
    """prospective_own.score's primary analysis (crossers) through prospective_own.comparisons; `lam_used` overrides C."""
    from .prospective_own import comparisons
    pr = pd.read_csv(RESULTS / "prospective_own_predictions.csv", float_precision="round_trip")
    r = d[d.family == "own-seed"][["window", "a", "seed", col]].merge(pr, on=["window", "a", "seed"])
    r["R_cross"] = r[col] * r.Ghat_lo / 2
    if lam_used is not None:
        r["C"] = [lam_used[round(a, 2)] * u for a, u in zip(r.a, r.U)]
    rows, cells = [], []
    for a in A_VALUES:
        g = r[r.a.round(2) == a]
        per_run = pd.DataFrame({"window": g.window, "e_own": np.log(g.R_cross / g.U_own),
                                "e_U": np.log(g.R_cross / g.U), "e_C": np.log(g.R_cross / g.C),
                                "e_Cown": np.log(g.R_cross / g.C_own)})
        for k, v in comparisons(per_run, a).items():
            rows.append({"a": a, "comparison": k, **v})
        cells.append({"a": a, "n_crossed": len(g), "median_residual_vs_U_own": float((g.R_cross / g.U_own).median() - 1)})
    return pd.DataFrame(rows), pd.DataFrame(cells)


def main():
    d = runs()
    assert d.reproduced.all(), "a replay did not reproduce its stored crossing"
    out = []
    # --- validation: the check-based path reproduces the committed numbers exactly
    from .prospective_own import LAMBDA
    lc = lam(d, "cross_w2")
    for a in A_VALUES:
        assert abs(lc[a]["lambda"] - LAMBDA[a]) < 5e-6, (a, lc[a], LAMBDA[a])
    cal = pd.read_csv(RESULTS / "prospective_calibration.csv")
    assert b2(d, "cross_w2") == float(cal.B2_pooled_median_cross_R.iloc[0]), "B2 not reproduced"
    m_chk, c_chk = block3(d, "cross_w2")
    ref = pd.read_csv(RESULTS / "prospective_comparisons.csv").set_index("comparison")
    for r in c_chk.itertuples():
        for k in ("mean_diff", "ci95_lo", "ci95_hi"):
            assert abs(getattr(r, k) - ref.loc[r.comparison, k]) < 1e-12, (r.comparison, k)
    o_chk, oc_chk = own_seed(d, "cross_w2")
    ref_o = pd.read_csv(RESULTS / "prospective_own_scores.csv")
    ref_o = ref_o[ref_o.analysis.str.startswith("primary")].set_index(["a", "comparison"])
    for r in o_chk.itertuples():
        for k in ("stat", "lo", "hi"):
            assert abs(getattr(r, k) - ref_o.loc[(r.a, r.comparison), k]) < 1e-12, (r.a, r.comparison, k)
    # --- sensitivity: interpolated crossings
    li = lam(d, "cross_w2_interp")
    for a in A_VALUES:
        out.append({"experiment": "Block G base (lambda)", "a": a, "quantity": "lambda(a)", "check_interval": 50,
                    "check_based": lc[a]["lambda"], "interpolated": li[a]["lambda"], "n": lc[a]["n"]})
    m_int, c_int = block3(d, "cross_w2_interp")
    m_mat, c_mat = block3(d, "cross_w2_interp", lam_used={a: li[a]["lambda"] for a in A_VALUES},
                          b1_w2={a: li[a]["median_cross_w2"] for a in A_VALUES}, b2_val=b2(d, "cross_w2_interp"))
    out.append({"experiment": "Block G all windows (B2)", "a": np.nan, "quantity": "B2 pooled median crossing R",
                "check_interval": 50, "check_based": b2(d, "cross_w2"), "interpolated": b2(d, "cross_w2_interp"),
                "n": int(d.family.str.startswith("Block G").sum())})
    for (lab, cc) in (("registered predictions", c_int), ("cadence-matched C, B1 and B2", c_mat)):
        for r0, r1 in zip(c_chk.itertuples(), cc.itertuples()):
            for k in ("mean_diff", "ci95_lo", "ci95_hi"):
                out.append({"experiment": "Block 3", "a": np.nan, "quantity": f"{r0.comparison} {k} [{lab}]",
                            "check_interval": 50, "check_based": getattr(r0, k), "interpolated": getattr(r1, k),
                            "n": len(m_chk), "check_pass": bool(r0.C_better_interval_excludes_zero),
                            "interp_pass": bool(r1.C_better_interval_excludes_zero)})
    for a in A_VALUES:
        for k in ("U", "C"):
            for (lab, mm) in (("registered predictions", m_int), ("cadence-matched", m_mat)):
                if k == "U" and lab == "cadence-matched":
                    continue
                out.append({"experiment": "Block 3", "a": a, "quantity": f"mean |log error| {k} [{lab}]",
                            "check_interval": 50,
                            "check_based": float(m_chk[m_chk.a == a][f"abslogerr_{k}"].mean()),
                            "interpolated": float(mm[mm.a == a][f"abslogerr_{k}"].mean()), "n": int((m_chk.a == a).sum())})
        R_U = lambda mm: float((mm[mm.a == a].obs_median_cross_R / mm[mm.a == a].U).median() - 1)
        out.append({"experiment": "Block 3", "a": a, "quantity": "median over settings of (median R / U) - 1",
                    "check_interval": 50, "check_based": R_U(m_chk), "interpolated": R_U(m_int),
                    "n": int((m_chk.a == a).sum())})
    o_int, oc_int = own_seed(d, "cross_w2_interp")
    o_mat, _ = own_seed(d, "cross_w2_interp", lam_used={a: li[a]["lambda"] for a in A_VALUES})
    for (lab, oo) in (("registered predictions", o_int), ("cadence-matched C", o_mat)):
        for k in range(len(o_chk)):
            r0, r1 = o_chk.iloc[k], oo.iloc[k]
            row = {"experiment": "own-seed", "a": r0["a"], "quantity": f"{r0['comparison']} stat [{lab}]",
                   "check_interval": 50, "check_based": r0["stat"], "interpolated": r1["stat"], "n": np.nan,
                   "check_lo": r0["lo"], "check_hi": r0["hi"], "interp_lo": r1["lo"], "interp_hi": r1["hi"],
                   "check_pass": bool(r0["pass"]), "interp_pass": bool(r1["pass"])}
            if "C_better_interval_above_0" in o_chk.columns and r0["comparison"] == "P2b":
                row.update(check_C_better=bool(r0["C_better_interval_above_0"]),
                           interp_C_better=bool(r1["C_better_interval_above_0"]))
            out.append(row)
    for r0, r1 in zip(oc_chk.itertuples(), oc_int.itertuples()):
        out.append({"experiment": "own-seed", "a": r0.a, "quantity": "median residual R_cross / U_own - 1",
                    "check_interval": 50, "check_based": r0.median_residual_vs_U_own,
                    "interpolated": r1.median_residual_vs_U_own, "n": r0.n_crossed})
    s = pd.DataFrame(out)
    s.to_csv(RESULTS / "cadence_sensitivity.csv", index=False)
    pd.concat([o_chk.assign(crossing="check"), o_int.assign(crossing="interpolated"),
               o_mat.assign(crossing="interpolated, cadence-matched C")]).to_csv(
        RESULTS / "cadence_sensitivity_own_seed.csv", index=False)
    pd.concat([c_chk.assign(crossing="check"), c_int.assign(crossing="interpolated"),
               c_mat.assign(crossing="interpolated, cadence-matched C, B1 and B2")]).to_csv(
        RESULTS / "cadence_sensitivity_block3.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
