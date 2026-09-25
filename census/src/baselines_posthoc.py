"""Stronger baselines for the Block 3 held-out test and the prospective own-seed test (POST HOC, Track 6).

POST HOC: every baseline here was specified after both tests were scored.  No registered verdict changes.

Training data (the only data any baseline is fitted on): the ten Block G settings (five windows x a in {1.30, 1.50};
results/blockG_crossings.csv crossers, R = crossing |w2| x certified Ĝ(window, a) / 2 exactly as
prospective.calibrate), the certified R_glob of those ten settings (prospective_rglob_all.csv), their certified Ĝ
(prospective_ghat.csv) and the limit-gap constant K of each Block G window (limit_windows.csv K_lo; κ0 = K / D∞,
D∞ = 4√2/3, as prospective_windows).  Nothing from Block 3 runs or own-seed runs enters any fit; `fit_baselines`
refuses a training table containing any other window.

Baselines (fixed in this file before any of them was scored):
  PL   U with a pooled lag factor: λ_pool(a) = median over every Block G crosser at a (all five windows pooled) of
       R_cross / R_glob(window, a);  prediction λ_pool(a) · U.
  PL5  variant of PL: λ5(a) = median over the five Block G windows of (median crossing R / R_glob);  λ5(a) · U.
  RK   no conditional threshold: OLS over the ten Block G settings of log(median crossing R) on
       [1, log κ0(window), 1{a = 1.50}];  prediction exp(β0 + β1 log κ0 + β2 1{a = 1.50}).
  RKa  variant of RK with a separate intercept and slope per a: per-a OLS (five settings) of log(median crossing R)
       on [1, log κ0(window)].
  RG   gap size alone: per-a OLS (five settings) of log(median crossing R) on [1, log Ĝ(window, a)].

Scoring reuses the registered code: Block 3 = prospective.score's primary metric (|log(pred / median crossing R)| per
setting) and prospective._boot_diff (8 settings, 10,000 resamples, seed 0) for mean(err_C - err_X); own-seed =
prospective_own.score's per-run residuals (crossers) and prospective_own._win_boot (window-level, 10,000 resamples,
seed 0) for mean over windows of (mean |e_C| - mean |e_X|), per a.  Before anything new is reported, the committed
C, B1, B2, U numbers (prospective_comparisons.csv, prospective_scores.csv, prospective_own_scores.csv,
prospective_own_settings_scored.csv, prospective_calibration.csv) are reproduced to 1e-12.

    python -m src.baselines_posthoc
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .prospective import RESULTS, _boot_diff, _cert_ghat
from .prospective_own import B_WIN, _win_boot, comparisons

A_VALUES = (1.30, 1.50)
D_INF = 4 * math.sqrt(2) / 3
BLOCKG = ("base", "G1_wide_gap", "G2_narrow_gap", "G3_far_outer", "G4_shifted")
BASELINES = ("PL", "PL5", "RK", "RKa", "RG")
TOL = 1e-12


# ------------------------------------------------------------------------------------------ pure pieces (tested)
def abslogerr(pred, obs):
    return np.abs(np.log(np.asarray(pred, float) / np.asarray(obs, float)))


def ols(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fit_baselines(settings, runs):
    """settings: one row per training setting with window, a, median_cross_R, R_glob, Ghat, kappa0.
    runs: one row per training crosser with window, a, R (crossing R).  Returns a dict of fitted parameters."""
    bad = set(settings.window) - set(BLOCKG) | set(runs.window) - set(BLOCKG)
    if bad:
        raise ValueError(f"training data outside Block G: {sorted(bad)}")
    s = settings.assign(a=settings.a.round(2))
    r = runs.assign(a=runs.a.round(2))
    rg = {(w, a): v for w, a, v in zip(s.window, s.a, s.R_glob)}
    fit = {"PL": {}, "PL5": {}, "RKa": {}, "RG": {}}
    for a in A_VALUES:
        ra = r[r.a == a]
        fit["PL"][a] = float(np.median(ra.R.values / np.array([rg[(w, a)] for w in ra.window])))
        sa = s[s.a == a]
        fit["PL5"][a] = float(np.median(sa.median_cross_R.values / sa.R_glob.values))
        y = np.log(sa.median_cross_R.values)
        fit["RKa"][a] = ols(np.column_stack([np.ones(len(sa)), np.log(sa.kappa0.values)]), y)
        fit["RG"][a] = ols(np.column_stack([np.ones(len(sa)), np.log(sa.Ghat.values)]), y)
    X = np.column_stack([np.ones(len(s)), np.log(s.kappa0.values), (s.a == 1.50).astype(float)])
    fit["RK"] = ols(X, np.log(s.median_cross_R.values))
    return fit


def predict_baselines(fit, a, U, kappa0, Ghat):
    a = round(float(a), 2)
    lk, lg = math.log(kappa0), math.log(Ghat)
    b = fit["RK"]
    return {"PL": fit["PL"][a] * U, "PL5": fit["PL5"][a] * U,
            "RK": math.exp(b[0] + b[1] * lk + b[2] * (a == 1.50)),
            "RKa": math.exp(fit["RKa"][a][0] + fit["RKa"][a][1] * lk),
            "RG": math.exp(fit["RG"][a][0] + fit["RG"][a][1] * lg)}


def setting_compare(m, models, ref="C"):
    """Block 3's registered comparison: per-setting |log error|, bootstrap of mean(err_ref - err_X)."""
    rows = []
    for x in models:
        d, lo, hi = _boot_diff(m[f"abslogerr_{ref}"].values, m[f"abslogerr_{x}"].values)
        rows.append({"comparison": f"{ref} - {x}", "mean_diff": d, "ci95_lo": lo, "ci95_hi": hi,
                     "C_better_interval_excludes_zero": hi < 0, "X_better_interval_excludes_zero": lo > 0})
    return pd.DataFrame(rows)


def window_compare(per_run, ref, x, B=B_WIN):
    """Own-seed: per-run signed errors e_<model>; window means of |e|; window bootstrap of mean(|e_ref| - |e_x|)."""
    g = per_run.assign(**{c: per_run[c].abs() for c in (f"e_{ref}", f"e_{x}")}).groupby("window")
    mean = g[[f"e_{ref}", f"e_{x}"]].mean()
    d, lo, hi = _win_boot(mean[f"e_{ref}"] - mean[f"e_{x}"], B)
    return {"mean_diff": d, "ci95_lo": lo, "ci95_hi": hi, "C_better_interval_excludes_zero": hi < 0,
            "X_better_interval_excludes_zero": lo > 0}


# ------------------------------------------------------------------------------------------ data
def _kappa0():
    lw = pd.read_csv(RESULTS / "limit_windows.csv", float_precision="round_trip")
    k = {w: K / D_INF for w, K in zip(lw.window, lw.K_lo)}
    held = pd.read_csv(RESULTS / "prospective_windows.csv", float_precision="round_trip")
    for r in held.itertuples():
        tag = f"H{int(round(r.quantile * 100)):02d}"
        assert abs(r.kappa0_lo - r.K_lo / D_INF) < 1e-15 and abs(k[tag] - r.kappa0_lo) < 1e-15, tag
    own = pd.read_csv(RESULTS / "prospective_own_windows.csv", float_precision="round_trip")
    for r in own.itertuples():
        assert abs(r.kappa0_lo - r.K_lo / D_INF) < 1e-15
        k[r.name] = r.kappa0_lo
    return k


def training_data():
    """Block G only, exactly as prospective.calibrate."""
    gh = _cert_ghat()
    bg = pd.read_csv(RESULTS / "blockG_crossings.csv")
    bg = bg[bg.cross_step.notna()].copy()
    bg["R"] = [w2 * gh[(win, round(a, 2))] / 2 for win, a, w2 in zip(bg.window, bg.a, bg.cross_w2)]
    bg["a"] = bg.a.round(2)
    rg = pd.read_csv(RESULTS / "prospective_rglob_all.csv")
    rg = {(r.window, round(r.a, 2)): r.R_glob for r in rg.itertuples()}
    k0 = _kappa0()
    rows = []
    for (win, a), g in bg.groupby(["window", "a"]):
        rows.append({"window": win, "a": a, "n_crossed": len(g), "median_cross_R": g.R.median(),
                     "median_cross_w2": g.cross_w2.median(), "R_glob": rg[(win, a)], "Ghat": gh[(win, a)],
                     "kappa0": k0[win]})
    return pd.DataFrame(rows), bg[["window", "a", "seed", "cross_w2", "R"]]


# ------------------------------------------------------------------------------------------ validation
def validate(settings, runs):
    checks = []

    def chk(name, got, ref):
        err = abs(float(got) - float(ref))
        checks.append({"check": name, "reproduced": float(got), "committed": float(ref), "abs_diff": err,
                       "ok": err <= TOL})

    ex = pd.read_csv(RESULTS / "prospective_existing_settings.csv").set_index(["window", "a"])
    for r in settings.itertuples():
        chk(f"Block G median crossing R {r.window} {r.a}", r.median_cross_R, ex.loc[(r.window, r.a), "median_cross_R"])
        chk(f"Block G R_glob {r.window} {r.a}", r.R_glob, ex.loc[(r.window, r.a), "R_glob"])
    cal = pd.read_csv(RESULTS / "prospective_calibration.csv").set_index("a")
    for a in A_VALUES:
        b = settings[(settings.window == "base") & (settings.a == a)].iloc[0]
        chk(f"lambda({a})", b.median_cross_R / b.R_glob, cal.loc[a, "lambda_fitted"])
        chk(f"B1 base median |w2| ({a})", b.median_cross_w2, cal.loc[a, "B1_base_median_cross_w2"])
        chk(f"B2 pooled median R ({a})", runs.R.median(), cal.loc[a, "B2_pooled_median_cross_R"])
    return checks


# ------------------------------------------------------------------------------------------ Block 3
def block3(fit, checks):
    gh, k0 = _cert_ghat(), _kappa0()
    runs = pd.read_csv(RESULTS / "prospective_runs.csv")
    runs["R_cert"] = [w2 * gh[(w, round(a, 2))] / 2 if w2 == w2 else np.nan
                      for w, a, w2 in zip(runs.window, runs.a, runs.cross_w2)]
    obs = runs[runs.cross_step.notna()].groupby(["window", runs.a.round(2)]).agg(
        n_crossed=("R_cert", "size"), obs_median_cross_R=("R_cert", "median")).reset_index()
    pred = pd.read_csv(RESULTS / "prospective_predictions.csv")
    p1 = pred[pred.target == "median_cross_R"].pivot_table(index=["window", "a"], columns="model",
                                                          values="prediction").reset_index()
    m = obs.merge(p1, on=["window", "a"])
    m["a"] = m.a.round(2)
    m["kappa0"] = [k0[w] for w in m.window]
    m["Ghat"] = [gh[(w, a)] for w, a in zip(m.window, m.a)]
    for b in BASELINES:
        m[b] = [predict_baselines(fit, a, u, k, g)[b] for a, u, k, g in zip(m.a, m.U, m.kappa0, m.Ghat)]
    for x in ("U", "C", "B1", "B2") + BASELINES:
        m[f"abslogerr_{x}"] = abslogerr(m[x], m.obs_median_cross_R)
    # validation against the committed scoring
    sc = pd.read_csv(RESULTS / "prospective_scores.csv").assign(a=lambda d: d.a.round(2)).set_index(["window", "a"])
    for r in m.itertuples():
        for c in ("obs_median_cross_R", "U", "C", "B1", "B2", "abslogerr_U", "abslogerr_C", "abslogerr_B1",
                  "abslogerr_B2"):
            got, ref = getattr(r, c), sc.loc[(r.window, r.a), c]
            checks.append({"check": f"Block 3 {c} {r.window} {r.a}", "reproduced": got, "committed": ref,
                           "abs_diff": abs(got - ref), "ok": abs(got - ref) <= TOL})
    reg = setting_compare(m, ("B1", "B2", "U"))
    ref = pd.read_csv(RESULTS / "prospective_comparisons.csv").set_index("comparison")
    for r in reg.itertuples():
        for k in ("mean_diff", "ci95_lo", "ci95_hi"):
            got, rv = getattr(r, k), ref.loc[r.comparison, k]
            checks.append({"check": f"Block 3 {r.comparison} {k}", "reproduced": got, "committed": rv,
                           "abs_diff": abs(got - rv), "ok": abs(got - rv) <= TOL})
    return m, reg, setting_compare(m, BASELINES)


# ------------------------------------------------------------------------------------------ own-seed
def own_seed(fit, checks):
    k0 = _kappa0()
    pr = pd.read_csv(RESULTS / "prospective_own_predictions.csv", float_precision="round_trip")
    rn = pd.read_csv(RESULTS / "prospective_own_runs.csv", float_precision="round_trip")
    d = rn.merge(pr, on=["window", "a", "seed"])
    d["a"] = d.a.round(2)
    d["R_cross"] = d.cross_w2 * d.Ghat_lo / 2
    d = d[d.R_cross.notna()].copy()                     # primary analysis: crossers
    for b in BASELINES:
        d[b] = [predict_baselines(fit, a, u, k0[w], g)[b] for a, u, w, g in zip(d.a, d.U, d.window, d.Ghat_lo)]
    ref = pd.read_csv(RESULTS / "prospective_own_scores.csv")
    ref = ref[ref.analysis.str.startswith("primary")].assign(a=lambda x: x.a.round(2)).set_index(["a", "comparison"])
    rows, per_window = [], []
    for a in A_VALUES:
        g = d[d.a == a]
        per_run = pd.DataFrame({"window": g.window, "e_own": np.log(g.R_cross / g.U_own),
                                "e_U": np.log(g.R_cross / g.U), "e_C": np.log(g.R_cross / g.C),
                                "e_Cown": np.log(g.R_cross / g.C_own)})
        for k, v in comparisons(per_run, a).items():          # validation: the registered comparisons
            for s in ("stat", "lo", "hi"):
                got, rv = v[s], ref.loc[(a, k), s]
                checks.append({"check": f"own-seed {a} {k} {s}", "reproduced": got, "committed": rv,
                               "abs_diff": abs(got - rv), "ok": abs(got - rv) <= TOL})
        for b in BASELINES:
            per_run[f"e_{b}"] = np.log(g.R_cross / g[b])
        for x in BASELINES:
            rows.append({"test": "own-seed (per-run, window bootstrap)", "a": a, "comparison": f"C - {x}",
                         **window_compare(per_run, "C", x)})
        wm = per_run.assign(**{c: per_run[c].abs() for c in per_run.columns if c.startswith("e_")}).groupby("window").mean()
        for x in ("U", "C", "own", "Cown") + BASELINES:
            per_window.append({"a": a, "model": x, "mean_over_windows_of_window_mean_abslogerr": float(wm[f"e_{x}"].mean()),
                               "mean_over_runs_abslogerr": float(per_run[f"e_{x}"].abs().mean()), "n_runs": len(per_run),
                               "n_windows": int(wm.shape[0])})
    # setting-level (median crossing R per window) errors, validated against prospective_own_settings_scored.csv
    ss = pd.read_csv(RESULTS / "prospective_own_settings_scored.csv").assign(a=lambda x: x.a.round(2)).set_index(["window", "a"])
    cells = []
    for (w, a), g in d.groupby(["window", "a"]):
        obs = float(g.R_cross.median())
        row = {"window": w, "a": a, "n_crossed": len(g), "obs_median_R": obs, "U": g.U.iloc[0], "C": g.C.iloc[0]}
        for x in ("U", "C") + BASELINES:
            row[x] = float(g[x].iloc[0])
            row[f"abslogerr_{x}"] = abs(math.log(obs / row[x]))
        for x in ("U", "C"):
            got, rv = row[f"abslogerr_{x}"], ss.loc[(w, a), f"abslogerr_setting_{x}"]
            checks.append({"check": f"own-seed setting |log err| {x} {w} {a}", "reproduced": got, "committed": rv,
                           "abs_diff": abs(got - rv), "ok": abs(got - rv) <= TOL})
        cells.append(row)
    cells = pd.DataFrame(cells)
    for a in A_VALUES:
        ca = cells[cells.a == a]
        for x in BASELINES:
            d_, lo, hi = _boot_diff(ca.abslogerr_C.values, ca[f"abslogerr_{x}"].values)
            rows.append({"test": "own-seed (setting median R, setting bootstrap)", "a": a, "comparison": f"C - {x}",
                         "mean_diff": d_, "ci95_lo": lo, "ci95_hi": hi, "C_better_interval_excludes_zero": hi < 0,
                         "X_better_interval_excludes_zero": lo > 0})
    return pd.DataFrame(rows), pd.DataFrame(per_window), cells


# ------------------------------------------------------------------------------------------ main
def _fit_table(fit, settings):
    rows = []
    src = "Block G only: 10 settings (5 windows x a in {1.30, 1.50}), blockG_crossings.csv crossers"
    for a in A_VALUES:
        rows.append({"baseline": "PL", "a": a, "parameter": "lambda_pool", "value": fit["PL"][a],
                     "fitted_on": src + " + certified R_glob; pooled per-run R/R_glob at a",
                     "n": int((settings[settings.a == a].n_crossed).sum())})
        rows.append({"baseline": "PL5", "a": a, "parameter": "lambda_5", "value": fit["PL5"][a],
                     "fitted_on": src + " + certified R_glob; median of 5 per-window ratios at a", "n": 5})
        for i, p in enumerate(("intercept", "slope_log_kappa0")):
            rows.append({"baseline": "RKa", "a": a, "parameter": p, "value": fit["RKa"][a][i],
                         "fitted_on": src + " + K (limit_windows.csv); 5 settings at a", "n": 5})
        for i, p in enumerate(("intercept", "slope_log_Ghat")):
            rows.append({"baseline": "RG", "a": a, "parameter": p, "value": fit["RG"][a][i],
                         "fitted_on": src + " + certified Ghat; 5 settings at a", "n": 5})
    for i, p in enumerate(("intercept", "slope_log_kappa0", "shift_a1.50")):
        rows.append({"baseline": "RK", "a": np.nan, "parameter": p, "value": fit["RK"][i],
                     "fitted_on": src + " + K (limit_windows.csv); all 10 settings", "n": 10})
    return pd.DataFrame(rows)


def main():
    settings, runs = training_data()
    checks = validate(settings, runs)
    fit = fit_baselines(settings, runs)
    m, reg, new = block3(fit, checks)
    own_cmp, own_pw, own_cells = own_seed(fit, checks)
    ch = pd.DataFrame(checks)
    ch.to_csv(RESULTS / "baselines_posthoc_validation.csv", index=False)
    if not ch.ok.all():
        print(ch[~ch.ok].to_string(index=False))
        raise SystemExit("VALIDATION FAILED: committed numbers not reproduced; nothing new reported")
    print(f"validation: {len(ch)} checks, max |diff| = {ch.abs_diff.max():.3g}")
    # in-sample fit on the training settings (context only)
    ins = settings.copy()
    for b in BASELINES:
        ins[b] = [predict_baselines(fit, a, rg, k, g)[b] for a, rg, k, g in zip(ins.a, ins.R_glob, ins.kappa0, ins.Ghat)]
        ins[f"abslogerr_{b}"] = abslogerr(ins[b], ins.median_cross_R)
    ins.to_csv(RESULTS / "baselines_posthoc_training_fit.csv", index=False)
    _fit_table(fit, settings).to_csv(RESULTS / "baselines_posthoc_fits.csv", index=False)
    m.to_csv(RESULTS / "baselines_posthoc_block3.csv", index=False)
    comp = pd.concat([reg.assign(test="Block 3 (registered models, reproduced)", a=np.nan),
                      new.assign(test="Block 3 (post hoc baselines)", a=np.nan), own_cmp], ignore_index=True)
    comp = comp[["test", "a", "comparison", "mean_diff", "ci95_lo", "ci95_hi", "C_better_interval_excludes_zero",
                 "X_better_interval_excludes_zero"]]
    comp.to_csv(RESULTS / "baselines_posthoc_comparisons.csv", index=False)
    own_pw.to_csv(RESULTS / "baselines_posthoc_own_errors.csv", index=False)
    own_cells.to_csv(RESULTS / "baselines_posthoc_own_settings.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(_fit_table(fit, settings)[["baseline", "a", "parameter", "value"]].to_string(index=False))
    print(ins[["window", "a", "median_cross_R"] + [f"abslogerr_{b}" for b in BASELINES]].to_string(index=False))
    cols = ["window", "a", "obs_median_cross_R", "C", "B1", "B2", "U"] + list(BASELINES)
    print(m[cols].to_string(index=False))
    print(m[[c for c in m.columns if c.startswith("abslogerr_")]].mean().to_string())
    print(comp.to_string(index=False))
    print(own_pw.to_string(index=False))


if __name__ == "__main__":
    main()
