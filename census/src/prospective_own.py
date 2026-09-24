"""Prospective test of the own-seed predictor U_own (registration: results/prospective_own_prediction.md).

    python -m src.prospective_own select          # the eight windows (structural rule, no training)
    python -m src.prospective_own power           # power simulation from existing per-run errors
    python -m src.prospective_own predict [w]     # certified Ĝ and R_glob per setting, U, C, and every run's R_own
    python -m src.prospective_own validate [w]    # certified branch and bound at both bracket ends, 3 seeds/setting
    python -m src.prospective_own train [w]       # Block 3's protocol (blockG_windows.train, 12,000 steps)
    python -m src.prospective_own score
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.30, 1.50)
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SEED0, N_SEEDS = 400_000, 60
BUDGET = 12_000
LAMBDA = {1.30: 1.11487, 1.50: 1.16440}          # Block 3, frozen
RHO_RES = {1.30: 1.0311, 1.50: 1.0660}           # frozen from base-window data (crossings <= 12,000)
RANGE_P4 = {1.30: (0.001, 0.061), 1.50: (0.034, 0.094)}
NI_MARGIN = 0.01
B_WIN = 10_000
WORKERS = int(os.environ.get("PO_WORKERS", "3"))


# ------------------------------------------------------------------------------------------ selection
def _trained_symmetric():
    from .blockG_windows import WINDOWS
    held = pd.read_csv(RESULTS / "prospective_windows.csv")
    tr = {(round(-w.i_lo, 4), round(w.o_lo, 4), round(w.o_hi, 4)) for w in WINDOWS if abs(w.i_lo + w.i_hi) < 1e-12}
    return tr | set(zip(held.i.round(4), held.o1.round(4), held.o2.round(4)))


def select():
    c = pd.read_csv(RESULTS / "prospective_candidates.csv")
    c = c[c.K_lo > 0].reset_index(drop=True)
    tr = _trained_symmetric()
    c = c[[k not in tr for k in zip(c.i.round(4), c.o1.round(4), c.o2.round(4))]].reset_index(drop=True)
    picked = []
    for q in QUANTILES:
        t = c.kappa0_lo.quantile(q)
        picked.append(next(k for k in (c.kappa0_lo - t).abs().sort_values().index if k not in picked))
    sel = c.loc[picked].copy()
    sel.insert(0, "quantile", QUANTILES)
    sel.insert(0, "name", [f"V{int(round(q * 100)):02d}" for q in QUANTILES])
    assert not any(k in tr for k in zip(sel.i.round(4), sel.o1.round(4), sel.o2.round(4)))
    sel.to_csv(RESULTS / "prospective_own_windows.csv", index=False)
    print(sel.to_string(index=False))
    return sel


def windows():
    from .blockG_windows import Window
    return [Window(r.name, -r.i, r.i, r.o1, r.o2) for r in pd.read_csv(RESULTS / "prospective_own_windows.csv").itertuples()]


# ------------------------------------------------------------------------------------------ statistics
def _win_boot(vals, B, seed=0):
    """Window-level bootstrap of the mean of per-window values (resample windows with replacement)."""
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    m = vals[rng.integers(0, len(vals), (B, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def comparisons(per_run, a, B=B_WIN):
    """per_run: DataFrame with window, e_own, e_U, e_C, e_Cown (signed log(R_cross/pred)) for one a."""
    g = per_run.assign(**{k: per_run[k].abs() for k in ("e_own", "e_U", "e_C", "e_Cown")}).groupby("window")
    mean, med = g.mean(numeric_only=True), g.median(numeric_only=True)
    out = {}
    d, lo, hi = _win_boot(mean.e_own - mean.e_U, B)
    out["P1"] = {"stat": d, "lo": lo, "hi": hi, "pass": hi < 0}
    if round(a, 2) == 1.30:
        d, lo, hi = _win_boot(med.e_own - med.e_C, B)
        out["P2a"] = {"stat": d, "lo": lo, "hi": hi, "pass": hi < NI_MARGIN}
    else:
        d, lo, hi = _win_boot(mean.e_own - mean.e_C, B)
        out["P2b"] = {"stat": d, "lo": lo, "hi": hi, "pass": hi >= 0, "C_better_interval_above_0": lo > 0}
    d, lo, hi = _win_boot(mean.e_Cown - mean.e_C, B)
    out["P3"] = {"stat": d, "lo": lo, "hi": hi, "pass": hi < 0}
    m4 = float(per_run.e_own.abs().median())
    out["P4"] = {"stat": m4, "lo": RANGE_P4[round(a, 2)][0], "hi": RANGE_P4[round(a, 2)][1],
                 "pass": RANGE_P4[round(a, 2)][0] <= m4 <= RANGE_P4[round(a, 2)][1]}
    return out


# ------------------------------------------------------------------------------------------ power
def power(n_sim=1000, seed=1):
    rows = []
    for label, mult, indep in (("as observed", 1.0, False), ("stress: window sd x3, independent shifts", 3.0, True)):
        rows += _power(n_sim, seed, mult, indep, label)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "prospective_own_power.csv", index=False)
    print(d.to_string(index=False))


def _power(n_sim, seed, mult, indep, label):
    """Simulated experiments (8 windows x 60 runs per a) from existing data:
    per-run signed errors (e_own, e_U) resampled jointly from the base window's phase 2b runs crossing within
    12,000 steps (own thresholds from own_threshold_crossing.csv); a window effect common to all predictors,
    drawn N(0, sd_a) with sd_a the spread of Block 3's per-setting log(obs/U) at that a; e_C = e_U - log λ(a),
    e_Cown = e_own - log ρ_res(a).  Each comparison is scored exactly as registered (window bootstrap reduced to
    2,000 resamples for speed)."""
    cr = pd.read_csv(RESULTS / "own_threshold_crossing.csv")
    runs = pd.read_csv(RESULTS / "wi_crossing_runs.csv")
    b3 = pd.read_csv(RESULTS / "prospective_scores.csv")
    rng = np.random.default_rng(seed)
    rows = []
    for a in A_VALUES:
        o = cr[cr.a.round(2) == a]
        Rp = float(o.w2_pop.iloc[0] * o.Ghat_cert.iloc[0] / 2)
        m = runs[(runs.a.round(2) == a) & (runs.budget == 32_000) & (runs.cross_step <= BUDGET)].merge(
            o[["seed", "R_own"]], on="seed")
        pairs = np.c_[np.log(m.R_cert / m.R_own), np.log(m.R_cert / Rp)]
        sd = mult * float(np.std(-b3[b3.a.round(2) == a].U_logerr, ddof=1))
        hits = {}
        for _ in range(n_sim):
            parts = []
            for w in range(len(QUANTILES)):
                idx = rng.integers(0, len(pairs), N_SEEDS)
                dlt = rng.normal(0, sd)
                dlt2 = rng.normal(0, sd) if indep else dlt      # stress: the two predictors shifted independently
                e_own, e_U = pairs[idx, 0] + dlt, pairs[idx, 1] + dlt2
                parts.append(pd.DataFrame({"window": w, "e_own": e_own, "e_U": e_U,
                                           "e_C": e_U - math.log(LAMBDA[a]), "e_Cown": e_own - math.log(RHO_RES[a])}))
            res = comparisons(pd.concat(parts), a, B=2_000)
            for k, v in res.items():
                hits.setdefault(k, []).append(v["pass"])
        for k, v in hits.items():
            rows.append({"scenario": label, "a": a, "comparison": k, "power_or_prob_pass": float(np.mean(v)),
                         "n_sim": n_sim, "window_sd": sd, "n_base_runs": len(pairs)})
    return rows


# ------------------------------------------------------------------------------------------ predictions
def _win_tuple(w):
    return (w.i_lo, w.i_hi, w.o_lo, w.o_hi)


def _setting_job(args):
    """Certified Ĝ and population R_glob of a window at a (Block 3's procedures)."""
    from .prospective import _rglob_job, ghat_window
    w, a = args
    g = ghat_window((w, a))
    r = _rglob_job((w, a))
    return {"window": w.tag, "a": a, "Ghat_lo": g["Ghat_lo"], "Ghat_hi": g["Ghat_hi"],
            "w2_glob_lo": r["w2_glob_lo"], "w2_glob_hi": r["w2_glob_hi"]}


def _own_job(args):
    from .own_threshold import BRACKET_STEP, WIDTH, global_min
    w, a, seed, w2_start = args
    x, y = (t.numpy() for t in w.data(200, seed))

    def placed(s):
        return global_min(s, a, x, y, win=_win_tuple(w))[3] > 0
    lo = hi = round(w2_start, 6)
    p = placed(lo)
    if p:
        while p and lo > 0.3:
            hi = lo; lo = round(lo - BRACKET_STEP, 6); p = placed(lo)
    else:
        while not p and hi < 30:
            lo = hi; hi = round(hi + BRACKET_STEP, 6); p = placed(hi)
    while hi - lo > WIDTH + 1e-12:
        mid = round(0.5 * (lo + hi), 6)
        if placed(mid):
            hi = mid
        else:
            lo = mid
    return {"window": w.tag, "a": a, "seed": seed, "w2_own_lo": lo, "w2_own_hi": hi, "w2_own": 0.5 * (lo + hi)}


def predict(workers=WORKERS):
    wins = windows()
    with Pool(workers) as p:
        st = pd.DataFrame(p.map(_setting_job, [(w, a) for w in wins for a in A_VALUES], chunksize=1))
    st["w2_pop"] = 0.5 * (st.w2_glob_lo + st.w2_glob_hi)
    st["U"] = st.w2_pop * st.Ghat_lo / 2
    st["C"] = st.U * st.a.round(2).map(LAMBDA)
    st.to_csv(RESULTS / "prospective_own_settings.csv", index=False)
    start = {(r.window, round(r.a, 2)): r.w2_pop for r in st.itertuples()}
    jobs = [(w, a, SEED0 + i, start[(w.tag, a)]) for w in wins for a in A_VALUES for i in range(N_SEEDS)]
    with Pool(workers) as p:
        own = pd.DataFrame(p.map(_own_job, jobs, chunksize=1))
    own = own.merge(st[["window", "a", "Ghat_lo", "U", "C"]], on=["window", "a"])
    own["U_own"] = own.w2_own * own.Ghat_lo / 2
    own["C_own"] = own.U_own * own.a.round(2).map(RHO_RES)
    f = RESULTS / "prospective_own_predictions.csv"
    own.to_csv(f, index=False)
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    (RESULTS / "prospective_own_predictions.sha256").write_text(h + "\n")
    print("SHA-256", h, "(commit the file and its hash before training)")


def gate_hash(results=None):
    """Registered: the predictions file must be unchanged from its recorded (committed) SHA-256."""
    R = results or RESULTS
    rec = (R / "prospective_own_predictions.sha256").read_text().strip()
    now = hashlib.sha256((R / "prospective_own_predictions.csv").read_bytes()).hexdigest()
    if rec != now:
        raise SystemExit("STOP (registered): the predictions file changed after its hash was recorded")
    return now


def gate_validation(results=None):
    """Registered stop: any certified status contradicting a bracket end."""
    v = pd.read_csv((results or RESULTS) / "prospective_own_validation.csv")
    bad = v[(v.cert_lo == "plus") | (v.cert_hi == "minus")]
    if len(bad):
        raise SystemExit(f"STOP (registered): {len(bad)} certified statuses contradict a bracket end")
    return len(v)


def _val_job(args):
    from .profiled_bnb import certify
    w, a, seed, lo, hi = args
    x, y = (t.numpy() for t in w.data(200, seed))
    out = {"window": w.tag, "a": a, "seed": seed}
    for end, s in (("lo", lo), ("hi", hi)):
        st = "unresolved"
        for tol in (1e-7, 1e-9):
            rm = certify(s, a, x, y, "-", tol=tol, win=_win_tuple(w))
            rp = certify(s, a, x, y, "+", tol=tol, win=_win_tuple(w))
            st = "minus" if rp["lower"] > rm["upper"] else "plus" if rm["lower"] > rp["upper"] else "unresolved"
            if st != "unresolved":
                break
        out[f"cert_{end}"] = st
    out["contradiction"] = out["cert_lo"] == "plus" or out["cert_hi"] == "minus"
    return out


def validate(workers=WORKERS):
    gate_hash()
    wins = {w.tag: w for w in windows()}
    own = pd.read_csv(RESULTS / "prospective_own_predictions.csv", float_precision="round_trip")
    sub = own[own.seed.isin([SEED0, SEED0 + 1, SEED0 + 2])]
    with Pool(workers) as p:
        v = pd.DataFrame(p.map(_val_job, [(wins[r.window], r.a, r.seed, r.w2_own_lo, r.w2_own_hi)
                                          for r in sub.itertuples()], chunksize=1))
    v.to_csv(RESULTS / "prospective_own_validation.csv", index=False)
    print(v.to_string(index=False)); print("contradictions:", int(v.contradiction.sum()))
    gate_validation()


# ------------------------------------------------------------------------------------------ training and scoring
def _train_job(args):
    import torch
    from .blockG_windows import train
    torch.set_num_threads(1)
    w, a, seed = args
    r = train(w, a, seed, gs=1.0, budget=BUDGET)
    return {k: r[k] for k in ("window", "a", "seed", "cross_step", "cross_w2", "final_placed", "final_solved")} | \
        {"final_w2": 2 * r["R_final"]}


def train(workers=WORKERS):
    gate_hash()
    gate_validation()
    jobs = [(w, a, SEED0 + i) for w in windows() for a in A_VALUES for i in range(N_SEEDS)]
    with Pool(workers) as p:
        pd.DataFrame(p.map(_train_job, jobs, chunksize=1)).to_csv(RESULTS / "prospective_own_runs.csv", index=False)


def score():
    gate_hash()
    gate_validation()
    pr = pd.read_csv(RESULTS / "prospective_own_predictions.csv", float_precision="round_trip")
    rn = pd.read_csv(RESULTS / "prospective_own_runs.csv", float_precision="round_trip")
    d = rn.merge(pr, on=["window", "a", "seed"])
    d["R_cross"] = d.cross_w2 * d.Ghat_lo / 2
    d["R_final"] = d.final_w2 * d.Ghat_lo / 2
    rows, cells = [], []
    for analysis in ("primary (crossers)", "sensitivity (non-crossers at their final R)"):
        dd = d.copy()
        if analysis.startswith("sensitivity"):
            dd["R_cross"] = dd.R_cross.fillna(dd.R_final)     # crossing R >= final R; the bound is used as the value
        dd = dd[dd.R_cross.notna()]
        for a in A_VALUES:
            g = dd[dd.a.round(2) == a]
            per_run = pd.DataFrame({"window": g.window, "e_own": np.log(g.R_cross / g.U_own),
                                    "e_U": np.log(g.R_cross / g.U), "e_C": np.log(g.R_cross / g.C),
                                    "e_Cown": np.log(g.R_cross / g.C_own)})
            for k, v in comparisons(per_run, a).items():
                rows.append({"analysis": analysis, "a": a, "comparison": k, **v})
    for (w, a), g in d.groupby(["window", d.a.round(2)]):
        x = g[g.R_cross.notna()]
        obs = float(x.R_cross.median())
        cells.append({"window": w, "a": a, "n": len(g), "n_crossed": len(x), "non_crossers": len(g) - len(x),
                      "frac_crossed": len(x) / len(g), "obs_median_R": obs,
                      "abslogerr_setting_U_own": abs(math.log(obs / g.U_own.median())),
                      "abslogerr_setting_C_own": abs(math.log(obs / g.C_own.median())),
                      "abslogerr_setting_U": abs(math.log(obs / g.U.iloc[0])),
                      "abslogerr_setting_C": abs(math.log(obs / g.C.iloc[0])),
                      "spearman_R_cross_vs_R_own": float(x.R_cross.rank().corr(x.U_own.rank()))})
    pd.DataFrame(rows).to_csv(RESULTS / "prospective_own_scores.csv", index=False)
    pd.DataFrame(cells).to_csv(RESULTS / "prospective_own_settings_scored.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(cells).to_string(index=False))



# ------------------------------------------------------------------ early-branch rule (for the amendment; not yet registered)
PLATEAU = 0.05


def train_logged(win, a, seed, budget=BUDGET):
    """blockG_windows.train, unchanged in every operation, plus a read-only log of (step, w1, w2) at each 50-step check
    until the crossing.  Returns (cross_step, cross_w2, final_w2, log)."""
    import torch
    from torch.nn import functional as F
    from .blockG_windows import CHECK_EVERY, LR, f_a, oriented_gap
    f = f_a(a)
    x, y = win.data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    cross_step = cross_w2 = None
    log = []
    for i in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
        if i % CHECK_EVERY == 0 and cross_step is None:
            with torch.no_grad():
                w1, b1, w2, b2 = (float(v) for v in th)
            log.append((i, w1, w2))
            if oriented_gap(win, a, w1, b1) > 0:
                cross_step, cross_w2 = i, abs(w2)
    with torch.no_grad():
        final_w2 = abs(float(th[2]))
    return cross_step, cross_w2, final_w2, log


def early_branch(log, plateau="w1"):
    """The rule: the first logged check at which the run has left the constant-predictor plateau (having been on it);
    a run never on the plateau takes its first logged check.  plateau="w1": |w1| < 0.05; "w1w2": |w1| < 0.05 or
    |w2| < 0.05.  Returns (index, step, branch in canonical orientation, |w2|) or None if the run is still on the
    plateau at every check."""
    on = [(abs(w1) < PLATEAU) or (plateau == "w1w2" and abs(w2) < PLATEAU) for _, w1, w2 in log]
    if not any(on):
        i = 0
    else:
        i0 = on.index(True)
        rest = [k for k in range(i0, len(on)) if not on[k]]
        if not rest:
            return None
        i = rest[0]
    step, w1, w2 = log[i]
    return i, step, int(np.sign(w1) * np.sign(w2)), abs(w2)


def _calib_job(args):
    from .blockG_windows import WINDOWS
    a, seed = args
    base = next(w for w in WINDOWS if w.tag == "base")
    cs, cw2, fw2, log = train_logged(base, a, seed)
    out = {"a": a, "seed": seed, "cross_step": cs, "cross_w2": cw2, "final_w2": fw2, "n_checks": len(log)}
    if cs is not None:
        out["branch_cross"] = int(np.sign(log[-1][1]) * np.sign(log[-1][2]))
    for pl in ("w1", "w1w2"):
        e = early_branch(log, pl)
        out.update({f"early_{pl}_defined": e is not None, f"early_{pl}_step": e[1] if e else np.nan,
                    f"early_{pl}_branch": e[2] if e else np.nan, f"early_{pl}_w2": e[3] if e else np.nan})
    out["ever_on_plateau_w1"] = any(abs(w1) < PLATEAU for _, w1, _ in log)
    out["ever_on_plateau_w1w2"] = any(abs(w1) < PLATEAU or abs(w2) < PLATEAU for _, w1, w2 in log)
    return out


def calibrate_early(workers=3):
    """On existing settings only (Block G base-window runs, seeds 0-39, a = 1.30 and 1.50): (i) exact reproduction of
    blockG_crossings.csv by train_logged; (ii) the early-branch rule's match rate, lead time and per-run errors under
    both plateau definitions.  Own branch thresholds: mirror_branch_thresholds.csv (group 'cross', the same training
    sets up to float32 rounding of fold1d.make_data)."""
    with Pool(workers) as p:
        d = pd.DataFrame(p.map(_calib_job, [(a, s) for a in A_VALUES for s in range(40)], chunksize=1))
    ref = pd.read_csv(RESULTS / "blockG_crossings.csv", float_precision="round_trip")
    ref = ref[ref.window == "base"][["a", "seed", "cross_step", "cross_w2"]]
    m = d.merge(ref, on=["a", "seed"], suffixes=("", "_ref"))
    d["reproduced"] = bool(np.array_equal(m.cross_step.values, m.cross_step_ref.values, equal_nan=True) and
                           np.array_equal(m.cross_w2.values, m.cross_w2_ref.values, equal_nan=True))
    T = pd.read_csv(RESULTS / "mirror_branch_thresholds.csv", float_precision="round_trip")
    T = T[T.group == "cross"].set_index(["a", "seed"])
    from .own_threshold import _pop
    rows = []
    for pl in ("w1", "w1w2"):
        for a in A_VALUES:
            G = _pop(a)[0]
            r_fit = RESID[a]
            g = d[(d.a.round(2) == a) & d.cross_step.notna()].copy()
            g["T_early"] = [T.loc[(a, s), "T_plus"] if b > 0 else T.loc[(a, s), "T_minus"] if b < 0 else np.nan
                            for s, b in zip(g.seed, g[f"early_{pl}_branch"].fillna(0))]
            g["R_cross"] = g.cross_w2 * G / 2
            e_un = np.log(g.R_cross / (g.T_early * G / 2))
            e_fit = e_un - np.log(1 + r_fit)
            rows.append({"plateau": pl, "a": a, "n_cross": len(g), "defined": int(g[f"early_{pl}_defined"].sum()),
                         "ever_on_plateau": float(g[f"ever_on_plateau_{pl}"].mean()),
                         "match_rate": float((g[f"early_{pl}_branch"] == g.branch_cross).mean()),
                         "lead_steps_median": float((g.cross_step - g[f"early_{pl}_step"]).median()),
                         "lead_steps_min": float((g.cross_step - g[f"early_{pl}_step"]).min()),
                         "lead_R_median": float(((g.cross_w2 - g[f"early_{pl}_w2"]) * G / 2).median()),
                         "abs_err_unfitted_median": float(np.nanmedian(np.abs(e_un))),
                         "abs_err_unfitted_p10": float(np.nanpercentile(np.abs(e_un), 10)),
                         "abs_err_unfitted_p90": float(np.nanpercentile(np.abs(e_un), 90)),
                         "abs_err_fitted_median": float(np.nanmedian(np.abs(e_fit))),
                         "abs_err_fitted_p10": float(np.nanpercentile(np.abs(e_fit), 10)),
                         "abs_err_fitted_p90": float(np.nanpercentile(np.abs(e_fit), 90))})
    d.to_csv(RESULTS / "prospective_own_early_calibration_runs.csv", index=False)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "prospective_own_early_calibration.csv", index=False)
    print("reproduced blockG_crossings exactly:", bool(d.reproduced.iloc[0]), "on", len(m), "runs")
    print(t.to_string(index=False))


RESID = {1.30: 0.030, 1.50: 0.063}               # r(a), frozen from base-window crossing runs (S3 crossing-branch residual)


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS
    {"select": select, "power": power, "predict": lambda: predict(w), "validate": lambda: validate(w),
     "train": lambda: train(w), "score": score, "calibrate_early": lambda: calibrate_early(w)}[cmd]()
