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
    # amendment 2: the global minimiser's branch at the placed end, and the other branch's own threshold
    from .mirror_branches import branch_threshold
    gbr = int(np.sign(global_min(hi, a, x, y, win=_win_tuple(w))[1]))
    w2_other = branch_threshold(a, seed, -gbr, 0.5 * (lo + hi), data=(x, y), win=_win_tuple(w))
    return {"window": w.tag, "a": a, "seed": seed, "w2_own_lo": lo, "w2_own_hi": hi, "w2_own": 0.5 * (lo + hi),
            "global_branch": gbr, "w2_other": w2_other}


PO_PARTS = RESULTS / "prospective_own_parts"


def _append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def predict(workers=WORKERS):
    """Every setting and every run's prediction is appended to prospective_own_parts/ as it finishes, and a restart
    skips finished ones (crash resilience only).  The predictions file is assembled, sorted, once every run is done,
    and only then hashed."""
    PO_PARTS.mkdir(exist_ok=True)
    wins = windows()
    sf = PO_PARTS / "settings.csv"
    done = set() if not sf.exists() else set(zip(*[pd.read_csv(sf)[c] for c in ("window", "a")]))
    todo = [(w, a) for w in wins for a in A_VALUES if (w.tag, a) not in {(d0, round(d1, 2)) for d0, d1 in done}]
    with Pool(workers) as p:
        for r in p.imap_unordered(_setting_job, todo, chunksize=1):
            _append(sf, r)
    st = pd.read_csv(sf, float_precision="round_trip").sort_values(["window", "a"]).reset_index(drop=True)
    st["w2_pop"] = 0.5 * (st.w2_glob_lo + st.w2_glob_hi)
    st["U"] = st.w2_pop * st.Ghat_lo / 2
    st["C"] = st.U * st.a.round(2).map(LAMBDA)
    st.to_csv(RESULTS / "prospective_own_settings.csv", index=False)
    start = {(r.window, round(r.a, 2)): r.w2_pop for r in st.itertuples()}
    of = PO_PARTS / "own.csv"
    done = set() if not of.exists() else {(w_, round(a_, 2), int(s_)) for w_, a_, s_ in
                                          zip(*[pd.read_csv(of)[c] for c in ("window", "a", "seed")])}
    jobs = [(w, a, SEED0 + i, start[(w.tag, a)]) for w in wins for a in A_VALUES for i in range(N_SEEDS)
            if (w.tag, a, SEED0 + i) not in done]
    with Pool(workers) as p:
        for r in p.imap_unordered(_own_job, jobs, chunksize=1):
            _append(of, r)
    own = pd.read_csv(of, float_precision="round_trip").sort_values(["window", "a", "seed"]).reset_index(drop=True)
    assert len(own) == len(wins) * len(A_VALUES) * N_SEEDS and not own.duplicated(["window", "a", "seed"]).any()
    own = own.merge(st[["window", "a", "Ghat_lo", "U", "C"]], on=["window", "a"])
    own["U_own"] = own.w2_own * own.Ghat_lo / 2
    own["C_own"] = own.U_own * own.a.round(2).map(RHO_RES)
    own["R_other"] = own.w2_other * own.Ghat_lo / 2
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
    torch.set_num_threads(1)
    w, a, seed = args
    cs, cw2, fw2, log = train_logged(w, a, seed, BUDGET)
    out = {"window": w.tag, "a": a, "seed": seed, "cross_step": cs, "cross_w2": cw2, "final_w2": fw2}
    e = early_branch(log, "w1w2")
    out.update({"early_defined": e is not None, "early_step": e[1] if e else np.nan,
                "early_branch": e[2] if e else np.nan, "early_w2": e[3] if e else np.nan,
                "branch_cross": int(np.sign(log[-1][1]) * np.sign(log[-1][2])) if cs is not None else np.nan})
    return out


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
        loss = F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y)
        loss.backward()
        opt.step()
        if i % CHECK_EVERY == 0 and cross_step is None:
            with torch.no_grad():
                w1, b1, w2, b2 = (float(v) for v in th)
            log.append((i, w1, w2, float(loss.detach())))
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
    on = [(abs(e[1]) < PLATEAU) or (plateau == "w1w2" and abs(e[2]) < PLATEAU) for e in log]
    if not any(on):
        i = 0
    else:
        i0 = on.index(True)
        rest = [k for k in range(i0, len(on)) if not on[k]]
        if not rest:
            return None
        i = rest[0]
    step, w1, w2 = log[i][:3]
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
    out["ever_on_plateau_w1"] = any(abs(e[1]) < PLATEAU for e in log)
    out["ever_on_plateau_w1w2"] = any(abs(e[1]) < PLATEAU or abs(e[2]) < PLATEAU for e in log)
    out["log"] = log
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
    logs = [(r.a, r.seed, e) for r in d.itertuples() for e in r.log]
    pd.DataFrame([{"a": a, "seed": sd, "step": e[0], "w1": e[1], "w2": e[2], "loss": e[3]} for a, sd, e in logs]).to_csv(
        RESULTS / "prospective_own_early_calibration_logs.csv", index=False)
    d = d.drop(columns="log")
    d.to_csv(RESULTS / "prospective_own_early_calibration_runs.csv", index=False)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "prospective_own_early_calibration.csv", index=False)
    print("reproduced blockG_crossings exactly:", bool(d.reproduced.iloc[0]), "on", len(m), "runs")
    print(t.to_string(index=False))


RESID = {1.30: 0.030, 1.50: 0.063}               # r(a), frozen from base-window crossing runs (S3 crossing-branch residual)



# ------------------------------------------------------------------ amendment 2: secondary predictors
W_GLOBAL = 0.8919                                      # global-branch crossing share, Block G base-window runs (both a)
EXPECT_MATCH = {1.30: 0.865, 1.50: 0.892}              # early branch vs crossing branch, base-window runs
MATCH_TOL = 0.10
EXPECT_ERR = {("unfitted", 1.30): (0.029, 0.053), ("unfitted", 1.50): (0.065, 0.124),
              ("fitted", 1.30): (0.004, 0.023), ("fitted", 1.50): (0.005, 0.063)}   # 10th-90th pct, base window


def weighted_median(vals, wts):
    vals, wts = np.asarray(vals, float), np.asarray(wts, float)
    o = np.argsort(vals)
    c = np.cumsum(wts[o]) / wts.sum()
    return float(vals[o][np.searchsorted(c, 0.5)])


def early_prediction(row):
    """Early-branch predictor: the threshold of the branch recorded at the early checkpoint (global if it is the global
    minimiser's branch, else the other); undefined -> None (a miss)."""
    if not row["early_defined"]:
        return None
    return row["U_own"] if row["early_branch"] == row["global_branch"] else row["R_other"]


def score_secondary(results=None):
    R = results or RESULTS
    gate_hash(R); gate_validation(R)
    pr = pd.read_csv(R / "prospective_own_predictions.csv", float_precision="round_trip")
    rn = pd.read_csv(R / "prospective_own_runs.csv", float_precision="round_trip")
    st = pd.read_csv(R / "prospective_own_settings.csv", float_precision="round_trip")
    d = rn.merge(pr, on=["window", "a", "seed"])
    d["R_cross"] = d.cross_w2 * d.Ghat_lo / 2
    d["early_R"] = d.early_w2 * d.Ghat_lo / 2
    rows, sett = [], []
    for a in A_VALUES:
        g = d[d.a.round(2) == a]
        x = g[g.R_cross.notna()].copy()
        match = (x.early_branch == x.branch_cross) & x.early_defined
        pred = np.array([early_prediction(r) if early_prediction(r) is not None else np.nan for r in x.to_dict("records")])
        e_un = np.abs(np.log(x.R_cross.values / pred))
        e_fit = np.abs(np.log(x.R_cross.values / (pred * (1 + RESID[a]))))
        lead = x.cross_step - x.early_step
        rows.append({"a": a, "n_crossed": len(x), "n_undefined": int((~x.early_defined).sum()),
                     "match_rate": float(match.mean()), "match_expected": EXPECT_MATCH[a],
                     "match_consistent": abs(float(match.mean()) - EXPECT_MATCH[a]) <= MATCH_TOL,
                     "lead_steps_median": float(lead.median()), "lead_steps_min": float(lead.min()),
                     "lead_R_median": float((x.R_cross - x.early_R).median()), "lead_R_min": float((x.R_cross - x.early_R).min()),
                     **{f"err_{k}_median": float(np.nanmedian(v)) for k, v in (("unfitted", e_un), ("fitted", e_fit))},
                     **{f"err_{k}_in_expected_range": EXPECT_ERR[(k, a)][0] <= float(np.nanmedian(v)) <= EXPECT_ERR[(k, a)][1]
                        for k, v in (("unfitted", e_un), ("fitted", e_fit))}})
        for wname, gw in g.groupby("window"):
            xw = gw[gw.R_cross.notna()]
            obs = float(xw.R_cross.median())
            mix = weighted_median(np.r_[gw.U_own.values, gw.R_other.values],
                                  np.r_[np.full(len(gw), W_GLOBAL), np.full(len(gw), 1 - W_GLOBAL)])
            s0 = st[(st.window == wname) & (st.a.round(2) == a)].iloc[0]
            sett.append({"a": a, "window": wname, "obs_median_R": obs,
                         "err_mix": abs(np.log(obs / mix)), "err_mix_fitted": abs(np.log(obs / (mix * (1 + RESID[a])))),
                         "err_U": abs(np.log(obs / s0.U)), "err_C": abs(np.log(obs / s0.C)),
                         "err_U_own_setting": abs(np.log(obs / gw.U_own.median()))})
    t = pd.DataFrame(rows)
    se = pd.DataFrame(sett)
    comp = []
    for a in A_VALUES:
        s_a = se[se.a.round(2) == a]
        for m in ("err_mix", "err_mix_fitted"):
            m0, lo, hi = _win_boot(s_a[m].values, B_WIN)
            comp.append({"a": a, "statistic": f"mean per-setting {m}", "value": m0, "lo": lo, "hi": hi})
            for other in ("err_U", "err_C", "err_U_own_setting"):
                dd, lo, hi = _win_boot((s_a[m] - s_a[other]).values, B_WIN)
                comp.append({"a": a, "statistic": f"{m} - {other}", "value": dd, "lo": lo, "hi": hi})
    t.to_csv(R / "prospective_own_early_scores.csv", index=False)
    se.to_csv(R / "prospective_own_mixture_settings.csv", index=False)
    pd.DataFrame(comp).to_csv(R / "prospective_own_mixture_scores.csv", index=False)
    print(t.T.to_string()); print(pd.DataFrame(comp).to_string(index=False))


def descriptive():
    """EXPLORATORY, base-window runs (prospective protocol): (i) when the branch commits -- match rate between the branch
    at each check and at the crossing, by time and by R; (ii) branch switches between initialisation and crossing and
    the training loss at each switch, and each run's loss at its last switch."""
    from .own_threshold import _pop
    lg = pd.read_csv(RESULTS / "prospective_own_early_calibration_logs.csv", float_precision="round_trip")
    runs = pd.read_csv(RESULTS / "prospective_own_early_calibration_runs.csv")
    runs = runs[runs.cross_step.notna()]
    lg = lg.merge(runs[["a", "seed", "cross_step", "branch_cross"]], on=["a", "seed"])
    lg["branch"] = np.sign(lg.w1) * np.sign(lg.w2)
    lg["G"] = lg.a.round(2).map({a: _pop(a)[0] for a in A_VALUES})
    lg["R"] = lg.w2.abs() * lg.G / 2
    lg["frac_time"] = lg.step / lg.cross_step
    lg["match"] = lg.branch == lg.branch_cross
    by_t = lg.groupby([lg.a.round(2), pd.cut(lg.frac_time, [0, .01, .05, .1, .25, .5, .75, .9, 1.0001], include_lowest=True)],
                      observed=True).match.agg(["mean", "size"]).reset_index()
    by_R = lg.groupby([lg.a.round(2), pd.cut(lg.R, [0, .02, .05, .1, .15, .2, .25, 1])], observed=True).match.agg(["mean", "size"]).reset_index()
    sw = []
    for (a, sd), g in lg.sort_values("step").groupby([lg.a.round(2), "seed"]):
        b = g.branch.values
        idx = np.where(np.diff(b) != 0)[0] + 1
        for k in idx:
            sw.append({"a": a, "seed": sd, "step": int(g.step.values[k]), "loss": float(g.loss.values[k]),
                       "loss_before": float(g.loss.values[k - 1]), "last": k == idx[-1]})
    sw = pd.DataFrame(sw)
    by_t.to_csv(RESULTS / "prospective_own_commit_by_time.csv", index=False)
    by_R.to_csv(RESULTS / "prospective_own_commit_by_R.csv", index=False)
    sw.to_csv(RESULTS / "prospective_own_switch_losses.csv", index=False)
    print(by_t.to_string(index=False)); print(by_R.to_string(index=False))
    if len(sw):
        print("switches:", len(sw), " loss at switch: median %.4f, min %.4f, max %.4f;  log 2 = %.4f" %
              (sw.loss.median(), sw.loss.min(), sw.loss.max(), np.log(2)))
        print("loss at each run's last switch:", sw[sw["last"]].loss.describe().to_string())

if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else WORKERS
    {"select": select, "power": power, "predict": lambda: predict(w), "validate": lambda: validate(w),
     "train": lambda: train(w), "score": score, "calibrate_early": lambda: calibrate_early(w),
     "score_secondary": score_secondary, "descriptive": descriptive}[cmd]()
