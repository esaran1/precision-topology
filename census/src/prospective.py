"""Block 3: prospective predictions on held-out window geometries (registered before training).

Pieces (all computed independently of unconstrained training in the new settings):
  ghat_window     certified Ĝ(window, a): branch and bound on the exact one-sided gap over
                  w1 ∈ [-W', W'], b1 ∈ [0, 2π), W' = 2a / min(i_hi + o_hi, o_hi - i_lo) (beyond it both
                  orientations are negative, as for the base window, T76); the symmetry
                  (w1, b1, w2) -> (-w1, -b1, -w2) makes the one-sided sup equal Ĝ.
  rglob_window    certified R_glob(window, a): Block 1c's profiled branch and bound on the window's
                  800-point quadrature population, bisected to 0.005 in |w2|.
  calibration     from frozen Block G base-window runs (same protocol as the new runs: Adam lr 1e-2,
                  200 points per class from Window.data, 12,000 steps, crossing checked every 50):
                  λ(a) = median crossing R / certified R_glob(base, a)     [FITTED];
                  B1: base median crossing |w2| at a; B2: pooled median crossing R over all existing
                  settings; B3: base fraction placed by each budget at a.
  predictions     U, C, B1, B2 (primary: median crossing R) and C, B3 (secondary: fraction placed by
                  budget) for each held-out setting.
  train / score   the registered runs and scoring (after registration only).

    python -m src.prospective ghat | rglob | calibrate | predict | train | score
"""

from __future__ import annotations

import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.30, 1.50)
BUDGETS = (1_000, 2_000, 4_000, 8_000, 12_000)
N_SEEDS = 90
SEED_OFFSET = 100_000          # disjoint from every earlier window run (seeds 0-39)
WIDTH = 0.005
WORKERS = int(os.environ.get("PR_WORKERS", "4"))


def windows():
    """Existing (Block G) and held-out windows as blockG_windows.Window objects."""
    from .blockG_windows import WINDOWS, Window
    held = []
    f = RESULTS / "prospective_windows.csv"
    if f.exists():
        for r in pd.read_csv(f).itertuples():
            held.append(Window(f"H{int(round(r.quantile * 100)):02d}", -r.i, r.i, r.o1, r.o2))
    return list(WINDOWS), held


def _win_tuple(w):
    return (w.i_lo, w.i_hi, w.o_lo, w.o_hi)


def ghat_window(args, rel=1e-3, h0=0.05):
    w, a = args
    from .profiled_bnb import gap
    i_lo, i_hi, o_lo, o_hi = _win_tuple(w)
    W = 2 * a / min(i_hi + o_hi, o_hi - i_lo)
    nw, nb = int(math.ceil(2 * W / h0)), int(math.ceil(2 * math.pi / h0))
    hw, hb = W / nw, math.pi / nb
    wc = -W + (np.arange(nw) + 0.5) * 2 * hw
    bc = (np.arange(nb) + 0.5) * 2 * hb
    cw, cb = np.meshgrid(wc, bc, indexing="ij"); cw, cb = cw.ravel(), cb.ravel()
    best, arg = -np.inf, None
    reach = max(abs(i_lo), abs(i_hi)) + o_hi
    for _ in range(60):
        G = gap(cw, cb, a, inner=(i_lo, i_hi), outer=(o_lo, o_hi))
        j = int(np.argmax(G))
        if G[j] > best:
            best, arg = float(G[j]), (float(cw[j]), float(cb[j]))
        ub = G + (1 + a) * (reach * hw + 2 * hb)
        keep = ub > best
        hi = float(ub[keep].max()) if keep.any() else best
        if hi - best <= rel * best:
            return {"window": w.tag, "a": a, "Ghat_lo": best, "Ghat_hi": hi, "w1": arg[0], "b1": arg[1]}
        cw, cb = cw[keep], cb[keep]
        hw, hb = hw / 2, hb / 2
        cw = np.concatenate([cw - hw, cw - hw, cw + hw, cw + hw])
        cb = np.concatenate([cb - hb, cb + hb, cb - hb, cb + hb])
    raise RuntimeError("Ĝ not converged")


def ghat():
    existing, held = windows()
    jobs = [(w, a) for w in existing + held for a in A_VALUES]
    with Pool(WORKERS) as p:
        out = p.map(ghat_window, jobs)
    pd.DataFrame(out).to_csv(RESULTS / "prospective_ghat.csv", index=False)
    print(pd.DataFrame(out).to_string(index=False))


def _rglob_job(args):
    w, a = args
    from .profiled_bnb import certify
    x, y = (t.numpy() for t in w.population(400))
    win = _win_tuple(w)

    def plus(s):
        rm = certify(s, a, x, y, "-", win=win)
        rp = certify(s, a, x, y, "+", win=win)
        if rp["lower"] > rm["upper"]:
            return False, rm, rp
        if rm["lower"] > rp["upper"]:
            return True, rm, rp
        rm = certify(s, a, x, y, "-", tol=1e-9, win=win)
        rp = certify(s, a, x, y, "+", tol=1e-9, win=win)
        if rm["lower"] > rp["upper"]:
            return True, rm, rp
        if rp["lower"] > rm["upper"]:
            return False, rm, rp
        raise RuntimeError(f"sign unresolved at s={s} ({w.tag}, a={a})")
    from .prospective import ghat_window as _gw
    G = _gw((w, a))["Ghat_lo"]
    # bracket in R starting from [0.20, 0.24], stepping 0.04 outward until the sign changes,
    # then bisection in |w2| until the width is below one grid step (0.05)
    Rlo, Rhi, evals = 0.20, 0.24, 0
    flo, fhi = plus(2 * Rlo / G)[0], plus(2 * Rhi / G)[0]
    evals += 2
    while flo and Rlo > 0.02:
        Rhi, fhi = Rlo, flo
        Rlo = round(Rlo - 0.04, 6); flo = plus(2 * Rlo / G)[0]; evals += 1
    while not fhi and Rhi < 1.0:
        Rlo, flo = Rhi, fhi
        Rhi = round(Rhi + 0.04, 6); fhi = plus(2 * Rhi / G)[0]; evals += 1
    if flo or not fhi:
        return {"window": w.tag, "a": a, "w2_glob_lo": np.nan, "w2_glob_hi": np.nan, "evaluations": evals}
    lo, hi = 2 * Rlo / G, 2 * Rhi / G
    while hi - lo >= 0.05:
        mid = 0.5 * (lo + hi)
        evals += 1
        if plus(mid)[0]:
            hi = mid
        else:
            lo = mid
    return {"window": w.tag, "a": a, "w2_glob_lo": lo, "w2_glob_hi": hi, "evaluations": evals}


def rglob(which="all"):
    existing, held = windows()
    ws = {"all": existing + held, "held": held, "existing": existing}[which]
    jobs = [(w, a) for w in ws for a in A_VALUES]
    with Pool(WORKERS) as p:
        out = p.map(_rglob_job, jobs)
    d = pd.DataFrame(out)
    g = pd.read_csv(RESULTS / "prospective_ghat.csv")
    d = d.merge(g[["window", "a", "Ghat_lo"]], on=["window", "a"], how="left")
    d["R_glob"] = 0.5 * (d.w2_glob_lo + d.w2_glob_hi) * d.Ghat_lo / 2
    d["R_glob_lo"], d["R_glob_hi"] = d.w2_glob_lo * d.Ghat_lo / 2, d.w2_glob_hi * d.Ghat_lo / 2
    out_f = RESULTS / f"prospective_rglob_{which}.csv"
    d.to_csv(out_f, index=False)
    print(d.to_string(index=False))


if __name__ == "__main__" and sys.argv[1] in ("ghat", "rglob"):
    cmd = sys.argv[1]
    if cmd == "ghat":
        ghat()
    elif cmd == "rglob":
        rglob(sys.argv[2] if len(sys.argv) > 2 else "all")


# ------------------------------------------------------------------ calibration and predictions
def _cert_ghat():
    g = pd.read_csv(RESULTS / "prospective_ghat.csv")
    return {(r.window, round(r.a, 2)): r.Ghat_lo for r in g.itertuples()}


def calibrate():
    """Frozen existing data only (Block G runs, same protocol as the new runs)."""
    gh = _cert_ghat()
    bg = pd.read_csv(RESULTS / "blockG_crossings.csv")
    bg = bg[bg.cross_step.notna()].copy()
    bg["R_cert"] = [w2 * gh[(win, round(a, 2))] / 2 for win, a, w2 in zip(bg.window, bg.a, bg.cross_w2)]
    rg = pd.read_csv(RESULTS / "prospective_rglob_all.csv")
    rg = {(r.window, round(r.a, 2)): r.R_glob for r in rg.itertuples()}
    rows = []
    for (win, a), g in bg.groupby(["window", bg.a.round(2)]):
        rows.append({"window": win, "a": a, "n_crossed": len(g), "median_cross_R": g.R_cert.median(),
                     "median_cross_w2": g.cross_w2.median(), "R_glob": rg[(win, a)],
                     "log_ratio": math.log(g.R_cert.median() / rg[(win, a)])})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "prospective_existing_settings.csv", index=False)
    all_g = pd.read_csv(RESULTS / "blockG_crossings.csv")
    cal = []
    for a in A_VALUES:
        b = t[(t.window == "base") & (t.a == a)].iloc[0]
        base_all = all_g[(all_g.window == "base") & (all_g.a.round(2) == a)]
        frac = {f"B3_frac_by_{B}": float((base_all.cross_step <= B).mean()) for B in BUDGETS}
        cal.append({"a": a, "lambda_fitted": b.median_cross_R / b.R_glob,
                    "B1_base_median_cross_w2": b.median_cross_w2,
                    "B2_pooled_median_cross_R": float(bg.R_cert.median()), **frac})
    c = pd.DataFrame(cal)
    c["U_expected_log_error_min"] = t.log_ratio.min()
    c["U_expected_log_error_max"] = t.log_ratio.max()
    c.to_csv(RESULTS / "prospective_calibration.csv", index=False)
    print(t.to_string(index=False)); print(c.T.to_string())


def _growth_curves(a):
    """Base-task |w2| trajectories (phase2b checkpoints, budget 32k, 40 seeds) -> |w2| at each budget."""
    cols = ["a", "budget", "seed", "step", "w2"]
    parts = []
    for ch in pd.read_csv(RESULTS / "phase2b_checkpoints.csv", usecols=cols, chunksize=2_000_000):
        parts.append(ch[(ch.a.round(2) == a) & (ch.budget == 32_000)])
    d = pd.concat(parts)
    out = {}
    for s, g in d.groupby("seed"):
        g = g.sort_values("step")
        out[s] = {B: float(np.interp(B, g.step, g.w2.abs())) for B in BUDGETS}
    return pd.DataFrame(out).T


def predict():
    gh = _cert_ghat()
    cal = pd.read_csv(RESULTS / "prospective_calibration.csv").set_index("a")
    rg = pd.read_csv(RESULTS / "prospective_rglob_all.csv")
    rg = rg[rg.window.str.match(r"H\d\d")]
    rows = []
    for r in rg.itertuples():
        a = round(r.a, 2)
        c = cal.loc[a]
        G = gh[(r.window, a)]
        pred = {"U": r.R_glob, "C": c.lambda_fitted * r.R_glob,
                "B1": c.B1_base_median_cross_w2 * G / 2, "B2": c.B2_pooled_median_cross_R}
        for model, v in pred.items():
            rows.append({"window": r.window, "a": a, "target": "median_cross_R", "model": model,
                         "budget": np.nan, "prediction": v, "R_glob": r.R_glob, "Ghat": G})
        curves = _growth_curves(a)
        w2_c = 2 * pred["C"] / G
        for B in BUDGETS:
            rows.append({"window": r.window, "a": a, "target": "frac_placed", "model": "C", "budget": B,
                         "prediction": float((curves[B] >= w2_c).mean()), "R_glob": r.R_glob, "Ghat": G})
            rows.append({"window": r.window, "a": a, "target": "frac_placed", "model": "B3", "budget": B,
                         "prediction": float(c[f"B3_frac_by_{B}"]), "R_glob": r.R_glob, "Ghat": G})
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "prospective_predictions.csv", index=False)
    print(d[d.target == "median_cross_R"].pivot_table(index=["window", "a"], columns="model",
                                                      values="prediction").to_string())


# ------------------------------------------------------------------ training (after registration)
def _train_job(args):
    w, a, seed = args
    import torch
    from .blockG_windows import train
    torch.set_num_threads(1)
    r = train(w, a, seed, gs=1.0)            # R recomputed below with the certified Ĝ
    return {k: r[k] for k in ("window", "a", "seed", "cross_step", "cross_w2", "final_placed",
                              "final_solved")}


def train_runs(seeds=None):
    _, held = windows()
    seeds = seeds or range(SEED_OFFSET, SEED_OFFSET + N_SEEDS)
    out = RESULTS / "prospective_runs.csv"
    done = set()
    if out.exists():
        d0 = pd.read_csv(out)
        done = set(zip(d0.window, d0.a.round(2), d0.seed))
    jobs = [(w, a, s) for w in held for a in A_VALUES for s in seeds if (w.tag, a, s) not in done]
    with Pool(WORKERS) as p:
        rows = p.map(_train_job, jobs)
    new = pd.DataFrame(rows)
    d = pd.concat([pd.read_csv(out), new]) if out.exists() else new
    d.to_csv(out, index=False)
    print(d.groupby(["window", "a"]).cross_step.apply(lambda s: s.notna().sum()).to_string())


def _boot_diff(err_a, err_b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(err_a)
    idx = rng.integers(0, k, (n, k))
    diffs = (err_a[idx] - err_b[idx]).mean(axis=1)
    return float(np.mean(err_a - err_b)), *np.percentile(diffs, [2.5, 97.5])


def score():
    gh = _cert_ghat()
    runs = pd.read_csv(RESULTS / "prospective_runs.csv")
    runs["R_cert"] = [w2 * gh[(w, round(a, 2))] / 2 if w2 == w2 else np.nan
                      for w, a, w2 in zip(runs.window, runs.a, runs.cross_w2)]
    obs = runs[runs.cross_step.notna()].groupby(["window", runs.a.round(2)]).agg(
        n_crossed=("R_cert", "size"), obs_median_cross_R=("R_cert", "median")).reset_index()
    pred = pd.read_csv(RESULTS / "prospective_predictions.csv")
    p1 = pred[pred.target == "median_cross_R"].pivot_table(index=["window", "a"], columns="model",
                                                          values="prediction").reset_index()
    m = obs.merge(p1, on=["window", "a"])
    for model in ("U", "C", "B1", "B2"):
        m[f"abslogerr_{model}"] = np.abs(np.log(m[model] / m.obs_median_cross_R))
    rows = []
    for other in ("B1", "B2", "U"):
        d, lo, hi = _boot_diff(m["abslogerr_C"].values, m[f"abslogerr_{other}"].values)
        rows.append({"comparison": f"C - {other}", "mean_diff": d, "ci95_lo": lo, "ci95_hi": hi,
                     "C_better_interval_excludes_zero": hi < 0})
    comp = pd.DataFrame(rows)
    sec = []
    for (w, a), g in runs.groupby(["window", runs.a.round(2)]):
        for B in BUDGETS:
            o = float((g.cross_step <= B).mean())
            for model in ("C", "B3"):
                pv = float(pred[(pred.window == w) & (pred.a.round(2) == a) & (pred.target == "frac_placed")
                                & (pred.model == model) & (pred.budget == B)].prediction.iloc[0])
                sec.append({"window": w, "a": a, "budget": B, "model": model, "pred": pv, "obs": o,
                            "abs_err": abs(pv - o)})
    sec = pd.DataFrame(sec)
    per = sec.pivot_table(index=["window", "a"], columns="model", values="abs_err", aggfunc="mean")
    d2, lo2, hi2 = _boot_diff(per["C"].values, per["B3"].values)
    comp = pd.concat([comp, pd.DataFrame([{"comparison": "secondary: C - B3 (MAE of placed fraction)",
                                           "mean_diff": d2, "ci95_lo": lo2, "ci95_hi": hi2,
                                           "C_better_interval_excludes_zero": hi2 < 0}])])
    cal = pd.read_csv(RESULTS / "prospective_calibration.csv").iloc[0]
    m["U_logerr"] = np.log(m["U"] / m.obs_median_cross_R)
    m["U_in_expected_range"] = ((-m.U_logerr) >= cal.U_expected_log_error_min - 1e-12) & \
                               ((-m.U_logerr) <= cal.U_expected_log_error_max + 1e-12)
    m.to_csv(RESULTS / "prospective_scores.csv", index=False)
    comp.to_csv(RESULTS / "prospective_comparisons.csv", index=False)
    sec.to_csv(RESULTS / "prospective_secondary.csv", index=False)
    print(m.to_string(index=False)); print(comp.to_string(index=False))
    print(sec.groupby("model").abs_err.mean().to_string())


if __name__ == "__main__" and sys.argv[1] in ("calibrate", "predict", "train", "score"):
    {"calibrate": calibrate, "predict": predict, "train": train_runs, "score": score}[sys.argv[1]]()
