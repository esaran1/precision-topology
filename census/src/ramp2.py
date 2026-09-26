"""Follow-up (2), registered: the Adam ramp REDESIGNED with no stationary warm-up (design-corrected follow-up to Track
1B; registration results/ramp2_registration.md).  The original ramp's verdicts are unchanged.

Why: in the original Adam ramp the 4,000-step warm-up at a stationary point let Adam's v̂ decay 8–88× below its
free-training size, so ηλ_min(P^{1/2}HP^{1/2}) > 1 at crossing and the frozen-preconditioner linearisation did not
apply (ramp.adam_contrast, post hoc).  Here the ramp starts at initialisation: (w₁, b₁, b₂) from the standard
U(−1, 1) initialisation (float32 draw, cast), w₂ = +s(t), s(t) = S0·exp(γt) from the first step, Adam lr 0.01, and
every run is scored against the switch of the branch it actually tracks (continuation on its own sample from its
crossing state).

    python -m src.ramp2 design | run | branch | score
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import ramp as R

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "ramp2"
A_VALUES = (1.30, 1.50)
SEEDS = tuple(range(861_000, 861_040))
PILOT_SEEDS = tuple(range(869_100, 869_108))
S0 = 0.5                                   # |w₂| at the start (the median |w₂| of the standard initialisation is ≈ 0.5)
END_FRAC = 3.0
SWEEP = tuple(np.geomspace(1e-4, 3e-2, 11))
N_GAMMA = 6
MIN_PILOT_CROSS = 6                        # of 8 pilot runs, for a γ to be admissible
MAX_ETA_LAMBDA = 0.5                       # validity: median ηλ_min(P^{1/2}HP^{1/2}) at crossing (relaxation ≥ 2 steps)


def _placed_at_init(a, seeds):
    import torch
    th = R._init(seeds)
    inner, outer = R._windows()
    ti, to = th[:, 0:1] * inner + th[:, 1:2], th[:, 0:1] * outer + th[:, 1:2]
    G = (to + a * torch.sin(to)).min(dim=1).values - (ti + a * torch.sin(ti)).max(dim=1).values
    return (G > 0).numpy()


def run_cell(a, seeds, gamma):
    """batch_ramp from initialisation: no warm-up, s₀ = S0, random init; runs placed at initialisation are flagged."""
    s_end = END_FRAC * R._s_star_pop(a)
    steps = math.ceil(math.log(s_end / S0) / gamma)
    res = R.batch_ramp(a, seeds, "adam", gamma, s_star_pop=S0 / R.S0_FRAC, warmup=0, max_steps=steps, init="random")
    pi = _placed_at_init(a, seeds)
    for x, p in zip(res, pi):
        x["placed_at_init"] = bool(p)
    return res


def _eta_lambda(L, rec):
    p = 1.0 / (np.sqrt([rec["vhat_w1"], rec["vhat_b1"], rec["vhat_b2"]]) + 1e-8)
    H = L["H"]
    return R.LR["adam"] * float(np.linalg.eigvalsh((np.sqrt(p)[:, None] * H) * np.sqrt(p)[None, :]).min())


def design():
    """γ grid fixed before any registered run, from PILOT_SEEDS (outside every registered range), predictions only:
    admissible γ = sweep points where ≥ MIN_PILOT_CROSS of 8 pilot runs cross and the median ηλ_min at crossing is
    ≤ MAX_ETA_LAMBDA; the grid is N_GAMMA log-spaced values spanning the admissible range."""
    L = R._landscape()
    sweep, rows = [], []
    for a in A_VALUES:
        for g in SWEEP:
            res = run_cell(a, PILOT_SEEDS, g)
            cr = [x for x in res if x["crossed"] and not x["placed_at_init"]]
            el = [_eta_lambda(L[round(a, 2)], x) for x in cr]
            pr = [R.predict_run(L[round(a, 2)], "adam", g, x)["pred_r"] for x in cr]
            sweep.append({"a": a, "gamma": g, "n_crossed": len(cr), "median_eta_lambda": float(np.median(el)) if el else np.nan,
                          "median_pred_r": float(np.median(pr)) if pr else np.nan})
            print(sweep[-1], flush=True)
        sw = pd.DataFrame([r for r in sweep if r["a"] == a])
        ok = sw[(sw.n_crossed >= MIN_PILOT_CROSS) & (sw.median_eta_lambda <= MAX_ETA_LAMBDA)]
        if len(ok) < 2:
            rows.append({"a": a, "cell": -1, "gamma": np.nan, "note": "no admissible range"})
            continue
        for j, g in enumerate(np.geomspace(ok.gamma.min(), ok.gamma.max(), N_GAMMA)):
            rows.append({"a": a, "cell": j, "gamma": float(g), "note": "",
                         "steps": math.ceil(math.log(END_FRAC * R._s_star_pop(a) / S0) / g)})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(sweep).to_csv(OUT / "design_sweep.csv", index=False)
    d = pd.DataFrame(rows); d.to_csv(OUT / "design.csv", index=False)
    print(d.to_string(index=False))
    return d


def run():
    os.nice(15)
    d = pd.read_csv(OUT / "design.csv"); d = d[d.cell >= 0]
    f = OUT / "runs.json"
    done = json.loads(f.read_text()) if f.exists() else []
    have = {(round(r["a"], 2), r["cell"]) for r in done}
    for r in d.itertuples():
        if (round(r.a, 2), r.cell) in have:
            continue
        res = run_cell(r.a, SEEDS, r.gamma)
        done += [{"a": r.a, "cell": int(r.cell), "gamma": r.gamma, **{k: (v if not isinstance(v, (np.floating,)) else float(v))
                                                                          for k, v in x.items()}} for x in res]
        f.write_text(json.dumps(done, default=float))
        print(r.a, r.cell, sum(x["crossed"] and not x["placed_at_init"] for x in res), flush=True)


def _branch_job(args):
    """The switch of the branch a run tracks: Newton on its own sample at its crossing state and scale (lag_law.branch_point),
    then continuation in s (down if that branch point is placed, up if not; 1% steps) to the gap's sign change, bisected
    to 1e-6 relative."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    from .lag_law import branch_point, gap
    a, seed, w1, b1, b2, s_c = args
    X, Y = R._data([seed]); x, y = X[0].numpy(), Y[0].numpy()
    out = {"a": a, "seed": seed, "s_cross": s_c}
    z, gr = branch_point([w1, b1, b2], s_c, a, x, y)
    out["dist_to_branch"] = float(np.linalg.norm(z - np.array([w1, b1, b2])))
    if gr > 1e-8:
        return {**out, "s_branch": np.nan, "note": "no branch point at the crossing"}
    G = lambda z_: gap(z_[0], z_[1], a)
    placed = G(z) > 0
    step = 0.99 if placed else 1.01
    s, zs = s_c, z
    for _ in range(400):
        s_new = s * step
        z_new, gr = branch_point(zs, s_new, a, x, y)
        if gr > 1e-8 or np.linalg.norm(z_new - zs) > 0.5:
            return {**out, "s_branch": np.nan, "note": f"continuation lost at s={s_new:.4f}"}
        if (G(z_new) > 0) != placed:
            lo, hi = sorted([s, s_new]); zlo = z_new if s_new < s else zs
            while (hi - lo) / lo > 1e-6:
                mid = 0.5 * (lo + hi)
                zm, _ = branch_point(zlo, mid, a, x, y)
                if G(zm) > 0:
                    hi = mid
                else:
                    lo, zlo = mid, zm
            return {**out, "s_branch": 0.5 * (lo + hi), "note": ""}
        s, zs = s_new, z_new
    return {**out, "s_branch": np.nan, "note": "no sign change within 400 steps"}


def branch(workers=1):
    from multiprocessing import get_context
    runs = json.loads((OUT / "runs.json").read_text())
    f = OUT / "branch.json"
    done = json.loads(f.read_text()) if f.exists() else []
    have = {(round(r["a"], 2), r["seed"], round(r["s_cross"], 12)) for r in done}
    jobs = [(r["a"], r["seed"], r["w1"], r["b1"], r["b2"], r["s_cross"]) for r in runs
            if r["crossed"] and not r["placed_at_init"] and (round(r["a"], 2), r["seed"], round(r["s_cross"], 12)) not in have]
    with get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_branch_job, jobs)):
            done.append({k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()})
            if i % 20 == 0:
                f.write_text(json.dumps(done, default=float))
    f.write_text(json.dumps(done, default=float))


def cells_for_scoring(runs, br, L):
    """Per (a, cell): observed r = s_cross/s_branch − 1 and predicted κ_kχ (Adam, the run's measured v̂) for crossing
    runs with a branch switch; the cell's median ηλ_min at crossing; counts."""
    bk = {(round(b["a"], 2), b["seed"], round(b["s_cross"], 12)): b for b in br}
    out = {}
    for a in A_VALUES:
        cells = []
        for (cell, g), grp in pd.DataFrame([r for r in runs if round(r["a"], 2) == a]).groupby(["cell", "gamma"]):
            obs, pred, el = [], [], []
            for r in grp.to_dict("records"):
                if not r["crossed"] or r["placed_at_init"]:
                    continue
                b = bk.get((a, r["seed"], round(r["s_cross"], 12)))
                if b is None or not np.isfinite(b["s_branch"]):
                    continue
                obs.append(r["s_cross"] / b["s_branch"] - 1)
                pred.append(R.predict_run(L[a], "adam", g, r)["pred_r"])
                el.append(_eta_lambda(L[a], r))
            cells.append({"gamma": g, "cell": int(cell), "obs_r": np.array(obs), "pred_r": np.array(pred), "n_runs": len(grp),
                          "n_warmup_placed": 0, "n_placed_at_init": int(grp.placed_at_init.sum()),
                          "median_eta_lambda": float(np.median(el)) if el else np.nan})
        out[a] = cells
    return out


def score_setting2(cells):
    """Validity first: cells whose median ηλ_min at crossing exceeds MAX_ETA_LAMBDA are INVALID and dropped; then
    ramp.score_setting (R1–R3; ≥ 30 scored runs per cell, ≥ 4 resolved cells)."""
    valid = [c for c in cells if np.isfinite(c["median_eta_lambda"]) and c["median_eta_lambda"] <= MAX_ETA_LAMBDA]
    sc = R.score_setting(valid)
    return {**sc, "cells_valid": len(valid), "cells_total": len(cells)}


def score():
    L = R._landscape()
    runs = json.loads((OUT / "runs.json").read_text()); br = json.loads((OUT / "branch.json").read_text())
    C = cells_for_scoring(runs, br, L)
    out = []
    for a, cells in C.items():
        sc = score_setting2(cells)
        out.append({"a": a, **sc, "cells": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in cells]})
    (OUT / "verdicts.json").write_text(json.dumps(out, indent=1, default=float))
    for o in out:
        print({k: v for k, v in o.items() if k != "cells"})
    return out


if __name__ == "__main__":
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"design": design, "run": run, "branch": lambda: branch(w), "score": score}[sys.argv[1]]()
