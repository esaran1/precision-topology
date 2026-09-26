"""Registered boundary test (final night; registration results/ramp_boundary_registration.md): width-1 SGD forced ramps
at five growth rates chosen a priori so that χ targets are 0.03, 0.06, 0.1, 0.2 and 0.4, at a = 1.30 (winding −1) and
1.50 (winding 0), 40 fresh seeds per cell (862,000–862,039), scored against each seed's forced-branch switch computed
BEFORE any run.

Design (reuses src/ramp.py; SGD lr 0.3, full batch, 4,000-step warm-up at the population branch point θ*(s₀) on the
natural winding, s₀ = 0.5 s*, then s = s₀e^{γ(t−4000)}):
  γ(cell)   = χ_target · η · λ_min(H_pop)   (χ = γ/(ηλ_min) exactly for SGD; H_pop from lag_law/kappa.csv)
  end scale = s*·max(3, 1 + 4 κ χ_target)   (so the predicted crossing lies well inside the ramp)
  s_branch  = the forced branch's own-sample switch: Newton continuation from θ*(s₀) on the seed's own sample
              (ramp._branch_switch_job logic), computed and hashed before any run
  χ_own     = γ / (η λ_min(H_own)), H_own the Hessian at the forced branch point at s_branch (validity quantity)
  obs r     = s_cross/s_branch − 1;  pred r = κ_k χ with κ_k (committed lag_law) for the run's winding at crossing
  ratio     = obs r / pred r per run; the cell statistic is the median ratio.
Rules (ramp_boundary.score_cells):
  B1  median ratio in [0.75, 1.25] in the χ = 0.03 AND 0.06 cells at both a  (PASS iff all four in band)
  B2  median ratio outside [0.75, 1.25] in the χ = 0.4 cell at both a        (PASS iff both outside)
  χ*  (descriptive, per a): the smallest target whose median ratio leaves the band
  validity per cell: median χ_own within 30% of the target AND ≥ 30 crossings; an invalid cell is UNRESOLVED and
  a rule depending on it is UNRESOLVED.

    python -m src.ramp_boundary freeze | run | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import ramp as R

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "ramp_boundary"
SETTINGS = ((1.30, -1), (1.50, 0))
CHI_TARGETS = (0.03, 0.06, 0.1, 0.2, 0.4)
SEEDS = tuple(range(862_000, 862_040))
BAND = (0.75, 1.25)
CHI_TOL = 0.30
MIN_CROSS = 30


def design():
    kb = pd.read_csv(RESULTS / "lag_law" / "kappa_by_winding.csv"); L = R._landscape()
    rows = []
    for a, k in SETTINGS:
        lam = float(np.linalg.eigvalsh(L[round(a, 2)]["H"]).min())
        kap = float(kb[(kb.a.round(2) == round(a, 2)) & (kb.k == k)].kappa_sgd.iloc[0])
        for c in CHI_TARGETS:
            rows.append({"a": a, "winding": k, "chi_target": c, "gamma": c * R.LR["sgd"] * lam, "lambda_min_pop": lam,
                         "kappa": kap, "end_factor": max(3.0, 1 + 4 * kap * c), "s_star_pop": R._s_star_pop(a)})
    return pd.DataFrame(rows)


def _switch_job(args):
    """The forced branch's own-sample switch and λ_min of its Hessian there (before any run)."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    from .lag_law import hessian_and_tangent
    a, k, seed = args
    r = R._branch_switch_job((a, k, seed))
    out = {"a": a, "winding": k, "seed": seed, "s_branch": r["s_branch"], "note": r.get("note", "")}
    if np.isfinite(r["s_branch"]):
        from .lag_law import branch_point
        X, Y = R._data([seed]); x, y = X[0].numpy(), Y[0].numpy()
        s = R.S0_FRAC * R._s_star_pop(a)
        z, _ = branch_point(R.branch_init(a, k), s, a, x, y)
        # continue to just below the switch (unplaced side) in 1% steps, as the switch job does
        while s * 1.01 < r["s_branch"]:
            s *= 1.01
            z, _ = branch_point(z, s, a, x, y)
        z, _ = branch_point(z, r["s_branch"] * (1 - 1e-6), a, x, y)
        H, _ = hessian_and_tangent(z, r["s_branch"] * (1 - 1e-6), a, x, y)
        out["lambda_min_own"] = float(np.linalg.eigvalsh(H).min())
    return out


def freeze(workers=2):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    d = design(); d.to_csv(OUT / "design.csv", index=False)
    jobs = [(a, k, sd) for a, k in SETTINGS for sd in SEEDS]
    with get_context("spawn").Pool(workers) as pool:
        rows = list(pool.imap(_switch_job, jobs))
    f = pd.DataFrame(rows); f.to_csv(OUT / "frozen_switches.csv", index=False)
    h = {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in ("design.csv", "frozen_switches.csv")}
    (OUT / "frozen.sha256").write_text("".join(f"{v}  {n}\n" for n, v in h.items()))
    print(f.groupby(["a"]).agg(n=("s_branch", "size"), ok=("s_branch", lambda v: int(np.isfinite(v).sum()))), h)


def _cell_job(args):
    os.nice(15)
    a, k, c, gamma, end_factor, sp = args
    steps = math.ceil(math.log(end_factor / R.S0_FRAC) / gamma)
    res = R.batch_ramp(a, SEEDS, "sgd", gamma, sp, max_steps=R.WARMUP + steps, winding=k)
    return [{"a": a, "winding": k, "chi_target": c, "gamma": gamma, **x} for x in res]


def run(workers=2):
    from multiprocessing import get_context
    for n, v in (l.split()[::-1] for l in (OUT / "frozen.sha256").read_text().splitlines()):
        assert hashlib.sha256((OUT / n).read_bytes()).hexdigest() == v, n
    d = pd.read_csv(OUT / "design.csv")
    jobs = [(r.a, int(r.winding), r.chi_target, r.gamma, r.end_factor, r.s_star_pop) for r in d.itertuples()]
    with get_context("spawn").Pool(workers) as pool:
        out = [x for part in pool.imap(_cell_job, jobs) for x in part]
    (OUT / "runs.json").write_text(json.dumps(out, default=float))


def score_cells(cells):
    """cells: list of dicts {a, chi_target, ratios (per crossing run), chi_own_median, n_cross}.  Returns per-cell
    status and the rule verdicts."""
    lo, hi = BAND
    for c in cells:
        c["valid"] = (c["n_cross"] >= MIN_CROSS and np.isfinite(c["chi_own_median"])
                      and abs(c["chi_own_median"] / c["chi_target"] - 1) <= CHI_TOL)
        c["median_ratio"] = float(np.median(c["ratios"])) if len(c["ratios"]) else float("nan")
        c["in_band"] = bool(lo <= c["median_ratio"] <= hi) if np.isfinite(c["median_ratio"]) else None
    get = lambda a, x: [c for c in cells if round(c["a"], 2) == a and abs(c["chi_target"] - x) < 1e-12][0]
    avals = sorted({round(c["a"], 2) for c in cells})
    b1c = [get(a, x) for a in avals for x in (0.03, 0.06)]
    b2c = [get(a, 0.4) for a in avals]
    B1 = "UNRESOLVED" if not all(c["valid"] for c in b1c) else ("PASS" if all(c["in_band"] for c in b1c) else "FAIL")
    B2 = "UNRESOLVED" if not all(c["valid"] for c in b2c) else ("PASS" if all(not c["in_band"] for c in b2c) else "FAIL")
    chi_star = {}
    for a in avals:
        cs = sorted([c for c in cells if round(c["a"], 2) == a], key=lambda c: c["chi_target"])
        out = [c["chi_target"] for c in cs if c["in_band"] is False]
        chi_star[a] = out[0] if out else None
    return {"B1": B1, "B2": B2, "chi_star": chi_star, "cells": cells}


def score():
    for n, v in (l.split()[::-1] for l in (OUT / "frozen.sha256").read_text().splitlines()):
        assert hashlib.sha256((OUT / n).read_bytes()).hexdigest() == v, n
    L = R._landscape()
    d = pd.read_csv(OUT / "design.csv"); f = pd.read_csv(OUT / "frozen_switches.csv")
    runs = pd.DataFrame(json.loads((OUT / "runs.json").read_text()))
    runs = runs.merge(f[["a", "seed", "s_branch", "lambda_min_own"]], on=["a", "seed"], how="left")
    cells, rows = [], []
    for r in d.itertuples():
        g = runs[(runs.a.round(2) == round(r.a, 2)) & (np.isclose(runs.chi_target, r.chi_target))]
        cr = g[g.crossed & np.isfinite(g.s_branch)]
        ratios, chis = [], []
        for x in cr.to_dict("records"):
            pr = R.predict_run(L[round(r.a, 2)], "sgd", r.gamma, x)
            obs = x["s_cross"] / x["s_branch"] - 1
            ratios.append(obs / pr["pred_r"]); chis.append(r.gamma / (R.LR["sgd"] * x["lambda_min_own"]))
            rows.append({"a": r.a, "chi_target": r.chi_target, "seed": x["seed"], "obs_r": obs, "pred_r": pr["pred_r"],
                         "winding_k": pr["winding_k"], "chi_own": chis[-1], "ratio": ratios[-1]})
        cells.append({"a": r.a, "chi_target": r.chi_target, "ratios": ratios, "n_cross": int(len(cr)),
                      "n_runs": int(len(g)), "chi_own_median": float(np.median(chis)) if chis else float("nan"),
                      "n_warmup_placed": int(g.placed_in_warmup.sum())})
    S = score_cells(cells)
    pd.DataFrame(rows).to_csv(OUT / "scored_runs.csv", index=False)
    out = {**{k: v for k, v in S.items() if k != "cells"},
           "cells": [{k: v for k, v in c.items() if k != "ratios"} for c in S["cells"]]}
    (OUT / "verdicts.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    {"freeze": lambda: freeze(w), "run": lambda: run(w), "score": score}[sys.argv[1]]()
