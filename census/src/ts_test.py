"""Task B: a second prospective test of the timescale account (registration: results/ts_test_registration.md).

Fresh training sets fold1d.make_data(200, seed), seeds 830,000–830,079 (unused anywhere before), a ∈ {1.45, 1.60}
(activation values the timescale fit never saw).  Own global thresholds (own_threshold.own_threshold, the grid method
of the phase-2b own thresholds) are computed for every seed and frozen with a hash BEFORE any Adam run.  Adam, standard
protocol (as phase2b_ordering.run: U(−1, 1)⁴ in float32 then double, lr 0.01, full batch), budget 32,000, every-step
crossing detection (phase2b_ordering.state).  At each crossing: the timescale ratio as Track 3 (growth d log|w₂|/dt over
the last min(100, step − 1) steps; relaxation lr·λ_min(D^{-1/2} H D^{-1/2}) at the branch, residual_timescale).

TS-1  median residual (w₂,cross/w₂,own − 1) = median(α + β·ratioᵢ) from the frozen Adam fit, within
      max(0.01, 0.25·|pred|)  (the SGD extension's tolerance)
TS-2  mean(|log(cross/own)| − |log(cross/pop)|): paired run-level bootstrap 95% interval entirely below 0
UNRESOLVED at an a with fewer than 30 crossings.  Registered sensitivity: non-crossers imputed at their final |w₂|.

    python -m src.ts_test own | freeze | train | score
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

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "ts_test"
A_VALUES = (1.45, 1.60)
SEEDS = tuple(range(830_000, 830_080))
BUDGET = 32_000
MIN_CROSS = 30
FROZEN_T = RESULTS / "ts_test_own_frozen.csv"


def _own_job(args):
    os.nice(15)
    from .own_threshold import _pop, own_threshold
    a, seed = args
    return own_threshold(a, seed, _pop(a)[1])


def own(workers=1):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    f = OUT / "own_parts.csv"
    done = set() if not f.exists() else {(round(r.a, 2), r.seed) for r in pd.read_csv(f).itertuples()}
    jobs = [(a, s) for a in A_VALUES for s in SEEDS if (round(a, 2), s) not in done]
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_own_job, jobs):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
            print(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}), flush=True)


def freeze():
    from .own_threshold import _pop
    d = pd.read_csv(OUT / "own_parts.csv").sort_values(["a", "seed"])
    d["w2_own"] = 0.5 * (d.w2_lo + d.w2_hi)
    d["w2_pop"] = [_pop(a)[1] for a in d.a]
    d.to_csv(FROZEN_T, index=False)
    h = hashlib.sha256(FROZEN_T.read_bytes()).hexdigest()
    (RESULTS / "ts_test_own_frozen.sha256").write_text(h + "\n")
    print(len(d), int(d.w2_own.isna().sum()), h)


def run_one(a, seed, budget=BUDGET):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data
    from .fold1d_theorem import maximum_gap
    from .phase2b_ordering import state
    from .residual_timescale import branch_hessian, relax_rate
    torch.set_num_threads(1)
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=1e-2)
    x, y = x.double(), y.double()
    gs = maximum_gap(a, resolution=600)
    hist = {}
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        opt.step()
        w2 = abs(float(th.detach()[2]))
        hist[step] = w2; hist.pop(step - 101, None)
        if state(th.detach(), f, a, gs)["placement_ok"]:
            win = min(100, step - 1)
            growth = math.log(hist[step] / hist[step - win]) / win if win >= 1 else float("nan")
            q = th.detach().numpy()
            st = opt.state[th]
            vhat = st["exp_avg_sq"].numpy() / (1 - 0.999 ** int(st["step"]))
            _, H, _, _ = branch_hessian(a, x.numpy(), y.numpy(), q[0], q[1], q[2], q[3])
            lam = relax_rate(H, vhat[[0, 1, 3]])
            return {"a": a, "seed": seed, "crossed": True, "step": step, "w2_cross": w2, "w2_final": w2,
                    "growth": growth, "relax": lam, "ratio": growth / lam if lam > 0 else float("nan")}
    return {"a": a, "seed": seed, "crossed": False, "step": float("nan"), "w2_cross": float("nan"),
            "w2_final": abs(float(th.detach()[2])), "growth": float("nan"), "relax": float("nan"), "ratio": float("nan")}


def _run_job(args):
    os.nice(15)
    r = run_one(*args)
    print(json.dumps({"a": r["a"], "seed": r["seed"], "done": True}), flush=True)
    return r


def train(workers=1):
    from multiprocessing import get_context
    if not FROZEN_T.exists():
        raise SystemExit("own thresholds not frozen")
    f = OUT / "runs.csv"
    done = set() if not f.exists() else {(round(r.a, 2), r.seed) for r in pd.read_csv(f).itertuples()}
    jobs = [(a, s) for a in A_VALUES for s in SEEDS if (round(a, 2), s) not in done]
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_run_job, jobs):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)


def paired_boot(d, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = rng.choice(d, (n, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def score_a(w2_x, w2_own, w2_pop, ratios, alpha, beta, tol_abs=0.01, tol_rel=0.25):
    w2_x, w2_own, ratios = (np.asarray(v, float) for v in (w2_x, w2_own, ratios))
    ok = np.isfinite(w2_own) & np.isfinite(w2_x)
    w2_x, w2_own, ratios = w2_x[ok], w2_own[ok], ratios[ok]
    if len(w2_x) < MIN_CROSS:
        return {"TS-1": "UNRESOLVED", "TS-2": "UNRESOLVED", "n": len(w2_x)}
    use = np.isfinite(ratios) & (ratios > 0)
    pred = float(np.median(alpha + beta * ratios[use])) if use.sum() >= MIN_CROSS else float("nan")
    obs = float(np.median(w2_x / w2_own - 1))
    tol = max(tol_abs, tol_rel * abs(pred)) if math.isfinite(pred) else float("nan")
    ts1 = "UNRESOLVED" if not math.isfinite(pred) else ("PASS" if abs(obs - pred) <= tol else "FAIL")
    diff = np.abs(np.log(w2_x / w2_own)) - np.abs(np.log(w2_x / w2_pop))
    m, lo, hi = paired_boot(diff)
    return {"n": len(w2_x), "TS-1": ts1, "pred": pred, "obs": obs, "tol": tol,
            "median_ratio": float(np.median(ratios[use])) if use.any() else float("nan"),
            "TS-2": "PASS" if hi < 0 else "FAIL", "TS2_mean": m, "TS2_lo": lo, "TS2_hi": hi}


def score():
    fit = json.loads((RESULTS / "residual_timescale_fit.json").read_text())
    own_ = pd.read_csv(FROZEN_T); own_["a"] = own_.a.round(2)
    r = pd.read_csv(OUT / "runs.csv"); r["a"] = r.a.round(2)
    m = r.merge(own_[["a", "seed", "w2_own", "w2_pop"]], on=["a", "seed"])
    rows = []
    for a in A_VALUES:
        g = m[m.a == round(a, 2)]
        c = g[g.crossed]
        prim = score_a(c.w2_cross, c.w2_own, g.w2_pop.iloc[0], c.ratio, fit["alpha"], fit["beta"])
        imp = np.where(g.crossed, g.w2_cross, g.w2_final)                    # registered sensitivity
        sens = score_a(imp, g.w2_own, g.w2_pop.iloc[0], np.where(g.crossed, g.ratio, np.nan), fit["alpha"], fit["beta"])
        rows.append({"a": a, "runs": len(g), "crossed": len(c), **{k: v for k, v in prim.items()},
                     **{f"sens_{k}": v for k, v in sens.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "scores.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"own": lambda: own(w), "freeze": freeze, "train": lambda: train(w), "score": score}[sys.argv[1]]()
