"""Asymmetric windows, Track 2 Step 1: exploratory pilot (results/asym_pilot_design.md).  Not a registered test.

I = [−0.8, 0.8], O = [−2.0, −1.2] ∪ [1.2, 2.0 + Δ].  Width-2 conditional minimisation at fixed scale s (the validated
search's optimiser), with a width-1 reference (ṽ fixed at (±1, 0)).  Placement is the continuous-window G₊ on the
asymmetric windows.

    python -m src.asym_pilot run [width2|width1]      (3 workers, nice 15; checkpoints per scale)
    python -m src.asym_pilot report
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "asym_pilot.csv"
A = 1.30
DELTA = 0.8
SCALES = tuple(10 ** (-1 + k / 4) for k in range(13))
RESTARTS = {"width2": 400, "width1": 200}
CAP = 3000
POLISH = 5
TIE = 1e-9


def windows(delta):
    return (-0.8, 0.8), ((-2.0, -1.2), (1.2, 2.0 + delta))


def set_windows(delta):
    """Reset the window globals the exact-extrema and dense-gap code read (this process only)."""
    from . import width2_geometry as g
    inner, outer = windows(delta)
    g.INNER, g.OUTER = inner, outer
    g._XI = np.linspace(*inner, 801)
    g._XO = np.concatenate([np.linspace(*outer[0], 401), np.linspace(*outer[1], 401)])


def population(delta):
    """400 inner points as the base population; 400 outer points split by length (Δ = 0: the base population)."""
    import torch
    lin = lambda lo, hi, n: torch.linspace(lo, hi, n, dtype=torch.float64).numpy()   # as the base population
    nL = int(round(400 * 0.8 / (1.6 + delta)))
    nR = 400 - nL
    inner = lin(-0.8, 0.8, 400)
    right = lin(1.2, 2.0 + delta, nR)
    left = -lin(1.2, 2.0, nL)
    x = np.concatenate([inner, right, left])
    y = np.concatenate([np.zeros(400), np.ones(400)])
    return x, y


def _width1_batch(s, x, y, act, restarts, seed):
    """Width-1 reference: the batched search with t fixed at ±1 (gradient in t is zero at |t| = 1)."""
    from . import width2_conditional as wc
    rng = np.random.default_rng(seed)
    P0 = np.column_stack([rng.uniform(0, wc.BOX_ALPHA, restarts), rng.uniform(0, wc.TWO_PI, restarts),
                          np.zeros(restarts), np.zeros(restarts), rng.choice([-1.0, 1.0], restarts)])
    sg = np.ones(restarts)
    P, L, G, it = wc.bfgs_batch(lambda Q, rows: wc.loss_grad_batch(Q, sg[rows], s, x, y, act), P0, maxit=CAP)
    return [{"k": k, "p": P[k], "sigma": 1, "loss": float(L[k]), "gnorm": float(np.abs(G[k]).max())}
            for k in range(restarts)]


def _polish(c, s, x, y, act, width):
    from .width2_conditional import profile_b
    from .width2_finish import newton
    from .width2_unplaced import g_exact, loss
    p = c["p"]
    if width == "width2":
        t = float(np.clip(p[4], -1, 1))
        q = np.r_[p[:4], t, c["sigma"] * (1 - abs(t))]
    else:
        q = np.r_[p[:4], np.sign(p[4]), 0.0]
    z0 = np.r_[q, profile_b(s * (q[4] * act.u(q[0] * x + q[1]) + q[5] * act.u(q[2] * x + q[3])), y)]
    if width == "width2":
        z, it, gmax, conv = newton(z0, s, x, y, act)
        q = z[:6]
    L = loss(q, s, x, y, act)
    g = g_exact(q, act)
    return {"loss": L, "q": q, "G_lo": float(g[0]), "G_hi": float(g[1]), "placed": bool(g[0] > 0),
            "unplaced": bool(g[1] <= 0)}


def eval_scale(width, k):
    from .width2_conditional import constant_predictor_loss, search_batch
    from .width2_geometry import Act
    set_windows(DELTA)
    act = Act("fa", A)
    x, y = population(DELTA)
    s = SCALES[k]
    if width == "width2":
        _, cands = search_batch(s, x, y, act, restarts=RESTARTS[width], seed=9000 + k, maxit=CAP)
        cands = [c for c in cands if c["p"] is not None and np.all(np.isfinite(c["p"]))]
    else:
        cands = [c for c in _width1_batch(s, x, y, act, RESTARTS[width], 9500 + k) if np.all(np.isfinite(c["p"]))]
    cands.sort(key=lambda c: c["loss"])
    picked, seen = [], []
    for c in cands:                                        # five lowest distinct losses
        if all(abs(c["loss"] - l0) > 1e-7 for l0 in seen):
            picked.append(c); seen.append(c["loss"])
        if len(picked) == POLISH:
            break
    pol = [_polish(c, s, x, y, act, width) for c in picked]
    const = constant_predictor_loss(y)
    best = min(pol, key=lambda r: r["loss"])
    bp = min((r["loss"] for r in pol if r["placed"]), default=np.inf)
    bu = min((r["loss"] for r in pol if not r["placed"]), default=np.inf)
    row = {"width": width, "a": A, "delta": DELTA, "k": k, "s": s, "restarts": RESTARTS[width],
           "retained_loss": min(best["loss"], const), "retained_constant": const < best["loss"],
           "G_lo": best["G_lo"], "G_hi": best["G_hi"], "placed": best["placed"] and best["loss"] <= const,
           "best_placed_loss": bp, "best_unplaced_loss": bu,
           "undecided": bool(np.isfinite(bp) and np.isfinite(bu) and abs(bp - bu) < TIE),
           "n_polished_placed": sum(r["placed"] for r in pol),
           "q": json.dumps([float(v) for v in best["q"]]),
           "raw_best_loss": cands[0]["loss"], "raw_not_converged": sum(c["gnorm"] > 1e-6 for c in cands)}
    return row


def _job(args):
    os.nice(15)
    width, k = args
    row = eval_scale(width, k)
    print(json.dumps({key: row[key] for key in ("width", "k", "s", "retained_loss", "G_lo", "placed", "undecided")}),
          flush=True)
    return row


def run(widths):
    from multiprocessing import get_context
    done = set() if not OUT.exists() else {(r.width, r.k) for r in pd.read_csv(OUT).itertuples()}
    todo = [(w, k) for w in widths for k in range(len(SCALES)) if (w, k) not in done]
    with get_context("spawn").Pool(3) as pool:
        for row in pool.imap_unordered(_job, todo):
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def decide(d):
    """The design's go / no-go on the width-2 rows (sorted by s)."""
    d = d.sort_values("s")
    if len(d) < len(SCALES):
        return {"decision": "incomplete"}
    if d.undecided.any():
        return {"decision": "no-go", "why": "undecided scale(s)", "s": list(d.s[d.undecided])}
    pl = d.placed.values.astype(bool)
    changes = int((pl[1:] != pl[:-1]).sum())
    if changes != 1 or pl[0] or not pl[-1]:
        return {"decision": "no-go", "why": f"{changes} status changes; placed at smallest = {bool(pl[0])}"}
    su, sp = d.s[~pl], d.s[pl]
    span_u, span_p = float(su.max() / su.min()), float(sp.max() / sp.min())
    ok = span_u >= 3 and span_p >= 3
    return {"decision": "go" if ok else "no-go", "switch_between": (float(su.max()), float(sp.min())),
            "span_unplaced": span_u, "span_placed": span_p}


def report():
    d = pd.read_csv(OUT)
    pd.set_option("display.width", 250)
    print(d.drop(columns=["q"]).sort_values(["width", "s"]).to_string(index=False))
    print(json.dumps(decide(d[d.width == "width2"]), default=float))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:] or ["width2", "width1"])
    else:
        report()
