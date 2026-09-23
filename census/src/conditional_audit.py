"""Block 1a: rerun the frozen conditional search retaining every candidate.

Registered in results/conditional_audit_prediction.md (6b17508).

At each output scale s = |w2| the frozen search (blockB_landscape.best_conditional) is
replayed call for call through blockB_landscape.minimise, and every restart is kept:
screen (600 steps) for all restarts, full (3,000 steps) for the 8 carried.  Each row has
the final parameters, loss, class gap G, gradient norm, step count, and the discard reason
(degenerate / not carried / not lowest).  The exact constant predictor (w1 = 0, b2 = 0,
loss log 2) is added at every s.  The retained candidate must equal best_conditional's.

Scales: the full coarse grid 1.5..11 (0.5) plus the fine 0.05 grid over the bracket
[w2_thr - 0.5, w2_thr] of each reported threshold (blockB_switches.csv), i.e. every scale
the frozen scan evaluates for R_glob (50 restarts) and R_solve (24 restarts).

    python -m src.conditional_audit candidates
"""

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
OUT = RESULTS / "cond_audit_candidates.csv"


def _grad_norm(a, w2, p, x, y):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation
    f = activation("sin_family", a)
    v = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    out = torch.tensor(w2, dtype=torch.float64) * f(v[0] * x + v[1]) + v[2]
    F.binary_cross_entropy_with_logits(out, y).backward()
    return float(v.grad.norm())


def scales(a):
    sw = pd.read_csv(RESULTS / "blockB_switches.csv")
    r = sw[sw.a.round(2) == round(a, 2)].iloc[0]
    out = {("glob", round(float(s), 4)) for s in np.arange(1.5, 11.0 + 1e-9, 0.5)}
    out |= {("solve", round(float(s), 4)) for s in np.arange(1.5, 11.0 + 1e-9, 0.5)}
    for kind, col in (("glob", "w2_glob"), ("solve", "w2_solve")):
        thr = float(r[col])
        out |= {(kind, round(float(s), 4)) for s in np.arange(thr - 0.5, thr + 1e-9, 0.05)}
    return sorted(out)


def job(args):
    a, kind, s = args
    import torch
    from . import blockB_landscape as bb
    torch.set_num_threads(1)
    x, y = bb.population_data()
    restarts = bb.RESTARTS if kind == "glob" else 24
    rows = []
    cheap = []
    for seed in range(restarts):
        c = bb.minimise(a, s, x, y, seed=seed, steps=bb.SCREEN)
        c.update(seed=seed, stage="screen", steps=bb.SCREEN, degenerate=bb.is_degenerate(c))
        cheap.append(c)
    ok = sorted([c for c in cheap if not c["degenerate"]], key=lambda c: c["loss"])
    carried = {id(c) for c in ok[:bb.KEEP]}
    full = []
    for c in ok[:bb.KEEP]:
        d = bb.minimise(a, s, x, y, start=[c["w1"], c["b1"], c["b2"]], steps=bb.STEPS)
        d.update(seed=c["seed"], stage="full", steps=bb.SCREEN + bb.STEPS,
                 degenerate=bb.is_degenerate(d))
        full.append(d)
    good = [d for d in full if not d["degenerate"]]
    kept = min(good, key=lambda d: d["loss"]) if good else None
    ref = bb.best_conditional(a, s, x, y, restarts=restarts)
    same = (ref is None and kept is None) or (ref is not None and kept is not None
                                                and abs(ref["loss"] - kept["loss"]) < 1e-12)
    for c in cheap:
        reason = ("degenerate" if c["degenerate"] else
                  "carried" if id(c) in carried else "not carried (screen rank > 8)")
        rows.append({**{k: c[k] for k in ("seed", "stage", "steps", "w1", "b1", "b2", "loss", "gap",
                                          "solves", "degenerate")}, "status": reason})
    for d in full:
        reason = ("degenerate" if d["degenerate"] else
                  "RETAINED" if d is kept else "not lowest")
        rows.append({**{k: d[k] for k in ("seed", "stage", "steps", "w1", "b1", "b2", "loss", "gap",
                                          "solves", "degenerate")}, "status": reason})
    rows.append({"seed": -1, "stage": "constant", "steps": 0, "w1": 0.0, "b1": 0.0, "b2": 0.0,
                 "loss": math.log(2), "gap": 0.0, "solves": False, "degenerate": True,
                 "status": "constant predictor (exact)"})
    for r in rows:
        r.update(a=a, kind=kind, s=s, frozen_reproduced=same,
                 grad_norm=(_grad_norm(a, s, [r["w1"], r["b1"], r["b2"]], x, y)
                            if r["stage"] != "constant" else 0.0))
    return rows


def candidates(workers=10):
    jobs = [(a, k, s) for a in A_VALUES for (k, s) in scales(a)]
    print(f"{len(jobs)} (a, kind, s) jobs", flush=True)
    with Pool(workers) as p:
        out = [r for rs in p.imap_unordered(job, jobs) for r in rs]
    d = pd.DataFrame(out)
    d.to_csv(OUT, index=False)
    print(f"{len(d)} candidate rows; frozen reproduced at "
          f"{d.groupby(['a','kind','s']).frozen_reproduced.first().sum()} of {len(jobs)} scales")


def plot():
    """Loss against output scale for every candidate, per a (glob searches, 50 restarts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT)
    d = d[d.kind == "glob"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
    for ax, (a, g) in zip(axes.ravel(), d.groupby("a")):
        styles = [("degenerate", "x", "0.75", "degenerate (discarded)"),
                  ("not carried (screen rank > 8)", ".", "0.55", "screen, not carried"),
                  ("not lowest", "o", "#9ecae1", "full run, not lowest")]
        for st, mk, col, lab in styles:
            h = g[g.status == st]
            ax.scatter(h.s, h.loss, marker=mk, s=6, color=col, label=lab, lw=0.5)
        ret = g[g.status == "RETAINED"].sort_values("s")
        ax.plot(ret.s, ret.loss, "-", color="#08519c", lw=1.2, label="retained branch")
        ax.axhline(math.log(2), color="k", ls="--", lw=0.8, label="constant predictor (log 2)")
        for sign, mk, col, lab in ((-1, "v", "#d62728", "best G ≤ 0"), (1, "^", "#2ca02c", "best G > 0")):
            sub = g[(g.stage != "constant") & ((g.gap > 0) if sign > 0 else (g.gap <= 0))]
            best = sub.loc[sub.groupby("s").loss.idxmin()]
            ax.scatter(best.s, best.loss, marker=mk, s=10, color=col, label=lab, zorder=3)
        sw = pd.read_csv(RESULTS / "blockB_switches.csv")
        thr = float(sw[sw.a.round(2) == round(a, 2)].w2_glob.iloc[0])
        ax.axvline(thr, color="#2ca02c", ls=":", lw=0.8)
        ax.set_title(f"a = {a:.2f} (frozen $|w_2|_{{glob}}$ = {thr:.2f})", fontsize=8)
        ax.set_xlabel("output scale $|w_2|$", fontsize=8)
        ax.set_ylim(0.0, 1.6)
    axes[0, 0].set_ylabel("conditional loss", fontsize=8)
    axes[1, 0].set_ylabel("conditional loss", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="upper right", frameon=False)
    fig.tight_layout()
    out = RESULTS / "figures" / "cond_audit_candidates.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(out)


if __name__ == "__main__" and sys.argv[1] in ("candidates", "plot"):
    {"candidates": candidates, "plot": plot}[sys.argv[1]]()


# ---------------------------------------------------------------------------------------------
# 1e: stricter optimisation near each threshold (registered: 10x steps, grad-norm tolerance 1e-8)
STRICT_SCREEN, STRICT_FULL, GRAD_TOL = 6_000, 30_000, 1e-8
STRICT_OUT = RESULTS / "cond_audit_strict.csv"


def _minimise_strict(a, w2, x, y, start=None, seed=0, steps=STRICT_FULL):
    import torch
    from torch.nn import functional as F
    from . import blockB_landscape as bb
    from .fold1d import activation, solves
    f = activation("sin_family", a)
    if start is None:
        g = torch.Generator().manual_seed(seed)
        p = torch.stack([torch.rand(1, generator=g).double()[0] * 4 - 2,
                         torch.rand(1, generator=g).double()[0] * 8 - 4,
                         torch.rand(1, generator=g).double()[0] * 2 * abs(w2) - abs(w2)])
    else:
        p = torch.tensor(start, dtype=torch.float64)
    p = p.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=3e-2)
    w2t = torch.tensor(w2, dtype=torch.float64)
    used = steps
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(w2t * f(p[0] * x + p[1]) + p[2], y)
        loss.backward()
        if i % 100 == 0 and float(p.grad.norm()) <= GRAD_TOL:
            used = i
            break
        opt.step()
        if i == steps // 2:
            for gp in opt.param_groups:
                gp["lr"] = 5e-3
    q = p.detach()
    with torch.no_grad():
        loss = float(F.binary_cross_entropy_with_logits(w2t * f(q[0] * x + q[1]) + q[2], y))
    w1, b1, b2 = (float(v) for v in q)
    theta = torch.tensor([w1, b1, w2, b2], dtype=torch.float64)
    return {"w1": w1, "b1": b1, "b2": b2, "loss": loss, "gap": bb.gap_of(f, w1, b1, w2),
            "solves": bool(solves(theta, f)), "steps": used}


def _strict_job(args):
    a, kind, s = args
    import torch
    from . import blockB_landscape as bb
    torch.set_num_threads(1)
    x, y = bb.population_data()
    restarts = bb.RESTARTS if kind == "glob" else 24
    cheap = [_minimise_strict(a, s, x, y, seed=sd, steps=STRICT_SCREEN) for sd in range(restarts)]
    cheap = sorted([c for c in cheap if not bb.is_degenerate(c)], key=lambda c: c["loss"])
    full = [_minimise_strict(a, s, x, y, start=[c["w1"], c["b1"], c["b2"]]) for c in cheap[:bb.KEEP]]
    full = [c for c in full if not bb.is_degenerate(c)]
    b = min(full, key=lambda c: c["loss"])
    return {"a": a, "kind": kind, "s": s, **b}


def strict(workers=4):
    sw = pd.read_csv(RESULTS / "blockB_switches.csv")
    jobs = []
    for r in sw.itertuples():
        for kind, thr in (("glob", r.w2_glob), ("solve", r.w2_solve)):
            for s in np.arange(thr - 0.1, thr + 0.1 + 1e-9, 0.05):
                jobs.append((round(float(r.a), 2), kind, round(float(s), 4)))
    with Pool(workers) as p:
        out = p.map(_strict_job, jobs)
    d = pd.DataFrame(out)
    d.to_csv(STRICT_OUT, index=False)
    print(d.to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "strict":
    strict()
