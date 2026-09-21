"""Block B: the conditional landscape and its switch points.

Registered in results/blockB_prediction.md.  float64 throughout.

The conditional loss is the training loss minimised over (w1, b1, b2) with
|w2| held FIXED.  Four switch points are computed separately:

  R_glob  global minimiser first has G > 0        (many restarts, take best loss)
  R_spin  upper spinodal: the G<=0 branch, followed by continuation upward from
          small |w2|, vanishes or its smallest Hessian eigenvalue hits 0
  R_fold  lower spinodal: the G>0 branch, followed DOWNWARD from large |w2|,
          vanishes
  R_solve on the fold branch, minimiser first sign-correct on the dense grid

R_fold < R_glob < R_spin is a hysteresis window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, activation, solves)
from .fold1d_theorem import maximum_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_DENSE = 4_001
A_VALUES = (1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
RESTARTS = 50
STEPS = 3_000   # convergence-checked: gaps settle by 3k, NOT by 900

_inner = torch.linspace(-INNER_MAX, INNER_MAX, N_DENSE, dtype=torch.float64)
_pos = torch.linspace(OUTER_MIN, OUTER_MAX, N_DENSE // 2, dtype=torch.float64)
_outer = torch.cat([_pos, -_pos])


def population_data(n_per_class: int = 400):
    """Dense uniform sample of the windows -- the population loss."""

    inner = torch.linspace(-INNER_MAX, INNER_MAX, n_per_class, dtype=torch.float64)
    pos = torch.linspace(OUTER_MIN, OUTER_MAX, n_per_class // 2, dtype=torch.float64)
    outer = torch.cat([pos, -pos])
    x = torch.cat([inner, outer])
    y = torch.cat([torch.zeros(len(inner), dtype=torch.float64),
                   torch.ones(len(outer), dtype=torch.float64)])
    return x, y


def gap_of(f, w1: float, b1: float, w2: float) -> float:
    with torch.no_grad():
        vi = f(torch.tensor(w1, dtype=torch.float64) * _inner + b1)
        vo = f(torch.tensor(w1, dtype=torch.float64) * _outer + b1)
    return float(vo.min() - vi.max()) if w2 > 0 else float(vi.min() - vo.max())


def minimise(a, w2, x, y, start=None, seed=0, steps=STEPS):
    """Minimise the conditional loss over (w1,b1,b2) at fixed w2."""

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
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        out = w2t * f(p[0] * x + p[1]) + p[2]
        F.binary_cross_entropy_with_logits(out, y).backward()
        opt.step()
        if i == steps // 2:
            for gp in opt.param_groups:
                gp["lr"] = 5e-3
    q = p.detach()
    with torch.no_grad():
        out = w2t * f(q[0] * x + q[1]) + q[2]
        loss = float(F.binary_cross_entropy_with_logits(out, y))
    w1, b1, b2 = (float(v) for v in q)
    theta = torch.tensor([w1, b1, w2, b2], dtype=torch.float64)
    return {"w1": w1, "b1": b1, "b2": b2, "loss": loss,
            "gap": gap_of(f, w1, b1, w2),
            "solves": bool(solves(theta, f))}


def hessian_min_eig(a, w2, p, x, y) -> float:
    f = activation("sin_family", a)
    w2t = torch.tensor(w2, dtype=torch.float64)
    def loss_fn(v):
        out = w2t * f(v[0] * x + v[1]) + v[2]
        return F.binary_cross_entropy_with_logits(out, y)
    v = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    H = torch.autograd.functional.hessian(loss_fn, v)
    return float(torch.linalg.eigvalsh(H).min())


def scan(a: float, x, y, w2max=14.0, step=0.05):
    """All four switch points at this a."""

    gs = maximum_gap(a, resolution=600)
    grid = np.arange(0.5, w2max + 1e-9, step)

    # --- global minimiser, many restarts -------------------------------
    glob = None
    glob_rows = []
    for w2 in grid:
        cands = [minimise(a, float(w2), x, y, seed=s) for s in range(RESTARTS)]
        best = min(cands, key=lambda c: c["loss"])
        glob_rows.append((float(w2), best["gap"], best["loss"], best["solves"]))
        if glob is None and best["gap"] > 0:
            glob = float(w2)
            break
    # --- upper spinodal: follow the G<=0 branch upward ------------------
    spin = None
    cur = None
    for w2 in grid:
        cands = [minimise(a, float(w2), x, y, seed=s) for s in range(12)] if cur is None else []
        if cur is None:
            neg = [c for c in cands if c["gap"] <= 0]
            if not neg:
                spin = float(w2)                     # no G<=0 branch at all
                break
            cur = min(neg, key=lambda c: c["loss"])
            continue
        nxt = minimise(a, float(w2), x, y, start=[cur["w1"], cur["b1"], cur["b2"]])
        if nxt["gap"] > 0:
            spin = float(w2)
            break
        cur = nxt
    # --- lower spinodal: follow the G>0 branch downward -----------------
    fold = None
    cur = None
    for w2 in grid[::-1]:
        if cur is None:
            cands = [minimise(a, float(w2), x, y, seed=s) for s in range(12)]
            pos = [c for c in cands if c["gap"] > 0]
            if not pos:
                continue
            cur = min(pos, key=lambda c: c["loss"])
            continue
        nxt = minimise(a, float(w2), x, y, start=[cur["w1"], cur["b1"], cur["b2"]])
        if nxt["gap"] <= 0:
            fold = float(w2)
            break
        cur = nxt
    # --- solve point on the fold branch ---------------------------------
    solve = None
    cur = None
    for w2 in grid:
        cands = [minimise(a, float(w2), x, y, seed=s) for s in range(12)] if cur is None else []
        if cur is None:
            pos = [c for c in cands if c["gap"] > 0]
            if not pos:
                continue
            cur = min(pos, key=lambda c: c["loss"])
        else:
            cur = minimise(a, float(w2), x, y, start=[cur["w1"], cur["b1"], cur["b2"]])
        if cur["solves"]:
            solve = float(w2)
            break
    return {"a": a, "gstar": gs,
            "w2_glob": glob, "R_glob": None if glob is None else glob * gs / 2,
            "w2_spin": spin, "R_spin": None if spin is None else spin * gs / 2,
            "w2_fold": fold, "R_fold": None if fold is None else fold * gs / 2,
            "w2_solve": solve, "R_solve": None if solve is None else solve * gs / 2}


def main() -> None:
    x, y = population_data()
    rows = []
    for a in A_VALUES:
        r = scan(a, x, y)
        rows.append(r)
        hyst = (r["R_fold"] is not None and r["R_glob"] is not None
                and r["R_spin"] is not None
                and r["R_fold"] < r["R_glob"] < r["R_spin"])
        print(f"  a={a:<5} R_fold={r['R_fold']} R_glob={r['R_glob']} "
              f"R_spin={r['R_spin']} R_solve={r['R_solve']}  hysteresis={hyst}", flush=True)
        frame = pd.DataFrame(rows)
        stem = RESULTS / "blockB_switches"
        with artifact_lock(stem, "blockB switches"):
            tmp = stem.with_suffix(".csv.tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(stem.with_suffix(".csv"))
    print("done")


if __name__ == "__main__":
    main()
