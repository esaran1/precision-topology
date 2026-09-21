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
SCREEN = 600    # cheap screening pass before converging the best candidates
KEEP = 8        # candidates carried from screening to full convergence
LOG2 = float(np.log(2.0))
DEGEN = 1e-4    # |loss - log2| below this is the constant predictor

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


def is_degenerate(c) -> bool:
    """The constant predictor (w1 -> 0, loss -> log 2) is a stationary point of
    the loss with no placement content: its gap is identically 0 and it wins the
    cheap screen at large |w2|.  It must never be taken as THE minimiser."""

    return abs(c["loss"] - LOG2) < DEGEN or abs(c["w1"]) < 1e-3


def best_conditional(a, w2, x, y, restarts=RESTARTS):
    """Two-stage: screen cheaply, converge the best non-degenerate candidates."""

    cheap = [minimise(a, w2, x, y, seed=s, steps=SCREEN) for s in range(restarts)]
    cheap = [c for c in cheap if not is_degenerate(c)]
    if not cheap:
        return None
    cheap.sort(key=lambda c: c["loss"])
    full = [minimise(a, w2, x, y, start=[c["w1"], c["b1"], c["b2"]], steps=STEPS)
            for c in cheap[:KEEP]]
    full = [c for c in full if not is_degenerate(c)]
    return min(full, key=lambda c: c["loss"]) if full else None


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


def bracket_then_refine(pred, lo, hi, coarse=0.5, fine=0.05):
    """Find the smallest x in [lo,hi] with pred(x) true: coarse bracket, then refine.

    pred must be monotone in x over the range (false then true), which is the
    case for every switch here: all four are first-crossing definitions.
    """

    prev = lo
    for x in np.arange(lo, hi + 1e-9, coarse):
        if pred(float(x)):
            for xf in np.arange(prev, x + 1e-9, fine):
                if pred(float(xf)):
                    return float(xf)
            return float(x)
        prev = x
    return None


def scan(a: float, x, y, w2max=11.0, step=0.05, w2min=1.5):
    """All four switch points at this a.  Coarse bracket then fine refine."""

    gs = maximum_gap(a, resolution=600)
    grid = np.arange(0.5, w2max + 1e-9, step)

    # --- global minimiser, many restarts -------------------------------
    def glob_pos(w2):
        b = best_conditional(a, w2, x, y)
        return b is not None and b["gap"] > 0
    glob = bracket_then_refine(glob_pos, w2min, w2max)
    # --- upper spinodal: follow the G<=0 branch upward ------------------
    def spin_pos(w2):
        cur = None
        for v in np.arange(w2min, w2 + 1e-9, 0.25):
            if cur is None:
                cands = [minimise(a, float(v), x, y, seed=sd, steps=SCREEN)
                         for sd in range(16)]
                cands = [c for c in cands if not is_degenerate(c) and c["gap"] <= 0]
                if not cands:
                    continue
                c = min(cands, key=lambda z: z["loss"])
                cur = minimise(a, float(v), x, y, start=[c["w1"], c["b1"], c["b2"]])
                continue
            cur = minimise(a, float(v), x, y, start=[cur["w1"], cur["b1"], cur["b2"]])
            if is_degenerate(cur):
                return True                       # branch dissolved
        return cur is None or cur["gap"] > 0
    spin = bracket_then_refine(spin_pos, w2min, w2max)

    # --- lower spinodal: follow the G>0 branch downward -----------------
    # Seed the downward branch from a MID-range |w2| where the good placement is
    # the actual minimiser.  Screening at the top of the range fails: there the
    # log-2 constant predictor dominates and the seeding falls through to an
    # unrelated basin, which produced a spurious R_fold of 0.107 at a = 1.30.
    fold = None
    seed_w2 = None
    for v in np.arange(w2min, w2max + 1e-9, 0.5):
        b = best_conditional(a, float(v), x, y, restarts=24)
        if b is not None and b["gap"] > 0:
            seed_w2, cur = float(v) + 1.0, None
            break
    if seed_w2 is None:
        return {"a": a, "gstar": gs, "w2_glob": glob,
                "R_glob": None if glob is None else glob * gs / 2,
                "w2_spin": spin, "R_spin": None if spin is None else spin * gs / 2,
                "w2_fold": None, "R_fold": None, "w2_solve": None, "R_solve": None}
    cur = best_conditional(a, seed_w2, x, y, restarts=24)
    for w2 in np.arange(seed_w2 - 0.25, w2min - 1e-9, -0.25):
        if cur is None:
            break
        nxt = minimise(a, float(w2), x, y, start=[cur["w1"], cur["b1"], cur["b2"]])
        if is_degenerate(nxt) or nxt["gap"] <= 0:
            fold = float(w2)
            break
        cur = nxt

    # --- solve point on the fold branch ---------------------------------
    def solve_pos(w2):
        b = best_conditional(a, w2, x, y, restarts=24)
        return b is not None and b["solves"]
    solve = bracket_then_refine(solve_pos, w2min, w2max)

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
