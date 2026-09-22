"""Block B run directly on the scaling-limit activation h(sigma) = -sigma + sigma^3/6.

Registered in results/scaling_limit_prediction.md (S-2, S-3).

The frozen Block B procedure (src/blockB_landscape.py, sha256 9f1b10741d8bf48c)
is reused VERBATIM for its optimiser settings, screening, degeneracy test and
bracket-then-refine search.  The only substitutions are:

  activation      f_a(t) = t + a sin t        ->  h(sigma) = -sigma + sigma^3/6
  output weight   w2                          ->  w2_inf = w2 * eps^{3/2}
  gap normaliser  Ghat(a)                     ->  K = sup gap of h over placements

so R_inf = |w2_inf| * K / 2 is the same quantity as R, expressed in the limit.

The task windows I = [-0.8,0.8], O = +-[1.2,2.0] are UNCHANGED: they are fixed
in x, and the placement (u, v) = (w1/sqrt(eps), (b1-pi)/sqrt(eps)) absorbs the
horizontal rescaling.  This is exactly why the invariance is only asymptotic.

The degeneracy test needs care: for h the constant predictor is still w1 -> 0
with loss -> log 2, so the same test applies unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .blockB_landscape import (DEGEN, KEEP, LOG2, RESTARTS, SCREEN, STEPS,
                               bracket_then_refine, population_data)
from .fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_DENSE = 4_001

_inner = torch.linspace(-INNER_MAX, INNER_MAX, N_DENSE, dtype=torch.float64)
_pos = torch.linspace(OUTER_MIN, OUTER_MAX, N_DENSE // 2, dtype=torch.float64)
_outer = torch.cat([_pos, -_pos])


def h(z):
    return -z + z ** 3 / 6.0


def K_constant(n_u=1200, n_v=800, umax=6.0, vmax=6.0):
    """sup over affine placements of h's class gap on the fixed windows."""

    I = np.linspace(-INNER_MAX, INNER_MAX, 2001)
    P = np.linspace(OUTER_MIN, OUTER_MAX, 1000)
    O = np.concatenate([P, -P])
    best, arg = -np.inf, None
    for u in np.linspace(0.02, umax, n_u):
        for v in np.linspace(-vmax, vmax, n_v):
            hi, ho = h(u * I + v), h(u * O + v)
            g = max(ho.min() - hi.max(), hi.min() - ho.max())
            if g > best:
                best, arg = g, (float(u), float(v))
    return float(best), arg


def gap_of(u, v, w2):
    with torch.no_grad():
        vi = h(u * _inner + v)
        vo = h(u * _outer + v)
    return float(vo.min() - vi.max()) if w2 > 0 else float(vi.min() - vo.max())


def solves_h(u, v, w2, b2):
    with torch.no_grad():
        ni = w2 * h(u * _inner + v) + b2
        no = w2 * h(u * _outer + v) + b2
    return bool((ni < 0).all() and (no > 0).all())


def is_degenerate(c):
    return abs(c["loss"] - LOG2) < DEGEN or abs(c["w1"]) < 1e-3


def minimise(w2, x, y, start=None, seed=0, steps=STEPS):
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
        out = w2t * h(p[0] * x + p[1]) + p[2]
        F.binary_cross_entropy_with_logits(out, y).backward()
        opt.step()
        if i == steps // 2:
            for gp in opt.param_groups:
                gp["lr"] = 5e-3
    q = p.detach()
    with torch.no_grad():
        out = w2t * h(q[0] * x + q[1]) + q[2]
        loss = float(F.binary_cross_entropy_with_logits(out, y))
    u, v, b2 = (float(z) for z in q)
    return {"w1": u, "b1": v, "b2": b2, "loss": loss,
            "gap": gap_of(u, v, w2), "solves": solves_h(u, v, w2, b2)}


def best_conditional(w2, x, y, restarts=RESTARTS):
    cheap = [minimise(w2, x, y, seed=s, steps=SCREEN) for s in range(restarts)]
    cheap = [c for c in cheap if not is_degenerate(c)]
    if not cheap:
        return None
    cheap.sort(key=lambda c: c["loss"])
    full = [minimise(w2, x, y, start=[c["w1"], c["b1"], c["b2"]], steps=STEPS)
            for c in cheap[:KEEP]]
    full = [c for c in full if not is_degenerate(c)]
    return min(full, key=lambda c: c["loss"]) if full else None


def scan(K, x, y, w2max=11.0, step=0.05, w2min=0.5):
    """Same four switch points, in scaling-limit units w2_inf."""

    def glob_pos(w2):
        b = best_conditional(w2, x, y)
        return b is not None and b["gap"] > 0
    glob = bracket_then_refine(glob_pos, w2min, w2max)

    def spin_pos(w2):
        cur = None
        for vv in np.arange(w2min, w2 + 1e-9, 0.25):
            if cur is None:
                cands = [minimise(float(vv), x, y, seed=sd, steps=SCREEN)
                         for sd in range(16)]
                cands = [c for c in cands if not is_degenerate(c) and c["gap"] <= 0]
                if not cands:
                    continue
                c = min(cands, key=lambda z: z["loss"])
                cur = minimise(float(vv), x, y, start=[c["w1"], c["b1"], c["b2"]])
                continue
            cur = minimise(float(vv), x, y, start=[cur["w1"], cur["b1"], cur["b2"]])
            if is_degenerate(cur):
                return True
        return cur is None or cur["gap"] > 0
    spin = bracket_then_refine(spin_pos, w2min, w2max)

    def solve_pos(w2):
        b = best_conditional(w2, x, y, restarts=24)
        return b is not None and b["solves"]
    solve = bracket_then_refine(solve_pos, w2min, w2max)

    return {"K": K,
            "w2_glob": glob, "R_glob": None if glob is None else glob * K / 2,
            "w2_spin": spin, "R_spin": None if spin is None else spin * K / 2,
            "w2_solve": solve, "R_solve": None if solve is None else solve * K / 2}


def main():
    K, arg = K_constant()
    print(f"K = {K:.6f} at (u, v) = ({arg[0]:.4f}, {arg[1]:.4f})", flush=True)
    print(f"kappa_0 = K / (4sqrt2/3) = {K / (4 * np.sqrt(2) / 3):.6f}", flush=True)
    x, y = population_data()
    row = scan(K, x, y)
    for k, vv in row.items():
        print(f"  {k} = {vv}", flush=True)
    pd.DataFrame([row]).to_csv(RESULTS / "scaling_limit_switches.csv", index=False)


if __name__ == "__main__":
    main()
