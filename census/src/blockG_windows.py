"""Block G: does the training-free machinery transfer to task windows it was
never calibrated on?

Registered in results/blockG_prediction.md.  float64 throughout.

Every threshold in this paper is computed on I = [-0.8,0.8], O = +-[1.2,2.0].
Nothing in the account is specific to them: Ghat is a supremum of a class gap
over placements, and R = |w2| Ghat/2 a margin capacity, both defined for any
window pair.  This module recomputes Ghat, R_glob and R_solve on four modified
window pairs and measures crossings there, so the predictions are made with no
training and tested against training.

The window bounds are module-level constants in fold1d, so every primitive is
re-expressed here as a function of an explicit Window rather than by mutating
globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .blockB_landscape import DEGEN, KEEP, LOG2, RESTARTS, SCREEN, STEPS, bracket_then_refine

RESULTS = Path(__file__).resolve().parents[1] / "results"
LR = 1e-2
N_SEEDS = 40
BUDGET = 12_000
CHECK_EVERY = 50
GRID_STEP = 0.01     # the a=1.60 audit showed 0.05 is too coarse


@dataclass(frozen=True)
class Window:
    tag: str
    i_lo: float
    i_hi: float
    o_lo: float
    o_hi: float

    def dense(self, n=4_001):
        inner = torch.linspace(self.i_lo, self.i_hi, n, dtype=torch.float64)
        pos = torch.linspace(self.o_lo, self.o_hi, n // 2, dtype=torch.float64)
        return inner, torch.cat([pos, -pos])

    def data(self, n_per_class, seed):
        rng = np.random.default_rng(seed)
        inner = rng.uniform(self.i_lo, self.i_hi, n_per_class)
        mag = rng.uniform(self.o_lo, self.o_hi, n_per_class)
        outer = mag * rng.choice([-1.0, 1.0], n_per_class)
        x = np.concatenate([inner, outer])
        y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
        return (torch.tensor(x, dtype=torch.float64),
                torch.tensor(y, dtype=torch.float64))

    def population(self, n=400):
        inner = torch.linspace(self.i_lo, self.i_hi, n, dtype=torch.float64)
        pos = torch.linspace(self.o_lo, self.o_hi, n // 2, dtype=torch.float64)
        outer = torch.cat([pos, -pos])
        x = torch.cat([inner, outer])
        y = torch.cat([torch.zeros(len(inner), dtype=torch.float64),
                       torch.ones(len(outer), dtype=torch.float64)])
        return x, y


WINDOWS = (
    Window("base",          -0.8,  0.8, 1.2, 2.0),
    Window("G1_wide_gap",   -0.6,  0.6, 1.4, 2.0),
    Window("G2_narrow_gap", -0.95, 0.95, 1.05, 2.0),
    Window("G3_far_outer",  -0.8,  0.8, 1.2, 3.0),
    Window("G4_shifted",    -0.4,  1.2, 1.6, 2.4),
)


def f_a(a):
    return lambda v: v + a * torch.sin(v)


def oriented_gap(win, a, w1, b1, cache={}):
    key = (win.tag, 4001)
    if key not in cache:
        cache[key] = win.dense()
    inner, outer = cache[key]
    f = f_a(a)
    with torch.no_grad():
        w = torch.tensor(w1, dtype=torch.float64)
        vi, vo = f(w * inner + b1), f(w * outer + b1)
    return max(float(vo.min() - vi.max()), float(vi.min() - vo.max()))


def ghat(win, a, n_w1=1200, n_b1=1200, w1_max=6.0, b1_max=10.0):
    """Supremum of the oriented class gap over placements, on THIS window.

    Vectorised over b1: for each w1, evaluate all b1 at once.  f_a(w1 x + b1)
    is not f_a(w1 x) + b1, so b1 must enter inside f and cannot be hoisted.
    """

    inner, outer = win.dense(1501)
    best, arg = -np.inf, None
    b1s = torch.linspace(-b1_max, b1_max, n_b1, dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        for w1 in np.linspace(-w1_max, w1_max, n_w1):
            if abs(w1) < 1e-9:
                continue
            w = torch.tensor(w1, dtype=torch.float64)
            vi = (w * inner).unsqueeze(0) + b1s          # (n_b1, n_inner)
            vo = (w * outer).unsqueeze(0) + b1s
            vi = vi + a * torch.sin(vi)
            vo = vo + a * torch.sin(vo)
            plus = vo.min(dim=1).values - vi.max(dim=1).values
            minus = vi.min(dim=1).values - vo.max(dim=1).values
            g = torch.maximum(plus, minus)
            j = int(g.argmax())
            if float(g[j]) > best:
                best, arg = float(g[j]), (float(w1), float(b1s[j, 0]))
    return best, arg


def solves_win(win, a, th):
    inner, outer = win.dense()
    f = f_a(a)
    w1, b1, w2, b2 = (torch.tensor(float(v), dtype=torch.float64) for v in th)
    with torch.no_grad():
        return bool(((w2 * f(w1 * inner + b1) + b2) < 0).all()
                    and ((w2 * f(w1 * outer + b1) + b2) > 0).all())


def is_degenerate(c):
    return abs(c["loss"] - LOG2) < DEGEN or abs(c["w1"]) < 1e-3


def minimise(win, a, w2, x, y, start=None, seed=0, steps=STEPS):
    f = f_a(a)
    if start is None:
        g = torch.Generator().manual_seed(seed)
        p = torch.stack([torch.rand(1, generator=g).double()[0] * 4 - 2,
                         torch.rand(1, generator=g).double()[0] * 16 - 8,
                         torch.rand(1, generator=g).double()[0] * 2 * abs(w2) - abs(w2)])
    else:
        p = torch.tensor(start, dtype=torch.float64)
    p = p.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=3e-2)
    w2t = torch.tensor(w2, dtype=torch.float64)
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(w2t * f(p[0] * x + p[1]) + p[2], y).backward()
        opt.step()
        if i == steps // 2:
            for gp in opt.param_groups:
                gp["lr"] = 5e-3
    q = p.detach()
    with torch.no_grad():
        loss = float(F.binary_cross_entropy_with_logits(
            w2t * f(q[0] * x + q[1]) + q[2], y))
    w1, b1, b2 = (float(v) for v in q)
    return {"w1": w1, "b1": b1, "b2": b2, "loss": loss,
            "gap": oriented_gap(win, a, w1, b1),
            "solves": solves_win(win, a, [w1, b1, w2, b2])}


def best_conditional(win, a, w2, x, y, restarts=RESTARTS):
    cheap = [minimise(win, a, w2, x, y, seed=s, steps=SCREEN) for s in range(restarts)]
    cheap = [c for c in cheap if not is_degenerate(c)]
    if not cheap:
        return None
    cheap.sort(key=lambda c: c["loss"])
    full = [minimise(win, a, w2, x, y, start=[c["w1"], c["b1"], c["b2"]], steps=STEPS)
            for c in cheap[:KEEP]]
    full = [c for c in full if not is_degenerate(c)]
    return min(full, key=lambda c: c["loss"]) if full else None


def switches(win, a, gs, w2min=0.5, w2max=14.0):
    x, y = win.population()

    def glob_pos(w2):
        b = best_conditional(win, a, w2, x, y)
        return b is not None and b["gap"] > 0

    def solve_pos(w2):
        b = best_conditional(win, a, w2, x, y, restarts=24)
        return b is not None and b["solves"]

    glob = bracket_then_refine(glob_pos, w2min, w2max, coarse=0.5, fine=GRID_STEP)
    solve = bracket_then_refine(solve_pos, w2min, w2max, coarse=0.5, fine=GRID_STEP)
    return {"w2_glob": glob, "R_glob": None if glob is None else glob * gs / 2,
            "w2_solve": solve, "R_solve": None if solve is None else solve * gs / 2}


def train(win, a, seed, gs, budget=BUDGET):
    f = f_a(a)
    x, y = win.data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    cross_step = cross_R = cross_w2 = None
    for i in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
        if i % CHECK_EVERY == 0 and cross_step is None:
            with torch.no_grad():
                w1, b1, w2, b2 = (float(v) for v in th)
            if oriented_gap(win, a, w1, b1) > 0:
                cross_step, cross_w2 = i, abs(w2)
                cross_R = abs(w2) * gs / 2
    with torch.no_grad():
        w1, b1, w2, b2 = (float(v) for v in th)
    return {"window": win.tag, "a": a, "seed": seed,
            "cross_step": cross_step, "cross_R": cross_R, "cross_w2": cross_w2,
            "final_placed": oriented_gap(win, a, w1, b1) > 0,
            "final_solved": solves_win(win, a, [w1, b1, w2, b2]),
            "R_final": abs(w2) * gs / 2}


def main():
    print("=== Ghat on each window (training-free) ===", flush=True)
    gh = {}
    for win in WINDOWS:
        for a in (1.30, 1.50):
            g, arg = ghat(win, a)
            gh[(win.tag, a)] = g
            print(f"  {win.tag:15s} a={a}: Ghat={g:.6f} at w1={arg[0]:.4f} b1={arg[1]:.4f}",
                  flush=True)

    print("\n=== switch points on each window (training-free, grid 0.01) ===", flush=True)
    srows = []
    for win in WINDOWS:
        for a in (1.30, 1.50):
            s = switches(win, a, gh[(win.tag, a)])
            s.update({"window": win.tag, "a": a, "gstar": gh[(win.tag, a)]})
            srows.append(s)
            print(f"  {win.tag:15s} a={a}: R_glob={s['R_glob']} R_solve={s['R_solve']}",
                  flush=True)
    pd.DataFrame(srows).to_csv(RESULTS / "blockG_switches.csv", index=False)

    print("\n=== measured crossings ===", flush=True)
    rows = []
    for win in WINDOWS:
        for a in (1.30, 1.50):
            for s in range(N_SEEDS):
                rows.append(train(win, a, s, gh[(win.tag, a)]))
            d = pd.DataFrame([r for r in rows if r["window"] == win.tag and r["a"] == a])
            c = d[d.cross_R.notna()]
            print(f"  {win.tag:15s} a={a}: crossed {len(c)}/{len(d)}  "
                  f"solved {d.final_solved.mean():.3f}  "
                  f"median cross R {c.cross_R.median() if len(c) else float('nan'):.4f}  "
                  f"median cross |w2| {c.cross_w2.median() if len(c) else float('nan'):.4f}",
                  flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "blockG_crossings"
    with artifact_lock(stem, "blockG crossings"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
    print("\nwritten blockG_switches.csv and blockG_crossings.csv")


if __name__ == "__main__":
    main()
