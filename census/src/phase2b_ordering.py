"""Phase 2b: the ordering test -- does scale precede placement or follow it?

Registered in results/phase2b_prediction.md before any checkpoint was logged.

Logs log-spaced checkpoints across the budget, PLUS a fine record once G comes
within a small margin of zero, so the first good-placement step is located
precisely rather than bracketed by a coarse grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, LR, N_PER_CLASS,
                     activation, logits, make_data, solves)
from .fold1d_theorem import maximum_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_DENSE = 4_001
A_VALUES = (1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
BUDGETS = (2_000, 4_000, 8_000, 16_000, 32_000)
SEEDS = 40
NEAR = 0.05          # once |G| < NEAR * Gstar, log every step


def _windows():
    inner = torch.linspace(-INNER_MAX, INNER_MAX, N_DENSE, dtype=torch.float64)
    pos = torch.linspace(OUTER_MIN, OUTER_MAX, N_DENSE // 2, dtype=torch.float64)
    return inner, torch.cat([pos, -pos])


INNER, OUTER = _windows()


def state(theta, f, a, gstar):
    """Placement, bias, R and rho at the current parameters."""

    w1, b1, w2, b2 = (float(v) for v in theta)
    with torch.no_grad():
        vi = f(torch.tensor(w1, dtype=torch.float64) * INNER + b1)
        vo = f(torch.tensor(w1, dtype=torch.float64) * OUTER + b1)
    M_I, m_O = float(vi.max()), float(vo.min())
    mi_I, M_O = float(vi.min()), float(vo.max())
    if w2 > 0:
        gap, lo, hi = m_O - M_I, -w2 * m_O, -w2 * M_I
    else:
        gap, lo, hi = mi_I - M_O, -w2 * M_O, -w2 * mi_I
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "gap": gap,
            "placement_ok": gap > 0, "bias_ok": lo < b2 < hi,
            "R": abs(w2) * gstar / 2.0, "rho": gap / gstar}


def run(a: float, seed: int, budget: int, gstar: float):
    f = activation("sin_family", a)
    x, y = make_data(N_PER_CLASS, seed)
    torch.manual_seed(seed)
    init = torch.empty(4).uniform_(-1.0, 1.0)          # drawn once, then cast
    theta = init.double().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=LR)

    marks = set(np.unique(np.logspace(0, np.log10(budget), 40).astype(int)))
    rows, crossing, fine = [], None, False
    for step in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(logits(theta, x.double(), f), y.double())
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())
        if not fine or crossing is None:
            s = state(theta.detach(), f, a, gstar)
            if abs(s["gap"]) < NEAR * gstar:
                fine = True                             # log every step near zero
            if crossing is None and s["placement_ok"]:
                crossing = step
                rows.append({**s, "step": step, "loss": loss_val, "crossing": True})
                fine = False
                continue
        if step in marks or fine:
            s = state(theta.detach(), f, a, gstar)
            rows.append({**s, "step": step, "loss": loss_val, "crossing": False})
    t = theta.detach()
    final = state(t, f, a, gstar)
    return rows, crossing, bool(solves(t, f)), final


def main() -> None:
    out = []
    for a in A_VALUES:
        gs = maximum_gap(a, resolution=600)
        for B in BUDGETS:
            solved = crossed = 0
            for seed in range(SEEDS):
                rows, crossing, ok, final = run(a, seed, B, gs)
                solved += ok
                crossed += crossing is not None
                for r in rows:
                    out.append({"a": a, "budget": B, "seed": seed,
                                "gstar": gs, "crossing_step": crossing,
                                "final_solved": ok, **r})
            print(f"  a={a:<5} B={B:<6} solved={solved:2d}/{SEEDS} "
                  f"crossed={crossed:2d}/{SEEDS}", flush=True)
            frame = pd.DataFrame(out)
            stem = RESULTS / "phase2b_checkpoints"
            with artifact_lock(stem, "phase2b checkpoints"):
                tmp = stem.with_suffix(".csv.tmp")
                frame.to_csv(tmp, index=False)
                tmp.replace(stem.with_suffix(".csv"))
    print("done")


if __name__ == "__main__":
    main()
