"""Block E: intervene on |w2| and see whether placement follows.

Registered in results/blockF_lag_prediction.md (reversibility pair) and revised
in results/blockE_redesign.md after a pilot, which is reported there in full.

Key pilot finding, which sets the design: held at fixed |w2| from a COLD start,
placement is never achieved at any R -- 0 of 9 at R/R_glob = 0.85, 1.15, 1.50
over 20,000 steps.  Started from a placed configuration and held, placement is
RETAINED above R_glob and LOST below it (0/10 at 0.90 R_glob, 10/10 at 1.15).
So the switch governs STABILITY, not attainability, and the primary test is the
reversibility pair hold_low / hold_high.

Mechanics common to every arm: same initialisation per seed (drawn in one dtype
and cast, matching phase1_relog and Block A, so these are the same networks whose
crossings Block A measured); |w2| projected back to its held value after every
step; (w1,b1) and sign(w2) never touched; b2 scaled with w2 so the decision
threshold is carried along rather than destroyed, which would confound a
placement test with a bias failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .fold1d import activation, make_data, solves
from .fold1d_theorem import maximum_gap
from .r_variable import oriented_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"
A = 1.30
STEPS = 12_000
LR = 1e-2
N_SEEDS = 40
CHECK_EVERY = 50
HOLD_AFTER_PLACED = 200   # steps to let the placed run settle before intervening
LOW, HIGH = 0.85, 1.15    # multiples of R_glob for the two sides

ARMS = ("control", "noise_floor", "hold_low", "hold_high",
        "cold_low", "cold_high", "jump")
PROSPECTIVE = ("hold_low", "hold_high")   # the only arms scored as registered


def w2_for_R(R, gstar, sign):
    return sign * 2.0 * R / gstar


def _scale_to(th, R, gstar):
    """Set |w2| to the value giving margin capacity R; carry b2 with it."""

    with torch.no_grad():
        w2 = float(th[2])
        s = np.sign(w2) if w2 != 0 else 1.0
        new = w2_for_R(R, gstar, s)
        if w2 != 0:
            th[3] *= new / w2
        th[2] = torch.tensor(new, dtype=torch.float64)
        return abs(float(th[2]))


def run(a, seed, arm, R_glob, gstar, steps=STEPS):
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)

    hold = None
    applied = False
    placed_at = None          # first step at which gap > 0
    intervened_at = None
    lost_at = None            # first step after the intervention at which gap <= 0
    gap_at_intervention = None

    if arm == "cold_low":
        hold = _scale_to(th, LOW * R_glob, gstar); applied = True; intervened_at = 0
    elif arm == "cold_high":
        hold = _scale_to(th, HIGH * R_glob, gstar); applied = True; intervened_at = 0
    elif arm == "jump":
        _scale_to(th, 1.05 * R_glob, gstar); applied = True; intervened_at = 0

    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        out = th[2] * f(th[0] * x + th[1]) + th[3]
        F.binary_cross_entropy_with_logits(out, y).backward()
        opt.step()
        if hold is not None:
            with torch.no_grad():
                v = float(th[2])
                if v != 0:
                    th[2] = torch.tensor(np.sign(v) * hold, dtype=torch.float64)

        if i % CHECK_EVERY == 0 or i == steps - 1:
            with torch.no_grad():
                w1, b1, w2, b2 = (float(v) for v in th)
            gp = oriented_gap(f, w1, b1)
            if placed_at is None and gp > 0:
                placed_at = i
            if intervened_at is not None and i > intervened_at and gp <= 0 and lost_at is None:
                lost_at = i
            # the hold/null arms intervene once the run has placed and settled
            if (not applied and arm in ("hold_low", "hold_high", "noise_floor")
                    and placed_at is not None and i >= placed_at + HOLD_AFTER_PLACED):
                gap_at_intervention = gp
                if arm == "noise_floor":
                    with torch.no_grad():
                        th[2] *= 1.0 + 1e-6
                        th[3] *= 1.0 + 1e-6
                else:
                    hold = _scale_to(th, (LOW if arm == "hold_low" else HIGH) * R_glob,
                                     gstar)
                applied = True
                intervened_at = i
                lost_at = None

    with torch.no_grad():
        w1, b1, w2, b2 = (float(v) for v in th)
    gp = oriented_gap(f, w1, b1)
    return {"a": a, "seed": seed, "arm": arm,
            "placed_at": placed_at, "intervened_at": intervened_at,
            "gap_at_intervention": gap_at_intervention,
            "lost_at": lost_at,
            "placed_final": bool(gp > 0), "gap_final": gp,
            "kept": (None if arm in ("cold_low", "cold_high", "jump")
                     else (bool(gp > 0) if intervened_at is not None else None)),
            "solved_final": bool(solves(
                torch.tensor([w1, b1, w2, b2], dtype=torch.float64), f)),
            "R_final": abs(w2) * gstar / 2.0}


def main():
    gstar = maximum_gap(A, resolution=600)
    sw = pd.read_csv(RESULTS / "blockB_switches.csv")
    R_glob = float(sw[sw.a == A].R_glob.iloc[0])
    print(f"a={A}  Ghat={gstar:.6f}  R_glob={R_glob:.5f}  "
          f"low={LOW * R_glob:.5f}  high={HIGH * R_glob:.5f}", flush=True)
    rows = []
    for arm in ARMS:
        for s in range(N_SEEDS):
            rows.append(run(A, s, arm, R_glob, gstar))
        d = pd.DataFrame([r for r in rows if r["arm"] == arm])
        n_int = int(d.intervened_at.notna().sum())
        kept = d[d.kept.notna()]
        tag = " [PROSPECTIVE]" if arm in PROSPECTIVE else ""
        print(f"  {arm:11s} placed {d.placed_final.mean():.3f}  solved "
              f"{d.solved_final.mean():.3f}  intervened {n_int}/{len(d)}  "
              f"kept {kept.kept.sum() if len(kept) else 0}/{len(kept)}  "
              f"R_final {d.R_final.median():.4f}{tag}", flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "blockE_intervene"
    with artifact_lock(stem, "blockE intervene"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
    print("\nwritten results/blockE_intervene.csv")


if __name__ == "__main__":
    main()
