"""Block E: intervene on |w2| and see whether placement follows.

Registered in results/blockF_lag_prediction.md, section "Block E, reduced arm
set -- reversibility registered".  The reduced arm set is used because the
no-hysteresis falsifier was triggered: R_fold ~ R_glob ~ R_spin to within a grid
step at a = 1.30 and (after refinement to step 0.01) at a = 1.60 as well, so
there is no window in which a freeze-mid arm is defined.

Every arm trains the SAME run -- same seed, same data, same optimiser state
schedule -- and differs only in what is done to (w2, b2) at the intervention
step.  |w2| is rescaled and then HELD by projecting it back after every step;
(w1, b1) and the sign of w2 are never touched.  b2 is scaled with w2 so the
decision threshold is carried along rather than destroyed, which would confound
a placement test with a bias failure.

Arms:
  control        nothing done
  null           (w2,b2) scaled by 1+1e-6 and held -- the noise floor
  freeze_low     |w2| set so R = 0.85 * R_glob, held        -> placement never achieved
  freeze_high    |w2| set so R = 1.15 * R_glob, held        -> placement achieved
  jump           |w2| set so R = 1.05 * R_glob at step 0    -> placement early
  down_hold_low  run until placed, then R = 0.85 * R_glob   -> placement LOST
  down_hold_high run until placed, then R = 1.15 * R_glob   -> placement KEPT

The last two are the reversibility pair and the primary result.
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
STEPS = 8_000
LR = 1e-2
N_SEEDS = 40
INTERVENE_AT = 400        # early: before any run has placed (crossings are >= 1860)
PLACED_HOLD = 200         # for the down-hold arms: steps to wait after placement


def r_of(w2: float, gstar: float) -> float:
    return abs(w2) * gstar / 2.0


def w2_for_R(R: float, gstar: float, sign: float) -> float:
    return sign * 2.0 * R / gstar


def run(a, seed, arm, R_glob, gstar, steps=STEPS):
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    x, y = x.double(), y.double()
    # Same initialisation as phase1_relog / Block A: draw in ONE dtype and cast,
    # never per-dtype, so these seeds are the SAME networks as the runs whose
    # crossings Block A measured.
    torch.manual_seed(seed)
    init = torch.empty(4).uniform_(-1.0, 1.0)
    th = init.double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)

    hold = None                  # |w2| value to project back to, if any
    applied = False
    placed_at = None
    rows = []
    for i in range(steps):
        # --- interventions ------------------------------------------------
        if not applied and arm in ("null", "freeze_low", "freeze_high") and i == INTERVENE_AT:
            with torch.no_grad():
                w2 = float(th[2]); s = np.sign(w2) or 1.0
                if arm == "null":
                    scale = 1.0 + 1e-6
                    th[2] *= scale; th[3] *= scale
                else:
                    tgt = 0.85 * R_glob if arm == "freeze_low" else 1.15 * R_glob
                    new = w2_for_R(tgt, gstar, s)
                    scale = new / w2 if w2 != 0 else 1.0
                    th[2] = new; th[3] *= scale
                hold = abs(float(th[2]))
            applied = True
        if not applied and arm == "jump" and i == 0:
            with torch.no_grad():
                w2 = float(th[2]); s = np.sign(w2) or 1.0
                new = w2_for_R(1.05 * R_glob, gstar, s)
                scale = new / w2 if w2 != 0 else 1.0
                th[2] = new; th[3] *= scale
            applied = True
        if (not applied and arm in ("down_hold_low", "down_hold_high")
                and placed_at is not None and i >= placed_at + PLACED_HOLD):
            with torch.no_grad():
                w2 = float(th[2]); s = np.sign(w2) or 1.0
                tgt = 0.85 * R_glob if arm == "down_hold_low" else 1.15 * R_glob
                new = w2_for_R(tgt, gstar, s)
                scale = new / w2 if w2 != 0 else 1.0
                th[2] = new; th[3] *= scale
                hold = abs(float(th[2]))
            applied = True

        opt.zero_grad(set_to_none=True)
        out = th[2] * f(th[0] * x + th[1]) + th[3]
        F.binary_cross_entropy_with_logits(out, y).backward()
        opt.step()
        if hold is not None:                        # project |w2| back
            with torch.no_grad():
                w2 = float(th[2])
                if w2 != 0:
                    th[2] = torch.tensor(np.sign(w2) * hold, dtype=torch.float64)

        if i % 50 == 0 or i == steps - 1:
            with torch.no_grad():
                w1, b1, w2, b2 = (float(v) for v in th)
            gp = oriented_gap(f, w1, b1)
            if placed_at is None and gp > 0:
                placed_at = i
            rows.append({"step": i, "w1": w1, "b1": b1, "w2": w2, "b2": b2,
                         "gap": gp, "R": r_of(w2, gstar), "placed": gp > 0})
    with torch.no_grad():
        w1, b1, w2, b2 = (float(v) for v in th)
    gp = oriented_gap(f, w1, b1)
    frame = pd.DataFrame(rows)
    # placement at the END, and whether it was ever lost after the intervention
    post = frame[frame.step >= INTERVENE_AT] if arm != "jump" else frame
    return {"a": a, "seed": seed, "arm": arm,
            "placed_final": bool(gp > 0),
            "solved_final": bool(solves(torch.tensor([w1, b1, w2, b2], dtype=torch.float64), f)),
            "placed_ever": bool(frame.placed.any()),
            "placed_at": placed_at,
            "R_final": r_of(w2, gstar),
            "frac_placed_after": float(post.placed.mean()) if len(post) else np.nan,
            "gap_final": gp}


def main():
    gstar = maximum_gap(A, resolution=600)
    sw = pd.read_csv(RESULTS / "blockB_switches.csv")
    R_glob = float(sw[sw.a == A].R_glob.iloc[0])
    print(f"a={A}  Ghat={gstar:.6f}  R_glob={R_glob:.5f}", flush=True)
    arms = ("control", "null", "freeze_low", "freeze_high", "jump",
            "down_hold_low", "down_hold_high")
    rows = []
    for arm in arms:
        for s in range(N_SEEDS):
            rows.append(run(A, s, arm, R_glob, gstar))
        d = pd.DataFrame([r for r in rows if r["arm"] == arm])
        print(f"  {arm:15s} placed_final {d.placed_final.mean():.3f}  "
              f"solved {d.solved_final.mean():.3f}  "
              f"ever_placed {d.placed_ever.mean():.3f}  "
              f"R_final {d.R_final.median():.4f}", flush=True)
    frame = pd.DataFrame(rows)
    stem = RESULTS / "blockE_intervene"
    with artifact_lock(stem, "blockE intervene"):
        frame.to_csv(stem.with_suffix(".csv"), index=False)
    print("\nwritten results/blockE_intervene.csv")


if __name__ == "__main__":
    main()
