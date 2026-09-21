"""Phase 1: decompose every failure into PLACEMENT or BIAS.

For w2 > 0, with M_I = max_{x in I} f_a(w1 x + b1) and m_O = min_{x in O} f_a(...),
a network is sign-correct on the windows iff

    w2 M_I + b2 < 0 < w2 m_O + b2

which holds iff BOTH
    placement:  G(w1,b1) = m_O - M_I > 0
    bias:       b2 in (-w2 m_O, -w2 M_I),  an interval of width w2 G

Mirror for w2 < 0 (roles of I and O swap).  |w2| does NOT appear in the
placement condition; it only sets the admissible bias interval's width.  So any
predictive power of |w2| must act through the dynamics that produce placement
and bias -- which is the reviewer's point, made testable.

Every classification is cross-checked against solves(); disagreement is a bug.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .fold1d import (INNER_MAX, OUTER_MIN, OUTER_MAX, activation, logits, solves)
from .fold1d_theorem import maximum_gap

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_DENSE = 4_001


def _windows(dtype=torch.float64):
    inner = torch.linspace(-INNER_MAX, INNER_MAX, N_DENSE, dtype=dtype)
    pos = torch.linspace(OUTER_MIN, OUTER_MAX, N_DENSE // 2, dtype=dtype)
    return inner, torch.cat([pos, -pos])


def decompose(a: float, w1: float, b1: float, w2: float, b2: float) -> dict:
    """Placement gap at this run's own (w1,b1), and the bias condition."""

    f = activation("sin_family", a)
    inner, outer = _windows()
    with torch.no_grad():
        vi = f(torch.tensor(w1, dtype=torch.float64) * inner + b1)
        vo = f(torch.tensor(w1, dtype=torch.float64) * outer + b1)
    M_I, m_O = float(vi.max()), float(vo.min())
    mi_I, M_O = float(vi.min()), float(vo.max())
    if w2 > 0:                       # outer must exceed inner
        gap = m_O - M_I
        lo, hi = -w2 * m_O, -w2 * M_I
    else:                            # mirrored: inner must exceed outer
        gap = mi_I - M_O
        lo, hi = -w2 * mi_I, -w2 * M_O
    placement_ok = gap > 0
    bias_ok = lo < b2 < hi
    return {"gap": gap, "placement_ok": placement_ok, "bias_ok": bias_ok,
            "bias_lo": lo, "bias_hi": hi, "bias_width": abs(w2) * gap}


def main() -> None:
    runs = pd.read_csv(RESULTS / "phase1_runs.csv")
    gstar = {a: maximum_gap(float(a), resolution=600) for a in sorted(runs.a.unique())}
    rows, bugs = [], 0
    for _, r in runs.iterrows():
        d = decompose(r.a, r.w1, r.b1, r.w2, r.b2)
        predicted = d["placement_ok"] and d["bias_ok"]
        if predicted != bool(r.solved):
            bugs += 1
        rows.append({**r.to_dict(), **d,
                     "rho": d["gap"] / gstar[r.a] if gstar[r.a] > 0 else np.nan,
                     "R": abs(r.w2) * gstar[r.a] / 2.0,
                     "predicted_solved": predicted,
                     "failure": ("solved" if predicted else
                                 "placement" if not d["placement_ok"] else "bias")})
    frame = pd.DataFrame(rows)
    print(f"cross-check against solves(): {bugs} disagreements of {len(frame)}")
    if bugs:
        print("  *** DISAGREEMENT IS A BUG -- investigate before continuing ***")
    stem = RESULTS / "phase1_decomposition"
    with artifact_lock(stem, "phase1 decomposition"):
        tmp = stem.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    return frame


if __name__ == "__main__":
    main()
