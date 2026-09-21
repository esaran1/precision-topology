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


# Grid spacings of the windows solves() uses.  The identity between
# (placement and bias) and sign correctness holds EXACTLY only when M_I and
# m_O are extrema over the same points solves() evaluates.
H_INNER = 2.0 * INNER_MAX / (N_DENSE - 1)
H_OUTER = (OUTER_MAX - OUTER_MIN) / (N_DENSE // 2 - 1)
H_MAX = max(H_INNER, H_OUTER)


def decompose(a: float, w1: float, b1: float, w2: float, b2: float) -> dict:
    """Placement gap at this run's own (w1,b1), and the bias condition.

    Computed on EXACTLY the grid solves() uses, so the identity is exact.
    A continuous-extrema version is returned alongside as `gap_continuous`;
    any difference between the two is a grid effect, not a bug.
    """

    f = activation("sin_family", a)
    inner, outer = _windows()
    w = torch.tensor(w1, dtype=torch.float64)
    with torch.no_grad():
        vi = f(w * inner + b1)
        vo = f(w * outer + b1)
        # continuous-extrema reference: 20x finer, labelled separately
        fi = torch.linspace(-INNER_MAX, INNER_MAX, 20 * N_DENSE, dtype=torch.float64)
        fp = torch.linspace(OUTER_MIN, OUTER_MAX, 10 * N_DENSE, dtype=torch.float64)
        fo = torch.cat([fp, -fp])
        ci, co = f(w * fi + b1), f(w * fo + b1)
    M_I, m_O = float(vi.max()), float(vo.min())
    mi_I, M_O = float(vi.min()), float(vo.max())
    if w2 > 0:                       # outer must exceed inner
        gap = m_O - M_I
        gap_c = float(co.min()) - float(ci.max())
        lo, hi = -w2 * m_O, -w2 * M_I
    else:                            # mirrored: inner must exceed outer
        gap = mi_I - M_O
        gap_c = float(ci.min()) - float(co.max())
        # w2 < 0: need w2*mi_I + b2 < 0 and w2*M_O + b2 > 0, so
        #   b2 < -w2*mi_I  and  b2 > -w2*M_O.  Dividing by a negative w2
        #   flips the order, so the endpoints are (-w2*M_O, -w2*mi_I).
        lo, hi = -w2 * M_O, -w2 * mi_I
    placement_ok = gap > 0
    bias_ok = lo < b2 < hi
    # Region-wide certificate: |N'(x)| <= |w2||w1|(1+a), so a grid margin
    # exceeding that bound times h/2 certifies correctness BETWEEN grid points.
    lip = abs(w2) * abs(w1) * (1.0 + a)
    margin_grid = min(float(-(w2 * vi + b2).max()), float((w2 * vo + b2).min()))
    certified = bool(margin_grid > lip * H_MAX / 2.0)
    return {"gap": gap, "gap_continuous": gap_c,
            "placement_ok": placement_ok, "bias_ok": bias_ok,
            "bias_lo": lo, "bias_hi": hi, "bias_width": abs(w2) * gap,
            "margin_grid": margin_grid, "lipschitz": lip,
            "certify_threshold": lip * H_MAX / 2.0, "certified": certified}


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
    # Near-misses: 0 errors on the 200-point sample, fail the dense region.
    # These are the runs a reviewer will ask about; classify each.
    NEAR = [(1.5, 132), (1.5, 145), (1.5, 178), (3.0, 13)]
    near = frame[[(round(r.a, 4), int(r.seed)) in NEAR and r.precision == "float32"
                  for _, r in frame.iterrows()]]
    if len(near):
        print("\nnear-misses (0 sample errors, fail the dense region):")
        for _, r in near.iterrows():
            print(f"  a={r.a:<5} seed={int(r.seed):3d}  {r.failure:9s} "
                  f"gap={r.gap:+.5f}  |w2|={abs(r.w2):.3f}  "
                  f"bias in ({r.bias_lo:+.4f},{r.bias_hi:+.4f}) b2={r.b2:+.4f}")
    print(f"\ncross-check against solves(): {bugs} disagreements of {len(frame)}")
    if bugs:
        print("  *** DISAGREEMENT IS A BUG -- investigate before continuing ***")
    sol = frame[frame.predicted_solved]
    if len(sol):
        print(f"region-wide certification of solved runs: "
              f"{int(sol.certified.sum())} of {len(sol)} "
              f"({100 * sol.certified.mean():.1f}%)")
    gridfx = int((frame.placement_ok != (frame.gap_continuous > 0)).sum())
    print(f"grid vs continuous placement sign differs on {gridfx} runs "
          f"(grid effect, not a bug)")
    stem = RESULTS / "phase1_decomposition"
    with artifact_lock(stem, "phase1 decomposition"):
        tmp = stem.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))
    return frame


if __name__ == "__main__":
    main()
