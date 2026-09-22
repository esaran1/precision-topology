"""Items #13 and the certified-G* upgrade: EXACT extrema of f_a on a window.

f_a(t) = t + a sin t, so d/dx f_a(w1 x + b1) = w1 * (1 + a cos(w1 x + b1)).
For a > 1 the interior critical points are where cos(w1 x + b1) = -1/a, i.e.

    w1 x + b1 = +- arccos(-1/a) + 2 pi k

giving x = (+- arccos(-1/a) + 2 pi k - b1) / w1.  The extrema of f_a on [lo, hi]
are therefore the ENDPOINTS plus whichever of these finitely many points land
inside.  No grid is involved, and the result is exact up to float64 evaluation of
arccos and sin.

This replaces grid sampling in two places:
  #13  headline width-1 separations -- verify the class gap exactly
  #2   the certified G* search -- so the certificate does not rest on a grid
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def f_a(t, a):
    return t + a * math.sin(t)


def critical_x(a, w1, b1, lo, hi):
    """Interior x in (lo,hi) where d/dx f_a(w1 x + b1) = 0, exactly."""

    if a <= 1.0 or w1 == 0.0:
        return []                      # monotone: no interior critical point
    c = math.acos(-1.0 / a)            # in (pi/2, pi]
    out = []
    # w1*x + b1 = +-c + 2*pi*k  ->  x = (+-c + 2*pi*k - b1)/w1
    t_lo, t_hi = sorted((w1 * lo + b1, w1 * hi + b1))
    kmin = math.floor((t_lo - c) / (2 * math.pi)) - 1
    kmax = math.ceil((t_hi + c) / (2 * math.pi)) + 1
    for k in range(kmin, kmax + 1):
        for sgn in (+1.0, -1.0):
            t = sgn * c + 2 * math.pi * k
            x = (t - b1) / w1
            if lo < x < hi:
                out.append(x)
    return sorted(out)


def extrema_on(a, w1, b1, lo, hi):
    """(min, max) of f_a(w1 x + b1) over [lo, hi], EXACT."""

    xs = [lo, hi] + critical_x(a, w1, b1, lo, hi)
    vals = [f_a(w1 * x + b1, a) for x in xs]
    return min(vals), max(vals)


def exact_gap(a, w1, b1, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    """Oriented class gap, computed from exact extrema on the three intervals.

    The outer window is +-[1.2,2.0], i.e. two intervals.
    """

    i_lo, i_hi = extrema_on(a, w1, b1, inner[0], inner[1])
    op_lo, op_hi = extrema_on(a, w1, b1, outer[0], outer[1])
    on_lo, on_hi = extrema_on(a, w1, b1, -outer[1], -outer[0])
    o_lo, o_hi = min(op_lo, on_lo), max(op_hi, on_hi)
    return max(o_lo - i_hi, i_lo - o_hi), (i_lo, i_hi, o_lo, o_hi)


def separates(a, theta, inner=(-0.8, 0.8), outer=(1.2, 2.0)):
    """Does w2*f_a(w1 x + b1) + b2 have the correct sign on BOTH windows, exactly?

    Inner must be negative, outer positive.  Uses exact extrema, so this is a
    proof on the continuum, not a sample.
    """

    w1, b1, w2, b2 = (float(v) for v in theta)
    i_lo, i_hi = extrema_on(a, w1, b1, inner[0], inner[1])
    op_lo, op_hi = extrema_on(a, w1, b1, outer[0], outer[1])
    on_lo, on_hi = extrema_on(a, w1, b1, -outer[1], -outer[0])
    o_lo, o_hi = min(op_lo, on_lo), max(op_hi, on_hi)
    # logit extremes follow from w2's sign
    if w2 > 0:
        inner_max = w2 * i_hi + b2
        outer_min = w2 * o_lo + b2
    else:
        inner_max = w2 * i_lo + b2
        outer_min = w2 * o_hi + b2
    return bool(inner_max < 0 and outer_min > 0), inner_max, outer_min


def main():
    print("=== #13: exact verification of the headline separations ===\n")
    from .fold1d import activation, solves
    import torch
    # the two figure-1 networks and the recorded solvers
    cases = [
        ("fig1 solver a=1.5 seed0", 1.5,
         (0.9437292267391273, 2.3593122836987837, -4.400602112045125, 13.150372704129833)),
        ("fig1 monotone a=1.0 seed38", 1.0,
         (-0.8161124460820061, -2.3657288524708325, 2.6410341930922247, 7.321862160553577)),
    ]
    rows = []
    for name, a, th in cases:
        ok, im, om = separates(a, th)
        f = activation("sin_family", a)
        grid = solves(torch.tensor(th, dtype=torch.float64), f)
        rows.append({"case": name, "a": a, "exact_separates": ok,
                     "grid_solves": bool(grid), "agree": ok == bool(grid),
                     "inner_max_logit": im, "outer_min_logit": om,
                     "n_crit_inner": len(critical_x(a, th[0], th[1], -0.8, 0.8)),
                     "n_crit_outer_pos": len(critical_x(a, th[0], th[1], 1.2, 2.0)),
                     "n_crit_outer_neg": len(critical_x(a, th[0], th[1], -2.0, -1.2))})
        print(f"  {name}")
        print(f"    exact: separates={ok}  inner max logit={im:+.6e}  outer min logit={om:+.6e}")
        print(f"    grid solves()={bool(grid)}   agree={ok == bool(grid)}")
        print(f"    interior critical points: inner {rows[-1]['n_crit_inner']}, "
              f"outer+ {rows[-1]['n_crit_outer_pos']}, outer- {rows[-1]['n_crit_outer_neg']}")
    pd.DataFrame(rows).to_csv(RESULTS / "exact_separation.csv", index=False)
    print("\nwritten results/exact_separation.csv")


if __name__ == "__main__":
    main()
