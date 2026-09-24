"""Global certificate for Ĝ(a) by branch and bound, replacing the zoom certificate.

Ĝ(a) = sup over placements (w1, b1) of the oriented class gap
G = max(min_O φ - max_I φ, min_I φ - max_O φ),  φ(x) = f_a(w1 x + b1),
I = [-0.8, 0.8], O = ±[1.2, 2.0], f_a(t) = t + a sin t.

WHY: kappa_certify.certify_exact refines only a ±12h box around the coarse
argmax, so its upper end Ghat_hi is not a global bound, and its slack
(1+a)·2·h/2 under-states the Lipschitz step (below).  Its LOWER end is a value
attained at a real placement, hence a valid lower bound; R uses that end.

COMPACT DOMAIN (analytic).  f_a(u) - f_a(t) >= (u - t) - 2a for u >= t.
For w1 > 0: min_O φ <= φ(-2.0), max_I φ >= φ(0.8), and φ(0.8) - φ(-2.0) >=
2.8 w1 - 2a, so the first orientation is < 0 once w1 > a/1.4; likewise
min_I φ <= φ(-0.8), max_O φ >= φ(2.0) bounds the second.  G is invariant under
b1 -> b1 + 2π (f_a(t+2π) = f_a(t) + 2π shifts φ by a constant) and under
(w1, b1) -> (-w1, -b1) (f_a odd: φ -> -φ swaps the orientations), and G = 0 at
w1 = 0.  Hence Ĝ(a) = sup over w1 ∈ (0, a/1.4], b1 ∈ [0, 2π).  The search covers
w1 ∈ (0, a], b1 ∈ [0, 2π], which contains it.

LIPSCHITZ STEP.  Moving (w1, b1) by (δw, δb) moves φ(x) by at most
(1+a)(|x| δw + δb).  max_I φ moves by <= (1+a)(0.8 δw + δb) and min_O φ by
<= (1+a)(2.0 δw + δb); each orientation, and hence G, moves by at most
(1+a)(2.8 δw + 2 δb).  For a cell with half-widths (hw, hb), every point is
within that step of the centre (sup-norm cell geometry).

ALGORITHM.  Cells on (0, a] x [0, 2π]; evaluate G exactly (exact_extrema) at each
centre; lo = best centre value (attained, a valid lower bound); a cell survives if
G(centre) + step(hw, hb) > lo; surviving cells split 2x2; stop when the largest
surviving bound is within target_rel·lo of lo.  The enclosure [lo, hi] is global.

    python -m src.ghat_bnb            # all a in ghat_certified_all.csv
"""

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "ghat_bnb.csv"
TARGET_REL = 1e-3
H0 = 0.05


def step(a, hw, hb):
    return (1.0 + a) * (2.8 * hw + 2.0 * hb)


def certify(a, target_rel=TARGET_REL, h0=H0, max_rounds=40, record=None):
    """record (certificate export, Block 2): if a dict, it receives nw, nb, hw0, hb0 and the leaves as integer
    quadtree indices (level, iw, ib) with reason 1 (pruned: G(centre) + step <= running lo) or 0 (surviving at
    convergence).  The leaves tile [0, a] x [0, 2π] exactly.  Results are identical with and without it."""
    from .exact_extrema import exact_gap
    nw = max(1, math.ceil(a / h0))
    nb = math.ceil(2 * math.pi / h0)
    hw, hb = a / nw / 2, 2 * math.pi / nb / 2
    wc = (np.arange(nw) + 0.5) * 2 * hw
    bc = (np.arange(nb) + 0.5) * 2 * hb
    cells = np.array([(w, b) for w in wc for b in bc])
    if record is not None:
        idx = np.array([(i, j) for i in range(nw) for j in range(nb)], dtype=np.int64)
        record.update(nw=nw, nb=nb, hw0=hw, hb0=hb, leaves=[])
    lo, arg, evaluated = -np.inf, None, 0
    for rnd in range(max_rounds):
        g = np.array([exact_gap(a, float(w), float(b))[0] for w, b in cells])
        evaluated += len(cells)
        i = int(np.argmax(g))
        if g[i] > lo:
            lo, arg = float(g[i]), (float(cells[i, 0]), float(cells[i, 1]))
        ub = g + step(a, hw, hb)
        keep = ub > lo
        hi = float(max(ub[keep].max(), lo))
        if record is not None:
            record["leaves"].append((rnd, idx[~keep], 1))
        if hi - lo <= target_rel * lo:
            if record is not None:
                record["leaves"].append((rnd, idx[keep], 0))
            return {"a": a, "Ghat_lo": lo, "Ghat_hi": hi, "rel_width": (hi - lo) / lo,
                    "w1": arg[0], "b1": arg[1], "rounds": rnd + 1, "cells_evaluated": evaluated,
                    "final_hw": hw, "final_hb": hb, "surviving": int(keep.sum()), "converged": True}
        cells = cells[keep]
        hw, hb = hw / 2, hb / 2
        offs = np.array([(-hw, -hb), (-hw, hb), (hw, -hb), (hw, hb)])
        cells = (cells[:, None, :] + offs[None, :, :]).reshape(-1, 2)
        if record is not None:
            k = idx[keep]
            ch = np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=np.int64)
            idx = (2 * k[:, None, :] + ch[None, :, :]).reshape(-1, 2)
    return {"a": a, "Ghat_lo": lo, "Ghat_hi": hi, "rel_width": (hi - lo) / lo, "w1": arg[0],
            "b1": arg[1], "rounds": max_rounds, "cells_evaluated": evaluated, "final_hw": hw,
            "final_hb": hb, "surviving": int(keep.sum()), "converged": False}


def main(workers=10):
    base = pd.read_csv(RESULTS / "ghat_certified_all.csv")
    with Pool(workers) as p:
        rows = p.map(certify, [float(a) for a in base.a])
    d = pd.DataFrame(rows).merge(base[["a", "Ghat_certified", "Ghat_cert_hi"]], on="a")
    d["zoom_lo_minus_bnb_lo"] = d.Ghat_certified - d.Ghat_lo
    d["zoom_lo_inside_enclosure"] = (d.Ghat_certified >= d.Ghat_lo - 1e-15) & (d.Ghat_certified <= d.Ghat_hi)
    d.to_csv(OUT, index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(certify(float(sys.argv[1])))
    else:
        main()
