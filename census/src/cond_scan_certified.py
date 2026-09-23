"""Block 1 addendum: certified coarse scan across the full scale range at a = 1.30.

At each |w2| in 1.5..11 (step 0.5): the certified global conditional minimiser (profiled_bnb.certify on
both regions, tolerance 1e-9), the enclosure of its surviving cells (per sign of w1; the population is
x-symmetric so w1 -> -w1 is an exact symmetry), and a certified lower bound on L* outside balls of radius
0.1 around the minimiser and its mirror (profiled_bnb.competitor_gap).  A positive margin at every scale
certifies that no other basin comes within that margin of the branch at those scales.

    python -m src.cond_scan_certified
"""

from __future__ import annotations

import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A = 1.30
RADIUS = 0.1


def _job(s):
    from . import blockB_landscape as bb
    from .profiled_bnb import certify, competitor_gap, gap
    x, y = bb.population_data()
    x, y = x.numpy(), y.numpy()
    rm = certify(s, A, x, y, "-", tol=1e-9)
    rp = certify(s, A, x, y, "+", tol=1e-9)
    g = rp if rp["upper"] < rm["upper"] else rm
    w1, b1 = g["arg_w1"], g["arg_b1"]
    cg = competitor_gap(s, A, x, y, [(w1, b1), (-w1, b1)], RADIUS)
    enc = g.get("encl_pos") or g.get("encl_neg")
    return {"s": s, "global_region": g["region"], "m_minus_lo": rm["lower"], "m_minus_hi": rm["upper"],
            "m_plus_lo": rp["lower"], "m_plus_hi": rp["upper"], "argmin_w1": abs(w1), "argmin_b1": b1,
            "argmin_G": float(gap([abs(w1)], [b1], A)[0]) if w1 != 0 else 0.0,
            "encl_w1": None if enc is None else f"[{enc[0]:.6f}, {enc[1]:.6f}]",
            "encl_b1": None if enc is None else f"[{enc[2]:.6f}, {enc[3]:.6f}]",
            "competitor_lower": cg["lower"], "competitor_converged": cg["converged"],
            "competitor_margin": cg["lower"] - g["upper"], "constant_margin": math.log(2) - g["upper"]}


def main():
    grid = [round(v, 4) for v in np.arange(1.5, 11.0 + 1e-9, 0.5)]
    with Pool(int(os.environ.get("SC_WORKERS", "1"))) as p:
        rows = p.map(_job, grid)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "cond_scan_certified_a130.csv", index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
