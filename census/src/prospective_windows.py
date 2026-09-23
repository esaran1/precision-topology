"""Block 3a: choose held-out window geometries by a structural rule, before any training in them.

Candidate family: symmetric windows (inner half-width i, outer inner edge o1, outer outer edge o2)
  i  ∈ {0.5, 0.6, 0.7, 0.8, 0.9},  o1 = i + g, g ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.8},
  o2 = o1 + d, d ∈ {0.6, 0.8, 1.2}
excluding the five geometries already used (blockG_windows.WINDOWS; G4 is asymmetric and not in
the family anyway).  For each candidate, κ0 = K / D∞ with D∞ = 4√2/3 and K = sup over (u, v) of the
limit class gap G0 = max over orientations of (min_O h(σ) - max_I h(σ)), σ = u x + v,
h(σ) = -σ + σ³/6 — computed from the limiting cubic alone, certified by branch and bound
(|ΔG0| <= max|h'| over the cell · ((i + o2) hu + 2 hv), exact extrema of h on intervals).
Selection: the four candidates nearest the κ0 quantiles 0.1, 0.35, 0.65, 0.9 of the family.

    python -m src.prospective_windows kappa      # κ0 for all candidates -> prospective_candidates.csv
    python -m src.prospective_windows select     # the four held-out geometries
"""

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .limit_bnb import _h_extrema

RESULTS = Path(__file__).resolve().parents[1] / "results"
D_INF = 4 * math.sqrt(2) / 3
USED = {(0.8, 1.2, 2.0), (0.6, 1.4, 2.0), (0.95, 1.05, 2.0), (0.8, 1.2, 3.0)}
QUANTILES = (0.1, 0.35, 0.65, 0.9)


def candidates():
    out = []
    for i in (0.5, 0.6, 0.7, 0.8, 0.9):
        for g in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
            for d in (0.6, 0.8, 1.2):
                w = (i, round(i + g, 4), round(i + g + d, 4))
                if w not in USED:
                    out.append(w)
    return out


def g0(u, v, i, o1, o2):
    """Oriented limit gap, both orientations, exact extrema."""
    def ext(lo, hi):
        a1, a2 = u * lo + v, u * hi + v
        return _h_extrema(np.minimum(a1, a2), np.maximum(a1, a2))
    imn, imx = ext(-i, i)
    p_mn, p_mx = ext(o1, o2)
    n_mn, n_mx = ext(-o2, -o1)
    omn, omx = np.minimum(p_mn, n_mn), np.maximum(p_mx, n_mx)
    return np.maximum(omn - imx, imn - omx)


def K_certified(win, U=8.0, V=12.0, h0=0.05, rel=1e-4):
    i, o1, o2 = win
    nu, nv = int(U / h0), int(2 * V / h0)
    hu, hv = U / nu / 2, V / nv
    uc = (np.arange(nu) + 0.5) * 2 * hu
    vc = -V + (np.arange(nv) + 0.5) * 2 * hv
    cu, cv = np.meshgrid(uc, vc, indexing="ij"); cu, cv = cu.ravel(), cv.ravel()
    best, arg = -np.inf, None
    for _ in range(40):
        G = g0(cu, cv, i, o1, o2)
        smax = o2 * (np.abs(cu) + hu) + np.abs(cv) + hv
        step = (1 + smax ** 2 / 2) * ((i + o2) * hu + 2 * hv)
        j = int(np.argmax(G))
        if G[j] > best:
            best, arg = float(G[j]), (float(cu[j]), float(cv[j]))
        ub = G + step
        keep = ub > best
        hi = float(ub[keep].max()) if keep.any() else best
        if hi - best <= rel * best:
            edge = (arg[0] > U - 0.5) or (abs(arg[1]) > V - 0.5)
            return {"i": i, "o1": o1, "o2": o2, "K_lo": best, "K_hi": hi, "u": arg[0], "v": arg[1],
                    "argmax_near_box_edge": edge}
        cu, cv = cu[keep], cv[keep]
        hu, hv = hu / 2, hv / 2
        cu = np.concatenate([cu - hu, cu - hu, cu + hu, cu + hu])
        cv = np.concatenate([cv - hv, cv + hv, cv - hv, cv + hv])
    raise RuntimeError(f"K not converged for {win}")


def kappa():
    with Pool(int(__import__('os').environ.get('PW_WORKERS', '3'))) as p:
        rows = p.map(K_certified, candidates())
    d = pd.DataFrame(rows)
    d["kappa0_lo"], d["kappa0_hi"] = d.K_lo / D_INF, d.K_hi / D_INF
    d.to_csv(RESULTS / "prospective_candidates.csv", index=False)
    print(d.describe().to_string())


def select():
    d = pd.read_csv(RESULTS / "prospective_candidates.csv")
    d = d[d.K_lo > 0].reset_index(drop=True)
    rows = []
    for q in QUANTILES:
        target = float(d.kappa0_lo.quantile(q))
        j = int((d.kappa0_lo - target).abs().idxmin())
        rows.append({"quantile": q, "target_kappa0": target, **d.loc[j].to_dict()})
    t = pd.DataFrame(rows)
    assert t[["i", "o1", "o2"]].drop_duplicates().shape[0] == len(t)
    t.to_csv(RESULTS / "prospective_windows.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    {"kappa": kappa, "select": select}[sys.argv[1]]()
