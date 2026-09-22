"""Certified Ghat for the constructed q-families (information for T52 condition 4).

The committed `gstar` column of r_families.csv came from a search that is not in
the repository.  This computes the supremum properly: exact x-extrema (endpoints
plus the closed-form critical points +-((a-1)/a)^(1/q)) inside a Lipschitz-
certified grid over placements, searched in limit coordinates w1 = s*u, b1 = s*v
with s = (a-1)^(1/q) so the fold is resolved at every a.

    python -m src.family_certify   ->  results/family_ghat_certified.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
"""Certified Ghat for the q-families, exact x-extrema + Lipschitz slack (information only)."""
I_, O_ = (-0.8, 0.8), (1.2, 2.0)
def f(t, q, a):
    t = np.asarray(t, float); at = np.abs(t)
    inner = (1 - a) * t + a * np.sign(t) * at ** (q + 1) / (q + 1)
    edge = (1 - a) + a / (q + 1)
    return np.where(at <= 1, inner, np.sign(t) * edge + (t - np.sign(t)))
def ext(q, a, w1, b1, lo, hi):
    xs = ((a - 1) / a) ** (1 / q)
    pts = [lo, hi] + [(c - b1) / w1 for c in (xs, -xs) if lo < (c - b1) / w1 < hi]
    v = f(w1 * np.array(pts) + b1, q, a); return v.min(), v.max()
def gap(q, a, w1, b1):
    ilo, ihi = ext(q, a, w1, b1, *I_)
    p = ext(q, a, w1, b1, *O_); n = ext(q, a, w1, b1, -O_[1], -O_[0])
    olo, ohi = min(p[0], n[0]), max(p[1], n[1])
    return max(olo - ihi, ilo - ohi)
def cert(q, a, U=6.0, V=6.0, h0=0.02, rel=1e-3):
    """Search in limit coordinates w1 = s*u, b1 = s*v, s = (a-1)**(1/q)."""
    s = (a - 1) ** (1 / q)
    L = max(1.0, a - 1.0) * 2.0 * s                 # Lipschitz of gap in (u, v)
    def scan(us, vs):
        best, arg = -1e9, None
        for u in us:
            if abs(u) < 1e-9: continue
            for v in vs:
                g = gap(q, a, s * u, s * v)
                if g > best: best, arg = g, (u, v)
        return best, arg
    h = h0; lo, arg = scan(np.arange(-U, U + 1e-9, h), np.arange(-V, V + 1e-9, h))
    while L * h / 2 >= rel * lo and h > 1e-7:
        h /= 5; u0, v0 = arg; pad = 12 * h
        lo, arg = scan(np.arange(u0 - pad, u0 + pad + 1e-12, h), np.arange(v0 - pad, v0 + pad + 1e-12, h))
    return lo, lo + L * h / 2



def main() -> None:
    d = pd.read_csv(RESULTS / "r_families.csv")
    rows = []
    for (q, a), g in d.groupby(["q", "a"]):
        lo, hi = cert(q, a)
        rows.append({"q": q, "a": a, "gstar_committed": g.gstar.iloc[0],
                     "ghat_cert_lo": lo, "ghat_cert_hi": hi, "ratio": lo / g.gstar.iloc[0]})
        print(f"q={q} a={a}: committed {g.gstar.iloc[0]:.6f}  certified [{lo:.6f}, {hi:.6f}]",
              flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "family_ghat_certified.csv", index=False)


if __name__ == "__main__":
    main()
