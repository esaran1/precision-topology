"""The domain lemma for K = sup G₀ (math_note_v2 §8, the limit class gap), and its numerical check.

G₀(u, v) = max(min_O h(σ) − max_I h(σ), min_I h(σ) − max_O h(σ)), σ = u x + v, h(σ) = −σ + σ³/6, I = [−0.8, 0.8],
O = ±[1.2, 2.0].

Identity: for d >= 0, h(a) − h(a + d) = d·[1 − ((a + d/2)² + d²/12)/2].
Each orientation is bounded by one outer point against one inner point (the min over a set is at most any member,
the max at least any member), with σ increasing in x for u >= 0:
  orientation O over I:  <= h(σ(−2)) − h(σ(xi)),  xi = −0.8 (d = 1.2u) or xi = 0.8 (d = 2.8u, a + d/2 = v − 0.6u);
  orientation I over O:  <= h(σ(xi)) − h(σ(2)),   xi = 0.8 (d = 1.2u)  or xi = −0.8 (d = 2.8u, a + d/2 = v + 0.6u).
Lemma: G₀(u, v) <= 0 if u >= √(50/3) (then d²/12 >= 2 with d = 1.2u), or if |v| >= √2 + 0.6u (then (a + d/2)² >= 2 in
both orientations).  G₀(−u, v) = G₀(u, v) (x -> −x; I and O are symmetric).  K > 0, so
sup G₀ = sup over {|u| < √(50/3) ≈ 4.0825, |v| < √2 + 0.6·√(50/3) ≈ 3.864}, inside the certified box [0, 8] x [−12, 12]
(with u >= 0 by the symmetry).

    python -m src.k_domain
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
U0 = math.sqrt(50 / 3)


def h(s):
    return -s + s ** 3 / 6


def main():
    from .limit_windows import _g0_both
    rng = np.random.default_rng(0)
    rows = {}
    # (1) the identity, at random a, d
    a, d = rng.uniform(-50, 50, 100_000), rng.uniform(0, 50, 100_000)
    lhs = h(a) - h(a + d); rhs = d * (1 - ((a + d / 2) ** 2 + d ** 2 / 12) / 2)
    rows["identity_max_rel_err"] = float(np.max(np.abs(lhs - rhs) / (1 + np.abs(lhs))))
    # (2) G0 <= 0 on the excluded region: dense grid over u in [0, 60], v in [-80, 80], plus far random points
    worst, n_ex = -np.inf, 0
    us = np.linspace(0, 60, 1201)
    for u in us:
        v = np.linspace(-80, 80, 3201)
        ex = (u >= U0) | (np.abs(v) >= math.sqrt(2) + 0.6 * u)
        if ex.any():
            g = _g0_both(np.full(ex.sum(), u), v[ex], (-0.8, 0.8), (1.2, 2.0))
            worst = max(worst, float(g.max())); n_ex += int(ex.sum())
    uf, vf = rng.uniform(0, 1e4, 200_000), rng.uniform(-1e4, 1e4, 200_000)
    ex = (uf >= U0) | (np.abs(vf) >= math.sqrt(2) + 0.6 * uf)
    gf = _g0_both(uf[ex], vf[ex], (-0.8, 0.8), (1.2, 2.0))
    rows["excluded_points_checked"] = n_ex + int(ex.sum())
    rows["max_G0_on_excluded_region"] = float(max(worst, gf.max()))       # 0 exactly at u = 0 (σ constant)
    m = -np.inf
    for u in np.linspace(0.01, 60, 1200):
        v = np.linspace(-80, 80, 3201)
        e = (u >= U0) | (np.abs(v) >= math.sqrt(2) + 0.6 * u)
        if e.any():
            m = max(m, float(_g0_both(np.full(e.sum(), u), v[e], (-0.8, 0.8), (1.2, 2.0)).max()))
    rows["max_G0_on_excluded_region_u_positive"] = m
    # (3) symmetry G0(-u, v) = G0(u, v)
    u, v = rng.uniform(0, 8, 10_000), rng.uniform(-12, 12, 10_000)
    rows["symmetry_max_abs_diff"] = float(np.abs(_g0_both(u, v, (-0.8, 0.8), (1.2, 2.0))
                                                 - _g0_both(-u, v, (-0.8, 0.8), (1.2, 2.0))).max())
    # (4) the region fits the certified box, and K's argmax lies in it
    kb = pd.read_csv(RESULTS / "limit_K_base.csv").iloc[0]
    rows["u_bound"] = U0; rows["v_bound"] = math.sqrt(2) + 0.6 * U0
    rows["region_inside_box"] = bool(U0 < 8 and math.sqrt(2) + 0.6 * U0 < 12)
    rows["K_argmax_inside_region"] = bool(kb.u < U0 and abs(kb.v) < math.sqrt(2) + 0.6 * kb.u)
    rows["K_lo"] = float(kb.K_lo)
    pd.DataFrame([rows]).to_csv(RESULTS / "k_domain_check.csv", index=False)
    for k, v in rows.items():
        print(f"{k:40s} {v!r}")


if __name__ == "__main__":
    main()
