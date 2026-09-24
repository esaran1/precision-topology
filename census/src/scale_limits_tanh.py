"""The small-scale limit for tanh at width 2 (registered: results/scale_limits_tanh_prediction.md; theory:
math_note_v2 §10, §10.1).  Population objective (800 points), ‖ṽ‖₁ = 1, box ladder |αᵢ| ≤ A.

    python -m src.scale_limits_tanh
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .scale_limits import RESTARTS, TIE, _data

RESULTS = Path(__file__).resolve().parents[1] / "results"
BOXES = (5.0, 10.0, 20.0, 40.0)
BOUNDARY = 0.999


def _unpack(q, A):
    """q = (q₁, β₁, q₂, β₂, t) with αᵢ = A·sin qᵢ."""
    return np.array([A * math.sin(q[0]), q[1], A * math.sin(q[2]), q[3], float(np.clip(q[4], -1, 1))])


def dmu_q(q, sg, A, act, xO, xI):
    """Δμ and its gradient in q (chain rule through αᵢ = A sin qᵢ)."""
    from .scale_limits import dmu_w2
    p = _unpack(q, A)
    val, g = dmu_w2(p, sg, act, xO, xI)
    g = g.copy()
    g[0] *= A * math.cos(q[0]); g[2] *= A * math.cos(q[2])
    return val, g


def analytic(A, act, x, y, xO, xI, kind):
    """The single step tanh(A(x − c)) or the symmetric pair ½tanh(A(x − c)) − ½tanh(A(x + c)), c optimised on
    [0.8, 1.2].  Returns (p, sg) in the (α₁, β₁, α₂, β₂, t) parametrisation."""
    from .scale_limits import dmu_w2
    def mk(c):
        if kind == "single":
            return np.array([A, -A * c, A, A * c, 1.0]), 1
        return np.array([A, -A * c, A, A * c, 0.5]), -1
    f = lambda c: dmu_w2(*mk(c), act, xO, xI)[0]
    grid = np.arange(0.8, 1.2 + 1e-12, 1e-4)
    vals = np.array([f(c) for c in grid])
    k = int(np.argmax(vals)); lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(80):
        if f(a) > f(b):
            hi = b
        else:
            lo = a
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    c = 0.5 * (lo + hi)
    return mk(c)


def box(A, act, x, y, xO, xI, rng):
    from .cmaes import cma_es
    from .scale_limits import _phi_w2
    from .width2_conditional import bfgs
    from .width2_geometry import gaps
    cands = []
    for k in range(RESTARTS):
        q0 = np.array([rng.uniform(-math.pi / 2, math.pi / 2), rng.uniform(-2.5 * A, 2.5 * A),
                       rng.uniform(-math.pi / 2, math.pi / 2), rng.uniform(-2.5 * A, 2.5 * A), rng.uniform(-1, 1)])
        sg = int(rng.choice([-1, 1]))
        q, f, _, _ = bfgs(lambda z, sg=sg: tuple(-v for v in dmu_q(z, sg, A, act, xO, xI)), q0, gtol=1e-12, maxit=500)
        cands.append((-f, _unpack(q, A), sg, "search"))
    best = max(c[0] for c in cands)
    ladder_500 = max(c[0] for c in cands[:500])

    def negd(z):                                     # independent: v unconstrained, normalised; α clipped to the box
        v = z[4:6]; n1 = abs(v[0]) + abs(v[1])
        if n1 < 1e-12:
            return 1e9
        v = v / n1
        a1, a2 = np.clip(z[0], -A, A), np.clip(z[2], -A, A)
        u1 = act.u(a1 * xO + z[1]).mean() - act.u(a1 * xI + z[1]).mean()
        u2 = act.u(a2 * xO + z[3]).mean() - act.u(a2 * xI + z[3]).mean()
        return -(v[0] * u1 + v[1] * u2)
    ind = -min(cma_es(negd, np.r_[rng.uniform(-A, A), rng.uniform(-2 * A, 2 * A), rng.uniform(-A, A),
                                  rng.uniform(-2 * A, 2 * A), rng.uniform(-1, 1, 2)], 0.3 * A, max_generations=400,
                      seed=int(rng.integers(1 << 30))).best_f for _ in range(40))
    from .scale_limits import dmu_w2
    an = {kind: analytic(A, act, x, y, xO, xI, kind) for kind in ("single", "pair")}
    an_val = {k: dmu_w2(p, sg, act, xO, xI)[0] for k, (p, sg) in an.items()}
    top = max(best, *an_val.values())
    pool = [c for c in cands if c[0] >= top - TIE] + [(an_val[k], p, sg, f"analytic_{k}") for k, (p, sg) in an.items()
                                                      if an_val[k] >= top - TIE]
    rows = []
    for val, p, sg, src in pool:
        phi, v = _phi_w2(p, sg, act, x)
        typ = "single_unit" if abs(p[4]) >= 1 - 1e-9 else "pair"
        Gp = gaps(p[:4], v, act)["G+"]
        rows.append({"A": A, "source": src, "type": typ, "dmu": val, "var": float(np.var(phi)),
                     "max_abs_alpha_over_A": float(max(abs(p[0]) if abs(v[0]) > 0 else 0, abs(p[2]) if abs(v[1]) > 0 else 0) / A),
                     "t": float(p[4]), "G_lo": Gp[0], "G_hi": Gp[1],
                     "G_n": float(phi[y == 1].min() - phi[y == 0].max())})
    M = pd.DataFrame(rows)
    sel = M.loc[M["var"].idxmin()]
    out = {"A": A, "dmu_max_search": best, "one_minus_max": 1 - top, "ladder_500": ladder_500,
           "ladder_ok": abs(best - ladder_500) <= TIE, "independent_max": ind, "independent_ok": ind <= top + TIE,
           "analytic_single": an_val["single"], "analytic_pair": an_val["pair"],
           "analytic_attain": all(v >= top - TIE for v in an_val.values()),
           "search_not_above_analytic": best <= max(an_val.values()) + TIE,
           "n_maximisers": len(M), "maximisers_on_boundary": bool((M.max_abs_alpha_over_A >= BOUNDARY).all()),
           "min_max_abs_alpha_over_A": float(M.max_abs_alpha_over_A.min()),
           "G_range_single": (float(M[M.type == "single_unit"].G_hi.min()), float(M[M.type == "single_unit"].G_hi.max()))
           if (M.type == "single_unit").any() else None,
           "G_range_pair": (float(M[M.type == "pair"].G_lo.min()), float(M[M.type == "pair"].G_lo.max()))
           if (M.type == "pair").any() else None,
           "selected_source": sel.source, "selected_type": sel.type, "selected_t": sel.t, "selected_var": sel["var"],
           "selected_G_lo": sel.G_lo, "selected_G_hi": sel.G_hi, "selected_G_n": sel.G_n, "top": top}
    return out, M


def verdict(r):
    """r: the per-box summary DataFrame (sorted by A)."""
    valid = bool(r.ladder_ok.all() and r.independent_ok.all() and r.analytic_attain.all())
    if not valid:
        return "STOP: validation failed"
    if (r.selected_G_hi <= 0).any():
        return "STOP: Var-selected member unplaced at some A (contradicts the limit analysis)"
    inc = np.diff(r.top.values)
    rising = bool((inc > TIE).all() and (r.top < 1).all())
    if rising and bool(r.maximisers_on_boundary.all()) and bool((r.selected_G_lo > 0).all()):
        return "T1: supremum approached by placed configurations, not attained"
    if inc[-1] <= TIE and r.min_max_abs_alpha_over_A.iloc[-1] < BOUNDARY and r.selected_G_lo.iloc[-1] > 0:
        return "STOP (T2): attained placed maximiser, contradicting the bound dmu < 1 = sup"
    return "neither"


def main():
    from .width2_geometry import Act
    x, y, xO, xI = _data()
    act = Act("tanh")
    rng = np.random.default_rng(0)
    rows, Ms = [], []
    for A in BOXES:
        r, M = box(A, act, x, y, xO, xI, rng)
        rows.append(r); Ms.append(M)
        print(r, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / "scale_limits_tanh.csv", index=False)
    pd.concat(Ms).to_csv(RESULTS / "scale_limits_tanh_maximisers.csv", index=False)
    v = verdict(r)
    pd.DataFrame([{"verdict": v}]).to_csv(RESULTS / "scale_limits_tanh_summary.csv", index=False)
    print(v)


if __name__ == "__main__":
    main()
