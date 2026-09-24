"""The small-scale limit of the conditional minimiser (registered: results/scale_limits_prediction.md; theory:
math_note_v2 §10).  Population objective (800 points).

    python -m src.scale_limits            # D(α), width 1 at six a, width 2 at a = 1.30 and 1.50, summary + verdict
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
A1 = (1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
A2 = (1.30, 1.50)
RESTARTS = 2000
TIE = 1e-9


def _data():
    from .width2_conditional import population
    x, y = population()
    return x, y, x[y == 1], x[y == 0]


def D_of(alpha, xO, xI):
    a = np.atleast_1d(alpha)[:, None]
    return np.cos(a * xO[None, :]).mean(axis=1) - np.cos(a * xI[None, :]).mean(axis=1)


def alpha_star(xO, xI):
    grid = np.arange(0.001, 50.0, 0.001)
    Dg = np.concatenate([D_of(grid[i:i + 5000], xO, xI) for i in range(0, len(grid), 5000)])
    k = int(np.argmax(np.abs(Dg)))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    f = lambda al: abs(float(D_of(al, xO, xI)[0]))
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(80):
        if f(c) > f(d):
            hi = d
        else:
            lo = c
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    al = 0.5 * (lo + hi)
    return al, float(D_of(al, xO, xI)[0]), float(grid[k]), float(Dg[k])


# ------------------------------------------------------------------------------------------ width 1
def dmu_w1(p, a, xO, xI):
    w, b = p
    tO, tI = w * xO + b, w * xI + b
    fO, fI = tO + a * np.sin(tO), tI + a * np.sin(tI)
    dO, dI = 1 + a * np.cos(tO), 1 + a * np.cos(tI)
    val = fO.mean() - fI.mean()
    g = np.array([(dO * xO).mean() - (dI * xI).mean(), dO.mean() - dI.mean()])
    return val, g


def width1(a, xO, xI, x, y):
    from .width2_conditional import bfgs
    from .profiled_bnb import gap
    rng = np.random.default_rng(0)
    cands = []
    for _ in range(RESTARTS):
        p0 = np.array([rng.uniform(-10, 10), rng.uniform(0, 2 * math.pi)])
        p, f, g, _ = bfgs(lambda q: tuple(-v for v in dmu_w1(q, a, xO, xI)), p0, gtol=1e-12, maxit=500)
        cands.append((-f, p))
    best = max(c[0] for c in cands)
    M = [c for c in cands if c[0] >= best - TIE]
    var = [np.var(_phi_w1(c[1], a, x)) for c in M]
    sel = M[int(np.argmin(var))][1]
    G = float(gap(np.array([sel[0]]), np.array([sel[1]]), a)[0])
    phi = _phi_w1(sel, a, x)
    Gn = float(phi[y == 1].min() - phi[y == 0].max())
    # the author's check: Δμ equals a·[mean_O sin − mean_I sin]
    sinpart = a * (np.sin(sel[0] * xO + sel[1]).mean() - np.sin(sel[0] * xI + sel[1]).mean())
    return {"a": a, "dmu_max": best, "n_maximisers": len(M), "w1": float(sel[0]), "b1": float(sel[1] % (2 * math.pi)),
            "G": G, "G_n": Gn, "sine_check_abs_diff": abs(best - sinpart)}


def _phi_w1(p, a, x):
    t = p[0] * x + p[1]
    return t + a * np.sin(t)


# ------------------------------------------------------------------------------------------ width 2
def _phi_w2(p, sg, act, x):
    t = float(np.clip(p[4], -1, 1)); v = np.array([t, sg * (1 - abs(t))])
    return v[0] * act.u(p[0] * x + p[1]) + v[1] * act.u(p[2] * x + p[3]), v


def dmu_w2(p, sg, act, xO, xI):
    t = float(np.clip(p[4], -1, 1)); v0, v1 = t, sg * (1 - abs(t))
    TO1, TI1, TO2, TI2 = p[0] * xO + p[1], p[0] * xI + p[1], p[2] * xO + p[3], p[2] * xI + p[3]
    D1 = act.u(TO1).mean() - act.u(TI1).mean(); D2 = act.u(TO2).mean() - act.u(TI2).mean()
    val = v0 * D1 + v1 * D2
    g = np.array([v0 * ((act.du(TO1) * xO).mean() - (act.du(TI1) * xI).mean()),
                  v0 * (act.du(TO1).mean() - act.du(TI1).mean()),
                  v1 * ((act.du(TO2) * xO).mean() - (act.du(TI2) * xI).mean()),
                  v1 * (act.du(TO2).mean() - act.du(TI2).mean()),
                  (D1 - sg * np.sign(t) * D2) if abs(t) < 1 else 0.0])
    return val, g


def width2(a, xO, xI, x, y, al_star, D_star):
    from .width2_conditional import bfgs
    from .width2_geometry import Act, gaps
    act = Act("fa", a)
    rng = np.random.default_rng(0)
    cands = []
    for k in range(RESTARTS):
        p0 = np.array([rng.uniform(0, 10), rng.uniform(0, 2 * math.pi), rng.uniform(0, 10), rng.uniform(0, 2 * math.pi),
                       rng.uniform(-1, 1)]); sg = int(rng.choice([-1, 1]))
        p, f, g, _ = bfgs(lambda q, sg=sg: tuple(-v for v in dmu_w2(q, sg, act, xO, xI)), p0, gtol=1e-12, maxit=500)
        p[4] = np.clip(p[4], -1, 1)
        cands.append((-f, p, sg, k))
    best = max(c[0] for c in cands)
    ladder_500 = max(c[0] for c in cands[:500])
    # independent search: v unconstrained, normalised (CMA-ES)
    from .cmaes import cma_es
    def negd(q):
        v = q[4:6]; n1 = abs(v[0]) + abs(v[1])
        if n1 < 1e-12:
            return 1e9
        v = v / n1
        u1 = act.u(q[0] * xO + q[1]).mean() - act.u(q[0] * xI + q[1]).mean()
        u2 = act.u(q[2] * xO + q[3]).mean() - act.u(q[2] * xI + q[3]).mean()
        return -(v[0] * u1 + v[1] * u2)
    ind = -min(cma_es(negd, np.r_[rng.uniform(0, 10, 4), rng.uniform(-1, 1, 2)], 1.0, max_generations=400,
                      seed=int(rng.integers(1 << 30))).best_f for _ in range(40))
    M = [c for c in cands if c[0] >= best - TIE]
    types = {"single_unit": [], "pair_c_nonzero": [], "pair_c_zero": []}
    rows_M = []
    for val, p, sg, k in M:
        phi, v = _phi_w2(p, sg, act, x)
        c = v[0] * p[0] + v[1] * p[2]
        typ = "single_unit" if abs(p[4]) >= 1 - 1e-9 else ("pair_c_zero" if abs(c) < 1e-6 else "pair_c_nonzero")
        Gp = gaps(p[:4], v, act)["G+"][0]
        types[typ].append(Gp)
        rows_M.append({"type": typ, "dmu": val, "var": float(np.var(phi)), "c": float(c), "G": Gp})
    # the analytically derived Var-minimiser on M: the cancelling pair at α*, φ = const + a·sign(D*)·cos(α*x)
    sgnD = 1.0 if D_star > 0 else -1.0
    # ṽ = (½, −½), α₁ = α₂ = α*: v₁ sin(α x + β₁) + v₂ sin(α x + β₂) = sign(D)·cos(αx) needs β₁ = sign(D)·π/2, β₂ = −β₁
    b1 = sgnD * math.pi / 2
    th_pair = np.array([al_star, b1, al_star, -b1]); v_pair = np.array([0.5, -0.5])
    phi_pair = v_pair[0] * act.u(th_pair[0] * x + th_pair[1]) + v_pair[1] * act.u(th_pair[2] * x + th_pair[3])
    dmu_pair = float(phi_pair[y == 1].mean() - phi_pair[y == 0].mean())
    Gpair = gaps(th_pair, v_pair, act)["G+"]
    Gn_pair = float(phi_pair[y == 1].min() - phi_pair[y == 0].max())
    var_pair = float(np.var(phi_pair))
    sine_diffs = []
    for val, p, sg, k in M[:200]:
        phi, v = _phi_w2(p, sg, act, x)
        sp = act.a * sum(v[i] * (np.sin(p[2 * i] * xO + p[2 * i + 1]).mean() - np.sin(p[2 * i] * xI + p[2 * i + 1]).mean())
                         for i in (0, 1))
        sine_diffs.append(abs(val - sp))
    return ({"a": a, "dmu_max_search": best, "dmu_bound_a_absD": a * abs(D_star), "ladder_500": ladder_500,
             "ladder_ok": abs(best - ladder_500) <= TIE, "independent_max": ind, "independent_ok": ind <= best + TIE,
             "n_maximisers": len(M), **{f"n_{k}": len(v) for k, v in types.items()},
             **{f"G_range_{k}": (min(v), max(v)) if v else None for k, v in types.items()},
             "pair_dmu": dmu_pair, "pair_attains_max": dmu_pair >= best - TIE, "pair_var": var_pair,
             "pair_var_le_all_maximisers": all(var_pair <= r["var"] + 1e-12 for r in rows_M),
             "selected_G_lo": Gpair[0], "selected_G_hi": Gpair[1], "selected_G_n": Gn_pair,
             "sine_check_max_abs_diff": max(sine_diffs)}, pd.DataFrame(rows_M))


def main():
    x, y, xO, xI = _data()
    al, Ds, al_grid, D_grid = alpha_star(xO, xI)
    pd.DataFrame([{"alpha_star": al, "D_star": Ds, "alpha_grid": al_grid, "D_grid": D_grid}]).to_csv(
        RESULTS / "scale_limits_D.csv", index=False)
    print("alpha* =", al, "D* =", Ds, flush=True)
    w1 = pd.DataFrame([width1(a, xO, xI, x, y) for a in A1])
    w1.to_csv(RESULTS / "scale_limits_width1.csv", index=False)
    print(w1.to_string(index=False), flush=True)
    rows = []
    for a in A2:
        r, M = width2(a, xO, xI, x, y, al, Ds)
        rows.append(r); M.to_csv(RESULTS / f"scale_limits_width2_maximisers_a{a:.2f}.csv", index=False)
        print(r, flush=True)
    w2 = pd.DataFrame(rows)
    w2.to_csv(RESULTS / "scale_limits_width2.csv", index=False)
    w1_consistent = bool((w1.G <= 0).all())
    valid = bool(w2.ladder_ok.all() and w2.independent_ok.all() and w2.pair_attains_max.all()
                 and w2.pair_var_le_all_maximisers.all())
    placed = bool((w2.selected_G_lo > 0).all())
    unplaced = bool((w2.selected_G_hi <= 0).all())
    verdict = ("STOP: width-1 consistency check failed" if not w1_consistent else
               "STOP: width-2 validation failed" if not valid else
               "width 2 f_a: NO placement threshold (selected dmu-maximiser has G > 0): both arms NOT APPLICABLE; W1, W4 not run"
               if placed else "width 2 f_a: threshold exists (G <= 0): W0 continues" if unplaced else "undecided (G straddles 0)")
    pd.DataFrame([{"width1_all_G_le_0": w1_consistent, "width2_validation_ok": valid, "verdict": verdict}]).to_csv(
        RESULTS / "scale_limits_summary.csv", index=False)
    print(verdict)


if __name__ == "__main__":
    main()
