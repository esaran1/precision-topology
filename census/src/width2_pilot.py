"""Width 2 — EXPLORATORY pilot (not registered; nothing here tests a registered outcome).

1. The cosine identity: f_a(αx + π/2) − f_a(αx + 3π/2) = −π + 2a·cos(αx) (checked numerically on a grid).
2. The single-cosine problem: sup over α of G(cos(αx)) = 1/√2, attained at α = 5π/8 (analytic; checked).
3. Γ̂₂ at a = 1.30 and 1.50 by the two independent searches (4,000-start Nelder–Mead; differential evolution),
   with the maximiser's linear coefficient c = Σṽᵢαᵢ, and Γ̂₂/a.
4. The a-free problem obtained when c = 0: κ = sup G(Σṽᵢ sin(αᵢx + βᵢ)) over ‖ṽ‖₁ = 1 with Σṽᵢαᵢ = 0.
5. Conditional scans at a = 0.5 and 1.0 (f_a monotone) and at 1.30 / 1.50, coarse (100 restarts per scale):
   the retained minimiser's directional G₊ against R₂, and the restart iteration-cap hit rate.

    python -m src.width2_pilot [identity | gamma | kappa | scan]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .width2_geometry import Act, INNER, OUTER, F130, F150, differential_evolution, exact_gap, extrema, gamma2_de, \
    gamma2_multistart, nelder_mead, phi

RESULTS = Path(__file__).resolve().parents[1] / "results"


def identity():
    x = np.linspace(-3, 3, 100_001)
    rows = []
    for a in (0.5, 1.0, 1.3, 1.5):
        act = Act("fa", a)
        for alpha in (0.7, 1.79, 5 * math.pi / 8, 7.7):
            lhs = act.u(alpha * x + math.pi / 2) - act.u(alpha * x + 3 * math.pi / 2)
            rhs = -math.pi + 2 * a * np.cos(alpha * x)
            rows.append({"a": a, "alpha": alpha, "max_abs_diff": float(np.abs(lhs - rhs).max())})
    # single cosine: G(cos(αx)) on a fine α grid, exact extrema of cos (the f_a identity with ṽ = (½, −½), divided by a)
    al = np.linspace(0.01, 12, 12_000)
    def gcos(al_):
        th = np.array([al_, math.pi / 2, al_, 3 * math.pi / 2]); v = np.array([0.5, -0.5])
        return max(exact_gap(th, v, Act("fa", 1.0)))
    g = np.array([gcos(v) for v in al])
    k = int(np.argmax(g))
    single = {"alpha_argmax_grid": float(al[k]), "G_max_grid": float(g[k]), "alpha_5pi_8": 5 * math.pi / 8,
              "G_at_5pi_8": gcos(5 * math.pi / 8), "one_over_sqrt2": 1 / math.sqrt(2)}
    pd.DataFrame(rows).to_csv(RESULTS / "width2_pilot_identity.csv", index=False)
    pd.DataFrame([single]).to_csv(RESULTS / "width2_pilot_single_cosine.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(single)


def gamma():
    rows = []
    for act in (F130, F150):
        for name, r in (("multistart NM, 4000 starts", gamma2_multistart(act, starts=4000, seed=0)),
                        ("differential evolution", gamma2_de(act, seed=1))):
            th, v = r["theta"], r["v"]
            c = float(v[0] * th[0] + v[1] * th[2])
            rows.append({"a": act.a, "search": name, "gamma_lo": r["gamma_lo"], "gamma_hi": r["gamma_hi"],
                         "gamma_over_a": r["gamma_lo"] / act.a, "alpha1": th[0], "beta1": th[1], "alpha2": th[2],
                         "beta2": th[3], "v1": v[0], "v2": v[1], "linear_coefficient_c": c,
                         "c_over_sum_abs_v_alpha": abs(c) / (abs(v[0] * th[0]) + abs(v[1] * th[2]))})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "width2_pilot_gamma.csv", index=False)


def _g_sin(z):
    """G(Σṽᵢ sin(αᵢx + βᵢ)) with ṽ = (t, 1 − |t|) scaled to satisfy Σṽᵢαᵢ = 0 is not a free parameter: here the
    constraint is imposed by solving for α₂ = −ṽ₁α₁/ṽ₂ (ṽ₂ ≠ 0)."""
    a1, b1, b2, t = z
    t = float(np.clip(t, -0.999, 0.999)); v = np.array([t, 1 - abs(t)])
    a2 = -v[0] * a1 / v[1]
    xi, xo = np.linspace(*INNER, 1601), np.concatenate([np.linspace(*OUTER[0], 801), np.linspace(*OUTER[1], 801)])
    f = lambda x: v[0] * np.sin(a1 * x + b1) + v[1] * np.sin(a2 * x + b2)
    fi, fo = f(xi), f(xo)
    return max(fo.min() - fi.max(), fi.min() - fo.max())


def kappa(starts=3000, seed=0):
    """The a-free problem (c = 0): best value by multistart NM on the dense-grid gap, then exact extrema at the
    best point (via f_a with a = 1 and the linear parts cancelled: G(φ)/a is exactly this problem's value)."""
    rng = np.random.default_rng(seed)
    best = (-math.inf, None)
    for _ in range(starts):
        z0 = np.array([rng.uniform(0, 10), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(-1, 1)])
        z, fv = nelder_mead(lambda z: -_g_sin(z), z0, iters=800)
        if -fv > best[0]:
            best = (-fv, z)
    a1, b1, b2, t = best[1]; t = float(np.clip(t, -0.999, 0.999)); v = np.array([t, 1 - abs(t)]); a2 = -v[0] * a1 / v[1]
    # exact: φ = f_a with a = 1 at (α, β) has G(φ) = G(linear cancelled) + ... ; with c = 0 the linear part is constant
    th = np.array([a1, b1, a2, b2])
    lo, hi = exact_gap(th, v, Act("fa", 1.0))
    out = {"kappa_dense": best[0], "kappa_exact_lo": lo, "kappa_exact_hi": hi, "alpha1": a1, "beta1": b1,
           "alpha2": a2, "beta2": b2, "v1": v[0], "v2": v[1], "starts": starts}
    pd.DataFrame([out]).to_csv(RESULTS / "width2_pilot_kappa.csv", index=False)
    print(out)


def scan():
    from .width2_conditional import directional_gplus, population, search_batch
    x, y = population()
    g = pd.read_csv(RESULTS / "width2_pilot_gamma.csv")
    rows = []
    for a in (0.5, 1.0, 1.30, 1.50):
        act = Act("fa", a)
        gh = g[g.a == a].gamma_lo.max() if (g.a == a).any() else None
        if gh is None or not np.isfinite(gh):
            gh = gamma2_multistart(act, starts=400, seed=0)["gamma_lo"]
        for R2 in (0.02, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0):
            s = 2 * R2 / gh
            r, c = search_batch(s, x, y, act, restarts=100, seed=2, batch=100)
            cap = sum(q["iters"] >= 2000 for q in c[:-1])
            if r["p"] is None:
                gp = (np.nan, np.nan); t = np.nan
            else:
                gp = directional_gplus(r["p"], r["sigma"], act); t = float(r["p"][4])
            rows.append({"a": a, "gamma2_hat": gh, "R2": R2, "s": s, "retained_loss": r["loss"],
                         "retained_constant": r["p"] is None, "Gplus_lo": gp[0], "Gplus_hi": gp[1], "t": t,
                         "cap_hits": cap, "restarts": 100})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "width2_pilot_scan.csv", index=False)


if __name__ == "__main__":
    {"identity": identity, "gamma": gamma, "kappa": kappa, "scan": scan}[sys.argv[1]]()
