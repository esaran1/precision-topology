"""§17: small-scale compactness for the sine family (width 1): conditional minimisers stay in a compact set as s -> 0
iff the class-mean gap Delta-mu attains its supremum (Track 3 (3)).  This module produces every number quoted in §17
(results/small_scale_compactness.json).  Nothing here is part of a proof except exact integer counts; the proof is in
the note.  The checks are cheap (seconds, one process).

    python -m src.small_scale
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
Q = 0.4 / 79401                      # population lattice unit (math note §11 / theorem1_hypotheses.md)
TAU = 0.4                            # half-length of the excluded x-window in the spreading lemma
PSI2 = math.log(math.cosh(1.0))      # psi(2) = log cosh 1: psi(z) >= PSI2 for |z| >= 2
A_ALL = (1.02, 1.05, 1.10, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60)


def psi(z):
    """log cosh(z/2), computed stably."""
    z = np.abs(np.asarray(z, float)) / 2
    return z + np.log1p(np.exp(-2 * z)) - math.log(2)


def _bisect_root(fn, lo, hi, iters=200):
    """Root of an increasing function on [lo, hi] (fn(lo) <= 0 <= fn(hi)) by bisection."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def profiled_loss(phi, y, s):
    """L*(s) = min_b mean softplus(z) - y z, z = s phi + b; b* is the root of mean sigmoid(z) = mean y (bisection)."""
    zmin, zmax = float(np.min(s * phi)), float(np.max(s * phi))
    ybar = float(np.mean(y))
    f = lambda b: float(np.mean(1 / (1 + np.exp(-(s * phi + b))))) - ybar
    b = _bisect_root(f, -zmax - 50, -zmin + 50)
    z = s * phi + b
    return float(np.mean(np.logaddexp(0, z) - y * z))


def V_s(phi, s):
    """min_c mean psi(s (phi - c)); the derivative in c is -s mean tanh(s (phi - c)/2)/2, increasing in c."""
    f = lambda c: float(np.mean(np.tanh(s * (c - phi) / 2)))
    c = _bisect_root(f, float(np.min(phi)) - 1, float(np.max(phi)) + 1)
    return float(np.mean(psi(s * (phi - c))))


def spreading_fraction(x, tau=TAU):
    """Largest number of points in any closed x-interval of length 2 tau (exact sliding count), and 1 - that/n."""
    xs = np.sort(x)
    j = 0
    best = 0
    for i in range(len(xs)):
        while xs[i] - xs[j] > 2 * tau + 1e-12:
            j += 1
        best = max(best, i - j + 1)
    return best, 1 - best / len(xs)


def compact_bounds(var1, a, frac, tau=TAU):
    """Theorem C constants: if some theta_1 attains sup Delta-mu with Var(phi_1) = var1, then for s < s1 every
    conditional minimiser has |w1| <= Wc (§17.2)."""
    # (frac)(5/48) s^2 (tau |w1| - a)^2 <= s^2 var1 / 8  =>  (tau |w1| - a)^2 <= 48 var1 / (40 frac) = 1.2 var1 / frac
    Wc = (a + math.sqrt(1.2 * var1 / frac)) / tau
    s1 = math.sqrt(8 * frac * PSI2 / var1)
    return Wc, s1


def population():
    from .limit_bnb import population as pop
    return pop()


def main():
    os.nice(15)
    x, y = population()
    out = {}
    # the decomposition identity, checked on random theta (not a proof; the note proves it)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        w1, b1, a, s = rng.uniform(-5, 5), rng.uniform(0, 2 * math.pi), rng.uniform(1.0, 1.6), 10 ** rng.uniform(-3, 0.5)
        phi = w1 * x + b1 + a * np.sin(w1 * x + b1)
        dmu = phi[y == 1].mean() - phi[y == 0].mean()
        worst = max(worst, abs(profiled_loss(phi, y, s) - (math.log(2) - s * dmu / 4 + V_s(phi, s))))
    out["identity_max_abs_err"] = worst
    # spreading lemma on the population and on the continuous windows
    cnt, frac = spreading_fraction(x)
    out["population"] = {"n": len(x), "max_points_in_interval_0.8": int(cnt), "spread_fraction": frac,
                         "min_spacing": float(np.min(np.diff(np.sort(x))))}
    # the lattice alias attains sup Delta-mu = 2a exactly
    n = np.round(x / Q).astype(np.int64)
    assert np.max(np.abs(x - n * Q)) < 1e-12
    alias = math.pi / Q
    varx = float(np.var(x))
    rows = []
    for a in A_ALL:
        var_A = alias ** 2 * varx + a ** 2           # Var(alpha x - pi/2 - a cos(alpha x)) with cos = +1 inner, -1 outer
        phi = alias * x - math.pi / 2 - a * np.where(y == 0, 1.0, -1.0)
        var_num = float(np.var(phi))
        Wc, s1 = compact_bounds(var_A, a, frac)
        rows.append({"a": a, "var_alias_formula": var_A, "var_alias_numeric": var_num,
                     "Wc": Wc, "s1": s1, "Wc_over_alias": Wc / alias})
    out["population"].update({"lattice_q": Q, "alias_alpha": alias, "inner_parity_even": bool((n[y == 0] % 2 == 0).all()),
                              "outer_parity_odd": bool((n[y == 1] % 2 == 1).all()), "var_x": varx, "rows": rows})
    # continuous windows: attained at alpha*_c (computer-checked, theorem1_checks.json), mass spreading 0.75 exactly
    t1 = json.loads((RESULTS / "theorem1_checks.json").read_text())
    ac = t1["H-A2_classmean_attainment_and_H-Q_quadratic_growth"]["continuous"]
    gx, gw = np.polynomial.legendre.leggauss(400)
    crow = []
    for a in A_ALL:
        def mean_on(lo, hi, f):
            xs = 0.5 * (hi - lo) * gx + 0.5 * (hi + lo)
            return (f(xs) * gw).sum() / 2
        ph = lambda xs: ac["alpha_star"] * xs + math.pi / 2 + a * np.sin(ac["alpha_star"] * xs + math.pi / 2)
        # population measure: 1/2 uniform on I, 1/2 uniform on O = +-[1.2, 2]
        m1 = 0.5 * mean_on(-0.8, 0.8, ph) + 0.25 * (mean_on(1.2, 2.0, ph) + mean_on(-2.0, -1.2, ph))
        m2 = 0.5 * mean_on(-0.8, 0.8, lambda t: ph(t) ** 2) + 0.25 * (mean_on(1.2, 2.0, lambda t: ph(t) ** 2)
                                                                    + mean_on(-2.0, -1.2, lambda t: ph(t) ** 2))
        var_c = m2 - m1 ** 2
        Wc, s1 = compact_bounds(var_c, a, 0.75)
        crow.append({"a": a, "var_star_c": var_c, "Wc": Wc, "s1": s1})
    out["continuous"] = {"alpha_star_c": ac["alpha_star"], "global_maximiser_certified_by_scan": ac["global_maximiser_certified"],
                         "spread_fraction": 0.75, "rows": crow}
    # training samples: m = E_O x - E_I x != 0, so sup Delta-mu = +inf (not attained)
    from .fold1d import make_data
    ms = []
    for seed in range(50):
        xt, yt = make_data(200, seed)
        xt, yt = xt.double().numpy(), yt.double().numpy()
        ms.append(float(xt[yt == 1].mean() - xt[yt == 0].mean()))
    out["samples"] = {"seeds": 50, "min_abs_m": float(np.min(np.abs(ms))), "median_abs_m": float(np.median(np.abs(ms))),
                      "all_nonzero": bool(np.all(np.array(ms) != 0))}
    # illustration (not a proof): on seed 0, the ramp w1 = m/(s Var x) (b1 = 0, s -> 0) beats theta* at small s
    xt, yt = make_data(200, 0)
    xt, yt = xt.double().numpy(), yt.double().numpy()
    m0 = float(xt[yt == 1].mean() - xt[yt == 0].mean())
    ill = []
    for s in (1e-2, 1e-3, 1e-4):
        a = 1.30
        w = m0 / (s * np.var(xt))
        ramp = w * xt + a * np.sin(w * xt)
        star = 1.7913244 * xt + math.pi / 2 + a * np.sin(1.7913244 * xt + math.pi / 2)
        star2 = -1.7913244 * xt + math.pi / 2 + a * np.sin(-1.7913244 * xt + math.pi / 2)
        Ls = min(profiled_loss(star, yt, s), profiled_loss(-star, yt, s), profiled_loss(star2, yt, s),
                 profiled_loss(-star2, yt, s))
        ill.append({"s": s, "ramp_w1": w, "L_ramp": profiled_loss(ramp, yt, s), "L_theta_star": Ls,
                    "ramp_lower": profiled_loss(ramp, yt, s) < Ls})
    out["samples"]["illustration_seed0"] = {"m": m0, "rows": ill}
    pr = {r_["a"]: r_ for r_ in out["population"]["rows"]}
    cr = {r_["a"]: r_ for r_ in out["continuous"]["rows"]}
    out["ledger"] = {
        "C spread count (population, interval 0.8)": out["population"]["max_points_in_interval_0.8"],
        "C spread fraction F (population)": out["population"]["spread_fraction"],
        "C alias alpha = pi/q": alias,
        "C Var(x) population": varx,
        "C Wc population a=1.30": pr[1.30]["Wc"], "C s1 population a=1.30": pr[1.30]["s1"],
        "C Wc population a=1.60": pr[1.60]["Wc"], "C s1 population a=1.60": pr[1.60]["s1"],
        "C Wc continuous a=1.30": cr[1.30]["Wc"], "C s1 continuous a=1.30": cr[1.30]["s1"],
        "C Wc continuous a=1.60": cr[1.60]["Wc"], "C s1 continuous a=1.60": cr[1.60]["s1"],
        "C samples min |m| (seeds 0-49)": out["samples"]["min_abs_m"],
        "C samples median |m| (seeds 0-49)": out["samples"]["median_abs_m"],
        "C identity max abs err": out["identity_max_abs_err"]}
    (RESULTS / "small_scale_compactness.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
