"""Convergence of R_glob(a) to R_glob^inf, refitted on certified intervals (replaces the frozen-grid fit).

Inputs: certified finite-a brackets (cond_certified_brackets.csv, kind glob, R = s Ĝ_cert/2) and the certified
limit interval (limit_switch.csv).  dev(a) = R_glob(a)/R_glob^inf − 1, ε = a − 1.

1. Midpoint fit: log-log slope of dev on ε, c1 = Σε·dev/Σε² (R = R_inf(1 + c1 ε)), max residual, corr.
2. Exact ranges over the certified box.  For fixed R_inf, the OLS slope is linear in y_i = log dev_i and each
   y_i is increasing in R_i, so its extremes are attained at interval endpoints chosen by the sign of the
   OLS weight; R_inf is scanned on 4001 points of its interval.  c1 has positive weights: extremes at the
   corners.
3. Feasibility over the certified box: does some R_inf in its interval and some law R = R_inf + m ε (one O(ε)
   term), or R = R_inf + m ε + q ε² (plus O(ε²)), pass through all six certified intervals?  Exact for each
   R_inf on a 4001-point scan: m is an interval (linear case); for the quadratic case m is scanned on 4001
   points of its a-priori range and q is then an interval.
4. Limit-free check: R(a) = α + βε fitted to the six finite a alone; α (and its exact range over the box,
   same endpoint selection) compared with the certified limit interval.

    python -m src.rglob_refit
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _ols_weights(x):
    xc = x - x.mean()
    return xc / (xc @ xc)


def _extreme(weights, lo, hi, sense):
    pick_hi = weights > 0 if sense == "max" else weights < 0
    return float(weights @ np.where(pick_hi, hi, lo))


def main():
    b = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    b = b[b.kind == "glob"].sort_values("a")
    e = (b.a - 1).values
    Rlo, Rhi = b.R_lo.values, b.R_hi.values
    Rmid = 0.5 * (Rlo + Rhi)
    ls = pd.read_csv(RESULTS / "limit_switch.csv").iloc[0]
    Llo, Lhi = float(ls.R_glob_inf_lo), float(ls.R_glob_inf_hi)
    Lmid = 0.5 * (Llo + Lhi)
    le = np.log(e)
    w = _ols_weights(le)

    dev = Rmid / Lmid - 1
    slope_mid = float(w @ np.log(dev))
    c1_mid = float(e @ dev / (e @ e))
    resid_mid = float(np.abs(Rmid - Lmid * (1 + c1_mid * e)).max())
    corr_mid = float(np.corrcoef(e, dev)[0, 1])

    smin, smax = np.inf, -np.inf
    for L in np.linspace(Llo, Lhi, 4001):
        ylo, yhi = np.log(Rlo / L - 1), np.log(Rhi / L - 1)
        smin = min(smin, _extreme(w, ylo, yhi, "min"))
        smax = max(smax, _extreme(w, ylo, yhi, "max"))
    c1_min = float(e @ (Rlo / Lhi - 1) / (e @ e))
    c1_max = float(e @ (Rhi / Llo - 1) / (e @ e))

    X = np.vstack([np.ones_like(e), e]).T
    alpha_w = np.linalg.pinv(X)[0]                  # α = alpha_w · R
    alpha_mid = float(alpha_w @ Rmid)
    beta_mid = float(np.linalg.pinv(X)[1] @ Rmid)
    a_min, a_max = _extreme(alpha_w, Rlo, Rhi, "min"), _extreme(alpha_w, Rlo, Rhi, "max")
    lin_resid = float(np.abs(Rmid - (alpha_mid + beta_mid * e)).max())

    Ls = np.linspace(Llo, Lhi, 4001)
    lin_ok = [L for L in Ls if ((Rlo - L) / e).max() <= ((Rhi - L) / e).min()]
    quad_ok = []
    for L in Ls[::40]:
        for m in np.linspace(0.0, 0.15, 4001):
            qlo = ((Rlo - L - m * e) / e ** 2).max(); qhi = ((Rhi - L - m * e) / e ** 2).min()
            if qlo <= qhi:
                quad_ok.append((L, m, 0.5 * (qlo + qhi)))
    quad_ms = [m for _, m, _ in quad_ok]
    quad_qs = [q for _, _, q in quad_ok]
    quad_c1 = [m / L for L, m, _ in quad_ok]

    rows = [
        ("R_glob_inf certified lo", Llo), ("R_glob_inf certified hi", Lhi),
        ("finite-a certified R width max", float((Rhi - Rlo).max())),
        ("dev min (%), midpoints", float(dev.min() * 100)), ("dev max (%), midpoints", float(dev.max() * 100)),
        ("dev all positive (worst corner)", float((Rlo / Lhi - 1 > 0).all())),
        ("dev monotone in a, midpoints", float(np.all(np.diff(dev) > 0))),
        ("corr(eps, dev), midpoints", corr_mid),
        ("log-log slope, midpoints", slope_mid),
        ("log-log slope, min over certified box", smin), ("log-log slope, max over certified box", smax),
        ("slope range inside registered band [0.5, 2]", float(0.5 <= smin and smax <= 2)),
        ("c1, midpoints", c1_mid), ("c1 min over box", c1_min), ("c1 max over box", c1_max),
        ("c1 fit max residual, midpoints", resid_mid),
        ("one-term law R_inf(1 + c1 eps) feasible within all certified intervals", float(len(lin_ok) > 0)),
        ("two-term law (eps and eps^2) feasible within all certified intervals", float(len(quad_ok) > 0)),
        ("two-term law: m (= R_inf c1) min over feasible", min(quad_ms) if quad_ok else np.nan),
        ("two-term law: m max over feasible", max(quad_ms) if quad_ok else np.nan),
        ("two-term law: c1 = m/R_inf min over feasible", min(quad_c1) if quad_ok else np.nan),
        ("two-term law: c1 max over feasible", max(quad_c1) if quad_ok else np.nan),
        ("two-term law: q (eps^2 coefficient) min over feasible", min(quad_qs) if quad_ok else np.nan),
        ("two-term law: q max over feasible", max(quad_qs) if quad_ok else np.nan),
        ("limit-free linear fit: intercept alpha, midpoints", alpha_mid),
        ("limit-free intercept min over box", a_min), ("limit-free intercept max over box", a_max),
        ("limit-free slope beta, midpoints", beta_mid),
        ("limit-free linear fit max residual, midpoints", lin_resid),
        ("limit-free intercept range overlaps certified limit", float(a_max >= Llo and a_min <= Lhi)),
        ("limit-free intercept midpoint inside certified limit", float(Llo <= alpha_mid <= Lhi)),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value"])
    out.to_csv(RESULTS / "rglob_convergence_refit.csv", index=False)
    pts = pd.DataFrame({"a": b.a.values, "eps": e, "R_lo": Rlo, "R_hi": Rhi, "R_mid": Rmid,
                        "dev_mid_pct": dev * 100, "dev_lo_pct": (Rlo / Lhi - 1) * 100,
                        "dev_hi_pct": (Rhi / Llo - 1) * 100})
    pts.to_csv(RESULTS / "rglob_convergence_points.csv", index=False)
    print(pts.to_string(index=False))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
