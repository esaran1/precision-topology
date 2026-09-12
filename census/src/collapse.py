"""Items 1-3: data collapse, threshold independence, held-out prediction.

Pure reanalysis of committed rate curves -- no training.  Registered in
``results/collapse_prediction.md`` before it ran.

The scaling variable is u = eps * B^theta.  Note that rescaling u is a pure
translation in log-space (log u = log eps + theta log B), so a per-budget
logistic's *slope* is invariant to theta: if slopes differ across budgets,
no choice of theta can collapse the curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def logistic_fit(log_u: np.ndarray, rate: np.ndarray,
                 refine: int = 3) -> tuple[float, float, float]:
    """Least-squares 2-parameter logistic in log u.  Returns (resid, k, mid)."""

    best = None
    for k in np.linspace(0.2, 60, 120):
        for mid in np.linspace(log_u.min() - 1, log_u.max() + 1, 120):
            r = float(np.sqrt(np.mean(
                (1 / (1 + np.exp(-k * (log_u - mid))) - rate) ** 2)))
            if best is None or r < best[0]:
                best = (r, k, mid)
    resid, k, mid = best
    for _ in range(refine):
        for kk in np.linspace(max(0.05, k * 0.6), k * 1.6, 30):
            for mm in np.linspace(mid - 0.5, mid + 0.5, 30):
                r = float(np.sqrt(np.mean(
                    (1 / (1 + np.exp(-kk * (log_u - mm))) - rate) ** 2)))
                if r < resid:
                    resid, k, mid = r, kk, mm
    return resid, k, mid


def collapse_residual(frame: pd.DataFrame, theta: float) -> float:
    """RMS residual of a single logistic in u = eps * B^theta."""

    u = frame.eps.values * frame.budget.values ** theta
    if np.any(u <= 0):
        return float("nan")
    return logistic_fit(np.log(u), frame.rate.values)[0]


def onset_at(frame: pd.DataFrame, level: float) -> float | None:
    """Interpolated onset at a crossing level, or None if unbracketed.

    Bracketed iff some eps reaches `level` and a strictly smaller eps does not.
    """

    d = frame.sort_values("eps")
    above = d[d.rate >= level]
    below = d[d.rate < level]
    if above.empty or below.empty:
        return None
    hi = above.eps.min()
    lower = below[below.eps < hi]
    if lower.empty:
        return None
    lo = lower.eps.max()
    r_lo = float(d[d.eps == lo].rate.iloc[0])
    r_hi = float(d[d.eps == hi].rate.iloc[0])
    if r_hi == r_lo:
        return float(hi)
    return float(lo + (level - r_lo) * (hi - lo) / (r_hi - r_lo))


def loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and its standard error for log y against log x."""

    lx, ly = np.log(x), np.log(y)
    sx = lx.mean()
    slope = ((lx - sx) * (ly - ly.mean())).sum() / ((lx - sx) ** 2).sum()
    intercept = ly.mean() - slope * sx
    resid = ly - (slope * lx + intercept)
    se = np.sqrt((resid ** 2).sum() / (len(lx) - 2) / ((lx - sx) ** 2).sum())
    return float(slope), float(se)
