"""Phase 2A: a STRICTLY CAUSAL forecast of the fold crossing (design: results/designs/phase2a_slow_sb_design.md, approved
2026-10-04; registration: results/phase2a_registration.md).

BINDING RULE.  The forecast reads the output scale s only, through a GuardedArray view with cutoff t_c (rows ≥ t_c raise
CausalityViolation), and uses the FROZEN Phase 1A extrapolation (`src/causal_forecast.py`, 25aac9e:
`extrapolate_scale`, family quad, window ⌈0.05·t_c⌉ ≥ 50 steps, horizon 1.6·s_ref), unchanged.  It reads no parameter
row, no ρ₂ and no gradient.  tests/test_phase2a.py fails if it reads a row at or after its cutoff, and checks that its
output is identical when every row ≥ t_c is NaN.

THE FORECAST (the rule of `p2a_explore_f`, the page's "t̂_F, ṡ̂_F → ε̂_F, ŝ_c, t_fc"):
  ŝ_k, k = 0, 1, …   = extrapolate_scale(s_0 … s_{t_c−1}, quad / 0.05 / 50, s_ref = s_F, horizon 1.6·s_F); ŝ_k is the
                       forecast of s at step t_c + k (anchored at s_{t_c−1}); the path stops at the first ŝ ≥ 1.6·s_F
  t̂_F = t_c + k_F    k_F = the first k with ŝ_k ≥ s_F  (the no-delay forecast)
  ṡ̂_F = ŝ_{k_F} − ŝ_{k_F−1}   (ŝ_{−1} := s_{t_c−1}): the extrapolated one-step increment at t̂_F
  ε̂_F = (ṡ̂_F/s_F)/(ηΛ_F),  r_fc = Ω₀·ε̂_F^{2/3},  ŝ_c = s_F(1 + r_fc)
  t_fc = t_c + k_c    k_c = the first k with ŝ_k ≥ ŝ_c
  delay_fc = t_fc − t̂_F
NO FORECAST (status ≠ "ok"; t_fc = t̂_F = None) when the frozen extrapolation returns no path (window unusable, not
growing at the cutoff), or no ŝ ≥ s_F, or no ŝ ≥ ŝ_c, within the horizon 1.6·s_F.
"""

from __future__ import annotations

import math

import numpy as np

from . import causal_forecast as CF

OMEGA0 = 2.338107410459767          # −(first zero of Ai): the fold delay constant
FAMILY, WINDOW_FRAC, MIN_WINDOW, HORIZON = "quad", 0.05, 50, 1.6


def config(f):
    """The frozen Phase 1A configuration (quad, 5% window, ≥ 50 steps, horizon 1.6·s_ref) at cutoff fraction f."""
    return CF.ForecastConfig(f=float(f), family=FAMILY, window_frac=WINDOW_FRAC, min_window=MIN_WINDOW,
                             horizon_factor=HORIZON)


def _none(status, t_c, info, s_view):
    return {"status": status, "t_c": int(t_c), "t_F_fc": None, "t_fc": None, "delay_fc": None, "eps_fc": None,
            "r_fc": None, "s_c_fc": None, "sdot_F_fc": None, **info,
            "max_index_read": s_view.max_index_read if isinstance(s_view, CF.GuardedArray) else None}


def forecast_fold(s_view, t_c, s_F, Lambda_F, eta, budget_left, f):
    """The causal fold forecast from s_0 … s_{t_c−1}.  s_view: a GuardedArray with cutoff t_c (or, for the NaN
    recomputation, a plain array).  Returns a JSON-able dict (times are absolute steps)."""
    t_c = int(t_c)
    s = CF._prefix(s_view, t_c)                    # rows 0 … t_c − 1: the only read
    s_hat, st, info = CF.extrapolate_scale(s, config(f), s_F, int(budget_left))
    info = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in info.items()}
    if st != "ok":
        return _none(st, t_c, info, s_view)
    hitF = np.nonzero(s_hat >= s_F)[0]
    if not len(hitF):
        return _none("no forecast: no extrapolated s reaches s_F within 1.6·s_F", t_c, info, s_view)
    kF = int(hitF[0])
    prev = s_hat[kF - 1] if kF > 0 else s[-1]
    sdot = float(s_hat[kF] - prev)
    eps = (sdot / s_F) / (eta * Lambda_F)
    r_fc = OMEGA0 * eps ** (2.0 / 3.0)
    s_c = s_F * (1.0 + r_fc)
    hitc = np.nonzero(s_hat >= s_c)[0]
    if not len(hitc):
        return {**_none("no forecast: no extrapolated s reaches s_c within 1.6·s_F", t_c, info, s_view),
                "t_F_fc": t_c + kF, "eps_fc": float(eps), "r_fc": float(r_fc), "s_c_fc": float(s_c),
                "sdot_F_fc": sdot, "t_fc": None, "delay_fc": None}
    kc = int(hitc[0])
    return {"status": "ok", "t_c": t_c, "t_F_fc": t_c + kF, "t_fc": t_c + kc, "delay_fc": kc - kF,
            "eps_fc": float(eps), "r_fc": float(r_fc), "s_c_fc": float(s_c), "sdot_F_fc": sdot, **info,
            "max_index_read": s_view.max_index_read if isinstance(s_view, CF.GuardedArray) else None}


def has_forecast(rec):
    return rec is not None and rec.get("status") == "ok" and rec.get("t_fc") is not None


def same_forecast(a, b):
    """Every output field identical (None equal to None; the guard bookkeeping `max_index_read` ignored)."""
    ka = set(a) - {"max_index_read"}
    if ka != set(b) - {"max_index_read"}:
        return False
    for k in ka:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


def run_forecast(s_path, t_c, s_F, Lambda_F, eta, budget_left, f):
    """HARNESS: the forecast from a GUARDED view of s with cutoff t_c (the caller passes a path that may extend past
    t_c), with the guard's last row asserted < t_c, and the NaN recomputation (every row ≥ t_c set to NaN, unguarded)
    asserted identical.  Returns (record, nan_identical)."""
    s_path = np.asarray(s_path, float)
    g = CF.GuardedArray(s_path, t_c, name="s")
    rec = forecast_fold(g, t_c, s_F, Lambda_F, eta, budget_left, f)
    if g.max_index_read >= t_c or g.violations:
        raise CF.CausalityViolation(f"s: row {g.max_index_read} read with cutoff {t_c}")
    poisoned = s_path.copy()
    poisoned[t_c:] = np.nan
    rec2 = forecast_fold(poisoned, t_c, s_F, Lambda_F, eta, budget_left, f)
    same = same_forecast(rec, rec2)
    rec["max_index_read"] = int(g.max_index_read)
    rec["nan_recompute_identical"] = bool(same)
    return rec, same
