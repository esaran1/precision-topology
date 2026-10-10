"""Track A: the CAUSAL ADAM per-run forecaster (width 1, free Adam).

Design: results/designs/trackA_causal_adam_design.md (author's decisions 2026-10-09).  src/causal_forecast.py (25aac9e)
is used UNCHANGED (GuardedArray, the cutoff, the rule point, the branch rule, R4 = lag_forecast_1d, 1C's extrapolation).
Driver: src/trackA_causal.py.

THE TWO FROZEN RULES (chosen AFTER exploration on the 120 exploration seeds 9,771,000-119 at a = 1.85, out of 7
extrapolation families; results/designs/trackA_explore/).  Their source is hashed (`rule_hashes`) before registration and
a test recomputes the hash.

  slin_20      (extrapolation) s linear in t, least squares over the last n = max(3, min(20, t_c − t_R)) visible steps
               (t_c − n … t_c − 1), anchored at s_{t_c−1}, run until ŝ ≥ 1.6·s*_frozen (inclusive) or the step budget.
               A fitted slope ≤ 0 means no forecast.
  coupled_sw   (P at the forecast switch) R4 from t_R with the VISIBLE preconditioner P_t = 1/(√v̂_t + ε) for steps
               < t_c; after the cutoff the raw second moment is carried, v ← β₂v + (1 − β₂)(Hδ)², from
               v̂_{t_c−1}(1 − β₂^{t_c−1}), v̂_t = v_t/(1 − β₂^t), P_t = 1/(√v̂_t + ε).  P̂ = 1/(√v̂(t_sw,fc) + ε), where v̂(t_sw,fc)
               is: that carried v̂ if the coupled recursion reaches t_sw,fc; the visible row v̂_{t_sw,fc} if t_sw,fc < t_c;
               otherwise (the coupled recursion stopped before t_sw,fc: a crossing, divergence or the grid end) the
               decay-only carry v̂_{t_c−1}(1 − β₂^{t_c−1})·β₂^k/(1 − β₂^{t_sw,fc}) (the exploration's rule, unchanged).
               P̂ is frozen and R4 (lag_forecast_1d) is rerun with it: the forecast crossing t_fc.

The forecaster never receives a run's arrays: GuardedArray views with cutoff t_c of w₂, M and v̂ (all 4 columns) and of
the hidden state (w₁, b₁, b₂) with at most ONE readable row (the forecaster's rule point t_R < t_c).
"""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass

import numpy as np

from . import causal_forecast as C
from . import linear_response as LR

B1, B2, EPS = LR.BETA1, LR.BETA2, LR.EPS
HID = [0, 1, 3]                          # (w₁, b₁, b₂) columns of the parameter / moment rows
SLIN_K = 20
SLIN_MIN_POINTS = 3
HORIZON = 1.6
EXTRAPOLATIONS = ("slin_20", "ext_1c")   # ext_1c: 1C's frozen quad-in-log-s, 5% / ≥ 50 window (DESCRIPTIVE only)
P_RULES = ("coupled_sw", "cut")          # cut: v̂ at t_c − 1 (DESCRIPTIVE only)
CFG_1C = C.ForecastConfig(family="quad", window_frac=0.05, min_window=50, horizon_factor=HORIZON, adam_P="cutoff")


# ================================================================================================ FROZEN RULE 1: slin_20
def slin_extrapolate(s_vis, t_R, s_frozen, budget_left, K=SLIN_K, horizon=HORIZON):
    """(ŝ for t_c, t_c + 1, …; status; info).  s linear in t over the last max(3, min(K, t_c − t_R)) visible steps,
    anchored at s_{t_c−1}; stops at the first ŝ ≥ horizon·s_frozen (inclusive) or after the budget."""
    s_vis = np.asarray(s_vis, float)
    t_c = len(s_vis)
    n = max(SLIN_MIN_POINTS, min(K, t_c - t_R))
    w = np.asarray(s_vis[t_c - n:t_c], float)
    info = {"n_window": int(n)}
    tau = np.arange(-len(w) + 1, 1, dtype=float)
    s_stop = horizon * s_frozen
    sc = max(1.0, float(abs(tau).max()))
    c = np.polyfit(tau / sc, w, 1)
    rate = np.polyval(np.polyder(c), 0.0) / sc
    info["slope"] = float(rate)
    if rate <= 0:
        return np.array([]), "no forecast: not growing at the cutoff", info
    need = (s_stop - w[-1]) / rate
    n_ah = int(max(1, min(budget_left, math.ceil(4 * max(need, 1.0)) + 10)))
    k = np.arange(1, n_ah + 1, dtype=float)
    sh = w[-1] + np.polyval(c, k / sc) - np.polyval(c, 0.0)
    hit = np.nonzero(sh >= s_stop)[0]
    if len(hit):
        sh = sh[:hit[0] + 1]
    if len(sh) == 0:
        return np.array([]), "no forecast: empty extrapolation", info
    info["n_extrapolated"] = int(len(sh))
    return sh, "ok", info


# ================================================================================================ FROZEN RULE 2: coupled_sw
def coupled_vhat(s_ext, t0, B, a, delta0, lr, m0, P_vis, v_raw0, t_c):
    """R4 from t0 with the visible P rows before t_c and, after it, v̂ carried with g = Hδ (raw second moment from
    v_raw0 = v̂_{t_c−1}(1 − β₂^{t_c−1}), bias-corrected by the step count).  Returns (t_hit | None, status, {t: v̂ (3
    hidden coordinates) for every carried step t ≥ t_c reached})."""
    seg = np.asarray(s_ext, float)[t0:]
    inside = np.array([B.contains(v) for v in seg])
    T_end = len(seg) if inside.all() else int(np.argmin(inside))
    if T_end < 2:
        return None, "no forecast: rule point outside the branch grid", {}
    seg = seg[:T_end]
    th = [np.asarray(B.theta(v), float) for v in seg[:1]]
    d = np.asarray(delta0, float).copy()
    m = np.asarray(m0, float).copy()
    v = None
    vh = {}
    gap = lambda i, dd: LR.gap_exact(th[i][0] + dd[0], th[i][1] + dd[1], a)  # noqa: E731
    if gap(0, d) > 0:
        return t0, "ok", vh
    rad = 0.0
    for i in range(T_end - 1):
        t_next = t0 + i + 1
        if len(th) < i + 2:
            th.append(np.asarray(B.theta(seg[i + 1]), float))
        H = np.asarray(B.hess(seg[i]), float)
        g = H @ d
        m = B1 * m + (1 - B1) * g
        if t_next < t_c:
            P = P_vis[t_next]
        else:
            if v is None:
                v = v_raw0.copy()
            v = B2 * v + (1 - B2) * g ** 2
            vhat = v / (1 - B2 ** t_next)
            vh[t_next] = vhat
            P = 1.0 / (np.sqrt(vhat) + EPS)
        if i % 10 == 0:
            rad = max(rad, LR.step_map_radius(H, P, lr, True))
        d = d - lr * P * (m / (1 - B1 ** t_next)) - (th[i + 1] - th[i])
        if not np.all(np.isfinite(d)) or np.abs(d).max() > LR.DIVERGED:
            return None, "no forecast: unstable / diverged (Track 1 rule)", vh
        if gap(i + 1, d) > 0:
            if rad > 1.0:
                return None, "no forecast: unstable / diverged (Track 1 rule)", vh
            return t0 + i + 1, "ok", vh
    return None, "no forecast: no predicted crossing on the extrapolated path", vh


def coupled_sw_vhat(t_sw_fc, vh, Vg, t_c):
    """v̂ (4 columns; the hidden ones used) at the forecast switch: the coupled carry, the visible row, or the decay-only
    carry from v̂_{t_c−1}.  Returns (v̂, source)."""
    if t_sw_fc in vh:
        vhat = np.zeros(4)
        vhat[HID] = vh[t_sw_fc]
        return vhat, "coupled"
    if t_sw_fc < t_c:
        return C._row(Vg, t_sw_fc), "visible"
    last = t_c - 1
    v_last = C._row(Vg, last)
    k = t_sw_fc - last
    return v_last * (1 - B2 ** last) * B2 ** k / (1 - B2 ** t_sw_fc), "decay"


FROZEN_RULES = (slin_extrapolate, coupled_vhat, coupled_sw_vhat)


def rule_source():
    """The source text of the frozen rules (slin_20, coupled_sw) and their constants, as hashed."""
    consts = (f"SLIN_K={SLIN_K!r}\nSLIN_MIN_POINTS={SLIN_MIN_POINTS!r}\nHORIZON={HORIZON!r}\nHID={HID!r}\n"
              f"B1={B1!r}\nB2={B2!r}\nEPS={EPS!r}\n")
    return consts + "".join(inspect.getsource(f) for f in FROZEN_RULES)


def rule_hashes():
    """SHA-256 of the frozen rules' source (each function and all together, with the constants)."""
    out = {f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in FROZEN_RULES}
    out["all_with_constants"] = hashlib.sha256(rule_source().encode()).hexdigest()
    return out


# ================================================================================================ the forecast
@dataclass
class AdamInputs:
    """out: signed w₂ path; hid: (w₁, b₁, b₂) path (at most one row read); M, Vhat: Adam's first moment and
    bias-corrected second moment (4 columns).  Guarded views (cutoff t_c) in a forecast; plain arrays with every row
    ≥ t_c NaN in the NaN recomputation."""
    a: float
    x: np.ndarray
    y: np.ndarray
    s_frozen: float
    t_c: int
    budget: int
    out: object
    hid: object
    M: object
    Vhat: object
    lr: float = LR.LR_ADAM


def _no(status, info, t_c, guards, s_switch=float("nan")):
    return _record({"status": status, "s_switch": s_switch, "t_hit": None}, None, t_c, info, guards)


def _record(res, s_ext, t_c, info, guards):
    out = {"t_c": int(t_c), **info, "status": res["status"]}
    out["max_index_read"] = {g.name: g.max_index_read for g in guards if isinstance(g, C.GuardedArray)}
    for k in ("traj_radius_max", "traj_max_abs_delta", "segment_steps"):
        if k in res:
            out[k] = res[k]
    t_hit = res.get("t_hit")
    if t_hit is None:
        out.update(t_fc=None, s_fc=float("nan"), r_fc=float("nan"), lag_steps_fc=None)
        return out
    s_sw = info["s_run"]
    out.update(t_fc=int(t_hit), s_fc=float(s_ext[t_hit]), r_fc=float(s_ext[t_hit] / s_sw - 1),
               lag_steps_fc=int(t_hit) - int(info["t_sw_fc"]), hit_in_extrapolated_part=bool(t_hit >= t_c))
    return out


def forecast_adam(inp: AdamInputs, ext="slin_20", prule="coupled_sw"):
    """ONE causal forecast at the cutoff inp.t_c.  ext ∈ EXTRAPOLATIONS, prule ∈ P_RULES; the registered primary is
    (slin_20, coupled_sw).  Reads rows < t_c only (and one hidden row, the rule point)."""
    if ext not in EXTRAPOLATIONS or prule not in P_RULES:
        raise ValueError((ext, prule))
    t_c = int(inp.t_c)
    guards = (inp.out, inp.hid, inp.M, inp.Vhat)
    w2 = C._prefix(inp.out, t_c)
    s = np.abs(w2)
    info = {"ext": ext, "P_rule": prule}
    tR = C.causal_rule_step(s, inp.s_frozen)
    if tR is None:
        return _no("no forecast: no rule point before the cutoff", info, t_c, guards)
    flip = LR.D_FLIP if w2[tR] < 0 else np.ones(3)
    TH = C._row(inp.hid, tR) * flip
    info.update(t_rule=int(tR), s_rule=float(s[tR]))
    B, s_run = C.w1_branch(inp.a, inp.x, inp.y, inp.s_frozen, float(s[tR]), TH)
    if B is None:
        return _no("no forecast: no minimum at the rule point", info, t_c, guards)
    if s_run is None:
        return _no("no forecast: occupied branch has no switch", info, t_c, guards)
    info["s_run"] = float(s_run)
    m0 = C._row(inp.M, tR)[HID] * flip
    budget_left = max(0, inp.budget - t_c + 1)
    if ext == "slin_20":
        s_hat, st, xinfo = slin_extrapolate(s, tR, inp.s_frozen, budget_left)
    else:
        s_hat, st, xinfo = C.extrapolate_scale(s, CFG_1C, inp.s_frozen, budget_left, t_start=tR)
    info.update({f"ext_{k}": v for k, v in xinfo.items() if k != "t_c"})
    if st != "ok":
        return _no(st, info, t_c, guards, s_run)
    s_ext = np.concatenate([s, s_hat])
    t_sw_fc = LR._first_ge(s_ext, s_run)
    info["t_sw_fc"] = t_sw_fc
    if t_sw_fc is None:
        return _no("no forecast: no forecast switch on the extrapolated path", info, t_c, guards, s_run)
    info["s_sw_fc"] = float(s_ext[t_sw_fc])
    delta0 = TH - B.theta(float(s[tR]))
    if prule == "coupled_sw":
        P_vis = {t: 1.0 / (np.sqrt(C._row(inp.Vhat, t)[HID]) + EPS) for t in range(tR + 1, t_c)}
        v_raw0 = C._row(inp.Vhat, t_c - 1)[HID] * (1 - B2 ** (t_c - 1))
        t_cpl, st_cpl, vh = coupled_vhat(s_ext, tR, B, inp.a, delta0, inp.lr, m0, P_vis, v_raw0, t_c)
        info.update(coupled_t_hit=t_cpl, coupled_status=st_cpl)
        vhat, src = coupled_sw_vhat(t_sw_fc, vh, inp.Vhat, t_c)
    else:
        vhat, src = C._row(inp.Vhat, t_c - 1), "cutoff"
    P = 1.0 / (np.sqrt(vhat[HID]) + EPS)
    info.update(P_source=src, P_hat=[float(v) for v in P])
    gapf = lambda th: LR.gap_exact(th[0], th[1], inp.a)  # noqa: E731
    bc1 = lambda t: 1 - B1 ** t  # noqa: E731
    r = C.lag_forecast_1d(s_ext, tR, B.theta, B.hess, B.contains, gapf, delta0, inp.lr, s_run, P=P, m0=m0, bc1_of=bc1)
    return _record({**r, "t_hit": r.get("t_hit")}, s_ext, t_c, info, guards)


def guarded_inputs(a, x, y, s_frozen, t_c, budget, W, M, V):
    """The forecaster's inputs from a run's full arrays: GuardedArray views (cutoff t_c) of w₂, M, v̂ and of the hidden
    state with at most one readable row."""
    return AdamInputs(a=a, x=x, y=y, s_frozen=s_frozen, t_c=int(t_c), budget=budget,
                      out=C.guard(W[:, 2], t_c, name="w2"), hid=C.guard(W[:, HID], t_c, max_rows=1, name="hidden"),
                      M=C.guard(M, t_c, name="M"), Vhat=C.guard(V, t_c, name="Vhat"))


def nan_inputs(a, x, y, s_frozen, t_c, budget, W, M, V):
    """The NaN recomputation's inputs: plain copies with every row ≥ t_c set to NaN."""
    def cut(A):
        A = np.array(A, float, copy=True)
        A[int(t_c):] = np.nan
        return A
    Wn = cut(W)
    return AdamInputs(a=a, x=x, y=y, s_frozen=s_frozen, t_c=int(t_c), budget=budget, out=Wn[:, 2], hid=Wn[:, HID],
                      M=cut(M), Vhat=cut(V))


def run_forecast(inp: AdamInputs, ext="slin_20", prule="coupled_sw"):
    """The forecast with every guard's last row asserted < t_c."""
    out = forecast_adam(inp, ext, prule)
    for name, mi in out["max_index_read"].items():
        if mi >= inp.t_c:
            raise C.CausalityViolation(f"{name}: row {mi} read with cutoff {inp.t_c}")
    return out
