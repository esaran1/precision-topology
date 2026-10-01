"""Phase 1A (author's program of 2026-09-30): a STRICTLY CAUSAL forecaster of the crossing step and the signed lag.

Not a registration.  Code, tests and a pilot-seed error estimate only (driver: src/phase1a_pilot.py, results/phase1a/).

BINDING RULE.  Every forecast reads only data from BEFORE a declared cutoff t_c.  The forecaster never receives a run's
arrays: it receives GuardedArray views.  A guard raises CausalityViolation on any read of a row at or after its cutoff
(output path; Adam moments), or of any hidden-state row other than the one it is allowed to read (release, or the rule
point, at most one row and strictly before t_c).  tests/test_causal_forecast.py checks (a) that a deliberately leaky
forecaster is caught, (b) that the real forecasters pass, (c) that their output is identical when every row at or after
t_c is NaN (or garbage), (d) the same for every auxiliary input.

THE CUTOFF (a stopping time).  t_c = the first step t ≥ 1 with s_t ≥ f·s_ref, s_ref = the run's FROZEN own-branch switch
(width 1: s*_frozen; GELU-T: the occupied copy's frozen s_switch; W2-A: the occupied copy's frozen s_switch).  The
harness locates t_c (`cutoff_step`, which compares s_{t_c} with a frozen level and passes on only the integer t_c); the
forecaster reads rows 0 … t_c − 1.  f is a parameter (the final f is the author's decision).

THE FORECAST (frozen rule, `ForecastConfig`).
 (i)   Extrapolation of the output path past the cutoff.  Window = the last n_w = max(min_window, ⌈window_frac·t_c⌉)
       visible steps (t_c − n_w … t_c − 1).  y_t = log s_t.  Families:
         lin   y linear in t (constant growth rate), least squares over the window;
         quad  y quadratic in t;
         rate  the one-step growth rate y_{t+1} − y_t linear in y_t (an autonomous local model of the slaved output
               dynamics, ds/dt = F(s)), iterated forward.
       Every family is ANCHORED at the last visible point (ŷ = y_{t_c−1} + p(τ) − p(0)), so the path is continuous at t_c.
       Width 2 (W2-A): s = ‖v‖₁ as above and the share u = |v₁|/s (lin, quad: the same polynomial family in t; rate: u
       linear in y), anchored; v̂ = sign(v_{t_c−1})⊙ŝ·(û, 1 − û).  The extrapolation runs until ŝ ≥ horizon_factor·s_ref
       or the run's step budget.  No forecast if the extrapolated path does not grow at t_c.
 (ii)  The linear-response lag (Track A's R4 recursion, linear_response.simulate / width2_asym.r4_recursion, unchanged)
       integrated along [actual visible path, extrapolated path], from the release state (GELU-T, W2-A) or the causal rule
       point (width 1), using only frozen landscape quantities (the occupied branch θ*(s), H(s) on the own sample) and
       that one state.  W2-A T and T′: the lag-free switch is width2_asym.own_path_switch along the extrapolated v path;
       W2-A D, GELU-T and width 1: the first extrapolated step with ŝ ≥ the frozen switch (width 1: s*_run).
 (iii) Output: the forecast crossing step t_fc, s_fc, the forecast switch t_sw,fc and s_sw,fc, the signed lag in steps
       (t_fc − t_sw,fc) and in scale (r_fc = s_fc/s_sw,fc − 1), and the last row each guard saw.

Width 1, causal rule point: t_R = the first step ≥ 1 with s_t ≥ 0.5·s*_frozen at every t_R ≤ t < t_c (Track A's rule with
"before first reaching s*_frozen" replaced by "before the cutoff"; it reads s_1 … s_{t_c−1}).  Adam: P frozen at
1/(√v̂_{t_R} + ε) and m₀ = m_{t_R} (Track A), or with adam_P = "cutoff" at v̂_{t_c−1} (the causal analogue of 2A's P at
t_sw); bias correction 1 − β₁^t from the step count (not a state).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from . import linear_response as LR

FAMILIES = ("lin", "quad", "rate")


# ================================================================================================ the guard
class CausalityViolation(RuntimeError):
    """A forecaster read a row it is not allowed to read."""


class GuardedArray:
    """Read-only, access-checked view of a run array (rows = steps).

    cutoff     rows ≥ cutoff are never readable; len(view) = cutoff and negative indices count from the cutoff (the view
               behaves like the truncated array a[:cutoff]).  A slice whose explicit stop exceeds the cutoff is a violation
               (it asks for rows the view does not have), not a silent truncation.
    rows       if given, the only readable rows (hidden state: {release} or {rule point}).
    max_rows   if given, at most this many DISTINCT rows may be read (hidden state at the rule point: 1, chosen by the
               forecaster, strictly before the cutoff).
    Every read is recorded (max_index_read, rows_read); a violation raises CausalityViolation and is recorded."""

    def __init__(self, data, cutoff, rows=None, max_rows=None, name="array"):
        a = np.array(data, dtype=np.float64, copy=True)
        a.setflags(write=False)
        self.__data = a
        self.cutoff = int(cutoff)
        if not 0 <= self.cutoff <= len(a):
            raise ValueError("cutoff outside the array")
        self.allowed = None if rows is None else frozenset(int(r) for r in rows)
        self.max_rows = max_rows
        self.name = name
        self.rows_read: set[int] = set()
        self.max_index_read = -1
        self.violations: list[str] = []

    def __len__(self):
        return self.cutoff

    @property
    def shape(self):
        return (self.cutoff,) + self.__data.shape[1:]

    @property
    def ndim(self):
        return self.__data.ndim

    def _violate(self, msg):
        self.violations.append(msg)
        raise CausalityViolation(f"{self.name}: {msg}")

    def _row_index(self, key):
        """Absolute row indices requested by `key` (int, slice, list/array of ints) and whether it was a scalar."""
        n = self.cutoff
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i >= n:
                self._violate(f"row {i} read; cutoff {n}")
            if i < 0:
                i += n
                if i < 0:
                    self._violate(f"row {key} out of the view")
            return np.array([i]), True
        if isinstance(key, slice):
            for b in (key.start, key.stop):
                if b is not None and int(b) > n:
                    self._violate(f"slice bound {b} beyond cutoff {n}")
            return np.arange(*key.indices(n)), False
        if key is Ellipsis:
            return np.arange(n), False
        idx = np.asarray(key)
        if idx.dtype == bool:
            if len(idx) > n and idx[n:].any():
                self._violate("boolean mask selects rows at/after the cutoff")
            idx = np.nonzero(idx[:n])[0]
        idx = idx.astype(np.int64).ravel()
        if (idx >= n).any():
            self._violate(f"row {int(idx.max())} read; cutoff {n}")
        idx = np.where(idx < 0, idx + n, idx)
        if (idx < 0).any():
            self._violate("negative row out of the view")
        return idx, False

    def _record(self, idx):
        if self.allowed is not None:
            bad = [int(i) for i in idx if int(i) not in self.allowed]
            if bad:
                self._violate(f"row {bad[0]} is not an allowed row {sorted(self.allowed)}")
        if self.max_rows is not None:
            new = self.rows_read | {int(i) for i in idx}
            if len(new) > self.max_rows:
                self._violate(f"{len(new)} distinct rows read; at most {self.max_rows} allowed")
        self.rows_read |= {int(i) for i in idx}
        if len(idx):
            self.max_index_read = max(self.max_index_read, int(idx.max()))

    def __getitem__(self, key):
        rest = ()
        if isinstance(key, tuple):
            key, rest = key[0], key[1:]
        idx, scalar = self._row_index(key)
        self._record(idx)
        out = self.__data[idx]
        if scalar:
            out = out[0]
        if rest:
            out = out[(slice(None),) + rest] if not scalar else out[rest]
        return np.array(out, dtype=np.float64, copy=True)

    def row(self, t):
        return self[int(t)]

    def __array__(self, dtype=None, copy=None):
        out = self[:]
        return out.astype(dtype) if dtype is not None else out

    def __iter__(self):
        for i in range(self.cutoff):
            yield self[i]


def guard(data, cutoff, rows=None, max_rows=None, name="array"):
    return GuardedArray(data, cutoff, rows=rows, max_rows=max_rows, name=name)


def _prefix(a, t_c):
    """Rows 0 … t_c − 1 of a guarded view or a plain array (the only way the forecasters read a path)."""
    out = np.array(a[:t_c], dtype=np.float64, copy=True)
    if len(out) != t_c:
        raise ValueError("path shorter than the cutoff")
    return out


def _row(a, t):
    return np.array(a[int(t)], dtype=np.float64, copy=True)


def max_index_read(*arrs):
    return max((a.max_index_read for a in arrs if isinstance(a, GuardedArray)), default=-1)


# ================================================================================================ the cutoff
def cutoff_step(s_path, level, t_from=1):
    """HARNESS SIDE (a stopping time): the first step t ≥ t_from with s_t ≥ level, None if never.  Deciding to stop at
    t reads s_t only; nothing at or after the returned step is passed to a forecaster."""
    s = np.asarray(s_path, float)
    idx = np.nonzero(s[t_from:] >= level)[0]
    return None if len(idx) == 0 else int(idx[0] + t_from)


def w1_cutoff_on_s_run(s_path, w2_sign, f, s_frozen, a, x, y, hidden_row):
    """HARNESS SIDE, width 1 (a nested stopping time): the first step t with s_t ≥ f·s*_run(t), where s*_run(t) is the
    switch of the branch occupied at the causal rule point t_R(t) = causal_rule_step(s_0 … s_{t−1}) (Track A's branch
    rule; hidden state read at t_R(t) < t only).  Returns (t_c, t_R, s*_run) or (None, None, None).  Deciding to stop
    at t reads s_0 … s_t and the hidden state at t_R(t); the forecaster then recomputes t_R and s*_run from rows
    < t_c."""
    s = np.asarray(s_path, float)
    lvl = RULE_FRAC * s_frozen
    last_i, cache = None, {}               # last i in 1 … t − 1 with s_i < lvl (causal_rule_step's "below")
    for t in range(1, len(s)):
        if t - 1 >= 1 and s[t - 1] < lvl:
            last_i = t - 1
        if t < 2 or s[t - 1] < lvl:
            continue                       # causal_rule_step(s[:t]) is None
        tR = last_i + 1 if last_i is not None else 1
        if tR not in cache:
            flip = LR.D_FLIP if w2_sign(tR) < 0 else np.ones(3)
            B, s_run = w1_branch(a, x, y, s_frozen, float(s[tR]), np.asarray(hidden_row(tR), float) * flip)
            cache[tR] = s_run
        if cache[tR] is not None and s[t] >= f * cache[tR]:
            return t, tR, cache[tR]
    return None, None, None


# ================================================================================================ configuration
@dataclass(frozen=True)
class ForecastConfig:
    f: float = 0.9                 # cutoff fraction of the frozen own-branch switch
    family: str = "quad"           # extrapolation family (FAMILIES); quad/0.05 chosen on the Phase 1A dev seeds
    window_frac: float = 0.05      # window = the last ⌈window_frac·t_c⌉ visible steps …
    min_window: int = 50           # … at least this many
    horizon_factor: float = 1.6    # extrapolate until ŝ ≥ horizon_factor·s_ref (or the step budget)
    adam_P: str = "rule_point"     # width-1 Adam: P at the rule point (Track A) or at t_c − 1 ("cutoff")

    def with_f(self, f):
        return replace(self, f=float(f))


def window_length(t_c, cfg, t_start=0):
    """⌈window_frac·(t_c − t_start)⌉ steps, at least min_window, at most t_c; t_start = the forecaster's start state
    (release: 0; width 1: the rule point, so that a long stall before the final rise is not in the window)."""
    return int(min(t_c, max(cfg.min_window, math.ceil(cfg.window_frac * (t_c - t_start)))))


# ================================================================================================ (i) extrapolation
def _poly_anchor(tau, w, deg, k):
    """Least-squares polynomial of degree deg in τ/scale over (τ, w), evaluated at k (> 0) minus its value at 0."""
    sc = max(1.0, float(abs(tau).max()))
    c = np.polyfit(tau / sc, w, deg)
    return np.polyval(c, k / sc) - np.polyval(c, 0.0), float(np.polyval(np.polyder(c), 0.0) / sc)


def extrapolate_log(y_vis, n_win, n_ahead, family):
    """ŷ_k, k = 1 … n_ahead, from the visible log-scale y (last n_win points; the last visible point is k = 0) and the
    fitted growth rate dy/dt at the cutoff.  Anchored at y_vis[-1]."""
    if family not in FAMILIES:
        raise ValueError(family)
    w = np.asarray(y_vis[-n_win:], float)
    tau = np.arange(-len(w) + 1, 1, dtype=float)
    k = np.arange(1, n_ahead + 1, dtype=float)
    if family in ("lin", "quad"):
        d, rate = _poly_anchor(tau, w, 1 if family == "lin" else 2, k)
        return w[-1] + d, rate
    g, yy = np.diff(w), w[:-1]
    beta, alpha = np.polyfit(yy - w[-1], g, 1)              # g = α + β(y − y_last)
    out = np.empty(n_ahead)
    y = w[-1]
    for i in range(n_ahead):
        y = y + alpha + beta * (y - w[-1])
        out[i] = y
    return out, float(alpha)


def extrapolate_share(u_vis, y_vis, y_hat, n_win, family):
    """Share û (W2-A) for the extrapolated steps: lin/quad, the same polynomial family in t; rate, u linear in y."""
    w = np.asarray(u_vis[-n_win:], float)
    k = np.arange(1, len(y_hat) + 1, dtype=float)
    if family in ("lin", "quad"):
        tau = np.arange(-len(w) + 1, 1, dtype=float)
        d, _ = _poly_anchor(tau, w, 1 if family == "lin" else 2, k)
        return w[-1] + d
    yw = np.asarray(y_vis[-n_win:], float)
    gam = np.polyfit(yw - yw[-1], w, 1)[0] if np.ptp(yw) > 0 else 0.0
    return w[-1] + gam * (np.asarray(y_hat) - yw[-1])


def _n_ahead(y_last, rate, s_stop, budget_left):
    """Steps to extrapolate: enough for ŷ to reach log s_stop at the fitted rate (×4, for curvature), within the budget."""
    if rate <= 0:
        return 0
    need = (math.log(s_stop) - y_last) / rate
    return int(max(1, min(budget_left, math.ceil(4 * max(need, 1.0)) + 10)))


def extrapolate_scale(s_vis, cfg, s_ref, budget_left, t_start=0):
    """(ŝ for t_c, t_c + 1, …, status, info).  ŝ stops at the first step ≥ horizon_factor·s_ref (inclusive)."""
    s_vis = np.asarray(s_vis, float)
    t_c = len(s_vis)
    n_w = window_length(t_c, cfg, t_start)
    info = {"t_c": t_c, "n_window": n_w}
    if n_w < 3 or np.any(s_vis[-n_w:] <= 0) or not np.all(np.isfinite(s_vis[-n_w:])):
        return np.array([]), "no forecast: window unusable", info
    y = np.log(s_vis)
    _, rate = extrapolate_log(y, n_w, 1, cfg.family)
    info["growth_rate_at_cutoff"] = rate
    if not np.isfinite(rate) or rate <= 0:
        return np.array([]), "no forecast: extrapolated path not growing at the cutoff", info
    s_stop = cfg.horizon_factor * s_ref
    n = _n_ahead(y[-1], rate, s_stop, budget_left)
    yh, _ = extrapolate_log(y, n_w, n, cfg.family)
    hit = np.nonzero(yh >= math.log(s_stop))[0]
    if len(hit):
        yh = yh[:hit[0] + 1]
    info["n_extrapolated"] = int(len(yh))
    return np.exp(yh), "ok", info


def extrapolate_v(V_vis, cfg, s_ref, budget_left):
    """(V̂ rows for t_c, …, status, info) for width 2: ŝ as extrapolate_scale, share û, signs of v_{t_c−1}."""
    V_vis = np.asarray(V_vis, float)
    s_vis = np.abs(V_vis).sum(axis=1)
    s_hat, st, info = extrapolate_scale(s_vis, cfg, s_ref, budget_left)
    if st != "ok":
        return np.zeros((0, 2)), st, info
    n_w = info["n_window"]
    sg = np.sign(V_vis[-n_w:])
    if np.any(sg == 0) or np.any(sg != sg[-1]):
        return np.zeros((0, 2)), "no forecast: an output weight changes sign in the window", info
    u = np.abs(V_vis[:, 0]) / s_vis
    u_hat = np.clip(extrapolate_share(u, np.log(s_vis), np.log(s_hat), n_w, cfg.family), 0.0, 1.0)
    V_hat = sg[-1][None, :] * s_hat[:, None] * np.column_stack([u_hat, 1 - u_hat])
    return V_hat, "ok", info


# ================================================================================================ (ii) generic 1-D lag core
class _Lazy:
    """A cached sequence f(0), f(1), …, f(n − 1), evaluated on first access (the recursion stops at its hit)."""

    def __init__(self, fn, n):
        self.fn, self.n, self.cache = fn, int(n), {}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError(i)
        if i not in self.cache:
            self.cache[i] = self.fn(i)
        return self.cache[i]


def lag_forecast_1d(s_ext, t0, theta_of, hess_of, contains, gapf, delta0, lr, s_switch,
                    P=None, m0=None, bc1_of=None):
    """R4 recursion (linear_response.simulate, unchanged) along s_ext[t0:], from δ₀ at t0, on a branch given by
    callables (θ*(s), H(s), contains(s)) and a gap gapf(θ) of a PREDICTED state.  P: the frozen preconditioner (None:
    I).  m0, bc1_of(t): Adam's momentum state and bias correction (None: SGD).  Returns a dict: t_hit (absolute), status,
    traj_radius_max, traj_max_abs_delta, t_sw (first t with s_ext ≥ s_switch), s_switch."""
    s_ext = np.asarray(s_ext, float)
    seg = s_ext[t0:]
    inside = np.array([contains(v) for v in seg])
    T_end = len(seg) if inside.all() else int(np.argmin(inside))
    out = {"segment_steps": int(T_end), "s_switch": float(s_switch)}
    t_sw = LR._first_ge(s_ext, s_switch)
    out["t_sw"] = t_sw
    if T_end < 2:
        return {**out, "t_hit": None, "status": "no forecast: rule point outside the branch grid"}
    seg = seg[:T_end]
    th = _Lazy(lambda i: np.asarray(theta_of(seg[i]), float), T_end)      # computed only up to the hit
    Hs = _Lazy(lambda i: np.asarray(hess_of(seg[i]), float), T_end)
    dth = _Lazy(lambda i: th[i + 1] - th[i], T_end - 1)
    PP = np.tile(np.ones(len(np.atleast_1d(delta0))) if P is None else np.asarray(P, float), (T_end, 1))
    bc1 = None if bc1_of is None else np.array([bc1_of(t0 + i) for i in range(T_end)])
    t_hit, path = LR.simulate(seg, lambda t: Hs[t], PP, dth, lambda i, dd: gapf(th[i] + dd), np.asarray(delta0, float),
                              m0=m0, bc1=bc1, lr=lr)
    idx = sorted(set(range(0, len(path), 10)) | {len(path) - 1})
    rad = max(LR.step_map_radius(Hs[i], PP[0], lr, m0 is not None) for i in idx)
    mx = float(np.nanmax(np.abs(np.array(path))))
    out.update(traj_radius_max=rad, traj_max_abs_delta=mx)
    if t_hit is None:
        return {**out, "t_hit": None, "status": "no forecast: no predicted crossing on the extrapolated path"}
    if rad > 1.0 or not np.isfinite(mx) or mx > LR.DIVERGED:
        return {**out, "t_hit": None, "status": "no forecast: unstable / diverged (Track 1 rule)"}
    return {**out, "t_hit": int(t0 + t_hit), "status": "ok"}


def _finish(res, s_ext, t_c, info, guards):
    """Forecast record: crossing, switch, signed lags, the cutoff and the last row each guard read."""
    out = {"t_c": int(t_c), **info, **{k: v for k, v in res.items() if k != "t_hit"}}
    t_hit = res.get("t_hit")
    out["max_index_read"] = {g.name: g.max_index_read for g in guards if isinstance(g, GuardedArray)}
    if t_hit is None:
        out.update(t_fc=None, s_fc=float("nan"), r_fc=float("nan"), lag_steps_fc=None)
        return out
    s_sw = res["s_switch"]
    out.update(t_fc=int(t_hit), s_fc=float(s_ext[t_hit]), r_fc=float(s_ext[t_hit] / s_sw - 1),
               lag_steps_fc=(int(t_hit) - int(res["t_sw"])) if res.get("t_sw") is not None else None,
               hit_in_extrapolated_part=bool(t_hit >= t_c))
    return out


# ================================================================================================ adapters
@dataclass
class W1Inputs:
    """Width 1 (Track A family, sin activation a).  out: signed w₂ path (guarded, cutoff t_c); hid: (w₁, b₁, b₂) path
    (guarded: one row, the rule point); M, Vhat: Adam's first moment and bias-corrected second moment, 4 columns (w₁, b₁,
    w₂, b₂) (guarded, cutoff t_c; None for SGD)."""
    a: float
    x: np.ndarray
    y: np.ndarray
    s_frozen: float
    opt: str
    lr: float
    t_c: int
    budget: int
    out: object
    hid: object
    M: object = None
    Vhat: object = None


RULE_FRAC = 0.5


def causal_rule_step(s_vis, s_frozen, frac=RULE_FRAC):
    """The first step t_R ≥ 1 with s_t ≥ frac·s*_frozen at every t_R ≤ t ≤ len(s_vis) − 1 (reads s_1 … s_{t_c−1});
    None if s_{t_c−1} < frac·s*_frozen."""
    s = np.asarray(s_vis, float)
    if len(s) < 2 or s[-1] < frac * s_frozen:
        return None
    below = np.nonzero(s[1:] < frac * s_frozen)[0]
    return int(below[-1] + 2) if len(below) else 1


def w1_branch(a, x, y, s_frozen, s_R, TH):
    """Track A's branch rule: Newton at s_R from the (canonical) rule-point state, continued on the own sample (grid
    [0.3, 1.7]·s*_frozen, spacing 0.002·s*_frozen); s*_run = its switch nearest s*_frozen.  (None, None) if no minimum."""
    th_b, res, Hn = LR.newton(TH, s_R, a, x, y)
    if res > 1e-9 or np.linalg.eigvalsh(Hn).min() <= 0:
        return None, None
    B = LR.Branch(s_R, th_b, a, x, y, 0.3 * s_frozen, 1.7 * s_frozen, 0.002 * s_frozen)
    return B, B.switch(s_frozen)


def forecast_w1(inp: W1Inputs, cfg: ForecastConfig):
    t_c = int(inp.t_c)
    w2 = _prefix(inp.out, t_c)
    s = np.abs(w2)
    guards = (inp.out, inp.hid, inp.M, inp.Vhat)
    tR = causal_rule_step(s, inp.s_frozen)
    if tR is None:
        return _finish({"status": "no forecast: no rule point before the cutoff", "s_switch": float("nan")}, s, t_c,
                       {}, guards)
    flip = LR.D_FLIP if w2[tR] < 0 else np.ones(3)
    TH = _row(inp.hid, tR) * flip
    info = {"t_rule": tR, "s_rule": float(s[tR])}
    B, s_run = w1_branch(inp.a, inp.x, inp.y, inp.s_frozen, float(s[tR]), TH)
    if B is None:
        return _finish({"status": "no forecast: no minimum at the rule point", "s_switch": float("nan")}, s, t_c, info,
                       guards)
    if s_run is None:
        return _finish({"status": "no forecast: occupied branch has no switch", "s_switch": float("nan")}, s, t_c,
                       info, guards)
    info["s_run"] = float(s_run)
    adam = inp.opt == "adam"
    P = m0 = bc1_of = None
    if adam:
        tP = tR if cfg.adam_P == "rule_point" else t_c - 1
        P = 1.0 / (np.sqrt(_row(inp.Vhat, tP)[[0, 1, 3]]) + LR.EPS)
        m0 = _row(inp.M, tR)[[0, 1, 3]] * flip
        bc1_of = lambda t: 1 - LR.BETA1 ** t                       # noqa: E731  Adam's step count = the step index
        info["t_P"] = int(tP)
    s_hat, st, xinfo = extrapolate_scale(s, cfg, inp.s_frozen, max(0, inp.budget - t_c + 1), t_start=tR)
    info.update(xinfo)
    if st != "ok":
        return _finish({"status": st, "s_switch": s_run}, s, t_c, info, guards)
    s_ext = np.concatenate([s, s_hat])
    r = lag_forecast_1d(s_ext, tR, B.theta, B.hess, B.contains,
                        lambda th: LR.gap_exact(th[0], th[1], inp.a), TH - B.theta(float(s[tR])), inp.lr, s_run,
                        P=P, m0=m0, bc1_of=bc1_of)
    return _finish(r, s_ext, t_c, info, guards)


@dataclass
class GeluInputs:
    """GELU-T.  out: w₂ path (guarded, cutoff t_c); hid: (w₁, b₁, b₂) path (guarded: row 0, the release, only);
    copy_row: the occupied copy's frozen row (s_switch, κ, λ_min); branch: its own-sample branch grid (GBranch from the
    frozen copy point at s₀; no path input); act: the activation (gap); eta."""
    t_c: int
    budget: int
    out: object
    hid: object
    copy_row: dict
    branch: object
    act: object
    eta: float = 0.03


def forecast_gelu(inp: GeluInputs, cfg: ForecastConfig):
    from .act_general import gap_mid
    t_c = int(inp.t_c)
    s = np.abs(_prefix(inp.out, t_c))
    guards = (inp.out, inp.hid)
    z_rel = _row(inp.hid, 0)
    cf, B = inp.copy_row, inp.branch
    s_sw = float(cf["s_switch"])
    s_hat, st, info = extrapolate_scale(s, cfg, s_sw, max(0, inp.budget - t_c + 1))
    if st != "ok":
        return _finish({"status": st, "s_switch": s_sw}, s, t_c, info, guards)
    s_ext = np.concatenate([s, s_hat])
    r = lag_forecast_1d(s_ext, 0, B.theta, B.hess, B.contains, lambda th: gap_mid(th[0], th[1], inp.act),
                        z_rel - B.theta(float(s_ext[0])), inp.eta, s_sw)
    out = _finish(r, s_ext, t_c, info, guards)
    if r.get("t_sw") is not None and r["t_sw"] >= 1:     # secondary: κχ with ṡ on the extrapolated path
        t_sw, w = r["t_sw"], min(100, r["t_sw"])
        sdot = (s_ext[t_sw] - s_ext[t_sw - w]) / w
        out["r_cf_fc"] = float(cf["kappa"] * (sdot / s_sw) / (inp.eta * cf["lam_min_switch"]))
    return out


@dataclass
class W2AInputs:
    """W2-A.  out: v path (guarded, cutoff t_c; 2 columns); hid: z path (guarded: row 0, the release, only); copy_row:
    the occupied copy's frozen row; windings k (from the release classification); x, y: the own sample; arm."""
    arm: str
    t_c: int
    budget: int
    out: object
    hid: object
    copy_row: dict
    windings: tuple
    x: np.ndarray
    y: np.ndarray
    copy: str


def forecast_w2a(inp: W2AInputs, cfg: ForecastConfig):
    from . import width2_asym as W
    t_c = int(inp.t_c)
    V = _prefix(inp.out, t_c)
    guards = (inp.out, inp.hid)
    z_rel = _row(inp.hid, 0)
    cf, k = inp.copy_row, tuple(inp.windings)
    s_ref = float(cf["s_switch"])
    V_hat, st, info = extrapolate_v(V, cfg, s_ref, max(0, inp.budget - t_c + 1))
    s_vis = np.abs(V).sum(axis=1)
    if st != "ok":
        return _finish({"status": st, "s_switch": float("nan")}, s_vis, t_c, info, guards)
    V_ext = np.vstack([V, V_hat])
    s_ext = np.abs(V_ext).sum(axis=1)
    eta = W.ETA[inp.arm]
    br = W.OwnBranch(V_ext, W.shift(np.array(cf["z_s0"]), V_ext[0], k), inp.x, inp.y, inp.copy in W.DUPLICATE_COPIES)
    t_last = len(V_ext) - 1
    if W.OWN_PATH_SWITCH[inp.arm]:
        t_sw, s_sw = W.own_path_switch(br, t_last)
    else:
        t_sw, s_sw = W.first_ge(s_ext, s_ref), s_ref
    res = {"t_sw": t_sw, "s_switch": float(s_sw)}
    if t_sw is None or not np.isfinite(s_sw):
        return _finish({**res, "status": "no forecast: no switch on the extrapolated v path",
                        "branch_lost_at": br.lost_at}, s_ext, t_c, info, guards)
    t_hit, stt, mx, rad = W.r4_recursion(br, z_rel, eta, t_last)
    res.update(traj_radius_max=rad, traj_max_abs_delta=mx, branch_lost_at=br.lost_at, t_hit=t_hit,
               status="ok" if t_hit is not None else f"no forecast: {stt}")
    return _finish(res, s_ext, t_c, info, guards)


ADAPTERS = {"w1": forecast_w1, "gelu": forecast_gelu, "w2a": forecast_w2a}


def run_forecast(kind, inputs, cfg):
    """The forecast, with every guard's last row asserted < t_c (and the hidden row asserted to be the allowed one)."""
    out = ADAPTERS[kind](inputs, cfg)
    for name, mi in out["max_index_read"].items():
        if mi >= inputs.t_c:
            raise CausalityViolation(f"{name}: row {mi} read with cutoff {inputs.t_c}")
    return out
