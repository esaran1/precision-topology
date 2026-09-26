"""Track 1 (final round), POST HOC on existing runs: exact linear response along each run's own trajectory.

LABEL: POST HOC.  Every run here already exists and its observed crossing was known (in aggregate) before this analysis;
the per-run predicted crossings are nevertheless computed and committed (with SHA-256) BEFORE any observed crossing is
read.  Observed crossings are loaded only by `compare`, which first asserts the committed hash.

Why.  The Track 1A lag law r = κ(a)·χ (math note §13) under-predicts nothing but over-predicts the observed lag: the
observed lag is 0.82-0.91 of κχ at every a.  The derivation makes three simplifications: (1) coefficients (H, P, θ*′,
∇G) frozen at the switch, (2) a steady-state (slaved) displacement, (3) no momentum.  Here each is removed.

Model (for each run, along its actual trajectory; canonical orientation w₂ > 0, i.e. (w₁, b₁) → (−w₁, −b₁) and Adam's
first moment likewise when w₂ < 0):
    θ = (w₁, b₁, b₂) (b₂ TRAINED, as in 1A), s_t = |w₂| after step t (the run's actual path), θ*(s) the run's OWN tracked
    conditional stationary branch on its OWN training sample (the objective it trains on), H(s) its Hessian,
    δ_t = θ_t − θ*(s_t).  Adam (torch ordering; step t+1 uses the gradient at (θ_t, s_t)):
        m_{t+1} = β₁ m_t + (1 − β₁) H(s_t) δ_t
        δ_{t+1} = δ_t − η P_{t+1} m_{t+1} / (1 − β₁^{k_{t+1}}) − [θ*(s_{t+1}) − θ*(s_t)]
    with P_{t+1} = 1/(√v_{t+1}/√(1 − β₂^{k_{t+1}}) + ε) the run's ACTUAL preconditioner (v from the replay, not
    linearised) and k the run's actual Adam step count.  No-momentum form: δ_{t+1} = δ_t − ηP_{t+1}H(s_t)δ_t − Δθ*_t.
    SGD (plain torch SGD, lr 0.3, no momentum): P = I, no m state.
    Start: t₀ = the step at which the run last passes c·s* before the switch (s_{t₀−1} < c·s* ≤ s_t for all
    t₀ ≤ t < t_sw), c ∈ {0.7, 0.5}; δ₀ = θ_{t₀} − θ*(s_{t₀}); m₀ = the run's actual Adam first
    moment at t₀ (the gradient EMA; the variant full_m0H uses H(s_{t₀})δ₀ instead).
    Predicted crossing: the first step t ≥ t₀ at which G(θ*(s_t) + δ_t) > 0 (exact extrema on the continuous windows),
    s_pred = s_t there; predicted lag r_pred = s_pred/s* − 1.
    Interventions inside the window (Block 4b arms, applied at t*): the run's actual θ jump (teleport) is added to δ and
    the model's m is replaced by the run's actual first moment after the load (reset: zeros), exactly as in the run.

Variants (each committed before comparison).  Reading of the author's ablations: (a)-(c) change the coefficients of
the full model and keep its momentum state; (d) removes the momentum state.
    full      P_t, H_t, exact θ*(s) path and exact G; momentum (m₀ actual).
    full_m0H  as full with m₀ = H(s_{t₀})δ₀.
    a         (ablation a) all coefficients frozen at the switch: H_sw, P_sw, linear branch θ*(s*) + θ*′(s − s*) and
              linear G (g′(s*)(s − s*) + ∇G·δ, as in 1A); momentum kept.  Should reproduce κχ.
    b         (ablation b) time-varying P only (H, θ*′ frozen, linear branch and G as in a); momentum kept.
    c         (ablation c) time-varying H and θ*′ only (exact θ*(s) path, exact G); P frozen at the switch; momentum kept.
    c_path    (split of c) exact θ*(s) path and exact G, H and P frozen at the switch; momentum kept.
    d         (ablation d) as full without the momentum state.
    a_nomom   as a without momentum: the literal dynamics of the 1A derivation (discrete time).
    exact_grad  POST-COMPARISON DIAGNOSTIC ONLY (never in the committed predictions), not a linear model: as full but with the exact gradient ∇L(θ*(s_t) + δ_t; s_t) in place
              of H(s_t)δ_t.  With the actual s path and the actual P_t this is the run's own hidden-coordinate update, so
              it must reproduce the observed crossing; full differs from it ONLY by linearising the gradient in δ.
    slaved    the closed-form 1A formula with this run's own coefficients at its own switch:
              r = [∇G·(ηP_swH)⁻¹θ*′]·ṡ_sw / (s*·∇G·θ*′), ṡ_sw = (s_{t_sw} − s_{t_sw−100})/100.
    A variant whose δ path leaves sup|δ| ≤ 1 (or is non-finite) is reported as diverged, and a variant whose iterated
    one-step linear map has spectral radius > 1 (checked every 10th step up to the hit; `step_map_radius`) as
    unstable; neither has a prediction (the raw first-gap-positive scale is kept as `*_raw_s_pred`, not used).
    For SGD (plain torch SGD, no momentum): P ≡ I and no momentum in any variant, so b ≡ a, c ≡ d ≡ full ≡ full_m0H,
    a_nomom ≡ a.
    "At the switch": t_sw = first step with s_t ≥ s* (the branch switch, not the run's crossing); P_sw = P_{t_sw}.

Branch and s* (design decisions):
    - Objective: the run's own sample (lag tests and 4b: sample_size._data(6400, seed); SGD: make_data(200, seed);
      Task B: make_data(200, seed)); the loss is the run's mean BCE.
    - The run is replayed to a fixed horizon (s_t ≥ HORIZON·s*_pop(a) or its step budget) WITHOUT evaluating the gap of
      the actual trajectory and without reading any recorded crossing column.
    - Tracked branch: Newton (in θ at fixed s) from the run's state at the first step with s_t ≥ s*_pop(a) (the 1A
      population switch, a fixed a-level number), continued on the grid below; s* = the root of G(θ*(s)) on that branch
      nearest the anchor (bisection).  Refinement: Newton from the state at the first step with s_t ≥ s*; if it lands
      on a different branch (sup distance > 1e-6), the branch is re-identified from there (counted: `branch_refined`).
    - Grid: s from GRID_LO·s*_pop to GRID_HI·s*_pop, spacing GRID_H·s*_pop, Newton continuation with a tangent
      predictor; θ*(s) by cubic Hermite interpolation (exact tangents), H(s) linear.  A fold (Newton failure, H not
      positive definite, or a jump) ends the grid on that side.
    - Different branch at the start point: Newton from the actual state at s_{t₀} converges to a point at sup distance
      > 1e-6 from θ*(s_{t₀}) on the tracked branch (or does not converge).  Counted per arm; none excluded: the model is
      run with δ₀ = θ_{t₀} − θ*(s_{t₀}) on the tracked branch.

    python -m src.linear_response time            # timing sample (no predictions written)
    python -m src.linear_response predict         # resumable; results/linear_response/predictions_parts.jsonl
    python -m src.linear_response finalize        # predictions.csv + predictions.sha256 (commit before compare)
    python -m src.linear_response compare         # asserts the committed hash, then reads observed crossings
    python -m src.linear_response diagnose [n]    # POST-COMPARISON diagnostic: exact-gradient reconstruction
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "linear_response"
PARTS = OUT / "predictions_parts.jsonl"
PRED = OUT / "predictions.csv"
SHA = OUT / "predictions.sha256"

LR_ADAM, LR_SGD, BETA1, BETA2, EPS = 1e-2, 0.3, 0.9, 0.999, 1e-8
HORIZON = 1.6            # replay until s_t >= HORIZON * s*_pop(a) (or the run's budget)
GRID_LO, GRID_HI, GRID_H = 0.3, 1.7, 0.002
STARTS = (0.7, 0.5)
BRANCH_TOL = 1e-6
S_POP = {1.30: 4.958011429378877, 1.45: 2.8962428593646985, 1.50: 2.5270, 1.60: 2.0022}   # overwritten from kappa.csv
TWO_PI = 2 * math.pi
D_FLIP = np.array([-1.0, -1.0, 1.0])


def _s_pop():
    k = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    return {round(float(a), 2): float(s) for a, s in zip(k.a, k.s_star)}


# ------------------------------------------------------------------------------------------ loss, gradient, Hessian
def derivs(th, s, a, x, y, need_ds=False):
    """Mean BCE of z = s·f_a(w₁x + b₁) + b₂ in θ = (w₁, b₁, b₂): loss, gradient, Hessian (and ∂ₛ∇L)."""
    w1, b1, b2 = th
    t = w1 * x + b1
    st, ct = np.sin(t), np.cos(t)
    f = t + a * st; f1 = 1 + a * ct; f2 = -a * st
    z = s * f + b2
    p = 0.5 * (1 + np.tanh(0.5 * z))
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    r = p - y; q = p * (1 - p)
    dz = np.stack([s * f1 * x, s * f1, np.ones_like(x)])            # 3 × n
    n = len(x)
    g = dz @ r / n
    H = (dz * q) @ dz.T / n
    rs = r * s * f2
    H[0, 0] += np.mean(rs * x * x); H[0, 1] += np.mean(rs * x); H[1, 0] = H[0, 1]; H[1, 1] += np.mean(rs)
    if not need_ds:
        return loss, g, H
    dsg = dz @ (q * f) / n + np.array([np.mean(r * f1 * x), np.mean(r * f1), 0.0])
    return loss, g, H, dsg


def newton(th0, s, a, x, y, iters=60, tol=1e-12):
    th = np.asarray(th0, float).copy()
    L0, g, H = derivs(th, s, a, x, y)
    mu = 0.0
    for _ in range(iters):
        if np.abs(g).max() < tol:
            break
        ok = False
        for _ in range(30):
            try:
                d = np.linalg.solve(H + mu * np.eye(3), -g)
            except np.linalg.LinAlgError:
                mu = max(mu * 10, 1e-8); continue
            L1, g1, H1 = derivs(th + d, s, a, x, y)
            if L1 <= L0 + 1e-15 or np.abs(g1).max() < np.abs(g).max() * 0.5:
                th, L0, g, H, ok = th + d, L1, g1, H1, True
                mu = mu / 10 if mu > 1e-12 else 0.0
                break
            mu = max(mu * 10, 1e-8)
        if not ok:
            break
    return th, float(np.abs(g).max()), H


# ------------------------------------------------------------------------------------------ exact gap (continuous windows)
def _fmax(lo, hi, a, tc):
    """max of f_a(t) = t + a sin t on [lo, hi]; local maxima at tc + 2πn (cos tc = −1/a, sin tc > 0)."""
    best = max(lo + a * math.sin(lo), hi + a * math.sin(hi))
    n0 = math.ceil((lo - tc) / TWO_PI)
    t = tc + TWO_PI * n0
    while t <= hi:
        best = max(best, t + a * math.sin(t)); t += TWO_PI
    return best


def _fmin(lo, hi, a, tc):
    best = min(lo + a * math.sin(lo), hi + a * math.sin(hi))
    n0 = math.ceil((lo + tc) / TWO_PI)
    t = -tc + TWO_PI * n0
    while t <= hi:
        best = min(best, t + a * math.sin(t)); t += TWO_PI
    return best


def gap_exact(w1, b1, a):
    """G₊ = min_O f_a − max_I f_a with I = w₁[−0.8, 0.8] + b₁, O = w₁(±[1.2, 2.0]) + b₁ (orientation w₂ > 0)."""
    if not (math.isfinite(w1) and math.isfinite(b1)) or abs(w1) > 1e4 or abs(b1) > 1e4:
        return float("nan")                                   # diverged state: no crossing is declared
    tc = math.acos(-1.0 / a) if a > 1 else math.pi
    w = abs(w1)
    mI = _fmax(b1 - 0.8 * w, b1 + 0.8 * w, a, tc)
    mO = min(_fmin(b1 + 1.2 * w, b1 + 2.0 * w, a, tc), _fmin(b1 - 2.0 * w, b1 - 1.2 * w, a, tc))
    return mO - mI


def grad_gap(w1, b1, a, h=1e-6):
    return np.array([(gap_exact(w1 + h, b1, a) - gap_exact(w1 - h, b1, a)) / (2 * h),
                     (gap_exact(w1, b1 + h, a) - gap_exact(w1, b1 - h, a)) / (2 * h), 0.0])


# ------------------------------------------------------------------------------------------ branch by continuation on a grid
class Branch:
    """θ*(s) on a grid by Newton continuation from an anchor (s_a, θ_a); Hermite θ*, linear H."""

    def __init__(self, s_anchor, th_anchor, a, x, y, s_lo, s_hi, h):
        self.a, self.x, self.y = a, x, y
        th, res, H = newton(th_anchor, s_anchor, a, x, y)
        self.anchor_ok = res < 1e-9 and np.linalg.eigvalsh(H).min() > 0
        pts = {s_anchor: self._pt(th, s_anchor)}
        for direction in (+1, -1):
            s, th_c = s_anchor, th.copy()
            nxt = (math.floor(s_anchor / h) + (1 if direction > 0 else 0)) * h
            while (direction > 0 and nxt <= s_hi) or (direction < 0 and nxt >= s_lo):
                pt = pts[s] if s in pts else self._pt(th_c, s)
                pred = th_c + pt[2] * (nxt - s)
                th_n, res, Hn = newton(pred, nxt, a, x, y, iters=30)
                jump = np.abs(th_n - pred).max()
                if res > 1e-9 or np.linalg.eigvalsh(Hn).min() <= 0 or jump > 1e-2 + 0.5 * np.abs(pred - th_c).max():
                    break
                pts[nxt] = self._pt(th_n, nxt)
                s, th_c = nxt, th_n
                nxt = nxt + direction * h
        self.s = np.array(sorted(pts))
        self.th = np.array([pts[k][0] for k in self.s])
        self.H = np.array([pts[k][1] for k in self.s])
        self.tan = np.array([pts[k][2] for k in self.s])
        self.lo, self.hi = float(self.s[0]), float(self.s[-1])

    def _pt(self, th, s):
        _, g, H, dsg = derivs(th, s, self.a, self.x, self.y, need_ds=True)
        return th.copy(), H, -np.linalg.solve(H, dsg)

    def contains(self, s):
        return self.lo <= s <= self.hi

    def _idx(self, s):
        i = int(np.searchsorted(self.s, s) - 1)
        return min(max(i, 0), len(self.s) - 2)

    def theta(self, s):
        i = self._idx(s); s0, s1 = self.s[i], self.s[i + 1]; hh = s1 - s0; u = (s - s0) / hh
        h00 = 2 * u**3 - 3 * u**2 + 1; h10 = u**3 - 2 * u**2 + u; h01 = -2 * u**3 + 3 * u**2; h11 = u**3 - u**2
        return h00 * self.th[i] + h10 * hh * self.tan[i] + h01 * self.th[i + 1] + h11 * hh * self.tan[i + 1]

    def hess(self, s):
        i = self._idx(s); u = (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        return (1 - u) * self.H[i] + u * self.H[i + 1]

    def exact(self, s):
        """Newton-refined θ*(s), H and tangent (for the switch and the interpolation check)."""
        th, res, _ = newton(self.theta(s), s, self.a, self.x, self.y)
        _, _, H, dsg = derivs(th, s, self.a, self.x, self.y, need_ds=True)
        return th, H, -np.linalg.solve(H, dsg), res

    def gap_at(self, s):
        th = self.theta(s)
        return gap_exact(th[0], th[1], self.a)

    def switch(self, s_ref):
        """Root of g(s) = G(θ*(s)) (− → +) nearest s_ref, by bisection on the Newton-refined branch."""
        gs = np.array([gap_exact(t[0], t[1], self.a) for t in self.th])
        roots = [i for i in range(len(gs) - 1) if gs[i] <= 0 < gs[i + 1]]
        if not roots:
            return None
        i = min(roots, key=lambda j: abs(self.s[j] - s_ref))
        lo, hi = float(self.s[i]), float(self.s[i + 1])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            th, *_ = self.exact(mid)
            if gap_exact(th[0], th[1], self.a) > 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-12:
                break
        return 0.5 * (lo + hi)

    def on_branch(self, s, th, tol=BRANCH_TOL):
        if not self.contains(s):
            return False
        ref, *_ = self.exact(s)
        return float(np.abs(ref - th).max()) <= tol


# ------------------------------------------------------------------------------------------ the recursion
def step_map_radius(H, P, lr, momentum, beta1=BETA1):
    """Spectral radius of the one-step linear map (bias correction 1): no momentum δ' = (I − ηPH)δ; momentum
    state (δ, m): m' = β₁m + (1 − β₁)Hδ, δ' = δ − ηPm'.  > 1: the linearised dynamics is unstable."""
    H = np.asarray(H, float); P = np.asarray(P, float)
    if not momentum:
        return float(np.abs(np.linalg.eigvals(np.eye(3) - lr * P[:, None] * H)).max())
    A = np.zeros((6, 6))
    A[:3, :3] = np.eye(3) - lr * (1 - beta1) * P[:, None] * H
    A[:3, 3:] = -lr * beta1 * np.diag(P)
    A[3:, :3] = (1 - beta1) * H
    A[3:, 3:] = beta1 * np.eye(3)
    return float(np.abs(np.linalg.eigvals(A)).max())


def grad_only(th, s, a, x, y):
    w1, b1, b2 = th
    t = w1 * x + b1
    f1 = 1 + a * np.cos(t)
    z = s * (t + a * np.sin(t)) + b2
    r = 0.5 * (1 + np.tanh(0.5 * z)) - y
    n = len(x)
    return np.array([s * np.dot(r * f1, x) / n, s * np.sum(r * f1) / n, np.sum(r) / n])


def simulate(s, H_of, P, dth, gapf, delta0, m0=None, bc1=None, beta1=BETA1, lr=LR_ADAM, events=None, grad=None):
    """Linear response along a given path, from index 0.

    s[t]      scale after step t (t = 0 … T);  H_of(t): Hessian used for the gradient of step t+1 (at s[t]);
    P[t]      preconditioner of step t (t ≥ 1; P[0] unused);  dth[t] = θ*(s[t+1]) − θ*(s[t]) (or its linear form);
    gapf(t, δ) → gap at step t;  m0 None: no momentum state (m̂ = Hδ exactly);  bc1[t] = 1 − β₁^{k_t}.
    events: {t: (jump, m_new)} applied to the state after step t (δ += jump; m = m_new if not None).
    Returns (index of the first t with gap > 0 or None, δ path)."""
    T = len(s) - 1
    d = np.asarray(delta0, float).copy()
    m = None if m0 is None else np.asarray(m0, float).copy()
    path = [d.copy()]
    if gapf(0, d) > 0:
        return 0, path
    for t in range(T):
        g = H_of(t) @ d if grad is None else grad(t, d)
        if m is None:
            step = g
        else:
            m = beta1 * m + (1 - beta1) * g
            step = m / bc1[t + 1]
        d = d - lr * P[t + 1] * step - dth[t]
        if events and (t + 1) in events:
            jump, m_new = events[t + 1]
            d = d + jump
            if m is not None and m_new is not None:
                m = np.asarray(m_new, float).copy()
        path.append(d.copy())
        if not np.all(np.isfinite(d)) or np.abs(d).max() > 1e4:
            return None, path                                 # diverged (reported by the caller as no crossing)
        if gapf(t + 1, d) > 0:
            return t + 1, path
    return None, path


# ------------------------------------------------------------------------------------------ replays (recording, no gap)
def _record_state(th, opt, adam):
    q = th.detach().numpy()
    rec = {"w": q.copy()}
    if adam:
        st = opt.state[th]
        k = int(st["step"])
        v = st["exp_avg_sq"].numpy()
        rec.update(m=st["exp_avg"].numpy().copy(), k=k,
                   P=1.0 / (np.sqrt(v) / math.sqrt(1 - BETA2 ** k) + EPS) if k > 0 else np.full(4, np.nan))
    return rec


class Path_:
    """A recorded trajectory in canonical coordinates: s[t], θ[t] (3), m[t] (3), P[t] (3), bc1[t], events."""

    def __init__(self, adam):
        self.adam = adam
        self.w, self.m, self.P, self.k = [], [], [], []
        self.events = {}

    def add(self, rec):
        self.w.append(rec["w"])
        if self.adam:
            self.m.append(rec["m"]); self.P.append(rec["P"]); self.k.append(rec["k"])

    def finish(self):
        W = np.array(self.w)
        self.sign = np.sign(W[:, 2])
        self.s = np.abs(W[:, 2])
        flip = np.where(self.sign[:, None] < 0, D_FLIP[None, :], 1.0)
        self.th = W[:, [0, 1, 3]] * flip
        if self.adam:
            self.mm = np.array(self.m)[:, [0, 1, 3]] * flip
            self.PP = np.array(self.P)[:, [0, 1, 3]]
            self.bc1 = 1 - BETA1 ** np.array(self.k, float)
        else:
            self.mm = None; self.PP = np.ones((len(W), 3)); self.bc1 = None
        del self.w, self.m, self.P
        return self


def _horizon_loop(th, opt, step_fn, pth, adam, s_stop, t_from, t_budget):
    t = t_from
    while t < t_budget:
        step_fn()
        t += 1
        pth.add(_record_state(th, opt, adam))
        if abs(float(th.detach()[2])) >= s_stop:
            break
    return t


def replay_path(kind, a, seed, arm, s_stop, cache=None):
    """Replay a run from its initialisation to the horizon (s ≥ s_stop or budget), recording every step.  No gap of the
    actual trajectory is evaluated.  kind ∈ {lag1, lag2-prim, lag2-tstar, 4b, SGD, TaskB}."""
    import copy
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data
    torch.set_num_threads(1)
    f = activation("sin_family", a)
    if kind in ("SGD", "TaskB"):
        x, y = make_data(200, seed); x, y = x.double(), y.double()
        torch.manual_seed(seed)
        th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
        adam = kind == "TaskB"
        opt = torch.optim.Adam([th], lr=LR_ADAM) if adam else torch.optim.SGD([th], lr=LR_SGD)
        pth = Path_(adam); pth.add({"w": th.detach().numpy().copy(), "m": np.zeros(4), "k": 0, "P": np.full(4, np.nan)})

        def step_fn():
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
            opt.step()
        _horizon_loop(th, opt, step_fn, pth, adam, s_stop, 0, 32_000)
        return pth.finish(), x.numpy().astype(float), y.numpy().astype(float)

    from . import lag_test2 as lt
    from .lag_test import scale_w2_step
    fac = 1.0 if kind == "4b" else float(arm)
    f_, G_, xt, yt, th, opt = lt._setup(a, seed)
    xn, yn = xt.numpy().astype(float), yt.numpy().astype(float)
    pth = Path_(True)
    pth.add({"w": th.detach().numpy().copy(), "m": np.zeros(4), "k": 0, "P": np.full(4, np.nan)})

    def mk_step(theta, op, factor):
        def step_fn():
            op.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(logits(theta, xt, f), yt).backward()
            w2b = float(theta.detach()[2]); op.step(); scale_w2_step(theta, w2b, factor)
        return step_fn

    if kind == "lag1":
        budget = int(round(32_000 / fac))
        _horizon_loop(th, opt, mk_step(th, opt, fac), pth, True, s_stop, 0, budget)
        return pth.finish(), xn, yn

    rule = "tstar" if kind == "lag2-tstar" else "primary"
    ck = torch.load(lt.STATES / f"{rule}_a{a:.2f}_s{seed}.pt", weights_only=False)
    t_star = int(ck["t_star"])
    # pre-t* segment: the φ = 1 trajectory from initialisation (as lag_test2.checkpoint_one pass 2)
    t = 0
    while t < t_star:
        mk_step(th, opt, 1.0)(); t += 1
        pth.add(_record_state(th, opt, True))
    st = opt.state[th]
    same = (torch.equal(th.detach(), ck["theta"]) and torch.equal(st["exp_avg"], ck["opt"]["state"][0]["exp_avg"])
            and torch.equal(st["exp_avg_sq"], ck["opt"]["state"][0]["exp_avg_sq"]))
    if not same:
        raise RuntimeError(f"pre-t* replay does not reproduce the checkpoint ({kind}, {a}, {seed})")
    if kind in ("lag2-prim", "lag2-tstar"):
        theta = ck["theta"].clone().requires_grad_(True)
        op = torch.optim.Adam([theta], lr=1e-2); op.load_state_dict(ck["opt"])
        total = t_star + math.ceil((lt.BASE_BUDGET - t_star) / fac)
        _horizon_loop(theta, op, mk_step(theta, op, fac), pth, True, s_stop, t_star, total)
        return pth.finish(), xn, yn
    # Block 4b arm (replay_4b's setup, AS RUN)
    from . import residual_mechanism as rm
    theta0 = tuple(float(v) for v in ck["theta"])
    if arm.startswith("teleport"):
        thn, _ = rm.apply_teleport(theta0, rm.teleport_target(a, seed, theta0, xn, yn), a, xn, yn)
    else:
        thn = theta0
    opt_state = rm.reset_moments(ck["opt"]) if arm.endswith("reset") else ck["opt"]
    if arm == "teleport_reset":
        rm.continue_from(a, seed, theta0, opt_state, t_star)          # AS RUN (see timescale_consistency.replay_4b)
    if arm == "teleport":
        rm.continue_from(a, seed, theta0, ck["opt"], t_star)          # AS RUN; its return value is not read
    theta = torch.tensor(thn, dtype=torch.float64).requires_grad_(True)
    op = torch.optim.Adam([theta], lr=1e-2); op.load_state_dict(opt_state)
    # the intervention: replace the last recorded state (step t*) by the post-intervention state; keep the jump
    before = pth.w[-1].copy(); m_before = pth.m[-1].copy()
    rec = _record_state(theta, op, True) if int(op.state[theta]["step"]) > 0 else \
        {"w": theta.detach().numpy().copy(), "m": np.zeros(4), "k": 0, "P": np.full(4, np.nan)}
    # θ and the first moment at t* are the post-intervention ones (the state the continuation starts from); the step
    # count and P recorded at t* stay those of the step INTO t* (taken before the intervention)
    pth.w[-1] = rec["w"]; pth.m[-1] = rec["m"]
    pth.intervention = {"t": t_star, "theta_before": before, "m_before": m_before}
    _horizon_loop(theta, op, mk_step(theta, op, 1.0), pth, True, s_stop, t_star, lt.BASE_BUDGET)
    return pth.finish(), xn, yn


# ------------------------------------------------------------------------------------------ per-run predictions
VARIANTS = ("full", "full_m0H", "d", "c", "c_path", "b", "a", "a_nomom")
# The exact-gradient reconstruction reproduces the run's own hidden-coordinate trajectory, i.e. it would evaluate the
# gap of the actual trajectory; it is therefore NOT computed before the commit, only afterwards as a labelled
# post-comparison diagnostic (`diagnose`).
DIAG_VARIANTS = ("full", "exact_grad")
DIVERGED = 1.0           # a linear-response path with sup|δ| > 1 is reported as diverged (no prediction)


def _first_ge(s, v, start=0):
    idx = np.nonzero(s[start:] >= v)[0]
    return None if len(idx) == 0 else int(idx[0] + start)


def predict_run(pth, x, y, a, s_pop, branches, variants=None):
    """All variants, both start points, for one recorded path.  `branches`: list of Branch objects already computed on
    this sample (reused when the anchor lands on one)."""
    out = {"n_steps": len(pth.s) - 1, "s_max": float(pth.s.max())}
    lo, hi, h = GRID_LO * s_pop, GRID_HI * s_pop, GRID_H * s_pop

    def get_branch(t_anchor):
        s_a, th_a = float(pth.s[t_anchor]), pth.th[t_anchor]
        th_b, res, H = newton(th_a, s_a, a, x, y)
        if res > 1e-9 or np.linalg.eigvalsh(H).min() <= 0:
            return None, th_b
        for B in branches:
            if B.contains(s_a) and float(np.abs(B.theta(s_a) - th_b).max()) <= BRANCH_TOL:
                return B, th_b
        B = Branch(s_a, th_b, a, x, y, lo, hi, h)
        branches.append(B)
        return B, th_b

    t_ref = _first_ge(pth.s, s_pop)
    if t_ref is None:
        return {**out, "status": "horizon: s never reaches s*_pop"}
    B, _ = get_branch(t_ref)
    if B is None:
        return {**out, "status": "anchor: Newton did not converge to a minimum"}
    s_star = B.switch(s_pop)
    refined = False
    if s_star is not None:
        t_sw = _first_ge(pth.s, s_star)
        if t_sw is not None:
            B2, thb2 = get_branch(t_sw)
            if B2 is not None and B2 is not B:
                refined = True
                s2 = B2.switch(float(pth.s[t_sw]))
                if s2 is not None:
                    B, s_star = B2, s2
    out["branch_refined"] = refined
    if s_star is None:
        return {**out, "status": "no switch on the tracked branch"}
    t_sw = _first_ge(pth.s, s_star)
    if t_sw is None:
        return {**out, "s_star": s_star, "status": "horizon: s never reaches s*"}
    th_sw, H_sw, tan_sw, res_sw = B.exact(s_star)
    dG = grad_gap(th_sw[0], th_sw[1], a)
    P_sw = pth.PP[t_sw]
    gprime = float(dG @ tan_sw)
    win = min(100, t_sw)
    sdot = (pth.s[t_sw] - pth.s[t_sw - win]) / win
    lr = LR_ADAM if pth.adam else LR_SGD
    r_slaved = float(dG @ np.linalg.solve(lr * P_sw[:, None] * H_sw, tan_sw)) * sdot / (s_star * gprime)
    Ph = np.sqrt(P_sw)
    lam = float(np.linalg.eigvalsh((Ph[:, None] * H_sw) * Ph[None, :]).min())
    chi_sw = (sdot / s_star) / (lr * lam)
    kappa_own = lam * float(dG @ np.linalg.solve(P_sw[:, None] * H_sw, tan_sw)) / gprime
    lam_max = float(np.linalg.eigvalsh((Ph[:, None] * H_sw) * Ph[None, :]).max())
    out["eta_lam_max_sw"] = lr * lam_max
    out["rho_sw_nomom"] = step_map_radius(H_sw, P_sw, lr, False)
    out["rho_sw_mom"] = step_map_radius(H_sw, P_sw, lr, pth.adam)
    out["eta_lam_min_sw"] = lr * lam
    out.update(status="ok", s_star=s_star, t_sw_rel=None, winding_b1_sw=float(th_sw[1]), w1_sw=float(th_sw[0]),
               kappa_own=kappa_own, chi_sw=chi_sw, r_slaved=r_slaved, sdot_sw=float(sdot), lam_sw=lam,
               gprime=gprime, branch_lo=B.lo, branch_hi=B.hi, sign_changes=int(np.sum(np.diff(pth.sign) != 0)))
    # interpolation error check at the switch
    out["interp_err_theta_sw"] = float(np.abs(B.theta(s_star) - th_sw).max())
    out["interp_err_H_sw"] = float(np.abs(B.hess(s_star) - H_sw).max())
    iv = getattr(pth, "intervention", None)
    for c in STARTS:
        tag = f"{int(round(c * 10)):d}"
        below = np.nonzero(pth.s[:t_sw] < c * s_star)[0]
        t0 = int(below[-1] + 1) if len(below) else 0          # where the run last passes c·s* on its way to the switch
        if not B.contains(float(pth.s[t0])):
            out[f"start{tag}_status"] = "start outside the tracked branch's range (fold)"
            out[f"start{tag}_different_branch"] = True
            continue
        out[f"start{tag}_status"] = "ok"
        out[f"start{tag}_sign_changes"] = int(np.sum(np.diff(pth.sign[t0:]) != 0))
        # the model runs only while s stays inside the grid of the tracked branch
        s_seg = pth.s[t0:]
        inside = (s_seg >= B.lo) & (s_seg <= B.hi)
        T_end = len(s_seg) if inside.all() else int(np.argmin(inside))
        s_seg = s_seg[:T_end]
        th_seg = [B.theta(v) for v in s_seg]
        th0_act = pth.th[t0]
        d0 = th0_act - th_seg[0]
        # start-branch check
        thb0, res0, H0n = newton(th0_act, float(s_seg[0]), a, x, y)
        conv0 = res0 < 1e-9 and np.linalg.eigvalsh(H0n).min() > 0
        dist0 = float(np.abs(thb0 - th_seg[0]).max())
        out[f"start{tag}_step_offset"] = int(t0 - t_sw)
        out[f"start{tag}_s0_over_sstar"] = float(s_seg[0] / s_star)
        out[f"start{tag}_start_branch_dist"] = dist0
        out[f"start{tag}_start_newton_converged"] = bool(conv0)
        out[f"start{tag}_different_branch"] = bool((not conv0) or dist0 > BRANCH_TOL)
        out[f"start{tag}_start_branch_w1_sign_differs"] = bool(np.sign(thb0[0]) != np.sign(th_seg[0][0]))
        out[f"start{tag}_start_branch_b1_shift_2pi"] = int(round((thb0[1] - th_seg[0][1]) / TWO_PI))
        out[f"start{tag}_delta0_norm"] = float(np.abs(d0).max())
        Hs = [B.hess(v) for v in s_seg]
        P = pth.PP[t0:t0 + T_end]
        bc1 = None if pth.bc1 is None else pth.bc1[t0:t0 + T_end]
        dth_exact = [th_seg[i + 1] - th_seg[i] for i in range(T_end - 1)]
        dth_lin = [tan_sw * (s_seg[i + 1] - s_seg[i]) for i in range(T_end - 1)]
        th_lin = lambda i: th_sw + tan_sw * (s_seg[i] - s_star)
        gap_exact_f = lambda i, d: gap_exact(th_seg[i][0] + d[0], th_seg[i][1] + d[1], a)
        gap_lin_f = lambda i, d: gprime * (s_seg[i] - s_star) + float(dG @ d)
        Pfix = np.tile(P_sw, (T_end, 1))
        events = None
        if iv is not None and t0 < iv["t"] < t0 + T_end:
            i_ev = iv["t"] - t0
            jump_raw = pth.th[iv["t"]] - _canon(iv["theta_before"], pth.sign[iv["t"]])
            m_new = pth.mm[iv["t"]] if pth.adam else None
            events = {i_ev: (jump_raw, m_new)}
            out[f"start{tag}_intervention_in_window"] = True
        for v in (variants or VARIANTS):
            frozenH = v in ("a", "a_nomom", "b", "c_path")
            exact_path = v in ("full", "full_m0H", "d", "c", "c_path", "exact_grad")
            varyP = v in ("full", "full_m0H", "d", "b", "exact_grad")
            mom = pth.adam and v not in ("d", "a_nomom")          # SGD: plain SGD, no momentum in any variant
            H_of = (lambda t: H_sw) if frozenH else (lambda t, Hs=Hs: Hs[t])
            dth = dth_exact if exact_path else dth_lin
            gf = gap_exact_f if exact_path else gap_lin_f
            PP = P if varyP else Pfix
            # the linear-branch variants start from the displacement off the linear branch
            dd0 = d0 if exact_path else th0_act - th_lin(0)
            m0 = None
            if mom:
                m0 = pth.mm[t0] if v != "full_m0H" else (H_sw if frozenH else Hs[0]) @ dd0
            gfun = (lambda t, d: grad_only(th_seg[t] + d, float(s_seg[t]), a, x, y)) if v == "exact_grad" else None
            t_hit, path = simulate(s_seg, H_of, PP, dth, gf, dd0, m0=m0, bc1=bc1, lr=lr, events=events, grad=gfun)
            key = f"{v}_{tag}"
            mx = float(np.nanmax(np.abs(np.array(path)))) if len(path) else float("nan")
            n_sim = len(path)
            # stability of the linear map actually iterated (every 10th step up to the hit, and the last one)
            idx = sorted(set(range(0, n_sim, 10)) | {n_sim - 1})
            rho = max(step_map_radius(H_of(i), PP[min(i + 1, T_end - 1)], lr, mom) for i in idx)
            out[f"{key}_rho_max"] = rho
            out[f"{key}_unstable"] = bool(rho > 1.0)
            diverged = (not np.isfinite(mx)) or mx > DIVERGED
            out[f"{key}_max_abs_delta"] = mx
            out[f"{key}_diverged"] = bool(diverged)
            out[f"{key}_raw_s_pred"] = float("nan") if t_hit is None else float(s_seg[t_hit])
            if t_hit is None or diverged or rho > 1.0:
                out[f"{key}_s_pred"] = float("nan"); out[f"{key}_step_rel"] = float("nan")
            else:
                out[f"{key}_s_pred"] = float(s_seg[t_hit]); out[f"{key}_step_rel"] = int(t_hit)
            out[f"{key}_r_pred"] = out[f"{key}_s_pred"] / s_star - 1
        out[f"start{tag}_t0"] = int(t0)
        out[f"start{tag}_T"] = int(T_end)
    return out


def _canon(w4, sign):
    q = np.asarray(w4, float)[[0, 1, 3]]
    return q * D_FLIP if sign < 0 else q


# ------------------------------------------------------------------------------------------ run list and driver
def run_list():
    """The 1,750 runs of the 1A predictions (set, a, seed, arm) — only identifiers and the committed κχ are read."""
    p = pd.read_csv(RESULTS / "lag_law" / "predictions.csv")
    p["a"] = p.a.round(2); p["arm"] = p.arm.astype(str)
    return p[["set", "a", "seed", "arm", "chi", "winding_k", "mirror", "kappa_k", "pred_r"]].rename(
        columns={"chi": "chi_1A", "kappa_k": "kappa_1A", "pred_r": "pred_r_1A", "winding_k": "winding_1A",
                 "mirror": "mirror_1A"})


def _groups(runs):
    """Group by sample (set family, a, seed) so a sample's branches are computed once."""
    fam = runs["set"].map(lambda s: "n6400" if s in ("lag1", "lag2-prim", "lag2-tstar", "4b") else s)
    runs = runs.assign(fam=fam)
    return [(k, g) for k, g in runs.groupby(["fam", "a", "seed"], sort=True)]


def predict(limit_groups=None, subsample=None):
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    SP = _s_pop()
    runs = run_list()
    if subsample is not None:
        runs = subsample_runs(runs, subsample)
    done = set()
    if PARTS.exists():
        d = _read_parts()
        done = {(r.set, round(r.a, 2), int(r.seed), str(r.arm)) for r in d.itertuples()}
    groups = _groups(runs)
    if limit_groups:
        groups = groups[:limit_groups]
    for (fam, a, seed), g in groups:
        branches = []
        cache = {}
        for r in g.itertuples():
            key = (r.set, round(r.a, 2), int(r.seed), str(r.arm))
            if key in done:
                continue
            t0 = time.time()
            s_stop = HORIZON * SP[a]
            ck = _path_key(r.set, r.arm)
            t1 = t0
            try:
                if ck in cache:
                    pth, x, y = cache[ck]
                else:
                    pth, x, y = replay_path(r.set, a, int(r.seed), r.arm, s_stop)
                    cache[ck] = (pth, x, y)
                t1 = time.time()
                res = predict_run(pth, x, y, a, SP[a], branches)
            except Exception as e:                                    # recorded, never silently dropped
                res = {"status": f"error: {type(e).__name__}: {e}"}
            row = {"set": r.set, "a": a, "seed": int(r.seed), "arm": str(r.arm), "path_key": ck,
                   "t_replay": t1 - t0, "t_model": time.time() - t1, **res}
            with open(PARTS, "a") as fh:
                fh.write(json.dumps(_jsonable(row)) + "\n")
        print(json.dumps({"group": [fam, a, int(seed)], "n": len(g), "t": time.strftime("%H:%M:%S")}), flush=True)


def _jsonable(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.bool_,)):
            v = bool(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        out[k] = v
    return out


def _read_parts():
    rows = [json.loads(l) for l in PARTS.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def _path_key(st, arm):
    """Runs that are bitwise the same trajectory (φ = 1, no intervention, same sample and init) share one replay."""
    if st in ("lag1", "lag2-prim", "lag2-tstar") and float(arm) == 1.0:
        return "phi1"
    if st == "4b" and arm == "control":
        return "phi1"
    return f"{st}:{arm}"


def subsample_runs(runs, n):
    """Deterministic subsample fixed BEFORE any prediction: the first n seeds of every arm by seed order."""
    return runs.sort_values(["set", "a", "arm", "seed"]).groupby(["set", "a", "arm"], sort=False).head(n)


def finalize():
    d = _read_parts()
    forbidden = {"crossing_step", "cross_step", "s_cross", "residual", "obs_r", "w2_cross", "step", "w2_abs"}
    assert not (forbidden & set(d.columns)), forbidden & set(d.columns)
    d = d.sort_values(["set", "a", "arm", "seed"]).reset_index(drop=True)
    d.to_csv(PRED, index=False)
    h = hashlib.sha256(PRED.read_bytes()).hexdigest()
    SHA.write_text(f"{h}  predictions.csv\n")
    print(h, len(d), d.status.value_counts().to_dict())


def timing(n_groups=2):
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    SP = _s_pop()
    runs = run_list()
    for fam in ("n6400", "SGD", "TaskB"):
        gs = [g for g in _groups(runs) if g[0][0] == fam][:1]
        for (f_, a, seed), g in gs:
            branches, cache = [], {}
            for r in g.itertuples():
                t0 = time.time()
                ck = _path_key(r.set, r.arm)
                if ck in cache:
                    pth, x, y = cache[ck]
                else:
                    pth, x, y = replay_path(r.set, a, int(r.seed), r.arm, HORIZON * SP[a]); cache[ck] = (pth, x, y)
                t1 = time.time()
                res = predict_run(pth, x, y, a, SP[a], branches)
                print(r.set, a, r.seed, r.arm, f"replay {t1 - t0:.1f}s model {time.time() - t1:.1f}s steps {len(pth.s)}",
                      res.get("status"), flush=True)


# ------------------------------------------------------------------------------------------ comparison (after the commit)
ARMS_ORDER = ("lag1", "lag2-prim", "lag2-tstar", "4b", "SGD", "TaskB")
REPORT = ("full", "full_m0H", "a", "b", "c", "c_path", "d", "a_nomom")


def _assert_committed():
    import subprocess
    h = hashlib.sha256(PRED.read_bytes()).hexdigest()
    rec = SHA.read_text().split()[0]
    assert h == rec, f"predictions.csv hash {h} != committed {rec}"
    rel = PRED.relative_to(ROOT)
    out = subprocess.run(["git", "log", "--format=%H", "-1", "--", str(rel)], cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip(), "predictions.csv is not committed"
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel), str(SHA.relative_to(ROOT))], cwd=ROOT)
    assert dirty.returncode == 0, "predictions.csv or its hash differ from the committed version"
    return h, out.stdout.strip()


def observed():
    """The observed crossing |w₂| of every run (read ONLY here, after the committed-hash assertion)."""
    norm = lambda v: str(float(v)) if str(v).replace(".", "", 1).isdigit() else str(v)
    obs = []
    l1 = pd.read_csv(RESULTS / "lag_test_runs.csv", float_precision="round_trip"); l1 = l1[l1.cross_step.notna()]
    obs.append(pd.DataFrame({"set": "lag1", "a": l1.a.round(2), "seed": l1.seed, "arm": l1.factor.map(norm),
                             "s_obs": l1.w2_abs, "step_obs": l1.cross_step}))
    l2 = pd.read_csv(RESULTS / "lag_test2_runs.csv", float_precision="round_trip"); l2 = l2[l2.cross_step.notna()]
    obs.append(pd.DataFrame({"set": np.where(l2.rule == "primary", "lag2-prim", "lag2-tstar"), "a": l2.a.round(2),
                             "seed": l2.seed, "arm": l2.factor.map(norm), "s_obs": l2.w2_abs, "step_obs": l2.cross_step}))
    b = pd.read_csv(RESULTS / "residual_mechanism_parts.csv", float_precision="round_trip"); b = b[b.cross_step.notna()]
    obs.append(pd.DataFrame({"set": "4b", "a": b.a.round(2), "seed": b.seed, "arm": b.arm.astype(str), "s_obs": b.w2_abs,
                             "step_obs": b.cross_step}))
    sg = pd.read_csv(RESULTS / "sgd_own_runs.csv", float_precision="round_trip"); sg = sg[sg.crossed]
    obs.append(pd.DataFrame({"set": "SGD", "a": sg.a.round(2), "seed": sg.seed, "arm": "sgd", "s_obs": sg.w2_cross,
                             "step_obs": sg.step}))
    tb = pd.read_csv(RESULTS / "ts_test" / "runs.csv", float_precision="round_trip"); tb = tb[tb.crossed]
    obs.append(pd.DataFrame({"set": "TaskB", "a": tb.a.round(2), "seed": tb.seed, "arm": "adam", "s_obs": tb.w2_cross,
                             "step_obs": tb.step}))
    o = pd.concat(obs, ignore_index=True)
    o["seed"] = o.seed.astype(int)
    return o


def observed_1A():
    """The 1A observed residual (against each run's GLOBAL own-sample threshold, exactly as lag_law.compare) and that
    threshold, to reproduce the 0.82-0.91 and to separate the change of reference from the model."""
    norm = lambda v: str(float(v)) if str(v).replace(".", "", 1).isdigit() else str(v)
    tc = pd.read_csv(RESULTS / "timescale_consistency" / "runs.csv")
    o = [pd.DataFrame({"set": tc.test, "a": tc.a.round(2), "seed": tc.seed, "arm": tc.arm.astype(str), "obs_r_1A": tc.residual})]
    p = pd.read_csv(RESULTS / "residual_timescale_runs.csv")
    o.append(pd.DataFrame({"set": "lag2-prim", "a": p.a.round(2), "seed": p.seed, "arm": p.factor.astype(str), "obs_r_1A": p.residual}))
    s = pd.read_csv(RESULTS / "sgd_own_runs.csv"); s = s[s.crossed].copy(); s["a"] = s.a.round(2)
    own = pd.read_csv(RESULTS / "own_threshold_crossing.csv"); own["a"] = own.a.round(2)
    s = s.merge(own[["a", "seed", "w2_own"]], on=["a", "seed"])
    o.append(pd.DataFrame({"set": "SGD", "a": s.a, "seed": s.seed, "arm": "sgd", "obs_r_1A": s.w2_cross / s.w2_own - 1,
                           "w2_own_global": s.w2_own}))
    t = pd.read_csv(RESULTS / "ts_test" / "runs.csv"); t = t[t.crossed].copy(); t["a"] = t.a.round(2)
    fo = pd.read_csv(RESULTS / "ts_test_own_frozen.csv"); fo["a"] = fo.a.round(2)
    t = t.merge(fo[["a", "seed", "w2_own"]], on=["a", "seed"])
    o.append(pd.DataFrame({"set": "TaskB", "a": t.a, "seed": t.seed, "arm": "adam", "obs_r_1A": t.w2_cross / t.w2_own - 1,
                           "w2_own_global": t.w2_own}))
    d = pd.concat(o, ignore_index=True)
    d["arm"] = d.arm.map(norm); d["seed"] = d.seed.astype(int)
    so = pd.read_csv(RESULTS / "sample_size_own.csv", float_precision="round_trip")
    so = so[so.n == 6400][["a", "seed", "w2_own"]].copy(); so["a"] = so.a.round(2)
    d = d.merge(so, on=["a", "seed"], how="left")
    d["w2_own_global"] = d.w2_own_global.fillna(d.w2_own)
    return d.drop(columns="w2_own").drop_duplicates(["set", "a", "seed", "arm"])


def _slope(pred, obs):
    ok = np.isfinite(pred) & np.isfinite(obs)
    p, o = np.asarray(pred)[ok], np.asarray(obs)[ok]
    return float((p * o).sum() / (p * p).sum()) if len(p) else float("nan"), int(ok.sum())


def _stats(g, col):
    ok = g[np.isfinite(g[col]) & np.isfinite(g.r_obs)]
    if not len(ok):
        return {"n": 0}
    sl, n = _slope(ok[col].to_numpy(), ok.r_obs.to_numpy())
    rr = ok.r_obs / ok[col]
    both = g[np.isfinite(g[col]) & np.isfinite(g.pred_r_1A)]
    return {"n": n, "median_pred": float(ok[col].median()),
            "median_pred_over_median_1A": float(both[col].median() / both.pred_r_1A.median()) if len(both) else float("nan"), "median_obs": float(ok.r_obs.median()),
            "ratio_of_medians": float(ok.r_obs.median() / ok[col].median()), "median_run_ratio": float(rr.median()),
            "slope_obs_on_pred": sl}


def compare():
    h, commit = _assert_committed()
    pr = pd.read_csv(PRED); pr["arm"] = pr.arm.astype(str); pr["a"] = pr.a.round(2)
    ref = run_list()
    o = observed()
    m = pr.merge(o, on=["set", "a", "seed", "arm"], how="left").merge(ref, on=["set", "a", "seed", "arm"], how="left")
    assert len(m) == len(pr)
    m["r_obs"] = m.s_obs / m.s_star - 1
    m = m.merge(observed_1A(), on=["set", "a", "seed", "arm"], how="left")
    cols = {"1A": "pred_r_1A", "slaved": "r_slaved"}
    for c in STARTS:
        tag = f"{int(round(c * 10))}"
        for v in REPORT:
            cols[f"{v}_{tag}"] = f"{v}_{tag}_r_pred"
    rows = []
    for (st, a, arm), g in m.groupby(["set", "a", "arm"]):
        row = {"set": st, "a": a, "arm": arm, "n_runs": len(g), "n_status_ok": int((g.status == "ok").sum()),
               "n_obs": int(g.s_obs.notna().sum()), "median_r_obs": float(g.r_obs.median()),
               "n_diff_branch_start7": int(g.get("start7_different_branch", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
               "n_diff_branch_start5": int(g.get("start5_different_branch", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
               "n_branch_refined": int(g.branch_refined.fillna(False).astype(bool).sum())}
        for name, col in cols.items():
            if col in g:
                sd = _stats(g, col)
                row[f"{name}_n"] = sd.get("n", 0)
                for k in ("median_pred", "median_pred_over_median_1A", "ratio_of_medians", "median_run_ratio",
                          "slope_obs_on_pred"):
                    row[f"{name}_{k}"] = sd.get(k, float("nan"))
        rows.append(row)
    A = pd.DataFrame(rows)
    A["_o"] = A["set"].map({s_: i for i, s_ in enumerate(ARMS_ORDER)})
    A = A.sort_values(["_o", "a", "arm"]).drop(columns="_o")
    A.to_csv(OUT / "compare_arms.csv", index=False)
    per_a = []
    m["opt"] = np.where(m["set"] == "SGD", "sgd", "adam")
    for (a, opt), g in m.groupby(["a", "opt"]):
        row = {"a": a, "opt": opt, "n_runs": len(g)}
        for name, col in cols.items():
            if col in g:
                sd = _stats(g, col)
                row[f"{name}_n"] = sd.get("n", 0)
                for k in ("median_pred", "median_pred_over_median_1A", "ratio_of_medians", "median_run_ratio",
                          "slope_obs_on_pred"):
                    row[f"{name}_{k}"] = sd.get(k, float("nan"))
        per_a.append(row)
        # the 1A reference: residual against the run's global own-sample threshold (lag_law.compare), on κχ
        g1 = g[np.isfinite(g.obs_r_1A) & np.isfinite(g.pred_r_1A)]
        sl, n = _slope(g1.pred_r_1A.to_numpy(), g1.obs_r_1A.to_numpy())
        row["1A_global_ref_slope_obs_on_pred"] = sl; row["1A_global_ref_n"] = n
        row["median_s_star_over_global_own"] = float((g.s_star / g.w2_own_global).median())
        row["frac_s_star_off_global_own_1pct"] = float(((g.s_star / g.w2_own_global - 1).abs() > 0.01).mean())
    Pa = pd.DataFrame(per_a); Pa.to_csv(OUT / "compare_per_a.csv", index=False)
    m.to_csv(OUT / "compare_runs.csv", index=False)
    summary = {"label": "POST HOC", "predictions_sha256": h, "predictions_commit": commit, "n_runs": int(len(m)),
               "n_status_ok": int((m.status == "ok").sum()), "status_counts": m.status.value_counts().to_dict(),
               "n_obs": int(m.s_obs.notna().sum()), "per_a": Pa.to_dict(orient="records"),
               "arms": A.to_dict(orient="records")}
    for c in STARTS:
        tag = f"{int(round(c * 10))}"
        for v in REPORT:
            k = f"{v}_{tag}"
            summary[f"n_unstable_{k}"] = int(m.get(f"{k}_unstable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
            summary[f"n_diverged_{k}"] = int(m.get(f"{k}_diverged", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
            summary[f"n_pred_{k}"] = int(np.isfinite(m[f"{k}_r_pred"]).sum())
            sl, n = _slope(m[f"{k}_r_pred"].to_numpy(), m.r_obs.to_numpy())
            summary[f"slope_all_{k}"] = sl
        summary[f"n_diff_branch_start{tag}"] = int(m[f"start{tag}_different_branch"].fillna(False).astype(bool).sum())
    ok = m[m.status == "ok"]
    summary["interp_err_theta_sw_max"] = float(ok.interp_err_theta_sw.max())
    summary["interp_err_H_sw_max"] = float(ok.interp_err_H_sw.max())
    summary["rho_sw_nomom_adam_min"] = float(ok[ok.opt == "adam"].rho_sw_nomom.min())
    summary["rho_sw_mom_adam_max"] = float(ok[ok.opt == "adam"].rho_sw_mom.max())
    summary["rho_sw_sgd_max"] = float(ok[ok.opt == "sgd"].rho_sw_nomom.max())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    show = ["set", "a", "arm", "n_runs", "n_obs", "median_r_obs"] + [f"{k}_ratio_of_medians" for k in
                                                                   ("1A", "slaved", "full_7", "full_5", "a_7", "b_7", "c_7", "c_path_7", "d_7")]
    print(A[[c for c in show if c in A]].round(3).to_string(index=False))
    showp = ["a", "opt", "n_runs"] + [f"{k}_slope_obs_on_pred" for k in
                                     ("1A", "slaved", "full_7", "full_5", "a_7", "b_7", "c_7", "c_path_7", "d_7", "a_5", "b_5", "c_5")]
    print(Pa[[c for c in showp if c in Pa]].round(3).to_string(index=False))


def tables():
    """Markdown tables (results/linear_response/tables.md) from compare_arms.csv / compare_per_a.csv; every number in the
    writer inputs comes from here or summary.json."""
    A = pd.read_csv(OUT / "compare_arms.csv"); Pa = pd.read_csv(OUT / "compare_per_a.csv")
    f3 = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:.2f}"
    f4 = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:.4f}"
    keys = [("1A", "κχ (1A)"), ("slaved", "slaved own"), ("full_7", "full 0.7"), ("full_5", "full 0.5"),
            ("a_7", "(a) 0.7"), ("b_7", "(b) 0.7"), ("c_7", "(c) 0.7"), ("c_path_7", "c_path 0.7"), ("d_7", "(d) 0.7"),
            ("a_5", "(a) 0.5"), ("b_5", "(b) 0.5"), ("c_5", "(c) 0.5"), ("d_5", "(d) 0.5")]
    g = lambda r, k: r.get(k, float("nan"))
    L_ = ["### Per arm: observed / predicted lag (ratio of arm medians; number of runs with a prediction in brackets "
          "where fewer than n)", "",
          "| set | a | arm | n | median r_obs | " + " | ".join(k[1] for k in keys) + " | diff. branch 0.7 / 0.5 |",
          "|" + "---|" * (5 + len(keys) + 1)]
    for _, r in A.iterrows():
        cells = []
        for k, _ in keys:
            v, n = g(r, f"{k}_ratio_of_medians"), g(r, f"{k}_n")
            cells.append(f3(v) + (f" ({int(n)})" if np.isfinite(n) and n < r["n_obs"] else ""))
        L_.append(f"| {r['set']} | {r['a']:.2f} | {r['arm']} | {r['n_obs']} | {f4(r['median_r_obs'])} | " + " | ".join(cells)
                  + f" | {r['n_diff_branch_start7']} / {r['n_diff_branch_start5']} |")
    L_ += ["", "### Per a and optimiser: through-origin slope of r_obs on r_pred (median of per-run ratios in brackets)", "",
           "| a | opt | n | 1A vs global own threshold | " + " | ".join(k[1] for k in keys) + " |",
           "|" + "---|" * (4 + len(keys))]
    for _, r in Pa.iterrows():
        cells = [f"{f3(g(r, f'{k}_slope_obs_on_pred'))} ({f3(g(r, f'{k}_median_run_ratio'))})" for k, _ in keys]
        L_.append(f"| {r['a']:.2f} | {r['opt']} | {r['n_runs']} | {f3(r['1A_global_ref_slope_obs_on_pred'])} | "
                  + " | ".join(cells) + " |")
    L_ += ["", "### Per arm: median predicted lag / median κχ (predictions only; sanity of the machinery)", "",
           "| set | a | arm | " + " | ".join(k[1] for k in keys[1:]) + " |", "|" + "---|" * (3 + len(keys) - 1)]
    for _, r in A.iterrows():
        L_.append(f"| {r['set']} | {r['a']:.2f} | {r['arm']} | "
                  + " | ".join(f3(g(r, f"{k}_median_pred_over_median_1A")) for k, _ in keys[1:]) + " |")
    (OUT / "tables.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


# ------------------------------------------------------------------------------------------ post-comparison diagnostic
def diagnose(n_seeds=4):
    """POST HOC, AFTER the comparison (labelled; not a committed prediction).  On the first n_seeds seeds of every arm:
    the exact-gradient reconstruction (the run's own update with the recorded s path, P_t and momentum; see
    `exact_grad`) against the linear full model and the observed crossing.  It isolates the only approximation left in
    the full model, the linearisation of the gradient in δ, and checks the machinery (it must reproduce the observed
    crossing up to the grid-vs-exact gap definition)."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    _assert_committed()
    SP = _s_pop()
    runs = subsample_runs(run_list(), n_seeds)
    f = OUT / "diagnostic_exact_grad.jsonl"
    done = set()
    if f.exists():
        done = {(r["set"], round(r["a"], 2), int(r["seed"]), str(r["arm"])) for r in map(json.loads, f.read_text().splitlines())}
    for (fam, a, seed), g in _groups(runs):
        branches, cache = [], {}
        for r in g.itertuples():
            key = (r.set, round(r.a, 2), int(r.seed), str(r.arm))
            if key in done:
                continue
            ck = _path_key(r.set, r.arm)
            try:
                if ck not in cache:
                    cache[ck] = replay_path(r.set, a, int(r.seed), r.arm, HORIZON * SP[a])
                pth, x, y = cache[ck]
                res = predict_run(pth, x, y, a, SP[a], branches, variants=DIAG_VARIANTS)
            except Exception as e:
                res = {"status": f"error: {type(e).__name__}: {e}"}
            keep = {k: v for k, v in res.items() if k in ("status", "s_star") or k.startswith(("full_", "exact_grad_"))
                    or k in ("start7_t0", "start5_t0")}
            with open(f, "a") as fh:
                fh.write(json.dumps(_jsonable({"set": r.set, "a": a, "seed": int(r.seed), "arm": str(r.arm), **keep})) + "\n")
    d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()])
    d["arm"] = d.arm.astype(str); d["a"] = d.a.round(2)
    d = d.merge(observed(), on=["set", "a", "seed", "arm"], how="left")
    rows = []
    for tag in ("7", "5"):
        e = d[f"exact_grad_{tag}_s_pred"].astype(float); fu = d[f"full_{tag}_s_pred"].astype(float)
        step_e = d[f"start{tag}_t0"] + d[f"exact_grad_{tag}_step_rel"].astype(float)
        r_obs = d.s_obs / d.s_star - 1
        rows.append({"start": float(tag) / 10, "n": int(e.notna().sum()),
                     "n_exact_step_equals_observed": int((step_e == d.step_obs).sum()),
                     "n_exact_step_within_1_of_observed": int(((step_e - d.step_obs).abs() <= 1).sum()),
                     "max_abs_rel_s_exact_vs_obs": float((e / d.s_obs - 1).abs().max()),
                     "median_r_obs": float(r_obs.median()), "median_r_exact_grad": float((e / d.s_star - 1).median()),
                     "median_r_full": float((fu / d.s_star - 1).median()),
                     "slope_obs_on_exact_grad": _slope((e / d.s_star - 1).to_numpy(), r_obs.to_numpy())[0],
                     "slope_obs_on_full": _slope((fu / d.s_star - 1).to_numpy(), r_obs.to_numpy())[0],
                     "slope_exact_grad_on_full": _slope((fu / d.s_star - 1).to_numpy(), (e / d.s_star - 1).to_numpy())[0]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "diagnostic_exact_grad.csv", index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "time":
        timing()
    elif cmd == "predict":
        predict(subsample=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "finalize":
        finalize()
    elif cmd == "compare":
        compare()
    elif cmd == "tables":
        tables()
    elif cmd == "diagnose":
        diagnose(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
