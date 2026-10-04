"""Phase 2A: slow tracking on the simplicity-bias benchmark (design: results/designs/phase2a_slow_sb_design.md, approved
by the author 2026-10-04; registration: results/phase2a_registration.md).

Prediction: a run released on the linear-dominant branch M at s₀ = 1.7957 and trained with the output SCALE slowed by ρ
(v ← v − η[(I − ââᵀ) + ρââᵀ]∇_vL, â = sign(v)/√3 on the active units; η = 1 on W, c, b) crosses (the first upward
passage of q by ρ₂) at M's fold s_F = 4.7677, delayed by s_F·Ω₀ε_F^{2/3}, not at the switch s* = 3.5914.  ONE landscape
(the fixed v3 data) across 27 distinct rates: the rates, not the seeds, are the replicates.

ORDER (no step reads anything a later step produces):
  scan / freeze / pilot / manifest   before the registration commit.  pilot: pilot seeds only (2,937,900-2,937,959),
                                     three pilot rates off the ladder; sets τ₁, τ₂; the STOP rule.
  run                                after the registration commit and its timestamp: holds of the 120 registered seeds
                                     (fixed-s BFGS at s₀, labels), the gate, the seed → rate plan (the i-th M landing
                                     takes the i-th ladder rate, fastest first), then per rate: training from the exact
                                     M(s₀) to t_c (first s ≥ 0.95·s_F; NO ρ₂, no branch check), the state at t_c hashed,
                                     the causal fold forecast (GuardedArray, rows < t_c; NaN recomputation identical).
                                     -> holds.jsonl, runs.jsonl
  finalize                           forecasts.sha256 (holds.jsonl, runs.jsonl; no observed key); COMMIT before observe
  observe                            per rate: retrain from the release with ρ₂ EVERY step, the state at t_c asserted
                                     equal to the hashed one bit for bit, observation continued from it to the crossing
                                     (and s_F) or 1.25·s_F; follow check at 0.8·s_F; idle unit and signs every step;
                                     the tightening-cutoff forecasts (DESCRIPTIVE).  -> observed.jsonl
  score                              scores.json

Machine rules: one process, nice 15, one thread; a memory gate (free ≥ 25%, swap free ≥ 500 MB; waits) and a disk check
(free ≥ 20 GB; else STOP) before every job, logged to results/phase2a/memory_gate.log; stop above 3 GB RSS.  No global
torch / numpy RNG state: the init uses a LOCAL torch.Generator; bootstraps use local numpy Generators.

    python -m src.phase2a scan | freeze | pilot | manifest
    python -m src.phase2a run | finalize | observe | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import causal_forecast as CF
from . import causal_forecast_fold as CFF
from . import simplicity_bias_v2 as v2
from . import sb_fold as SBF

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase2a"
REGISTRATION_MD = RESULTS / "phase2a_registration.md"
DESIGN_MD = RESULTS / "designs" / "phase2a_slow_sb_design.md"
V2_FROZEN = RESULTS / "simplicity_bias_v2" / "frozen.json"

# ------------------------------------------------------------------------------------------ registered constants
S0 = 1.7957                                     # hold / release scale
ETA = 1.0
LAM = 1e-4
LADDER_LOG2 = (-13.0, -14.0, -15.0) + tuple(-(15 + k / 8) for k in range(1, 25))       # 27 rates, fastest first
PILOT_LOG2 = (-15.0625, -16.5625, -17.9375)     # off the ladder (half-steps k 0.5, 12.5, 23.5; author 2026-10-04,
                                                # S1 (b): the middle rate moved from 2^-16.5 = ladder k 12); the STOP
                                                # rule reads the first
SEEDS = tuple(range(2_937_000, 2_937_120))      # 120 registered seeds
PILOT_SEEDS = tuple(range(2_937_900, 2_937_960))  # 60 pilot seeds
N_RATES = len(LADDER_LOG2)
# Author 2026-10-04: every VERDICT is computed over the 22 ladder rates with no prior outcome.  The 5 rates already run
# in the exploration (p2a_explore_f, _g) are still RUN (all 27 run, one per M seed) and reported as DESCRIPTIVE only.
PREOBSERVED_LOG2 = (-13.0, -14.0, -15.0, -16.0, -17.0)
SCORED_LOG2 = tuple(l2 for l2 in LADDER_LOG2 if l2 not in PREOBSERVED_LOG2)
N_SCORED = len(SCORED_LOG2)                      # 22
GATE_MIN_M = 27                                 # gate: ≥ 27 of 120 seeds land on M
F_CUT = 0.95                                    # registered cutoff fraction of s_F
TIGHT_REF_LOG2, TIGHT_REF_GAP = -15.0, 0.05     # descriptive tightening cutoff 1 − f = 0.05·(ρ/2⁻¹⁵)^{2/3}
OBS_END = 1.25                                  # observation to 1.25·s_F
FOLLOW_FRAC = 0.8                               # follow check at 0.8·s_F
LABEL_TOL = 1e-3                                # branch label: function-space distance ≤ 1e-3
BUDGET_FACTOR = 64.0                            # step budget ⌈64/ρ⌉ (≈ 2× the steps to 1.25·s_F)
F_MIN_FRAC, F_HI, F_FALSIFIER = 0.90, 1.25, 1.25   # F: s_F ≤ s_obs ≤ 1.25·s_F at ≥ 90%; none ≤ 1.25·s*
WITHIN_MIN_FRAC = 0.80                          # C1, C2
C3_MIN_SPEARMAN = 0.8                           # C3: Spearman(ρ, r_obs/r_fc) ≥ 0.8 and every ratio > 1
E_BAND = (0.55, 0.80)                           # E: 95% CI of the slope inside the band
VALID_MIN_FRAC = 0.90                           # follow check, cutoff before the crossing
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 2_937_000, (2.5, 97.5)   # C4 and E (a fresh local generator each)
MIN_RATES_STAT = 3                              # C3 and E need ≥ 3 rates with a value; C4 ≥ 2
PILOT_EPS_MAX, PILOT_CHI_MAX = 0.01, 0.1        # STOP rule at the first pilot rate
CHI_WINDOW = 100                                # χ_t: d log s/dt over the last min(100, t) steps (v3's chi_at)
TAU_FACTOR, TAU_ROUND = 1.5, 0.05               # τ = 1.5 × the largest pilot error, rounded up to 0.05
OMEGA0 = CFF.OMEGA0
HOLD_GTOL = 1e-8
RSS_LIMIT = 3 * 1024 ** 3
DISK_MIN_BYTES = 20 * 1024 ** 3
ARTIFACTS = ("phase2a/seed_scan.json", "phase2a/frozen.json", "phase2a/pilot.json", "phase2a/registration.sha256",
             "phase2a/holds.jsonl", "phase2a/runs.jsonl", "phase2a/forecasts.sha256", "phase2a/observed.jsonl",
             "phase2a/scores.json")

X, Y = v2.data()
N = len(Y)


def rho_of(l2):
    return 2.0 ** float(l2)


def budget(rho):
    return int(math.ceil(BUDGET_FACTOR / rho))


def f_tight(rho):
    """DESCRIPTIVE tightening cutoff: 1 − f ∝ ρ^{2/3}, equal to 0.05 at 2⁻¹⁵."""
    return 1.0 - TIGHT_REF_GAP * (rho / 2.0 ** TIGHT_REF_LOG2) ** (2.0 / 3.0)


# ------------------------------------------------------------------------------------------ model (training coordinates)
def unpack(th):
    return th[:8].reshape(4, 2), th[8:12], th[12:16], th[16]


def loss_grad(th, lam=LAM):
    """BCE + (λ/2)(|W|² + |c|²) of z = tanh(XWᵀ + c)·v + b and its analytic gradient (17 = W 8, c 4, v 4, b)."""
    W, c, v, b = unpack(th)
    H = np.tanh(X @ W.T + c)
    z = H @ v + b
    L = float(np.mean(np.logaddexp(0.0, z) - Y * z) + 0.5 * lam * ((W ** 2).sum() + (c ** 2).sum()))
    r = (0.5 * (1 + np.tanh(0.5 * z)) - Y) / N
    D = (1 - H ** 2) * (r[:, None] * v[None, :])
    g = np.empty(17)
    g[:8] = (D.T @ X + lam * W).ravel()
    g[8:12] = D.sum(0) + lam * c
    g[12:16] = H.T @ r
    g[16] = r.sum()
    return L, g


def make_step(rho, active, eta=ETA):
    """The scale-only rule: (W, c, b) ← · − η∇; v ← v − η[(I − ââᵀ) + ρââᵀ]∇_vL, â = sign(v)/√n on the ACTIVE units
    (fixed at release; the idle unit's â component is 0)."""
    act = np.asarray(active, int)
    n = len(act)

    def step(th):
        _, g = loss_grad(th)
        new = th.copy()
        new[:12] -= eta * g[:12]
        new[16] -= eta * g[16]
        gv = g[12:16]
        a = np.zeros(4)
        a[act] = np.sign(th[12:16][act]) / math.sqrt(n)
        gp = a * (a @ gv)
        new[12:16] -= eta * ((gv - gp) + rho * gp)
        return new
    return step


def scale(th):
    return float(np.abs(th[12:16]).sum())


# ------------------------------------------------------------------------------------------ ρ₂ (every step)
_U = [np.unique(X[:, j]) for j in range(2)]
_A = np.searchsorted(_U[0], X[:, 0])
_B = np.searchsorted(_U[1], X[:, 1])
_N1, _N2 = len(_U[0]), len(_U[1])
_CNT = [np.bincount(_A, minlength=_N1).astype(float), np.bincount(_B, minlength=_N2).astype(float)]
_I0 = _A * _N2 + _B
_I1 = (np.arange(_N1)[:, None] * _N2 + _B[None, :]).ravel()          # φ(x₁ := t, x₂ = x_i2)
_I2 = (_A[None, :] * _N2 + np.arange(_N2)[:, None]).ravel()          # φ(x₁ = x_i1, x₂ := t)
_W1 = np.repeat(_CNT[0], N)
_W2 = np.repeat(_CNT[1], N)
_UX = _U[0][:, None, None]
_UY = _U[1][None, :, None]


def rho2(th):
    """ρ₂ = V₂/(V₁ + V₂) of φ(x) = Σ vₖ tanh(wₖ·x + cₖ) (sb.feature_usage's S-randomisation, every replacement value
    weighted by its count), evaluated on the product grid of the data's coordinate values (the data are a product grid,
    so every replaced point lies on it).  Equal to v2.rho2_batch to rounding (tested)."""
    W = th[:8].reshape(4, 2)
    G = (np.tanh(_UX * W[:, 0] + (_UY * W[:, 1] + th[8:12])) @ th[12:16]).ravel()
    base = G[_I0]
    V1 = _W1 @ np.abs(G[_I1] - np.tile(base, _N1))
    V2 = _W2 @ np.abs(G[_I2] - np.tile(base, _N2))
    return float(V2 / (V1 + V2))


# ------------------------------------------------------------------------------------------ init, hold, labels
def init_row(seed):
    """v3's init (src/simplicity_bias_v3.init_net: PyTorch-default Linear(2, 4) and Linear(4, 1), float32, cast to
    float64), drawn from a LOCAL torch.Generator seeded with `seed` in the same order (hidden weight, hidden bias, output
    weight, output bias) with the same bounds, so it is bit-identical to init_net after torch.manual_seed(seed) (tested)
    without touching the global RNG.  Returns the 17-row (W 8, c 4, v 4, b)."""
    import torch
    g = torch.Generator().manual_seed(int(seed))

    def kaiming(shape, fan_in):
        gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
        bound = math.sqrt(3.0) * (gain / math.sqrt(fan_in))
        return torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound, generator=g)

    def bias(n, fan_in):
        bound = 1 / math.sqrt(fan_in)
        return torch.empty(n, dtype=torch.float32).uniform_(-bound, bound, generator=g)
    W = kaiming((4, 2), 2); c = bias(4, 2); v = kaiming((1, 4), 4); b = bias(1, 4)
    return np.concatenate([t.double().numpy().ravel() for t in (W, c, v, b)])


def rescaled_init(seed, s0=S0):
    row = init_row(seed)
    row[12:16] *= s0 / np.abs(row[12:16]).sum()
    return row


def to_landscape(th):
    return SBF.run_to_full(np.atleast_2d(np.asarray(th, float)))


def branch_P(name, s):
    """16-vector of a branch's point at exactly s (sb_fold.branch_point: Newton from the nearest stored stable point);
    None outside its stable range or if Newton fails."""
    r = SBF.branch_point(name, s)
    return None if r is None else r[1].to_full(r[0])[0]


def branch_theta(name, s):
    """The branch point at exactly s in TRAINING coordinates (idle units exactly 0; v = s·η²/Ση² on the active units)."""
    r = SBF.branch_point(name, s)
    if r is None:
        return None
    u, R = r
    n = R.nA
    W = u[:2 * n].reshape(n, 2); c = u[2 * n:3 * n]; eta = np.r_[1.0, u[3 * n:4 * n - 1]]; b = u[-1]
    th = np.zeros(17)
    Wf = np.zeros((4, 2)); Wf[:n] = W
    th[:8] = Wf.ravel(); th[8:8 + n] = c; th[12:12 + n] = s * eta ** 2 / (eta ** 2).sum(); th[16] = b
    return th


def hold(seeds, s0=S0):
    """Fixed-s BFGS of each seed's rescaled v3 init (sb_fold.local_min_batch: 16 landscape coordinates, b profiled,
    gtol 1e-8; one batch in seed order).  Returns (P, loss, gnorm)."""
    rows = np.array([rescaled_init(s) for s in seeds])
    return SBF.local_min_batch(SBF.run_to_full(rows), np.full(len(seeds), s0), X, Y, gtol=HOLD_GTOL)


def label_at(P, branch_points):
    """Label of a held minimiser: the branch whose frozen point at s₀ lies within 1e-3 (function space, sb_fold.fdist);
    the nearest if several; else 'other'.  Returns (label, {branch: distance})."""
    d = {k: (float(SBF.fdist(np.atleast_2d(P), np.atleast_2d(np.asarray(Q, float)), X)[0]) if Q is not None else math.inf)
         for k, Q in branch_points.items()}
    k = min(d, key=d.get)
    return (k if d[k] <= LABEL_TOL else "other"), d


def plan(labels, seeds=SEEDS, ladder=LADDER_LOG2):
    """The seed → rate plan: the i-th seed (in seed order) labelled M takes the i-th ladder rate (fastest first); M
    seeds beyond the ladder and off-M seeds are counted, not trained."""
    m_seeds = [s for s, lab in zip(seeds, labels) if lab == "M"]
    return [(s, l2) for s, l2 in zip(m_seeds, ladder)], m_seeds[len(ladder):]


def in_M_basin(th):
    """Follow check: the state minimised at its own s (sb_fold.local_min_batch) lies within 1e-3 of M's point at that
    exact s.  (bool, distance)."""
    s = scale(th)
    P, _, _ = SBF.local_min_batch(to_landscape(th), np.array([s]), X, Y)
    Q = branch_P("M", s)
    if Q is None:
        return False, math.inf
    d = float(SBF.fdist(P, Q[None], X)[0])
    return bool(d <= LABEL_TOL), d


# ------------------------------------------------------------------------------------------ M table: λ_min, |m′c′|
def hessian(th, lam=LAM):
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float64); Yt = torch.as_tensor(Y, dtype=torch.float64)

    def Lf(q):
        W = q[:8].reshape(4, 2); c = q[8:12]; v = q[12:16]; b = q[16]
        z = torch.tanh(Xt @ W.T + c) @ v + b
        return torch.nn.functional.binary_cross_entropy_with_logits(z, Yt) + 0.5 * lam * ((W ** 2).sum() + (c ** 2).sum())
    return torch.autograd.functional.hessian(Lf, torch.tensor(th, dtype=torch.float64)).numpy()


def tangent_lmin(th):
    """λ_min of the training-coordinate Hessian on the fixed-s tangent of the ACTIVE units (Σ sign(v)·δv = 0; P = I:
    the scale-only rule's metric there), as p2a_explore_a with ρ = 1."""
    H = hessian(th)
    act = [k for k in range(4) if abs(th[12 + k]) > 1e-12]
    idx = [2 * k for k in act] + [2 * k + 1 for k in act] + [8 + k for k in act] + [12 + k for k in act] + [16]
    nv = len(act); m = len(idx)
    a = np.zeros(m); a[3 * nv:4 * nv] = np.sign(th[[12 + k for k in act]]); a /= np.linalg.norm(a)
    U, _, _ = np.linalg.svd(np.eye(m) - np.outer(a, a))
    Qb = U[:, :m - 1]
    return float(np.linalg.eigvalsh(Qb.T @ H[np.ix_(idx, idx)] @ Qb)[0])


def theta_from_u(u, nA, s):
    W = u[:2 * nA].reshape(nA, 2); c = u[2 * nA:3 * nA]; eta = np.r_[1.0, u[3 * nA:4 * nA - 1]]
    th = np.zeros(17); Wf = np.zeros((4, 2)); Wf[:nA] = W
    th[:8] = Wf.ravel(); th[8:8 + nA] = c; th[12:12 + nA] = s * eta ** 2 / (eta ** 2).sum(); th[16] = u[-1]
    return th


def fold_constant_fit(ss, lam, s_F, window):
    m = ss >= s_F - window
    d = s_F - ss[m]
    coef, *_ = np.linalg.lstsq(np.vstack([d, d ** 2]).T, lam[m] ** 2, rcond=None)
    return float(abs(coef[0]) / 4), int(m.sum())


def lmin_interp(s, table):
    """λ_min of M at s (linear interpolation in the frozen table; NaN outside it)."""
    ts, tl = np.asarray(table["s"], float), np.asarray(table["lam_min"], float)
    return np.interp(s, ts, tl, left=np.nan, right=np.nan)


# ------------------------------------------------------------------------------------------ training and forecast
def train_to_cutoff(theta0, rho, s_F, f=F_CUT, n_budget=None):
    """Training from the release to t_c = the first step t ≥ 1 with s_t ≥ f·s_F (a stopping time on s; the harness
    reads s_t only).  No ρ₂, no branch check.  Returns (s_0 … s_{t_c}, θ_{t_c}, t_c); t_c None if the budget ends first
    (then the path is the whole budget)."""
    th = np.array(theta0, float)
    active = [k for k in range(4) if th[12 + k] != 0.0]
    step = make_step(rho, active)
    nb = budget(rho) if n_budget is None else int(n_budget)
    S = np.empty(nb + 1); S[0] = scale(th)
    level = f * s_F
    for t in range(1, nb + 1):
        th = step(th)
        S[t] = scale(th)
        if S[t] >= level:
            return S[:t + 1].copy(), th, t
        if t % 1_000_000 == 0:
            rss_guard()
    return S, th, None


def state_sha(th):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(th, dtype="<f8")).tobytes()).hexdigest()


def chi_path(S, t_from, t_to, table, eta=ETA, window=CHI_WINDOW):
    """χ_t = (d log s/dt over the last min(100, t) steps)/(η λ_min(s_t)) for t_from ≤ t < t_to (v3's chi_at with the
    frozen M table's λ_min, P = I on the fixed-s tangent)."""
    t = np.arange(max(1, t_from), t_to)
    if not len(t):
        return np.zeros(0)
    w = np.minimum(window, t)
    rate = (np.log(S[t]) - np.log(S[t - w])) / w
    return rate / (eta * lmin_interp(S[t], table))


def run_rate(l2, fr, n_budget=None):
    """One rate: training from the frozen release to t_c, the state hashed, the causal forecast at f = 0.95 (guarded;
    NaN recomputation identical, else CausalityViolation), and the STOP-rule quantities (ε̂_F, max χ on [s*, t_c))."""
    rho = rho_of(l2)
    s_F, LamF = fr["s_F"], fr["Lambda_F"]
    S, th, t_c = train_to_cutoff(np.array(fr["release_theta"]), rho, s_F, F_CUT, n_budget)
    nb = budget(rho) if n_budget is None else int(n_budget)
    rec = {"log2rho": float(l2), "rho": rho, "budget": nb, "t_c": t_c}
    if t_c is None:
        rec.update(forecast={"status": "no forecast: no cutoff within the budget", "t_fc": None, "t_F_fc": None},
                   nan_recompute_identical=None)
        return rec
    fc, same = CFF.run_forecast(S, t_c, s_F, LamF, ETA, nb - t_c + 1, F_CUT)
    if not same:
        raise CF.CausalityViolation(f"NaN recomputation differs at log2rho {l2}")
    t_star = CF.cutoff_step(S[:t_c], fr["s_star"])          # first t with s ≥ s* (before t_c), None if none
    chis = chi_path(S, t_star, t_c, fr["M_table"]) if t_star is not None else np.zeros(0)
    rec.update(s_at_tc=float(S[t_c]), state_tc=[float(x) for x in th], state_tc_sha256=state_sha(th), forecast=fc,
               nan_recompute_identical=bool(same), t_sstar=t_star,
               chi_max_sstar_tc=float(np.nanmax(chis)) if len(chis) else None,
               chi_n_nan=int(np.isnan(chis).sum()))
    return rec


def observe_rate(l2, fr, run_rec, n_budget=None):
    """Observation (after the forecasts are committed): retrain from the release with ρ₂ at EVERY step (v3's event: the
    first t with ρ₂(t) ≥ q after an earlier step with ρ₂ < q); at t_c assert the state equals the hashed one bit for bit,
    and continue from it until the crossing and s_F are both reached, or s ≥ 1.25·s_F, or the budget.  Records t_obs,
    s_obs, t_F (first s ≥ s_F), the follow check at 0.8·s_F, the idle unit (exactly 0) and the active signs up to the
    crossing.  Returns (record, s path)."""
    rho = rho_of(l2)
    s_F, q = fr["s_F"], fr["q"]
    th = np.array(fr["release_theta"], float)
    active = [k for k in range(4) if th[12 + k] != 0.0]
    idle = [k for k in range(4) if k not in active]
    idle_idx = [i for k in idle for i in (2 * k, 2 * k + 1, 8 + k, 12 + k)]
    sg0 = np.sign(th[12:16][active])
    step = make_step(rho, active)
    nb = budget(rho) if n_budget is None else int(n_budget)
    t_c = run_rec["t_c"]
    S = np.empty(nb + 1); S[0] = scale(th)
    r0 = rho2(th)
    below = r0 < q
    t_obs = None; rho2_obs = None; t_F = None; t08 = None; th08 = None
    idle_viol = None; sign_viol = None; hash_ok = None
    t = 0
    for t in range(1, nb + 1):
        th = step(th)
        S[t] = scale(th)
        if t_obs is None:
            if idle_viol is None and idle_idx and np.any(th[idle_idx] != 0.0):
                idle_viol = t
            if sign_viol is None and np.any(np.sign(th[12:16][active]) != sg0):
                sign_viol = t
            r = rho2(th)
            if r >= q and below:
                t_obs, rho2_obs = t, r
            elif r < q:
                below = True
        if t08 is None and S[t] >= FOLLOW_FRAC * s_F:
            t08, th08 = t, th.copy()
        if t_F is None and S[t] >= s_F:
            t_F = t
        if t_c is not None and t == t_c:
            hash_ok = bool(state_sha(th) == run_rec["state_tc_sha256"]
                           and np.array_equal(th, np.array(run_rec["state_tc"], float)))
            if not hash_ok:
                raise AssertionError(f"state at t_c differs from the hashed one (log2rho {l2})")
            th = np.array(run_rec["state_tc"], float)           # observation resumes from the hashed state
        if S[t] >= OBS_END * s_F:
            break
        if t_obs is not None and t_F is not None and (t_c is None or t >= t_c):
            break
        if t % 1_000_000 == 0:
            rss_guard()
    S = S[:t + 1].copy()
    follow = in_M_basin(th08) if th08 is not None else (False, math.inf)
    rec = {"log2rho": float(l2), "steps_observed": int(t), "t_obs": t_obs,
           "s_obs": float(S[t_obs]) if t_obs is not None else None, "rho2_at_obs": rho2_obs, "rho2_release": r0,
           "t_F": t_F, "s_end": float(S[-1]), "t_08": t08, "follow_in_M": bool(follow[0]), "follow_dist_M": follow[1],
           "idle_zero_to_crossing": idle_viol is None, "idle_first_violation": idle_viol,
           "signs_fixed_to_crossing": sign_viol is None, "sign_first_violation": sign_viol, "state_tc_hash_ok": hash_ok}
    return rec, S


def tight_forecast(l2, fr, S):
    """DESCRIPTIVE: the same causal forecaster at the tightening cutoff f(ρ) = 1 − 0.05·(ρ/2⁻¹⁵)^{2/3} (guarded; NaN
    recomputation reported)."""
    rho = rho_of(l2)
    f = f_tight(rho)
    t_c = CF.cutoff_step(S, f * fr["s_F"])
    if t_c is None:
        return {"f": f, "t_c": None, "status": "no cutoff on the observed path"}
    fc, same = CFF.run_forecast(S, t_c, fr["s_F"], fr["Lambda_F"], ETA, budget(rho) - t_c + 1, f)
    return {"f": f, **fc, "nan_recompute_identical": bool(same)}


# ------------------------------------------------------------------------------------------ pure scoring rules
def _arr(v, dt=float):
    if isinstance(v, (list, tuple)):
        return np.array([np.nan if (dt is float and x is None) else x for x in v], dtype=dt)
    return np.asarray(v, dt)


def verdict(ok, computable=True):
    return "UNRESOLVED" if not computable else ("PASS" if ok else "FAIL")


def criterion_F(s_obs, s_F, s_star, n_rates):
    """F: s_F ≤ s_obs ≤ 1.25·s_F at ≥ 90% of the rates (a rate with no crossing is not within), and at no rate
    s_obs ≤ 1.25·s* (the falsifier).  No rate: UNRESOLVED."""
    s_obs = _arr(s_obs)
    with np.errstate(invalid="ignore"):
        within = np.isfinite(s_obs) & (s_obs >= s_F) & (s_obs <= F_HI * s_F)
        fals = np.isfinite(s_obs) & (s_obs <= F_FALSIFIER * s_star)
    n = int(n_rates)
    frac = int(within.sum()) / n if n else float("nan")
    ok = bool(n and frac >= F_MIN_FRAC and not fals.any())
    return {"n": n, "n_within": int(within.sum()), "n_no_crossing": int((~np.isfinite(s_obs)).sum()), "frac": frac,
            "min_frac": F_MIN_FRAC, "n_falsifier": int(fals.sum()), "falsified": bool(fals.any()),
            "verdict": verdict(ok, n > 0)}


def criterion_within(err, delay_fc, tau, has_fc, n_rates):
    """C1 / C2: |err| ≤ τ·delay_fc (inclusive) at ≥ 80% of the rates.  A rate with no forecast, or with err undefined
    (no crossing), is NOT within.  No rate: UNRESOLVED."""
    err, delay_fc, has_fc = _arr(err), _arr(delay_fc), _arr(has_fc, bool)
    with np.errstate(invalid="ignore"):
        ok = has_fc & np.isfinite(err) & np.isfinite(delay_fc) & (np.abs(err) <= tau * delay_fc)
    n = int(n_rates)
    frac = int(ok.sum()) / n if n else float("nan")
    return {"n": n, "n_within": int(ok.sum()), "n_no_forecast": int((~has_fc).sum()),
            "n_err_undefined": int((has_fc & ~np.isfinite(err)).sum()), "frac_within": frac, "tau": tau,
            "min_frac": WITHIN_MIN_FRAC, "verdict": verdict(n and frac >= WITHIN_MIN_FRAC, n > 0)}


def spearman(x, y):
    """Spearman correlation with average ranks for ties (NaN if a variable is constant)."""
    def rank(a):
        a = np.asarray(a, float)
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def criterion_C3(rho, r_obs, r_fc, has_fc):
    """C3 (asymptotic): over the rates with a forecast and a crossing (finite r_obs/r_fc, r_fc > 0): Spearman(ρ,
    r_obs/r_fc) ≥ 0.8 AND r_obs/r_fc > 1 at every one of them.  Fewer than 3 such rates, or Spearman NaN: UNRESOLVED."""
    rho, r_obs, r_fc, has_fc = _arr(rho), _arr(r_obs), _arr(r_fc), _arr(has_fc, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(has_fc & (r_fc > 0), r_obs / r_fc, np.nan)
    ok = np.isfinite(ratio)
    n = int(ok.sum())
    sp = spearman(rho[ok], ratio[ok]) if n >= MIN_RATES_STAT else float("nan")
    all_gt1 = bool(n and (ratio[ok] > 1).all())
    computable = n >= MIN_RATES_STAT and np.isfinite(sp)
    return {"n": n, "n_excluded": int(len(ratio) - n), "spearman": sp, "min_spearman": C3_MIN_SPEARMAN,
            "ratio_min": float(ratio[ok].min()) if n else None, "ratio_max": float(ratio[ok].max()) if n else None,
            "all_ratios_gt_1": all_gt1,
            "verdict": verdict(computable and sp >= C3_MIN_SPEARMAN and all_gt1, computable)}


def bootstrap_mean_ci(D, n_boot=BOOT_N, seed=BOOT_SEED):
    """(mean, lower, upper): 95% percentile bootstrap of the mean over rate resamples (local default_rng(seed))."""
    D = np.asarray(D, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    lo, hi = np.percentile(D[idx].mean(axis=1), BOOT_PCT)
    return float(D.mean()), float(lo), float(hi)


def criterion_C4(t_fc, t_F_fc, t_obs, has_fc):
    """C4: D = |t_fc − t_obs| − |t̂_F − t_obs| (t̂_F = the no-delay forecast), paired per rate.  Any rate with no
    forecast: FAIL (the page: no forecast fails C4).  Otherwise over the rates with a crossing: PASS iff the upper end of
    the 95% percentile bootstrap interval of mean D (10,000 resamples) is < 0; an interval above 0 or containing 0 is
    FAIL; fewer than 2 rates: UNRESOLVED."""
    t_fc, t_F_fc, t_obs, has_fc = _arr(t_fc), _arr(t_F_fc), _arr(t_obs), _arr(has_fc, bool)
    n_nofc = int((~has_fc).sum())
    ok = has_fc & np.isfinite(t_obs)
    D = np.abs(t_fc[ok] - t_obs[ok]) - np.abs(t_F_fc[ok] - t_obs[ok])
    n = int(ok.sum())
    mean, lo, hi = bootstrap_mean_ci(D) if n >= 2 else (float("nan"),) * 3
    computable = bool(n >= 2 and np.isfinite(hi))
    if n_nofc:
        v = "FAIL"
    else:
        v = verdict(hi < 0, computable)
    return {"n": n, "n_no_forecast": n_nofc, "n_no_crossing": int((has_fc & ~np.isfinite(t_obs)).sum()),
            "mean_D": mean, "ci95": [lo, hi], "boot_n": BOOT_N, "boot_seed": BOOT_SEED, "verdict": v,
            "DESCRIPTIVE_frac_forecast_closer": float((D < 0).mean()) if n else None}


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm = x - x.mean()
    sxx = float((xm ** 2).sum())
    return float((xm * (y - y.mean())).sum() / sxx) if sxx > 0 else float("nan")


def criterion_E(eps_fc, r_obs, has_fc):
    """E: the OLS slope of ln r_obs on ln ε̂_F over the rates with a forecast and a crossing above s_F (r_obs > 0); its
    95% CI = the percentile interval of the slope over 10,000 rate resamples (local default_rng(2,937,000); resamples
    with no spread in ε̂ are dropped, counted) must lie inside [0.55, 0.80] (closed).  Fewer than 3 rates:
    UNRESOLVED.  The OLS t-interval is DESCRIPTIVE."""
    eps_fc, r_obs, has_fc = _arr(eps_fc), _arr(r_obs), _arr(has_fc, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = has_fc & np.isfinite(eps_fc) & (eps_fc > 0) & np.isfinite(r_obs) & (r_obs > 0)
    x, y = np.log(eps_fc[ok]), np.log(r_obs[ok])
    n = int(ok.sum())
    if n < MIN_RATES_STAT:
        return {"n": n, "slope": None, "ci95": [None, None], "band": list(E_BAND), "verdict": "UNRESOLVED"}
    slope = ols_slope(x, y)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(BOOT_N, n))
    xs, ys = x[idx], y[idx]
    xm = xs - xs.mean(axis=1, keepdims=True)
    sxx = (xm ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(sxx > 0, (xm * (ys - ys.mean(axis=1, keepdims=True))).sum(axis=1) / sxx, np.nan)
    good = np.isfinite(b)
    lo, hi = np.percentile(b[good], BOOT_PCT) if good.any() else (float("nan"), float("nan"))
    # descriptive OLS t-interval
    resid = y - (y.mean() + slope * (x - x.mean()))
    se = math.sqrt(float((resid ** 2).sum()) / (n - 2) / float(((x - x.mean()) ** 2).sum())) if n > 2 else float("nan")
    computable = bool(np.isfinite(lo) and np.isfinite(hi) and np.isfinite(slope))
    return {"n": n, "slope": slope, "ci95": [float(lo), float(hi)], "band": list(E_BAND),
            "n_degenerate_resamples": int((~good).sum()), "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
            "DESCRIPTIVE_ols_se": se,
            "verdict": verdict(computable and E_BAND[0] <= lo and hi <= E_BAND[1], computable)}


def gate(labels):
    n_m = int(sum(1 for lab in labels if lab == "M"))
    return {"n_seeds": len(labels), "n_M": n_m, "min_M": GATE_MIN_M, "pass": bool(n_m >= GATE_MIN_M),
            "counts": {k: int(sum(1 for lab in labels if lab == k)) for k in sorted(set(labels))}}


def validity(follow_in_M, t_c, t_obs, idle_ok, sign_ok, n_rates):
    """At ≥ 90% of the rates: the state at 0.8·s_F lies in M's basin (a rate that never reached 0.8·s_F fails it); at
    ≥ 90% of the rates: t_c comes strictly before the crossing (a rate with no cutoff fails it; a rate with a cutoff
    and no crossing passes it); at EVERY rate: the idle unit stays exactly 0 and the active signs stay fixed up to the
    crossing (to the end of observation if none)."""
    follow, t_c, t_obs = _arr(follow_in_M, bool), _arr(t_c), _arr(t_obs)
    idle_ok, sign_ok = _arr(idle_ok, bool), _arr(sign_ok, bool)
    n = int(n_rates)
    with np.errstate(invalid="ignore"):
        before = np.isfinite(t_c) & (~np.isfinite(t_obs) | (t_c < t_obs))
    ff = float(follow.sum()) / n if n else float("nan")
    fb = float(before.sum()) / n if n else float("nan")
    out = {"n": n, "follow_frac": ff, "follow_ok": bool(n and ff >= VALID_MIN_FRAC),
           "cutoff_before_frac": fb, "cutoff_before_ok": bool(n and fb >= VALID_MIN_FRAC),
           "n_cutoff_not_before": int(n - before.sum()),
           "idle_zero_all": bool(n and idle_ok.all()), "signs_fixed_all": bool(n and sign_ok.all()),
           "min_frac": VALID_MIN_FRAC}
    out["ok"] = bool(out["follow_ok"] and out["cutoff_before_ok"] and out["idle_zero_all"] and out["signs_fixed_all"])
    return out


def outcome(verdicts, gate_ok=True, valid_ok=True):
    """UNRESOLVED (gate) / UNRESOLVED (validity); else PASS iff every criterion passes; UNRESOLVED if any is UNRESOLVED;
    otherwise FAIL naming each failing criterion (the registered GELU-T / W2-A / Phase 1C rule)."""
    if not gate_ok:
        return "UNRESOLVED (gate)"
    if not valid_ok:
        return "UNRESOLVED (validity)"
    vs = list(verdicts.values())
    if "UNRESOLVED" in vs:
        return "UNRESOLVED"
    if all(v == "PASS" for v in vs):
        return "PASS"
    return "FAIL " + "+".join(k for k, v in verdicts.items() if v == "FAIL")


def criteria(table, s_F, s_star, tau1, tau2, n_rates=N_SCORED):
    """F, C1-C4, E over the rate table (one row per rate: rho, has_fc, t_fc, t_F_fc, delay_fc, eps_fc, r_fc, t_obs,
    s_obs, t_F)."""
    g = {k: _arr([r.get(k) for r in table]) for k in ("rho", "t_fc", "t_F_fc", "delay_fc", "eps_fc", "r_fc", "t_obs",
                                                         "s_obs", "t_F")}
    has_fc = _arr([bool(r.get("has_fc")) for r in table], bool)
    with np.errstate(invalid="ignore"):
        r_obs = g["s_obs"] / s_F - 1
        delay_obs = g["t_obs"] - g["t_F"]
    return {"F": criterion_F(g["s_obs"], s_F, s_star, n_rates),
            "C1": criterion_within(g["t_fc"] - g["t_obs"], g["delay_fc"], tau1, has_fc, n_rates),
            "C2": criterion_within(g["delay_fc"] - delay_obs, g["delay_fc"], tau2, has_fc, n_rates),
            "C3": criterion_C3(g["rho"], r_obs, g["r_fc"], has_fc),
            "C4": criterion_C4(g["t_fc"], g["t_F_fc"], g["t_obs"], has_fc),
            "E": criterion_E(g["eps_fc"], r_obs, has_fc)}


def split_scored(table):
    """(scored rows: the 22 rates with no prior outcome; pre-observed rows: the 5 explored rates, DESCRIPTIVE)."""
    pre = {float(x) for x in PREOBSERVED_LOG2}
    return ([r for r in table if float(r["log2rho"]) not in pre], [r for r in table if float(r["log2rho"]) in pre])


def score_tables(T, s_F, s_star, tau1, tau2, gate_ok=True):
    """Verdicts over the 22 SCORED rates only (F, C1, C2 fractions over 22; C3 over the scored rates with a finite
    ratio; C4 and E bootstraps over the scored rates; the validity rules over the 22).  The 5 pre-observed rates get the
    same statistics as DESCRIPTIVE only (denominator 5); they never enter a verdict."""
    if not gate_ok:
        return {"criteria": {}, "validity": {"ok": False}, "verdicts": {}, "outcome": outcome({}, False, False),
                "DESCRIPTIVE_preobserved": None}
    sc, pre = split_scored(T)

    def val(rows, n):
        return validity([r["follow_in_M"] for r in rows], [r["t_c"] for r in rows], [r["t_obs"] for r in rows],
                        [r["idle_ok"] for r in rows], [r["sign_ok"] for r in rows], n)
    crit = criteria(sc, s_F, s_star, tau1, tau2, N_SCORED)
    v = val(sc, N_SCORED)
    verdicts = {k: c["verdict"] for k, c in crit.items()}
    return {"criteria": crit, "validity": v, "verdicts": verdicts, "outcome": outcome(verdicts, True, v["ok"]),
            "n_scored_rows": len(sc), "n_preobserved_rows": len(pre),
            "DESCRIPTIVE_preobserved": {"criteria": criteria(pre, s_F, s_star, tau1, tau2, len(PREOBSERVED_LOG2)),
                                        "validity": val(pre, len(PREOBSERVED_LOG2))}}


def pilot_errors(t_fc, t_F_fc, t_obs, t_F):
    """Pilot errors at f = 0.95: e₁ = |t_fc − t_obs|/delay_fc (C1), e₂ = |delay_fc − delay_obs|/delay_fc (C2); None if
    the forecast or the crossing is missing."""
    if None in (t_fc, t_F_fc, t_obs, t_F) or t_fc - t_F_fc <= 0:
        return None, None
    d = t_fc - t_F_fc
    return abs(t_fc - t_obs) / d, abs(d - (t_obs - t_F)) / d


def tau_rule(errors):
    """τ = 1.5 × the largest pilot error, rounded UP to a multiple of 0.05; None if any error is missing."""
    if not errors or any(e is None or not np.isfinite(e) for e in errors):
        return None
    x = TAU_FACTOR * max(errors) / TAU_ROUND
    k = math.ceil(round(x, 9))
    return round(k * TAU_ROUND, 10)


def pilot_rate_allowed(l2):
    """A pilot rate must lie OFF the ladder: every run is deterministic given ρ, so a pilot at a ladder rate would
    observe a registered rate before the registration."""
    return float(l2) not in {float(x) for x in LADDER_LOG2}


def pilot_stop(eps_fc, chi_max, n_pilot_M, tau1, tau2):
    """The pilot STOP rule: ε̂_F > 0.01 or max χ_t > 0.1 on [s*, t_c) at the first pilot rate (2⁻¹⁵·⁰⁶²⁵); also STOP if
    fewer than 3 pilot seeds land on M, or a τ cannot be set (a pilot run without a forecast or a crossing)."""
    reasons = []
    if n_pilot_M < len(PILOT_LOG2):
        reasons.append(f"only {n_pilot_M} pilot seeds on M")
    if eps_fc is None or not np.isfinite(eps_fc) or eps_fc > PILOT_EPS_MAX:
        reasons.append(f"eps_F_hat {eps_fc} > {PILOT_EPS_MAX} (or undefined)")
    if chi_max is None or not np.isfinite(chi_max) or chi_max > PILOT_CHI_MAX:
        reasons.append(f"max chi {chi_max} > {PILOT_CHI_MAX} (or undefined)")
    if tau1 is None or tau2 is None:
        reasons.append("tau undefined (a pilot run without a forecast or a crossing)")
    return {"stop": bool(reasons), "reasons": reasons}


def horizon(t_c, t_obs, t_F):
    """DESCRIPTIVE: t_obs − t_c in steps and in delays ((t_obs − t_c)/max(delay_obs, 1))."""
    if t_c is None or t_obs is None:
        return {"steps": None, "delays": None}
    d = max((t_obs - t_F) if t_F is not None else 1, 1)
    return {"steps": int(t_obs - t_c), "delays": (t_obs - t_c) / d}


# ------------------------------------------------------------------------------------------ machine rules and io
def memory_gate(tag):
    """free ≥ 25% and swap free ≥ 500 MB (act_fold.check_memory; waits, re-checking every 60 s); disk free ≥ 20 GB
    (else STOP).  Every check logged to results/phase2a/memory_gate.log."""
    from .act_fold import check_memory
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            ok, f, w = check_memory()
        except Exception:                                                 # noqa: BLE001
            ok, f, w = False, float("nan"), float("nan")
        disk = shutil.disk_usage(ROOT).free
        with open(OUT / "memory_gate.log", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB "
                     f"disk_free={disk / 1024 ** 3:.1f}GB {'OK' if ok and disk >= DISK_MIN_BYTES else 'WAIT'}\n")
        if disk < DISK_MIN_BYTES:
            raise SystemExit(f"STOP: disk free {disk / 1024 ** 3:.1f} GB < 20 GB")
        if ok:
            return f, w
        time.sleep(60)


def rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")


def _setup():
    cur = os.getpriority(os.PRIO_PROCESS, 0)     # raise the niceness to 15, never lower it (the launcher may have
    if cur < 15:                                # set it already: `nice -n 15`, and a sandbox may refuse re-nicing)
        os.nice(15 - cur)
    import torch
    torch.set_num_threads(1)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def _append_write(p, row):
    old = p.read_text() if p.exists() else ""
    p.write_text(old + json.dumps(_jsonable(row)) + "\n")


def _rows(p):
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


# ------------------------------------------------------------------------------------------ seed scan
SCAN_SKIP_FILES = ("phase2a.py", "test_phase2a.py", "phase2a_registration.md", "phase2a_slow_sb_design.md")
SCAN_PATTERNS = {
    "registered": r"(^|[^0-9.])29370(0[0-9]|1[0-1])[0-9]([^0-9]|$)|2_937_(0[0-9]|1[0-1])[0-9]"
                  r"|(^|[^0-9.,])2,937,(0[0-9]|1[0-1])[0-9]([^0-9,]|$)",
    "pilot": r"(^|[^0-9.])29379[0-5][0-9]([^0-9]|$)|2_937_9[0-5][0-9]|(^|[^0-9.,])2,937,9[0-5][0-9]([^0-9,]|$)"}


def scan():
    """The 2A seed ranges are unused: no number in 2,937,000-2,937,119 or 2,937,900-2,937,959 (also written with _ or
    ,) in any text file under src/, tests/, results/, paper/ (2A's own files and the design page excepted), and no
    overlap with the registered or pilot seeds of any test module."""
    import importlib
    from . import phase1c as P1C
    OUT.mkdir(parents=True, exist_ok=True)
    saved = P1C.SCAN_SKIP_FILES
    try:
        P1C.SCAN_SKIP_FILES = SCAN_SKIP_FILES
        hits = P1C._scan_tree(SCAN_PATTERNS)
    finally:
        P1C.SCAN_SKIP_FILES = saved
    hits = {k: [h for h in v if not h.startswith("results/phase2a/")] for k, v in hits.items()}
    mine = set(SEEDS) | set(PILOT_SEEDS)
    ov = {}
    for mod in ("track_a", "track2a", "track2b", "track2c", "gelu_transfer", "width2_asym", "phase1a_pilot", "phase1c",
                "simplicity_bias_v2", "simplicity_bias_v3"):
        m = importlib.import_module(f"src.{mod}")
        for name in dir(m):
            if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
                ov[f"{mod}.{name}"] = sorted(mine & set(P1C._flat_ints(getattr(m, name))))
    out = {"ranges": {"registered": [SEEDS[0], SEEDS[-1], len(SEEDS)],
                      "pilot": [PILOT_SEEDS[0], PILOT_SEEDS[-1], len(PILOT_SEEDS)]},
           "patterns": SCAN_PATTERNS, "pattern_files": hits, "registered_seed_overlap": ov,
           "exploration_seeds_disjoint": bool(not (mine & set(range(2_930_000, 2_930_200)))),
           "unused": bool(not any(hits.values()) and not any(ov.values()))}
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable(out), indent=1))
    if not out["unused"]:
        raise SystemExit("STOP: a 2A seed range is not unused")


# ------------------------------------------------------------------------------------------ freeze
def freeze():
    """results/phase2a/frozen.json: q, s* (v2 frozen.json), s_F (sb_fold), the M table (λ_min on M's Δs = 0.001 grid from
    s₀ to s_F, training coordinates, P = I on the fixed-s tangent), |m′c′| (last 0.05 below s_F; 0.025 as a check),
    Λ_F, the exact release θ_M(s₀) (training coordinates, idle unit 0), the branch points at s₀ used by the hold
    classifier, the 27-rate ladder and the pilot rates, budgets, the seed → rate plan rule.  No seed is drawn."""
    _setup()
    memory_gate("freeze")
    t0 = time.time()
    v2f = json.loads(V2_FROZEN.read_text())
    s_F = float(SBF.fold_of("M"))
    z = np.load(SBF.OUT / "grid_M.npz")
    R = SBF.Reduced(int(z["n_active"]), X, Y)
    sel = (z["s"] >= S0 - 0.0015) & (z["s"] < s_F)
    ss = z["s"][sel]
    lam = np.array([tangent_lmin(theta_from_u(u, R.nA, s)) for u, s in zip(z["U"][sel], ss)])
    mc, n_fit = fold_constant_fit(ss, lam, s_F, 0.05)
    mc2, n_fit2 = fold_constant_fit(ss, lam, s_F, 0.025)
    LamF = math.sqrt(mc * s_F)
    rel = branch_theta("M", S0)
    bp = {k: branch_P(k, S0) for k in ("L0", "M", "S", "S2")}
    fr = {"label": "Phase 2A frozen inputs (before any registered or pilot seed is drawn)",
          "q": float(v2f["q"]), "s_star": float(v2f["s_q"]), "lambda": float(v2f["lambda"]),
          "v2_frozen_sha256": _sha(V2_FROZEN), "s_F": s_F, "s_F_over_s_star": s_F / float(v2f["s_q"]),
          "s0": S0, "eta": ETA, "omega0": OMEGA0,
          "abs_mc": mc, "abs_mc_n": n_fit, "abs_mc_window_0.025": mc2, "abs_mc_window_0.025_n": n_fit2,
          "Lambda_F": LamF,
          "M_table": {"s": [float(x) for x in ss], "lam_min": [float(x) for x in lam],
                      "note": "λ_min of the training-coordinate Hessian on the fixed-s tangent of the active units, P = I"},
          "release_theta": [float(x) for x in rel], "release_sha256": state_sha(rel),
          "release_active_units": [k for k in range(4) if rel[12 + k] != 0.0],
          "release_rho2": rho2(rel), "release_loss": loss_grad(rel)[0],
          "release_grad_max": float(np.abs(loss_grad(rel)[1][[i for k in range(4) if rel[12 + k] != 0.0
                                                               for i in (2 * k, 2 * k + 1, 8 + k)] + [16]]).max()),
          "branch_points_s0": {k: (None if v is None else [float(x) for x in v]) for k, v in bp.items()},
          "ladder_log2": list(LADDER_LOG2), "pilot_log2": list(PILOT_LOG2),
          "scored_log2": list(SCORED_LOG2), "preobserved_log2_descriptive": list(PREOBSERVED_LOG2),
          "budgets": {f"{l2:g}": budget(rho_of(l2)) for l2 in LADDER_LOG2 + PILOT_LOG2},
          "tight_f": {f"{l2:g}": f_tight(rho_of(l2)) for l2 in LADDER_LOG2},
          "plan_rule": "holds of SEEDS in seed order (one fixed-s BFGS batch); the i-th seed labelled M takes "
                       "ladder_log2[i] (fastest first); M seeds beyond 27 and off-M seeds counted, not trained",
          "seeds": [SEEDS[0], SEEDS[-1]], "pilot_seeds": [PILOT_SEEDS[0], PILOT_SEEDS[-1]],
          "secs": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen.json").write_text(json.dumps(_jsonable(fr), indent=1))
    print(json.dumps({k: v for k, v in _jsonable(fr).items() if k not in ("M_table", "branch_points_s0")}, indent=1))
    rss_guard()


def _frozen():
    return json.loads((OUT / "frozen.json").read_text())


# ------------------------------------------------------------------------------------------ pilot (before registration)
def pilot():
    """Pilot seeds only.  Holds of the 60 pilot seeds (one batch); the first three M landings take the pilot rates in
    order; each: run_rate (to t_c, forecast) then observe_rate (the pilot's crossing) — resumable per rate.  The STOP
    rule; τ₁, τ₂ = 1.5 × the largest pilot error at f = 0.95, rounded up to 0.05.  -> pilot.json"""
    _setup()
    fr = _frozen()
    out_p = OUT / "pilot.json"
    P = json.loads(out_p.read_text()) if out_p.exists() else {}
    if "holds" not in P:
        memory_gate("pilot holds")
        t0 = time.time()
        Ph, L, gn = hold(PILOT_SEEDS)
        labs = []
        for i, s in enumerate(PILOT_SEEDS):
            lab, d = label_at(Ph[i], fr["branch_points_s0"])
            labs.append({"seed": s, "label": lab, "dist": d, "loss": float(L[i]), "gnorm": float(gn[i])})
        P["holds"] = labs
        P["holds_secs"] = round(time.time() - t0, 1)
        out_p.write_text(json.dumps(_jsonable(P), indent=1))
    labels = [h["label"] for h in P["holds"]]
    pl, _ = plan(labels, PILOT_SEEDS, PILOT_LOG2)
    P["gate_like_counts"] = gate(labels)
    P["assignment"] = [{"seed": s, "log2rho": l2} for s, l2 in pl]
    P.setdefault("runs", {})
    for key in [k for k in P["runs"] if float(k) not in {float(x) for x in PILOT_LOG2}]:
        P.setdefault("superseded_runs", {})[key] = {**P["runs"].pop(key), "superseded": "not a pilot rate after the "
                                                    "author's decision of 2026-10-04 (S1 (b): 2^-16.5 -> 2^-16.5625)"}
    for s, l2 in pl:
        key = f"{l2:g}"
        if key in P["runs"] and "observed" in P["runs"][key]:
            continue
        if not pilot_rate_allowed(l2):
            P["runs"][key] = {"seed": s, "log2rho": l2, "skipped": "pilot rate lies ON the ladder (each run is "
                              "deterministic given rho, so piloting it would observe a registered rate): not run; "
                              "the author's decision is needed"}
            out_p.write_text(json.dumps(_jsonable(P), indent=1))
            continue
        memory_gate(f"pilot run {l2:g}")
        t0 = time.time()
        rr = run_rate(l2, fr)
        rr["seed"] = s
        rr["secs_run"] = round(time.time() - t0, 1)
        rss_guard()
        memory_gate(f"pilot observe {l2:g}")
        t0 = time.time()
        ob, S = observe_rate(l2, fr, rr)
        ob["secs_observe"] = round(time.time() - t0, 1)
        fc = rr["forecast"]
        e1, e2 = pilot_errors(fc.get("t_fc"), fc.get("t_F_fc"), ob["t_obs"], ob["t_F"])
        ob.update(err_C1=e1, err_C2=e2, horizon=horizon(rr["t_c"], ob["t_obs"], ob["t_F"]),
                  r_obs=(ob["s_obs"] / fr["s_F"] - 1) if ob["s_obs"] is not None else None,
                  delay_obs=(ob["t_obs"] - ob["t_F"]) if ob["t_obs"] is not None and ob["t_F"] is not None else None)
        rr.pop("state_tc", None)
        P["runs"][key] = {**rr, "observed": ob}
        del S
        out_p.write_text(json.dumps(_jsonable(P), indent=1))
        print(json.dumps(_jsonable({"log2rho": l2, "t_c": rr["t_c"], "fc": {k: fc.get(k) for k in
                                    ("t_F_fc", "t_fc", "delay_fc", "eps_fc", "r_fc")}, "obs": ob})), flush=True)
        rss_guard()
    runs = [P["runs"].get(f"{l2:g}") for l2 in PILOT_LOG2]
    e1 = [r["observed"]["err_C1"] if r and "observed" in r else None for r in runs]
    e2 = [r["observed"]["err_C2"] if r and "observed" in r else None for r in runs]
    tau1, tau2 = tau_rule(e1), tau_rule(e2)
    first = runs[0]
    st = pilot_stop(first["forecast"].get("eps_fc") if first else None, first.get("chi_max_sstar_tc") if first else None,
                    len(pl), tau1, tau2)
    P.update(errors_C1=e1, errors_C2=e2, tau1=tau1, tau2=tau2, stop_rule=st,
             rule="tau = 1.5 x the largest pilot error at f = 0.95, rounded up to 0.05")
    out_p.write_text(json.dumps(_jsonable(P), indent=1))
    print(json.dumps(_jsonable({"tau1": tau1, "tau2": tau2, "errors_C1": e1, "errors_C2": e2, "stop": st}), indent=1))


def _pilot():
    P = json.loads((OUT / "pilot.json").read_text())
    assert not P["stop_rule"]["stop"], "the pilot STOP rule fired"
    return P


# ------------------------------------------------------------------------------------------ registration manifest
FROZEN_DATA = ("results/designs/phase2a_slow_sb_design.md", "results/phase2a_registration.md",
               "results/phase2a/seed_scan.json", "results/phase2a/frozen.json", "results/phase2a/pilot.json",
               "tests/test_phase2a.py", "tests/test_causal_forecast.py", "results/simplicity_bias_v2/frozen.json",
               "results/sb_fold/grid_M.npz", "results/sb_fold/branches.json",
               "results/sb_fold/branch_M_fwd.npz", "results/sb_fold/branch_M_bwd.npz",
               "results/sb_fold/branch_L0_fwd.npz", "results/sb_fold/branch_L0_bwd.npz",
               "results/sb_fold/branch_S_fwd.npz", "results/sb_fold/branch_S_bwd.npz",
               "results/sb_fold/branch_S2_fwd.npz", "results/sb_fold/branch_S2_bwd.npz")


def code_closure(start=("phase2a",)):
    from . import phase1c as P1C
    return P1C.code_closure(start)


def manifest_files():
    return sorted(set(code_closure()) | set(FROZEN_DATA))


def manifest():
    lines = []
    for rel in manifest_files():
        p = ROOT / rel
        assert p.exists(), f"missing {rel}"
        lines.append(f"{_sha(p)}  {rel}")
    (OUT / "registration.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _assert_committed(p):
    rel = str(Path(p).resolve().relative_to(ROOT))
    assert subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip(), f"{rel} not committed"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0, f"{rel} modified"


def assert_registration():
    m = OUT / "registration.sha256"
    for ln in m.read_text().splitlines():
        h, rel = ln.split()
        assert _sha(ROOT / rel) == h, f"registration hash mismatch: {rel}"
    _assert_committed(m)
    _assert_committed(OUT / "registration_stamp.txt")


# ------------------------------------------------------------------------------------------ run / finalize / observe / score
FORBIDDEN = {"t_obs", "s_obs", "rho2_at_obs", "t_F", "follow_in_M", "crossed", "r_obs", "delay_obs"}


def run():
    """After the registration commit and its timestamp: holds of the 120 seeds, the gate, the plan, then each planned
    rate to t_c with its forecast (resumable per rate)."""
    assert_registration()
    _setup()
    fr = _frozen()
    _pilot()
    hf = OUT / "holds.jsonl"
    if not hf.exists():
        memory_gate("holds")
        Ph, L, gn = hold(SEEDS)
        for i, s in enumerate(SEEDS):
            lab, d = label_at(Ph[i], fr["branch_points_s0"])
            _append_write(hf, {"seed": s, "label": lab, "dist": d, "loss": float(L[i]), "gnorm": float(gn[i])})
    holds = _rows(hf)
    assert [h["seed"] for h in holds] == list(SEEDS)
    labels = [h["label"] for h in holds]
    g = gate(labels)
    print(json.dumps(g), flush=True)
    if not g["pass"]:
        print("gate failed: no rate is trained (UNRESOLVED (gate))", flush=True)
        return
    pl, surplus = plan(labels)
    rf = OUT / "runs.jsonl"
    done = {r["log2rho"] for r in _rows(rf)}
    for s, l2 in pl:
        if float(l2) in done:
            continue
        memory_gate(f"run {l2:g}")
        t0 = time.time()
        rec = run_rate(l2, fr)
        rec.update(seed=s, secs=round(time.time() - t0, 1))
        _append_write(rf, rec)
        print(json.dumps({"seed": s, "log2rho": l2, "t_c": rec["t_c"], "status": rec["forecast"]["status"],
                          "secs": rec["secs"]}), flush=True)
        rss_guard()


def _keys(o):
    if isinstance(o, dict):
        return set(o) | {k for v in o.values() for k in _keys(v)}
    if isinstance(o, list):
        return {k for v in o for k in _keys(v)}
    return set()


def finalize():
    """forecasts.sha256 over holds.jsonl and runs.jsonl: every planned rate exactly once, no observed key.  COMMIT both
    files and forecasts.sha256 before observe."""
    holds = _rows(OUT / "holds.jsonl")
    pl, _ = plan([h["label"] for h in holds])
    runs = _rows(OUT / "runs.jsonl")
    assert sorted(r["log2rho"] for r in runs) == sorted(float(l2) for _, l2 in pl), "every planned rate exactly once"
    assert not (FORBIDDEN & _keys(runs)), "an observed quantity in a runs row"
    lines = [f"{_sha(OUT / f)}  {f}" for f in ("holds.jsonl", "runs.jsonl")]
    (OUT / "forecasts.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _assert_forecasts():
    hs = dict(reversed(ln.split()) for ln in (OUT / "forecasts.sha256").read_text().splitlines())
    for f in ("holds.jsonl", "runs.jsonl"):
        assert _sha(OUT / f) == hs[f], f"forecast hash mismatch ({f})"
        _assert_committed(OUT / f)
    _assert_committed(OUT / "forecasts.sha256")


def observe():
    assert_registration()
    _assert_forecasts()
    _setup()
    fr = _frozen()
    of = OUT / "observed.jsonl"
    done = {r["log2rho"] for r in _rows(of)}
    for rr in _rows(OUT / "runs.jsonl"):
        l2 = rr["log2rho"]
        if l2 in done:
            continue
        memory_gate(f"observe {l2:g}")
        t0 = time.time()
        ob, S = observe_rate(l2, fr, rr)
        ob["tight"] = tight_forecast(l2, fr, S)
        ob["secs"] = round(time.time() - t0, 1)
        del S
        _append_write(of, ob)
        print(json.dumps({"log2rho": l2, "t_obs": ob["t_obs"], "s_obs": ob["s_obs"], "secs": ob["secs"]}), flush=True)
        rss_guard()


def rate_table(fr, runs, obs):
    """One row per rate: forecast (registered f) and observation."""
    ob = {o["log2rho"]: o for o in obs}
    rows = []
    for r in sorted(runs, key=lambda r: -r["log2rho"]):
        fc = r["forecast"]; o = ob.get(r["log2rho"], {})
        rows.append({"log2rho": r["log2rho"], "rho": r["rho"], "seed": r.get("seed"), "t_c": r["t_c"],
                     "has_fc": CFF.has_forecast(fc), "status": fc.get("status"), "t_fc": fc.get("t_fc"),
                     "t_F_fc": fc.get("t_F_fc"), "delay_fc": fc.get("delay_fc"), "eps_fc": fc.get("eps_fc"),
                     "r_fc": fc.get("r_fc"), "t_obs": o.get("t_obs"), "s_obs": o.get("s_obs"), "t_F": o.get("t_F"),
                     "follow_in_M": o.get("follow_in_M", False), "idle_ok": o.get("idle_zero_to_crossing", False),
                     "sign_ok": o.get("signs_fixed_to_crossing", False),
                     "horizon": horizon(r["t_c"], o.get("t_obs"), o.get("t_F")), "tight": o.get("tight")})
    return rows


def score():
    assert_registration()
    _assert_forecasts()
    fr = _frozen(); P = _pilot()
    holds = _rows(OUT / "holds.jsonl")
    g = gate([h["label"] for h in holds])
    runs = _rows(OUT / "runs.jsonl"); obs = _rows(OUT / "observed.jsonl")
    T = rate_table(fr, runs, obs) if g["pass"] else []
    ST = score_tables(T, fr["s_F"], fr["s_star"], P["tau1"], P["tau2"], g["pass"])
    crit, val, verdicts = ST["criteria"], ST["validity"], ST["verdicts"]
    tight_T = [{**r, "t_c": (r["tight"] or {}).get("t_c"), "has_fc": CFF.has_forecast(r["tight"]),
                **{k: (r["tight"] or {}).get(k) for k in ("t_fc", "t_F_fc", "delay_fc", "eps_fc", "r_fc")}} for r in T]
    out = {"gate": g, "validity": val, "criteria": crit, "verdicts": verdicts, "outcome": ST["outcome"],
           "scored_log2rho": list(SCORED_LOG2), "preobserved_log2rho": list(PREOBSERVED_LOG2),
           "tau1": P["tau1"], "tau2": P["tau2"], "rates": T,
           "DESCRIPTIVE_preobserved_rates": ST["DESCRIPTIVE_preobserved"],
           "DESCRIPTIVE_tightening_cutoff": criteria(split_scored(tight_T)[0], fr["s_F"], fr["s_star"], P["tau1"],
                                                     P["tau2"]) if g["pass"] else None}
    (OUT / "scores.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("gate", "verdicts", "outcome")}), indent=1))


def main(argv):
    cmd = argv[1]
    fns = {"scan": scan, "freeze": freeze, "pilot": pilot, "manifest": manifest, "run": run, "finalize": finalize,
           "observe": observe, "score": score}
    if cmd not in fns:
        raise SystemExit(f"unknown command {cmd}")
    fns[cmd]()


if __name__ == "__main__":
    main(sys.argv)
