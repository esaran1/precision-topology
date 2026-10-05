"""Phase 2A-PS: the fold prediction with a SAMPLE PER SEED (design: results/designs/phase2a_perseed_design.md, approved by
the author 2026-10-05, commit cec2726; registration: results/phase2a_ps_registration.md).

Each seed draws its own 800-point sample (i.i.d. levels on the benchmark's product grid), and has its own M branch,
fold s_F, switch s*, fold constant Λ_F, class and causal forecast: the SEEDS are the replicates.  Otherwise as Phase 2A
(src/phase2a.py, registration 569b836): the scale-only rule (η = 1), the crossing (first upward passage of q by ρ₂,
every step), three active units (the idle unit exactly 0), the frozen causal fold forecaster
(src/causal_forecast_fold.py, unchanged, leading order Ω₀ε̂_F^{2/3}).

PER-SEED FREEZE (before the registration; no training):
  sample     i.i.d. levels on the product grid, numpy default_rng([seed, 7]) (p2a_explore_g.sample), 800 points
  identity   M by a data homotopy at s ≈ 4.49 from the fixed dataset's M point: X(λ) = (1 − λ)X_fixed + λX_seed,
             Newton at λ = 1/n … 1, n = 50 (200 if the 50-step homotopy fails); accepted iff Newton converges and the
             reduced λ_min > 0 at every λ, the branch has an upper fold (its stable part reaching it), and ρ₂ < q at
             every stable continuation point
  constants  s_F by pseudo-arclength continuation (h = 0.05), validated (h/2 agrees; λ_min → 0 at the fold; no
             perturbed minimum on M above it); |m′c′| from the λ_min² fit over the last 0.05 and 0.025 below s_F
             (agreeing within 5%), Λ_F = √(|m′c′|·s_F); s* = equal loss of the seed's own M and S
  class      30 perturbed minima at 1.01·s_F: n = #(ρ₂ ≥ q): clean n ≥ 27, none n = 0, mixed 1 ≤ n ≤ 26
  release    s₀ = lo + FRAC0·(s_F − lo) on the seed's own stable M (FRAC0 maps 2A's 1.7957); eligible iff
             s₀ ≤ 0.6·s_F; released at the exact M(s₀) (Newton), no BFGS hold
  gate       ≥ 24 clean eligible seeds of the 400.  Plan: the first 40 clean (2⁻¹⁶ scored, 2⁻¹⁴ exponent), 30 none and
             20 mixed (2⁻¹⁴) scoreable seeds in seed order; the rest are counted.

ORDER (no step reads anything a later step produces):
  scan / pilot / freeze / manifest     before the registration commit (pilot: pilot seeds only)
  run       after the registration commit and its timestamp: per planned (seed, rate) training from the seed's exact
            M(s₀) to t_c (first s ≥ 0.95·s_F(seed); NO ρ₂), the state at t_c hashed, the causal forecast with the
            seed's own (s_F, Λ_F) and the competitor with the fixed-dataset fold, at the same t_c -> runs.jsonl
  finalize  forecasts.sha256 over runs.jsonl (no observed key); COMMIT before observe
  observe   retrain with ρ₂ EVERY step, the t_c state asserted equal to the hashed one bit for bit, observation to the
            crossing (and s_F) or 1.25·s_F; follow check at 0.8·s_F on the seed's own M -> observed.jsonl
  score     scores.json (F, H, E_seed, C3, C4, P; C1, C2 secondary; validity; gate)

Machine rules: one process, nice 15, one thread; a memory gate (free ≥ 25%, swap free ≥ 500 MB; waits) and a disk check
(free ≥ 20 GB; else STOP) before every job and between seeds, logged to results/phase2a_ps/memory_gate.log; stop above
3 GB RSS; every training run and every forecast has a hard step budget (⌈64/ρ⌉).  No global torch / numpy RNG state.

    python -m src.phase2a_ps scan | gate TAG | pilot | freeze | summary | manifest | stamp
    python -m src.phase2a_ps run | finalize | observe | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import causal_forecast as CF
from . import causal_forecast_fold as CFF
from . import phase2a as P2A
from . import sb_fold as SBF
from . import simplicity_bias as sb
from . import simplicity_bias_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase2a_ps"
REGISTRATION_MD = RESULTS / "phase2a_ps_registration.md"
DESIGN_MD = RESULTS / "designs" / "phase2a_perseed_design.md"
V2_FROZEN = RESULTS / "simplicity_bias_v2" / "frozen.json"
P2A_FROZEN = RESULTS / "phase2a" / "frozen.json"
P2A_POSTHOC = RESULTS / "phase2a_posthoc.json"

# ------------------------------------------------------------------------------------------ registered constants
SEEDS = tuple(range(2_975_000, 2_975_400))       # 400 registered seeds (author-approved 2026-10-05)
PILOT_SEEDS = tuple(range(2_976_000, 2_976_060))  # 60 pilot seeds
EXPLORATION_SEEDS = range(2_930_000, 2_930_200)  # never registered
SAMPLE_STREAM = 7                                # default_rng([seed, 7])
N_POINTS = 800
LAM = 1e-4
ETA = 1.0
Q = 0.3914103370353161                           # v2 frozen q (asserted against results/simplicity_bias_v2/frozen.json)
S_F_FIXED = 4.767689442106793                    # the fixed dataset's fold (2A frozen.json)
LAMF_FIXED = 0.001509953872657247                # the fixed dataset's Λ_F (2A frozen.json)
S0_FIXED, LO_FIXED = 1.7957, 1.154               # 2A's s₀ and the fixed dataset's M lower end
FRAC0 = (S0_FIXED - LO_FIXED) / (S_F_FIXED - LO_FIXED)   # 0.177575… (the page's "0.1776")
ELIG_MAX = 0.6                                   # eligible iff s₀ ≤ 0.6·s_F
S_HOMOTOPY = 4.5                                 # the stored fixed-dataset branch point nearest 4.5 (s = 4.49485…)
HOMOTOPY_STEPS = (50, 200)                       # λ steps; 200 if the 50-step homotopy fails
H_CONT, H_CHECK = 0.05, 0.025                    # continuation step and the half step for the fold check
CONT_LENGTH = 40.0                               # n_max = 40/h continuation points
FOLD_AGREE_REL = 1e-6                            # |s_F(h) − s_F(h/2)| ≤ 1e-6·s_F
FOLD_EIG_TOL = 1e-6                              # |reduced λ_min at the refined fold| ≤ 1e-6 (λ_min → 0)
VALID_SCALES = (0.99, 0.999, 1.001, 1.01)        # perturbed minima (30 each); the class reads 1.01
ABOVE_SCALES = (1.001, 1.01)                     # no perturbed minimum on M above the fold
PERTURB_SEED, PERTURB_N = 0, 30                  # default_rng(0); δ = N(0, 1)·(0.05|P| + 0.02)
ON_TOL = 1e-3                                    # "on M": function-space distance ≤ 1e-3 (sb_fold.fdist)
CLASS_SCALE = 1.01
CLEAN_MIN = 27                                   # clean ≥ 27 of 30 with ρ₂ ≥ q; none 0; mixed 1–26
MC_DS, MC_WINDOWS, MC_AGREE = 0.001, (0.05, 0.025), 0.05   # D1 (c), author 2026-10-05: Λ_F from the 0.05 window;
                                                             # the ≤ 5% agreement with 0.025 is DESCRIPTIVE only
LOG2_SCORED, LOG2_EXPONENT, LOG2_OTHER = -16.0, -14.0, -14.0
CAPS = {"clean": 40, "none": 30, "mixed": 20}
GATE_MIN_CLEAN = 24
F_CUT = 0.95
BUDGET_FACTOR = 64.0                             # ⌈64/ρ⌉ steps per run; forecast budget ⌈64/ρ⌉ − t_c
OBS_END = 1.25
FOLLOW_FRAC = 0.8
F_MIN_FRAC, F_HI, F_FALSIFIER = 0.90, 1.25, 1.25
H_MIN_FRAC, H_MIN_N = 0.80, 10
E_BAND = (0.55, 0.80)
C3_MIN_FRAC = 0.90
TAU1, TAU2, WITHIN_MIN_FRAC = 0.15, 0.25, 0.80   # C1, C2 (secondary; registered expected FAIL)
VALID_MIN_FRAC = 0.90
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 2_975_000, (2.5, 97.5)
MIN_SEEDS_STAT = 3                               # E_seed ≥ 3 seeds; C4 and P ≥ 2
PRIMARY = ("F", "H", "E_seed", "C3", "C4", "P")
SECONDARY = ("C1", "C2")
OMEGA0 = CFF.OMEGA0
RSS_LIMIT = 3 * 1024 ** 3
DISK_MIN_BYTES = 20 * 1024 ** 3

X_FIXED, Y_FIXED = v2.data()


def rho_of(l2):
    return 2.0 ** float(l2)


def budget(rho):
    return int(math.ceil(BUDGET_FACTOR / rho))


# ------------------------------------------------------------------------------------------ the per-seed sample
def sample(seed):
    """The benchmark generator with i.i.d. levels in place of quantile levels (p2a_explore_g.sample): per class a 20 × 20
    product grid; x₁ class-1 levels: 4 from U[−0.1, 0.1] and 16 from U[0.1, 1] (sorted), class 0 the mirror; x₂: 20
    middle-slab levels U[−h, h] (class 0), 10 outer levels U[h + 0.2, 1] mirrored to ± (class 1); default_rng([seed, 7])."""
    rng = np.random.default_rng([int(seed), SAMPLE_STREAM])
    p, n1, n2 = sb.PILOT_P, sb.N1, sb.N2
    k_noise = int(round(p * n1))
    l1 = np.sort(np.r_[rng.uniform(-0.1, 0.1, k_noise), rng.uniform(0.1, 1.0, n1 - k_noise)])
    w = (2 - 2 * sb.GAP) / 3
    h = w / 2
    mid = np.sort(rng.uniform(-h, h, n2))
    upper = np.sort(rng.uniform(h + sb.GAP, 1.0, n2 // 2))
    outer = np.concatenate([-upper[::-1], upper])
    X1 = np.array([(a, b) for a in l1 for b in outer])
    X0 = np.array([(-a, b) for a in l1 for b in mid])
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    return X, y


def sample_sha(X, Y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X, dtype="<f8")).tobytes())
    h.update(np.ascontiguousarray(np.asarray(Y, dtype="<f8")).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ model on a seed's sample
def loss_grad(th, X, Y, lam=LAM):
    """P2A.loss_grad on the seed's sample (same expression, same operation order)."""
    W, c, v, b = th[:8].reshape(4, 2), th[8:12], th[12:16], th[16]
    H = np.tanh(X @ W.T + c)
    z = H @ v + b
    L = float(np.mean(np.logaddexp(0.0, z) - Y * z) + 0.5 * lam * ((W ** 2).sum() + (c ** 2).sum()))
    r = (0.5 * (1 + np.tanh(0.5 * z)) - Y) / len(Y)
    D = (1 - H ** 2) * (r[:, None] * v[None, :])
    g = np.empty(17)
    g[:8] = (D.T @ X + lam * W).ravel()
    g[8:12] = D.sum(0) + lam * c
    g[12:16] = H.T @ r
    g[16] = r.sum()
    return L, g


def make_step(rho, active, X, Y, eta=ETA):
    """P2A.make_step on the seed's sample: (W, c, b) ← · − η∇; v ← v − η[(I − ââᵀ) + ρââᵀ]∇_vL, â = sign(v)/√n on the
    active units (the idle unit's â component is 0)."""
    act = np.asarray(active, int)
    n = len(act)

    def step(th):
        _, g = loss_grad(th, X, Y)
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


scale = P2A.scale
state_sha = P2A.state_sha


class Rho2Grid:
    """ρ₂ of φ(x) = Σ vₖ tanh(wₖ·x + cₖ) on a product-grid sample: P2A.rho2 with the grid built from the seed's X (every
    replacement value weighted by its count).  Equal to v2.rho2_batch to rounding, and to P2A.rho2 bit for bit on the
    fixed data (tested)."""

    def __init__(self, X):
        X = np.asarray(X, float)
        N = len(X)
        U = [np.unique(X[:, j]) for j in range(2)]
        A = np.searchsorted(U[0], X[:, 0])
        B = np.searchsorted(U[1], X[:, 1])
        self.N1, self.N2 = len(U[0]), len(U[1])
        cnt = [np.bincount(A, minlength=self.N1).astype(float), np.bincount(B, minlength=self.N2).astype(float)]
        self.I0 = A * self.N2 + B
        self.I1 = (np.arange(self.N1)[:, None] * self.N2 + B[None, :]).ravel()
        self.I2 = (A[None, :] * self.N2 + np.arange(self.N2)[:, None]).ravel()
        self.W1 = np.repeat(cnt[0], N)
        self.W2 = np.repeat(cnt[1], N)
        self.UX = U[0][:, None, None]
        self.UY = U[1][None, :, None]

    def __call__(self, th):
        W = th[:8].reshape(4, 2)
        G = (np.tanh(self.UX * W[:, 0] + (self.UY * W[:, 1] + th[8:12])) @ th[12:16]).ravel()
        base = G[self.I0]
        V1 = self.W1 @ np.abs(G[self.I1] - np.tile(base, self.N1))
        V2 = self.W2 @ np.abs(G[self.I2] - np.tile(base, self.N2))
        return float(V2 / (V1 + V2))


def rho2_static(P16, X):
    """ρ₂ of a 16-vector (landscape parametrisation) by sb.feature_usage (the class rule and the identity check)."""
    return float(sb.feature_usage(np.asarray(P16, float), X)["rho2"])


def hessian(th, X, Y, lam=LAM):
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Yt = torch.as_tensor(Y, dtype=torch.float64)

    def Lf(q):
        W = q[:8].reshape(4, 2); c = q[8:12]; v = q[12:16]; b = q[16]
        z = torch.tanh(Xt @ W.T + c) @ v + b
        return torch.nn.functional.binary_cross_entropy_with_logits(z, Yt) + 0.5 * lam * ((W ** 2).sum() + (c ** 2).sum())
    return torch.autograd.functional.hessian(Lf, torch.tensor(th, dtype=torch.float64)).numpy()


def tangent_lmin(th, X, Y):
    """λ_min of the training-coordinate Hessian on the fixed-s tangent of the active units (P = I), as P2A.tangent_lmin
    on the seed's sample."""
    H = hessian(th, X, Y)
    act = [k for k in range(4) if abs(th[12 + k]) > 1e-12]
    idx = [2 * k for k in act] + [2 * k + 1 for k in act] + [8 + k for k in act] + [12 + k for k in act] + [16]
    nv = len(act); m = len(idx)
    a = np.zeros(m); a[3 * nv:4 * nv] = np.sign(th[[12 + k for k in act]]); a /= np.linalg.norm(a)
    U, _, _ = np.linalg.svd(np.eye(m) - np.outer(a, a))
    Qb = U[:, :m - 1]
    return float(np.linalg.eigvalsh(Qb.T @ H[np.ix_(idx, idx)] @ Qb)[0])


# ------------------------------------------------------------------------------------------ the seed's branches
def homotopy(name, X, Y, n_steps):
    """Data homotopy at fixed s (the fixed dataset's stored branch point nearest S_HOMOTOPY): X(λ) = (1 − λ)X_fixed +
    λX_seed, Newton at λ = 1/n … 1 from the previous solution.  Returns (u at λ = 1 or None, nA, s, info)."""
    xs, ev, Rp = SBF.load_branch(name)
    k = int(np.argmin(np.abs(xs[:, -1] - S_HOMOTOPY)))
    s = float(xs[k, -1])
    u0 = xs[k, :-1].copy()
    min_eig = math.inf
    for lt in np.linspace(1.0 / n_steps, 1.0, n_steps):
        Rt = SBF.Reduced(Rp.nA, (1 - lt) * X_FIXED + lt * X, Y)
        u1, ok, res = SBF.newton_fixed_s(Rt.F, Rt.JF, u0, s)
        if not ok:
            return None, Rp.nA, s, {"n_steps": n_steps, "newton_ok": False, "lambda_failed": float(lt),
                                    "homotopy_min_eig": None, "accepted": False}
        J = Rt.JF(u1, s)[:, :-1]
        min_eig = min(min_eig, float(np.linalg.eigvalsh(0.5 * (J + J.T))[0]))
        u0 = u1
    return u0, Rp.nA, s, {"n_steps": n_steps, "newton_ok": True, "lambda_failed": None, "homotopy_min_eig": min_eig,
                           "accepted": bool(min_eig > 0)}


def homotopy_accepted(name, X, Y):
    """The 50-step homotopy; the 200-step one if the 50-step one fails (Newton or λ_min ≤ 0 at some λ)."""
    attempts = []
    for n in HOMOTOPY_STEPS:
        u, nA, s, info = homotopy(name, X, Y, n)
        attempts.append(info)
        if info["accepted"]:
            return u, nA, s, {**info, "attempts": attempts}
    return None, nA, s, {**attempts[-1], "attempts": attempts}


def continue_both(R, u, s, h):
    x0 = np.r_[u, s]
    n_max = int(CONT_LENGTH / h)
    pf, ff = SBF.continuation(R.F, R.JF, x0, +1, h, n_max=n_max)
    pb, fb = SBF.continuation(R.F, R.JF, x0, -1, h, n_max=n_max)
    return pf, ff, pb, fb


class SeedBranch:
    """A branch of the seed's own landscape: the CONTIGUOUS stable run of continuation points containing the homotopy
    point, and Newton at any s from the nearest of them."""

    def __init__(self, R, pb, pf):
        pts = pb[::-1] + pf[1:]
        i0 = len(pb) - 1                                           # the homotopy point
        ev = np.array([p["eig_min"] for p in pts])
        st = ev > 0
        self.R = R
        self.start_stable = bool(st[i0])
        if not st[i0]:
            self.xs = np.zeros((0, len(pts[0]["x"])))
            self.lo = self.hi = float("nan")
            self.contiguous_equals_all_stable = False
            return
        a = i0
        while a - 1 >= 0 and st[a - 1]:
            a -= 1
        b = i0
        while b + 1 < len(pts) and st[b + 1]:
            b += 1
        self.seg = (a, b)
        self.xs = np.array([pts[i]["x"] for i in range(a, b + 1)])
        self.lo, self.hi = float(self.xs[:, -1].min()), float(self.xs[:, -1].max())
        allst = np.array([p["x"] for p, ok in zip(pts, st) if ok])
        self.contiguous_equals_all_stable = bool(len(allst) == len(self.xs))
        self.n_pf_stable_prefix = b - i0 + 1                      # forward points (incl. the start) in the segment

    def u(self, s):
        k = int(np.argmin(np.abs(self.xs[:, -1] - s)))
        u, ok, _ = SBF.newton_fixed_s(self.R.F, self.R.JF, self.xs[k, :-1], s)
        return u if ok else None

    def theta(self, s):
        u = self.u(s)
        return None if u is None else P2A.theta_from_u(u, self.R.nA, s)

    def P(self, s):
        u = self.u(s)
        return None if u is None else self.R.to_full(u)[0]

    def loss(self, s):
        u = self.u(s)
        return None if u is None else self.R.loss(u, s)


def equal_loss_s(M, S):
    """s* = the scale where the seed's own M and S have equal loss: bisection on [max lo + 1e-3, min hi − 1e-3] (to
    1e-8) if the loss difference changes sign there; else None."""
    lo, hi = max(M.lo, S.lo) + 1e-3, min(M.hi, S.hi) - 1e-3
    if not lo < hi:
        return None, "no common stable range"

    def diff(s):
        a, b = M.loss(s), S.loss(s)
        return None if a is None or b is None else a - b
    dl, dh = diff(lo), diff(hi)
    if dl is None or dh is None:
        return None, "Newton failed at an end"
    if not dl * dh < 0:
        return None, "no sign change of the loss difference"
    for _ in range(200):
        if hi - lo <= 1e-8:
            break
        mid = 0.5 * (lo + hi)
        dm = diff(mid)
        if dm is None:
            return None, "Newton failed in the bisection"
        if dm * dl > 0:
            lo, dl = mid, dm
        else:
            hi = mid
    return 0.5 * (lo + hi), "ok"


def perturbed_minima(M, s_F, X, Y):
    """30 perturbed local minima at each of 0.99, 0.999, 1.001, 1.01·s_F (one default_rng(0) per seed, in this order;
    p2a_explore_h): per scale the count on M (≤ 1e-3 of M's point at min(s, 0.9999·s_F)), the count with ρ₂ ≥ q and the
    median ρ₂."""
    rng = np.random.default_rng(PERTURB_SEED)
    out = {}
    for f in VALID_SCALES:
        s = f * s_F
        th = M.theta(min(s, 0.9999 * s_F))
        if th is None:
            return None
        Pb = SBF.run_to_full(th[None])[0]
        P0 = Pb[None] + rng.normal(0, 1, (PERTURB_N, 16)) * (0.05 * np.abs(Pb) + 0.02)
        P0[:, 12:] = np.abs(P0[:, 12:])
        P, L, gn = SBF.local_min_batch(P0, np.full(PERTURB_N, s), X, Y)
        d = SBF.fdist(P, Pb[None], X)
        r = np.array([rho2_static(p, X) for p in P])
        out[f"{f:g}"] = {"n_on_M": int((d <= ON_TOL).sum()), "n_rho2_ge_q": int((r >= Q).sum()),
                         "rho2_med": float(np.median(r)), "gnorm_max": float(np.max(gn))}
    return out


def lam_grid(M, s_F, X, Y):
    """λ_min of the training-coordinate tangent Hessian (P = I) on the Δs = 0.001 grid over the last 0.05 below s_F
    (p2a_explore_h's grid).  (s grid, λ_min) or None if Newton fails at a grid point."""
    ss = np.arange(s_F - MC_WINDOWS[0], s_F, MC_DS)
    ss = ss[ss < s_F - 1e-6]
    lam = []
    for s in ss:
        th = M.theta(s)
        if th is None:
            return None
        lam.append(tangent_lmin(th, X, Y))
    return ss, np.array(lam)


def mc_fit(ss, lam, s_F, window, form="quad"):
    """|m′c′| = |a|/4 from the least-squares fit λ_min² = a·d + b·d² (form 'quad', Phase 2A's estimator) or
    a·d + c·d^{3/2} + b·d² (form 'fold', DESCRIPTIVE), d = s_F − s, over the grid points with d ≤ window.
    (|m′c′|, a, n points)."""
    ss, lam = np.asarray(ss, float), np.asarray(lam, float)
    m = ss >= s_F - window
    d = s_F - ss[m]
    A = np.vstack([d, d ** 2]).T if form == "quad" else np.vstack([d, d ** 1.5, d ** 2]).T
    coef, *_ = np.linalg.lstsq(A, lam[m] ** 2, rcond=None)
    return float(abs(coef[0]) / 4), float(coef[0]), int(m.sum())


def fold_constant(ss, lam, s_F):
    """The page's |m′c′| (rule 4): Phase 2A's quadratic estimator over the last 0.05 (the value; Λ_F = √(|m′c′|·s_F))
    and over the last 0.025 (the check); agreement = |mc(0.025)/mc(0.05) − 1|.  The fold-form fit is DESCRIPTIVE."""
    mc, a, n = mc_fit(ss, lam, s_F, MC_WINDOWS[0])
    mc2, a2, n2 = mc_fit(ss, lam, s_F, MC_WINDOWS[1])
    out = {"abs_mc": mc, "abs_mc_n": n, "coef_linear": a, "abs_mc_0.025": mc2, "abs_mc_0.025_n": n2,
           "rel_diff": float(abs(mc2 / mc - 1)) if mc > 0 else None,
           "lam_min_first": float(lam[0]), "lam_min_last": float(lam[-1]), "Lambda_F": math.sqrt(mc * s_F)}
    out["DESCRIPTIVE_foldform"] = {f"{w:g}": mc_fit(ss, lam, s_F, w, "fold")[0] for w in MC_WINDOWS}
    return out


def classify(n_rho2_ge_q):
    n = int(n_rho2_ge_q)
    return "clean" if n >= CLEAN_MIN else ("none" if n == 0 else "mixed")


def release_scale(lo, s_F):
    """Rule 2: s₀ = lo + FRAC0·(s_F − lo) on the seed's own stable M."""
    return lo + FRAC0 * (s_F - lo)


def is_eligible(s0, s_F):
    """Rule 2: eligible iff s₀ ≤ 0.6·s_F (inclusive)."""
    return bool(s0 <= ELIG_MAX * s_F)


def freeze_seed(seed, timing=True):
    """Every per-seed frozen quantity (no training): the RAW record (freeze_raw) and its evaluation by the registered
    rules (evaluate: checks, status, class, Λ_F)."""
    rec = freeze_raw(seed, timing)
    rec["evaluation"] = evaluate(rec)
    return rec


def freeze_raw(seed, timing=True):
    """The raw per-seed quantities.  Stops early only when the seed's M cannot be traced (no branch to measure)."""
    t0 = time.time()
    X, Y = sample(seed)
    rec = {"seed": int(seed), "sample_sha256": sample_sha(X, Y), "n_points": int(len(Y))}

    def done():
        if timing:
            rec["secs"] = round(time.time() - t0, 2)
        return rec
    u, nA, s_h, hinfo = homotopy_accepted("M", X, Y)
    rec["M_homotopy"] = hinfo
    if u is None:
        return done()
    R = SBF.Reduced(nA, X, Y)
    rec["nA"] = int(nA)
    pf, ff, pb, fb = continue_both(R, u, s_h, H_CONT)
    rec["fold_up"] = ff[0]["s_fold"] if ff else None
    rec["fold_up_eig"] = ff[0]["eig_min"] if ff else None
    rec["fold_down"] = fb[0]["s_fold"] if fb else None
    M = SeedBranch(R, pb, pf)
    rec["M_start_stable"] = M.start_stable
    if not ff or not M.start_stable:
        return done()
    s_F = float(ff[0]["s_fold"])
    rec["s_F"] = s_F
    rec["stable_to_fold"] = bool(M.n_pf_stable_prefix >= len(pf) - 1)    # pf[-1] is the point past the fold
    rec["M_stable"] = [M.lo, M.hi]
    rec["M_contiguous_equals_all_stable"] = M.contiguous_equals_all_stable
    r2 = [rho2_static(R.to_full(x[:-1])[0], X) for x in M.xs]           # every stable continuation point
    P_end = M.P(0.9999 * s_F)                                            # and M at 0.9999·s_F
    r2.append(rho2_static(P_end, X) if P_end is not None else math.inf)
    rec["rho2_M_max"] = float(max(r2))
    rec["rho2_M_n_points"] = len(r2)
    if not (rec["stable_to_fold"] and rec["rho2_M_max"] < Q):
        return done()
    # ---- s_F validation, class, fold constant
    _, ff2 = SBF.continuation(R.F, R.JF, np.r_[u, s_h], +1, H_CHECK, n_max=int(CONT_LENGTH / H_CHECK))
    rec["fold_up_h2"] = ff2[0]["s_fold"] if ff2 else None
    rec["validation"] = perturbed_minima(M, s_F, X, Y)
    g = lam_grid(M, s_F, X, Y)
    rec["lam_grid"] = None if g is None else {"s": [float(x) for x in g[0]], "lam_min": [float(x) for x in g[1]]}
    rec["fold_constant"] = None if g is None else fold_constant(g[0], g[1], s_F)
    # ---- S and s*
    uS, nAS, sS, hS = homotopy_accepted("S", X, Y)
    rec["S_homotopy"] = hS
    s_star, why = None, "S homotopy failed"
    if uS is not None:
        RS = SBF.Reduced(nAS, X, Y)
        pfS, ffS, pbS, fbS = continue_both(RS, uS, sS, H_CONT)
        S = SeedBranch(RS, pbS, pfS)
        rec["S_fold_up"] = ffS[0]["s_fold"] if ffS else None
        rec["S_fold_down"] = fbS[0]["s_fold"] if fbS else None
        if S.start_stable:
            rec["S_stable"] = [S.lo, S.hi]
            s_star, why = equal_loss_s(M, S)
            if S.lo <= CLASS_SCALE * s_F <= S.hi:
                PS_ = S.P(CLASS_SCALE * s_F)
                rec["rho2_S_at_1.01sF"] = rho2_static(PS_, X) if PS_ is not None else None
        else:
            why = "S homotopy point not stable"
    rec["s_star"] = s_star
    rec["s_star_note"] = why
    rec["sF_over_sstar"] = s_F / s_star if s_star else None
    # ---- release (exact M(s₀), Newton) and the follow-check anchor M(0.8·s_F)
    s0 = release_scale(M.lo, s_F)
    rec["s0"] = s0
    rec["s0_over_sF"] = s0 / s_F
    th0 = M.theta(s0)
    uf = M.u(FOLLOW_FRAC * s_F)
    if th0 is not None:
        rec["release_theta"] = [float(x) for x in th0]
        rec["release_sha256"] = state_sha(th0)
        rec["release_active_units"] = [k for k in range(4) if th0[12 + k] != 0.0]
        rec["release_rho2"] = Rho2Grid(X)(th0)
        Lr, gr = loss_grad(th0, X, Y)
        rec["release_loss"] = Lr
        rec["release_grad_max"] = float(np.abs(gr[[i for k in rec["release_active_units"]
                                                   for i in (2 * k, 2 * k + 1, 8 + k)] + [16]]).max())
    rec["u_follow"] = None if uf is None else [float(x) for x in uf]
    rec["s_follow"] = FOLLOW_FRAC * s_F
    return done()


def evaluate(rec):
    """The registered rules applied to a raw record: the identity (untraceable reasons), the s_F checks, the |m′c′|
    window agreement, s*, the release, eligibility, the class, Λ_F.  status = the FIRST failing reason in STATUS_ORDER,
    else 'scoreable'.  The class is reported for every seed with perturbed minima (counted only when scoreable)."""
    out = {"status": None, "class": None, "checks": {}, "Lambda_F": None, "abs_mc": None, "eligible": None,
           "s_F": rec.get("s_F"), "s_star": rec.get("s_star"), "s0": rec.get("s0")}
    if not rec["M_homotopy"]["accepted"]:
        out["status"] = "untraceable: M homotopy failed"
        return out
    if rec.get("fold_up") is None:
        out["status"] = "untraceable: no upper fold"
        return out
    if not rec.get("M_start_stable"):
        out["status"] = "untraceable: homotopy point not stable"
        return out
    if not rec.get("stable_to_fold"):
        out["status"] = "untraceable: stable part does not reach the upper fold"
        return out
    if not (rec.get("rho2_M_max") is not None and rec["rho2_M_max"] < Q):
        out["status"] = "untraceable: rho2 >= q on the stable part"
        return out
    s_F, val, fc = rec["s_F"], rec.get("validation"), rec.get("fold_constant")
    if val is not None:
        out["class_n_rho2_ge_q"] = val[f"{CLASS_SCALE:g}"]["n_rho2_ge_q"]
        out["class"] = classify(out["class_n_rho2_ge_q"])
    c = {"h_half_agrees": bool(rec.get("fold_up_h2") is not None
                               and abs(rec["fold_up_h2"] - s_F) <= FOLD_AGREE_REL * s_F),
         "fold_eig_zero": bool(rec.get("fold_up_eig") is not None and abs(rec["fold_up_eig"]) <= FOLD_EIG_TOL),
         "lam_min_decreasing_to_fold": bool(fc is not None and fc["coef_linear"] > 0
                                            and fc["lam_min_last"] < fc["lam_min_first"]),
         "none_on_M_above": bool(val is not None and all(val[f"{f:g}"]["n_on_M"] == 0 for f in ABOVE_SCALES)),
         "mc_windows_agree": bool(fc is not None and fc["rel_diff"] is not None and fc["rel_diff"] <= MC_AGREE)}
    out["checks"] = c
    out["mc_agree_DESCRIPTIVE"] = c["mc_windows_agree"]       # author 2026-10-05, D1 (c): descriptive only
    if fc is not None:
        out["abs_mc"], out["Lambda_F"] = fc["abs_mc"], fc["Lambda_F"]
    out["eligible"] = is_eligible(rec["s0"], s_F)
    if not (c["h_half_agrees"] and c["fold_eig_zero"] and c["lam_min_decreasing_to_fold"] and c["none_on_M_above"]):
        out["status"] = "s_F not validated"
    elif fc is None or not (fc["abs_mc"] > 0 and math.isfinite(fc["abs_mc"])):
        out["status"] = "Lambda_F not validated"       # D1 (c): the window agreement no longer removes seeds
    elif rec.get("s_star") is None:
        out["status"] = "no s*"
    elif rec.get("release_theta") is None or rec.get("u_follow") is None:
        out["status"] = "release Newton failed"
    elif not out["eligible"]:
        out["status"] = "ineligible"
    else:
        out["status"] = "scoreable"
    return out


STATUS_ORDER = ("untraceable: M homotopy failed", "untraceable: no upper fold",
                "untraceable: homotopy point not stable", "untraceable: stable part does not reach the upper fold",
                "untraceable: rho2 >= q on the stable part", "s_F not validated", "Lambda_F not validated", "no s*",
                "release Newton failed", "ineligible", "scoreable")


def _st(r):
    return r["evaluation"]["status"]


def _cl(r):
    return r["evaluation"]["class"]


def counts(rows):
    """Counts by status (untraceable, constants not validated, no s*, ineligible, scoreable) and by class."""
    st = [_st(r) for r in rows]
    c = {k: int(sum(1 for x in st if x == k)) for k in STATUS_ORDER}
    sc = [r for r in rows if _st(r) == "scoreable"]
    by_class = {k: int(sum(1 for r in sc if _cl(r) == k)) for k in ("clean", "none", "mixed")}
    traced = [r for r in rows if not _st(r).startswith("untraceable")]
    return {"n_seeds": len(rows), "by_status": c,
            "n_untraceable": int(sum(1 for x in st if x.startswith("untraceable"))),
            "n_constants_not_validated": c["s_F not validated"] + c["Lambda_F not validated"],
            "n_no_s_star": c["no s*"], "n_release_failed": c["release Newton failed"], "n_ineligible": c["ineligible"],
            "n_scoreable": c["scoreable"], "scoreable_by_class": by_class,
            "traced_by_class_all": {k: int(sum(1 for r in traced if _cl(r) == k)) for k in ("clean", "none", "mixed")}}


def gate(rows):
    n = int(sum(1 for r in rows if _st(r) == "scoreable" and _cl(r) == "clean"))
    return {"n_clean_eligible": n, "min": GATE_MIN_CLEAN, "pass": bool(n >= GATE_MIN_CLEAN)}


def plan(rows):
    """The first 40 clean (at 2⁻¹⁴ and 2⁻¹⁶), 30 none and 20 mixed (at 2⁻¹⁴) scoreable seeds in seed order; the rest
    are counted, not trained.  Returns ([(seed, class, log2rho), ...], {class: [surplus seeds]})."""
    rows = sorted(rows, key=lambda r: r["seed"])
    chosen = {k: [] for k in CAPS}
    surplus = {k: [] for k in CAPS}
    for r in rows:
        if _st(r) != "scoreable":
            continue
        k = _cl(r)
        (chosen[k] if len(chosen[k]) < CAPS[k] else surplus[k]).append(r["seed"])
    pl = []
    for s in chosen["clean"]:
        pl += [(s, "clean", LOG2_EXPONENT), (s, "clean", LOG2_SCORED)]
    pl += [(s, "none", LOG2_OTHER) for s in chosen["none"]]
    pl += [(s, "mixed", LOG2_OTHER) for s in chosen["mixed"]]
    return pl, surplus


def class_listing(rows):
    """The canonical class listing hashed into frozen.json: 'seed status class' per seed, in seed order."""
    return "".join(f"{r['seed']} {_st(r)} {_cl(r)}\n" for r in sorted(rows, key=lambda r: r["seed"]))


# ------------------------------------------------------------------------------------------ training and forecasts
def train_to_cutoff(theta0, rho, s_F, X, Y, f=F_CUT, n_budget=None):
    """P2A.train_to_cutoff on the seed's sample: from the release to t_c = the first t ≥ 1 with s_t ≥ f·s_F (a stopping
    time on s); no ρ₂, no branch check; hard budget ⌈64/ρ⌉.  Returns (s_0 … s_{t_c}, θ_{t_c}, t_c) or (path, θ, None)."""
    th = np.array(theta0, float)
    active = [k for k in range(4) if th[12 + k] != 0.0]
    step = make_step(rho, active, X, Y)
    nb = budget(rho) if n_budget is None else int(n_budget)
    S = np.empty(nb + 1)
    S[0] = scale(th)
    level = f * s_F
    for t in range(1, nb + 1):
        th = step(th)
        S[t] = scale(th)
        if S[t] >= level:
            return S[:t + 1].copy(), th, t
        if t % 1_000_000 == 0:
            P2A.rss_guard()
    return S, th, None


def run_one(fs, l2, n_budget=None):
    """One planned run: training from the seed's frozen release to t_c, the state hashed, the causal forecast with the
    seed's own (s_F, Λ_F) and the same forecaster at the same t_c with the fixed-dataset fold (S_F_FIXED, LAMF_FIXED);
    both guarded, NaN recomputation identical (else CausalityViolation)."""
    rho = rho_of(l2)
    X, Y = sample(fs["seed"])
    assert sample_sha(X, Y) == fs["sample_sha256"], "sample hash mismatch"
    th0 = np.array(fs["release_theta"], float)
    assert state_sha(th0) == fs["release_sha256"], "release hash mismatch"
    nb = budget(rho) if n_budget is None else int(n_budget)
    S, th, t_c = train_to_cutoff(th0, rho, fs["s_F"], X, Y, F_CUT, nb)
    rec = {"seed": fs["seed"], "class": fs["class"], "log2rho": float(l2), "rho": rho, "budget": nb, "t_c": t_c}
    if t_c is None:
        nofc = {"status": "no forecast: no cutoff within the budget", "t_fc": None, "t_F_fc": None, "s_c_fc": None}
        rec.update(forecast=nofc, forecast_fixed=dict(nofc), nan_recompute_identical=None)
        return rec
    fc, same = CFF.run_forecast(S, t_c, fs["s_F"], fs["Lambda_F"], ETA, nb - t_c, F_CUT)
    fx, same_x = CFF.run_forecast(S, t_c, S_F_FIXED, LAMF_FIXED, ETA, nb - t_c, F_CUT)
    if not (same and same_x):
        raise CF.CausalityViolation(f"NaN recomputation differs (seed {fs['seed']}, log2rho {l2})")
    rec.update(s_at_tc=float(S[t_c]), state_tc=[float(x) for x in th], state_tc_sha256=state_sha(th), forecast=fc,
               forecast_fixed=fx, nan_recompute_identical=bool(same and same_x))
    return rec


def follow_check(th, fs, X, Y):
    """The state minimised at its own s (sb_fold.local_min_batch) lies within 1e-3 of the seed's own M point at that
    exact s (Newton from the frozen M point at 0.8·s_F).  (bool, distance)."""
    s = scale(th)
    P, _, _ = SBF.local_min_batch(SBF.run_to_full(np.atleast_2d(th)), np.array([s]), X, Y)
    R = SBF.Reduced(fs["nA"], X, Y)
    u, ok, _ = SBF.newton_fixed_s(R.F, R.JF, np.array(fs["u_follow"], float), s)
    if not ok:
        return False, math.inf
    d = float(SBF.fdist(P, R.to_full(u)[0][None], X)[0])
    return bool(d <= ON_TOL), d


def observe_one(fs, l2, run_rec, n_budget=None):
    """P2A.observe_rate on the seed's sample: retrain from the release with ρ₂ at EVERY step; at t_c assert the state
    equals the hashed one bit for bit and continue from it until the crossing and s_F are both reached (t ≥ t_c), or
    s ≥ 1.25·s_F, or the budget.  Records t_obs, s_obs, t_F, the follow check at 0.8·s_F, the idle unit and the active
    signs up to the crossing.  Returns (record, s path)."""
    rho = rho_of(l2)
    X, Y = sample(fs["seed"])
    assert sample_sha(X, Y) == fs["sample_sha256"], "sample hash mismatch"
    rho2 = Rho2Grid(X)
    s_F = fs["s_F"]
    th = np.array(fs["release_theta"], float)
    active = [k for k in range(4) if th[12 + k] != 0.0]
    idle = [k for k in range(4) if k not in active]
    idle_idx = [i for k in idle for i in (2 * k, 2 * k + 1, 8 + k, 12 + k)]
    sg0 = np.sign(th[12:16][active])
    step = make_step(rho, active, X, Y)
    nb = budget(rho) if n_budget is None else int(n_budget)
    t_c = run_rec["t_c"]
    S = np.empty(nb + 1)
    S[0] = scale(th)
    r0 = rho2(th)
    below = r0 < Q
    t_obs = rho2_obs = t_F = t08 = th08 = None
    idle_viol = sign_viol = hash_ok = None
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
            if r >= Q and below:
                t_obs, rho2_obs = t, r
            elif r < Q:
                below = True
        if t08 is None and S[t] >= FOLLOW_FRAC * s_F:
            t08, th08 = t, th.copy()
        if t_F is None and S[t] >= s_F:
            t_F = t
        if t_c is not None and t == t_c:
            hash_ok = bool(state_sha(th) == run_rec["state_tc_sha256"]
                           and np.array_equal(th, np.array(run_rec["state_tc"], float)))
            if not hash_ok:
                raise AssertionError(f"state at t_c differs from the hashed one (seed {fs['seed']}, log2rho {l2})")
            th = np.array(run_rec["state_tc"], float)
        if S[t] >= OBS_END * s_F:
            break
        if t_obs is not None and t_F is not None and (t_c is None or t >= t_c):
            break
        if t % 1_000_000 == 0:
            P2A.rss_guard()
    S = S[:t + 1].copy()
    follow = follow_check(th08, fs, X, Y) if th08 is not None else (False, math.inf)
    rec = {"seed": fs["seed"], "class": fs["class"], "log2rho": float(l2), "steps_observed": int(t), "t_obs": t_obs,
           "s_obs": float(S[t_obs]) if t_obs is not None else None, "rho2_at_obs": rho2_obs, "rho2_release": r0,
           "t_F": t_F, "s_end": float(S[-1]), "reached_obs_end": bool(S[-1] >= OBS_END * s_F),
           "t_08": t08, "follow_in_M": bool(follow[0]), "follow_dist_M": follow[1],
           "idle_zero_to_crossing": idle_viol is None, "idle_first_violation": idle_viol,
           "signs_fixed_to_crossing": sign_viol is None, "sign_first_violation": sign_viol, "state_tc_hash_ok": hash_ok}
    return rec, S


# ------------------------------------------------------------------------------------------ pure scoring rules
_arr = P2A._arr
verdict = P2A.verdict


def boot_ci(D, stat="mean", n_boot=BOOT_N, seed=BOOT_SEED):
    """(statistic, lower, upper): the 95% percentile bootstrap of the mean or median over SEED resamples (a fresh local
    default_rng(2,975,000))."""
    D = np.asarray(D, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    f = np.mean if stat == "mean" else np.median
    lo, hi = np.percentile(f(D[idx], axis=1), BOOT_PCT)
    return float(f(D)), float(lo), float(hi)


def criterion_F(s_obs, s_F, s_star):
    """F (clean seeds, 2⁻¹⁶): s_F ≤ s_obs ≤ 1.25·s_F (inclusive, each seed's own s_F) at ≥ 90% of the trained clean
    seeds (no crossing: not within), AND no falsifier: a crossing BELOW s_F and at or below 1.25·s* (own s*).  No seed:
    UNRESOLVED."""
    s_obs, s_F, s_star = _arr(s_obs), _arr(s_F), _arr(s_star)
    with np.errstate(invalid="ignore"):
        within = np.isfinite(s_obs) & (s_obs >= s_F) & (s_obs <= F_HI * s_F)
        fals = np.isfinite(s_obs) & (s_obs < s_F) & (s_obs <= F_FALSIFIER * s_star)
    n = len(s_obs)
    frac = int(within.sum()) / n if n else float("nan")
    ok = bool(n and frac >= F_MIN_FRAC and not fals.any())
    return {"n": n, "n_within": int(within.sum()), "n_no_crossing": int((~np.isfinite(s_obs)).sum()),
            "n_below_sF": int((np.isfinite(s_obs) & (s_obs < s_F)).sum()), "frac": frac, "min_frac": F_MIN_FRAC,
            "n_falsifier": int(fals.sum()), "falsified": bool(fals.any()), "verdict": verdict(ok, n > 0)}


def criterion_H(crossed, reached_end):
    """H (none seeds, 2⁻¹⁴): no crossing by 1.25·s_F at ≥ 80% of the trained none seeds.  A seed counts iff it reached
    s ≥ 1.25·s_F with no crossing (a run ending at the step budget below 1.25·s_F without crossing does NOT count).
    Fewer than 10 none seeds: UNRESOLVED."""
    crossed, reached_end = _arr(crossed, bool), _arr(reached_end, bool)
    ok = (~crossed) & reached_end
    n = len(ok)
    frac = int(ok.sum()) / n if n else float("nan")
    return {"n": n, "n_no_crossing_to_1.25sF": int(ok.sum()), "n_crossed": int(crossed.sum()),
            "n_budget_end_no_crossing": int(((~crossed) & ~reached_end).sum()), "frac": frac, "min_frac": H_MIN_FRAC,
            "min_n": H_MIN_N, "verdict": verdict(n >= H_MIN_N and frac >= H_MIN_FRAC, n >= H_MIN_N)}


def seed_exponent(r14, r16, e14, e16):
    """Δln r_obs/Δln ε̂_F between 2⁻¹⁴ and 2⁻¹⁶ (None unless both r > 0 and both ε̂ > 0, finite, distinct)."""
    vals = (r14, r16, e14, e16)
    if any(v is None or not np.isfinite(v) or v <= 0 for v in vals) or e14 == e16:
        return None
    return (math.log(r16) - math.log(r14)) / (math.log(e16) - math.log(e14))


def criterion_E_seed(exponents):
    """E_seed: the median over clean seeds of the per-seed exponent; its 95% percentile bootstrap interval (10,000 seed
    resamples, default_rng(2,975,000)) must lie inside [0.55, 0.80] (closed).  Seeds without an exponent (no forecast
    or no crossing above s_F at either rate) are excluded and counted.  Fewer than 3: UNRESOLVED."""
    x = _arr(exponents)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n < MIN_SEEDS_STAT:
        return {"n": n, "n_excluded": int((~ok).sum()), "median": None, "ci95": [None, None], "band": list(E_BAND),
                "verdict": "UNRESOLVED"}
    med, lo, hi = boot_ci(x[ok], "median")
    return {"n": n, "n_excluded": int((~ok).sum()), "median": med, "ci95": [lo, hi], "band": list(E_BAND),
            "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
            "verdict": verdict(E_BAND[0] <= lo and hi <= E_BAND[1], bool(np.isfinite(lo) and np.isfinite(hi)))}


def criterion_C3(r_obs, r_fc, has_fc):
    """C3: r_obs/r_fc > 1 at ≥ 90% of the trained clean seeds (2⁻¹⁶); a seed with no forecast or no crossing is not
    > 1.  No seed: UNRESOLVED."""
    r_obs, r_fc, has_fc = _arr(r_obs), _arr(r_fc), _arr(has_fc, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(has_fc & (r_fc > 0), r_obs / r_fc, np.nan)
        gt = np.isfinite(ratio) & (ratio > 1)
    n = len(ratio)
    frac = int(gt.sum()) / n if n else float("nan")
    fin = np.isfinite(ratio)
    return {"n": n, "n_gt_1": int(gt.sum()), "n_undefined": int((~fin).sum()), "frac": frac, "min_frac": C3_MIN_FRAC,
            "ratio_min": float(ratio[fin].min()) if fin.any() else None,
            "ratio_median": float(np.median(ratio[fin])) if fin.any() else None,
            "ratio_max": float(ratio[fin].max()) if fin.any() else None,
            "verdict": verdict(n and frac >= C3_MIN_FRAC, n > 0)}


def criterion_C4(t_fc, t_F_fc, t_obs, has_fc):
    """C4: D = |t_fc − t_obs| − |t̂_F − t_obs| per clean seed (2⁻¹⁶).  Any trained clean seed with no forecast: FAIL.
    Otherwise over the seeds with a crossing: PASS iff the upper end of the 95% percentile bootstrap interval of mean D
    (10,000 seed resamples) is < 0; fewer than 2 seeds: UNRESOLVED."""
    t_fc, t_F_fc, t_obs, has_fc = _arr(t_fc), _arr(t_F_fc), _arr(t_obs), _arr(has_fc, bool)
    n_nofc = int((~has_fc).sum())
    ok = has_fc & np.isfinite(t_obs)
    D = np.abs(t_fc[ok] - t_obs[ok]) - np.abs(t_F_fc[ok] - t_obs[ok])
    n = int(ok.sum())
    mean, lo, hi = boot_ci(D) if n >= 2 else (float("nan"),) * 3
    computable = bool(n >= 2 and np.isfinite(hi))
    v = "FAIL" if n_nofc else verdict(hi < 0, computable)
    return {"n": n, "n_no_forecast": n_nofc, "n_no_crossing": int((has_fc & ~np.isfinite(t_obs)).sum()),
            "mean_D": mean, "ci95": [lo, hi], "boot_n": BOOT_N, "boot_seed": BOOT_SEED, "verdict": v,
            "DESCRIPTIVE_frac_forecast_closer": float((D < 0).mean()) if n else None}


def criterion_P(s_c_own, s_c_fixed, s_obs):
    """P: D = |ln ŝ_c/s_obs| − |ln ŝ_c^fixed/s_obs| per clean seed with a crossing (2⁻¹⁶; ŝ_c from the seed's own
    fold, ŝ_c^fixed from the same forecaster at the same t_c with the fixed-dataset fold).  A crossing seed without its
    own ŝ_c: FAIL.  A crossing seed without ŝ_c^fixed: excluded, counted.  PASS iff the upper end of the 95% percentile
    bootstrap interval of mean D (10,000 seed resamples) is < 0; fewer than 2 seeds: UNRESOLVED."""
    a, b, o = _arr(s_c_own), _arr(s_c_fixed), _arr(s_obs)
    cross = np.isfinite(o)
    with np.errstate(invalid="ignore"):
        own_missing = cross & ~(np.isfinite(a) & (a > 0))
        fx_missing = cross & ~own_missing & ~(np.isfinite(b) & (b > 0))
    ok = cross & ~own_missing & ~fx_missing
    D = np.abs(np.log(a[ok] / o[ok])) - np.abs(np.log(b[ok] / o[ok]))
    n = int(ok.sum())
    mean, lo, hi = boot_ci(D) if n >= 2 else (float("nan"),) * 3
    computable = bool(n >= 2 and np.isfinite(hi))
    v = "FAIL" if own_missing.any() else verdict(hi < 0, computable)
    return {"n": n, "n_no_crossing": int((~cross).sum()), "n_own_missing": int(own_missing.sum()),
            "n_fixed_missing_excluded": int(fx_missing.sum()), "mean_D": mean, "ci95": [lo, hi], "boot_n": BOOT_N,
            "boot_seed": BOOT_SEED, "verdict": v,
            "DESCRIPTIVE_frac_own_closer": float((D < 0).mean()) if n else None}


def criterion_within(err, delay_fc, tau, has_fc):
    """C1 / C2 (secondary): P2A.criterion_within over the trained clean seeds (2⁻¹⁶) at ≥ 80%."""
    return P2A.criterion_within(err, delay_fc, tau, has_fc, len(_arr(has_fc, bool)))


def validity(follow_in_M, t_c, t_obs, idle_ok_all, sign_ok_all):
    """At ≥ 90% of the trained clean seeds (2⁻¹⁶): the follow check at 0.8·s_F lands on the seed's own M (never
    reaching 0.8·s_F fails it); at ≥ 90%: t_c < t_obs (no cutoff fails; a cutoff and no crossing passes); evaluated
    separately.  The idle unit exactly 0 and the active signs fixed up to the crossing at EVERY clean run (2⁻¹⁶ and
    2⁻¹⁴; to the end of observation if no crossing)."""
    follow, t_c, t_obs = _arr(follow_in_M, bool), _arr(t_c), _arr(t_obs)
    n = len(follow)
    with np.errstate(invalid="ignore"):
        before = np.isfinite(t_c) & (~np.isfinite(t_obs) | (t_c < t_obs))
    ff = float(follow.sum()) / n if n else float("nan")
    fb = float(before.sum()) / n if n else float("nan")
    ia, sa = _arr(idle_ok_all, bool), _arr(sign_ok_all, bool)
    out = {"n": n, "follow_frac": ff, "follow_ok": bool(n and ff >= VALID_MIN_FRAC),
           "cutoff_before_frac": fb, "cutoff_before_ok": bool(n and fb >= VALID_MIN_FRAC),
           "n_cutoff_not_before": int(n - before.sum()), "n_runs_idle_sign": len(ia),
           "idle_zero_all": bool(len(ia) and ia.all()), "signs_fixed_all": bool(len(sa) and sa.all()),
           "min_frac": VALID_MIN_FRAC}
    out["ok"] = bool(out["follow_ok"] and out["cutoff_before_ok"] and out["idle_zero_all"] and out["signs_fixed_all"])
    return out


def outcome(verdicts, gate_ok=True, valid_ok=True):
    """The registered rule (P2A.outcome) over the PRIMARY criteria F, H, E_seed, C3, C4, P (C1, C2 excluded)."""
    return P2A.outcome({k: verdicts[k] for k in PRIMARY if k in verdicts}, gate_ok, valid_ok)


def run_row(fs, rr, ob):
    """One scored row: a planned run's frozen constants, forecasts and observation."""
    fc, fx = rr.get("forecast") or {}, rr.get("forecast_fixed") or {}
    s_obs = ob.get("s_obs")
    return {"seed": fs["seed"], "class": fs["class"], "log2rho": rr["log2rho"], "s_F": fs["s_F"],
            "s_star": fs["s_star"], "Lambda_F": fs["Lambda_F"], "t_c": rr.get("t_c"), "has_fc": CFF.has_forecast(fc),
            "status": fc.get("status"), "t_fc": fc.get("t_fc"), "t_F_fc": fc.get("t_F_fc"),
            "delay_fc": fc.get("delay_fc"), "eps_fc": fc.get("eps_fc"), "r_fc": fc.get("r_fc"),
            "s_c_fc": fc.get("s_c_fc"), "s_c_fixed": fx.get("s_c_fc"), "has_fc_fixed": CFF.has_forecast(fx),
            "t_fc_fixed": fx.get("t_fc"), "t_obs": ob.get("t_obs"), "s_obs": s_obs, "t_F": ob.get("t_F"),
            "r_obs": (s_obs / fs["s_F"] - 1) if s_obs is not None else None,
            "reached_obs_end": ob.get("reached_obs_end", False), "follow_in_M": ob.get("follow_in_M", False),
            "idle_ok": ob.get("idle_zero_to_crossing", False), "sign_ok": ob.get("signs_fixed_to_crossing", False),
            "mc_agree": bool(fs.get("mc_agree"))}


def descriptive_split(rows):
    """DESCRIPTIVE ONLY (author 2026-10-05, with D1 (c)): every criterion (F, H, E_seed, C3, C4, P, C1, C2) and
    validity recomputed separately over the seeds whose |m′c′| two-window agreement check (0.025 vs 0.05, ≤ 5%) passes
    and over those where it fails.  Never a verdict; the registered outcome is score_tables over all rows."""
    out = {"label": "DESCRIPTIVE: split by the |m'c'| two-window agreement check (not a criterion)"}
    for tag, flag in (("agree", True), ("disagree", False)):
        sub = [r for r in rows if bool(r.get("mc_agree")) is flag]
        st = score_tables(sub)
        out[tag] = {"n_rows": len(sub), "n_clean_scored": st["n_clean_scored"], "n_none": st["n_none"],
                    "verdicts_DESCRIPTIVE": st["verdicts"], "secondary_verdicts_DESCRIPTIVE": st["secondary_verdicts"],
                    "validity_DESCRIPTIVE": st["validity"], "criteria_DESCRIPTIVE": st["criteria"],
                    "secondary_DESCRIPTIVE": st["secondary"]}
    return out


def score_tables(rows, gate_ok=True):
    """Verdicts from the scored rows (run_row): F, C3, C4, P, C1, C2 and validity over the clean seeds at 2⁻¹⁶;
    E_seed over the clean seeds paired across 2⁻¹⁴ and 2⁻¹⁶; H over the none seeds at 2⁻¹⁴.  Mixed: descriptive."""
    if not gate_ok:
        return {"criteria": {}, "validity": {"ok": False}, "verdicts": {}, "outcome": outcome({}, False, False),
                "secondary": {}}
    c16 = [r for r in rows if r["class"] == "clean" and r["log2rho"] == LOG2_SCORED]
    c14 = {r["seed"]: r for r in rows if r["class"] == "clean" and r["log2rho"] == LOG2_EXPONENT}
    none = [r for r in rows if r["class"] == "none" and r["log2rho"] == LOG2_OTHER]
    g = {k: [r.get(k) for r in c16] for k in ("s_obs", "s_F", "s_star", "t_fc", "t_F_fc", "t_obs", "t_F", "delay_fc",
                                              "r_obs", "r_fc", "s_c_fc", "s_c_fixed")}
    has = [bool(r["has_fc"]) for r in c16]
    t_fc, t_obs, t_F, dly = _arr(g["t_fc"]), _arr(g["t_obs"]), _arr(g["t_F"]), _arr(g["delay_fc"])
    expo = []
    for r in c16:
        q = c14.get(r["seed"])
        ok14 = q is not None and q["has_fc"]
        expo.append(seed_exponent(q["r_obs"] if ok14 else None, r["r_obs"] if r["has_fc"] else None,
                                  q["eps_fc"] if ok14 else None, r["eps_fc"] if r["has_fc"] else None))
    crit = {"F": criterion_F(g["s_obs"], g["s_F"], g["s_star"]),
            "H": criterion_H([r["t_obs"] is not None for r in none], [bool(r["reached_obs_end"]) for r in none]),
            "E_seed": criterion_E_seed(expo),
            "C3": criterion_C3(g["r_obs"], g["r_fc"], has),
            "C4": criterion_C4(g["t_fc"], g["t_F_fc"], g["t_obs"], has),
            "P": criterion_P(g["s_c_fc"], g["s_c_fixed"], g["s_obs"])}
    with np.errstate(invalid="ignore"):
        sec = {"C1": criterion_within(t_fc - t_obs, dly, TAU1, has),
               "C2": criterion_within(dly - (t_obs - t_F), dly, TAU2, has)}
    allc = c16 + list(c14.values())
    val = validity([r["follow_in_M"] for r in c16], [r["t_c"] for r in c16], [r["t_obs"] for r in c16],
                   [r["idle_ok"] for r in allc], [r["sign_ok"] for r in allc])
    verdicts = {k: c["verdict"] for k, c in crit.items()}
    return {"criteria": crit, "secondary": sec, "secondary_verdicts": {k: c["verdict"] for k, c in sec.items()},
            "validity": val, "verdicts": verdicts, "outcome": outcome(verdicts, True, val["ok"]),
            "seed_exponents": expo, "n_clean_scored": len(c16), "n_clean_exponent": len(c14), "n_none": len(none)}


# ------------------------------------------------------------------------------------------ machine rules and io
def memory_gate(tag):
    """free ≥ 25% and swap free ≥ 500 MB (act_fold.check_memory; waits, re-checking every 60 s); disk free ≥ 20 GB
    (else STOP).  Every check logged to results/phase2a_ps/memory_gate.log."""
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


_setup = P2A._setup
_sha = P2A._sha
_jsonable = P2A._jsonable
_rows = P2A._rows


def _append_write(p, row):
    with open(p, "a") as fh:
        fh.write(json.dumps(_jsonable(row)) + "\n")


def _peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


# ------------------------------------------------------------------------------------------ seed scan
SCAN_DIRS = ("src", "tests", "results", "paper", "notes", "independent", "data", "dist")
SCAN_SKIP_FILES = ("phase2a_ps.py", "test_phase2a_ps.py", "phase2a_ps_registration.md", "phase2a_perseed_design.md")
SCAN_SKIP_SUFFIX = (".npz", ".npy", ".pdf", ".png", ".pt", ".pkl", ".ots", ".bak", ".gz", ".zip", ".pyc", ".jpg")
SCAN_PATTERNS = {
    "registered": r"(^|[^0-9.])2975[0-3][0-9][0-9]([^0-9]|$)|(^|[^0-9])2_975_[0-3][0-9][0-9]([^0-9]|$)"
                  r"|(^|[^0-9.,])2,975,[0-3][0-9][0-9]([^0-9,]|$)",
    "pilot": r"(^|[^0-9.])29760[0-5][0-9]([^0-9]|$)|(^|[^0-9])2_976_0[0-5][0-9]([^0-9]|$)"
             r"|(^|[^0-9.,])2,976,0[0-5][0-9]([^0-9,]|$)"}
SEED_MODULES = ("track_a", "track2a", "track2b", "track2c", "gelu_transfer", "width2_asym", "phase1a_pilot", "phase1c",
                "phase2a", "phase2b", "simplicity_bias_v2", "simplicity_bias_v3")


def scan_tree(patterns=SCAN_PATTERNS, dirs=SCAN_DIRS, chunk=16 * 1024 * 1024, overlap=256):
    """Regex scan of every text file under `dirs` (binary files skipped; 2A-PS's own files and results/phase2a_ps/
    skipped), streamed in 16 MB chunks with a 256-byte overlap.  {pattern key: [files with a match]}."""
    import re
    rx = {k: re.compile(v.encode()) for k, v in patterns.items()}
    hits = {k: [] for k in patterns}
    for top in dirs:
        for dp, dns, fns in os.walk(ROOT / top):
            dns[:] = [d for d in dns if d not in ("__pycache__", ".git")]
            if Path(dp).resolve() == OUT.resolve():
                dns[:] = []
                continue
            for fn in fns:
                p = Path(dp) / fn
                if fn in SCAN_SKIP_FILES or p.suffix in SCAN_SKIP_SUFFIX:
                    continue
                try:
                    with open(p, "rb") as fh:
                        if b"\0" in fh.read(4096):
                            continue
                        fh.seek(0)
                        found, tail = set(), b""
                        while True:
                            buf = fh.read(chunk)
                            if not buf:
                                break
                            data = tail + buf
                            for k, r in rx.items():
                                if k not in found and r.search(data):
                                    found.add(k)
                            tail = data[-overlap:]
                except OSError:
                    continue
                for k in found:
                    hits[k].append(str(p.relative_to(ROOT)))
    return hits


def scan():
    """Both 2A-PS seed ranges are unused: no number in 2,975,000-2,975,399 or 2,976,000-2,976,059 (also written with _
    or ,) in any text file under src/, tests/, results/, paper/, notes/, independent/, data/, dist/ (2A-PS's own files
    and the design page excepted), and no overlap with the SEEDS* / PILOT_SEEDS* constants of the registered modules."""
    import importlib
    from .phase2b import _flat_ints
    OUT.mkdir(parents=True, exist_ok=True)
    memory_gate("scan")
    hits = scan_tree()
    mine = set(SEEDS) | set(PILOT_SEEDS)
    ov = {}
    for mod in SEED_MODULES:
        m = importlib.import_module(f"src.{mod}")
        for name in dir(m):
            if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
                ov[f"{mod}.{name}"] = sorted(mine & set(_flat_ints(getattr(m, name))))
    out = {"ranges": {"registered": [SEEDS[0], SEEDS[-1], len(SEEDS)],
                      "pilot": [PILOT_SEEDS[0], PILOT_SEEDS[-1], len(PILOT_SEEDS)]}, "dirs": list(SCAN_DIRS),
           "patterns": SCAN_PATTERNS, "pattern_files": hits, "registered_seed_overlap": ov,
           "exploration_seeds_disjoint": bool(not (mine & set(EXPLORATION_SEEDS))),
           "unused": bool(not any(hits.values()) and not any(ov.values()))}
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable(out), indent=1))
    if not out["unused"]:
        raise SystemExit("STOP: a 2A-PS seed range is not unused")


# ------------------------------------------------------------------------------------------ inputs, freeze, pilot
def check_inputs():
    """The fixed-dataset constants equal the committed v2 and 2A frozen files."""
    v2f = json.loads(V2_FROZEN.read_text())
    p2f = json.loads(P2A_FROZEN.read_text())
    assert float(v2f["q"]) == Q, "q"
    assert float(p2f["s_F"]) == S_F_FIXED and float(p2f["Lambda_F"]) == LAMF_FIXED, "fixed-dataset fold"
    assert float(p2f["s0"]) == S0_FIXED
    return {"v2_frozen_sha256": _sha(V2_FROZEN), "phase2a_frozen_sha256": _sha(P2A_FROZEN)}


def _freeze_seeds(seeds, path, tag, stop_when=None):
    """Resumable per-seed freeze to `path` (one JSON line per seed, in order); memory gate and RSS guard between seeds."""
    done = {r["seed"] for r in _rows(path)}
    for s in seeds:
        if s in done:
            continue
        memory_gate(f"{tag} {s}")
        rec = freeze_seed(s)
        rec["peak_rss_gb"] = round(_peak_rss_gb(), 3)
        _append_write(path, rec)
        print(json.dumps({"seed": s, "status": _st(rec), "class": _cl(rec), "s_F": rec.get("s_F"),
                          "secs": rec.get("secs"), "rss_gb": rec["peak_rss_gb"]}), flush=True)
        P2A.rss_guard()
        if stop_when is not None and stop_when(rec):
            break
    return _rows(path)


def freeze():
    """The FREEZE of all 400 registered seeds (no training): per seed -> frozen_parts.jsonl (resumable); then
    summary() -> frozen.json (counts, classes and their hash, gate, plan)."""
    _setup()
    check_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _freeze_seeds(SEEDS, OUT / "frozen_parts.jsonl", "freeze")
    if {r["seed"] for r in rows} == set(SEEDS):
        summary()


def reevaluate():
    """Re-apply the registered rules (evaluate) to the stored raw freeze records, WITHOUT refreezing: every raw field is
    kept bit for bit and only 'evaluation' is replaced (author's decision 2026-10-05, D1 (c)); then summary()."""
    path = OUT / "frozen_parts.jsonl"
    rows = _rows(path)
    new = []
    for r in rows:
        raw = {k: v for k, v in r.items() if k != "evaluation"}
        new.append({**raw, "evaluation": evaluate(raw)})
        assert {k: v for k, v in new[-1].items() if k != "evaluation"} == raw
    path.write_text("".join(json.dumps(_jsonable(r)) + "\n" for r in new))
    summary()


def summary():
    """frozen.json from frozen_parts.jsonl: every seed's status and class, the counts (untraceable, constants not
    validated, no s*, ineligible, clean/none/mixed), the class listing and its SHA-256, the gate, the plan."""
    rows = sorted(_rows(OUT / "frozen_parts.jsonl"), key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == list(SEEDS), "every registered seed frozen exactly once"
    pl, surplus = plan(rows)
    listing = class_listing(rows)
    for r in rows:                                    # the stored evaluation is the registered rules' evaluation
        assert evaluate(r) == r["evaluation"], f"evaluation of seed {r['seed']} differs from the stored one"
    sc = [seed_record(rows, r["seed"]) for r in rows if _st(r) == "scoreable"]
    agree = {k: {"agree": int(sum(1 for r in sc if r["class"] == k and r["mc_agree"])),
                 "disagree": int(sum(1 for r in sc if r["class"] == k and not r["mc_agree"]))}
             for k in ("clean", "none", "mixed")}
    fr = {"label": "Phase 2A-PS frozen per-seed inputs (before the registration; no training)",
          **check_inputs(), "frozen_parts_sha256": _sha(OUT / "frozen_parts.jsonl"),
          "counts": counts(rows), "gate": gate(rows),
          "class_listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
          "classes": {str(r["seed"]): [_st(r), _cl(r)] for r in rows},
          "plan": [{"seed": s, "class": k, "log2rho": l2} for s, k, l2 in pl],
          "plan_counts": {k: int(sum(1 for _, kk, l2 in pl if kk == k and (k != "clean" or l2 == LOG2_SCORED)))
                          for k in CAPS},
          "surplus_counted_not_trained": surplus,
          "DESCRIPTIVE_mc_agreement_scoreable_by_class": agree,
          "DESCRIPTIVE_mc_agreement_planned": {k: {"agree": int(sum(1 for s_, kk, l2 in pl if kk == k and (
              k != "clean" or l2 == LOG2_SCORED) and seed_record(rows, s_)["mc_agree"])), "disagree": int(sum(
              1 for s_, kk, l2 in pl if kk == k and (k != "clean" or l2 == LOG2_SCORED)
              and not seed_record(rows, s_)["mc_agree"]))} for k in CAPS},
          "D1_decision": "author 2026-10-05: (c) Lambda_F from the 0.05 window; the two-window agreement (<= 5%) is "
                         "DESCRIPTIVE only; clarification made at freeze from landscape counts only, before any "
                         "training ((a) had given Lambda_F not validated 91, clean 45)",
          "DESCRIPTIVE": {"s_F_range_scoreable": [min(r["s_F"] for r in sc), max(r["s_F"] for r in sc)] if sc else None,
                          "sF_over_sstar_range_scoreable": ([min(r["sF_over_sstar"] for r in sc),
                                                             max(r["sF_over_sstar"] for r in sc)] if sc else None),
                          "Lambda_F_range_scoreable": ([min(r["Lambda_F"] for r in sc),
                                                        max(r["Lambda_F"] for r in sc)] if sc else None),
                          "freeze_secs_total": round(sum(r.get("secs", 0) for r in rows), 1),
                          "peak_rss_gb": max(r.get("peak_rss_gb", 0) for r in rows),
                          "homotopy_200_used": int(sum(1 for r in rows if r["M_homotopy"]["n_steps"] == 200)),
                          "M_contiguous_differs_from_all_stable": [r["seed"] for r in rows
                                                                   if r.get("M_contiguous_equals_all_stable") is False]},
          "rules": {"FRAC0": FRAC0, "ELIG_MAX": ELIG_MAX, "CLEAN_MIN": CLEAN_MIN, "CAPS": CAPS,
                    "GATE_MIN_CLEAN": GATE_MIN_CLEAN, "FOLD_AGREE_REL": FOLD_AGREE_REL, "FOLD_EIG_TOL": FOLD_EIG_TOL,
                    "MC_AGREE": MC_AGREE, "HOMOTOPY_STEPS": HOMOTOPY_STEPS, "S_HOMOTOPY": S_HOMOTOPY}}
    (OUT / "frozen.json").write_text(json.dumps(_jsonable(fr), indent=1))
    print(json.dumps(_jsonable({k: fr[k] for k in ("counts", "gate", "class_listing_sha256", "plan_counts")}),
                     indent=1))


def seed_record(rows, seed):
    """A seed's raw record flattened with its evaluation (status, class, Λ_F)."""
    r = next(x for x in rows if x["seed"] == seed)
    e = r["evaluation"]
    return {**r, "status": e["status"], "class": e["class"], "Lambda_F": e["Lambda_F"], "eligible": e["eligible"],
            "mc_agree": e.get("mc_agree_DESCRIPTIVE")}


def pilot():
    """Pilot seeds only (2,976,000-2,976,059), a MACHINE check that fixes no rule, threshold or criterion: freeze pilot
    seeds in order until the first scoreable clean one; then that seed's runs at 2⁻¹⁴ and 2⁻¹⁶ through run_one and
    observe_one (the hashed-state resume, the guarded forecasts, time and peak RSS at the slowest rate).  STOP if a run
    breaks a machine rule (RSS > 3 GB, a NaN recomputation differing, a hash mismatch).  -> pilot_parts.jsonl, pilot.json"""
    _setup()
    check_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _freeze_seeds(PILOT_SEEDS, OUT / "pilot_parts.jsonl", "pilot freeze",
                         stop_when=lambda r: _st(r) == "scoreable" and _cl(r) == "clean")
    pc = [r for r in rows if _st(r) == "scoreable" and _cl(r) == "clean"]
    out_p = OUT / "pilot.json"
    Pj = json.loads(out_p.read_text()) if out_p.exists() else {}
    Pj.update(label="PILOT (pilot seeds only; a machine check: fixes no rule, threshold or criterion)",
              n_frozen=len(rows), counts=counts(rows), freeze_secs=[r.get("secs") for r in rows])
    if not pc:
        Pj["runs"] = {}
        out_p.write_text(json.dumps(_jsonable(Pj), indent=1))
        print("no scoreable clean pilot seed", flush=True)
        return
    fs = seed_record(rows, pc[0]["seed"])
    Pj["seed"] = fs["seed"]
    Pj.setdefault("runs", {})
    for l2 in (LOG2_EXPONENT, LOG2_SCORED):
        key = f"{l2:g}"
        if key in Pj["runs"]:
            continue
        memory_gate(f"pilot run {fs['seed']} {key}")
        t0 = time.time()
        rr = run_one(fs, l2)
        rr["secs_run"] = round(time.time() - t0, 1)
        P2A.rss_guard()
        memory_gate(f"pilot observe {fs['seed']} {key}")
        t0 = time.time()
        ob, S = observe_one(fs, l2, rr)
        ob["secs_observe"] = round(time.time() - t0, 1)
        del S
        rr.pop("state_tc", None)
        Pj["runs"][key] = {**rr, "observed": ob, "peak_rss_gb": round(_peak_rss_gb(), 3)}
        out_p.write_text(json.dumps(_jsonable(Pj), indent=1))
        print(json.dumps(_jsonable({"log2rho": l2, "t_c": rr["t_c"], "status": rr["forecast"]["status"],
                                    "t_obs": ob["t_obs"], "s_obs_over_sF": (ob["s_obs"] or float("nan")) / fs["s_F"],
                                    "secs": [rr["secs_run"], ob["secs_observe"]], "rss": _peak_rss_gb()})), flush=True)
        P2A.rss_guard()


# ------------------------------------------------------------------------------------------ registration manifest
FROZEN_DATA = ("results/designs/phase2a_perseed_design.md", "results/phase2a_ps_registration.md",
               "results/phase2a_ps/seed_scan.json", "results/phase2a_ps/frozen.json",
               "results/phase2a_ps/frozen_parts.jsonl", "results/phase2a_ps/pilot.json",
               "results/phase2a_ps/pilot_parts.jsonl", "tests/test_phase2a_ps.py", "tests/test_causal_forecast.py",
               "results/simplicity_bias_v2/frozen.json", "results/phase2a/frozen.json", "results/phase2a_posthoc.json",
               "results/sb_fold/branches.json",
               "results/sb_fold/branch_M_fwd.npz", "results/sb_fold/branch_M_bwd.npz",
               "results/sb_fold/branch_S_fwd.npz", "results/sb_fold/branch_S_bwd.npz",
               "results/designs/phase2a_explore/p2a_explore_g.py", "results/designs/phase2a_explore/p2a_explore_h.py",
               "results/designs/phase2a_explore/p2a_explore_h.jsonl")


def code_closure(start=("phase2a_ps",)):
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


def stamp():
    """registration_stamp.txt: the registration commit (the last commit touching the manifest) and the SHA-256 of the
    registration file and of the manifest."""
    _assert_committed(OUT / "registration.sha256")
    h = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(OUT / "registration.sha256")], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()
    txt = (f"Phase 2A-PS registration\nregistration commit: {h}\n"
           f"sha256 results/phase2a_ps_registration.md: {_sha(REGISTRATION_MD)}\n"
           f"sha256 results/phase2a_ps/registration.sha256: {_sha(OUT / 'registration.sha256')}\n")
    (OUT / "registration_stamp.txt").write_text(txt)
    print(txt)


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


def _frozen():
    fr = json.loads((OUT / "frozen.json").read_text())
    assert fr["frozen_parts_sha256"] == _sha(OUT / "frozen_parts.jsonl")
    rows = _rows(OUT / "frozen_parts.jsonl")
    assert hashlib.sha256(class_listing(rows).encode()).hexdigest() == fr["class_listing_sha256"]
    return fr, rows


def run():
    """After the registration commit and its timestamp: every planned (seed, rate) to t_c with both forecasts
    (resumable per run)."""
    assert_registration()
    _setup()
    fr, rows = _frozen()
    if not fr["gate"]["pass"]:
        print("gate failed: nothing is trained (UNRESOLVED (gate))", flush=True)
        return
    rf = OUT / "runs.jsonl"
    done = {(r["seed"], r["log2rho"]) for r in _rows(rf)}
    for p in fr["plan"]:
        if (p["seed"], float(p["log2rho"])) in done:
            continue
        fs = seed_record(rows, p["seed"])
        memory_gate(f"run {p['seed']} {p['log2rho']:g}")
        t0 = time.time()
        rec = run_one(fs, p["log2rho"])
        rec["secs"] = round(time.time() - t0, 1)
        _append_write(rf, rec)
        print(json.dumps({"seed": p["seed"], "class": p["class"], "log2rho": p["log2rho"], "t_c": rec["t_c"],
                          "status": rec["forecast"]["status"], "secs": rec["secs"]}), flush=True)
        P2A.rss_guard()


def finalize():
    """forecasts.sha256 over runs.jsonl: every planned run exactly once, no observed key.  COMMIT both before observe."""
    fr, _ = _frozen()
    runs = _rows(OUT / "runs.jsonl")
    assert sorted((r["seed"], r["log2rho"]) for r in runs) == sorted((p["seed"], float(p["log2rho"]))
                                                                    for p in fr["plan"]), "every planned run once"
    assert not (FORBIDDEN & P2A._keys(runs)), "an observed quantity in a runs row"
    line = f"{_sha(OUT / 'runs.jsonl')}  runs.jsonl"
    (OUT / "forecasts.sha256").write_text(line + "\n")
    print(line)


def _assert_forecasts():
    h, f = (OUT / "forecasts.sha256").read_text().split()
    assert _sha(OUT / f) == h, "forecast hash mismatch"
    _assert_committed(OUT / f)
    _assert_committed(OUT / "forecasts.sha256")


def observe():
    assert_registration()
    _assert_forecasts()
    _setup()
    fr, rows = _frozen()
    of = OUT / "observed.jsonl"
    done = {(r["seed"], r["log2rho"]) for r in _rows(of)}
    for rr in _rows(OUT / "runs.jsonl"):
        if (rr["seed"], rr["log2rho"]) in done:
            continue
        fs = seed_record(rows, rr["seed"])
        memory_gate(f"observe {rr['seed']} {rr['log2rho']:g}")
        t0 = time.time()
        ob, S = observe_one(fs, rr["log2rho"], rr)
        ob["secs"] = round(time.time() - t0, 1)
        del S
        _append_write(of, ob)
        print(json.dumps({"seed": rr["seed"], "log2rho": rr["log2rho"], "t_obs": ob["t_obs"], "s_obs": ob["s_obs"],
                          "secs": ob["secs"]}), flush=True)
        P2A.rss_guard()


def posthoc_coef():
    """2A's POST HOC theory-backed fit r = A ε^{2/3} + C ε ln(1/ε) (ε = ε̂_F, all 27 rates): DESCRIPTIVE only."""
    J = json.loads(P2A_POSTHOC.read_text())
    c = J["fits_by_eps_source"]["eps_fc"]["fits"]["KS_eps_ln_free_A"]["coef"]
    return float(c["e23"]), float(c["eL"])


def descriptive(rows_scored):
    """DESCRIPTIVE (never a verdict): 2A's POST HOC correction per clean 2⁻¹⁶ seed; mixed seeds; per-class crossing
    counts and falsifier flags at every run."""
    A, C = posthoc_coef()
    out = {"posthoc_label": "POST HOC: 2A's fitted correction r = A eps^(2/3) + C eps ln(1/eps) (phase2a_posthoc.json, "
                            "eps_fc, KS_eps_ln_free_A); descriptive only",
           "posthoc_A": A, "posthoc_C": C, "per_seed": []}
    for r in rows_scored:
        e = r.get("eps_fc")
        rp = A * e ** (2 / 3) + C * e * math.log(1 / e) if e and e > 0 else None
        out["per_seed"].append({"seed": r["seed"], "class": r["class"], "log2rho": r["log2rho"], "r_obs": r["r_obs"],
                                "r_fc": r["r_fc"], "r_posthoc": rp,
                                "ratio_obs_posthoc": (r["r_obs"] / rp) if (rp and r["r_obs"] is not None) else None,
                                "falsifier_flag": bool(r["s_obs"] is not None and r["s_obs"] < r["s_F"]
                                                       and r["s_obs"] <= F_FALSIFIER * r["s_star"]),
                                "crossed": r["t_obs"] is not None})
    by = {}
    for r in rows_scored:
        k = f"{r['class']} {r['log2rho']:g}"
        b = by.setdefault(k, {"n": 0, "n_crossed": 0, "n_in_F_window": 0, "n_falsifier_flag": 0})
        b["n"] += 1
        b["n_crossed"] += int(r["t_obs"] is not None)
        b["n_in_F_window"] += int(r["s_obs"] is not None and r["s_F"] <= r["s_obs"] <= F_HI * r["s_F"])
        b["n_falsifier_flag"] += int(r["s_obs"] is not None and r["s_obs"] < r["s_F"]
                                     and r["s_obs"] <= F_FALSIFIER * r["s_star"])
    out["by_class_rate"] = by
    return out


def score():
    assert_registration()
    _assert_forecasts()
    fr, rows = _frozen()
    runs = _rows(OUT / "runs.jsonl")
    obs = {(o["seed"], o["log2rho"]): o for o in _rows(OUT / "observed.jsonl")}
    T = [run_row(seed_record(rows, r["seed"]), r, obs.get((r["seed"], r["log2rho"]), {})) for r in runs]
    ST = score_tables(T, fr["gate"]["pass"])
    out = {"counts_PROMINENT": fr["counts"], "gate": fr["gate"], "outcome": ST["outcome"],
           "verdicts": ST["verdicts"], "secondary_verdicts": ST.get("secondary_verdicts"),
           "criteria": ST["criteria"], "secondary": ST.get("secondary"), "validity": ST["validity"],
           "seed_exponents": ST.get("seed_exponents"), "rows": T, "DESCRIPTIVE": descriptive(T),
           "DESCRIPTIVE_split_by_mc_agreement": descriptive_split(T) if fr["gate"]["pass"] else None}
    (OUT / "scores.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("counts_PROMINENT", "gate", "verdicts", "secondary_verdicts",
                                                    "outcome")}), indent=1))


def main(argv):
    cmd = argv[1]
    if cmd == "gate":
        f, w = memory_gate(" ".join(argv[2:]) or "gate")
        print(f"memory gate OK: free {f}% swap_free {w} MB disk_free {shutil.disk_usage(ROOT).free / 1024 ** 3:.1f} GB")
        return
    fns = {"scan": scan, "pilot": pilot, "freeze": freeze, "summary": summary, "reevaluate": reevaluate, "manifest": manifest, "stamp": stamp,
           "run": run, "finalize": finalize, "observe": observe, "score": score}
    if cmd not in fns:
        raise SystemExit(f"unknown command {cmd}")
    fns[cmd]()


if __name__ == "__main__":
    main(sys.argv)
