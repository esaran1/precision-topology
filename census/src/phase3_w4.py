"""PHASE 3 (registered): the lag law at width 4 on the asymmetric windows (arms Q, S, T4; separate verdicts).

Approved page (author, 2026-10-06; commit bd2c48d): results/designs/phase3_width4_asym_design.md.  Registration text:
results/phase3_registration.md.  Tests of every decision rule: tests/test_phase3_w4.py.  Exploratory producers:
results/designs/phase3_explore/ (w4core.py, whose numerics this module promotes; p3_activity.py, p3_budget.py).

Setting (W2-A's, at width 4).  a = 1.30, Δ = 0.4: I = [−0.8, 0.8] class 0, O = [−2.0, −1.2] ∪ [1.2, 2.4] class 1.
N(x) = Σ₁⁴ v_i f(α_i x + β_i) + b, f(t) = t + a·sin t.  Layout: hidden z = (α₁..α₄, β₁..β₄, b), output v = (v₁..v₄),
s = ‖v‖₁.  Each seed's own 400-point sample asym_register.training_set(seed); mean BCE with logits; float64 numpy,
analytic gradient and Hessian.  G₊ = min_O φ − max_I φ (φ = N − b): the exact-extrema enclosure (width2_geometry's
algorithm with K units), screened by a dense grid whose value bounds G₊ from above.  Observation and hold: placed iff
the enclosure's lower end > 0.  Branch points (switches, own-path switch, forecasts): placed iff the midpoint > 0.
s₀ = 0.5·s_pop2 = 0.2225397 (T2-1's validated bracket, asym_scores.json); held output v₀ = s₀·(0.1, 0.2, 0.3, 0.4).

Arms (hold: v fixed at v₀, hidden GD at lr 1.0 for W steps; release: full-batch SGD, η = 0.03 on z, ρη on v):
  Q   HEADLINE, a MECHANISM test: the four-function copy (type 1111:++--), branch-point start θ*_Q,pop(s₀); seeds
      7,430,000-119; gate ≥ 96/120 on Q and no run with G > 0 at the start or any hold step.
  T4  CONTROL: W2-A's T embedded (3+1, type 31:+-), random hold (z ~ U(−1, 1)⁹ from default_rng(seed)); seeds
      7,432,000-709; gate ≥ 60 runs on T4 at release, regardless of G in the hold.
  S   SIGN TEST: the three-function copy (type 121:+-+, κ₀ < 0), branch-point start θ*_S,pop(s₀); seeds 7,431,000-119;
      gate ≥ 96/120 on S and no run with G > 0 in the hold.
Release classification (W2-A's Newton rule + the partition + the activity test): damped Newton at v₀ from the release
state, accepted iff max|∇| < 1e−8 and H_zz PD; on copy c at winding k ∈ ℤ⁴ iff the Newton point is within 1e−6 (sup) of
shift(θ*_own,c(s₀), v₀, k), the release state within 1e−3 of it, the Newton point's coincidence partition equals the
copy's, and every unit is ACTIVE (|v_i| ≥ θ·s, θ = 1e−8; an inactive unit gives 'neither (inactive unit)').  The
winding k is fixed at release; κ_k = κ₀ + a·k (exact); the table is ‖k‖₁ ≤ 3 (129 windings).

Budgets: B = min(⌈1.5·t*⌉ + 3000, 10⁶), t* = (1/ρη)·∫ ds/D along the copy's frozen adiabatic path, D = −sign(v)·∇_vL.
A seed whose ⌈1.5·t*⌉ + 3000 exceeds 10⁶ is counted and NOT trained.  Hard per-run caps: 10⁶ steps and RSS 1 GB.

Strictly causal forecasts (f = 0.95): t_c = the first t ≥ 1 with s_t ≥ 0.95·(the copy's frozen switch), a stopping time.
The forecaster (forecast_w4) reads GuardedArray views of the v path rows < t_c and the release row of z only;
causal_forecast's frozen quad rule extrapolates log s and EACH UNIT'S SHARE (clipped at 0, renormalised); the own-path
switch and R4 run along [visible, extrapolated].  A NaN recomputation (rows ≥ t_c set to NaN) must be identical.
Forecasts are committed before observation.

    python -m src.phase3_w4 gate TAG         # the memory gate as a separate logged step (before every job)
    python -m src.phase3_w4 scan             # the seed ranges are unused          -> results/phase3/seed_scan.json
    python -m src.phase3_w4 landscape        # population copies and switches      -> results/phase3/landscape.json
    python -m src.phase3_w4 pilot            # pilot seeds only: ρ per arm, τ, V6  -> pilot_frozen_parts.jsonl, pilot_parts.jsonl, pilot.json
    python -m src.phase3_w4 freeze ARM       # registered seeds, no training       -> frozen_parts.jsonl (resumable)
    python -m src.phase3_w4 summary          # frozen.json (gates, budgets, the budget condition, the collapse table)
    python -m src.phase3_w4 manifest         # registration.sha256 (the registration commit); then: stamp
    python -m src.phase3_w4 run ARM          # AFTER the push and the OpenTimestamps proof: train to t_c, forecasts
    python -m src.phase3_w4 finalize         # forecasts.sha256 (commit runs + hash before observe)
    python -m src.phase3_w4 observe ARM; python -m src.phase3_w4 score
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import causal_forecast as C

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase3"
REGISTRATION_MD = RESULTS / "phase3_registration.md"
DESIGN_MD = RESULTS / "designs" / "phase3_width4_asym_design.md"
ASYM_SCORES = RESULTS / "asym_scores.json"

# ------------------------------------------------------------------------------------------ setting
K = 4
A_ACT, DELTA = 1.30, 0.4
INNER = (-0.8, 0.8)
OUTER = ((-2.0, -1.2), (1.2, 2.0 + DELTA))
XI_DENSE = np.linspace(INNER[0], INNER[1], 801)
XO_DENSE = np.concatenate([np.linspace(*OUTER[0], 401), np.linspace(*OUTER[1], 401)])
TWO_PI = 2 * math.pi
S_POP2_EXPECTED = 0.44507940623559955          # W2-A's √(s_lo·s_hi) of T2-1's validated bracket
S0_PAGE = 0.2225397
SHARES = (0.1, 0.2, 0.3, 0.4)
ETA = 0.03

Q, S, T4 = "Q", "S", "T4"
ARMS = (Q, S, T4)
HEADLINE = Q
ROLE = {Q: "headline (four-function copy, branch-point start; a MECHANISM test)",
        S: "sign test (three-function copy, κ₀ < 0, branch-point start)",
        T4: "control (W2-A's T embedded 3+1, random hold)"}
SEEDS = {Q: tuple(range(7_430_000, 7_430_120)), S: tuple(range(7_431_000, 7_431_120)),
         T4: tuple(range(7_432_000, 7_432_710))}
SEEDS_Q, SEEDS_S, SEEDS_T4 = SEEDS[Q], SEEDS[S], SEEDS[T4]
PILOT_SEEDS = tuple(range(7_439_000, 7_439_010))          # Q and S (branch-point starts; 10 seeds each)
PILOT_SEEDS_T4 = tuple(range(7_439_100, 7_439_300))       # T4: random holds drawn in order until 10 land on T4
PILOT_N_ON_COPY = 10
PILOT_RANGES = {Q: PILOT_SEEDS, S: PILOT_SEEDS, T4: PILOT_SEEDS_T4}
EXPLORATION_SEEDS = range(7_410_000, 7_410_060)           # never registered
START = {Q: "branch point", S: "branch point", T4: "random hold"}
COPY_OF = {Q: "Q", S: "S", T4: "T4"}
POINT_COPIES = {Q: (), S: (), T4: ("Q", "S")}             # T4 seeds: Q and S own points (landing counts, descriptive)
RHO_START = {Q: 2.0 ** -13, S: 2.0 ** -12, T4: 2.0 ** -10}  # the page's table (‡); the pilot rule halves from here

# population copy points at v₀ (exploratory class points, 6 decimals; results/designs/phase3_explore/README "Labels")
POP_START = {"Q": (-1.648094, -3.687251, 3.856339, 1.714031, -1.46324, 1.887936, 1.386163, -1.683263, -0.082595),
             "S": (-1.679706, 1.690561, -1.679706, 3.805367, -1.451355, -1.690496, -1.451355, 1.380376, -0.011128),
             "T4": (-1.707199, 1.670937, 1.670937, 1.670937, -1.450349, -1.692048, -1.692048, -1.692048, 0.296007)}
COPY_TYPE = {"Q": "1111:++--", "S": "121:+-+", "T4": "31:+-"}
COPY_PARTITION = {"Q": ((0,), (1,), (2,), (3,)), "S": ((0, 2), (1,), (3,)), "T4": ((0,), (1, 2, 3))}
POP_SWITCH_PAGE = {"Q": 0.482, "S": 0.440, "T4": 0.762}
POP_SWITCH_REL = 5e-3
POP_KAPPA_SIGN_PAGE = {"Q": 1, "S": -1, "T4": 1}        # the page: Q κ₀ +0.017, S κ₀ −0.012; T4 (W2-A's T) > 0

# numerics (W2-A's)
HOLD_LR, W_MIN, W_RELAX = 1.0, 4000, 25.0
# D1, APPROVED by the author 2026-10-06 (T4 only): T4's random hold uses W ≥ 8000, the exploration's hold length on
# which the page's T4 gate power rests (53/400 at 8000 vs 46/400 at W2-A's 4000 on exploration seeds 7,410,000-019)
W_FLOOR = {"Q": W_MIN, "S": W_MIN, "T4": 8000}
NEWTON_GTOL, NEWTON_ITER_TOL, ON_TOL, STATE_TOL = 1e-8, 1e-12, 1e-6, 1e-3
DUP_TOL = 1e-6
FOLLOW_FRAC = 0.8
S_HI_FRAC = 1.6
S_MAX_POP = 30.0
ADIAB_HS = 0.005
HALVING_REL = 1e-6
DECIDE_REL = 1e-3
DG_ONESIDED_MAX = 1e-5
WINDING_L1_MAX = 3
SDOT_WIN = 100
BRANCH_JUMP = 0.2
DIVERGED = 1.0
RADIUS_EVERY = 10

# activity test (Phase 3 only; after the 2A-PS post hoc diagnosis 2f0fb33)
THETA_ACTIVE = 1e-8

# budgets and caps
BUDGET_MULT, BUDGET_ADD, STEP_CAP = 1.5, 3000, 1_000_000
RSS_RUN_CAP = 1024 ** 3                       # per run (current RSS of the process, checked every RSS_EVERY steps)
RSS_EVERY = 20_000
RSS_LIMIT = 3 * 1024 ** 3                     # the machine's process limit (peak RSS)
DISK_MIN_BYTES = 20 * 1024 ** 3
BUDGET_CONDITION_MAX_OVER = 24                # the author's condition: > 24 over-cap seeds in Q or S → STOP

# the causal forecast (Phase 1A's frozen rule)
F_CUT = 0.95
FAMILY, WINDOW_FRAC, MIN_WINDOW, HORIZON = "quad", 0.05, 50, 1.6

# criteria (1C), gates, validity
WITHIN_MIN_FRAC = 0.80
C3_BAND = (0.90, 1.10)
MIN_ABS_LAG = 6                               # C3 and S use runs with |lag_fc| ≥ 6 only
S_MIN_FRAC = 0.80
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 7_430_000, (2.5, 97.5)
CUTOFF_BEFORE_MIN_FRAC = 0.90
CRITERIA = {Q: ("C1", "C2", "C3", "C4"), S: ("C1", "C2", "C3", "C4", "S"), T4: ("C1", "C2", "C3", "C4")}
GATE_MIN = {Q: 96, S: 96, T4: 60}
GATE_HOLD_CONDITION = {Q: True, S: True, T4: False}
MIN_SCORED = 60
TSW_MIN_FRAC = 0.90
REGIME_MAX, REGIME_MIN_FRAC = 0.5, 0.80
LAG_MIN_STEPS = 10.0
KC_Q90_MAX = 0.1
CHI_REL_TOL = 0.30
CHI_PATH_Q90_MAX = 0.25
PILOT_CHI_MAX, PILOT_MAX_HALVINGS = 0.1, 3
TAU_MULT, TAU_ROUND = 1.5, 5

FORBIDDEN = {"t_obs", "s_obs", "crossed", "step_obs", "placed", "r_obs", "lag_obs", "follows_branch", "t_follow"}


# ------------------------------------------------------------------------------------------ the network
def _f(t):
    st = np.sin(t)
    return t + A_ACT * st, 1 + A_ACT * np.cos(t), -A_ACT * st


def _sig(n):
    return 0.5 * (1 + np.tanh(0.5 * n))


def n_units(z):
    return (len(z) - 1) // 2


def loss(z, v, x, y):
    k = n_units(z)
    t = np.outer(x, z[:k]) + z[k:2 * k]
    n = (t + A_ACT * np.sin(t)) @ v + z[-1]
    return float(np.mean(np.logaddexp(0, n) - y * n))


def _parts(z, v, x):
    k = n_units(z)
    t = np.outer(x, z[:k]) + z[k:2 * k]
    f, d, s = _f(t)
    return k, f, d, s, f @ v + z[-1]


def grad(z, v, x, y):
    """(∇_z L, ∇_v L)."""
    k, f, d, s, n = _parts(z, v, x)
    r = (_sig(n) - y) / len(x)
    vd = d * v
    return np.concatenate([(r * x) @ vd, r @ vd, [r.sum()]]), r @ f


def gz(z, v, x, y):
    return grad(z, v, x, y)[0]


def hess(z, v, x, y):
    """The full Hessian in (z, v) order (3K + 1 coordinates)."""
    k, f, d, s, n = _parts(z, v, x)
    m = len(x)
    p = _sig(n)
    r = p - y
    w = p * (1 - p)
    vd = d * v
    J = np.concatenate([vd * x[:, None], vd, np.ones((m, 1)), f], 1)
    H = (J * w[:, None]).T @ J
    vs = s * v
    for i in range(k):
        ia, ib, iv = i, k + i, 2 * k + 1 + i
        H[ia, ia] += np.sum(r * vs[:, i] * x * x)
        H[ib, ib] += np.sum(r * vs[:, i])
        c = np.sum(r * vs[:, i] * x)
        H[ia, ib] += c
        H[ib, ia] += c
        c = np.sum(r * d[:, i] * x)
        H[ia, iv] += c
        H[iv, ia] += c
        c = np.sum(r * d[:, i])
        H[ib, iv] += c
        H[iv, ib] += c
    return H / m


def blocks(z, v, x, y):
    """(H_zz, H_zv)."""
    H = hess(z, v, x, y)
    nz = len(z)
    return H[:nz, :nz], H[:nz, nz:]


def newton(z0, v, x, y, maxit=100):
    """W2-A's damped Newton on ∇_z L(·, v) = 0 (backtracking on ‖∇‖), iterated to 1e−12.  Returns (z, max|∇|, λ_min of
    H_zz, accepted); accepted iff max|∇| < 1e−8 and H_zz is positive definite."""
    z = np.asarray(z0, float).copy()
    v = np.asarray(v, float)
    g = gz(z, v, x, y)
    for _ in range(maxit):
        if not np.all(np.isfinite(g)) or np.abs(g).max() < NEWTON_ITER_TOL:
            break
        H = blocks(z, v, x, y)[0]
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        t, moved = 1.0, False
        for _ in range(40):
            zn = z + t * d
            gn = gz(zn, v, x, y)
            if np.all(np.isfinite(gn)) and np.linalg.norm(gn) < (1 - 1e-4 * t) * np.linalg.norm(g):
                moved = True
                break
            t *= 0.5
        if not moved:
            break
        z, g = zn, gn
    H = blocks(z, v, x, y)[0]
    lam = float(np.linalg.eigvalsh(0.5 * (H + H.T)).min()) if np.all(np.isfinite(H)) else float("nan")
    gm = float(np.abs(g).max()) if np.all(np.isfinite(g)) else float("nan")
    return z, gm, lam, bool(np.isfinite(gm) and gm < NEWTON_GTOL and np.isfinite(lam) and lam > 0)


def dz_dv(z, v, x, y):
    Hzz, Hzv = blocks(z, v, x, y)
    return -np.linalg.solve(Hzz, Hzv)


def drive(z, v, x, y):
    """D = −sign(v)·∇_vL(z, v): ds/step = ρη·D under SGD at a branch point."""
    return float(-np.sign(v) @ grad(z, v, x, y)[1])


# ------------------------------------------------------------------------------------------ the gap
_ACT = None


def act():
    global _ACT
    if _ACT is None:
        from .width2_geometry import Act
        _ACT = Act("fa", A_ACT)
    return _ACT


def phi(xs, z, v):
    k = n_units(z)
    t = np.outer(np.asarray(xs, float), z[:k]) + z[k:2 * k]
    return (t + A_ACT * np.sin(t)) @ np.asarray(v, float)


def dphi(xs, z, v):
    k = n_units(z)
    t = np.outer(np.asarray(xs, float), z[:k]) + z[k:2 * k]
    return (1 + A_ACT * np.cos(t)) @ (np.asarray(v, float) * z[:k])


def _m2(m, h, z, v):
    k = n_units(z)
    a = act()
    out = 0.0
    for i in range(k):
        al, be = z[i], z[k + i]
        out = out + abs(v[i]) * al ** 2 * a.d2u_bound(al * (m - h) + be, al * (m + h) + be)
    return out


def extrema(z, v, lo, hi, tol=1e-9, vtol=1e-13, n0=64, max_cells=400_000):
    """width2_geometry.extrema with K units: certified enclosures (min_lo, min_hi, max_lo, max_hi) of φ on [lo, hi]."""
    ends = phi(np.array([lo, hi]), z, v)
    mn_lo = mn_hi = float(ends.min())
    mx_lo = mx_hi = float(ends.max())
    edges = np.linspace(lo, hi, n0 + 1)
    m = 0.5 * (edges[:-1] + edges[1:])
    h = np.full(n0, 0.5 * (hi - lo) / n0)
    total = 0
    while len(m):
        total += len(m)
        if total > max_cells:
            raise RuntimeError("extrema: cell cap")
        d = dphi(m, z, v)
        M2 = _m2(m, h, z, v)
        mono = np.abs(d) > M2 * h
        if mono.any():
            e = phi(np.concatenate([m[mono] - h[mono], m[mono] + h[mono]]), z, v)
            mn_lo = min(mn_lo, float(e.min()))
            mn_hi = min(mn_hi, float(e.min()))
            mx_lo = max(mx_lo, float(e.max()))
            mx_hi = max(mx_hi, float(e.max()))
        m, h, d, M2 = m[~mono], h[~mono], d[~mono], M2[~mono]
        r_all = np.abs(d) * h + 0.5 * M2 * h ** 2
        small = (h < tol) | (r_all < vtol)
        if small.any():
            fv = phi(m[small], z, v)
            r = r_all[small]
            mn_lo = min(mn_lo, float((fv - r).min()))
            mn_hi = min(mn_hi, float(fv.min()))
            mx_lo = max(mx_lo, float(fv.max()))
            mx_hi = max(mx_hi, float((fv + r).max()))
        m, h = m[~small], h[~small]
        h = h / 2
        m = np.concatenate([m - h, m + h])
        h = np.concatenate([h, h])
    return mn_lo, mn_hi, mx_lo, mx_hi


def gap_enclosure(z, v):
    _, _, ixl, ixh = extrema(z, v, *INNER)
    oL, oR = extrema(z, v, *OUTER[0]), extrema(z, v, *OUTER[1])
    return float(min(oL[0], oR[0]) - ixh), float(min(oL[1], oR[1]) - ixl)


def gap_mid(z, v):
    lo, hi = gap_enclosure(z, v)
    return 0.5 * (lo + hi)


def gap_dense(z, v):
    return float(phi(XO_DENSE, z, v).min() - phi(XI_DENSE, z, v).max())


def placed_state(z, v):
    """Observation and hold rule: (placed, undecided).  Placed iff the enclosure's lower end > 0 (a dense value ≤ 0
    means G₊ ≤ 0); undecided iff the enclosure straddles 0 (counted, not placed)."""
    if gap_dense(z, v) <= 0:
        return False, False
    lo, hi = gap_enclosure(z, v)
    return bool(lo > 0), bool(lo <= 0 < hi)


def branch_placed(z, v):
    """Branch-point and forecast rule: the enclosure midpoint > 0 (screened by the dense value)."""
    return bool(gap_dense(z, v) > 0 and gap_mid(z, v) > 0)


# ------------------------------------------------------------------------------------------ symmetries, partitions
def shift(z, v, k):
    """The 2π winding copy: β_i + 2πk_i, b − 2π Σ k_i v_i (the same function up to the bias, the same loss)."""
    z = np.asarray(z, float).copy()
    n = n_units(z)
    k = np.asarray(k, float)
    z[n:2 * n] += TWO_PI * k
    z[-1] -= TWO_PI * float(k @ np.asarray(v, float))
    return z


def windings_to(z, zc):
    """The winding k with shift(zc, v, k) nearest z in the β coordinates."""
    n = n_units(z)
    return tuple(int(round((z[n + i] - zc[n + i]) / TWO_PI)) for i in range(n))


def groups(z, tol=DUP_TOL):
    """The coincidence partition: groups of units with α equal and β equal modulo 2π (tolerance 1e−6), each a tuple of
    unit indices in increasing order, groups ordered by their first unit."""
    n = n_units(z)
    red = (z[n:2 * n] + math.pi) % TWO_PI - math.pi
    left, out = list(range(n)), []
    while left:
        i = left.pop(0)
        g = [i]
        for j in list(left):
            db = red[i] - red[j]
            db -= TWO_PI * round(db / TWO_PI)
            if abs(z[i] - z[j]) <= tol * max(1.0, abs(z[i])) and abs(db) <= tol:
                g.append(j)
                left.remove(j)
        out.append(tuple(g))
    return tuple(out)


def partition(z):
    return tuple(sorted(groups(z)))


def btype(z, v):
    """Branch type: group sizes ordered by total share (descending) and, per group, the sign of α ('+', '-', '0' if
    |α| < 1e−6); e.g. '1111:++--' (four distinct units), '31:+-' (three units coincide), '4:+' (all coincide)."""
    gs = groups(z)
    sh = [sum(abs(v[i]) for i in g) for g in gs]
    order = np.argsort(-np.asarray(sh), kind="stable")
    sizes = "".join(str(len(gs[o])) for o in order)
    sg = "".join("0" if abs(z[gs[o][0]]) < 1e-6 else ("+" if z[gs[o][0]] > 0 else "-") for o in order)
    return f"{sizes}:{sg}"


def split_basis(z, v):
    """Orthonormal basis of the split subspace: per coinciding group, the 2(m − 1) first-order function-preserving
    directions δ(α_i, β_i) = v_j e, δ(α_j, β_j) = −v_i e (pairs (first unit, j))."""
    n = n_units(z)
    cols = []
    for g in groups(z):
        i = g[0]
        for j in g[1:]:
            for off in (0, n):
                c = np.zeros(2 * n + 1)
                c[off + i], c[off + j] = v[j], -v[i]
                cols.append(c)
    if not cols:
        return None
    return np.linalg.qr(np.column_stack(cols))[0]


def reduced_lam(H, z, v):
    """(λ_min on the complement of the split subspace, λ_min of the split block or None).  No coinciding group: the
    full H_zz's λ_min and None."""
    H = 0.5 * (np.asarray(H, float) + np.asarray(H, float).T)
    B = split_basis(z, v)
    if B is None:
        return float(np.linalg.eigvalsh(H).min()), None
    n, m = H.shape[0], B.shape[1]
    Qc = np.linalg.qr(np.column_stack([B, np.eye(n)]))[0][:, m:]
    return float(np.linalg.eigvalsh(Qc.T @ H @ Qc).min()), float(np.linalg.eigvalsh(B.T @ H @ B).min())


# ------------------------------------------------------------------------------------------ the activity test
def active_units(v, theta=THETA_ACTIVE):
    """Unit i is ACTIVE iff |v_i| ≥ θ·s, s = ‖v‖₁ (inclusive; θ = 1e−8)."""
    v = np.asarray(v, float)
    s = float(np.abs(v).sum())
    if s == 0.0:
        return []
    return [i for i in range(len(v)) if abs(v[i]) >= theta * s]


def all_units_active(v, theta=THETA_ACTIVE):
    return len(active_units(v, theta)) == len(np.asarray(v))


def scale_direction(v, theta=THETA_ACTIVE):
    """â = sign(v_A)/√|A| on the active set A, 0 elsewhere (any scale direction is built from active units only)."""
    v = np.asarray(v, float)
    a = np.zeros_like(v)
    A = active_units(v, theta)
    if A:
        a[A] = np.sign(v[A]) / np.sqrt(len(A))
    return a


def scale_only_step(v, g, eta, rho, a_hat):
    """The 2A-PS scale-only update, for the activity test's constructed cases: v − η[ρ(â·g)â + (I − ââᵀ)g]."""
    v, g, a = (np.asarray(t, float) for t in (v, g, a_hat))
    par = (a @ g) * a
    return v - eta * (rho * par + (g - par))


# ------------------------------------------------------------------------------------------ κ (P = I, total derivative)
WINDING_TABLE = tuple(k for k in itertools.product(range(-WINDING_L1_MAX, WINDING_L1_MAX + 1), repeat=K)
                      if sum(abs(i) for i in k) <= WINDING_L1_MAX)


def in_winding_table(k):
    """The table: ‖k‖₁ ≤ 3 (129 windings for K = 4)."""
    return k is not None and len(k) == K and sum(abs(int(i)) for i in k) <= WINDING_L1_MAX


def kappa_winding(kap0, coef, k):
    """κ_k = κ₀ + a·k (exact: a winding leaves H, ∇_zG and ∂_vG unchanged and moves the tangent by −2π(k·v′) in b)."""
    return float(kap0 + sum(float(c) * int(i) for c, i in zip(coef, k)))


def grad_gap(z, v, vprime, h=1e-6):
    """∇_zG (central differences of the enclosure midpoint in α, β; b does not enter G), ∂_vG·v′ (central, along v′),
    and the largest forward-backward disagreement over the 2K + 1 derivatives relative to their sup norm."""
    n = n_units(z)
    g0 = gap_mid(z, v)
    out, dis = np.zeros(2 * n + 1), []
    for i in range(2 * n):
        e = np.zeros(2 * n + 1)
        e[i] = h
        gp, gm = gap_mid(z + e, v), gap_mid(z - e, v)
        out[i] = (gp - gm) / (2 * h)
        dis.append(abs((gp - g0) - (g0 - gm)) / h)
    vp = np.asarray(vprime, float)
    gp, gm = gap_mid(z, np.asarray(v) + h * vp), gap_mid(z, np.asarray(v) - h * vp)
    dv = (gp - gm) / (2 * h)
    dis.append(abs((gp - g0) - (g0 - gm)) / h)
    sc = max(float(np.abs(out).max()), abs(dv), 1e-300)
    return out, float(dv), float(max(dis) / sc)


def kappa_at(z, v, vprime, x, y):
    """κ₀ = λ_min·[∇_zG·H⁻¹θ*′]/[∇_zG·θ*′ + ∂_vG·v′] (reduced λ_min: every group's split directions excluded) and the
    winding coefficients a_i = −2π·λ_min·(H⁻¹∇_zG)_b·v′_i/[dG/ds] (κ_k = κ₀ + a·k)."""
    Hzz, Hzv = blocks(z, v, x, y)
    vp = np.asarray(vprime, float)
    tan = -np.linalg.solve(Hzz, Hzv @ vp)
    lam, lam_split = reduced_lam(Hzz, z, v)
    dGz, dGv, dis = grad_gap(z, v, vp)
    u = np.linalg.solve(Hzz, dGz)
    den = float(dGz @ tan + dGv)
    kap0 = float(lam * (u @ tan) / den)
    coef = [float(-TWO_PI * lam * u[-1] * vp[i] / den) for i in range(len(vp))]
    return {"kappa0": kap0, "kappa_winding_coef": coef, "lam_min_switch": lam, "lam_split_switch": lam_split,
            "dGz_dot_tangent": float(dGz @ tan), "dGv_along": dGv, "dG_ds": den, "gradG_onesided_rel": dis,
            "gradG_ok": bool(dis <= DG_ONESIDED_MAX), "vprime": vp.tolist()}


# ------------------------------------------------------------------------------------------ continuation (freeze)
def flow_rhs(z, v, x, y):
    """dv/ds on the adiabatic reference: g/(sign(v)·g), g = −∇_vL(z*(v), v); None at a turning point."""
    g = -grad(z, v, x, y)[1]
    ds = float(np.sign(v) @ g)
    return (g / ds) if ds > 0 else None


def rk4_step(z, v, hs, x, y):
    """One RK4 step of length hs in s along the adiabatic reference, z* by tangent-predicted damped Newton."""
    def stage(zb, vb, vn):
        zz, _, _, ok = newton(zb + dz_dv(zb, vb, x, y) @ (vn - vb), vn, x, y)
        return zz, ok
    k1 = flow_rhs(z, v, x, y)
    if k1 is None:
        return z, v, False
    v2 = v + 0.5 * hs * k1
    z2, ok = stage(z, v, v2)
    k2 = flow_rhs(z2, v2, x, y) if ok else None
    if k2 is None:
        return z, v, False
    v3 = v + 0.5 * hs * k2
    z3, ok = stage(z, v, v3)
    k3 = flow_rhs(z3, v3, x, y) if ok else None
    if k3 is None:
        return z, v, False
    v4 = v + hs * k3
    z4, ok = stage(z, v, v4)
    k4 = flow_rhs(z4, v4, x, y) if ok else None
    if k4 is None:
        return z, v, False
    vn = v + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    zn, ok = stage(z, v, vn)
    return zn, vn, ok


def continue_adiabatic(z0, v0, x, y, s_max, hs_rel=ADIAB_HS, s_stop=None):
    """The adiabatic reference: RK4 in s with steps hs_rel·s.  Checks at every accepted point: Newton accepted, reduced
    λ_min > 0, the split block's λ_min > 0 (coinciding groups), the coincidence partition kept, s increasing (no turning
    point).  Stops at the first branch-gap sign change (bisected to 1e−12 relative) -> 'switch', or at exactly s_stop ->
    'stop'.  Also the drive integral ∫ ds/D (trapezoid on the accepted points, the last piece to the switch)."""
    z, v = np.asarray(z0, float), np.asarray(v0, float)
    g0 = gap_mid(z, v)
    part0 = partition(z)
    min_lam, min_split, part_ok, n = float("inf"), float("inf"), True, 0
    s_prev, d_prev = float(np.abs(v).sum()), drive(z, v, x, y)
    integral = 0.0
    while True:
        s = float(np.abs(v).sum())
        if s_stop is not None and s >= s_stop * (1 - 1e-14):
            return {"status": "stop", "z": z, "v": v, "min_lam_reduced": min_lam, "min_lam_split": min_split,
                    "partition_kept": part_ok, "n_steps": n}
        if s >= s_max:
            return {"status": "none", "s_last": s}
        hs = hs_rel * s
        if s_stop is not None and s + hs >= s_stop:
            hs = s_stop - s
        zn, vn, ok = rk4_step(z, v, hs, x, y)
        if not ok:
            return {"status": "failed", "s_last": s}
        lr, ls = reduced_lam(blocks(zn, vn, x, y)[0], zn, vn)
        min_lam, n = min(min_lam, lr), n + 1
        if ls is not None:
            min_split = min(min_split, ls)
        part_ok = part_ok and partition(zn) == part0
        gn = gap_mid(zn, vn)
        if (gn > 0) != (g0 > 0) and s_stop is None:
            lo, hi = 0.0, hs
            while hi - lo > 1e-12 * s:
                mid = 0.5 * (lo + hi)
                zm, vm, okm = rk4_step(z, v, mid, x, y)
                if okm and (gap_mid(zm, vm) > 0) == (g0 > 0):
                    lo = mid
                else:
                    hi = mid
            zs, vs, oks = rk4_step(z, v, 0.5 * (lo + hi), x, y)
            ssw = float(np.abs(vs).sum())
            d_sw = drive(zs, vs, x, y)
            integral += 0.5 * (ssw - s_prev) * (1 / d_prev + 1 / d_sw)
            return {"status": "switch", "s_switch": ssw, "z_switch": zs, "v_switch": vs,
                    "vprime": flow_rhs(zs, vs, x, y), "newton_ok_switch": oks, "min_lam_reduced": min_lam,
                    "min_lam_split": min_split, "partition_kept": bool(part_ok and partition(zs) == part0),
                    "n_steps": n, "to_placed": bool(g0 <= 0), "G_start": g0, "drive_integral": float(integral),
                    "shares_switch": [float(t) for t in vs / ssw]}
        if (gn > 0) != (g0 > 0):
            return {"status": "failed", "s_last": s, "note": "sign change before s_stop"}
        s_new = float(np.abs(vn).sum())
        d_new = drive(zn, vn, x, y)
        integral += 0.5 * (s_new - s_prev) * (1 / d_prev + 1 / d_new)
        s_prev, d_prev = s_new, d_new
        z, v = zn, vn


def _decided(zsw, vsw, x, y):
    ssw = float(np.abs(vsw).sum())
    zb, vb, okb = rk4_step(zsw, vsw, -DECIDE_REL * ssw, x, y)
    za, va, oka = rk4_step(zsw, vsw, DECIDE_REL * ssw, x, y)
    below, above = gap_enclosure(zb, vb), gap_enclosure(za, va)
    return below, above, bool(below[1] < 0 < above[0] and okb and oka)


def switch_of_copy(z_s0, v0, x, y, s_max):
    """Validated switch along the adiabatic reference: step halving agrees to 1e−6 relative; no turning point; reduced
    λ_min > 0 along it; coinciding groups: the split block PD along it and the partition kept throughout; G(s₀) < 0 and
    the switch goes to placed; decided at (1 ± 1e−3)·s_switch; ∇G one-sided agreement ≤ 1e−5; Newton accepted at the
    switch; λ_min(switch) > 0; κ finite; the drive integral finite and positive."""
    a = continue_adiabatic(z_s0, v0, x, y, s_max)
    b = continue_adiabatic(z_s0, v0, x, y, s_max, hs_rel=ADIAB_HS / 2)
    row = {"path": "adiabatic", "status": a["status"], "status_halved": b["status"]}
    if a["status"] != "switch" or b["status"] != "switch":
        return {**row, "valid": False, "note": f"no validated switch below {s_max:.4f}"}
    ssw = a["s_switch"]
    has_groups = len(partition(np.asarray(z_s0, float))) < K
    row.update(s_switch=ssw, s_switch_halved=b["s_switch"],
               halving_ok=bool(abs(ssw - b["s_switch"]) <= HALVING_REL * ssw), to_placed=a["to_placed"],
               G_start=a["G_start"], min_lam_reduced_path=a["min_lam_reduced"],
               min_lam_split_path=a["min_lam_split"] if has_groups else None, partition_kept=a["partition_kept"],
               v_switch=[float(t) for t in a["v_switch"]], z_switch=[float(t) for t in a["z_switch"]],
               shares_switch=a["shares_switch"], newton_ok_switch=a["newton_ok_switch"], n_steps=a["n_steps"],
               drive_integral=a["drive_integral"], drive_integral_halved=b["drive_integral"])
    below, above, dec = _decided(a["z_switch"], a["v_switch"], x, y)
    row.update(G_enc_below=list(below), G_enc_above=list(above), decided_ok=dec)
    k = kappa_at(a["z_switch"], a["v_switch"], a["vprime"], x, y)
    row.update(k)
    checks = [row["halving_ok"], row["to_placed"], row["min_lam_reduced_path"] > 0, row["partition_kept"],
              row["decided_ok"], row["gradG_ok"], row["newton_ok_switch"], k["lam_min_switch"] > 0,
              bool(np.isfinite(k["kappa0"])), bool(np.isfinite(a["drive_integral"]) and a["drive_integral"] > 0)]
    if has_groups:
        checks.append(bool(row["min_lam_split_path"] > 0))
    row["checks"] = [bool(c) for c in checks]
    row["valid"] = bool(all(checks))
    return row


def follow_point(z_s0, v0, s_sw, x, y):
    """The copy's branch point at s_f = 0.8·s_switch on its adiabatic reference, recomputed with the step halved
    (agreement 1e−6 sup in z and v), partition kept, with ∂z*/∂v there."""
    s_f = FOLLOW_FRAC * s_sw
    r1 = continue_adiabatic(z_s0, v0, x, y, 10 * s_sw, s_stop=s_f)
    r2 = continue_adiabatic(z_s0, v0, x, y, 10 * s_sw, hs_rel=ADIAB_HS / 2, s_stop=s_f)
    if r1["status"] != "stop" or r2["status"] != "stop":
        return {"s": s_f, "valid": False, "note": f"{r1['status']}/{r2['status']}"}
    zf, _, _, okn = newton(r1["z"], r1["v"], x, y)
    sup = float(max(np.abs(r1["z"] - r2["z"]).max(), np.abs(r1["v"] - r2["v"]).max()))
    return {"s": float(s_f), "v": [float(t) for t in r1["v"]], "z": [float(t) for t in zf],
            "dz_dv": dz_dv(zf, r1["v"], x, y).tolist(), "halving_sup": sup,
            "partition": [list(g) for g in partition(zf)],
            "valid": bool(okn and sup <= HALVING_REL and partition(zf) == partition(np.asarray(z_s0, float))
                          and r1["partition_kept"])}


# ------------------------------------------------------------------------------------------ helpers
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array_sha(*arrs):
    """SHA-256 of the little-endian float64 bytes of the arrays, concatenated."""
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a, dtype="<f8")).tobytes())
    return h.hexdigest()


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
    with open(p, "a") as fh:
        fh.write(json.dumps(_jsonable(row)) + "\n")


def _rows(p):
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()] if Path(p).exists() else []


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS


def current_rss():
    """The process's current resident set (bytes), from ps."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
        return int(out.strip()) * 1024
    except (OSError, ValueError):
        return 0


def rss_guard():
    if peak_rss() > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {peak_rss() / 1e9:.2f} GB > 3 GB")


def memory_gate(tag):
    """free ≥ 25% and swap free ≥ 500 MB (act_fold.check_memory; waits, re-checking every 60 s); disk free ≥ 20 GB
    (else STOP).  Every check logged to results/phase3/memory_gate.log."""
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


def _setup():
    try:
        os.nice(15)
    except OSError:
        pass


def s_pop2():
    d = json.loads(ASYM_SCORES.read_text())
    v = math.sqrt(d["s_lo"] * d["s_hi"])
    assert abs(v - S_POP2_EXPECTED) <= 1e-15, "asym_scores.json bracket differs"
    return v


def s0_value():
    return 0.5 * s_pop2()


def v_held(s0=None):
    s0 = s0_value() if s0 is None else s0
    return s0 * np.array(SHARES)


def population():
    from .asym_pilot import population as pop
    x, y = pop(DELTA)
    return np.asarray(x, float), np.asarray(y, float)


def own_sample(seed):
    from .asym_register import training_set
    x, y = training_set(int(seed))
    return np.asarray(x, float), np.asarray(y, float)


def random_hold_start(seed):
    """T4's hidden start: z ~ U(−1, 1)⁹, the first draw of numpy default_rng(seed) (a local generator; the exploration's
    p3_own draw)."""
    return np.random.default_rng(int(seed)).uniform(-1.0, 1.0, 2 * K + 1)


# ------------------------------------------------------------------------------------------ landscape (population)
def own_point(copy, z_start, v0, x, y):
    """Damped Newton at v₀ from z_start; accepted, of the copy's type and partition, unplaced (G(s₀) enclosure upper
    end < 0 is not required; the switch validation checks G(s₀) < 0).  With λ_min (full and reduced) and W."""
    z, gm, lam, ok = newton(np.asarray(z_start, float), v0, x, y)
    H = blocks(z, v0, x, y)[0]
    lr, ls = reduced_lam(H, z, v0) if ok else (float("nan"), None)
    typ = btype(z, v0) if ok else None
    part = partition(z) if ok else None
    pl = placed_state(z, v0)[0] if ok else None
    point_ok = bool(ok and typ == COPY_TYPE[copy] and part == COPY_PARTITION[copy] and not pl)
    return {"copy": copy, "z_s0": z.tolist(), "grad_s0": gm, "lam_min_s0": lam, "lam_reduced_s0": lr,
            "lam_split_s0": ls, "newton_ok": ok, "type": typ, "partition": [list(g) for g in part] if part else None,
            "placed_s0": pl, "G_s0": list(gap_enclosure(z, v0)), "point_ok": point_ok,
            "W": w_hold(lam) if ok else None}


def landscape():
    """Population copies at v₀ and their population switches (L4's comparators, descriptive).  STOP if a copy is invalid,
    its switch is more than 0.5% from the page's value, or κ₀'s sign differs from the page's."""
    OUT.mkdir(parents=True, exist_ok=True)
    x, y = population()
    s0 = s0_value()
    v0 = v_held(s0)
    out = {"s_pop2": s_pop2(), "s0": s0, "v0": v0.tolist(), "shares": list(SHARES), "copies": {}}
    for c in ("Q", "S", "T4"):
        row = own_point(c, POP_START[c], v0, x, y)
        row["start_distance"] = float(np.abs(np.array(row["z_s0"]) - np.array(POP_START[c])).max())
        sw = switch_of_copy(np.array(row["z_s0"]), v0, x, y, S_MAX_POP) if row["point_ok"] else {"valid": False}
        row["switch"] = sw
        row["s_pop_branch"] = sw.get("s_switch")
        row["matches_page"] = bool(sw.get("s_switch") is not None
                                   and abs(sw["s_switch"] / POP_SWITCH_PAGE[c] - 1) <= POP_SWITCH_REL)
        row["kappa_sign_matches_page"] = bool(sw.get("kappa0") is not None
                                              and np.sign(sw["kappa0"]) == POP_KAPPA_SIGN_PAGE[c])
        row["valid"] = bool(row["point_ok"] and sw.get("valid") and row["matches_page"]
                            and row["kappa_sign_matches_page"])
        out["copies"][c] = row
        print(c, row["type"], row["point_ok"], sw.get("s_switch"), sw.get("valid"), sw.get("kappa0"), flush=True)
    out["validated"] = all(out["copies"][c]["valid"] for c in out["copies"])
    (OUT / "landscape.json").write_text(json.dumps(_jsonable(out), indent=1))
    if not out["validated"]:
        raise SystemExit("STOP: population landscape not validated")
    return out


def _land():
    return json.loads((OUT / "landscape.json").read_text())


# ------------------------------------------------------------------------------------------ rules (pure functions)
def w_hold(lam, floor=W_MIN):
    """W2-A's hold length: max(floor, ⌈25/(1.0·λ_min)⌉), λ_min of the full H_zz at the own copy; None for λ ≤ 0.
    floor = W_FLOOR[arm] (4000, W2-A's; T4 8000, D1 approved 2026-10-06)."""
    if lam is None or not np.isfinite(lam) or lam <= 0:
        return None
    return int(max(floor, math.ceil(W_RELAX / (HOLD_LR * lam))))


def drive_steps(integral, rho, eta=ETA):
    """t* = (1/ρη)·∫ ds/D."""
    return float(integral) / (float(rho) * float(eta))


def budget_uncapped(t_star):
    return int(math.ceil(BUDGET_MULT * float(t_star))) + BUDGET_ADD


def budget_of(t_star):
    """B = min(⌈1.5·t*⌉ + 3000, 10⁶)."""
    return int(min(budget_uncapped(t_star), STEP_CAP))


def over_cap(t_star):
    """A seed whose budget would exceed the 10⁶-step cap: counted, NOT trained."""
    return bool(budget_uncapped(t_star) > STEP_CAP)


def first_ge(s, level, start=0):
    idx = np.nonzero(np.asarray(s, float)[start:] >= level)[0]
    return None if len(idx) == 0 else int(idx[0] + start)


def closed_form(s_path, t_sw, s_sw, kappa, lam_sw, eta=ETA):
    """W2-A's (r_cf, χ, ṡ): χ = (ṡ/s_switch)/(η·λ_min), ṡ = (s_{t_sw} − s_{t_sw−w})/w, w = min(100, t_sw); r_cf = κχ."""
    if t_sw is None or t_sw < 1 or not np.isfinite(s_sw):
        return float("nan"), float("nan"), float("nan")
    s = np.asarray(s_path, float)
    w = min(SDOT_WIN, int(t_sw))
    sdot = (s[t_sw] - s[t_sw - w]) / w
    chi = (sdot / s_sw) / (eta * lam_sw)
    return float(kappa * chi), float(chi), float(sdot)


def chi_path(s_path, lam, eta, t_end):
    s = np.asarray(s_path, float)
    lam = np.asarray(lam, float)
    n = min(int(t_end), len(s) - 1, len(lam))
    return (s[1:n + 1] - s[:n]) / s[:n] / (eta * lam[:n])


def chi_window_max(s_path, t_sw, chi, s_sw, frac=FOLLOW_FRAC):
    """V7's statistic (and the pilot's): max χ_t over 0 ≤ t < t_sw with s_t ≥ 0.8·s_switch; NaN if undefined."""
    if t_sw is None or t_sw < 1 or len(chi) < int(t_sw):
        return float("nan")
    s = np.asarray(s_path, float)[:int(t_sw)]
    w = np.asarray(chi, float)[:int(t_sw)][s >= frac * s_sw]
    return float(w.max()) if len(w) and np.all(np.isfinite(w)) else float("nan")


def q90(x):
    x = np.asarray(x, float)
    if len(x) == 0 or not np.all(np.isfinite(x)):
        return float("nan")
    return float(np.percentile(x, 90))


def ceil_to(x, m):
    """Phase 1A's rounding (phase1a_pilot._ceil_to)."""
    return round(float(math.ceil(x / m - 1e-12) * m), 10)


def tolerances(err_cross, err_lag):
    """τ = 1.5 × the pilot q90, rounded up to a multiple of 5 steps (Phase 1A's rule: τ_cross = ⌈1.5·q90|e_cross|⌉₅,
    τ_lag = ⌈max(1.5·q90|e_lag|, 1)⌉₅); q90 over the finite errors (numpy percentile, linear).  None without errors."""
    ec = np.abs(np.asarray([e for e in err_cross if e is not None and np.isfinite(e)], float))
    el = np.abs(np.asarray([e for e in err_lag if e is not None and np.isfinite(e)], float))
    qc = float(np.percentile(ec, 90)) if len(ec) else None
    ql = float(np.percentile(el, 90)) if len(el) else None
    return {"q90_abs_err_cross": qc, "q90_abs_err_lag": ql, "n_cross": int(len(ec)), "n_lag": int(len(el)),
            "tau_cross": ceil_to(TAU_MULT * qc, TAU_ROUND) if qc is not None else None,
            "tau_lag": ceil_to(max(TAU_MULT * ql, 1), TAU_ROUND) if ql is not None else None}


def pilot_rule(run_pilot, rho_start):
    """The page's pilot rule (W2-A's T rule from the arm's table ρ): run_pilot(ρ) → the on-copy pilot runs' window max χ_t
    (V7's statistic); keep ρ if q90 ≤ 0.1, else halve and rerun, at most three halvings, then STOP.  No value: STOP."""
    hist = []
    rho = float(rho_start)
    for k in range(PILOT_MAX_HALVINGS + 1):
        vals = np.asarray(run_pilot(rho), float)
        q = q90(vals)
        hist.append({"rho": rho, "n": int(len(vals)), "q90": q})
        if len(vals) == 0 or not np.isfinite(q):
            return {"status": "STOP", "rho": None, "history": hist, "reason": "no pilot value"}
        if q <= PILOT_CHI_MAX:
            return {"status": "ok", "rho": rho, "history": hist}
        if k == PILOT_MAX_HALVINGS:
            break
        rho /= 2
    return {"status": "STOP", "rho": None, "history": hist, "reason": "q90 > 0.1 after three halvings"}


# ------------------------------------------------------------------------------------------ hold and release
def hold(z_start, v0, W, x, y):
    """v fixed at v₀, z by full-batch GD at lr 1.0 for W steps; placed_state at the start state and after every step.
    Returns (z at release, record)."""
    z = np.asarray(z_start, float).copy()
    p0, u0 = placed_state(z, v0)
    npos, first, und = int(p0), (0 if p0 else None), int(u0)
    for k in range(1, int(W) + 1):
        z = z - HOLD_LR * gz(z, v0, x, y)
        p, u = placed_state(z, v0)
        und += int(u)
        if p:
            npos += 1
            first = k if first is None else first
    return z, {"W_hold": int(W), "init_placed": bool(p0), "hold_n_G_pos": npos, "hold_first_G_pos": first,
               "hold_G_positive": npos > 0, "hold_n_undecided": und}


def classify_release(z_state, v, copies, x, y, newton_result=None):
    """W2-A's Newton rule + the partition + the activity test.  copies = {name: (the copy's own point at v, canonical
    winding, or None; its partition)}.  On copy c at winding k iff Newton accepted, the Newton point within 1e−6 (sup) of
    shift(z_c, v, k), the state within 1e−3 of it, the Newton point's partition equal to the copy's, and every unit
    active.  Returns (copy or None, k or None, record)."""
    zn, gn, lamn, okn = newton(z_state, v, x, y) if newton_result is None else newton_result
    zn, z_state = np.asarray(zn, float), np.asarray(z_state, float)
    act = active_units(v)
    rec = {"newton_ok": bool(okn), "newton_grad": gn, "newton_lam": lamn, "active_units": act,
           "all_units_active": len(act) == len(v), "release_dist": float(np.abs(z_state - zn).max())}
    if okn:
        part = partition(zn)
        rec.update(newton_type=btype(zn, v), newton_partition=[list(g) for g in part], n_groups=len(part),
                   newton_placed=placed_state(zn, v)[0])
        rec["genuine_four_unit"] = bool(len(part) == K and not rec["newton_placed"])
    else:
        rec.update(newton_type="not accepted", newton_partition=None, n_groups=None, newton_placed=None,
                   genuine_four_unit=False)
    hits = []
    if okn:
        for c, (zc, pc) in copies.items():
            if zc is None:
                continue
            zc = np.asarray(zc, float)
            k = windings_to(zn, zc)
            zk = shift(zc, v, k)
            if (np.abs(zn - zk).max() <= ON_TOL and np.abs(z_state - zk).max() <= STATE_TOL
                    and partition(zn) == tuple(tuple(g) for g in pc)):
                hits.append((c, k))
    assert len(hits) <= 1, "a state cannot be on two copies"
    if hits and not rec["all_units_active"]:
        rec["note"] = "neither (inactive unit)"
        return None, None, rec
    return (hits[0][0], hits[0][1], rec) if hits else (None, None, rec)


def start_state(arm, seed, land):
    """Q, S: θ*_pop(s₀) of the arm's copy (branch-point start); T4: the seed's random hold start."""
    if arm == T4:
        return random_hold_start(seed)
    return np.array(land["copies"][COPY_OF[arm]]["z_s0"], float)


def release(arm, seed, copies_rows, W, land, x, y):
    """Start, hold and release classification (release information only).  The winding k is fixed HERE."""
    v0 = np.array(land["v0"], float)
    z0 = start_state(arm, seed, land)
    z_rel, rec = hold(z0, v0, W, x, y)
    nres = newton(z_rel, v0, x, y)
    c = COPY_OF[arm]
    own = {c: (copies_rows[c]["z_s0"] if copies_rows[c].get("point_ok") else None, COPY_PARTITION[c])}
    copy, k, crec = classify_release(z_rel, v0, own, x, y, newton_result=nres)
    rec.update(crec)
    rec.update(arm=arm, seed=int(seed), start=START[arm], z_init=z0.tolist(), copy_at_release=copy,
               windings=list(k) if k is not None else None, on_copy=bool(copy == c))
    others = {}
    for oc in POINT_COPIES[arm]:                                         # descriptive landing on Q / S own points
        o = {oc: (copies_rows[oc]["z_s0"] if copies_rows[oc].get("point_ok") else None, COPY_PARTITION[oc])}
        cc, kk, _ = classify_release(z_rel, v0, o, x, y, newton_result=nres)
        others[oc] = {"on": bool(cc == oc), "windings": list(kk) if kk is not None else None}
    rec["landing_on_other_copies"] = others
    return z_rel, rec


def eligibility(fs):
    """(eligible, reason): on the arm's copy at release, the copy validated in full, the winding in the table, and the
    budget within the 10⁶-step cap.  Otherwise the run is counted and not trained."""
    rel, cf = fs["release"], fs["copies"][COPY_OF[fs["arm"]]]
    if not rel.get("on_copy"):
        return False, "not on the arm's copy at release (counted)"
    if not cf.get("valid"):
        return False, "the occupied copy has no validated frozen switch (counted)"
    if not in_winding_table(tuple(rel["windings"])):
        return False, "winding outside the table ‖k‖₁ ≤ 3 (counted)"
    b = fs.get("budget") or {}
    if b.get("over_cap"):
        return False, "over the 10⁶-step cap: counted, not trained"
    if b.get("B") is None:
        return False, "no budget (counted)"
    return True, "eligible"


# ------------------------------------------------------------------------------------------ the branch on a v path
class StreamBranch:
    """z*(v_t) along a v path, one step at a time, by warm-started damped Newton from the copy (predictor z*(v_{t−1}) +
    ∂z*/∂v·(v_t − v_{t−1})).  Lost at the first step whose Newton is not accepted or whose correction exceeds 0.2 (sup).
    Keeps the current and previous points only, and the reduced λ_min at every computed point (W2-A's OwnBranch, with
    O(1) memory)."""

    def __init__(self, z_anchor, v0, x, y):
        self.x, self.y = x, y
        self.t, self.lost_at = 0, None
        self.lam = []
        self.z_prev = self.v_prev = self.H_prev = None
        z, _, _, ok = newton(z_anchor, np.asarray(v0, float), x, y)
        if ok:
            self._set(z, np.asarray(v0, float))
        else:
            self.lost_at = 0
            self.z = self.v = self.H = self.hzv = None

    def _set(self, z, v):
        Hzz, Hzv = blocks(z, v, self.x, self.y)
        self.z, self.v, self.H, self.hzv = z, v, Hzz, Hzv
        self.lam.append(reduced_lam(Hzz, z, v)[0])

    def advance(self, v_next):
        if self.lost_at is not None:
            return False
        v_next = np.asarray(v_next, float)
        pred = self.z + (-np.linalg.solve(self.H, self.hzv)) @ (v_next - self.v)
        zn, _, _, ok = newton(pred, v_next, self.x, self.y)
        if not ok or np.abs(zn - pred).max() > BRANCH_JUMP:
            self.lost_at = self.t + 1
            return False
        self.z_prev, self.v_prev, self.H_prev = self.z, self.v, self.H
        self._set(zn, v_next)
        self.t += 1
        return True


def bisect_switch(z_lo, v0, v1, x, y):
    """s_switch on the segment v0 → v1: the root of the branch gap bisected to 1e−12 in the segment parameter, Newton
    warm-started from the last unplaced point (W2-A's own_path_switch)."""
    lo, hi, zl = 0.0, 1.0, np.asarray(z_lo, float)
    v0, v1 = np.asarray(v0, float), np.asarray(v1, float)
    while hi - lo > 1e-12:
        mid = 0.5 * (lo + hi)
        vm = v0 + mid * (v1 - v0)
        zm, _, _, okm = newton(zl, vm, x, y)
        if okm and gap_mid(zm, vm) <= 0:
            lo, zl = mid, zm
        else:
            hi = mid
    return float(np.abs(v0 + 0.5 * (lo + hi) * (v1 - v0)).sum())


class OwnSwitch:
    """THE LAG-FREE SWITCH ON A v PATH (W2-A's own_path_switch, incremental): t_sw = the first t ≥ 1 at which z*(v_t) is
    placed (midpoint > 0); s_switch by bisection on v_{t_sw−1} → v_{t_sw}.  Rows are fed in order (row 0 at creation).
    Reads only v and the frozen copy (no gap of a training state)."""

    def __init__(self, z_anchor, v0, x, y):
        self.br = StreamBranch(z_anchor, v0, x, y)
        self.x, self.y = x, y
        self.t_sw, self.s_sw = None, float("nan")
        self.done = self.br.lost_at is not None

    def feed(self, v_row):
        if self.done:
            return
        if not self.br.advance(v_row):
            self.done = True
            return
        if branch_placed(self.br.z, self.br.v):
            self.t_sw = self.br.t
            self.s_sw = bisect_switch(self.br.z_prev, self.br.v_prev, self.br.v, self.x, self.y)
            self.done = True


def walk(V, z_anchor, x, y, z_release, eta, t_max=None):
    """One pass along the v path V: the own-path switch (W2-A's own_path_switch) and Track A's R4 recursion, SGD form
    (W2-A's r4_recursion), with identical semantics: δ₀ = z_release − z*(v₀); δ_{t+1} = (I − ηH_t)δ_t − (z*(v_{t+1}) −
    z*(v_t)); the crossing is the first t ≥ 1 with z*(v_t) + δ_t placed (midpoint); no crossing if the one-step map's
    spectral radius exceeds 1 (every 10th step), sup|δ| > 1, or the branch is lost first.  Returns a dict."""
    V = np.asarray(V, float)
    t_max = len(V) - 1 if t_max is None else int(t_max)
    br = StreamBranch(z_anchor, V[0], x, y)
    out = {"t_sw": None, "s_switch": float("nan"), "t_hit": None, "r4_status": None, "traj_max_abs_delta": float("nan"),
           "traj_radius_max": float("nan"), "branch_lost_at": None}
    if br.lost_at is not None:
        out.update(r4_status="branch lost", branch_lost_at=0)
        return out
    sw_done, r4_done = False, False
    d = np.asarray(z_release, float) - br.z
    mx, rad = float(np.abs(d).max()), 0.0
    for t in range(0, t_max + 1):
        if not sw_done and t >= 1 and branch_placed(br.z, br.v):
            out["t_sw"] = t
            out["s_switch"] = bisect_switch(br.z_prev, br.v_prev, br.v, x, y)
            sw_done = True
        if not r4_done:
            if t % RADIUS_EVERY == 0:
                rad = max(rad, float(np.abs(1.0 - eta * np.linalg.eigvalsh(0.5 * (br.H + br.H.T))).max()))
                if rad > 1.0:
                    out["r4_status"], r4_done = "unstable", True
            if not r4_done and t >= 1 and branch_placed(br.z + d, br.v):
                out["t_hit"], out["r4_status"], r4_done = t, "ok", True
        if sw_done and r4_done:
            break
        if t == t_max:
            if not r4_done:
                out["r4_status"] = "no predicted crossing within the budget"
            break
        if not br.advance(V[t + 1]):
            if not r4_done:
                out["r4_status"] = "branch lost"
            break
        if not r4_done:
            d = d - eta * (br.H_prev @ d) - (br.z - br.z_prev)
            mx = max(mx, float(np.abs(d).max()))
            if not np.isfinite(mx) or mx > DIVERGED:
                out["r4_status"], r4_done = "diverged", True
                if sw_done:
                    break
    out.update(traj_max_abs_delta=mx, traj_radius_max=rad, branch_lost_at=br.lost_at, branch_steps=br.t)
    return out


# ------------------------------------------------------------------------------------------ the causal forecast
def config(f=F_CUT):
    return C.ForecastConfig(f=float(f), family=FAMILY, window_frac=WINDOW_FRAC, min_window=MIN_WINDOW,
                            horizon_factor=HORIZON)


def extrapolate_v4(V_vis, cfg, s_ref, budget_left):
    """(V̂ rows for t_c, …, status, info) at width 4: ŝ by causal_forecast.extrapolate_scale (log s, the frozen family,
    anchored), EACH UNIT'S SHARE u_i = |v_i|/s by causal_forecast.extrapolate_share (the same family in t, anchored),
    clipped at 0 and renormalised; the signs of v_{t_c−1}.  No forecast if a sign changes in the window."""
    V_vis = np.asarray(V_vis, float)
    s_vis = np.abs(V_vis).sum(axis=1)
    s_hat, st, info = C.extrapolate_scale(s_vis, cfg, s_ref, budget_left)
    if st != "ok":
        return np.zeros((0, V_vis.shape[1])), st, info
    n_w = info["n_window"]
    sg = np.sign(V_vis[-n_w:])
    if np.any(sg == 0) or np.any(sg != sg[-1]):
        return np.zeros((0, V_vis.shape[1])), "no forecast: an output weight changes sign in the window", info
    U = np.abs(V_vis) / s_vis[:, None]
    Uh = np.column_stack([C.extrapolate_share(U[:, i], np.log(s_vis), np.log(s_hat), n_w, cfg.family)
                          for i in range(V_vis.shape[1])])
    Uh = np.clip(Uh, 0.0, None)
    tot = Uh.sum(axis=1)
    if np.any(tot <= 0) or not np.all(np.isfinite(tot)):
        return np.zeros((0, V_vis.shape[1])), "no forecast: extrapolated shares degenerate", info
    Uh = Uh / tot[:, None]
    return sg[-1][None, :] * s_hat[:, None] * Uh, "ok", info


@dataclass
class W4Inputs:
    """out: the v path (guarded, cutoff t_c; 4 columns); hid: the z path (guarded: row 0, the release, only); copy_row:
    the occupied copy's frozen row (z_s0, s_switch); windings k (fixed at release); x, y: the own sample."""
    t_c: int
    budget: int
    out: object
    hid: object
    copy_row: dict
    windings: tuple
    x: np.ndarray
    y: np.ndarray
    eta: float = ETA


def forecast_w4(inp: W4Inputs, cfg):
    t_c = int(inp.t_c)
    V = C._prefix(inp.out, t_c)
    guards = (inp.out, inp.hid)
    z_rel = C._row(inp.hid, 0)
    cf, k = inp.copy_row, tuple(inp.windings)
    s_ref = float(cf["s_switch"])
    V_hat, st, info = extrapolate_v4(V, cfg, s_ref, max(0, inp.budget - t_c + 1))
    s_vis = np.abs(V).sum(axis=1)
    if st != "ok":
        return C._finish({"status": st, "s_switch": float("nan")}, s_vis, t_c, info, guards)
    V_ext = np.vstack([V, V_hat])
    s_ext = np.abs(V_ext).sum(axis=1)
    w = walk(V_ext, shift(np.array(cf["z_s0"]), V_ext[0], k), inp.x, inp.y, z_rel, inp.eta)
    res = {"t_sw": w["t_sw"], "s_switch": float(w["s_switch"])}
    if w["t_sw"] is None or not np.isfinite(w["s_switch"]):
        return C._finish({**res, "status": "no forecast: no switch on the extrapolated v path",
                          "branch_lost_at": w["branch_lost_at"]}, s_ext, t_c, info, guards)
    res.update(traj_radius_max=w["traj_radius_max"], traj_max_abs_delta=w["traj_max_abs_delta"],
               branch_lost_at=w["branch_lost_at"], t_hit=w["t_hit"],
               status="ok" if w["t_hit"] is not None else f"no forecast: {w['r4_status']}")
    return C._finish(res, s_ext, t_c, info, guards)


ADAPTERS = {"w4": forecast_w4}


def run_forecast(inp, cfg):
    """The forecast with every guard's last row asserted < t_c (causal_forecast.run_forecast's check)."""
    out = ADAPTERS["w4"](inp, cfg)
    for name, mi in out["max_index_read"].items():
        if mi >= inp.t_c:
            raise C.CausalityViolation(f"{name}: row {mi} read with cutoff {inp.t_c}")
    return out


def same_forecast(a, b):
    """Every output field identical (NaN equal to NaN; `max_index_read` and timings ignored): phase1c.same_forecast."""
    ka = {k for k in a if k not in ("max_index_read", "secs")}
    kb = {k for k in b if k not in ("max_index_read", "secs")}
    if ka != kb:
        return False
    for k in ka:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if isinstance(x, (np.ndarray, list, tuple)) or isinstance(y, (np.ndarray, list, tuple)):
            if not np.array_equal(np.asarray(x, float), np.asarray(y, float), equal_nan=True):
                return False
            continue
        if x != y:
            return False
    return True


def forecast_one(V, Z, t_c, budget, copy_row, windings, x, y, f=F_CUT):
    """ONE causal forecast: the forecaster receives GuardedArray views (v rows < t_c; z row 0 only) and run_forecast
    re-asserts every guard's last row < t_c; then the NaN recomputation: the same forecaster, unguarded, on copies with
    every row ≥ t_c set to NaN.  The record says whether the output is identical."""
    t0 = time.time()
    cfg = config(f)
    inp = W4Inputs(int(t_c), int(budget), C.guard(V, t_c, name="v"), C.guard(Z, t_c, rows=[0], name="z"),
                   copy_row, tuple(windings), x, y)
    fc = run_forecast(inp, cfg)
    Vn = np.array(V, float, copy=True)
    Vn[t_c:] = np.nan
    Zn = np.array(Z, float, copy=True)
    Zn[t_c:] = np.nan
    fc2 = ADAPTERS["w4"](W4Inputs(int(t_c), int(budget), Vn, Zn, copy_row, tuple(windings), x, y), cfg)
    rec = dict(fc)
    rec["t_sw_fc"] = fc.get("t_sw")
    rec["lag_fc"] = fc.get("lag_steps_fc")
    rec["nan_recompute_identical"] = bool(same_forecast(fc, fc2))
    rec["secs"] = round(time.time() - t0, 2)
    return rec


# ------------------------------------------------------------------------------------------ training
def train_to_cutoff(z0, v0, rho, level, n_max, x, y, eta=ETA):
    """Full-batch SGD (η on z, ρη on v) from the release to t_c = the first t ≥ 1 with s_t ≥ level (a stopping time on
    s; no gap or placement evaluated), or the budget.  Hard caps: n_max ≤ 10⁶ steps and the current RSS ≤ 1 GB
    (checked every 20,000 steps).  Returns (Z rows 0 … t, V rows 0 … t, t_c or None, status)."""
    assert int(n_max) <= STEP_CAP, "budget above the 10⁶-step cap"
    z, v = np.asarray(z0, float).copy(), np.asarray(v0, float).copy()
    Zs, Vs = np.empty((int(n_max) + 1, len(z))), np.empty((int(n_max) + 1, len(v)))   # pages touched as written
    Zs[0], Vs[0] = z, v
    for t in range(1, int(n_max) + 1):
        g_z, g_v = grad(z, v, x, y)
        z = z - eta * g_z
        v = v - rho * eta * g_v
        Zs[t], Vs[t] = z, v
        if float(np.abs(v).sum()) >= level:
            return Zs[:t + 1].copy(), Vs[:t + 1].copy(), t, "ok"
        if t % RSS_EVERY == 0 and current_rss() > RSS_RUN_CAP:
            return Zs[:t + 1].copy(), Vs[:t + 1].copy(), None, "aborted: per-run RSS cap (1 GB)"
    return Zs, Vs, None, "no cutoff within the budget"


def run_one(fs, rho, land):
    """Train from the frozen release to t_c (no observation), hash the state at t_c and the v prefix, and forecast.  The
    row holds no observed quantity (FORBIDDEN keys asserted in finalize)."""
    t0 = time.time()
    arm, seed = fs["arm"], fs["seed"]
    rel = fs["release"]
    row = {"arm": arm, "seed": int(seed), "rho": float(rho), "copy_at_release": rel.get("copy_at_release"),
           "windings": rel.get("windings"), "on_copy": bool(rel.get("on_copy")),
           "hold_G_positive": bool(rel.get("hold_G_positive"))}
    ok, why = eligibility(fs)
    row.update(eligible=ok, status=why)
    if not ok:
        return row
    cf = fs["copies"][COPY_OF[arm]]
    x, y = own_sample(seed)
    z_rel = np.array(fs["z_release"], float)
    assert array_sha(z_rel) == fs["z_release_sha256"], "release hash mismatch"
    v0 = np.array(land["v0"], float)
    B = int(fs["budget"]["B"])
    s_ref = float(cf["s_switch"])
    Z, V, t_c, st = train_to_cutoff(z_rel, v0, rho, F_CUT * s_ref, B, x, y)
    row.update(budget=B, s_ref=s_ref, t_c=t_c, status=st, steps_trained=int(len(V) - 1))
    if t_c is None:
        row["forecast"] = {"t_c": None, "status": st, "t_fc": None, "t_sw_fc": None}
        row["secs_run"] = round(time.time() - t0, 1)
        return row
    row.update(state_tc_sha256=array_sha(Z[t_c], V[t_c]), v_prefix_sha256=array_sha(V), s_at_tc=float(np.abs(V[t_c]).sum()))
    row["forecast"] = forecast_one(V, Z, t_c, B, cf, rel["windings"], x, y)
    row["secs_run"] = round(time.time() - t0, 1)
    return row


def follow_decision(zn, okn, zc, okc, part_copy):
    """Follows iff the run's Newton point and the copy's carried branch point are both accepted, within 1e−6 (sup) at
    the release winding, and the run's Newton point keeps the copy's coincidence partition."""
    if not (okn and okc):
        return False
    return bool(np.abs(np.asarray(zn, float) - np.asarray(zc, float)).max() <= ON_TOL
                and partition(np.asarray(zn, float)) == tuple(tuple(g) for g in part_copy))


def follow_check(z_t, v_t, cf, copy, k, x, y):
    """At t₀.₈ (the first step with s_t ≥ 0.8·the copy's frozen switch, where its follow point is): Newton at v_{t₀.₈} from
    the run's z; the copy's branch point there = Newton from the frozen follow point carried by one tangent predictor,
    at the release winding.  The state at t₀.₈ only (no gap)."""
    fp = cf["follow_point"]
    vf, zf = np.array(fp["v"]), np.array(fp["z"])
    guess = shift(zf + np.array(fp["dz_dv"]) @ (np.asarray(v_t) - vf), v_t, k)
    zc, _, _, okc = newton(guess, v_t, x, y)
    zn, _, _, okn = newton(z_t, v_t, x, y)
    fol = follow_decision(zn, okn, zc, okc, COPY_PARTITION[copy])
    return {"follows_branch": fol, "follow_newton_ok": okn, "follow_copy_point_ok": okc,
            "follow_newton_dist": float(np.abs(zn - zc).max()) if okc and okn else None,
            "follow_partition": [list(g) for g in partition(zn)] if okn else None}


def observe_one(fs, rho, run_row, land, chunk=5000):
    """AFTER the forecasts commit: retrain from the release (deterministic), every-step detection (enclosure lower end
    > 0), the state at t_c asserted equal to the hashed one bit for bit, continued until the crossing, t_c and the
    actual own-path switch are all decided (or the budget).  Then the registered validity inputs (W2-A's): κ_k, ηλ,
    κ/(ηλ), r_cf and χ at the actual t_sw, the window max χ_t, and the follow check at t₀.₈."""
    arm, seed = fs["arm"], fs["seed"]
    cf, rel = fs["copies"][COPY_OF[arm]], fs["release"]
    k = tuple(rel["windings"])
    x, y = own_sample(seed)
    z = np.array(fs["z_release"], float)
    assert array_sha(z) == fs["z_release_sha256"], "release hash mismatch"
    v = np.array(land["v0"], float)
    B = int(fs["budget"]["B"])
    t_c = run_row.get("t_c")
    sw = OwnSwitch(shift(np.array(cf["z_s0"]), v, k), v, x, y)
    s_f = cf["follow_point"]["s"]
    Vs = np.empty((B + 1, len(v)))
    Vs[0] = v
    t_obs, und, t08, st08, hash_ok = None, 0, None, None, None
    t = 0
    while t < B:
        for _ in range(min(chunk, B - t)):
            t += 1
            g_z, g_v = grad(z, v, x, y)
            z = z - ETA * g_z
            v = v - rho * ETA * g_v
            Vs[t] = v
            if t_obs is None:
                p, u = placed_state(z, v)
                und += int(u)
                if p:
                    t_obs = t
            if t08 is None and float(np.abs(v).sum()) >= s_f:
                t08, st08 = t, (z.copy(), v.copy())
            if t_c is not None and t == t_c:
                hash_ok = bool(array_sha(z, v) == run_row["state_tc_sha256"]
                               and array_sha(Vs[:t + 1]) == run_row["v_prefix_sha256"])
                if not hash_ok:
                    raise AssertionError(f"state at t_c differs from the hashed one ({arm} {seed})")
            sw.feed(v)
            if t % RSS_EVERY == 0 and current_rss() > RSS_RUN_CAP:
                raise SystemExit(f"STOP: per-run RSS cap during observation ({arm} {seed})")
        if t_obs is not None and sw.done and (t_c is None or t >= t_c):
            break
    V = Vs[:t + 1]
    s = np.abs(V).sum(axis=1)
    o = {"arm": arm, "seed": int(seed), "steps_observed": int(t), "crossed": t_obs is not None, "t_obs": t_obs,
         "s_obs": float(s[t_obs]) if t_obs is not None else None, "n_undecided": und, "state_tc_hash_ok": hash_ok,
         "t_sw": sw.t_sw, "s_sw": sw.s_sw if sw.t_sw is not None else None, "branch_lost_at": sw.br.lost_at}
    kap = kappa_winding(cf["kappa0"], cf["kappa_winding_coef"], k)
    lam_sw = cf["lam_min_switch"]
    o.update(kappa=kap, lam_min_switch=lam_sw, eta_lam=ETA * lam_sw, lag_steps_pred=kap / (ETA * lam_sw),
             s_pop=land["copies"][COPY_OF[arm]]["s_pop_branch"])
    if sw.t_sw is not None:
        r_cf, chi, sdot = closed_form(s, sw.t_sw, sw.s_sw, kap, lam_sw)
        cp = chi_path(s, sw.br.lam, ETA, sw.t_sw)
        o.update(r_cf=r_cf, chi_tsw=chi, sdot=sdot, chi_window_max=chi_window_max(s, sw.t_sw, cp, sw.s_sw))
    if t08 is not None:
        o.update(t_follow=t08, **follow_check(st08[0], st08[1], cf, COPY_OF[arm], k, x, y))
    else:
        o.update(t_follow=None, follows_branch=False)
    return o


# ------------------------------------------------------------------------------------------ pure scoring rules
def _arr(v, dt=float):
    if isinstance(v, list):
        return np.array([np.nan if (dt is float and x is None) else x for x in v], dtype=dt)
    return np.asarray(v, dt)


def gate(arm, on_copy, hold_positive):
    """Q, S: ≥ 96/120 on the arm's copy AND no run with G > 0 at the start or any hold step; T4: ≥ 60 runs (of 710) on
    T4, regardless of G in the hold."""
    on_copy, hold_positive = _arr(on_copy, bool), _arr(hold_positive, bool)
    n_on = int(on_copy.sum())
    ok_count, ok_hold = bool(n_on >= GATE_MIN[arm]), not bool(hold_positive.any())
    return {"n_runs": int(len(on_copy)), "n_on_copy": n_on, "min_on_copy": GATE_MIN[arm],
            "n_hold_G_positive": int(hold_positive.sum()), "count_ok": ok_count, "hold_ok": ok_hold,
            "hold_condition_applies": GATE_HOLD_CONDITION[arm],
            "pass": bool(ok_count and (ok_hold or not GATE_HOLD_CONDITION[arm]))}


def cutoff_before(t_c, t_obs):
    t_c, t_obs = _arr(t_c), _arr(t_obs)
    with np.errstate(invalid="ignore"):
        return np.isfinite(t_c) & np.isfinite(t_obs) & (t_c < t_obs)


def cutoff_validity(before, base):
    """t_c strictly before the crossing in ≥ 90% of the crossing runs `base` (no cutoff = not before); none: fails."""
    before, base = _arr(before, bool), _arr(base, bool)
    n = int(base.sum())
    k = int((before & base).sum())
    frac = k / n if n else float("nan")
    return {"n_crossing_runs": n, "n_cutoff_before_crossing": k, "n_cutoff_not_before_unscored": n - k, "frac": frac,
            "min_frac": CUTOFF_BEFORE_MIN_FRAC, "ok": bool(n and frac >= CUTOFF_BEFORE_MIN_FRAC)}


def is_forecast(t_fc, t_sw_fc):
    return np.isfinite(_arr(t_fc)) & np.isfinite(_arr(t_sw_fc))


def criterion_within(err, tau, has_fc):
    """C1 / C2: |err| ≤ τ (inclusive) in ≥ 80% of the scored runs; a miss or an undefined error is not within; no
    scored run: UNRESOLVED."""
    err, has_fc = _arr(err), _arr(has_fc, bool)
    n = len(err)
    with np.errstate(invalid="ignore"):
        ok = has_fc & np.isfinite(err) & (np.abs(err) <= tau)
    k = int(ok.sum())
    frac = k / n if n else float("nan")
    return {"n": n, "n_within": k, "n_miss": int((~has_fc).sum()),
            "n_err_undefined": int((has_fc & ~np.isfinite(err)).sum()), "frac_within": frac, "tau": tau,
            "min_frac": WITHIN_MIN_FRAC,
            "verdict": "UNRESOLVED" if n == 0 else ("PASS" if frac >= WITHIN_MIN_FRAC else "FAIL")}


def criterion_c3(r_obs, r_fc, lag_fc, has_fc):
    """C3: the median of r_obs/r_fc over the scored runs with a forecast, |lag_fc| ≥ 6 and r_fc ≠ 0 lies in [0.9, 1.1]
    (closed; float64).  None: UNRESOLVED."""
    r_obs, r_fc, lag_fc, has_fc = _arr(r_obs), _arr(r_fc), _arr(lag_fc), _arr(has_fc, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        el = has_fc & np.isfinite(lag_fc) & (np.abs(lag_fc) >= MIN_ABS_LAG) & (r_fc != 0)
        q = np.where(el, r_obs / r_fc, np.nan)
    ok = np.isfinite(q)
    n = int(ok.sum())
    m = float(np.median(q[ok])) if n else float("nan")
    return {"n": n, "n_excluded": int(len(q) - n), "median_ratio": m, "band": list(C3_BAND),
            "min_abs_lag_fc": MIN_ABS_LAG,
            "verdict": "UNRESOLVED" if n == 0 else ("PASS" if C3_BAND[0] <= m <= C3_BAND[1] else "FAIL")}


def bootstrap_mean_ci(D, n_boot=BOOT_N, seed=BOOT_SEED):
    """(mean, lower, upper): the 95% percentile bootstrap of the mean (local default_rng(seed), n_boot resamples with
    replacement; the registered GELU-T / W2-A / 1C procedure)."""
    D = np.asarray(D, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    means = D[idx].mean(axis=1)
    lo, hi = np.percentile(means, BOOT_PCT)
    return float(D.mean()), float(lo), float(hi)


def criterion_c4(t_fc, t_sw_fc, t_obs, has_fc):
    """C4: D = |t_fc − t_obs| − |t_sw,fc − t_obs| (the no-lag forecast), paired per run over the scored runs with a
    forecast; PASS iff the upper end of the 95% bootstrap interval of mean D is < 0 (above 0 or containing 0: FAIL);
    fewer than 2 runs: UNRESOLVED."""
    t_fc, t_sw_fc, t_obs, has_fc = _arr(t_fc), _arr(t_sw_fc), _arr(t_obs), _arr(has_fc, bool)
    ok = has_fc & np.isfinite(t_fc) & np.isfinite(t_sw_fc) & np.isfinite(t_obs)
    D = np.abs(t_fc[ok] - t_obs[ok]) - np.abs(t_sw_fc[ok] - t_obs[ok])
    n = int(ok.sum())
    mean, lo, hi = bootstrap_mean_ci(D) if n >= 2 else (float("nan"),) * 3
    computable = bool(n >= 2 and np.isfinite(hi))
    return {"n": n, "mean_D": mean, "ci95": [lo, hi], "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
            "verdict": "UNRESOLVED" if not computable else ("PASS" if hi < 0 else "FAIL"),
            "DESCRIPTIVE_n_forecast_closer": int((D < 0).sum()),
            "DESCRIPTIVE_frac_forecast_closer": float((D < 0).mean()) if n else float("nan")}


def criterion_sign(lag_fc, lag_obs, has_fc):
    """S: lag_fc < 0 and lag_obs < 0 in ≥ 80% of the eligible scored runs: those with a forecast and |lag_fc| ≥ 6, plus
    every miss (counted as not both negative; 1C's rule).  None eligible: UNRESOLVED."""
    lag_fc, lag_obs, has_fc = _arr(lag_fc), _arr(lag_obs), _arr(has_fc, bool)
    with np.errstate(invalid="ignore"):
        el_fc = has_fc & np.isfinite(lag_fc) & (np.abs(lag_fc) >= MIN_ABS_LAG)
        miss = ~has_fc
        el = el_fc | miss
        both = el_fc & (lag_fc < 0) & np.isfinite(lag_obs) & (lag_obs < 0)
    n, k = int(el.sum()), int(both.sum())
    frac = k / n if n else float("nan")
    return {"n_eligible": n, "n_eligible_miss": int(miss.sum()), "n_both_negative": k, "frac": frac,
            "min_frac": S_MIN_FRAC, "min_abs_lag": MIN_ABS_LAG,
            "verdict": "UNRESOLVED" if n == 0 else ("PASS" if frac >= S_MIN_FRAC else "FAIL")}


def outcome(verdicts):
    vs = list(verdicts.values())
    if "UNRESOLVED" in vs:
        return "UNRESOLVED"
    if all(v == "PASS" for v in vs):
        return "PASS"
    return "FAIL " + "+".join(k for k, v in verdicts.items() if v == "FAIL")


def criteria(arm, t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs, has_fc, tau_cross, tau_lag, names=None):
    t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs = (_arr(v) for v in (t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs))
    has_fc = _arr(has_fc, bool)
    lag_fc, lag_obs = t_fc - t_sw_fc, t_obs - t_sw
    out = {"C1": criterion_within(t_fc - t_obs, tau_cross, has_fc),
           "C2": criterion_within(lag_fc - lag_obs, tau_lag, has_fc),
           "C3": criterion_c3(r_obs, r_fc, lag_fc, has_fc),
           "C4": criterion_c4(t_fc, t_sw_fc, t_obs, has_fc)}
    if "S" in (names or CRITERIA[arm]):
        out["S"] = criterion_sign(lag_fc, lag_obs, has_fc)
    return out


def validity_v(scored, on_copy, crossed, t_obs, t_sw, kappa, eta_lam, lag_steps, r_cf, chi_tsw, chi_win_max,
               pilot_chi_median):
    """W2-A's V1-V7 (with its change 3: V4 on |κ/(ηλ)|, V5 on |κχ|) over the scored runs."""
    A = np.asarray
    scored, on_copy, crossed = A(scored, bool), A(on_copy, bool), A(crossed, bool)
    t_obs, t_sw, kappa, eta_lam, lag_steps, r_cf, chi_tsw, chi_win_max = (
        _arr(v) for v in (t_obs, t_sw, kappa, eta_lam, lag_steps, r_cf, chi_tsw, chi_win_max))
    n = int(scored.sum())
    with np.errstate(invalid="ignore"):
        kpos = on_copy & crossed & np.isfinite(kappa) & (kappa > 0)
        tsw_before = kpos & np.isfinite(t_sw) & np.isfinite(t_obs) & (t_sw < t_obs)
    frac_tsw = float(tsw_before.sum() / kpos.sum()) if kpos.sum() else float("nan")

    def med(v):
        return float(np.median(v)) if len(v) else float("nan")
    regime_frac = float(np.mean(eta_lam[scored] <= REGIME_MAX)) if n else float("nan")
    lag_med = med(np.abs(lag_steps[scored]))
    kc_q90 = q90(np.abs(r_cf[scored]))
    chi_med = med(chi_tsw[scored])
    chi_rel = (abs(chi_med / pilot_chi_median - 1) if n and pilot_chi_median is not None
               and np.isfinite(pilot_chi_median) and pilot_chi_median != 0 else float("nan"))
    chi_path_q90 = q90(chi_win_max[scored])
    V = {"V1_min_scored_60": n >= MIN_SCORED,
         "V2_tsw_before_crossing_90pct_kappa_pos": bool(kpos.sum() == 0 or frac_tsw >= TSW_MIN_FRAC),
         "V3_regime_eta_lam_le_0p5_in_80pct": bool(n and regime_frac >= REGIME_MIN_FRAC),
         "V4_median_abs_predicted_lag_ge_10_steps": bool(np.isfinite(lag_med) and lag_med >= LAG_MIN_STEPS),
         "V5_q90_abs_kappa_chi_le_0p1": bool(np.isfinite(kc_q90) and kc_q90 <= KC_Q90_MAX),
         "V6_median_chi_within_30pct_of_pilot": bool(np.isfinite(chi_rel) and chi_rel <= CHI_REL_TOL),
         "V7_q90_max_chi_window_le_0p25": bool(np.isfinite(chi_path_q90) and chi_path_q90 <= CHI_PATH_Q90_MAX)}
    stats = {"n_scored": n, "n_kappa_pos_crossing": int(kpos.sum()), "frac_tsw_before_crossing_kappa_pos": frac_tsw,
             "regime_frac": regime_frac, "median_abs_predicted_lag_steps": lag_med, "q90_abs_kappa_chi_tsw": kc_q90,
             "median_chi_tsw": chi_med, "pilot_median_chi_tsw": pilot_chi_median, "chi_rel_to_pilot": chi_rel,
             "q90_max_chi_window": chi_path_q90}
    return V, stats


def spearman(a, b):
    """Spearman with average ranks (track_a.spearman's pandas ranking)."""
    import pandas as pd
    ra = pd.Series(np.asarray(a, float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, float)).rank().to_numpy()
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def descriptive_L(R, fs, r_obs):
    """DESCRIPTIVE ONLY (never a verdict): W2-A's L1-L5 with the causal forecast in place of the registered prediction
    (r_traj := r_fc, s_traj := s_fc, t_traj := t_fc), over the scored runs with a forecast fs."""
    n = int(fs.sum())
    if n == 0:
        return {"n": 0}
    ro, rf, rc = r_obs[fs], R["r_fc"][fs], R["r_cf"][fs]
    so, sf, sp, ss = R["s_obs"][fs], R["s_fc"][fs], R["s_pop"][fs], R["s_sw"][fs]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"n": n, "L1_median_r_obs_over_r_fc": float(np.median(ro / rf)),
               "L2_median_r_obs_over_r_cf": float(np.median(ro / rc)),
               "L3_spearman_r_fc_r_obs": spearman(rf, ro) if n >= 2 else float("nan")}
        if n >= 2:
            D4 = np.abs(np.log(so / sf)) - np.abs(np.log(so / sp))
            out["L4_mean_D_ci95"] = list(bootstrap_mean_ci(D4))
        el = np.abs(R["t_fc"][fs] - R["t_sw"][fs]) >= MIN_ABS_LAG
        if el.sum() >= 2:
            D5 = np.abs(np.log(so[el] / sf[el])) - np.abs(np.log(so[el] / ss[el]))
            out["L5_n"] = int(el.sum())
            out["L5_mean_D_ci95"] = list(bootstrap_mean_ci(D5))
    return out


BOOL_KEYS = ("crossed", "on_copy", "follows", "hold_positive", "nan_identical", "eligible", "over_cap", "aborted")


def score_arm(arm, R, pilot_chi, tau_cross, tau_lag):
    """The registered verdict for one arm.  R: arrays over ALL the arm's seeds (NaN / False where undefined).
    1. Gate (first).  Fails: every criterion UNRESOLVED, nothing scored: 'UNRESOLVED (gate)'.
    2. Base (crossing runs): on the arm's copy at release (trained: eligible), follows its branch (the follow check,
       partition kept), crossed within the budget, finite r_cf and r_obs.
    3. Scored: the base with the cutoff STRICTLY before the crossing; a scored run without a forecast is a MISS.
    4. Validity: V1-V7 over the scored runs (W2-A's, misses included), the cutoff before the crossing in ≥ 90% of the
       base, and the NaN recomputation identical in every run with a cutoff.  Fails: 'UNRESOLVED (validity)'.
    5. C1-C4 (+ S for arm S) over the scored runs; outcome()."""
    R = {k: (_arr(v, bool) if k in BOOL_KEYS else _arr(v)) for k, v in R.items()}
    n = len(R["t_obs"])
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = R["s_obs"] / R["s_sw"] - 1
    names = CRITERIA[arm]
    out = {"arm": arm, "role": ROLE[arm], "n_runs": n, "f": F_CUT, "tau_cross": tau_cross, "tau_lag": tau_lag}
    g = gate(arm, R["on_copy"], R["hold_positive"])
    out["gate"] = g
    if not g["pass"]:
        out.update({k: {"verdict": "UNRESOLVED"} for k in names})
        out.update(valid=False, outcome="UNRESOLVED (gate)", n_scored=0, scored_index=[])
        return out
    before = cutoff_before(R["t_c"], R["t_obs"])
    has_fc = is_forecast(R["t_fc"], R["t_sw_fc"])
    base = (R["on_copy"] & R["eligible"] & R["follows"] & R["crossed"] & np.isfinite(R["r_cf"])
            & np.isfinite(r_obs))
    scored = base & before
    V, vstats = validity_v(scored, R["on_copy"] & R["eligible"], R["crossed"], R["t_obs"], R["t_sw"], R["kappa"],
                           R["eta_lam"], R["lag_steps"], R["r_cf"], R["chi_tsw"], R["chi_win_max"], pilot_chi)
    cv = cutoff_validity(before, base)
    withc = np.isfinite(R["t_c"])
    V["cutoff_before_crossing_90pct"] = cv["ok"]
    V["nan_recomputation_identical"] = bool(np.all(R["nan_identical"][withc])) if withc.any() else True
    valid = all(bool(v) for v in V.values())
    out.update(validity={k: bool(v) for k, v in V.items()}, valid=valid, validity_inputs=vstats, cutoff_validity=cv,
               n_on_copy=int(R["on_copy"].sum()), n_eligible=int(R["eligible"].sum()),
               n_over_cap_not_trained=int((R["on_copy"] & R["over_cap"]).sum()),
               n_aborted_rss=int(R["aborted"].sum()),
               n_on_copy_not_following=int((R["on_copy"] & R["eligible"] & R["crossed"] & ~R["follows"]).sum()),
               n_base_crossing_runs=int(base.sum()), n_scored=int(scored.sum()),
               n_scored_miss=int((scored & ~has_fc).sum()), n_scored_with_forecast=int((scored & has_fc).sum()),
               n_crossed_cutoff_not_before=int((base & ~before).sum()), n_nan_recompute_checked=int(withc.sum()),
               n_nan_recompute_differs=int((withc & ~R["nan_identical"]).sum()),
               scored_index=np.nonzero(scored)[0].tolist())
    Sx = lambda k: R[k][scored]                                              # noqa: E731
    crit = criteria(arm, Sx("t_fc"), Sx("t_sw_fc"), Sx("r_fc"), Sx("t_obs"), Sx("t_sw"), r_obs[scored],
                    has_fc[scored], tau_cross, tau_lag, names=("C1", "C2", "C3", "C4", "S"))
    for k in names:
        c = crit[k]
        if not valid:
            c = {**c, "verdict_if_valid": c["verdict"], "verdict": "UNRESOLVED"}
        out[k] = c
    out["outcome"] = "UNRESOLVED (validity)" if not valid else outcome({k: out[k]["verdict"] for k in names})
    d = {}
    if "S" not in names:
        d["S_descriptive"] = crit["S"]
    fs = scored & has_fc
    d["L1_L5_on_r_fc"] = descriptive_L(R, fs, r_obs)
    with np.errstate(invalid="ignore"):
        d["horizon_steps_t_obs_minus_t_c"] = (lambda v: [float(np.min(v)), float(np.median(v)), float(np.max(v))]
                                              if len(v) else None)((R["t_obs"] - R["t_c"])[scored])
    if arm == T4:
        keep = scored & ~R["hold_positive"]
        d["sensitivity_excluding_hold_G_positive"] = {
            "n_scored": int(keep.sum()),
            **{k: v["verdict"] for k, v in criteria(arm, R["t_fc"][keep], R["t_sw_fc"][keep], R["r_fc"][keep],
                                                    R["t_obs"][keep], R["t_sw"][keep], r_obs[keep], has_fc[keep],
                                                    tau_cross, tau_lag).items()}}
    out["DESCRIPTIVE"] = d
    return out


def collapse_table(releases):
    """DESCRIPTIVE (the finding reported first): where random holds land at s₀.  releases: T4 release records."""
    n = len(releases)
    types = {}
    for r in releases:
        key = (r.get("newton_type") or "not accepted") + (" placed" if r.get("newton_placed") else "")
        types[key] = types.get(key, 0) + 1
    acc = [r for r in releases if r.get("newton_ok")]
    return {"n_random_holds": n, "n_newton_accepted": len(acc),
            "n_genuine_four_unit_unplaced": int(sum(bool(r.get("genuine_four_unit")) for r in releases)),
            "n_coinciding_unit_points": int(sum(1 for r in acc if (r.get("n_groups") or K) < K)),
            "n_on_T4": int(sum(bool(r.get("on_copy")) for r in releases)),
            "n_on_Q_own_point": int(sum(bool((r.get("landing_on_other_copies") or {}).get("Q", {}).get("on"))
                                        for r in releases)),
            "n_on_S_own_point": int(sum(bool((r.get("landing_on_other_copies") or {}).get("S", {}).get("on"))
                                        for r in releases)),
            "n_placed": int(sum(bool(r.get("newton_placed")) for r in acc)),
            "by_type": dict(sorted(types.items(), key=lambda t: -t[1]))}


# ------------------------------------------------------------------------------------------ seed scan
SCAN_DIRS = ("src", "tests", "results", "paper", "notes", "independent", "data", "dist")
SCAN_SKIP_FILES = ("phase3_w4.py", "test_phase3_w4.py", "phase3_registration.md", "phase3_width4_asym_design.md")
SCAN_SKIP_SUFFIX = (".npz", ".npy", ".pdf", ".png", ".pt", ".pkl", ".ots", ".bak", ".gz", ".zip", ".pyc", ".jpg")
SEED_PREFIXES = ("7430", "7431", "7432", "7439")


def scan_patterns():
    pats = {}
    for p in SEED_PREFIXES:
        pats[p] = (rf"(^|[^0-9.]){p}[0-9]{{3}}([^0-9]|$)|(^|[^0-9]){p[0]}_{p[1:4]}_[0-9]{{3}}([^0-9]|$)"
                   rf"|(^|[^0-9.,]){p[0]},{p[1:4]},[0-9]{{3}}([^0-9,]|$)")
    return pats


def scan_tree(patterns, dirs=SCAN_DIRS, chunk=16 * 1024 * 1024, overlap=256):
    """Regex scan of every text file under `dirs` (binary files skipped; Phase 3's own files, the approved page and
    results/phase3/ skipped), streamed in 16 MB chunks with a 256-byte overlap.  {pattern key: [files with a match]}."""
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


def _flat_ints(v):
    if isinstance(v, dict):
        return [i for x in v.values() for i in _flat_ints(x)]
    if isinstance(v, (tuple, list, range, set, frozenset)):
        return [i for x in v for i in _flat_ints(x)]
    return [int(v)] if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else []


def all_phase3_seeds():
    return set(SEEDS_Q) | set(SEEDS_S) | set(SEEDS_T4) | set(PILOT_SEEDS) | set(PILOT_SEEDS_T4)


def registered_overlap():
    """Phase 3 seeds among the SEEDS* / PILOT_SEEDS* constants of every src module that defines one."""
    import importlib
    import re
    mine = all_phase3_seeds()
    found = {}
    for p in sorted((ROOT / "src").glob("*.py")):
        if p.stem == "phase3_w4":
            continue
        if not re.search(r"^(SEEDS|PILOT_SEEDS)\w*\s*=", p.read_text(), re.M):
            continue
        try:
            m = importlib.import_module(f"src.{p.stem}")
        except Exception as e:                                            # noqa: BLE001
            found[p.stem] = f"import failed: {type(e).__name__}"
            continue
        for name in dir(m):
            if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
                found[f"{p.stem}.{name}"] = sorted(mine & set(_flat_ints(getattr(m, name))))
    return found


def scan():
    """Every Phase 3 range (prefixes 7430, 7431, 7432, 7439: every number 7,43x,000-999, also written with _ or ,) is
    absent from every text file under src/, tests/, results/, paper/, notes/, independent/, data/, dist/ (Phase 3's own
    files, the approved page and results/phase3/ excepted), and no Phase 3 seed is a registered or pilot seed of any
    module.  Disjoint from the exploration seeds 7,410,000-059."""
    OUT.mkdir(parents=True, exist_ok=True)
    pats = scan_patterns()
    hits = scan_tree(pats)
    ov = registered_overlap()
    bad_import = {k: v for k, v in ov.items() if isinstance(v, str)}
    out = {"ranges": {"Q": [SEEDS_Q[0], SEEDS_Q[-1], len(SEEDS_Q)], "S": [SEEDS_S[0], SEEDS_S[-1], len(SEEDS_S)],
                      "T4": [SEEDS_T4[0], SEEDS_T4[-1], len(SEEDS_T4)],
                      "pilot_Q_S": [PILOT_SEEDS[0], PILOT_SEEDS[-1], len(PILOT_SEEDS)],
                      "pilot_T4": [PILOT_SEEDS_T4[0], PILOT_SEEDS_T4[-1], len(PILOT_SEEDS_T4)]},
           "dirs": list(SCAN_DIRS), "patterns": pats, "pattern_files": hits, "registered_seed_overlap": ov,
           "modules_not_importable": bad_import,
           "exploration_seeds_disjoint": bool(not (all_phase3_seeds() & set(EXPLORATION_SEEDS))),
           "unused": bool(not any(hits.values())
                          and not any(v for v in ov.values() if isinstance(v, list) and v)
                          and not (all_phase3_seeds() & set(EXPLORATION_SEEDS)))}
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("pattern_files", "modules_not_importable", "unused")}), indent=1))
    if not out["unused"]:
        raise SystemExit("STOP: a Phase 3 seed range is not unused")
    return out


# ------------------------------------------------------------------------------------------ freeze (per seed)
def freeze_seed(arm, seed, land, rho):
    """Per seed: the own copy point (Newton from θ*_pop on the own sample; type, partition, unplaced), λ_min and W;
    the hold and the release classification (the winding fixed here); the release state and its SHA-256.  The arm's
    copy in FULL (validated switch, κ₀ and a, the follow point, the drive integral) for Q and S always, for T4 when the
    release is on T4.  With ρ: t* = ∫ds/D / (ρη) and B = min(⌈1.5·t*⌉ + 3000, 10⁶), over_cap.  No training."""
    t0 = time.time()
    x, y = own_sample(seed)
    v0 = np.array(land["v0"], float)
    c = COPY_OF[arm]
    rows = {}
    for cc in (c,) + POINT_COPIES[arm]:
        rows[cc] = own_point(cc, land["copies"][cc]["z_s0"], v0, x, y)
        rows[cc]["full"] = False
    lam_c = rows[c]["lam_min_s0"] if rows[c]["point_ok"] else None
    W = w_hold(lam_c, W_FLOOR[arm]) or W_FLOOR[arm]
    z_rel, rel = release(arm, seed, rows, W, land, x, y)
    full = rows[c]["point_ok"] and (arm != T4 or rel["on_copy"])
    if full:
        rows[c]["full"] = True
        sw = switch_of_copy(np.array(rows[c]["z_s0"]), v0, x, y, S_HI_FRAC * land["copies"][c]["s_pop_branch"])
        rows[c].update(sw)
        if sw["valid"]:
            rows[c]["follow_point"] = follow_point(np.array(rows[c]["z_s0"]), v0, sw["s_switch"], x, y)
            rows[c]["valid"] = bool(rows[c]["follow_point"]["valid"])
            if not rows[c]["valid"]:
                rows[c]["note"] = "follow point invalid"
    else:
        rows[c]["valid"] = False
        rows[c]["note"] = ("no accepted own point of the copy's type and partition at s0" if not rows[c]["point_ok"]
                           else "T4: release not on T4 (point only)")
    for cc in POINT_COPIES[arm]:
        rows[cc]["valid"] = False
        rows[cc]["note"] = "point only (landing counts)"
    fs = {"arm": arm, "seed": int(seed), "copies": rows, "W": int(W), "release": rel,
          "z_release": z_rel.tolist(), "z_release_sha256": array_sha(z_rel)}
    fs["budget"] = budget_record(fs, rho)
    fs["secs"] = round(time.time() - t0, 1)
    return _jsonable(fs)


def budget_record(fs, rho):
    cf = fs["copies"][COPY_OF[fs["arm"]]]
    if not cf.get("valid") or rho is None:
        return {"rho": rho, "B": None, "over_cap": None, "t_star": None}
    t_star = drive_steps(cf["drive_integral"], rho)
    return {"rho": float(rho), "eta": ETA, "drive_integral": cf["drive_integral"], "t_star": t_star,
            "B_uncapped": budget_uncapped(t_star), "B": budget_of(t_star), "over_cap": over_cap(t_star)}


def _pilot_rho():
    p = OUT / "pilot.json"
    return json.loads(p.read_text()) if p.exists() else None


def freeze(arm):
    """Registered seeds of one arm (resumable; the memory gate before every seed).  Needs pilot.json (ρ)."""
    _setup()
    land = _land()
    pj = _pilot_rho()
    assert pj is not None and pj["arms"][arm]["rule"]["status"] == "ok", "the pilot has not set ρ"
    rho = pj["arms"][arm]["rho"]
    f = OUT / "frozen_parts.jsonl"
    done = {(r["arm"], r["seed"]) for r in _rows(f)}
    for seed in SEEDS[arm]:
        if (arm, seed) in done:
            continue
        memory_gate(f"freeze {arm} {seed}")
        r = freeze_seed(arm, seed, land, rho)
        _append_write(f, r)
        c = r["copies"][COPY_OF[arm]]
        print(json.dumps({"arm": arm, "seed": seed, "on_copy": r["release"]["on_copy"], "valid": c.get("valid"),
                          "s_switch": c.get("s_switch"), "kappa0": c.get("kappa0"), "B": r["budget"].get("B"),
                          "W": r["W"], "secs": r["secs"]}), flush=True)
        rss_guard()


def _frozen_rows(arm=None):
    rows = _rows(OUT / "frozen_parts.jsonl")
    return [r for r in rows if arm is None or r["arm"] == arm]


def _dist(v):
    v = np.asarray([t for t in v if t is not None and np.isfinite(t)], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "min": float(v.min()), "q10": float(np.percentile(v, 10)),
            "median": float(np.median(v)), "q90": float(np.percentile(v, 90)), "max": float(v.max())}


def budget_condition(rows_by_arm):
    """The author's condition (2026-10-06): if MORE THAN 24 seeds in Q or in S would exceed the 10⁶-step cap, STOP
    before registering."""
    out = {}
    for arm in (Q, S):
        rs = rows_by_arm.get(arm, [])
        b = [r["budget"] for r in rs]
        n_over = int(sum(bool(x.get("over_cap")) for x in b))
        out[arm] = {"n_seeds": len(rs), "n_with_budget": int(sum(x.get("B") is not None for x in b)),
                    "n_over_cap": n_over, "max_allowed": BUDGET_CONDITION_MAX_OVER,
                    "B_distribution": _dist([x.get("B") for x in b]),
                    "B_uncapped_distribution": _dist([x.get("B_uncapped") for x in b]),
                    "t_star_distribution": _dist([x.get("t_star") for x in b]),
                    "ok": n_over <= BUDGET_CONDITION_MAX_OVER}
    out["pass"] = all(out[a]["ok"] for a in (Q, S))
    return out


def summary():
    """frozen.json: per arm the counts, the gate at the freeze, the budgets (and the author's budget condition), κ₀, W,
    the windings, and the random-hold collapse table (T4); the SHA-256 of frozen_parts.jsonl and of the per-seed listing."""
    rows = _frozen_rows()
    by = {a: sorted([r for r in rows if r["arm"] == a], key=lambda r: r["seed"]) for a in ARMS}
    for a in ARMS:
        assert [r["seed"] for r in by[a]] == list(SEEDS[a]), f"every {a} seed exactly once"
    pj = _pilot_rho()
    out = {"label": "PHASE 3 freeze (no training of any registered seed)", "s0": _land()["s0"],
           "frozen_parts_sha256": _sha(OUT / "frozen_parts.jsonl"), "pilot_sha256": _sha(OUT / "pilot.json"),
           "rho": {a: pj["arms"][a]["rho"] for a in ARMS}, "tau": {a: pj["arms"][a]["tau"] for a in ARMS},
           "pilot_median_chi_tsw": {a: pj["arms"][a]["pilot_median_chi_tsw"] for a in ARMS}, "arms": {}}
    listing = []
    for a in ARMS:
        rs = by[a]
        c = COPY_OF[a]
        rel = [r["release"] for r in rs]
        cps = [r["copies"][c] for r in rs]
        elig = [eligibility(r) for r in rs]
        g = gate(a, [x["on_copy"] for x in rel], [x["hold_G_positive"] for x in rel])
        k0 = [cp.get("kappa0") for cp in cps if cp.get("valid")]
        wins = {}
        for x in rel:
            if x.get("windings") is not None:
                wins[str(tuple(x["windings"]))] = wins.get(str(tuple(x["windings"])), 0) + 1
        out["arms"][a] = {
            "n_seeds": len(rs), "n_point_ok": int(sum(bool(cp.get("point_ok")) for cp in cps)),
            "n_on_copy_at_release": int(sum(bool(x["on_copy"]) for x in rel)),
            "n_hold_G_positive": int(sum(bool(x["hold_G_positive"]) for x in rel)),
            "n_inactive_unit": int(sum(not x.get("all_units_active", True) for x in rel)),
            "n_full_valid": int(sum(bool(cp.get("valid")) for cp in cps)),
            "n_on_copy_valid": int(sum(bool(x["on_copy"] and cp.get("valid")) for x, cp in zip(rel, cps))),
            "n_eligible_trained": int(sum(e[0] for e in elig)),
            "not_trained_reasons": {k: sum(1 for e in elig if not e[0] and e[1] == k) for k in sorted({e[1] for e in elig if not e[0]})},
            "windings_at_release": wins, "gate_at_freeze": g,
            "kappa0": _dist(k0), "n_kappa0_negative": int(sum(v < 0 for v in k0)),
            "s_switch": _dist([cp.get("s_switch") for cp in cps if cp.get("valid")]),
            "W": _dist([r["W"] for r in rs]),
            "t_star": _dist([r["budget"].get("t_star") for r in rs]),
            "B": _dist([r["budget"].get("B") for r in rs]),
            "n_over_cap": int(sum(bool(r["budget"].get("over_cap")) for r in rs)),
            "invalid_notes": {}}
        for cp in cps:
            if not cp.get("valid"):
                nt = cp.get("note", "invalid")
                out["arms"][a]["invalid_notes"][nt] = out["arms"][a]["invalid_notes"].get(nt, 0) + 1
        for r in rs:
            listing.append(f"{a} {r['seed']} {r['release']['copy_at_release']} {r['release']['windings']} "
                           f"{r['copies'][c].get('valid')} {r['budget'].get('B')} {r['z_release_sha256']}")
    out["collapse_random_holds_T4"] = collapse_table([r["release"] for r in by[T4]])
    out["budget_condition"] = budget_condition(by)
    out["listing_sha256"] = hashlib.sha256("\n".join(listing).encode()).hexdigest()
    (OUT / "frozen.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: v for k, v in out.items() if k != "arms"}), indent=1))
    for a in ARMS:
        print(a, json.dumps(_jsonable({k: out["arms"][a][k] for k in ("n_on_copy_at_release", "n_full_valid",
                                                                         "n_eligible_trained", "gate_at_freeze",
                                                                         "n_over_cap", "B", "kappa0")})))
    if not out["budget_condition"]["pass"]:
        raise SystemExit("STOP: the author's budget condition fails (more than 24 over-cap seeds in Q or S)")
    return out


# ------------------------------------------------------------------------------------------ the pilot (pilot seeds only)
def train_to_switch(fs, rho, land, chunk=5000):
    """Pilot: train from the frozen release in chunks until the own-path switch on the actual v path (or the branch is
    lost, or the budget).  Reads v only; no gap of a training state.  Returns (V, OwnSwitch)."""
    cf, rel = fs["copies"][COPY_OF[fs["arm"]]], fs["release"]
    k = tuple(rel["windings"])
    x, y = own_sample(fs["seed"])
    z = np.array(fs["z_release"], float)
    v = np.array(land["v0"], float)
    B = budget_of(drive_steps(cf["drive_integral"], rho))
    sw = OwnSwitch(shift(np.array(cf["z_s0"]), v, k), v, x, y)
    Vs = np.empty((B + 1, len(v)))
    Vs[0] = v
    t = 0
    while t < B and not sw.done:
        for _ in range(min(chunk, B - t)):
            t += 1
            g_z, g_v = grad(z, v, x, y)
            z = z - ETA * g_z
            v = v - rho * ETA * g_v
            Vs[t] = v
            sw.feed(v)
            if sw.done:
                break
    return Vs[:t + 1].copy(), sw, B


def pilot_switch_run(fs, rho, land):
    """One pilot run to t_sw: χ at t_sw and V7's window max χ_t (the pilot rule's value)."""
    t0 = time.time()
    ok, why = pilot_eligible(fs)
    rec = {"arm": fs["arm"], "seed": fs["seed"], "rho": rho, "on_copy": fs["release"]["on_copy"],
           "copy_at_release": fs["release"]["copy_at_release"], "windings": fs["release"]["windings"],
           "hold_n_G_pos": fs["release"]["hold_n_G_pos"]}
    if not ok:
        return {**rec, "status": why}
    cf = fs["copies"][COPY_OF[fs["arm"]]]
    V, sw, B = train_to_switch(fs, rho, land)
    s = np.abs(V).sum(axis=1)
    rec.update(budget=B, n_steps_run=int(len(V) - 1), t_sw=sw.t_sw, s_switch=sw.s_sw if sw.t_sw else None,
               branch_lost_at=sw.br.lost_at, s_switch_frozen=cf["s_switch"])
    if sw.t_sw is None:
        return {**rec, "status": "no switch within the budget", "secs": round(time.time() - t0, 1)}
    kap = kappa_winding(cf["kappa0"], cf["kappa_winding_coef"], tuple(fs["release"]["windings"]))
    r_cf, chi, _ = closed_form(s, sw.t_sw, sw.s_sw, kap, cf["lam_min_switch"])
    cp = chi_path(s, sw.br.lam, ETA, sw.t_sw)
    rec.update(kappa=kap, chi_tsw=chi, kc_tsw=r_cf, chi_window_max=chi_window_max(s, sw.t_sw, cp, sw.s_sw),
               status="ok", secs=round(time.time() - t0, 1))
    return rec


def pilot_eligible(fs):
    rel, cf = fs["release"], fs["copies"][COPY_OF[fs["arm"]]]
    if not rel.get("on_copy"):
        return False, "not on the arm's copy at release"
    if not cf.get("valid"):
        return False, "copy invalid"
    if not in_winding_table(tuple(rel["windings"])):
        return False, "winding outside the table"
    return True, "ok"


def pilot_freeze(land, arms=ARMS):
    """Pilot seeds frozen (no ρ yet; t* is recomputed per ρ): Q and S on 7,439,000-009 (branch-point starts); T4 on
    7,439,100 onward IN ORDER until 10 releases land on T4 (all drawn seeds counted)."""
    f = OUT / "pilot_frozen_parts.jsonl"
    done = {(r["arm"], r["seed"]): r for r in _rows(f)}
    out = {}
    for arm in arms:
        rows = []
        n_on = 0
        for seed in PILOT_RANGES[arm]:
            if arm == T4 and n_on >= PILOT_N_ON_COPY:
                break
            key = (arm, seed)
            if key in done:
                r = done[key]
            else:
                memory_gate(f"pilot freeze {arm} {seed}")
                r = freeze_seed(arm, seed, land, None)
                _append_write(f, r)
                print(json.dumps({"pilot_freeze": arm, "seed": seed, "on_copy": r["release"]["on_copy"],
                                  "valid": r["copies"][COPY_OF[arm]].get("valid"), "secs": r["secs"]}), flush=True)
                rss_guard()
            rows.append(r)
            n_on += int(bool(r["release"]["on_copy"]))
        out[arm] = rows
    return out


def pilot(arms=ARMS):
    """The pilot rules on pilot seeds only.  (1) ρ per arm: from the page's ρ, the on-copy pilot runs to t_sw (no gap
    of a state after release), q90 of V7's window max χ_t ≤ 0.1, else halve (≤ 3 halvings), else STOP.  (2) At the
    chosen ρ: the registered run_one and observe_one on the pilot runs -> τ_cross, τ_lag (1.5 × q90, rounded up to 5) and
    V6's pilot median χ at t_sw.  Resumable (pilot_parts.jsonl)."""
    _setup()
    land = _land()
    fz = pilot_freeze(land, arms)
    parts = OUT / "pilot_parts.jsonl"
    done = {(r["kind"], r["arm"], r["seed"], r["rho"]): r for r in _rows(parts)}
    res = {}
    for arm in arms:
        runs = {}

        def run_p(rho, arm=arm, runs=runs):
            vals, rows = [], []
            for fs in fz[arm]:
                key = ("switch", arm, fs["seed"], rho)
                if key in done:
                    r = done[key]
                else:
                    memory_gate(f"pilot {arm} {fs['seed']} rho={rho}")
                    r = {"kind": "switch", **pilot_switch_run(fs, rho, land)}
                    _append_write(parts, r)
                    done[key] = r
                    print(json.dumps(_jsonable(r)), flush=True)
                    rss_guard()
                rows.append(r)
                vv = r.get("chi_window_max")
                if r.get("status") == "ok" and vv is not None and np.isfinite(vv):
                    vals.append(vv)
            runs[rho] = rows
            return vals
        rule = pilot_rule(run_p, RHO_START[arm])
        entry = {"rule": rule, "rho": rule["rho"], "n_pilot_seeds_drawn": len(fz[arm]),
                 "n_on_copy": int(sum(bool(fs["release"]["on_copy"]) for fs in fz[arm])),
                 "seeds": [fs["seed"] for fs in fz[arm]]}
        if rule["status"] == "ok":
            final = runs[rule["rho"]]
            chis = [r["chi_tsw"] for r in final if r.get("status") == "ok" and r.get("chi_tsw") is not None]
            entry["pilot_median_chi_tsw"] = float(np.median(chis)) if chis else None
            entry["n_with_value"] = len(chis)
            errs = []
            for fs in fz[arm]:
                if not pilot_eligible(fs)[0]:
                    continue
                key = ("forecast", arm, fs["seed"], rule["rho"])
                if key in done:
                    r = done[key]
                else:
                    memory_gate(f"pilot forecast {arm} {fs['seed']}")
                    r = {"kind": "forecast", **pilot_forecast_run(fs, rule["rho"], land)}
                    _append_write(parts, r)
                    done[key] = r
                    print(json.dumps(_jsonable({k: r.get(k) for k in ("arm", "seed", "t_c", "t_fc", "t_obs",
                                                                        "err_cross", "err_lag", "nan_ok", "secs")})),
                          flush=True)
                    rss_guard()
                errs.append(r)
            ec = [r.get("err_cross") for r in errs]
            el = [r.get("err_lag") for r in errs]
            entry["tau"] = tolerances(ec, el)
            entry["forecast_runs"] = [{k: r.get(k) for k in ("seed", "t_c", "t_obs", "t_sw", "t_fc", "t_sw_fc",
                                                              "lag_obs", "lag_fc", "err_cross", "err_lag", "r_obs",
                                                              "r_fc", "nan_ok", "cutoff_before_crossing",
                                                              "follows_branch")} for r in errs]
            entry["DISCLOSURE_pilot_criteria_at_tau"] = pilot_criteria(arm, errs, entry["tau"])
        else:
            entry.update(pilot_median_chi_tsw=None, tau=None)
        res[arm] = entry
    prev = _pilot_rho() or {}
    out = {"label": "PHASE 3 pilot (pilot seeds only; no registered seed drawn)", "pilot_seeds_Q_S": list(PILOT_SEEDS),
           "pilot_seeds_T4_range": [PILOT_SEEDS_T4[0], PILOT_SEEDS_T4[-1]],
           "arms": {**prev.get("arms", {}), **res}}
    (OUT / "pilot.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({a: {k: v for k, v in r.items() if k in ("rho", "rule", "tau", "pilot_median_chi_tsw",
                                                                         "n_on_copy", "n_pilot_seeds_drawn")}
                                for a, r in res.items()}), indent=1))
    stops = [a for a in arms if res[a]["rule"]["status"] != "ok" or not res[a].get("tau")
             or res[a]["tau"].get("tau_cross") is None or res[a]["tau"].get("tau_lag") is None]
    if stops:
        raise SystemExit("STOP: the pilot sets no ρ or τ for " + ", ".join(stops))
    return out


def pilot_forecast_run(fs, rho, land):
    """A pilot run through the registered run_one (train to t_c, causal forecast, NaN recomputation) and observe_one
    (retraining, detection, the actual own-path switch, the follow check): the errors that set τ."""
    t0 = time.time()
    fs2 = dict(fs)
    fs2["budget"] = budget_record(fs, rho)
    rr = run_one(fs2, rho, land)
    rec = {"arm": fs["arm"], "seed": fs["seed"], "rho": rho, "t_c": rr.get("t_c"), "status": rr.get("status"),
           "budget": rr.get("budget")}
    if not rr.get("eligible"):
        return {**rec, "secs": round(time.time() - t0, 1)}
    ob = observe_one(fs2, rho, rr, land)
    fc = rr.get("forecast") or {}
    rec.update(t_fc=fc.get("t_fc"), t_sw_fc=fc.get("t_sw_fc"), lag_fc=fc.get("lag_fc"), r_fc=fc.get("r_fc"),
               forecast_status=fc.get("status"), nan_ok=fc.get("nan_recompute_identical"), t_obs=ob["t_obs"],
               t_sw=ob["t_sw"], follows_branch=ob.get("follows_branch"), chi_tsw=ob.get("chi_tsw"),
               chi_window_max=ob.get("chi_window_max"), r_cf=ob.get("r_cf"), kappa=ob.get("kappa"))
    if ob["t_obs"] is not None and ob["t_sw"] is not None:
        rec["lag_obs"] = ob["t_obs"] - ob["t_sw"]
        rec["r_obs"] = ob["s_obs"] / ob["s_sw"] - 1
    if fc.get("t_fc") is not None and ob["t_obs"] is not None:
        rec["err_cross"] = fc["t_fc"] - ob["t_obs"]
    if rec.get("lag_fc") is not None and rec.get("lag_obs") is not None:
        rec["err_lag"] = rec["lag_fc"] - rec["lag_obs"]
    rec["cutoff_before_crossing"] = bool(rr.get("t_c") is not None and ob["t_obs"] is not None
                                         and rr["t_c"] < ob["t_obs"])
    rec["secs"] = round(time.time() - t0, 1)
    return rec


def pilot_criteria(arm, errs, tau):
    """DISCLOSURE: C1-C4 (+ S) on the pilot runs at the pilot's own τ (the pilot has no gate or validity)."""
    rs = [r for r in errs if r.get("t_obs") is not None and r.get("t_sw") is not None and r.get("cutoff_before_crossing")]
    if not rs or tau.get("tau_cross") is None:
        return None
    g = lambda k: [r.get(k) for r in rs]                                     # noqa: E731
    c = criteria(arm, g("t_fc"), g("t_sw_fc"), g("r_fc"), g("t_obs"), g("t_sw"), g("r_obs"),
                 is_forecast(g("t_fc"), g("t_sw_fc")), tau["tau_cross"], tau["tau_lag"],
                 names=("C1", "C2", "C3", "C4", "S"))
    return {k: {kk: v[kk] for kk in v if kk in ("verdict", "frac_within", "median_ratio", "ci95", "frac", "n",
                                                  "n_eligible")} for k, v in c.items()}


# ------------------------------------------------------------------------------------------ registration
FROZEN_DATA = ("results/designs/phase3_width4_asym_design.md", "results/phase3_registration.md",
               "results/phase3/seed_scan.json", "results/phase3/landscape.json",
               "results/phase3/pilot_frozen_parts.jsonl", "results/phase3/pilot_parts.jsonl",
               "results/phase3/pilot.json", "results/phase3/frozen_parts.jsonl", "results/phase3/frozen.json",
               "tests/test_phase3_w4.py", "tests/test_causal_forecast.py", "results/asym_scores.json",
               "results/width2_asym/landscape.json",
               "results/designs/phase3_explore/w4core.py", "results/designs/phase3_explore/p3_activity.py",
               "results/designs/phase3_explore/test_p3_activity.py", "results/designs/phase3_explore/p3_budget.py",
               "results/designs/phase3_explore/p3_runs.py", "results/designs/phase3_explore/p3_runs.jsonl",
               "results/designs/phase3_explore/p3_own.py", "results/designs/phase3_explore/README.md")


def code_closure(start=("phase3_w4",)):
    """Every src module reached by relative imports (`from . import X`, `from .X import`), recursively (phase1c's rule)."""
    import ast
    seen, todo = set(), list(start)
    while todo:
        m = todo.pop()
        if m in seen or not (ROOT / "src" / f"{m}.py").exists():
            continue
        seen.add(m)
        tree = ast.parse((ROOT / "src" / f"{m}.py").read_text())
        for nd in ast.walk(tree):
            if isinstance(nd, ast.ImportFrom) and nd.level == 1:
                if nd.module:
                    todo.append(nd.module.split(".")[0])
                else:
                    todo.extend(a.name for a in nd.names)
    return sorted(f"src/{m}.py" for m in seen)


def manifest_files():
    return sorted(set(code_closure()) | set(FROZEN_DATA))


def manifest():
    """results/phase3/registration.sha256 (written for the registration commit)."""
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


def stamp():
    """registration_stamp.txt: the registration commit (the last commit touching the manifest) and the SHA-256 of the
    registration file and of the manifest."""
    _assert_committed(OUT / "registration.sha256")
    h = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(OUT / "registration.sha256")], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()
    txt = (f"Phase 3 registration\nregistration commit: {h}\n"
           f"sha256 results/phase3_registration.md: {_sha(REGISTRATION_MD)}\n"
           f"sha256 results/phase3/registration.sha256: {_sha(OUT / 'registration.sha256')}\n")
    (OUT / "registration_stamp.txt").write_text(txt)
    print(txt)


def assert_registration():
    m = OUT / "registration.sha256"
    for ln in m.read_text().splitlines():
        h, rel = ln.split()
        assert _sha(ROOT / rel) == h, f"registration hash mismatch: {rel}"
    _assert_committed(m)
    _assert_committed(OUT / "registration_stamp.txt")
    assert (OUT / "registration_stamp.txt.ots").exists(), "no OpenTimestamps proof yet: no registered training"


def _frozen_by_seed(arm):
    fr = json.loads((OUT / "frozen.json").read_text())
    assert fr["frozen_parts_sha256"] == _sha(OUT / "frozen_parts.jsonl"), "frozen_parts.jsonl changed"
    rows = {r["seed"]: r for r in _frozen_rows(arm)}
    assert sorted(rows) == list(SEEDS[arm])
    return fr, rows


def _runs_file(arm):
    return OUT / f"runs_{arm}.jsonl"


def _obs_file(arm):
    return OUT / f"observed_{arm}.jsonl"


def run(arm):
    """AFTER the registration commit is pushed and OpenTimestamped: every seed of the arm (resumable; memory gate
    before each).  Ineligible seeds (not on copy, invalid copy, winding outside the table, OVER CAP) get a counted row
    and are not trained."""
    assert_registration()
    _setup()
    fr, rows = _frozen_by_seed(arm)
    land = _land()
    rho = fr["rho"][arm]
    f = _runs_file(arm)
    done = {r["seed"] for r in _rows(f)}
    for seed in SEEDS[arm]:
        if seed in done:
            continue
        memory_gate(f"run {arm} {seed}")
        r = run_one(rows[seed], rho, land)
        _append_write(f, r)
        fc = r.get("forecast") or {}
        print(json.dumps(_jsonable({"arm": arm, "seed": seed, "status": r.get("status"), "t_c": r.get("t_c"),
                                    "t_fc": fc.get("t_fc"), "nan_ok": fc.get("nan_recompute_identical"),
                                    "secs": r.get("secs_run")})), flush=True)
        rss_guard()


def _keys(o):
    if isinstance(o, dict):
        return set(o) | {k for v in o.values() for k in _keys(v)}
    if isinstance(o, list):
        return {k for v in o for k in _keys(v)}
    return set()


def finalize():
    """forecasts.sha256 over the three runs files: every seed exactly once, no observed key.  COMMIT before observe."""
    lines = []
    for arm in ARMS:
        rows = _rows(_runs_file(arm))
        assert sorted(r["seed"] for r in rows) == list(SEEDS[arm]), f"every seed exactly once ({arm})"
        assert not (FORBIDDEN & _keys(rows)), f"an observed quantity in a runs row ({arm})"
        lines.append(f"{_sha(_runs_file(arm))}  runs_{arm}.jsonl")
    (OUT / "forecasts.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _assert_forecasts():
    hs = dict(reversed(ln.split()) for ln in (OUT / "forecasts.sha256").read_text().splitlines())
    for arm in ARMS:
        assert _sha(_runs_file(arm)) == hs[f"runs_{arm}.jsonl"], f"forecast hash mismatch ({arm})"
        _assert_committed(_runs_file(arm))
    _assert_committed(OUT / "forecasts.sha256")


def observe(arm):
    """AFTER the forecasts commit: eligible runs only (others counted from the runs file); resumable."""
    assert_registration()
    _assert_forecasts()
    _setup()
    fr, rows = _frozen_by_seed(arm)
    land = _land()
    rho = fr["rho"][arm]
    f = _obs_file(arm)
    done = {r["seed"] for r in _rows(f)}
    for rr in _rows(_runs_file(arm)):
        if rr["seed"] in done or not rr.get("eligible") or str(rr.get("status", "")).startswith("aborted"):
            continue
        memory_gate(f"observe {arm} {rr['seed']}")
        t0 = time.time()
        o = observe_one(rows[rr["seed"]], rho, rr, land)
        o["secs"] = round(time.time() - t0, 1)
        _append_write(f, o)
        print(json.dumps(_jsonable({k: o.get(k) for k in ("arm", "seed", "crossed", "t_obs", "t_sw", "follows_branch",
                                                          "secs")})), flush=True)
        rss_guard()


def table(arm, frozen_rows, runs, obs):
    """Arrays over ALL the arm's seeds for score_arm (NaN / False where undefined)."""
    rb = {r["seed"]: r for r in runs}
    ob = {o["seed"]: o for o in obs}
    keys = ("seed", "on_copy", "eligible", "over_cap", "aborted", "hold_positive", "crossed", "t_obs", "s_obs", "t_sw",
            "s_sw", "t_c", "t_fc", "t_sw_fc", "s_fc", "r_fc", "nan_identical", "follows", "r_cf", "kappa", "eta_lam",
            "lag_steps", "chi_tsw", "chi_win_max", "s_pop")
    R = {k: [] for k in keys}
    g = lambda d, k: d.get(k) if d.get(k) is not None else np.nan          # noqa: E731
    for seed in SEEDS[arm]:
        fs, r, o = frozen_rows[seed], rb.get(seed, {}), ob.get(seed, {})
        fc = r.get("forecast") or {}
        R["seed"].append(seed)
        R["on_copy"].append(bool(fs["release"]["on_copy"]))
        R["eligible"].append(bool(r.get("eligible", False)))
        R["over_cap"].append(bool(fs["budget"].get("over_cap")))
        R["aborted"].append(str(r.get("status", "")).startswith("aborted"))
        R["hold_positive"].append(bool(fs["release"]["hold_G_positive"]))
        R["crossed"].append(bool(o.get("crossed", False)))
        for k in ("t_obs", "s_obs", "t_sw", "s_sw", "r_cf", "kappa", "eta_lam", "chi_tsw", "s_pop"):
            R[k].append(g(o, k))
        R["lag_steps"].append(g(o, "lag_steps_pred"))
        R["chi_win_max"].append(g(o, "chi_window_max"))
        R["follows"].append(bool(o.get("follows_branch", False)))
        R["t_c"].append(g(r, "t_c"))
        for k in ("t_fc", "t_sw_fc", "s_fc", "r_fc"):
            R[k].append(g(fc, k))
        R["nan_identical"].append(bool(fc.get("nan_recompute_identical", True)) if r.get("t_c") is not None else True)
    return R


def score():
    assert_registration()
    _assert_forecasts()
    fr = json.loads((OUT / "frozen.json").read_text())
    res = {"label": "PHASE 3 (registered)", "forecasts_sha256": _sha(OUT / "forecasts.sha256"),
           "registration_sha256_file_sha256": _sha(OUT / "registration.sha256"),
           "collapse_random_holds_T4": fr["collapse_random_holds_T4"], "arms": {}}
    for arm in ARMS:
        _, rows = _frozen_by_seed(arm)
        R = table(arm, rows, _rows(_runs_file(arm)), _rows(_obs_file(arm)))
        res["arms"][arm] = score_arm(arm, R, fr["pilot_median_chi_tsw"][arm], fr["tau"][arm]["tau_cross"],
                                     fr["tau"][arm]["tau_lag"])
    (OUT / "scores.json").write_text(json.dumps(_jsonable(res), indent=1))
    print(json.dumps(_jsonable({a: v["outcome"] for a, v in res["arms"].items()})))


def main(argv):
    cmd = argv[1]
    if cmd == "gate":
        memory_gate(" ".join(argv[2:]) or "gate")
        print("memory gate OK", flush=True)
    elif cmd == "scan":
        scan()
    elif cmd == "landscape":
        _setup()
        landscape()
    elif cmd == "pilot":
        pilot(tuple(argv[2:]) or ARMS)
    elif cmd == "freeze":
        freeze(argv[2])
    elif cmd == "summary":
        summary()
    elif cmd == "manifest":
        manifest()
    elif cmd == "stamp":
        stamp()
    elif cmd == "run":
        run(argv[2])
    elif cmd == "finalize":
        finalize()
    elif cmd == "observe":
        observe(argv[2])
    elif cmd == "score":
        score()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv)
