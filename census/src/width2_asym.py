"""W2-A: a registered test of the lag law at width 2 on the asymmetric windows (three arms, separate verdicts).

Design (approved by the author 2026-09-29 with changes A-D; A and D approved 2026-09-29):
results/designs/width2_asym_design.md (draft with the width-2 background: ccce4fa; exploratory producers:
results/designs/width2_asym_explore/, whose w2core.py this module's numerics are taken from).  Registration text:
results/width2_asym_registration.md.  Tests of every decision rule: tests/test_width2_asym.py.

Setting.  a = 1.30, Δ = 0.4 (as T2-1-T2-3): I = [-0.8, 0.8] class 0, O = [-2.0, -1.2] ∪ [1.2, 2.4] class 1.  Width 2,
N = v₁f(α₁x + β₁) + v₂f(α₂x + β₂) + b, f(t) = t + a·sin t; q = (α₁, β₁, v₁, α₂, β₂, v₂, b) (width2_train's layout);
hidden z = (α₁, β₁, α₂, β₂, b), output v = (v₁, v₂), s = ‖v‖₁.  Each seed's own 400-point sample
asym_register.training_set(seed); mean BCE with logits; float64 numpy, analytic gradient and Hessian.
G₊ = min_O φ − max_I φ, φ = v₁f(t₁) + v₂f(t₂): the exact-extrema enclosure (width2_geometry.extrema on explicit windows;
no module global is changed), screened by a dense grid whose value is an upper bound on G₊.  Observation and hold:
placed iff the enclosure's lower end > 0.  Branch points (switches, own-path switch, predictions): placed iff the
enclosure midpoint > 0.  s_pop2 = √(s_lo·s_hi) of T2-1's validated bracket (asym_scores.json) = 0.44508; s₀ = s_pop2/2.

Arms (hold: v fixed at v₀, hidden GD at lr 1.0 for W steps; release: full-batch SGD, η on z, ρ·η on v):
  T   HEADLINE.  v₀ = s₀·(0.1, 0.9); hidden = coordinates 0, 1, 3, 4, 6 of the seed's U(−1, 1)⁷ draw (width2_train's
      draw, from a local torch Generator); η = 0.03; seeds 884,000-884,119; scored copy T; gate ≥ 60/120 on T.
  D   CONTROL.  v₀ = s₀·(½, ½); the same hidden draw; η = 0.3; the same seeds; scored copies D and D′ (D′ as GELU-T's
      mirror copy: it gets its own prediction); gate ≥ 60/120 on D or D′.
  Tp  T′ (change A).  v₀ = s₀·(0.1, 0.9); hidden start θ*_T′,pop(s₀); η = 0.03; seeds 884,200-884,319; scored copy T′;
      gate ≥ 108/120 on T′ and no run with G > 0 at the start or any hold step.
Copies (population points at v₀, landscape()): D = the unplaced duplicate with α > 0, D′ = the unplaced duplicate with
α < 0 (equal shares); T = the unplaced two-unit point with the larger-share unit's α > 0 and the smaller's α < 0, T′ =
the reverse (shares 0.1/0.9).  A copy is identified up to 2π windings (β_i + 2πk_i, b − 2π(k₁v₁ + k₂v₂)).

    python -m src.width2_asym landscape   # population copies and switches -> results/width2_asym/landscape.json
    python -m src.width2_asym freeze      # per seed and copy (resumable)   -> frozen_parts.jsonl, frozen_seeds.json
    python -m src.width2_asym pilot       # the pilot rules (pilot seeds)    -> pilot_parts.jsonl, pilot.json
    python -m src.width2_asym hashes      # registration.sha256
    python -m src.width2_asym train       # registered runs, predictions     -> predictions_parts.jsonl, paths/
    python -m src.width2_asym finalize    # predictions.csv + predictions.sha256
    python -m src.width2_asym observe     # AFTER the predictions commit: every-step detection, gates, scores
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "width2_asym"
PATHS = OUT / "paths"
ASYM_SCORES = RESULTS / "asym_scores.json"

A_ACT, DELTA = 1.30, 0.4
INNER = (-0.8, 0.8)
OUTER = ((-2.0, -1.2), (1.2, 2.0 + DELTA))
XI_DENSE = np.linspace(INNER[0], INNER[1], 801)
XO_DENSE = np.concatenate([np.linspace(*OUTER[0], 401), np.linspace(*OUTER[1], 401)])
ZI, VI = [0, 1, 3, 4, 6], [2, 5]
TWO_PI = 2 * math.pi
S_POP2_EXPECTED = 0.44507940623559955

SEEDS = tuple(range(884_000, 884_120))
SEEDS_TP = tuple(range(884_200, 884_320))
PILOT_SEEDS = tuple(range(884_900, 884_910))
T, D, TP = "T", "D", "Tp"
ARMS = (T, D, TP)
HEADLINE = T
ROLE = {T: "headline", D: "control", TP: "T-prime arm (change A; own verdict)"}
SHARES = {T: (0.1, 0.9), D: (0.5, 0.5), TP: (0.1, 0.9)}
ETA = {T: 0.03, D: 0.3, TP: 0.03}
ARM_SEEDS = {T: SEEDS, D: SEEDS, TP: SEEDS_TP}
SCORED_COPIES = {T: ("T",), D: ("D", "Dp"), TP: ("Tp",)}
CLASSIFY_COPIES = {T: ("T", "Tp"), D: ("D", "Dp"), TP: ("Tp", "T")}     # the copies a release is classified against
COPY_SHARES = {"T": SHARES[T], "Tp": SHARES[T], "D": SHARES[D], "Dp": SHARES[D]}
DUPLICATE_COPIES = ("D", "Dp")
OWN_PATH_SWITCH = {T: True, D: False, TP: True}      # T, T′: lag-free switch read from the run's own v path (change C)
COPY_NAMES = {"T": "T", "Tp": "T′", "D": "D", "Dp": "D′"}

# population copy points: damped Newton at v₀ on the population from the exploratory class points
# (designs/width2_asym_explore/w2_explore_a.json and w2_explore_a_s0.2225_u0.1.json, first row of each class)
POP_START = {"D": (1.641692, -1.708088, 1.641692, -1.708088, 0.296361),
             "Dp": (-1.340431, -1.723233, -1.340431, -1.723233, 0.483756),
             "T": (-1.707199, -1.450349, 1.670937, -1.692048, 0.296007),
             "Tp": (1.877688, -1.671324, -1.381723, -1.680716, 0.446559)}
# the page's population switches (D 5.080, T 0.482, T′ 0.328) and D′'s exploratory one (4.971), 3 digits: a recomputed
# switch more than 0.5% away from them stops the landscape (the exploratory adiabatic switches were not bisected: the
# exploration reported the first 0.5% step past the sign change, so they lie up to one step above the switch)
POP_SWITCH_PAGE = {"D": 5.080, "Dp": 4.971, "T": 0.482, "Tp": 0.328}
POP_SWITCH_REL = 5e-3

HOLD_LR, W_MIN, W_RELAX = 1.0, 4000, 25.0
BUDGET_D, BUDGET_T_BASE, RHO_T_START = 40_000, 100_000, 2.0 ** -10
NEWTON_GTOL, NEWTON_ITER_TOL, ON_TOL, STATE_TOL = 1e-8, 1e-12, 1e-6, 1e-3
DUP_TOL = 1e-6
FOLLOW_FRAC = 0.8
FOLLOW_STATE_TOL = None            # GELU-T registration §13 (carried over): the Newton condition only
S_HI_FRAC = 1.6
S_MAX_POP = 30.0
RAY_H, ADIAB_HS = 0.02, 0.005      # continuation steps: log s on the diagonal; relative s steps on the adiabatic path
HALVING_REL = 1e-6
DECIDE_REL = 1e-3
DG_ONESIDED_MAX = 1e-5
KSUM_RANGE = (-3, 3)
SDOT_WIN = 100
BRANCH_JUMP = 0.2                  # a Newton correction along the own path larger than this (sup) ends the branch
DIVERGED = 1.0                     # R4: sup|δ| > 1 is no prediction (Track A)
RADIUS_EVERY = 10

# registered criteria and conditions
GATE_MIN = {T: 60, D: 60, TP: 108}
GATE_HOLD_CONDITION = {T: False, D: False, TP: True}
L1_BAND, L2_BAND, L3_MIN = (0.90, 1.10), (0.80, 1.20), 0.5
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 884_000, (2.5, 97.5)
MIN_SCORED = 60
TSW_MIN_FRAC = 0.90
REGIME_MAX, REGIME_MIN_FRAC = 0.5, 0.80
LAG_MIN_STEPS = 10.0
KC_Q90_MAX = 0.1
CHI_REL_TOL = 0.30
CHI_PATH_Q90_MAX = 0.25
L5_NMIN = 6                        # change D: N_min = 2(g + τ + d) = 2(1 + 1 + 1) steps, on |t_pred − t_sw|
PILOT_KC_MAX, PILOT_CHI_MAX, PILOT_MAX_HALVINGS = 0.1, 0.1, 3

RSS_LIMIT = 3 * 1024 ** 3

REGISTERED_FILES = ("src/width2_asym.py", "tests/test_width2_asym.py", "results/width2_asym_registration.md",
                    "results/designs/width2_asym_design.md",
                    "results/width2_asym/landscape.json", "results/width2_asym/frozen_seeds.json",
                    "results/width2_asym/pilot.json", "results/asym_scores.json",
                    "src/width2_geometry.py", "src/asym_register.py", "src/asym_pilot.py", "src/width2_train.py",
                    "src/track_a.py", "src/act_fold.py")


# ------------------------------------------------------------------------------------------ numerics (numpy, analytic)
def _f(t):
    return t + A_ACT * np.sin(t), 1 + A_ACT * np.cos(t), -A_ACT * np.sin(t)


def _sig(n):
    return 0.5 * (1 + np.tanh(0.5 * n))


def logits(q, x):
    t1, t2 = q[0] * x + q[1], q[3] * x + q[4]
    return q[2] * (t1 + A_ACT * np.sin(t1)) + q[5] * (t2 + A_ACT * np.sin(t2)) + q[6]


def loss(q, x, y):
    n = logits(q, x)
    return float(np.mean(np.logaddexp(0, n) - y * n))


def _jac(q, x):
    t1, t2 = q[0] * x + q[1], q[3] * x + q[4]
    f1, d1, s1 = _f(t1)
    f2, d2, s2 = _f(t2)
    J = np.stack([q[2] * d1 * x, q[2] * d1, f1, q[5] * d2 * x, q[5] * d2, f2, np.ones_like(x)], 1)
    return J, q[2] * f1 + q[5] * f2 + q[6], (d1, s1, d2, s2)


def grad(q, x, y):
    J, n, _ = _jac(q, x)
    return (_sig(n) - y) @ J / len(x)


def hess(q, x, y):
    J, n, (d1, s1, d2, s2) = _jac(q, x)
    p = _sig(n)
    r = p - y
    H = (J * (p * (1 - p))[:, None]).T @ J
    for ia, ib, iv, d, s in ((0, 1, 2, d1, s1), (3, 4, 5, d2, s2)):
        v = q[iv]
        c = np.sum(r * v * s * x)
        H[ia, ia] += np.sum(r * v * s * x * x)
        H[ib, ib] += np.sum(r * v * s)
        H[ia, ib] += c
        H[ib, ia] += c
        e = np.sum(r * d * x)
        H[ia, iv] += e
        H[iv, ia] += e
        e = np.sum(r * d)
        H[ib, iv] += e
        H[iv, ib] += e
    return H / len(x)


def qof(z, v):
    q = np.empty(7)
    q[ZI] = z
    q[VI] = v
    return q


def gz(z, v, x, y):
    return grad(qof(z, v), x, y)[ZI]


def hz_blocks(z, v, x, y):
    H = hess(qof(z, v), x, y)
    return H[np.ix_(ZI, ZI)], H[np.ix_(ZI, VI)]


def newton(z0, v, x, y, maxit=100):
    """Damped Newton on ∇_z L(·, v) = 0 (backtracking on ‖∇‖), iterated to 1e−12.  Returns (z, max|∇|, λ_min of the
    z-Hessian, accepted); accepted iff max|∇| < 1e−8 and the z-Hessian is positive definite."""
    z = np.asarray(z0, float).copy()
    v = np.asarray(v, float)
    g = gz(z, v, x, y)
    for _ in range(maxit):
        if not np.all(np.isfinite(g)) or np.abs(g).max() < NEWTON_ITER_TOL:
            break
        H, _ = hz_blocks(z, v, x, y)
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
    H, _ = hz_blocks(z, v, x, y)
    lam = float(np.linalg.eigvalsh(0.5 * (H + H.T)).min()) if np.all(np.isfinite(H)) else float("nan")
    gm = float(np.abs(g).max()) if np.all(np.isfinite(g)) else float("nan")
    return z, gm, lam, bool(np.isfinite(gm) and gm < NEWTON_GTOL and np.isfinite(lam) and lam > 0)


def dz_dv(z, v, x, y):
    """Branch derivative ∂z*/∂v = −H_zz⁻¹H_zv (5 × 2)."""
    Hzz, Hzv = hz_blocks(z, v, x, y)
    return -np.linalg.solve(Hzz, Hzv)


# ------------------------------------------------------------------------------------------ the gap
_ACT = None


def act():
    global _ACT
    if _ACT is None:
        from .width2_geometry import Act
        _ACT = Act("fa", A_ACT)
    return _ACT


def _phi(xs, z, v):
    a = act()
    return v[0] * a.u(z[0] * xs + z[1]) + v[1] * a.u(z[2] * xs + z[3])


def gap_dense(z, v):
    """Dense-grid G₊ (an upper bound on the exact G₊: each grid is a subset of its window, endpoints included)."""
    return float(_phi(XO_DENSE, z, v).min() - _phi(XI_DENSE, z, v).max())


def gap_enclosure(z, v):
    """(lo, hi): enclosure of the exact G₊ on the asymmetric windows (width2_geometry.extrema on explicit intervals)."""
    from .width2_geometry import extrema
    th, vv, a = np.array([z[0], z[1], z[2], z[3]], float), np.asarray(v, float), act()
    _, _, ixl, ixh = extrema(th, vv, a, *INNER)
    oL, oR = extrema(th, vv, a, *OUTER[0]), extrema(th, vv, a, *OUTER[1])
    oml, omh = min(oL[0], oR[0]), min(oL[1], oR[1])
    return float(oml - ixh), float(omh - ixl)


def gap_mid(z, v):
    lo, hi = gap_enclosure(z, v)
    return 0.5 * (lo + hi)


def placed_state(z, v):
    """Observation and hold rule: (placed, undecided).  Placed iff the exact enclosure's lower end > 0 (screened: a dense
    value ≤ 0 means the exact G₊ ≤ 0); undecided iff the enclosure straddles 0 (counted, not placed)."""
    if gap_dense(z, v) <= 0:
        return False, False
    lo, hi = gap_enclosure(z, v)
    return bool(lo > 0), bool(lo <= 0 < hi)


def branch_placed(z, v):
    """Branch-point and prediction rule: the enclosure midpoint > 0 (screened by the dense value)."""
    return bool(gap_dense(z, v) > 0 and gap_mid(z, v) > 0)


# ------------------------------------------------------------------------------------------ symmetries and copies
def shift(z, v, k):
    """The 2π winding copy: β_i + 2πk_i, b − 2π(k₁v₁ + k₂v₂) (the same function up to the bias, the same loss)."""
    z = np.asarray(z, float).copy()
    z[1] += TWO_PI * k[0]
    z[3] += TWO_PI * k[1]
    z[4] -= TWO_PI * (k[0] * v[0] + k[1] * v[1])
    return z


def windings_to(z, zc):
    """The winding k with shift(zc, v, k) nearest z in the β coordinates."""
    return (int(round((z[1] - zc[1]) / TWO_PI)), int(round((z[3] - zc[3]) / TWO_PI)))


def is_duplicate(z, tol=DUP_TOL):
    """Units equal modulo 2π (α₁ = α₂, β₁ ≡ β₂): the width-1 function, for output weights of one sign."""
    db = z[1] - z[3]
    db -= TWO_PI * round(db / TWO_PI)
    return bool(abs(z[0] - z[2]) <= tol * max(1.0, abs(z[0])) and abs(db) <= tol)


def copy_type(z, v, placed):
    """Type of a stationary point at v > 0: 'D' / 'Dp' (duplicate, α > 0 / α < 0), 'T' / 'Tp' (two-unit: the
    larger-share unit's α > 0 and the smaller's α < 0 / the reverse), 'other'; ' placed' appended if placed."""
    if is_duplicate(z):
        t = "D" if z[0] > 0 else "Dp"
    else:
        big, small = (0, 2) if abs(v[0]) >= abs(v[1]) else (2, 0)
        sb, ss = np.sign(z[big]), np.sign(z[small])
        t = "T" if (sb > 0 > ss) else ("Tp" if (sb < 0 < ss) else "other")
    return t + (" placed" if placed else "")


def classify(z_state, z_newton, newton_ok, copies, v, state_tol=STATE_TOL, duplicate_required=()):
    """(copy, winding k) or (None, None).  copies = {name: the copy's point at v (canonical winding) or None}.  On copy c
    at winding k iff Newton was accepted, the Newton point is within 1e−6 (sup) of shift(z_c, v, k), the state is within
    state_tol of it (None: no state condition), and, for c in duplicate_required, the Newton point is a duplicate."""
    if not newton_ok:
        return None, None
    z_state, z_newton = np.asarray(z_state, float), np.asarray(z_newton, float)
    hits = []
    for c, zc in copies.items():
        if zc is None:
            continue
        k = windings_to(z_newton, np.asarray(zc, float))
        zk = shift(zc, v, k)
        if (np.abs(z_newton - zk).max() <= ON_TOL
                and (state_tol is None or np.abs(z_state - zk).max() <= state_tol)
                and (c not in duplicate_required or is_duplicate(z_newton))):
            hits.append((c, k))
    assert len(hits) <= 1, "a state cannot be on two copies"
    return hits[0] if hits else (None, None)


def split_basis(v):
    """Orthonormal basis (z coordinates) of a duplicate point's split subspace δ(α₁, β₁) = v₂e, δ(α₂, β₂) = −v₁e
    (first-order function-preserving)."""
    B = np.zeros((5, 2))
    B[0, 0], B[2, 0] = v[1], -v[0]
    B[1, 1], B[3, 1] = v[1], -v[0]
    return np.linalg.qr(B)[0]


def reduced_lam(H, v, duplicate):
    """(λ_min of the reduced Hessian, λ_min of the split block or None).  Duplicate: the split subspace is excluded
    (λ_min on its orthogonal complement); otherwise the full z-Hessian."""
    H = 0.5 * (np.asarray(H, float) + np.asarray(H, float).T)
    if not duplicate:
        return float(np.linalg.eigvalsh(H).min()), None
    B = split_basis(v)
    Q = np.linalg.qr(np.column_stack([B, np.eye(5)]))[0][:, 2:]
    return float(np.linalg.eigvalsh(Q.T @ H @ Q).min()), float(np.linalg.eigvalsh(B.T @ H @ B).min())


# ------------------------------------------------------------------------------------------ κ (P = I, total derivative)
def kappa_formula(H, tan, dGz, dGv_along, lam):
    """κ = λ_min·[∇_zG·H⁻¹θ*′]/[dG/ds], dG/ds = ∇_zG·θ*′ + ∂_vG·v′ (the total derivative of the branch gap along the
    path; change B).  Signed.  Returns (κ, u = H⁻¹∇_zG, dG/ds)."""
    H, tan, dGz = (np.asarray(a, float) for a in (H, tan, dGz))
    u = np.linalg.solve(H, dGz)
    den = float(dGz @ tan + dGv_along)
    return float(lam * (u @ tan) / den), u, den


def kappa_winding(kap0, coef, k):
    """κ at winding k = κ₀ + a₁k₁ + a₂k₂.  Exact: the winding leaves H, ∇_zG and ∂_vG unchanged and moves the tangent
    by −2π(k·v′) in b, so a_i = −2π·λ_min·u_b·v′_i/[dG/ds].  For D (v′ = (½, ½)) it depends on k₁ + k₂ only."""
    return float(kap0 + coef[0] * k[0] + coef[1] * k[1])


def in_winding_table(k):
    """The page's winding table: k₁ + k₂ ∈ −3..3."""
    return k is not None and KSUM_RANGE[0] <= k[0] + k[1] <= KSUM_RANGE[1]


def grad_gap(z, v, vprime, h=1e-6):
    """∇_zG (central differences of the enclosure midpoint in α₁, β₁, α₂, β₂; b does not enter G), ∂_vG·v′ (central,
    along v′), and the largest forward-backward disagreement over the five derivatives relative to their sup norm."""
    g0 = gap_mid(z, v)
    out, dis = np.zeros(5), []
    for i in range(4):
        e = np.zeros(5)
        e[i] = h
        gp, gm = gap_mid(z + e, v), gap_mid(z - e, v)
        out[i] = (gp - gm) / (2 * h)
        dis.append(abs((gp - g0) - (g0 - gm)) / h)
    vp = np.asarray(vprime, float)
    gp, gm = gap_mid(z, np.asarray(v) + h * vp), gap_mid(z, np.asarray(v) - h * vp)
    dv = (gp - gm) / (2 * h)
    dis.append(abs((gp - g0) - (g0 - gm)) / h)
    scale = max(float(np.abs(out).max()), abs(dv), 1e-300)
    return out, float(dv), float(max(dis) / scale)


def kappa_at(z, v, vprime, x, y, duplicate):
    """Frozen κ quantities at a switch point with path direction v′ = dv/ds: κ₀ (winding 0), the winding coefficients,
    κ over the winding table (k₁ + k₂ = −3..3 with (k₁ + k₂, 0); D's values depend on the sum only), reduced λ_min,
    the split block's λ_min (D) and the parts of dG/ds."""
    Hzz, Hzv = hz_blocks(z, v, x, y)
    vprime = np.asarray(vprime, float)
    tan = -np.linalg.solve(Hzz, Hzv @ vprime)
    lam, lam_split = reduced_lam(Hzz, v, duplicate)
    dGz, dGv, dis = grad_gap(z, v, vprime)
    kap0, u, den = kappa_formula(Hzz, tan, dGz, dGv, lam)
    coef = [float(-TWO_PI * lam * u[4] * vprime[i] / den) for i in range(2)]
    return {"kappa0": kap0, "kappa_winding_coef": coef, "lam_min_switch": lam, "lam_split_switch": lam_split,
            "dGz_dot_tangent": float(dGz @ tan), "dGv_along": dGv, "dG_ds": den, "gradG_z": dGz.tolist(),
            "gradG_onesided_rel": dis, "gradG_ok": bool(dis <= DG_ONESIDED_MAX), "tangent": tan.tolist(),
            "vprime": vprime.tolist(),
            "kappa_table_ksum": {str(ks): kappa_winding(kap0, coef, (ks, 0))
                                 for ks in range(KSUM_RANGE[0], KSUM_RANGE[1] + 1)}}


# ------------------------------------------------------------------------------------------ rules (pure functions)
def w_hold(lam):
    """max(4000, ⌈25/(1.0·λ_min)⌉); None for λ ≤ 0 or non-finite."""
    if lam is None or not np.isfinite(lam) or lam <= 0:
        return None
    return int(max(W_MIN, math.ceil(W_RELAX / (HOLD_LR * lam))))


def arm_hold_steps(arm, W_by_copy):
    """W for an arm's hold: the largest W over the arm's scored copies (D: D and D′, the copy is not known before
    release; T: T; T′: T′), over the copies with an accepted own point; the floor 4000 if none has one."""
    ws = [W_by_copy.get(c) for c in SCORED_COPIES[arm]]
    ws = [w for w in ws if w is not None]
    return int(max(ws)) if ws else W_MIN


def budget(arm, rho):
    """D: 40,000 steps; T, T′: 100,000·2⁻¹⁰/ρ."""
    return BUDGET_D if arm == D else int(round(BUDGET_T_BASE * RHO_T_START / rho))


def is_on_branch(arm, copy):
    return copy in SCORED_COPIES[arm]


def first_ge(s, level, start=0):
    idx = np.nonzero(np.asarray(s, float)[start:] >= level)[0]
    return None if len(idx) == 0 else int(idx[0] + start)


def closed_form(s_path, t_sw, s_sw, kappa, lam_sw, eta):
    """(r_cf, χ, ṡ): χ = (ṡ/s_switch)/(η·λ_min), ṡ = (s_{t_sw} − s_{t_sw−w})/w, w = min(100, t_sw); r_cf = κ·χ.  NaN
    if t_sw is undefined or 0."""
    if t_sw is None or t_sw < 1 or not np.isfinite(s_sw):
        return float("nan"), float("nan"), float("nan")
    s = np.asarray(s_path, float)
    w = min(SDOT_WIN, int(t_sw))
    sdot = (s[t_sw] - s[t_sw - w]) / w
    chi = (sdot / s_sw) / (eta * lam_sw)
    return float(kappa * chi), float(chi), float(sdot)


def chi_path(s_path, lam, eta, t_end):
    """χ_t = ((s_{t+1} − s_t)/s_t)/(η·λ_min,t), t = 0 … t_end − 1 (Corollary L3's χ_t with the one-step ṡ; λ_min the
    reduced λ_min at the branch point z*(v_t))."""
    s = np.asarray(s_path, float)
    lam = np.asarray(lam, float)
    n = min(int(t_end), len(s) - 1, len(lam))
    return (s[1:n + 1] - s[:n]) / s[:n] / (eta * lam[:n])


def chi_window_max(s_path, t_sw, chi, s_sw, frac=FOLLOW_FRAC):
    """V7's statistic (and T's and T′'s pilot statistic): max χ_t over 0 ≤ t < t_sw with s_t ≥ 0.8·s_switch.  NaN if
    t_sw is undefined, the window is empty or a χ_t in it is not finite."""
    if t_sw is None or t_sw < 1 or len(chi) < int(t_sw):
        return float("nan")
    s = np.asarray(s_path, float)[:int(t_sw)]
    w = np.asarray(chi, float)[:int(t_sw)][s >= frac * s_sw]
    return float(w.max()) if len(w) and np.all(np.isfinite(w)) else float("nan")


def follows_branch(copy_release, copy_follow):
    return bool(copy_release is not None and copy_follow == copy_release)


def l5_eligible(t_pred, t_sw, nmin=L5_NMIN):
    """Change D with the author's clarification (2): a run enters L5 only if |t_pred − t_sw| ≥ N_min = 6 steps (the
    ABSOLUTE predicted lag; a negative lag is treated exactly as a positive one)."""
    t_pred, t_sw = np.asarray(t_pred, float), np.asarray(t_sw, float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(t_pred) & np.isfinite(t_sw) & (np.abs(t_pred - t_sw) >= nmin)


def q90(x):
    x = np.asarray(x, float)
    if len(x) == 0 or not np.all(np.isfinite(x)):
        return float("nan")
    return float(np.percentile(x, 90))


def pilot_rho_D(q):
    """GELU-T's: ρ = min(1, 2^⌊log₂(0.1/q90)⌋); q90 ≤ 0 gives 1."""
    if q <= 0:
        return 1.0
    return float(min(1.0, 2.0 ** math.floor(math.log2(PILOT_KC_MAX / q))))


def pilot_rule_D(run_pilot):
    """GELU-T's pilot rule (arm D): run_pilot(ρ) → the on-branch pilot runs' SIGNED κ·χ at t_sw.  At ρ = 1:
    ρ = pilot_rho_D(q90); if ρ < 1, rerun at ρ and keep it if q90 ≤ 0.1, else halve (at most three halvings), then
    STOP.  No pilot value: STOP."""
    hist = []
    kc = np.asarray(run_pilot(1.0), float)
    q = q90(kc)
    hist.append({"rho": 1.0, "n": int(len(kc)), "q90": q})
    if not np.isfinite(q):
        return {"status": "STOP", "rho": None, "history": hist, "reason": "no pilot value"}
    rho = pilot_rho_D(q)
    if rho == 1.0:
        return {"status": "ok", "rho": 1.0, "history": hist}
    for k in range(PILOT_MAX_HALVINGS + 1):
        kc = np.asarray(run_pilot(rho), float)
        q = q90(kc)
        hist.append({"rho": rho, "n": int(len(kc)), "q90": q})
        if np.isfinite(q) and q <= PILOT_KC_MAX:
            return {"status": "ok", "rho": rho, "history": hist}
        if k == PILOT_MAX_HALVINGS:
            break
        rho /= 2
    return {"status": "STOP", "rho": None, "history": hist, "reason": "q90 > 0.1 after three halvings"}


def pilot_rule_T(run_pilot):
    """T and T′ (approved page): from ρ = 2⁻¹⁰, run_pilot(ρ) → the on-branch pilot runs' window max χ_t (V7's
    statistic); keep ρ if q90 ≤ 0.1, else halve and rerun, at most three halvings, then STOP.  No pilot value: STOP."""
    hist = []
    rho = RHO_T_START
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


def bootstrap_mean_ci(Dv, n_boot=BOOT_N, seed=BOOT_SEED):
    Dv = np.asarray(Dv, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(Dv), size=(n_boot, len(Dv)))
    means = Dv[idx].mean(axis=1)
    lo, hi = np.percentile(means, BOOT_PCT)
    return float(Dv.mean()), float(lo), float(hi)


def gate(arm, on_branch, hold_positive):
    """T: ≥ 60/120 on T; D: ≥ 60/120 on D or D′ (both regardless of G in the hold); T′: ≥ 108/120 on T′ AND no run
    with G > 0 at the start state or any hold step."""
    on_branch = np.asarray(on_branch, bool)
    hold_positive = np.asarray(hold_positive, bool)
    n_on = int(on_branch.sum())
    ok_count = bool(n_on >= GATE_MIN[arm])
    ok_hold = not bool(hold_positive.any())
    return {"n_runs": int(len(on_branch)), "n_on_branch": n_on, "min_on_branch": GATE_MIN[arm],
            "n_hold_G_positive": int(hold_positive.sum()), "count_ok": ok_count, "hold_ok": ok_hold,
            "hold_condition_applies": GATE_HOLD_CONDITION[arm],
            "pass": bool(ok_count and (ok_hold or not GATE_HOLD_CONDITION[arm]))}


def score_arm(arm, on_branch, hold_positive, crossed, step_obs, s_obs, s_sw, s_traj, t_traj, r_traj, r_cf, kappa, t_sw,
              eta_lam, lag_steps, chi_tsw, chi_win_max, pilot_chi_median, s_pop_branch, follows=None, exclude=None):
    """Registered verdicts for one arm (arrays over the arm's 120 runs; NaN where undefined).

    Gate first (fail: every criterion UNRESOLVED, nothing scored).  A run is scored if it is on a scored copy at release,
    follows its branch, crossed, has finite r_traj, r_cf, r_obs and s_traj, and is not in `exclude` (the DESCRIPTIVE
    sensitivity analysis only).  All lags SIGNED.  V1-V7 as GELU-T with V1 ≥ 60/120 (any fails: L1-L5 UNRESOLVED).
    L4: D = |log(s_obs/s_traj)| − |log(s_obs/s_pop,branch)| (the run's branch's population switch).  L5: the same against
    s_switch (the lag-free own switch) over the scored runs with |t_traj − t_sw| ≥ 6 (change D); UNRESOLVED if fewer
    than half the scored runs, or fewer than 2, remain."""
    A = np.asarray
    on_branch, hold_positive, crossed = A(on_branch, bool), A(hold_positive, bool), A(crossed, bool)
    (step_obs, s_obs, s_sw, s_traj, t_traj, r_traj, r_cf, kappa, t_sw, eta_lam, lag_steps, chi_tsw, chi_win_max,
     s_pop_branch) = (A(v, float) for v in (step_obs, s_obs, s_sw, s_traj, t_traj, r_traj, r_cf, kappa, t_sw, eta_lam,
                                            lag_steps, chi_tsw, chi_win_max, s_pop_branch))
    names = ("L1", "L2", "L3", "L4", "L5")
    follows = np.ones(len(on_branch), bool) if follows is None else A(follows, bool)
    exclude = np.zeros(len(on_branch), bool) if exclude is None else A(exclude, bool)
    g = gate(arm, on_branch, hold_positive)
    out = {"arm": arm, "role": ROLE[arm], "gate": g}
    if not g["pass"]:
        for k in names:
            out[k] = {"verdict": "UNRESOLVED"}
        out.update(valid=False, outcome="UNRESOLVED (gate)", scored_index=[])
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = s_obs / s_sw - 1
    scored = (on_branch & follows & ~exclude & crossed & np.isfinite(r_traj) & np.isfinite(r_cf) & np.isfinite(r_obs)
              & np.isfinite(s_traj))
    n = int(scored.sum())
    with np.errstate(invalid="ignore"):
        kpos = on_branch & ~exclude & crossed & np.isfinite(kappa) & (kappa > 0)
        tsw_before = kpos & np.isfinite(t_sw) & np.isfinite(step_obs) & (t_sw < step_obs)
    frac_tsw = float(tsw_before.sum() / kpos.sum()) if kpos.sum() else float("nan")

    def S(v):
        return v[scored]

    def med(v):
        return float(np.median(v)) if len(v) else float("nan")
    regime_frac = float(np.mean(S(eta_lam) <= REGIME_MAX)) if n else float("nan")
    lag_med = med(S(lag_steps))
    kc_q90 = q90(S(r_cf))
    chi_med = med(S(chi_tsw))
    chi_rel = (abs(chi_med / pilot_chi_median - 1) if n and pilot_chi_median is not None
               and np.isfinite(pilot_chi_median) and pilot_chi_median != 0 else float("nan"))
    chi_path_q90 = q90(S(chi_win_max))
    V = {"V1_min_scored_60": n >= MIN_SCORED,
         "V2_tsw_before_crossing_90pct_kappa_pos": bool(kpos.sum() == 0 or frac_tsw >= TSW_MIN_FRAC),
         "V3_regime_eta_lam_le_0p5_in_80pct": bool(n and regime_frac >= REGIME_MIN_FRAC),
         "V4_median_predicted_lag_ge_10_steps": bool(np.isfinite(lag_med) and lag_med >= LAG_MIN_STEPS),
         "V5_q90_kappa_chi_le_0p1": bool(np.isfinite(kc_q90) and kc_q90 <= KC_Q90_MAX),
         "V6_median_chi_within_30pct_of_pilot": bool(np.isfinite(chi_rel) and chi_rel <= CHI_REL_TOL),
         "V7_q90_max_chi_window_le_0p25": bool(np.isfinite(chi_path_q90) and chi_path_q90 <= CHI_PATH_Q90_MAX)}
    valid = all(V.values())
    elig = scored & l5_eligible(t_traj, t_sw)
    out.update({"n_runs": int(len(on_branch)), "n_on_branch": int(on_branch.sum()),
                "n_off_branch": int((~on_branch).sum()), "n_crossed": int(crossed.sum()),
                "n_on_branch_not_following": int((on_branch & ~follows).sum()),
                "n_on_branch_hold_G_positive": int((on_branch & hold_positive).sum()),
                "n_excluded_sensitivity": int((on_branch & exclude).sum()),
                "n_on_branch_crossed": int((on_branch & crossed).sum()), "n_scored": n,
                "n_on_branch_crossed_not_scored": int((on_branch & crossed & ~scored).sum()),
                "n_kappa_nonpos_scored": int((scored & ~(kappa > 0)).sum()),
                "n_kappa_pos_crossing": int(kpos.sum()), "frac_tsw_before_crossing_kappa_pos": frac_tsw,
                "regime_frac": regime_frac, "median_predicted_lag_steps": lag_med, "q90_kappa_chi_tsw": kc_q90,
                "median_chi_tsw": chi_med, "pilot_median_chi_tsw": pilot_chi_median, "chi_rel_to_pilot": chi_rel,
                "q90_max_chi_window": chi_path_q90, "n_L5_eligible": int(elig.sum()),
                "validity": {k: bool(v) for k, v in V.items()}, "valid": bool(valid),
                "scored_index": np.nonzero(scored)[0].tolist()})
    ro, rt, rc = S(r_obs), S(r_traj), S(r_cf)
    so, sp = S(s_obs), S(s_traj)

    def verdict(ok, computable):
        if not valid or not computable:
            return "UNRESOLVED"
        return "PASS" if ok else "FAIL"
    m1 = med(ro / rt) if n else float("nan")
    m2 = med(ro / rc) if n else float("nan")
    out["L1"] = {"n": n, "median_ratio": m1, "verdict": verdict(L1_BAND[0] <= m1 <= L1_BAND[1], n >= 1)}
    out["L2"] = {"n": n, "median_ratio": m2, "verdict": verdict(L2_BAND[0] <= m2 <= L2_BAND[1], n >= 1)}
    from .track_a import spearman
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = spearman(rt, ro) if n >= 2 else float("nan")
    out["L3"] = {"n": n, "spearman": rho, "verdict": verdict(rho >= L3_MIN, bool(np.isfinite(rho)))}
    if n >= 2:
        D4 = np.abs(np.log(so / sp)) - np.abs(np.log(so / S(s_pop_branch)))
        mean, lo, hi = bootstrap_mean_ci(D4)
    else:
        mean, lo, hi = (float("nan"),) * 3
    out["L4"] = {"n": n, "mean_D": mean, "ci95": [lo, hi], "verdict": verdict(hi < 0, bool(np.isfinite(hi))),
                 "comparator": "the run's branch's population switch"}
    ne = int(elig.sum())
    enough = bool(ne >= 2 and ne >= 0.5 * n)
    if enough:
        so5, sp5, ss5 = s_obs[elig], s_traj[elig], s_sw[elig]
        D5 = np.abs(np.log(so5 / sp5)) - np.abs(np.log(so5 / ss5))
        mean, lo, hi = bootstrap_mean_ci(D5)
    else:
        mean, lo, hi = (float("nan"),) * 3
    out["L5"] = {"n": ne, "n_scored": n, "mean_D": mean, "ci95": [lo, hi],
                 "verdict": verdict(hi < 0, bool(enough and np.isfinite(hi))),
                 "comparator": "s_switch (the lag-free own switch)", "resolution_rule": f"|t_pred - t_sw| >= {L5_NMIN}",
                 "enough_eligible": enough}
    vs = [out[k]["verdict"] for k in names]
    if not valid:
        out["outcome"] = "UNRESOLVED (validity)"
    elif "UNRESOLVED" in vs:
        out["outcome"] = "UNRESOLVED"
    elif all(v == "PASS" for v in vs):
        out["outcome"] = "PASS"
    else:
        out["outcome"] = "FAIL " + "+".join(k for k in names if out[k]["verdict"] == "FAIL")
    return out


# ------------------------------------------------------------------------------------------ the branch on the own v path
class OwnBranch:
    """z*(v_t) along a run's own v path, computed lazily by warm-started damped Newton from the release copy (predictor
    z*(v_{t−1}) + ∂z*/∂v·(v_t − v_{t−1})).  The branch is lost at the first step whose Newton is not accepted or whose
    correction exceeds 0.2 (sup); nothing is computed beyond it.  Stores z*, the z-Hessian, the reduced λ_min and (D)
    the split block's λ_min.  `V` may be replaced by a longer path (the pilot trains in chunks)."""

    def __init__(self, V, z_anchor, x, y, duplicate):
        self.V = np.asarray(V, float)
        self.x, self.y, self.dup = x, y, bool(duplicate)
        self.Z, self.H, self.lam, self.lam_split = [], [], [], []
        self.lost_at = None
        self._hzv = None
        z, _, _, ok = newton(z_anchor, self.V[0], x, y)
        if ok:
            self._append(z, 0)
        else:
            self.lost_at = 0

    def _append(self, z, t):
        Hzz, Hzv = hz_blocks(z, self.V[t], self.x, self.y)
        lr, ls = reduced_lam(Hzz, self.V[t], self.dup)
        self.Z.append(z)
        self.H.append(Hzz)
        self.lam.append(lr)
        self.lam_split.append(ls)
        self._hzv = Hzv

    def ensure(self, t):
        """True iff z*(v_t) is available (computed up to t if needed)."""
        t = int(t)
        while len(self.Z) <= t:
            k = len(self.Z)
            if self.lost_at is not None or k >= len(self.V):
                return False
            pred = self.Z[-1] + (-np.linalg.solve(self.H[-1], self._hzv)) @ (self.V[k] - self.V[k - 1])
            zn, _, _, ok = newton(pred, self.V[k], self.x, self.y)
            if not ok or np.abs(zn - pred).max() > BRANCH_JUMP:
                self.lost_at = k
                return False
            self._append(zn, k)
        return True

    def point(self, v, z_guess):
        z, _, _, ok = newton(z_guess, v, self.x, self.y)
        return z, ok


def own_path_switch(br, t_max, placed_fn=branch_placed, gap_fn=gap_mid, t_from=1):
    """THE LAG-FREE SWITCH ON THE RUN'S OWN v PATH (change C; frozen and hashed; T and T′).  Uses the v path and the
    release copy only (no gap of the training state).  t_sw = the first step t ≥ 1 (t ≤ t_max) at which the branch
    point z*(v_t) is placed (placed_fn); s_switch = ‖v‖₁ at the root of the branch gap (gap_fn) on the straight segment
    v_{t_sw−1} → v_{t_sw}, bisected in the segment parameter to 1e−12 with Newton warm-started from the last unplaced
    point.  (None, nan) if the branch is lost or has no placed point up to t_max."""
    for t in range(max(1, t_from), int(t_max) + 1):
        if not br.ensure(t):
            return None, float("nan")
        if placed_fn(br.Z[t], br.V[t]):
            lo, hi, zl = 0.0, 1.0, br.Z[t - 1]
            v0, v1 = br.V[t - 1], br.V[t]
            while hi - lo > 1e-12:
                mid = 0.5 * (lo + hi)
                vm = v0 + mid * (v1 - v0)
                zm, okm = br.point(vm, zl)
                if okm and gap_fn(zm, vm) <= 0:
                    lo, zl = mid, zm
                else:
                    hi = mid
            vr = v0 + 0.5 * (lo + hi) * (v1 - v0)
            return int(t), float(np.abs(vr).sum())
    return None, float("nan")


def r4_recursion(br, z_release, eta, t_max, placed_fn=branch_placed):
    """Track A's R4 recursion, SGD form (P = I, no momentum), along the run's own v path: δ₀ = z_release − z*(v₀);
    δ_{t+1} = (I − ηH_t)δ_t − (z*(v_{t+1}) − z*(v_t)); the predicted crossing is the first t ≥ 1 with z*(v_t) + δ_t
    placed (placed_fn at v_t).  No prediction if the one-step map's spectral radius exceeds 1 (checked at every 10th
    step) or sup|δ| > 1, or the branch is lost first.  Returns (t_hit or None, status, max|δ|, max radius)."""
    if not br.ensure(0):
        return None, "branch lost", float("nan"), float("nan")
    d = np.asarray(z_release, float) - br.Z[0]
    mx, rad = float(np.abs(d).max()), 0.0
    for t in range(0, int(t_max) + 1):
        if not br.ensure(t):
            return None, "branch lost", mx, rad
        if t % RADIUS_EVERY == 0:
            rad = max(rad, float(np.abs(1.0 - eta * np.linalg.eigvalsh(0.5 * (br.H[t] + br.H[t].T))).max()))
            if rad > 1.0:
                return None, "unstable", mx, rad
        if t >= 1 and placed_fn(br.Z[t] + d, br.V[t]):
            return t, "ok", mx, rad
        if t == t_max:
            break
        if not br.ensure(t + 1):
            return None, "branch lost", mx, rad
        d = d - eta * (br.H[t] @ d) - (br.Z[t + 1] - br.Z[t])
        mx = max(mx, float(np.abs(d).max()))
        if not np.isfinite(mx) or mx > DIVERGED:
            return None, "diverged", mx, rad
    return None, "no predicted crossing within the budget", mx, rad


# ------------------------------------------------------------------------------------------ helpers
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")


def _memory_guard(tag=""):
    """The machine's memory gate (free ≥ 25%, swap free ≥ 500 MB; act_fold.check_memory).  Pauses (re-checking every
    60 s) while it fails; every check is logged to results/width2_asym/memory_gate.log."""
    from .act_fold import check_memory
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            ok, f, w = check_memory()
        except Exception:                                                 # noqa: BLE001
            ok, f, w = False, float("nan"), float("nan")
        with open(OUT / "memory_gate.log", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB {'OK' if ok else 'WAIT'}\n")
        if ok:
            return f, w
        time.sleep(60)


def _setup_process():
    import os
    os.nice(15)                         # threads: OMP_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1 in the launching shell


def s_pop2():
    """√(s_lo·s_hi) of T2-1's validated bracket (asym_scores.json)."""
    d = json.loads(ASYM_SCORES.read_text())
    v = math.sqrt(d["s_lo"] * d["s_hi"])
    assert abs(v - S_POP2_EXPECTED) <= 1e-15, "asym_scores.json bracket differs"
    return v


def s0_value():
    return 0.5 * s_pop2()


def v_held(shares, s0=None):
    s0 = s0_value() if s0 is None else s0
    return np.array([s0 * shares[0], s0 * shares[1]])


def population():
    from .asym_pilot import population as pop
    x, y = pop(DELTA)
    return np.asarray(x, float), np.asarray(y, float)


def own_sample(seed):
    from .asym_register import training_set
    return training_set(int(seed))


def hidden_draw(seed):
    """Coordinates 0, 1, 3, 4, 6 of U(−1, 1)⁷ (float64) from a local torch Generator seeded with the seed: the same
    numbers as width2_train.init_params(seed), with no global RNG state touched."""
    import torch
    gen = torch.Generator().manual_seed(int(seed))
    q = torch.empty(7, dtype=torch.float64).uniform_(-1.0, 1.0, generator=gen).numpy()
    return q[ZI].copy()


def _jsonable(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, np.bool_):
            v = bool(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        out[k] = v
    return out


# ------------------------------------------------------------------------------------------ continuation (freeze)
DIAG = np.array([0.5, 0.5])


def continue_diagonal(z0, s0, x, y, s_max, h=RAY_H, s_stop=None):
    """Natural-parameter continuation in log s along v = (s/2, s/2) from (z0, s0): tangent predictor, damped Newton;
    the step is halved on a Newton failure or a correction > 0.2 (sup), down to 1e−6.  Checks at every accepted point:
    reduced λ_min, the split block's λ_min, still duplicate.  Stops at the first branch-gap sign change (bisected to
    1e−10 in log s) -> 'switch', or, with s_stop, at the point at exactly s_stop -> 'stop'."""
    z, p, hh = np.asarray(z0, float), math.log(s0), h
    g0 = gap_mid(z, s0 * DIAG)
    min_lam, min_split, dup_ok, n = float("inf"), float("inf"), bool(is_duplicate(z)), 0
    p_end = math.log(s_stop) if s_stop is not None else math.log(s_max)

    def tan_log(zz, pp):
        return (dz_dv(zz, math.exp(pp) * DIAG, x, y) @ DIAG) * math.exp(pp)
    while True:
        if p >= p_end - 1e-15:
            if s_stop is not None:
                return {"status": "stop", "z": z, "v": math.exp(p) * DIAG, "min_lam_reduced": min_lam,
                        "min_lam_split": min_split, "duplicate_throughout": dup_ok, "n_steps": n}
            return {"status": "none", "s_last": math.exp(p)}
        step = min(hh, p_end - p)
        pn = p + step
        pred = z + step * tan_log(z, p)
        zn, _, _, ok = newton(pred, math.exp(pn) * DIAG, x, y)
        if not ok or np.abs(zn - pred).max() > BRANCH_JUMP:
            hh /= 2
            if hh < 1e-6:
                return {"status": "failed", "s_last": math.exp(p)}
            continue
        vn = math.exp(pn) * DIAG
        Hn, _ = hz_blocks(zn, vn, x, y)
        lr, ls = reduced_lam(Hn, vn, True)
        min_lam, min_split, dup_ok, n = min(min_lam, lr), min(min_split, ls), dup_ok and is_duplicate(zn), n + 1
        gn = gap_mid(zn, vn)
        if (gn > 0) != (g0 > 0):
            lo, hi, zl = p, pn, z
            while hi - lo > 1e-10:
                mid = 0.5 * (lo + hi)
                zm, _, _, okm = newton(zl + (mid - lo) * tan_log(zl, lo), math.exp(mid) * DIAG, x, y)
                if okm and (gap_mid(zm, math.exp(mid) * DIAG) > 0) == (g0 > 0):
                    lo, zl = mid, zm
                else:
                    hi = mid
            ssw = math.exp(0.5 * (lo + hi))
            zsw, _, _, oksw = newton(zl, ssw * DIAG, x, y)
            return {"status": "switch", "s_switch": ssw, "z_switch": zsw, "v_switch": ssw * DIAG, "vprime": DIAG.copy(),
                    "newton_ok_switch": oksw, "min_lam_reduced": min_lam, "min_lam_split": min_split,
                    "duplicate_throughout": dup_ok and is_duplicate(zsw), "n_steps": n, "to_placed": bool(g0 <= 0),
                    "G_start": g0}
        z, p = zn, pn
        hh = min(h, 1.5 * hh)


def flow_rhs(z, v, x, y):
    """dv/ds on the adiabatic reference: g/(sign(v)·g), g = −∇_vL(z*(v), v) (the reduced gradient flow of v, by the
    envelope theorem, parametrised by s).  None where s does not increase along the flow (a turning point)."""
    g = -grad(qof(z, v), x, y)[VI]
    ds = float(np.sign(v) @ g)
    return (g / ds) if ds > 0 else None


def rk4_step(z, v, hs, x, y):
    """One RK4 step of length hs in s along the adiabatic reference, z* by damped Newton (tangent-predicted) at every
    stage.  Returns (z_new, v_new, ok)."""
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
    """The adiabatic reference (T, T′): RK4 in s with steps hs_rel·s, z* by damped Newton at every stage.  Checks at every
    accepted point: Newton accepted, λ_min > 0, s increasing along the flow (no turning point).  Stops at the first
    branch-gap sign change (the last step bisected in its length to 1e−12 relative) -> 'switch', or, with s_stop, at the
    point at exactly s_stop -> 'stop'."""
    z, v = np.asarray(z0, float), np.asarray(v0, float)
    g0 = gap_mid(z, v)
    min_lam, n = float("inf"), 0
    while True:
        s = float(np.abs(v).sum())
        if s_stop is not None and s >= s_stop * (1 - 1e-14):
            return {"status": "stop", "z": z, "v": v, "min_lam_reduced": min_lam, "n_steps": n}
        if s >= s_max:
            return {"status": "none", "s_last": s}
        hs = hs_rel * s
        if s_stop is not None and s + hs >= s_stop:
            hs = s_stop - s
        zn, vn, ok = rk4_step(z, v, hs, x, y)
        if not ok:
            return {"status": "failed", "s_last": s}
        lam = float(np.linalg.eigvalsh(hz_blocks(zn, vn, x, y)[0]).min())
        min_lam, n = min(min_lam, lam), n + 1
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
            return {"status": "switch", "s_switch": float(np.abs(vs).sum()), "z_switch": zs, "v_switch": vs,
                    "vprime": flow_rhs(zs, vs, x, y), "newton_ok_switch": oks, "min_lam_reduced": min_lam,
                    "n_steps": n, "to_placed": bool(g0 <= 0), "G_start": g0,
                    "share_switch": float(vs[0] / np.abs(vs).sum())}
        if (gn > 0) != (g0 > 0):
            return {"status": "failed", "s_last": s, "note": "sign change before s_stop"}
        z, v = zn, vn


def _decided_diag(zsw, ssw, x, y):
    res = {}
    for f in (1 - DECIDE_REL, 1 + DECIDE_REL):
        zf, _, _, okf = newton(zsw, f * ssw * DIAG, x, y)
        res[f] = (gap_enclosure(zf, f * ssw * DIAG), okf)
    below, above = res[1 - DECIDE_REL][0], res[1 + DECIDE_REL][0]
    return below, above, bool(below[1] < 0 < above[0] and res[1 - DECIDE_REL][1] and res[1 + DECIDE_REL][1])


def _decided_adiab(zsw, vsw, x, y):
    ssw = float(np.abs(vsw).sum())
    zb, vb, okb = rk4_step(zsw, vsw, -DECIDE_REL * ssw, x, y)
    za, va, oka = rk4_step(zsw, vsw, DECIDE_REL * ssw, x, y)
    below, above = gap_enclosure(zb, vb), gap_enclosure(za, va)
    return below, above, bool(below[1] < 0 < above[0] and okb and oka)


def switch_of_copy(copy, z_s0, v0, x, y, s_max):
    """Validated switch of one copy (D, D′: along the diagonal; T, T′: along the adiabatic reference), recomputed with the
    step halved (agreement 1e−6 relative); no turning point (a path that fails is invalid); reduced λ_min > 0 along it;
    D, D′: split block PD and duplicate throughout; G(s₀) < 0 and the switch goes to placed; the enclosure is decided at
    (1 ± 1e−3)·s_switch; ∇G forward/backward agreement ≤ 1e−5; Newton accepted at the switch; κ finite."""
    dup = copy in DUPLICATE_COPIES
    s0 = float(np.abs(v0).sum())
    if dup:
        a = continue_diagonal(z_s0, s0, x, y, s_max)
        b = continue_diagonal(z_s0, s0, x, y, s_max, h=RAY_H / 2)
    else:
        a = continue_adiabatic(z_s0, v0, x, y, s_max)
        b = continue_adiabatic(z_s0, v0, x, y, s_max, hs_rel=ADIAB_HS / 2)
    row = {"path": "diagonal" if dup else "adiabatic", "status": a["status"], "status_halved": b["status"]}
    if a["status"] != "switch" or b["status"] != "switch":
        return {**row, "valid": False, "note": f"no validated switch below {s_max:.4f}"}
    ssw = a["s_switch"]
    row.update(s_switch=ssw, s_switch_halved=b["s_switch"],
               halving_ok=bool(abs(ssw - b["s_switch"]) <= HALVING_REL * ssw),
               to_placed=a["to_placed"], G_start=a["G_start"], min_lam_reduced_path=a["min_lam_reduced"],
               v_switch=[float(t) for t in a["v_switch"]], z_switch=[float(t) for t in a["z_switch"]],
               newton_ok_switch=a["newton_ok_switch"], n_steps=a["n_steps"])
    if dup:
        row.update(min_lam_split_path=a["min_lam_split"], duplicate_throughout=a["duplicate_throughout"])
        below, above, dec = _decided_diag(a["z_switch"], ssw, x, y)
    else:
        row.update(share_switch=a["share_switch"])
        below, above, dec = _decided_adiab(a["z_switch"], a["v_switch"], x, y)
    row.update(G_enc_below=list(below), G_enc_above=list(above), decided_ok=dec)
    k = kappa_at(a["z_switch"], a["v_switch"], a["vprime"], x, y, dup)
    row.update(k)
    checks = [row["halving_ok"], row["to_placed"], row["min_lam_reduced_path"] > 0, row["decided_ok"],
              row["gradG_ok"], row["newton_ok_switch"], k["lam_min_switch"] > 0, bool(np.isfinite(k["kappa0"]))]
    if dup:
        checks += [row["min_lam_split_path"] > 0, row["duplicate_throughout"]]
    row["checks"] = [bool(c) for c in checks]
    row["valid"] = bool(all(checks))
    return row


def follow_point(copy, z_s0, v0, s_sw, x, y):
    """The copy's branch point at s_f = 0.8·s_switch on its reference path (D, D′: diagonal; T, T′: adiabatic),
    recomputed with the step halved (agreement 1e−6 sup, z and v), with ∂z*/∂v there."""
    s_f = FOLLOW_FRAC * s_sw
    s0 = float(np.abs(v0).sum())
    if copy in DUPLICATE_COPIES:
        r1 = continue_diagonal(z_s0, s0, x, y, s_f, s_stop=s_f)
        r2 = continue_diagonal(z_s0, s0, x, y, s_f, h=RAY_H / 2, s_stop=s_f)
    else:
        r1 = continue_adiabatic(z_s0, v0, x, y, 10 * s_sw, s_stop=s_f)
        r2 = continue_adiabatic(z_s0, v0, x, y, 10 * s_sw, hs_rel=ADIAB_HS / 2, s_stop=s_f)
    if r1["status"] != "stop" or r2["status"] != "stop":
        return {"s": s_f, "valid": False, "note": f"{r1['status']}/{r2['status']}"}
    zf, _, _, okn = newton(r1["z"], r1["v"], x, y)
    sup = float(max(np.abs(r1["z"] - r2["z"]).max(), np.abs(r1["v"] - r2["v"]).max()))
    return {"s": float(s_f), "v": [float(t) for t in r1["v"]], "z": [float(t) for t in zf],
            "dz_dv": dz_dv(zf, r1["v"], x, y).tolist(), "halving_sup": sup,
            "valid": bool(okn and sup <= HALVING_REL and (copy not in DUPLICATE_COPIES or is_duplicate(zf)))}


# ------------------------------------------------------------------------------------------ landscape (population)
def landscape():
    """Population copies at v₀ and their population switches (L4's comparators).  Each copy: damped Newton at v₀ on
    the population from the exploratory class point (POP_START); accepted, of its type, unplaced.  Switches validated
    as switch_of_copy (s_max = 30).  STOP if a copy is invalid or its switch is more than 0.1% from the page's value."""
    OUT.mkdir(parents=True, exist_ok=True)
    x, y = population()
    s0 = s0_value()
    out = {"s_pop2": s_pop2(), "s0": s0, "v_D": v_held(SHARES[D], s0).tolist(), "v_T": v_held(SHARES[T], s0).tolist(),
           "copies": {}}
    for c in ("D", "Dp", "T", "Tp"):
        v = v_held(COPY_SHARES[c], s0)
        zz, gm, lam, ok = newton(np.array(POP_START[c]), v, x, y)
        H, _ = hz_blocks(zz, v, x, y)
        lr, ls = reduced_lam(H, v, c in DUPLICATE_COPIES)
        row = {"z_s0": zz.tolist(), "grad_s0": gm, "lam_min_s0": lam, "lam_reduced_s0": lr, "lam_split_s0": ls,
               "newton_ok": ok, "G_s0": list(gap_enclosure(zz, v)), "type": copy_type(zz, v, placed_state(zz, v)[0]),
               "loss": loss(qof(zz, v), x, y), "start_distance": float(np.abs(zz - np.array(POP_START[c])).max())}
        sw = switch_of_copy(c, zz, v, x, y, s_max=S_MAX_POP)
        row["switch"] = sw
        row["s_pop_branch"] = sw.get("s_switch")
        row["matches_page"] = bool(sw.get("s_switch") is not None
                                   and abs(sw["s_switch"] / POP_SWITCH_PAGE[c] - 1) <= POP_SWITCH_REL)
        row["valid"] = bool(ok and row["type"] == c and sw["valid"] and row["matches_page"])
        out["copies"][c] = row
        print(c, row["type"], ok, sw.get("s_switch"), sw["valid"], sw.get("kappa0"), flush=True)
    out["validated"] = all(out["copies"][c]["valid"] for c in ("D", "Dp", "T", "Tp"))
    (OUT / "landscape.json").write_text(json.dumps(_jsonable_tree(out), indent=1))
    if not out["validated"]:
        raise SystemExit("STOP: population landscape not validated")
    return out


def _jsonable_tree(o):
    if isinstance(o, dict):
        return {str(k): _jsonable_tree(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable_tree(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable_tree(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def _land():
    return json.loads((OUT / "landscape.json").read_text())


# ------------------------------------------------------------------------------------------ per-seed frozen inputs
def seed_copies(seed):
    """(copies frozen in full, copies frozen as points only).  Registered seeds 884,000-884,119: T, D, D′ in full (arms
    T and D), T′ as a point (arm T's release classification); 884,200-884,319: T′ in full, T as a point; pilot seeds:
    all four in full (every arm is piloted on them)."""
    if seed in SEEDS_TP:
        return ("Tp",), ("T",)
    if seed in PILOT_SEEDS:
        return ("T", "Tp", "D", "Dp"), ()
    return ("T", "D", "Dp"), ("Tp",)


def freeze_copy(copy, land, x, y, full):
    v = np.array(land["v_D"] if copy in DUPLICATE_COPIES else land["v_T"])
    zp = np.array(land["copies"][copy]["z_s0"])
    z, gm, lam, ok = newton(zp, v, x, y)
    typ = copy_type(z, v, placed_state(z, v)[0]) if ok else None
    H, _ = hz_blocks(z, v, x, y)
    lr, ls = reduced_lam(H, v, copy in DUPLICATE_COPIES)
    row = {"copy": copy, "z_s0": z.tolist(), "grad_s0": gm, "lam_min_s0": lam, "lam_reduced_s0": lr,
           "lam_split_s0": ls, "newton_ok": ok, "type": typ, "point_ok": bool(ok and typ == copy),
           "W": w_hold(lam) if ok else None, "G_s0": list(gap_enclosure(z, v)), "full": bool(full)}
    if not row["point_ok"]:
        return {**row, "valid": False, "note": "no accepted own-sample point of this type at s0"}
    if not full:
        return {**row, "valid": False, "note": "point only (classification)"}
    sw = switch_of_copy(copy, z, v, x, y, S_HI_FRAC * land["copies"][copy]["s_pop_branch"])
    row.update(sw)
    if sw["valid"]:
        row["follow_point"] = follow_point(copy, z, v, sw["s_switch"], x, y)
        row["valid"] = bool(row["follow_point"]["valid"])
        if not row["valid"]:
            row["note"] = "follow point invalid"
    return row


def freeze_one(seed, land):
    x, y = own_sample(seed)
    full, point = seed_copies(seed)
    rows = {c: freeze_copy(c, land, x, y, True) for c in full}
    rows.update({c: freeze_copy(c, land, x, y, False) for c in point})
    W = {c: r["W"] for c, r in rows.items() if r.get("point_ok")}
    arms = [a for a in ARMS if all(c in full for c in SCORED_COPIES[a])]
    return {"seed": int(seed), "pilot": seed in PILOT_SEEDS, "copies": rows,
            "W_arm": {a: arm_hold_steps(a, W) for a in arms}}


def freeze():
    _setup_process()
    land = _land()
    f = OUT / "frozen_parts.jsonl"
    done = set() if not f.exists() else {json.loads(ln)["seed"] for ln in f.read_text().splitlines()}
    for seed in PILOT_SEEDS + SEEDS + SEEDS_TP:
        if seed in done:
            continue
        _memory_guard(f"freeze {seed}")
        t0 = time.time()
        r = freeze_one(seed, land)
        with open(f, "a") as fh:
            fh.write(json.dumps(_jsonable_tree(r)) + "\n")
        print(json.dumps({"seed": seed, "secs": round(time.time() - t0, 1),
                          **{c: [v.get("valid"), v.get("s_switch"), v.get("kappa0"), v.get("W")]
                             for c, v in r["copies"].items()}}, default=float), flush=True)
        _rss_guard()
    rows = sorted((json.loads(ln) for ln in f.read_text().splitlines()), key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == sorted(PILOT_SEEDS + SEEDS + SEEDS_TP)
    (OUT / "frozen_seeds.json").write_text(json.dumps(rows, indent=1))
    nv = {}
    for r in rows:
        for c, v in r["copies"].items():
            if v.get("full"):
                nv.setdefault(c, [0, 0])
                nv[c][0] += bool(v.get("valid"))
                nv[c][1] += 1
    print("frozen", len(rows), "valid full copies", nv, flush=True)


def _frozen():
    return {r["seed"]: r for r in json.loads((OUT / "frozen_seeds.json").read_text())}


# ------------------------------------------------------------------------------------------ runs
def train_path(q0, eta, rho, n_steps, x, y):
    """Full-batch SGD (no momentum) from q0: q ← q − (η on z, ρη on v)·∇L, n_steps steps.  Row 0 is q0."""
    lr = np.full(7, float(eta))
    lr[VI] = rho * eta
    P = np.empty((int(n_steps) + 1, 7))
    P[0] = q0
    q = np.asarray(q0, float).copy()
    for t in range(1, int(n_steps) + 1):
        q = q - lr * grad(q, x, y)
        P[t] = q
    return P


def hold(z_start, v0, W, x, y):
    """Hold: v fixed at v₀, z by full-batch GD at lr 1.0 for W steps; placed_state at the start state and after every
    step (the start state counts as a hold step).  Returns (z at release, record)."""
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


def run_start(arm, seed, fr, land):
    """Start, hold and release classification (information up to release only)."""
    x, y = own_sample(seed)
    v0 = v_held(SHARES[arm], land["s0"])
    z0 = np.array(land["copies"]["Tp"]["z_s0"]) if arm == TP else hidden_draw(seed)
    W = fr["W_arm"][arm]
    z_rel, rec = hold(z0, v0, W, x, y)
    zn, gn, lamn, okn = newton(z_rel, v0, x, y)
    copies = {c: (fr["copies"][c]["z_s0"] if fr["copies"].get(c, {}).get("point_ok") else None)
              for c in CLASSIFY_COPIES[arm]}
    dup_req = tuple(c for c in CLASSIFY_COPIES[arm] if c in DUPLICATE_COPIES)
    copy, k = classify(z_rel, zn, okn, copies, v0, duplicate_required=dup_req)
    typ = copy_type(zn, v0, placed_state(zn, v0)[0]) if okn else "not accepted"
    rec.update(arm=arm, seed=int(seed), z_init=z0.tolist(), z_release=z_rel.tolist(), v0=v0.tolist(),
               newton_grad=gn, newton_lam=lamn, newton_ok=okn, newton_type=typ, copy_at_release=copy,
               windings=list(k) if k else None, on_branch=bool(is_on_branch(arm, copy)),
               release_dist=float(np.abs(z_rel - zn).max()))
    return z_rel, v0, rec, (x, y)


def follow_classify(z_newton, newton_ok, z_copy, copy_ok, v, k, copy):
    """The follow check's decision: the release copy (at the release winding k) iff the run's Newton point is accepted,
    the copy's carried branch point is accepted, the two are within 1e−6 (sup; FOLLOW_STATE_TOL = None: no state
    distance) and, for D and D′, the run's Newton point is still a duplicate.  Otherwise None."""
    if not (newton_ok and copy_ok):
        return None
    zc = shift(np.asarray(z_copy, float), v, (-k[0], -k[1]))            # the copy's point at the canonical winding
    c, kk = classify(z_newton, z_newton, True, {copy: zc}, v, state_tol=FOLLOW_STATE_TOL,
                     duplicate_required=(copy,) if copy in DUPLICATE_COPIES else ())
    return c if (c is not None and tuple(kk) == tuple(k)) else None


def follow_check(P, s, cf, copy, k, x, y):
    """At t₀.₈ = the first step with s_t ≥ 0.8·s_switch (the copy's frozen switch, where its follow point is): damped
    Newton at v_{t₀.₈} from the run's z; the copy's branch point there = Newton at v_{t₀.₈} from the frozen follow point
    (winding-shifted) carried by one tangent predictor.  Decision: follow_classify.  The state at t₀.₈ only (no gap)."""
    fp = cf.get("follow_point") or {}
    t08 = first_ge(s, fp["s"]) if fp.get("s") is not None else None
    if t08 is None:
        return {"t_follow": None, "copy_at_follow": None, "follows_branch": False, "follow_note": "s never reaches s_f"}
    if not fp.get("valid"):
        return {"t_follow": int(t08), "copy_at_follow": None, "follows_branch": False,
                "follow_note": "no valid follow point"}
    zr, vr = P[t08][ZI], P[t08][VI]
    vf, zf = np.array(fp["v"]), np.array(fp["z"])
    guess = shift(zf + np.array(fp["dz_dv"]) @ (vr - vf), vr, k)        # tangent predictor, then the release winding
    zc, _, _, okc = newton(guess, vr, x, y)
    zn, _, _, okn = newton(zr, vr, x, y)
    c08 = follow_classify(zn, okn, zc, okc, vr, k, copy)
    return {"t_follow": int(t08), "s_follow_run": float(s[t08]), "copy_at_follow": c08, "follow_newton_ok": okn,
            "follow_copy_point_ok": okc, "follow_newton_dist": float(np.abs(zn - zc).max()) if okc and okn else None,
            "follow_state_dist": float(np.abs(zr - zc).max()) if okc else None,
            "follows_branch": follows_branch(copy, c08),
            "follow_duplicate": bool(is_duplicate(zn)) if copy in DUPLICATE_COPIES else None}


def _prediction_setup(arm, rec, fr):
    """The occupied copy's frozen row and the run's κ, or a reason for no prediction."""
    if not rec["on_branch"]:
        return None, "no prediction: not on a scored copy at release"
    copy, k = rec["copy_at_release"], tuple(rec["windings"])
    cf = fr["copies"][copy]
    if not cf.get("valid"):
        return None, "no prediction: the occupied copy has no validated frozen switch"
    if not in_winding_table(k):
        return None, "no prediction: winding outside the table (k1 + k2 not in -3..3)"
    return (copy, k, cf), None


def predict_one(arm, seed, P, rec, fr, land, x, y, with_traj=True):
    """Predictions for one run from its release classification, the frozen inputs and its v path (no gap of the training
    state after release is evaluated).  with_traj False (pilot): up to t_sw only, no recursion."""
    out = dict(rec)
    setup, why = _prediction_setup(arm, rec, fr)
    if setup is None:
        return {**out, "status": why}
    copy, k, cf = setup
    eta = ETA[arm]
    V = P[:, VI]
    s = np.abs(V).sum(axis=1)
    kap = kappa_winding(cf["kappa0"], cf["kappa_winding_coef"], k)
    lam_sw = cf["lam_min_switch"]
    out.update(kappa=kap, lam_min_switch=lam_sw, eta_lam=eta * lam_sw, lag_steps_pred=kap / (eta * lam_sw),
               s_switch_frozen=cf["s_switch"], s_pop_branch=land["copies"][copy]["s_pop_branch"],
               s_min_after_release=float(s.min()))
    dup = copy in DUPLICATE_COPIES
    br = OwnBranch(V, shift(np.array(cf["z_s0"]), V[0], k), x, y, dup)
    t_last = len(s) - 1
    if OWN_PATH_SWITCH[arm]:
        t_sw, s_sw = own_path_switch(br, t_last)
        out["s_switch_own_path"] = s_sw
    else:
        t_sw, s_sw = first_ge(s, cf["s_switch"]), float(cf["s_switch"])
    out.update(s_switch=s_sw, t_sw=t_sw)
    if t_sw is None or not np.isfinite(s_sw):
        return {**out, "branch_lost_at": br.lost_at,
                "status": "no prediction: no switch on the run's own branch within the budget"}
    br.ensure(t_sw)
    r_cf, chi, sdot = closed_form(s, t_sw, s_sw, kap, lam_sw, eta)
    out.update(sdot=sdot, chi_tsw=chi, r_cf=r_cf, kc_tsw=r_cf, pred_signed_s_cross_cf=s_sw * (1 + r_cf))
    cp = chi_path(s, br.lam, eta, t_sw)
    out["chi_path_max_all"] = float(cp.max()) if len(cp) and np.all(np.isfinite(cp)) else float("nan")
    out["chi_window_max"] = chi_window_max(s, t_sw, cp, s_sw)
    if dup:
        spl = [v for v in br.lam_split[:t_sw + 1] if v is not None]
        out["min_lam_split_to_tsw"] = float(min(spl)) if spl else float("nan")
    out.update(follow_check(P, s, cf, copy, k, x, y))
    if not with_traj:
        return {**out, "branch_lost_at": br.lost_at, "status": "ok (pilot: to t_sw, no recursion)"}
    t_hit, st, mx, rad = r4_recursion(br, P[0][ZI], eta, t_last)
    out.update(traj_radius_max=rad, traj_max_abs_delta=mx, traj_status=st, branch_lost_at=br.lost_at,
               branch_steps_computed=len(br.Z))
    if t_hit is None:
        out.update(r_traj=float("nan"), s_traj=float("nan"), t_traj=None)
    else:
        out.update(t_traj=int(t_hit), s_traj=float(s[t_hit]), r_traj=float(s[t_hit]) / s_sw - 1,
                   lag_steps_traj=int(t_hit) - int(t_sw), l5_eligible=bool(l5_eligible([t_hit], [t_sw])[0]))
    out["status"] = "ok"
    return out


# ------------------------------------------------------------------------------------------ pilot
def train_to_switch(arm, q0, rho, cf, k, x, y, chunk=2000):
    """Train in chunks until t_sw (D: s ≥ the frozen switch; T, T′: the own-path switch, the branch extended
    incrementally), the budget, or the branch is lost.  The v path only is read; no gap of the state."""
    n_max = budget(arm, rho)
    P = np.asarray(q0, float)[None, :]
    dup = cf["copy"] in DUPLICATE_COPIES
    br = OwnBranch(P[:, VI], shift(np.array(cf["z_s0"]), P[0][VI], k), x, y, dup)
    checked = 0
    while len(P) - 1 < n_max:
        n = min(chunk, n_max - (len(P) - 1))
        P = np.vstack([P, train_path(P[-1], ETA[arm], rho, n, x, y)[1:]])
        if arm == D:
            if first_ge(np.abs(P[:, VI]).sum(axis=1), cf["s_switch"]) is not None:
                break
            continue
        br.V = P[:, VI]
        t_sw, _ = own_path_switch(br, len(P) - 1, t_from=max(1, checked))
        checked = len(P)
        if t_sw is not None or br.lost_at is not None:
            break
    return P


def pilot_run(arm, seed, rho, fr, land):
    z_rel, v0, rec, (x, y) = run_start(arm, seed, fr, land)
    setup, why = _prediction_setup(arm, rec, fr)
    if setup is None:
        return {**rec, "rho": rho, "status": why}
    copy, k, cf = setup
    P = train_to_switch(arm, qof(z_rel, v0), rho, cf, k, x, y)
    r = predict_one(arm, seed, P, rec, fr, land, x, y, with_traj=False)
    r.update(rho=rho, n_steps_run=int(len(P) - 1))
    return r


PILOT_KEYS = ("arm", "seed", "rho", "copy_at_release", "windings", "on_branch", "hold_n_G_pos", "t_sw", "s_switch",
              "kappa", "chi_tsw", "kc_tsw", "chi_window_max", "chi_path_max_all", "follows_branch", "t_follow",
              "follow_state_dist", "n_steps_run", "branch_lost_at", "status", "W_hold", "newton_type",
              "release_dist", "s_switch_frozen")


def pilot():
    """The pilot rules on the pilot seeds, per arm, up to t_sw only (no gap of the state after release; no crossing).
    D: GELU-T's rule on the SIGNED κχ at t_sw; T, T′: from ρ = 2⁻¹⁰ on q90 of V7's window max χ_t.  Each arm's pilot
    median χ at t_sw (at the chosen ρ) is kept for V6.  Resumable (pilot_parts.jsonl)."""
    _setup_process()
    land, fz = _land(), _frozen()
    parts = OUT / "pilot_parts.jsonl"
    done = {} if not parts.exists() else {(r["arm"], r["seed"], r["rho"]): r
                                           for r in map(json.loads, parts.read_text().splitlines())}
    res = {}
    for arm in (D, T, TP):
        runs = {}

        def run_pilot(rho, arm=arm, runs=runs):
            vals, rows = [], []
            for seed in PILOT_SEEDS:
                key = (arm, seed, rho)
                if key in done:
                    r = done[key]
                else:
                    _memory_guard(f"pilot {arm} {seed} rho={rho}")
                    t0 = time.time()
                    r = _jsonable_tree({kk: vv for kk, vv in pilot_run(arm, seed, rho, fz[seed], land).items()
                                        if kk in PILOT_KEYS})
                    r["secs"] = round(time.time() - t0, 1)
                    with open(parts, "a") as fh:
                        fh.write(json.dumps(r) + "\n")
                    print(json.dumps(r), flush=True)
                    _rss_guard()
                rows.append(r)
                v = r.get("kc_tsw") if arm == D else r.get("chi_window_max")
                if r.get("on_branch") and v is not None and np.isfinite(v):
                    vals.append(v)
            runs[rho] = rows
            return vals
        rule = pilot_rule_D(run_pilot) if arm == D else pilot_rule_T(run_pilot)
        final = runs.get(rule["rho"], []) if rule["status"] == "ok" else []
        chis = [r["chi_tsw"] for r in final if r.get("on_branch") and r.get("chi_tsw") is not None]
        res[arm] = {"rule": rule, "rho": rule["rho"], "pilot_median_chi_tsw": float(np.median(chis)) if chis else None,
                    "n_on_branch_final": int(sum(bool(r.get("on_branch")) for r in final)),
                    "n_with_value_final": int(len(chis)), "runs": {str(kk): vv for kk, vv in runs.items()}}
    out = {"pilot_seeds": list(PILOT_SEEDS), "arms": res}
    (OUT / "pilot.json").write_text(json.dumps(_jsonable_tree(out), indent=1))
    print(json.dumps({a: {k: v for k, v in r.items() if k != "runs"} for a, r in res.items()}, indent=1, default=float))
    stops = [a for a in ARMS if res[a]["rule"]["status"] != "ok"]
    if stops:
        raise SystemExit("STOP: pilot rule for " + ", ".join(stops))
    return out


# ------------------------------------------------------------------------------------------ registration hashes
def hashes():
    lines = [f"{_sha(ROOT / p)}  {p}" for p in REGISTERED_FILES]
    (OUT / "registration.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _assert_registration():
    for line in (OUT / "registration.sha256").read_text().splitlines():
        h, p = line.split()
        assert _sha(ROOT / p) == h, f"{p} changed since registration"


# ------------------------------------------------------------------------------------------ registered training
def _path_file(arm, seed):
    return PATHS / f"{arm}_{seed}.npy"


def train():
    """All arms to the full budget; no gap or placement of the state after release is evaluated.  Each path is saved
    (untracked, results/width2_asym/paths/) and hashed; each row is written as the run completes (resumable)."""
    _setup_process()
    _assert_registration()
    land, fz = _land(), _frozen()
    pil = json.loads((OUT / "pilot.json").read_text())["arms"]
    PATHS.mkdir(parents=True, exist_ok=True)
    f = OUT / "predictions_parts.jsonl"
    done = set() if not f.exists() else {(r["arm"], r["seed"]) for r in map(json.loads, f.read_text().splitlines())}
    for arm in ARMS:
        rho = pil[arm]["rho"]
        for seed in ARM_SEEDS[arm]:
            if (arm, seed) in done:
                continue
            _memory_guard(f"train {arm} {seed}")
            t0 = time.time()
            fr = fz[seed]
            z_rel, v0, rec, (x, y) = run_start(arm, seed, fr, land)
            P = train_path(qof(z_rel, v0), ETA[arm], rho, budget(arm, rho), x, y)
            pf = _path_file(arm, seed)
            np.save(pf, P)
            h = _sha(pf)
            try:
                r = predict_one(arm, seed, P, rec, fr, land, x, y)
            except Exception as e:                                         # noqa: BLE001  counted, not replaced
                r = {**rec, "status": f"error: {type(e).__name__}: {e}"}
            r.update(rho=rho, budget=budget(arm, rho), path_sha256=h, secs=round(time.time() - t0, 1))
            row = {kk: (json.dumps(vv) if isinstance(vv, (list, tuple)) else vv) for kk, vv in r.items()}
            with open(f, "a") as fh:
                fh.write(json.dumps(_jsonable(row)) + "\n")
            print(json.dumps({kk: r.get(kk) for kk in ("arm", "seed", "copy_at_release", "on_branch", "hold_n_G_pos",
                                                       "status", "t_sw", "t_traj", "r_traj", "r_cf", "follows_branch",
                                                       "secs")}, default=float), flush=True)
            del P
            _rss_guard()


FORBIDDEN = {"cross_step", "s_obs", "r_obs", "step_obs", "placed", "crossed"}


def finalize():
    d = pd.DataFrame([json.loads(ln) for ln in (OUT / "predictions_parts.jsonl").read_text().splitlines()])
    d = d.sort_values(["arm", "seed"])
    for arm in ARMS:
        assert list(d[d.arm == arm].seed) == list(ARM_SEEDS[arm]), f"every registered seed exactly once ({arm})"
    assert not (FORBIDDEN & set(d.columns))
    p = OUT / "predictions.csv"
    d.to_csv(p, index=False)
    (OUT / "predictions.sha256").write_text(f"{_sha(p)}  predictions.csv\n")
    print(_sha(p), len(d), d.groupby("arm").status.value_counts().to_dict())


# ------------------------------------------------------------------------------------------ observation and scoring
def _assert_committed(p, sha):
    h = _sha(p)
    assert h == sha.read_text().split()[0], "hash mismatch"
    rel = str(p.relative_to(ROOT))
    assert subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip(), f"{rel} not committed"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0, f"{rel} modified"
    return h


def observe_path(P):
    """Every-step detection after release: the first t ≥ 1 with placed_state (enclosure lower end > 0)."""
    und = 0
    for t in range(1, len(P)):
        p, u = placed_state(P[t][ZI], P[t][VI])
        und += int(u)
        if p:
            return t, und
    return None, und


def observe():
    """AFTER the predictions commit: assert the registration and prediction hashes and each path hash, gates first,
    then every-step detection for the arms whose gate passed, then the registered scoring."""
    _setup_process()
    _assert_registration()
    _assert_committed(OUT / "predictions.csv", OUT / "predictions.sha256")
    pr = pd.read_csv(OUT / "predictions.csv")
    pil = json.loads((OUT / "pilot.json").read_text())["arms"]
    parts = OUT / "observed_parts.jsonl"
    done = set() if not parts.exists() else {(r["arm"], r["seed"]) for r in map(json.loads, parts.read_text().splitlines())}
    for r in pr.itertuples():                              # every path hash first
        assert _sha(_path_file(r.arm, r.seed)) == r.path_sha256, f"path hash mismatch {r.arm} {r.seed}"
    gates = {arm: gate(arm, pr[pr.arm == arm].on_branch.astype(bool), pr[pr.arm == arm].hold_G_positive.astype(bool))
             for arm in ARMS}
    for r in pr.itertuples():
        if not gates[r.arm]["pass"] or (r.arm, int(r.seed)) in done:
            continue
        _memory_guard(f"observe {r.arm} {r.seed}")
        P = np.load(_path_file(r.arm, r.seed))
        t_c, und = observe_path(P)
        row = {"arm": r.arm, "seed": int(r.seed), "crossed": t_c is not None, "step_obs": t_c, "n_undecided": und,
               "s_obs": float(np.abs(P[t_c][VI]).sum()) if t_c is not None else float("nan")}
        with open(parts, "a") as fh:
            fh.write(json.dumps(_jsonable(row)) + "\n")
        del P
        _rss_guard()
    res = {"predictions_sha256": _sha(OUT / "predictions.csv"),
           "registration_sha256_file_sha256": _sha(OUT / "registration.sha256"),
           "rho": {a: pil[a]["rho"] for a in ARMS}, "gates": gates, "arms": {}}
    obs = pd.DataFrame([json.loads(ln) for ln in parts.read_text().splitlines()]) if parts.exists() else \
        pd.DataFrame(columns=["arm", "seed", "crossed", "step_obs", "s_obs", "n_undecided"])
    for arm in ARMS:
        g = pr[pr.arm == arm].copy()
        med_chi = pil[arm]["pilot_median_chi_tsw"]
        med_chi = float("nan") if med_chi is None else med_chi
        if not gates[arm]["pass"]:
            nan = np.full(len(g), np.nan)
            res["arms"][arm] = score_arm(arm, g.on_branch.astype(bool), g.hold_G_positive.astype(bool),
                                         np.zeros(len(g), bool), *([nan] * 14), med_chi, nan)
            continue
        d = g.merge(obs[obs.arm == arm], on=["arm", "seed"], how="left")
        assert len(d) == len(ARM_SEEDS[arm])

        def col(c, d=d):
            return pd.to_numeric(d[c], errors="coerce").to_numpy(float) if c in d else np.full(len(d), np.nan)
        fol = d["follows_branch"].fillna(False).astype(bool) if "follows_branch" in d else np.zeros(len(d), bool)
        args = (arm, d.on_branch.astype(bool), d.hold_G_positive.astype(bool), d.crossed.fillna(False).astype(bool),
                col("step_obs"), col("s_obs"), col("s_switch"), col("s_traj"), col("t_traj"), col("r_traj"),
                col("r_cf"), col("kappa"), col("t_sw"), col("eta_lam"), col("lag_steps_pred"), col("chi_tsw"),
                col("chi_window_max"), med_chi, col("s_pop_branch"))
        sc = score_arm(*args, follows=fol)
        if arm in (T, D):          # GELU-T's registered DESCRIPTIVE sensitivity analysis: runs with G > 0 in the hold out
            sens = score_arm(*args, follows=fol, exclude=d.hold_G_positive.astype(bool))
            sens.pop("scored_index")
            sc["sensitivity_excluding_hold_G_positive_DESCRIPTIVE"] = sens
        d["scored"] = False
        d.loc[d.index[sc["scored_index"]], "scored"] = True
        d["r_obs"] = col("s_obs") / col("s_switch") - 1
        d.to_csv(OUT / f"observed_runs_{arm}.csv", index=False)
        S = d[d.scored]
        sc["descriptive"] = {
            "copy_counts": d["copy_at_release"].fillna("none").value_counts().to_dict(),
            "newton_type_counts": d["newton_type"].fillna("none").value_counts().to_dict(),
            "n_init_placed": int(d.init_placed.astype(bool).sum()),
            "median_r_obs_scored": float(S.r_obs.median()), "median_r_traj_scored": float(S.r_traj.median()),
            "median_r_cf_scored": float(S.r_cf.median()),
            "median_steps_tsw_to_crossing_scored": float((S.step_obs - S.t_sw).median()),
            "n_kappa_negative_scored": int((S.kappa < 0).sum()),
            "n_hold_G_positive": int(d.hold_G_positive.astype(bool).sum()),
            "n_undecided_steps_total": int(pd.to_numeric(d.n_undecided, errors="coerce").fillna(0).sum()),
            "n_follow_check_at_or_after_crossing": int((d.crossed.fillna(False).astype(bool)
                                                        & (col("t_follow") >= col("step_obs"))).sum()),
            "status_counts": d["status"].value_counts().to_dict()}
        res["arms"][arm] = sc
    res["headline"] = {"arm": HEADLINE, "outcome": res["arms"][HEADLINE]["outcome"]}
    (OUT / "scores.json").write_text(json.dumps(_jsonable_tree(res), indent=1))
    print(json.dumps(_jsonable_tree(res), indent=1))
    return res


if __name__ == "__main__":
    cmds = {"landscape": lambda: (_setup_process(), landscape()), "freeze": freeze, "pilot": pilot, "hashes": hashes,
            "train": train, "finalize": finalize, "observe": observe}
    cmds[sys.argv[1]]()
