"""Registered test 2B: the validity boundary of the lag law in κχ, redesigned (width-1 SGD forced ramps with the
learning rate set a priori for linear stability along the whole ramp).

Design (author-approved as designed, 2026-09-29): results/designs/2B_boundary_design.md.
Registration: results/track2b_registration.md.  Tests: tests/test_track2b.py.

Reuses the first boundary test's machinery (src/ramp_boundary.py, src/ramp.py): the population branch point θ*(s₀) on
the natural winding (ramp.branch_init), each seed's 400-point sample (ramp._data), the forced branch's own-sample
switch and λ_min there (ramp_boundary._switch_job), the dense crossing windows (ramp._windows) and the committed SGD κ
(lag_law/kappa_by_winding.csv, lag_law/kappa.csv).  What is new:

  η_cell   = min(0.3, 1/max λ_max(H_pop(s))) over s ∈ [s₀, s_end] on the population branch (validated continuation)
  γ        = (κχ_target/κ)·η_cell·λ_min(H_pop at the switch)        so χ = γ/(η_cell λ_min) = κχ_target/κ exactly
  s_end    = s*·(1 + 6κχ_target), s* the population switch; s₀ = 0.5·s*; 4,000 held steps at s₀, then s = s₀e^{γ(t−4000)}
  realised stability: η_cell·λ_max(H_own(θ_{t−1}, s_t)) at every step of every run, from the first held step to the
             run's crossing step (crossing runs) or to the last ramp step (other runs); H_own is the Hessian of the
             run's own training loss at the point where that step's gradient is evaluated.
  pred r   = κ_k·χ_target (κ_k the committed SGD κ for the run's winding at its crossing)
  obs r    = s_cross/s_branch − 1 (every-step crossing detection on the saved path)
Rules (score_cells):
  C1  median obs/pred in [0.75, 1.25] in every cell with κχ ≤ 0.1 (6 cells)
  C2  median obs/pred outside [0.75, 1.25] in every cell with κχ ≥ 0.3 (4 cells)
  κχ* (descriptive, per a): the smallest target whose median ratio leaves the band
  validity per cell: ≥ 30 of 40 runs cross; median χ_own within 30% of χ_target; realised stability holds.
  A criterion with any invalid cell is UNRESOLVED.

    python -m src.track2b branch     # population branch, λ along [s₀, s_end], design     -> results/track2b/
    python -m src.track2b freeze     # per-seed forced-branch switch and λ_min_own         -> frozen_switches.csv
    python -m src.track2b feasibility
    python -m src.track2b hashes     # registration manifest                               -> registration.sha256
    python -m src.track2b train      # registered runs, no gap evaluated                  -> predictions_parts.jsonl
    python -m src.track2b finalize   # predictions.csv + predictions.sha256
    python -m src.track2b observe    # AFTER the predictions commit: every-step detection, validity, scores
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import ramp as R

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "track2b"
PATHS = OUT / "paths"

SETTINGS = ((1.30, -1), (1.50, 0))                      # (a, natural winding)
KX_TARGETS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4)
SEEDS = tuple(range(2_020_000, 2_020_040))             # fresh (see the registration §3)
ETA_CAP = 0.3
WARMUP = 4000
S0_FRAC = 0.5
END_MULT = 6.0                                         # s_end = s*·(1 + 6κχ)
BAND = (0.75, 1.25)
CHI_TOL = 0.30
MIN_CROSS = 30
N_SEEDS = 40
C1_MAX_KX = 0.1
C2_MIN_KX = 0.3
STAB_MAX = 1.0
KAPPA_K_RANGE = tuple(range(-5, 6))                    # windings tabulated in the frozen prediction table
RSS_LIMIT = 3 * 1024 ** 3

# continuation grid and its validation
GRID_RATIO = 1.0025
RESID_MAX = 1e-10
STEP_MAX = 0.01                                        # continuation step in (w₁, b₁, b₂/s)

REGISTERED_FILES = ("src/track2b.py", "src/ramp.py", "src/ramp_boundary.py", "src/lag_law.py", "src/fold1d.py",
                    "src/width2_conditional.py", "tests/test_track2b.py", "results/track2b_registration.md",
                    "results/designs/2B_boundary_design.md", "results/lag_law/kappa.csv",
                    "results/lag_law/kappa_by_winding.csv", "results/cond_certified_brackets.csv",
                    "results/track2b/population_branch.csv", "results/track2b/branch_validation.json",
                    "results/track2b/design.csv",
                    "results/track2b/kappa_k.csv", "results/track2b/frozen_switches.csv",
                    "results/track2b/feasibility.json")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _read(p):
    """Frozen CSVs are read back exactly (the default parser can differ from the written value by one ulp)."""
    return pd.read_csv(p, float_precision="round_trip")


# ------------------------------------------------------------------------------------------ a-priori rules (pure)
def eta_cell(lam_max_max):
    """The stability rule: η = min(0.3, 1/λ_max), λ_max the largest population-Hessian eigenvalue over [s₀, s_end]."""
    lam = float(lam_max_max)
    assert np.isfinite(lam) and lam > 0, lam
    return min(ETA_CAP, 1.0 / lam)


def chi_target(kx, kappa):
    return float(kx) / float(kappa)


def gamma_for(kx, kappa, eta, lam_min):
    """γ with χ = γ/(η λ_min) equal to κχ_target/κ."""
    return chi_target(kx, kappa) * float(eta) * float(lam_min)


def end_factor(kx):
    """s_end / s*."""
    return 1.0 + END_MULT * float(kx)


def ramp_steps(gamma, kx):
    """Steps after the held phase until s = s₀e^{γt} reaches s_end = s*(1 + 6κχ)."""
    return math.ceil(math.log(end_factor(kx) / S0_FRAC) / gamma)


def scale_at(t, s0, gamma, warmup=WARMUP):
    """The output scale used by step t (t = 1, 2, ...): s₀ in the held phase, then s₀e^{γ(t − warmup)}."""
    return s0 if t <= warmup else s0 * math.exp(gamma * (t - warmup))


def realised_stability(eta, lam_max_steps, cross_step, n_steps):
    """Max over the run's realised trajectory of η·λ_max: steps 1..cross_step for a crossing run, 1..n_steps
    otherwise.  lam_max_steps[t−1] is λ_max(H_own(θ_{t−1}, s_t)) for step t.  Returns (max η·λ, holds?)."""
    lam = np.asarray(lam_max_steps, float)
    end = int(cross_step) if cross_step is not None and np.isfinite(cross_step) else int(n_steps)
    seg = lam[:end]
    assert len(seg) == end and end >= 1, (len(lam), end)
    m = float(eta * seg.max()) if np.all(np.isfinite(seg)) else float("inf")
    return m, bool(m <= STAB_MAX)


def cell_valid(n_cross, chi_own_median, chi_tgt, stab_ok):
    """Validity per cell: ≥ 30 crossings (of 40), median χ_own within 30% of χ_target, realised stability in every run."""
    chi_ok = bool(np.isfinite(chi_own_median) and abs(chi_own_median / chi_tgt - 1) <= CHI_TOL)
    return {"crossings_ok": bool(n_cross >= MIN_CROSS), "chi_ok": chi_ok, "stability_ok": bool(stab_ok),
            "valid": bool(n_cross >= MIN_CROSS and chi_ok and stab_ok)}


def score_cells(cells):
    """cells: dicts {a, kx_target, chi_target, ratios (per scored crossing run), n_cross, chi_own_median, stab_ok}.
    Returns the per-cell status, C1, C2 and κχ* per a (the first boundary test's conventions)."""
    lo, hi = BAND
    for c in cells:
        c.update(cell_valid(c["n_cross"], c["chi_own_median"], c["chi_target"], c["stab_ok"]))
        c["median_ratio"] = float(np.median(c["ratios"])) if len(c["ratios"]) else float("nan")
        c["in_band"] = bool(lo <= c["median_ratio"] <= hi) if np.isfinite(c["median_ratio"]) else None
    c1 = [c for c in cells if c["kx_target"] <= C1_MAX_KX + 1e-12]
    c2 = [c for c in cells if c["kx_target"] >= C2_MIN_KX - 1e-12]
    avals = sorted({round(c["a"], 2) for c in cells})
    assert len(c1) == 3 * len(avals) and len(c2) == 2 * len(avals), (len(c1), len(c2))

    def verdict(cs, ok):
        if not all(c["valid"] for c in cs):
            return "UNRESOLVED"
        return "PASS" if all(ok(c) for c in cs) else "FAIL"
    C1 = verdict(c1, lambda c: c["in_band"] is True)
    C2 = verdict(c2, lambda c: c["in_band"] is False)
    kx_star = {}
    for a in avals:
        cs = sorted([c for c in cells if round(c["a"], 2) == a], key=lambda c: c["kx_target"])
        out = [c["kx_target"] for c in cs if c["in_band"] is False]
        kx_star[a] = out[0] if out else None
    return {"C1": C1, "C2": C2, "kx_star": kx_star, "C1_cells_invalid": [(round(c["a"], 2), c["kx_target"]) for c in c1
                                                                       if not c["valid"]],
            "C2_cells_invalid": [(round(c["a"], 2), c["kx_target"]) for c in c2 if not c["valid"]], "cells": cells}


def feasibility_counts(s_branch, s_star, kx, ratio=1.0):
    """Seeds whose crossing at the lag ratio·κχ would lie at or beyond the ramp's end s*(1 + 6κχ), or which have no
    frozen switch (these cannot count as crossings against s_branch)."""
    sb = np.asarray(s_branch, float)
    beyond = ~np.isfinite(sb) | (sb * (1 + ratio * kx) >= s_star * end_factor(kx))
    return int(beyond.sum())


def feasible(n_beyond, n=N_SEEDS):
    """A cell can meet the crossing rule only if at most n − 30 seeds are ruled out a priori."""
    return bool(n - n_beyond >= MIN_CROSS)


# ------------------------------------------------------------------------------------------ population branch (a priori)
def _polish(z, s, a, x, y, iters=8):
    """Plain Newton steps after lag_law.branch_point (whose damped search can stop at a residual of ~1e-9 when the loss
    decrease is below round-off), kept while the gradient residual decreases."""
    import torch
    from .lag_law import _loss_t
    Xt, Yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    f = lambda q: _loss_t(q, s, a, Xt, Yt)

    def grad(zz):
        q = torch.tensor(zz, dtype=torch.float64, requires_grad=True)
        return torch.autograd.grad(f(q), q)[0].numpy()
    g = grad(z); r = float(np.abs(g).max())
    for _ in range(iters):
        H = torch.autograd.functional.hessian(f, torch.tensor(z, dtype=torch.float64)).numpy()
        zn = z - np.linalg.solve(H, g)
        gn = grad(zn); rn = float(np.abs(gn).max())
        if not rn < r:
            break
        z, g, r = zn, gn, rn
    return z, r


def population_branch(a, k, s_hi, extra=()):
    """Newton continuation (lag_law.branch_point, then plain-Newton polishing) of the population branch from θ*(s₀) on
    winding k up to s_hi, on a geometric grid (ratio GRID_RATIO) plus the points `extra`.  Returns a frame with s, z,
    gradient residual, the continuation step in scale-free coordinates (w₁, b₁, b₂/s), λ_min and λ_max of H_pop."""
    from .lag_law import _pop, branch_point, hessian_and_tangent
    x, y = _pop()
    sp = R._s_star_pop(a); s0 = S0_FRAC * sp
    grid = [s0]
    while grid[-1] * GRID_RATIO < s_hi:
        grid.append(grid[-1] * GRID_RATIO)
    grid = sorted(set(grid) | {float(s_hi)} | {float(e) for e in extra if s0 <= e <= s_hi})
    z = R.branch_init(a, k); s_prev = s0
    rows = []
    for s in grid:
        zn, _ = branch_point(z, s, a, x, y)
        zn, gr = _polish(zn, s, a, x, y)
        H, _ = hessian_and_tangent(zn, s, a, x, y)
        ev = np.linalg.eigvalsh(H)
        step = float(np.linalg.norm([zn[0] - z[0], zn[1] - z[1], zn[2] / s - z[2] / s_prev]))
        rows.append({"a": a, "winding": k, "s": s, "s_over_sstar": s / sp, "w1": zn[0], "b1": zn[1], "b2": zn[2],
                     "grad_residual": gr, "step_norm": step, "lambda_min": float(ev[0]), "lambda_max": float(ev[-1])})
        z, s_prev = zn, s
    return pd.DataFrame(rows)


def validate_branch(br, a):
    """Validation of the continuation: every point a stationary point (residual < 1e-10) with H positive definite, no
    continuation step larger than 0.01 in (w₁, b₁, b₂/s), λ_max nondecreasing in s (so the maximum over the continuous
    interval [s₀, s_end] is attained at the grid point s_end), and the branch passes through the committed switch point
    (lag_law/kappa.csv, up to the winding copy)."""
    from .lag_law import TWO_PI, _pop, branch_point
    L = R._landscape()[round(a, 2)]
    kt = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    s_sw = float(kt[kt.a.round(2) == round(a, 2)].s_star.iloc[0])
    k = int(br.winding.iloc[0])
    x, y = _pop()
    # the continued branch evaluated at the committed switch scale
    near = br.iloc[(br.s - s_sw).abs().argmin()]
    zc, _ = branch_point(np.array([near.w1, near.b1, near.b2]), s_sw, a, x, y)
    shift = np.array([0.0, TWO_PI * k, -TWO_PI * k * s_sw])
    dz = float(np.abs(zc - (L["z"] + shift)).max())
    out = {"max_residual": float(br.grad_residual.max()), "min_lambda_min": float(br.lambda_min.min()),
           "max_step_norm": float(br.step_norm.iloc[1:].max()), "switch_point_max_abs_diff": dz,
           "lambda_max_nondecreasing": bool((np.diff(br.lambda_max.to_numpy()) >= 0).all()), "n_points": int(len(br))}
    out["validated"] = bool(out["max_residual"] < RESID_MAX and out["min_lambda_min"] > 0
                            and out["max_step_norm"] < STEP_MAX and dz < 1e-6 and out["lambda_max_nondecreasing"])
    return out


def lam_on(br, s_lo, s_hi):
    """λ_max max and λ_min min over the grid points of the branch in [s_lo, s_hi] (both ends are grid points)."""
    seg = br[(br.s >= s_lo * (1 - 1e-12)) & (br.s <= s_hi * (1 + 1e-12))]
    assert np.isclose(seg.s.min(), s_lo, rtol=1e-12) and np.isclose(seg.s.max(), s_hi, rtol=1e-12)
    return float(seg.lambda_max.max()), float(seg.lambda_min.min())


def design_rows(branches):
    """The 12 cells from the validated branches (dict a -> frame) and the committed κ and switch Hessian."""
    kb = pd.read_csv(RESULTS / "lag_law" / "kappa_by_winding.csv"); L = R._landscape()
    rows = []
    for a, k in SETTINGS:
        br = branches[round(a, 2)]
        sp = R._s_star_pop(a); s0 = S0_FRAC * sp
        lam_sw = float(np.linalg.eigvalsh(L[round(a, 2)]["H"]).min())
        kap = float(kb[(kb.a.round(2) == round(a, 2)) & (kb.k == k)].kappa_sgd.iloc[0])
        for kx in KX_TARGETS:
            s_end = sp * end_factor(kx)
            lmax, lmin_min = lam_on(br, s0, s_end)
            eta = eta_cell(lmax)
            g = gamma_for(kx, kap, eta, lam_sw)
            n = ramp_steps(g, kx)
            rows.append({"a": a, "winding": k, "kx_target": kx, "kappa": kap, "chi_target": chi_target(kx, kap),
                         "s_star_pop": sp, "s0": s0, "s_end": s_end, "end_factor": end_factor(kx),
                         "lambda_max_max": lmax, "lambda_min_min_branch": lmin_min, "lambda_min_switch": lam_sw,
                         "eta_cell": eta, "eta_cap_binds": bool(eta == ETA_CAP), "gamma": g,
                         "eta_lambda_max_max": eta * lmax, "warmup": WARMUP, "ramp_steps": n, "total_steps": WARMUP + n,
                         "s_last": scale_at(WARMUP + n, s0, g)})
    return pd.DataFrame(rows)


def kappa_k_table():
    """κ_k (committed SGD landscape at the population switch, P = I) for k in KAPPA_K_RANGE at both a."""
    from .lag_law import kappa_k
    L = R._landscape()
    rows = []
    for a, _ in SETTINGS:
        l = L[round(a, 2)]
        for k in KAPPA_K_RANGE:
            rows.append({"a": a, "k": k, "kappa_k": float(kappa_k(l["H"], l["tan"], l["dG"], np.ones(3), k))})
    return pd.DataFrame(rows)


def branch():
    """Population branch and λ along [s₀, s_end] for every cell, validated; the design (η_cell, γ per cell)."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    frames, val, brs = [], {}, {}
    for a, k in SETTINGS:
        sp = R._s_star_pop(a)
        ends = [sp * end_factor(kx) for kx in KX_TARGETS]
        br = population_branch(a, k, max(ends), extra=ends)
        v = validate_branch(br, a)
        assert v["validated"], (a, v)
        val[str(round(a, 2))] = v
        frames.append(br); brs[round(a, 2)] = br
        print(a, k, v, flush=True)
    pd.concat(frames).to_csv(OUT / "population_branch.csv", index=False)
    d = design_rows(brs)
    d.to_csv(OUT / "design.csv", index=False)
    kappa_k_table().to_csv(OUT / "kappa_k.csv", index=False)
    (OUT / "branch_validation.json").write_text(json.dumps(val, indent=1))
    pd.set_option("display.width", 250)
    print(d.to_string(index=False))
    return d


# ------------------------------------------------------------------------------------------ per-seed frozen inputs
def freeze():
    """Each seed's forced-branch own-sample switch and λ_min(H_own) just below it (ramp_boundary._switch_job, the first
    boundary test's code), one process, written as each seed completes."""
    from . import ramp_boundary as B
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "frozen_parts.jsonl"
    done = set() if not f.exists() else {(round(r["a"], 2), r["seed"]) for r in map(json.loads, f.read_text().splitlines())}
    for a, k in SETTINGS:
        for sd in SEEDS:
            if (round(a, 2), sd) in done:
                continue
            r = B._switch_job((a, k, sd))
            with open(f, "a") as fh:
                fh.write(json.dumps({kk: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                     for kk, v in r.items()}) + "\n")
            print(json.dumps(r), flush=True)
            _rss_guard()
    d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()])
    d = d.sort_values(["a", "seed"]).reset_index(drop=True)
    assert len(d) == len(SETTINGS) * len(SEEDS) and not d.duplicated(["a", "seed"]).any()
    d.to_csv(OUT / "frozen_switches.csv", index=False)
    print(d.groupby("a").agg(n=("seed", "size"), ok=("s_branch", lambda v: int(np.isfinite(v).sum()))))


def feasibility():
    """A-priori check (before registration): per cell, the seeds whose predicted crossing (lag = κχ, and at the band's
    edge 1.25κχ) lies at or beyond the ramp's end.  If more than 10 are ruled out at lag κχ in any cell, the crossing
    rule cannot be met even if the law holds exactly, and the test STOPS before registration."""
    d = _read(OUT / "design.csv"); f = _read(OUT / "frozen_switches.csv")
    rows = []
    for r in d.itertuples():
        sb = f[f.a.round(2) == round(r.a, 2)].s_branch.to_numpy(float)
        n1 = feasibility_counts(sb, r.s_star_pop, r.kx_target, 1.0)
        n125 = feasibility_counts(sb, r.s_star_pop, r.kx_target, BAND[1])
        n2 = feasibility_counts(sb, r.s_star_pop, r.kx_target, 2.0)
        lmo = f[f.a.round(2) == round(r.a, 2)].lambda_min_own.to_numpy(float)
        chi_rel = (r.gamma / (r.eta_cell * lmo)) / r.chi_target
        rows.append({"a": r.a, "kx_target": r.kx_target, "n_beyond_at_pred": n1, "n_beyond_at_1.25pred": n125,
                     "n_beyond_at_2pred": n2, "feasible_at_pred": feasible(n1), "feasible_at_1.25pred": feasible(n125),
                     "feasible_at_2pred": feasible(n2), "chi_own_over_target_median_all40": float(np.median(chi_rel)),
                     "chi_own_over_target_min": float(chi_rel.min()), "chi_own_over_target_max": float(chi_rel.max())})
    out = {"cells": rows, "all_feasible_at_pred": all(x["feasible_at_pred"] for x in rows),
           "s_branch_over_s_star": {str(round(a, 2)): {"min": float((g.s_branch / R._s_star_pop(a)).min()),
                                                       "median": float((g.s_branch / R._s_star_pop(a)).median()),
                                                       "max": float((g.s_branch / R._s_star_pop(a)).max()),
                                                       "n_finite": int(np.isfinite(g.s_branch).sum())}
                                    for a, g in f.groupby(f.a.round(2))}}
    (OUT / "feasibility.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def hashes():
    lines = [f"{_sha(ROOT / p)}  {p}" for p in REGISTERED_FILES]
    (OUT / "registration.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _assert_registration():
    for line in (OUT / "registration.sha256").read_text().splitlines():
        h, p = line.split()
        assert _sha(ROOT / p) == h, f"{p} changed since registration"


def _rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")


# ------------------------------------------------------------------------------------------ dynamics
def hessian_batch(p, X, Y, s, a):
    """Closed-form Hessian of each seed's own mean BCE loss in (w₁, b₁, b₂) at w₂ = s.  p: (n, 3); X, Y: (n, m).
    Returns (n, 3, 3)."""
    import torch
    w1, b1, b2 = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    u = w1 * X + b1
    f1 = 1 + a * torch.cos(u)
    f2 = -a * torch.sin(u)
    z = s * (u + a * torch.sin(u)) + b2
    sg = torch.sigmoid(z)
    e = sg - Y
    q = sg * (1 - sg)
    A = q * (s * f1) ** 2 + e * s * f2                  # ∂²L/∂u² per point (before the chain rule in w₁, b₁)
    B = q * s * f1                                      # ∂²L/∂u∂b₂
    m = lambda v: v.mean(dim=1)
    H = torch.stack([torch.stack([m(A * X * X), m(A * X), m(B * X)], -1),
                     torch.stack([m(A * X), m(A), m(B)], -1),
                     torch.stack([m(B * X), m(B), m(q)], -1)], -2)
    return H


def train_cell(a, k, eta, gamma, s_star_pop, n_steps, seeds=SEEDS, warmup=WARMUP):
    """Vectorised full-batch SGD (lr η) over seeds, as ramp.batch_ramp but with the cell's η and without evaluating any
    gap: w₂ = +s_t is set every step, the hidden parameters start at the population branch point θ*(s₀) on winding k.
    Returns W (n_steps + 1, n, 3), the parameters after each step (row 0 = start), and lam (n_steps, n), λ_max of the
    own-sample Hessian at the point where each step's gradient is evaluated."""
    import torch
    torch.set_num_threads(1)
    X, Y = R._data(seeds)
    p = torch.tensor(np.tile(R.branch_init(a, k), (len(seeds), 1)), dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.SGD([p], lr=eta)
    s0 = S0_FRAC * s_star_pop
    W = np.empty((n_steps + 1, len(seeds), 3)); lam = np.empty((n_steps, len(seeds)))
    W[0] = p.detach().numpy()
    for t in range(1, n_steps + 1):
        s = scale_at(t, s0, gamma, warmup)
        with torch.no_grad():
            lam[t - 1] = torch.linalg.eigvalsh(hessian_batch(p.detach(), X, Y, s, a))[:, -1].numpy()
        opt.zero_grad(set_to_none=True)
        tt = p[:, 0:1] * X + p[:, 1:2]
        z = s * (tt + a * torch.sin(tt)) + p[:, 2:3]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none").mean(dim=1).sum()
        loss.backward()
        opt.step()
        W[t] = p.detach().numpy()
    return W, lam


def gap_path(Wr, a, chunk=500):
    """G(θ_t) for t = 0..T on the dense windows of ramp.batch_ramp (w₂ > 0 orientation), for one run's path (T+1, 3)."""
    import torch
    inner, outer = R._windows()
    out = []
    Wt = torch.tensor(np.asarray(Wr), dtype=torch.float64)
    for i in range(0, len(Wt), chunk):
        w1, b1 = Wt[i:i + chunk, 0:1], Wt[i:i + chunk, 1:2]
        ti, to = w1 * inner + b1, w1 * outer + b1
        out.append(((to + a * torch.sin(to)).min(dim=1).values - (ti + a * torch.sin(ti)).max(dim=1).values).numpy())
    return np.concatenate(out)


def first_crossing(G, warmup=WARMUP):
    """ramp.batch_ramp's rule, every step: the first step t ≥ 1 with G(θ_t) > 0.  If t ≤ warmup the run was placed in
    the held phase (not a crossing; the run is done).  Returns (crossed, step t or None, placed_in_warmup)."""
    hit = np.nonzero(np.asarray(G[1:], float) > 0)[0]
    if not len(hit):
        return False, None, False
    t = int(hit[0]) + 1
    return (False, None, True) if t <= warmup else (True, t, False)


def predict_r(L, gamma, eta, b1_cross, kappa_tab=None):
    """pred r = κ_k·χ with χ = γ/(η λ_min(H_pop)) (= χ_target by construction) and k = round((b₁ − b₁*)/2π) at the
    crossing (ramp.predict_run's SGD rule, with the cell's η)."""
    from .lag_law import TWO_PI, kappa_k
    H, tan, dG, z = L["H"], L["tan"], L["dG"], L["z"]
    k = int(round((b1_cross - z[1]) / TWO_PI))
    lam = float(np.linalg.eigvalsh(H).min())
    chi = gamma / (eta * lam)
    kap = float(kappa_k(H, tan, dG, np.ones(3), k))
    if kappa_tab is not None and k in kappa_tab:
        assert abs(kappa_tab[k] - kap) <= 1e-12 * max(1.0, abs(kap)), (k, kappa_tab[k], kap)
    return {"winding_k": k, "chi": chi, "kappa_k": kap, "pred_r": kap * chi}


def _path_file(a, kx, seed):
    return PATHS / f"a{a:.2f}_kx{kx:g}_seed{seed}.npz"


FORBIDDEN = {"crossed", "step_cross", "s_cross", "obs_r", "ratio", "placed_in_warmup", "gap", "G"}


def train():
    """All 12 cells × 40 seeds to the end of the ramp; NO gap or placement evaluated.  Each run's path (parameters after
    every step, and λ_max of its own-sample Hessian at every step) is saved (paths/*.npz, untracked) and hashed; one
    row per run is written as its cell completes (resumable by cell)."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    _assert_registration()
    PATHS.mkdir(parents=True, exist_ok=True)
    d = _read(OUT / "design.csv")
    fz = _read(OUT / "frozen_switches.csv"); fz["a2"] = fz.a.round(2)
    f = OUT / "predictions_parts.jsonl"
    done = set() if not f.exists() else {(round(r["a"], 2), r["kx_target"]) for r in map(json.loads, f.read_text().splitlines())}
    for r in d.itertuples():
        if (round(r.a, 2), r.kx_target) in done:
            continue
        W, lam = train_cell(r.a, int(r.winding), r.eta_cell, r.gamma, r.s_star_pop, int(r.total_steps))
        rows = []
        for i, sd in enumerate(SEEDS):
            pf = _path_file(r.a, r.kx_target, sd)
            np.savez(pf, W=W[:, i, :], lam_max=lam[:, i])
            fr = fz[(fz.a2 == round(r.a, 2)) & (fz.seed == sd)].iloc[0]
            sb, lmo = float(fr.s_branch), float(fr.lambda_min_own) if pd.notna(fr.lambda_min_own) else float("nan")
            rows.append({"a": r.a, "winding": int(r.winding), "kx_target": r.kx_target, "seed": sd, "eta_cell": r.eta_cell,
                         "gamma": r.gamma, "chi_target": r.chi_target, "n_steps": int(r.total_steps),
                         "s_branch": sb, "lambda_min_own": lmo, "chi_own": r.gamma / (r.eta_cell * lmo),
                         "pred_r_natural_winding": r.kappa * r.chi_target,
                         "max_eta_lambda_full_path": float(r.eta_cell * lam[:, i].max()),
                         "all_finite": bool(np.isfinite(W[:, i, :]).all() and np.isfinite(lam[:, i]).all()),
                         "path_sha256": _sha(pf)})
        with open(f, "a") as fh:
            for x in rows:
                fh.write(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in x.items()}) + "\n")
        print(r.a, r.kx_target, r.eta_cell, int(r.total_steps), "max η·λ_max over full paths",
              round(max(x["max_eta_lambda_full_path"] for x in rows), 4), flush=True)
        del W, lam
        _rss_guard()


def finalize():
    d = pd.DataFrame([json.loads(l) for l in (OUT / "predictions_parts.jsonl").read_text().splitlines()])
    d = d.sort_values(["a", "kx_target", "seed"]).reset_index(drop=True)
    assert len(d) == len(SETTINGS) * len(KX_TARGETS) * len(SEEDS) and not d.duplicated(["a", "kx_target", "seed"]).any()
    assert not (FORBIDDEN & set(d.columns))
    p = OUT / "predictions.csv"
    d.to_csv(p, index=False)
    h = _sha(p)
    (OUT / "predictions.sha256").write_text(f"{h}  predictions.csv\n")
    print(h, len(d))


def _assert_predictions():
    h = (OUT / "predictions.sha256").read_text().split()[0]
    assert _sha(OUT / "predictions.csv") == h, "predictions.csv changed since its commit"
    return h


# ------------------------------------------------------------------------------------------ observation and scoring
def observe():
    """After the predictions commit: assert every hash, then every-step crossing detection on each saved path, the
    realised stability check, validity per cell and the registered verdicts."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    _assert_registration()
    ph = _assert_predictions()
    L = R._landscape()
    kt = _read(OUT / "kappa_k.csv")
    pr = _read(OUT / "predictions.csv")
    parts = OUT / "observed_parts.jsonl"
    done = set() if not parts.exists() else {(round(r["a"], 2), r["kx_target"], r["seed"])
                                             for r in map(json.loads, parts.read_text().splitlines())}
    d = _read(OUT / "design.csv")
    for r in pr.itertuples():
        if (round(r.a, 2), r.kx_target, r.seed) in done:
            continue
        pf = _path_file(r.a, r.kx_target, r.seed)
        assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
        z = np.load(pf); W, lam = z["W"], z["lam_max"]
        assert len(W) == r.n_steps + 1 and len(lam) == r.n_steps
        G = gap_path(W, r.a)
        crossed, t_c, pw = first_crossing(G)
        s0 = S0_FRAC * R._s_star_pop(r.a)
        m_stab, stab = realised_stability(r.eta_cell, lam, t_c, r.n_steps)
        row = {"a": r.a, "kx_target": r.kx_target, "seed": int(r.seed), "crossed": crossed, "step_cross": t_c,
               "placed_in_warmup": pw, "max_eta_lambda_realised": m_stab, "stability_holds": stab,
               "G_end": float(G[-1]), "w1_end": float(W[-1, 0]), "b1_end": float(W[-1, 1])}
        if crossed:
            s_c = scale_at(t_c, s0, r.gamma)
            ktab = dict(zip(kt[kt.a.round(2) == round(r.a, 2)].k, kt[kt.a.round(2) == round(r.a, 2)].kappa_k))
            p = predict_r(L[round(r.a, 2)], r.gamma, r.eta_cell, float(W[t_c, 1]), ktab)
            obs = s_c / r.s_branch - 1 if np.isfinite(r.s_branch) else float("nan")
            row.update(s_cross=s_c, w1_cross=float(W[t_c, 0]), b1_cross=float(W[t_c, 1]), b2_cross=float(W[t_c, 2]),
                       **p, obs_r=obs, ratio=obs / p["pred_r"])
        with open(parts, "a") as fh:
            fh.write(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}) + "\n")
        _rss_guard()
    o = pd.DataFrame([json.loads(l) for l in parts.read_text().splitlines()])
    o = o.sort_values(["a", "kx_target", "seed"]).reset_index(drop=True)
    runs = pr.merge(o, on=["a", "kx_target", "seed"], how="left", validate="one_to_one")
    assert len(runs) == len(pr) and runs.crossed.notna().all()
    runs.to_csv(OUT / "observed_runs.csv", index=False)
    cells = []
    for r in d.itertuples():
        g = runs[(runs.a.round(2) == round(r.a, 2)) & np.isclose(runs.kx_target, r.kx_target)]
        assert len(g) == N_SEEDS
        sc = g[g.crossed.astype(bool) & np.isfinite(g.s_branch.astype(float))]
        cells.append({"a": r.a, "kx_target": r.kx_target, "chi_target": r.chi_target, "eta_cell": r.eta_cell,
                      "gamma": r.gamma, "ratios": sc.ratio.astype(float).tolist(), "n_cross": int(len(sc)),
                      "n_crossed_any": int(g.crossed.astype(bool).sum()), "n_runs": int(len(g)),
                      "n_placed_in_warmup": int(g.placed_in_warmup.astype(bool).sum()),
                      "chi_own_median": float(sc.chi_own.median()) if len(sc) else float("nan"),
                      "chi_own_median_all40": float(g.chi_own.median()),
                      "stab_ok": bool(g.stability_holds.astype(bool).all()),
                      "max_eta_lambda_realised": float(g.max_eta_lambda_realised.max()),
                      "n_runs_stability_violated": int((~g.stability_holds.astype(bool)).sum()),
                      "n_winding_changed": int((sc.winding_k != r.winding).sum()),
                      "median_obs_r": float(sc.obs_r.median()) if len(sc) else float("nan"),
                      "median_pred_r": float(sc.pred_r.median()) if len(sc) else float("nan"),
                      "ratio_q25": float(sc.ratio.quantile(0.25)) if len(sc) else float("nan"),
                      "ratio_q75": float(sc.ratio.quantile(0.75)) if len(sc) else float("nan")})
    S = score_cells(cells)
    out = {**{k: v for k, v in S.items() if k != "cells"},
           "cells": [{k: v for k, v in c.items() if k != "ratios"} for c in S["cells"]],
           "predictions_sha256": ph, "registration_sha256_file_sha256": _sha(OUT / "registration.sha256"),
           "n_runs": int(len(runs)), "n_crossed": int(runs.crossed.astype(bool).sum())}
    (OUT / "scores.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    {"branch": branch, "freeze": freeze, "feasibility": feasibility, "hashes": hashes, "train": train,
     "finalize": finalize, "observe": observe}[sys.argv[1]]()
