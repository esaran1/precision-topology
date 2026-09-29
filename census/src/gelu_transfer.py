"""GELU-T: a registered test of the lag law at GELU under free SGD from a declared start (two arms).

Design (approved by the author 2026-09-29): results/designs/GELU_transfer_design.md (draft 5e6323f).
Registration: results/gelu_transfer_registration.md.  Tests: tests/test_gelu_transfer.py.

Setting (Track 3A): width 1, z = w₂·GELU(w₁x + b₁) + b₂ (exact erf form), I = [−0.8, 0.8] class 0, O = ±[1.2, 2.0]
class 1, each seed's own 400-point sample fold1d.make_data(200, seed), float64, s = |w₂|.  Every-step detection with
phase2b_ordering.state (dense windows; G > 0 in w₂'s orientation).  s_pop = 6.645633 (kappa_gelu_frozen.json s_star, the
tracked population branch) is used only for s₀ = 0.5·s_pop and the branch-point start; s_glob = 6.641492 (3A's
registered switch) is L4's comparator.

Arms (the same 80 seeds 876,000–876,079; pilot seeds 876,900–876,909):
  random  PRIMARY.  Hidden (w₁, b₁, b₂) = coordinates 0, 1, 3 of U(−1, 1)⁴ drawn in float32 from a torch Generator
          seeded with the seed (the same draw as 3A's torch.manual_seed(seed); w₂'s draw is discarded) and cast to
          double; w₂ = +s₀.
  branch  MECHANISM CONTROL.  θ*_pop(s₀), the population branch point at s₀; even seeds the w₁ > 0 copy, odd seeds the
          mirror (w₁ → −w₁).
Hold: w₂ fixed at +s₀; (w₁, b₁, b₂) by full-batch GD, lr 0.3, for W steps; G (dense) checked at the start state and
after every hold step.  W = W_copy = max(4000, ⌈25/(0.3·λ_min,own(s₀))⌉) of the assigned copy (branch arm), or the larger
of the two copies' W (random arm, whose copy is not known before release).
Release: fresh torch SGD (no momentum), lr η = 0.03 on (w₁, b₁, b₂) and ρ·η on w₂ (w₂'s gradient multiplied by ρ before
the step), 40,000 steps.  Nothing after release is evaluated for gap or placement until `observe`.
Classification at release: damped Newton at s₀ from the release state (accepted iff max|∇| < 1e−8 and H positive
definite); on copy c iff the Newton point is within 1e−6 (sup) of the frozen θ*_own,c(s₀) AND the release state is
within 1e−3 of it.  random arm: on-branch iff on either copy (the run gets that copy's prediction); branch arm: on
target iff on the assigned copy.  Other runs get no prediction and are counted, never replaced.
Gates (author's decision 2026-09-29): random ≥ 80% on-branch at release, regardless of G in the hold (runs with G > 0
at the start or in the hold are scored, flagged, and excluded only in a DESCRIPTIVE sensitivity analysis); branch ≥ 90%
on target AND no run with G > 0 at any hold step.
Follow check (author's decision): at the first step with s_t ≥ 0.8·s_switch,branch, Newton at that s from the run's
state, classified against both copies' frozen branch points at 0.8·s_switch carried to that s; a run whose copy there
differs from its release copy (or is neither) is not scored (counted).  V7 is the q90 of max χ_t over the part of the
path with s_t ≥ 0.8·s_switch,branch up to t_sw.  κχ is SIGNED everywhere (pilot rule, V5, predictions): κ's sign is
invariant under the mirror coordinates, the sign convention of G and the direction of s (kappa_transforms,
kappa_mirror_recomputed), so a negative κ is a predicted crossing before the switch.

Frozen per seed and copy (freeze): θ*_own(s₀) (Newton from θ*_pop(s₀) or its mirror), λ_min(s₀), W_copy,
s_switch,branch (act_fold pseudo-arclength continuation from s₀ to 1.6·s_pop; the first G sign change, bisected to
1e−10; validated by step halving (1e−6 relative), no turning point or branch point and λ_min > 0 on [s₀, s_switch], and
the exact-extrema enclosure decided unplaced at 0.999·s_switch and placed at 1.001·s_switch), κ_seed (own H, θ*′ and ∇G
at s_switch, P = I; ∇G one-sided agreement ≤ 1e−5 in both components) and λ_min(H(s_switch)).

Predictions per on-branch run (from the release state and the s path only):
  r_traj  Track A's R4 recursion (linear_response.simulate, SGD form: P = I, no momentum, lr η) from release with
          δ₀ = θ_release − θ*(s₀) on the occupied branch, θ*(s_t) and H(s_t) of that branch along the run's own s_t,
          predicted crossing at the first step with G(θ*(s_t) + δ_t) > 0 (exact extrema); r_traj = s_traj/s_switch − 1.
          No prediction if the one-step map has spectral radius > 1 (checked every 10th step) or sup|δ| > 1.
  r_cf    κ_seed·χ, χ = (ṡ/s_switch)/(η·λ_min(H(s_switch))), ṡ = (s_{t_sw} − s_{t_sw−w})/w, w = min(100, t_sw),
          t_sw = the first step with s_t ≥ s_switch.

    python -m src.gelu_transfer landscape   # population branch point at s₀          -> results/gelu_transfer/landscape.json
    python -m src.gelu_transfer freeze      # per seed and copy (pilot + registered) -> frozen_seeds.json
    python -m src.gelu_transfer pilot       # the pilot rule (pilot seeds, to t_sw)  -> pilot.json
    python -m src.gelu_transfer hashes      # registration.sha256
    python -m src.gelu_transfer train       # registered runs, predictions           -> predictions_parts.jsonl
    python -m src.gelu_transfer finalize    # predictions.csv + predictions.sha256
    python -m src.gelu_transfer observe     # AFTER the predictions commit: every-step detection, gates, scores
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import linear_response as LR

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "gelu_transfer"
PATHS = OUT / "paths"
KAPPA_JSON = RESULTS / "act_general" / "kappa_gelu_frozen.json"
KAPPA_SHA256 = "9a4848d27bbfb10e1ecf031f01abe8fec5973d75d4dd974f15e81ee3ac1db779"   # registered in Track 3A

ACT = "gelu"
SEEDS = tuple(range(876_000, 876_080))
PILOT_SEEDS = tuple(range(876_900, 876_910))
PRIMARY, CONTROL = "random", "branch"
ARMS = (PRIMARY, CONTROL)
COPIES = (1, -1)                       # +1: the w₁ > 0 copy (target); −1: the mirror copy

S0_FRAC = 0.5
HOLD_LR, W_MIN, W_RELAX = 0.3, 4000, 25.0
ETA, BUDGET = 0.03, 40_000
NEWTON_GTOL, ON_TOL, STATE_TOL = 1e-8, 1e-6, 1e-3
FOLLOW_FRAC = 0.8                      # branch identity re-checked at the first step with s_t ≥ 0.8·s_switch,branch
FOLLOW_STATE_TOL = None                # OPEN (author): state-to-branch tolerance at that step; None = Newton identity only
S_HI_FRAC = 1.6
CONT_H0, CONT_HMAX = 0.01, 0.05        # act_fold defaults; step halving uses half of both
HALVING_REL = 1e-6
DECIDE_REL = 1e-3
DG_ONESIDED_MAX = 1e-5
GRID_H_FRAC = 0.002                    # branch grid spacing for the recursion (Track A: 0.002·s*)
SDOT_WIN = 100

# registered criteria and conditions
GATE_FRAC = {PRIMARY: 0.80, CONTROL: 0.90}
L1_BAND, L2_BAND, L3_MIN = (0.90, 1.10), (0.80, 1.20), 0.5
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 876_000, (2.5, 97.5)
MIN_CROSS = 60
TSW_MIN_FRAC = 0.90
REGIME_MAX, REGIME_MIN_FRAC = 0.5, 0.80
LAG_MIN_STEPS = 10.0
KC_Q90_MAX = 0.1
CHI_REL_TOL = 0.30
CHI_PATH_Q90_MAX = 0.25
PILOT_KC_MAX, PILOT_MAX_HALVINGS = 0.1, 3

RSS_LIMIT = 3 * 1024 ** 3              # machine rule: stop any job over 3 GB RSS

REGISTERED_FILES = ("src/gelu_transfer.py", "tests/test_gelu_transfer.py", "results/gelu_transfer_registration.md",
                    "results/designs/GELU_transfer_design.md",
                    "results/gelu_transfer/landscape.json", "results/gelu_transfer/frozen_seeds.json",
                    "results/gelu_transfer/pilot.json", "results/act_general/kappa_gelu_frozen.json",
                    "src/act_general.py", "src/act_fold.py", "src/linear_response.py", "src/lag_law.py",
                    "src/track_a.py", "src/phase2b_ordering.py", "src/fold1d.py", "src/width2_geometry.py",
                    "src/width2_conditional.py")


# ------------------------------------------------------------------------------------------ rules (pure functions)
def w_hold(lam):
    """Hold length for a copy with λ_min,own(s₀) = lam: max(4000, ⌈25/(0.3·lam)⌉).  lam ≤ 0 or non-finite: None."""
    if lam is None or not np.isfinite(lam) or lam <= 0:
        return None
    return int(max(W_MIN, math.ceil(W_RELAX / (HOLD_LR * lam))))


def assigned_copy(seed):
    """Branch arm: even seeds the w₁ > 0 copy (+1), odd seeds the mirror (−1)."""
    return 1 if int(seed) % 2 == 0 else -1


def arm_hold_steps(arm, seed, W_by_copy):
    """W for a run: the assigned copy's W (branch arm); the larger of the two copies' W (random arm).  None if a needed
    copy has no W."""
    if arm == CONTROL:
        return W_by_copy.get(assigned_copy(seed))
    ws = [W_by_copy.get(c) for c in COPIES]
    return None if any(w is None for w in ws) else int(max(ws))


def classify_release(z_release, z_newton, newton_ok, copies, state_tol=STATE_TOL):
    """Copy the run is on: +1, −1 or 0 (neither).  copies = {c: that copy's branch point at this s, or None}.  On c iff
    the Newton point is accepted and within ON_TOL (sup) of the branch point and the state within state_tol of it
    (state_tol None: no state condition)."""
    if not newton_ok:
        return 0
    z_release, z_newton = np.asarray(z_release, float), np.asarray(z_newton, float)
    hits = [c for c, zc in copies.items() if zc is not None
            and np.abs(z_newton - np.asarray(zc, float)).max() <= ON_TOL
            and (state_tol is None or np.abs(z_release - np.asarray(zc, float)).max() <= state_tol)]
    assert len(hits) <= 1, "a state cannot be on both copies"
    return hits[0] if hits else 0


def is_on_branch(arm, seed, copy):
    """random arm: on either copy; branch arm: on the assigned copy."""
    if arm == PRIMARY:
        return copy in COPIES
    return copy == assigned_copy(seed)


def t_switch(s_path, s_sw):
    """First index t with s_t ≥ s_switch (None if never)."""
    if s_sw is None or not np.isfinite(s_sw):
        return None
    return LR._first_ge(np.asarray(s_path, float), float(s_sw))


def closed_form(s_path, t_sw, s_sw, kappa, lam_sw, eta=ETA):
    """(r_cf, χ, ṡ): χ = (ṡ/s_switch)/(η·λ_min), ṡ = (s_{t_sw} − s_{t_sw−w})/w with w = min(100, t_sw) (Track A's R5).
    NaN if t_sw is undefined or 0."""
    if t_sw is None or t_sw < 1:
        return float("nan"), float("nan"), float("nan")
    s = np.asarray(s_path, float)
    w = min(SDOT_WIN, int(t_sw))
    sdot = (s[t_sw] - s[t_sw - w]) / w
    chi = (sdot / s_sw) / (eta * lam_sw)
    return float(kappa * chi), float(chi), float(sdot)


def chi_window_max(s_path, t_sw, chi, s_sw, frac=FOLLOW_FRAC):
    """V7's statistic: max of χ_t over 0 ≤ t < t_sw with s_t ≥ frac·s_switch (NaN if the window is empty or any χ_t in
    it is not finite)."""
    if t_sw is None or t_sw < 1 or len(chi) == 0:
        return float("nan")
    s = np.asarray(s_path, float)[:int(t_sw)]
    w = np.asarray(chi, float)[s >= frac * s_sw]
    return float(w.max()) if len(w) and np.all(np.isfinite(w)) else float("nan")


def first_at_fraction(s_path, s_sw, frac=FOLLOW_FRAC):
    """The first step with s_t ≥ frac·s_switch (None if never)."""
    return t_switch(s_path, frac * s_sw) if s_sw is not None and np.isfinite(s_sw) else None


def follows_branch(copy_release, copy_follow):
    """The run is on the same copy at the follow check as at release (neither at the check: does not follow)."""
    return bool(copy_release in COPIES and copy_follow == copy_release)


def chi_path(s_path, t_sw, lam_of_s, eta=ETA):
    """χ_t = ((s_{t+1} − s_t)/s_t)/(η·λ_min(H(s_t))) for t = 0 … t_sw − 1 (Corollary L3's χ_t with the one-step ṡ).
    lam_of_s(s) → λ_min of the occupied branch's Hessian at s (NaN outside the branch range)."""
    if t_sw is None or t_sw < 1:
        return np.array([])
    s = np.asarray(s_path, float)
    return np.array([((s[t + 1] - s[t]) / s[t]) / (eta * lam_of_s(s[t])) for t in range(int(t_sw))])


def q90(x):
    """90th percentile (numpy's default linear interpolation); NaN for an empty input or any NaN."""
    x = np.asarray(x, float)
    if len(x) == 0 or not np.all(np.isfinite(x)):
        return float("nan")
    return float(np.percentile(x, 90))


def pilot_rho(q):
    """ρ = min(1, 2^⌊log₂(0.1/q90)⌋); q90 ≤ 0 gives ρ = 1."""
    if q <= 0:
        return 1.0
    return float(min(1.0, 2.0 ** math.floor(math.log2(PILOT_KC_MAX / q))))


def pilot_rule(run_pilot):
    """The registered pilot rule.  run_pilot(ρ) → the pilot runs' SIGNED κ_seed·χ at t_sw (on-branch runs only); q90 is
    the 90th percentile of the signed values.  ρ = min(1, 2^⌊log₂(0.1/q90)⌋) at ρ = 1; if ρ < 1 the
    pilot is rerun at ρ and ρ is kept only if q90 ≤ 0.1, otherwise ρ is halved and the pilot rerun, up to three halvings;
    then STOP.  No pilot value at all: STOP."""
    hist = []
    kc = np.asarray(run_pilot(1.0), float)
    q = q90(kc)
    hist.append({"rho": 1.0, "n": int(len(kc)), "q90_kc": q})
    if not np.isfinite(q):
        return {"status": "STOP", "rho": None, "history": hist, "reason": "no pilot value"}
    rho = pilot_rho(q)
    if rho == 1.0:
        return {"status": "ok", "rho": 1.0, "history": hist}
    for k in range(PILOT_MAX_HALVINGS + 1):
        kc = np.asarray(run_pilot(rho), float)
        q = q90(kc)
        hist.append({"rho": rho, "n": int(len(kc)), "q90_kc": q})
        if np.isfinite(q) and q <= PILOT_KC_MAX:
            return {"status": "ok", "rho": rho, "history": hist}
        if k == PILOT_MAX_HALVINGS:
            break
        rho /= 2
    return {"status": "STOP", "rho": None, "history": hist, "reason": "q90 > 0.1 after three halvings"}


def bootstrap_mean_ci(D, n_boot=BOOT_N, seed=BOOT_SEED):
    """(mean, lower, upper): 95% percentile bootstrap interval of the mean of D over run resamples (numpy
    default_rng(seed), n_boot resamples of size len(D) with replacement)."""
    D = np.asarray(D, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    means = D[idx].mean(axis=1)
    lo, hi = np.percentile(means, BOOT_PCT)
    return float(D.mean()), float(lo), float(hi)


def gate(arm, on_branch, hold_positive):
    """Registered gate.  random: fraction on-branch (target or mirror) at release ≥ 0.80, regardless of G in the hold.
    branch: fraction on target ≥ 0.90 AND no run with G > 0 at any hold step (the start state included)."""
    on_branch = np.asarray(on_branch, bool); hold_positive = np.asarray(hold_positive, bool)
    frac = float(on_branch.mean()) if len(on_branch) else float("nan")
    ok_frac = bool(len(on_branch) and frac >= GATE_FRAC[arm])
    ok_hold = not bool(hold_positive.any())
    return {"n_runs": int(len(on_branch)), "n_on_branch": int(on_branch.sum()), "frac_on_branch": frac,
            "min_frac": GATE_FRAC[arm], "n_hold_G_positive": int(hold_positive.sum()),
            "frac_ok": ok_frac, "hold_ok": ok_hold,
            "pass": bool(ok_frac and (ok_hold or arm == PRIMARY)), "hold_condition_applies": arm == CONTROL}


def score_arm(arm, on_branch, hold_positive, crossed, step_obs, s_obs, s_sw, s_traj, r_traj, r_cf, kappa, t_sw,
              eta_lam, lag_steps, chi_tsw, chi_path_max, pilot_chi_median, s_glob, follows=None, exclude=None):
    """Registered verdicts for one arm.  Arrays over the arm's 80 runs (NaN where undefined).

    Gate first (fail: every criterion UNRESOLVED, nothing scored).  Scored run: on-branch at release, on the same copy
    at the follow check (`follows`; None: all True), crossed, finite r_traj, r_cf and r_obs, and not in `exclude` (the
    DESCRIPTIVE sensitivity analysis only; None in the registered scoring).  All lags are SIGNED: a negative-κ run has a
    negative predicted lag (predicted crossing before its switch); ratios r_obs/r_pred are of signed lags.  Validity
    V1-V7 (any fails: L1-L5 UNRESOLVED); V2 counts only crossing runs with κ_seed > 0 (predicted-late); predicted-early
    runs (κ_seed ≤ 0) are scored and not in V2.  V5 is the q90 of the SIGNED κχ.  L4/L5 use s_pred = s_traj."""
    A = lambda v, t=float: np.asarray(v, t)
    on_branch, hold_positive, crossed = A(on_branch, bool), A(hold_positive, bool), A(crossed, bool)
    step_obs, s_obs, s_sw, s_traj, r_traj, r_cf, kappa, t_sw, eta_lam, lag_steps, chi_tsw, chi_path_max = (
        A(v) for v in (step_obs, s_obs, s_sw, s_traj, r_traj, r_cf, kappa, t_sw, eta_lam, lag_steps, chi_tsw,
                       chi_path_max))
    names = ("L1", "L2", "L3", "L4", "L5")
    follows = np.ones(len(on_branch), bool) if follows is None else A(follows, bool)
    exclude = np.zeros(len(on_branch), bool) if exclude is None else A(exclude, bool)
    g = gate(arm, on_branch, hold_positive)
    out = {"arm": arm, "role": "primary" if arm == PRIMARY else "mechanism control", "gate": g}
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
    kpos = on_branch & crossed & np.isfinite(kappa) & (kappa > 0)
    tsw_before = kpos & np.isfinite(t_sw) & np.isfinite(step_obs) & (t_sw < step_obs)
    frac_tsw = float(tsw_before.sum() / kpos.sum()) if kpos.sum() else float("nan")
    S = lambda v: v[scored]
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    regime_frac = float(np.mean(S(eta_lam) <= REGIME_MAX)) if n else float("nan")
    lag_med = med(S(lag_steps))
    kc_q90 = q90(S(r_cf))
    chi_med = med(S(chi_tsw))
    chi_rel = abs(chi_med / pilot_chi_median - 1) if n and np.isfinite(pilot_chi_median) and pilot_chi_median else float("nan")
    chi_path_q90 = q90(S(chi_path_max))
    V = {"V1_min_crossing_with_prediction": n >= MIN_CROSS,
         "V2_tsw_before_crossing_90pct_kappa_pos": bool(kpos.sum() == 0 or frac_tsw >= TSW_MIN_FRAC),
         "V3_regime_eta_lam_le_0p5_in_80pct": bool(n and regime_frac >= REGIME_MIN_FRAC),
         "V4_median_predicted_lag_ge_10_steps": bool(np.isfinite(lag_med) and lag_med >= LAG_MIN_STEPS),
         "V5_q90_kappa_chi_le_0p1": bool(np.isfinite(kc_q90) and kc_q90 <= KC_Q90_MAX),
         "V6_median_chi_within_30pct_of_pilot": bool(np.isfinite(chi_rel) and chi_rel <= CHI_REL_TOL),
         "V7_q90_max_chi_window_le_0p25": bool(np.isfinite(chi_path_q90) and chi_path_q90 <= CHI_PATH_Q90_MAX)}
    valid = all(V.values())
    out.update({"n_runs": int(len(on_branch)), "n_on_branch": int(on_branch.sum()),
                "n_neither_or_off_target": int((~on_branch).sum()), "n_crossed": int(crossed.sum()),
                "n_on_branch_not_following": int((on_branch & ~follows).sum()),
                "n_on_branch_hold_G_positive": int((on_branch & hold_positive).sum()),
                "n_excluded_sensitivity": int((on_branch & exclude).sum()),
                "n_kappa_nonpos_scored": int((scored & ~(kappa > 0)).sum()),
                "n_on_branch_crossed": int((on_branch & crossed).sum()), "n_scored": n,
                "n_on_branch_crossed_no_prediction": int((on_branch & crossed & ~scored).sum()),
                "n_kappa_pos_crossing": int(kpos.sum()), "frac_tsw_before_crossing_kappa_pos": frac_tsw,
                "regime_frac": regime_frac, "median_predicted_lag_steps": lag_med, "q90_kappa_chi_tsw": kc_q90,
                "median_chi_tsw": chi_med, "pilot_median_chi_tsw": float(pilot_chi_median),
                "chi_rel_to_pilot": chi_rel, "q90_max_chi_window": chi_path_q90,
                "validity": {k: bool(v) for k, v in V.items()}, "valid": bool(valid),
                "scored_index": np.nonzero(scored)[0].tolist()})
    ro, rt, rc = S(r_obs), S(r_traj), S(r_cf)
    so, sp, ssw = S(s_obs), S(s_traj), S(s_sw)

    def verdict(ok, computable):
        if not valid or not computable:
            return "UNRESOLVED"
        return "PASS" if ok else "FAIL"
    m1 = med(ro / rt) if n else float("nan")
    m2 = med(ro / rc) if n else float("nan")
    out["L1"] = {"n": n, "median_ratio": m1, "verdict": verdict(L1_BAND[0] <= m1 <= L1_BAND[1], n >= 1)}
    out["L2"] = {"n": n, "median_ratio": m2, "verdict": verdict(L2_BAND[0] <= m2 <= L2_BAND[1], n >= 1)}
    from .track_a import spearman
    rho = spearman(rt, ro) if n >= 3 else float("nan")
    out["L3"] = {"n": n, "spearman": rho, "verdict": verdict(rho >= L3_MIN, n >= 3 and np.isfinite(rho))}
    for k, ref in (("L4", np.full(n, float(s_glob))), ("L5", ssw)):
        D = np.abs(np.log(so / sp)) - np.abs(np.log(so / ref))
        mean, lo, hi = bootstrap_mean_ci(D) if n >= 2 else (float("nan"),) * 3
        out[k] = {"n": n, "mean_D": mean, "ci95": [lo, hi], "verdict": verdict(hi < 0, n >= 2 and np.isfinite(hi)),
                  "comparator": "s_glob" if k == "L4" else "s_switch_branch"}
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


# ------------------------------------------------------------------------------------------ helpers
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")


def _memory_guard():
    from .act_fold import check_memory
    ok, f, w = check_memory()
    if not ok:
        raise SystemExit(f"STOP (memory): free {f}%, swap free {w} MB (need >= 25% and >= 500 MB)")
    return f, w


def _setup_process():
    os.nice(15)
    import torch
    torch.set_num_threads(1)


def kappa_frozen():
    assert _sha(KAPPA_JSON) == KAPPA_SHA256, "kappa_gelu_frozen.json differs from its registered version"
    return json.loads(KAPPA_JSON.read_text())


def s_values():
    K = kappa_frozen()
    return float(K["s_star"]), float(K["s_glob"]), S0_FRAC * float(K["s_star"])


def _act():
    from .act_general import GAct
    return GAct(ACT)


class PopProblem:
    """The 800-point population objective in act_fold.Problem's interface (σ = +1)."""

    def __init__(self):
        import torch
        from .act_fold import Problem
        from .act_general import population
        self.act = _act(); self.sigma = 1.0
        x, y = population()
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.X, self.Y = torch.tensor(self.x, dtype=torch.float64), torch.tensor(self.y, dtype=torch.float64)
        for k in ("_L", "F", "J", "loss_grad", "gap"):
            setattr(self, k, getattr(Problem, k).__get__(self))


def own_problem(seed):
    from .act_fold import Problem
    return Problem(ACT, int(seed), 1.0)


def hessian_z(z, s, P):
    """Hessian of the loss in z = (w₁, b₁, b₂) at w₂ = +s (no solve, so a singular H is returned, not raised)."""
    import torch
    q = torch.tensor(np.r_[np.asarray(z, float), s], dtype=torch.float64)
    return torch.autograd.functional.hessian(P._L, q).numpy()[:3, :3]


def newton_at(z0, s, P):
    """Damped Newton (act_general.branch_point) at w₂ = +s; returns (z, max|∇|, λ_min(H), accepted).  Accepted iff
    max|∇| < 1e−8 and H positive definite; a singular or non-finite H is not accepted."""
    from .act_general import branch_point
    z, g = branch_point(np.asarray(z0, float), s, P.x, P.y, P.act)
    H = hessian_z(z, s, P)
    lam = float(np.linalg.eigvalsh(0.5 * (H + H.T)).min()) if np.all(np.isfinite(H)) else float("nan")
    return z, float(g), lam, bool(np.isfinite(g) and g < NEWTON_GTOL and np.isfinite(lam) and lam > 0)


def dLds(z, s, P):
    import torch
    q = torch.tensor(np.r_[z, s], dtype=torch.float64, requires_grad=True)
    return float(torch.autograd.grad(P._L(q), q)[0][3])


def grad_gap_checked(w1, b1, act, h=1e-6):
    """∇G (central differences of the exact-extrema midpoint) and the larger one-sided relative disagreement over the
    two components."""
    from .act_general import gap_mid
    g0 = gap_mid(w1, b1, act)
    out, asym = [], 0.0
    for dw, db in ((h, 0.0), (0.0, h)):
        gp, gm = gap_mid(w1 + dw, b1 + db, act), gap_mid(w1 - dw, b1 - db, act)
        c = (gp - gm) / (2 * h)
        asym = max(asym, abs((gp - g0) / h - (g0 - gm) / h) / max(abs(c), 1e-12))
        out.append(c)
    return np.array([out[0], out[1], 0.0]), float(asym)


class GBranch(LR.Branch):
    """θ*(s) of one own-sample GELU branch on a grid (spacing h) by natural-parameter continuation with a tangent
    predictor and damped Newton, from an anchor; Hermite θ*, linear H (the interpolation of linear_response.Branch, which
    Track A's R4 uses).  A Newton failure, a non-positive-definite H or a jump ends the grid on that side."""

    def __init__(self, s_anchor, z_anchor, P, s_lo, s_hi, h):
        self.P = P
        z, g, lam, ok = newton_at(z_anchor, s_anchor, P)
        self.anchor_ok = ok
        pts = {s_anchor: self._pt(z, s_anchor)}
        for direction in (+1, -1):
            s, zc = s_anchor, z.copy()
            nxt = (math.floor(s_anchor / h) + (1 if direction > 0 else 0)) * h
            while (direction > 0 and nxt <= s_hi) or (direction < 0 and nxt >= s_lo):
                pred = zc + pts[s][2] * (nxt - s)
                zn, gn, lamn, okn = newton_at(pred, nxt, P)
                jump = np.abs(zn - pred).max()
                if not okn or jump > 1e-2 + 0.5 * np.abs(pred - zc).max():
                    break
                pts[nxt] = self._pt(zn, nxt)
                s, zc = nxt, zn
                nxt = nxt + direction * h
        self.s = np.array(sorted(pts))
        self.th = np.array([pts[k][0] for k in self.s])
        self.H = np.array([pts[k][1] for k in self.s])
        self.tan = np.array([pts[k][2] for k in self.s])
        self.lo, self.hi = float(self.s[0]), float(self.s[-1])

    def _pt(self, z, s):
        from .act_general import hessian_and_tangent
        H, tan = hessian_and_tangent(z, s, self.P.x, self.P.y, self.P.act)
        return np.asarray(z, float).copy(), H, tan

    def lam_min(self, s):
        if not self.contains(s):
            return float("nan")
        return float(np.linalg.eigvalsh(self.hess(s)).min())


# ------------------------------------------------------------------------------------------ landscape (population)
def landscape():
    """θ*_pop(s₀): act_fold continuation of the tracked population branch from (z*, s_pop) down to s₀, then damped
    Newton at s₀.  Validated: no turning point or branch point on the way, λ_min > 0, step halving agrees to 1e−6
    (sup), direct damped Newton from z* agrees, G(s₀) < 0, dL/ds(s₀) < 0.  Writes landscape.json; STOP if invalid."""
    from .act_fold import continue_branch
    from .act_general import hessian_and_tangent
    from .lag_law import kappa
    OUT.mkdir(parents=True, exist_ok=True)
    K = kappa_frozen()
    s_pop, s_glob, s0 = s_values()
    P = PopProblem()
    X0 = np.r_[K["z_star"], s_pop]
    res = {}
    for tag, h0, hm in (("base", CONT_H0, CONT_HMAX), ("halved", CONT_H0 / 2, CONT_HMAX / 2)):
        ev, path = continue_branch(P.F, P.J, X0, -1, s_stop=s0, gapf=P.gap, h0=h0, hmax=hm)
        near = min(path, key=lambda p: abs(p["s"] - s0))
        z, g, lam, ok = newton_at(near["z"], s0, P)
        res[tag] = {"ev": ev, "z": z, "grad": g, "lam": lam, "ok": ok, "n_path": len(path),
                    "min_lam_path": float(min(p["lam"] for p in path))}
    zd, gd, lamd, okd = newton_at(K["z_star"], s0, P)
    z0 = res["base"]["z"]
    H0, tan0 = hessian_and_tangent(z0, s0, P.x, P.y, P.act)
    G0 = P.gap(np.r_[z0, s0])
    dl = dLds(z0, s0, P)
    zs, gs, lams, oks = newton_at(K["z_star"], s_pop, P)
    Hs, tans = hessian_and_tangent(zs, s_pop, P.x, P.y, P.act)
    dGs, asym_s = grad_gap_checked(zs[0], zs[1], P.act)
    ev = res["base"]["ev"]
    checks = {"continuation_reached_s0": ev["end"] == "s_stop",
              "no_turning_point": ev["fold"] is None and res["halved"]["ev"]["fold"] is None,
              "no_branch_point": ev["branch_point"] is None and res["halved"]["ev"]["branch_point"] is None,
              "lam_min_positive_on_path": res["base"]["min_lam_path"] > 0 and res["halved"]["min_lam_path"] > 0,
              "newton_accepted": res["base"]["ok"] and res["halved"]["ok"],
              "halving_agrees_1e-6": float(np.abs(res["base"]["z"] - res["halved"]["z"]).max()) <= HALVING_REL,
              "direct_newton_agrees_1e-6": okd and float(np.abs(zd - z0).max()) <= HALVING_REL,
              "unplaced_at_s0": G0 < 0, "dLds_negative_at_s0": dl < 0,
              "z_star_reproduced_1e-8": float(np.abs(zs - np.array(K["z_star"])).max()) <= 1e-8}
    out = {"act": ACT, "s_pop": s_pop, "s_glob": s_glob, "s0": s0, "kappa_json_sha256": KAPPA_SHA256,
           "z_pop_s0": z0.tolist(), "grad_s0": res["base"]["grad"], "lam_min_s0": res["base"]["lam"],
           "H_s0": H0.tolist(), "tangent_s0": tan0.tolist(), "G_s0": G0, "dLds_s0": dl,
           "z_pop_s0_halved": res["halved"]["z"].tolist(), "z_pop_s0_direct_newton": zd.tolist(),
           "continuation_end": ev["end"], "continuation_n_path": res["base"]["n_path"],
           "min_lam_path": res["base"]["min_lam_path"], "min_lam_path_halved": res["halved"]["min_lam_path"],
           "at_s_pop": {"z": zs.tolist(), "grad": gs, "H": Hs.tolist(), "tangent": tans.tolist(), "gradG": dGs.tolist(),
                        "gradG_onesided_rel": asym_s, "kappa_sgd": kappa(Hs, tans, dGs, np.ones(3))[0],
                        "lam_min": lams},
           "checks": checks, "validated": bool(all(checks.values()))}
    (OUT / "landscape.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("H_s0", "at_s_pop")}, indent=1))
    if not out["validated"]:
        raise SystemExit("STOP: population landscape at s0 not validated")
    return out


def _land():
    return json.loads((OUT / "landscape.json").read_text())


# ------------------------------------------------------------------------------------------ per-seed frozen inputs
def freeze_copy(P, z_pop, c, s0, s_pop):
    """Own-sample branch of copy c at s₀ and its validated switch and κ."""
    from .act_fold import continue_branch
    from .act_general import gplus, hessian_and_tangent
    from .lag_law import kappa
    z_start = np.array([c * z_pop[0], z_pop[1], z_pop[2]])
    z, g, lam0, ok = newton_at(z_start, s0, P)
    row = {"copy": c, "z_s0": z.tolist(), "grad_s0": g, "lam_min_s0": lam0, "newton_ok": ok,
           "W": w_hold(lam0) if ok else None}
    if not ok:
        return {**row, "valid": False, "note": "no accepted own-sample minimum at s0"}
    X0 = np.r_[z, s0]
    row["G_s0"] = P.gap(X0)
    runs = {}
    for tag, h0, hm in (("base", CONT_H0, CONT_HMAX), ("halved", CONT_H0 / 2, CONT_HMAX / 2)):
        ev, path = continue_branch(P.F, P.J, X0, +1, s_stop=S_HI_FRAC * s_pop, gapf=P.gap, h0=h0, hmax=hm)
        runs[tag] = (ev, path)
    ev, path = runs["base"]
    sw = ev["switch"]
    sw_h = runs["halved"][0]["switch"]
    row.update(end=ev["end"], fold_s=ev["fold"]["s"] if ev["fold"] else None,
               branch_point_s=ev["branch_point"]["s"] if ev["branch_point"] else None,
               s_switch=sw["s"] if sw else None, s_switch_halved=sw_h["s"] if sw_h else None,
               switch_to_placed=sw["to_placed"] if sw else None)
    if not sw or not sw_h:
        return {**row, "valid": False, "note": "no switch on [s0, 1.6 s_pop]"}
    s_sw = sw["s"]
    lam_seg = [p["lam"] for p in path if p["s"] <= s_sw]
    row["min_lam_to_switch"] = float(min(lam_seg + [sw["lam"]]))
    row["halving_ok"] = bool(abs(s_sw - sw_h["s"]) <= HALVING_REL * abs(s_sw))
    row["no_turn_before_switch"] = bool(ev["fold"] is None or ev["fold"]["s"] > s_sw)
    row["no_branch_point_before_switch"] = bool(ev["branch_point"] is None or ev["branch_point"]["s"] > s_sw)
    zsw, gsw, lamsw_n, oksw = newton_at(sw["z"], s_sw, P)
    dec = {}
    for f in (1 - DECIDE_REL, 1 + DECIDE_REL):
        zf, gf, lf, okf = newton_at(zsw, f * s_sw, P)
        dec[f] = (gplus(zf[0], zf[1], 1.0, P.act), okf, float(np.abs(zf - zsw).max()))
    lo_enc, hi_enc = dec[1 - DECIDE_REL][0], dec[1 + DECIDE_REL][0]
    row.update(G_enc_below=list(lo_enc), G_enc_above=list(hi_enc),
               decided_ok=bool(lo_enc[1] < 0 and hi_enc[0] > 0 and dec[1 - DECIDE_REL][1] and dec[1 + DECIDE_REL][1]))
    H, tan = hessian_and_tangent(zsw, s_sw, P.x, P.y, P.act)
    dG, asym = grad_gap_checked(zsw[0], zsw[1], P.act)
    kap, lam_sw, _, _ = kappa(H, tan, dG, np.ones(3))
    row.update(z_switch=zsw.tolist(), grad_switch=gsw, kappa=kap, lam_min_switch=lam_sw, gradG=dG.tolist(),
               gradG_onesided_rel=asym, gradG_ok=bool(asym <= DG_ONESIDED_MAX), H_switch=H.tolist(),
               tangent_switch=tan.tolist())
    row["valid"] = bool(ok and row["G_s0"] < 0 and sw["to_placed"] and row["halving_ok"] and row["no_turn_before_switch"]
                        and row["no_branch_point_before_switch"] and row["min_lam_to_switch"] > 0 and row["decided_ok"]
                        and row["gradG_ok"] and oksw)
    return row


D_MIRROR = np.array([-1.0, 1.0, 1.0])      # z = (w₁, b₁, b₂) → (−w₁, b₁, b₂) under x → −x


def kappa_transforms(H, tan, dG):
    """κ (P = I) of (H, θ*′, ∇G) and under the three reparametrisations/conventions: the mirror coordinates
    (H → DHD, θ*′ → Dθ*′, ∇G → D∇G, D = diag(−1, 1, 1)), the sign of G (∇G → −∇G) and the direction of s (θ*′ → −θ*′).
    κ = λ_min·[∇G·H⁻¹θ*′]/[∇G·θ*′] is invariant under all three (each rescales numerator and denominator alike)."""
    from .lag_law import kappa
    H, tan, dG = (np.asarray(v, float) for v in (H, tan, dG))
    D = np.diag(D_MIRROR)
    one = np.ones(3)
    return {"kappa": kappa(H, tan, dG, one)[0], "mirror_coords": kappa(D @ H @ D, D @ tan, D @ dG, one)[0],
            "G_sign": kappa(H, tan, -dG, one)[0], "s_direction": kappa(H, -tan, dG, one)[0]}


def mirrored_problem(seed):
    """The seed's own sample mirrored, x → −x (same labels): the loss at (−w₁, b₁, b₂) equals the original loss at
    (w₁, b₁, b₂)."""
    P = own_problem(seed)
    P.X = -P.X
    P.x = P.X.numpy().astype(float)
    return P


def kappa_mirror_recomputed(seed, z_switch, s_sw):
    """κ recomputed from scratch on the mirrored sample at the mirror image of a frozen switch point (Newton there,
    H, θ*′ and ∇G of the mirrored problem).  Equal to the frozen κ if its sign and size are not a coordinate artefact."""
    from .act_general import hessian_and_tangent
    from .lag_law import kappa
    Q = mirrored_problem(seed)
    zm, g, lam, ok = newton_at(D_MIRROR * np.asarray(z_switch, float), s_sw, Q)
    H, tan = hessian_and_tangent(zm, s_sw, Q.x, Q.y, Q.act)
    dG, _ = grad_gap_checked(zm[0], zm[1], Q.act)
    return {"kappa": kappa(H, tan, dG, np.ones(3))[0], "newton_ok": ok,
            "dist_from_mirror_image": float(np.abs(zm - D_MIRROR * np.asarray(z_switch, float)).max())}


def branch_point_at(P, z_s0, s0, s_target):
    """Copy branch point at s_target by act_fold continuation from (θ*_own(s₀), s₀), then damped Newton at s_target;
    validated by step halving (1e−6 sup) and no turning point on the way.  Returns z, θ*′ and the checks."""
    from .act_fold import continue_branch
    from .act_general import hessian_and_tangent
    X0 = np.r_[np.asarray(z_s0, float), s0]
    res = []
    for h0, hm in ((CONT_H0, CONT_HMAX), (CONT_H0 / 2, CONT_HMAX / 2)):
        ev, path = continue_branch(P.F, P.J, X0, +1, s_stop=s_target, gapf=None, h0=h0, hmax=hm)
        below = [p for p in path if p["s"] <= s_target] or path[:1]
        z, g, lam, ok = newton_at(below[-1]["z"], s_target, P)
        res.append((ev, z, g, lam, ok))
    (ev, z, g, lam, ok), (ev2, z2, *_ , ok2) = res
    _, tan = hessian_and_tangent(z, s_target, P.x, P.y, P.act) if ok else (None, np.full(3, np.nan))
    valid = bool(ok and ok2 and ev["fold"] is None and ev2["fold"] is None and ev["end"] == "s_stop"
                 and float(np.abs(z - z2).max()) <= HALVING_REL)
    return {"s": float(s_target), "z": z.tolist(), "tangent": np.asarray(tan).tolist(), "grad": g, "lam_min": lam,
            "halving_sup": float(np.abs(z - z2).max()), "valid": valid}


def branch_point_near(fp, s, P):
    """A frozen follow point (at s_f = 0.8·s_switch) carried to the run's s by one tangent predictor + damped Newton."""
    if fp is None or not fp.get("valid"):
        return None
    z, g, lam, ok = newton_at(np.asarray(fp["z"]) + np.asarray(fp["tangent"]) * (s - fp["s"]), s, P)
    return z if ok else None


def freeze_one(seed):
    land = _land()
    s_pop, s0 = land["s_pop"], land["s0"]
    P = own_problem(seed)
    rows = {c: freeze_copy(P, np.array(land["z_pop_s0"]), c, s0, s_pop) for c in COPIES}
    for c in COPIES:                     # both copies' branch points at 0.8·s_switch of copy c (the follow check)
        r = rows[c]
        if not r.get("valid"):
            continue
        s_f = FOLLOW_FRAC * r["s_switch"]
        r["s_follow"] = s_f
        r["follow_points"] = {str(c2): (branch_point_at(P, rows[c2]["z_s0"], s0, s_f) if rows[c2]["newton_ok"] else None)
                              for c2 in COPIES}
        r["kappa_transforms"] = kappa_transforms(r["H_switch"], r["tangent_switch"], r["gradG"])
        r["kappa_mirror_recomputed"] = kappa_mirror_recomputed(seed, r["z_switch"], r["s_switch"])
    Wc = {c: rows[c]["W"] for c in COPIES}
    return {"seed": int(seed), "pilot": seed in PILOT_SEEDS, "copies": {str(c): rows[c] for c in COPIES},
            "W_branch_arm": arm_hold_steps(CONTROL, seed, Wc), "W_random_arm": arm_hold_steps(PRIMARY, seed, Wc)}


def freeze():
    _setup_process()
    f = OUT / "frozen_parts.jsonl"
    done = set() if not f.exists() else {json.loads(l)["seed"] for l in f.read_text().splitlines()}
    for seed in PILOT_SEEDS + SEEDS:
        if seed in done:
            continue
        _memory_guard()
        r = freeze_one(seed)
        with open(f, "a") as fh:
            fh.write(json.dumps(r, default=float) + "\n")
        c = r["copies"]
        print(json.dumps({"seed": seed, **{k: [c[k].get("valid"), c[k].get("s_switch"), c[k].get("kappa"), c[k].get("W")]
                                            for k in c}}), flush=True)
        _rss_guard()
    rows = sorted((json.loads(l) for l in f.read_text().splitlines()), key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == sorted(PILOT_SEEDS + SEEDS)
    (OUT / "frozen_seeds.json").write_text(json.dumps(rows, indent=1, default=float))
    nv = {c: sum(r["copies"][c]["valid"] for r in rows) for c in ("1", "-1")}
    print("frozen", len(rows), "valid copies", nv)


def _frozen():
    return {r["seed"]: r for r in json.loads((OUT / "frozen_seeds.json").read_text())}


# ------------------------------------------------------------------------------------------ training
def init_state(arm, seed, z_pop, s0):
    """(w₁, b₁, w₂, b₂) before the hold.  random: coordinates 0, 1, 3 of U(−1, 1)⁴ (float32, torch Generator seeded
    with the seed, cast to double), w₂ = +s₀.  branch: θ*_pop(s₀), w₁ mirrored for odd seeds."""
    import torch
    if arm == PRIMARY:
        gen = torch.Generator().manual_seed(int(seed))
        q = torch.empty(4, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen).to(torch.float64).numpy()
        return np.array([q[0], q[1], s0, q[3]], dtype=np.float64)
    c = assigned_copy(seed)
    return np.array([c * z_pop[0], z_pop[1], s0, z_pop[2]], dtype=np.float64)


def _loss(th, X, Y, u):
    from torch.nn import functional as F
    return F.binary_cross_entropy_with_logits(th[2] * u(th[0] * X + th[1]) + th[3], Y)


def hold(th0, W, X, Y, u):
    """Hold: w₂ fixed; (w₁, b₁, b₂) by full-batch GD at lr 0.3 for W steps.  G (dense, phase2b_ordering.state) is
    evaluated at the start state and after every step.  Returns (θ at release, record)."""
    import torch
    from .phase2b_ordering import state
    th = torch.tensor(th0, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    st0 = state(th.detach(), u, None, 1.0)
    npos, first, gmax = int(st0["placement_ok"]), (0 if st0["placement_ok"] else None), float(st0["gap"])
    for k in range(1, int(W) + 1):
        g = torch.autograd.grad(_loss(th, X, Y, u), th)[0]
        with torch.no_grad():
            th.sub_(HOLD_LR * g * mask)
        st = state(th.detach(), u, None, 1.0)
        gmax = max(gmax, float(st["gap"]))
        if st["placement_ok"]:
            npos += 1
            first = k if first is None else first
    rec = {"W_hold": int(W), "init_placed": bool(st0["placement_ok"]), "hold_n_G_pos": npos, "hold_first_G_pos": first,
           "hold_max_G": gmax, "hold_G_positive": npos > 0}
    return th.detach().numpy().copy(), rec


def release_train(th_rel, X, Y, u, rho, budget, stop_s=None):
    """Free SGD from the release state: fresh torch SGD, lr η, w₂'s gradient multiplied by ρ.  Returns the parameter
    path (row 0 = release).  stop_s: stop at the first step with |w₂| ≥ stop_s (the pilot; t_sw)."""
    import torch
    th = torch.tensor(th_rel, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([th], lr=ETA)
    Wp = np.empty((budget + 1, 4)); Wp[0] = th_rel
    for t in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        _loss(th, X, Y, u).backward()
        if rho != 1.0:
            th.grad[2].mul_(rho)
        opt.step()
        Wp[t] = th.detach().numpy()
        if stop_s is not None and abs(Wp[t, 2]) >= stop_s:
            return Wp[:t + 1]
    return Wp


def run_start(arm, seed, fr, land):
    """Init + hold + release classification (pre-release information only)."""
    import torch
    from .fold1d import make_data
    s0 = land["s0"]
    Wc = {int(c): v["W"] for c, v in fr["copies"].items()}
    W = arm_hold_steps(arm, seed, Wc)
    xt, yt = make_data(200, int(seed))
    X, Y = xt.to(torch.float64), yt.to(torch.float64)
    u = _act().torch_u
    th0 = init_state(arm, seed, land["z_pop_s0"], s0)
    th_rel, rec = hold(th0, W, X, Y, u)
    assert th_rel[2] == s0, "w₂ must be exactly +s0 at release"
    P = own_problem(seed)
    z_rel = th_rel[[0, 1, 3]]
    zn, gn, lamn, okn = newton_at(z_rel, s0, P)
    copies = {int(c): (v["z_s0"] if v.get("newton_ok") else None) for c, v in fr["copies"].items()}
    copy = classify_release(z_rel, zn, okn, copies)
    rec.update(arm=arm, seed=int(seed), init=th0.tolist(), release=th_rel.tolist(), newton_grad=gn, newton_lam=lamn,
               newton_ok=okn, copy_at_release=copy, on_branch=is_on_branch(arm, seed, copy),
               **{f"dist_newton_copy{c:+d}": (float(np.abs(zn - np.asarray(z)).max()) if z is not None else None)
                  for c, z in copies.items()})
    return th_rel, rec, (X, Y, u, P)


def follow_check(Wp, s, copy_release, cf, P):
    """Branch at the first step with s_t ≥ 0.8·s_switch,branch (of the release copy): Newton at that s from the run's
    (w₁, b₁, b₂), classified against both copies' frozen branch points (0.8·s_switch) carried to that s, tolerances as at
    release with the state condition FOLLOW_STATE_TOL.  Uses the state at that step only (no gap)."""
    t08 = first_at_fraction(s, cf["s_switch"])
    if t08 is None:
        return {"t_follow": None, "copy_at_follow": 0, "follows_branch": False}
    st = float(s[t08]); z = Wp[t08, [0, 1, 3]]
    pts = {int(c2): branch_point_near(fp, st, P) for c2, fp in cf["follow_points"].items()}
    zn, gn, ln, okn = newton_at(z, st, P)
    c08 = classify_release(z, zn, okn, pts, state_tol=FOLLOW_STATE_TOL)
    dist = {f"follow_state_dist_copy{c2:+d}": (float(np.abs(z - zc).max()) if zc is not None else None) for c2, zc in pts.items()}
    return {"t_follow": int(t08), "s_follow_run": st, "copy_at_follow": c08, "follow_newton_ok": okn,
            "follows_branch": follows_branch(copy_release, c08), **dist}


def predict_one(arm, seed, Wp, rec, fr, land, P, with_traj=True):
    """Predictions for one run from its release classification and its s path (no gap of the actual path)."""
    out = dict(rec)
    if not rec["on_branch"]:
        return {**out, "status": "no prediction: not on-branch at release"}
    cf = fr["copies"][str(rec["copy_at_release"])]
    if not cf.get("valid"):
        return {**out, "status": "no prediction: occupied copy has no validated frozen switch"}
    s = np.abs(Wp[:, 2])
    out["w2_min_after_release"] = float(Wp[:, 2].min())
    s_sw, kap, lam_sw = cf["s_switch"], cf["kappa"], cf["lam_min_switch"]
    t_sw = t_switch(s, s_sw)
    r_cf, chi, sdot = closed_form(s, t_sw, s_sw, kap, lam_sw)
    out.update(s_switch=s_sw, kappa=kap, lam_min_switch=lam_sw, t_sw=t_sw, sdot=sdot, chi_tsw=chi, r_cf=r_cf,
               kc_tsw=r_cf, eta_lam=ETA * lam_sw, lag_steps_pred=kap / (ETA * lam_sw))
    s0, s_pop = land["s0"], land["s_pop"]
    B = GBranch(s0, np.array(cf["z_s0"]), P, 0.95 * s0, S_HI_FRAC * s_pop, GRID_H_FRAC * s_pop)
    out.update(branch_lo=B.lo, branch_hi=B.hi, branch_anchor_ok=B.anchor_ok)
    cp = chi_path(s, t_sw, B.lam_min) if t_sw is not None else np.array([])
    out["chi_path_max_all"] = float(cp.max()) if len(cp) and np.all(np.isfinite(cp)) else float("nan")   # descriptive
    out["chi_window_max"] = chi_window_max(s, t_sw, cp, s_sw)                                          # V7
    out.update(follow_check(Wp, s, rec["copy_at_release"], cf, P))
    out["pred_signed_s_cross_cf"] = s_sw * (1 + r_cf) if np.isfinite(r_cf) else float("nan")
    if not with_traj:
        return {**out, "status": "ok (pilot: no trajectory prediction)"}
    from .act_general import gap_mid
    z_rel = Wp[0, [0, 1, 3]]
    inside = (s >= B.lo) & (s <= B.hi)
    T_end = len(s) if inside.all() else int(np.argmin(inside))
    s_seg = s[:T_end]
    th_seg = np.array([B.theta(v) for v in s_seg])
    Hs = [B.hess(v) for v in s_seg]
    d0 = z_rel - th_seg[0]
    dth = np.diff(th_seg, axis=0)
    PP = np.ones((T_end, 3))
    gapf = lambda i, dd: gap_mid(th_seg[i][0] + dd[0], th_seg[i][1] + dd[1], P.act)
    t_hit, path = LR.simulate(s_seg, lambda t: Hs[t], PP, dth, gapf, d0, m0=None, lr=ETA)
    idx = sorted(set(range(0, len(path), 10)) | {len(path) - 1})
    rad = max(LR.step_map_radius(Hs[i], np.ones(3), ETA, False) for i in idx)
    mx = float(np.nanmax(np.abs(np.array(path))))
    out.update(traj_rho_max=rad, traj_max_abs_delta=mx, traj_T=int(T_end), delta0_norm=float(np.abs(d0).max()))
    if t_hit is None or rad > 1.0 or not np.isfinite(mx) or mx > LR.DIVERGED:
        out.update(r_traj=float("nan"), s_traj=float("nan"), status_traj="no prediction (no hit / unstable / diverged)")
    else:
        out.update(s_traj=float(s_seg[t_hit]), r_traj=float(s_seg[t_hit]) / s_sw - 1, t_traj=int(t_hit),
                   status_traj="ok")
    out["status"] = "ok"
    return out


# ------------------------------------------------------------------------------------------ pilot
def pilot():
    """The pilot rule on the pilot seeds, both arms, up to t_sw only (no gap after release, no crossing computed).
    κχ values are pooled over both arms' on-branch pilot runs; each arm's pilot median χ at t_sw is kept (V6)."""
    _setup_process()
    _memory_guard()
    land, fz = _land(), _frozen()
    runs = {}

    def run_pilot(rho):
        vals, rows = [], []
        for arm in ARMS:
            for seed in PILOT_SEEDS:
                fr = fz[seed]
                th_rel, rec, (X, Y, u, P) = run_start(arm, seed, fr, land)
                row = dict(rec)
                if rec["on_branch"] and fr["copies"][str(rec["copy_at_release"])].get("valid"):
                    s_sw = fr["copies"][str(rec["copy_at_release"])]["s_switch"]
                    Wp = release_train(th_rel, X, Y, u, rho, BUDGET, stop_s=s_sw)
                    row = predict_one(arm, seed, Wp, rec, fr, land, P, with_traj=False)
                    row["n_steps_run"] = int(len(Wp) - 1)
                    if np.isfinite(row["kc_tsw"]):
                        vals.append(row["kc_tsw"])
                row["rho"] = rho
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in ("arm", "seed", "copy_at_release", "on_branch", "hold_n_G_pos", "t_sw",
                                                          "chi_tsw", "kc_tsw", "chi_window_max", "copy_at_follow",
                                                          "follows_branch")}, default=float), flush=True)
                _rss_guard()
        runs[rho] = rows
        return vals
    rule = pilot_rule(run_pilot)
    final = runs[rule["rho"]] if rule["status"] == "ok" else []
    med = {arm: float(np.median([r["chi_tsw"] for r in final if r["arm"] == arm and np.isfinite(r.get("chi_tsw", np.nan))]))
           if any(r["arm"] == arm and np.isfinite(r.get("chi_tsw", np.nan)) for r in final) else float("nan")
           for arm in ARMS}
    out = {"rule": rule, "rho": rule["rho"], "pilot_median_chi_tsw": med,
           "runs": {str(k): v for k, v in runs.items()}, "pilot_seeds": list(PILOT_SEEDS)}
    (OUT / "pilot.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1, default=float))
    if rule["status"] != "ok":
        raise SystemExit("STOP: pilot rule " + rule.get("reason", ""))
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
    return PATHS / f"{arm}_{seed}.npz"


def train():
    """Both arms, 80 seeds each, to the full budget.  No gap or placement after release is evaluated.  Each path is
    saved (untracked) and hashed; each row is written as the run completes."""
    _setup_process()
    _assert_registration()
    land, fz = _land(), _frozen()
    rho = json.loads((OUT / "pilot.json").read_text())["rho"]
    PATHS.mkdir(parents=True, exist_ok=True)
    f = OUT / "predictions_parts.jsonl"
    done = set() if not f.exists() else {(r["arm"], r["seed"]) for r in map(json.loads, f.read_text().splitlines())}
    for arm in ARMS:
        for seed in SEEDS:
            if (arm, seed) in done:
                continue
            _memory_guard()
            fr = fz[seed]
            th_rel, rec, (X, Y, u, P) = run_start(arm, seed, fr, land)
            Wp = release_train(th_rel, X, Y, u, rho, BUDGET)
            pf = _path_file(arm, seed)
            np.savez(pf, W=Wp)
            h = _sha(pf)
            try:
                r = predict_one(arm, seed, Wp, rec, fr, land, P)
            except Exception as e:                                         # counted, not replaced
                r = {**rec, "status": f"error: {type(e).__name__}: {e}"}
            r.update(rho=rho, path_sha256=h)
            with open(f, "a") as fh:
                fh.write(json.dumps(LR._jsonable({k: (json.dumps(v) if isinstance(v, list) else v)
                                                  for k, v in r.items()})) + "\n")
            print(json.dumps({k: r.get(k) for k in ("arm", "seed", "copy_at_release", "on_branch", "hold_n_G_pos", "status",
                                                    "t_sw", "r_traj", "r_cf")}, default=float), flush=True)
            _rss_guard()


FORBIDDEN = {"cross_step", "s_obs", "r_obs", "step_obs", "placed", "crossed"}


def finalize():
    d = pd.DataFrame([json.loads(l) for l in (OUT / "predictions_parts.jsonl").read_text().splitlines()])
    d = d.sort_values(["arm", "seed"])
    for arm in ARMS:
        assert list(d[d.arm == arm].seed) == list(SEEDS), f"every registered seed exactly once ({arm})"
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


def observe():
    """After the predictions commit: assert hashes; per arm, the gate (from the committed predictions) first; then
    every-step detection on each saved path of a gated arm; scores."""
    _setup_process()
    import torch
    from .phase2b_ordering import state
    _assert_registration()
    _assert_committed(OUT / "predictions.csv", OUT / "predictions.sha256")
    pr = pd.read_csv(OUT / "predictions.csv")
    land = _land()
    pil = json.loads((OUT / "pilot.json").read_text())
    u = _act().torch_u
    parts = OUT / "observed_parts.jsonl"
    done = {} if not parts.exists() else {(r["arm"], r["seed"]): r for r in map(json.loads, parts.read_text().splitlines())}
    gates = {arm: gate(arm, pr[pr.arm == arm].on_branch.astype(bool), pr[pr.arm == arm].hold_G_positive.astype(bool))
             for arm in ARMS}
    for r in pr.itertuples():
        if not gates[r.arm]["pass"] or (r.arm, r.seed) in done:
            continue
        pf = _path_file(r.arm, r.seed)
        assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
        Wp = np.load(pf)["W"]
        t_c = None
        for t in range(1, len(Wp)):
            if state(torch.tensor(Wp[t], dtype=torch.float64), u, None, 1.0)["placement_ok"]:
                t_c = t
                break
        row = {"arm": r.arm, "seed": int(r.seed), "crossed": t_c is not None, "step_obs": t_c,
               "s_obs": abs(float(Wp[t_c, 2])) if t_c is not None else float("nan")}
        with open(parts, "a") as fh:
            fh.write(json.dumps(LR._jsonable(row)) + "\n")
        _rss_guard()
    res = {"predictions_sha256": _sha(OUT / "predictions.csv"),
           "registration_sha256_file_sha256": _sha(OUT / "registration.sha256"), "rho": pil["rho"], "arms": {}}
    obs = pd.DataFrame([json.loads(l) for l in parts.read_text().splitlines()]) if parts.exists() else \
        pd.DataFrame(columns=["arm", "seed", "crossed", "step_obs", "s_obs"])
    for arm in ARMS:
        g = pr[pr.arm == arm].copy()
        if not gates[arm]["pass"]:
            nan = np.full(len(g), np.nan)
            res["arms"][arm] = score_arm(arm, g.on_branch.astype(bool), g.hold_G_positive.astype(bool),
                                         np.zeros(len(g), bool), *([nan] * 12), pil["pilot_median_chi_tsw"][arm],
                                         land["s_glob"])
            continue
        d = g.merge(obs[obs.arm == arm], on=["arm", "seed"], how="left")
        assert len(d) == len(SEEDS)
        col = lambda c: d[c].to_numpy(float) if c in d else np.full(len(d), np.nan)
        fol = d["follows_branch"].fillna(False).astype(bool) if "follows_branch" in d else np.zeros(len(d), bool)
        args = (arm, d.on_branch.astype(bool), d.hold_G_positive.astype(bool), d.crossed.fillna(False).astype(bool),
                col("step_obs"), col("s_obs"), col("s_switch"), col("s_traj"), col("r_traj"), col("r_cf"),
                col("kappa"), col("t_sw"), col("eta_lam"), col("lag_steps_pred"), col("chi_tsw"),
                col("chi_window_max"), pil["pilot_median_chi_tsw"][arm], land["s_glob"])
        sc = score_arm(*args, follows=fol)
        if arm == PRIMARY:           # registered DESCRIPTIVE sensitivity analysis: runs with G > 0 in the hold excluded
            sens = score_arm(*args, follows=fol, exclude=d.hold_G_positive.astype(bool))
            sens.pop("scored_index")
            sc["sensitivity_excluding_hold_G_positive_DESCRIPTIVE"] = sens
        d["scored"] = False
        d.loc[d.index[sc["scored_index"]], "scored"] = True
        d["r_obs"] = d.s_obs / d.s_switch - 1
        d.to_csv(OUT / f"observed_runs_{arm}.csv", index=False)
        S = d[d.scored]
        sc["descriptive"] = {
            "n_copy_plus": int((d["copy_at_release"] == 1).sum()), "n_copy_minus": int((d["copy_at_release"] == -1).sum()),
            "n_neither": int((d["copy_at_release"] == 0).sum()), "n_init_placed": int(d.init_placed.astype(bool).sum()),
            "median_r_obs_scored": float(S.r_obs.median()), "median_r_traj_scored": float(S.r_traj.median()),
            "median_r_cf_scored": float(S.r_cf.median()),
            "median_steps_tsw_to_crossing_scored": float((S.step_obs - S.t_sw).median()),
            "median_s_obs_over_s_switch_scored": float((S.s_obs / S.s_switch).median()),
            "n_kappa_negative_scored": int((S.kappa < 0).sum()),
            "n_on_branch_not_following": int((d.on_branch.astype(bool) & ~fol).sum()),
            "n_follow_check_at_or_after_crossing": int((d.crossed.fillna(False).astype(bool)
                                                        & (d.t_follow >= d.step_obs)).sum()),
            "n_hold_G_positive": int(d.hold_G_positive.astype(bool).sum()),
            "n_hold_G_positive_scored": int((S.hold_G_positive.astype(bool)).sum()),
            "median_chi_path_max_all_scored_descriptive": float(S.chi_path_max_all.median()),
            "n_signed_r_traj_negative_scored": int((S.r_traj < 0).sum()),
            "n_r_obs_negative_scored": int((S.r_obs < 0).sum())}
        res["arms"][arm] = sc
    res["headline"] = {"arm": PRIMARY, "outcome": res["arms"][PRIMARY]["outcome"]}
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    {"landscape": lambda: (_setup_process(), landscape()), "freeze": freeze, "pilot": pilot, "hashes": hashes,
     "train": train, "finalize": finalize, "observe": observe}[sys.argv[1]]()
