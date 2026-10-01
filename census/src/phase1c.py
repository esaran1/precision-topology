"""Phase 1C: the registered causal replication on fresh seeds.

Design: results/designs/phase1c_causal_design.md (approved by the author 2026-10-01 with changes 1-4).  Registration:
results/phase1c_registration.md.  A THIN LAYER over the replicated pipelines (Track A width 1 at a = 1.58 via the Phase 1A
driver, GELU-T random start, W2-A T and T′, each called unchanged) and the frozen causal forecaster
(src/causal_forecast.py, 25aac9e; the Phase 1A driver's make_inputs / context, unchanged).

Arms: W1 (width-1 SGD, a = 1.58; required), G (GELU-T random start), T (W2-A T), Tp (W2-A T′; sign test, own verdict).
Gate to Phase 2: W1 PASS and (G PASS or T PASS).

ORDER (no step reads anything a later step produces):
  scan / freeze / manifest   before the registration commit (no training of a registered seed)
  run ARM                    per seed: start, hold, release (G, T, T′), training to the full budget WITHOUT evaluating
                             any gap or placement after release; the 1C forecasts at the arm's f and at the other f
                             (descriptive), each from GUARDED views of rows < t_c only, each recomputed with every row
                             ≥ t_c NaN (must be identical); the output path prefix and the forecaster's hidden rows saved
                             (untracked, SHA-256 in the row).  -> runs_ARM.jsonl
  finalize                   forecasts.sha256 over the four runs files (to be COMMITTED before observe)
  observe ARM                asserts the registration and the committed forecasts; retrains each scored-copy run
                             deterministically (the saved prefix asserted bit for bit), every-step detection over the
                             full budget, the actual switch, the follow check and the registered validity quantities
                             (registered predict_one, with_traj=False).  -> observed_ARM.jsonl
  score                      scores.json (score_arm_1c; the registered gate and V1-V7 functions, unchanged)

Machine rules: one process, nice 15, one thread; a memory gate (free ≥ 25%, swap free ≥ 500 MB) before every job and
between seeds, logged to results/phase1c/memory_gate.log; stop above 3 GB RSS.  No global torch / numpy RNG state.

    python -m src.phase1c scan | landscape_check | freeze ARM | manifest
    python -m src.phase1c run ARM | finalize | observe ARM | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import causal_forecast as C

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase1c"
PATHS = OUT / "paths"
REGISTRATION_MD = RESULTS / "phase1c_registration.md"
DESIGN_MD = RESULTS / "designs" / "phase1c_causal_design.md"

# ------------------------------------------------------------------------------------------ registered constants
ARMS = ("W1", "G", "T", "Tp")
ARM_LABEL = {"W1": "W1 (width-1 SGD lr 0.3, a = 1.58)", "G": "G (GELU-T random start)", "T": "T (W2-A T)",
             "Tp": "T′ (W2-A T′; sign test, own verdict)"}
SEEDS = {"W1": tuple(range(9_350_000, 9_350_100)),          # 100
         "G": tuple(range(9_360_000, 9_360_080)),           # 80
         "T": tuple(range(9_370_000, 9_370_200)),           # 200
         "Tp": tuple(range(9_380_000, 9_380_120))}          # 120
SETTING = {"W1": "w1_sgd_run", "G": "gelu_random", "T": "w2a_T", "Tp": "w2a_Tp"}     # the Phase 1A driver's names
KIND = {"W1": "w1", "G": "gelu", "T": "w2a", "Tp": "w2a"}
F = {"W1": 0.90, "G": 0.95, "T": 0.95, "Tp": 0.95}             # changes 1 and 2
F_OTHER = {"W1": 0.95, "G": 0.90, "T": 0.90, "Tp": 0.90}       # change 3: descriptive only
TAU_CROSS = {"W1": 10, "G": 10, "T": 15, "Tp": 30}             # steps
TAU_LAG = {"W1": 15, "G": 10, "T": 5, "Tp": 5}                 # steps
BAND = {"W1": 0.20, "G": 0.15, "T": 0.10, "Tp": 0.10}          # C3: median r_obs/r_fc in [1 - b, 1 + b]
WITHIN_MIN_FRAC = 0.80                                         # C1, C2
S_MIN_FRAC, S_MIN_ABS_LAG = 0.80, 6                            # S (T′ only)
CUTOFF_BEFORE_MIN_FRAC = 0.90                                  # validity: t_c < t_obs in ≥ 90% of crossing runs
W1_MIN = 60                                                    # W1: ≥ 60 crossings and ≥ 60 scored runs (Track A)
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 9_350_000, (2.5, 97.5)   # C4
FAMILY, WINDOW_FRAC, MIN_WINDOW, HORIZON = "quad", 0.05, 50, 1.6
SAVE_MARGIN = 500
# every committed artifact, by name (the ledger's provenance check reads these): pilot_check (`pilot_check`), seed scan
# (`scan`), frozen per-seed inputs (`freeze`), manifest (`manifest`), runs + forecasts (`run`), forecast hashes
# (`finalize`), observations (`observe`), verdicts (`score`)
ARTIFACTS = ("phase1c/pilot_check.json", "phase1c/seed_scan.json", "phase1c/frozen_W1.jsonl", "phase1c/frozen_G.jsonl",
             "phase1c/frozen_T.jsonl", "phase1c/frozen_Tp.jsonl", "phase1c/registration.sha256",
             "phase1c/runs_W1.jsonl", "phase1c/runs_G.jsonl", "phase1c/runs_T.jsonl", "phase1c/runs_Tp.jsonl",
             "phase1c/forecasts.sha256", "phase1c/observed_W1.jsonl", "phase1c/observed_G.jsonl",
             "phase1c/observed_T.jsonl", "phase1c/observed_Tp.jsonl", "phase1c/scores.json")
RSS_LIMIT = 3 * 1024 ** 3
CRITERIA = {"W1": ("C1", "C2", "C3", "C4"), "G": ("C1", "C2", "C3", "C4"), "T": ("C1", "C2", "C3", "C4"),
            "Tp": ("C1", "C2", "C3", "C4", "S")}


def config(arm, f=None):
    """The frozen forecaster configuration (Phase 1A primary quad / 0.05, at least 50 steps, horizon 1.6·s_ref)."""
    return C.ForecastConfig(f=F[arm] if f is None else float(f), family=FAMILY, window_frac=WINDOW_FRAC,
                            min_window=MIN_WINDOW, horizon_factor=HORIZON)


# ------------------------------------------------------------------------------------------ pure scoring rules
def _arr(v, dt=float):
    return np.array([np.nan if (dt is float and x is None) else x for x in v], dtype=dt) if isinstance(v, list) \
        else np.asarray(v, dt)


def cutoff_before(t_c, t_obs):
    """Validity rule, per run: a cutoff exists and is STRICTLY before the crossing (t_c < t_obs)."""
    t_c, t_obs = _arr(t_c), _arr(t_obs)
    with np.errstate(invalid="ignore"):
        return np.isfinite(t_c) & np.isfinite(t_obs) & (t_c < t_obs)


def cutoff_validity(before, base):
    """t_c strictly before the crossing in ≥ 90% of the crossing runs `base` (runs with no cutoff count as not before).
    With no crossing run it fails (nothing to score)."""
    before, base = _arr(before, bool), _arr(base, bool)
    n = int(base.sum())
    k = int((before & base).sum())
    frac = k / n if n else float("nan")
    return {"n_crossing_runs": n, "n_cutoff_before_crossing": k, "n_cutoff_not_before_unscored": n - k,
            "frac": frac, "min_frac": CUTOFF_BEFORE_MIN_FRAC, "ok": bool(n and frac >= CUTOFF_BEFORE_MIN_FRAC)}


def is_forecast(t_fc, t_sw_fc):
    """A run HAS a forecast iff both the forecast crossing and the forecast (lag-free) switch exist; otherwise (cutoff
    but no forecast, or no forecast switch) it is a MISS."""
    return np.isfinite(_arr(t_fc)) & np.isfinite(_arr(t_sw_fc))


def criterion_within(err, tau, has_fc):
    """C1 / C2 over the scored runs: |err| ≤ τ in ≥ 80% of them.  A miss (no forecast) or an undefined error counts as
    NOT within.  No scored run: UNRESOLVED."""
    err, has_fc = _arr(err), _arr(has_fc, bool)
    n = len(err)
    with np.errstate(invalid="ignore"):
        ok = has_fc & np.isfinite(err) & (np.abs(err) <= tau)
    k = int(ok.sum())
    frac = k / n if n else float("nan")
    return {"n": n, "n_within": k, "n_miss": int((~has_fc).sum()), "n_err_undefined": int((has_fc & ~np.isfinite(err)).sum()),
            "frac_within": frac, "tau": tau, "min_frac": WITHIN_MIN_FRAC,
            "verdict": "UNRESOLVED" if n == 0 else ("PASS" if frac >= WITHIN_MIN_FRAC else "FAIL")}


def criterion_ratio(r_obs, r_fc, has_fc, b):
    """C3: the median of r_obs/r_fc over the scored runs with a forecast and a finite ratio lies in [1 − b, 1 + b]
    (closed; float64, no rounding).  None: UNRESOLVED."""
    r_obs, r_fc, has_fc = _arr(r_obs), _arr(r_fc), _arr(has_fc, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(has_fc & (r_fc != 0), r_obs / r_fc, np.nan)
    ok = np.isfinite(q)
    n = int(ok.sum())
    m = float(np.median(q[ok])) if n else float("nan")
    return {"n": n, "n_excluded": int(len(q) - n), "median_ratio": m, "band": [1 - b, 1 + b],
            "verdict": "UNRESOLVED" if n == 0 else ("PASS" if (1 - b) <= m <= (1 + b) else "FAIL")}


def bootstrap_mean_ci(D, n_boot=BOOT_N, seed=BOOT_SEED):
    """(mean, lower, upper): 95% percentile bootstrap interval of the mean of D over run resamples (numpy
    default_rng(seed), a LOCAL generator; n_boot resamples of size len(D) with replacement; the registered GELU-T / W2-A
    procedure)."""
    D = np.asarray(D, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    means = D[idx].mean(axis=1)
    lo, hi = np.percentile(means, BOOT_PCT)
    return float(D.mean()), float(lo), float(hi)


def criterion_c4(t_fc, t_sw_fc, t_obs, has_fc):
    """C4 (change 4): D = |t_fc − t_obs| − |t_nolag − t_obs|, t_nolag = t_sw,fc (the no-lag forecast: the crossing at
    the forecast switch), paired per run, over the scored runs with a forecast.  PASS iff the upper end of the 95%
    percentile bootstrap interval of mean D is < 0; an interval entirely above 0, or containing 0, is FAIL.  Fewer than
    2 runs (or a non-finite interval): UNRESOLVED.  The run fraction with D < 0 is DESCRIPTIVE (the draft's C4)."""
    t_fc, t_sw_fc, t_obs, has_fc = _arr(t_fc), _arr(t_sw_fc), _arr(t_obs), _arr(has_fc, bool)
    ok = has_fc & np.isfinite(t_fc) & np.isfinite(t_sw_fc) & np.isfinite(t_obs)
    D = np.abs(t_fc[ok] - t_obs[ok]) - np.abs(t_sw_fc[ok] - t_obs[ok])
    n = int(ok.sum())
    mean, lo, hi = bootstrap_mean_ci(D) if n >= 2 else (float("nan"),) * 3
    computable = bool(n >= 2 and np.isfinite(hi))
    return {"n": n, "mean_D": mean, "ci95": [lo, hi], "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
            "reading": "PASS iff the upper end < 0 (interval entirely above 0 or containing 0: FAIL)",
            "verdict": "UNRESOLVED" if not computable else ("PASS" if hi < 0 else "FAIL"),
            "DESCRIPTIVE_n_forecast_closer": int((D < 0).sum()), "DESCRIPTIVE_n_ties": int((D == 0).sum()),
            "DESCRIPTIVE_frac_forecast_closer": float((D < 0).mean()) if n else float("nan")}


def criterion_sign(lag_fc, lag_obs, has_fc):
    """S (T′): lag_fc < 0 and lag_obs < 0 in ≥ 80% of the eligible scored runs.  Eligible: the scored runs with a
    forecast and |lag_fc| ≥ 6 steps, AND every scored run without a forecast (a MISS), which counts as not both negative
    (author's decision 2026-10-01; as D9 does for an undefined observed lag).  No eligible run: UNRESOLVED."""
    lag_fc, lag_obs, has_fc = _arr(lag_fc), _arr(lag_obs), _arr(has_fc, bool)
    with np.errstate(invalid="ignore"):
        el_fc = has_fc & np.isfinite(lag_fc) & (np.abs(lag_fc) >= S_MIN_ABS_LAG)
        miss = ~has_fc
        el = el_fc | miss
        both = el_fc & (lag_fc < 0) & np.isfinite(lag_obs) & (lag_obs < 0)
    n, k = int(el.sum()), int(both.sum())
    frac = k / n if n else float("nan")
    return {"n_eligible": n, "n_eligible_miss": int(miss.sum()), "n_both_negative": k, "frac": frac,
            "min_frac": S_MIN_FRAC, "min_abs_lag": S_MIN_ABS_LAG,
            "verdict": "UNRESOLVED" if n == 0 else ("PASS" if frac >= S_MIN_FRAC else "FAIL")}


def horizons(t_c, t_obs, t_sw, lag_obs):
    """DESCRIPTIVE (change 3): forecast horizon t_obs − t_c and t_sw − t_c, in steps and in lags (÷ max(|lag_obs|, 1))."""
    t_c, t_obs, t_sw, lag_obs = _arr(t_c), _arr(t_obs), _arr(t_sw), _arr(lag_obs)
    L = np.maximum(np.abs(lag_obs), 1.0)

    def st(v):
        v = v[np.isfinite(v)]
        return {"n": int(len(v)), "min": float(v.min()) if len(v) else None,
                "median": float(np.median(v)) if len(v) else None, "max": float(v.max()) if len(v) else None}
    return {"cross_steps": st(t_obs - t_c), "switch_steps": st(t_sw - t_c), "cross_lags": st((t_obs - t_c) / L),
            "switch_lags": st((t_sw - t_c) / L)}


def outcome(verdicts):
    """PASS iff every criterion passes; UNRESOLVED if any is UNRESOLVED; otherwise FAIL naming each failing one
    (the registered GELU-T / W2-A outcome rule)."""
    vs = list(verdicts.values())
    if "UNRESOLVED" in vs:
        return "UNRESOLVED"
    if all(v == "PASS" for v in vs):
        return "PASS"
    return "FAIL " + "+".join(k for k, v in verdicts.items() if v == "FAIL")


def criteria(arm, t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs, has_fc):
    """C1-C4 (+ S for T′) over the given (scored) runs."""
    t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs = (_arr(v) for v in (t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs))
    has_fc = _arr(has_fc, bool)
    lag_fc, lag_obs = t_fc - t_sw_fc, t_obs - t_sw
    out = {"C1": criterion_within(t_fc - t_obs, TAU_CROSS[arm], has_fc),
           "C2": criterion_within(lag_fc - lag_obs, TAU_LAG[arm], has_fc),
           "C3": criterion_ratio(r_obs, r_fc, has_fc, BAND[arm]),
           "C4": criterion_c4(t_fc, t_sw_fc, t_obs, has_fc)}
    if "S" in CRITERIA[arm]:
        out["S"] = criterion_sign(lag_fc, lag_obs, has_fc)
    return out


def same_forecast(a, b):
    """The per-run NaN recomputation: every output field identical (NaN equal to NaN; the guards' bookkeeping
    `max_index_read` and timings ignored)."""
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


# ------------------------------------------------------------------------------------------ verdict per arm
def _registered_score(arm, R, r_traj, s_traj, t_traj, r_cf, pilot_chi, s_pop):
    """The REGISTERED scoring function of the replicated test (gelu_transfer.score_arm / width2_asym.score_arm),
    called unchanged."""
    if arm == "G":
        from . import gelu_transfer as G
        return G.score_arm(G.PRIMARY, R["on_branch"], R["hold_positive"], R["crossed"], R["t_obs"], R["s_obs"],
                           R["s_sw"], s_traj, r_traj, r_cf, R["kappa"], R["t_sw"], R["eta_lam"], R["lag_steps"],
                           R["chi_tsw"], R["chi_win_max"], pilot_chi, float(s_pop), follows=R["follows"])
    from . import width2_asym as W
    return W.score_arm(arm, R["on_branch"], R["hold_positive"], R["crossed"], R["t_obs"], R["s_obs"], R["s_sw"],
                       s_traj, t_traj, r_traj, r_cf, R["kappa"], R["t_sw"], R["eta_lam"], R["lag_steps"], R["chi_tsw"],
                       R["chi_win_max"], pilot_chi, s_pop, follows=R["follows"])


def score_arm_1c(arm, R, pilot_chi=None, s_pop=None):
    """The registered 1C verdict for one arm.  R: arrays over ALL the arm's seeds (NaN / False where undefined):
    crossed, t_obs, s_obs (observed); t_sw, s_sw (the actual lag-free switch: W1 the first step with s ≥ s*_run of the
    cutoff's rule-point branch; G the copy's frozen switch; T, T′ the registered own-path switch on the actual v path);
    t_c, t_fc, t_sw_fc, s_fc, r_fc (the 1C forecast at the arm's f); nan_identical; hold_positive; W1: frozen_ok,
    r_cf (descriptive); G, T, T′: on_branch, follows and the registered validity inputs r_cf, kappa, eta_lam,
    lag_steps, chi_tsw, chi_win_max.  Optional *_other keys: the forecast at the other f (descriptive).

    1. Gate (G, T, T′; the registered gate, unchanged).  Fails: every criterion UNRESOLVED, nothing scored.
    2. Crossing runs (the base): W1, a frozen s*_frozen and a crossing within the budget; G, T, T′, the registered
       scored-run conditions without the registered predictions (on a scored copy at release, follows its branch,
       crossed, finite r_cf and r_obs).
    3. Scored runs: the base with the cutoff strictly before the crossing (W1 also: the actual switch t_sw defined).
       A scored run without a forecast is a MISS.
    4. Validity: W1, ≥ 60 crossings and ≥ 60 scored runs (Track A's counts); G, T, T′, the registered V1-V7 over the
       scored runs (the registered function, with a finite placeholder for the registered predictions exactly on the
       scored runs, so its scored set is the 1C scored set, misses included); every arm, the cutoff before the crossing
       in ≥ 90% of the base and the per-run NaN recomputation identical in every run with a cutoff.  Fails: every
       criterion UNRESOLVED.
    5. C1-C4 (+ S) over the scored runs; outcome() of the verdicts."""
    R = {k: (_arr(v, bool) if k in ("crossed", "on_branch", "follows", "hold_positive", "frozen_ok", "nan_identical")
             else _arr(v)) for k, v in R.items()}
    n = len(R["t_obs"])
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = R["s_obs"] / R["s_sw"] - 1
    before = cutoff_before(R["t_c"], R["t_obs"])
    has_fc = is_forecast(R["t_fc"], R["t_sw_fc"])
    out = {"arm": arm, "label": ARM_LABEL[arm], "n_runs": n, "f": F[arm], "tau_cross": TAU_CROSS[arm],
           "tau_lag": TAU_LAG[arm], "band": BAND[arm]}
    names = CRITERIA[arm]
    if arm == "W1":
        base = R["frozen_ok"] & R["crossed"]
        scored = base & before & np.isfinite(R["t_sw"]) & np.isfinite(r_obs)
        out["gate"] = {"applies": False, "pass": True, "note": "no hold (Track A)"}
        V = {"W1_min_60_crossings": int(base.sum()) >= W1_MIN, "W1_min_60_scored": int(scored.sum()) >= W1_MIN}
        reg = None
    else:
        ph = np.where(before, 1.0, np.nan)
        reg = _registered_score(arm, R, ph, np.where(before, R["s_obs"], np.nan), R["t_obs"], R["r_cf"], pilot_chi,
                                s_pop)
        out["gate"] = reg["gate"]
        if not reg["gate"]["pass"]:
            out.update({k: {"verdict": "UNRESOLVED"} for k in names})
            out.update(valid=False, outcome="UNRESOLVED (gate)", n_scored=0, scored_index=[])
            return out
        base = R["on_branch"] & R["follows"] & R["crossed"] & np.isfinite(R["r_cf"]) & np.isfinite(r_obs)
        scored = np.zeros(n, bool)
        scored[reg["scored_index"]] = True
        assert np.array_equal(scored, base & before), "registered scored set != 1C scored set"
        V = dict(reg["validity"])
        out["registered_validity_inputs"] = {k: reg.get(k) for k in (
            "n_on_branch", "n_crossed", "n_on_branch_not_following", "n_on_branch_hold_G_positive",
            "n_kappa_pos_crossing", "frac_tsw_before_crossing_kappa_pos", "regime_frac", "median_predicted_lag_steps",
            "q90_kappa_chi_tsw", "median_chi_tsw", "pilot_median_chi_tsw", "chi_rel_to_pilot", "q90_max_chi_window")}
    cv = cutoff_validity(before, base)
    withc = np.isfinite(R["t_c"])
    nan_ok = bool(np.all(R["nan_identical"][withc])) if withc.any() else True
    V["cutoff_before_crossing_90pct"] = cv["ok"]
    V["nan_recomputation_identical"] = nan_ok
    valid = all(bool(v) for v in V.values())
    out.update(validity={k: bool(v) for k, v in V.items()}, valid=valid, cutoff_validity=cv,
               n_base_crossing_runs=int(base.sum()), n_scored=int(scored.sum()),
               n_scored_miss=int((scored & ~has_fc).sum()), n_scored_with_forecast=int((scored & has_fc).sum()),
               n_crossed_cutoff_not_before=int((base & ~before).sum()),
               n_nan_recompute_checked=int(withc.sum()),
               n_nan_recompute_differs=int((withc & ~R["nan_identical"]).sum()),
               scored_index=np.nonzero(scored)[0].tolist())
    S = lambda k: R[k][scored]                                              # noqa: E731
    crit = criteria(arm, S("t_fc"), S("t_sw_fc"), S("r_fc"), S("t_obs"), S("t_sw"), r_obs[scored], has_fc[scored])
    for k in names:
        c = crit[k]
        if not valid:
            c = {**c, "verdict_if_valid": c["verdict"], "verdict": "UNRESOLVED"}
        out[k] = c
    out["outcome"] = "UNRESOLVED (validity)" if not valid else outcome({k: out[k]["verdict"] for k in names})
    # ---- DESCRIPTIVE (never a verdict)
    d = {"horizon": horizons(S("t_c"), S("t_obs"), S("t_sw"), S("t_obs") - S("t_sw"))}
    if "t_fc_other" in R:
        bo = cutoff_before(R["t_c_other"], R["t_obs"])
        so = base & bo & (np.isfinite(R["t_sw"]) if arm == "W1" else True)
        ho = is_forecast(R["t_fc_other"], R["t_sw_fc_other"])
        d["other_f"] = {"f": F_OTHER[arm], "n_cutoff_before_crossing": int(so.sum()),
                        "criteria_stats": criteria(arm, R["t_fc_other"][so], R["t_sw_fc_other"][so], R["r_fc_other"][so],
                                                   R["t_obs"][so], R["t_sw"][so], r_obs[so], ho[so]),
                        "horizon": horizons(R["t_c_other"][so], R["t_obs"][so], R["t_sw"][so],
                                            R["t_obs"][so] - R["t_sw"][so])}
    fs = scored & has_fc
    with np.errstate(invalid="ignore"):
        d["L1_L5_on_r_fc"] = _descriptive_L(arm, R, fs, r_obs, before, pilot_chi, s_pop)
    if arm in ("G", "T"):
        keep = scored & ~R["hold_positive"]
        d["sensitivity_excluding_hold_G_positive"] = {
            "n_scored": int(keep.sum()),
            **{k: v["verdict"] for k, v in criteria(arm, R["t_fc"][keep], R["t_sw_fc"][keep], R["r_fc"][keep],
                                                    R["t_obs"][keep], R["t_sw"][keep], r_obs[keep],
                                                    has_fc[keep]).items()}}
    out["DESCRIPTIVE"] = d
    return out


def _descriptive_L(arm, R, fs, r_obs, before, pilot_chi, s_pop):
    """DESCRIPTIVE: the replicated test's L criteria with the 1C forecast in place of its registered prediction
    (r_traj := r_fc, s_traj := s_fc, t_traj := t_fc), over the scored runs with a forecast; the registered functions,
    unchanged (W1: Track A's L1-L3; L2 there is Track A's closed form, G, T, T′ L2 the registered r_cf; neither is a
    1C forecast)."""
    if arm == "W1":
        from .track_a import score_opt
        r = score_opt(r_obs, np.where(fs, R["r_fc"], np.nan), np.where(fs, R["r_cf"], np.nan), R["crossed"], before)
        return {k: r[k] for k in ("L1", "L2", "L3")}
    r = _registered_score(arm, {**R, "follows": R["follows"] & fs}, np.where(fs, R["r_fc"], np.nan),
                          np.where(fs, R["s_fc"], np.nan), R["t_fc"], R["r_cf"], pilot_chi, s_pop)
    return {k: r.get(k) for k in ("L1", "L2", "L3", "L4", "L5", "n_scored", "valid")}


def phase2_gate(scores):
    """Gate to Phase 2: W1 PASS and (G PASS or T PASS)."""
    p = {a: scores.get(a, {}).get("outcome") == "PASS" for a in ARMS}
    return {"W1_pass": p["W1"], "G_pass": p["G"], "T_pass": p["T"], "pass": bool(p["W1"] and (p["G"] or p["T"]))}


# ------------------------------------------------------------------------------------------ machine rules and io
def memory_gate(tag):
    """free ≥ 25% and swap free ≥ 500 MB (act_fold.check_memory); waits (re-checking every 60 s) while it fails; every
    check logged to results/phase1c/memory_gate.log."""
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


def rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")


def _setup():
    os.nice(15)
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
    """Append one JSON row (the file is rewritten whole: a partial line is never left behind)."""
    old = p.read_text() if p.exists() else ""
    p.write_text(old + json.dumps(_jsonable(row)) + "\n")


def _rows(p):
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


# ------------------------------------------------------------------------------------------ seed scan
SEED_PREFIX = {"W1": "9350", "G": "9360", "T": "9370", "Tp": "9380"}
SCAN_SKIP_FILES = ("phase1c.py", "test_phase1c.py", "phase1c_registration.md")


def _scan_tree(patterns, chunk=16 * 1024 * 1024, overlap=256):
    """Regex scan of every text file under src, tests, results, paper (binary files and untracked path stores skipped;
    1C's own files skipped), streamed in 16 MB chunks with a 256-byte overlap.  {pattern key: [files with a match]}."""
    import re
    rx = {k: re.compile(v.encode()) for k, v in patterns.items()}
    hits = {k: [] for k in patterns}
    for top in ("src", "tests", "results", "paper"):
        for dp, _, fns in os.walk(ROOT / top):
            if dp.rstrip("/").endswith("paths") or os.path.basename(dp.rstrip("/")) == "phase1c":
                continue
            for fn in fns:
                p = Path(dp) / fn
                if fn in SCAN_SKIP_FILES or p.suffix in (".npz", ".npy", ".pdf", ".png", ".pt", ".pkl", ".ots", ".bak",
                                                          ".gz", ".zip"):
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
    return [int(v)] if isinstance(v, (int, np.integer)) else []


def registered_overlap():
    """1C seeds that are registered or pilot seeds of any test module (constants SEEDS* / PILOT_SEEDS*)."""
    import importlib
    mine = {s for ss in SEEDS.values() for s in ss}
    found = {}
    for mod in ("track_a", "track2a", "track2b", "track2c", "gelu_transfer", "width2_asym", "phase1a_pilot"):
        m = importlib.import_module(f"src.{mod}")
        for name in dir(m):
            if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
                v = getattr(m, name)
                found[f"{mod}.{name}"] = sorted(mine & set(_flat_ints(v)))
    return found


def scan():
    """The 1C seed ranges are unused: no 7-digit number with an arm's 4-digit prefix (also written 9_350_xxx or
    9,350,xxx) in any text file under src/, tests/, results/, paper/ (1C's own files excepted), and no overlap with the
    registered or pilot seeds of any test module."""
    OUT.mkdir(parents=True, exist_ok=True)
    pats = {}
    for arm, p in SEED_PREFIX.items():
        pats[arm] = (rf"(^|[^0-9.]){p}[0-9]{{3}}([^0-9]|$)|{p[0]}_{p[1:4]}_[0-9]{{3}}"
                     rf"|(^|[^0-9.,]){p[0]},{p[1:4]},[0-9]{{3}}([^0-9,]|$)")      # comma form: not inside a CSV row
    hits = _scan_tree(pats)
    ov = registered_overlap()
    out = {"ranges": {a: [s[0], s[-1], len(s)] for a, s in SEEDS.items()}, "prefix_pattern": pats,
           "prefix_pattern_files": hits, "registered_seed_overlap": ov,
           "unused": bool(not any(hits.values()) and not any(ov.values()))}
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable(out), indent=1))
    if not out["unused"]:
        raise SystemExit("STOP: a 1C seed range is not unused")


# ------------------------------------------------------------------------------------------ frozen per-seed inputs
def _frozen_file(arm):
    return OUT / f"frozen_{arm}.jsonl"


def w2a_arm(arm):
    from . import width2_asym as W
    return W.T if arm == "T" else W.TP


def freeze_one(arm, seed):
    """The per-seed inputs the replicated registration froze before training, by its own functions, unchanged.
    W1 (Track A R0 at a = 1.58, the Phase 1A landscape): s*_frozen (phase1a_pilot.w1_frozen) and the seed's global
    own-sample threshold (own_threshold; DESCRIPTIVE, as in Track A).  G: gelu_transfer.freeze_one (both copies, follow
    points, κ).  T, T′: width2_asym.freeze_copy per copy as the registration froze its arms (the arm's scored copy in
    full, the other classified copy as a point) and the arm's hold length (arm_hold_steps)."""
    if arm == "W1":
        from . import phase1a_pilot as PP
        from .own_threshold import own_threshold
        fr = PP.w1_frozen(seed)
        if fr.get("s_frozen") is not None:
            ot = own_threshold(PP.A_W1, seed, fr["s_frozen"])
            fr.update(w2_own_lo=ot["w2_lo"], w2_own_hi=ot["w2_hi"], own_evals=ot["evals"], own_note=ot["note"])
        return {"arm": arm, **fr}
    if arm == "G":
        from . import gelu_transfer as G
        return {"arm": arm, **G.freeze_one(seed)}
    from . import width2_asym as W
    a = w2a_arm(arm)
    land = W._land()
    x, y = W.own_sample(seed)
    full, point = (("T",), ("Tp",)) if a == W.T else (("Tp",), ("T",))
    copies = {c: W.freeze_copy(c, land, x, y, True) for c in full}
    copies.update({c: W.freeze_copy(c, land, x, y, False) for c in point})
    Wc = {c: r["W"] for c, r in copies.items() if r.get("point_ok")}
    return {"arm": arm, "seed": int(seed), "copies": W._jsonable_tree(copies), "W_arm": {a: W.arm_hold_steps(a, Wc)}}


def freeze(arm):
    """Resumable; the memory gate before every seed; one row per seed, written as it completes."""
    _setup()
    OUT.mkdir(parents=True, exist_ok=True)
    f = _frozen_file(arm)
    done = {r["seed"] for r in _rows(f)}
    for seed in SEEDS[arm]:
        if seed in done:
            continue
        memory_gate(f"freeze {arm} {seed}")
        t0 = time.time()
        r = freeze_one(arm, seed)
        r["secs"] = round(time.time() - t0, 1)
        _append_write(f, r)
        print(json.dumps({"arm": arm, "seed": seed, "secs": r["secs"]}), flush=True)
        rss_guard()
    rows = _rows(f)
    assert sorted(r["seed"] for r in rows) == list(SEEDS[arm]), "every seed exactly once"


def _frozen(arm):
    rows = {r["seed"]: r for r in _rows(_frozen_file(arm))}
    assert sorted(rows) == list(SEEDS[arm])
    return rows


# ------------------------------------------------------------------------------------------ registration manifest
FROZEN_DATA = ("results/designs/phase1c_causal_design.md", "results/phase1c_registration.md",
               "results/phase1c/seed_scan.json", "results/phase1c/seed_scan_v1.json", "results/phase1c/frozen_W1.jsonl", "results/phase1c/frozen_G.jsonl",
               "results/phase1c/frozen_T.jsonl", "results/phase1c/frozen_Tp.jsonl", "tests/test_phase1c.py",
               "tests/test_causal_forecast.py", "results/phase1a/landscape_w1.json", "results/phase1a/summary.json",
               "results/phase1a/fixtures/gelu_random.npz", "results/phase1a/fixtures/w2a_T.npz",
               "results/gelu_transfer/landscape.json", "results/gelu_transfer/pilot.json",
               "results/act_general/kappa_gelu_frozen.json", "results/width2_asym/landscape.json",
               "results/width2_asym/pilot.json", "results/asym_scores.json", "results/lag_law/kappa.csv")


def code_closure(start=("phase1c",)):
    """Every src module phase1c reaches by relative imports (`from . import X`, `from .X import`), recursively."""
    import ast
    seen, todo = set(), list(start)
    while todo:
        m = todo.pop()
        if m in seen or not (ROOT / "src" / f"{m}.py").exists():
            continue
        seen.add(m)
        tree = ast.parse((ROOT / "src" / f"{m}.py").read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level == 1:
                if n.module:
                    todo.append(n.module.split(".")[0])
                else:
                    todo.extend(a.name for a in n.names)
    return sorted(f"src/{m}.py" for m in seen)


def manifest_files():
    return sorted(set(code_closure()) | set(FROZEN_DATA))


def manifest():
    """results/phase1c/registration.sha256: SHA-256 of the registration md, the design page, the code closure, the
    tests, the frozen per-seed files and every frozen input the replicated pipelines read."""
    lines = []
    for rel in manifest_files():
        p = ROOT / rel
        assert p.exists(), f"missing {rel}"
        lines.append(f"{_sha(p)}  {rel}")
    (OUT / "registration.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def assert_registration():
    """Before any registered training or observation: every manifest hash equal and the manifest committed and clean."""
    m = OUT / "registration.sha256"
    for ln in m.read_text().splitlines():
        h, rel = ln.split()
        assert _sha(ROOT / rel) == h, f"registration hash mismatch: {rel}"
    _assert_committed(m)


def _assert_committed(p):
    rel = str(Path(p).resolve().relative_to(ROOT))
    assert subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip(), f"{rel} not committed"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0, f"{rel} modified"


# ------------------------------------------------------------------------------------------ the 1C prediction
def _s_of(out):
    return np.abs(out) if out.ndim == 1 else np.abs(out).sum(axis=1)


def harness_cutoff(arm, row, out, hid_rows, hid, ctx, f):
    """HARNESS SIDE (a stopping time; only the integer t_c and, for W1, the rule point and s*_run go on).  W1: the nested
    stopping time on s*_run (causal_forecast.w1_cutoff_on_s_run); G, T, T′: the first step with s_t ≥ f·s_ref, s_ref
    the occupied copy's frozen switch."""
    from . import phase1a_pilot as PP
    s = _s_of(out)
    if arm == "W1":
        hr = {int(r_): hid[i] for i, r_ in enumerate(hid_rows)}
        t_c, tR, s_run = C.w1_cutoff_on_s_run(s, lambda k: out[k], f, row["s_ref"], PP.A_W1, ctx["x"], ctx["y"],
                                              lambda k: hr[int(k)])
        return t_c, {"cutoff_rule_point": tR, "cutoff_s_run": s_run}
    return C.cutoff_step(s, f * row["s_ref"]), {}


def forecast_one(arm, row, out, hid_rows, hid, ctx, f):
    """ONE 1C prediction at cutoff fraction f.  The forecaster (causal_forecast.run_forecast, which re-asserts every
    guard's last row < t_c) receives GUARDED views built by the Phase 1A driver's make_inputs (rows < t_c; hidden state:
    one allowed row).  Then the per-run NaN recomputation: the same forecaster, unguarded, on copies with every row
    ≥ t_c (output path and hidden state) set to NaN; the record says whether the output is identical."""
    from . import phase1a_pilot as PP
    t0 = time.time()
    cfg = config(arm, f)
    rec = {"f": float(f)}
    t_c, extra = harness_cutoff(arm, row, out, hid_rows, hid, ctx, f)
    rec.update(extra)
    if t_c is None:
        return {**rec, "t_c": None, "status": "no cutoff: s never reaches f·s_ref", "secs": round(time.time() - t0, 2)}
    st = SETTING[arm]
    fc = C.run_forecast(KIND[arm], PP.make_inputs(st, row, out, hid_rows, hid, t_c, ctx, guarded=True), cfg)
    out_nan = np.array(out, float, copy=True)
    out_nan[t_c:] = np.nan
    hid_nan = np.array(hid, float, copy=True)
    hid_nan[np.asarray(hid_rows) >= t_c] = np.nan
    fc2 = C.ADAPTERS[KIND[arm]](PP.make_inputs(st, row, out_nan, hid_rows, hid_nan, t_c, ctx, guarded=False), cfg)
    rec.update({k: v for k, v in fc.items()})
    rec["t_sw_fc"] = fc.get("t_sw")
    rec["nan_recompute_identical"] = bool(same_forecast(fc, fc2))
    rec["secs"] = round(time.time() - t0, 2)
    return rec


# ------------------------------------------------------------------------------------------ training (no observation)
def _t_save(s, s_ref, n):
    """The saved output prefix: up to the first step with s ≥ 1.6·s_ref (the forecast horizon) plus 500 steps, or the
    whole path.  It never uses a crossing; observation retrains the run to the full budget."""
    t_hi = C.cutoff_step(s, HORIZON * s_ref) if s_ref else None
    return int(n - 1 if t_hi is None else min(n - 1, t_hi + SAVE_MARGIN))


def _save(arm, seed, out, hid_rows, hid):
    PATHS.mkdir(parents=True, exist_ok=True)
    p = PATHS / f"{arm}_{seed}.npz"
    np.savez_compressed(p, out=out, hid_rows=np.asarray(hid_rows, np.int64), hid=hid)
    return p.name, _sha(p)


def _load(row):
    p = PATHS / row["path"]
    assert _sha(p) == row["path_sha256"], f"path hash mismatch {row['path']}"
    d = np.load(p)
    return d["out"], d["hid_rows"], d["hid"]


def train_full(arm, seed, fr):
    """The replicated pipeline's run, unchanged, to the FULL budget: (record, output path, hidden rows, hidden rows'
    values, full parameter path, context).  No gap or placement of the state after release is evaluated."""
    if arm == "W1":
        from . import phase1a_pilot as PP
        W, _, _ = PP.w1_train(seed)
        s = np.abs(W[:, 2])
        rr = PP.w1_rule_rows(s, fr["s_frozen"])
        rec = {"setting": "w1_sgd", "budget": PP.W1_BUDGET, "s_ref": fr["s_frozen"], "rule_rows": rr, "eligible": True}
        return rec, W[:, 2], rr, (W[rr][:, [0, 1, 3]] if rr else np.zeros((0, 3))), W
    if arm == "G":
        from . import gelu_transfer as G
        land = G._land()
        rho = json.loads((G.OUT / "pilot.json").read_text())["rho"]
        th_rel, rel, (X, Y, u, P) = G.run_start(G.PRIMARY, seed, fr, land)
        Wp = G.release_train(th_rel, X, Y, u, rho, G.BUDGET)
        copy = rel["copy_at_release"]
        cf = fr["copies"].get(str(copy), {}) if copy is not None else {}
        ok = bool(rel["on_branch"] and cf.get("valid"))
        rec = {"setting": "gelu_random", "budget": G.BUDGET, "rho": rho, "release": rel, "eligible": ok,
               "s_ref": cf["s_switch"] if ok else None, "frozen": fr}
        return rec, Wp[:, 2], [0], Wp[[0]][:, [0, 1, 3]], Wp
    from . import width2_asym as W
    a = w2a_arm(arm)
    land = W._land()
    rho = json.loads((W.OUT / "pilot.json").read_text())["arms"][a]["rho"]
    z_rel, v0, rel, (x, y) = W.run_start(a, seed, fr, land)
    n = W.budget(a, rho)
    Pp = W.train_path(W.qof(z_rel, v0), W.ETA[a], rho, n, x, y)
    copy = rel["copy_at_release"]
    cf = fr["copies"].get(copy, {}) if copy is not None else {}
    ok = bool(rel["on_branch"] and cf.get("valid") and rel.get("windings") is not None
              and W.in_winding_table(tuple(rel["windings"])))
    rec = {"setting": SETTING[arm], "budget": n, "rho": rho, "release": rel, "eligible": ok,
           "s_ref": cf["s_switch"] if ok else None, "frozen": fr}
    return rec, Pp[:, W.VI], [0], Pp[[0]][:, W.ZI], Pp


def context(arm, row):
    from . import phase1a_pilot as PP
    return PP.context(SETTING[arm], row)


def run_one(arm, seed, fr):
    """Train (full budget, no observation), then the 1C forecasts at f and at the other f.  The row holds no observed
    quantity (FORBIDDEN keys asserted in finalize)."""
    t0 = time.time()
    row = {"arm": arm, "seed": int(seed)}
    if arm == "W1" and fr.get("s_frozen") is None:
        return {**row, "status": "no frozen switch", "eligible": False}
    rec, out, hid_rows, hid, full = train_full(arm, seed, fr)
    del full
    row.update(rec)
    row["status"] = "ok"
    if not row["eligible"]:
        row["status"] = "not eligible: not on a scored copy at release (or invalid copy / winding outside the table)"
        row["secs_run"] = round(time.time() - t0, 1)
        return row
    s = _s_of(out)
    t_end = _t_save(s, row["s_ref"], len(out))
    out = np.array(out[:t_end + 1], copy=True)
    hid_rows = list(hid_rows)
    name, h = _save(arm, seed, out, hid_rows, hid)
    row.update(n_saved=t_end + 1, path=name, path_sha256=h)
    ctx = context(arm, row)
    row["forecast"] = forecast_one(arm, row, out, hid_rows, hid, ctx, F[arm])
    row["forecast_other_f"] = forecast_one(arm, row, out, hid_rows, hid, ctx, F_OTHER[arm])
    row["secs_run"] = round(time.time() - t0, 1)
    return row


def _runs_file(arm):
    return OUT / f"runs_{arm}.jsonl"


def run(arm, n=None):
    """AFTER the registration commit and its timestamp: every seed of the arm (resumable), memory gate before each."""
    _setup()
    assert_registration()
    f = _runs_file(arm)
    done = {r["seed"] for r in _rows(f)}
    fz = _frozen(arm)
    for seed in SEEDS[arm][: (int(n) if n else None)]:
        if seed in done:
            continue
        memory_gate(f"run {arm} {seed}")
        r = run_one(arm, seed, fz[seed])
        _append_write(f, {k: v for k, v in r.items() if k != "frozen"})
        fc = r.get("forecast") or {}
        print(json.dumps(_jsonable({"arm": arm, "seed": seed, "status": r.get("status"), "t_c": fc.get("t_c"),
                                    "t_fc": fc.get("t_fc"), "nan_ok": fc.get("nan_recompute_identical"),
                                    "secs": r.get("secs_run")})), flush=True)
        rss_guard()


FORBIDDEN = {"t_obs", "s_obs", "crossed", "step_obs", "placed", "r_obs", "follows_branch", "t_follow"}


def _keys(o):
    if isinstance(o, dict):
        return set(o) | {k for v in o.values() for k in _keys(v)}
    if isinstance(o, list):
        return {k for v in o for k in _keys(v)}
    return set()


def finalize():
    """forecasts.sha256 over the four runs files: every seed exactly once, no observed quantity anywhere in a row.  The
    runs files and forecasts.sha256 are COMMITTED before observe."""
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


# ------------------------------------------------------------------------------------------ observation (after the forecasts commit)
def _w1_landscape():
    from . import phase1a_pilot as PP
    return json.loads((PP.OUT / "landscape_w1.json").read_text())


def observe_one(arm, row, fr):
    """Retrain the run (deterministic; the saved prefix asserted bit for bit), every-step detection over the full budget
    (the replicated test's detector), the actual lag-free switch, and for G, T, T′ the registered validity quantities
    and the follow check (registered predict_one with with_traj=False, unchanged)."""
    from . import linear_response as LR
    out_s, _, _ = _load(row)
    rec, out, _, _, full = train_full(arm, row["seed"], fr)
    assert np.array_equal(out[:len(out_s)], out_s), "retrained path differs from the saved prefix"
    o = {"arm": arm, "seed": row["seed"]}
    if arm == "W1":
        from . import phase1a_pilot as PP
        t_obs = PP.w1_observe(full)
        s = np.abs(full[:, 2])
        s_run = (row.get("forecast") or {}).get("cutoff_s_run")
        t_sw = LR._first_ge(s, s_run) if s_run is not None else None
        o.update(crossed=t_obs is not None, t_obs=t_obs, s_obs=float(s[t_obs]) if t_obs is not None else None,
                 s_sw=s_run, t_sw=t_sw)
        if t_sw is not None and t_sw >= 1:                     # DESCRIPTIVE: Track A's closed form (R5) at a = 1.58
            L = _w1_landscape()
            w = min(100, int(t_sw))
            sdot = (s[t_sw] - s[t_sw - w]) / w
            o["r_cf"] = float(L["kappa_sgd"] * (sdot / s_run) / (PP.W1_LR * L["lam_min"]))
        return o
    if arm == "G":
        import torch
        from . import gelu_transfer as G
        from .phase2b_ordering import state
        land = G._land()
        P = G.own_problem(row["seed"])
        u = G._act().torch_u
        t_obs = None
        for t in range(1, len(full)):
            if state(torch.tensor(full[t], dtype=torch.float64), u, None, 1.0)["placement_ok"]:
                t_obs = t
                break
        pr = G.predict_one(G.PRIMARY, row["seed"], full, row["release"], fr, land, P, with_traj=False)
        s = np.abs(full[:, 2])
        o.update(crossed=t_obs is not None, t_obs=t_obs, s_obs=float(s[t_obs]) if t_obs is not None else None)
    else:
        from . import width2_asym as W
        a = w2a_arm(arm)
        x, y = W.own_sample(row["seed"])
        t_obs, und = W.observe_path(full)
        pr = W.predict_one(a, row["seed"], full, row["release"], fr, W._land(), x, y, with_traj=False)
        s = np.abs(full[:, W.VI]).sum(axis=1)
        o.update(crossed=t_obs is not None, t_obs=t_obs, s_obs=float(s[t_obs]) if t_obs is not None else None,
                 n_undecided=und, s_pop=pr.get("s_pop_branch"))
    o.update(s_sw=pr.get("s_switch"), t_sw=pr.get("t_sw"), registered_status=pr.get("status"),
             **{k: pr.get(k) for k in ("kappa", "eta_lam", "lag_steps_pred", "chi_tsw", "r_cf", "chi_window_max",
                                       "follows_branch", "t_follow", "copy_at_follow")})
    return o


def _obs_file(arm):
    return OUT / f"observed_{arm}.jsonl"


def observe(arm):
    """AFTER the forecasts commit: the registration and forecast hashes asserted; eligible runs only (others are counted
    from the runs file); resumable; memory gate before each run."""
    _setup()
    assert_registration()
    _assert_forecasts()
    f = _obs_file(arm)
    done = {r["seed"] for r in _rows(f)}
    fz = _frozen(arm)
    for row in _rows(_runs_file(arm)):
        if row["seed"] in done or not row.get("eligible") or row.get("status") != "ok":
            continue
        memory_gate(f"observe {arm} {row['seed']}")
        t0 = time.time()
        o = observe_one(arm, row, fz[row["seed"]])
        o["secs"] = round(time.time() - t0, 1)
        _append_write(f, o)
        print(json.dumps(_jsonable({k: o.get(k) for k in ("arm", "seed", "crossed", "t_obs", "t_sw", "follows_branch",
                                                          "secs")})), flush=True)
        rss_guard()


# ------------------------------------------------------------------------------------------ scoring
def table(arm, runs, obs):
    """Arrays over ALL the arm's seeds for score_arm_1c (NaN / False where undefined)."""
    ob = {o["seed"]: o for o in obs}
    R = {k: [] for k in ("seed", "crossed", "t_obs", "s_obs", "t_sw", "s_sw", "t_c", "t_fc", "t_sw_fc", "s_fc", "r_fc",
                         "nan_identical", "hold_positive", "t_c_other", "t_fc_other", "t_sw_fc_other", "r_fc_other",
                         "r_cf", "frozen_ok", "on_branch", "follows", "kappa", "eta_lam", "lag_steps", "chi_tsw",
                         "chi_win_max", "s_pop")}
    for r in sorted(runs, key=lambda r: r["seed"]):
        o = ob.get(r["seed"], {})
        fc, fo = r.get("forecast") or {}, r.get("forecast_other_f") or {}
        rel = r.get("release") or {}
        g = lambda d, k: d.get(k) if d.get(k) is not None else np.nan          # noqa: E731
        R["seed"].append(r["seed"])
        R["crossed"].append(bool(o.get("crossed", False)))
        for k in ("t_obs", "s_obs", "t_sw", "s_sw", "r_cf", "kappa", "eta_lam", "chi_tsw", "s_pop"):
            R[k].append(g(o, k))
        R["lag_steps"].append(g(o, "lag_steps_pred"))
        R["chi_win_max"].append(g(o, "chi_window_max"))
        R["follows"].append(bool(o.get("follows_branch", False)))
        for k, k2 in (("t_c", "t_c"), ("t_fc", "t_fc"), ("t_sw_fc", "t_sw_fc"), ("s_fc", "s_fc"), ("r_fc", "r_fc")):
            R[k].append(g(fc, k2))
        for k in ("t_c", "t_fc", "t_sw_fc", "r_fc"):
            R[f"{k}_other"].append(g(fo, k))
        R["nan_identical"].append(bool(fc.get("nan_recompute_identical", True)) if fc.get("t_c") is not None else True)
        R["hold_positive"].append(bool(rel.get("hold_G_positive", False)))
        R["frozen_ok"].append(r.get("status") != "no frozen switch")
        R["on_branch"].append(bool(rel.get("on_branch", False)) if arm != "W1" else True)
    return R


def score():
    """scores.json: per arm the registered 1C verdict (score_arm_1c), and the gate to Phase 2."""
    from . import gelu_transfer as G
    from . import width2_asym as W
    assert_registration()
    _assert_forecasts()
    res = {"label": "PHASE 1C (registered causal replication)", "forecasts_sha256": _sha(OUT / "forecasts.sha256"),
           "registration_sha256_file_sha256": _sha(OUT / "registration.sha256"), "arms": {}}
    for arm in ARMS:
        R = table(arm, _rows(_runs_file(arm)), _rows(_obs_file(arm)))
        if arm == "W1":
            pc, sp = None, None
        elif arm == "G":
            pc = json.loads((G.OUT / "pilot.json").read_text())["pilot_median_chi_tsw"][G.PRIMARY]
            sp = G._land()["s_glob"]
        else:
            pc = json.loads((W.OUT / "pilot.json").read_text())["arms"][w2a_arm(arm)]["pilot_median_chi_tsw"]
            sp = R["s_pop"]
        res["arms"][arm] = score_arm_1c(arm, R, pilot_chi=pc, s_pop=sp)
    res["phase2_gate"] = phase2_gate(res["arms"])
    (OUT / "scores.json").write_text(json.dumps(_jsonable(res), indent=1))
    print(json.dumps(_jsonable({a: v["outcome"] for a, v in res["arms"].items()} | {"phase2": res["phase2_gate"]})))


PILOT_SETTING = {"W1": "w1_sgd_run", "G": "gelu_random", "T": "w2a_T", "Tp": "w2a_Tp"}


def pilot_check():
    """DISCLOSURE (pre-registration; Phase 1A est seeds only): the 1C criteria code applied to the committed Phase 1A
    est forecasts (primary quad / 0.05) at each arm's f and other f -> results/phase1c/pilot_check.json.  Every est
    run with a cutoff before its crossing is treated as scored (the pilot has no gate or V1-V7)."""
    from . import phase1a_pilot as PP
    out = {"label": "PRE-REGISTRATION DISCLOSURE: the 1C criteria on the Phase 1A est seeds (not 1C data)", "arms": {}}
    for arm in ARMS:
        rows = PP._rows(PP.OUT / f"forecasts_{PILOT_SETTING[arm]}.jsonl")
        out["arms"][arm] = {}
        for f in (F[arm], F_OTHER[arm]):
            rs = [r for r in rows if r["split"] == "est" and r["family"] == FAMILY and r["window_frac"] == WINDOW_FRAC
                  and r["f"] == f and (r.get("actual_PILOT_ONLY") or {}).get("t_obs") is not None]
            A = lambda k: [((r.get("actual_PILOT_ONLY") or {}).get(k)) for r in rs]          # noqa: E731
            t_obs, t_sw = _arr(A("t_obs")), _arr(A("t_sw_act"))
            s_obs, s_sw = _arr(A("s_obs")), _arr(A("s_sw_act"))
            t_c = _arr([r.get("t_c") for r in rs])
            bef = cutoff_before(t_c, t_obs)
            t_fc, t_sw_fc = _arr([r.get("t_fc") for r in rs]), _arr([r.get("t_sw") for r in rs])
            r_fc = _arr([r.get("r_fc") for r in rs])
            with np.errstate(invalid="ignore", divide="ignore"):
                r_obs = s_obs / s_sw - 1
            sc = bef & np.isfinite(t_sw)
            hf = is_forecast(t_fc, t_sw_fc)
            out["arms"][arm][str(f)] = {"n_crossed": len(rs), "cutoff_validity": cutoff_validity(bef, np.ones(len(rs), bool)),
                                        "n_scored": int(sc.sum()),
                                        "criteria": criteria(arm, t_fc[sc], t_sw_fc[sc], r_fc[sc], t_obs[sc], t_sw[sc],
                                                             r_obs[sc], hf[sc]),
                                        "horizon": horizons(t_c[sc], t_obs[sc], t_sw[sc], t_obs[sc] - t_sw[sc])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_check.json").write_text(json.dumps(_jsonable(out), indent=1))
    return out


def main(argv):
    cmd = argv[1]
    if cmd == "scan":
        scan()
    elif cmd == "freeze":
        freeze(argv[2])
    elif cmd == "pilot_check":
        pilot_check()
    elif cmd == "manifest":
        manifest()
    elif cmd == "run":
        run(argv[2], argv[3] if len(argv) > 3 else None)
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
