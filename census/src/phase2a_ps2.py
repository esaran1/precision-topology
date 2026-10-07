"""Phase 2A-PS2: the per-seed fold test with the pipeline repaired (design: results/designs/phase2a_ps2_design.md,
approved by the author 2026-10-07 "as written, including the worker's edits", commit 820ef52; registration:
results/phase2a_ps2_registration.md).

Registered after 2A-PS's UNRESOLVED (validity) outcome, which STAYS.  This is a new registration, not a re-scoring.
Everything is 2A-PS's (src/phase2a_ps.py, imported and NEVER modified: it is hashed in 2A-PS's registration) except
three changes, each set after 2A-PS data (‡, from the POST HOC diagnosis 2f0fb33):

  (a) ACTIVITY TEST at the release (Phase 3's): unit k is active iff |v_k| ≥ 1e−8·s, s = ‖v‖₁.  A release with a unit
      that 2A-PS's rule (v ≠ 0.0) counts active but this test does not is "inactive unit at release": counted, never
      trained or scored (status right after "release Newton failed").
  (b) PER-SEED BUDGET (Phase 3's): t* = (1/ρ)∫ds/(3·(−dL_M/ds)) from s₀ to 0.9999·s_F on the seed's frozen M
      (dL_M/ds ≥ 0 anywhere: "stall", counted); B = min(⌈k·t*⌉ + 3000, ⌈192/ρ⌉), k = 1.5 clean, 2.5 none and mixed;
      ⌈k·t*⌉ + 3000 > ⌈192/ρ⌉: "over cap", counted, not trained.  Run, forecast and observation budgets: B (forecast
      budget B − t_c).  Hard per-run caps: ⌈192/ρ⌉ steps and 1 GB current RSS ("aborted: per-run RSS cap", counted,
      not scored).
  (c) H on the REGISTERED 30-minimum class rule, unchanged.  The posthoc WIDER search (360 minima at 1.01/1.02/1.05/
      1.10·s_F × perturbation ×1/×2/×4 × 30, default_rng([seed, 20261005])) is frozen per seed and reported
      DESCRIPTIVELY only, as crossing rates by wider-search class.  It is never a criterion.

Seeds: registered 2,987,000–2,987,599 (600), pilot 2,988,000–2,988,059.  Gate ≥ 24 clean scoreable; caps 40/30/20 in
seed order; fewer than 10 trained none seeds: H UNRESOLVED.  Author's pre-registration check: STOP if more than 5
scoreable clean seeds or more than 5 scoreable none seeds are over cap.

ORDER: scan / pilot / freeze / manifest before the registration commit; run → finalize (commit) → observe → score
after it (as 2A-PS).  Machine rules: one process, nice 15, one thread; memory gate (free ≥ 25%, swap free ≥ 500 MB;
waits) and disk check (≥ 20 GB, else STOP) before every job and between seeds, logged to
results/phase2a_ps2/memory_gate.log; process stop above 3 GB peak RSS; per-run caps as in (b).  No global RNG state.

    python -m src.phase2a_ps2 gate TAG | scan | pilot | freeze | summary | reevaluate | manifest | stamp
    python -m src.phase2a_ps2 run | finalize | observe | score
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
from . import phase2a_ps as PS
from . import phase2a_ps_posthoc as PH
from . import sb_fold as SBF

ROOT = PS.ROOT
RESULTS = PS.RESULTS
OUT = RESULTS / "phase2a_ps2"
REGISTRATION_MD = RESULTS / "phase2a_ps2_registration.md"
DESIGN_MD = RESULTS / "designs" / "phase2a_ps2_design.md"
EXPLORE_DIR = RESULTS / "designs" / "phase2a_ps2_explore"

# ------------------------------------------------------------------------------------------ registered constants
SEEDS = tuple(range(2_987_000, 2_987_600))       # 600 registered seeds (author-approved 2026-10-07)
PILOT_SEEDS = tuple(range(2_988_000, 2_988_060))  # 60 pilot seeds
EXPLORATION_SEEDS = range(2_930_112, 2_930_200)  # the design's exploration (never registered)

THETA = 1e-8                                     # (a) active iff |v_k| ≥ THETA·s (inclusive)
N_ACTIVE = 3                                     # (b) the page's 3 in t* = (1/ρ)∫ds/(3·(−dL_M/ds))
TSTAR_N = 121                                    # t*'s grid: linspace(s₀, 0.9999·s_F, 121) (the exploration's)
TSTAR_END = 0.9999                               # upper end of the integral (× s_F)
SLOPE_DS_REL = PH.SLOPE_DS_REL                   # dL_M/ds: central difference, half-width 1e−6·s_F
SLOPE_TOP = 0.99995                              # the upper difference point is capped at 0.99995·s_F (below the fold)
K_BUDGET = {"clean": 1.5, "none": 2.5, "mixed": 2.5}
BUDGET_ADD = 3000
CAP_FACTOR = 192.0                               # hard per-run step cap ⌈192/ρ⌉
RSS_RUN_CAP = 1024 ** 3                          # per-run cap on the CURRENT RSS (1 GB)
RSS_EVERY = 20_000                               # checked every 20,000 steps (Phase 3's)
OVERCAP_STOP_MAX = 5                             # author's check: STOP if > 5 scoreable clean or > 5 none over cap
RATES = {"clean": (PS.LOG2_EXPONENT, PS.LOG2_SCORED), "none": (PS.LOG2_OTHER,), "mixed": (PS.LOG2_OTHER,)}
EXTRAS_FOR = ("ineligible", "scoreable")         # 2A-PS statuses for which t* and the wider search are frozen

CAPS = PS.CAPS                                   # 40 / 30 / 20 (unchanged)
GATE_MIN_CLEAN = PS.GATE_MIN_CLEAN               # 24 (unchanged)
H_MIN_N = PS.H_MIN_N                             # 10 (unchanged)

INACTIVE = "inactive unit at release"
TSTAR_FAILED = "t* not computable"
STALL = "stall"
OVER_CAP = "over cap"
STATUS_ORDER = (PS.STATUS_ORDER[:PS.STATUS_ORDER.index("release Newton failed") + 1] + (INACTIVE, "ineligible")
                + (TSTAR_FAILED, STALL, OVER_CAP, "scoreable"))
ABORTED = "aborted: per-run RSS cap (1 GB)"


def rho_of(l2):
    return PS.rho_of(l2)


def step_cap(rho):
    """The hard per-run step cap ⌈192/ρ⌉ (3,145,728 at 2⁻¹⁴; 12,582,912 at 2⁻¹⁶)."""
    return int(math.ceil(CAP_FACTOR / rho))


# ------------------------------------------------------------------------------------------ (a) the activity test
def active_units(v, theta=THETA):
    """Units with |v_k| ≥ θ·s, s = ‖v‖₁ (inclusive).  [] if s = 0."""
    v = np.asarray(v, float)
    s = float(np.abs(v).sum())
    if s == 0.0:
        return []
    return [k for k in range(len(v)) if abs(v[k]) >= theta * s]


def scale_direction(v, theta=THETA):
    """â = sign(v_A)/√|A| on the active set A only (0 elsewhere)."""
    v = np.asarray(v, float)
    a = np.zeros_like(v)
    A = active_units(v, theta)
    if A:
        a[A] = np.sign(v[A]) / math.sqrt(len(A))
    return a


def activity(release_theta, theta=THETA):
    """The release's units: 2A-PS's active set (v ≠ 0.0), the activity test's, their shares |v_k|/s, and the units
    2A-PS counts active that the test does not (inactive)."""
    v = np.asarray(release_theta, float)[12:16]
    s = float(np.abs(v).sum())
    reg = [k for k in range(4) if v[k] != 0.0]
    act = active_units(v, theta)
    return {"active_registered": reg, "active_theta": act, "inactive": [k for k in reg if k not in act],
            "shares": [float(abs(v[k]) / s) if s > 0 else None for k in range(4)],
            "min_share_registered": float(min(abs(v[k]) / s for k in reg)) if reg and s > 0 else None}


# ------------------------------------------------------------------------------------------ (b) t* and the budget
def dLds_grid(M, s0, s_F):
    """dL_M/ds on linspace(s₀, 0.9999·s_F, 121) by central differences (half-width 1e−6·s_F; the upper point capped at
    0.99995·s_F), Newton on the seed's own M at each point (the exploration's p2_explore.rho_t_star).  NaN where Newton
    fails."""
    ss = np.linspace(s0, TSTAR_END * s_F, TSTAR_N)
    h = SLOPE_DS_REL * s_F
    d = []
    for s in ss:
        up = min(s + h, SLOPE_TOP * s_F)
        a, b = M.loss(up), M.loss(s - h)
        d.append(np.nan if a is None or b is None else (a - b) / (up - (s - h)))
    return ss, np.array(d)


def rho_t_star_of(ss, d, s0, s_F, n=N_ACTIVE):
    """(ρ·t*, status): ρ·t* = ∫ds/(n·(−dL_M/ds)) from s₀ to 0.9999·s_F (posthoc quasi_static_rho_time, trapezoid on
    the grid).  status 'ok'; 'stall' if dL_M/ds ≥ 0 at any grid point (ρ·t* None); 'not computable' if any grid value
    is missing or not finite (ρ·t* None)."""
    d = np.array([np.nan if x is None else x for x in d], float)
    if not np.all(np.isfinite(d)):
        return None, "not computable"
    if np.any(d >= 0):
        return None, "stall"
    rt = PH.quasi_static_rho_time(np.asarray(ss, float), d, s0, TSTAR_END * s_F, n_active=n)
    if not math.isfinite(rt):
        return None, "stall"
    return float(rt), "ok"


def tstar_record(M, s0, s_F):
    ss, d = dLds_grid(M, s0, s_F)
    rt, st = rho_t_star_of(ss, d, s0, s_F)
    return {"grid_s": [float(x) for x in ss], "dLds": [float(x) if np.isfinite(x) else None for x in d],
            "rho_t_star": rt, "status": st, "n_active": N_ACTIVE,
            "stall_scale": PH.stall_scale(ss, d) if st == "stall" else None}


def budget_steps(rho_t_star, k, rho):
    """{t_star, B_uncapped = ⌈k·t*⌉ + 3000, cap = ⌈192/ρ⌉, B = min(B_uncapped, cap), over_cap = B_uncapped > cap}."""
    t_star = float(rho_t_star) / float(rho)
    unc = int(math.ceil(float(k) * t_star)) + BUDGET_ADD
    cap = step_cap(rho)
    return {"rho": float(rho), "k": float(k), "t_star": t_star, "B_uncapped": unc, "cap": cap, "B": int(min(unc, cap)),
            "over_cap": bool(unc > cap)}


def budgets(cls, rho_t_star):
    """The seed's budget at each of its class's rates (clean 2⁻¹⁴ and 2⁻¹⁶; none, mixed 2⁻¹⁴)."""
    return {f"{l2:g}": budget_steps(rho_t_star, K_BUDGET[cls], rho_of(l2)) for l2 in RATES[cls]}


# ------------------------------------------------------------------------------------------ (c) the wider search
def wide_search(M, s_F, X, Y, seed):
    """DESCRIPTIVE ONLY (never a criterion): the posthoc wider search exactly as the exploration ran it — 30 perturbed
    minima per (scale, amplitude) at 1.01/1.02/1.05/1.10·s_F × ×1/×2/×4 around M(0.9999·s_F), one
    default_rng([seed, 20261005]) per seed in that order; ρ₂ by sb.feature_usage; class by posthoc wider_class."""
    Pb = SBF.run_to_full(M.theta(TSTAR_END * s_F)[None])[0]
    rng = np.random.default_rng([int(seed), PH.WIDE_STREAM])
    blocks = []
    for f in PH.WIDE_SCALES:
        for amp in PH.WIDE_AMPS:
            P0 = Pb[None] + rng.normal(0, 1, (PH.WIDE_N, 16)) * amp * (0.05 * np.abs(Pb) + 0.02)
            P0[:, 12:] = np.abs(P0[:, 12:])
            Pm, L, gn = SBF.local_min_batch(P0, np.full(PH.WIDE_N, f * s_F), X, Y)
            r = np.array([PS.rho2_static(p, X) for p in Pm])
            blocks.append({"scale": f, "amp": amp, "n": PH.WIDE_N, "n_ge_q": int((r >= PS.Q).sum())})
    return {"blocks": blocks, "class": PH.wider_class(blocks)}


# ------------------------------------------------------------------------------------------ the per-seed freeze
def freeze_raw(seed, timing=True):
    """2A-PS's raw record (phase2a_ps.freeze_raw, unchanged) plus 'ps2': the activity test at the release (every seed
    with a release), and for seeds whose 2A-PS status is ineligible or scoreable, the seed's M rebuilt by the
    registered steps (s_F and the release hash asserted equal), t* on it and the wider search."""
    t0 = time.time()
    rec = PS.freeze_raw(seed, timing=False)
    ev0 = PS.evaluate(rec)
    ps2 = {"ps_status": ev0["status"]}
    if rec.get("release_theta") is not None:
        ps2["activity"] = activity(rec["release_theta"])
    if ev0["status"] in EXTRAS_FOR:
        X, Y = PS.sample(seed)
        M = PH.seed_M(rec, X, Y)
        ps2["tstar"] = tstar_record(M, rec["s0"], rec["s_F"])
        ps2["wide"] = wide_search(M, rec["s_F"], X, Y, seed)
    rec["ps2"] = ps2
    if timing:
        rec["secs"] = round(time.time() - t0, 2)
    return rec


def freeze_seed(seed, timing=True):
    rec = freeze_raw(seed, timing)
    rec["evaluation"] = evaluate(rec)
    return rec


def evaluate(rec, repairs=True):
    """2A-PS's evaluation (phase2a_ps.evaluate, unchanged); with repairs=False it is returned as is (2A-PS's rules).
    With the repairs, a seed 2A-PS calls ineligible or scoreable is re-checked, in STATUS_ORDER:
      inactive unit at release (a unit with v ≠ 0 but |v| < 1e−8·s) → ineligible (2A-PS's) → t* not computable →
      stall → over cap (at any of its class's rates) → scoreable.
    The class (the registered 30-minimum rule) is never changed.  Budgets and the wider-search class are attached."""
    ev = PS.evaluate(rec)
    if not repairs:
        return ev
    ps2 = rec.get("ps2") or {}
    act = ps2.get("activity")
    ev["activity_inactive"] = None if act is None else list(act["inactive"])
    ts = ps2.get("tstar")
    ev["rho_t_star"] = None if ts is None else ts["rho_t_star"]
    ev["tstar_status"] = None if ts is None else ts["status"]
    ev["budgets"] = None
    ev["over_cap"] = None
    w = ps2.get("wide")
    ev["wide_class_DESCRIPTIVE"] = None if w is None else w["class"]["pooled"]
    if ev["status"] not in EXTRAS_FOR:
        return ev
    if act is None or act["inactive"]:
        ev["status"] = INACTIVE
        return ev
    if ev["status"] == "ineligible":
        return ev
    assert act["active_theta"] == act["active_registered"] and len(act["active_theta"]) == N_ACTIVE, \
        f"seed {rec['seed']}: active set {act}"
    if ts is None or ts["status"] == "not computable":
        ev["status"] = TSTAR_FAILED
        return ev
    if ts["status"] == "stall":
        ev["status"] = STALL
        return ev
    b = budgets(ev["class"], ts["rho_t_star"])
    ev["budgets"] = b
    ev["over_cap"] = bool(any(x["over_cap"] for x in b.values()))
    ev["status"] = OVER_CAP if ev["over_cap"] else "scoreable"
    return ev


def _st(r):
    return r["evaluation"]["status"]


def _cl(r):
    return r["evaluation"]["class"]


def counts(rows):
    """Counts by status (untraceable FIRST, as the headline) and the class of every status (counted when scoreable)."""
    st = [_st(r) for r in rows]
    c = {k: int(sum(1 for x in st if x == k)) for k in STATUS_ORDER}
    assert sum(c.values()) == len(rows), "a status outside STATUS_ORDER"
    by = {k: {cl: int(sum(1 for r in rows if _st(r) == k and _cl(r) == cl)) for cl in ("clean", "none", "mixed")}
          for k in STATUS_ORDER if not k.startswith("untraceable")}
    n_unt = int(sum(1 for x in st if x.startswith("untraceable")))
    return {"n_seeds": len(rows), "n_untraceable": n_unt,
            "untraceable_frac": n_unt / len(rows) if rows else None, "by_status": c,
            "class_by_status": by, "n_scoreable": c["scoreable"], "scoreable_by_class": by["scoreable"],
            "n_inactive_unit": c[INACTIVE], "n_stall": c[STALL], "n_over_cap": c[OVER_CAP],
            "n_tstar_not_computable": c[TSTAR_FAILED]}


def gate(rows):
    n = int(sum(1 for r in rows if _st(r) == "scoreable" and _cl(r) == "clean"))
    return {"n_clean_scoreable": n, "min": GATE_MIN_CLEAN, "pass": bool(n >= GATE_MIN_CLEAN)}


def over_cap_check(rows):
    """The author's pre-registration check: the number of seeds over cap (every earlier check passed) per class; FAIL
    (STOP before registering) if more than 5 clean or more than 5 none."""
    n = {k: int(sum(1 for r in rows if _st(r) == OVER_CAP and _cl(r) == k)) for k in ("clean", "none", "mixed")}
    return {"n_over_cap": n, "max_allowed": OVERCAP_STOP_MAX,
            "pass": bool(n["clean"] <= OVERCAP_STOP_MAX and n["none"] <= OVERCAP_STOP_MAX),
            "seeds": {k: [r["seed"] for r in rows if _st(r) == OVER_CAP and _cl(r) == k] for k in n}}


def plan(rows):
    """The first 40 clean (2⁻¹⁴ and 2⁻¹⁶), 30 none and 20 mixed (2⁻¹⁴) SCOREABLE seeds in seed order, each run with its
    own frozen budget B; the rest counted, not trained.  ([{seed, class, log2rho, budget}], {class: [surplus]})."""
    rows = sorted(rows, key=lambda r: r["seed"])
    chosen = {k: [] for k in CAPS}
    surplus = {k: [] for k in CAPS}
    for r in rows:
        if _st(r) != "scoreable":
            continue
        k = _cl(r)
        (chosen[k] if len(chosen[k]) < CAPS[k] else surplus[k]).append(r)
    pl = []
    for k in ("clean", "none", "mixed"):
        for r in chosen[k]:
            for l2 in RATES[k]:
                b = r["evaluation"]["budgets"][f"{l2:g}"]
                assert not b["over_cap"] and b["B"] <= b["cap"]
                pl.append({"seed": r["seed"], "class": k, "log2rho": l2, "budget": b["B"]})
    return pl, {k: [r["seed"] for r in v] for k, v in surplus.items()}


def class_listing(rows):
    """'seed status class' per seed in seed order (hashed into frozen.json)."""
    return "".join(f"{r['seed']} {_st(r)} {_cl(r)}\n" for r in sorted(rows, key=lambda r: r["seed"]))


def wide_table(rows):
    """DESCRIPTIVE: registered class × wider-search class over the scoreable seeds."""
    out = {}
    for r in rows:
        if _st(r) != "scoreable":
            continue
        w = r["evaluation"].get("wide_class_DESCRIPTIVE")
        d = out.setdefault(_cl(r), {})
        d[str(w)] = d.get(str(w), 0) + 1
    return out


# ------------------------------------------------------------------------------------------ training and forecasts
def current_rss():
    """The process's current resident set in bytes (ps)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
        return int(out.strip()) * 1024
    except (OSError, ValueError):
        return 0


def _rss_over(rss_fn):
    return rss_fn() > RSS_RUN_CAP


def train_to_cutoff(theta0, rho, s_F, X, Y, n_budget, f=PS.F_CUT, rss_fn=current_rss):
    """phase2a_ps.train_to_cutoff (the same step and the same stopping time; equal bit for bit, tested) with the budget
    B and the per-run caps: n_budget ≤ ⌈192/ρ⌉ (asserted) and the current RSS ≤ 1 GB every 20,000 steps.
    Returns (s_0 … s_t, θ_t, t_c or None, status ∈ {'ok', 'no cutoff within the budget', ABORTED})."""
    nb = int(n_budget)
    assert 1 <= nb <= step_cap(rho), "budget above the per-run step cap"
    th = np.array(theta0, float)
    active = [k for k in range(4) if th[12 + k] != 0.0]
    step = PS.make_step(rho, active, X, Y)
    S = np.empty(nb + 1)
    S[0] = PS.scale(th)
    level = f * s_F
    for t in range(1, nb + 1):
        th = step(th)
        S[t] = PS.scale(th)
        if S[t] >= level:
            return S[:t + 1].copy(), th, t, "ok"
        if t % RSS_EVERY == 0 and _rss_over(rss_fn):
            return S[:t + 1].copy(), th, None, ABORTED
        if t % 1_000_000 == 0:
            P2A.rss_guard()
    return S, th, None, "no cutoff within the budget"


def run_one(fs, l2, n_budget, rss_fn=current_rss):
    """One planned run (phase2a_ps.run_one with the seed's budget B): training from the frozen release to t_c, the
    state hashed, the causal forecast with the seed's own (s_F, Λ_F) and the fixed-dataset competitor at the same
    t_c, forecast budget B − t_c; both guarded, NaN recomputation identical (else CausalityViolation)."""
    rho = rho_of(l2)
    X, Y = PS.sample(fs["seed"])
    assert PS.sample_sha(X, Y) == fs["sample_sha256"], "sample hash mismatch"
    th0 = np.array(fs["release_theta"], float)
    assert PS.state_sha(th0) == fs["release_sha256"], "release hash mismatch"
    nb = int(n_budget)
    S, th, t_c, st = train_to_cutoff(th0, rho, fs["s_F"], X, Y, nb, PS.F_CUT, rss_fn)
    rec = {"seed": fs["seed"], "class": fs["class"], "log2rho": float(l2), "rho": rho, "budget": nb,
           "step_cap": step_cap(rho), "t_c": t_c, "run_status": st, "aborted": st == ABORTED}
    if t_c is None:
        why = ABORTED if st == ABORTED else "no forecast: no cutoff within the budget"
        nofc = {"status": why, "t_fc": None, "t_F_fc": None, "s_c_fc": None}
        rec.update(forecast=nofc, forecast_fixed=dict(nofc), nan_recompute_identical=None)
        return rec
    fc, same = CFF.run_forecast(S, t_c, fs["s_F"], fs["Lambda_F"], PS.ETA, nb - t_c, PS.F_CUT)
    fx, same_x = CFF.run_forecast(S, t_c, PS.S_F_FIXED, PS.LAMF_FIXED, PS.ETA, nb - t_c, PS.F_CUT)
    if not (same and same_x):
        raise CF.CausalityViolation(f"NaN recomputation differs (seed {fs['seed']}, log2rho {l2})")
    rec.update(s_at_tc=float(S[t_c]), state_tc=[float(x) for x in th], state_tc_sha256=PS.state_sha(th), forecast=fc,
               forecast_fixed=fx, nan_recompute_identical=bool(same and same_x))
    return rec


def observe_one(fs, l2, run_rec, n_budget, rss_fn=current_rss):
    """phase2a_ps.observe_one (equal bit for bit, tested) with the budget B and the per-run RSS cap (aborted: the
    record carries 'aborted' True and is counted, not scored).  ρ₂ at EVERY step; at t_c the state is asserted equal
    to the hashed one bit for bit; stop when the crossing and s_F are both reached (t ≥ t_c), or s ≥ 1.25·s_F, or B."""
    rho = rho_of(l2)
    X, Y = PS.sample(fs["seed"])
    assert PS.sample_sha(X, Y) == fs["sample_sha256"], "sample hash mismatch"
    rho2 = PS.Rho2Grid(X)
    s_F = fs["s_F"]
    th = np.array(fs["release_theta"], float)
    active = [k for k in range(4) if th[12 + k] != 0.0]
    idle = [k for k in range(4) if k not in active]
    idle_idx = [i for k in idle for i in (2 * k, 2 * k + 1, 8 + k, 12 + k)]
    sg0 = np.sign(th[12:16][active])
    step = PS.make_step(rho, active, X, Y)
    nb = int(n_budget)
    assert 1 <= nb <= step_cap(rho), "budget above the per-run step cap"
    t_c = run_rec["t_c"]
    S = np.empty(nb + 1)
    S[0] = PS.scale(th)
    r0 = rho2(th)
    below = r0 < PS.Q
    t_obs = rho2_obs = t_F = t08 = th08 = None
    idle_viol = sign_viol = hash_ok = None
    aborted = False
    t = 0
    for t in range(1, nb + 1):
        th = step(th)
        S[t] = PS.scale(th)
        if t_obs is None:
            if idle_viol is None and idle_idx and np.any(th[idle_idx] != 0.0):
                idle_viol = t
            if sign_viol is None and np.any(np.sign(th[12:16][active]) != sg0):
                sign_viol = t
            r = rho2(th)
            if r >= PS.Q and below:
                t_obs, rho2_obs = t, r
            elif r < PS.Q:
                below = True
        if t08 is None and S[t] >= PS.FOLLOW_FRAC * s_F:
            t08, th08 = t, th.copy()
        if t_F is None and S[t] >= s_F:
            t_F = t
        if t_c is not None and t == t_c:
            hash_ok = bool(PS.state_sha(th) == run_rec["state_tc_sha256"]
                           and np.array_equal(th, np.array(run_rec["state_tc"], float)))
            if not hash_ok:
                raise AssertionError(f"state at t_c differs from the hashed one (seed {fs['seed']}, log2rho {l2})")
            th = np.array(run_rec["state_tc"], float)
        if S[t] >= PS.OBS_END * s_F:
            break
        if t_obs is not None and t_F is not None and (t_c is None or t >= t_c):
            break
        if t % RSS_EVERY == 0 and _rss_over(rss_fn):
            aborted = True
            break
        if t % 1_000_000 == 0:
            P2A.rss_guard()
    S = S[:t + 1].copy()
    follow = PS.follow_check(th08, fs, X, Y) if th08 is not None else (False, math.inf)
    rec = {"seed": fs["seed"], "class": fs["class"], "log2rho": float(l2), "budget": nb, "steps_observed": int(t),
           "t_obs": t_obs, "s_obs": float(S[t_obs]) if t_obs is not None else None, "rho2_at_obs": rho2_obs,
           "rho2_release": r0, "t_F": t_F, "s_end": float(S[-1]), "reached_obs_end": bool(S[-1] >= PS.OBS_END * s_F),
           "t_08": t08, "follow_in_M": bool(follow[0]), "follow_dist_M": follow[1],
           "idle_zero_to_crossing": idle_viol is None, "idle_first_violation": idle_viol,
           "signs_fixed_to_crossing": sign_viol is None, "sign_first_violation": sign_viol, "state_tc_hash_ok": hash_ok,
           "aborted": aborted}
    return rec, S


# ------------------------------------------------------------------------------------------ scoring
def run_row(fs, rr, ob):
    """phase2a_ps.run_row plus the abort flag and the (DESCRIPTIVE) wider-search class."""
    row = PS.run_row(fs, rr, ob)
    row["aborted"] = bool(rr.get("aborted") or ob.get("aborted"))
    row["budget"] = rr.get("budget")
    row["wide_class_DESCRIPTIVE"] = fs.get("wide_class")
    return row


def score_tables(rows, gate_ok=True):
    """phase2a_ps.score_tables (every criterion, validity and the outcome rule unchanged) over the scored rows MINUS
    the aborted runs, which are counted (n_aborted) and not scored."""
    kept = [r for r in rows if not r.get("aborted")]
    out = PS.score_tables(kept, gate_ok)
    out["n_aborted_not_scored"] = int(len(rows) - len(kept))
    out["aborted_runs"] = [[r["seed"], r["log2rho"]] for r in rows if r.get("aborted")]
    return out


def wide_crossing_rates(rows):
    """DESCRIPTIVE ONLY (never a verdict): crossing by 1.25·s_F per (registered class, rate, wider-search class) over
    the trained, not aborted runs."""
    out = {}
    for r in rows:
        if r.get("aborted"):
            continue
        k = f"{r['class']} {r['log2rho']:g} wide-{r.get('wide_class_DESCRIPTIVE')}"
        b = out.setdefault(k, {"n": 0, "n_crossed": 0, "n_reached_1.25_no_crossing": 0})
        b["n"] += 1
        b["n_crossed"] += int(r["t_obs"] is not None)
        b["n_reached_1.25_no_crossing"] += int(r["t_obs"] is None and bool(r["reached_obs_end"]))
    for b in out.values():
        b["crossing_rate"] = b["n_crossed"] / b["n"] if b["n"] else None
    return {"label": "DESCRIPTIVE: crossing rates by the posthoc wider-search class (not a criterion)", "by": out}


# ------------------------------------------------------------------------------------------ machine rules and io
def memory_gate(tag):
    """free ≥ 25% and swap free ≥ 500 MB (act_fold.check_memory; waits, re-checking every 60 s); disk ≥ 20 GB (else
    STOP).  Every check logged to results/phase2a_ps2/memory_gate.log."""
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
                     f"disk_free={disk / 1024 ** 3:.1f}GB {'OK' if ok and disk >= PS.DISK_MIN_BYTES else 'WAIT'}\n")
        if disk < PS.DISK_MIN_BYTES:
            raise SystemExit(f"STOP: disk free {disk / 1024 ** 3:.1f} GB < 20 GB")
        if ok:
            return f, w
        time.sleep(60)


_setup = P2A._setup
_sha = P2A._sha
_jsonable = P2A._jsonable
_rows = P2A._rows
_append_write = PS._append_write
_peak_rss_gb = PS._peak_rss_gb


# ------------------------------------------------------------------------------------------ seed scan
SCAN_DIRS = PS.SCAN_DIRS
SCAN_SKIP_FILES = ("phase2a_ps2.py", "test_phase2a_ps2.py", "phase2a_ps2_registration.md", "phase2a_ps2_design.md")
SCAN_SKIP_PATHS = ("results/designs/phase2a_ps2_explore/p2_seedscan.py",
                   "results/designs/phase2a_ps2_explore/p2_seedscan.json",
                   "results/designs/phase2a_ps2_explore/README.md")
SCAN_PATTERNS = {
    "registered": r"(^|[^0-9.])2987[0-5][0-9][0-9]([^0-9]|$)|(^|[^0-9])2_987_[0-5][0-9][0-9]([^0-9]|$)"
                  r"|(^|[^0-9.,])2,987,[0-5][0-9][0-9]([^0-9,]|$)",
    "pilot": r"(^|[^0-9.])29880[0-5][0-9]([^0-9]|$)|(^|[^0-9])2_988_0[0-5][0-9]([^0-9]|$)"
             r"|(^|[^0-9.,])2,988,0[0-5][0-9]([^0-9,]|$)"}
SEED_MODULES = PS.SEED_MODULES + ("phase2a_ps", "phase3_w4")


def scan_tree(patterns=SCAN_PATTERNS, dirs=SCAN_DIRS, chunk=16 * 1024 * 1024, overlap=256):
    """phase2a_ps.scan_tree with 2A-PS2's own files and results/phase2a_ps2/ skipped (and the exploration's seed-range
    check files, which name the candidate ranges)."""
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
                rel = str(p.relative_to(ROOT))
                if fn in SCAN_SKIP_FILES or rel in SCAN_SKIP_PATHS or p.suffix in PS.SCAN_SKIP_SUFFIX:
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
                    hits[k].append(rel)
    return hits


def range_literal_overlaps(lo_hi=((SEEDS[0], SEEDS[-1] + 1), (PILOT_SEEDS[0], PILOT_SEEDS[-1] + 1))):
    """Every integer range(...) literal in src/ and tests/ (except 2A-PS2's own files) overlapping a 2A-PS2 range."""
    import re
    rx = re.compile(r"range\(\s*([0-9_]+)\s*(?:,\s*([0-9_]+)\s*)?[,)]")
    over, n = [], 0
    for d in ("src", "tests"):
        for p in sorted((ROOT / d).rglob("*.py")):
            if p.name in SCAN_SKIP_FILES:
                continue
            for m in rx.finditer(p.read_text(errors="ignore")):
                a, b = m.group(1), m.group(2)
                lo, hi = (0, int(a)) if b is None else (int(a), int(b))
                n += 1
                for c0, c1 in lo_hi:
                    if lo < c1 and c0 < hi:
                        over.append([str(p.relative_to(ROOT)), m.group(0)])
    return n, over


def parquet_seed_hits():
    """Every column whose name contains 'seed' in every parquet file under the scan dirs: values in a 2A-PS2 range."""
    import pyarrow.parquet as pqm
    out = []
    for d in SCAN_DIRS:
        for p in sorted((ROOT / d).rglob("*.parquet")):
            t = pqm.read_table(p)
            for c in t.column_names:
                if "seed" in c.lower():
                    col = t.column(c).to_pylist()
                    n = sum(1 for x in col if isinstance(x, (int, float)) and x == x
                            and (SEEDS[0] <= x <= SEEDS[-1] or PILOT_SEEDS[0] <= x <= PILOT_SEEDS[-1]))
                    out.append([str(p.relative_to(ROOT)), c, n])
    return out


def scan():
    """Both 2A-PS2 seed ranges are unused: no text match (digits, '_' and ',' spellings) in any text file under the
    eight trees; no overlap with the SEEDS*/PILOT_SEEDS* constants of the registered modules, nor with any integer
    range literal in src/ and tests/; no hit in any parquet seed column."""
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
    n_lit, lit = range_literal_overlaps()
    pq = parquet_seed_hits()
    out = {"ranges": {"registered": [SEEDS[0], SEEDS[-1], len(SEEDS)],
                      "pilot": [PILOT_SEEDS[0], PILOT_SEEDS[-1], len(PILOT_SEEDS)]}, "dirs": list(SCAN_DIRS),
           "patterns": SCAN_PATTERNS, "pattern_files": hits, "registered_seed_overlap": ov,
           "range_literals_checked": n_lit, "range_literal_overlaps": lit,
           "parquet_seed_columns": pq, "parquet_hits": int(sum(x[2] for x in pq)),
           "exploration_seeds_disjoint": bool(not (mine & set(EXPLORATION_SEEDS))),
           "ps_seeds_disjoint": bool(not (mine & (set(PS.SEEDS) | set(PS.PILOT_SEEDS)))),
           "unused": bool(not any(hits.values()) and not any(ov.values()) and not lit
                          and not any(x[2] for x in pq))}
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("pattern_files", "range_literals_checked",
                                                    "range_literal_overlaps", "parquet_hits", "unused")})))
    if not out["unused"]:
        raise SystemExit("STOP: a 2A-PS2 seed range is not unused")


# ------------------------------------------------------------------------------------------ freeze, pilot, summary
def _freeze_seeds(seeds, path, tag, stop_when=None):
    """Resumable per-seed freeze to `path` (one JSON line per seed, in order); memory gate and RSS guard between seeds."""
    done = {r["seed"] for r in _rows(path)}
    for s in seeds:
        if stop_when is not None and stop_when(_rows(path)):
            break
        if s in done:
            continue
        memory_gate(f"{tag} {s}")
        rec = freeze_seed(s)
        rec["peak_rss_gb"] = round(_peak_rss_gb(), 3)
        _append_write(path, rec)
        e = rec["evaluation"]
        print(json.dumps(_jsonable({"seed": s, "status": e["status"], "class": e["class"], "s_F": rec.get("s_F"),
                                    "rho_t_star": e.get("rho_t_star"), "wide": e.get("wide_class_DESCRIPTIVE"),
                                    "secs": rec.get("secs"), "rss_gb": rec["peak_rss_gb"]})), flush=True)
        P2A.rss_guard()
    return _rows(path)


def freeze():
    """The FREEZE of all 600 registered seeds (no training) -> frozen_parts.jsonl (resumable); then summary()."""
    _setup()
    PS.check_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _freeze_seeds(SEEDS, OUT / "frozen_parts.jsonl", "freeze")
    if {r["seed"] for r in rows} == set(SEEDS):
        summary()


def reevaluate():
    """Re-apply the rules (evaluate) to the stored raw records WITHOUT refreezing (every raw field kept bit for bit);
    then summary()."""
    path = OUT / "frozen_parts.jsonl"
    new = []
    for r in _rows(path):
        raw = {k: v for k, v in r.items() if k != "evaluation"}
        new.append({**raw, "evaluation": _jsonable(evaluate(raw))})
    path.write_text("".join(json.dumps(_jsonable(r)) + "\n" for r in new))
    summary()


def seed_record(rows, seed):
    """A seed's raw record flattened with its evaluation (status, class, Λ_F, budgets, wider-search class)."""
    r = next(x for x in rows if x["seed"] == seed)
    e = r["evaluation"]
    return {**r, "status": e["status"], "class": e["class"], "Lambda_F": e["Lambda_F"], "eligible": e["eligible"],
            "mc_agree": e.get("mc_agree_DESCRIPTIVE"), "budgets": e.get("budgets"),
            "wide_class": e.get("wide_class_DESCRIPTIVE")}


def _dist(x):
    x = [float(v) for v in x if v is not None]
    if not x:
        return None
    return {"n": len(x), "min": min(x), "median": float(np.median(x)), "q90": float(np.percentile(x, 90)),
            "max": max(x)}


def summary():
    """frozen.json: counts (untraceable first), every seed's status and class (listing and its SHA-256), the gate, the
    plan with each run's budget, the author's over-cap check, budget distributions, the activity drops, the stalls, the
    wider-search table (DESCRIPTIVE).  Writes frozen.json, then STOPs (SystemExit) if the over-cap check fails."""
    rows = sorted(_rows(OUT / "frozen_parts.jsonl"), key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == list(SEEDS), "every registered seed frozen exactly once"
    for r in rows:
        assert _jsonable(evaluate(r)) == r["evaluation"], f"evaluation of seed {r['seed']} differs from the stored one"
    pl, surplus = plan(rows)
    listing = class_listing(rows)
    sc = [seed_record(rows, r["seed"]) for r in rows if _st(r) == "scoreable"]
    bud = {}
    for k in ("clean", "none", "mixed"):
        for l2 in RATES[k]:
            key = f"{l2:g}"
            bs = [r["evaluation"]["budgets"][key] for r in rows if _st(r) in ("scoreable", OVER_CAP) and _cl(r) == k]
            bud[f"{k} {key}"] = {"rho_t_star": _dist([r["evaluation"]["rho_t_star"] for r in rows
                                                      if _st(r) in ("scoreable", OVER_CAP) and _cl(r) == k]),
                                 "B": _dist([b["B"] for b in bs if not b["over_cap"]]),
                                 "B_uncapped": _dist([b["B_uncapped"] for b in bs]),
                                 "n_over_cap": int(sum(b["over_cap"] for b in bs)), "cap": step_cap(rho_of(l2))}
    oc = over_cap_check(rows)
    fr = {"label": "Phase 2A-PS2 frozen per-seed inputs (before the registration; no training)",
          **PS.check_inputs(), "frozen_parts_sha256": _sha(OUT / "frozen_parts.jsonl"),
          "counts": counts(rows), "gate": gate(rows), "over_cap_check": oc,
          "class_listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
          "classes": {str(r["seed"]): [_st(r), _cl(r)] for r in rows},
          "plan": pl,
          "plan_counts": {k: int(sum(1 for p in pl if p["class"] == k and (k != "clean" or p["log2rho"] ==
                                                                          PS.LOG2_SCORED))) for k in CAPS},
          "n_runs_planned": len(pl),
          "plan_steps_total": int(sum(p["budget"] for p in pl)),
          "surplus_counted_not_trained": surplus,
          "budgets": bud,
          "activity": {"inactive_unit_seeds": [[r["seed"], _cl(r), r["ps2"]["activity"]["min_share_registered"]]
                                               for r in rows if _st(r) == INACTIVE],
                       "smallest_live_share_scoreable": min((r["ps2"]["activity"]["min_share_registered"]
                                                            for r in rows if _st(r) == "scoreable"), default=None)},
          "stall_seeds": [[r["seed"], _cl(r), r["ps2"]["tstar"].get("stall_scale")] for r in rows if _st(r) == STALL],
          "DESCRIPTIVE_wide_class_by_registered_class_scoreable": wide_table(rows),
          "DESCRIPTIVE_wide_class_by_registered_class_planned": {
              k: {w: int(sum(1 for p in pl if p["class"] == k and p["log2rho"] == RATES[k][-1]
                             and seed_record(rows, p["seed"])["wide_class"] == w)) for w in ("clean", "none", "mixed")}
              for k in CAPS},
          "DESCRIPTIVE": {"s_F_range_scoreable": [min(r["s_F"] for r in sc), max(r["s_F"] for r in sc)] if sc else None,
                          "sF_over_sstar_range_scoreable": ([min(r["sF_over_sstar"] for r in sc),
                                                             max(r["sF_over_sstar"] for r in sc)] if sc else None),
                          "Lambda_F_range_scoreable": ([min(r["Lambda_F"] for r in sc),
                                                        max(r["Lambda_F"] for r in sc)] if sc else None),
                          "mc_agree_scoreable_by_class": {k: int(sum(1 for r in sc if r["class"] == k and r["mc_agree"]))
                                                          for k in ("clean", "none", "mixed")},
                          "freeze_secs_total": round(sum(r.get("secs", 0) for r in rows), 1),
                          "peak_rss_gb": max(r.get("peak_rss_gb", 0) for r in rows),
                          "homotopy_200_used": int(sum(1 for r in rows if r["M_homotopy"]["n_steps"] == 200))},
          "rules": {"THETA": THETA, "N_ACTIVE": N_ACTIVE, "TSTAR_N": TSTAR_N, "TSTAR_END": TSTAR_END,
                    "SLOPE_DS_REL": SLOPE_DS_REL, "K_BUDGET": K_BUDGET, "BUDGET_ADD": BUDGET_ADD,
                    "CAP_FACTOR": CAP_FACTOR, "RSS_RUN_CAP": RSS_RUN_CAP, "STATUS_ORDER": STATUS_ORDER,
                    "CAPS": CAPS, "GATE_MIN_CLEAN": GATE_MIN_CLEAN, "H_MIN_N": H_MIN_N,
                    "OVERCAP_STOP_MAX": OVERCAP_STOP_MAX}}
    (OUT / "frozen.json").write_text(json.dumps(_jsonable(fr), indent=1))
    print(json.dumps(_jsonable({k: fr[k] for k in ("counts", "gate", "over_cap_check", "class_listing_sha256",
                                                    "plan_counts", "n_runs_planned", "plan_steps_total")}), indent=1))
    if not oc["pass"]:
        raise SystemExit("STOP: the author's over-cap check fails (more than 5 scoreable clean or none seeds over cap)")


def pilot():
    """Pilot seeds only (2,988,000-2,988,059), a MACHINE check that fixes no rule, threshold or criterion: freeze pilot
    seeds in order until the first scoreable clean AND the first scoreable none seed; then the clean seed at 2⁻¹⁴ and
    2⁻¹⁶ and the none seed at 2⁻¹⁴ through run_one and observe_one with their frozen budgets B (the hashed-state
    resume, the guarded forecasts, whether the none observation reaches 1.25·s_F within B, time and RSS).
    -> pilot_parts.jsonl, pilot.json"""
    _setup()
    PS.check_inputs()
    OUT.mkdir(parents=True, exist_ok=True)

    def found(rows):
        cl = {_cl(r) for r in rows if _st(r) == "scoreable"}
        return "clean" in cl and "none" in cl
    rows = _freeze_seeds(PILOT_SEEDS, OUT / "pilot_parts.jsonl", "pilot freeze", stop_when=found)
    out_p = OUT / "pilot.json"
    Pj = json.loads(out_p.read_text()) if out_p.exists() else {}
    Pj.update(label="PILOT (pilot seeds only; a machine check: fixes no rule, threshold or criterion)",
              n_frozen=len(rows), counts=counts(rows), freeze_secs=[r.get("secs") for r in rows],
              statuses={str(r["seed"]): [_st(r), _cl(r)] for r in rows})
    Pj.setdefault("runs", {})
    for k in ("clean", "none"):
        pc = [r for r in rows if _st(r) == "scoreable" and _cl(r) == k]
        if not pc:
            continue
        fs = seed_record(rows, pc[0]["seed"])
        for l2 in RATES[k]:
            key = f"{k} {fs['seed']} {l2:g}"
            if key in Pj["runs"]:
                continue
            B = fs["budgets"][f"{l2:g}"]["B"]
            memory_gate(f"pilot run {key}")
            t0 = time.time()
            rr = run_one(fs, l2, B)
            rr["secs_run"] = round(time.time() - t0, 1)
            P2A.rss_guard()
            memory_gate(f"pilot observe {key}")
            t0 = time.time()
            ob, S = observe_one(fs, l2, rr, B)
            ob["secs_observe"] = round(time.time() - t0, 1)
            del S
            rr.pop("state_tc", None)
            Pj["runs"][key] = {**rr, "observed": ob, "rho_t_star": fs["evaluation"]["rho_t_star"],
                               "budget_record": fs["budgets"][f"{l2:g}"], "peak_rss_gb": round(_peak_rss_gb(), 3),
                               "current_rss_gb_after": round(current_rss() / 1e9, 3)}
            out_p.write_text(json.dumps(_jsonable(Pj), indent=1))
            print(json.dumps(_jsonable({"run": key, "B": B, "t_c": rr["t_c"], "status": rr["forecast"]["status"],
                                        "t_obs": ob["t_obs"], "steps": ob["steps_observed"],
                                        "s_end_over_sF": ob["s_end"] / fs["s_F"], "reached_1.25": ob["reached_obs_end"],
                                        "secs": [rr["secs_run"], ob["secs_observe"]], "rss": _peak_rss_gb()})),
                  flush=True)
            P2A.rss_guard()
    out_p.write_text(json.dumps(_jsonable(Pj), indent=1))


# ------------------------------------------------------------------------------------------ registration manifest
FROZEN_DATA = ("results/designs/phase2a_ps2_design.md", "results/phase2a_ps2_registration.md",
               "results/phase2a_ps2/seed_scan.json", "results/phase2a_ps2/frozen.json",
               "results/phase2a_ps2/frozen_parts.jsonl", "results/phase2a_ps2/pilot.json",
               "results/phase2a_ps2/pilot_parts.jsonl", "tests/test_phase2a_ps2.py", "tests/test_phase2a_ps.py",
               "tests/test_causal_forecast.py",
               "results/simplicity_bias_v2/frozen.json", "results/phase2a/frozen.json", "results/phase2a_posthoc.json",
               "results/sb_fold/branches.json",
               "results/sb_fold/branch_M_fwd.npz", "results/sb_fold/branch_M_bwd.npz",
               "results/sb_fold/branch_S_fwd.npz", "results/sb_fold/branch_S_bwd.npz",
               "results/phase2a_ps_registration.md", "results/phase2a_ps/registration.sha256",
               "results/phase2a_ps/frozen_parts.jsonl", "results/phase2a_ps/runs.jsonl",
               "results/phase2a_ps/observed.jsonl", "results/phase2a_ps/scores.json",
               "results/phase2a_ps_posthoc.json",
               "results/designs/phase2a_ps2_explore/README.md", "results/designs/phase2a_ps2_explore/p2_explore.py",
               "results/designs/phase2a_ps2_explore/p2_explore.jsonl",
               "results/designs/phase2a_ps2_explore/p2_summary.py",
               "results/designs/phase2a_ps2_explore/p2_summary.log",
               "results/designs/phase2a_ps2_explore/p2_seedscan.py",
               "results/designs/phase2a_ps2_explore/p2_seedscan.json",
               "results/designs/phase3_explore/p3_activity.py")


def code_closure(start=("phase2a_ps2",)):
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


def stamp():
    """registration_stamp.txt: the registration commit (the last commit touching the manifest) and the SHA-256 of the
    registration file and of the manifest."""
    _assert_committed(OUT / "registration.sha256")
    h = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(OUT / "registration.sha256")], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()
    txt = (f"Phase 2A-PS2 registration\nregistration commit: {h}\n"
           f"sha256 results/phase2a_ps2_registration.md: {_sha(REGISTRATION_MD)}\n"
           f"sha256 results/phase2a_ps2/registration.sha256: {_sha(OUT / 'registration.sha256')}\n")
    (OUT / "registration_stamp.txt").write_text(txt)
    print(txt)


def assert_registration():
    m = OUT / "registration.sha256"
    for ln in m.read_text().splitlines():
        h, rel = ln.split()
        assert _sha(ROOT / rel) == h, f"registration hash mismatch: {rel}"
    _assert_committed(m)
    _assert_committed(OUT / "registration_stamp.txt")


# ------------------------------------------------------------------------------------------ run / finalize / observe / score
FORBIDDEN = PS.FORBIDDEN


def _frozen():
    fr = json.loads((OUT / "frozen.json").read_text())
    assert fr["frozen_parts_sha256"] == _sha(OUT / "frozen_parts.jsonl")
    rows = _rows(OUT / "frozen_parts.jsonl")
    assert hashlib.sha256(class_listing(rows).encode()).hexdigest() == fr["class_listing_sha256"]
    return fr, rows


def run():
    """After the registration commit and its timestamp: every planned (seed, rate) to t_c with both forecasts, each
    with its frozen budget B (resumable per run)."""
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
        assert fs["budgets"][f"{p['log2rho']:g}"]["B"] == p["budget"]
        memory_gate(f"run {p['seed']} {p['log2rho']:g}")
        t0 = time.time()
        rec = run_one(fs, p["log2rho"], p["budget"])
        rec["secs"] = round(time.time() - t0, 1)
        _append_write(rf, rec)
        print(json.dumps({"seed": p["seed"], "class": p["class"], "log2rho": p["log2rho"], "B": p["budget"],
                          "t_c": rec["t_c"], "status": rec["forecast"]["status"], "secs": rec["secs"]}), flush=True)
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
    """Per run (not an aborted one): observation with ρ₂ every step, the t_c state asserted, budget B."""
    assert_registration()
    _assert_forecasts()
    _setup()
    fr, rows = _frozen()
    of = OUT / "observed.jsonl"
    done = {(r["seed"], r["log2rho"]) for r in _rows(of)}
    for rr in _rows(OUT / "runs.jsonl"):
        if (rr["seed"], rr["log2rho"]) in done:
            continue
        if rr.get("aborted"):
            _append_write(of, {"seed": rr["seed"], "class": rr["class"], "log2rho": rr["log2rho"], "aborted": True,
                               "not_observed": "run aborted (per-run RSS cap): counted, not scored"})
            continue
        fs = seed_record(rows, rr["seed"])
        memory_gate(f"observe {rr['seed']} {rr['log2rho']:g}")
        t0 = time.time()
        ob, S = observe_one(fs, rr["log2rho"], rr, rr["budget"])
        ob["secs"] = round(time.time() - t0, 1)
        del S
        _append_write(of, ob)
        print(json.dumps({"seed": rr["seed"], "log2rho": rr["log2rho"], "t_obs": ob["t_obs"], "s_obs": ob["s_obs"],
                          "reached_1.25": ob["reached_obs_end"], "secs": ob["secs"]}), flush=True)
        P2A.rss_guard()


def score():
    assert_registration()
    _assert_forecasts()
    fr, rows = _frozen()
    runs = _rows(OUT / "runs.jsonl")
    obs = {(o["seed"], o["log2rho"]): o for o in _rows(OUT / "observed.jsonl")}
    T = [run_row(seed_record(rows, r["seed"]), r, obs.get((r["seed"], r["log2rho"]), {})) for r in runs]
    ST = score_tables(T, fr["gate"]["pass"])
    out = {"counts_PROMINENT": fr["counts"], "gate": fr["gate"], "over_cap_check": fr["over_cap_check"],
           "outcome": ST["outcome"], "verdicts": ST["verdicts"], "secondary_verdicts": ST.get("secondary_verdicts"),
           "criteria": ST["criteria"], "secondary": ST.get("secondary"), "validity": ST["validity"],
           "n_aborted_not_scored": ST["n_aborted_not_scored"], "aborted_runs": ST["aborted_runs"],
           "seed_exponents": ST.get("seed_exponents"), "rows": T, "DESCRIPTIVE": PS.descriptive(T),
           "DESCRIPTIVE_wide_crossing_rates": wide_crossing_rates(T),
           "DESCRIPTIVE_split_by_mc_agreement": PS.descriptive_split([r for r in T if not r.get("aborted")])
           if fr["gate"]["pass"] else None}
    (OUT / "scores.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("counts_PROMINENT", "gate", "verdicts", "secondary_verdicts",
                                                    "outcome")}), indent=1))


def main(argv):
    cmd = argv[1]
    if cmd == "gate":
        f, w = memory_gate(" ".join(argv[2:]) or "gate")
        print(f"memory gate OK: free {f}% swap_free {w} MB disk_free {shutil.disk_usage(ROOT).free / 1024 ** 3:.1f} GB")
        return
    fns = {"scan": scan, "pilot": pilot, "freeze": freeze, "summary": summary, "reevaluate": reevaluate,
           "manifest": manifest, "stamp": stamp, "run": run, "finalize": finalize, "observe": observe, "score": score}
    if cmd not in fns:
        raise SystemExit(f"unknown command {cmd}")
    fns[cmd]()


if __name__ == "__main__":
    main(sys.argv)
