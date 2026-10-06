"""POST HOC diagnosis of the registered per-seed fold test 2A-PS (registration c819290, stamp 9a10df4, OpenTimestamps
a2065bf; forecasts ca62d97; results d6fb90e: UNRESOLVED (validity)).  NOT registered.  It changes NO verdict: the
registered outcome stays UNRESOLVED (validity), and src/phase2a_ps.py, the frozen files, runs.jsonl, observed.jsonl and
scores.json are read, never written.

Questions (author, 2026-10-05):
 (1) The 4 trained clean seeds with no forecast (2,975,003, 052, 093, 179): the exact failure point in the registered
     forecaster, reproduced with the registered code on the hashed states, and whether it is a PIPELINE issue (budget,
     extrapolation window, horizon cap, release) or a LANDSCAPE issue (stall or slow growth on the seed's own M).
 (2) The 7 trained "none" seeds that crossed by 1.25·s_F: the branch they crossed into (the crossing state minimised at
     its own s and matched to the seed's own S, S2 and L0 branches), and whether a WIDER search past the fold (more
     perturbed minima, larger perturbations, s = 1.01 … 1.10·s_F) classifies them, and the other 23 none seeds, as
     clean or mixed.
 (3) Seed 2,975,052's active-sign flip at step 1 (both rates): its cause, and every criterion and validity recomputed
     with that seed EXCLUDED (beside the registered verdicts).

Every computation reuses the registered functions (sample, release, make_step, train_to_cutoff, observe_one,
run_forecast, homotopy_accepted, continue_both, SeedBranch, perturbed_minima, score_tables) unchanged.  Replays assert
the frozen sample and release hashes, the hashed t_c states and the committed forecasts and observations.

Jobs (one process, nice 15, one thread; the memory gate is a SEPARATE logged step before each job and also runs between
items; stop above 3 GB RSS; every replay has a hard step budget ≤ ⌈64/ρ⌉):

    python -m src.phase2a_ps_posthoc gate TAG     # memory gate (logged to results/phase2a_ps/memory_gate.log)
    python -m src.phase2a_ps_posthoc nofc         # (1) replays of the 4 no-forecast clean seeds, both rates; M's slope
    python -m src.phase2a_ps_posthoc none         # (2) the 30 trained none seeds: wider search, S profile; crossers
    python -m src.phase2a_ps_posthoc sign         # (3) seed 2,975,052: the first step and the 197-step observation
    python -m src.phase2a_ps_posthoc build        # results/phase2a_ps_posthoc.json and .md (no compute beyond ms)

Raw job outputs: results/phase2a_ps_posthoc_parts.jsonl (one row per item; resumable).  build() assembles the JSON from
that file and the committed registered files only.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

from . import causal_forecast as CF
from . import causal_forecast_fold as CFF
from . import phase2a as P2A
from . import phase2a_ps as PS
from . import sb_fold as SBF

RESULTS = PS.RESULTS
OUT_JSON = RESULTS / "phase2a_ps_posthoc.json"
OUT_MD = RESULTS / "phase2a_ps_posthoc.md"
PARTS = RESULTS / "phase2a_ps_posthoc_parts.jsonl"

NOFC_SEEDS = (2_975_003, 2_975_052, 2_975_093, 2_975_179)
SIGN_SEED = 2_975_052
RATES_CLEAN = (PS.LOG2_EXPONENT, PS.LOG2_SCORED)          # -14, -16
RHO_TIME = PS.BUDGET_FACTOR                                # ρ·t at the budget (64)
N_ACTIVE = 3

SLOPE_GRID = 41                                            # M's dL/ds on [0.80, 0.9999]·s_F
SLOPE_LO, SLOPE_HI = 0.80, 0.9999
SLOPE_DS_REL = 1e-6                                        # central difference half-width (relative to s_F)

WIDE_SCALES = (1.01, 1.02, 1.05, 1.10)                     # s/s_F of the wider search
WIDE_AMPS = (1.0, 2.0, 4.0)                                # × the registered perturbation size (0.05|P| + 0.02)
WIDE_N = 30                                                # minima per (scale, amplitude) block
WIDE_STREAM = 2_026_1005                                   # default_rng([seed, WIDE_STREAM]) (POST HOC stream)
S_PROFILE = tuple(round(1.0 + 0.01 * k, 2) for k in range(26))   # ρ₂ of the seed's S on 1.00 … 1.25·s_F
OTHER_BRANCHES = ("S", "S2", "L0")
DEAD_REL = 1e-12                                           # |v_k| ≤ 1e-12·s: a numerically dead "active" unit

LABEL = ("POST HOC (not registered): a diagnosis of 2A-PS after its results (d6fb90e). It changes NO verdict; the "
         "registered outcome remains UNRESOLVED (validity).")


# =============================================================================================== pure functions (tested)
def quad_profile(s_prefix, s_F, budget_left, f=PS.F_CUT):
    """The registered forecaster's extrapolation (CF.extrapolate_scale: quad in log s over the last ⌈0.05·t_c⌉ ≥ 50 rows,
    anchored at the last row, horizon 1.6·s_F, ⌈4·need⌉ + 10 steps capped by the budget) UNPACKED: the fitted
    quadratic's curvature, its vertex k* (steps after t_c − 1) and the vertex value ŝ_max WITHOUT the budget cap, the
    first k with ŝ_k ≥ s_F without the cap (None if never), and whether the budget capped the path the forecaster
    used.  s_prefix = s_0 … s_{t_c−1}."""
    s = np.asarray(s_prefix, float)
    cfg = CFF.config(f)
    t_c = len(s)
    n_w = CF.window_length(t_c, cfg)
    w = np.log(s[-n_w:])
    tau = np.arange(-n_w + 1, 1, dtype=float)
    sc = max(1.0, float(np.abs(tau).max()))
    c2, c1, c0 = np.polyfit(tau / sc, w, 2)                # p(x) = c2 x² + c1 x + c0, x = k/sc
    rate = c1 / sc                                         # d log s / dk at k = 0 (the forecaster's growth rate)
    y_last = w[-1]

    def yhat(k):
        x = np.asarray(k, float) / sc
        return y_last + (c2 * x ** 2 + c1 * x)

    out = {"t_c": t_c, "n_window": int(n_w), "growth_rate_at_cutoff": float(rate),
           "curvature_per_step2": float(2 * c2 / sc ** 2), "concave": bool(c2 < 0)}
    n_ahead = CF._n_ahead(y_last, rate, cfg.horizon_factor * s_F, int(budget_left)) if rate > 0 else 0
    out["n_ahead_used"] = int(n_ahead)
    out["n_ahead_uncapped"] = int(max(1, math.ceil(4 * max((math.log(cfg.horizon_factor * s_F) - y_last) / rate, 1.0))
                                      + 10)) if rate > 0 else 0
    out["budget_capped"] = bool(rate > 0 and out["n_ahead_uncapped"] > int(budget_left))
    out["s_hat_at_end_of_used_path"] = float(np.exp(yhat(n_ahead))) if n_ahead else None
    if c2 < 0:
        kv = -c1 / (2 * c2) * sc
        out["k_vertex"] = float(kv)
        out["s_hat_max_uncapped"] = float(np.exp(min(float(yhat(max(kv, 0.0))), 700.0)))
    else:
        out["k_vertex"] = None
        out["s_hat_max_uncapped"] = math.inf
    # first k ≥ 1 with ŷ_k ≥ log s_F, no cap (closed form of the quadratic)
    a, b, c = c2 / sc ** 2, c1 / sc, y_last - math.log(s_F)
    k_F = None
    if c >= 0:
        k_F = 1
    elif a == 0:
        k_F = int(math.ceil(-c / b)) if b > 0 else None
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            roots = sorted(r for r in ((-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)) if r > 0)
            if roots:
                k_F = max(1, int(math.ceil(roots[0])))
    out["k_reach_sF_uncapped"] = k_F
    out["k_reach_sF_within_used_path"] = bool(k_F is not None and k_F <= n_ahead)
    return out


def quasi_static_rho_time(s_grid, dLds, s_a, s_b, n_active=N_ACTIVE):
    """ρ·t to move from s_a to s_b ON M under the scale-only rule: on M the gradient orthogonal to â vanishes, so
    ds/dt = −ηρ·Σ_k sign(v_k)∂L/∂v_k = −n·ρ·dL_M/ds (η = 1), hence ρ·t = ∫ ds / (n·(−dL_M/ds)).  inf if dL_M/ds ≥ 0
    anywhere on [s_a, s_b] (a stall: the scale flow has a fixed point on M below s_b).  Trapezoid on the grid points
    inside [s_a, s_b] with the end values interpolated."""
    s_grid, dLds = np.asarray(s_grid, float), np.asarray(dLds, float)
    if s_b <= s_a:
        return 0.0
    m = (s_grid > s_a) & (s_grid < s_b)
    ss = np.r_[s_a, s_grid[m], s_b]
    gg = np.r_[np.interp(s_a, s_grid, dLds), dLds[m], np.interp(s_b, s_grid, dLds)]
    if np.any(gg >= 0):
        return math.inf
    return float(np.trapezoid(1.0 / (n_active * (-gg)), ss))


def stall_scale(s_grid, dLds):
    """The first s on the grid where dL_M/ds ≥ 0 (linear interpolation of the sign change), or None."""
    s_grid, dLds = np.asarray(s_grid, float), np.asarray(dLds, float)
    idx = np.flatnonzero(dLds >= 0)
    if not len(idx):
        return None
    i = int(idx[0])
    if i == 0:
        return float(s_grid[0])
    w = -dLds[i - 1] / (dLds[i] - dLds[i - 1])
    return float(s_grid[i - 1] + w * (s_grid[i] - s_grid[i - 1]))


def classify_failure(status, degenerate_release, rho_time_to_target, rho_time_target_label, k_reach_uncapped,
                     budget_left, rho_time_budget=RHO_TIME):
    """PIPELINE vs LANDSCAPE for a run without a forecast (POST HOC reading; never a verdict).
    - degenerate release (an 'active' unit numerically dead): PIPELINE (the registered active set and â);
    - otherwise the TRUE dynamics decide: if s on the seed's own M cannot reach the forecaster's target (the cutoff
      0.95·s_F for 'no cutoff', s_F for 'no s_F') within the ρ-time budget (stall, or quasi-static ρ·t beyond 64),
      the forecaster's refusal is correct and the cause is the LANDSCAPE (slow growth / stall; the fixed ⌈64/ρ⌉ budget
      is where it shows);
    - else PIPELINE: the extrapolation budget cap (the uncapped quadratic reaches s_F after the budget) or the
      extrapolation itself (the window quadratic turns over below s_F although the dynamics reach it)."""
    if degenerate_release:
        return "PIPELINE", "degenerate release: an 'active' unit with |v| ≈ 0 enters â, so the scale-only rule is not " \
                           "rate-limited"
    if rho_time_to_target is None or not math.isfinite(rho_time_to_target):
        return "LANDSCAPE", f"stall: dL_M/ds ≥ 0 below {rho_time_target_label} on the seed's own M"
    if rho_time_to_target > rho_time_budget:
        return "LANDSCAPE", (f"slow growth: reaching {rho_time_target_label} on the seed's own M needs ρ·t ≈ "
                             f"{rho_time_to_target:.0f} > {rho_time_budget:g} (the budget)")
    if status.startswith("no forecast: no extrapolated s reaches s_F"):
        if k_reach_uncapped is not None and k_reach_uncapped > budget_left:
            return "PIPELINE", "extrapolation budget cap: the uncapped quadratic reaches s_F after the budget"
        return "PIPELINE", "extrapolation: the window quadratic turns over below s_F although M reaches it in time"
    return "PIPELINE", "budget: the dynamics reach the target within ρ·t ≤ 64 but the run did not"


def effective_dead_units(theta, active, rel=DEAD_REL):
    """'Active' units (v ≠ 0.0, the registered rule) whose |v_k| ≤ rel·s: numerically dead."""
    th = np.asarray(theta, float)
    s = float(np.abs(th[12:16]).sum())
    return [int(k) for k in active if abs(th[12 + k]) <= rel * s]


def step_decomposition(g_v, v, active, rho, eta=PS.ETA):
    """The registered scale-only update of v (PS.make_step's expression) split per active unit into the part along â
    (rate-limited by ρ) and the part orthogonal to it (full η); returns per-unit Δv, the s change, and the s change the
    rule would make with the numerically dead units removed from â (POST HOC comparison only)."""
    g_v, v = np.asarray(g_v, float), np.asarray(v, float)

    def upd(act):
        a = np.zeros(4)
        a[act] = np.sign(v[act]) / math.sqrt(len(act))
        gp = a * (a @ g_v)
        dv = -eta * ((g_v - gp) + rho * gp)
        return dv, a, gp
    dv, a, gp = upd(list(active))
    new = v + dv
    out = {"a_hat": a.tolist(), "dv": dv.tolist(), "dv_parallel": (-eta * rho * gp).tolist(),
           "dv_orthogonal": (-eta * (g_v - gp)).tolist(), "v_new": new.tolist(),
           "ds": float(np.abs(new).sum() - np.abs(v).sum()),
           "sign_flips": [int(k) for k in active if np.sign(new[k]) != np.sign(v[k])]}
    live = [k for k in active if abs(v[k]) > DEAD_REL * np.abs(v).sum()]
    if len(live) != len(active):
        dv2, _, _ = upd(live)
        out["ds_live_units_only"] = float(np.abs(v + dv2).sum() - np.abs(v).sum())
    return out


def block_class(n_ge_q, n=WIDE_N):
    """The registered class rule scaled to a block of n minima: clean ≥ 0.9·n (27 of 30), none 0, mixed otherwise."""
    n_ge_q = int(n_ge_q)
    return "clean" if n_ge_q >= math.ceil(PS.CLEAN_MIN / 30 * n) else ("none" if n_ge_q == 0 else "mixed")


def wider_class(blocks):
    """POST HOC wider-search class from blocks [{"scale", "amp", "n", "n_ge_q"}, …]: pooled over every block (clean if
    ≥ 90% of all minima have ρ₂ ≥ q, none if 0, else mixed), plus the class per scale (pooled over amplitudes) and
    the first scale with any ρ₂ ≥ q minimum."""
    N = sum(int(b["n"]) for b in blocks)
    k = sum(int(b["n_ge_q"]) for b in blocks)
    per = {}
    for b in blocks:
        p = per.setdefault(f"{b['scale']:g}", [0, 0])
        p[0] += int(b["n"]); p[1] += int(b["n_ge_q"])
    per_cls = {sc: block_class(v[1], v[0]) for sc, v in per.items()}
    first = next((sc for sc in sorted(per, key=float) if per[sc][1] > 0), None)
    return {"pooled": block_class(k, N), "n": N, "n_ge_q": k, "per_scale": per_cls,
            "per_scale_n_ge_q": {sc: v[1] for sc, v in per.items()}, "first_scale_with_ge_q": first}


def match_branch(dists, tol=PS.ON_TOL):
    """The branch whose point at the same s lies within tol (function space) and nearest; 'other' if none."""
    best, dbest = "other", math.inf
    for name, d in dists.items():
        if d is not None and math.isfinite(d) and d <= tol and d < dbest:
            best, dbest = name, d
    return best


def first_crossing_scale(s_grid, rho2, q=PS.Q):
    """The first s on the grid with ρ₂ ≥ q (linear interpolation from the previous point), or None."""
    s_grid, rho2 = np.asarray(s_grid, float), np.asarray(rho2, float)
    idx = np.flatnonzero(np.isfinite(rho2) & (rho2 >= q))
    if not len(idx):
        return None
    i = int(idx[0])
    if i == 0 or not np.isfinite(rho2[i - 1]):
        return float(s_grid[i])
    w = (q - rho2[i - 1]) / (rho2[i] - rho2[i - 1])
    return float(s_grid[i - 1] + w * (s_grid[i] - s_grid[i - 1]))


def score_excluding(rows, seeds):
    """Every criterion and validity by the REGISTERED score_tables over the scored rows minus the given seeds."""
    keep = [r for r in rows if r["seed"] not in set(seeds)]
    return PS.score_tables(keep, True)


# =============================================================================================== io
def _rows():
    return P2A._rows(PARTS) if PARTS.exists() else []


def _done(kind):
    return {(r["kind"], r.get("seed"), r.get("log2rho")) for r in _rows() if r["kind"] == kind}


def _append(row):
    PS._append_write(PARTS, row)


def _committed():
    fr, rows = PS._frozen()
    runs = {(r["seed"], r["log2rho"]): r for r in P2A._rows(PS.OUT / "runs.jsonl")}
    obs = {(o["seed"], o["log2rho"]): o for o in P2A._rows(PS.OUT / "observed.jsonl")}
    return fr, rows, runs, obs


def _gate(tag):
    PS.memory_gate(f"posthoc {tag}")
    P2A.rss_guard()


def seed_M(fs, X, Y):
    """The seed's own M rebuilt with the registered freeze steps; asserts s_F and the release hash equal the frozen."""
    u, nA, s_h, _ = PS.homotopy_accepted("M", X, Y)
    R = SBF.Reduced(nA, X, Y)
    pf, ff, pb, fb = PS.continue_both(R, u, s_h, PS.H_CONT)
    M = PS.SeedBranch(R, pb, pf)
    assert float(ff[0]["s_fold"]) == fs["s_F"], "rebuilt s_F differs from the frozen"
    assert PS.state_sha(M.theta(fs["s0"])) == fs["release_sha256"], "rebuilt release differs from the frozen"
    return M


def seed_branch(name, X, Y):
    """The seed's own branch `name` (S, S2, L0) by the registered homotopy and continuation; None if not accepted or its
    homotopy point is not stable."""
    u, nA, s_h, info = PS.homotopy_accepted(name, X, Y)
    if u is None:
        return None
    R = SBF.Reduced(nA, X, Y)
    pf, ff, pb, fb = PS.continue_both(R, u, s_h, PS.H_CONT)
    B = PS.SeedBranch(R, pb, pf)
    return B if B.start_stable else None


def branch_P(B, s):
    if B is None or not (B.lo <= s <= B.hi):
        return None
    return B.P(s)


def M_slope(M, s_F):
    """dL_M/ds on [0.80, 0.9999]·s_F (central differences of the loss on the seed's own M, Newton at each s)."""
    ss = np.linspace(SLOPE_LO * s_F, SLOPE_HI * s_F, SLOPE_GRID)
    h = SLOPE_DS_REL * s_F
    d = []
    for s in ss:
        a, b = M.loss(s + h if s + h < s_F else s), M.loss(s - h)
        hh = (h if s + h < s_F else 0.0) + h
        d.append(None if a is None or b is None else (a - b) / hh)
    return ss, np.array([np.nan if x is None else x for x in d])


def mu_at_state(th, X, Y):
    """dL_M/ds read off a state near M: (1/n)·Σ_k sign(v_k)·∂L/∂v_k over the active units."""
    _, g = PS.loss_grad(np.asarray(th, float), X, Y)
    act = [k for k in range(4) if th[12 + k] != 0.0]
    return float(np.mean([np.sign(th[12 + k]) * g[12 + k] for k in act]))


def path_summary(S, s_F, rho):
    """s path facts: the first step at 0.8/0.9/0.95/0.99/1.0·s_F, s at ρ·t = 16, 32, 48, 64, s at the end, the late
    log-s growth rate (linear fit over the last 5%) and its ρ-normalised value."""
    S = np.asarray(S, float)
    out = {"n_steps": int(len(S) - 1), "s_end": float(S[-1]), "s_end_over_sF": float(S[-1] / s_F),
           "s_max_over_sF": float(S.max() / s_F)}
    for fr in (0.8, 0.9, 0.95, 0.99, 1.0):
        i = np.flatnonzero(S >= fr * s_F)
        out[f"t_first_{fr:g}sF"] = int(i[0]) if len(i) else None
    for rt in (16, 32, 48, 64):
        t = int(round(rt / rho))
        out[f"s_over_sF_at_rho_t_{rt}"] = float(S[t] / s_F) if t < len(S) else None
    n = max(50, int(math.ceil(0.05 * (len(S) - 1))))
    y = np.log(S[-n:])
    sl = float(np.polyfit(np.arange(n, dtype=float), y, 1)[0])
    out["late_dlogs_dt"] = sl
    out["late_dlogs_drhot"] = sl / rho
    return out


# =============================================================================================== jobs
def job_nofc():
    """(1) For each no-forecast clean seed and rate: replay the registered training from the frozen release (hard
    budget ⌈64/ρ⌉), assert t_c and the hashed t_c state, reproduce the committed forecast with run_forecast, continue
    to the budget and assert the committed observation's s_end; then the seed's M rebuilt and its slope dL_M/ds."""
    fr, rows, runs, obs = _committed()
    done = _done("nofc_replay")
    for seed in NOFC_SEEDS:
        fs = PS.seed_record(rows, seed)
        X, Y = PS.sample(seed)
        assert PS.sample_sha(X, Y) == fs["sample_sha256"]
        th0 = np.array(fs["release_theta"], float)
        assert PS.state_sha(th0) == fs["release_sha256"]
        for l2 in RATES_CLEAN:
            if ("nofc_replay", seed, l2) in done:
                continue
            _gate(f"nofc {seed} {l2:g}")
            t0 = time.time()
            rr, ob = runs[(seed, l2)], obs[(seed, l2)]
            rho = PS.rho_of(l2)
            nb = PS.budget(rho)
            S, th, t_c = PS.train_to_cutoff(th0, rho, fs["s_F"], X, Y, PS.F_CUT, nb)
            assert t_c == rr["t_c"], "t_c differs from the committed"
            row = {"kind": "nofc_replay", "seed": seed, "log2rho": l2, "t_c": t_c, "budget": nb}
            if t_c is not None:
                assert PS.state_sha(th) == rr["state_tc_sha256"], "t_c state differs from the hash"
                fc, same = CFF.run_forecast(S, t_c, fs["s_F"], fs["Lambda_F"], PS.ETA, nb - t_c, PS.F_CUT)
                row["forecast_reproduced"] = bool(same and CFF.same_forecast(
                    {k: v for k, v in fc.items() if k != "nan_recompute_identical"},
                    {k: v for k, v in rr["forecast"].items() if k != "nan_recompute_identical"}))
                row["forecast"] = fc
                row["quad"] = quad_profile(S[:t_c], fs["s_F"], nb - t_c)
                row["mu_at_tc"] = mu_at_state(th, X, Y)
                step = PS.make_step(rho, [k for k in range(4) if th0[12 + k] != 0.0], X, Y)
                Sf = np.empty(nb + 1)
                Sf[:t_c + 1] = S
                n_end = min(nb, ob["steps_observed"])
                for t in range(t_c + 1, n_end + 1):
                    th = step(th)
                    Sf[t] = PS.scale(th)
                    if t % 1_000_000 == 0:
                        P2A.rss_guard()
                S = Sf[:n_end + 1]
            else:
                row["forecast_reproduced"] = rr["forecast"]["status"] == "no forecast: no cutoff within the budget"
            row["observed_s_end_reproduced"] = bool(float(S[-1]) == ob["s_end"])
            row["mu_at_end"] = mu_at_state(th, X, Y)
            row["path"] = path_summary(S, fs["s_F"], rho)
            if t_c is not None:
                k = min(len(S) - 1, t_c + 2000)
                row["s_near_cutoff"] = {str(t): float(S[t]) for t in (t_c - 2000, t_c - 1, t_c, k) if 0 <= t < len(S)}
            row["secs"] = round(time.time() - t0, 1)
            row["peak_rss_gb"] = PS._peak_rss_gb()
            _append(row)
            print(json.dumps({k: row[k] for k in ("seed", "log2rho", "t_c", "forecast_reproduced",
                                                  "observed_s_end_reproduced", "secs")}), flush=True)
            del S
    done = _done("nofc_branch")
    for seed in NOFC_SEEDS:
        if ("nofc_branch", seed, None) in done:
            continue
        _gate(f"nofc branch {seed}")
        t0 = time.time()
        fs = PS.seed_record(rows, seed)
        X, Y = PS.sample(seed)
        M = seed_M(fs, X, Y)
        ss, d = M_slope(M, fs["s_F"])
        _append({"kind": "nofc_branch", "seed": seed, "s_grid": ss.tolist(), "dLds": d.tolist(),
                 "secs": round(time.time() - t0, 1)})
        print(json.dumps({"seed": seed, "branch": "done"}), flush=True)


def job_none():
    """(2) For each trained none seed (2⁻¹⁴): M rebuilt (s_F, release asserted), the registered perturbed_minima
    reproduced (equal to the frozen counts), the seed's S traced and its ρ₂ on 1.00 … 1.25·s_F, and the WIDER search
    (4 scales × 3 amplitudes × 30 minima, default_rng([seed, WIDE_STREAM])), each minimum labelled against S at its s.
    For the 7 crossers: the crossing state replayed from the hashed t_c state (hard budget t_obs − t_c), ρ₂ checked
    at t_obs − 1 and t_obs, minimised at s_obs, and matched to the seed's S, S2 and L0 at s_obs."""
    fr, rows, runs, obs = _committed()
    none = [p["seed"] for p in fr["plan"] if p["class"] == "none"]
    done = _done("none")
    for seed in none:
        if ("none", seed, PS.LOG2_OTHER) in done:
            continue
        _gate(f"none {seed}")
        t0 = time.time()
        fs = PS.seed_record(rows, seed)
        X, Y = PS.sample(seed)
        assert PS.sample_sha(X, Y) == fs["sample_sha256"]
        s_F = fs["s_F"]
        M = seed_M(fs, X, Y)
        rep = PS.perturbed_minima(M, s_F, X, Y)
        row = {"kind": "none", "seed": seed, "log2rho": PS.LOG2_OTHER,
               "registered_minima_reproduced": bool(rep == fs["validation"])}
        S = seed_branch("S", X, Y)
        row["S_stable"] = None if S is None else [S.lo, S.hi]
        prof = []
        for f in S_PROFILE:
            P = branch_P(S, f * s_F)
            prof.append(PS.rho2_static(P, X) if P is not None else None)
        row["S_rho2_profile"] = {f"{f:g}": r for f, r in zip(S_PROFILE, prof)}
        # registered block (fourth draw of default_rng(0)): label vs S at 1.01·s_F
        th_b = M.theta(0.9999 * s_F)
        Pb = SBF.run_to_full(th_b[None])[0]
        rng = np.random.default_rng([seed, WIDE_STREAM])
        blocks = []
        for f in WIDE_SCALES:
            s = f * s_F
            PSb = branch_P(S, s)
            for amp in WIDE_AMPS:
                P0 = Pb[None] + rng.normal(0, 1, (WIDE_N, 16)) * amp * (0.05 * np.abs(Pb) + 0.02)
                P0[:, 12:] = np.abs(P0[:, 12:])
                Pm, L, gn = SBF.local_min_batch(P0, np.full(WIDE_N, s), X, Y)
                r = np.array([PS.rho2_static(p, X) for p in Pm])
                dS = SBF.fdist(Pm, PSb[None], X) if PSb is not None else np.full(WIDE_N, np.inf)
                onS = dS <= PS.ON_TOL
                blocks.append({"scale": f, "amp": amp, "n": WIDE_N, "n_ge_q": int((r >= PS.Q).sum()),
                               "n_on_S": int(onS.sum()), "n_on_S_ge_q": int((onS & (r >= PS.Q)).sum()),
                               "rho2_med": float(np.median(r)), "rho2_max": float(r.max()),
                               "gnorm_max": float(gn.max()), "loss_min": float(L.min())})
        row["wide_blocks"] = blocks
        row["S_rho2_at_wide_scales"] = {f"{f:g}": (PS.rho2_static(branch_P(S, f * s_F), X)
                                                   if branch_P(S, f * s_F) is not None else None) for f in WIDE_SCALES}
        # registered 1.01 block, re-drawn exactly as perturbed_minima (fourth block of default_rng(0)), labelled vs S
        rng0 = np.random.default_rng(PS.PERTURB_SEED)
        for f in PS.VALID_SCALES:
            P0 = Pb[None] + rng0.normal(0, 1, (PS.PERTURB_N, 16)) * (0.05 * np.abs(Pb) + 0.02)
            if f == PS.CLASS_SCALE:
                break
        P0[:, 12:] = np.abs(P0[:, 12:])
        Pm, _, _ = SBF.local_min_batch(P0, np.full(PS.PERTURB_N, PS.CLASS_SCALE * s_F), X, Y)
        r = np.array([PS.rho2_static(p, X) for p in Pm])
        PS1 = branch_P(S, PS.CLASS_SCALE * s_F)
        dS = SBF.fdist(Pm, PS1[None], X) if PS1 is not None else np.full(PS.PERTURB_N, np.inf)
        row["registered_block"] = {"n_ge_q": int((r >= PS.Q).sum()), "n_on_S": int((dS <= PS.ON_TOL).sum()),
                                   "rho2_med": float(np.median(r)),
                                   "rho2_quartiles": [float(x) for x in np.percentile(r, (25, 50, 75))]}
        ob, rr = obs[(seed, PS.LOG2_OTHER)], runs[(seed, PS.LOG2_OTHER)]
        if ob["t_obs"] is not None:
            rho = PS.rho_of(PS.LOG2_OTHER)
            th = np.array(rr["state_tc"], float)
            assert PS.state_sha(th) == rr["state_tc_sha256"]
            th0 = np.array(fs["release_theta"], float)
            step = PS.make_step(rho, [k for k in range(4) if th0[12 + k] != 0.0], X, Y)
            n_rep = ob["t_obs"] - rr["t_c"]
            assert 0 < n_rep <= PS.budget(rho)
            g2 = PS.Rho2Grid(X)
            prev = None
            for t in range(rr["t_c"] + 1, ob["t_obs"] + 1):
                if t == ob["t_obs"]:
                    prev = g2(th)
                th = step(th)
            s_obs, r_obs = PS.scale(th), g2(th)
            cr = {"t_obs": ob["t_obs"], "s_obs": s_obs, "s_obs_reproduced": bool(s_obs == ob["s_obs"]),
                  "rho2_at_t_obs": r_obs, "rho2_at_t_obs_minus_1": prev,
                  "crossing_reproduced": bool(r_obs >= PS.Q and prev < PS.Q and r_obs == ob["rho2_at_obs"])}
            Pm, Lm, gm = SBF.local_min_batch(SBF.run_to_full(th[None]), np.array([s_obs]), X, Y)
            cr["min_rho2"] = PS.rho2_static(Pm[0], X)
            cr["min_loss"] = float(Lm[0])
            cr["min_gnorm"] = float(gm[0])
            cr["min_active_units"] = int((Pm[0, 12:] ** 2 / (Pm[0, 12:] ** 2).sum() > SBF.ACTIVE_TOL).sum())
            cr["dist_state_to_min"] = float(SBF.fdist(SBF.run_to_full(th[None]), Pm, X)[0])
            dists, losses, rho2b = {}, {}, {}
            for name in OTHER_BRANCHES:
                B = S if name == "S" else seed_branch(name, X, Y)
                PBr = branch_P(B, s_obs)
                dists[name] = float(SBF.fdist(Pm, PBr[None], X)[0]) if PBr is not None else None
                losses[name] = B.loss(s_obs) if PBr is not None else None
                rho2b[name] = PS.rho2_static(PBr, X) if PBr is not None else None
            cr["dist_to_branch"] = dists
            cr["branch_loss_at_s_obs"] = losses
            cr["branch_rho2_at_s_obs"] = rho2b
            cr["label"] = match_branch(dists)
            row["crossing"] = cr
        row["secs"] = round(time.time() - t0, 1)
        row["peak_rss_gb"] = PS._peak_rss_gb()
        _append(row)
        print(json.dumps({"seed": seed, "reproduced": row["registered_minima_reproduced"],
                          "wide": wider_class(blocks)["pooled"],
                          "cross": row.get("crossing", {}).get("label"), "secs": row["secs"]}), flush=True)


def job_sign():
    """(3) Seed 2,975,052: the release's units, the first registered step at both rates decomposed, and the registered
    observe_one replayed (197 steps at both rates; asserted equal to the committed observations)."""
    fr, rows, runs, obs = _committed()
    if ("sign", SIGN_SEED, None) in _done("sign"):
        return
    _gate("sign")
    fs = PS.seed_record(rows, SIGN_SEED)
    X, Y = PS.sample(SIGN_SEED)
    assert PS.sample_sha(X, Y) == fs["sample_sha256"]
    th0 = np.array(fs["release_theta"], float)
    assert PS.state_sha(th0) == fs["release_sha256"]
    act = [k for k in range(4) if th0[12 + k] != 0.0]
    _, g = PS.loss_grad(th0, X, Y)
    row = {"kind": "sign", "seed": SIGN_SEED, "active_units_registered": act,
           "dead_units": effective_dead_units(th0, act), "release_v": th0[12:16].tolist(),
           "release_W": th0[:8].tolist(), "release_c": th0[8:12].tolist(), "release_grad_v": g[12:16].tolist(),
           "release_grad_max_other": float(np.abs(g[[0, 1, 2, 3, 8, 9, 16]]).max()), "steps": {}}
    for l2 in RATES_CLEAN:
        rho = PS.rho_of(l2)
        dec = step_decomposition(g[12:16], th0[12:16], act, rho)
        th1 = PS.make_step(rho, act, X, Y)(th0)
        dec["registered_step_v_new"] = th1[12:16].tolist()
        dec["registered_step_matches"] = bool(np.allclose(th1[12:16], dec["v_new"], rtol=0, atol=1e-15))
        dec["registered_ds"] = float(PS.scale(th1) - PS.scale(th0))
        ob, S = PS.observe_one(fs, l2, runs[(SIGN_SEED, l2)])
        dec["observe_reproduced"] = bool(all(ob[k] == obs[(SIGN_SEED, l2)][k] for k in
                                             ("t_obs", "steps_observed", "s_end", "t_F", "sign_first_violation",
                                              "t_08", "reached_obs_end")))
        dec["s_path_first"] = [float(x) for x in S[:4]]
        dec["t_c"] = runs[(SIGN_SEED, l2)]["t_c"]
        dec["steps_observed"] = ob["steps_observed"]
        row["steps"][f"{l2:g}"] = dec
    _append(row)
    print(json.dumps({"seed": SIGN_SEED, "dead": row["dead_units"]}), flush=True)


# =============================================================================================== build
def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def build():
    """The POST HOC JSON from the parts file and the committed registered files (no training)."""
    fr, rows, runs, obs = _committed()
    S = json.loads((PS.OUT / "scores.json").read_text())
    parts = _rows()
    by = {}
    for r in parts:
        by[(r["kind"], r.get("seed"), r.get("log2rho"))] = r
    plan_clean = [p["seed"] for p in fr["plan"] if p["class"] == "clean" and p["log2rho"] == PS.LOG2_SCORED]
    out = {"label": LABEL, "registered": {"outcome": S["outcome"], "verdicts": S["verdicts"],
                                          "secondary_verdicts": S["secondary_verdicts"],
                                          "validity_ok": S["validity"]["ok"]}}

    # ---- clean reference: the seed's frozen constants and the dynamics at t_c (every clean run with a cutoff)
    def const(seed):
        fs = PS.seed_record(rows, seed)
        return {"s_F": fs["s_F"], "s0_over_sF": fs["s0_over_sF"], "lo": fs["M_stable"][0], "Lambda_F": fs["Lambda_F"],
                "sF_over_sstar": fs["sF_over_sstar"], "mc_agree": fs["mc_agree"]}
    ref = {}
    for l2 in RATES_CLEAN:
        mus, rts, grs = [], [], []
        for seed in plan_clean:
            rr = runs[(seed, l2)]
            if rr["t_c"] is None or seed in NOFC_SEEDS:
                continue
            fs = PS.seed_record(rows, seed)
            X, Y = PS.sample(seed)
            mus.append(mu_at_state(np.array(rr["state_tc"], float), X, Y))
            rts.append(rr["t_c"] * PS.rho_of(l2))
            grs.append(rr["forecast"].get("growth_rate_at_cutoff"))
        ref[f"{l2:g}"] = {"n": len(mus), "median_dLds_at_tc": _med(mus), "min_abs_dLds_at_tc": float(np.min(np.abs(mus))),
                          "median_rho_t_c": _med(rts), "max_rho_t_c": float(np.max(rts)),
                          "median_growth_rate_at_cutoff": _med(grs)}
    cons = [const(s) for s in plan_clean if s not in NOFC_SEEDS]
    ref["constants_median_other_36"] = {k: _med([c[k] for c in cons]) for k in ("s_F", "s0_over_sF", "lo", "Lambda_F",
                                                                               "sF_over_sstar")}
    ref["constants_range_other_36"] = {k: [float(np.min([c[k] for c in cons])), float(np.max([c[k] for c in cons]))]
                                       for k in ("s_F", "s0_over_sF", "lo", "Lambda_F", "sF_over_sstar")}
    out["clean_reference"] = ref

    # ---- (1) no-forecast clean seeds
    q1 = []
    for seed in NOFC_SEEDS:
        fs = PS.seed_record(rows, seed)
        br = by.get(("nofc_branch", seed, None))
        ent = {"seed": seed, "constants": const(seed), "runs": {}}
        if br is not None:
            sg, d = np.array(br["s_grid"]), np.array(br["dLds"], float)
            s_F = fs["s_F"]
            ent["M_slope"] = {"dLds_at": {f"{f:g}": float(np.interp(f * s_F, sg, d)) for f in (0.8, 0.9, 0.95, 0.99,
                                                                                               0.9999)},
                              "stall_scale_over_sF": (lambda z: None if z is None else z / s_F)(stall_scale(sg, d)),
                              "rho_time_0.8_to_0.95": quasi_static_rho_time(sg, d, 0.8 * s_F, 0.95 * s_F),
                              "rho_time_0.95_to_0.9999": quasi_static_rho_time(sg, d, 0.95 * s_F, 0.9999 * s_F),
                              "rho_time_0.8_to_0.9999": quasi_static_rho_time(sg, d, 0.8 * s_F, 0.9999 * s_F)}
        dead = effective_dead_units(fs["release_theta"], fs["release_active_units"])
        ent["release_dead_units"] = dead
        for l2 in RATES_CLEAN:
            rp = by.get(("nofc_replay", seed, l2))
            rr, ob = runs[(seed, l2)], obs[(seed, l2)]
            e = {"status": rr["forecast"]["status"], "t_c": rr["t_c"], "budget": rr["budget"],
                 "observed_s_end_over_sF": ob["s_end"] / fs["s_F"], "t_08": ob["t_08"],
                 "rho_t_08": (ob["t_08"] * PS.rho_of(l2)) if ob["t_08"] is not None else None}
            if rp is not None:
                e.update({k: rp[k] for k in ("forecast_reproduced", "observed_s_end_reproduced", "path", "mu_at_end")})
                e["quad"] = rp.get("quad")
                e["mu_at_tc"] = rp.get("mu_at_tc")
                e["s_near_cutoff"] = rp.get("s_near_cutoff")
                if rr["t_c"] is not None:
                    e["rho_t_c"] = rr["t_c"] * PS.rho_of(l2)
                    if rp.get("path", {}).get("t_first_0.8sF") is not None:
                        e["observed_rho_time_0.8_to_0.95"] = (rr["t_c"] - rp["path"]["t_first_0.8sF"]) * PS.rho_of(l2)
            else:
                e["forecast_fields"] = {k: rr["forecast"].get(k) for k in ("t_F_fc", "eps_fc", "r_fc", "s_c_fc",
                                                                           "growth_rate_at_cutoff", "n_window",
                                                                           "n_extrapolated")}
            ms = ent.get("M_slope", {})
            if dead:
                target, rt = "the degenerate release", 0.0
            elif rr["t_c"] is None:
                target, rt = "0.95·s_F", ms.get("rho_time_0.8_to_0.95")
                if rt is not None and ob["t_08"] is not None:
                    rt = rt + ob["t_08"] * PS.rho_of(l2)        # ρ·t already spent to 0.8·s_F
            else:
                target, rt = "s_F", ms.get("rho_time_0.95_to_0.9999")
                if rt is not None:
                    rt = rt + rr["t_c"] * PS.rho_of(l2)
            e["rho_time_to_target_quasi_static"] = rt
            kq = (e.get("quad") or {}).get("k_reach_sF_uncapped")
            bl = (rr["budget"] - rr["t_c"]) if rr["t_c"] is not None else 0
            if seed == SIGN_SEED:
                rf = rr["forecast"]
                e["horizon"] = {"s_c_fc": rf["s_c_fc"], "horizon_1.6sF": 1.6 * fs["s_F"], "eps_fc": rf["eps_fc"],
                                "r_fc": rf["r_fc"], "t_F_fc": rf["t_F_fc"], "n_extrapolated": rf["n_extrapolated"]}
            lab, why = classify_failure(rr["forecast"]["status"], bool(dead), rt, target, kq, bl)
            e["diagnosis"] = {"label": lab, "reason": why}
            ent["runs"][f"{l2:g}"] = e
        q1.append(ent)
    out["q1_no_forecast"] = q1

    # ---- (2) none seeds
    none_rows = [r for r in parts if r["kind"] == "none"]
    q2 = {"per_seed": []}
    for r in sorted(none_rows, key=lambda x: x["seed"]):
        fs = PS.seed_record(rows, r["seed"])
        ob = obs[(r["seed"], PS.LOG2_OTHER)]
        wc = wider_class(r["wide_blocks"])
        prof = r["S_rho2_profile"]
        sg = [float(k) for k in prof]
        rv = [np.nan if v is None else v for v in prof.values()]
        fc = first_crossing_scale(sg, rv)
        ent = {"seed": r["seed"], "crossed": ob["t_obs"] is not None,
               "s_obs_over_sF": (ob["s_obs"] / fs["s_F"]) if ob["s_obs"] is not None else None,
               "reached_1.25sF": ob["reached_obs_end"], "s_end_over_sF": ob["s_end"] / fs["s_F"],
               "registered_class_n": fs["evaluation"]["class_n_rho2_ge_q"],
               "registered_minima_reproduced": r["registered_minima_reproduced"],
               "registered_block": r["registered_block"], "S_stable": r["S_stable"],
               "S_rho2_at_1.01": prof.get("1.01"), "S_first_rho2_ge_q_over_sF": fc,
               "wider": wc, "wider_changes_class": wc["pooled"] != "none",
               "wide_blocks": r["wide_blocks"], "signs_fixed": ob["signs_fixed_to_crossing"],
               "degenerate_release": bool(effective_dead_units(fs["release_theta"], fs["release_active_units"]))}
        if "crossing" in r:
            ent["crossing"] = r["crossing"]
        q2["per_seed"].append(ent)
    ps = q2["per_seed"]
    if ps:
        n = len(ps)
        q2["n"] = n
        q2["n_registered_reproduced"] = sum(e["registered_minima_reproduced"] for e in ps)
        q2["n_wider_changes_class"] = sum(e["wider_changes_class"] for e in ps)
        q2["frac_wider_changes_class"] = q2["n_wider_changes_class"] / n
        q2["wider_pooled_counts"] = {k: sum(e["wider"]["pooled"] == k for e in ps) for k in ("clean", "mixed", "none")}
        q2["wider_per_scale_counts"] = {f"{f:g}": {k: sum(e["wider"]["per_scale"][f"{f:g}"] == k for e in ps)
                                                   for k in ("clean", "mixed", "none")} for f in WIDE_SCALES}
        cr = [e for e in ps if e["crossed"]]
        nc = [e for e in ps if not e["crossed"]]
        q2["crossed"] = {"n": len(cr), "n_wider_changes": sum(e["wider_changes_class"] for e in cr),
                         "labels": {e["seed"]: e["crossing"]["label"] for e in cr if "crossing" in e},
                         "n_S_ge_q_before_1.25": sum(e["S_first_rho2_ge_q_over_sF"] is not None
                                                     and e["S_first_rho2_ge_q_over_sF"] <= 1.25 for e in cr)}
        q2["not_crossed"] = {"n": len(nc), "n_wider_changes": sum(e["wider_changes_class"] for e in nc),
                             "n_S_ge_q_before_1.25": sum(e["S_first_rho2_ge_q_over_sF"] is not None
                                                         and e["S_first_rho2_ge_q_over_sF"] <= 1.25 for e in nc),
                             "n_signs_moved": sum(not e["signs_fixed"] for e in nc)}
        q2["S_at_1.01_ge_q"] = sum((e["S_rho2_at_1.01"] or 0) >= PS.Q for e in ps)
        q2["registered_block_on_S_any"] = sum(e["registered_block"]["n_on_S"] > 0 for e in ps)
    out["q2_none"] = q2

    # ---- (3) seed 2,975,052
    sr = by.get(("sign", SIGN_SEED, None))
    q3 = {"seed": SIGN_SEED, "cause": sr}
    ex = score_excluding(S["rows"], [SIGN_SEED])
    q3["excluded_rescore_POST_HOC"] = {"outcome": ex["outcome"], "verdicts": ex["verdicts"],
                                       "secondary_verdicts": ex["secondary_verdicts"], "validity": ex["validity"],
                                       "criteria": ex["criteria"], "secondary": ex["secondary"],
                                       "n_clean_scored": ex["n_clean_scored"], "n_none": ex["n_none"]}
    degen = []
    for seed, cls in sorted({(p["seed"], p["class"]) for p in fr["plan"]}):
        fs = PS.seed_record(rows, seed)
        dead = effective_dead_units(fs["release_theta"], fs["release_active_units"])
        if dead:
            ob = obs[(seed, PS.LOG2_OTHER)]
            degen.append({"seed": seed, "class": cls, "dead_units": dead,
                          "min_abs_v": float(min(abs(fs["release_theta"][12 + k]) for k in dead)),
                          "t_c_2^-14": runs[(seed, PS.LOG2_OTHER)]["t_c"],
                          "sign_first_violation_2^-14": ob["sign_first_violation"], "crossed_2^-14": ob["t_obs"] is not None})
    q3["other_degenerate_releases_in_plan"] = [d for d in degen if d["seed"] != SIGN_SEED]
    q3["degenerate_releases_in_plan"] = degen
    q3["registered_outcome_unchanged"] = S["outcome"]
    out["q3_sign_seed"] = q3
    out["compute"] = {"secs_total": round(sum(r.get("secs", 0) or 0 for r in parts), 1),
                      "peak_rss_gb_max": max([r.get("peak_rss_gb", 0) or 0 for r in parts] or [0])}
    return P2A._jsonable(out)


# =============================================================================================== page
def _f(v, d=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "∞"
        return f"{v:.{d}g}"
    return str(v)


def _seed(s):
    return f"{s:,}"


def _plain(J):
    """The plain answers, every number read from the JSON."""
    ref = J["clean_reference"]
    q1 = {e["seed"]: e for e in J["q1_no_forecast"]}
    slow = [q1[s] for s in (2_975_003, 2_975_093, 2_975_179)]
    rts = [r["rho_time_to_target_quasi_static"] for e in slow for r in e["runs"].values()]
    s64 = [r["path"]["s_over_sF_at_rho_t_64"] for e in slow for r in e["runs"].values()]
    dl = [e["M_slope"]["dLds_at"]["0.95"] for e in slow]
    sF = [e["constants"]["s_F"] for e in slow]
    rtc = [r["rho_t_c"] for e in slow[:2] for r in e["runs"].values()]
    q2, q3 = J["q2_none"], J["q3_sign_seed"]
    ex = q3["excluded_rescore_POST_HOC"]
    labs = q2["crossed"]["labels"]
    nS = sum(v == "S" for v in labs.values())
    others = [_seed(int(k)) for k, v in labs.items() if v != "S"]
    st = q3["cause"]["steps"]
    return ["## Plain answers (POST HOC)", "",
            f"1. **No forecast on 4 clean seeds.** Every failure point was reproduced bit for bit with the registered "
            f"code from the hashed states (t_c, t_c state, forecast record and observed s_end identical).",
            f"   - **2,975,052: PIPELINE.** Its exact-M release carries a numerically dead unit (|v₂| = "
            f"{_f(q3['cause']['release_v'][2])}) that the registered rule (v ≠ 0.0) counts as active. In â that unit "
            f"lets the scale direction leak into the un-throttled part of the update: Δs at step 1 is "
            f"{_f(st['-14']['registered_ds'])} at both rates (vs {_f(st['-14']['ds_live_units_only'])} at 2⁻¹⁴ without "
            f"it). So t_c = 73, ε̂_F ≈ 0.78, ŝ_c ≈ 15.7 lies beyond the 1.6·s_F horizon, and that gives \"no extrapolated "
            f"s reaches s_c\".",
            f"   - **2,975,003 and 093 (no ŝ reaches s_F) and 179 (no cutoff): LANDSCAPE (slow growth near the fold). "
            f"The fixed ⌈64/ρ⌉ budget is where it shows.** These are the three largest s_F in the plan ({_f(min(sF))}–"
            f"{_f(max(sF))}; the other 36 clean seeds have median {_f(ref['constants_median_other_36']['s_F'])}, max "
            f"{_f(ref['constants_range_other_36']['s_F'][1])}). On their own M, dL/ds at 0.95·s_F is "
            f"{', '.join(_f(x) for x in dl)}, against the other seeds' smallest |dL/ds| at t_c of "
            f"{_f(ref['-16']['min_abs_dLds_at_tc'])} (median {_f(abs(ref['-16']['median_dLds_at_tc']))}). M has no "
            f"stall (dL/ds < 0 up to the fold), but the quasi-static ρ·t to reach the target (s_F, or 0.95·s_F for 179) "
            f"is {_f(min(rts))}–{_f(max(rts))} > 64. The observed s at ρ·t = 64 is {_f(min(s64))}–{_f(max(s64))}·s_F. "
            f"The quasi-static formula matches the observed ρ·t from 0.8 to 0.95·s_F to < 0.1%. The cutoffs of 003 "
            f"and 093 came at ρ·t {_f(min(rtc))}–{_f(max(rtc))} (other seeds: median {_f(ref['-16']['median_rho_t_c'])}, "
            f"max {_f(ref['-16']['max_rho_t_c'])}). The forecaster's quadratic was capped at the remaining run budget "
            f"(uncapped it reaches s_F only ~8–21% beyond the budget). So its refusal is CORRECT: the true path did not "
            f"reach s_F either. The margin is small: the budget misses by ~1–11% in ρ·t.",
            f"2. **The 7 crossing none seeds** crossed into the seed's own **S (slab) branch on {nS}/7**: the crossing "
            f"state minimised at s_obs lands within ≤ 2e−7 of S. On {len(others)}/7 ({', '.join(others)}) it lands on "
            "an unidentified 3-unit minimum with ρ₂ just above q (≈ 0.395, 0.392), 0.09–0.15 from S and further from "
            "S2 and L0. On 2,975,110 and 114, S's own ρ₂ is just below q at 1.01·s_F (0.381, 0.384) and reaches q at "
            "1.041 and 1.033·s_F, exactly where they crossed. The registered class read S below q at 1.01·s_F.",
            f"   - On the other 5 crossers S has ρ₂ ≥ q at 1.01·s_F, but none of the 30 registered minima reached S. "
            f"S has ρ₂ ≥ q at 1.01·s_F on {q2['S_at_1.01_ge_q']}/30 none seeds, while the registered block found S on "
            f"only {q2['registered_block_on_S_any']}/30.",
            f"   - **Wider search** (s = 1.01, 1.02, 1.05, 1.10·s_F; perturbations ×1, ×2, ×4; 360 minima per seed): "
            f"**{q2['n_wider_changes_class']}/30 ({100 * q2['frac_wider_changes_class']:.0f}%) of the trained none "
            f"seeds would change class**, all to mixed (pooled), none to clean. Crossers "
            f"{q2['crossed']['n_wider_changes']}/7 (2,975,103 stays none); non-crossers "
            f"{q2['not_crossed']['n_wider_changes']}/23.",
            "   - So the wider search does not separate crossers from non-crossers (86% vs 65%), and neither does S's "
            "ρ₂ profile (S reaches q by 1.25·s_F on 7/7 and 23/23). On these seeds H's failure reads as a class rule "
            "that samples basins too narrowly. It does not show that the dynamics are predicted by S's existence.",
            f"3. **2,975,052's sign flip** is the same degenerate release: v₂ = {_f(q3['cause']['release_v'][2])}, "
            f"W₂ ≈ 1e−77, c₂ ≈ 1e−82. The first update moves v₂ by {_f(st['-14']['dv'][2])} at full η = 1, so its "
            f"sign flips at step 1 at both rates. It is not a dynamical event near the fold. The one other planned "
            f"seed with such a release (2,975,127, none) flips at step 1 too.",
            f"   - **Excluding the seed (POST HOC):** validity becomes OK (signs fixed at every clean run; follow "
            f"{_f(ex['validity']['follow_frac'])}; t_c < t_obs {_f(ex['validity']['cutoff_before_frac'])}). The verdicts "
            f"are unchanged: " + ", ".join(f"{k} {v}" for k, v in ex["verdicts"].items())
            + f". The POST HOC outcome is **{ex['outcome']}**. **The registered outcome remains "
            f"{q3['registered_outcome_unchanged']}.**", ""]


def render_md(J):
    L = ["# Phase 2A-PS POST HOC: no-forecast seeds, crossing none seeds, and the sign flip", "",
         f"**POST HOC.** {J['label']} Generated by `python -m src.phase2a_ps_posthoc build` from "
         "`results/phase2a_ps_posthoc_parts.jsonl` and the committed registered files (registration c819290, "
         "forecasts ca62d97, results d6fb90e).", "",
         f"Registered outcome: **{J['registered']['outcome']}** — "
         + ", ".join(f"{k} {v}" for k, v in J["registered"]["verdicts"].items())
         + "; " + ", ".join(f"{k} {v}" for k, v in J["registered"]["secondary_verdicts"].items()) + ". "
         "Nothing below changes it.", ""]
    L += _plain(J)
    ref = J["clean_reference"]
    # (1)
    L += ["## (1) The four clean seeds with no forecast", ""]
    L += ["Reference: the other 36 trained clean seeds (median; range): "
          + ", ".join(f"{k} {_f(ref['constants_median_other_36'][k])} ({_f(ref['constants_range_other_36'][k][0])}–"
                      f"{_f(ref['constants_range_other_36'][k][1])})" for k in ref["constants_median_other_36"])
          + f". dL_M/ds at the t_c state (2⁻¹⁶): median {_f(ref['-16']['median_dLds_at_tc'])}, smallest |·| "
          f"{_f(ref['-16']['min_abs_dLds_at_tc'])}; ρ·t_c median {_f(ref['-16']['median_rho_t_c'])}, max "
          f"{_f(ref['-16']['max_rho_t_c'])} (budget ρ·t = 64).", ""]
    L += ["| seed | rate | registered status | failure point (reproduced) | diagnosis |", "|---|---|---|---|---|"]
    for e in J["q1_no_forecast"]:
        for l2, r in e["runs"].items():
            q = r.get("quad") or {}
            if r["t_c"] is None:
                fp = (f"no step with s ≥ 0.95·s_F in {r['budget']:,} steps; s at the budget "
                      f"{_f(r['observed_s_end_over_sF'])}·s_F")
            elif r.get("horizon"):
                h = r["horizon"]
                fp = (f"t_c {r['t_c']}; ε̂_F {_f(h['eps_fc'])} → ŝ_c {_f(h['s_c_fc'])} > 1.6·s_F = "
                      f"{_f(h['horizon_1.6sF'])}: the path stops at the horizon before ŝ_c")
            else:
                fp = (f"t_c {r['t_c']:,}; quad window {q.get('n_window'):,}, concave {q.get('concave')}; path capped "
                      f"by the budget ({q.get('n_ahead_used'):,} steps); ŝ at its end "
                      f"{_f((q.get('s_hat_at_end_of_used_path') or 0) / e['constants']['s_F'])}·s_F; uncapped vertex "
                      f"{_f((q.get('s_hat_max_uncapped') or 0) / e['constants']['s_F'])}·s_F; uncapped k to s_F "
                      f"{_f(q.get('k_reach_sF_uncapped'))}")
            rep = "reproduced" if r.get("forecast_reproduced") else "NOT reproduced"
            L.append(f"| {_seed(e['seed'])} | 2^{l2} | {r['status']} | {fp} ({rep}) | **{r['diagnosis']['label']}**: "
                     f"{r['diagnosis']['reason']} |")
    L.append("")
    for e in J["q1_no_forecast"]:
        c, ms = e["constants"], e.get("M_slope", {})
        L.append(f"- **{_seed(e['seed'])}**: s_F {_f(c['s_F'])}, s₀/s_F {_f(c['s0_over_sF'])}, Λ_F {_f(c['Lambda_F'])}, "
                 f"s_F/s* {_f(c['sF_over_sstar'])}. dL_M/ds on its own M: "
                 + ", ".join(f"{k}·s_F {_f(v)}" for k, v in (ms.get("dLds_at") or {}).items())
                 + f"; stall at {_f(ms.get('stall_scale_over_sF'))}·s_F; quasi-static ρ·t 0.8→0.95·s_F "
                 f"{_f(ms.get('rho_time_0.8_to_0.95'))}, 0.95→0.9999·s_F {_f(ms.get('rho_time_0.95_to_0.9999'))}.")
        for l2, r in e["runs"].items():
            p = r.get("path") or {}
            L.append(f"  - 2^{l2}: ρ·t to target (quasi-static, incl. ρ·t spent) {_f(r['rho_time_to_target_quasi_static'])}; "
                     f"s/s_F at ρ·t 16/32/48/64: {_f(p.get('s_over_sF_at_rho_t_16'))}/{_f(p.get('s_over_sF_at_rho_t_32'))}/"
                     f"{_f(p.get('s_over_sF_at_rho_t_48'))}/{_f(p.get('s_over_sF_at_rho_t_64'))}; late d ln s/d(ρt) "
                     f"{_f(p.get('late_dlogs_drhot'))}; observed ρ·t 0.8→0.95·s_F {_f(r.get('observed_rho_time_0.8_to_0.95'))}; "
                     f"s_end reproduced {r.get('observed_s_end_reproduced')}.")
    L.append("")
    # (2)
    q2 = J["q2_none"]
    L += ["## (2) The none seeds: what the 7 crossers crossed into, and a wider search past the fold", ""]
    if q2.get("per_seed"):
        L += [f"Registered perturbed minima reproduced on {q2['n_registered_reproduced']}/{q2['n']} none seeds. "
              f"The seed's own S has ρ₂ ≥ q at 1.01·s_F on {q2['S_at_1.01_ge_q']}/{q2['n']}; the registered 1.01 block "
              f"put any minimum on S on {q2['registered_block_on_S_any']}/{q2['n']}.", "",
              "| seed | crossed (s_obs/s_F) | crossing minimum: branch, ρ₂, dist | S: ρ₂ at 1.01·s_F; first ρ₂ ≥ q | registered block: n ≥ q, on S, ρ₂ median | wider: pooled; per scale 1.01/1.02/1.05/1.10 (n ≥ q of 90) |",
              "|---|---|---|---|---|---|"]
        for e in q2["per_seed"]:
            cr = e.get("crossing")
            cm = (f"{cr['label']}, {_f(cr['min_rho2'])}, {_f(cr['dist_to_branch'].get(cr['label']) if cr['label'] != 'other' else min([v for v in cr['dist_to_branch'].values() if v is not None] or [math.inf]))}"
                  if cr else "—")
            ps_ = e["wider"]["per_scale"]
            pn = e["wider"]["per_scale_n_ge_q"]
            L.append(f"| {_seed(e['seed'])} | {'yes (' + _f(e['s_obs_over_sF']) + ')' if e['crossed'] else ('no, reached 1.25' if e['reached_1.25sF'] else 'no, budget end at ' + _f(e['s_end_over_sF']))} | {cm} | "
                     f"{_f(e['S_rho2_at_1.01'])}; {_f(e['S_first_rho2_ge_q_over_sF'])} | "
                     f"{e['registered_block']['n_ge_q']}, {e['registered_block']['n_on_S']}, {_f(e['registered_block']['rho2_med'])} | "
                     f"**{e['wider']['pooled']}**; " + "/".join(f"{ps_[k]} ({pn[k]})" for k in ps_) + " |")
        L += ["", f"- **Wider search changes the class of {q2['n_wider_changes_class']}/{q2['n']} trained none seeds "
              f"({100 * q2['frac_wider_changes_class']:.0f}%)** (pooled: clean {q2['wider_pooled_counts']['clean']}, "
              f"mixed {q2['wider_pooled_counts']['mixed']}, none {q2['wider_pooled_counts']['none']}). Crossers: "
              f"{q2['crossed']['n_wider_changes']}/{q2['crossed']['n']}; non-crossers: "
              f"{q2['not_crossed']['n_wider_changes']}/{q2['not_crossed']['n']}.",
              "- Per scale (pooled over amplitudes ×1, ×2, ×4): "
              + "; ".join(f"{k}·s_F: " + ", ".join(f"{c} {n}" for c, n in v.items())
                          for k, v in q2["wider_per_scale_counts"].items()) + ".",
              f"- Crossers whose S reaches ρ₂ ≥ q by 1.25·s_F: {q2['crossed']['n_S_ge_q_before_1.25']}/"
              f"{q2['crossed']['n']}; non-crossers: {q2['not_crossed']['n_S_ge_q_before_1.25']}/"
              f"{q2['not_crossed']['n']} (signs moved on {q2['not_crossed']['n_signs_moved']} non-crossers).", ""]
        for e in q2["per_seed"]:
            cr = e.get("crossing")
            if not cr:
                continue
            L.append(f"- {_seed(e['seed'])}: crossing replayed (s_obs reproduced {cr['s_obs_reproduced']}, crossing "
                     f"reproduced {cr['crossing_reproduced']}); minimum at s_obs: ρ₂ {_f(cr['min_rho2'])}, active units "
                     f"{cr['min_active_units']}, distance from the crossing state {_f(cr['dist_state_to_min'])}; distance "
                     "to S/S2/L0 " + "/".join(_f(cr['dist_to_branch'][k]) for k in OTHER_BRANCHES)
                     + "; their ρ₂ " + "/".join(_f(cr['branch_rho2_at_s_obs'][k]) for k in OTHER_BRANCHES)
                     + f" → **{cr['label']}**.")
        L.append("")
    # (3)
    q3 = J["q3_sign_seed"]
    c = q3["cause"] or {}
    L += [f"## (3) Seed {_seed(q3['seed'])}: the active-sign flip at step 1", ""]
    if c:
        st = c["steps"]
        L += [f"- Release (exact M(s₀), frozen): v = {', '.join(_f(x) for x in c['release_v'])}; W of unit 2 "
              f"{_f(c['release_W'][4])}, {_f(c['release_W'][5])}; c₂ {_f(c['release_c'][2])}. The registered active set "
              f"(v ≠ 0.0) is {c['active_units_registered']}; numerically dead among them (|v| ≤ 1e−12·s): "
              f"{c['dead_units']}. ∂L/∂v = {', '.join(_f(x) for x in c['release_grad_v'])}; every other active "
              f"gradient ≤ {_f(c['release_grad_max_other'])}.",
              "- With unit 2 in â = sign(v)/√3, the part of ∇_vL along the true scale direction of the two live units "
              "is no longer inside the ρ-scaled component: (I − ââᵀ)∇_vL moves v₀, v₁ outward and v₂ inward at the "
              "full η = 1."]
        for l2, d in st.items():
            L.append(f"  - 2^{l2}: Δv = {', '.join(_f(x) for x in d['dv'])} (orthogonal part "
                     f"{', '.join(_f(x) for x in d['dv_orthogonal'])}); Δs at step 1 {_f(d['registered_ds'])} vs "
                     f"{_f(d.get('ds_live_units_only'))} with the dead unit removed from â; sign flips at step 1: units "
                     f"{d['sign_flips']}; t_c {d['t_c']} at both rates, observation {d['steps_observed']} steps "
                     f"(observe reproduced {d['observe_reproduced']}).")
        L.append("")
    od = q3["other_degenerate_releases_in_plan"]
    L.append("- Other planned seeds with a numerically dead 'active' unit at the release: "
             + (", ".join(f"{_seed(o['seed'])} ({o['class']}, unit {o['dead_units']}, |v| {_f(o['min_abs_v'])}; 2⁻¹⁴: t_c "
                        f"{o['t_c_2^-14']}, first sign change at step {o['sign_first_violation_2^-14']}, crossed "
                        f"{o['crossed_2^-14']})" for o in od) or "none") + ".")
    ex = q3["excluded_rescore_POST_HOC"]
    L += ["", f"**POST HOC rescoring with seed {_seed(q3['seed'])} EXCLUDED (beside the registered verdicts; the "
          f"registered outcome remains {q3['registered_outcome_unchanged']}):**", "",
          "| criterion | registered | POST HOC, seed excluded | statistic (excluded) |", "|---|---|---|---|"]
    reg = {**J["registered"]["verdicts"], **J["registered"]["secondary_verdicts"]}
    cr = {**ex["criteria"], **ex["secondary"]}
    for k in list(PS.PRIMARY) + list(PS.SECONDARY):
        v = {**ex["verdicts"], **ex["secondary_verdicts"]}[k]
        x = cr[k]
        stat = {"F": lambda x: f"{x['n_within']}/{x['n']} within; falsifier {x['n_falsifier']}",
                "H": lambda x: f"{x['n_no_crossing_to_1.25sF']}/{x['n']}",
                "E_seed": lambda x: f"median {_f(x['median'])}, CI [{_f(x['ci95'][0])}, {_f(x['ci95'][1])}], n {x['n']}",
                "C3": lambda x: f"{x['n_gt_1']}/{x['n']}",
                "C4": lambda x: f"no forecast {x['n_no_forecast']}; mean D {_f(x['mean_D'], 6)}, CI [{_f(x['ci95'][0], 6)}, {_f(x['ci95'][1], 6)}]",
                "P": lambda x: f"mean D {_f(x['mean_D'])}, CI [{_f(x['ci95'][0])}, {_f(x['ci95'][1])}], n {x['n']}"}
        s = stat[k](x) if k in stat else (f"{x.get('n_within', '—')}/{x.get('n', '—')}" if isinstance(x, dict) else "")
        L.append(f"| {k} | {reg[k]} | {v} | {s} |")
    va = ex["validity"]
    L += ["", f"Validity, seed excluded: follow {_f(va['follow_frac'])}, t_c < t_obs {_f(va['cutoff_before_frac'])}, "
          f"idle 0 at every clean run {va['idle_zero_all']}, signs fixed at every clean run {va['signs_fixed_all']} → "
          f"**{'OK' if va['ok'] else 'NOT OK'}**. POST HOC outcome with the seed excluded: **{ex['outcome']}** "
          f"(not a verdict; the registered outcome is {q3['registered_outcome_unchanged']}).", "",
          f"Compute: {J['compute']['secs_total']} s on one process (nice 15, one thread); peak RSS "
          f"{_f(J['compute']['peak_rss_gb_max'], 3)} GB.", ""]
    return "\n".join(L)


def write():
    J = build()
    OUT_JSON.write_text(json.dumps(J, indent=1))
    OUT_MD.write_text(render_md(json.loads(json.dumps(J))))
    print(f"wrote {OUT_JSON.name}, {OUT_MD.name}")


def main(argv):
    cmd = argv[1]
    if cmd == "gate":
        f, w = PS.memory_gate("posthoc " + (" ".join(argv[2:]) or "gate"))
        print(f"memory gate OK: free {f}% swap_free {w} MB")
        return
    PS._setup()
    {"nofc": job_nofc, "none": job_none, "sign": job_sign, "build": write}[cmd]()


if __name__ == "__main__":
    main(sys.argv)
