"""POST HOC (not registered; changes NO verdict): the two final checks of the experimental program (author, 2026-10-08).

Nothing here is registered and no registered verdict changes: Phase 2A stays PASS (registration 569b836, results
7b15d9f), 2A-PS stays UNRESOLVED (validity) (registration c819290, results d6fb90e), 2A-PS2 stays FAIL H (registration
a1bd625, stamp 3a6f57b, OpenTimestamps b2564fa, forecasts 678f43d, results 4d10abc).  Registered code (src/phase2a.py,
src/phase2a_ps.py, src/phase2a_ps2.py, ...) is imported and NEVER modified; frozen files, runs, observed rows and scores
are read, never written.  No new registration, no new design.

(1) The fold-delay law with one correction term, r = A ε^{2/3} + C ε ln(1/ε) ([KS01] Rem. 2.11; math note §14.3).
    2A: the committed POST HOC fit (src/phase2a_posthoc.py, results/phase2a_posthoc.json, commit 9e56ebb) is CITED, not
    recomputed.  2A-PS2: every clean seed has two rates (2⁻¹⁴, 2⁻¹⁶), so per seed only the two-point slope
    Δln r_obs/Δln ε (= the registered E_seed exponent with ε = ε̂_F) is available: (a) its distribution and whether it
    falls with the seed's ε (Spearman, ε-quartile bins); (b) pooled fits with per-seed A_s and a common C (and A_s = Ω₀;
    and a common A); (c) C against 2A's.  ε sources: the forecast ε̂_F (registered) and, as sensitivity, an observed ε_F
    from the committed first-passage steps (secant over [0.95, 1]·s_F; quadratic t(s) through 0.8, 0.95, 1.0·s_F).

(2) The 2A-PS2 trained none seeds: the 7 that crossed by 1.25·s_F against the 23 that did not, on every FROZEN landscape
    quantity, plus a DISTANCE from the fold point to the nearest slab minimum (definition below) recomputed from the
    frozen states without training; Mann–Whitney U (exact, conditional on ties) with Holm adjustment; DESCRIPTIVE.
    2A-PS's 7 vs 23 as a replication check (its wider-search counts and t* are POST HOC there, not frozen).

DISTANCE (exact definition).  P_F = sb_fold.run_to_full(M.theta(0.9999·s_F)): the seed's own M at 0.9999·s_F, in the
unit-ℓ₁ landscape parametrisation, which is the starting point of the frozen wider search.  The wider search is
re-drawn exactly as src/phase2a_ps2.wide_search (default_rng([seed, 20261005]); 30 perturbed minima per (scale, amplitude)
at 1.01/1.02/1.05/1.10·s_F × ×1/×2/×4; sb_fold.local_min_batch at s = scale·s_F) and its counts are asserted equal to
the frozen ones.  d_slab = min over the 360 minima with ρ₂ ≥ q of the function-space distance sb_fold.fdist(P_F, P_m)
(RMS over the seed's 800 points of the difference of unit-ℓ₁ network functions; invariant to unit permutations and
sign flips); +∞ if no minimum has ρ₂ ≥ q.  Also per scale, the same with the ℓ₂ distance in P (minimised over the 24
unit permutations), and d_S(f) = fdist(P_F, S(f·s_F)) to the seed's own slab branch S (registered homotopy and
continuation, src/phase2a_ps_posthoc.seed_branch).

    python -m src.final_posthoc gate TAG     # memory gate (logged to results/final_posthoc_memory_gate.log)
    python -m src.final_posthoc dist         # (2) distances for the 60 trained none seeds (2A-PS2 and 2A-PS); resumable
    python -m src.final_posthoc build        # results/final_posthoc.json and .md (no compute beyond seconds)

Machine: one process, nice 15, one thread; memory gate as a separate logged step before each job and between seeds
(free ≥ 25 %, swap free ≥ 500 MB; waits); disk free ≥ 20 GB else STOP; STOP above 1 GB peak RSS.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import sys
import time

import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "final_posthoc.json"
OUT_MD = RESULTS / "final_posthoc.md"
PARTS = RESULTS / "final_posthoc_parts.jsonl"
GATE_LOG = RESULTS / "final_posthoc_memory_gate.log"
P2A_POSTHOC_JSON = RESULTS / "phase2a_posthoc.json"
PHASES = {"ps2": RESULTS / "phase2a_ps2", "ps": RESULTS / "phase2a_ps"}
PS_POSTHOC_PARTS = RESULTS / "phase2a_ps_posthoc_parts.jsonl"

OMEGA0 = 2.338107410459767
TWO_THIRDS = 2.0 / 3.0
Q = 0.3914103370353161                       # phase2a_ps.Q (asserted equal in the job)
WIDE_SCALES = (1.01, 1.02, 1.05, 1.10)
WIDE_AMPS = (1.0, 2.0, 4.0)
WIDE_N = 30
WIDE_STREAM = 2_026_1005
S_DIST_SCALES = (1.01, 1.05)
RSS_STOP = 1024 ** 3                          # 1 GB peak RSS: STOP
DISK_MIN = 20 * 1024 ** 3
PERM_N = 20_000                               # Spearman permutation p
PERM_SEED = 2_026_1008

LABEL = ("POST HOC (not registered; author 2026-10-08): the two final checks. Changes NO verdict: 2A PASS, 2A-PS "
         "UNRESOLVED (validity), 2A-PS2 FAIL H stand.")
REGISTERED = {"phase2a": {"registration": "569b836", "results": "7b15d9f", "posthoc": "9e56ebb"},
              "phase2a_ps": {"registration": "c819290", "forecasts": "ca62d97", "results": "d6fb90e",
                             "posthoc": "2f0fb33"},
              "phase2a_ps2": {"registration": "a1bd625", "stamp": "3a6f57b", "opentimestamps": "b2564fa",
                              "forecasts": "678f43d", "results": "4d10abc"}}


# =============================================================================================== pure functions (tested)
def ks_value(A, C, eps):
    eps = np.asarray(eps, float)
    return A * eps ** TWO_THIRDS + C * eps * np.log(1.0 / eps)


def ks_local_slope(A, C, eps):
    """d ln r/d ln ε of r = A ε^{2/3} + C ε ln(1/ε) (analytic)."""
    eps = np.asarray(eps, float)
    num = TWO_THIRDS * A * eps ** TWO_THIRDS + C * eps * (np.log(1.0 / eps) - 1.0)
    return num / ks_value(A, C, eps)


def two_point_slope(r1, r2, e1, e2):
    """Δln r/Δln ε between two points (the registered seed_exponent's formula); None unless all > 0 and e1 ≠ e2."""
    vals = (r1, r2, e1, e2)
    if any(v is None or not np.isfinite(v) or v <= 0 for v in vals) or e1 == e2:
        return None
    return (math.log(r2) - math.log(r1)) / (math.log(e2) - math.log(e1))


def ks_two_point_slope(A, C, e1, e2):
    """The two-point slope a law r = A ε^{2/3} + C ε ln(1/ε) gives between ε₁ and ε₂."""
    return two_point_slope(float(ks_value(A, C, e1)), float(ks_value(A, C, e2)), e1, e2)


def ks_exact_two_point(r1, r2, e1, e2):
    """(A, C) through two points exactly (2 × 2 linear solve)."""
    M = np.array([[e1 ** TWO_THIRDS, e1 * math.log(1 / e1)], [e2 ** TWO_THIRDS, e2 * math.log(1 / e2)]])
    A, C = np.linalg.solve(M, np.array([r1, r2], float))
    return float(A), float(C)


def fit_pooled_ks(eps, r, group, mode="free_A"):
    """Relative least squares Σ((model − r)/r)² for r = A_g ε^{2/3} + C ε ln(1/ε) over pooled points.
    mode 'free_A': one A per group, common C; 'Omega0': A_g = Ω₀ for all, common C; 'common_A': one A, one C.
    Returns {A: {group: A_g}, C, rel_resid (list), rms_rel, max_abs_rel, n, n_par}."""
    eps, r = np.asarray(eps, float), np.asarray(r, float)
    group = list(group)
    gs = sorted(set(group), key=lambda g: group.index(g))
    e23, eL = eps ** TWO_THIRDS / r, eps * np.log(1 / eps) / r
    if mode == "free_A":
        X = np.zeros((len(r), len(gs) + 1))
        for i, g in enumerate(group):
            X[i, gs.index(g)] = e23[i]
        X[:, -1] = eL
        sol = np.linalg.lstsq(X, np.ones(len(r)), rcond=None)[0]
        A = {g: float(sol[j]) for j, g in enumerate(gs)}
        C = float(sol[-1])
    elif mode == "Omega0":
        y = 1.0 - OMEGA0 * e23
        C = float((eL @ y) / (eL @ eL))
        A = {g: OMEGA0 for g in gs}
    elif mode == "common_A":
        sol = np.linalg.lstsq(np.column_stack([e23, eL]), np.ones(len(r)), rcond=None)[0]
        A = {g: float(sol[0]) for g in gs}
        C = float(sol[1])
    else:
        raise ValueError(mode)
    model = np.array([A[g] for g in group]) * eps ** TWO_THIRDS + C * eps * np.log(1 / eps)
    rel = model / r - 1.0
    npar = {"free_A": len(gs) + 1, "Omega0": 1, "common_A": 2}[mode]
    return {"A": A, "C": C, "rel_resid": rel.tolist(), "rms_rel": float(np.sqrt(np.mean(rel ** 2))),
            "max_abs_rel": float(np.max(np.abs(rel))), "n": int(len(r)), "n_par": npar}


def ranks(x):
    """Average (mid) ranks, 1-based; +∞ ranks last (ties averaged)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    rk = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and (xs[j + 1] == xs[i]):
            j += 1
        rk[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return rk


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else float("nan")


def spearman_perm_p(x, y, n_perm=PERM_N, seed=PERM_SEED):
    """Two-sided permutation p of Spearman's ρ (default_rng(seed); (k + 1)/(n_perm + 1))."""
    rho = spearman(x, y)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    k = sum(abs(spearman(x, rng.permutation(y))) >= abs(rho) - 1e-12 for _ in range(n_perm))
    return rho, (k + 1) / (n_perm + 1)


def mann_whitney_exact(x, y):
    """Mann–Whitney U of x against y with the EXACT two-sided p conditional on the observed ties (the null
    distribution of x's mid-rank sum over all C(n_x + n_y, n_x) splits, by dynamic programming on doubled ranks).
    p = P(|W − E W| ≥ |w − E W|).  Returns {U (for x), auc = U/(n_x n_y), p, n_x, n_y}; p = 1 if every value ties."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m, n = len(x), len(y)
    rk = ranks(np.r_[x, y])
    r2 = np.rint(2 * rk).astype(int)                       # doubled mid-ranks are integers
    w2 = int(r2[:m].sum())
    U = w2 / 2.0 - m * (m + 1) / 2.0
    total = int(r2.sum())
    # counts[k][s]: number of k-subsets with doubled-rank sum s (Python ints: exact)
    smax = int(np.sort(r2)[::-1][:m].sum())
    counts = [[0] * (smax + 1) for _ in range(m + 1)]
    counts[0][0] = 1
    for v in r2:
        for k in range(m, 0, -1):
            row, prev = counts[k], counts[k - 1]
            for s in range(smax, v - 1, -1):
                if prev[s - v]:
                    row[s] += prev[s - v]
    dist = counts[m]
    N = sum(dist)
    mean2 = m * total / (m + n)                             # E of the doubled rank sum
    dev = abs(w2 - mean2)
    hit = sum(c for s, c in enumerate(dist) if c and abs(s - mean2) >= dev - 1e-9)
    return {"U": U, "auc": U / (m * n), "p": hit / N, "n_x": m, "n_y": n}


def holm(pvals):
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    k = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(k)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (k - i) * p[j]))
        adj[j] = run
    return adj.tolist()


UNIT_PERMS = tuple(itertools.permutations(range(4)))


def perm_index(perm):
    """Column order of a 16-vector with its units permuted (unit k: W at 2k, 2k+1; c at 8+k; η at 12+k)."""
    return [i for k in perm for i in (2 * k, 2 * k + 1)] + [8 + k for k in perm] + [12 + k for k in perm]


def param_dist(P, Q16):
    """ℓ₂ distance in the 16-vector landscape parametrisation, minimised over the 24 permutations of the 4 units
    (rows of Q16 against one P)."""
    P, Q16 = np.asarray(P, float), np.atleast_2d(np.asarray(Q16, float))
    return np.min(np.stack([np.linalg.norm(Q16[:, perm_index(pm)] - P[None], axis=1) for pm in UNIT_PERMS]), axis=0)


def nearest_slab(dist, rho2, q=Q):
    """min of dist over entries with ρ₂ ≥ q; +∞ if none."""
    dist, rho2 = np.asarray(dist, float), np.asarray(rho2, float)
    sel = rho2 >= q
    return float(dist[sel].min()) if sel.any() else math.inf


def observed_eps(s_F, Lambda_F, t08, t_c, s_at_tc, t_F, eta=1.0):
    """Observed ε_F = (ṡ_F/s_F)/(ηΛ_F) from the committed first-passage steps: secant ṡ over [s(t_c), s_F]
    (t_c is the first step with s ≥ 0.95·s_F, s(t_c) committed), and the quadratic t(s) through (0.8·s_F, t₀₈),
    (s(t_c), t_c), (s_F, t_F) differentiated at s_F.  Level steps are first-passage steps (s within one step's increment
    of the level)."""
    sec = (s_F - s_at_tc) / (t_F - t_c)
    S = np.array([0.8 * s_F, s_at_tc, s_F]); T = np.array([t08, t_c, t_F], float)
    a, b, _ = np.polyfit(S - s_F, T, 2)
    quad = 1.0 / b                                          # dt/ds at s_F is b (centered at s_F)
    return {"secant": sec / s_F / (eta * Lambda_F), "quadratic": quad / s_F / (eta * Lambda_F)}


def summary(x):
    x = np.asarray([v for v in x if v is not None], float)
    fin = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    out = {"n": int(len(x)), "n_inf": int((~np.isfinite(x)).sum()), "median": float(np.median(x)),
           "min": float(x.min()), "max": float(x.max())}
    if len(fin) == len(x):
        out["q25"], out["q75"] = (float(v) for v in np.percentile(x, (25, 75)))
    return out


# =============================================================================================== (2) the distance job
def _gate(tag):
    from .act_fold import check_memory
    while True:
        try:
            ok, f, w = check_memory()
        except Exception:                                                  # noqa: BLE001
            ok, f, w = False, float("nan"), float("nan")
        disk = shutil.disk_usage(ROOT).free
        with open(GATE_LOG, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB "
                     f"disk_free={disk / 1024 ** 3:.1f}GB {'OK' if ok and disk >= DISK_MIN else 'WAIT'}\n")
        if disk < DISK_MIN:
            raise SystemExit(f"STOP: disk free {disk / 1024 ** 3:.1f} GB < 20 GB")
        if ok:
            return f, w
        time.sleep(60)


def _peak_rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS


def _rss_guard():
    if _peak_rss() > RSS_STOP:
        raise SystemExit(f"STOP: peak RSS {_peak_rss() / 1e9:.2f} GB > 1 GB")


def _setup():
    cur = os.getpriority(os.PRIO_PROCESS, 0)
    if cur < 15:
        os.nice(15 - cur)
    import torch
    torch.set_num_threads(1)


def _append_write(row):
    with open(PARTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _jsonl(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def none_seeds(phase):
    fr = json.loads((PHASES[phase] / "frozen.json").read_text())
    return [p["seed"] for p in fr["plan"] if p["class"] == "none"]


def _wide_minima(M, s_F, X, Y, seed):
    """phase2a_ps2.wide_search re-drawn (same operations, same stream), keeping the minima."""
    from . import phase2a_ps as PS
    from . import sb_fold as SBF
    Pb = SBF.run_to_full(M.theta(0.9999 * s_F)[None])[0]
    rng = np.random.default_rng([int(seed), WIDE_STREAM])
    out = []
    for f in WIDE_SCALES:
        for amp in WIDE_AMPS:
            P0 = Pb[None] + rng.normal(0, 1, (WIDE_N, 16)) * amp * (0.05 * np.abs(Pb) + 0.02)
            P0[:, 12:] = np.abs(P0[:, 12:])
            Pm, L, gn = SBF.local_min_batch(P0, np.full(WIDE_N, f * s_F), X, Y)
            r = np.array([PS.rho2_static(p, X) for p in Pm])
            out.append((f, amp, Pm, r, L, gn))
    return Pb, out


def job_dist():
    from . import phase2a_ps as PS
    from . import phase2a_ps2 as PS2
    from . import phase2a_ps_posthoc as PH
    from . import sb_fold as SBF
    _setup()
    assert PS.Q == Q and PH.WIDE_STREAM == WIDE_STREAM and PH.WIDE_SCALES == WIDE_SCALES and PH.WIDE_AMPS == WIDE_AMPS
    done = {(r["phase"], r["seed"]) for r in _jsonl(PARTS)}
    ref_ps = {r["seed"]: r for r in _jsonl(PS_POSTHOC_PARTS) if r["kind"] == "none"}
    for phase in ("ps2", "ps"):
        rows = {r["seed"]: r for r in _jsonl(PHASES[phase] / "frozen_parts.jsonl")}
        for seed in none_seeds(phase):
            if (phase, seed) in done:
                continue
            _gate(f"dist {phase} {seed}")
            _rss_guard()
            t0 = time.time()
            fs = rows[seed]
            X, Y = PS.sample(seed)
            assert PS.sample_sha(X, Y) == fs["sample_sha256"]
            s_F = fs["s_F"]
            M = PH.seed_M(fs, X, Y)                       # asserts s_F and the release hash equal the frozen
            Pb, blocks = _wide_minima(M, s_F, X, Y, seed)
            counts = [{"scale": f, "amp": a, "n": WIDE_N, "n_ge_q": int((r >= Q).sum())} for f, a, _, r, _, _ in blocks]
            if phase == "ps2":
                ref = fs["ps2"]["wide"]["blocks"]
            else:
                ref = [{k: b[k] for k in ("scale", "amp", "n", "n_ge_q")} for b in ref_ps[seed]["wide_blocks"]]
            row = {"phase": phase, "seed": seed, "wide_counts": counts, "wide_reproduced": counts == ref}
            per = []
            for f, a, Pm, r, L, gn in blocks:
                fd = SBF.fdist(Pm, Pb[None], X)
                pd_ = param_dist(Pb, Pm)
                per.append({"scale": f, "amp": a, "n_ge_q": int((r >= Q).sum()),
                            "d_slab_f": nearest_slab(fd, r), "d_slab_param": nearest_slab(pd_, r),
                            "d_any_f_min": float(fd.min()), "rho2_max": float(r.max()),
                            "gnorm_max": float(gn.max())})
            row["blocks"] = per
            S = PH.seed_branch("S", X, Y)
            row["S_range"] = None if S is None else [float(S.lo), float(S.hi)]
            row["d_S_f"], row["rho2_S"] = {}, {}
            for f in S_DIST_SCALES:
                P = PH.branch_P(S, f * s_F)
                row["d_S_f"][f"{f:g}"] = float(SBF.fdist(P[None], Pb[None], X)[0]) if P is not None else None
                row["rho2_S"][f"{f:g}"] = PS.rho2_static(P, X) if P is not None else None
            if phase == "ps":                              # 2A-PS froze neither t* nor the activity test: POST HOC here
                row["rho_t_star_posthoc"] = PS2.tstar_record(M, fs["s0"], s_F)["rho_t_star"]
                row["activity_posthoc"] = PS2.activity(fs["release_theta"])
            row["secs"] = round(time.time() - t0, 1)
            row["peak_rss_gb"] = round(_peak_rss() / 1e9, 3)
            _append_write(row)
            print(json.dumps({"phase": phase, "seed": seed, "reproduced": row["wide_reproduced"],
                              "d_slab": min(b["d_slab_f"] for b in per), "secs": row["secs"],
                              "rss": row["peak_rss_gb"]}), flush=True)
            _rss_guard()


# =============================================================================================== build: part (1)
def _cite_2a():
    J = json.loads(P2A_POSTHOC_JSON.read_text())
    fc = J["fits_by_eps_source"]["eps_fc"]["fits"]
    ks, ks0 = fc["KS_eps_ln_free_A"], fc["KS_eps_ln_A_Omega0"]
    cub = J["fits_by_eps_source"]["eps_obs_cubic"]["fits"]["KS_eps_ln_free_A"]
    sec = J["fits_by_eps_source"]["eps_obs_secant"]["fits"]["KS_eps_ln_free_A"]
    rows = {x["log2rho"]: x for x in J["rows"]}
    a, b = rows[-14.0], rows[-16.0]
    tp = two_point_slope(a["r_obs"], b["r_obs"], a["eps_fc"], b["eps_fc"])
    tp_sec = two_point_slope(a["r_obs"], b["r_obs"], a["eps_obs_secant"], b["eps_obs_secant"])
    return {"source": "results/phase2a_posthoc.json (src/phase2a_posthoc.py, commit 9e56ebb); cited, not recomputed",
            "n_rates": len(J["rows"]), "A": ks["coef"]["e23"], "A_over_Omega0": ks["A_over_Omega0"],
            "C": ks["coef"]["eL"], "rms_rel": ks["rms_rel"], "max_abs_rel": ks["max_abs_rel"],
            "local_slope_at_slowest": ks["local_slope_at_slowest"],
            "local_slope_extrapolated": ks["local_slope_extrapolated"],
            "eps_within_0.01": ks["eps_where_slope_within_0.01_of_2/3"],
            "Omega0_fixed_C": ks0["coef"]["eL"], "Omega0_fixed_rms_rel": ks0["rms_rel"],
            "obs_cubic": {"A_over_Omega0": cub["A_over_Omega0"], "C": cub["coef"]["eL"]},
            "obs_secant": {"A_over_Omega0": sec["A_over_Omega0"], "C": sec["coef"]["eL"]},
            "eps_fc_2^-14": a["eps_fc"], "eps_fc_2^-16": b["eps_fc"],
            "two_point_slope_2^-14_2^-16_derived": tp, "two_point_slope_2^-14_2^-16_obs_secant_derived": tp_sec,
            "fit_two_point_slope_2^-14_2^-16": ks_two_point_slope(ks["coef"]["e23"], ks["coef"]["eL"],
                                                                   a["eps_fc"], b["eps_fc"])}


def _clean_pairs():
    """2A-PS2 clean seeds with both rates: committed scores rows + observed + runs (read only)."""
    D = PHASES["ps2"]
    S = json.loads((D / "scores.json").read_text())
    obs = {(o["seed"], o["log2rho"]): o for o in _jsonl(D / "observed.jsonl")}
    runs = {(r["seed"], r["log2rho"]): r for r in _jsonl(D / "runs.jsonl")}
    by = {}
    for r in S["rows"]:
        if r["class"] == "clean":
            by.setdefault(r["seed"], {})[r["log2rho"]] = r
    out = []
    for seed, d in by.items():
        a, b = d[-14.0], d[-16.0]
        rec = {"seed": seed, "s_F": b["s_F"], "Lambda_F": b["Lambda_F"]}
        for l2, x in ((-14.0, a), (-16.0, b)):
            o, rr = obs[(seed, l2)], runs[(seed, l2)]
            assert o["t_obs"] == x["t_obs"] and rr["t_c"] == x["t_c"]
            oe = observed_eps(x["s_F"], x["Lambda_F"], o["t_08"], rr["t_c"], rr["s_at_tc"], o["t_F"])
            rec[f"{l2:g}"] = {"r_obs": x["r_obs"], "eps_fc": x["eps_fc"], "eps_obs_secant": oe["secant"],
                              "eps_obs_quadratic": oe["quadratic"], "r_fc": x["r_fc"]}
        out.append(rec)
    return out, S


def _bins(x, key, n_bins=4):
    order = np.argsort(key)
    return [order[i * len(x) // n_bins:(i + 1) * len(x) // n_bins] for i in range(n_bins)]


def part1():
    two_a = _cite_2a()
    A2, C2 = two_a["A"], two_a["C"]
    pairs, S = _clean_pairs()
    reg_expo = S["seed_exponents"]
    src = ("eps_fc", "eps_obs_secant", "eps_obs_quadratic")
    per_seed = []
    for p in pairs:
        a, b = p["-14"], p["-16"]
        e = {"seed": p["seed"], "s_F": p["s_F"], "Lambda_F": p["Lambda_F"],
             "r_obs_14": a["r_obs"], "r_obs_16": b["r_obs"], "eps_fc_14": a["eps_fc"], "eps_fc_16": b["eps_fc"]}
        for k in src:
            e[f"slope_{k}"] = two_point_slope(a["r_obs"], b["r_obs"], a[k], b[k])
            e[f"{k}_16"] = b[k]
            e[f"{k}_14"] = a[k]
            e[f"slope_2A_fit_{k}"] = ks_two_point_slope(A2, C2, a[k], b[k])
            e[f"exact_AC_{k}"] = ks_exact_two_point(a["r_obs"], b["r_obs"], a[k], b[k])
        per_seed.append(e)
    assert [round(x["slope_eps_fc"], 15) for x in per_seed] == [round(x, 15) for x in reg_expo], \
        "per-seed slope differs from the registered seed_exponents"
    out = {"two_A": two_a, "n_seeds": len(per_seed), "per_seed": per_seed, "by_eps_source": {}}
    for k in src:
        sl = np.array([x[f"slope_{k}"] for x in per_seed])
        e16 = np.array([x[f"{k}_16"] for x in per_seed])
        pred = np.array([x[f"slope_2A_fit_{k}"] for x in per_seed])
        rho, p = spearman_perm_p(sl, e16)
        rho_d, p_d = spearman_perm_p(sl - pred, e16)
        bins = []
        for idx in _bins(sl, e16):
            bins.append({"n": int(len(idx)), "eps16_range": [float(e16[idx].min()), float(e16[idx].max())],
                         "slope_median": float(np.median(sl[idx])), "slope_mean": float(np.mean(sl[idx])),
                         "slope_2A_fit_median": float(np.median(pred[idx])),
                         "n_below_2_3": int((sl[idx] < TWO_THIRDS).sum())})
        Cs = np.array([x[f"exact_AC_{k}"][1] for x in per_seed])
        As = np.array([x[f"exact_AC_{k}"][0] for x in per_seed]) / OMEGA0
        ent = {"slope": summary(sl), "n_below_2_3": int((sl < TWO_THIRDS).sum()),
               "slope_sd": float(np.std(sl, ddof=1)), "slope_2A_fit_sd": float(np.std(pred, ddof=1)),
               "n_below_2A_fit": int((sl < pred).sum()),
               "slope_2A_fit": summary(pred), "slope_minus_2A_fit": summary(sl - pred),
               "eps16": summary(e16), "eps14_over_eps16": summary([x[f"{k}_14"] / x[f"{k}_16"] for x in per_seed]),
               "spearman_slope_vs_eps16": {"rho": rho, "p_perm": p},
               "spearman_slope_minus_fit_vs_eps16": {"rho": rho_d, "p_perm": p_d},
               "bins_by_eps16_quartile": bins,
               "per_seed_exact": {"A_over_Omega0": summary(As), "C": summary(Cs), "n_C_negative": int((Cs < 0).sum())}}
        e_all = np.array([[x[f"{k}_14"], x[f"{k}_16"]] for x in per_seed]).ravel()
        r_all = np.array([[x["r_obs_14"], x["r_obs_16"]] for x in per_seed]).ravel()
        g_all = np.repeat([x["seed"] for x in per_seed], 2)
        fits = {}
        for mode in ("free_A", "Omega0", "common_A"):
            f = fit_pooled_ks(e_all, r_all, g_all, mode)
            Aseed = np.array([f["A"][x["seed"]] for x in per_seed])
            ls = np.array([float(ks_local_slope(f["A"][x["seed"]], f["C"], x[f"{k}_16"])) for x in per_seed])
            ls_2a = np.array([float(ks_local_slope(A2, C2, x[f"{k}_16"])) for x in per_seed])
            rel = np.array(f["rel_resid"])
            fits[mode] = {"C": f["C"], "C_over_2A_C": f["C"] / C2, "n": f["n"], "n_par": f["n_par"],
                          "rms_rel": f["rms_rel"], "max_abs_rel": f["max_abs_rel"],
                          "rel_resid_14": summary(rel[0::2]), "rel_resid_16": summary(rel[1::2]),
                          "A_over_Omega0": summary(Aseed / OMEGA0),
                          "local_slope_at_eps16": summary(ls), "local_slope_2A_fit_at_eps16": summary(ls_2a),
                          "n_local_slope_below_2_3": int((ls < TWO_THIRDS).sum())}
        ent["pooled_fits"] = fits
        out["by_eps_source"][k] = ent
    out["registered_E_seed"] = S["criteria"]["E_seed"]
    return out


# =============================================================================================== build: part (2)
def _num(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return float(x)
    return float(x)


QUANT_FROZEN = (   # key, label, getter on (frozen row, evaluation, parts row), frozen in ps2?, in Holm family?
    ("abs_mc", "fold constant |m′c′| (registered window)", lambda f, e, d: f["fold_constant"]["abs_mc"]),
    ("abs_mc_0.025", "|m′c′|, half window (0.025)", lambda f, e, d: f["fold_constant"]["abs_mc_0.025"]),
    ("mc_rel_diff", "|m′c′| two-window relative difference", lambda f, e, d: f["fold_constant"]["rel_diff"]),
    ("mc_agree", "|m′c′| windows agree (1/0)", lambda f, e, d: e.get("mc_agree_DESCRIPTIVE")),
    ("Lambda_F", "Λ_F", lambda f, e, d: e["Lambda_F"]),
    ("s_F", "s_F", lambda f, e, d: f["s_F"]),
    ("s_star", "s*", lambda f, e, d: f["s_star"]),
    ("sF_over_sstar", "s_F/s*", lambda f, e, d: f["sF_over_sstar"]),
    ("s0", "s₀", lambda f, e, d: f["s0"]),
    ("s0_over_sF", "s₀/s_F", lambda f, e, d: f["s0_over_sF"]),
    ("S_fold_up_over_sF", "S's upper fold / s_F", lambda f, e, d: (f["S_fold_up"] / f["s_F"]
                                                                   if f.get("S_fold_up") is not None else None)),
    ("rho2_S_101", "S's ρ₂ at 1.01·s_F", lambda f, e, d: f.get("rho2_S_at_1.01sF")),
    ("rho2_M_max", "max ρ₂ on M (stable part)", lambda f, e, d: f["rho2_M_max"]),
    ("release_rho2", "ρ₂ at the release", lambda f, e, d: f["release_rho2"]),
    ("release_loss", "loss at the release", lambda f, e, d: f["release_loss"]),
    ("min_share", "smallest active-unit share at the release", None),
    ("rho_t_star", "ρ·t* (t* of the budget rule)", None),
    ("class_n_rho2_ge_q", "registered 30-minimum count with ρ₂ ≥ q", lambda f, e, d: e["class_n_rho2_ge_q"]),
)
WIDE_KEYS = (("wide_all", "wider search: n of 360 with ρ₂ ≥ q", None, None),) + tuple(
    (f"wide_s{f:g}", f"wider search at {f:g}·s_F: n of 90", f, None) for f in WIDE_SCALES) + tuple(
    (f"wide_x{a:g}", f"wider search at ×{a:g}: n of 120", None, a) for a in WIDE_AMPS)
DIST_KEYS = (
    ("d_slab", "d_slab: fdist(P_F, nearest ρ₂ ≥ q minimum), 360", lambda d: min(b["d_slab_f"] for b in d["blocks"])),
    ("d_slab_101", "d_slab at 1.01·s_F only (90)", lambda d: min(b["d_slab_f"] for b in d["blocks"]
                                                                 if b["scale"] == 1.01)),
    ("d_slab_param", "ℓ₂ distance in P (permutation-minimised) to nearest ρ₂ ≥ q minimum",
     lambda d: min(b["d_slab_param"] for b in d["blocks"])),
    ("d_S_101", "fdist(P_F, S(1.01·s_F))", lambda d: d["d_S_f"].get("1.01")),
    ("d_S_105", "fdist(P_F, S(1.05·s_F))", lambda d: d["d_S_f"].get("1.05")),
)


def _phase_table(phase, parts):
    D = PHASES[phase]
    fz = {r["seed"]: r for r in _jsonl(D / "frozen_parts.jsonl")}
    obs = {(o["seed"], o["log2rho"]): o for o in _jsonl(D / "observed.jsonl")}
    dist = {r["seed"]: r for r in parts if r["phase"] == phase}
    out = []
    for seed in none_seeds(phase):
        f, d = fz[seed], dist.get(seed)
        e = f["evaluation"]
        o = obs[(seed, -14.0)]
        crossed = o["t_obs"] is not None and o["s_obs"] <= 1.25 * f["s_F"]
        row = {"seed": seed, "crossed": bool(crossed), "s_obs_over_sF": (o["s_obs"] / f["s_F"]
                                                                          if o["t_obs"] is not None else None)}
        for key, _, g in QUANT_FROZEN:
            if key == "min_share":
                row[key] = (f["ps2"]["activity"]["min_share_registered"] if phase == "ps2" else
                            (d["activity_posthoc"]["min_share_registered"] if d else None))
            elif key == "rho_t_star":
                row[key] = (e["rho_t_star"] if phase == "ps2" else (d["rho_t_star_posthoc"] if d else None))
            else:
                row[key] = _num(g(f, e, d))
        if phase == "ps2":
            row["B"] = e["budgets"]["-14"]["B"]
        blocks = (f["ps2"]["wide"]["blocks"] if phase == "ps2" else (d["wide_counts"] if d else None))
        for key, _, sc, amp in WIDE_KEYS:
            row[key] = (None if blocks is None else
                        float(sum(b["n_ge_q"] for b in blocks if (sc is None or b["scale"] == sc)
                                  and (amp is None or b["amp"] == amp))))
        for key, _, g in DIST_KEYS:
            row[key] = (_num(g(d)) if d is not None else None)
        row["wide_reproduced"] = (d["wide_reproduced"] if d else None)
        out.append(row)
    return out


def compare(table):
    """Per quantity: crossers vs non-crossers medians/ranges, exact Mann–Whitney, Holm over the tested quantities."""
    keys = [k for k, _, _ in QUANT_FROZEN] + [k for k, *_ in WIDE_KEYS] + [k for k, _, _ in DIST_KEYS]
    labels = {k: l for k, l, *_ in QUANT_FROZEN + WIDE_KEYS + DIST_KEYS}
    res, tested = [], []
    for k in keys:
        xc = [r[k] for r in table if r["crossed"] and r[k] is not None]
        xn = [r[k] for r in table if not r["crossed"] and r[k] is not None]
        ent = {"key": k, "label": labels[k], "crossers": summary(xc), "non_crossers": summary(xn)}
        allv = xc + xn
        if len(xc) < 2 or len(xn) < 2 or len(set(allv)) <= 1:
            ent["test"] = None
            ent["note"] = "constant (not tested)" if len(set(allv)) <= 1 else "too few values (not tested)"
        else:
            ent["test"] = mann_whitney_exact(xc, xn)
            tested.append(len(res))
        res.append(ent)
    adj = holm([res[i]["test"]["p"] for i in tested])
    for i, a in zip(tested, adj):
        res[i]["test"]["p_holm"] = a
    return {"n_crossers": sum(r["crossed"] for r in table), "n_non_crossers": sum(not r["crossed"] for r in table),
            "n_tested": len(tested), "quantities": res,
            "min_p": min((res[i]["test"]["p"] for i in tested), default=None),
            "min_p_holm": min((res[i]["test"]["p_holm"] for i in tested), default=None)}


def part2():
    parts = _jsonl(PARTS)
    out = {}
    for phase in ("ps2", "ps"):
        tab = _phase_table(phase, parts)
        c = compare(tab)
        out[phase] = {"table": tab, "comparison": c,
                      "n_dist_rows": sum(1 for r in parts if r["phase"] == phase),
                      "n_wide_reproduced": sum(1 for r in tab if r["wide_reproduced"]),
                      "max_peak_rss_gb": max((r["peak_rss_gb"] for r in parts if r["phase"] == phase), default=None),
                      "secs_total": round(sum(r["secs"] for r in parts if r["phase"] == phase), 1)}
        wm = [r for r in tab if r["wide_all"] and r["wide_all"] > 0]
        out[phase]["DESCRIPTIVE_wide_mixed_only"] = {
            "n_seeds": len(wm), "n_crossers": sum(r["crossed"] for r in wm),
            "tests": {k: mann_whitney_exact([r[k] for r in wm if r["crossed"]], [r[k] for r in wm if not r["crossed"]])
                      for k in ("wide_all", "wide_s1.01", "wide_x1", "d_slab")}}
        if phase == "ps2":
            out[phase]["B_same_ranks_as_rho_t_star"] = bool(
                np.array_equal(ranks([r["B"] for r in tab]), ranks([r["rho_t_star"] for r in tab])))
    return out


def build():
    return {"label": LABEL, "registered": REGISTERED,
            "part1_delay_law": part1(), "part2_crossers": part2(),
            "distance_definition": __doc__.split("DISTANCE (exact definition).")[1].split("\n\n")[0].strip()}


# =============================================================================================== markdown
def _g(x, d=4):
    if x is None:
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e5:
        return f"{int(x)}"
    if isinstance(x, float) and x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.{d - 1}e}"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def _rng(s, d=4):
    if not s or s.get("n", 0) == 0:
        return "—"
    return f"{_g(s['median'], d)} [{_g(s['min'], d)}, {_g(s['max'], d)}]"


def render_md(J):
    P1, P2 = J["part1_delay_law"], J["part2_crossers"]
    A = P1["two_A"]
    fc = P1["by_eps_source"]["eps_fc"]
    L = ["# Final POST HOC checks: the delay law per seed, and what separates crossing none seeds", "",
         f"**POST HOC.** {J['label']} Registered code, frozen files, runs, observations and scores are read, never "
         "written. Generated by `python -m src.final_posthoc build` from the committed registered files, "
         "`results/phase2a_posthoc.json` (cited) and `results/final_posthoc_parts.jsonl` (the distance job, "
         "`python -m src.final_posthoc dist`). The experimental program is complete with this page: no new "
         "registration, no new design.", "",
         "## Plain conclusions", ""]
    fA = fc["pooled_fits"]["free_A"]
    sp = fc["spearman_slope_vs_eps16"]
    q1p, q4p = fc["bins_by_eps16_quartile"][0]["slope_2A_fit_median"], fc["bins_by_eps16_quartile"][-1]["slope_2A_fit_median"]
    q1o, q4o = fc["bins_by_eps16_quartile"][0]["slope_median"], fc["bins_by_eps16_quartile"][-1]["slope_median"]
    sps = [(k, e["spearman_slope_vs_eps16"]["rho"], e["spearman_slope_vs_eps16"]["p_perm"])
           for k, e in P1["by_eps_source"].items()]
    L += [f"1. **Delay law with one correction term.** 2A (one landscape, {A['n_rates']} rates; cited from 9e56ebb): "
          f"r = A ε^{{2/3}} + C ε ln(1/ε) gives A/Ω₀ = {A['A_over_Omega0']:.4f}, C = {A['C']:.4f}, rms rel. residual "
          f"{100 * A['rms_rel']:.3f} %; local exponent {A['local_slope_at_slowest']['2^-18']:.4f} at 2⁻¹⁸, approaching "
          f"2/3 from above (within 0.01 of 2/3 only below ε ≈ {A['eps_within_0.01']:.2g}). "
          f"2A-PS2 ({P1['n_seeds']} clean seeds, two rates each): the per-seed two-point slope (= the registered E_seed "
          f"exponent) has median {fc['slope']['median']:.4f}, range {fc['slope']['min']:.4f}–{fc['slope']['max']:.4f}"
          f" (IQR {fc['slope']['q25']:.4f}–{fc['slope']['q75']:.4f}); {fc['n_below_2_3']}/{P1['n_seeds']} lie below "
          f"2/3. 2A's fit predicts {_rng(fc['slope_2A_fit'])} for the same ε pairs, i.e. a spread across seeds of "
          f"{fc['slope_2A_fit']['max'] - fc['slope_2A_fit']['min']:.4f}, against an observed spread of "
          f"{fc['slope']['max'] - fc['slope']['min']:.4f}. Spearman of slope against the seed's ε̂_F at 2⁻¹⁶: "
          f"ρ = {sp['rho']:.3f} (permutation p = {sp['p_perm']:.3f}); the sign is "
          f"{'OPPOSITE to' if sp['rho'] < 0 else 'the same as'} a trend toward 2/3 at smaller ε. "
          f"Pooled fit (per-seed A_s, common C): C = {fA['C']:.4f} ({fA['C_over_2A_C']:.2f} × 2A's {A['C']:.4f}), "
          f"A_s/Ω₀ {_rng(fA['A_over_Omega0'])}; implied local slope at each seed's 2⁻¹⁶ ε: "
          f"{_rng(fA['local_slope_at_eps16'])} (2A's fit at the same ε: {_rng(fA['local_slope_2A_fit_at_eps16'])}). "
          f"**The per-seed data do not trend toward 2/3 as ε decreases.** Across the seeds' ε range 2A's law predicts "
          f"the two-point slope to drop by {q4p - q1p:.4f} from the largest-ε to the smallest-ε quartile (median "
          f"{q4p:.4f} → {q1p:.4f}); the observed quartile medians rise instead ({q4o:.4f} → {q1o:.4f}), and the Spearman sign is negative with "
          f"every ε source ({', '.join(f'{k}: {v:.3f} (p {pv:.3f})' for k, v, pv in sps)}"
          f"{'; none with p < 0.05' if all(pv >= 0.05 for _, _, pv in sps) else ''}). The seed-to-seed SD of the slope is {fc['slope_sd']:.4f}, "
          f"{fc['slope_sd'] / fc['slope_2A_fit_sd']:.0f} × the SD 2A's law gives the same ε pairs "
          f"({fc['slope_2A_fit_sd']:.4f}): the per-seed slope is dominated by landscape-to-landscape variation (and the "
          "forecaster's per-rate ε̂ error, which the observed-ε rows remove without changing the picture), not by the "
          "ε-dependence of a common law. There is no sampling noise per seed (each run is deterministic), and two rates "
          "per seed fix A_s and C_s exactly, so a per-seed trend toward 2/3 is not testable here. On average the "
          f"per-seed data agree with 2A: median slope {fc['slope']['median']:.4f} vs 2A's {A['two_point_slope_2^-14_2^-16_derived']:.4f} "
          f"over the same rates, and a common C of the same sign and size ({fA['C']:.3f} vs {A['C']:.3f})."]
    c2, c1 = P2["ps2"]["comparison"], P2["ps"]["comparison"]

    def _q(c, key):
        return next(q for q in c["quantities"] if q["key"] == key)

    def _desc(q):
        t = q["test"]
        return (f"{q['label']} (crossers {_g(q['crossers']['median'], 4)} vs non-crossers "
                f"{_g(q['non_crossers']['median'], 4)}; AUC {t['auc']:.3f}, exact p {t['p']:.2g}, Holm p "
                f"{t['p_holm']:.2g})")

    def _rep(key):
        q = _q(c1, key)
        t = q["test"]
        return ("not testable in 2A-PS" if t is None else
                f"2A-PS: AUC {t['auc']:.3f}, p {t['p']:.2g}, Holm p {t['p_holm']:.2g}")
    sig2 = [q for q in c2["quantities"] if q["test"] and q["test"]["p_holm"] < 0.05]
    sig1 = [q for q in c1["quantities"] if q["test"] and q["test"]["p_holm"] < 0.05]
    a1 = [_q(c1, q["key"])["test"]["auc"] for q in sig2 if _q(c1, q["key"])["test"]]
    land2 = [q for q in sig2 if not q["key"].startswith(("wide", "d_"))]
    ds2, ds1 = _q(c2, "d_slab"), _q(c1, "d_slab")
    txt = (f"2. **Crossing vs non-crossing none seeds (2A-PS2, {c2['n_crossers']} vs {c2['n_non_crossers']}; "
           f"{c2['n_tested']} quantities, exact Mann–Whitney, Holm).** ")
    if sig2:
        txt += ("Holm-significant at 0.05: " + "; ".join(f"{_desc(q)} [{_rep(q['key'])}]" for q in sig2) + ". ")
    else:
        txt += "No quantity is Holm-significant at 0.05. "
    txt += (f"Nearest-slab distance d_slab: crossers {_g(ds2['crossers']['median'], 4)} vs non-crossers "
            f"{_g(ds2['non_crossers']['median'], 4)} (AUC {ds2['test']['auc']:.3f}, p {ds2['test']['p']:.2g}, Holm "
            f"{ds2['test']['p_holm']:.2g}); 2A-PS: {_g(ds1['crossers']['median'], 4)} vs "
            f"{_g(ds1['non_crossers']['median'], 4)} (p {ds1['test']['p']:.2g}). ")
    txt += (f"**{'No' if not land2 else 'A'} landscape quantity frozen before training (fold constant, Λ_F, s_F, s*, "
            "s_F/s*, s₀, t*, B, S's ρ₂, activity) separates crossers from non-crossers at Holm 0.05"
            + (".** " if not land2 else ": " + "; ".join(q["label"] for q in land2) + ".** "))
    txt += (("The only separation is " + ("the wider search's own count of slab minima past the fold" if all(
        q["key"].startswith("wide") for q in sig2) else ", ".join(q["label"] for q in sig2)) +
             f" (frozen before training, reported DESCRIPTIVELY only); in 2A-PS, where the same count was computed POST "
             f"HOC, {len(sig1)} quantit{'y' if len(sig1) == 1 else 'ies'} pass{'es' if len(sig1) == 1 else ''} Holm"
             + (f"; the same counts point the same way there (AUC {min(a1):.3f}–{max(a1):.3f}) but "
                f"{'do not reach' if not any(_q(c1, q['key'])['test'] and _q(c1, q['key'])['test']['p_holm'] < 0.05 for q in sig2) else 'partly reach'}"
                " Holm 0.05, so the separation does not replicate at that level. " if a1 else ". "))
            if sig2 else "")
    raw2 = [q for q in c2["quantities"] if q["test"] and q["test"]["p"] < 0.05
            and not q["key"].startswith(("wide", "d_"))]
    if raw2:
        txt += ("Landscape quantities with raw p < 0.05 in 2A-PS2 (none survives Holm): " + "; ".join(
            f"{q['label']} (AUC {q['test']['auc']:.3f}, p {q['test']['p']:.2g}, Holm {q['test']['p_holm']:.2g}; "
            f"{_rep(q['key'])})" for q in raw2) + ". ")
    both = [q for q in raw2 if _q(c1, q["key"])["test"] and _q(c1, q["key"])["test"]["p"] < 0.05
            and (q["test"]["auc"] - 0.5) * (_q(c1, q["key"])["test"]["auc"] - 0.5) > 0]
    if both:
        txt += ("Raw p < 0.05 in BOTH phases with the same direction: " + ", ".join(q["label"] for q in both) +
                " (" + "; ".join(f"crossers {'larger' if q['test']['auc'] > 0.5 else 'smaller'}" for q in both) +
                "). This is suggestive, not established: none survives Holm in either phase, and the "
                "two phases are separate seed sets of the same landscape family, chosen after both results. ")
    wm = P2["ps2"]["DESCRIPTIVE_wide_mixed_only"]
    txt += (f"Restricted to the {wm['n_seeds']} seeds whose wider search found any slab minimum ({wm['n_crossers']} "
            f"crossers), the count still differs (n of 360: AUC {wm['tests']['wide_all']['auc']:.3f}, p "
            f"{wm['tests']['wide_all']['p']:.2g}; ×1: AUC {wm['tests']['wide_x1']['auc']:.3f}, p "
            f"{wm['tests']['wide_x1']['p']:.2g}), so it is not only the wide-none seeds (0/6 crossed). ")
    txt += ("Descriptive: 7 vs 23 per phase gives little power against moderate shifts, so 'no separation' means no "
            "strong predictor among these quantities, not that none exists.")
    L += [txt, ""]
    # ---------------------------------------------------------------- part 1 detail
    L += ["## (1) The delay law with one correction term", "",
          "### 2A (cited; results/phase2a_posthoc.json, commit 9e56ebb; not recomputed)", "",
          "| quantity | value |", "|---|---|",
          f"| fit r = A ε^{{2/3}} + C ε ln(1/ε), ε̂_F, {A['n_rates']} rates: A | {A['A']:.4f} |",
          f"| A/Ω₀ | {A['A_over_Omega0']:.4f} |", f"| C | {A['C']:.4f} |",
          f"| rms / max rel. residual | {100 * A['rms_rel']:.3f} % / {100 * A['max_abs_rel']:.3f} % |",
          "| local exponent at 2⁻¹⁷, 2⁻¹⁷·⁵, 2⁻¹⁸ | " + ", ".join(f"{v:.4f}" for v in
                                                           A["local_slope_at_slowest"].values()) + " |",
          "| local exponent at ε = 1e-6, 1e-8, 1e-10, 1e-12 | " + ", ".join(
              f"{v:.4f}" for v in A["local_slope_extrapolated"].values()) + " |",
          f"| within 0.01 of 2/3 below ε ≈ | {A['eps_within_0.01']:.3g} |",
          f"| A = Ω₀ fixed: C (rms) | {A['Omega0_fixed_C']:.4f} ({100 * A['Omega0_fixed_rms_rel']:.3f} %) |",
          f"| observed ε_F (cubic): A/Ω₀, C | {A['obs_cubic']['A_over_Omega0']:.4f}, {A['obs_cubic']['C']:.4f} |",
          f"| observed ε_F (secant): A/Ω₀, C | {A['obs_secant']['A_over_Omega0']:.4f}, "
          f"{A['obs_secant']['C']:.4f} |",
          f"| 2A two-point slope 2⁻¹⁴→2⁻¹⁶ (derived from the committed rows; ε̂_F {A['eps_fc_2^-14']:.3e} → "
          f"{A['eps_fc_2^-16']:.3e}) | {A['two_point_slope_2^-14_2^-16_derived']:.4f} (fit: "
          f"{A['fit_two_point_slope_2^-14_2^-16']:.4f}; observed secant ε: "
          f"{A['two_point_slope_2^-14_2^-16_obs_secant_derived']:.4f}) |", ""]
    E = P1["registered_E_seed"]
    L += ["### 2A-PS2 per seed (40 clean seeds, 2⁻¹⁴ and 2⁻¹⁶)", "",
          f"Registered E_seed (unchanged): median {E['median']:.4f}, 95 % CI [{E['ci95'][0]:.4f}, {E['ci95'][1]:.4f}],"
          f" PASS. The per-seed slope with ε̂_F below equals the registered seed exponents exactly (asserted).", "",
          "| ε source | slope median [min, max] | IQR | < 2/3 | 2A fit's slope at the same ε pairs | slope − 2A fit "
          "| Spearman(slope, ε at 2⁻¹⁶) (perm. p) | Spearman(slope − fit, ε) (p) |", "|---|---|---|---|---|---|---|---|"]
    for k, ent in P1["by_eps_source"].items():
        s = ent["slope"]
        L.append(f"| {k} | {_rng(s)} | {s['q25']:.4f}–{s['q75']:.4f} | {ent['n_below_2_3']}/40 | "
                 f"{_rng(ent['slope_2A_fit'])} | {_rng(ent['slope_minus_2A_fit'])} | "
                 f"{ent['spearman_slope_vs_eps16']['rho']:.3f} ({ent['spearman_slope_vs_eps16']['p_perm']:.3f}) | "
                 f"{ent['spearman_slope_minus_fit_vs_eps16']['rho']:.3f} "
                 f"({ent['spearman_slope_minus_fit_vs_eps16']['p_perm']:.3f}) |")
    L += ["", "Binned by the seed's ε at 2⁻¹⁶ (quartiles, 10 seeds each; smallest ε first):", "",
          "| ε source | bin | ε at 2⁻¹⁶ range | slope median | slope mean | 2A fit's slope (median) | < 2/3 |",
          "|---|---|---|---|---|---|---|"]
    for k, ent in P1["by_eps_source"].items():
        for i, b in enumerate(ent["bins_by_eps16_quartile"]):
            L.append(f"| {k} | Q{i + 1} | {b['eps16_range'][0]:.3e}–{b['eps16_range'][1]:.3e} | "
                     f"{b['slope_median']:.4f} | {b['slope_mean']:.4f} | {b['slope_2A_fit_median']:.4f} | "
                     f"{b['n_below_2_3']}/{b['n']} |")
    L += ["", "Pooled fits of r_obs = A_s ε^{2/3} + C ε ln(1/ε) (relative least squares over 80 points):", "",
          "| ε source | A | n par | C | C / 2A's C | rms rel | max rel | A_s/Ω₀ median [min, max] | implied local "
          "slope at the seed's 2⁻¹⁶ ε | 2A fit's local slope at the same ε | local slope < 2/3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    names = {"free_A": "per-seed A_s", "Omega0": "A_s = Ω₀", "common_A": "one common A"}
    for k, ent in P1["by_eps_source"].items():
        for m, f in ent["pooled_fits"].items():
            L.append(f"| {k} | {names[m]} | {f['n_par']} | {f['C']:.4f} | {f['C_over_2A_C']:.2f} | "
                     f"{100 * f['rms_rel']:.2f} % | {100 * f['max_abs_rel']:.2f} % | {_rng(f['A_over_Omega0'])} | "
                     f"{_rng(f['local_slope_at_eps16'])} | {_rng(f['local_slope_2A_fit_at_eps16'])} | "
                     f"{f['n_local_slope_below_2_3']}/40 |")
    L += ["", "Per-seed exact solve (two points, two unknowns A_s, C_s; descriptive):", "",
          "| ε source | A_s/Ω₀ median [min, max] | C_s median [min, max] | C_s < 0 |", "|---|---|---|---|"]
    for k, ent in P1["by_eps_source"].items():
        pe = ent["per_seed_exact"]
        L.append(f"| {k} | {_rng(pe['A_over_Omega0'])} | {_rng(pe['C'])} | {pe['n_C_negative']}/40 |")
    L += ["", "Limits: two rates per seed, so per seed only a two-point slope is available and a per-seed A and C are "
          "exactly determined (no residual, no test). The runs are deterministic given (seed, ρ): there is no "
          "sampling noise per seed, but the 40 seeds are 40 different landscapes, and ε̂_F carries the forecaster's "
          "per-rate ṡ-extrapolation error, which enters the slope directly. The observed ε (secant over "
          "[0.95, 1]·s_F; quadratic t(s) through the 0.8, 0.95, 1.0·s_F first-passage steps) removes the forecaster "
          "but is a finite-difference estimate.", ""]
    # ---------------------------------------------------------------- part 2 detail
    L += ["## (2) Crossing vs non-crossing none seeds", "", "Distance definition: " + J["distance_definition"], ""]
    for phase, title in (("ps2", "2A-PS2 (registered a1bd625; wider search frozen before training)"),
                         ("ps", "2A-PS replication (registered c819290; wider search, t* and the activity test "
                                "computed POST HOC, not frozen)")):
        P = P2[phase]
        c = P["comparison"]
        L += [f"### {title}", "",
              f"{c['n_crossers']} crossers vs {c['n_non_crossers']} non-crossers; {c['n_tested']} quantities tested; "
              f"Holm over those {c['n_tested']}. Wider-search counts reproduced exactly on "
              f"{P['n_wide_reproduced']}/{len(P['table'])} seeds (re-drawn, asserted against the "
              f"{'frozen' if phase == 'ps2' else '2A-PS POST HOC (2f0fb33)'} counts). Distance job: "
              f"{P['secs_total']:.0f} s, peak RSS {_g(P['max_peak_rss_gb'], 3)} GB." +
              (" B (the budget) is a monotone function of ρ·t* (same ranks: "
               f"{P['B_same_ranks_as_rho_t_star']}), so it is not tested separately." if phase == "ps2" else ""), "",
              "| quantity | crossers median [min, max] | non-crossers median [min, max] | U | AUC | exact p | Holm p |",
              "|---|---|---|---|---|---|---|"]
        for q in c["quantities"]:
            t = q["test"]
            L.append(f"| {q['label']} | {_rng(q['crossers'])} | {_rng(q['non_crossers'])} | " +
                     (f"{t['U']:g} | {t['auc']:.3f} | {t['p']:.4f} | {t['p_holm']:.3f} |" if t else
                      f"— | — | {q.get('note', '—')} | — |"))
        L += ["", "Per seed (crossers first):", "",
              "| seed | crossed | s_obs/s_F | |m′c′| | Λ_F | s_F/s* | ρ·t* | S ρ₂ at 1.01 | wide n/360 | at 1.01 "
              "| d_slab | d_slab 1.01 | d_S(1.01) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(P["table"], key=lambda r: (not r["crossed"], r["seed"])):
            L.append(f"| {r['seed']:,} | {'yes' if r['crossed'] else 'no'} | {_g(r['s_obs_over_sF'], 3)} | "
                     f"{_g(r['abs_mc'], 3)} | {_g(r['Lambda_F'], 3)} | {_g(r['sF_over_sstar'], 3)} | "
                     f"{_g(r['rho_t_star'], 2)} | {_g(r['rho2_S_101'], 3)} | {_g(r['wide_all'], 0)} | "
                     f"{_g(r['wide_s1.01'], 0)} | {_g(r['d_slab'], 4)} | {_g(r['d_slab_101'], 4)} | "
                     f"{_g(r['d_S_101'], 4)} |")
        L.append("")
    L += ["## Caveats", "",
          "- POST HOC, after every result; nothing here is a verdict or changes one. 7 vs 23 per phase; the exact "
          "test's smallest attainable two-sided p is about 1/C(30, 7) · 2 ≈ 1e-6, but the power at these sizes against "
          "moderate shifts is low; Holm controls the family-wise error over the tested quantities of each phase.",
          "- The distance is to minima the wider search FOUND (360 local minimisations from perturbations of the fold "
          "point); it is an upper bound on the distance to the nearest slab basin, and +∞ where none was found (ranked "
          "last).",
          "- 2A-PS's wider-search counts and t* were computed after its results (2f0fb33 and here); in 2A-PS2 they "
          "were frozen before training.",
          "- The [KS01] Rem. 2.11 ε ln(1/ε) term is cited, not proved for this reduction (math note §14.5)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "build"
    if cmd == "gate":
        _gate(argv[1] if len(argv) > 1 else "gate")
        print("gate OK")
    elif cmd == "dist":
        job_dist()
    elif cmd == "build":
        J = json.loads(json.dumps(build()))
        OUT_JSON.write_text(json.dumps(J, indent=1) + "\n")
        OUT_MD.write_text(render_md(J))
        print(f"wrote {OUT_JSON.name}, {OUT_MD.name}")
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
