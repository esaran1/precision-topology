"""Registered test 2C: the band task in R^d against each seed's own-sample R^d switch, prospectively.

Design (author-approved 2026-09-29, unchanged): results/designs/2C_band_own_sample_design.md.
Registration: results/track2c_registration.md.  Tests: tests/test_track2c.py.

Setting (Track 3B, src/band_rd.py): x₁ exactly the width-1 task (fold1d.make_data(200, seed)); x₂…x_d i.i.d. U(−2, 2)
from numpy default_rng([seed, 3]); d ∈ {2, 4}; a ∈ {1.30, 1.50}; a single unit z = w₂·f_a(w·x + b₁) + b₂,
f_a(t) = t + a sin t; p = (w₁, b₁, w₂, b₂, w_noise) drawn U(−1, 1)^{d+3} in float32 from a torch Generator seeded with
the seed (bit-identical to band_rd.init_rd, without touching the global torch RNG); free Adam lr 0.01, full batch,
float64, budget 64,000 steps.  Crossing: the first step t ≥ 1 with G > 0 (band_rd.gap_rd, the exact R^d gap on the
continuous support, oriented by w₂); a run with G > 0 at initialisation is placed at init and has no crossing.

Prediction (per run):  s_pred = s_own,d·(1 + κ_k·χ).
  s_own,d  frozen before training: the placement switch of the seed's own-sample conditional minimiser in all hidden
           coordinates (w₁, b₁, b₂, w_noise) at w₂ = s, by the validated conditional search (`own_switch_rd`):
           bracket by steps of 0.1 from the seed's x₁-only own threshold, bisect to width ≤ 0.01, midpoint; both
           bracket ends validated (restart ladder 200 → 800, independent CMA-ES, audit).
  κ_k      the width-1 κ_k (lag_law.kappa_k with the width-1 population landscape of lag_law/kappa.csv and its median
           Adam preconditioner), frozen per (a, k); k = the winding of the branch occupied at the rule point.
  χ        pre-crossing information only.  Rule point t_R = the last upward passage of 0.5·s_own,d before s_t = |w₂(t)|
           first reaches s_own,d (track_a.rule_step).  t_sw = the first step with s_t ≥ s_own,d (the 2A rule; 2A
           passed).  The branch occupied at t_R (damped Newton in all hidden coordinates at w₂ = w₂(t_R) from the
           state at t_R, band_rd.branch_rd) is continued in s to s_{t_sw}; H_sig = its (w₁, b₁, b₂) Hessian block
           there.  P frozen at t_sw: relax = lr·λ_min(P^{1/2} H_sig P^{1/2}), P = 1/(√v̂(t_sw) + ε)
           (residual_timescale.relax_rate); growth = log(s_{t_sw}/s_{t_sw−100})/100; χ = growth/relax.
Crossings are kept out of the predictions as in Track A / 2A: `train` runs the full budget without evaluating any gap,
saves each path and its SHA-256, and computes the predictions; `observe` runs only after the predictions are committed.

    python -m src.track2c timing        # one non-registered seed: cost of the frozen inputs (no training)
    python -m src.track2c freeze        # per seed: x₁-only own threshold and validated s_own,d  -> frozen_seeds.csv
    python -m src.track2c kappa         # frozen width-1 κ_k table                               -> kappa_k.csv
    python -m src.track2c hashes        # registration manifest                                  -> registration.sha256
    python -m src.track2c train         # registered runs, no gap evaluated                      -> predictions_parts.jsonl
    python -m src.track2c finalize      # predictions.csv + predictions.sha256
    python -m src.track2c observe       # AFTER the predictions commit: crossings, scores         -> scores.json
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "track2c"
PATHS = OUT / "paths"

A_VALUES = (1.30, 1.50)
D_VALUES = (2, 4)
SEEDS = tuple(range(2_030_000, 2_030_060))          # 60 fresh seeds, the same seeds in all four cells (3B convention)
TIMING_SEED = 889_000                                # 3B's disclosed non-registered pilot seed (timing only)
BUDGET = 64_000                                      # 3B's budget
LR = 1e-2
WINDOW = 100                                         # growth window (3B)
RULE_FRAC = 0.5                                      # Track A rule point
TWO_PI = 2 * math.pi
K_RANGE = tuple(range(-3, 4))                        # windings tabulated for κ_k

# conditional search (validated, not certified)
RESTARTS = 200
LADDER = (200, 800)
CMA_STARTS, CMA_GENS, CMA_SIGMA0 = 20, 300, 1.0
POLISH = 5
GTOL, MAXIT, BATCH = 1e-8, 3000, 100
TIE = 1e-9
NEAR = 1e-7
NOISE_START_FRAC = 0.1                               # odd restarts start with w_noise in the box scaled by 0.1
BRACKET_STEP, WIDTH = 0.1, 0.01                      # own_threshold's bracket conventions
S_FLOOR, S_CEIL = 0.5, 20.0
CONT_STEP, CONT_JUMP, BRANCH_GTOL = 0.01, 0.5, 1e-8  # band_rd post hoc continuation conventions

# registered criteria and validity
C1_MAX = 0.05
C2_BAND = (0.75, 1.25)
MIN_CROSS = 30
RHO_MAX, RHO_FRAC = 0.05, 0.90
PRED_FRAC = 0.90
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 2_030_000, (2.5, 97.5)

RSS_LIMIT = 3 * 1024 ** 3
MEM_FREE_MIN, SWAP_FREE_MIN_MB = 25, 500

REGISTERED_FILES = ("src/track2c.py", "src/band_rd.py", "src/track_a.py", "src/own_threshold.py", "src/profiled_bnb.py",
                    "src/width2_conditional.py", "src/residual_timescale.py", "src/lag_law.py", "src/fold1d.py",
                    "src/cmaes.py", "src/linear_response.py", "tests/test_track2c.py",
                    "results/track2c_registration.md", "results/designs/2C_band_own_sample_design.md",
                    "results/lag_law/kappa.csv", "results/cond_certified_brackets.csv",
                    "results/track2c/kappa_k.csv", "results/track2c/frozen_seeds.csv")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonable(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, np.bool_):
            v = bool(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        out[k] = v
    return out


# ------------------------------------------------------------------------------------------ machine guards
def gate_ok():
    """Memory gate: memory_pressure free ≥ 25% and swap free ≥ 500 MB."""
    try:
        mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
        free = int(mp.rsplit("percentage:", 1)[1].strip().rstrip("%"))
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        swap_free = float(sw.split("free =")[1].split("M")[0])
    except Exception:
        return False
    return free >= MEM_FREE_MIN and swap_free >= SWAP_FREE_MIN_MB


def wait_gate(log=print):
    while not gate_ok():
        log(json.dumps({"memory_gate": "failed; pausing 120 s"}))
        time.sleep(120)


def rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")


# ------------------------------------------------------------------------------------------ data and initialisation
def make_data(seed, d):
    from .band_rd import make_data_rd
    return make_data_rd(int(seed), int(d))


def init_params(seed, d):
    """U(−1, 1)^{d+3} in float32 from a local torch Generator (bit-identical to band_rd.init_rd), cast to float64."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    return torch.empty(int(d) + 3).uniform_(-1.0, 1.0, generator=g).double()


# ------------------------------------------------------------------------------------------ own-sample R^d loss
def loss_grad_rd(P, s, a, X, y):
    """Profiled own-sample loss at w₂ = +s for rows P = (w₁, b₁, w_noise…) (b₂ profiled exactly), its gradient
    (envelope theorem) and the profiled b₂.  X: (n, d) with x₁ first."""
    from .width2_conditional import _sig, _softplus, profile_b_batch
    P = np.atleast_2d(np.asarray(P, float))
    W = np.concatenate([P[:, :1], P[:, 2:]], axis=1)
    T = W @ X.T + P[:, 1:2]
    Z0 = s * (T + a * np.sin(T))
    b = profile_b_batch(Z0, y)
    Z = Z0 + b[:, None]
    L = (_softplus(Z) - y[None, :] * Z).mean(axis=1)
    RD = (_sig(Z) - y[None, :]) * s * (1 + a * np.cos(T))
    gw = RD @ X / X.shape[0]
    G = np.column_stack([gw[:, :1], RD.mean(axis=1), gw[:, 1:]])
    return L, G, b


def loss_full(h, s, a, X, y):
    """Own-sample loss at w₂ = +s, h = (w₁, b₁, b₂, w_noise…)."""
    from .width2_conditional import _softplus
    h = np.asarray(h, float)
    t = X @ np.r_[h[0], h[3:]] + h[1]
    z = s * (t + a * np.sin(t)) + h[2]
    return float((_softplus(z) - y * z).mean())


def status_of(h, a):
    """Placement of a hidden configuration at w₂ > 0: 'placed' iff the exact R^d gap G₊ > 0 (the crossing rule)."""
    from .band_rd import gap_rd
    G = float(gap_rd(h[0], h[1], np.asarray(h[3:], float), 1.0, a))
    return ("placed" if G > 0 else "unplaced"), G


def start_box(s, a, x1, y):
    from .profiled_bnb import w_bound
    return float(w_bound(s, a, x1, y))


def starts(n, rng, d, W, sigma=None):
    """Restart k: w₁ ~ U(−W, W), b₁ ~ U(0, 2π), w_noise ~ U(−c W, c W)^{d−1} with c = 1 (k even) or 0.1 (k odd).
    sigma = ±1 (the mirror-restricted search): w₁ ~ sigma·U(0, W) instead."""
    P = np.empty((n, d + 1))
    P[:, 0] = rng.uniform(-W, W, n) if sigma is None else sigma * np.abs(rng.uniform(-W, W, n))
    P[:, 1] = rng.uniform(0.0, TWO_PI, n)
    c = np.where(np.arange(n) % 2 == 0, 1.0, NOISE_START_FRAC)
    P[:, 2:] = rng.uniform(-1.0, 1.0, (n, d - 1)) * (c * W)[:, None]
    return P


def search_rd(s, a, X, y, restarts, rng, sigma=None):
    """Batched BFGS (width2_conditional.bfgs_batch) from `restarts` starts; every restart is a candidate (loss, point,
    profiled b₂, gradient norm, flags), plus the constant predictor (k = −1)."""
    from .width2_conditional import bfgs_batch, constant_predictor_loss
    d = X.shape[1]
    W = start_box(s, a, X[:, 0], y)
    P0 = starts(restarts, rng, d, W, sigma)
    cands = []
    for i in range(0, restarts, BATCH):
        p0 = P0[i:i + BATCH]
        P, L, G, it = bfgs_batch(lambda Q, rows: loss_grad_rd(Q, s, a, X, y)[:2], p0, gtol=GTOL, maxit=MAXIT)
        _, _, B = loss_grad_rd(P, s, a, X, y)
        for k in range(len(p0)):
            gn = float(np.abs(G[k]).max())
            flags = []
            if not (np.all(np.isfinite(P[k])) and np.isfinite(L[k])):
                flags.append("nonfinite")
            if gn > 100 * GTOL:
                flags.append("not_converged")
            h = np.r_[P[k, 0], P[k, 1], B[k], P[k, 2:]]
            cands.append({"k": i + k, "h": h, "loss": float(L[k]), "gnorm": gn, "flags": flags})
    cands.append({"k": -1, "h": None, "loss": constant_predictor_loss(y), "gnorm": 0.0, "flags": ["constant_predictor"]})
    return cands, W


def eligible(c):
    return c["k"] >= 0 and not ({"nonfinite", "not_converged"} & set(c["flags"]))


def finish(cands, s, a, X, y, polish=POLISH):
    """Polish the `polish` lowest distinct eligible candidates by damped Newton in all hidden coordinates
    (band_rd.branch_rd); keep the BFGS point if Newton does not lower the loss.  Retained = the lowest polished loss,
    constant predictor included (status 'unplaced')."""
    from .band_rd import branch_rd
    ok = sorted([c for c in cands if eligible(c)], key=lambda c: c["loss"])
    picked, seen = [], []
    for c in ok:
        if all(abs(c["loss"] - l0) > 1e-10 for l0 in seen):
            picked.append(c); seen.append(c["loss"])
        if len(picked) == polish:
            break
    pol = []
    for c in picked:
        h0 = c["h"]
        hn, _, gmax, _ = branch_rd(a, X, y, np.r_[h0[0], h0[1], s, h0[2], h0[3:]])
        Ln = loss_full(hn, s, a, X, y)
        L0 = loss_full(h0, s, a, X, y)
        h, L, g = (hn, Ln, gmax) if Ln <= L0 + 1e-12 else (h0, L0, c["gnorm"])
        st, G = status_of(h, a)
        pol.append({"k": c["k"], "h": h, "loss": L, "gnorm_polished": g, "G": G, "status": st})
    const = next(c for c in cands if c["k"] == -1)
    pol.append({"k": -1, "h": None, "loss": const["loss"], "gnorm_polished": 0.0, "G": 0.0, "status": "unplaced"})
    ret = min(pol, key=lambda c: c["loss"])
    return ret, pol


def eval_status(s, a, X, y, restarts, rng):
    cands, W = search_rd(s, a, X, y, restarts, rng)
    ret, pol = finish(cands, s, a, X, y)
    return ret, pol, cands, W


# ------------------------------------------------------------------------------------------ validation (pure rules)
def ladder_ok(ret_small, ret_big, tol=TIE):
    """Restart ladder: retained loss unchanged within tol and the same status."""
    return bool(abs(ret_small["loss"] - ret_big["loss"]) <= tol and ret_small["status"] == ret_big["status"])


def independent_ok(retained_loss, other_loss, tol=TIE):
    """The independent search must not find a loss below the retained one by more than tol."""
    return bool(other_loss >= retained_loss - tol)


def audit_ok(ret, others, tol=TIE):
    """No candidate of the other status within tol of the retained loss.  `others`: dicts with 'loss', 'status'."""
    bad = [c for c in others if c is not ret and c["status"] != ret["status"] and c["loss"] <= ret["loss"] + tol]
    return bool(not bad), len(bad)


def end_validated(row, expected):
    """A bracket end is validated iff ladder, independent search and audit all pass and its status is the expected
    one ('unplaced' at s_lo, 'placed' at s_hi)."""
    return bool(row["ladder_ok"] and row["independent_ok"] and row["audit_ok"] and row["status"] == expected)


def near_top(cands, ret, a, window=NEAR):
    out = []
    for c in cands:
        if c["k"] < 0:
            if c["loss"] <= ret["loss"] + window:
                out.append({"loss": c["loss"], "status": "unplaced", "k": -1})
            continue
        if eligible(c) and c["loss"] <= ret["loss"] + window:
            out.append({"loss": c["loss"], "status": status_of(c["h"], a)[0], "k": c["k"]})
    return out


def cma_rd(s, a, X, y, rng, starts_n=CMA_STARTS, gens=CMA_GENS):
    """Independent search: CMA-ES (src/cmaes.py) on q = (w₁, b₁, w_noise) with b₂ profiled, starts uniform in the box
    [−W, W] × [0, 2π) × [−W, W]^{d−1} (every coordinate at full width)."""
    from .cmaes import cma_es
    d = X.shape[1]
    W = start_box(s, a, X[:, 0], y)

    def f(q):
        L = loss_grad_rd(q[None, :], s, a, X, y)[0][0]
        return float(L) if np.isfinite(L) else 1e9
    best = (math.inf, None)
    for _ in range(starts_n):
        q0 = np.r_[rng.uniform(-W, W), rng.uniform(0, TWO_PI), rng.uniform(-W, W, d - 1)]
        r = cma_es(f, q0, CMA_SIGMA0, max_generations=gens, seed=int(rng.integers(1 << 31)))
        if r.best_f < best[0]:
            best = (float(r.best_f), [float(v) for v in r.best_x])
    return {"loss": best[0], "q": best[1]}


def validate_end(s, a, X, y, rngs):
    """Ladder (200 then 800 restarts, independent streams), independent CMA-ES, audit on the 800-restart search."""
    r_small, _, _, _ = eval_status(s, a, X, y, LADDER[0], rngs[0])
    r_big, pol_big, c_big, W = eval_status(s, a, X, y, LADDER[1], rngs[1])
    cm = cma_rd(s, a, X, y, rngs[2])
    near = near_top(c_big, r_big, a) + [c for c in pol_big if c is not r_big]
    aud, n_other = audit_ok(r_big, near)
    return {"s": s, "status": r_big["status"], "loss": r_big["loss"], "G": r_big["G"],
            "h": None if r_big["h"] is None else [float(v) for v in r_big["h"]],
            "loss_ladder_small": r_small["loss"], "status_ladder_small": r_small["status"],
            "ladder_ok": ladder_ok(r_small, r_big), "cma_loss": cm["loss"],
            "independent_ok": independent_ok(r_big["loss"], cm["loss"]), "audit_ok": aud,
            "audit_n_other_within_tie": n_other, "audit_n_near_top": len(near),
            "n_eligible": sum(1 for c in c_big if eligible(c)), "box_W": W}


# ------------------------------------------------------------------------------------------ the bracket (pure driver)
def bracket_switch(status_fn, s_start, step=BRACKET_STEP, width=WIDTH, s_floor=S_FLOOR, s_ceil=S_CEIL):
    """own_threshold's bracketing with an arbitrary status function (True = placed): from s_start, step down by `step`
    while placed (or up while unplaced) until the status changes, then bisect to hi − lo ≤ width.  Returns
    (lo, hi, evals, note); lo or hi NaN with a note if the floor / ceiling is reached."""
    n = 0
    lo = hi = round(float(s_start), 6)
    p = status_fn(lo); n += 1
    if p:
        while p:
            hi = lo; lo = round(lo - step, 6); p = status_fn(lo); n += 1
            if p and lo <= s_floor:
                return float("nan"), hi, n, f"placed down to {s_floor}"
    else:
        while not p:
            lo = hi; hi = round(hi + step, 6); p = status_fn(hi); n += 1
            if not p and hi >= s_ceil:
                return lo, float("nan"), n, f"unplaced up to {s_ceil}"
    while hi - lo > width + 1e-12:
        mid = round(0.5 * (lo + hi), 6)
        if status_fn(mid):
            hi = mid
        else:
            lo = mid
        n += 1
    return lo, hi, n, ""


def _rng(*key):
    return np.random.default_rng([int(k) for k in key])


def own_switch_rd(seed, d, a, s_start):
    """The seed's own-sample R^d switch: bracket (RESTARTS restarts per scale), then validation at both ends."""
    X, y = make_data(seed, d)
    a100 = int(round(a * 100))
    evals = []

    def placed(s):
        r, _, _, _ = eval_status(s, a, X, y, RESTARTS, _rng(seed, d, a100, 1, len(evals)))
        evals.append({"s": s, "status": r["status"], "loss": r["loss"], "G": r["G"]})
        return r["status"] == "placed"
    t0 = time.time()
    lo, hi, n, note = bracket_switch(placed, s_start)
    out = {"seed": int(seed), "d": int(d), "a": a, "s_start": float(s_start), "s_lo": lo, "s_hi": hi,
           "s_own_d": 0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) else float("nan"), "evals": n,
           "note": note, "bracket_seconds": time.time() - t0, "scan": json.dumps(evals)}
    if note:
        out.update(validated=False)
        return out
    t1 = time.time()
    vlo = validate_end(lo, a, X, y, [_rng(seed, d, a100, 2, 0, j) for j in range(3)])
    vhi = validate_end(hi, a, X, y, [_rng(seed, d, a100, 2, 1, j) for j in range(3)])
    ok_lo, ok_hi = end_validated(vlo, "unplaced"), end_validated(vhi, "placed")
    out.update(validated=bool(ok_lo and ok_hi), validated_lo=ok_lo, validated_hi=ok_hi,
               validation_seconds=time.time() - t1, validation_lo=json.dumps(_jsonable(vlo)),
               validation_hi=json.dumps(_jsonable(vhi)),
               wnoise_l2_at_hi=float(np.linalg.norm(vhi["h"][3:])) if vhi["h"] is not None else float("nan"),
               w1_at_hi=vhi["h"][0] if vhi["h"] is not None else float("nan"))
    return out


def mirror_of(h):
    """The mirror half-space of a hidden configuration at w₂ > 0: '+' iff w₁ > 0."""
    return "+" if float(h[0]) > 0 else "-"


def _restricted_placed(seed, d, a, sigma, key=3):
    """Status function of the minimiser restricted to the mirror half-space sign(w₁) = sigma (at w₂ > 0): the same
    search with w₁ started in that half-space; only candidates ending in it are retained (the constant predictor
    included).  Used for the registered DESCRIPTIVE mirror switches (not validated; not scored)."""
    X, y = make_data(seed, d)
    k = [0]

    def f(s):
        rng = _rng(seed, d, int(round(a * 100)), key, int(sigma > 0), k[0]); k[0] += 1
        cands, _ = search_rd(s, a, X, y, RESTARTS, rng, sigma=sigma)
        ok = [c for c in cands if eligible(c) and np.sign(c["h"][0]) == sigma]
        _, pol = finish(ok + [cands[-1]], s, a, X, y)
        pol = [p for p in pol if p["h"] is None or np.sign(p["h"][0]) == sigma]
        return min(pol, key=lambda p: p["loss"])["status"] == "placed"
    return f


def mirror_switches(seed, d, a, g):
    """Registered DESCRIPTIVE inputs (not scored): the own-sample R^d switch of each mirror half-space.  If the
    validated global minimiser lies in one mirror at both bracket ends, that mirror's restricted minimiser is the global
    one there, so its switch is bracketed by the global bracket and s_mirror = s_own,d; the other mirror's switch is
    bracketed by `_restricted_placed` from the x₁-only threshold (steps of 0.1, width 0.01, midpoint).  Otherwise both
    mirrors are bracketed."""
    out = {}
    try:
        m_lo = mirror_of(json.loads(g["validation_lo"])["h"]) if json.loads(g["validation_lo"])["h"] else None
        m_hi = mirror_of(json.loads(g["validation_hi"])["h"])
    except (KeyError, TypeError):
        m_lo = m_hi = None
    out.update(global_mirror_lo=m_lo, global_mirror_hi=m_hi)
    todo = ("+", "-")
    if m_lo is not None and m_lo == m_hi and g.get("validated"):
        out["s_mirror_" + m_hi] = g["s_own_d"]
        todo = tuple(m for m in ("+", "-") if m != m_hi)
    t0 = time.time()
    for m in todo:
        lo, hi, n, note = bracket_switch(_restricted_placed(seed, d, a, 1.0 if m == "+" else -1.0), g["s_start"])
        out["s_mirror_" + m] = 0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) else float("nan")
        out["mirror_note_" + m] = note
    out["mirror_seconds"] = time.time() - t0
    return out


def own_x1(seed, a):
    """The seed's x₁-only own threshold (own_threshold.own_threshold, the width-1 global conditional search on the x₁
    sample; 3B's convention), bracket midpoint."""
    from .own_threshold import _pop, own_threshold
    r = own_threshold(a, int(seed), _pop(a)[1])
    return {"s_own_x1_lo": float(r["w2_lo"]), "s_own_x1_hi": float(r["w2_hi"]),
            "s_own_x1": 0.5 * (float(r["w2_lo"]) + float(r["w2_hi"])), "x1_evals": int(r["evals"]), "x1_note": r["note"]}


def frozen_seed_a(seed, a):
    """Frozen inputs of one seed at one a: the x₁-only own threshold, then s_own,d for d = 2 and 4 (each started from
    the x₁-only threshold)."""
    t0 = time.time()
    x1 = own_x1(seed, a)
    rows = []
    for d in D_VALUES:
        start = x1["s_own_x1"] if np.isfinite(x1["s_own_x1"]) else _pop_mid(a)
        r = own_switch_rd(seed, d, a, start)
        r.update(mirror_switches(seed, d, a, r))
        rows.append({**r, **x1, "x1_seconds": time.time() - t0})
    return rows


def _pop_mid(a):
    from .band_rd import s_bracket
    return s_bracket(a)[2]


# ------------------------------------------------------------------------------------------ frozen inputs
def timing():
    """Cost of the frozen inputs on one NON-registered seed (3B pilot seed 889,000; no training): both a, d = 2, 4."""
    os.nice(15)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in A_VALUES:
        t0 = time.time()
        for r in frozen_seed_a(TIMING_SEED, a):
            r["seconds_total_seed_a"] = time.time() - t0
            rows.append(r)
            print(json.dumps({k: r[k] for k in ("seed", "d", "a", "s_own_x1", "s_own_d", "evals", "validated",
                                                "bracket_seconds", "validation_seconds")}, default=float), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "timing.csv", index=False)


def freeze():
    """Per seed and a (in seed order): the frozen inputs, each seed-cell appended to frozen_parts.jsonl as it completes
    (resumable); the memory gate is re-checked between seeds."""
    os.nice(15)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "frozen_parts.jsonl"
    done = set() if not f.exists() else {(r["seed"], round(r["a"], 2)) for r in map(json.loads, f.read_text().splitlines())}
    for seed in SEEDS:
        for a in A_VALUES:
            if (seed, round(a, 2)) in done:
                continue
            wait_gate()
            t0 = time.time()
            rows = frozen_seed_a(seed, a)
            with open(f, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(_jsonable({**r, "seconds_seed_a": time.time() - t0})) + "\n")
            print(json.dumps({"seed": seed, "a": a, "s_own_x1": rows[0]["s_own_x1"],
                              "s_own_d": [r["s_own_d"] for r in rows], "validated": [r["validated"] for r in rows],
                              "seconds": round(time.time() - t0, 1)}, default=float), flush=True)
            rss_guard()
    write_frozen()


def write_frozen():
    d = pd.DataFrame([json.loads(l) for l in (OUT / "frozen_parts.jsonl").read_text().splitlines()])
    d["a"] = d.a.round(2)
    d = d.sort_values(["d", "a", "seed"])
    assert len(d) == len(SEEDS) * 4 and not d.duplicated(["seed", "d", "a"]).any(), "every seed-cell exactly once"
    cols = ["seed", "d", "a", "s_own_d", "s_lo", "s_hi", "validated", "validated_lo", "validated_hi", "note", "evals",
            "s_start", "s_own_x1", "s_own_x1_lo", "s_own_x1_hi", "x1_evals", "x1_note", "wnoise_l2_at_hi", "w1_at_hi",
            "global_mirror_lo", "global_mirror_hi", "s_mirror_+", "s_mirror_-", "mirror_seconds",
            "bracket_seconds", "validation_seconds", "validation_lo", "validation_hi", "scan"]
    d[cols].to_csv(OUT / "frozen_seeds.csv", index=False)
    print(len(d), "seed-cells;", int(d.validated.sum()), "validated;", int(d.s_own_d.isna().sum()), "undefined")


def kappa_table():
    """Frozen width-1 κ_k: lag_law.kappa_k(H, θ*′, ∇G, median P, k) from lag_law/kappa.csv, a = 1.30, 1.50, k = −3…3."""
    from .lag_law import kappa_k
    kt = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    rows = []
    for a in A_VALUES:
        r = kt[kt.a.round(2) == round(a, 2)].iloc[0]
        H, tan, dG, p = (np.array(json.loads(getattr(r, c))) for c in ("H", "tangent", "gradG", "p_median"))
        z = np.array(json.loads(r.z_star))
        for k in K_RANGE:
            rows.append({"a": a, "k": k, "kappa_k": float(kappa_k(H, tan, dG, p, k)), "z_star_b1": float(z[1]),
                         "z_star_w1": float(z[0]), "s_star": float(r.s_star)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "kappa_k.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def load_kappa():
    k = pd.read_csv(OUT / "kappa_k.csv")
    return {(round(r.a, 2), int(r.k)): float(r.kappa_k) for r in k.itertuples()}, \
        {round(r.a, 2): float(r.z_star_b1) for r in k.itertuples()}


def hashes():
    lines = [f"{_sha(ROOT / p)}  {p}" for p in REGISTERED_FILES]
    (OUT / "registration.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def assert_registration():
    for line in (OUT / "registration.sha256").read_text().splitlines():
        h, p = line.split()
        assert _sha(ROOT / p) == h, f"{p} changed since registration"


# ------------------------------------------------------------------------------------------ rules (pure functions)
def rule_step(s_path, s_own):
    """Track A's rule point (track_a.rule_step): the last upward passage of 0.5·s_own before s first reaches s_own."""
    from .track_a import rule_step as rs
    return rs(s_path, s_own, RULE_FRAC)


def t_switch(s_path, s_own):
    """t_sw: the first step t ≥ 1 with s_t ≥ s_own (None if never).  Output-scale trajectory only."""
    s = np.asarray(s_path, float)
    idx = np.nonzero(s[1:] >= s_own)[0]
    return None if len(idx) == 0 else int(idx[0] + 1)


def growth_at(s_path, t, window=WINDOW):
    """log(s_t / s_{t−w})/w with w = min(window, t) (Track A's window convention)."""
    w = min(window, t)
    return math.log(s_path[t] / s_path[t - w]) / w


def winding(b1, w2, z_b1):
    """3B's winding: k = round((canonical b₁ − b₁ of the width-1 switch point)/2π), canonical b₁ = b₁·sign(w₂)."""
    b1c = b1 if w2 > 0 else -b1
    return int(round((b1c - z_b1) / TWO_PI))


def continue_branch(a, X, y, p_rule, s_target, step=CONT_STEP):
    """The branch occupied at the rule point, continued in s: damped Newton in all hidden coordinates at w₂ = w₂(t_R)
    from the state at t_R (band_rd.branch_rd), then log-steps of at most 1% to s_target, each solved from the previous
    point.  Lost if a solve does not converge (gradient ≥ 1e−8) or the branch jumps (> 0.5 in any coordinate).
    Returns (h_rule, h_target, H_target, note)."""
    from .band_rd import branch_rd
    p_rule = np.asarray(p_rule, float)
    sg = 1.0 if p_rule[2] > 0 else -1.0
    h, H, g, conv = branch_rd(a, X, y, p_rule)
    if not conv:
        return None, None, None, "no converged branch at the rule point"
    h_rule = h.copy()
    s0 = abs(float(p_rule[2]))
    n = max(1, int(math.ceil(abs(math.log(s_target / s0)) / math.log1p(step))))
    for s in np.exp(np.linspace(math.log(s0), math.log(s_target), n + 1))[1:]:
        hn, H, g, conv = branch_rd(a, X, y, np.r_[h[0], h[1], sg * s, h[2], h[3:]])
        if not conv or np.abs(hn - h).max() > CONT_JUMP:
            return h_rule, None, None, f"continuation lost at s = {s:.5f}"
        h = hn
    return h_rule, h, H, ""


def predict_run(seed, d, a, W, V, s_own, kap, z_b1):
    """Registered prediction for one run from its path W (T+1, d+3) and bias-corrected Adam v̂ V (T+1, d+3), using
    only the state up to t_sw.  No gap or placement is evaluated."""
    from .residual_timescale import relax_rate
    out = {"seed": int(seed), "d": int(d), "a": a, "s_own_d": s_own}
    if not np.isfinite(s_own):
        return {**out, "status": "no frozen s_own,d"}
    s = np.abs(W[:, 2])
    t_sw = t_switch(s, s_own)
    t_r = rule_step(s, s_own)
    if t_sw is None or t_r is None:
        return {**out, "status": "s never reaches s_own,d"}
    out.update(t_rule=int(t_r), t_sw=int(t_sw), s_rule=float(s[t_r]), s_at_tsw=float(s[t_sw]))
    X, y = make_data(seed, d)
    h_rule, h, H, note = continue_branch(a, X, y, W[t_r], float(s[t_sw]))
    if h_rule is not None:
        out.update(k=winding(float(h_rule[1]), float(W[t_r, 2]), z_b1), branch_b1_rule=float(h_rule[1]),
                   mirror_rule="+" if float(h_rule[0]) * float(W[t_r, 2]) > 0 else "-",
                   branch_wnoise_l2_rule=float(np.linalg.norm(h_rule[3:])))
    if note:
        return {**out, "status": "branch rule: " + note}
    k = out["k"]
    if (round(a, 2), k) not in kap:
        return {**out, "status": f"winding {k} outside the frozen κ table"}
    v = V[t_sw]
    relax = relax_rate(H[:3, :3], v[[0, 1, 3]])
    relax_full = relax_rate(H, v[[0, 1, 3] + list(range(4, d + 3))])
    growth = growth_at(s, t_sw)
    chi = growth / relax if relax > 0 else float("nan")
    kappa = kap[(round(a, 2), k)]
    r_pred = kappa * chi
    s_pred = s_own * (1 + r_pred)
    out.update(growth=growth, relax=relax, relax_full=relax_full, chi=chi, kappa=kappa, r_pred=r_pred, s_pred=s_pred,
               branch_wnoise_l2_tsw=float(np.linalg.norm(h[3:])), branch_w1_tsw=float(h[0]),
               P_w1=float(1 / (math.sqrt(v[0]) + 1e-8)), P_b1=float(1 / (math.sqrt(v[1]) + 1e-8)),
               P_b2=float(1 / (math.sqrt(v[3]) + 1e-8)),
               k_state_tsw=winding(float(W[t_sw, 1]), float(W[t_sw, 2]), z_b1))
    ok = np.isfinite(r_pred) and np.isfinite(s_pred) and s_pred > 0
    out["status"] = "ok" if ok else "no finite prediction"
    return out


# ------------------------------------------------------------------------------------------ registered scoring (pure)
def bootstrap_mean_ci(D, n_boot=BOOT_N, seed=BOOT_SEED):
    """(mean, lower, upper): 95% percentile bootstrap interval of the mean of D (numpy default_rng(seed), n_boot
    resamples of size len(D) with replacement; the 2.5th and 97.5th percentiles of the resampled means)."""
    D = np.asarray(D, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    means = D[idx].mean(axis=1)
    lo, hi = np.percentile(means, BOOT_PCT)
    return float(D.mean()), float(lo), float(hi)


def score_cell(crossed, placed_at_init, rho_cross, s_obs, s_own, r_pred, s_x1, t_sw, step_obs):
    """Registered verdicts for one (d, a) cell.  Arrays over the cell's runs (NaN where undefined).
      crossing runs: crossed and not placed at initialisation.
      scored runs:   crossing runs with t_sw strictly before the crossing step and a finite prediction
                     (s_pred = s_own·(1 + r_pred) > 0) and a finite x₁-only own threshold.
      V1  ≥ 30 crossing runs.
      V2  ρ at crossing < 0.05 in ≥ 90% of crossing runs.
      V3  scored runs ≥ 90% of crossing runs (the prediction exists before the crossing).
      C1  median |log(s_obs/s_pred)| ≤ 0.05.
      C2  median of r_obs/r_pred ∈ [0.75, 1.25] (signed lags, r_obs = s_obs/s_own − 1).
      C3  D = |log(s_obs/s_pred)| − |log(s_obs/s_x1)|: upper end of the 95% percentile bootstrap interval of mean D < 0.
    Any validity condition failing makes C1–C3 UNRESOLVED; a statistic that cannot be computed (no scored run; C3
    with fewer than 2) is UNRESOLVED.  Outcome: PASS if all pass; UNRESOLVED if any UNRESOLVED; else FAIL naming each."""
    crossed = np.asarray(crossed, bool) & ~np.asarray(placed_at_init, bool)
    rho, s_obs, s_own, r_pred, s_x1, t_sw, step_obs = (np.asarray(v, float) for v in
                                                         (rho_cross, s_obs, s_own, r_pred, s_x1, t_sw, step_obs))
    s_pred = s_own * (1 + r_pred)
    n_cross = int(crossed.sum())
    tsw_before = crossed & np.isfinite(t_sw) & np.isfinite(step_obs) & (t_sw < step_obs)
    finite_pred = np.isfinite(s_pred) & (s_pred > 0) & np.isfinite(r_pred)
    scored = tsw_before & finite_pred & np.isfinite(s_x1) & (s_x1 > 0) & np.isfinite(s_obs)
    n = int(scored.sum())
    rho_ok = crossed & np.isfinite(rho) & (rho < RHO_MAX)
    frac_rho = float(rho_ok.sum() / n_cross) if n_cross else float("nan")
    frac_scored = float(n / n_cross) if n_cross else float("nan")
    v1 = n_cross >= MIN_CROSS
    v2 = bool(n_cross > 0 and frac_rho >= RHO_FRAC)
    v3 = bool(n_cross > 0 and frac_scored >= PRED_FRAC)
    valid = bool(v1 and v2 and v3)
    so, sp, sx, rp = s_obs[scored], s_pred[scored], s_x1[scored], r_pred[scored]
    err = np.abs(np.log(so / sp))
    ratio = (so / s_own[scored] - 1) / rp
    D = err - np.abs(np.log(so / sx))

    def verdict(ok, computable):
        if not valid or not computable:
            return "UNRESOLVED"
        return "PASS" if ok else "FAIL"
    med_err = float(np.median(err)) if n else float("nan")
    med_ratio = float(np.median(ratio)) if n else float("nan")
    mean_d, lo, hi = bootstrap_mean_ci(D) if n >= 2 else (float("nan"),) * 3
    out = {"n_runs": int(len(crossed)), "n_placed_at_init": int(np.asarray(placed_at_init, bool).sum()),
           "n_crossing": n_cross, "n_rho_below": int(rho_ok.sum()), "frac_rho_below": frac_rho,
           "n_tsw_before_crossing": int(tsw_before.sum()), "n_scored": n, "frac_scored": frac_scored,
           "n_crossing_unscored": n_cross - n,
           "validity": {"V1_min_crossings": bool(v1), "V2_rho": v2, "V3_prediction_before_crossing": v3},
           "valid": valid,
           "C1": {"n": n, "median_abs_log_err": med_err, "threshold": C1_MAX,
                  "verdict": verdict(med_err <= C1_MAX, n >= 1 and np.isfinite(med_err))},
           "C2": {"n": n, "median_ratio": med_ratio, "band": list(C2_BAND),
                  "verdict": verdict(C2_BAND[0] <= med_ratio <= C2_BAND[1], n >= 1 and not np.isnan(med_ratio))},
           "C3": {"n": n, "mean_D": mean_d, "ci95": [lo, hi], "comparator": "x1-only own threshold",
                  "verdict": verdict(hi < 0, n >= 2 and np.isfinite(hi))},
           "scored_mask": scored.tolist()}
    vs = [out[c]["verdict"] for c in ("C1", "C2", "C3")]
    out["outcome"] = ("UNRESOLVED" if "UNRESOLVED" in vs else "PASS" if all(v == "PASS" for v in vs)
                      else "FAIL " + "+".join(c for c in ("C1", "C2", "C3") if out[c]["verdict"] == "FAIL"))
    return out


def first_placed(G):
    """Index of the first step t ≥ 1 with G > 0 (None if never)."""
    idx = np.nonzero(np.asarray(G, float)[1:] > 0)[0]
    return None if len(idx) == 0 else int(idx[0] + 1)


# ------------------------------------------------------------------------------------------ training (no gap evaluated)
def train_one(seed, d, a, budget=BUDGET, p0=None):
    """Free Adam (lr 0.01, full batch, float64, all d + 3 parameters; band_rd.train_single's formulation) for the full
    budget, recording the parameters and Adam's bias-corrected v̂ at every step.  No gap or placement is evaluated.
    p0: tests only."""
    import torch
    Xn, Yn = make_data(seed, d)
    X, Y = torch.tensor(Xn), torch.tensor(Yn)
    p = (init_params(seed, d) if p0 is None else torch.tensor(np.asarray(p0, float))).clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=LR)
    n = d + 3
    W = np.empty((budget + 1, n)); V = np.full((budget + 1, n), np.nan)
    W[0] = p.detach().numpy()
    for t in range(1, budget + 1):
        opt.zero_grad(set_to_none=True)
        w = torch.cat([p[0:1], p[4:]])
        tt = (X * w[None, :]).sum(dim=-1) + p[1]
        z = p[2] * (tt + a * torch.sin(tt)) + p[3]
        torch.nn.functional.binary_cross_entropy_with_logits(z, Y).backward()
        opt.step()
        W[t] = p.detach().numpy()
        st = opt.state[p]
        V[t] = st["exp_avg_sq"].numpy() / (1 - 0.999 ** int(st["step"]))
    return W, V


def _path_file(seed, d, a):
    return PATHS / f"d{d}_a{a:.2f}_{seed}.npz"


def runs():
    return [(d, a, s) for d in D_VALUES for a in A_VALUES for s in SEEDS]


def train():
    """The 240 registered runs in fixed order (d, a, seed).  Each path is saved (paths/, untracked) and hashed; each
    prediction row is written as its run completes (resumable)."""
    os.nice(15)
    assert_registration()
    PATHS.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(OUT / "frozen_seeds.csv")
    fr["a"] = fr.a.round(2)
    fr = fr.set_index(["seed", "d", "a"])
    kap, zb1 = load_kappa()
    f = OUT / "predictions_parts.jsonl"
    done = set() if not f.exists() else {(r["d"], round(r["a"], 2), r["seed"]) for r in map(json.loads, f.read_text().splitlines())}
    for i, (d, a, seed) in enumerate(runs()):
        if (d, round(a, 2), seed) in done:
            continue
        if i % 20 == 0:
            wait_gate()
        W, V = train_one(seed, d, a)
        pf = _path_file(seed, d, a)
        np.savez(pf, W=W)
        h = _sha(pf)
        fz = fr.loc[(seed, d, round(a, 2))]
        s_own = float(fz.s_own_d) if bool(fz.validated) else float("nan")
        try:
            r = predict_run(seed, d, a, W, V, s_own, kap, zb1[round(a, 2)])
        except Exception as e:                                            # counted, not replaced
            r = {"seed": seed, "d": d, "a": a, "s_own_d": s_own, "status": f"error: {type(e).__name__}: {e}"}
        r.update(path_sha256=h, s_own_x1=float(fz.s_own_x1), s_own_d_validated=bool(fz.validated))
        r.update(mirror_matched(seed, d, a, W, V, r, fz, kap, zb1[round(a, 2)]))
        with open(f, "a") as fh:
            fh.write(json.dumps(_jsonable(r)) + "\n")
        print(json.dumps({k: r.get(k) for k in ("d", "a", "seed", "status", "t_rule", "t_sw", "k")}), flush=True)
        rss_guard()


def mirror_matched(seed, d, a, W, V, r, fz, kap, z_b1):
    """Registered DESCRIPTIVE prediction (not scored): the same rules with s_own,d replaced by the frozen switch of the
    mirror the run occupies at its (registered) rule point.  Identical to the registered prediction when that mirror is
    the global minimiser's."""
    m = r.get("mirror_rule")
    if m is None:
        return {"mm_status": "no rule-point mirror"}
    s_m = float(fz["s_mirror_" + m])
    if s_m == float(r.get("s_own_d", float("nan"))):
        return {"mm_same_as_registered": True, "mm_s_switch": s_m, **{"mm_" + k: r.get(k) for k in
                ("status", "t_rule", "t_sw", "chi", "kappa", "r_pred", "s_pred", "mirror_rule")}}
    try:
        q = predict_run(seed, d, a, W, V, s_m, kap, z_b1)
    except Exception as e:
        q = {"status": f"error: {type(e).__name__}: {e}"}
    return {"mm_same_as_registered": False, "mm_s_switch": s_m, **{"mm_" + k: q.get(k) for k in
            ("status", "t_rule", "t_sw", "chi", "kappa", "r_pred", "s_pred", "mirror_rule")}}


FORBIDDEN = {"crossed", "step_obs", "s_obs", "r_obs", "placed", "placed_at_init", "rho_cross"}


def finalize():
    d = pd.DataFrame([json.loads(l) for l in (OUT / "predictions_parts.jsonl").read_text().splitlines()])
    d["a"] = d.a.round(2)
    d = d.sort_values(["d", "a", "seed"])
    assert [tuple(x) for x in d[["d", "a", "seed"]].itertuples(index=False)] == \
        [(dd, round(aa, 2), s) for dd, aa, s in runs()], "every registered run exactly once"
    assert not (FORBIDDEN & set(d.columns))
    p = OUT / "predictions.csv"
    d.to_csv(p, index=False)
    h = _sha(p)
    (OUT / "predictions.sha256").write_text(f"{h}  predictions.csv\n")
    print(h, len(d), d.status.value_counts().to_dict())


# ------------------------------------------------------------------------------------------ observation and scoring
def _assert_committed(p, sha):
    h = _sha(p)
    assert h == sha.read_text().split()[0], "hash mismatch"
    rel = str(p.relative_to(ROOT))
    assert subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip(), f"{rel} not committed"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0, f"{rel} modified"
    return h


def observe_path(W, a):
    """Every-step detection on a saved path: the R^d gap at every step (band_rd.gap_rd, oriented by w₂)."""
    from .band_rd import gap_rd
    G = gap_rd(W[:, 0], W[:, 1], W[:, 4:], W[:, 2], a)
    t_c = first_placed(G)
    out = {"placed_at_init": bool(G[0] > 0), "crossed": bool(t_c is not None and not G[0] > 0), "step_obs": t_c}
    if out["crossed"]:
        q = W[t_c]
        out.update(s_obs=abs(float(q[2])), G_at_cross=float(G[t_c]), rho_cross=float(np.linalg.norm(q[4:]) / abs(q[0])),
                   wnoise_l2_cross=float(np.linalg.norm(q[4:])), w1_cross=float(q[0]), b1_cross=float(q[1]),
                   w2_cross=float(q[2]))
    return out


def observe():
    os.nice(15)
    assert_registration()
    _assert_committed(OUT / "predictions.csv", OUT / "predictions.sha256")
    pr = pd.read_csv(OUT / "predictions.csv")
    parts = OUT / "observed_parts.jsonl"
    done = set() if not parts.exists() else {(r["d"], round(r["a"], 2), r["seed"]) for r in
                                             map(json.loads, parts.read_text().splitlines())}
    for r in pr.itertuples():
        if (r.d, round(r.a, 2), r.seed) in done:
            continue
        pf = _path_file(r.seed, r.d, r.a)
        assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
        W = np.load(pf)["W"]
        row = {"seed": int(r.seed), "d": int(r.d), "a": float(r.a), **observe_path(W, r.a)}
        with open(parts, "a") as fh:
            fh.write(json.dumps(_jsonable(row)) + "\n")
        rss_guard()
    return score()


def score():
    """Registered scoring (results/track2c_registration.md §7) and the registered descriptives (§8)."""
    pr = pd.read_csv(OUT / "predictions.csv")
    o = pd.DataFrame([json.loads(l) for l in (OUT / "observed_parts.jsonl").read_text().splitlines()])
    for t in (pr, o):
        t["a"] = t.a.round(2)
    fz = pd.read_csv(OUT / "frozen_seeds.csv"); fz["a"] = fz.a.round(2)
    m = pr.merge(o, on=["seed", "d", "a"], how="left").merge(fz[["seed", "d", "a", "global_mirror_hi"]],
                                                            on=["seed", "d", "a"], how="left")
    assert len(m) == len(runs()) and m.crossed.notna().all()
    m["r_obs"] = m.s_obs / m.s_own_d - 1
    m["s_pred_floor"] = np.where(np.isfinite(m.s_pred), m.s_pred, m.s_own_d)
    cells, rows = {}, []
    for (d, a), g in m.groupby(["d", "a"]):
        sc = score_cell(g.crossed.astype(bool), g.placed_at_init.astype(bool), g.rho_cross, g.s_obs, g.s_own_d,
                        g.r_pred, g.s_own_x1, g.t_sw, g.step_obs)
        mask = np.array(sc.pop("scored_mask"), bool)
        m.loc[g.index, "scored"] = mask
        desc = describe(g, mask)
        mm = score_cell(g.crossed.astype(bool), g.placed_at_init.astype(bool), g.rho_cross, g.s_obs, g.mm_s_switch,
                        g.mm_r_pred, g.s_own_x1, g.mm_t_sw, g.step_obs)
        mm.pop("scored_mask")
        desc["mirror_matched_NOT_SCORED"] = mm
        cr = g[g.crossed.astype(bool) & ~g.placed_at_init.astype(bool)]
        desc["n_crossing_rule_mirror_is_global_mirror"] = int((cr.mirror_rule == cr.global_mirror_hi).sum())
        desc["n_crossing_with_rule_mirror"] = int(cr.mirror_rule.notna().sum())
        cells[f"d{d}_a{a:.2f}"] = {"d": int(d), "a": a, **sc, "descriptive": desc}
    m["scored"] = m.scored.astype(bool)
    m.to_csv(OUT / "observed_runs.csv", index=False)
    res = {"cells": cells, "predictions_sha256": (OUT / "predictions.sha256").read_text().split()[0],
           "registration_sha256_file_sha256": _sha(OUT / "registration.sha256")}
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: {c: v[c] for c in ("n_crossing", "n_scored", "valid", "outcome")} | {
        c: v[c]["verdict"] for c in ("C1", "C2", "C3")} for k, v in cells.items()}, indent=1, default=float))
    return res


def describe(g, mask):
    """Registered descriptives (not scored)."""
    q = lambda v, p: float(np.nanpercentile(np.asarray(v, float), p)) if len(v) and np.isfinite(np.asarray(v, float)).any() else float("nan")
    c = g[g.crossed.astype(bool) & ~g.placed_at_init.astype(bool)]
    S = g[mask]
    out = {"n_not_crossed": int((~g.crossed.astype(bool) & ~g.placed_at_init.astype(bool)).sum()),
           "status_counts": g.status.value_counts().to_dict(),
           "median_s_own_d_over_x1": q(g.s_own_d / g.s_own_x1, 50),
           "q10_q90_s_own_d_over_x1": [q(g.s_own_d / g.s_own_x1, 10), q(g.s_own_d / g.s_own_x1, 90)],
           "rho_cross_median": q(c.rho_cross, 50), "rho_cross_max": q(c.rho_cross, 100),
           "n_crossing_below_s_own_d": int((c.s_obs < c.s_own_d).sum())}
    if len(S):
        so = S.s_obs.to_numpy(float)
        x1lag = S.s_own_x1 * (1 + S.r_pred)
        Dx = np.abs(np.log(so / S.s_pred)) - np.abs(np.log(so / x1lag))
        mD, lo, hi = bootstrap_mean_ci(Dx) if len(S) >= 2 else (float("nan"),) * 3
        out.update(median_r_obs=q(S.r_obs, 50), median_r_pred=q(S.r_pred, 50), median_chi=q(S.chi, 50),
                   median_kappa=q(S.kappa, 50), winding_counts={str(k): int(v) for k, v in
                                                               S.k.value_counts().sort_index().items()},
                   median_abs_log_err_x1_only=q(np.abs(np.log(so / S.s_own_x1)), 50),
                   median_abs_log_err_static_s_own_d=q(np.abs(np.log(so / S.s_own_d)), 50),
                   stricter_comparator_x1_plus_lag={"mean_D": mD, "ci95": [lo, hi],
                                                    "median_abs_log_err": q(np.abs(np.log(so / x1lag)), 50)},
                   spearman_r_pred_r_obs=float(pd.Series(S.r_pred.to_numpy(float)).rank().corr(
                       pd.Series(S.r_obs.to_numpy(float)).rank())) if len(S) >= 3 else float("nan"),
                   median_steps_tsw_to_crossing=q(S.step_obs - S.t_sw, 50),
                   median_steps_rule_to_crossing=q(S.step_obs - S.t_rule, 50))
    if len(c):
        out["all_crossing_runs_floor_median_abs_log_err"] = q(np.abs(np.log(c.s_obs / c.s_pred_floor)), 50)
    return out


# ------------------------------------------------------------------------------------------ PRE-REGISTRATION DIAGNOSTIC
# NON-REGISTERED.  Run on Track 3B's committed primary seeds (880,000-880,005; not 2C seeds), no training: does the
# global own-sample R^d switch (the design's s_own,d) equal the switch of the branch each 3B run actually followed
# (results/band_rd/posthoc_branch_switch.csv)?  And does the switch of the minimiser restricted to the run's mirror
# half-space (sign of w₁ at w₂ > 0) equal it?
def _diag_restricted_placed(seed, d, a, sigma):
    """The diagnostic's restricted status (as run for diag_mirror_3b.csv): unrestricted starts, candidates filtered to
    sign(w₁) = sigma, the constant predictor not eligible."""
    X, y = make_data(seed, d)
    k = [0]

    def f(s):
        rng = _rng(seed, d, int(round(a * 100)), 7, int(sigma > 0), k[0]); k[0] += 1
        cands, _ = search_rd(s, a, X, y, RESTARTS, rng)
        ok = [c for c in cands if eligible(c) and np.sign(c["h"][0]) == sigma]
        _, pol = finish(ok + [cands[-1]], s, a, X, y)
        pol = [p for p in pol if p["h"] is not None and np.sign(p["h"][0]) == sigma]
        return min(pol, key=lambda p: p["loss"])["status"] == "placed"
    return f


def diag_mirror():
    os.nice(15)
    R = pd.DataFrame([json.loads(l) for l in (RESULTS / "band_rd" / "runs.jsonl").read_text().splitlines()])
    R = R[R.arm == "primary"]; R["a"] = R.a.round(2)
    b = pd.read_csv(RESULTS / "band_rd" / "posthoc_branch_switch.csv"); b = b[b.arm == "primary"]; b["a"] = b.a.round(2)
    o = pd.read_csv(RESULTS / "band_rd" / "own_x1_frozen.csv"); o["a"] = o.a.round(2)
    rows = []
    for seed in range(880_000, 880_006):
        for d in D_VALUES:
            for a in A_VALUES:
                a = round(a, 2)
                s0 = float(o[(o.a == a) & (o.seed == seed)].w2_own.iloc[0])
                g = own_switch_rd(seed, d, a, s0)
                out = {"label": "NON-REGISTERED pre-registration diagnostic (3B seeds)", "seed": seed, "d": d, "a": a,
                       "s_own_x1": s0, "s_global": g["s_own_d"], "global_validated": g["validated"],
                       "global_mirror": ("+" if json.loads(g["validation_hi"])["h"][0] > 0 else "-")
                       if g.get("validation_hi") else None}
                for sg in (1.0, -1.0):
                    lo, hi, n, note = bracket_switch(_diag_restricted_placed(seed, d, a, sg), s0)
                    out["s_mirror_" + ("+" if sg > 0 else "-")] = 0.5 * (lo + hi)
                r = R[(R.d == d) & (R.a == a) & (R.seed == seed)].iloc[0]
                if r.crossed:
                    p = np.array(json.loads(r.p_cross))
                    out["run_mirror"] = "+" if p[0] * np.sign(p[2]) > 0 else "-"
                    out["s_branch_3B"] = float(b[(b.d == d) & (b.a == a) & (b.seed == seed)].s_branch.iloc[0])
                    out["global_over_branch"] = out["s_global"] / out["s_branch_3B"]
                    out["run_mirror_switch_over_branch"] = out["s_mirror_" + out["run_mirror"]] / out["s_branch_3B"]
                rows.append(out)
                print(json.dumps(_jsonable(out)), flush=True)
                pd.DataFrame(rows).to_csv(OUT / "diag_mirror_3b.csv", index=False)


if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    {"diag_mirror": diag_mirror, "timing": timing,"freeze": freeze, "write_frozen": write_frozen, "kappa": kappa_table, "hashes": hashes,
     "train": train, "finalize": finalize, "observe": observe, "score": score}[sys.argv[1]]()
