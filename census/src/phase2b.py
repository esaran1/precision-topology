"""Phase 2B: the lever (the output learning rate) on the simplicity-bias benchmark (design:
results/designs/phase2b_lever_design.md, approved by the author 2026-10-02; registration: results/phase2b_registration.md;
registered after Phase 2A's result, registration 569b836, result 7b15d9f).

Question: does slowing the OUTPUT weights' Adam learning rate (not the step count) change what training learns, at matched
scale and at matched training loss, while slab use still starts above the switch s*?

Setting (v3, unchanged): the 800 points of src/simplicity_bias_v2.data(); tanh width 4; BCE (mean) + (λ/2)(|W|² + |c|²),
λ = 1e-4; float64; full batch; s = |v|₁; q = 0.3914; s* = 3.5914.  Init: src/simplicity_bias_v3.init_net (PyTorch default
init after torch.manual_seed(seed), output weights capped at 0.5·s*) called inside torch.random.fork_rng, so the global
torch RNG state is restored; no warm start, no idle-unit zeroing.  Adam (0.9, 0.999, 1e-8) in numpy (torch.optim.Adam's
formula; tested), per-parameter learning rates.  Every run trains T_C = 40,000 steps; ρ₂ and BCE at every step.

Arms (shared seeds 2,962,000-2,962,079):
  std     arm 1   lr 0.01 on everything                                    reference
  out16   arm 2   lr 0.01/16 on v (4 output weights); 0.01 on W, c, b     PRIMARY: R_s, R_l-rho2, R_l-acc, O
  out16b  arm 2b  lr 0.01/16 on v and b; 0.01 on W, c                     SECONDARY: its own verdict, same criteria
  glob    arm 3   lr 0.01/12.18 on everything                             N
  globcm  arm 3cm lr 0.01/16.14 on everything (cost-matched)              N

STRICT CAUSAL RULE: no criterion is a forecast.  Every criterion compares OBSERVED outcomes after every run has ended
(score() refuses an incomplete set of runs), so no cutoff applies.  Nothing here imports or calls a forecaster (tested).

ORDER:
  scan / freeze / manifest    before the registration commit (no registered seed drawn).  freeze reproduces committed
                              exploration runs (non-registered seeds 2,953,000+) and recomputes the global factors.
  run                         after the registration commit and its timestamp: per seed (in order), arm 1 first (the
                              matched step reads its step to 3 s*), then 2, 2b, 3, 3cm; resumable.  -> runs.jsonl
  score                       after all 400 runs.  -> scores.json

Machine rules: one process, nice 15, one thread; a memory gate (free ≥ 25%, swap free ≥ 500 MB; waits) and a disk check
(free ≥ 20 GB; else STOP) before every job, logged to results/phase2b/memory_gate.log; stop above 3 GB RSS.  No global
numpy RNG; bootstraps use local numpy Generators; the init runs inside torch.random.fork_rng.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
        nice -n 15 .venv/bin/python -m src.phase2b scan | freeze | manifest | run | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import simplicity_bias_v2 as v2
from . import simplicity_bias_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase2b"
REGISTRATION_MD = RESULTS / "phase2b_registration.md"
DESIGN_MD = RESULTS / "designs" / "phase2b_lever_design.md"
EXPLORE = RESULTS / "designs" / "phase2b_explore"
V2_FROZEN = RESULTS / "simplicity_bias_v2" / "frozen.json"
P2A_FROZEN = RESULTS / "phase2a" / "frozen.json"

# ------------------------------------------------------------------------------------------ registered constants
LAM = 1e-4
S_STAR = 3.5913755424683727                     # v2 frozen s_q (checked in freeze and tests)
Q = 0.3914103370353161                          # v2 frozen q
S_FOLD = 4.767689442106793                      # M's fold s_F (phase2a/frozen.json); DESCRIPTIVE onset location only
CAP = 0.5 * S_STAR
ADAM_LR, B1, B2, EPS = 0.01, 0.9, 0.999, 1e-8
T_C = 40_000                                    # every run trains exactly T_C steps (states 0 … T_C)
G_GLOB = 12.18                                  # ‡ arm 3: median over the exploration seeds of arm-2 ÷ arm-1 steps to 3 s*
G_CM = 16.14                                    # ‡ arm 3cm: 12.18^(ln 12.177 / ln 9.454), cost-matched to arm 2
OUT_FACTOR = 16.0                               # arms 2, 2b
SEEDS = tuple(range(2_962_000, 2_962_080))      # 80 registered seeds, shared by all arms
EXPLORE_SEEDS = tuple(range(2_953_000, 2_953_020))   # non-registered exploration seeds (never to be registered)
ARMS = ("std", "out16", "out16b", "glob", "globcm")   # run order per seed (arm 1 first: the matched step reads it)
ARM_LABEL = {"std": "1", "out16": "2", "out16b": "2b", "glob": "3", "globcm": "3cm"}
OUTPUT_ARMS = ("out16", "out16b")               # R_s, R_l-rho2, R_l-acc, O
GLOBAL_ARMS = ("glob", "globcm")                # N
PRIMARY, SECONDARY = "out16", "out16b"
SCALE_LEVELS = (("scale_1.25s*", 1.25), ("scale_2s*", 2.0), ("scale_3s*", 3.0))      # s_m = factor · s*
BCE_LEVELS = (("bce_0.1", 0.1), ("bce_0.03", 0.03), ("bce_0.01", 0.01))             # ‡ ℓ
POINTS = tuple(k for k, _ in SCALE_LEVELS) + tuple(k for k, _ in BCE_LEVELS)
QTY = ("rho2", "acc_shuffled", "acc_reversed")
CELLS = tuple(f"{p}|{q}" for p in POINTS for q in QTY)                  # 18 cells, point-major
CRITERION_CELLS = {
    "R_s": tuple(f"{p}|{q}" for p, _ in SCALE_LEVELS for q in QTY),                 # 9
    "R_l_rho2": tuple(f"{p}|rho2" for p, _ in BCE_LEVELS),                         # 3
    "R_l_acc": tuple(f"{p}|{q}" for p, _ in BCE_LEVELS for q in ("acc_shuffled", "acc_reversed")),   # 6
    "N": CELLS}                                                                     # 18
DELTA = {"rho2": 0.03, "acc_shuffled": 0.02, "acc_reversed": 0.03}      # ‡ N margins (one-sided)
O_MIN_FRAC = 0.75                               # ‡ O: ≥ 75% of runs with s(t_on) > s*
REACH_MIN_FRAC = 0.90                           # UNRESOLVED if < 90% of seeds reach a cell in both arms
BOOT_N, BOOT_SEED, BOOT_PCT = 10_000, 20261002, (2.5, 97.5)
NO_CRITERION_IS_A_FORECAST = True               # every criterion compares observed outcomes after all runs end
CHUNK = 250                                     # ρ₂ evaluated in chunks of 250 states (as the exploration)
TRAJ_POINTS = 64
RSS_LIMIT = 3 * 1024 ** 3
DISK_MIN_BYTES = 20 * 1024 ** 3
ARTIFACTS = ("phase2b/seed_scan.json", "phase2b/frozen.json", "phase2b/registration.sha256", "phase2b/runs.jsonl",
             "phase2b/scores.json")

X, Y = v2.data()
N = len(Y)
U1, I1 = np.unique(X[:, 0], return_inverse=True)
U2, I2 = np.unique(X[:, 1], return_inverse=True)
W1 = np.bincount(I1).astype(float)              # multiplicity of each x₁ value
W2 = np.bincount(I2).astype(float)
NEG1 = np.searchsorted(U1, -U1)                 # index of −x₁ (the x₁ level set is exactly symmetric)
assert np.array_equal(U1[NEG1], -U1)
YB = Y.astype(bool)


def lr_vec(arm):
    """Per-parameter Adam learning rates over the 17 training coordinates (W 8, c 4, v 4, b)."""
    lr = np.full(17, ADAM_LR)
    if arm == "std":
        pass
    elif arm == "out16":
        lr[12:16] /= OUT_FACTOR
    elif arm == "out16b":
        lr[12:17] /= OUT_FACTOR
    elif arm == "glob":
        lr /= G_GLOB
    elif arm == "globcm":
        lr = np.full(17, ADAM_LR / G_CM)
    else:
        raise ValueError(arm)
    return lr


# ------------------------------------------------------------------------------------------ model
def loss_grad(th):
    """Objective BCE + (λ/2)(|W|² + |c|²) of z = tanh(XWᵀ + c)·v + b and its analytic gradient."""
    W = th[:8].reshape(4, 2); c = th[8:12]; v = th[12:16]; b = th[16]
    H = np.tanh(X @ W.T + c)
    z = H @ v + b
    L = float(np.mean(np.logaddexp(0.0, z) - Y * z) + 0.5 * LAM * ((W ** 2).sum() + (c ** 2).sum()))
    r = (0.5 * (1 + np.tanh(0.5 * z)) - Y) / N
    D = (1 - H ** 2) * (r[:, None] * v[None, :])
    g = np.empty(17)
    g[:8] = (D.T @ X + LAM * W).ravel()
    g[8:12] = D.sum(0) + LAM * c
    g[12:16] = H.T @ r
    g[16] = r.sum()
    return L, g


def bce(L, th):
    """The data loss (mean BCE): the objective minus (λ/2)(|W|² + |c|²)."""
    return L - 0.5 * LAM * ((th[:8] ** 2).sum() + (th[8:12] ** 2).sum())


def init_row(seed):
    """v3.init_net(seed, cap 0.5·s*) inside torch.random.fork_rng (the global torch RNG state is restored afterwards)."""
    import torch
    with torch.random.fork_rng(devices=[]):
        hid, out, s0 = v3.init_net(seed, CAP, torch)
    th = np.concatenate([hid.weight.detach().numpy().ravel(), hid.bias.detach().numpy(),
                         out.weight.detach().numpy().ravel(), out.bias.detach().numpy()]).astype(float)
    return th, float(s0)


def adam_update(th, g, m, v, k, lr):
    """One torch.optim.Adam step (k = 1, 2, …): m ← m + (1 − β₁)(g − m); v ← β₂v + (1 − β₂)g²;
    θ ← θ − (lr/(1 − β₁ᵏ))·m/(√v/√(1 − β₂ᵏ) + ε).  Returns (θ, m, v)."""
    m = m + (1 - B1) * (g - m)
    v = v * B2 + (1 - B2) * g * g
    th = th - (lr / (1 - B1 ** k)) * m / (np.sqrt(v) / math.sqrt(1 - B2 ** k) + EPS)
    return th, m, v


# ------------------------------------------------------------------------------------------ measurements
def grid_z(P):
    """Network output without the bias on the 40 × 40 grid of unique (x₁, x₂) values: (m, 40, 40)."""
    W = P[:, :8].reshape(-1, 4, 2); c = P[:, 8:12]; v = P[:, 12:16]
    A = W[:, None, None, :, 0] * U1[None, :, None, None] + W[:, None, None, :, 1] * U2[None, None, :, None]
    return np.einsum("mabk,mk->mab", np.tanh(A + c[:, None, None, :]), v)


def rho2_rows(P):
    """ρ₂ = V₂/(V₁ + V₂) for parameter rows (m, 17), V_j = (1/N²) Σ_i Σ_t mult(t)|φ(x_i, x_j := t) − φ(x_i)| (v2/v3)."""
    Z = grid_z(P)
    base = Z[:, I1, I2]
    V1 = np.einsum("a,man->m", W1, np.abs(Z[:, :, I2] - base[:, None, :])) / N ** 2
    V2 = np.einsum("b,mbn->m", W2, np.abs(np.transpose(Z, (0, 2, 1))[:, :, I1] - base[:, None, :])) / N ** 2
    return V2 / (V1 + V2)


def metrics(th):
    """At one state: s, objective, BCE, ρ₂, G₊, train accuracy, and the two shifted accuracies.
    shuffled: x₁ replaced by every x₁ of the 800 points (the exact expectation over a uniform permutation of x₁).
    reversed: x₁ → −x₁ (swaps the classes' x₁ distributions exactly; x₂ unchanged).  Prediction: z + b > 0."""
    P = th[None]
    Z = grid_z(P)[0]
    b = th[16]
    base = Z[I1, I2]
    sv = np.abs(th[12:16]).sum()
    corr = lambda z: (z + b > 0) == YB                                                    # noqa: E731
    shuf = (W1[:, None] * corr(Z[:, I2])).sum() / (W1.sum() * N)
    rev = corr(Z[NEG1[I1], I2]).mean()
    L, _ = loss_grad(th)
    return {"s": float(sv), "loss": L, "rho2": float(rho2_rows(P)[0]),
            "gplus": float(0.5 * (base[YB].min() - base[~YB].max()) / sv),
            "acc_train": float(corr(base).mean()), "acc_shuffled": float(shuf), "acc_reversed": float(rev),
            "bce": float(bce(L, th))}


def lasting_onset(R, S, end):
    """t_on = the first step such that ρ₂ ≥ q at EVERY step in [t_on, end]; None if ρ₂(end) < q (no such step).
    Returns {t, s, s_over_s_star, s_over_s_F, from_init} or None."""
    r = np.asarray(R[:end + 1])
    if not len(r) or not np.isfinite(r).all() or r[-1] < Q:
        return None
    below = np.nonzero(r < Q)[0]
    t_on = int(below[-1] + 1) if len(below) else 0
    s = float(S[t_on])
    return {"t": t_on, "s": s, "s_over_s_star": s / S_STAR, "s_over_s_F": s / S_FOLD, "from_init": t_on == 0}


def _path_sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a, dtype="<f8")).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ one run
def run_one(arm, seed, t_match=None, T=T_C, extra_points=None):
    """One run: Adam with lr_vec(arm) for T steps from v3's init of `seed`.  Records the first passage (no
    interpolation) of each matched scale (s ≥ factor·s*) and BCE level (BCE ≤ ℓ) with the metrics there, the state at
    `t_match` (arm 1's step to 3 s*: the DESCRIPTIVE matched step), the endpoint at T, the lasting onset to T, a
    non-finite check at every step, and the SHA-256 of the (s, BCE, ρ₂) paths.  A non-finite state stops the run
    (finite = False).  extra_points: {name: (kind, level)} additional first passages (exploration reproduction only)."""
    lr = lr_vec(arm)
    th, s_raw = init_row(seed)
    m = np.zeros(17); v = np.zeros(17)
    S = np.full(T + 1, np.nan); B = np.full(T + 1, np.nan); R = np.full(T + 1, np.nan)
    levels = [(k, "scale", f * S_STAR) for k, f in SCALE_LEVELS] + [(k, "bce", lv) for k, lv in BCE_LEVELS]
    for k, (kind, lv) in (extra_points or {}).items():
        levels.append((k, kind, lv))
    t_first, keep = {}, {}
    buf, buf_t = [], []
    t_nonfinite = None
    t0 = time.time()

    def flush():
        if buf:
            R[buf_t[0]:buf_t[-1] + 1] = rho2_rows(np.array(buf)); buf.clear(); buf_t.clear()
            rss_guard()

    t_last = -1
    for t in range(T + 1):
        if not np.isfinite(th).all():
            t_nonfinite = t
            break
        L, g = loss_grad(th)
        s = float(np.abs(th[12:16]).sum()); b_ = bce(L, th)
        if not (np.isfinite(L) and np.isfinite(g).all()):
            t_nonfinite = t
            break
        S[t] = s; B[t] = b_; buf.append(th.copy()); buf_t.append(t); t_last = t
        for k, kind, lv in levels:
            if k not in t_first and ((kind == "scale" and s >= lv) or (kind == "bce" and b_ <= lv)):
                t_first[k] = t; keep[k] = th.copy()
        if t_match is not None and t == t_match:
            keep["match_step"] = th.copy()
        if t == T:
            keep["end_T_C"] = th.copy()
        if len(buf) >= CHUNK:
            flush()
        if t == T:
            break
        th, m, v = adam_update(th, g, m, v, t + 1, lr)
    flush()
    if t_nonfinite is None and not np.isfinite(R[:t_last + 1]).all():
        t_nonfinite = int(np.nonzero(~np.isfinite(R[:t_last + 1]))[0][0])
    finite = t_nonfinite is None
    at = {k: metrics(p) for k, p in keep.items()}
    if not all(np.isfinite(list(d.values())).all() for d in at.values()):
        finite = False
        t_nonfinite = t_nonfinite if t_nonfinite is not None else -1
    lev = {k: lv for k, _, lv in levels}
    idx = np.unique(np.r_[0, np.geomspace(1, T, TRAJ_POINTS).astype(int)])
    rec = {"arm": arm, "arm_label": ARM_LABEL[arm], "seed": int(seed), "lr": lr.tolist(), "T": int(T),
           "s_init_raw": s_raw, "finite": bool(finite), "t_nonfinite": t_nonfinite, "t_match": t_match,
           "t_first": t_first,
           "overshoot": {k: float(S[t] / lev[k] - 1) for k, t in t_first.items() if k.startswith("scale")},
           "s_at_first": {k: float(S[t]) for k, t in t_first.items()},
           "at": at,
           "onset": lasting_onset(R, S, T) if finite else None,
           "rho2_init": float(R[0]),
           "path_sha256": _path_sha(S, B, R),
           "traj": {"t": idx.tolist(), "s": S[idx].tolist(), "bce": B[idx].tolist(), "rho2": R[idx].tolist()},
           "secs": round(time.time() - t0, 2), "max_rss_gb": _rss() / 1e9}
    return rec, (S, B, R)


# ------------------------------------------------------------------------------------------ pure scoring rules
def verdict(ok, resolved=True):
    return "UNRESOLVED" if not resolved else ("PASS" if ok else "FAIL")


def paired(rows_arm, rows_ref, cells=CELLS):
    """(n_seeds, n_cells) paired differences arm − arm 1 on the same seed (rows aligned by seed); NaN where a seed does
    not reach the cell's point in both arms."""
    assert [r["seed"] for r in rows_arm] == [r["seed"] for r in rows_ref], "rows must be aligned by seed"
    D = np.full((len(rows_arm), len(cells)), np.nan)
    for i, (a, b) in enumerate(zip(rows_arm, rows_ref)):
        for j, c in enumerate(cells):
            p, q = c.split("|")
            x, y = (a.get("at") or {}).get(p), (b.get("at") or {}).get(p)
            if x is not None and y is not None:
                D[i, j] = x[q] - y[q]
    return D


def boot_ci_median(D, n_boot=None, seed=None):
    """95% percentile bootstrap interval of the median paired difference per column: seeds (rows) resampled jointly
    across all columns (one index matrix from a FRESH default_rng(seed)); the median over the resampled seeds that reach
    the cell (nanmedian); percentiles 2.5 and 97.5 (numpy's linear rule).  Returns (lo, hi, median)."""
    D = np.asarray(D, float)
    n_boot = BOOT_N if n_boot is None else int(n_boot)
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(D[idx], axis=1)
            lo, hi = np.percentile(med, BOOT_PCT[0], axis=0), np.percentile(med, BOOT_PCT[1], axis=0)
            point = np.nanmedian(D, axis=0)
    return lo, hi, point


def reach_ok(D, cols, n_seeds):
    """The 90% reach rule for a criterion: every one of its cells is reached in BOTH arms by ≥ 90% of the seeds."""
    n_both = np.isfinite(np.asarray(D, float)[:, cols]).sum(axis=0)
    return bool((n_both >= REACH_MIN_FRAC * n_seeds).all()), [int(x) for x in n_both]


def criterion_lower(D, lo, crit, n_seeds, any_nonfinite):
    """R_s / R_l-rho2 / R_l-acc: PASS if every cell's lower end > 0; UNRESOLVED (overriding PASS and FAIL) if a cell is
    reached in both arms by < 90% of the seeds, or a run of an arm it uses (this arm or arm 1) is non-finite
    (`nonfinite_used`; author's decision D5, 2026-10-05); else FAIL."""
    cols = [CELLS.index(c) for c in CRITERION_CELLS[crit]]
    ok_reach, n_both = reach_ok(D, cols, n_seeds)
    lo_c = np.asarray(lo, float)[cols]
    ok = bool(np.all(lo_c > 0))
    return {"criterion": crit, "cells": list(CRITERION_CELLS[crit]), "lower": lo_c.tolist(), "n_both": n_both,
            "reach_ok": ok_reach, "nonfinite_in_used_arms": bool(any_nonfinite), "all_lower_gt_0": ok,
            "verdict": verdict(ok, ok_reach and not any_nonfinite)}


def criterion_N(D, hi, n_seeds, any_nonfinite):
    """N (a global arm): PASS if every one of the 18 cells' upper end < δ (0.03 for ρ₂ and reversed accuracy, 0.02 for
    shuffled accuracy); UNRESOLVED as for R (reach; a non-finite run of this arm or of arm 1); else FAIL."""
    cols = [CELLS.index(c) for c in CRITERION_CELLS["N"]]
    ok_reach, n_both = reach_ok(D, cols, n_seeds)
    marg = np.array([DELTA[c.split("|")[1]] for c in CRITERION_CELLS["N"]])
    hi_c = np.asarray(hi, float)[cols]
    ok = bool(np.all(hi_c < marg))
    return {"criterion": "N", "cells": list(CRITERION_CELLS["N"]), "upper": hi_c.tolist(), "delta": marg.tolist(),
            "n_both": n_both, "reach_ok": ok_reach, "nonfinite_in_used_arms": bool(any_nonfinite), "all_upper_lt_delta": ok,
            "verdict": verdict(ok, ok_reach and not any_nonfinite)}


def criterion_O(onset_s, n_runs, any_nonfinite):
    """O (an output arm): t_on = the first step with ρ₂ ≥ q at every step to T_C.  PASS if ≥ 75% of the arm's runs have
    s(t_on) > s*; a run with no such step (onset None) counts against.  UNRESOLVED if a run of this arm or of arm 1
    is non-finite (D5: every criterion compares against arm 1, so a non-finite arm-1 run voids every criterion)."""
    n_runs = int(n_runs)
    n_above = int(sum(1 for x in onset_s if x is not None and x > S_STAR))
    frac = n_above / n_runs if n_runs else float("nan")
    ok = bool(n_runs and frac >= O_MIN_FRAC)
    return {"criterion": "O", "n_runs": n_runs, "n_above_s_star": n_above,
            "n_no_onset": int(sum(1 for x in onset_s if x is None)), "frac": frac, "min_frac": O_MIN_FRAC,
            "nonfinite_in_used_arms": bool(any_nonfinite), "verdict": verdict(ok, n_runs > 0 and not any_nonfinite)}


def arm_verdicts(arm, rows_arm, rows_ref, any_nonfinite):
    """Every registered criterion of one arm vs arm 1 (rows aligned by seed).  Output arms: R_s, R_l_rho2, R_l_acc, O;
    global arms: N.  The bootstrap runs once per arm over all 18 cells (fresh default_rng(20261002)).
    any_nonfinite: whether a run of an arm these criteria use (this arm or arm 1) is non-finite (D5)."""
    D = paired(rows_arm, rows_ref)
    lo, hi, med = boot_ci_median(D)
    n = len(rows_arm)
    cells = {c: {"median": float(med[j]) if np.isfinite(med[j]) else None, "lo": float(lo[j]), "hi": float(hi[j]),
                 "n_both": int(np.isfinite(D[:, j]).sum()), "n_up": int((D[:, j] > 0).sum()),
                 "n_down": int((D[:, j] < 0).sum())} for j, c in enumerate(CELLS)}
    out = {"arm": arm, "arm_label": ARM_LABEL[arm], "n_seeds": n, "cells": cells, "criteria": {}}
    if arm in OUTPUT_ARMS:
        for crit in ("R_s", "R_l_rho2", "R_l_acc"):
            out["criteria"][crit] = criterion_lower(D, lo, crit, n, any_nonfinite)
        out["criteria"]["O"] = criterion_O([(r.get("onset") or {}).get("s") for r in rows_arm], n, any_nonfinite)
    elif arm in GLOBAL_ARMS:
        out["criteria"]["N"] = criterion_N(D, hi, n, any_nonfinite)
    else:
        raise ValueError(arm)
    return out


def outcome(v):
    """The page's outcome statements from the verdicts v = {arm: {criterion: verdict}}.  PRIMARY (arm 2 with both
    global arms' N); arm 2b's verdict is SECONDARY and reported on its own."""
    p = v[PRIMARY]
    n = {a: v[a]["N"] for a in GLOBAL_ARMS}
    allv = list(p.values()) + list(n.values())
    res = {"primary_verdicts": {**{f"arm 2 {k}": x for k, x in p.items()}, **{f"arm {ARM_LABEL[a]} N": x
                                                                             for a, x in n.items()}}}
    if "UNRESOLVED" in allv:
        res["primary"] = "UNRESOLVED"
        res["statements"] = ["UNRESOLVED: " + ", ".join(k for k, x in res["primary_verdicts"].items()
                                                      if x == "UNRESOLVED")]
        failed = [k for k, x in res["primary_verdicts"].items() if x == "FAIL"]
        if failed:
            res["statements"].append("Also FAIL: " + ", ".join(failed) + ".")
    elif all(x == "PASS" for x in allv):
        res["primary"] = "PASS"
        res["statements"] = ["All pass: the lever is the rate ratio, not the step count, and slab use still starts "
                             "above s*."]
    else:
        res["primary"] = "FAIL"
        failed = [k for k, x in res["primary_verdicts"].items() if x == "FAIL"]
        st = [f"N fails (arm {ARM_LABEL[a]}): global slowing helps too." for a, x in n.items() if x == "FAIL"]
        if failed and all(k in ("arm 2 R_l_rho2", "arm 2 R_l_acc") for k in failed):
            st.append("Only R_l fails: a scale-only effect.")
        if p["O"] == "FAIL":
            st.append("O fails: the onset moves to or below s*.")
        named = {"arm 2 O"} | {f"arm {ARM_LABEL[a]} N" for a in GLOBAL_ARMS}
        if not all(k in ("arm 2 R_l_rho2", "arm 2 R_l_acc") for k in failed):
            other = [k for k in failed if k not in named]
            if other:
                st.append("Also FAIL (no outcome statement on the page for this combination): " + ", ".join(other)
                          + ".")
        res["statements"] = st
    s2 = v[SECONDARY]
    res["secondary_arm_2b"] = {"verdicts": s2, "all_pass": all(x == "PASS" for x in s2.values()),
                               "any_unresolved": "UNRESOLVED" in s2.values()}
    return res


def score_tables(runs):
    """All registered verdicts from the complete set of runs (a list of run records): every arm × every seed exactly
    once, else an AssertionError (no partial scoring; nothing is scored before every run has ended)."""
    by = {(r["arm"], r["seed"]): r for r in runs}
    assert len(by) == len(runs), "a run appears twice"
    seeds = sorted({r["seed"] for r in runs})
    assert set(by) == {(a, s) for a in ARMS for s in seeds}, "every arm × every seed exactly once"
    rows = {a: [by[(a, s)] for s in seeds] for a in ARMS}
    any_nonfinite = any(not r["finite"] for r in runs)
    # D5 (author, 2026-10-05): a non-finite run makes UNRESOLVED only the criteria that use its arm.  Every criterion of
    # arm a uses arm a and arm 1 (the reference), so a non-finite arm-1 run voids every criterion.
    nonfinite_arm = {a: any(not r["finite"] for r in rows[a]) for a in ARMS}
    per_arm, v = {}, {}
    for a in OUTPUT_ARMS + GLOBAL_ARMS:
        per_arm[a] = arm_verdicts(a, rows[a], rows["std"], nonfinite_arm[a] or nonfinite_arm["std"])
        v[a] = {k: c["verdict"] for k, c in per_arm[a]["criteria"].items()}
    return {"n_seeds": len(seeds), "any_nonfinite": any_nonfinite, "nonfinite_by_arm": nonfinite_arm,
            "nonfinite_runs": [[r["arm"], r["seed"], r["t_nonfinite"]] for r in runs if not r["finite"]],
            "arms": per_arm, "verdicts": v, "outcome": outcome(v)}


# ------------------------------------------------------------------------------------------ descriptive (never a verdict)
def _q(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if not len(a):
        return None
    return {"n": int(len(a)), "median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}


def descriptive(runs):
    """DESCRIPTIVE, registered as such: the endpoint comparison at T_C (ρ₂, both accuracies, s; NO endpoint advantage is
    claimed), the matched step (arm 1's step to 3 s*), steps to each matched point and their ratio to arm 1 (step cost),
    the onset location (s/s*, s/s_F; not tested against the fold), two-sided equivalence for the global arms, overshoot."""
    by = {(r["arm"], r["seed"]): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    rows = {a: [by[(a, s)] for s in seeds] for a in ARMS if all((a, s) in by for s in seeds)}
    out = {"label": "DESCRIPTIVE (never a verdict)", "endpoint_no_advantage_claimed": True, "arms": {}}
    pts = POINTS + ("match_step", "end_T_C")
    for a, R in rows.items():
        d = {"values": {}, "paired_vs_arm1": {}, "steps_to": {}, "steps_ratio_vs_arm1": {}}
        for p in pts:
            d["values"][p] = {q: _q([(r["at"].get(p) or {}).get(q) for r in R]) for q in QTY + ("s", "bce")}
            if a != "std":
                ref_p = "scale_3s*" if p == "match_step" else p
                for q in QTY + ("s",):
                    dd = [r["at"][p][q] - b["at"][ref_p][q] for r, b in zip(R, rows["std"])
                          if r["at"].get(p) and b["at"].get(ref_p)]
                    d["paired_vs_arm1"][f"{p}|{q}"] = {**(_q(dd) or {}), "n_up": int(sum(x > 0 for x in dd)),
                                                       "n_down": int(sum(x < 0 for x in dd))}
        for p in POINTS:
            d["steps_to"][p] = _q([r["t_first"].get(p) for r in R])
            if a != "std":
                d["steps_ratio_vs_arm1"][p] = _q([r["t_first"][p] / b["t_first"][p] for r, b in zip(R, rows["std"])
                                                  if p in r["t_first"] and p in b["t_first"] and b["t_first"][p] > 0])
        on = [r.get("onset") for r in R]
        d["onset"] = {"n_lasting": sum(o is not None for o in on), "n_from_init": sum(bool(o and o["from_init"])
                                                                                       for o in on),
                      "s_over_s_star": _q([o["s_over_s_star"] for o in on if o]),
                      "s_over_s_F": _q([o["s_over_s_F"] for o in on if o]),
                      "n_in_[1,1.25]s*": int(sum(1 for o in on if o and 1 <= o["s_over_s_star"] <= 1.25)),
                      "n_below_or_at_s*": int(sum(1 for o in on if o and o["s"] <= S_STAR))}
        d["max_overshoot_scale"] = max([max(r["overshoot"].values()) for r in R if r["overshoot"]], default=None)
        if a in GLOBAL_ARMS and "std" in rows:
            D = paired(R, rows["std"])
            lo, hi, _ = boot_ci_median(D)
            marg = np.array([DELTA[c.split("|")[1]] for c in CELLS])
            d["two_sided_equivalence"] = {c: {"lo": float(lo[j]), "hi": float(hi[j]), "delta": float(marg[j]),
                                              "within": bool(lo[j] > -marg[j] and hi[j] < marg[j])}
                                          for j, c in enumerate(CELLS)}
        if a != "std" and "std" in rows:
            D = paired(R, rows["std"], cells=tuple(f"end_T_C|{q}" for q in QTY))
            lo, hi, med = boot_ci_median(D)
            d["endpoint_bootstrap_DESCRIPTIVE"] = {q: {"median": float(med[j]), "lo": float(lo[j]), "hi": float(hi[j])}
                                                   for j, q in enumerate(QTY)}
        out["arms"][a] = d
    return out


# ------------------------------------------------------------------------------------------ machine rules and io
def memory_gate(tag):
    """free ≥ 25% and swap free ≥ 500 MB (act_fold.check_memory; waits, re-checking every 60 s); disk free ≥ 20 GB
    (else STOP).  Every check logged to results/phase2b/memory_gate.log."""
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


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS


def rss_guard():
    if _rss() > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {_rss() / 1e9:.2f} GB > 3 GB")


def _setup():
    cur = os.getpriority(os.PRIO_PROCESS, 0)
    if cur < 15:
        os.nice(15 - cur)
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
    old = p.read_text() if p.exists() else ""
    p.write_text(old + json.dumps(_jsonable(row)) + "\n")


def _rows(p):
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


# ------------------------------------------------------------------------------------------ seed scan
SCAN_DIRS = ("src", "tests", "results", "paper", "notes", "independent", "data", "dist")
SCAN_SKIP_FILES = ("phase2b.py", "test_phase2b.py", "phase2b_registration.md", "phase2b_lever_design.md")
SCAN_SKIP_SUFFIX = (".npz", ".npy", ".pdf", ".png", ".pt", ".pkl", ".ots", ".bak", ".gz", ".zip", ".pyc", ".jpg")
SCAN_PATTERNS = {
    "registered": r"(^|[^0-9.])29620[0-7][0-9]([^0-9]|$)|(^|[^0-9])2_962_0[0-7][0-9]([^0-9]|$)"
                  r"|(^|[^0-9.,])2,962,0[0-7][0-9]([^0-9,]|$)"}
SEED_MODULES = ("track_a", "track2a", "track2b", "track2c", "gelu_transfer", "width2_asym", "phase1a_pilot", "phase1c",
                "phase2a", "simplicity_bias_v2", "simplicity_bias_v3")


def scan_tree(patterns=SCAN_PATTERNS, dirs=SCAN_DIRS, chunk=16 * 1024 * 1024, overlap=256):
    """Regex scan of every text file under `dirs` (binary files skipped; 2B's own files and results/phase2b/ skipped),
    streamed in 16 MB chunks with a 256-byte overlap.  {pattern key: [files with a match]}."""
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


def scan():
    """The 80 registered seeds are unused: no number in 2,962,000-2,962,079 (also written with _ or ,) in any text file
    under src/, tests/, results/, paper/, notes/, independent/, data/, dist/ (2B's own files and the design page
    excepted), and no overlap with the SEEDS* / PILOT_SEEDS* constants of the registered test modules."""
    import importlib
    OUT.mkdir(parents=True, exist_ok=True)
    memory_gate("scan")
    hits = scan_tree()
    mine = set(SEEDS)
    ov = {}
    for mod in SEED_MODULES:
        m = importlib.import_module(f"src.{mod}")
        for name in dir(m):
            if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
                ov[f"{mod}.{name}"] = sorted(mine & set(_flat_ints(getattr(m, name))))
    out = {"ranges": {"registered": [SEEDS[0], SEEDS[-1], len(SEEDS)]}, "dirs": list(SCAN_DIRS),
           "patterns": SCAN_PATTERNS, "pattern_files": hits, "registered_seed_overlap": ov,
           "exploration_seeds_disjoint": bool(not (mine & set(range(2_953_000, 2_953_100)))),
           "unused": bool(not any(hits.values()) and not any(ov.values()))}
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable(out), indent=1))
    if not out["unused"]:
        raise SystemExit("STOP: the 2B seed range is not unused")


# ------------------------------------------------------------------------------------------ freeze (no registered seed)
def explore_run(cond, seed):
    return json.loads((EXPLORE / "runs_lever" / f"{cond}_{seed}.json").read_text())


EXPLORE_POINT = {"scale_1.25s*": "scale_1.25s*", "scale_2s*": "scale_2s*", "scale_3s*": "scale_3s*",
                 "bce_0.1": "loss_0.1", "bce_0.03": "loss_0.03", "bce_0.01": "loss_0.01", "end_T_C": "step_40000",
                 "match_step": "match_step"}


def compare_with_exploration(rec, paths, old):
    """Max absolute differences between a run (record and (s, BCE, ρ₂) paths, T = T_C) and a committed exploration run
    of the same arm and seed (p2b_lever.run): steps to every matched point and to the matched step, every metric there
    and at step 40,000, the lasting onset to 40,000, and the committed trajectory samples at steps ≤ T_C.  Exact
    reproduction: every difference 0."""
    d = {"n_compared": 0, "max_abs_metric": 0.0, "steps_equal": True, "onset_equal": True, "max_abs_traj": 0.0,
         "n_traj": 0}
    for mine, theirs in EXPLORE_POINT.items():
        if mine == "end_T_C":
            tm, to = T_C, 40_000
        elif mine == "match_step":
            tm, to = rec["t_match"], old["t_match"]
        elif theirs.startswith("scale_"):
            tm, to = rec["t_first"].get(mine), old["t_scale"].get(theirs[6:])
        else:
            tm, to = rec["t_first"].get(mine), old["t_loss"].get(theirs)
        if tm != to:
            d["steps_equal"] = False
        a, b = rec["at"].get(mine), old["at"].get(theirs)
        if (a is None) != (b is None):
            d["steps_equal"] = False
            continue
        if a is None:
            continue
        for k in ("s", "loss", "rho2", "gplus", "acc_train", "acc_shuffled", "acc_reversed", "bce"):
            d["max_abs_metric"] = max(d["max_abs_metric"], abs(a[k] - b[k])); d["n_compared"] += 1
    oo, on = old["onset"].get("T40000"), rec["onset"]
    if (oo is None) != (on is None) or (oo and (oo["t"] != on["t"] or oo["s"] != on["s"])):
        d["onset_equal"] = False
    S, B, R = paths
    ot = old["traj"]
    for i, t in enumerate(ot["t"]):
        if t <= T_C:
            for k, arr in (("s", S), ("bce", B), ("rho2", R)):
                d["max_abs_traj"] = max(d["max_abs_traj"], abs(float(arr[t]) - ot[k][i]))
            d["n_traj"] += 1
    d["exact"] = bool(d["steps_equal"] and d["onset_equal"] and d["max_abs_metric"] == 0 and d["max_abs_traj"] == 0
                      and d["n_compared"] > 0 and d["n_traj"] > 0)
    return d


def global_factors():
    """From the committed exploration runs: G = median over the 20 seeds of out16 ÷ std steps to 3 s* (rounded to 2
    decimals: 12.18); r_g = the same for glob; G_cm = 12.18^(ln r_o / ln r_g) rounded to 2 decimals (16.14)."""
    t3 = lambda c, s: explore_run(c, s)["t_scale"]["3s*"]                                          # noqa: E731
    r_o = float(np.median([t3("out16", s) / t3("std", s) for s in EXPLORE_SEEDS]))
    r_g = float(np.median([t3("glob", s) / t3("std", s) for s in EXPLORE_SEEDS]))
    return {"r_o": r_o, "G": round(r_o, 2), "r_g": r_g, "a": math.log(r_g) / math.log(12.18),
            "G_cm": round(12.18 ** (math.log(r_o) / math.log(r_g)), 2)}


FREEZE_REPRO_SEED = EXPLORE_SEEDS[0]


def freeze():
    """results/phase2b/frozen.json (no registered seed drawn): the constants (q, s*, λ from v2's frozen.json; s_F from
    phase2a/frozen.json), the arms' learning-rate vectors, the cells, margins and thresholds, the global factors
    recomputed from the committed exploration runs, and the exact reproduction of the committed exploration run of
    every arm on exploration seed 2,953,000 (p2b_lever.run / p2b_lever_cm.py) by run_one."""
    _setup()
    v2f = json.loads(V2_FROZEN.read_text())
    p2a = json.loads(P2A_FROZEN.read_text())
    gf = global_factors()
    repro = {}
    for arm in ARMS:
        memory_gate(f"freeze repro {arm} {FREEZE_REPRO_SEED}")
        old = explore_run(arm, FREEZE_REPRO_SEED)
        rec, paths = run_one(arm, FREEZE_REPRO_SEED, t_match=old["t_match"])
        repro[arm] = {**compare_with_exploration(rec, paths, old), "secs": rec["secs"]}
        print(arm, json.dumps(repro[arm]), flush=True)
        rss_guard()
    fr = {"label": "Phase 2B frozen inputs (before any registered seed is drawn)",
          "q": float(v2f["q"]), "s_star": float(v2f["s_q"]), "lambda": float(v2f["lambda"]),
          "constants_match_v2_frozen": bool(v2f["q"] == Q and v2f["s_q"] == S_STAR and v2f["lambda"] == LAM),
          "v2_frozen_sha256": _sha(V2_FROZEN), "s_F": float(p2a["s_F"]), "s_F_matches_phase2a": p2a["s_F"] == S_FOLD,
          "cap": CAP, "adam": {"lr": ADAM_LR, "betas": [B1, B2], "eps": EPS}, "T_C": T_C,
          "arms": {a: {"label": ARM_LABEL[a], "lr": lr_vec(a).tolist()} for a in ARMS},
          "G_glob": G_GLOB, "G_cm": G_CM, "global_factors_from_exploration": gf,
          "global_factors_reproduce": bool(gf["G"] == G_GLOB and gf["G_cm"] == G_CM),
          "scale_levels": {k: f * S_STAR for k, f in SCALE_LEVELS}, "bce_levels": dict(BCE_LEVELS),
          "cells": list(CELLS), "criterion_cells": {k: list(v) for k, v in CRITERION_CELLS.items()},
          "delta": DELTA, "O_min_frac": O_MIN_FRAC, "reach_min_frac": REACH_MIN_FRAC,
          "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "pct": list(BOOT_PCT)},
          "seeds": [SEEDS[0], SEEDS[-1], len(SEEDS)], "arm_order": list(ARMS),
          "exploration_reproduction_seed": FREEZE_REPRO_SEED, "exploration_reproduction": repro,
          "exploration_reproduction_exact": bool(all(r["exact"] for r in repro.values()))}
    (OUT / "frozen.json").write_text(json.dumps(_jsonable(fr), indent=1))
    print(json.dumps({k: v for k, v in _jsonable(fr).items() if k != "exploration_reproduction"}, indent=1))
    if not (fr["constants_match_v2_frozen"] and fr["s_F_matches_phase2a"] and fr["global_factors_reproduce"]
            and fr["exploration_reproduction_exact"]):
        raise SystemExit("STOP: a frozen check failed")


# ------------------------------------------------------------------------------------------ registration manifest
EXPLORE_FILES = ("README_lever.md", "p2b_explore.py", "p2b_lever.py", "p2b_lever_cm.py", "p2b_lever_cm.json",
                 "p2b_lever_power.py", "p2b_lever_power2.py", "p2b_lever_power2.json", "p2b_lever_power80.py",
                 "p2b_lever_power80.json", "p2b_lever_power_out16_glob.json", "p2b_lever.json", "p2b_check.json")
FROZEN_DATA = (("results/designs/phase2b_lever_design.md", "results/phase2b_registration.md",
                "results/phase2b/seed_scan.json", "results/phase2b/frozen.json", "tests/test_phase2b.py",
                "results/simplicity_bias_v2/frozen.json", "results/phase2a/frozen.json")
               + tuple(f"results/designs/phase2b_explore/{f}" for f in EXPLORE_FILES)
               + tuple(f"results/designs/phase2b_explore/runs_lever/{a}_{s}.json" for a in ARMS for s in EXPLORE_SEEDS))


def code_closure(start=("phase2b",)):
    """Every src module phase2b reaches by relative imports (`from . import X`, `from .X import`), recursively."""
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


def assert_registration():
    m = OUT / "registration.sha256"
    for ln in m.read_text().splitlines():
        h, rel = ln.split()
        assert _sha(ROOT / rel) == h, f"registration hash mismatch: {rel}"
    _assert_committed(m)
    _assert_committed(OUT / "registration_stamp.txt")


# ------------------------------------------------------------------------------------------ run / score
def run():
    """After the registration commit and its timestamp: for each registered seed in order, arm 1 then 2, 2b, 3, 3cm
    (resumable per (arm, seed)); a memory gate before every run."""
    assert_registration()
    _setup()
    rf = OUT / "runs.jsonl"
    done = {(r["arm"], r["seed"]) for r in _rows(rf)}
    t3 = {r["seed"]: r["t_first"].get("scale_3s*") for r in _rows(rf) if r["arm"] == "std"}
    for s in SEEDS:
        for arm in ARMS:
            if (arm, s) in done:
                continue
            memory_gate(f"run {arm} {s}")
            rec, _ = run_one(arm, s, t_match=None if arm == "std" else t3.get(s))
            _append_write(rf, rec)
            if arm == "std":
                t3[s] = rec["t_first"].get("scale_3s*")
            print(json.dumps({k: rec[k] for k in ("arm", "seed", "finite", "t_first", "secs")}), flush=True)
            rss_guard()


def score():
    assert_registration()
    runs = _rows(OUT / "runs.jsonl")
    assert sorted({r["seed"] for r in runs}) == list(SEEDS), "every registered seed"
    ST = score_tables(runs)
    out = {**ST, "DESCRIPTIVE": descriptive(runs), "no_criterion_is_a_forecast": NO_CRITERION_IS_A_FORECAST}
    (OUT / "scores.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("verdicts", "outcome")}), indent=1))


def main(argv):
    cmd = argv[1]
    fns = {"scan": scan, "freeze": freeze, "manifest": manifest, "run": run, "score": score}
    if cmd not in fns:
        raise SystemExit(f"unknown command {cmd}")
    fns[cmd]()


if __name__ == "__main__":
    main(sys.argv)
