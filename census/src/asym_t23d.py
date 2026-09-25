"""T2-3d (registered after the item 2 diagnosis; asym_registration.md, amendment 4): does the width-1 own-sample threshold
predict where width-2 training crosses in the SLOWED regime?

a = 1.30, Δ = 0.4, matched initialisation (k as T2-3), output learning-rate factor φ₂ = 0.01778 (T2-3c's), budget 32,000,
placement from step 0 with the 20% stop.  Fresh seeds 600,480–600,559 (extension 600,560–600,639 if fewer than 40 cross).
Before any training: each seed's width-1 own-sample threshold (width2_diagnosis.own_w1: width-1 conditional search on the
run's own training set, 100 restarts, scan s = 10^(k/8), bisection to 5%) is computed, frozen and hashed.

Criteria (on crossing runs with a defined own threshold; UNRESOLVED with fewer than 40):
  (i)  median |log(s_cross / s_w1own)| <= 0.10;
  (ii) mean(|log(s_cross/s_w1own)| − |log(s_cross/s_glob,w2)|), paired run-level bootstrap 95% interval entirely below 0
       (s_glob,w2 = 0.44508, the T2-1 bracket midpoint).
PASS iff both.  This covers the slowed regime only; the full-speed regime (T2-3) remains unexplained.

    python -m src.asym_t23d own [workers] | freeze | train [workers] | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "asym_t23d"
SEEDS = tuple(range(600_480, 600_560))
SEEDS_EXT = tuple(range(600_560, 600_640))
PHI2 = 0.01778279410038923
BUDGET = 32_000
S_GLOB_W2 = 0.44507940623559955
MIN_N = 40
FROZEN_D = RESULTS / "asym_t23d_own_frozen.csv"


def _own_job(seed):
    os.nice(15)
    from .width2_diagnosis import own_w1
    r = own_w1(seed)
    print(json.dumps({"seed": seed, "done": True}), flush=True)
    return r


def own(workers=2, seeds=SEEDS):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    f = OUT / "own_parts.csv"
    done = set() if not f.exists() else set(pd.read_csv(f).seed)
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_own_job, [s for s in seeds if s not in done]):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)


def freeze():
    d = pd.read_csv(OUT / "own_parts.csv").sort_values("seed")
    d.to_csv(FROZEN_D, index=False)
    h = hashlib.sha256(FROZEN_D.read_bytes()).hexdigest()
    (RESULTS / "asym_t23d_own_frozen.sha256").write_text(h + "\n")
    print(len(d), int(d.s_w1_own.isna().sum()), h)


def _train_job(seed):
    os.nice(15)
    from .asym_t23c import train_one
    k = json.loads((RESULTS / "asym_frozen.json").read_text())["k"]
    r = train_one(seed, k, PHI2, BUDGET)
    print(json.dumps({"seed": seed, "done": True}), flush=True)
    return r


def train(workers=3):
    from multiprocessing import get_context
    if not FROZEN_D.exists():
        raise SystemExit("own thresholds not frozen")
    f = OUT / "train.csv"
    done = set() if not f.exists() else set(pd.read_csv(f).seed)
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_train_job, [s for s in SEEDS if s not in done]):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)


def score_arrays(s_cross, s_own, s_glob=S_GLOB_W2, n_boot=10_000, seed=0):
    s_cross, s_own = np.asarray(s_cross, float), np.asarray(s_own, float)
    ok = np.isfinite(s_cross) & np.isfinite(s_own)
    s_cross, s_own = s_cross[ok], s_own[ok]
    if len(s_cross) < MIN_N:
        return {"verdict": "UNRESOLVED", "n": len(s_cross)}
    e_own = np.abs(np.log(s_cross / s_own)); e_pop = np.abs(np.log(s_cross / s_glob))
    med = float(np.median(e_own))
    d = e_own - e_pop
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    ok1, ok2 = med <= 0.10, hi < 0
    return {"verdict": "PASS" if ok1 and ok2 else "FAIL", "n": len(s_cross), "median_abs_log_err": med,
            "criterion_i": ok1, "diff_mean": float(d.mean()), "diff_lo": lo, "diff_hi": hi, "criterion_ii": ok2,
            "median_cross_over_own": float(np.median(s_cross / s_own)), "median_cross_over_pop": float(np.median(s_cross / s_glob))}


def score():
    own_ = pd.read_csv(FROZEN_D)
    d = pd.read_csv(OUT / "train.csv").merge(own_[["seed", "s_w1_own"]], on="seed", how="left")
    out = {"runs": len(d), "placed_at_init": int(d.placed_at_init.sum()), "crossed": int(d.crossed.sum()),
           "own_undefined": int(own_.s_w1_own.isna().sum())}
    if d.placed_at_init.mean() > 0.20:
        out["T2-3d"] = "STOP (more than 20% placed at step 0)"
    else:
        c = d[d.crossed & ~d.placed_at_init]
        sc = score_arrays(c.s_cross, c.s_w1_own)
        out.update({"criteria": sc, "T2-3d": sc["verdict"]})
    (OUT / "scores.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    {"own": lambda: own(w), "freeze": freeze, "train": lambda: train(w), "score": score}[sys.argv[1]]()
