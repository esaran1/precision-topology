"""T2-3c (registered AFTER T2-3's FAIL and T2-3b's UNRESOLVED; asym_registration.md, amendment 3): width-2 training in
width 1's timescale regime.  Same geometry (a = 1.30, Δ = 0.4), matched initialisation, bracketed threshold
s ∈ [0.4371, 0.4532] and criteria as T2-3; the output-weight learning-rate factor φ₂ chosen so that the timescale ratio
(growth / relaxation) sits at width 1's Adam regime.

Rule for φ₂: calibration seeds 520,000–520,039 (never used before); each pilot run trains with φ₂ until ‖v‖₁ first
reaches s_glob (the T2-1 bracket midpoint) and records ONLY the timescale ratio there (growth d log‖v‖₁/dt over the
last min(100, step − 1) steps under φ₂; relaxation = lr·λ_min(D^{-1/2} H D^{-1/2}) at the branch, as asym_t23b._relax).
Placement and crossings are never evaluated.  Bisection in log φ₂ on [1e−4, 1] until the pilot median ratio is within
20% of 0.0057 (the geometric midpoint of [0.0014, 0.023]).
Budget: at the chosen φ₂, pilot runs continue (still without placement) until ‖v‖₁ reaches 3·s_glob; budget =
ceil(1.5 × the 95th percentile of those step counts / 1000) × 1000, at least 32,000.
Validity: achieved median ratio at crossing in [0.0014, 0.023], and ≥ 40 crossings; otherwise UNRESOLVED.

    python -m src.asym_t23c pilot | freeze | train | score
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

from .asym_register import FROZEN, MIN_CROSS, STEP0_STOP, _act, _setup, score_t2_3, training_set
from .asym_t23b import _relax, validity

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "asym_t23c"
CAL_SEEDS = tuple(range(520_000, 520_040))
SEEDS = tuple(range(600_320, 600_400))
SEEDS_EXT = tuple(range(600_400, 600_480))
TARGET, TOL = 0.0057, 0.20
PHI_RANGE = (1e-4, 1.0)
PILOT_CAP = 100_000
FROZEN_C = RESULTS / "asym_t23c_frozen.json"


def _k():
    return json.loads(FROZEN.read_text())["k"]


def _s_glob():
    f = json.loads(FROZEN.read_text())
    return math.sqrt(f["s_lo"] * f["s_hi"])


def _start(seed, phi):
    import torch
    from .width2_train import LR, init_params
    torch.set_num_threads(1)
    _setup(); act = _act()
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); k = _k(); q0[2] *= k; q0[5] *= k
    q = q0.clone().requires_grad_(True)
    return q, torch.optim.Adam([q], lr=LR), X, Y, act


def _step(q, opt, X, Y, act, phi):
    import torch
    from .width2_train import logits
    opt.zero_grad()
    torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
    vb = q.detach()[[2, 5]].clone()
    opt.step()
    with torch.no_grad():
        q[[2, 5]] = vb + phi * (q[[2, 5]] - vb)
    return float(q.detach()[2].abs() + q.detach()[5].abs())


def pilot_ratio(seed, phi, target, cap=PILOT_CAP, continue_to=None):
    """Timescale ratio at the first step with ‖v‖₁ ≥ target (no placement evaluated); optionally keep training (still
    without placement) until ‖v‖₁ ≥ continue_to and return that step count too."""
    q, opt, X, Y, act = _start(seed, phi)
    hist, ratio, reach3 = {}, float("nan"), None
    for step in range(1, cap + 1):
        s = _step(q, opt, X, Y, act, phi)
        hist[step] = s; hist.pop(step - 101, None)
        if not math.isfinite(ratio) and s >= target:
            win = min(100, step - 1)
            growth = math.log(hist[step] / hist[step - win]) / win if win >= 1 else float("nan")
            relax = _relax(q.detach().clone(), opt.state[q], X, Y, act)
            ratio = growth / relax if relax > 0 else float("nan")
            if continue_to is None:
                return {"seed": seed, "step": step, "ratio": ratio}
        if continue_to is not None and s >= continue_to:
            return {"seed": seed, "ratio": ratio, "steps_to_3x": step}
    return {"seed": seed, "step": cap, "ratio": ratio, "steps_to_3x": cap}


def _pilot_job(args):
    os.nice(15)
    return pilot_ratio(*args)


def choose_phi(med_of, lo=PHI_RANGE[0], hi=PHI_RANGE[1], target=TARGET, tol=TOL, max_iter=40):
    """Bisection in log φ₂ (ratio increases with φ₂) until the median is within tol of target.  The endpoints are not
    evaluated: the bracket is known from T2-3 (φ₂ = 1: ratio 2.73 at crossing) and T2-3b (φ₂ = 0.0093: 6.8e−6); a trial
    at which no pilot run reaches the threshold scale within the cap counts as below target (median 0)."""
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        m = med_of(mid)
        if abs(m / target - 1) <= tol:
            return mid, m
        lo, hi = (mid, hi) if m < target else (lo, mid)
    return None, "bisection did not reach the tolerance"


def budget_from(steps_to_3x, floor=32_000):
    return int(max(floor, math.ceil(1.5 * float(np.percentile(steps_to_3x, 95)) / 1000) * 1000))


def pilot(workers=1):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    tgt = _s_glob()
    log = []
    with get_context("spawn").Pool(workers) as pool:
        def med_of(phi):
            r = pool.map(_pilot_job, [(s, phi, tgt) for s in CAL_SEEDS])
            v = np.array([x["ratio"] for x in r], float)
            m = float(np.median(v[np.isfinite(v)])) if np.isfinite(v).sum() >= len(v) / 2 else 0.0   # most never reached
            log.append({"phi": phi, "median_ratio": m, "finite": int(np.isfinite(v).sum()),
                        "median_step": float(np.median([x["step"] for x in r]))})
            print(json.dumps(log[-1]), flush=True)
            return m
        phi, m = choose_phi(med_of)
        steps3 = None
        if phi is not None:
            r3 = pool.map(_pilot_job, [(s, phi, tgt, PILOT_CAP, 3 * tgt) for s in CAL_SEEDS])
            steps3 = [x["steps_to_3x"] for x in r3]
    res = {"phi2": phi, "median_ratio": m, "bisection": log, "target_s": tgt,
           "steps_to_3x": steps3, "budget": budget_from(steps3) if steps3 else None}
    (OUT / "pilot.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("phi2", "median_ratio", "budget")}, default=float))


def freeze():
    p = json.loads((OUT / "pilot.json").read_text())
    if p["phi2"] is None:
        raise SystemExit("no phi2: T2-3c is not registered")
    d = {"phi2": p["phi2"], "k": _k(), "budget": p["budget"], "seeds": [SEEDS[0], SEEDS[-1]],
         "seeds_ext": [SEEDS_EXT[0], SEEDS_EXT[-1]], "pilot_median_ratio": p["median_ratio"],
         "ratio_range": [0.0014, 0.023]}
    txt = json.dumps(d, indent=1, sort_keys=True)
    FROZEN_C.write_text(txt)
    (RESULTS / "asym_t23c_frozen.sha256").write_text(hashlib.sha256(txt.encode()).hexdigest() + "\n")
    print(txt)


def train_one(seed, k, phi, budget):
    import torch
    from .width2_train import placed, unpack
    q, opt, X, Y, act = _start(seed, phi)
    with torch.no_grad():
        pass
    if placed(q.detach().numpy(), act)[0]:
        return {"seed": seed, "placed_at_init": True, "crossed": False, "step": 0, "s_cross": np.nan, "ratio": np.nan}
    hist = {}
    for step in range(1, budget + 1):
        s = _step(q, opt, X, Y, act, phi)
        hist[step] = s; hist.pop(step - 101, None)
        qn = q.detach().numpy().copy()
        if placed(qn, act)[0]:
            _, v, _ = unpack(qn)
            win = min(100, step - 1)
            growth = math.log(hist[step] / hist[step - win]) / win if win >= 1 else float("nan")
            relax = _relax(q.detach().clone(), opt.state[q], X, Y, act)
            return {"seed": seed, "placed_at_init": False, "crossed": True, "step": step,
                    "s_cross": float(np.abs(v).sum()), "growth": growth, "relax": relax,
                    "ratio": growth / relax if relax > 0 else np.nan}
    return {"seed": seed, "placed_at_init": False, "crossed": False, "step": np.nan, "s_cross": np.nan, "ratio": np.nan}


def _train_job(seed):
    os.nice(15)
    fz = json.loads(FROZEN_C.read_text())
    r = train_one(seed, fz["k"], fz["phi2"], fz["budget"])
    print(json.dumps({"seed": seed, "done": True}), flush=True)
    return r


def train(workers=1):
    from multiprocessing import get_context
    f = OUT / "train.csv"
    done = set() if not f.exists() else set(pd.read_csv(f).seed)
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_train_job, [s for s in SEEDS if s not in done]):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
    d = pd.read_csv(f)
    if d.placed_at_init.mean() <= STEP0_STOP and d.crossed.sum() < MIN_CROSS:
        with get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(_train_job, [s for s in SEEDS_EXT if s not in set(d.seed)]):
                pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)


def verdict(crossed_ratios, s_cross, s_lo, s_hi, n_cross):
    ok, m = validity(crossed_ratios)
    sc = score_t2_3(s_cross, s_lo, s_hi)
    if n_cross < MIN_CROSS:
        return "UNRESOLVED (fewer than 40 crossings)", m, sc
    if not ok:
        return f"UNRESOLVED (median ratio {m:.4g} outside [0.0014, 0.023])", m, sc
    return sc["verdict"], m, sc


def score():
    fz = json.loads(FROZEN.read_text())
    d = pd.read_csv(OUT / "train.csv")
    out = {"runs": len(d), "placed_at_init": int(d.placed_at_init.sum()), "crossed": int(d.crossed.sum())}
    if d.placed_at_init.mean() > STEP0_STOP:
        out["T2-3c"] = "STOP (more than 20% placed at step 0)"
    else:
        c = d[d.crossed & ~d.placed_at_init]
        v, m, sc = verdict(c.ratio.values, c.s_cross.values, fz["s_lo"], fz["s_hi"], len(c))
        out.update({"median_ratio_at_cross": m, "criteria": sc, "T2-3c": v})
    (OUT / "scores.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"pilot": lambda: pilot(w), "freeze": freeze, "train": lambda: train(w), "score": score}[sys.argv[1]]()
