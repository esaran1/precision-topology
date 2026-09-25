"""Asymmetric windows, Track 2 Step 2: the REGISTERED test (results/asym_registration.md).  a = 1.30, Δ = 0.4.

T2-1 (landscape): a validated conditional-minimiser bracket search at width 2 finds exactly one placement switch
      (unplaced below, placed above).
T2-2 (secondary, directional): the switch scale lies below the pilot's (Δ = 0.8) lower bracket end, 0.5623.
T2-3 (training, W1's criterion unchanged): width-2 Adam training on asymmetric training sets crosses above the
      threshold: ≥ 90% of crossing runs have s_cross ≥ s_hi; the bootstrap 95% interval of median(s_cross/s_hi) − 1
      lies above 0; and median(s_cross/s_lo) ≤ 1.25.  s = ‖w₂‖₁; ratios need no Γ̂₂.

    python -m src.asym_register scan | bisect | validate | train | score
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .asym_pilot import population, set_windows

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "asym_parts"
A = 1.30
DELTA = 0.4
PILOT_LO = 0.5623413251903491
GRID = tuple(10 ** (-1 + k / 8) for k in range(17))          # 0.1 .. 10
GRID_DOWN = tuple(10 ** (-2 + k / 8) for k in range(8))       # extension below 0.1 (registered)
GRID_UP = tuple(10 ** (1 + k / 8) for k in range(1, 9))       # extension above 10 (registered)
RESTARTS, CAP, POLISH, TIE = 1000, 3000, 5, 1e-9
BISECT_REL = 0.05
LADDER = (500, 1000, 2000, 4000)
SEEDS = tuple(range(80))
SEEDS_EXT = tuple(range(80, 160))
MIN_CROSS = 40
BUDGET = 32_000


def _act():
    from .width2_geometry import Act
    return Act("fa", A)


def _setup():
    set_windows(DELTA)
    from . import width2_train as wt
    from . import width2_geometry as g
    wt._XI = np.linspace(*g.INNER, 401)
    wt._XO = np.concatenate([np.linspace(*g.OUTER[0], 201), np.linspace(*g.OUTER[1], 201)])
    return population(DELTA)


# ------------------------------------------------------------------------------------------ one scale
def eval_scale(s, restarts=RESTARTS, seed=0, cap=CAP, keep=False):
    """Retained conditional minimiser at scale s: batched BFGS restarts, the POLISH lowest distinct candidates polished
    by damped Newton, the constant predictor included.  Status from the exact continuous-window G₊ enclosure."""
    from .asym_pilot import _polish
    from .width2_conditional import constant_predictor_loss, search_batch
    x, y = _setup()
    act = _act()
    _, cands = search_batch(s, x, y, act, restarts=restarts, seed=seed, maxit=cap)
    cands = [c for c in cands if c["p"] is not None and np.all(np.isfinite(c["p"]))]
    return _retain(cands, s, x, y, act, constant_predictor_loss(y), keep)


def _retain(cands, s, x, y, act, const, keep=False):
    from .asym_pilot import _polish
    cands = sorted(cands, key=lambda c: c["loss"])
    picked, seen = [], []
    for c in cands:
        if all(abs(c["loss"] - l0) > 1e-7 for l0 in seen):
            picked.append(c); seen.append(c["loss"])
        if len(picked) == POLISH:
            break
    pol = [_polish(c, s, x, y, act, "width2") for c in picked]
    best = min(pol, key=lambda r: r["loss"])
    status = status_of(best["G_lo"], best["G_hi"]) if best["loss"] <= const else "unplaced"
    bp = min((r["loss"] for r in pol if r["placed"]), default=np.inf)
    bu = min((r["loss"] for r in pol if r["unplaced"]), default=np.inf)
    row = {"s": s, "retained_loss": min(best["loss"], const), "G_lo": best["G_lo"], "G_hi": best["G_hi"],
           "status": status, "best_placed_loss": bp, "best_unplaced_loss": bu,
           "tie": bool(np.isfinite(bp) and np.isfinite(bu) and abs(bp - bu) < TIE),
           "raw_min_loss": cands[0]["loss"], "q": json.dumps([float(v) for v in best["q"]])}
    if keep:
        row["_cands"] = cands
    return row


def status_of(g_lo, g_hi):
    if g_lo > 0:
        return "placed"
    if g_hi <= 0:
        return "unplaced"
    return "undecided"


# ------------------------------------------------------------------------------------------ decisions
def switch_of(d):
    """T2-1's pattern on a scan (sorted by s): 'switch' iff no undecided/tied scale, exactly one status change, from
    unplaced to placed.  Returns (verdict, bracket or None, reason)."""
    d = d.sort_values("s")
    if (d.status == "undecided").any() or d.tie.any():
        return "undecided", None, "undecided or tied scale"
    pl = (d.status == "placed").values
    ch = int((pl[1:] != pl[:-1]).sum())
    if ch == 0:
        return ("all_placed" if pl[0] else "all_unplaced"), None, "no change"
    if ch == 1 and not pl[0]:
        k = int(np.argmax(pl))
        return "switch", (float(d.s.iloc[k - 1]), float(d.s.iloc[k])), "one change"
    return "alternation", None, f"{ch} changes"


def ladder_ok(losses, statuses, tol=TIE):
    return (max(losses) - min(losses) <= tol) and len(set(statuses)) == 1


def validated(end_rows):
    """Both bracket ends: ladder unchanged (loss within 1e−9, same status), the independent search not below the
    retained loss by more than 1e−9, and the audit (no candidate of the other status within 1e−9 below)."""
    return all(r["ladder_ok"] and r["independent_ok"] and r["audit_ok"] for r in end_rows)


def score_t2_3(s_cross, s_lo, s_hi, seed=0):
    from .width2_train import _boot_median_ci
    s_cross = np.asarray(s_cross, float)
    if len(s_cross) < MIN_CROSS:
        return {"verdict": "UNRESOLVED", "n": len(s_cross)}
    r_hi = s_cross / s_hi
    frac = float((r_hi >= 1).mean())
    lo, hi = _boot_median_ci(r_hi - 1)
    med_lo = float(np.median(s_cross / s_lo))
    ok = frac >= 0.90 and lo > 0 and med_lo <= 1.25
    return {"verdict": "PASS" if ok else "FAIL", "n": len(s_cross), "frac_above_s_hi": frac,
            "median_ratio_s_hi": float(np.median(r_hi)), "ci_median_minus_1": (lo, hi), "median_ratio_s_lo": med_lo}


# ------------------------------------------------------------------------------------------ stages
def _append(name, row):
    PARTS.mkdir(exist_ok=True)
    f = PARTS / name
    pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")}]).to_csv(f, mode="a", header=not f.exists(),
                                                                                   index=False)


def _read(name):
    f = PARTS / name
    return pd.read_csv(f, float_precision="round_trip") if f.exists() else pd.DataFrame()


def _scale_job(args):
    os.nice(15)
    s, seed = args
    r = eval_scale(s, seed=seed)
    print(json.dumps({"s": s, "status": r["status"], "loss": r["retained_loss"], "G_lo": r["G_lo"]}), flush=True)
    return r


def scan(workers=1):
    from multiprocessing import get_context
    grid = list(GRID)
    done = _read("scan.csv")
    have = set() if done.empty else set(np.round(done.s, 12))
    todo = [(s, 100 + k) for k, s in enumerate(grid) if round(s, 12) not in have]
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_scale_job, todo):
            _append("scan.csv", r)
    d = _read("scan.csv")
    v, br, why = switch_of(d)
    # registered extensions: all placed -> extend down; all unplaced -> extend up (once each)
    ext = GRID_DOWN if v == "all_placed" else GRID_UP if v == "all_unplaced" else ()
    if ext:
        todo = [(s, 200 + k) for k, s in enumerate(ext) if round(s, 12) not in set(np.round(d.s, 12))]
        with get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(_scale_job, todo):
                _append("scan.csv", r)
        v, br, why = switch_of(_read("scan.csv"))
    print(json.dumps({"scan": v, "bracket": br, "why": why}))
    return v, br


def bisect():
    v, br = switch_of(_read("scan.csv"))[:2]
    if v != "switch":
        print(json.dumps({"bisect": "not applicable", "scan": v})); return None
    lo, hi = br
    done = _read("bisect.csv")
    for r in (done.itertuples() if not done.empty else []):
        if r.status == "placed":
            hi = min(hi, r.s)
        elif r.status == "unplaced":
            lo = max(lo, r.s)
    k = len(done)
    while hi / lo - 1 > BISECT_REL:
        mid = math.sqrt(lo * hi)
        r = eval_scale(mid, seed=300 + k); k += 1
        _append("bisect.csv", r)
        print(json.dumps({"s": mid, "status": r["status"]}), flush=True)
        if r["status"] == "placed":
            hi = mid
        elif r["status"] == "unplaced":
            lo = mid
        else:
            print(json.dumps({"bisect": "undecided at", "s": mid})); break
    print(json.dumps({"bracket": (lo, hi)}))
    return lo, hi


def _validate_end(s):
    from .width2_conditional import constant_predictor_loss, independent_search, search_batch
    x, y = _setup()
    act = _act()
    _, cands = search_batch(s, x, y, act, restarts=max(LADDER), seed=500, maxit=CAP)
    cands = [c for c in cands if c["p"] is not None and np.all(np.isfinite(c["p"]))]
    const = constant_predictor_loss(y)
    lad = [_retain([c for c in cands if c["k"] < n], s, x, y, act, const) for n in LADDER]
    top = lad[-1]
    ind = independent_search(s, x, y, act, starts=40, seed=7)
    other = "unplaced" if top["status"] == "placed" else "placed"
    other_loss = top["best_unplaced_loss"] if other == "unplaced" else top["best_placed_loss"]
    return {"s": s, "status": top["status"], "retained_loss": top["retained_loss"],
            "ladder_losses": json.dumps([r["retained_loss"] for r in lad]),
            "ladder_ok": ladder_ok([r["retained_loss"] for r in lad], [r["status"] for r in lad]),
            "independent_loss": ind["loss"], "independent_ok": bool(ind["loss"] >= top["retained_loss"] - TIE),
            "audit_ok": bool(not (other_loss < top["retained_loss"] + TIE)), "G_lo": top["G_lo"], "G_hi": top["G_hi"]}


def validate():
    d = pd.concat([_read("scan.csv"), _read("bisect.csv")])
    lo = float(d[d.status == "unplaced"].s.max()); hi = float(d[(d.status == "placed") & (d.s > lo)].s.min())
    rows = [_validate_end(lo), _validate_end(hi)]
    for r in rows:
        _append("validate.csv", r)
    print(json.dumps({"validated": validated(rows), "s_lo": lo, "s_hi": hi}))


# ------------------------------------------------------------------------------------------ training
def training_set(seed, delta=DELTA, n_per_class=200):
    """Asymmetric analogue of fold1d.make_data: inner U(I); outer uniform on O (side by length); float32 rounding."""
    rng = np.random.default_rng(seed)
    inner = rng.uniform(-0.8, 0.8, n_per_class)
    pR = (0.8 + delta) / (1.6 + delta)
    right = rng.uniform(0, 1, n_per_class) < pR
    outer = np.where(right, rng.uniform(1.2, 2.0 + delta, n_per_class), -rng.uniform(1.2, 2.0, n_per_class))
    x = np.concatenate([inner, outer]).astype(np.float32).astype(float)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return x, y


def _train_job(seed):
    os.nice(15)
    from .width2_train import train
    _setup()
    x, y = training_set(seed)
    r = train(seed, _act(), 2.0, budget=BUDGET, log_every=10**9, x=x, y=y)   # gamma2_hat = 2 → R2 field = ‖v‖₁
    c = r["cross"]
    row = {"seed": seed, "crossed": c is not None, "step": c["step"] if c else np.nan,
           "s_cross": c["R2"] if c else np.nan, "q": json.dumps([float(v) for v in c["q"]]) if c else ""}
    print(json.dumps({k: row[k] for k in ("seed", "crossed", "step", "s_cross")}), flush=True)
    return row


def train_all(workers=1):
    from multiprocessing import get_context
    done = _read("train.csv")
    have = set() if done.empty else set(done.seed)
    todo = [s for s in SEEDS if s not in have]
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_train_job, todo):
            _append("train.csv", r)
    d = _read("train.csv")
    if d.crossed.sum() < MIN_CROSS:                               # registered extension
        todo = [s for s in SEEDS_EXT if s not in set(d.seed)]
        with get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(_train_job, todo):
                _append("train.csv", r)


def score():
    sc = _read("scan.csv"); v, br, why = switch_of(sc)
    val = _read("validate.csv")
    out = {"T2-1_scan": v, "why": why}
    if v == "switch" and len(val) == 2:
        ok = validated(val.to_dict("records"))
        s_lo, s_hi = float(val.s.min()), float(val.s.max())
        out.update({"T2-1": "PASS" if ok else "UNRESOLVED (bracket not validated)", "s_lo": s_lo, "s_hi": s_hi,
                    "T2-2": "PASS" if s_hi < PILOT_LO else "FAIL"})
        tr = _read("train.csv")
        if ok and not tr.empty:
            out["T2-3"] = score_t2_3(tr[tr.crossed].s_cross.values, s_lo, s_hi)
            out["runs"] = len(tr); out["crossed"] = int(tr.crossed.sum())
    elif v in ("all_placed", "all_unplaced", "alternation"):
        out.update({"T2-1": "FAIL", "T2-2": "not applicable", "T2-3": "not applicable (no threshold)"})
    else:
        out.update({"T2-1": "UNRESOLVED", "T2-3": "not applicable"})
    (RESULTS / "asym_scores.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    {"scan": lambda: scan(w), "bisect": bisect, "validate": validate, "train": lambda: train_all(w),
     "score": score}[cmd]()
