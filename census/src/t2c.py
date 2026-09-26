"""Track 2C, step 1: calibration of the early-scale basin rule on the EXISTING slowed width-2 runs (T2-3b, T2-3c, T2-3d),
per the author's binding amendment:

    "don't fix the rule at 0.25 × s_glob: T2-3d had early crossers at s = 0.016.  Calibrate the rule scale on the existing
     slowed runs as the earliest scale at which the basin classification is final in at least 95% of runs and below the
     earliest crossing observed, state it, then apply it to fresh seeds."  If no scale satisfies both, stop and report.

Definitions (fixed before the calibration was computed):
  runs         every run of T2-3b, T2-3c and T2-3d not placed at step 0 (crossing or not); per-arm figures beside the pooled one.
  rule scale   s_r on width2_lag.GRID (10^(−2.5 + k/8), 0.0032 .. 5.6).  The rule is applied to a run at the FIRST PASSAGE of
               ‖v‖₁ through s_r (the first step t ≥ 1 with s_r between s(t−1) and s(t), or s(0) = s_r), and only if that
               passage is strictly before the run's crossing step.  A run whose ‖v‖₁ never passes s_r before it crosses (it
               starts above s_r and never comes down to it, or it crosses first) has NO classification at s_r.
  basin        the branch point z*(v(t_r)) (width2_lag.branch_point: damped Newton at fixed v from the state at t_r, a minimum).
  final        the basin followed by Newton continuation along the run's own recorded v-path (width2_lag's predictor–corrector)
               from t_r to the crossing step ends at the crossing branch point z*(v_c) (Newton from the crossing state), to
               max|Δz| ≤ 1e−6(1 + max|z|).  Non-crossing runs: never counted final (no crossing branch to compare).
  earliest crossing   the smallest s_cross over the three arms' crossing runs (0.0164, T2-3d).
  choice       the smallest grid scale with (fraction of runs final) ≥ 0.95 AND s_r < earliest crossing; none → STOP.
Because a run can be final only if it has a classification, the fraction AVAILABLE (classification exists) bounds the final
fraction from above at every scale: if it is below 0.95 at every candidate scale, no definition of "final" can pass.

    python -m src.t2c availability          # uses width2_lag's saved first-passage states; replays non-crossers
    python -m src.t2c finality [s_r ...]    # the basin continuation at the given grid scales (all available runs)
    python -m src.t2c calibrate             # the decision (results/t2c/calibration.json)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .width2_lag import GRID, IDX, STATES, STEP_MAX, STEP_MIN, arms

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "t2c"
SLOWED = ("T2-3b", "T2-3c", "T2-3d")
NEED = 0.95
MATCH = 1e-6
FIN_COLS = ["arm", "seed", "s_r", "available", "final", "status", "t_rule", "cross_step", "end_dist"]


def choose_rule_scale(scales, final_frac, earliest_crossing, need=NEED):
    """The earliest scale with final fraction ≥ need and below the earliest crossing; None if there is none.  Also
    returns the earliest scale meeting the finality condition alone (for the report)."""
    scales = np.asarray(scales, float); final_frac = np.asarray(final_frac, float)
    o = np.argsort(scales); scales, final_frac = scales[o], final_frac[o]
    ok_final = final_frac >= need
    both = ok_final & (scales < earliest_crossing)
    pick = float(scales[both][0]) if both.any() else None
    first_final = float(scales[ok_final][0]) if ok_final.any() else None
    return {"rule_scale": pick, "earliest_final_scale_any": first_final, "decision": "REGISTER" if pick else "STOP"}


def available(passage_steps, cross_step, s_r):
    """Classification at s_r exists iff the first passage through s_r happens strictly before the crossing step
    (cross_step = inf for a run that never crosses)."""
    t = passage_steps.get(s_r)
    return t is not None and t < cross_step


def _replay_noncrosser(seed, phi, k, budget=32_000):
    from .width2_lag import replay
    r = replay(seed, phi, k, budget + 1)                     # 'step' beyond the budget: every passage within it counts
    return {g: p["t"] for g, p in r["passage"].items() if p["t"] <= budget}


def availability():
    try:
        os.nice(15)
    except OSError:
        pass
    OUT.mkdir(exist_ok=True)
    rows = []
    for arm in SLOWED:
        f, phi, k = arms()[arm]
        d = pd.read_csv(f, float_precision="round_trip")
        d = d[~d.placed_at_init.astype(bool)]
        for r in d.itertuples():
            if r.crossed:
                with open(STATES / f"{arm}_{int(r.seed)}.pkl", "rb") as fh:
                    st = pickle.load(fh)
                steps = {g: p["t"] for g, p in st["passage"].items()}
                cs, sc, s0 = float(r.step), float(r.s_cross), st["s_init"]
            else:
                steps = _replay_noncrosser(int(r.seed), phi, k)
                cs, sc, s0 = math.inf, math.inf, float("nan")
            for g in GRID:
                rows.append({"arm": arm, "seed": int(r.seed), "crossed": bool(r.crossed), "s_r": g, "s_cross": sc,
                             "cross_step": cs, "passage_step": steps.get(g, np.nan), "available": available(steps, cs, g)})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "availability_runs.csv", index=False)
    earliest = float(A[A.crossed].s_cross.min())
    S = A.groupby("s_r").agg(n_runs=("seed", "size"), n_available=("available", "sum")).reset_index()
    S["frac_available"] = S.n_available / S.n_runs
    S["below_earliest_crossing"] = S.s_r < earliest
    for arm in SLOWED:
        g = A[A.arm == arm].groupby("s_r").available.mean()
        S[f"frac_available_{arm}"] = S.s_r.map(g)
    S.to_csv(OUT / "availability.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False)); print("earliest crossing", earliest)
    return S, earliest


def follow(z0, V, t0, t1, x, y, act, h_max=STEP_MAX):
    """Continuation of the fixed-v branch along the recorded path from t0 to t1 (t0 < t1 ≤ len(V) − 1).  Returns
    (z at t1, status)."""
    from .width2_lag import _is_min, branch_point, corrector_ok, jac_v, v_at
    V = np.asarray(V, float); tc = len(V) - 1
    vd = V[tc] - V[tc - 1] if tc >= 1 else np.zeros(2)
    z, gm, H, conv = branch_point(z0, V[int(t0)], x, y, act)
    if not (conv and _is_min(H)):
        return None, "no_basin_at_rule"
    t, h = float(t0), h_max
    while t < t1:
        v_prev = v_at(V, t, tc, vd); s_prev = float(np.abs(v_prev).sum())
        j = min(int(math.floor(t)), tc - 1)
        sp = max(float(np.abs(V[j + 1] - V[j]).sum()), 1e-300)
        tn = min(t + max(h * s_prev / sp, 1e-6), float(t1))
        v = v_at(V, tn, tc, vd)
        z_pred = z - np.linalg.solve(H, jac_v(z, v_prev, x, y, act) @ (v - v_prev))
        zn, gm, Hn, conv = branch_point(z_pred, v, x, y, act)
        if not (conv and _is_min(Hn) and corrector_ok(zn, z_pred, z)):
            if h / 2 < STEP_MIN:
                return None, ("branch_lost" if (not conv or not _is_min(Hn)) else "jump")
            h /= 2
            continue
        t, z, H, h = tn, zn, Hn, min(h_max, 2 * h)
    return z, "ok"


def finality_one(arm, seed, s_r):
    from .asym_register import _act, _setup
    from .width2_lag import branch_point
    _setup(); act = _act()
    from .asym_register import training_set
    x, y = training_set(seed)
    with open(STATES / f"{arm}_{seed}.pkl", "rb") as fh:
        st = pickle.load(fh)
    p = st["passage"].get(s_r)
    V = st["V"]; tc = len(V) - 1
    if p is None or p["t"] >= tc:
        return {"arm": arm, "seed": seed, "s_r": s_r, "available": False, "final": False, "status": "unavailable"}
    q = p["q"]
    zc, _, _, _ = branch_point(st["q_cross"][IDX], V[tc], x, y, act)
    z_end, status = follow(q[IDX], V, p["t"], tc, x, y, act)
    final = bool(z_end is not None and np.abs(z_end - zc).max() <= MATCH * (1 + np.abs(zc).max()))
    return {"arm": arm, "seed": seed, "s_r": s_r, "available": True, "final": final, "status": status,
            "t_rule": int(p["t"]), "cross_step": tc,
            "end_dist": float(np.abs(z_end - zc).max()) if z_end is not None else float("nan")}


def finality(scales):
    try:
        os.nice(15)
    except OSError:
        pass
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    f = OUT / "finality_runs.csv"
    done = set() if not f.exists() else {(r.arm, int(r.seed), round(r.s_r, 12)) for r in pd.read_csv(f).itertuples()}
    A = pd.read_csv(OUT / "availability_runs.csv")
    for s_r in scales:
        g = min(GRID, key=lambda v: abs(math.log(v / s_r)))
        sub = A[(np.isclose(A.s_r, g)) & A.available & A.crossed]
        for r in sub.itertuples():
            if (r.arm, int(r.seed), round(g, 12)) in done:
                continue
            row = finality_one(r.arm, int(r.seed), g)
            pd.DataFrame([row]).reindex(columns=FIN_COLS).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(json.dumps({"s_r": g, "done": True}), flush=True)


def calibrate():
    S = pd.read_csv(OUT / "availability.csv")
    A = pd.read_csv(OUT / "availability_runs.csv")
    earliest = float(A[A.crossed].s_cross.min())
    n_runs = int(A.groupby("s_r").size().iloc[0])
    fin = OUT / "finality_runs.csv"
    F = pd.read_csv(fin) if fin.exists() else pd.DataFrame(columns=["s_r", "final"])
    rows = []
    for r in S.itertuples():
        g = F[np.isclose(F.s_r.astype(float), r.s_r)] if len(F) else F
        rows.append({"s_r": r.s_r, "frac_available": r.frac_available,
                     "final_computed": bool(len(g)), "frac_final": float(g.final.sum() / n_runs) if len(g) else float("nan"),
                     "frac_final_upper_bound": r.frac_available, "below_earliest_crossing": r.s_r < earliest})
    C = pd.DataFrame(rows)
    # the decision uses the computed final fraction where computed, else its upper bound (availability): a scale whose
    # upper bound is below 0.95 cannot pass whatever the classification finds
    ff = np.where(C.final_computed, C.frac_final, C.frac_final_upper_bound)
    dec = choose_rule_scale(C.s_r, ff, earliest)
    undecided = C[(~C.final_computed) & (C.frac_final_upper_bound >= NEED)]
    out = {"label": "Track 2C calibration (existing slowed runs T2-3b, T2-3c, T2-3d; author's binding amendment)",
           "n_runs": n_runs, "earliest_crossing": earliest, "need": NEED, **dec,
           "max_frac_available": float(C.frac_available.max()),
           "s_r_at_max_available": float(C.s_r[C.frac_available.idxmax()]),
           "max_frac_available_below_earliest": float(C[C.below_earliest_crossing].frac_available.max()),
           "scales_needing_finality_check": [float(v) for v in undecided.s_r]}
    if len(undecided):
        out["decision"] = "INCOMPLETE (finality not computed where availability allows 0.95)"
    C.to_csv(OUT / "calibration_scales.csv", index=False)
    (OUT / "calibration.json").write_text(json.dumps(out, indent=1))
    print(C.round(4).to_string(index=False)); print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    {"availability": availability, "calibrate": calibrate,
     "finality": lambda: finality([float(v) for v in sys.argv[2:]])}[sys.argv[1]]()
