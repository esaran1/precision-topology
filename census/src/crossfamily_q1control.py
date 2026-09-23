"""Negative control for Y1: q1 at the follow-up's a, same scale-equivariant procedure.

Registered in results/crossfamily_q1control_prediction.md after Y1 passed and
before any q1 threshold at these a was computed.

    python -m src.crossfamily_q1control verify     # q1 domain condition, before registering
    python -m src.crossfamily_q1control validity   # gate: q1 at a = 1.30 vs the frozen procedure
    python -m src.crossfamily_q1control run        # q1 R_glob, R_solve at a = 1.01-1.04
    python -m src.crossfamily_q1control score      # vs family A (crossfamily_followup_thresholds.csv)

Family A's thresholds are not recomputed: they are the committed Y1 values.
"""

from __future__ import annotations

import csv
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .crossfamily_followup import (A_VALUES, DR_COARSE, DR_FINE, OUT as FOLLOWUP_OUT, R_HI, R_LO,
                                   RESULTS, _t_range)

Q = 1.0
FAM = "q1"
VERIFY = RESULTS / "crossfamily_q1control_domain.csv"
GATE = RESULTS / "crossfamily_q1control_validity.csv"
OUT = RESULTS / "crossfamily_q1control_thresholds.csv"
SCORES = RESULTS / "crossfamily_q1control_scores.csv"
SCAN_UV = 6.0                     # certification scan box |u|, |v| <= 6 in q1 limit coordinates
X_MAX = 2.0
S_BOX = SCAN_UV * X_MAX + SCAN_UV  # every scanned placement has |sigma| <= 18
FIELDS = ["family", "a", "threshold", "ghat", "w2", "R", "t_min", "t_max", "frac_outside"]


def ghat_q(q, a):
    """Certified Ghat for q-family q, limit coordinates w1 = s u, b1 = s v, s = eps**(1/q)."""
    from . import family_certify as fc
    s = (a - 1) ** (1 / q)
    L = max(1.0, a - 1.0) * 2.0 * s

    def scan(us, vs):
        b, g = -1e9, None
        for u in us:
            if abs(u) < 1e-9:
                continue
            for v in vs:
                val = fc.gap(q, a, s * u, s * v)
                if val > b:
                    b, g = val, (u, v)
        return b, g
    h = 0.02
    lo, arg = scan(np.arange(-SCAN_UV, SCAN_UV + 1e-9, h), np.arange(-SCAN_UV, SCAN_UV + 1e-9, h))
    while L * h / 2 >= 1e-3 * lo and h > 1e-7:
        h /= 5
        u0, v0 = arg
        pad = 12 * h
        lo, arg = scan(np.arange(u0 - pad, u0 + pad + 1e-12, h), np.arange(v0 - pad, v0 + pad + 1e-12, h))
    return lo, lo + L * h / 2, (s * arg[0], s * arg[1])


def verify():
    rows = []
    for a in A_VALUES:
        eps = a - 1
        bound = eps ** (1 / Q) * S_BOX
        lo, hi, (w1, b1) = ghat_q(Q, a)
        tmin, tmax, frac = _t_range(w1, b1)
        rows.append({"a": a, "eps": eps, "eps_1q_times_Sbox": bound, "condition_holds": bound <= 1,
                     "ghat_q1_lo": lo, "ghat_q1_hi": hi, "argmax_w1": w1, "argmax_b1": b1,
                     "t_min_at_argmax": tmin, "t_max_at_argmax": tmax,
                     "frac_outside_at_argmax": frac})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(VERIFY, index=False)


def job(args):
    family, a, which = args
    import torch
    from . import blockB_landscape as bb
    from .blockB_scaled import best_conditional
    torch.set_num_threads(1)
    g = ghat_q(Q, a)[0]
    x, y = bb.population_data()
    restarts = 50 if which == "glob" else 24

    def pred(w2):
        b = best_conditional(family, a, w2, x, y, restarts=restarts)
        return b is not None and (b["gap"] > 0 if which == "glob" else b["solves"])
    w2 = bb.bracket_then_refine(pred, 2 * R_LO / g, 2 * R_HI / g,
                                coarse=2 * DR_COARSE / g, fine=2 * DR_FINE / g)
    tmin = tmax = frac = None
    if w2 is not None:
        b = best_conditional(family, a, w2, x, y, restarts=restarts)
        if b is not None:
            tmin, tmax, frac = _t_range(b["w1"], b["b1"])
    return {"family": family, "a": a, "threshold": which, "ghat": g, "w2": w2,
            "R": None if w2 is None else w2 * g / 2, "t_min": tmin, "t_max": tmax,
            "frac_outside": frac}


def validity():
    """Gate: at a = 1.30 the scaled procedure must reproduce q1's frozen thresholds
    (crossfamily_thresholds.csv) within one frozen grid step, as A and q2 did."""
    ref = pd.read_csv(RESULTS / "crossfamily_thresholds.csv")
    ref = ref[(ref.family == FAM) & (ref.a.round(2) == 1.30)].set_index("threshold").R
    with Pool(2) as pool:
        res = pool.map(job, [(FAM, 1.30, w) for w in ("glob", "solve")])
    rows = [{"family": FAM, "threshold": r["threshold"], "R_frozen": float(ref[r["threshold"]]),
             "R_scaled": r["R"], "diff": r["R"] - float(ref[r["threshold"]]),
             "one_grid_step": DR_FINE,
             "pass": abs(r["R"] - float(ref[r["threshold"]])) <= DR_FINE + 1e-9} for r in res]
    for r in rows:
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(GATE, index=False)


def run(workers=8):
    jobs = [(FAM, a, w) for a in A_VALUES for w in ("glob", "solve")]
    with OUT.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        wr.writeheader()
        with Pool(workers) as pool:
            for r in pool.imap_unordered(job, jobs):
                wr.writerow(r)
                fh.flush()
                print(f"  q1 a={r['a']:.2f} {r['threshold']:5s}: R = {r['R']}  "
                      f"frac|t|>1 = {r['frac_outside']}", flush=True)


def score():
    A = pd.read_csv(FOLLOWUP_OUT)
    A = A[A.family == "A"]
    d = pd.read_csv(OUT)
    rows = []
    for a in A_VALUES:
        for w in ("glob", "solve"):
            ra = float(A[(A.a.round(2) == a) & (A.threshold == w)].R.iloc[0])
            rq = d[(d.a.round(2) == a) & (d.threshold == w)].iloc[0]
            diff = float(rq.R) - ra
            rows.append({"a": a, "threshold": w, "R_A": ra, "R_q1": float(rq.R), "diff": diff,
                         "steps": diff / DR_FINE, "beyond_two_steps": abs(diff) > 2 * DR_FINE + 1e-9,
                         "within_one_step": abs(diff) <= DR_FINE + 1e-9,
                         "q1_frac_outside": rq.frac_outside})
    s = pd.DataFrame(rows)
    s.to_csv(SCORES, index=False)
    print(s.to_string(index=False))
    for w in ("glob", "solve"):
        sub = s[s.threshold == w]
        print(f"Z1-{w}: q1 beyond two grid steps of A at all four a: "
              f"{'PASS' if sub.beyond_two_steps.all() else 'FAIL'} "
              f"({int(sub.beyond_two_steps.sum())}/4; within one step: {int(sub.within_one_step.sum())}/4)")
    return s


if __name__ == "__main__":
    {"verify": verify, "validity": validity, "run": run, "score": score}[sys.argv[1]]()
