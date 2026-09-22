"""Cross-family test of the scaling reduction (results/crossfamily_prediction.md).

Runs the FROZEN Block B procedure (src/blockB_landscape.py, sha256 9f1b1074...)
on the q-families by substituting only the activation: `blockB_landscape.activation`
is rebound to return the q-family function, so `minimise`, `best_conditional`,
`gap_of`, the degeneracy exclusion and `solves` run unchanged.  Grid steps are
scaled by Ghat_A(a)/Ghat_q(a) so the R-resolution equals family A's frozen grid.

    python -m src.crossfamily_landscape run      # q2, q1 at six a; resumable
    python -m src.crossfamily_landscape score    # X1, X2 against family A
"""

from __future__ import annotations

import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "crossfamily_thresholds.csv"
SCORES = RESULTS / "crossfamily_scores.csv"
A_VALUES = (1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
FAMILIES = (2.0, 1.0)
R_LO, R_HI = 0.05, 0.60
R_INF = {"glob": 0.19991194947, "solve": 0.30711111078}
WORKERS = int(os.environ.get("X_WORKERS", "2"))
FIELDS = ["family", "q", "a", "threshold", "ghat", "w2", "R", "coarse", "fine"]


def ghat_A(a):
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv").set_index("a")
    return float(g.loc[round(a, 2), "Ghat_certified"])


def job(args):
    q, a, which = args
    import torch
    from . import blockB_landscape as bb
    from .depth_families import make_family
    from .family_certify import cert

    torch.set_num_threads(1)
    fq, _ = make_family(q)
    f = lambda v, _a=a: fq(v, _a)          # noqa: E731
    bb.activation = lambda name, par, _f=f: _f   # substitute ONLY the activation
    gq, _ = cert(q, a)
    k = ghat_A(a) / gq
    x, y = bb.population_data()
    lo, hi = 2 * R_LO / gq, 2 * R_HI / gq
    if which == "glob":
        def pred(w2):
            b = bb.best_conditional(a, w2, x, y)
            return b is not None and b["gap"] > 0
    else:
        def pred(w2):
            b = bb.best_conditional(a, w2, x, y, restarts=24)
            return b is not None and b["solves"]
    w2 = bb.bracket_then_refine(pred, lo, hi, coarse=0.5 * k, fine=0.05 * k)
    return {"family": f"q{int(q)}", "q": q, "a": a, "threshold": which, "ghat": gq,
            "w2": w2, "R": None if w2 is None else w2 * gq / 2,
            "coarse": 0.5 * k, "fine": 0.05 * k}


def run():
    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT)
        done = set(zip(d.q, d.a.round(2), d.threshold))
    jobs = [(q, a, w) for q in FAMILIES for a in A_VALUES for w in ("glob", "solve")
            if (q, round(a, 2), w) not in done]
    new = not OUT.exists()
    with OUT.open("a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if new:
            wr.writeheader()
        with Pool(WORKERS) as pool:
            for r in pool.imap_unordered(job, jobs):
                wr.writerow(r)
                fh.flush()
                print(f"  {r['family']} a={r['a']:.2f} {r['threshold']:5s}: R = {r['R']}", flush=True)


def score():
    sw = pd.read_csv(RESULTS / "blockB_switches.csv").set_index("a")
    d = pd.read_csv(OUT)
    rows = []
    for a in A_VALUES:
        gA = ghat_A(a)
        dA = 0.05 * gA / 2
        RA = {"glob": float(sw.loc[a, "w2_glob"]) * gA / 2,
              "solve": float(sw.loc[a, "w2_solve"]) * gA / 2}
        for which in ("glob", "solve"):
            tol = max(abs(RA[which] - R_INF[which]), 2 * dA)
            for fam in ("q2", "q1"):
                r = d[(d.family == fam) & (d.a.round(2) == round(a, 2)) & (d.threshold == which)]
                Rq = float(r.R.iloc[0]) if len(r) and pd.notna(r.R.iloc[0]) else np.nan
                rows.append({"a": a, "threshold": which, "family": fam, "R_A": RA[which],
                             "R_q": Rq, "diff": Rq - RA[which], "tol": tol,
                             "within": bool(abs(Rq - RA[which]) <= tol)})
    s = pd.DataFrame(rows)
    s.to_csv(SCORES, index=False)
    q2 = s[s.family == "q2"]
    q1 = s[s.family == "q1"]
    x1 = bool(q2.within.all())
    x2 = any(int((~q1[q1.threshold == w].within).sum()) >= 4 for w in ("glob", "solve"))
    print(s.to_string(index=False))
    print(f"\nX1 (q2 within tol at all six a, both thresholds): {'PASS' if x1 else 'FAIL'}")
    print(f"X2 (q1 outside tol at >= 4 of 6 a, some threshold): {'PASS' if x2 else 'FAIL'}")
    return s, x1, x2


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
