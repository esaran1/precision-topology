"""Follow-up cross-family test, inside the q2 reduction's domain.

Registered in results/crossfamily_followup_prediction.md AFTER the original
test's X1 failed.

    python -m src.crossfamily_followup verify   # domain check at the chosen a, before any threshold
    python -m src.crossfamily_followup run      # R_glob, R_solve for family A and q2
    python -m src.crossfamily_followup score
"""

from __future__ import annotations

import csv
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
VERIFY = RESULTS / "crossfamily_followup_domain.csv"
OUT = RESULTS / "crossfamily_followup_thresholds.csv"
SCORES = RESULTS / "crossfamily_followup_scores.csv"
A_VALUES = (1.01, 1.02, 1.03, 1.04)
U_MAX, V_MAX, X_MAX = 2.2, 2.0, 2.0                  # T70 compact set, family-A coordinates
S_A = U_MAX * X_MAX + V_MAX                          # 6.4
S_Q2 = S_A / math.sqrt(2)                            # q2 coordinates: sigma_q2 = sigma_A / sqrt2
# R-grid equal to family A's frozen resolution at a = 1.30 (0.5 then 0.05 in |w2|)
GHAT_A_130 = 0.0861018
DR_COARSE, DR_FINE = 0.5 * GHAT_A_130 / 2, 0.05 * GHAT_A_130 / 2
R_LO, R_HI = 0.05, 0.60
R_INF = {"glob": 0.19991194947, "solve": 0.30711111078}
WORKERS = int(os.environ.get("X_WORKERS", "3"))
FIELDS = ["family", "a", "threshold", "ghat", "w2", "R", "t_min", "t_max", "frac_outside"]


def _q2():
    from .depth_families import make_family
    return make_family(2.0)[0]


def ghat_q2(a):
    """Certified Ghat for q2, with its argmax placement."""
    from . import family_certify as fc
    s = (a - 1) ** 0.5
    L = max(1.0, a - 1.0) * 2.0 * s
    best, arg = -1e9, None
    def scan(us, vs):
        b, g = -1e9, None
        for u in us:
            if abs(u) < 1e-9:
                continue
            for v in vs:
                val = fc.gap(2.0, a, s * u, s * v)
                if val > b:
                    b, g = val, (u, v)
        return b, g
    h = 0.02
    lo, arg = scan(np.arange(-6, 6 + 1e-9, h), np.arange(-6, 6 + 1e-9, h))
    while L * h / 2 >= 1e-3 * lo and h > 1e-7:
        h /= 5
        u0, v0 = arg
        pad = 12 * h
        lo, arg = scan(np.arange(u0 - pad, u0 + pad + 1e-12, h), np.arange(v0 - pad, v0 + pad + 1e-12, h))
    return lo, lo + L * h / 2, (s * arg[0], s * arg[1])


def ghat_A(a):
    from .kappa_certify import certify_exact
    c = certify_exact(a)
    return c["Ghat_lo"], c["Ghat_hi"], (c["w1"], c["b1"])


def _t_range(w1, b1):
    x = np.r_[np.linspace(-0.8, 0.8, 4001), np.linspace(1.2, 2.0, 2001), -np.linspace(1.2, 2.0, 2001)]
    t = w1 * x + b1
    return float(t.min()), float(t.max()), float(np.mean(np.abs(t) > 1))


def verify():
    rows = []
    print(f"analytic bound: every placement in the T70 compact set has max|sigma_q2| <= {S_Q2:.4f}")
    for a in A_VALUES:
        eps = a - 1
        bound = math.sqrt(eps) * S_Q2
        lo, hi, (w1, b1) = ghat_q2(a)
        tmin, tmax, frac = _t_range(w1, b1)
        rows.append({"a": a, "eps": eps, "sqrt_eps_times_Smax": bound, "condition_holds": bound <= 1,
                     "ghat_q2_lo": lo, "ghat_q2_hi": hi, "argmax_w1": w1, "argmax_b1": b1,
                     "t_min_at_argmax": tmin, "t_max_at_argmax": tmax,
                     "frac_outside_at_argmax": frac})
        print(f"  a={a}: sqrt(eps)*Smax = {bound:.4f} ({'<= 1' if bound <= 1 else '> 1 VIOLATED'});"
              f"  Ghat_q2 in [{lo:.6e}, {hi:.6e}];  t at argmax in [{tmin:+.4f}, {tmax:+.4f}];"
              f"  fraction |t|>1 = {frac:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(VERIFY, index=False)


def job(args):
    """Thresholds with the scale-equivariant Block B minimisation (src/blockB_scaled.py).
    The frozen raw-coordinate version fails its validity gate at these eps
    (results/blockB_scaled_validity.md)."""
    family, a, which = args
    import torch
    from . import blockB_landscape as bb
    from .blockB_scaled import best_conditional
    torch.set_num_threads(1)
    g = ghat_q2(a)[0] if family == "q2" else ghat_A(a)[0]
    x, y = bb.population_data()
    lo, hi = 2 * R_LO / g, 2 * R_HI / g
    coarse, fine = 2 * DR_COARSE / g, 2 * DR_FINE / g
    restarts = 50 if which == "glob" else 24
    def pred(w2):
        b = best_conditional(family, a, w2, x, y, restarts=restarts)
        return b is not None and (b["gap"] > 0 if which == "glob" else b["solves"])
    w2 = bb.bracket_then_refine(pred, lo, hi, coarse=coarse, fine=fine)
    tmin = tmax = frac = None
    if w2 is not None:
        b = best_conditional(family, a, w2, x, y, restarts=restarts)
        if family == "q2" and b is not None:
            tmin, tmax, frac = _t_range(b["w1"], b["b1"])
    return {"family": family, "a": a, "threshold": which, "ghat": g, "w2": w2,
            "R": None if w2 is None else w2 * g / 2, "t_min": tmin, "t_max": tmax,
            "frac_outside": frac}


def validity():
    """Gate: at a = 1.30 the scaled procedure must reproduce the frozen thresholds."""
    ref = {("A", "glob"): 0.215254, ("A", "solve"): 0.307814,
           ("q2", "glob"): 0.204983, ("q2", "solve"): 0.295390}
    jobs = [(f, 1.30, w) for f in ("A", "q2") for w in ("glob", "solve")]
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    rows = []
    for r in res:
        k = (r["family"], r["threshold"])
        rows.append({"family": k[0], "threshold": k[1], "R_frozen": ref[k], "R_scaled": r["R"],
                     "diff": r["R"] - ref[k], "one_grid_step": DR_FINE,
                     "pass": abs(r["R"] - ref[k]) <= DR_FINE + 1e-9})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "blockB_scaled_validity.csv", index=False)


def run():
    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT)
        done = set(zip(d.family, d.a.round(2), d.threshold))
    jobs = [(f, a, w) for f in ("A", "q2") for a in A_VALUES for w in ("glob", "solve")
            if (f, round(a, 2), w) not in done]
    new = not OUT.exists()
    with OUT.open("a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        if new:
            wr.writeheader()
        with Pool(WORKERS) as pool:
            for r in pool.imap_unordered(job, jobs):
                wr.writerow(r)
                fh.flush()
                print(f"  {r['family']:2s} a={r['a']:.2f} {r['threshold']:5s}: R = {r['R']}  "
                      f"frac|t|>1 = {r['frac_outside']}", flush=True)


def score():
    d = pd.read_csv(OUT)
    rows = []
    for a in A_VALUES:
        for w in ("glob", "solve"):
            A = d[(d.family == "A") & (d.a.round(2) == a) & (d.threshold == w)].iloc[0]
            Q = d[(d.family == "q2") & (d.a.round(2) == a) & (d.threshold == w)].iloc[0]
            tol = max(abs(A.R - R_INF[w]), 2 * DR_FINE)
            rows.append({"a": a, "threshold": w, "R_A": A.R, "R_q2": Q.R, "diff": Q.R - A.R,
                         "tol": tol, "within": bool(abs(Q.R - A.R) <= tol),
                         "q2_frac_outside": Q.frac_outside})
    s = pd.DataFrame(rows)
    s.to_csv(SCORES, index=False)
    print(s.to_string(index=False))
    for w in ("glob", "solve"):
        ok = bool(s[s.threshold == w].within.all())
        print(f"Y1-{w}: q2 within tol of A at all four a: {'PASS' if ok else 'FAIL'}")
    return s


if __name__ == "__main__":
    {"verify": verify, "run": run, "score": score, "validity": validity}[sys.argv[1]]()
