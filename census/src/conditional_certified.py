"""Block 1c/1d: certified conditional thresholds from the profiled branch and bound.

Registered in results/conditional_audit_prediction.md (6b17508).

At each output scale s: certified m₋ = min_{G<=0} L*, m₊ = min_{G>0} L* (src/profiled_bnb.certify,
absolute tolerance 1e-7; tightened to 1e-9 if the two intervals overlap).  The globally preferred
placement has G > 0 iff m₊ < m₋.  Certified R_glob = the s interval where that sign changes,
bisected to width 0.005 in |w2|.  The global minimiser's solve margin (exact extrema, profiled b2)
defines the solve switch, bisected the same way (numerically checked, not certified: see note).
Continuity test: the m₋ and m₊ argmins (w1 folded to >= 0 when the data are x-symmetric) and their
separation at the switch.

    python -m src.conditional_certified population      # the six Block B a, quadrature objective
    python -m src.conditional_certified seeds A          # 1d: 50 seeds' training sets at a = A
"""

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .profiled_bnb import _interval_extrema, certify, gap, profile

RESULTS = Path(__file__).resolve().parents[1] / "results"
A_VALUES = (1.30, 1.35, 1.40, 1.45, 1.50, 1.60)
WIDTH = 0.005


def _population():
    from . import blockB_landscape as bb
    x, y = bb.population_data()
    return x.numpy(), y.numpy()


def _seed_data(seed):
    from .fold1d import make_data
    x, y = make_data(200, seed)
    return x.double().numpy(), y.double().numpy()


def solve_margin(w1, b1, s, a, x, y):
    L, b2, *_ = profile([w1], [b1], s, a, x, y)
    t1, t2 = w1 * -0.8 + b1, w1 * 0.8 + b1
    _, imax = _interval_extrema(min(t1, t2), max(t1, t2), a)
    mins = []
    for lo, hi in ((1.2, 2.0), (-2.0, -1.2)):
        u1, u2 = w1 * lo + b1, w1 * hi + b1
        mn, _ = _interval_extrema(min(u1, u2), max(u1, u2), a)
        mins.append(mn)
    zi = s * imax + b2[0]
    zo = s * min(mins) + b2[0]
    return float(min(-zi, zo))


def evaluate(s, a, x, y, symmetric):
    out = {}
    for tol in (1e-7, 1e-9):
        rm = certify(s, a, x, y, "-", tol=tol)
        rp = certify(s, a, x, y, "+", tol=tol)
        if rp["lower"] > rm["upper"]:
            status = "minus"
        elif rm["lower"] > rp["upper"]:
            status = "plus"
        else:
            status = "unresolved"
        if status != "unresolved":
            break
    glob = rp if (status == "plus") else rm
    w1g, b1g = glob["arg_w1"], glob["arg_b1"]
    fold = (lambda w: abs(w)) if symmetric else (lambda w: w)
    sep = math.hypot(fold(rm["arg_w1"]) - fold(rp["arg_w1"]), rm["arg_b1"] - rp["arg_b1"]) \
        if np.isfinite(rp["arg_w1"]) else np.nan
    out.update(a=a, s=s, status=status, tol=tol,
               m_minus_lo=rm["lower"], m_minus_hi=rm["upper"], m_plus_lo=rp["lower"],
               m_plus_hi=rp["upper"], plus_above_log2=rp["above_log2"],
               minus_arg_w1=rm["arg_w1"], minus_arg_b1=rm["arg_b1"],
               plus_arg_w1=rp["arg_w1"], plus_arg_b1=rp["arg_b1"], argmin_separation=sep,
               glob_w1=w1g, glob_b1=b1g,
               glob_G=float(gap([w1g], [b1g], a)[0]) if np.isfinite(w1g) else np.nan,
               glob_is_constant=(status != "plus" and w1g == 0.0),
               solve_margin=solve_margin(w1g, b1g, s, a, x, y) if np.isfinite(w1g) and w1g != 0.0 else -np.inf,
               converged=rm["converged"] and rp["converged"], W=rm["W"])
    return out


def _bisect(fn, lo, hi, a, x, y, symmetric, rows):
    """fn(row) -> bool, false at lo and true at hi; shrink to width WIDTH."""
    while hi - lo > WIDTH:
        mid = round(0.5 * (lo + hi), 6)
        r = evaluate(mid, a, x, y, symmetric)
        rows.append(r)
        if fn(r):
            hi = mid
        else:
            lo = mid
    return lo, hi


def scan(a, x, y, symmetric, grid=None):
    grid = grid if grid is not None else [round(v, 4) for v in np.arange(1.5, 11.0 + 1e-9, 0.5)]
    rows = [evaluate(s, a, x, y, symmetric) for s in grid]
    plus = [r["status"] == "plus" for r in rows]
    solves = [r["solve_margin"] > 0 for r in rows]
    res = {"a": a}
    for name, flags, fn in (("glob", plus, lambda r: r["status"] == "plus"),
                            ("solve", solves, lambda r: r["solve_margin"] > 0)):
        changes = [i for i in range(1, len(flags)) if flags[i] != flags[i - 1]]
        res[f"{name}_sign_changes_on_coarse_grid"] = len(changes)
        first = next((i for i, f in enumerate(flags) if f), None)
        if first is None or first == 0:
            res[f"w2_{name}_lo"] = res[f"w2_{name}_hi"] = np.nan
            continue
        lo, hi = _bisect(fn, grid[first - 1], grid[first], a, x, y, symmetric, rows)
        res[f"w2_{name}_lo"], res[f"w2_{name}_hi"] = lo, hi
    res["unresolved_evaluations"] = sum(r["status"] == "unresolved" for r in rows)
    res["all_converged"] = all(r["converged"] for r in rows)
    return res, rows


def _pop_job(a):
    x, y = _population()
    return scan(a, x, y, symmetric=True)


def population():
    with Pool(int(__import__('os').environ.get('CC_WORKERS', '3'))) as p:
        out = p.map(_pop_job, A_VALUES)
    thr = pd.DataFrame([r for r, _ in out])
    evals = pd.DataFrame([row for _, rows in out for row in rows]).sort_values(["a", "s"])
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv")
    gm = dict(zip(g.a.round(2), g.Ghat_certified))
    sw = pd.read_csv(RESULTS / "blockB_switches.csv")
    thr["w2_glob_frozen"] = thr.a.map(dict(zip(sw.a.round(2), sw.w2_glob)))
    thr["w2_solve_frozen"] = thr.a.map(dict(zip(sw.a.round(2), sw.w2_solve)))
    for k in ("glob", "solve"):
        thr[f"R_{k}_lo"] = thr[f"w2_{k}_lo"] * thr.a.round(2).map(gm) / 2
        thr[f"R_{k}_hi"] = thr[f"w2_{k}_hi"] * thr.a.round(2).map(gm) / 2
        # frozen reports the first grid point where the predicate holds; agreement within one step
        thr[f"{k}_frozen_minus_certified_hi"] = thr[f"w2_{k}_frozen"] - thr[f"w2_{k}_hi"]
    thr.to_csv(RESULTS / "cond_certified_thresholds.csv", index=False)
    evals.to_csv(RESULTS / "cond_certified_evaluations.csv", index=False)
    print(thr.to_string(index=False))


def _seed_job(args):
    a, seed = args
    x, y = _seed_data(seed)
    lo = 1.5
    grid = [round(v, 4) for v in np.arange(lo, 11.0 + 1e-9, 0.5)]
    res, rows = scan(a, x, y, symmetric=False, grid=grid)
    res["seed"] = seed
    return res


def seeds(a):
    a = float(a)
    with Pool(int(__import__('os').environ.get('CC_WORKERS', '4'))) as p:
        out = p.map(_seed_job, [(a, s) for s in range(50)])
    d = pd.DataFrame(out)
    d.to_csv(RESULTS / f"cond_certified_seeds_a{a:.2f}.csv", index=False)
    print(d[["seed", "w2_glob_lo", "w2_glob_hi", "w2_solve_lo", "w2_solve_hi"]].describe().to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "population":
        population()
    elif cmd == "seeds":
        seeds(sys.argv[2])


# ------------------------------------------------------------------ brackets (reallocated compute)
BRACKET_WIDTH = 0.0125      # two bisections of the 0.05 grid step: below one grid step


def _bracket_job(args):
    a, kind, thr = args
    x, y = _population()
    pred = (lambda r: r["status"] == "plus") if kind == "glob" else (lambda r: r["solve_margin"] > 0)
    rows = []
    lo, hi = round(thr - 0.05, 6), round(thr, 6)
    rlo, rhi = evaluate(lo, a, x, y, True), evaluate(hi, a, x, y, True)
    rows += [rlo, rhi]
    steps_out = 0
    while pred(rlo) and steps_out < 20:           # threshold lies below the frozen bracket
        hi, rhi = lo, rlo
        lo = round(lo - 0.05, 6)
        rlo = evaluate(lo, a, x, y, True); rows.append(rlo); steps_out += 1
    while not pred(rhi) and steps_out < 20:       # threshold lies above the frozen bracket
        lo, rlo = hi, rhi
        hi = round(hi + 0.05, 6)
        rhi = evaluate(hi, a, x, y, True); rows.append(rhi); steps_out += 1
    while hi - lo > BRACKET_WIDTH + 1e-12:
        mid = round(0.5 * (lo + hi), 6)
        r = evaluate(mid, a, x, y, True); rows.append(r)
        if pred(r):
            hi, rhi = mid, r
        else:
            lo, rlo = mid, r
    return ({"a": a, "kind": kind, "w2_frozen": thr, "w2_lo": lo, "w2_hi": hi,
             "steps_outside_frozen_bracket": steps_out,
             "argmin_separation_lo": rlo["argmin_separation"], "argmin_separation_hi": rhi["argmin_separation"],
             "glob_G_lo": rlo["glob_G"], "glob_G_hi": rhi["glob_G"],
             "solve_margin_lo": rlo["solve_margin"], "solve_margin_hi": rhi["solve_margin"],
             "all_converged": all(r["converged"] for r in rows),
             "unresolved": sum(r["status"] == "unresolved" for r in rows)}, rows)


def brackets():
    sw = pd.read_csv(RESULTS / "blockB_switches.csv")
    jobs = [(round(float(r.a), 2), k, float(getattr(r, f"w2_{k}"))) for r in sw.itertuples()
            for k in ("glob", "solve")]
    with Pool(int(__import__('os').environ.get('CC_WORKERS', '6'))) as p:
        out = p.map(_bracket_job, jobs)
    t = pd.DataFrame([r for r, _ in out])
    ev = pd.DataFrame([row for _, rows in out for row in rows]).sort_values(["a", "s"])
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv")
    gm = dict(zip(g.a.round(2), g.Ghat_certified))
    t["Ghat_cert"] = t.a.map(gm)
    t["R_lo"], t["R_hi"] = t.w2_lo * t.Ghat_cert / 2, t.w2_hi * t.Ghat_cert / 2
    t["R_frozen"] = t.w2_frozen * t.Ghat_cert / 2
    t["frozen_within_one_step"] = (t.w2_frozen > t.w2_lo - 1e-9) & (t.w2_frozen - t.w2_hi <= 0.05 + 1e-9)
    t.to_csv(RESULTS / "cond_certified_brackets.csv", index=False)
    ev.to_csv(RESULTS / "cond_certified_bracket_evaluations.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__" and sys.argv[1] == "brackets":
    brackets()
