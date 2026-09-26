"""Track T redesign (v2): weight decay on the hidden weights + the ρ₂ path.  Design: results/simplicity_bias_v2_design.md.

Objective at fixed s (the same data, network, s convention and profiled b as v1, src/simplicity_bias.py):
    L_λ(θ; s) = min_b mean ℓ(s·Σₖ ṽₖ tanh(wₖ·x + cₖ) + b, y) + (λ/2)(‖W‖² + ‖c‖²).
θ = 0 gives L_λ = log 2, so every minimiser has ‖(W, c)‖ ≤ B(λ) = √(2 log 2 / λ) (attained: coercive, continuous).

    python -m src.simplicity_bias_v2 lambda          # the λ rule (smallest grid value passing the attainment check)
    python -m src.simplicity_bias_v2 grid SET        # pilot path on the grid, set A or B
    python -m src.simplicity_bias_v2 bisect SET      # q from both sets' grid paths, then s_q by bisection, CMA-ES
    python -m src.simplicity_bias_v2 summarise       # gate verdict
    python -m src.simplicity_bias_v2 freeze          # (gate pass only) s_q, κ_q landscape, hashed
    python -m src.simplicity_bias_v2 train W NW      # seeds W::NW, trajectories only (ρ₂ not evaluated)
    python -m src.simplicity_bias_v2 predict         # χ, κ_q per run from pre-crossing information; hashed
    python -m src.simplicity_bias_v2 score           # only after the predictions are committed
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import simplicity_bias as sb
from .width2_conditional import bfgs_batch, profile_b_batch

OUT = sb.RESULTS / "simplicity_bias_v2"
RUNS = OUT / "runs"
LAMBDA_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
CHECK_SCALES = (1.0, 4.0, 16.0)
CHECK_RESTARTS = 200
GRID = tuple(float(v) for v in np.round(0.5 * 1.25 ** np.arange(21), 6))      # 0.5 … 43.4
GTOL = 1e-8
LADDER = (200, 400, 800)
TOL = 1e-7
BISECT_REL = 0.005
AGREE_TOL = 0.02
SEEDS = tuple(range(2_710_000, 2_710_040))
LR = 0.01
BUDGET = 40_000
STOP_FACTOR = 3.0                    # a run stops once s ≥ 3·s_q (a rule on s only) or at the budget
T0_LEVEL = 0.5                       # χ is measured at the last upward passage of 0.5·s_q before the run stops
SDOT_WINDOW = 100


def bound(lam):
    return math.sqrt(2 * math.log(2) / lam)


def data():
    return sb.make_data(sb.PILOT_P)


def loss_grad(P, s, X, y, lam):
    L, G = sb.loss_grad_batch(P, s, X, y)
    h = P[:, :12]
    return L + 0.5 * lam * (h ** 2).sum(axis=1), G + np.concatenate([lam * h, np.zeros((len(P), 4))], axis=1)


def search(s, X, y, lam, restarts, family, seed, maxit=3000, batch=200, P0=None):
    rng = np.random.default_rng(seed)
    P0 = sb.starts(rng, restarts, family) if P0 is None else P0
    cands = []
    for i in range(0, len(P0), batch):
        P, L, G, it = bfgs_batch(lambda Q, rows: loss_grad(Q, s, X, y, lam), P0[i:i + batch], gtol=GTOL, maxit=maxit)
        f90 = bfgs_batch.last_f90
        for k in range(len(P)):
            eta = P[k, 12:]; P[k, 12:] = eta / np.linalg.norm(eta)
            gn = float(np.abs(G[k]).max())
            cands.append({"k": i + k, "p": P[k].copy(), "loss": float(L[k]), "gnorm": gn, "iters": int(it[k]),
                          "flags": sb.flags(gn, float(L[k]), float(f90[k]))})
    return cands


def retain(cands):
    ok = [c for c in cands if not (set(c["flags"]) & sb.EXCLUDING)]
    return min(ok, key=lambda c: c["loss"]) if ok else None


def attained(ret, cands, lam):
    """Attainment check at one scale: the retained minimiser is converged (gradient ≤ 100·gtol, no flag at all, so
    not a stalled saturating tail), inside the a-priori bound B(λ), and reproduced by at least 2 restarts within 1e−7."""
    if ret is None:
        return {"ok": False, "reason": "no converged candidate"}
    norm = float(np.linalg.norm(ret["p"][:12]))
    hits = int(sum(abs(c["loss"] - ret["loss"]) <= TOL for c in cands if not (set(c["flags"]) & sb.EXCLUDING)))
    ok = (ret["flags"] == []) and norm <= bound(lam) and hits >= 2
    return {"ok": bool(ok), "norm": norm, "bound": bound(lam), "hits": hits, "gnorm": ret["gnorm"],
            "flags": ret["flags"]}


def hessian_min_eig(p, s, X, y, lam, h=1e-5):
    """Smallest eigenvalue of the profiled objective's Hessian in the 16 search coordinates, with the exactly flat
    η-radial direction projected out (central differences of the analytic gradient)."""
    n = 16
    H = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        H[:, j] = (loss_grad((p + e)[None], s, X, y, lam)[1][0] - loss_grad((p - e)[None], s, X, y, lam)[1][0]) / (2 * h)
    H = 0.5 * (H + H.T)
    r = np.zeros(n); r[12:] = p[12:] / np.linalg.norm(p[12:])
    Q = np.eye(n) - np.outer(r, r)
    ev = np.linalg.eigvalsh(Q @ H @ Q)
    ev = np.sort(ev)
    return float(ev[1]) if abs(ev[0]) < 1e-6 * max(1.0, abs(ev[-1])) else float(ev[0])   # skip the projected zero


# ------------------------------------------------------------------------------------------ λ rule
def run_lambda():
    sb._init_worker()
    X, y = data()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    chosen = None
    for lam in LAMBDA_GRID:
        ok_all = True
        for s in CHECK_SCALES:
            cands = search(s, X, y, lam, CHECK_RESTARTS, "A", 5_000 + int(s * 10))
            ret = retain(cands)
            a = attained(ret, cands, lam)
            rows.append({"lambda": lam, "s": s, **{k: v for k, v in a.items() if k != "flags"},
                         "flags": ",".join(a.get("flags", [])), "loss": None if ret is None else ret["loss"]})
            print(json.dumps(rows[-1]), flush=True)
            ok_all &= a["ok"]
        if ok_all:
            chosen = lam
            break
    res = {"grid": LAMBDA_GRID, "check_scales": CHECK_SCALES, "restarts": CHECK_RESTARTS, "chosen_lambda": chosen,
           "rows": rows}
    (OUT / "lambda_rule.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"chosen_lambda": chosen}), flush=True)


def chosen_lambda():
    return json.loads((OUT / "lambda_rule.json").read_text())["chosen_lambda"]


# ------------------------------------------------------------------------------------------ the path
def evaluate(s, X, y, lam, family, validate=False):
    cands = search(s, X, y, lam, max(LADDER), family, sb.SET_SEED[family] * 100_003 + int(round(s * 1e6)) % 99_991)
    lad = []
    for n in LADDER:
        r = retain(cands[:n])
        lad.append({"n": n, "loss": None if r is None else r["loss"],
                    "rho2": None if r is None else sb.feature_usage(r["p"], X)["rho2"]})
    ret = retain(cands)
    fu = sb.feature_usage(ret["p"], X)
    a = attained(ret, cands, lam)
    au = sb.audit(cands, ret, TOL)
    rec = {"s": s, "set": family, "lambda": lam, "loss": ret["loss"], "rho2": fu["rho2"], "V1": fu["V1"], "V2": fu["V2"],
           "S1": fu["S1"], "S2": fu["S2"], "gplus": float(sb.gplus(ret["p"][None], X, y)[0]),
           "p": [float(t) for t in ret["p"]], "ladder": lad,
           "ladder_ok": bool(lad[1]["loss"] is not None and abs(lad[1]["loss"] - lad[2]["loss"]) <= TOL),
           "attained": a, "audit": au}
    if validate:
        rec["hess_min_eig"] = hessian_min_eig(ret["p"], s, X, y, lam)
        ind = independent(s, X, y, lam, seed=sb.SET_SEED[family] + 1)
        rec["cma"] = {"loss": ind, "ok": bool(ind >= ret["loss"] - TOL)}
    return rec


def independent(s, X, y, lam, starts_n=16, seed=7, gens=300):
    """CMA-ES, signed output weights on the ℓ₁ sphere (v1's independent parametrisation), plus the same decay."""
    from .cmaes import cma_es
    rng = np.random.default_rng(seed)

    def to_P(q):
        W = q[:8].reshape(4, 2).copy(); c = q[8:12].copy(); u = q[12:16]
        sg = np.where(u < 0, -1.0, 1.0); W *= sg[:, None]; c *= sg
        return np.concatenate([W.ravel(), c, np.sqrt(np.abs(u) / max(np.abs(u).sum(), 1e-300))])

    def f(q):
        if np.abs(q[12:16]).sum() < 1e-12:
            return 1e9
        return float(loss_grad(to_P(q)[None], s, X, y, lam)[0][0])
    best = math.inf
    for _ in range(starts_n):
        q0 = np.concatenate([rng.uniform(-3, 3, 12), rng.uniform(-1, 1, 4)])
        r = cma_es(f, q0, 1.0, max_generations=gens, seed=int(rng.integers(1 << 31)))
        best = min(best, r.best_f)
    return best


def _part(fam, s):
    return OUT / "pilot_parts" / f"{fam}_{s:.6f}.json"


def _point(s, X, y, lam, fam, validate=False):
    f = _part(fam, s)
    if f.exists():
        r = json.loads(f.read_text())
        if not validate or "cma" in r:
            return r
    t0 = time.time()
    r = evaluate(s, X, y, lam, fam, validate)
    r["seconds"] = time.time() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(r)); tmp.replace(f)
    print(json.dumps({k: r[k] for k in ("set", "s", "loss", "rho2", "gplus", "seconds")} |
                     {"attained": r["attained"]["ok"], "hits": r["attained"].get("hits")}), flush=True)
    return r


def run_grid(fam):
    sb._init_worker()
    X, y = data(); lam = chosen_lambda()
    for s in GRID:
        _point(s, X, y, lam, fam)


def q_rule(rows_a, rows_b):
    """q = midway between the small-scale and large-scale ends of the path: ½(ρ̄₂(s_min) + ρ̄₂(s_max)), ρ̄₂ the mean of
    the two restart sets' retained minimisers at the first and last grid scales."""
    def at(rows, s):
        return [r["rho2"] for r in rows if r["s"] == s][0]
    lo = 0.5 * (at(rows_a, GRID[0]) + at(rows_b, GRID[0])); hi = 0.5 * (at(rows_a, GRID[-1]) + at(rows_b, GRID[-1]))
    return {"rho2_small": lo, "rho2_large": hi, "q": 0.5 * (lo + hi)}


def first_crossing(scales, rho, q):
    """First upward passage of q along increasing s (bracket), and the number of passages (either direction)."""
    o = np.argsort(scales); s = np.asarray(scales, float)[o]; r = np.asarray(rho, float)[o]
    above = r >= q
    up = np.flatnonzero(~above[:-1] & above[1:])
    return {"bracket": (float(s[up[0]]), float(s[up[0] + 1])) if len(up) else None,
            "passages": int((above[1:] != above[:-1]).sum()), "starts_below": bool(not above[0])}


def _rows(fam):
    return [json.loads(f.read_text()) for f in sorted((OUT / "pilot_parts").glob(f"{fam}_*.json"))]


def run_bisect(fam):
    sb._init_worker()
    X, y = data(); lam = chosen_lambda()
    while True:
        ra = [r for r in _rows("A") if r["s"] in GRID]; rb = [r for r in _rows("B") if r["s"] in GRID]
        if len(ra) == len(GRID) and len(rb) == len(GRID):
            break
        time.sleep(10)
    q = q_rule(ra, rb)["q"]
    rows = _rows(fam)
    fc = first_crossing([r["s"] for r in rows], [r["rho2"] for r in rows], q)
    if fc["bracket"] is None:
        print(json.dumps({"set": fam, "status": "q not reached", "q": q}), flush=True)
        return
    lo, hi = fc["bracket"]
    while hi / lo - 1 > BISECT_REL:
        mid = float(np.round(math.sqrt(lo * hi), 6))
        r = _point(mid, X, y, lam, fam)
        if r["rho2"] >= q:
            hi = mid
        else:
            lo = mid
    for s in (lo, hi):
        _point(s, X, y, lam, fam, validate=True)
    print(json.dumps({"set": fam, "status": "done", "q": q, "bracket": [lo, hi]}), flush=True)


# ------------------------------------------------------------------------------------------ the gate
def gate(sets, q):
    """sets: {name: {"rows": [...]}}.  Gate (all must hold):
      A attained: at each set's final bracket ends (the points that fix s_q) the retained minimiser passes the
        attainment check (converged, no flag, inside B(λ), reproduced by ≥ 2 restarts) and the projected Hessian is
        positive definite;
      B agreement: both sets reach q inside the grid and their s_q agree within 2% (|s_A − s_B| / min ≤ 0.02);
      C independent search: CMA-ES finds no lower loss (beyond 1e−7) at any bracket end."""
    per = {}
    sq = {}
    okA = okC = True
    for name, S in sets.items():
        rows = S["rows"]
        fc = first_crossing([r["s"] for r in rows], [r["rho2"] for r in rows], q)
        if fc["bracket"] is None:
            per[name] = {"reached": False, **fc}
            okA = okC = False
            sq[name] = None
            continue
        lo, hi = fc["bracket"]
        ends = [r for r in rows if r["s"] in (lo, hi)]
        a = all(r["attained"]["ok"] and r.get("hess_min_eig", -1) > 0 for r in ends) and len(ends) == 2
        c = all(r.get("cma", {}).get("ok", False) for r in ends) and len(ends) == 2
        sq[name] = math.sqrt(lo * hi)
        per[name] = {"reached": True, **fc, "s_q": sq[name], "attained": bool(a), "cma_ok": bool(c),
                     "bracket_rel": hi / lo - 1}
        okA &= a; okC &= c
    vals = [v for v in sq.values() if v is not None]
    if len(vals) == len(sq) and len(vals) >= 2:
        rel = (max(vals) - min(vals)) / min(vals); okB = rel <= AGREE_TOL
    else:
        rel, okB = None, False
    return {"per_set": per, "A_attained": bool(okA), "B_agreement": bool(okB), "C_independent": bool(okC),
            "agreement_rel": rel, "s_q": math.sqrt(vals[0] * vals[1]) if okB else None,
            "pass": bool(okA and okB and okC)}


def summarise():
    ra, rb = _rows("A"), _rows("B")
    q = q_rule([r for r in ra if r["s"] in GRID], [r for r in rb if r["s"] in GRID])
    v = gate({"A": {"rows": ra}, "B": {"rows": rb}}, q["q"])
    import pandas as pd
    tab = []
    for r in ra + rb:
        tab.append({"set": r["set"], "s": r["s"], "loss": r["loss"], "rho2": r["rho2"], "gplus": r["gplus"],
                    "S1": r["S1"], "S2": r["S2"], "attained": r["attained"]["ok"], "norm": r["attained"].get("norm"),
                    "hits": r["attained"].get("hits"), "ladder_ok": r["ladder_ok"], "audit_ok": r["audit"]["ok"],
                    "hess_min_eig": r.get("hess_min_eig"), "cma_loss": r.get("cma", {}).get("loss"),
                    "grid": r["s"] in GRID})
    pd.DataFrame(tab).sort_values(["set", "s"]).to_csv(OUT / "pilot_scan.csv", index=False)
    summ = {"label": "Track T v2 pilot (gate computation)", "lambda": chosen_lambda(), "q_rule": q, "gate": v,
            "n_points": len(tab), "n_attained": int(sum(t["attained"] for t in tab))}
    (OUT / "pilot_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


# ------------------------------------------------------------------------------------------ scoring rules (registered)
def log_errors(s_cross, pred, base):
    s_cross, pred, base = map(lambda a: np.asarray(a, float), (s_cross, pred, base))
    return np.abs(np.log(s_cross / pred)), np.abs(np.log(s_cross / base))


def paired_bootstrap(d, n_boot=10_000, seed=12345, level=0.95):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def score_rule(s_cross, pred, base, n_runs, min_cross=30, t1_max=0.10):
    """T1: median |log(s_cross/pred)| ≤ 0.10.  T3 (beats the baseline s_q): paired run-level bootstrap 95% interval of
    mean(|log err_pred| − |log err_base|) entirely below 0.  Validity: at least 30 crossings of n_runs."""
    n = len(s_cross)
    valid = n >= min_cross
    if n == 0:
        return {"valid": False, "n_cross": 0, "n_runs": n_runs}
    ep, eb = log_errors(s_cross, pred, base)
    lo, hi = paired_bootstrap(ep - eb)
    return {"valid": bool(valid), "n_cross": n, "n_runs": n_runs, "median_abs_log_err_pred": float(np.median(ep)),
            "median_abs_log_err_base": float(np.median(eb)), "T1": bool(valid and np.median(ep) <= t1_max),
            "T3_interval": [lo, hi], "T3": bool(valid and hi < 0), "pass": bool(valid and np.median(ep) <= t1_max and hi < 0)}


def crossing_step(rho_traj, q):
    """Training crossing: the first upward passage of q by ρ₂ (the first step t with ρ₂(t) ≥ q after some earlier step
    with ρ₂ < q).  None if there is none."""
    r = np.asarray(rho_traj, float)
    below = np.flatnonzero(r < q)
    if not len(below):
        return None
    after = np.flatnonzero(r[below[0]:] >= q)
    return None if not len(after) else int(below[0] + after[0])


def t0_step(s_traj, s_q, level=T0_LEVEL):
    """The last upward passage of level·s_q over the recorded trajectory (a rule on s only)."""
    s = np.asarray(s_traj, float); L = level * s_q
    up = np.flatnonzero((s[:-1] < L) & (s[1:] >= L))
    return None if not len(up) else int(up[-1] + 1)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "lambda":
        run_lambda()
    elif cmd == "grid":
        run_grid(sys.argv[2])
    elif cmd == "bisect":
        run_bisect(sys.argv[2])
    elif cmd == "summarise":
        summarise()
