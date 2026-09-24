"""The best UNPLACED conditional configuration at each small scale of the direct check (author's request 2026-09-24,
option (a)): so that the selection of the placed minimiser can be seen to be resolved, not within numerical noise.

At scale s (R₂ = sΓ̂₂/2), minimise L*(θ; s) over {G₊(φ) ≤ 0}:
  * parametrisation: q = (α₁, β₁, α₂, β₂, v₁, v₂), v normalised to ‖v‖₁ = 1 (as width2_w0._local_from_stall);
  * feasibility during the search by the dense-grid G₊ (width2_train.dense_gplus), an UPPER bound on the exact G₊,
    so a feasible point is truly unplaced; the reported point is re-verified with exact extrema;
  * starts: the unplaced candidates (exact G₊ ≤ 0) of an unconstrained 400-restart search at the same scale, the
    analytic single unit at α*, and random points; CMA-ES with an infeasibility penalty from each;
  * classification of the best point: an unconstrained local CMA-ES from it.  If the loss can still fall by more than
    1e−12 and the descent ends placed, the minimum is on the boundary G₊ = 0 (the constraint is active); if it cannot
    fall, it is an interior local minimum of the unconstrained problem; otherwise 'unresolved'.

    python -m src.width2_unplaced <act>           # all four scales of width2_w0.SMALL_R2
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARTS = RESULTS / "width2_w0_parts"
DESCENT_TOL = 1e-12


def _split(q):
    v = np.asarray(q[4:6], float)
    n1 = abs(v[0]) + abs(v[1])
    return np.asarray(q[:4], float), v / n1 if n1 > 0 else v


def loss(q, s, x, y, act):
    from .width2_conditional import _softplus, profile_b
    th, v = _split(q)
    if not np.isfinite(v).all() or abs(v).sum() == 0:
        return 1e9
    z0 = s * (v[0] * act.u(th[0] * x + th[1]) + v[1] * act.u(th[2] * x + th[3]))
    z = z0 + profile_b(z0, y)
    return float((_softplus(z) - y * z).mean())


def g_dense(q, act):
    from .width2_train import dense_gplus
    th, v = _split(q)
    return dense_gplus(th, v, act)


def g_exact(q, act):
    from .width2_geometry import gaps
    th, v = _split(q)
    return gaps(th, v, act)["G+"]


def classify(best_loss, g_hi, local_loss, local_g_lo, tol=DESCENT_TOL):
    """'boundary' (constraint active), 'interior' (unconstrained local minimum), or 'unresolved'."""
    if g_hi > 0:
        return "infeasible"
    fell = local_loss < best_loss - tol
    if fell and local_g_lo is not None and local_g_lo > 0:
        return "boundary"
    if not fell:
        return "interior"
    return "unresolved"


def best_unplaced(act_name, R2, restarts=400, n_random=24, gens=600, seed=11):
    from .cmaes import cma_es
    from .width2_conditional import population, search_batch
    from .width2_w0 import ACTS, _read
    act = ACTS[act_name]
    gh = float(_read(f"gamma_{act_name}.csv").query("search == 'nm'").gamma_lo.iloc[0])
    x, y = population()
    s = 2 * R2 / gh
    f = lambda q: loss(q, s, x, y, act)

    def pen(q):
        L = f(q)
        g = g_dense(q, act)
        return L if g <= 0 else L + 1.0 + g
    rng = np.random.default_rng(seed)
    starts = []
    _, cands = search_batch(s, x, y, act, restarts=restarts, seed=seed, maxit=2000)
    un = []
    for c in cands[:-1]:
        if c["p"] is None or not np.isfinite(c["loss"]):
            continue
        t = float(np.clip(c["p"][4], -1, 1))
        q = np.r_[c["p"][:4], t, c["sigma"] * (1 - abs(t))]
        if g_exact(q, act)[1] <= 0:
            un.append((c["loss"], q))
    un.sort(key=lambda z: z[0])
    starts += [(q, 0.05, "unplaced candidate") for _, q in un[:12]]
    al = 1.7913244
    for sg in (1.0, -1.0):                                             # analytic single units at α*
        starts.append((np.array([al, sg * math.pi / 2, 1.0, 0.0, 1.0, 0.0]), 0.05, "single unit at alpha*"))
    for _ in range(n_random):
        starts.append((np.r_[rng.uniform(-6, 6), rng.uniform(0, 2 * math.pi), rng.uniform(-6, 6),
                             rng.uniform(0, 2 * math.pi), rng.uniform(-1, 1, 2)], 1.0, "random"))
    found = []
    for q0, sig, kind in starts:
        r = cma_es(pen, q0, sig, max_generations=gens, seed=int(rng.integers(1 << 30)))
        found.append((r.best_f, r.best_x, kind))
    feas = [(fv, q, k) for fv, q, k in found if g_exact(q, act)[1] <= 0]
    bl, bq, bk = min(feas, key=lambda z: z[0])
    ge = g_exact(bq, act)
    loc = cma_es(f, bq, 0.01, max_generations=400, seed=1)
    lg = g_exact(loc.best_x, act)
    cls = classify(bl, ge[1], loc.best_f, lg[0])
    th, v = _split(bq)
    c_lin = float(v[0] * th[0] + v[1] * th[2])
    return {"act": act_name, "R2": R2, "s": s, "best_unplaced_loss": bl, "G_lo": ge[0], "G_hi": ge[1],
            "classification": cls, "local_descent_loss": loc.best_f, "local_descent_G_lo": lg[0],
            "local_descent_drop": bl - loc.best_f, "start_kind": bk, "n_starts": len(found), "n_feasible": len(feas),
            "n_unplaced_candidates": len(un), "weights": json.dumps([float(v[0]), float(v[1])]),
            "alphas": json.dumps([float(th[0]), float(th[2])]), "linear_c": c_lin}


def run(act_name):
    from .width2_w0 import SMALL_R2
    out = PARTS / f"unplaced_{act_name}.csv"
    done = set() if not out.exists() else set(np.round(pd.read_csv(out).R2, 10))
    for R2 in SMALL_R2:
        if round(R2, 10) in done:
            continue
        row = best_unplaced(act_name, R2)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(json.dumps(row), flush=True)


def report(act_name):
    """Join with the direct check's retained (placed) minimiser, scale by scale."""
    sm = pd.read_csv(PARTS / f"smallscale_{act_name}.csv", float_precision="round_trip")
    un = pd.read_csv(PARTS / f"unplaced_{act_name}.csv", float_precision="round_trip")
    m = sm.merge(un, on=["act", "R2"], suffixes=("", "_u"))
    from .width2_conditional import population
    x, _ = population()
    alpha_star = 1.7913244
    m["best_placed_loss"] = np.where(m.placed, m.retained_loss, np.nan)
    m["difference"] = m.best_unplaced_loss - m.best_placed_loss
    m["difference_over_tie_tol_1e-9"] = m.difference / 1e-9
    m["predicted_single_unit_gap"] = m.s ** 2 / 8 * alpha_star ** 2 * float(np.var(x))
    return m


if __name__ == "__main__":
    run(sys.argv[1])
