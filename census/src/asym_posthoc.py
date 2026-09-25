"""POST HOC, EXPLORATORY (author's request 2026-09-25, after T2-3 failed): what do the T2-3 runs cross on?

Definitions (fixed before running):
  knockout class at the crossing: 'single-unit' if exactly one unit alone (φ = vᵢuᵢ) has G₊ > 0; 'pair' (cooperating)
      if neither alone does but both together do; 'redundant' if each alone does.
  trajectory: at log-spaced steps up to the crossing, each unit's weight share |vᵢ|/‖v‖₁ and the ramp-cancellation
      index |Σvᵢαᵢ|/Σ|vᵢαᵢ|; 'dominant unit' = share ≥ 0.75.
  branch switch scale (per run): from the run's configuration at its crossing, damped Newton on the width-2 joint loss
      at a trial scale s; the smallest s whose local minimiser is placed (bisection in log s on [0.02, s_cross], to 2%).
  width-1 threshold: the width-1 conditional minimiser (ṽ fixed on one unit; asym_pilot's width-1 search, 400
      restarts, Newton-free polish) scanned on s = 10^(−1 + k/8) and bisected to 2%.
Comparison: per-run |log(s_cross / s_ref)| for s_ref = the width-2 global threshold (T2-1 bracket midpoint), the width-1
threshold, and the run's own branch switch scale.

    python -m src.asym_posthoc run
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .asym_register import BUDGET, FROZEN, PARTS, _act, _setup, training_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "asym_posthoc"


def _gplus(th, v, act):
    from .width2_geometry import gaps
    return gaps(th, v, act)["G+"]


def features(q, act):
    """q = (α₁, β₁, v₁, α₂, β₂, v₂, b) in width2_train's layout."""
    th = np.array([q[0], q[1], q[3], q[4]]); v = np.array([q[2], q[5]])
    n1 = np.abs(v).sum()
    share = np.abs(v) / n1
    alone = [_gplus(th, np.array([np.sign(v[0]), 0.0]), act)[0] > 0, _gplus(th, np.array([0.0, np.sign(v[1])]), act)[0] > 0]
    both = _gplus(th, v / n1, act)[0] > 0
    if alone[0] and alone[1]:
        ko = "redundant"
    elif alone[0] or alone[1]:
        ko = "single-unit"
    else:
        ko = "pair" if both else "unplaced"
    va = np.array([v[0] * q[0], v[1] * q[3]])
    cancel = float(abs(va.sum()) / max(np.abs(va).sum(), 1e-300))
    return {"max_share": float(share.max()), "cancel_index": cancel, "knockout": ko, "s": float(n1)}


def trajectory(seed, k, cross_step):
    import torch
    from .width2_train import LR, init_params, logits
    torch.set_num_threads(1)
    _setup(); act = _act()
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); q0[2] *= k; q0[5] *= k
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    marks = set(np.unique(np.logspace(0, math.log10(max(cross_step, 2)), 25).astype(int))) | {cross_step}
    traj = []
    for step in range(1, int(cross_step) + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        opt.step()
        if step in marks:
            f = features(q.detach().numpy(), act)
            traj.append({"step": step, **f})
    return traj, q.detach().numpy().copy()


def branch_switch(qc, act, x, y, s_cross, lo=0.02, rel=0.02):
    """Bisection in log s for the smallest s at which Newton from the crossing configuration lands placed."""
    from .width2_conditional import profile_b
    from .width2_finish import newton
    from .width2_unplaced import g_exact
    z0 = np.r_[qc[0], qc[1], qc[3], qc[4], qc[2], qc[5]]
    n1 = abs(z0[4]) + abs(z0[5]); z0[4:6] /= n1

    def placed_at(s):
        b = profile_b(s * (z0[4] * act.u(z0[0] * x + z0[1]) + z0[5] * act.u(z0[2] * x + z0[3])), y)
        z, *_ = newton(np.r_[z0, b], s, x, y, act, maxit=200)
        return bool(g_exact(z[:6], act)[0] > 0)
    hi = s_cross
    if not placed_at(hi):
        hi = s_cross * 4
        if not placed_at(hi):
            return float("nan"), "not placed at 4x s_cross"
    if placed_at(lo):
        return lo, "placed at the lower end"
    while hi / lo - 1 > rel:
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if placed_at(mid) else (mid, hi)
    return math.sqrt(lo * hi), "ok"


def _run_job(args):
    os.nice(15)
    seed, k, step, s_cross = args
    x, y = _setup()
    act = _act()
    traj, qc = trajectory(seed, k, step)
    sb, note = branch_switch(qc, act, x, y, s_cross)
    f = features(qc, act)
    pre = [t for t in traj if t["step"] < step]
    return {"seed": seed, "cross_step": step, "s_cross": s_cross, "s_replay": f["s"], "knockout": f["knockout"],
            "max_share_at_cross": f["max_share"], "cancel_at_cross": f["cancel_index"],
            "frac_pre_dominant": float(np.mean([t["max_share"] >= 0.75 for t in pre])) if pre else np.nan,
            "s_branch": sb, "branch_note": note, "traj": json.dumps(traj)}


def width1_threshold():
    """Width-1 conditional minimiser on the Δ = 0.4 population (asym_pilot's width-1 search), scan then bisection."""
    from . import asym_pilot as ap
    from .width2_conditional import constant_predictor_loss
    x, y = _setup(); act = _act()
    const = constant_predictor_loss(y)

    def status(s, seed):
        cands = [c for c in ap._width1_batch(s, x, y, act, 400, seed) if np.all(np.isfinite(c["p"]))]
        cands.sort(key=lambda c: c["loss"])
        pol = [ap._polish(c, s, x, y, act, "width1") for c in cands[:5]]
        best = min(pol, key=lambda r: r["loss"])
        return bool(best["placed"] and best["loss"] <= const), best["loss"]
    grid = [10 ** (-1 + k / 8) for k in range(25)]
    rows = [{"s": s, "placed": status(s, 9700 + i)[0]} for i, s in enumerate(grid)]
    pl = [r["placed"] for r in rows]
    changes = sum(pl[i] != pl[i + 1] for i in range(len(pl) - 1))
    k = next((i for i, p in enumerate(pl) if p), None)
    if k is None or k == 0:
        return {"scan": rows, "changes": changes, "s_w1": float("nan")}
    lo, hi = grid[k - 1], grid[k]
    j = 0
    while hi / lo - 1 > 0.02:
        mid = math.sqrt(lo * hi); j += 1
        lo, hi = (lo, mid) if status(mid, 9900 + j)[0] else (mid, hi)
    return {"scan": rows, "changes": changes, "s_w1_lo": lo, "s_w1_hi": hi, "s_w1": math.sqrt(lo * hi)}


def run(workers=1):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    k = json.loads(FROZEN.read_text())["k"]
    tr = pd.read_csv(PARTS / "train.csv")
    tr = tr[tr.crossed & ~tr.placed_at_init]
    jobs = [(int(r.seed), k, int(r.step), float(r.s_cross)) for r in tr.itertuples()]
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_run_job, jobs)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "runs.csv", index=False)
    w1 = width1_threshold()
    (OUT / "width1.json").write_text(json.dumps(w1, indent=1, default=float))
    summarise()


def summarise():
    d = pd.read_csv(OUT / "runs.csv")
    w1 = json.loads((OUT / "width1.json").read_text())
    sc = json.loads((RESULTS / "asym_scores.json").read_text())
    s_glob = math.sqrt(sc["s_lo"] * sc["s_hi"])
    ok = d[np.isfinite(d.s_branch)]
    err = lambda ref: np.abs(np.log(ok.s_cross / ref))
    rng = np.random.default_rng(0)

    def boot(v):
        m = np.array([rng.choice(v, len(v)).mean() for _ in range(10_000)])
        return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    e_glob, e_w1, e_br = err(s_glob), err(w1["s_w1"]), err(ok.s_branch)
    out = {"n_runs": len(d), "n_branch_defined": len(ok), "replay_reproduces": bool(np.allclose(d.s_replay, d.s_cross, rtol=0, atol=1e-12)),
           "knockout_counts": d.knockout.value_counts().to_dict(),
           "median_max_share_at_cross": float(d.max_share_at_cross.median()),
           "median_cancel_at_cross": float(d.cancel_at_cross.median()),
           "median_frac_pre_dominant": float(d.frac_pre_dominant.median()),
           "s_glob_w2": s_glob, "s_w1": w1["s_w1"], "w1_scan_changes": w1["changes"],
           "median_s_branch": float(ok.s_branch.median()),
           "median_ratio_glob": float(np.median(ok.s_cross / s_glob)),
           "median_ratio_w1": float(np.median(ok.s_cross / w1["s_w1"])),
           "median_ratio_branch": float(np.median(ok.s_cross / ok.s_branch)),
           "mean_abslog_glob": boot(e_glob.values), "mean_abslog_w1": boot(e_w1.values),
           "mean_abslog_branch": boot(e_br.values),
           "branch_minus_glob": boot((e_br - e_glob).values), "w1_minus_glob": boot((e_w1 - e_glob).values),
           "spearman_cross_branch": float(np.corrcoef(ok.s_cross.rank(), ok.s_branch.rank())[0, 1]),
           "branch_notes": d.branch_note.value_counts().to_dict()}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def single_unit_branch(grid=None, share2=0.01):
    """The explicit single-unit branch of the width-2 conditional loss: at each s, the width-1 conditional minimiser
    (400 restarts, polished) embedded with the second unit at weight share 1% (a copy of the first unit's (α, β)),
    Newton-minimised on the width-2 joint loss.  Recorded: placed?, knockout class, max share.  The branch's switch
    scale is the first grid scale at which the result is placed while still single-unit (share of the carrier ≥ 0.75)."""
    from . import asym_pilot as ap
    from .width2_conditional import profile_b
    from .width2_finish import newton
    from .width2_unplaced import g_exact
    x, y = _setup(); act = _act()
    grid = grid or [10 ** (-0.5 + k / 16) for k in range(25)]
    rows = []
    for i, s in enumerate(grid):
        cands = [c for c in ap._width1_batch(s, x, y, act, 400, 9800 + i) if np.all(np.isfinite(c["p"]))]
        cands.sort(key=lambda c: c["loss"])
        w1 = ap._polish(cands[0], s, x, y, act, "width1")["q"]
        sg = np.sign(w1[4])
        q = np.r_[w1[0], w1[1], w1[0], w1[1], sg * (1 - share2), sg * share2]
        b = profile_b(s * (q[4] * act.u(q[0] * x + q[1]) + q[5] * act.u(q[2] * x + q[3])), y)
        z, *_ = newton(np.r_[q, b], s, x, y, act, maxit=300)
        qq = np.r_[z[0], z[1], z[4], z[2], z[3], z[5], z[6]]
        f = features(qq, act)
        placed = bool(g_exact(z[:6], act)[0] > 0)
        rows.append({"s": s, "placed": placed, "knockout": f["knockout"], "max_share": f["max_share"],
                     "single_unit": f["max_share"] >= 0.75})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "single_unit_branch.csv", index=False)
    hit = d[d.placed & d.single_unit]
    lost = d[~d.single_unit]
    return {"s_single_first_placed": float(hit.s.min()) if len(hit) else float("nan"),
            "s_single_last_unplaced_before": float(d[(d.s < hit.s.min()) & ~d.placed].s.max()) if len(hit) else float("nan"),
            "branch_leaves_single_unit_at": float(lost.s.min()) if len(lost) else float("nan")}


if __name__ == "__main__":
    {"run": lambda: run(int(sys.argv[2]) if len(sys.argv) > 2 else 1), "summarise": summarise,
     "single": lambda: print(json.dumps(single_unit_branch(), indent=1))}[sys.argv[1]]()
