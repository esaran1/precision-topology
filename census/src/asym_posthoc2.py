"""EXPLORATORY (author's request, 2026-09-25): two further looks at the T2-3 runs.  Nothing here changes T2-3's registered
verdict or WP-15's wording.

(1) Own-sample width-2 thresholds.  The T2-3 runs trained on their own sampled training sets
    (asym_register.training_set(seed)), not on the 800-point population the registered threshold used.  For 10 seeds
    (the first 10 crossing seeds in seed order), the width-2 conditional minimiser on the run's own training set is
    scanned at s = 10^(−1 + k/8) (200 restarts, cap 3,000, 5 polished; placement = continuous-window G₊ on the asymmetric
    windows, as T2-3's crossing) until the first placed scale after an unplaced one, then bisected to 5%.
(2) Timescale ratio at crossing (as Track 3): growth = d log‖v‖₁/dt over the last min(100, step − 1) steps; relax =
    lr·λ_min(D^{-1/2} H D^{-1/2}), H the Hessian of the training loss in (α₁, β₁, α₂, β₂, b) at v fixed, at the branch
    (damped Newton from the crossing state), D = diag(√v̂ + eps) from Adam's second moments of those coordinates.

    python -m src.asym_posthoc2 own | ratio | summary
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .asym_register import FROZEN, PARTS, _act, _setup, training_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "asym_posthoc"
N_SEEDS, RESTARTS, CAP = 10, 200, 3000
LR, EPS, BETA2 = 1e-2, 1e-8, 0.999


def _status(s, x, y, act, seed):
    from .asym_register import _retain
    from .width2_conditional import constant_predictor_loss, search_batch
    _, cands = search_batch(s, x, y, act, restarts=RESTARTS, seed=seed, maxit=CAP)
    cands = [c for c in cands if c["p"] is not None and np.all(np.isfinite(c["p"]))]
    return _retain(cands, s, x, y, act, constant_predictor_loss(y))["status"]


def own_threshold(seed):
    _setup(); act = _act()
    x, y = training_set(seed)
    grid = [10 ** (-1 + k / 8) for k in range(17)]
    prev, rows = None, []
    for i, s in enumerate(grid):
        st = _status(s, x, y, act, 20_000 + i)
        rows.append((s, st))
        if st == "placed" and prev == "unplaced":
            lo, hi = grid[i - 1], s
            j = 0
            while hi / lo - 1 > 0.05:
                mid = math.sqrt(lo * hi); j += 1
                stm = _status(mid, x, y, act, 21_000 + j)
                rows.append((mid, stm))
                if stm == "undecided":
                    return {"seed": seed, "s_own": float("nan"), "note": "undecided in bisection", "scan": rows}
                lo, hi = (lo, mid) if stm == "placed" else (mid, hi)
            later = [st2 for s2, st2 in rows if s2 < lo and st2 == "placed"]
            return {"seed": seed, "s_own_lo": lo, "s_own_hi": hi, "s_own": math.sqrt(lo * hi),
                    "note": "ok" if not later else "placed below the bracket (alternation)", "scan": rows}
        if st != "undecided":
            prev = st
    return {"seed": seed, "s_own": float("nan"), "note": "no unplaced-to-placed change on the grid", "scan": rows}


def _own_job(seed):
    os.nice(15)
    r = own_threshold(seed)
    r["scan"] = json.dumps(r["scan"])
    print(json.dumps({"seed": seed, "s_own": r["s_own"], "note": r["note"]}), flush=True)
    return r


def own(workers=1):
    from multiprocessing import get_context
    tr = pd.read_csv(PARTS / "train.csv")
    seeds = sorted(tr[tr.crossed & ~tr.placed_at_init].seed)[:N_SEEDS]
    with get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_own_job, seeds)
    pd.DataFrame(rows).to_csv(OUT / "own_thresholds.csv", index=False)


def ratio_one(seed, k, step):
    import torch
    from .width2_train import init_params, logits
    torch.set_num_threads(1)
    _setup(); act = _act()
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); q0[2] *= k; q0[5] *= k
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    win = min(100, int(step) - 1)
    path = {0: float(q0[2].abs() + q0[5].abs())}
    for t in range(1, int(step) + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        opt.step()
        if t >= step - win:
            path[t] = float(q.detach()[2].abs() + q.detach()[5].abs())
    qc = q.detach().clone()
    st = opt.state[q]
    vhat = (st["exp_avg_sq"].numpy() / (1 - BETA2 ** int(st["step"])))[[0, 1, 3, 4, 6]]
    growth = math.log(path[int(step)] / path[int(step) - win]) / win if win >= 1 else float("nan")
    idx = [0, 1, 3, 4, 6]

    def L(z):
        qq = qc.clone(); qq[idx] = z
        return torch.nn.functional.binary_cross_entropy_with_logits(logits(qq, X, act), Y)
    z = qc[idx].clone()
    mu, f0 = 1e-3, float(L(z))
    for _ in range(200):
        zz = z.clone().requires_grad_(True)
        g = torch.autograd.grad(L(zz), zz)[0]
        if float(g.abs().max()) < 1e-12:
            break
        H = torch.autograd.functional.hessian(L, z)
        ok = False
        for _ in range(30):
            d = torch.linalg.solve(H + mu * torch.eye(5, dtype=torch.float64) * max(1.0, float(H.diag().abs().max())), -g)
            f1 = float(L(z + d))
            if f1 < f0:
                z, f0, ok, mu = z + d, f1, True, max(mu / 3, 1e-12)
                break
            mu *= 10
        if not ok:
            break
    H = torch.autograd.functional.hessian(L, z).numpy()
    dd = 1.0 / np.sqrt(np.sqrt(vhat) + EPS)
    M = dd[:, None] * H * dd[None, :]
    relax = LR * float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())
    return {"seed": seed, "growth": growth, "relax": relax, "ratio": growth / relax if relax > 0 else float("nan"),
            "hess_min_eig": float(np.linalg.eigvalsh(H).min())}


def ratio():
    k = json.loads(FROZEN.read_text())["k"]
    tr = pd.read_csv(PARTS / "train.csv")
    tr = tr[tr.crossed & ~tr.placed_at_init]
    rows = [ratio_one(int(r.seed), k, int(r.step)) for r in tr.itertuples()]
    pd.DataFrame(rows).to_csv(OUT / "timescale.csv", index=False)


def summary():
    tr = pd.read_csv(PARTS / "train.csv"); tr = tr[tr.crossed & ~tr.placed_at_init]
    sc = json.loads((RESULTS / "asym_scores.json").read_text()); s_glob = math.sqrt(sc["s_lo"] * sc["s_hi"])
    out = {"label": "EXPLORATORY; does not change T2-3's registered verdict (FAIL)"}
    f = OUT / "own_thresholds.csv"
    if f.exists():
        o = pd.read_csv(f).merge(tr[["seed", "s_cross"]], on="seed")
        ok = o[np.isfinite(o.s_own)]
        e_own = np.abs(np.log(ok.s_cross / ok.s_own)); e_pop = np.abs(np.log(ok.s_cross / s_glob))
        out.update({"own_n": len(o), "own_defined": len(ok), "own_notes": o.note.value_counts().to_dict(),
                    "own_median": float(ok.s_own.median()), "own_range": [float(ok.s_own.min()), float(ok.s_own.max())],
                    "pop_threshold": s_glob, "median_cross_over_own": float(np.median(ok.s_cross / ok.s_own)),
                    "median_cross_over_pop": float(np.median(ok.s_cross / s_glob)),
                    "mean_abslog_own": float(e_own.mean()), "mean_abslog_pop": float(e_pop.mean()),
                    "runs_closer_to_own": int((e_own < e_pop).sum()),
                    "spearman_cross_own": float(np.corrcoef(ok.s_cross.rank(), ok.s_own.rank())[0, 1]) if len(ok) > 2 else None})
    f = OUT / "timescale.csv"
    if f.exists():
        t = pd.read_csv(f).merge(tr[["seed", "s_cross"]], on="seed")
        fit = json.loads((RESULTS / "residual_timescale_fit.json").read_text())
        good = t[np.isfinite(t.ratio) & (t.ratio > 0)]
        out.update({"ts_n": len(t), "ts_usable": len(good), "ts_nonpositive_relax": int((t.relax <= 0).sum()),
                    "ts_median_ratio": float(good.ratio.median()) if len(good) else None,
                    "ts_ratio_range": [float(good.ratio.min()), float(good.ratio.max())] if len(good) else None,
                    "width1_fitted_ratio_range": fit["ratio_range_fitted"],
                    "width1_fit_predicted_median_residual": float(np.median(fit["alpha"] + fit["beta"] * good.ratio)) if len(good) else None,
                    "observed_median_residual_vs_pop": float(np.median(tr.s_cross / s_glob) - 1),
                    "spearman_residual_ratio_within_T2_3": float(np.corrcoef((good.s_cross / s_glob).rank(),
                                                                             good.ratio.rank())[0, 1])})
    (OUT / "exploratory2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"own": lambda: own(int(sys.argv[2]) if len(sys.argv) > 2 else 1), "ratio": ratio, "summary": summary}[sys.argv[1]]()
