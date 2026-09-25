"""POST HOC, EXPLORATORY (author's request 2026-09-25): do the T2-3 and T2-3c runs pass through Track 7's stuck states?

States are not logged, so each run is replayed deterministically (its recorded crossing step and ‖v‖₁ must reproduce).
At 50 evenly spaced pre-crossing steps, each state is classified with Track 7's criteria (width2_nogating): stationary
two-unit configuration = unplaced (exact G₊ ≤ 0), both weight shares > SINGLE_SHARE (0.01), scale-relative tangent gradient
max|∇L|/‖w₂‖₁ ≤ STATIONARY_TOL (1e−6) and free-direction Hessian λ_min/‖w₂‖₁ ≥ −HESS_TOL (−1e−6).
Reported per arm: fraction of pre-crossing checks that are stationary two-unit configurations; fraction of runs with any
such episode; among those, whether the run had left it (non-stationary at the last check before crossing: 'escape then
cross') or was still in it at the last check.  Also the weight share and cancellation index at those states, beside
Track 7's unplaced endpoints.  No verdict changes.

    python -m src.asym_stuck
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .asym_register import _act, _setup, training_set

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "asym_posthoc"
N_CHECK = 50


def classify(q, act, x, y):
    from .width2_nogating import HESS_TOL, SINGLE_SHARE, STATIONARY_TOL, tangent_grad_max, tangent_hessian_min
    from .width2_train import placed
    v = np.array([q[2], q[5]]); n1 = float(np.abs(v).sum())
    share = np.abs(v) / n1
    va = np.array([v[0] * q[0], v[1] * q[3]])
    cancel = float(abs(va.sum()) / max(np.abs(va).sum(), 1e-300))
    pl = bool(placed(q, act)[0])
    g = tangent_grad_max(q, act, x, y) / n1
    lam = tangent_hessian_min(q, act, x, y) / n1
    stat2 = (not pl) and share.min() > SINGLE_SHARE and g <= STATIONARY_TOL and lam >= -HESS_TOL
    return {"placed": pl, "min_share": float(share.min()), "cancel": cancel, "grad_rel": g, "lam_rel": lam,
            "stationary_two_unit": bool(stat2)}


def replay(arm, seed, k, phi, step, s_cross):
    import torch
    from .width2_train import LR, init_params, logits
    torch.set_num_threads(1)
    _setup(); act = _act()
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); q0[2] *= k; q0[5] *= k
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    checks = set(np.unique(np.linspace(1, max(step - 1, 1), N_CHECK).astype(int))) if step > 1 else set()
    rows = []
    for t in range(1, int(step) + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        vb = q.detach()[[2, 5]].clone()
        opt.step()
        if phi != 1.0:
            with torch.no_grad():
                q[[2, 5]] = vb + phi * (q[[2, 5]] - vb)
        if t in checks and t < step:
            rows.append({"arm": arm, "seed": seed, "t": t, "frac_time": t / step,
                         **classify(q.detach().numpy().copy(), act, x, y)})
    s_end = float(q.detach()[2].abs() + q.detach()[5].abs())
    return rows, abs(s_end - s_cross) <= 1e-12


def _job(args):
    os.nice(15)
    rows, ok = replay(*args)
    return rows, (args[0], args[1], ok)


def run(workers=1):
    from multiprocessing import get_context
    k = json.loads((RESULTS / "asym_frozen.json").read_text())["k"]
    jobs = []
    t3 = pd.read_csv(RESULTS / "asym_parts" / "train.csv"); t3 = t3[t3.crossed & ~t3.placed_at_init]
    jobs += [("T2-3", int(r.seed), k, 1.0, int(r.step), float(r.s_cross)) for r in t3.itertuples()]
    tc = pd.read_csv(RESULTS / "asym_t23c" / "train.csv"); tc = tc[tc.crossed & ~tc.placed_at_init]
    phi = json.loads((RESULTS / "asym_t23c_frozen.json").read_text())["phi2"]
    jobs += [("T2-3c", int(r.seed), k, phi, int(r.step), float(r.s_cross)) for r in tc.itertuples()]
    with get_context("spawn").Pool(workers) as pool:
        out = pool.map(_job, jobs)
    d = pd.DataFrame([r for rows, _ in out for r in rows])
    rep = pd.DataFrame([o for _, o in out], columns=["arm", "seed", "reproduced"])
    d.to_csv(OUT / "stuck_states.csv", index=False)
    return summarise(d, rep)


def summarise(d, rep):
    ng = pd.read_csv(RESULTS / "width2_nogating_parts" / "replays.csv")
    ng = ng[(ng.step == 16000) & (ng.act != "tanh") & (ng.variant == "preserved") & (ng.placed == False)]
    out = {"label": "POST HOC, EXPLORATORY; no verdict changes", "replays_reproduce": bool(rep.reproduced.all())}
    for arm, g in d.groupby("arm"):
        per = g.groupby("seed")
        any_ep = per.stationary_two_unit.any()
        last = g.sort_values("t").groupby("seed").tail(1).set_index("seed").stationary_two_unit
        ep = any_ep[any_ep].index
        st = g[g.stationary_two_unit]
        out[arm] = {"runs": int(g.seed.nunique()),
                    "frac_checks_stationary_two_unit": float(g.stationary_two_unit.mean()),
                    "median_run_frac_time_stationary": float(per.stationary_two_unit.mean().median()),
                    "runs_with_episode": int(len(ep)),
                    "of_those_escaped_before_crossing": int((~last.loc[ep]).sum()),
                    "of_those_still_stationary_at_last_check": int(last.loc[ep].sum()),
                    "stationary_median_min_share": float(st.min_share.median()) if len(st) else None,
                    "stationary_median_cancel": float(st.cancel.median()) if len(st) else None}
    out["track7_unplaced_endpoints"] = {"n": int(len(ng)),
                                        "median_min_share": float(np.minimum(ng.share1, ng.share2).median()) if "share1" in ng else None,
                                        "median_cancel": float(ng.cancel_index.median()) if "cancel_index" in ng else None,
                                        "median_grad_rel": float(ng.grad_rel.median())}
    (OUT / "stuck_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
