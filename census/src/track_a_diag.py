"""Track A, POST HOC diagnostic (author-approved, after the registered score 74b3c54): Adam ordering.

LABEL: POST HOC.  P at the observed crossing uses crossing-time information, so it is a DIAGNOSTIC, never a prediction;
the registered Adam L3 FAIL stands.

For every registered Adam run at a = 1.65 the registered pipeline (`track_a.predict_one`, unchanged) is re-run with the
preconditioner frozen at a different step, everything else identical (rule point, branch, δ₀, m₀, H and θ* path, s path,
stability rule).  Implementation: predict_one reads v̂ only at the rule point t_R, so v̂[t_R] is replaced by v̂ at the
chosen step.  Variants:
    registered   v̂ unchanged (must reproduce the committed predictions exactly: machinery check);
    p_cross      v̂ at the run's OBSERVED crossing step (diagnostic; uses crossing information);
    p_switch     v̂ at t_sw = the first step with s_t ≥ s*_run (the occupied branch's switch; pre-crossing whenever the
                 lag is positive; a candidate rule for a future registered test, checked here to precede the crossing).
The Adam moments are recomputed by deterministic retraining (`track_a.train_one`); the retrained parameter path must
equal the saved, hashed path bit for bit.

    python -m src.track_a_diag          -> results/track_a/diag_p_at_crossing.csv / .json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import track_a as T

OUT = T.OUT


def metrics(obs, pred):
    """Spearman, fraction with |obs/pred − 1| ≤ 0.10, median obs/pred, q10/q90 of both, over finite pairs."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[ok], pred[ok]
    if len(o) < 3:
        return {"n": int(len(o))}
    r = o / p
    return {"n": int(len(o)), "spearman": T.spearman(p, o), "frac_within_10pct": float(np.mean(np.abs(r - 1) <= 0.10)),
            "median_obs_over_pred": float(np.median(r)), "pred_q10": float(np.quantile(p, 0.1)),
            "pred_q90": float(np.quantile(p, 0.9)), "obs_q10": float(np.quantile(o, 0.1)), "obs_q90": float(np.quantile(o, 0.9)),
            "ratio_q10": float(np.quantile(r, 0.1)), "ratio_q90": float(np.quantile(r, 0.9))}


def run():
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    obs = pd.read_csv(OUT / "observed_runs.csv")
    fr = pd.read_csv(OUT / "frozen_seeds.csv").set_index("seed")
    g = obs[(obs.opt == "adam") & (obs.status == "ok")]
    rows = []
    for r in g.itertuples():
        W, M, V, K = T.train_one("adam", int(r.seed))
        saved = np.load(T._path_file("adam", int(r.seed)))["W"]
        assert np.array_equal(W, saved), f"retrained path differs from the saved path (seed {r.seed})"
        s_fr = float(fr.loc[int(r.seed), "s_frozen"])
        tR = int(r.t_rule)
        s = np.abs(W[:, 2])
        t_sw = T.LR._first_ge(s, float(r.s_run))
        row = {"seed": int(r.seed), "crossed": bool(r.crossed), "step_obs": r.step_obs, "t_rule": tR, "t_sw": t_sw,
               "r_obs": r.r_obs, "r_traj_committed": r.r_traj, "r_cf_committed": r.r_cf}
        steps = {"registered": tR, "p_switch": t_sw,
                 "p_cross": int(r.step_obs) if bool(r.crossed) and np.isfinite(r.step_obs) else None}
        for name, t_p in steps.items():
            if t_p is None:
                row[f"{name}_r_traj"] = row[f"{name}_r_cf"] = float("nan")
                continue
            Vm = V.copy(); Vm[tR] = V[t_p]
            p = T.predict_one("adam", int(r.seed), W, M, Vm, K, s_fr)
            row[f"{name}_r_traj"] = p.get("r_traj", float("nan")); row[f"{name}_r_cf"] = p.get("r_cf", float("nan"))
            row[f"{name}_status_traj"] = p.get("status_traj")
        row["p_switch_before_crossing"] = bool(t_sw is not None and np.isfinite(r.step_obs) and t_sw < r.step_obs)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("seed", "r_obs", "registered_r_traj", "p_cross_r_traj", "p_switch_r_traj")},
                         default=float), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "diag_p_at_crossing.csv", index=False)
    # the committed values went through a CSV round trip (last-ulp rounding): compare to 1e-12 relative
    same = np.allclose(d.registered_r_traj, d.r_traj_committed, equal_nan=True, rtol=1e-12, atol=0) and \
        np.allclose(d.registered_r_cf, d.r_cf_committed, equal_nan=True, rtol=1e-12, atol=0)
    c = d[d.crossed]
    res = {"label": "POST HOC diagnostic (P at crossing uses crossing-time information; never a prediction; the "
                    "registered Adam L3 FAIL stands)",
           "registered_reproduced_exactly": bool(same), "n_runs": int(len(d)), "n_crossed": int(len(c)),
           "n_p_switch_before_crossing": int(c.p_switch_before_crossing.sum())}
    for name in ("registered", "p_cross", "p_switch"):
        res[name] = {"traj": metrics(c.r_obs, c[f"{name}_r_traj"]), "closed_form": metrics(c.r_obs, c[f"{name}_r_cf"])}
    (OUT / "diag_p_at_crossing.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    run()
