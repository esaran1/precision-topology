"""Final checks (2026-09-26; existing data only).

(2) Appendix (Track A, a = 1.65, registered): per optimiser, observed and predicted median lag, median per-run ratio,
    fraction within 10%, median absolute error and Spearman, for the registered trajectory-integrated (L1/L3) and
    closed-form (L2) predictions, on the registered set (crossed, status ok, rule before crossing).
(4) POST HOC, no fitting to the a = 1.65 runs: the mechanistic predictors (registered trajectory-integrated κχ and
    closed-form κχ) against two baselines, per optimiser:
      constant   the pooled median lag of the same optimiser's earlier runs at a = 1.30–1.60 (1,750 runs of the lag-law
                 comparison; lag from each run's tracked-branch switch, the same reference as the a = 1.65 target);
      frozen fit residual = α + β·ratio (residual_timescale_fit.json, SHA-256 16792946…, fitted on 384 Adam runs at
                 a = 1.30/1.50 before any a = 1.65 data existed), applied with each run's pre-crossing χ from Track A.
    Metrics against the registered target r_obs (lag from the run's frozen branch switch): median |obs − pred|,
    fraction with |obs/pred − 1| ≤ 0.10, Spearman(pred, obs) (undefined for a constant). Sensitivity: the same against
    r_obs_vs_global_own (the reference the frozen fit was built on).

    python -m src.final_checks
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "final_checks"


def _metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.broadcast_to(np.asarray(pred, float), np.shape(obs)).astype(float)
    ok = np.isfinite(obs) & np.isfinite(pred) & (pred != 0)
    o, p = obs[ok], pred[ok]
    sp = float(pd.Series(p).rank().corr(pd.Series(o).rank())) if np.ptp(p) > 0 else float("nan")
    return {"n": int(ok.sum()), "median_obs": float(np.median(o)), "median_pred": float(np.median(p)),
            "median_ratio": float(np.median(o / p)), "frac_within_10": float((np.abs(o / p - 1) <= 0.10).mean()),
            "median_abs_err": float(np.median(np.abs(o - p))), "spearman": sp}


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(RESULTS / "track_a" / "observed_runs.csv")
    d = d[(d.crossed == True) & (d.status == "ok") & (d.rule_before_cross == True)]            # noqa: E712
    fit = json.loads((RESULTS / "residual_timescale_fit.json").read_text())
    lr = pd.read_csv(RESULTS / "linear_response" / "compare_runs.csv")
    const = {"adam": float(lr[lr["set"] != "SGD"].r_obs.median()), "sgd": float(lr[lr["set"] == "SGD"].r_obs.median())}
    out = {"appendix": {}, "comparison": {}, "comparison_global_own_reference": {}, "constant_lag": const,
           "frozen_fit": {"alpha": fit["alpha"], "beta": fit["beta"]}}
    for opt, g in d.groupby("opt"):
        out["appendix"][opt] = {"trajectory": _metrics(g.r_obs, g.r_traj), "closed_form": _metrics(g.r_obs, g.r_cf)}
        preds = {"mechanistic, trajectory (registered)": g.r_traj, "mechanistic, closed form (registered)": g.r_cf,
                 "constant lag from a = 1.30–1.60": np.full(len(g), const[opt]),
                 "frozen fitted relationship": fit["alpha"] + fit["beta"] * g.chi}
        out["comparison"][opt] = {k: _metrics(g.r_obs, v) for k, v in preds.items()}
        out["comparison_global_own_reference"][opt] = {k: _metrics(g.r_obs_vs_global_own, v) for k, v in preds.items()}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    rows = [{"reference": ref, "opt": o, "predictor": k, **v} for ref, key in (("branch switch (registered target)", "comparison"),
            ("global own threshold (sensitivity)", "comparison_global_own_reference")) for o, dd in out[key].items() for k, v in dd.items()]
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(json.dumps(out["appendix"], indent=1)); print(pd.DataFrame(rows).round(4).to_string(index=False)); print(const)


if __name__ == "__main__":
    main()
