"""Metric-artifact check (Fig 4): 10–90% crossings of the continuous and binary metrics in R.

The original computation (metric_artifact_results.md, 2026-09-12) has no committed
producer.  This reimplements it from the stated definition:

    W = range of R over which a metric moves from 10% to 90% of its total observed
    swing, each metric normalized by its own range

on 0.025-wide R bins with at least MIN_N runs (the bins of Fig 4), by linear
interpolation between bin midpoints, first crossing of each level.  The share of
the error improvement at zero solve rate is (first-bin errors − errors in the last
bin with solve rate 0) / first-bin errors.

`ghat` = "restricted" reproduces the original numbers (validation);
`ghat` = "certified" is the paper's R.

    python -m src.metric_check
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "metric_check.csv"
EDGES = np.arange(0.0, 0.55, 0.025)
MIN_N = 8


def runs(ghat: str) -> pd.DataFrame:
    s = pd.concat([pd.read_csv(RESULTS / "fold1d_sweep.csv"),
                   pd.read_csv(RESULTS / "fold1d_refine.csv")])
    s = s[(s.activation == "sin_family") & (s.parameter > 1.0)].copy()
    g = pd.read_csv(RESULTS / "ghat_certified_all.csv").set_index(pd.read_csv(
        RESULTS / "ghat_certified_all.csv").a.round(2))
    col = "Ghat_certified" if ghat == "certified" else "Ghat_restricted"
    s["R"] = s.w2_abs * s.parameter.round(2).map(g[col]) / 2
    s["solved"] = s.solved.astype(str).str.lower() == "true"
    return s


def binned(s: pd.DataFrame) -> pd.DataFrame:
    s = s.assign(bin=pd.cut(s.R, EDGES))
    g = s.groupby("bin", observed=True).agg(
        n=("solved", "size"), rate=("solved", "mean"), err=("eval_errors", "mean"),
        q25=("eval_errors", lambda v: v.quantile(.25)),
        q75=("eval_errors", lambda v: v.quantile(.75))).reset_index()
    g["mid"] = [b.mid for b in g["bin"]]
    return g[g.n >= MIN_N].reset_index(drop=True)


def crossing(x, y, level):
    """First R where the normalized curve y reaches `level` (linear interpolation)."""
    for i in range(1, len(y)):
        if (y[i - 1] - level) * (y[i] - level) <= 0 and y[i] != y[i - 1]:
            return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return float("nan")


def analyse(ghat: str) -> dict:
    s = runs(ghat)
    g = binned(s)
    x = g.mid.values.astype(float)
    rate = (g.rate - g.rate.min()) / (g.rate.max() - g.rate.min())
    err = (g.err.max() - g.err) / (g.err.max() - g.err.min())           # improvement, 0 -> 1
    b10, b90 = crossing(x, rate.values, 0.1), crossing(x, rate.values, 0.9)
    c10, c90 = crossing(x, err.values, 0.1), crossing(x, err.values, 0.9)
    s05 = s.assign(bin=pd.cut(s.R, np.arange(0.0, 0.3001, 0.05)))
    t = s05.groupby("bin", observed=True).agg(n=("solved", "size"), rate=("solved", "mean"),
                                               err=("eval_errors", "mean")).reset_index()
    first = float(t.err.iloc[0])
    zero = t[t.rate == 0]
    last_zero = float(zero.err.iloc[-1])
    n_zero = int(s[s.R <= float(zero.bin.iloc[-1].right)].shape[0])
    return {"ghat": ghat, "runs": len(s), "binary_10": b10, "binary_90": b90, "W_binary": b90 - b10,
            "continuous_10": c10, "continuous_90": c90, "W_continuous": c90 - c10,
            "ratio": (c90 - c10) / (b90 - b10),
            "err_first_bin": first, "err_last_zero_rate_bin": last_zero,
            "improvement_at_zero_rate_pct": (first - last_zero) / first * 100,
            "runs_in_zero_rate_bins": n_zero,
            "last_zero_rate_bin_right": float(zero.bin.iloc[-1].right)}


def sensitivity(ghat: str = "certified") -> pd.DataFrame:
    """Binary and continuous crossings under other bin widths (the original binning is
    not recorded; its binary 10% crossing 0.332 is reproduced with 0.02 bins)."""
    s = runs(ghat)
    rows = []
    for w in (0.02, 0.025, 0.03, 0.04, 0.05):
        e = np.arange(0.0, 0.62, w)
        g = s.assign(bin=pd.cut(s.R, e)).groupby("bin", observed=True).agg(
            n=("solved", "size"), rate=("solved", "mean"), err=("eval_errors", "mean")).reset_index()
        g = g[g.n >= MIN_N]
        x = np.array([b.mid for b in g.bin])
        r = ((g.rate - g.rate.min()) / (g.rate.max() - g.rate.min())).values
        c = ((g.err.max() - g.err) / (g.err.max() - g.err.min())).values
        rows.append({"ghat": ghat, "bin_width": w, "binary_10": crossing(x, r, 0.1),
                     "binary_90": crossing(x, r, 0.9), "continuous_10": crossing(x, c, 0.1),
                     "continuous_90": crossing(x, c, 0.9)})
    t = pd.DataFrame(rows)
    t["ratio"] = (t.continuous_90 - t.continuous_10) / (t.binary_90 - t.binary_10)
    return t


def main():
    d = pd.DataFrame([analyse("restricted"), analyse("certified")])
    d.to_csv(OUT, index=False)
    print(d.T.to_string())
    t = pd.concat([sensitivity("restricted"), sensitivity("certified")])
    t.to_csv(RESULTS / "metric_check_binning.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
