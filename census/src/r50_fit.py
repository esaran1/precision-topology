"""The R50 logistic fit, reimplemented and committed.

The original script that produced the committed `R50 = 0.3705` is NOT in the
repository and is not recoverable: `git log -S "R50" --all` and `-S "r50" --all`
return only .md and .csv blobs; an exhaustive scan of every blob ever committed
finds no script; no untracked .py or .ipynb on disk contains it; git fsck's
dangling objects are checkpoint CSVs.  See results/r50_provenance.md.

This module is the documented replacement.  Definition, stated exactly:

  POPULATION  the pooled union, r_pooled.csv + r_adamw.csv = 3,150 runs
              (12 values of a, 8 budgets, three optimisers).  R is recomputed
              with the CERTIFIED Ghat (T65), not the restricted search.

  BINNING     41 quantile edges over R, giving <=40 bins; a bin is kept only if
              it holds >= 10 runs.  Each kept bin contributes (mean R, solve rate).

  MODEL       two-parameter logistic  P(solve) = 1 / (1 + exp(-k (R - mid)))

  FIT         least squares on the binned rates, by grid search over
              (k, mid) at 900 x 1600 resolution.  The result is sensitive to this
              resolution -- coarser grids give values up to 0.005 higher -- so the
              grid is fixed here and the convergence is documented.

  CROSSING    R50 := mid, the logistic's own midpoint, where P = 0.5 exactly.
              R25 and R75 are the analytic quantiles mid + log(q/(1-q))/k.

  INTERVAL    cluster bootstrap over runs, 400 resamples, percentile 95%.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_BIN_EDGES = 41
MIN_PER_BIN = 10
K_GRID, MID_GRID = 900, 1600
N_BOOT = 400


def pooled(certified: bool = True) -> pd.DataFrame:
    gg = pd.read_csv(RESULTS / "ghat_certified_all.csv").set_index("a")
    col = "Ghat_certified" if certified else "Ghat_restricted"
    frames = []
    for fn in ("r_pooled.csv", "r_adamw.csv"):
        t = pd.read_csv(RESULTS / fn)
        if "a" not in t:
            t = t.assign(a=1.25)
        if certified:                               # the rigorous Ĝ_cert (Block 2, author's decision 2026-09-24)
            from .ghat_rigorous import ghat_R_of
            g = t.a.map(ghat_R_of)
        else:
            g = t.a.map(lambda v: float(gg.loc[round(float(v), 2), col]))
        frames.append(pd.DataFrame({"R": t.w2.abs() * g / 2,
                                    "solved": t.solved.astype(bool)}))
    return pd.concat(frames, ignore_index=True)


def fit(d: pd.DataFrame) -> dict:
    x, y = d.R.values, d.solved.values.astype(float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BIN_EDGES)))
    cx, cy = [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() >= MIN_PER_BIN:
            cx.append(x[m].mean())
            cy.append(y[m].mean())
    cx, cy = np.array(cx), np.array(cy)
    best = (np.inf, None)
    for k in np.linspace(5, 300, K_GRID):
        pr = 1.0 / (1.0 + np.exp(-np.clip(k * (cx[None, :] -
                    np.linspace(cx.min(), cx.max(), MID_GRID)[:, None]), -50, 50)))
        r = np.sqrt(((pr - cy[None, :]) ** 2).mean(axis=1))
        j = int(r.argmin())
        if r[j] < best[0]:
            best = (float(r[j]), (float(k),
                    float(np.linspace(cx.min(), cx.max(), MID_GRID)[j])))
    rms, (k, mid) = best
    q = lambda t: mid + np.log(t / (1 - t)) / k
    return {"R50": mid, "R25": q(0.25), "R75": q(0.75), "k": k, "rms": rms,
            "n_bins": len(cx)}


def main() -> None:
    d = pooled(certified=True)
    f = fit(d)
    print(f"certified Ghat, n = {len(d)}, bins = {f['n_bins']}")
    print(f"  R50 = {f['R50']:.4f}   R25 = {f['R25']:.4f}   R75 = {f['R75']:.4f}")
    print(f"  k = {f['k']:.2f}   rms = {f['rms']:.4f}")
    W = f["R75"] - f["R25"]
    print(f"  W = R75 - R25 = {W:.4f}   materiality 0.5W = {0.5 * W:.4f}")
    rng = np.random.default_rng(0)
    boots = []
    for i in range(N_BOOT):
        s = d.iloc[rng.choice(len(d), len(d), replace=True)]
        try:
            boots.append(fit(s)["R50"])
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  95% CI [{lo:.4f}, {hi:.4f}]  ({len(boots)} resamples)")
    r = fit(pooled(certified=False))
    print(f"\nrestricted Ghat (for the shift only): R50 = {r['R50']:.4f}")
    print(f"  shift from the Ghat switch: {f['R50'] - r['R50']:+.4f}")
    pd.DataFrame([{**f, "ci_lo": lo, "ci_hi": hi, "W": W,
                   "materiality": 0.5 * W,
                   "R50_restricted": r["R50"],
                   "shift": f["R50"] - r["R50"]}]).to_csv(
        RESULTS / "r50_fit.csv", index=False)
    print("\nwritten results/r50_fit.csv")


if __name__ == "__main__":
    main()
