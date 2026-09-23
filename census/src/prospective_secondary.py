"""Block 3 secondary analyses (labelled secondary; the registered analysis in prospective.score is primary).

1. U's range prediction under its registered rule (per setting), and read as a joint prediction.
2. Window-clustered robustness: the 8 settings are 4 windows x 2 a, paired within window.  Bootstrap over
   windows (all 4^4 = 256 resamples enumerated exactly) for mean(err_C - err_X), X in {B1, B2}; and
   leave-one-window-out means.
3. C's signed log errors log(C/obs) per setting, their mean with a setting-level bootstrap interval
   (10,000 resamples, seed 0) and a window-level interval (256 enumerated resamples).

    python -m src.prospective_secondary
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _window_boot(values_by_window):
    """Exact window-level bootstrap: all ordered resamples of the windows with replacement."""
    wins = list(values_by_window)
    means = []
    for combo in itertools.product(wins, repeat=len(wins)):
        vals = np.concatenate([values_by_window[w] for w in combo])
        means.append(vals.mean())
    return np.percentile(means, [2.5, 97.5])


def main():
    m = pd.read_csv(RESULTS / "prospective_scores.csv")
    cal = pd.read_csv(RESULTS / "prospective_calibration.csv").iloc[0]
    lo, hi = cal.U_expected_log_error_min, cal.U_expected_log_error_max
    rows = []
    # 1. U's range
    m["U_obs_over_pred_log"] = np.log(m.obs_median_cross_R / m.U)
    m["U_under_prediction_pct"] = (1 - m.U / m.obs_median_cross_R) * 100
    inside = int(m.U_in_expected_range.sum())
    rows.append({"analysis": "U range, per setting (registered rule)", "result": f"{inside} of {len(m)} inside",
                 "detail": "; ".join(f"{r.window}/{r.a:.2f}: {r.U_obs_over_pred_log:.4f}"
                                     f"{'' if r.U_in_expected_range else ' OUTSIDE'}" for r in m.itertuples()),
                 "lo": lo, "hi": hi})
    rows.append({"analysis": "U range, read as a joint prediction (all settings in range)",
                 "result": "PASS" if inside == len(m) else "FAIL (narrowly)",
                 "detail": f"U under-predicts in all {len(m)} settings by "
                           f"{m.U_under_prediction_pct.min():.1f}% to {m.U_under_prediction_pct.max():.1f}%",
                 "lo": lo, "hi": hi})
    # 2. window-clustered
    for other in ("B1", "B2"):
        diff = m["abslogerr_C"] - m[f"abslogerr_{other}"]
        byw = {w: diff[m.window == w].values for w in sorted(m.window.unique())}
        wlo, whi = _window_boot(byw)
        rows.append({"analysis": f"window-clustered bootstrap: C - {other}", "result":
                     f"mean {diff.mean():.4f}, 95% [{wlo:.4f}, {whi:.4f}]",
                     "detail": "C better, interval excludes 0" if whi < 0 else "interval includes 0",
                     "lo": wlo, "hi": whi})
        for w in sorted(m.window.unique()):
            d = diff[m.window != w]
            rows.append({"analysis": f"leave-one-window-out: C - {other}, without {w}",
                         "result": f"mean {d.mean():.4f}", "detail": "C better" if d.mean() < 0 else "C not better",
                         "lo": np.nan, "hi": np.nan})
    # 3. signed errors of C
    m["C_signed_logerr"] = np.log(m.C / m.obs_median_cross_R)
    s = m.C_signed_logerr.values
    rng = np.random.default_rng(0)
    boot = rng.choice(s, (10_000, len(s))).mean(axis=1)
    slo, shi = np.percentile(boot, [2.5, 97.5])
    byw = {w: m.C_signed_logerr[m.window == w].values for w in sorted(m.window.unique())}
    wlo, whi = _window_boot(byw)
    rows.append({"analysis": "C signed log error log(C/obs)", "result":
                 f"mean {s.mean():.4f}; setting-level 95% [{slo:.4f}, {shi:.4f}]; window-level 95% [{wlo:.4f}, {whi:.4f}]",
                 "detail": "; ".join(f"{r.window}/{r.a:.2f}: {r.C_signed_logerr:+.4f}" for r in m.itertuples()),
                 "lo": slo, "hi": shi})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "prospective_secondary_analyses.csv", index=False)
    m[["window", "a", "obs_median_cross_R", "C", "C_signed_logerr", "U", "U_obs_over_pred_log",
       "U_under_prediction_pct", "U_in_expected_range"]].to_csv(RESULTS / "prospective_signed.csv", index=False)
    pd.set_option("display.max_colwidth", 200)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
