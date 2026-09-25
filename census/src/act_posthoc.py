"""Track 3A, POST HOC descriptive statistics of the registered training runs (after scoring; nothing here is a
registered verdict).  Output: results/act_general/posthoc_training.csv.

    python -m src.act_posthoc
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .act_general import NAMES, OUT


def main():
    rows = []
    for n in NAMES:
        k = json.loads((OUT / f"kappa_{n}_frozen.json").read_text())
        d = pd.concat([pd.read_csv(OUT / f"train_{n}.csv"), pd.read_csv(OUT / f"train_ext_{n}.csv")])
        c = d[d.crossed & ~d.placed_at_init]
        r = c.s_cross / k["s_glob"] - 1
        q = np.percentile(r, [10, 25, 50, 75, 90])
        chi = c.chi[np.isfinite(c.chi) & (c.chi > 0)]
        rows.append({"act": n, "runs": len(d), "crossing": len(c), "r_q10": q[0], "r_q25": q[1], "r_median": q[2],
                     "r_q75": q[3], "r_q90": q[4], "frac_abs_r_le_0.10": float((r.abs() <= 0.10).mean()),
                     "n_early_below_half_s_glob": int((c.s_cross < 0.5 * k["s_glob"]).sum()),
                     "median_step": float(c.step.median()), "chi_q25": float(np.percentile(chi, 25)),
                     "chi_q75": float(np.percentile(chi, 75)),
                     "spearman_r_chi": float(pd.Series(r[chi.index]).rank().corr(chi.rank())),
                     "noncross_median_w2_final": float(d[~d.crossed].w2_final.median())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "posthoc_training.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
