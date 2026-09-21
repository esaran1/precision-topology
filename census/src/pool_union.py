"""Build the pooled R union as a committed artifact.

r_pooled.csv holds 2,910 Adam+SGD rows and r_adamw.csv holds 240 AdamW rows.
The 3,150 union the abstract quotes was previously formed only at render time,
so verify_ledger could not see it (Phase 0, item 3).  This writes it down.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifact_lock import artifact_lock

RESULTS = Path(__file__).resolve().parents[1] / "results"
COLUMNS = ["a", "opt", "budget", "w2", "gstar", "R", "solved"]


def build() -> pd.DataFrame:
    pooled = pd.read_csv(RESULTS / "r_pooled.csv")
    adamw = pd.read_csv(RESULTS / "r_adamw.csv").rename(columns={"optimizer": "opt"})
    adamw["a"] = 1.25                      # the whole AdamW sweep is at a = 1.25
    frame = pd.concat([pooled[COLUMNS], adamw[COLUMNS]], ignore_index=True)
    frame["solved"] = frame.solved.astype(bool)
    return frame


def main() -> None:
    frame = build()
    low = frame[frame.R < 0.30]
    high = frame[frame.R > 0.50]
    print(f"pooled union: {len(frame)} runs, {frame.opt.nunique()} optimisers, "
          f"{frame.a.nunique()} activation values, {frame.budget.nunique()} budgets")
    print(f"  below R=0.30: {int(low.solved.sum())} of {len(low)}")
    print(f"  above R=0.50: {int(high.solved.sum())} of {len(high)}")
    stem = RESULTS / "r_pooled_union"
    with artifact_lock(stem, "pooled R union"):
        tmp = stem.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
