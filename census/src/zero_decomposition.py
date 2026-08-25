"""The monotonic width-3 zero: stratum decomposition from raw artifacts.

Single committed source for the T1/T2 numbers.  Every stratum is
recomputed from its artifact CSV — never from a summary document — with
the monotonicity predicate stated per stratum.  Reading-B bit-identical
duplicates (T23) are excluded by the same key used in the audit.
Output: printed table + ``results/monotonic_zero_decomposition.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .threshold_report import zero_rate_upper_bound

RESULTS = Path(__file__).resolve().parents[1] / "results"


def strata() -> list[dict]:
    rows = []

    def add(name: str, frame: pd.DataFrame, note: str) -> None:
        rows.append({"stratum": name, "runs": len(frame),
                     "separations": int(frame.perfect_eval.sum()), "note": note})

    w = pd.read_csv(RESULTS / "width_sweep.csv")
    add("width_sweep", w[(w.width == 3) & w.activation.isin(["tanh", "relu", "leaky_relu"])],
        "fixed monotonic activations")

    t = pd.read_csv(RESULTS / "threshold_sweep.csv")
    add("threshold_sweep", t[(t.width == 3) & (t.monotonic == True)],  # noqa: E712
        "family A a<=1, family B alpha>=0, fixed monotonic")

    p = pd.read_csv(RESULTS / "parametrization_sweep.csv")
    add("parametrization_sweep", p[(p.width == 3) & (p.monotonic == True)],  # noqa: E712
        "monotonic column")

    c = pd.read_csv(RESULTS / "corrugation_sweep.csv")
    cw = c[(c.width == 3) & (c.monotonic == True)].copy()  # noqa: E712
    duplicate_key = ["activation", "depth", "seed", "frequency", "noise",
                     "final_train_accuracy", "final_eval_accuracy", "eval_errors"]
    offset = cw[cw.reading == "offset"]
    duplicates = int(offset.duplicated(subset=duplicate_key, keep="first").sum())
    distinct = cw.drop(cw[cw.reading == "offset"].index).pipe(
        lambda core: pd.concat([core, offset.drop_duplicates(subset=duplicate_key, keep="first")]))
    add("corrugation_sweep (distinct)", distinct,
        f"{len(cw)} rows minus {duplicates} bit-identical Reading-B amplitude duplicates (T23)")

    pr = pd.read_csv(RESULTS / "protocol_sweep.csv")
    add("protocol_sweep", pr[(pr.width == 3) & (pr.monotonic == True)],  # noqa: E712
        "author-protocol stratum")

    r = pd.read_csv(RESULTS / "search_restarts.csv")
    add("search_restarts", r[r.width == 3],
        "200 tanh + 200 sin_family a=0.95, all monotonic by construction")

    wd = pd.read_csv(RESULTS / "winding_sweep.csv")
    add("winding_sweep", wd[(wd.width == 3) & (wd.monotonic == True)],  # noqa: E712
        "monotonic column, q=1..4 links")

    return rows


def main() -> None:
    rows = strata()
    frame = pd.DataFrame(rows)
    total_runs = int(frame.runs.sum())
    total_separations = int(frame.separations.sum())
    print(frame.to_string(index=False))
    print(f"\nTOTAL: {total_separations} separations in {total_runs} distinct runs")
    bound = zero_rate_upper_bound(total_runs)
    print(f"one-sided exact 95% upper bound on the separation rate: "
          f"{bound:.6f} = {bound * 100:.4f}%  [solves (1-p)^{total_runs} = 0.05]")
    if total_separations != 0:
        raise SystemExit("NONZERO SEPARATION COUNT — stop and investigate")
    frame.to_csv(RESULTS / "monotonic_zero_decomposition.csv", index=False)


if __name__ == "__main__":
    main()
