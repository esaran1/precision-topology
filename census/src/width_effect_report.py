"""Task E 1b/1c analysis of width_effect.csv against P-W1..P-W4.

Pure stdlib statistics (no scipy in this environment): two-sided Fisher
exact via the hypergeometric distribution, Clopper-Pearson from
threshold_report.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import pandas as pd

from .threshold_report import clopper_pearson, zero_rate_upper_bound


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for table [[a, b], [c, d]].

    Sums hypergeometric probabilities of all tables with the observed
    margins whose probability does not exceed the observed table's.
    """

    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    denominator = comb(n, col1)

    def probability(k: int) -> float:
        return comb(row1, k) * comb(row2, col1 - k) / denominator

    observed = probability(a)
    lo, hi = max(0, col1 - row2), min(col1, row1)
    return min(1.0, sum(p for k in range(lo, hi + 1)
                        if (p := probability(k)) <= observed * (1 + 1e-9)))


def cell_label(row_activation: str, parameter) -> str:
    if row_activation == "sin_family":
        return f"sin({parameter})"
    return row_activation


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    data = pd.read_csv(directory / "width_effect.csv")

    for depth in (3, 6):
        d = data[data.depth == depth]
        print(f"=== depth {depth}: dense-verified solves per cell ===")
        table = d.pivot_table(index="width", columns="activation",
                              values="solved", aggfunc=["sum", "count"])
        print(table.to_string())

        print(f"\n=== depth {depth}: P-W1 Fisher tests, non-monotonic vs monotonic ===")
        for width in sorted(d.width.unique()):
            w = d[d.width == width]
            n_cell = len(w[w.activation == "gelu"])
            for non in ("gelu", "sin_family"):
                for mono in ("tanh", "leaky_relu", "relu"):
                    a = int(w[w.activation == non].solved.sum())
                    c = int(w[w.activation == mono].solved.sum())
                    p = fisher_two_sided(a, n_cell - a, c, n_cell - c)
                    marker = " *" if p < 0.05 else ""
                    print(f"width {width:>2} {cell_label(non, 1.5):>9} {a:>3}/{n_cell}"
                          f" vs {mono:>10} {c:>3}/{n_cell}  p={p:.4g}{marker}")

        print(f"\n=== depth {depth}: convergence speed (steps to criterion, solvers only) ===")
        solved = d[d.solved]
        speed = solved[solved.steps_to_criterion > 0].groupby(
            ["width", "activation"]).steps_to_criterion.agg(["median", "mean", "count"])
        print(speed.to_string())
        censored = d[(d.steps_to_criterion < 0)].groupby(
            ["width", "activation"]).size()
        print("\ncensored (never hit train accuracy 1.0):")
        print(censored.to_string() if len(censored) else "none")

    print("\n=== width-3 zero bounds and CP intervals (depth 3) ===")
    w3 = data[(data.depth == 3) & (data.width == 3)]
    for activation in ("gelu", "sin_family", "tanh", "leaky_relu", "relu"):
        cell = w3[w3.activation == activation]
        k, n = int(cell.solved.sum()), len(cell)
        if k == 0:
            print(f"{activation}: 0/{n}, zero-rate upper bound "
                  f"{zero_rate_upper_bound(n):.4f}")
        else:
            lo, hi = clopper_pearson(k, n)
            print(f"{activation}: {k}/{n}, CP95 [{lo:.4f}, {hi:.4f}]")

    print("\n=== minimum eval errors at width 3-4 (depth 3), per activation ===")
    for width in (3, 4):
        w = data[(data.depth == 3) & (data.width == width)]
        for activation in ("gelu", "sin_family", "tanh", "leaky_relu", "relu"):
            cell = w[w.activation == activation]
            errors = cell.eval_errors.sort_values().tolist()
            print(f"width {width} {activation:>10}: min={errors[0]} "
                  f"quartiles={errors[len(errors)//4]},{errors[len(errors)//2]},"
                  f"{errors[3*len(errors)//4]} zeros={sum(e == 0 for e in errors)}"
                  f" dense_fail={int(((cell.eval_errors == 0) & (cell.dense_errors > 0)).sum())}")

    print("\n=== dying-ReLU flag: chance collapses and dead units (depth 3) ===")
    relu = data[(data.depth == 3) & (data.activation == "relu")]
    print(relu.groupby("width").agg(
        at_chance=("at_chance", "sum"),
        dead_fraction_mean=("inactive_unit_fraction", "mean")).to_string())


if __name__ == "__main__":
    main()
