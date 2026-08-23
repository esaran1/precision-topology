"""Analysis for the monotonicity-threshold sweep.

Everything here reads ``results/threshold_sweep.parquet`` and reports; it
never trains and never writes result artifacts, so it is safe to run while
the sweep is still writing (partial frames simply produce partial tables).

Statistics are the ones registered in ``results/threshold_prediction.md``:
exact binomial (Clopper-Pearson) intervals on separation counts, full error
distributions per parameter value, the transition interval at grid
resolution, and the confound trio (final loss, final gradient norm,
inactive-unit fraction).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ERROR_BANDS = ((0, 0), (1, 5), (6, 8), (9, 15), (16, 25), (26, 10_000))


def _binomial_cdf(successes: int, n: int, p: float) -> float:
    """P[X <= successes] for X ~ Binomial(n, p), via log-gamma."""

    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if successes < n else 1.0
    total = 0.0
    log_p, log_q = math.log(p), math.log(1.0 - p)
    for k in range(successes + 1):
        log_term = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_p
            + (n - k) * log_q
        )
        total += math.exp(log_term)
    return min(total, 1.0)


def _bisect(function, low: float, high: float, iterations: int = 200) -> float:
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if function(middle):
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    alpha = 1.0 - confidence
    if successes == 0:
        lower = 0.0
    else:
        # Smallest p whose upper tail P[X >= successes] exceeds alpha/2.
        lower = _bisect(
            lambda p: 1.0 - _binomial_cdf(successes - 1, n, p) > alpha / 2.0,
            0.0,
            1.0,
        )
    if successes == n:
        upper = 1.0
    else:
        # Largest p whose lower tail P[X <= successes] exceeds alpha/2 —
        # found as the smallest p where the tail drops below it.
        upper = _bisect(
            lambda p: _binomial_cdf(successes, n, p) < alpha / 2.0,
            0.0,
            1.0,
        )
    return lower, upper


def zero_rate_upper_bound(n: int, confidence: float = 0.95) -> float:
    """One-sided exact upper bound on the rate when zero successes observed.

    Solves (1 - p)^n = 1 - confidence, the exact form of the rule of three.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def condition_table(frame: pd.DataFrame, width: int) -> pd.DataFrame:
    """Per-(family, parameter) summary at one width, all depths and seeds pooled."""

    rows = []
    subset = frame[frame.width == width]
    for (family, activation, parameter), group in subset.groupby(
        ["family", "activation", "parameter"], dropna=False, sort=False
    ):
        n = len(group)
        separations = int(group.perfect_eval.sum())
        lower, upper = clopper_pearson(separations, n)
        errors = group.eval_errors
        bands = {
            f"band_{low}_{high}": int(((errors >= low) & (errors <= high)).sum())
            for low, high in ERROR_BANDS
        }
        rows.append(
            {
                "family": family,
                "activation": activation,
                "parameter": parameter,
                "monotonic": bool(group.monotonic.iloc[0]),
                "n": n,
                "separations": separations,
                "rate": separations / n,
                "ci_low": lower,
                "ci_high": upper,
                "errors_min": int(errors.min()),
                "errors_p25": float(errors.quantile(0.25)),
                "errors_median": float(errors.median()),
                "errors_p75": float(errors.quantile(0.75)),
                "errors_max": int(errors.max()),
                "final_loss_median": float(group.final_train_loss.median()),
                "gradient_norm_median": float(group.final_gradient_norm.median()),
                "inactive_fraction_mean": float(group.inactive_unit_fraction.mean()),
                **bands,
            }
        )
    return pd.DataFrame(rows)


def transition_interval(
    table: pd.DataFrame, family: str
) -> tuple[float | None, float | None]:
    """(last zero-separation value, first positive value) along the parameter
    axis ordered from the monotonic side toward the non-monotonic side.

    Family A is ordered ascending (monotonic a <= 1), Family B descending
    (monotonic alpha >= 0).  Returns grid values, never interpolates.
    """

    subset = table[table.family == family].copy()
    ascending = family == "A"
    subset = subset.sort_values("parameter", ascending=ascending)
    last_zero: float | None = None
    for _, row in subset.iterrows():
        if row.separations == 0:
            last_zero = row.parameter
        else:
            return last_zero, float(row.parameter)
    return last_zero, None


def monotonic_side_zero(table: pd.DataFrame) -> tuple[int, int]:
    """(total separations, total runs) across all monotonic-side conditions."""

    monotonic = table[table.monotonic]
    return int(monotonic.separations.sum()), int(monotonic.n.sum())


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    frame = pd.read_parquet(directory / "threshold_sweep.parquet")
    print(f"{len(frame)} runs loaded")
    for width in sorted(frame.width.unique()):
        table = condition_table(frame, width)
        print(f"\n=== width {width} ===")
        print(
            table[
                [
                    "family",
                    "parameter",
                    "monotonic",
                    "n",
                    "separations",
                    "ci_high",
                    "errors_min",
                    "errors_median",
                ]
            ].to_string(index=False)
        )
        for family in ("A", "B"):
            print(f"family {family} transition: {transition_interval(table, family)}")
        zero = monotonic_side_zero(table)
        print(f"monotonic side: {zero[0]} separations in {zero[1]} runs")


if __name__ == "__main__":
    main()
