"""Summarise Half A: distributions, never means alone.

The width-5 ReLU data in the original census was trimodal (runs at exactly
chance, a cluster near 0.96, and runs at exactly 1.0).  A mean over that
mixture describes none of its components, so every summary here carries the
distribution alongside it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Accuracy bands used for the histogram.  The two exact endpoints are separated
# out because they are the signature of the mixture: chance means the run
# learned nothing, 1.0 means it separated the evaluation set completely.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("chance (=0.50)", 0.0, 0.5),
    ("(0.50,0.90)", 0.5, 0.90),
    ("[0.90,0.99)", 0.90, 0.99),
    ("[0.99,1.00)", 0.99, 1.0),
    ("perfect (=1.00)", 1.0, 1.0),
)


def band_of(accuracy: float) -> str:
    if accuracy <= 0.5:
        return BANDS[0][0]
    if accuracy >= 1.0:
        return BANDS[-1][0]
    for name, low, high in BANDS[1:-1]:
        if low <= accuracy < high:
            return name
    return "unclassified"


def add_bands(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["band"] = result.final_eval_accuracy.map(band_of)
    return result


def per_width_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Pass rate, chance and perfect fractions, mean and SD, per cell."""

    grouped = frame.groupby(["activation", "depth", "width"])
    summary = grouped.agg(
        n=("seed", "count"),
        pass_rate=("passed", "mean"),
        at_chance=("at_chance", "mean"),
        perfect=("perfect_eval", "mean"),
        mean_accuracy=("final_eval_accuracy", "mean"),
        sd_accuracy=("final_eval_accuracy", "std"),
        min_accuracy=("final_eval_accuracy", "min"),
        max_accuracy=("final_eval_accuracy", "max"),
    ).reset_index()
    return summary


def pooled_by_width(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool depths within each activation, retaining distribution shape."""

    grouped = frame.groupby(["activation", "width"])
    return grouped.agg(
        n=("seed", "count"),
        pass_rate=("passed", "mean"),
        at_chance=("at_chance", "mean"),
        perfect=("perfect_eval", "mean"),
        mean_accuracy=("final_eval_accuracy", "mean"),
        sd_accuracy=("final_eval_accuracy", "std"),
    ).reset_index()


def histogram(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Counts per accuracy band, which is the primary reported form."""

    banded = add_bands(frame)
    counts = (
        banded.groupby(by + ["band"]).size().rename("count").reset_index()
    )
    order = [name for name, _, _ in BANDS]
    counts["band"] = pd.Categorical(counts["band"], categories=order, ordered=True)
    return counts.pivot_table(
        index=by, columns="band", values="count", fill_value=0, observed=False
    ).reset_index()


def trimodality_index(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Fraction of runs at the two exact endpoints versus in between.

    A high combined endpoint fraction with few intermediate runs indicates a
    mixture of failures and successes rather than a smooth accuracy ceiling.
    """

    grouped = frame.groupby(by)
    result = grouped.agg(
        n=("seed", "count"),
        at_chance=("at_chance", "mean"),
        perfect=("perfect_eval", "mean"),
    ).reset_index()
    result["endpoints"] = result.at_chance + result.perfect
    result["intermediate"] = 1.0 - result.endpoints
    return result


def load(results_directory: Path) -> pd.DataFrame:
    return pd.read_parquet(results_directory / "width_sweep.parquet")
