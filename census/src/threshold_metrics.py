"""Criterion-robustness metrics for the perfect-separation threshold.

Exact separation on a finite sample is not a property of the learned map.  It
is the event that no sampled point falls on the wrong side, which depends on
how many points were drawn and where they sit relative to the decision surface.
Thin tubes concentrate points near that surface, so a configuration can be
*easier* on average while producing *fewer* exactly-perfect runs.

These helpers therefore report the accuracy distribution, the error count, and
the activation gap under more than one criterion, so a reader can see whether a
conclusion depends on the choice of threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .models import MLP
from .parametrization import TorusLink, core_curves


# Two criteria for "separates".  The strict one is the event used so far; the
# tolerant one allows a couple of stray points out of a 2,000-point sample.
PERFECT = 1.0
NEAR_PERFECT = 0.999


def error_count(accuracy: float, n_points: int) -> int:
    """Misclassified points implied by an accuracy on a known sample size."""

    return int(round((1.0 - accuracy) * n_points))


def add_error_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach explicit error counts; one wrong point and forty are different."""

    result = frame.copy()
    n_points = result.n_per_class * 2
    result["eval_points"] = n_points
    result["eval_errors"] = [
        error_count(accuracy, points)
        for accuracy, points in zip(result.final_eval_accuracy, n_points)
    ]
    result["near_perfect"] = result.final_eval_accuracy >= NEAR_PERFECT
    result["above_99"] = result.final_eval_accuracy >= 0.99
    return result


def distribution_summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median and threshold fractions side by side, never a mean alone."""

    prepared = add_error_counts(frame)
    grouped = prepared.groupby(by)
    summary = grouped.agg(
        n=("seed", "count"),
        median_accuracy=("final_eval_accuracy", "median"),
        max_accuracy=("final_eval_accuracy", "max"),
        sd_accuracy=("final_eval_accuracy", "std"),
        fraction_perfect=("perfect_eval", "mean"),
        fraction_near_perfect=("near_perfect", "mean"),
        fraction_above_99=("above_99", "mean"),
        median_errors=("eval_errors", "median"),
        min_errors=("eval_errors", "min"),
    ).reset_index()
    return summary


def criterion_gap(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Non-monotonic minus monotonic, under each criterion.

    The claim under test is that the non-monotonic activation separates where
    the monotonic ones do not.  Reporting the gap under both a strict and a
    tolerant criterion shows whether that claim depends on the threshold.
    """

    prepared = add_error_counts(frame)
    rows: list[dict[str, object]] = []
    for key, group in prepared.groupby(by):
        gelu = group[group.activation == "gelu"]
        monotonic = group[group.activation != "gelu"]
        if gelu.empty or monotonic.empty:
            continue
        keys = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(by, keys)),
                "n_gelu": len(gelu),
                "n_monotonic": len(monotonic),
                "gelu_perfect": float(gelu.perfect_eval.mean()),
                "monotonic_perfect": float(monotonic.perfect_eval.mean()),
                "gap_perfect": float(gelu.perfect_eval.mean() - monotonic.perfect_eval.mean()),
                "gelu_near_perfect": float(gelu.near_perfect.mean()),
                "monotonic_near_perfect": float(monotonic.near_perfect.mean()),
                "gap_near_perfect": float(
                    gelu.near_perfect.mean() - monotonic.near_perfect.mean()
                ),
                "gelu_median": float(gelu.final_eval_accuracy.median()),
                "monotonic_median": float(monotonic.final_eval_accuracy.median()),
                "gap_median": float(
                    gelu.final_eval_accuracy.median() - monotonic.final_eval_accuracy.median()
                ),
                "gelu_max": float(gelu.final_eval_accuracy.max()),
                "monotonic_max": float(monotonic.final_eval_accuracy.max()),
                "monotonic_min_errors": int(
                    min(error_count(a, n) for a, n in zip(monotonic.final_eval_accuracy, monotonic.n_per_class * 2))
                ),
            }
        )
    return pd.DataFrame(rows)


def ordering_disagreement(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Where the activation ranking differs between criteria.

    If "fraction perfect" and "median accuracy" order the activations
    differently, the threshold is measuring something the median is not.
    """

    prepared = add_error_counts(frame)
    rows: list[dict[str, object]] = []
    for key, group in prepared.groupby(by):
        by_perfect = (
            group.groupby("activation").perfect_eval.mean().sort_values(ascending=False)
        )
        by_median = (
            group.groupby("activation").final_eval_accuracy.median().sort_values(ascending=False)
        )
        keys = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(by, keys)),
                "order_by_perfect": " > ".join(by_perfect.index),
                "order_by_median": " > ".join(by_median.index),
                "orders_agree": list(by_perfect.index) == list(by_median.index),
                "top_by_perfect": by_perfect.index[0],
                "top_by_median": by_median.index[0],
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BoundaryProximity:
    """Where a run's misclassified points sit within their tube."""

    errors: int
    median_relative_radius: float | None
    fraction_in_outer_decile: float | None


@torch.no_grad()
def error_boundary_proximity(
    model: MLP,
    link: TorusLink,
    features: np.ndarray,
    labels: np.ndarray,
    n_core_points: int = 2048,
) -> BoundaryProximity:
    """Relative radius of misclassified points within their own tube.

    A value near 1 means the point sits at the tube surface.  If near-miss runs
    fail only on surface points, the exact-separation threshold is tracking
    where the sample happened to land rather than whether the map separates.
    """

    points = torch.as_tensor(features, dtype=torch.float32)
    truth = torch.as_tensor(labels, dtype=torch.int64)
    predicted = model(points).argmax(dim=1)
    wrong = predicted != truth
    count = int(wrong.sum().item())
    if count == 0:
        return BoundaryProximity(0, None, None)

    first_core, second_core = core_curves(link, n_core_points)
    cores = {
        0: (torch.tensor(first_core, dtype=torch.float64), link.tube_radius_a),
        1: (torch.tensor(second_core, dtype=torch.float64), link.tube_radius_b),
    }
    relative: list[float] = []
    wrong_points = points[wrong].to(torch.float64)
    wrong_labels = truth[wrong]
    for label in (0, 1):
        selected = wrong_points[wrong_labels == label]
        if selected.shape[0] == 0:
            continue
        core, tube = cores[label]
        distance = torch.cdist(selected, core).min(dim=1).values
        relative.extend((distance / tube).tolist())

    values = torch.tensor(relative, dtype=torch.float64)
    return BoundaryProximity(
        errors=count,
        median_relative_radius=float(values.median().item()),
        fraction_in_outer_decile=float((values >= 0.9).to(torch.float64).mean().item()),
    )
