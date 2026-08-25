"""Projected linking across many projections, with cancellation diagnostics.

The single PCA-to-R^3 convention used in ``linking_projected`` reports one
number per layer.  Since a projection can cancel a nonzero linking number to
zero, one number cannot distinguish absence from cancellation.  This module
reports the *distribution* instead:

- many independent random 3-frames per representation,
- every triple drawn from the top PCA components rather than only the top 3,
- the fraction of projections returning zero, alongside the distinct values,
- a measure of how far the layer map departs from an isometry, so cancellation
  rate can be checked against distortion.

A cell that returns zero under every one of many random projections is in a
different situation from one that returns zero under most but not all, and the
difference bears directly on whether cancellation is the explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from .cancellation import random_projection_matrix
from .linking import linking_number
from .linking_trace import ARTIFACT_DISTANCE  # noqa: F401
from .models import MLP


@dataclass(frozen=True)
class ProjectionDistribution:
    """Linking values over a family of projections of one representation."""

    n_projections: int
    values: tuple[int, ...]
    fraction_zero: float
    fraction_nonzero: float
    distinct_values: tuple[int, ...]
    all_zero: bool
    min_distance: float
    projected_distances: tuple[float, ...] = ()
    reportable_values: tuple[int, ...] = ()

    @property
    def any_nonzero(self) -> bool:
        return any(value != 0 for value in self.values)


def distortion(model: MLP, upto_layer: int) -> float:
    """How far the composed linear part departs from an isometry.

    The product of the layer weight matrices is compared to the nearest scaled
    orthogonal map: a value of 0 means the composition is a similarity, which
    preserves linking exactly, and larger values mean more distortion.  Computed
    from singular values, as the normalised spread ``(s_max - s_min) / s_max``.
    """

    composed: torch.Tensor | None = None
    for index in range(upto_layer):
        weight = model.hidden_layers[index].weight.to(torch.float64)
        composed = weight if composed is None else weight @ composed
    if composed is None:
        return 0.0
    singular = torch.linalg.svdvals(composed)
    largest = float(singular.max().item())
    smallest = float(singular.min().item())
    if largest <= 0.0:
        return 1.0
    return (largest - smallest) / largest


def random_projection_distribution(
    first: torch.Tensor,
    second: torch.Tensor,
    n_projections: int = 64,
    seed_base: int = 0,
) -> ProjectionDistribution:
    """Linking under many independent random 3-frames."""

    dimension = first.shape[1]
    native_distance = float(torch.cdist(first, second).min().item())
    values: list[int] = []
    projected_distances: list[float] = []
    reportable_values: list[int] = []
    for index in range(n_projections):
        if dimension == 3:
            left, right = first, second
        else:
            basis = random_projection_matrix(dimension, seed_base + index)
            left, right = first @ basis, second @ basis
        estimate = linking_number(left, right)
        if estimate.defined:
            values.append(estimate.rounded)
            projected = float(torch.cdist(left, right).min().item())
            projected_distances.append(projected)
            if projected > ARTIFACT_DISTANCE:
                reportable_values.append(estimate.rounded)
        if dimension == 3:
            break

    array = np.array(values, dtype=int)
    return ProjectionDistribution(
        n_projections=len(values),
        values=tuple(values),
        fraction_zero=float((array == 0).mean()) if len(array) else float("nan"),
        fraction_nonzero=float((array != 0).mean()) if len(array) else float("nan"),
        distinct_values=tuple(sorted(set(values))),
        all_zero=bool(len(array) and (array == 0).all()),
        min_distance=native_distance,
        projected_distances=tuple(projected_distances),
        reportable_values=tuple(reportable_values),
    )


def pca_triple_distribution(
    first: torch.Tensor,
    second: torch.Tensor,
    n_components: int = 6,
) -> ProjectionDistribution:
    """Linking under every triple drawn from the leading PCA components.

    The standard convention takes only the top 3.  Structure carried by lower
    components is invisible to that choice, so all triples from the top
    ``n_components`` are evaluated.
    """

    combined = torch.cat([first, second], dim=0)
    centred = combined - combined.mean(dim=0, keepdim=True)
    _, _, components = torch.linalg.svd(centred, full_matrices=False)
    available = min(n_components, components.shape[0])
    native_distance = float(torch.cdist(first, second).min().item())

    values: list[int] = []
    projected_distances: list[float] = []
    reportable_values: list[int] = []
    for triple in combinations(range(available), 3):
        basis = components[list(triple)].T
        left = (first - combined.mean(dim=0, keepdim=True)) @ basis
        right = (second - combined.mean(dim=0, keepdim=True)) @ basis
        estimate = linking_number(left, right)
        if not estimate.defined:
            continue
        values.append(estimate.rounded)
        # Distance in the PROJECTED image, which is what governs whether the
        # Gauss integrand is well conditioned.  Native-space separation says
        # nothing about this: a projection can bring the curves arbitrarily
        # close together.
        projected = float(torch.cdist(left, right).min().item())
        projected_distances.append(projected)
        if projected > ARTIFACT_DISTANCE:
            reportable_values.append(estimate.rounded)

    array = np.array(values, dtype=int)
    return ProjectionDistribution(
        n_projections=len(values),
        values=tuple(values),
        fraction_zero=float((array == 0).mean()) if len(array) else float("nan"),
        fraction_nonzero=float((array != 0).mean()) if len(array) else float("nan"),
        distinct_values=tuple(sorted(set(values))),
        all_zero=bool(len(array) and (array == 0).all()),
        min_distance=native_distance,
        projected_distances=tuple(projected_distances),
        reportable_values=tuple(reportable_values),
    )


@torch.no_grad()
def propagate(model: MLP, first: torch.Tensor, second: torch.Tensor, layer: int):
    """Push both curves through the first ``layer`` hidden layers."""

    left, right = first, second
    for index in range(layer):
        hidden = model.hidden_layers[index]
        weight = hidden.weight.to(torch.float64)
        bias = hidden.bias.to(torch.float64)
        left = model._activate(torch.nn.functional.linear(left, weight, bias))
        right = model._activate(torch.nn.functional.linear(right, weight, bias))
    return left, right


def reportable(distribution: ProjectionDistribution) -> bool:
    """Whether the curves are far enough apart for any value to be meaningful."""

    return distribution.min_distance > ARTIFACT_DISTANCE
