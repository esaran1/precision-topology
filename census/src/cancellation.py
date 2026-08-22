"""Can projection cancel a nonzero linking number to zero?

Our projected-linking null rests on a control that cannot see the failure mode
the author raises.  A Hopf link has |lk| = 1, so there is nothing to cancel: any
projection either recovers 1 or is degenerate.  The concern is different --
projecting to R^3 creates self-intersections, the image can be several joined
rings, and positive and negative crossings can cancel so the estimator reports 0
while nontrivial linking is present.

To test that, we need a configuration whose linking number is nonzero but whose
*signed crossings are separable into cancelling groups*.  Two circles with
higher winding are the natural candidate: a (p, q) torus link carries linking
number that grows with winding, and its crossings are distributed around the
torus so that a projection can, in principle, cancel them.

This module builds such configurations, verifies their true linking in R^3 with
the validated estimator, and measures what many projections return.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .linking import linking_number
from .linking_trace import _project_to_r3


@dataclass(frozen=True)
class ProjectionOutcome:
    """What a family of projections returned for one configuration."""

    name: str
    true_linking: int
    n_projections: int
    values: tuple[int, ...]
    fraction_zero: float
    fraction_correct: float
    distinct_values: tuple[int, ...]
    min_distance: float

    @property
    def cancels(self) -> bool:
        """Whether any projection returned 0 while true linking is nonzero."""

        return self.true_linking != 0 and 0 in self.values


def torus_link(p: int, q: int, n_points: int = 4096, major: float = 2.0, minor: float = 0.8):
    """Two curves on a torus forming a (p, q)-style link with winding ``q``.

    Each component winds ``q`` times around the tube while going ``p`` times
    around the core.  For two components offset by half a period, the linking
    number scales with the winding.
    """

    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)

    def component(offset: float) -> np.ndarray:
        angle = q * t + offset
        radius = major + minor * np.cos(angle)
        return np.column_stack(
            [radius * np.cos(p * t), radius * np.sin(p * t), minor * np.sin(angle)]
        )

    return component(0.0), component(math.pi)


def coiled_link(turns: int, n_points: int = 4096, radius: float = 1.0, coil: float = 0.25):
    """Two interlocked coils: a Hopf link whose components wind ``turns`` times.

    Higher winding multiplies the number of signed crossings, which is what
    gives a projection something to cancel.
    """

    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)
    first = np.column_stack(
        [
            radius * np.cos(t) + coil * np.cos(turns * t),
            radius * np.sin(t) + coil * np.sin(turns * t),
            coil * np.sin(turns * t),
        ]
    )
    second = np.column_stack(
        [
            radius + radius * np.cos(t) + coil * np.cos(turns * t),
            coil * np.sin(turns * t),
            radius * np.sin(t) + coil * np.cos(turns * t),
        ]
    )
    return first, second


def random_projection_matrix(dimension: int, seed: int) -> torch.Tensor:
    """A random orthonormal 3-frame in ``dimension`` dimensions."""

    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(dimension, 3, dtype=torch.float64, generator=generator)
    basis, _ = torch.linalg.qr(gaussian)
    return basis


def project_random(
    first: torch.Tensor, second: torch.Tensor, seed: int
) -> torch.Tensor | None:
    """Project both curves through a shared random 3-frame."""

    basis = random_projection_matrix(first.shape[1], seed)
    return first @ basis, second @ basis


def evaluate_projections(
    first: np.ndarray,
    second: np.ndarray,
    name: str,
    n_projections: int = 200,
    seed_base: int = 0,
    true_linking: int | None = None,
) -> ProjectionOutcome:
    """Measure true linking, then linking under many random projections."""

    left = torch.tensor(first, dtype=torch.float64)
    right = torch.tensor(second, dtype=torch.float64)
    if left.shape[1] == 3:
        truth = linking_number(left, right)
        if not truth.defined:
            raise ValueError(f"{name}: components intersect; true linking undefined")
        true_value = truth.rounded
        true_distance = truth.min_distance
    else:
        # Above R^3 the classical linking number is not defined for two curves.
        # The embedding used here is an isometry onto its image, so the value
        # carried over from R^3 must be supplied by the caller.
        if true_linking is None:
            raise ValueError(
                f"{name}: true linking must be supplied for dimension {left.shape[1]}"
            )
        true_value = true_linking
        true_distance = float(torch.cdist(left, right).min().item())

    values: list[int] = []
    for index in range(n_projections):
        projected_left, projected_right = project_random(left, right, seed_base + index)
        estimate = linking_number(projected_left, projected_right)
        if estimate.defined:
            values.append(estimate.rounded)

    array = np.array(values, dtype=int)
    return ProjectionOutcome(
        name=name,
        true_linking=true_value,
        n_projections=len(values),
        values=tuple(values),
        fraction_zero=float((array == 0).mean()) if len(array) else float("nan"),
        fraction_correct=float((np.abs(array) == abs(true_value)).mean())
        if len(array)
        else float("nan"),
        distinct_values=tuple(sorted(set(values))),
        min_distance=true_distance,
    )


def embed(curve: np.ndarray, dimension: int, seed: int) -> np.ndarray:
    """Pad an R^3 curve to ``dimension`` and rotate it into general position."""

    if dimension == 3:
        return curve
    padded = np.concatenate(
        [curve, np.zeros((curve.shape[0], dimension - 3), dtype=np.float64)], axis=1
    )
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(dimension, dimension, dtype=torch.float64, generator=generator)
    rotation, upper = torch.linalg.qr(gaussian)
    rotation = rotation * torch.sign(torch.diagonal(upper)).unsqueeze(0)
    return (torch.tensor(padded, dtype=torch.float64) @ rotation).numpy()
