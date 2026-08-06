"""Does the projection convention recover linking from a known-linked pair?

The projected linking measurement above width 3 returns zero everywhere.  That
is either a real null -- the obstruction is a width-3 phenomenon and nothing is
linked in the wider layers -- or the projection is blind and would return zero
whatever it was given.

This module separates the two.  A configuration with known linking number is
embedded in R^k for the same widths the sweep used, rotated by a random
orthogonal map so the link does not lie in the first three coordinates, and
then passed through the identical joint-PCA-to-R^3 convention.  If the
estimator recovers the known value, the convention can see linking at that
dimension and the sweep's uniform zero is a real null.  If it does not, the
measure is blind above d=3 and the sweep result is uninformative.

The embedding is isometric onto its image, so the *intrinsic* linking of the
pair is unchanged by construction; only the ambient dimension and the basis
differ.  Note that two circles in R^4 or higher are always ambient-isotopic to
an unlink, so this is deliberately a test of the estimator and the projection,
not a claim that the embedded pair is "linked in R^k".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import linked_core_circles
from .linking import linking_number
from .linking_trace import _project_to_r3


@dataclass(frozen=True)
class FidelityResult:
    """One embed-rotate-project-estimate trial."""

    dimension: int
    seed: int
    expected: int
    recovered: int | None
    raw: float | None
    residual: float | None
    min_distance: float
    defined: bool

    @property
    def correct(self) -> bool:
        return self.recovered is not None and abs(self.recovered) == abs(self.expected)


def random_rotation(dimension: int, seed: int) -> torch.Tensor:
    """A Haar-distributed orthogonal matrix via QR of a Gaussian."""

    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(
        dimension, dimension, dtype=torch.float64, generator=generator
    )
    orthogonal, upper = torch.linalg.qr(gaussian)
    # Fix the sign convention so the result is Haar-distributed rather than
    # biased by QR's arbitrary choice of signs.
    return orthogonal * torch.sign(torch.diagonal(upper)).unsqueeze(0)


def embed_and_rotate(
    curve: torch.Tensor, dimension: int, rotation: torch.Tensor
) -> torch.Tensor:
    """Pad an R^3 curve with zeros out to ``dimension`` and rotate it."""

    if dimension < 3:
        raise ValueError("dimension must be at least 3")
    padding = torch.zeros(curve.shape[0], dimension - 3, dtype=torch.float64)
    return torch.cat([curve, padding], dim=1) @ rotation


def fidelity_trial(
    dimension: int,
    seed: int,
    n_points: int = 512,
    expected: int = -1,
) -> FidelityResult:
    """Embed the census Hopf link in R^dimension, rotate, project, estimate."""

    first_array, second_array = linked_core_circles(n_points)
    first = torch.as_tensor(first_array, dtype=torch.float64)
    second = torch.as_tensor(second_array, dtype=torch.float64)

    rotation = random_rotation(dimension, seed)
    embedded_first = embed_and_rotate(first, dimension, rotation)
    embedded_second = embed_and_rotate(second, dimension, rotation)

    # Distance is preserved by the rotation, so this equals the R^3 value; it is
    # recomputed rather than assumed so a broken embedding would show up.
    native_distance = float(torch.cdist(embedded_first, embedded_second).min().item())

    left, right = _project_to_r3(embedded_first, embedded_second)
    estimate = linking_number(left, right)
    return FidelityResult(
        dimension=dimension,
        seed=seed,
        expected=expected,
        recovered=estimate.rounded if estimate.defined else None,
        raw=estimate.raw if estimate.defined else None,
        residual=estimate.residual if estimate.defined else None,
        min_distance=native_distance,
        defined=estimate.defined,
    )


def run_fidelity(
    dimensions: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 10, 12, 15),
    seeds: tuple[int, ...] = tuple(range(10)),
    n_points: int = 512,
) -> list[FidelityResult]:
    return [
        fidelity_trial(dimension, seed, n_points)
        for dimension in dimensions
        for seed in seeds
    ]
