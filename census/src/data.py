"""Deterministic linked-solid-tori and Gaussian-blob datasets in R^3.

The paper does not specify enough geometry to reproduce D-II point-for-point.
We therefore use an explicit Hopf-link construction.  For major radius R, the
core curves are

    C0(t) = (R cos(t), R sin(t), 0)
    C1(s) = (R + R cos(s), 0, R sin(s)).

C1 crosses the spanning disk of C0 at x=0 (inside) and x=2R (outside), so the
oriented core circles have linking number +/-1.  Points are sampled uniformly
by volume from a disk normal to each core curve, producing solid tori.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Dataset:
    """A balanced binary dataset with float32 coordinates and int64 labels."""

    features: NDArray[np.float32]
    labels: NDArray[np.int64]


def _validate_common(n_per_class: int, seed: int) -> None:
    if not isinstance(n_per_class, int) or n_per_class <= 0:
        raise ValueError("n_per_class must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def linked_core_circles(
    n_points: int = 512,
    major_radius: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Return equally spaced samples from the two oriented Hopf-link cores."""

    if not isinstance(n_points, int) or n_points < 3:
        raise ValueError("n_points must be an integer of at least 3")
    if major_radius <= 0.0:
        raise ValueError("major_radius must be positive")

    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    zeros = np.zeros_like(angles)
    first = major_radius * np.column_stack((cosine, sine, zeros))
    second = major_radius * np.column_stack((1.0 + cosine, zeros, sine))
    return first, second


def gauss_linking_integral(first: FloatArray, second: FloatArray) -> float:
    """Approximate the Gauss linking integral for two closed polygonal curves.

    Each polygon segment contributes at its midpoint with its full chord vector.
    Convergence is rapid for the smooth, disjoint circles used here.
    """

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != (3,) or second.shape[1:] != (3,):
        raise ValueError("curves must have shape (n_points, 3)")
    if first.shape[0] < 3 or second.shape[0] < 3:
        raise ValueError("each closed curve needs at least three points")

    first_next = np.roll(first, -1, axis=0)
    second_next = np.roll(second, -1, axis=0)
    first_mid = 0.5 * (first + first_next)
    second_mid = 0.5 * (second + second_next)
    first_step = first_next - first
    second_step = second_next - second

    displacement = first_mid[:, None, :] - second_mid[None, :, :]
    distance_squared = np.einsum("ijk,ijk->ij", displacement, displacement)
    if np.any(distance_squared == 0.0):
        raise ValueError("curves intersect at segment midpoints")
    cross_steps = np.cross(first_step[:, None, :], second_step[None, :, :])
    numerator = np.einsum("ijk,ijk->ij", displacement, cross_steps)
    integral = np.sum(numerator / np.power(distance_squared, 1.5), dtype=np.float64)
    return float(integral / (4.0 * np.pi))


def linked_tori(
    n_per_class: int,
    tube_radius: float = 0.2,
    seed: int = 0,
    major_radius: float = 1.0,
) -> Dataset:
    """Sample two linked solid tori, with one balanced class per torus.

    Cross-section radii use ``tube_radius * sqrt(U)`` so points are uniform in
    cross-sectional area rather than concentrated near the core.
    """

    _validate_common(n_per_class, seed)
    if major_radius <= 0.0:
        raise ValueError("major_radius must be positive")
    # The minimum distance between the two core circles is exactly R.  Equal
    # tube radii are therefore disjoint when 2*tube_radius < R.
    if not 0.0 < tube_radius < major_radius / 2.0:
        raise ValueError(
            "tube_radius must be positive and less than half the minimum core separation"
        )

    rng = np.random.default_rng(seed)
    features: list[FloatArray] = []
    for class_index in range(2):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        cross_angle = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        radial_distance = tube_radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_per_class))

        cosine = np.cos(theta)
        sine = np.sin(theta)
        if class_index == 0:
            core = major_radius * np.column_stack((cosine, sine, np.zeros_like(theta)))
            normal = np.column_stack((cosine, sine, np.zeros_like(theta)))
            binormal = np.broadcast_to(np.array([0.0, 0.0, 1.0]), core.shape)
        else:
            core = major_radius * np.column_stack((1.0 + cosine, np.zeros_like(theta), sine))
            normal = np.column_stack((cosine, np.zeros_like(theta), sine))
            binormal = np.broadcast_to(np.array([0.0, 1.0, 0.0]), core.shape)

        offsets = radial_distance[:, None] * (
            np.cos(cross_angle)[:, None] * normal
            + np.sin(cross_angle)[:, None] * binormal
        )
        features.append(core + offsets)

    coordinates = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    labels = np.repeat(np.arange(2, dtype=np.int64), n_per_class)
    permutation = rng.permutation(coordinates.shape[0])
    return Dataset(coordinates[permutation], labels[permutation])


def gaussian_blobs(
    n_per_class: int,
    standard_deviation: float = 0.2,
    separation: float = 4.0,
    seed: int = 0,
) -> Dataset:
    """Sample two isotropic, well-separated Gaussian blobs in R^3."""

    _validate_common(n_per_class, seed)
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive")
    if separation <= 0.0:
        raise ValueError("separation must be positive")

    rng = np.random.default_rng(seed)
    centers = np.array([[-separation / 2.0, 0.0, 0.0], [separation / 2.0, 0.0, 0.0]])
    features = np.concatenate(
        [
            rng.normal(center, standard_deviation, size=(n_per_class, 3))
            for center in centers
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    labels = np.repeat(np.arange(2, dtype=np.int64), n_per_class)
    permutation = rng.permutation(features.shape[0])
    return Dataset(features[permutation], labels[permutation])
