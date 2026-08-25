"""Parametrised linked tori, for testing sensitivity to the configuration.

``data.linked_tori`` fixes almost everything about the link: both tori share one
tube radius and one major radius, the second core is offset by exactly one major
radius along ``x``, and the two cores lie in the coordinate planes ``z = 0`` and
``y = 0``.  That last property is the one the author's objection is about --
with the components axis-aligned, the direction a network must fold along is
exposed to a single coordinate, and a first affine layer can find it trivially.

This module makes those choices explicit parameters so the width-3 result can
be checked against configurations that do not hand the fold direction over.

Every configuration is verified before use: the two cores must be a genuine
link (|linking number| = 1 by the validated Gauss estimator) and the two solid
tori must not intersect each other or self-intersect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from .data import Dataset, FloatArray
from .linking import linking_number


@dataclass(frozen=True)
class TorusLink:
    """A parametrised pair of linked solid tori.

    ``offset`` is the centre of the second core relative to the first, in units
    of the first major radius.  ``rotation_degrees`` tilts the second component
    away from the coordinate planes, which is what removes the axis alignment.
    """

    major_radius_a: float = 1.0
    major_radius_b: float = 1.0
    tube_radius_a: float = 0.2
    tube_radius_b: float = 0.2
    offset: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "baseline"

    def aspect_a(self) -> float:
        return self.tube_radius_a / self.major_radius_a

    def aspect_b(self) -> float:
        return self.tube_radius_b / self.major_radius_b


def _rotation_matrix(degrees: tuple[float, float, float]) -> FloatArray:
    """Extrinsic x-y-z rotation, in degrees."""

    x, y, z = (math.radians(value) for value in degrees)
    rx = np.array(
        [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]],
        dtype=np.float64,
    )
    ry = np.array(
        [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]],
        dtype=np.float64,
    )
    rz = np.array(
        [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def core_curves(link: TorusLink, n_points: int = 512) -> tuple[FloatArray, FloatArray]:
    """Ordered samples of the two core circles."""

    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)
    cosine, sine, zeros = np.cos(angles), np.sin(angles), np.zeros_like(angles)

    first = link.major_radius_a * np.column_stack((cosine, sine, zeros))
    # The second core starts in the x-z plane, the configuration that links it
    # with the first, and is then rotated and translated.
    second = link.major_radius_b * np.column_stack((cosine, zeros, sine))
    second = second @ _rotation_matrix(link.rotation_degrees).T
    second = second + np.asarray(link.offset, dtype=np.float64) * link.major_radius_a
    return first, second


def _frames(link: TorusLink, theta: FloatArray, component: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Core point, normal, and binormal for one component at angles ``theta``."""

    cosine, sine, zeros = np.cos(theta), np.sin(theta), np.zeros_like(theta)
    if component == 0:
        core = link.major_radius_a * np.column_stack((cosine, sine, zeros))
        normal = np.column_stack((cosine, sine, zeros))
        binormal = np.broadcast_to(np.array([0.0, 0.0, 1.0]), core.shape)
        return core, normal, binormal

    core = link.major_radius_b * np.column_stack((cosine, zeros, sine))
    normal = np.column_stack((cosine, zeros, sine))
    binormal = np.broadcast_to(np.array([0.0, 1.0, 0.0]), core.shape)
    rotation = _rotation_matrix(link.rotation_degrees).T
    shift = np.asarray(link.offset, dtype=np.float64) * link.major_radius_a
    return core @ rotation + shift, normal @ rotation, np.asarray(binormal) @ rotation


def sample_link(link: TorusLink, n_per_class: int, seed: int) -> Dataset:
    """Sample points uniformly by cross-sectional area from both solid tori."""

    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")
    rng = np.random.default_rng(seed)
    features: list[FloatArray] = []
    for component in (0, 1):
        tube = link.tube_radius_a if component == 0 else link.tube_radius_b
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        cross_angle = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        radius = tube * np.sqrt(rng.uniform(0.0, 1.0, size=n_per_class))
        core, normal, binormal = _frames(link, theta, component)
        offsets = radius[:, None] * (
            np.cos(cross_angle)[:, None] * normal + np.sin(cross_angle)[:, None] * binormal
        )
        features.append(core + offsets)

    coordinates = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    labels = np.repeat(np.arange(2, dtype=np.int64), n_per_class)
    permutation = rng.permutation(coordinates.shape[0])
    return Dataset(coordinates[permutation], labels[permutation])


@dataclass(frozen=True)
class LinkValidation:
    """Whether a configuration is a genuine, embedded, disjoint link."""

    name: str
    linking_number: int | None
    linking_residual: float | None
    core_separation: float
    tube_gap: float
    self_intersects_a: bool
    self_intersects_b: bool
    tori_intersect: bool

    @property
    def valid(self) -> bool:
        return (
            self.linking_number is not None
            and abs(self.linking_number) == 1
            and not self.self_intersects_a
            and not self.self_intersects_b
            and not self.tori_intersect
        )

    def reason(self) -> str:
        if self.linking_number is None:
            return "cores intersect; linking undefined"
        if abs(self.linking_number) != 1:
            return f"linking number is {self.linking_number}, not +/-1"
        if self.self_intersects_a or self.self_intersects_b:
            return "a tube self-intersects (tube radius exceeds major radius)"
        if self.tori_intersect:
            return f"tubes overlap; gap {self.tube_gap:.4f}"
        return "valid"


def validate(link: TorusLink, n_points: int = 512) -> LinkValidation:
    """Check linking number and embeddedness before any training happens."""

    first, second = core_curves(link, n_points)
    estimate = linking_number(
        torch.tensor(first, dtype=torch.float64),
        torch.tensor(second, dtype=torch.float64),
    )
    separation = estimate.min_distance
    # A solid torus is embedded exactly when its tube radius is below its major
    # radius; at or beyond that the tube closes through the hole.
    self_a = link.tube_radius_a >= link.major_radius_a
    self_b = link.tube_radius_b >= link.major_radius_b
    gap = separation - (link.tube_radius_a + link.tube_radius_b)
    from .linking_trace import ARTIFACT_DISTANCE

    usable = estimate.defined and separation > ARTIFACT_DISTANCE
    return LinkValidation(
        name=link.name,
        linking_number=estimate.rounded if usable else None,
        linking_residual=estimate.residual if usable else None,
        core_separation=separation,
        tube_gap=gap,
        self_intersects_a=self_a,
        self_intersects_b=self_b,
        tori_intersect=gap <= 0.0,
    )


def axis_alignment(link: TorusLink, n_points: int = 512) -> tuple[float, float]:
    """How closely each core lies within a coordinate plane.

    Returns the smallest coordinate-wise standard deviation of each core.  A
    value near zero means the component is confined to a coordinate plane, so a
    single input coordinate distinguishes it and the fold direction is exposed.
    """

    first, second = core_curves(link, n_points)
    return float(first.std(axis=0).min()), float(second.std(axis=0).min())


# The grid.  `baseline` reproduces data.linked_tori exactly.
GRID: tuple[TorusLink, ...] = (
    TorusLink(name="baseline"),
    # Tube radius, holding everything else fixed.
    TorusLink(tube_radius_a=0.05, tube_radius_b=0.05, name="thin_tube"),
    TorusLink(tube_radius_a=0.35, tube_radius_b=0.35, name="thick_tube"),
    # Aspect ratio, including the asymmetric case the baseline cannot express.
    # A larger second major radius pulls its core towards the first, so the
    # offset is raised to keep the tubes disjoint; validate() rejects the
    # configurations where it is not.
    TorusLink(major_radius_b=1.6, offset=(1.6, 0.0, 0.0), name="unequal_major"),
    TorusLink(tube_radius_a=0.1, tube_radius_b=0.35, name="asymmetric_tube"),
    TorusLink(
        major_radius_b=1.6,
        offset=(1.6, 0.0, 0.0),
        tube_radius_a=0.1,
        tube_radius_b=0.35,
        name="asymmetric_both",
    ),
    # Offset, holding the linking number at +/-1.
    TorusLink(offset=(0.7, 0.0, 0.0), name="near_offset"),
    TorusLink(offset=(1.3, 0.0, 0.0), name="far_offset"),
    TorusLink(offset=(1.0, 0.25, 0.15), name="oblique_offset"),
    # Rotations that break the coordinate-plane alignment of the second core.
    TorusLink(rotation_degrees=(0.0, 0.0, 30.0), name="rotated_30"),
    TorusLink(rotation_degrees=(20.0, 35.0, 15.0), name="rotated_generic"),
    TorusLink(
        offset=(1.0, 0.2, 0.1),
        rotation_degrees=(25.0, 40.0, 20.0),
        tube_radius_a=0.15,
        tube_radius_b=0.25,
        major_radius_b=1.3,
        name="generic",
    ),
)
