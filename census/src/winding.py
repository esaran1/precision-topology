"""Links of controllable linking number: a ring and a solenoid around it.

Component A is a plain circle.  Component B winds ``q`` times around A's
core while traversing once around, giving linking number ±q by
construction -- "increasing winding between two rings" rather than copies
of Hopf links.  Every configuration is verified with the validated Gauss
estimator before use, and both tubes are checked for embeddedness.

Solid tubes are sampled with parallel-transport (double-reflection) frames,
because B has no closed-form normal frame and a naive Frenet frame flips at
inflections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import Dataset, FloatArray
from .linking import linking_number


RING_RADIUS = 1.0
WINDING_MINOR_RADIUS = 0.35
TUBE_RADIUS_A = 0.12
TUBE_RADIUS_B = 0.08


@dataclass(frozen=True)
class WindingLink:
    q: int  # winding count; |linking number| equals q
    minor_radius: float = WINDING_MINOR_RADIUS
    tube_radius_a: float = TUBE_RADIUS_A
    tube_radius_b: float = TUBE_RADIUS_B

    @property
    def name(self) -> str:
        return f"winding_q{self.q}"


def core_a(n_points: int = 4096) -> FloatArray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)
    return np.column_stack(
        [RING_RADIUS * np.cos(t), RING_RADIUS * np.sin(t), np.zeros_like(t)]
    )


def core_b(link: WindingLink, n_points: int = 4096) -> FloatArray:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)
    radius = RING_RADIUS + link.minor_radius * np.cos(link.q * t)
    return np.column_stack(
        [
            radius * np.cos(t),
            radius * np.sin(t),
            link.minor_radius * np.sin(link.q * t),
        ]
    )


def _transport_frames(core: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Parallel-transport normal and binormal along a closed polyline."""

    tangents = np.roll(core, -1, axis=0) - np.roll(core, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.empty_like(core)
    binormals = np.empty_like(core)
    # Initial normal: any vector orthogonal to the first tangent.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(tangents[0] @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    normal = reference - (reference @ tangents[0]) * tangents[0]
    normal /= np.linalg.norm(normal)
    for index in range(core.shape[0]):
        tangent = tangents[index]
        normal = normal - (normal @ tangent) * tangent
        normal /= np.linalg.norm(normal)
        normals[index] = normal
        binormals[index] = np.cross(tangent, normal)
    return normals, binormals


def sample(link: WindingLink, n_per_class: int, seed: int) -> Dataset:
    """Sample the two solid tubes, uniform by cross-sectional area."""

    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")
    rng = np.random.default_rng(seed)
    features: list[FloatArray] = []
    dense = 8192
    for component, tube in ((0, link.tube_radius_a), (1, link.tube_radius_b)):
        core = core_a(dense) if component == 0 else core_b(link, dense)
        normals, binormals = _transport_frames(core)
        indices = rng.integers(0, dense, size=n_per_class)
        cross_angle = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        radius = tube * np.sqrt(rng.uniform(0.0, 1.0, size=n_per_class))
        offsets = radius[:, None] * (
            np.cos(cross_angle)[:, None] * normals[indices]
            + np.sin(cross_angle)[:, None] * binormals[indices]
        )
        features.append(core[indices] + offsets)
    coordinates = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    labels = np.repeat(np.arange(2, dtype=np.int64), n_per_class)
    permutation = rng.permutation(coordinates.shape[0])
    return Dataset(coordinates[permutation], labels[permutation])


@dataclass(frozen=True)
class WindingValidation:
    name: str
    q: int
    linking_number: int | None
    linking_residual: float | None
    core_gap: float
    b_self_gap: float
    tube_clearance: float
    b_self_clearance: float
    min_between_class_distance: float
    between_class_collisions: int

    @property
    def valid(self) -> bool:
        return (
            self.linking_number is not None
            and abs(self.linking_number) == self.q
            and self.tube_clearance > 0.0
            and self.b_self_clearance > 0.0
            and self.between_class_collisions == 0
        )


def _self_gap(core: FloatArray, tube: float) -> float:
    """Minimum distance between non-neighbouring points of one core."""

    points = torch.tensor(core, dtype=torch.float64)
    n = points.shape[0]
    steps = points - torch.roll(points, 1, dims=0)
    arc = float(steps.norm(dim=1).sum().item())
    spacing = arc / n
    window = max(4, int(np.ceil(3.0 * tube / max(spacing, 1e-12))))
    index = torch.arange(n)
    separation = (index[:, None] - index[None, :]).abs()
    circular = torch.minimum(separation, n - separation)
    mask = circular > window
    distance = torch.cdist(points, points)
    return float(distance[mask].min().item())


def validate(link: WindingLink, n_points: int = 4096, sample_seed: int = 0) -> WindingValidation:
    first = core_a(n_points)
    second = core_b(link, n_points)
    estimate = linking_number(
        torch.tensor(first, dtype=torch.float64),
        torch.tensor(second, dtype=torch.float64),
    )
    core_gap = estimate.min_distance
    b_gap = _self_gap(second, link.tube_radius_b)
    data = sample(link, 1_000, sample_seed)
    points = torch.tensor(data.features, dtype=torch.float64)
    labels = torch.tensor(data.labels)
    pairwise = torch.cdist(points[labels == 0], points[labels == 1])
    return WindingValidation(
        name=link.name,
        q=link.q,
        linking_number=estimate.rounded if estimate.defined else None,
        linking_residual=estimate.residual if estimate.defined else None,
        core_gap=core_gap,
        b_self_gap=b_gap,
        tube_clearance=core_gap - (link.tube_radius_a + link.tube_radius_b),
        b_self_clearance=b_gap - 2.0 * link.tube_radius_b,
        min_between_class_distance=float(pairwise.min().item()),
        between_class_collisions=int((pairwise == 0.0).sum().item()),
    )


GRID: tuple[WindingLink, ...] = tuple(WindingLink(q=q) for q in (1, 2, 3, 4))
