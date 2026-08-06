"""Gauss linking-number estimation on ordered polygonal cycles.

This follows the direct Appendix G.8 approach rather than the Appendix H
point-cloud detector, because our class cores are known parametrised curves:
we can propagate ordered samples and evaluate the Gauss integral on the
resulting polygon directly, with no PCA projection and no cycle reconstruction.

Two facts govern the implementation.

First, the linking number is only defined for *disjoint* curves.  As the two
components approach intersection the Gauss integrand blows up like
``1/|x-y|^2`` and the quadrature becomes ill-conditioned; Ren and Lim's own
Table 8 reports fractional values at exactly that point and stars them as
artifacts.  :func:`linking_number` therefore reports the minimum inter-curve
distance alongside the estimate and flags the undefined case rather than
silently returning a fraction.

Second, the exact value is an integer.  The distance of the raw estimate from
the nearest integer is the natural residual, and it is reported so that a
caller can tell a converged estimate from a meaningless one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class LinkingEstimate:
    """One Gauss-integral estimate with the diagnostics needed to trust it."""

    raw: float
    rounded: int
    residual: float
    min_distance: float
    segments: tuple[int, int]
    defined: bool

    @property
    def converged(self) -> bool:
        """Whether the estimate is close enough to an integer to be usable."""

        return self.defined and self.residual < 0.25


def _as_cycle(points: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(points, dtype=torch.float64, device="cpu")
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("cycle must have shape (n_points, 3)")
    if values.shape[0] < 3:
        raise ValueError("a cycle needs at least three points")
    return values


def subdivide(cycle: torch.Tensor, factor: int) -> torch.Tensor:
    """Insert ``factor - 1`` evenly spaced points along every edge."""

    if factor < 1:
        raise ValueError("subdivision factor must be at least 1")
    values = _as_cycle(cycle)
    if factor == 1:
        return values
    following = torch.roll(values, shifts=-1, dims=0)
    weights = torch.arange(factor, dtype=torch.float64).reshape(1, factor, 1) / factor
    points = values.unsqueeze(1) * (1.0 - weights) + following.unsqueeze(1) * weights
    return points.reshape(-1, 3)


def minimum_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Smallest distance between any two vertices of the two cycles."""

    left = _as_cycle(first)
    right = _as_cycle(second)
    return float(torch.cdist(left, right).min().item())


def linking_number(
    first: torch.Tensor,
    second: torch.Tensor,
    subdivisions: int = 1,
) -> LinkingEstimate:
    """Estimate the Gauss linking integral of two ordered closed polygons.

    Midpoint quadrature over segment pairs, matching the discretisation the
    paper describes.  Both inputs are treated as closed: the final vertex is
    joined back to the first.
    """

    left = subdivide(_as_cycle(first), subdivisions)
    right = subdivide(_as_cycle(second), subdivisions)

    left_next = torch.roll(left, shifts=-1, dims=0)
    right_next = torch.roll(right, shifts=-1, dims=0)
    left_mid = 0.5 * (left + left_next)
    right_mid = 0.5 * (right + right_next)
    left_tangent = left_next - left
    right_tangent = right_next - right

    separation = left_mid.unsqueeze(1) - right_mid.unsqueeze(0)
    distance = separation.norm(dim=2)
    cross = torch.cross(
        left_tangent.unsqueeze(1).expand(-1, right_tangent.shape[0], -1),
        right_tangent.unsqueeze(0).expand(left_tangent.shape[0], -1, -1),
        dim=2,
    )
    numerator = (separation * cross).sum(dim=2)
    # Guard the singular case; `defined` below reports whether it was hit.
    safe = distance.clamp_min(1e-300)
    raw = float((numerator / safe.pow(3)).sum().item() / (4.0 * math.pi))

    closest = minimum_distance(left, right)
    # The integral is only defined for disjoint curves, and it degrades
    # continuously as they approach: `defined` records disjointness, while
    # `converged` (below) records whether the estimate is actually usable.
    defined = bool(closest > 0.0) and math.isfinite(raw)
    rounded = int(round(raw)) if defined else 0
    residual = abs(raw - rounded) if defined else float("nan")
    return LinkingEstimate(
        raw=raw,
        rounded=rounded,
        residual=residual,
        min_distance=closest,
        segments=(int(left.shape[0]), int(right.shape[0])),
        defined=defined,
    )


# --- Reference configurations with known linking number -------------------


def circle(
    n_points: int,
    radius: float = 1.0,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    plane: str = "xy",
    reverse: bool = False,
    seed: int | None = None,
    jitter: float = 0.0,
) -> torch.Tensor:
    """An ordered circular cycle, optionally jittered for noise-floor tests."""

    angles = torch.arange(n_points, dtype=torch.float64) * (2.0 * math.pi / n_points)
    if reverse:
        angles = torch.flip(angles, dims=[0])
    cosine = radius * torch.cos(angles)
    sine = radius * torch.sin(angles)
    zero = torch.zeros_like(cosine)
    if plane == "xy":
        points = torch.stack([cosine, sine, zero], dim=1)
    elif plane == "xz":
        points = torch.stack([cosine, zero, sine], dim=1)
    elif plane == "yz":
        points = torch.stack([zero, cosine, sine], dim=1)
    else:
        raise ValueError(f"unknown plane: {plane}")
    points = points + torch.tensor(centre, dtype=torch.float64)
    if jitter > 0.0:
        generator = torch.Generator().manual_seed(0 if seed is None else seed)
        points = points + jitter * torch.randn(
            points.shape, dtype=torch.float64, generator=generator
        )
    return points


def unlink(n_points: int = 200, separation: float = 6.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two far-apart coplanar circles; linking number 0."""

    return (
        circle(n_points, centre=(-separation / 2.0, 0.0, 0.0)),
        circle(n_points, centre=(separation / 2.0, 0.0, 0.0)),
    )


def hopf_link(n_points: int = 200, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """The standard Hopf link; linking number -1, or +1 when reversed."""

    return (
        circle(n_points, radius=1.0, centre=(0.0, 0.0, 0.0), plane="xy"),
        circle(
            n_points,
            radius=1.0,
            centre=(1.0, 0.0, 0.0),
            plane="xz",
            reverse=reverse,
        ),
    )


def torus_link_2_4(n_points: int = 400) -> tuple[torch.Tensor, torch.Tensor]:
    """The (2,4) torus link: two curves on a torus with linking number 2.

    Each component winds once around the tube while the pair winds twice
    around the core, giving |link| = 2.
    """

    parameter = torch.arange(n_points, dtype=torch.float64) * (2.0 * math.pi / n_points)
    major, minor = 2.0, 0.6

    def component(offset: float) -> torch.Tensor:
        angle = 2.0 * parameter + offset
        radius = major + minor * torch.cos(angle)
        return torch.stack(
            [radius * torch.cos(parameter), radius * torch.sin(parameter), minor * torch.sin(angle)],
            dim=1,
        )

    return component(0.0), component(math.pi)


def chain(n_points: int = 200) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three circles in a chain: adjacent pairs link, the ends do not.

    The middle circle is enlarged so that the two end circles are separated
    rather than tangent.  Coplanar unit circles at centre distance 2 touch at a
    single point, which makes the end pair a degenerate rather than a clean
    unlink and leaves the integral undefined.
    """

    first = circle(n_points, radius=1.0, centre=(-1.6, 0.0, 0.0), plane="xy")
    second = circle(n_points, radius=1.8, centre=(0.0, 0.0, 0.0), plane="xz")
    third = circle(n_points, radius=1.0, centre=(1.6, 0.0, 0.0), plane="xy")
    return first, second, third


def intersecting(n_points: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Two circles that genuinely cross, so the linking number is undefined.

    Unit circles in the xy- and xz-planes sharing a centre intersect
    transversally at (±1, 0, 0).
    """

    return (
        circle(n_points, radius=1.0, centre=(0.0, 0.0, 0.0), plane="xy"),
        circle(n_points, radius=1.0, centre=(0.0, 0.0, 0.0), plane="xz"),
    )
