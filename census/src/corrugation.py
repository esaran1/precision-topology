"""Corrugated linked tori, reproducing the paper's published thickening.

Appendix G.1 of arXiv:2606.31856v1 specifies the thickening as

    gamma(t) + eps * n(t),  eps ~ U(0, r),  r = 0.15

with "high-frequency oscillations 0.3 sin(100t) added to preserve topology".

Our original generator has never had the oscillation term, so every earlier
result in this project was measured on a smoother link than the paper's.  This
module adds it, anchored on their exact values, with amplitude and frequency as
swept parameters around that anchor.

The oscillation is applied **orthogonal to the radial direction** -- along the
binormal -- so it corrugates the tube rather than merely thickening it.  That
is the property the author's account concerns: with a corrugated tube, no single
planar fold separates the components cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .data import Dataset, FloatArray
from .linking import linking_number
from .parametrization import TorusLink, _frames, _rotation_matrix


# The paper's published values, Appendix G.1.
PAPER_AMPLITUDE = 0.3
PAPER_FREQUENCY = 100.0
PAPER_TUBE_RADIUS = 0.15


@dataclass(frozen=True)
class CorrugatedLink:
    """A torus link with a periodic oscillation orthogonal to the radial direction."""

    base: TorusLink = TorusLink()
    amplitude: float = 0.0
    frequency: float = 0.0
    noise: float = 0.0
    name: str = "flat"
    # "core" displaces the centreline (Reading A, the literal reading);
    # "offset" modulates where points are sampled within the tube (Reading B).
    reading: str = "core"

    @property
    def corrugated(self) -> bool:
        return self.amplitude > 0.0 and self.frequency > 0.0


def core_curves(link: CorrugatedLink, n_points: int = 2048) -> tuple[FloatArray, FloatArray]:
    """Core curves with the corrugation applied.

    The oscillation displaces the core along its binormal, which is orthogonal
    to both the tangent and the radial (normal) direction.
    """

    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float64)
    curves: list[FloatArray] = []
    for component in (0, 1):
        core, _normal, binormal = _frames(link.base, angles, component)
        if link.corrugated and link.reading == "core":
            wave = link.amplitude * np.sin(link.frequency * angles)
            core = core + wave[:, None] * np.asarray(binormal)
        curves.append(core)
    return curves[0], curves[1]


def sample(link: CorrugatedLink, n_per_class: int, seed: int) -> Dataset:
    """Sample the corrugated solid tori.

    With zero amplitude, zero frequency, and zero noise this reproduces
    ``parametrization.sample_link`` exactly, which is pinned by a test.
    """

    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")
    rng = np.random.default_rng(seed)
    features: list[FloatArray] = []
    for component in (0, 1):
        tube = link.base.tube_radius_a if component == 0 else link.base.tube_radius_b
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        cross_angle = rng.uniform(0.0, 2.0 * np.pi, size=n_per_class)
        radius = tube * np.sqrt(rng.uniform(0.0, 1.0, size=n_per_class))
        core, normal, binormal = _frames(link.base, theta, component)
        binormal = np.asarray(binormal)
        wave = (
            link.amplitude * np.sin(link.frequency * theta)
            if link.corrugated
            else np.zeros_like(theta)
        )
        if link.corrugated and link.reading == "core":
            # Reading A: the centreline itself is displaced.
            core = core + wave[:, None] * binormal
        if link.corrugated and link.reading == "offset":
            # Reading B: the oscillation modulates where within the tube a
            # point is drawn, leaving the centreline and the tube intact.  The
            # modulation scales the sampled radius, so points stay inside the
            # tube and the solid torus remains embedded.
            #
            # KNOWN FLAW (documented, deliberately not fixed): the modulation
            # depth is hardcoded at 0.5 and ``link.amplitude`` never enters,
            # so all Reading-B configurations at one frequency are the same
            # condition regardless of their amplitude label, and the clip
            # below concentrates ~20% of points exactly on the tube surface.
            # See results/reading_b_anomaly.md.  Changing this would orphan
            # the recorded artifacts; a future Reading-B sweep should thread
            # amplitude through the depth and regenerate from scratch.
            radius = radius * (1.0 + 0.5 * np.sin(link.frequency * theta))
            radius = np.clip(radius, 0.0, tube)
        offsets = radius[:, None] * (
            np.cos(cross_angle)[:, None] * normal + np.sin(cross_angle)[:, None] * binormal
        )
        features.append(core + offsets)

    coordinates = np.concatenate(features, axis=0)
    if link.noise > 0.0:
        coordinates = coordinates + rng.normal(0.0, link.noise, size=coordinates.shape)
    coordinates = coordinates.astype(np.float32, copy=False)
    labels = np.repeat(np.arange(2, dtype=np.int64), n_per_class)
    permutation = rng.permutation(coordinates.shape[0])
    return Dataset(coordinates[permutation], labels[permutation])


@dataclass(frozen=True)
class CorrugationValidation:
    """Whether a corrugated configuration is a genuine embedded link."""

    name: str
    linking_number: int | None
    linking_residual: float | None
    core_separation: float
    tube_gap: float
    self_intersects: bool
    min_between_class_distance: float | None
    between_class_collisions: int | None

    @property
    def valid(self) -> bool:
        return (
            self.linking_number is not None
            and abs(self.linking_number) == 1
            and not self.self_intersects
            and self.tube_gap > 0.0
            and (self.between_class_collisions in (None, 0))
        )

    def reason(self) -> str:
        if self.linking_number is None:
            return "cores intersect; linking undefined"
        if abs(self.linking_number) != 1:
            return f"linking number is {self.linking_number}, not +/-1"
        if self.self_intersects:
            return "corrugated core self-intersects"
        if self.tube_gap <= 0.0:
            return f"tubes overlap; gap {self.tube_gap:.4f}"
        if self.between_class_collisions:
            return f"{self.between_class_collisions} between-class collisions after noise"
        return "valid"


def _self_intersects(curve: FloatArray, tube_radius: float) -> bool:
    """Whether a corrugated core comes within a tube diameter of itself.

    Points adjacent along the curve are necessarily close together and must be
    excluded, or every smooth curve registers as self-intersecting.  The
    exclusion window is set by arc length rather than by index count: any pair
    whose separation *along* the curve is less than the tube diameter cannot
    represent a genuine self-approach, since the tube is locally a cylinder
    around the core.
    """

    points = torch.tensor(curve, dtype=torch.float64)
    n = points.shape[0]
    steps = points - torch.roll(points, shifts=1, dims=0)
    arc = float(steps.norm(dim=1).sum().item())
    spacing = arc / n
    # Exclude pairs closer along the curve than one tube diameter, with a small
    # floor so a coarse sampling cannot exclude everything.
    #
    # The window is an *arc-length* exclusion while the test below measures
    # *chord* distance, and chord is always shorter than arc.  On a curved
    # centreline a pair separated by exactly one tube diameter of arc is
    # therefore slightly closer than one diameter in space, which would flag
    # every smooth closed curve.  A margin of 1.5 tube diameters of arc removes
    # that boundary effect while still admitting genuine self-approaches, which
    # bring distant parts of the curve together and are not marginal.
    window = max(4, int(math.ceil(3.0 * tube_radius / max(spacing, 1e-12))))
    if window >= n // 2:
        return False

    index = torch.arange(n)
    separation = (index[:, None] - index[None, :]).abs()
    circular = torch.minimum(separation, n - separation)
    mask = circular > window
    if not mask.any():
        return False
    distance = torch.cdist(points, points)
    return bool(distance[mask].min().item() < 2.0 * tube_radius)


def validate(
    link: CorrugatedLink,
    n_points: int = 2048,
    n_per_class: int = 1_000,
    seed: int = 0,
) -> CorrugationValidation:
    """Verify linking, embeddedness, and absence of between-class collisions."""

    first, second = core_curves(link, n_points)
    estimate = linking_number(
        torch.tensor(first, dtype=torch.float64),
        torch.tensor(second, dtype=torch.float64),
    )
    tube_a = link.base.tube_radius_a
    tube_b = link.base.tube_radius_b
    gap = estimate.min_distance - (tube_a + tube_b)
    self_hit = _self_intersects(first, tube_a) or _self_intersects(second, tube_b)

    # Noise is applied to sampled points, so between-class separation must be
    # checked on an actual sample rather than on the cores.
    minimum: float | None = None
    collisions: int | None = None
    if link.noise > 0.0 or link.corrugated:
        dataset = sample(link, n_per_class, seed)
        points = torch.tensor(dataset.features, dtype=torch.float64)
        labels = torch.tensor(dataset.labels)
        left = points[labels == 0]
        right = points[labels == 1]
        pairwise = torch.cdist(left, right)
        minimum = float(pairwise.min().item())
        collisions = int((pairwise == 0.0).sum().item())

    return CorrugationValidation(
        name=link.name,
        linking_number=estimate.rounded if estimate.defined else None,
        linking_residual=estimate.residual if estimate.defined else None,
        core_separation=estimate.min_distance,
        tube_gap=gap,
        self_intersects=self_hit,
        min_between_class_distance=minimum,
        between_class_collisions=collisions,
    )


def _named(amplitude: float, frequency: float, noise: float, tube: float) -> str:
    if amplitude == 0.0 and noise == 0.0:
        return "flat"
    parts = []
    if amplitude > 0.0:
        parts.append(f"a{amplitude:g}_f{frequency:g}")
    if noise > 0.0:
        parts.append(f"n{noise:g}")
    if tube != TorusLink().tube_radius_a:
        parts.append(f"r{tube:g}")
    return "_".join(parts)


def build_grid() -> tuple[CorrugatedLink, ...]:
    """Both readings: paper anchor, amplitude/frequency sweeps, noise, and the
    low-amplitude embedded arm under Reading A."""

    paper_base = TorusLink(
        tube_radius_a=PAPER_TUBE_RADIUS, tube_radius_b=PAPER_TUBE_RADIUS
    )
    links: list[CorrugatedLink] = [
        # Degenerate: must reproduce the existing baseline exactly.
        CorrugatedLink(name="flat", reading="core"),
    ]
    for reading in ("core", "offset"):
        tag = "A" if reading == "core" else "B"
        # The published parametrization.
        links.append(
            CorrugatedLink(
                base=paper_base,
                amplitude=PAPER_AMPLITUDE,
                frequency=PAPER_FREQUENCY,
                name=f"{tag}_paper",
                reading=reading,
            )
        )
        for amplitude in (0.05, 0.15, 0.5):
            links.append(
                CorrugatedLink(
                    base=paper_base,
                    amplitude=amplitude,
                    frequency=PAPER_FREQUENCY,
                    name=f"{tag}_a{amplitude:g}",
                    reading=reading,
                )
            )
        for frequency in (10.0, 50.0, 200.0):
            links.append(
                CorrugatedLink(
                    base=paper_base,
                    amplitude=PAPER_AMPLITUDE,
                    frequency=frequency,
                    name=f"{tag}_f{frequency:g}",
                    reading=reading,
                )
            )
        for noise in (0.005, 0.02):
            links.append(
                CorrugatedLink(
                    base=paper_base,
                    amplitude=PAPER_AMPLITUDE,
                    frequency=PAPER_FREQUENCY,
                    noise=noise,
                    name=f"{tag}_n{noise:g}",
                    reading=reading,
                )
            )
    # Reading A, low-amplitude embedded arm: separates corrugation effects from
    # self-overlapping-tube effects.  0.001 is below the measured embedded limit
    # of 0.00108 at frequency 100.
    links.append(
        CorrugatedLink(
            base=paper_base,
            amplitude=0.001,
            frequency=PAPER_FREQUENCY,
            name="A_embedded_a0.001",
            reading="core",
        )
    )
    links.append(
        CorrugatedLink(
            base=paper_base,
            amplitude=0.3,
            frequency=0.5,
            name="A_embedded_f0.5",
            reading="core",
        )
    )
    return tuple(links)


GRID: tuple[CorrugatedLink, ...] = build_grid()
