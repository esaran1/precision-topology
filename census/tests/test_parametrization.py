"""Tests for the parametrised torus-link generator.

The generator's job is to vary the link configuration without ever producing a
configuration that is not a genuine embedded link.  A silent unlink or a
self-intersecting tube would invalidate its cell of the sweep without failing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import linked_tori
from src.linking import linking_number
from src.parametrization import (
    GRID,
    TorusLink,
    axis_alignment,
    core_curves,
    sample_link,
    validate,
)


def test_baseline_reproduces_the_original_generator():
    """The baseline configuration must match data.linked_tori exactly."""

    link = TorusLink(name="baseline")
    generated = sample_link(link, 500, seed=7)
    original = linked_tori(500, tube_radius=0.2, seed=7, major_radius=1.0)
    assert np.array_equal(generated.features, original.features)
    assert np.array_equal(generated.labels, original.labels)


def test_every_grid_configuration_is_a_valid_link():
    for link in GRID:
        report = validate(link)
        assert report.valid, f"{link.name}: {report.reason()}"
        assert abs(report.linking_number) == 1
        assert report.tube_gap > 0.0


def test_validation_rejects_overlapping_tubes():
    """Tubes that intersect must be caught, not silently trained on.

    The cores are separated by 1.0 in the baseline geometry, so the tubes
    overlap once their radii sum past that.
    """

    link = TorusLink(tube_radius_a=0.55, tube_radius_b=0.55, name="overlapping")
    report = validate(link)
    assert report.tube_gap < 0.0
    assert report.tori_intersect
    assert not report.valid


def test_validation_rejects_a_self_intersecting_tube():
    link = TorusLink(major_radius_b=0.3, tube_radius_b=0.4, name="self_intersecting")
    report = validate(link)
    assert not report.valid


def test_validation_rejects_an_unlinked_configuration():
    """A large offset separates the components entirely; linking becomes 0."""

    link = TorusLink(offset=(8.0, 0.0, 0.0), name="unlinked")
    report = validate(link)
    assert report.linking_number == 0
    assert not report.valid


def test_rotation_breaks_axis_alignment():
    """The property the author's objection is about."""

    aligned = axis_alignment(TorusLink(name="baseline"))
    rotated = axis_alignment(TorusLink(rotation_degrees=(20.0, 35.0, 15.0)))
    assert aligned[1] < 1e-9
    assert rotated[1] > 1e-3


def test_rotation_preserves_the_linking_number():
    for degrees in ((0.0, 0.0, 30.0), (20.0, 35.0, 15.0), (45.0, 10.0, 60.0)):
        link = TorusLink(rotation_degrees=degrees)
        assert abs(validate(link).linking_number) == 1


def test_asymmetric_configurations_are_expressible():
    """The original generator cannot express these at all."""

    link = TorusLink(tube_radius_a=0.1, tube_radius_b=0.35, major_radius_b=1.6, offset=(1.6, 0.0, 0.0))
    assert link.aspect_a() != link.aspect_b()
    assert validate(link).valid


def test_sampled_points_lie_within_their_tubes():
    link = TorusLink(tube_radius_a=0.2, tube_radius_b=0.3, name="check")
    dataset = sample_link(link, 400, seed=1)
    first_core, second_core = core_curves(link, 2048)
    points = torch.tensor(dataset.features, dtype=torch.float64)
    labels = torch.tensor(dataset.labels)
    for label, core, tube in (
        (0, first_core, link.tube_radius_a),
        (1, second_core, link.tube_radius_b),
    ):
        selected = points[labels == label]
        distance = torch.cdist(selected, torch.tensor(core, dtype=torch.float64)).min(dim=1).values
        # Core sampling is discrete, so allow a small polygonal slack.
        assert float(distance.max()) <= tube + 1e-2


def test_sampling_is_deterministic_and_seed_dependent():
    link = TorusLink(name="baseline")
    first = sample_link(link, 200, seed=3)
    again = sample_link(link, 200, seed=3)
    different = sample_link(link, 200, seed=4)
    assert np.array_equal(first.features, again.features)
    assert not np.array_equal(first.features, different.features)


def test_core_curves_are_linked_by_the_validated_estimator():
    for link in GRID:
        first, second = core_curves(link, 512)
        estimate = linking_number(
            torch.tensor(first, dtype=torch.float64),
            torch.tensor(second, dtype=torch.float64),
        )
        assert estimate.defined
        assert abs(estimate.rounded) == 1


def test_classes_are_balanced():
    dataset = sample_link(TorusLink(), 250, seed=0)
    counts = np.bincount(dataset.labels)
    assert counts.tolist() == [250, 250]


def test_grid_covers_the_intended_axes():
    names = {link.name for link in GRID}
    assert {"thin_tube", "thick_tube"} <= names
    assert {"asymmetric_tube", "unequal_major"} <= names
    assert {"near_offset", "far_offset", "oblique_offset"} <= names
    assert {"rotated_30", "rotated_generic", "generic"} <= names
    # At least one configuration must break axis alignment.
    assert any(axis_alignment(link)[1] > 1e-3 for link in GRID)
