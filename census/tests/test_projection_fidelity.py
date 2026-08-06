"""Tests for the projection-fidelity control.

The projected linking sweep returns zero everywhere above width 3.  These tests
pin the control that distinguishes a real null from a blind measure.
"""

from __future__ import annotations

import pytest
import torch

from src.linking import linking_number
from src.linking_trace import _project_to_r3
from src.projection_fidelity import (
    embed_and_rotate,
    fidelity_trial,
    random_rotation,
    run_fidelity,
)


def test_random_rotation_is_orthogonal():
    for dimension in (3, 5, 15):
        rotation = random_rotation(dimension, seed=0)
        identity = torch.eye(dimension, dtype=torch.float64)
        assert torch.allclose(rotation @ rotation.T, identity, atol=1e-10)
        assert abs(abs(float(torch.linalg.det(rotation))) - 1.0) < 1e-10


def test_rotation_depends_on_seed():
    first = random_rotation(6, seed=0)
    second = random_rotation(6, seed=1)
    assert not torch.allclose(first, second)


def test_embedding_preserves_distances():
    """The embedding must be isometric, or the control tests the wrong thing.

    Pairwise differences are compared by explicit norm rather than via cdist,
    whose squared-expansion accumulates error around 1e-8 at these magnitudes
    and would force a tolerance loose enough to hide a real defect.
    """

    curve = torch.randn(40, 3, dtype=torch.float64)
    rotation = random_rotation(9, seed=3)
    embedded = embed_and_rotate(curve, 9, rotation)
    original = (curve.unsqueeze(1) - curve.unsqueeze(0)).norm(dim=2)
    mapped = (embedded.unsqueeze(1) - embedded.unsqueeze(0)).norm(dim=2)
    assert torch.allclose(original, mapped, atol=1e-12)


def test_embedding_rejects_dimensions_below_three():
    with pytest.raises(ValueError):
        embed_and_rotate(torch.zeros(5, 3, dtype=torch.float64), 2, torch.eye(2, dtype=torch.float64))


@pytest.mark.parametrize("dimension", [3, 4, 5, 8, 15])
def test_projection_recovers_known_linking(dimension: int):
    """The decisive control: joint PCA must see a link it is given."""

    result = fidelity_trial(dimension, seed=0, n_points=256)
    assert result.defined
    assert abs(result.recovered) == 1
    assert result.correct


def test_projection_recovers_linking_across_seeds():
    results = run_fidelity(dimensions=(4, 8, 15), seeds=tuple(range(5)), n_points=256)
    assert all(result.correct for result in results)


def test_projection_survives_a_nonorthogonal_map_and_activation():
    """The realistic case: what an untrained layer actually applies."""

    from src.data import linked_core_circles

    first_array, second_array = linked_core_circles(256)
    first = torch.as_tensor(first_array, dtype=torch.float64)
    second = torch.as_tensor(second_array, dtype=torch.float64)
    for dimension in (4, 8, 15):
        generator = torch.Generator().manual_seed(dimension)
        weight = torch.randn(3, dimension, dtype=torch.float64, generator=generator) / (3**0.5)
        bias = torch.randn(dimension, dtype=torch.float64, generator=generator) * 0.1
        left = torch.tanh(first @ weight + bias)
        right = torch.tanh(second @ weight + bias)
        projected_left, projected_right = _project_to_r3(left, right)
        estimate = linking_number(projected_left, projected_right)
        assert estimate.defined
        assert abs(estimate.rounded) == 1


def test_unlinked_configuration_projects_to_zero():
    """The control must also report zero when there is genuinely no link."""

    from src.linking import unlink

    first, second = unlink(256)
    rotation = random_rotation(7, seed=11)
    embedded_first = embed_and_rotate(first, 7, rotation)
    embedded_second = embed_and_rotate(second, 7, rotation)
    projected_left, projected_right = _project_to_r3(embedded_first, embedded_second)
    assert linking_number(projected_left, projected_right).rounded == 0
