"""Tests for between-class margin measured in quantization steps."""

from __future__ import annotations

import pytest
import torch

from src.margin import between_class_margin, local_ulp


def test_fixed_point_step_is_uniform_grid_spacing():
    values = torch.tensor([-0.9, 0.0, 0.3, 0.95], dtype=torch.float64)
    assert torch.allclose(local_ulp(values, "fixed-4"), torch.full_like(values, 0.125))
    assert torch.allclose(local_ulp(values, "fixed-6"), torch.full_like(values, 0.03125))


def test_ieee_ulp_is_exponent_dependent():
    """The whole point of the local convention: spacing is not constant."""

    at_one = local_ulp(torch.tensor([1.0], dtype=torch.float64), "bfloat16").item()
    at_half = local_ulp(torch.tensor([0.5], dtype=torch.float64), "bfloat16").item()
    assert at_one == pytest.approx(2.0**-7)
    assert at_half == pytest.approx(2.0**-8)
    assert at_one != at_half

    float32_ulp = local_ulp(torch.tensor([1.0], dtype=torch.float64), "float32").item()
    assert float32_ulp == pytest.approx(2.0**-23)


def test_margin_below_one_when_pair_shares_a_cell():
    # Two opposite-class points far closer than one fixed-4 cell.
    preactivations = torch.tensor(
        [[0.0, 0.0], [1e-9, 0.0], [4.0, 4.0]], dtype=torch.float64
    )
    labels = torch.tensor([0, 1, 0])
    measurement = between_class_margin(preactivations, labels, "fixed-4", "tanh")
    assert measurement.margin_in_steps_chebyshev < 1.0
    assert measurement.below_one
    assert measurement.between_class_collision_pairs == 1


def test_margin_above_one_when_classes_are_well_separated():
    preactivations = torch.tensor(
        [[-3.0, -3.0], [-3.0, -3.0], [3.0, 3.0], [3.0, 3.0]], dtype=torch.float64
    )
    labels = torch.tensor([0, 0, 1, 1])
    measurement = between_class_margin(preactivations, labels, "fixed-4", "tanh")
    assert measurement.margin_in_steps_chebyshev > 1.0
    assert not measurement.below_one
    assert measurement.between_class_collision_pairs == 0


def test_margin_is_measured_on_post_activation_values():
    """tanh saturation compresses a wide preactivation gap into a small margin."""

    # A gap of 6.0 in preactivation space between the two classes.
    preactivations = torch.tensor([[3.0], [9.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    measurement = between_class_margin(preactivations, labels, "fixed-6", "tanh")
    expected = float(torch.tanh(torch.tensor(9.0, dtype=torch.float64)) - torch.tanh(torch.tensor(3.0, dtype=torch.float64)))
    assert measurement.min_between_class_distance == pytest.approx(expected)
    # Post-activation the gap is under 0.01 despite spanning 6.0 before tanh,
    # which is why the distance must not be taken on preactivations.
    assert measurement.min_between_class_distance < 0.01


def test_between_class_margin_requires_two_classes():
    preactivations = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        between_class_margin(preactivations, torch.tensor([0, 0]), "fixed-4", "tanh")


def test_labels_must_match_input_count():
    preactivations = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        between_class_margin(preactivations, torch.tensor([0, 1, 0]), "fixed-4", "tanh")


def test_unknown_quantizer_is_rejected():
    with pytest.raises(ValueError):
        local_ulp(torch.tensor([0.5], dtype=torch.float64), "float8")


def test_collision_pairs_counted_across_classes_only():
    """Three inputs in one cell, two of class 0 and one of class 1 -> 2 pairs."""

    preactivations = torch.tensor(
        [[0.0], [1e-9], [2e-9], [4.0]], dtype=torch.float64
    )
    labels = torch.tensor([0, 0, 1, 1])
    measurement = between_class_margin(preactivations, labels, "fixed-4", "tanh")
    assert measurement.between_class_collision_pairs == 2
