"""Tests for the parametric activation families and the threshold sweep."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models import MLP, parametric_monotonic
from src.threshold_sweep import (
    FAMILY_A,
    FAMILY_B,
    ThresholdSweepConfig,
    condition_monotonic,
    conditions,
    inactive_unit_fraction,
)
from src.train import TrainingConfig, train_mlp
from src.data import linked_tori


def _numeric_monotonic(model: MLP, low: float = -12.0, high: float = 12.0) -> bool:
    """Empirical monotonicity of the model's activation on a dense grid."""

    values = torch.linspace(low, high, 20_001, dtype=torch.float64)
    outputs = model._activate(values)
    return bool((outputs[1:] >= outputs[:-1] - 1e-12).all().item())


def test_family_a_monotonic_exactly_up_to_one():
    """f_a(x) = x + a sin x is monotonic iff a <= 1, including a = 1 itself."""

    for a in FAMILY_A:
        model = MLP(1, 1, 1, "sin_family", activation_parameter=a)
        assert _numeric_monotonic(model) == (a <= 1.0), a
        assert parametric_monotonic("sin_family", a) == (a <= 1.0)


def test_family_b_monotonic_exactly_down_to_zero():
    """g_alpha is monotonic iff alpha >= 0, including ReLU at alpha = 0."""

    for alpha in FAMILY_B:
        model = MLP(1, 1, 1, "pwl_family", activation_parameter=alpha)
        assert _numeric_monotonic(model) == (alpha >= 0.0), alpha
        assert parametric_monotonic("pwl_family", alpha) == (alpha >= 0.0)


def test_family_a_zero_is_identity():
    model = MLP(1, 1, 1, "sin_family", activation_parameter=0.0)
    values = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
    assert torch.equal(model._activate(values), values)


def test_family_b_one_is_identity():
    model = MLP(1, 1, 1, "pwl_family", activation_parameter=1.0)
    values = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
    assert torch.equal(model._activate(values), values)


def test_family_b_zero_is_relu():
    model = MLP(1, 1, 1, "pwl_family", activation_parameter=0.0)
    values = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
    expected = torch.nn.functional.relu(values)
    assert torch.equal(model._activate(values), expected)


def test_family_b_negative_is_a_v():
    """alpha < 0 folds the negative half-line upward: g(-x) = -alpha*x > 0."""

    model = MLP(1, 1, 1, "pwl_family", activation_parameter=-0.5)
    assert float(model._activate(torch.tensor([-2.0], dtype=torch.float64))) == 1.0
    assert float(model._activate(torch.tensor([2.0], dtype=torch.float64))) == 2.0


def test_parametric_requires_parameter():
    with pytest.raises(ValueError):
        MLP(3, 2, 3, "sin_family")
    with pytest.raises(ValueError):
        MLP(3, 2, 3, "tanh", activation_parameter=1.0)


def test_parameter_does_not_change_initialization():
    """The activation parameter must not perturb weight init RNG consumption.

    If it did, existing recorded runs would no longer be reconstructible.
    """

    torch.manual_seed(7)
    with_parameter = MLP(3, 2, 3, "sin_family", activation_parameter=2.0)
    torch.manual_seed(7)
    plain = MLP(3, 2, 3, "tanh")
    for left, right in zip(with_parameter.parameters(), plain.parameters()):
        assert torch.equal(left, right)


def test_conditions_grid_size():
    assert len(conditions()) == len(FAMILY_A) + len(FAMILY_B) + 4
    assert ThresholdSweepConfig().total_runs() == (
        len(conditions()) * 2 * 4 * 20
    )


def test_condition_monotonic_matches_analytic_thresholds():
    assert condition_monotonic("sin_family", 1.0)
    assert not condition_monotonic("sin_family", 1.05)
    assert condition_monotonic("pwl_family", 0.0)
    assert not condition_monotonic("pwl_family", -0.05)
    assert condition_monotonic("tanh", None)
    assert not condition_monotonic("gelu", None)


def test_training_runs_with_parametric_activation():
    """One short end-to-end run per family, checking bookkeeping not accuracy."""

    train = linked_tori(40, tube_radius=0.2, seed=1)
    evaluation = linked_tori(40, tube_radius=0.2, seed=2)
    for activation, parameter in (("sin_family", 1.5), ("pwl_family", -0.25)):
        result = train_mlp(
            train,
            evaluation,
            hidden_depth=2,
            hidden_width=3,
            activation=activation,  # type: ignore[arg-type]
            config=TrainingConfig(seed=0, max_steps=20),
            activation_parameter=parameter,
        )
        assert result.model.activation_parameter == parameter
        assert 0.0 <= result.final_eval_accuracy <= 1.0
        fraction = inactive_unit_fraction(result)
        assert 0.0 <= fraction <= 1.0


def test_identity_parameter_network_is_affine():
    """a = 0 makes the whole network affine: outputs must be an affine map."""

    model = MLP(3, 3, 3, "sin_family", activation_parameter=0.0)
    inputs = torch.randn(8, 3, dtype=torch.float32)
    base = model(torch.zeros(1, 3))
    # An affine map satisfies f(x + y) - f(0) == (f(x) - f(0)) + (f(y) - f(0)).
    left = model(inputs[:4] + inputs[4:]) - base
    right = (model(inputs[:4]) - base) + (model(inputs[4:]) - base)
    assert torch.allclose(left, right, atol=1e-5)
