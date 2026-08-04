import math

import pytest
import torch

from src.precision import (
    DELTA_TABLE,
    collision_metrics,
    quantize_values,
    saturation_metrics,
    threshold_comparison,
)


@pytest.mark.parametrize("criterion", ["paper", "exact"])
@pytest.mark.parametrize("spec", DELTA_TABLE, ids=lambda spec: f"{spec.format}-{spec.convention}")
def test_threshold_exact_identity_agrees_with_direct_atanh_when_representable(spec, criterion):
    result = threshold_comparison(spec.delta, criterion)
    if result.direct_atanh is None:
        assert 1.0 - result.effective_delta == 1.0
    else:
        assert result.direct_atanh == pytest.approx(result.exact, rel=0.0, abs=2e-12)


@pytest.mark.parametrize("criterion", ["paper", "exact"])
@pytest.mark.parametrize("spec", DELTA_TABLE, ids=lambda spec: f"{spec.format}-{spec.convention}")
def test_threshold_log_form_has_expected_asymptotic_error(spec, criterion):
    result = threshold_comparison(spec.delta, criterion)
    # log(2/d) omits 0.5*log(1-d/2); its exact absolute error is bounded by d/2.
    # Two ULPs cover independent rounding in the two float64 log expressions.
    rounding_allowance = 2.0 * math.ulp(max(abs(result.exact), abs(result.log_asymptotic)))
    assert result.asymptotic_error <= result.effective_delta / 2.0 + rounding_allowance


@pytest.mark.parametrize(
    ("delta", "criterion", "expected"),
    [
        (2.0**-6, "paper", 2.4220935432292956),
        (2.0**-6, "exact", 2.770631772579213),
        (2.0**-4, "paper", 1.7169936022425731),
        (2.0**-4, "exact", 2.0715673631957663),
    ],
)
def test_coarse_fixed_thresholds_are_exact_not_asymptotic(delta, criterion, expected):
    assert threshold_comparison(delta, criterion).exact == pytest.approx(
        expected, rel=0.0, abs=2e-15
    )


@pytest.mark.parametrize("criterion", ["paper", "exact"])
def test_saturation_known_inside_is_exactly_zero(criterion):
    values = torch.zeros((7, 3), dtype=torch.float64)
    metrics = saturation_metrics(values, 2.0**-4, criterion)
    assert metrics["upper_saturation_fraction"] == 0.0
    assert metrics["lower_saturation_fraction"] == 0.0
    assert metrics["total_saturation_fraction"] == 0.0
    assert metrics["per_unit_saturation_max"] == 0.0


@pytest.mark.parametrize("criterion", ["paper", "exact"])
def test_saturation_known_outside_is_exactly_one(criterion):
    values = torch.tensor([[100.0, -100.0], [-100.0, 100.0]], dtype=torch.float64)
    metrics = saturation_metrics(values, 2.0**-53, criterion)
    assert metrics["upper_saturation_fraction"] == 0.5
    assert metrics["lower_saturation_fraction"] == 0.5
    assert metrics["total_saturation_fraction"] == 1.0
    assert metrics["per_unit_saturation_min"] == 1.0
    assert metrics["fraction_units_over_50pct_saturation"] == 1.0


def test_float32_values_are_compared_against_unrounded_float64_threshold():
    threshold = threshold_comparison(2.0**-8, "paper").exact
    nearest_float32 = torch.tensor([[threshold]], dtype=torch.float32)
    assert float(nearest_float32.item()) > threshold
    # A direct float32 tensor/scalar comparison rounds the threshold and is false.
    assert not bool((nearest_float32 > threshold).item())
    metrics = saturation_metrics(nearest_float32, 2.0**-8, "paper")
    assert metrics["upper_saturation_fraction"] == 1.0


def test_per_unit_median_uses_standard_even_sample_interpolation():
    values = torch.tensor(
        [
            [0.0, 10.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 0.0, 10.0],
        ],
        dtype=torch.float64,
    )
    metrics = saturation_metrics(values, 2.0**-4, "paper")
    assert metrics["per_unit_saturation_median"] == 0.5


def test_bfloat16_quantization_matches_torch_cast_exactly():
    values = torch.tensor([0.1, 0.1001, -0.3333, math.pi / 4], dtype=torch.float64)
    actual = quantize_values(values, "bfloat16")
    expected = values.to(torch.bfloat16).to(torch.float64)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, values)


def test_fixed4_has_exactly_16_levels_on_closed_unit_interval():
    dense = torch.linspace(-1.0, 1.0, 10001, dtype=torch.float64)
    quantized = quantize_values(dense, "fixed-4")
    assert torch.unique(quantized).numel() == 16
    assert quantized.min().item() == -0.9375
    assert quantized.max().item() == 0.9375


def test_per_unit_collisions_do_not_pool_scalars_across_units():
    values = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    metrics = collision_metrics(values, "float64")
    assert metrics["per_unit_collision_mean"] == pytest.approx(1.0 / 3.0)
    assert metrics["per_unit_collision_min"] == pytest.approx(1.0 / 3.0)
    assert metrics["per_unit_collision_max"] == pytest.approx(1.0 / 3.0)
    assert metrics["vector_collision_rate"] == 0.0


def test_vector_collision_detects_duplicate_complete_outputs():
    values = torch.tensor([[0.01, 0.01], [0.02, 0.02], [0.8, -0.8]], dtype=torch.float64)
    metrics = collision_metrics(values, "fixed-4")
    assert metrics["vector_collision_rate"] == pytest.approx(1.0 / 3.0)


def test_collision_metrics_are_null_without_real_quantizer():
    metrics = collision_metrics(torch.zeros((5, 2)), None)
    assert all(value is None for value in metrics.values())


def test_collision_metrics_apply_the_models_actual_activation():
    values = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], dtype=torch.float64)
    assert collision_metrics(values, "float64", "relu")["vector_collision_rate"] == 0.5
    assert collision_metrics(values, "float64", "leaky_relu")["vector_collision_rate"] == 0.0
    assert collision_metrics(values, "float64", "tanh")["vector_collision_rate"] == 0.0


def test_fixed_point_collision_rejects_unbounded_activations():
    with pytest.raises(ValueError, match="only for bounded tanh"):
        collision_metrics(torch.zeros((2, 2)), "fixed-4", "relu")


def test_collision_group_purity_and_size_distribution():
    values = torch.tensor(
        [[0.01, 0.01], [0.02, 0.02], [0.80, -0.80], [0.81, -0.81]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 0, 1])
    metrics = collision_metrics(values, "fixed-4", "tanh", labels)
    assert metrics["collision_group_count"] == 2.0
    assert metrics["collision_group_pure_fraction"] == 0.5
    assert metrics["collision_group_size_mean"] == 2.0
    assert metrics["collision_group_size_median"] == 2.0
    assert metrics["collision_group_size_max"] == 2.0
    assert metrics["fraction_inputs_in_collision_groups"] == 1.0
