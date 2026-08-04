import math

import pytest
import torch

from src.precision import (
    DELTA_TABLE,
    collision_rate,
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


def test_collision_rate_counts_repeated_quantized_outputs():
    values = torch.zeros((5, 2), dtype=torch.float64)
    assert collision_rate(values, "float64") == pytest.approx(0.9)
    assert collision_rate(values, None) is None
