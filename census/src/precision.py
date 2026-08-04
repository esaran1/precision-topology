"""Precision definitions, saturation thresholds, and output quantization.

``delta`` always means unit roundoff (maximum rounding error).  The primary
saturation criterion follows the paper, tanh(x) > 1 - delta.  We also measure
the closer bit-level proxy tanh(x) > 1 - delta/2 separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


Criterion = Literal["paper", "exact"]
Activation = Literal["tanh", "relu", "leaky_relu"]


@dataclass(frozen=True)
class PrecisionSpec:
    """One saturation-threshold row and its corresponding real quantizer, if any."""

    format: str
    mantissa_bits: int | None
    delta: float
    convention: str
    quantizer: str | None
    note: str


# The single authoritative delta table.  Duplicate format names intentionally
# preserve the paper/IEEE discrepancy as separate experimental rows.
DELTA_TABLE: tuple[PrecisionSpec, ...] = (
    PrecisionSpec("float64", 52, 2.0**-53, "IEEE", "float64", "Standard IEEE unit roundoff."),
    PrecisionSpec("float64", 52, 2.0**-54, "paper", "float64", "Paper's stated double value."),
    PrecisionSpec("float32", 23, 2.0**-24, "IEEE", "float32", "Standard IEEE unit roundoff."),
    PrecisionSpec("float16", 10, 2.0**-11, "IEEE", "float16", "Standard IEEE unit roundoff."),
    PrecisionSpec(
        "half",
        None,
        2.0**-9,
        "paper",
        None,
        "Paper's stated half value; no corresponding IEEE format; saturation only.",
    ),
    PrecisionSpec("bfloat16", 7, 2.0**-8, "IEEE", "bfloat16", "Standard bfloat16 unit roundoff."),
    PrecisionSpec("fixed-8", None, 2.0**-8, "simulated", "fixed-8", "Uniform 8-bit quantizer."),
    PrecisionSpec("fixed-6", None, 2.0**-6, "simulated", "fixed-6", "Uniform 6-bit quantizer."),
    PrecisionSpec("fixed-4", None, 2.0**-4, "simulated", "fixed-4", "Uniform 4-bit quantizer."),
)


@dataclass(frozen=True)
class ThresholdComparison:
    """Stable exact threshold and the two requested cross-checks."""

    delta: float
    criterion: Criterion
    effective_delta: float
    exact: float
    direct_atanh: float | None
    log_asymptotic: float

    @property
    def direct_error(self) -> float | None:
        return None if self.direct_atanh is None else abs(self.direct_atanh - self.exact)

    @property
    def asymptotic_error(self) -> float:
        return abs(self.log_asymptotic - self.exact)


def threshold_comparison(delta: float, criterion: Criterion = "paper") -> ThresholdComparison:
    """Compute atanh(1-d) robustly and retain direct/asymptotic checks.

    The identity ``atanh(1-d) = 0.5 * (log(2-d) - log(d))`` is exact and
    remains finite when subtraction makes ``1-d`` round to 1.0.  For the exact
    rounding proxy, ``d`` is ``delta/2``.
    """

    if not 0.0 < delta < 2.0:
        raise ValueError("delta must lie strictly between 0 and 2")
    if criterion not in ("paper", "exact"):
        raise ValueError(f"unknown saturation criterion: {criterion}")

    effective_delta = delta if criterion == "paper" else delta / 2.0
    exact = 0.5 * (math.log(2.0 - effective_delta) - math.log(effective_delta))
    rounded_argument = 1.0 - effective_delta
    direct = (
        math.atanh(rounded_argument)
        if -1.0 < rounded_argument < 1.0
        else None
    )
    asymptotic = 0.5 * math.log(2.0 / effective_delta)
    return ThresholdComparison(
        delta=delta,
        criterion=criterion,
        effective_delta=effective_delta,
        exact=exact,
        direct_atanh=direct,
        log_asymptotic=asymptotic,
    )


def saturation_metrics(
    preactivations: torch.Tensor,
    delta: float,
    criterion: Criterion = "paper",
) -> dict[str, float]:
    """Return pooled and per-unit saturation summaries for a 2-D activation matrix."""

    if preactivations.ndim != 2:
        raise ValueError("preactivations must have shape (inputs, units)")
    if preactivations.numel() == 0 or preactivations.shape[1] == 0:
        raise ValueError("preactivations must be non-empty")

    threshold = threshold_comparison(delta, criterion).exact
    # Preactivations are stored in their genuine training dtype (float32), but
    # comparison must occur in float64. PyTorch otherwise treats the Python
    # scalar as wrapped and rounds the threshold to float32 at the boundary.
    comparison_values = preactivations.to(dtype=torch.float64, device="cpu")
    upper = comparison_values > threshold
    lower = comparison_values < -threshold
    saturated = upper | lower
    per_unit = saturated.to(torch.float64).mean(dim=0)

    return {
        "upper_saturation_fraction": float(upper.to(torch.float64).mean().item()),
        "lower_saturation_fraction": float(lower.to(torch.float64).mean().item()),
        "total_saturation_fraction": float(saturated.to(torch.float64).mean().item()),
        "per_unit_saturation_min": float(per_unit.min().item()),
        "per_unit_saturation_median": float(torch.quantile(per_unit, 0.5).item()),
        "per_unit_saturation_max": float(per_unit.max().item()),
        "fraction_units_over_50pct_saturation": float((per_unit > 0.5).to(torch.float64).mean().item()),
    }


def _fixed_bits(quantizer: str) -> int:
    if not quantizer.startswith("fixed-"):
        raise ValueError(f"not a fixed-point quantizer: {quantizer}")
    try:
        bits = int(quantizer.removeprefix("fixed-"))
    except ValueError as exc:
        raise ValueError(f"invalid fixed-point quantizer: {quantizer}") from exc
    if bits <= 0:
        raise ValueError("fixed-point bit count must be positive")
    return bits


def quantize_values(values: torch.Tensor, quantizer: str) -> torch.Tensor:
    """Quantize values and return float64 reconstruction values on CPU.

    IEEE quantizers use actual PyTorch dtype casts.  Fixed-N is a midpoint
    round-to-nearest uniform quantizer with 2**N levels on [-1, 1], spacing
    2**(-(N-1)), and maximum rounding error delta=2**(-N).
    """

    source = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    ieee_dtypes = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if quantizer in ieee_dtypes:
        return source.to(ieee_dtypes[quantizer]).to(torch.float64)

    bits = _fixed_bits(quantizer)
    levels = 2**bits
    spacing = 2.0 / levels
    indices = torch.floor((source.clamp(-1.0, 1.0) + 1.0) / spacing)
    indices = indices.clamp(0, levels - 1)
    return -1.0 + spacing / 2.0 + indices * spacing


def quantized_activation(
    preactivations: torch.Tensor,
    quantizer: str,
    activation: Activation = "tanh",
) -> torch.Tensor:
    """Apply the model's activation in float64, then its real quantizer."""

    values = torch.as_tensor(preactivations, dtype=torch.float64, device="cpu")
    if activation == "tanh":
        activated = torch.tanh(values)
    elif activation == "relu":
        activated = torch.relu(values)
    elif activation == "leaky_relu":
        activated = torch.nn.functional.leaky_relu(values, negative_slope=0.01)
    else:
        raise ValueError(f"unsupported activation: {activation}")
    if quantizer.startswith("fixed-") and activation != "tanh":
        raise ValueError("fixed-point collision metrics are defined only for bounded tanh outputs")
    return quantize_values(activated, quantizer)


def quantized_tanh(preactivations: torch.Tensor, quantizer: str) -> torch.Tensor:
    """Backward-compatible explicit tanh quantization helper."""

    return quantized_activation(preactivations, quantizer, "tanh")


def collision_metrics(
    preactivations: torch.Tensor,
    quantizer: str | None,
    activation: Activation = "tanh",
    labels: torch.Tensor | None = None,
) -> dict[str, float | None]:
    """Measure scalar collisions per unit and true vector-level collisions.

    No scalar values are pooled across units: doing so makes format cardinality,
    rather than the network, dominate the result.  Vector collisions count
    duplicate complete layer-output rows and are the primary injectivity metric.
    ``None`` values denote a delta row without a real quantizer.
    """

    metric_names = (
        "per_unit_collision_mean",
        "per_unit_collision_std",
        "per_unit_collision_min",
        "per_unit_collision_median",
        "per_unit_collision_max",
        "vector_collision_rate",
        "collision_group_count",
        "collision_group_pure_fraction",
        "collision_group_size_mean",
        "collision_group_size_std",
        "collision_group_size_min",
        "collision_group_size_median",
        "collision_group_size_max",
        "fraction_inputs_in_collision_groups",
    )
    if quantizer is None:
        return {name: None for name in metric_names}
    if preactivations.ndim != 2 or preactivations.shape[0] == 0 or preactivations.shape[1] == 0:
        raise ValueError("preactivations must have non-empty shape (inputs, units)")

    outputs = quantized_activation(preactivations, quantizer, activation)
    n_inputs = outputs.shape[0]
    per_unit = _per_unit_rates_from_outputs(outputs)
    _, inverse, counts = torch.unique(outputs, dim=0, return_inverse=True, return_counts=True)
    vector_rate = 1.0 - float(counts.numel()) / float(n_inputs)
    collision_sizes = counts[counts > 1].to(torch.float64)
    if labels is not None:
        label_values = torch.as_tensor(labels, dtype=torch.int64, device="cpu").reshape(-1)
        if label_values.shape[0] != n_inputs:
            raise ValueError("labels must contain one entry per input")
    else:
        label_values = None

    if collision_sizes.numel() == 0:
        group_metrics: dict[str, float | None] = {
            "collision_group_count": 0.0,
            "collision_group_pure_fraction": None,
            "collision_group_size_mean": None,
            "collision_group_size_std": None,
            "collision_group_size_min": None,
            "collision_group_size_median": None,
            "collision_group_size_max": None,
            "fraction_inputs_in_collision_groups": 0.0,
        }
    else:
        pure_fraction = None
        if label_values is not None:
            collision_group_ids = torch.nonzero(counts > 1, as_tuple=False).reshape(-1)
            pure_groups = sum(
                int(torch.unique(label_values[inverse == group_id]).numel() == 1)
                for group_id in collision_group_ids
            )
            pure_fraction = pure_groups / int(collision_group_ids.numel())
        group_metrics = {
            "collision_group_count": float(collision_sizes.numel()),
            "collision_group_pure_fraction": pure_fraction,
            "collision_group_size_mean": float(collision_sizes.mean().item()),
            "collision_group_size_std": float(collision_sizes.std(correction=0).item()),
            "collision_group_size_min": float(collision_sizes.min().item()),
            "collision_group_size_median": float(torch.quantile(collision_sizes, 0.5).item()),
            "collision_group_size_max": float(collision_sizes.max().item()),
            "fraction_inputs_in_collision_groups": float(collision_sizes.sum().item()) / n_inputs,
        }
    return {
        "per_unit_collision_mean": float(per_unit.mean().item()),
        "per_unit_collision_std": float(per_unit.std(correction=0).item()),
        "per_unit_collision_min": float(per_unit.min().item()),
        "per_unit_collision_median": float(torch.quantile(per_unit, 0.5).item()),
        "per_unit_collision_max": float(per_unit.max().item()),
        "vector_collision_rate": vector_rate,
        **group_metrics,
    }


def _per_unit_rates_from_outputs(outputs: torch.Tensor) -> torch.Tensor:
    n_inputs = outputs.shape[0]
    return torch.tensor(
        [
            1.0 - float(torch.unique(outputs[:, unit]).numel()) / float(n_inputs)
            for unit in range(outputs.shape[1])
        ],
        dtype=torch.float64,
    )


def per_unit_collision_rates(
    preactivations: torch.Tensor,
    quantizer: str | None,
    activation: Activation = "tanh",
) -> torch.Tensor | None:
    """Return one scalar collision rate per unit, preserving unit identity."""

    if quantizer is None:
        return None
    if preactivations.ndim != 2 or preactivations.shape[0] == 0 or preactivations.shape[1] == 0:
        raise ValueError("preactivations must have non-empty shape (inputs, units)")
    return _per_unit_rates_from_outputs(quantized_activation(preactivations, quantizer, activation))
