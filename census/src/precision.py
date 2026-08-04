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
    upper = preactivations > threshold
    lower = preactivations < -threshold
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


def quantized_tanh(preactivations: torch.Tensor, quantizer: str) -> torch.Tensor:
    """Evaluate tanh in float64, then apply the specified real quantizer."""

    values = torch.as_tensor(preactivations, dtype=torch.float64, device="cpu")
    return quantize_values(torch.tanh(values), quantizer)


def collision_rate(preactivations: torch.Tensor, quantizer: str | None) -> float | None:
    """Return 1 - distinct quantized tanh outputs / scalar inputs.

    The measure is pooled over all input-unit pairs.  ``None`` denotes a delta
    row without a corresponding real quantizer (the paper's 2^-9 half value).
    """

    if quantizer is None:
        return None
    if preactivations.numel() == 0:
        raise ValueError("preactivations must be non-empty")
    outputs = quantized_tanh(preactivations, quantizer).reshape(-1)
    return 1.0 - float(torch.unique(outputs).numel()) / float(outputs.numel())
