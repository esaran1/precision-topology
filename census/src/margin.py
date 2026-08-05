"""Between-class margin measured in units of local quantization resolution.

The question this answers is whether class purity of quantized collision groups
is forced by margin: two inputs of different classes can only land in the same
quantization cell if their post-activation representations are separated by
less than the resolution of that cell.  A ratio below one is therefore the
condition under which an impure group becomes possible.

Defining "the quantization step" needs care.  Fixed-point quantizers use a
uniform grid, so the step is a single constant.  IEEE formats do not: ULP
spacing depends on the exponent, so it varies across coordinates and across the
range of a single coordinate.  Rather than collapse that to one number, the
step is evaluated *locally* at the two points being compared, using the widest
spacing encountered along the separating displacement.  That is the
conservative choice: it is the resolution that would have to be exceeded for
the pair to be guaranteed distinguishable in every coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .precision import Activation, quantize_values, quantized_activation


@dataclass(frozen=True)
class MarginMeasurement:
    """Between-class separation of one layer under one quantizer."""

    min_between_class_distance: float
    min_between_class_chebyshev: float
    quantization_step: float
    margin_in_steps: float
    margin_in_steps_chebyshev: float
    below_one: bool
    between_class_collision_pairs: int
    nearest_pair_indices: tuple[int, int]


def local_ulp(values: torch.Tensor, quantizer: str) -> torch.Tensor:
    """Return the local quantization spacing at each element of ``values``.

    For fixed-point quantizers this is the constant grid spacing.  For IEEE
    formats it is the true local ULP, obtained by taking the distance to the
    next representable value at each element.
    """

    source = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if quantizer.startswith("fixed-"):
        bits = int(quantizer.removeprefix("fixed-"))
        spacing = 2.0 / (2**bits)
        return torch.full_like(source, spacing)

    dtypes = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if quantizer not in dtypes:
        raise ValueError(f"unknown quantizer: {quantizer}")
    dtype = dtypes[quantizer]
    # Round to the target format first: ULP is a property of the representable
    # neighbour, so it must be evaluated at the quantized location.
    rounded = source.to(dtype)
    nudged = torch.nextafter(
        rounded, torch.full_like(rounded, float("inf"), dtype=dtype)
    )
    return (nudged.to(torch.float64) - rounded.to(torch.float64)).abs()


def between_class_margin(
    preactivations: torch.Tensor,
    labels: torch.Tensor,
    quantizer: str,
    activation: Activation = "tanh",
) -> MarginMeasurement:
    """Measure the smallest between-class separation relative to resolution.

    Distances are computed on post-activation values, which is the quantity the
    quantizer actually consumes, matching the existing collision metric.
    """

    values = quantized_activation_unquantized(preactivations, activation)
    label_values = torch.as_tensor(labels, dtype=torch.int64, device="cpu").reshape(-1)
    if label_values.shape[0] != values.shape[0]:
        raise ValueError("labels must contain one entry per input")

    classes = torch.unique(label_values)
    if classes.numel() != 2:
        raise ValueError("between-class margin is defined for exactly two classes")
    left = values[label_values == classes[0]]
    right = values[label_values == classes[1]]
    left_index = torch.nonzero(label_values == classes[0], as_tuple=False).reshape(-1)
    right_index = torch.nonzero(label_values == classes[1], as_tuple=False).reshape(-1)

    distances = torch.cdist(left, right, p=2)
    flat = int(torch.argmin(distances).item())
    row, column = divmod(flat, distances.shape[1])
    min_distance = float(distances[row, column].item())
    nearest = (int(left_index[row].item()), int(right_index[column].item()))

    # Chebyshev distance is the decision-relevant one for coordinate-wise
    # quantization: a pair survives as distinct if any single coordinate lands
    # in a different cell, which is governed by the largest coordinate gap.
    differences = (left.unsqueeze(1) - right.unsqueeze(0)).abs()
    chebyshev = differences.amax(dim=2)
    min_chebyshev = float(chebyshev.min().item())

    step = _pair_step(values, nearest, quantizer)
    # Count genuine between-class collisions under this quantizer.
    quantized = quantize_values(values, quantizer)
    collisions = _between_class_collision_pairs(quantized, label_values)

    margin = min_distance / step if step > 0 else float("inf")
    margin_chebyshev = min_chebyshev / step if step > 0 else float("inf")
    return MarginMeasurement(
        min_between_class_distance=min_distance,
        min_between_class_chebyshev=min_chebyshev,
        quantization_step=step,
        margin_in_steps=margin,
        margin_in_steps_chebyshev=margin_chebyshev,
        below_one=margin_chebyshev < 1.0,
        between_class_collision_pairs=collisions,
        nearest_pair_indices=nearest,
    )


def quantized_activation_unquantized(
    preactivations: torch.Tensor, activation: Activation
) -> torch.Tensor:
    """Apply the model activation in float64 without quantizing."""

    values = torch.as_tensor(preactivations, dtype=torch.float64, device="cpu")
    if activation == "tanh":
        return torch.tanh(values)
    if activation == "relu":
        return torch.relu(values)
    if activation == "leaky_relu":
        return torch.nn.functional.leaky_relu(values, negative_slope=0.01)
    raise ValueError(f"unsupported activation: {activation}")


def _pair_step(values: torch.Tensor, pair: tuple[int, int], quantizer: str) -> float:
    """Widest local spacing across the coordinates separating one pair."""

    left = values[pair[0]]
    right = values[pair[1]]
    spacing = torch.maximum(local_ulp(left, quantizer), local_ulp(right, quantizer))
    return float(spacing.max().item())


def _between_class_collision_pairs(
    quantized: torch.Tensor, labels: torch.Tensor
) -> int:
    """Count unordered input pairs of different classes with identical rows."""

    _, inverse, counts = torch.unique(
        quantized, dim=0, return_inverse=True, return_counts=True
    )
    total = 0
    for group_id in torch.nonzero(counts > 1, as_tuple=False).reshape(-1):
        members = labels[inverse == group_id]
        positive = int((members == members.max()).sum().item())
        negative = int(members.numel() - positive)
        if members.unique().numel() > 1:
            total += positive * negative
    return total
