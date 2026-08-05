"""Collision structure under post-hoc versus interleaved quantization.

Three forward regimes are compared on the same inputs:

``F``
    Full precision throughout, no quantization anywhere.
``QF``
    Full precision forward pass, quantized once at the output.
``G``
    Quantization applied to the activations at every layer, so each layer
    consumes a quantized representation.

``QF`` is a deterministic function of the ``F`` output, so ``collisions(F)``
must be contained in ``collisions(QF)``: equal inputs to a function give equal
outputs.  That containment is a correctness check on this module rather than an
empirical question, and a violation is a bug.

``G`` is not a function of the ``F`` output.  Quantization inside the network
changes what later layers receive, so two inputs that coincide by the final
layer in full precision may be perturbed onto trajectories that never meet.
Whether ``collisions(F)`` is contained in ``collisions(G)`` is therefore an
open empirical question, and is what this module measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .models import MLP
from .precision import Activation, quantize_values


@dataclass(frozen=True)
class Partition:
    """Collision structure of one set of output vectors."""

    group_ids: torch.Tensor
    group_count: int

    @property
    def n_inputs(self) -> int:
        return int(self.group_ids.numel())

    def colliding_pairs(self) -> set[tuple[int, int]]:
        """Unordered index pairs sharing an output vector.

        Groups are materialized one at a time; only colliding groups contribute,
        so this stays far below the full n^2 pair count in practice.
        """

        pairs: set[tuple[int, int]] = set()
        for members in self.groups():
            for position, left in enumerate(members):
                for right in members[position + 1 :]:
                    pairs.add((left, right))
        return pairs

    def groups(self) -> Iterable[list[int]]:
        """Yield the member indices of each group of size at least two."""

        order = torch.argsort(self.group_ids, stable=True)
        sorted_ids = self.group_ids[order]
        boundaries = torch.nonzero(
            torch.diff(sorted_ids, prepend=sorted_ids[:1] - 1), as_tuple=False
        ).reshape(-1)
        limits = list(boundaries.tolist()) + [int(sorted_ids.numel())]
        for start, stop in zip(limits[:-1], limits[1:]):
            if stop - start > 1:
                yield [int(index) for index in order[start:stop].tolist()]


def partition_of(outputs: torch.Tensor) -> Partition:
    """Partition rows into groups of exactly equal vectors."""

    values = torch.as_tensor(outputs, dtype=torch.float64, device="cpu")
    unique, inverse = torch.unique(values, dim=0, return_inverse=True)
    return Partition(group_ids=inverse, group_count=int(unique.shape[0]))


def _activate(values: torch.Tensor, activation: Activation) -> torch.Tensor:
    if activation == "tanh":
        return torch.tanh(values)
    if activation == "relu":
        return torch.relu(values)
    if activation == "leaky_relu":
        return torch.nn.functional.leaky_relu(values, negative_slope=0.01)
    raise ValueError(f"unsupported activation: {activation}")


@torch.no_grad()
def forward_full_precision(model: MLP, inputs: torch.Tensor) -> torch.Tensor:
    """Regime F: float64 forward pass with no quantization."""

    hidden = torch.as_tensor(inputs, dtype=torch.float64, device="cpu")
    for layer in model.hidden_layers:
        preactivation = torch.nn.functional.linear(
            hidden,
            layer.weight.to(torch.float64),
            layer.bias.to(torch.float64),
        )
        hidden = _activate(preactivation, model.activation_name)
    return torch.nn.functional.linear(
        hidden,
        model.output_layer.weight.to(torch.float64),
        model.output_layer.bias.to(torch.float64),
    )


@torch.no_grad()
def forward_interleaved(
    model: MLP,
    inputs: torch.Tensor,
    quantizer: str,
    quantize_input: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Regime G: quantize the activations entering every layer.

    Returns the output and the per-layer quantized activations, so the layer at
    which two trajectories first diverge can be located.
    """

    hidden = torch.as_tensor(inputs, dtype=torch.float64, device="cpu")
    if quantize_input:
        hidden = quantize_values(hidden, quantizer)
    trace: list[torch.Tensor] = []
    for layer in model.hidden_layers:
        preactivation = torch.nn.functional.linear(
            hidden,
            layer.weight.to(torch.float64),
            layer.bias.to(torch.float64),
        )
        hidden = quantize_values(_activate(preactivation, model.activation_name), quantizer)
        trace.append(hidden.clone())
    output = torch.nn.functional.linear(
        hidden,
        model.output_layer.weight.to(torch.float64),
        model.output_layer.bias.to(torch.float64),
    )
    return quantize_values(output, quantizer), trace


@dataclass(frozen=True)
class RegimeComparison:
    """Set relations between the collision partitions of F, QF, and G."""

    quantizer: str
    n_inputs: int
    full_precision_pairs: int
    post_hoc_pairs: int
    interleaved_pairs: int
    post_hoc_superset_holds: bool
    post_hoc_violations: tuple[tuple[int, int], ...]
    interleaved_superset_holds: bool
    interleaved_violations: tuple[tuple[int, int], ...]
    within_class_violations: int
    between_class_violations: int
    divergence_layers: tuple[int, ...]


@torch.no_grad()
def final_hidden_full_precision(model: MLP, inputs: torch.Tensor) -> torch.Tensor:
    """Full-precision activation of the final hidden layer.

    This is the representation the census collision metric is computed on, and
    unlike the two output logits it is where quantized collisions are dense.
    """

    hidden = torch.as_tensor(inputs, dtype=torch.float64, device="cpu")
    for layer in model.hidden_layers:
        preactivation = torch.nn.functional.linear(
            hidden,
            layer.weight.to(torch.float64),
            layer.bias.to(torch.float64),
        )
        hidden = _activate(preactivation, model.activation_name)
    return hidden


@dataclass(frozen=True)
class LayerRegimeComparison:
    """QF versus G collision structure at the final hidden layer.

    ``collisions(F)`` is empty at this observation point in trained networks --
    exact float64 equality of a real-valued activation vector essentially never
    occurs -- so the containment question is only non-trivial when the
    comparison is made at a fixed quantization level.  ``QF`` quantizes the
    full-precision activation once; ``G`` reaches the same layer through a
    network that quantized every preceding activation.
    """

    quantizer: str
    post_hoc_pairs: int
    interleaved_pairs: int
    post_hoc_not_interleaved: int
    interleaved_not_post_hoc: int
    post_hoc_not_interleaved_within: int
    post_hoc_not_interleaved_between: int
    interleaved_not_post_hoc_within: int
    interleaved_not_post_hoc_between: int
    containment_holds: bool


def compare_final_layer(
    model: MLP,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    quantizer: str,
) -> LayerRegimeComparison:
    """Compare QF and G collision partitions at the final hidden layer."""

    full = final_hidden_full_precision(model, inputs)
    post_hoc = partition_of(quantize_values(full, quantizer)).colliding_pairs()
    _, trace = forward_interleaved(model, inputs, quantizer)
    interleaved = partition_of(trace[-1]).colliding_pairs()

    label_values = torch.as_tensor(labels, dtype=torch.int64, device="cpu").reshape(-1)

    def split(pairs: set[tuple[int, int]]) -> tuple[int, int]:
        within = sum(
            1 for left, right in pairs if label_values[left] == label_values[right]
        )
        return within, len(pairs) - within

    post_hoc_only = post_hoc - interleaved
    interleaved_only = interleaved - post_hoc
    post_within, post_between = split(post_hoc_only)
    inter_within, inter_between = split(interleaved_only)

    return LayerRegimeComparison(
        quantizer=quantizer,
        post_hoc_pairs=len(post_hoc),
        interleaved_pairs=len(interleaved),
        post_hoc_not_interleaved=len(post_hoc_only),
        interleaved_not_post_hoc=len(interleaved_only),
        post_hoc_not_interleaved_within=post_within,
        post_hoc_not_interleaved_between=post_between,
        interleaved_not_post_hoc_within=inter_within,
        interleaved_not_post_hoc_between=inter_between,
        containment_holds=not post_hoc_only,
    )


def compare_regimes(
    model: MLP,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    quantizer: str,
) -> RegimeComparison:
    """Compare collision partitions of F, QF, and G on the same inputs."""

    full = forward_full_precision(model, inputs)
    post_hoc = quantize_values(full, quantizer)
    interleaved, trace = forward_interleaved(model, inputs, quantizer)

    full_partition = partition_of(full)
    post_hoc_partition = partition_of(post_hoc)
    interleaved_partition = partition_of(interleaved)

    full_pairs = full_partition.colliding_pairs()
    post_hoc_pairs = post_hoc_partition.colliding_pairs()
    interleaved_pairs = interleaved_partition.colliding_pairs()

    post_hoc_missing = tuple(sorted(full_pairs - post_hoc_pairs))
    interleaved_missing = tuple(sorted(full_pairs - interleaved_pairs))

    label_values = torch.as_tensor(labels, dtype=torch.int64, device="cpu").reshape(-1)
    within = sum(
        1 for left, right in interleaved_missing if label_values[left] == label_values[right]
    )
    between = len(interleaved_missing) - within

    divergence = tuple(
        _first_divergence_layer(trace, left, right) for left, right in interleaved_missing
    )

    return RegimeComparison(
        quantizer=quantizer,
        n_inputs=int(full.shape[0]),
        full_precision_pairs=len(full_pairs),
        post_hoc_pairs=len(post_hoc_pairs),
        interleaved_pairs=len(interleaved_pairs),
        post_hoc_superset_holds=not post_hoc_missing,
        post_hoc_violations=post_hoc_missing,
        interleaved_superset_holds=not interleaved_missing,
        interleaved_violations=interleaved_missing,
        within_class_violations=within,
        between_class_violations=between,
        divergence_layers=divergence,
    )


def _first_divergence_layer(
    trace: Sequence[torch.Tensor], left: int, right: int
) -> int:
    """Index (1-based) of the first quantized layer where two inputs differ."""

    for layer_index, activations in enumerate(trace, start=1):
        if not torch.equal(activations[left], activations[right]):
            return layer_index
    return 0
