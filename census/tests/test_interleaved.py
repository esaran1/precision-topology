"""Tests for the F / QF / G collision-regime detector.

The positive control is the load-bearing test here.  A zero result from a
detector that can never fire is indistinguishable from a zero result that
means something, so a case where a pair provably collides in F but not in G is
constructed explicitly and the detector is required to find it.
"""

from __future__ import annotations

import pytest
import torch

from src.interleaved import (
    compare_regimes,
    forward_full_precision,
    forward_interleaved,
    partition_of,
)
from src.models import MLP


def _network(
    weights: list[tuple[torch.Tensor, torch.Tensor]],
    output: tuple[torch.Tensor, torch.Tensor],
    activation: str,
) -> MLP:
    """Build an MLP with hand-set parameters."""

    input_dim = weights[0][0].shape[1]
    width = weights[0][0].shape[0]
    model = MLP(input_dim, len(weights), width, activation)  # type: ignore[arg-type]
    with torch.no_grad():
        for layer, (weight, bias) in zip(model.hidden_layers, weights):
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        model.output_layer.weight.copy_(output[0])
        model.output_layer.bias.copy_(output[1])
    return model.to(torch.float32).eval()


def _control_network() -> MLP:
    """A network where one pair provably collides in F but separates in G.

    Construction.  Layer 1 passes the input through unchanged on unit 0.  Layer
    2 computes ``8 * u0 - 3.2`` and applies ReLU, so it clamps to exactly 0.0
    whenever ``u0 < 0.4``.

    In full precision the two control inputs are 0.36 and 0.39.  Both are below
    0.4, so ReLU clamps both to exactly 0.0 and the pair collides in F.

    Under interleaved quantization the layer-1 activations are first rounded to
    the fixed-4 grid (step 0.125, cell centres ..., 0.3125, 0.4375, ...).  The
    cell boundary at 0.375 falls between the two inputs, so 0.36 rounds down to
    0.3125 and 0.39 rounds up to 0.4375 -- landing on opposite sides of the
    0.4 ReLU threshold.  Layer 2 then gives ``8*0.3125-3.2 = -0.7`` (clamped to
    0) against ``8*0.4375-3.2 = +0.3``.  The gain of 8 is what makes the
    surviving difference exceed one quantization cell, so the separation is not
    rounded away again at the next quantization step.
    """

    weight_1 = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    bias_1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    weight_2 = torch.tensor([[8.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    bias_2 = torch.tensor([-3.2, 0.0], dtype=torch.float32)
    output = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        torch.tensor([0.0, 0.0], dtype=torch.float32),
    )
    return _network([(weight_1, bias_1), (weight_2, bias_2)], output, "relu")


CONTROL_INPUTS = torch.tensor([[0.36], [0.39]], dtype=torch.float64)


def test_positive_control_pair_collides_in_f_but_not_in_g():
    """Mandatory positive control: the detector must find a real F-not-G pair.

    Without this, a zero result on real weights would be indistinguishable from
    a detector that can never fire.
    """

    model = _control_network()
    labels = torch.tensor([0, 1])

    full = forward_full_precision(model, CONTROL_INPUTS)
    assert torch.equal(full[0], full[1]), "control requires an exact F collision"

    interleaved, _ = forward_interleaved(model, CONTROL_INPUTS, "fixed-4")
    assert not torch.equal(
        interleaved[0], interleaved[1]
    ), "control requires the pair to separate under G"

    comparison = compare_regimes(model, CONTROL_INPUTS, labels, "fixed-4")
    # The detector must report exactly this pair as an F-not-G violation.
    assert comparison.full_precision_pairs == 1
    assert comparison.interleaved_superset_holds is False
    assert comparison.interleaved_violations == ((0, 1),)
    assert comparison.between_class_violations == 1
    assert comparison.within_class_violations == 0
    # The trajectories first differ at the quantized layer-1 activation.
    assert comparison.divergence_layers == (1,)
    # QF must still hold: it is a deterministic function of the F output.
    assert comparison.post_hoc_superset_holds is True


def test_positive_control_within_class_variant():
    """The detector distinguishes within-class from between-class violations."""

    model = _control_network()
    comparison = compare_regimes(
        model, CONTROL_INPUTS, torch.tensor([1, 1]), "fixed-4"
    )
    assert comparison.interleaved_violations == ((0, 1),)
    assert comparison.within_class_violations == 1
    assert comparison.between_class_violations == 0


def test_positive_control_is_specific_to_interleaving():
    """The same pair must remain collided under QF, which quantizes once.

    This separates the two regimes: the F-not-G violation is caused by
    quantization inside the network, not by quantization as such.
    """

    model = _control_network()
    full = forward_full_precision(model, CONTROL_INPUTS)
    from src.precision import quantize_values

    post_hoc = quantize_values(full, "fixed-4")
    assert torch.equal(post_hoc[0], post_hoc[1])


def test_post_hoc_superset_holds_on_random_networks():
    """QF containment is determinism, so it must hold for every random case."""

    for seed in range(8):
        torch.manual_seed(seed)
        model = MLP(3, 4, 6, "tanh").to(torch.float32).eval()
        inputs = torch.randn(64, 3, dtype=torch.float64)
        labels = torch.randint(0, 2, (64,))
        for quantizer in ("bfloat16", "float16", "fixed-4", "fixed-6"):
            comparison = compare_regimes(model, inputs, labels, quantizer)
            assert comparison.post_hoc_superset_holds, (
                f"QF containment failed at seed {seed} quantizer {quantizer}: "
                f"{comparison.post_hoc_violations}"
            )


def test_quantizing_output_only_never_splits_a_collision():
    """Directly: equal vectors stay equal after a deterministic map."""

    values = torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    full = partition_of(values)
    from src.precision import quantize_values

    quantized = partition_of(quantize_values(values, "fixed-4"))
    assert full.colliding_pairs() <= quantized.colliding_pairs()


def test_partition_groups_and_pairs():
    values = torch.tensor(
        [[0.0], [0.0], [1.0], [0.0], [2.0]], dtype=torch.float64
    )
    partition = partition_of(values)
    groups = sorted(sorted(group) for group in partition.groups())
    assert groups == [[0, 1, 3]]
    assert partition.colliding_pairs() == {(0, 1), (0, 3), (1, 3)}


def test_partition_with_no_collisions_is_empty():
    values = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    partition = partition_of(values)
    assert partition.colliding_pairs() == set()
    assert list(partition.groups()) == []


def test_interleaved_forward_quantizes_every_layer():
    """Every traced activation must be exactly representable in the format."""

    torch.manual_seed(0)
    model = MLP(3, 3, 5, "tanh").to(torch.float32).eval()
    inputs = torch.randn(32, 3, dtype=torch.float64)
    from src.precision import quantize_values

    _, trace = forward_interleaved(model, inputs, "fixed-4")
    assert len(trace) == 3
    for activations in trace:
        assert torch.equal(activations, quantize_values(activations, "fixed-4"))


def test_full_precision_forward_matches_model_forward():
    """Regime F in float64 must agree with the model's own float32 forward."""

    torch.manual_seed(0)
    model = MLP(3, 4, 8, "tanh").to(torch.float32).eval()
    inputs = torch.randn(16, 3, dtype=torch.float64)
    with torch.no_grad():
        reference = model(inputs.to(torch.float32))
    computed = forward_full_precision(model, inputs)
    assert torch.allclose(computed, reference.to(torch.float64), atol=1e-5)


def test_unsupported_activation_is_rejected():
    torch.manual_seed(0)
    model = MLP(3, 2, 4, "tanh").to(torch.float32).eval()
    model.activation_name = "sigmoid"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        forward_full_precision(model, torch.randn(4, 3, dtype=torch.float64))
