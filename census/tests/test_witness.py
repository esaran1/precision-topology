"""The witness must be reproducible, separating, and exactly expressible."""

from __future__ import annotations

import torch

from src.census import SweepConfig, _make_data
from src.models import MLP
from src.witness import (
    WITNESS_SEED,
    FoldWitness,
    dense_verification,
    train_witness,
)


def _eval_errors(model: FoldWitness, seed: int) -> int:
    config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=2_000,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    _, eval_data, *_ = _make_data("linked_tori", seed, config)
    features = torch.as_tensor(eval_data.features, dtype=torch.float32)
    labels = torch.as_tensor(eval_data.labels, dtype=torch.int64)
    with torch.no_grad():
        return int((model(features).argmax(dim=1) != labels).sum().item())


def test_witness_separates_and_is_deterministic():
    first = train_witness(WITNESS_SEED)
    assert _eval_errors(first, WITNESS_SEED) == 0
    second = train_witness(WITNESS_SEED)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_witness_dense_sample():
    model = train_witness(WITNESS_SEED)
    result = dense_verification(model, 10_000, 91_000)
    assert result["errors"] == 0
    assert result["min_margin"] > 0.0


def test_fold_is_expressible_as_pwl_layer():
    """(|x-1|, y, z) equals a pwl_family(-1) layer with W=I, b=(-1,4,4),
    minus the constant (0,4,4), on the data's support."""

    layer = MLP(3, 1, 3, "pwl_family", activation_parameter=-1.0)
    with torch.no_grad():
        layer.hidden_layers[0].weight.copy_(torch.eye(3))
        layer.hidden_layers[0].bias.copy_(torch.tensor([-1.0, 4.0, 4.0]))
    inputs = torch.tensor(
        [[0.3, -1.1, 0.9], [2.1, 1.2, -1.15], [1.0, 0.0, 0.0], [-1.2, 0.4, 0.2]]
    )
    _, pre = layer(inputs, return_preactivations=True)
    activated = torch.where(pre[0] >= 0, pre[0], -pre[0])
    expected = FoldWitness().fold(inputs) + torch.tensor([0.0, 4.0, 4.0])
    assert torch.allclose(activated, expected, atol=1e-6)
