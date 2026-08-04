import math

import pytest
import torch

from src.models import DEFAULT_INITIALIZATION_GAIN, MLP


def test_stored_preactivations_match_hand_computed_forward_pass():
    model = MLP(input_dim=3, hidden_depth=2, hidden_width=2, activation="tanh")
    with torch.no_grad():
        model.hidden_layers[0].weight.copy_(
            torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]])
        )
        model.hidden_layers[0].bias.copy_(torch.tensor([0.25, -0.5]))
        model.hidden_layers[1].weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 2.0]]))
        model.hidden_layers[1].bias.copy_(torch.tensor([0.1, -0.2]))
        model.output_layer.weight.zero_()
        model.output_layer.bias.zero_()

    inputs = torch.tensor([[2.0, -1.0, 0.5]])
    stored = model.collect_preactivations(inputs)
    first_expected = torch.tensor([[-0.25, -3.5]])
    first_post = torch.tensor([[math.tanh(-0.25), math.tanh(-3.5)]])
    second_expected = torch.tensor(
        [[first_post[0, 0] - first_post[0, 1] + 0.1,
          0.5 * first_post[0, 0] + 2.0 * first_post[0, 1] - 0.2]]
    )

    assert len(stored) == 2
    assert torch.equal(stored[0], first_expected)
    assert torch.allclose(stored[1], second_expected, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("activation", ["tanh", "relu", "leaky_relu"])
def test_all_activations_return_one_preactivation_per_hidden_layer(activation):
    model = MLP(3, hidden_depth=4, hidden_width=5, activation=activation)
    logits, preactivations = model(torch.zeros((7, 3)), return_preactivations=True)
    assert logits.shape == (7, 2)
    assert len(preactivations) == 4
    assert all(values.shape == (7, 5) for values in preactivations)


def test_default_initialization_is_seed_reproducible_and_metadata_is_explicit():
    torch.manual_seed(123)
    first = MLP(3, 2, 5, "tanh")
    torch.manual_seed(123)
    second = MLP(3, 2, 5, "tanh")
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert "kaiming_uniform_(a=sqrt(5))" in first.initialization_scheme
    assert first.initialization_gain == pytest.approx(1.0 / math.sqrt(3.0))
    assert first.initialization_gain == DEFAULT_INITIALIZATION_GAIN
