"""Plain CPU MLPs with explicit pre-activation capture."""

from __future__ import annotations

import math
from typing import Literal, overload

import torch
from torch import nn
from torch.nn import functional as F


ActivationName = Literal["tanh", "relu", "leaky_relu"]

# nn.Linear.reset_parameters calls kaiming_uniform_(a=sqrt(5)) for weights and
# uniform_(-1/sqrt(fan_in), 1/sqrt(fan_in)) for biases. No reset or rescaling is
# applied after module construction in the main experiment.
DEFAULT_INITIALIZATION_SCHEME = (
    "torch.nn.Linear.reset_parameters: kaiming_uniform_(a=sqrt(5)); "
    "bias_uniform_bound=1/sqrt(fan_in)"
)
DEFAULT_INITIALIZATION_GAIN = float(torch.nn.init.calculate_gain("leaky_relu", math.sqrt(5.0)))


class MLP(nn.Module):
    """Constant-width MLP with a two-logit classifier head."""

    def __init__(
        self,
        input_dim: int,
        hidden_depth: int,
        hidden_width: int,
        activation: ActivationName,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if hidden_depth <= 0 or hidden_width <= 0:
            raise ValueError("hidden_depth and hidden_width must be positive")
        if activation not in ("tanh", "relu", "leaky_relu"):
            raise ValueError(f"unsupported activation: {activation}")

        dimensions = [input_dim] + [hidden_width] * hidden_depth
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(hidden_depth)
        )
        self.output_layer = nn.Linear(hidden_width, output_dim)
        self.hidden_depth = hidden_depth
        self.hidden_width = hidden_width
        self.activation_name: ActivationName = activation
        self.initialization_scheme = DEFAULT_INITIALIZATION_SCHEME
        self.initialization_gain = DEFAULT_INITIALIZATION_GAIN

    def _activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "tanh":
            return torch.tanh(values)
        if self.activation_name == "relu":
            return F.relu(values)
        return F.leaky_relu(values, negative_slope=0.01)

    @overload
    def forward(self, inputs: torch.Tensor, return_preactivations: Literal[False] = False) -> torch.Tensor: ...

    @overload
    def forward(
        self, inputs: torch.Tensor, return_preactivations: Literal[True]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]: ...

    def forward(
        self, inputs: torch.Tensor, return_preactivations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = inputs
        preactivations: list[torch.Tensor] = []
        for layer in self.hidden_layers:
            preactivation = layer(hidden)
            preactivations.append(preactivation)
            hidden = self._activate(preactivation)
        logits = self.output_layer(hidden)
        if return_preactivations:
            return logits, preactivations
        return logits

    @torch.no_grad()
    def collect_preactivations(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Return detached CPU inputs to each hidden activation function."""

        was_training = self.training
        self.eval()
        _, values = self(inputs.to(device="cpu", dtype=torch.float32), return_preactivations=True)
        if was_training:
            self.train()
        return [value.detach().to(device="cpu").clone() for value in values]
