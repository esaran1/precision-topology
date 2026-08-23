"""Plain CPU MLPs with explicit pre-activation capture."""

from __future__ import annotations

import math
from typing import Literal, overload

import torch
from torch import nn
from torch.nn import functional as F


ActivationName = Literal["tanh", "relu", "leaky_relu", "gelu", "sin_family", "pwl_family"]

# tanh, ReLU, and leaky-ReLU are all continuous and coordinate-wise monotonic,
# so under Ren and Lim they occupy a single expressivity class; GELU is
# non-monotonic and is the only activation here that can bear on that ordering.
MONOTONIC_ACTIVATIONS = ("tanh", "relu", "leaky_relu")
NONMONOTONIC_ACTIVATIONS = ("gelu",)

# Continuously parametrized families with analytically known monotonicity
# thresholds, for the threshold sweep (results/threshold_prediction.md):
#   sin_family: f_a(x) = x + a*sin(x); f' = 1 + a*cos(x) >= 0 iff a <= 1.
#   pwl_family: g_alpha(x) = x for x >= 0, alpha*x for x < 0; monotonic iff
#               alpha >= 0 (alpha = 0 is ReLU, alpha = 1 is the identity).
PARAMETRIC_ACTIVATIONS = ("sin_family", "pwl_family")


def parametric_monotonic(activation: str, parameter: float) -> bool:
    """Analytic monotonicity of a parametric activation, no fitting involved."""

    if activation == "sin_family":
        return parameter <= 1.0
    if activation == "pwl_family":
        return parameter >= 0.0
    raise ValueError(f"not a parametric activation: {activation}")

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
        activation_parameter: float | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if hidden_depth <= 0 or hidden_width <= 0:
            raise ValueError("hidden_depth and hidden_width must be positive")
        if activation not in (
            MONOTONIC_ACTIVATIONS + NONMONOTONIC_ACTIVATIONS + PARAMETRIC_ACTIVATIONS
        ):
            raise ValueError(f"unsupported activation: {activation}")
        if activation in PARAMETRIC_ACTIVATIONS:
            if activation_parameter is None:
                raise ValueError(f"{activation} requires activation_parameter")
        elif activation_parameter is not None:
            raise ValueError(f"{activation} takes no activation_parameter")

        dimensions = [input_dim] + [hidden_width] * hidden_depth
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(hidden_depth)
        )
        self.output_layer = nn.Linear(hidden_width, output_dim)
        self.hidden_depth = hidden_depth
        self.hidden_width = hidden_width
        self.activation_name: ActivationName = activation
        self.activation_parameter = activation_parameter
        self.initialization_scheme = DEFAULT_INITIALIZATION_SCHEME
        self.initialization_gain = DEFAULT_INITIALIZATION_GAIN

    def _activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "tanh":
            return torch.tanh(values)
        if self.activation_name == "relu":
            return F.relu(values)
        if self.activation_name == "gelu":
            return F.gelu(values)
        if self.activation_name == "sin_family":
            return values + self.activation_parameter * torch.sin(values)
        if self.activation_name == "pwl_family":
            return torch.where(
                values >= 0.0, values, self.activation_parameter * values
            )
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
