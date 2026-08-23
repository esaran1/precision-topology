"""The F.2 construction pipeline, generalized and precondition-checked.

Recipe (Appendix F.2): find a strict local extremum of the activation,
shrink the fold coordinate into its neighbourhood with a frozen affine
map, apply the activation, amplify the folded coordinate with a second
frozen affine map, and let a trained monotone continuation finish.

Step 1 is the precondition: ``fold_point`` locates a strict local
extremum by finding a sign change of f' and verifying strictness.  For
continuous coordinate-wise monotonic activations this step FAILS — f'
never changes sign — and the failure is raised as ``PipelineFailure``
naming the reason.  The pipeline is not a general solution finder; it is
a recipe whose precondition monotonic activations provably do not meet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .census import SweepConfig, _make_data
from .data import linked_tori
from .train import seed_everything


class PipelineFailure(RuntimeError):
    """The construction cannot proceed, with the failing step named."""


@dataclass(frozen=True)
class FoldPoint:
    location: float          # t*: the strict local extremum of f
    kind: str                # "max" or "min"
    window: tuple[float, float]  # interval around t* where f is 2-to-1


def fold_point(
    f: Callable[[np.ndarray], np.ndarray],
    search_interval: tuple[float, float] = (-8.0, 8.0),
    resolution: int = 200_001,
    slope_floor: float = 1e-6,
) -> FoldPoint:
    """Step 1: locate a strict local extremum via a sign change of f'.

    ``slope_floor`` guards against numerical-underflow "extrema": both
    branches around the crossing must carry real slope, or the fold is
    degenerate at working precision (GELU's tail produces sign flicker in
    f' at magnitudes ~1e-13 near x = -7.6, which is not a usable fold).
    """

    xs = np.linspace(*search_interval, resolution)
    values = f(xs)
    derivative = np.gradient(values, xs)
    signs = np.sign(derivative)
    changes = np.nonzero(np.diff(signs) != 0)[0]
    step = (search_interval[1] - search_interval[0]) / (resolution - 1)
    halo = max(3, int(round(0.25 / step)))
    genuine = []
    for i in changes:
        left = derivative[max(i - halo, 0): i + 1]
        right = derivative[i + 1: min(i + halo + 1, resolution)]
        if left.size == 0 or right.size == 0:
            continue
        # Sign test on immediately adjacent samples (a fixed halo overshoots
        # narrow folds); slope test only over samples belonging to each
        # branch, i.e. matching that side's sign.
        left_sign = np.sign(left[-1]) if left[-1] != 0 else np.sign(left[left != 0][-1]) if (left != 0).any() else 0.0
        right_sign = np.sign(right[0]) if right[0] != 0 else np.sign(right[right != 0][0]) if (right != 0).any() else 0.0
        if left_sign * right_sign >= 0:
            continue
        left_branch = left[np.sign(left) == left_sign]
        right_branch = right[np.sign(right) == right_sign]
        if abs(left_branch).max() < slope_floor or abs(right_branch).max() < slope_floor:
            continue
        genuine.append(i)
    if not genuine:
        raise PipelineFailure(
            "step 1 (precondition): f' has no sign change with usable slope on "
            f"{search_interval} — no strict local extremum exists at working "
            "precision; the activation is monotonic there and the recipe "
            "cannot start"
        )
    index = genuine[0]
    location = float(xs[index])
    kind = "max" if derivative[max(index - halo, 0)] > 0 else "min"
    return FoldPoint(location=location, kind=kind, window=(location - 1.0, location + 1.0))


class ConstructedFold(nn.Module):
    """Frozen shrink -> activation -> frozen amplify; trained tanh continuation."""

    def __init__(
        self,
        activation: Callable[[torch.Tensor], torch.Tensor],
        t_star: float,
        shrink: float,
        amplification: float,
        yz_rescale: float,
        sign: float = 1.0,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.t_star = t_star
        self.shrink = shrink
        self.amplification = amplification
        self.yz_rescale = yz_rescale
        self.sign = sign
        self.hidden = nn.ModuleList(nn.Linear(3, 3) for _ in range(depth))
        self.head = nn.Linear(3, 2)

    def frozen_representation(self, points: torch.Tensor) -> torch.Tensor:
        f_t_star = self.activation(torch.tensor(self.t_star)).item()
        z1 = self.activation(self.t_star + self.shrink * (points[:, 0] - 1.0))
        z2 = self.activation(self.shrink * points[:, 1])
        z3 = self.activation(self.shrink * points[:, 2])
        f_zero = self.activation(torch.tensor(0.0)).item()
        return torch.stack(
            [
                self.sign * self.amplification * (z1 - f_t_star),
                self.yz_rescale * (z2 - f_zero),
                self.yz_rescale * (z3 - f_zero),
            ],
            dim=1,
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        hidden = self.frozen_representation(points)
        for layer in self.hidden:
            hidden = torch.tanh(layer(hidden))
        return self.head(hidden)


def construct_and_train(
    activation: Callable[[torch.Tensor], torch.Tensor],
    numpy_activation: Callable[[np.ndarray], np.ndarray],
    shrink: float,
    amplification: float,
    yz_slope: float,
    seeds: range,
    steps: int = 3_000,
    dense_check_per_class: int = 50_000,
) -> dict:
    """Run the full pipeline; returns outcome including any dense separation.

    ``yz_slope`` is the activation's slope near 0, used to undo its action
    on the pass-through coordinates.
    """

    point = fold_point(numpy_activation)  # raises PipelineFailure if monotonic
    config = SweepConfig(
        n_train_per_class=1_000, n_eval_per_class=1_000,
        max_steps=steps, learning_rate=1e-2, tube_radius=0.2,
    )
    sign = 1.0 if point.kind == "max" else -1.0
    outcomes = []
    for seed in seeds:
        train_data, eval_data, *_ = _make_data("linked_tori", seed, config)
        tf = torch.as_tensor(train_data.features); tl = torch.as_tensor(train_data.labels)
        ef = torch.as_tensor(eval_data.features); el = torch.as_tensor(eval_data.labels)
        seed_everything(seed)
        model = ConstructedFold(
            activation, point.location, shrink, amplification,
            yz_rescale=1.0 / (shrink * yz_slope), sign=sign,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(tf), tl).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            errors = int((model(ef).argmax(1) != el).sum().item())
        dense_errors = None
        if errors == 0:
            dense = linked_tori(dense_check_per_class, tube_radius=0.2, seed=960_000 + seed)
            df = torch.as_tensor(dense.features); dl = torch.as_tensor(dense.labels)
            with torch.no_grad():
                dense_errors = int((model(df).argmax(1) != dl).sum().item())
        outcomes.append({"seed": seed, "eval_errors": errors, "dense_errors": dense_errors,
                         "model": model if errors == 0 and dense_errors == 0 else None})
    return {"fold_point": point, "outcomes": outcomes}
