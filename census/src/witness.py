"""Part 2a: a semi-analytic width-3 witness for non-monotonic separation.

The witness decomposes the computation into exactly one hand-designed
non-monotonic fold and a purely monotonic continuation:

  layer 1 (fixed, by hand):  (x, y, z) -> (|x - 1|, y, z)
  layers 2..k (trained):     width-3 tanh
  head (trained):            linear, 2 logits

The fold plane ``x = 1`` contains the second component's core circle, so the
fold maps B 2-to-1 onto an arc while A -- which lies in ``x <= 1`` except for
a 0.2-thick tube cap -- is mapped almost affinely.  The claim the witness
supports is structural: one non-injective fold at the right plane is the
entire non-monotonic requirement; everything after it can be monotone.

The fold layer is exactly expressible inside the project's own activation
zoo as a ``pwl_family(-1)`` (absolute value) layer with weights I and bias
(-1, 4, 4): coordinates two and three stay positive on the data, where
``|.|`` is the identity, so only the first coordinate folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import linked_tori
from .census import SweepConfig, _make_data
from .train import seed_everything


FOLD_CENTER = 1.0
CONTINUATION_DEPTH = 2
WITNESS_SEED = 2  # chosen by the selection sweep in results/witness.md


class FoldWitness(nn.Module):
    """Hand-designed fold + monotone tanh continuation + linear head."""

    def __init__(self, depth: int = CONTINUATION_DEPTH) -> None:
        super().__init__()
        self.depth = depth
        self.hidden = nn.ModuleList(nn.Linear(3, 3) for _ in range(depth))
        self.head = nn.Linear(3, 2)

    def fold(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [(inputs[:, 0] - FOLD_CENTER).abs(), inputs[:, 1], inputs[:, 2]],
            dim=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fold(inputs)
        for layer in self.hidden:
            hidden = torch.tanh(layer(hidden))
        return self.head(hidden)

    def intermediate(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Representations after the fold and after each tanh layer."""

        values = [self.fold(inputs)]
        hidden = values[0]
        for layer in self.hidden:
            hidden = torch.tanh(layer(hidden))
            values.append(hidden)
        return values


def train_witness(seed: int = WITNESS_SEED, steps: int = 2_000) -> FoldWitness:
    """Deterministically train the monotone continuation."""

    config = SweepConfig(
        n_train_per_class=1_000,
        n_eval_per_class=1_000,
        max_steps=steps,
        learning_rate=1e-2,
        tube_radius=0.2,
    )
    train_data, _, *_ = _make_data("linked_tori", seed, config)
    features = torch.as_tensor(train_data.features, dtype=torch.float32)
    labels = torch.as_tensor(train_data.labels, dtype=torch.int64)
    seed_everything(seed)
    model = FoldWitness()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(features), labels).backward()
        optimizer.step()
    model.eval()
    return model


def dense_verification(
    model: nn.Module, n_per_class: int, seed: int
) -> dict[str, float | int]:
    """Errors and worst logit margin on a large fresh sample."""

    data = linked_tori(n_per_class, tube_radius=0.2, seed=seed)
    features = torch.as_tensor(data.features, dtype=torch.float32)
    labels = torch.as_tensor(data.labels, dtype=torch.int64)
    with torch.no_grad():
        logits = model(features)
        correct = logits.gather(1, labels[:, None]).squeeze(1)
        wrong = logits.gather(1, (1 - labels)[:, None]).squeeze(1)
        margin = correct - wrong
    return {
        "n": 2 * n_per_class,
        "errors": int((margin <= 0.0).sum().item()),
        "min_margin": float(margin.min().item()),
        "seed": seed,
    }


def export_weights(model: FoldWitness, path: Path) -> None:
    payload: dict[str, object] = {
        "fold": {
            "description": "(x, y, z) -> (|x - 1|, y, z); as pwl_family(-1): W=I, b=(-1, 4, 4), then subtract (0, 4, 4) fused into the next layer's affine",
            "fold_center": FOLD_CENTER,
        },
        "continuation": {
            name: value.detach().numpy().tolist()
            for name, value in model.state_dict().items()
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def load_witness(path: Path) -> FoldWitness:
    payload = json.loads(path.read_text())
    model = FoldWitness()
    state = {
        name: torch.tensor(value, dtype=torch.float32)
        for name, value in payload["continuation"].items()
    }
    model.load_state_dict(state)
    model.eval()
    return model
