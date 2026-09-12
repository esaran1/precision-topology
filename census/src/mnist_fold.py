"""The budget law in a standard setting: MNIST MLP with an f_a bottleneck.

Registered in ``results/mnist_budget_law_prediction.md`` BEFORE anything
here ran.

Architecture 784 -> 256 -> [w] -> 128 -> 10, ReLU everywhere except the
bottleneck layer, whose activation is f_a(x) = x + a*sin(x).  Monotone
iff a <= 1 exactly; fold depth ~ (a-1)^{3/2}, so beta = 3/2.

Capability is a test-accuracy threshold reached within the budget, not an
exact separation -- real tasks have no exact criterion.  The threshold is
chosen from a pilot and its sensitivity is reported at two other values.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .mnist_data import load
from .train import seed_everything

RESULTS = Path(__file__).resolve().parents[1] / "results"
BATCH = 128
LR = 1e-3


def fold(x: torch.Tensor, a: float) -> torch.Tensor:
    """f_a(x) = x + a sin x.  Monotone iff a <= 1."""

    return x + a * torch.sin(x)


class FoldNet(torch.nn.Module):
    """784 -> 256 -> [w] -> 128 -> 10 with f_a at the bottleneck."""

    def __init__(self, width: int, a: float):
        super().__init__()
        self.a = a
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, width)
        self.fc3 = torch.nn.Linear(width, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Representation entering the bottleneck (the ID axis)."""

        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = fold(self.fc2(h), self.a)
        h = F.relu(self.fc3(h))
        return self.fc4(h)


def _batches(n: int, batch: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n - batch + 1, batch):
        yield order[start:start + batch]


def train_one(width: int, a: float, seed: int, steps: int,
              data: dict, record_scale: bool = False) -> dict:
    """One run.  Returns test accuracy and terminal bottleneck weight scale."""

    seed_everything(seed)
    x, y = data["train_x"], data["train_y"]
    model = FoldNet(width, a)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(seed)

    done = 0
    while done < steps:
        for index in _batches(len(x), BATCH, generator):
            if done >= steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[index]), y[index])
            loss.backward()
            optimizer.step()
            done += 1

    model.eval()
    with torch.no_grad():
        logits = model(data["test_x"])
        accuracy = float((logits.argmax(1) == data["test_y"]).float().mean())
        train_logits = model(x[:10_000])
        train_accuracy = float((train_logits.argmax(1) == y[:10_000]).float().mean())
    out = {"width": width, "a": a, "seed": seed, "steps": steps,
           "test_accuracy": accuracy, "train_accuracy": train_accuracy}
    if record_scale:
        # terminal weight scale at the bottleneck layer: the |w2| analogue
        out["w_scale"] = float(model.fc2.weight.detach().norm())
        out["w_scale_out"] = float(model.fc3.weight.detach().norm())
        out["spectral_fc2"] = float(torch.linalg.matrix_norm(
            model.fc2.weight.detach(), ord=2))
    return out


def mnist_tensors() -> dict:
    raw = load()
    def prep(images, labels):
        x = torch.tensor(images.reshape(len(images), -1), dtype=torch.float32) / 255.0
        x = (x - 0.1307) / 0.3081
        return x, torch.tensor(labels, dtype=torch.long)
    train_x, train_y = prep(raw["train_images"], raw["train_labels"])
    test_x, test_y = prep(raw["test_images"], raw["test_labels"])
    return {"train_x": train_x, "train_y": train_y,
            "test_x": test_x, "test_y": test_y}


def write(frame: pd.DataFrame, name: str) -> None:
    stem = RESULTS / name
    with artifact_lock(stem, name.replace("_", " ")):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))
