"""Task E Part 2a/2b: MNIST bottleneck width sweep.

Registered in ``results/bottleneck_prediction.md`` BEFORE this ran.
Architecture 784 -> 128 -> w -> 128 -> 10; the tanh-vs-GELU comparison
carries the prediction; ReLU/leaky-ReLU are context (Part 1c isolated
their family-specific optimization deficit).
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

WIDTHS = (2, 4, 6, 8, 10, 12, 16, 24, 32, 48)
ACTIVATIONS = ("gelu", "tanh", "relu", "leaky_relu")
SEEDS = {"gelu": range(10), "tanh": range(10), "relu": range(5), "leaky_relu": range(5)}
EPOCHS = 3
BATCH = 256
LR = 1e-3
PROBE_EVERY = 100  # batches; convergence = first probe with >=99% on probe set
PROBE_N = 5_000


def _act(name: str, v: torch.Tensor) -> torch.Tensor:
    if name == "gelu":
        return F.gelu(v)
    if name == "tanh":
        return torch.tanh(v)
    if name == "relu":
        return F.relu(v)
    return F.leaky_relu(v, 0.01)


class BottleneckNet(torch.nn.Module):
    def __init__(self, width: int, activation: str):
        super().__init__()
        self.activation = activation
        self.layer1 = torch.nn.Linear(784, 128)
        self.layer2 = torch.nn.Linear(128, width)   # the bottleneck
        self.layer3 = torch.nn.Linear(width, 128)
        self.head = torch.nn.Linear(128, 10)

    def forward(self, x: torch.Tensor, return_bottleneck: bool = False):
        h1 = _act(self.activation, self.layer1(x))
        pre = self.layer2(h1)
        h2 = _act(self.activation, pre)
        h3 = _act(self.activation, self.layer3(h2))
        out = self.head(h3)
        return (out, pre) if return_bottleneck else out


def run_one(width: int, activation: str, seed: int, x: torch.Tensor,
            y: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor) -> dict:
    started = time.monotonic()
    seed_everything(seed)
    model = BottleneckNet(width, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    probe_x, probe_y = x[:PROBE_N], y[:PROBE_N]

    batches_seen = 0
    batches_to_criterion = -1
    n = len(x)
    for _ in range(EPOCHS):
        order = torch.randperm(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            optimizer.step()
            batches_seen += 1
            if batches_to_criterion < 0 and batches_seen % PROBE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    accuracy = (model(probe_x).argmax(1) == probe_y).float().mean()
                model.train()
                if accuracy >= 0.99:
                    batches_to_criterion = batches_seen

    model.eval()
    with torch.no_grad():
        test_errors = int((model(xt).argmax(1) != yt).sum().item())
        train_logits, bottleneck_pre = model(x[:10_000], return_bottleneck=True)
        train_errors_10k = int((train_logits.argmax(1) != y[:10_000]).sum().item())
        dead = float((bottleneck_pre <= 0).all(dim=0).float().mean().item()) \
            if activation in ("relu",) else float("nan")

    return {
        "width": width, "activation": activation, "seed": seed,
        "test_errors": test_errors, "test_accuracy": 1.0 - test_errors / len(yt),
        "train_errors_10k": train_errors_10k,
        "batches_to_criterion": batches_to_criterion,
        "dead_bottleneck_fraction": dead,
        "duration_seconds": time.monotonic() - started,
    }


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    stem = directory / "bottleneck_sweep"
    data = load()
    x = torch.tensor(data["train_images"].reshape(-1, 784).astype(np.float32) / 255.0)
    y = torch.tensor(data["train_labels"].astype(np.int64))
    xt = torch.tensor(data["test_images"].reshape(-1, 784).astype(np.float32) / 255.0)
    yt = torch.tensor(data["test_labels"].astype(np.int64))

    rows: list[dict] = []
    for width in WIDTHS:
        for activation in ACTIVATIONS:
            for seed in SEEDS[activation]:
                rows.append(run_one(width, activation, seed, x, y, xt, yt))
            cell = [r for r in rows if r["width"] == width and r["activation"] == activation]
            mean_error = sum(r["test_errors"] for r in cell) / len(cell)
            print(f"width={width} {activation}: mean test errors {mean_error:.1f}/10000",
                  flush=True)
            frame = pd.DataFrame(rows)
            with artifact_lock(stem, "bottleneck sweep"):
                temp = stem.with_suffix(".csv.tmp")
                frame.to_csv(temp, index=False)
                temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
