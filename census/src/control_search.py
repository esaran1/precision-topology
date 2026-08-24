"""Task F 1a: search map for the GELU-over-ReLU positive control.

Criterion fixed beforehand in ``results/control_criterion.md``.  Two
arms: deep MNIST MLPs (cheap) and small CIFAR-10 CNNs (main).  tanh is
run everywhere (the 1c kill switch).  Pilot seeds only locate
candidates; declarations need the n >= 10 replication stage.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .train import seed_everything

ACTS = ("gelu", "relu", "tanh")


def _act(name: str, v: torch.Tensor) -> torch.Tensor:
    if name == "gelu":
        return F.gelu(v)
    if name == "relu":
        return F.relu(v)
    return torch.tanh(v)


class DeepMLP(torch.nn.Module):
    def __init__(self, depth: int, activation: str, width: int = 256):
        super().__init__()
        self.activation = activation
        sizes = [784] + [width] * depth
        self.hidden = torch.nn.ModuleList(
            torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.head = torch.nn.Linear(width, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = _act(self.activation, layer(x))
        return self.head(x)


class SmallCNN(torch.nn.Module):
    """VGG-style: 3x3 convs, max-pool every second conv, small FC head."""

    def __init__(self, conv_depth: int, activation: str):
        super().__init__()
        self.activation = activation
        channels = [3]
        plan = []
        c = 32
        for i in range(conv_depth):
            plan.append((channels[-1], c))
            channels.append(c)
            if i % 2 == 1:
                c = min(c * 2, 128)
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv2d(a, b, 3, padding=1) for a, b in plan)
        spatial = 32 // (2 ** (conv_depth // 2))
        spatial = max(spatial, 1)
        self.flat = channels[-1] * spatial * spatial
        self.fc = torch.nn.Linear(self.flat, 128)
        self.head = torch.nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = _act(self.activation, conv(x))
            if i % 2 == 1 and x.shape[-1] > 1:
                x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = _act(self.activation, self.fc(x))
        return self.head(x)


def train_and_eval(model, x, y, xt, yt, epochs: int, batch: int, lr: float,
                   warmup_steps: int = 0) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    epoch_losses = []
    step = 0
    for _ in range(epochs):
        order = torch.randperm(n)
        running = 0.0
        model.train()
        for start in range(0, n, batch):
            step += 1
            if warmup_steps:
                scale = min(1.0, step / warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr * scale
            idx = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(idx)
        epoch_losses.append(running / n)
    model.eval()
    with torch.no_grad():
        test_errors = 0
        for start in range(0, len(xt), 1_000):
            test_errors += int(
                (model(xt[start:start + 1_000]).argmax(1) != yt[start:start + 1_000]).sum().item())
        train_errors = 0
        for start in range(0, 10_000, 1_000):
            train_errors += int(
                (model(x[start:start + 1_000]).argmax(1) != y[start:start + 1_000]).sum().item())
    plateaued = len(epoch_losses) >= 2 and (
        epoch_losses[-2] - epoch_losses[-1]) < 0.01 * abs(epoch_losses[-2])
    return {"test_errors": test_errors, "train_errors_10k": train_errors,
            "final_train_loss": epoch_losses[-1],
            "penultimate_train_loss": epoch_losses[-2] if len(epoch_losses) >= 2 else float("nan"),
            "loss_plateaued": plateaued}


def run_mnist(rows: list, persist, seeds=range(3)) -> None:
    from .mnist_data import load
    data = load()
    x = torch.tensor(data["train_images"].reshape(-1, 784).astype(np.float32) / 255.0)
    y = torch.tensor(data["train_labels"].astype(np.int64))
    xt = torch.tensor(data["test_images"].reshape(-1, 784).astype(np.float32) / 255.0)
    yt = torch.tensor(data["test_labels"].astype(np.int64))
    for depth in (4, 8, 12):
        for activation in ACTS:
            for seed in seeds:
                started = time.monotonic()
                seed_everything(seed)
                model = DeepMLP(depth, activation)
                metrics = train_and_eval(model, x, y, xt, yt,
                                         epochs=15, batch=256, lr=1e-3)
                rows.append({"arm": "mnist_mlp", "depth": depth,
                             "activation": activation, "seed": seed,
                             "warmup": 0, **metrics,
                             "duration_seconds": time.monotonic() - started})
            cell = [r for r in rows if r["arm"] == "mnist_mlp"
                    and r["depth"] == depth and r["activation"] == activation]
            print(f"mnist_mlp depth={depth} {activation}: "
                  f"mean test errors {np.mean([r['test_errors'] for r in cell]):.0f}",
                  flush=True)
            persist()


def run_cifar(rows: list, persist, seeds=range(3)) -> None:
    from .cifar_data import load
    data = load()
    mean = data["train_images"].mean(axis=(0, 2, 3), keepdims=True) / 255.0
    std = data["train_images"].std(axis=(0, 2, 3), keepdims=True) / 255.0
    x = torch.tensor((data["train_images"] / 255.0 - mean) / std, dtype=torch.float32)
    y = torch.tensor(data["train_labels"])
    xt = torch.tensor((data["test_images"] / 255.0 - mean) / std, dtype=torch.float32)
    yt = torch.tensor(data["test_labels"])
    for depth, warmup in ((4, 0), (8, 0), (8, 500)):
        for activation in ACTS:
            for seed in seeds:
                started = time.monotonic()
                seed_everything(seed)
                model = SmallCNN(depth, activation)
                metrics = train_and_eval(model, x, y, xt, yt,
                                         epochs=12, batch=128, lr=1e-3,
                                         warmup_steps=warmup)
                rows.append({"arm": "cifar_cnn", "depth": depth,
                             "activation": activation, "seed": seed,
                             "warmup": warmup, **metrics,
                             "duration_seconds": time.monotonic() - started})
                persist()
            cell = [r for r in rows if r["arm"] == "cifar_cnn" and r["depth"] == depth
                    and r["activation"] == activation and r["warmup"] == warmup]
            print(f"cifar_cnn depth={depth} warmup={warmup} {activation}: "
                  f"mean test errors {np.mean([r['test_errors'] for r in cell]):.0f}",
                  flush=True)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    stem = directory / "control_search"
    rows: list[dict] = []

    def persist() -> None:
        frame = pd.DataFrame(rows)
        with artifact_lock(stem, "control search"):
            temp = stem.with_suffix(".csv.tmp")
            frame.to_csv(temp, index=False)
            temp.replace(stem.with_suffix(".csv"))

    run_mnist(rows, persist)
    run_cifar(rows, persist)
    print("done", flush=True)


if __name__ == "__main__":
    main()
