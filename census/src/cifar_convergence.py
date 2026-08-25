"""Task F: depth-8 CIFAR cells trained to plateau (restoration criterion 4).

Same architecture, data, optimizer, and fresh seeds (3-12) as
``cifar_replication.csv``; the only change is the budget: up to 40
epochs with an explicit plateau stop (relative training-loss improvement
< 1% on two consecutive epochs).  Per-epoch losses and test errors are
recorded so the 12-epoch snapshot is a strict prefix of this run's
trajectory.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .cifar_data import load
from .control_search import SmallCNN
from .train import seed_everything

MAX_EPOCHS = 40
PLATEAU_RELATIVE = 0.01
PLATEAU_CONSECUTIVE = 2
CELLS = (("gelu", range(3, 13)), ("relu", range(3, 13)), ("tanh", range(3, 8)))


def run_one(activation: str, seed: int, x, y, xt, yt) -> dict:
    started = time.monotonic()
    seed_everything(seed)
    model = SmallCNN(8, activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    losses: list[float] = []
    consecutive = 0
    epochs_run = 0
    for _ in range(MAX_EPOCHS):
        order = torch.randperm(n)
        running = 0.0
        model.train()
        for start in range(0, n, 128):
            idx = order[start:start + 128]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(idx)
        losses.append(running / n)
        epochs_run += 1
        if len(losses) >= 2:
            improvement = (losses[-2] - losses[-1]) / max(abs(losses[-2]), 1e-12)
            consecutive = consecutive + 1 if improvement < PLATEAU_RELATIVE else 0
            if consecutive >= PLATEAU_CONSECUTIVE:
                break
    model.eval()
    with torch.no_grad():
        test_errors = 0
        for start in range(0, len(xt), 1_000):
            test_errors += int((model(xt[start:start + 1_000]).argmax(1)
                                != yt[start:start + 1_000]).sum().item())
        train_errors = 0
        for start in range(0, 10_000, 1_000):
            train_errors += int((model(x[start:start + 1_000]).argmax(1)
                                 != y[start:start + 1_000]).sum().item())
    return {"depth": 8, "activation": activation, "seed": seed,
            "epochs_run": epochs_run, "plateaued": consecutive >= PLATEAU_CONSECUTIVE,
            "test_errors": test_errors, "train_errors_10k": train_errors,
            "final_train_loss": losses[-1],
            "loss_trajectory": "|".join(f"{v:.5f}" for v in losses),
            "duration_seconds": time.monotonic() - started}


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    stem = directory / "cifar_convergence"
    data = load()
    mean = data["train_images"].mean(axis=(0, 2, 3), keepdims=True) / 255.0
    std = data["train_images"].std(axis=(0, 2, 3), keepdims=True) / 255.0
    x = torch.tensor((data["train_images"] / 255.0 - mean) / std, dtype=torch.float32)
    y = torch.tensor(data["train_labels"])
    xt = torch.tensor((data["test_images"] / 255.0 - mean) / std, dtype=torch.float32)
    yt = torch.tensor(data["test_labels"])
    rows: list[dict] = []
    for activation, seeds in CELLS:
        for seed in seeds:
            rows.append(run_one(activation, seed, x, y, xt, yt))
            frame = pd.DataFrame(rows)
            with artifact_lock(stem, "cifar convergence"):
                temp = stem.with_suffix(".csv.tmp")
                frame.to_csv(temp, index=False)
                temp.replace(stem.with_suffix(".csv"))
            r = rows[-1]
            print(f"{activation} seed={seed}: {r['epochs_run']} epochs "
                  f"plateaued={r['plateaued']} test={r['test_errors']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
