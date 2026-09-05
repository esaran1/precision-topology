"""Part 1: does an architecture comparison flip with training budget?

CIFAR-10, depth-8 SmallCNN, GELU/ReLU/tanh, fresh seeds (100+), budgets
2/5/12/30/60 epochs.  Records test error, train error, and terminal weight
norms (the account is about how far the optimizer travels, so norms are
recorded alongside every cell).

Cells are written after every run so partial results are usable.
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

RESULTS = Path(__file__).resolve().parents[1] / "results"
BUDGETS = (2, 5, 12, 30, 60)
ACTS = ("gelu", "relu", "tanh")
SEEDS = range(100, 108)          # 8 fresh seeds, disjoint from earlier work


def weight_norms(model) -> dict:
    total = 0.0
    spectral = 1.0
    for p in model.parameters():
        if p.dim() >= 2:
            total += float(p.detach().norm()) ** 2
            m = p.detach().reshape(p.shape[0], -1)
            spectral *= float(torch.linalg.matrix_norm(m, ord=2))
    return {"weight_l2": total ** 0.5, "spectral_product": spectral}


def run_one(activation: str, seed: int, epochs: int, x, y, xt, yt) -> dict:
    started = time.monotonic()
    seed_everything(seed)
    model = SmallCNN(8, activation)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(n)
        running = 0.0
        model.train()
        for start in range(0, n, 128):
            idx = order[start:start + 128]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(idx)
        losses.append(running / n)
    model.eval()
    with torch.no_grad():
        test_err = sum(int((model(xt[i:i+1000]).argmax(1) != yt[i:i+1000]).sum())
                       for i in range(0, len(xt), 1000))
        train_err = sum(int((model(x[i:i+1000]).argmax(1) != y[i:i+1000]).sum())
                        for i in range(0, 10_000, 1000))
    return {"activation": activation, "seed": seed, "epochs": epochs,
            "test_errors": test_err, "train_errors_10k": train_err,
            "final_train_loss": losses[-1], **weight_norms(model),
            "duration_seconds": time.monotonic() - started}


def main() -> None:
    data = load()
    mean = data["train_images"].mean(axis=(0, 2, 3), keepdims=True) / 255.0
    std = data["train_images"].std(axis=(0, 2, 3), keepdims=True) / 255.0
    x = torch.tensor((data["train_images"] / 255.0 - mean) / std, dtype=torch.float32)
    y = torch.tensor(data["train_labels"])
    xt = torch.tensor((data["test_images"] / 255.0 - mean) / std, dtype=torch.float32)
    yt = torch.tensor(data["test_labels"])

    stem = RESULTS / "budget_flip"
    rows = []
    # budget-major so every activation is comparable at each budget as it lands
    for epochs in BUDGETS:
        for activation in ACTS:
            for seed in SEEDS:
                rows.append(run_one(activation, seed, epochs, x, y, xt, yt))
                frame = pd.DataFrame(rows)
                with artifact_lock(stem, "budget flip"):
                    tmp = stem.with_suffix(".csv.tmp")
                    frame.to_csv(tmp, index=False)
                    tmp.replace(stem.with_suffix(".csv"))
            cell = [r for r in rows if r["epochs"] == epochs
                    and r["activation"] == activation]
            print(f"epochs={epochs} {activation}: mean test errors "
                  f"{np.mean([c['test_errors'] for c in cell]):.1f} "
                  f"(n={len(cell)})", flush=True)
        print(f"--- budget {epochs} complete ---", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
