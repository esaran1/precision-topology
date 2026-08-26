"""Intrinsic dimension of the CIFAR CNN's bottleneck input (Task F Part 2).

The width axis for Part 2 is the width of a fully-connected bottleneck
inserted at the flatten point of the depth-8 CNN:

    conv1..conv8 -> flatten (512) -> [BOTTLENECK w] -> fc 128 -> head 10

This is the one place in the architecture where "width" is unambiguous
(a flat vector, one number), which is why it is chosen over channel
count or spatial extent; see `results/cifar_width_axis.md`.

The quantity that matters for the prediction is the intrinsic dimension
of the representation *arriving* at the bottleneck, i.e. of the
512-dimensional flattened conv output of a network trained without a
bottleneck.  Three nonlinear estimators are reported (TwoNN, Levina-
Bickel MLE at k = 10 and k = 20), with PCA-95% as a linear reference
only.  Estimated on both GELU and tanh reference nets and two disjoint
subsamples each, so estimator spread and representation dependence are
both visible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifact_lock import artifact_lock
from .cifar_data import load
from .control_search import SmallCNN, train_and_eval
from .intrinsic_dimension import levina_bickel, pca_dimension, two_nn
from .train import seed_everything

SUBSAMPLE = 5_000
EPOCHS = 12


def flatten_features(model: SmallCNN, x: torch.Tensor,
                     batch: int = 500) -> np.ndarray:
    """The 512-d conv output that a bottleneck would receive."""

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            h = x[start:start + batch]
            for i, conv in enumerate(model.convs):
                h = torch.nn.functional.gelu(conv(h)) if model.activation == "gelu" \
                    else torch.tanh(conv(h)) if model.activation == "tanh" \
                    else torch.nn.functional.relu(conv(h))
                if i % 2 == 1 and h.shape[-1] > 1:
                    h = torch.nn.functional.max_pool2d(h, 2)
            outputs.append(h.flatten(1).numpy())
    return np.concatenate(outputs)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    data = load()
    mean = data["train_images"].mean(axis=(0, 2, 3), keepdims=True) / 255.0
    std = data["train_images"].std(axis=(0, 2, 3), keepdims=True) / 255.0
    x = torch.tensor((data["train_images"] / 255.0 - mean) / std, dtype=torch.float32)
    y = torch.tensor(data["train_labels"])
    xt = torch.tensor((data["test_images"] / 255.0 - mean) / std, dtype=torch.float32)
    yt = torch.tensor(data["test_labels"])

    rows = []
    for activation in ("gelu", "tanh"):
        for seed in (0, 1):
            seed_everything(seed)
            model = SmallCNN(8, activation)
            train_and_eval(model, x, y, xt, yt, epochs=EPOCHS, batch=128, lr=1e-3)
            features = flatten_features(model, x[:2 * SUBSAMPLE + 1000])
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(features))
            for name, index in (("sub1", order[:SUBSAMPLE]),
                                ("sub2", order[SUBSAMPLE:2 * SUBSAMPLE])):
                sample = features[index]
                rows.append({
                    "representation": f"flatten512_{activation}_seed{seed}",
                    "subsample": name,
                    "two_nn": two_nn(sample),
                    "mle_k10": levina_bickel(sample, 10),
                    "mle_k20": levina_bickel(sample, 20),
                    "pca_95_linear_reference": pca_dimension(sample),
                })
                print(rows[-1], flush=True)
            frame = pd.DataFrame(rows)
            stem = directory / "cifar_intrinsic"
            with artifact_lock(stem, "cifar intrinsic dimension"):
                temp = stem.with_suffix(".csv.tmp")
                frame.to_csv(temp, index=False)
                temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
