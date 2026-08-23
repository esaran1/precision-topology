"""Task E 2a: intrinsic dimension of MNIST, three-plus estimators.

Estimators (all stdlib/numpy/torch, no sklearn in this venv):
- TwoNN (Facco et al. 2017): d from the distribution of mu = r2/r1,
  via the ML estimator d = (n - 1) / sum(log mu) with the standard
  10% largest-mu discard.
- Levina-Bickel MLE at k = 10 and k = 20 (two of the three primary
  estimators; the k-dependence is part of the reported spread).
- PCA dimension at 95% explained variance — reported as a *linear
  reference only*, not part of the nonlinear spread (it upper-bounds
  heavily on curved manifolds).

Run on raw pixels (canonical literature range ~10-15) and on the
128-unit first-hidden-layer representation of trained reference
networks (tanh and gelu, 3 seeds each), since the registered width axis
is the intrinsic dimension of the data as it arrives at the bottleneck.
Distances are computed in float64 on 5,000-point subsamples, two
disjoint subsamples per input to expose sampling noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .artifact_lock import artifact_lock
from .mnist_data import load
from .train import seed_everything

SUBSAMPLE = 5_000


def _knn_distances(points: np.ndarray, k: int) -> np.ndarray:
    """Distances to the k nearest neighbours, chunked, float64."""

    n = len(points)
    points = points.astype(np.float64)
    norms = (points ** 2).sum(axis=1)
    out = np.empty((n, k))
    step = 512
    for start in range(0, n, step):
        chunk = points[start:start + step]
        d2 = norms[start:start + step, None] + norms[None, :] - 2.0 * chunk @ points.T
        np.maximum(d2, 0.0, out=d2)
        d2[np.arange(len(chunk)), np.arange(start, start + len(chunk))] = np.inf
        part = np.partition(d2, k - 1, axis=1)[:, :k]
        out[start:start + len(chunk)] = np.sort(np.sqrt(part), axis=1)
    return out


def two_nn(points: np.ndarray) -> float:
    r = _knn_distances(points, 2)
    mu = r[:, 1] / np.maximum(r[:, 0], 1e-300)
    mu = mu[(mu > 1.0) & np.isfinite(mu)]
    mu.sort()
    kept = mu[: int(0.9 * len(mu))]  # standard 10% tail discard
    return float(len(kept) / np.log(kept).sum())


def levina_bickel(points: np.ndarray, k: int) -> float:
    r = _knn_distances(points, k)
    with np.errstate(divide="ignore"):
        logs = np.log(r[:, k - 1][:, None] / r[:, : k - 1])
    inverse = logs.sum(axis=1) / (k - 2)  # MacKay-Ghahramani correction uses k-2
    inverse = inverse[np.isfinite(inverse) & (inverse > 0)]
    return float(1.0 / inverse.mean())


def pca_dimension(points: np.ndarray, variance: float = 0.95) -> int:
    centered = points.astype(np.float64) - points.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    ratio = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(ratio, variance) + 1)


class ReferenceNet(torch.nn.Module):
    """784 -> 128 -> (bottleneck-position) trained without a bottleneck.

    Only the first layer matters here: its 128-d output is "the data as
    it arrives at the bottleneck" for the sweep architecture
    784 -> 128 -> w -> 128 -> 10.
    """

    def __init__(self, activation: str):
        super().__init__()
        self.activation = activation
        self.layer1 = torch.nn.Linear(784, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.head = torch.nn.Linear(128, 10)

    def _act(self, v: torch.Tensor) -> torch.Tensor:
        return torch.tanh(v) if self.activation == "tanh" else F.gelu(v)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self._act(self.layer1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._act(self.layer2(self.hidden(x)))
        return self.head(h)


def train_reference(x: torch.Tensor, y: torch.Tensor, activation: str,
                    seed: int, epochs: int = 3, batch: int = 256) -> ReferenceNet:
    seed_everything(seed)
    model = ReferenceNet(activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            optimizer.step()
    return model


def estimate_all(points: np.ndarray, label: str, rows: list[dict], seed: int) -> None:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    for half, name in ((order[:SUBSAMPLE], "sub1"), (order[SUBSAMPLE:2 * SUBSAMPLE], "sub2")):
        sample = points[half]
        rows.append({
            "input": label, "subsample": name,
            "two_nn": two_nn(sample),
            "mle_k10": levina_bickel(sample, 10),
            "mle_k20": levina_bickel(sample, 20),
            "pca_95_linear_reference": pca_dimension(sample),
        })
        print(rows[-1], flush=True)


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "results"
    data = load()
    pixels = data["train_images"].reshape(60_000, 784).astype(np.float32) / 255.0
    labels = data["train_labels"].astype(np.int64)
    rows: list[dict] = []

    estimate_all(pixels, "raw_pixels", rows, seed=0)

    x = torch.tensor(pixels)
    y = torch.tensor(labels)
    for activation in ("tanh", "gelu"):
        for seed in (0, 1, 2):
            model = train_reference(x, y, activation, seed)
            model.eval()
            with torch.no_grad():
                hidden = model.hidden(x).numpy()
            estimate_all(hidden, f"hidden_{activation}_seed{seed}", rows, seed=seed + 10)

    frame = pd.DataFrame(rows)
    stem = directory / "intrinsic_dimension"
    with artifact_lock(stem, "MNIST intrinsic dimension"):
        temp = stem.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        temp.replace(stem.with_suffix(".csv"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
