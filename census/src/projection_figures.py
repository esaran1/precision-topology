"""Render projected core curves for manual inspection.

The author's recommendation is to look at the projected configurations rather
than trust the number.  A projection whose image is several joined rings, or
whose components pass through each other, is identifiable by eye in a way the
integer alone does not convey.

Each figure shows the same representation under several projections side by
side, annotated with the linking value and the projected minimum distance, so a
disagreement between projections is visible directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from .cancellation import random_projection_matrix  # noqa: E402
from .linking import linking_number  # noqa: E402
from .linking_trace import ARTIFACT_DISTANCE  # noqa: E402


def _panel(axis, left: torch.Tensor, right: torch.Tensor, title: str) -> None:
    axis.plot(left[:, 0], left[:, 1], left[:, 2], linewidth=0.9, color="#1f77b4")
    axis.plot(right[:, 0], right[:, 1], right[:, 2], linewidth=0.9, color="#d62728")
    axis.set_title(title, fontsize=8)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_zticklabels([])
    axis.grid(False)


def _describe(left: torch.Tensor, right: torch.Tensor) -> str:
    estimate = linking_number(left, right)
    distance = float(torch.cdist(left, right).min().item())
    usable = estimate.defined and distance > ARTIFACT_DISTANCE
    value = estimate.rounded if usable else ("artifact" if estimate.defined else "undef")
    return f"link {value}   d={distance:.3f}"


def figure_for_projections(
    first: torch.Tensor,
    second: torch.Tensor,
    bases: list[tuple[str, torch.Tensor | None]],
    suptitle: str,
    output_path: Path,
) -> Path:
    """Render one representation under several projections."""

    columns = len(bases)
    figure = plt.figure(figsize=(3.1 * columns, 3.4))
    for index, (label, basis) in enumerate(bases, start=1):
        axis = figure.add_subplot(1, columns, index, projection="3d")
        if basis is None:
            left, right = first, second
        else:
            left, right = first @ basis, second @ basis
        _panel(axis, left, right, f"{label}\n{_describe(left, right)}")
    figure.suptitle(suptitle, fontsize=9)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def pca_bases(
    first: torch.Tensor, second: torch.Tensor, triples: list[tuple[int, int, int]]
) -> list[tuple[str, torch.Tensor]]:
    """Projection bases for the named PCA component triples."""

    combined = torch.cat([first, second], dim=0)
    centred = combined - combined.mean(dim=0, keepdim=True)
    _, _, components = torch.linalg.svd(centred, full_matrices=False)
    return [
        (f"PCA {triple}", components[list(triple)].T)
        for triple in triples
        if max(triple) < components.shape[0]
    ]


def random_bases(dimension: int, seeds: list[int]) -> list[tuple[str, torch.Tensor]]:
    return [
        (f"random {seed}", random_projection_matrix(dimension, seed)) for seed in seeds
    ]
