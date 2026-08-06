"""Propagate ordered core circles through a trained network and track linking.

Two things make this different from the collision census.

The class supports the network is trained on are sampled *solid tori*, but the
linking number is defined on the one-dimensional core curves.  Ordered samples
of those cores are therefore propagated separately from the training data; the
tubes are never fed to the estimator.

The invariant is only the Theorem 4.7 invariant at width 3.  For two 1-D curves
`m = n = 1`, complementary dimension requires `m + n + 1 = d = 3`.  At any
larger width the curves are not complementary-dimensional, so a linking number
can only be obtained by projecting back to `R^3`, and that projection is a
stated convention rather than a property of the layer.  :class:`LayerLinking`
records which regime produced each number and refuses to let the two be
confused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import linked_core_circles
from .linking import LinkingEstimate, linking_number
from .models import MLP


# Below this inter-curve distance the Gauss integrand is ill-conditioned and any
# value it returns is an artifact rather than a linking number.  Calibrated from
# the failure boundary in results/linking_validation.md, where estimates degrade
# once perturbation drives minimum distance to roughly 0.05 or below.
ARTIFACT_DISTANCE = 0.02

# The exact projection convention used above width 3, stated once here so every
# reported number can name it.
PROJECTION_CONVENTION = (
    "PCA to R^3 fitted jointly on both propagated core curves at that layer, "
    "components ordered by descending explained variance"
)


@dataclass(frozen=True)
class LayerLinking:
    """Linking diagnostics for one layer of one network."""

    layer: int
    width: int
    raw: float | None
    rounded: int | None
    residual: float | None
    min_distance: float
    defined: bool
    artifact_regime: bool
    projected: bool

    @property
    def reportable(self) -> bool:
        """Whether a linking value may be quoted for this layer at all."""

        return self.defined and not self.artifact_regime

    @property
    def regime(self) -> str:
        if not self.defined:
            return "undefined (curves meet)"
        if self.artifact_regime:
            return "artifact (too close)"
        return "exact (d=3)" if not self.projected else "projected"


def _project_to_r3(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint PCA of both curves onto three components.

    Fitting on the union keeps a single shared frame, so the relative geometry
    of the two curves is preserved; fitting per curve would not be meaningful.
    """

    combined = torch.cat([first, second], dim=0)
    centred = combined - combined.mean(dim=0, keepdim=True)
    # SVD rather than an eigendecomposition of the covariance: better
    # conditioned, and V columns are already ordered by singular value.
    _, _, components = torch.linalg.svd(centred, full_matrices=False)
    basis = components[:3].T
    projected = centred @ basis
    return projected[: first.shape[0]], projected[first.shape[0] :]


@torch.no_grad()
def trace_linking(
    model: MLP,
    n_core_points: int = 512,
    major_radius: float = 1.0,
) -> list[LayerLinking]:
    """Track linking of the two class cores through every layer of a network.

    Layer 0 is the input configuration.  Each subsequent entry is the
    post-activation representation of one hidden layer.
    """

    first_array, second_array = linked_core_circles(n_core_points, major_radius)
    first = torch.as_tensor(first_array, dtype=torch.float64, device="cpu")
    second = torch.as_tensor(second_array, dtype=torch.float64, device="cpu")

    trace: list[LayerLinking] = [_measure(first, second, layer=0, width=3)]
    for index, layer in enumerate(model.hidden_layers, start=1):
        weight = layer.weight.to(torch.float64)
        bias = layer.bias.to(torch.float64)
        first = model._activate(torch.nn.functional.linear(first, weight, bias))
        second = model._activate(torch.nn.functional.linear(second, weight, bias))
        trace.append(_measure(first, second, layer=index, width=model.hidden_width))
    return trace


def _measure(
    first: torch.Tensor,
    second: torch.Tensor,
    layer: int,
    width: int,
) -> LayerLinking:
    """Estimate linking at one layer, projecting only when width exceeds 3."""

    # Minimum distance is measured in the layer's own space, before any
    # projection, because that is where the curves either meet or do not.
    native_distance = float(torch.cdist(first, second).min().item())

    projected = width > 3
    if projected:
        left, right = _project_to_r3(first, second)
    else:
        left, right = first, second

    estimate: LinkingEstimate = linking_number(left, right)
    artifact = native_distance <= ARTIFACT_DISTANCE
    usable = estimate.defined and not artifact
    return LayerLinking(
        layer=layer,
        width=width,
        raw=estimate.raw if usable else None,
        rounded=estimate.rounded if usable else None,
        residual=estimate.residual if usable else None,
        min_distance=native_distance,
        defined=estimate.defined,
        artifact_regime=artifact,
        projected=projected,
    )


def trace_to_records(
    trace: list[LayerLinking],
    **metadata: object,
) -> list[dict[str, object]]:
    """Flatten a trace into tidy rows, carrying run metadata."""

    return [
        {
            **metadata,
            "layer": entry.layer,
            "layer_width": entry.width,
            "linking_raw": entry.raw,
            "linking_rounded": entry.rounded,
            "linking_residual": entry.residual,
            "min_distance": entry.min_distance,
            "defined": entry.defined,
            "artifact_regime": entry.artifact_regime,
            "projected": entry.projected,
            "reportable": entry.reportable,
            "regime": entry.regime,
            "projection_convention": PROJECTION_CONVENTION if entry.projected else None,
        }
        for entry in trace
    ]


def first_change_layer(trace: list[LayerLinking]) -> int | None:
    """Layer at which the reportable linking number first differs from input.

    Returns ``None`` if it never changes while remaining reportable.
    """

    baseline = trace[0].rounded
    for entry in trace[1:]:
        if not entry.reportable:
            continue
        if entry.rounded != baseline:
            return entry.layer
    return None


def first_unreportable_layer(trace: list[LayerLinking]) -> int | None:
    """Layer at which linking first stops being measurable."""

    for entry in trace[1:]:
        if not entry.reportable:
            return entry.layer
    return None
