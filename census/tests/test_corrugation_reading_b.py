"""Pins the documented Reading-B sampling flaw (results/reading_b_anomaly.md).

These tests document recorded behaviour rather than desired behaviour: the
recorded corrugation artifacts were generated with amplitude ignored under
Reading B and with a probability atom on the tube surface.  If either test
starts failing, the generator's semantics changed and every Reading-B
artifact must be regenerated and re-labelled.
"""

from __future__ import annotations

import numpy as np

from src.corrugation import CorrugatedLink, sample
from src.parametrization import TorusLink


def _reading_b(amplitude: float) -> CorrugatedLink:
    return CorrugatedLink(
        base=TorusLink(tube_radius_a=0.15, tube_radius_b=0.15),
        amplitude=amplitude,
        frequency=100.0,
        name=f"test_b_a{amplitude:g}",
        reading="offset",
    )


def test_reading_b_ignores_amplitude():
    first = sample(_reading_b(0.05), 500, seed=11)
    second = sample(_reading_b(0.5), 500, seed=11)
    assert np.array_equal(first.features, second.features)


def test_reading_b_surface_atom():
    """The clip puts a finite fraction of points exactly on the tube surface."""

    data = sample(_reading_b(0.3), 5_000, seed=3)
    points = data.features.astype(np.float64)
    labels = data.labels
    a = points[labels == 0]
    radial = np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)
    distance = np.sqrt((radial - 1.0) ** 2 + a[:, 2] ** 2)
    fraction = float((np.abs(distance - 0.15) < 1e-6).mean())
    assert fraction > 0.15  # ~0.20 in the recorded artifacts, 0 for flat
