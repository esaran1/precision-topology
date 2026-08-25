"""No code path may emit a linking value inside the artifact regime.

Two layers of protection (audit 2026-08-25, AUDIT.md finding "live
bypass"): a behavioural check that every gated emitter returns None for
curves closer than ARTIFACT_DISTANCE, and a static source scan that
fails if a new ``estimate.rounded if …`` emission appears without an
artifact-gate term.  ``cancellation.py`` is the single documented
exemption: it deliberately measures raw projection outcomes (that is
the control that overturned T16), and its conclusion does not rest on
any individual emitted value being trustworthy.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from src.linking import linking_number
from src.linking_trace import ARTIFACT_DISTANCE, _measure

SRC = Path(__file__).resolve().parents[1] / "src"
EXEMPT = {"cancellation.py", "linking.py"}


def _crushed_pair() -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.arange(64, dtype=torch.float64) * (2 * torch.pi / 64)
    first = torch.stack([angles.cos(), angles.sin(), torch.zeros_like(angles)], dim=1)
    # Second curve: same circle nudged by less than the artifact threshold.
    second = first + torch.tensor([0.5 * ARTIFACT_DISTANCE, 0.0, 0.0])
    return first, second


def test_estimator_reports_distance_and_definedness():
    first, second = _crushed_pair()
    estimate = linking_number(first, second)
    assert estimate.min_distance <= ARTIFACT_DISTANCE
    # the estimator itself only guards exact coincidence; callers must gate
    assert estimate.defined


def test_layer_linking_refuses_artifact_regime():
    first, second = _crushed_pair()
    entry = _measure(first.to(torch.float64), second.to(torch.float64),
                     layer=0, width=3)
    assert entry.artifact_regime
    assert entry.rounded is None and entry.raw is None


def test_winding_and_validators_gate():
    from src.corrugation import GRID as CORRUGATION_GRID, validate as validate_corrugation
    from src.parametrization import GRID as PARAM_GRID, validate as validate_parametrization

    # Designed geometry sits far outside the artifact regime, so values must
    # be reported (the gate must not over-fire) …
    assert validate_parametrization(PARAM_GRID[0]).linking_number is not None
    assert validate_corrugation(CORRUGATION_GRID[0]).linking_number is not None


def test_static_scan_every_rounded_emission_is_gated():
    pattern = re.compile(r"\.rounded if ([\w.]+)")
    for path in sorted(SRC.glob("*.py")):
        if path.name in EXEMPT:
            continue
        text = path.read_text()
        for match in pattern.finditer(text):
            guard = match.group(1).split(".")[-1]
            assert guard in {"usable", "reportable"}, (
                f"{path.name}: linking value emitted behind guard "
                f"'{guard}' — every emission must use an artifact-gated "
                f"'usable'/'reportable' flag (see linking_trace.py)"
            )
