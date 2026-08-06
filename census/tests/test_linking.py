"""Validation of the Gauss linking-number estimator.

This is the blocking gate on Half B.  Until the estimator recovers known
integers, has a calibrated noise floor, and refuses the undefined case, no
linking measurement on trained representations is interpretable.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.linking import (
    chain,
    circle,
    hopf_link,
    intersecting,
    linking_number,
    minimum_distance,
    subdivide,
    torus_link_2_4,
    unlink,
)


# --- Known configurations -------------------------------------------------


def test_unlink_is_zero():
    first, second = unlink()
    estimate = linking_number(first, second)
    assert estimate.rounded == 0
    assert estimate.converged


def test_hopf_link_is_minus_one():
    first, second = hopf_link()
    estimate = linking_number(first, second)
    assert estimate.rounded == -1
    assert estimate.residual < 1e-3
    assert estimate.converged


def test_orientation_reversal_flips_the_sign():
    forward = linking_number(*hopf_link())
    reversed_ = linking_number(*hopf_link(reverse=True))
    assert forward.rounded == -1
    assert reversed_.rounded == 1
    assert forward.raw == pytest.approx(-reversed_.raw, abs=1e-9)


def test_torus_link_2_4_has_magnitude_two():
    estimate = linking_number(*torus_link_2_4())
    assert abs(estimate.rounded) == 2
    assert estimate.residual < 1e-2
    assert estimate.converged


def test_chain_links_adjacent_pairs_only():
    first, second, third = chain()
    assert abs(linking_number(first, second).rounded) == 1
    assert abs(linking_number(second, third).rounded) == 1
    # The two end circles are not linked with each other.
    ends = linking_number(first, third)
    assert ends.rounded == 0
    assert ends.converged


def test_argument_order_does_not_matter():
    first, second = hopf_link()
    assert linking_number(first, second).raw == pytest.approx(
        linking_number(second, first).raw, abs=1e-9
    )


# --- The undefined case ---------------------------------------------------


def test_intersecting_curves_are_reported_undefined():
    """Linking number is undefined for curves that meet; do not return a value."""

    first, second = intersecting()
    estimate = linking_number(first, second)
    assert estimate.min_distance == 0.0
    assert not estimate.defined
    assert not estimate.converged
    assert math.isnan(estimate.residual)


def test_near_intersection_still_recovers_the_integer():
    """Ill-conditioning is gradual: a small positive gap is still usable."""

    first = circle(400, 1.0, (0.0, 0.0, 0.0), "xy")
    second = circle(400, 1.0, (0.05, 0.0, 0.0), "xz")
    estimate = linking_number(first, second)
    assert estimate.defined
    assert estimate.rounded == -1


# --- Noise floor ----------------------------------------------------------


def test_refinement_converges_quadratically():
    """Residual should fall by roughly 4x per doubling of sample count."""

    residuals = []
    for n_points in (100, 200, 400, 800):
        estimate = linking_number(*hopf_link(n_points))
        residuals.append(estimate.residual)
    for coarse, fine in zip(residuals[:-1], residuals[1:]):
        assert fine < coarse
        assert 3.0 < coarse / fine < 5.0


def test_subdivision_refines_a_coarse_cycle():
    coarse = linking_number(*hopf_link(100), subdivisions=1)
    fine = linking_number(*hopf_link(100), subdivisions=4)
    assert fine.residual < coarse.residual
    assert fine.rounded == coarse.rounded == -1


def test_estimator_is_deterministic():
    first, second = hopf_link()
    assert linking_number(first, second).raw == linking_number(first, second).raw


def test_moderate_jitter_does_not_change_the_integer():
    """Across seeds, sampling noise up to 0.2 leaves the rounded value intact."""

    for seed in range(10):
        first = circle(200, 1.0, (0.0, 0.0, 0.0), "xy", seed=100 + seed, jitter=0.2)
        second = circle(200, 1.0, (1.0, 0.0, 0.0), "xz", seed=200 + seed, jitter=0.2)
        assert linking_number(first, second).rounded == -1


def test_heavy_jitter_is_detectably_unreliable():
    """The estimator must not silently return a wrong integer at high noise.

    At jitter 0.5 the curves are perturbed into near-intersection and the
    estimate becomes meaningless.  What matters is that the diagnostics show
    it: either the value stops rounding to the truth or the residual is large,
    rather than a clean-looking wrong answer.
    """

    suspicious = 0
    for seed in range(10):
        first = circle(200, 1.0, (0.0, 0.0, 0.0), "xy", seed=100 + seed, jitter=0.5)
        second = circle(200, 1.0, (1.0, 0.0, 0.0), "xz", seed=200 + seed, jitter=0.5)
        estimate = linking_number(first, second)
        if estimate.rounded != -1 or not estimate.converged:
            suspicious += 1
    assert suspicious >= 8


# --- Mechanics ------------------------------------------------------------


def test_subdivide_multiplies_the_vertex_count():
    cycle = circle(50)
    assert subdivide(cycle, 1).shape == (50, 3)
    assert subdivide(cycle, 4).shape == (200, 3)


def test_subdivide_preserves_the_original_vertices():
    cycle = circle(20)
    refined = subdivide(cycle, 3)
    assert torch.allclose(refined[::3], cycle)


def test_minimum_distance_matches_a_known_configuration():
    first, second = hopf_link()
    assert minimum_distance(first, second) == pytest.approx(1.0, abs=1e-9)


def test_malformed_cycles_are_rejected():
    with pytest.raises(ValueError):
        linking_number(torch.zeros(10, 2), torch.zeros(10, 3))
    with pytest.raises(ValueError):
        linking_number(torch.zeros(2, 3), torch.zeros(10, 3))
    with pytest.raises(ValueError):
        subdivide(circle(10), 0)
