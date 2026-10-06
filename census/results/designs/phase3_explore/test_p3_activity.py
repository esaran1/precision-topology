"""Constructed cases for the Phase 3 activity test (p3_activity.py).  Run: python -m pytest -q test_p3_activity.py"""
import numpy as np
import pytest
import p3_activity as P

# the 2A-PS degenerate release (2f0fb33, phase2a_ps_posthoc.json): v and ∂L/∂v of seed 2,975,052
V52 = np.array([1.499465234754688, 1.446173704391896, 1.1793476661290464e-93, 0.0])
G52 = np.array([-0.07707232379345327, -0.07707232379283815, 3.1581182483505337e-78, 0.0])


def test_theta_fixed():
    assert P.THETA == 1e-8


def test_dead_unit_inactive():
    assert P.active_units(V52) == [0, 1]
    assert P.active_units(np.array([1.0, 1e-93, 2.0, 0.5])) == [0, 2, 3]


def test_zero_unit_inactive():
    assert P.active_units(np.array([0.3, 0.0, 0.2, 0.1])) == [0, 2, 3]


def test_genuine_small_units_active():
    s0 = 0.2225397
    assert P.active_units(s0 * np.array([0.1, 0.2, 0.3, 0.4])) == [0, 1, 2, 3]       # every Phase 3 release
    v = 0.6264 * np.array([0.03593, 0.32934, 0.31737, 0.31737])                      # smallest exploratory share
    assert P.active_units(v) == [0, 1, 2, 3]
    assert P.active_units(np.array([1.0, 1e-6, 1.0, 1.0])) == [0, 1, 2, 3]            # 1e-6 relative: active


def test_threshold_inclusive_and_relative():
    """Exact dyadic arithmetic (θ passed as 2⁻²⁰ so that θ·s is exact): |v_i| = θ·s is active, just below is not."""
    th = 2.0 ** -20
    v = np.array([0.5, 0.25, 0.25 - 2.0 ** -20, 2.0 ** -20])                        # s = 1 exactly
    assert P.active_units(v, theta=th) == [0, 1, 2, 3]
    v2 = np.array([0.5, 0.25, 0.25 - 2.0 ** -21, 2.0 ** -21])                       # |v_3| = θ·s/2
    assert P.active_units(v2, theta=th) == [0, 1, 2]
    assert P.active_units(2.0 ** 30 * v, theta=th) == [0, 1, 2, 3]                   # scale invariant
    assert P.active_units(2.0 ** 30 * v2, theta=th) == [0, 1, 2]


def test_scale_direction_excludes_inactive():
    a = P.scale_direction(V52)
    assert np.allclose(a, [2 ** -0.5, 2 ** -0.5, 0, 0])
    a_old = np.array([1, 1, 1, 0]) / np.sqrt(3)                                        # the registered v ≠ 0 rule
    assert not np.allclose(a, a_old)


@pytest.mark.parametrize("rho, ds_new", [(2.0 ** -14, 9.408e-06), (2.0 ** -16, 2.352e-06)])
def test_effect_on_scale_step(rho, ds_new):
    """With the dead unit in â the first step moves s by 0.1028 (2f0fb33); with the activity test, ρ-limited."""
    old = P.scale_only_step(V52, G52, 1.0, rho, np.array([1, 1, 1, 0]) / np.sqrt(3))
    new = P.scale_only_step(V52, G52, 1.0, rho, P.scale_direction(V52))
    ds_old = np.abs(old).sum() - np.abs(V52).sum()
    ds = np.abs(new).sum() - np.abs(V52).sum()
    assert ds_old == pytest.approx(0.10277, abs=2e-5)
    assert ds == pytest.approx(ds_new, rel=1e-3)
    assert np.sign(old[2]) < 0 and new[2] == pytest.approx(V52[2], abs=1e-70)       # no sign flip of the dead unit
