import numpy as np

from src import width2_lag as wl


def test_kappa_identity_and_scale_invariance():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5)); H = A @ A.T + 5 * np.eye(5)
    tan, dG = rng.standard_normal(5), rng.standard_normal(5)
    # H ∝ I and P = I: the lag is isotropic, κ = 1
    assert abs(wl.kappa(3.0 * np.eye(5), tan, dG, np.ones(5))[0] - 1.0) < 1e-12
    p = rng.uniform(0.5, 2.0, 5)
    k1 = wl.kappa(H, tan, dG, p)[0]
    k2 = wl.kappa(H, tan, dG, 7.3 * p)[0]
    assert abs(k1 - k2) < 1e-10 * max(1, abs(k1))


def test_kappa_matches_width1_formula():
    from src import lag_law
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3)); H = A @ A.T + np.eye(3)
    tan, dG, p = rng.standard_normal(3), rng.standard_normal(3), rng.uniform(0.2, 3, 3)
    assert abs(wl.kappa(H, tan, dG, p)[0] - lag_law.kappa(H, tan, dG, p)[0]) < 1e-12


def test_v_at_interpolates_and_extrapolates():
    V = np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 2.0]])
    assert np.allclose(wl.v_at(V, 0.5, 2, np.array([1.0, 0.0])), [0.5, 1.5])
    assert np.allclose(wl.v_at(V, 2.0, 2, np.array([1.0, 0.0])), [3.0, 2.0])
    assert np.allclose(wl.v_at(V, 4.0, 2, np.array([1.0, -1.0])), [5.0, 0.0])
    assert np.allclose(wl.v_at(V, -1.0, 2, np.array([1.0, 0.0])), [0.0, 1.0])


def test_corrector_acceptance():
    zp = np.zeros(5); pred = np.full(5, 0.1)
    assert wl.corrector_ok(pred + 0.01, pred, zp)          # small correction: smooth branch
    assert not wl.corrector_ok(pred + 0.2, pred, zp)       # correction larger than the step: a jump
    assert wl.corrector_ok(zp + 5e-4, zp, zp)              # zero step, tiny correction allowed (1e-3 floor)


def test_within_tolerance():
    assert wl.within(0.015, 0.01)                          # floor 0.01
    assert not wl.within(0.025, 0.01)
    assert wl.within(0.12, 0.1) and not wl.within(0.13, 0.1)   # 25% of |pred|


def test_path_kappa_reduces_to_ray_kappa():
    """With no direction drift (v̇ ∥ v) the path κ₂ equals 1A's κ on the ray."""
    from src.asym_register import _act, _setup, training_set
    _setup(); act = _act()
    x, y = training_set(600_320)
    z0 = np.array([1.8, 1.6, 1.7, 4.6, 0.1])
    v = np.array([0.3, -0.25])
    z, gm, H, conv = wl.branch_point(z0, v, x, y, act)
    assert conv and wl._is_min(H)
    s = float(np.abs(v).sum()); vt = v / s
    p = np.array([1.0, 2.0, 0.5, 1.5, 3.0])
    Hr, tan = wl.tangent(z, s, vt, x, y, act)
    dG, _ = wl.grad_gap(z, vt, act)
    k_ray = wl.kappa(Hr, tan, dG, p)[0]
    sdot = 1e-3
    k_path = wl.path_kappa(z, v, vt * sdot, sdot, p, x, y, act)
    assert abs(k_path["gdot_dir_part"]) < 1e-12
    assert abs(k_path["kappa2"] - k_ray) < 1e-6 * max(1, abs(k_ray))
