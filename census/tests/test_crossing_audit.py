import numpy as np

from src.crossing_audit import interp_R, summarise


def test_interp_exact_on_linear_path():
    # G crosses zero at step 3.25 on a straight line; R linear in the step: interpolation recovers R(3.25) exactly
    steps = np.arange(0, 6)
    G = (steps - 3.25) * 0.1
    R = 1.0 + 0.01 * steps
    out = summarise(steps, G, R, 1, 1.0)
    assert out["first_positive_step"] == 4 and out["cross_step_check"] == 4
    assert abs(out["R_interp_step"] - (1.0 + 0.0325)) < 1e-14
    assert abs(out["growth_one_step_rel"] - 0.01) < 1e-14


def test_check_grid_bias_and_interpolation():
    # checks every 50 from 0; true zero at 120: first positive check 150, previous 100
    steps = np.arange(0, 151)
    G = (steps - 120.0) * 1e-3
    R = 0.2 + 1e-4 * steps
    out = summarise(steps, G, R, 50, 0.2)
    assert out["cross_step_check"] == 150 and out["first_positive_step"] == 121
    assert abs(out["R_check"] - (0.2 + 0.015)) < 1e-14
    assert abs(out["R_interp_check"] - (0.2 + 0.012)) < 1e-14         # linear G: check interpolation is exact
    assert abs(out["growth_over_interval_rel"] - 0.005 / 0.2) < 1e-12
    # the recorded (check) residual exceeds the interpolated one by the within-interval growth
    assert out["residual_check"] > out["residual_interp_check"] > 0


def test_nonlinear_G_check_interpolation_differs_from_step():
    # a fail case for the check-level definition: G convex in the step, so the chord between checks misplaces the zero
    steps = np.arange(0, 101)
    G = ((steps / 100.0) ** 3) - 0.3 ** 3 * 1.0001
    R = steps * 1.0
    out = summarise(steps, G, R, 50, 100.0)
    assert abs(out["R_interp_step"] - 30.0) < 1.0
    assert abs(out["R_interp_check"] - 30.0) > 10.0


def test_interp_R_endpoints():
    assert interp_R(0.0, 1.0, 5.0, 6.0) == 5.0
    assert abs(interp_R(-1.0, 1e-300, 5.0, 6.0) - 6.0) < 1e-12
