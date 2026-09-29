"""§14 fold-tracking checks: the Airy constant, the normal form, and the reduced delay law on known systems."""

import math

from src import theory_checks as T


def test_omega0_is_first_airy_zero():
    assert abs(float(T.omega0()) - T.OMEGA0) < 1e-14


def test_riccati_blowup_at_omega0():
    assert abs(T.riccati_blowup(-6.0) - T.OMEGA0) < 1e-8


def test_scalar_delay_law_converges():
    r = T.scalar_case(0.3, 1.0, 1.0, 1.0, 1.0, 1e-5)
    assert abs(r["ratio"] - 1) < 0.01


def test_two_d_delay_law_with_preconditioner():
    r = T.two_d_case(0.1, 1e-6)
    assert abs(r["ratio"] - 1) < 0.03 and r["H_null_check"] < 1e-12


def test_normalisation_matters():
    # with P = 0.4 the normalised constants give the right delay; ignoring P (r = 1) misses it by a factor p^(-2/3)
    r = T.scalar_case(0.1, 2.0, 0.5, 0.4, 3.0, 1e-6)
    eta_p = T.fold_delay_pred(1e-6, 0.1 * 0.4, -2.0, 0.5)          # equivalent: eta*p as a single rate, r = 1
    unnormalised = T.fold_delay_pred(1e-6, 0.1, -2.0, 0.5)
    assert abs(r["ratio"] - 1) < 0.03
    assert abs(r["delay_obs"] / unnormalised - 1) > 0.3
    assert abs(r["delay_obs"] / eta_p - 1) < 0.03


def test_lambda_sq_slope_approaches_four_mc():
    d = T.lambda_sq_slope()
    last = d["rows"][-1]["lambda_sq_over_d"]
    assert abs(last / d["four_abs_mc"] - 1) < 0.03


def test_ks01_bessel_definition_equals_airy_zero():
    d = T.bessel_form_at_omega0()
    assert abs(float(d["smallest_positive_zero"]) - T.OMEGA0) < 1e-12


def test_theorem_L_bounds_hold_on_toy_ramps():
    for eta in (0.25, 0.1):
        for sdot in (1e-3, 2.5e-4):
            r = T.lag_toy_run(sdot, eta)
            assert r["conditions_hold"] and r["bound_holds"] and r["dev_bound_holds"]
    r = T.lag_toy_run(1.25e-4, 0.1, ramp="exponential")
    assert r["conditions_hold"] and r["bound_holds"] and r["dev_bound_holds"]


def test_theorem_L_detects_a_wrong_prediction():
    # constructed fail: shifting the prediction by 5x the bound must violate it
    r = T.lag_toy_run(2.5e-4, 0.1)
    assert abs(r["lag_minus_ell"] + 5 * r["lag_error_bound"]) > r["lag_error_bound"]


def test_second_order_residual_with_interpolated_crossing():
    a = T.lag_toy_run(5e-4, 0.1); b = T.lag_toy_run(1.25e-4, 0.1)
    ra = (a["r_interp"] - a["ell_over_sstar"]) / a["chi"] ** 2
    rb = (b["r_interp"] - b["ell_over_sstar"]) / b["chi"] ** 2
    assert abs(ra - rb) < 0.01 * abs(rb)


def test_free_training_lag_eta_invariant():
    r = [T.free_training_toy(eta)["r_obs"] for eta in (0.1, 0.025)]
    assert abs(r[0] / r[1] - 1) < 0.01
