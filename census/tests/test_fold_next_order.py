"""Track C: the next-order fold-passage delay (src/fold_next_order.py, math note §18).

Constructed cases: polynomial normal forms (flow and map) whose next-order coefficients follow from the formula; the
derived σ_q is checked against direct numerical integration at small ε (the error divided by ε must go to 0), with
FAIL cases (a wrong or missing term leaves an O(ε) or O(ε ln ε) error)."""

import json
import math

import numpy as np
import pytest

from src import fold_next_order as FN

FLOW_CASES = [dict(p=1.0, q=1.0, f3=0.3, f11=-0.4, g0=1.0, g1=0.5, xq=1.0),
              dict(p=0.7, q=1.6, f3=-0.5, f11=0.6, g0=0.8, g1=-0.3, xq=0.8)]


def _flow_pred(c, discrete=False):
    A, B = FN.inner_AB(c["p"], c["q"], c["f3"], c["f11"], c["g0"], c["g1"], discrete=discrete)
    J = FN.J_flow(c["q"], c["f3"], c["g0"], c["g1"], A, c["xq"])
    return A, B, J, FN.coefficients(c["p"], c["q"], c["g0"], A, B, J, discrete=discrete)


def test_airy_constants():
    a = FN.airy_constants()
    assert abs(a["Omega0"] - FN.OMEGA0) < 1e-14
    assert abs(a["c_A"] - FN.C_A) < 1e-14
    assert abs(a["c_B"] - 0.5) < 1e-25 or abs(a["c_B"] - 0.5) < 1e-14


def test_inner_AB_definitions():
    A, B = FN.inner_AB(2.0, 0.5, 0.3, -0.2, 1.5, 0.6, discrete=False)
    assert A == pytest.approx(0.3 / 0.5 - 0.6 / 1.5) and B == pytest.approx(-0.2 / 2.0 - 0.6 / 1.5)
    Ad, Bd = FN.inner_AB(2.0, 0.5, 0.3, -0.2, 1.5, 0.6, discrete=True)
    assert Ad == pytest.approx(A - 0.5) and Bd == pytest.approx(B - 0.5)


def test_expansion_slope_is_log_derivative():
    for rho in (1e-4, 1e-6):
        f = lambda r: FN.expansion(r, 1e-3, 5e-4, 0.1, -3.0, 500.0)["sigma"]        # noqa: E731
        h = 1e-6
        num = (math.log(f(rho * math.exp(h))) - math.log(f(rho * math.exp(-h)))) / (2 * h)
        assert FN.expansion(rho, 1e-3, 5e-4, 0.1, -3.0, 500.0)["slope"] == pytest.approx(num, abs=1e-8)


@pytest.mark.parametrize("c", FLOW_CASES)
def test_flow_formula_matches_integration(c):
    A, B, J, KD = _flow_pred(c)
    errs = []
    for eps in (1e-3, 1e-4, 1e-5, 1e-6):
        s = FN.simulate_flow(eps, c["p"], c["q"], c["f3"], c["f11"], c["g0"], c["g1"], c["xq"])
        errs.append((s - FN.expansion(eps, c["p"], c["q"], c["g0"], KD["K"], KD["D"])["sigma"]) / eps)
    # the remainder is o(ε): error/ε falls (like ε^{1/3} up to logs) and is small against D
    assert abs(errs[-1]) < 0.01 * max(1.0, abs(KD["D"]))
    assert abs(errs[2]) < abs(errs[0])


@pytest.mark.parametrize("c", FLOW_CASES)
def test_flow_FAIL_cases(c):
    """Leading term only: error/ε drifts like K ln(1/ε) (not o(1)); dropping c₁ or J leaves an O(1) error/ε."""
    A, B, J, KD = _flow_pred(c)
    eps_list = (1e-4, 1e-6)
    sims = [FN.simulate_flow(e, c["p"], c["q"], c["f3"], c["f11"], c["g0"], c["g1"], c["xq"]) for e in eps_list]
    lead = [(s - FN.expansion(e, c["p"], c["q"], c["g0"], 0.0, 0.0)["sigma"]) / e for s, e in zip(sims, eps_list)]
    assert abs((lead[1] - lead[0]) - KD["K"] * math.log(100.0)) < 0.02          # the log coefficient, measured
    assert abs(lead[1]) > 0.05
    D_no_c1 = KD["D"] + (c["g0"] / c["q"]) * KD["c1"]
    bad = (sims[1] - FN.expansion(eps_list[1], c["p"], c["q"], c["g0"], KD["K"], D_no_c1)["sigma"]) / eps_list[1]
    assert abs(bad) > 0.05
    wrongK = (sims[1] - FN.expansion(eps_list[1], c["p"], c["q"], c["g0"], -KD["K"], KD["D"])["sigma"]) / eps_list[1]
    assert abs(wrongK) > 0.05


def test_map_formula_with_discrete_terms():
    h = 0.25
    c = {k: (h * v if k != "xq" else v) for k, v in FLOW_CASES[0].items()}
    A, B = FN.inner_AB(c["p"], c["q"], c["f3"], c["f11"], c["g0"], c["g1"], discrete=True)
    J = FN.J_map(c["q"], c["f3"], c["g0"], c["g1"], A, c["xq"])
    KD = FN.coefficients(c["p"], c["q"], c["g0"], A, B, J, discrete=True)
    Af, Bf = FN.inner_AB(c["p"], c["q"], c["f3"], c["f11"], c["g0"], c["g1"], discrete=False)
    KDf = FN.coefficients(c["p"], c["q"], c["g0"], Af, Bf, J, discrete=False)
    for eps in (1e-4, 1e-6):
        s = FN.simulate_map(eps, c["p"], c["q"], c["f3"], c["f11"], c["g0"], c["g1"], c["xq"])
        ok = (s - FN.expansion(eps, c["p"], c["q"], c["g0"], KD["K"], KD["D"])["sigma"]) / eps
        nohalf = (s - FN.expansion(eps, c["p"], c["q"], c["g0"], KD["K"], KD["D"] - KD["D_half_step"])["sigma"]) / eps
        flow = (s - FN.expansion(eps, c["p"], c["q"], c["g0"], KDf["K"], KDf["D"])["sigma"]) / eps
        assert abs(ok) < 0.02
        assert abs(nohalf - 0.5 * c["g0"]) < 0.02                                    # FAIL: the half step is real
        assert abs(flow) > 0.5                                                       # FAIL: flow A, B for a map


# ------------------------------------------------------------------------------------- the committed prediction
def _J():
    if not FN.OUT_JSON.exists():
        pytest.skip("results/fold_next_order.json not produced")
    return json.loads(FN.OUT_JSON.read_text())


def test_prediction_rows_reproduce_from_coefficients():
    d = _J()
    nf, KD = d["normal_form"], d["expansion_sigma"]
    A, B = FN.inner_AB(nf["p"], nf["q"], nf["f3"], nf["f11"], nf["g0"], nf["g1"], discrete=True)
    assert A == d["inner"]["A"] and B == d["inner"]["B"]
    KD2 = FN.coefficients(nf["p"], nf["q"], nf["g0"], A, B, d["escape"]["J"], discrete=True)
    assert KD2 == KD
    s_F = d["frozen_inputs"]["s_F"]
    for r in d["rows"]:
        ex = FN.expansion(r["rho"], nf["p"], nf["q"], nf["g0"], KD["K"], KD["D"])
        assert r["r_pred"] == ex["sigma"] / s_F and r["slope_pred"] == ex["slope"]
    assert len(d["rows"]) == 27


def test_prediction_landscape_consistency():
    d = _J()
    nf = d["normal_form"]
    assert abs(d["fold"]["s_F_minus_frozen"]) < 1e-12
    assert abs(nf["g1_over_3p"] - 1) < 1e-9                       # G's slope along e is 3p (gradient symmetry)
    assert abs(nf["abs_mc"] / d["frozen_inputs"]["abs_mc_frozen"] - 1) < 0.02
    assert nf["hessian_eigs"][0] < 1e-12 < nf["hessian_eigs"][1]
    qd = d["escape"]["quartic_diagnostic"]                       # f₃ and g₁ seen along the escape run
    f3_lin = (qd[2]["x"] * qd[0]["f3_est"] - qd[0]["x"] * qd[2]["f3_est"]) / (qd[2]["x"] - qd[0]["x"])
    assert abs(f3_lin / nf["f3"] - 1) < 0.02
    assert abs(qd[0]["g1_est"] / nf["g1"] - 1) < 1e-3
    assert d["escape"]["J_numerical_uncertainty"] < 0.5


def test_prediction_reads_no_observation():
    d = _J()
    text = json.dumps(d)
    for key in ("r_obs", "s_obs", "t_obs", "\"observed"):
        assert key not in text
    src = FN.__file__ and open(FN.__file__).read().split("def compare")[0]
    for name in ("observed.jsonl", "runs.jsonl", "scores.json", "phase2a_posthoc"):
        assert name not in src
