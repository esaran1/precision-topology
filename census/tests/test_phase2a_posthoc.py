"""POST HOC Phase 2A delay-law fit (src/phase2a_posthoc.py): constructed cases."""
import json

import numpy as np
import pytest

import src.phase2a_posthoc as PH

OM = PH.OMEGA0
EPS = 1.77e-3 * 2.0 ** -np.r_[0.0, 1.0, 2.0, np.arange(2.125, 5.01, 0.125)]     # 27 points over 5 octaves


def _ks(eps, A=OM, C=0.8):
    return A * eps ** (2 / 3) + C * eps * np.log(1 / eps)


def test_recovers_ks_form():
    r = _ks(EPS)
    f = PH.fit_linear(EPS, r, ("e23", "eL"))
    assert f["coef"]["e23"] == pytest.approx(OM, rel=1e-10)
    assert f["coef"]["eL"] == pytest.approx(0.8, rel=1e-8)
    assert f["max_abs_rel"] < 1e-12


def test_fixed_A_recovers_C():
    r = _ks(EPS, C=-0.3)
    f = PH.fit_linear(EPS, r, ("eL",), (("e23", OM),))
    assert f["coef"]["eL"] == pytest.approx(-0.3, rel=1e-9) and f["coef"]["e23"] == OM


def test_three_term_recovers_D():
    r = _ks(EPS) - 1.2 * EPS
    c = PH.fit_linear(EPS, r, ("e23", "eL", "e"))["coef"]
    assert c["e23"] == pytest.approx(OM, rel=1e-7) and c["eL"] == pytest.approx(0.8, rel=1e-6)
    assert c["e"] == pytest.approx(-1.2, rel=1e-5)


def test_leading_only_misfits_ks_data():
    r = _ks(EPS)
    f = PH.fit_linear(EPS, r, ("e23",))
    assert f["coef"]["e23"] > OM and f["rms_rel"] > 1e-2


def test_model_local_slope_analytic():
    c = {"e23": OM, "eL": 0.8}
    e = np.array([1e-3, 1e-5, 1e-8])
    num = np.gradient(np.log(_ks(np.exp(np.log(e)[:, None] + np.r_[-1e-6, 0, 1e-6]))), 1e-6, axis=1)[:, 1]
    assert np.allclose(PH.model_local_slope(c, e), num, rtol=1e-7)
    # closed form: 2/3 + C ε (L/3 − 1) / r
    L = np.log(1 / e)
    assert np.allclose(PH.model_local_slope(c, e), 2 / 3 + 0.8 * e * (L / 3 - 1) / _ks(e))
    assert np.all(PH.model_local_slope({"e23": 2.0}, e) == pytest.approx(2 / 3))


def test_local_slope_approaches_two_thirds_from_above():
    c = {"e23": OM, "eL": 0.8}
    s = PH.model_local_slope(c, np.logspace(-4, -14, 30))
    assert np.all(s > 2 / 3) and np.all(np.diff(s) < 0) and s[-1] - 2 / 3 < 1e-3
    e1 = PH.eps_slope_within(c, 0.01)
    assert abs(PH.model_local_slope(c, e1) - 2 / 3) <= 0.01
    assert abs(PH.model_local_slope(c, e1 * 3) - 2 / 3) > 0.01
    assert PH.eps_slope_within({"e23": 1.0}, 0.01) == pytest.approx(0.1)


def test_finite_difference_and_window_slopes_of_power_law():
    r = 3.0 * EPS ** 0.7
    fd = PH.finite_difference_slopes(EPS, r)
    assert np.allclose(fd["slope"], 0.7) and len(fd["slope"]) == len(EPS) - 1
    w = PH.window_slopes(EPS, r, 5)
    assert np.allclose(w["slope"], 0.7) and len(w["slope"]) == len(EPS) - 4
    assert w["eps_mid"][0] == pytest.approx(np.exp(np.log(EPS[:5]).mean()))


def test_window_slopes_of_ks_track_analytic_and_fall():
    r = _ks(EPS)
    w = PH.window_slopes(EPS, r, 5)
    an = PH.model_local_slope({"e23": OM, "eL": 0.8}, np.array(w["eps_mid"]))
    assert np.max(np.abs(np.array(w["slope"]) - an)) < 2e-3
    assert np.all(np.diff(w["slope"][2:]) < 0) and min(w["slope"]) > 2 / 3


def test_sdot_cubic_exact_and_duplicate_point():
    # t(s) = 1000 s + 50 s^2 + 3 s^3 → ṡ at s = 1 is 1/(1000 + 100 + 9)
    t = lambda s: 1000 * s + 50 * s ** 2 + 3 * s ** 3
    lv = [0.8, 0.874, 0.95, 1.0]
    assert PH.sdot_cubic(lv, [t(x) for x in lv]) == pytest.approx(1 / 1109, rel=1e-9)
    # a repeated first-passage step (f_tight = 0.95): dropped, quadratic through three → exact for a quadratic
    q = lambda s: 1000 * s + 50 * s ** 2
    lv2 = [0.8, 0.95, 0.95 + 1e-6, 1.0]
    tt = [q(0.8), q(0.95), q(0.95), q(1.0)]
    assert PH.sdot_cubic(lv2, tt) == pytest.approx(1 / 1100, rel=1e-4)


def test_beta_profile_recovers_beta():
    r = 2.0 * EPS ** (2 / 3) * (1 + 3.0 * EPS ** 0.5)
    bp = PH.fit_beta_profile(EPS, r)
    assert bp["beta"] == pytest.approx(0.5) and bp["A"] == pytest.approx(2.0, rel=1e-8)
    assert bp["B"] == pytest.approx(3.0, rel=1e-6)


def test_committed_page_regenerates():
    if not PH.OUT_JSON.exists():
        pytest.skip("not produced")
    J = json.loads(PH.OUT_JSON.read_text())
    assert J == json.loads(json.dumps(PH.build()))
    assert PH.OUT_MD.read_text() == PH.render_md(J)
    assert PH.OUT_MD.read_text().startswith("# Phase 2A POST HOC")
    assert len(J["rows"]) == 27 and sum(x["status"] == "scored" for x in J["rows"]) == 22
