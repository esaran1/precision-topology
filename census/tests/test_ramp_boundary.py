import numpy as np

from src import ramp_boundary as B


def _cells(ratio_by_chi, n=40, chi_scale=1.0, n_cross=40):
    out = []
    for a in (1.3, 1.5):
        for c in B.CHI_TARGETS:
            out.append({"a": a, "chi_target": c, "ratios": list(np.full(n_cross, ratio_by_chi[c])), "n_cross": n_cross,
                        "chi_own_median": c * chi_scale})
    return out


def test_pass_pass_and_chi_star():
    S = B.score_cells(_cells({0.03: 1.0, 0.06: 1.1, 0.1: 1.2, 0.2: 1.4, 0.4: 2.0}))
    assert S["B1"] == "PASS" and S["B2"] == "PASS" and S["chi_star"][1.3] == 0.2


def test_fail_cases():
    assert B.score_cells(_cells({0.03: 1.0, 0.06: 1.4, 0.1: 1.5, 0.2: 1.6, 0.4: 2.0}))["B1"] == "FAIL"
    S = B.score_cells(_cells({0.03: 1.0, 0.06: 1.0, 0.1: 1.0, 0.2: 1.0, 0.4: 1.1}))
    assert S["B2"] == "FAIL" and S["chi_star"][1.3] is None


def test_unresolved_by_chi_and_crossings():
    assert B.score_cells(_cells({0.03: 1, 0.06: 1, 0.1: 1, 0.2: 1, 0.4: 2}, chi_scale=1.4))["B1"] == "UNRESOLVED"
    S = B.score_cells(_cells({0.03: 1, 0.06: 1, 0.1: 1, 0.2: 1, 0.4: 2}, n_cross=20))
    assert S["B1"] == "UNRESOLVED" and S["B2"] == "UNRESOLVED"


def test_design_is_a_priori():
    d = B.design()
    assert len(d) == 10
    assert np.allclose(d.gamma / (0.3 * d.lambda_min_pop), d.chi_target)
