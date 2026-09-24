import math

import numpy as np
import pandas as pd

from src.scale_limits import _data, _phi_w2, dmu_w2
from src.scale_limits_tanh import _unpack, analytic, dmu_q, verdict
from src.width2_geometry import Act, gaps

ACT = Act("tanh")
X, Y, XO, XI = _data()


def test_first_order_tie_and_signs():
    for A in (5.0, 20.0):
        (ps, ss), (pp, sp) = analytic(A, ACT, X, Y, XO, XI, "single"), analytic(A, ACT, X, Y, XO, XI, "pair")
        ds, dp = dmu_w2(ps, ss, ACT, XO, XI)[0], dmu_w2(pp, sp, ACT, XO, XI)[0]
        assert abs(ds - dp) < 1e-12 and ds < 1
        Gs = gaps(ps[:4], _phi_w2(ps, ss, ACT, X)[1], ACT)["G+"]
        Gp = gaps(pp[:4], _phi_w2(pp, sp, ACT, X)[1], ACT)["G+"]
        assert Gs[1] < 0 < Gp[0]
        assert np.var(_phi_w2(pp, sp, ACT, X)[0]) < np.var(_phi_w2(ps, ss, ACT, X)[0])


def test_supremum_rises_toward_one():
    v = [dmu_w2(*analytic(A, ACT, X, Y, XO, XI, "single"), ACT, XO, XI)[0] for A in (5.0, 10.0, 20.0, 40.0)]
    assert all(b > a for a, b in zip(v, v[1:])) and v[-1] < 1 and 1 - v[-1] < 1e-5


def test_chain_rule_gradient():
    rng = np.random.default_rng(1)
    q = np.r_[rng.uniform(-1, 1), rng.uniform(-3, 3), rng.uniform(-1, 1), rng.uniform(-3, 3), 0.3]
    f, g = dmu_q(q, -1, 10.0, ACT, XO, XI)
    h = 1e-6
    for i in range(5):
        e = np.zeros(5); e[i] = h
        fd = (dmu_q(q + e, -1, 10.0, ACT, XO, XI)[0] - dmu_q(q - e, -1, 10.0, ACT, XO, XI)[0]) / (2 * h)
        assert abs(fd - g[i]) < 1e-6
    assert abs(_unpack(np.array([math.pi / 2, 0, 0, 0, 0]), 7.0)[0] - 7.0) < 1e-15


def _table(top, onb, Glo, Ghi=None, minr=None, ok=True):
    n = len(top)
    return pd.DataFrame({"top": top, "maximisers_on_boundary": onb, "selected_G_lo": Glo,
                         "selected_G_hi": Ghi if Ghi is not None else [g + 1e-6 for g in Glo],
                         "min_max_abs_alpha_over_A": minr if minr is not None else [1.0] * n,
                         "ladder_ok": [ok] * n, "independent_ok": [True] * n, "analytic_attain": [True] * n})


def test_verdict_cases():
    assert verdict(_table([0.9, 0.99, 0.9999, 0.99999999], [True] * 4, [0.5, 0.8, 0.9, 0.95])).startswith("T1")
    assert verdict(_table([0.9, 0.95, 0.96, 0.96], [True, True, False, False], [0.5] * 4,
                          minr=[1, 1, 0.5, 0.5])).startswith("STOP (T2)")
    assert verdict(_table([0.9, 0.99, 0.9999, 0.99999999], [True] * 4, [0.5, -0.1, 0.9, 0.95],
                          Ghi=[0.6, -0.05, 1, 1])).startswith("STOP: Var-selected")
    assert verdict(_table([0.9, 0.99, 0.9999, 0.99999999], [True] * 4, [0.5] * 4, ok=False)).startswith("STOP: valid")
    assert verdict(_table([0.9, 0.99, 0.9999, 0.99999999], [True, True, True, False], [0.5] * 4)) == "neither"
