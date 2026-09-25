import math

import numpy as np
import pandas as pd
import pytest

from src import harsh_review_a as h


def _adam_ratio(g, b1=0.9, b2=0.999):
    m = v = 0.0
    out = []
    for t, gt in enumerate(g, 1):
        m = b1 * m + (1 - b1) * gt; v = b2 * v + (1 - b2) * gt * gt
        out.append(abs(m / (1 - b1 ** t)) / math.sqrt(v / (1 - b2 ** t)))
    return np.array(out)


def test_adam_bound_holds_on_random_and_adversarial_sequences():
    _, B, Binf = h.adam_bound(600)
    rng = np.random.default_rng(0)
    for _ in range(50):                                        # pass cases: never above B_t
        g = rng.standard_normal(600) * np.exp(rng.uniform(-3, 3, 600))
        assert (_adam_ratio(g) <= B + 1e-12).all()
    got, want = h.adam_worst_case_check(t=300)                 # the equality case attains it
    assert abs(got - want) < 1e-9
    assert B.max() < Binf


def test_adam_bound_fails_if_bias_correction_dropped():
    # constructed fail case: the uncorrected ratio at t = 1 is (1-b1)/sqrt(1-b2) = 3.16 > B_1 = 1
    _, B, _ = h.adam_bound(1)
    assert B[0] == pytest.approx(1.0) and (1 - 0.9) / math.sqrt(1 - 0.999) > B[0]


def test_sup_abs_on_covers_true_max_and_undercovers_with_wrong_lipschitz():
    f = lambda a: np.cos(a * 40.0)                             # |f'| <= 40
    assert h.sup_abs_on(f, 1.0, 40.0, h=1e-2) >= 1.0          # pass: Lipschitz step covers the grid gap
    g = lambda a: np.where(np.abs(a - 0.5034) < 1e-4, 2.0, 0.0)   # narrow spike between grid points
    assert h.sup_abs_on(g, 1.0, 0.0, h=1e-2) < 2.0            # fail: a wrong (zero) Lipschitz constant misses it


def test_continuous_D_closed_form_matches_quadrature_and_detects_wrong_window():
    gl, gw = np.polynomial.legendre.leggauss(200)
    q = lambda f, lo, hi: 0.5 * (hi - lo) * (gw * f(0.5 * (hi - lo) * gl + 0.5 * (hi + lo))).sum()
    for al in (0.3, 1.79, 5.0):
        D = q(lambda x: np.cos(al * x), 1.2, 2.0) / 0.8 - q(lambda x: np.cos(al * x), -0.8, 0.8) / 1.6
        assert abs(D - h.D_c(al)[0]) < 1e-13
        Dw = q(lambda x: np.cos(al * x), 1.2, 2.2) / 1.0 - q(lambda x: np.cos(al * x), -0.8, 0.8) / 1.6
        assert abs(Dw - h.D_c(al)[0]) > 1e-4                 # a different window is detected


def test_domain_bound_on_random_parameters():
    x, y, xO, xI = h._data()
    rng = np.random.default_rng(1)
    for a in (1.3, 1.5, 2.0):
        for _ in range(2000):
            w, b = rng.uniform(-5, 5), rng.uniform(0, 2 * math.pi)
            phi = w * x + b + a * np.sin(w * x + b)
            G = max(phi[y == 1].min() - phi[y == 0].max(), phi[y == 0].min() - phi[y == 1].max())
            assert G <= 2 * a * abs(math.sin(1.4 * w)) - 2.8 * abs(w) + 1e-12


def test_census_relevance_rejects_unclassified_block(monkeypatch):
    from src import census_relevance as cr
    monkeypatch.setattr(cr, "BLOCKS", {k: v for k, v in cr.BLOCKS.items() if k != "B"})
    with pytest.raises(RuntimeError):
        cr.build()
    monkeypatch.undo()
    d, _ = cr.build()
    assert len(d) == len(pd.read_csv(cr.RESULTS / "registration_census.csv"))
