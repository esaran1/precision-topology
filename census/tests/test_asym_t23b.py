import math

from src import asym_t23b as tb


def test_choose_phi_pass_and_fail():
    f = lambda phi: 3000.0 / phi ** 0.5                       # steps fall as phi grows
    phi, m = tb.choose_phi(9000.0, f)
    assert phi is not None and abs(m / 9000 - 1) <= 0.02 and abs(phi - (1 / 3) ** 2) / (1 / 9) < 0.05
    assert tb.choose_phi(1000.0, f)[0] is None                # width-1 faster than width 2 at phi = 1: no phi
    assert tb.choose_phi(1e9, f)[0] is None                   # slower than any phi in range: no phi


def test_validity_check():
    assert tb.validity([0.005] * 50)[0]
    assert not tb.validity([2.7] * 50)[0]
    assert not tb.validity([0.0005] * 50)[0]
    assert not tb.validity([float("nan"), -1.0])[0]
