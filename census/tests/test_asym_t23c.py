import math

from src import asym_t23c as tc


def test_choose_phi_cases():
    f = lambda phi: 3.0 * phi ** 3                 # increasing; target 0.0057 at phi ~ 0.124
    phi, m = tc.choose_phi(f)
    assert phi is not None and abs(m / 0.0057 - 1) <= 0.2
    assert tc.choose_phi(lambda phi: 0.0)[0] is None            # never reaches the target: no phi
    assert tc.choose_phi(lambda phi: 10.0)[0] is None           # always above: no phi


def test_budget_and_verdict():
    assert tc.budget_from([1000] * 40) == 32_000
    assert tc.budget_from([30_000] * 40) == 45_000
    good = [1.1] * 60
    assert tc.verdict([0.005] * 60, good, 0.99, 1.0, 60)[0] == "PASS"
    assert tc.verdict([0.005] * 60, [3.0] * 60, 0.99, 1.0, 60)[0] == "FAIL"
    assert tc.verdict([2.7] * 60, good, 0.99, 1.0, 60)[0].startswith("UNRESOLVED")      # ratio outside
    assert tc.verdict([0.005] * 30, good[:30], 0.99, 1.0, 30)[0].startswith("UNRESOLVED")  # < 40 crossings
