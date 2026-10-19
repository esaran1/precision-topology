"""Item 3 in the producer (src/u2_sample.py): the sheared switch and the first-order ring on ε-intervals that contain
0, with constructed pass / undecided cases, and the sheared gap's sign on rows left and right of the switch."""

import pytest

from src import u2_sample as m


@pytest.fixture(scope="module")
def xy():
    from src.limit_bnb import population
    return population()


def test_sheared_switch_pass_and_undecided(xy):
    x, y = xy
    vs, J = m.tube(5e-4, x, y)
    Jn = (vs[2] - 0.003, vs[2] + 0.003)
    assert m.switch_unit_shear(0.0, 1e-3, Jn, x, y, vs)["unique_sign_change"]
    r = m.switch_unit_shear(0.0, 1e-2, Jn, x, y, vs)                  # too wide: Z does not contract
    assert not r["unique_sign_change"]


def test_sheared_gap_sign_left_and_right(xy):
    x, y = xy
    vs, J = m.tube(5e-4, x, y)
    for dA, sign in ((-0.004, -1), (0.004, 1)):
        A = vs[2] + dA
        v = m.branch_point(A, 5e-4, x, y, start=(vs[0], vs[1]))
        _, b, *_ = m.point_eval([v[0]], [v[1]], A, 5e-4, x, y)
        xt = [v[0], v[1], b[0]]
        ok, R, info = m.krawczyk_modes(xt, [1e-7] * 3, 0.0, 1e-3, x, y, "shear", A_iv=m._iv(A - 1e-5, A + 1e-5),
                                       with_gap=False)
        assert ok
        lo, hi = m.gap_sheared(xt, R, info["X1"], 0.0, 1e-3)
        assert (hi < 0) if sign < 0 else (lo > 0)


def test_first_order_ring_passes_where_lipschitz_fails(xy):
    x, y = xy
    vs, J = m.tube(5e-3, x, y)
    w = (J[1] - J[0]) / 40
    A_lo, A_hi = J[0] + 20 * w, J[0] + 21 * w
    c = m.branch_point(0.5 * (A_lo + A_hi), 5e-3, x, y, start=(vs[0], vs[1]))
    _, b, *_ = m.point_eval([c[0]], [c[1]], 0.5 * (A_lo + A_hi), 5e-3, x, y)
    assert not m.ring_unit(0.0, 1e-2, A_lo, A_hi, tuple(c), x, y)["ring_ok"]
    assert m.ring3d((c[0], c[1], b[0]), A_lo, A_hi, 0.0, 1e-2, x, y, "mv")["certified"]
