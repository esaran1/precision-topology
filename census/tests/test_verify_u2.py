"""check_u2_unit (src/verify_certificates.py, Track U, U2): the ε-interval activation against 60-digit values, including
ε-balls that CONTAIN 0 (series in v = εσ², never through √ε); NaN propagation; slab closedness; and the outer / ring / PD /
switch checks on constructed pass, fail and undecided cases."""

import math

import numpy as np
import pytest

flint = pytest.importorskip("flint")
from flint import arb
from mpmath import mp

import src.verify_certificates as vc


def _exact(sig, eps, which):
    """φ_ε and its σ- and ε-derivatives at 60 digits from the ε-power series
    φ_ε(σ) = −σ + (1+ε) Σ_m (−1)^m ε^m σ^{2m+3}/(2m+3)!   (entire in ε; valid at ε = 0, where it is h),
    cross-checked against the closed form (u − (1+ε) sin u)/ε^{3/2} in test_series_truth_matches_the_closed_form."""
    with mp.workdps(60):
        s, e = mp.mpf(sig), mp.mpf(eps)
        tot = mp.mpf(0)
        for m in range(160):
            if which == "f":
                c = (-1) ** m * s ** (2 * m + 3) / mp.factorial(2 * m + 3); g = e ** m * (1 + e)
            elif which == "df":
                c = (-1) ** m * s ** (2 * m + 2) / mp.factorial(2 * m + 2); g = e ** m * (1 + e)
            elif which == "d2f":
                c = (-1) ** m * s ** (2 * m + 1) / mp.factorial(2 * m + 1); g = e ** m * (1 + e)
            elif which == "de":
                c = (-1) ** m * s ** (2 * m + 3) / mp.factorial(2 * m + 3)
                g = (m * e ** (m - 1) if m else 0) + (m + 1) * e ** m
            else:
                c = (-1) ** m * s ** (2 * m + 3) / mp.factorial(2 * m + 3)
                g = (m * (m - 1) * e ** (m - 2) if m > 1 else 0) + ((m + 1) * m * e ** (m - 1) if m else 0)
            tot += c * g
        base = {"f": -s, "df": mp.mpf(-1)}.get(which, mp.mpf(0))
        return base + tot


def test_series_truth_matches_the_closed_form():
    with mp.workdps(60):
        for s, e in ((3.7, 0.3), (-12.0, 0.6), (0.4, 1e-3)):
            u = mp.sqrt(mp.mpf(e)) * mp.mpf(s)
            closed = (u - (1 + mp.mpf(e)) * mp.sin(u)) / mp.mpf(e) ** 1.5
            assert abs(closed - _exact(s, e, "f")) < mp.mpf(10) ** -30 * max(1, abs(closed))


def _in(ball, val):
    lo, hi = float(ball.lower()), float(ball.upper())
    v = float(val)
    tol = 1e-12 * max(1.0, abs(v))
    return lo - tol <= v <= hi + tol


@pytest.mark.parametrize("which", ["f", "df", "d2f", "de", "dee"])
def test_activation_encloses_60_digit_values_on_balls_containing_zero(which):
    rng = np.random.default_rng(3)
    for _ in range(25):
        e_hi = float(10 ** rng.uniform(-6, math.log10(0.6)))
        s_lo = float(rng.uniform(-15, 15)); s_hi = s_lo + float(rng.uniform(0, 0.2))
        act = vc.U2Act(vc._ival(0.0, e_hi))                     # ε-ball containing 0
        b = getattr(act, which)(vc._ival(s_lo, s_hi))
        assert b.is_finite()
        for e in (0.0, e_hi * 1e-3, 0.5 * e_hi, e_hi):
            for s in (s_lo, 0.5 * (s_lo + s_hi), s_hi):
                assert _in(b, _exact(s, e, which)), (which, e, s)


def test_activation_at_eps_zero_is_the_limit_profile():
    act = vc.U2Act(arb(0))
    for s in (-3.1, -0.4, 0.0, 1.0, 2.5, 7.0):
        assert _in(act.f(arb(s)), -s + s ** 3 / 6)
        assert _in(act.de(arb(s)), s ** 3 / 6 - s ** 5 / 120)               # φ_ε at ε = 0 is r(σ)


def test_activation_never_takes_sqrt_of_eps():
    """A ball reaching slightly negative ε (rounding of a ball at 0) stays finite: no √ε is ever formed."""
    act = vc.U2Act(arb(0, 1e-6))
    for t in (arb(2.0), vc._ival(-3, 3), vc._ival(0.1, 0.2)):
        for fn in (act.f, act.df, act.d2f, act.de, act.dee):
            assert fn(t).is_finite()


def test_closed_form_and_series_agree_across_v_equal_one():
    for name in ("Gt", "J", "S", "K1", "K2"):
        a = vc._kernel_series(name, arb(1))
        b = vc._kernel_closed(name, arb(1))
        assert a.overlaps(b)
        c = vc.u2_kernel(name, vc._ival(0.5, 3.0))
        for v in (0.5, 1.0, 2.0, 3.0):
            assert c.overlaps(vc.u2_kernel(name, arb(v)))
        a4, b4 = vc._kernel_series(name, arb(400)), vc._kernel_closed(name, arb(400))
        assert a4.overlaps(b4)                                  # the series/closed-form switch point
        c = vc.u2_kernel(name, vc._ival(350.0, 450.0))           # a ball straddling it is split
        for v in (350.0, 400.0, 450.0):
            assert c.overlaps(vc.u2_kernel(name, arb(v)))


def test_nan_propagates_to_a_conservative_range():
    act = vc.U2Act(arb("inf"))
    lo, hi = act.range(arb(1), arb(2))
    assert not lo.is_finite() and not hi.is_finite()


def test_slabs_closed_and_contiguous():
    assert vc.u2_slabs_closed_ok([(0.0, 0.005), (0.005, 0.01), (0.01, 0.02)])
    assert not vc.u2_slabs_closed_ok([(0.0, 0.005), (0.0050000001, 0.01)])          # a gap
    assert not vc.u2_slabs_closed_ok([(0.0, 0.005), (0.004, 0.01)])                 # an overlap, endpoints not shared
    assert not vc.u2_slabs_closed_ok([(0.01, 0.01)])
    assert not vc.u2_slabs_closed_ok([])


# ------------------------------------------------------------------ outer / ring / PD / switch on constructed cases
@pytest.fixture(scope="module")
def xy():
    from src.limit_bnb import population
    return population()


@pytest.fixture(scope="module")
def branch(xy):
    from src.u2_sample import branch_point              # the test (not the checker) may use the producer to set up data
    x, y = xy
    A_lo, A_hi = 0.6850, 0.68625
    c = branch_point(0.5 * (A_lo + A_hi), 5e-4, x, y)
    return tuple(float(v) for v in c), A_lo, A_hi


def test_outer_pass_on_a_region_outside_the_box(xy, branch):
    x, y = xy
    c, A_lo, A_hi = branch
    r = vc.u2_outer_arb(x, y, c, A_lo, A_hi, 0.0, 1e-3, "R", region=(2.0, 2.5, 1.0, 1.5), h0=0.25, max_depth=4)
    assert r["verdict"] == "pass", r


def test_outer_fail_with_a_shifted_centre(xy, branch):
    """θ_c moved by 0.3: the true branch lies outside the box; a witness that beats the shifted 'branch' is found."""
    x, y = xy
    c, A_lo, A_hi = branch
    r = vc.u2_outer_arb(x, y, (c[0] + 0.3, c[1]), A_lo, A_hi, 0.0, 1e-3, "R", region=(1.5, 1.75, 1.25, 1.5), h0=0.25,
                        max_depth=6)
    assert r["verdict"] == "fail" and r["witness"] is not None


def test_outer_unresolved_within_a_depth_cap(xy, branch):
    x, y = xy
    c, A_lo, A_hi = branch
    r = vc.u2_outer_arb(x, y, c, A_lo, A_hi, 0.0, 1e-3, "R", region=(1.75, 2.0, 1.25, 1.5), h0=0.25, max_depth=0,
                        witness_from_depth=99)
    assert r["verdict"] == "unresolved"


def test_ring_pass_and_fail(xy, branch):
    x, y = xy
    c, A_lo, A_hi = branch
    ok = vc.u2_ring_arb(x, y, c, A_lo, A_hi, 0.0, 1e-3, boxes=[(0.1, 0.0, 0.00625), (-0.1, 0.05, 0.00625)])
    assert ok["certified"]
    # a 'ring' box that contains the critical point itself (rho_in = 0): never certifiable
    bad = vc.u2_ring_arb(x, y, c, A_lo, A_hi, 0.0, 1e-3, rho_in=0.0, boxes=[(0.0, 0.0, 0.002)], max_depth=2)
    assert not bad["certified"]


def test_pd_pass_and_fail(xy, branch):
    x, y = xy
    c, A_lo, A_hi = branch
    p0, q0 = c
    r = vc.u2_pd_box_arb(x, y, p0 - 0.0125, p0 + 0.0125, q0 - 0.0125, q0 + 0.0125, A_lo, A_hi, 0.0, 1e-3)
    assert r["b_ok"] and r["pd"]
    r = vc.u2_pd_box_arb(x, y, p0 - 0.0125, p0 + 0.0125, q0 - 0.0125, q0 + 0.0125, A_lo, A_hi, 0.0, 1e-3, c=1.0)
    assert not r["pd"]                                      # λ_min is about 0.04, so H − 1·I is not PD


def test_switch_krawczyk_pass_and_fail(xy):
    from src.u2_sample import switch_float
    x, y = xy
    vs, b = switch_float(5e-5, x, y, [1.668, 1.369, 0.6855])
    zf = [float(vs[0]), float(vs[1]), float(b), float(vs[2])]
    xs = [arb(float(v)) for v in x]; ys = [int(v) for v in y]
    act = vc.U2Act(vc._ival(0.0, 1e-4))                      # an ε-ball containing 0
    fun = lambda X, Jf=True: vc.u2_system_arb(X, xs, ys, act, with_gap=True, want_J=Jf)
    ok, K, X, _ = vc.krawczyk_arb(fun, zf, [6e-4, 6e-4, 2e-4, 5e-4])
    assert ok
    # undecided: the same radii over the ten-times wider ε-ball [0, 1e-3] (the switch moves by about 4.5e-4 in A)
    act_w = vc.U2Act(vc._ival(0.0, 1e-3))
    fun_w = lambda X, Jf=True: vc.u2_system_arb(X, xs, ys, act_w, with_gap=True, want_J=Jf)
    assert not vc.krawczyk_arb(fun_w, zf, [6e-4, 6e-4, 2e-4, 5e-4])[0]
    valid, G = vc.u2_active_pair(X[0], X[1], act)
    assert valid and G.contains(0)
    shifted = [zf[0] + 5e-3, zf[1], zf[2], zf[3]]
    ok2, *_ = vc.krawczyk_arb(fun, shifted, [6e-4, 6e-4, 2e-4, 5e-4])
    assert not ok2
