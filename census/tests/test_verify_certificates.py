"""The independent certificate checker (src/verify_certificates.py): its primitives against dense sampling and
high precision, the exact-tiling check, and constructed cases it must reject, including one a float-only check
would accept."""

import math

import numpy as np
import pytest

flint = pytest.importorskip("flint")
from flint import arb

import src.verify_certificates as vc


@pytest.fixture(scope="module")
def obj():
    from src.conditional_certified import _population
    x, y = _population()
    return vc.Objective(x, y, 4.95, 1.30)


def test_fa_range_encloses_dense_samples():
    fa = vc.FA(1.30)
    rng = np.random.default_rng(0)
    for _ in range(200):
        lo, hi = np.sort(rng.uniform(-20, 20, 2))
        mn, mx = fa.range(arb(lo), arb(hi))
        t = np.linspace(lo, hi, 20001); f = t + 1.30 * np.sin(t)
        assert float(mn.mid()) <= f.min() + 1e-12 and float(mx.mid()) >= f.max() - 1e-12


def test_coverage_accepts_an_exact_tiling_and_rejects_gaps_and_overlaps():
    lv = np.array([0, 1, 1, 1, 1]); iw = np.array([0, 2, 2, 3, 3]); ib = np.array([0, 0, 1, 0, 1])
    ok, _ = vc.coverage_ok(lv, iw, ib, nw=2, nb=1)                # cell (0,0) + the four children of (1,0)
    assert ok
    assert not vc.coverage_ok(lv[:-1], iw[:-1], ib[:-1], 2, 1)[0]                           # a gap
    assert not vc.coverage_ok(np.r_[lv, 0], np.r_[iw, 1], np.r_[ib, 0], 2, 1)[0]            # parent + children


def test_convexity_bound_at_the_centre_is_below_the_profiled_value(obj):
    """L*(c) ≥ g(b̂) − |F(b̂)|·|b* − b̂| must hold; check against the float profile at random centres."""
    from src.profiled_bnb import profile
    rng = np.random.default_rng(1)
    for _ in range(10):
        w, b = rng.uniform(-2.5, 2.5), rng.uniform(0, 2 * math.pi)
        lb = obj.lower_bound(w, b, 0.0, 0.0)
        L = profile(np.array([w]), np.array([b]), 4.95, 1.30, obj.xf, obj.yf)[0][0]
        assert float(lb.lower().mid()) <= L + 1e-13


def test_rejects_a_claim_that_float_arithmetic_accepts(obj):
    """Constructed case: at a point where the float evaluation of L(c, b̂) rounds ABOVE the true value, the claim
    'L* ≥ fl(L)' is accepted by a float-only check (the float value equals the claim) but is false; the rigorous
    check must reject it, and must accept a claim 1e−12 lower."""
    from src.profiled_bnb import profile
    rng = np.random.default_rng(2)
    for _ in range(400):
        w, b = rng.uniform(-2.5, 2.5), rng.uniform(0, 2 * math.pi)
        L, b2, _, _, _ = profile(np.array([w]), np.array([b]), 4.95, 1.30, obj.xf, obj.yf)
        fl = float(L[0])
        true = obj.upper_at(w, b, float(b2[0]))                 # ball containing L(c, b̂) ≥ L*(c)
        if arb(fl) > true:                                      # float value lies above the true L(c, b̂) ≥ L*(c)
            break
    else:
        pytest.skip("no upward-rounded point found")
    assert fl >= fl                                             # the float-only check passes (claim == float value)
    st = {"evals": 0, "max_depth": 0}
    assert not vc._leaf_ok(obj, w, b, 0.0, 0.0, "lb", fl, vc.MAX_DEPTH, st)      # rigorous: rejected
    assert vc._leaf_ok(obj, w, b, 0.0, 0.0, "lb", fl - 1e-12, vc.MAX_DEPTH, st)  # a true claim: accepted


def test_rejects_a_false_region_claim(obj):
    """A cell containing a point with G > 0 cannot be certified as outside {G > 0}."""
    st = {"evals": 0, "max_depth": 0}
    # the certified R_glob branch argmin at a = 1.30 has G ≈ 0; a nearby point on the placed side has G > 0
    assert not vc._leaf_ok(obj, 0.8337, 3.832, 1e-2, 1e-2, "out+", None, 2, st)


def test_lemma_matches_the_search_bound():
    from src.conditional_certified import _population
    from src.profiled_bnb import w_bound
    x, y = _population()
    for a, s in ((1.30, 4.95), (1.50, 2.525), (1.60, 2.0)):
        assert abs(vc.lemma_W(x, y, s, a) - w_bound(s, a, x, y)) <= 1e-12
