"""§21 (Track K): the ball-arithmetic and exact-rational pieces of the small-scale compactness proof, on constructed
pass and fail cases, and the committed numbers."""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import arb

from src import small_scale as ss
from src import small_scale_k as sk

ROOT = Path(__file__).resolve().parents[1]


def _Dc_float(al):
    return (math.sin(2 * al) - math.sin(1.2 * al) - math.sin(0.8 * al)) / (0.8 * al)


def test_Dc_enclosure_contains_closed_form_and_quadrature():
    gx, gw = np.polynomial.legendre.leggauss(200)
    rng = np.random.default_rng(1)
    for al in rng.uniform(0.05, 10, 40):
        enc = sk.Dc(arb(float(al)))
        assert enc.lower() <= _Dc_float(al) + 1e-14 and _Dc_float(al) - 1e-14 <= enc.upper()
        xi = 0.8 * gx; xo = 0.4 * gx + 1.6                       # nodes on I = [-0.8, 0.8] and [1.2, 2]
        quad = (np.cos(al * xo) * gw).sum() / 2 - (np.cos(al * xi) * gw).sum() / 2
        assert abs(quad - _Dc_float(al)) < 1e-12


def test_small_alpha_bound():
    for al in np.linspace(1e-4, 0.5, 2000):
        assert abs(_Dc_float(al)) <= float(sk.Dc_small_bound(Fraction(float(al)))) + 1e-15
    assert sk.Dc_small_bound(Fraction(1, 2)) == Fraction(8, 15)


def test_decay_bound():
    for al in np.linspace(0.1, 200, 5000):
        assert abs(_Dc_float(al)) <= 3 / (0.8 * al) + 1e-15


def test_var_cos_enclosure_against_quadrature():
    gx, gw = np.polynomial.legendre.leggauss(400)
    for al in (0.7, 1.7922917, 3.1, 8.0):
        def mean(f):
            xi = 0.8 * gx; xo = 0.4 * gx + 1.6
            return 0.5 * (f(xi) * gw).sum() / 2 + 0.5 * (f(xo) * gw).sum() / 2   # O is symmetric: one side suffices
        v = mean(lambda t: np.cos(al * t) ** 2) - mean(lambda t: np.cos(al * t)) ** 2
        enc = sk.var_cos_c(arb(al))
        assert enc.lower() - 1e-12 <= v <= enc.upper() + 1e-12


def test_localisation_pass():
    D_lo = sk.D_lower_at(sk.ALPHA0)
    r = sk.certify_localisation(D_lo)
    assert r["ok"] and r["test_point_cell_kept"] and not r["failed_cells"]


def test_localisation_fail_threshold_above_maximum():
    # constructed fail: a "lower bound" above the true maximum 1.5732419 clears every cell, including the test point's
    r = sk.certify_localisation(arb("1.58"))
    assert not r["ok"] and not r["test_point_cell_kept"]


def test_localisation_fail_box_too_small():
    # constructed fail: a box much narrower than the region where |D_c| >= D_lo - 0.01 cannot be resolved
    r = sk.certify_localisation(arb("1.56"), r=Fraction(1, 10 ** 6), max_depth=12)
    assert not r["ok"] and r["failed_cells"]


def test_localisation_fail_wrong_centre():
    # constructed fail: a box placed away from the maximiser leaves uncleared cells outside it
    r = sk.certify_localisation(sk.D_lower_at(sk.ALPHA0), centre=Fraction(17, 10), r=Fraction(1, 100), max_depth=20)
    assert not r["ok"]


def test_theorem_c_constants_match_section17_and_are_monotone():
    for V in (0.5, 5.6586, 5.4995e11):
        s1, Wc = sk.theorem_c_constants(Fraction(V), 1.3, Fraction(3, 4))
        Wf, sf = ss.compact_bounds(V, 1.3, 0.75)
        assert s1 * (1 - 1e-15) <= sf <= s1 * (1 + 1e-12) and Wc * (1 - 1e-12) <= Wf <= Wc * (1 + 1e-15)
    a1 = sk.theorem_c_constants(Fraction(5), 1.3, Fraction(3, 4))
    a2 = sk.theorem_c_constants(Fraction(6), 1.3, Fraction(3, 4))
    assert a2[0] < a1[0] and a2[1] > a1[1]


def test_spread_count_exact_constructed():
    assert sk.spread_count_exact([0, Fraction(2, 5), Fraction(4, 5)]) == 3           # closed endpoints count
    assert sk.spread_count_exact([0, Fraction(2, 5), Fraction(4, 5) + Fraction(1, 10 ** 30)]) == 2
    assert sk.spread_count_exact([0, 1, 2, 3]) == 1


def test_dyadic_lattice_constructed():
    E, g, ints = sk.dyadic_lattice([0.5, 0.75, -0.25])
    assert E == 2 and g == 1 and ints == [2, 3, -1]
    E, g, _ = sk.dyadic_lattice([0.5, -1.5, 2.0])
    assert E == 1 and g == 1
    E, g, _ = sk.dyadic_lattice([0.5, -1.0, 2.0])
    assert Fraction(g, 2 ** E) == Fraction(1, 2)


def test_exact_symmetry_pass_and_fail():
    xs = [-0.5, 0.5, -1.5, 1.5]; ys = [0, 0, 1, 1]
    assert sk.exact_symmetry(xs, ys) == (True, 0)
    sym, m = sk.exact_symmetry([-0.5, 0.5, -1.5, 1.5 + 2 ** -50], ys)       # constructed fail: one ulp-scale shift
    assert not sym and m == Fraction(1, 2 ** 51)


def test_alias_minimality_constructed():
    q = Fraction(1, 7)
    gi, go = sk.alias_minimality(q, [q * n for n in (-6, -2, 2, 6)], [q * n for n in (9, 15, 21)])
    assert gi == [4] and go == [6]


def test_float_alias_enclosure_matches_float():
    from src.blockB_landscape import population_data
    x, y = population_data()
    xs, ys = x.numpy(), y.numpy()
    al = math.pi / (0.4 / 79401)
    d_float = np.cos(al * xs[ys == 1]).mean() - np.cos(al * xs[ys == 0]).mean()
    assert abs(d_float + 2) < 1e-9


def test_committed_numbers():
    d = json.loads((ROOT / "results/small_scale_k.json").read_text())
    L = d["ledger"]
    assert L["K localisation ok"] == 1.0 and L["K float symmetric"] == 1.0
    assert L["K float spread count"] == 200 and L["K exact spread count"] == 200
    assert L["K float dyadic exponent"] == 60 and L["K float gcd"] == 1
    assert L["K exact inner gap"] == 796 and L["K exact outer gap"] == 798
    assert d["float_quadrature"]["m_exact"] == "0"
    assert abs(L["K Wc exact a=1.30"] - 2345103.67) < 1 and abs(L["K s1 exact a=1.30"] - 2.17545e-6) < 1e-10
    assert 10.8 < L["K Wc continuous box a=1.30"] < 10.81 and 0.675 < L["K s1 continuous box a=1.30"] < 0.676
    # §17's float-scan numbers (10.77, 0.678 at a = 1.30) sit inside the rigorous ones
    s17 = json.loads((ROOT / "results/small_scale_compactness.json").read_text())
    c = {r["a"]: r for r in s17["continuous"]["rows"]}[1.3]
    assert c["Wc"] <= L["K Wc continuous box a=1.30"] and c["s1"] >= L["K s1 continuous box a=1.30"]
    for r in d["continuous"]["rows"]:
        assert r["Wc_upper_box"] <= r["Wc_upper_crude"] and r["s1_lower_box"] >= r["s1_lower_crude"]
