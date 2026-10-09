"""POST HOC final checks (src/final_posthoc.py): constructed cases for the fits, local slopes, Mann–Whitney/Holm and
the distance functions."""
import itertools
import math

import numpy as np
import pytest

import src.final_posthoc as F

OM = F.OMEGA0


def _ks(eps, A=OM, C=0.8):
    return A * eps ** (2 / 3) + C * eps * np.log(1 / eps)


# ------------------------------------------------------------------------------------------ fits and slopes
def test_local_slope_matches_numeric_derivative():
    for A, C, e in ((OM, 0.8, 1e-4), (2.0, -0.5, 3e-5), (OM, 0.0, 1e-3)):
        h = 1e-6
        num = (math.log(_ks(e * math.exp(h), A, C)) - math.log(_ks(e * math.exp(-h), A, C))) / (2 * h)
        assert float(F.ks_local_slope(A, C, e)) == pytest.approx(num, abs=1e-8)


def test_local_slope_pure_power_is_two_thirds_and_above_with_positive_C():
    assert float(F.ks_local_slope(OM, 0.0, 1e-4)) == pytest.approx(2 / 3, abs=1e-14)
    assert float(F.ks_local_slope(OM, 0.8, 1e-4)) > 2 / 3
    assert float(F.ks_local_slope(OM, -0.8, 1e-4)) < 2 / 3
    # approaches 2/3 as eps -> 0
    s = [float(F.ks_local_slope(OM, 0.8, e)) for e in (1e-3, 1e-5, 1e-7, 1e-9)]
    assert all(a > b for a, b in zip(s, s[1:])) and s[-1] - 2 / 3 < 0.01


def test_two_point_slope_and_exact_solve():
    assert F.two_point_slope(1.0, 2.0, 1.0, 4.0) == pytest.approx(0.5)
    assert F.two_point_slope(1.0, 2.0, 1.0, 1.0) is None and F.two_point_slope(None, 2.0, 1.0, 4.0) is None
    e1, e2 = 8.8e-4, 2.2e-4
    A, C = F.ks_exact_two_point(_ks(e1, 2.1, 0.6), _ks(e2, 2.1, 0.6), e1, e2)
    assert A == pytest.approx(2.1, rel=1e-10) and C == pytest.approx(0.6, rel=1e-8)
    # pure power: two-point slope is exactly 2/3
    assert F.ks_two_point_slope(OM, 0.0, e1, e2) == pytest.approx(2 / 3, abs=1e-13)


def test_pooled_fit_recovers_per_seed_A_and_common_C():
    rng = np.random.default_rng(0)
    As = rng.uniform(0.9, 1.1, 12) * OM
    e14 = rng.uniform(1e-4, 6e-4, 12)
    eps = np.c_[e14, e14 / 4].ravel()
    g = np.repeat(np.arange(12), 2)
    r = np.array([_ks(e, As[k], 0.8) for e, k in zip(eps, g)])
    f = F.fit_pooled_ks(eps, r, g, "free_A")
    assert f["C"] == pytest.approx(0.8, rel=1e-8) and f["max_abs_rel"] < 1e-12 and f["n_par"] == 13
    assert all(f["A"][k] == pytest.approx(As[k], rel=1e-10) for k in range(12))
    f0 = F.fit_pooled_ks(eps, _ks(eps, OM, -0.3), g, "Omega0")
    assert f0["C"] == pytest.approx(-0.3, rel=1e-9) and f0["n_par"] == 1 and set(f0["A"].values()) == {OM}
    fc = F.fit_pooled_ks(eps, _ks(eps, 2.0, 0.5), g, "common_A")
    assert fc["C"] == pytest.approx(0.5, rel=1e-8) and fc["A"][3] == pytest.approx(2.0, rel=1e-10)


def test_observed_eps_on_a_known_path():
    # s(t) = s_F (1 - a (T - t)) linear near s_F: ṡ = a s_F everywhere; the secant and the quadratic agree
    s_F, a, T, Lam = 5.0, 1e-6, 1e6, 1e-3
    t_of = lambda s: T - (1 - s / s_F) / a
    e = F.observed_eps(s_F, Lam, t_of(0.8 * s_F), t_of(0.95 * s_F), 0.95 * s_F, t_of(s_F))
    assert e["secant"] == pytest.approx(a / Lam, rel=1e-9)
    assert e["quadratic"] == pytest.approx(a / Lam, rel=1e-6)


# ------------------------------------------------------------------------------------------ ranks, Spearman, MWU, Holm
def test_ranks_midranks_and_infinity_last():
    assert F.ranks([3.0, 1.0, 3.0, math.inf]).tolist() == [2.5, 1.0, 2.5, 4.0]


def test_spearman_monotone_and_permutation_p():
    x = np.arange(20.0)
    assert F.spearman(x, x ** 3) == pytest.approx(1.0)
    assert F.spearman(x, -x) == pytest.approx(-1.0)
    rho, p = F.spearman_perm_p(x, x ** 2, n_perm=500, seed=1)
    assert rho == pytest.approx(1.0) and p == pytest.approx(1 / 501)


def _brute_mw(x, y):
    allv = np.r_[x, y]
    rk = F.ranks(allv)
    m = len(x)
    E = m * (len(allv) + 1) / 2
    w = rk[:m].sum()
    c = [rk[list(i)].sum() for i in itertools.combinations(range(len(allv)), m)]
    return np.mean([abs(v - E) >= abs(w - E) - 1e-9 for v in c])


def test_mann_whitney_exact_against_enumeration_with_and_without_ties():
    cases = (([1, 2, 3, 3.5, 9], [4, 5, 6, 7, 8, 10, 11]),
             ([1, 1, 2, 5], [2, 2, 3, 3, 4, 6]),
             ([0, 0, 1], [0, 1, 1, 1, math.inf]))
    for x, y in cases:
        r = F.mann_whitney_exact(x, y)
        assert r["p"] == pytest.approx(_brute_mw(x, y), abs=1e-12)
        # U from its definition: pairs x > y plus half ties
        U = sum((a > b) + 0.5 * (a == b) for a in x for b in y)
        assert r["U"] == pytest.approx(U) and r["auc"] == pytest.approx(U / (len(x) * len(y)))


def test_mann_whitney_complete_separation_7_vs_23():
    r = F.mann_whitney_exact(np.arange(7.0) + 100, np.arange(23.0))
    assert r["U"] == 7 * 23 and r["auc"] == 1.0
    assert r["p"] == pytest.approx(2 / math.comb(30, 7), rel=1e-12)


def test_mann_whitney_all_tied_is_one():
    assert F.mann_whitney_exact([1, 1], [1, 1, 1])["p"] == pytest.approx(1.0)


def test_holm():
    assert F.holm([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert F.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0]) and F.holm([0.2, 0.01]) == pytest.approx([0.2, 0.02])
    assert F.holm([0.3, 0.3, 0.3, 0.3]) == pytest.approx([1.0, 1.0, 1.0, 1.0])


# ------------------------------------------------------------------------------------------ distances
def test_param_dist_is_permutation_invariant():
    rng = np.random.default_rng(3)
    P = rng.normal(size=16)
    perm = (2, 0, 3, 1)
    Qp = P[F.perm_index(perm)]
    assert F.param_dist(P, Qp[None])[0] == pytest.approx(0.0, abs=1e-14)
    Q2 = P.copy(); Q2[5] += 0.3
    assert F.param_dist(P, np.stack([Q2, Qp]))[0] == pytest.approx(0.3)


def test_perm_index_is_a_permutation_keeping_units_together():
    idx = F.perm_index((1, 0, 2, 3))
    assert sorted(idx) == list(range(16))
    assert idx[:4] == [2, 3, 0, 1] and idx[8:10] == [9, 8] and idx[12:14] == [13, 12]


def test_nearest_slab():
    d = [0.3, 0.1, 0.2, 0.05]
    r = [0.5, 0.2, 0.4, 0.39]
    assert F.nearest_slab(d, r, q=0.3914) == pytest.approx(0.2)
    assert math.isinf(F.nearest_slab(d, [0.1] * 4, q=0.3914))


def test_fdist_identity_and_symmetry_on_unit_l1_functions():
    from src import sb_fold as SBF
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, (50, 2))
    P = np.abs(rng.normal(size=16)); Q = np.abs(rng.normal(size=16))
    assert SBF.fdist(P[None], P[None], X)[0] == pytest.approx(0.0, abs=1e-15)
    assert SBF.fdist(P[None], Q[None], X)[0] == pytest.approx(SBF.fdist(Q[None], P[None], X)[0])
    # unit permutation leaves the function (hence fdist) unchanged
    assert SBF.fdist(P[None], P[F.perm_index((3, 1, 0, 2))][None], X)[0] == pytest.approx(0.0, abs=1e-14)


def test_summary_handles_infinity():
    s = F.summary([1.0, 2.0, math.inf])
    assert s["n"] == 3 and s["n_inf"] == 1 and s["median"] == 2.0 and math.isinf(s["max"]) and "q25" not in s
