"""Theorem 1 hypothesis checks: the check functions on constructed pass / fail cases, and the committed results."""

import json
import math
from pathlib import Path

import numpy as np

from src import theorem1_checks as T

ROOT = Path(__file__).resolve().parents[1]


def test_lipschitz_scan_pass_and_fail():
    ok, bad = T.lipschitz_scan(lambda t: 0.5 * np.abs(np.sin(t)), 0, 10, 1e-3, 0.5, 0.6)
    assert ok and bad is None
    ok, bad = T.lipschitz_scan(lambda t: np.where(np.abs(t - 7) < 0.01, 0.7, 0.1), 0, 10, 1e-3, 0.0, 0.6)
    assert not ok and abs(bad - 7) < 0.02


def test_adaptive_scan_finds_first_exceedance():
    f = lambda t: 1 - (t - 3.0) ** 2                       # reaches level 0.99 only on |t - 3| <= 0.1
    ok, ex, unres = T.adaptive_scan(f, 0, 2.5, 0.05, 6.0, 0.99)
    assert ok and ex is None and unres is None
    ok, ex, unres = T.adaptive_scan(f, 0, 5, 0.05, 10.0, 0.99)
    assert not ok and ex is not None and 2.89 <= ex <= 2.93 and (unres is None or 2.89 <= unres <= 2.91)


def test_quadratic_growth_pass_and_fail():
    ok, c = T.quadratic_growth(lambda t: -2 + 0.1 * (t - 1), 0.9, 1.1, 1e-3, 0.1)
    assert ok and c > 1.9
    ok, c = T.quadratic_growth(lambda t: -12 * (t - 1) ** 2, 0.9, 1.1, 1e-3, 3.0)   # quartic flat maximum
    assert not ok


def test_remainder_ratio_bounded_and_detects_wrong_coefficient():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(50), np.zeros(50)]
    phi = rng.normal(size=100) + y
    for s in (0.1, 0.03):
        r, R, m4 = T.remainder_ratio(phi, y, s)
        assert r <= 1 / 192 * (1 + 1e-3)
    r_bad, *_ = T.remainder_ratio(phi, y, 0.03, var_coef=1 / 6)             # wrong s^2 coefficient
    assert r_bad > 100 / 192


def test_lattice_alias_on_population():
    x, y = T._data()
    Dl, runs = T.D_lattice_factory(x, y)
    Dd, *_ = T.D_data_factory(x, y)
    t = np.array([1.7913244, 10.0, 1234.5, math.pi / T.Q])
    assert np.max(np.abs(Dl(t) - Dd(t))) < 1e-9
    assert abs(Dd(np.array([math.pi / T.Q]))[0] + 2.0) < 1e-9                # |D| = 2 > |D*| at the alias


def test_committed_results():
    d = json.loads((ROOT / "results/theorem1_checks.json").read_text())
    assert all(r["Gamma_n_positive"] and r["argmax_interior_margin"] > 0 and r["Gamma_n_ge_Ghat_lo"]
               for r in d["H-A1_data_gap_attainment"])
    cm = d["H-A2_classmean_attainment_and_H-Q_quadratic_growth"]
    assert cm["continuous"]["global_maximiser_certified"] and cm["data"]["quad_growth_ok"]
    assert not cm["data"]["alpha_star_global_on_(0, pi/q)"] and cm["data"]["first_alpha_with_absD_ge_Dstar"] > 6e5
    assert max(r["max_ratio_all"] for r in d["H-R_uniform_remainder"]) < 1 / 192 * (1 + 1e-4)
