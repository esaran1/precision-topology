"""§17 (small-scale compactness): the decomposition identity, psi bounds, the spreading count, and Theorem C / C' on
constructed data (a lattice set where the supremum is attained, and an asymmetric set where it is not)."""

import json
import math
from pathlib import Path

import numpy as np

from src import small_scale as ss

ROOT = Path(__file__).resolve().parents[1]


def test_decomposition_identity():
    rng = np.random.default_rng(3)
    x = np.r_[rng.uniform(-0.8, 0.8, 30), rng.uniform(1.2, 2, 15), -rng.uniform(1.2, 2, 15)]
    y = np.r_[np.zeros(30), np.ones(30)]
    for _ in range(50):
        w1, b1, s = rng.uniform(-6, 6), rng.uniform(0, 6.3), 10 ** rng.uniform(-3, 0.5)
        phi = w1 * x + b1 + 1.3 * np.sin(w1 * x + b1)
        dmu = phi[y == 1].mean() - phi[y == 0].mean()
        assert abs(ss.profiled_loss(phi, y, s) - (math.log(2) - s * dmu / 4 + ss.V_s(phi, s))) < 1e-10


def test_psi_bounds():
    z = np.linspace(-6, 6, 20001)
    p = ss.psi(z)
    assert np.all(p <= z ** 2 / 8 + 1e-15)
    inner = np.abs(z) <= 2
    assert np.all(p[inner] >= 5 / 48 * z[inner] ** 2 - 1e-15)
    assert np.all(p[~inner] >= ss.PSI2 - 1e-15)


def test_spreading_count_population():
    x, _ = ss.population()
    cnt, frac = ss.spreading_fraction(x)
    assert cnt == 200 and frac == 0.75


def _grid_min_w1(x, y, s, a=1.3, W=80.0, nw=321, nb=12):
    """Brute-force conditional minimiser over a (w1, b1) grid; returns |w1| of the best point."""
    best, arg = np.inf, None
    for w1 in np.linspace(-W, W, nw):
        for b1 in np.linspace(0, 2 * math.pi, nb, endpoint=False):
            phi = w1 * x + b1 + a * np.sin(w1 * x + b1)
            for sg in (1, -1):
                L = ss.profiled_loss(sg * phi, y, s)
                if L < best:
                    best, arg = L, abs(w1)
    return arg


def _constructed():
    inner = np.linspace(-0.8, 0.8, 9); outer = np.r_[np.linspace(1.2, 2.0, 5), -np.linspace(1.2, 2.0, 5)]
    x = np.r_[inner, outer, 0.0]; y = np.r_[np.zeros(9), np.ones(10), 1.0]      # balanced 10/10, symmetric, m = 0
    return x, y


def test_theorem_C_pass_on_constructed_symmetric_lattice():
    # x on the lattice 0.2 Z: D is periodic, so sup Delta-mu is attained -> minimisers stay bounded as s decreases
    x, y = _constructed()
    assert abs(x[y == 1].mean() - x[y == 0].mean()) < 1e-12
    w = [_grid_min_w1(x, y, s) for s in (1e-2, 1e-3, 1e-4)]
    assert max(w) < 5


def test_C_prime_fail_on_constructed_asymmetric_data():
    # constructed fail: shift class 1 by 0.05 (m = 0.05 != 0): sup Delta-mu = +inf, the argmin |w1| escapes
    x, y = _constructed()
    xa = x + np.where(y == 1, 0.05, 0.0)
    w = [_grid_min_w1(xa, y, s) for s in (1e-2, 1e-4)]
    assert w[0] < 5 and w[1] > 30


def test_committed_numbers():
    d = json.loads((ROOT / "results/small_scale_compactness.json").read_text())
    assert d["population"]["inner_parity_even"] and d["population"]["outer_parity_odd"]
    assert d["samples"]["all_nonzero"] and d["identity_max_abs_err"] < 1e-12
    r = {x["a"]: x for x in d["population"]["rows"]}[1.3]
    assert abs(r["Wc"] - 2345103.67) < 1 and abs(r["s1"] - 2.1754e-6) < 1e-9
