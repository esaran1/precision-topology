import numpy as np

from src import residual_timescale as rt


def test_relax_rate_diagonal_and_preconditioning():
    H = np.diag([2.0, 8.0, 1.0]); v = np.array([1.0, 16.0, 0.25])       # sqrt(v) = 1, 4, 0.5
    lam = rt.relax_rate(H, v, eps=0.0, lr=0.01)
    assert abs(lam - 0.01 * min(2 / 1, 8 / 4, 1 / 0.5)) < 1e-15
    assert rt.relax_rate(np.diag([1.0, -1.0, 1.0]), np.ones(3)) < 0     # fail case: saddle gives a negative rate


def test_spearman_and_bootstrap():
    x = np.arange(50.0)
    assert abs(rt.spearman(x, x ** 3) - 1) < 1e-12 and abs(rt.spearman(x, -x) + 1) < 1e-12
    lo, hi = rt.boot_spearman(x, x + np.random.default_rng(0).normal(0, 5, 50), n=500)
    assert lo > 0.8
    lo, hi = rt.boot_spearman(x, np.random.default_rng(1).normal(0, 1, 50), n=500)
    assert lo < 0 < hi


def test_branch_hessian_converges_on_separable_data():
    x = np.r_[np.linspace(-0.8, 0.8, 50), np.linspace(1.2, 2.0, 25), -np.linspace(1.2, 2.0, 25)]
    y = np.r_[np.zeros(50), np.ones(50)]
    z, H, g, conv = rt.branch_hessian(1.3, x, y, 0.8, 2.45, 1.0, 0.0)
    assert conv and np.all(np.linalg.eigvalsh(H) > 0)
