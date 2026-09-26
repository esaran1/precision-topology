"""Tests for src/linear_response.py: the linear-response recursion on constructed cases, the analytic derivatives and the
exact gap (Track 1, final round; POST HOC analysis)."""

import numpy as np
import pytest

from src import linear_response as L


def _const_case(T=4000, sdot=1e-3, s_star=2.0, lr=0.01):
    H = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])
    P = np.array([3.0, 1.5, 2.0])
    v = np.array([0.4, -0.7, 1.3])                       # θ*′ (constant: linear branch)
    c = np.array([1.0, -1.0, 0.0])                       # ∇G (linear gap, no b₂ dependence)
    s = s_star * 0.7 + sdot * np.arange(T + 1)
    th_star = lambda si: v * (si - s_star)               # c·θ*(s*) = 0: the branch switch is at s*
    dth = [v * (s[t + 1] - s[t]) for t in range(T)]
    Ps = np.tile(P, (T + 1, 1))
    slaved = -np.linalg.solve(lr * P[:, None] * H, v * sdot)
    return H, P, v, c, s, th_star, dth, Ps, slaved, lr, s_star, sdot


def test_slaved_start_gives_constant_delta():
    H, P, v, c, s, th_star, dth, Ps, slaved, lr, *_ = _const_case(T=500)
    _, path = L.simulate(s, lambda t: H, Ps, dth, lambda t, d: -1.0, slaved, lr=lr)
    assert np.abs(np.array(path) - slaved).max() < 1e-10


def test_constant_coefficients_reproduce_analytic_slaved_lag():
    H, P, v, c, s, th_star, dth, Ps, slaved, lr, s_star, sdot = _const_case()
    gapf = lambda t, d: float(c @ (th_star(s[t]) + d))
    t_hit, _ = L.simulate(s, lambda t: H, Ps, dth, gapf, slaved, lr=lr)
    r_pred = s[t_hit] / s_star - 1
    lam = np.linalg.eigvalsh((np.sqrt(P)[:, None] * H) * np.sqrt(P)[None, :]).min()
    kappa = lam * (c @ np.linalg.solve(P[:, None] * H, v)) / (c @ v)
    chi = (sdot / s_star) / (lr * lam)
    assert r_pred == pytest.approx(kappa * chi, abs=sdot / s_star + 1e-12)
    # from a non-slaved start the transient decays and the same lag results
    t2, _ = L.simulate(s, lambda t: H, Ps, dth, gapf, np.zeros(3), lr=lr)
    assert s[t2] / s_star - 1 == pytest.approx(kappa * chi, abs=2 * sdot / s_star)


def test_momentum_with_beta1_zero_is_the_no_momentum_recursion():
    H, P, v, c, s, th_star, dth, Ps, slaved, lr, *_ = _const_case(T=800)
    d0 = np.array([0.05, -0.02, 0.1])
    gapf = lambda t, d: -1.0
    _, p0 = L.simulate(s, lambda t: H, Ps, dth, gapf, d0, lr=lr)
    _, p1 = L.simulate(s, lambda t: H, Ps, dth, gapf, d0, m0=np.zeros(3), bc1=np.ones(len(s)), beta1=0.0, lr=lr)
    assert np.abs(np.array(p0) - np.array(p1)).max() < 1e-14


def test_momentum_leaves_the_steady_lag_unchanged():
    H, P, v, c, s, th_star, dth, Ps, slaved, lr, *_ = _const_case(T=3000)
    bc1 = np.ones(len(s))
    _, p = L.simulate(s, lambda t: H, Ps, dth, lambda t, d: -1.0, slaved, m0=H @ slaved, bc1=bc1, lr=lr)
    assert np.abs(np.array(p) - slaved).max() < 1e-10                 # slaved state with m = Hδ is stationary
    _, p2 = L.simulate(s, lambda t: H, Ps, dth, lambda t, d: -1.0, np.zeros(3), m0=np.zeros(3), bc1=bc1, lr=lr)
    assert np.abs(p2[-1] - slaved).max() < 1e-6 * np.abs(slaved).max()


def test_events_add_the_jump_and_replace_the_moment():
    H, P, v, c, s, th_star, dth, Ps, slaved, lr, *_ = _const_case(T=10)
    jump = np.array([0.1, 0.0, -0.3])
    _, p0 = L.simulate(s, lambda t: H, Ps, dth, lambda t, d: -1.0, slaved, lr=lr)
    _, p1 = L.simulate(s, lambda t: H, Ps, dth, lambda t, d: -1.0, slaved, lr=lr, events={1: (jump, None)})
    assert np.allclose(p1[1] - p0[1], jump)


def test_derivatives_match_autograd():
    torch = pytest.importorskip("torch")
    from src import lag_law
    from src.fold1d import make_data
    x, y = make_data(200, 3); x, y = x.double().numpy(), y.double().numpy()
    rng = np.random.default_rng(1)
    for _ in range(3):
        th = rng.normal(size=3); s, a = 2.3, 1.5
        l, g, H, ds = L.derivs(th, s, a, x, y, need_ds=True)
        Ht, tan = lag_law.hessian_and_tangent(th, s, a, x, y)
        X, Y = torch.tensor(x), torch.tensor(y)
        zt = torch.tensor(th, requires_grad=True)
        gt = torch.autograd.grad(lag_law._loss_t(zt, s, a, X, Y), zt)[0].numpy()
        assert np.abs(g - gt).max() < 1e-12
        assert np.abs(H - Ht).max() < 1e-12
        assert np.abs(-np.linalg.solve(H, ds) - tan).max() < 1e-9


def test_exact_gap_matches_lag_law():
    from src import lag_law
    rng = np.random.default_rng(2)
    for _ in range(100):
        w1, b1, a = rng.uniform(-4, 4), rng.uniform(-8, 8), rng.choice([1.3, 1.45, 1.5, 1.6])
        assert abs(L.gap_exact(w1, b1, a) - lag_law.gap(w1, b1, a)) < 1e-10


def test_branch_interpolation_is_accurate():
    from src.fold1d import make_data
    x, y = make_data(200, 830_000); x, y = x.double().numpy(), y.double().numpy()
    a, s0 = 1.45, 2.9
    th, res, H = L.newton(np.array([1.0, 0.0, -2.0]), s0, a, x, y)
    if res > 1e-9 or np.linalg.eigvalsh(H).min() <= 0:
        pytest.skip("no minimum from this start")
    B = L.Branch(s0, th, a, x, y, 0.9 * s0, 1.1 * s0, 0.002 * 2.9)
    for sv in np.linspace(B.lo, B.hi, 7)[1:-1]:
        ex, Hx, _, r = B.exact(sv)
        assert np.abs(B.theta(sv) - ex).max() < 1e-7
        assert np.abs(B.hess(sv) - Hx).max() < 1e-3
