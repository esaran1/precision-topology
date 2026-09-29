"""Track 1A fold detector on constructed saddle-nodes (before applying it), and the reduced landscape's consistency with
the v2 objective."""

import numpy as np
import pytest

from src import sb_fold as sf
from src import simplicity_bias as sb
from src import simplicity_bias_v2 as v2


def _saddle_node(s_f, k=0.0, dim=1):
    """L(x, z; s) = x³/3 − (s_f − s)x (+ k(z − x)²/2): minimum x = +√(s_f − s), fold at s = s_f."""
    if dim == 1:
        F = lambda u, s: np.array([u[0] ** 2 - (s_f - s)])
        JF = lambda u, s: np.array([[2 * u[0], 1.0]])
        return F, JF
    F = lambda u, s: np.array([u[0] ** 2 - (s_f - s) - k * (u[1] - u[0]), k * (u[1] - u[0])])
    JF = lambda u, s: np.array([[2 * u[0] + k, -k, 1.0], [-k, k, 0.0]])
    return F, JF


@pytest.mark.parametrize("h", [0.05, 0.025, 0.0125])
def test_detects_saddle_node_1d(h):
    s_f = 2.7
    F, JF = _saddle_node(s_f)
    x0 = np.array([1.0, s_f - 1.0])
    pts, folds = sf.continuation(F, JF, x0, +1, h, n_max=2000)
    assert len(folds) == 1
    assert folds[0]["s_fold"] == pytest.approx(s_f, abs=1e-9)
    assert abs(folds[0]["eig_min"]) < 1e-5                               # Hessian 2x → 0 at the fold
    assert pts[0]["eig_min"] > 0 and pts[-1]["eig_min"] < 0              # minimum branch → saddle branch
    assert max(p["s"] for p in pts) <= s_f + 1e-9


def test_detects_saddle_node_coupled_2d():
    s_f = 5.0
    F, JF = _saddle_node(s_f, k=3.0, dim=2)
    x0 = np.array([2.0, 2.0, s_f - 4.0])
    pts, folds = sf.continuation(F, JF, x0, +1, 0.1, n_max=2000)
    assert len(folds) == 1 and folds[0]["s_fold"] == pytest.approx(s_f, abs=1e-9)
    assert abs(folds[0]["eig_min"]) < 1e-4
    ev = [p["eig_min"] for p in pts]
    assert ev[0] > 0 and ev[-1] < 0


def test_no_false_fold_on_a_regular_branch():
    F = lambda u, s: np.array([u[0] - s ** 2])                            # x = s², no turning point
    JF = lambda u, s: np.array([[1.0, -2 * s]])
    pts, folds = sf.continuation(F, JF, np.array([0.25, 0.5]), +1, 0.05, n_max=200, s_bounds=(0, 3))
    assert folds == [] and pts[-1]["s"] > 2.9


def test_backward_direction_and_newton():
    s_f = 1.0
    F, JF = _saddle_node(s_f)
    u, ok, r = sf.newton_fixed_s(F, JF, np.array([0.9]), 0.0)
    assert ok and u[0] == pytest.approx(1.0)
    pts, folds = sf.continuation(F, JF, np.r_[u, 0.0], -1, 0.1, n_max=50, s_bounds=(-2, 3))
    assert folds == [] and pts[-1]["s"] < -1.9


def test_reduced_loss_matches_v2_objective():
    X, y = sb.make_data(0.2)
    rng = np.random.default_rng(3)
    for nA in (1, 3):
        p = np.concatenate([rng.normal(0, 2, 12), np.r_[rng.uniform(0.3, 1, nA), np.zeros(4 - nA)]])
        s = 2.3
        u, n = sf.Reduced.from_full(p, s, X, y)
        assert n == nA
        R = sf.Reduced(nA, X, y)
        L2 = v2.loss_grad(p[None], s, X, y, sf.LAM)[0][0]
        idle = 0.5 * sf.LAM * (p[2 * nA:8] ** 2).sum() + 0.5 * sf.LAM * (p[8 + nA:12] ** 2).sum()
        assert R.loss(u, s) + idle == pytest.approx(L2, abs=1e-12)
        assert abs(R.F(u, s)[-1]) < 1e-10                                 # b profiled: ∂L/∂b = 0
        P, b = R.to_full(u)
        assert np.allclose(sb.phi(P[None], X), sb.phi(p[None], X), atol=1e-12)


def test_function_distance_is_permutation_invariant():
    X, _ = sb.make_data(0.2)
    rng = np.random.default_rng(4)
    p = np.concatenate([rng.normal(0, 2, 12), rng.uniform(0.3, 1, 4)])
    W = p[:8].reshape(4, 2); perm = [2, 0, 3, 1]
    q = np.concatenate([W[perm].ravel(), p[8:12][perm], p[12:][perm]])
    assert sf.fdist(p[None], q[None], X)[0] < 1e-14
    r = p.copy(); r[0] += 0.5
    assert sf.fdist(p[None], r[None], X)[0] > 1e-3


def test_no_global_torch_state_change():
    import torch
    before = torch.get_default_dtype()
    X, y = sb.make_data(0.2)
    R = sf.Reduced(1, X, y)
    u = np.r_[3.0, 0.0, 0.0, 0.0]
    R.F(u, 1.0); R.JF(u, 1.0); R.loss(u, 1.0)
    assert torch.get_default_dtype() == before
