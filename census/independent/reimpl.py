"""Independent reimplementation from mathematical definitions.

RULE: imports nothing from src/ except data loading (make_data). Every
quantity is derived from its definition, not from how the original computed it.
"""
from __future__ import annotations
import numpy as np

# --- task definition, from the paper's statement of it -------------------
# Task: realize sign(|x| - 1) on the sampled regions.
# class 0 (label 0): |x| <= 0.8          -> network output must be NEGATIVE
# class 1 (label 1): 1.2 <= |x| <= 2.0   -> network output must be POSITIVE
INNER_HALF = 0.8
OUTER_LO, OUTER_HI = 1.2, 2.0


def f_sin(t, a):
    """Family A activation: f_a(t) = t + a sin t."""
    return t + a * np.sin(t)


def net(theta, x, a, act=f_sin):
    """N(x) = w2 * f(w1 x + b1) + b2."""
    w1, b1, w2, b2 = theta
    return w2 * act(w1 * x + b1, a) + b2


def separates(theta, a, n=4001, act=f_sin) -> bool:
    """Definition: correct sign everywhere on both class regions.

    Both regions are continuous intervals, so 'everywhere' is approximated by
    a dense sweep. Independent choice: sample each region uniformly at n
    points, outer region on both signs.
    """
    inner = np.linspace(-INNER_HALF, INNER_HALF, n)
    outer_pos = np.linspace(OUTER_LO, OUTER_HI, n // 2)
    outer = np.concatenate([outer_pos, -outer_pos])
    return bool((net(theta, inner, a, act) < 0).all()
                and (net(theta, outer, a, act) > 0).all())


def fold_depth(a, samples=2_000_001):
    """D(a) = f(local max) - f(local min), by direct numerical search.

    Definition: the fold is the dip between f_a's local maximum and the
    following local minimum. Found by dense evaluation, no analytic input.
    """
    if a <= 1.0:
        return 0.0
    t = np.linspace(0.0, 2.0 * np.pi, samples)
    v = f_sin(t, a)
    # interior local extrema by first difference sign changes
    dv = np.diff(v)
    sign_change = np.nonzero(np.diff(np.sign(dv)) != 0)[0] + 1
    if len(sign_change) < 2:
        return 0.0
    vals = v[sign_change]
    return float(vals.max() - vals.min())


def gauss_linking(c1, c2):
    """Gauss linking integral, from its definition.

    lk = (1/4pi) * sum over segment pairs of
         (dr1 x dr2) . (r1 - r2) / |r1 - r2|^3
    evaluated at segment midpoints.
    """
    c1 = np.asarray(c1, float); c2 = np.asarray(c2, float)
    t1 = np.roll(c1, -1, 0) - c1          # tangent vectors
    t2 = np.roll(c2, -1, 0) - c2
    m1 = c1 + 0.5 * t1                     # midpoints
    m2 = c2 + 0.5 * t2
    diff = m1[:, None, :] - m2[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    cross = np.cross(t1[:, None, :], t2[None, :, :])
    num = (diff * cross).sum(axis=2)
    return float((num / dist**3).sum() / (4.0 * np.pi))


def grad_fd(theta, a, x, y, h=1e-6):
    """Gradient of mean BCE by central finite differences (no autograd)."""
    def loss(th):
        z = net(th, x, a)
        # numerically stable BCE-with-logits, from the definition
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))
    g = np.zeros(4)
    for i in range(4):
        tp = np.array(theta, float); tm = np.array(theta, float)
        tp[i] += h; tm[i] -= h
        g[i] = (loss(tp) - loss(tm)) / (2 * h)
    return g, loss(np.array(theta, float))


def hess_fd(theta, a, x, y, h=1e-4):
    """Hessian by central finite differences of the finite-difference gradient."""
    H = np.zeros((4, 4))
    for i in range(4):
        tp = np.array(theta, float); tm = np.array(theta, float)
        tp[i] += h; tm[i] -= h
        gp, _ = grad_fd(tp, a, x, y); gm, _ = grad_fd(tm, a, x, y)
        H[i] = (gp - gm) / (2 * h)
    return 0.5 * (H + H.T)


def loglog_slope(xs, ys):
    """Least-squares slope of log y against log x, from the normal equations."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    n = len(lx)
    return float((n * (lx * ly).sum() - lx.sum() * ly.sum())
                 / (n * (lx**2).sum() - lx.sum()**2))
