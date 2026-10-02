"""EXPLORATORY helpers for the Phase 2A design (NOT registered).  Training coordinates of the v3 model
(W 4x2, c 4, v 4, b; z = tanh(XW^T + c)·v + b; BCE + (λ/2)(|W|² + |c|²), λ = 1e-4, the v2/v3 800 points), analytic
gradient, autograd Hessian, branch points of src/sb_fold.py mapped to training coordinates, branch labels by function
space distance (sb_fold.fdist ≤ 1e-3) at the run's own s.  Population = training set (the benchmark has one fixed point set).
No global torch/numpy state: torch is used with explicit float64 tensors and set_num_threads(1)."""
from __future__ import annotations

import math
import numpy as np

from src import simplicity_bias as sb
from src import simplicity_bias_v2 as v2
from src import sb_fold as F

LAM = 1e-4
S_STAR = 3.5913755424683727
S_FOLD = 4.767689442106793
Q = 0.3914103370353161
OMEGA0 = 2.338107410459767
X, Y = v2.data()
N = len(Y)


def unpack(th):
    return th[:8].reshape(4, 2), th[8:12], th[12:16], th[16]


def loss_grad(th, lam=LAM):
    W, c, v, b = unpack(th)
    H = np.tanh(X @ W.T + c)
    z = H @ v + b
    L = float(np.mean(np.logaddexp(0.0, z) - Y * z) + 0.5 * lam * ((W ** 2).sum() + (c ** 2).sum()))
    r = (0.5 * (1 + np.tanh(0.5 * z)) - Y) / N
    D = (1 - H ** 2) * (r[:, None] * v[None, :])
    g = np.empty(17)
    g[:8] = (D.T @ X + lam * W).ravel()
    g[8:12] = D.sum(0) + lam * c
    g[12:16] = H.T @ r
    g[16] = r.sum()
    return L, g


def hessian(th, lam=LAM):
    import torch
    torch.set_num_threads(1)
    Xt = torch.as_tensor(X, dtype=torch.float64); Yt = torch.as_tensor(Y, dtype=torch.float64)

    def Lf(q):
        W = q[:8].reshape(4, 2); c = q[8:12]; v = q[12:16]; b = q[16]
        z = torch.tanh(Xt @ W.T + c) @ v + b
        return torch.nn.functional.binary_cross_entropy_with_logits(z, Yt) + 0.5 * lam * ((W ** 2).sum() + (c ** 2).sum())
    return torch.autograd.functional.hessian(Lf, torch.tensor(th, dtype=torch.float64)).numpy()


def scale(th):
    return float(np.abs(th[12:16]).sum())


def to_landscape(th):
    """Training row -> 16-vector of the landscape parametrisation (signs absorbed, η = √(|v|/‖v‖₁)) (sb_fold.run_to_full)."""
    return F.run_to_full(np.asarray(th, float)[None])[0]


def branch_theta(name, s):
    """Branch point at exactly s (Newton from the stored stable points) in training coordinates (idle units zero)."""
    r = F.branch_point(name, s)
    if r is None:
        return None
    u, R = r
    n = R.nA
    W = u[:2 * n].reshape(n, 2); c = u[2 * n:3 * n]; eta = np.r_[1.0, u[3 * n:4 * n - 1]]; b = u[-1]
    v = s * eta ** 2 / (eta ** 2).sum()
    th = np.zeros(17)
    Wf = np.zeros((4, 2)); Wf[:n] = W
    th[:8] = Wf.ravel(); th[8:8 + n] = c; th[12:12 + n] = v; th[16] = b
    return th


def branch_dists(th, names=("L0", "M", "S", "S2")):
    """Function-space distance (unit-ℓ₁ φ, RMS over the 800 points) to each branch's point at the row's own s."""
    s = scale(th); P = to_landscape(th)
    out = {}
    for nm in names:
        bt = branch_theta(nm, s)
        out[nm] = math.inf if bt is None else float(F.fdist(P[None], to_landscape(bt)[None], X)[0])
    return out


def label(th, tol=1e-3):
    d = branch_dists(th)
    nm = min(d, key=d.get)
    return (nm if d[nm] <= tol else "other"), d


def rho2(th):
    return float(sb.feature_usage(to_landscape(th), X)["rho2"])


def gplus(th):
    return float(sb.gplus(to_landscape(th)[None], X, Y)[0])
