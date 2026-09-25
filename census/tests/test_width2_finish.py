import numpy as np

from src.width2_conditional import population, profile_b
from src.width2_finish import newton
from src.width2_unplaced import loss
from src.width2_w0 import ACTS


def test_newton_reaches_the_cancelling_pair_from_a_perturbed_start():
    act = ACTS["f1.30"]
    x, y = population()
    s = 2 * 0.001 / 1.1868728875468348
    al = 1.7913244
    q = np.array([al, -np.pi / 2, al, np.pi / 2, 0.5, -0.5])          # the cancelling pair (sign D* < 0)
    L_pair = loss(q, s, x, y, act)
    rng = np.random.default_rng(0)
    q0 = q + rng.normal(0, 0.02, 6)
    z0 = np.r_[q0, profile_b(s * (q0[4] * act.u(q0[0] * x + q0[1]) + q0[5] * act.u(q0[2] * x + q0[3])), y)]
    z, it, gmax, conv = newton(z0, s, x, y, act)
    assert conv and loss(z[:6], s, x, y, act) <= L_pair + 1e-13


def test_joint_minimum_equals_the_profiled_loss():
    act = ACTS["f1.50"]
    x, y = population()
    s = 0.01
    q0 = np.array([1.2, 0.3, -2.0, 1.0, 0.7, 0.3])
    z0 = np.r_[q0, 0.0]
    z, *_ = newton(z0, s, x, y, act)
    import torch
    from src.width2_finish import _joint_loss
    J = float(_joint_loss(torch.tensor(z), s, torch.tensor(x), torch.tensor(y), act))
    assert abs(J - loss(z[:6], s, x, y, act)) < 1e-14                  # b is at its profile optimum
