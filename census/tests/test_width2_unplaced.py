import math

import numpy as np

from src.width2_unplaced import classify, g_dense, g_exact, loss


def test_classify_constructed_cases():
    assert classify(0.5, -1e-3, 0.5 - 1e-9, 0.02) == "boundary"          # falls and becomes placed: constraint active
    assert classify(0.5, -0.3, 0.5, -0.3) == "interior"                  # cannot fall
    assert classify(0.5, -0.3, 0.5 - 1e-13, -0.3) == "interior"          # below the descent tolerance
    assert classify(0.5, -0.3, 0.5 - 1e-9, -0.2) == "unresolved"         # falls but stays unplaced
    assert classify(0.5, 1e-6, 0.5, 1e-6) == "infeasible"


def test_loss_equals_the_validated_batch_objective():
    from src.width2_conditional import loss_grad_batch, population
    from src.width2_w0 import ACTS
    act = ACTS["f1.30"]
    x, y = population()
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = np.r_[rng.uniform(-5, 5), rng.uniform(0, 6), rng.uniform(-5, 5), rng.uniform(0, 6), rng.uniform(-1, 1)]
        sg = float(rng.choice([-1, 1]))
        L, _ = loss_grad_batch(p[None, :], np.array([sg]), 0.01, x, y, act)
        q = np.r_[p[:4], p[4], sg * (1 - abs(p[4]))]
        assert abs(loss(q, 0.01, x, y, act) - float(L[0])) < 1e-13


def test_dense_gap_is_an_upper_bound_of_the_exact_gap():
    from src.width2_w0 import ACTS
    act = ACTS["f1.30"]
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = np.r_[rng.uniform(-4, 4), rng.uniform(0, 6), rng.uniform(-4, 4), rng.uniform(0, 6), rng.uniform(-1, 1, 2)]
        assert g_dense(q, act) >= g_exact(q, act)[1] - 1e-12
