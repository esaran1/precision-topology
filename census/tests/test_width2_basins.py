import numpy as np

from src import width2_basins as wb


def _dup_q(sigma=-1.0):
    a, b = 1.8, 1.57
    return np.array([a, b, 0.2, sigma * a, sigma * b, sigma * 0.2 if sigma > 0 else -0.2, -0.05])


def test_duplicate_detection():
    assert wb.duplicate(_dup_q(-1.0))[0] and wb.duplicate(_dup_q(-1.0))[1] == -1.0
    assert wb.duplicate(_dup_q(1.0))[0] and wb.duplicate(_dup_q(1.0))[1] == 1.0
    q = _dup_q(-1.0); q[3] += 1e-3
    assert not wb.duplicate(q)[0]


def test_transfer_direction_is_flat_for_duplicates():
    """The weight transfer between copies leaves the network function (hence the loss) unchanged."""
    import torch
    from src.width2_conditional import training_set
    from src.width2_nogating import act_of
    from src.width2_train import logits
    act = act_of("f1.30"); x, y = training_set(630_000)
    X = torch.tensor(x, dtype=torch.float64)
    q = _dup_q(-1.0)
    e = wb.transfer_direction(q)
    f0 = logits(torch.tensor(q), X, act)
    f1 = logits(torch.tensor(q + 0.05 * e), X, act)
    assert float((f0 - f1).abs().max()) < 1e-12
    assert abs(abs(q[2] + 0.05 * e[2]) + abs(q[5] + 0.05 * e[5]) - (abs(q[2]) + abs(q[5]))) < 1e-15


def _spectrum_case(evals, zero_vec_along_transfer, q):
    from src.width2_nogating import free_basis
    B = free_basis(q)
    e = B.T @ wb.transfer_direction(q); e /= np.linalg.norm(e)
    rng = np.random.default_rng(0)
    M = rng.standard_normal((6, 6))
    M[:, 0] = e if zero_vec_along_transfer else rng.standard_normal(6)
    Q, _ = np.linalg.qr(M)
    if zero_vec_along_transfer:
        Q[:, 0] = e * np.sign(Q[:, 0] @ e)
    return np.asarray(evals, float), Q, B


def test_classify_cases():
    q = _dup_q(-1.0)
    ev, V, B = _spectrum_case([0.0, 0.004, 0.004, 0.006, 0.007, 0.25], True, q)
    assert wb.classify(ev, V, B, q) == "duplicate-unit minimum (Morse-Bott)"
    ev, V, B = _spectrum_case([1e-3, 0.004, 0.004, 0.006, 0.007, 0.25], True, q)
    assert wb.classify(ev, V, B, q) == "strict minimum"
    ev, V, B = _spectrum_case([-1e-4, 0.004, 0.004, 0.006, 0.007, 0.25], True, q)
    assert wb.classify(ev, V, B, q) == "saddle"
    ev, V, B = _spectrum_case([0.0, 0.004, 0.004, 0.006, 0.007, 0.25], False, q)   # zero mode not the transfer
    assert wb.classify(ev, V, B, q) == "degenerate, undetermined"
    q2 = q.copy(); q2[3] += 0.3                                                      # not a duplicate
    ev, V, B = _spectrum_case([0.0, 0.004, 0.004, 0.006, 0.007, 0.25], True, q2)
    assert wb.classify(ev, V, B, q2) == "degenerate, undetermined"
    ev, V, B = _spectrum_case([0.0, 0.0, 0.004, 0.006, 0.007, 0.25], True, q)       # two zero modes
    assert wb.classify(ev, V, B, q) == "degenerate, undetermined"
