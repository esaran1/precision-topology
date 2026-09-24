import math

import numpy as np

from src.width2_nogating import (PILOT_EVERY, PILOT_STEPS, act_of, breakdown, choose_H, control_ok, exclusion_ok,
                                 score_levels, settling_time)

N_CHECKS = PILOT_STEPS // PILOT_EVERY


def test_settling_time():
    tr = [False] * 30 + [True] * (N_CHECKS - 30)
    assert settling_time(tr) == 31 * PILOT_EVERY
    assert settling_time([True] * N_CHECKS) == PILOT_EVERY                  # never changes
    late = [False] * (N_CHECKS - 50) + [True] * 50                           # changes in the final 10%
    assert settling_time(late) == PILOT_STEPS
    flicker = [False] * 30 + [True, False] * 10 + [True] * (N_CHECKS - 50)
    assert settling_time(flicker) == 51 * PILOT_EVERY                       # last change at check index 50
    und = [False] * 30 + [None] + [True] * (N_CHECKS - 31)                   # an undecided check counts as a change
    assert settling_time(und) == 32 * PILOT_EVERY


def test_choose_H():
    assert choose_H([1000] * 40)[0] == 16_000
    assert choose_H([5000] * 40)[0] == 32_000
    assert choose_H([30_000] * 40)[0] == 128_000
    assert choose_H([40_000] * 40)[0] is None                                # stop


def _mat(fracs, n=80, seed=0):
    rng = np.random.default_rng(seed)
    out, base = {}, rng.random(n)
    for L, f in fracs.items():
        out[L] = (base < f).astype(int)                                      # nested: monotone in f
    return {L: float(v.mean()) for L, v in out.items()}, out


def test_score_levels_outcomes():
    fr, m = _mat({0.003: 0.97, 0.01: 0.97, 0.03: 0.98, 0.1: 1.0})
    assert score_levels(fr, 80, m)[0] == "no gating (predicted)"
    fr, m = _mat({0.003: 0.2, 0.01: 0.5, 0.03: 0.8, 0.1: 0.95})
    assert score_levels(fr, 80, m)[0] == "gating at small scale (competing)"
    fr, m = _mat({0.003: 0.7, 0.01: 0.8, 0.03: 0.9, 0.1: 0.95})
    assert score_levels(fr, 80, m)[0] == "neither"
    # a significant decrease between levels blocks the competing outcome
    m = {0.003: np.r_[np.ones(40), np.zeros(40)].astype(int), 0.01: np.zeros(80, int),
         0.03: np.ones(80, int), 0.1: np.ones(80, int)}
    fr = {L: float(v.mean()) for L, v in m.items()}
    assert score_levels(fr, 80, m)[0] == "neither"
    # at n = 80 the fraction binds: 72/80 = 0.90 has CP lower 0.812 (passes); 71/80 = 0.8875 fails the fraction
    def exact(k):
        return np.r_[np.ones(k), np.zeros(80 - k)].astype(int)
    m = {0.003: exact(73), 0.01: exact(76), 0.03: exact(76), 0.1: exact(76)}
    fr = {L: float(v.mean()) for L, v in m.items()}
    out, lo = score_levels(fr, 80, m)
    assert lo[0.003] >= 0.80 and out == "no gating (predicted)"
    m[0.003] = exact(72)
    fr = {L: float(v.mean()) for L, v in m.items()}
    assert score_levels(fr, 80, m)[0] == "no gating (predicted)"
    m[0.003] = exact(71)
    fr = {L: float(v.mean()) for L, v in m.items()}
    assert score_levels(fr, 80, m)[0] == "neither"


def test_stops():
    assert control_ok(0.2) and not control_ok(0.25)
    assert exclusion_ok(16, 80) and not exclusion_ok(17, 80)


def test_breakdown_pair_vs_single():
    act = act_of("f1.30")
    al = 1.7913244
    # the cancelling cosine pair: v = (½, −½)·s, β = −π/2 and +π/2 (sign(D*) < 0), α₁ = α₂ = α*
    q = np.array([al, -math.pi / 2, 0.05, al, math.pi / 2, -0.05, 0.0])
    b = breakdown(q, act)
    assert b["knockout"] == "shared" and b["pair"] and b["pair_strict"] and b["cancel_index"] < 1e-12
    # a single unit (second unit's weight 0): not a pair
    q1 = np.array([al, -math.pi / 2, 0.1, 3.0, 0.3, 0.0, 0.0])
    b1 = breakdown(q1, act)
    assert not b1["pair"] and not b1["pair_strict"] and b1["knockout"] in ("unplaced", "single-unit")
    # the pair with unequal |α| beyond 10%: not a pair
    q2 = np.array([al, -math.pi / 2, 0.05, 1.3 * al, math.pi / 2, -0.05, 0.0])
    assert not breakdown(q2, act)["pair"]


def test_endpoint_types():
    from src.width2_nogating import endpoint_type
    pair = {"pair": True, "share1": 0.5, "share2": 0.5}
    single = {"pair": False, "share1": 0.999, "share2": 0.001}
    two = {"pair": False, "share1": 0.6, "share2": 0.4}
    assert endpoint_type(True, pair, 1e-3) == "placed pair"
    assert endpoint_type(False, single, 1e-8) == "single-unit local minimum"
    assert endpoint_type(False, single, 1e-3) == "other: unplaced single unit, not stationary"
    assert endpoint_type(True, two, 1e-8) == "other: placed, not the pair"
    assert endpoint_type(False, two, 1e-8) == "other: unplaced, two units"
    assert endpoint_type(None, two, 1e-8) == "other: undecided"


def test_tangent_gradient_matches_a_finite_difference_along_the_sphere():
    import numpy as np
    import torch
    from src.width2_conditional import population
    from src.width2_nogating import act_of, tangent_grad
    from src.width2_train import logits
    x, y = population()
    act = act_of("f1.30")
    q = np.array([1.79, -1.2, 0.004, 2.3, 0.3, -0.002, 0.1])
    g, gv = tangent_grad(q, act, x, y)
    t = np.array([np.sign(q[2]), -np.sign(q[5])]) / np.sqrt(2)      # tangent to |v1| + |v2| = const (fixed signs)
    X, Y = torch.tensor(x), torch.tensor(y)
    L = lambda qq: float(torch.nn.functional.binary_cross_entropy_with_logits(logits(torch.tensor(qq), X, act), Y))
    h = 1e-6
    e = np.zeros(7); e[2], e[5] = t
    fd = (L(q + h * e) - L(q - h * e)) / (2 * h)
    assert abs(fd - gv @ t) < 1e-8
    n = np.sign([q[2], q[5]]) / np.sqrt(2)
    assert abs(gv @ n) < 1e-15                                       # the normal component is removed
