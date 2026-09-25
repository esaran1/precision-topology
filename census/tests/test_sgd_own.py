import numpy as np

from src import sgd_own as so


def test_choose_budget_cases():
    pop = {1.3: 5.0, 1.5: 2.5}
    run = lambda v13, v15: {1.3: [dict(zip(so.BUDGETS, v13))] * 20, 1.5: [dict(zip(so.BUDGETS, v15))] * 20}
    assert so.choose_budget(run((8, 9, 10, 11), (4, 5, 6, 7)), pop) == 16_000
    assert so.choose_budget(run((5, 7.6, 10, 11), (3, 4, 6, 7)), pop) == 32_000
    assert so.choose_budget(run((1, 2, 3, 4), (1, 2, 3, 4)), pop) == 128_000       # never reached: largest
    mixed = {1.3: [dict(zip(so.BUDGETS, (8, 9, 10, 11)))] * 18 + [dict(zip(so.BUDGETS, (1, 1, 1, 8)))] * 2,
             1.5: [dict(zip(so.BUDGETS, (4, 5, 6, 7)))] * 20}
    assert so.choose_budget(mixed, pop) == 128_000                                   # 90% < 95% until 128k


def test_score_cases():
    rng = np.random.default_rng(0)
    own = 5 * np.exp(0.15 * rng.standard_normal(40))
    good = own * 1.04 * np.exp(0.005 * rng.standard_normal(40))
    s = so.score_a(good, own, 5.0)
    assert s["G1"] == "PASS" and s["G2"] == "PASS"
    noise = 5.2 * np.exp(0.01 * rng.standard_normal(40))                            # tracks pop, not own
    s = so.score_a(noise, own, 5.0)
    assert s["G1"] == "FAIL" and s["G2"] == "FAIL"
    assert so.score_a(good[:20], own[:20], 5.0)["G1"] == "UNRESOLVED"


def test_init_matches_phase2b():
    import torch
    for seed in (0, 7):
        _, _, _, th = so._setup(1.3, seed)
        torch.manual_seed(seed)
        ref = torch.empty(4).uniform_(-1.0, 1.0).double()
        assert torch.equal(th.detach(), ref)
