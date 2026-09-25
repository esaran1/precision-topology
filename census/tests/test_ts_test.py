import numpy as np

from src import ts_test as tt


def test_score_cases():
    rng = np.random.default_rng(0)
    own = 3.0 * np.exp(0.1 * rng.standard_normal(60))
    r = np.full(60, 0.01)                                      # pred = 0.0157 + 2.658*0.01 = 0.0423
    good = own * 1.042 * np.exp(0.002 * rng.standard_normal(60))
    s = tt.score_a(good, own, 3.0, r, 0.0157, 2.658)
    assert s["TS-1"] == "PASS" and s["TS-2"] == "PASS"
    far = own * 1.2
    assert tt.score_a(far, own, 3.0, r, 0.0157, 2.658)["TS-1"] == "FAIL"
    pop_like = np.full(60, 3.1)                               # tracks the population, not own
    assert tt.score_a(pop_like, own, 3.0, r, 0.0157, 2.658)["TS-2"] == "FAIL"
    assert tt.score_a(good[:20], own[:20], 3.0, r[:20], 0.0157, 2.658)["TS-1"] == "UNRESOLVED"
    bad_r = np.full(60, -1.0)                                  # unusable ratios: TS-1 unresolved
    assert tt.score_a(good, own, 3.0, bad_r, 0.0157, 2.658)["TS-1"] == "UNRESOLVED"
