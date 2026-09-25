import numpy as np

from src import asym_t23d as td


def test_score_cases():
    rng = np.random.default_rng(0)
    own = 4.8 * np.exp(0.1 * rng.standard_normal(60))
    good = own * np.exp(0.02 * rng.standard_normal(60))
    assert td.score_arrays(good, own)["verdict"] == "PASS"
    loose = own * 1.2                                           # tracks own but 18% off: criterion (i) fails
    r = td.score_arrays(loose, own); assert r["verdict"] == "FAIL" and not r["criterion_i"] and r["criterion_ii"]
    near_pop = np.full(60, 0.46)                                 # sits at the population width-2 threshold
    assert not td.score_arrays(near_pop, own)["criterion_ii"]
    assert td.score_arrays(good[:30], own[:30])["verdict"] == "UNRESOLVED"
    nan_own = own.copy(); nan_own[:30] = np.nan                   # undefined thresholds are dropped, then too few
    assert td.score_arrays(good, nan_own)["verdict"] == "UNRESOLVED"
