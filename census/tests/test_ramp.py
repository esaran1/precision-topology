import numpy as np

from src import ramp as R


def _cells(slope=1.0, gammas=(1, 2, 4, 8, 16, 32), n=40, noise=0.0, wp=0, rng=np.random.default_rng(0)):
    out = []
    for g in gammas:
        pred = np.full(n, 0.001 * g)
        out.append({"gamma": g, "obs_r": slope * pred + noise * rng.standard_normal(n), "pred_r": pred, "n_runs": n,
                    "n_warmup_placed": wp})
    return out


def test_score_setting_cases():
    s = R.score_setting(_cells(1.0))
    assert s["R1"] == "PASS" and s["R2"] == "PASS" and s["R3"] == "PASS"
    assert R.score_setting(_cells(1.5))["R1"] == "FAIL"                        # slope outside [0.7, 1.3]
    flat = _cells(0.0)
    for c in flat:
        c["obs_r"] = np.full(40, 0.02)
    f = R.score_setting(flat)
    assert f["R2"] == "FAIL" and f["R3"] == "FAIL"                               # no increase; slowest not small
    assert R.score_setting(_cells(1.0, n=20))["R1"] == "UNRESOLVED"              # < 30 crossings per cell
    assert R.score_setting(_cells(1.0, wp=10))["R1"] == "STOP"                   # > 20% placed in warm-up


def test_score_r4_cases():
    rng = np.random.default_rng(0)
    base = 0.03 + 0.002 * rng.standard_normal(40)
    assert R.score_r4({0.01: base, 0.005: base + 0.001, 0.0025: base - 0.001})["R4"] == "PASS"
    assert R.score_r4({0.01: base, 0.005: base * 0.2, 0.0025: base * 0.05})["R4"] == "FAIL"   # lag vanishing with η
    assert R.score_r4({0.01: base[:20], 0.005: base, 0.0025: base})["R4"] == "UNRESOLVED"


def test_slope_through_origin():
    x = np.array([1.0, 2.0, 3.0])
    assert abs(R.slope_through_origin(2 * x, x) - 2.0) < 1e-12


def test_score_setting_negative_kappa():
    cells = _cells(1.0)
    for c in cells:
        c["pred_r"] = -c["pred_r"]; c["obs_r"] = -c["obs_r"]
    s = R.score_setting(cells)
    assert s["R1"] == "PASS" and s["R2"] == "PASS" and s["R3"] == "PASS"
    for c in cells:
        c["obs_r"] = -c["obs_r"]                                   # wrong sign
    s = R.score_setting(cells)
    assert s["R1"] == "FAIL" and s["R2"] == "FAIL"


def test_batch_equals_single_and_branch_init():
    """Batched == single-seed training: exact (≤1e-12) for SGD through the ramp start and for both optimisers through
    1,000 warm-up steps.  (Adam from the branch point later reaches roundoff-scale gradients, where m/√v̂ amplifies
    one-ulp reduction differences to ~1e-5; recorded in the registration.)"""
    import numpy as np
    z = R.branch_init(1.3, -1); z0 = R.branch_init(1.3, 0)
    assert abs(z[1] - z0[1] + 2 * np.pi) < 1e-12 and abs(z[2] - z0[2] - 2 * np.pi * R.S0_FRAC * 4.958011429378877) < 1e-6
    seeds = (869_000, 869_001, 869_002)
    for opt, steps in (("adam", 1000), ("sgd", 1000), ("sgd", R.WARMUP + 50)):
        ref = R.single_ramp_reference(1.3, seeds[1], opt, 1e-3, 4.95625, steps, winding=-1)
        res = R.batch_ramp(1.3, seeds, opt, 1e-3, 4.95625, max_steps=steps, winding=-1)
        assert all(not x["placed_in_warmup"] for x in res)
        got = np.array([res[1]["w1_end"], res[1]["b1_end"], res[1]["b2_end"]])
        assert np.max(np.abs(got - ref)) < 1e-12, (opt, steps, got, ref)
