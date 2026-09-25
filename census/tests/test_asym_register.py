import numpy as np
import pandas as pd

from src import asym_register as ar


def _scan(statuses, ties=None):
    s = np.array(ar.GRID[:len(statuses)])
    return pd.DataFrame({"s": s, "status": statuses, "tie": ties or [False] * len(s)})


def test_switch_of_cases():
    U, P, X = "unplaced", "placed", "undecided"
    v, br, _ = ar.switch_of(_scan([U, U, U, P, P, P]))
    assert v == "switch" and br == (ar.GRID[2], ar.GRID[3])
    assert ar.switch_of(_scan([P] * 6))[0] == "all_placed"
    assert ar.switch_of(_scan([U] * 6))[0] == "all_unplaced"
    assert ar.switch_of(_scan([U, P, U, P]))[0] == "alternation"
    assert ar.switch_of(_scan([P, P, U, U]))[0] == "alternation"          # placed -> unplaced is not a switch
    assert ar.switch_of(_scan([U, X, P, P]))[0] == "undecided"
    assert ar.switch_of(_scan([U, U, P, P], [False, True, False, False]))[0] == "undecided"


def test_ladder_and_validation():
    assert ar.ladder_ok([0.3, 0.3 + 5e-10, 0.3, 0.3], ["placed"] * 4)
    assert not ar.ladder_ok([0.3, 0.3 - 2e-9, 0.3, 0.3], ["placed"] * 4)
    assert not ar.ladder_ok([0.3] * 4, ["placed", "placed", "unplaced", "placed"])
    ok = {"ladder_ok": True, "independent_ok": True, "audit_ok": True}
    assert ar.validated([ok, ok])
    for k in ok:
        assert not ar.validated([ok, {**ok, k: False}])


def test_score_t2_3_cases():
    rng = np.random.default_rng(0)
    good = 1.0 * (1.05 + 0.03 * rng.random(60))
    assert ar.score_t2_3(good, 0.98, 1.0)["verdict"] == "PASS"
    assert ar.score_t2_3(good[:30], 0.98, 1.0)["verdict"] == "UNRESOLVED"
    below = np.r_[good[:50], 0.9 * np.ones(10)]                       # 50/60 = 0.83 above < 0.90
    assert ar.score_t2_3(below, 0.98, 1.0)["verdict"] == "FAIL"
    far = 1.4 * np.ones(60) + 0.01 * rng.random(60)                   # median/s_lo > 1.25
    assert ar.score_t2_3(far, 0.98, 1.0)["verdict"] == "FAIL"
    at = np.r_[np.ones(40), 1.01 * np.ones(20)]                        # median ratio 1: CI not above 0
    assert ar.score_t2_3(at, 0.98, 1.0)["verdict"] == "FAIL"


def test_training_set_geometry():
    x, y = ar.training_set(3)
    xo, xi = x[y == 1], x[y == 0]
    assert len(xo) == len(xi) == 200
    assert (np.abs(xi) <= 0.8).all()
    assert ((xo >= 1.2) & (xo <= 2.4 + 1e-6) | (xo <= -1.2) & (xo >= -2.0 - 1e-6)).all()
    fr = np.mean([(ar.training_set(s)[0][200:] > 0).mean() for s in range(50)])
    assert abs(fr - 1.2 / 2.0) < 0.02                                   # side chosen by length
    x0, _ = ar.training_set(3, delta=0.0)
    assert (np.abs(x0[200:]) <= 2.0 + 1e-6).all()


def test_matched_k_and_scale_invariance_of_placement():
    k = ar.k_matched(0.3, 0.32)
    assert abs(k - 0.09261 * 0.31 / 0.97946) < 1e-15
    from src.width2_train import init_params
    ar._setup()
    from src.width2_train import placed
    for seed in range(600_000, 600_020):                       # placement status invariant to scaling v
        q = init_params(seed).numpy().copy()
        q2 = q.copy(); q2[2] *= k; q2[5] *= k
        assert placed(q, ar._act())[0] == placed(q2, ar._act())[0]


def test_freeze_refuses_without_validated_bracket(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(ar, "PARTS", tmp_path)
    with pytest.raises(SystemExit):
        ar.freeze()
    pd.DataFrame([{"s": 0.3, "ladder_ok": True, "independent_ok": False, "audit_ok": True},
                  {"s": 0.32, "ladder_ok": True, "independent_ok": True, "audit_ok": True}]).to_csv(tmp_path / "validate.csv", index=False)
    with pytest.raises(SystemExit):
        ar.freeze()
