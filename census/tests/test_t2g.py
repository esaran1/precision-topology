import numpy as np
import pandas as pd

from src import t2g


def test_truth_and_prediction():
    assert t2g.truth(True, 0.05) == "EARLY" and t2g.truth(True, 4.8) == "LATE"
    assert t2g.truth(False, np.nan) == "LATE" and t2g.truth(True, 1.0) == "LATE"
    assert t2g.prediction("EARLY", 0.04, 4.8) == 0.04 and t2g.prediction("LATE", 0.04, 4.8) == 4.8


def _gate_df(n=100, frac_early=0.4, acc=1.0, late_err=0.02, seed=0):
    rng = np.random.default_rng(seed)
    ne = int(n * frac_early)
    tru = np.array(["EARLY"] * ne + ["LATE"] * (n - ne))
    pred = tru.copy()
    nflip = int(round(n * (1 - acc)))
    pred[:nflip] = np.where(pred[:nflip] == "EARLY", "LATE", "EARLY")
    s_own = 4.8 * np.exp(0.1 * rng.standard_normal(n))
    s_cross = np.where(tru == "EARLY", 0.06, s_own * np.exp(late_err * rng.choice([-1, 1], n)))
    return pd.DataFrame({"class_pred": pred, "truth": tru, "crossed": True, "s_cross": s_cross, "s_own": s_own})


def test_gate_cases():
    assert t2g.gate_decision(_gate_df())["gate"] == "PASS"
    assert t2g.gate_decision(_gate_df(acc=0.93))["gate"] == "FAIL"                 # accuracy below 0.95
    assert t2g.gate_decision(_gate_df(late_err=0.2))["gate"] == "FAIL"             # late-class error above 0.10
    assert t2g.gate_decision(_gate_df(n=30, frac_early=0.4))["gate"] == "FAIL"     # fewer than 20 predicted-late runs
    d = _gate_df(); d.loc[d.truth == "LATE", "s_own"] = np.nan                        # own undefined: not scored
    assert t2g.gate_decision(d)["gate"] == "FAIL"


def _reg_df(n=80, acc=1.0, late_err=0.02, early_err=0.3, frac_early=0.4, crossed=True, seed=1):
    d = _gate_df(n, frac_early, acc, late_err, seed)
    d["s_pred"] = np.where(d.class_pred == "EARLY", d.s_cross * np.exp(early_err), d.s_own)
    d["crossed"] = crossed
    return d


def test_registered_cases():
    r = t2g.score_registered(_reg_df(), 0, 80)
    assert r["verdict"] == "PASS" and r["criterion_i"] and r["criterion_ii"] and r["criterion_iii"]
    r = t2g.score_registered(_reg_df(acc=0.85), 0, 80)                               # (i) fails
    assert r["verdict"] == "FAIL" and not r["criterion_i"]
    r = t2g.score_registered(_reg_df(late_err=0.3), 0, 80)                           # (ii) fails
    assert r["verdict"] == "FAIL" and not r["criterion_ii"]
    d = _reg_df(); d["s_pred"] = 0.44508 * np.exp(0.5)                               # constant worse than population: (iii)
    r = t2g.score_registered(d, 0, 80)
    assert not r["criterion_iii"] and r["verdict"] == "FAIL"
    assert t2g.score_registered(_reg_df(n=30), 0, 30)["verdict"].startswith("UNRESOLVED")
    d = _reg_df(); d.loc[:50, "crossed"] = False
    assert t2g.score_registered(d, 0, 80)["verdict"].startswith("UNRESOLVED")        # fewer than 40 crossings
    assert t2g.score_registered(_reg_df(), 20, 80)["verdict"].startswith("STOP")     # 25% placed at step 0


def test_classifier_smoke_on_unused_seed():
    r = t2g.classify(999_999, relax_steps=50)
    assert r["class_pred"] in ("EARLY", "LATE", "EXCLUDED")
    if r["class_pred"] == "EARLY":
        assert np.isfinite(r["s_pred_early"]) and r["s_pred_early"] < t2g.S_CUT
