import math

import numpy as np
import pandas as pd
import pytest

from src.baselines_posthoc import (BASELINES, abslogerr, fit_baselines, ols, predict_baselines, setting_compare,
                                   window_compare)


def _settings(lam=(1.1, 1.2), slope=0.2, gslope=0.3):
    wins = ("base", "G1_wide_gap", "G2_narrow_gap", "G3_far_outer", "G4_shifted")
    kap = (0.30, 0.60, 0.07, 0.25, 0.55)
    rows, runs = [], []
    for a, l in zip((1.30, 1.50), lam):
        for i, (w, k) in enumerate(zip(wins, kap)):
            rg = 0.15 + 0.02 * i
            G = 0.05 * (i + 1) * (1 + (a == 1.50))
            # median crossing R constructed to satisfy all three model families exactly is impossible, so each
            # test below checks one family on data built for it
            rows.append({"window": w, "a": a, "R_glob": rg, "kappa0": k, "Ghat": G, "median_cross_R": l * rg,
                         "n_crossed": 3})
            for f in (0.9, 1.0, 1.1):
                runs.append({"window": w, "a": a, "R": f * l * rg})
    return pd.DataFrame(rows), pd.DataFrame(runs)


def test_pooled_lag_recovers_constant_ratio():
    s, r = _settings()
    fit = fit_baselines(s, r)
    assert abs(fit["PL"][1.30] - 1.1) < 1e-12 and abs(fit["PL"][1.50] - 1.2) < 1e-12
    assert abs(fit["PL5"][1.30] - 1.1) < 1e-12 and abs(fit["PL5"][1.50] - 1.2) < 1e-12
    p = predict_baselines(fit, 1.3, 0.2, 0.3, 0.1)
    assert abs(p["PL"] - 0.22) < 1e-12


def test_regressions_recover_noiseless_coefficients():
    s, r = _settings()
    s["median_cross_R"] = np.exp(-1.0 + 0.25 * np.log(s.kappa0) + 0.1 * (s.a == 1.50))
    fit = fit_baselines(s, r)
    assert np.allclose(fit["RK"], [-1.0, 0.25, 0.1], atol=1e-12)
    for a in (1.30, 1.50):
        assert np.allclose(fit["RKa"][a], [-1.0 + 0.1 * (a == 1.50), 0.25], atol=1e-12)
    s["median_cross_R"] = np.exp(-0.5 + 0.3 * np.log(s.Ghat))
    fit = fit_baselines(s, r)
    for a in (1.30, 1.50):
        assert np.allclose(fit["RG"][a], [-0.5, 0.3], atol=1e-12)
    p = predict_baselines(fit, 1.5, 0.2, 0.3, 0.2)
    assert abs(p["RG"] - math.exp(-0.5 + 0.3 * math.log(0.2))) < 1e-12
    assert set(p) == set(BASELINES)


def test_fit_refuses_non_blockG_windows():
    s, r = _settings()
    s.loc[0, "window"] = "H10"
    with pytest.raises(ValueError):
        fit_baselines(s, r)
    s, r = _settings()
    r.loc[0, "window"] = "V10"
    with pytest.raises(ValueError):
        fit_baselines(s, r)


def test_ols_exact():
    X = np.column_stack([np.ones(4), [1.0, 2.0, 3.0, 5.0]])
    assert np.allclose(ols(X, 2 - 3 * X[:, 1]), [2, -3], atol=1e-12)


def _block3(obs, preds):
    m = pd.DataFrame({"obs_median_cross_R": obs, **preds})
    for k in preds:
        m[f"abslogerr_{k}"] = abslogerr(m[k], m.obs_median_cross_R)
    return m


def test_perfect_baseline_has_zero_error_and_beats_C():
    obs = np.array([0.21, 0.23, 0.24, 0.27, 0.22, 0.25, 0.26, 0.28])
    m = _block3(obs, {"C": obs * 1.05, "X": obs.copy()})
    assert np.all(m.abslogerr_X == 0)
    c = setting_compare(m, ("X",)).iloc[0]
    assert c.mean_diff > 0 and c.ci95_lo > 0 and c.X_better_interval_excludes_zero
    assert not c.C_better_interval_excludes_zero


def test_constant_baseline_loses_to_tracking_model():
    obs = np.array([0.14, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36])
    rng = np.random.default_rng(3)
    m = _block3(obs, {"C": obs * np.exp(rng.normal(0, 0.01, 8)), "K": np.full(8, np.median(obs))})
    c = setting_compare(m, ("K",)).iloc[0]
    assert c.ci95_hi < 0 and c.C_better_interval_excludes_zero


def test_identical_models_tie():
    obs = np.array([0.2, 0.25, 0.3])
    m = _block3(obs, {"C": obs * 1.1, "X": obs * 1.1})
    c = setting_compare(m, ("X",)).iloc[0]
    assert c.mean_diff == 0 and c.ci95_lo == 0 and c.ci95_hi == 0
    assert not c.C_better_interval_excludes_zero and not c.X_better_interval_excludes_zero


def test_window_compare_pass_and_fail():
    rng = np.random.default_rng(0)
    rows = []
    for w in range(8):
        shift = rng.normal(0, 0.05)
        for _ in range(40):
            e = rng.normal(0, 0.03)
            rows.append({"window": f"V{w}", "e_C": e, "e_X": e + 0.2 + shift, "e_Y": e})
    pr = pd.DataFrame(rows)
    out = window_compare(pr, "C", "X", B=2000)
    assert out["ci95_hi"] < 0 and out["C_better_interval_excludes_zero"]
    tie = window_compare(pr, "C", "Y", B=2000)
    assert tie["mean_diff"] == 0 and not tie["C_better_interval_excludes_zero"]
    rev = window_compare(pr.rename(columns={"e_C": "e_X", "e_X": "e_C"}), "C", "X", B=2000)
    assert rev["ci95_lo"] > 0 and rev["X_better_interval_excludes_zero"]
