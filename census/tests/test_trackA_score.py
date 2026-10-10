"""The Track A scoring entry point (post-registration plumbing fix, author 2026-10-10): the full path from table() to the
verdict on constructed rows.  The registered path (score_track_a on table()'s full output) crashes on the string column
"status"; the entry point passes only the documented columns and gives the same verdict as score_track_a on the
constructed arrays directly."""
import numpy as np
import pytest

from src import trackA_causal as T
from src import trackA_score as S

TOL = T.TOLERANCES
FK = str(T.F)


def _rows(n=100, err=1, lag=30, miss=0, nan_bad=0, c4_bad=False):
    """Run rows and observation rows in the format run() and observe() write.  Miss rows carry a non-"ok" status."""
    rows, obs = [], []
    for i in range(n):
        t_sw = 1000
        t_obs = t_sw + lag
        fc = {"t_fc": t_obs + err, "t_sw_fc": t_obs if c4_bad else t_sw, "r_fc": 0.1, "status": "ok",
              "nan_recompute_identical": i >= nan_bad, "P_hat": [1.0, 1.0, 1.0]}
        if i < miss:
            fc = {"status": "no forecast: no predicted crossing on the extrapolated path",
                  "nan_recompute_identical": True}
        rows.append({"seed": 9_776_000 + i, "status": "ok", "fc": {FK: {"t_c": t_sw - 50, "primary": fc}}})
        obs.append({"seed": 9_776_000 + i, "crossed": True, "t_obs": t_obs, "s_obs": 1.1,
                    "by_f": {FK: {"t_sw": t_sw, "s_sw": 1.0, "P_true_at_t_sw": [1.0, 1.0, 1.0]}}})
    return rows, obs


def test_registered_path_crashes_on_the_status_column():
    rows, obs = _rows()
    with pytest.raises(ValueError, match="could not convert string to float"):
        T.score_track_a(T.table(rows, obs, T.F, "primary"), TOL)


def test_documented_columns_are_the_docstring_columns():
    rows, obs = _rows()
    R = S.documented_columns(T.table(rows, obs, T.F, "primary"))
    assert set(R) == set(S.SCORE_COLUMNS)
    assert "status" not in R and "P_relerr" not in R
    for k in ("frozen_ok", "crossed", "t_obs", "s_obs", "t_sw", "s_sw", "t_c", "t_fc", "t_sw_fc", "r_fc",
              "nan_identical"):
        assert k in (T.score_track_a.__doc__ or "")


@pytest.mark.parametrize("kw,outcome_has,c", [
    ({}, "PASS", None),
    ({"err": 11}, "FAIL", "C1"),
    ({"c4_bad": True}, "FAIL", "C4"),
    ({"nan_bad": 1}, "UNRESOLVED", None),
])
def test_full_path_verdicts(kw, outcome_has, c):
    rows, obs = _rows(**kw)
    res = S.score_table(rows, obs, TOL)
    assert outcome_has in res["outcome"]
    if c:
        assert res[c]["verdict"] == "FAIL"


def test_full_path_misses_count_against_c1_c2():
    rows, obs = _rows(miss=25)                                  # 75/100 within < 80%
    res = S.score_table(rows, obs, TOL)
    assert res["n_scored_miss"] == 25
    assert res["C1"]["verdict"] == "FAIL" and res["C2"]["verdict"] == "FAIL"


def test_full_path_equals_score_track_a_on_the_arrays():
    rows, obs = _rows(err=3, miss=5)
    via = S.score_table(rows, obs, TOL)
    R = S.documented_columns(T.table(rows, obs, T.F, "primary"))
    direct = T.score_track_a({k: np.asarray(v, bool if k in ("crossed", "frozen_ok", "nan_identical") else float)
                              for k, v in R.items()}, TOL)
    for k in ("C1", "C2", "C3", "C4", "outcome", "validity", "n_scored", "n_scored_miss"):
        assert via[k] == direct[k]
