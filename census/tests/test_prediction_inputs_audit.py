"""The prediction-inputs audit's counting functions on constructed cases (no path, no registered run is read)."""

import math

import numpy as np
import pytest

from src import prediction_inputs_audit as A


def test_read_beyond_counts():
    # t_last vs t_obs: before (−3), at (0), at (0), after by 1, after by 5; one run without a value
    r = A.read_beyond([7, 10, 20, 31, 45, np.nan], [10, 10, 20, 30, 40, 50])
    assert r["n"] == 5
    assert (r["n_before"], r["n_at"], r["n_strictly_after"], r["n_at_or_after"]) == (1, 2, 2, 4)
    assert r["max_steps_after"] == 5.0
    assert r["median_steps_after_among_after"] == 3.0
    assert (r["d_min"], r["d_median"], r["d_max"]) == (-3.0, 0.0, 5.0)


def test_read_beyond_none_after():
    r = A.read_beyond([1, 2], [5, 5])
    assert r["n_at_or_after"] == 0 and r["max_steps_after"] == 0.0 and r["median_steps_after_among_after"] is None


def test_sdot_window():
    # t_sw = 50 (w = 50, window [0, 50]); t_obs 60: ends before.  t_sw 300, t_obs 250: straddles.
    # t_sw 400, t_obs 250: starts at 300 ≥ 250, entirely after.  t_sw 250 = t_obs: ends at the crossing (straddles).
    r = A.sdot_window([50, 300, 400, 250], [60, 250, 250, 250])
    assert r["n"] == 4
    assert r["n_end_at_or_after"] == 3
    assert r["n_entirely_at_or_after"] == 1
    assert r["n_straddling"] == 2
    assert r["max_steps_tsw_after"] == 150.0


def test_sdot_window_short_t_sw_uses_min_window():
    # t_sw = 30 < 100: w = 30, window [0, 30]; t_obs = 0 → entirely at or after
    r = A.sdot_window([30], [0])
    assert r["n_entirely_at_or_after"] == 1


def test_extrapolate_linear_1d_and_2d():
    x = np.arange(300, dtype=float) ** 2
    y = A.extrapolate_linear(x, 200, w=100)
    assert np.array_equal(y[:201], x[:201])
    slope = (x[200] - x[100]) / 100
    assert np.allclose(y[201:], x[200] + slope * np.arange(1, 100))
    X = np.stack([x, -x], 1)
    Y = A.extrapolate_linear(X, 200, w=100)
    assert np.array_equal(Y[:, 0], y) and np.array_equal(Y[:, 1], -y)
    assert np.array_equal(X[:, 0], x)                       # the input is not modified
    with pytest.raises(ValueError):
        A.extrapolate_linear(x, 50, w=100)


def test_extrapolate_exact_on_a_line():
    x = 3.0 + 0.25 * np.arange(500)
    assert np.allclose(A.extrapolate_linear(x, 250), x, rtol=0, atol=1e-12)


def test_first_ge():
    s = np.array([0.0, 1.0, 0.5, 2.0, 3.0])
    assert A.first_ge(s, 2.0) == 3
    assert A.first_ge(s, 0.5, 1) == 1
    assert A.first_ge(s, 9.0) is None


def test_rel_equal():
    assert A.rel_equal(6004, 6004.0)
    assert A.rel_equal(0.0459689715156018, 0.04596897151560186)         # 16 vs 17 significant digits
    assert not A.rel_equal(0.0459689715156018, 0.04596897151561)
    assert A.rel_equal(0.0007189570137939917, 0.0007189570137939)       # 16 decimal places kept (< 1e-16)
    assert not A.rel_equal(0.0007189570137941, 0.0007189570137939)
    assert A.rel_equal(None, float("nan")) and not A.rel_equal(None, 1.0) and not A.rel_equal(1.0, None)
    assert A.rel_equal(True, 1.0) and not A.rel_equal(False, 1.0)


def test_spearman_monotone():
    assert A.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert A.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_ta_inputs_masks_everything_but_the_named_rows():
    n = 12
    W = np.arange(n * 4, dtype=float).reshape(n, 4)
    Mr = np.full((n, 4), 7.0)
    Vr = np.full((n, 4), 9.0)
    Wm, M, V, K = A._ta_inputs(W, 5, [5, 8], Mr, Vr, adam=True)
    hid = [0, 1, 3]
    assert np.array_equal(Wm[:, 2], W[:, 2])                         # the output weight is untouched
    assert np.array_equal(Wm[5, hid], W[5, hid])                     # the rule-point state is kept
    others = np.delete(np.arange(n), 5)
    assert np.isnan(Wm[np.ix_(others, hid)]).all()
    assert np.array_equal(M[5], Mr[5]) and np.isnan(np.delete(M, 5, 0)).all()
    assert np.array_equal(V[5], Vr[5]) and np.array_equal(V[8], Vr[8])
    assert np.isnan(np.delete(V, [5, 8], 0)).all()
    assert np.array_equal(K, np.arange(n))
    Wm, M, V, K = A._ta_inputs(W, 5, [5], None, None, adam=False)    # SGD: no moment is passed at all
    assert np.isnan(M).all() and np.isnan(V).all() and not K.any()


def test_code_refs_check(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("a = 1\nb = compute(a)\n")
    refs = (("X", "p", "input", A.FROZEN, "src/m.py", 2, "b = compute("),
            ("X", "p", "input", A.FROZEN, "src/m.py", 1, "b = compute("),
            ("X", "p", "input", A.FROZEN, "src/m.py", 9, "a = 1"))
    bad = A.code_refs_check(refs, root=tmp_path)
    assert len(bad) == 2 and "src/m.py:1" in bad[0] and "src/m.py:9" in bad[1]


def test_registered_code_refs_hold():
    assert A.code_refs_check() == []


def test_clean_nan_to_none():
    assert A._clean({"a": float("nan"), "b": np.int64(3), "c": [np.float64(1.5), math.inf]}) == \
        {"a": None, "b": 3, "c": [1.5, None]}


def test_last_read_elementwise_max_ignoring_nan():
    r = A.last_read([10, 5, np.nan, np.nan], [3, 8, 4, np.nan])
    assert np.array_equal(r[:3], [10.0, 8.0, 4.0]) and np.isnan(r[3])
    assert np.array_equal(A.last_read([1, 9], [2, 3], [7, 0]), [7.0, 9.0])


def test_ratio_split_identity_and_sides():
    # run 0: t_traj = t_obs, s identical, r differs only by rounding (identity);
    # run 1: t_traj = t_obs (identity, exact); run 2: after, ratio < 1; run 3: before, ratio > 1;
    # run 4: t_traj != t_obs but ratio exactly 1; run 5: no prediction (dropped)
    r = A.ratio_split(t_traj=[10, 20, 31, 38, 50, np.nan], t_obs=[10, 20, 30, 40, 51, 60],
                      s_traj=[1.5, 2.0, 2.2, 1.9, 3.0, np.nan], s_obs=[1.5, 2.0, 2.1, 2.0, 3.0, 1.0],
                      r_traj=[1e-5, 0.2, 0.3, 0.1, 0.4, np.nan], r_obs=[1e-5 * (1 + 5e-12), 0.2, 0.2, 0.15, 0.4, 0.1])
    assert r["n"] == 5
    assert (r["n_t_traj_equal_t_obs"], r["n_identity_s_traj_identical_s_obs"]) == (2, 2)
    assert 4e-12 < r["max_abs_ratio_minus_1_among_identity"] < 6e-12
    assert (r["n_other"], r["n_other_ratio_below_1"], r["n_other_ratio_above_1"], r["n_other_ratio_exactly_1"]) == \
        (3, 1, 1, 1)


def test_ratio_split_identity_with_different_s_is_flagged():
    # t_traj = t_obs but s_traj != s_obs would break the identity: counted in n_t_traj_equal_t_obs only
    r = A.ratio_split([5], [5], [1.0], [1.1], [0.1], [0.2])
    assert r["n_t_traj_equal_t_obs"] == 1 and r["n_identity_s_traj_identical_s_obs"] == 0


def test_truncate_after():
    X = np.arange(20, dtype=float).reshape(10, 2)
    Y = A.truncate_after(X, 4)
    assert np.array_equal(Y[:5], X[:5]) and np.isnan(Y[5:]).all()
    assert not np.isnan(X).any()                                   # the input is not modified
    y = A.truncate_after(np.arange(5.0), 4)
    assert not np.isnan(y).any()                                   # L = last row: nothing removed


def test_md_refs_and_check():
    text = ("r_traj = s[t] (`width2_asym.py:1396`); the switch (width2_asym.py:1370, gelu_transfer.py:849); "
            "again width2_asym.py:1396; not a ref: width2_asym.py and 12:30.")
    assert A.md_refs(text) == [("gelu_transfer.py", 849), ("width2_asym.py", 1370), ("width2_asym.py", 1396)]
    refs = (("X", "p", "i", A.OUTPUT, "src/width2_asym.py", 1396, "tok"),
            ("X", "p", "i", A.OUTPUT, "src/gelu_transfer.py", 849, "tok"))
    assert A.md_refs_check(text, refs) == ["width2_asym.py:1370"]
    assert A.md_refs_check("nothing cited", refs) == []


def test_audit_md_cites_only_checked_lines():
    assert A.md_refs_check(A.AUDIT_MD.read_text()) == []


def test_committed_json_regenerates():
    import json
    assert json.loads(A.OUT_JSON.read_text()) == json.loads(json.dumps(A.build()))
