"""Test 2C (registered; band task in R^d against each seed's own-sample R^d switch): every decision rule on constructed
pass, fail and unresolved cases (C1–C3, the bootstrap rule, validity, UNRESOLVED propagation), the t_sw and rule-point
rules, the own-sample switch search and its validation rules, and the separation of predictions from crossings."""

import math

import numpy as np
import pytest

from src import track2c as C


# ------------------------------------------------------------------------------------------ constants
def test_registered_constants():
    assert C.SEEDS == tuple(range(2_030_000, 2_030_060)) and len(C.SEEDS) == 60
    assert C.A_VALUES == (1.30, 1.50) and C.D_VALUES == (2, 4) and C.BUDGET == 64_000 and C.LR == 0.01
    assert (C.C1_MAX, C.C2_BAND, C.MIN_CROSS, C.RHO_MAX, C.RHO_FRAC, C.PRED_FRAC) == (0.05, (0.75, 1.25), 30, 0.05, 0.90, 0.90)
    assert (C.BOOT_N, C.BOOT_SEED, C.BOOT_PCT) == (10_000, 2_030_000, (2.5, 97.5))
    assert (C.RESTARTS, C.LADDER, C.CMA_STARTS, C.TIE, C.BRACKET_STEP, C.WIDTH) == (200, (200, 800), 20, 1e-9, 0.1, 0.01)
    assert C.WINDOW == 100 and C.RULE_FRAC == 0.5 and C.TIMING_SEED not in C.SEEDS
    assert len(C.runs()) == 240


# ------------------------------------------------------------------------------------------ data, init, training
def test_init_is_band_rd_init_without_touching_global_rng():
    import torch
    from src.band_rd import init_rd
    for d in (2, 4):
        with torch.random.fork_rng():
            ref = init_rd(2_030_007, d)
        before = torch.random.get_rng_state().clone()
        p = C.init_params(2_030_007, d)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert torch.equal(p, ref) and p.dtype == torch.float64 and len(p) == d + 3


def test_data_is_band_rd_data():
    from src.band_rd import make_data_rd
    for d in (2, 4):
        X, y = C.make_data(2_030_001, d)
        X0, y0 = make_data_rd(2_030_001, d)
        assert np.array_equal(X, X0) and np.array_equal(y, y0) and X.shape == (400, d)
    assert np.array_equal(C.make_data(5, 4)[0][:, :2], C.make_data(5, 2)[0])


def test_train_one_equals_band_rd_single_seed_training():
    import torch
    from src.band_rd import train_single
    with torch.random.fork_rng():
        ps, _ = train_single(1.50, 4, 889_003, 300)
    W, V = C.train_one(889_003, 4, 1.50, budget=300)
    assert np.array_equal(W[-1], ps) and W.shape == (301, 7) and np.isnan(V[0]).all() and np.all(V[1:] >= 0)


# ------------------------------------------------------------------------------------------ own-sample loss and status
def test_profiled_loss_gradient_and_full_loss():
    X, y = C.make_data(889_001, 4)
    rng = np.random.default_rng(3)
    P = rng.uniform(-1.5, 1.5, (4, 5)); P[:, 1] = rng.uniform(0, 6.28, 4)
    L, G, b = C.loss_grad_rd(P, 4.0, 1.30, X, y)
    for j in range(5):
        E = np.zeros(5); E[j] = 1e-6
        fd = (C.loss_grad_rd(P + E, 4.0, 1.30, X, y)[0] - C.loss_grad_rd(P - E, 4.0, 1.30, X, y)[0]) / 2e-6
        assert np.abs(fd - G[:, j]).max() < 1e-7
    for i in range(4):
        h = np.r_[P[i, 0], P[i, 1], b[i], P[i, 2:]]
        assert abs(C.loss_full(h, 4.0, 1.30, X, y) - L[i]) < 1e-14
        hb = h.copy(); hb[2] += 1e-3                                  # b₂ is profiled: moving it raises the loss
        assert C.loss_full(hb, 4.0, 1.30, X, y) > L[i]


def test_status_is_the_rd_gap_sign_and_reduces_to_width1():
    from src.band_rd import gap_rd
    from src.profiled_bnb import gap
    for w1, b1 in ((2.4, -2.3), (1.1, 0.4), (-2.0, 2.2)):
        st, G = C.status_of(np.array([w1, b1, 0.3, 0.0]), 1.30)
        assert G == float(gap([w1], [b1], 1.30)[0]) and st == ("placed" if G > 0 else "unplaced")
    h = np.array([2.4, -2.3, 0.0, 0.2, -0.1, 0.05])
    st, G = C.status_of(h, 1.50)
    assert G == float(gap_rd(2.4, -2.3, h[3:], 1.0, 1.50)) and st == ("placed" if G > 0 else "unplaced")


def test_starts_cover_full_and_small_noise_boxes():
    P = C.starts(1000, np.random.default_rng(0), 4, 2.5)
    assert np.abs(P[:, 0]).max() <= 2.5 and P[:, 1].min() >= 0 and P[:, 1].max() < 2 * math.pi
    assert np.abs(P[0::2, 2:]).max() > 2.0 and np.abs(P[1::2, 2:]).max() <= 0.25


# ------------------------------------------------------------------------------------------ bracket and validation rules
def test_bracket_steps_up_then_bisects_to_width():
    calls = []
    f = lambda s: (calls.append(s), s >= 3.2371)[1]
    lo, hi, n, note = C.bracket_switch(f, 3.0)
    assert note == "" and lo < 3.2371 <= hi and hi - lo <= 0.01 + 1e-12 and not f(lo) and f(hi)
    assert calls[:4] == [3.0, 3.1, 3.2, 3.3] and n == len(calls) - 2


def test_bracket_steps_down_when_placed_at_start_and_notes_limits():
    lo, hi, n, note = C.bracket_switch(lambda s: s >= 2.6049, 3.0)
    assert note == "" and lo < 2.6049 <= hi and hi - lo <= 0.01 + 1e-12
    lo, hi, n, note = C.bracket_switch(lambda s: True, 1.0)
    assert np.isnan(lo) and note.startswith("placed down to")
    lo, hi, n, note = C.bracket_switch(lambda s: False, 19.5)
    assert np.isnan(hi) and note.startswith("unplaced up to")


def test_ladder_independent_audit_rules():
    a = {"loss": 0.3, "status": "placed"}
    assert C.ladder_ok(a, {"loss": 0.3 + 5e-10, "status": "placed"})
    assert not C.ladder_ok(a, {"loss": 0.3 - 2e-9, "status": "placed"})          # the larger search found lower
    assert not C.ladder_ok(a, {"loss": 0.3, "status": "unplaced"})
    assert C.independent_ok(0.3, 0.3 - 5e-10) and C.independent_ok(0.3, 0.4) and not C.independent_ok(0.3, 0.3 - 2e-9)
    ret = {"loss": 0.3, "status": "placed"}
    assert C.audit_ok(ret, [ret, {"loss": 0.3 + 5e-8, "status": "unplaced"}]) == (True, 0)
    assert C.audit_ok(ret, [{"loss": 0.3 + 5e-10, "status": "unplaced"}]) == (False, 1)
    assert C.audit_ok(ret, [{"loss": 0.3 + 5e-10, "status": "placed"}]) == (True, 0)


def test_end_validated_needs_all_checks_and_the_expected_status():
    ok = {"ladder_ok": True, "independent_ok": True, "audit_ok": True, "status": "placed"}
    assert C.end_validated(ok, "placed") and not C.end_validated(ok, "unplaced")
    for k in ("ladder_ok", "independent_ok", "audit_ok"):
        assert not C.end_validated({**ok, k: False}, "placed")


def test_search_and_validation_on_a_real_sample():
    """A non-registered seed: the 200-restart search, the 800-restart search and CMA-ES agree at one scale; the
    validation row carries every check; placement from the search is monotone across a wide scale gap."""
    X, y = C.make_data(889_002, 2)
    r1, _, c1, _ = C.eval_status(3.6, 1.50, X, y, 200, np.random.default_rng(1))
    r2, _, _, _ = C.eval_status(3.6, 1.50, X, y, 200, np.random.default_rng(2))
    assert C.ladder_ok(r1, r2) and r1["gnorm_polished"] < 1e-8 and sum(C.eligible(c) for c in c1) >= 150
    lo, _, _, _ = C.eval_status(1.5, 1.50, X, y, 200, np.random.default_rng(3))
    hi, _, _, _ = C.eval_status(8.0, 1.50, X, y, 200, np.random.default_rng(4))
    assert lo["status"] == "unplaced" and hi["status"] == "placed"
    v = C.validate_end(8.0, 1.50, X, y, [np.random.default_rng(j) for j in (5, 6, 7)])
    assert v["ladder_ok"] and v["independent_ok"] and v["audit_ok"] and C.end_validated(v, "placed")
    assert v["cma_loss"] >= v["loss"] - 1e-9


# ------------------------------------------------------------------------------------------ rule point, t_sw, χ inputs
def test_t_switch_and_rule_point():
    s = np.array([0.9, 0.3, 1.2, 0.8, 1.1, 1.6, 2.0, 1.9, 2.5])
    assert C.t_switch(s, 2.0) == 6 and C.t_switch(s, 1.95) == 6 and C.t_switch(s, 3.0) is None
    assert C.t_switch(np.array([2.5, 1.0, 2.1]), 2.0) == 2                    # step 0 does not count
    # rule point: last upward passage of 0.5·2.0 = 1.0 before t_sw = 6 (s dips to 0.8 at step 3)
    assert C.rule_step(s, 2.0) == 4
    assert C.rule_step(s, 3.0) is None
    s2 = np.array([0.1, 1.2, 1.5, 1.8, 2.1])                                 # never below 0.5·s_own after step 1
    assert C.rule_step(s2, 2.0) == 1 and C.t_switch(s2, 2.0) == 4


def test_growth_window_and_winding():
    s = np.exp(0.001 * np.arange(300))
    assert abs(C.growth_at(s, 250) - 0.001) < 1e-12 and abs(C.growth_at(s, 40) - 0.001) < 1e-12
    z = -2.3
    assert C.winding(-2.3 - 2 * math.pi + 0.4, 1.0, z) == -1 and C.winding(-2.1, 1.0, z) == 0
    assert C.winding(2.3 + 2 * math.pi, -3.0, z) == -1                         # canonical b₁ = −b₁ when w₂ < 0


def test_width1_kappa_values_match_3b():
    import json
    import pandas as pd
    from src.lag_law import kappa_k
    kt = pd.read_csv(C.RESULTS / "lag_law" / "kappa.csv")
    got = {}
    for a, k in ((1.30, -1), (1.50, 0), (1.30, 0)):
        r = kt[kt.a.round(2) == a].iloc[0]
        H, tan, dG, p = (np.array(json.loads(getattr(r, c))) for c in ("H", "tangent", "gradG", "p_median"))
        got[(a, k)] = kappa_k(H, tan, dG, p, k)
    assert abs(got[(1.30, -1)] - 7.61) < 0.01 and abs(got[(1.50, 0)] - 4.05) < 0.01 and got[(1.30, 0)] < 0


def test_continue_branch_constructed():
    X, y = C.make_data(889_004, 2)
    p = np.array([2.4, -2.4, 1.5, 0.0, 0.01])
    h_rule, h, H, note = C.continue_branch(1.50, X, y, p, 1.5)                 # target = own s: no step moves it
    assert note == "" and np.abs(h - h_rule).max() < 1e-9 and H.shape == (4, 4)
    h_rule, h, H, note = C.continue_branch(1.50, X, y, p, 2.2)
    assert note == "" and np.linalg.eigvalsh(H).min() > 0 and h.shape == (4,)


def test_prediction_uses_no_gap(monkeypatch):
    """predict_run on a short real path must not evaluate any gap (crossings are kept out of the predictions)."""
    import src.band_rd as B
    W, V = C.train_one(889_005, 2, 1.50, budget=1500)
    s = np.abs(W[:, 2])
    s_own = float(0.97 * s.max())
    kap = {(1.50, k): 4.0 for k in C.K_RANGE}

    def boom(*a, **k):
        raise AssertionError("gap evaluated")
    monkeypatch.setattr(B, "gap_rd", boom)
    monkeypatch.setattr(B, "gap_rd_t", boom)
    r = C.predict_run(889_005, 2, 1.50, W, V, s_own, kap, -2.3)
    assert r["status"] == "ok", r
    assert r["t_rule"] < r["t_sw"] and s[r["t_sw"]] >= s_own > s[r["t_sw"] - 1]
    assert abs(r["r_pred"] - r["kappa"] * r["chi"]) < 1e-15 and abs(r["s_pred"] - s_own * (1 + r["r_pred"])) < 1e-12
    assert r["relax"] > 0 and r["chi"] > 0 and not (C.FORBIDDEN & set(r))
    r2 = C.predict_run(889_005, 2, 1.50, W, V, float(s.max() * 1.1), kap, -2.3)
    assert r2["status"] == "s never reaches s_own,d"


# ------------------------------------------------------------------------------------------ observation
def test_observe_path_constructed():
    W = np.zeros((5, 5))
    W[:, 0] = 0.9; W[:, 1] = -2.39; W[:, 2] = [0.5, 1.0, 1.5, 2.0, 2.5]; W[:, 4] = 0.0     # G = 0.171 > 0
    W[:3, 0] = 0.1                                                            # unplaced until step 3
    o = C.observe_path(W, 1.50)
    assert o["crossed"] and o["step_obs"] == 3 and o["s_obs"] == 2.0 and o["rho_cross"] == 0.0
    W4 = W.copy(); W4[3, 4] = 0.3                                            # a noise weight widens the windows
    o4 = C.observe_path(W4, 1.50)
    assert o4["step_obs"] == 4 and abs(o["G_at_cross"] - 0.171) < 1e-3
    W2 = W.copy(); W2[:, 0] = 0.9                                            # placed at initialisation
    o2 = C.observe_path(W2, 1.50)
    assert o2["placed_at_init"] and not o2["crossed"]
    W3 = W.copy(); W3[:, 0] = 0.1
    assert not C.observe_path(W3, 1.50)["crossed"] and C.first_placed([1.0, -1.0, -2.0]) is None


# ------------------------------------------------------------------------------------------ scoring
def _cell(n=60, n_cross=50, lag_ratio=1.0, noise=0.01, x1_factor=0.88, rho=0.01, tsw_late=0, seed=0, pred_nan=0):
    """Constructed runs: s_own ~ U(2.5, 3.5); r_pred ~ U(0.02, 0.06); s_obs = s_own(1 + lag_ratio·r_pred)(1 + noise·u);
    x₁-only threshold = x1_factor·s_own; t_sw 100 steps before the crossing (tsw_late runs: after)."""
    rng = np.random.default_rng(seed)
    s_own = rng.uniform(2.5, 3.5, n)
    r_pred = rng.uniform(0.02, 0.06, n)
    s_obs = s_own * (1 + lag_ratio * r_pred) * (1 + noise * rng.uniform(-1, 1, n))
    crossed = np.zeros(n, bool); crossed[:n_cross] = True
    step = np.where(crossed, 3000.0, np.nan)
    t_sw = np.full(n, 2900.0); t_sw[:tsw_late] = 3100.0
    r_pred = r_pred.copy(); r_pred[tsw_late:tsw_late + pred_nan] = np.nan
    return dict(crossed=crossed, placed_at_init=np.zeros(n, bool), rho_cross=np.where(crossed, rho, np.nan),
                s_obs=np.where(crossed, s_obs, np.nan), s_own=s_own, r_pred=r_pred, s_x1=x1_factor * s_own,
                t_sw=t_sw, step_obs=step)


def _v(s):
    return s["C1"]["verdict"], s["C2"]["verdict"], s["C3"]["verdict"]


def test_all_pass():
    s = C.score_cell(**_cell())
    assert s["valid"] and _v(s) == ("PASS", "PASS", "PASS") and s["outcome"] == "PASS"
    assert s["n_crossing"] == 50 and s["n_scored"] == 50 and s["C3"]["ci95"][1] < 0


def test_c1_fails_on_large_errors_and_c2_on_wrong_lag():
    s = C.score_cell(**_cell(noise=0.15))                     # errors ~ 7% > 5%, lag ratio still ~1 in median
    assert s["C1"]["verdict"] == "FAIL" and s["C1"]["median_abs_log_err"] > 0.05
    s = C.score_cell(**_cell(lag_ratio=2.0, noise=0.0))       # obs lag twice the prediction: C2 fails, C1 passes
    assert _v(s)[:2] == ("PASS", "FAIL") and abs(s["C2"]["median_ratio"] - 2.0) < 1e-9 and s["outcome"] == "FAIL C2"
    s = C.score_cell(**_cell(lag_ratio=0.5, noise=0.0))
    assert s["C2"]["verdict"] == "FAIL" and s["C2"]["median_ratio"] < 0.75


def test_c2_band_edges_inclusive_and_signed():
    s = C.score_cell(**_cell(lag_ratio=1.25, noise=0.0))
    assert abs(s["C2"]["median_ratio"] - 1.25) < 1e-12
    base = _cell(noise=0.0)
    base["s_obs"] = np.where(base["crossed"], base["s_own"] * 0.99, np.nan)   # crossings below s_own: negative ratio
    s = C.score_cell(**base)
    assert s["C2"]["median_ratio"] < 0 and s["C2"]["verdict"] == "FAIL"


def test_c3_fails_when_x1_threshold_is_as_good():
    c = _cell(noise=0.02)
    c["s_x1"] = np.where(c["crossed"], c["s_obs"], c["s_x1"])           # the x₁-only threshold sits at the observed scale
    s = C.score_cell(**c)
    assert s["C3"]["verdict"] == "FAIL" and s["C3"]["ci95"][1] >= 0
    assert "C3" in s["outcome"]


def test_bootstrap_rule():
    D = np.full(30, -0.1) + np.random.default_rng(0).normal(0, 0.01, 30)
    m, lo, hi = C.bootstrap_mean_ci(D)
    assert hi < 0 and lo < m < hi and C.bootstrap_mean_ci(D) == (m, lo, hi)       # deterministic (fixed seed)
    rng = np.random.default_rng(C.BOOT_SEED)
    idx = rng.integers(0, 30, size=(C.BOOT_N, 30))
    assert np.allclose(np.percentile(D[idx].mean(axis=1), [2.5, 97.5]), [lo, hi], rtol=0, atol=0)
    m, lo, hi = C.bootstrap_mean_ci(np.random.default_rng(1).normal(0, 1, 40))
    assert lo < 0 < hi


def test_unresolved_min_crossings():
    s = C.score_cell(**_cell(n_cross=29))
    assert not s["validity"]["V1_min_crossings"] and _v(s) == ("UNRESOLVED",) * 3 and s["outcome"] == "UNRESOLVED"
    assert C.score_cell(**_cell(n_cross=30))["validity"]["V1_min_crossings"]


def test_unresolved_noise_ratio():
    c = _cell()
    rho = np.where(c["crossed"], 0.01, np.nan); rho[:6] = 0.05                   # 44/50 = 0.88 < 0.90 (0.05 is not < 0.05)
    s = C.score_cell(**{**c, "rho_cross": rho})
    assert not s["validity"]["V2_rho"] and s["outcome"] == "UNRESOLVED" and abs(s["frac_rho_below"] - 0.88) < 1e-12
    rho[:6] = 0.01; rho[:5] = 0.2                                                 # 45/50 = 0.90: holds
    assert C.score_cell(**{**c, "rho_cross": rho})["validity"]["V2_rho"]


def test_unresolved_when_predictions_come_too_late():
    s = C.score_cell(**_cell(tsw_late=6))                                         # 44/50 scored
    assert not s["validity"]["V3_prediction_before_crossing"] and s["outcome"] == "UNRESOLVED" and s["n_scored"] == 44
    s = C.score_cell(**_cell(tsw_late=3, pred_nan=3))                             # 6 unscored (late or no prediction)
    assert s["n_scored"] == 44 and s["outcome"] == "UNRESOLVED"
    s = C.score_cell(**_cell(tsw_late=5))                                         # 45/50 = 0.90: valid
    assert s["valid"] and s["n_scored"] == 45 and s["outcome"] == "PASS"
    c = _cell(); c["t_sw"][:5] = 3000.0                                           # t_sw on the crossing step: not before
    assert C.score_cell(**c)["n_scored"] == 45


def test_placed_at_init_and_non_crossers_excluded():
    c = _cell(n_cross=50)
    pai = np.zeros(60, bool); pai[:3] = True                                      # 3 crossing-flagged runs placed at init
    s = C.score_cell(**{**c, "placed_at_init": pai})
    assert s["n_crossing"] == 47 and s["n_placed_at_init"] == 3 and s["n_scored"] == 47


def test_uncomputable_statistics_are_unresolved(monkeypatch):
    monkeypatch.setattr(C, "MIN_CROSS", 1)
    c = _cell(n=2, n_cross=1)
    s = C.score_cell(**c)
    assert s["valid"] and s["n_scored"] == 1 and s["C3"]["verdict"] == "UNRESOLVED" and s["outcome"] == "UNRESOLVED"
    assert s["C1"]["verdict"] in ("PASS", "FAIL")


def test_outcome_names_every_failing_criterion():
    s = C.score_cell(**_cell(lag_ratio=3.0, noise=0.1, x1_factor=1.1))
    assert s["outcome"].startswith("FAIL") and all(c in s["outcome"] for c, v in zip(("C1", "C2", "C3"), _v(s))
                                                   if v == "FAIL")


# ------------------------------------------------------------------------------------------ mirrors (registered descriptive)
def test_mirror_of_and_restricted_search():
    assert C.mirror_of([0.8, 3.8, -18.0]) == "+" and C.mirror_of([-0.8, -2.5, 17.0]) == "-"
    P = C.starts(200, np.random.default_rng(0), 2, 2.0, sigma=-1.0)
    assert (P[:, 0] <= 0).all()
    f = C._restricted_placed(889_002, 2, 1.50, 1.0)
    g = C._restricted_placed(889_002, 2, 1.50, -1.0)
    assert f(8.0) and g(8.0) and not f(1.5) and not g(1.5)


def test_mirror_switches_reuses_the_global_bracket(monkeypatch):
    import json
    calls = []
    monkeypatch.setattr(C, "bracket_switch", lambda f, s0: (calls.append(s0), (3.0, 3.01, 5, ""))[1])
    monkeypatch.setattr(C, "_restricted_placed", lambda *a, **k: None)
    g = {"validation_lo": json.dumps({"h": [0.9, 3.9, -9.0, 0.01]}), "validation_hi": json.dumps({"h": [0.9, 3.9, -9.4, 0.01]}),
         "validated": True, "s_own_d": 2.7125, "s_start": 2.3656}
    out = C.mirror_switches(1, 2, 1.5, g)
    assert out["s_mirror_+"] == 2.7125 and out["s_mirror_-"] == 3.005 and calls == [2.3656]
    g2 = {**g, "validation_lo": json.dumps({"h": [-0.9, -2.3, 9.0, 0.01]})}          # mirror changes across the bracket
    calls.clear()
    out = C.mirror_switches(1, 2, 1.5, g2)
    assert out["s_mirror_+"] == 3.005 and out["s_mirror_-"] == 3.005 and len(calls) == 2


def test_mirror_matched_prediction(monkeypatch):
    fz = {"s_mirror_+": 2.7, "s_mirror_-": 3.3}
    r = {"mirror_rule": "+", "s_own_d": 2.7, "status": "ok", "t_rule": 5, "t_sw": 9, "chi": 0.01, "kappa": 4.0,
         "r_pred": 0.04, "s_pred": 2.808}
    out = C.mirror_matched(1, 2, 1.5, None, None, r, fz, {}, -2.3)
    assert out["mm_same_as_registered"] and out["mm_s_pred"] == 2.808 and out["mm_s_switch"] == 2.7
    monkeypatch.setattr(C, "predict_run", lambda seed, d, a, W, V, s, kap, z: {"status": "ok", "s_pred": s * 1.05,
                                                                                 "r_pred": 0.05, "t_sw": 11})
    out = C.mirror_matched(1, 2, 1.5, None, None, {**r, "mirror_rule": "-"}, fz, {}, -2.3)
    assert not out["mm_same_as_registered"] and abs(out["mm_s_pred"] - 3.465) < 1e-12 and out["mm_t_sw"] == 11
    assert C.mirror_matched(1, 2, 1.5, None, None, {"status": "x"}, fz, {}, -2.3) == {"mm_status": "no rule-point mirror"}


def test_score_end_to_end(tmp_path, monkeypatch):
    """score() on constructed prediction / observation / frozen tables: verdicts equal score_cell's, the mirror-matched
    descriptive is reported and not scored."""
    import json
    import pandas as pd
    monkeypatch.setattr(C, "OUT", tmp_path)
    seeds = list(range(40))
    monkeypatch.setattr(C, "runs", lambda: [(2, 1.5, s) for s in seeds])
    rng = np.random.default_rng(0)
    s_own = rng.uniform(2.5, 3.5, 40); r_pred = rng.uniform(0.02, 0.06, 40)
    s_obs = s_own * (1 + r_pred) * (1 + 0.005 * rng.uniform(-1, 1, 40))
    pr = pd.DataFrame({"seed": seeds, "d": 2, "a": 1.5, "s_own_d": s_own, "r_pred": r_pred, "s_pred": s_own * (1 + r_pred),
                       "t_sw": 100, "t_rule": 50, "status": "ok", "chi": 0.01, "kappa": 4.0, "k": 0, "s_own_x1": 0.88 * s_own,
                       "mirror_rule": "+", "mm_s_switch": s_own, "mm_r_pred": r_pred, "mm_t_sw": 100})
    pr.to_csv(tmp_path / "predictions.csv", index=False)
    (tmp_path / "predictions.sha256").write_text("x  predictions.csv\n"); (tmp_path / "registration.sha256").write_text("")
    with open(tmp_path / "observed_parts.jsonl", "w") as fh:
        for i in seeds:
            fh.write(json.dumps({"seed": i, "d": 2, "a": 1.5, "placed_at_init": False, "crossed": i < 36,
                                 "step_obs": 200 if i < 36 else None, "s_obs": float(s_obs[i]) if i < 36 else None,
                                 "rho_cross": 0.01 if i < 36 else None}) + "\n")
    pd.DataFrame({"seed": seeds, "d": 2, "a": 1.5, "global_mirror_hi": "+"}).to_csv(tmp_path / "frozen_seeds.csv", index=False)
    res = C.score()
    c = res["cells"]["d2_a1.50"]
    assert c["outcome"] == "PASS" and c["n_crossing"] == 36 and c["n_scored"] == 36
    assert c["descriptive"]["mirror_matched_NOT_SCORED"]["outcome"] == "PASS"
    assert c["descriptive"]["n_crossing_rule_mirror_is_global_mirror"] == 36
    o = pd.read_csv(tmp_path / "observed_runs.csv")
    assert o.scored.sum() == 36 and (tmp_path / "scores.json").exists()
