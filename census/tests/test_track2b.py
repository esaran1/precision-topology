"""Test 2B (registered; the validity boundary of the lag law in κχ): every decision rule on constructed PASS, FAIL and
UNRESOLVED cases, the η_cell stability rule, the realised-trajectory check, and the reuse of the ramp machinery."""

import math

import numpy as np
import pandas as pd
import pytest

from src import ramp as R
from src import track2b as X

OTHER_SEEDS = (862_000, 862_001, 862_002)          # first boundary test seeds (already used); never 2B seeds


# ------------------------------------------------------------------------------------------ registered constants
def test_registered_constants():
    assert X.SETTINGS == ((1.30, -1), (1.50, 0))
    assert X.KX_TARGETS == (0.02, 0.05, 0.1, 0.2, 0.3, 0.4)
    assert X.SEEDS == tuple(range(2_020_000, 2_020_040)) and len(X.SEEDS) == X.N_SEEDS == 40
    assert (X.ETA_CAP, X.WARMUP, X.S0_FRAC, X.END_MULT) == (0.3, 4000, 0.5, 6.0)
    assert (X.BAND, X.CHI_TOL, X.MIN_CROSS, X.C1_MAX_KX, X.C2_MIN_KX, X.STAB_MAX) == ((0.75, 1.25), 0.30, 30, 0.1, 0.3, 1.0)
    assert not set(X.SEEDS) & set(OTHER_SEEDS)


# ------------------------------------------------------------------------------------------ η_cell and γ (a priori)
def test_eta_cell_rule():
    assert X.eta_cell(2.0) == 0.3                        # 1/λ = 0.5 > 0.3: the cap binds
    assert X.eta_cell(1 / 0.3) == pytest.approx(0.3)
    assert X.eta_cell(4.0) == 0.25                       # 1/λ < 0.3
    assert X.eta_cell(3.87) * 3.87 == pytest.approx(1.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(AssertionError):
            X.eta_cell(bad)


def test_gamma_gives_the_target_kappa_chi():
    for kap, eta, lam in ((7.62, 0.3, 0.1507), (4.07, 0.26, 0.1493)):
        for kx in X.KX_TARGETS:
            g = X.gamma_for(kx, kap, eta, lam)
            chi = g / (eta * lam)
            assert chi == pytest.approx(kx / kap, rel=1e-14) and kap * chi == pytest.approx(kx, rel=1e-14)


def test_ramp_length_and_schedule_reach_s_end():
    s_star, g, kx = 4.95625, 1.2e-4, 0.02
    n = X.ramp_steps(g, kx)
    s0 = 0.5 * s_star
    assert X.scale_at(X.WARMUP + n, s0, g) >= s_star * (1 + 6 * kx) > X.scale_at(X.WARMUP + n - 1, s0, g)
    assert X.scale_at(1, s0, g) == X.scale_at(X.WARMUP, s0, g) == s0            # held for 4,000 steps
    assert X.scale_at(X.WARMUP + 1, s0, g) == pytest.approx(s0 * math.exp(g))
    assert X.end_factor(0.4) == pytest.approx(3.4) and X.end_factor(0.02) == pytest.approx(1.12)


# ------------------------------------------------------------------------------------------ realised stability
def test_realised_stability_is_checked_up_to_the_crossing():
    lam = np.full(100, 3.0); lam[80] = 5.0               # a violation at step 81 (η·λ = 1.25 at η = 0.25)
    m, ok = X.realised_stability(0.25, lam, 60, 100)     # crossed at step 60: the violation is after it
    assert ok and m == pytest.approx(0.75)
    m, ok = X.realised_stability(0.25, lam, 81, 100)     # crossed at step 81: the violation is on the trajectory
    assert not ok and m == pytest.approx(1.25)
    m, ok = X.realised_stability(0.25, lam, None, 100)   # not crossed: the whole run
    assert not ok
    m, ok = X.realised_stability(0.25, lam, float("nan"), 100)
    assert not ok


def test_realised_stability_boundary_and_nonfinite():
    lam = np.full(10, 4.0)
    assert X.realised_stability(0.25, lam, None, 10) == (1.0, True)             # η·λ = 1 exactly holds (≤ 1)
    assert X.realised_stability(0.25, np.full(10, 4.0 + 1e-9), None, 10)[1] is False
    lam[3] = np.nan
    assert X.realised_stability(0.25, lam, None, 10)[1] is False               # a non-finite point fails
    assert X.realised_stability(0.25, lam, 3, 10)[1] is True                   # ... unless after the crossing
    with pytest.raises(AssertionError):
        X.realised_stability(0.25, lam, 11, 10)


# ------------------------------------------------------------------------------------------ validity per cell
def test_cell_validity_each_condition():
    assert X.cell_valid(30, 0.0129, 0.01, True)["valid"]                       # 30 crossings, +29%
    assert X.cell_valid(40, 0.0071, 0.01, True)["valid"]                       # −29%
    assert not X.cell_valid(40, 0.0069, 0.01, True)["chi_ok"]                  # −31%
    assert X.cell_valid(40, 0.65, 0.5, True)["chi_ok"] is (abs(0.65 / 0.5 - 1) <= 0.30)   # compared as computed
    v = X.cell_valid(29, 0.01, 0.01, True); assert not v["valid"] and not v["crossings_ok"]
    v = X.cell_valid(40, 0.0131, 0.01, True); assert not v["valid"] and not v["chi_ok"]
    v = X.cell_valid(40, float("nan"), 0.01, True); assert not v["valid"] and not v["chi_ok"]
    v = X.cell_valid(40, 0.01, 0.01, False); assert not v["valid"] and not v["stability_ok"]


# ------------------------------------------------------------------------------------------ C1, C2, κχ*
def _cells(ratio_by_kx, n_cross=40, chi_scale=1.0, stab=True, override=None):
    out = []
    for a, kap in ((1.3, 7.62), (1.5, 4.07)):
        for kx in X.KX_TARGETS:
            c = {"a": a, "kx_target": kx, "chi_target": kx / kap, "ratios": list(np.full(n_cross, ratio_by_kx[kx])),
                 "n_cross": n_cross, "chi_own_median": kx / kap * chi_scale, "stab_ok": stab}
            if override and (a, kx) in override:
                c.update(override[(a, kx)])
            out.append(c)
    return out


EXPECTED = {0.02: 1.0, 0.05: 1.05, 0.1: 1.2, 0.2: 1.4, 0.3: 1.6, 0.4: 2.0}


def test_pass_pass_and_kx_star():
    S = X.score_cells(_cells(EXPECTED))
    assert S["C1"] == "PASS" and S["C2"] == "PASS" and S["kx_star"] == {1.3: 0.2, 1.5: 0.2}
    assert S["C1_cells_invalid"] == [] and S["C2_cells_invalid"] == []


def test_band_edges_are_inclusive():
    S = X.score_cells(_cells({0.02: 0.75, 0.05: 1.25, 0.1: 1.25, 0.2: 1.25, 0.3: 1.2500001, 0.4: 0.7499999}))
    assert S["C1"] == "PASS" and S["C2"] == "PASS" and S["kx_star"][1.3] == 0.3


def test_C1_fails_if_any_small_kx_cell_leaves_the_band():
    for kx in (0.02, 0.05, 0.1):
        r = dict(EXPECTED); r[kx] = 1.3
        assert X.score_cells(_cells(r))["C1"] == "FAIL"
    S = X.score_cells(_cells(EXPECTED, override={(1.5, 0.05): {"ratios": [0.5] * 40}}))   # one a only, below band
    assert S["C1"] == "FAIL" and S["C2"] == "PASS" and S["kx_star"] == {1.3: 0.2, 1.5: 0.05}


def test_C2_fails_if_any_large_kx_cell_stays_in_the_band():
    S = X.score_cells(_cells({0.02: 1.0, 0.05: 1.0, 0.1: 1.0, 0.2: 1.0, 0.3: 1.1, 0.4: 1.2}))
    assert S["C1"] == "PASS" and S["C2"] == "FAIL" and S["kx_star"] == {1.3: None, 1.5: None}
    S = X.score_cells(_cells(EXPECTED, override={(1.3, 0.3): {"ratios": [1.0] * 40}}))
    assert S["C2"] == "FAIL" and S["kx_star"][1.3] == 0.2


def test_unresolved_by_crossings_chi_or_stability():
    S = X.score_cells(_cells(EXPECTED, n_cross=29))
    assert S["C1"] == "UNRESOLVED" and S["C2"] == "UNRESOLVED"
    S = X.score_cells(_cells(EXPECTED, chi_scale=1.31))
    assert S["C1"] == "UNRESOLVED" and S["C2"] == "UNRESOLVED"
    S = X.score_cells(_cells(EXPECTED, stab=False))
    assert S["C1"] == "UNRESOLVED" and S["C2"] == "UNRESOLVED"


def test_unresolved_propagates_per_criterion_only():
    # an invalid κχ = 0.4 cell leaves C1 resolved; an invalid κχ = 0.2 cell touches neither criterion
    S = X.score_cells(_cells(EXPECTED, override={(1.5, 0.4): {"n_cross": 12, "ratios": [2.0] * 12},
                                                 (1.3, 0.2): {"stab_ok": False}}))
    assert S["C1"] == "PASS" and S["C2"] == "UNRESOLVED" and S["C2_cells_invalid"] == [(1.5, 0.4)]
    S = X.score_cells(_cells(EXPECTED, override={(1.3, 0.1): {"stab_ok": False}}))
    assert S["C1"] == "UNRESOLVED" and S["C2"] == "PASS" and S["C1_cells_invalid"] == [(1.3, 0.1)]


def test_unresolved_takes_precedence_over_a_failing_valid_cell():
    S = X.score_cells(_cells(EXPECTED, override={(1.3, 0.02): {"ratios": [1.7] * 40},
                                                 (1.5, 0.1): {"n_cross": 20, "ratios": [1.0] * 20}}))
    assert S["C1"] == "UNRESOLVED" and S["cells"][0]["in_band"] is False and S["cells"][0]["valid"]


def test_kx_star_skips_cells_without_crossings_and_ignores_validity():
    S = X.score_cells(_cells(EXPECTED, override={(1.3, 0.2): {"n_cross": 0, "ratios": []},
                                                 (1.5, 0.05): {"n_cross": 10, "ratios": [1.5] * 10}}))
    assert S["kx_star"] == {1.3: 0.3, 1.5: 0.05}
    c = [c for c in S["cells"] if c["a"] == 1.3 and c["kx_target"] == 0.2][0]
    assert c["in_band"] is None and not np.isfinite(c["median_ratio"]) and not c["valid"]


def test_median_ratio_is_the_cell_statistic():
    r = [0.1, 0.8, 1.0, 1.1, 9.0]
    S = X.score_cells(_cells(EXPECTED, override={(1.3, 0.02): {"ratios": r * 8}}))
    c = S["cells"][0]
    assert c["median_ratio"] == 1.0 and c["in_band"]


# ------------------------------------------------------------------------------------------ a-priori feasibility
def test_feasibility_counts():
    sb = np.array([1.0, 1.05, 1.10, 1.2, np.nan])
    # κχ = 0.02: s_end = 1.12 s*; predicted crossing s_branch·1.02 ≥ 1.12 for s_branch ≥ 1.098; NaN counts as out
    assert X.feasibility_counts(sb, 1.0, 0.02) == 3
    assert X.feasibility_counts(sb, 1.0, 0.02, ratio=1.25) == 3
    assert X.feasibility_counts(sb, 1.0, 0.4) == 1
    assert X.feasible(10) and not X.feasible(11)


# ------------------------------------------------------------------------------------------ crossing detection
def test_first_crossing_rule():
    w = X.WARMUP
    G = -np.ones(w + 50)
    assert X.first_crossing(G) == (False, None, False)
    G[w + 7] = 0.01; G[w + 9] = 0.02
    assert X.first_crossing(G) == (True, w + 7, False)
    G[0] = 1.0                                           # the start state is not a step
    assert X.first_crossing(G) == (True, w + 7, False)
    G[w + 7] = 0.0                                       # G = 0 is not placed
    assert X.first_crossing(G) == (True, w + 9, False)
    G[w] = 0.5                                           # placed in the held phase: not a crossing, the run is done
    assert X.first_crossing(G) == (False, None, True)


# ------------------------------------------------------------------------------------------ dynamics = the ramp machinery
def test_hessian_batch_matches_autograd():
    import torch
    from src.lag_law import hessian_and_tangent
    Xd, Yd = R._data(list(OTHER_SEEDS))
    rng = np.random.default_rng(0)
    for a, k in X.SETTINGS:
        z0 = R.branch_init(a, k)
        P = z0 + 0.05 * rng.standard_normal((3, 3))
        s = 0.8 * R._s_star_pop(a)
        Hb = X.hessian_batch(torch.tensor(P), Xd, Yd, s, a).numpy()
        for i in range(3):
            H, _ = hessian_and_tangent(P[i], s, a, Xd[i].numpy(), Yd[i].numpy())
            assert np.allclose(Hb[i], H, rtol=1e-10, atol=1e-12)


def test_train_cell_equals_batch_ramp_at_the_shared_learning_rate():
    """At η = 0.3 the 2B loop reproduces ramp.batch_ramp exactly: same crossing step and scale, same parameters."""
    a, k, sp = 1.50, 0, R._s_star_pop(1.50)
    gamma, warm, n = 0.003, 200, 700
    ref = R.batch_ramp(a, OTHER_SEEDS, "sgd", gamma, sp, warmup=warm, max_steps=n, winding=k)
    W, lam = X.train_cell(a, k, 0.3, gamma, sp, n, seeds=OTHER_SEEDS, warmup=warm)
    assert W.shape == (n + 1, 3, 3) and lam.shape == (n, 3) and np.isfinite(lam).all()
    stop = ref[0]["steps_run"]
    for i, r in enumerate(ref):
        crossed, t, pw = X.first_crossing(X.gap_path(W[:, i, :], a), warmup=warm)
        assert (crossed, pw) == (r["crossed"], r["placed_in_warmup"])
        if crossed:
            assert t == r["step"] + warm
            assert X.scale_at(t, 0.5 * sp, gamma, warm) == r["s_cross"]
            assert (W[t, i, 0], W[t, i, 1], W[t, i, 2]) == (r["w1"], r["b1"], r["b2"])
        assert (W[stop, i, 0], W[stop, i, 1], W[stop, i, 2]) == (r["w1_end"], r["b1_end"], r["b2_end"])
    assert any(r["crossed"] for r in ref)


def test_predict_r_equals_ramp_predict_run_at_eta_0_3():
    L = R._landscape()
    for a, k in X.SETTINGS:
        l = L[round(a, 2)]
        for kk in (-1, 0, 1):
            b1 = l["z"][1] + 2 * math.pi * kk + 0.01
            ref = R.predict_run(l, "sgd", 1e-3, {"w1": l["z"][0], "b1": b1, "b2": 0.0})
            p = X.predict_r(l, 1e-3, 0.3, b1)
            assert p["winding_k"] == ref["winding_k"] == kk
            assert p["pred_r"] == pytest.approx(ref["pred_r"], rel=1e-14)
    # with the cell's η, χ is γ/(η λ_min): the prediction at the natural winding is exactly the target κχ
    kb = pd.read_csv(R.RESULTS / "lag_law" / "kappa_by_winding.csv")
    for a, k in X.SETTINGS:
        l = L[round(a, 2)]
        lam = float(np.linalg.eigvalsh(l["H"]).min())
        kap = float(kb[(kb.a.round(2) == round(a, 2)) & (kb.k == k)].kappa_sgd.iloc[0])
        for kx, eta in ((0.02, 0.3), (0.4, 0.26)):
            p = X.predict_r(l, X.gamma_for(kx, kap, eta, lam), eta, l["z"][1] + 2 * math.pi * k)
            assert p["pred_r"] == pytest.approx(kx, rel=1e-6)


# ------------------------------------------------------------------------------------------ frozen design (after `branch`)
DESIGN = X.OUT / "design.csv"


@pytest.mark.skipif(not DESIGN.exists(), reason="design not frozen yet")
def test_frozen_design_follows_the_rules():
    d = X._read(DESIGN)
    br = X._read(X.OUT / "population_branch.csv")
    assert len(d) == 12 and sorted(set(d.kx_target)) == list(X.KX_TARGETS)
    for r in d.itertuples():
        b = br[(br.a.round(2) == round(r.a, 2)) & (br.s >= r.s0 * (1 - 1e-12)) & (br.s <= r.s_end * (1 + 1e-12))]
        assert b.s.min() == pytest.approx(r.s0, rel=1e-12) and b.s.max() == pytest.approx(r.s_end, rel=1e-12)
        assert r.lambda_max_max == b.lambda_max.max()
        assert r.eta_cell == min(0.3, 1 / r.lambda_max_max) and r.eta_cell * r.lambda_max_max <= 1 + 1e-15
        assert r.gamma / (r.eta_cell * r.lambda_min_switch) == pytest.approx(r.chi_target, rel=1e-14)
        assert r.kappa * r.chi_target == pytest.approx(r.kx_target, rel=1e-14)
        assert r.s_end == pytest.approx(r.s_star_pop * (1 + 6 * r.kx_target), rel=1e-14)
        assert r.s_last >= r.s_end and r.total_steps == r.warmup + r.ramp_steps
    assert (br.grad_residual < X.RESID_MAX).all() and (br.lambda_min > 0).all()
