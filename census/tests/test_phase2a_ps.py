"""Phase 2A-PS (src/phase2a_ps.py): the registered settings; the per-seed pieces (sample = the exploration generator,
loss/step and ρ₂ equal to Phase 2A's on the fixed data, the idle unit exactly 0); the class rule, eligibility, the
status order, counts, the gate and the training caps; constructed PASS / FAIL / UNRESOLVED cases for EVERY criterion
(F incl. the falsifier read as stated, H, E_seed, C3, C4, P, C1, C2), every validity rule and the outcome rule; the
ENFORCEMENT test (a forecaster that reads s at or after its cutoff makes this file fail; a guard bypass is caught by the
NaN recomputation; run_one aborts on a differing recomputation) and the NaN recomputation on a real path; the
reproduction of an exploratory seed's freeze (p2a_explore_h, seed 2,930,006) and of its 2⁻¹³ run."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import causal_forecast as CF
from src import causal_forecast_fold as CFF
from src import phase2a as P2A
from src import phase2a_ps as PS
from src import simplicity_bias_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
EXPLORE_H = ROOT / "results/designs/phase2a_explore/p2a_explore_h.jsonl"
FROZEN = ROOT / "results/phase2a_ps/frozen.json"
PILOT = ROOT / "results/phase2a_ps/pilot.json"


def _explore_h(seed, part="freeze", l2=None):
    for ln in EXPLORE_H.read_text().splitlines():
        r = json.loads(ln)
        if r.get("seed") != seed or r.get("mode") != "grid":
            continue
        if part == "freeze" and "traceable" in r:
            return r
        if part == "run" and r.get("part") == "run" and r.get("log2rho") == l2:
            return r
    raise KeyError(seed)


# ================================================================================================ registered settings
def test_registered_settings_match_the_approved_page():
    assert (PS.SEEDS[0], PS.SEEDS[-1], len(PS.SEEDS)) == (2_975_000, 2_975_399, 400)
    assert (PS.PILOT_SEEDS[0], PS.PILOT_SEEDS[-1], len(PS.PILOT_SEEDS)) == (2_976_000, 2_976_059, 60)
    mine = set(PS.SEEDS) | set(PS.PILOT_SEEDS)
    assert not set(PS.SEEDS) & set(PS.PILOT_SEEDS)
    assert not mine & set(range(2_930_000, 2_930_200))                         # exploration seeds
    assert not mine & (set(P2A.SEEDS) | set(P2A.PILOT_SEEDS))
    assert PS.SAMPLE_STREAM == 7 and PS.N_POINTS == 800
    assert round(PS.FRAC0, 4) == 0.1776 and PS.FRAC0 == (1.7957 - 1.154) / (PS.S_F_FIXED - 1.154)   # rule 2
    assert PS.ELIG_MAX == 0.6
    assert PS.S_HOMOTOPY == 4.5 and PS.HOMOTOPY_STEPS == (50, 200)                                    # rule 3
    assert (PS.H_CONT, PS.H_CHECK) == (0.05, 0.025) and PS.MC_AGREE == 0.05 and PS.MC_WINDOWS == (0.05, 0.025)
    assert PS.PERTURB_N == 30 and PS.CLASS_SCALE == 1.01 and PS.CLEAN_MIN == 27                       # rule 5
    assert (PS.LOG2_SCORED, PS.LOG2_EXPONENT, PS.LOG2_OTHER) == (-16.0, -14.0, -14.0)                  # rule 6
    assert PS.F_CUT == 0.95 and PS.BUDGET_FACTOR == 64.0                                              # rule 7
    assert (PS.S_F_FIXED, PS.LAMF_FIXED) == (4.767689442106793, 0.001509953872657247)
    assert PS.CAPS == {"clean": 40, "none": 30, "mixed": 20} and PS.GATE_MIN_CLEAN == 24
    assert (PS.F_MIN_FRAC, PS.F_HI, PS.F_FALSIFIER, PS.H_MIN_FRAC, PS.E_BAND, PS.C3_MIN_FRAC) == (
        0.9, 1.25, 1.25, 0.8, (0.55, 0.80), 0.9)
    assert (PS.TAU1, PS.TAU2, PS.WITHIN_MIN_FRAC, PS.VALID_MIN_FRAC) == (0.15, 0.25, 0.8, 0.9)
    assert (PS.BOOT_N, PS.BOOT_SEED, PS.BOOT_PCT) == (10_000, 2_975_000, (2.5, 97.5))
    assert PS.PRIMARY == ("F", "H", "E_seed", "C3", "C4", "P") and PS.SECONDARY == ("C1", "C2")
    assert (PS.OBS_END, PS.FOLLOW_FRAC, PS.ON_TOL) == (1.25, 0.8, 1e-3)
    cfg = CFF.config(PS.F_CUT)
    assert (cfg.f, cfg.family, cfg.window_frac, cfg.min_window, cfg.horizon_factor) == (0.95, "quad", 0.05, 50, 1.6)


def test_fixed_dataset_inputs_equal_the_committed_frozen_files():
    info = PS.check_inputs()
    assert len(info["v2_frozen_sha256"]) == 64 and len(info["phase2a_frozen_sha256"]) == 64
    assert PS.Q == json.loads((ROOT / "results/simplicity_bias_v2/frozen.json").read_text())["q"]


def test_scan_patterns_cover_exactly_the_two_ranges():
    import re
    reg = re.compile(PS.SCAN_PATTERNS["registered"])
    pil = re.compile(PS.SCAN_PATTERNS["pilot"])
    for s in ("2975000", "x 2975399,", "2,975,000", "2,975,399.", "2_975_123"):
        assert reg.search(s), s
    for s in ("2975400", "29750001", "1.2975000", "2,975,400", "2_975_400", "2,974,999", "12975000"):
        assert not reg.search(s), s
    for s in ("2976000", "2,976,059", "2_976_031"):
        assert pil.search(s), s
    for s in ("2976060", "2,976,060", "2_976_060", "29760599"):
        assert not pil.search(s), s


# ================================================================================================ per-seed model pieces
def test_sample_is_the_exploration_generator_and_touches_no_global_rng():
    from results.designs.phase2a_explore import p2a_explore_g as G
    st = np.random.get_state()[1].copy()
    X, Y = PS.sample(2_930_006)
    assert np.array_equal(np.random.get_state()[1], st)
    X2, Y2 = G.sample(2_930_006)
    assert np.array_equal(X, X2) and np.array_equal(Y, Y2)
    assert X.shape == (800, 2) and (Y[:400] == 0).all() and (Y[400:] == 1).all()
    assert len(np.unique(X[:, 0])) == 40 and len(np.unique(X[:, 1])) == 40
    assert PS.sample_sha(X, Y) == PS.sample_sha(*PS.sample(2_930_006))
    assert PS.sample_sha(X, Y) != PS.sample_sha(*PS.sample(2_930_007))


def test_loss_grad_step_and_rho2_equal_phase2a_on_the_fixed_data():
    th = P2A.branch_theta("M", P2A.S0)
    L1, g1 = PS.loss_grad(th, P2A.X, P2A.Y)
    L2, g2 = P2A.loss_grad(th)
    assert L1 == L2 and np.array_equal(g1, g2)
    a, b = PS.make_step(2.0 ** -9, [0, 1, 2], P2A.X, P2A.Y), P2A.make_step(2.0 ** -9, [0, 1, 2])
    x, y = th.copy(), th.copy()
    for _ in range(50):
        x, y = a(x), b(y)
    assert np.array_equal(x, y)
    assert PS.Rho2Grid(P2A.X)(x) == P2A.rho2(x)


def test_rho2_grid_equals_v2_rho2_batch_on_a_seed_sample():
    import torch
    X, Y = PS.sample(2_930_011)
    rng = np.random.default_rng(5)
    rows = rng.normal(0, 1, (6, 17))
    rows[:, 15] = 0.0
    ref = v2.rho2_batch(rows, X, torch)
    r = PS.Rho2Grid(X)
    assert max(abs(r(th) - ref[i]) for i, th in enumerate(rows)) < 1e-13


def test_idle_unit_stays_exactly_zero_on_a_seed_sample():
    X, Y = PS.sample(2_930_011)
    rng = np.random.default_rng(1)
    th = rng.normal(0, 0.5, 17)
    th[[6, 7, 11, 15]] = 0.0                                          # unit 3 idle
    th[12:15] = np.abs(th[12:15])
    step = PS.make_step(2.0 ** -10, [0, 1, 2], X, Y)
    s0 = PS.scale(th)
    for _ in range(200):
        th = step(th)
    assert (th[[6, 7, 11, 15]] == 0.0).all() and PS.scale(th) != s0


# ================================================================================================ class, eligibility, status
@pytest.mark.parametrize("n,cls", [(30, "clean"), (27, "clean"), (26, "mixed"), (1, "mixed"), (0, "none")])
def test_class_rule(n, cls):
    assert PS.classify(n) == cls


def test_release_scale_and_eligibility_boundary():
    lo, s_F = 1.154, PS.S_F_FIXED
    assert PS.release_scale(lo, s_F) == pytest.approx(1.7957, abs=1e-12)                 # maps 2A's s₀
    assert PS.is_eligible(0.6 * 5.0, 5.0) and not PS.is_eligible(np.nextafter(3.0, 4.0), 5.0)
    # eligibility: s₀ ≤ 0.6·s_F ⟺ lo ≤ (0.6 − FRAC0)/(1 − FRAC0)·s_F
    k = (0.6 - PS.FRAC0) / (1 - PS.FRAC0)
    assert PS.is_eligible(PS.release_scale(0.99 * k * 5.0, 5.0), 5.0)
    assert not PS.is_eligible(PS.release_scale(1.01 * k * 5.0, 5.0), 5.0)


def _raw(**kw):
    """A constructed raw record that passes every rule (status 'scoreable', class 'clean')."""
    s_F = 5.0
    r = {"seed": 1, "M_homotopy": {"accepted": True}, "fold_up": s_F, "fold_up_eig": 1e-12, "M_start_stable": True,
         "stable_to_fold": True, "rho2_M_max": 0.3, "s_F": s_F, "fold_up_h2": s_F * (1 + 1e-9),
         "validation": {"0.99": {"n_on_M": 30, "n_rho2_ge_q": 0}, "0.999": {"n_on_M": 20, "n_rho2_ge_q": 0},
                        "1.001": {"n_on_M": 0, "n_rho2_ge_q": 30}, "1.01": {"n_on_M": 0, "n_rho2_ge_q": 30}},
         "fold_constant": {"abs_mc": 1e-7, "coef_linear": 4e-7, "rel_diff": 0.03, "lam_min_first": 1e-4,
                           "lam_min_last": 2e-5, "Lambda_F": math.sqrt(1e-7 * s_F)},
         "s_star": 4.0, "s0": 2.0, "release_theta": [0.0] * 17, "u_follow": [0.0] * 11}
    r.update(kw)
    return r


def _setv(r, scale, **kw):
    r["validation"] = {k: dict(v) for k, v in r["validation"].items()}
    r["validation"][scale].update(kw)
    return r


@pytest.mark.parametrize("mod,status", [
    (lambda r: r.update(M_homotopy={"accepted": False}), "untraceable: M homotopy failed"),
    (lambda r: r.update(fold_up=None), "untraceable: no upper fold"),
    (lambda r: r.update(M_start_stable=False), "untraceable: homotopy point not stable"),
    (lambda r: r.update(stable_to_fold=False), "untraceable: stable part does not reach the upper fold"),
    (lambda r: r.update(rho2_M_max=PS.Q), "untraceable: rho2 >= q on the stable part"),
    (lambda r: r.update(rho2_M_max=None), "untraceable: rho2 >= q on the stable part"),
    (lambda r: r.update(fold_up_h2=5.0 * (1 + 2e-6)), "s_F not validated"),
    (lambda r: r.update(fold_up_h2=None), "s_F not validated"),
    (lambda r: r.update(fold_up_eig=2e-6), "s_F not validated"),
    (lambda r: _setv(r, "1.001", n_on_M=1), "s_F not validated"),
    (lambda r: _setv(r, "1.01", n_on_M=1), "s_F not validated"),
    (lambda r: r["fold_constant"].update(coef_linear=-1e-7), "s_F not validated"),
    (lambda r: r["fold_constant"].update(lam_min_last=2e-4), "s_F not validated"),
    (lambda r: r["fold_constant"].update(rel_diff=0.0500001), "Lambda_F not validated"),
    (lambda r: r.update(s_star=None), "no s*"),
    (lambda r: r.update(release_theta=None), "release Newton failed"),
    (lambda r: r.update(u_follow=None), "release Newton failed"),
    (lambda r: r.update(s0=3.0000001), "ineligible"),
    (lambda r: None, "scoreable"),
])
def test_evaluate_status_order(mod, status):
    r = _raw(fold_constant=dict(_raw()["fold_constant"]))
    mod(r)
    e = PS.evaluate(r)
    assert e["status"] == status
    if status == "scoreable":
        assert e["class"] == "clean" and e["eligible"] and e["Lambda_F"] == r["fold_constant"]["Lambda_F"]


def test_evaluate_boundaries_inclusive():
    assert PS.evaluate(_raw(s0=3.0))["status"] == "scoreable"                        # s₀ = 0.6·s_F
    fc = dict(_raw()["fold_constant"], rel_diff=0.05)
    assert PS.evaluate(_raw(fold_constant=fc))["status"] == "scoreable"              # agreement 5% inclusive
    assert PS.evaluate(_raw(fold_up_eig=-1e-6))["status"] == "scoreable"
    assert PS.evaluate(_raw(fold_up_h2=5.0 * (1 + 1e-6)))["status"] == "scoreable"
    for n, cls in ((27, "clean"), (26, "mixed"), (0, "none")):
        assert PS.evaluate(_setv(_raw(), "1.01", n_rho2_ge_q=n))["class"] == cls
    assert PS.evaluate(_setv(_raw(), "1.001", n_rho2_ge_q=0))["class"] == "clean"   # the class reads 1.01·s_F only


def _row(seed, status="scoreable", cls="clean"):
    return {"seed": seed, "evaluation": {"status": status, "class": cls}}


def test_counts_gate_and_class_listing():
    rows = ([_row(i, "untraceable: M homotopy failed", None) for i in range(5)]
            + [_row(10 + i, "s_F not validated", "clean") for i in range(2)]
            + [_row(20 + i, "Lambda_F not validated", "mixed") for i in range(3)]
            + [_row(30 + i, "no s*", "none") for i in range(4)] + [_row(40 + i, "ineligible", "clean") for i in range(6)]
            + [_row(100 + i, cls=c) for i, c in enumerate(["clean"] * 24 + ["none"] * 7 + ["mixed"] * 3)])
    c = PS.counts(rows)
    assert c["n_seeds"] == len(rows) and c["n_untraceable"] == 5 and c["n_constants_not_validated"] == 5
    assert c["n_no_s_star"] == 4 and c["n_ineligible"] == 6 and c["n_scoreable"] == 34
    assert c["scoreable_by_class"] == {"clean": 24, "none": 7, "mixed": 3}
    assert sum(c["by_status"].values()) == len(rows)
    assert PS.gate(rows) == {"n_clean_eligible": 24, "min": 24, "pass": True}
    rows[-34]["evaluation"]["class"] = "mixed"                                       # 23 clean eligible
    assert PS.gate(rows)["pass"] is False
    # an ineligible or unvalidated clean seed never counts toward the gate
    assert PS.gate([_row(1, "ineligible"), _row(2, "Lambda_F not validated")])["n_clean_eligible"] == 0
    a = PS.class_listing(rows)
    assert a == PS.class_listing(list(reversed(rows))) and a.splitlines()[0] == "0 untraceable: M homotopy failed None"


def test_plan_caps_in_seed_order():
    cls = (["clean"] * 45 + ["none"] * 33 + ["mixed"] * 25)
    rng = np.random.default_rng(3)
    rng.shuffle(cls)
    rows = [_row(1000 + i, cls=c) for i, c in enumerate(cls)] + [_row(5000 + i, "ineligible", "clean") for i in range(9)]
    rows = rows[::-1]                                                              # input order is irrelevant
    pl, surplus = PS.plan(rows)
    seeds = {k: [s for s, kk, l2 in pl if kk == k and (k != "clean" or l2 == -16.0)] for k in PS.CAPS}
    for k, cap in PS.CAPS.items():
        allk = sorted(1000 + i for i, c in enumerate(cls) if c == k)
        assert seeds[k] == allk[:cap] and surplus[k] == allk[cap:]
    assert [(s, l2) for s, k, l2 in pl if k == "clean"] == [x for s in seeds["clean"] for x in ((s, -14.0), (s, -16.0))]
    assert all(l2 == -14.0 for s, k, l2 in pl if k != "clean") and len(pl) == 2 * 40 + 30 + 20
    assert not any(s >= 5000 for s, _, _ in pl)
    # fewer than the caps: everything scoreable is trained
    pl2, sur2 = PS.plan([_row(1, cls="clean"), _row(2, cls="none")])
    assert pl2 == [(1, "clean", -14.0), (1, "clean", -16.0), (2, "none", -14.0)] and not any(sur2.values())


# ================================================================================================ criteria
def _cl16(n=40, s_F=5.0, s_star=4.0, s_obs=None):
    return [s_F * 1.05 if s_obs is None else s_obs] * n, [s_F] * n, [s_star] * n


@pytest.mark.parametrize("k_bad,verdict", [(4, "PASS"), (5, "FAIL")])           # 36/40 = 0.9; 35/40 = 0.875
def test_F_90pct(k_bad, verdict):
    s_obs, s_F, s_star = _cl16()
    for i in range(k_bad):
        s_obs[i] = None if i % 2 else 5.0 * 1.3                                    # no crossing / above 1.25·s_F
    c = PS.criterion_F(s_obs, s_F, s_star)
    assert c["verdict"] == verdict and c["n"] == 40 and not c["falsified"]


def test_F_window_inclusive_and_per_seed_s_F():
    c = PS.criterion_F([5.0, 6.25, 4.0, 5.0], [5.0, 5.0, 4.0, 4.0], [4.5, 4.5, 3.0, 3.0])
    assert c["n_within"] == 4 and c["verdict"] == "PASS"


def test_F_falsifier_read_as_stated():
    # a crossing below s_F and at or below 1.25·s*: FAIL even with 39/40 within
    s_obs, s_F, s_star = _cl16()
    s_obs[0] = 4.9                                                                  # < s_F = 5, ≤ 1.25·4 = 5
    c = PS.criterion_F(s_obs, s_F, s_star)
    assert c["falsified"] and c["n_falsifier"] == 1 and c["verdict"] == "FAIL"
    s_obs[0] = 5.0 * 1.25 / 1.25 * 0.99                                            # 4.95 < s_F, 1.25·s* = 5.0: fires
    assert PS.criterion_F(s_obs, s_F, s_star)["falsified"]
    # below s_F but above 1.25·s* (s_F/s* > 1.25): not the falsifier (an early exit), counted outside the window
    s_obs, s_F, s_star = _cl16(s_star=3.0)
    s_obs[0] = 4.0                                                                  # 1.25·3 = 3.75 < 4 < 5
    c = PS.criterion_F(s_obs, s_F, s_star)
    assert not c["falsified"] and c["n_below_sF"] == 1 and c["verdict"] == "PASS"
    # at or above s_F with s_F/s* < 1.25: s_obs ≤ 1.25·s* but NOT below s_F: the falsifier does not fire
    s_obs, s_F, s_star = _cl16(s_star=4.5, s_obs=5.0)                              # 1.25·4.5 = 5.625 ≥ 5.0 = s_F
    c = PS.criterion_F(s_obs, s_F, s_star)
    assert not c["falsified"] and c["verdict"] == "PASS"


def test_F_unresolved_without_seeds():
    assert PS.criterion_F([], [], [])["verdict"] == "UNRESOLVED"


@pytest.mark.parametrize("n_bad,verdict", [(6, "PASS"), (7, "FAIL")])            # 24/30 = 0.8; 23/30
def test_H_80pct(n_bad, verdict):
    crossed = [i < n_bad for i in range(30)]
    c = PS.criterion_H(crossed, [True] * 30)
    assert c["verdict"] == verdict and c["n"] == 30


def test_H_budget_end_without_crossing_does_not_count_and_min_n():
    crossed, reached = [False] * 30, [True] * 30
    for i in range(7):
        reached[i] = False                                                          # ended at the budget below 1.25·s_F
    c = PS.criterion_H(crossed, reached)
    assert c["n_budget_end_no_crossing"] == 7 and c["verdict"] == "FAIL"
    assert PS.criterion_H([False] * 9, [True] * 9)["verdict"] == "UNRESOLVED"
    assert PS.criterion_H([False] * 10, [True] * 10)["verdict"] == "PASS"
    assert PS.criterion_H([], [])["verdict"] == "UNRESOLVED"


def test_seed_exponent():
    r14, e14 = 0.04, 1.6e-3
    e16 = e14 / 4
    r16 = r14 * 4 ** (-2 / 3)
    assert PS.seed_exponent(r14, r16, e14, e16) == pytest.approx(2 / 3, rel=1e-12)
    for bad in ((None, r16, e14, e16), (r14, -0.01, e14, e16), (r14, r16, 0.0, e16), (r14, r16, e14, e14),
                (r14, float("nan"), e14, e16)):
        assert PS.seed_exponent(*bad) is None


def test_E_seed_band_pass_fail_unresolved():
    rng = np.random.default_rng(0)
    assert PS.criterion_E_seed(list(0.67 + 0.02 * rng.standard_normal(40)))["verdict"] == "PASS"
    assert PS.criterion_E_seed(list(1.0 + 0.02 * rng.standard_normal(40)))["verdict"] == "FAIL"      # linear lag
    assert PS.criterion_E_seed(list(0.0 + 0.02 * rng.standard_normal(40)))["verdict"] == "FAIL"      # no delay
    straddle = list(0.80 + 0.05 * rng.standard_normal(40))                                           # CI over 0.80
    assert PS.criterion_E_seed(straddle)["verdict"] == "FAIL"
    e = PS.criterion_E_seed([0.6, 0.7, None, None])
    assert e["verdict"] == "UNRESOLVED" and e["n"] == 2 and e["n_excluded"] == 2
    e = PS.criterion_E_seed([0.66, 0.67, 0.68, None])
    assert e["n"] == 3 and e["n_excluded"] == 1 and e["verdict"] == "PASS" and e["median"] == 0.67


def test_E_seed_bootstrap_is_the_median_with_the_registered_generator():
    x = np.array([0.6, 0.65, 0.7, 0.75, 0.62, 0.71])
    e = PS.criterion_E_seed(list(x))
    rng = np.random.default_rng(2_975_000)
    idx = rng.integers(0, len(x), size=(10_000, len(x)))
    lo, hi = np.percentile(np.median(x[idx], axis=1), (2.5, 97.5))
    assert e["ci95"] == [lo, hi] and e["median"] == float(np.median(x))


@pytest.mark.parametrize("k_bad,verdict", [(4, "PASS"), (5, "FAIL")])
def test_C3_90pct_and_no_forecast_is_not_gt_1(k_bad, verdict):
    r_obs, r_fc, has = [0.05] * 40, [0.04] * 40, [True] * 40
    for i in range(k_bad):
        if i % 3 == 0:
            r_obs[i] = 0.03                                                         # ratio < 1
        elif i % 3 == 1:
            has[i] = False                                                          # no forecast
        else:
            r_obs[i] = None                                                         # no crossing
    c = PS.criterion_C3(r_obs, r_fc, has)
    assert c["verdict"] == verdict and c["n"] == 40
    assert PS.criterion_C3([0.04], [0.04], [True])["verdict"] == "FAIL"             # ratio = 1 is not > 1
    assert PS.criterion_C3([], [], [])["verdict"] == "UNRESOLVED"


def test_C4_pass_fail_no_forecast_unresolved():
    n = 40
    t_obs = np.full(n, 1000.0)
    t_F_fc = t_obs - 100                                                            # no-delay forecast 100 early
    good = PS.criterion_C4(list(t_obs - 10), list(t_F_fc), list(t_obs), [True] * n)
    assert good["verdict"] == "PASS" and good["ci95"][1] < 0 and good["n"] == n
    bad = PS.criterion_C4(list(t_obs + 300), list(t_F_fc), list(t_obs), [True] * n)
    assert bad["verdict"] == "FAIL"
    mix = list(t_obs - 10)
    mix[:20] = list(t_obs[:20] + 190)                                               # mean D straddles 0
    assert PS.criterion_C4(mix, list(t_F_fc), list(t_obs), [True] * n)["verdict"] == "FAIL"
    has = [True] * n
    has[0] = False
    t_fc = list(t_obs - 10)
    t_fc[0] = None
    nf = PS.criterion_C4(t_fc, list(t_F_fc), list(t_obs), has)
    assert nf["verdict"] == "FAIL" and nf["n_no_forecast"] == 1
    t_obs2 = list(t_obs)
    t_obs2[1] = None                                                                # no crossing: out of D, counted
    c = PS.criterion_C4(list(t_obs - 10), list(t_F_fc), t_obs2, [True] * n)
    assert c["n"] == n - 1 and c["n_no_crossing"] == 1 and c["verdict"] == "PASS"
    assert PS.criterion_C4([990.0], [900.0], [1000.0], [True])["verdict"] == "UNRESOLVED"


def test_P_pass_fail_missing_unresolved():
    n = 40
    s_obs = [5.2] * n
    own = [5.19] * n                                                                # closer to s_obs
    fixed = [5.0] * n
    assert PS.criterion_P(own, fixed, s_obs)["verdict"] == "PASS"
    assert PS.criterion_P(fixed, own, s_obs)["verdict"] == "FAIL"
    half = [5.19] * 20 + [4.9] * 20                                                 # straddles 0
    assert PS.criterion_P(half, fixed, s_obs)["verdict"] == "FAIL"
    o2 = list(own)
    o2[3] = None                                                                    # own ŝ_c missing at a crossing seed
    c = PS.criterion_P(o2, fixed, s_obs)
    assert c["verdict"] == "FAIL" and c["n_own_missing"] == 1
    f2 = list(fixed)
    f2[3] = None                                                                    # competitor missing: excluded
    c = PS.criterion_P(own, f2, s_obs)
    assert c["verdict"] == "PASS" and c["n_fixed_missing_excluded"] == 1 and c["n"] == n - 1
    s2 = list(s_obs)
    s2[5] = None                                                                    # no crossing: excluded, counted
    c = PS.criterion_P(o2[:3] + [5.19] + o2[4:], fixed, s2)
    assert c["n_no_crossing"] == 1 and c["n"] == n - 1
    assert PS.criterion_P([5.19], [5.0], [5.2])["verdict"] == "UNRESOLVED"
    rng = np.random.default_rng(2_975_000)
    D = np.abs(np.log(np.array(own) / 5.2)) - np.abs(np.log(np.array(fixed) / 5.2))
    idx = rng.integers(0, n, size=(10_000, n))
    lo, hi = np.percentile(D[idx].mean(axis=1), (2.5, 97.5))
    assert PS.criterion_P(own, fixed, s_obs)["ci95"] == [lo, hi]


@pytest.mark.parametrize("k_bad,verdict", [(8, "PASS"), (9, "FAIL")])           # 32/40 = 0.8; 31/40
def test_C1_C2_within_80pct(k_bad, verdict):
    delay = [1000.0] * 40
    err = [100.0] * 40                                                              # 0.10 ≤ τ₁ = 0.15
    has = [True] * 40
    for i in range(k_bad):
        if i % 2:
            err[i] = 200.0
        else:
            has[i] = False
    assert PS.criterion_within(err, delay, PS.TAU1, has)["verdict"] == verdict
    assert PS.criterion_within([150.0], [1000.0], PS.TAU1, [True])["verdict"] == "PASS"   # inclusive
    assert PS.criterion_within([250.0], [1000.0], PS.TAU2, [True])["verdict"] == "PASS"


@pytest.mark.parametrize("k_bad,ok", [(4, True), (5, False)])                   # 36/40 = 0.9; 35/40
def test_validity_follow_and_cutoff_before_90pct(k_bad, ok):
    n = 40
    follow = [True] * n
    for i in range(k_bad):
        follow[i] = False
    v = PS.validity(follow, [100] * n, [200] * n, [True] * 80, [True] * 80)
    assert v["follow_ok"] is ok and v["cutoff_before_ok"] and v["ok"] is ok
    t_c, t_obs = [100] * n, [200] * n
    for i in range(k_bad):
        if i == 0:
            t_c[i] = None                                                           # no cutoff: not before
        elif i == 1:
            t_obs[i] = 100                                                          # t_c = t_obs: not before
        else:
            t_obs[i] = 50
    v = PS.validity([True] * n, t_c, t_obs, [True] * 80, [True] * 80)
    assert v["cutoff_before_ok"] is ok and v["follow_ok"] and v["n_cutoff_not_before"] == k_bad


def test_validity_cutoff_and_no_crossing_passes_and_idle_signs_every_clean_run():
    v = PS.validity([True] * 10, [100] * 10, [None] * 10, [True] * 20, [True] * 20)
    assert v["cutoff_before_ok"] and v["ok"]
    idle = [True] * 20
    idle[17] = False                                                                # one 2⁻¹⁴ clean run
    assert not PS.validity([True] * 10, [100] * 10, [200] * 10, idle, [True] * 20)["ok"]
    sg = [True] * 20
    sg[0] = False
    assert not PS.validity([True] * 10, [100] * 10, [200] * 10, [True] * 20, sg)["ok"]
    assert not PS.validity([], [], [], [], [])["ok"]


def test_outcome_rule_ignores_C1_C2():
    v = {k: "PASS" for k in PS.PRIMARY}
    assert PS.outcome({**v, "C1": "FAIL", "C2": "FAIL"}) == "PASS"
    assert PS.outcome({**v, "H": "FAIL", "P": "FAIL"}) == "FAIL H+P"
    assert PS.outcome({**v, "E_seed": "UNRESOLVED", "F": "FAIL"}) == "UNRESOLVED"
    assert PS.outcome(v, gate_ok=False) == "UNRESOLVED (gate)"
    assert PS.outcome(v, valid_ok=False) == "UNRESOLVED (validity)"


def _table(n_clean=40, n_none=30, n_mixed=5):
    """Scored rows (run_row's keys) of a constructed PASS: clean seeds at 2⁻¹⁶ and 2⁻¹⁴, none and mixed at 2⁻¹⁴."""
    rows = []
    for i in range(n_clean):
        for l2, eps in ((-14.0, 1.6e-3), (-16.0, 4e-4)):
            r_fc = OMEGA0 * eps ** (2 / 3)
            r_obs = 1.2 * r_fc
            s_F = 5.0
            rows.append({"seed": i, "class": "clean", "log2rho": l2, "s_F": s_F, "s_star": 4.0, "Lambda_F": 1e-3,
                         "t_c": 1000, "has_fc": True, "status": "ok", "t_fc": 2000 - 10, "t_F_fc": 1800,
                         "delay_fc": 190, "eps_fc": eps, "r_fc": r_fc, "s_c_fc": s_F * (1 + 0.95 * r_obs),
                         "s_c_fixed": s_F * (1 + 0.5 * r_obs), "has_fc_fixed": True, "t_fc_fixed": 1900,
                         "t_obs": 2000, "s_obs": s_F * (1 + r_obs), "t_F": 1800, "r_obs": r_obs,
                         "reached_obs_end": False, "follow_in_M": True, "idle_ok": True, "sign_ok": True})
    for k, n in (("none", n_none), ("mixed", n_mixed)):
        for i in range(n):
            rows.append({"seed": 1000 * (k == "none") + 2000 * (k == "mixed") + i, "class": k, "log2rho": -14.0,
                         "s_F": 5.0, "s_star": 4.0, "t_c": 1000, "has_fc": True, "t_fc": 1990, "t_F_fc": 1800,
                         "delay_fc": 190, "eps_fc": 1.6e-3, "r_fc": 0.03, "s_c_fc": 5.1, "s_c_fixed": 5.05,
                         "t_obs": None, "s_obs": None, "t_F": 1800, "r_obs": None, "reached_obs_end": True,
                         "follow_in_M": True, "idle_ok": k == "none", "sign_ok": False})
    return rows


OMEGA0 = CFF.OMEGA0


def test_score_tables_constructed_pass_and_each_failure():
    T = _table()
    S = PS.score_tables(T)
    assert S["outcome"] == "PASS", S["verdicts"]
    assert S["criteria"]["E_seed"]["median"] == pytest.approx(2 / 3, rel=1e-9)
    assert S["n_clean_scored"] == 40 and S["n_clean_exponent"] == 40 and S["n_none"] == 30
    assert S["secondary_verdicts"] == {"C1": "PASS", "C2": "PASS"}       # |−10| ≤ 0.15·190; |190 − 200| ≤ 0.25·190
    T2 = _table()
    for r in T2:
        if r["class"] == "none" and r["seed"] < 1007:
            r["t_obs"], r["s_obs"] = 900, 5.5                                       # 7 of 30 none seeds cross
    assert PS.score_tables(T2)["outcome"] == "FAIL H"
    T3 = _table()
    for r in T3:
        if r["class"] == "clean" and r["log2rho"] == -16.0 and r["seed"] == 0:
            r["s_obs"] = 4.9                                                        # the falsifier
    S3 = PS.score_tables(T3)
    assert S3["verdicts"]["F"] == "FAIL" and S3["criteria"]["F"]["falsified"]
    T4 = _table()
    for r in T4:
        if r["class"] == "clean" and r["seed"] < 5 and r["log2rho"] == -16.0:
            r["follow_in_M"] = False
    assert PS.score_tables(T4)["outcome"] == "UNRESOLVED (validity)"
    T5 = _table()
    for r in T5:
        if r["class"] == "clean" and r["log2rho"] == -14.0 and r["seed"] == 3:
            r["sign_ok"] = False                                                    # a 2⁻¹⁴ clean run
    assert PS.score_tables(T5)["outcome"] == "UNRESOLVED (validity)"
    assert PS.score_tables(_table(), gate_ok=False)["outcome"] == "UNRESOLVED (gate)"
    T6 = _table(n_none=9)
    assert PS.score_tables(T6)["outcome"] == "UNRESOLVED"                         # H: fewer than 10 none seeds


def test_score_tables_C2_is_secondary_and_mixed_never_scored():
    T = _table()
    for r in T:
        if r["class"] == "clean" and r["log2rho"] == -16.0:
            r["delay_fc"] = 100                                                     # |100 − 200|/100 = 1 > 0.25
    S = PS.score_tables(T)
    assert S["secondary_verdicts"]["C2"] == "FAIL" and S["outcome"] == "PASS"
    T = _table()
    for r in T:
        if r["class"] == "mixed":
            r.update(t_obs=10, s_obs=1.0, follow_in_M=False, idle_ok=False)
    assert PS.score_tables(T)["outcome"] == "PASS"


# ================================================================================================ ENFORCEMENT
def _exp_path(t_c, alpha=1e-3, extra=400):
    return np.exp(alpha * np.arange(t_c + extra, dtype=float))


def test_enforcement_leaky_forecaster_is_caught(monkeypatch):
    s = _exp_path(300)
    real = CFF.forecast_fold

    def leaky(s_view, t_c, *a, **k):
        _ = s_view[t_c]
        return real(s_view, t_c, *a, **k)
    monkeypatch.setattr(CFF, "forecast_fold", leaky)
    with pytest.raises(CF.CausalityViolation):
        CFF.run_forecast(s, 300, 2.0, 1e-3, 1.0, 10 ** 4, 0.95)

    def leaky_slice(s_view, t_c, *a, **k):
        _ = s_view[: t_c + 1]
        return real(s_view, t_c, *a, **k)
    monkeypatch.setattr(CFF, "forecast_fold", leaky_slice)
    with pytest.raises(CF.CausalityViolation):
        CFF.run_forecast(s, 300, 2.0, 1e-3, 1.0, 10 ** 4, 0.95)


def test_enforcement_guard_bypass_is_caught_by_the_nan_recomputation(monkeypatch):
    s = _exp_path(300)
    real = CFF.forecast_fold
    private = {"a": s}

    def bypass(s_view, t_c, *a, **k):
        out = real(s_view, t_c, *a, **k)
        out["peek"] = float(private["a"][t_c]) if isinstance(s_view, CF.GuardedArray) else float(np.asarray(s_view)[t_c])
        return out
    monkeypatch.setattr(CFF, "forecast_fold", bypass)
    rec, same = CFF.run_forecast(s, 300, 2.0, 1e-3, 1.0, 10 ** 4, 0.95)
    assert not same and rec["nan_recompute_identical"] is False


# ================================================================================================ the exploratory seed
@pytest.fixture(scope="module")
def frozen_2930006():
    import torch
    torch.set_num_threads(1)
    return PS.freeze_seed(2_930_006, timing=False)


def test_reproduces_the_exploratory_freeze_of_seed_2930006(frozen_2930006):
    """p2a_explore_h (grid, seed 2,930,006): every frozen quantity the page's exploration numbers rest on, bit for bit
    (the fold, the h/2 fold, the lower fold, M's stable range, |m′c′| and Λ_F, s*, s₀, every perturbed-minimum count,
    the homotopy record)."""
    r, h = frozen_2930006, _explore_h(2_930_006)
    assert r["fold_up"] == h["M"]["fold_up"] and r["fold_down"] == h["M"]["fold_down"]
    assert r["fold_up_h2"] == h["fold_h0.025"] and r["M_stable"] == h["M_stable"]
    assert r["M_homotopy"]["n_steps"] == h["M"]["n_steps"] == 50
    assert r["M_homotopy"]["homotopy_min_eig"] == h["M"]["homotopy_min_eig"]
    assert r["fold_constant"]["abs_mc"] == h["abs_mc"] and r["fold_constant"]["Lambda_F"] == h["Lambda_F"]
    assert r["s_star"] == h["s_star"] and r["s0"] == h["s0_frac"]
    names = {"0.99": "below_1pct", "0.999": "below_0.1pct", "1.001": "above_0.1pct", "1.01": "above_1pct"}
    for k, hk in names.items():
        for f in ("n_on_M", "n_rho2_ge_q", "rho2_med"):
            assert r["validation"][k][f] == h["validation"][hk][f], (k, f)
    assert r["S_homotopy"]["homotopy_min_eig"] == h["S"]["homotopy_min_eig"] and r["S_fold_up"] == h["S"]["fold_up"]
    assert r["rho2_S_at_1.01sF"] == pytest.approx(h["rho2_S_at_1.01sF"], rel=1e-12)
    assert r["rho2_M_max"] >= h["rho2_M_max"] - 1e-3 and r["rho2_M_max"] < PS.Q
    e = r["evaluation"]
    assert e["class"] == "clean" and e["eligible"] and e["checks"]["h_half_agrees"] and e["checks"]["fold_eig_zero"]
    assert e["checks"]["none_on_M_above"] and e["checks"]["lam_min_decreasing_to_fold"]
    # this clean seed's |m′c′| windows disagree by 9.7% (> 5%): the page's rule 4 leaves it unscored (disclosed)
    assert e["status"] == "Lambda_F not validated" and r["fold_constant"]["rel_diff"] == pytest.approx(0.0974, abs=1e-3)
    assert r["release_active_units"] == [0, 1, 2] and r["release_rho2"] < PS.Q and r["release_grad_max"] < 1e-9
    assert PS.evaluate(json.loads(json.dumps(PS._jsonable(r)))) == e                  # JSON round trip


def test_untraceable_exploratory_seed():
    r = PS.freeze_seed(2_930_003, timing=False)
    h = _explore_h(2_930_003)
    assert r["evaluation"]["status"] == "untraceable: M homotopy failed" and not h["traceable"]
    assert [a["n_steps"] for a in r["M_homotopy"]["attempts"]] == [50, 200]
    assert r["M_homotopy"]["attempts"][0]["lambda_failed"] == h["M"]["first"]["lambda_failed"]
    assert r["M_homotopy"]["attempts"][1]["lambda_failed"] == h["M"]["lambda_failed"]


def _fs(rec):
    return {**rec, "status": rec["evaluation"]["status"], "class": rec["evaluation"]["class"],
            "Lambda_F": rec["evaluation"]["Lambda_F"]}


def test_real_run_and_observation_reproduce_the_exploration_at_2_to_minus_13(frozen_2930006):
    """p2a_explore_h's 2⁻¹³ run of seed 2,930,006 (ρ₂ every 20 steps refined to the step; its step written as
    (g − a(a·g)) + ρa(a·g)): t_c 109,399; t_F 126,577; t_obs 141,615; own forecast t̂_F 126,783, t_fc 139,682; the
    fixed-dataset forecast t_fc 138,260.  The registered code (Phase 2A's step, ρ₂ every step) reproduces them."""
    fs = _fs(frozen_2930006)
    h = _explore_h(2_930_006, "run", -13.0)
    rr = PS.run_one(fs, -13.0)
    assert rr["t_c"] == h["t_c"] and rr["nan_recompute_identical"]
    fc, fx = rr["forecast"], rr["forecast_fixed"]
    assert fc["max_index_read"] == rr["t_c"] - 1 and fx["max_index_read"] == rr["t_c"] - 1
    assert (fc["t_F_fc"], fc["t_fc"]) == (h["fc_own"]["t_F_fc"], h["fc_own"]["t_fc"])
    assert fc["eps_fc"] == pytest.approx(h["fc_own"]["eps_fc"], rel=1e-9)
    assert fx["t_fc"] == h["fc_pop"]["t_fc"] and fx["s_c_fc"] == pytest.approx(h["fc_pop"]["s_c_fc"], rel=1e-12)
    assert rr["budget"] == PS.budget(2.0 ** -13) == 524_288
    ob, S = PS.observe_one(fs, -13.0, rr)
    assert ob["state_tc_hash_ok"] and ob["t_obs"] == h["t_obs"] and ob["t_F"] == h["t_F"]
    assert ob["s_obs"] == pytest.approx(h["s_obs"], rel=1e-12) and ob["rho2_at_obs"] >= PS.Q
    assert ob["follow_in_M"] and ob["follow_dist_M"] < 1e-6
    assert ob["idle_zero_to_crossing"] and ob["signs_fixed_to_crossing"] and not ob["reached_obs_end"]
    row = PS.run_row(fs, rr, ob)
    assert row["has_fc"] and row["r_obs"] == pytest.approx(h["r_obs"], rel=1e-12)
    assert row["s_c_fixed"] == fx["s_c_fc"]
    # the real path: the forecaster reads rows < t_c only; NaN / garbage beyond the cutoff changes nothing
    t_c = rr["t_c"]
    g = CF.GuardedArray(S, t_c, name="s")
    rec = CFF.forecast_fold(g, t_c, fs["s_F"], fs["Lambda_F"], 1.0, rr["budget"] - t_c, 0.95)
    assert g.max_index_read == t_c - 1 and not g.violations
    for poison in (np.nan, 1e9, -3.0):
        S2 = S.copy()
        S2[t_c:] = poison
        assert CFF.same_forecast(rec, CFF.forecast_fold(S2, t_c, fs["s_F"], fs["Lambda_F"], 1.0, rr["budget"] - t_c,
                                                        0.95))


def test_run_one_aborts_on_a_differing_recomputation(monkeypatch, frozen_2930006):
    fs = _fs(frozen_2930006)
    monkeypatch.setattr(CFF, "run_forecast", lambda *a, **k: ({"status": "ok"}, False))
    with pytest.raises(CF.CausalityViolation):
        PS.run_one(fs, -8.0)


def test_run_one_leaky_forecaster_raises(monkeypatch, frozen_2930006):
    fs = _fs(frozen_2930006)
    real = CFF.forecast_fold

    def leaky(s_view, t_c, *a, **k):
        _ = s_view[t_c]
        return real(s_view, t_c, *a, **k)
    monkeypatch.setattr(CFF, "forecast_fold", leaky)
    with pytest.raises(CF.CausalityViolation):
        PS.run_one(fs, -8.0)


def test_run_one_hard_budget_and_forecast_budget(monkeypatch, frozen_2930006):
    fs = _fs(frozen_2930006)
    r = PS.run_one(fs, -8.0, n_budget=10)
    assert r["t_c"] is None and r["forecast"]["status"].startswith("no forecast") and r["budget"] == 10
    seen = []
    real = CFF.run_forecast

    def spy(S, t_c, s_F, L, eta, budget_left, f):
        seen.append((t_c, s_F, L, budget_left, f))
        return real(S, t_c, s_F, L, eta, budget_left, f)
    monkeypatch.setattr(CFF, "run_forecast", spy)
    r = PS.run_one(fs, -8.0)
    nb = PS.budget(2.0 ** -8)
    assert seen == [(r["t_c"], fs["s_F"], fs["Lambda_F"], nb - r["t_c"], 0.95),
                    (r["t_c"], PS.S_F_FIXED, PS.LAMF_FIXED, nb - r["t_c"], 0.95)]


def test_cutoff_is_a_stopping_time_and_deterministic(frozen_2930006):
    fs = _fs(frozen_2930006)
    X, Y = PS.sample(fs["seed"])
    S, th, t_c = PS.train_to_cutoff(np.array(fs["release_theta"]), 2.0 ** -8, fs["s_F"], X, Y)
    assert S[t_c] >= 0.95 * fs["s_F"] and (S[1:t_c] < 0.95 * fs["s_F"]).all() and len(S) == t_c + 1
    S2, th2, t2 = PS.train_to_cutoff(np.array(fs["release_theta"]), 2.0 ** -8, fs["s_F"], X, Y)
    assert t2 == t_c and np.array_equal(th, th2)


def test_observe_detects_a_tampered_state(frozen_2930006):
    fs = _fs(frozen_2930006)
    X, Y = PS.sample(fs["seed"])
    S, th, t_c = PS.train_to_cutoff(np.array(fs["release_theta"]), 2.0 ** -8, fs["s_F"], X, Y)
    bad = th.copy()
    bad[0] = np.nextafter(bad[0], np.inf)
    rr = {"t_c": t_c, "state_tc": [float(x) for x in bad], "state_tc_sha256": PS.state_sha(bad)}
    with pytest.raises(AssertionError):
        PS.observe_one(fs, -8.0, rr, n_budget=t_c + 10)


def test_run_one_refuses_a_wrong_sample_or_release(frozen_2930006):
    fs = _fs(frozen_2930006)
    with pytest.raises(AssertionError):
        PS.run_one({**fs, "sample_sha256": "0" * 64}, -8.0, n_budget=5)
    th = list(fs["release_theta"])
    th[0] += 1e-12
    with pytest.raises(AssertionError):
        PS.run_one({**fs, "release_theta": th}, -8.0, n_budget=5)


def test_forbidden_keys_never_in_a_run_record(frozen_2930006):
    fs = _fs(frozen_2930006)
    r = PS.run_one(fs, -8.0)
    assert not (PS.FORBIDDEN & P2A._keys(r))


def test_code_closure_has_no_observation_reader_in_the_forecaster():
    import ast
    tree = ast.parse((ROOT / "src/causal_forecast_fold.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not {"rho2", "Rho2Grid", "t_obs", "s_obs"} & names
    assert "src/causal_forecast_fold.py" in PS.code_closure() and "src/phase2a.py" in PS.code_closure()


# ================================================================================================ frozen artifacts
@pytest.mark.skipif(not FROZEN.exists(), reason="not yet frozen")
def test_frozen_summary_consistent():
    fr = json.loads(FROZEN.read_text())
    rows = PS._rows(ROOT / "results/phase2a_ps/frozen_parts.jsonl")
    assert sorted(r["seed"] for r in rows) == list(PS.SEEDS)
    assert fr["frozen_parts_sha256"] == PS._sha(ROOT / "results/phase2a_ps/frozen_parts.jsonl")
    import hashlib
    assert hashlib.sha256(PS.class_listing(rows).encode()).hexdigest() == fr["class_listing_sha256"]
    for r in rows:
        assert PS.evaluate(r) == r["evaluation"]
    assert fr["counts"] == PS.counts(rows) and fr["gate"] == PS.gate(rows)
    pl, _ = PS.plan(rows)
    assert [(p["seed"], p["class"], p["log2rho"]) for p in fr["plan"]] == pl
    assert sum(fr["counts"]["by_status"].values()) == 400


@pytest.mark.skipif(not PILOT.exists(), reason="pilot not yet run")
def test_pilot_is_pilot_seeds_only_and_ran_clean():
    P = json.loads(PILOT.read_text())
    rows = PS._rows(ROOT / "results/phase2a_ps/pilot_parts.jsonl")
    assert all(r["seed"] in PS.PILOT_SEEDS for r in rows)
    if P.get("runs"):
        assert P["seed"] in PS.PILOT_SEEDS
        for k, r in P["runs"].items():
            assert r["nan_recompute_identical"] and r["observed"]["state_tc_hash_ok"] and r["peak_rss_gb"] < 3.0
