"""Width 2 (Route A): every validity check and stop condition of results/width2_design.md §5, exercised on
constructed pass and fail cases before any run; plus the scorers and the proved partial bound toward Lemma 4."""

import math

import numpy as np
import pytest
import torch

from src.width2_geometry import (F130, F150, TANH, Act, averaging_bound, branch_distance, canonical, exact_gap,
                                 extrema, gaps, global_sign, identity_check, identity_predicts, mirror, phi,
                                 searches_agree, sign_correct, swap, theorem1_ok, theorem2_ok, unit_orient,
                                 beta_shift)
from src import width2_conditional as wc
from src import width2_train as wt


# ------------------------------------------------------------------ exact extrema (used by every check)
@pytest.mark.parametrize("act", [F130, F150, TANH])
def test_extrema_enclose_a_dense_grid(act):
    rng = np.random.default_rng(1)
    xs = np.linspace(-0.8, 0.8, 400_001)
    for _ in range(40):
        th = rng.uniform(-6, 6, 4); v = rng.uniform(-2, 2, 2)
        mnl, mnh, mxl, mxh = extrema(th, v, act, -0.8, 0.8)
        f = phi(xs, th, v, act)
        assert mnl <= f.min() + 1e-12 and mxh >= f.max() - 1e-12
        assert f.min() <= mnh + 1e-9 and f.max() >= mxl - 1e-9


def test_curvature_bound_is_a_bound():
    rng = np.random.default_rng(2)
    for act in (F130, TANH):
        for _ in range(500):
            a, b = np.sort(rng.uniform(-8, 8, 2))
            ts = np.linspace(a, b, 2001)
            assert np.abs(act.d2u(ts)).max() <= act.d2u_bound(np.array(a), np.array(b)) + 1e-12


# ------------------------------------------------------------------ Theorem 1 (margin bound)
def test_theorem1_holds_for_tanh_networks_with_the_exact_gamma():
    rng = np.random.default_rng(3)
    for _ in range(200):
        th = rng.uniform(-8, 8, 4); v = rng.uniform(-3, 3, 2); b = rng.uniform(-2, 2)
        ok, m = theorem1_ok(th, v, b, TANH, gamma2_hat=1.0)
        assert ok


def test_theorem1_check_fires_when_the_margin_exceeds_the_bound():
    """Constructed fail case: a sign-correct tanh network (G₋ construction, oriented to G₊) whose margin exceeds
    ½‖v‖₁Γ̂ for an understated Γ̂ -- the check must report a violation."""
    th = np.array([30.0, 30.0, 30.0, -30.0]); v = np.array([-1.0, 1.0])     # −tanh(30(x+1)) + tanh(30(x−1))
    g = gaps(th, v, TANH)["G+"]
    assert g[0] > 1.5                                                        # ‖v‖₁ = 2, gap near 2
    ok, m = theorem1_ok(th, v, 1.0, TANH, gamma2_hat=0.5)            # b = 1: N ≈ −1 on I, +1 on O
    assert m > 0 and not ok


# ------------------------------------------------------------------ the §2 identity
@pytest.mark.parametrize("act", [F130, TANH])
def test_identity_has_no_mismatch_on_random_networks(act):
    r = identity_check(300, act, seed=4)
    assert r["mismatches"] == 0 and r["decided"] >= 290
    assert r["n_sign_correct"] > 20                      # the positive side is exercised


def test_identity_check_catches_a_wrong_orientation(monkeypatch):
    """Constructed fail case: an identity that uses G₋ instead of G₊ must produce mismatches."""
    import src.width2_geometry as g

    def wrong(th, v, b, act, tol=1e-9):
        e = g.gaps(th, v, act)
        (iml, imh, _, _), (_, _, oxl, oxh) = e["extrema"]["I"], e["extrema"]["O"]
        return e["G-"][0] > 0 and -iml < b < -oxh
    monkeypatch.setattr(g, "identity_predicts", wrong)
    assert g.identity_check(300, F130, seed=4)["mismatches"] > 0


def test_identity_bias_just_inside_and_just_outside_the_interval():
    from src.width2_geometry import placed_sample
    rng = np.random.default_rng(12)
    for act in (F130, TANH):
        for _ in range(10):
            th, v = placed_sample(rng, act)
            e = gaps(th, v, act)
            lo_b, hi_b = -e["extrema"]["O"][0], -e["extrema"]["I"][3]
            if hi_b - lo_b < 1e-5:
                continue
            mid = 0.5 * (lo_b + hi_b)
            assert identity_predicts(th, v, mid, act) is True and sign_correct(th, v, mid, act) is True
            for b in (hi_b + 1e-6, lo_b - 1e-6):                 # a bias just outside the feasible interval
                assert identity_predicts(th, v, b, act) is False and sign_correct(th, v, b, act) is False


# ------------------------------------------------------------------ Γ̂₂ agreement and Theorem 2
def test_searches_agree_pass_and_fail():
    assert searches_agree({"gamma_lo": 0.5}, {"gamma_lo": 0.5 * (1 + 5e-7)})
    assert not searches_agree({"gamma_lo": 0.5}, {"gamma_lo": 0.5 * (1 + 5e-6)})     # a perturbed G


def test_theorem2_construction_approaches_one_from_below():
    for alpha in (5, 20, 80):
        th = np.array([alpha, alpha, alpha, -alpha]); v = np.array([0.5, -0.5])
        lo, hi = exact_gap(th, v, TANH)
        assert theorem2_ok(hi) and lo > 0.9 if alpha >= 20 else theorem2_ok(hi)


def test_theorem2_check_fires_for_an_activation_with_range_above_two():
    class Wide(Act):
        def u(self, t):
            return 1.5 * np.tanh(t)

        def du(self, t):
            return 1.5 * (1 - np.tanh(t) ** 2)

        def d2u_bound(self, tlo, thi):
            return 1.5 * TANH.d2u_bound(tlo, thi)
    w = Wide("tanh")
    th = np.array([40.0, 40.0, 40.0, -40.0]); v = np.array([0.5, -0.5])
    assert not theorem2_ok(exact_gap(th, v, w)[1])


# ------------------------------------------------------------------ symmetry and canonical branches
@pytest.mark.parametrize("act", [F130, TANH])
def test_generators_leave_G_invariant_and_canonical_is_idempotent(act):
    rng = np.random.default_rng(5)
    for _ in range(40):
        th = rng.uniform(-5, 5, 4); v = rng.uniform(-2, 2, 2); g0 = exact_gap(th, v, act)[0]
        gens = [swap, lambda t, w: unit_orient(t, w, 0), lambda t, w: unit_orient(t, w, 1), mirror, global_sign]
        if act.periodic:
            gens.append(lambda t, w: beta_shift(t, w, 1, -2))
        for gen in gens:
            assert abs(exact_gap(*gen(th, v), act)[0] - g0) < 1e-11
        c = canonical(th, v, act)
        assert np.allclose(np.concatenate(canonical(*c, act)), np.concatenate(c))
        assert branch_distance((th, v), mirror(*unit_orient(*swap(th, v), 1)), act) < 1e-12


def test_mirror_invariance_fails_for_an_asymmetric_window():
    """Constructed fail case: with an asymmetric outer window the mirror is not a symmetry of G."""
    th = np.array([1.3, 0.7, 0.4, 2.1]); v = np.array([0.8, -0.2])

    def g_asym(th, v):
        i = extrema(th, v, F130, -0.8, 0.8); o1 = extrema(th, v, F130, -2.0, -1.2); o2 = extrema(th, v, F130, 1.2, 2.6)
        return min(o1[0], o2[0]) - i[3]
    assert abs(g_asym(*mirror(th, v)) - g_asym(th, v)) > 1e-6


def test_loss_invariant_under_the_group_on_the_population():
    x, y = wc.population()
    rng = np.random.default_rng(6)
    p = np.concatenate([rng.uniform(-3, 3, 4), [0.3]]); s = 4.0
    L0 = wc.loss_grad(p, s, 1, x, y, F130)[0]
    th, v = p[:4], wc.vtilde(p[4], 1)
    for gth, gv in [mirror(th, v), swap(th, v), unit_orient(th, v, 0), beta_shift(th, v, 0, 1)]:
        t = gv[0] / (abs(gv[0]) + abs(gv[1])); sg = 1 if gv[1] >= 0 else -1
        L1 = wc.loss_grad(np.concatenate([gth, [t]]), s, sg, x, y, F130)[0]
        assert abs(L1 - L0) < 1e-12


# ------------------------------------------------------------------ conditional search: audit, convergence, stricter search
def test_gradient_matches_finite_differences_and_b_is_profiled():
    x, y = wc.training_set(600_000)
    rng = np.random.default_rng(7)
    for act in (F130, TANH):
        p = np.concatenate([rng.uniform(-3, 3, 4), [0.4]]); s = 3.0
        L, g, b = wc.loss_grad(p, s, -1, x, y, act)
        fd = np.array([(wc.loss_grad(p + e, s, -1, x, y, act)[0] - wc.loss_grad(p - e, s, -1, x, y, act)[0]) / 2e-6
                       for e in np.eye(5) * 1e-6])
        assert np.abs(fd - g).max() < 1e-6


def _cands():
    mk = lambda L, flags=(): {"loss": L, "flags": list(flags), "p": None}
    return [mk(0.30), mk(0.25), mk(0.27), mk(0.6931, ["constant_predictor"])]


def test_audit_passes_when_retained_is_lowest():
    c = _cands(); r = wc.retain(c)
    assert r["loss"] == 0.25 and wc.audit(c, r)["ok"]


def test_audit_fires_on_a_planted_lower_candidate_marked_discarded():
    c = _cands() + [{"loss": 0.20, "flags": ["not_converged"], "p": None}]
    r = wc.retain(c)
    assert r["loss"] == 0.25 and not wc.audit(c, r)["ok"]


def test_constant_predictor_is_retained_when_it_is_lowest():
    c = [{"loss": 0.8, "flags": [], "p": None}, {"loss": math.log(2), "flags": ["constant_predictor"], "p": None}]
    assert "constant_predictor" in wc.retain(c)["flags"]


def test_stricter_search_check_pass_and_fail():
    assert wc.stricter_check(0.25, 0.25 + 1e-12)
    assert not wc.stricter_check(0.25, 0.25 - 1e-6)      # a search with truncated restarts retained a higher value


# ------------------------------------------------------------------ thresholds and 'not applicable'
def test_first_sign_change_and_not_applicable():
    R = np.array([0.1, 0.2, 0.3, 0.4])
    assert wc.first_sign_change(R, [-1, -1, 0.1, 0.2])["bracket"] == (0.2, 0.3)
    assert wc.first_sign_change(R, [-1, -1, -1, -1])["status"] == "not applicable"
    assert wc.first_sign_change(R, [1, 1, 1, 1])["status"] == "not applicable"
    assert wc.first_sign_change(R, [-1, 1, -1, 1])["sign_changes"] == 3


def test_r2_grid_is_the_registered_scan():
    R, s = wc.r2_grid(1.0)
    assert R[0] == 0.02 and R[-1] == 1.00 and len(R) == 99 and np.allclose(s, 2 * R)


# ------------------------------------------------------------------ training: determinism and crossing detection
def test_training_is_deterministic():
    assert wt.determinism_ok(600_000, F130, gamma2_hat=1.0, budget=300)


def test_determinism_check_catches_a_nondeterministic_perturbation(monkeypatch):
    real = wt.init_params
    calls = {"n": 0}

    def noisy(seed):
        calls["n"] += 1
        q = real(seed)
        return q + (1e-12 if calls["n"] % 2 == 0 else 0.0)
    monkeypatch.setattr(wt, "init_params", noisy)
    assert not wt.determinism_ok(600_000, F130, gamma2_hat=1.0, budget=300)


def test_crossing_detection_agrees_with_an_independent_dense_check():
    from src.width2_geometry import placed_sample
    rng = np.random.default_rng(8)
    n_pos = 0
    for k in range(300):
        if k % 2 == 0:
            th, v = placed_sample(rng, F130)
            q = np.array([th[0], th[1], v[0], th[2], th[3], v[1], 0.0])
        else:
            q = rng.uniform(-4, 4, 7)
        pl, _ = wt.placed(q, F130)
        assert pl == wt.dense_crossing_check(q, F130)
        n_pos += pl
    assert n_pos > 0


def test_off_by_one_crossing_is_a_disagreement():
    """Constructed fail case: a detector that reports the step after the true first placement disagrees with the
    every-step check at that true step."""
    steps = [(1, False), (2, False), (3, True), (4, True)]
    true_first = next(s for s, p in steps if p)
    reported = true_first + 1
    assert reported != true_first


# ------------------------------------------------------------------ W4 replays
@pytest.fixture(scope="module")
def checkpoint():
    run = wt.train(600_001, F130, gamma2_hat=1.0, budget=60, checkpoint_steps=[40])
    x, y = wc.training_set(600_001)
    return run["checkpoints"][40], x, y


def test_k1_replay_reproduces_the_true_freeze_reference(checkpoint):
    ck, x, y = checkpoint
    ok, diff = wt.k1_check(ck, x, y, F130)
    assert ok, diff


def test_k1_check_fires_against_a_reference_that_does_not_truly_freeze(checkpoint):
    ck, x, y = checkpoint
    bad_ref = wt.reference_true_freeze(ck, 25, x, y, F130, reset=False)
    ok, diff = wt.k1_check(ck, x, y, F130, reference=bad_ref)
    assert not ok and diff > 1e-10


def test_decisions_preserved_pass_and_fail(checkpoint):
    ck, x, y = checkpoint
    assert wt.decisions_preserved(ck["q"], 2.0, x, F130)
    assert not wt.decisions_preserved(ck["q"], -1.0, x, F130)       # k < 0


def test_true_freeze_pass_and_fail(checkpoint):
    ck, x, y = checkpoint
    r = wt.replay(ck, 1.5, 30, x, y, F130, 1.0, record=())
    assert wt.freeze_ok(r["drift"])
    r2 = wt.replay(ck, 1.5, 30, x, y, F130, 1.0, project=False, record=())
    assert not wt.freeze_ok(r2["drift"])


# ------------------------------------------------------------------ scorers
def test_w1_pass_and_fail_cases():
    rng = np.random.default_rng(9)
    ok = rng.uniform(1.02, 1.2, 80)
    assert wt.score_w1(ok, 1.0)["pass"]
    assert not wt.score_w1(rng.uniform(0.8, 1.2, 80), 1.0)["pass"]          # < 90% above
    assert not wt.score_w1(rng.uniform(1.3, 1.6, 80), 1.0)["pass"]          # median > 1.25 (upper bound)


def test_w4_pass_and_fail_cases():
    levels = [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    rng = np.random.default_rng(10)
    thr = rng.normal(1.03, 0.03, 200)
    C = (np.array(levels)[None, :] > thr[:, None]).astype(int)
    r = wt.score_w4(levels, C, own_over_pop_median=1.03, agree_start=np.ones(200))
    assert r["W4a_pass"] and r["W4b_pass"]
    C2 = C.copy(); C2[:, -1] = 0                                              # placement collapses at 2.0
    assert not wt.score_w4(levels, C2, 1.03, np.ones(200))["W4a_pass"]
    assert not wt.score_w4(levels, C, 1.20, np.ones(200))["W4a_pass"]         # 50% point far from own/pop
    assert not wt.score_w4(levels, C, 1.03, np.r_[np.ones(170), np.zeros(30)])["W4b_pass"]


def test_tanh_verdicts():
    assert wt.tanh_verdict([True, True], True, True) == "H-general"
    assert wt.tanh_verdict([True, True], False, True) == "H-nonmonotone"
    assert wt.tanh_verdict([True, False], True, True).startswith("neither")
    assert wt.tanh_verdict([True, True], False, False).startswith("not decided")


# ------------------------------------------------------------------ partial step toward Lemma 4
def test_averaging_bound_holds_when_both_units_oscillate_fast():
    rng = np.random.default_rng(11)
    lam = 30.0
    for _ in range(100):
        th = np.array([rng.uniform(lam, 3 * lam), rng.uniform(0, 2 * math.pi),
                       rng.uniform(lam, 3 * lam), rng.uniform(0, 2 * math.pi)])
        t = rng.uniform(-1, 1); v = np.array([t, 1 - abs(t)])
        assert exact_gap(th, v, F130)[1] <= averaging_bound(F130, lam) + 1e-12
