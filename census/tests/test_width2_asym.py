"""W2-A (registered lag-law test at width 2 on the asymmetric windows; arms T, D, T′): every decision rule on constructed
pass, fail and unresolved cases -- the three gates (T′'s hold condition included), release classification (windings,
on neither, unconverged), the follow check (D still duplicate), the own-v-path switch procedure, signed κ with the total
derivative and D's split block excluded, the winding table, L1-L5 with the |lag| ≥ 6 rule, the bootstrap, V1-V7,
UNRESOLVED propagation, the pilot rules (STOP branches) -- plus the analytic gradient and Hessian against finite
differences and integration checks on the population (no registered or pilot seed is drawn, held or trained)."""

import math

import numpy as np
import pytest

from src import width2_asym as X


# ------------------------------------------------------------------------------------------ fixtures (population only)
@pytest.fixture(scope="module")
def pop():
    return X.population()


@pytest.fixture(scope="module")
def s0():
    return X.s0_value()


@pytest.fixture(scope="module")
def pts(pop, s0):
    """The four population copy points at v₀ (damped Newton from the exploratory class points)."""
    x, y = pop
    out = {}
    for c in ("D", "Dp", "T", "Tp"):
        v = X.v_held(X.COPY_SHARES[c], s0)
        z, gm, lam, ok = X.newton(np.array(X.POP_START[c]), v, x, y)
        assert ok
        out[c] = (z, v)
    return out


# ------------------------------------------------------------------------------------------ registered constants
def test_registered_constants():
    assert X.SEEDS == tuple(range(884_000, 884_120)) and X.SEEDS_TP == tuple(range(884_200, 884_320))
    assert X.PILOT_SEEDS == tuple(range(884_900, 884_910))
    assert not (set(X.SEEDS) & set(X.SEEDS_TP)) and not (set(X.SEEDS + X.SEEDS_TP) & set(X.PILOT_SEEDS))
    assert X.ARMS == ("T", "D", "Tp") and X.HEADLINE == "T"
    assert X.SHARES == {"T": (0.1, 0.9), "D": (0.5, 0.5), "Tp": (0.1, 0.9)}
    assert X.ETA == {"T": 0.03, "D": 0.3, "Tp": 0.03}
    assert X.ARM_SEEDS == {"T": X.SEEDS, "D": X.SEEDS, "Tp": X.SEEDS_TP}
    assert X.SCORED_COPIES == {"T": ("T",), "D": ("D", "Dp"), "Tp": ("Tp",)}
    assert X.OWN_PATH_SWITCH == {"T": True, "D": False, "Tp": True}
    assert (X.HOLD_LR, X.W_MIN, X.W_RELAX) == (1.0, 4000, 25.0)
    assert (X.BUDGET_D, X.BUDGET_T_BASE, X.RHO_T_START) == (40_000, 100_000, 2.0 ** -10)
    assert (X.NEWTON_GTOL, X.ON_TOL, X.STATE_TOL, X.FOLLOW_FRAC, X.FOLLOW_STATE_TOL) == (1e-8, 1e-6, 1e-3, 0.8, None)
    assert X.GATE_MIN == {"T": 60, "D": 60, "Tp": 108}
    assert X.GATE_HOLD_CONDITION == {"T": False, "D": False, "Tp": True}
    assert (X.L1_BAND, X.L2_BAND, X.L3_MIN) == ((0.90, 1.10), (0.80, 1.20), 0.5)
    assert (X.BOOT_N, X.BOOT_PCT) == (10_000, (2.5, 97.5))
    assert (X.MIN_SCORED, X.TSW_MIN_FRAC, X.REGIME_MAX, X.REGIME_MIN_FRAC, X.LAG_MIN_STEPS, X.KC_Q90_MAX,
            X.CHI_REL_TOL, X.CHI_PATH_Q90_MAX) == (60, 0.90, 0.5, 0.80, 10.0, 0.1, 0.30, 0.25)
    assert X.L5_NMIN == 6 == 2 * (1 + 1 + 1)
    assert (X.PILOT_KC_MAX, X.PILOT_CHI_MAX, X.PILOT_MAX_HALVINGS) == (0.1, 0.1, 3)
    assert X.KSUM_RANGE == (-3, 3)
    assert X.OUTER == ((-2.0, -1.2), (1.2, 2.4)) and X.INNER == (-0.8, 0.8)


def test_s0_is_half_of_t2_1_bracket_midpoint():
    assert X.s_pop2() == pytest.approx(0.44508, abs=1e-5)
    assert X.s0_value() == 0.5 * X.s_pop2()
    assert list(X.v_held((0.1, 0.9), 1.0)) == [0.1, 0.9]


def test_hidden_draw_is_width2_train_draw_without_touching_global_state():
    import torch
    before = torch.random.get_rng_state().clone()
    z = X.hidden_draw(884_123)
    assert torch.equal(before, torch.random.get_rng_state())
    with torch.random.fork_rng():
        from src.width2_train import init_params
        ref = init_params(884_123).numpy()
    assert np.array_equal(z, ref[[0, 1, 3, 4, 6]]) and z.dtype == np.float64


# ------------------------------------------------------------------------------------------ numerics
def test_gradient_and_hessian_match_finite_differences(pop):
    x, y = pop
    rng = np.random.default_rng(7)
    for _ in range(3):
        q = rng.uniform(-2, 2, 7)
        g, H = X.grad(q, x, y), X.hess(q, x, y)
        e = 1e-6
        gfd = np.array([(X.loss(q + e * u, x, y) - X.loss(q - e * u, x, y)) / (2 * e) for u in np.eye(7)])
        Hfd = np.array([(X.grad(q + e * u, x, y) - X.grad(q - e * u, x, y)) / (2 * e) for u in np.eye(7)])
        assert np.abs(g - gfd).max() < 1e-8 and np.abs(H - Hfd).max() < 1e-7
        assert np.allclose(H, H.T, atol=1e-14)


def test_newton_accepts_a_minimum_and_rejects_a_saddle(pop, pts):
    x, y = pop
    z, v = pts["T"]
    zn, gm, lam, ok = X.newton(z + 1e-3, v, x, y)
    assert ok and gm < 1e-8 and lam > 0 and np.abs(zn - z).max() < 1e-9
    zn, gm, lam, ok = X.newton(z, v, x, y, maxit=0)                       # accepted as it stands
    assert ok
    q = np.zeros(5)                                                        # z = 0: not stationary, no iterations
    assert not X.newton(q, v, x, y, maxit=0)[3]


def test_branch_derivative_matches_finite_differences(pop, pts):
    x, y = pop
    z, v = pts["T"]
    J = X.dz_dv(z, v, x, y)
    for i in range(2):
        e = np.zeros(2); e[i] = 1e-6
        zp = X.newton(z, v + e, x, y)[0]
        zm = X.newton(z, v - e, x, y)[0]
        assert np.abs((zp - zm) / 2e-6 - J[:, i]).max() < 1e-5


def test_dense_gap_bounds_the_enclosure_and_placement_rules(pts):
    z, v = pts["T"]
    lo, hi = X.gap_enclosure(z, v)
    assert lo <= hi and hi - lo < 1e-9 and X.gap_dense(z, v) >= lo
    assert X.placed_state(z, v) == (False, False) and not X.branch_placed(z, v)
    # a placed state: the population's placed two-unit class at equal shares (exploration: G₊ = 0.168 > 0)
    zp = np.array([-1.547774, -1.533133, 1.780219, -1.659123, 0.336591])
    vD = X.v_held((0.5, 0.5), X.s0_value())
    lo, hi = X.gap_enclosure(zp, vD)
    assert lo > 0 and X.placed_state(zp, vD) == (True, False) and X.branch_placed(zp, vD)


# ------------------------------------------------------------------------------------------ symmetries and copies
def test_winding_shift_keeps_loss_and_stationarity(pop, pts):
    x, y = pop
    z, v = pts["T"]
    for k in ((1, 0), (0, -1), (2, -1)):
        zk = X.shift(z, v, k)
        assert X.loss(X.qof(zk, v), x, y) == pytest.approx(X.loss(X.qof(z, v), x, y), abs=1e-12)
        assert np.abs(X.gz(zk, v, x, y)).max() < 1e-9
        assert X.windings_to(zk, z) == k
        assert X.gap_mid(zk, v) == pytest.approx(X.gap_mid(z, v), abs=1e-9)


def test_duplicate_and_copy_types(pts):
    zD, vD = pts["D"]
    zT, vT = pts["T"]
    zTp, _ = pts["Tp"]
    zDp, _ = pts["Dp"]
    assert X.is_duplicate(zD) and X.is_duplicate(zDp) and not X.is_duplicate(zT)
    assert X.is_duplicate(X.shift(zD, vD, (1, 0)))                      # units equal modulo 2π
    assert X.copy_type(zD, vD, False) == "D" and X.copy_type(zDp, vD, False) == "Dp"
    assert X.copy_type(zT, vT, False) == "T" and X.copy_type(zTp, vT, False) == "Tp"
    assert X.copy_type(zT, vT, True) == "T placed"
    assert X.copy_type(np.array([1.0, 0.0, 2.0, 0.5, 0.0]), vT, False) == "other"       # both α > 0
    assert X.copy_type(zD + np.array([1e-5, 0, 0, 0, 0]), vD, False) == "other"          # no longer duplicate


COP = {"T": np.array([-1.7, -1.45, 1.67, -1.69, 0.3]), "Tp": np.array([1.88, -1.67, -1.38, -1.68, 0.45])}
V0 = np.array([0.0222, 0.2003])


def test_classify_on_copy_with_windings_neither_and_unconverged():
    z = COP["T"]
    assert X.classify(z + 5e-4, z + 5e-7, True, COP, V0) == ("T", (0, 0))
    zk = X.shift(z, V0, (1, -1))
    assert X.classify(zk + 5e-4, zk, True, COP, V0) == ("T", (1, -1))
    assert X.classify(COP["Tp"], COP["Tp"], True, COP, V0) == ("Tp", (0, 0))
    assert X.classify(z, z, False, COP, V0) == (None, None)                 # Newton not accepted (unconverged)
    assert X.classify(z, z + 2e-6, True, COP, V0) == (None, None)           # Newton point off the copy
    assert X.classify(z + 2e-3, z, True, COP, V0) == (None, None)           # release state > 1e-3 away
    assert X.classify(z + 2e-3, z, True, COP, V0, state_tol=None) == ("T", (0, 0))
    assert X.classify(z, z, True, {"T": None, "Tp": COP["Tp"]}, V0) == (None, None)
    zbad = z.copy(); zbad[4] += 1e-3                                         # b off by a winding-inconsistent amount
    assert X.classify(zbad, zbad, True, COP, V0) == (None, None)


def test_classify_duplicate_required():
    zD = np.array([1.64, -1.71, 1.64, -1.71, 0.3])
    v = np.array([0.11, 0.11])
    assert X.classify(zD, zD, True, {"D": zD}, v, duplicate_required=("D",)) == ("D", (0, 0))
    znd = zD + np.array([9.5e-7, 0, -9.5e-7, 0, 0])                              # within 1e-6 of D but not duplicate
    assert X.classify(znd, znd, True, {"D": zD}, v) == ("D", (0, 0))
    assert X.classify(znd, znd, True, {"D": zD}, v, duplicate_required=("D",)) == (None, None)


def test_follow_classify_rule_including_d_still_duplicate():
    zT = COP["T"]
    k = (1, 0)
    zc = X.shift(zT, V0, k)                                                  # the copy's carried point at winding k
    assert X.follow_classify(zc + 5e-7, True, zc, True, V0, k, "T") == "T"
    assert X.follow_classify(zc + 5e-2, True, zc, True, V0, k, "T") is None     # Newton lands elsewhere
    assert X.follow_classify(zc, False, zc, True, V0, k, "T") is None           # run's Newton not accepted
    assert X.follow_classify(zc, True, zc, False, V0, k, "T") is None           # carried point not accepted
    assert X.follow_classify(X.shift(zT, V0, (0, 0)), True, zc, True, V0, k, "T") is None   # another winding
    zD = np.array([1.64, -1.71, 1.64, -1.71, 0.3])
    v = np.array([2.0, 2.0])
    assert X.follow_classify(zD, True, zD, True, v, (0, 0), "D") == "D"
    znd = zD + np.array([9.5e-7, 0, -9.5e-7, 0, 0])
    assert X.follow_classify(znd, True, zD, True, v, (0, 0), "D") is None       # D no longer duplicate
    assert X.follows_branch("T", "T") and not X.follows_branch("T", None) and not X.follows_branch(None, None)


# ------------------------------------------------------------------------------------------ κ
def test_split_directions_are_function_preserving_orthogonal_and_excluded(pop, pts):
    x, y = pop
    z, v = pts["D"]
    B = X.split_basis(v)
    assert np.allclose(B.T @ B, np.eye(2), atol=1e-12)
    for j in range(2):                                                       # first order: φ unchanged
        e = 1e-6 * B[:, j]
        g = [(X.gap_mid(z + e, v) - X.gap_mid(z - e, v)) / 2e-6]
        assert abs(g[0]) < 1e-6
    dGz, _, _ = X.grad_gap(z, v, X.DIAG)
    Hzz, Hzv = X.hz_blocks(z, v, x, y)
    tan = -np.linalg.solve(Hzz, Hzv @ X.DIAG)
    assert np.abs(B.T @ dGz).max() < 1e-6 and np.abs(B.T @ tan).max() < 1e-9
    lr, ls = X.reduced_lam(Hzz, v, True)
    ev = np.linalg.eigvalsh(Hzz)
    Q = np.linalg.qr(np.column_stack([B, np.eye(5)]))[0][:, 2:]
    assert lr == pytest.approx(np.linalg.eigvalsh(Q.T @ Hzz @ Q).min())
    assert ls == pytest.approx(np.linalg.eigvalsh(B.T @ Hzz @ B).min()) and ev.min() <= min(lr, ls) + 1e-12
    assert X.reduced_lam(Hzz, v, False) == (pytest.approx(ev.min()), None)


def test_kappa_formula_signed_and_invariant_under_g_sign_and_s_direction():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5)); H = A @ A.T + np.eye(5)
    tan, dG = rng.normal(size=5), rng.normal(size=5)
    k, u, den = X.kappa_formula(H, tan, dG, 0.3, 0.7)
    assert k == pytest.approx(0.7 * (dG @ np.linalg.solve(H, tan)) / (dG @ tan + 0.3))
    assert X.kappa_formula(H, tan, -dG, -0.3, 0.7)[0] == pytest.approx(k)          # G → −G
    assert X.kappa_formula(H, -tan, dG, -0.3, 0.7)[0] == pytest.approx(k)          # s → −s (v′ → −v′)
    assert np.sign(X.kappa_formula(H, tan, dG, 0.0, 0.7)[0]) != 0


def test_kappa_denominator_is_the_total_derivative_along_the_path(pop, pts):
    """dG/ds = ∇_zG·θ*′ + ∂_vG·v′ equals the finite difference of the branch gap G(z*(v(s)), v(s)) along v′."""
    x, y = pop
    z, v = pts["T"]
    vp = np.array([0.4, 0.6])
    k = X.kappa_at(z, v, vp, x, y, False)
    h = 1e-5
    zp = X.newton(z + X.dz_dv(z, v, x, y) @ (h * vp), v + h * vp, x, y)[0]
    zm = X.newton(z - X.dz_dv(z, v, x, y) @ (h * vp), v - h * vp, x, y)[0]
    fd = (X.gap_mid(zp, v + h * vp) - X.gap_mid(zm, v - h * vp)) / (2 * h)
    assert k["dG_ds"] == pytest.approx(fd, rel=1e-5)
    assert abs(k["dGv_along"]) > 0.1 * abs(k["dG_ds"])                     # the v-term matters at width 2 (change B)
    assert k["gradG_ok"]


def test_kappa_by_winding_is_exact_and_depends_on_the_sum_only_for_d(pop, pts):
    x, y = pop
    for c, vp in (("T", np.array([0.52, 0.48])), ("D", X.DIAG)):
        z, v = pts[c]
        dup = c == "D"
        k0 = X.kappa_at(z, v, vp, x, y, dup)
        for kk in ((1, 0), (0, 1), (2, -1), (-3, 0)):
            direct = X.kappa_at(X.shift(z, v, kk), v, vp, x, y, dup)["kappa0"]
            assert X.kappa_winding(k0["kappa0"], k0["kappa_winding_coef"], kk) == pytest.approx(direct, rel=1e-6,
                                                                                                abs=1e-9)
        a1, a2 = k0["kappa_winding_coef"]
        if dup:
            assert a1 == pytest.approx(a2) and len(k0["kappa_table_ksum"]) == 7
        else:
            assert abs(a1 - a2) > 1e-3 * max(abs(a1), abs(a2))             # T: κ depends on k₁ and k₂ separately


def test_winding_table_range():
    assert X.in_winding_table((0, 0)) and X.in_winding_table((3, 0)) and X.in_winding_table((-2, -1))
    assert X.in_winding_table((5, -5))                                    # the page's rule is on k₁ + k₂
    assert not X.in_winding_table((4, 0)) and not X.in_winding_table((-2, -2)) and not X.in_winding_table(None)


# ------------------------------------------------------------------------------------------ hold, budgets, rules
def test_w_hold_arm_hold_steps_and_budgets():
    assert X.w_hold(0.0476) == 4000 and X.w_hold(25 / 5000) == 5000 and X.w_hold(0.001) == 25_000
    assert X.w_hold(0.0) is None and X.w_hold(-1.0) is None and X.w_hold(float("nan")) is None
    W = {"T": 4000, "Tp": 9000, "D": 4500, "Dp": 6000}
    assert X.arm_hold_steps("T", W) == 4000 and X.arm_hold_steps("Tp", W) == 9000
    assert X.arm_hold_steps("D", W) == 6000 and X.arm_hold_steps("D", {"D": 4500}) == 4500
    assert X.arm_hold_steps("T", {}) == 4000
    assert X.budget("D", 1.0) == 40_000 and X.budget("D", 0.25) == 40_000
    assert X.budget("T", 2 ** -10) == 100_000 and X.budget("Tp", 2 ** -12) == 400_000


def test_on_branch_rule_per_arm():
    assert X.is_on_branch("T", "T") and not X.is_on_branch("T", "Tp") and not X.is_on_branch("T", None)
    assert X.is_on_branch("D", "D") and X.is_on_branch("D", "Dp") and not X.is_on_branch("D", "T")
    assert X.is_on_branch("Tp", "Tp") and not X.is_on_branch("Tp", "T")


def test_first_ge_closed_form_chi_path_and_window():
    s = np.array([1.0, 1.1, 1.2, 1.3, 1.5])
    assert X.first_ge(s, 1.2) == 2 and X.first_ge(s, 2.0) is None and X.first_ge(s, 0.5) == 0
    r, chi, sd = X.closed_form(s, 3, 1.25, 0.5, 0.2, 0.1)
    assert sd == pytest.approx(0.1) and chi == pytest.approx((0.1 / 1.25) / 0.02) and r == pytest.approx(0.5 * chi)
    assert all(math.isnan(t) for t in X.closed_form(s, None, 1.25, 0.5, 0.2, 0.1))
    assert all(math.isnan(t) for t in X.closed_form(s, 0, 1.25, 0.5, 0.2, 0.1))
    cp = X.chi_path(s, np.full(5, 0.5), 0.1, 4)
    assert np.allclose(cp, (np.diff(s) / s[:-1]) / 0.05)
    assert X.chi_window_max(s, 4, cp, 1.5) == pytest.approx(cp[1:4].max())          # s_t ≥ 1.2 for t = 2, 3 only
    assert X.chi_window_max(s, 4, cp, 1.5) == pytest.approx(max(cp[2], cp[3]))
    assert math.isnan(X.chi_window_max(s, 1, cp, 10.0))                              # empty window
    cp2 = cp.copy(); cp2[3] = np.nan
    assert math.isnan(X.chi_window_max(s, 4, cp2, 1.5))
    assert math.isnan(X.chi_window_max(s, None, cp, 1.5))


def test_l5_resolution_rule_uses_the_absolute_lag():
    e = X.l5_eligible([106, 105, 94, 95, 100, np.nan], [100, 100, 100, 100, np.nan, 100])
    assert list(e) == [True, False, True, False, False, False]                   # |−6| ≥ 6 in, |−5| out


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    rng = np.random.default_rng(3)
    Dv = rng.normal(-0.01, 0.005, 70)
    a, b = X.bootstrap_mean_ci(Dv), X.bootstrap_mean_ci(Dv)
    assert a == b and a[1] < a[0] < a[2] < 0
    m, lo, hi = X.bootstrap_mean_ci(np.full(10, -0.2))
    assert m == lo == hi == pytest.approx(-0.2)


# ------------------------------------------------------------------------------------------ pilot rules
def test_pilot_rule_d_inactive_active_and_stop():
    calls = []
    assert X.pilot_rule_D(lambda r: calls.append(r) or [0.03, 0.02, -0.5]) == {
        "status": "ok", "rho": 1.0, "history": [{"rho": 1.0, "n": 3, "q90": pytest.approx(0.028)}]}
    assert calls == [1.0]
    # q90 0.3 at ρ = 1 → ρ = 2^⌊log₂(1/3)⌋ = 1/4; kept if q90 there ≤ 0.1
    rule = X.pilot_rule_D(lambda r: [0.3] if r == 1.0 else [0.3 * r])
    assert rule["status"] == "ok" and rule["rho"] == 0.25
    calls = []
    rule = X.pilot_rule_D(lambda r: calls.append(r) or [0.3])             # never ≤ 0.1: three halvings, then STOP
    assert rule["status"] == "STOP" and calls == [1.0, 0.25, 0.125, 0.0625, 0.03125]
    assert X.pilot_rule_D(lambda r: [])["status"] == "STOP"
    assert X.pilot_rho_D(0.1) == 1.0 and X.pilot_rho_D(0.11) == 0.5 and X.pilot_rho_D(-1.0) == 1.0


def test_pilot_rule_t_halves_from_2_to_minus_10_then_stops():
    calls = []
    rule = X.pilot_rule_T(lambda r: calls.append(r) or [0.05, 0.1])
    assert rule["status"] == "ok" and rule["rho"] == 2 ** -10 and calls == [2 ** -10]
    rule = X.pilot_rule_T(lambda r: [0.176 * r / 2 ** -10])                 # χ ∝ ρ: 0.176 → 0.088 at 2⁻¹¹
    assert rule["status"] == "ok" and rule["rho"] == 2 ** -11
    calls = []
    rule = X.pilot_rule_T(lambda r: calls.append(r) or [0.2])
    assert rule["status"] == "STOP" and calls == [2 ** -10, 2 ** -11, 2 ** -12, 2 ** -13]
    assert X.pilot_rule_T(lambda r: [])["status"] == "STOP"
    assert X.pilot_rule_T(lambda r: [0.05, float("nan")])["status"] == "STOP"


# ------------------------------------------------------------------------------------------ gates
def test_gates_t_d_and_t_prime():
    on = np.zeros(120, bool); on[:60] = True
    hp = np.zeros(120, bool)
    assert X.gate("T", on, hp)["pass"] and X.gate("D", on, hp)["pass"]
    on[59] = False
    assert not X.gate("T", on, hp)["pass"] and not X.gate("D", on, hp)["pass"]
    on[:] = True; hp[7] = True
    assert X.gate("T", on, hp)["pass"] and X.gate("D", on, hp)["pass"]      # G > 0 in the hold: scored, flagged
    g = X.gate("Tp", on, hp)
    assert not g["pass"] and g["count_ok"] and not g["hold_ok"]              # T′: no G > 0 in the hold
    hp[:] = False
    on[:] = False; on[:108] = True
    assert X.gate("Tp", on, hp)["pass"]
    on[107] = False
    assert not X.gate("Tp", on, hp)["pass"]


# ------------------------------------------------------------------------------------------ scoring
def _case(arm="T", n=120, seed=0, ratio=1.0, noise=0.02, n_on=120, n_cross=120, lag=0.0013, s_pop=0.4816,
          t_lag=28.0, kappa=0.0486):
    """Constructed runs: own switches around s_pop; r_traj ≈ lag; obs = pred·ratio·(1 + noise); t_pred − t_sw = t_lag."""
    rng = np.random.default_rng(seed)
    s_sw = s_pop * rng.uniform(0.9, 1.1, n)
    r_traj = lag * rng.uniform(0.8, 1.2, n)
    r_cf = r_traj * 1.01
    r_obs = r_traj * ratio * (1 + noise * rng.uniform(-1, 1, n))
    on = np.zeros(n, bool); on[:n_on] = True
    crossed = np.zeros(n, bool); crossed[:n_cross] = True
    t_sw = np.full(n, 10_000.0)
    return dict(arm=arm, on_branch=on, hold_positive=np.zeros(n, bool), crossed=crossed,
                step_obs=np.where(crossed, t_sw + t_lag, np.nan), s_obs=np.where(crossed, s_sw * (1 + r_obs), np.nan),
                s_sw=s_sw, s_traj=s_sw * (1 + r_traj), t_traj=t_sw + t_lag, r_traj=r_traj, r_cf=r_cf,
                kappa=np.full(n, kappa), t_sw=t_sw, eta_lam=np.full(n, 0.0017), lag_steps=np.full(n, t_lag),
                chi_tsw=np.full(n, 0.027), chi_win_max=np.full(n, 0.05), pilot_chi_median=0.027,
                s_pop_branch=np.full(n, s_pop))


def _v(s):
    return tuple(s[k]["verdict"] for k in ("L1", "L2", "L3", "L4", "L5"))


def test_all_pass_each_arm():
    for arm in ("T", "D", "Tp"):
        s = X.score_arm(**_case(arm=arm))
        assert s["gate"]["pass"] and s["valid"] and _v(s) == ("PASS",) * 5 and s["outcome"] == "PASS", arm
        assert s["n_scored"] == 120 and s["n_L5_eligible"] == 120


def test_gate_fail_makes_everything_unresolved():
    s = X.score_arm(**_case(n_on=59))
    assert s["outcome"] == "UNRESOLVED (gate)" and _v(s) == ("UNRESOLVED",) * 5 and s["scored_index"] == []
    c = _case(arm="Tp"); c["hold_positive"] = c["hold_positive"].copy(); c["hold_positive"][3] = True
    assert X.score_arm(**c)["outcome"] == "UNRESOLVED (gate)"
    c["arm"] = "T"
    s = X.score_arm(**c)
    assert s["outcome"] == "PASS" and s["n_on_branch_hold_G_positive"] == 1 and s["n_scored"] == 120
    assert X.score_arm(**_case(arm="Tp", n_on=107))["outcome"] == "UNRESOLVED (gate)"


def test_off_branch_and_not_following_runs_are_counted_not_scored():
    c = _case(n_on=70)
    c["s_obs"] = c["s_obs"].copy(); c["s_obs"][70:] = 100.0
    s = X.score_arm(**c)
    assert s["n_scored"] == 70 and s["n_off_branch"] == 50 and s["outcome"] == "PASS"
    c = _case(); fol = np.ones(120, bool); fol[:61] = False
    s = X.score_arm(**c, follows=fol)
    assert s["n_scored"] == 59 and s["n_on_branch_not_following"] == 61 and s["outcome"] == "UNRESOLVED (validity)"


def test_each_criterion_fails_alone():
    c = _case(ratio=1.12); c["r_cf"] = c["r_traj"] * 1.12 / 1.05
    s = X.score_arm(**c)
    assert s["L1"]["verdict"] == "FAIL" and s["L2"]["verdict"] == "PASS" and s["outcome"].startswith("FAIL")
    for ratio, v in ((0.9005, "PASS"), (1.0995, "PASS"), (0.8995, "FAIL"), (1.1005, "FAIL")):
        c = _case(ratio=ratio, noise=0.0); c["r_cf"] = c["r_traj"] * ratio
        assert X.score_arm(**c)["L1"]["verdict"] == v, ratio
    c = _case(); c["r_cf"] = c["r_traj"] * 0.8
    assert X.score_arm(**c)["outcome"] == "FAIL L2"
    c = _case(noise=0.0)
    perm = np.random.default_rng(9).permutation(120)
    c["s_obs"] = c["s_sw"] * (1 + c["r_traj"][perm])
    s = X.score_arm(**c)
    assert s["L3"]["verdict"] == "FAIL" and s["L3"]["spearman"] < 0.5


def test_l4_uses_the_branch_population_switch():
    c = _case()
    s = X.score_arm(**c)
    assert s["L4"]["verdict"] == "PASS" and s["L4"]["mean_D"] < 0
    c["s_sw"] = np.full(120, 0.4816); c["s_traj"] = c["s_sw"] * 1.05
    c["r_traj"] = np.full(120, 0.05); c["r_cf"] = np.full(120, 0.05)
    c["s_obs"] = 0.4816 * (1 + 1e-5 * np.linspace(-1, 1, 120))               # observed at the population switch
    s = X.score_arm(**c)
    assert s["L4"]["verdict"] == "FAIL" and s["L4"]["mean_D"] > 0


def test_l5_fails_when_the_lag_is_not_resolved_and_passes_when_it_is():
    c = _case(noise=0.0, lag=0.005)
    c["s_obs"] = c["s_sw"] * (1 + 1e-6 * np.linspace(0.5, 1.5, 120))        # crossing at the switch itself
    s = X.score_arm(**c)
    assert s["L5"]["verdict"] == "FAIL" and s["L5"]["mean_D"] > 0
    assert X.score_arm(**_case(noise=0.01))["L5"]["verdict"] == "PASS"


def test_l5_resolution_rule_and_its_unresolved_branch():
    c = _case(); c["t_traj"] = c["t_traj"].copy()
    c["t_traj"][:60] = c["t_sw"][:60] + 5                                    # 60 of 120 below N_min: exactly half remain
    s = X.score_arm(**c)
    assert s["L5"]["n"] == 60 and s["L5"]["verdict"] == "PASS" and s["n_scored"] == 120
    c["t_traj"][60] = c["t_sw"][60] - 5                                      # 59 < half: UNRESOLVED
    s = X.score_arm(**c)
    assert s["L5"]["n"] == 59 and s["L5"]["verdict"] == "UNRESOLVED" and s["outcome"] == "UNRESOLVED"
    assert _v(s)[:4] == ("PASS",) * 4
    c["t_traj"][:] = c["t_sw"] - 6                                           # a negative lag of 6 steps is eligible
    assert X.score_arm(**c)["L5"]["n"] == 120


def test_negative_lags_t_prime_pass_l1_to_l5_with_signed_ratios_and_the_literal_v4():
    c = _case(arm="Tp", lag=-0.0040, t_lag=-48.0, kappa=-0.0564, s_pop=0.3272)
    c["r_cf"] = c["r_traj"] * 1.01
    s = X.score_arm(**c)
    assert s["L1"]["median_ratio"] == pytest.approx(1.0, abs=0.03) and s["L3"]["spearman"] > 0.9
    # GELU-T's V4 is the SIGNED median predicted lag ≥ 10 steps: a negative-κ arm fails it by construction
    assert not s["validity"]["V4_median_predicted_lag_ge_10_steps"] and s["outcome"] == "UNRESOLVED (validity)"
    assert s["n_kappa_nonpos_scored"] == 120 and s["n_kappa_pos_crossing"] == 0
    assert s["validity"]["V2_tsw_before_crossing_90pct_kappa_pos"]         # no κ > 0 run: V2 holds
    c["lag_steps"] = np.full(120, 10.0)                                     # (if V4 held, everything else passes)
    assert X.score_arm(**c)["outcome"] == "PASS"
    c["s_obs"] = c["s_sw"] * (1 - c["r_traj"])                              # observed on the wrong side
    s = X.score_arm(**c)
    assert s["L1"]["verdict"] == "FAIL" and s["L1"]["median_ratio"] < 0


def test_each_validity_condition_makes_l1_l5_unresolved():
    def check(c, key):
        s = X.score_arm(**c)
        assert not s["validity"][key] and s["outcome"] == "UNRESOLVED (validity)" and _v(s) == ("UNRESOLVED",) * 5
    check(_case(n_cross=59), "V1_min_scored_60")
    c = _case(); c["t_sw"] = c["t_sw"].copy(); c["t_sw"][:13] = c["step_obs"][:13]          # 107/120 < 0.9
    check(c, "V2_tsw_before_crossing_90pct_kappa_pos")
    c = _case(); c["t_sw"] = c["t_sw"].copy(); c["t_sw"][:12] = np.nan                     # 108/120 = 0.9: holds
    assert X.score_arm(**c)["validity"]["V2_tsw_before_crossing_90pct_kappa_pos"]
    c = _case(); c["eta_lam"] = np.where(np.arange(120) < 25, 0.6, 0.01)                   # 95/120 < 0.8
    check(c, "V3_regime_eta_lam_le_0p5_in_80pct")
    c = _case(); c["eta_lam"] = np.where(np.arange(120) < 24, 0.6, 0.01)                   # 96/120 = 0.8
    assert X.score_arm(**c)["valid"]
    c = _case(); c["lag_steps"] = np.full(120, 9.99)
    check(c, "V4_median_predicted_lag_ge_10_steps")
    c = _case(); c["r_cf"] = np.full(120, 0.1001); c["r_traj"] = c["r_cf"] / 1.01
    c["s_traj"] = c["s_sw"] * (1 + c["r_traj"]); c["s_obs"] = c["s_sw"] * (1 + c["r_cf"])
    check(c, "V5_q90_kappa_chi_le_0p1")
    c = _case(); c["pilot_chi_median"] = 0.027 / 1.301
    check(c, "V6_median_chi_within_30pct_of_pilot")
    c = _case(); c["pilot_chi_median"] = 0.027 / 1.299
    assert X.score_arm(**c)["valid"]
    c = _case(); c["pilot_chi_median"] = None
    check(c, "V6_median_chi_within_30pct_of_pilot")
    c = _case(); c["chi_win_max"] = np.where(np.arange(120) < 100, 0.1, 0.3)
    check(c, "V7_q90_max_chi_window_le_0p25")
    c = _case(); c["chi_win_max"] = np.full(120, 0.1); c["chi_win_max"][3] = np.nan
    check(c, "V7_q90_max_chi_window_le_0p25")


def test_invalid_even_if_statistics_would_fail_and_multiple_failures_named():
    s = X.score_arm(**_case(n_cross=50, ratio=1.5))
    assert _v(s) == ("UNRESOLVED",) * 5 and s["L1"]["median_ratio"] == pytest.approx(1.5, rel=0.03)
    c = _case(n_cross=70, ratio=1.3); c["r_cf"] = c["r_traj"] * 0.5
    s = X.score_arm(**c)
    assert s["n_scored"] == 70 and s["outcome"].startswith("FAIL") and "L1" in s["outcome"] and "L2" in s["outcome"]


def test_l3_with_all_predictions_tied_is_unresolved():
    c = _case(); c["r_traj"] = np.full(120, 0.0013); c["r_cf"] = c["r_traj"] * 1.01
    c["s_traj"] = c["s_sw"] * 1.0013
    s = X.score_arm(**c)
    assert s["L3"]["verdict"] == "UNRESOLVED" and s["outcome"] == "UNRESOLVED"


def test_sensitivity_exclusion_is_separate():
    c = _case(); ex = np.zeros(120, bool); ex[:70] = True
    s = X.score_arm(**c, exclude=ex)
    assert s["n_scored"] == 50 and s["outcome"] == "UNRESOLVED (validity)"
    assert X.score_arm(**c)["n_scored"] == 120


# ------------------------------------------------------------------------------------------ R4 and the own-path switch
class FakeBranch:
    """A constructed branch: z*(v_t) = zs[t], H_t = Hs[t]; lost at `lost` (ensure fails there)."""

    def __init__(self, zs, Hs, V, lost=None, gap=None):
        self.Z, self.H, self.V, self._lost = list(zs), list(Hs), np.asarray(V, float), lost
        self._gap = gap

    def ensure(self, t):
        return t < len(self.Z) and (self._lost is None or t < self._lost)

    def point(self, v, zg):
        return np.array([self._gap(v), 0, 0, 0, 0.0]), True


def test_r4_recursion_matches_hand_computation():
    n = 80
    zs = [np.array([0.01 * t, 0, 0, 0, 0.0]) for t in range(n)]           # the branch point moves by 0.01 per step
    Hs = [np.diag([0.5, 1, 1, 1, 1.0])] * n
    V = np.ones((n, 2))
    placed = lambda z, v: z[0] > 0.2                                       # noqa: E731
    eta = 0.1
    t, st, mx, rad = X.r4_recursion(FakeBranch(zs, Hs, V), np.zeros(5), eta, n - 1, placed_fn=placed)
    d, tt = 0.0, None
    for k in range(n - 1):                                                  # δ_{t+1} = (1 − 0.05)δ_t − 0.01
        if k >= 1 and 0.01 * k + d > 0.2:
            tt = k
            break
        d = (1 - eta * 0.5) * d - 0.01
    assert tt is not None and tt > 20 + 5                                  # the lag: later than the branch's 0.2 at t = 20
    assert st == "ok" and t == tt and rad == pytest.approx(0.95)
    assert X.r4_recursion(FakeBranch(zs, Hs, V), np.zeros(5), eta, 5, placed_fn=placed)[1].startswith("no predicted")
    assert X.r4_recursion(FakeBranch(zs, Hs, V, lost=10), np.zeros(5), eta, n - 1, placed_fn=placed)[1] == \
        "branch lost"
    assert X.r4_recursion(FakeBranch(zs, [np.eye(5) * 25] * n, V), np.zeros(5), eta, n - 1,
                          placed_fn=placed)[1] == "unstable"                # |1 − 0.1·25| = 1.5 > 1
    zs2 = [np.array([0.0, 0, 0, 0, 3.0 * t]) for t in range(n)]            # a runaway branch: sup|δ| > 1
    assert X.r4_recursion(FakeBranch(zs2, Hs, V), np.zeros(5), eta, n - 1, placed_fn=placed)[1] == "diverged"
    t0 = X.r4_recursion(FakeBranch(zs, Hs, V), np.array([0.5, 0, 0, 0, 0]), eta, n - 1, placed_fn=placed)[0]
    assert t0 == 1                                                          # a placed release is not a crossing at t = 0


def test_own_path_switch_first_placed_step_and_segment_root():
    V = np.array([[0.0, 0.1 * t] for t in range(20)])                      # s_t = 0.1·t
    gapf = lambda v: v[1] - 0.93                                           # noqa: E731  the branch gap: root at s = 0.93
    zs = [np.array([gapf(v), 0, 0, 0, 0.0]) for v in V]
    br = FakeBranch(zs, [np.eye(5)] * 20, V, gap=gapf)
    t, s = X.own_path_switch(br, 19, placed_fn=lambda z, v: z[0] > 0, gap_fn=lambda z, v: z[0])
    assert t == 10 and s == pytest.approx(0.93, abs=1e-11)
    assert X.own_path_switch(br, 9, placed_fn=lambda z, v: z[0] > 0, gap_fn=lambda z, v: z[0]) == (None, pytest.approx(
        float("nan"), nan_ok=True))
    br = FakeBranch(zs, [np.eye(5)] * 20, V, lost=7, gap=gapf)
    t, s = X.own_path_switch(br, 19, placed_fn=lambda z, v: z[0] > 0, gap_fn=lambda z, v: z[0])
    assert t is None and math.isnan(s)


def test_own_path_switch_on_the_diagonal_equals_the_frozen_continuation(pop, pts):
    """D on the population: a v path s_t(½, ½) through the switch; the own-path procedure and the diagonal continuation
    give the same switch (D's G depends on s only)."""
    x, y = pop
    z, v = pts["D"]
    ref = X.continue_diagonal(z, float(v.sum()), x, y, 6.0)
    assert ref["status"] == "switch" and ref["s_switch"] == pytest.approx(5.0796, abs=1e-4)
    s = np.linspace(4.9, 5.2, 301)
    V = np.column_stack([s / 2, s / 2])
    zstart = X.newton(ref["z_switch"], V[0], x, y)[0]
    br = X.OwnBranch(V, zstart, x, y, True)
    t, s_sw = X.own_path_switch(br, 300)
    assert t == X.first_ge(s, ref["s_switch"]) and s_sw == pytest.approx(ref["s_switch"], rel=1e-9)


def test_d_branch_off_the_diagonal_is_the_duplicate_point_of_the_same_s(pop, pts):
    x, y = pop
    z, v = pts["D"]
    s0 = float(v.sum())
    V = np.array([[s0 / 2 + 0.001 * t, s0 / 2 - 0.0005 * t] for t in range(40)])   # s grows, shares drift
    br = X.OwnBranch(V, z, x, y, True)
    assert br.ensure(39)
    for t in (10, 39):
        assert X.is_duplicate(br.Z[t], tol=1e-8)
        zd = X.newton(br.Z[t], np.full(2, V[t].sum() / 2), x, y)[0]
        assert np.abs(zd - br.Z[t]).max() < 1e-9
        assert br.lam_split[t] is not None and br.lam[t] > 0


def test_adiabatic_continuation_stop_and_follow_point(pop, pts):
    x, y = pop
    z, v = pts["T"]
    r = X.continue_adiabatic(z, v, x, y, 30.0, s_stop=0.3)
    assert r["status"] == "stop" and float(r["v"].sum()) == pytest.approx(0.3, rel=1e-12)
    assert np.abs(X.gz(r["z"], r["v"], x, y)).max() < 1e-8
    g = -X.grad(X.qof(r["z"], r["v"]), x, y)[X.VI]
    assert np.all(X.flow_rhs(r["z"], r["v"], x, y) == g / g.sum())


# ------------------------------------------------------------------------------------------ runs (population, tiny)
def test_hold_keeps_v_fixed_and_counts_g_and_train_applies_rho_to_v_only(pop, pts):
    x, y = pop
    zT, vT = pts["T"]
    z0 = zT + 0.01
    zr, rec = X.hold(z0, vT, 5, x, y)
    assert rec["W_hold"] == 5 and rec["hold_n_G_pos"] == 0 and not rec["hold_G_positive"]
    assert not np.allclose(zr, z0)
    zp = np.array([-1.547774, -1.533133, 1.780219, -1.659123, 0.336591])      # a placed start at equal shares
    vD = X.v_held((0.5, 0.5), X.s0_value())
    _, rec = X.hold(zp, vD, 3, x, y)
    assert rec["init_placed"] and rec["hold_first_G_pos"] == 0 and rec["hold_n_G_pos"] == 4 and rec["hold_G_positive"]
    q0 = X.qof(zr, vT)
    g = X.grad(q0, x, y)
    for rho in (1.0, 2 ** -10):
        P = X.train_path(q0, 0.03, rho, 1, x, y)
        assert np.allclose(P[1][X.ZI], q0[X.ZI] - 0.03 * g[X.ZI], atol=0, rtol=0)
        assert np.allclose(P[1][X.VI], q0[X.VI] - rho * 0.03 * g[X.VI], atol=0, rtol=0)


def test_observe_path_first_certified_placed_step(pts):
    zT, vT = pts["T"]
    zp = np.array([-1.547774, -1.533133, 1.780219, -1.659123, 0.336591])
    vD = X.v_held((0.5, 0.5), X.s0_value())
    P = np.array([X.qof(zT, vT)] * 4 + [X.qof(zp, vD)] * 2)
    assert X.observe_path(P) == (4, 0)
    assert X.observe_path(P[:4]) == (None, 0)
    assert X.observe_path(np.array([X.qof(zp, vD)] + [X.qof(zT, vT)] * 3))[0] is None   # t = 0 is not observed


def test_prediction_setup_reasons():
    fr = {"copies": {"T": {"valid": True}, "Tp": {"valid": False}}}
    rec = {"on_branch": False, "copy_at_release": None, "windings": None}
    assert X._prediction_setup("T", rec, fr)[1].startswith("no prediction: not on a scored copy")
    rec = {"on_branch": True, "copy_at_release": "Tp", "windings": [0, 0]}
    assert "no validated frozen switch" in X._prediction_setup("Tp", rec, fr)[1]
    rec = {"on_branch": True, "copy_at_release": "T", "windings": [4, 0]}
    assert "winding outside the table" in X._prediction_setup("T", rec, fr)[1]
    rec = {"on_branch": True, "copy_at_release": "T", "windings": [1, 0]}
    (copy, k, cf), why = X._prediction_setup("T", rec, fr)
    assert copy == "T" and k == (1, 0) and why is None


def test_t_prime_arm_starts_at_the_population_branch_point_and_is_classified(pop, pts):
    """run_start on the POPULATION as sample (monkeypatched), a 5-step hold: T′ releases on T′ at winding (0, 0)."""
    x, y = pop
    zTp, vT = pts["Tp"]
    land = {"s0": X.s0_value(), "copies": {"Tp": {"z_s0": zTp.tolist()}}}
    fr = {"W_arm": {"Tp": 5}, "copies": {"Tp": {"z_s0": zTp.tolist(), "point_ok": True},
                                          "T": {"z_s0": pts["T"][0].tolist(), "point_ok": True}}}
    orig = X.own_sample
    try:
        X.own_sample = lambda seed: (x, y)
        z_rel, v0, rec, _ = X.run_start("Tp", 1, fr, land)
    finally:
        X.own_sample = orig
    assert np.allclose(v0, vT) and rec["copy_at_release"] == "Tp" and rec["windings"] == [0, 0]
    assert rec["on_branch"] and rec["W_hold"] == 5 and not rec["hold_G_positive"]
    assert rec["z_init"] == zTp.tolist()


@pytest.mark.parametrize("copy,arm,n_steps", [("D", "D", 4700), ("T", "T", 11_500), ("Tp", "Tp", 4300)])
def test_end_to_end_on_the_population_reproduces_the_exploration(pop, pts, copy, arm, n_steps):
    """The whole prediction pipeline (frozen switch, κ, own-path switch, R4, closed form, follow check) on the
    POPULATION from the exact branch point (the exploratory setting of w2_explore_c: D η 0.3 ρ 1, lag ≈ 200 steps; T η 0.03
    ρ 2⁻¹⁰, ≈ +28 steps; T′ ≈ −48 steps), then every-step observation: the recursion lands within 3 steps (or 5% of the
    lag) of the crossing and the lag has the predicted sign."""
    x, y = pop
    z, v = pts[copy]
    cf = {"copy": copy, "z_s0": z.tolist(), **X.switch_of_copy(copy, z, v, x, y, X.S_MAX_POP)}
    cf["follow_point"] = X.follow_point(copy, z, v, cf["s_switch"], x, y)
    assert cf["valid"] and cf["follow_point"]["valid"]
    fr = {"copies": {copy: cf}}
    land = {"copies": {copy: {"s_pop_branch": cf["s_switch"]}}}
    rho = 1.0 if arm == "D" else 2 ** -10
    rec = {"arm": arm, "on_branch": True, "copy_at_release": copy, "windings": [0, 0]}
    P = X.train_path(X.qof(z, v), X.ETA[arm], rho, n_steps, x, y)
    r = X.predict_one(arm, 0, P, rec, fr, land, x, y)
    assert r["status"] == "ok" and r["traj_status"] == "ok" and r["follows_branch"]
    t_obs, und = X.observe_path(P)
    assert t_obs is not None and abs(t_obs - r["t_traj"]) <= max(3, 0.05 * abs(r["t_traj"] - r["t_sw"]))
    r_obs = float(np.abs(P[t_obs][X.VI]).sum()) / r["s_switch"] - 1
    assert np.sign(r_obs) == np.sign(r["r_traj"]) == np.sign(r["kappa"]) == np.sign(r["r_cf"])
    assert 0.9 <= r_obs / r["r_traj"] <= 1.1 and 0.8 <= r_obs / r["r_cf"] <= 1.2
    assert abs(r["lag_steps_traj"]) >= X.L5_NMIN and r["chi_window_max"] < 0.25
    if arm == "D":
        assert r["t_sw"] == X.first_ge(np.abs(P[:, X.VI]).sum(axis=1), cf["s_switch"])
        assert r["min_lam_split_to_tsw"] > 0
    else:
        assert r["s_switch_own_path"] == r["s_switch"] and abs(r["s_switch"] / cf["s_switch"] - 1) < 0.01
