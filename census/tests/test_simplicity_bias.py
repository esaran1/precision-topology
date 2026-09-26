"""Track T (simplicity bias): the data, the parameter rule, the gradient, the feature-usage classifier and the gate,
exercised on constructed pass and fail cases before any pilot computation (results/simplicity_bias_design.md)."""

import math

import numpy as np
import pytest

from src import simplicity_bias as sb


# ------------------------------------------------------------------ data
def test_data_is_balanced_product_grid_with_stated_error():
    X, y = sb.make_data(0.2)
    assert (y == 1).sum() == (y == 0).sum() == 400
    for c in (0, 1):
        Xc = X[y == c]
        assert len(np.unique(Xc[:, 0])) == 20 and len(np.unique(Xc[:, 1])) == 20
        assert len({tuple(r) for r in Xc}) == 400                     # product grid: x₁ ⟂ x₂ | y exactly
    assert sb.best_threshold_error(X[:, 0], y) == pytest.approx(0.10)
    assert set(np.round(X[y == 1, 0], 12)) & set(np.round(X[y == 0, 0], 12)) == set()   # no cross-class x₁ tie
    mid = np.abs(X[y == 0, 1]).max(); out = np.abs(X[y == 1, 1]).min()
    assert out - mid > 0.2 - 1e-9                                      # the slab coordinate separates every point


@pytest.mark.parametrize("p,err", [(0.1, 0.05), (0.2, 0.10), (0.3, 0.15)])
def test_linear_error_is_half_the_noise(p, err):
    X, y = sb.make_data(p)
    assert sb.best_threshold_error(X[:, 0], y) == pytest.approx(err)


# ------------------------------------------------------------------ the rule
def test_halfspace_gap_constructed():
    u = np.array([0., 1, 2, 3]); y = np.array([0., 0, 1, 1])
    assert sb.halfspace_gap(u, y) == pytest.approx(2.0)
    u = np.array([0., 1, 1, 2]); y = np.array([0., 1, 0, 1])          # tie at 1 kept on one side
    assert sb.halfspace_gap(u, y) == pytest.approx(1.0)


def test_rule_on_pilot_geometry():
    X, y = sb.make_data(0.2)
    r = sb.gap_rule(X, y, n_dir=1800)
    assert r["delta_lin"] == pytest.approx(1.6) and r["delta_slab"] == pytest.approx(1.0)
    assert r["pure_x1_attains_sup"] and sb.rule_ok(r)


def test_rule_fails_when_slab_competes():
    X, y = sb.make_data(0.6)                                           # 30% error: 2(1 − 0.6) = 0.8 < 1
    r = sb.gap_rule(X, y, n_dir=900)
    assert not sb.rule_ok(r)


def test_tanh_unit_gaps_never_exceed_the_halfspace_sup():
    """tanh = ∫ sign dF, so every unit's class-mean gap is at most Δ*; steep units approach it."""
    X, y = sb.make_data(0.2)
    d_all = sb.gap_rule(X, y, n_dir=1800)["delta_all"]
    rng = np.random.default_rng(0)
    for _ in range(300):
        w = rng.normal(0, 5, 2); c = rng.normal(0, 2)
        h = np.tanh(X @ w + c)
        assert abs(h[y == 1].mean() - h[y == 0].mean()) <= d_all + 1e-12
    h = np.tanh(2000 * X[:, 0] + 2000 * 1e-6)
    assert h[y == 1].mean() - h[y == 0].mean() == pytest.approx(1.6, abs=1e-6)


def test_first_order_slope_is_quarter_gap():
    X, y = sb.make_data(0.2)
    P = np.zeros((1, 16)); P[0, 0] = 400.0; P[0, 12] = 1.0            # one steep x₁ unit carries all of ṽ
    g = sb.phi(P, X)[0]; gap = g[y == 1].mean() - g[y == 0].mean()
    for s in (1e-3, 1e-4):
        L = sb.loss_grad_batch(P, s, X, y)[0][0]
        assert (math.log(2) - L) / s == pytest.approx(gap / 4, rel=2e-3)


# ------------------------------------------------------------------ gradient
def test_gradient_matches_finite_differences():
    X, y = sb.make_data(0.2)
    rng = np.random.default_rng(3)
    P = np.concatenate([rng.normal(0, 2, (3, 12)), rng.uniform(0.3, 1, (3, 4))], axis=1)
    for s in (0.7, 4.0):
        L, G = sb.loss_grad_batch(P, s, X, y)
        h = 1e-6
        for j in range(16):
            E = np.zeros_like(P); E[:, j] = h
            fd = (sb.loss_grad_batch(P + E, s, X, y)[0] - sb.loss_grad_batch(P - E, s, X, y)[0]) / (2 * h)
            assert np.allclose(G[:, j], fd, atol=1e-8, rtol=1e-5)


def test_simplex_parametrisation_covers_signed_outputs():
    """tanh is odd: a negative output weight equals a positive one on the negated unit."""
    X, _ = sb.make_data(0.2)
    rng = np.random.default_rng(4)
    W = rng.normal(0, 2, (4, 2)); c = rng.normal(0, 1, 4); u = np.array([0.4, -0.3, 0.2, -0.1])
    f_signed = (np.tanh(X @ W.T + c) * u).sum(axis=1)
    sg = np.sign(u)
    P = np.concatenate([(W * sg[:, None]).ravel(), c * sg, np.sqrt(np.abs(u))])
    assert np.allclose(sb.phi(P[None], X)[0], f_signed)


# ------------------------------------------------------------------ feature usage (constructed networks)
K = 3000.0


def _linear_net():
    P = np.zeros(16); P[0] = K; P[12] = 1.0                            # unit 1: tanh(K x₁)
    return P


def _combo(alpha, beta, gamma=0.0):
    """α·tanh(K x₁) + (β/2)[tanh(K(x₂ − t)) + tanh(−K(x₂ + t))] (+ γ spare on x₁), t in the upper slab gap."""
    X, y = sb.make_data(0.2)
    t = 0.5 * (np.abs(X[y == 0, 1]).max() + np.abs(X[y == 1, 1]).min())
    P = np.zeros(16)
    P[0] = K                                                           # unit 1 on x₁
    P[3] = K; P[9] = -K * t                                            # unit 2: x₂ upper boundary
    P[5] = -K; P[10] = -K * t                                          # unit 3: x₂ lower boundary
    P[6] = K                                                           # unit 4: spare on x₁
    w = np.array([alpha, beta / 2, beta / 2, gamma]); w = w / w.sum()
    P[12:] = np.sqrt(w)
    return P


def test_feature_usage_pure_features():
    X, y = sb.make_data(0.2)
    f = sb.feature_usage(_linear_net(), X)
    assert f["feature"] == "linear" and f["rho2"] < 1e-9
    f = sb.feature_usage(_combo(0.0, 1.0), X)
    assert f["feature"] == "slab" and f["rho2"] > 1 - 1e-9
    assert sb.gplus(_combo(0.0, 1.0)[None], X, y)[0] > 0 and sb.gplus(_linear_net()[None], X, y)[0] < 0


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3, 0.32, 0.35, 0.4, 0.6, 0.9])
def test_feature_share_and_separation_agree_on_hard_mixtures(alpha):
    """For hard units, α (linear) + β = 1 − α (slab): V₁ = α, V₂ = β/2, and G₊ = (β − 2α)/2; so G₊ > 0 exactly when the
    slab is dominant (ρ₂ > ½).  The design's classifier threshold ½ is this identity."""
    X, y = sb.make_data(0.2)
    P = _combo(alpha, 1 - alpha)
    f = sb.feature_usage(P, X)
    assert f["V1"] == pytest.approx(alpha, abs=1e-6) and f["V2"] == pytest.approx((1 - alpha) / 2, abs=1e-6)
    g = sb.gplus(P[None], X, y)[0]
    assert g == pytest.approx((1 - alpha - 2 * alpha) / 2, abs=1e-6)
    assert (g > 0) == (f["feature"] == "slab")


def test_classify_feature():
    assert sb.classify_feature(0.2) == "linear" and sb.classify_feature(0.8) == "slab"
    assert sb.classify_feature(0.5) == "tie" and sb.classify_feature(float("nan")) == "none"


# ------------------------------------------------------------------ search bookkeeping
def test_flags_and_audit():
    assert sb.flags(1e-9, 0.3, 0.3) == []
    assert sb.flags(1e-3, 0.3, 0.3 + 1e-12) == ["info:stalled_tail"]
    assert sb.flags(1e-3, 0.3, 0.31) == ["not_converged"]
    assert sb.flags(float("nan"), 0.3, 0.3) == ["nonfinite"]
    y = np.array([0., 1.])
    c = [{"k": 0, "p": np.zeros(16), "loss": 0.40, "flags": []},
         {"k": 1, "p": np.zeros(16), "loss": 0.30, "flags": ["not_converged"]},
         {"k": 2, "p": np.zeros(16), "loss": 0.45, "flags": ["info:stalled_tail"]}]
    r = sb.retain(c, y)
    assert r["loss"] == 0.40
    assert not sb.audit(c, r)["ok"]
    c[1]["loss"] = 0.5
    assert sb.audit(c, sb.retain(c, y))["ok"]
    assert sb.retain([{"k": 0, "p": None, "loss": 9.0, "flags": []}], y)["flags"] == ["constant_predictor"]


def test_search_runs_and_retains_the_minimum_on_a_toy_problem():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, (60, 2)); y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(float)
    cands = sb.search(1.0, X, y, 12, "A", 1, maxit=400, batch=6)
    r = sb.retain(cands, y)
    ok = [c["loss"] for c in cands if not set(c["flags"]) & sb.EXCLUDING]
    assert r["loss"] == min(ok + [sb.constant_loss(y)])
    assert r["loss"] < sb.constant_loss(y)


# ------------------------------------------------------------------ the gate (constructed pass and fail cases)
def _set(switch, n=12, flip=None, bad_feature=None, invalid=None, alternate=False):
    s = np.round(0.5 * 1.2 ** np.arange(n), 6)
    pts = []
    for i, si in enumerate(s):
        pos = si > switch
        if alternate and i == n - 2:
            pos = False
        g = 0.1 if pos else -0.1
        feat = "slab" if pos else "linear"
        if bad_feature == i:
            feat = "slab" if feat == "linear" else "linear"
        pts.append({"s": float(si), "gplus": g, "feature": feat, "validated": invalid != i})
    return {"points": pts, "switch": switch}


def test_gate_passes_clean_case():
    v = sb.gate({"A": _set(2.0), "B": _set(2.03)})
    assert v["pass"] and v["G1"] and v["G2"] and v["G3"] and v["G4"]


def test_gate_fails_on_alternation():
    v = sb.gate({"A": _set(2.0, alternate=True), "B": _set(2.0)})
    assert not v["pass"] and not v["G1"] and v["per_set"]["A"]["sign_changes"] == 3


def test_gate_fails_on_feature_mismatch():
    v = sb.gate({"A": _set(2.0, bad_feature=1), "B": _set(2.0)})
    assert not v["pass"] and v["G1"] and not v["G2"]
    v = sb.gate({"A": _set(2.0, bad_feature=10), "B": _set(2.0)})          # linear dominant above the switch
    assert not v["pass"] and not v["G2"]


def test_gate_fails_on_disagreement():
    v = sb.gate({"A": _set(2.0), "B": _set(2.05)})
    assert not v["pass"] and not v["G3"] and v["agreement_rel"] == pytest.approx(0.025)


def test_gate_fails_on_validation_or_missing_switch():
    assert not sb.gate({"A": _set(2.0, invalid=3), "B": _set(2.0)})["pass"]
    assert not sb.gate({"A": _set(100.0), "B": _set(2.0)})["pass"]         # no switch inside the scanned range
    v = sb.gate({"A": _set(0.1), "B": _set(0.1)})                          # positive already at the first scale
    assert not v["pass"] and not v["G1"]


def test_switch_from_sequence():
    r = sb.switch_from_sequence([1, 2, 3, 4], [-1, -1, 1, 1])
    assert r["sign_changes"] == 1 and r["bracket"] == (2.0, 3.0)
    r = sb.switch_from_sequence([4, 1, 3, 2], [1, -1, 1, -1])              # order-independent
    assert r["bracket"] == (2.0, 3.0)


# ------------------------------------------------------------------ diagnostics added after the first pilot points
def test_hedge_closed_form_matches_network_loss():
    X, y = sb.make_data(0.2)
    lo = 0.5 * (X[y == 0, 0][X[y == 0, 0] < -0.1].max() + X[:, 0][(X[:, 0] > -0.1)].min())
    hi = 0.5 * (X[:, 0][X[:, 0] < 0.1].max() + X[y == 1, 0][X[y == 1, 0] > 0.1].min())
    P = np.zeros(16); P[0] = 3000.0; P[8] = -3000.0 * lo; P[2] = 3000.0; P[9] = -3000.0 * hi; P[12:14] = 1.0
    for s in (0.5, 2.0):
        assert sb.loss_grad_batch(P[None], s, X, y)[0][0] == pytest.approx(sb.hedge_loss(s), abs=1e-9)
    assert abs(sb.gplus(P[None], X, y)[0]) < 1e-9                       # on the separation boundary


def test_halfspace_oracle_matches_brute_force():
    rng = np.random.default_rng(8)
    X = rng.uniform(-1, 1, (40, 2)); r = rng.normal(size=40)
    val, d, tau, sg = sb._halfspace_oracle(X, r, n_dir=360)
    h = sg * np.sign(X @ d - tau)
    assert (r * h).sum() == pytest.approx(-val)
    best = 0.0
    for t in np.pi * np.arange(360) / 360:
        u = X @ np.array([math.cos(t), math.sin(t)])
        for tt in np.r_[u - 1e-9, u.max() + 1e-9]:
            best = max(best, abs((r * np.sign(u - tt)).sum()))
    assert val == pytest.approx(best)
