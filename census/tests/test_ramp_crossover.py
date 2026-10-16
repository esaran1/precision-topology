"""Ramp-limit crossover check (POST HOC, exploratory): the profiled ramp-limit loss, its 4-D Newton polish, the
restart-agreement rule, the mirror range, the bisection and the classification of registered quantities, on
constructed pass and fail cases, and the committed numbers."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import ramp_crossover as rc

ROOT = Path(__file__).resolve().parents[1]
A = 1.3
f = lambda t, a=A: t + a * np.sin(t)


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-0.8, 0.8, n), rng.uniform(1.2, 2.4, n), rng.uniform(-2, -1.2, n // 2)])
    y = np.concatenate([np.zeros(n), np.ones(n + n // 2)])
    return x, y


def test_inner_newton_symmetric_case_is_log2():
    """x = (-1, -1, 1, 1), y = (0, 1, 0, 1): no information, so kappa = b = 0 and L = log 2."""
    x, y = np.array([-1.0, -1, 1, 1]), np.array([0.0, 1, 0, 1])
    L, k, b, g = rc.inner_newton(np.zeros((1, 4)), x, y)
    assert abs(L[0] - math.log(2)) < 1e-15 and abs(k[0]) < 1e-15 and abs(b[0]) < 1e-15


def test_inner_newton_is_the_minimum():
    """Stationary, and every small perturbation of (kappa, b) increases the loss (convex problem)."""
    x, y = _toy()
    W = 0.1 * A * np.sin(1.7 * x + 0.4)[None, :]
    L, k, b, g = rc.inner_newton(W, x, y)
    assert g[0] < 1e-13
    fl = lambda kk, bb: float(np.mean(np.logaddexp(0, kk * x + W[0] + bb) - y * (kk * x + W[0] + bb)))
    for dk, db in [(1e-4, 0), (-1e-4, 0), (0, 1e-4), (0, -1e-4), (1e-4, -1e-4)]:
        assert fl(k[0] + dk, b[0] + db) > L[0]


def test_gradient_and_hessian_against_finite_differences():
    x, y = _toy(seed=1)
    p, s, h = np.array([0.3, -0.1, 1.6, 0.7]), 0.15, 1e-6
    L, g, H = rc.loss_grad_hess(p, s, x, y)
    for i in range(4):
        e = np.zeros(4); e[i] = h
        Lp, gp, _ = rc.loss_grad_hess(p + e, s, x, y)
        Lm, gm, _ = rc.loss_grad_hess(p - e, s, x, y)
        assert abs((Lp - Lm) / (2 * h) - g[i]) < 1e-8
        assert np.max(np.abs((gp - gm) / (2 * h) - H[:, i])) < 1e-7
    # fail case: the Gauss-Newton part alone (the second-derivative term of z dropped) does not match
    u = p[2] * x + p[3]; z = p[0] * x + p[1] + s * A * np.sin(u)
    q = 1 / (1 + np.exp(-z)); q = q * (1 - q)
    J = np.stack([x, np.ones_like(x), s * A * np.cos(u) * x, s * A * np.cos(u)])
    GN = (q * J) @ J.T / len(x)
    e = np.zeros(4); e[2] = h
    fd = (rc.loss_grad_hess(p + e, s, x, y)[1] - rc.loss_grad_hess(p - e, s, x, y)[1]) / (2 * h)
    assert np.max(np.abs(fd - GN[:, 2])) > 1e-6


def test_newton_polish_reaches_a_stationary_point():
    x, y = _toy(seed=2)
    s = 0.2
    L0, k0, b0, _ = rc.inner_newton(s * A * np.sin(1.5 * x + 1.0)[None, :], x, y)
    p, L, gn = rc.newton_polish([k0[0], b0[0], 1.5, 1.0], s, x, y)
    assert gn < 1e-12 and L <= L0[0] + 1e-15
    assert np.linalg.eigvalsh(rc.loss_grad_hess(p, s, x, y)[2])[0] > 0


def test_canonical_preserves_the_function():
    """alpha < 0 maps to (|alpha|, pi - beta): the same sine.  A wrong map (pi + beta) changes it."""
    x = np.linspace(-2, 2.4, 501)
    p = (0.3, 0.1, -1.7, 0.6)
    c = rc.canonical(p, 0.15)
    assert c["alpha"] > 0 and abs(c["c_bar"] - 2.0) < 1e-15
    assert np.max(np.abs(np.sin(p[2] * x + p[3]) - np.sin(c["alpha"] * x + c["beta"]))) < 1e-12
    assert np.max(np.abs(np.sin(p[2] * x + p[3]) - np.sin(1.7 * x + math.pi + 0.6))) > 0.5


def test_agreement_rule():
    r = {"loss": 0.6, "alpha": 1.686, "beta": 1.0, "c_bar": 1.5}
    assert rc.agree(r, dict(r))
    assert rc.agree(r, {**r, "beta": 1.0 + 2 * math.pi})                 # beta mod 2 pi
    assert not rc.agree(r, {**r, "loss": 0.6 + 1e-10})
    assert not rc.agree(r, {**r, "beta": 1.01})
    assert not rc.agree(r, {**r, "c_bar": 1.5 + 1e-5})


def test_mirror_range_realisation():
    """|c| <= alpha: units (alpha, beta), (-alpha, pi - beta) with weights p, q = (1 + c/alpha)/2, (1 - c/alpha)/2 give
    c x + const + a sin(alpha x + beta) exactly (Lemma K8); |c| > alpha needs p - q outside [-1, 1]."""
    x = np.linspace(-2, 2.4, 801)
    al, be = 1.686, 0.9
    for c in (-1.6, -0.3, 0.0, 0.8, 1.686):
        p_, q_ = (1 + c / al) / 2, (1 - c / al) / 2
        phi = p_ * f(al * x + be) + q_ * f(-al * x + math.pi - be)
        tgt = c * x + A * np.sin(al * x + be)
        d = phi - tgt
        assert 0 <= p_ <= 1 and 0 <= q_ <= 1 and np.ptp(d) < 1e-12
        assert rc.mirror_margin({"alpha": al, "c_bar": c}) >= 0
    for c in (2.0, -2.7):
        assert abs(c / al) > 1 and rc.mirror_margin({"alpha": al, "c_bar": c}) < 0


def test_bisection_brackets_a_known_root():
    h = lambda s: (math.log(s / 0.15), {"agree": True})
    lo, hi, ev = rc.bisect_sign(h, 0.13, 0.18, steps=20)
    assert lo <= 0.15 <= hi and hi / lo - 1 < 1e-6 and len(ev) == 22
    with pytest.raises(ValueError):
        rc.bisect_sign(lambda s: (1.0, {}), 0.13, 0.18)                    # no sign change
    with pytest.raises(ValueError):
        rc.bisect_sign(lambda s: (-math.log(s / 0.15), {}), 0.13, 0.18)    # wrong orientation


def test_classification_flags_quantities_at_or_below():
    q = [{"name": "far above", "s": 0.44}, {"name": "at the bracket end", "s": 0.16}, {"name": "below", "s": 0.04}]
    rows = rc.classify(q, 0.16)
    assert [r["above"] for r in rows] == [True, False, False]
    assert abs(rows[0]["margin_rel"] - 1.75) < 1e-12 and rows[2]["margin_rel"] < 0


def test_registered_quantities_read_from_committed_files():
    q = {r["name"]: r["s"] for r in rc.registered_quantities()}
    assert abs(q["T2-1 bracket s_lo"] - 0.43714448126110894) < 1e-15
    assert abs(q["T2-1 bracket s_hi"] - 0.4531583637600818) < 1e-15
    assert abs(q["W2-A hold scale s0 (also Phase 1C W2-A arm, Phase 3)"] - 0.22253970311779978) < 1e-15
    assert abs(q["W2-A population switch T'"] - 0.3271666370541715) < 1e-15


def test_committed_numbers():
    d = json.loads((ROOT / "results" / "ramp_crossover.json").read_text())
    L = d["ledger"]
    assert d["a"] == 1.30 and d["delta"] == 0.4 and d["label"].startswith("POST HOC")
    assert 0.1333521432163324 < L["RC s_ramp lo"] < L["RC s_ramp hi"] < 0.1778279410038923
    assert L["RC bisection rel width"] < 1e-5 and L["RC bisection all agree"] == 1 and L["RC committed all agree"] == 1
    assert L["RC max gap above"] < 1e-14 and L["RC min mirror margin above"] > 0
    assert L["RC max abs gap 0.178 to 7.5"] < 1e-15
    assert L["RC gap s=10"] < -1e-4 and L["RC min gap above"] == L["RC gap s=10"]   # retained beats the ramp at s = 10
    assert L["RC gap s=0.1"] > 1e-3 and L["RC gap s=0.1334"] > 3e-4
    assert abs(L["RC regpath gap hi*1.01"]) < 1e-12 and L["RC regpath gap lo*0.95"] > 0
    assert L["RC quantities below"] == 0 and L["RC min quantity margin"] > 0.4
    for c in d["committed"]:
        assert (c["mirror_margin"] > 0) == (c["s"] > L["RC s_ramp hi"])
