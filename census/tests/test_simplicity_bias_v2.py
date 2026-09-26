"""Track T v2: the weight-decayed objective, the attainment check, the q rule, the gate and every registered scoring
rule, on constructed pass and fail cases before the pilot (results/simplicity_bias_v2_design.md)."""

import math

import numpy as np
import pytest

from src import simplicity_bias as sb
from src import simplicity_bias_v2 as v2
from src.lag_law import kappa


def test_decayed_gradient_matches_finite_differences():
    X, y = sb.make_data(0.2)
    rng = np.random.default_rng(1)
    P = np.concatenate([rng.normal(0, 2, (2, 12)), rng.uniform(0.3, 1, (2, 4))], axis=1)
    for lam in (1e-3, 1e-2):
        L, G = v2.loss_grad(P, 3.0, X, y, lam)
        for j in range(16):
            E = np.zeros_like(P); E[:, j] = 1e-6
            fd = (v2.loss_grad(P + E, 3.0, X, y, lam)[0] - v2.loss_grad(P - E, 3.0, X, y, lam)[0]) / 2e-6
            assert np.allclose(G[:, j], fd, atol=1e-8, rtol=1e-5)


def test_a_priori_bound():
    """θ = 0 has loss log 2, so any point with ‖(W, c)‖ > B(λ) is worse than the origin."""
    X, y = sb.make_data(0.2)
    lam = 1e-3; B = v2.bound(lam)
    P = np.zeros((1, 16)); P[0, 12:] = 0.5
    assert v2.loss_grad(P, 5.0, X, y, lam)[0][0] == pytest.approx(math.log(2))
    rng = np.random.default_rng(2)
    for _ in range(50):
        Q = np.concatenate([rng.normal(size=12), rng.uniform(0.2, 1, 4)])
        Q[:12] *= 1.001 * B / np.linalg.norm(Q[:12])
        assert v2.loss_grad(Q[None], 5.0, X, y, lam)[0][0] > math.log(2)


def _c(loss, flags=(), p=None):
    return {"loss": loss, "flags": list(flags), "p": np.r_[np.ones(12) * 0.1, np.ones(4) * 0.5] if p is None else p,
            "gnorm": 1e-9}


def test_attainment_check():
    c = [_c(0.3), _c(0.3 + 5e-8), _c(0.4)]
    assert v2.attained(v2.retain(c), c, 1e-3)["ok"]
    c = [_c(0.3), _c(0.4)]                                               # single hit
    assert not v2.attained(v2.retain(c), c, 1e-3)["ok"]
    c = [_c(0.3, ["info:stalled_tail"]), _c(0.3)]                        # a stalled tail retained
    r = min(c, key=lambda t: (t["loss"], len(t["flags"]) == 0))
    assert not v2.attained(r, c, 1e-3)["ok"]
    big = np.r_[np.ones(12) * 100, np.ones(4) * 0.5]                     # outside B(λ)
    c = [_c(0.3, p=big), _c(0.3, p=big)]
    assert not v2.attained(v2.retain(c), c, 1e-3)["ok"]
    assert not v2.attained(None, [], 1e-3)["ok"]


def _rows(rhos, scales=None, name="A"):
    scales = v2.GRID if scales is None else scales
    return [{"s": s, "rho2": r, "set": name, "attained": {"ok": True}, "hess_min_eig": 1.0, "cma": {"ok": True}}
            for s, r in zip(scales, rhos)]


def test_q_rule_and_first_crossing():
    n = len(v2.GRID)
    ra = _rows(np.linspace(0.0, 0.4, n)); rb = _rows(np.linspace(0.02, 0.38, n), name="B")
    q = v2.q_rule(ra, rb)
    assert q["q"] == pytest.approx(0.5 * (0.01 + 0.39))
    fc = v2.first_crossing([r["s"] for r in ra], [r["rho2"] for r in ra], q["q"])
    assert fc["passages"] == 1 and fc["bracket"][0] < fc["bracket"][1]
    fc = v2.first_crossing([1, 2, 3, 4, 5], [0.1, 0.3, 0.1, 0.3, 0.3], 0.2)
    assert fc["bracket"] == (1.0, 2.0) and fc["passages"] == 3
    assert v2.first_crossing([1, 2], [0.3, 0.4], 0.2)["bracket"] is None


def _set_rows(s_q, name, attained=True, cma=True, hess=1.0):
    lo, hi = s_q / 1.002, s_q * 1.002
    sc = [0.5, 1.0, lo, hi, 20.0]
    rows = _rows([0.0, 0.1, 0.19, 0.21, 0.4], sc, name)
    for r in rows:
        if r["s"] in (lo, hi):
            r["attained"] = {"ok": attained}; r["cma"] = {"ok": cma}; r["hess_min_eig"] = hess
    return {"rows": rows}


def test_gate_pass_and_fail_cases():
    v = v2.gate({"A": _set_rows(5.0, "A"), "B": _set_rows(5.05, "B")}, 0.2)
    assert v["pass"] and v["agreement_rel"] == pytest.approx(0.01, abs=1e-9)
    assert not v2.gate({"A": _set_rows(5.0, "A"), "B": _set_rows(5.2, "B")}, 0.2)["pass"]          # 4% apart
    v = v2.gate({"A": _set_rows(5.0, "A", attained=False), "B": _set_rows(5.0, "B")}, 0.2)
    assert not v["pass"] and not v["A_attained"]
    v = v2.gate({"A": _set_rows(5.0, "A", hess=-1e-3), "B": _set_rows(5.0, "B")}, 0.2)
    assert not v["pass"] and not v["A_attained"]
    v = v2.gate({"A": _set_rows(5.0, "A"), "B": _set_rows(5.0, "B", cma=False)}, 0.2)
    assert not v["pass"] and not v["C_independent"]
    v = v2.gate({"A": _set_rows(5.0, "A"), "B": _set_rows(5.0, "B")}, 0.9)                        # q never reached
    assert not v["pass"] and not v["per_set"]["A"]["reached"]


# ------------------------------------------------------------------ registered scoring rules
def test_crossing_step_is_an_upward_passage():
    assert v2.crossing_step([0.5, 0.3, 0.1, 0.15, 0.25, 0.3], 0.2) == 4      # starts above: needs a dip first
    assert v2.crossing_step([0.1, 0.1, 0.2], 0.2) == 2
    assert v2.crossing_step([0.5, 0.4, 0.3], 0.2) is None
    assert v2.crossing_step([0.1, 0.15], 0.2) is None


def test_t0_is_last_upward_passage():
    s = [0.1, 0.4, 0.6, 0.4, 0.55, 1.2, 2.0]
    assert v2.t0_step(s, 1.0) == 4
    assert v2.t0_step([0.1, 0.2], 1.0) is None


def test_score_rule_pass_and_fail():
    rng = np.random.default_rng(0)
    n = 40
    sq = 5.0; pred = sq * (1 + rng.uniform(0.2, 0.4, n))
    sc = pred * np.exp(rng.normal(0, 0.03, n))                             # prediction accurate, lag large
    r = v2.score_rule(sc, pred, np.full(n, sq), n)
    assert r["valid"] and r["T1"] and r["T3"] and r["pass"]
    sc = np.full(n, sq) * np.exp(rng.normal(0, 0.03, n))                   # no lag: baseline wins
    r = v2.score_rule(sc, pred, np.full(n, sq), n)
    assert not r["T3"] and not r["pass"]
    sc = pred * np.exp(rng.normal(0, 0.5, n))                              # too noisy: T1 fails
    assert not v2.score_rule(sc, pred, np.full(n, sq), n)["T1"]
    r = v2.score_rule(pred[:29], pred[:29], np.full(29, sq), n)            # 29 crossings: invalid
    assert not r["valid"] and not r["pass"]


def test_paired_bootstrap_deterministic_and_centered():
    d = np.r_[np.full(20, -0.1), np.full(20, -0.05)]
    lo, hi = v2.paired_bootstrap(d)
    assert lo < -0.075 < hi < 0 and v2.paired_bootstrap(d) == (lo, hi)


def test_kappa_formula_reused_from_1a():
    H = 2.0 * np.eye(3); tan = np.array([1.0, 0.5, -0.2]); dG = np.array([0.3, -1.0, 0.4])
    assert kappa(H, tan, dG, np.ones(3))[0] == pytest.approx(1.0)
    assert kappa(H, tan, dG, np.full(3, 7.0))[0] == pytest.approx(1.0)       # invariant to rescaling P
