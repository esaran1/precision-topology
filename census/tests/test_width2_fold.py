import math

import numpy as np

from src import width2_fold as wf


class SaddleNode:
    """L(y; p) = y0³/3 − (c − p)·y0 + y1²/2 + 0.2·y0·y1.  Stationary: y1 = −0.2·y0, y0² − 0.04·y0 − (c − p) = 0.
    The minimum branch y0 = (0.04 + √(0.0016 + 4(c − p)))/2 exists for p < p_f = c + 0.0004 and folds there (y0 = 0.02)."""
    def __init__(self, c=0.5):
        self.c = c

    def loss(self, y, p):
        return y[0] ** 3 / 3 - (self.c - p) * y[0] + y[1] ** 2 / 2 + 0.2 * y[0] * y[1]

    def grad(self, y, p):
        return np.array([y[0] ** 2 - (self.c - p) + 0.2 * y[1], y[1] + 0.2 * y[0]])

    def hess(self, y, p):
        return np.array([[2 * y[0], 0.2], [0.2, 1.0]])

    def gp(self, y, p):
        return np.array([1.0, 0.0])

    def branch(self, p):
        y0 = (0.04 + math.sqrt(0.0016 + 4 * (self.c - p))) / 2
        return np.array([y0, -0.2 * y0])


class NoFold(SaddleNode):
    """L = y0²/2 − p·y0 + y1²/2: a minimum for every p, no fold."""
    def loss(self, y, p):
        return y[0] ** 2 / 2 - p * y[0] + y[1] ** 2 / 2

    def grad(self, y, p):
        return np.array([y[0] - p, y[1]])

    def hess(self, y, p):
        return np.eye(2)

    def gp(self, y, p):
        return np.array([-1.0, 0.0])


def test_fold_detected_and_located():
    prob = SaddleNode()
    p0 = -0.5
    r = wf.continue_branch(prob, prob.branch(p0), p0, p_max=3.0)
    assert r["status"] == "fold"
    assert abs(r["p_event"] - (prob.c + 0.0004)) < 1e-6
    assert abs(r["lam_rel_event"]) < 1e-3
    assert all(q["lam_rel"] > 0 for q in r["path"])                      # a minimum all the way to the fold


def test_fold_validation_pass_and_fail():
    prob = SaddleNode(); p0 = -0.5
    r = wf.continue_branch(prob, prob.branch(p0), p0, p_max=3.0)
    r2 = wf.continue_branch(prob, prob.branch(p0), p0, p_max=3.0, h_max=wf.H_MAX / 2)
    rng = np.random.default_rng(0)
    # the fold point with p scaled so that s_f·(1 ± 0.01) straddles it: use p directly as log s
    v = wf.validate_fold(prob, r["x"][:-1], r["p_event"], r2["p_event"], rng)
    assert v["halving_ok"] and v["eig_ok"] and v["found_beyond"] == 0 and v["exists_before"] and v["validated"]
    # fail cases: halving disagreement; a point that is not a fold (a minimum persists beyond it)
    assert not wf.validate_fold(prob, r["x"][:-1], r["p_event"], r["p_event"] + 0.01, rng)["validated"]
    nf = NoFold()
    v = wf.validate_fold(nf, np.array([0.2, 0.0]), 0.2, 0.2, rng)
    assert not v["eig_ok"] and v["found_beyond"] > 0 and not v["validated"]


def test_no_fold_and_switch_event():
    nf = NoFold()
    r = wf.continue_branch(nf, np.array([0.1, 0.0]), 0.1, p_max=1.0)
    assert r["status"] == "none"
    r = wf.continue_branch(nf, np.array([0.1, 0.0]), 0.1, p_max=1.0, event=lambda y, p: y[0] - 0.6)
    assert r["status"] == "switch" and abs(r["p_event"] - 0.6) < 1e-8
    r = wf.continue_branch(nf, np.array([0.1, 0.0]), 0.1, p_max=1.0, boundary=lambda y: y[0] > 0.5)
    assert r["status"] == "boundary"


def test_chart_classification():
    w = np.array([1.8, 1.5, -1.8, -1.5, 0.5, 0.1])
    assert wf.chart(w, np.array([1.0, -1.0]))[0] == "duplicate"           # σ₁ = o·σ₂ with o = −1: transfer flat
    assert wf.chart(w, np.array([1.0, 1.0]))[0] == "two-unit"            # copies, but the transfer is not flat
    assert wf.chart(np.array([1.8, 1.5, 0.3, 2.0, 0.9995, 0.1]), np.array([1.0, 1.0]))[:2] == ("single", [0, 1, 5])
    assert wf.chart(np.array([1.8, 1.5, 0.3, 2.0, 0.4, 0.1]), np.array([1.0, 1.0]))[0] == "two-unit"


def test_width2_problem_derivatives():
    from src.asym_register import _act, _setup, training_set
    _setup(); act = _act(); x, y = training_set(600_000)
    w = np.array([1.7, 1.4, -0.9, 4.0, 0.6, 0.05]); sg = np.array([1.0, -1.0])
    prob = wf.Width2Problem(x, y, act, w, sg, [0, 1, 2, 3, 4, 5])
    ya, p, h = w.copy(), math.log(0.8), 1e-6
    g = prob.grad(ya, p)
    fd = np.array([(prob.loss(ya + h * e, p) - prob.loss(ya - h * e, p)) / (2 * h) for e in np.eye(6)])
    assert np.abs(g - fd).max() < 1e-7
    gp_fd = (prob.grad(ya, p + h) - prob.grad(ya, p - h)) / (2 * h)
    assert np.abs(prob.gp(ya, p) - gp_fd).max() < 1e-6


def test_dlam2_ds_on_saddle_node():
    """Near the fold λ_min² is linear in p (the synthetic 'p' plays s here): check the slope estimator against a finite difference."""
    prob = SaddleNode(); p0 = -0.5
    r = wf.continue_branch(prob, prob.branch(p0), p0, p_max=3.0, h_max=0.005)
    pf = r["p_event"]
    # dlam2_ds takes s = exp(p); build an equivalent path in s = p + 10 to keep s > 0 and linear
    path = [{"p": math.log(q["p"] + 10), "lam": q["lam"]} for q in r["path"]]
    est = wf.dlam2_ds(path, pf + 10, rel=0.01)
    lam = lambda p: np.linalg.eigvalsh(prob.hess(prob.branch(p), p))[0]
    ref = (lam(pf - 0.002) ** 2 - lam(pf - 0.004) ** 2) / 0.002
    assert np.isfinite(est) and abs(est / ref - 1) < 0.2
