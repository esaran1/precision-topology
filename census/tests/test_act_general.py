import math

import numpy as np
import pytest
import torch

from src import act_general as ag


@pytest.fixture(scope="module")
def acts():
    return {n: ag.GAct(n) for n in ag.NAMES}


def test_derivatives_match_autograd(acts):
    t = torch.linspace(-25, 25, 20001, dtype=torch.float64).requires_grad_(True)
    for n, a in acts.items():
        f = a.torch_u(t)
        g1 = torch.autograd.grad(f.sum(), t, create_graph=True)[0]
        g2 = torch.autograd.grad(g1.sum(), t, create_graph=True)[0]
        g3 = torch.autograd.grad(g2.sum(), t)[0]
        T = t.detach().numpy()
        assert np.abs(a.d3u(T) - g3.numpy()).max() < 1e-12
        assert np.abs(a.u(T) - f.detach().numpy()).max() < 1e-12
        assert np.abs(a.du(T) - g1.detach().numpy()).max() < 1e-12
        assert np.abs(a.d2u(T) - g2.detach().numpy()).max() < 1e-12


def test_d2u_bound_pass_and_fail(acts):
    for a in acts.values():
        assert ag.bound_check(a, a.d2u_bound, n=400) == 0                    # validated bound: no violation
        naive = lambda lo, hi, a=a: np.maximum(np.abs(a.d2u(lo)), np.abs(a.d2u(hi)))   # endpoints only
        assert ag.bound_check(a, naive, n=400) > 0                           # the check catches an invalid bound


def test_gap_enclosure_placed_unplaced(acts):
    a = acts["gelu"]
    lo, hi = ag.gplus(0.8642113, -0.9036362, 1.0, a)
    assert lo > 0.037 and hi - lo < 1e-8
    assert abs(lo - ag.dense_gplus(0.8642113, -0.9036362, 1.0, a)) < 1e-4
    assert ag.classify(ag.gplus(0.8642113, -0.9036362, -1.0, a)) == "unplaced"
    assert ag.classify(ag.gplus(20.0, 0.0, 1.0, a)) == "unplaced"            # relu-like kink at x = 0


def test_flat_tail_extrema_terminate(acts):
    for a in acts.values():
        for w1, b1 in ((0.3, -9.0), (1e-9, 5.0), (30.0, 0.1), (2.0, -30.0)):
            lo, hi = ag.gplus(w1, b1, 1.0, a)
            assert hi - lo < 1e-8


def test_classify():
    assert ag.classify((0.1, 0.1)) == "placed"
    assert ag.classify((-0.1, -0.05)) == "unplaced"
    assert ag.classify((-1e-12, 1e-12)) == "undecided"
    assert ag.classify((-1e-12, 0.0)) == "unplaced"


def test_solve_check(acts):
    a = acts["silu"]
    ok, b = ag.solve_check(1.5828347, -1.6359120, 1.0, 1.0, a)
    assert ok is True
    bad, _ = ag.solve_check(1.5828347, -1.6359120, -1.0, 1.0, a)               # unplaced orientation
    assert bad is False


def test_loss_grad_fd(acts):
    x, y = ag.population()
    for a in acts.values():
        P = np.array([[1.3, -0.7], [4.0, 2.0]]); sg = np.array([1.0, -1.0])
        L, G = ag.loss_grad_w1(P, sg, 3.0, x, y, a)
        h = 1e-6
        for j in range(2):
            e = np.zeros(2); e[j] = h
            Lp, _ = ag.loss_grad_w1(P + e, sg, 3.0, x, y, a)
            Lm, _ = ag.loss_grad_w1(P - e, sg, 3.0, x, y, a)
            assert np.allclose((Lp - Lm) / (2 * h), G[:, j], atol=1e-8)


def test_fa_status_either_side_of_certified_bracket():
    act = ag.get_act("fa1.30")
    x, y = ag.population()
    assert ag.status_at(4.90, x, y, act, restarts=100, seed=3)[0]["status"] == "unplaced"
    assert ag.status_at(5.00, x, y, act, restarts=100, seed=4)[0]["status"] == "placed"


def test_criterion_verdict():
    assert ag.criterion_verdict(["unplaced", "unplaced"], 0.04) == "switch predicted"
    assert ag.criterion_verdict(["unplaced", "unplaced"], 0.0) == "undetermined"      # no placed configuration
    assert ag.criterion_verdict(["placed", "placed"], 0.04) == "no switch predicted"
    assert ag.criterion_verdict(["placed", "unplaced"], 0.04) == "undetermined"
    assert ag.criterion_verdict(["undecided", "unplaced"], 0.04) == "undetermined"


def test_first_switch():
    s = [1, 2, 3, 4, 5]
    r = ag.first_switch(s, ["unplaced", "unplaced", "placed", "placed", "placed"])
    assert r["bracket"] == (2, 3) and r["changes"] == 1 and not r["undecided"]
    r = ag.first_switch(s, ["unplaced", "placed", "unplaced", "placed", "placed"])
    assert r["bracket"] == (1, 2) and r["changes"] == 3
    assert ag.first_switch(s, ["placed"] * 5)["bracket"] is None
    assert ag.first_switch(s, ["unplaced"] * 5)["bracket"] is None
    assert ag.first_switch(s, ["unplaced", "undecided", "placed", "placed", "placed"])["undecided"]


def test_validity_checks():
    a = {"loss": 0.3, "status": "placed"}
    assert ag.ladder_ok(a, {"loss": 0.3 + 5e-10, "status": "placed"})
    assert not ag.ladder_ok(a, {"loss": 0.3 - 1e-8, "status": "placed"})
    assert not ag.ladder_ok(a, {"loss": 0.3, "status": "unplaced"})
    assert ag.independent_ok(0.3, 0.3 - 5e-10)
    assert not ag.independent_ok(0.3, 0.3 - 1e-8)
    ret = {"loss": 0.3, "status": "placed"}
    far = {"loss": 0.31, "status": "unplaced"}
    near = {"loss": 0.3 + 5e-10, "status": "unplaced"}
    assert ag.audit_ok(ret, [ret, far])[0]
    assert ag.audit_ok(ret, [ret, {"loss": 0.3, "status": "placed"}])[0]          # same-status ties are fine
    assert not ag.audit_ok(ret, [ret, near])[0]
    assert not ag.audit_ok(ret, [{"loss": 0.3 - 1e-6, "status": "undecided"}])[0]
    und = {"loss": 0.3, "status": "undecided"}
    assert not ag.audit_ok(und, [und, far])[0]


def test_near_top_status_sees_other_status_ties(acts):
    a = acts["gelu"]
    ret = {"k": 0, "w1": 0.8642113, "b1": -0.9036362, "sigma": 1.0, "loss": 0.5, "flags": []}
    tie_other = {"k": 1, "w1": 20.0, "b1": 0.0, "sigma": 1.0, "loss": 0.5 + 1e-10, "flags": []}
    far = {"k": 2, "w1": 20.0, "b1": 0.0, "sigma": 1.0, "loss": 0.6, "flags": []}
    ret["status"] = "placed"
    near = ag.near_top_status([ret, tie_other, far], ret, a)
    assert len(near) == 2
    assert not ag.audit_ok(ret, near)[0]
    assert ag.audit_ok(ret, ag.near_top_status([ret, far], ret, a))[0]
