"""Uniqueness certificates v2 (src/certificates_v2.py): interval primitives against 50-digit arithmetic, enclosure of
the profiled bias and gradient, and the ring / outer / coverage checks on constructed pass and fail cases."""

import numpy as np
import pandas as pd
import pytest

mp = pytest.importorskip("mpmath").mp


@pytest.fixture(scope="module")
def data():
    from src.limit_bnb import population
    return population()


@pytest.fixture(scope="module")
def centre():
    from src.certificates_v2 import RESULTS
    ls = pd.read_csv(RESULTS / "limit_switch.csv", float_precision="round_trip").iloc[0]
    return float(ls.argmin_p), float(ls.argmin_q)


# ------------------------------------------------------------------ interval primitives
def test_primitives_enclose_50_digit_values():
    from src.certificates_v2 import Iv, h_iv, hp_iv, mean_iv, sigmoid_iv
    mp.dps = 50
    rng = np.random.default_rng(1)
    lo = rng.normal(0, 6, 400)
    hi = lo + np.abs(rng.normal(0, 1, 400)) * 10.0 ** rng.integers(-16, 0, 400)
    X = Iv(lo, hi)
    S, H, Hp = sigmoid_iv(X), h_iv(X), hp_iv(X)
    for i in range(400):
        for t in (mp.mpf(lo[i]), mp.mpf(hi[i]), (mp.mpf(lo[i]) + mp.mpf(hi[i])) / 2):
            assert S.lo[i] <= 1 / (1 + mp.exp(-t)) <= S.hi[i]
            assert H.lo[i] <= -t + t ** 3 / 6 <= H.hi[i]
            assert Hp.lo[i] <= -1 + t ** 2 / 2 <= Hp.hi[i]
    v = rng.normal(0, 1, (3, 801)) * 10.0 ** rng.integers(-8, 8, (3, 801))
    M = mean_iv(Iv(v), axis=1)
    for k in range(3):
        exact = mp.fsum(mp.mpf(t) for t in v[k]) / 801
        assert M.lo[k] <= exact <= M.hi[k]


def test_widening_is_not_vacuous():
    from src.certificates_v2 import Iv, sigmoid_iv
    S = sigmoid_iv(Iv(np.array([0.3])))
    assert 0 < S.hi[0] - S.lo[0] < 1e-14


# ------------------------------------------------------------------ bias and gradient enclosures
def test_b_and_gradient_enclose_float_profile_inside_boxes(data):
    from src.certificates_v2 import Iv, b_enclosure, grad_enclosure, h_iv, sigma_iv, ybar_iv
    from src.limit_bnb import profile
    x, y = data
    rng = np.random.default_rng(2)
    m = 12
    pc, qc = 1.67 + rng.uniform(-0.2, 0.2, m), 1.37 + rng.uniform(-0.2, 0.2, m)
    w = 0.004
    P, Q, A = Iv(pc - w, pc + w), Iv(qc - w, qc + w), Iv(np.full(m, 0.66), np.full(m, 0.6625))
    gp, gq, ok = grad_enclosure(P, Q, A, x, y)
    B, okb = b_enclosure(Iv(A.lo[:, None], A.hi[:, None]) * h_iv(sigma_iv(P, Q, x)), ybar_iv(y))
    assert ok.all() and okb.all()
    for i in range(m):
        for _ in range(6):
            p, q, a = pc[i] + rng.uniform(-w, w), qc[i] + rng.uniform(-w, w), rng.uniform(0.66, 0.6625)
            _, b, fp, fq = profile([p], [q], a, x, y)
            assert B.lo[i] <= b[0] <= B.hi[i]
            assert gp.lo[i] <= fp[0] <= gp.hi[i] and gq.lo[i] <= fq[0] <= gq.hi[i]


# ------------------------------------------------------------------ ring: no critical point
def test_ring_certifies_on_the_branch(data, centre):
    from src.certificates_v2 import ring_no_critical
    x, y = data
    r = ring_no_critical(0.684, 0.685, centre, x, y)
    assert r["ring_certified"] and r["ring_grad_lower"] > 0


def test_ring_fails_when_the_critical_point_is_in_the_ring(data, centre):
    """Constructed fail case: shift the centre by 0.1 in p, so the true argmin lies at |Δ|_∞ = 0.1, inside the ring."""
    from src.certificates_v2 import ring_no_critical
    x, y = data
    r = ring_no_critical(0.684, 0.685, (centre[0] + 0.1, centre[1]), x, y, max_boxes=20_000)
    assert not r["ring_certified"] and r["ring_left"] > 0


def test_ring_fails_when_the_critical_point_is_inside_only_by_the_inner_box_being_removed(data, centre):
    """With rho_in = 0 (no PD box to hand the centre to), the critical point at the centre must block certification."""
    from src.certificates_v2 import ring_no_critical
    x, y = data
    r = ring_no_critical(0.684, 0.685, centre, x, y, rho_in=0.0, max_boxes=20_000)
    assert not r["ring_certified"]


# ------------------------------------------------------------------ outer exclusion
def test_outer_certifies_on_the_branch_small_region(data, centre):
    from src.certificates_v2 import MARGIN, outer_exclusion
    x, y = data
    r = outer_exclusion(0.684, 0.685, centre, x, y, P=4.0)
    assert r["outer_certified"] and r["outer_margin"] >= MARGIN and r["localised"]


def test_outer_fails_for_a_wrong_centre(data, centre):
    """Constructed fail case: a centre 0.3 away from the argmin; the argmin lies outside its box."""
    from src.certificates_v2 import outer_exclusion
    x, y = data
    r = outer_exclusion(0.684, 0.685, (centre[0] + 0.3, centre[1]), x, y, P=4.0)
    assert not r["outer_certified"] and "outer_note" in r


def test_outer_fails_when_the_box_is_too_small_for_the_A_interval(data, centre):
    """rho = 0.001: the loss rise at the box edge is far below the branch variation, so a point outside comes within
    margin of the branch value; the check must not certify."""
    from src.certificates_v2 import outer_exclusion
    x, y = data
    r = outer_exclusion(0.66, 0.71, centre, x, y, P=4.0, rho=0.001, max_cells=200_000)
    assert not r["outer_certified"]


def test_outer_stops_at_the_cell_cap(data, centre):
    from src.certificates_v2 import outer_exclusion
    x, y = data
    r = outer_exclusion(0.684, 0.685, centre, x, y, P=4.0, max_cells=100)
    assert not r["outer_certified"] and "cell cap" in r["outer_note"]


# ------------------------------------------------------------------ coverage bookkeeping
def test_coverage_uses_the_leaves_of_the_halving_tree():
    from src.certificates_v2 import _covered, _leaves_to_split
    rows = [(0.0, 1.0, False), (1.0, 2.0, True), (0.0, 0.5, True), (0.5, 1.0, False)]
    d = pd.DataFrame(rows, columns=["A_lo", "A_hi", "certified"])
    assert not _covered(d, [(0.0, 1.0), (1.0, 2.0)])
    assert _leaves_to_split(d, 0.0, 1.0) == [(0.5, 0.75), (0.75, 1.0)]
    d2 = pd.concat([d, pd.DataFrame([(0.5, 0.75, True), (0.75, 1.0, True)], columns=d.columns)])
    assert _covered(d2, [(0.0, 1.0), (1.0, 2.0)])
    d3 = pd.concat([d, pd.DataFrame([(0.5, 0.75, True)], columns=d.columns)])      # one half missing
    assert not _covered(d3, [(0.0, 1.0), (1.0, 2.0)])


def test_job_requires_all_three_parts(monkeypatch, data, centre):
    import src.certificates_v2 as c
    monkeypatch.setattr(c, "ring_no_critical", lambda *a, **k: {"ring_certified": True})
    monkeypatch.setattr(c, "outer_exclusion", lambda *a, **k: {"outer_certified": True, "localised": False})
    assert not c._job(("t", 0.684, 0.685, centre))["certified"]
    monkeypatch.setattr(c, "outer_exclusion", lambda *a, **k: {"outer_certified": True, "localised": True})
    assert c._job(("t", 0.684, 0.685, centre))["certified"]
    monkeypatch.setattr(c, "ring_no_critical", lambda *a, **k: {"ring_certified": False})
    assert not c._job(("t", 0.684, 0.685, centre))["certified"]
