"""§20 (Track U, step U1): each new bound on constructed cases (one that must hold, one that must fail), the
margin-vs-perturbation decision on constructed pass / fail / undecided inputs, the ε-uniform outer exclusion on a
certified and on a constructed failing case, and the committed artifacts.  The links sweep (Krawczyk over [0.66, 0.71])
is not re-run here."""

import hashlib
import json
import math
from pathlib import Path

import mpmath as mpm
import numpy as np
import pytest
from mpmath import iv, mp

from src import uniform_eps as ue
from src.limit_bnb import population, profile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "uniform_eps"


@pytest.fixture(scope="module")
def data():
    return population()


def _phi_mp(sig, eps, k=0):
    """φ_ε and its σ-derivatives at 60 digits."""
    with mp.workdps(60):
        sig, eps = mp.mpf(sig), mp.mpf(eps)
        u = mp.sqrt(eps) * sig
        if k == 0:
            return (u - (1 + eps) * mp.sin(u)) / eps ** mp.mpf(1.5)
        if k == 1:
            return (1 - (1 + eps) * mp.cos(u)) / eps
        return (1 + eps) * mp.sin(u) / mp.sqrt(eps)


def _rem(sig, eps, k):
    """φ_ε^{(k)} − (h^{(k)} + ε r^{(k)}) at 60 digits (sig, eps: floats, taken exactly)."""
    with mp.workdps(60):
        s, e = mp.mpf(sig), mp.mpf(eps)
        base = [ue.h(s) + e * ue.r(s), -1 + s ** 2 / 2 + e * ue.rp(s), s + e * (s - s ** 3 / 6)][k]
        return _phi_mp(sig, eps, k) - base


def test_lemma_U1_holds_and_halved_bound_fails():
    rng = np.random.default_rng(1)
    for _ in range(1500):
        eps = 10 ** rng.uniform(-7, math.log10(0.6))
        sig = rng.uniform(-60, 60)
        assert abs(_rem(sig, eps, 0)) <= eps ** 2 * ue.m2(abs(sig)) * (1 + 1e-9)
    # constructed fail: the leading remainder is ε²(σ⁷/5040 − σ⁵/120); at σ = 1 it is −0.00813 ε², above m2/2 = 0.00425
    eps, sig = 1e-4, 1.0
    assert abs(_rem(sig, eps, 0)) > eps ** 2 * 0.5 * ue.m2(sig)


def test_lemma_U1_derivatives_hold_and_first_order_term_is_needed():
    rng = np.random.default_rng(2)
    for _ in range(1000):
        eps = 10 ** rng.uniform(-7, math.log10(0.6))
        s = rng.uniform(-40, 40)
        assert abs(_rem(s, eps, 1)) <= eps ** 2 * ue.m2_1(abs(s)) * (1 + 1e-9)
        assert abs(_rem(s, eps, 2)) <= eps ** 2 * ue.m2_2(abs(s)) * (1 + 1e-9)
    # constructed fail: without the ε r' term the ε² bound is violated
    eps, s = 1e-3, 3.0
    with mp.workdps(60):
        assert abs(_phi_mp(s, eps, 1) - (-1 + mp.mpf(s) ** 2 / 2)) > eps ** 2 * ue.m2_1(s)


def test_lemma_U2_profiled_remainder_holds_and_omitting_L1_fails(data):
    x, y = data
    rng = np.random.default_rng(3)
    pts = [(1.6686, 1.3697)] + [(rng.uniform(0, 6), rng.uniform(-8, 8)) for _ in range(12)]
    for p, q in pts:
        for A in (0.66, 0.6854, 0.71):
            for eps in (1e-4, 1e-3, 1e-2):
                L0, L1, C = ue.point_terms([p], [q], A, x, y, eps)
                Le = ue.profiled_exact(p, q, A, eps, x, y)
                assert abs(Le - L0[0] - eps * L1[0]) <= eps ** 2 * C[0] + 1e-10 * max(1.0, abs(Le))
    # constructed fail: |L*_ε − L*_0| <= ε² C is false at the branch centre (the first-order shift is ε·0.083)
    L0, L1, C = ue.point_terms([1.6686], [1.3697], 0.6854, x, y, 1e-3)
    Le = ue.profiled_exact(1.6686, 1.3697, 0.6854, 1e-3, x, y)
    assert abs(Le - L0[0]) > (1e-3) ** 2 * C[0]
    assert abs(L1[0] - 0.0828) < 1e-3


def test_decision_pass_fail_undecided():
    assert ue.decide(1e-4, 5e-3, -2.0, 2.0, 10.0, 1e-5) == "pass"            # 1e-4 − 2e-5 − 1e-9 > 0
    assert ue.decide(1e-4, 5e-3, -2.0, 2.0, 10.0, 1e-3) == "undecided"       # lower end < 0 < upper end
    assert ue.decide(-3e-3, -1e-3, -0.1, 0.1, 10.0, 1e-3) == "fail"          # the competitor certainly wins
    assert ue.decide(0.25, 0.25, -0.5, -0.5, 0.0, 0.5) == "undecided"        # the first-order shift of the difference
    assert ue.decide(0.25, 0.25, -0.5, -0.5, 0.0, 0.75) == "fail"            # cancels the margin exactly / overturns it


def test_uniform_eps0_reproduces_artifact_and_explains_section16(data):
    x, y = data
    c = ue.limit_inputs()
    cons = json.loads((OUT / "constants.json").read_text())
    u = ue.uniform_eps0(c, x, y, cons["margin"]["m_lower"], 0.68625)
    assert u["eps0"] == cons["uniform_eps0_producer_margin"]["eps0"] == 1.5e-12
    assert abs(u["root"] - 1.9143e-12) < 1e-15
    assert u["decision_at_eps0"] == "pass"
    assert u["first_order_share_at_root"] > 0.999                     # the ε L_1 term binds, not the ε² remainder
    assert ue.decide(cons["margin"]["m_lower"], math.inf, -0.68625 * ue.r_max(u["S_K"]) - u["L1_branch_hi"],
                     math.inf, u["C_K_at_eps0"] + u["C_branch"], 3e-12) == "undecided"


def test_r_max_closed_form():
    s = np.linspace(0, 98.83, 200001)
    for S in (1.0, 3.0, 3.4641, 4.0, 4.4721, 5.157, 30.0, 98.83):
        grid = np.abs(ue.r(np.append(s[s <= S], S)))
        assert ue.r_max(S) >= grid.max() - 1e-9 * max(1, grid.max())
        assert ue.r_max(S) <= grid.max() * (1 + 1e-6) + 1e-9


def test_separation_and_favourable_bounds_hold_and_uninflated_fails(data):
    x, y = data
    A_lo, A_hi = 0.68375, 0.685
    rng = np.random.default_rng(4)
    for (pc, qc, hp, hq) in [(9.2, 10.8, 0.01, 0.01), (17.9, 32.9, 0.004, 0.004), (5.0, -12.0, 0.02, 0.02)]:
        S = (pc * x + qc)[None, :]
        d = (np.abs(x) * hp + hq)[None, :]
        for eps0 in (1e-4, 1e-3):
            sep = ue.separation_bound(S, d, x, A_lo, eps0)[0]
            fav = ue.favourable_profile(S, d, y, A_lo, A_hi, eps0)[0]
            for _ in range(5):
                p, q = pc + rng.uniform(-hp, hp), qc + rng.uniform(-hq, hq)
                A, eps = rng.uniform(A_lo, A_hi), eps0 * rng.uniform(0.01, 1)
                Le = ue.profiled_exact(p, q, A, eps, x, y)
                assert sep <= Le and fav <= Le
    # constructed fail: with no inflation (eps0 = 0) the 'bound' at a point exceeds L*_ε at ε = 1e-3
    S = (9.2 * x + 10.8)[None, :]
    fav0 = ue.favourable_profile(S, np.zeros_like(S), y, 0.684, 0.684, 0.0)[0]
    assert fav0 > ue.profiled_exact(9.2, 10.8, 0.684, 1e-3, x, y)
    assert ue.favourable_profile(S, np.zeros_like(S), y, 0.684, 0.684, 1e-3)[0] <= \
        ue.profiled_exact(9.2, 10.8, 0.684, 1e-3, x, y)


def test_isotonic_retained_bound_holds_and_full_bound_fails_at_branch(data):
    x, y = data
    order = np.argsort(x, kind="stable")
    ys = y[order].astype(int)
    assert abs(ue.iso_retained(ys, len(ys)) - 0.47738562622110961) < 1e-12        # nothing dropped: B_full
    rng = np.random.default_rng(5)
    for _ in range(8):
        p, q = rng.uniform(2, 20), rng.uniform(-30, 30)
        drop = np.abs(p * x + q) < 2 * math.sqrt(2)
        b = ue.iso_retained(ys[~drop[order]], len(ys))
        for eps in (1e-5, 1e-3):
            assert b <= ue.profiled_exact(p, q, 0.685, eps, x, y) + 1e-12
    # constructed fail: at the branch some points lie in |σ| < 2√2, and B_full exceeds its loss
    L0 = profile([1.6686], [1.3697], 0.685, x, y)[0][0]
    assert ue.iso_retained(ys, len(ys)) > L0


def test_inflated2_encloses_and_fails_beyond_eps0():
    rng = np.random.default_rng(6)
    eps0 = 4.1e-6
    ph = ue.Inflated2(eps0)
    for _ in range(300):
        s = rng.uniform(-6, 6)
        S = iv.mpf([mpm.mpf(s) - 1e-6, mpm.mpf(s) + 1e-6])
        eps = eps0 * rng.uniform(0, 1)
        if eps == 0:
            continue
        for k, f in ((0, ph.phi), (1, ph.phi1), (2, ph.phi2)):
            v = _phi_mp(s, eps, k)
            enc = f(S)
            assert mpm.mpf(enc.a) <= v <= mpm.mpf(enc.b)
    # constructed fail: at ε = 100 eps0 and |σ| = 30 the second-order enclosure at eps0 misses φ_ε
    S = iv.mpf([30, 30])
    v = _phi_mp(30.0, 100 * eps0, 0)
    enc = ph.phi(S)
    assert not (mpm.mpf(enc.a) <= v <= mpm.mpf(enc.b))


def test_outer_eps_certifies_switch_subinterval_and_fails_with_shifted_centre(data):
    x, y = data
    c = ue.limit_inputs()
    cons = json.loads((OUT / "constants.json").read_text())
    pr = c["parts"].iloc[20]
    res, _ = ue.outer_eps(float(pr.A_lo), float(pr.A_hi), c["centre"], x, y, 4.1e-6, cons["B_table"], c["B_full"],
                          record=False)
    assert res["certified"] and res["min_D_pert"] > ue.TAU
    # constructed fail: a centre 0.3 off the branch; the branch minimiser is then a competitor below it
    bad = (c["centre"][0] + 0.3, c["centre"][1])
    res, _ = ue.outer_eps(float(pr.A_lo), float(pr.A_hi), bad, x, y, 4.1e-6, cons["B_table"], c["B_full"],
                          record=False, max_rounds=3)
    assert not res["certified"]


def test_committed_artifacts():
    s = json.loads((OUT / "summary.json").read_text())
    assert s["theorem_U1_holds"] and s["eps0"] == 4.1e-6 and s["eps0_matches_rule"]
    assert s["outer_certified"] and s["switch_unique_on_J"] and s["binding_link"] == "pd"
    assert s["ring_slack"] > 0 and s["pd_slack"] > 0 and s["localisation_slack"] > 0 and s["eps_in_G2_range"]
    cons = json.loads((OUT / "constants.json").read_text())
    assert cons["lemma_U1_spot_check_max_ratio"] <= 1.0
    assert cons["switch_subintervals"] == [19, 20]
    assert abs(cons["margin"]["m_lower"] - 1.0301e-4) < 1e-8 and abs(cons["margin"]["m_upper"] - 4.7027e-3) < 1e-7
    assert abs(cons["G2_range"]["eps"] - 0.0462) < 1e-9
    for k in range(40):
        o = json.loads((OUT / f"outer_eps_4.1e-6_k{k:02d}.json").read_text())
        assert o["certified"] and o["eps0"] == 4.1e-6
        if k in (19, 20):
            f = OUT / o["leaves_file"]
            assert hashlib.sha256(f.read_bytes()).hexdigest() == o["leaves_sha256"]
            lv = np.load(f)
            assert len(lv["p"]) == o["n_leaves"] and set(np.unique(lv["reason"])) <= {1, 2, 3, 4, 5, 6}
            arr = b"".join(np.ascontiguousarray(lv[n]).tobytes() for n in ("p", "q", "hp", "hq", "reason", "D"))
            assert hashlib.sha256(arr).hexdigest() == o["leaves_array_sha256"]
