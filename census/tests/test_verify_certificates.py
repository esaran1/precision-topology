"""The independent certificate checker (src/verify_certificates.py): its primitives against dense sampling and
high precision, the exact-tiling check, and constructed cases it must reject, including one a float-only check
would accept."""

import math

import numpy as np
import pytest

flint = pytest.importorskip("flint")
from flint import arb, ctx

import src.verify_certificates as vc


@pytest.fixture(scope="module")
def obj():
    from src.conditional_certified import _population
    x, y = _population()
    return vc.Objective(x, y, 4.95, 1.30)


def test_fa_range_encloses_dense_samples():
    fa = vc.FA(1.30)
    rng = np.random.default_rng(0)
    for _ in range(200):
        lo, hi = np.sort(rng.uniform(-20, 20, 2))
        mn, mx = fa.range(arb(lo), arb(hi))
        t = np.linspace(lo, hi, 20001); f = t + 1.30 * np.sin(t)
        assert float(mn.mid()) <= f.min() + 1e-12 and float(mx.mid()) >= f.max() - 1e-12


def test_coverage_accepts_an_exact_tiling_and_rejects_gaps_and_overlaps():
    lv = np.array([0, 1, 1, 1, 1]); iw = np.array([0, 2, 2, 3, 3]); ib = np.array([0, 0, 1, 0, 1])
    ok, _ = vc.coverage_ok(lv, iw, ib, nw=2, nb=1)                # cell (0,0) + the four children of (1,0)
    assert ok
    assert not vc.coverage_ok(lv[:-1], iw[:-1], ib[:-1], 2, 1)[0]                           # a gap
    assert not vc.coverage_ok(np.r_[lv, 0], np.r_[iw, 1], np.r_[ib, 0], 2, 1)[0]            # parent + children


def test_coverage_duplicate_and_ancestor_branches_with_the_area_preserved():
    # grid 2 x 2 at level 0; tiling: (0,0,0), (0,0,1), (0,1,0), and the four children of (0,1,1)
    lv = np.array([0, 0, 0, 1, 1, 1, 1]); iw = np.array([0, 0, 1, 2, 2, 3, 3]); ib = np.array([0, 1, 0, 2, 3, 2, 3])
    assert vc.coverage_ok(lv, iw, ib, 2, 2)[0]
    # duplicate one level-1 child, drop another: area unchanged
    ok, why = vc.coverage_ok(np.r_[lv[:-1], 1], np.r_[iw[:-1], 2], np.r_[ib[:-1], 2], 2, 2)
    assert not ok and why == "duplicate leaf"
    # replace the level-0 leaf (0,0,0) by its four children and add... instead: add parent (0,1,1) of the children and
    # drop the level-0 leaf (0,0,0) (area 4 at level 1 either way): area unchanged, ancestor present
    ok, why = vc.coverage_ok(np.r_[lv[1:], 0], np.r_[iw[1:], 1], np.r_[ib[1:], 1], 2, 2)
    assert not ok and "ancestor" in why
    assert not vc.coverage_ok(np.array([-1, 0]), np.array([0, 0]), np.array([0, 0]), 1, 1)[0]    # negative level


def test_convexity_bound_at_the_centre_is_below_the_profiled_value(obj):
    """L*(c) ≥ g(b̂) − |F(b̂)|·|b* − b̂| must hold; check against the float profile at random centres."""
    from src.profiled_bnb import profile
    rng = np.random.default_rng(1)
    for _ in range(10):
        w, b = rng.uniform(-2.5, 2.5), rng.uniform(0, 2 * math.pi)
        lb = obj.lower_bound(w, b, 0.0, 0.0)
        L = profile(np.array([w]), np.array([b]), 4.95, 1.30, obj.xf, obj.yf)[0][0]
        assert float(lb.lower().mid()) <= L + 1e-13


def test_rejects_a_claim_that_float_arithmetic_accepts(obj):
    """Constructed case: at a point where the float evaluation of L(c, b̂) rounds ABOVE the true value, the claim
    'L* ≥ fl(L)' is accepted by a float-only check (the float value equals the claim) but is false; the rigorous
    check must reject it, and must accept a claim 1e−12 lower."""
    from src.profiled_bnb import profile
    rng = np.random.default_rng(2)
    for _ in range(400):
        w, b = rng.uniform(-2.5, 2.5), rng.uniform(0, 2 * math.pi)
        L, b2, _, _, _ = profile(np.array([w]), np.array([b]), 4.95, 1.30, obj.xf, obj.yf)
        fl = float(L[0])
        true = obj.upper_at(w, b, float(b2[0]))                 # ball containing L(c, b̂) ≥ L*(c)
        if arb(fl) > true:                                      # float value lies above the true L(c, b̂) ≥ L*(c)
            break
    else:
        pytest.skip("no upward-rounded point found")
    assert fl >= fl                                             # the float-only check passes (claim == float value)
    st = {"evals": 0, "max_depth": 0}
    assert not vc._leaf_ok(obj, w, b, 0.0, 0.0, "lb", fl, vc.MAX_DEPTH, st)      # rigorous: rejected
    assert vc._leaf_ok(obj, w, b, 0.0, 0.0, "lb", fl - 1e-12, vc.MAX_DEPTH, st)  # a true claim: accepted


def test_rejects_a_false_region_claim(obj):
    """A cell containing a point with G > 0 cannot be certified as outside {G > 0}."""
    st = {"evals": 0, "max_depth": 0}
    # the certified R_glob branch argmin at a = 1.30 has G ≈ 0; a nearby point on the placed side has G > 0
    assert not vc._leaf_ok(obj, 0.8337, 3.832, 1e-2, 1e-2, "out+", None, 2, st)


def test_lemma_matches_the_search_bound():
    from src.conditional_certified import _population
    from src.profiled_bnb import w_bound
    x, y = _population()
    for a, s in ((1.30, 4.95), (1.50, 2.525), (1.60, 2.0)):
        assert abs(vc.lemma_W(x, y, s, a) - w_bound(s, a, x, y)) <= 1e-12


# ------------------------------------------------------------------------------------------ Ĝ enclosure certificates
def test_gap_bounds_bracket_an_independent_float_evaluation():
    from src.exact_extrema import exact_gap
    fa = vc.FA(1.30)
    rng = np.random.default_rng(3)
    for _ in range(300):
        w, b = float(rng.uniform(0, 1.3)), float(rng.uniform(0, 2 * math.pi))
        g = exact_gap(1.30, w, b)[0]
        up = vc._oriented_gap_upper(fa, arb(w), arb(b))
        lo = vc._oriented_gap_lower(fa, w, b)
        assert float(lo.mid()) <= g + 1e-13 and float(up.mid()) >= g - 1e-13
        assert float(up.mid()) - float(lo.mid()) < 1e-9                    # tight at a point


@pytest.fixture(scope="module")
def ghat_cert(tmp_path_factory):
    import shutil
    import src.cert_export as ce
    root = tmp_path_factory.mktemp("results")
    certs = root / "certificates"
    for f in ("ghat_bnb.csv", "ghat_certified_all.csv"):
        shutil.copy(ce.CERTS.parent / f, root / f)
    old = ce.CERTS
    ce.CERTS = certs
    try:
        ce.ghat(3.0, "g3")
    finally:
        ce.CERTS = old
    return certs


def _check(certs, monkeypatch, edit=None):
    import json as _json
    if edit:
        m = _json.loads((certs / "g3.json").read_text())
        d = dict(np.load(certs / "g3.npz"))
        edit(m, d)
        (certs / "t.json").write_text(_json.dumps(m)); np.savez_compressed(certs / "t.npz", **d)
        name = "t"
    else:
        name = "g3"
    monkeypatch.setattr(vc, "CERTS", certs)
    return vc.check_ghat(name, verbose=False, workers=1)


def test_ghat_certificate_structure_and_rounding_level_discrepancies(ghat_cert, monkeypatch):
    """The real a = 3.0 certificate: exact tiling, and the float search's published endpoints miss the rigorous
    enclosure only at rounding level (the strict endpoint checks report that; they are not relaxed)."""
    r = _check(ghat_cert, monkeypatch)
    c = r["checks"]
    assert c["coverage"] and c["domain_contains_reduction"] and c["ghat_cert_below_hi"]
    assert abs(r["claim_hi_excess_over_published"]) < 1e-14 and abs(r["ghat_cert_deficit"]) < 1e-14
    assert r["rigorous_lower"] <= r["rigorous_upper"]
    assert not c["files_match_committed_hashes"]                    # no manifest in the temporary directory


def test_ghat_rejects_a_lowered_upper_claim(ghat_cert, monkeypatch):
    r = _check(ghat_cert, monkeypatch, lambda m, d: m.update(claim_hi=m["bnb_lo"]))
    assert not r["checks"]["every_leaf_below_claim_hi"]


def test_ghat_rejects_a_missing_leaf(ghat_cert, monkeypatch):
    def drop(m, d):
        for k in ("level", "iw", "ib", "reason"):
            d[k] = d[k][1:]
    assert not _check(ghat_cert, monkeypatch, drop)["checks"]["coverage"]


def test_ghat_rejects_an_overstated_cert_value(ghat_cert, monkeypatch):
    r = _check(ghat_cert, monkeypatch, lambda m, d: m.update(ghat_cert=m["ghat_cert"] + 1e-9))
    assert not r["checks"]["ghat_cert_attained"]


@pytest.fixture(scope="module")
def ghat_stream_cert(tmp_path_factory):
    import shutil
    import src.cert_export as ce
    root = tmp_path_factory.mktemp("results_s")
    certs = root / "certificates"
    for f in ("ghat_bnb.csv", "ghat_certified_all.csv"):
        shutil.copy(ce.CERTS.parent / f, root / f)
    old = ce.CERTS
    ce.CERTS = certs
    try:
        ce.ghat_stream(3.0, "s3")
        ce.ghat(3.0, "g3")
    finally:
        ce.CERTS = old
    return certs


def test_streamed_format_gives_the_same_verdicts(ghat_stream_cert, monkeypatch):
    monkeypatch.setattr(vc, "CERTS", ghat_stream_cert)
    a = vc.check_ghat("s3", verbose=False, workers=1)
    b = vc.check_ghat("g3", verbose=False, workers=1)
    assert a["checks"] == b["checks"] and a["leaves"] == b["leaves"] and a["worst_leaf_upper"] == b["worst_leaf_upper"]
    assert a["checks"]["coverage"]


def test_per_round_tiling_rejects_constructed_faults(ghat_stream_cert):
    import json as _json
    d = np.load(ghat_stream_cert / "s3.npz")
    m = _json.loads((ghat_stream_cert / "s3.json").read_text())
    R = len(m["rounds"])
    base = {k: d[k] for k in d.files}

    class Z(dict):
        @property
        def files(self):
            return list(self.keys())
    assert vc.coverage_per_round(Z(base), m["nw"], m["nb"], R)[0]
    k = [k for k in base if len(base[k]) > 3][0]
    drop = dict(base); drop[k] = base[k][1:]
    assert not vc.coverage_per_round(Z(drop), m["nw"], m["nb"], R)[0]                     # a gap
    dup = dict(base); dup[k] = np.r_[base[k], base[k][:1]]
    assert not vc.coverage_per_round(Z(dup), m["nw"], m["nb"], R)[0]                      # a duplicate
    lvl = int(k[1:3])
    if lvl + 1 < R:                                                                        # a child of a level-l leaf
        nk = f"r{lvl + 1:02d}_pruned"
        anc = dict(base); anc[nk] = np.r_[base.get(nk, np.empty((0, 2), np.int32)), 2 * base[k][:1]]
        assert not vc.coverage_per_round(Z(anc), m["nw"], m["nb"], R)[0]


# ------------------------------------------------------------------------------------------ limit-problem pieces
def test_h_range_encloses_dense_samples():
    h = vc.HAct()
    rng = np.random.default_rng(2)
    for _ in range(200):
        lo, hi = np.sort(rng.uniform(-6, 6, 2))
        mn, mx = h.range(arb(lo), arb(hi))
        t = np.linspace(lo, hi, 20001); f = -t + t ** 3 / 6
        assert float(mn.mid()) <= f.min() + 1e-12 and float(mx.mid()) >= f.max() - 1e-12


def test_pava_matches_an_independent_float_implementation():
    from src.limit_bnb import isotonic_logloss
    rng = np.random.default_rng(3)
    for _ in range(30):
        ys = (rng.random(rng.integers(5, 200)) < rng.random()).astype(int)
        assert abs(float(vc.pava_logloss(ys).mid()) - isotonic_logloss(ys)) < 1e-9


def test_localisation_bounds_match_the_search_values():
    from src.limit_bnb import R2, isotonic_logloss, monotone_bound, population
    x, y = population()
    BP, Bf = vc.localisation_bounds(x, y, 24.0)
    ys = y[np.argsort(x)]
    full = min(isotonic_logloss(ys), isotonic_logloss(ys[::-1])) / len(x)
    assert abs(float(Bf.mid()) - full) < 1e-9
    assert float(BP.mid()) <= monotone_bound(x, y, 4 * R2 / 24.0) + 1e-9        # never above the search's value


def test_rejects_a_constant_predictor_claim_that_is_false_for_limit_objective():
    # a leaf whose claimed lower bound exceeds log 2 at the constant predictor must be rejected
    from src.limit_bnb import population
    x, y = population()
    obj = vc.Objective(x, y, 0.68, None, vc.HAct())
    st = {"evals": 0, "max_depth": 0}
    assert not vc._leaf_ok(obj, 0.0, 0.0, 1e-3, 1e-3, "lb", 0.7, vc.MAX_DEPTH, st)   # L0* ≈ log 2 < 0.7 there


def test_direct_bound_is_a_lower_bound_of_the_profiled_loss():
    # the combined (mean-value, direct) cell bound never exceeds the profiled loss at points inside the cell,
    # including far out where the direct bound is the active one
    from src.limit_bnb import population, profile
    x, y = population()
    obj = vc.Objective(x, y, 0.69, None, vc.HAct())
    rng = np.random.default_rng(5)
    for pc, qc, h in ((0.5, 1.0, 0.01), (7.5, -40.0, 0.125), (20.0, -20.0, 0.125), (0.02, -9.8, 0.023)):
        lb = obj.lower_bound(pc, qc, h, h)
        assert lb is not None
        ps, qs = pc + rng.uniform(-h, h, 20), qc + rng.uniform(-h, h, 20)
        L, *_ = profile(ps, qs, 0.69, x, y)
        assert float(vc.lo_(lb).mid()) <= L.min() + 1e-9


def test_h_on_a_ball_containing_zero_is_finite_and_encloses():
    # regression: arb ** 3 returns nan on a ball containing 0 (python-flint); HAct must not
    h = vc.HAct()
    t = vc.ball(-0.03, 0.03)
    v = h.f(t)
    assert v.is_finite()
    for tt in np.linspace(-0.03, 0.03, 101):
        assert float(vc.lo_(v).mid()) <= -tt + tt ** 3 / 6 <= float(vc.hi_(v).mid())
    mn, mx = h.range(vc.ball(-0.5, -0.49), vc.ball(0.49, 0.5))     # interval endpoints given as balls
    assert mn.is_finite() and mx.is_finite()
    tt = np.linspace(-0.5, 0.5, 2001); f = -tt + tt ** 3 / 6
    assert float(mn.mid()) <= f.min() + 1e-12 and float(mx.mid()) >= f.max() - 1e-12


def test_range_is_conservative_when_a_candidate_is_not_finite():
    h = vc.HAct()
    mn, mx = h.range(arb("nan"), arb(1.0))
    assert not mn.is_finite() and not mx.is_finite()


def test_F_tails_resolves_sign_when_every_sigmoid_is_saturated():
    # 800 logits of size ~1e5 (the far-out limit-problem cells): the direct sum cannot resolve F's sign at 80 bits,
    # the tail form can, and agrees with a high-precision direct evaluation
    from src.limit_bnb import population
    x, y = population()
    obj = vc.Objective(x, y, 0.69, None, vc.HAct())
    Z = [obj.s * obj.fa.f(arb(21.625) * xi + arb(-44.709)) for xi in obj.X]
    zs = sorted(float(z.mid()) for z in Z)
    b_mid = -0.5 * (zs[399] + zs[400])                      # between the two middle logits
    for b in (b_mid - 1.0, b_mid + 1.0):
        Ft = obj._F_tails(Z, arb(b))
        assert Ft.is_finite() and (Ft < 0 or Ft > 0)       # sign decided
        old = ctx.prec
        try:
            ctx.prec = 4000                                  # direct sum at very high precision decides it too
            acc = arb(0)
            for z in Z:
                acc += 1 / (1 + (-(arb(z.mid()) + b)).exp())
            Fd = acc / len(Z) - arb(1) / 2
            assert (Fd < 0) == (Ft < 0)
        finally:
            ctx.prec = old


def test_K_bounds_enclose_float_G0_and_cells_fail_below_K():
    import numpy as np
    from src import verify_certificates as vc
    h = lambda s: -s + s ** 3 / 6
    def g0(u, v):
        xs = lambda lo, hi: np.linspace(lo, hi, 20001)
        I, O = h(u * xs(-0.8, 0.8) + v), h(np.r_[u * xs(1.2, 2) + v, u * xs(-2, -1.2) + v])
        return max(O.min() - I.max(), I.min() - O.max())
    rng = np.random.default_rng(0)
    for u, v in zip(rng.uniform(0, 8, 40), rng.uniform(-12, 12, 40)):
        g = g0(u, v)
        tol = 1e-8 * (1 + abs(g))            # the grid value is >= the true G0 by the grid spacing
        assert float(vc.g0_upper(u, v).mid()) >= g - tol and float(vc.g0_lower(u, v).mid()) <= g + tol
    uK, vK = 1.605574798583984, 1.2041816711425795          # published argmax: G0 ~ 0.57946
    i, j = int(uK / vc.K_H0), int((vK + 12) / vc.K_H0)
    assert vc._k_cells([(i, j)], 0, 0.5)[0] == [(i, j)]     # fail case: target below K cannot be proved
    assert vc._k_cells([(i, j)], 0, 50.0)[0] == []          # pass case: a loose target is proved
    ex = vc.k_domain_lemma_exact()
    assert all(ex.values())


def test_solve_margin_sign_pass_and_fail_cases():
    import json
    from src import verify_certificates as vc
    from src.conditional_certified import _population
    w, b = json.load(open("results/certificates/ghat_a1.30.json"))["ghat_cert_witness"]
    x, y = _population()
    if float(vc.Objective(x, y, 60.0, 1.3).gap_bounds(w, w, b, b)[0].mid()) <= 0:
        w, b = -w, -b                                            # the w2 > 0 orientation
    box = (w - 1e-7, w + 1e-7, b - 1e-7, b + 1e-7)
    assert vc.solve_sign_on_box(vc.Objective(x, y, 60.0, 1.3), *box) == "solves"
    assert vc.solve_sign_on_box(vc.Objective(x, y, 0.5, 1.3), *box) == "fails"
    wide = (w - 0.5, w + 0.5, b - 0.5, b + 0.5)                # a box too wide to decide: undecided, never a guess
    assert vc.solve_sign_on_box(vc.Objective(x, y, 60.0, 1.3), *wide, max_depth=1) is None


def test_pd_box_pass_and_fail_cases():
    from src import verify_certificates as vc
    box = (1.6185, 1.6285, 1.3197, 1.3297, 0.66, 0.67)
    assert vc._pd_box(box + (0.0473,))["pd"]
    assert not vc._pd_box(box + (5.0,))["pd"]                  # margin above lambda_min: must fail


def test_krawczyk_pass_and_fail_cases():
    import csv
    from src import verify_certificates as vc
    row = next(csv.DictReader((vc.CERTS.parent / "first_order_c1.csv").open()))
    xs, ys = vc._limit_data()
    fun = lambda X, J=True: vc.switch_system_arb(X, xs, ys, J)[:2] if J else vc.switch_system_arb(X, xs, ys, False)
    xt = [float(row["p_star"]), float(row["q_star"]), float(row["b_star"]), float(row["A_star_lo"])]
    assert vc.krawczyk_arb(fun, xt, [1e-9] * 4)[0]
    off = [xt[0] + 1e-4] + xt[1:]
    assert not vc.krawczyk_arb(fun, off, [1e-9] * 4)[0]      # a box that misses the zero cannot pass


def test_ring_pass_and_fail_cases():
    from src import verify_certificates as vc
    c = (1.6685791015625, 1.3697166410041663)
    ok = vc._ring_interval((0.66, 0.66125, c[0], c[1], 0.05, 0.15, 0.0125, 4))
    assert ok["certified"]
    # a "ring" that contains the minimiser (rho_in = 0) cannot be certified free of critical points
    bad = vc._ring_interval((0.684, 0.685, 1.66808, 1.36923, 0.0, 0.004, 0.004, 2))
    assert not bad["certified"]
