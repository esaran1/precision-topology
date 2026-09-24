"""Registered validity checks and stop conditions, exercised on constructed pass and fail cases, including the
file write-and-read round trips they depend on."""

import math

import numpy as np
import pandas as pd
import pytest


def _awkward(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n) * 10.0 ** rng.integers(-17, 1, n)        # values whose default parse can be off by 1 ulp


# ------------------------------------------------------------------ Block 4 horizon reproduction (amended)
def _horizon_files(tmp, G, placed, perturb=None):
    keys = pd.DataFrame({"seed": np.arange(len(G)), "ck_step": 7, "level": 0.9, "variant": "preserved"})
    b4 = keys.assign(placed_end=placed, G_end=G)
    hz = keys.assign(placed_4000=placed.copy(), G_4000=G.copy())
    if perturb:
        perturb(hz)
    b4.to_csv(tmp / "fixed_scale_block4.csv", index=False)
    hz.to_csv(tmp / "fixed_scale_horizons.csv", index=False)


def test_horizon_check_passes_on_identical_awkward_floats(tmp_path):
    from src.fixed_scale import check_horizons
    G = _awkward(3000)
    _horizon_files(tmp_path, G, G > 0)
    assert check_horizons(tmp_path)


def test_horizon_check_fails_on_one_ulp(tmp_path):
    from src.fixed_scale import check_horizons
    G = _awkward(3000)

    def bump(hz):
        hz.loc[5, "G_4000"] = np.nextafter(hz.loc[5, "G_4000"], np.inf)
    _horizon_files(tmp_path, G, G > 0, bump)
    assert not check_horizons(tmp_path)


def test_horizon_check_fails_on_placement(tmp_path):
    from src.fixed_scale import check_horizons
    G = _awkward(3000)

    def flip(hz):
        hz.loc[9, "placed_4000"] = not hz.loc[9, "placed_4000"]
    _horizon_files(tmp_path, G, G > 0, flip)
    assert not check_horizons(tmp_path)


def test_default_parser_is_not_round_trip(tmp_path):
    """The fired check's cause: the default parser does not round-trip every float."""
    G = _awkward(3000)
    pd.DataFrame({"G": G}).to_csv(tmp_path / "g.csv", index=False)
    default = pd.read_csv(tmp_path / "g.csv").G.values
    exact = pd.read_csv(tmp_path / "g.csv", float_precision="round_trip").G.values
    assert np.array_equal(exact, G)
    assert not np.array_equal(default, G)


# ------------------------------------------------------------------ sample-size test
def test_sample_size_validate400_gate(tmp_path):
    from src import sample_size as ss
    pd.DataFrame({"overlap": [True] * 45 + [False] * 5}).to_csv(tmp_path / "sample_size_validate400.csv", index=False)
    assert ss.gate_validate400(tmp_path) == 45
    pd.DataFrame({"overlap": [True] * 44 + [False] * 6}).to_csv(tmp_path / "sample_size_validate400.csv", index=False)
    with pytest.raises(SystemExit):
        ss.gate_validate400(tmp_path)


def test_sample_size_certify_gate(tmp_path):
    from src import sample_size as ss
    ok = pd.DataFrame({"cert_lo": ["minus", "unresolved"], "cert_hi": ["plus", "plus"]})
    ok.to_csv(tmp_path / "sample_size_certify.csv", index=False)
    assert ss.gate_certify(tmp_path) == 2
    for lo, hi in (("plus", "plus"), ("minus", "minus")):
        pd.DataFrame({"cert_lo": [lo], "cert_hi": [hi]}).to_csv(tmp_path / "sample_size_certify.csv", index=False)
        with pytest.raises(SystemExit):
            ss.gate_certify(tmp_path)


def test_sample_size_seed_extension():
    from src import sample_size as ss
    base = pd.DataFrame({"a": 1.3, "n": 400, "seed": range(ss.SEED0, ss.SEED0 + 50),
                         "cross_step": [np.nan] * 15 + [100.0] * 35})
    calls = []

    def run(more):                                  # every added run crosses
        calls.append(len(more))
        return [{"a": a, "n": n, "seed": s, "cross_step": 1.0} for a, n, s in more]
    d = ss.extend_seeds(base, run)
    assert calls == [20] and len(d) == 70 and d.cross_step.notna().sum() == 55
    never = base.assign(cross_step=np.nan)
    d = ss.extend_seeds(never, lambda more: [{"a": a, "n": n, "seed": s, "cross_step": np.nan} for a, n, s in more])
    assert len(d) == 110                            # capped


def _ss_files(tmp, own_excess, free_off, n_seeds=50, spread=0.05, seed=0):
    from src import sample_size as ss, own_threshold as ot
    for f in ("cond_certified_brackets.csv", "ghat_certified_all.csv"):
        (tmp / f).write_bytes((ot.Path(ot.__file__).resolve().parents[1] / "results" / f).read_bytes())
    rng = np.random.default_rng(seed)
    rows_o, rows_f = [], []
    for a in ss.A_VALUES:
        _, w2p, Rp = ss._pop(a)
        for n in ss.N_VALUES:
            for i in range(n_seeds):
                rows_o.append({"a": a, "n": n, "seed": i, "w2_own": w2p * (1 + own_excess[n] + rng.normal(0, spread))})
                rows_f.append({"a": a, "n": n, "seed": i, "cross_step": 10.0,
                               "R_cross": Rp * (1 + free_off[n] + rng.normal(0, spread))})
    pd.DataFrame(rows_o).to_csv(tmp / "sample_size_own.csv", index=False)
    pd.DataFrame(rows_f).to_csv(tmp / "sample_size_free.csv", index=False)
    pd.DataFrame({"overlap": [True] * 50}).to_csv(tmp / "sample_size_validate400.csv", index=False)
    pd.DataFrame({"cert_lo": ["minus"], "cert_hi": ["plus"]}).to_csv(tmp / "sample_size_certify.csv", index=False)


def test_sample_size_score_pass_and_fail(tmp_path, monkeypatch):
    from src import sample_size as ss, own_threshold as ot
    real = ss.RESULTS
    monkeypatch.setattr(ot, "RESULTS", tmp_path)
    try:
        _ss_files(tmp_path, {400: 0.06, 1600: 0.04, 6400: 0.02}, {400: 0.10, 1600: 0.06, 6400: 0.03}, spread=0.01)
        ss.score(tmp_path)
        t = pd.read_csv(tmp_path / "sample_size_tests.csv")
        assert t.N1_pass.all() and t.N2_pass.all() and t.N3_pass.all() and not t.competing_offset_not_finite_sample.any()
        _ss_files(tmp_path, {400: 0.06, 1600: 0.06, 6400: 0.06}, {400: 0.10, 1600: 0.10, 6400: 0.10}, spread=0.0, seed=1)
        ss.score(tmp_path)
        t = pd.read_csv(tmp_path / "sample_size_tests.csv")
        assert not t.N1_pass.any() and not t.N2_pass.any() and t.competing_offset_not_finite_sample.all()
        # indistinguishable flag: 1,600 and 6,400 both consistent with zero, ordering fails between them
        _ss_files(tmp_path, {400: 0.06, 1600: 0.0, 6400: 0.002}, {400: 0.10, 1600: 0.0, 6400: 0.002}, spread=0.05, seed=2)
        ss.score(tmp_path)
        pr = pd.read_csv(tmp_path / "sample_size_detail_pairs.csv")
        flagged = pr[(pr.n_small == 1600) & (pr.n_large == 6400) & ~pr.strictly_decreasing]
        assert len(flagged) and flagged.note.str.contains("indistinguishable").all()
    finally:
        ss.RESULTS = real


# ------------------------------------------------------------------ prospective own-seed test
def test_prospective_own_hash_gate(tmp_path):
    from src import prospective_own as po
    import hashlib
    f = tmp_path / "prospective_own_predictions.csv"
    pd.DataFrame({"x": _awkward(50)}).to_csv(f, index=False)
    (tmp_path / "prospective_own_predictions.sha256").write_text(hashlib.sha256(f.read_bytes()).hexdigest() + "\n")
    assert po.gate_hash(tmp_path)
    f.write_bytes(f.read_bytes() + b"\n")
    with pytest.raises(SystemExit):
        po.gate_hash(tmp_path)


def test_prospective_own_validation_gate(tmp_path):
    from src import prospective_own as po
    pd.DataFrame({"cert_lo": ["minus", "unresolved"], "cert_hi": ["plus", "plus"]}).to_csv(
        tmp_path / "prospective_own_validation.csv", index=False)
    assert po.gate_validation(tmp_path) == 2
    pd.DataFrame({"cert_lo": ["plus"], "cert_hi": ["plus"]}).to_csv(tmp_path / "prospective_own_validation.csv", index=False)
    with pytest.raises(SystemExit):
        po.gate_validation(tmp_path)


def test_prospective_own_comparisons_pass_and_fail():
    from src import prospective_own as po
    rng = np.random.default_rng(0)
    rows = []
    for w in range(8):
        e_own = rng.normal(0.03, 0.01, 60); e_U = e_own + 0.12        # C = λU still worse than U_own
        rows.append(pd.DataFrame({"window": w, "e_own": e_own, "e_U": e_U, "e_C": e_U - np.log(1.11487),
                                  "e_Cown": e_own - np.log(1.0311)}))
    good = po.comparisons(pd.concat(rows), 1.30, B=2000)
    assert good["P1"]["pass"] and good["P2a"]["pass"] and good["P3"]["pass"] and good["P4"]["pass"]
    bad = po.comparisons(pd.concat(rows).assign(e_own=lambda d: d.e_U + 0.05), 1.30, B=2000)
    assert not bad["P1"]["pass"] and not bad["P2a"]["pass"] and not bad["P4"]["pass"]
    res = po.comparisons(pd.concat(rows), 1.50, B=2000)                 # U_own beats C clearly: P2b fails
    assert not res["P2b"]["pass"]


# ------------------------------------------------------------------ c1 test
def test_first_order_score_verdicts(tmp_path, monkeypatch):
    from src import first_order as fo
    monkeypatch.setattr(fo, "RESULTS", tmp_path)
    (tmp_path / "first_order_c1.csv").write_bytes((fo.Path(fo.__file__).resolve().parents[1] / "results" /
                                                   "first_order_c1.csv").read_bytes())
    c = pd.read_csv(tmp_path / "first_order_c1.csv", float_precision="round_trip").iloc[0]
    eps = np.array([0.01, 0.02, 0.03, 0.04])

    def write(c1, rel_w):
        R = c.R_glob_inf_lo * (1 + c1 * eps - 0.1 * eps ** 2)
        K = c.K_bnb_lo * (1 + c.k1_lo * eps); A = c.A_star_lo * (1 + c.A1_over_A_lo * eps)
        pd.DataFrame({"a": 1 + eps, "eps": eps, "R_lo": R * (1 - rel_w / 2), "R_hi": R * (1 + rel_w / 2),
                      "K_lo": K * (1 - 1e-7), "K_hi": K * (1 + 1e-7), "A_lo": A * (1 - rel_w / 2),
                      "A_hi": A * (1 + rel_w / 2)}).to_csv(tmp_path / "first_order_finite.csv", index=False)
        fo.score()
        return pd.read_csv(tmp_path / "first_order_scores.csv").set_index("quantity").verdict
    assert write(0.2852, 2e-4).iloc[0] == "PASS"
    assert write(0.49, 2e-4).iloc[0] == "FAIL"
    assert write(0.2852, 5e-2).iloc[0].startswith("INCONCLUSIVE")


def test_first_order_bracket_stops_at_unresolved(monkeypatch, tmp_path):
    """The registered rule: an unresolved midpoint stops the bisection; the last resolved bracket is kept."""
    from src import first_order as fo, conditional_certified as cc, ghat_bnb
    monkeypatch.setattr(fo, "RESULTS", tmp_path)
    pd.DataFrame({"A_star_lo": [0.685]}).to_csv(tmp_path / "first_order_c1.csv", index=False)
    s_true = 0.685 * 1.006 / 0.04 ** 1.5
    monkeypatch.setattr(cc, "_population", lambda: (np.zeros(2), np.zeros(2)))
    monkeypatch.setattr(ghat_bnb, "certify", lambda a, target_rel: {"Ghat_lo": 1.0, "Ghat_hi": 1.0, "converged": True,
                                                                     "w1": 0.1, "b1": 3.2})

    def ev(s, a, x, y, sym):
        st = "unresolved" if abs(s - s_true) < 0.02 else ("plus" if s > s_true else "minus")
        return {"status": st, "converged": True}
    monkeypatch.setattr(cc, "evaluate", ev)
    r, rows = fo._finite_job(1.04)
    assert r["stopped_unresolved"] and r["s_lo"] < s_true < r["s_hi"]
    assert all(e["status"] != "unresolved" for e in rows[:-1])


# ------------------------------------------------------------------ distance-column rerun
def test_block4_rerun_check(tmp_path):
    from src.fixed_scale import check_rerun
    G = _awkward(200)
    a = pd.DataFrame({"seed": range(200), "ck_step": 5, "level": 0.9, "variant": "preserved", "placed_end": G > 0,
                      "G_end": G, "first_hit": np.where(G > 0, 25.0, np.nan), "decisions_preserved": True,
                      "dist_branch_end": 1.0})
    a.to_csv(tmp_path / "fixed_scale_block4.csv", index=False)
    a.assign(dist_branch_end=0.5).to_csv(tmp_path / "fixed_scale_block4_rerun.csv", index=False)
    assert check_rerun(tmp_path)
    b = a.copy(); b.loc[3, "G_end"] = np.nextafter(b.loc[3, "G_end"], np.inf)
    b.to_csv(tmp_path / "fixed_scale_block4_rerun.csv", index=False)
    with pytest.raises(SystemExit):
        check_rerun(tmp_path)


# ------------------------------------------------------------------ lag test (design)
def test_lag_reproduce_check(tmp_path):
    from src import lag_test as lt
    G = _awkward(100) + 5.0
    ref = pd.DataFrame({"a": 1.3, "n": 6400, "seed": range(100), "cross_step": np.arange(100, dtype=float), "w2_abs": G})
    ref.loc[3, ["cross_step", "w2_abs"]] = np.nan                       # a non-crosser on both sides
    ref.to_csv(tmp_path / "sample_size_free.csv", index=False)
    new = ref.drop(columns="n").assign(factor=1.0)
    new.to_csv(tmp_path / "lag_test_reproduce.csv", index=False)
    assert lt.check_reproduce(tmp_path) == 100
    bad = new.copy(); bad.loc[7, "w2_abs"] = np.nextafter(bad.loc[7, "w2_abs"], np.inf)
    bad.to_csv(tmp_path / "lag_test_reproduce.csv", index=False)
    with pytest.raises(SystemExit):
        lt.check_reproduce(tmp_path)
    bad = new.copy(); bad.loc[3, "cross_step"] = 50.0                     # crosses where the reference did not
    bad.to_csv(tmp_path / "lag_test_reproduce.csv", index=False)
    with pytest.raises(SystemExit):
        lt.check_reproduce(tmp_path)


def test_lag_w2_scaling_is_per_parameter_lr():
    """The rescaled step equals Adam with a per-parameter learning rate for w2 (hand-written reference Adam)."""
    import torch
    from src.lag_test import scale_w2_step
    torch.manual_seed(0)
    target = torch.randn(4, dtype=torch.float64)
    for factor in (0.25, 1.0, 2.0):
        th = torch.tensor([0.3, -0.2, 0.7, 0.1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([th], lr=1e-2)
        ref = th.detach().clone(); m = torch.zeros(4, dtype=torch.float64); v = torch.zeros(4, dtype=torch.float64)
        lr = torch.tensor([1e-2, 1e-2, 1e-2 * factor, 1e-2], dtype=torch.float64)
        for t in range(1, 51):
            opt.zero_grad(); ((th - target) ** 2 * torch.arange(1, 5)).sum().backward()
            w2b = float(th.detach()[2]); opt.step(); scale_w2_step(th, w2b, factor)
            g = 2 * (ref - target) * torch.arange(1, 5)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            ref = ref - lr * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8)
        assert torch.allclose(th.detach(), ref, rtol=0, atol=1e-12)
    th = torch.tensor([0.3, -0.2, 0.7, 0.1], dtype=torch.float64, requires_grad=True)
    before = th.detach().clone(); scale_w2_step(th, float(before[2]), 1.0)
    assert torch.equal(th.detach(), before)                               # factor 1: no operation at all


def test_lag_score_pass_and_fail(tmp_path, monkeypatch):
    from src import lag_test as lt, own_threshold as ot
    for f in ("cond_certified_brackets.csv", "ghat_certified_all.csv"):
        (tmp_path / f).write_bytes((ot.Path(ot.__file__).resolve().parents[1] / "results" / f).read_bytes())
    monkeypatch.setattr(ot, "RESULTS", tmp_path)
    rng = np.random.default_rng(0)

    def build(resid):
        from src.sample_size import _pop
        own, runs, free = [], [], []
        for a in lt.A_VALUES:
            G, w2p, _ = _pop(a)
            for i in range(50):
                s = lt.SEED0 + i; w_own = w2p * (1 + rng.normal(0.01, 0.02))
                own.append({"a": a, "n": lt.N, "seed": s, "w2_own": w_own})
                for fct in lt.FACTORS:
                    w2 = w_own * (1 + resid[fct] + rng.normal(0, 0.005))
                    runs.append({"a": a, "seed": s, "factor": fct, "cross_step": 100.0, "w2_abs": w2, "R_cross": w2 * G / 2,
                                 "w1": 0.8, "b1": 3.8, "w2": w2, "plateau_steps": 0})
                    if fct == 1.0:
                        free.append({"a": a, "n": lt.N, "seed": s, "cross_step": 100.0, "w2_abs": w2})
        pd.DataFrame(own).to_csv(tmp_path / "sample_size_own.csv", index=False)
        pd.DataFrame(runs).to_csv(tmp_path / "lag_test_runs.csv", index=False)
        pd.DataFrame(free).to_csv(tmp_path / "sample_size_free.csv", index=False)
    build({0.25: 0.005, 0.5: 0.015, 1.0: 0.04, 2.0: 0.07})
    lt.score(tmp_path)
    t = pd.read_csv(tmp_path / "lag_test_tests.csv")
    assert t.L1_pass.all() and not t.competing_no_dependence.any()
    assert t.L2_pass.all()                                              # 0.015/0.04 = 0.375, 0.005/0.04 = 0.125
    build({0.25: 0.04, 0.5: 0.04, 1.0: 0.04, 2.0: 0.04})
    lt.score(tmp_path)
    t = pd.read_csv(tmp_path / "lag_test_tests.csv")
    assert not t.L1_pass.any() and t.competing_no_dependence.all() and not t.L2_pass.any()
    pr = pd.read_csv(tmp_path / "lag_test_pairs.csv")
    failed = pr[~pr.strictly_increasing]
    assert len(failed) and failed.note.str.contains("indistinguishable").all()


def test_lag_branch_rule(tmp_path):
    from src import lag_test as lt
    occ = pd.DataFrame({"branch": [1] * 90 + [-1] * 10, "init_branch": [1] * 100})
    occ.to_csv(tmp_path / "mirror_occupancy.csv", index=False)
    assert lt.branch_rule(tmp_path)[0] == "init-selected branch"
    occ = pd.DataFrame({"branch": [1] * 89 + [-1] * 11, "init_branch": [1] * 100})
    occ.to_csv(tmp_path / "mirror_occupancy.csv", index=False)
    assert lt.branch_rule(tmp_path)[0] == "global"
