"""Track 3B (band task in R^d): the gap, data/init conventions, batching, lag quantities, and every registered scoring
rule and validity check on constructed pass and fail cases (results/band_rd_registration.md)."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src import band_rd as B
from src.profiled_bnb import gap as gap1

A = 1.30


# ------------------------------------------------------------------------------------------ gap
def test_gap_without_noise_equals_width1_gap_bitwise():
    rng = np.random.default_rng(0)
    w1, b1 = rng.uniform(-4, 4, 500), rng.uniform(-8, 8, 500)
    for a in (1.30, 1.50):
        g = B.gap_rd(w1, b1, np.zeros((500, 1)), np.ones(500), a)
        assert np.array_equal(g, gap1(w1, b1, a))
        g0 = B.gap_rd(w1, b1, np.zeros((500, 0)), np.ones(500), a)          # d = 1: no noise coordinates
        assert np.array_equal(g0, gap1(w1, b1, a))


def test_gap_negative_orientation_is_the_mirror():
    rng = np.random.default_rng(1)
    w1, b1, wn = rng.uniform(-4, 4, 200), rng.uniform(-8, 8, 200), rng.uniform(-0.1, 0.1, (200, 3))
    gm = B.gap_rd(w1, b1, wn, -np.ones(200), A)
    gp = B.gap_rd(-w1, -b1, -wn, np.ones(200), A)
    assert np.allclose(gm, gp, atol=1e-12, rtol=0)


def _brute(w1, b1, wn, a, n=4001):
    """Dense grid over x₁ in each window and over the noise term ξ ∈ [−N, N] (its exact range on the box)."""
    N = B.NOISE_C * np.abs(wn).sum()
    xi = np.linspace(-N, N, 201)
    f = lambda t: t + a * np.sin(t)
    xi_in = np.linspace(-0.8, 0.8, n)
    xo = np.r_[np.linspace(1.2, 2.0, n // 2), -np.linspace(1.2, 2.0, n // 2)]
    ti = (w1 * xi_in + b1)[:, None] + xi[None, :]
    to = (w1 * xo + b1)[:, None] + xi[None, :]
    return f(to).min() - f(ti).max()


def test_gap_with_noise_matches_dense_support_and_bounds_it():
    rng = np.random.default_rng(2)
    for _ in range(30):
        w1, b1 = rng.uniform(-3, 3), rng.uniform(-6, 6)
        wn = rng.uniform(-0.2, 0.2, 3)
        g, gb = float(B.gap_rd(w1, b1, wn, 1.0, A)), _brute(w1, b1, wn, A)
        assert gb >= g - 1e-12                                  # grid extremes lie inside the true extremes
        assert gb - g < 2e-3                                    # and converge to them


def test_gap_decreases_with_noise_weight_and_constructed_sign_change():
    w1, b1 = -0.82176164, 3.81952138                             # width-1 population branch at 1.1 s* (a = 1.30)
    g0 = float(B.gap_rd(w1, b1, np.array([0.0]), 1.0, A))
    assert g0 > 0                                                # placed without noise weight
    gs = [float(B.gap_rd(w1, b1, np.array([r]), 1.0, A)) for r in (0.0, 0.005, 0.01, 0.02, 0.05)]
    assert all(np.diff(gs) < 0) and gs[-1] < 0                  # widening un-places it
    # only ‖w_noise‖₁ matters (box support)
    assert float(B.gap_rd(w1, b1, np.array([0.01, -0.01, 0.0]), 1.0, A)) == float(B.gap_rd(w1, b1, np.array([0.02]), 1.0, A))


def test_torch_gap_matches_numpy():
    rng = np.random.default_rng(3)
    for d in (1, 2, 4):
        P = np.c_[rng.uniform(-4, 4, 300), rng.uniform(-8, 8, 300), rng.uniform(-6, 6, 300), rng.uniform(-3, 3, 300),
                  rng.uniform(-0.2, 0.2, (300, d - 1))]
        G, G1 = B.gap_rd_t(torch.tensor(P), A)
        gn = B.gap_rd(P[:, 0], P[:, 1], P[:, 4:], P[:, 2], A)
        g1n = B.gap_rd(P[:, 0], P[:, 1], np.zeros((300, 0)), P[:, 2], A)
        assert np.allclose(G.numpy(), gn, atol=1e-12, rtol=0) and np.allclose(G1.numpy(), g1n, atol=1e-12, rtol=0)


# ------------------------------------------------------------------------------------------ data and init
def test_data_and_init_conventions():
    from src.fold1d import make_data
    for d in (2, 4):
        X, Y = B.make_data_rd(889_500, d)
        x1, y = make_data(200, 889_500)
        assert X.shape == (400, d) and np.array_equal(X[:, 0], x1.double().numpy()) and np.array_equal(Y, y.double().numpy())
        assert np.all(np.abs(X[:, 1:]) <= 2.0)
        torch.manual_seed(889_500)
        w = torch.empty(4).uniform_(-1.0, 1.0).double()
        assert torch.equal(B.init_rd(889_500, d)[:4], w)             # width-1 init of the same seed
    X2, _ = B.make_data_rd(889_500, 2); X4, _ = B.make_data_rd(889_500, 4)
    assert np.array_equal(X2[:, 1], X4[:, 1])                        # nested noise
    assert B.make_data_rd(889_500, 1)[0].shape == (400, 1)


def test_registered_seed_range_is_fresh_and_disjoint_from_pilot():
    assert B.SEEDS == tuple(range(880_000, 880_040))
    assert not set(B.SEEDS) & set(B.PILOT_SEEDS)


# ------------------------------------------------------------------------------------------ training
def _p0(n=2, wn=0.03, d=2):
    z = [-0.82176164, 3.81952138, -17.73569729]                     # population branch at 1.1 s* (a = 1.30)
    return np.array([[z[0], z[1], 1.1 * 4.958011429378877, z[2]] + [wn] * (d - 1)] * n)


def test_batched_equals_single_seed():
    seeds = (889_600, 889_601, 889_602)
    for d in (2, 4):
        rec, _ = B.train_batch(A, d, seeds, 300, stop_when_done=False)
        for i, s in enumerate(seeds):
            ps, _ = B.train_single(A, d, s, 300)
            assert np.abs(np.array(json.loads(rec[i]["p_end"])) - ps).max() < 1e-12


def test_crossing_detection_constructed():
    rec, tr = B.train_batch(A, 2, (889_700, 889_701), 2000, p0=_p0(), record_every=1, s_half=1.0)
    for r in rec:
        assert not r["placed_at_init"] and r["crossed"] and r["G1_first_step"] == 0
        p = np.array(json.loads(r["p_cross"]))
        assert B.gap_rd(p[0], p[1], p[4:], p[2], A) > 0 and r["s_cross"] == abs(p[2])
        assert r["step"] >= 1 and np.isnan(r["s_cross_minus_window"])  # crossed before a full growth window
    # the step before the crossing was unplaced (trace at every step: column 4 = G)
    i = 0
    t = int(rec[i]["step"])
    assert tr[t - 1, i, 4] > 0 and (t == 1 or tr[t - 2, i, 4] <= 0)
    # placed at init: no crossing is recorded
    rec2, _ = B.train_batch(A, 2, (889_700,), 50, p0=_p0(1, wn=0.0))
    assert rec2[0]["placed_at_init"] and not rec2[0]["crossed"]


# ------------------------------------------------------------------------------------------ lag quantities
def test_lag_quantities_reduce_to_width1_pipeline():
    from src.lag_law import kappa_k
    from src.residual_timescale import branch_hessian, relax_rate
    X, Y = B.make_data_rd(889_800, 1)
    p = np.array([-0.83, 3.83, 5.2, -17.0])
    vh = np.array([1e-4, 2e-4, 3e-3, 5e-5])
    rec = {"seed": 889_800, "p_cross": json.dumps(list(p)), "vhat_cross": json.dumps(list(vh)), "s_cross": 5.2,
           "s_cross_minus_window": 5.1}
    L = B._landscape()[1.3]
    q = B.lag_quantities(A, 1, rec, L)
    z, H, g, conv = branch_hessian(A, X[:, 0], Y, p[0], p[1], p[2], p[3])
    assert conv and q["branch_converged"]
    assert abs(q["relax_sig"] - relax_rate(H, vh[[0, 1, 3]])) < 1e-9 * abs(q["relax_sig"])
    assert abs(q["growth"] - math.log(5.2 / 5.1) / 100) < 1e-15
    assert q["winding_k"] == 0 and q["kappa_k"] == kappa_k(L["H"], L["tan"], L["dG"], L["p"], 0)
    assert abs(q["pred_r"] - q["kappa_k"] * q["chi"]) < 1e-15
    # the other winding copy: b₁ − 2π (and b₂ shifted to keep the logits) → k = −1
    p2 = p + np.array([0, -2 * math.pi, 0, 2 * math.pi * 5.2])
    q2 = B.lag_quantities(A, 1, {**rec, "p_cross": json.dumps(list(p2))}, L)
    assert q2["winding_k"] == -1 and abs(q2["relax_sig"] - q["relax_sig"]) < 1e-6 * abs(q["relax_sig"])


def test_lag_quantities_in_rd_block_and_noise_branch():
    X, Y = B.make_data_rd(889_800, 2)
    p = np.array([-0.83, 3.83, 5.2, -17.0, 0.01])
    rec = {"seed": 889_800, "p_cross": json.dumps(list(p)), "vhat_cross": json.dumps([1e-4, 2e-4, 3e-3, 5e-5, 1e-4]),
           "s_cross": 5.2, "s_cross_minus_window": float("nan")}
    q = B.lag_quantities(A, 2, rec, B._landscape()[1.3])
    assert q["branch_converged"] and np.isnan(q["chi"]) and np.isnan(q["pred_r"])   # no growth window → unusable
    assert q["branch_wnoise_l2"] < 0.2                                               # finite-sample noise weight is small


# ------------------------------------------------------------------------------------------ registered scoring rules
REF = {"q1": 0.035, "q3": 0.15}
S_LO = 4.95


def _cell(n=40, frac_above=1.0, med_r=0.09, lag_obs=0.03, lag_pred=0.03, usable_frac=1.0, g_ok=True):
    k = int(round(frac_above * n))
    s = np.r_[np.full(k, 5.3), np.full(n - k, 4.9)]
    r_pop = np.full(n, med_r)
    obs = np.full(n, lag_obs); pred = np.full(n, lag_pred)
    usable = np.arange(n) < int(round(usable_frac * n))
    g = np.full(n, 0.01) if g_ok else np.r_[np.full(n - 1, 0.01), 0.0]
    return B.score_cell(s, r_pop, obs, pred, usable, S_LO, REF, g)


def test_P1():
    assert _cell(frac_above=1.0)["P1"] == "PASS"
    assert _cell(frac_above=0.9)["P1"] == "PASS"                    # boundary: exactly 90% passes
    assert _cell(frac_above=0.875)["P1"] == "FAIL"
    assert _cell(frac_above=0.5)["P1"] == "FAIL"


def test_P2a():
    assert _cell(med_r=0.09)["P2a"] == "PASS"
    assert _cell(med_r=0.035)["P2a"] == "PASS" and _cell(med_r=0.15)["P2a"] == "PASS"   # quartiles inclusive
    c = _cell(med_r=0.20); assert c["P2a"] == "FAIL" and c["P2a_side"] == "above"
    c = _cell(med_r=0.01); assert c["P2a"] == "FAIL" and c["P2a_side"] == "below"


def test_P2b():
    assert _cell(lag_obs=0.035, lag_pred=0.03)["P2b"] == "PASS"            # within the absolute floor 0.01
    assert _cell(lag_obs=0.045, lag_pred=0.03)["P2b"] == "FAIL"            # 0.015 > max(0.01, 0.0075)
    assert _cell(lag_obs=0.13, lag_pred=0.10)["P2b"] == "FAIL"             # 0.03 > max(0.01, 0.025)
    assert _cell(lag_obs=0.12, lag_pred=0.10)["P2b"] == "PASS"             # 0.02 <= 0.025
    assert _cell(lag_obs=-0.03, lag_pred=0.03)["P2b"] == "FAIL"            # wrong sign
    assert _cell(usable_frac=0.75)["P2b"] == "UNRESOLVED"                  # > 20% unusable
    assert _cell(usable_frac=0.85)["P2b"] == "PASS"                        # 34 usable, 15% unusable


def test_unresolved_and_stop():
    c = _cell(n=29)
    assert c["P1"] == c["P2a"] == c["P2b"] == "UNRESOLVED"
    assert _cell(n=30)["P1"] == "PASS"
    c = _cell(g_ok=False)
    assert c["P1"] == c["P2a"] == c["P2b"] == "STOP"
    c = B.score_cell([], [], [], [], [], S_LO, REF, [])
    assert c["P1"] == "UNRESOLVED"


def test_frozen_input_check(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT", tmp_path)
    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    B._freeze("x.csv")
    B.check_frozen("x.csv")
    (tmp_path / "x.csv").write_text("a,b\n1,3\n")
    with pytest.raises(AssertionError):
        B.check_frozen("x.csv")


# ------------------------------------------------------------------------------------------ population check pieces
def test_population_loss_reduces_to_width1_and_noise_block():
    from src.lag_law import _loss_t
    from src.width2_conditional import population
    x, y = population()
    s = 4.95
    h1 = np.array([-0.83, 3.83, -16.0])
    L2 = B.pop_loss_fn(A, s, 2, 16)
    l1 = float(_loss_t(torch.tensor(h1), s, A, torch.tensor(x), torch.tensor(y)))
    assert abs(float(L2(torch.tensor(np.r_[h1, 0.0]))) - l1) < 1e-13
    H = torch.autograd.functional.hessian(L2, torch.tensor(np.r_[h1, 0.0])).numpy()
    assert abs(H[3, 3] - (4 / 3) * H[1, 1]) < 1e-10 * abs(H[1, 1])             # noise block = E[x_n²]·∂²L/∂b₁²
    assert np.abs(H[:3, 3]).max() < 1e-12                                      # decoupled from (w₁, b₁, b₂)
    # averaging: any noise weight raises the loss above the width-1 minimum over b₁ shifts
    L4 = B.pop_loss_fn(A, s, 4, 4)
    assert abs(float(L4(torch.tensor(np.r_[h1, 0.0, 0.0, 0.0]))) - l1) < 1e-13


def test_newton_min_fixed_coordinates():
    L = lambda h: ((h - torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)) ** 2).sum()
    h, f, g = B.newton_min(L, [0.0, 0.0, 0.0], fixed=np.array([False, True, False]))
    assert np.allclose(h, [1.0, 0.0, 3.0]) and abs(f - 4.0) < 1e-12


# ------------------------------------------------------------------------------------------ end to end (scoring pipeline)
def test_score_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT", tmp_path)
    lines = []
    for arm, seeds, d in (("primary", (889_900, 889_901), 2), ("primary", (889_900, 889_901), 1),
                          ("secondary", (889_910, 889_911), 2)):
        p0 = _p0(2, 0.03, 2) if d == 2 else _p0(2, 0.0, 2)[:, :4]          # d = 1: placed at init (no crossing)
        rec, tr = B.train_batch(1.30, d, seeds, 400, p0=p0, s_half=1.0)
        np.savez_compressed(tmp_path / f"traces_{arm}_d{d}_a1.30.npz", trace=tr, seeds=np.array(seeds))
        lines += [json.dumps({"arm": arm, **r}, default=float) for r in rec]
    (tmp_path / "runs.jsonl").write_text("\n".join(lines) + "\n")
    ref = {"1.30": {"q1": 0.035, "q3": 0.15}, "1.50": {"q1": 0.07, "q3": 0.18}}
    (tmp_path / "width1_reference.json").write_text(json.dumps(ref)); B._freeze("width1_reference.json")
    pd.DataFrame({"a": [1.3, 1.3], "seed": [889_900, 889_901], "w2_own": [5.0, 5.1]}).to_csv(
        tmp_path / "own_x1_frozen.csv", index=False)
    B._freeze("own_x1_frozen.csv")
    V, D = B.score()
    assert set(V.P1) == {"UNRESOLVED"}                                  # 2 or 4 runs per cell < 30
    assert list(V.arm) == ["primary", "primary", "secondary"] and list(V.d) == [1, 2, 2]
    assert V[V.arm == "secondary"].n_runs.iloc[0] == 4                  # pooled: primary + secondary seeds
    assert V[V.arm == "secondary"].P2b.iloc[0] == "not scored (secondary)"
    assert (tmp_path / "scored_runs.csv").exists() and (tmp_path / "verdicts.json").exists()
    sr = pd.read_csv(tmp_path / "scored_runs.csv")
    assert (sr[sr.crossed].G_numpy_at_cross > 0).all()
    assert D[(D.arm == "primary") & (D.d == 1)].n_crossing.iloc[0] == 0
    # tampering with a frozen input stops scoring
    (tmp_path / "own_x1_frozen.csv").write_text("a,seed,w2_own\n1.3,889900,4.0\n")
    with pytest.raises(AssertionError):
        B.score()
    with pytest.raises(AssertionError):
        B.run()


# ------------------------------------------------------------------------------------------ POST HOC helper
def test_branch_switch_job_constructed():
    p = _p0(1, 0.0, 2)[0, :4]                                         # d = 1: the R^d and G1 switches coincide
    r = B._branch_switch_job(("t", 1, 1.30, 889_700, json.dumps(list(p))))
    assert np.isfinite(r["s_branch"]) and r["s_branch"] == r["s_branch_G1"] and r["s_branch_note"] == ""
    X, Y = B.make_data_rd(889_700, 1)
    for f, sign in ((1.001, 1), (0.999, -1)):                         # placed just above the switch, unplaced below
        s = r["s_branch"] * f
        h, _, g, conv = B.branch_rd(1.30, X, Y, np.r_[p[0], p[1], s, p[3]])
        assert conv and np.sign(B.gap_rd(h[0], h[1], h[3:], 1.0, 1.30)) == sign
    p2 = _p0(1, 0.02, 2)[0]                                           # d = 2: noise widening raises the switch
    r2 = B._branch_switch_job(("t", 2, 1.30, 889_700, json.dumps(list(p2))))
    assert r2["s_branch"] > r2["s_branch_G1"]
