"""Phase 2B (the lever; results/designs/phase2b_lever_design.md, approved 2026-10-02): registered settings, the model
and measurements, the exact reproduction of committed exploration runs, and constructed PASS / FAIL / UNRESOLVED cases
for every criterion (R_s, R_l-rho2, R_l-acc, O; N for each global arm), the 90%-reach validity rule, non-finite runs and
the outcome rule.  No criterion is a forecast (tested).  Exploration seeds only (2,953,000+); no registered seed drawn."""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

from src import phase2b as P
from src import simplicity_bias_v2 as v2
from src import simplicity_bias_v3 as v3

EXP = P.EXPLORE
FROZEN = P.OUT / "frozen.json"
SCAN = P.OUT / "seed_scan.json"


# ------------------------------------------------------------------------------------------ settings
def test_registered_settings_match_the_approved_page():
    assert P.T_C == 40_000
    assert P.SEEDS == tuple(range(2_962_000, 2_962_080)) and len(P.SEEDS) == 80
    assert (P.G_GLOB, P.G_CM, P.OUT_FACTOR) == (12.18, 16.14, 16.0)
    assert P.ARMS[0] == "std" and set(P.ARMS) == {"std", "out16", "out16b", "glob", "globcm"}
    assert (P.PRIMARY, P.SECONDARY) == ("out16", "out16b")
    assert P.OUTPUT_ARMS == ("out16", "out16b") and P.GLOBAL_ARMS == ("glob", "globcm")
    assert [f for _, f in P.SCALE_LEVELS] == [1.25, 2.0, 3.0]
    assert [lv for _, lv in P.BCE_LEVELS] == [0.1, 0.03, 0.01]
    assert {k: len(v) for k, v in P.CRITERION_CELLS.items()} == {"R_s": 9, "R_l_rho2": 3, "R_l_acc": 6, "N": 18}
    assert P.DELTA == {"rho2": 0.03, "acc_shuffled": 0.02, "acc_reversed": 0.03}
    assert (P.O_MIN_FRAC, P.REACH_MIN_FRAC) == (0.75, 0.90)
    assert (P.BOOT_N, P.BOOT_SEED, P.BOOT_PCT) == (10_000, 20261002, (2.5, 97.5))
    assert (P.ADAM_LR, P.B1, P.B2, P.EPS, P.LAM) == (0.01, 0.9, 0.999, 1e-8, 1e-4)
    assert P.CAP == 0.5 * P.S_STAR
    assert not set(P.SEEDS) & set(range(2_953_000, 2_953_100))


def test_learning_rate_vectors_v_only_vs_v_and_b():
    std = P.lr_vec("std")
    assert np.all(std == 0.01)
    o = P.lr_vec("out16")
    assert np.all(o[12:16] == 0.01 / 16) and np.all(o[:12] == 0.01) and o[16] == 0.01          # b NOT slowed
    ob = P.lr_vec("out16b")
    assert np.all(ob[12:17] == 0.01 / 16) and np.all(ob[:12] == 0.01)                            # v AND b slowed
    assert np.all(P.lr_vec("glob") == 0.01 / 12.18)
    assert np.all(P.lr_vec("globcm") == 0.01 / 16.14)
    with pytest.raises(ValueError):
        P.lr_vec("other")


def test_constants_match_v2_frozen_and_phase2a():
    f = json.loads(P.V2_FROZEN.read_text())
    assert (f["q"], f["s_q"], f["lambda"]) == (P.Q, P.S_STAR, P.LAM)
    assert json.loads(P.P2A_FROZEN.read_text())["s_F"] == P.S_FOLD


def test_global_factors_reproduce_from_the_committed_exploration():
    g = P.global_factors()
    assert g["G"] == 12.18 and g["G_cm"] == 16.14
    assert abs(g["r_o"] - 12.1773) < 1e-4 and abs(g["r_g"] - 9.4541) < 1e-4


# ------------------------------------------------------------------------------------------ model and measurements
def test_init_is_v3_init_net_and_restores_the_global_torch_rng():
    import torch
    before = torch.random.get_rng_state().clone()
    th, s0 = P.init_row(2_953_000)
    assert torch.equal(torch.random.get_rng_state(), before)
    with torch.random.fork_rng(devices=[]):
        hid, out, s0b = v3.init_net(2_953_000, P.CAP, torch)
    ref = np.concatenate([hid.weight.detach().numpy().ravel(), hid.bias.detach().numpy(),
                          out.weight.detach().numpy().ravel(), out.bias.detach().numpy()])
    assert np.array_equal(th, ref) and s0 == s0b
    assert np.abs(th[12:16]).sum() <= P.CAP * (1 + 1e-12)


def test_numpy_adam_matches_torch_adam_with_param_groups():
    import torch
    seed = 2_953_099                                    # exploration check seed (never registered)
    for arm in ("out16", "out16b"):
        th, _ = P.init_row(seed)
        lr = P.lr_vec(arm)
        X = torch.as_tensor(P.X); Y = torch.as_tensor(P.Y)
        W = torch.tensor(th[:8].reshape(4, 2), requires_grad=True); c = torch.tensor(th[8:12], requires_grad=True)
        v = torch.tensor(th[12:16], requires_grad=True); b = torch.tensor(th[16:17], requires_grad=True)
        opt = torch.optim.Adam([{"params": [W, c], "lr": 0.01}, {"params": [v], "lr": float(lr[12])},
                                {"params": [b], "lr": float(lr[16])}], betas=(P.B1, P.B2), eps=P.EPS)
        m = np.zeros(17); vv = np.zeros(17)
        for k in range(1, 201):
            z = torch.tanh(X @ W.T + c) @ v + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y) + 0.5 * P.LAM * ((W ** 2).sum()
                                                                                              + (c ** 2).sum())
            opt.zero_grad(); loss.backward(); opt.step()
            _, g = P.loss_grad(th)
            th, m, vv = P.adam_update(th, g, m, vv, k, lr)
        tt = np.concatenate([W.detach().numpy().ravel(), c.detach().numpy(), v.detach().numpy(), b.detach().numpy()])
        assert np.abs(tt - th).max() / np.abs(tt).max() < 1e-12


def test_rho2_equals_v2_rho2_batch():
    import torch
    rng = np.random.default_rng(5)
    Pm = rng.normal(size=(12, 17))
    assert np.abs(P.rho2_rows(Pm) - v2.rho2_batch(Pm, P.X, torch)).max() < 1e-12


def test_bce_is_the_objective_minus_the_penalty():
    th, _ = P.init_row(2_953_001)
    L, _ = P.loss_grad(th)
    z = np.tanh(P.X @ th[:8].reshape(4, 2).T + th[8:12]) @ th[12:16] + th[16]
    assert abs(P.bce(L, th) - np.mean(np.logaddexp(0, z) - P.Y * z)) < 1e-14


def test_shifted_accuracies_constructed():
    # a net that reads x2 only: shuffling or reversing x1 changes nothing
    th = np.zeros(17); th[1] = 1.0; th[12] = 2.0
    m = P.metrics(th)
    assert m["acc_shuffled"] == m["acc_train"] == m["acc_reversed"]
    # a net that reads x1 only: reversing x1 swaps the classes (accuracy 1 - train when no output is exactly 0)
    th = np.zeros(17); th[0] = 1.0; th[12] = 2.0
    m = P.metrics(th)
    z = np.tanh(P.X[:, 0])
    assert np.all(z != 0)
    assert abs(m["acc_reversed"] - (1 - m["acc_train"])) < 1e-15
    # shuffled: the exact permutation expectation (every x1 of the 800 points at every point)
    pred = (np.tanh(P.X[:, 0])[:, None] * 2 > 0)                             # (N x1 values, 1)
    exp = np.mean([(pred[j, 0] == P.YB).mean() for j in range(P.N)])
    assert abs(m["acc_shuffled"] - exp) < 1e-12


def test_lasting_onset_constructed():
    S = np.linspace(1, 11, 11)
    q = P.Q
    R = np.array([0.5, 0.1, 0.1, q, 0.1, q, q + 0.1, q, q, q, q])       # last dip below q at t = 4
    o = P.lasting_onset(R, S, 10)
    assert o["t"] == 5 and o["s"] == 6.0 and not o["from_init"]
    assert P.lasting_onset(R, S, 4) is None                                # below q at the end: no such step
    o = P.lasting_onset(np.full(11, 0.9), S, 10)
    assert o["t"] == 0 and o["from_init"] and o["s"] == 1.0                 # never below: t_on = 0 (s ≤ cap < s*)
    assert P.lasting_onset(np.r_[R[:-1], np.nan], S, 10) is None


def test_non_finite_run_is_flagged_and_stopped(monkeypatch):
    real = P.loss_grad
    calls = {"n": 0}

    def bad(th):
        calls["n"] += 1
        L, g = real(th)
        if calls["n"] > 30:
            g = g * np.nan
        return L, g
    monkeypatch.setattr(P, "loss_grad", bad)
    rec, (S, B, R) = P.run_one("std", 2_953_002, T=100)
    assert not rec["finite"] and rec["t_nonfinite"] is not None and rec["t_nonfinite"] <= 31
    assert rec["onset"] is None and "end_T_C" not in rec["at"]


# ------------------------------------------------------------------------------------------ exact reproduction
def test_reproduces_a_committed_exploration_run_exactly():
    """The primary arm on exploration seed 2,953,000 (40,000 steps) against runs_lever/out16_2953000.json (p2b_lever.run):
    every step to a matched point, every metric there, at step 40,000 and at the matched step, the lasting onset and
    the committed trajectory samples: difference exactly 0."""
    old = P.explore_run("out16", 2_953_000)
    rec, paths = P.run_one("out16", 2_953_000, t_match=old["t_match"])
    d = P.compare_with_exploration(rec, paths, old)
    assert d["exact"], d
    assert rec["finite"] and d["n_compared"] >= 56


@pytest.mark.parametrize("arm", P.ARMS)
def test_every_arm_reproduces_the_exploration_path_exactly_over_3000_steps(arm):
    seed = 2_953_001
    old = P.explore_run(arm, seed)
    rec, (S, B, R) = P.run_one(arm, seed, T=3_000)
    n = 0
    for i, t in enumerate(old["traj"]["t"]):
        if t <= 3_000:
            # rho2 is evaluated in the same 250-state chunks; the last partial chunk differs only in batch size
            assert (S[t], B[t]) == (old["traj"]["s"][i], old["traj"]["bce"][i])
            assert R[t] == old["traj"]["rho2"][i]
            n += 1
    assert n > 100
    for k, t in rec["t_first"].items():
        if k.startswith("scale"):
            assert old["t_scale"][k[6:]] == t


@pytest.mark.skipif(not FROZEN.exists(), reason="not yet frozen")
def test_frozen_inputs():
    fr = json.loads(FROZEN.read_text())
    assert fr["constants_match_v2_frozen"] and fr["s_F_matches_phase2a"] and fr["global_factors_reproduce"]
    assert fr["exploration_reproduction_exact"] and set(fr["exploration_reproduction"]) == set(P.ARMS)
    assert fr["T_C"] == P.T_C and fr["delta"] == P.DELTA and fr["cells"] == list(P.CELLS)
    assert fr["arms"] == {a: {"label": P.ARM_LABEL[a], "lr": P.lr_vec(a).tolist()} for a in P.ARMS}


@pytest.mark.skipif(not SCAN.exists(), reason="not yet scanned")
def test_seed_scan_unused():
    sc = json.loads(SCAN.read_text())
    assert sc["unused"] and sc["ranges"]["registered"] == [2_962_000, 2_962_079, 80]
    assert set(sc["dirs"]) == {"src", "tests", "results", "paper", "notes", "independent", "data", "dist"}


# ------------------------------------------------------------------------------------------ scoring on the exploration
def _explore_rows(arm):
    """Exploration runs (20 seeds) in run_one's record format (points renamed; onset lasting to 40,000)."""
    rows = []
    for s in P.EXPLORE_SEEDS:
        o = P.explore_run(arm, s)
        at = {mine: o["at"][theirs] for mine, theirs in P.EXPLORE_POINT.items() if theirs in o["at"]}
        rows.append({"arm": arm, "seed": s, "finite": True, "t_nonfinite": None, "at": at,
                     "onset": o["onset"]["T40000"], "t_first": {}})
    return rows


def test_bootstrap_reproduces_the_committed_exploration_interval():
    """boot_ci_median (fresh default_rng(20261002), 10,000 resamples, seeds jointly) on the exploration's out16 − std
    differences equals the lower ends committed in p2b_lever_power_out16_glob.json (numpy rounding to 4 decimals, as committed)."""
    D = P.paired(_explore_rows("out16"), _explore_rows("std"))
    lo, _, med = P.boot_ci_median(D)
    J = json.loads((EXP / "p2b_lever_power_out16_glob.json").read_text())["on_exploration_seeds"]
    for j, c in enumerate(P.CELLS):
        key = c.replace("bce_", "loss_")
        assert float(np.round(lo[j], 4)) == J["arm2_ci_low"][key]
        assert float(np.round(med[j], 4)) == J["arm2_median_diff"][key]


def test_verdicts_on_the_exploration_seeds_match_the_committed_power_analysis():
    std = _explore_rows("std")
    J = json.loads((EXP / "p2b_lever_power2.json").read_text())["arms"]
    for arm in ("out16", "out16b"):
        v = P.arm_verdicts(arm, _explore_rows(arm), std, False)["criteria"]
        e = J[arm]["on_exploration_seeds"]
        assert {k: v[k]["verdict"] == "PASS" for k in ("R_s", "R_l_rho2", "R_l_acc", "O")} == \
            {"R_s": e["R_s"], "R_l_rho2": e["R_l_rho2"], "R_l_acc": e["R_l_acc"], "O": e["O"]}
    assert P.arm_verdicts("out16b", _explore_rows("out16b"), std, False)["criteria"]["R_l_acc"]["verdict"] == "FAIL"
    for arm in ("glob", "globcm"):
        assert P.arm_verdicts(arm, _explore_rows(arm), std, False)["criteria"]["N"]["verdict"] == "PASS"


# ------------------------------------------------------------------------------------------ constructed cases
@pytest.fixture
def fast_boot(monkeypatch):
    monkeypatch.setattr(P, "BOOT_N", 2_000)


def _runs(n=80, delta=None, noise=0.002, onset_above=None, drop=None, nonfinite=(), seed=0):
    """Constructed runs: every arm × n seeds.  Arm 1 values 0.5 + 0.01·N(0,1) per cell; arm a = arm 1 + delta[a][cell]
    + noise·N(0,1).  Default delta: +0.1 on output arms, 0 on global arms.  onset_above[a] = number of runs with
    s(t_on) > s* (the rest: no onset if 'none' in the key list, else 0.9 s*).  drop = {(arm, point): k} removes the
    point from the first k seeds of that arm.  nonfinite = [(arm, seed index)]."""
    rng = np.random.default_rng(seed)
    delta = delta or {}
    onset_above = onset_above or {}
    drop = drop or {}
    base = {i: {p: {q: 0.5 + 0.01 * rng.normal() for q in P.QTY} for p in P.POINTS} for i in range(n)}
    runs = []
    for a in P.ARMS:
        d0 = 0.1 if a in P.OUTPUT_ARMS else 0.0
        k_above = onset_above.get(a, n)
        for i in range(n):
            at = {}
            for p in P.POINTS:
                if i < drop.get((a, p), 0):
                    continue
                at[p] = {q: base[i][p][q] + (0.0 if a == "std" else delta.get(a, {}).get(f"{p}|{q}", d0)
                                             + noise * rng.normal()) for q in P.QTY}
                at[p].update(s=2.0 * P.S_STAR, bce=0.05)
            if i < k_above:
                on = {"s": 1.5 * P.S_STAR}
            else:
                on = None if onset_above.get(a + ":none") else {"s": 0.9 * P.S_STAR}
            runs.append({"arm": a, "seed": 1000 + i, "finite": (a, i) not in nonfinite,
                         "t_nonfinite": 5 if (a, i) in nonfinite else None, "at": at, "onset": on, "t_first": {}})
    return runs


def _v(runs):
    return P.score_tables(runs)["verdicts"]


def test_all_pass_constructed(fast_boot):
    S = P.score_tables(_runs())
    assert S["verdicts"] == {"out16": {"R_s": "PASS", "R_l_rho2": "PASS", "R_l_acc": "PASS", "O": "PASS"},
                             "out16b": {"R_s": "PASS", "R_l_rho2": "PASS", "R_l_acc": "PASS", "O": "PASS"},
                             "glob": {"N": "PASS"}, "globcm": {"N": "PASS"}}
    assert S["outcome"]["primary"] == "PASS" and "All pass" in S["outcome"]["statements"][0]


@pytest.mark.parametrize("crit,cell", [("R_s", "scale_1.25s*|acc_shuffled"), ("R_s", "scale_3s*|rho2"),
                                       ("R_l_rho2", "bce_0.03|rho2"), ("R_l_acc", "bce_0.1|acc_reversed"),
                                       ("R_l_acc", "bce_0.01|acc_shuffled")])
@pytest.mark.parametrize("arm", ["out16", "out16b"])
def test_lower_end_criteria_fail_on_one_cell(fast_boot, crit, cell, arm):
    v = _v(_runs(delta={arm: {cell: 0.0}}))[arm]                  # one cell centred on 0: its lower end < 0
    assert v[crit] == "FAIL"
    assert all(x == "PASS" for k, x in v.items() if k != crit)


def test_lower_end_must_be_strictly_positive(fast_boot):
    # a cell whose paired differences are exactly 0 for every seed: lower end = 0, not > 0 -> FAIL
    runs = _runs(delta={"out16": {"bce_0.1|rho2": 0.0}}, noise=0.0)
    assert _v(runs)["out16"]["R_l_rho2"] == "FAIL"


@pytest.mark.parametrize("arm", ["glob", "globcm"])
@pytest.mark.parametrize("cell,d,verdict", [("scale_2s*|rho2", 0.025, "PASS"), ("scale_2s*|rho2", 0.035, "FAIL"),
                                            ("bce_0.01|acc_shuffled", 0.015, "PASS"),
                                            ("bce_0.01|acc_shuffled", 0.025, "FAIL"),
                                            ("scale_3s*|acc_reversed", 0.035, "FAIL"),
                                            ("scale_1.25s*|rho2", -0.2, "PASS")])        # one-sided: a loss passes N
def test_N_one_sided_margins(fast_boot, arm, cell, d, verdict):
    v = _v(_runs(delta={arm: {cell: d}}, noise=0.0005))
    assert v[arm]["N"] == verdict
    other = [a for a in P.GLOBAL_ARMS if a != arm][0]
    assert v[other]["N"] == "PASS"


@pytest.mark.parametrize("k,verdict", [(60, "PASS"), (59, "FAIL")])          # 60/80 = 0.75 ≥ 0.75; 59/80 < 0.75
@pytest.mark.parametrize("arm", ["out16", "out16b"])
def test_O_threshold(fast_boot, k, verdict, arm):
    v = _v(_runs(onset_above={arm: k}))[arm]
    assert v["O"] == verdict and v["R_s"] == "PASS"


def test_O_run_without_onset_counts_against(fast_boot):
    S = P.score_tables(_runs(onset_above={"out16": 59, "out16:none": True}))
    c = S["arms"]["out16"]["criteria"]["O"]
    assert c["verdict"] == "FAIL" and c["n_no_onset"] == 21 and c["n_above_s_star"] == 59


def test_O_onset_exactly_at_s_star_is_not_above():
    c = P.criterion_O([P.S_STAR] * 80, 80, False)
    assert c["verdict"] == "FAIL" and c["n_above_s_star"] == 0


@pytest.mark.parametrize("k,expect", [(8, "PASS"), (9, "UNRESOLVED")])       # 72/80 = 0.90 reach; 71/80 < 0.90
@pytest.mark.parametrize("arm,point,crit", [("out16", "scale_3s*", "R_s"), ("out16", "bce_0.01", "R_l_rho2"),
                                            ("out16b", "bce_0.03", "R_l_acc"), ("glob", "bce_0.01", "N"),
                                            ("globcm", "scale_2s*", "N")])
def test_reach_validity_90pct(fast_boot, k, expect, arm, point, crit):
    v = _v(_runs(drop={(arm, point): k}))
    assert v[arm][crit] == expect


def test_reach_counts_both_arms(fast_boot):
    # arm 1 misses 3 s* at 5 seeds and arm 2 at 5 other seeds: 70/80 reach in both -> UNRESOLVED
    runs = _runs()
    for r in runs:
        if r["arm"] == "std" and r["seed"] < 1005:
            r["at"].pop("scale_3s*")
        if r["arm"] == "out16" and 1005 <= r["seed"] < 1010:
            r["at"].pop("scale_3s*")
    v = _v(runs)
    assert v["out16"]["R_s"] == "UNRESOLVED"
    # arm 2b misses nothing, so its cells at 3 s* are reached in both arms by 75/80 (≥ 72): still resolved
    assert v["out16b"]["R_s"] == "PASS"
    assert P.score_tables(runs)["arms"]["out16b"]["criteria"]["R_s"]["n_both"][6:9] == [75, 75, 75]


def test_reach_unresolved_overrides_a_fail(fast_boot):
    v = _v(_runs(delta={"out16": {"scale_3s*|rho2": 0.0}}, drop={("out16", "scale_3s*"): 9}))
    assert v["out16"]["R_s"] == "UNRESOLVED"


@pytest.mark.parametrize("bad,voided", [
    (("glob", 3), {("glob", "N")}),                                                     # arm 3: only N(3)
    (("globcm", 40), {("globcm", "N")}),                                                # arm 3cm: only N(3cm)
    (("out16b", 79), {("out16b", c) for c in ("R_s", "R_l_rho2", "R_l_acc", "O")}),     # arm 2b: only 2b's
    (("out16", 0), {("out16", c) for c in ("R_s", "R_l_rho2", "R_l_acc", "O")}),        # arm 2: only 2's
    (("std", 0), "all")])                                                               # arm 1: every criterion
def test_non_finite_run_voids_only_the_criteria_that_use_its_arm(fast_boot, bad, voided):
    """D5 (author, 2026-10-05): a non-finite run makes UNRESOLVED only the criteria that use its arm; every criterion
    compares against arm 1, so a non-finite arm-1 run voids all of them.  Every other criterion keeps its verdict."""
    S = P.score_tables(_runs(nonfinite=[bad]))
    assert S["any_nonfinite"] and S["nonfinite_runs"] == [[bad[0], 1000 + bad[1], 5]]
    assert S["nonfinite_by_arm"] == {a: a == bad[0] for a in P.ARMS}
    for a, crits in S["verdicts"].items():
        for c, x in crits.items():
            expect = "UNRESOLVED" if voided == "all" or (a, c) in voided else "PASS"
            assert x == expect, (a, c, x)
    prim = voided == "all" or any(a in ("out16", "glob", "globcm") for a, _ in voided)
    assert S["outcome"]["primary"] == ("UNRESOLVED" if prim else "PASS")


def test_non_finite_run_does_not_hide_a_fail_elsewhere(fast_boot):
    # a non-finite arm-3cm run voids N(3cm) only; arm 2's failing R_l_acc stays FAIL
    v = _v(_runs(delta={"out16": {"bce_0.1|acc_shuffled": 0.0}}, nonfinite=[("globcm", 7)]))
    assert v["globcm"]["N"] == "UNRESOLVED" and v["glob"]["N"] == "PASS" and v["out16"]["R_l_acc"] == "FAIL"


def test_incomplete_or_duplicate_runs_are_refused():
    runs = _runs(n=4)
    with pytest.raises(AssertionError):
        P.score_tables(runs[:-1])
    with pytest.raises(AssertionError):
        P.score_tables(runs + runs[:1])


# ------------------------------------------------------------------------------------------ outcome rule
def _vd(**kw):
    v = {"out16": {"R_s": "PASS", "R_l_rho2": "PASS", "R_l_acc": "PASS", "O": "PASS"},
         "out16b": {"R_s": "PASS", "R_l_rho2": "PASS", "R_l_acc": "PASS", "O": "PASS"},
         "glob": {"N": "PASS"}, "globcm": {"N": "PASS"}}
    for k, x in kw.items():
        a, c = k.split("__")
        v[a][c] = x
    return v


def test_outcome_rule():
    o = P.outcome(_vd())
    assert o["primary"] == "PASS" and o["statements"][0].startswith("All pass")
    o = P.outcome(_vd(glob__N="FAIL"))
    assert o["primary"] == "FAIL" and o["statements"] == ["N fails (arm 3): global slowing helps too."]
    o = P.outcome(_vd(globcm__N="FAIL", glob__N="FAIL"))
    assert len(o["statements"]) == 2
    o = P.outcome(_vd(out16__R_l_acc="FAIL"))
    assert o["statements"] == ["Only R_l fails: a scale-only effect."]
    o = P.outcome(_vd(out16__R_l_acc="FAIL", out16__R_l_rho2="FAIL"))
    assert o["statements"] == ["Only R_l fails: a scale-only effect."]
    o = P.outcome(_vd(out16__O="FAIL"))
    assert o["statements"] == ["O fails: the onset moves to or below s*."]
    o = P.outcome(_vd(out16__O="FAIL", out16__R_l_rho2="FAIL"))
    assert "O fails: the onset moves to or below s*." in o["statements"]
    assert not any(s.startswith("Only R_l") for s in o["statements"])
    o = P.outcome(_vd(out16__R_s="FAIL"))
    assert o["primary"] == "FAIL" and "arm 2 R_s" in o["statements"][-1]
    o = P.outcome(_vd(glob__N="UNRESOLVED", out16__O="FAIL"))
    assert o["primary"] == "UNRESOLVED" and "arm 3 N" in o["statements"][0] and "arm 2 O" in o["statements"][1]


def test_secondary_arm_never_changes_the_primary_outcome():
    o = P.outcome(_vd(out16b__R_l_acc="FAIL", out16b__O="UNRESOLVED"))
    assert o["primary"] == "PASS"
    assert o["secondary_arm_2b"]["verdicts"]["R_l_acc"] == "FAIL" and o["secondary_arm_2b"]["any_unresolved"]


# ------------------------------------------------------------------------------------------ causal rule, descriptive
def test_no_criterion_reads_a_forecast():
    assert P.NO_CRITERION_IS_A_FORECAST is True
    closure = P.code_closure()
    assert not any("forecast" in f for f in closure), closure
    for fn in (P.paired, P.boot_ci_median, P.reach_ok, P.criterion_lower, P.criterion_N, P.criterion_O,
               P.arm_verdicts, P.outcome, P.score_tables):
        src = inspect.getsource(fn).lower()
        assert "forecast" not in src.replace("no criterion is a forecast", ""), fn.__name__
    # a forecast key in a run row is never read: adding one changes nothing
    runs = _runs(n=20)
    base = P.score_tables(runs)["verdicts"]
    for r in runs:
        r["forecast"] = {"t_fc": float("nan")}
    assert P.score_tables(runs)["verdicts"] == base


def test_descriptive_never_carries_a_verdict(fast_boot):
    runs = _runs(n=20)
    for r in runs:
        r["at"]["match_step"] = r["at"]["scale_3s*"]
        r["at"]["end_T_C"] = r["at"]["scale_3s*"]
        r["t_first"] = {p: 100 for p in P.POINTS}
        r["overshoot"] = {"scale_3s*": 0.001}
        if r["onset"]:
            r["onset"] = {**r["onset"], "s_over_s_star": r["onset"]["s"] / P.S_STAR,
                          "s_over_s_F": r["onset"]["s"] / P.S_FOLD, "from_init": False}
    D = P.descriptive(runs)
    assert D["endpoint_no_advantage_claimed"] is True
    txt = json.dumps(D["arms"])
    assert "PASS" not in txt and "FAIL" not in txt and "verdict" not in txt
    assert set(D["arms"]["glob"]["two_sided_equivalence"]) == set(P.CELLS)


def test_manifest_lists_the_registration_inputs():
    files = P.manifest_files()
    for f in ("src/phase2b.py", "src/simplicity_bias_v2.py", "src/simplicity_bias_v3.py", "tests/test_phase2b.py",
              "results/phase2b_registration.md", "results/designs/phase2b_lever_design.md",
              "results/phase2b/frozen.json", "results/phase2b/seed_scan.json",
              "results/designs/phase2b_explore/runs_lever/globcm_2953019.json"):
        assert f in files
    assert not any(f.endswith("registration.sha256") or "registration_stamp" in f for f in files)
