"""Track L (registered; src/trackL.py): constructed PASS / FAIL / UNRESOLVED cases for R, N and validity; the data
construction (counts, p flips, test sets) reproducing the exploration's dataset hash; the OpenTimestamps guard of `run`;
resumability; no criterion reads a forecast; and one committed exploration run reproduced bit for bit."""
import gc
import inspect
import json
import sys

import numpy as np
import pytest

from src import trackL as L

LV = L.LEVEL_KEYS


# ------------------------------------------------------------------------------------------ the exploration's data hash
def test_exploration_code_builds_the_registered_dataset_hash():
    """The committed exploration code (results/designs/trackL_explore/L_data.py, b08843d) builds, at p = 0.8, exactly the
    dataset whose hash is registered.  Runs first (before the module data fixture) to bound memory."""
    sys.path.insert(0, str(L.EXPLORE))
    try:
        import L_data
        d = L_data.build(0.8)
        assert L.dataset_sha256(d) == L.EXPLORATION_DATA_SHA256
        assert L_data.CONSTRUCTION_SEED == L.CONSTRUCTION_SEED and L_data.PUBLISHED == L.PUBLISHED
        del d
    finally:
        sys.path.remove(str(L.EXPLORE))
        gc.collect()


@pytest.fixture(scope="module")
def data():
    d = L.build()
    yield d
    del d
    gc.collect()


def test_registered_build_reproduces_the_exploration_dataset_hash(data):
    assert L.dataset_sha256(data) == L.EXPLORATION_DATA_SHA256


def test_counts_flips_and_test_sets(data):
    n = L.N_TRAIN_PER_CLASS
    assert data["Xtr"].shape == (10_000, 6144) and data["Xtr"].dtype == np.float32
    assert data["n_train_per_class"] == 5000 and data["n_test_per_class"] == 980
    for k in ("Xorig", "Xrand", "Xrev"):
        assert data[k].shape == (1960, 6144)
    y = data["ytr"]
    assert (y[:n] == 0).all() and (y[n:] == 1).all()
    fl = data["flip"]
    assert fl[:n].sum() == 1000 and fl[n:].sum() == 1000                       # p = 0.8: exactly 20% per class
    ml = data["src_train_mnist_label"]
    assert (ml[~fl] == y[~fl]).all() and (ml[fl] == 1 - y[fl]).all()          # flipped: the other class's digit
    for d_ in (0, 1):
        assert (ml == d_).sum() == n                                           # every digit used exactly n times
    cifar = np.array(L.CIFAR_CLASSES)
    assert (data["src_train_cifar_label"] == cifar[y]).all()                   # CIFAR always correct
    yt = data["yte"]
    assert (data["src_test_cifar_label"] == cifar[yt]).all()
    assert (data["src_test_mnist_label_orig"] == yt).all()
    assert (data["src_test_mnist_label_rev"] == 1 - yt).all()
    agree = float(np.mean(data["src_test_mnist_label_rand"] == yt))
    assert agree == data["rand_mnist_agrees"] == L.RAND_MNIST_AGREES == 938 / 1960
    assert round(agree, 3) == 0.479


def test_image_layout(data):
    X = data["Xtr"][:50].reshape(-1, 3, 64, 32)
    assert X.min() >= 0.0 and X.max() <= 1.0 and data["Xtr"].max() == 1.0
    top = X[:, :, :32]
    assert (top[:, 0] == top[:, 1]).all() and (top[:, 0] == top[:, 2]).all()     # MNIST repeated to 3 channels
    assert (top[:, :, :2] == 0).all() and (top[:, :, 30:] == 0).all()
    assert (top[:, :, :, :2] == 0).all() and (top[:, :, :, 30:] == 0).all()     # zero padding by 2
    assert (X[:, :, 32:] > 0).mean() > 0.9                                      # CIFAR below


def test_verify_data_refuses_mismatch_and_missing(monkeypatch, tmp_path):
    assert set(L.verify_data()["npy"].values()) == set(L.NPY_SHA256.values())
    bad = dict(L.NPY_SHA256)
    bad["mnist/train_labels.npy"] = "0" * 64
    monkeypatch.setattr(L, "NPY_SHA256", bad)
    with pytest.raises(RuntimeError, match="SHA-256"):
        L.verify_data()
    monkeypatch.undo()
    for ds in ("mnist", "cifar10"):
        (tmp_path / ds).mkdir()
        (tmp_path / ds / "SHA256SUMS").write_text((L.DATA / ds / "SHA256SUMS").read_text())
    monkeypatch.setattr(L, "DATA", tmp_path)
    with pytest.raises(RuntimeError, match="no downloads"):
        L.verify_data()


def test_no_download_path():
    src = inspect.getsource(L)
    for word in ("urllib", "mnist_data", "cifar_data", "requests", "torchvision"):
        assert word not in src.split('"""', 2)[2], word


# ------------------------------------------------------------------------------------------ network, schedule, arms
def test_constants_match_the_approved_page():
    assert L.P == 0.8 and L.CONSTRUCTION_SEED == 20261009 and L.WIDTH == 256
    assert (L.LR, L.MOMENTUM, L.BATCH, L.STOP_LOSS) == (0.01, 0.9, 128, 0.002)
    assert L.F_OUT == 16.0 and L.ARMS == ("std", "out16", "glob16")
    assert dict(L.LEVELS) == {"bce_0.6": 0.6, "bce_0.3": 0.3, "bce_0.03": 0.03}
    assert L.DELTA == 0.02 and L.REACH_MIN_FRAC == 0.90 and L.BOOT_N == 10_000 and L.BOOT_PCT == (2.5, 97.5)
    assert L.SEEDS == tuple(range(2_992_000, 2_992_040)) and len(L.SEEDS) == 40
    assert L.PILOT_SEEDS == tuple(range(2_993_000, 2_993_010))
    s = set(L.SEEDS)
    assert not (s & set(L.PILOT_SEEDS)) and not (s & set(L.EXPLORE_SEEDS)) and not (s & set(L.ALPHA_EXPLORE_SEEDS))
    assert not (set(L.PILOT_SEEDS) & (set(L.EXPLORE_SEEDS) | set(L.ALPHA_EXPLORE_SEEDS)))
    assert L.FREEZE_REPRO_SEED in L.EXPLORE_SEEDS
    assert L.STEP_CAP == 200_000 and L.NO_CRITERION_IS_A_FORECAST


def test_lr_groups():
    import torch
    net = L.make_net(2_991_000, torch)
    g = L.lr_groups(net, "std")
    assert len(g) == 1 and g[0]["lr"] == 0.01 and len(g[0]["params"]) == 6
    g = L.lr_groups(net, "glob16")
    assert len(g) == 1 and g[0]["lr"] == 0.01 / 16
    g = L.lr_groups(net, "out16")
    assert g[0]["lr"] == 0.01 / 16 and g[0]["params"] == [net[4].weight] and g[1]["lr"] == 0.01
    assert any(q is net[4].bias for q in g[1]["params"]) and len(g[1]["params"]) == 5
    with pytest.raises(ValueError):
        L.lr_groups(net, "out4")


def test_init_inside_fork_rng_leaves_global_state():
    import torch
    before = torch.get_rng_state().clone()
    a = L.make_net(2_991_000, torch)
    b = L.make_net(2_991_000, torch)
    assert torch.equal(torch.get_rng_state(), before)
    assert L._params_sha(a) == L._params_sha(b)
    assert L._params_sha(L.make_net(2_991_001, torch)) != L._params_sha(a)


def test_check_schedule_matches_exploration():
    old = L.explore_row("out16", 2_991_000)["rec"]["t"]
    t, sched = 0, [0]
    while sched[-1] < old[-1]:
        t = L.next_check(t)
        sched.append(t)
    assert sched == old
    assert sched[:3] == [0, 5, 10] and 500 in sched and L.next_check(500) == 510 and L.next_check(1000) == 1020


def test_matched_first_check_at_or_below_level():
    rec = {"loss": [0.7, 0.61, 0.6, 0.31, 0.29, 0.05, 0.03]}
    assert L.matched(rec) == {"bce_0.6": 2, "bce_0.3": 4, "bce_0.03": 6}
    assert L.matched({"loss": [0.7, 0.5]}) == {"bce_0.6": 1, "bce_0.3": None, "bce_0.03": None}


def _tiny(n=256, bad=False):
    rng = np.random.default_rng(0)
    X = rng.random((n, 6144), dtype=np.float32)
    if bad:
        X[0, 0] = np.inf
    y = np.repeat(np.array([0, 1], dtype=np.int64), n // 2)
    T = rng.random((20, 6144), dtype=np.float32)
    return {"Xtr": X, "ytr": y, "flip": np.arange(n) % 5 == 0, "Xorig": T, "Xrand": T, "Xrev": T,
            "yte": np.repeat(np.array([0, 1], dtype=np.int64), 10)}


def test_run_stops_at_cap_and_at_a_nonfinite_check():
    r = L.run_one("std", 2_991_000, _tiny(), cap=10, rss_cap=None)
    assert r["stop"] == "cap" and r["steps"] == 10 and r["finite"] and r["rec"]["t"] == [0, 5, 10]
    r = L.run_one("out16", 2_991_000, _tiny(bad=True), cap=10, rss_cap=None)
    assert r["stop"] == "nonfinite" and not r["finite"] and r["t_nonfinite"] == 0 and r["steps"] == 0


# ------------------------------------------------------------------------------------------ constructed criterion cases
def _runs(d2=0.01, d3=0.0, n=40, miss2=(), miss3=(), nonfinite=(), per_level2=None, per_level3=None):
    """Rows for n seeds: arm 1 rand = 0 (so every paired difference is exactly the constructed value); arm 2 = d2 (or
    per_level2[j]); arm 3 likewise.
    miss2/miss3: [(seed index, level key)] not reached in that arm.  nonfinite: arms with one non-finite run."""
    runs = []
    for i in range(n):
        base = 0.0
        for arm in L.ARMS:
            at = {}
            for j, k in enumerate(LV):
                dd = 0.0
                if arm == "out16":
                    dd = per_level2[j] if per_level2 is not None else d2
                elif arm == "glob16":
                    dd = per_level3[j] if per_level3 is not None else d3
                miss = (arm == "out16" and (i, k) in miss2) or (arm == "glob16" and (i, k) in miss3)
                at[k] = None if miss else {"t": 100 * (j + 1), "rand": base + dd, "rev": 0.3, "orig": 0.9,
                                           "acc_flip": 0.5, "acc_pred": 0.9, "outw": 1.0, "hid": 10.0, "loss": 0.1}
            fin = not (arm in nonfinite and i == 3)
            runs.append({"arm": arm, "seed": 1000 + i, "finite": fin, "t_nonfinite": None if fin else 50, "at": at,
                         "steps": 1000, "stop": "loss" if fin else "nonfinite",
                         "end": {"t": 1000, "rand": base, "rev": 0.3, "orig": 0.9, "acc_flip": 1.0, "acc_pred": 1.0,
                                 "outw": 1.0, "hid": 10.0, "loss": 0.002}})
    return runs


def test_R_pass_fail_strict():
    st = L.score_tables(_runs(d2=0.01))
    assert st["verdicts"]["R"] == "PASS" and np.allclose(st["arms"]["out16"]["criterion"]["lower"], 0.01)
    st = L.score_tables(_runs(per_level2=(0.01, 0.01, 0.0)))                     # a lower end of exactly 0 fails
    assert st["verdicts"]["R"] == "FAIL"
    st = L.score_tables(_runs(per_level2=(0.01, -0.01, 0.01)))
    assert st["verdicts"]["R"] == "FAIL"


def test_N_pass_fail_strict_and_one_sided():
    assert L.score_tables(_runs(d3=0.019))["verdicts"]["N"] == "PASS"
    assert L.score_tables(_runs(per_level3=(0.0, 0.0, 0.02)))["verdicts"]["N"] == "FAIL"      # upper = delta fails
    assert L.score_tables(_runs(per_level3=(0.0, 0.03, 0.0)))["verdicts"]["N"] == "FAIL"
    assert L.score_tables(_runs(d3=-0.5))["verdicts"]["N"] == "PASS"                           # worse than arm 1 passes


def test_reach_rule_36_of_40():
    four = [(i, "bce_0.03") for i in range(4)]
    five = [(i, "bce_0.03") for i in range(5)]
    st = L.score_tables(_runs(d2=0.01, miss2=four, d3=0.0, miss3=four))
    assert st["verdicts"] == {"R": "PASS", "N": "PASS"}
    assert st["arms"]["out16"]["criterion"]["n_both"] == [40, 40, 36]
    st = L.score_tables(_runs(d2=0.01, miss2=five))
    assert st["verdicts"] == {"R": "UNRESOLVED", "N": "PASS"}
    st = L.score_tables(_runs(d2=-0.01, miss2=five, d3=0.5, miss3=five))                 # reach overrides FAIL
    assert st["verdicts"] == {"R": "UNRESOLVED", "N": "UNRESOLVED"}
    # reach is counted in BOTH arms: a seed missing in arm 1 counts against both criteria
    runs = _runs(d2=0.01)
    for r in runs:
        if r["arm"] == "std" and r["seed"] < 1005:
            r["at"]["bce_0.6"] = None
    assert L.score_tables(runs)["verdicts"] == {"R": "UNRESOLVED", "N": "UNRESOLVED"}


def test_nonfinite_scope_2B_D5():
    assert L.score_tables(_runs(d2=-0.01, nonfinite=("out16",)))["verdicts"] == {"R": "UNRESOLVED", "N": "PASS"}
    assert L.score_tables(_runs(d2=-0.01, d3=0.5, nonfinite=("glob16",)))["verdicts"] == {"R": "FAIL",
                                                                                          "N": "UNRESOLVED"}
    assert L.score_tables(_runs(d2=0.01, nonfinite=("std",)))["verdicts"] == {"R": "UNRESOLVED", "N": "UNRESOLVED"}


def test_outcome_statements():
    o = L.score_tables(_runs(d2=0.0))["outcome"]
    assert o["outcome"] == "R FAIL, N PASS" and o["matches_prediction"]
    assert o["statements"][0].startswith("R FAIL, N PASS (predicted)")
    o = L.score_tables(_runs(d2=0.01))["outcome"]
    assert o["outcome"] == "R PASS, N PASS" and not o["matches_prediction"]
    assert any("the lever transfers, against the prediction" in s for s in o["statements"])
    o = L.score_tables(_runs(d2=0.0, d3=0.05))["outcome"]
    assert o["outcome"] == "R FAIL, N FAIL" and any("global slowing helps" in s for s in o["statements"])
    o = L.score_tables(_runs(d2=-0.01))["outcome"]
    assert any("the lever hurts" in s for s in o["statements"]) and o["verdicts"]["R"] == "FAIL"
    assert not any("the lever hurts" in s for s in L.score_tables(_runs(d2=0.0))["outcome"]["statements"])
    o = L.score_tables(_runs(d2=0.0, nonfinite=("std",)))["outcome"]
    assert o["outcome"] == "UNRESOLVED" and not o["matches_prediction"]


def test_incomplete_or_duplicated_runs_refused():
    runs = _runs()
    with pytest.raises(AssertionError):
        L.score_tables(runs[:-1])
    with pytest.raises(AssertionError):
        L.score_tables(runs + runs[:1])


def test_bootstrap_is_2Bs_registered_procedure():
    from src import phase2b as P2B
    rng = np.random.default_rng(5)
    D = rng.normal(0, 0.01, size=(40, 3))
    D[0, 2] = np.nan
    a = L.boot_ci_median(D)
    b = P2B.boot_ci_median(D, seed=L.BOOT_SEED)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    # explicit: a fresh default_rng(20261009), one 10,000 x 40 index matrix, nanmedian, linear percentiles
    idx = np.random.default_rng(20261009).integers(0, 40, size=(10_000, 40))
    med = np.nanmedian(D[idx], axis=1)
    assert np.array_equal(a[0], np.percentile(med, 2.5, axis=0)) and np.array_equal(a[1], np.percentile(med, 97.5, axis=0))


def test_descriptive_is_never_a_verdict():
    d = L.descriptive(_runs(d2=0.01, d3=-0.01))
    assert d["endpoint_no_advantage_claimed"] and "verdict" not in json.dumps(d).replace("never a verdict", "")
    assert d["arms"]["out16"]["step_cost_vs_arm1"]["bce_0.6"]["median"] == 1.0
    assert "bootstrap_rev_DESCRIPTIVE" in d["arms"]["out16"]


# ------------------------------------------------------------------------------------------ strict causal rule
def test_no_criterion_reads_a_forecast():
    assert L.code_closure() == ["src/trackL.py"]
    for fn in (L.paired, L.boot_ci_median, L.reach_ok, L.criterion_R, L.criterion_N, L.score_tables, L.outcome,
               L.verdict, L.arm_cells):
        assert "forecast" not in inspect.getsource(fn).lower(), fn.__name__
    for runs in (_runs(d2=0.0), _runs(d2=0.01, d3=0.5)):
        base = L.score_tables(runs)
        for r in runs:
            r["forecast"] = {"rand": 99.0, "verdict": "PASS"}
            for k in LV:
                if r["at"][k] is not None:
                    r["at"][k]["forecast_rand"] = -99.0
        assert L.score_tables(runs)["verdicts"] == base["verdicts"]


# ------------------------------------------------------------------------------------------ the OTS guard, resumability
def _fake_registration(monkeypatch, tmp_path, ots):
    monkeypatch.setattr(L, "OUT", tmp_path)
    (tmp_path / "registration.sha256").write_text("")
    (tmp_path / "registration_stamp.txt").write_text("x\n")
    if ots:
        (tmp_path / "registration_stamp.txt.ots").write_bytes(b"proof")
    monkeypatch.setattr(L, "_assert_committed", lambda p: None)
    monkeypatch.setattr(L, "_setup", lambda: None)


def test_run_refuses_without_the_ots_proof(monkeypatch, tmp_path):
    _fake_registration(monkeypatch, tmp_path, ots=False)
    called = []
    for name in ("memory_gate", "build", "run_one"):
        monkeypatch.setattr(L, name, lambda *a, _n=name, **k: called.append(_n))
    with pytest.raises(SystemExit, match="registration_stamp.txt.ots"):
        L.run()
    with pytest.raises(SystemExit, match="OpenTimestamps"):
        L.score()
    assert called == [] and not (tmp_path / "runs.jsonl").exists()


def test_run_checks_the_proof_before_anything_else():
    # whatever the repository state, assert_registration checks the .ots FIRST
    src = inspect.getsource(L.assert_registration)
    assert src.index("registration_stamp.txt.ots") < src.index("registration.sha256")
    assert inspect.getsource(L.run).split('"""', 2)[2].lstrip().startswith("assert_registration()")


def test_run_proceeds_with_the_proof_and_resumes_per_seed_arm(monkeypatch, tmp_path):
    _fake_registration(monkeypatch, tmp_path, ots=True)
    monkeypatch.setattr(L, "memory_gate", lambda tag: (50, 1000))
    monkeypatch.setattr(L, "build", lambda: {"fake": True})
    monkeypatch.setattr(L, "dataset_sha256", lambda d: L.EXPLORATION_DATA_SHA256)
    monkeypatch.setattr(L, "SEEDS", (7, 8))
    calls = []

    def fake_run_one(arm, seed, data):
        calls.append((seed, arm))
        return {"arm": arm, "seed": seed, "steps": 1, "stop": "loss", "finite": True, "secs": 0, "max_rss_gb": 0}

    monkeypatch.setattr(L, "run_one", fake_run_one)
    L._append_write(tmp_path / "runs.jsonl", fake_run_one("std", 7, None))
    L._append_write(tmp_path / "runs.jsonl", fake_run_one("out16", 7, None))
    calls.clear()
    L.run()
    assert calls == [(7, "glob16"), (8, "std"), (8, "out16"), (8, "glob16")]
    assert sorted((r["seed"], r["arm"]) for r in L._rows(tmp_path / "runs.jsonl")) == sorted(
        (s, a) for s in (7, 8) for a in L.ARMS)
    calls.clear()
    L.run()
    assert calls == []


def test_manifest_covers_code_tests_page_registration_and_data_check():
    files = L.manifest_files()
    for f in ("src/trackL.py", "tests/test_trackL.py", "results/designs/trackL_dominoes_design.md",
              "results/trackL_registration.md", "results/trackL/seed_scan.json", "results/trackL/frozen.json",
              "results/designs/trackL_explore/L_data_check.log", "results/designs/trackL_explore/L_data.py",
              "results/designs/trackL_alpha_explore/A_summary.json", "data/mnist/SHA256SUMS"):
        assert f in files, f


# ------------------------------------------------------------------------------------------ bit-for-bit reproduction
def test_reproduces_the_committed_exploration_run_bit_for_bit(data):
    """Arm 2 (out16) on exploration seed 2,991,000, the full run to train BCE <= 0.002 (8,110 steps): every check step
    and every recorded value (L_explore.py rounds to 5 decimals) equal to explore_lever.jsonl."""
    rec = L.run_one("out16", 2_991_000, data, rss_cap=None)
    old = L.explore_row("out16", 2_991_000)
    cmp = L.compare_with_exploration(rec, old)
    assert cmp["exact"], cmp
    assert rec["steps"] == 8110 and rec["stop"] == "loss" and cmp["n_values"] == 8 * len(old["rec"]["t"])
    assert rec["at"]["bce_0.03"]["t"] == old["rec"]["t"][L.matched(old["rec"])["bce_0.03"]]
