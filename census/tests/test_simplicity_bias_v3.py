"""Track T v3: decision rules on constructed PASS / FAIL / UNRESOLVED cases, the init rule, χ and the event, before
any registered training (results/simplicity_bias_v3_registration.md)."""

import math

import numpy as np
import pytest
import torch

from src import simplicity_bias_v3 as v3

SW = 3.5914


def test_decide_pass_pass():
    r = v3.decide(SW * np.linspace(1.01, 1.2, 36), np.full(36, 0.02), 40, SW)
    assert r["valid"] and r["C1"] == "PASS" and r["C2"] == "PASS"


def test_decide_c1_fail_c2_pass():
    s = SW * np.r_[np.full(6, 0.95), np.linspace(1.05, 1.2, 30)]            # 6/36 below the switch
    r = v3.decide(s, np.full(36, 0.02), 40, SW)
    assert r["valid"] and r["C1"] == "FAIL" and r["C2"] == "PASS"


def test_decide_c2_fail_high_and_low():
    r = v3.decide(SW * np.linspace(1.3, 1.6, 36), np.full(36, 0.02), 40, SW)
    assert r["C1"] == "PASS" and r["C2"] == "FAIL"
    r = v3.decide(SW * np.linspace(0.6, 0.9, 36), np.full(36, 0.02), 40, SW)
    assert r["C1"] == "FAIL" and r["C2"] == "FAIL"


def test_boundaries():
    s = SW * np.r_[np.full(3, 0.99), np.full(27, 1.10)]                      # exactly 0.90 at or above
    r = v3.decide(s, np.full(30, 0.06), 40, SW)                              # 30 crossings, median χ = 0.06
    assert r["valid"] and r["C1"] == "PASS"
    r = v3.decide(np.full(30, SW * 1.25), np.full(30, 0.01), 40, SW)
    assert r["C2"] == "PASS"
    r = v3.decide(np.full(30, SW), np.full(30, 0.01), 40, SW)                # s_cross = switch counts as at-or-above
    assert r["C1"] == "PASS" and r["C2"] == "PASS"


def test_unresolved_by_crossings_and_by_chi():
    r = v3.decide(SW * np.full(29, 1.1), np.full(29, 0.01), 40, SW)
    assert not r["valid"] and r["C1"] == r["C2"] == "UNRESOLVED" and r["unresolved_by"] == ["crossings"]
    r = v3.decide(SW * np.full(36, 1.1), np.full(36, 0.07), 40, SW)
    assert r["unresolved_by"] == ["chi"] and r["C1"] == "UNRESOLVED" and r["C1_would_be"] == "PASS"
    r = v3.decide([], [], 40, SW)
    assert r["C1"] == "UNRESOLVED" and r["n_cross"] == 0


def test_init_rule_caps_output_scale():
    cap = 0.5 * SW
    caps = []
    for seed in range(3_000_000, 3_000_040):                                  # test seeds outside the registered range
        hid, out, s0 = v3.init_net(seed, cap, torch)
        s = float(out.weight.abs().sum())
        assert s <= cap + 1e-12
        if s0 <= cap:
            assert s == pytest.approx(s0)
        else:
            caps.append(seed); assert s == pytest.approx(cap)
    hid2, out2, _ = v3.init_net(3_000_000, cap, torch)                        # deterministic
    assert torch.equal(hid2.weight, v3.init_net(3_000_000, cap, torch)[0].weight)


def test_chi_definition():
    H = np.diag(np.r_[np.full(12, 2.0), 4.0])
    s = np.exp(0.01 * np.arange(300)); vh = np.full((300, 13), 1e-4)
    p = 1 / (math.sqrt(1e-4) + 1e-8)
    chi = v3.chi_at(s, vh, 250, H)
    assert chi == pytest.approx(0.01 / (v3.LR * 2.0 * p))
    assert v3.chi_at(s, vh, 50, H) == pytest.approx(chi)                      # window min(100, t)
    assert v3.chi_at(s, vh, None, H) is None


def test_frozen_inputs_match_v2_commit():
    fz = v3.frozen_inputs()
    assert fz["s_switch"] == pytest.approx(3.5913755424683727) and fz["q"] == pytest.approx(0.3914103370353161)
    assert fz["lambda"] == 1e-4 and fz["init_cap"] == pytest.approx(0.5 * fz["s_switch"])
    assert fz["v2_frozen_sha256"].startswith("deb8853e")


def test_train_run_records_every_step_and_respects_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "RUNS", tmp_path)
    fz = v3.frozen_inputs() | {"s_switch": 0.6}                              # small stop scale: a short machinery run
    fz["init_cap"] = 0.3
    steps, s_end, s0 = v3.train_run(3_000_001, fz)
    d = np.load(tmp_path / "run_3000001.npz")
    assert d["s"][0] <= 0.3 + 1e-12 and len(d["s"]) == steps + 1 and d["params"].shape == (steps + 1, 17)
    assert s_end >= 3 * 0.6 and (d["s"][:-1] < 3 * 0.6).all()
