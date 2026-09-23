"""Tests for the Block 4/5 replay machinery and every registered stop condition.

Required by the amendment of results/fixed_scale_prediction.md (2026-09-23) before any replay runs:
  1. the reference freeze leaves w2 bit-identical over a multi-step run, under both optimiser-state
     variants;
  2. the k = 1 replay matches the reference exactly, under both variants;
  3. each registered stop condition fires on a hand-constructed case where it should, and does not
     fire on one where it should not.
Also documents why the original check was invalid: zeroing w2's gradient does not freeze w2 under Adam.
"""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src import fixed_scale as fs
from src.fold1d import activation, make_data

SEED = 424242


def _checkpoint(steps=300):
    """A real mid-training checkpoint: full 4-parameter Adam run on the phase 2b protocol."""
    f = activation("sin_family", fs.A)
    x, y = make_data(200, SEED)
    x, y = x.double(), y.double()
    torch.manual_seed(SEED)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=fs.LR)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
    st = opt.state[th]
    return {"seed": SEED, "step": steps, "theta": th.detach().clone().numpy(),
            "exp_avg": st["exp_avg"].clone().numpy(), "exp_avg_sq": st["exp_avg_sq"].clone().numpy(),
            "adam_step": int(st["step"]), "G": 0.0}


@pytest.fixture(scope="module")
def ck():
    return _checkpoint()


@pytest.mark.parametrize("variant", ["preserved", "reset"])
def test_reference_freeze_keeps_w2_bit_identical(ck, variant):
    _, w2_traj = fs.reference_freeze(ck, variant, steps=200)
    assert all(w == float(ck["theta"][2]) for w in w2_traj)


@pytest.mark.parametrize("variant", ["preserved", "reset"])
def test_k1_replay_matches_reference_exactly(ck, variant):
    ref, _ = fs.reference_freeze(ck, variant, steps=200)
    rep = fs.replay_params(ck, variant, steps=200)
    assert np.array_equal(ref, rep)


def test_original_check_was_invalid(ck):
    """Zeroing w2's gradient alone lets Adam's stored momentum move w2 (why the original check fired)."""
    f = activation("sin_family", fs.A)
    x, y = make_data(200, SEED)
    x, y = x.double(), y.double()
    th = torch.tensor(ck["theta"], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([th], lr=fs.LR)
    opt.state[th] = {"step": torch.tensor(float(ck["adam_step"])),
                     "exp_avg": torch.tensor(ck["exp_avg"], dtype=torch.float64),
                     "exp_avg_sq": torch.tensor(ck["exp_avg_sq"], dtype=torch.float64)}
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        th.grad[2] = 0.0
        opt.step()
    assert abs(float(th.detach()[2]) - float(ck["theta"][2])) > 1e-6


# ---- stop condition 1: the rescaling changes a decision at the moment of intervention
def test_stop_decisions_does_not_fire_for_positive_k(ck):
    for k in (0.3, 1.0, 2.5):
        assert fs._decisions_preserved(ck, k)


def test_stop_decisions_fires_for_sign_flip(ck):
    assert not fs._decisions_preserved(ck, -1.0)


# ---- stop condition 2 (amended): k = 1 replay against the true-freeze reference
def test_stop_k1_does_not_fire_on_correct_replay(ck):
    fires, reasons = fs.stop_conditions(ck, k=1.0)
    assert not fires, reasons


def test_stop_k1_fires_on_broken_replay(ck, monkeypatch):
    real = fs.replay_params

    def broken(ck_, variant, steps, k=1.0, x=None, y=None):
        out = real(ck_, variant, steps, k, x, y)
        return out + 1e-6                     # a replay that has drifted from the reference
    monkeypatch.setattr(fs, "replay_params", broken)
    fires, reasons = fs.stop_conditions(ck, k=1.0)
    assert fires and any("k=1" in r for r in reasons)


def test_stop_k1_fires_on_variant_mismatch(ck):
    ref, _ = fs.reference_freeze(ck, "preserved", 25)
    rep = fs.replay_params(ck, "reset", 25)
    assert np.abs(ref - rep).max() > fs.K1_TOL


def test_stop_conditions_combined_fire_on_sign_flip(ck):
    fires, reasons = fs.stop_conditions(ck, k=-1.0)
    assert fires and "rescaling changed a decision" in reasons
