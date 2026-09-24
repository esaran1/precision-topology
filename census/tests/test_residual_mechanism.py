"""Block 4b validity checks and scorer on constructed pass and fail cases (before any intervention is run)."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

import src.residual_mechanism as rm
from src import lag_test2 as L2
from src.sample_size import _data

A, SEED = 1.30, 300_000


@pytest.fixture(scope="module")
def state():
    ck = torch.load(L2.STATES / f"primary_a{A:.2f}_s{SEED}.pt", weights_only=False)
    x, y = _data(L2.N, SEED)
    return ck, tuple(float(v) for v in ck["theta"]), x, y


def test_reset_zeroes_everything_and_a_partial_reset_fails(state):
    ck = state[0]
    assert rm.reset_ok(rm.reset_moments(ck["opt"]))
    import copy
    part = copy.deepcopy(ck["opt"])
    for v in part["state"].values():
        v["exp_avg"] = torch.zeros_like(v["exp_avg"])          # exp_avg only: a partial reset
    assert not rm.reset_ok(part)


def test_teleport_lands_on_the_minimiser_on_the_same_branch(state):
    ck, theta, x, y = state
    tgt = rm.teleport_target(A, SEED, theta, x, y)
    th, diag = rm.apply_teleport(theta, tgt, A, x, y)
    assert diag["branch_after"] == diag["branch_before"]
    assert diag["grad_norm"] <= rm.GRAD_TOL and diag["b2_residual"] <= rm.B_TOL
    assert rm.checks(diag)[0] in ("ok", "exclude")


def test_a_target_on_the_other_branch_stops(state):
    ck, theta, x, y = state
    mw, mb, br = rm.teleport_target(A, SEED, theta, x, y)
    _, diag = rm.apply_teleport(theta, (-mw, mb, br), A, x, y)
    assert rm.checks(diag) == ("stop", "teleport changed the mirror branch")


def test_a_target_off_the_minimiser_stops(state):
    ck, theta, x, y = state
    mw, mb, br = rm.teleport_target(A, SEED, theta, x, y)
    _, diag = rm.apply_teleport(theta, (mw + 0.05, mb, br), A, x, y)
    assert rm.checks(diag)[0] == "stop"


def test_a_planted_placed_target_is_excluded():
    diag = {"branch_before": 1, "branch_after": 1, "grad_norm": 0.0, "b2_residual": 0.0, "G_after": 0.01}
    assert rm.checks(diag)[0] == "exclude"


def test_control_reproduces_the_lag_test_and_a_one_ulp_change_fails(state):
    ck, theta, x, y = state
    out = rm.continue_from(A, SEED, theta, ck["opt"], ck["t_star"])
    ref = pd.read_csv(rm.RESULTS / "lag_test2_runs.csv", float_precision="round_trip")
    ref = ref[(ref.rule == "primary") & (ref.factor == 1.0)]
    d = pd.DataFrame([{"a": A, "seed": SEED, "arm": "control", **out}])
    assert rm.control_reproduces(d, ref)
    d2 = d.copy(); d2.loc[0, "w2_abs"] = np.nextafter(d2.loc[0, "w2_abs"], np.inf)
    assert not rm.control_reproduces(d2, ref)
    # a state perturbed by one ulp in w1 does not reproduce
    th = (np.nextafter(theta[0], np.inf),) + theta[1:]
    out2 = rm.continue_from(A, SEED, th, ck["opt"], ck["t_star"])
    d3 = pd.DataFrame([{"a": A, "seed": SEED, "arm": "control", **out2}])
    assert not rm.control_reproduces(d3, ref)


@pytest.mark.parametrize("res, want", [
    ({"control": (0.03, None, None), "teleport": (0.01, 0.01, 0.03), "reset": (0.028, -0.002, 0.005),
      "teleport_reset": (0.01, 0.01, 0.03)}, "ID"),
    ({"control": (0.03, None, None), "teleport": (0.028, -0.002, 0.005), "reset": (0.01, 0.01, 0.03),
      "teleport_reset": (0.01, 0.01, 0.03)}, "OM"),
    ({"control": (0.03, None, None), "teleport": (0.029, -0.002, 0.004), "reset": (0.029, -0.002, 0.004),
      "teleport_reset": (0.028, -0.002, 0.004)}, "competing"),
    ({"control": (0.03, None, None), "teleport": (0.01, 0.01, 0.03), "reset": (0.012, 0.01, 0.03),
      "teleport_reset": (0.005, 0.01, 0.03)}, "both"),
])
def test_scorer_classifies_constructed_patterns(res, want):
    assert rm.verdict(res).startswith(want)
