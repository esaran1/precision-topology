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
