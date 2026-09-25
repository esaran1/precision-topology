import math

import numpy as np
import pandas as pd

from src import asym_posthoc as ap
from src.asym_register import _act, _setup


def test_knockout_classes_on_constructed_configurations():
    _setup(); act = _act()
    pair = [1.6, math.pi / 2, 0.5, -1.6, math.pi / 2, 0.5, 0.0]           # ramp-cancelling cosine pair
    assert ap.features(pair, act)["knockout"] == "pair"
    sb = pd.read_csv(ap.OUT / "single_unit_branch.csv")                  # a placed single-unit-branch configuration
    assert sb.placed.iloc[-1]
    import json
    # a placed width-1 unit plus a constant (alpha = 0) second unit: single-unit; two copies: redundant
    from src import asym_pilot as apl
    x, y = _setup()
    c = sorted(apl._width1_batch(10.0, x, y, act, 50, 1), key=lambda r: r["loss"])[0]
    q = apl._polish(c, 10.0, x, y, act, "width1")["q"]
    assert apl._polish(c, 10.0, x, y, act, "width1")["placed"]
    sg = np.sign(q[4])
    single = [q[0], q[1], sg * 0.9, 0.0, 0.3, 0.1, 0.0]
    assert ap.features(single, act)["knockout"] == "single-unit"
    red = [q[0], q[1], sg * 0.5, q[0], q[1], sg * 0.5, 0.0]
    assert ap.features(red, act)["knockout"] == "redundant"
    flat = [0.0, 0.3, 0.5, 0.0, 0.2, 0.5, 0.0]                              # fail case: nothing placed
    assert ap.features(flat, act)["knockout"] == "unplaced"
