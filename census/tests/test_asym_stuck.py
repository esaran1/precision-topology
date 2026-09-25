import math

import numpy as np

from src import asym_stuck as st
from src.asym_register import _act, _setup, training_set


def test_classify_rejects_placed_and_moving_states():
    _setup(); act = _act()
    x, y = training_set(600_000)
    pair = [1.6, math.pi / 2, -0.25, -1.6, math.pi / 2, -0.25, 0.0]        # placed cancelling pair: not stuck
    c = st.classify(pair, act, x, y)
    assert c["placed"] and not c["stationary_two_unit"]
    moving = [0.7, 0.3, 0.2, -1.1, 2.0, 0.3, 0.1]                           # generic state: large gradient, not stuck
    c = st.classify(moving, act, x, y)
    assert not c["placed"] and c["grad_rel"] > 1e-6 and not c["stationary_two_unit"]
