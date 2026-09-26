import math

from src import t2c


def test_choose_rule_scale_pass_and_fail():
    scales = [0.005, 0.01, 0.02, 0.05, 0.1]
    # final in >= 95% at 0.01 and below the earliest crossing 0.016: registered at 0.01
    r = t2c.choose_rule_scale(scales, [0.5, 0.96, 0.97, 0.99, 0.99], 0.016)
    assert r["decision"] == "REGISTER" and r["rule_scale"] == 0.01 and r["earliest_final_scale_any"] == 0.01
    # finality reached only above the earliest crossing: STOP, and the finality-only scale is reported
    r = t2c.choose_rule_scale(scales, [0.1, 0.5, 0.96, 0.99, 0.99], 0.016)
    assert r["decision"] == "STOP" and r["rule_scale"] is None and r["earliest_final_scale_any"] == 0.02
    # never final in 95%: STOP with no finality scale
    r = t2c.choose_rule_scale(scales, [0.1, 0.5, 0.7, 0.76, 0.6], 0.016)
    assert r["decision"] == "STOP" and r["earliest_final_scale_any"] is None
    # unsorted input is handled
    r = t2c.choose_rule_scale(scales[::-1], [0.99, 0.99, 0.97, 0.96, 0.5], 0.016)
    assert r["rule_scale"] == 0.01


def test_available_requires_passage_before_crossing():
    steps = {0.01: 5, 0.02: 40}
    assert t2c.available(steps, 30, 0.01)
    assert not t2c.available(steps, 30, 0.02)          # passage after the crossing
    assert not t2c.available(steps, 40, 0.02)          # passage at the crossing step is not before it
    assert not t2c.available(steps, 30, 0.005)         # never passed (started above it)
    assert t2c.available(steps, math.inf, 0.02)        # non-crossing run
