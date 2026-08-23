"""Tests for the threshold-sweep analysis, especially the exact intervals."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.threshold_report import (
    clopper_pearson,
    condition_table,
    transition_interval,
    zero_rate_upper_bound,
)


def test_clopper_pearson_zero_successes_matches_closed_form():
    """With x = 0 the exact upper bound solves (1-p)^n = alpha/2."""

    for n in (20, 160, 1000):
        _, upper = clopper_pearson(0, n)
        assert math.isclose((1.0 - upper) ** n, 0.025, rel_tol=1e-6)


def test_clopper_pearson_all_successes_symmetric():
    lower, upper = clopper_pearson(160, 160)
    assert upper == 1.0
    assert math.isclose((lower) ** 160, 0.025, rel_tol=1e-6)


def test_clopper_pearson_known_value():
    """x=5, n=50 gives the textbook interval (0.0333, 0.2181) to 3 decimals."""

    lower, upper = clopper_pearson(5, 50)
    assert abs(lower - 0.0333) < 5e-4
    assert abs(upper - 0.2181) < 5e-4


def test_zero_rate_upper_bound_rule_of_three():
    """Exact bound is close to, and slightly below, 3/n for moderate n."""

    for n in (100, 1000, 3330):
        bound = zero_rate_upper_bound(n)
        assert bound < 3.0 / n
        assert bound > 2.9 / n


def test_zero_rate_upper_bound_exact():
    bound = zero_rate_upper_bound(100)
    assert math.isclose((1.0 - bound) ** 100, 0.05, rel_tol=1e-9)


def _fake_frame() -> pd.DataFrame:
    rows = []
    for parameter, separations in ((0.9, 0), (1.0, 0), (1.1, 3), (1.5, 10)):
        for index in range(20):
            rows.append(
                {
                    "family": "A",
                    "activation": "sin_family",
                    "parameter": parameter,
                    "monotonic": parameter <= 1.0,
                    "width": 3,
                    "perfect_eval": index < separations,
                    "eval_errors": 0 if index < separations else 30 + index,
                    "final_train_loss": 0.1,
                    "final_gradient_norm": 1.0,
                    "inactive_unit_fraction": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_condition_table_counts():
    table = condition_table(_fake_frame(), width=3)
    assert list(table.separations) == [0, 0, 3, 10]
    assert (table.n == 20).all()
    zero_rows = table[table.separations == 0]
    assert (zero_rows.ci_low == 0.0).all()
    assert (zero_rows.ci_high > 0.0).all()


def test_transition_interval_at_grid_resolution():
    table = condition_table(_fake_frame(), width=3)
    last_zero, first_positive = transition_interval(table, "A")
    assert last_zero == 1.0
    assert first_positive == 1.1


def test_transition_interval_no_positive():
    frame = _fake_frame()
    frame["perfect_eval"] = False
    table = condition_table(frame, width=3)
    last_zero, first_positive = transition_interval(table, "A")
    assert last_zero == 1.5
    assert first_positive is None


def test_clopper_pearson_rejects_bad_input():
    with pytest.raises(ValueError):
        clopper_pearson(5, 0)
    with pytest.raises(ValueError):
        clopper_pearson(-1, 10)
