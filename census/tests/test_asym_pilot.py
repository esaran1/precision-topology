import numpy as np
import pandas as pd

from src import asym_pilot as ap


def test_population_delta0_is_base():
    from src.width2_conditional import population
    x0, y0 = population()
    x, y = ap.population(0.0)
    assert np.array_equal(np.sort(x0[y0 == 0]), np.sort(x[y == 0]))
    assert np.array_equal(np.sort(x0[y0 == 1]), np.sort(x[y == 1]))


def test_population_delta08_counts_and_ramp():
    x, y = ap.population(0.8)
    assert (y == 1).sum() == 400 and (x[y == 1] < 0).sum() == 133
    assert x[y == 1].max() == 2.8 and x[y == 1].min() == -2.0
    m = x[y == 1].mean() - x[y == 0].mean()
    assert abs(m - 0.803) < 1e-3


def test_set_windows_delta0_leaves_gaps_unchanged_and_delta_changes_them():
    from src import width2_geometry as g
    th, v = np.array([-1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0])      # decreasing: min_O at the right end
    act = g.Act("fa", 1.3)
    ap.set_windows(0.0)                        # other tests may have left the asymmetric windows set globally
    ref = g.gaps(th, v, act)["G+"]
    ap.set_windows(0.0)
    assert g.gaps(th, v, act)["G+"] == ref
    ap.set_windows(0.8)
    assert g.OUTER[1] == (1.2, 2.8) and g.gaps(th, v, act)["G+"] != ref
    ap.set_windows(0.0)


def _rows(placed, und=None):
    s = np.array(ap.SCALES)
    return pd.DataFrame({"s": s, "placed": placed, "undecided": und if und is not None else [False] * len(s)})


def test_decide_pass_and_fail_cases():
    n = len(ap.SCALES)
    assert ap.decide(_rows([False] * 6 + [True] * 7))["decision"] == "go"
    assert ap.decide(_rows([False] * 6 + [True] * 7, [False] * 5 + [True] + [False] * 7))["decision"] == "no-go"
    assert ap.decide(_rows([False] * 5 + [True, False] + [True] * 6))["decision"] == "no-go"      # alternation
    assert ap.decide(_rows([True] * n))["decision"] == "no-go"                                    # never unplaced
    assert ap.decide(_rows([False] * n))["decision"] == "no-go"                                   # never placed
    assert ap.decide(_rows([False] * 1 + [True] * 12))["decision"] == "no-go"                     # span < 3 below
    assert ap.decide(_rows([False] * 11 + [True] * 2))["decision"] == "no-go"                     # 1.78 < 3 above
    assert ap.decide(_rows([False] * 10 + [True] * 3))["decision"] == "go"                        # 3.16 >= 3
    assert ap.decide(_rows([False] * 6 + [True] * 7).iloc[:6])["decision"] == "incomplete"
