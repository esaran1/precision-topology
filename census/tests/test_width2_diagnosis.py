import numpy as np
import pandas as pd

from src import width2_diagnosis as wd


def test_metrics_constant_has_no_spearman_and_exact_predictor_has_zero_error():
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(0, 0.5, 60))
    t = pd.DataFrame({"arm": np.repeat(["A", "B", "C"], 20), "s_cross": s, "P1": 0.5, "P2": s * 1.02,
                      "P3": np.nan, "P4": 5.0, "P5": s[::-1]})
    m = wd.metrics(t, cols=("P1", "P2", "P5")).set_index(["predictor", "arm"])
    assert np.isnan(m.loc[("P1", "pooled"), "spearman"])                       # constant: undefined, cannot pass
    assert abs(m.loc[("P2", "pooled"), "median_abs_log_err"] - np.log(1.02)) < 1e-12
    assert m.loc[("P2", "pooled"), "spearman"] > 0.99
    assert m.loc[("P5", "pooled"), "spearman"] < 0.5                            # an unrelated predictor fails
