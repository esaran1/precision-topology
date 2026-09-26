import numpy as np

from src import ramp2 as R2


def _cell(g, n=40, slope=1.0, el=0.1, k=0.001):
    pred = np.full(n, k * g)
    return {"gamma": g, "obs_r": slope * pred, "pred_r": pred, "n_runs": 40, "n_warmup_placed": 0, "median_eta_lambda": el}


def test_validity_drops_cells_with_large_eta_lambda():
    cells = [_cell(g) for g in (1, 2, 4, 8, 16, 32)]
    s = R2.score_setting2(cells)
    assert s["cells_valid"] == 6 and s["R1"] == "PASS" and s["R2"] == "PASS"
    cells = [_cell(g, el=(0.9 if g > 2 else 0.1)) for g in (1, 2, 4, 8, 16, 32)]   # 4 cells invalid -> only 2 remain
    s = R2.score_setting2(cells)
    assert s["cells_valid"] == 2 and s["R1"] == "UNRESOLVED"
    cells = [_cell(g, slope=2.0) for g in (1, 2, 4, 8, 16, 32)]
    assert R2.score_setting2(cells)["R1"] == "FAIL"
    cells = [_cell(g, el=float("nan")) for g in (1, 2, 4, 8, 16, 32)]              # no crossings -> invalid
    assert R2.score_setting2(cells)["R1"] == "UNRESOLVED"


def test_run_cell_from_initialisation_matches_single_seed():
    """No warm-up: s = S0·exp(γt) from step 1 and the standard initialisation; batch equals single-seed training."""
    import math
    import torch
    from src import ramp as R
    a, g, steps = 1.5, 2e-3, 300
    res = R.batch_ramp(a, (861_990, 861_991), "adam", g, R2.S0 / R.S0_FRAC, warmup=0, max_steps=steps, init="random")
    X, Y = R._data([861_991])
    p = R._init([861_991])[:, [0, 1, 3]].clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.01)
    for t in range(1, steps + 1):
        s = R2.S0 * math.exp(g * t)
        opt.zero_grad(set_to_none=True)
        tt = p[:, 0:1] * X + p[:, 1:2]
        torch.nn.functional.binary_cross_entropy_with_logits(s * (tt + a * torch.sin(tt)) + p[:, 2:3], Y).backward()
        opt.step()
        if res[1]["crossed"] and t == res[1]["step"]:
            break
    q = p.detach().numpy()[0]
    got = np.array([res[1]["w1"], res[1]["b1"], res[1]["b2"]]) if res[1]["crossed"] else np.array([res[1]["w1_end"], res[1]["b1_end"], res[1]["b2_end"]])
    assert np.max(np.abs(got - q)) < 1e-10


def test_branch_switch_from_crossing_state_matches_the_ramp_posthoc_value():
    """Continuation from a crossing state (ramp2._branch_job) finds the same switch as the committed ramp post hoc, which
    continued up from the branch point at s₀ (results/ramp/posthoc_branch_switch.csv)."""
    import pandas as pd
    from src import ramp as R
    d = R.read_runs(); b = pd.read_csv(R.OUT / "posthoc_branch_switch.csv")
    for a, seed in ((1.5, 860000), (1.3, 860003)):
        r = d[(d.a.round(2) == a) & (d.seed == seed) & (d.opt == "sgd") & (d.cell == 0) & (d.winding == (0 if a == 1.5 else -1))].iloc[0]
        want = b[(b.a.round(2) == a) & (b.seed == seed) & (b.winding == r.winding)].s_branch.iloc[0]
        got = R2._branch_job((a, seed, r.w1, r.b1, r.b2, r.s_cross))
        assert abs(got["s_branch"] / want - 1) < 1e-5, (a, seed, got, want)
