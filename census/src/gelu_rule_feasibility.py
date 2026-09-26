"""Follow-up (1), GELU prospective test: feasibility of the author's rule-scale calibration, checked before any design
or training.  The rule scale must be (a) the earliest scale at which the branch classification is final in ≥ 95% of the
existing GELU runs and (b) below the earliest crossing observed.  This computes (b)'s bound from the existing runs
(Track 3A, 200 seeds, act_general/train*_gelu.csv, crossing runs not placed at initialisation) and, for the same
seeds, the initial output scale |w₂(0)| (the run_one initialisation: torch.manual_seed(seed), U(−1, 1)⁴ in float32),
i.e. how many runs even start below that bound.  Written to act_general/gelu_rule_feasibility.json.

    python -m src.gelu_rule_feasibility
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    import torch
    from .act_posthoc2 import _runs
    r = _runs("gelu")
    all_runs = pd.concat([pd.read_csv(RESULTS / "act_general" / f) for f in ("train_gelu.csv", "train_ext_gelu.csv")])
    s_min = float(r.s_cross.min())
    w2_0 = []
    for sd in all_runs.seed:
        torch.manual_seed(int(sd))
        w2_0.append(abs(float(torch.empty(4).uniform_(-1.0, 1.0)[2])))
    w2_0 = np.array(w2_0)
    out = {"n_runs": int(len(all_runs)), "n_crossing": int(len(r)), "earliest_crossing_s": s_min,
           "crossings_below_0.05": int((r.s_cross < 0.05).sum()), "crossings_below_half_s_glob": int((r.s_cross < 0.5 * 6.6415).sum()),
           "init_abs_w2_median": float(np.median(w2_0)),
           "runs_starting_below_earliest_crossing": int((w2_0 < s_min).sum()),
           "frac_starting_below_earliest_crossing": float((w2_0 < s_min).mean())}
    (RESULTS / "act_general" / "gelu_rule_feasibility.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def _init_job(seed):
    import torch
    from .act_posthoc2 import _branch_job
    torch.manual_seed(int(seed))
    th = torch.empty(4).uniform_(-1.0, 1.0).double().numpy()
    r = _branch_job(("gelu", int(seed), th[0], th[1], th[2], th[3]))
    r["w2_init"] = float(th[2])
    return r


def finality_at_init(workers=1):
    """(a) at the only admissible rule scale: is the branch identified at initialisation (Newton on the run's own sample
    at its initial signed w₂, then continuation to the branch's switch, as act_posthoc2) the branch the run crosses on?
    Final = both switches exist and agree within 1e-4 relative, or neither has a switch.  Crossing runs only."""
    from multiprocessing import get_context
    from .act_posthoc2 import _runs, read_switches
    r = _runs("gelu")
    with get_context("spawn").Pool(workers) as pool:
        rows = list(pool.imap(_init_job, list(r.seed)))
    d = pd.DataFrame(rows).rename(columns={"s_branch": "s_branch_init", "note": "note_init"})
    b = read_switches(); b = b[b.act == "gelu"][["seed", "s_branch"]]
    d = d.merge(b, on="seed")
    both = np.isfinite(d.s_branch_init) & np.isfinite(d.s_branch)
    d["final"] = (both & (np.abs(d.s_branch_init / d.s_branch - 1) < 1e-4)) | (~np.isfinite(d.s_branch_init) & ~np.isfinite(d.s_branch))
    d.to_csv(RESULTS / "act_general" / "gelu_rule_finality_init.csv", index=False)
    f = json.loads((RESULTS / "act_general" / "gelu_rule_feasibility.json").read_text())
    f.update({"final_at_init_frac": float(d.final.mean()), "final_at_init_n": int(d.final.sum()),
              "rule_feasible": bool(d.final.mean() >= 0.95)})
    (RESULTS / "act_general" / "gelu_rule_feasibility.json").write_text(json.dumps(f, indent=1))
    print(json.dumps(f, indent=1))


if __name__ == "__main__":
    main()
    finality_at_init()
