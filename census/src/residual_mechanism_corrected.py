"""Block 4b CORRECTION (author's instruction, 2026-09-25): the two teleport arms rerun from uncontaminated optimiser state.

Bug: in residual_mechanism.run_one the arms shared optimiser-state tensors (torch's load_state_dict keeps references), so
  teleport        started from the control continuation's end-of-run Adam state (not the state at the switch), and
  teleport_reset  started from the reset continuation's end state (not zeroed moments).
Here each teleport arm starts from a fresh deep copy of the checkpoint: teleport with the saved Adam state exactly as
registered ("Adam state kept"); teleport_reset with zeroed moments.  The starting state is checked bit for bit against
the checkpoint before each continuation.  Control and reset rows are the registered ones (unaffected).  Scoring uses the
registered criteria (residual_mechanism.verdict and _boot_diff), written to separate files; the registered files are
not modified.

    python -m src.residual_mechanism_corrected run [workers] | score
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "residual_mechanism_corrected"


def state_equal(sd_a, sd_b):
    """Bit-for-bit equality of two optimiser state dicts (tensors and scalars)."""
    import torch
    if sd_a["param_groups"] != sd_b["param_groups"] or set(sd_a["state"]) != set(sd_b["state"]):
        return False
    for k in sd_a["state"]:
        for f in ("exp_avg", "exp_avg_sq", "step"):
            x, y = sd_a["state"][k][f], sd_b["state"][k][f]
            x = x if torch.is_tensor(x) else torch.tensor(x)
            y = y if torch.is_tensor(y) else torch.tensor(y)
            if not torch.equal(x, y):
                return False
    return True


def run_one(args):
    import torch
    from . import lag_test2 as L2
    from . import residual_mechanism as rm
    from .sample_size import _data
    os.nice(15)
    a, seed = args
    path = L2.STATES / f"primary_a{a:.2f}_s{seed}.pt"
    ck = torch.load(path, weights_only=False)
    theta = tuple(float(v) for v in ck["theta"])
    x, y = _data(L2.N, seed)
    th_tel, diag = rm.apply_teleport(theta, rm.teleport_target(a, seed, theta, x, y), a, x, y)
    status, why = rm.checks(diag)
    out = []
    for arm in ("teleport", "teleport_reset"):
        fresh = torch.load(path, weights_only=False)["opt"]                       # an independent copy per arm
        opt_state = rm.reset_moments(fresh) if arm == "teleport_reset" else copy.deepcopy(fresh)
        ref = torch.load(path, weights_only=False)["opt"]
        start_ok = (rm.reset_ok(opt_state) if arm == "teleport_reset" else state_equal(opt_state, ref))
        row = {"a": a, "seed": seed, "arm": arm, "t_switch": ck["t_star"], "check": status, "check_reason": why,
               "start_state_matches_design": bool(start_ok), **diag}
        if status == "stop":
            row.update(cross_step=np.nan, w2_abs=np.nan, R_cross=np.nan)
        else:
            row.update(rm.continue_from(a, seed, th_tel, opt_state, ck["t_star"]))
        out.append(row)
    return out


def run(workers=3):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    reg = pd.read_csv(RESULTS / "residual_mechanism_parts.csv", float_precision="round_trip")
    keys = sorted({(round(r.a, 2), int(r.seed)) for r in reg.itertuples()})
    with get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.map(run_one, keys) for r in part]
    pd.DataFrame(rows).to_csv(OUT / "teleport_arms.csv", index=False)


def score():
    from . import residual_mechanism as rm
    from .lag_test2 import _own
    reg = pd.read_csv(RESULTS / "residual_mechanism_parts.csv", float_precision="round_trip")
    cor = pd.read_csv(OUT / "teleport_arms.csv", float_precision="round_trip")
    if not cor.start_state_matches_design.all():
        raise SystemExit("STOP: a corrected arm's starting state does not match the design")
    d = pd.concat([reg[reg.arm.isin(["control", "reset"])], cor[reg.columns.intersection(cor.columns)]], ignore_index=True)
    own = _own()
    d["R_own"] = [own.loc[(round(a, 2), int(s))] for a, s in zip(d.a, d.seed)]
    d["residual"] = d.R_cross / d.R_own - 1
    out = []
    for a, g in d.groupby(d.a.round(2)):
        excl = g[(g.arm == "teleport") & (g.check == "exclude")].seed.unique()
        keep = g[~g.seed.isin(excl)]
        both = keep.pivot_table(index="seed", columns="arm", values="residual").dropna()
        res = {}
        for arm in rm.ARMS:
            med = float(both[arm].median())
            ci = rm._boot_diff(both["control"].values, both[arm].values) if arm != "control" else (np.nan, np.nan)
            res[arm] = (med, ci[0], ci[1])
            out.append({"a": a, "arm": arm, "n_crossed": int(keep[keep.arm == arm].R_cross.notna().sum()),
                        "median_residual": med, "control_minus_arm_ci_lo": ci[0], "control_minus_arm_ci_hi": ci[1],
                        "excluded": len(excl)})
        v = rm.verdict(res)
        out.append({"a": a, "arm": "VERDICT", "verdict": v,
                    "4b-ID": "PASS" if v.startswith("ID") else "FAIL", "4b-OM": "PASS" if v.startswith("OM") else "FAIL"})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "scores.csv", index=False)
    regs = pd.read_csv(RESULTS / "residual_mechanism_scores.csv")
    pd.set_option("display.width", 220)
    print("REGISTERED (as run):"); print(regs.to_string(index=False))
    print("CORRECTED:"); print(s.to_string(index=False))


if __name__ == "__main__":
    {"run": lambda: run(int(sys.argv[2]) if len(sys.argv) > 2 else 3), "score": score}[sys.argv[1]]()
