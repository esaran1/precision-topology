"""Contamination audit (author's request, 2026-09-25): does every arm that restarts from saved optimiser state start
from exactly its checkpoint?

Method: torch.optim.Adam.step is wrapped so that, at the FIRST step of each optimiser, the parameters and the Adam state
(step, exp_avg, exp_avg_sq; or "empty") are snapshotted before the step.  Each experiment's own arm code is then run in
its original arm order on sample checkpoints, sharing one in-memory checkpoint object across arms (the worst case for
aliasing; the originals shared objects within a process or a Pool chunk), with short horizons where the code allows.
Each snapshot is compared bit for bit with the start state the design specifies, computed from a FRESH load of the
checkpoint from disk.  The known contaminated case, Block 4b's original run_one, is the audit's positive control: it
must be flagged.

    python -m src.state_audit
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
SNAPS: list = []


def _install_hook():
    import torch
    orig = torch.optim.Adam.step
    if getattr(torch.optim.Adam, "_audit_hooked", False):
        return

    def step(self, *a, **k):
        if not getattr(self, "_audit_seen", False):
            self._audit_seen = True
            ps = [p for g in self.param_groups for p in g["params"]]
            SNAPS.append({"params": [p.detach().clone() for p in ps],
                          "state": [({kk: (vv.detach().clone() if torch.is_tensor(vv) else torch.tensor(float(vv)))
                                      for kk, vv in self.state[p].items()} if p in self.state and self.state[p] else None)
                                    for p in ps]})
        return orig(self, *a, **k)
    torch.optim.Adam.step = step
    torch.optim.Adam._audit_hooked = True


def _eq(a, b):
    import torch
    a = a if torch.is_tensor(a) else torch.tensor(a, dtype=torch.float64)
    b = b if torch.is_tensor(b) else torch.tensor(b, dtype=torch.float64)
    return bool(torch.equal(a.double().reshape(-1), b.double().reshape(-1)))


def _state_matches(snap_state, want):
    """want: None (empty state expected) or dict with step, exp_avg, exp_avg_sq."""
    if want is None:
        return snap_state is None or all(float(v.abs().max()) == 0 for v in snap_state.values())
    if snap_state is None:
        return False
    return all(_eq(snap_state[k], want[k]) for k in ("step", "exp_avg", "exp_avg_sq"))


def audit_fixed_scale(n_ck=4):
    """Blocks 4 and 5 (replay; LEVELS x (preserved, reset) in the original job order) and the horizon extension
    (replay_multi, with HORIZONS shortened to 3 steps)."""
    from . import fixed_scale as fs
    G_hat, w2_glob, _ = fs._ghat_and_glob()
    rows = []
    for block, sel in ((4, lambda s: fs.select4(s, G_hat, w2_glob)), (5, fs.select5)):
        cks = sel(fs._load())[:n_ck]
        fresh = sel(fs._load())[:n_ck]
        br = fs._branch_argmins(G_hat, w2_glob)
        for ck, fr in zip(cks, fresh):
            for lv in fs.LEVELS:
                for var in ("preserved", "reset"):
                    SNAPS.clear()
                    fs.replay(ck, lv, var, 3, "place" if block == 4 else "lose", G_hat, w2_glob, branch=br[lv])
                    w1, b1, w2, b2 = (float(v) for v in fr["theta"])
                    k = lv * w2_glob / abs(w2)
                    want_p = np.array([w1, b1, b2 * k])
                    want_s = ({"step": float(fr["adam_step"]), "exp_avg": fr["exp_avg"][[0, 1, 3]],
                               "exp_avg_sq": fr["exp_avg_sq"][[0, 1, 3]]} if var == "preserved" else None)
                    s = SNAPS[0]
                    rows.append({"experiment": f"Block {block}", "unit": f"seed {fr['seed']} step {fr['step']}",
                                 "arm": f"{lv}/{var}", "params_match": _eq(s["params"][0], want_p),
                                 "state_match": _state_matches(s["state"][0], want_s)})
    import importlib
    old = fs.HORIZONS
    try:
        fs.HORIZONS = (1, 2, 3)
        cks = fs.select4(fs._load(), G_hat, w2_glob)[:n_ck]
        fresh = fs.select4(fs._load(), G_hat, w2_glob)[:n_ck]
        brl = fs._branch_argmins_levels(fs.H_LEVELS, w2_glob)
        for ck, fr in zip(cks, fresh):
            for lv in fs.H_LEVELS:
                for var in ("preserved", "reset"):
                    SNAPS.clear()
                    fs.replay_multi(ck, lv, var, w2_glob, brl[lv])
                    w1, b1, w2, b2 = (float(v) for v in fr["theta"])
                    k = lv * w2_glob / abs(w2)
                    want_s = ({"step": float(fr["adam_step"]), "exp_avg": fr["exp_avg"][[0, 1, 3]],
                               "exp_avg_sq": fr["exp_avg_sq"][[0, 1, 3]]} if var == "preserved" else None)
                    s = SNAPS[0]
                    rows.append({"experiment": "Block 4 horizon extension", "unit": f"seed {fr['seed']} step {fr['step']}",
                                 "arm": f"{lv}/{var}", "params_match": _eq(s["params"][0], np.array([w1, b1, b2 * k])),
                                 "state_match": _state_matches(s["state"][0], want_s)})
    finally:
        fs.HORIZONS = old
    return rows


def audit_nogating(n_seed=3):
    """The no-gating replays (replay_ng): R2 levels x (preserved, reset), one shared checkpoint object per seed."""
    import torch
    from . import width2_nogating as wn
    from .width2_conditional import training_set
    from .width2_train import rescale
    rows = []
    for name in ("f1.30", "f1.50"):
        for seed in wn.SEEDS[:n_seed]:
            path = wn.PARTS / f"ck_{name}_{seed}.pkl"
            tr = pickle.load(open(path, "rb"))
            x, y = training_set(seed)
            for R2 in wn.R2_LEVELS:
                for var in wn.VARIANTS:
                    SNAPS.clear()
                    radius = 2 * R2 / wn.gamma2(name)
                    wn.replay_ng(tr["ck"], radius, 3, x, y, wn.act_of(name), var, record=(3,))
                    fr = pickle.load(open(path, "rb"))["ck"]
                    k = radius / (abs(fr["q"][2]) + abs(fr["q"][5]))
                    want_s = None
                    if var == "preserved" and fr["adam"]:
                        want_s = {kk: fr["adam"][kk] for kk in ("step", "exp_avg", "exp_avg_sq")}
                    s = SNAPS[0]
                    rows.append({"experiment": "no-gating (Track 7)", "unit": f"{name} seed {seed}", "arm": f"{R2}/{var}",
                                 "params_match": _eq(s["params"][0], torch.tensor(rescale(fr["q"], k))),
                                 "state_match": _state_matches(s["state"][0], want_s)})
    return rows


def audit_lag2(n_seed=2):
    """The deconfounded lag test's continuations (continue_one), φ arms in order, both rules."""
    import torch
    from . import lag_test2 as lt
    d = pd.read_csv(RESULTS / "lag_test2_runs.csv")
    rows = []
    for rule in ("primary", "tstar"):
        for a in (1.3, 1.5):
            seeds = sorted(d[(d.rule == rule) & (d.a.round(2) == a) & d.cross_step.notna()].seed.unique())[:n_seed]
            for seed in seeds:
                for fac in lt.FACTORS:
                    SNAPS.clear()
                    lt.continue_one((a, int(seed), fac, rule))
                    fr = torch.load(lt.STATES / f"{rule}_a{a:.2f}_s{int(seed)}.pt", weights_only=False)
                    st = fr["opt"]["state"][0]
                    s = SNAPS[0]
                    rows.append({"experiment": f"lag test 2 ({rule})", "unit": f"a {a} seed {int(seed)}", "arm": f"phi {fac}",
                                 "params_match": _eq(s["params"][0], fr["theta"]),
                                 "state_match": _state_matches(s["state"][0], {k: st[k] for k in ("step", "exp_avg", "exp_avg_sq")})})
    return rows


def audit_block4b(n_seed=2, corrected=False):
    """Block 4b: the original run_one (positive control, must flag the teleport arms) or the corrected arms."""
    import torch
    from . import lag_test2 as lt
    from . import residual_mechanism as rm
    from .sample_size import _data
    rows = []
    for a in (1.3, 1.5):
        d = pd.read_csv(RESULTS / "residual_mechanism_parts.csv")
        for seed in sorted(d[d.a.round(2) == a].seed.unique())[:n_seed]:
            SNAPS.clear()
            if corrected:
                from .residual_mechanism_corrected import run_one as rc
                import os
                arms = ("teleport", "teleport_reset")
                _nice = os.nice
                os.nice = lambda k: 0
                try:
                    rc((a, int(seed)))
                finally:
                    os.nice = _nice
            else:
                arms = rm.ARMS
                rm.run_one((a, int(seed)))
            fr = torch.load(lt.STATES / f"primary_a{a:.2f}_s{int(seed)}.pt", weights_only=False)
            theta0 = tuple(float(v) for v in fr["theta"])
            x, y = _data(lt.N, int(seed))
            th_tel, _ = rm.apply_teleport(theta0, rm.teleport_target(a, int(seed), theta0, x, y), a, x, y)
            st = fr["opt"]["state"][0]
            want = {k: st[k] for k in ("step", "exp_avg", "exp_avg_sq")}
            for arm, s in zip(arms, SNAPS):
                want_p = torch.tensor(th_tel if arm.startswith("teleport") else theta0, dtype=torch.float64)
                rows.append({"experiment": "Block 4b" + (" (corrected)" if corrected else " (as registered)"),
                             "unit": f"a {a} seed {int(seed)}", "arm": arm, "params_match": _eq(s["params"][0], want_p),
                             "state_match": _state_matches(s["state"][0], None if arm.endswith("reset") else want)})
    return rows


def main():
    _install_hook()
    rows = []
    rows += audit_block4b(corrected=False)          # positive control first
    rows += audit_block4b(corrected=True)
    rows += audit_fixed_scale()
    rows += audit_nogating()
    rows += audit_lag2()
    d = pd.DataFrame(rows)
    d["clean"] = d.params_match & d.state_match
    d.to_csv(RESULTS / "state_audit.csv", index=False)
    summ = d.groupby(["experiment"]).agg(arms=("clean", "size"), clean=("clean", "sum")).reset_index()
    flagged = d[~d.clean].groupby(["experiment", "arm"]).size().reset_index(name="n")
    out = {"summary": summ.to_dict("records"), "flagged": flagged.to_dict("records"),
           "positive_control_flagged": bool((~d[d.experiment == "Block 4b (as registered)"].clean).any())}
    (RESULTS / "state_audit.json").write_text(json.dumps(out, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False)); print(flagged.to_string(index=False)); print(out["positive_control_flagged"])


if __name__ == "__main__":
    main()
