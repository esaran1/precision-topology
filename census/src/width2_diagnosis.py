"""Item 2, step 1 (POST HOC diagnosis; rules and gate fixed in results/width2_diagnosis_gate.md before any result).

    python -m src.width2_diagnosis compute [workers]     # P2 (T2-3b, T2-3c seeds), P3 (all 30), P5 (T2-3b, T2-3c)
    python -m src.width2_diagnosis evaluate             # metrics, selection and gate
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "width2_diagnosis"
P1, P4 = 0.44507940623559955, None
ARMS = {"T2-3": (RESULTS / "asym_parts" / "train.csv", 1.0),
        "T2-3b": (RESULTS / "asym_t23b" / "train.csv", None),
        "T2-3c": (RESULTS / "asym_t23c" / "train.csv", None)}


def _crossers(arm):
    d = pd.read_csv(ARMS[arm][0])
    return d[d.crossed & ~d.placed_at_init].sort_values("seed")


def own_w1(seed, restarts=100):
    """P3: width-1 own-sample threshold on the run's training set (asymmetric windows): scan s = 10^(k/8), k = 0..13,
    then bisection to 5%."""
    from . import asym_pilot as ap
    from .asym_register import _act, _setup, training_set
    from .width2_conditional import constant_predictor_loss
    _setup(); act = _act()
    x, y = training_set(seed)
    const = constant_predictor_loss(y)

    def placed(s, sd):
        c = [r for r in ap._width1_batch(s, x, y, act, restarts, sd) if np.all(np.isfinite(r["p"]))]
        c.sort(key=lambda r: r["loss"])
        pol = [ap._polish(r, s, x, y, act, "width1") for r in c[:5]]
        b = min(pol, key=lambda r: r["loss"])
        return bool(b["placed"] and b["loss"] <= const)
    grid = [10 ** (k / 8) for k in range(14)]
    prev = None
    for i, s in enumerate(grid):
        p = placed(s, 31_000 + i)
        if p and prev is False:
            lo, hi = grid[i - 1], s
            j = 0
            while hi / lo - 1 > 0.05:
                mid = math.sqrt(lo * hi); j += 1
                lo, hi = (lo, mid) if placed(mid, 32_000 + j) else (mid, hi)
            return {"seed": seed, "s_w1_own": math.sqrt(lo * hi), "note": "ok"}
        prev = p
    return {"seed": seed, "s_w1_own": float("nan"), "note": "no unplaced-to-placed change on the grid"}


def _job(args):
    os.nice(15)
    kind, arm, seed, extra = args
    if kind == "P2":
        from .asym_posthoc2 import own_threshold
        r = own_threshold(seed)
        return {"kind": kind, "arm": arm, "seed": seed, "value": r["s_own"], "note": r["note"]}
    if kind == "P3":
        r = own_w1(seed)
        return {"kind": kind, "arm": arm, "seed": seed, "value": r["s_w1_own"], "note": r["note"]}
    # P5: branch switch from the crossing configuration (replay to the crossing, then continuation)
    from .asym_posthoc import branch_switch
    from .asym_register import _act, _setup, training_set
    from .asym_t23c import _start, _step
    from .width2_train import placed
    phi, step, s_cross = extra
    q, opt, X, Y, act = _start(seed, phi)
    for _ in range(int(step)):
        _step(q, opt, X, Y, act, phi)
    qc = q.detach().numpy().copy()
    x, y = _setup()[0], None
    x, y = training_set(seed)
    sb, note = branch_switch(qc, act, x, y, s_cross)
    rep = abs(float(abs(qc[2]) + abs(qc[5])) - s_cross) <= 1e-12
    return {"kind": kind, "arm": arm, "seed": seed, "value": sb, "note": note + ("" if rep else " (replay did not reproduce)")}


def compute(workers=1):
    from multiprocessing import get_context
    OUT.mkdir(exist_ok=True)
    f = OUT / "predictors.csv"
    done = set() if not f.exists() else {(r.kind, r.arm, r.seed) for r in pd.read_csv(f).itertuples()}
    phis = {"T2-3b": json.loads((RESULTS / "asym_t23b_frozen.json").read_text())["phi2"],
            "T2-3c": json.loads((RESULTS / "asym_t23c_frozen.json").read_text())["phi2"]}
    J = []
    for arm in ARMS:
        c = _crossers(arm)
        first10 = list(c.seed[:10])
        if arm != "T2-3":
            J += [("P2", arm, int(s), None) for s in first10]
            J += [("P5", arm, int(r.seed), (phis[arm], int(r.step), float(r.s_cross))) for r in c.itertuples()]
        J += [("P3", arm, int(s), None) for s in first10]
    J = [j for j in J if (j[0], j[1], j[2]) not in done]
    J.sort(key=lambda j: {"P5": 0, "P3": 1, "P2": 2}[j[0]])
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_job, J):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
            print(json.dumps({k: r[k] for k in ("kind", "arm", "seed", "note")}), flush=True)


def table():
    """One row per crossing run with every predictor available for it."""
    rows = []
    pr = pd.read_csv(OUT / "predictors.csv")
    own3 = pd.read_csv(RESULTS / "asym_posthoc" / "own_thresholds.csv")[["seed", "s_own"]]
    br3 = pd.read_csv(RESULTS / "asym_posthoc" / "runs.csv")[["seed", "s_branch"]]
    p4 = json.loads((RESULTS / "asym_posthoc" / "width1.json").read_text())["s_w1"]
    for arm in ARMS:
        c = _crossers(arm)[["seed", "s_cross"]].copy(); c["arm"] = arm
        c["P1"] = P1; c["P4"] = p4
        g = pr[pr.arm == arm]
        get = lambda k: dict(zip(g[g.kind == k].seed, g[g.kind == k].value))
        c["P3"] = c.seed.map(get("P3"))
        if arm == "T2-3":
            c["P2"] = c.seed.map(dict(zip(own3.seed, own3.s_own)))
            c["P5"] = c.seed.map(dict(zip(br3.seed, br3.s_branch)))
        else:
            c["P2"] = c.seed.map(get("P2")); c["P5"] = c.seed.map(get("P5"))
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


def metrics(t, cols=("P1", "P2", "P3", "P4", "P5")):
    out = []
    for p in cols:
        for arm, g in list(t.groupby("arm")) + [("pooled", t)]:
            ok = g[np.isfinite(g[p])]
            if not len(ok):
                continue
            e = np.abs(np.log(ok.s_cross / ok[p]))
            sp = float(np.corrcoef(ok.s_cross.rank(), ok[p].rank())[0, 1]) if ok[p].nunique() > 1 else float("nan")
            out.append({"predictor": p, "arm": arm, "n": len(ok), "median_abs_log_err": float(e.median()), "spearman": sp})
    return pd.DataFrame(out)


def evaluate():
    t = table()
    t.to_csv(OUT / "table.csv", index=False)
    common = t[np.isfinite(t.P2) & np.isfinite(t.P3)]
    mc = metrics(common)
    sel = mc[mc.arm == "pooled"].sort_values("median_abs_log_err").iloc[0].predictor
    mall = metrics(t)
    ms = mall[(mall.predictor == sel)]
    per_arm = ms[ms.arm != "pooled"]
    pooled_sp = float(ms[ms.arm == "pooled"].spearman.iloc[0])
    go = bool((per_arm.median_abs_log_err <= 0.10).all() and len(per_arm) == 3 and np.isfinite(pooled_sp) and pooled_sp >= 0.6)
    res = {"label": "POST HOC diagnosis; gate fixed in width2_diagnosis_gate.md", "common_subset_n": len(common),
           "selected": sel, "gate": "GO" if go else "STOP",
           "selected_per_arm": per_arm[["arm", "n", "median_abs_log_err", "spearman"]].to_dict("records"),
           "selected_pooled_spearman": pooled_sp}
    mc.to_csv(OUT / "metrics_common.csv", index=False); mall.to_csv(OUT / "metrics_all.csv", index=False)
    (OUT / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(mc.to_string(index=False)); print(mall.to_string(index=False)); print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"compute": lambda: compute(int(sys.argv[2]) if len(sys.argv) > 2 else 1), "evaluate": evaluate}[sys.argv[1]]()
