"""Score Block A5d against its registration (results/blockA5d_prediction.md).

  python -m src.blockA5d_analyze bracket   # which a values get stage-2 seeds
  python -m src.blockA5d_analyze score     # A1-A5, writes blockA5d_scores.csv + figure data
  python -m src.blockA5d_analyze bracket k1 | score k1   # the k = 1 registration
                                           # (blockA5d_k1_prediction.md; perfect = 0 uniform AND 0 stratified)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = RESULTS / "blockA5d_runs.csv"
SCORES = RESULTS / "blockA5d_scores.csv"
CURVE = RESULTS / "blockA5d_perfect_fraction.csv"
BUDGETS = (1_000, 4_000, 16_000, 64_000)
A_VALUES = (0.9, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0)
BETA = 1.5


def use_k1():
    """Point the module at the k = 1 artifacts (registration blockA5d_k1_prediction.md)."""
    global RUNS, SCORES, CURVE, CONTROLS_OUT, K1
    RUNS = RESULTS / "blockA5d_k1_runs.csv"
    SCORES = RESULTS / "blockA5d_k1_scores.csv"
    CURVE = RESULTS / "blockA5d_k1_perfect_fraction.csv"
    CONTROLS_OUT = RESULTS / "blockA5d_k1_controls.csv"
    K1 = True


CONTROLS_OUT = RESULTS / "blockA5d_controls.csv"
K1 = False


def load():
    d = pd.read_csv(RUNS, dtype={"act": str})
    d["perfect"] = d.perfect.astype(str).str.lower() == "true"
    if K1:   # secondary: also zero errors on the 200,000-point uniform sample
        d["perfect_big"] = d.perfect & (pd.to_numeric(d.big_errors, errors="coerce") == 0)
    return d


def curve(d):
    fa = d[~d.act.isin(["relu", "gelu"])].copy()
    fa["a"] = fa.a.astype(float)
    extra = {"k_big": ("perfect_big", "sum")} if K1 else {}
    g = fa.groupby(["budget", "a"]).agg(n=("perfect", "size"), k=("perfect", "sum"),
                                        acc_med=("heldout_acc", "median"), **extra).reset_index()
    g["frac"] = g.k / g.n
    c = d[d.act.isin(["relu", "gelu"])].groupby(["budget", "act"]).agg(
        n=("perfect", "size"), k=("perfect", "sum"), acc_med=("heldout_acc", "median"),
        **({"k_big": ("perfect_big", "sum")} if K1 else {})).reset_index()
    c["frac"] = c.k / c.n
    return g, c


def onset(g, budget):
    """Smallest grid a with >= half perfect; bracketed iff a smaller grid a has < half."""
    sub = g[g.budget == budget].sort_values("a")
    above = sub[sub.frac >= 0.5]
    if above.empty:
        return None, False, None
    a_on = float(above.a.iloc[0])
    below = sub[(sub.a < a_on) & (sub.frac < 0.5)]
    prev = float(below.a.iloc[-1]) if not below.empty else None
    return a_on, not below.empty, prev


def bracket():
    g, _ = curve(load())
    acts = set()
    for B in BUDGETS:
        a_on, br, prev = onset(g, B)
        print(f"  B={B:>6}: onset {a_on}  bracketed={br}  (lower bracket {prev})")
        if br:
            acts |= {str(prev), str(a_on)}
    print("stage-2 activations:", ",".join(sorted(acts, key=float)))
    return sorted(acts, key=float)


def score():
    d = load()
    g, c = curve(d)
    g.to_csv(CURVE, index=False)
    c.to_csv(CONTROLS_OUT, index=False)
    rows = []
    # A1
    low = d[d.act.isin(["0.9", "1.0"])]
    if K1:   # violation rule: zero on uniform, stratified AND the 200,000-point sample
        viol = int(low.perfect_big.sum())
        rows.append({"id": "A1", "prediction": "zero perfect runs for a <= 1, every budget",
                     "measured": f"{int((low.heldout_errors == 0).sum())} zero-uniform, "
                                 f"{int(low.perfect.sum())} zero on uniform+stratified, {viol} also zero "
                                 f"on 200k, of {len(low)} run-budgets; min uniform errors "
                                 f"{int(low.heldout_errors.min())}, min stratified {int(low.strat_errors.min())}",
                     "pass": viol == 0})
    else:
        rows.append({"id": "A1", "prediction": "zero perfect runs for a <= 1, every budget",
                     "measured": f"{int(low.perfect.sum())} perfect of {len(low)} run-budgets; "
                                 f"min held-out errors {int(low.heldout_errors.min())}",
                     "pass": int(low.perfect.sum()) == 0})
    # A2 / A3
    ons = {}
    for B in BUDGETS:
        a_on, br, prev = onset(g, B)
        ons[B] = (a_on, br, prev)
    brk = {B: v for B, v in ons.items() if v[1]}
    rows.append({"id": "A2", "prediction": "a bracketed onset a_on(B) > 1 at every budget",
                 "measured": "; ".join(f"B={B}: a_on={v[0]} bracketed={v[1]}" for B, v in ons.items()),
                 "pass": all(v[1] and v[0] is not None and v[0] > 1 for v in ons.values())})
    seq = [ons[B][0] for B in BUDGETS if ons[B][1]]
    nonincr = all(x >= y for x, y in zip(seq[:-1], seq[1:])) if len(seq) >= 2 else None
    strict = (ons[1_000][1] and ons[64_000][1] and ons[64_000][0] < ons[1_000][0])
    rows.append({"id": "A3", "prediction": "a_on non-increasing in B, strictly lower at 64k than 1k",
                 "measured": "bracketed onsets " + ", ".join(f"{B}:{ons[B][0]}" for B in BUDGETS if ons[B][1]),
                 "pass": bool(nonincr) and bool(strict)})
    # A4
    top = d[(d.act == "3.0") & (d.budget == 64_000)]
    rows.append({"id": "A4", "prediction": "at a = 3.0, B = 64k, at least one perfect run",
                 "measured": f"{int(top.perfect.sum())} of {len(top)}",
                 "pass": int(top.perfect.sum()) >= 1})
    # A5 (exploratory, direction only)
    Bs = [B for B in BUDGETS if ons[B][1]]
    if len(Bs) >= 2:
        eps = np.array([ons[B][0] - 1.0 for B in Bs])
        gamma = -np.polyfit(np.log(Bs), np.log(eps), 1)[0]
    else:
        gamma = float("nan")
    onset_region = d[~d.act.isin(["relu", "gelu"]) & (d.a.astype(str) != "")]
    onset_region = onset_region[onset_region.a.astype(float) > 1.0]
    med = onset_region.groupby("budget").wout_fro.median()
    alpha_L = np.polyfit(np.log(med.index.values), np.log(med.values), 1)[0]
    rows.append({"id": "A5", "prediction": "gamma and alpha_L/beta (beta = 3/2) both positive (direction only)",
                 "measured": f"gamma = {gamma:.4f} from {len(Bs)} bracketed budgets; alpha_L = {alpha_L:.4f}, "
                             f"alpha_L/beta = {alpha_L / BETA:.4f}",
                 "pass": bool(np.isfinite(gamma) and gamma > 0 and alpha_L > 0)})
    out = pd.DataFrame(rows)
    out.to_csv(SCORES, index=False)
    print(out.to_string(index=False))
    print("\nperfect fraction by budget and a:")
    print(g.pivot(index="a", columns="budget", values="frac").to_string())
    print("\ncontrols:")
    print(c.pivot(index="act", columns="budget", values="frac").to_string())
    return out


if __name__ == "__main__":
    if sys.argv[2:] == ["k1"]:
        use_k1()
    {"bracket": bracket, "score": score}[sys.argv[1]]()
