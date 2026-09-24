"""Producers for the numbers in WRITER_INPUTS.md that no earlier artifact holds.

Every R here uses the certified Ĝ as a rigorous enclosure's lower end (results/ghat_rigorous.csv, column
Ghat_cert_rigorous: Arb's lower bound at the witness of the old float Ghat_certified; author's decision 2026-09-24).  R is linear in Ĝ, so
switch points found by bracketing in |w2| (Block B) and crossings recorded as
|w2| (phase2b checkpoints) convert exactly: R = |w2|·Ĝ_cert/2.

    python -m src.writer_inputs crossings   # per-run crossings from phase2b_checkpoints.csv (2.9 GB; slow)
    python -m src.writer_inputs tables      # everything else (fast)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
CROSS_RUNS = RESULTS / "wi_crossing_runs.csv"


def ghat_cert():
    from .ghat_rigorous import ghat_R               # the rigorous Ĝ_cert (Block 2, author's decision 2026-09-24)
    return ghat_R()


def crossings():
    """One row per run that crossed: the checkpoint flagged crossing == True."""
    cols = ["a", "budget", "seed", "w2", "step", "crossing", "final_solved"]
    parts = []
    for ch in pd.read_csv(RESULTS / "phase2b_checkpoints.csv", usecols=cols,
                          chunksize=2_000_000, dtype={"crossing": str, "final_solved": str}):
        parts.append(ch[ch.crossing.str.lower() == "true"])
    d = pd.concat(parts, ignore_index=True)
    g = ghat_cert()
    d["w2_abs"] = d.w2.abs()
    d["R_cert"] = d.w2_abs * d.a.round(2).map(g) / 2
    d = d.drop(columns=["crossing", "w2"]).rename(columns={"step": "cross_step"})
    assert not d.duplicated(["a", "budget", "seed"]).any()
    d.to_csv(CROSS_RUNS, index=False)
    print(f"{len(d)} crossing runs written")


def _boot_median_ci(v, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def per_a_table():
    """Certified per-a table: R_glob, R_solve (Block B), crossings (phase2b)."""
    g = ghat_cert()
    sw = pd.read_csv(RESULTS / "blockB_switches.csv")
    fw = pd.read_csv(RESULTS / "blockB_fine_windows.csv").set_index("a")
    cr = pd.read_csv(CROSS_RUNS)
    rows = []
    for _, s in sw.iterrows():
        a = round(float(s.a), 2)
        c = cr[cr.a.round(2) == a]
        cell = c.groupby("budget").R_cert.median()
        lo, hi = _boot_median_ci(c.R_cert.values)
        w2g_fine = float(fw.loc[a, "w2_glob"]) if a in fw.index else np.nan
        rows.append({
            "a": a, "Ghat_cert": g[a],
            "w2_glob": s.w2_glob, "R_glob_cert": s.w2_glob * g[a] / 2,
            "w2_glob_fine": w2g_fine,
            "R_glob_fine_cert": w2g_fine * g[a] / 2 if np.isfinite(w2g_fine) else np.nan,
            "w2_solve": s.w2_solve, "R_solve_cert": s.w2_solve * g[a] / 2,
            "n_runs": 5 * 40, "n_crossings": len(c),
            "budgets": "/".join(str(int(b)) for b in sorted(c.budget.unique())),
            "cross_R_median_pooled": float(c.R_cert.median()),
            "cross_R_ci95_lo": float(lo), "cross_R_ci95_hi": float(hi),
            "cross_R_median_of_cell_medians": float(cell.median()),
            "cross_w2_median_pooled": float(c.w2_abs.median()),
            "cross_w2_median_of_cell_medians": float(c.groupby("budget").w2_abs.median().median()),
            "offset_vs_R_glob_pct": (float(cell.median()) / (s.w2_glob * g[a] / 2) - 1) * 100,
        })
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "wi_per_a_certified.csv", index=False)
    cvR = t.cross_R_median_of_cell_medians.std(ddof=0) / t.cross_R_median_of_cell_medians.mean()
    cvW = t.cross_w2_median_of_cell_medians.std(ddof=0) / t.cross_w2_median_of_cell_medians.mean()
    pd.DataFrame([{"cv_R_cert": cvR, "cv_w2": cvW, "n_a": len(t)}]).to_csv(
        RESULTS / "wi_per_a_cv.csv", index=False)
    return t


def phase1():
    d = pd.read_csv(RESULTS / "phase1_decomposition.csv")
    d["solved"] = d.solved.astype(str).str.lower() == "true"
    rows = []
    for prec, g in list(d.groupby("precision")) + [("both", d)]:
        rows.append({"precision": prec, "runs": len(g), "solved": int(g.solved.sum()),
                     "failed": int((~g.solved).sum()),
                     "placement": int((g.failure == "placement").sum()),
                     "bias": int((g.failure == "bias").sum()),
                     "disagreements": int((g.predicted_solved.astype(str).str.lower() == "true") != g.solved).sum()
                     if False else int(((g.predicted_solved.astype(str).str.lower() == "true") != g.solved).sum())})
    pd.DataFrame(rows).to_csv(RESULTS / "wi_phase1_reconciliation.csv", index=False)


def e2_rescore():
    """E-2 under its registered rule (blockE_redesign.md:63, 72-74)."""
    d = pd.read_csv(RESULTS / "blockE_intervene.csv")
    rows = []
    for arm in ("hold_low", "hold_high"):
        k = d[(d.arm == arm) & d.kept.notna()]
        kept = int((k.kept.astype(str).str.lower() == "true").sum())
        n = len(k)
        rows.append({"arm": arm, "intervened": n, "kept": kept, "lost": n - kept,
                     "kept_frac": kept / n, "lost_frac": (n - kept) / n})
    t = pd.DataFrame(rows).set_index("arm")
    hh = t.loc["hold_high"]
    t["registered_rule"] = ["placement LOST, falsified if kept in > 1 of 10",
                            "placement KEPT >= 9/10, falsified if lost in > 1 of 10"]
    t.loc["hold_high", "verdict_as_registered"] = "PASS" if hh.kept_frac >= 0.9 else "FAIL"
    hl = t.loc["hold_low"]
    t.loc["hold_low", "verdict_as_registered"] = "PASS" if hl.kept_frac <= 0.1 else "FAIL"
    t.reset_index().to_csv(RESULTS / "wi_e2_rescore.csv", index=False)


def blockF():
    d = pd.read_csv(RESULTS / "blockF_optimisers.csv")
    rows = []
    for opt, g in d.groupby("opt" if "opt" in d else "optimiser"):
        c = g[g.local_rate.notna()]
        rows.append({"optimiser": opt, "crossed": len(c), "median_rate": c.local_rate.median(),
                     "median_cross_R_restricted": c.cross_R.median() if "cross_R" in c else np.nan})
    pd.DataFrame(rows).to_csv(RESULTS / "wi_blockF_rates.csv", index=False)


def branch_tracking():
    d = pd.read_csv(RESULTS / "blockC_adiabatic.csv")
    d["before"] = d.before_crossing.astype(str).str.lower() == "true"
    rows = []
    for lab, g in (("before crossing", d[d.before]), ("after crossing", d[~d.before]), ("all", d)):
        q = g.rel_dist.quantile([0.25, 0.5, 0.75])
        rows.append({"split": lab, "n": len(g), "q1": q[0.25], "median": q[0.5], "q3": q[0.75],
                     "max": g.rel_dist.max()})
    pd.DataFrame(rows).to_csv(RESULTS / "wi_branch_tracking.csv", index=False)


def crossing_steps():
    """Crossing step range per (a, budget): claim 13's 1,860-6,839 is a = 1.30 at budget 8,000."""
    c = pd.read_csv(CROSS_RUNS)
    t = c.groupby([c.a.round(2), "budget"]).cross_step.agg(["size", "min", "median", "max"]).reset_index()
    t.to_csv(RESULTS / "wi_crossing_steps.csv", index=False)


def tables():
    crossing_steps()
    per_a_table()
    phase1()
    e2_rescore()
    branch_tracking()
    for f in ("wi_per_a_certified", "wi_per_a_cv", "wi_phase1_reconciliation", "wi_e2_rescore",
              "wi_branch_tracking"):
        print(f"--- {f}")
        print(pd.read_csv(RESULTS / f"{f}.csv").to_string(index=False))


if __name__ == "__main__":
    {"crossings": crossings, "tables": tables}[sys.argv[1]]()
