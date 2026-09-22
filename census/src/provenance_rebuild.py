"""Producers for artifacts that were committed without one.

Found 2026-09-22 when the verifier's provenance check was tightened from
"some committed script mentions the artifact" to "some committed function
references it AND writes a file".  Each function below regenerates one artifact
from committed sources or committed training code; `check()` compares every
regenerated artifact with the committed one and reports the largest difference.

    python -m src.provenance_rebuild check        # regenerate in memory, compare
    python -m src.provenance_rebuild write NAME    # overwrite results/NAME

Status (see results/provenance_rebuild.md):
  r_pooled.csv   -- rebuilt EXACTLY from four committed source artifacts
  r_adamw.csv    -- regenerated EXACTLY by retraining (AdamW lr 1e-2, wd 0.01)
  r_families.csv -- w2 and solved regenerate EXACTLY by retraining; the `gstar`
                    column came from a search that is not in the repository and is
                    NOT reproduced here (see the note).
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


# --- r_pooled.csv: a join of committed artifacts ------------------------------
def r_pooled() -> pd.DataFrame:
    from .fold1d_theorem import maximum_gap

    sw = pd.read_csv(RESULTS / "fold1d_sweep.csv")
    sw = sw[(sw.activation == "sin_family") & (sw.parameter > 1.0)]
    rf = pd.read_csv(RESULTS / "fold1d_refine.csv")
    rf = rf[rf.activation == "sin_family"]
    te = pd.read_csv(RESULTS / "termination.csv")
    te = te[(te.budget == 2000) & (te.seed >= 0)]          # seed -1 is a sentinel row
    ac = pd.read_csv(RESULTS / "alpha_composition.csv")
    parts = [
        pd.DataFrame({"a": sw.parameter, "w2": sw.w2_abs, "solved": sw.solved,
                      "budget": 2000, "opt": "adam", "src": "sweep"}),
        pd.DataFrame({"a": rf.parameter, "w2": rf.w2_abs, "solved": rf.solved,
                      "budget": 2000, "opt": "adam", "src": "refine"}),
        pd.DataFrame({"a": te.a, "w2": te.w2_final.abs(),
                      "solved": te.solved.astype(str) == "True",
                      "budget": te.budget, "opt": "adam", "src": "termination"}),
        pd.DataFrame({"a": 1.25, "w2": ac.w2, "solved": ac.solved,
                      "budget": ac.budget, "opt": ac.optimizer, "src": "alpha_comp"}),
    ]
    out = pd.concat(parts, ignore_index=True)
    gs = {a: maximum_gap(float(a), resolution=600) for a in sorted(out.a.unique())}
    out["gstar"] = out.a.map(gs)
    out["R"] = out.w2 * out.gstar / 2
    return out


# --- r_adamw.csv: retrain ------------------------------------------------------
ADAMW_BUDGETS = (1_000, 2_000, 4_000, 5_000, 6_000, 8_000, 16_000, 40_000)


def _adamw_one(args):
    budget, seed = args
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data, solves

    torch.set_num_threads(1)
    f = activation("sin_family", 1.25)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = torch.optim.AdamW([theta], lr=1e-2, weight_decay=0.01)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, x, f), y).backward()
        opt.step()
    t = theta.detach()
    return {"optimizer": "adamw", "budget": budget, "seed": seed,
            "w2": abs(float(t[2])), "solved": bool(solves(t, f))}


def r_adamw(workers: int = 1) -> pd.DataFrame:
    from .fold1d_theorem import maximum_gap

    jobs = [(b, s) for b in ADAMW_BUDGETS for s in range(30)]
    with Pool(workers) as p:
        rows = p.map(_adamw_one, jobs)
    out = pd.DataFrame(rows)
    out["gstar"] = maximum_gap(1.25, resolution=600)
    out["R"] = out.w2 * out.gstar / 2
    return out


# --- r_families.csv: w2 and solved only -----------------------------------------
def _family_one(args):
    q, a, budget, seed = args
    import torch
    from torch.nn import functional as F
    from .depth_families import make_family
    from .family_onsets import solves_family
    from .fold1d import make_data

    torch.set_num_threads(1)
    f, _ = make_family(q)
    x, y = make_data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).requires_grad_(True)
    opt = torch.optim.Adam([th], lr=1e-2)
    for _ in range(budget):
        opt.zero_grad(set_to_none=True)
        w1, b1, w2, b2 = th[0], th[1], th[2], th[3]
        F.binary_cross_entropy_with_logits(w2 * f(w1 * x + b1, a) + b2, y).backward()
        opt.step()
    t = th.detach()
    return {"q": q, "a": a, "budget": budget, "seed": seed,
            "w2": abs(float(t[2])), "solved": bool(solves_family(t, f, a))}


def r_families_primary(workers: int = 1) -> pd.DataFrame:
    """The q, a, budget, seed, w2, solved columns of r_families.csv."""

    ref = pd.read_csv(RESULTS / "r_families.csv")
    jobs = [(float(r.q), float(r.a), int(r.budget), int(r.seed))
            for r in ref[["q", "a", "budget", "seed"]].itertuples(index=False)]
    with Pool(workers) as p:
        return pd.DataFrame(p.map(_family_one, jobs))


# --- comparison -----------------------------------------------------------------
def _compare(new, ref, key, cols):
    A = new.sort_values(key).reset_index(drop=True)
    B = ref.sort_values(key).reset_index(drop=True)
    out = {"rows_new": len(A), "rows_ref": len(B)}
    if len(A) != len(B):
        return out
    for c in cols:
        if A[c].dtype.kind in "fi":
            out[f"max_abs_diff_{c}"] = float(np.abs(A[c].astype(float) - B[c].astype(float)).max())
        else:
            out[f"equal_{c}"] = bool((A[c].astype(str).str.lower().values
                                      == B[c].astype(str).str.lower().values).all())
    return out


def check(workers: int = 1):
    res = {}
    ref = pd.read_csv(RESULTS / "r_pooled.csv")
    res["r_pooled.csv"] = _compare(r_pooled(), ref, ["src", "opt", "a", "budget", "w2"],
                                   ["a", "budget", "w2", "gstar", "R", "solved", "opt", "src"])
    ref = pd.read_csv(RESULTS / "r_adamw.csv")
    res["r_adamw.csv"] = _compare(r_adamw(workers), ref, ["budget", "seed"],
                                  ["w2", "gstar", "R", "solved"])
    ref = pd.read_csv(RESULTS / "r_families.csv")
    res["r_families.csv (w2, solved)"] = _compare(
        r_families_primary(workers), ref, ["q", "a", "budget", "seed"], ["w2", "solved"])
    for k, v in res.items():
        print(k, v)
    pd.DataFrame([{"artifact": k, **v} for k, v in res.items()]).to_csv(
        RESULTS / "provenance_rebuild_check.csv", index=False)
    return res


def write(name: str, workers: int = 1) -> None:
    """Overwrite results/<name> with the regenerated artifact."""
    if name == "r_pooled.csv":
        r_pooled().to_csv(RESULTS / "r_pooled.csv", index=False)
    elif name == "r_adamw.csv":
        r_adamw(workers).to_csv(RESULTS / "r_adamw.csv", index=False)
    else:
        raise ValueError(f"{name}: no full producer (see provenance_rebuild.md)")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif sys.argv[1] == "write":
        write(sys.argv[2])
