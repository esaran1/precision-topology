"""POST HOC (author's request after READY-FOR-WRITER planning, 2026-09-25): two checks on the existing Track 3A training
runs (GELU, SiLU, Mish; the registered 200-seed arm).  No new training; the registered verdicts stay as scored.

(1) Validity: the growth-to-relaxation ratio χ at crossing (stored per run by act_general.run_one: d log s/dt over the
    last ≤ 100 steps divided by lr·λ_min(D^{-1/2}HD^{-1/2}) with the run's own Adam D, the same definition as the sine
    runs' ratio) against the width-1 sine range in which the lag law was verified (Track 1A: lag_law/predictions.csv
    and compare_arms.csv).
(2) Each crossing run scored against its tracked branch's own-sample switch (as ramp.posthoc_branch): Newton on the run's
    own 400-point sample from its crossing state (w₁, b₁, b₂) at its signed w₂, then continuation in |w₂| down (if the
    branch is placed there) or up (if not) in 1% steps until the branch's gap changes sign, then bisection to 1e-6.
    Observed r_b = s_cross/s_branch − 1 against κ_act·χ (κ frozen by Track 3A).

    python -m src.act_posthoc2 [workers]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "act_general"
ACTS = ("gelu", "silu", "mish")


def _runs(name):
    d = pd.concat([pd.read_csv(OUT / f"train_{name}.csv"), pd.read_csv(OUT / f"train_ext_{name}.csv")], ignore_index=True)
    return d[(d.crossed == True) & (d.placed_at_init == False)].copy()      # as the registered scoring


def _branch_job(args):
    try:
        return _branch_job_inner(args)
    except ValueError as e:
        return {"act": args[0], "seed": int(args[1]), "s_cross": abs(args[4]), "s_branch": np.nan, "note": str(e)}


def _branch_job_inner(args):
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    from .act_general import GAct, branch_point, gplus
    from .fold1d import make_data
    name, seed, w1, b1, w2, b2 = args
    act = GAct(name)
    xt, yt = make_data(200, int(seed)); x, y = xt.double().numpy(), yt.double().numpy()
    sg = 1.0 if w2 > 0 else -1.0
    def G(z):                                            # cell cap 400,000 (act_general's original); undecided -> nan
        e = gplus(z[0], z[1], sg, act, max_cells=400_000)
        if not np.isfinite(e[0]) or not np.isfinite(e[1]):
            raise ValueError("gap undecided at the cell cap")
        return 0.5 * (e[0] + e[1])
    s_c = abs(w2)
    out = {"act": name, "seed": int(seed), "s_cross": s_c}
    z, gr = branch_point([w1, b1, b2], sg * s_c, x, y, act)
    out.update(grad_at_cross=gr, dist_to_branch=float(np.linalg.norm(z - np.array([w1, b1, b2]))))
    if gr > 1e-8:
        return {**out, "s_branch": np.nan, "note": "no branch point at the crossing"}
    placed = G(z) > 0
    step = 0.99 if placed else 1.01
    s, zs = s_c, z
    for _ in range(400):
        s_new = s * step
        z_new, gr = branch_point(zs, sg * s_new, x, y, act)
        if gr > 1e-8 or np.linalg.norm(z_new - zs) > 0.5:
            return {**out, "s_branch": np.nan, "note": f"continuation lost at s={s_new:.4f}"}
        if (G(z_new) > 0) != placed:
            lo_, hi_ = sorted([s, s_new]); zlo = z_new if s_new < s else zs
            while (hi_ - lo_) / lo_ > 1e-6:
                mid = 0.5 * (lo_ + hi_)
                zm, _ = branch_point(zlo, sg * mid, x, y, act)
                if G(zm) > 0:
                    hi_ = mid
                else:
                    lo_, zlo = mid, zm
            return {**out, "s_branch": 0.5 * (lo_ + hi_), "placed_at_cross_on_branch": placed, "note": ""}
        s, zs = s_new, z_new
    return {**out, "s_branch": np.nan, "note": "no sign change within 400 steps"}


def branch_switches(workers=1, limit=None):
    from multiprocessing import get_context
    f = OUT / "posthoc2_branch_switch.csv"
    done = set() if not f.exists() else {(r.act, r.seed) for r in pd.read_csv(f).itertuples()}
    jobs = [(n, r.seed, r.w1, r.b1, r.w2, r.b2) for n in ACTS for r in _runs(n).itertuples() if (n, r.seed) not in done]
    jobs = jobs[:limit] if limit else jobs
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_branch_job, jobs):
            pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)


def read_switches():
    """posthoc2_branch_switch.csv is appended row by row; rows without a switch have no placed_at_cross_on_branch field
    (7 fields: the note is the last), rows with one have 8.  Parsed by field count."""
    import csv
    with open(OUT / "posthoc2_branch_switch.csv") as f:
        r = csv.reader(f); hdr = next(r)
        rows = []
        for line in r:
            if len(line) == 8:
                rows.append(dict(zip(hdr, line)))
            elif len(line) == 7:
                rows.append(dict(zip(hdr[:6] + ["note"], line)) | {"placed_at_cross_on_branch": ""})
            elif len(line) == 5:          # the cell-cap rows: act, seed, s_cross, s_branch, note
                rows.append(dict(zip(["act", "seed", "s_cross", "s_branch", "note"], line)))
            else:
                raise ValueError(line)
    d = pd.DataFrame(rows)
    for c in ("seed", "s_cross", "grad_at_cross", "dist_to_branch", "s_branch"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["seed"] = d.seed.astype(int)
    return d


def main(workers=1):
    branch_switches(workers)
    pr = pd.read_csv(RESULTS / "lag_law" / "predictions.csv"); ca = pd.read_csv(RESULTS / "lag_law" / "compare_arms.csv")
    sine = {"chi_run_min": float(pr.chi.min()), "chi_run_q95": float(pr.chi.quantile(0.95)), "chi_run_max": float(pr.chi.max()),
            "chi_arm_median_max": float(ca.median_chi.max())}
    b = read_switches()
    rows = []
    for n in ACTS:
        r = _runs(n).merge(b[b.act == n][["seed", "s_branch", "note"]], on="seed", how="left")
        kap = json.loads((OUT / f"kappa_{n}_frozen.json").read_text())["kappa_adam"]
        ok = r[np.isfinite(r.s_branch) & np.isfinite(r.chi)].copy()
        ok["r_b"] = ok.s_cross / ok.s_branch - 1; ok["pred"] = kap * ok.chi
        pred, obs = float(ok.pred.median()), float(ok.r_b.median()); tol = max(0.01, 0.25 * abs(pred))
        rows.append({"act": n, "crossing_runs": len(r), "with_branch_switch": len(ok),
                     "chi_median": float(r.chi.median()), "chi_q25": float(r.chi.quantile(0.25)), "chi_q75": float(r.chi.quantile(0.75)),
                     "frac_chi_le_sine_run_max": float((r.chi <= sine["chi_run_max"]).mean()),
                     "frac_chi_le_sine_arm_median_max": float((r.chi <= sine["chi_arm_median_max"]).mean()),
                     "validity_met": bool(r.chi.median() <= sine["chi_arm_median_max"]),
                     "kappa": kap, "pred_median": pred, "obs_branch_median": obs, "tol": tol, "within": abs(obs - pred) <= tol,
                     "frac_at_or_above_branch": float((ok.r_b >= 0).mean()),
                     "n_chi_missing": int(r.chi.isna().sum()),
                     "n_no_switch_on_branch": int(r.s_branch.isna().sum()),
                     "no_switch_notes": "; ".join(f"{k}: {v}" for k, v in r[r.s_branch.isna()].note.value_counts().items()),
                     "no_switch_s_cross_max": float(r[r.s_branch.isna()].s_cross.max()) if r.s_branch.isna().any() else float("nan"),
                     "branch_over_pop_median": float((ok.s_branch / json.loads((OUT / f"kappa_{n}_frozen.json").read_text())["s_glob"]).median()),
                     "branch_over_pop_q10": float((ok.s_branch / json.loads((OUT / f"kappa_{n}_frozen.json").read_text())["s_glob"]).quantile(0.1)),
                     "branch_over_pop_q90": float((ok.s_branch / json.loads((OUT / f"kappa_{n}_frozen.json").read_text())["s_glob"]).quantile(0.9)),
                     "frac_branch_off_pop_1pct": float((np.abs(ok.s_branch / json.loads((OUT / f"kappa_{n}_frozen.json").read_text())["s_glob"] - 1) > 0.01).mean()),
                     "obs_branch_q25": float(ok.r_b.quantile(0.25)), "obs_branch_q75": float(ok.r_b.quantile(0.75)),
                     "frac_abs_rb_le_0.01": float((ok.r_b.abs() <= 0.01).mean()),
                     "frac_branch_placed_at_cross": float(r.merge(b[b.act == n][["seed", "placed_at_cross_on_branch"]], on="seed")
                                                             .placed_at_cross_on_branch.astype(str).eq("True").mean())})
        ok.to_csv(OUT / f"posthoc2_runs_{n}.csv", index=False)
    d = pd.DataFrame(rows)
    (OUT / "posthoc2_summary.json").write_text(json.dumps({"sine_width1": sine, "acts": d.to_dict("records")}, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(sine); print(d.T.to_string())
    return d


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
