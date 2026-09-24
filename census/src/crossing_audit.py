"""Crossing-detection discretisation audit (POST HOC, exploratory; no registered verdict changes).

Placement-check cadence in each experiment that measures crossing R (from the code):
  every step    phase 2b (phase2b_ordering.run; the crossings behind S3), the size test (sample_size.free_one), the lag
                test (lag_test.run_one; every φ arm, budget 32,000/φ, checks NOT scaled with the budget), the
                deconfounded lag test (lag_test2.continue_one; every φ arm, likewise) and 4b (residual_mechanism
                .continue_from);
  every 50      Block 3 (blockG_windows.train, CHECK_EVERY = 50), the Block G base-window runs that λ was fitted on (the
                same function), and the prospective own-seed test (prospective_own.train_logged, the same loop).
The gap is in every case the dense-grid gap of the experiment itself (phase2b_ordering.state or
blockG_windows.oriented_gap).

Per-step G:
  phase 2b      STORED: phase2b_checkpoints.csv logs every step once |G| < 0.05·G*, so the last negative step and the
                first positive step are both on disk; no rerun (all 38 crossing runs per a at budget 32,000).
  the others    NOT stored: 20 runs per a (a = 1.30, 1.50) are rerun with the experiment's own operations, logging G and
                |w₂| at every step up to the stored crossing.  A rerun counts only if its check-based crossing step and
                |w₂| equal the stored ones bit for bit.

Per run:
  R_check        the crossing R as the experiment recorded it (first check with G > 0);
  R_interp_step  crossing located by linear interpolation of G between the last negative and the first positive STEP;
  R_interp_check crossing located by linear interpolation of G between the last negative and the first positive CHECK
                 (the definition asked for; identical to R_interp_step for every-step checks);
  growth         R(check) − R(previous check), relative to the reference threshold: the upper bound on the upward bias.

    python -m src.crossing_audit [workers]
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_RUNS = RESULTS / "crossing_audit_runs.csv"
A_VALUES = (1.30, 1.50)
N_RUNS = 20


def interp_R(G0, G1, R0, R1):
    """R at the zero of the straight line through (0, G0) and (1, G1), G0 <= 0 < G1."""
    f = G0 / (G0 - G1)
    return R0 + f * (R1 - R0)


def summarise(steps, G, R, interval, R_ref):
    """steps, G, R: per-step series ending at (or beyond) the first positive check; checks at steps ≡ first step of
    the series' check grid (every `interval` steps, aligned so that steps % interval == offset)."""
    steps, G, R = np.asarray(steps), np.asarray(G, float), np.asarray(R, float)
    is_chk = (steps % interval == 0) if interval > 1 else np.ones(len(steps), bool)
    chk_idx = np.flatnonzero(is_chk)
    c = int([k for k in chk_idx if G[k] > 0][0])                  # first positive check
    j = int(np.flatnonzero(G > 0)[0])                             # first positive step
    pc = chk_idx[chk_idx < c]
    out = {"cross_step_check": int(steps[c]), "first_positive_step": int(steps[j]),
           "R_check": float(R[c]), "R_step": float(R[j]),
           "R_interp_step": float(interp_R(G[j - 1], G[j], R[j - 1], R[j])) if j > 0 else np.nan,
           "R_ref": float(R_ref), "check_interval": interval}
    if len(pc):
        p = int(pc[-1])
        out["R_interp_check"] = float(interp_R(G[p], G[c], R[p], R[c])) if G[p] <= 0 else np.nan
        out["growth_over_interval_rel"] = float((R[c] - R[p]) / R_ref)
    else:                                                          # crossing at the first check
        out["R_interp_check"], out["growth_over_interval_rel"] = np.nan, np.nan
    out["growth_one_step_rel"] = float((R[j] - R[j - 1]) / R_ref) if j > 0 else np.nan
    for k in ("check", "interp_step", "interp_check"):
        out[f"residual_{k}"] = out[f"R_{k}"] / R_ref - 1
    return out


# ------------------------------------------------------------------------------------------ phase 2b (stored)
def phase2b_stored():
    from .writer_inputs import ghat_cert
    gc = ghat_cert()
    own = pd.read_csv(RESULTS / "own_threshold_crossing.csv").set_index(["a", "seed"]).R_own
    keep = []
    cols = ["a", "budget", "seed", "w2", "gap", "step", "crossing"]
    for ch in pd.read_csv(RESULTS / "phase2b_checkpoints.csv", usecols=cols, chunksize=2_000_000,
                          dtype={"crossing": str}, float_precision="round_trip"):
        ch = ch[ch.budget == 32_000]
        keep.append(ch[ch.gap.abs() < 0.05])
    d = pd.concat(keep)
    runs = pd.read_csv(RESULTS / "wi_crossing_runs.csv", float_precision="round_trip")
    runs = runs[runs.budget == 32_000]
    rows = []
    for r in runs.itertuples():
        a = round(float(r.a), 2)
        g = d[(d.a.round(2) == a) & (d.seed == r.seed)].sort_values("step")
        c = g[g.crossing == "True"]
        s = int(c.step.iloc[0])
        pre = g[g.step == s - 1]
        # wi_crossing_runs.csv stores |w₂| to 15 significant digits, the checkpoint rows at full precision: relative 1e-13
        ok = (len(c) == 1 and s == int(r.cross_step) and len(pre) == 1
              and abs(float(abs(c.w2.iloc[0])) - float(r.w2_abs)) <= 1e-13 * float(r.w2_abs))
        R_ref = float(own.loc[(a, int(r.seed))]) if (a, int(r.seed)) in own.index else np.nan
        G = [float(pre.gap.iloc[0]), float(c.gap.iloc[0])] if ok else [np.nan, np.nan]
        R = [abs(float(pre.w2.iloc[0])) * gc[a] / 2, abs(float(c.w2.iloc[0])) * gc[a] / 2] if ok else [np.nan] * 2
        row = {"family": "phase 2b (S3)", "a": a, "seed": int(r.seed), "window": "base", "source": "stored",
               "stored_cross_step": int(r.cross_step), "reproduced": ok}
        if ok:
            row.update(summarise([s - 1, s], G, R, 1, R_ref if np.isfinite(R_ref) else R[1]))
            row["R_ref_kind"] = "R_own (n = 400)" if np.isfinite(R_ref) else "none (growth relative to R_check)"
            if not np.isfinite(R_ref):                     # no own threshold at this a: growth only, no residual
                row.update(R_ref=np.nan, residual_check=np.nan, residual_interp_step=np.nan,
                           residual_interp_check=np.nan)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------ reruns
def _size_job(args):
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits
    from .phase2b_ordering import state
    from .sample_size import _data, _pop
    a, seed, stored_step, stored_w2, R_ref = args
    Gp = _pop(a)[0]
    f = activation("sin_family", a)
    xt, yt = (torch.tensor(v) for v in _data(6400, seed))
    torch.manual_seed(seed)
    theta = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=1e-2)
    steps, G, R, w2c = [], [], [], None
    for step in range(1, stored_step + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(theta, xt, f), yt).backward()
        opt.step()
        st = state(theta.detach(), f, a, Gp)
        steps.append(step); G.append(st["gap"]); R.append(abs(st["w2"]) * Gp / 2)
        if st["placement_ok"]:
            w2c = abs(st["w2"])
            break
    row = {"family": "size test n = 6,400 (lag tests phi = 1, 4b control)", "a": a, "seed": seed, "window": "base",
           "source": "rerun", "stored_cross_step": stored_step, "R_ref_kind": "R_own (n = 6,400)"}
    row["reproduced"] = w2c is not None and steps[-1] == stored_step and w2c == stored_w2
    if row["reproduced"]:
        row.update(summarise(steps, G, R, 1, R_ref))
    return row


def _blockg_job(args):
    import torch
    from torch.nn import functional as F
    from .blockG_windows import CHECK_EVERY, LR, f_a, oriented_gap
    family, win, a, seed, stored_step, stored_w2, ghat, R_ref, ref_kind = args
    f = f_a(a)
    x, y = win.data(200, seed)
    torch.manual_seed(seed)
    th = torch.empty(4).uniform_(-1.0, 1.0).double().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    steps, G, R, w2c, cs = [], [], [], None, None
    for i in range(stored_step + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(th[2] * f(th[0] * x + th[1]) + th[3], y).backward()
        opt.step()
        with torch.no_grad():
            w1, b1, w2, b2 = (float(v) for v in th)
        g = oriented_gap(win, a, w1, b1)
        steps.append(i); G.append(g); R.append(abs(w2) * ghat / 2)
        if i % CHECK_EVERY == 0 and cs is None and g > 0:
            cs, w2c = i, abs(w2)
    row = {"family": family, "a": a, "seed": seed, "window": win.tag, "source": "rerun",
           "stored_cross_step": stored_step, "R_ref_kind": ref_kind}
    row["reproduced"] = cs == stored_step and w2c == stored_w2
    if row["reproduced"]:
        row.update(summarise(steps, G, R, CHECK_EVERY, R_ref))
    return row


def jobs():
    from .lag_test2 import _own
    from .prospective import windows as pwins
    from .prospective_own import windows as owins
    size, blk = [], []
    ownL = _own()
    sf = pd.read_csv(RESULTS / "sample_size_free.csv", float_precision="round_trip")
    for a in A_VALUES:
        g = sf[(sf.a.round(2) == a) & (sf.n == 6400) & sf.cross_step.notna()].sort_values("seed").head(N_RUNS)
        size += [(a, int(r.seed), int(r.cross_step), float(r.w2_abs), float(ownL.loc[(a, int(r.seed))]))
                 for r in g.itertuples()]
    gh = {(r.window, round(r.a, 2)): r.Ghat_lo for r in pd.read_csv(RESULTS / "prospective_ghat.csv").itertuples()}
    rg = {(r.window, round(r.a, 2)): r.R_glob for r in pd.read_csv(RESULTS / "prospective_rglob_all.csv").itertuples()}
    old, held = pwins()
    base = {w.tag: w for w in old}["base"]
    bg = pd.read_csv(RESULTS / "blockG_crossings.csv", float_precision="round_trip")
    for a in A_VALUES:                                     # λ calibration: Block G base window, 20 runs per a
        g = bg[(bg.window == "base") & (bg.a.round(2) == a) & bg.cross_step.notna()].sort_values("seed").head(N_RUNS)
        blk += [("Block G base window (lambda calibration)", base, a, int(r.seed), int(r.cross_step),
                 float(r.cross_w2), gh[("base", a)], rg[("base", a)], "R_glob (base)") for r in g.itertuples()]
    pr = pd.read_csv(RESULTS / "prospective_runs.csv", float_precision="round_trip")
    for a in A_VALUES:                                     # Block 3: 20 runs per a, 5 per held-out window
        for w in held:
            g = pr[(pr.window == w.tag) & (pr.a.round(2) == a) & pr.cross_step.notna()].sort_values("seed")
            g = g.head(N_RUNS // len(held))
            blk += [("Block 3 (prospective)", w, a, int(r.seed), int(r.cross_step), float(r.cross_w2),
                     gh[(w.tag, a)], rg[(w.tag, a)], "U = R_glob (window)") for r in g.itertuples()]
    vw = {w.tag: w for w in owins()}
    rn = pd.read_csv(RESULTS / "prospective_own_runs.csv", float_precision="round_trip").merge(
        pd.read_csv(RESULTS / "prospective_own_predictions.csv", float_precision="round_trip"),
        on=["window", "a", "seed"])
    for a in A_VALUES:                                     # prospective own-seed: 20 runs per a, spread over windows
        g = rn[(rn.a.round(2) == a) & rn.cross_step.notna()].sort_values(["window", "seed"])
        g = g.groupby("window").head(-(-N_RUNS // len(vw))).head(N_RUNS)
        blk += [("prospective own-seed", vw[r.window], a, int(r.seed), int(r.cross_step), float(r.cross_w2),
                 float(r.Ghat_lo), float(r.U_own), "U_own (run's own)") for r in g.itertuples()]
    return size, blk


def summary(d):
    rows = []
    for (fam, a), g in d[d.reproduced].groupby(["family", "a"], sort=False):
        rows.append({"family": fam, "a": a, "n_runs": len(g), "check_interval": int(g.check_interval.iloc[0]),
                     "R_ref": g.R_ref_kind.iloc[0],
                     "growth_over_interval_rel_median": g.growth_over_interval_rel.median(),
                     "growth_over_interval_rel_max": g.growth_over_interval_rel.max(),
                     "growth_one_step_rel_median": g.growth_one_step_rel.median(),
                     "median_residual_check": g.residual_check.median(),
                     "median_residual_interp_check": g.residual_interp_check.median(),
                     "median_residual_interp_step": g.residual_interp_step.median(),
                     "median_shift_check_minus_interp_check": (g.residual_check - g.residual_interp_check).median(),
                     "max_shift_check_minus_interp_check": (g.residual_check - g.residual_interp_check).max()})
    return pd.DataFrame(rows)


def main(workers=1):
    rows = phase2b_stored()
    print("phase 2b (stored):", sum(r["reproduced"] for r in rows), "of", len(rows), "runs located", flush=True)
    size, blk = jobs()
    with Pool(workers) as p:
        for fn, js in ((_size_job, size), (_blockg_job, blk)):
            for row in p.imap_unordered(fn, js, chunksize=1):
                rows.append(row)
                print(row["family"], row["a"], row["window"], row["seed"], "reproduced" if row["reproduced"]
                      else "NOT REPRODUCED", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT_RUNS, index=False)
    s = summary(d)
    s.to_csv(RESULTS / "crossing_audit_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(s.to_string(index=False))
    bad = d[~d.reproduced]
    if len(bad):
        print("NOT REPRODUCED:", bad[["family", "a", "seed", "window"]].to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
