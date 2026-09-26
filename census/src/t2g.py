"""Track 2 (final round): a prospective two-class width-2 prediction, GATED.  A NEW PREDICTOR DESIGNED FROM THE DIAGNOSIS OF
EARLIER FAILURES (T2-3d; Track 2B).  T2-3 stays FAIL, T2-3b and T2-3c stay UNRESOLVED, T2-3d stays FAIL, whatever this shows.

Setting: a = 1.30, Δ = 0.4 (asymmetric windows), matched initialisation (k = 0.04209), each run on its own training set.

Everything below was committed BEFORE the classifier was evaluated on any run.

Classifier (computed from the initial state alone, by a pre-computation; never from training):
  A. fixed-v relaxation: from the run's initial state q₀ (init_params(seed), v scaled by k), Adam (lr 0.01, fresh state) on the
     hidden coordinates z = (α₁, β₁, α₂, β₂, b) with the output weights v held at v₀, for RELAX_STEPS = 2,000 steps, placement
     (exact G₊ > 0, width2_train.placed) checked at every step.  Placed at any step → class EARLY, predicted crossing s₀ = ‖v₀‖₁.
  B. otherwise: the basin it relaxed into, z*(v₀) (damped Newton at fixed v₀ from the relaxed state, a minimum), followed by
     Newton continuation (predictor–corrector, steps ≤ 2%) along the ray v = s·v₀/s₀ upward from s₀ to S_CUT = 1.0.  If its gap
     G₊ turns positive at s_sw ≤ S_CUT → class EARLY, predicted crossing s_sw.  Otherwise (never placed up to S_CUT, or the
     branch is lost / jumps before a switch) → class LATE, predicted crossing = the run's width-1 own-sample threshold
     (width2_diagnosis.own_w1, as T2-3d: 100 restarts, scan 10^(k/8), bisection to 5%).
Ground truth (observed): EARLY iff the run crossed and s_cross < S_CUT = 1.0; LATE otherwise (a non-crossing run is LATE).
  (Disclosed: 1.0 lies in the gap of the existing crossing scales, 0.52 < gap < 4.16, seen in T2-3d / Track 2B.)
Runs placed at step 0 are excluded (as T2-3..T2-3d) and counted.

STEP 1, GATE (POST HOC, existing slowed runs T2-3b, T2-3c, T2-3d; 239 runs not placed at step 0):
  PASS iff classification accuracy ≥ 0.95 AND median |log(s_cross / s_w1own)| ≤ 0.10 over the predicted-LATE crossing runs
  (at least 20 such runs with a defined own threshold; otherwise the gate is not passed).  Fail → stop; nothing registered.
STEP 2 (only on a gate pass): REGISTERED test on fresh seeds 870,000–870,079 (extension 870,080–870,159 if fewer than 40 cross)
  at T2-3c's φ₂ = 0.01778, budget 32,000, with the classifier and every prediction frozen and hashed before any training.
  (i) accuracy ≥ 0.90; (ii) median |log(s_cross/s_pred)| ≤ 0.10 over crossing runs with a defined prediction;
  (iii) mean(|log(s_cross/s_pred)| − |log(s_cross/0.44508)|), paired run-level bootstrap 95% interval (10,000) entirely below 0.
  Validity: ≥ 40 crossings (else UNRESOLVED); > 20% placed at step 0 → STOP.  PASS iff (i), (ii) and (iii).

    python -m src.t2g classify <arm|fresh>     # the classifier on existing arms (or fresh seeds); resumable
    python -m src.t2g own <arm|fresh>          # width-1 own thresholds for predicted-LATE runs (all runs for fresh)
    python -m src.t2g gate
    python -m src.t2g freeze | train | score   # step 2 only
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "t2g"
RELAX_STEPS, S_CUT, STEP_MAX, STEP_MIN = 2000, 1.0, 0.02, 1e-4
PHI2, BUDGET, S_GLOB_W2 = 0.01778279410038923, 32_000, 0.44507940623559955
FRESH = tuple(range(870_000, 870_080))
FRESH_EXT = tuple(range(870_080, 870_160))
GATE_ACC, GATE_ERR, GATE_MIN_LATE = 0.95, 0.10, 20
REG_ACC, REG_ERR, MIN_CROSS, STEP0_STOP = 0.90, 0.10, 40, 0.20
EXISTING = {"T2-3b": RESULTS / "asym_t23b" / "train.csv", "T2-3c": RESULTS / "asym_t23c" / "train.csv",
            "T2-3d": RESULTS / "asym_t23d" / "train.csv"}
IDX = [0, 1, 3, 4, 6]
CLS_COLS = ["arm", "seed", "s0", "placed_at_init", "relax_placed_step", "branch_status", "s_switch", "class_pred",
            "s_pred_early"]


def _nice():
    try:
        os.nice(15)
    except OSError:
        pass


def _k():
    return json.loads((RESULTS / "asym_frozen.json").read_text())["k"]


# ------------------------------------------------------------------------------------------ rules (pure)
def truth(crossed, s_cross, cut=S_CUT):
    return "EARLY" if bool(crossed) and np.isfinite(s_cross) and s_cross < cut else "LATE"


def prediction(class_pred, s_pred_early, s_own):
    return float(s_pred_early) if class_pred == "EARLY" else float(s_own)


def gate_decision(df):
    """df: one row per run (not placed at step 0) with class_pred, truth, crossed, s_cross, s_own."""
    acc = float((df.class_pred == df.truth).mean())
    late = df[(df.class_pred == "LATE") & df.crossed.astype(bool) & np.isfinite(df.s_own) & np.isfinite(df.s_cross)]
    err = np.abs(np.log(late.s_cross / late.s_own))
    med = float(np.median(err)) if len(late) else float("nan")
    ok = acc >= GATE_ACC and len(late) >= GATE_MIN_LATE and med <= GATE_ERR
    return {"n_runs": int(len(df)), "accuracy": acc, "n_pred_late_scored": int(len(late)), "median_abs_log_err_pred_late": med,
            "confusion": {f"pred {p} / true {t}": int(((df.class_pred == p) & (df.truth == t)).sum())
                          for p in ("EARLY", "LATE") for t in ("EARLY", "LATE")},
            "gate": "PASS" if ok else "FAIL"}


def score_registered(df, n_placed_init, n_total, s_glob=S_GLOB_W2, n_boot=10_000, seed=0):
    """df: runs not placed at step 0, with class_pred, truth, crossed, s_cross, s_pred."""
    if n_total and n_placed_init / n_total > STEP0_STOP:
        return {"verdict": "STOP (more than 20% placed at step 0)"}
    c = df[df.crossed.astype(bool) & np.isfinite(df.s_cross)]
    if len(c) < MIN_CROSS:
        return {"verdict": "UNRESOLVED (fewer than 40 crossings)", "n_cross": int(len(c))}
    acc = float((df.class_pred == df.truth).mean())
    ok = c[np.isfinite(c.s_pred) & (c.s_pred > 0)]
    e = np.abs(np.log(ok.s_cross / ok.s_pred)); ep = np.abs(np.log(ok.s_cross / s_glob))
    med = float(np.median(e))
    d = (e - ep).values
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    c1, c2, c3 = acc >= REG_ACC, med <= REG_ERR, hi < 0
    return {"verdict": "PASS" if (c1 and c2 and c3) else "FAIL", "n_runs": int(len(df)), "n_cross": int(len(c)),
            "n_scored": int(len(ok)), "accuracy": acc, "criterion_i": c1, "median_abs_log_err": med, "criterion_ii": c2,
            "diff_mean": float(d.mean()), "diff_lo": lo, "diff_hi": hi, "criterion_iii": c3,
            "median_abs_log_err_pop": float(np.median(ep))}


# ------------------------------------------------------------------------------------------ the classifier
def classify(seed, k=None, relax_steps=RELAX_STEPS, s_cut=S_CUT):
    import torch
    from .asym_register import _act, _setup, training_set
    from .width2_lag import _dz_ds, _is_min, branch_point, corrector_ok, gap
    from .width2_train import LR, init_params, logits, placed
    torch.set_num_threads(1)
    _setup(); act = _act()
    k = _k() if k is None else k
    x, y = training_set(seed)
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    q0 = init_params(seed); q0[2] *= k; q0[5] *= k
    s0 = float(q0[2].abs() + q0[5].abs())
    row = {"seed": seed, "s0": s0, "placed_at_init": bool(placed(q0.numpy(), act)[0]), "relax_placed_step": np.nan,
           "branch_status": "", "s_switch": np.nan}
    if row["placed_at_init"]:
        return {**row, "class_pred": "EXCLUDED", "s_pred_early": np.nan}
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)
    for t in range(1, relax_steps + 1):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(logits(q, X, act), Y).backward()
        q.grad[[2, 5]] = 0.0
        opt.step()
        if placed(q.detach().numpy(), act)[0]:
            row.update({"relax_placed_step": t, "branch_status": "placed_in_relaxation"})
            return {**row, "class_pred": "EARLY", "s_pred_early": s0}
    v0 = q0[[2, 5]].numpy(); vt = v0 / s0
    z, gm, H, conv = branch_point(q.detach().numpy()[IDX], v0, x, y, act)
    if not (conv and _is_min(H)):
        return {**row, "branch_status": "no_basin", "class_pred": "LATE", "s_pred_early": np.nan}
    g = gap(z, vt, act)
    if g > 0:
        return {**row, "branch_status": "basin_placed_at_s0", "s_switch": s0, "class_pred": "EARLY", "s_pred_early": s0}
    s_prev, h = s0, STEP_MAX
    while True:
        s = min(s_prev * math.exp(h), s_cut)
        if s <= s_prev:
            return {**row, "branch_status": "unplaced_to_cut", "class_pred": "LATE", "s_pred_early": np.nan}
        z_pred = z + _dz_ds(z, s_prev, vt, x, y, act, H) * (s - s_prev)
        zn, gm, Hn, conv = branch_point(z_pred, s * vt, x, y, act)
        if not (conv and _is_min(Hn) and corrector_ok(zn, z_pred, z)):
            if h / 2 < STEP_MIN:
                return {**row, "branch_status": "branch_lost_or_jump", "class_pred": "LATE", "s_pred_early": np.nan}
            h /= 2
            continue
        gn = gap(zn, vt, act)
        if gn > 0:
            lo, hi, zl = s_prev, s, z
            for _ in range(60):
                if hi / lo - 1 < 1e-8:
                    break
                mid = math.sqrt(lo * hi)
                zm, _, _, _ = branch_point(zl, mid * vt, x, y, act)
                if gap(zm, vt, act) > 0:
                    hi = mid
                else:
                    lo, zl = mid, zm
            s_sw = math.sqrt(lo * hi)
            return {**row, "branch_status": "switch_below_cut", "s_switch": s_sw, "class_pred": "EARLY", "s_pred_early": s_sw}
        s_prev, z, H, h = s, zn, Hn, min(STEP_MAX, 2 * h)


def _seeds(which):
    if which == "fresh":
        return list(FRESH)
    if which == "fresh_ext":
        return list(FRESH_EXT)
    d = pd.read_csv(EXISTING[which])
    return sorted(int(s) for s in d.seed)


def run_classify(which, seeds=None):
    _nice()
    OUT.mkdir(exist_ok=True)
    f = OUT / f"classify_{which}.csv"
    done = set() if not f.exists() else set(pd.read_csv(f).seed)
    for sd in (seeds or _seeds(which)):
        if sd in done:
            continue
        r = {"arm": which, **classify(sd)}
        pd.DataFrame([r]).reindex(columns=CLS_COLS).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(json.dumps({"arm": which, "seed": sd, "done": True}), flush=True)     # nothing outcome-bearing printed


def run_own(which):
    """Width-1 own thresholds: existing arms, predicted-LATE runs only (T2-3d: the frozen values are reused); fresh: all."""
    _nice()
    from .width2_diagnosis import own_w1
    f = OUT / f"own_{which}.csv"
    done = set() if not f.exists() else set(pd.read_csv(f).seed)
    if which in ("fresh", "fresh_ext"):
        seeds = _seeds(which)
    else:
        c = pd.read_csv(OUT / f"classify_{which}.csv")
        seeds = sorted(int(s) for s in c[c.class_pred == "LATE"].seed)
        if which == "T2-3d":
            fr = pd.read_csv(RESULTS / "asym_t23d_own_frozen.csv")
            rows = [{"seed": int(r.seed), "s_w1_own": r.s_w1_own, "note": "frozen T2-3d value"}
                    for r in fr.itertuples() if int(r.seed) in set(seeds) and int(r.seed) not in done]
            if rows:
                pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
            return
    for sd in seeds:
        if sd in done:
            continue
        r = own_w1(sd)
        pd.DataFrame([{"seed": sd, "s_w1_own": r["s_w1_own"], "note": r["note"]}]).to_csv(
            f, mode="a", header=not f.exists(), index=False)
        print(json.dumps({"own": which, "seed": sd, "done": True}), flush=True)


def gate():
    rows = []
    for arm, path in EXISTING.items():
        tr = pd.read_csv(path)
        c = pd.read_csv(OUT / f"classify_{arm}.csv")
        own = pd.read_csv(OUT / f"own_{arm}.csv") if (OUT / f"own_{arm}.csv").exists() else pd.DataFrame(columns=["seed", "s_w1_own"])
        m = tr.merge(c[["seed", "class_pred", "s_pred_early", "branch_status", "s0"]], on="seed").merge(
            own[["seed", "s_w1_own"]], on="seed", how="left")
        m = m[~m.placed_at_init.astype(bool)].copy()
        m["arm"] = arm
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    d["truth"] = [truth(c_, s_) for c_, s_ in zip(d.crossed, d.s_cross)]
    d["s_own"] = d.s_w1_own.astype(float)
    d.to_csv(OUT / "gate_runs.csv", index=False)
    out = {"label": "Step 1 gate, POST HOC on existing slowed runs (T2-3b, T2-3c, T2-3d); new predictor designed from the "
                    "diagnosis of earlier failures", **gate_decision(d),
           "per_arm": {a: gate_decision(g) for a, g in d.groupby("arm")},
           "excluded_mismatch": int(((d.class_pred == "EXCLUDED")).sum())}
    (OUT / "gate.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out


# ------------------------------------------------------------------------------------------ step 2
FROZEN = OUT / "fresh_frozen.csv"


def freeze(which="fresh"):
    """Freeze classes and predictions (fresh, or the registered extension 'fresh_ext') with SHA-256, before any training."""
    global FROZEN
    c = pd.read_csv(OUT / f"classify_{which}.csv")
    own = pd.read_csv(OUT / f"own_{which}.csv")
    FROZEN = OUT / f"{which}_frozen.csv"
    m = c.merge(own[["seed", "s_w1_own"]], on="seed", how="left").sort_values("seed")
    m["s_pred"] = [prediction(p, e, o) if p != "EXCLUDED" else np.nan
                   for p, e, o in zip(m.class_pred, m.s_pred_early, m.s_w1_own)]
    m.to_csv(FROZEN, index=False)
    h = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    (OUT / f"{which}_frozen.sha256").write_text(h + "\n")
    print(len(m), h)


def train(seeds=FRESH):
    _nice()
    from .asym_t23c import train_one
    need = OUT / ("fresh_frozen.csv" if seeds == FRESH else "fresh_ext_frozen.csv")
    if not need.exists():
        raise SystemExit("predictions not frozen")
    f = OUT / "fresh_train.csv"
    done = set() if not f.exists() else set(pd.read_csv(f).seed)
    for sd in seeds:
        if sd in done:
            continue
        r = train_one(sd, _k(), PHI2, BUDGET)
        pd.DataFrame([r]).reindex(columns=["seed", "placed_at_init", "crossed", "step", "s_cross", "growth", "relax", "ratio"]).to_csv(
            f, mode="a", header=not f.exists(), index=False)
        print(json.dumps({"seed": sd, "done": True}), flush=True)


def score():
    fz = pd.concat([pd.read_csv(f) for f in (OUT / "fresh_frozen.csv", OUT / "fresh_ext_frozen.csv") if f.exists()])
    tr = pd.read_csv(OUT / "fresh_train.csv")
    d = tr.merge(fz[["seed", "class_pred", "s_pred", "s0"]], on="seed")
    n_total, n_pi = len(d), int(d.placed_at_init.astype(bool).sum())
    d = d[~d.placed_at_init.astype(bool)].copy()
    d["truth"] = [truth(c_, s_) for c_, s_ in zip(d.crossed, d.s_cross)]
    sc = score_registered(d, n_pi, n_total)
    out = {"label": "Track 2 final round, step 2, REGISTERED (new predictor designed from the diagnosis of earlier failures); "
                    "T2-3 FAIL, T2-3b/T2-3c UNRESOLVED, T2-3d FAIL unchanged",
           "runs": n_total, "placed_at_init": n_pi, "sha256_frozen": (OUT / "fresh_frozen.sha256").read_text().strip(),
           "confusion": {f"pred {p} / true {t}": int(((d.class_pred == p) & (d.truth == t)).sum())
                         for p in ("EARLY", "LATE") for t in ("EARLY", "LATE")}, **sc}
    (OUT / "fresh_scores.json").write_text(json.dumps(out, indent=1, default=float))
    d.to_csv(OUT / "fresh_scored_runs.csv", index=False)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    {"classify": lambda: run_classify(arg), "own": lambda: run_own(arg), "gate": gate, "freeze": lambda: freeze(arg or "fresh"),
     "train": lambda: train(FRESH if arg != "ext" else FRESH_EXT), "score": score}[cmd]()
