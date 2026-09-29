"""Registered test 2A: Adam per-run ordering at an unseen activation value (a = 1.85).

Design (author-approved): results/designs/2A_adam_ordering_design.md.  Registration: results/track2a_registration.md.

2A is the Track A pipeline (src/track_a.py, registered in fcd2e46) UNCHANGED, with one rule changed: WHEN the Adam
preconditioner is read.
  - Track A froze P at the rule point t_R (the last upward passage of 0.5·s*_frozen).
  - 2A freezes P at t_sw = the first step at which |w₂| reaches s*_run, the switch of the branch occupied at the rule
    point (the rule found POST HOC on the a = 1.65 runs, src/track_a_diag.py `p_switch`; tested prospectively here).
Everything else is Track A's code: landscape construction and validation, winding rule, per-seed frozen switch and own
threshold, Adam training (lr 0.01, 32,000 steps, make_data(200, seed), init uniform(−1, 1)⁴), rule point, branch rule,
trajectory-integrated recursion (primary), closed form κ_k·χ (secondary), every-step detection.

How the Track A code is reused unchanged: `pipeline()` loads src/track_a.py as a SEPARATE module instance (its SHA-256
must equal the registered Track A hash) and rebinds only its data constants (A = 1.85, the 80 seeds, the output
directory, Adam only).  The imported `src.track_a` module is never modified.  The P-freeze change is made exactly as in
the post hoc diagnostic: `predict_one` reads v̂ only at t_R, so v̂[t_R] is replaced by v̂[t_sw] and `predict_one` is
called unchanged.  t_sw needs s*_run, which `predict_one` computes from the rule-point state alone; it is therefore
called once with v̂ unchanged (the Track A rule; kept as a DESCRIPTIVE column, never scored) and once with v̂[t_sw].

Registered criteria (scored runs only; pred = the trajectory-integrated r_traj with P = P(t_sw), obs = r_obs):
  A1  Spearman(pred, obs) ≥ 0.5
  A2  fraction with |obs/pred − 1| ≤ 0.10 ≥ 0.75   (threshold supplied by the author)
  A3  median obs/pred ∈ [0.9, 1.1]
Validity (else every criterion is UNRESOLVED): ≥ 60 of 80 runs cross; t_sw precedes the crossing in ≥ 90% of crossing
runs (the others are unscored and reported); the rule point precedes the crossing in every scored run.
Scored run: crossed, t_sw defined and strictly before the crossing step, and a finite primary prediction.

    python -m src.track2a landscape   # s*_pop, κ, validation              -> results/track2a/landscape.json
    python -m src.track2a freeze      # per-seed frozen inputs               -> results/track2a/frozen_seeds.csv
    python -m src.track2a hashes      # registration hashes                  -> results/track2a/registration.sha256
    python -m src.track2a train       # registered runs, no gap evaluated    -> results/track2a/predictions_parts.jsonl
    python -m src.track2a finalize    # predictions.csv + predictions.sha256
    python -m src.track2a observe     # AFTER the predictions commit: every-step detection, scores
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import linear_response as LR
from . import track_a as T

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "track2a"
PATHS = OUT / "paths"
A = 1.85
SEEDS = tuple(range(1_850_000, 1_850_080))
OPT = "adam"

# registered criteria
A1_MIN = 0.5
A2_MIN = 0.75                  # the author's number (2026-09-28)
A2_TOL = 0.10
A3_BAND = (0.9, 1.1)
MIN_CROSS = 60
TSW_MIN_FRAC = 0.90

# src/track_a.py as registered in fcd2e46 (results/track_a/registration.sha256)
TRACK_A_SHA256 = "06d21cf36511a1797df1f8d20691c1d6f5f8d29bcbc812780b1ddc2a46121b09"
RSS_LIMIT = 3 * 1024 ** 3      # machine rule: stop any job over 3 GB RSS

# files frozen by the registration commit (SHA-256 in registration.sha256)
REGISTERED_FILES = ("src/track2a.py", "src/track_a.py", "src/linear_response.py", "src/fold1d.py",
                    "src/phase2b_ordering.py", "src/own_threshold.py", "src/width2_conditional.py",
                    "tests/test_track2a.py", "results/track2a_registration.md",
                    "results/track2a/landscape.json", "results/track2a/frozen_seeds.csv")

_PIPE = None


# ------------------------------------------------------------------------------------------ the unchanged pipeline
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_track_a(**consts):
    """A fresh module instance of src/track_a.py (byte-identical to its registered version) with the given module
    constants rebound.  The imported src.track_a is untouched."""
    src = Path(__file__).with_name("track_a.py")
    assert _sha(src) == TRACK_A_SHA256, "src/track_a.py differs from its registered Track A version"
    spec = importlib.util.spec_from_file_location("src._track_a_instance", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k, v in consts.items():
        assert hasattr(m, k), k
        setattr(m, k, v)
    return m


def pipeline():
    """The Track A pipeline bound to a = 1.85, the 80 2A seeds, results/track2a and Adam only."""
    global _PIPE
    if _PIPE is None:
        _PIPE = load_track_a(A=A, SEEDS=SEEDS, OPTS=(OPT,), OUT=OUT, PATHS=PATHS)
    return _PIPE


# ------------------------------------------------------------------------------------------ rules (pure functions)
def t_switch(s_path, s_run):
    """t_sw: the first step at which s_t = |w₂(t)| reaches s*_run (None if never).  Output-scale trajectory only."""
    if s_run is None or not np.isfinite(s_run):
        return None
    return LR._first_ge(np.asarray(s_path, float), float(s_run))


def v_at_switch(V, t_rule, t_sw):
    """v̂ with the row that predict_one reads (t_R) replaced by v̂ at t_sw: P is frozen at P(t_sw)."""
    Vm = np.array(V, float, copy=True)
    Vm[t_rule] = V[t_sw]
    return Vm


def _stat(pred, obs):
    r = obs / pred
    return {"spearman": T.spearman(pred, obs) if len(pred) >= 3 else float("nan"),
            "frac_within_10pct": float(np.mean(np.abs(r - 1) <= A2_TOL)) if len(r) else float("nan"),
            "median_ratio": float(np.median(r)) if len(r) else float("nan")}


def score_2a(r_obs, r_pred, crossed, step_obs, t_rule, t_sw, r_secondary=None):
    """Registered verdicts.  Arrays over the 80 runs (NaN where undefined).
    Crossing runs whose t_sw is undefined or not strictly before the crossing are unscored (counted, reported), as are
    crossing runs with no primary prediction.  Validity: ≥ 60 crossings; t_sw strictly before the crossing in ≥ 90% of
    crossing runs; t_rule strictly before the crossing in every scored run.  If invalid, A1-A3 are UNRESOLVED.  A
    criterion whose statistic cannot be computed (Spearman: < 3 scored runs; A2/A3: 0 scored runs) is UNRESOLVED."""
    r_obs, r_pred = np.asarray(r_obs, float), np.asarray(r_pred, float)
    step_obs, t_rule, t_sw = (np.asarray(v, float) for v in (step_obs, t_rule, t_sw))
    crossed = np.asarray(crossed, bool)
    n_cross = int(crossed.sum())
    tsw_before = crossed & np.isfinite(t_sw) & np.isfinite(step_obs) & (t_sw < step_obs)
    scored = tsw_before & np.isfinite(r_pred) & np.isfinite(r_obs)
    rule_before = np.isfinite(t_rule) & np.isfinite(step_obs) & (t_rule < step_obs)
    n_tsw = int(tsw_before.sum())
    frac_tsw = n_tsw / n_cross if n_cross else float("nan")
    v1 = n_cross >= MIN_CROSS
    v2 = n_cross > 0 and frac_tsw >= TSW_MIN_FRAC
    v3 = bool(rule_before[scored].all())
    valid = bool(v1 and v2 and v3)
    idx = np.nonzero(scored)[0]
    st = _stat(r_pred[scored], r_obs[scored])
    n = int(scored.sum())

    def verdict(ok, computable):
        if not valid or not computable:
            return "UNRESOLVED"
        return "PASS" if ok else "FAIL"
    out = {"n_runs": int(len(crossed)), "n_crossed": n_cross, "n_tsw_before_crossing": n_tsw,
           "frac_tsw_before_crossing": frac_tsw,
           "n_crossed_tsw_not_before": int((crossed & ~tsw_before).sum()),
           "n_crossed_tsw_before_no_prediction": int((tsw_before & ~scored).sum()),
           "n_scored": n, "n_scored_rule_not_before_crossing": int((scored & ~rule_before).sum()),
           "validity": {"min_crossings": bool(v1), "tsw_before_90pct": bool(v2), "rule_before_all_scored": v3},
           "valid": valid, "scored_index": idx.tolist()}
    out["A1"] = {"n": n, "spearman": st["spearman"],
                 "verdict": verdict(st["spearman"] >= A1_MIN, n >= 3 and np.isfinite(st["spearman"]))}
    out["A2"] = {"n": n, "frac_within_10pct": st["frac_within_10pct"],
                 "verdict": verdict(st["frac_within_10pct"] >= A2_MIN, n >= 1)}
    out["A3"] = {"n": n, "median_ratio": st["median_ratio"],
                 "verdict": verdict(A3_BAND[0] <= st["median_ratio"] <= A3_BAND[1], n >= 1)}
    vs = [out[k]["verdict"] for k in ("A1", "A2", "A3")]
    out["outcome"] = "UNRESOLVED" if "UNRESOLVED" in vs else ("PASS" if all(v == "PASS" for v in vs) else
                                                             "FAIL " + "+".join(k for k in ("A1", "A2", "A3")
                                                                                if out[k]["verdict"] == "FAIL"))
    if r_secondary is not None:
        rs = np.asarray(r_secondary, float)
        ok2 = scored & np.isfinite(rs)
        out["secondary_closed_form"] = {"n": int(ok2.sum()), **_stat(rs[ok2], r_obs[ok2]),
                                        "note": "reported, not scored"}
    return out


# ------------------------------------------------------------------------------------------ landscape and frozen inputs
def landscape():
    """Track A's landscape construction at a = 1.85 (continuation from the certified 1.60 switch; validated by
    own_threshold.global_min at 0.995 and 1.005 s*_pop).  Writes results/track2a/landscape.json."""
    OUT.mkdir(parents=True, exist_ok=True)
    return pipeline().landscape()


def freeze():
    """Per-seed frozen inputs (Track A's frozen_one): own-sample branch switch and global own threshold.
    Writes results/track2a/frozen_parts.jsonl as each seed completes, then results/track2a/frozen_seeds.csv."""
    pipeline().freeze()


def hashes():
    lines = [f"{_sha(ROOT / p)}  {p}" for p in REGISTERED_FILES]
    (OUT / "registration.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _assert_registration():
    for line in (OUT / "registration.sha256").read_text().splitlines():
        h, p = line.split()
        assert _sha(ROOT / p) == h, f"{p} changed since registration"


# ------------------------------------------------------------------------------------------ training and predictions
def predict_2a(seed, W, M, V, K, s_frozen, m=None):
    """Primary and secondary predictions with P frozen at t_sw (everything else is track_a.predict_one).  `m`: the
    pipeline instance (default: a = 1.85; another instance is used only to check the machinery at a = 1.65)."""
    m = pipeline() if m is None else m
    base = m.predict_one(OPT, seed, W, M, V, K, s_frozen)          # Track A's rule (P at t_R): DESCRIPTIVE only
    s = np.abs(W[:, 2])
    t_sw = t_switch(s, base.get("s_run"))
    desc = {"descr_r_traj_P_rule": base.get("r_traj", float("nan")), "descr_r_cf_P_rule": base.get("r_cf", float("nan")),
            "descr_P_rule_w1": base.get("P_w1"), "descr_P_rule_b1": base.get("P_b1"), "descr_P_rule_b2": base.get("P_b2")}
    if base.get("status") != "ok" or t_sw is None:
        return {**base, "r_traj": float("nan"), "r_cf": float("nan"), "t_sw": t_sw,
                "status_2a": "no prediction: " + (base.get("status") if base.get("status") != "ok" else "s never reaches s*_run"),
                **desc}
    p = m.predict_one(OPT, seed, W, M, v_at_switch(V, base["t_rule"], t_sw), K, s_frozen)
    assert p["t_rule"] == base["t_rule"] and p["s_run"] == base["s_run"], "rule point / branch must not depend on P"
    return {**p, "t_sw": int(t_sw), "s_at_tsw": float(s[t_sw]), "status_2a": "ok" if np.isfinite(p.get("r_traj", np.nan))
            else "no primary prediction (" + str(p.get("status_traj")) + ")", **desc}


def _rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")


def train():
    """The 80 registered Adam runs for the full budget.  No placement or gap is evaluated.  Each run's parameter path is
    saved (paths/*.npz, untracked) and hashed; each row is written as the run completes."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    _assert_registration()
    m = pipeline()
    PATHS.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(OUT / "frozen_seeds.csv").set_index("seed")
    f = OUT / "predictions_parts.jsonl"
    done = set() if not f.exists() else {json.loads(l)["seed"] for l in f.read_text().splitlines()}
    for seed in SEEDS:
        if seed in done:
            continue
        W, M, V, K = m.train_one(OPT, seed)
        pf = m._path_file(OPT, seed)
        np.savez(pf, W=W)
        h = _sha(pf)
        s_fr = float(fr.loc[seed, "s_frozen"])
        try:
            r = predict_2a(seed, W, M, V, K, s_fr) if np.isfinite(s_fr) else \
                {"opt": OPT, "seed": seed, "status": "no frozen switch", "status_2a": "no prediction: no frozen switch"}
        except Exception as e:                                             # counted, not replaced
            r = {"opt": OPT, "seed": seed, "status": f"error: {type(e).__name__}: {e}", "status_2a": "no prediction: error"}
        r["path_sha256"] = h
        with open(f, "a") as fh:
            fh.write(json.dumps(LR._jsonable(r)) + "\n")
        print(json.dumps({k: r.get(k) for k in ("seed", "status_2a", "t_rule", "t_sw", "r_traj", "r_cf")}), flush=True)
        _rss_guard()


FORBIDDEN = {"cross_step", "s_obs", "r_obs", "step_obs", "placed", "crossed"}


def finalize():
    d = pd.DataFrame([json.loads(l) for l in (OUT / "predictions_parts.jsonl").read_text().splitlines()])
    d = d.sort_values("seed")
    assert list(d.seed) == list(SEEDS), "every registered seed exactly once"
    assert not (FORBIDDEN & set(d.columns))
    p = OUT / "predictions.csv"
    d.to_csv(p, index=False)
    h = _sha(p)
    (OUT / "predictions.sha256").write_text(f"{h}  predictions.csv\n")
    print(h, len(d), d.status_2a.value_counts().to_dict())


# ------------------------------------------------------------------------------------------ observation and scoring
def observe():
    """After the predictions commit: assert the committed hashes, then every-step detection on each saved path."""
    os.nice(15)
    import torch
    torch.set_num_threads(1)
    from .fold1d import activation
    from .phase2b_ordering import state
    m = pipeline()
    _assert_registration()
    m._assert_committed(OUT / "predictions.csv", OUT / "predictions.sha256")
    pr = pd.read_csv(OUT / "predictions.csv")
    f = activation("sin_family", A)
    parts = OUT / "observed_parts.jsonl"
    done = {} if not parts.exists() else {r["seed"]: r for r in map(json.loads, parts.read_text().splitlines())}
    for r in pr.itertuples():
        if r.seed in done:
            continue
        pf = m._path_file(OPT, r.seed)
        assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
        W = np.load(pf)["W"]
        t_c = None
        for t in range(1, len(W)):
            if state(torch.tensor(W[t]), f, A, 1.0)["placement_ok"]:
                t_c = t
                break
        row = {"seed": int(r.seed), "crossed": t_c is not None, "step_obs": t_c,
               "s_obs": abs(float(W[t_c, 2])) if t_c is not None else float("nan")}
        with open(parts, "a") as fh:
            fh.write(json.dumps(LR._jsonable(row)) + "\n")
        print(json.dumps(row), flush=True)
        _rss_guard()
    o = pd.DataFrame([json.loads(l) for l in parts.read_text().splitlines()])
    fr = pd.read_csv(OUT / "frozen_seeds.csv")
    d = pr.merge(o, on="seed").merge(fr[["seed", "w2_own"]], on="seed", how="left")
    assert len(d) == len(SEEDS)
    d["r_obs"] = d.s_obs / d.s_run - 1
    d["tsw_before_cross"] = d.crossed & (d.t_sw < d.step_obs)
    d["rule_before_cross"] = d.crossed & (d.t_rule < d.step_obs)
    sc = score_2a(d.r_obs, d.r_traj, d.crossed, d.step_obs, d.t_rule, d.t_sw, r_secondary=d.r_cf)
    d["scored"] = False
    d.loc[d.index[sc.pop("scored_index")], "scored"] = True
    d.to_csv(OUT / "observed_runs.csv", index=False)
    S = d[d.scored]
    un = d[d.crossed & ~d.tsw_before_cross]
    sc["unscored_crossing_runs"] = [{"seed": int(x.seed), "t_sw": None if pd.isna(x.t_sw) else int(x.t_sw),
                                     "step_obs": int(x.step_obs), "status_2a": x.status_2a} for x in un.itertuples()]
    sc["descriptive"] = {
        "n_status_2a": d.status_2a.value_counts().to_dict(),
        "n_winding_rule_agrees": int(d.winding_rule_agrees.fillna(False).astype(bool).sum()),
        "median_r_obs_scored": float(S.r_obs.median()), "median_r_traj_scored": float(S.r_traj.median()),
        "r_obs_q10_q90_scored": [float(S.r_obs.quantile(0.1)), float(S.r_obs.quantile(0.9))],
        "r_traj_q10_q90_scored": [float(S.r_traj.quantile(0.1)), float(S.r_traj.quantile(0.9))],
        "ratio_q10_q90_scored": [float((S.r_obs / S.r_traj).quantile(0.1)), float((S.r_obs / S.r_traj).quantile(0.9))],
        "median_steps_tsw_to_crossing_scored": float((S.step_obs - S.t_sw).median()),
        "median_steps_rule_to_crossing_scored": float((S.step_obs - S.t_rule).median()),
        "median_s_run_over_s_frozen": float((d.s_run / d.s_frozen).median()),
        "track_a_rule_P_at_rule_point_NOT_SCORED": _stat(S.descr_r_traj_P_rule.to_numpy(float), S.r_obs.to_numpy(float))
        if len(S) else {}}
    sc["predictions_sha256"] = (OUT / "predictions.sha256").read_text().split()[0]
    sc["registration_sha256_file_sha256"] = _sha(OUT / "registration.sha256")
    (OUT / "scores.json").write_text(json.dumps(sc, indent=1, default=float))
    print(json.dumps(sc, indent=1, default=float))
    return sc


if __name__ == "__main__":
    {"landscape": landscape, "freeze": freeze, "hashes": hashes, "train": train, "finalize": finalize,
     "observe": observe}[sys.argv[1]]()
