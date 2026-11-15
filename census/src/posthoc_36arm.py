"""POST HOC (rebuttal program, author's instruction 2026-11-14): the submitted paper's 36-arm trajectory-integrated
agreement re-scored with the FROZEN strictly causal forecaster.

LABEL: POST HOC.  No registration, no registered verdict changes.  The 36 arms and their observed crossings were known
(the submitted §5.1, Fig. 5 grey crosses, App. V.3 Table 36, Table 39) before this analysis.  Nothing below is tuned on
these runs: every rule is an existing frozen rule, applied unchanged.

WHAT THE SUBMITTED PAPER SAID.  "integrating the linearized dynamics along measured output-scale paths brings every
defined arm median within 5%" (p6); Table 39 "Integrated, tracked branch 1,708 runs, 99.9% within 10%".  The construction
(src/linear_response.py, Track 1 final round) integrates the linear-response recursion along the run's OWN measured
s_t path up to the predicted step: it is path-conditioned, not a forecast (prediction-inputs audit, condition (iii)).

THE RUNS.  The 1,750 runs of linear_response.run_list() (36 arms: lag1 ×8, lag2-prim ×8, lag2-tstar ×8, 4b ×8, SGD ×2,
TaskB ×2; a ∈ {1.30, 1.45, 1.50, 1.60}).  No paths were stored; each run is REPLAYED deterministically by
linear_response.replay_path, called unchanged (the AS-RUN protocols, including the lag-2 checkpoints and the Block 4b
interventions, with its own checkpoint-equality assertion), with a recorder that additionally keeps the raw θ path and
Adam's raw first moment and bias-corrected second moment (exactly prediction_inputs_audit.adam_replay's definitions).
Replay checks, per run, against committed values: the replayed horizon length and max output scale equal
linear_response/compare_runs.csv's n_steps and s_max exactly, and (looked up only AFTER the run's forecasts are
computed; the observed table is passed to no forecasting code) the replayed |w₂| at the committed crossing step equals
the committed crossing |w₂| exactly.

THE FORECASTER.  src/causal_forecast.py, asserted identical to the frozen file (phase1b.assert_forecaster: the Phase 1C
manifest hash and the 25aac9e blob), called through phase1b.W1Case / phase1b.forecast_one unchanged (guarded views of
rows < t_c, then the per-run NaN recomputation on copies with every row ≥ t_c set to NaN), with Phase 1C's frozen W1
configuration (quad extrapolation of log s, window 5% (≥ 50 steps), horizon 1.6·s_ref).

DECISIONS (fixed here before any re-score outcome was computed; each is Phase 1B's for the width-one Track A runs):
  s_frozen  the run's frozen own-sample switch by Track A's freeze rule (track_a.frozen_one), unchanged except for the
            activation value: Newton from the 1A population switch point (lag_law/kappa.csv z_star, s_star at the arm's
            a) onto the run's own training sample, continued (grid [0.3, 1.7]·s_pop, spacing 0.002·s_pop); s_frozen =
            its switch nearest s_pop.  Pre-run information only (the landscape and the sample).
  f         0.90 (Phase 1C W1 / Phase 1B width 1: the cutoff on s*_run at the causal rule point) and 0.95, both
            reported.  f = 0.90 for W1 was set after 1C's estimation seeds; f = 0.95 was chosen after the 1C pilot (for
            G, T, T′).  Neither is tuned here.
  Adam P    "rule_point" (Track A's rule, P frozen at the causal rule point), primary; "cutoff" (P at t_c − 1)
            reported beside it.  SGD: P = I.
  lr        the run's hidden-coordinate learning rate: Adam 0.01, SGD 0.3 (lag-1/lag-2 factors rescale only w₂'s step;
            the forecaster reads the measured s path before t_c and extrapolates it).
  budget    the run's own step budget (lag1: round(32,000/φ); lag2: t* + ⌈(32,000 − t*)/φ⌉; 4b, SGD, TaskB: 32,000).
  switch    actual switch t_sw = the first step with s ≥ s*_run of the cutoff's rule-point branch (1C's W1 rule);
            r_obs = s_obs/s*_run − 1 (the lag against the same switch the forecast uses).
  scored    crossed, cutoff strictly before the crossing (t_c < t_obs) and t_sw defined; a scored run without a
            forecast is a MISS (counts as not within in every fraction).
  yardsticks (per arm; DESCRIPTIVE, no verdict)
            - the submitted claim's: observed/predicted ratio of arm medians, median r_obs / median r_fc over scored
              runs with a forecast, within 5% (|ratio − 1| ≤ 0.05);
            - Table 39's: fraction of scored runs with |r_obs/r_fc − 1| ≤ 0.10;
            - 1C W1's tolerances: |t_fc − t_obs| ≤ 10 (C1), |lag error in steps| ≤ 15 (C2), median r_obs/r_fc in
              [0.80, 1.20] (C3).
  Interventions (4b teleport/reset, lag-2 slowdown from t*): applied as run.  The forecaster reads one hidden-state row
            (the rule point) and the s path before t_c; it is not told about an intervention.  Counted per arm
            (t* relative to the rule point and the cutoff).

Machine rules: launched only after a separate, logged gate entry (src/u2_gate, U2_GATE_DIR=results/rebuttal/arm36,
--disk-gb 15); one process per worker, nice 15, one thread; memory checked between sample groups (free ≥ 25%, swap free
≥ 500 MB; waits while failing); stops below 15 GB free disk or above 3 GB RSS.  Resumable.

    U2_GATE_DIR=results/rebuttal/arm36 python -m src.u2_gate gate "arm36 run" --disk-gb 15     # its own step
    U2_GATE_DIR=results/rebuttal/arm36 python -m src.posthoc_36arm run [k i]                  # worker i of k
    U2_GATE_DIR=results/rebuttal/arm36 python -m src.posthoc_36arm summarize                  # csv + json + README

Artifacts (paths under results/, for the ledger's provenance check): rebuttal/arm36/parts_0of1.jsonl (run,
appended per run), rebuttal/arm36/arm36_causal.csv and rebuttal/arm36/arm36_causal.json (summarize).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import linear_response as LR

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "rebuttal" / "arm36"
LABEL = "POST HOC: the 36-arm agreement re-scored with the frozen causal forecaster (rebuttal, 2026-11-14)"

FS = (0.90, 0.95)
F_PRIMARY = 0.90
ADAM_PS = ("rule_point", "cutoff")
TAU_CROSS, TAU_LAG, BAND = 10, 15, 0.20            # Phase 1C W1 (phase1c.TAU_CROSS / TAU_LAG / BAND["W1"])
WITHIN_ARM = 0.05                                   # the submitted claim's yardstick (arm medians)
WITHIN_RUN = 0.10                                   # Table 39's yardstick (per run)
BASE_BUDGET = 32_000
DISK_STOP_GB = 10.0                                 # lowered from 15 by the author, 2026-11-14
RSS_LIMIT = 3 * 1024 ** 3
INPUTS = ("results/linear_response/compare_runs.csv", "results/linear_response/compare_arms.csv",
          "results/lag_law/kappa.csv", "results/lag_law/predictions.csv", "results/lag_test_runs.csv",
          "results/lag_test2_runs.csv", "results/residual_mechanism_parts.csv", "results/sgd_own_runs.csv",
          "results/ts_test/runs.csv", "src/causal_forecast.py", "src/linear_response.py", "src/phase1b.py",
          "src/phase1c.py", "src/posthoc_36arm.py")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ recorder (replay unchanged)
class RawPath(LR.Path_):
    """linear_response.Path_ that also keeps the raw θ path W (4 columns), Adam's raw first moment M and bias-corrected
    second moment V̂ (prediction_inputs_audit.adam_replay's definitions; NaN where the step count is 0)."""

    def __init__(self, adam):
        super().__init__(adam)
        self.v = []

    def add(self, rec):
        super().add(rec)
        if self.adam:
            self.v.append(rec.get("v", np.full(4, np.nan)))

    def finish(self):
        self.W = np.array(self.w, float)
        if self.adam:
            self.M = np.array(self.m, float)
            self.V = np.array(self.v, float)
        else:
            self.M = self.V = None
        return super().finish()


_ORIG_RECORD = LR._record_state


def _record_raw(th, opt, adam):
    rec = _ORIG_RECORD(th, opt, adam)
    if adam:
        st = opt.state[th]
        k = int(st["step"])
        rec["v"] = st["exp_avg_sq"].numpy() / (1 - LR.BETA2 ** k) if k > 0 else np.full(4, np.nan)
    return rec


@contextlib.contextmanager
def raw_recording():
    """replay_path, called unchanged, records through RawPath (module-global lookups at call time)."""
    old_p, old_r = LR.Path_, LR._record_state
    LR.Path_, LR._record_state = RawPath, _record_raw
    try:
        yield
    finally:
        LR.Path_, LR._record_state = old_p, old_r


def replay(kind, a, seed, arm, s_stop):
    with raw_recording():
        pth, x, y = LR.replay_path(kind, a, int(seed), str(arm), s_stop)
    assert isinstance(pth, RawPath)
    return pth, x, y


# ------------------------------------------------------------------------------------------ per-run frozen inputs
def population_point(a):
    k = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    r = k[k.a.round(2) == round(float(a), 2)].iloc[0]
    return np.array(json.loads(r.z_star), float), float(r.s_star)


def frozen_s(a, x, y, z_pop, s_pop):
    """Track A's freeze rule (track_a.frozen_one) at activation a: the own-sample switch of the landscape branch."""
    th, res, H = LR.newton(z_pop, s_pop, a, x, y)
    if res > 1e-9 or np.linalg.eigvalsh(H).min() <= 0:
        return float("nan"), "no own-sample minimum from the landscape point"
    B = LR.Branch(s_pop, th, a, x, y, 0.3 * s_pop, 1.7 * s_pop, 0.002 * s_pop)
    s_fr = B.switch(s_pop)
    return (float("nan"), "landscape branch has no own-sample switch") if s_fr is None else (float(s_fr), "ok")


def opt_lr(kind):
    return ("sgd", LR.LR_SGD) if kind == "SGD" else ("adam", LR.LR_ADAM)


def t_star_of(kind, a, seed):
    if kind not in ("lag2-prim", "lag2-tstar", "4b"):
        return None
    import torch
    from . import lag_test2 as lt
    rule = "tstar" if kind == "lag2-tstar" else "primary"
    return int(torch.load(lt.STATES / f"{rule}_a{a:.2f}_s{int(seed)}.pt", weights_only=False)["t_star"])


def budget_of(kind, arm, t_star):
    if kind == "lag1":
        return int(round(BASE_BUDGET / float(arm)))
    if kind in ("lag2-prim", "lag2-tstar"):
        return int(t_star + math.ceil((BASE_BUDGET - t_star) / float(arm)))
    return BASE_BUDGET


def cfg_of(f, adam_P):
    from . import phase1b as P1B
    return P1B.config("track_a/adam", f, adam_P)


# ------------------------------------------------------------------------------------------ one run
def forecast_run(kind, a, seed, arm, pth, x, y, s_frozen, t_star):
    """Every forecast of one run (no observed crossing is read here)."""
    from . import phase1b as P1B
    opt, lr = opt_lr(kind)
    budget = budget_of(kind, arm, t_star)
    case = P1B.W1Case(a, x, y, s_frozen, opt, lr, budget, pth.W, pth.M, pth.V)
    row = {"budget": budget, "opt": opt, "t_star": t_star, "fc": {}}
    for f in FS:
        t_c, extra = case.cutoff(f)
        Ps = ADAM_PS if opt == "adam" else (None,)
        row["fc"][str(f)] = {str(P): P1B.forecast_one(case, cfg_of(f, P)) for P in Ps}
        s_run = extra.get("cutoff_s_run")
        row["fc"][str(f)]["_switch"] = {"t_c": t_c, "s_run": s_run, "t_rule": extra.get("cutoff_rule_point"),
                                        "t_sw": LR._first_ge(case.s, s_run) if s_run is not None else None}
    return row


def verify_obs(s, step_obs, s_obs):
    """AFTER the forecasts: the replayed |w₂| at the committed crossing step equals the committed crossing |w₂|."""
    t = int(step_obs)
    return {"obs_index_ok": bool(t < len(s) and float(s[t]) == float(s_obs))}


# ------------------------------------------------------------------------------------------ machine rules and driver
def _setup():
    # read the priority (os.nice(0) is refused in the sandbox); raise it to 15 only if the launcher did not
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.setpriority(os.PRIO_PROCESS, 0, 15)
    import torch
    torch.set_num_threads(1)


def _guards(tag):
    from .act_fold import check_memory
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")
    free_gb = shutil.disk_usage("/").free / 2 ** 30
    if free_gb < DISK_STOP_GB:
        raise SystemExit(f"STOP: disk free {free_gb:.1f} GB < {DISK_STOP_GB} GB")
    while True:
        try:
            ok, f, w = check_memory()
        except Exception:                                                   # noqa: BLE001
            ok, f, w = False, float("nan"), float("nan")
        with open(OUT / "memory_checks.log", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB disk={free_gb:.1f}G "
                     f"{'OK' if ok else 'WAIT'}\n")
        if ok:
            return
        time.sleep(60)


def _parts(k, i):
    return OUT / f"parts_{i}of{k}.jsonl"


def _done():
    keys = set()
    for p in OUT.glob("parts_*of*.jsonl"):
        for ln in p.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                keys.add((r["set"], round(r["a"], 2), int(r["seed"]), str(r["arm"])))
    return keys


def _committed_runs():
    c = pd.read_csv(RESULTS / "linear_response" / "compare_runs.csv", float_precision="round_trip")
    c["a"] = c.a.round(2); c["arm"] = c.arm.astype(str)
    return c


def run(k=1, i=0):
    from . import phase1b as P1B
    from .u2_gate import require_gate
    OUT.mkdir(parents=True, exist_ok=True)
    require_gate(f"posthoc_36arm run worker {i} of {k}")
    _setup()
    P1B.assert_forecaster()
    runs = LR.run_list()
    comm = _committed_runs().set_index(["set", "a", "seed", "arm"])
    obs = LR.observed(); obs["arm"] = obs.arm.astype(str)
    obs = obs.set_index(["set", "a", "seed", "arm"])
    done = _done()
    groups = LR._groups(runs)
    for gi, ((fam, a, seed), g) in enumerate(groups):
        if gi % k != i:
            continue
        todo = [r for r in g.itertuples() if (r.set, round(r.a, 2), int(r.seed), str(r.arm)) not in done]
        if not todo:
            continue
        _guards(f"group {fam} {a} {seed}")
        s_pop_kappa = LR._s_pop()[a]
        z_pop, s_pop = population_point(a)
        assert s_pop == s_pop_kappa
        cache, frozen = {}, {}
        for r in todo:
            t0 = time.time()
            key = (r.set, round(r.a, 2), int(r.seed), str(r.arm))
            row = {"set": r.set, "a": a, "seed": int(r.seed), "arm": str(r.arm)}
            try:
                ck = LR._path_key(r.set, r.arm)
                if ck not in cache:
                    cache[ck] = replay(r.set, a, int(r.seed), str(r.arm), LR.HORIZON * s_pop)
                pth, x, y = cache[ck]
                cm = comm.loc[key]
                row.update(n_steps=len(pth.s) - 1, s_max=float(pth.s.max()),
                           replay_n_steps_ok=bool(len(pth.s) - 1 == int(cm.n_steps)),
                           replay_s_max_ok=bool(float(pth.s.max()) == float(cm.s_max)))
                fk = (fam, x.shape[0])
                if fk not in frozen:
                    frozen[fk] = frozen_s(a, x, y, z_pop, s_pop)
                row["s_frozen"], row["frozen_note"] = frozen[fk]
                t_star = t_star_of(r.set, a, int(r.seed))
                if math.isfinite(row["s_frozen"]):
                    row.update(forecast_run(r.set, a, int(r.seed), str(r.arm), pth, x, y, row["s_frozen"], t_star))
                    row["status"] = "ok"
                else:
                    row.update(t_star=t_star, status="no s_frozen")
                # observed crossing: read only now, after the forecasts are in the row
                if key in obs.index:
                    o = obs.loc[key]
                    row.update(crossed=True, t_obs=int(o.step_obs), s_obs=float(o.s_obs),
                               **verify_obs(pth.s, o.step_obs, o.s_obs))
                else:
                    row.update(crossed=False)
            except Exception as e:                                          # recorded, never silently dropped
                row["status"] = f"error: {type(e).__name__}: {e}"
            row["secs"] = round(time.time() - t0, 2)
            with open(_parts(k, i), "a") as fh:
                fh.write(json.dumps(_jsonable(row)) + "\n")
        print(json.dumps({"group": [fam, a, int(seed)], "n": len(todo), "t": time.strftime("%H:%M:%S")}), flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


# ------------------------------------------------------------------------------------------ pure scoring
def table(rows, f, adam_P="rule_point"):
    """One row per run at cutoff fraction f (Adam: P rule adam_P)."""
    out = []
    for r in rows:
        d = {"set": r["set"], "a": round(float(r["a"]), 2), "arm": str(r["arm"]), "seed": int(r["seed"]),
             "status": r.get("status"), "crossed": bool(r.get("crossed", False)),
             "t_obs": r.get("t_obs"), "s_obs": r.get("s_obs"), "t_star": r.get("t_star"),
             "replay_ok": bool(r.get("replay_n_steps_ok", False) and r.get("replay_s_max_ok", False)
                               and (r.get("obs_index_ok", True) if r.get("crossed") else True))}
        fcs = (r.get("fc") or {}).get(str(f))
        if fcs:
            sw = fcs["_switch"]
            fc = fcs[str(adam_P) if r.get("opt") == "adam" else "None"]
            d.update(t_c=sw["t_c"], s_run=sw["s_run"], t_sw=sw["t_sw"], t_rule=sw["t_rule"],
                     fc_status=fc.get("status"), t_fc=fc.get("t_fc"), t_sw_fc=fc.get("t_sw_fc"), r_fc=fc.get("r_fc"),
                     nan_identical=fc.get("nan_recompute_identical"))
        out.append(d)
    T = pd.DataFrame(out)
    for c in ("t_c", "s_run", "t_sw", "t_rule", "t_fc", "t_sw_fc", "r_fc", "t_obs", "s_obs", "t_star"):
        if c not in T:
            T[c] = np.nan
        T[c] = pd.to_numeric(T[c], errors="coerce").astype(float)
    if "fc_status" not in T:
        T["fc_status"] = None
    if "nan_identical" not in T:
        T["nan_identical"] = None
    T["cutoff_before"] = T.crossed & np.isfinite(T.t_c) & (T.t_c < T.t_obs)
    T["scored"] = T.cutoff_before & np.isfinite(T.t_sw)
    T["has_fc"] = T.scored & (T.fc_status == "ok") & np.isfinite(T.r_fc) & np.isfinite(T.t_fc)
    T["r_obs"] = T.s_obs / T.s_run - 1
    T["ratio"] = np.where(T.has_fc & (T.r_fc != 0), T.r_obs / T.r_fc, np.nan)
    T["cross_err"] = T.t_fc - T.t_obs
    T["lag_err"] = (T.t_fc - T.t_sw_fc) - (T.t_obs - T.t_sw)
    T["intervention_after_rule"] = np.isfinite(T.t_star) & np.isfinite(T.t_rule) & (T.t_star > T.t_rule)
    T["intervention_after_cutoff"] = np.isfinite(T.t_star) & np.isfinite(T.t_c) & (T.t_star >= T.t_c)
    return T


def _frac(mask, n):
    return float(mask.sum() / n) if n else float("nan")


def arm_stats(T):
    """Per arm (set, a, arm): counts, the three yardsticks (DESCRIPTIVE).  Misses count as not within."""
    rows = []
    for (st, a, arm), g in T.groupby(["set", "a", "arm"], sort=False):
        sc, fc = g[g.scored], g[g.has_fc]
        n = len(sc)
        mo, mp = (float(np.median(fc.r_obs)), float(np.median(fc.r_fc))) if len(fc) else (float("nan"),) * 2
        rom = mo / mp if len(fc) and mp != 0 else float("nan")
        mrr = float(np.median(fc.ratio)) if len(fc) else float("nan")
        rows.append({
            "set": st, "a": a, "arm": arm, "n_runs": len(g), "n_crossed": int(g.crossed.sum()),
            "n_cutoff_before": int(g.cutoff_before.sum()), "n_scored": n, "n_forecast": len(fc), "n_miss": n - len(fc),
            "n_nan_differs": int((g.scored & (g.nan_identical == False)).sum()),         # noqa: E712
            "n_replay_not_ok": int((~g.replay_ok).sum()),
            "n_intervention_after_rule": int((g.scored & g.intervention_after_rule).sum()),
            "median_r_obs": mo, "median_r_fc": mp, "ratio_of_medians": rom,
            "arm_within_5pct": bool(np.isfinite(rom) and abs(rom - 1) <= WITHIN_ARM),
            "median_run_ratio": mrr,
            "frac_runs_within_10pct": _frac(fc.ratio.sub(1).abs() <= WITHIN_RUN, n),
            "C1_frac_cross_within_10": _frac(fc.cross_err.abs() <= TAU_CROSS, n),
            "C2_frac_lag_within_15": _frac(fc.lag_err.abs() <= TAU_LAG, n),
            "C3_median_in_band": bool(np.isfinite(mrr) and abs(mrr - 1) <= BAND)})
    return pd.DataFrame(rows)


def overall(A, T):
    d = A[A.n_forecast > 0]
    sc = T[T.scored]
    return {"n_arms": int(len(A)), "n_arms_defined": int(len(d)), "n_arms_within_5pct": int(d.arm_within_5pct.sum()),
            "ratio_of_medians_min": float(d.ratio_of_medians.min()) if len(d) else None,
            "ratio_of_medians_max": float(d.ratio_of_medians.max()) if len(d) else None,
            "n_runs": int(len(T)), "n_scored": int(len(sc)), "n_forecast": int(T.has_fc.sum()),
            "n_miss": int(len(sc) - T.has_fc.sum()),
            "frac_scored_runs_within_10pct": _frac(T[T.has_fc].ratio.sub(1).abs() <= WITHIN_RUN, len(sc)),
            "n_nan_differs": int(A.n_nan_differs.sum()), "n_replay_not_ok": int(A.n_replay_not_ok.sum())}


# ------------------------------------------------------------------------------------------ summarize
def _rows():
    rows = []
    for p in sorted(OUT.glob("parts_*of*.jsonl")):
        rows += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    return rows


def summarize():
    rows = _rows()
    keys = [(r["set"], round(r["a"], 2), int(r["seed"]), str(r["arm"])) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate run rows"
    ref = pd.read_csv(RESULTS / "linear_response" / "compare_arms.csv")
    ref["a"] = ref.a.round(2); ref["arm"] = ref.arm.astype(str)
    ref = ref.set_index(["set", "a", "arm"])
    out, res = [], {}
    for f in FS:
        for P in ADAM_PS:
            T = table(rows, f, P)
            A = arm_stats(T)
            A.insert(0, "adam_P", P); A.insert(0, "f", f)
            A["submitted_path_conditioned_ratio_of_medians_full7"] = [
                float(ref.loc[(s, a, m), "full_7_ratio_of_medians"]) if (s, a, m) in ref.index else float("nan")
                for s, a, m in zip(A.set, A.a, A.arm)]
            out.append(A)
            res[f"f={f}, adam_P={P}"] = overall(A, T)
    csv = pd.concat(out, ignore_index=True)
    csv.to_csv(OUT / "arm36_causal.csv", index=False)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    errs = [r for r in rows if str(r.get("status", "")).startswith("error")]
    js = {"label": LABEL, "n_rows": len(rows), "n_error_rows": len(errs),
          "errors": [{k: r[k] for k in ("set", "a", "seed", "arm", "status")} for r in errs][:20],
          "primary": f"f = {F_PRIMARY}, adam_P = rule_point", "overall": res,
          "inputs_sha256": {p: _sha(ROOT / p) for p in INPUTS},
          "parts_sha256": {p.name: _sha(p) for p in sorted(OUT.glob("parts_*of*.jsonl"))},
          "git_head": git("rev-parse", "HEAD")}
    (OUT / "arm36_causal.json").write_text(json.dumps(_jsonable(js), indent=1) + "\n")
    print(json.dumps(_jsonable({"overall": res, "n_rows": len(rows), "n_error_rows": len(errs)}), indent=1))
    return js


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(*(int(v) for v in sys.argv[2:4])) if len(sys.argv) > 2 else run()
    elif cmd == "summarize":
        summarize()
    else:
        raise SystemExit(__doc__)
