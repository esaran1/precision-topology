"""POST HOC forecast-horizon analysis (rebuttal, author's request 2026-11-14).  NOT A REGISTRATION; no verdict changes.

For each existing run set, the FROZEN causal forecaster of that set is re-run on the EXISTING runs at every cutoff
fraction f in FGRID (0.80, 0.82, ..., 0.98 and 0.95), each forecast with its per-run NaN recomputation, and scored with
the set's own scoring function, unchanged:

  W1, G, T, Tp   Phase 1C arms (registered 6433edc): phase1c.forecast_one on the saved paths (results/phase1c/paths,
                 hash-asserted by phase1c._load); scored by phase1c.score_arm_1c (W1: the actual switch is 1C's, from
                 the registered-f cutoff's rule-point branch, as 1C's own other-f descriptive uses it).
  A177           width-1 sine, Adam, a = 1.77 (registered 173a612): the runs stored no paths, so each run is RETRAINED
                 with the registered training (trackA_causal.train, deterministic) and used only if its training hash
                 equals the committed runs.jsonl train_sha256; forecast trackA_causal.forecast_with_nan_check (primary
                 slin_20 / coupled_sw); actual switch per f from that f's cutoff branch (trackA_causal.observe_one's
                 rule); scored by trackA_causal.score_track_a on the documented columns (trackA_score).
  A165_sgd/adam  width-1 sine, a = 1.65 (pre-tag registration; causal re-score is Phase 1B, post hoc): the registered
                 scored set, phase1b.forecast_one with phase1b.config (Adam P at the rule point) on the saved paths
                 (hash-asserted); actual switch from the f = 0.90 cutoff's branch (Phase 1B's rule); scored by
                 phase1b.rescore with W1's tolerances.

Before the grid is trusted, every forecast at the f values each set stored (registered f and its other f) must equal
the stored forecast field for field (phase1c.same_forecast), and the score at the registered f must reproduce the
committed verdict numbers.  Machine rules: launched only after a separate gate step (src.u2_gate gate, logs in
results/rebuttal/horizon/); 2 worker processes, nice 15, one thread; between units the parent waits while memory free
< 25 %, swap free < 500 MB or disk free < 10 GB (logged); a worker stops above 3 GB RSS; resumable (parts.jsonl).

    python -m src.u2_gate gate horizon --disk-gb 10        (U2_GATE_DIR=results/rebuttal/horizon; its own step)
    python -m src.posthoc_horizon run                      (after the gate; detached)
    python -m src.posthoc_horizon score                    (-> horizon.csv, horizon.json)
    python -m src.posthoc_horizon figure                   (-> results/rebuttal/figures/fig_horizon.pdf/.png)

Artifacts (paths under results/, for the ledger's provenance check): rebuttal/horizon/horizon.csv and
rebuttal/horizon/horizon.json (score); results/rebuttal/figures/fig_horizon.pdf/.png (figure).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "rebuttal" / "horizon"
FIG = RESULTS / "rebuttal" / "figures"
PARTS = OUT / "parts.jsonl"
LABEL = "POST HOC forecast-horizon analysis (existing runs only; not a registration; no verdict changes)"

FGRID = tuple(sorted({round(0.80 + 0.02 * i, 2) for i in range(10)} | {0.95}))
SETS = ("W1", "A177", "A165_sgd", "A165_adam", "G", "T", "Tp")
SET_LABEL = {"W1": "width-1 sine, SGD, a = 1.58", "A177": "width-1 sine, Adam, a = 1.77",
             "A165_sgd": "width-1 sine, SGD, a = 1.65", "A165_adam": "width-1 sine, Adam, a = 1.65",
             "G": "width-1 GELU, SGD, warm start", "T": "width-2 sine, slowed output",
             "Tp": "width-2 sine, branch-point start"}
F_USED = {"W1": 0.90, "G": 0.95, "T": 0.95, "Tp": 0.95, "A177": 0.90, "A165_sgd": 0.90, "A165_adam": 0.90}
F_NOTE = {"W1": "registered (chosen after the est seeds)", "G": "registered (chosen after the pilot)",
          "T": "registered (chosen after the pilot)", "Tp": "registered (chosen after the pilot)",
          "A177": "registered", "A165_sgd": "Phase 1B post hoc re-score", "A165_adam": "Phase 1B post hoc re-score"}
P1B_ARM = {"A165_sgd": "track_a/sgd", "A165_adam": "track_a/adam"}

MEM_PCT, SWAP_MB, DISK_GB = 25.0, 500.0, 10.0   # disk 10 GB: author, 2026-11-14
RSS_LIMIT = 3 * 1024 ** 3
WORKERS = 2   # 2, not 3: the 36-arm re-score runs one worker at the same time (three-worker cap)


def _fk(f):
    return str(round(float(f), 2))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonable(o):
    from .phase1c import _jsonable as J
    return J(o)


# ------------------------------------------------------------------------------------------ units
def units():
    """(set, seed) of every existing run each set's forecaster reads."""
    from . import phase1c as P1C
    from . import phase1b as P1B
    from . import trackA_causal as T
    out = []
    for arm in ("W1",):
        out += [(arm, r["seed"]) for r in P1C._rows(P1C._runs_file(arm)) if r.get("status") == "ok" and r.get("path")]
    out += [("A177", s) for s in T.SEEDS]
    for k, arm_id in P1B_ARM.items():
        d = P1B.registered_table(arm_id)
        out += [(k, int(s)) for s in d[d.reg_scored].seed]
    for arm in ("G", "T", "Tp"):
        out += [(arm, r["seed"]) for r in P1C._rows(P1C._runs_file(arm)) if r.get("status") == "ok" and r.get("path")]
    return out


def _same(a, b):
    from .phase1c import same_forecast
    return bool(same_forecast(json.loads(json.dumps(_jsonable(a))), json.loads(json.dumps(_jsonable(b)))))


def _strip(rec):
    return {k: v for k, v in rec.items() if k != "secs"}


def compute_1c(arm, seed):
    from . import phase1c as P1C
    from . import linear_response as LR
    row = next(r for r in P1C._rows(P1C._runs_file(arm)) if r["seed"] == seed)
    row = {**row, "frozen": P1C._frozen(arm)[seed]}
    out, hr, hid = P1C._load(row)
    ctx = P1C.context(arm, row)
    fc = {_fk(f): P1C.forecast_one(arm, row, out, hr, hid, ctx, f) for f in FGRID}
    res = {"set": arm, "seed": seed, "status": "ok", "fc": fc,
           "repro": {"registered_f": _same(fc[_fk(P1C.F[arm])], row["forecast"]),
                     "other_f": _same(fc[_fk(P1C.F_OTHER[arm])], row["forecast_other_f"])}}
    if arm == "W1":
        s = P1C._s_of(out)
        res["t_sw_own_f"] = {k: (LR._first_ge(s, v["cutoff_s_run"]) if v.get("cutoff_s_run") is not None else None)
                             for k, v in fc.items()}
    return res


def compute_a177(seed):
    from . import trackA_causal as T
    from . import linear_response as LR
    row = next(r for r in T._rows(T.OUT / "runs.jsonl") if r["seed"] == seed)
    fr = T._frozen("registered")[seed]
    if fr.get("s_frozen") is None:
        return {"set": "A177", "seed": seed, "status": "no frozen switch"}
    x, y = T.sample(seed)
    W, M, V = T.train(seed)
    h = T._train_sha(W, M, V)
    if h != row["train_sha256"]:
        return {"set": "A177", "seed": seed, "status": "REFUSED: retrained hash differs", "train_sha256": h}
    s = np.abs(W[:, 2])
    fc, t_sw = {}, {}
    for f in FGRID:
        t_c, tR, s_run = T.harness_cutoff(W, f, fr["s_frozen"], x, y)
        rec = {"f": f, "t_c": t_c, "cutoff_rule_point": tR, "cutoff_s_run": s_run}
        if t_c is not None:
            rec["primary"] = T.forecast_with_nan_check(W, M, V, x, y, fr["s_frozen"], t_c)
        fc[_fk(f)] = rec
        t_sw[_fk(f)] = LR._first_ge(s, s_run) if s_run is not None else None
    rp = {}
    for k in ("0.9", "0.95"):
        a, b = fc[k], row["fc"][k]
        rp[k] = all(a[q] == b[q] for q in ("t_c", "cutoff_rule_point", "cutoff_s_run")) and \
            ((a.get("primary") is None and b.get("primary") is None) or _same(a["primary"], b["primary"]))
    return {"set": "A177", "seed": seed, "status": "ok", "train_sha256": h, "fc": fc, "t_sw_own_f": t_sw,
            "repro": {"registered_f": rp["0.9"], "other_f": rp["0.95"]}}


def compute_a165(key, seed):
    from . import phase1b as P1B
    from . import linear_response as LR
    arm_id = P1B_ARM[key]
    d = P1B.registered_table(arm_id)
    r = next(q for q in d.itertuples() if int(q.seed) == seed)
    X, pname, h = P1B.load_verified(arm_id, r)
    case = P1B.make_case(arm_id, r, X)
    cuts = [case.cutoff(f)[0] for f in FGRID]
    if case.kind == "w1" and case.opt == "adam":
        tcs = [v for v in cuts if v is not None]
        if tcs:
            P1B.attach_adam(case, arm_id, seed, max(tcs))
    fc = {_fk(f): P1B.forecast_one(case, P1B.config(arm_id, f, P1B.ADAM_P.get(arm_id))) for f in FGRID}
    s_run = fc[_fk(P1B.F[arm_id])].get("cutoff_s_run")
    stored = next(q for q in P1B._rows(P1B.OUT / "runs_track_a.jsonl") if q["arm_id"] == arm_id and q["seed"] == seed)
    return {"set": key, "seed": seed, "status": "ok", "arm_id": arm_id, "path_sha256": h,
            "s_run_cutoff_branch": s_run,
            "t_sw_cutoff_branch": LR._first_ge(case.s, s_run) if s_run is not None else None,
            "fc": fc,
            "repro": {"registered_f": _same(_strip(fc["0.9"]), _strip(stored["fc"]["0.9"])),
                      "other_f": _same(_strip(fc["0.95"]), _strip(stored["fc"]["0.95"]))}}


def compute(unit):
    k, seed = unit
    t0 = time.time()
    if k in ("W1", "G", "T", "Tp"):
        r = compute_1c(k, seed)
    elif k == "A177":
        r = compute_a177(seed)
    else:
        r = compute_a165(k, seed)
    r["secs"] = round(time.time() - t0, 1)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r["max_rss_gb"] = round(rss / 2 ** 30, 3)
    if rss > RSS_LIMIT:
        r["rss_stop"] = True
    return r


# ------------------------------------------------------------------------------------------ the launcher
def _worker_init():
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # read the priority (os.nice(0) is refused in the sandbox); raise it to 15 only if the launcher did not
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.setpriority(os.PRIO_PROCESS, 0, 15)
    import torch
    torch.set_num_threads(1)


def _gate_log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "memory_waits.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def wait_for_memory():
    """Between units: wait (re-check every 60 s) while memory, swap or disk is below the thresholds; logged."""
    from .u2_gate import measure, evaluate
    while True:
        m = measure(str(OUT))
        if evaluate(m, MEM_PCT, SWAP_MB, DISK_GB):
            return m
        _gate_log(f"WAIT free={m['mem_free_pct']:.0f}% swap={m['swap_free_mb']:.0f}M disk={m['disk_free_gb']:.1f}G")
        time.sleep(60)


def done_keys(p=PARTS):
    if not Path(p).exists():
        return set()
    return {(r["set"], r["seed"]) for r in (json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())}


def run():
    os.environ.setdefault("U2_GATE_DIR", str(OUT))
    from . import u2_gate as G
    G.require_gate("posthoc_horizon run", gate_log=OUT / "gate_entries.jsonl", launch_log=OUT / "launches.jsonl")
    _worker_init()
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    import multiprocessing as mp
    todo = [u for u in units() if tuple(u) not in done_keys()]
    print(f"{len(todo)} units to do", flush=True)
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn"), initializer=_worker_init) as ex:
        pend, it = set(), iter(todo)
        stop = False
        while True:
            while not stop and len(pend) < WORKERS:
                u = next(it, None)
                if u is None:
                    break
                wait_for_memory()
                pend.add(ex.submit(compute, u))
            if not pend:
                break
            dn, pend = wait(pend, return_when=FIRST_COMPLETED)
            for fu in dn:
                r = fu.result()
                with open(PARTS, "a") as fh:
                    fh.write(json.dumps(_jsonable(r)) + "\n")
                print(json.dumps({k: r.get(k) for k in ("set", "seed", "status", "repro", "secs", "max_rss_gb")}),
                      flush=True)
                if r.get("rss_stop"):
                    print("STOP: a worker exceeded 3 GB RSS", flush=True)
                    stop = True
        if stop:
            raise SystemExit("stopped on RSS")


# ------------------------------------------------------------------------------------------ scoring (pure tables)
def _q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else None


def summary_row(set_, f, crit, n_with_cutoff, n_base, n_scored, n_miss, nan_checked, nan_differs, err_cross,
                lag_obs, valid, outcome, cutoff_before_frac, horizon_steps):
    """One CSV row from the scoring function's output (crit: its C1..C4 dicts) and the per-run error arrays over the
    scored runs with a forecast (err_cross = t_fc − t_obs, lag_obs = t_obs − t_sw)."""
    e = np.abs(np.asarray(err_cross, float))
    L = np.maximum(np.abs(np.asarray(lag_obs, float)), 1.0)
    return {"set": set_, "label": SET_LABEL[set_], "f": round(float(f), 2),
            "f_is_used": bool(round(float(f), 2) == F_USED[set_]), "f_used": F_USED[set_], "f_used_note": F_NOTE[set_],
            "n_with_cutoff": int(n_with_cutoff), "n_crossing_base": int(n_base), "n_scored": int(n_scored),
            "n_forecast": int(n_scored - n_miss), "n_miss": int(n_miss),
            "C1_n_within": crit["C1"]["n_within"], "C1_frac_within": crit["C1"]["frac_within"],
            "C1_tau": crit["C1"]["tau"],
            "C2_n_within": crit["C2"]["n_within"], "C2_frac_within": crit["C2"]["frac_within"],
            "C2_tau": crit["C2"]["tau"], "C3_median_ratio": crit["C3"]["median_ratio"],
            "abs_err_cross_q50": _q(e, 50), "abs_err_cross_q90": _q(e, 90),
            "abs_err_cross_in_lags_q50": _q(e / L, 50) if len(e) else None,
            "nan_recompute_checked": int(nan_checked), "nan_recompute_differs": int(nan_differs),
            "cutoff_before_crossing_frac": cutoff_before_frac, "horizon_steps_q50": horizon_steps,
            "valid_by_set_rules": bool(valid), "outcome_DESCRIPTIVE": outcome}


def _parts(p=PARTS):
    rows = {}
    for l in Path(p).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            rows[(r["set"], r["seed"])] = r
    return rows


def score_1c(arm, mine, f):
    from . import phase1c as P1C
    from . import gelu_transfer as G
    from . import width2_asym as W
    runs = []
    for r in P1C._rows(P1C._runs_file(arm)):
        m = mine.get(r["seed"])
        r = dict(r)
        if r.get("status") == "ok" and r.get("path"):
            assert m is not None and m["status"] == "ok", f"missing horizon row {arm} {r['seed']}"
            r["forecast"] = m["fc"][_fk(f)]
        r["forecast_other_f"] = None
        runs.append(r)
    obs = P1C._rows(P1C._obs_file(arm))
    R = P1C.table(arm, runs, obs)
    if arm == "W1":
        pc, sp = None, None
    elif arm == "G":
        pc = json.loads((G.OUT / "pilot.json").read_text())["pilot_median_chi_tsw"][G.PRIMARY]
        sp = G._land()["s_glob"]
    else:
        pc = json.loads((W.OUT / "pilot.json").read_text())["arms"][P1C.w2a_arm(arm)]["pilot_median_chi_tsw"]
        sp = R["s_pop"]
    res = P1C.score_arm_1c(arm, R, pilot_chi=pc, s_pop=sp)
    A = {k: P1C._arr(v) for k, v in R.items() if k in ("t_fc", "t_sw_fc", "t_obs", "t_sw", "t_c")}
    idx = np.asarray(res.get("scored_index", []), int)
    hf = P1C.is_forecast(A["t_fc"], A["t_sw_fc"])
    fs = idx[hf[idx]] if len(idx) else idx
    crit = {k: (res[k] if "verdict_if_valid" not in res[k] else res[k]) for k in ("C1", "C2", "C3", "C4")}
    hz = res.get("DESCRIPTIVE", {}).get("horizon", {}).get("cross_steps", {}).get("median")
    row = summary_row(arm, f, crit, int(np.isfinite(A["t_c"]).sum()), res.get("n_base_crossing_runs", 0),
                      res.get("n_scored", 0), res.get("n_scored_miss", 0), res.get("n_nan_recompute_checked", 0),
                      res.get("n_nan_recompute_differs", 0), A["t_fc"][fs] - A["t_obs"][fs],
                      A["t_obs"][fs] - A["t_sw"][fs], res.get("valid", False), res.get("outcome"),
                      (res.get("cutoff_validity") or {}).get("frac"), hz)
    return row, res


def score_a177(mine, f):
    from . import trackA_causal as T
    from . import trackA_score as S
    rows, obs = [], []
    ob = {o["seed"]: o for o in T._rows(T.OUT / "observed.jsonl")}
    for r in T._rows(T.OUT / "runs.jsonl"):
        m = mine.get(r["seed"])
        r = dict(r)
        o = dict(ob.get(r["seed"], {}))
        if r.get("status") == "ok":
            assert m is not None and m["status"] == "ok", f"missing or refused horizon row A177 {r['seed']}"
            r["fc"] = {_fk(f): m["fc"][_fk(f)]}
            s_run = m["fc"][_fk(f)].get("cutoff_s_run")
            o["by_f"] = {_fk(f): {"t_sw": m["t_sw_own_f"][_fk(f)], "s_sw": s_run, "P_true_at_t_sw": None}}
        rows.append(r)
        if o:
            obs.append(o)
    R = S.documented_columns(T.table(rows, obs, float(_fk(f)), "primary"))
    res = T.score_track_a(R, T.TOLERANCES)
    A = {k: np.asarray(v, float) for k, v in R.items() if k in ("t_fc", "t_sw_fc", "t_obs", "t_sw", "t_c")}
    idx = np.asarray(res["scored_index"], int)
    hf = np.isfinite(A["t_fc"]) & np.isfinite(A["t_sw_fc"])
    fs = idx[hf[idx]] if len(idx) else idx
    with np.errstate(invalid="ignore"):
        hz = _q((A["t_obs"] - A["t_c"])[idx], 50) if len(idx) else None
    row = summary_row("A177", f, res, int(np.isfinite(A["t_c"]).sum()), res["n_base_crossing_runs"], res["n_scored"],
                      res["n_scored_miss"], res["n_nan_recompute_checked"], res["n_nan_recompute_differs"],
                      A["t_fc"][fs] - A["t_obs"][fs], A["t_obs"][fs] - A["t_sw"][fs], res["valid"], res["outcome"],
                      res["cutoff_validity"]["frac"], hz)
    return row, res


def score_a165(key, mine, f):
    from . import phase1b as P1B
    arm_id = P1B_ARM[key]
    d = P1B.registered_table(arm_id)
    base = d.reg_scored.to_numpy(bool)
    rows = []
    for s in d.seed[d.reg_scored].astype(int):
        m = mine[int(s)]
        rows.append({"arm_id": arm_id, "seed": int(s), "fc": {_fk(f): m["fc"][_fk(f)]},
                     "s_run_cutoff_branch": m["s_run_cutoff_branch"], "t_sw_cutoff_branch": m["t_sw_cutoff_branch"]})
    R = P1B.arm_arrays(arm_id, d, rows, "fc", float(_fk(f)))
    res = P1B.rescore(P1B.TOL_KEY[arm_id], base, R["t_c"], R["t_fc"], R["t_sw_fc"], R["r_fc"], R["t_obs"], R["t_sw"],
                      R["r_obs"], R["nan_identical"], True)
    scm, hf = res.pop("scored_mask"), res.pop("has_fc_mask")
    res.pop("before_mask")
    fs = scm & hf
    row = summary_row(key, f, res["criteria"], int((base & np.isfinite(R["t_c"])).sum()), res["n_registered_scored"],
                      res["n_scored"], res["n_scored_miss"], res["n_nan_recompute_checked"],
                      res["n_nan_recompute_differs"], R["t_fc"][fs] - R["t_obs"][fs], R["t_obs"][fs] - R["t_sw"][fs],
                      res["valid_1c"], res["outcome_1c_rule"], res["cutoff_validity"]["frac"],
                      (res["horizon"].get("cross_steps") or {}).get("median"))
    return row, res


def _core(c):
    keep = ("n", "n_within", "frac_within", "median_ratio", "mean_D", "ci95", "verdict")
    return {k: c.get(k) for k in keep if k in c}


def _close(a, b):
    if isinstance(a, dict):
        return all(_close(a[k], b.get(k)) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def registered_check(set_, res):
    """At the set's used f: the scoring output equals the committed one (C1-C4 core numbers and the outcome)."""
    if set_ in ("W1", "G", "T", "Tp"):
        ref = json.loads((RESULTS / "phase1c" / "scores.json").read_text())["arms"][set_]
        got = res
    elif set_ == "A177":
        ref = json.loads((RESULTS / "trackA" / "scores.json").read_text())
        got = res
    else:
        ref = json.loads((RESULTS / "phase1b" / "summary.json").read_text())["arms"][P1B_ARM[set_]]["at_f"]
        got = {**res, **res["criteria"], "outcome": res["outcome_1c_rule"]}
        ref = {**ref, **ref["criteria"], "outcome": ref["outcome_1c_rule"]}
    got = json.loads(json.dumps(_jsonable(got)))
    ok = {k: _close(_core(ref[k]), _core(got[k])) for k in ("C1", "C2", "C3", "C4")}
    ok["outcome"] = ref["outcome"] == got["outcome"]
    return ok


def score():
    P = _parts()
    by = {}
    for (k, s), r in P.items():
        by.setdefault(k, {})[s] = r
    out_rows, checks, repro = [], {}, {}
    for k in SETS:
        mine = by.get(k, {})
        rr = [r for r in mine.values() if r["status"] == "ok"]
        repro[k] = {"n_rows": len(mine), "n_ok": len(rr),
                    "n_refused": sum(1 for r in mine.values() if r["status"].startswith("REFUSED")),
                    "forecast_equal_at_used_f": sum(bool(r["repro"]["registered_f"]) for r in rr),
                    "forecast_equal_at_other_stored_f": sum(bool(r["repro"]["other_f"]) for r in rr)}
        for f in FGRID:
            if k in ("W1", "G", "T", "Tp"):
                row, res = score_1c(k, mine, f)
            elif k == "A177":
                row, res = score_a177(mine, f)
            else:
                row, res = score_a165(k, mine, f)
            out_rows.append(row)
            if round(f, 2) == F_USED[k]:
                checks[k] = registered_check(k, res)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "horizon.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    inputs = {str(p.relative_to(ROOT)): _sha(p) for p in (
        RESULTS / "phase1c" / "scores.json", RESULTS / "phase1c" / "forecasts.sha256", RESULTS / "trackA" / "runs.jsonl",
        RESULTS / "trackA" / "observed.jsonl", RESULTS / "trackA" / "scores.json",
        RESULTS / "phase1b" / "runs_track_a.jsonl", RESULTS / "phase1b" / "summary.json",
        RESULTS / "track_a" / "observed_runs.csv", RESULTS / "track_a" / "predictions.csv",
        ROOT / "src" / "causal_forecast.py", ROOT / "src" / "causal_adam.py", ROOT / "src" / "phase1c.py",
        ROOT / "src" / "trackA_causal.py", ROOT / "src" / "phase1b.py", PARTS)}
    js = {"label": LABEL, "producer": "src/posthoc_horizon.py score", "fgrid": list(FGRID), "f_used": F_USED,
          "f_used_note": F_NOTE, "set_label": SET_LABEL, "forecast_reproduction": repro,
          "score_reproduction_at_used_f": checks, "inputs_sha256": inputs, "rows": out_rows}
    (OUT / "horizon.json").write_text(json.dumps(_jsonable(js), indent=1) + "\n")
    print(json.dumps(_jsonable({"repro": repro, "checks": checks}), indent=1))
    return js


# ------------------------------------------------------------------------------------------ figure
def wilson(k, n, z=1.959963984540054):
    if not n:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


PANELS = (("Width-1 sine", ("W1", "A165_sgd", "A165_adam", "A177")),
          ("Outside width-1 sine", ("G", "T", "Tp")))
STYLE = {"W1": ("#1b6ca8", "o"), "A177": ("#c0392b", "s"), "A165_sgd": ("#2e8b57", "^"), "A165_adam": ("#8e44ad", "D"),
         "G": ("#d35400", "o"), "T": ("#1b6ca8", "s"), "Tp": ("#2e8b57", "^")}


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(OUT / "horizon.csv")))
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5), sharey=True)
    for ax, (title, sets) in zip(axes, PANELS):
        for k in sets:
            rs = sorted((r for r in rows if r["set"] == k), key=lambda r: float(r["f"]))
            f = np.array([float(r["f"]) for r in rs])
            kk = np.array([int(r["C1_n_within"]) for r in rs])
            n = np.array([int(r["n_scored"]) for r in rs])
            p = np.where(n > 0, kk / np.maximum(n, 1), np.nan)
            lo, hi = np.array([wilson(a, b) for a, b in zip(kk, n)]).T
            col, mk = STYLE[k]
            ax.errorbar(f, p, yerr=[np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)],   # rounding at p = 0 or 1
                         color=col, marker=mk, ms=3.5, lw=1.0, capsize=1.5, elinewidth=0.7,
                        label=SET_LABEL[k])
            fu = F_USED[k]
            j = int(np.argmin(np.abs(f - fu)))
            ax.plot([f[j]], [p[j]], marker=mk, ms=8, mfc="none", mec=col, mew=1.2, ls="none")
        ax.axhline(0.80, color="0.4", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("cutoff fraction f")
        ax.set_xticks([0.80, 0.85, 0.90, 0.95])
        ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, handlelength=1.6)   # below: never over data
    axes[0].set_ylabel("fraction of runs with the\ncrossing step within tolerance")
    fig.text(0.995, 0.995, "post hoc", ha="right", va="top", fontsize=8, style="italic", color="0.3")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_horizon.pdf")
    fig.savefig(FIG / "fig_horizon.png", dpi=200)


def _launch(label, fn):
    """Any computation here starts only after a separate, earlier gate step (one gate entry licenses one launch)."""
    from . import u2_gate as G
    G.require_gate(label, gate_log=OUT / "gate_entries.jsonl", launch_log=OUT / "launches.jsonl")
    _worker_init()
    return fn()


def selftest():
    import pytest
    return pytest.main(["-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_posthoc_horizon.py")])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run()
    else:
        rc = _launch(f"posthoc_horizon {cmd}", {"score": score, "figure": figure, "selftest": selftest}[cmd])
        sys.exit(rc if isinstance(rc, int) else 0)
