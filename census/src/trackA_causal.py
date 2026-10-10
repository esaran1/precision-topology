"""Track A: the causal Adam per-run forecast at an unseen activation value (a = 1.77).

Design: results/designs/trackA_causal_adam_design.md (33de250; author's decisions 2026-10-09: slin_20, coupled_sw,
a = 1.77, f = 0.90 (0.95 descriptive), 100 registered seeds 9,776,000-099, a 40-seed pilot 9,774,000-039, R4-unstable
runs count as misses; the two rules frozen and hashed before registration; their forecast error estimated on PILOT SEEDS
ONLY).  Registration text: results/trackA_registration.md.  Forecaster: src/causal_adam.py.

Pipeline (Track A's width 1, unchanged): free Adam lr 0.01 (phase1a_pilot.w1_train: make_data(200, seed), local torch
Generator, no global RNG state), budget 32,000, every-step detection (phase1a_pilot.w1_observe); landscape = Track A's
continuation from the certified 1.60 switch and its validation (src/track_a.py byte-identical to its registered version,
loaded through track2a.load_track_a with a, the seeds and OUT rebound); per-seed s*_frozen = Newton from the landscape
switch point onto the own sample, continued (Track A's R0).  Cutoff: 1C's W1 nested stopping time on s*_run
(causal_forecast.w1_cutoff_on_s_run) at f.

ORDER (no step reads anything a later step produces):
  gate                       memory gate (a separate logged step before each job): exit 0 iff free ≥ 25%, swap ≥ 500 MB
  scan / landscape           seed and a = 1.77 scan; landscape at 1.77 + validation (a failure stops before the pilot)
  freeze_rules               SHA-256 of the frozen rules' source (slin_20, coupled_sw) -> frozen_rules.json
  freeze pilot               s*_frozen for the pilot seeds
  pilot                      the 40 pilot seeds: train, forecasts (rules asserted frozen), observation -> pilot_runs.jsonl
  pilot_summary              the pilot tolerances (τ = 1.5 × pilot q90 rounded up to 5; b = |median − 1| + 2SE rounded
                             up to 0.05) -> pilot.json   [to the author before registration]
  freeze registered / manifest / stamp      before training a registered seed
  run                        REFUSES TO START until registration_stamp.txt.ots exists; per seed: train (full budget,
                             no observation), forecasts from GUARDED views of rows < t_c, NaN recomputation -> runs.jsonl
  finalize                   forecasts.sha256 (committed before observe)
  observe                    retrain (hash asserted), every-step detection, actual switch, P(t_sw) -> observed.jsonl
  score                      scores.json

    python -m src.trackA_causal gate | scan | landscape | freeze_rules | freeze pilot|registered
    python -m src.trackA_causal pilot | pilot_summary | manifest | stamp | run | finalize | observe | score
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import causal_adam as CA
from . import causal_forecast as C
from . import linear_response as LR
from . import phase1c as P1

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "trackA"
REGISTRATION_MD = RESULTS / "trackA_registration.md"
DESIGN_MD = RESULTS / "designs" / "trackA_causal_adam_design.md"

# ------------------------------------------------------------------------------------------ registered constants
A = 1.77
BUDGET = 32_000
LR_ADAM = LR.LR_ADAM                                   # 0.01
F = 0.90                                               # cutoff fraction (primary)
F_DESC = 0.95                                          # descriptive
SEEDS = tuple(range(9_776_000, 9_776_100))             # 100 registered seeds
PILOT_SEEDS = tuple(range(9_774_000, 9_774_040))       # 40 pilot seeds
EXT, PRULE = "slin_20", "coupled_sw"                   # the frozen forecast (src/causal_adam.py)
DESC_VARIANTS = {"P_cut": ("slin_20", "cut"), "ext_1c": ("ext_1c", "coupled_sw")}   # DESCRIPTIVE, at f = F
WITHIN_MIN_FRAC = P1.WITHIN_MIN_FRAC                   # 0.80 (C1, C2)
CUTOFF_BEFORE_MIN_FRAC = P1.CUTOFF_BEFORE_MIN_FRAC     # 0.90
MIN_CROSS = 60                                         # ≥ 60 crossings and ≥ 60 scored runs
BOOT_N, BOOT_SEED = P1.BOOT_N, 9_776_000               # C4 (1C's procedure; seed = the first registered seed)
TOL_STEP, TOL_BAND = 5, 0.05                           # pilot rule: round up to multiples of these
TOL_FACTOR = 1.5
SE_MEDIAN = "1.2533 × (IQR / 1.349) / √n (Phase 1A's pilot rule)"
RSS_LIMIT = 1024 ** 3                                  # stop over 1 GB RSS
ARTIFACTS = ("trackA/seed_scan.json", "trackA/landscape.json", "trackA/frozen_rules.json", "trackA/frozen_pilot.jsonl",
             "trackA/pilot_runs.jsonl", "trackA/pilot.json", "trackA/frozen_registered.jsonl",
             "trackA/registration.sha256", "trackA/runs.jsonl", "trackA/forecasts.sha256", "trackA/observed.jsonl",
             "trackA/scores.json")


# ------------------------------------------------------------------------------------------ machine rules and io
def _check_memory():
    from .act_fold import check_memory
    try:
        return check_memory()
    except Exception:                                                   # noqa: BLE001
        return False, float("nan"), float("nan")


def _gate_log(tag, ok, f, w):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "memory_gate.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB {'OK' if ok else 'WAIT'}\n")


def gate():
    """The SEPARATE memory-gate step before each job: one check, logged; exit status 0 iff ok."""
    ok, f, w = _check_memory()
    _gate_log("gate", ok, f, w)
    print(ok, f, w)
    return 0 if ok else 1


def memory_gate(tag):
    """Between seeds: waits (re-checking every 60 s) while the gate fails; every check logged."""
    while True:
        ok, f, w = _check_memory()
        _gate_log(tag, ok, f, w)
        if ok:
            return f, w
        time.sleep(60)


def rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 1 GB")


def _setup():
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.setpriority(os.PRIO_PROCESS, 0, 15)
    import torch
    torch.set_num_threads(1)


_sha = P1._sha
_jsonable = P1._jsonable
_append_write = P1._append_write
_rows = P1._rows


def _train_sha(W, M, V):
    h = hashlib.sha256()
    for X in (W, M, V):
        h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ landscape and frozen inputs
def pipeline():
    """Track A's pipeline (src/track_a.py, hash-asserted by track2a.load_track_a) bound to a = 1.77, Adam, results/trackA."""
    from .track2a import load_track_a
    return load_track_a(A=A, SEEDS=SEEDS, OPTS=("adam",), OUT=OUT, PATHS=OUT / "paths")


def landscape():
    """Track A's landscape construction at a = 1.77 (continuation in a from the certified 1.60 switch point, principal
    copy, switch; validated by own_threshold.global_min at 0.995 and 1.005 s*_pop).  A validation failure raises
    SystemExit inside Track A's function: it stops before the pilot.  -> results/trackA/landscape.json"""
    OUT.mkdir(parents=True, exist_ok=True)
    out = pipeline().landscape()
    assert out["validated"] and out["a"] == A
    d = json.loads((OUT / "landscape.json").read_text())
    d["built_by"] = "src/track_a.py landscape() (registered, unchanged) via track2a.load_track_a(A=1.77, OUT=results/trackA)"
    (OUT / "landscape.json").write_text(json.dumps(d, indent=1))
    return d


def _land(path=None):
    return json.loads(Path(path or OUT / "landscape.json").read_text())


def sample(seed):
    from .fold1d import make_data
    x, y = make_data(200, seed)
    return x.double().numpy(), y.double().numpy()


def s_frozen_of(seed, land, a):
    """Track A's R0: Newton from the landscape switch point onto the own sample, continued (grid [0.3, 1.7]·s*_pop,
    spacing 0.002·s*_pop); its switch.  (row, x, y)."""
    s_pop = land["s_star_pop"]
    x, y = sample(seed)
    th, res, H = LR.newton(np.array(land["theta_star"]), s_pop, a, x, y)
    row = {"seed": int(seed), "newton_res": float(res)}
    if res > 1e-9 or np.linalg.eigvalsh(H).min() <= 0:
        return {**row, "s_frozen": None, "note": "no own-sample minimum from the landscape point"}, x, y
    B = LR.Branch(s_pop, th, a, x, y, 0.3 * s_pop, 1.7 * s_pop, 0.002 * s_pop)
    return {**row, "s_frozen": B.switch(s_pop)}, x, y


def _frozen_file(which):
    return OUT / ("frozen_pilot.jsonl" if which == "pilot" else "frozen_registered.jsonl")


def freeze(which):
    """s*_frozen per seed (pilot or registered); resumable; no training."""
    _setup()
    seeds = PILOT_SEEDS if which == "pilot" else SEEDS
    land = _land()
    f = _frozen_file(which)
    done = {r["seed"] for r in _rows(f)}
    for seed in seeds:
        if seed in done:
            continue
        r, _, _ = s_frozen_of(seed, land, A)
        _append_write(f, r)
        rss_guard()
    assert sorted(r["seed"] for r in _rows(f)) == list(seeds), "every seed exactly once"


def _frozen(which):
    rows = {r["seed"]: r for r in _rows(_frozen_file(which))}
    assert sorted(rows) == list(PILOT_SEEDS if which == "pilot" else SEEDS)
    return rows


def freeze_rules():
    """results/trackA/frozen_rules.json: SHA-256 of the frozen rules' source (src/causal_adam.py: slin_20 and
    coupled_sw, with their constants)."""
    OUT.mkdir(parents=True, exist_ok=True)
    d = {"rules": {"extrapolation": EXT, "preconditioner": PRULE}, "sha256": CA.rule_hashes(),
         "functions": [f.__name__ for f in CA.FROZEN_RULES]}
    (OUT / "frozen_rules.json").write_text(json.dumps(d, indent=1) + "\n")
    return d


def assert_rules_frozen():
    """The committed rule hashes equal the rules' current source."""
    d = json.loads((OUT / "frozen_rules.json").read_text())
    assert d["sha256"] == CA.rule_hashes(), "the frozen rules changed"
    return d


# ------------------------------------------------------------------------------------------ seed and a scan
SEED_PREFIX = {"registered": "9776", "pilot": "9774"}
SCAN_TOPS = ("src", "tests", "results", "paper", "notes", "independent", "data")
SCAN_SKIP_FILES = ("causal_adam.py", "trackA_causal.py", "test_trackA_causal.py", "trackA_registration.md",
                   "trackA_causal_adam_design.md")
SCAN_SKIP_SUFFIX = (".npz", ".npy", ".pdf", ".png", ".pt", ".pkl", ".ots", ".bak", ".gz", ".zip", ".pyc", ".jpg",
                    ".parquet")
A_PATTERN = r"(^|[^0-9])1\.(770*|769999[0-9]*)([^0-9]|$)"
# Every file where the token occurs, reviewed (2026-10-09): none is an activation value.  {file: (matching lines, what)}
A_HITS_REVIEWED = {
    "src/verify_ledger.py": (1, "Track A's own PRODUCERS comment ('at a = 1.77')"),
    "tests/test_phase2a_posthoc.py": (1, "EPS = 1.77e-3 (a step size)"),
    "results/phase2b_results.md": (1, "a bootstrap interval bound [1.77–2.43]"),
    "results/phase1b_results.md": (1, "a ratio quantile 1.77 / 2.01 / 2.39"),
    "results/designs/phase3_explore/p3_spop4_scan.log": (1, "gap values 'G 1.77' (Phase 3 exploration log)"),
    "results/sb_fold/branch_losses.csv": (1, "the scale grid s = 1.77 of the simplicity-bias landscape (no sin activation)"),
    "results/phase2b_checkpoints.csv": (4, "the R column (a ratio) 1.769999…; its a column holds only 1.3, 1.35, 1.4, "
                                           "1.45, 1.5, 1.6 (gitignored 2.7 GB per-step log)"),
    "results/phase2a_ps/freeze.log": (1, "a timing 'secs': 1.77"),
    "results/phase2a_ps/frozen_parts.jsonl": (1, "a timing 'secs': 1.77"),
    "results/phase2a_ps2/freeze.log": (2, "timings 'secs': 1.77"),
    "results/phase2a_ps2/frozen_parts.jsonl": (2, "timings 'secs': 1.77"),
}


def _scan_tree(patterns, chunk=16 * 1024 * 1024, overlap=256):
    """Regex scan of every text file under SCAN_TOPS (binary files skipped; Track A's own files and results/trackA
    skipped), streamed in 16 MB chunks with a 256-byte overlap.  {pattern key: [files with a match]}."""
    import re
    rx = {k: re.compile(v.encode()) for k, v in patterns.items()}
    hits = {k: [] for k in patterns}
    for top in SCAN_TOPS:
        for dp, dns, fns in os.walk(ROOT / top):
            dns[:] = [d for d in dns if d not in ("trackA", "__pycache__", ".pytest_cache")]
            for fn in fns:
                p = Path(dp) / fn
                if fn in SCAN_SKIP_FILES or p.suffix in SCAN_SKIP_SUFFIX:
                    continue
                try:
                    with open(p, "rb") as fh:
                        if b"\0" in fh.read(4096):
                            continue
                        fh.seek(0)
                        found, tail = set(), b""
                        while True:
                            buf = fh.read(chunk)
                            if not buf:
                                break
                            data = tail + buf
                            for k, r in rx.items():
                                if k not in found and r.search(data):
                                    found.add(k)
                            tail = data[-overlap:]
                except OSError:
                    continue
                for k in found:
                    hits[k].append(str(p.relative_to(ROOT)))
    return hits


def registered_overlap():
    """Track A seeds among the SEEDS* / PILOT_SEEDS* constants of the registered / pilot modules."""
    import importlib
    mine = set(SEEDS) | set(PILOT_SEEDS)
    found = {}
    for mod in ("track_a", "track2a", "track2b", "track2c", "gelu_transfer", "width2_asym", "phase1a_pilot", "phase1b",
                "phase1c", "phase2a", "phase2a_ps", "phase2a_ps2", "phase2b", "phase3_w4"):
        m = importlib.import_module(f"src.{mod}")
        for name in dir(m):
            if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
                found[f"{mod}.{name}"] = sorted(mine & set(P1._flat_ints(getattr(m, name))))
    return found


def scan():
    """Seed ranges unused (no 7-digit number with the 9774 / 9776 prefix, also written 9_774_xxx or 9,774,xxx; no
    overlap with the registered modules' seed constants) and a = 1.77 unused (no "1.77", "1.770…" or "1.769999" token)
    in any text file under SCAN_TOPS, Track A's own files excepted."""
    OUT.mkdir(parents=True, exist_ok=True)
    pats = {}
    for k, p in SEED_PREFIX.items():
        pats[k] = (rf"(^|[^0-9.]){p}[0-9]{{3}}([^0-9]|$)|{p[0]}_{p[1:4]}_[0-9]{{3}}"
                   rf"|(^|[^0-9.,]){p[0]},{p[1:4]},[0-9]{{3}}([^0-9,]|$)")
    pats["a_1.77"] = A_PATTERN
    hits = _scan_tree(pats)
    import re
    rx = re.compile(A_PATTERN.encode())
    a_ctx = {}
    for rel in hits["a_1.77"]:
        lines = []
        with open(ROOT / rel, "rb") as fh:
            for i, ln in enumerate(fh):
                m = rx.search(ln)
                if m:
                    lines.append([i + 1, ln[max(0, m.start() - 60):m.end() + 30].decode(errors="replace")])
        a_ctx[rel] = {"n_lines": len(lines), "lines": lines[:5],
                      "reviewed": A_HITS_REVIEWED.get(rel, (None, None))[1]}
    a_ok = all(rel in A_HITS_REVIEWED and A_HITS_REVIEWED[rel][0] == c["n_lines"] for rel, c in a_ctx.items())
    ov = registered_overlap()
    out = {"ranges": {"registered": [SEEDS[0], SEEDS[-1], len(SEEDS)],
                      "pilot": [PILOT_SEEDS[0], PILOT_SEEDS[-1], len(PILOT_SEEDS)]},
           "a": A, "patterns": pats, "dirs": list(SCAN_TOPS), "skipped_own_files": list(SCAN_SKIP_FILES),
           "pattern_files": hits, "registered_seed_overlap": ov,
           "seeds_unused": bool(not hits["registered"] and not hits["pilot"] and not any(ov.values())),
           "a_token_hits": a_ctx,
           "a_rule": "a = 1.77 unused iff every file with the token is in A_HITS_REVIEWED (reviewed: not an activation "
                     "value) with the same number of matching lines",
           "a_unused": bool(a_ok)}
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("pattern_files", "a_token_hits", "seeds_unused", "a_unused")}),
                     indent=1))
    if not (out["seeds_unused"] and out["a_unused"]):
        raise SystemExit("STOP: a Track A seed range or a = 1.77 is not unused")
    return out


# ------------------------------------------------------------------------------------------ cutoff and forecasts
def harness_cutoff(W, f, s_frozen, x, y, a=A):
    """HARNESS SIDE (1C's W1 nested stopping time on s*_run): (t_c, t_R, s*_run) or (None, None, None)."""
    return C.w1_cutoff_on_s_run(np.abs(W[:, 2]), lambda k: W[k, 2], f, s_frozen, a, x, y, lambda k: W[k, CA.HID])


def forecast_with_nan_check(W, M, V, x, y, s_frozen, t_c, ext=EXT, prule=PRULE, a=A, budget=BUDGET):
    """The forecast from GUARDED views (rows < t_c; one hidden row), then the same forecaster, unguarded, on copies with
    every row ≥ t_c NaN; nan_recompute_identical says whether the outputs are identical (phase1c.same_forecast).  An
    exception in the NaN recomputation (a NaN reached the forecaster) counts as NOT identical."""
    t0 = time.time()
    fc = CA.run_forecast(CA.guarded_inputs(a, x, y, s_frozen, t_c, budget, W, M, V), ext, prule)
    try:
        fc2 = CA.forecast_adam(CA.nan_inputs(a, x, y, s_frozen, t_c, budget, W, M, V), ext, prule)
        fc["nan_recompute_identical"] = bool(P1.same_forecast(fc, fc2))
    except Exception as e:                                  # noqa: BLE001  NaN reached the forecaster: not identical
        fc["nan_recompute_identical"] = False
        fc["nan_recompute_error"] = f"{type(e).__name__}: {e}"
    fc["secs"] = round(time.time() - t0, 2)
    return fc


def forecasts(W, M, V, x, y, s_frozen, a=A, budget=BUDGET):
    """Every forecast of one run: at f = F the primary (slin_20, coupled_sw) and the descriptive variants; at f = F_DESC
    the primary rule (descriptive).  Each with its cutoff record (harness side) and NaN recomputation."""
    out = {}
    for f in (F, F_DESC):
        t_c, tR, s_run = harness_cutoff(W, f, s_frozen, x, y, a)
        rec = {"f": f, "t_c": t_c, "cutoff_rule_point": tR, "cutoff_s_run": s_run}
        if t_c is not None:
            rec["primary"] = forecast_with_nan_check(W, M, V, x, y, s_frozen, t_c, EXT, PRULE, a, budget)
            if f == F:
                for k, (e, p) in DESC_VARIANTS.items():
                    rec[k] = forecast_with_nan_check(W, M, V, x, y, s_frozen, t_c, e, p, a, budget)
        out[str(f)] = rec
    return out


def train(seed, a=A, budget=BUDGET):
    from .phase1a_pilot import w1_train
    return w1_train(seed, "adam", budget, a=a)


def run_one(seed, fr, a=A, budget=BUDGET):
    """Train (full budget, no observation) and every forecast.  The row holds no observed quantity."""
    t0 = time.time()
    row = {"seed": int(seed), "a": a, "s_frozen": fr.get("s_frozen")}
    if fr.get("s_frozen") is None:
        return {**row, "status": "no frozen switch"}
    x, y = sample(seed)
    W, M, V = train(seed, a, budget)
    row.update(status="ok", train_sha256=_train_sha(W, M, V), secs_train=round(time.time() - t0, 1))
    row["fc"] = forecasts(W, M, V, x, y, fr["s_frozen"], a, budget)
    row["secs_run"] = round(time.time() - t0, 1)
    return row


def observe_one(row, a=A, budget=BUDGET):
    """Retrain (the training hash asserted), every-step detection over the full budget (w1_observe), and per f the
    actual switch t_sw (first step with s ≥ that cutoff's s*_run) and the true P(t_sw) for the descriptive P̂ error."""
    from .phase1a_pilot import w1_observe
    W, M, V = train(row["seed"], a, budget)
    assert _train_sha(W, M, V) == row["train_sha256"], "retrained run differs from the forecast run"
    s = np.abs(W[:, 2])
    t_obs = w1_observe(W, a=a)
    o = {"seed": row["seed"], "crossed": t_obs is not None, "t_obs": t_obs,
         "s_obs": float(s[t_obs]) if t_obs is not None else None, "by_f": {}}
    for f, rec in row["fc"].items():
        s_run = rec.get("cutoff_s_run")
        t_sw = LR._first_ge(s, s_run) if s_run is not None else None
        P_true = (1.0 / (np.sqrt(V[t_sw][CA.HID]) + CA.EPS)).tolist() if t_sw is not None and t_sw >= 1 else None
        o["by_f"][f] = {"t_sw": t_sw, "s_sw": s_run, "P_true_at_t_sw": P_true}
    return o


# ------------------------------------------------------------------------------------------ pure scoring rules
def criterion_c4(t_fc, t_sw_fc, t_obs, has_fc, seed=BOOT_SEED):
    """C4 (1C's): D = |t_fc − t_obs| − |t_sw,fc − t_obs| over the scored runs with a forecast; PASS iff the upper end of
    the 95% percentile bootstrap interval of mean D (BOOT_N resamples, numpy default_rng(seed)) is < 0."""
    t_fc, t_sw_fc, t_obs, has_fc = P1._arr(t_fc), P1._arr(t_sw_fc), P1._arr(t_obs), P1._arr(has_fc, bool)
    ok = has_fc & np.isfinite(t_fc) & np.isfinite(t_sw_fc) & np.isfinite(t_obs)
    D = np.abs(t_fc[ok] - t_obs[ok]) - np.abs(t_sw_fc[ok] - t_obs[ok])
    n = int(ok.sum())
    mean, lo, hi = P1.bootstrap_mean_ci(D, BOOT_N, seed) if n >= 2 else (float("nan"),) * 3
    computable = bool(n >= 2 and np.isfinite(hi))
    return {"n": n, "mean_D": mean, "ci95": [lo, hi], "boot_n": BOOT_N, "boot_seed": seed,
            "reading": "PASS iff the upper end < 0 (interval entirely above 0 or containing 0: FAIL)",
            "verdict": "UNRESOLVED" if not computable else ("PASS" if hi < 0 else "FAIL"),
            "DESCRIPTIVE_n_forecast_closer": int((D < 0).sum()), "DESCRIPTIVE_n_ties": int((D == 0).sum())}


def criteria(tol, t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs, has_fc):
    """C1-C4 over the given (scored) runs at tolerances tol = {tau_cross, tau_lag, band} (1C's functions)."""
    t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs = (P1._arr(v) for v in (t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs))
    has_fc = P1._arr(has_fc, bool)
    return {"C1": P1.criterion_within(t_fc - t_obs, tol["tau_cross"], has_fc),
            "C2": P1.criterion_within((t_fc - t_sw_fc) - (t_obs - t_sw), tol["tau_lag"], has_fc),
            "C3": P1.criterion_ratio(r_obs, r_fc, has_fc, tol["band"]),
            "C4": criterion_c4(t_fc, t_sw_fc, t_obs, has_fc)}


def score_track_a(R, tol):
    """The registered verdict.  R: arrays over ALL registered seeds (NaN / False where undefined): frozen_ok, crossed,
    t_obs, s_obs, t_sw, s_sw (= the cutoff's s*_run), t_c, t_fc, t_sw_fc, r_fc, nan_identical.

    base    = frozen s*_frozen and a crossing within the budget;
    scored  = base with t_c < t_obs, t_sw defined and r_obs finite; a scored run without a forecast (t_fc or t_sw,fc
              undefined; R4-unstable included) is a MISS: in C1 and C2's denominator, not within; excluded from C3, C4.
    Validity (otherwise every criterion UNRESOLVED): ≥ 60 base runs and ≥ 60 scored runs; t_c < t_obs in ≥ 90% of the
    base; the actual switch t_sw defined in every base run with t_c < t_obs; the NaN recomputation identical in every
    run with a cutoff.  PASS = C1-C4 all PASS (phase1c.outcome)."""
    bool_keys = ("crossed", "frozen_ok", "nan_identical")
    R = {k: (P1._arr(v, bool) if k in bool_keys else P1._arr(v)) for k, v in R.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = R["s_obs"] / R["s_sw"] - 1
    before = P1.cutoff_before(R["t_c"], R["t_obs"])
    has_fc = P1.is_forecast(R["t_fc"], R["t_sw_fc"])
    base = R["frozen_ok"] & R["crossed"]
    scored = base & before & np.isfinite(R["t_sw"]) & np.isfinite(r_obs)
    cv = P1.cutoff_validity(before, base)
    withc = np.isfinite(R["t_c"])
    V = {"min_60_crossings": int(base.sum()) >= MIN_CROSS, "min_60_scored": int(scored.sum()) >= MIN_CROSS,
         "cutoff_before_crossing_90pct": cv["ok"],
         "actual_switch_defined": bool(np.all(np.isfinite(R["t_sw"][base & before]))),
         "nan_recomputation_identical": bool(np.all(R["nan_identical"][withc])) if withc.any() else True}
    valid = all(V.values())
    out = {"a": A, "f": F, "tolerances": dict(tol), "n_runs": int(len(base)), "validity": V, "valid": valid,
           "cutoff_validity": cv, "n_base_crossing_runs": int(base.sum()), "n_scored": int(scored.sum()),
           "n_scored_miss": int((scored & ~has_fc).sum()), "n_scored_with_forecast": int((scored & has_fc).sum()),
           "n_crossed_cutoff_not_before": int((base & ~before).sum()),
           "n_tsw_undefined_in_base_before": int((base & before & ~np.isfinite(R["t_sw"])).sum()),
           "n_nan_recompute_checked": int(withc.sum()),
           "n_nan_recompute_differs": int((withc & ~R["nan_identical"]).sum()),
           "scored_index": np.nonzero(scored)[0].tolist()}
    S = lambda k: R[k][scored]                                                  # noqa: E731
    crit = criteria(tol, S("t_fc"), S("t_sw_fc"), S("r_fc"), S("t_obs"), S("t_sw"), r_obs[scored], has_fc[scored])
    for k, c in crit.items():
        out[k] = c if valid else {**c, "verdict_if_valid": c["verdict"], "verdict": "UNRESOLVED"}
    out["outcome"] = "UNRESOLVED (validity)" if not valid else P1.outcome({k: out[k]["verdict"] for k in crit})
    out["falsifier_C4_fails"] = out["C4"]["verdict"] == "FAIL"
    return out


# ------------------------------------------------------------------------------------------ tables (run + observation)
def table(rows, obs, f=F, key="primary"):
    """Arrays over the given runs (sorted by seed) for score_track_a / the pilot rule, for forecast `key` at f."""
    ob = {o["seed"]: o for o in obs}
    R = {k: [] for k in ("seed", "frozen_ok", "crossed", "t_obs", "s_obs", "t_sw", "s_sw", "t_c", "t_fc", "t_sw_fc",
                         "r_fc", "nan_identical", "P_relerr", "status")}
    nan = float("nan")
    g = lambda d, k: d.get(k) if d.get(k) is not None else nan                 # noqa: E731
    for r in sorted(rows, key=lambda r: r["seed"]):
        o = ob.get(r["seed"], {})
        c = (r.get("fc") or {}).get(str(f), {})
        fc = c.get(key) or {}
        of = (o.get("by_f") or {}).get(str(f), {})
        R["seed"].append(r["seed"])
        R["frozen_ok"].append(r.get("status") != "no frozen switch")
        R["crossed"].append(bool(o.get("crossed", False)))
        R["t_obs"].append(g(o, "t_obs"))
        R["s_obs"].append(g(o, "s_obs"))
        R["t_sw"].append(g(of, "t_sw"))
        R["s_sw"].append(g(of, "s_sw"))
        R["t_c"].append(g(c, "t_c"))
        R["t_fc"].append(g(fc, "t_fc"))
        R["t_sw_fc"].append(g(fc, "t_sw_fc"))
        R["r_fc"].append(g(fc, "r_fc"))
        R["nan_identical"].append(bool(fc.get("nan_recompute_identical", True)) if c.get("t_c") is not None else True)
        R["status"].append(fc.get("status", r.get("status")))
        Pt, Ph = of.get("P_true_at_t_sw"), fc.get("P_hat")
        R["P_relerr"].append((np.asarray(Ph) / np.asarray(Pt) - 1).tolist() if Pt is not None and Ph is not None
                             else [nan] * 3)
    return R


def _ceil_to(x, m):
    return round(float(math.ceil(x / m - 1e-12) * m), 10)


def pilot_errors(R):
    """The forecast error on the pilot runs: pilot set = frozen, crossed, t_c < t_obs, t_sw defined, r_obs finite (the
    scored-run definition); errors over the pilot-set runs WITH a forecast."""
    R = {k: (np.asarray(v, bool) if k in ("crossed", "frozen_ok", "nan_identical") else
             (np.asarray(v, float) if k not in ("status", "seed") else v)) for k, v in R.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = R["s_obs"] / R["s_sw"] - 1
    before = P1.cutoff_before(R["t_c"], R["t_obs"])
    base = R["frozen_ok"] & R["crossed"]
    ps = base & before & np.isfinite(R["t_sw"]) & np.isfinite(r_obs)
    has = P1.is_forecast(R["t_fc"], R["t_sw_fc"])
    w = ps & has
    e1 = np.abs(R["t_fc"][w] - R["t_obs"][w])
    e2 = np.abs((R["t_fc"][w] - R["t_sw_fc"][w]) - (R["t_obs"][w] - R["t_sw"][w]))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = r_obs[w] / R["r_fc"][w]
    ratio = ratio[np.isfinite(ratio)]
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else None              # noqa: E731
    miss = {}
    for i in np.nonzero(ps & ~has)[0]:
        miss[R["status"][i]] = miss.get(R["status"][i], 0) + 1
    return {"n_seeds": int(len(base)), "n_frozen_ok": int(R["frozen_ok"].sum()), "n_crossed": int(base.sum()),
            "n_cutoff_before": int((base & before).sum()), "n_pilot_set": int(ps.sum()), "n_with_forecast": int(w.sum()),
            "n_miss": int((ps & ~has).sum()), "miss_status": miss,
            "nan_identical_all": bool(np.all(R["nan_identical"][np.isfinite(R["t_c"])])),
            "abs_err_cross": {"q50": q(e1, 50), "q90": q(e1, 90), "max": q(e1, 100)},
            "abs_err_lag": {"q50": q(e2, 50), "q90": q(e2, 90), "max": q(e2, 100)},
            "ratio_r_obs_over_r_fc": {"n": int(len(ratio)), "q10": q(ratio, 10), "q25": q(ratio, 25),
                                      "median": q(ratio, 50), "q75": q(ratio, 75), "q90": q(ratio, 90),
                                      "sd": float(np.std(ratio, ddof=1)) if len(ratio) > 1 else None},
            "_mask": ps, "_has": has, "_r_obs": r_obs}


def pilot_tolerances(err):
    """THE PILOT RULE (page): τ_cross = 1.5 × pilot q90 |t_fc − t_obs| and τ_lag = 1.5 × pilot q90 |lag error|, each
    rounded up to a multiple of 5; b = |median r_obs/r_fc − 1| + 2SE, rounded up to 0.05, SE = 1.2533·(IQR/1.349)/√n
    (Phase 1A's pilot rule).  Percentiles: numpy linear interpolation."""
    rr = err["ratio_r_obs_over_r_fc"]
    se = 1.2533 * ((rr["q75"] - rr["q25"]) / 1.349) / math.sqrt(rr["n"])
    return {"tau_cross": int(_ceil_to(TOL_FACTOR * err["abs_err_cross"]["q90"], TOL_STEP)),
            "tau_lag": int(_ceil_to(TOL_FACTOR * err["abs_err_lag"]["q90"], TOL_STEP)),
            "band": _ceil_to(abs(rr["median"] - 1) + 2 * se, TOL_BAND), "se_median": se, "se_rule": SE_MEDIAN}


def pilot_alternatives(err):
    """DISCLOSURE ONLY (not the rule): the exploration's SE (1.2533·sd/√n) and Phase 1A's floors (τ_lag ≥ 5 via
    max(1.5·q90, 1); b ≥ 0.10), to show whether either would change a number."""
    rr = err["ratio_r_obs_over_r_fc"]
    se_sd = 1.2533 * rr["sd"] / math.sqrt(rr["n"])
    se_iqr = 1.2533 * ((rr["q75"] - rr["q25"]) / 1.349) / math.sqrt(rr["n"])
    return {"band_with_sd_se": _ceil_to(abs(rr["median"] - 1) + 2 * se_sd, TOL_BAND), "se_sd": se_sd,
            "phase1a_floors": {"tau_lag": int(_ceil_to(max(TOL_FACTOR * err["abs_err_lag"]["q90"], 1), TOL_STEP)),
                               "band": _ceil_to(max(0.10, abs(rr["median"] - 1) + 2 * se_iqr), TOL_BAND)}}


# ------------------------------------------------------------------------------------------ pilot (pilot seeds only)
def pilot():
    """The 40 pilot seeds (PILOT seeds only): train, every forecast (the frozen rules asserted), observation.  Resumable;
    the memory gate before every seed.  -> results/trackA/pilot_runs.jsonl"""
    _setup()
    assert_rules_frozen()
    land = _land()
    assert land["validated"] and land["a"] == A
    fz = _frozen("pilot")
    f = OUT / "pilot_runs.jsonl"
    done = {r["seed"] for r in _rows(f)}
    for seed in PILOT_SEEDS:
        if seed in done:
            continue
        memory_gate(f"pilot {seed}")
        r = run_one(seed, fz[seed])
        if r.get("status") == "ok":
            r["observed_PILOT_ONLY"] = observe_one(r)
        _append_write(f, r)
        p = (r.get("fc") or {}).get(str(F), {}).get("primary") or {}
        print(json.dumps(_jsonable({"seed": seed, "t_c": p.get("t_c"), "t_fc": p.get("t_fc"),
                                    "t_obs": (r.get("observed_PILOT_ONLY") or {}).get("t_obs"),
                                    "nan_ok": p.get("nan_recompute_identical"), "secs": r.get("secs_run")})),
              flush=True)
        rss_guard()


def _desc_stats(R, tol):
    err = pilot_errors(R)
    m, has, r_obs = err.pop("_mask"), err.pop("_has"), err.pop("_r_obs")
    A_ = {k: np.asarray(v, float) for k, v in R.items() if k in ("t_fc", "t_sw_fc", "r_fc", "t_obs", "t_sw")}
    err["criteria_at_pilot_tolerances_IN_SAMPLE"] = {
        k: {kk: v.get(kk) for kk in ("frac_within", "n_within", "n_miss", "median_ratio", "verdict", "ci95", "n")}
        for k, v in criteria(tol, A_["t_fc"][m], A_["t_sw_fc"][m], A_["r_fc"][m], A_["t_obs"][m], A_["t_sw"][m],
                             r_obs[m], has[m]).items()}
    pe = np.asarray([p for p, ok in zip(R["P_relerr"], m & has) if ok], float)
    if len(pe):
        err["P_hat_relerr_vs_P_at_actual_t_sw"] = {"coords": ["w1", "b1", "b2"],
                                                   "median_abs": np.nanmedian(np.abs(pe), axis=0).tolist(),
                                                   "q90_abs": np.nanpercentile(np.abs(pe), 90, axis=0).tolist()}
    return err


def pilot_summary():
    """results/trackA/pilot.json: the pilot forecast error (primary rule at f = 0.90) and the pilot tolerances; the same
    statistics for the descriptive variants and f = 0.95; the rule hashes the pilot ran with."""
    rules = assert_rules_frozen()
    rows = _rows(OUT / "pilot_runs.jsonl")
    assert sorted(r["seed"] for r in rows) == list(PILOT_SEEDS), "every pilot seed exactly once"
    obs = [r["observed_PILOT_ONLY"] for r in rows if r.get("observed_PILOT_ONLY")]
    R = table(rows, obs, F, "primary")
    err = pilot_errors(R)
    for k in ("_mask", "_has", "_r_obs"):
        err.pop(k)
    tol = pilot_tolerances(err)
    out = {"label": "TRACK A PILOT (pilot seeds 9,774,000-039 only; NOT registered data)", "a": A, "f": F,
           "rule_sha256": rules["sha256"], "primary": {"ext": EXT, "P_rule": PRULE},
           "pilot_error": err, "tolerances": tol, "alternatives_DISCLOSURE_ONLY": pilot_alternatives(err),
           "primary_with_criteria": _desc_stats(R, tol),
           "DESCRIPTIVE": {k: _desc_stats(table(rows, obs, F, k), tol) for k in DESC_VARIANTS},
           "secs_run_total": round(sum(r.get("secs_run", 0) for r in rows), 1)}
    out["DESCRIPTIVE"]["f_0.95_primary_rule"] = _desc_stats(table(rows, obs, F_DESC, "primary"), tol)
    sr = [abs(((r.get("fc") or {}).get(str(F), {}).get("primary") or {}).get("s_run", np.nan)
              - (r.get("fc") or {}).get(str(F), {}).get("cutoff_s_run", np.nan)) for r in rows if r.get("fc")]
    out["max_abs_forecaster_minus_harness_s_run"] = float(np.nanmax(sr)) if sr else None
    out["horizon_steps_t_obs_minus_t_c"] = _horizon(rows, obs)
    (OUT / "pilot.json").write_text(json.dumps(_jsonable(out), indent=1) + "\n")
    print(json.dumps(_jsonable({"tolerances": tol, "pilot_error": err}), indent=1))
    return out


def _horizon(rows, obs):
    ob = {o["seed"]: o for o in obs}
    h = [ob[r["seed"]]["t_obs"] - r["fc"][str(F)]["t_c"] for r in rows
         if r.get("fc") and r["seed"] in ob and ob[r["seed"]].get("t_obs") is not None
         and r["fc"][str(F)].get("t_c") is not None]
    return {"q10": float(np.percentile(h, 10)), "median": float(np.median(h)), "q90": float(np.percentile(h, 90))} \
        if h else None


# ------------------------------------------------------------------------------------------ registration manifest
FROZEN_DATA = ("results/designs/trackA_causal_adam_design.md", "results/trackA_registration.md",
               "results/trackA/seed_scan.json", "results/trackA/landscape.json", "results/trackA/frozen_rules.json",
               "results/trackA/frozen_pilot.jsonl", "results/trackA/pilot_runs.jsonl", "results/trackA/pilot.json",
               "results/trackA/frozen_registered.jsonl", "tests/test_trackA_causal.py",
               "tests/test_causal_forecast.py", "results/lag_law/kappa.csv",
               "results/designs/trackA_explore/explore.py", "results/designs/trackA_explore/explore2.py",
               "results/designs/trackA_explore/explore_runs.jsonl", "results/designs/trackA_explore/explore2_runs.jsonl",
               "results/designs/trackA_explore/summary2.json")


def code_closure(start=("trackA_causal",)):
    """Every src module reached by relative imports (phase1c's rule)."""
    import ast
    seen, todo = set(), list(start)
    while todo:
        m = todo.pop()
        if m in seen or not (ROOT / "src" / f"{m}.py").exists():
            continue
        seen.add(m)
        tree = ast.parse((ROOT / "src" / f"{m}.py").read_text())
        for nd in ast.walk(tree):
            if isinstance(nd, ast.ImportFrom) and nd.level == 1:
                if nd.module:
                    todo.append(nd.module.split(".")[0])
                else:
                    todo.extend(a.name for a in nd.names)
    return sorted(f"src/{m}.py" for m in seen)


def manifest_files():
    return sorted(set(code_closure()) | set(FROZEN_DATA))


def manifest():
    """results/trackA/registration.sha256 (for the registration commit)."""
    lines = []
    for rel in manifest_files():
        p = ROOT / rel
        assert p.exists(), f"missing {rel}"
        lines.append(f"{_sha(p)}  {rel}")
    (OUT / "registration.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


_assert_committed = P1._assert_committed


def stamp():
    """registration_stamp.txt: the registration commit and the SHA-256 of the registration file and the manifest."""
    _assert_committed(OUT / "registration.sha256")
    h = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(OUT / "registration.sha256")], cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()
    txt = (f"Track A (causal Adam) registration\nregistration commit: {h}\n"
           f"sha256 results/trackA_registration.md: {_sha(REGISTRATION_MD)}\n"
           f"sha256 results/trackA/registration.sha256: {_sha(OUT / 'registration.sha256')}\n")
    (OUT / "registration_stamp.txt").write_text(txt)
    print(txt)


def assert_registration():
    """Before any registered training: the OpenTimestamps proof EXISTS (checked first), every manifest hash equal, the
    manifest and the stamp committed and clean, the frozen rules unchanged."""
    if not (OUT / "registration_stamp.txt.ots").exists():
        raise SystemExit("REFUSED: no OpenTimestamps proof (results/trackA/registration_stamp.txt.ots): "
                         "no registered training")
    m = OUT / "registration.sha256"
    for ln in m.read_text().splitlines():
        h, rel = ln.split()
        assert _sha(ROOT / rel) == h, f"registration hash mismatch: {rel}"
    _assert_committed(m)
    _assert_committed(OUT / "registration_stamp.txt")
    assert_rules_frozen()


def tolerances():
    """The registered tolerances: the pilot rule's output in the hashed pilot.json."""
    t = json.loads((OUT / "pilot.json").read_text())["tolerances"]
    return {k: t[k] for k in ("tau_cross", "tau_lag", "band")}


# ------------------------------------------------------------------------------------------ run / finalize / observe / score
FORBIDDEN = {"t_obs", "s_obs", "crossed", "t_sw", "observed_PILOT_ONLY", "P_true_at_t_sw"}


def run():
    """AFTER the registration commit is pushed and OpenTimestamped: every registered seed (resumable)."""
    assert_registration()
    _setup()
    fz = _frozen("registered")
    f = OUT / "runs.jsonl"
    done = {r["seed"] for r in _rows(f)}
    for seed in SEEDS:
        if seed in done:
            continue
        memory_gate(f"run {seed}")
        r = run_one(seed, fz[seed])
        _append_write(f, r)
        p = (r.get("fc") or {}).get(str(F), {}).get("primary") or {}
        print(json.dumps(_jsonable({"seed": seed, "t_c": p.get("t_c"), "t_fc": p.get("t_fc"),
                                    "nan_ok": p.get("nan_recompute_identical"), "secs": r.get("secs_run")})),
              flush=True)
        rss_guard()


def finalize():
    """forecasts.sha256 over runs.jsonl: every seed exactly once, no observed quantity in any row."""
    rows = _rows(OUT / "runs.jsonl")
    assert sorted(r["seed"] for r in rows) == list(SEEDS), "every seed exactly once"
    assert not (FORBIDDEN & P1._keys(rows)), "an observed quantity in a runs row"
    (OUT / "forecasts.sha256").write_text(f"{_sha(OUT / 'runs.jsonl')}  runs.jsonl\n")


def _assert_forecasts():
    h = (OUT / "forecasts.sha256").read_text().split()[0]
    assert _sha(OUT / "runs.jsonl") == h, "forecast hash mismatch"
    _assert_committed(OUT / "runs.jsonl")
    _assert_committed(OUT / "forecasts.sha256")


def observe():
    """AFTER the forecasts commit: retrain each run (hash asserted) and observe."""
    assert_registration()
    _assert_forecasts()
    _setup()
    f = OUT / "observed.jsonl"
    done = {r["seed"] for r in _rows(f)}
    for row in _rows(OUT / "runs.jsonl"):
        if row["seed"] in done or row.get("status") != "ok":
            continue
        memory_gate(f"observe {row['seed']}")
        o = observe_one(row)
        _append_write(f, o)
        rss_guard()


def score():
    """scores.json: the registered verdict and the DESCRIPTIVE statistics (P̂ error; 1C's extrapolation; P at t_c − 1;
    f = 0.95)."""
    assert_registration()
    _assert_forecasts()
    rows, obs = _rows(OUT / "runs.jsonl"), _rows(OUT / "observed.jsonl")
    tol = tolerances()
    res = score_track_a(table(rows, obs, F, "primary"), tol)
    res["DESCRIPTIVE"] = {"primary_errors": _desc_stats(table(rows, obs, F, "primary"), tol),
                          **{k: _desc_stats(table(rows, obs, F, k), tol) for k in DESC_VARIANTS},
                          "f_0.95_primary_rule": _desc_stats(table(rows, obs, F_DESC, "primary"), tol)}
    res["forecasts_sha256"] = _sha(OUT / "forecasts.sha256")
    (OUT / "scores.json").write_text(json.dumps(_jsonable(res), indent=1) + "\n")
    print(json.dumps(_jsonable({k: res[k] for k in ("outcome", "validity")})))


def main(argv):
    cmd = argv[1]
    if cmd == "gate":
        sys.exit(gate())
    fn = {"scan": scan, "landscape": landscape, "freeze_rules": freeze_rules, "pilot": pilot,
          "pilot_summary": pilot_summary, "manifest": manifest, "stamp": stamp, "run": run, "finalize": finalize,
          "observe": observe, "score": score}
    if cmd == "freeze":
        freeze(argv[2])
    elif cmd in fn:
        fn[cmd]()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv)
