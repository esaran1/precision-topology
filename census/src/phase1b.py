"""Phase 1B (author's program): POST HOC causal re-scoring of the already-run registered tests with the FROZEN causal
forecaster.

POST HOC.  Nothing here is a registration and no registered verdict changes: every registered verdict is read from its
committed scores.json and printed beside the re-score unchanged.  No writer input, revision plan or reply draft.

THE FORECASTER.  src/causal_forecast.py at commit 25aac9e (quad extrapolation of log s, window 5% (at least 50 steps),
horizon 1.6·s_ref; GuardedArray enforcement), called UNCHANGED through `causal_forecast.run_forecast` with Phase 1C's
configuration (`phase1c.config`).  `assert_forecaster` checks, before any forecast, that the file's SHA-256 equals its
entry in results/phase1c/registration.sha256 and that its git blob equals the blob at 25aac9e.

THE RUNS.  The registered runs of Track A (a = 1.65, SGD and Adam), Test 2A (a = 1.85, Adam), GELU-T (random and branch
arms) and W2-A (T, D, T′).  Only the REGISTERED SCORED SET of each arm is re-scored (Track A: crossed, rule point before
the crossing, finite r_traj and r_obs, as track_a.score_opt; the others: the committed `scored` column).  Each saved path
(untracked, results/<test>/paths/) is loaded only after its SHA-256 is asserted equal to the committed predictions.csv
row (the prediction-inputs audit's rule).  Track A / 2A Adam moments are rebuilt by the audit's `adam_replay` (local torch
Generator), asserting the replayed θ equals the saved path bit for bit.  Observed crossings (step_obs, s_obs) and the
actual lag-free switch (G, W2-A: t_sw, s_switch) come from the committed observed CSVs; nothing is re-detected.  Width 1:
the actual switch is Phase 1C's W1 rule, the first step with s ≥ s*_run of the cutoff's causal rule-point branch (the
path's first passage of a level; no placement or gap is evaluated).

DECISIONS (fixed in this file and committed BEFORE any re-score outcome was computed):
  f        Phase 1C's f per family (registration 6433edc): width 1 (Track A SGD and Adam, Test 2A) f = 0.90 with the cutoff
           on s*_run at the causal rule point (1C's W1 rule, `causal_forecast.w1_cutoff_on_s_run`); GELU-T (both arms)
           and W2-A (T, D, T′) f = 0.95.  Every arm is also reported at the other f (0.95 width 1; 0.90 otherwise),
           DESCRIPTIVELY.
  Adam P   the forecaster's existing width-1 Adam adapter.  Each test keeps its own registered P rule, made causal:
           Track A Adam: P = 1/(√v̂ + ε) at the causal rule point (`adam_P = "rule_point"`; Track A froze P at its rule
           point).  Test 2A: P at t_c − 1 (`adam_P = "cutoff"`; the causal analogue of 2A's registered P at t_sw, which
           lies after the cutoff).  The other choice is reported for both, DESCRIPTIVELY.
  criteria Phase 1C's C1-C4 (+ S for T′) with the tolerances of the corresponding 1C family: width 1 W1's
           (τ_cross 10, τ_lag 15, b 0.20); GELU-T G's (10, 10, 0.15); W2-A T T's (15, 5, 0.10); T′ T′'s (30, 5, 0.10).
           W2-A D: NO tolerance was ever set (D was not piloted in Phase 1A and has no 1C arm): raw errors only, no
           verdict.  C4: phase1c.criterion_c4 (10,000 resamples, numpy default_rng(9350000), local).
  scored   the registered scored set with the cutoff strictly before the crossing (t_c < t_obs; width 1 also: the
           actual switch defined).  A scored run without a forecast is a MISS.  The 1C validity rules are reported
           beside the criteria (cutoff before the crossing in ≥ 90% of the registered scored set; NaN recomputation
           identical; width 1: ≥ 60 runs), with the 1C-rule outcome (UNRESOLVED if one fails).
  L1-L5    each test's REGISTERED criteria (Track A L1-L3, Test 2A A1-A3, GELU-T and W2-A L1-L5) recomputed by the
           registered function, unchanged, with r_fc (s_fc, t_fc) in place of the registered trajectory prediction over
           the scored runs with a forecast; every other input (r_obs, r_cf, validity inputs) is the committed one.

Machine rules: one process, nice 15, one thread; a memory gate (free ≥ 25%, swap free ≥ 500 MB; act_fold.check_memory)
logged to results/phase1b/memory_gate.log, run as a separate step before every job (`gate`) and inside the job between
runs; stop above 3 GB RSS.  No global torch / numpy RNG state.

    python -m src.phase1b gate TAG                       # memory gate (separate, logged step)
    python -m src.phase1b run track_a|track2a|gelu|w2a   # resumable -> results/phase1b/runs_<test>.jsonl
    python -m src.phase1b summarize                      # -> results/phase1b/summary.json
    python -m src.phase1b report                         # -> results/phase1b_results.md
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
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import causal_forecast as C
from . import phase1c as P1C

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase1b"
RESULTS_MD = RESULTS / "phase1b_results.md"
FORECASTER = ROOT / "src" / "causal_forecast.py"
FORECASTER_COMMIT = "25aac9e"
FORECASTER_SHA256 = "d6039f50c202531b6f255fb90af185a0df867232eaafbde12db881efb3a1750f"
LABEL = "PHASE 1B: POST HOC causal re-scoring (no registered verdict changes)"

# ------------------------------------------------------------------------------------------ decisions (before outcomes)
TESTS = {"track_a": ("track_a/sgd", "track_a/adam"), "track2a": ("track2a/adam",),
         "gelu": ("gelu/random", "gelu/branch"), "w2a": ("w2a/T", "w2a/D", "w2a/Tp")}
ARMS = tuple(a for v in TESTS.values() for a in v)
KIND = {a: ("w1" if a.startswith("track") else "gelu" if a.startswith("gelu") else "w2a") for a in ARMS}
TOL_KEY = {"track_a/sgd": "W1", "track_a/adam": "W1", "track2a/adam": "W1", "gelu/random": "G", "gelu/branch": "G",
           "w2a/T": "T", "w2a/D": None, "w2a/Tp": "Tp"}           # the Phase 1C family whose tolerances apply
F = {a: (0.90 if KIND[a] == "w1" else 0.95) for a in ARMS}
F_OTHER = {a: (0.95 if KIND[a] == "w1" else 0.90) for a in ARMS}
ADAM_P = {"track_a/adam": "rule_point", "track2a/adam": "cutoff"}
ADAM_P_OTHER = {"track_a/adam": "cutoff", "track2a/adam": "rule_point"}
ARM_LABEL = {"track_a/sgd": "Track A SGD (a = 1.65)", "track_a/adam": "Track A Adam (a = 1.65)",
             "track2a/adam": "Test 2A Adam (a = 1.85)", "gelu/random": "GELU-T random", "gelu/branch": "GELU-T branch",
             "w2a/T": "W2-A T", "w2a/D": "W2-A D", "w2a/Tp": "W2-A T′"}
W1_MIN = P1C.W1_MIN
ARTIFACTS = ("phase1b/runs_track_a.jsonl", "phase1b/runs_track2a.jsonl", "phase1b/runs_gelu.jsonl",
             "phase1b/runs_w2a.jsonl", "phase1b/summary.json")
RUNS_FILE = {"track_a": "runs_track_a.jsonl", "track2a": "runs_track2a.jsonl", "gelu": "runs_gelu.jsonl",
             "w2a": "runs_w2a.jsonl"}
RSS_LIMIT = 3 * 1024 ** 3
REPRO_RTOL = 1e-9


def test_of(arm_id):
    return arm_id.split("/")[0]


def arm_of(arm_id):
    return arm_id.split("/")[1]


def config(arm_id, f, adam_P=None):
    """Phase 1C's frozen configuration (quad / 0.05 / ≥ 50 steps / horizon 1.6) at cutoff fraction f; width-1 Adam: the
    given P rule."""
    cfg = P1C.config("W1", f)
    return cfg if adam_P is None else C.ForecastConfig(**{**cfg.__dict__, "adam_P": adam_P})


# ------------------------------------------------------------------------------------------ the frozen forecaster
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assert_forecaster():
    """src/causal_forecast.py is the frozen forecaster: its SHA-256 equals the registered Phase 1C manifest entry and
    this module's constant, and its git blob equals the blob at 25aac9e."""
    h = _sha(FORECASTER)
    reg = dict(reversed(ln.split()) for ln in (RESULTS / "phase1c" / "registration.sha256").read_text().splitlines())
    assert h == FORECASTER_SHA256 == reg["src/causal_forecast.py"], "causal_forecast.py differs from the frozen forecaster"
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    blob = git("rev-parse", f"{FORECASTER_COMMIT}:./src/causal_forecast.py")
    assert blob and blob == git("hash-object", "src/causal_forecast.py"), "causal_forecast.py differs from 25aac9e"
    return h


# ------------------------------------------------------------------------------------------ registered tables
def _bool(s):
    return pd.Series(s).fillna(False).astype(bool).to_numpy()


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce").to_numpy(float) if c in d else np.full(len(d), np.nan)


def registered_table(arm_id):
    """All runs of an arm, from the committed observed CSV (registered order), with `reg_scored` (the registered scored
    set) and `path_sha256` (asserted equal to the committed predictions.csv row)."""
    t, arm = test_of(arm_id), arm_of(arm_id)
    if t == "track_a":
        d = pd.read_csv(RESULTS / "track_a" / "observed_runs.csv")
        d = d[d.opt == arm].reset_index(drop=True)
        d["reg_scored"] = (_bool(d.crossed) & _bool(d.rule_before_cross) & np.isfinite(_num(d, "r_traj"))
                           & np.isfinite(_num(d, "r_obs")))
        pr = pd.read_csv(RESULTS / "track_a" / "predictions.csv")
        pr = pr[pr.opt == arm]
    elif t == "track2a":
        d = pd.read_csv(RESULTS / "track2a" / "observed_runs.csv")
        d["reg_scored"] = _bool(d.scored)
        pr = pd.read_csv(RESULTS / "track2a" / "predictions.csv")
    elif t == "gelu":
        d = pd.read_csv(RESULTS / "gelu_transfer" / f"observed_runs_{arm}.csv")
        d["reg_scored"] = _bool(d.scored)
        pr = pd.read_csv(RESULTS / "gelu_transfer" / "predictions.csv")
        pr = pr[pr.arm == arm]
    else:
        d = pd.read_csv(RESULTS / "width2_asym" / f"observed_runs_{arm}.csv")
        d["reg_scored"] = _bool(d.scored)
        pr = pd.read_csv(RESULTS / "width2_asym" / "predictions.csv")
        pr = pr[pr.arm == arm]
    ph = dict(zip(pr.seed.astype(int), pr.path_sha256))
    assert all(ph[int(s)] == h for s, h in zip(d.seed, d.path_sha256)), "observed and predictions path hashes differ"
    return d


def registered_scores(arm_id):
    """The committed registered verdict block of an arm (scores.json), unchanged."""
    t, arm = test_of(arm_id), arm_of(arm_id)
    if t == "track_a":
        return json.loads((RESULTS / "track_a" / "scores.json").read_text())[arm]
    if t == "track2a":
        return json.loads((RESULTS / "track2a" / "scores.json").read_text())
    sub = "gelu_transfer" if t == "gelu" else "width2_asym"
    return json.loads((RESULTS / sub / "scores.json").read_text())["arms"][arm]


REG_CRITERIA = {"track_a": ("L1", "L2", "L3"), "track2a": ("A1", "A2", "A3"), "gelu": ("L1", "L2", "L3", "L4", "L5"),
                "w2a": ("L1", "L2", "L3", "L4", "L5")}


def registered_outcome(arm_id, sc):
    """Track A registered no combined outcome (its L verdicts are listed); the others' committed `outcome`."""
    if test_of(arm_id) == "track_a":
        return "no combined outcome (" + ", ".join(f"{k} {sc[k]['verdict']}" for k in REG_CRITERIA["track_a"]) + ")"
    return sc["outcome"]


# ------------------------------------------------------------------------------------------ cases (one run's inputs)
def _nan_from(a, t_c):
    b = np.array(a, float, copy=True)
    b[int(t_c):] = np.nan
    return b


class W1Case:
    """A width-1 run (Track A / 2A): θ path W (budget+1, 4); Adam moments rebuilt to the largest cutoff."""
    kind = "w1"

    def __init__(self, a, x, y, s_frozen, opt, lr, budget, W, M=None, V=None):
        self.a, self.x, self.y, self.s_frozen, self.opt, self.lr, self.budget = a, x, y, float(s_frozen), opt, lr, budget
        self.W, self.M, self.V = np.asarray(W, float), M, V
        self.s = np.abs(self.W[:, 2])
        self._cut = {}

    def cutoff(self, f):
        """HARNESS SIDE: 1C's nested stopping time on s*_run (reads s_0 … s_t and the hidden state at t_R(t) < t)."""
        f = float(f)
        if f not in self._cut:
            W = self.W
            t_c, tR, s_run = C.w1_cutoff_on_s_run(self.s, lambda k: W[k, 2], f, self.s_frozen, self.a, self.x, self.y,
                                                  lambda k: W[k, [0, 1, 3]])
            self._cut[f] = (t_c, {"cutoff_rule_point": tR, "cutoff_s_run": s_run})
        return self._cut[f]

    def inputs(self, t_c, guarded=True, nan=False):
        out, hid = self.W[:, 2], self.W[:, [0, 1, 3]]
        M, V = self.M, self.V
        if nan:
            out, hid = _nan_from(out, t_c), _nan_from(hid, t_c)
            M = None if M is None else _nan_from(M, t_c)
            V = None if V is None else _nan_from(V, t_c)
        gd = (lambda a_, **kw: C.guard(a_, t_c, **kw)) if guarded else (lambda a_, **kw: a_)
        return C.W1Inputs(a=self.a, x=self.x, y=self.y, s_frozen=self.s_frozen, opt=self.opt, lr=self.lr, t_c=int(t_c),
                          budget=self.budget, out=gd(out, name="w2"), hid=gd(hid, max_rows=1, name="hidden"),
                          M=None if M is None else gd(M, name="M"), Vhat=None if V is None else gd(V, name="Vhat"))


class GeluCase:
    """A GELU-T run: w₂ path `out`, hidden (w₁, b₁, b₂) path `hid` (only row 0, the release, readable); the occupied
    copy's frozen row and its own-sample branch grid (GBranch from the frozen copy point; no path input)."""
    kind = "gelu"

    def __init__(self, out, hid, copy_row, branch, act, budget):
        self.out, self.hid = np.asarray(out, float), np.asarray(hid, float)
        self.copy_row, self.branch, self.act, self.budget = copy_row, branch, act, int(budget)
        self.s = np.abs(self.out)
        self.s_ref = float(copy_row["s_switch"])

    def cutoff(self, f):
        return C.cutoff_step(self.s, float(f) * self.s_ref), {}

    def inputs(self, t_c, guarded=True, nan=False):
        out, hid = (_nan_from(self.out, t_c), _nan_from(self.hid, t_c)) if nan else (self.out, self.hid)
        gd = (lambda a_, **kw: C.guard(a_, t_c, **kw)) if guarded else (lambda a_, **kw: a_)
        return C.GeluInputs(t_c=int(t_c), budget=self.budget, out=gd(out, name="w2"), hid=gd(hid, rows={0}, name="hidden"),
                            copy_row=self.copy_row, branch=self.branch, act=self.act)


class W2ACase:
    """A W2-A run: v path `out` (n, 2), z path `hid` (n, 5) (only row 0, the release, readable); the occupied copy's
    frozen row, the windings fixed at release, the own sample."""
    kind = "w2a"

    def __init__(self, arm, out, hid, copy_row, windings, x, y, copy, budget):
        self.arm, self.out, self.hid = arm, np.asarray(out, float), np.asarray(hid, float)
        self.copy_row, self.windings, self.x, self.y, self.copy = copy_row, tuple(windings), x, y, copy
        self.budget = int(budget)
        self.s = np.abs(self.out).sum(axis=1)
        self.s_ref = float(copy_row["s_switch"])

    def cutoff(self, f):
        return C.cutoff_step(self.s, float(f) * self.s_ref), {}

    def inputs(self, t_c, guarded=True, nan=False):
        out, hid = (_nan_from(self.out, t_c), _nan_from(self.hid, t_c)) if nan else (self.out, self.hid)
        gd = (lambda a_, **kw: C.guard(a_, t_c, **kw)) if guarded else (lambda a_, **kw: a_)
        return C.W2AInputs(arm=self.arm, t_c=int(t_c), budget=self.budget, out=gd(out, name="v"),
                           hid=gd(hid, rows={0}, name="hidden"), copy_row=self.copy_row, windings=self.windings,
                           x=self.x, y=self.y, copy=self.copy)


def forecast_one(case, cfg):
    """ONE causal forecast at cfg.f.  The harness locates t_c (a stopping time); the frozen forecaster
    (`causal_forecast.run_forecast`, which re-asserts every guard's last row < t_c) receives GUARDED views of rows < t_c
    only; then the per-run NaN recomputation (1C's): the same adapter, unguarded, on copies with every row ≥ t_c NaN."""
    t0 = time.time()
    rec = {"f": float(cfg.f)}
    if case.kind == "w1" and case.opt == "adam":
        rec["adam_P"] = cfg.adam_P
    t_c, extra = case.cutoff(cfg.f)
    rec.update(extra)
    if t_c is None:
        return {**rec, "t_c": None, "status": "no cutoff: s never reaches f·s_ref", "secs": round(time.time() - t0, 2)}
    fc = C.run_forecast(case.kind, case.inputs(t_c, guarded=True), cfg)
    fc2 = C.ADAPTERS[case.kind](case.inputs(t_c, guarded=False, nan=True), cfg)
    rec.update(fc)
    rec["t_sw_fc"] = fc.get("t_sw")
    rec["nan_recompute_identical"] = bool(P1C.same_forecast(fc, fc2))
    rec["secs"] = round(time.time() - t0, 2)
    return rec


# ------------------------------------------------------------------------------------------ loading one registered run
def _path_of(arm_id, seed):
    t, arm = test_of(arm_id), arm_of(arm_id)
    if t == "track_a":
        from . import track_a as TA
        return TA._path_file(arm, int(seed))
    if t == "track2a":
        return RESULTS / "track2a" / "paths" / f"adam_{int(seed)}.npz"
    if t == "gelu":
        from . import gelu_transfer as G
        return G._path_file(arm, int(seed))
    from . import width2_asym as W
    return W._path_file(arm, int(seed))


def load_verified(arm_id, r):
    """The saved path, after asserting its SHA-256 equals the committed row's."""
    p = _path_of(arm_id, r.seed)
    h = _sha(p)
    assert h == r.path_sha256, f"path hash mismatch {p}"
    return (np.load(p) if p.suffix == ".npy" else np.load(p)["W"]), p.name, h


def make_case(arm_id, r, X):
    """The case of one registered run (X: its verified saved array)."""
    t, arm = test_of(arm_id), arm_of(arm_id)
    if t in ("track_a", "track2a"):
        from . import track2a as T2
        from . import track_a as TA
        return w1_case(TA.A if t == "track_a" else T2.A, int(r.seed), float(r.s_frozen), arm, X)
    if t == "gelu":
        from . import gelu_transfer as G
        cf = G._frozen()[int(r.seed)]["copies"][str(int(r.copy_at_release))]
        assert float(cf["s_switch"]) == float(r.s_switch), "frozen s_switch differs from the committed row"
        return gelu_case(int(r.seed), cf, X[:, 2], X[:, [0, 1, 3]], G.BUDGET)
    from . import width2_asym as W
    cf = W._frozen()[int(r.seed)]["copies"][r.copy_at_release]
    assert float(cf["s_switch"]) == float(r.s_switch_frozen), "frozen s_switch differs from the committed row"
    return w2a_case(arm, int(r.seed), cf, json.loads(r.windings), r.copy_at_release, X[:, W.VI], X[:, W.ZI],
                    int(r.budget))


def w1_case(a, seed, s_frozen, opt, W):
    """Width-1 case (Track A's protocol: make_data(200, seed), SGD lr 0.3 / Adam lr 0.01, budget 32,000) at activation a."""
    from . import track_a as TA
    x, y = TA._sample(int(seed))
    return W1Case(a, x, y, float(s_frozen), opt, TA.LR_ADAM if opt == "adam" else TA.LR_SGD, TA.BUDGET, W)


def gelu_case(seed, cf, out, hid, budget):
    """GELU-T case from the occupied copy's frozen row cf: its own-sample branch grid exactly as the registered
    predict_one builds it (GBranch from the frozen copy point at s₀; no path input)."""
    from . import gelu_transfer as G
    land = G._land()
    P = G.own_problem(int(seed))
    B = G.GBranch(land["s0"], np.array(cf["z_s0"]), P, 0.95 * land["s0"], G.S_HI_FRAC * land["s_pop"],
                  G.GRID_H_FRAC * land["s_pop"])
    return GeluCase(out, hid, cf, B, P.act, budget)


def w2a_case(arm, seed, cf, windings, copy, out, hid, budget):
    """W2-A case from the occupied copy's frozen row cf, the windings fixed at release and the seed's own sample."""
    from . import width2_asym as W
    x, y = W.own_sample(int(seed))
    return W2ACase(arm, out, hid, cf, windings, x, y, copy, budget)


def attach_adam(case, arm_id, seed, n):
    """Adam's m and bias-corrected v̂ (rows 0 … n) by the audit's replay (local Generator); the replayed θ must equal the
    saved path bit for bit."""
    from .prediction_inputs_audit import adam_replay
    Wr, M, V = adam_replay(int(seed), case.a, int(n), lr=case.lr)
    ok = bool(np.array_equal(Wr, case.W[:int(n) + 1]))
    assert ok, f"Adam replay differs from the saved path ({arm_id} {seed})"
    case.M, case.V = M, V
    return ok


def forecast_run(arm_id, r, X, pname, h):
    """Every forecast of one registered run: at f and the other f (Adam: also the other P rule); width 1: the actual
    switch t_sw (1C's W1 rule: first s ≥ s*_run of the arm-f cutoff's rule-point branch)."""
    t0 = time.time()
    case = make_case(arm_id, r, X)
    row = {"arm_id": arm_id, "seed": int(r.seed), "path": pname, "path_sha256": h, "path_sha256_ok": True,
           "budget": int(case.budget)}
    fs = (F[arm_id], F_OTHER[arm_id])
    cuts = {f: case.cutoff(f)[0] for f in fs}
    if case.kind == "w1" and case.opt == "adam":
        tcs = [v for v in cuts.values() if v is not None]
        row["adam_replay_bit_identical"] = attach_adam(case, arm_id, r.seed, max(tcs)) if tcs else None
    P_main = ADAM_P.get(arm_id)
    row["fc"] = {str(f): forecast_one(case, config(arm_id, f, P_main)) for f in fs}
    if arm_id in ADAM_P_OTHER:
        row["fc_other_P"] = {str(f): forecast_one(case, config(arm_id, f, ADAM_P_OTHER[arm_id])) for f in fs}
    if case.kind == "w1":
        s_run = row["fc"][str(F[arm_id])].get("cutoff_s_run")
        from . import linear_response as LR
        row["s_run_cutoff_branch"] = s_run
        row["t_sw_cutoff_branch"] = LR._first_ge(case.s, s_run) if s_run is not None else None
    row["secs"] = round(time.time() - t0, 1)
    return row


# ------------------------------------------------------------------------------------------ machine rules and io
def memory_gate(tag):
    """free ≥ 25% and swap free ≥ 500 MB (act_fold.check_memory); waits (re-checking every 60 s) while it fails; every
    check logged to results/phase1b/memory_gate.log."""
    from .act_fold import check_memory
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            ok, f, w = check_memory()
        except Exception:                                                 # noqa: BLE001
            ok, f, w = False, float("nan"), float("nan")
        with open(OUT / "memory_gate.log", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag} free={f}% swap_free={w}MB {'OK' if ok else 'WAIT'}\n")
        if ok:
            return f, w
        time.sleep(60)


def rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss            # bytes on macOS
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")


def _setup():
    if os.nice(0) < 15:
        os.nice(15 - os.nice(0))
    import torch
    torch.set_num_threads(1)


def _runs_path(test):
    return OUT / RUNS_FILE[test]


def _rows(p):
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def run(test, n=None):
    """Every registered scored run of the test's arms (resumable; the memory gate before each run; one row per run)."""
    _setup()
    assert_forecaster()
    OUT.mkdir(parents=True, exist_ok=True)
    f = _runs_path(test)
    done = {(r["arm_id"], r["seed"]) for r in _rows(f)}
    k = 0
    for arm_id in TESTS[test]:
        d = registered_table(arm_id)
        for r in d[d.reg_scored].sort_values("seed").itertuples():
            if (arm_id, int(r.seed)) in done:
                continue
            if n is not None and k >= int(n):
                return
            memory_gate(f"run {arm_id} {r.seed}")
            X, pname, h = load_verified(arm_id, r)
            row = forecast_run(arm_id, r, X, pname, h)
            del X
            P1C._append_write(f, row)
            fc = row["fc"][str(F[arm_id])]
            print(json.dumps(P1C._jsonable({"arm": arm_id, "seed": row["seed"], "t_c": fc.get("t_c"),
                                            "status": fc.get("status"), "nan_ok": fc.get("nan_recompute_identical"),
                                            "secs": row["secs"]})), flush=True)
            k += 1
            rss_guard()


# ------------------------------------------------------------------------------------------ pure scoring (re-score)
def _q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else None


def raw_errors(t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs, has_fc):
    """RAW errors over the given runs (no tolerance, no verdict): |t_fc − t_obs| and |lag_fc − lag_obs| (min, q10,
    median, q90, max; signed medians), r_obs/r_fc (q10, median, q90), D = |t_fc − t_obs| − |t_sw,fc − t_obs| (mean with
    phase1c's bootstrap interval, the run count with D < 0), over the runs with a forecast; misses counted."""
    A = P1C._arr
    t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs = (A(v) for v in (t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs))
    has_fc = A(has_fc, bool)
    ec, el = t_fc - t_obs, (t_fc - t_sw_fc) - (t_obs - t_sw)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(has_fc & (r_fc != 0), r_obs / r_fc, np.nan)
    ok = has_fc & np.isfinite(t_fc) & np.isfinite(t_sw_fc) & np.isfinite(t_obs)
    D = np.abs(t_fc[ok] - t_obs[ok]) - np.abs(t_sw_fc[ok] - t_obs[ok])
    mean, lo, hi = P1C.bootstrap_mean_ci(D) if len(D) >= 2 else (float("nan"),) * 3

    def st(v):
        v = np.where(has_fc, v, np.nan)
        return {"min": _q(np.abs(v), 0), "q10": _q(np.abs(v), 10), "median": _q(np.abs(v), 50), "q90": _q(np.abs(v), 90),
                "max": _q(np.abs(v), 100), "signed_median": _q(v, 50)}
    return {"n": int(len(t_fc)), "n_miss": int((~has_fc).sum()), "abs_err_cross": st(ec), "abs_err_lag": st(el),
            "ratio_r_obs_over_r_fc": {"n": int(np.isfinite(q).sum()), "q10": _q(q, 10), "median": _q(q, 50),
                                      "q90": _q(q, 90)},
            "D_vs_no_lag": {"n": int(len(D)), "mean": mean, "ci95": [lo, hi], "n_D_negative": int((D < 0).sum())}}


def rescore(tol_key, base, t_c, t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs, nan_identical, width1):
    """The 1C-style re-score of one arm at one f.  base: the registered scored set.  Scored: base with t_c < t_obs (and
    t_sw, r_obs defined).  Criteria: phase1c.criteria with the family's tolerances (tol_key None: no tolerance, raw
    errors only).  1C validity (reported; the 1C-rule outcome is UNRESOLVED if one fails): t_c < t_obs in ≥ 90% of the
    base, the NaN recomputation identical in every base run with a cutoff, width 1: ≥ 60 base and scored runs."""
    A = P1C._arr
    base = A(base, bool)
    t_c, t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs = (A(v) for v in (t_c, t_fc, t_sw_fc, r_fc, t_obs, t_sw, r_obs))
    nan_identical = A(nan_identical, bool)
    before = P1C.cutoff_before(t_c, t_obs)
    scored = base & before & np.isfinite(t_sw) & np.isfinite(r_obs)
    has_fc = P1C.is_forecast(t_fc, t_sw_fc)
    cv = P1C.cutoff_validity(before, base)
    withc = base & np.isfinite(t_c)
    V = {"cutoff_before_crossing_90pct": bool(cv["ok"]),
         "nan_recomputation_identical": bool(np.all(nan_identical[withc])) if withc.any() else True}
    if width1:
        V["W1_min_60_base"] = int(base.sum()) >= W1_MIN
        V["W1_min_60_scored"] = int(scored.sum()) >= W1_MIN
    S = lambda v: v[scored]                                                  # noqa: E731
    out = {"n_registered_scored": int(base.sum()), "cutoff_validity": cv, "n_scored": int(scored.sum()),
           "n_scored_miss": int((scored & ~has_fc).sum()), "n_scored_with_forecast": int((scored & has_fc).sum()),
           "n_base_no_cutoff": int((base & ~np.isfinite(t_c)).sum()),
           "n_base_cutoff_not_before": int((base & np.isfinite(t_c) & ~before).sum()),
           "n_base_no_actual_switch": int((base & before & ~np.isfinite(t_sw)).sum()),
           "n_nan_recompute_checked": int(withc.sum()), "n_nan_recompute_differs": int((withc & ~nan_identical).sum()),
           "validity_1c": V, "valid_1c": all(V.values()), "tol_key": tol_key,
           "raw": raw_errors(S(t_fc), S(t_sw_fc), S(r_fc), S(t_obs), S(t_sw), S(r_obs), S(has_fc)),
           "horizon": P1C.horizons(S(t_c), S(t_obs), S(t_sw), S(t_obs) - S(t_sw))}
    if tol_key is None:
        out["criteria"] = None
        out["outcome_criteria"] = out["outcome_1c_rule"] = "no tolerance set: raw errors only"
    else:
        crit = P1C.criteria(tol_key, S(t_fc), S(t_sw_fc), S(r_fc), S(t_obs), S(t_sw), S(r_obs), S(has_fc))
        oc = P1C.outcome({k: v["verdict"] for k, v in crit.items()})
        out.update(criteria=crit, outcome_criteria=oc, outcome_1c_rule=oc if out["valid_1c"] else "UNRESOLVED (validity)")
    out["scored_mask"] = scored
    out["has_fc_mask"] = has_fc
    out["before_mask"] = before
    return out


# ------------------------------------------------------------------------------------------ registered functions
def registered_L(arm_id, d, r_traj, s_traj=None, t_traj=None, before=None, r_cf=None):
    """The test's REGISTERED criteria by its registered function, unchanged.  r_traj (s_traj, t_traj): the prediction
    (the committed one, or r_fc); before: Track A's rule-before-crossing argument; r_cf: the closed form (Track A L2,
    2A's secondary), default the committed one."""
    t, arm = test_of(arm_id), arm_of(arm_id)
    r_cf = _num(d, "r_cf") if r_cf is None else np.asarray(r_cf, float)
    if t == "track_a":
        from . import track_a as TA
        rb = _bool(d.rule_before_cross) if before is None else np.asarray(before, bool)
        return TA.score_opt(_num(d, "r_obs"), r_traj, r_cf, _bool(d.crossed), rb)
    if t == "track2a":
        from . import track2a as T2
        out = T2.score_2a(_num(d, "r_obs"), r_traj, _bool(d.crossed), _num(d, "step_obs"), _num(d, "t_rule"),
                          _num(d, "t_sw"), r_secondary=r_cf)
        out.pop("scored_index", None)
        return out
    fol = _bool(d.follows_branch)
    if t == "gelu":
        from . import gelu_transfer as G
        pil = json.loads((G.OUT / "pilot.json").read_text())["pilot_median_chi_tsw"][arm]
        out = G.score_arm(arm, _bool(d.on_branch), _bool(d.hold_G_positive), _bool(d.crossed), _num(d, "step_obs"),
                          _num(d, "s_obs"), _num(d, "s_switch"), s_traj, r_traj, r_cf, _num(d, "kappa"), _num(d, "t_sw"),
                          _num(d, "eta_lam"), _num(d, "lag_steps_pred"), _num(d, "chi_tsw"), _num(d, "chi_window_max"),
                          pil, G._land()["s_glob"], follows=fol)
    else:
        from . import width2_asym as W
        pil = json.loads((W.OUT / "pilot.json").read_text())["arms"][arm]["pilot_median_chi_tsw"]
        pil = float("nan") if pil is None else pil
        out = W.score_arm(arm, _bool(d.on_branch), _bool(d.hold_G_positive), _bool(d.crossed), _num(d, "step_obs"),
                          _num(d, "s_obs"), _num(d, "s_switch"), s_traj, t_traj, r_traj, r_cf, _num(d, "kappa"),
                          _num(d, "t_sw"), _num(d, "eta_lam"), _num(d, "lag_steps_pred"), _num(d, "chi_tsw"),
                          _num(d, "chi_window_max"), pil, _num(d, "s_pop_branch"), follows=fol)
    out.pop("scored_index", None)
    return out


def _crit_core(c):
    """The comparable core of a criterion block (verdict and its statistic)."""
    keep = ("verdict", "n", "median_ratio", "spearman", "frac_within_10pct", "mean_D", "ci95")
    return {k: c[k] for k in keep if k in c}


def same_core(a, b):
    """Two criterion cores agree: verdicts and counts exactly, statistics to relative 1e-9 (the committed CSVs keep 16
    decimal places; small r values (T′ r_cf ~ 1e-3) carry that rounding to ~1e-13 relative in a median or a mean)."""
    from .prediction_inputs_audit import rel_equal
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, list) or isinstance(y, list):
            if not (isinstance(x, list) and isinstance(y, list) and len(x) == len(y)
                    and all(rel_equal(u, v, REPRO_RTOL) for u, v in zip(x, y))):
                return False
        elif isinstance(x, str) or isinstance(y, str):
            if x != y:
                return False
        elif not rel_equal(x, y, REPRO_RTOL):
            return False
    return True


def registered_reproduced(arm_id, d):
    """The registered function on the committed predictions reproduces the committed verdicts and statistics."""
    sc = registered_scores(arm_id)
    rec = registered_L(arm_id, d, _num(d, "r_traj"), _num(d, "s_traj"), _num(d, "t_traj"))
    rec = json.loads(json.dumps(P1C._jsonable(rec)))
    ok = all(same_core(_crit_core(rec[k]), _crit_core(sc[k])) for k in REG_CRITERIA[test_of(arm_id)])
    if "outcome" in sc:
        ok = ok and rec.get("outcome") == sc["outcome"]
    return bool(ok)


# ------------------------------------------------------------------------------------------ assembling one arm
def _g(o, k):
    v = (o or {}).get(k)
    return np.nan if v is None else v


def arm_arrays(arm_id, d, rows, key="fc", f=None):
    """Per run (registered order; NaN where not forecast): the forecast at f (`key` fc or fc_other_P) and the observed /
    actual quantities (G, W2-A: committed t_sw, s_switch; width 1: the arm-f cutoff branch's switch)."""
    f = F[arm_id] if f is None else f
    by = {r["seed"]: r for r in rows if r["arm_id"] == arm_id}
    R = {k: [] for k in ("t_c", "t_fc", "t_sw_fc", "s_fc", "r_fc", "nan_identical", "t_sw", "s_sw", "s_run_cutoff",
                         "have_row")}
    for s in d.seed.astype(int):
        row = by.get(int(s), {})
        fc = (row.get(key) or {}).get(str(f)) or {}
        R["have_row"].append(bool(row))
        for k in ("t_c", "t_fc", "t_sw_fc", "s_fc", "r_fc"):
            R[k].append(_g(fc, k))
        R["nan_identical"].append(bool(fc.get("nan_recompute_identical", True)) if fc.get("t_c") is not None else True)
        R["s_run_cutoff"].append(_g(row, "s_run_cutoff_branch"))
        if KIND[arm_id] == "w1":
            R["t_sw"].append(_g(row, "t_sw_cutoff_branch"))
            R["s_sw"].append(_g(row, "s_run_cutoff_branch"))
    R = {k: np.array(v, dtype=bool if k in ("nan_identical", "have_row") else float) for k, v in R.items() if v}
    R["t_obs"], R["s_obs"] = _num(d, "step_obs"), _num(d, "s_obs")
    if KIND[arm_id] != "w1":
        R["t_sw"], R["s_sw"] = _num(d, "t_sw"), _num(d, "s_switch")
    with np.errstate(invalid="ignore", divide="ignore"):
        R["r_obs"] = R["s_obs"] / R["s_sw"] - 1
    return R


def _status_counts(arm_id, rows, mask_seeds, key="fc", f=None):
    f = F[arm_id] if f is None else f
    c = Counter()
    for r in rows:
        if r["arm_id"] == arm_id and r["seed"] in mask_seeds:
            c[((r.get(key) or {}).get(str(f)) or {}).get("status", "?")] += 1
    return dict(sorted(c.items()))


def score_arm_1b(arm_id, d, rows):
    """POST HOC re-score of one arm: the 1C-style criteria at f (and, descriptively, at the other f and, Adam, with the
    other P rule), the registered criteria with r_fc, horizons, raw errors, and the committed registered verdict."""
    base = d.reg_scored.to_numpy(bool)
    seeds = d.seed.astype(int).to_numpy()
    assert {r["seed"] for r in rows if r["arm_id"] == arm_id} == set(seeds[base].tolist()), \
        f"{arm_id}: the runs file must hold exactly the registered scored set"
    w1 = KIND[arm_id] == "w1"
    out = {"arm_id": arm_id, "label": ARM_LABEL[arm_id], "f": F[arm_id], "f_other": F_OTHER[arm_id],
           "tol_key": TOL_KEY[arm_id], "adam_P": ADAM_P.get(arm_id), "adam_P_other": ADAM_P_OTHER.get(arm_id),
           "n_runs": int(len(d)), "n_registered_scored": int(base.sum())}
    sc = registered_scores(arm_id)
    out["registered"] = {"outcome": registered_outcome(arm_id, sc),
                         **{k: _crit_core(sc[k]) for k in REG_CRITERIA[test_of(arm_id)]},
                         "reproduced_by_registered_function": registered_reproduced(arm_id, d)}

    def one(key, f):
        R = arm_arrays(arm_id, d, rows, key, f)
        if w1 and f != F[arm_id]:                     # 1C: the actual switch from the arm-f cutoff's branch
            R0 = arm_arrays(arm_id, d, rows, "fc", F[arm_id])
            R["t_sw"], R["s_sw"], R["r_obs"] = R0["t_sw"], R0["s_sw"], R0["r_obs"]
        res = rescore(TOL_KEY[arm_id], base, R["t_c"], R["t_fc"], R["t_sw_fc"], R["r_fc"], R["t_obs"], R["t_sw"],
                      R["r_obs"], R["nan_identical"], w1)
        scm, hf, bf = res.pop("scored_mask"), res.pop("has_fc_mask"), res.pop("before_mask")
        fs = scm & hf
        with np.errstate(invalid="ignore"):
            L = registered_L(arm_id, d, np.where(fs, R["r_fc"], np.nan), np.where(fs, R["s_fc"], np.nan),
                             np.where(fs, R["t_fc"], np.nan), before=bf,
                             r_cf=np.where(fs, _num(d, "r_cf"), np.nan) if test_of(arm_id).startswith("track") else None)
        res["registered_criteria_on_r_fc"] = {"outcome": L.get("outcome"), "valid": L.get("valid"),
                                              "n_scored": L.get("n_scored"),
                                              **{k: _crit_core(L[k]) for k in REG_CRITERIA[test_of(arm_id)]}}
        res["miss_statuses"] = _status_counts(arm_id, rows, set(seeds[scm & ~hf].tolist()), key, f)
        res["no_cutoff_or_late_statuses"] = _status_counts(arm_id, rows, set(seeds[base & ~bf].tolist()), key, f)
        res["n_scored_with_forecast_hit_in_extrapolated_part"] = int(sum(
            bool(((r.get(key) or {}).get(str(f)) or {}).get("hit_in_extrapolated_part"))
            for r in rows if r["arm_id"] == arm_id and r["seed"] in set(seeds[fs].tolist())))
        return res
    out["at_f"] = one("fc", F[arm_id])
    out["at_f_other_DESCRIPTIVE"] = one("fc", F_OTHER[arm_id])
    if arm_id in ADAM_P_OTHER:
        out["other_P_DESCRIPTIVE"] = {"adam_P": ADAM_P_OTHER[arm_id], "at_f": one("fc_other_P", F[arm_id]),
                                      "at_f_other": one("fc_other_P", F_OTHER[arm_id])}
    if w1:
        R = arm_arrays(arm_id, d, rows)
        reg = _num(d, "s_run")
        okm = base & np.isfinite(R["s_run_cutoff"])
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(R["s_run_cutoff"][okm] / reg[okm] - 1)
        out["s_run_cutoff_vs_registered"] = {"n": int(okm.sum()), "n_equal_rel_1e-9": int((rel <= 1e-9).sum()),
                                             "max_rel_diff": float(rel.max()) if len(rel) else None}
    return out


def _guard_audit(rows):
    """Every forecast in the rows: the last row each guard read is < t_c; NaN recomputations identical; paths hashed."""
    n_fc = n_bad = n_nan = 0
    for r in rows:
        for key in ("fc", "fc_other_P"):
            for fc in (r.get(key) or {}).values():
                if fc.get("t_c") is None:
                    continue
                n_fc += 1
                n_bad += int(max((fc.get("max_index_read") or {}).values() or [-1]) >= fc["t_c"])
                n_nan += int(not fc.get("nan_recompute_identical"))
    return {"n_forecasts_with_cutoff": n_fc, "n_read_at_or_after_cutoff": n_bad, "n_nan_recompute_differs": n_nan,
            "n_rows": len(rows), "n_path_sha256_ok": sum(bool(r.get("path_sha256_ok")) for r in rows),
            "n_adam_replay_bit_identical": sum(r.get("adam_replay_bit_identical") is True for r in rows),
            "n_adam_rows": sum("fc_other_P" in r for r in rows)}


def build_summary():
    """summary.json from the committed runs files and the committed registered CSVs / scores (no path is read)."""
    out = {"label": LABEL, "forecaster": {"file": "src/causal_forecast.py", "commit": FORECASTER_COMMIT,
                                          "sha256": FORECASTER_SHA256},
           "decisions": {"f": F, "f_other": F_OTHER, "tol_key": TOL_KEY, "adam_P": ADAM_P, "adam_P_other": ADAM_P_OTHER,
                         "config": {k: v for k, v in P1C.config("W1", 0.9).__dict__.items() if k not in ("f", "adam_P")},
                         "tolerances": {k: {"tau_cross": P1C.TAU_CROSS[k], "tau_lag": P1C.TAU_LAG[k], "band": P1C.BAND[k]}
                                        for k in P1C.ARMS}},
           "runs_files": {}, "arms": {}}
    for test in TESTS:
        rows = _rows(_runs_path(test))
        out["runs_files"][RUNS_FILE[test]] = {"sha256": _sha(_runs_path(test)), **_guard_audit(rows)}
        for arm_id in TESTS[test]:
            out["arms"][arm_id] = score_arm_1b(arm_id, registered_table(arm_id), rows)
    return P1C._jsonable(out)


def summarize():
    S = build_summary()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    print(json.dumps({a: (v["registered"]["outcome"], v["at_f"]["outcome_criteria"]) for a, v in S["arms"].items()}))
    return S


# ------------------------------------------------------------------------------------------ results page
def _f(v, d=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def _cell(c):
    if not c or "verdict" not in c:
        return "—"
    v = c["verdict"]
    if "n_within" in c:
        return f"{v} ({c['n_within']}/{c['n']}, misses {c['n_miss']})"
    if "median_ratio" in c and "band" in c:
        return f"{v} (median {_f(c['median_ratio'])}, n {c['n']})"
    if "ci95" in c and "boot_n" in c:
        lo, hi = c["ci95"]
        return f"{v} ([{_f(lo, 1)}, {_f(hi, 1)}], n {c['n']})"
    if "n_both_negative" in c:
        return f"{v} ({c['n_both_negative']}/{c['n_eligible']}, misses {c.get('n_eligible_miss', 0)})"
    return v


def _reg_cell(c):
    """A registered criterion block (verdict and its statistic)."""
    if not c:
        return "—"
    for k, d in (("median_ratio", 3), ("spearman", 3), ("frac_within_10pct", 3)):
        if k in c:
            return f"{c['verdict']} ({_f(c[k], d)}, n {c.get('n')})"
    if "ci95" in c:
        lo, hi = c["ci95"]
        return f"{c['verdict']} ([{_f(lo, 4)}, {_f(hi, 4)}], n {c.get('n')})"
    return c.get("verdict", "—")


def _hz(h, d=0):
    if not h or not h.get("n"):
        return "—"
    return f"{_f(h['min'], d)} / {_f(h['median'], d)} / {_f(h['max'], d)}"


def _raw_cell(r):
    a, l_, q = r["abs_err_cross"], r["abs_err_lag"], r["ratio_r_obs_over_r_fc"]
    D = r["D_vs_no_lag"]
    lo, hi = D["ci95"]
    return (f"{_f(a['median'], 1)} / {_f(a['q90'], 1)} / {_f(a['max'], 0)} | {_f(l_['median'], 1)} / {_f(l_['q90'], 1)} / "
            f"{_f(l_['max'], 0)} | {_f(q['q10'])} / {_f(q['median'])} / {_f(q['q90'])} | {_f(D['mean'], 1)} "
            f"[{_f(lo, 1)}, {_f(hi, 1)}], {D['n_D_negative']}/{D['n']}")


def render_md(S):
    A = S["arms"]
    tol = S["decisions"]["tolerances"]
    L = ["# Phase 1B: POST HOC causal re-scoring of the registered tests", "",
         "**POST HOC. No registered verdict changes.** The registered verdicts below are read unchanged from each test's "
         "committed `scores.json`; the causal re-score is a diagnostic beside them. Every number on this page comes from "
         "`results/phase1b/summary.json` (`src/phase1b.py summarize`), built from the committed runs files "
         "`results/phase1b/runs_*.jsonl` (`src/phase1b.py run`) and the committed registered CSVs. This page is generated "
         "by `src/phase1b.py report`.", "",
         "## 1. What was done", "",
         f"- **Forecaster:** `src/causal_forecast.py` at {S['forecaster']['commit']} (SHA-256 "
         f"`{S['forecaster']['sha256'][:16]}…`, asserted equal to the Phase 1C manifest entry and to the git blob at "
         f"{S['forecaster']['commit']} before any forecast), called unchanged with Phase 1C's configuration "
         f"(family {S['decisions']['config']['family']}, window {S['decisions']['config']['window_frac']} of the visible "
         f"steps (at least {S['decisions']['config']['min_window']}), horizon {S['decisions']['config']['horizon_factor']}"
         "·s_ref). It reads guarded views of rows < t_c only; each forecast is recomputed with every row ≥ t_c NaN.",
         "- **Runs:** the registered scored set of each arm. Each saved path's SHA-256 was asserted equal to its committed "
         "predictions.csv row before use; Adam moments were rebuilt by the prediction-inputs audit's replay, bit-identical "
         "to the saved θ path. Crossings (step_obs, s_obs) and, for GELU-T and W2-A, the actual switch (t_sw, s_switch) "
         "are the committed observed values; nothing was re-detected. Width 1: the actual switch is Phase 1C's W1 rule "
         "(the first s ≥ s\\*_run of the cutoff's causal rule-point branch).",
         "- **Decisions, fixed in `src/phase1b.py` and committed before any re-score outcome was computed:**",
         "  - f per family (Phase 1C's): width 1 (Track A SGD and Adam, Test 2A) **f = 0.90** with the cutoff on s\\*_run "
         "at the causal rule point; GELU-T (both arms) and W2-A (T, D, T′) **f = 0.95**. Every arm is also reported at the "
         "other f (0.95 width 1; 0.90 otherwise), descriptively.",
         "  - Adam P (the forecaster's existing width-1 Adam adapter): **Track A Adam, P at the causal rule point** "
         "(Track A froze P at its rule point); **Test 2A, P at t_c − 1** (the causal analogue of 2A's registered P at "
         "t_sw, which lies after the cutoff). The other choice is reported for both, descriptively.",
         f"  - Tolerances (τ_cross / τ_lag / b) of the corresponding 1C family: width 1 W1's "
         f"{tol['W1']['tau_cross']} / {tol['W1']['tau_lag']} / {tol['W1']['band']}; GELU-T G's {tol['G']['tau_cross']} / "
         f"{tol['G']['tau_lag']} / {tol['G']['band']}; W2-A T T's {tol['T']['tau_cross']} / {tol['T']['tau_lag']} / "
         f"{tol['T']['band']}; T′ T′'s {tol['Tp']['tau_cross']} / {tol['Tp']['tau_lag']} / {tol['Tp']['band']}.",
         "  - **W2-A D: no tolerance was ever set.** D was not piloted in Phase 1A and has no Phase 1C arm, so it gets raw "
         "errors only, with no verdict.",
         "- **Scored:** the registered scored set with the cutoff strictly before the crossing (width 1 also needs the "
         "actual switch). A scored run without a forecast is a miss. The 1C validity rules are reported (cutoff before the "
         "crossing in ≥ 90% of the registered scored set; NaN recomputation identical; width 1 ≥ 60 runs); the 1C-rule "
         "outcome is UNRESOLVED if one fails.", "",
         "## 2. Side by side: registered verdict (unchanged) and POST HOC causal re-score", "",
         "| arm | registered verdict (committed) | registered criteria re-run with r_fc (POST HOC) | causal re-score at f, "
         "C1–C4 (+S) (POST HOC) | 1C-rule outcome (POST HOC) |", "|---|---|---|---|---|"]
    for a, o in A.items():
        reg = o["registered"]
        rf = o["at_f"]["registered_criteria_on_r_fc"]
        names = [k for k in rf if k[0] in "LA" and k[1:].isdigit()]
        rf_cell = (f"{rf['outcome']}: " if rf.get("outcome") else "") + ", ".join(f"{k} {rf[k]['verdict']}" for k in names)
        L.append(f"| {o['label']} | {reg['outcome']} | {rf_cell} | f = {o['f']}: {o['at_f']['outcome_criteria']} | "
                 f"{o['at_f']['outcome_1c_rule']} |")
    L += ["", "The registered verdicts are unchanged. Each one was reproduced from the committed predictions by the "
          "registered scoring function: " + ", ".join(f"{A[a]['label']} {'yes' if A[a]['registered']['reproduced_by_registered_function'] else 'NO'}"
                                                         for a in A) + ".", "",
          "## 3. Causal re-score at f, per arm (POST HOC)", "",
          "| arm | f | registered scored | cutoff before crossing | scored | misses | C1 \\|t_fc − t_obs\\| ≤ τ_cross | "
          "C2 \\|lag_fc − lag_obs\\| ≤ τ_lag | C3 median r_obs/r_fc ∈ [1 ± b] | C4 interval of mean D (PASS iff upper < 0) | "
          "S (T′) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, o in A.items():
        x = o["at_f"]
        cv = x["cutoff_validity"]
        c = x["criteria"] or {}
        L.append(f"| {o['label']} | {o['f']} | {x['n_registered_scored']} | {cv['n_cutoff_before_crossing']}/"
                 f"{cv['n_crossing_runs']} ({_f(cv['frac'])}) | {x['n_scored']} | {x['n_scored_miss']} | "
                 + (" | ".join(_cell(c.get(k)) for k in ("C1", "C2", "C3", "C4"))
                    if c else "no tolerance | no tolerance | no band | raw only (§5)") +
                 f" | {_cell(c.get('S')) if 'S' in c else '—'} |")
    L += ["", "1C validity conditions at f (POST HOC; the 1C-rule outcome is UNRESOLVED if one fails):", ""]
    for a, o in A.items():
        x = o["at_f"]
        L.append(f"- {o['label']}: " + ", ".join(f"{k} {'ok' if v else 'FAILS'}" for k, v in x["validity_1c"].items())
                 + f"; NaN recomputation differs {x['n_nan_recompute_differs']}/{x['n_nan_recompute_checked']}"
                 + (f"; misses by status {x['miss_statuses']}" if x["miss_statuses"] else "")
                 + (f"; registered scored runs not scored (no cutoff / cutoff not before the crossing / no actual "
                    f"switch): {x['n_base_no_cutoff']} / {x['n_base_cutoff_not_before']} / {x['n_base_no_actual_switch']}"
                    if x["n_scored"] < x["n_registered_scored"] else "") + ".")
    L += ["", "## 4. Registered criteria recomputed with r_fc in place of the registered prediction (POST HOC)", "",
          "The registered function, unchanged, over the scored runs with a forecast (r_traj := r_fc, s_traj := s_fc, "
          "t_traj := t_fc). Every other input is the committed one: r_obs (width 1: against the registered s\\*_run), r_cf "
          "(Track A's L2 and 2A's secondary are the registered closed form, not a causal forecast), and the validity "
          "inputs.", "",
          "| arm | registered (committed) | with r_fc at f |", "|---|---|---|"]
    for a, o in A.items():
        reg, rf = o["registered"], o["at_f"]["registered_criteria_on_r_fc"]
        names = [k for k in reg if k[0] in "LA" and k[1:].isdigit()]
        L.append(f"| {o['label']} | " + "; ".join(f"{k} {_reg_cell(reg[k])}" for k in names) + " | "
                 + "; ".join(f"{k} {_reg_cell(rf[k])}" for k in names)
                 + (f"; valid {_f(rf['valid'])}" if rf.get("valid") is not None else "") + " |")
    L += ["", "## 5. Raw errors and horizons at f (POST HOC; scored runs with a forecast)", "",
          "Cells: \\|t_fc − t_obs\\| median / q90 / max (steps) | \\|lag_fc − lag_obs\\| median / q90 / max | r_obs/r_fc "
          "q10 / median / q90 | mean D = \\|t_fc − t_obs\\| − \\|t_sw,fc − t_obs\\| [bootstrap 95%], runs with D < 0. "
          "W2-A D has no tolerance; these raw errors are all that is reported for it.", "",
          "| arm | \\|err cross\\| \\| \\|err lag\\| \\| ratio \\| D | t_obs − t_c (steps) | t_sw − t_c (steps) | "
          "t_obs − t_c (lags) |", "|---|---|---|---|---|"]
    for a, o in A.items():
        x = o["at_f"]
        h = x["horizon"]
        L.append(f"| {o['label']} | {_raw_cell(x['raw'])} | {_hz(h['cross_steps'])} | {_hz(h['switch_steps'])} | "
                 f"{_hz(h['cross_lags'], 2)} |")
    L += ["", "## 6. Descriptive: the other f, and the other Adam P rule (POST HOC)", "",
          "| arm | variant | cutoff before crossing | scored | misses | C1 | C2 | C3 | C4 | S | raw: \\|err cross\\| "
          "\\| \\|err lag\\| \\| ratio \\| D |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, o in A.items():
        vs = [(f"other f = {o['f_other']}", o["at_f_other_DESCRIPTIVE"])]
        if "other_P_DESCRIPTIVE" in o:
            p = o["other_P_DESCRIPTIVE"]
            vs += [(f"P {p['adam_P']}, f = {o['f']}", p["at_f"]), (f"P {p['adam_P']}, f = {o['f_other']}", p["at_f_other"])]
        for name, x in vs:
            cv = x["cutoff_validity"]
            c = x["criteria"] or {}
            L.append(f"| {o['label']} | {name} | {cv['n_cutoff_before_crossing']}/{cv['n_crossing_runs']} | "
                     f"{x['n_scored']} | {x['n_scored_miss']} | "
                     + (" | ".join(_cell(c.get(k)) for k in ("C1", "C2", "C3", "C4")) if c else "— | — | — | —")
                     + f" | {_cell(c.get('S')) if 'S' in c else '—'} | {_raw_cell(x['raw'])} |")
    L += ["", "Registered criteria with r_fc under each descriptive variant:", ""]
    for a, o in A.items():
        vs = [(f"other f = {o['f_other']}", o["at_f_other_DESCRIPTIVE"])]
        if "other_P_DESCRIPTIVE" in o:
            p = o["other_P_DESCRIPTIVE"]
            vs += [(f"P {p['adam_P']}, f = {o['f']}", p["at_f"]), (f"P {p['adam_P']}, f = {o['f_other']}", p["at_f_other"])]
        for name, x in vs:
            rf = x["registered_criteria_on_r_fc"]
            names = [k for k in rf if k[0] in "LA" and k[1:].isdigit()]
            L.append(f"- {o['label']}, {name}: " + "; ".join(f"{k} {_reg_cell(rf[k])}" for k in names) + ".")
    L += ["", "## 7. Checks", ""]
    for fn, v in S["runs_files"].items():
        L.append(f"- `{fn}`: {v['n_rows']} runs, every path hash asserted ({v['n_path_sha256_ok']}/{v['n_rows']}); "
                 f"{v['n_forecasts_with_cutoff']} forecasts with a cutoff, {v['n_read_at_or_after_cutoff']} read a row at "
                 f"or after it, {v['n_nan_recompute_differs']} NaN recomputations differ"
                 + (f"; Adam replay bit-identical {v['n_adam_replay_bit_identical']}/{v['n_adam_rows']}"
                    if v["n_adam_rows"] else "") + ".")
    for a, o in A.items():
        if "s_run_cutoff_vs_registered" in o:
            s = o["s_run_cutoff_vs_registered"]
            mx = "—" if s["max_rel_diff"] is None else f"{s['max_rel_diff']:.1e}"
            L.append(f"- {o['label']}: the cutoff branch's s\\*_run equals the registered s\\*_run (rel. ≤ 1e-9) in "
                     f"{s['n_equal_rel_1e-9']}/{s['n']} registered scored runs with a cutoff (max rel. difference {mx}).")
    L += ["", "## 8. Reproduce", "", "```",
          "python -m src.phase1b gate TAG; python -m src.phase1b run track_a|track2a|gelu|w2a   # needs the saved paths",
          "python -m src.phase1b summarize; python -m src.phase1b report",
          "```", ""]
    return "\n".join(L)


def report():
    S = json.loads((OUT / "summary.json").read_text())
    RESULTS_MD.write_text(render_md(S))
    print(RESULTS_MD)


def main(argv):
    cmd = argv[1]
    if cmd == "gate":
        print(memory_gate(" ".join(argv[2:]) or "gate"))
    elif cmd == "run":
        run(argv[2], argv[3] if len(argv) > 3 else None)
    elif cmd == "summarize":
        summarize()
    elif cmd == "report":
        report()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv)
