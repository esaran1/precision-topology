"""Phase 1A pilot: the extrapolation-error estimate of the causal forecaster (src/causal_forecast.py) on FRESH PILOT SEEDS.

Not a registration.  No registered seed is used; no registered run is read or re-scored (that is Phase 1B).  The
crossing of a PILOT run is measured with the existing exact detectors, for the error estimate only.

Settings (the ones Phase 1C will use):
  w1_sgd       width 1, f(t) = t + a·sin t at the CANDIDATE a = 1.58 (no existing crossing or training data at it:
               results/phase1a/a_candidate_scan.json), plain torch SGD lr 0.3, Track A's protocol (make_data(200, seed),
               U(−1, 1)⁴ in float32 from a LOCAL torch Generator, cast to float64; budget 32,000); s*_frozen = the own-sample
               switch of the landscape branch (Track A's R0).  Landscape at 1.58: Track A's construction (continuation in
               a at fixed s from the 1.60 certified switch point; validated by own_threshold.global_min).
  gelu_random  GELU-T's random-start arm, its registered pipeline (freeze_one, hold, release, ρ = 1, η = 0.03, 40,000).
  w2a_T        W2-A arm T, its registered pipeline (hold, ρ = 2⁻¹⁰, η = 0.03, budget 100,000), copy T.
  w2a_Tp       W2-A arm T′, its registered pipeline (ρ = 2⁻¹², budget 400,000), copy T′.
Seeds: dev (choice of the extrapolation family and window) and est (the error estimate), disjoint, fresh (verified by
`seed_scan`).

Saved per run (results/phase1a/paths/, untracked, SHA-256 in the rows): the output path only (w₂, or v), up to a margin
past the crossing, and the hidden state at release (GELU-T, W2-A) or at the candidate rule points (width 1).

    python -m src.phase1a_pilot scan          # a candidate and seed checks       -> a_candidate_scan.json, seed_scan.json
    python -m src.phase1a_pilot landscape     # width-1 landscape at a = 1.58     -> landscape_w1.json
    python -m src.phase1a_pilot run SETTING SPLIT [n]   # train + observe (resumable) -> runs_SETTING.jsonl, paths/
    python -m src.phase1a_pilot forecast SETTING SPLIT  # forecasts and errors       -> forecasts_SETTING.jsonl
    python -m src.phase1a_pilot summarize     # summary.json + error_estimate.md
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from . import causal_forecast as C
from . import linear_response as LR

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "phase1a"
PATHS = OUT / "paths"
A_W1 = 1.58
W1_LR, W1_BUDGET = 0.3, 32_000
F_GRID = (0.8, 0.9, 0.95, 0.98)
FAMILY_GRID = ("lin", "quad", "rate")
WINDOW_GRID = (0.05, 0.1, 0.25)
SETTINGS = ("w1_sgd", "w1_sgd_run", "gelu_random", "w2a_T", "w2a_Tp")     # w1_sgd_run: the same runs, cutoff on s*_run
SELECT = ("w1_sgd_run", "gelu_random", "w2a_T", "w2a_Tp")                  # dev selection and the proposed f
RUNS_OF = {"w1_sgd_run": "w1_sgd"}
# width 1: 9,310,100-119 were the first est range; the width-1 cutoff and window were revised after inspecting them, so
# they are DEV (dev2) and the width-1 est range is 9,310,200-239 (fresh).
SEEDS = {"w1_sgd": {"dev": tuple(range(9_310_000, 9_310_005)) + tuple(range(9_310_100, 9_310_120)),
                    "est": tuple(range(9_310_200, 9_310_240))},
         "gelu_random": {"dev": tuple(range(9_320_000, 9_320_005)), "est": tuple(range(9_320_100, 9_320_120))},
         "w2a_T": {"dev": tuple(range(9_330_000, 9_330_010)), "est": tuple(range(9_330_100, 9_330_160))},
         "w2a_Tp": {"dev": tuple(range(9_340_000, 9_340_005)), "est": tuple(range(9_340_100, 9_340_120))}}
# every committed artifact, by name (the ledger's provenance check reads these): run rows (`run`), forecast rows
# (`forecast`), test fixtures (`make_fixtures`), summary and report (`summarize`), scans (`scan`), landscape (`landscape`)
ARTIFACTS = ("phase1a/runs_w1_sgd.jsonl", "phase1a/runs_gelu_random.jsonl", "phase1a/runs_w2a_T.jsonl",
             "phase1a/runs_w2a_Tp.jsonl", "phase1a/forecasts_w1_sgd.jsonl", "phase1a/forecasts_w1_sgd_run.jsonl",
             "phase1a/forecasts_gelu_random.jsonl", "phase1a/forecasts_w2a_T.jsonl", "phase1a/forecasts_w2a_Tp.jsonl",
             "phase1a/fixtures/gelu_random.npz", "phase1a/fixtures/w2a_T.npz", "phase1a/summary.json",
             "phase1a/error_estimate.md", "phase1a/a_candidate_scan.json", "phase1a/seed_scan.json",
             "phase1a/landscape_w1.json")
SAVE_MARGIN = 500
RSS_LIMIT = 3 * 1024 ** 3


# ------------------------------------------------------------------------------------------ machine rules
def memory_gate(tag):
    """free ≥ 25% and swap free ≥ 500 MB (act_fold.check_memory); waits (re-checking every 60 s) while it fails; every
    check logged to results/phase1a/memory_gate.log."""
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
    os.nice(15)
    import torch
    torch.set_num_threads(1)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def _append_write(p, row):
    """Append one JSON row (the file is rewritten whole, so a partial line can never be left behind)."""
    old = p.read_text() if p.exists() else ""
    p.write_text(old + json.dumps(_jsonable(row)) + "\n")


def _rows(p):
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


# ------------------------------------------------------------------------------------------ scans (a and seeds)
def _scan_tree(patterns, chunk=16 * 1024 * 1024, overlap=256):
    """Regex scan of every text file under src, tests, results, paper (binary files skipped), streamed in 16 MB chunks
    with a 256-byte overlap (a 2.7 GB CSV is never held in memory).  Returns {pattern key: [files with a match]}."""
    import re
    rx = {k: re.compile(v.encode()) for k, v in patterns.items()}
    hits = {k: [] for k in patterns}
    for top in ("src", "tests", "results", "paper"):
        for dp, _, fns in os.walk(ROOT / top):
            if dp.rstrip("/").endswith("paths") or "phase1a" in dp:
                continue
            for fn in fns:
                p = Path(dp) / fn
                if fn in ("causal_forecast.py", "phase1a_pilot.py", "test_causal_forecast.py"):
                    continue
                if p.suffix in (".npz", ".npy", ".pdf", ".png", ".pt", ".pkl", ".ots", ".bak", ".gz", ".zip"):
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


def scan():
    """(1) The candidate a: literal occurrences of 1.58 as a number (and its float artefacts) and every numeric a-column
    value of every results CSV in [1.575, 1.585]; (2) the pilot seed ranges: no 7-digit seed of any range appears."""
    import csv
    OUT.mkdir(parents=True, exist_ok=True)
    hits = _scan_tree({"literal_1.58": r"(^|[^0-9.])1\.580*([^0-9]|$)", "float_artefact": r"1\.5799999|1\.580000000[0-9]"})
    col_hits = {}
    for dp, _, fns in os.walk(RESULTS):
        if "phase1a" in dp:
            continue
        for fn in fns:
            if not fn.endswith(".csv"):
                continue
            p = Path(dp) / fn
            try:
                with open(p, newline="") as fh:
                    rd = csv.reader(fh)
                    hdr = next(rd)
                    cols = [i for i, h in enumerate(hdr) if h.strip().lower() in ("a", "act_a", "parameter")]
                    if not cols:
                        continue
                    for row in rd:
                        for i in cols:
                            try:
                                v = float(row[i])
                            except (ValueError, IndexError):
                                continue
                            if 1.575 <= v <= 1.585:
                                col_hits.setdefault(str(p.relative_to(ROOT)), set()).add(v)
            except (OSError, StopIteration, csv.Error):
                continue
    a_scan = {"a": A_W1, "literal_files": hits["literal_1.58"], "float_artefact_files": hits["float_artefact"],
              "csv_a_column_hits": {k: sorted(v) for k, v in col_hits.items()},
              "note": ("literal_files: every text file with the token 1.58 (any role: a scale value, a statistic); "
                       "csv_a_column_hits: results CSVs whose a / act_a / parameter column has a value in [1.575, 1.585] "
                       "(training or crossing data at a = 1.58 would appear there).")}
    (OUT / "a_candidate_scan.json").write_text(json.dumps(_jsonable(a_scan), indent=1))
    pats = {}
    for st, d in SEEDS.items():
        for split, ss in d.items():
            p = str(ss[0])[:4]
            pats[f"{st}_{split}"] = rf"(^|[^0-9.]){p}[0-9]{{3}}([^0-9]|$)|{p[0]}_{p[1:4]}_[0-9]{{3}}"
    sh = _scan_tree(pats)
    seed_scan = {"ranges": {f"{st}_{sp}": [ss[0], ss[-1], len(ss)] for st, d in SEEDS.items() for sp, ss in d.items()},
                 "prefix_pattern_files": sh,
                 "registered_seed_overlap": registered_overlap()}
    (OUT / "seed_scan.json").write_text(json.dumps(_jsonable(seed_scan), indent=1))
    print(json.dumps(_jsonable({"a": a_scan, "seeds": seed_scan}), indent=1))


def registered_overlap():
    """Pilot seeds that are registered (or pilot) seeds of any test module (constants SEEDS / PILOT_SEEDS / SEEDS_*)."""
    import importlib
    mine = {s for d in SEEDS.values() for ss in d.values() for s in ss}
    found = {}
    for mod in ("track_a", "track2a", "track2b", "track2c", "gelu_transfer", "width2_asym"):
        m = importlib.import_module(f"src.{mod}")
        for name in dir(m):
            if name.startswith("SEEDS") or name.startswith("PILOT_SEEDS"):
                v = getattr(m, name)
                if isinstance(v, (tuple, list, range)):
                    found[f"{mod}.{name}"] = sorted(mine & set(int(s) for s in v))
    return found


# ------------------------------------------------------------------------------------------ width 1 at a = 1.58
def _w1_sample(seed):
    from .fold1d import make_data
    x, y = make_data(200, seed)
    return x.double().numpy(), y.double().numpy()


def landscape():
    """Track A's landscape construction at a = 1.58: continuation in a at fixed s from the 1.60 certified switch point
    (lag_law kappa.csv), principal copy, branch on the 800-point population, switch; validated by own_threshold.global_min
    (unplaced at 0.995·s*, placed at 1.005·s*, same branch up to the mirror and 2π)."""
    import pandas as pd
    from .own_threshold import global_min
    from .track_a import kappa_closed, principal_copy
    from .width2_conditional import population
    _setup()
    memory_gate("landscape w1")
    x, y = population()
    k = pd.read_csv(RESULTS / "lag_law" / "kappa.csv")
    r = k[k.a.round(2) == 1.60].iloc[0]
    z = np.array(json.loads(r.z_star))
    s0 = float(r.s_star)
    for aa in np.linspace(1.60, A_W1, 11)[1:]:
        z, res, H = LR.newton(z, s0, aa, x, y)
        assert res < 1e-9 and np.linalg.eigvalsh(H).min() > 0, (aa, res)
    z, kcopy = principal_copy(z, s0)
    z, res, _ = LR.newton(z, s0, A_W1, x, y)
    B = LR.Branch(s0, z, A_W1, x, y, 0.5 * s0, 1.5 * s0, 0.001 * s0)
    s_star = B.switch(s0)
    th, Hs, tan, res = B.exact(s_star)
    dG = LR.grad_gap(th[0], th[1], A_W1)
    tp = 2 * math.pi
    val = {}
    for f_ in (0.995, 1.005):
        Lb, w1, b1, G, _ = global_min(f_ * s_star, A_W1, x, y)
        thb, *_ = B.exact(f_ * s_star)
        same = min(abs(abs(w1) - abs(thb[0])), 9) < 2e-3 and \
            min(abs((b1 - thb[1]) % tp), tp - abs((b1 - thb[1]) % tp)) < 2e-3
        val[str(f_)] = {"global_min_w1": w1, "global_min_b1": b1, "global_min_G": G, "branch_w1": float(thb[0]),
                        "branch_b1": float(thb[1]), "same_branch_mod_mirror_2pi": bool(same)}
    ok = val["0.995"]["global_min_G"] <= 0 < val["1.005"]["global_min_G"] and val["1.005"]["same_branch_mod_mirror_2pi"]
    k_sgd, lam = kappa_closed(Hs, tan, dG, np.ones(3))
    out = {"a": A_W1, "s_star_pop": s_star, "theta_star": th, "H": Hs, "tangent": tan, "gradG": dG,
           "grad_residual": res, "H_pd": bool(np.linalg.eigvalsh(Hs).min() > 0), "winding_shift_from_1p60_copy": kcopy,
           "kappa_sgd": k_sgd, "lam_min": lam, "validation_global_min": val, "validated": bool(ok)}
    (OUT / "landscape_w1.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k_: v for k_, v in out.items() if k_ != "H"}), indent=1))
    if not ok:
        raise SystemExit("STOP: landscape switch not validated")


def _land_w1():
    return json.loads((OUT / "landscape_w1.json").read_text())


def w1_frozen(seed):
    """s*_frozen: Newton from the landscape switch point onto the own sample, continued; its switch (Track A's R0)."""
    d = _land_w1()
    s_pop = d["s_star_pop"]
    x, y = _w1_sample(seed)
    th, res, H = LR.newton(np.array(d["theta_star"]), s_pop, A_W1, x, y)
    if res > 1e-9 or np.linalg.eigvalsh(H).min() <= 0:
        return {"seed": seed, "s_frozen": None, "note": "no own-sample minimum"}
    B = LR.Branch(s_pop, th, A_W1, x, y, 0.3 * s_pop, 1.7 * s_pop, 0.002 * s_pop)
    s_fr = B.switch(s_pop)
    return {"seed": seed, "s_frozen": s_fr, "newton_res": res}


def w1_train(seed, opt="sgd", budget=W1_BUDGET, a=A_W1):
    """Track A's run (local torch Generator; no global RNG state).  Returns W (budget+1, 4) and, for Adam, M, V̂."""
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data
    f = activation("sin_family", a)
    x, y = make_data(200, seed)
    gen = torch.Generator().manual_seed(int(seed))
    th = torch.empty(4, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen).to(torch.float64).requires_grad_(True)
    x, y = x.double(), y.double()
    adam = opt == "adam"
    o = torch.optim.Adam([th], lr=1e-2) if adam else torch.optim.SGD([th], lr=W1_LR)
    W = np.empty((budget + 1, 4))
    M = np.zeros((budget + 1, 4)) if adam else None
    V = np.full((budget + 1, 4), np.nan) if adam else None
    W[0] = th.detach().numpy()
    for t in range(1, budget + 1):
        o.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        o.step()
        W[t] = th.detach().numpy()
        if adam:
            st = o.state[th]
            k = int(st["step"])
            M[t] = st["exp_avg"].numpy()
            V[t] = st["exp_avg_sq"].numpy() / (1 - 0.999 ** k)
    return W, M, V


def w1_observe(W, a=A_W1):
    """Every-step detection (phase2b_ordering.state, Track A's observe): the first placed step t ≥ 1."""
    import torch
    from .fold1d import activation
    from .phase2b_ordering import state
    f = activation("sin_family", a)
    for t in range(1, len(W)):
        if state(torch.tensor(W[t], dtype=torch.float64), f, a, 1.0)["placement_ok"]:
            return t
    return None


def w1_rule_rows(s, s_frozen):
    """Every step the causal rule point can be, for any cutoff: 1 and every i + 1 with s_i < 0.5·s*_frozen ≤ s_{i+1}
    (i ≥ 1)."""
    lvl = C.RULE_FRAC * s_frozen
    s = np.asarray(s, float)
    up = np.nonzero((s[1:-1] < lvl) & (s[2:] >= lvl))[0] + 2
    return sorted({1} | {int(v) for v in up})


# ------------------------------------------------------------------------------------------ runs (train + observe)
def _save_path(setting, seed, out, hid_rows, hid):
    PATHS.mkdir(parents=True, exist_ok=True)
    p = PATHS / f"{setting}_{seed}.npz"
    np.savez_compressed(p, out=out, hid_rows=np.asarray(hid_rows, np.int64), hid=hid)
    return p.name, _sha(p)


def _t_end(s, t_obs, s_ref, n):
    t_hi = C.cutoff_step(s, 1.1 * s_ref) if s_ref else None
    cands = [v for v in (t_obs, t_hi) if v is not None]
    return int(min(n - 1, (max(cands) if cands else n - 1) + SAVE_MARGIN))


def run_one(setting, seed):
    t0 = time.time()
    if setting == "w1_sgd":
        fr = w1_frozen(seed)
        row = {"setting": setting, "seed": seed, "frozen": fr}
        if fr["s_frozen"] is None:
            return {**row, "status": "no frozen switch"}
        W, _, _ = w1_train(seed)
        s = np.abs(W[:, 2])
        t_obs = w1_observe(W)
        rr = w1_rule_rows(s, fr["s_frozen"])
        t_end = _t_end(s, t_obs, fr["s_frozen"], len(W))
        name, h = _save_path(setting, seed, W[:t_end + 1, 2], rr, W[rr][:, [0, 1, 3]] if rr else np.zeros((0, 3)))
        row.update(status="ok", t_obs=t_obs, s_obs=float(s[t_obs]) if t_obs else None, budget=W1_BUDGET,
                   n_saved=t_end + 1, path=name, path_sha256=h, rule_rows=rr, s_ref=fr["s_frozen"])
    elif setting == "gelu_random":
        import torch
        from . import gelu_transfer as G
        from .phase2b_ordering import state
        land = G._land()
        fr = G.freeze_one(seed)
        th_rel, rec, (X, Y, u, P) = G.run_start("random", seed, fr, land)
        row = {"setting": setting, "seed": seed, "frozen": fr, "release": {k: rec[k] for k in (
            "copy_at_release", "on_branch", "hold_n_G_pos", "hold_G_positive", "W_hold", "newton_ok")}}
        Wp = G.release_train(th_rel, X, Y, u, 1.0, G.BUDGET)
        t_obs = None
        for t in range(1, len(Wp)):
            if state(torch.tensor(Wp[t], dtype=torch.float64), u, None, 1.0)["placement_ok"]:
                t_obs = t
                break
        s = np.abs(Wp[:, 2])
        copy = rec["copy_at_release"]
        s_ref = fr["copies"][str(copy)]["s_switch"] if rec["on_branch"] else None
        fol = G.follow_check(Wp, s, copy, fr["copies"][str(copy)], P) if rec["on_branch"] else {}
        t_end = _t_end(s, t_obs, s_ref, len(Wp))
        name, h = _save_path(setting, seed, Wp[:t_end + 1, 2], [0], Wp[[0]][:, [0, 1, 3]])
        row.update(status="ok", t_obs=t_obs, s_obs=float(s[t_obs]) if t_obs else None, budget=G.BUDGET,
                   n_saved=t_end + 1, path=name, path_sha256=h, s_ref=s_ref,
                   follow_PILOT_ONLY={k: fol.get(k) for k in ("t_follow", "copy_at_follow", "follows_branch")})
    else:
        from . import width2_asym as W
        arm = W.T if setting == "w2a_T" else W.TP
        land = W._land()
        x, y = W.own_sample(seed)
        full, point = (("T",), ("Tp",)) if arm == W.T else (("Tp",), ("T",))
        copies = {c: W.freeze_copy(c, land, x, y, True) for c in full}
        copies.update({c: W.freeze_copy(c, land, x, y, False) for c in point})
        Wc = {c: r["W"] for c, r in copies.items() if r.get("point_ok")}
        fr = {"seed": seed, "copies": copies, "W_arm": {arm: W.arm_hold_steps(arm, Wc)}}
        rho = json.loads((W.OUT / "pilot.json").read_text())["arms"][arm]["rho"]
        z_rel, v0, rec, _ = W.run_start(arm, seed, fr, land)
        row = {"setting": setting, "seed": seed, "frozen": W._jsonable_tree(fr), "rho": rho,
               "release": {k: rec[k] for k in ("copy_at_release", "windings", "on_branch", "hold_n_G_pos",
                                                "hold_G_positive", "W_hold", "newton_ok", "newton_type")}}
        n = W.budget(arm, rho)
        Pp = W.train_path(W.qof(z_rel, v0), W.ETA[arm], rho, n, x, y)
        t_obs, und = W.observe_path(Pp)
        V = Pp[:, W.VI]
        s = np.abs(V).sum(axis=1)
        copy = rec["copy_at_release"]
        ok = rec["on_branch"] and copies[copy].get("valid")
        s_ref = copies[copy]["s_switch"] if ok else None
        fol = W.follow_check(Pp, s, copies[copy], copy, tuple(rec["windings"]), x, y) if ok else {}
        t_end = _t_end(s, t_obs, s_ref, len(Pp))
        name, h = _save_path(setting, seed, V[:t_end + 1], [0], Pp[[0]][:, W.ZI])
        row.update(status="ok", t_obs=t_obs, s_obs=float(s[t_obs]) if t_obs else None, n_undecided=und, budget=n,
                   n_saved=t_end + 1, path=name, path_sha256=h, s_ref=s_ref,
                   follow_PILOT_ONLY={k: fol.get(k) for k in ("t_follow", "copy_at_follow", "follows_branch")})
        del Pp
    row["secs_run"] = round(time.time() - t0, 1)
    return row


def run(setting, split, n=None):
    _setup()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"runs_{setting}.jsonl"
    done = {(r["seed"]) for r in _rows(f)}
    seeds = SEEDS[setting][split][: (int(n) if n else None)]
    for seed in seeds:
        if seed in done:
            continue
        memory_gate(f"run {setting} {split} {seed}")
        r = run_one(setting, seed)
        r["split"] = split
        _append_write(f, r)
        print(json.dumps(_jsonable({k: r.get(k) for k in ("setting", "seed", "split", "status", "t_obs", "s_ref",
                                                          "secs_run")})), flush=True)
        rss_guard()


# ------------------------------------------------------------------------------------------ forecasts and errors
def _load(row):
    d = np.load(PATHS / row["path"])
    assert _sha(PATHS / row["path"]) == row["path_sha256"], f"path hash mismatch {row['path']}"
    return d["out"], d["hid_rows"], d["hid"]


def _hid_full(n, rows, hid, k):
    H = np.full((n, k), np.nan)
    H[np.asarray(rows, int)] = hid
    return H


def _branch_gelu(row):
    from . import gelu_transfer as G
    land = G._land()
    cf = row["frozen"]["copies"][str(row["release"]["copy_at_release"])]
    P = G.own_problem(row["seed"])
    B = G.GBranch(land["s0"], np.array(cf["z_s0"]), P, 0.95 * land["s0"], G.S_HI_FRAC * land["s_pop"],
                  G.GRID_H_FRAC * land["s_pop"])
    return B, cf, P


def _base(setting):
    return RUNS_OF.get(setting, setting)


def make_inputs(setting, row, out, hid_rows, hid, t_c, ctx, guarded=True):
    """The forecaster's inputs: every run array behind a guard (cutoff t_c; hidden state: one allowed row)."""
    setting = _base(setting)
    n = len(out)
    gd = (lambda a, **kw: C.guard(a, t_c, **kw)) if guarded else (lambda a, **kw: a)
    if setting == "w1_sgd":
        Hf = _hid_full(n, hid_rows, hid, 3)
        return C.W1Inputs(a=A_W1, x=ctx["x"], y=ctx["y"], s_frozen=row["s_ref"], opt="sgd", lr=W1_LR, t_c=t_c,
                          budget=row["budget"], out=gd(out, name="w2"), hid=gd(Hf, max_rows=1, name="hidden"))
    Hf = _hid_full(n, hid_rows, hid, hid.shape[1])
    if setting == "gelu_random":
        return C.GeluInputs(t_c=t_c, budget=row["budget"], out=gd(out, name="w2"), hid=gd(Hf, rows={0}, name="hidden"),
                            copy_row=ctx["cf"], branch=ctx["B"], act=ctx["act"])
    from . import width2_asym as W
    arm = W.T if setting == "w2a_T" else W.TP
    copy = row["release"]["copy_at_release"]
    return C.W2AInputs(arm=arm, t_c=t_c, budget=row["budget"], out=gd(out, name="v"), hid=gd(Hf, rows={0}, name="hidden"),
                       copy_row=row["frozen"]["copies"][copy], windings=tuple(row["release"]["windings"]),
                       x=ctx["x"], y=ctx["y"], copy=copy)


def context(setting, row):
    setting = _base(setting)
    if setting == "w1_sgd":
        x, y = _w1_sample(row["seed"])
        return {"x": x, "y": y}
    if setting == "gelu_random":
        B, cf, P = _branch_gelu(row)
        return {"B": B, "cf": cf, "act": P.act}
    from . import width2_asym as W
    x, y = W.own_sample(row["seed"])
    return {"x": x, "y": y}


def actual_and_reference(setting, row, out, hid_rows, hid, ctx, fc_any):
    """PILOT ONLY, NOT CAUSAL (reads the whole saved path): the actual switch and lag, and the path-conditioned
    reference (the same lag integration along the ACTUAL output path, no extrapolation; POST HOC)."""
    setting = _base(setting)
    t_obs = row["t_obs"]
    res = {"t_obs": t_obs, "s_obs": row["s_obs"]}
    if setting == "w1_sgd":
        s = np.abs(out)
        s_run = fc_any.get("s_run")
        if s_run is None:
            return res
        res["s_sw_act"] = s_run
        res["t_sw_act"] = LR._first_ge(s, s_run)
        tR = fc_any["t_rule"]  # present whenever s_run is
        TH = hid[list(hid_rows).index(tR)] * (LR.D_FLIP if out[tR] < 0 else np.ones(3))
        B, _ = C.w1_branch(A_W1, ctx["x"], ctx["y"], row["s_ref"], float(s[tR]), TH)
        r = C.lag_forecast_1d(s, tR, B.theta, B.hess, B.contains, lambda th: LR.gap_exact(th[0], th[1], A_W1),
                              TH - B.theta(float(s[tR])), W1_LR, s_run)
        res["t_ref"] = r["t_hit"]
        return res
    if setting == "gelu_random":
        s = np.abs(out)
        s_sw = float(ctx["cf"]["s_switch"])
        res.update(s_sw_act=s_sw, t_sw_act=LR._first_ge(s, s_sw))
        from .act_general import gap_mid
        B = ctx["B"]
        r = C.lag_forecast_1d(s, 0, B.theta, B.hess, B.contains, lambda th: gap_mid(th[0], th[1], ctx["act"]),
                              hid[0] - B.theta(float(s[0])), 0.03, s_sw)
        res["t_ref"] = r["t_hit"]
        return res
    from . import width2_asym as W
    arm = W.T if setting == "w2a_T" else W.TP
    copy = row["release"]["copy_at_release"]
    cf, k = row["frozen"]["copies"][copy], tuple(row["release"]["windings"])
    br = W.OwnBranch(out, W.shift(np.array(cf["z_s0"]), out[0], k), ctx["x"], ctx["y"], False)
    t_sw, s_sw = W.own_path_switch(br, len(out) - 1)
    res.update(s_sw_act=s_sw, t_sw_act=t_sw)
    t_hit, st, _, _ = W.r4_recursion(br, hid[0], W.ETA[arm], len(out) - 1)
    res["t_ref"] = t_hit
    return res


def _err_row(fc, act):
    """Forecast vs actual (signed, steps; lag ratios)."""
    e = {}
    t_fc, t_obs = fc.get("t_fc"), act.get("t_obs")
    t_sw_fc, t_sw_act = fc.get("t_sw"), act.get("t_sw_act")
    if t_fc is not None and t_obs is not None:
        e["err_cross"] = t_fc - t_obs
    if t_sw_fc is not None and t_sw_act is not None:
        e["err_sw"] = t_sw_fc - t_sw_act
    if t_sw_fc is not None and t_obs is not None:
        e["err_lagfree_baseline"] = t_sw_fc - t_obs
    if t_obs is not None and t_sw_act is not None:
        e["lag_act"] = t_obs - t_sw_act
        if act.get("s_obs") and act.get("s_sw_act"):
            e["r_obs"] = act["s_obs"] / act["s_sw_act"] - 1
    if fc.get("lag_steps_fc") is not None and "lag_act" in e:
        e["err_lag"] = fc["lag_steps_fc"] - e["lag_act"]
    if np.isfinite(fc.get("r_fc", np.nan)) and "r_obs" in e and fc["r_fc"] != 0:
        e["ratio_r_obs_over_r_fc"] = e["r_obs"] / fc["r_fc"]
    if fc.get("s_switch") is not None and act.get("s_sw_act"):
        e["rel_err_s_switch"] = fc["s_switch"] / act["s_sw_act"] - 1
    if act.get("t_ref") is not None and t_obs is not None:
        e["err_ref"] = act["t_ref"] - t_obs
    if t_obs is not None:
        e["cutoff_before_crossing"] = bool(fc["t_c"] < t_obs)
        e["steps_cutoff_to_crossing"] = t_obs - fc["t_c"]
    return e


def eligible(setting, row):
    setting = _base(setting)
    if row.get("status") != "ok" or row.get("s_ref") is None:
        return False
    if setting == "w1_sgd":
        return True
    return bool(row["release"]["on_branch"])


def forecast(setting, split, families=FAMILY_GRID, windows=WINDOW_GRID, f_grid=F_GRID):
    _setup()
    fo = OUT / f"forecasts_{setting}.jsonl"
    done = {(r["seed"], r["family"], r["window_frac"], r["f"]) for r in _rows(fo)}
    for row in _rows(OUT / f"runs_{_base(setting)}.jsonl"):
        if row["split"] != split or not eligible(setting, row):
            continue
        todo = [(fam, w, f_) for fam in families for w in windows for f_ in f_grid
                if (row["seed"], fam, w, f_) not in done]
        if not todo:
            continue
        memory_gate(f"forecast {setting} {row['seed']}")
        out, hid_rows, hid = _load(row)
        s = np.abs(out) if out.ndim == 1 else np.abs(out).sum(axis=1)
        ctx = context(setting, row)
        act, cut_cache = None, {}
        for fam, w, f_ in todo:
            t0 = time.time()
            cfg = C.ForecastConfig(f=f_, family=fam, window_frac=w)
            rec = {"setting": setting, "seed": row["seed"], "split": split, "family": fam, "window_frac": w, "f": f_}
            if setting == "w1_sgd_run":                                    # nested stopping time on s*_run
                hr = {int(r_): hid[i] for i, r_ in enumerate(hid_rows)}
                if f_ not in cut_cache:                                     # depends on f only
                    cut_cache[f_] = C.w1_cutoff_on_s_run(s, lambda k: out[k], f_, row["s_ref"], A_W1, ctx["x"],
                                                         ctx["y"], lambda k: hr[int(k)])
                t_c, tR_c, srun_c = cut_cache[f_]
                rec.update(cutoff_rule_point=tR_c, cutoff_s_run=srun_c)
            else:
                t_c = C.cutoff_step(s, f_ * row["s_ref"])
            if t_c is None:
                rec.update(status="no cutoff: s never reaches f·s_ref in the saved path")
            else:
                fc = C.run_forecast({"w1_sgd": "w1", "gelu_random": "gelu"}.get(_base(setting), "w2a"),
                                    make_inputs(setting, row, out, hid_rows, hid, t_c, ctx), cfg)
                if act is None or _base(setting) == "w1_sgd":
                    act = actual_and_reference(setting, row, out, hid_rows, hid, ctx, fc)
                rec.update({k: v for k, v in fc.items() if k not in ("max_index_read",)})
                rec["max_index_read"] = fc["max_index_read"]
                rec["actual_PILOT_ONLY"] = act
                rec.update(_err_row(fc, act))
            rec["secs"] = round(time.time() - t0, 2)
            _append_write(fo, rec)
            print(json.dumps(_jsonable({k: rec.get(k) for k in ("seed", "family", "window_frac", "f", "status", "t_c",
                                                                "t_fc", "err_cross", "err_lag", "err_ref", "secs")})),
                  flush=True)
        rss_guard()



# ------------------------------------------------------------------------------------------ summary and report
# Decision rules.
#  PRIMARY (family, window), on the DEV seeds only.  FIRST RULE (written before the dev forecasts were inspected): the
#    configuration with the fewest missing dev forecasts (over every setting and f), ties broken by the pooled median
#    (over settings and f) of the per-setting median of |t_fc − t_obs| / max(|lag_act|, 1).  It selected lin/0.25 on
#    the first dev grid (lin/0.05 on the final grid), among the least accurate configurations, because one long-stall
#    dev run (seed 9,310,000) has missing forecasts in most other configurations.  REVISED RULE (after inspecting the dev results; the est seeds were not yet run): the same pooled
#    median, with a missing forecast counted as +∞ in its per-setting median (the worst outcome rather than a
#    lexicographic first key).  The window 0.05 was added to the dev grid after the first dev pass (families quad, rate,
#    lin).  Both rules and their picks are reported.
#  PROPOSED f (est seeds).  FIRST RULE (written before the dev pass): the largest f with the cutoff before the crossing in
#    every crossed run and at least 2·max|lag| steps from the cutoff to the crossing.  It admits only f = 0.8 on the dev
#    runs (width-1 SGD: lag ~78 steps on a fast path), where the W2-A forecasts are off by 10²–10³ steps.  REVISED RULE
#    (after the dev pass, before any est forecast; disclosed): the largest f such that, in every setting and every crossed
#    est run with a forecast, the cutoff precedes BOTH the crossing and the actual lag-free switch by at least that run's
#    |actual lag| (at least one lag of genuine horizon, and the forecaster never sees the switch pass).  Both reported.
#  PROPOSED TOLERANCES (est seeds, primary configuration, proposed f), per setting:
#    step tolerances τ_cross = 1.5·q90 |t_fc − t_obs| and τ_lag = 1.5·q90 |lag_fc − lag_act| (signed lags in steps),
#                    each rounded up to a multiple of 5 steps (1C criterion: ≥ 80% of scored runs within τ; the pilot
#                    fraction within τ is reported);
#    ratio band      b = max(0.10, |m − 1| + 2·SE), rounded up to 0.05, m = the pilot median of r_obs/r_fc and SE =
#                    1.2533·(IQR/1.349)/√n (1C criterion: the median of r_obs/r_fc in [1 − b, 1 + b]);
#    baseline        the forecast beats the lag-free forecast (t_sw,fc) in at least half the runs (median paired
#                    difference |t_sw,fc − t_obs| − |t_fc − t_obs| > 0).
DEV_RULE = ("pooled median over settings and f of the per-setting median |t_fc - t_obs| / max(|lag_act|, 1), a missing "
            "forecast counted as +inf (revised rule; the first rule, fewest missing forecasts first, is also reported)")


def _q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.percentile(x, p)) if len(x) else None


def _stats(rows, f_):
    rs = [r for r in rows if r["f"] == f_]
    ok = [r for r in rs if r.get("status") == "ok" and r.get("err_cross") is not None]
    crossed = [r for r in rs if (r.get("actual_PILOT_ONLY") or {}).get("t_obs") is not None]
    cb = [r for r in rs if r.get("cutoff_before_crossing") is not None]
    a = lambda k: [abs(r[k]) for r in ok if r.get(k) is not None]             # noqa: E731
    s = lambda k: [r[k] for r in ok if r.get(k) is not None]                  # noqa: E731
    rel = [abs(r["err_cross"]) / max(abs(r["lag_act"]), 1) for r in ok if r.get("lag_act") is not None]
    lagrel = [abs(r["err_lag"]) / max(abs(r["lag_act"]), 1) for r in ok if r.get("err_lag") is not None]
    ratio = s("ratio_r_obs_over_r_fc")
    beats = [abs(r["err_lagfree_baseline"]) - abs(r["err_cross"]) for r in ok if r.get("err_lagfree_baseline") is not None]
    margin = [r["steps_cutoff_to_crossing"] for r in cb]
    hz = []
    for r in ok:
        a_ = r.get("actual_PILOT_ONLY") or {}
        if a_.get("t_obs") is not None and a_.get("t_sw_act") is not None and r.get("lag_act") is not None:
            hz.append((min(a_["t_obs"], a_["t_sw_act"]) - r["t_c"]) / max(abs(r["lag_act"]), 1))
    out = {"n_runs": len(rs), "n_crossed": len(crossed), "n_forecast": len(ok),
           "n_missing": len(rs) - len([r for r in rs if r.get("status") == "ok"]),
           "statuses": {k: sum(1 for r in rs if r.get("status") == k) for k in sorted({r.get("status") for r in rs})},
           "n_cutoff_before_crossing": sum(1 for r in cb if r["cutoff_before_crossing"]), "n_with_cutoff_and_crossing": len(cb),
           "min_steps_cutoff_to_crossing": min(margin) if margin else None,
           "n_horizon_at_least_one_lag": sum(1 for v in hz if v >= 1), "n_horizon_checked": len(hz),
           "n_cutoff_before_both_events": sum(1 for v in hz if v > 0),
           "min_horizon_over_abs_lag": min(hz) if hz else None,
           "median_steps_cutoff_to_crossing": _q(margin, 50),
           "abs_err_cross_median": _q(a("err_cross"), 50), "abs_err_cross_q90": _q(a("err_cross"), 90),
           "err_cross_signed_median": _q(s("err_cross"), 50),
           "abs_err_cross_over_abs_lag_median": _q(rel, 50), "abs_err_cross_over_abs_lag_q90": _q(rel, 90),
           "abs_err_lag_median": _q(a("err_lag"), 50), "abs_err_lag_q90": _q(a("err_lag"), 90),
           "err_lag_signed_median": _q(s("err_lag"), 50),
           "abs_err_lag_over_abs_lag_median": _q(lagrel, 50), "abs_err_lag_over_abs_lag_q90": _q(lagrel, 90),
           "abs_err_sw_median": _q(a("err_sw"), 50), "abs_err_sw_q90": _q(a("err_sw"), 90),
           "ratio_r_obs_over_r_fc": {"n": len(ratio), "q10": _q(ratio, 10), "q25": _q(ratio, 25), "median": _q(ratio, 50),
                                     "q75": _q(ratio, 75), "q90": _q(ratio, 90)},
           "abs_err_lagfree_baseline_median": _q(a("err_lagfree_baseline"), 50),
           "abs_err_lagfree_baseline_q90": _q(a("err_lagfree_baseline"), 90),
           "n_beats_lagfree_baseline": sum(1 for v in beats if v > 0), "n_ties_lagfree_baseline": sum(1 for v in beats if v == 0),
           "median_paired_baseline_minus_forecast": _q(beats, 50),
           "abs_err_ref_POSTHOC_median": _q(a("err_ref"), 50), "abs_err_ref_POSTHOC_q90": _q(a("err_ref"), 90),
           "abs_rel_err_s_switch_median": _q(a("rel_err_s_switch"), 50),
           "abs_rel_err_s_switch_q90": _q(a("rel_err_s_switch"), 90),
           "actual_lag_steps": {"min": _q([r["lag_act"] for r in ok if r.get("lag_act") is not None], 0),
                                "median": _q([r["lag_act"] for r in ok if r.get("lag_act") is not None], 50),
                                "max": _q([r["lag_act"] for r in ok if r.get("lag_act") is not None], 100)},
           "max_index_read_below_cutoff": all(max(r.get("max_index_read", {}).values() or [-1]) < r["t_c"]
                                              for r in rs if r.get("t_c") is not None)}
    return out


def _run_stats(setting, split):
    rows = [r for r in _rows(OUT / f"runs_{_base(setting)}.jsonl") if r["split"] == split]
    el = [r for r in rows if eligible(setting, r)]
    ratio = [r["s_obs"] / r["s_ref"] for r in el if r.get("s_obs") and r.get("s_ref")]
    lag = []
    return {"n_seeds": len(rows), "n_eligible": len(el), "n_eligible_crossed": sum(1 for r in el if r.get("t_obs")),
            "min_s_obs_over_s_ref": min(ratio) if ratio else None, "max_s_obs_over_s_ref": max(ratio) if ratio else None,
            "n_follow_check_fails_PILOT_ONLY": sum(1 for r in el if (r.get("follow_PILOT_ONLY") or {}).get("follows_branch")
                                                   is False),
            "secs_run_total": round(sum(r.get("secs_run", 0) for r in rows), 1), "lag": lag}


def build_summary():
    out = {"label": "PHASE 1A PILOT (not a registration; fresh pilot seeds only)", "a_w1": A_W1, "f_grid": list(F_GRID),
           "families": list(FAMILY_GRID), "windows": list(WINDOW_GRID), "seeds": {k: {sp: [v[0], v[-1], len(v)]
                                                                                       for sp, v in d.items()}
                                                                                   for k, d in SEEDS.items()},
           "dev_rule": DEV_RULE, "landscape_s_star": round(_land_w1()["s_star_pop"], 6), "settings": {}}
    fc = {st: _rows(OUT / f"forecasts_{st}.jsonl") for st in SETTINGS}
    configs = [(fam, w) for fam in FAMILY_GRID for w in WINDOW_GRID]
    # primary configuration on the dev seeds
    dev = {}
    for fam, w in configs:
        miss, meds, meds_inf = 0, [], []
        for st in SELECT:
            rows = [r for r in fc[st] if r["split"] == "dev" and r["family"] == fam and r["window_frac"] == w]
            for f_ in F_GRID:
                rs = [r for r in rows if r["f"] == f_]
                if not rs:
                    continue
                S = _stats(rows, f_)
                miss += S["n_missing"]
                if S["abs_err_cross_over_abs_lag_median"] is not None:
                    meds.append(S["abs_err_cross_over_abs_lag_median"])
                v = [abs(r["err_cross"]) / max(abs(r["lag_act"]), 1)
                     if r.get("err_cross") is not None and r.get("lag_act") is not None else math.inf for r in rs]
                meds_inf.append(float(np.median(v)))
        if not meds_inf:
            continue
        dev[f"{fam}/{w}"] = {"n_missing": miss, "pooled_median_rel_err": float(np.median(meds)) if meds else None,
                             "pooled_median_rel_err_missing_inf": float(np.median(meds_inf)), "n_cells": len(meds_inf)}
    full = [k for k in dev if dev[k]["n_cells"] == len(SELECT) * len(F_GRID)]
    first = min(full, key=lambda k: (dev[k]["n_missing"], dev[k]["pooled_median_rel_err"]
                                     if dev[k]["pooled_median_rel_err"] is not None else float("inf")))
    primary = min(full, key=lambda k: dev[k]["pooled_median_rel_err_missing_inf"])
    out["dev_selection"] = dev
    out["primary_first_rule"] = first
    out["primary"] = primary
    pfam, pw = primary.split("/")[0], float(primary.split("/")[1])
    secs = 0.0
    for st in SETTINGS:
        S = {"runs": {sp: _run_stats(st, sp) for sp in ("dev", "est")}, "est": {}, "est_sensitivity": {}}
        if st not in RUNS_OF:                                              # each run's training counted once
            secs += sum(S["runs"][sp]["secs_run_total"] for sp in ("dev", "est"))
        secs += sum(r.get("secs", 0) for r in fc[st])
        est = [r for r in fc[st] if r["split"] == "est"]
        for fam, w in configs:
            rows = [r for r in est if r["family"] == fam and r["window_frac"] == w]
            if not rows:
                continue
            tab = {str(f_): _stats(rows, f_) for f_ in F_GRID}
            if (fam, w) == (pfam, pw):
                S["est"] = tab
            S["est_sensitivity"][f"{fam}/{w}"] = {k: {"n_forecast": v["n_forecast"], "n_missing": v["n_missing"],
                                                      "abs_err_cross_median": v["abs_err_cross_median"],
                                                      "abs_err_cross_q90": v["abs_err_cross_q90"],
                                                      "abs_err_lag_q90": v["abs_err_lag_q90"]}
                                                  for k, v in tab.items()}
        out["settings"][st] = S
    out["compute_secs_total"] = round(secs, 1)
    adj = secs
    for st in SETTINGS:                     # rows > 60 s (the overnight idle period) replaced by the setting's median row
        v = [r.get("secs", 0) for r in fc[st]]
        if v:
            med = float(np.median(v))
            adj -= sum(x - med for x in v if x > 60)
    out["compute_secs_rows_over_60s_at_median"] = round(adj, 1)
    out["n_rows_over_60s"] = sum(1 for st in SETTINGS for r in fc[st] if r.get("secs", 0) > 60)
    out["proposal"] = proposal(out)
    return out


def _ceil_to(x, m):
    return round(float(math.ceil(x / m - 1e-12) * m), 10)


def _f_admissible(summ, first_rule, settings=SELECT, minimal=False):
    ok_f = []
    for f_ in F_GRID:
        good = True
        for st in settings:
            T = summ["settings"][st]["est"].get(str(f_))
            if not T or T["n_with_cutoff_and_crossing"] == 0:
                good = False
                continue
            if first_rule:
                lag = T["actual_lag_steps"]
                mx = max(abs(lag["min"] or 0), abs(lag["max"] or 0))
                if T["n_cutoff_before_crossing"] < T["n_with_cutoff_and_crossing"] or \
                        (T["min_steps_cutoff_to_crossing"] or 0) < 2 * mx:
                    good = False
            elif T["n_cutoff_before_crossing"] < T["n_with_cutoff_and_crossing"] or T["n_horizon_checked"] == 0 or \
                    (T["n_cutoff_before_both_events"] if minimal else T["n_horizon_at_least_one_lag"]) \
                    < T["n_horizon_checked"]:
                good = False
        if good:
            ok_f.append(f_)
    return ok_f


def _tolerances(summ, st, f_):
    T = summ["settings"][st]["est"][str(f_)]
    rr = T["ratio_r_obs_over_r_fc"]
    fc_rows = [r for r in _rows(OUT / f"forecasts_{st}.jsonl") if r["split"] == "est" and r["f"] == f_
               and f"{r['family']}/{r['window_frac']}" == summ["primary"] and r.get("err_cross") is not None]
    tau = _ceil_to(1.5 * T["abs_err_cross_q90"], 5) if T["abs_err_cross_q90"] is not None else None
    tau_l = _ceil_to(max(1.5 * T["abs_err_lag_q90"], 1), 5) if T["abs_err_lag_q90"] is not None else None
    b = None
    if rr["n"] >= 2:
        se = 1.2533 * ((rr["q75"] - rr["q25"]) / 1.349) / math.sqrt(rr["n"])
        b = _ceil_to(max(0.10, abs(rr["median"] - 1) + 2 * se), 0.05)
    return {"f": f_, "step_tolerance_cross": tau,
            "pilot_n_within_cross": sum(1 for r in fc_rows if abs(r["err_cross"]) <= tau) if tau is not None else None,
            "step_tolerance_lag": tau_l,
            "pilot_n_within_lag": sum(1 for r in fc_rows if r.get("err_lag") is not None and abs(r["err_lag"]) <= tau_l)
            if tau_l is not None else None,
            "pilot_n": len(fc_rows), "ratio_band_halfwidth": b, "pilot_median_ratio": rr["median"],
            "pilot_n_beats_lagfree": T["n_beats_lagfree_baseline"], "pilot_n_forecast": T["n_forecast"]}


RECOMMENDED_F = 0.95   # a recommendation made AFTER the est results were seen (flagged in the report); the rules above
                       # are reported unchanged


def proposal(summ):
    ok_first = _f_admissible(summ, True)
    ok_f = _f_admissible(summ, False)
    ok_literal = _f_admissible(summ, False, ("w1_sgd",) + SELECT[1:])
    per = {st: _f_admissible(summ, False, (st,)) for st in SETTINGS}
    ok_min = _f_admissible(summ, False, minimal=True)
    ok_min_literal = _f_admissible(summ, False, ("w1_sgd",) + SELECT[1:], minimal=True)
    f_star = max(ok_f) if ok_f else None
    f_per = {st: (max(v) if v else None) for st, v in per.items()}
    return {"f_ok_first_rule": ok_first, "f_ok": ok_f, "f_ok_with_w1_cutoff_on_s_frozen": ok_literal,
            "f_ok_minimal": ok_min, "f_ok_minimal_with_w1_cutoff_on_s_frozen": ok_min_literal,
            "f_recommended": RECOMMENDED_F,
            "tolerances_recommended_f": {st: _tolerances(summ, st, RECOMMENDED_F) for st in SETTINGS},
            "f_proposed": f_star, "f_ok_per_setting": per, "f_proposed_per_setting": f_per,
            "window_frac": float(summ["primary"].split("/")[1]), "family": summ["primary"].split("/")[0],
            "tolerances": {st: _tolerances(summ, st, f_per[st]) for st in SETTINGS if f_per[st] is not None},
            "tolerances_common_f": ({st: _tolerances(summ, st, f_star) for st in SETTINGS}
                                    if f_star is not None else {})}


SETTING_LABEL = {"w1_sgd": "width 1, SGD lr 0.3, a = 1.58, cutoff on s*_frozen (spec-literal)",
                 "w1_sgd_run": "width 1, SGD lr 0.3, a = 1.58, cutoff on s*_run (nested stopping time)", "gelu_random": "GELU-T random start (ρ = 1, η = 0.03)",
                 "w2a_T": "W2-A T (ρ = 2⁻¹⁰, η = 0.03)", "w2a_Tp": "W2-A T′ (ρ = 2⁻¹², η = 0.03)"}


def _fmt(v, d=0):
    if v is None:
        return "—"
    if isinstance(v, float) and d == 0:
        return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)


def render_md(S):
    """results/phase1a/error_estimate.md, generated whole from summary.json (the ledger requires equality)."""
    P = S["proposal"]
    L = []
    L.append("# Phase 1A: causal forecaster, extrapolation-error estimate on pilot seeds")
    L.append("")
    L.append("**PHASE 1A PILOT, 2026-09-30.** Not a registration. Code, tests and an error estimate on FRESH PILOT SEEDS "
             "only. No registered seed was used, no registered run was read or re-scored (that is Phase 1B, after the "
             "forecaster is frozen). The crossings below are of pilot runs, measured with the existing exact detectors, "
             "for the error estimate only. Every number here is generated from `results/phase1a/summary.json` by "
             "`src/phase1a_pilot.py` (`render_md`); the ledger regenerates both.")
    L.append("")
    L.append("## 1. The forecaster (`src/causal_forecast.py`)")
    L.append("")
    L.append("- **Cutoff (a stopping time).** t_c = the first step t ≥ 1 with s_t ≥ f·s_ref, s_ref the run's FROZEN "
             "own-branch switch (width 1: s\\*_frozen; GELU-T and W2-A: the occupied copy's frozen s_switch). The harness "
             "locates t_c (it compares s_{t_c} with a frozen level and passes on only the integer); the forecaster reads "
             "rows 0 … t_c − 1 and nothing else of the run's path. **Width 1 has two variants:** `w1_sgd` (the spec "
             "literally: s_ref = s\\*_frozen) and `w1_sgd_run` (a nested stopping time: t_c = the first t with "
             "s_t ≥ f·s\\*_run(t), s\\*_run(t) the switch of the branch occupied at the causal rule point of s_0 … "
             "s_{t−1}; the harness reads the hidden state at that rule point only; `w1_cutoff_on_s_run`). The run's own "
             "branch at a = 1.58 often differs from the landscape branch (s\\*_run/s\\*_frozen 0.82–1.25 on the "
             "dev runs), so the spec-literal cutoff can land after the crossing.")
    L.append("- **(i) Extrapolation.** Window = the last ⌈w·(t_c − t_start)⌉ visible steps (at least 50; t_start = the "
             "release, or the width-1 rule point). y = log s. Families: `lin` "
             "(y linear in t), `quad` (quadratic in t), `rate` (one-step growth rate of y linear in y: an autonomous local "
             "model of the slaved output dynamics). Anchored at the last visible point. Width 2: s and the share "
             "u = |v₁|/s (lin/quad: the same family in t; rate: u linear in log s); signs of v_{t_c−1}. Extrapolated until "
             "ŝ ≥ 1.6·s_ref or the run's budget.")
    L.append("- **(ii) Integration.** Track A's R4 recursion (`linear_response.simulate`; W2-A `width2_asym.r4_recursion`, "
             "unchanged) along [visible path, extrapolated path], from the release state (GELU-T, W2-A) or the causal "
             "rule point (width 1: the first step ≥ 1 with s ≥ 0.5·s\\*_frozen at every step up to t_c − 1), with the "
             "frozen branch θ\\*(s), H(s) of the occupied copy on the own sample. Lag-free switch: W2-A T and T′, "
             "`width2_asym.own_path_switch` along the EXTRAPOLATED v path; otherwise the first extrapolated step with "
             "ŝ ≥ the frozen switch (width 1: s\\*_run of the rule-point branch).")
    L.append("- **(iii) Output.** t_fc, s_fc, t_sw,fc, s_sw,fc, signed lag in steps (t_fc − t_sw,fc) and scale "
             "(r_fc = s_fc/s_sw,fc − 1), and the last row each guard saw.")
    L.append("- **Width-1 Adam (implemented and tested, not piloted).** P = 1/(√v̂ + ε) frozen at the rule point (Track A) "
             "or at t_c − 1 (`adam_P = \"cutoff\"`, the causal analogue of 2A's P at t_sw); m₀ at the rule point; "
             "moments guarded with the cutoff.")
    L.append("")
    L.append("## 2. Enforcement (`tests/test_causal_forecast.py`)")
    L.append("")
    L.append("- The forecasters receive `GuardedArray` views, never arrays. A read of a row ≥ t_c (int, slice bound, "
             "index list, mask) raises `CausalityViolation`; the hidden-state view admits only the release row (GELU-T, "
             "W2-A) or ONE row before t_c (width 1: the rule point); Adam moments are cut at t_c. `run_forecast` "
             "re-asserts every guard's last row < t_c.")
    L.append("- (a) leaky forecasters (reading s_{t_c}; a second hidden row; a hidden state after release) are caught; a "
             "guard BYPASS (reading the private array) is caught by poisoning; (b) the real forecasters pass "
             "(width-1 SGD and Adam on fresh runs; GELU-T and W2-A T on committed fixtures); (c) NaN and garbage in every "
             "row ≥ t_c leave the output identical (guarded and unguarded); (d) Adam moments poisoned at rows ≥ t_c, "
             "or NaN except the rows the rule names, and the hidden state NaN except the allowed row: identical.")
    L.append("- Constructed cases: exact exponential growth is extrapolated exactly by every family and gives the "
             "analytic crossing and the known lag 1/(ηh) = 40 steps (PASS); a growth rate that changes after the cutoff "
             "is an extrapolation error the comparison catches (FAIL); a non-growing path or a switch beyond the budget "
             "gives no forecast (UNRESOLVED).")
    L.append("")
    L.append("## 3. Settings, seeds and the candidate a")
    L.append("")
    L.append("| setting | dev seeds | est seeds | est: eligible (on-branch) / crossed | s_obs/s_ref min–max (est) |")
    L.append("|---|---|---|---|---|")
    for st in SETTINGS:
        sd, R = S["seeds"][_base(st)], S["settings"][st]["runs"]
        L.append(f"| {SETTING_LABEL[st]} | {sd['dev'][0]:,}–{sd['dev'][1]:,} ({sd['dev'][2]}) | "
                 f"{sd['est'][0]:,}–{sd['est'][1]:,} ({sd['est'][2]}) | {R['est']['n_eligible']} / "
                 f"{R['est']['n_eligible_crossed']} | {_fmt(R['est']['min_s_obs_over_s_ref'], 4)}–"
                 f"{_fmt(R['est']['max_s_obs_over_s_ref'], 4)} |")
    L.append("")
    L.append("- **Seeds.** Fresh: no 7-digit seed with these prefixes occurs in src/, tests/, results/ or paper/ "
             "(`seed_scan.json`), and none is a registered or pilot seed of any test module. Why 20 (T: 60): q90 needs "
             "≥ 10 points to be more than the maximum; with ~20 per setting the q90 is the 18th–19th order statistic and "
             "the median's 95% order-statistic interval spans ranks 6–15. T has about half its runs on T at release "
             "(98/200 registered), hence 40 seeds, extended to 60 when only 15 of the first 40 were on T (the 20 added seeds "
             "were run after the first 40 were forecast; nothing else changed). Width 1: the first est range "
             "(9,310,100–119) was inspected, the width-1 cutoff variant and the window start were revised, and that "
             "range was RELABELLED dev; the width-1 est range is 9,310,200–239 (40 seeds; about 15% of width-1 runs do "
             "not cross within 32,000 steps). Dev seeds (5; T: 10) chose the extrapolation family and window and "
             "are not in the error estimate.")
    L.append("- **Candidate a = 1.58** (width-1 SGD). `a_candidate_scan.json`: no a-column of any results CSV equals "
             "1.58; the only value in [1.575, 1.585] is 1.575 in `corner_tracking.csv` (a landscape table, no training; "
             "the same file Track A cited for 1.65). The token 1.58 occurs only as a scale value in "
             "`sb_fold/branch_losses.csv` (an s-grid at a = 1.50) and in this phase's own ledger check. Landscape "
             "(`landscape_w1.json`, Track A's construction): s\\*_pop = "
             f"{S.get('landscape_s_star', '—')}, validated by the conditional search (unplaced at 0.995·s\\*, placed at "
             "1.005·s\\*). The pilot crossings at 1.58 exist only on the pilot seeds (disclosed).")
    L.append("- W2-A uses its registered pipeline (hold, ρ from its pilot.json, budgets, copies); GELU-T its registered "
             "pipeline (ρ = 1). The follow check (a hidden-state read at t₀.₈) is NOT a forecaster input; it is "
             "recorded for the pilot runs only.")
    L.append("")
    L.append("## 4. Choice of the extrapolation family and window (dev seeds only)")
    L.append("")
    L.append("- **First rule** (written before the dev forecasts were inspected): fewest missing dev forecasts, ties by "
             "the pooled median (over settings and f) of the per-setting median \\|t_fc − t_obs\\|/max(\\|lag\\|, 1). On "
             "the first dev pass it picked lin/0.25, among the least accurate configurations, because one long-stall "
             "width-1 dev run (seed 9,310,000: s fell to 0.21·s_ref, then rose to the switch in ~500 steps) had missing "
             "forecasts in most other configurations. With the width-1 window counted from the rule point (§1) those "
             f"forecasts exist, and on the final dev data the first rule picks `{S['primary_first_rule']}`.")
    L.append("- **Revised rule (DISCLOSED DEVIATION; after inspecting the first dev pass, before any est forecast of the "
             "final est ranges):** the same pooled median with a missing forecast counted as +∞ in its per-setting "
             f"median. The window 0.05 was added to the dev grid after the first dev pass. **Primary: `{S['primary']}`** "
             "(both rules agree on the final dev data). The est seeds played no part in either choice. \"Missing\" "
             "includes the dev runs that never reach the cutoff (never cross), the same in every configuration.")
    L.append("- **Horizon (disclosed change during the dev pass):** the extrapolation horizon was raised from 1.3·s_ref to "
             "1.6·s_ref after the first, aborted dev pass: at 1.3 the width-1 extrapolation ended before the predicted "
             "crossing in a dev run with s\\*_run/s\\*_frozen = 1.18 and r ≈ 0.08 (1.6·s\\*_frozen is within the "
             "branch grid, 1.7·s\\*_frozen). The R4 integration was made lazy (θ\\*, H evaluated only up to the hit); "
             "its output is identical (checked on a logged GELU-T forecast).")
    L.append("")
    L.append("| family / window | dev cells | missing dev forecasts | pooled median rel. error (missing ignored) | "
             "pooled median rel. error (missing = +∞) |")
    L.append("|---|---|---|---|---|")
    for k, v in S["dev_selection"].items():
        tag = (" **(primary)**" if k == S["primary"] else "") + (" (first rule)" if k == S["primary_first_rule"] else "")
        L.append(f"| {k}{tag} | {v['n_cells']} | {v['n_missing']} | {_fmt(v['pooled_median_rel_err'], 3)} | "
                 f"{_fmt(v['pooled_median_rel_err_missing_inf'], 3)} |")
    L.append("")
    L.append(f"## 5. Error estimate (est seeds; primary `{S['primary']}`)")
    L.append("")
    L.append("Errors are forecast − actual, over the runs with a forecast and a crossing. lag = t_obs − t_sw,act (steps). "
             "\"/|lag|\" = in units of the run's actual lag (denominator at least 1 step). ratio = r_obs/r_fc (signed "
             "scale lags). baseline = the lag-free forecast t_sw,fc. ref = the same integration along the ACTUAL path "
             "(path-conditioned, POST HOC; not causal; shows how much of the error is extrapolation).")
    for st in SETTINGS:
        L.append("")
        L.append(f"### {SETTING_LABEL[st]}")
        L.append("")
        L.append("| f | forecasts / runs | cutoff < crossing; < both events; ≥ 1 lag before both | min / median steps cutoff→crossing | \\|err\\| median / q90 "
                 "(steps) | \\|err\\|/\\|lag\\| median / q90 | lag err median / q90 (steps) | ratio q10 / median / q90 | "
                 "\\|switch err\\| median / q90 | baseline \\|err\\| median / q90; beats | ref \\|err\\| median / q90 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for f_ in S["f_grid"]:
            T = S["settings"][st]["est"].get(str(f_))
            if not T:
                continue
            rr = T["ratio_r_obs_over_r_fc"]
            L.append(f"| {f_} | {T['n_forecast']} / {T['n_runs']} | {T['n_cutoff_before_crossing']}/"
                     f"{T['n_with_cutoff_and_crossing']}; {T['n_cutoff_before_both_events']}/{T['n_horizon_checked']}; "
                     f"{T['n_horizon_at_least_one_lag']}/{T['n_horizon_checked']} | {_fmt(T['min_steps_cutoff_to_crossing'])} / "
                     f"{_fmt(T['median_steps_cutoff_to_crossing'])} | {_fmt(T['abs_err_cross_median'])} / "
                     f"{_fmt(T['abs_err_cross_q90'])} | {_fmt(T['abs_err_cross_over_abs_lag_median'], 2)} / "
                     f"{_fmt(T['abs_err_cross_over_abs_lag_q90'], 2)} | {_fmt(T['abs_err_lag_median'])} / "
                     f"{_fmt(T['abs_err_lag_q90'])} | {_fmt(rr['q10'], 3)} / {_fmt(rr['median'], 3)} / "
                     f"{_fmt(rr['q90'], 3)} | {_fmt(T['abs_err_sw_median'])} / {_fmt(T['abs_err_sw_q90'])} | "
                     f"{_fmt(T['abs_err_lagfree_baseline_median'])} / {_fmt(T['abs_err_lagfree_baseline_q90'])}; "
                     f"{T['n_beats_lagfree_baseline']}/{T['n_forecast']} | {_fmt(T['abs_err_ref_POSTHOC_median'])} / "
                     f"{_fmt(T['abs_err_ref_POSTHOC_q90'])} |")
        T0 = S["settings"][st]["est"].get(str(S["f_grid"][0])) or {}
        lag = T0.get("actual_lag_steps") or {}
        L.append("")
        L.append(f"Actual lag (steps, est runs with a forecast at f = {S['f_grid'][0]}): min {_fmt(lag.get('min'))}, "
                 f"median {_fmt(lag.get('median'))}, max {_fmt(lag.get('max'))}. Missing forecasts by status (f = "
                 f"{S['f_grid'][-1]}): " + "; ".join(f"{k}: {v}" for k, v in
                                                    (S["settings"][st]["est"].get(str(S["f_grid"][-1])) or {})
                                                    .get("statuses", {}).items()) + ".")
        if st.startswith("w2a"):
            L.append(f"Switch along the extrapolated vs the actual v path: \\|s_sw,fc/s_sw,act − 1\\| median / q90 = " +
                     ", ".join(f"f {f_}: {_fmt((S['settings'][st]['est'].get(str(f_)) or {}).get('abs_rel_err_s_switch_median'), 6)}"
                               f" / {_fmt((S['settings'][st]['est'].get(str(f_)) or {}).get('abs_rel_err_s_switch_q90'), 6)}"
                               for f_ in S["f_grid"]) + ".")
    L.append("")
    L.append("### Sensitivity: every family and window (est seeds; \\|err\\| median / q90 in steps, missing)")
    L.append("")
    L.append("| setting | config | " + " | ".join(f"f = {f_}" for f_ in S["f_grid"]) + " |")
    L.append("|---|---|" + "---|" * len(S["f_grid"]))
    for st in SETTINGS:
        for k, tab in S["settings"][st]["est_sensitivity"].items():
            L.append(f"| {st} | {k} | " + " | ".join(
                f"{_fmt(tab[str(f_)]['abs_err_cross_median'])} / {_fmt(tab[str(f_)]['abs_err_cross_q90'])} "
                f"({tab[str(f_)]['n_missing']})" for f_ in S["f_grid"]) + " |")
    L.append("")
    L.append("## 6. Proposal for the author (checkpoint 1)")
    L.append("")
    L.append("Rules (fixed before any est forecast; the f rule REVISED after the dev pass, disclosed): **f** = the largest f "
             "in the grid such that, in every setting and every crossed est run with a forecast, the cutoff precedes "
             "both the crossing and the actual lag-free switch by at least that run's \\|lag\\| (the first rule, "
             "≥ 2·max\\|lag\\| steps before the crossing, admitted only f = 0.8 on the dev runs, where the W2-A crossing "
             "forecasts are off by hundreds to thousands of steps). **Step tolerances** τ_cross = 1.5·q90 "
             "\\|t_fc − t_obs\\| and τ_lag = 1.5·q90 \\|lag_fc − lag_act\\|, rounded up to 5 steps (1C criterion: ≥ 80% "
             "of scored runs within τ). **Ratio band** b = max(0.10, \\|m − 1\\| + 2·SE) rounded up to 0.05 (m the pilot "
             "median of r_obs/r_fc, SE = 1.2533·(IQR/1.349)/√n; 1C criterion: median r_obs/r_fc in [1 − b, 1 + b]). "
             "**Baseline**: the forecast beats the lag-free forecast t_sw,fc in at least half the runs.")
    L.append("")
    L.append(f"- **One-lag rule, one common f** (width 1 with the s\\*_run cutoff): "
             f"{', '.join(str(v) for v in P['f_ok']) or 'none'} (with the spec-literal width-1 cutoff: "
             f"{', '.join(str(v) for v in P['f_ok_with_w1_cutoff_on_s_frozen']) or 'none'}; first rule: "
             f"{', '.join(str(v) for v in P['f_ok_first_rule']) or 'none'}): largest {P['f_proposed']}.")
    L.append("- **One-lag rule, per setting:** " + "; ".join(
        f"{st} {P['f_proposed_per_setting'][st]}" for st in P["f_ok_per_setting"]) + ".")
    L.append(f"- **Minimal rule** (the cutoff strictly before both the crossing and the actual lag-free switch in every "
             f"crossed est run, no horizon requirement), one common f: "
             f"{', '.join(str(v) for v in P['f_ok_minimal']) or 'none'} (with the spec-literal width-1 cutoff: "
             f"{', '.join(str(v) for v in P['f_ok_minimal_with_w1_cutoff_on_s_frozen']) or 'none'}).")
    L.append("- **Why they differ:** width-1 SGD (lr 0.3, ρ = 1) reaches its switch within about one lag (~80 steps) of a "
             "0.9–0.95 cutoff, so the one-lag rule leaves only f = 0.8 there; at f = 0.8 the slow W2-A paths are "
             "extrapolated over thousands of steps and their crossing forecasts are off by hundreds to thousands of steps "
             "(their lag forecasts much less: §5).")
    L.append(f"- **Recommendation (FLAGGED: made after the est results were seen; the rules above are unchanged): one "
             f"common f = {P['f_recommended']}, family `{P['family']}`, window {P['window_frac']}·(t_c − t_start), at "
             f"least 50 steps, and the width-1 cutoff on s\\*_run.** At f = {P['f_recommended']} the cutoff precedes "
             "both events in every crossed est run of every setting with a forecast (§5, third column), by hundreds of steps on the GELU-T and W2-A paths (several "
             "lags; s margin ≈ 5% against \\|r\\| ≲ 0.2%); on the fast width-1 path it precedes the crossing by 1.5–1.9 "
             "lags but the lag-free switch by only 37–76 steps (0.4–0.9 lag; f = 0.9: 73–151 steps, 1.9–2.7 lags to the "
             "crossing), so for width 1 the author may prefer f = 0.9 (errors: §5). The crossing and lag errors at 0.95 "
             "are a few steps. f = 0.98 is more accurate still but its W2-A and GELU-T horizon is a few "
             "hundred steps; f = 0.8 extrapolates the slow paths too far. The final f is the author's decision.")

    def tol_table(title, tols):
        if not tols:
            return
        L.append("")
        L.append(title)
        L.append("")
        L.append("| setting | f | τ_cross (steps) | within | τ_lag (steps) | within | ratio band ±b | pilot median ratio | "
                 "beats lag-free |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for st, t in tols.items():
            L.append(f"| {SETTING_LABEL[st]} | {t['f']} | {_fmt(t['step_tolerance_cross'])} | {t['pilot_n_within_cross']}/"
                     f"{t['pilot_n']} | {_fmt(t['step_tolerance_lag'])} | {t['pilot_n_within_lag']}/{t['pilot_n']} | "
                     f"{_fmt(t['ratio_band_halfwidth'], 2)} | {_fmt(t['pilot_median_ratio'], 3)} | "
                     f"{t['pilot_n_beats_lagfree']}/{t['pilot_n_forecast']} |")
    tol_table(f"**Tolerances at the recommended f = {P['f_recommended']} (proposal):**", P["tolerances_recommended_f"])
    tol_table("**Tolerances at the per-setting one-lag f:**", P["tolerances"])
    tol_table(f"**Tolerances at the common one-lag f = {P['f_proposed']}:**", P["tolerances_common_f"])
    L.append("")
    L.append(f"The final f, window and tolerances are the author's decision. Compute (one process, one thread, nice 15; "
             f"runs + forecasts, all seeds, summed per-row wall clock): {_fmt(S['compute_secs_total'] / 3600, 2)} h, of "
             f"which {S['n_rows_over_60s']} forecast rows ran during the machine's overnight idle period and are "
             f"inflated (single forecasts of 6–34 min that take ~4 s by day); with those rows at their setting's median: "
             f"{_fmt(S['compute_secs_rows_over_60s_at_median'] / 3600, 2)} h. Peak RSS of any job < 0.5 GB.")
    L.append("")
    L.append("## 7. Reproduce")
    L.append("")
    L.append("```")
    L.append("python -m src.phase1a_pilot scan; python -m src.phase1a_pilot landscape")
    L.append("python -m src.phase1a_pilot run SETTING dev|est        # resumable; paths/ untracked, hashed in the rows")
    L.append("python -m src.phase1a_pilot forecast SETTING dev|est   # every family, window and f")
    L.append("python -m src.phase1a_pilot summarize                  # summary.json, error_estimate.md")
    L.append("```")
    L.append("")
    return "\n".join(L)


def summarize():
    S = _jsonable(build_summary())
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    (OUT / "error_estimate.md").write_text(render_md(S))
    print(json.dumps(S["proposal"], indent=1))


FIXTURES = {"gelu_random": 9_320_004, "w2a_T": 9_330_002}      # dev runs (the shortest paths of their setting)
FIXTURE_F = 0.95


def make_fixtures():
    """Test fixtures (tests/test_causal_forecast.py): the output path of one dev run per setting cut 200 rows past the
    cutoff at f = 0.95, the release row, the run row, and the forecast of the default configuration (which the test
    must reproduce)."""
    _setup()
    d = OUT / "fixtures"
    d.mkdir(parents=True, exist_ok=True)
    for st, seed in FIXTURES.items():
        memory_gate(f"fixture {st} {seed}")
        row = [r for r in _rows(OUT / f"runs_{st}.jsonl") if r["seed"] == seed][0]
        out, hid_rows, hid = _load(row)
        s = np.abs(out) if out.ndim == 1 else np.abs(out).sum(axis=1)
        cfg = C.ForecastConfig(f=FIXTURE_F)
        t_c = C.cutoff_step(s, FIXTURE_F * row["s_ref"])
        ctx = context(st, row)
        fc = C.run_forecast({"gelu_random": "gelu"}.get(st, "w2a"), make_inputs(st, row, out, hid_rows, hid, t_c, ctx),
                            cfg)
        meta = {"row": row, "config": cfg.__dict__, "t_c": t_c, "expected": _jsonable(fc)}
        p = d / f"{st}.npz"
        np.savez_compressed(p, out=out[:t_c + 200], hid_rows=hid_rows, hid=hid, meta=np.array(json.dumps(meta)))
        (d / f"{st}.sha256").write_text(f"{_sha(p)}  {p.name}\n")
        print(st, seed, t_c, fc["status"], fc["t_fc"], flush=True)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3], *(sys.argv[4:5]))
    elif cmd == "forecast":
        fams = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else FAMILY_GRID
        wins = tuple(float(v) for v in sys.argv[5].split(",")) if len(sys.argv) > 5 else WINDOW_GRID
        forecast(sys.argv[2], sys.argv[3], fams, wins)
    else:
        {"scan": scan, "landscape": landscape, "summarize": summarize, "fixtures": make_fixtures}[cmd]()
