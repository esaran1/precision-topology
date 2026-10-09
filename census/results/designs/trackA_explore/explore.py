"""TRACK A design: EXPLORATORY computation for the causal Adam per-run forecast (NOT a registration; NOT a pilot).

Exploration seeds only: 9,771,000 + i (prefix 9771 unused anywhere in src/, tests/, results/, paper/, notes/ at the
time of writing).  Activation a = 1.85 (Test 2A's value: landscape results/track2a/landscape.json; it already has
data, so exploring there spends no fresh activation value).  Width 1, free Adam lr 0.01, make_data(200, seed), local
torch Generator (phase1a_pilot.w1_train; no global RNG state), budget 32,000, every-step detection (w1_observe).

Question: can Adam's preconditioner at the switch be predicted from rows < t_c only, and does that give a causal per-run
forecast that meets 1C-style criteria?  Per seed and f ∈ {0.90, 0.95} (1C's W1 nested cutoff on s*_run):

  path   ext     the frozen 1C extrapolation (quad / 5% / 1.6), causal;
         oracle  the ACTUAL s path (non-causal reference; isolates extrapolation error from P error).
  P      cut          v̂[t_c − 1] (Phase 1B's rule for 2A);
         decay        g² = 0 after the cutoff: v_raw decays by β₂ per step, v̂ = v_raw/(1 − β₂^t), to t_sw,fc;
         gwin         g² held at its mean over the window (g recovered from M: g_t = (m_t − β₁m_{t−1})/(1 − β₁));
         logquad      per-coordinate quadratic extrapolation of log v̂ over the window, to t_sw,fc;
         coupled      v̂ rolled forward with g = Hδ from the linear-response recursion itself (time-varying P after the
                      cutoff; actual v̂ rows before it) -- the "model of v̂ given the extrapolated gradient path";
         coupled_sw   P frozen at the coupled model's v̂ at t_sw,fc;
         oracle_sw    v̂ at the ACTUAL t_sw (non-causal; = Test 2A's registered rule; the ceiling).
Causal modes read only GuardedArray views with cutoff t_c (output, M, v̂; hidden: one row, the rule point).

    python results/designs/trackA_explore/explore.py gate        # memory gate (separate, logged); exit 1 if not ok
    python results/designs/trackA_explore/explore.py run N       # seeds 9,771,000 … +N−1 (resumable)
    python results/designs/trackA_explore/explore.py summarize   # summary.json
"""
from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import causal_forecast as C  # noqa: E402
from src import linear_response as LR  # noqa: E402

A = 1.85
SEED0 = 9_771_000
BUDGET = 32_000
FS = (0.90, 0.95)
PMODES = ("cut", "decay", "gwin", "logquad", "coupled", "coupled_sw", "oracle_sw")
CAUSAL_P = ("cut", "decay", "gwin", "logquad", "coupled", "coupled_sw")
RUNS = HERE / "explore_runs.jsonl"
RSS_LIMIT = 1024 ** 3
B1, B2, EPS = LR.BETA1, LR.BETA2, LR.EPS
HID = [0, 1, 3]


def log(msg):
    with open(HERE / "explore.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def gate():
    from src.act_fold import check_memory
    ok, f, w = check_memory()
    with open(HERE / "memory_gate.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} free={f}% swap_free={w}MB {'OK' if ok else 'WAIT'}\n")
    print(ok, f, w)
    return 0 if ok else 1


def rss_guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > RSS_LIMIT:
        raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 1 GB")


# ------------------------------------------------------------------------------------------ per-seed inputs
def frozen(seed):
    from src.fold1d import make_data
    d = json.loads((ROOT / "results" / "track2a" / "landscape.json").read_text())
    s_pop = d["s_star_pop"]
    x, y = make_data(200, seed)
    x, y = x.double().numpy(), y.double().numpy()
    th, res, H = LR.newton(np.array(d["theta_star"]), s_pop, A, x, y)
    if res > 1e-9 or np.linalg.eigvalsh(H).min() <= 0:
        return x, y, None
    B = LR.Branch(s_pop, th, A, x, y, 0.3 * s_pop, 1.7 * s_pop, 0.002 * s_pop)
    return x, y, B.switch(s_pop)


# ------------------------------------------------------------------------------------------ P predictors (rows < t_c)
def _raw(vhat, t):
    return vhat * (1 - B2 ** t)


def predict_vhat(mode, Vg, Mg, t_c, n_w, t_target):
    """v̂ (4 coords) at step t_target ≥ t_c − 1 from guarded rows < t_c only."""
    last = t_c - 1
    v_last = C._row(Vg, last)
    k = t_target - last
    if mode == "cut" or k <= 0:
        return v_last
    if mode == "decay":
        return _raw(v_last, last) * B2 ** k / (1 - B2 ** t_target)
    if mode == "gwin":
        Mw = np.array(Mg[last - n_w:t_c])                     # rows last − n_w … last
        g = (Mw[1:] - B1 * Mw[:-1]) / (1 - B1)
        g2 = np.mean(g ** 2, axis=0)
        v = _raw(v_last, last)
        v = B2 ** k * v + (1 - B2 ** k) * g2
        return v / (1 - B2 ** t_target)
    if mode == "logquad":
        Vw = np.array(Vg[t_c - n_w:t_c])
        tau = np.arange(-n_w + 1, 1, dtype=float)
        out = np.empty(4)
        for j in range(4):
            d, _ = C._poly_anchor(tau, np.log(Vw[:, j]), 2, np.array([float(k)]))
            out[j] = math.exp(math.log(Vw[-1, j]) + d[0])
        return out
    raise ValueError(mode)


# ------------------------------------------------------------------------------------------ R4 with time-varying P
def lag_tv(s_ext, t0, B, a, delta0, lr, s_switch, m0, P_vis, v_raw0, t_c, couple=True):
    """lag_forecast_1d with a time-varying preconditioner: for steps < t_c the visible P rows (P_vis[t], causal), after
    the cutoff v̂ rolled forward with g = Hδ (couple) and P = 1/(√v̂ + ε).  Returns (t_hit|None, status, v̂ at each
    step after the cutoff as {t: v̂ hid})."""
    seg = np.asarray(s_ext, float)[t0:]
    inside = np.array([B.contains(v) for v in seg])
    T_end = len(seg) if inside.all() else int(np.argmin(inside))
    if T_end < 2:
        return None, "no forecast: rule point outside the branch grid", {}
    seg = seg[:T_end]
    th = [np.asarray(B.theta(v), float) for v in seg[:1]]
    d = np.asarray(delta0, float).copy()
    m = np.asarray(m0, float).copy()
    v = None
    vh = {}
    gap = lambda i, dd: LR.gap_exact(th[i][0] + dd[0], th[i][1] + dd[1], a)  # noqa: E731
    if gap(0, d) > 0:
        return t0, "ok", vh
    rad = 0.0
    for i in range(T_end - 1):
        t_next = t0 + i + 1
        if len(th) < i + 2:
            th.append(np.asarray(B.theta(seg[i + 1]), float))
        H = np.asarray(B.hess(seg[i]), float)
        g = H @ d
        m = B1 * m + (1 - B1) * g
        if t_next < t_c:
            P = P_vis[t_next]
        else:
            if v is None:
                v = v_raw0.copy()                            # raw v at t_c − 1 (hidden coords)
            v = B2 * v + (1 - B2) * (g ** 2 if couple else 0.0)
            vhat = v / (1 - B2 ** t_next)
            vh[t_next] = vhat
            P = 1.0 / (np.sqrt(vhat) + EPS)
        if i % 10 == 0:
            rad = max(rad, LR.step_map_radius(H, P, lr, True))
        d = d - lr * P * (m / (1 - B1 ** t_next)) - (th[i + 1] - th[i])
        if not np.all(np.isfinite(d)) or np.abs(d).max() > LR.DIVERGED:
            return None, "no forecast: unstable / diverged (Track 1 rule)", vh
        if gap(i + 1, d) > 0:
            if rad > 1.0:
                return None, "no forecast: unstable / diverged (Track 1 rule)", vh
            return t0 + i + 1, "ok", vh
    return None, "no forecast: no predicted crossing on the extrapolated path", vh


# ------------------------------------------------------------------------------------------ one forecast
def forecast(W, M, V, x, y, s_frozen, t_c, cfg, path_mode, t_sw_act, pmodes=None):
    """Every P mode at one cutoff and one path mode.  Returns {pmode: record}, plus shared info."""
    out = C.guard(W[:, 2], t_c, name="w2")
    hid = C.guard(W[:, HID], t_c, max_rows=1, name="hidden")
    Mg = C.guard(M, t_c, name="M")
    Vg = C.guard(V, t_c, name="Vhat")
    w2 = C._prefix(out, t_c)
    s = np.abs(w2)
    tR = C.causal_rule_step(s, s_frozen)
    if tR is None:
        return {"status": "no rule point"}
    flip = LR.D_FLIP if w2[tR] < 0 else np.ones(3)
    TH = C._row(hid, tR) * flip
    B, s_run = C.w1_branch(A, x, y, s_frozen, float(s[tR]), TH)
    if B is None or s_run is None:
        return {"status": "no branch / switch"}
    m0 = C._row(Mg, tR)[HID] * flip
    s_hat, st, xinfo = C.extrapolate_scale(s, cfg, s_frozen, max(0, BUDGET - t_c + 1), t_start=tR)
    n_w = xinfo.get("n_window", C.window_length(t_c, cfg, tR))
    info = {"t_rule": tR, "s_run": float(s_run), "n_window": n_w, "ext_status": st}
    if callable(path_mode):                                # exploratory Adam-specific extrapolation (explore2.py)
        s_hat2, st2 = path_mode(s, tR, t_c, s_frozen)
        if st2 != "ok":
            return {**info, "status": st2}
        s_ext = np.concatenate([s, s_hat2])
    elif path_mode == "ext":
        if st != "ok":
            return {**info, "status": st}
        s_ext = np.concatenate([s, s_hat])
    else:                                                  # oracle: the actual path (NON-CAUSAL reference)
        s_ext = np.abs(W[:, 2])
    t_sw_fc = LR._first_ge(s_ext, s_run)
    info["t_sw_fc"] = t_sw_fc
    if t_sw_fc is None:
        return {**info, "status": "crossing forecast, no forecast switch"}
    delta0 = TH - B.theta(float(s[tR]))
    gapf = lambda th: LR.gap_exact(th[0], th[1], A)  # noqa: E731
    bc1 = lambda t: 1 - B1 ** t  # noqa: E731
    recs = {}
    v_true_sw = V[t_sw_act] if t_sw_act is not None else None
    # P visible rows (causal) for the time-varying mode
    P_vis = {t: 1.0 / (np.sqrt(C._row(Vg, t)[HID]) + EPS) for t in range(tR + 1, t_c)}
    v_raw0 = _raw(C._row(Vg, t_c - 1)[HID], t_c - 1)
    t_hit, stt, vh = lag_tv(s_ext, tR, B, A, delta0, LR.LR_ADAM, s_run, m0, P_vis, v_raw0, t_c, couple=True)
    recs["coupled"] = {"t_fc": t_hit, "status": stt}
    for pm in (pmodes or PMODES):
        if pm == "coupled":
            continue
        if pm == "oracle_sw":
            if v_true_sw is None:
                continue
            vhat = v_true_sw
        elif pm == "coupled_sw":
            if t_sw_fc in vh:
                vhat = np.zeros(4)
                vhat[HID] = vh[t_sw_fc]
            elif t_sw_fc < t_c:
                vhat = C._row(Vg, t_sw_fc)
            else:                                           # crossing before t_sw,fc: roll the decay-only model
                vhat = predict_vhat("decay", Vg, Mg, t_c, n_w, t_sw_fc)
        else:
            vhat = predict_vhat(pm, Vg, Mg, t_c, n_w, t_sw_fc)
        P = 1.0 / (np.sqrt(vhat[HID]) + EPS)
        r = C.lag_forecast_1d(s_ext, tR, B.theta, B.hess, B.contains, gapf, delta0, LR.LR_ADAM, s_run, P=P, m0=m0,
                              bc1_of=bc1)
        rec = {"t_fc": r.get("t_hit"), "status": r["status"]}
        if v_true_sw is not None:
            P_true = 1.0 / (np.sqrt(v_true_sw[HID]) + EPS)
            rec["P_relerr"] = (P / P_true - 1).tolist()
        recs[pm] = rec
    for pm, rec in recs.items():
        t = rec["t_fc"]
        rec["s_fc"] = float(s_ext[t]) if t is not None else None
        rec["r_fc"] = float(s_ext[t] / s_run - 1) if t is not None else None
    mi = max(g.max_index_read for g in (out, hid, Mg, Vg))
    assert mi < t_c, ("causality", mi, t_c)
    info["max_index_read"] = int(mi)
    return {**info, "status": "ok", "P": recs}


def run_seed(seed):
    from src.phase1a_pilot import w1_observe, w1_train
    t0 = time.time()
    x, y, s_fr = frozen(seed)
    row = {"seed": seed, "a": A, "s_frozen": s_fr}
    if s_fr is None:
        return {**row, "status": "no frozen switch"}
    W, M, V = w1_train(seed, "adam", BUDGET, a=A)
    t_train = time.time() - t0
    t_obs = w1_observe(W, a=A)
    row.update(t_obs=t_obs, s_obs=float(abs(W[t_obs, 2])) if t_obs is not None else None,
               secs_train=round(t_train, 1))
    cfg0 = C.ForecastConfig(adam_P="cutoff")
    row["fc"] = {}
    for f in FS:
        t_c, tR, s_run = C.w1_cutoff_on_s_run(np.abs(W[:, 2]), lambda k: W[k, 2], f, s_fr, A, x, y,
                                              lambda k: W[k, HID])
        t_sw = LR._first_ge(np.abs(W[:, 2]), s_run) if s_run is not None else None
        rec = {"t_c": t_c, "t_sw": t_sw, "s_run_cut": s_run}
        if t_c is not None:
            for pmode in ("ext", "oracle"):
                rec[pmode] = forecast(W, M, V, x, y, s_fr, t_c, cfg0.with_f(f), pmode, t_sw)
        row["fc"][str(f)] = rec
    row["secs"] = round(time.time() - t0, 1)
    return row


def run(n):
    if os.nice(0) < 15:
        os.nice(15 - os.nice(0))
    import torch
    torch.set_num_threads(1)
    done = set() if not RUNS.exists() else {json.loads(l)["seed"] for l in RUNS.read_text().splitlines()}
    log(f"run start n={n} done={len(done)}")
    for i in range(n):
        seed = SEED0 + i
        if seed in done:
            continue
        try:
            row = run_seed(seed)
        except Exception as e:  # noqa: BLE001
            row = {"seed": seed, "status": f"error: {type(e).__name__}: {e}"}
        with open(RUNS, "a") as fh:
            fh.write(json.dumps(LR._jsonable(row)) + "\n")
        log(f"seed {seed} secs {row.get('secs')} t_obs {row.get('t_obs')}")
        rss_guard()
    log("run end")


# ------------------------------------------------------------------------------------------ summary
def _q(v, p):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.quantile(v, p)) if v else None


def _ceil5(x):
    return 5 * math.ceil(x / 5 - 1e-12)


def crit(rows, tol):
    """1C criteria on a list of per-run dicts {t_fc, t_swfc, r_fc, t_obs, t_sw, r_obs}; misses: t_fc None."""
    n = len(rows)
    if n == 0:
        return None
    fcs = [r for r in rows if r["t_fc"] is not None]
    e1 = [abs(r["t_fc"] - r["t_obs"]) for r in fcs]
    e2 = [abs((r["t_fc"] - r["t_swfc"]) - (r["t_obs"] - r["t_sw"])) for r in fcs]
    ratio = [r["r_obs"] / r["r_fc"] for r in fcs if r["r_fc"]]
    D = np.array([abs(r["t_fc"] - r["t_obs"]) - abs(r["t_swfc"] - r["t_obs"]) for r in fcs], float)
    out = {"n": n, "misses": n - len(fcs), "q50_cross": _q(e1, .5), "q90_cross": _q(e1, .9), "max_cross": _q(e1, 1),
           "q50_lag": _q(e2, .5), "q90_lag": _q(e2, .9), "median_ratio": _q(ratio, .5),
           "ratio_q10": _q(ratio, .1), "ratio_q90": _q(ratio, .9)}
    if tol:
        tc, tl, b = tol
        out["C1"] = sum(e <= tc for e in e1) / n
        out["C2"] = sum(e <= tl for e in e2) / n
        out["C3_ok"] = out["median_ratio"] is not None and abs(out["median_ratio"] - 1) <= b
        if len(D) >= 2:
            g = np.random.default_rng(9771000)
            mu = D[g.integers(0, len(D), (10000, len(D)))].mean(axis=1)
            out["C4_hi"] = float(np.percentile(mu, 97.5))
        out["PASS"] = bool(out["C1"] >= .8 and out["C2"] >= .8 and out["C3_ok"] and out.get("C4_hi", 1) < 0)
    return out


def per_run(rows, f, path, pm):
    res, skipped = [], {"no_cross": 0, "cutoff_not_before": 0, "no_tsw": 0}
    for r in rows:
        if "fc" not in r or r.get("t_obs") is None:
            skipped["no_cross"] += 1
            continue
        c = r["fc"][str(f)]
        if c["t_c"] is None or c["t_c"] >= r["t_obs"]:
            skipped["cutoff_not_before"] += 1
            continue
        if c["t_sw"] is None:
            skipped["no_tsw"] += 1
            continue
        fc = c[path]
        p = fc.get("P", {}).get(pm, {}) if fc.get("status") == "ok" else {}
        s_run = c["s_run_cut"]
        res.append({"seed": r["seed"], "t_fc": p.get("t_fc"), "t_swfc": fc.get("t_sw_fc"), "r_fc": p.get("r_fc"),
                    "t_obs": r["t_obs"], "t_sw": c["t_sw"], "r_obs": r["s_obs"] / s_run - 1,
                    "status": p.get("status", fc.get("status")), "P_relerr": p.get("P_relerr"),
                    "t_c": c["t_c"]})
    return res, skipped


def summarize():
    rows = [json.loads(l) for l in RUNS.read_text().splitlines()]
    S = {"label": "EXPLORATORY (Track A design; not a registration, not a pilot)", "a": A, "seeds": len(rows),
         "seed_range": [min(r["seed"] for r in rows), max(r["seed"] for r in rows)],
         "crossed": sum(r.get("t_obs") is not None for r in rows),
         "secs_per_seed_median": _q([r.get("secs") for r in rows], .5), "by_f": {}}
    half = sorted(r["seed"] for r in rows)[len(rows) // 2]
    for f in FS:
        Sf = {}
        for path in ("ext", "oracle"):
            for pm in PMODES:
                pr, sk = per_run(rows, f, path, pm)
                key = f"{path}/{pm}"
                d = {"all": crit(pr, (10, 15, 0.2)), "skipped": sk}
                # pilot-rule split: tolerances from the first half (1.5·q90 → next multiple of 5; b = |med − 1| + 2SE
                # → next 0.05), applied to the second half
                p1 = [r for r in pr if r["seed"] < half]
                p2 = [r for r in pr if r["seed"] >= half]
                c1 = crit(p1, None)
                if c1 and c1["q90_cross"] is not None:
                    rat = [r["r_obs"] / r["r_fc"] for r in p1 if r["t_fc"] is not None and r["r_fc"]]
                    se = 1.2533 * np.std(rat, ddof=1) / math.sqrt(len(rat)) if len(rat) > 1 else float("nan")
                    tol = (max(5, _ceil5(1.5 * c1["q90_cross"])), max(5, _ceil5(1.5 * c1["q90_lag"])),
                           math.ceil((abs(c1["median_ratio"] - 1) + 2 * se) / 0.05 - 1e-12) * 0.05)
                    d["split"] = {"tol_from_first_half": tol, "first_half": crit(p1, tol), "second_half": crit(p2, tol)}
                if pm in CAUSAL_P or pm == "oracle_sw":
                    pe = np.array([r["P_relerr"] for r in pr if r["P_relerr"] is not None])
                    if len(pe):
                        d["P_relerr_vs_true_tsw"] = {"median_abs": np.median(np.abs(pe), axis=0).tolist(),
                                                    "q90_abs": np.quantile(np.abs(pe), .9, axis=0).tolist(),
                                                    "coords": ["w1", "b1", "b2"]}
                st = {}
                for r in pr:
                    if r["t_fc"] is None:
                        st[r["status"]] = st.get(r["status"], 0) + 1
                d["miss_status"] = st
                Sf[key] = d
        hz = [(r["t_obs"] - r["t_c"], r["t_sw"] - r["t_c"]) for r in per_run(rows, f, "ext", "cut")[0]]
        Sf["horizon_steps_obs_minus_tc_q10_50_90"] = [_q([h[0] for h in hz], p) for p in (.1, .5, .9)]
        Sf["horizon_steps_tsw_minus_tc_q10_50_90"] = [_q([h[1] for h in hz], p) for p in (.1, .5, .9)]
        S["by_f"][str(f)] = Sf
    (HERE / "summary.json").write_text(json.dumps(LR._jsonable(S), indent=1))
    print(json.dumps({f: {k: (v["all"] if isinstance(v, dict) and "all" in v else v) for k, v in S["by_f"][f].items()}
                      for f in S["by_f"]}, indent=0, default=str)[:6000])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gate":
        sys.exit(gate())
    if cmd == "run":
        run(int(sys.argv[2]))
    elif cmd == "summarize":
        summarize()
