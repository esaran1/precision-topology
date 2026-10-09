"""TRACK A design, EXPLORATORY step 2 (not a registration; not a pilot): Adam-specific causal extrapolations of the output
path.  Step 1 (explore.py) found that P at the switch is predictable from rows < t_c (median |error| ≈ 0.1–0.3%) but the
1C extrapolation (quad in log s, 5% window, ≥ 50 steps) misses 27% of Adam runs at f = 0.9 (window reaching back before
the rule point, into Adam's fast burst; log-concave fits that saturate below s*_run).  Same 120 exploration seeds and
a = 1.85; P rules: cut (v̂[t_c − 1]) and coupled_sw (v̂ at t_sw,fc from the coupled v̂ model).

Families (all read s rows < t_c only; window = the last n steps, never before the rule point t_R; at least 3 points):
  logquad_R   1C's quad in log s, window confined to [t_R, t_c) (n = max(3, min(50, t_c − t_R)))
  slin_K      s linear in t over the last K steps (K ∈ {10, 20, 40}, confined to ≥ t_R): Adam's near-constant step
  squad_K     s quadratic in t over the last K steps (anchored; stops at the vertex)

    python results/designs/trackA_explore/explore2.py run      # seeds 9,771,000 … 9,771,119 (resumable)
    python results/designs/trackA_explore/explore2.py summarize
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explore as E  # noqa: E402

import numpy as np  # noqa: E402

from src import causal_forecast as C  # noqa: E402
from src import linear_response as LR  # noqa: E402

RUNS2 = HERE / "explore2_runs.jsonl"
PM = ("cut", "coupled_sw")
HORIZON = 1.6


def _window(s, tR, t_c, K):
    n = max(3, min(K, t_c - tR))
    return np.asarray(s[t_c - n:t_c], float)


def make_family(name, K):
    def ext(s, tR, t_c, s_frozen):
        w = _window(s, tR, t_c, K)
        tau = np.arange(-len(w) + 1, 1, dtype=float)
        s_stop = HORIZON * s_frozen
        if name == "logquad_R":
            d, rate = C._poly_anchor(tau, np.log(w), 2, np.array([1.0]))
            if rate <= 0:
                return None, "no forecast: not growing at the cutoff"
            n = C._n_ahead(math.log(w[-1]), rate, s_stop, E.BUDGET - t_c + 1)
            yh = math.log(w[-1]) + C._poly_anchor(tau, np.log(w), 2, np.arange(1, n + 1, dtype=float))[0]
            sh = np.exp(yh)
        else:
            deg = 1 if name == "slin" else 2
            sc = max(1.0, float(abs(tau).max()))
            c = np.polyfit(tau / sc, w, deg)
            rate = np.polyval(np.polyder(c), 0.0) / sc
            if rate <= 0:
                return None, "no forecast: not growing at the cutoff"
            need = (s_stop - w[-1]) / rate
            n = int(max(1, min(E.BUDGET - t_c + 1, math.ceil(4 * max(need, 1.0)) + 10)))
            k = np.arange(1, n + 1, dtype=float)
            sh = w[-1] + np.polyval(c, k / sc) - np.polyval(c, 0.0)
            if deg == 2 and c[0] < 0:                               # stop at the vertex (no turning back)
                kv = -c[1] / (2 * c[0]) * sc
                sh = sh[k <= kv] if kv > 0 else sh[:0]
        hit = np.nonzero(sh >= s_stop)[0]
        if len(hit):
            sh = sh[:hit[0] + 1]
        if len(sh) == 0:
            return None, "no forecast: empty extrapolation"
        return sh, "ok"
    return ext


FAMS = {"logquad_R": make_family("logquad_R", 50)}
for K in (10, 20, 40):
    FAMS[f"slin_{K}"] = make_family("slin", K)
    FAMS[f"squad_{K}"] = make_family("squad", K)


def run():
    if os.nice(0) < 15:
        os.nice(15 - os.nice(0))
    import torch
    torch.set_num_threads(1)
    from src.phase1a_pilot import w1_train
    prev = {json.loads(l)["seed"]: json.loads(l) for l in E.RUNS.read_text().splitlines()}
    done = set() if not RUNS2.exists() else {json.loads(l)["seed"] for l in RUNS2.read_text().splitlines()}
    E.log("explore2 start")
    for seed, r in sorted(prev.items()):
        if seed in done or r.get("t_obs") is None:
            continue
        t0 = time.time()
        x, y, s_fr = E.frozen(seed)
        W, M, V = w1_train(seed, "adam", E.BUDGET, a=E.A)
        out = {"seed": seed, "fc": {}}
        for f in E.FS:
            c = r["fc"][str(f)]
            if c["t_c"] is None:
                continue
            cfg = C.ForecastConfig(adam_P="cutoff").with_f(f)
            out["fc"][str(f)] = {name: E.forecast(W, M, V, x, y, s_fr, c["t_c"], cfg, fn, c["t_sw"], pmodes=PM)
                                 for name, fn in FAMS.items()}
        out["secs"] = round(time.time() - t0, 1)
        with open(RUNS2, "a") as fh:
            fh.write(json.dumps(LR._jsonable(out)) + "\n")
        E.log(f"explore2 seed {seed} secs {out['secs']}")
        E.rss_guard()
    E.log("explore2 end")


def summarize():
    base = {json.loads(l)["seed"]: json.loads(l) for l in E.RUNS.read_text().splitlines()}
    rows2 = {json.loads(l)["seed"]: json.loads(l) for l in RUNS2.read_text().splitlines()}
    half = sorted(base)[len(base) // 2]
    S = {"label": "EXPLORATORY step 2 (Track A design)", "by_f": {}}
    for f in E.FS:
        Sf = {}
        for fam in FAMS:
            for pm in PM:
                merged = []
                for seed, r in base.items():
                    if seed not in rows2 or str(f) not in rows2[seed]["fc"]:
                        continue
                    rr = dict(r)
                    rr["fc"] = {str(f): {**r["fc"][str(f)], "ext": rows2[seed]["fc"][str(f)][fam]}}
                    merged.append(rr)
                pr, sk = E.per_run(merged, f, "ext", pm)
                a = E.crit(pr, (10, 15, 0.2))
                p1 = [q for q in pr if q["seed"] < half]
                p2 = [q for q in pr if q["seed"] >= half]
                c1 = E.crit(p1, None)
                rat = [q["r_obs"] / q["r_fc"] for q in p1 if q["t_fc"] is not None and q["r_fc"]]
                se = 1.2533 * np.std(rat, ddof=1) / math.sqrt(len(rat))
                tol = (max(5, E._ceil5(1.5 * c1["q90_cross"])), max(5, E._ceil5(1.5 * c1["q90_lag"])),
                       math.ceil((abs(c1["median_ratio"] - 1) + 2 * se) / 0.05 - 1e-12) * 0.05)
                st = {}
                for q in pr:
                    if q["t_fc"] is None:
                        st[q["status"]] = st.get(q["status"], 0) + 1
                Sf[f"{fam}/{pm}"] = {"all_at_W1_tol": a, "split_tol": tol, "second_half": E.crit(p2, tol),
                                     "miss_status": st, "skipped": sk}
        S["by_f"][str(f)] = Sf
    (HERE / "summary2.json").write_text(json.dumps(LR._jsonable(S), indent=1))
    for f, Sf in S["by_f"].items():
        for k, d in Sf.items():
            a, h = d["all_at_W1_tol"], d["second_half"]
            print(f"f {f} {k:22s} n {a['n']} miss {a['misses']} C1 {a['C1']:.2f} C2 {a['C2']:.2f} q90x {a['q90_cross']} "
                  f"q90l {a['q90_lag']} med {a['median_ratio']:.3f} q10/q90 {a['ratio_q10']:.2f}/{a['ratio_q90']:.2f} "
                  f"C4hi {a.get('C4_hi', float('nan')):.1f} | tol {d['split_tol']} 2nd-half C1 {h['C1']:.2f} "
                  f"C2 {h['C2']:.2f} med {h['median_ratio']:.3f} PASS {h['PASS']}")


if __name__ == "__main__":
    {"run": run, "summarize": summarize}[sys.argv[1]]()
