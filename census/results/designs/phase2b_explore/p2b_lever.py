"""EXPLORATORY (NOT a registration), for the Phase 2B design page (results/designs/phase2b_lever_design.md): the global
learning-rate control, matched-training-loss comparison, converged-budget behaviour and onset definitions, on the
NON-registered exploration seeds 2,953,000-2,953,019 only (range 2,953,000-2,953,099 never to be registered).

Benchmark, initialisation, Adam update, rho2 and the shifted test sets: imported unchanged from p2b_explore.py (v3's
init_net inside torch.random.fork_rng; numpy Adam checked against torch.optim.Adam in p2b_check.json).

Conditions (Adam, betas (0.9, 0.999), eps 1e-8; every run trained a FIXED T_RUN = 100,000 steps, no early stop):
  std      lr 0.01 on every parameter (v3 protocol; arm 1)
  out16    lr 0.01/16 on the output weights v; 0.01 on W, c and the output bias b (arm 2 as explored)
  out16b   lr 0.01/16 on v AND b; 0.01 on W, c (arm 2 variant: whole output layer)
  glob     lr 0.01/G on every parameter (arm 3), G = median over the 20 seeds of
           t(adam_r16 to 3 s*)/t(adam_r1 to 3 s*) from the committed runs/ (= 12.177 -> G = 12.18)
  globb    lr 0.01/Gb on every parameter, Gb = the same median for out16b (computed here from this script's runs)
  glob16   lr 0.01/16 on every parameter (added after seeing glob: a cost-matched candidate, since glob's realised step
           cost to 3 s* is below out16's)
Training loss for matching = the data loss BCE (mean), i.e. the objective minus (lam/2)(|W|^2 + |c|^2).

    python p2b_lever.py run COND SEED        # one run -> runs_lever/COND_SEED.json
    python p2b_lever.py summarise            # -> p2b_lever.json, p2b_lever_tables.md
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2b_explore as X  # noqa: E402

SEEDS = tuple(range(2_953_000, 2_953_020))
RUNS = HERE / "runs_lever"
T_RUN = 100_000
CKPT = (1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000)
LOSSES = (0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01)
SCALES = dict(X.SCALES); SCALES["4s*"] = 4 * X.S_STAR
G_EXPLORE = 12.18


def g_factor_b():
    r = [json.loads((RUNS / f"out16b_{s}.json").read_text())["t_scale"]["3s*"]
         / json.loads((RUNS / f"std_{s}.json").read_text())["t_scale"]["3s*"] for s in SEEDS]
    return round(float(np.median(r)), 2)


def lr_vec(cond):
    lr = np.full(17, X.ADAM_LR)
    if cond == "out16":
        lr[12:16] /= 16
    elif cond == "out16b":
        lr[12:17] /= 16
    elif cond == "glob":
        lr /= G_EXPLORE
    elif cond == "globb":
        lr /= g_factor_b()
    elif cond == "glob16":
        lr /= 16
    return lr


def bce(L, th):
    return L - 0.5 * X.LAM * ((th[:8] ** 2).sum() + (th[8:12] ** 2).sum())


def run(cond, seed):
    lr = lr_vec(cond)
    t_match = None
    if cond != "std":
        t_match = json.loads((RUNS / f"std_{seed}.json").read_text())["t_scale"]["3s*"]
    th, _ = X.init_row(seed)
    m = np.zeros(17); v = np.zeros(17)
    S = np.empty(T_RUN + 1); B = np.empty(T_RUN + 1); R = np.empty(T_RUN + 1)
    keep, t_scale, t_loss = {}, {}, {}
    buf, buf_t = [], []
    t0 = time.time()

    def flush():
        if buf:
            R[buf_t[0]:buf_t[-1] + 1] = X.rho2_rows(np.array(buf)); buf.clear(); buf_t.clear()
            if X.rss() > X.RSS_STOP:
                raise SystemExit("RSS > 3 GB: stopped")

    for t in range(T_RUN + 1):
        L, g = X.loss_grad(th)
        s = float(np.abs(th[12:16]).sum()); b_ = bce(L, th)
        if not np.isfinite(th).all():
            raise SystemExit("nonfinite")
        S[t] = s; B[t] = b_; buf.append(th.copy()); buf_t.append(t)
        for nm, val in SCALES.items():
            if nm not in t_scale and s >= val:
                t_scale[nm] = t; keep["scale_" + nm] = th.copy()
        for lv in LOSSES:
            k = f"loss_{lv}"
            if k not in t_loss and b_ <= lv:
                t_loss[k] = t; keep[k] = th.copy()
        if t in CKPT:
            keep[f"step_{t}"] = th.copy()
        if t_match is not None and t == t_match:
            keep["match_step"] = th.copy()
        if len(buf) >= X.CHUNK:
            flush()
        if t == T_RUN:
            break
        m = m + (1 - X.B1) * (g - m); v = v * X.B2 + (1 - X.B2) * g * g
        k1 = t + 1
        th = th - (lr / (1 - X.B1 ** k1)) * m / (np.sqrt(v) / math.sqrt(1 - X.B2 ** k1) + X.EPS)
    flush()

    def onset(end):
        """Lasting onset for a run ending at step `end`: the first step t_on such that rho2 >= q at every step in
        [t_on, end]; reported with s(t_on), and whether t_on > 0 (an upward passage) or 0 (never below q)."""
        r = R[:end + 1]
        if r[-1] < X.Q:
            return None
        below = np.nonzero(r < X.Q)[0]
        t_on = int(below[-1] + 1) if len(below) else 0
        return {"t": t_on, "s": float(S[t_on]), "s_over_s_star": float(S[t_on] / X.S_STAR), "from_init": t_on == 0}

    ends = {f"T{t}": t for t in (40_000, 100_000)}
    if "3s*" in t_scale:
        ends["3s*"] = t_scale["3s*"]
    res = {"label": "EXPLORATORY (non-registered seeds 2,953,000+; never to be registered)", "cond": cond, "seed": seed,
           "lr": lr.tolist(), "t_match": t_match, "t_scale": t_scale, "t_loss": t_loss,
           "s_at_loss": {k: float(S[t]) for k, t in t_loss.items()},
           "overshoot_scale": {k: float(S[t] / SCALES[k] - 1) for k, t in t_scale.items()},
           "onset": {k: onset(e) for k, e in ends.items()},
           "rho2_init": float(R[0]), "n_rho2_down_passages_after_3s*": None,
           "seconds": time.time() - t0, "max_rss_gb": X.rss() / 1e9}
    if "3s*" in t_scale:
        r = R[t_scale["3s*"]:]
        res["n_rho2_down_passages_after_3s*"] = int(((r[1:] < X.Q) & (r[:-1] >= X.Q)).sum())
    res["at"] = {k: X.metrics(p) | {"bce": float(bce(X.loss_grad(p)[0], p))} for k, p in keep.items()}
    idx = np.unique(np.r_[0, np.geomspace(1, T_RUN, 300).astype(int)])
    res["traj"] = {"t": idx.tolist(), "s": S[idx].tolist(), "bce": B[idx].tolist(), "rho2": R[idx].tolist()}
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{cond}_{seed}.json"
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, default=float)); tmp.replace(out)
    print(json.dumps({k: res[k] for k in ("cond", "seed", "t_scale", "seconds", "max_rss_gb")}, default=float), flush=True)


# ------------------------------------------------------------------------------------------ summary
def _q(a):
    a = np.asarray([x for x in a if x is not None], float)
    if not len(a):
        return None
    return {"n": int(len(a)), "median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None}


def summarise():
    conds = [c for c in ("std", "out16", "out16b", "glob", "globb", "glob16") if all((RUNS / f"{c}_{s}.json").exists() for s in SEEDS)]
    runs = {c: {s: json.loads((RUNS / f"{c}_{s}.json").read_text()) for s in SEEDS} for c in conds}
    pts = ([f"scale_{k}" for k in SCALES] + [f"loss_{lv}" for lv in LOSSES] + [f"step_{t}" for t in CKPT]
           + ["match_step"])
    out = {"label": "EXPLORATORY (non-registered seeds 2,953,000-2,953,019; never to be registered)", "seeds": SEEDS,
           "G_explore": G_EXPLORE, "Gb": g_factor_b() if "out16b" in runs else None, "conds": {}}
    for c in conds:
        R = runs[c]
        d = {"steps_to": {k: _q([R[s]["t_scale"].get(k) for s in SEEDS]) for k in SCALES},
             "steps_to_over_std": {k: _q([R[s]["t_scale"][k] / runs["std"][s]["t_scale"][k] for s in SEEDS
                                          if k in R[s]["t_scale"]]) for k in SCALES},
             "steps_to_loss": {k: _q([R[s]["t_loss"].get(f"loss_{k}") for s in SEEDS]) for k in LOSSES},
             "s_at_loss": {k: _q([R[s]["s_at_loss"].get(f"loss_{k}") for s in SEEDS]) for k in LOSSES},
             "max_overshoot_scale": max(max(R[s]["overshoot_scale"].values()) for s in SEEDS),
             "points": {}, "onset": {}}
        for p in pts:
            key = "scale_3s*" if (p == "match_step" and c == "std") else p
            e = {"n": sum(R[s]["at"].get(key) is not None for s in SEEDS)}
            for qn in ("rho2", "acc_shuffled", "acc_reversed", "s", "bce", "gplus"):
                e[qn] = _q([R[s]["at"][key][qn] for s in SEEDS if R[s]["at"].get(key)])
                if c != "std":
                    k2 = "scale_3s*" if p == "match_step" else p
                    dd = [R[s]["at"][key][qn] - runs["std"][s]["at"][k2][qn] for s in SEEDS
                          if R[s]["at"].get(key) and runs["std"][s]["at"].get(k2)]
                    e["d_" + qn] = _q(dd)
                    e["n_up_" + qn] = int(sum(x > 0 for x in dd)); e["n_down_" + qn] = int(sum(x < 0 for x in dd))
            d["points"][p] = e
        for end in ("3s*", "T40000", "T100000"):
            o = [R[s]["onset"].get(end) for s in SEEDS]
            ss = [x["s_over_s_star"] for x in o if x]
            d["onset"][end] = {"n_lasting": len(ss), "n_above_s*": int(sum(x > 1 for x in ss)),
                               "n_from_init": int(sum(x["from_init"] for x in o if x)), "s_over_s_star": _q(ss),
                               "n_in_s*_band": int(sum(1 <= x <= 1.25 for x in ss))}
        d["n_down_after_3s*"] = _q([R[s]["n_rho2_down_passages_after_3s*"] for s in SEEDS])
        d["seconds"] = _q([R[s]["seconds"] for s in SEEDS]); d["max_rss_gb"] = max(R[s]["max_rss_gb"] for s in SEEDS)
        out["conds"][c] = d
    # consistency with the committed exploration (same seeds): std vs adam_r1, out16 vs adam_r16
    cons = {}
    for mine, theirs in (("std", "adam_r1"), ("out16", "adam_r16")):
        if mine in runs:
            dt, dr = [], []
            for s in SEEDS:
                old = json.loads((HERE / "runs" / f"{theirs}_{s}.json").read_text())
                dt.append(abs(old["t_scale"]["3s*"] - runs[mine][s]["t_scale"]["3s*"]))
                dr.append(abs(old["at"]["scale_3s*"]["rho2"] - runs[mine][s]["at"]["scale_3s*"]["rho2"]))
            cons[f"{mine}_vs_{theirs}"] = {"max_abs_dsteps_3s*": int(max(dt)), "max_abs_drho2_3s*": float(max(dr))}
    out["consistency_with_committed_runs"] = cons
    (HERE / "p2b_lever.json").write_text(json.dumps(out, indent=1, default=float))
    (HERE / "p2b_lever_tables.md").write_text(tables(out))
    print(tables(out))


def tables(out):
    f = lambda d, k="median": "—" if d is None else f"{d[k]:.3f}"  # noqa: E731
    iq = lambda d: "—" if d is None else f"{d['median']:.3f} [{d['q25']:.3f}–{d['q75']:.3f}]"  # noqa: E731
    L = ["EXPLORATORY (non-registered seeds 2,953,000-2,953,019; never to be registered). Median [IQR]; paired "
         "differences vs std on the same seed: median [IQR] (mean, sd), seeds up/down.",
         f"G (glob) = {out['G_explore']}; Gb (globb) = {out['Gb']}. Consistency with committed runs: "
         f"{out['consistency_with_committed_runs']}"]
    for qn in ("rho2", "acc_shuffled", "acc_reversed"):
        L += [f"\n#### {qn}: value and paired difference vs std\n",
              "| point | " + " | ".join(out["conds"]) + " | " + " | ".join(f"Δ {c}" for c in out["conds"] if c != "std") + " |",
              "|---" * (2 * len(out["conds"])) + "|"]
        for p in out["conds"]["std"]["points"]:
            row = [iq(out["conds"][c]["points"][p][qn]) + f" (n={out['conds'][c]['points'][p]['n']})" for c in out["conds"]]
            for c in out["conds"]:
                if c == "std":
                    continue
                e = out["conds"][c]["points"][p]; d = e.get("d_" + qn)
                row.append("—" if d is None else f"{iq(d)} ({d['mean']:+.3f}, {d['sd'] or 0:.3f}); "
                           f"{e['n_up_' + qn]}/{e['n_down_' + qn]}")
            L.append(f"| {p} | " + " | ".join(row) + " |")
    L += ["\n#### s and steps at matched loss (BCE first passage)\n", "| BCE level | " + " | ".join(
        f"{c}: s / steps" for c in out["conds"]) + " |", "|---" * (1 + len(out["conds"])) + "|"]
    for lv in LOSSES:
        L.append(f"| {lv} | " + " | ".join(
            f"{iq(out['conds'][c]['s_at_loss'][lv])} / {f(out['conds'][c]['steps_to_loss'][lv])}" for c in out["conds"]) + " |")
    L += ["\n#### steps to matched scale (× std, per seed)\n", "| scale | " + " | ".join(out["conds"]) + " |",
          "|---" * (1 + len(out["conds"])) + "|"]
    for k in SCALES:
        L.append(f"| {k} | " + " | ".join(f"{f(out['conds'][c]['steps_to'][k])} (×{f(out['conds'][c]['steps_to_over_std'][k])})"
                                           for c in out["conds"]) + " |")
    L += ["\n#### lasting onset (rho2 >= q at every step from t_on to the end), by end of run\n",
          "| cond | end | n lasting | n with s_on > s* | n in [1, 1.25] s* | n from init | s_on/s* median [IQR] |",
          "|---|---|---|---|---|---|---|"]
    for c, d in out["conds"].items():
        for end, o in d["onset"].items():
            L.append(f"| {c} | {end} | {o['n_lasting']} | {o['n_above_s*']} | {o['n_in_s*_band']} | {o['n_from_init']} | "
                     f"{iq(o['s_over_s_star'])} |")
    L.append("\nmax relative overshoot at matched scale: " + ", ".join(
        f"{c} {d['max_overshoot_scale']:.4f}" for c, d in out["conds"].items()))
    L.append("down-passages of q after 3 s*: " + ", ".join(f"{c} {iq(d['n_down_after_3s*'])}" for c, d in out["conds"].items()))
    L.append("seconds per run: " + ", ".join(f"{c} {f(d['seconds'])}" for c, d in out["conds"].items())
             + "; max RSS GB: " + ", ".join(f"{c} {d['max_rss_gb']:.2f}" for c, d in out["conds"].items()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "summarise":
        summarise()
