"""EXPLORATORY summary (Track LN; NOT a registration): the two gates written in README.md before training, per-arm and
per-level tables, the arm-1 rand trajectory, the DESCRIPTIVE power of R and N from the 8 exploration seeds (Track L's
L_power.py method), and Track L's registered arm-level numbers (results/trackL/scores.json) side by side.
Reads runs.jsonl (LN_explore.py).  Writes LN_summary.json and LN_tables.md.
Usage (from census/): python results/designs/trackLN_explore/LN_summary.py"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import trackL as TL  # noqa: E402  (read-only)

ARMS = ("std", "out16", "glob16")
LEVELS = TL.LEVELS                    # (("bce_0.6", 0.6), ("bce_0.3", 0.3), ("bce_0.03", 0.03))
LK = [k for k, _ in LEVELS]
P = 0.8
H = -(P * math.log(P) + (1 - P) * math.log(1 - P))
NSIM, NBOOT, DELTA = 300, 1000, 0.02


def first(rec, ell):
    for i, v in enumerate(rec["loss"]):
        if v <= ell:
            return {k: rec[k][i] for k in rec}
    return None


def q(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if not len(a):
        return None
    return {"n": int(len(a)), "median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}


def load():
    rows = TL._rows(HERE / "runs.jsonl")
    by = {(r["arm"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows if all((a, r["seed"]) in by for a in ARMS)})
    return by, seeds


def gate2(R):
    """Track L's L2 gate rule, verbatim, on the arm-1 runs."""
    a = [first(r["rec"], H + 0.10) for r in R]
    b = [first(r["rec"], 0.02) for r in R]
    ra = np.array([x["rand"] if x else np.nan for x in a])
    rb = np.array([x["rand"] if x else np.nan for x in b])
    end = np.array([r["end"]["rand"] for r in R])
    d = rb - ra
    med_a, med_d, up = float(np.nanmedian(ra)), float(np.nanmedian(d)), float(np.nanmean(d > 0))
    return {"ell_A_threshold": H + 0.10, "ell_B_threshold": 0.02,
            "steps_at_A": [x["t"] if x else None for x in a], "steps_at_B": [x["t"] if x else None for x in b],
            "rand_at_A": ra.tolist(), "rand_at_B": rb.tolist(), "increase_A_to_B": d.tolist(), "rand_end": end.tolist(),
            "median_rand_A": med_a, "median_increase": med_d, "frac_increasing": up, "median_rand_end": float(np.median(end)),
            "PASS": bool(med_a <= 0.60 and med_d >= 0.10 and up >= 0.80),
            "never_uses_CIFAR": bool(np.median(end) <= 0.60), "uses_CIFAR_from_start": bool(med_a >= 0.70)}


def trajectory(R, ells=(0.69, 0.65, H + 0.10, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002)):
    out = []
    for e in ells:
        pts = [first(r["rec"], e) for r in R]
        out.append({"bce_le": round(e, 4), "n": sum(p is not None for p in pts),
                    "median_t": q([p["t"] for p in pts if p])["median"] if any(pts) else None,
                    "median_rand": q([p["rand"] for p in pts if p])["median"] if any(pts) else None,
                    "rand": [round(p["rand"], 4) if p else None for p in pts]})
    return out


def lower_upper(d, rng, nboot=NBOOT):
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    med = np.median(d[idx], axis=1)
    return np.quantile(med, 0.025), np.quantile(med, 0.975)


def power(diffs, n, crit, rng):
    """Track L's L_power.power: seeds resampled jointly across levels; R all lower ends > 0, N all upper ends < delta."""
    ok = 0
    for _ in range(NSIM):
        sel = rng.integers(0, len(diffs[0]), size=n)
        good = True
        for d in diffs:
            lo, hi = lower_upper(np.asarray(d)[sel], rng)
            if (crit == "R" and not lo > 0) or (crit == "N" and not hi < DELTA):
                good = False
                break
        ok += good
    return ok / NSIM


def main():
    by, seeds = load()
    rows = {a: [by[(a, s)] for s in seeds] for a in ARMS}
    S = {"label": "EXPLORATORY (Track LN; 8 exploration seeds; NOT a registration)", "seeds": seeds,
         "runs": {f"{r['seed']}|{r['arm']}": {k: r[k] for k in ("steps", "stop", "finite", "secs", "max_rss_gb")}
                  for a in ARMS for r in rows[a]},
         "arms": {}}
    for a in ARMS:
        R = rows[a]
        d = {"convergence": {"steps": q([r["steps"] for r in R]), "stop": {s: sum(r["stop"] == s for r in R)
                                                                           for s in ("loss", "cap", "nonfinite")},
                             "secs": q([r["secs"] for r in R])},
             "levels": {}}
        for k in LK + ["end"]:
            get = (lambda r: r["end"]) if k == "end" else (lambda r, k=k: r["at"].get(k))
            c = {"per_seed_t": [(get(r) or {}).get("t") for r in R],
                 **{m: q([(get(r) or {}).get(m) for r in R]) for m in ("t", "rand", "rev", "orig", "acc_flip", "outw", "hid", "absf")}}
            c["bound_16v_plus_b_median"] = None
            if a != "std":
                ref = rows["std"]
                pairs = [(get(r), get(b)) for r, b in zip(R, ref)]
                cost = [x["t"] / y["t"] if x and y and y["t"] > 0 else None for x, y in pairs]
                c["step_cost"] = {**(q(cost) or {}), "per_seed": cost}
                for m in ("rand", "rev", "orig"):
                    dd = [x[m] - y[m] if x and y else None for x, y in pairs]
                    v = [x for x in dd if x is not None]
                    c[f"d_{m}"] = {**(q(v) or {}), "n_up": int(sum(x > 0 for x in v)),
                                   "n_down": int(sum(x < 0 for x in v)), "per_seed": dd}
                if k != "end":
                    D = TL.paired(R, ref, q="rand", levels=(k,))
                    lo, hi, med = TL.boot_ci_median(D)
                    c["d_rand_boot95_DESCRIPTIVE"] = [float(lo[0]), float(hi[0])]
            d["levels"][k] = c
        S["arms"][a] = d
    # gate (1), proposed rule and the alternatives shown for comparison
    g1 = {}
    for k in LK:
        c = S["arms"]["out16"]["levels"][k]["step_cost"]
        ps = [x for x in c["per_seed"] if x is not None]
        g1[k] = {"median": c["median"], "n": c["n"], "frac_ge_2": float(np.mean([x >= 2.0 for x in ps])),
                 "n_missing": len(c["per_seed"]) - len(ps)}
    S["gate1_PROPOSED"] = {"rule": "median arm-2/arm-1 step cost >= 2.0 at each of 0.6, 0.3, 0.03", "cells": g1,
                           "holds": bool(all(g1[k]["median"] >= 2.0 for k in LK)),
                           "alt_per_seed_80pct": bool(all(g1[k]["frac_ge_2"] >= 0.8 for k in LK)),
                           "alt_0.3_and_0.03_only": bool(all(g1[k]["median"] >= 2.0 for k in LK[1:]))}
    S["gate2_L2_rule_arm1"] = gate2(rows["std"])
    S["arm1_rand_trajectory"] = trajectory(rows["std"])
    # DESCRIPTIVE power (from 8 exploration seeds)
    rng = np.random.default_rng(20261009)
    pw = {}
    for a, crit in (("out16", "R"), ("glob16", "N")):
        diffs = [[x for x in S["arms"][a]["levels"][k]["d_rand"]["per_seed"]] for k in LK]
        if any(x is None for dd in diffs for x in dd):
            pw[a] = "unequal reach"
            continue
        res = {"criterion": crit, "n_explore": len(diffs[0]), "medians": [float(np.median(dd)) for dd in diffs],
               "sd": [float(np.std(dd, ddof=1)) for dd in diffs]}
        for n in (20, 40, 80):
            res[f"P({crit}) n={n}"] = power(diffs, n, crit, rng)
        pw[a] = res
    S["power_DESCRIPTIVE_from_8_exploration_seeds"] = pw
    # Track L's registered arm-level numbers
    L = json.loads((ROOT / "results" / "trackL" / "scores.json").read_text())
    side = {}
    for a in ARMS:
        LD = L["DESCRIPTIVE"]["arms"][a]
        side[a] = {"steps_to_median": {k: LD["steps_to"][k]["median"] for k in LK},
                   "rand_median": {k: LD["values"][k]["rand"]["median"] for k in LK + ["end"]},
                   "outw_median": {k: LD["values"][k]["outw"]["median"] for k in LK + ["end"]}}
        if a != "std":
            side[a]["step_cost_median"] = {k: LD["step_cost_vs_arm1"][k]["median"] for k in LK}
            side[a]["step_cost_max"] = {k: LD["step_cost_vs_arm1"][k]["max"] for k in LK}
            side[a]["d_rand"] = {k: {x: L["arms"][a]["cells_rand"][k][x] for x in ("median", "lo", "hi", "n_up", "n_down")}
                                 for k in LK}
    S["trackL_registered_side_by_side"] = {"n_seeds": L["n_seeds"], "verdicts": L["verdicts"], "arms": side}
    (HERE / "LN_summary.json").write_text(json.dumps(TL._jsonable(S), indent=1))
    tables(S)


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def tables(S):
    o = ["# Track LN exploration tables (EXPLORATORY; 8 seeds 2,995,000–007; generated by LN_summary.py)", ""]
    o += ["## Per arm and level (medians over seeds; [q25, q75])", "",
          "| arm | level | steps | step cost vs arm 1 | rand | rev | orig | Δrand vs arm 1 (up/down) | ‖v‖₂ | mean abs f |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for k in LK + ["end"]:
            c = S["arms"][a]["levels"][k]
            sc = c.get("step_cost")
            dr = c.get("d_rand")
            o.append(f"| {a} | {k} | {f(c['t']['median'], 0)} [{f(c['t']['q25'], 0)}, {f(c['t']['q75'], 0)}] | "
                     + (f"×{f(sc['median'], 2)} [{f(sc['q25'], 2)}, {f(sc['q75'], 2)}]" if sc else "–") + " | "
                     + f"{f(c['rand']['median'])} | {f(c['rev']['median'])} | {f(c['orig']['median'])} | "
                     + (f"{dr['median']:+.4f} ({dr['n_up']}/{dr['n_down']})" if dr else "–")
                     + f" | {f(c['outw']['median'], 2)} | {f(c['absf']['median'], 2)} |")
    o += ["", "## Per seed: step cost (arm / arm 1) and Δrand at each level", ""]
    for a in ("out16", "glob16"):
        for k in LK:
            c = S["arms"][a]["levels"][k]
            o.append(f"- {a} {k}: cost " + ", ".join(f(x, 2) for x in c["step_cost"]["per_seed"])
                     + "; Δrand " + ", ".join("–" if x is None else f"{x:+.4f}" for x in c["d_rand"]["per_seed"])
                     + f"; bootstrap 95% of median Δrand (DESCRIPTIVE) [{c['d_rand_boot95_DESCRIPTIVE'][0]:+.4f}, "
                       f"{c['d_rand_boot95_DESCRIPTIVE'][1]:+.4f}]")
    g1 = S["gate1_PROPOSED"]
    o += ["", "## Gate (1), PROPOSED rule: median arm-2 step cost ≥ 2.0 at each level", "",
          "| level | median cost | n | fraction of seeds ≥ 2.0 |", "|---|---|---|---|"]
    for k in LK:
        c = g1["cells"][k]
        o.append(f"| {k} | ×{f(c['median'], 3)} | {c['n']} | {c['frac_ge_2']:.2f} |")
    o += ["", f"Holds (proposed rule): **{g1['holds']}**; per-seed 80% alternative: {g1['alt_per_seed_80pct']}; "
              f"0.3 and 0.03 only: {g1['alt_0.3_and_0.03_only']}."]
    g2 = S["gate2_L2_rule_arm1"]
    o += ["", "## Gate (2): Track L's L2 rule on arm 1", "",
          f"ℓ_A threshold BCE ≤ {g2['ell_A_threshold']:.4f}, ℓ_B BCE ≤ 0.02. Median rand at ℓ_A {g2['median_rand_A']:.4f}; "
          f"median increase ℓ_A→ℓ_B {g2['median_increase']:+.4f}; seeds increasing {g2['frac_increasing']:.2f}; "
          f"median rand at end {g2['median_rand_end']:.4f}. **PASS: {g2['PASS']}**; never uses CIFAR: "
          f"{g2['never_uses_CIFAR']}; uses CIFAR from the start: {g2['uses_CIFAR_from_start']}.", "",
          "- steps at ℓ_A: " + ", ".join(str(x) for x in g2["steps_at_A"]),
          "- steps at ℓ_B: " + ", ".join(str(x) for x in g2["steps_at_B"]),
          "- rand at ℓ_A: " + ", ".join(f(x, 4) for x in g2["rand_at_A"]),
          "- rand at ℓ_B: " + ", ".join(f(x, 4) for x in g2["rand_at_B"]),
          "- rand at end: " + ", ".join(f(x, 4) for x in g2["rand_end"])]
    o += ["", "## Arm-1 rand trajectory (first check at or below each train BCE; medians over seeds)", "",
          "| train BCE ≤ | n | median step | median rand |", "|---|---|---|---|"]
    for t in S["arm1_rand_trajectory"]:
        o.append(f"| {t['bce_le']} | {t['n']} | {f(t['median_t'], 0)} | {f(t['median_rand'], 4)} |")
    o += ["", "## Power (DESCRIPTIVE, from 8 exploration seeds; Track L's L_power.py method, δ = 0.02)", ""]
    for a, r in S["power_DESCRIPTIVE_from_8_exploration_seeds"].items():
        if isinstance(r, str):
            o.append(f"- {a}: {r}")
            continue
        c = r["criterion"]
        o.append(f"- {a} ({c}): medians " + ", ".join(f"{x:+.4f}" for x in r["medians"]) + "; sd "
                 + ", ".join(f"{x:.4f}" for x in r["sd"]) + "; "
                 + ", ".join(f"P({c}) n={n} {r[f'P({c}) n={n}']:.2f}" for n in (20, 40, 80)))
    L = S["trackL_registered_side_by_side"]
    o += ["", f"## Side by side with Track L (registered, {L['n_seeds']} seeds, results/trackL/scores.json; no claim)", "",
          "| arm | level | steps LN / L | step cost LN / L (L max) | rand LN / L | Δrand LN / L | ‖v‖₂ LN / L |",
          "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for k in LK:
            c, l = S["arms"][a]["levels"][k], L["arms"][a]
            sc = (f"×{f(c['step_cost']['median'], 2)} / ×{f(l['step_cost_median'][k], 2)} (×{f(l['step_cost_max'][k], 2)})"
                  if a != "std" else "–")
            dr = (f"{c['d_rand']['median']:+.4f} / {l['d_rand'][k]['median']:+.4f}" if a != "std" else "–")
            o.append(f"| {a} | {k} | {f(c['t']['median'], 0)} / {f(l['steps_to_median'][k], 0)} | {sc} | "
                     f"{f(c['rand']['median'])} / {f(l['rand_median'][k])} | {dr} | "
                     f"{f(c['outw']['median'], 2)} / {f(l['outw_median'][k], 2)} |")
    o += ["", "Runs: " + "; ".join(f"{k} {v['steps']} steps {v['secs']:.0f} s {v['max_rss_gb']} GB {v['stop']}"
                                   for k, v in S["runs"].items())]
    (HERE / "LN_tables.md").write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
