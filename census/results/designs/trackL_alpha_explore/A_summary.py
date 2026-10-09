"""EXPLORATORY summary (Track L output-multiplier follow-up, MOTIVATED BY THE TRACK L NULL; NOT a registration).
Reads JSONL files written by A_explore.py.  Writes A_tables.md and A_summary.json.
  * reference = alpha 1, lr 0.01, std (identical in the two modes: lr / 1^2 = lr) on the same seed;
  * paired differences (arm - reference, same seed) of MNIST-randomised (primary), reversed (secondary) and original
    (descriptive) test accuracy at the matched train-BCE levels {0.6, 0.3, 0.03} (first check at or below; no
    interpolation); median, seeds up/down, n reaching the level in both runs; step cost = median per-seed ratio of
    steps to the level (arm / reference);
  * the analogue of N: glob16 vs std at the SAME alpha and mode on the same seed;
  * L2 gate statistics (rule in ../trackL_explore/README.md: l_A = first check <= H(0.8) + 0.10, l_B = first check
    <= 0.02; PASS if median rand at l_A <= 0.60 and median paired increase l_A -> l_B >= 0.10 with >= 80% up) per cell;
  * end-of-training values (descriptive; a run ends at train BCE <= 0.002, its step cap, or a non-finite loss);
  * power of R (lower end of the 95% percentile bootstrap interval of the median paired difference > 0 at every level)
    at n = 20/40/80 by resampling the per-seed paired differences (../trackL_explore/L_power.py method: 300
    simulations, inner bootstrap 1,000, numpy default_rng(20261009)); cells where some seed misses a level are skipped.
Usage (from census/): python results/designs/trackL_alpha_explore/A_summary.py explore_pilot.jsonl explore_main.jsonl"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trackL_explore"))
import L_power  # noqa: E402

LEVELS = (0.6, 0.3, 0.03)
P = 0.8
H = -(P * math.log(P) + (1 - P) * math.log(1 - P))


def load(files):
    """rows by key; a key in a later file replaces the same key from an earlier file (pass the pilot first: the
    main batch re-ran seed 2,994,000 with longer caps; the shared prefix of the records is identical)"""
    rows = {}
    for f in files:
        p = Path(f) if Path(f).is_absolute() else HERE / f
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows[r["key"]] = r
    return list(rows.values())


def at(rec, ell):
    for i, v in enumerate(rec["loss"]):
        if v <= ell:
            return {k: rec[k][i] for k in rec}
    return None


def cellname(r):
    return (r["mode"] if r["alpha"] != 1.0 else "fix", r["arm"], r["alpha"])


def paired(rs, refs, qty=("rand", "rev", "orig")):
    cell = {}
    for ell in LEVELS:
        for q in qty:
            ds, cost, miss = [], [], 0
            for r in rs:
                s = refs.get(r["seed"])
                if s is None:
                    continue
                x, y = at(r["rec"], ell), at(s["rec"], ell)
                if x and y:
                    ds.append(x[q] - y[q])
                    cost.append(x["t"] / max(y["t"], 1))
                else:
                    miss += 1
            ds = np.array(ds)
            cell[f"{ell}|{q}"] = {"n": len(ds), "miss": miss,
                                  "median": round(float(np.median(ds)), 4) if len(ds) else None,
                                  "up": int((ds > 0).sum()), "down": int((ds < 0).sum()),
                                  "vals": np.round(ds, 4).tolist()}
            cell[f"{ell}|cost"] = round(float(np.median(cost)), 3) if cost else None
    return cell


def gate(rs):
    a = [at(r["rec"], H + 0.10) for r in rs]
    b = [at(r["rec"], 0.02) for r in rs]
    ra = np.array([x["rand"] if x else np.nan for x in a])
    rb = np.array([x["rand"] if x else np.nan for x in b])
    d = rb - ra
    ok = ~np.isnan(d)
    med_a = float(np.nanmedian(ra)) if (~np.isnan(ra)).any() else float("nan")
    med_d = float(np.median(d[ok])) if ok.any() else float("nan")
    up = float(np.mean(d[ok] > 0)) if ok.any() else float("nan")
    return {"n": len(rs), "n_reach_A": int((~np.isnan(ra)).sum()), "n_reach_B": int((~np.isnan(rb)).sum()),
            "median_rand_A": med_a, "median_increase": med_d, "frac_up": up,
            "PASS": bool(med_a <= 0.60 and med_d >= 0.10 and up >= 0.80 and ok.sum() == len(rs))}


def fmt(c, ell, q):
    v = c[f"{ell}|{q}"]
    if v["median"] is None:
        return f"— (0 of {v['n'] + v['miss']})"
    s = f"{v['median']:+.4f} ({v['up']}/{v['down']})"
    return s + (f" [n {v['n']}, {v['miss']} unreached]" if v["miss"] else "")


def main(files):
    rows = [r for r in load(files) if r["seed"] >= 2_994_000]
    by = defaultdict(list)
    for r in rows:
        by[cellname(r)].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["seed"])
    ref = {r["seed"]: r for r in by.get(("fix", "std", 1.0), [])}
    order = sorted(by, key=lambda k: (k[1] != "std", k[0], -k[2]))
    out = {"cells": {}, "vs_alpha1": {}, "glob_vs_std": {}, "power": {}}
    L = ["# Track L output-multiplier exploration tables (EXPLORATORY; motivated by the Track L null; NOT a registration)",
         "", "p = 0.8; f = alpha * g; SGD momentum 0.9, batch 128, base lr 0.01; `fix` = lr 0.01 for every alpha;",
         "`fs` = lr 0.01 / alpha^2 (function-space step at init matched); `glob16` = the mode's lr / 16.",
         "Matched train-BCE levels {0.6, 0.3, 0.03} (first check at or below; no interpolation). Δ = arm − reference on",
         "the same seed: median (seeds up/down). Reference: alpha 1, lr 0.01, std (the same run in both modes).", "",
         "## Per cell: MNIST-randomised accuracy at each level (median over seeds reaching it), end of training, gate", "",
         "| mode, arm, alpha | lr | n | 0.6 | 0.3 | 0.03 | mean abs f / ‖v‖ / ‖W2‖ at 0.03 | end rand | end orig | end train BCE | steps (median) | stop reached | L2 gate: rand@ℓ_A → Δ to ℓ_B (up) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        rs = by[k]
        vals = []
        for ell in LEVELS:
            xs = [at(r["rec"], ell) for r in rs]
            got = [x["rand"] for x in xs if x]
            vals.append(f"{np.median(got):.3f} ({len(got)})" if got else f"— (0)")
        end = {q: float(np.median([r["rec"][q][-1] for r in rs])) for q in ("rand", "orig", "loss")}
        x3 = [at(r["rec"], 0.03) for r in rs]
        x3 = [x for x in x3 if x]
        sc = (f"{np.median([x['absf'] for x in x3]):.2f} / {np.median([x['outw'] for x in x3]):.2f} / "
              f"{np.median([x['hid'] for x in x3]):.2f}") if x3 else "—"
        vals.append(sc)
        g = gate(rs)
        out["cells"]["|".join(map(str, k))] = {"n": len(rs), "lr": rs[0]["lr"], "end": end, "gate": g,
                                               "steps": [r["steps"] for r in rs],
                                               "finite": [r["finite"] for r in rs],
                                               "reached_stop": [r["reached_stop"] for r in rs]}
        gs = (f"{g['median_rand_A']:.3f} → {g['median_increase']:+.3f} ({g['frac_up']:.0%}){' PASS' if g['PASS'] else ''}"
              if not math.isnan(g["median_increase"]) else f"{g['median_rand_A']:.3f} → — (ℓ_B reached {g['n_reach_B']}/{g['n']})")
        L.append(f"| {k[0]}, {k[1]}, {k[2]:g} | {rs[0]['lr']:g} | {len(rs)} | " + " | ".join(vals) +
                 f" | {end['rand']:.3f} | {end['orig']:.3f} | {end['loss']:.4f} | {int(np.median([r['steps'] for r in rs]))}"
                 f" | {sum(r['reached_stop'] for r in rs)}/{len(rs)}{'' if all(r['finite'] for r in rs) else ' (non-finite: %d)' % sum(not r['finite'] for r in rs)} | {gs} |")
    L += ["", "## Paired differences vs alpha 1 (same seed)", "",
          "| mode, arm, alpha | n | Δrand 0.6 | Δrand 0.3 | Δrand 0.03 | Δrev 0.6 | Δrev 0.3 | Δrev 0.03 | Δorig 0.03 | cost 0.6 / 0.3 / 0.03 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(20261009)
    for k in order:
        if k == ("fix", "std", 1.0) or k[1] != "std":
            continue
        c = paired(by[k], ref)
        out["vs_alpha1"]["|".join(map(str, k))] = c
        L.append(f"| {k[0]}, {k[1]}, {k[2]:g} | {len(by[k])} | " + " | ".join(fmt(c, e, "rand") for e in LEVELS) + " | " +
                 " | ".join(fmt(c, e, "rev") for e in LEVELS) + f" | {fmt(c, 0.03, 'orig')} | " +
                 " / ".join(f"×{c[f'{e}|cost']}" if c[f'{e}|cost'] is not None else "—" for e in LEVELS) + " |")
        diffs = [c[f"{e}|rand"]["vals"] for e in LEVELS]
        if k[2] < 1 and all(c[f"{e}|rand"]["miss"] == 0 for e in LEVELS) and len({len(d) for d in diffs}) == 1 and len(diffs[0]) >= 3:
            out["power"]["|".join(map(str, k))] = {f"P(R) n={n}": L_power.power(diffs, n, "R", 0.0, rng) for n in (20, 40, 80)}
    # monotonicity in alpha (fix mode, std): per seed, least-squares slope of rand at the level on log10(alpha)
    fa = sorted(a for (m, arm, a) in by if m == "fix" and arm == "std" and len(by[("fix", "std", a)]) >= 3)
    L += ["", "## Monotonicity in alpha (fix mode, std arm; alphas with ≥ 3 seeds: " + ", ".join(f"{a:g}" for a in fa) + ")", "",
          "Per seed: least-squares slope of MNIST-randomised accuracy at the level on log10(alpha), over the alphas the",
          "seed reaches the level in (≥ 3). A smaller-alpha-helps account predicts a NEGATIVE slope.", "",
          "| level | n seeds | median slope per decade | seeds negative / positive | median rand by alpha (ascending alpha) |", "|---|---|---|---|---|"]
    out["monotone"] = {}
    seeds = sorted({r["seed"] for a in fa for r in by[("fix", "std", a)]})
    for ell in LEVELS:
        sl = []
        for s in seeds:
            pts = [(math.log10(a), at(r["rec"], ell)["rand"]) for a in fa for r in by[("fix", "std", a)]
                   if r["seed"] == s and at(r["rec"], ell)]
            if len(pts) >= 3:
                x, y = np.array(pts).T
                sl.append(float(np.polyfit(x, y, 1)[0]))
        meds = [np.median([at(r["rec"], ell)["rand"] for r in by[("fix", "std", a)] if at(r["rec"], ell)] or [np.nan]) for a in fa]
        out["monotone"][str(ell)] = {"slopes": sl, "median_by_alpha": dict(zip(map(str, fa), map(float, meds)))}
        sl = np.array(sl)
        L.append(f"| {ell} | {len(sl)} | {np.median(sl):+.4f} | {(sl < 0).sum()} / {(sl > 0).sum()} | " +
                 ", ".join(f"{a:g}: {m:.3f}" for a, m in zip(fa, meds)) + " |")
    L += ["", "## Analogue of N: glob16 vs std at the same alpha and mode (same seed)", "",
          "| mode, alpha | n | Δrand 0.6 | Δrand 0.3 | Δrand 0.03 | Δrev 0.6 | Δrev 0.3 | Δrev 0.03 | cost 0.6 / 0.3 / 0.03 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        if k[1] == "std":
            continue
        base = {r["seed"]: r for r in by.get((k[0], "std", k[2]), [])}
        c = paired(by[k], base)
        out["glob_vs_std"]["|".join(map(str, k))] = c
        L.append(f"| {k[0]}, {k[2]:g} ({k[1]}) | {len(by[k])} | " + " | ".join(fmt(c, e, "rand") for e in LEVELS) + " | " +
                 " | ".join(fmt(c, e, "rev") for e in LEVELS) + " | " +
                 " / ".join(f"×{c[f'{e}|cost']}" if c[f'{e}|cost'] is not None else "—" for e in LEVELS) + " |")
    L += ["", "## Power of R (smaller alpha > alpha 1 at all three levels; lower 95% bootstrap end of the median > 0)",
          "", "Resampling this exploration's per-seed paired differences (L_power.py method). Exploratory seeds only.", "",
          "| mode, arm, alpha | P(R) n = 20 | n = 40 | n = 80 |", "|---|---|---|---|"]
    for k, v in out["power"].items():
        L.append(f"| {k.replace('|', ', ')} | {v['P(R) n=20']:.2f} | {v['P(R) n=40']:.2f} | {v['P(R) n=80']:.2f} |")
    (HERE / "A_tables.md").write_text("\n".join(L) + "\n")
    (HERE / "A_summary.json").write_text(json.dumps(out, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
