"""EXPLORATORY summary (Track L, L2): gate statistics on std runs and paired arm-vs-std differences at matched train
BCE (first passage at the checks, no interpolation).  Reads JSONL files written by L_explore.py.
Usage (from census/): python results/designs/trackL_explore/L_summary.py FILE.jsonl [FILE2.jsonl ...] [--levels a,b,c]"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def H(p):
    return 0.0 if p >= 1 else -(p * math.log(p) + (1 - p) * math.log(1 - p))


def load(files):
    rows = []
    for f in files:
        p = Path(f) if Path(f).is_absolute() else HERE / f
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def at(rec, ell):
    """first check with loss <= ell: dict of values, or None"""
    for i, v in enumerate(rec["loss"]):
        if v <= ell:
            return {k: rec[k][i] for k in rec}
    return None


def gate(rows):
    out = {}
    by = defaultdict(list)
    for r in rows:
        if r["arm"] == "std":
            by[(r["p"], r["lr"], r["width"], r.get("opt", "sgd") + ("" if r.get("act", "relu") == "relu" else "-" + r["act"]) + ("" if r.get("vscale", 1.0) == 1.0 else f"-v{r['vscale']}"))].append(r)
    for (p, lr, w, o), rs in sorted(by.items()):
        a = [at(r["rec"], H(p) + 0.10) for r in rs]
        b = [at(r["rec"], 0.02) for r in rs]
        ra = np.array([x["rand"] if x else np.nan for x in a])
        rb = np.array([x["rand"] if x else np.nan for x in b])
        end = np.array([r["rec"]["rand"][-1] for r in rs])
        d = rb - ra
        med_a, med_d = float(np.nanmedian(ra)), float(np.nanmedian(d))
        up = float(np.nanmean(d > 0))
        verdict = ("PASS" if (med_a <= 0.60 and med_d >= 0.10 and up >= 0.80) else
                   "never-uses-CIFAR" if np.median(end) <= 0.60 else
                   "uses-CIFAR-from-start" if med_a >= 0.70 else "FAIL-other")
        out[f"p={p} lr={lr} w={w} {o}"] = {
            "n": len(rs), "ell_A": round(H(p) + 0.10, 4), "rand_at_A": np.round(ra, 3).tolist(),
            "rand_at_B": np.round(rb, 3).tolist(), "rand_end": np.round(end, 3).tolist(),
            "median_rand_A": med_a, "median_increase_A_to_B": med_d, "frac_up": up,
            "steps_to_B": [x["t"] if x else None for x in b], "steps_total": [r["steps"] for r in rs],
            "secs": [r["secs"] for r in rs], "verdict": verdict}
    return out


def paired(rows, levels, qty=("rand", "rev", "orig")):
    K = lambda r: (r["p"], r["lr"], r["width"], r.get("opt", "sgd") + ("" if r.get("act", "relu") == "relu" else "-" + r["act"]) + ("" if r.get("vscale", 1.0) == 1.0 else f"-v{r['vscale']}"))
    std = {K(r) + (r["seed"],): r for r in rows if r["arm"] == "std"}
    by = defaultdict(list)
    for r in rows:
        if r["arm"] != "std" and K(r) + (r["seed"],) in std:
            by[K(r) + (r["arm"],)].append(r)
    out = {}
    for (p, lr, w, o, arm), rs in sorted(by.items()):
        cell = {}
        for ell in levels:
            for q in qty:
                ds, cost = [], []
                for r in rs:
                    s = std[(p, lr, w, o, r["seed"])]
                    x, y = at(r["rec"], ell), at(s["rec"], ell)
                    if x and y:
                        ds.append(x[q] - y[q])
                        if q == qty[0]:
                            cost.append(x["t"] / max(y["t"], 1))
                ds = np.array(ds)
                cell[f"{ell}|{q}"] = {"n": len(ds), "median": round(float(np.median(ds)), 4) if len(ds) else None,
                                      "up": int((ds > 0).sum()), "down": int((ds < 0).sum()),
                                      "vals": np.round(ds, 4).tolist()}
                if q == qty[0]:
                    cell[f"{ell}|cost"] = round(float(np.median(cost)), 3) if cost else None
        out[f"p={p} lr={lr} w={w} {o} {arm}"] = cell
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    levels = [0.3, 0.1, 0.03]
    for a in sys.argv[1:]:
        if a.startswith("--levels"):
            levels = [float(x) for x in a.split("=")[1].split(",")]
    rows = load(args)
    g = gate(rows)
    for k, v in g.items():
        print("GATE", k, {kk: vv for kk, vv in v.items() if kk not in ("rand_at_A", "rand_at_B", "rand_end")})
        print("   rand@A", v["rand_at_A"], "\n   rand@B", v["rand_at_B"], "\n   end", v["rand_end"])
    pr = paired(rows, levels)
    for k, cell in pr.items():
        print("PAIRED", k)
        for ell in levels:
            print("   ", ell, "cost x%s" % cell[f"{ell}|cost"],
                  "  ".join(f"{q}: {cell[f'{ell}|{q}']['median']} ({cell[f'{ell}|{q}']['up']}/{cell[f'{ell}|{q}']['down']}, n={cell[f'{ell}|{q}']['n']})"
                            for q in ("rand", "rev", "orig")))
