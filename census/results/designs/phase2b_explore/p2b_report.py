"""EXPLORATORY: headline tables (markdown) from p2b_summary.json and the per-run JSONs.  python p2b_report.py > p2b_tables.md"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2b_explore as E  # noqa: E402

S = json.loads((HERE / "p2b_summary.json").read_text())
seeds = S["seeds_complete_in_every_condition"]
runs = {c: {s: json.loads((E.RUNS / f"{c}_{s}.json").read_text()) for s in seeds} for c in S["conditions"]}


def mi(d, f="{:.2f}"):
    if d is None:
        return "—"
    return (f + " [" + f + "–" + f + "]").format(d["median"], d["q25"], d["q75"])


def onset_last(r):
    e = r["events"]["last_up"]
    if r["rho2_end"] < E.Q:
        return None
    if e is None:                          # ρ₂ ≥ q throughout (from init)
        return 0.0
    return e["s"]


out = [f"Seeds: {len(seeds)} ({seeds[0]}–{seeds[-1]}), every condition. Cells: median [IQR] across seeds.\n"]
out.append("#### A. Slab use ρ₂ at matched output scale (first step with s ≥ the scale); n reached if < all\n")
out.append("| condition | 0.5 s* | s* | 1.25 s* | s_F | 2 s* | 3 s* | ρ₂ ≥ q at 3 s* |")
out.append("|---|---|---|---|---|---|---|---|")
for c, C in S["conditions"].items():
    cells = []
    for pt in list(E.SCALES):
        d = C["points"][pt]
        x = mi(d["rho2"])
        if d["n_reached"] < C["n_runs"]:
            x += f" (n={d['n_reached']})"
        cells.append(x)
    d3 = C["points"]["3s*"]
    fr = "—" if d3["frac_rho2_ge_q"] is None else f"{round(d3['frac_rho2_ge_q'] * d3['n_reached'])}/{d3['n_reached']}"
    out.append(f"| {c} | " + " | ".join(cells) + f" | {fr} |")

for k, title in (("acc_shuffled", "B. Accuracy, x₁ shuffled (expectation over permutations; pure-x₁ rule 0.50, pure slab 1.00)"),
                 ("acc_reversed", "C. Accuracy, x₁ → −x₁ (pure-x₁ rule 0.10, pure slab 1.00)"),
                 ("gplus", "C2. Gap G₊ = ½(min_{y=1} φ − max_{y=0} φ) of the unit-ℓ₁ function (> 0 iff some bias separates all 800 points)")):
    out.append(f"\n#### {title}\n")
    out.append("| condition | s* | 1.25 s* | s_F | 2 s* | 3 s* | at matched step |")
    out.append("|---|---|---|---|---|---|---|")
    for c, C in S["conditions"].items():
        cells = []
        for pt in ("s*", "1.25s*", "s_F", "2s*", "3s*", "match_step"):
            d = C["points"][pt]
            x = mi(d[k], "{:.3f}" if k == "gplus" else "{:.2f}") if d[k] else "—"
            if d["n_reached"] < C["n_runs"] and d["n_reached"]:
                x += f" (n={d['n_reached']})"
            cells.append(x)
        out.append(f"| {c} | " + " | ".join(cells) + " |")

out.append("\n#### D. Cost (steps) and the matched step budget (= the same seed's adam_r1 steps to 3 s*)\n")
out.append("| condition | steps to s* | steps to s_F | steps to 3 s* | ×adam_r1 to 3 s* | at matched step: s | ρ₂ | ρ₂ ≥ q | output sign changes |")
out.append("|---|---|---|---|---|---|---|---|---|")
for c, C in S["conditions"].items():
    P = C["points"]
    ms = P["match_step"]
    fr = "—" if ms["frac_rho2_ge_q"] is None else f"{round(ms['frac_rho2_ge_q'] * ms['n_reached'])}/{ms['n_reached']}"
    n3 = P["3s*"]["n_reached"]
    out.append(f"| {c} | {mi(P['s*']['steps'], '{:.0f}')} | {mi(P['s_F']['steps'], '{:.0f}')} | "
               f"{mi(P['3s*']['steps'], '{:.0f}')}{'' if n3 == C['n_runs'] else f' (n={n3})'} | "
               f"{mi(P['3s*']['steps_over_standard'], '{:.1f}')} | {mi(ms.get('s'))} | {mi(ms['rho2'])} | {fr} | "
               f"{mi(C['n_output_sign_changes'], '{:.0f}')} |")

out.append("\n#### E. Onset of persistent slab use: s at the LAST upward passage of q among runs ending with ρ₂ ≥ q "
           "(0 if ρ₂ ≥ q from init on); first upward passage for reference\n")
out.append("| condition | runs ending ρ₂ ≥ q | last up: s/s* median [IQR] | last up: below s* / [1, 1.25]s* / (1.25s*, s_F) / [1, 1.25]s_F / above | first up: s/s* median [IQR] |")
out.append("|---|---|---|---|---|---|")
for c, C in S["conditions"].items():
    xs = [onset_last(runs[c][s]) for s in seeds]
    xs = [x for x in xs if x is not None]
    cls = [E.onset_class(x) for x in xs]
    order = ["below s*", "at s* [1, 1.25]s*", "between 1.25s* and s_F", "at s_F [1, 1.25]s_F", "above 1.25s_F"]
    q = np.percentile([x / E.S_STAR for x in xs], [25, 50, 75]) if xs else None
    lu = "—" if q is None else f"{q[1]:.2f} [{q[0]:.2f}–{q[2]:.2f}]"
    out.append(f"| {c} | {len(xs)}/{len(seeds)} | {lu} | {' / '.join(str(cls.count(o)) for o in order)} | "
               f"{mi(C['onset']['first_up']['s_over_s_star'])} |")

out.append("\n#### F. Paired differences vs adam_r1 (same seed) at matched scale: median [IQR]; seeds above/below\n")
out.append("| condition | Δρ₂ at s_F | Δρ₂ at 3 s* | Δacc shuffled at 3 s* | Δacc reversed at 3 s* | Δρ₂ at matched step |")
out.append("|---|---|---|---|---|---|")
for c, C in S["conditions"].items():
    if c == "adam_r1":
        continue
    P = C["points"]

    def pd(pt, k):
        d = P[pt]
        return f"{mi(d[f'paired_diff_vs_adam_r1_{k}'], '{:+.3f}')}; {d[f'n_above_adam_r1_{k}']}/{d[f'n_below_adam_r1_{k}']}"
    out.append(f"| {c} | {pd('s_F', 'rho2')} | {pd('3s*', 'rho2')} | {pd('3s*', 'acc_shuffled')} | "
               f"{pd('3s*', 'acc_reversed')} | {pd('match_step', 'rho2')} |")

wn = runs.get("gd_wn_r9", {})
if wn:
    stuck = [(s, r["s_end"], r["rho2_end"], r["traj"]["s"][-40]) for s, r in wn.items() if r["status"] == "budget"]
    out.append(f"\ngd_wn_r9 runs stopped by the 1M-step budget: {len(stuck)}/{len(wn)}; their s_end "
               f"{', '.join(f'{x[1]:.3f}' for x in stuck)}; ρ₂_end max {max([x[2] for x in stuck] or [float('nan')]):.2e}; "
               f"s at the 40th-last log-spaced record (≈ step {wn[stuck[0][0]]['traj']['t'][-40] if stuck else '—'}): "
               f"{', '.join(f'{x[3]:.3f}' for x in stuck)}")
print("\n".join(out))
