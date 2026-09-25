"""Task C: the mechanism figure (width 1).  The conditional minimiser's |w₁| against held R/R_glob at a = 1.30 and 1.50,
with the class-mean optimum α* (horizontal), the placement bound a/1.4 (horizontal; placement needs |w₁| < a/1.4), R_glob
(vertical) and the free-training crossings (phase 2b, budget 32,000) overlaid.

Data (existing search machinery only):
  s ≥ 1.5   the conditional audit's retained minimiser (cond_audit_candidates.csv, status RETAINED)
  s < 1.5   the same frozen conditional search (blockB_landscape.best_conditional) at s = 0.05 ... 1.25 (added here)
  R_glob    the certified glob bracket midpoint (cond_certified_brackets.csv); x = |w₂| / w₂,glob
  crossings phase2b_checkpoints.csv crossing rows at budget 32,000 (|w₁|, |w₂| at the first placed step)
The minimiser path is a validated search result, not a certificate; R_glob is certified.

    python -m src.mechanism_w1_figure
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = RESULTS / "mechanism_w1_path.csv"
SMALL = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25)
A_VALUES = (1.30, 1.50)
ALPHA_STAR = 1.7913244


def _point(args):
    import torch
    from . import blockB_landscape as bb
    torch.set_num_threads(1)
    a, w2 = args
    x, y = bb.population_data()
    b = bb.best_conditional(a, float(w2), x, y, restarts=bb.RESTARTS)
    return {"a": a, "s": w2, "w1": np.nan if b is None else abs(b["w1"]), "gap": np.nan if b is None else b["gap"],
            "source": "frozen search (added)"}


def data(workers=1):
    with Pool(workers) as p:
        small = p.map(_point, [(a, s) for a in A_VALUES for s in SMALL])
    d = pd.read_csv(RESULTS / "cond_audit_candidates.csv")
    r = d[(d.status == "RETAINED") & d.a.round(2).isin(A_VALUES)].drop_duplicates(["a", "s"])
    big = pd.DataFrame({"a": r.a.round(2), "s": r.s, "w1": r.w1.abs(), "gap": r.gap, "source": "conditional audit"})
    out = pd.concat([pd.DataFrame(small), big]).sort_values(["a", "s"])
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv").query("kind == 'glob'")
    wg = {round(a, 2): 0.5 * (lo + hi) for a, lo, hi in zip(br.a, br.w2_lo, br.w2_hi)}
    out["w2_glob"] = out.a.round(2).map(wg)
    out["x"] = out.s / out.w2_glob
    out.to_csv(DATA, index=False)
    return out


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import figures_v5 as F
    d = pd.read_csv(DATA)
    ck = pd.read_csv(RESULTS / "phase2b_checkpoints.csv")
    cr = ck[(ck.crossing == True) & (ck.budget == 32_000)]
    fig, axes = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.4), sharey=True)
    stats = {}
    for ax, a in zip(axes, A_VALUES):
        g = d[d.a.round(2) == a].sort_values("x")
        wg = float(g.w2_glob.iloc[0])
        c = cr[cr.a.round(2) == a]
        ax.axhline(ALPHA_STAR, color=F.GREY, lw=0.8, ls="--", zorder=0)
        ax.axhline(a / 1.4, color=F.GREY, lw=0.8, ls=":", zorder=0)
        ax.axvline(1.0, color=F.BLUE, lw=0.8, zorder=0)
        ax.plot(g.x, g.w1, color=F.BLUE, lw=1.2, marker="o", ms=2.6, zorder=2)
        ax.plot(c.w2.abs() / wg, c.w1.abs(), ls="none", marker="o", ms=2.8, mfc="white", mec=F.VERM, mew=0.8, zorder=3)
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"output scale $R/R_{\mathrm{glob}}$" + f"  ($a = {a:.2f}$)")
        ax.text(0.012 if a == 1.30 else 0.02, ALPHA_STAR + 0.04, "class-mean optimum α* = 1.79", color="0.35", va="bottom")
        ax.text(0.012 if a == 1.30 else 0.02, a / 1.4 - 0.05, r"placement bound $a/1.4$", color="0.35", va="top")
        F._clean(ax)
        stats[a] = {"w2_glob": wg, "n_cross": len(c), "x_min": float(g.x.min()), "x_max": float(g.x.max()),
                    "w1_at_smallest": float(g.w1.iloc[0]),
                    "w1_below_bound_above_switch": bool((g[g.x > 1.02].w1 < a / 1.4).all()),
                    "w1_above_bound_below_switch": bool((g[g.x < 0.98].w1 > a / 1.4).all()),
                    "cross_w1_below_bound": float((c.w1.abs() < a / 1.4).mean())}
    axes[0].set_ylabel(r"$|w_1|$ of the minimiser")
    fig.tight_layout(pad=0.3)
    F.save(fig, "mechanism_w1")
    pd.DataFrame(stats).T.to_csv(RESULTS / "mechanism_w1_stats.csv")
    return stats


if __name__ == "__main__":
    data()
    print(figure())
