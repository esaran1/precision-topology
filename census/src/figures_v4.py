"""v4 figures (NOT yet in paper/figures: held until S1-S3 and the c1 test are scored).

    PYTHONPATH=. python -m src.figures_v4

Output: results/figures/v4/*.pdf (vector) and .png previews.  Style as paper/make_figures.py (single column
3.25 in, nothing below 7 pt, markers and line styles so nothing is colour-only).  Every panel states n and
shows uncertainty: bootstrap 95% intervals of medians (10,000 resamples, seed 0), Clopper-Pearson 95% intervals
of fractions, certified intervals of thresholds.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "figures" / "v4"
OUT.mkdir(parents=True, exist_ok=True)
COL = 3.25

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "grid.linewidth": 0.4, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  ok  {name}.pdf")


def boot_median(v, n=10_000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(v, (n, len(v))), axis=1)
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def clopper(k, n, alpha=0.05):
    def cdf(p, kk):
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))

    def solve(fn, lo=0.0, hi=1.0):
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if fn(mid) else (lo, mid)
        return 0.5 * (lo + hi)
    lo = 0.0 if k == 0 else solve(lambda p: 1 - cdf(p, k - 1) < alpha / 2)
    hi = 1.0 if k == n else solve(lambda p: cdf(p, k) > alpha / 2)
    return lo, hi


# ------------------------------------------------------------------------------------------ Block 3
def fig_prospective():
    from .prospective import _cert_ghat
    gh = _cert_ghat()
    runs = pd.read_csv(RESULTS / "prospective_runs.csv")
    runs = runs[runs.cross_step.notna()].copy()
    runs["R"] = [w2 * gh[(w, round(a, 2))] / 2 for w, a, w2 in zip(runs.window, runs.a, runs.cross_w2)]
    sc = pd.read_csv(RESULTS / "prospective_scores.csv")
    cal = pd.read_csv(RESULTS / "prospective_calibration.csv").set_index("a")
    fig, ax = plt.subplots(figsize=(COL, 3.1))
    models = [("C", "o", "#1f77b4", True, r"C $=\lambda(a)\,R_{\rm glob}$"),
              ("U", "o", "#1f77b4", False, r"U $=R_{\rm glob}$ (no lag)"),
              ("B1", "^", "#d62728", True, "B1: base crossing $|w_2|$"),
              ("B2", "s", "#2ca02c", True, "B2: pooled crossing $R$")]
    ns = []
    for r in sc.itertuples():
        g = runs[(runs.window == r.window) & (runs.a.round(2) == round(r.a, 2))]
        med, lo, hi = boot_median(g.R.values)
        ns.append(len(g))
        for m, mk, col, filled, _ in models:
            y = getattr(r, m)
            ax.errorbar(med, y, xerr=[[med - lo], [hi - med]], fmt=mk, ms=4 if m in ("C", "U") else 3.5,
                        mfc=col if filled else "white", mec=col, ecolor="0.6", elinewidth=0.6, capsize=1.5,
                        zorder=3 if m == "C" else 2)
        ax.plot([med, med], [r.U, r.C], color="#1f77b4", lw=0.5, zorder=1)      # λ scales U to C
    lim = [0.13, 0.33]
    ax.plot(lim, lim, color="0.3", lw=0.7, ls="--", zorder=1)
    ax.set_xlim(0.195, 0.285); ax.set_ylim(*lim)
    ax.set_xlabel(r"observed median crossing $R$ (bootstrap 95%)")
    ax.set_ylabel(r"predicted crossing $R$")
    for m, mk, col, filled, lab in models:
        ax.plot([], [], mk, color=col, mfc=col if filled else "white", mec=col, ms=4, ls="none", label=lab)
    ax.plot([], [], color="#1f77b4", lw=0.5, label=r"U$\to$C: fitted lag $\lambda$")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)
    ax.text(0.98, 0.03, f"$\\lambda$ fitted on the base window:\n"
            f"$\\lambda(1.30)={cal.loc[1.3, 'lambda_fitted']:.3f}$, $\\lambda(1.50)={cal.loc[1.5, 'lambda_fitted']:.3f}$\n"
            f"8 held-out settings; $n={min(ns)}$–${max(ns)}$ crossings of 90 each",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    save(fig, "v4_prospective")
    return {"n_min": min(ns), "n_max": max(ns)}


# ------------------------------------------------------------------------------------------ Blocks 4 and 5


def fig_fixed_scale(horizons=False):
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    r13 = br[(br.a.round(2) == 1.3) & (br.kind == "glob")].iloc[0]
    R_cert = 0.5 * (r13.R_lo + r13.R_hi)
    e2_x = 1.15 * 0.21446 / R_cert             # Block E held 1.15 x its frozen R_glob (0.21446), in certified units
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 4.3), sharex=True)
    for ax, blk, band, ylab in ((a1, 4, (0.9, 1.25), "placed at 4,000 steps"),
                                (a2, 5, (0.9, 1.1), "retained through 12,000 steps")):
        cv = pd.read_csv(RESULTS / f"fixed_scale_block{blk}_curve.csv")
        ts = pd.read_csv(RESULTS / f"fixed_scale_block{blk}_tests.csv").set_index("variant")
        ax.axvspan(*band, color="0.92", zorder=0, lw=0)
        ax.axvline(1.0, color="0.5", lw=0.6, ls=":")
        for var, mk, ls, col, dx in (("preserved", "o", "-", "#1f77b4", -0.006), ("reset", "s", "--", "#d62728", 0.006)):
            g = cv[cv.variant == var].sort_values("level")
            ax.errorbar(g.level + dx, g.frac, yerr=[g.frac - g.ci95_lo, g.ci95_hi - g.frac], fmt=mk, ls=ls, color=col,
                        ms=3, capsize=1.5, elinewidth=0.6, label=f"{var} (x$_{{50}}$ = {ts.loc[var, 'x50']:.3f})")
        n = int(cv.n.max())
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.04, 1.08)
        ax.text(0.99, 0.04, f"$n={n}$ checkpoints per level\nshaded: registered 50% band {band[0]}–{band[1]}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
        ax.set_title(f"Block {blk}: " + ("before placement" if blk == 4 else "just after placement"), loc="left")
    if horizons and (RESULTS / "fixed_scale_horizons_curve.csv").exists():
        hc = pd.read_csv(RESULTS / "fixed_scale_horizons_curve.csv")
        hc = hc[hc.variant == "preserved"]
        for H, ls in ((16000, (0, (3, 1))), (64000, (0, (1, 1)))):
            row = hc[hc.horizon == H].iloc[0]
            lv = [0.9, 0.95, 1.0, 1.05, 1.1]
            a1.plot(lv, [row[f"frac_{l}"] for l in lv], ls=ls, color="k", lw=0.9, label=f"preserved, {H:,} steps")
    # E-2: Block E's registered criterion, retention >= 0.9 at 1.15 x R_glob (frozen), and its observed 33/37
    lo, hi = clopper(33, 37)
    a2.plot([e2_x - 0.03, e2_x + 0.03], [0.9, 0.9], color="k", lw=1.2)
    a2.annotate("E-2 criterion: kept $\\geq$ 0.9\nat $1.15\\times$ Block E's $R_{\\rm glob}$", (e2_x + 0.03, 0.9),
                xytext=(1.42, 0.62), textcoords="data",
                fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    a2.errorbar([e2_x], [33 / 37], yerr=[[33 / 37 - lo], [hi - 33 / 37]], fmt="*", color="k", ms=6, capsize=1.5,
                elinewidth=0.6, label="Block E, E-2: 33/37 (FAIL)")
    a1.legend(loc="center right", frameon=False)
    a2.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), frameon=False)
    a2.set_xlabel(r"held $R/R_{\rm glob}$ ($a=1.30$, certified $R_{\rm glob}$)")
    fig.tight_layout(h_pad=0.6)
    save(fig, "v4_fixed_scale")
    return {"e2_x": e2_x}


# ------------------------------------------------------------------------------------------ certified thresholds
def fig_thresholds():
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    kb = pd.read_csv(RESULTS / "limit_K_base.csv").iloc[0]
    sl = pd.read_csv(RESULTS / "mn2_solve_limit.csv").iloc[0]
    runs = pd.read_csv(RESULTS / "wi_crossing_runs.csv")
    runs = runs[runs.budget == 32_000]
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    eps = sorted(br.a.round(2).unique() - 1)
    w = 0.018
    ns = []
    for e in eps:
        g = runs[runs.a.round(2) == round(1 + e, 2)].R_cert.values
        ns.append(len(g))
        vp = ax.violinplot([g], positions=[e], widths=0.035, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor("0.85"); b.set_edgecolor("0.6"); b.set_linewidth(0.4); b.set_alpha(1)
        med, lo, hi = boot_median(g)
        ax.errorbar([e], [med], yerr=[[med - lo], [hi - med]], fmt="_", color="0.25", ms=6, capsize=1.5, elinewidth=0.7)
    for kind, col, mk, lab in (("glob", "#1f77b4", "o", r"$R_{\rm glob}$ (certified)"),
                               ("solve", "#d62728", "s", r"$R_{\rm solve}$ (certified)")):
        g = br[br.kind == kind].sort_values("a")
        for r in g.itertuples():
            ax.add_patch(plt.Rectangle((r.a - 1 - w / 2, r.R_lo), w, r.R_hi - r.R_lo, color=col, lw=0))
            ax.plot([r.a - 1 - w / 2, r.a - 1 + w / 2], [0.5 * (r.R_lo + r.R_hi)] * 2, color=col, lw=0.8)
        ax.plot(g.a - 1, 0.5 * (g.R_lo + g.R_hi), color=col, lw=0.6, ls="-" if kind == "glob" else "--")
        ax.plot([], [], color=col, marker=mk, ls="-" if kind == "glob" else "--", ms=3, label=lab)
    # limit (ε = 0): certified intervals
    for lo, hi, col in ((kb.R_glob_inf_lo, kb.R_glob_inf_hi, "#1f77b4"), (sl.R_solve_inf_lo, sl.R_solve_inf_hi, "#d62728")):
        ax.add_patch(plt.Rectangle((-w / 2, lo), w, hi - lo, color=col, lw=0, alpha=0.85))
    ax.text(0.035, 0.212, r"limit $\varepsilon\to0$" + "\n(certified)", fontsize=6, va="center")
    for yv in (kb.R_glob_inf_hi, sl.R_solve_inf_lo):
        ax.annotate("", (w / 2, yv), xytext=(0.034, 0.212), textcoords="data",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"))
    nlab = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}$–${max(ns)}"
    ax.plot([], [], color="0.6", lw=4, label=f"free-training crossing $R$\n(budget 32k, $n={nlab}$ per $a$)")
    ax.set_xlim(-0.04, 0.64)
    ax.set_xlabel(r"$\varepsilon=a-1$")
    ax.set_ylabel(r"$R=|w_2|\hat G/2$")
    ax.legend(loc="center left", bbox_to_anchor=(0.07, 0.60), frameon=False)
    ax.text(0.99, 0.02, "interval heights are the certified widths;\nmedian bars: bootstrap 95%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="0.35")
    save(fig, "v4_thresholds")
    return {"n_min": min(ns), "n_max": max(ns)}


def main():
    info = {"prospective": fig_prospective(), "fixed_scale": fig_fixed_scale(), "thresholds": fig_thresholds()}
    print(info)


if __name__ == "__main__":
    main()
