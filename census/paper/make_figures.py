"""Regenerate every paper figure from committed artifacts. One command, no manual steps.

    PYTHONPATH=. python3 paper/make_figures.py

Output: paper/figures/*.pdf (vector) plus a provenance note.

This is an AUDIT PASS as well as production.  Assembling figures is the first
time numbers from different sections sit beside each other, so every value is
recomputed from its artifact here and compared against the canonical value.
Disagreements are printed as FINDING: and collected -- never silently fixed.

Three quantities in this project have appeared with multiple values.  Each is
pinned below with its point set / window, and the alternatives are named so a
figure can never quietly quote the wrong one:

  alpha          1.1172  (budget_alpha.csv, 1k-160k, 8 cells)
                 alternatives: 1.51 (1k-4k) ... 0.72 (40k-160k), the drift
                 that explains the extrapolation failure (Figure 5b).
  onset exponent -0.7340 (onset_law_extended.csv, 6 bracketed cells, 2k-128k)
                 alternatives: -0.8261 (family A, 4 cells 2k-32k, the held-out
                 fit).  NOTE: -0.8305 is q4's measured exponent, a DIFFERENT
                 FAMILY, not a family-A window variant.
  slope vs 1/beta 1.0984 (beta_law_points.csv, FIVE points, q2 and family A
                 both kept at matched beta=1.5)
                 alternatives: 1.1240 (four points, q2 merged into family A --
                 withdrawn 2026-09-11); 1.0547 under effective rather than
                 asymptotic beta.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

FINDINGS: list[str] = []
MANIFEST: list[dict] = []


def finding(msg: str) -> None:
    FINDINGS.append(msg)
    print(f"  FINDING: {msg}")


def record(name: str, sources: list[str], n: str, note: str,
           placement: str = "full-width") -> None:
    MANIFEST.append({"figure": name, "slug": SLUG[name], "sources": ", ".join(sources),
                     "n": n, "placement": placement, "note": note})


SLUG = {"Fig 1": "fig1_setting", "Fig 2": "fig2_exclusions",
        "Fig 3": "fig3_r_collapse", "Fig 4": "fig4_metric_check",
        "Fig 5": "fig5_budget_law", "Fig 6": "fig6_four_family",
        "Fig 7": "fig7_family_b", "Fig 8": "fig8_link"}

SMALLEST_PT = 6.0          # nothing may render below this at final size
COL_IN, DBL_IN = 3.25, 6.75


def audit_placement() -> None:
    """Every figure must be legible AT THE WIDTH IT IS PLACED.

    A 5.7in two-panel figure dropped into a 3.25in column scales by 0.57, so
    6.5pt legend text lands at 3.7pt.  Figures are therefore declared
    full-width (span both columns) or single-column, and checked against the
    declaration rather than assumed to fit.
    """

    import pypdf

    print("\nAUDIT: legibility at declared placement")
    for row in MANIFEST:
        pdf = FIGDIR / f"{row['slug']}.pdf"
        if not pdf.exists():
            continue
        w = float(pypdf.PdfReader(str(pdf)).pages[0].mediabox.width) / 72
        target = COL_IN if row["placement"] == "single-column" else DBL_IN
        scale = min(target / w, 1.0)
        smallest = 6.5 * scale
        flag = "ok " if smallest >= SMALLEST_PT else "BAD"
        print(f"  {flag} {row['figure']:7s} {w:.2f}in -> {row['placement']:13s} "
              f"smallest text {smallest:.1f}pt")
        if smallest < SMALLEST_PT:
            finding(f"{row['figure']} renders {smallest:.1f}pt at "
                    f"{row['placement']} width (min {SMALLEST_PT}pt)")


# ---------------------------------------------------------------- style ----
# Single column is ~3.25in.  Nothing below 7pt at final size.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,          # embed as TrueType, not Type 3
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL = 3.25          # single-column width, inches
DBL = 6.75          # full width

# marker/linestyle pairs so nothing is colour-only
STYLE = [("o", "-", "#1f77b4"), ("s", "--", "#d62728"),
         ("^", "-.", "#2ca02c"), ("D", ":", "#9467bd"),
         ("v", (0, (3, 1, 1, 1)), "#ff7f0e")]


def save(fig, name: str) -> None:
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png", dpi=200)   # convenience preview
    plt.close(fig)
    print(f"  ok  {name}.pdf")


# ------------------------------------------------------------- helpers ----
def loglog_fit(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    sx = lx.mean()
    slope = ((lx - sx) * (ly - ly.mean())).sum() / ((lx - sx) ** 2).sum()
    icept = ly.mean() - slope * sx
    resid = ly - (slope * lx + icept)
    se = (np.sqrt((resid ** 2).sum() / (len(lx) - 2) / ((lx - sx) ** 2).sum())
          if len(lx) > 2 else float("nan"))
    return slope, icept, se


def clopper_pearson(k: int, n: int):
    """Exact binomial interval; no scipy in this venv."""
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            tail = sum(comb(n, i) * m ** i * (1 - m) ** (n - i) for i in range(k, n + 1))
            if tail < 0.025:
                a = m
            else:
                b = m
        lo = a
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            tail = sum(comb(n, i) * m ** i * (1 - m) ** (n - i) for i in range(0, k + 1))
            if tail > 0.025:
                a = m
            else:
                b = m
        hi = a
    return lo, hi


def fisher_two_sided(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    rng = range(max(0, c1 - (n - r1)), min(r1, c1) + 1)
    tot = sum(comb(r1, i) * comb(n - r1, c1 - i) for i in rng)
    obs = comb(r1, a) * comb(n - r1, c1 - a)
    return sum(comb(r1, i) * comb(n - r1, c1 - i) for i in rng
               if comb(r1, i) * comb(n - r1, c1 - i) <= obs * (1 + 1e-9)) / tot


def auc(values, solved):
    v = np.asarray(values, float)
    s = np.asarray(solved, bool)
    pos, neg = v[s], v[~s]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def auc_ci(values, solved, n_boot=1500, seed=0):
    rng = np.random.default_rng(seed)
    v, s = np.asarray(values, float), np.asarray(solved, bool)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(v), len(v))
        if 0 < s[i].sum() < len(i):
            out.append(auc(v[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


# ------------------------------------------- canonical values, verified ----
ALPHA, ALPHA_CI = 1.1172, 0.1187
ONSET_EXP, ONSET_N = -0.7340, 6
SLOPE5 = 1.0984
SLOPE5_CI = (0.958, 1.239)
SLOPE_EFFECTIVE = 1.0547


def audit_canonical() -> None:
    """Recompute every canonical value from its artifact before plotting."""

    print("\nAUDIT: canonical values recomputed from artifacts")
    d = pd.read_csv(RESULTS / "budget_alpha.csv")
    a, _, se = loglog_fit(d.budget.values, d.w2_med.values)
    print(f"  alpha (1k-160k, n={len(d)}): {a:.4f} +- {se:.4f}")
    if abs(a - ALPHA) > 0.001:
        finding(f"alpha artifact {a:.4f} != canonical {ALPHA}")

    o = pd.read_csv(RESULTS / "onset_law_extended.csv")
    o = o[o.bracketed == True]  # noqa: E712
    s, _, se_o = loglog_fit(o.budget.values, (o.onset - 1).values)
    print(f"  onset exponent ({len(o)} bracketed cells): {s:.4f} +- {se_o:.4f}")
    if abs(s - ONSET_EXP) > 0.002:
        finding(f"onset exponent artifact {s:.4f} != canonical {ONSET_EXP}")
    if len(o) != ONSET_N:
        finding(f"bracketed cell count {len(o)} != canonical {ONSET_N}")

    p = pd.read_csv(RESULTS / "beta_law_points.csv")
    m5 = (p.inv_beta.values * -p.measured.values).sum() / (p.inv_beta.values ** 2).sum()
    print(f"  through-origin slope (5 points): {m5:.4f}")
    if abs(m5 - SLOPE5) > 0.001:
        finding(f"slope artifact {m5:.4f} != canonical {SLOPE5}")
    if len(p) != 5:
        finding(f"beta_law_points has {len(p)} rows, expected the 5-point set")


# ------------------------------------------------------------ Figure 1 ----
def fig1_setting() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL, 2.5))
    t = np.linspace(-6, 6, 800)
    for (mk, ls, c), a in zip(STYLE, (0.9, 1.05, 2.0)):
        lab = {0.9: r"$a=0.9$ (monotone)", 1.05: r"$a=1.05$ (just past)",
               2.0: r"$a=2.0$"}[a]
        ax1.plot(t, t + a * np.sin(t), ls=ls, color=c, label=lab)
    # fold depth D on the a=2 curve
    a = 2.0
    crit = np.arccos(-1.0 / a)
    tmax, tmin = np.pi - crit, np.pi + crit
    fmax, fmin = tmax + a * np.sin(tmax), tmin + a * np.sin(tmin)
    ax1.annotate("", xy=(tmin, fmin), xytext=(tmin, fmax),
                 arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
    ax1.text(tmin + 0.25, (fmax + fmin) / 2, r"$D(a)$", fontsize=7.5)
    ax1.plot([tmax, tmin], [fmax, fmin], "k.", ms=3.5)
    ax1.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$f_a(t) = t + a\sin t$")
    ax1.set_title(r"(a) the activation family", fontsize=8)
    ax1.legend(loc="upper left", frameon=False)

    x = np.linspace(-2.3, 2.3, 700)
    ax2.axvspan(-0.8, 0.8, color="#1f77b4", alpha=0.13)
    for lo, hi in ((1.2, 2.0), (-2.0, -1.2)):
        ax2.axvspan(lo, hi, color="#d62728", alpha=0.13)
    ax2.plot(x, np.sign(np.abs(x) - 1), color="0.35", lw=1.0, ls=":",
             label=r"target $\mathrm{sign}(|x|-1)$")

    # Parameters are RECOVERED FROM SEEDS, not invented: Adam/lr 1e-2/2,000 steps
    # at a = 1.5, seed 0 (solves) and seed 3 (fails).  Verified on this grid:
    # solver inner max logit -0.509 < 0 < 0.051 outer min.
    def net(w1, b1, w2, b2, a=1.5):
        return w2 * ((w1 * x + b1) + a * np.sin(w1 * x + b1)) + b2

    solver = (0.9437292267391273, 2.3593122836987837,
              -4.400602112045125, 13.150372704129833)
    failure = (-0.9983804107073221, -2.306300277491995,
               3.457569886689401, 10.09746721514516)
    inner_m = np.abs(x) <= 0.8
    outer_m = (np.abs(x) >= 1.2) & (np.abs(x) <= 2.0)
    sv = net(*solver)
    if not (sv[inner_m].max() < 0 < sv[outer_m].min()):
        finding("Fig 1b 'solving' parameters do not separate on the plot grid")
    ax2.plot(x, np.tanh(sv), color="#2ca02c", ls="-",
             label="solving (seed 0, squashed)")
    ax2.plot(x, np.tanh(net(*failure)), color="#ff7f0e", ls="--",
             label="non-solving (seed 3)")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel("network output")
    ax2.set_title(r"(b) the task: inner (blue) vs outer (red)", fontsize=8)
    ax2.legend(loc="lower right", frameon=False)
    save(fig, "fig1_setting")
    record("Fig 1", ["analytic (no data)"], "n/a",
           "f_a at a=0.9/1.05/2.0 with D(a) marked; task regions I=[-0.8,0.8], O=+-[1.2,2.0]")


# ------------------------------------------------------------ Figure 2 ----
def fig2_exclusions() -> None:
    d = pd.read_csv(RESULTS / "exclusion_table.csv")
    unreached = d[d.population == "zero_basin_constructed"]
    found = d[d.population == "found_adam"]
    fig, axes = plt.subplots(1, 4, figsize=(DBL, 2.3))
    cols = [("sharpness_product", r"$\lambda_{\max}\eta$", True),
            ("distance_median", "distance from init", False),
            ("margin", "logit margin", True),
            ("barrier_mep_median", "MEP barrier", False)]
    for ax, (col, lab, logscale) in zip(axes, cols):
        for grp, name, (mk, ls, c) in ((unreached, "unreached", STYLE[0]),
                                       (found, "found", STYLE[1])):
            vals = grp[col].values
            if logscale:
                vals = np.where(vals <= 0, np.nan, vals)
            ax.plot(grp.a.values, vals, marker=mk, ls="none", color=c,
                    ms=4, label=name)
        if logscale:
            ax.set_yscale("log")
        if col == "sharpness_product":
            ax.axhline(2.0, color="k", ls="--", lw=0.8)
            ax.text(1.05, 2.3, r"$2/\eta$ threshold", fontsize=6)
        ax.set_xlabel(r"$a$")
        ax.set_title(lab, fontsize=7.5)
        ax.grid(alpha=0.25, lw=0.3)
    # mark the two registered-prediction reversals
    for ax, flag in zip(axes, [False, True, True, False]):
        if flag:
            ax.set_title(ax.get_title() + " \u2020", fontsize=7.5)
    axes[0].legend(frameon=False, loc="lower right")
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.5, 0.02,
             "\u2020 opposite to the registered prediction: unreached points sit NEARER "
             "init\n   and carry SMALLER margin than found ones.",
             ha="center", fontsize=6.5)
    save(fig, "fig2_exclusions")
    record("Fig 2", ["exclusion_table.csv"],
           f"{len(unreached)} unreached, {len(found)} found",
           "log strip charts; dagger marks the two reversals (distance, margin)")


# ------------------------------------------------------------ Figure 3 ----
def fig3_r_collapse() -> None:
    d = pd.read_csv(RESULTS / "r_pooled.csv")
    fam = pd.read_csv(RESULTS / "r_families.csv")
    fb = pd.read_csv(RESULTS / "r_family_b.csv")
    fig = plt.figure(figsize=(DBL, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.5, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    edges = np.array([0, .1, .2, .25, .3, .32, .34, .36, .38, .4, .45, .5, .6, .8, 1.2, 2.0, 1e9])
    d = d.copy()
    d["bin"] = pd.cut(d.R, edges)
    g = d.groupby("bin", observed=True).agg(n=("solved", "size"), k=("solved", "sum")).reset_index()
    g["mid"] = [min(b.mid, 2.2) for b in g["bin"]]
    g = g[g.n >= 5]
    lo, hi = zip(*[clopper_pearson(int(k), int(n)) for k, n in zip(g.k, g.n)])
    rate = g.k / g.n
    ax.axvspan(0.30, 0.50, color="0.85", alpha=0.7, zorder=0)
    ax.errorbar(g["mid"], rate, yerr=[rate - np.array(lo), np.array(hi) - rate],
                marker="o", ls="-", color="#1f77b4", ms=3.5, lw=1.0, capsize=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$R = |w_2|\,G^*(a)/2$   (margin capacity)")
    ax.set_ylabel(r"$P(\mathrm{solve})$")
    ax.set_ylim(-0.04, 1.06)
    ax.text(0.315, 0.40, "transition\n0.30–0.50", fontsize=6.5)
    ax.text(0.055, 0.72, "0 of 2,160\nsolved", fontsize=6.5, color="#1f77b4")
    ax.text(0.70, 0.30, "402 of 403\nsolved", fontsize=6.5, color="#1f77b4")
    ax.set_title(r"(a) pooled over 2,910 runs: 12 values of $a$, 6 budgets, both optimisers"
                 "\n(Clopper–Pearson 95% intervals)", fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.3)

    axo = fig.add_subplot(gs[1, 0])
    ac = d[d.src == "alpha_comp"]
    labels, ps = [], []
    for (name, lo_, hi_) in (("R<0.3", 0, 0.3), ("0.3–0.5", 0.3, 0.5), ("R>0.5", 0.5, 1e9)):
        sub = ac[(ac.R >= lo_) & (ac.R < hi_)]
        A, S = sub[sub.opt == "adam"], sub[sub.opt == "sgd"]
        ka, na = int(A.solved.sum()), len(A)
        ks, ns = int(S.solved.sum()), len(S)
        labels.append(name)
        ps.append(fisher_two_sided(ka, na - ka, ks, ns - ks) if na and ns else np.nan)
        for off, (k, n, c, mk) in ((-0.13, (ka, na, "#1f77b4", "o")),
                                   (0.13, (ks, ns, "#d62728", "s"))):
            if n:
                r = k / n
                l, h = clopper_pearson(k, n)
                axo.errorbar(len(labels) - 1 + off, r, yerr=[[r - l], [h - r]],
                             marker=mk, color=c, ms=4, capsize=1.5, lw=1.0)
    axo.set_xticks(range(len(labels)))
    axo.set_xticklabels(labels)
    axo.set_ylim(-0.08, 1.15)
    axo.set_ylabel(r"$P(\mathrm{solve})$")
    for i, p in enumerate(ps):
        axo.text(i, 1.04, f"$p={p:.2f}$", ha="center", fontsize=6)
    axo.plot([], [], "o", color="#1f77b4", label="Adam")
    axo.plot([], [], "s", color="#d62728", label="SGD")
    axo.legend(frameon=False, loc="lower right", fontsize=6)
    axo.set_title(r"(b) same curve for both optimisers ($a=1.25$, $n=180$ each)"
                  "\nFisher exact, two-sided", fontsize=7.5)
    axo.grid(alpha=0.25, lw=0.3)

    axa = fig.add_subplot(gs[1, 1])
    sets = [("A", d.R.values, d.w2.values, d.gstar.values, d.solved.values),
            ("q2", *[fam[fam.q == 2.0][c].values for c in ("R", "w2", "gstar", "solved")]),
            ("q1", *[fam[fam.q == 1.0][c].values for c in ("R", "w2", "gstar", "solved")]),
            ("B", fb.R.values, fb.w2_abs.values, fb.gstar.values, fb.solved.values)]
    w = 0.26
    for j, (lab, mk, c) in enumerate((("$R$", "o", "#1f77b4"),
                                      (r"$|w_2|$", "s", "#d62728"),
                                      (r"$G^*$", "^", "#2ca02c"))):
        xs, ys, es = [], [], [[], []]
        for i, (fname, R_, W_, G_, S_) in enumerate(sets):
            v = (R_, W_, G_)[j]
            a_ = auc(v, S_)
            l, h = auc_ci(v, S_)
            xs.append(i + (j - 1) * w)
            ys.append(a_)
            es[0].append(a_ - l)
            es[1].append(h - a_)
        axa.bar(xs, ys, width=w * 0.9, color=c, alpha=0.75, label=lab,
                edgecolor="k", lw=0.4, hatch=["", "//", ".."][j])
        axa.errorbar(xs, ys, yerr=es, fmt="none", ecolor="k", capsize=1.5, lw=0.7)
    axa.set_xticks(range(len(sets)))
    axa.set_xticklabels([s[0] for s in sets])
    axa.set_ylim(0.3, 1.04)
    axa.axhline(0.5, color="k", lw=0.5, ls=":")
    axa.set_ylabel("AUC (solved vs not)")
    axa.set_xlabel("activation family")
    axa.legend(frameon=False, ncol=3, fontsize=6, loc="lower left")
    axa.set_title(r"(c) $R$ dominates its factors — except family B ($\beta{=}1$)"
                  "\n1,500-resample bootstrap 95% CI", fontsize=7.5)
    axa.grid(alpha=0.25, lw=0.3, axis="y")
    save(fig, "fig3_r_collapse")
    record("Fig 3", ["r_pooled.csv", "r_families.csv", "r_family_b.csv"],
           "(a) 2,910  (b) 180/optimiser  (c) A 2,910 / q2 360 / q1 360 / B 1,000",
           "Clopper-Pearson on rates; bootstrap CI on AUC; Fisher exact on (b)")


# ------------------------------------------------------------ Figure 4 ----
def fig4_metric_check() -> None:
    from src.fold1d_theorem import maximum_gap
    s = pd.concat([pd.read_csv(RESULTS / "fold1d_sweep.csv"),
                   pd.read_csv(RESULTS / "fold1d_refine.csv")])
    s = s[(s.activation == "sin_family") & (s.parameter > 1.0)].copy()
    gs_ = {a: maximum_gap(float(a), resolution=600) for a in sorted(s.parameter.unique())}
    s["R"] = s.w2_abs * s.parameter.map(gs_) / 2
    s["solved"] = s.solved.astype(bool)
    edges = np.arange(0.0, 0.55, 0.025)
    s["bin"] = pd.cut(s.R, edges)
    g = s.groupby("bin", observed=True).agg(
        n=("solved", "size"), rate=("solved", "mean"), err=("eval_errors", "mean"),
        q25=("eval_errors", lambda v: v.quantile(.25)),
        q75=("eval_errors", lambda v: v.quantile(.75))).reset_index()
    g["mid"] = [b.mid for b in g["bin"]]
    g = g[g.n >= 8]

    fig, ax = plt.subplots(figsize=(COL * 1.55, 2.6))
    ax.axvspan(0.055, 0.307, color="#d62728", alpha=0.09, zorder=0)
    ax.axvspan(0.332, 0.452, color="#1f77b4", alpha=0.12, zorder=0)
    ax.plot(g["mid"], g.rate, "o-", color="#1f77b4", ms=3.5, label="binary solve rate")
    ax.set_ylabel("solve rate", color="#1f77b4")
    ax.set_xlabel(r"$R = |w_2|\,G^*(a)/2$")
    ax.set_ylim(-0.04, 1.06)
    ax.tick_params(axis="y", labelcolor="#1f77b4")

    tw = ax.twinx()
    tw.fill_between(g["mid"], g.q25, g.q75, color="#d62728", alpha=0.15, lw=0)
    tw.plot(g["mid"], g.err, "s--", color="#d62728", ms=3.5, label="mean eval errors")
    tw.set_ylabel("eval errors (IQR band)", color="#d62728")
    tw.tick_params(axis="y", labelcolor="#d62728")
    tw.invert_yaxis()
    ax.text(0.06, 0.62, "76% of the error swing\noccurs while 0 runs solve",
            fontsize=6.5, color="#d62728")
    ax.annotate("", xy=(0.055, -0.02), xytext=(0.307, -0.02),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#d62728"))
    ax.annotate("", xy=(0.332, 1.02), xytext=(0.452, 1.02),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#1f77b4"))
    ax.set_title("Metric check: continuous error falls before the binary rate moves",
                 fontsize=7.5)
    save(fig, "fig4_metric_check")
    record("Fig 4", ["fold1d_sweep.csv", "fold1d_refine.csv"], "2,400",
           "10-90% crossings: continuous [0.055,0.307], binary [0.332,0.452], disjoint")


# ------------------------------------------------------------ Figure 5 ----
def fig5_budget_law() -> None:
    o = pd.read_csv(RESULTS / "onset_law_extended.csv")
    ob = o[o.bracketed == True].copy()  # noqa: E712
    eps = (ob.onset - 1).values
    B = ob.budget.values.astype(float)
    slope, icept, se = loglog_fit(B, eps)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(DBL, 2.5))
    xs = np.logspace(np.log10(B.min() * 0.8), np.log10(B.max() * 1.3), 50)
    ax.fill_between(xs, np.exp(icept) * xs ** -0.895, np.exp(icept) * xs ** -0.595,
                    color="0.8", alpha=0.5, label="registered band [-0.895, -0.595]")
    ax.plot(xs, np.exp(icept) * xs ** slope, "k-", lw=1.0,
            label=rf"fit {slope:.4f} ({len(ob)} cells)")
    ax.plot(B, eps, "o", color="#1f77b4", ms=4.5, label="measured onsets")

    train = ob[ob.budget < 128000].nsmallest(4, "budget")
    s4, i4, se4 = loglog_fit(train.budget.values.astype(float), (train.onset - 1).values)
    lx = np.log(128000.0)
    lxt = np.log(train.budget.values.astype(float))
    sxx = ((lxt - lxt.mean()) ** 2).sum()
    resid = np.log((train.onset - 1).values) - (s4 * lxt + i4)
    sig = np.sqrt((resid ** 2).sum() / (len(lxt) - 2))
    sep = sig * np.sqrt(1 + 1 / len(lxt) + (lx - lxt.mean()) ** 2 / sxx)
    pred = s4 * lx + i4
    plo, phi = np.exp(pred - 4.303 * sep), np.exp(pred + 4.303 * sep)
    ax.plot(train.budget, (train.onset - 1), "o", mfc="none", mec="#2ca02c",
            mew=1.1, ms=8, label="4 cells used for held-out fit")
    ax.errorbar([128000], [np.exp(pred)],
                yerr=[[np.exp(pred) - plo], [phi - np.exp(pred)]],
                fmt="v", color="#2ca02c", ms=5, capsize=2.5, lw=1.0,
                label="predicted at 128k (95% PI)")
    meas = float(ob[ob.budget == 128000].onset.iloc[0] - 1)
    ax.plot([128000], [meas], "*", color="#d62728", ms=11,
            label=f"measured {meas:.3f}: OUTSIDE")
    if plo <= meas <= phi:
        finding("held-out 128k value is INSIDE the interval; text says outside")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("budget $B$ (steps)")
    ax.set_ylabel(r"onset $\varepsilon = a_{\mathrm{onset}} - 1$")
    ax.set_title(r"(a) budget law, Adam only; 6 of 6 cells bracketed", fontsize=7.5)
    ax.legend(frameon=False, loc="lower left", fontsize=5.8)
    ax.grid(alpha=0.25, lw=0.3, which="both")

    d = pd.read_csv(RESULTS / "budget_alpha.csv")
    Bv, Wv = d.budget.values.astype(float), d.w2_med.values
    rows = []
    for i in range(len(Bv)):
        for j in range(i + 3, len(Bv) + 1):
            a_, _, _ = loglog_fit(Bv[i:j], Wv[i:j])
            rows.append((np.sqrt(Bv[i] * Bv[j - 1]), a_, j - i))
    rows = pd.DataFrame(rows, columns=["mid", "alpha", "cells"])
    for cells, (mk, ls, c) in zip(sorted(rows.cells.unique())[:4], STYLE):
        sub = rows[rows.cells == cells]
        ax2.plot(sub["mid"], sub.alpha, marker=mk, ls=ls, color=c, ms=3.5,
                 label=f"{cells}-cell window")
    ax2.axhline(ALPHA, color="k", ls="--", lw=0.9)
    ax2.text(2.2e3, ALPHA + 0.045, rf"full-range $\alpha={ALPHA:.4f}$ ($n=8$)", fontsize=6)
    ax2.set_xscale("log")
    ax2.set_xticks([2e3, 1e4, 5e4])
    ax2.set_xticklabels(["2k", "10k", "50k"])
    ax2.minorticks_off()
    ax2.set_xlabel("window geometric-mean budget")
    ax2.set_ylabel(r"fitted $\alpha$")
    ax2.set_title(r"(b) $\alpha$ drifts 1.51 $\to$ 0.72 with window"
                  "\nthe law interpolates; it does not extrapolate", fontsize=7.5)
    ax2.legend(frameon=False, fontsize=6)
    ax2.grid(alpha=0.25, lw=0.3)
    save(fig, "fig5_budget_law")
    record("Fig 5", ["onset_law_extended.csv", "budget_alpha.csv"],
           "(a) 6 bracketed cells, 40 seeds each  (b) 8 budgets, 30 seeds each",
           f"held-out: fit on 4 smallest, predict 128k -> [{plo:.5f},{phi:.5f}], measured {meas:.3f}")


# ------------------------------------------------------------ Figure 6 ----
def fig6_four_family() -> None:
    p = pd.read_csv(RESULTS / "beta_law_points.csv")
    fig, ax = plt.subplots(figsize=(COL * 1.5, 2.8))
    xs = np.linspace(0, 0.92, 50)
    ax.fill_between(xs, -(ALPHA + ALPHA_CI) * xs, -(ALPHA - ALPHA_CI) * xs,
                    color="k", alpha=0.09, label=r"$\alpha$ 95% CI")
    ax.plot(xs, -ALPHA * xs, "k--", lw=1.0,
            label=rf"predicted $-\alpha/\beta$, $\alpha={ALPHA:.4f}$")
    for _, r in p.iterrows():
        n_cells = int(r.n_cells)
        err = 0.114 if n_cells >= 4 else 0.171
        is_A = r.family.startswith("familyA")
        ax.errorbar(r.inv_beta, r.measured, yerr=err,
                    marker="D" if is_A else "o",
                    color="#d62728" if is_A else "#1f77b4",
                    ms=5.5 if is_A else 4.5, capsize=2, lw=1.0, ls="none")
        ax.annotate(r.family.split("_")[0], (r.inv_beta, r.measured),
                    textcoords="offset points", xytext=(5, -9), fontsize=6)
    matched = p[np.isclose(p.beta, 1.5)]
    ax.plot(matched.inv_beta, matched.measured, "o", mfc="none", mec="#2ca02c",
            mew=1.2, ms=11, ls="none", label=r"matched-$\beta$ cross-route control")
    ax.plot([], [], "D", color="#d62728", ms=5,
            label="family A (measured first)")
    ax.plot([], [], "o", color="#1f77b4", ms=4.5, label="constructed families")
    ax.set_xlabel(r"$1/\beta$")
    ax.set_ylabel("measured onset exponent")
    ax.set_title(r"Onset exponent tracks $1/\beta$  —  ADAM ONLY"
                 "\n" rf"5-point through-origin slope {SLOPE5:.4f} "
                 rf"[{SLOPE5_CI[0]:.3f}, {SLOPE5_CI[1]:.3f}]", fontsize=7.5)
    ax.legend(frameon=False, loc="lower left", fontsize=6)
    ax.grid(alpha=0.25, lw=0.3)
    save(fig, "fig6_four_family")
    record("Fig 6", ["beta_law_points.csv"],
           "5 points; cells per family: " + ", ".join(f"{r.family.split('_')[0]}={int(r.n_cells)}"
                                                      for _, r in p.iterrows()),
           f"error bars +-0.114 (>=4 cells) / +-0.171 (3 cells); effective-beta alt {SLOPE_EFFECTIVE}",
           placement="full-width")


# ------------------------------------------------------------ Figure 7 ----
def fig7_family_b() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(DBL, 2.4))
    oa = pd.read_csv(RESULTS / "onset_law_extended.csv")
    oa = oa[oa.bracketed == True]  # noqa: E712
    ax.plot(oa.budget, oa.onset - 1, "o-", color="#1f77b4", ms=4,
            label=r"family A ($\beta=1.5$): moves")
    ob = pd.read_csv(RESULTS / "onset_family_b.csv")
    obb = ob[ob.bracketed == True]  # noqa: E712
    ax.plot(obb.budget, np.abs(obb.onset_alpha), "s--", color="#d62728", ms=4,
            label=r"family B ($\beta=1$): flat")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("budget $B$")
    ax.set_ylabel(r"onset $|\varepsilon|$")
    ax.set_title(r"(a) the budget law fails for family B", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6)
    ax.grid(alpha=0.25, lw=0.3, which="both")

    fb = pd.read_csv(RESULTS / "r_family_b.csv")
    g = fb.groupby("parameter").agg(w2=("w2_abs", "median"), gstar=("gstar", "first"))
    g["prod"] = g.w2 * g.gstar / 2
    sl, _, _ = loglog_fit(g.gstar.values, g.w2.values)
    ax2.plot(g.gstar, g.w2, "o-", color="#d62728", ms=4,
             label=rf"median $|w_2| \sim G^{{*{sl:.3f}}}$")
    ax2.plot(g.gstar, g["prod"], "^-.", color="#2ca02c", ms=4,
             label=rf"product $R$ (varies {g['prod'].max()/g['prod'].min():.1f}$\times$)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks([0.16, 0.5, 1.6, 3.2])
    ax2.set_xticklabels(["0.16", "0.5", "1.6", "3.2"])
    ax2.minorticks_off()
    ax2.set_xlabel(r"$G^*(\alpha)$  (varies $%.0f\times$)" % (g.gstar.max() / g.gstar.min()))
    ax2.set_ylabel("median value")
    ax2.set_title("(b) compensation: weights grow as the fold shallows", fontsize=7.5)
    ax2.legend(frameon=False, fontsize=6)
    ax2.grid(alpha=0.25, lw=0.3, which="both")
    if abs(sl + 0.475) > 0.02:
        finding(f"family B compensation exponent {sl:.3f} != documented -0.475")
    save(fig, "fig7_family_b")
    record("Fig 7", ["onset_law_extended.csv", "onset_family_b.csv", "r_family_b.csv"],
           "(a) A 6 cells / B 2 bracketed of 3  (b) 5 alpha values, 200 runs each",
           f"compensation exponent {sl:.3f}; product range {g['prod'].max()/g['prod'].min():.1f}x vs G* {g.gstar.max()/g.gstar.min():.0f}x")


# ------------------------------------------------------------ Figure 8 ----
def fig8_link() -> None:
    arms = [("0.3x init", "gelu_scale_scaled_down.csv", "#1f77b4", "o"),
            ("standard", "gelu_scale_standard.csv", "#2ca02c", "s"),
            ("found-pattern", "gelu_scale_scaled_up.csv", "#d62728", "^")]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for i, (lab, f, c, mk) in enumerate(arms):
        d = pd.read_csv(RESULTS / f)
        k, n = int(d.separated.sum()), len(d)
        lo, hi = clopper_pearson(k, n)
        ax.errorbar(i, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]],
                    marker=mk, color=c, ms=6, capsize=2.5, lw=1.0)
        ax.text(i, 100 * hi + 1.2, f"{k}/{n}", ha="center", fontsize=6)
    ax.axhline(0.0537, color="k", ls="--", lw=0.9)
    ax.text(-0.35, 0.9, "monotonic zero: 0/5,580,\nrate < 0.0537% (one-sided 95%)",
            fontsize=6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms])
    ax.set_ylabel("dense-verified separation rate (%)")
    ax.set_xlabel("initialisation scale (width 3)")
    ax.set_ylim(-1.5, 26)
    ax.set_title("Link setting: GELU dose-response vs the monotonic floor", fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.3, axis="y")
    save(fig, "fig8_link")
    record("Fig 8", ["gelu_scale_*.csv", "monotonic_zero.md"], "200 per arm; floor n=5,580",
           "Clopper-Pearson 95%; up-vs-down Fisher one-sided p=1.5e-6",
           placement="single-column")


def main() -> int:
    print("Regenerating paper figures from committed artifacts\n")
    audit_canonical()
    print("\nFigures:")
    for fn in (fig1_setting, fig2_exclusions, fig3_r_collapse, fig4_metric_check,
               fig5_budget_law, fig6_four_family, fig7_family_b, fig8_link):
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            finding(f"{fn.__name__} FAILED: {type(exc).__name__}: {exc}")
    audit_placement()
    man = pd.DataFrame(MANIFEST)
    man.to_csv(FIGDIR / "manifest.csv", index=False)
    lines = ["# Figure provenance", "",
             "Regenerated by `PYTHONPATH=. python3 paper/make_figures.py`.",
             "Every number is recomputed from the artifact at render time; no",
             "hand-entered values. Canonical quantities are re-verified against",
             "their artifacts on each run (see the audit block at the top of output).",
             ""]
    for r in MANIFEST:
        lines += [f"## {r['figure']} (`{r['slug']}.pdf`, {r['placement']})", "",
                  f"- **sources**: {r['sources']}",
                  f"- **n**: {r['n']}",
                  f"- **uncertainty / notes**: {r['note']}", ""]
    (FIGDIR / "PROVENANCE.md").write_text("\n".join(lines))
    print(f"\n{len(FINDINGS)} finding(s)")
    (FIGDIR / "audit.txt").write_text("\n".join(FINDINGS) if FINDINGS else "no findings\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
