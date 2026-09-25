"""v4 figures for the paper's core chain (NOT yet in paper/figures).

    PYTHONPATH=. python -m src.figures_v4

Output: results/figures/v4/*.pdf (vector) and 300-dpi .png previews at true size, then
  - a layout check per figure (layout_check: no legend or in-axes text over data, no text over text), and
  - an audit per PDF (audit: width <= 5.5 in, every glyph actually set in the PDF >= 8 pt, every text run inside the
    page box).
Built at the ICLR text width (5.5 in, single-column template) and placed at built size; panels side by side, one
shared legend row above the panels, one note block (n, uncertainty, verdicts) below them.  One style table (STYLE) sets colour, marker, line style and fill for every series type;
nothing is distinguished by colour alone; fitted predictors are filled, unfitted open.  Every panel states n and
shows uncertainty: bootstrap 95% intervals (10,000 resamples, seed 0), Clopper-Pearson 95% intervals of fractions,
certified intervals of thresholds.

Plotting only: every number is read from a committed artifact in results/ (checked against `git ls-files` at the
end); the only things computed here are medians, means, fractions, Clopper-Pearson and bootstrap intervals, and
unit conversions R = |w2| G_hat / 2 that the producing scripts define.  Deterministic: PDFs are written without a
creation date, and every bootstrap uses seed 0.

Figures (the core chain, in order):
  v4_decomposition      Phase 1: failures are placement failures, not bias failures
  v4_thresholds         certified R_glob / R_solve against free-training crossings; limit and small-eps c1 test
  v4_cond_candidates    Block 1 audit at a = 1.30: every candidate, one continuous branch
  v4_fixed_scale        Blocks 4/5: placement and retention at fixed scale, E-2's criterion, horizon extension
  v4_mirror_branches    post hoc: crossing R against the occupied mirror branch's own threshold
  v4_prospective        Block 3: held-out windows, per setting
  v4_prospective_own    own-seed prospective test, per run
"""

from __future__ import annotations

import re
import subprocess
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures" / "v4"
OUT.mkdir(parents=True, exist_ok=True)
TEXT_W = 5.5               # ICLR \textwidth (iclr2027_conference.sty); figures are placed at built size
SMALLEST_PT = 8.0          # every glyph in every PDF, sub- and superscripts included
TICK_PT = 8.0              # tick labels (never contain scripts)
NOTE_PT = 8.5              # legends, titles, in-axes text (may contain scripts)
LABEL_PT = 9.0             # axis labels
TITLE_PAD = 8.0            # pt between the axes top and a panel title (clears the top tick label)
SOURCES: set[str] = set()
LAYOUT: dict[str, list[str]] = {}

# Mathtext sets sub- and superscripts at 0.7x the base size (R_glob in an 8.5 pt legend would put 6 pt glyphs in the
# PDF).  Scripts are set at SMALLEST_PT / NOTE_PT of their base instead: still lowered or raised, so still read as
# scripts, and every glyph actually in the PDF stays >= 8 pt.  No text uses second-level scripts.  audit() checks the
# written files.
import matplotlib._mathtext as _mathtext
assert hasattr(_mathtext, "SHRINK_FACTOR")
_mathtext.SHRINK_FACTOR = SMALLEST_PT / NOTE_PT + 1e-9

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": NOTE_PT, "axes.labelsize": LABEL_PT, "axes.titlesize": NOTE_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT, "legend.fontsize": NOTE_PT, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "grid.linewidth": 0.4, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "legend.handlelength": 1.8, "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.3, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.0, "legend.frameon": False,
})

# ------------------------------------------------------------------------------------------ the one style table
# Every series type in every figure: colour, marker, line style, filled.  filled=True for fitted/calibrated predictors,
# False for unfitted ones.  Series that share a colour always differ in marker, fill or line style.
_V = plt.get_cmap("viridis")
STYLE = {
    # thresholds and certified intervals
    "R_glob": dict(color="#1f77b4", marker="o", ls="-", filled=True),
    "R_solve": dict(color="#d62728", marker="s", ls="--", filled=True),
    "R_glob_bnb": dict(color="#9ecae1", marker="s", ls="none", filled=True),   # unconditional B&B limit bracket
    "first_order": dict(color="k", marker="none", ls="--", filled=False),
    "crossing": dict(color="0.72", marker="_", ls="none", filled=True),       # free-training crossing distributions
    "identity": dict(color="0.35", marker="none", ls=(0, (4, 2)), filled=False),
    "criterion": dict(color="k", marker="none", ls="-", filled=True),         # registered criterion lines
    "blockE": dict(color="k", marker="*", ls="none", filled=True),
    "band": dict(color="0.9", marker="none", ls="none", filled=True),         # registered 50% bands
    # outcome classes (Phase 1)
    "solved": dict(color="#2ca02c", marker="o", ls=":", filled=True),
    "placement": dict(color="#ff7f0e", marker="s", ls="-", filled=True),
    "bias": dict(color="#e377c2", marker="^", ls="--", filled=True),
    # predictors
    "U": dict(color="#1f77b4", marker="o", ls="none", filled=False),
    "C": dict(color="#1f77b4", marker="o", ls="none", filled=True),
    "U_own": dict(color="#9467bd", marker="s", ls="none", filled=False),
    "C_own": dict(color="#9467bd", marker="s", ls="none", filled=True),
    "B1": dict(color="#8c564b", marker="^", ls="none", filled=True),
    "B2": dict(color="#bcbd22", marker="D", ls="none", filled=True),
    # mirror branches
    "branch+": dict(color="#17becf", marker="o", ls="none", filled=True),
    "branch-": dict(color="#393b79", marker="v", ls="none", filled=True),
    "branch_init": dict(color="0.55", marker="x", ls="none", filled=False),
    # fixed-scale optimiser state and horizon (4,000 steps with preserved moments is Block 4's preserved curve)
    "preserved": dict(color="k", marker="o", ls="-", filled=True),
    "reset": dict(color="0.45", marker="s", ls="--", filled=False),
    "h16000": dict(color="k", marker="D", ls=(0, (3, 1)), filled=False),
    "h64000": dict(color="k", marker="^", ls=(0, (1, 1)), filled=False),
    # conditional-search candidates (Block 1 audit)
    "cand_degenerate": dict(color="0.7", marker="x", ls="none", filled=False),
    "cand_screen": dict(color="0.45", marker=".", ls="none", filled=True),
    "cand_full": dict(color="#9ecae1", marker="o", ls="none", filled=False),
    "branch_retained": dict(color="#1f77b4", marker="none", ls="-", filled=True),
    "certified_min": dict(color="k", marker="s", ls="none", filled=False),
    "strict": dict(color="k", marker="D", ls="none", filled=False),
    "constant": dict(color="k", marker="none", ls=":", filled=False),
}
# values of a: sequential colour (a is ordered) plus a distinct marker each
for _a, _m in ((1.30, "o"), (1.35, "v"), (1.40, "<"), (1.45, "D"), (1.50, "s"), (1.60, "^")):
    STYLE[f"a{_a:.2f}"] = dict(color=matplotlib.colors.to_hex(_V(0.35 + (_a - 1.30) / 0.30 * 0.55)), marker=_m, ls="none",
                               filled=True)

# the paper's symbols, uniform across figures
RG, RS, GH = r"$R_{\mathrm{glob}}$", r"$R_{\mathrm{solve}}$", r"$\hat G$"
RDEF = r"$R=|w_2|\hat G/2$"


def sty(key, filled=None, ms=4.0, **kw):
    """Keyword arguments for plot/errorbar from the style table."""
    s = STYLE[key]
    f = s["filled"] if filled is None else filled
    out = dict(color=s["color"], marker=s["marker"], ls=s["ls"], ms=ms, mec=s["color"],
               mfc=s["color"] if f else "white")
    out.update(kw)
    return out


def handle(key, filled=None, ms=4.0, **kw):
    return Line2D([], [], **sty(key, filled, ms, **kw))


def read(name, **kw):
    """Read a results artifact and record it for the provenance check."""
    SOURCES.add(name)
    return pd.read_csv(RESULTS / name, **kw)


# ------------------------------------------------------------------------------------------ layout helpers
def title(ax, text):
    """Short panel title, left-aligned with the panel's y-axis labels (see finish)."""
    ax.set_title(text, loc="left", pad=TITLE_PAD, fontsize=NOTE_PT)
    ax._v4_head = (text, TITLE_PAD)


def right_legend(ax, handles, labels, note=None, ncol=1):
    """Legend to the right of the axes (never over data); an optional note block sits above its entries."""
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.0), ncol=ncol, title=note,
              title_fontsize=NOTE_PT, alignment="left")


def finish(fig, w_pad=1.2, h_pad=1.0, rounds=3):
    """tight_layout, then start every panel title at the left edge of that panel's y tick labels; repeat until the
    layout settles."""
    for _ in range(rounds):
        fig.tight_layout(w_pad=w_pad, h_pad=h_pad)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax in fig.axes:
            if hasattr(ax, "_v4_head"):
                bb = ax.get_window_extent(r)
                yt = [t.get_window_extent(r).x0 for t in _in_view_ticklabels(ax) if t in ax.get_yticklabels()]
                xoff = (min(yt + [bb.x0]) - bb.x0) / bb.width           # start at the y tick labels, clear of the y label
                text, pad = ax._v4_head
                ax.set_title(text, loc="left", pad=pad, fontsize=NOTE_PT, x=xoff)


def _outer(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(r) for ax in fig.axes]
    x0 = min(ax.yaxis.get_tightbbox(r).x0 for ax in fig.axes)
    return r, x0, min(b.y0 for b in boxes), max(b.y1 for b in boxes)


def _tokens(line):
    """Words of a line, never splitting inside $...$ mathtext."""
    out, cur = [], None
    for w in line.split(" "):
        cur = w if cur is None else cur + " " + w
        if cur.count("$") % 2 == 0:
            out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    return out


def _wrap(fig, text, width_px, fontsize=NOTE_PT):
    """Greedy line wrap of text (hard breaks kept) to a measured width in display pixels."""
    r = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=fontsize)
    lines = []
    for para in text.split("\n"):
        line = ""
        for tok in _tokens(para):
            cand = tok if not line else line + " " + tok
            probe.set_text(cand)
            if line and probe.get_window_extent(r).width > width_px:
                lines.append(line); line = tok
            else:
                line = cand
        lines.append(line)
    probe.remove()
    return "\n".join(lines)


def _avail(fig, x0):
    return fig.bbox.x1 - 2 * fig.dpi / 72 - x0


def legend_row(fig, handles, labels, ncol):
    """One shared legend row above all panels (call after finish); labels wrapped to the column width."""
    r, x0, _, y1 = _outer(fig)
    fs = NOTE_PT * fig.dpi / 72
    rc = plt.rcParams
    while True:
        col_w = ((_avail(fig, x0) - 2 * rc["legend.borderpad"] * fs - (ncol - 1) * rc["legend.columnspacing"] * fs) / ncol
                 - (rc["legend.handlelength"] + rc["legend.handletextpad"]) * fs - 1)
        wrapped = [_wrap(fig, lab, col_w) for lab in labels]
        fx, fy = fig.transFigure.inverted().transform((x0, y1 + 4 * fig.dpi / 72))
        leg = fig.legend(handles, wrapped, loc="lower left", bbox_to_anchor=(fx, fy), bbox_transform=fig.transFigure,
                         ncol=ncol)
        fig.canvas.draw()
        if leg.get_window_extent(r).x1 <= fig.bbox.x1 or ncol == 1:
            return leg
        leg.remove()
        ncol -= 1


def note_row(fig, text):
    """Figure-level note (n, uncertainty, verdicts) below all panels, wrapped to the text width (call after finish)."""
    r, x0, y0, _ = _outer(fig)
    fx, fy = fig.transFigure.inverted().transform((x0, y0 - 3 * fig.dpi / 72))
    fig.text(fx, fy, _wrap(fig, text, _avail(fig, x0)), ha="left", va="top", fontsize=NOTE_PT)


def _in_view_ticklabels(ax):
    out = []
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = min(lim), max(lim)
        eps = 1e-9 * max(1.0, abs(hi - lo))
        for ticks, locs in ((axis.get_major_ticks(), axis.get_majorticklocs()),
                            (axis.get_minor_ticks(), axis.get_minorticklocs())):
            for t, v in zip(ticks, locs):
                if lo - eps <= v <= hi + eps and t.label1.get_visible() and t.label1.get_text().strip():
                    out.append(t.label1)
    return out


def _data_geometry(ax):
    """Display-space sample points and filled polygons of everything drawn in the axes (clipped to the axes)."""
    pts, polys = [], []

    def sample(xy, n=24):
        xy = np.asarray(xy, float)
        if len(xy) < 2:
            return xy
        t = np.linspace(0, 1, n)[:, None]
        return np.concatenate([xy[i] + t * (xy[i + 1] - xy[i]) for i in range(len(xy) - 1)])

    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if not ln.get_visible() or len(xy) == 0:
            continue
        d = ln.get_transform().transform(np.asarray(xy, float))
        d = d[np.all(np.isfinite(d), axis=1)]
        pad = 0.5 * ln.get_markersize() * ax.figure.dpi / 72 if ln.get_marker() not in (None, "None", "none", "") else 0
        pts.append((d, pad))
        if ln.get_linestyle() not in ("None", "none", "", " "):
            pts.append((sample(d), 0.5 * ln.get_linewidth() * ax.figure.dpi / 72))
    for c in ax.collections:
        if not c.get_visible():
            continue
        if isinstance(c, matplotlib.collections.LineCollection):
            for seg in c.get_segments():
                if len(seg):
                    pts.append((sample(c.get_transform().transform(np.asarray(seg, float))), 1.0))
        elif isinstance(c, matplotlib.collections.PathCollection) and len(c.get_offsets()):
            size = np.sqrt(np.max(c.get_sizes())) if len(c.get_sizes()) else 4
            pts.append((c.get_offset_transform().transform(np.asarray(c.get_offsets(), float)),
                        0.5 * size * ax.figure.dpi / 72))
        else:
            for p in c.get_paths():
                polys.append(c.get_transform().transform_path(p))
    for p in ax.patches:
        if p.get_visible():
            polys.append(p.get_transform().transform_path(p.get_path()))
    return pts, polys


def _box_hits_data(box, ax, pts, polys):
    clip = ax.get_window_extent()
    x0, x1 = max(box.x0, clip.x0), min(box.x1, clip.x1)
    y0, y1 = max(box.y0, clip.y0), min(box.y1, clip.y1)
    if x0 >= x1 or y0 >= y1:
        return False
    for d, pad in pts:
        if len(d) and np.any((d[:, 0] >= x0 - pad) & (d[:, 0] <= x1 + pad) & (d[:, 1] >= y0 - pad) & (d[:, 1] <= y1 + pad)
                             & (d[:, 0] >= clip.x0) & (d[:, 0] <= clip.x1) & (d[:, 1] >= clip.y0) & (d[:, 1] <= clip.y1)):
            return True
    gx, gy = np.meshgrid(np.linspace(x0, x1, 16), np.linspace(y0, y1, 16))
    grid = np.c_[gx.ravel(), gy.ravel()]
    return any(p.contains_points(grid).any() for p in polys)


def layout_check(fig):
    """Problems: any legend or in-axes text over data; any two texts/legends overlapping; returns a list of strings."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    legends = [lg for lg in fig.legends] + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    legend_texts = {id(t) for lg in legends for t in lg.get_texts() + [lg.get_title()]}
    ticklabels = {id(t) for ax in fig.axes for t in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both")}
    items = []                       # (name, bbox, axes-or-None-for-data-check)
    for ax in fig.axes:
        items += [(f"tick '{t.get_text()}'", t.get_window_extent(r), None) for t in _in_view_ticklabels(ax)]
    for t in fig.findobj(Text):
        if (id(t) in legend_texts or id(t) in ticklabels or not t.get_visible() or not t.get_text().strip()
                or t.figure is None):
            continue
        ax = t.axes if (t.axes is not None and t in t.axes.texts) else None
        # an Annotation's extent includes its leader line, which touches data by design: measure the text only
        items.append((f"text '{t.get_text()[:30]}'", Text.get_window_extent(t, r), ax))
    for lg in legends:
        ax = lg.axes if lg.axes is not None else None
        items.append((f"legend '{lg.get_texts()[0].get_text()[:20]}'", lg.get_window_extent(r), "legend"))
    probs = []
    fb = fig.bbox
    for name, bb, _ in items:
        if bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5:
            probs.append(f"{name} beyond the {TEXT_W} in text width ({(bb.x1 - fb.x1) / fig.dpi:+.2f} in)")
    geo = {id(ax): _data_geometry(ax) for ax in fig.axes}
    for name, bb, ax in items:
        targets = [ax] if isinstance(ax, matplotlib.axes.Axes) else (fig.axes if ax == "legend" else [])
        for a in targets:
            if _box_hits_data(bb, a, *geo[id(a)]):
                probs.append(f"{name} over data")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
                probs.append(f"{items[i][0]} overlaps {items[j][0]}")
    return probs


def save(fig, name):
    LAYOUT[name] = layout_check(fig)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  ok  {name}.pdf" + (f"  LAYOUT: {LAYOUT[name]}" if LAYOUT[name] else ""))


# ------------------------------------------------------------------------------------------ summaries
def boot_median(v, n=10_000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(v, (n, len(v))), axis=1)
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_mean(v, n=10_000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_spearman(x, y, n=10_000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    r = [pd.Series(x[i]).rank().corr(pd.Series(y[i]).rank()) for i in idx]
    return float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


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


def _ghat(a):
    """Certified G_hat of the base window (the value R uses throughout; own_threshold._pop)."""
    from .ghat_rigorous import ghat_R_of            # the rigorous Ĝ_cert (Block 2, author's decision 2026-09-24)
    return ghat_R_of(a)


# ------------------------------------------------------------------------------------------ Phase 1
def fig_decomposition():
    """Every failure is a placement failure (G <= 0) or a bias failure (G > 0, b2 outside its interval)."""
    d = read("phase1_decomposition.csv")
    rec = read("wi_phase1_reconciliation.csv").set_index("precision")
    tot = d.failure.value_counts()
    both = rec.loc["both"]
    assert (len(d), tot["solved"], tot["placement"], tot["bias"]) == (both.runs, both.solved, both.placement, both.bias)
    assert both.disagreements == 0
    g = d.groupby("a")
    eps = np.array(sorted(d.a.unique())) - 1
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.55))
    ns = g.size().values
    hs, ls = [], []
    for cls, dx, lab in (("placement", 0.97, r"placement failure ($G\leq0$)"),
                         ("bias", 1.0, "bias failure ($G>0$,\n$b_2$ outside its interval)"),
                         ("solved", 1.03, "solved")):
        k = g.failure.apply(lambda f: int((f == cls).sum())).values
        fr = k / ns
        ci = np.array([clopper(int(kk), int(nn)) for kk, nn in zip(k, ns)])
        ax.errorbar(eps * dx, fr, yerr=[fr - ci[:, 0], ci[:, 1] - fr], capsize=1.5, elinewidth=0.6, **sty(cls, ms=3.5))
        hs.append(handle(cls, ms=3.5)); ls.append(f"{lab}: {int(k.sum()):,} runs")
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$\varepsilon=a-1$ (log scale)")
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    n_each = int(ns.min()) if ns.min() == ns.max() else f"{ns.min()}–{ns.max()}"
    title(ax, "Phase 1: every failure is a placement or a bias failure")
    right_legend(ax, hs, ls, note=f"$n={n_each}$ runs per $a$ (200 seeds $\\times$ 2 precisions);\n"
                                  f"{len(d):,} runs, 0 disagreements with the\nsolve check; bars: Clopper–Pearson 95%")
    finish(fig)
    save(fig, "v4_decomposition")
    return {"n_per_a": n_each, "n_total": len(d), "solved": int(tot["solved"]), "placement": int(tot["placement"]),
            "bias": int(tot["bias"])}


# ------------------------------------------------------------------------------------------ Block 3
def fig_prospective():
    from .prospective import _cert_ghat
    SOURCES.add("prospective_ghat.csv")
    gh = _cert_ghat()
    runs = read("prospective_runs.csv")
    runs = runs[runs.cross_step.notna()].copy()
    runs["R"] = [w2 * gh[(w, round(a, 2))] / 2 for w, a, w2 in zip(runs.window, runs.a, runs.cross_w2)]
    sc = read("prospective_scores.csv")
    cal = read("prospective_calibration.csv").set_index("a")
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.9))
    models = [("C", r"C $=\lambda(a)\,R_{\mathrm{glob}}$ ($\lambda$ fitted)"),
              ("U", r"U $=R_{\mathrm{glob}}$ (nothing fitted)"),
              ("B1", r"B1: base window's crossing $|w_2|$"),
              ("B2", r"B2: pooled crossing $R$")]
    ns = []
    for r in sc.itertuples():
        g = runs[(runs.window == r.window) & (runs.a.round(2) == round(r.a, 2))]
        med, lo, hi = boot_median(g.R.values)
        ns.append(len(g))
        for m, _ in models:
            ax.errorbar(med, getattr(r, m), xerr=[[med - lo], [hi - med]], ecolor="0.6", elinewidth=0.6, capsize=1.5,
                        zorder=3 if m == "C" else 2, **sty(m, ms=4.5 if m in ("C", "U") else 4))
        ax.plot([med, med], [r.U, r.C], color=STYLE["C"]["color"], lw=0.6, zorder=1)      # λ scales U to C
    lim = [0.13, 0.33]
    ax.plot(lim, lim, lw=0.7, zorder=1, **sty("identity"))
    ax.set_xlim(0.195, 0.285); ax.set_ylim(*lim)
    ax.set_xlabel(r"observed median crossing $R$ (bootstrap 95%)")
    ax.set_ylabel(r"predicted crossing $R$")
    hs = [handle(m, ms=4.5) for m, _ in models] + [Line2D([], [], color=STYLE["C"]["color"], lw=0.6),
                                                   Line2D([], [], lw=0.7, **sty("identity"))]
    ls = [lab for _, lab in models] + [r"U$\to$C: fitted lag factor $\lambda$", "identity"]
    title(ax, "Block 3: held-out windows, per setting")
    right_legend(ax, hs, ls, note=f"8 held-out settings, $n={min(ns)}$–${max(ns)}$\ncrossings of 90 each; "
                                  f"$\\lambda$ fitted on the\nbase window: $\\lambda(1.30)={cal.loc[1.3, 'lambda_fitted']:.3f}$,\n"
                                  f"$\\lambda(1.50)={cal.loc[1.5, 'lambda_fitted']:.3f}$")
    finish(fig)
    save(fig, "v4_prospective")
    return {"n_min": min(ns), "n_max": max(ns)}


# ------------------------------------------------------------------------------------------ own-seed prospective test
def fig_prospective_own():
    """Registered own-seed test: per run, observed crossing R against the run's own threshold (U_own, nothing fitted),
    and each model's mean per-run |log error| (per-window mean, then mean over the 8 windows, as registered)."""
    pr = read("prospective_own_predictions.csv", float_precision="round_trip")
    rn = read("prospective_own_runs.csv", float_precision="round_trip")
    sc = read("prospective_own_scores.csv")
    ss = read("prospective_own_settings_scored.csv")
    d = rn.merge(pr, on=["window", "a", "seed"])
    d["R_cross"] = d.cross_w2 * d.Ghat_lo / 2                          # as registered (prospective_own.score)
    x = d[d.R_cross.notna()].copy()
    x["own_lo"], x["own_hi"] = x.w2_own_lo * x.Ghat_lo / 2, x.w2_own_hi * x.Ghat_lo / 2
    prim = sc[sc.analysis == "primary (crossers)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 1.75), gridspec_kw={"width_ratios": [1.2, 1]})
    info = {}
    lim = [0.15, 0.36]
    hs, ls = [], []
    ncross, rho_res_all = {}, []
    for a in (1.3, 1.5):
        k = f"a{a:.2f}"
        g = x[x.a.round(2) == a]
        a1.errorbar(g.U_own, g.R_cross, xerr=[g.U_own - g.own_lo, g.own_hi - g.U_own], elinewidth=0.4, alpha=0.8,
                    zorder=2, ecolor=STYLE[k]["color"], **sty(k, filled=False, ms=2.4, mew=0.45))
        hs.append(handle(k, filled=False, ms=3.5)); ls.append(f"U$_{{\\mathrm{{own}}}}$ per run, $a={a:.2f}$")
        rho_res = float(np.median(g.C_own / g.U_own))
        assert np.allclose(g.C_own / g.U_own, rho_res, rtol=1e-12)
        dash = (0, (5, 1.5)) if a == 1.3 else (0, (2, 1))       # line style differs by a, not colour alone
        a1.plot(lim, [rho_res * v for v in lim], color=STYLE[k]["color"], lw=0.9, ls=dash, zorder=3)
        hs.append(Line2D([], [], color=STYLE[k]["color"], lw=0.9, ls=dash))
        ls.append(f"C$_{{\\mathrm{{own}}}}$, $a={a:.2f}$ (fitted)")
        rho_res_all.append(f"{rho_res:.4f} ($a={a:.2f}$)")
        info[f"n_{a}"] = len(g)
        ncross[a] = (len(g), len(d[d.a.round(2) == a]))
    a1.plot(lim, lim, lw=0.7, zorder=1, **sty("identity"))
    s13, s15 = ss[ss.a.round(2) == 1.3].spearman_R_cross_vs_R_own, ss[ss.a.round(2) == 1.5].spearman_R_cross_vs_R_own
    wmax = float((x.own_hi - x.own_lo).max())
    a1.set_xlim(*lim); a1.set_ylim(*lim)
    a1.set_xlabel(r"own threshold $R_{\mathrm{own}}$ (frozen before training)")
    a1.set_ylabel(r"observed crossing $R$ of the run")
    title(a1, "(a) own-seed test, per run")
    # (b) mean per-run |log error| per model, window-level bootstrap as registered
    models = [("U", "U (nothing fitted)"), ("C", "C (fitted)"), ("U_own", "U$_{\\mathrm{own}}$ (nothing fitted)"),
              ("C_own", "C$_{\\mathrm{own}}$ (fitted)")]
    verdicts = []
    for i, a in enumerate((1.3, 1.5)):
        g = x[x.a.round(2) == a]
        lev = {}
        for j, (m, lab) in enumerate(models):
            per_w = g.groupby("window").apply(lambda h: float(np.abs(np.log(h.R_cross / h[m])).mean()),
                                              include_groups=False)
            mean, lo, hi = boot_mean(per_w.values)
            lev[m] = mean
            a2.errorbar([i + (j - 1.5) * 0.17], [mean], yerr=[[mean - lo], [hi - mean]], ecolor=STYLE[m]["color"],
                        elinewidth=0.7, capsize=1.5, **sty(m, ms=5))
        p1 = prim[(prim.a.round(2) == a) & (prim.comparison == "P1")].stat.iloc[0]
        p3 = prim[(prim.a.round(2) == a) & (prim.comparison == "P3")].stat.iloc[0]
        assert abs((lev["U_own"] - lev["U"]) - p1) < 1e-12 and abs((lev["C_own"] - lev["C"]) - p3) < 1e-12
        v = prim[prim.a.round(2) == a]
        verdicts.append(", ".join(f"{r.comparison}" for _, r in v.iterrows()) + f" at $a={a:.2f}$"
                        + (": all PASS" if v["pass"].all() else ": see scores"))
    a2.set_xticks([0, 1]); a2.set_xticklabels(["$a=1.30$", "$a=1.50$"])
    a2.set_xlim(-0.5, 1.5); a2.set_ylim(0, 0.2)
    a2.set_ylabel(r"mean per-run $|\log(R_{\mathrm{cross}}/\mathrm{pred})|$")
    title(a2, "(b) per-run error by model (crossers)")
    finish(fig, w_pad=1.5)
    legend_row(fig, hs + [Line2D([], [], lw=0.7, **sty("identity"))] + [handle(m, ms=5) for m, _ in models],
               ls + ["identity"] + [lab for _, lab in models], ncol=3)
    crossed = (f"$n={ncross[1.3][0]}$ of {ncross[1.3][1]} crossed at each $a$" if ncross[1.3] == ncross[1.5] else
               f"$n={ncross[1.3][0]}$ of {ncross[1.3][1]} ($a=1.30$) and {ncross[1.5][0]} of {ncross[1.5][1]} "
               f"($a=1.50$) crossed")
    note_row(fig, f"(a) 16 never-trained settings (8 windows $\\times$ 2 $a$), 60 runs each; {crossed}; $x$ bars: frozen bracket of $R_{{\\mathrm{{own}}}}$ ($\\leq${wmax:.1e}); "
                  f"per-setting Spearman $\\rho$: {s13.min():.2f}–{s13.max():.2f} ($a=1.30$), "
                  f"{s15.min():.2f}–{s15.max():.2f} ($a=1.50$).\n"
                  f"U $=R_{{\\mathrm{{glob}}}}$ (population); C $=\\lambda(a)R_{{\\mathrm{{glob}}}}$, $\\lambda$ fitted on the "
                  f"base window (Block 3); U$_{{\\mathrm{{own}}}}=R_{{\\mathrm{{own}}}}$ (own threshold); "
                  f"C$_{{\\mathrm{{own}}}}=\\rho_{{\\mathrm{{res}}}}(a)R_{{\\mathrm{{own}}}}$, fitted $\\rho_{{\\mathrm{{res}}}}$ = "
                  f"{', '.join(rho_res_all)}.\n(b) 8 windows per $a$; window-level bootstrap 95% (10,000 resamples, seed 0); filled: fitted, "
                  f"open: nothing fitted. Registered: {verdicts[0]}; {verdicts[1]}.")
    save(fig, "v4_prospective_own")
    return info


# ------------------------------------------------------------------------------------------ Blocks 4 and 5
def _block_e_rglob():
    SOURCES.add("blockE_results.md")
    m = re.search(r"`R_glob = (0\.\d+)`", (RESULTS / "blockE_results.md").read_text())
    return float(m.group(1))


def fig_fixed_scale(horizons=True):
    br = read("cond_certified_brackets.csv")
    r13 = br[(br.a.round(2) == 1.3) & (br.kind == "glob")].iloc[0]
    R_cert = 0.5 * (r13.R_lo + r13.R_hi)
    e2_x = 1.15 * _block_e_rglob() / R_cert      # Block E held 1.15 x its frozen R_glob, in certified units
    e2 = read("wi_e2_rescore.csv").set_index("arm").loc["hold_high"]
    kept, n_e = int(e2.kept), int(e2.intervened)
    if horizons:
        fig = plt.figure(figsize=(TEXT_W, 1.72))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85])
        a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharex=a1, sharey=a1); a3 = fig.add_subplot(gs[2])
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 1.95), sharex=True, sharey=True)
    notes = []
    for ax, blk, band, ylab in ((a1, 4, (0.9, 1.25), "placed at 4k steps"),
                                (a2, 5, (0.9, 1.1), "retained to 12k steps")):
        cv = read(f"fixed_scale_block{blk}_curve.csv")
        ts = read(f"fixed_scale_block{blk}_tests.csv").set_index("variant")
        ax.axvspan(*band, color=STYLE["band"]["color"], zorder=0, lw=0)
        ax.axvline(1.0, color="0.5", lw=0.6, ls=":")
        for var, dx in (("preserved", -0.006), ("reset", 0.006)):
            g = cv[cv.variant == var].sort_values("level")
            ax.errorbar(g.level + dx, g.frac, yerr=[g.frac - g.ci95_lo, g.ci95_hi - g.frac], capsize=1.5,
                        elinewidth=0.6, **sty(var, ms=3.5))
        n = int(cv.n.max())
        ax.set_ylabel(ylab, labelpad=2)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xticks([0.6, 1.0, 1.5, 2.0])
        ax.set_xlabel(r"held $R/R_{\mathrm{glob}}$")
        title(ax, f"({'a' if blk == 4 else 'b'}) Block {blk}: " + ("before placement" if blk == 4 else "after placement"))
        notes.append(f"({'a' if blk == 4 else 'b'}) $n={n}$ {'checkpoints' if blk == 4 else 'runs'} per level, "
                     f"x$_{{50}}$ = {ts.loc['preserved', 'x50']:.3f} (preserved), {ts.loc['reset', 'x50']:.3f} (reset), "
                     f"shaded: registered 50% band {band[0]}–{band[1]}")
        if blk == 5:
            lo, hi = clopper(kept, n_e)
            # E-2: Block E's registered criterion, retention >= 0.9 at 1.15 x R_glob (frozen), and its observed kept/n
            ax.plot([e2_x - 0.05, e2_x + 0.05], [0.9, 0.9], color=STYLE["criterion"]["color"], lw=1.6, zorder=4)
            ax.errorbar([e2_x], [kept / n_e], yerr=[[kept / n_e - lo], [hi - kept / n_e]], capsize=1.5,
                        elinewidth=0.6, zorder=5, **sty("blockE", ms=6))
    hs = [handle("preserved", ms=3.5), handle("reset", ms=3.5), Line2D([], [], color="k", lw=1.6),
          handle("blockE", ms=6)]
    ls = ["moments preserved (4k in c)", "moments reset", "E-2 criterion", f"Block E: {kept}/{n_e} kept, FAIL"]
    out = {"e2_x": e2_x}
    if horizons:
        hz = read("fixed_scale_horizons.csv", float_precision="round_trip")
        hc = read("fixed_scale_horizons_curve.csv").set_index(["variant", "horizon"])
        ht = read("fixed_scale_horizons_tests.csv").set_index("variant")
        hz = hz[hz.variant == "preserved"]
        a3.axvline(1.0, color="0.5", lw=0.6, ls=":")
        ns = set()
        x50s = []
        for H, key, dx in ((4000, "preserved", -0.008), (16000, "h16000", 0.0), (64000, "h64000", 0.008)):
            g = hz.groupby("level")[f"placed_{H}"]
            k, n = g.sum().astype(int), g.size()
            ns.update(n.values.tolist())
            fr = k / n
            for lv, f in fr.items():                        # the committed curve must be these counts
                assert abs(f - hc.loc[("preserved", H), f"frac_{lv:g}" if lv != 1 else "frac_1.0"]) < 1e-12
            ci = np.array([clopper(int(kk), int(nn)) for kk, nn in zip(k, n)])
            x50 = hc.loc[("preserved", H), "x50"]
            a3.errorbar(fr.index + dx, fr.values, yerr=[fr.values - ci[:, 0], ci[:, 1] - fr.values], lw=0.7,
                        capsize=1.5, elinewidth=0.6, **sty(key, ms=3.5))
            if H != 4000:
                hs.append(handle(key, ms=3.5, lw=0.7)); ls.append(f"{H:,} steps")
            x50s.append(f"{x50:.4f} ({H // 1000}k)")
        q1, q2 = ht.loc["preserved", "Q1_pass"], ht.loc["preserved", "Q2_pass"]
        a3.set_xlim(0.87, 1.13); a3.set_ylim(-0.04, 0.8)
        a3.set_xticks([0.9, 1.0, 1.1])
        a3.set_xlabel(r"held $R/R_{\mathrm{glob}}$")
        a3.set_ylabel("placed (preserved)", labelpad=2)
        title(a3, "(c) horizon extension")
        notes.append(f"(c) Block 4 horizon extension, $n={min(ns)}$ replays per level, x$_{{50}}$ = "
                     + ", ".join(x50s) + f"; registered Q1 {'PASS' if q1 else 'FAIL'}, Q2 {'PASS' if q2 else 'FAIL'}")
        out["horizon_n"] = min(ns)
    finish(fig, w_pad=1.0)
    legend_row(fig, hs, ls, ncol=3)
    note_row(fig, ";\n".join(notes) + ".\n"
                  f"E-2 criterion (Block E, registered): kept $\\geq$ 0.9 at 1.15$\\times$ Block E's {RG} = {e2_x:.3f} "
                  f"certified units; Block E observed {kept}/{n_e} kept (E-2 FAIL). Adam moments preserved or reset; "
                  f"(c) preserved only. $a=1.30$, certified {RG}; bars: Clopper–Pearson 95%.")
    save(fig, "v4_fixed_scale")
    return out


# ------------------------------------------------------------------------------------------ certified thresholds
def fig_thresholds():
    br = read("cond_certified_brackets.csv")
    kb = read("limit_K_base.csv").iloc[0]
    c1 = read("first_order_c1.csv").iloc[0]
    fo = read("first_order_finite.csv")
    fs = read("first_order_scores.csv").set_index("quantity")
    sl = read("mn2_solve_limit.csv").iloc[0]
    cv = read("certv2_solve_summary.csv").iloc[0]
    assert bool(cv.covered) and cv.sub_intervals_certified == cv.sub_intervals_evaluated   # R_solve^inf is global
    assert bool(c1.switch_krawczyk_ok) and kb.R_glob_inf_lo <= c1.R_glob_inf_lo <= c1.R_glob_inf_hi <= kb.R_glob_inf_hi
    runs = read("wi_crossing_runs.csv")
    runs = runs[runs.budget == 32_000]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT_W, 1.8), gridspec_kw={"width_ratios": [1.45, 1]})
    eps = sorted(br.a.round(2).unique() - 1)
    w = 0.018
    ns = []
    for e in eps:
        g = runs[runs.a.round(2) == round(1 + e, 2)].R_cert.values
        ns.append(len(g))
        vp = ax.violinplot([g], positions=[e], widths=0.035, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(STYLE["crossing"]["color"]); b.set_edgecolor("0.5"); b.set_linewidth(0.4); b.set_alpha(1)
        med, lo, hi = boot_median(g)
        ax.errorbar([e], [med], yerr=[[med - lo], [hi - med]], fmt="_", color="0.2", ms=6, capsize=1.5, elinewidth=0.7)
    for kind in ("glob", "solve"):
        s = STYLE[f"R_{kind}"]
        g = br[br.kind == kind].sort_values("a")
        for r in g.itertuples():
            ax.add_patch(plt.Rectangle((r.a - 1 - w / 2, r.R_lo), w, r.R_hi - r.R_lo, color=s["color"], lw=0))
            ax.plot([r.a - 1], [0.5 * (r.R_lo + r.R_hi)], marker=s["marker"], color=s["color"], ms=3, ls="none")
        ax.plot(g.a - 1, 0.5 * (g.R_lo + g.R_hi), color=s["color"], lw=0.6, ls=s["ls"])
    # limit (ε = 0): unconditional B&B bracket (light), sharp certified value (line), global R_solve^inf
    ax.add_patch(plt.Rectangle((-w / 2, kb.R_glob_inf_lo), w, kb.R_glob_inf_hi - kb.R_glob_inf_lo,
                               color=STYLE["R_glob_bnb"]["color"], lw=0))
    ax.plot([-w / 2, w / 2], [c1.R_glob_inf_lo] * 2, color=STYLE["R_glob"]["color"], lw=1.2)
    ax.add_patch(plt.Rectangle((-w / 2, sl.R_solve_inf_lo), w, sl.R_solve_inf_hi - sl.R_solve_inf_lo,
                               color=STYLE["R_solve"]["color"], lw=0))
    ax.text(0.03, 0.262, r"limit $\varepsilon\to0$" + "\n(certified, global)", va="center")
    for yv in (kb.R_glob_inf_hi, sl.R_solve_inf_lo):
        ax.annotate("", (w / 2 + 0.002, yv), xytext=(0.028, 0.262), textcoords="data",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4", shrinkA=0, shrinkB=0))
    ax.add_patch(plt.Rectangle((-0.006, 0.1955), 0.054, 0.0075, fill=False, lw=0.5, ls=":", ec="k"))
    ax.text(0.052, 0.1972, "zoom: (b)", va="center")
    nlab = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}$–${max(ns)}"
    ax.set_xlim(-0.04, 0.64)
    ax.set_xlabel(r"$\varepsilon=a-1$")
    ax.set_ylabel(RDEF)
    title(ax, "(a) certified thresholds and free-training crossings")
    # (b) small ε: the registered c1 test's brackets and the first-order line (certified range |ε| <= 0.05)
    wb = 0.0028
    bx.add_patch(plt.Rectangle((-wb / 2, kb.R_glob_inf_lo), wb, kb.R_glob_inf_hi - kb.R_glob_inf_lo,
                               color=STYLE["R_glob_bnb"]["color"], lw=0))
    bx.plot([-wb / 2 - 0.0008, wb / 2 + 0.0008], [0.5 * (c1.R_glob_inf_lo + c1.R_glob_inf_hi)] * 2,
            color=STYLE["R_glob"]["color"], lw=1.6)
    for r in fo.itertuples():
        bx.add_patch(plt.Rectangle((r.eps - wb / 2, r.R_lo), wb, r.R_hi - r.R_lo, color=STYLE["R_glob"]["color"], lw=0))
    ee = np.linspace(0, 0.05, 51)
    corners = [c1.R_glob_inf_lo * (1 + c * ee) for c in (c1.c1_lo, c1.c1_hi)] + \
              [c1.R_glob_inf_hi * (1 + c * ee) for c in (c1.c1_lo, c1.c1_hi)]
    bx.fill_between(ee, np.min(corners, axis=0), np.max(corners, axis=0), color="k", lw=0)
    c1m = 0.5 * (c1.c1_lo + c1.c1_hi)
    bx.plot(ee, 0.5 * (c1.R_glob_inf_lo + c1.R_glob_inf_hi) * (1 + c1m * ee), lw=0.8, **sty("first_order"))
    f = fs.loc["c1 (primary): R_glob"]
    bx.set_xlim(-0.004, 0.048)
    bx.set_ylim(0.1970, 0.2016)
    bx.set_xticks([0, 0.02, 0.04])
    bx.set_xlabel(r"$\varepsilon=a-1$")
    bx.set_ylabel(RG)
    title(bx, r"(b) small $\varepsilon$ and the registered $c_1$ test")
    finish(fig, w_pad=1.2)
    legend_row(fig,
               [Patch(color=STYLE["R_glob"]["color"], lw=0),
                Line2D([], [], color=STYLE["R_glob"]["color"], marker="o", ls="-", ms=3, lw=0.6),
                Line2D([], [], color=STYLE["R_solve"]["color"], marker="s", ls="--", ms=3, lw=0.6),
                Patch(color=STYLE["R_glob_bnb"]["color"], lw=0), Line2D([], [], color=STYLE["R_glob"]["color"], lw=1.6),
                Line2D([], [], lw=0.8, **sty("first_order")),
                Patch(facecolor=STYLE["crossing"]["color"], edgecolor="0.5", lw=0.4)],
               [f"certified {RG} (boxes)", f"{RG}, $a=1.30$–$1.60$", f"{RS} (certified)",
                r"$R^{\infty}_{\mathrm{glob}}$ B&B bracket", r"$R^{\infty}_{\mathrm{glob}}$ sharp",
                f"first order, $c_1={c1m:.4f}$", "crossing $R$ (budget 32k)"], ncol=3)
    note_row(fig, f"(a) free-training crossings, $n={nlab}$ runs per $a$ (violins; medians with bootstrap 95%); boxes: "
                  f"certified intervals to scale; limit $R^{{\\infty}}_{{\\mathrm{{solve}}}}$ global.\n(b) $n$ = 4 "
                  f"certified brackets ($a=1.01$–$1.04$) and the limit (no runs); unconditional B&B bracket of "
                  f"$R^{{\\infty}}_{{\\mathrm{{glob}}}}$ and sharp value [{c1.R_glob_inf_lo:.7f}, {c1.R_glob_inf_hi:.7f}]; "
                  f"first order $R^{{\\infty}}(1+c_1\\varepsilon)$, $c_1$ certified; first-order range certified for "
                  f"$|\\varepsilon|\\leq0.05$; registered $c_1$ test: {f.verdict.split(' ')[0]}: feasible "
                  f"$c_1\\in$ [{f.feasible_lo:.3f}, {f.feasible_hi:.3f}], width {f.feasible_width:.3f} > 0.1.")
    save(fig, "v4_thresholds")
    return {"n_min": min(ns), "n_max": max(ns)}


# ------------------------------------------------------------------------------------------ Block 1 audit
def fig_cond_candidates(a=1.30):
    """Every candidate the frozen conditional search evaluates at one a, the retained branch, the certified scan, and the
    gap of the minimiser changing sign once along one continuous branch.  a = 1.30: the a of the certified scan."""
    d = read("cond_audit_candidates.csv")
    g = d[d.a.round(2) == round(a, 2)]
    st = read("cond_audit_strict.csv")
    st = st[st.a.round(2) == round(a, 2)]
    scan = read("cond_scan_certified_a130.csv") if round(a, 2) == 1.3 else None
    br = read("cond_certified_brackets.csv")
    br = br[br.a.round(2) == round(a, 2)].set_index("kind")
    ret = g[g.status == "RETAINED"].drop_duplicates("s").sort_values("s")
    assert g.frozen_reproduced.all()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 1.85), sharex=True)
    hs, ls = [], []
    for sts, key, ms, lab in ((("degenerate",), "cand_degenerate", 2.6, "degenerate (discarded)"),
                              (("not carried (screen rank > 8)", "carried"), "cand_screen", 2.4, "screening runs"),
                              (("not lowest",), "cand_full", 2.8, "full runs, not lowest")):
        h = g[g.status.isin(sts)]
        a1.plot(h.s, h.loss, zorder=1, mew=0.5, **sty(key, ms=ms))
        hs.append(handle(key, ms=4, mew=0.6)); ls.append(f"{lab}: {len(h):,}")
    a1.plot(ret.s, ret.loss, lw=1.1, zorder=3, **sty("branch_retained"))
    a1.axhline(np.log(2), lw=0.8, zorder=2, **sty("constant"))
    if scan is not None:
        mmin = np.minimum(scan.m_minus_lo, scan.m_plus_lo)
        wmax = float(np.maximum(scan.m_minus_hi - scan.m_minus_lo, scan.m_plus_hi - scan.m_plus_lo).max())
        a1.plot(scan.s, mmin, zorder=4, mew=0.6, **sty("certified_min", ms=3.2))
    hs += [Line2D([], [], lw=1.1, **sty("branch_retained")), Line2D([], [], lw=0.8, **sty("constant")),
           handle("certified_min", ms=3.5, mew=0.6), handle("strict", ms=3.5, mew=0.6),
           Line2D([], [], color=STYLE["R_glob"]["color"], lw=0.9, ls=STYLE["R_glob"]["ls"]),
           Line2D([], [], color=STYLE["R_solve"]["color"], lw=0.9, ls=STYLE["R_solve"]["ls"])]
    ls += [f"retained branch ({len(ret)} scales)", "constant predictor ($\\log 2$)",
           f"certified global min. ({len(scan)})", f"stricter optimisation, 1e ({len(st)})",
           f"certified {RG} bracket", f"certified {RS} bracket"]
    a1.set_yscale("log")
    a1.set_ylim(0.17, 25)
    a1.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20])
    a1.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_ylabel("conditional loss")
    a1.set_xlabel(f"output scale $|w_2|$ ($a={a:.2f}$)")
    a1.tick_params(axis="x", labelbottom=True)
    nk = g.groupby(["kind", "s"]).ngroups
    # (b) gap of the retained minimiser: one sign change, one continuous branch
    a2.axhline(0, color="0.5", lw=0.6)
    for kind in ("glob", "solve"):
        r = br.loc[kind]
        a2.axvspan(r.w2_lo, r.w2_hi, color=STYLE[f"R_{kind}"]["color"], lw=0, zorder=0)
        a2.axvline(0.5 * (r.w2_lo + r.w2_hi), color=STYLE[f"R_{kind}"]["color"], lw=0.9, ls=STYLE[f"R_{kind}"]["ls"],
                   zorder=0)
    a2.plot(ret.s, ret.gap, lw=1.1, zorder=2, **sty("branch_retained"))
    a2.plot(st.s, st.gap, zorder=4, mew=0.6, **sty("strict", ms=3.2))
    if scan is not None:
        a2.plot(scan.s, scan.argmin_G, zorder=3, mew=0.6, **sty("certified_min", ms=3.2))
        cm = float(scan.competitor_margin.min())
    sep = float(max(br.loc["glob", "argmin_separation_lo"], br.loc["glob", "argmin_separation_hi"]))
    a2.set_xlabel(f"output scale $|w_2|$ ($a={a:.2f}$)")
    a2.set_ylabel(r"minimiser gap $G$")
    a2.set_ylim(-0.5, 0.12)
    title(a1, f"(a) every candidate of the frozen search")
    title(a2, "(b) the branch crosses $G=0$ continuously")
    finish(fig, w_pad=1.2)
    legend_row(fig, hs, ls, ncol=3)
    note_row(fig, f"(a) $a={a:.2f}$: $n=${len(g):,} candidates at {nk} evaluations; frozen result reproduced "
                  f"{nk}/{nk}; none below the retained branch (full runs not lowest lie on it); certified widths "
                  f"$\\leq${wmax:.0e}.\n(b) certified brackets: {RG} at $|w_2|\\in$ ({br.loc['glob', 'w2_lo']:.4f}, "
                  f"{br.loc['glob', 'w2_hi']:.4f}], {RS} at ({br.loc['solve', 'w2_lo']:.4f}, "
                  f"{br.loc['solve', 'w2_hi']:.4f}]; "
                  f"$G$ changes sign once; minimisers either side of the switch $\\leq${sep:.1e} apart (certified); "
                  f"other basins $\\geq$ +{cm:.4f} above (certified, radius 0.1).")
    save(fig, "v4_cond_candidates")
    return {"a": a, "n_candidates": len(g), "evaluations": nk, "scan": 0 if scan is None else len(scan), "strict": len(st)}


# ------------------------------------------------------------------------------------------ mirror branches (post hoc)
def fig_mirror_branches():
    """Post hoc: free-training crossing R (budget 32k) against the own threshold of the mirror branch the run occupies at
    the crossing; the initialisation-selected branch's threshold for contrast.  rho: mirror_s3.csv, spearman_rho."""
    occ = read("mirror_occupancy.csv", float_precision="round_trip")
    T = read("mirror_branch_thresholds.csv", float_precision="round_trip")
    s3 = read("mirror_s3.csv").set_index(["a", "threshold"])
    T = T[T.group == "cross"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 1.95))
    out = {}
    counts = {}
    notes = []
    for ax, a, tag in zip(axes, (1.3, 1.5), "ab"):
        G = _ghat(a)
        o = occ[occ.group == f"phase 2b crossing a={a:.2f}"].merge(T[T.a.round(2) == a][["seed", "T_plus", "T_minus"]],
                                                                      on="seed")
        o["R_occ"] = np.where(o.branch > 0, o.T_plus, o.T_minus) * G / 2
        o["R_init"] = np.where(o.init_branch > 0, o.T_plus, o.T_minus) * G / 2
        rho = s3.loc[(a, "branch occupied at the crossing")]
        rho_i = s3.loc[(a, "branch selected by initialisation")]
        rho_g = s3.loc[(a, "registered: own global threshold")]
        rho_chk = o.R_cross.rank().corr(o.R_occ.rank())
        assert abs(rho_chk - rho.spearman_rho) < 1e-12 and len(o) == rho.n
        lo, hi = boot_spearman(o.R_occ, o.R_cross)
        allv = np.r_[o.R_cross, o.R_occ, o.R_init]
        lim = [allv.min() - 0.006, allv.max() + 0.006]
        ax.plot(lim, lim, lw=0.7, zorder=1, **sty("identity"))
        ax.plot(o.R_init, o.R_cross, zorder=2, mew=0.6, **sty("branch_init", ms=3.5))
        for b, key in ((1, "branch+"), (-1, "branch-")):
            h = o[o.branch == b]
            counts[key] = counts.get(key, set()) | {len(h)}
            ax.plot(h.R_occ, h.R_cross, zorder=3, mew=0.3, **sty(key, ms=3.8))
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_ylabel("crossing $R$")
        ax.set_xlabel(r"own threshold $T$ of the branch ($R$ units)")
        title(ax, f"({tag}) $a={a:.2f}$: Spearman $\\rho={rho.spearman_rho:.3f}$ [{lo:.4f}, {hi:.4f}]")
        notes.append(f"({tag}) $n={len(o)}$ runs; initialisation branch $\\rho={rho_i.spearman_rho:.2f}$; "
                     f"own global threshold $\\rho={rho_g.spearman_rho:.2f}$; median residual $\\log(R/T)$ = "
                     f"{rho.residual_median_log:.3f}")
        out[a] = {"spearman": float(rho.spearman_rho), "ci": (lo, hi), "n": len(o)}
    finish(fig, w_pad=1.5)
    cnt = {k: "/".join(str(c) for c in sorted(v)) for k, v in counts.items()}
    legend_row(fig, [handle("branch+", ms=3.8), handle("branch-", ms=3.8), handle("branch_init", ms=3.8, mew=0.6),
                     Line2D([], [], lw=0.7, **sty("identity"))],
               [f"branch + at the crossing ({cnt['branch+']} runs per $a$)",
                f"branch $-$ at the crossing ({cnt['branch-']} runs per $a$)",
                "same runs, initialisation-selected branch's $T$", "identity"], ncol=2)
    note_row(fig, "Post hoc, exploratory. Free-training crossings, budget 32k; $T$ as $R=|w_2|\\hat G/2$; "
                  "titles: $\\rho$ with the branch occupied at the crossing, bootstrap 95%.\n" + ";\n".join(notes) + ".")
    save(fig, "v4_mirror_branches")
    return out


# ------------------------------------------------------------------------------------------ audit and provenance
def _pdf_text_runs(pdf):
    """Every text run in a matplotlib PDF: (effective font size in pt, bbox in pt) from the content stream, glyph advance
    widths from each Type0 font's /W array.  Height: descent 0.25 size, ascent 0.9 size."""
    import pypdf
    from pypdf.generic import ContentStream

    def mul(m, n):
        return [m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3], m[2] * n[0] + m[3] * n[2],
                m[2] * n[1] + m[3] * n[3], m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5]]

    rd = pypdf.PdfReader(str(pdf))
    page = rd.pages[0]
    widths = {}
    for k, f in page["/Resources"]["/Font"].items():
        f = f.get_object()
        w = {}
        dfont = f["/DescendantFonts"][0].get_object() if "/DescendantFonts" in f else f
        dw = float(dfont.get("/DW", 1000))
        arr = list(dfont.get("/W", []))
        i = 0
        while i < len(arr):
            c = int(arr[i])
            if isinstance(arr[i + 1], list) or hasattr(arr[i + 1], "__iter__"):
                for j, v in enumerate(arr[i + 1]):
                    w[c + j] = float(v)
                i += 2
            else:
                for cc in range(c, int(arr[i + 1]) + 1):
                    w[cc] = float(arr[i + 2])
                i += 3
        widths[k] = (w, dw)
    runs = []
    ctm, stack = [1, 0, 0, 1, 0, 0], []
    tm = tlm = [1, 0, 0, 1, 0, 0]
    font, size = None, 0.0
    for ops, op in ContentStream(page.get_contents(), rd).operations:
        if op == b"q":
            stack.append(ctm)
        elif op == b"Q":
            ctm = stack.pop()
        elif op == b"cm":
            ctm = mul([float(v) for v in ops], ctm)
        elif op == b"BT":
            tm = tlm = [1, 0, 0, 1, 0, 0]
        elif op == b"Tf":
            font, size = ops[0], float(ops[1])
        elif op == b"Td":
            tlm = mul([1, 0, 0, 1, float(ops[0]), float(ops[1])], tlm); tm = tlm
        elif op == b"Tm":
            tlm = [float(v) for v in ops]; tm = tlm
        elif op in (b"Tj", b"TJ"):
            items = ops[0] if op == b"TJ" else [ops[0]]
            w, dw = widths[font]
            adv, text = 0.0, ""
            for it in items:
                if isinstance(it, (str, bytes)):
                    raw = it.get_original_bytes() if hasattr(it, "get_original_bytes") else bytes(it)
                    raw = raw[2:] if raw[:2] == b"\xfe\xff" else raw          # pypdf re-adds a UTF-16 BOM
                    cids = [int.from_bytes(raw[i:i + 2], "big") for i in range(0, len(raw) - 1, 2)]
                    adv += sum(w.get(c, dw) for c in cids) / 1000 * size
                    text += str(it)
                else:
                    adv -= float(it) / 1000 * size
            if not text.strip():
                continue
            m = mul(tm, ctm)
            eff = size * abs(m[0] * m[3] - m[1] * m[2]) ** 0.5
            corners = [(x, y) for x in (0.0, adv) for y in (-0.25 * size, 0.9 * size)]
            pts = [(m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]) for x, y in corners]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            runs.append((eff, (min(xs), min(ys), max(xs), max(ys)), text))
    return runs, page.mediabox, len(rd.pages)


def audit():
    """Width of every v4 PDF (<= 5.5 in), the smallest glyph actually set in it (>= 8 pt, scripts included), every text
    run inside the page box, and the layout check (no legend/text over data, no overlapping text)."""
    print("\nAUDIT: text width, smallest glyph, text inside page box, layout")
    ok_all = True
    for pdf in sorted(OUT.glob("v4_*.pdf")):
        runs, mb, npages = _pdf_text_runs(pdf)
        W, H = float(mb.width), float(mb.height)
        smallest = min(r[0] for r in runs)
        outside = [r[2] for r in runs if r[1][0] < -0.01 or r[1][1] < -0.01 or r[1][2] > W + 0.01 or r[1][3] > H + 0.01]
        lay = LAYOUT.get(pdf.stem)
        ok = npages == 1 and W / 72 <= TEXT_W + 0.005 and smallest >= SMALLEST_PT - 1e-6 and not outside and lay == []
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {pdf.name:24s} width {W / 72:.3f} in  height {H / 72:.2f} in  "
              f"smallest glyph {smallest:.2f} pt  ({len(runs)} runs; outside page box: {len(outside)}; "
              f"layout problems: {'not checked' if lay is None else len(lay)})")
        for s in outside:
            print(f"        outside page box: {s!r}")
        for s in lay or []:
            print(f"        layout: {s}")
    return ok_all


def provenance():
    """In the repository: every source artifact must be committed. Outside a git checkout (e.g. the supplementary
    archive, which ships files without history): every source artifact must be present and non-empty."""
    in_git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip() == "true"
    if in_git:
        tracked = set(subprocess.run(["git", "ls-files", "results"], capture_output=True, text=True,
                                     cwd=ROOT).stdout.split())
        missing = sorted(s for s in SOURCES if f"results/{s}" not in tracked)
        print(f"\nPROVENANCE: {len(SOURCES)} source artifacts; not committed: {missing or 'none'}")
    else:
        missing = sorted(s for s in SOURCES if not (RESULTS / s).is_file() or (RESULTS / s).stat().st_size == 0)
        print(f"\nPROVENANCE (not a git checkout): {len(SOURCES)} source artifacts; missing or empty: {missing or 'none'}")
    return not missing


def main():
    info = {"decomposition": fig_decomposition(), "thresholds": fig_thresholds(), "cond_candidates": fig_cond_candidates(),
            "fixed_scale": fig_fixed_scale(horizons=True), "mirror_branches": fig_mirror_branches(),
            "prospective": fig_prospective(), "prospective_own": fig_prospective_own()}
    for k, v in info.items():
        print(f"  {k}: {v}")
    ok = audit() & provenance()
    print("\nALL CHECKS PASS" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
