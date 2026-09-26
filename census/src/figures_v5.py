"""Main-text figures, redesigned to the author's rules (2026-09-24), and their appendix companions.

Rules: one message per figure; no caption text in the image (captions are LaTeX); no internal names (no Block numbers,
test IDs, "certified units", PASS/FAIL); direct labels where a curve can be labelled, otherwise at most three legend
entries; two or three colours from a colourblind-safe palette (Okabe-Ito), each with one meaning across figures, grey
for context only; only informative series (coinciding curves go to an appendix figure); no clipping; at most about
3 in tall at the 5.5 in text width, every glyph >= 8 pt, in the paper's serif font (Times, the ICLR template default).

Colour meanings (every figure):
  BLUE   #0072B2  conditional minimisation: its threshold R_glob and predictions built on it
  VERM   #D55E00  not placed: networks that start unplaced, placement failures
  GREEN  #009E73  placed: networks that stay placed, runs that solve
  GREY            context only

    python -m src.figures_v5
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import figures_v4 as V4                       # rcParams (Times, 8 pt floor), data readers, audits
from .figures_v4 import RESULTS, TEXT_W, clopper, read

OUT = RESULTS / "figures" / "v5"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "0.6"
MAX_H = 3.0
RG = r"$R_{\mathrm{glob}}$"
LAYOUT: dict[str, list[str]] = {}


def save(fig, name):
    LAYOUT[name] = V4.layout_check(fig)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  {name}" + (f"  LAYOUT: {LAYOUT[name]}" if LAYOUT[name] else ""))


def _curve(blk, variant="preserved"):
    cv = read(f"fixed_scale_block{blk}_curve.csv")
    g = cv[cv.variant == variant].sort_values("level")
    x50 = float(read(f"fixed_scale_block{blk}_tests.csv").set_index("variant").loc[variant, "x50"])
    return g, x50


def _errbar(ax, g, color, filled=True, marker="o", ls="-", dx=0.0, label=None, ms=3.8, lw=1.2):
    return ax.errorbar(g.level + dx, g.frac, yerr=[g.frac - g.ci95_lo, g.ci95_hi - g.frac], color=color, marker=marker,
                       ls=ls, lw=lw, ms=ms, mfc=color if filled else "white", mec=color, capsize=1.8, elinewidth=0.7,
                       label=label, zorder=3)


# ------------------------------------------------------------------------------------------ fixed scale (main)
def fig_fixed_scale():
    """One message: held at a fixed output scale, an unplaced network becomes placed, and a placed one stays placed,
    only above the threshold from conditional minimization; both transitions sit just above it."""
    g4, x4 = _curve(4)
    g5, x5 = _curve(5)
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.35))
    ax.axvline(1.0, color=BLUE, lw=1.1, zorder=1)
    ax.text(0.985, 0.97, "threshold from\nconditional minimization", color=BLUE, ha="right", va="top",
            transform=ax.get_xaxis_transform())
    _errbar(ax, g4, VERM, label="starting unplaced: becomes placed")
    _errbar(ax, g5, GREEN, marker="s", label="starting placed: stays placed")
    for x50, col, dy, ha, txt_dx in ((x4, VERM, 0.5, "left", 0.03), (x5, GREEN, 0.5, "left", 0.03)):
        ax.plot([x50], [0.5], marker="|", ms=9, mew=1.3, color=col, zorder=4)
    ax.annotate(f"midpoint {x4:.2f}", xy=(x4, 0.5), xytext=(1.30, 0.62), color=VERM, va="center",
                arrowprops=dict(arrowstyle="-", color=VERM, lw=0.6, shrinkA=1, shrinkB=2))
    ax.annotate(f"midpoint {x5:.2f}", xy=(x5, 0.5), xytext=(1.30, 0.40), color=GREEN, va="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6, shrinkA=1, shrinkB=2))
    ax.set_xlim(0.55, 2.05); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_xlabel(r"held output scale $R/R_{\mathrm{glob}}$")
    ax.set_ylabel("fraction placed")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, "fixed_scale")
    return {"x50_becomes": x4, "x50_stays": x5}


# ------------------------------------------------------------------------------------------ fixed scale (appendix)
def fig_fixed_scale_appendix():
    """The coinciding curves: optimiser moments reset, the longer horizons, and the earlier retention criterion."""
    import re
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(TEXT_W, 2.4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, blk, col, mk, lab in ((a1, 4, VERM, "o", "becomes placed"), (a2, 5, GREEN, "s", "stays placed")):
        gp, _ = _curve(blk, "preserved"); gr, _ = _curve(blk, "reset")
        ax.plot([1.0, 1.0], [-0.03, 1.06], color=BLUE, lw=0.9, zorder=1)          # stops below the legend band
        _errbar(ax, gp, col, marker=mk, dx=-0.008, ms=3.2, lw=1.0)
        _errbar(ax, gr, col, marker=mk, filled=False, ls="--", dx=0.008, ms=3.2, lw=1.0)
        ax.set_xlim(0.55, 2.05); ax.set_ylim(-0.03, 1.5)
        ax.set_xticks([0.6, 1.0, 1.5, 2.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(r"held $R/R_{\mathrm{glob}}$")
        ax.set_title(f"({'a' if blk == 4 else 'b'}) {lab}", loc="left", fontsize=V4.NOTE_PT)
    a1.set_ylabel("fraction")
    # (b): the earlier retention criterion (>= 0.9 kept at 1.15 x an earlier estimate of R_glob) and its outcome
    br = read("cond_certified_brackets.csv")
    r13 = br[(br.a.round(2) == 1.3) & (br.kind == "glob")].iloc[0]
    e2_x = 1.15 * V4._block_e_rglob() / (0.5 * (r13.R_lo + r13.R_hi))
    e2 = read("wi_e2_rescore.csv").set_index("arm").loc["hold_high"]
    kept, n_e = int(e2.kept), int(e2.intervened)
    lo, hi = clopper(kept, n_e)
    a2.plot([e2_x - 0.06, e2_x + 0.06], [0.9, 0.9], color="k", lw=1.4, zorder=4)
    a2.errorbar([e2_x], [kept / n_e], yerr=[[kept / n_e - lo], [hi - kept / n_e]], color="k", marker="*", ms=6,
                capsize=1.5, elinewidth=0.6, ls="none", zorder=5)
    a2.annotate(f"earlier run:\n{kept}/{n_e} stayed placed\n(criterion 0.9)", xy=(e2_x + 0.03, kept / n_e),
                xytext=(1.32, 0.45), va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    # (c) horizon extension, moments preserved
    hz = read("fixed_scale_horizons.csv", float_precision="round_trip")
    hz = hz[hz.variant == "preserved"]
    a3.plot([1.0, 1.0], [-0.03, 0.78], color=BLUE, lw=0.9, zorder=1)
    for H, mk, ls, dx in ((4000, "o", "-", -0.006), (16000, "D", "--", 0.0), (64000, "^", ":", 0.006)):
        g = hz.groupby("level")[f"placed_{H}"]
        k, n = g.sum().astype(int), g.size()
        fr = k / n
        ci = np.array([clopper(int(kk), int(nn)) for kk, nn in zip(k, n)])
        a3.errorbar(fr.index + dx, fr.values, yerr=[fr.values - ci[:, 0], ci[:, 1] - fr.values], color=VERM,
                    marker=mk, ls=ls, lw=0.9, ms=3.0, mfc="white" if H != 4000 else VERM, mec=VERM, capsize=1.5,
                    elinewidth=0.6, label=f"{H // 1000}k steps")
    a3.set_xlim(0.87, 1.13); a3.set_ylim(-0.03, 1.2); a3.set_xticks([0.9, 1.0, 1.1]); a3.set_yticks([0, 0.25, 0.5, 0.75])
    a3.set_xlabel(r"held $R/R_{\mathrm{glob}}$")
    a3.set_title("(c) longer horizons", loc="left", fontsize=V4.NOTE_PT)
    a3.legend(loc="upper left", handlelength=2.2, ncol=1, borderaxespad=0.1)
    hs = [Line2D([], [], color="0.3", marker="o", ls="-", ms=3.2), Line2D([], [], color="0.3", marker="o", ls="--",
                                                                           mfc="white", ms=3.2)]
    a1.legend(hs, ["moments kept", "moments reset"], loc="upper left", handlelength=2.2, borderaxespad=0.1)
    for ax in (a1, a2, a3):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "app_fixed_scale")


def _clean(*axes):
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)


# ------------------------------------------------------------------------------------------ outcome decomposition
def fig_decomposition(precision=None, name="decomposition"):
    """One message: training fails by not placing the hidden unit; below ε ≈ 0.25 every run fails this way, and bias
    failures appear only in a narrow band around ε ≈ 0.4.  precision="float32": the single-draw 2,400-run population
    (final round: the population the main text should use; WP-26)."""
    d = read("phase1_decomposition.csv")
    if precision is not None:
        d = d[d.precision == precision]
    g = d.groupby("a")
    eps = np.array(sorted(d.a.unique())) - 1
    ns = g.size().values
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.3))
    for cls, col, mk, ls, filled in (("placement", VERM, "o", "-", True), ("bias", GREEN, "^", "--", False),
                                      ("solved", GREEN, "s", "-", True)):
        k = g.failure.apply(lambda f: int((f == cls).sum())).values
        fr = k / ns
        ci = np.array([clopper(int(kk), int(nn)) for kk, nn in zip(k, ns)])
        ax.errorbar(eps, fr, yerr=[fr - ci[:, 0], ci[:, 1] - fr], color=col, marker=mk, ls=ls, lw=1.1, ms=3.6,
                    mfc=col if filled else "white", mec=col, capsize=1.6, elinewidth=0.6, zorder=3)
    ax.text(0.021, 0.90, "fails to place the hidden unit", color=VERM, va="top")
    ax.text(1.4, 0.80, "solves", color=GREEN, ha="center", va="bottom")
    ax.text(0.28, 0.30, "places it,\noutput bias wrong", color=GREEN, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$\varepsilon=a-1$ (log scale)")
    ax.set_ylabel("fraction of runs")
    ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    _clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, name)
    return {"n_per_a": int(ns.min())}


# ------------------------------------------------------------------------------------------ thresholds against crossings
def fig_thresholds():
    """One message: at every a, free training places the hidden unit just above the conditional-minimization threshold
    R_glob and far below R_solve; both thresholds have certified limits as ε -> 0."""
    br = read("cond_certified_brackets.csv")
    c1 = read("first_order_c1.csv").iloc[0]
    sl = read("mn2_solve_limit.csv").iloc[0]
    runs = read("wi_crossing_runs.csv")
    runs = runs[runs.budget == 32_000]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.45))
    eps = sorted(br.a.round(2).unique() - 1)
    for e in eps:
        g = runs[runs.a.round(2) == round(1 + e, 2)].R_cert.values
        vp = ax.violinplot([g], positions=[e], widths=0.036, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(VERM); b.set_edgecolor("none"); b.set_alpha(0.28)
        med, lo, hi = V4.boot_median(g)
        ax.errorbar([e], [med], yerr=[[med - lo], [hi - med]], color=VERM, marker="o", ms=3.2, capsize=1.5,
                    elinewidth=0.7, zorder=4)
    for kind, ls, mk in (("glob", "-", "o"), ("solve", "--", "s")):
        g = br[br.kind == kind].sort_values("a")
        mid = 0.5 * (g.R_lo + g.R_hi)
        ax.plot(g.a - 1, mid, color=BLUE, ls=ls, lw=1.1, marker=mk, ms=3.0, zorder=3)
    Rg0 = 0.5 * (c1.R_glob_inf_lo + c1.R_glob_inf_hi)
    Rs0 = 0.5 * (sl.R_solve_inf_lo + sl.R_solve_inf_hi)
    ax.plot([0], [Rg0], color=BLUE, marker="o", ms=4.2, ls="none", zorder=4)
    ax.plot([0], [Rs0], color=BLUE, marker="s", ms=4.2, ls="none", mfc="white", zorder=4)
    ax.text(0.02, Rg0, r"$\varepsilon\to0$ limit", color=BLUE, va="center")
    ax.text(0.02, Rs0, r"$\varepsilon\to0$ limit", color=BLUE, va="center")
    ax.text(0.62, 0.314, r"$R_{\mathrm{solve}}$ (conditional minimization)", color=BLUE, ha="right", va="bottom")
    ax.text(0.28, 0.2135, r"$R_{\mathrm{glob}}$ (conditional minimization)", color=BLUE, ha="right", va="center")
    ax.text(0.27, 0.262, "free-training\ncrossings", color=VERM, ha="right", va="center")
    ax.set_xlim(-0.03, 0.64); ax.set_ylim(0.185, 0.325)
    ax.set_xlabel(r"$\varepsilon=a-1$")
    ax.set_ylabel(r"$R=|w_2|\hat G/2$")
    _clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "thresholds")


def fig_thresholds_appendix():
    """Panel removed from the main figure: small ε, the certified first-order line and the four small-ε brackets."""
    kb = read("limit_K_base.csv").iloc[0]
    c1 = read("first_order_c1.csv").iloc[0]
    fo = read("first_order_finite.csv")
    fig, bx = plt.subplots(figsize=(TEXT_W * 0.6, 2.2))
    wb = 0.0028
    bx.add_patch(plt.Rectangle((-wb / 2, kb.R_glob_inf_lo), wb, kb.R_glob_inf_hi - kb.R_glob_inf_lo, color=GREY,
                               alpha=0.5, lw=0))
    bx.plot([-wb / 2 - 0.0008, wb / 2 + 0.0008], [0.5 * (c1.R_glob_inf_lo + c1.R_glob_inf_hi)] * 2, color=BLUE, lw=1.6)
    for r in fo.itertuples():
        bx.add_patch(plt.Rectangle((r.eps - wb / 2, r.R_lo), wb, r.R_hi - r.R_lo, color=BLUE, lw=0))
    ee = np.linspace(0, 0.05, 51)
    c1m = 0.5 * (c1.c1_lo + c1.c1_hi)
    bx.plot(ee, 0.5 * (c1.R_glob_inf_lo + c1.R_glob_inf_hi) * (1 + c1m * ee), color="k", lw=0.8, ls="--")
    bx.text(0.003, 0.2011, f"first order, $c_1={c1m:.4f}$", va="bottom")
    bx.set_xlim(-0.004, 0.048); bx.set_ylim(0.1970, 0.2016); bx.set_xticks([0, 0.02, 0.04])
    bx.set_xlabel(r"$\varepsilon=a-1$"); bx.set_ylabel(RG)
    _clean(bx)
    fig.tight_layout(pad=0.3)
    save(fig, "app_thresholds_small_eps")


# ------------------------------------------------------------------------------------------ the conditional minimiser's gap
def fig_minimiser_gap(a=1.30):
    """One message: along the conditional minimizer at a = 1.30, the class gap changes sign exactly once, at R_glob.
    x = R/R_glob = |w2| / (midpoint of the certified R_glob bracket in |w2|), as in the fixed-scale figure."""
    d = read("cond_audit_candidates.csv")
    g = d[d.a.round(2) == round(a, 2)]
    ret = g[g.status == "RETAINED"].drop_duplicates("s").sort_values("s")
    scan = read("cond_scan_certified_a130.csv")
    br = read("cond_certified_brackets.csv")
    br = br[br.a.round(2) == round(a, 2)].set_index("kind")
    w_glob = 0.5 * (br.loc["glob", "w2_lo"] + br.loc["glob", "w2_hi"])
    xs = 0.5 * (br.loc["solve", "w2_lo"] + br.loc["solve", "w2_hi"]) / w_glob
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.3))
    ax.axhline(0, color=GREY, lw=0.7, zorder=0)
    ax.plot([1.0, 1.0], [-0.5, 0.06], color=BLUE, lw=1.0, zorder=1)
    ax.plot([xs, xs], [-0.5, 0.06], color=BLUE, lw=1.0, ls="--", zorder=1)
    ax.text(1.0 + 0.025, -0.40, r"$R_{\mathrm{glob}}$", color=BLUE, va="center")
    ax.text(xs + 0.025, -0.20, r"$R_{\mathrm{solve}}$", color=BLUE, va="center")
    ax.plot(ret.s / w_glob, ret.gap, color=BLUE, lw=1.2, zorder=2)
    ax.plot(scan.s / w_glob, scan.argmin_G, color=BLUE, marker="s", ms=3.4, mfc="white", ls="none", zorder=3)
    ax.text(9.0 / w_glob, 0.10, "conditional minimizer", color=BLUE, ha="center", va="bottom")
    ax.set_xlabel(r"output scale $R/R_{\mathrm{glob}}$" + f" ($a={a:.2f}$)")
    ax.set_ylabel("class gap $G$ of the minimizer")
    ax.set_ylim(-0.5, 0.16)
    _clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "minimiser_gap")
    return {"w2_glob_mid": w_glob, "R_solve_over_R_glob": xs, "x_range": (float(ret.s.min() / w_glob), float(ret.s.max() / w_glob))}


def fig_candidates_appendix(a=1.30):
    """Removed from the main figure: every candidate of the search, the stricter optimisation."""
    d = read("cond_audit_candidates.csv")
    g = d[d.a.round(2) == round(a, 2)]
    ret = g[g.status == "RETAINED"].drop_duplicates("s").sort_values("s")
    st = read("cond_audit_strict.csv"); st = st[st.a.round(2) == round(a, 2)]
    scan = read("cond_scan_certified_a130.csv")
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.4))
    for sts, col, mk, ms, lab in ((("degenerate",), "0.75", "x", 2.6, "discarded as degenerate"),
                                  (("not carried (screen rank > 8)", "carried"), "0.45", ".", 2.4, "screening runs"),
                                  (("not lowest",), BLUE, "o", 2.6, "full runs, not the lowest")):
        h = g[g.status.isin(sts)]
        ax.plot(h.s, h.loss, color=col, marker=mk, ls="none", ms=ms, mfc="none" if mk == "o" else col, mew=0.5,
                zorder=1, label=lab)
    ax.plot(ret.s, ret.loss, color=BLUE, lw=1.1, zorder=3, label="conditional minimizer (retained)")
    ax.plot(scan.s, np.minimum(scan.m_minus_lo, scan.m_plus_lo), color="k", marker="s", ms=3.0, mfc="white", ls="none",
            zorder=4, label="certified global minimum")
    ax.plot(st.s, st.loss, color="k", marker="D", ms=3.0, mfc="white", ls="none", zorder=5,
            label="stricter optimisation")
    ax.axhline(np.log(2), color="0.3", lw=0.8, ls=":", zorder=2)
    ax.text(11.2, np.log(2) * 1.08, "constant predictor", ha="right", va="bottom", color="0.3")
    ax.set_yscale("log"); ax.set_ylim(0.17, 25)
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(f"output scale $|w_2|$ ($a={a:.2f}$)"); ax.set_ylabel("conditional loss")
    ax.legend(loc="upper left", markerscale=1.4, ncol=2, columnspacing=1.2)
    ax.set_ylim(0.17, 60)
    _clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "app_candidates")


# ------------------------------------------------------------------------------------------ mirror branches
def fig_mirror_branches():
    """One message: a run crosses just above the own threshold of the mirror branch it occupies at the crossing, not
    that of the branch its initialisation selects."""
    occ = read("mirror_occupancy.csv", float_precision="round_trip")
    T = read("mirror_branch_thresholds.csv", float_precision="round_trip")
    T = T[T.group == "cross"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.7))
    for ax, a in zip(axes, (1.3, 1.5)):
        G = V4._ghat(a)
        o = occ[occ.group == f"phase 2b crossing a={a:.2f}"].merge(T[T.a.round(2) == a][["seed", "T_plus", "T_minus"]],
                                                                      on="seed")
        o["R_occ"] = np.where(o.branch > 0, o.T_plus, o.T_minus) * G / 2
        o["R_init"] = np.where(o.init_branch > 0, o.T_plus, o.T_minus) * G / 2
        allv = np.r_[o.R_cross, o.R_occ, o.R_init]
        lim = [allv.min() - 0.006, allv.max() + 0.006]
        ax.plot(lim, lim, color=GREY, lw=0.8, ls="--", zorder=1)
        ax.plot(o.R_init, o.R_cross, color=GREY, marker="x", ms=3.4, mew=0.7, ls="none", zorder=2)
        ax.plot(o.R_occ, o.R_cross, color=VERM, marker="o", ms=3.4, ls="none", zorder=3)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_title(f"$a={a:.2f}$", loc="left", fontsize=V4.NOTE_PT)
        ax.set_xlabel("threshold of a mirror branch ($R$)")
        _clean(ax)
    axes[0].set_ylabel("crossing $R$ of the run")
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 0.9))
    fig.legend([Line2D([], [], color=VERM, marker="o", ls="none", ms=3.6),
                Line2D([], [], color=GREY, marker="x", ls="none", ms=3.6, mew=0.8),
                Line2D([], [], color=GREY, ls="--", lw=0.8)],
               ["branch occupied at the crossing", "branch selected at initialisation", "crossing = threshold"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.4)
    save(fig, "mirror_branches")


# ------------------------------------------------------------------------------------------ held-out settings
def fig_prospective():
    """One message: in eight held-out settings, the conditional threshold with one fitted lag factor predicts the median
    crossing better than either baseline."""
    from .prospective import _cert_ghat
    gh = _cert_ghat()
    runs = read("prospective_runs.csv")
    runs = runs[runs.cross_step.notna()].copy()
    runs["R"] = [w2 * gh[(w, round(a, 2))] / 2 for w, a, w2 in zip(runs.window, runs.a, runs.cross_w2)]
    sc = read("prospective_scores.csv")
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.7))
    for r in sc.itertuples():
        g = runs[(runs.window == r.window) & (runs.a.round(2) == round(r.a, 2))]
        med, lo, hi = V4.boot_median(g.R.values)
        ax.plot([med, med], [r.U, r.C], color=BLUE, lw=0.6, zorder=1)
        for yv, mk, col, filled, ms in ((r.B1, "^", GREY, True, 4.0), (r.B2, "D", GREY, True, 3.6),
                                        (r.U, "o", BLUE, False, 4.4), (r.C, "o", BLUE, True, 4.4)):
            ax.errorbar(med, yv, xerr=[[med - lo], [hi - med]], color=col, ecolor="0.75", elinewidth=0.6, capsize=1.4,
                        marker=mk, ms=ms, mfc=col if filled else "white", mec=col, ls="none", zorder=3 if col == BLUE else 2)
    lim = [0.13, 0.335]
    ax.plot(lim, lim, color=GREY, lw=0.8, ls="--", zorder=0)
    ax.text(0.2825, 0.2885, "prediction = observed", color="0.45", rotation=0, ha="right", va="bottom")
    ax.set_xlim(0.195, 0.285); ax.set_ylim(*lim)
    ax.set_xlabel(r"observed median crossing $R$")
    ax.set_ylabel(r"predicted crossing $R$")
    hs = [Line2D([], [], color=BLUE, marker="o", ls="none", ms=4.4),
          Line2D([], [], color=BLUE, marker="o", ls="none", ms=4.4, mfc="white"),
          (Line2D([], [], color=GREY, marker="^", ls="none", ms=4.0), Line2D([], [], color=GREY, marker="D", ls="none", ms=3.6))]
    from matplotlib.legend_handler import HandlerTuple
    ax.legend(hs, [r"$R_{\mathrm{glob}}$ $\times$ fitted lag factor", r"$R_{\mathrm{glob}}$ alone",
                   "baselines from earlier runs"], loc="lower right", handler_map={tuple: HandlerTuple(ndivide=None)})
    _clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "prospective")


# ------------------------------------------------------------------------------------------ own-seed test
def fig_prospective_own():
    """One message: a run's own threshold, fixed before training, predicts its crossing better than the population
    threshold, with or without a fitted factor."""
    pr = read("prospective_own_predictions.csv", float_precision="round_trip")
    rn = read("prospective_own_runs.csv", float_precision="round_trip")
    d = rn.merge(pr, on=["window", "a", "seed"])
    d["R_cross"] = d.cross_w2 * d.Ghat_lo / 2
    x = d[d.R_cross.notna()].copy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W - 0.1, 2.45), gridspec_kw={"width_ratios": [1.15, 1]})
    lim = [0.15, 0.36]
    a1.plot(lim, lim, color=GREY, lw=0.8, ls="--", zorder=1)
    for a, mk in ((1.3, "o"), (1.5, "s")):
        g = x[x.a.round(2) == a]
        a1.plot(g.U_own, g.R_cross, color=VERM, marker=mk, ms=2.3, mfc="none", mew=0.45, ls="none", alpha=0.85,
                zorder=2, label=f"$a={a:.2f}$")
    a1.text(0.357, 0.258, "crossing =\nown threshold", color="0.45", ha="right", va="top")
    a1.set_xlim(*lim); a1.set_ylim(*lim)
    a1.set_xlabel(r"own threshold of the run, fixed before training")
    a1.set_ylabel("crossing $R$ of the run")
    a1.legend(loc="lower right", markerscale=1.8)
    models = [("U", "o", False, "population"), ("C", "o", True, "population, fitted"),
              ("U_own", "s", False, "own"), ("C_own", "s", True, "own, fitted")]
    for i, a in enumerate((1.3, 1.5)):
        g = x[x.a.round(2) == a]
        for j, (m, mk, filled, lab) in enumerate(models):
            per_w = g.groupby("window").apply(lambda h: float(np.abs(np.log(h.R_cross / h[m])).mean()),
                                              include_groups=False)
            mean, lo, hi = V4.boot_mean(per_w.values)
            xp = i + (j - 1.5) * 0.2
            a2.errorbar([xp], [mean], yerr=[[mean - lo], [hi - mean]], color=BLUE, marker=mk, ms=4.6,
                        mfc=BLUE if filled else "white", mec=BLUE, capsize=1.5, elinewidth=0.7, ls="none",
                        label=lab if i == 0 else None)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["$a=1.30$", "$a=1.50$"])
    a2.set_xlim(-0.55, 1.55); a2.set_ylim(0, 0.30)
    a2.legend(loc="upper right", ncol=2, columnspacing=0.8, handletextpad=0.2, borderaxespad=0.1)
    a2.set_ylabel("mean per-run |log error|")
    _clean(a1, a2)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "prospective_own")


# ------------------------------------------------------------------------------------------ setting (redesign of fig1_setting)
SETTING_SOLVER = (0.9437292267391273, 2.3593122836987837, -4.400602112045125, 13.150372704129833)   # a = 1.5, seed 0
SETTING_MONO = (-0.8161124460820061, -2.3657288524708325, 2.6410341930922247, 7.321862160553577)    # a = 1.0, seed 38


def fig_setting():
    """One message: for a > 1 the activation folds, and only a folded unit can put both outer windows on the same side of
    the inner one.  (a) f_a for a = 0.9, 1.05, 2.0 with the fold depth D(2) (exact closed form); (b) the task windows and
    two trained networks (parameters as in paper/make_figures.fig1_setting, recovered from seeds; sign patterns asserted)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.3))
    t = np.linspace(-6.5, 6.5, 900)
    for a_, c, ls, lw in ((0.9, GREY, "-", 1.2), (1.05, "0.35", "--", 1.0), (2.0, "k", "-", 1.2)):
        ax1.plot(t, t + a_ * np.sin(t), color=c, ls=ls, lw=lw, zorder=2)
    ax1.text(-6.2, 5.0, "$a = 2$", color="k")
    ax1.text(1.2, -5.8, "$a = 0.9$, $1.05$", color="0.35")
    a = 2.0
    crit = np.arccos(1.0 / a)
    tmax, tmin = np.pi - crit, np.pi + crit
    fmax, fmin = tmax + a * np.sin(tmax), tmin + a * np.sin(tmin)
    assert abs((fmax - fmin) - 2.0 * (np.sqrt(a * a - 1.0) - np.arccos(1.0 / a))) < 1e-9
    ax1.plot([tmax, tmin], [fmax, fmin], "k.", ms=4, zorder=5)
    xb = tmin + 1.4
    ax1.annotate("", xy=(xb, fmin), xytext=(xb, fmax), arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
    ax1.plot([tmax, xb], [fmax, fmax], color="k", lw=0.5, ls=":")
    ax1.plot([tmin, xb], [fmin, fmin], color="k", lw=0.5, ls=":")
    ax1.text(xb + 0.2, (fmax + fmin) / 2, "fold", va="center")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(r"$f_a(t) = t + a\,\sin t$")
    x = np.linspace(-2.3, 2.3, 900)
    lo_y, hi_y = -1.45, 1.65
    band = (0.12, 0.80)                                            # shading kept off the label strips (layout rule)
    ax2.axvspan(-0.8, 0.8, ymin=band[0], ymax=band[1], color="0.88", lw=0, zorder=0)
    for lo_, hi_ in ((1.2, 2.0), (-2.0, -1.2)):
        ax2.axvspan(lo_, hi_, ymin=band[0], ymax=band[1], color="0.94", lw=0, zorder=0)
    y_top = lo_y + (band[1] + 0.07) * (hi_y - lo_y)
    ax2.text(0, y_top, "inner", ha="center", va="center")
    ax2.text(1.6, y_top, "outer", ha="center", va="center")
    ax2.text(-1.6, y_top, "outer", ha="center", va="center")
    net = lambda p, a_: p[2] * ((p[0] * x + p[1]) + a_ * np.sin(p[0] * x + p[1])) + p[3]
    sv, mv = net(SETTING_SOLVER, 1.5), net(SETTING_MONO, 1.0)

    def signs(v):
        w = [np.sign(v[(x >= -2.0) & (x <= -1.2)]), np.sign(v[np.abs(x) <= 0.8]), np.sign(v[(x >= 1.2) & (x <= 2.0)])]
        return tuple(float(u[0]) for u in w) if all((u == u[0]).all() for u in w) else None
    assert signs(sv) == (1.0, -1.0, 1.0)
    assert signs(mv) != (1.0, -1.0, 1.0)
    ax2.plot(x, np.tanh(sv), color=GREEN, lw=1.3, zorder=3)
    ax2.plot(x, np.tanh(mv), color=GREY, lw=1.3, ls="--", zorder=3)
    ax2.axhline(0, color="k", lw=0.6, zorder=1)
    y_bot = lo_y + 0.05 * (hi_y - lo_y)
    ax2.text(1.55, y_bot, "folded: solves", color=GREEN, ha="center", va="center")
    ax2.text(-1.2, y_bot, "monotone: cannot", color="0.4", ha="center", va="center")
    ax2.set_xlabel("input $x$")
    ax2.set_ylabel("network output (tanh)")
    ax2.set_ylim(lo_y, hi_y)
    ax2.set_xlim(-2.3, 2.3)
    ax2.set_yticks([-1, 0, 1])
    for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(0.01, 0.99, lab, transform=ax.transAxes, ha="left", va="top")
    _clean(ax1, ax2)
    fig.tight_layout(pad=0.3)
    save(fig, "setting")
    return {"fold_depth_a2": float(fmax - fmin)}


def fig_metric_check():
    """One message: the continuous test error falls over a wide range of R while the binary solve rate is still zero.
    Two stacked panels sharing R (no twin axis); the 10-90% transition intervals as bars; numbers in the caption."""
    from .metric_check import analyse, binned, runs
    s_ = runs("certified"); g = binned(s_); m = analyse("certified")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TEXT_W, 2.9), sharex=True)
    ax1.plot(g["mid"], g.rate, color="k", marker="o", ms=3.4, lw=1.2, zorder=3)
    ax1.plot([m["binary_10"], m["binary_90"]], [1.14, 1.14], color="k", lw=2.2, solid_capstyle="butt")
    ax1.set_ylim(-0.05, 1.26)
    ax1.set_ylabel("solve rate")
    ax1.text(m["binary_10"] - 0.008, 1.14, "rate rises", ha="right", va="center")
    ax2.fill_between(g["mid"], g.q25, g.q75, color=GREY, alpha=0.35, lw=0, zorder=1)
    ax2.plot(g["mid"], g.err, color="0.25", marker="s", ms=3.2, lw=1.2, zorder=3)
    top = float(g.q75.max()) * 1.12
    ax2.plot([m["continuous_10"], m["continuous_90"]], [top, top], color="0.25", lw=2.2, solid_capstyle="butt")
    ax2.text(m["continuous_90"] + 0.008, top, "error falls", ha="left", va="center", color="0.25")
    ax2.set_ylim(-5, top * 1.14)
    ax2.set_ylabel("test errors")
    ax2.set_xlabel(r"output scale $R = |w_2|\,\hat G(a)/2$")
    for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(0.005, 0.97, lab, transform=ax.transAxes, ha="left", va="top")
    _clean(ax1, ax2)
    fig.tight_layout(pad=0.3)
    save(fig, "metric_check")
    return m


# ------------------------------------------------------------------------------------------ captions (for the writer)# ------------------------------------------------------------------------------------------ lag law (2026-09-25, final round)
LAG_A = {1.30: ("o", -1), 1.45: ("s", 0), 1.50: ("^", 0), 1.60: ("D", 0)}


def lag_data():
    A = read("lag_law/compare_arms.csv"); A["a"] = A.a.round(2)
    kb = read("lag_law/kappa_by_winding.csv"); kb["a"] = kb.a.round(2)
    kap = {a: float(kb[(kb.a == a) & (kb.k == k)].kappa_adam.iloc[0]) for a, (_, k) in LAG_A.items()}
    lr = RESULTS / "linear_response" / "compare_arms.csv"
    if not lr.exists():
        return A, kap, None
    L = pd.read_csv(lr); L["a"] = L.a.round(2)
    norm = lambda v: str(float(v)) if str(v).replace(".", "", 1).isdigit() else str(v)
    L["arm"] = L.arm.map(norm); A["arm"] = A.arm.map(norm)
    M = A[["set", "a", "arm", "median_chi"]].merge(L[["set", "a", "arm", "median_r_obs", "full_7_median_pred"]],
                                                    on=["set", "a", "arm"], how="inner")
    assert len(M) == len(A) == 36
    M["obs"] = M.median_r_obs                     # lag from each run's tracked-branch switch (Track 1, final round)
    return M, kap, M


def fig_lag():
    """One message: the crossing lag grows linearly with the growth-to-relaxation ratio, with the slope κ(a) predicted
    with no fitted parameter.  Median lag against median χ for all 36 free-training arms and tests; κ(a)χ lines."""
    A, kap, lr = lag_data()
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.6))
    xmax = float(A.median_chi.max()) * 1.12
    for a, (m, _) in LAG_A.items():
        xs = np.array([0.0, xmax])
        ax.plot(xs, kap[a] * xs, color=BLUE, lw=1.0, zorder=1)
        g = A[A.a == a]
        sgd = g["set"] == "SGD"
        ax.plot(g[~sgd].median_chi, g[~sgd].obs, ls="none", marker=m, ms=4.2, mfc="k", mec="k", zorder=3)
        ax.plot(g[sgd].median_chi, g[sgd].obs, ls="none", marker=m, ms=4.8, mfc="white", mec="k", mew=0.9, zorder=3)
        xe = min(0.09 / kap[a], xmax)                       # where the line leaves the plot
        ax.text(xe, 0.0915 if xe < xmax else kap[a] * xmax, (f"$a = {a:.2f}$" if a == 1.30 else f"${a:.2f}$"), color=BLUE,
                ha="center" if xe < xmax else "left", va="bottom" if xe < xmax else "center", clip_on=False)
    if lr is not None:
        ax.plot(lr.median_chi, lr.full_7_median_pred, ls="none", marker="x", ms=4.6, color="0.45", mew=0.9, zorder=4)
    ax.set_xlim(0, xmax); ax.set_ylim(0, 0.09)
    ax.set_xlabel(r"growth-to-relaxation ratio $\chi$ (median per arm)")
    ax.set_ylabel("crossing lag $r$ (median)")
    _clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "lag")


def scoreboard_data():
    """Every registered prospective width-1 test with a point or range prediction: (panel, family, label, pred, lo, hi,
    obs, pass).  Range predictions are drawn at their centre with the range as the bar."""
    rows = []
    ts = read("ts_test/scores.csv")
    for r in ts.itertuples():
        rows.append(("lag", "Task B", f"a={r.a:.2f}", r.pred, r.pred - r.tol, r.pred + r.tol, r.obs, abs(r.obs - r.pred) <= r.tol))
    ex = read("sgd_own_extension_scores.csv")
    for r in ex.itertuples():
        rows.append(("lag", "SGD", f"a={r.a:.2f}", r.pred, r.pred - r.tol, r.pred + r.tol, r.obs, abs(r.obs - r.pred) <= r.tol))
    import json as _j
    V = _j.loads((RESULTS / "ramp" / "verdicts.json").read_text())
    for x in V["R4"]:
        for e in (0.005, 0.0025):
            p_, t_, o_ = x["median_0.01"], x["tol"], x[f"median_{e}"]
            rows.append(("lag", "learning rate", f"a={x['a']:.2f} eta={e}", p_, p_ - t_, p_ + t_, o_, abs(o_ - p_) <= t_))
    ta = read("act_general/training_scores.csv"); ta = ta[ta.arm == "secondary_pooled_200"]
    for r in ta.itertuples():
        rows.append(("lag", "other activations", r.act, r.pred, r.pred - r.tol, r.pred + r.tol, r.obs, abs(r.obs - r.pred) <= r.tol))
    bd = read("band_rd/verdicts.csv"); bd = bd[(bd.arm == "primary") & (bd.d == 2)]
    for r in bd.itertuples():
        rows.append(("lag", "R^2", f"a={r.a:.2f}", r.median_pred_r, r.median_pred_r - r.tol, r.median_pred_r + r.tol,
                     r.median_obs_r_own, abs(r.median_obs_r_own - r.median_pred_r) <= r.tol))
    po = read("prospective_own_scores.csv"); po = po[(po.analysis == "primary (crossers)") & (po.comparison == "P4")]
    for r in po.itertuples():
        rows.append(("err", "own-sample threshold", f"a={r.a:.2f}", 0.5 * (r.lo + r.hi), r.lo, r.hi, r.stat, r.lo <= r.stat <= r.hi))
    es = read("prospective_own_early_scores.csv")
    # registered ranges (results/prospective_own_prediction.md:288-290)
    RU = {1.3: (0.029, 0.053), 1.5: (0.065, 0.124)}; RF = {1.3: (0.004, 0.023), 1.5: (0.005, 0.063)}
    for r in es.itertuples():
        a = round(r.a, 2)
        for fam, R_, v in (("early branch", RU, r.err_unfitted_median), ("early branch, lag-corrected", RF, r.err_fitted_median)):
            lo, hi = R_[a]
            rows.append(("err", fam, f"a={a:.2f}", 0.5 * (lo + hi), lo, hi, v, lo <= v <= hi))
    b3 = read("prospective_scores.csv")                   # registered range [0.0801, 0.1824] (Block 3 registration)
    for r in b3.itertuples():
        rows.append(("err", "held-out windows", f"{r.window} a={r.a:.2f}", 0.5 * (0.0801 + 0.1824), 0.0801, 0.1824,
                     r.abslogerr_U, bool(r.U_in_expected_range)))
    return pd.DataFrame(rows, columns=["panel", "family", "label", "pred", "lo", "hi", "obs", "passed"])


SB_MARK = {"Task B": "o", "SGD": "s", "learning rate": "^", "other activations": "v", "R^2": "P",
           "own-sample threshold": "o", "early branch": "s", "early branch, lag-corrected": "^", "held-out windows": "D"}


def fig_scoreboard():
    """One message: every registered prospective width-1 prediction with a stated value or range, observed against
    predicted; passes filled, failures open."""
    d = scoreboard_data()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.7))
    for ax, (panel, lim, lab) in zip(axes, (("lag", (-0.06, 0.22), "lag"), ("err", (0.0, 0.2), "|log error|"))):
        g = d[d.panel == panel]
        ax.plot(lim, lim, color=GREY, lw=0.7, zorder=0)
        fams = list(dict.fromkeys(g.family))
        for i, fam in enumerate(fams):
            q = g[g.family == fam].reset_index(drop=True)
            dx = (np.arange(len(q)) - (len(q) - 1) / 2) * 0.0025 if fam == "held-out windows" else np.zeros(len(q))
            for j, r in q.iterrows():
                x = r.pred + dx[j]
                ax.plot([x, x], [r.lo, r.hi], color=GREY, lw=1.6, alpha=0.55, zorder=1, solid_capstyle="butt")
                ax.plot([x], [r.obs], ls="none", marker=SB_MARK[fam], ms=4.4, mec="k", mew=0.8,
                        mfc="k" if r.passed else "white", zorder=3)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel(f"predicted {lab}"); ax.set_ylabel(f"observed {lab}")
        _clean(ax)
    for ax, lab in zip(axes, ("(a)", "(b)")):
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout(pad=0.3)
    save(fig, "scoreboard")
    d.to_csv(OUT / "scoreboard_data.csv", index=False)



def captions():
    """results/figures/v5/captions.md: for each figure, its PDF path, one-sentence message, main text or appendix, the
    population and n, what the uncertainty shows, and every number that moved from the image into the caption.  Every
    number is computed here from the committed artifacts."""
    f3 = lambda v: f"{v:.3f}"
    E = []

    def entry(name, where, message, population, uncertainty, numbers, note=None):
        E.append(f"## {name}\n\n- **PDF**: `results/figures/v5/{name}.pdf` (PNG preview beside it)\n- **Placement**: {where}\n"
                 f"- **Message**: {message}\n- **Population and n**: {population}\n- **Uncertainty shown**: {uncertainty}\n"
                 "- **Numbers for the caption** (moved out of the image):\n" + "".join(f"  - {x}\n" for x in numbers)
                 + (f"- **Note**: {note}\n" if note else "") + "\n")
    # setting and metric check (redesigns of the earlier fig1_setting and fig4_metric_check), 2026-09-25
    import math as _m
    entry("setting", "main text",
          "For a > 1 the activation f_a folds, and only a folded unit can put both outer windows on the same side of the "
          "inner window: a trained non-monotone network solves the task and the best monotone one cannot.",
          "(a) f_a(t) = t + a sin t for a = 0.9, 1.05 and 2.0 (closed form). (b) Inner window [−0.8, 0.8] (class 0, "
          "negative) and outer windows ±[1.2, 2.0] (class 1, positive); two trained width-1 networks recovered "
          "deterministically from seeds (float64, Adam lr 0.01, 2,000 steps): a = 1.5, seed 0 (solves; dense 4,001-point "
          "check) and a = 1.0, seed 38 (the best monotone run, 91/200 sample errors). Outputs shown through tanh.",
          "None (a closed form and two individual runs).",
          [f"fold depth at a = 2: D(2) = 2(√(a²−1) − arccos(1/a)) = {2 * (_m.sqrt(3) - _m.acos(0.5)):.4f} (the double arrow)",
           "the folded unit's sign pattern across the windows is (+, −, +); the monotone unit changes sign only once across them"],
          note="Replaces paper/figures/fig1_setting.pdf (v4 styling).")
    from .metric_check import analyse as _an
    _mm = _an("certified")
    entry("metric_check", "main text (or appendix)",
          "The continuous test error falls over a wide range of output scale while the binary solve rate is still zero, "
          "so the sharp threshold is not an artifact of a binary metric.",
          f"Width 1, {_mm['runs']:,} training runs (fold1d sweep and refinement; certified Ĝ), binned by R in 0.025 bins. "
          "(a) Fraction of runs that solve. (b) Test errors per run: mean (squares) and interquartile band.",
          "Interquartile band of the per-run test errors in (b); none in (a).",
          [f"binary solve rate: 10–90% transition over R ∈ [{_mm['binary_10']:.3f}, {_mm['binary_90']:.3f}] (bar in (a))",
           f"continuous error: 10–90% of its fall over R ∈ [{_mm['continuous_10']:.3f}, {_mm['continuous_90']:.3f}] (bar in (b)); "
           f"width ratio {_mm['ratio']:.1f}",
           f"{_mm['improvement_at_zero_rate_pct']:.0f}% of the error improvement occurs in bins where no run solves "
           f"({_mm['runs_in_zero_rate_bins']:,} runs, R < {_mm['last_zero_rate_bin_right']:.2f})"],
          note="Replaces paper/figures/fig4_metric_check.pdf (v4 styling, twin axis).")
    _p32 = read("track4_float32_decomposition.csv")
    entry("decomposition_float32", "main text (replaces decomposition if the main text uses the float32 population; WP-26)",
          "Training fails by not placing the hidden unit; below ε ≈ 0.25 every run fails this way, and bias failures appear "
          "only in a narrow band around ε ≈ 0.4.",
          f"Width 1, the single-draw float32 population: {int(_p32.runs.sum()):,} runs, {int(_p32.runs.min())} per a at "
          f"{len(_p32)} values of a (Adam lr 0.01, 2,000 steps).",
          "Clopper–Pearson 95% intervals per a.",
          [f"totals: {int(_p32.solved.sum())} solved, {int(_p32.placement.sum()):,} placement failures, {int(_p32.bias.sum())} bias failures",
           "the same runs as the width-1 sweep and refinement (and the metric check)"],
          note="Final round: same design as 'decomposition', restricted to the float32 draw.")
    # lag law and scoreboard (final round, 2026-09-25)
    _A, _kap, _lr = lag_data()
    _L = read("linear_response/compare_arms.csv")
    entry("lag", "main text",
          "The crossing lag grows linearly with the ratio of output growth to branch relaxation, with a slope κ(a) "
          "computed from the landscape with no fitted parameter (derived after a fitted relationship was known).",
          f"Width 1, all {len(_A)} free-training arms and tests (the four intervention experiments, SGD, and the fresh-sample "
          "test at a = 1.45 and 1.60); each point is an arm's median lag against its median ratio. The lag is measured from "
          "the switch of the branch each run tracks, on its own training sample (math note §13; WP-31). Filled markers: Adam; "
          "open markers: SGD. Marker shape: a = 1.30 circles, 1.45 squares, 1.50 triangles, 1.60 diamond. Grey crosses: the "
          "exact linear response along each run's own trajectory, started at 0.7 s* (arm medians; post hoc, predictions "
          "committed before the comparison).",
          "None drawn (arm medians); lines and crosses are predictions, not fits.",
          [", ".join(f"κ({a:.2f}) = {_kap[a]:.2f}" for a in _kap) + " (lines r = κ(a)χ)",
           "measured from each sample's global own threshold instead (Track 1A, registered tolerance: 36 of 36 arms within), "
           "the observed/predicted through-origin slope per a is "
           + ", ".join(f"{r.ratio_obs_to_pred:.2f} ({r.a:.2f})" for r in read("lag_law/compare_per_a.csv").itertuples()),
           "exact linear response, observed/predicted lag per arm (ratio of arm medians over the runs with a prediction, "
           f"{int(_L.full_7_n.sum()):,} of {int(_L.n_runs.sum()):,} runs; tracked-branch reference): "
           f"{_L.full_7_ratio_of_medians.min():.2f}–{_L.full_7_ratio_of_medians.max():.2f}"],
          note="New figure (final round).")
    _sb = scoreboard_data()
    _fam = _sb.groupby(["panel", "family"], sort=False).passed.agg(["sum", "size"]).reset_index()
    entry("scoreboard", "main text or appendix",
          "Every registered prospective width-1 prediction that states a value or a range, observed against predicted: "
          "the timescale and threshold predictions hold, and the tests outside the sine family and in R² fail.",
          "(a) Lag predictions: fresh-sample timescale test (circles, a = 1.45, 1.60), SGD extension (squares), learning-rate "
          "invariance (upward triangles; η = 0.005 and 0.0025 against η = 0.01), GELU/SiLU/Mish (downward triangles, 200 "
          "seeds), single unit in R² (plus signs). (b) Threshold-error predictions: own-sample threshold (circles), early "
          "branch (squares) and lag-corrected early branch (triangles), held-out windows (diamonds, 8 settings, spread "
          "horizontally for visibility). Filled: within the registered tolerance or range; open: outside it.",
          "Grey bars: the registered tolerance (prediction ± tolerance) or registered range.",
          [f"{r['family']}: {int(r['sum'])} of {int(r['size'])} within" for _, r in _fam.iterrows()]
          + ["Not plotted (no stated predicted value; registered as paired comparisons, all PASS): own-threshold vs "
             "population (Task B, SGD, own-seed P1), own-seed P2a/P2b/P3, held-out windows C vs B1/B2/B3",
           "Not plotted (not a point prediction): the forced ramp (slopes and rank correlations; WP-24)"],
          note="New figure (final round).")
    # mechanism (width 1): |w1| of the conditional minimiser against R/R_glob (Task C, 2026-09-25)
    if (RESULTS / "mechanism_w1_path.csv").exists():
        mp_ = read("mechanism_w1_path.csv"); ms_ = read("mechanism_w1_stats.csv").set_index("Unnamed: 0")
        def _cross_bound(a):
            g = mp_[mp_.a.round(2) == a].sort_values("x")
            i = int(np.argmax(g.w1.values < a / 1.4))
            x0, x1, y0, y1 = g.x.values[i - 1], g.x.values[i], g.w1.values[i - 1], g.w1.values[i]
            return float(np.exp(np.log(x0) + (a / 1.4 - y0) / (y1 - y0) * (np.log(x1) - np.log(x0))))
        entry("mechanism_w1", "main text",
              "As output scale grows, the conditional minimizer's first-layer weight moves from the class-mean optimum "
              "α* toward the worst-case optimum, dropping below the placement bound a/1.4 before placement switches on at "
              "R_glob, where free training crosses.",
              "Width 1, a = 1.30 and 1.50, 800-point population. Minimizer: the conditional audit's retained minimizer "
              "(s ≥ 1.5) and the same frozen search at s = 0.05–1.25 (a validated search, not a certificate). R_glob: the "
              "certified bracket midpoint. Crossings: phase 2b, budget 32,000, "
              f"{int(ms_.loc[1.3, 'n_cross'])} and {int(ms_.loc[1.5, 'n_cross'])} crossing runs.",
              "None (point values; the bracket for R_glob is narrower than the line).",
              [f"α* = 1.7913 (the class-mean optimum, independent of a); a/1.4 = {1.3 / 1.4:.3f} (a = 1.30), {1.5 / 1.4:.3f} (a = 1.50)",
               f"|w1| at the smallest scale shown: {ms_.loc[1.3, 'w1_at_smallest']:.3f} (a = 1.30), {ms_.loc[1.5, 'w1_at_smallest']:.3f} (a = 1.50)",
               f"the minimizer drops below a/1.4 at R/R_glob ≈ {_cross_bound(1.3):.2f} (a = 1.30) and {_cross_bound(1.5):.2f} (a = 1.50): "
               "the bound is necessary for placement, not sufficient",
               f"crossings with |w1| < a/1.4: {100 * ms_.loc[1.3, 'cross_w1_below_bound']:.0f}% and {100 * ms_.loc[1.5, 'cross_w1_below_bound']:.0f}%",
               f"w2_glob = {ms_.loc[1.3, 'w2_glob']:.4f} (a = 1.30), {ms_.loc[1.5, 'w2_glob']:.4f} (a = 1.50)"],
              note="Do not describe the minimizer path as certified; only R_glob is.")
    # fixed scale
    t4 = read("fixed_scale_block4_tests.csv").set_index("variant"); t5 = read("fixed_scale_block5_tests.csv").set_index("variant")
    n4 = int(read("fixed_scale_block4_curve.csv").n.max()); n5 = int(read("fixed_scale_block5_curve.csv").n.max())
    lv = sorted(read("fixed_scale_block4_curve.csv").level.unique())
    entry("fixed_scale", "main text",
          "Held at a fixed output scale, an unplaced network becomes placed, and a placed one stays placed, only above the "
          "threshold from conditional minimization; both transitions sit just above it.",
          f"a = 1.30. Becomes placed: {n4} pre-placement checkpoints per held level, read after 4,000 steps. Stays placed: "
          f"{n5} runs per level, read after 12,000 steps. Held levels R/R_glob = {', '.join(f'{v:g}' for v in lv)}; "
          "R_glob is the certified threshold at a = 1.30 (midpoint of its bracket).",
          "Clopper–Pearson 95% intervals on each fraction.",
          [f"n = {n4} checkpoints per level (becomes placed); n = {n5} runs per level (stays placed)",
           f"midpoints (50% crossing of the fraction): {t4.loc['preserved', 'x50']:.3f} (becomes placed), "
           f"{t5.loc['preserved', 'x50']:.3f} (stays placed); shown rounded to 1.05 and 1.07 on the plot",
           "the two curves are read at different horizons: 4,000 steps (becomes placed) and 12,000 steps (stays placed)",
           "Adam's moments are preserved from the checkpoint in both curves (the reset variant is in the appendix figure)"])
    # fixed scale appendix
    hc = read("fixed_scale_horizons_curve.csv").set_index(["variant", "horizon"])
    hz = read("fixed_scale_horizons.csv"); nh = int(hz[hz.variant == "preserved"].groupby("level").size().min())
    br = read("cond_certified_brackets.csv"); r13 = br[(br.a.round(2) == 1.3) & (br.kind == "glob")].iloc[0]
    e2_x = 1.15 * V4._block_e_rglob() / (0.5 * (r13.R_lo + r13.R_hi))
    e2 = read("wi_e2_rescore.csv").set_index("arm").loc["hold_high"]
    entry("app_fixed_scale", "appendix (companion to fixed_scale)",
          "Resetting the optimiser's moments and running 4× or 16× longer leave both transitions where they are.",
          f"As fixed_scale; (c): {nh} replays per level at held R/R_glob = 0.90–1.10, moments preserved.",
          "Clopper–Pearson 95% intervals.",
          [f"(a) midpoints {t4.loc['preserved', 'x50']:.3f} (moments kept) and {t4.loc['reset', 'x50']:.3f} (reset)",
           f"(b) midpoints {t5.loc['preserved', 'x50']:.3f} (kept) and {t5.loc['reset', 'x50']:.3f} (reset)",
           f"(b) earlier run: held at 1.15× an earlier estimate of R_glob = {e2_x:.3f} R_glob; {int(e2.kept)}/{int(e2.intervened)} "
           "stayed placed against a criterion of at least 0.9",
           "(c) midpoints " + ", ".join(f"{hc.loc[('preserved', H), 'x50']:.4f} ({H // 1000}k steps)" for H in (4000, 16000, 64000)),
           f"(c) n = {nh} replays per level"])
    # decomposition
    d = read("phase1_decomposition.csv"); tot = d.failure.value_counts(); ns = d.groupby("a").size()
    entry("decomposition", "main text",
          "Training fails by not placing the hidden unit: below ε ≈ 0.25 every run fails this way, and bias failures "
          "appear only in a narrow band around ε ≈ 0.4.",
          f"{d.a.nunique()} values of a (ε = {min(d.a) - 1:.2f}–{max(d.a) - 1:.0f}); n = {int(ns.min())} runs per a (200 seeds × "
          f"2 precisions); {len(d):,} runs in total.",
          "Clopper–Pearson 95% intervals on each fraction.",
          [f"n = {int(ns.min())} runs per a; {len(d):,} runs in total",
           f"outcome counts: failed to place (G ≤ 0) {int(tot['placement']):,}; placed but output bias outside its interval "
           f"{int(tot['bias']):,}; solved {int(tot['solved']):,}",
           "0 disagreements between this classification and the independent solve check"])
    # thresholds
    runs = read("wi_crossing_runs.csv"); runs = runs[runs.budget == 32_000]; nr = runs.groupby(runs.a.round(2)).size()
    c1 = read("first_order_c1.csv").iloc[0]; sl = read("mn2_solve_limit.csv").iloc[0]; kb = read("limit_K_base.csv").iloc[0]
    bg = br[br.kind == "glob"].sort_values("a"); bs = br[br.kind == "solve"].sort_values("a")
    entry("thresholds", "main text",
          "At every a, free training places the hidden unit just above the conditional-minimization threshold R_glob and "
          "far below R_solve; both thresholds have certified limits as ε → 0.",
          f"a = 1.30–1.60 (six values); free-training crossings at budget 32,000: n = {int(nr.min())} runs per a.",
          "Violins: the distribution of crossing R; points: medians with bootstrap 95% intervals. The certified threshold "
          "brackets are narrower than the markers (plotted at their midpoints).",
          [f"n = {int(nr.min())} crossing runs per a (budget 32,000)",
           f"certified R_glob brackets: widths ≤ {float((bg.R_hi - bg.R_lo).max()):.1e}; R_solve brackets: widths ≤ "
           f"{float((bs.R_hi - bs.R_lo).max()):.1e}",
           f"ε → 0 limits: R_glob^∞ ∈ [{c1.R_glob_inf_lo:.7f}, {c1.R_glob_inf_hi:.7f}] (sharp value); "
           f"R_solve^∞ ∈ [{sl.R_solve_inf_lo:.4f}, {sl.R_solve_inf_hi:.4f}] (global)",
           "R_glob^∞ rests on K = sup G₀, certified by branch and bound over a box with a domain lemma (math note §8); its "
           "independent Arb check (verify_certificates.py, python-flint/Arb) verifies it (WP-17)"])
    fs = read("first_order_scores.csv").set_index("quantity").loc["c1 (primary): R_glob"]
    fo = read("first_order_finite.csv")
    entry("app_thresholds_small_eps", "appendix (companion to thresholds)",
          "At small ε the certified thresholds follow the certified first-order line R_glob^∞(1 + c₁ε).",
          f"{len(fo)} certified brackets at a = {fo.eps.min() + 1:.2f}–{fo.eps.max() + 1:.2f} and the limit; no training runs.",
          "Boxes are the certified brackets, to scale.",
          [f"c₁ ∈ [{c1.c1_lo:.7f}, {c1.c1_hi:.7f}]; the first-order range is certified for |ε| ≤ 0.05",
           f"R_glob^∞: sharp value [{c1.R_glob_inf_lo:.7f}, {c1.R_glob_inf_hi:.7f}] (line) inside the unconditional "
           f"bracket [{kb.R_glob_inf_lo:.5f}, {kb.R_glob_inf_hi:.5f}] (grey box)",
           f"the registered c₁ test was INCONCLUSIVE: feasible c₁ ∈ [{fs.feasible_lo:.3f}, {fs.feasible_hi:.3f}], width "
           f"{fs.feasible_width:.3f} > 0.1"])
    # minimiser gap
    b = br[br.a.round(2) == 1.3].set_index("kind")
    scan = read("cond_scan_certified_a130.csv")
    cand = read("cond_audit_candidates.csv"); cg = cand[cand.a.round(2) == 1.3]
    ret = cg[cg.status == "RETAINED"].drop_duplicates("s")
    sep = float(max(b.loc["glob", "argmin_separation_lo"], b.loc["glob", "argmin_separation_hi"]))
    wg = 0.5 * (b.loc["glob", "w2_lo"] + b.loc["glob", "w2_hi"])
    entry("minimiser_gap", "main text",
          "Along the conditional minimizer at a = 1.30, the class gap changes sign exactly once, at R_glob.",
          f"a = 1.30, population objective (800 points); the conditional minimizer at {len(ret)} output scales (line) and "
          f"the certified global minimizer at {len(scan)} scales (squares).",
          f"None drawn: the certified values are exact to ≤ {float(np.maximum(scan.m_minus_hi - scan.m_minus_lo, scan.m_plus_hi - scan.m_plus_lo).max()):.0e} in loss.",
          [f"x = R/R_glob = |w₂| / {wg:.5f} (midpoint of the certified R_glob bracket in |w₂|)",
           f"certified brackets: R_glob at |w₂| ∈ ({b.loc['glob', 'w2_lo']:.4f}, {b.loc['glob', 'w2_hi']:.4f}]; R_solve at "
           f"|w₂| ∈ ({b.loc['solve', 'w2_lo']:.4f}, {b.loc['solve', 'w2_hi']:.4f}] (R_solve/R_glob = "
           f"{0.5 * (b.loc['solve', 'w2_lo'] + b.loc['solve', 'w2_hi']) / wg:.3f})",
           f"the minimizers either side of the switch are at most {sep:.1e} apart (certified); other basins are at least "
           f"+{float(scan.competitor_margin.min()):.4f} above (certified, radius 0.1)"])
    st = read("cond_audit_strict.csv"); st = st[st.a.round(2) == 1.3]
    nk = cg.groupby(["kind", "s"]).ngroups
    counts = {k: int(cg.status.isin(v).sum()) for k, v in (("degenerate", ("degenerate",)),
              ("screen", ("not carried (screen rank > 8)", "carried")), ("full", ("not lowest",)))}
    entry("app_candidates", "appendix (companion to minimiser_gap)",
          "No candidate of the conditional search lies below the retained minimizer at any scale.",
          f"a = 1.30: {len(cg):,} candidates at {nk} evaluations.", "None.",
          [f"{counts['degenerate']:,} discarded as degenerate; {counts['screen']:,} screening runs; {counts['full']:,} full runs "
           f"that were not the lowest (they lie on the retained minimizer)",
           f"frozen search result reproduced at {nk}/{nk} evaluations; {len(scan)} certified global minima; {len(st)} stricter-"
           "optimisation points"])
    # mirror branches
    s3 = read("mirror_s3.csv").set_index(["a", "threshold"])
    occ = read("mirror_occupancy.csv", float_precision="round_trip")
    T = read("mirror_branch_thresholds.csv", float_precision="round_trip"); T = T[T.group == "cross"]
    rhos, bcount = [], []
    for a in (1.3, 1.5):
        o = occ[occ.group == f"phase 2b crossing a={a:.2f}"].merge(T[T.a.round(2) == a][["seed", "T_plus", "T_minus"]], on="seed")
        o["R_occ"] = np.where(o.branch > 0, o.T_plus, o.T_minus) * V4._ghat(a) / 2
        lo, hi = V4.boot_spearman(o.R_occ, o.R_cross)
        r = s3.loc[(a, "branch occupied at the crossing")]
        rhos.append(f"a = {a:.2f}: ρ = {r.spearman_rho:.3f} [{lo:.4f}, {hi:.4f}] (n = {int(r.n)}); initialisation-selected branch "
                    f"ρ = {s3.loc[(a, 'branch selected by initialisation')].spearman_rho:.2f}; own global threshold ρ = "
                    f"{s3.loc[(a, 'registered: own global threshold')].spearman_rho:.2f}; median residual log(R/T) = "
                    f"{r.residual_median_log:.3f}")
        bcount.append(f"a = {a:.2f}: {int((o.branch > 0).sum())} runs on branch +, {int((o.branch < 0).sum())} on branch −")
    entry("mirror_branches", "main text",
          "A run crosses just above the own threshold of the mirror branch it occupies at the crossing, not that of the "
          "branch its initialisation selects.",
          "Free-training crossings at budget 32,000, a = 1.30 and 1.50, n = 38 runs each; thresholds in R units "
          "(R = |w₂|Ĝ/2).",
          "None drawn; the caption gives Spearman ρ with bootstrap 95% intervals.",
          rhos + bcount + ["the caption must say this analysis is POST HOC (exploratory); the label is no longer in the image"])
    # held-out settings
    from .prospective import _cert_ghat
    pr_ = read("prospective_runs.csv"); pr_ = pr_[pr_.cross_step.notna()]
    nset = pr_.groupby(["window", pr_.a.round(2)]).size()
    cal = read("prospective_calibration.csv").set_index("a")
    entry("prospective", "main text",
          "In eight held-out settings, the conditional threshold with one fitted lag factor predicts the median crossing "
          "better than either baseline.",
          f"8 held-out settings (4 window geometries × 2 a), 90 runs each; n = {int(nset.min())}–{int(nset.max())} crossings per setting.",
          "Horizontal bars: bootstrap 95% intervals of the observed median crossing R.",
          [f"n = {int(nset.min())}–{int(nset.max())} crossings of 90 per setting",
           f"lag factor fitted on the base window: λ(1.30) = {cal.loc[1.3, 'lambda_fitted']:.3f}, λ(1.50) = {cal.loc[1.5, 'lambda_fitted']:.3f}",
           f"baselines: the base window's median crossing |w₂| (triangles); the pooled median crossing R of earlier windows, "
           f"{cal.loc[1.3, 'B2_pooled_median_cross_R']:.4f} (diamonds)",
           "placement was checked every 50 steps; the detection-sensitivity analysis (WP-6: crossings interpolated between "
           "checks, calibration and observations both) keeps every registered comparison's sign, each excluding 0"])
    # own-seed test
    prp = read("prospective_own_predictions.csv", float_precision="round_trip")
    rn = read("prospective_own_runs.csv", float_precision="round_trip")
    dd = rn.merge(prp, on=["window", "a", "seed"]); dd["R_cross"] = dd.cross_w2 * dd.Ghat_lo / 2
    x = dd[dd.R_cross.notna()]
    ss = read("prospective_own_settings_scored.csv")
    rho_res = {a: float(np.median(x[x.a.round(2) == a].C_own / x[x.a.round(2) == a].U_own)) for a in (1.3, 1.5)}
    wmax = float(((x.w2_own_hi - x.w2_own_lo) * x.Ghat_lo / 2).max())
    ncr = {a: (int((x.a.round(2) == a).sum()), int((dd.a.round(2) == a).sum())) for a in (1.3, 1.5)}
    s13, s15 = ss[ss.a.round(2) == 1.3].spearman_R_cross_vs_R_own, ss[ss.a.round(2) == 1.5].spearman_R_cross_vs_R_own
    entry("prospective_own", "main text",
          "A run's own threshold, fixed before training, predicts its crossing better than the population threshold, with "
          "or without a fitted factor.",
          f"16 never-trained settings (8 windows × 2 a), 60 runs each; n = {ncr[1.3][0]} of {ncr[1.3][1]} crossed at a = 1.30 "
          f"and {ncr[1.5][0]} of {ncr[1.5][1]} at a = 1.50.",
          "(b): window-level bootstrap 95% intervals (10,000 resamples, seed 0) of the mean per-run |log error|.",
          [f"(a) the own thresholds' frozen brackets are at most {wmax:.1e} wide (no longer drawn)",
           f"(a) per-setting Spearman ρ between crossing and own threshold: {s13.min():.2f}–{s13.max():.2f} (a = 1.30), "
           f"{s15.min():.2f}–{s15.max():.2f} (a = 1.50)",
           "(b) the four predictors: population = R_glob; population, fitted = λ(a)·R_glob (λ fitted on the base window); "
           "own = R_own (the run's own threshold); own, fitted = ρ_res(a)·R_own",
           f"(b) fitted ρ_res (C_own): {rho_res[1.3]:.4f} (a = 1.30), {rho_res[1.5]:.4f} (a = 1.50)",
           "(b) the registered criteria (P1, P2a, P3, P4 at a = 1.30; P1, P2b, P3, P4 at a = 1.50) all passed",
           "placement was checked every 50 steps; the detection-sensitivity analysis (WP-6) finds every registered criterion "
           "passes with interpolated crossings too, but at a = 1.50 the reading 'C beats U_own' does not survive "
           "(−0.003 [−0.013, 0.008])"])
    head = ("# Figure captions (v5): inputs for the LaTeX captions\n\nGenerated by `python -m src.figures_v5` from the "
            "committed artifacts. Figures are 5.5 in wide, at most 3 in tall, every glyph at least 8 pt (Times). Colours: "
            "blue = conditional minimization (thresholds and predictions built on it); vermillion = free training's placement "
            "from an unplaced start; green = stays placed / solves; grey = context.\n\n")
    (OUT / "captions.md").write_text(head + "".join(E))
    return len(E)


# ------------------------------------------------------------------------------------------ audit
def audit():
    print("\nAUDIT (v5): width <= 5.5 in, height <= 3 in, smallest glyph >= 8 pt, text inside the page, layout")
    ok_all = True
    for pdf in sorted(OUT.glob("*.pdf")):
        runs, mb, npages = V4._pdf_text_runs(pdf)
        W, H = float(mb.width), float(mb.height)
        smallest = min(r[0] for r in runs)
        outside = [r[2] for r in runs if r[1][0] < -0.01 or r[1][1] < -0.01 or r[1][2] > W + 0.01 or r[1][3] > H + 0.01]
        lay = LAYOUT.get(pdf.stem)
        ok = (npages == 1 and W / 72 <= TEXT_W + 0.005 and H / 72 <= MAX_H + 0.005 and smallest >= V4.SMALLEST_PT - 1e-6
              and not outside and lay == [])
        ok_all &= ok
        print(f"  {'ok  ' if ok else 'FAIL'} {pdf.name:28s} {W / 72:.2f} x {H / 72:.2f} in  smallest {smallest:.2f} pt  "
              f"outside {len(outside)}  layout {'?' if lay is None else len(lay)}")
        for s in outside:
            print("      outside:", repr(s))
        for s in lay or []:
            print("      layout:", s)
    return ok_all


def main():
    out = {}
    out.update(fig_fixed_scale())
    fig_fixed_scale_appendix()
    fig_decomposition()
    fig_thresholds(); fig_thresholds_appendix()
    out.update(fig_minimiser_gap()); fig_candidates_appendix()
    fig_mirror_branches()
    fig_prospective()
    fig_prospective_own()
    fig_setting()
    fig_metric_check()
    fig_lag()
    fig_scoreboard()
    fig_decomposition("float32", "decomposition_float32")
    if (RESULTS / "mechanism_w1_path.csv").exists():
        from . import figures_v5 as _F
        from .mechanism_w1_figure import figure as fig_mechanism_w1
        fig_mechanism_w1()
        LAYOUT.update(_F.LAYOUT)                      # the module copy the figure's save() recorded into
    print(out)
    print("captions:", captions())
    audit()


if __name__ == "__main__":
    main()
