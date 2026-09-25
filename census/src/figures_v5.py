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
    only above the threshold from conditional minimisation; both transitions sit just above it."""
    g4, x4 = _curve(4)
    g5, x5 = _curve(5)
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.35))
    ax.axvline(1.0, color=BLUE, lw=1.1, zorder=1)
    ax.text(0.985, 0.97, "threshold from\nconditional minimisation", color=BLUE, ha="right", va="top",
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
def fig_decomposition():
    """One message: training fails by not placing the hidden unit; below ε ≈ 0.25 every run fails this way, and bias
    failures appear only in a narrow band around ε ≈ 0.4."""
    d = read("phase1_decomposition.csv")
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
    save(fig, "decomposition")
    return {"n_per_a": int(ns.min())}


# ------------------------------------------------------------------------------------------ thresholds against crossings
def fig_thresholds():
    """One message: at every a, free training places the hidden unit just above the conditional-minimisation threshold
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
    ax.text(0.62, 0.314, r"$R_{\mathrm{solve}}$ (conditional minimisation)", color=BLUE, ha="right", va="bottom")
    ax.text(0.28, 0.2135, r"$R_{\mathrm{glob}}$ (conditional minimisation)", color=BLUE, ha="right", va="center")
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
    """One message: along the conditional minimiser at a = 1.30, the class gap changes sign exactly once, at R_glob."""
    d = read("cond_audit_candidates.csv")
    g = d[d.a.round(2) == round(a, 2)]
    ret = g[g.status == "RETAINED"].drop_duplicates("s").sort_values("s")
    scan = read("cond_scan_certified_a130.csv")
    br = read("cond_certified_brackets.csv")
    br = br[br.a.round(2) == round(a, 2)].set_index("kind")
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.3))
    ax.axhline(0, color=GREY, lw=0.7, zorder=0)
    wg = 0.5 * (br.loc["glob", "w2_lo"] + br.loc["glob", "w2_hi"])
    ws = 0.5 * (br.loc["solve", "w2_lo"] + br.loc["solve", "w2_hi"])
    ax.plot([wg, wg], [-0.5, 0.06], color=BLUE, lw=1.0, zorder=1)
    ax.plot([ws, ws], [-0.5, 0.06], color=BLUE, lw=1.0, ls="--", zorder=1)
    ax.text(wg + 0.12, -0.40, r"$R_{\mathrm{glob}}$", color=BLUE, va="center")
    ax.text(ws + 0.12, -0.20, r"$R_{\mathrm{solve}}$", color=BLUE, va="center")
    ax.plot(ret.s, ret.gap, color=BLUE, lw=1.2, zorder=2)
    ax.plot(scan.s, scan.argmin_G, color=BLUE, marker="s", ms=3.4, mfc="white", ls="none", zorder=3)
    ax.text(9.0, 0.10, "conditional minimiser", color=BLUE, ha="center", va="bottom")
    ax.set_xlabel(f"output scale $|w_2|$ ($a={a:.2f}$)")
    ax.set_ylabel("class gap $G$ of the minimiser")
    ax.set_ylim(-0.5, 0.16)
    _clean(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "minimiser_gap")


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
    ax.plot(ret.s, ret.loss, color=BLUE, lw=1.1, zorder=3, label="conditional minimiser (retained)")
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
                        mfc=BLUE if filled else "white", mec=BLUE, capsize=1.5, elinewidth=0.7, ls="none")
            if i == 0:
                a2.text(xp, hi + 0.006, lab, color=BLUE, ha="center", va="bottom", rotation=90)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["$a=1.30$", "$a=1.50$"])
    a2.set_xlim(-0.55, 1.55); a2.set_ylim(0, 0.30)
    a2.set_ylabel("mean per-run |log error|")
    _clean(a1, a2)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "prospective_own")


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
    fig_minimiser_gap(); fig_candidates_appendix()
    fig_mirror_branches()
    fig_prospective()
    fig_prospective_own()
    print(out)
    audit()


if __name__ == "__main__":
    main()
