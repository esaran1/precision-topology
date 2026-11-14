"""Rebuttal figures 1-4 (Part C, 2026-11-14): the mechanism, the switch-scale law, the strictly causal lag forecasts, and
the scope by training regime.  Every plotted number is read from a committed artifact through the registered table and
scoring functions; nothing is re-derived.  The horizon figure (fig 5) is made by src/posthoc_horizon.py.

Author's figure rules: fonts >= 8 pt at the 5.5 in text width, no internal labels, <= 4 series per panel, colour plus
marker shape for every series, <= ~3 in tall.  Style (Times, 8-pt floor) and the layout check come from figures_v4.

    python -m src.figures_rebuttal       (the launcher: refuses to start without a separate, fresh gate entry,
                                          U2_GATE_DIR=results/rebuttal/figures python -m src.u2_gate gate <label> --disk-gb 15)

Writes results/rebuttal/figures/fig_{mechanism,switch_law,lag_causal,scope}.{pdf,png}, a caption file per figure
(fig_<name>_caption.md) and figure_data.json (the plotted numbers, for the ledger and the tests).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "rebuttal" / "figures"
BLUE, VERM, GREEN, GREY, ORANGE = "#0072B2", "#D55E00", "#009E73", "0.55", "#E69F00"
A_VALUES = (1.30, 1.50)
ALPHA_STAR = 1.7913244            # src/mechanism_w1_figure.py (the submitted figure's constant)


# ------------------------------------------------------------------------------------------ data (pure; tested)
def mech_crossings(path=None, chunksize=500_000):
    """Free-training crossing rows (budget 32,000) of the checkpoint table, streamed in chunks (the file is ~2.9 GB):
    a, |w1|, |w2| at the first placed step.  Same selection as the submitted mechanism figure."""
    path = RESULTS / "phase2b_checkpoints.csv" if path is None else Path(path)
    keep = []
    for ch in pd.read_csv(path, usecols=["a", "budget", "w1", "w2", "crossing"], chunksize=chunksize,
                          dtype={"crossing": str}):
        m = (ch.crossing.astype(str).str.lower() == "true") & (ch.budget == 32_000)
        if m.any():
            keep.append(ch.loc[m, ["a", "w1", "w2"]])
    d = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=["a", "w1", "w2"])
    d["a"] = d.a.round(2)
    d["w1"], d["w2"] = d.w1.abs(), d.w2.abs()
    return d


def mech_path():
    return pd.read_csv(RESULTS / "mechanism_w1_path.csv")


def switch_law_data():
    """The certified global switch scale s = |w2| at each certified a (two committed bracket tables), and the
    equation-(5) constants: A* bracket (limit_switch.csv), A* Krawczyk value and c_s = A'/A* (first_order_c1.csv)."""
    br = pd.read_csv(RESULTS / "cond_certified_brackets.csv")
    g = br[br.kind == "glob"].sort_values("a")
    finite = pd.DataFrame({"a": g.a.round(2).to_numpy(), "eps": (g.a - 1).to_numpy(), "s_lo": g.w2_lo.to_numpy(),
                           "s_hi": g.w2_hi.to_numpy(), "source": "finite-a certificate"})
    c1 = pd.read_csv(RESULTS / "c1_followup_finite.csv").sort_values("a")
    small = pd.DataFrame({"a": c1.a.round(2).to_numpy(), "eps": c1.eps.to_numpy(), "s_lo": c1.s_lo.to_numpy(),
                          "s_hi": c1.s_hi.to_numpy(), "source": "small-eps certificate"})
    lsw = pd.read_csv(RESULTS / "limit_switch.csv").iloc[0]
    fo = pd.read_csv(RESULTS / "first_order_c1.csv").iloc[0]
    const = {"A_lo": float(lsw.A_lo), "A_hi": float(lsw.A_hi), "A_star": float(fo.A_star_lo),
             "c_s": float(fo.A1_over_A_lo)}
    return pd.concat([finite, small], ignore_index=True), const


def law(eps, A, c_s):
    eps = np.asarray(eps, float)
    return A * eps ** -1.5 * (1 + c_s * eps)


def lag_w1_158():
    """Registered (1C, width-1 sine a = 1.58, SGD, f = 0.90): per scored run r_obs and r_fc, exactly as scored."""
    from . import phase1c as P
    R = P.table("W1", P._rows(P._runs_file("W1")), P._rows(P._obs_file("W1")))
    sc = P.score_arm_1c("W1", R)
    A = {k: P._arr(v) for k, v in R.items() if k in ("s_obs", "s_sw", "r_fc", "t_fc", "t_sw_fc")}
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = A["s_obs"] / A["s_sw"] - 1
    idx = np.array(sc["scored_index"], int)
    has = P.is_forecast(A["t_fc"], A["t_sw_fc"])[idx]
    return {"r_obs": r_obs[idx][has], "r_fc": A["r_fc"][idx][has], "n_scored": int(sc["n_scored"]),
            "n_miss": int(sc["n_scored_miss"]), "f": float(P.F["W1"]), "c1_within": int(sc["C1"]["n_within"]),
            "median_ratio": float(sc["C3"]["median_ratio"])}


def lag_adam_177():
    """Registered (width-1 sine a = 1.77, Adam, f = 0.90): through the post-registration scoring entry point."""
    from . import trackA_causal as T
    from . import trackA_score as TS
    rows, obs = T._rows(T.OUT / "runs.jsonl"), T._rows(T.OUT / "observed.jsonl")
    R = TS.documented_columns(T.table(rows, obs, T.F, "primary"))
    sc = T.score_track_a(R, T.TOLERANCES)
    A = {k: np.asarray(R[k], float) for k in ("s_obs", "s_sw", "r_fc", "t_fc", "t_sw_fc")}
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = A["s_obs"] / A["s_sw"] - 1
    idx = np.array(sc["scored_index"], int)
    from . import phase1c as P
    has = P.is_forecast(A["t_fc"], A["t_sw_fc"])[idx]
    return {"r_obs": r_obs[idx][has], "r_fc": A["r_fc"][idx][has], "n_scored": int(sc["n_scored"]),
            "n_miss": int(sc["n_scored_miss"]), "f": float(T.F), "c1_within": int(sc["C1"]["n_within"]),
            "median_ratio": float(sc["C3"]["median_ratio"])}


def lag_165(opt):
    """POST HOC causal re-score (width-1 sine a = 1.65, `opt` sgd|adam, f = 0.90): the committed re-score's arrays."""
    from . import phase1b as B
    arm = f"track_a/{opt}"
    d = B.registered_table(arm)
    rows = B._rows(B._runs_path("track_a"))
    R = B.arm_arrays(arm, d, rows, "fc", B.F[arm])
    res = B.rescore(B.TOL_KEY[arm], d.reg_scored.to_numpy(bool), R["t_c"], R["t_fc"], R["t_sw_fc"], R["r_fc"],
                    R["t_obs"], R["t_sw"], R["r_obs"], R["nan_identical"], True)
    m = res["scored_mask"] & res["has_fc_mask"]
    return {"r_obs": R["r_obs"][m], "r_fc": R["r_fc"][m], "n_scored": int(res["n_scored"]),
            "n_miss": int(res["n_scored_miss"]), "f": float(B.F[arm]), "c1_within": int(res["criteria"]["C1"]["n_within"]),
            "median_ratio": float(res["criteria"]["C3"]["median_ratio"])}


def scope_forecasts():
    """Regime (a): crossing-step forecasts within tolerance, at the registered f and (where scored) at f = 0.90."""
    C = json.loads((RESULTS / "phase1c" / "scores.json").read_text())["arms"]
    P3 = json.loads((RESULTS / "phase3" / "scores.json").read_text())["arms"]
    A2 = json.loads((RESULTS / "phase2a" / "scores.json").read_text())["criteria"]
    rows = []
    for arm, lab in (("G", "width-1 GELU, warm start"), ("T", "width-2, output LR ÷1024"),
                     ("Tp", "width-2, branch-point start")):
        o, of = C[arm], C[arm]["DESCRIPTIVE"]["other_f"]
        rows.append({"label": lab, "k": o["C1"]["n_within"], "n": o["C1"]["n"], "f": o["f"],
                     "k90": of["criteria_stats"]["C1"]["n_within"], "n90": of["criteria_stats"]["C1"]["n"],
                     "f90": of["f"]})
    for arm, lab in (("Q", "width-4, branch-point start"), ("S", "width-4, branch-point start, sign")):
        o = P3[arm]
        rows.append({"label": lab, "k": o["C1"]["n_within"], "n": o["C1"]["n"], "f": o["f"],
                     "k90": None, "n90": None, "f90": None})
    rows.append({"label": "tanh width-4, delay (22 rates)", "k": A2["C1"]["n_within"], "n": A2["C1"]["n"],
                 "f": 0.95, "k90": None, "n90": None, "f90": None})
    return rows


def scope_standard():
    """Regime (b): fraction of standard-training crossings with |κχ| <= 0.1 (descriptive map)."""
    K = json.loads((RESULTS / "phase2c" / "kappa_chi_map.json").read_text())["pooled"]
    out = []
    for key, lab in (("width-1 sine | Adam", "width-1 sine, Adam"), ("width-1 sine | SGD", "width-1 sine, SGD"),
                     ("width-1 GELU | Adam", "width-1 GELU, Adam"), ("band task in R^d | Adam", "band task, Adam"),
                     ("width 2 | Adam", "width-2 sine, Adam")):
        s = K[key]["summary"]
        out.append({"label": lab, "k": int(s["n_abs_le_0.1"]), "n": int(s["n"]), "frac": float(s["frac_abs_le_0.1"])})
    return out


def scope_mlp():
    """The registered MLP null: median paired change in randomised-feature accuracy at matched loss, 95% percentile
    bootstrap interval of the median (registered procedure), output LR ÷16 vs the baseline arm (the registered test)."""
    L = json.loads((RESULTS / "trackL" / "scores.json").read_text())["arms"]
    out = []
    for arm, lab in (("out16", "output LR ÷16"),):
        for lv in ("bce_0.6", "bce_0.3", "bce_0.03"):
            c = L[arm]["cells_rand"][lv]
            out.append({"arm": lab, "level": lv.replace("bce_", "loss "), "median": c["median"], "lo": c["lo"],
                        "hi": c["hi"], "n": c["n_both"]})
    return out


# ------------------------------------------------------------------------------------------ figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    from . import figures_v4 as V4                   # rcParams (Times, 8-pt floor), layout_check, clopper
    return V4


def _clean(*axes):
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)


LAYOUT: dict[str, list[str]] = {}


def _save(fig, name):
    import matplotlib.pyplot as plt
    V4 = _style()
    LAYOUT[name] = V4.layout_check(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"fig_{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"fig_{name}.png", dpi=300)
    plt.close(fig)
    print(f"  fig_{name}" + (f"  LAYOUT: {LAYOUT[name]}" if LAYOUT[name] else ""))


def fig_mechanism(cross=None):
    V4 = _style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    d = mech_path()
    cr = mech_crossings() if cross is None else cross
    fig, axes = plt.subplots(1, 2, figsize=(V4.TEXT_W, 2.4), sharey=True)
    stats = {}
    for ax, a in zip(axes, A_VALUES):
        g = d[d.a.round(2) == a].sort_values("x")
        wg = float(g.w2_glob.iloc[0])
        c = cr[cr.a.round(2) == a]
        ax.axhline(ALPHA_STAR, color=GREY, lw=0.8, ls="--", zorder=0)
        ax.axhline(a / 1.4, color=GREY, lw=0.8, ls=":", zorder=0)
        ax.axvline(1.0, color=BLUE, lw=0.8, zorder=0)
        ax.plot(g.x, g.w1, color=BLUE, lw=1.2, marker="o", ms=2.6, zorder=2)
        ax.plot(c.w2 / wg, c.w1, ls="none", marker="D", ms=3.0, mfc="white", mec=VERM, mew=0.8, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"output scale $s/s_{\mathrm{switch}}$" + f"  ($a = {a:.2f}$)")
        x0 = 0.012 if a == 1.30 else 0.02
        ax.text(x0, ALPHA_STAR + 0.04, "class-mean optimum", color="0.35", va="bottom")
        ax.text(x0, a / 1.4 - 0.05, r"placement bound $a/1.4$", color="0.35", va="top")
        _clean(ax)
        stats[a] = {"s_switch": wg, "n_cross": int(len(c)), "n_path": int(len(g)),
                    "cross_below_bound": float((c.w1 < a / 1.4).mean()) if len(c) else None}
    axes[1].text(1.12, 1.62, "minimiser", color=BLUE, ha="left")
    axes[1].text(1.12, 1.50, "training\ncrossings", color=VERM, ha="left", va="top")
    axes[0].set_ylabel(r"$|w_1|$")
    fig.tight_layout(pad=0.3)
    _save(fig, "mechanism")
    return stats


def fig_switch_law():
    V4 = _style()
    import matplotlib.pyplot as plt
    d, k = switch_law_data()
    fig, ax = plt.subplots(figsize=(V4.TEXT_W, 2.6))
    e = np.geomspace(0.06, 0.66, 200)
    ax.fill_between(e, law(e, k["A_lo"], k["c_s"]), law(e, k["A_hi"], k["c_s"]), color=BLUE, alpha=0.18, lw=0,
                    zorder=1)
    h = [ax.plot(e, law(e, k["A_star"], k["c_s"]), color=BLUE, lw=1.2, ls="-", zorder=2)[0],
         ax.plot(e, law(e, k["A_star"], 0.0), color=GREY, lw=1.0, ls="--", zorder=2)[0]]
    for src, mk, col in (("finite-a certificate", "o", "k"), ("small-eps certificate", "s", VERM)):
        g = d[d.source == src]
        mid = 0.5 * (g.s_lo + g.s_hi)
        h.append(ax.errorbar(g.eps, mid, yerr=[mid - g.s_lo, g.s_hi - mid], ls="none", marker=mk, ms=3.6, color=col,
                             mfc="white" if mk == "s" else col, mec=col, capsize=1.5, elinewidth=0.7, zorder=4))
    ax.legend(h, [r"$A^{*}\varepsilon^{-3/2}(1+c_s\varepsilon)$", r"$A^{*}\varepsilon^{-3/2}$ (leading term)",
                  "certified, a = 1.30–1.60", "certified, a = 1.08–1.12"], loc="lower left")
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.08, 0.1, 0.2, 0.3, 0.4, 0.6]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 40]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.06, 0.66)
    ax.set_xlabel(r"$\varepsilon = a - 1$")
    ax.set_ylabel(r"switch scale $s$")
    _clean(ax)
    fig.tight_layout(pad=0.3)
    _save(fig, "switch_law")
    rel = (0.5 * (d.s_lo + d.s_hi)) / law(d.eps, k["A_star"], k["c_s"]) - 1
    return {"n_finite": int((d.source == "finite-a certificate").sum()), "n_small": int((d.source != "finite-a certificate").sum()),
            "max_abs_rel_dev_from_law": float(np.abs(rel).max()), **k}


def fig_lag_causal(data=None):
    V4 = _style()
    import matplotlib.pyplot as plt
    D = data or {"sgd_158": lag_w1_158(), "sgd_165": lag_165("sgd"), "adam_177": lag_adam_177(),
                 "adam_165": lag_165("adam")}
    fig, axes = plt.subplots(1, 2, figsize=(V4.TEXT_W, 2.7))
    logs = {}
    for ax, (kreg, lreg), (kph, lph), ttl in (
            (axes[0], ("sgd_158", "a = 1.58, registered"), ("sgd_165", "a = 1.65, post hoc"), "(a) SGD"),
            (axes[1], ("adam_177", "a = 1.77, registered"), ("adam_165", "a = 1.65, post hoc"), "(b) Adam")):
        ph, rg = D[kph], D[kreg]
        h2 = ax.plot(ph["r_fc"], ph["r_obs"], ls="none", marker="^", ms=3.0, mfc="white", mec=GREY, mew=0.7,
                     zorder=2)[0]
        h1 = ax.plot(rg["r_fc"], rg["r_obs"], ls="none", marker="o", ms=2.8, color=BLUE, alpha=0.85, zorder=3)[0]
        v = np.concatenate([ph["r_fc"], ph["r_obs"], rg["r_fc"], rg["r_obs"]])
        v = v[np.isfinite(v)]
        lo = min(0.0, float(np.quantile(v, 0.01)))
        hi = float(v.max()) * 1.08
        lo -= 0.03 * (hi - lo)
        ax.plot([lo, hi], [lo, hi], color="0.35", lw=0.8, ls=(0, (4, 2)), zorder=1)
        clip = {}
        for key, col, mfc in ((kreg, BLUE, BLUE), (kph, GREY, "white")):     # off-scale runs: drawn at the edge
            x, y = D[key]["r_fc"], D[key]["r_obs"]
            off = (x < lo) | (y < lo)
            clip[key] = [(float(xi), float(yi)) for xi, yi in zip(x[off], y[off])]
            if off.any():
                ax.plot(np.maximum(x[off], lo), np.maximum(y[off], lo), ls="none",
                        marker="<" if (x[off] < lo).all() else "v", ms=4.0, color=col, mfc=mfc, mec=col,
                        clip_on=False, zorder=4)
        logs[ttl] = clip
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("forecast lag")
        ax.set_title(ttl, loc="left", fontsize=V4.NOTE_PT)
        ax.legend([h1, h2], [lreg, lph], loc="upper left", handletextpad=0.2)
        _clean(ax)
    axes[0].set_ylabel("observed lag")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    _save(fig, "lag_causal")
    return {k: {kk: (vv if not isinstance(vv, np.ndarray) else int(len(vv))) for kk, vv in v.items()}
            for k, v in D.items()} | {"off_scale": {k: v for c in logs.values() for k, v in c.items()}}


def fig_scope():
    V4 = _style()
    import matplotlib.pyplot as plt
    fa, sb, ml = scope_forecasts(), scope_standard(), scope_mlp()
    W, H = V4.TEXT_W, 3.0
    fig = plt.figure(figsize=(W, H))
    box = lambda l, b, w, h: fig.add_axes([l / W, b / H, w / W, h / H])          # noqa: E731  (inches)
    axa = box(1.72, 0.52, 1.00, 2.13)
    axb = box(4.00, 1.62, 1.40, 1.03)
    axc = box(4.00, 0.52, 1.40, 0.55)
    y = np.arange(len(fa))[::-1]
    for yi, r in zip(y, fa):
        lo, hi = V4.clopper(r["k"], r["n"]); p = r["k"] / r["n"]
        axa.errorbar([p], [yi + 0.12], xerr=[[p - lo], [hi - p]], marker="o", ms=3.4, color=BLUE, ls="none",
                     capsize=1.5, elinewidth=0.7, zorder=3)
        if r["k90"] is not None:
            lo, hi = V4.clopper(r["k90"], r["n90"]); p = r["k90"] / r["n90"]
            axa.errorbar([p], [yi - 0.12], xerr=[[p - lo], [hi - p]], marker="s", ms=3.4, color=VERM, mfc="white",
                         ls="none", capsize=1.5, elinewidth=0.7, zorder=3)
    axa.axvline(0.8, color="0.35", lw=0.7, ls=":", zorder=0)
    axa.set_yticks(y); axa.set_yticklabels([r["label"] for r in fa])
    axa.set_xlim(-0.03, 1.03); axa.set_ylim(-0.6, len(fa) - 0.4)
    axa.set_xlabel("crossing forecast\nwithin tolerance")
    axa.set_title("(a) slow growth or branch start", loc="right", fontsize=V4.NOTE_PT)
    from matplotlib.lines import Line2D
    axa.legend([Line2D([], [], marker="o", color=BLUE, ls="none", ms=3.4),
                Line2D([], [], marker="s", color=VERM, mfc="white", ls="none", ms=3.4)],
               ["0.95", "0.90"], title="cutoff", loc="lower left", bbox_to_anchor=(0.04, 0.02), handletextpad=0.2,
               handlelength=1.0)
    yb = np.arange(len(sb))[::-1]
    for yi, r in zip(yb, sb):
        lo, hi = V4.clopper(r["k"], r["n"]); p = r["k"] / r["n"]
        axb.errorbar([p], [yi], xerr=[[p - lo], [hi - p]], marker="D", ms=3.2, color=GREEN, ls="none", capsize=1.5,
                     elinewidth=0.7, zorder=3)
    axb.set_yticks(yb); axb.set_yticklabels([r["label"] for r in sb])
    axb.set_xlim(-0.03, 1.03); axb.set_ylim(-0.5, len(sb) - 0.5)
    axb.set_xlabel(r"crossings with $|\kappa\chi|\leq 0.1$")
    axb.set_title("(b) standard training", loc="left", fontsize=V4.NOTE_PT)
    levels = list(dict.fromkeys(r["level"] for r in ml))
    yc = np.arange(len(levels))[::-1]
    for yi, r in zip(yc, ml):
        axc.errorbar([r["median"]], [yi], xerr=[[r["median"] - r["lo"]], [r["hi"] - r["median"]]], marker="s",
                     ms=3.2, color=ORANGE, ls="none", capsize=1.5, elinewidth=0.7, zorder=3)
    axc.axvline(0, color="0.35", lw=0.7, zorder=0)
    axc.set_yticks(yc); axc.set_yticklabels(levels); axc.set_ylim(-0.5, len(levels) - 0.5)
    axc.set_xlabel("change in accuracy on the\nrandomised feature")
    axc.set_title("(c) standard MLP", loc="left", fontsize=V4.NOTE_PT)
    _clean(axa, axb, axc)
    _save(fig, "scope")
    return {"forecasts": fa, "standard": sb, "mlp": ml}


# ------------------------------------------------------------------------------------------ captions
_RUNSET = {"sgd_158": "a = 1.58 SGD", "sgd_165": "a = 1.65 SGD", "adam_177": "a = 1.77 Adam", "adam_165": "a = 1.65 Adam"}


def _offscale_text(off):
    parts = [f"{_RUNSET[k]}: forecast {x:.2f}, observed {y:.2f}" for k, pts in off.items() for x, y in pts]
    if not parts:
        return "Every run with a forecast is inside the axes."
    return (f"{len(parts)} run(s) off the axes, drawn as a triangle at the edge (" + "; ".join(parts) + ").")


def captions(st):
    m, sw, lg, sc = st["mechanism"], st["switch_law"], st["lag_causal"], st["scope"]
    n13, n15 = m[1.30]["n_cross"], m[1.50]["n_cross"]
    cap = {
        "mechanism": (
            "As the output scale grows towards the switch scale, the fixed-scale minimiser's input weight falls from "
            "the class-mean optimum to below the placement bound, and free training crosses at the switch scale, on "
            "this path and below the bound.",
            f"Width-one sine network on the 800-point population objective, a = 1.30 and 1.50; minimiser path from a "
            f"validated search ({m[1.30]['n_path']} and {m[1.50]['n_path']} scales), not a certificate; the switch scale "
            f"is the certified bracket midpoint. Diamonds: free-training crossings (the submitted figure's crossing set, "
            f"budget 32,000 steps), n = {n13} and {n15}.",
            "No interval is drawn: the path is a deterministic minimisation and the crossings are individual runs."),
        "switch_law": (
            "The certified global switch scale follows the branch law s = A*ε^{-3/2}(1 + c_s ε) of equation (5) across "
            "ε = 0.08–0.60, and the first-order correction is needed at finite ε.",
            f"Certified brackets of the global placement threshold: {sw['n_finite']} finite activation values "
            f"(circles) and {sw['n_small']} small-ε values (squares). Curve: A* = {sw['A_star']:.6f} (Krawczyk value), "
            f"c_s = {sw['c_s']:.5f}; band: A* ∈ [{sw['A_lo']:.5f}, {sw['A_hi']:.4f}]; dashed: the leading term alone. "
            f"Largest relative distance of a bracket midpoint from the curve: {100 * sw['max_abs_rel_dev_from_law']:.1f}%.",
            "Error bars are the certified brackets (computer-assisted interval enclosures), not statistical intervals; "
            "the law is proved for the continued branch and tied to the global threshold by these certificates."),
        "lag_causal": (
            "Strictly causal per-run forecasts of the lag track the observed lag within the registered ratio bands in "
            "the two registered tests; in the post hoc re-score at a = 1.65 the observed lag exceeds the forecast by a "
            f"median {100 * (lg['sgd_165']['median_ratio'] - 1):.0f}% (SGD) and "
            f"{100 * (lg['adam_165']['median_ratio'] - 1):.0f}% (Adam).",
            f"Width-one sine network. Every forecast reads the run only before its cutoff, the first step where the "
            f"output scale reaches 0.90 of the run's causally estimated switch scale (f = 0.90). (a) SGD: registered test at a = 1.58 "
            f"(n = {lg['sgd_158']['n_scored']} scored, {lg['sgd_158']['n_miss']} without a forecast); post hoc re-score "
            f"at a = 1.65 (n = {lg['sgd_165']['n_scored']}, {lg['sgd_165']['n_miss']} without a forecast). (b) Adam: "
            f"registered test at a = 1.77 (n = {lg['adam_177']['n_scored']}, {lg['adam_177']['n_miss']} without a "
            f"forecast, not plotted); post hoc re-score at a = 1.65 (n = {lg['adam_165']['n_scored']}, "
            f"{lg['adam_165']['n_miss']} without a forecast, not plotted). Lag: relative excess of the crossing scale "
            f"over the switch scale. Dashed: identity. " + _offscale_text(lg.get("off_scale", {})),
            "Each point is one run; no interval is drawn. Median observed/forecast ratio: "
            f"{lg['sgd_158']['median_ratio']:.3f} (a = 1.58), {lg['sgd_165']['median_ratio']:.3f} (a = 1.65 SGD), "
            f"{lg['adam_177']['median_ratio']:.3f} (a = 1.77), {lg['adam_165']['median_ratio']:.3f} (a = 1.65 Adam)."),
        "scope": (
            "Every registered causal pass outside the width-one sine network is in the slow-growth or branch-start "
            "regime; in standard training the lag law's regime is rare at width two and the output-layer lever has no "
            "effect in a standard MLP.",
            "(a) Registered strictly causal forecasts of the crossing step: fraction of scored runs within the registered "
            "tolerance at the registered cutoff 0.95 (filled; 0.95 was chosen after the pilot) and at 0.90 (open; "
            "scored descriptively, not computed for the width-4 and tanh tests). n = "
            + ", ".join(f"{r['n']}" for r in sc["forecasts"]) + " (top to bottom; the tanh row counts 22 rates, each a "
            "deterministic run, and its tolerance is 15% of the forecast delay). Dotted: the registered 80% bar. "
            "(b) Standard training, descriptive: fraction of crossings with predicted lag |κχ| ≤ 0.1, n = "
            + ", ".join(f"{r['n']}" for r in sc["standard"]) + ". (c) Registered test in a ReLU MLP (6144-256-256-1) on "
            "MNIST-CIFAR dominoes, predicted to show no effect: median paired change, output-layer learning rate ÷16 against the "
            "baseline arm, at three matched training losses, n = 40 seeds.",
            "(a, b) Clopper–Pearson 95% intervals. (c) 95% percentile bootstrap intervals of the median paired "
            "difference (10,000 resamples, the registered procedure)."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (msg, pop, unc) in cap.items():
        (OUT / f"fig_{name}_caption.md").write_text(
            f"# Figure: {name.replace('_', ' ')}\n\n**Message.** {msg}\n\n**Population and n.** {pop}\n\n"
            f"**Uncertainty.** {unc}\n")


def main():
    from . import u2_gate as G
    G.require_gate("figures_rebuttal")
    os.nice(15)
    import resource

    def guard():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # bytes on macOS
        if rss > 3e9:
            raise SystemExit(f"STOP: RSS {rss / 1e9:.2f} GB > 3 GB")
    st = {}
    for name, fn in (("mechanism", fig_mechanism), ("switch_law", fig_switch_law), ("lag_causal", fig_lag_causal),
                     ("scope", fig_scope)):
        st[name] = fn()
        guard()
    captions(st)

    def _j(o):
        if isinstance(o, dict):
            return {str(k): _j(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_j(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    (OUT / "figure_data.json").write_text(json.dumps(_j({**st, "layout": LAYOUT}), indent=1) + "\n")
    print(json.dumps(_j({"layout": LAYOUT}), indent=1))


if __name__ == "__main__":
    main()
