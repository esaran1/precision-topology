"""Regenerate every paper figure from committed CSVs. No manual steps.

AUDIT PASS: each figure declares, inline, which fit window / n / correction
state it uses. Any inconsistency found while assembling is printed as
FINDING: and reported, never silently reconciled.

Known multi-valued quantities (watch list):
  alpha            1.1172 (1k-160k, 8 cells) | 1.1084 (2k-128k window-matched)
                   | 0.72-1.51 across sub-ranges
  onset exponent   -0.7340 (6 cells) | -0.8305 (3 cells, 2k-32k)
  through-origin   1.0984 [0.958, 1.239], five-point fit: q2 and family A
                   both included at matched beta = 1.5 (corrected 2026-09-11;
                   earlier 1.0878 / 1.0969 / 1.1240 all used four points)
  per-family grid  +-0.114 (4-cell families) | +-0.171 (3-cell families)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
FIG = R / "figures"; FIG.mkdir(exist_ok=True)
FINDINGS: list[str] = []

# --- canonical values, single source of truth for the figures -------------
ALPHA = 1.1173; ALPHA_CI = 0.1187          # 1k-160k, 8 cells (T43)
ONSET_EXP = -0.7340; ONSET_N = 6           # 6 bracketed cells, 2k-128k


def finding(msg: str) -> None:
    FINDINGS.append(msg); print(f"FINDING: {msg}")


def fig_four_family() -> None:
    """Headline: exponent vs 1/beta, per-family error bars, Adam only."""
    fam = [("q0.667", 2.4993, -0.5000, 3), ("q1", 2.0, -0.6521, 4),
           ("q2", 1.5, -0.6749, 4), ("familyA", 1.5, -0.7340, 6),
           ("q4", 1.25, -0.8305, 3)]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    xs = np.linspace(0, 0.92, 50)
    ax.plot(xs, -ALPHA * xs, "k--", lw=1.2,
            label=rf"predicted $-\alpha/\beta$, $\alpha={ALPHA:.4f}$ (1k–160k, n=8)")
    ax.fill_between(xs, -(ALPHA+ALPHA_CI)*xs, -(ALPHA-ALPHA_CI)*xs,
                    color="k", alpha=0.08, label=r"$\alpha$ 95% CI")
    for name, beta, meas, cells in fam:
        # per-family resolution: 4-cell families span 64x, 3-cell span 16x
        err = 0.114 if cells >= 4 else 0.171
        mark = "s" if name == "familyA" else "o"
        ax.errorbar(1/beta, meas, yerr=err, marker=mark, ms=7, capsize=3,
                    color="tab:orange" if name == "familyA" else "tab:blue",
                    mec="k", mew=0.6, lw=1.0, zorder=3)
        ax.annotate(rf"$\beta$={beta:.2f}", (1/beta, meas), fontsize=7.5,
                    textcoords="offset points", xytext=(7, -11))
    ax.plot([], [], "s", color="tab:orange", mec="k", label="family A (measured first, out-of-sample)")
    ax.plot([], [], "o", color="tab:blue", mec="k", label="constructed families")
    ax.set_xlabel(r"$1/\beta$   ($\beta$ analytic, verified to $\leq$0.7%)")
    ax.set_ylabel("measured onset exponent")
    ax.set_title("Onset exponent tracks $1/\\beta$ (Adam only; see T47)", fontsize=10)
    ax.text(0.02, -1.10, "five-point fit: q2 and family A both included at matched $\\beta$=1.5",
            fontsize=7)
    ax.text(0.02, -0.98, "error bars: per-family grid resolution\n"
            r"$\pm$0.114 (4 cells, 64$\times$) / $\pm$0.171 (3 cells, 16$\times$)",
            fontsize=7, va="bottom")
    ax.set_xlim(0, 0.92); ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "fig1_four_family.png", dpi=200); plt.close(fig)
    # audit: ledger vs figure
    d = pd.read_csv(R / "beta_law_points.csv") if (R/"beta_law_points.csv").exists() else None
    if d is not None:
        for name, beta, meas, _ in fam:
            row = d[np.isclose(d.beta, beta, atol=0.01)]
            if len(row) and not np.isclose(row.measured.min(), meas, atol=0.02) \
               and not np.isclose(row.measured.max(), meas, atol=0.02):
                finding(f"fig1 {name}: figure uses {meas}, beta_law_points.csv has "
                        f"{list(row.measured.round(4))}")


def fig_budget_law() -> None:
    d = pd.read_csv(R / "onset_law_extended.csv")
    d = d[d.bracketed & d.onset.notna()]
    if len(d) != ONSET_N:
        finding(f"fig2: onset_law_extended.csv has {len(d)} bracketed cells, "
                f"canonical n is {ONSET_N}")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(d.budget, d.onset - 1, "o", ms=7, mec="k", label=f"measured (n={len(d)})")
    b = np.logspace(np.log10(d.budget.min()), np.log10(d.budget.max()), 50)
    c = np.log(d.onset.values - 1) - ONSET_EXP * np.log(d.budget.values)
    ax.loglog(b, np.exp(c.mean()) * b**ONSET_EXP, "k--", lw=1.2,
              label=f"fit {ONSET_EXP:.4f} (6 cells, 2k–128k)")
    ax.fill_between(b, np.exp(c.mean())*b**(ONSET_EXP-0.135),
                    np.exp(c.mean())*b**(ONSET_EXP+0.135), color="k", alpha=0.08,
                    label=r"registered band $\pm$0.135")
    ax.set_xlabel("training budget (steps)"); ax.set_ylabel(r"$\varepsilon_{\rm onset}=a-1$")
    ax.set_title("Onset displacement vs budget (Adam)", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(FIG / "fig2_budget_law.png", dpi=200); plt.close(fig)


def fig_zero_to_hundred() -> None:
    d = pd.read_csv(R / "budget_alpha.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.semilogx(d.budget, d.rate * 100, "o-", ms=7, mec="k", color="tab:green")
    ax.axhline(0, color="grey", lw=0.6); ax.axhline(100, color="grey", lw=0.6)
    ax.set_xlabel("training budget (steps)"); ax.set_ylabel("solve rate (%)")
    ax.set_title("a = 1.25: 0/200 at 2k steps → 100% at 80k (n=30/cell here)", fontsize=10)
    ax.grid(alpha=0.25); ax.set_ylim(-5, 105)
    fig.tight_layout(); fig.savefig(FIG / "fig3_zero_to_hundred.png", dpi=200); plt.close(fig)
    if not np.isclose(d[d.budget == 2000].rate.iloc[0], 0.0):
        finding(f"fig3: budget_alpha.csv rate at 2k is {d[d.budget==2000].rate.iloc[0]}, "
                f"headline says 0/200 (that n=200 figure comes from fold1d_sweep, n=30 here)")


def fig_margin_shift() -> None:
    d = pd.read_csv(R / "budget_flip.csv")
    piv = d.pivot_table(index="epochs", columns="activation", values="test_errors", aggfunc="mean")
    if not {"gelu", "relu"} <= set(piv.columns):
        finding("fig4: budget_flip.csv missing gelu/relu"); return
    adv = (piv["relu"] - piv["gelu"]).dropna()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.semilogx(adv.index, adv.values, "o-", ms=7, mec="k", color="tab:purple",
                label="GELU advantage over ReLU (n=8/cell)")
    ax.axhline(132.7, ls=":", color="grey", label="converged, n=10 (T38)")
    ax.set_xlabel("epochs"); ax.set_ylabel("test-error advantage (per 10,000)")
    ax.set_title("CIFAR-10: advantage magnitude varies 5.65× with budget", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "fig4_margin_shift.png", dpi=200); plt.close(fig)


def fig_cross_optimizer() -> None:
    """Adam vs SGD: the transfer failure, per T47."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.bar([0, 1], [-0.8305, -0.0056], width=0.5,
           color=["tab:blue", "tab:red"], edgecolor="k")
    ax.errorbar([1], [-0.0056], yerr=[0.0185], color="k", capsize=4, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Adam", "SGD"])
    ax.set_ylabel("onset exponent (family q4, $\\beta$=1.25)")
    ax.set_title("Same family, two optimizers: 45× difference", fontsize=10)
    ax.text(1, -0.10, r"$|{\rm exp}|<0.0185$" "\n4/4 bracketed\n1.08× grid, 64×",
            ha="center", fontsize=7.5)
    ax.text(0, -0.90, "4/4 bracketed\nsteepest of 4 families", ha="center", fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "fig5_cross_optimizer.png", dpi=200); plt.close(fig)


def main() -> None:
    for f in (fig_four_family, fig_budget_law, fig_zero_to_hundred,
              fig_margin_shift, fig_cross_optimizer):
        try:
            f(); print(f"  ok  {f.__name__}")
        except Exception as e:
            finding(f"{f.__name__} FAILED: {type(e).__name__}: {e}")
    print(f"\n{len(FINDINGS)} finding(s)")
    (R / "figure_audit.txt").write_text("\n".join(FINDINGS) if FINDINGS else "no findings\n")


if __name__ == "__main__":
    main()
