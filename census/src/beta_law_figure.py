"""Section 6 headline object: measured onset exponent vs 1/beta.

The law predicts exponent = -alpha * (1/beta), a line through the origin
whose slope is alpha -- measured independently from the budget sweep and
NEVER fitted to these families.  Family A (sin) is included as an
independent point measured before the constructed families existed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALPHA = 1.1172          # committed, from the 8-budget 1k-160k sweep


def family_exponents() -> pd.DataFrame:
    rows = []
    f = pd.read_csv(RESULTS / "family_onsets.csv")
    for name, sub in f.groupby("family"):
        ok = sub[sub.bracketed & sub.onset_eps.notna()]
        if len(ok) >= 2:
            s, _ = np.polyfit(np.log(ok.budget.values), np.log(ok.onset_eps.values), 1)
            resid = np.log(ok.onset_eps.values) - (
                s * np.log(ok.budget.values) + _)
            r2 = 1 - resid.var() / np.log(ok.onset_eps.values).var() \
                if np.log(ok.onset_eps.values).var() > 0 else float("nan")
            rows.append({"family": name, "beta": float(ok.beta.iloc[0]),
                         "inv_beta": 1.0 / float(ok.beta.iloc[0]),
                         "measured": s, "predicted": -ALPHA / float(ok.beta.iloc[0]),
                         "n_cells": len(ok), "r2": r2, "source": "constructed"})
    # family A: measured before the constructed families existed
    rows.append({"family": "familyA_sin", "beta": 1.5, "inv_beta": 1/1.5,
                 "measured": -0.7340, "predicted": -ALPHA / 1.5,
                 "n_cells": 6, "r2": 0.990, "source": "independent (pre-existing)"})
    return pd.DataFrame(rows).sort_values("inv_beta")


def main() -> None:
    d = family_exponents()
    print(d.round(4).to_string(index=False))
    d.to_csv(RESULTS / "beta_law_points.csv", index=False)

    fit = d[d.n_cells >= 2]
    if len(fit) >= 2:
        # slope through the origin: exponent = -alpha_eff * (1/beta)
        alpha_eff = -np.sum(fit.inv_beta * fit.measured) / np.sum(fit.inv_beta ** 2)
        pred = -alpha_eff * fit.inv_beta
        ss_res = float(((fit.measured - pred) ** 2).sum())
        ss_tot = float(((fit.measured - fit.measured.mean()) ** 2).sum())
        print(f"\nthrough-origin slope (alpha_eff) = {alpha_eff:.4f}")
        print(f"independently measured alpha     = {ALPHA:.4f}  "
              f"({100*abs(alpha_eff-ALPHA)/ALPHA:.1f}% apart)")
        print(f"R^2 about the predicted line     = {1 - ss_res/ss_tot:.4f}"
              if ss_tot > 0 else "")
        Path(RESULTS / "beta_law_fit.txt").write_text(
            f"alpha_eff={alpha_eff}\nalpha_measured={ALPHA}\n"
            f"n_points={len(fit)}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        xs = np.linspace(0, max(d.inv_beta) * 1.15, 50)
        ax.plot(xs, -ALPHA * xs, "k--", lw=1.2,
                label=rf"predicted: $-\alpha/\beta$, $\alpha={ALPHA:.3f}$ (measured)")
        for src, mark in (("constructed", "o"), ("independent (pre-existing)", "s")):
            s = d[d.source == src]
            if len(s):
                ax.scatter(s.inv_beta, s.measured, marker=mark, s=60, zorder=3,
                           label=src, edgecolor="k", linewidth=0.6)
        for _, r in d.iterrows():
            ax.annotate(rf"$\beta$={r.beta:.2f}", (r.inv_beta, r.measured),
                        textcoords="offset points", xytext=(6, -10), fontsize=8)
        ax.set_xlabel(r"$1/\beta$  (inverse fold-depth exponent)")
        ax.set_ylabel("measured onset exponent")
        ax.set_xlim(left=0); ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = RESULTS / "figures" / "beta_law.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"figure -> {out}")
    except ImportError:
        print("(matplotlib unavailable; points saved to beta_law_points.csv)")


if __name__ == "__main__":
    main()
