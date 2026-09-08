"""Verify every headline number in CLAIMS.md against its committed artifact.

Recomputes from CSVs rather than trusting the ledger text. Any mismatch is
printed as FINDING: and written to results/ledger_verification.txt.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
F: list[str] = []


def chk(label, got, want, tol=0.01):
    ok = (got is not None) and abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'MISMATCH'} {label}: artifact={got} ledger={want}")
    if not ok:
        F.append(f"{label}: artifact={got} ledger={want}")


def main() -> None:
    print("T1/T2 monotonic zero")
    d = pd.read_csv(R / "monotonic_zero_decomposition.csv")
    chk("total runs", float(d.runs.sum()), 5580.0, 0)
    chk("separations", float(d.separations.sum()), 0.0, 0)

    print("T43 budget law")
    d = pd.read_csv(R / "onset_law_extended.csv"); d = d[d.bracketed & d.onset.notna()]
    chk("bracketed cells", float(len(d)), 6.0, 0)
    s = np.polyfit(np.log(d.budget), np.log(d.onset - 1), 1)[0]
    chk("onset exponent", s, -0.7340, 0.002)
    a = pd.read_csv(R / "budget_alpha.csv")
    al = np.polyfit(np.log(a.budget), np.log(a.w2_med), 1)[0]
    chk("alpha (1k-160k)", al, 1.1172, 0.002)
    chk("rate at 80k", float(a[a.budget == 80000].rate.iloc[0]), 1.0, 0)

    print("T44 four-family")
    fam = {"q4": (1.25, -0.8305), "q2": (1.50, -0.6749),
           "q1": (2.00, -0.6521), "q0.667": (2.4993, -0.5000)}
    d = pd.read_csv(R / "family_onsets.csv"); d = d[d.bracketed & d.onset_eps.notna()]
    for name, sub in d.groupby("family"):
        key = name.split("_")[0]
        if key in fam and len(sub) >= 2:
            s = np.polyfit(np.log(sub.budget), np.log(sub.onset_eps), 1)[0]
            chk(f"{key} exponent", s, fam[key][1], 0.02)

    print("T47 cross-optimizer")
    d = pd.read_csv(R / "sgd_q4_refined.csv"); d = d[d.bracketed & d.onset_eps.notna()]
    chk("q4 SGD bracketed", float(len(d)), 4.0, 0)
    s = np.polyfit(np.log(d.budget), np.log(d.onset_eps), 1)[0]
    chk("q4 SGD exponent", s, 0.0056, 0.002)
    d = pd.read_csv(R / "sgd_onsets.csv"); d = d[d.bracketed & d.onset.notna()]
    s = np.polyfit(np.log(d.budget), np.log(d.onset - 1), 1)[0]
    chk("family A SGD exponent", s, -0.3255, 0.002)

    print("T38/T43 CIFAR")
    d = pd.read_csv(R / "budget_flip.csv")
    p = d.pivot_table(index="epochs", columns="activation", values="test_errors", aggfunc="mean")
    chk("GELU adv at 2ep", float(p.loc[2, "relu"] - p.loc[2, "gelu"]), 750.4, 1.0)
    chk("GELU adv at 12ep", float(p.loc[12, "relu"] - p.loc[12, "gelu"]), 150.6, 1.0)

    print("T41 exclusion table")
    d = pd.read_csv(R / "criticality.csv")
    z = d[d.population == "zero_basin_constructed"]
    chk("min zero-basin grad norm", float(z.grad_norm.min()), 0.0192, 0.002)
    chk("negative lambda_min count", float((z.lambda_min < 0).sum()), 6.0, 0)

    print(f"\n{len(F)} finding(s)")
    (R / "ledger_verification.txt").write_text("\n".join(F) if F else "no findings\n")


if __name__ == "__main__":
    main()
