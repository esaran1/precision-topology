"""Harsh-review Phase A, item A4: the registration census sorted by relevance (POST HOC classification, 2026-09-25).

Rule (fixed by block topic, before the counts were tabulated; ties resolved toward CENTRAL so that no failure is
hidden as peripheral): a registration block is CENTRAL if its predictions concern
  threshold    -- the conditional threshold: its value, what it predicts about training crossings, and the residual;
  scaling      -- the scaling reduction (κ, K, A*, c₁, the scaling limit, the cross-family test);
  prospective  -- the prospective held-out predictions (Block G calibration, Block 3, own-seed);
  width2       -- the width-2 result.
Everything else is PERIPHERAL: budget laws, sharpness/barriers, trapping, optimiser equivalence, and the exploratory
probes of the early census.  The classification is by block, never by outcome.

    python -m src.census_relevance
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

BLOCKS = {
    # central
    "B": ("central", "threshold"), "Block 4 (fixed scale)": ("central", "threshold"),
    "Block 4 horizon extension": ("central", "threshold"), "Block 5 (retention)": ("central", "threshold"),
    "Block 4b (residual mechanism)": ("central", "threshold"), "F": ("central", "threshold"),
    "H": ("central", "threshold"), "R-collapse 09-12": ("central", "threshold"),
    "lag test": ("central", "threshold"), "lag test 2 (deconfounded)": ("central", "threshold"),
    "metric artifact 09-12": ("central", "threshold"), "own-seed thresholds": ("central", "threshold"),
    "sample size": ("central", "threshold"), "phase1": ("central", "threshold"), "phase2b": ("central", "threshold"),
    "phase2b across a": ("central", "threshold"),
    "kappa": ("central", "scaling"), "scaling limit": ("central", "scaling"), "c1 first order": ("central", "scaling"),
    "cross-family": ("central", "scaling"), "K": ("central", "scaling"),
    "G": ("central", "prospective"), "Block 3 (held-out windows)": ("central", "prospective"),
    "prospective own-seed": ("central", "prospective"),
    "scale limits (width 2)": ("central", "width2"),
    "SGD own thresholds (Track 4)": ("central", "threshold"),
    "asymmetric windows (Track 2)": ("central", "width2"),
    # peripheral
    "Arrhenius 08-27": ("peripheral", "budget law / barriers"), "MNIST budget law 09-11": ("peripheral", "budget law"),
    "collapse 09-12": ("peripheral", "budget law"), "nu": ("peripheral", "budget law"),
    "S2 (withdrawn)": ("peripheral", "budget law"), "A5d k=1": ("peripheral", "5-D probe"),
    "A5d k=10": ("peripheral", "5-D probe"), "C": ("peripheral", "optimiser equivalence"),
    "third optimizer 09-14": ("peripheral", "optimiser equivalence"), "E": ("peripheral", "trapping"),
    "E-stall": ("peripheral", "trapping"), "early census 08-23": ("peripheral", "exploratory probe"),
    "early census 08-23 (basin)": ("peripheral", "barriers / basin"),
    "early census 08-23 (width)": ("peripheral", "exploratory probe"),
    "corrugation 08-06": ("peripheral", "exploratory probe"),
    "corrugation readings 08-22": ("peripheral", "exploratory probe"),
    "interleaved 08-05": ("peripheral", "exploratory probe"), "localization 08-22": ("peripheral", "exploratory probe"),
    "search 08-22": ("peripheral", "exploratory probe"), "threshold 08-22": ("peripheral", "exploratory probe"),
    "winding 08-22": ("peripheral", "exploratory probe"), "precision": ("peripheral", "retracted"),
}


def build():
    d = pd.read_csv(RESULTS / "registration_census.csv")
    missing = set(d.block) - set(BLOCKS)
    if missing:
        raise RuntimeError(f"unclassified blocks: {sorted(missing)}")
    d["relevance"] = d.block.map(lambda b: BLOCKS[b][0])
    d["claim"] = d.block.map(lambda b: BLOCKS[b][1])
    d[["id", "block", "scoring", "verdict", "relevance", "claim"]].to_csv(RESULTS / "census_relevance.csv", index=False)
    t = d.pivot_table(index=["scoring", "relevance"], columns="verdict", values="id", aggfunc="size", fill_value=0)
    t.to_csv(RESULTS / "census_relevance_counts.csv")
    return d, t


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d, t = build()
    print(t.to_string())
    f = d[(d.verdict == "FAIL") & (d.relevance == "central")]
    print(f.groupby(["scoring", "claim"]).size().to_string())
    print(", ".join(f.id))
