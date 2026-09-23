"""Registration census: every registered prediction in the repository, one row each.

Input  results/registration_census_enumeration.csv  hand-curated enumeration of every
       registration document (id, block, registration file and commit, verdict as
       previously scored, where it was scored, whether it was in the old "64").
Output results/registration_census.csv  the enumeration with the corrections below
       applied, each recorded in `changed_from` / `verdict_status`, and the verdicts
       first assigned in the census flagged as post hoc.

    python -m src.registration_census
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
SRC = RESULTS / "registration_census_enumeration.csv"
OUT = RESULTS / "registration_census.csv"
TALLY = RESULTS / "registration_tally.csv"
VERDICTS = ("PASS", "FAIL", "PARTIAL", "UNRESOLVED")

# (id, corrected verdict, previously scored verdict, reason)
CORRECTIONS = [
    ("E-2", "FAIL", "PASS",
     "33/37 = 0.892 kept < registered >= 9/10; lost in 4/37 > 1 of 10 "
     "(blockE_redesign.md:63,72-74); wi_e2_rescore.csv"),
    ("P-SGD", "PARTIAL", "PASS",
     "sgd_law_results.md:21-37 scores it OUTSIDE the registered band, DIRECTIONAL-ONLY by the "
     "rule fixed in advance; the summary row PASS recorded only that the arm ran"),
    ("P-null", "UNRESOLVED", "PASS",
     "geometric_transfer.md:1-41: the test did not execute and no ratio was measured, so the "
     "registered null was not excluded"),
]
# verdicts first assigned in the census (post hoc scoring against the registered criterion)
POST_HOC = ["B-solve", "B-CV", "collapse-link", "thr-P5", "B-spin", "loc-2", "rc-1d", "S-3",
            "gelu-up", "amp-2a-match", "E-cold_low", "metric-1a", "metric-2"]


def build() -> pd.DataFrame:
    d = pd.read_csv(SRC)
    d["changed_from"] = ""
    d["verdict_status"] = "as previously scored"
    for i, v, old, why in CORRECTIONS:
        m = d.id == i
        assert m.sum() == 1, i
        d.loc[m, "changed_from"] = old
        d.loc[m, "verdict"] = v
        d.loc[m, "verdict_status"] = "CORRECTED 2026-09-23: " + why
    for i in POST_HOC:
        m = d.id == i
        assert m.sum() == 1, i
        d.loc[m, "verdict_status"] = ("SCORED IN CENSUS 2026-09-23 (post hoc scoring against the "
                                      "registered criterion; judgement call, see note)")
    d["non_directional"] = d.note.fillna("").str.startswith("NON-DIRECTIONAL")
    d["scoring"] = np.where(d.id.isin(POST_HOC), "post hoc (census)", "registered rule")
    assert not d.id.duplicated().any()
    assert d.verdict.isin(VERDICTS).all()
    d.to_csv(OUT, index=False)
    rows = []
    for scope, g in (("scored by registered rules", d[d.scoring == "registered rule"]),
                     ("assigned post hoc in the census", d[d.scoring != "registered rule"]),
                     ("all", d)):
        c = g.verdict.value_counts().reindex(VERDICTS).fillna(0).astype(int)
        rows.append({"scope": scope, "n": len(g), **c.to_dict()})
    pd.DataFrame(rows).to_csv(TALLY, index=False)
    return d


def check() -> pd.DataFrame:
    d = build()
    tally = lambda x: x.verdict.value_counts().reindex(VERDICTS).fillna(0).astype(int).to_dict()
    print("all", len(d), tally(d))
    print("existing 64", tally(d[d.counted_in_existing_64 == "yes"]))
    print(d.groupby("block").verdict.value_counts().unstack(fill_value=0).to_string())
    return d


if __name__ == "__main__":
    check()
