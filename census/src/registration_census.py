"""Registration census: every registered prediction in the repository, one row each.

Input  results/registration_census_enumeration.csv  hand-curated enumeration of every
       registration document (id, block, registration file and commit, verdict as
       previously scored, where it was scored, whether it was in the old "64").
Output results/registration_census.csv  HEADLINE convention (as in the 2026-09-23 census, 151): one row per
       registered prediction, counted once across the values of a at which it was scored; a
       prediction with different verdicts at different a is PARTIAL, assigned post hoc.
       results/registration_census_by_unit.csv  appendix convention: one row per registered unit
       (per a where a registration scores each a separately), corrections below applied,
       each recorded in `changed_from` / `verdict_status`, census-assigned verdicts flagged.

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
OUT_UNITS = RESULTS / "registration_census_by_unit.csv"
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
# second round (2026-09-24): v4 registrations scored since the first census; one row per registered unit
# (per a where the registration scores each a separately)
POST_HOC_V4 = ["B3-U-range"]
OTHER = RESULTS / "registration_census_v4_gates_and_reported.csv"   # validity gates + no-criterion items


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
    for i in POST_HOC_V4:
        m = d.id == i
        assert m.sum() == 1, i
        d.loc[m, "verdict_status"] = ("SCORED IN CENSUS 2026-09-24 (post hoc scoring against the "
                                      "registered criterion; judgement call, see note)")
    d["non_directional"] = d.note.fillna("").str.startswith("NON-DIRECTIONAL")
    d["scoring"] = np.where(d.id.isin(POST_HOC + POST_HOC_V4), "post hoc (census)", "registered rule")
    assert not d.id.duplicated().any()
    assert d.verdict.isin(VERDICTS).all()
    d["prediction_id"] = d.id.str.split("@").str[0]
    d.to_csv(OUT_UNITS, index=False)
    p = collapse(d)
    p.to_csv(OUT, index=False)
    rows = []
    other = pd.read_csv(OTHER)
    gates = other[other.kind == "validity gate"]
    assert not set(other.id) & set(d.id)
    for label, t in (("", p), ("by registered unit: ", d)):
        v4 = t[t.census_round == "2026-09-24"]
        for scope, g in (("scored by registered rules", t[t.scoring == "registered rule"]),
                         ("assigned post hoc in the census", t[t.scoring != "registered rule"]),
                         ("all", t),
                         ("since 2026-09-23: scored by registered rules", v4[v4.scoring == "registered rule"]),
                         ("since 2026-09-23: assigned post hoc", v4[v4.scoring != "registered rule"])):
            c = g.verdict.value_counts().reindex(VERDICTS).fillna(0).astype(int)
            rows.append({"scope": label + scope, "n": len(g), **c.to_dict()})
    c = gates.verdict.value_counts().reindex(VERDICTS).fillna(0).astype(int)
    rows.append({"scope": "since 2026-09-23: validity gates (not predictions; not in the headline)", "n": len(gates),
                 **c.to_dict()})
    pd.DataFrame(rows).to_csv(TALLY, index=False)
    return p


def collapse(d: pd.DataFrame) -> pd.DataFrame:
    """One row per registered prediction (the 2026-09-23 convention): per-a rows of the same prediction are merged.
    Same verdict at every a -> that verdict, scored as its rows were. Different verdicts -> PARTIAL, assigned post
    hoc in the census (as S-3 was)."""
    out = []
    for pid, g in d.groupby("prediction_id", sort=False):
        if len(g) == 1 and g.id.iloc[0] == pid:
            out.append(g.iloc[0].to_dict())
            continue
        r = g.iloc[0].to_dict()
        units = ", ".join(f"{i.split('@')[1]}: {v}" for i, v in zip(g.id, g.verdict))
        r["id"] = pid
        r["original_verdict_text"] = " || ".join(g.original_verdict_text.astype(str))
        if g.verdict.nunique() == 1:
            r["verdict"] = g.verdict.iloc[0]
            r["scoring"] = "registered rule" if (g.scoring == "registered rule").all() else "post hoc (census)"
            r["verdict_status"] = f"merged across a ({units}); same verdict at every a"
        else:
            r["verdict"] = "PARTIAL"
            r["scoring"] = "post hoc (census)"
            r["verdict_status"] = (f"SCORED IN CENSUS 2026-09-24 (merged across a: {units}; different verdicts at "
                                   f"different a, so PARTIAL, assigned post hoc as S-3 was)")
        out.append(r)
    p = pd.DataFrame(out)
    assert not p.id.duplicated().any() and p.verdict.isin(VERDICTS).all()
    return p


def check() -> pd.DataFrame:
    d = build()
    tally = lambda x: x.verdict.value_counts().reindex(VERDICTS).fillna(0).astype(int).to_dict()
    print("all", len(d), tally(d))
    print("existing 64", tally(d[d.counted_in_existing_64 == "yes"]))
    print(d.groupby("block").verdict.value_counts().unstack(fill_value=0).to_string())
    return d


if __name__ == "__main__":
    check()
