"""Phase 2A-PS2 results page (src/phase2a_ps2_report.py): scores.json regenerates from the committed rows with the
registered scoring code, the page regenerates from scores.json, and the page carries the required headline items."""

import json
from pathlib import Path

import pytest

from src import phase2a_ps2 as P
from src import phase2a_ps2_report as R

SCORES = P.OUT / "scores.json"


@pytest.mark.skipif(not SCORES.exists(), reason="not yet scored")
def test_scores_regenerate_and_page_regenerates():
    S = json.loads(SCORES.read_text())
    assert S == json.loads(json.dumps(P._jsonable(R.score_result())))
    assert R.RESULTS_MD.read_text() == R.render_md(S)


@pytest.mark.skipif(not SCORES.exists(), reason="not yet scored")
def test_page_headline_items():
    S = json.loads(SCORES.read_text())
    md = R.render_md(S)
    assert "Registered after 2A-PS's UNRESOLVED (validity) outcome; 2A-PS's verdict stays." in md
    head = md.split("## Verdicts")[0]
    assert "**Untraceable:" in head and "**Inactive unit at release:" in head and "over-cap check" in head
    assert "KNOWN RISK" in md and "whether slab basins lie near the fold" in md
    assert "DESCRIPTIVE ONLY: crossing by 1.25·s_F by wider-search class" in md
    assert "Misses by reason" in md and "Counted, not trained" in md and "1.19 GB" in md
    for k in ("F", "H", "E_seed", "C3", "C4", "P", "C1", "C2"):
        assert f"| {k}" in md
