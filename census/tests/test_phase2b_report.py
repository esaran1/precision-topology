"""Phase 2B results page (src/phase2b_report.py; not part of the registration): the page regenerates from scores.json,
scores.json regenerates from the committed runs with the registered scoring code, and the page claims no endpoint
advantage."""

import json

import pytest

from src import phase2b as P
from src import phase2b_report as R

SCORES = P.OUT / "scores.json"


@pytest.mark.skipif(not SCORES.exists(), reason="not yet scored")
def test_results_page_and_scores_regenerate():
    S = json.loads(SCORES.read_text())
    assert S == json.loads(json.dumps(P._jsonable(R.score_result())))
    md = R.render_md(S)
    assert R.RESULTS_MD.read_text() == md
    assert "no endpoint advantage is claimed" in md
    assert S["no_criterion_is_a_forecast"] is True
    for a, vs in S["verdicts"].items():
        for k, v in vs.items():
            assert f"| {R.CRIT_NAME[k]} | {v} |" in md
