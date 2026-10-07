"""The Phase 3 reproducibility check (POST HOC): its comparison rule and, if present, its committed output."""
import json
from pathlib import Path

from src import phase3_repro_check as RC

ROOT = Path(__file__).resolve().parents[1]


def test_strip_drops_timing_only():
    a = {"seed": 1, "secs_run": 3.0, "forecast": {"t_fc": 10, "secs": 2.0, "max_index_read": {"v": 9}}, "rss": 5}
    assert RC.strip(a) == {"seed": 1, "forecast": {"t_fc": 10, "max_index_read": {"v": 9}}}
    b = json.loads(json.dumps(a))
    b["secs_run"] = 99.0
    b["forecast"]["secs"] = 0.1
    assert RC.strip(a) == RC.strip(b)
    b["forecast"]["t_fc"] = 11
    assert RC.strip(a) != RC.strip(b) and RC.differing_keys(RC.strip(a), RC.strip(b)) == ["forecast"]


def test_seeds_are_pre_stall_q_seeds():
    assert RC.SEEDS == (7_430_009, 7_430_005, 7_430_012) and all(s < 7_430_013 for s in RC.SEEDS)


def test_committed_output_if_present():
    p = ROOT / "results" / "phase3" / "repro_check.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    assert sorted(d["seeds"]) == sorted(str(s) for s in RC.SEEDS)
    assert d["all_identical"] is all(v["identical"] for v in d["seeds"].values())
