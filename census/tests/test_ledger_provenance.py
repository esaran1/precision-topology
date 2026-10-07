"""Provenance rule in verify_ledger: artifacts in a results/ subdirectory (constructed pass and fail cases)."""
import pytest

import src.verify_ledger as vl

GOOD = 'OUT = RESULTS / "sub"\n\ndef main():\n    (OUT / "x.json").write_text("")\n'
NO_SUBDIR = 'def main():\n    (RESULTS / "x.json").write_text("")\n'
OTHER_FILE = 'OUT = RESULTS / "sub"\n\ndef main():\n    (OUT / "y.json").write_text("")\n'
NO_WRITE = 'OUT = RESULTS / "sub"\n\ndef main():\n    return OUT / "x.json"\n'


@pytest.mark.parametrize("src,n_bad", [(GOOD, 0), (NO_SUBDIR, 1), (OTHER_FILE, 1), (NO_WRITE, 1)])
def test_subdirectory_provenance(tmp_path, monkeypatch, src, n_bad):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "prod.py").write_text(src)
    (tmp_path / "results" / "sub").mkdir(parents=True)
    (tmp_path / "results" / "sub" / "x.json").write_text("{}")
    monkeypatch.setattr(vl, "__file__", str(tmp_path / "src" / "verify_ledger.py"))
    monkeypatch.setattr(vl, "R", tmp_path / "results")
    monkeypatch.setattr(vl, "PRODUCERS", {"sub/x.json": ("prod", "main", "full", "")})
    monkeypatch.setattr(vl, "F", [])
    monkeypatch.setattr(vl, "REC", [])
    vl.provenance_check()
    assert vl.REC[0][1] == float(n_bad)
    assert len([f for f in vl.F if f.startswith("PROVENANCE")]) == n_bad


# The f-string-prefix branch (added with Phase 3, 5a3cb7a): OUT = RESULTS / "sub"; OUT / f"runs_{arm}.jsonl".
FSTR_GOOD = 'OUT = RESULTS / "sub"\n\ndef main(arm):\n    (OUT / f"runs_{arm}.jsonl").write_text("")\n'
FSTR_NO_SUBDIR = 'def main(arm):\n    (RESULTS / f"runs_{arm}.jsonl").write_text("")\n'
FSTR_OTHER_PREFIX = 'OUT = RESULTS / "sub"\n\ndef main(arm):\n    (OUT / f"observed_{arm}.jsonl").write_text("")\n'


@pytest.mark.parametrize("src,n_bad", [(FSTR_GOOD, 0), (FSTR_NO_SUBDIR, 1), (FSTR_OTHER_PREFIX, 1)])
def test_subdirectory_fstring_provenance(tmp_path, monkeypatch, src, n_bad):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "prod.py").write_text(src)
    (tmp_path / "results" / "sub").mkdir(parents=True)
    (tmp_path / "results" / "sub" / "runs_Q.jsonl").write_text("{}")
    monkeypatch.setattr(vl, "__file__", str(tmp_path / "src" / "verify_ledger.py"))
    monkeypatch.setattr(vl, "R", tmp_path / "results")
    monkeypatch.setattr(vl, "PRODUCERS", {"sub/runs_Q.jsonl": ("prod", "main", "full", "")})
    monkeypatch.setattr(vl, "F", [])
    monkeypatch.setattr(vl, "REC", [])
    vl.provenance_check()
    assert vl.REC[0][1] == float(n_bad)


# A prefix OUTSIDE the module's own results subfolder must fail: the module writes OUT = RESULTS / "sub",
# f"runs_{arm}.jsonl", but the artifact claimed for it lives in another subfolder ("other/").
# NOTE (documents existing behaviour, not an endorsement): a TOP-LEVEL artifact ("runs_Q.jsonl") is still accepted by the
# OLDER flat-file branch (stem.startswith(prefix + "_"), predating Phase 3), which does not check where the module writes.
@pytest.mark.parametrize("artifact,n_bad", [("sub/runs_Q.jsonl", 0), ("other/runs_Q.jsonl", 1), ("runs_Q.jsonl", 0)])
def test_fstring_prefix_only_in_own_subfolder(tmp_path, monkeypatch, artifact, n_bad):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "prod.py").write_text(FSTR_GOOD)
    p = tmp_path / "results" / artifact
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    monkeypatch.setattr(vl, "__file__", str(tmp_path / "src" / "verify_ledger.py"))
    monkeypatch.setattr(vl, "R", tmp_path / "results")
    monkeypatch.setattr(vl, "PRODUCERS", {artifact: ("prod", "main", "full", "")})
    monkeypatch.setattr(vl, "F", [])
    monkeypatch.setattr(vl, "REC", [])
    vl.provenance_check()
    assert vl.REC[0][1] == float(n_bad)
