"""The ledger's exception for anonymised registered files in the review supplement (author's decision 2026-11-14):
constructed pass and fail cases, and the unchanged path when no record exists."""
import hashlib
import json
import types

import pytest

from src import verify_ledger as VL


def _mod(tmp_path, files):
    root = tmp_path / "repo"
    out = root / "results" / "p3"
    out.mkdir(parents=True)
    lines = []
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        lines.append(f"{hashlib.sha256(text.encode()).hexdigest()}  {rel}")
    (out / "registration.sha256").write_text("\n".join(lines) + "\n")
    (out / "registration_stamp.txt").write_text("stamp\n")
    (out / "registration_stamp.txt.ots").write_bytes(b"ots")
    calls = []
    m = types.SimpleNamespace(OUT=out, ROOT=root, _sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
                              _assert_committed=lambda p: calls.append(p.name),
                              assert_registration=lambda: calls.append("registered function"))
    return m, root, calls


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.fixture
def results(tmp_path, monkeypatch):
    r = tmp_path / "results_dir"
    r.mkdir()
    monkeypatch.setattr(VL, "R", r)
    return r


def test_no_record_calls_the_registered_function_unchanged(tmp_path, results):
    m, _, calls = _mod(tmp_path, {"a.py": "x\n"})
    VL.registration_with_anonymised(m)
    assert calls == ["registered function"]


def _record(results, rel, registered, shipped):
    (results / VL.ANON_REGISTERED).write_text(json.dumps(
        {"files": [{"path": rel, "registered_sha256": registered, "shipped_sha256": shipped}]}))


def test_pass_when_the_anonymised_file_matches_both_recorded_hashes(tmp_path, results):
    m, root, calls = _mod(tmp_path, {"a.py": "path = '/home/someone'\n", "b.py": "y\n"})
    orig = "path = '/home/someone'\n"
    (root / "a.py").write_text("path = '<repo>/'\n")
    _record(results, "a.py", _sha(orig), _sha("path = '<repo>/'\n"))
    VL.registration_with_anonymised(m)
    assert calls == ["registration.sha256", "registration_stamp.txt"]


def test_fail_when_the_shipped_bytes_differ_from_the_record(tmp_path, results):
    m, root, _ = _mod(tmp_path, {"a.py": "orig\n"})
    (root / "a.py").write_text("something else\n")
    _record(results, "a.py", _sha("orig\n"), _sha("anonymised\n"))
    with pytest.raises(AssertionError, match="does not match its record"):
        VL.registration_with_anonymised(m)


def test_fail_when_the_recorded_registered_hash_is_not_the_manifest_hash(tmp_path, results):
    m, root, _ = _mod(tmp_path, {"a.py": "orig\n"})
    (root / "a.py").write_text("anonymised\n")
    _record(results, "a.py", _sha("some other original\n"), _sha("anonymised\n"))
    with pytest.raises(AssertionError, match="does not match its record"):
        VL.registration_with_anonymised(m)


def test_fail_when_an_unlisted_registered_file_changed(tmp_path, results):
    m, root, _ = _mod(tmp_path, {"a.py": "orig\n", "b.py": "y\n"})
    (root / "a.py").write_text("anonymised\n")
    (root / "b.py").write_text("tampered\n")
    _record(results, "a.py", _sha("orig\n"), _sha("anonymised\n"))
    with pytest.raises(AssertionError, match="registration hash mismatch: b.py"):
        VL.registration_with_anonymised(m)


def test_fail_without_the_ots_proof(tmp_path, results):
    m, root, _ = _mod(tmp_path, {"a.py": "orig\n"})
    (m.OUT / "registration_stamp.txt.ots").unlink()
    (root / "a.py").write_text("anonymised\n")
    _record(results, "a.py", _sha("orig\n"), _sha("anonymised\n"))
    with pytest.raises(AssertionError, match="no OpenTimestamps proof"):
        VL.registration_with_anonymised(m)
