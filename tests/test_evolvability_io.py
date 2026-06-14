"""
tests/test_evolvability_io.py — unit tests for ecology.evolvability.io.
"""
import json
import pathlib
import pytest

from ecology.evolvability.io import (
    new_run_dir,
    write_jsonl,
    read_jsonl,
    append_jsonl,
    write_csv,
    write_json,
    config_hash,
    git_commit,
)


# ---------------------------------------------------------------------------
# new_run_dir
# ---------------------------------------------------------------------------

class TestNewRunDir:
    def test_creates_run_dir_and_raw(self, tmp_path):
        d = new_run_dir(tmp_path, "myslug", run_id="r1")
        assert d == tmp_path / "myslug" / "r1"
        assert d.is_dir()
        assert (d / "raw").is_dir()

    def test_no_overwrite_raises(self, tmp_path):
        new_run_dir(tmp_path, "myslug", run_id="r1")
        with pytest.raises(FileExistsError):
            new_run_dir(tmp_path, "myslug", run_id="r1")

    def test_different_run_ids_are_independent(self, tmp_path):
        d1 = new_run_dir(tmp_path, "myslug", run_id="r1")
        d2 = new_run_dir(tmp_path, "myslug", run_id="r2")
        assert d1 != d2
        assert d1.is_dir()
        assert d2.is_dir()

    def test_auto_timestamp_if_no_run_id(self, tmp_path):
        d = new_run_dir(tmp_path, "myslug")
        assert d.parent == tmp_path / "myslug"
        assert d.is_dir()


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

class TestJsonl:
    def test_write_read_roundtrip(self, tmp_path):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        p = tmp_path / "out.jsonl"
        write_jsonl(p, rows)
        back = read_jsonl(p)
        assert back == rows

    def test_append_jsonl_order(self, tmp_path):
        p = tmp_path / "append.jsonl"
        rows = [{"n": i} for i in range(5)]
        for r in rows:
            append_jsonl(p, r)
        back = read_jsonl(p)
        assert back == rows

    def test_empty_write(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        write_jsonl(p, [])
        assert read_jsonl(p) == []


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def test_header_and_rows(self, tmp_path):
        p = tmp_path / "data.csv"
        rows = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        write_csv(p, rows)
        lines = p.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "x,y"          # header (sorted keys)
        assert len(lines) == 3            # header + 2 rows

    def test_empty_with_fieldnames(self, tmp_path):
        p = tmp_path / "empty.csv"
        write_csv(p, [], fieldnames=["a", "b"])
        lines = p.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "a,b"
        assert len(lines) == 1

    def test_empty_without_fieldnames(self, tmp_path):
        p = tmp_path / "empty2.csv"
        write_csv(p, [])
        assert p.read_text(encoding="utf-8") == ""

    def test_extra_action_ignore(self, tmp_path):
        p = tmp_path / "extra.csv"
        rows = [{"a": 1, "b": 2, "c": 99}]
        write_csv(p, rows, fieldnames=["a", "b"])
        lines = p.read_text(encoding="utf-8").splitlines()
        assert "c" not in lines[0]


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------

class TestWriteJson:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "obj.json"
        obj = {"key": "value", "n": 42}
        write_json(p, obj)
        with open(p, encoding="utf-8") as fh:
            back = json.load(fh)
        assert back == obj


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_deterministic(self):
        d = {"slug": "x", "n": 1}
        assert config_hash(d) == config_hash(d)

    def test_changes_on_modification(self):
        d1 = {"slug": "x"}
        d2 = {"slug": "y"}
        assert config_hash(d1) != config_hash(d2)

    def test_length(self):
        h = config_hash({"a": 1})
        assert len(h) == 64   # sha256 hex digest


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------

class TestGitCommit:
    def test_returns_none_or_40_hex(self):
        result = git_commit()
        if result is None:
            # Not a git repo or git not installed — acceptable
            return
        assert len(result) == 40
        assert all(c in "0123456789abcdef" for c in result)

    def test_invalid_path_returns_none(self, tmp_path):
        # tmp_path is not a git repo
        result = git_commit(repo_root=tmp_path)
        assert result is None
