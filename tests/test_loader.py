"""Tests for the loader — a wrong question/label join corrupts every result,
so the failure contracts (mismatch, bad JSON) must be enforced loudly."""

import pytest

from da_verify.tasks.loader import load_tasks


def _write(tmp_path, q_lines, l_lines):
    q = tmp_path / "q.jsonl"
    lab = tmp_path / "l.jsonl"
    q.write_text("\n".join(q_lines) + "\n", encoding="utf-8")
    lab.write_text("\n".join(l_lines) + "\n", encoding="utf-8")
    return q, lab


def test_happy_join(tmp_path):
    q, lab = _write(
        tmp_path,
        ['{"id": 1, "question": "q", "concepts": ["A"], "constraints": "c", '
         '"format": "@x[v]", "file_name": "t.csv", "level": "easy"}'],
        ['{"id": 1, "common_answers": [["x", "1.0"]]}'],
    )
    tasks = load_tasks(q, lab)
    assert len(tasks) == 1
    assert tasks[0].id == 1
    assert tasks[0].gold[0].name == "x" and tasks[0].gold[0].value == "1.0"
    assert tasks[0].n_subanswers == 1


def test_id_mismatch_raises(tmp_path):
    q, lab = _write(
        tmp_path,
        ['{"id": 1, "question": "q", "concepts": [], "constraints": "", '
         '"format": "", "file_name": "t.csv", "level": "easy"}'],
        ['{"id": 2, "common_answers": [["x", "1"]]}'],
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_tasks(q, lab)


def test_bad_json_raises_with_line(tmp_path):
    q, lab = _write(
        tmp_path,
        ['{"id": 1, "question": "q", "concepts": [], "constraints": "", '
         '"format": "", "file_name": "t.csv", "level": "easy"}',
         "{ not valid json"],
        ['{"id": 1, "common_answers": [["x", "1"]]}'],
    )
    with pytest.raises(ValueError, match="Bad JSON"):
        load_tasks(q, lab)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "nope.jsonl", tmp_path / "nope2.jsonl")
