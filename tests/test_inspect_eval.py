"""Inspect-adapter unit tests that do not need Docker or DAEval tables."""

import pytest

pytest.importorskip("inspect_ai")

from da_verify.inspect_eval import da_verify, score_completion


def _meta(*, gold):
    return {
        "task_id": 132,
        "question": "q",
        "concepts": [],
        "constraints": "",
        "answer_format": "@outlier_count[value]",
        "file_name": "t.csv",
        "level": "easy",
        "gold": gold,
    }


def test_strict_correct_is_scored_correct():
    result = score_completion(_meta(gold=[["outlier_count", "20"]]), "@outlier_count[20]")
    assert result["correct"] is True
    assert result["format_ok"] is True


def test_wrong_field_name_is_not_headline_correct():
    result = score_completion(_meta(gold=[["outlier_count", "20"]]), "@answer_count[20]")
    assert result["correct"] is False
    assert result["lenient_correct"] is True


def test_rejects_unknown_condition():
    with pytest.raises(ValueError, match="condition"):
        da_verify(condition="c9")  # type: ignore[arg-type]


def test_load_samples_requires_fetched_data():
    from da_verify.inspect_eval import load_inspect_samples
    from da_verify.tasks.loader import QUESTIONS_PATH

    if QUESTIONS_PATH.exists():
        pytest.skip("DAEval already fetched")
    with pytest.raises(FileNotFoundError):
        load_inspect_samples(n=1)
