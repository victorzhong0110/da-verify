"""Tests for score_response — especially the lenient/format-forgiving metric,
which must credit ONLY the unambiguous single-field name-mismatch and never
leak into the strict headline `correct`."""

from da_verify.eval.scoring import score_response
from da_verify.tasks.loader import GoldAnswer, Task


def _task(gold):
    return Task(id=1, question="q", concepts=(), constraints="", answer_format="",
                file_name="t.csv", level="easy", gold=gold)


def test_strict_correct_is_also_lenient():
    sc = score_response(_task((GoldAnswer("outlier_count", "20"),)), "@outlier_count[20]")
    assert sc.correct and sc.lenient_correct and sc.format_ok


def test_single_field_right_value_wrong_name_is_lenient_only():
    # the id=132 shape: value 20 right, @name wrong
    sc = score_response(_task((GoldAnswer("outlier_count", "20"),)), "@answer_count[20]")
    assert not sc.correct          # headline: wrong (didn't follow format)
    assert sc.lenient_correct      # format-forgiving: credited
    assert not sc.format_ok


def test_single_field_wrong_name_wrong_value_is_neither():
    sc = score_response(_task((GoldAnswer("outlier_count", "20"),)), "@answer_count[99]")
    assert not sc.correct and not sc.lenient_correct


def test_multifield_wrong_names_not_leniently_credited():
    # 2 fields -> positional matching would be ambiguous, so lenient must NOT fire
    t = _task((GoldAnswer("a", "1"), GoldAnswer("b", "2")))
    sc = score_response(t, "@x[1] @y[2]")  # right values, wrong names, multi-field
    assert not sc.correct and not sc.lenient_correct
