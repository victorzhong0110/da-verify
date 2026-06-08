"""Tests for the verifier — one block per behaviour, named for what it proves.

These tests ARE the spec. If a rule in verifier.py changes, a test here should
have to change too — that is how we keep "what counts as correct" honest.
"""

import pytest

from da_verify.tasks.verifier import (
    CompareOptions,
    compare_value,
    extract_answers,
    infer_type,
    official_extract_format,
    official_is_equal,
    verify_response,
)

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extracts_single_answer():
    assert extract_answers("The mean is @mean_fare[34.65].") == {"mean_fare": "34.65"}


def test_extracts_multiple_answers():
    got = extract_answers("@a[1] then @b[hello world]")
    assert got == {"a": "1", "b": "hello world"}


def test_balanced_scanner_handles_bracketed_list_value():
    # Our scanner keeps the inner brackets; the official non-greedy regex would
    # truncate at the first ']'. This is the concrete improvement.
    assert extract_answers("@outliers[[1, 2, 3]]") == {"outliers": "[1, 2, 3]"}
    names, vals = official_extract_format("@outliers[[1, 2, 3]]")
    assert vals == ["[1, 2, 3"]  # documents the official truncation bug


def test_unbalanced_field_is_skipped_not_crashing():
    # A stray '[' that never closes (the id=273 malformed-gold shape).
    assert extract_answers("@outlier_list[[]") == {}


def test_nested_tag_inside_value_not_double_extracted():
    # Regression (CRITICAL-1): a @tag inside another tag's value must NOT become
    # its own field. Only the outer 'result' is captured; 'detail' is not a key.
    got = extract_answers("@result[see @detail[42] here]")
    assert got == {"result": "see @detail[42] here"}
    assert "detail" not in got


def test_duplicate_name_last_wins():
    assert extract_answers("@x[1] @x[2]") == {"x": "2"}


def test_missing_field_returns_empty():
    assert extract_answers("no markers here") == {}
    assert extract_answers(None) == {}


# ---------------------------------------------------------------------------
# Numeric comparison
# ---------------------------------------------------------------------------


def test_numeric_exact_and_within_tolerance():
    assert compare_value("34.65", "34.65")
    assert compare_value("34.650000001", "34.65")  # within 1e-6? no -> rel test below
    assert not compare_value("34.70", "34.65")  # 0.05 diff > 1e-6


def test_numeric_relative_tolerance_knob():
    opts = CompareOptions(num_rel_tol=1e-2)  # robustness mode
    # 0.212 is ~0.95% off 0.21 -> within the 1% band; 0.214 (~1.9%) is not.
    assert compare_value("0.212", "0.21", opts)
    assert not compare_value("0.214", "0.21", opts)  # still rejected at 1%
    assert not compare_value("0.212", "0.21")  # default abs-only mode rejects


def test_numeric_normalises_currency_percent_commas():
    assert compare_value("$1,234.00", "1234.0")
    assert compare_value("42%", "42")  # we strip %, do NOT divide by 100


def test_numeric_missing_prediction_is_wrong():
    assert not compare_value(None, "34.65")
    assert not compare_value("not a number", "34.65")


def test_negative_and_int_floats():
    assert compare_value("-0.50", "-0.5")
    assert compare_value("100", "100.0")


# ---------------------------------------------------------------------------
# Categorical comparison
# ---------------------------------------------------------------------------


def test_categorical_casefold_and_whitespace():
    assert compare_value("Linear", "linear")
    assert compare_value("  not   normal ", "not normal")


def test_categorical_semantic_variant_is_correctly_marked_wrong():
    # We do NOT pretend to solve synonymy; this is a documented limitation.
    assert not compare_value("not normal", "False")
    assert not compare_value("no", "False")


# ---------------------------------------------------------------------------
# List comparison
# ---------------------------------------------------------------------------


def test_list_inferred_from_comma_or_brackets():
    assert infer_type("1, 2018, 88.32") == "list"
    assert infer_type("[a, b, c]") == "list"
    assert infer_type("34.65") == "numeric"
    assert infer_type("linear") == "categorical"


def test_list_ordered_default_respects_order():
    assert compare_value("1, 2018, 88.32", "1, 2018, 88.32")
    # order matters for tuples like (month, year, price)
    assert not compare_value("2018, 1, 88.32", "1, 2018, 88.32")


def test_list_set_mode_ignores_order():
    opts = CompareOptions(list_mode="set")
    assert compare_value("c, a, b", "a, b, c", opts)


def test_list_length_mismatch_is_wrong():
    assert not compare_value("1, 2", "1, 2, 3")


def test_list_elements_use_numeric_tolerance():
    assert compare_value("[1.0, 2.0]", "[1, 2]")


def test_short_comma_list_not_misread_as_thousands():
    # Regression (CRITICAL-2): "2,3" is the list [2,3], NOT the number 23.
    assert infer_type("2,3") == "list"
    assert not compare_value("23", "2,3")        # the false-positive we fixed
    assert compare_value("2, 3", "2,3")          # same list, spacing-insensitive


def test_real_thousands_separator_still_numeric():
    # Valid 3-digit grouping IS a number (model may output thousands separators).
    assert infer_type("1,234.5") == "numeric"
    assert compare_value("1,234.5", "1234.5")
    assert compare_value("$1,234", "1234")


def test_dict_literal_gold_is_categorical_not_list():
    # Regression (CRITICAL-3): {...} gold must not be split on its inner commas.
    g = "{'month_1': 7.17, 'month_2': 6.53}"
    assert infer_type(g) == "categorical"
    assert compare_value(g, g)                    # identical dict strings match
    assert not compare_value("7.17, 6.53", g)     # not naively comma-split


# ---------------------------------------------------------------------------
# Multi-part task verification
# ---------------------------------------------------------------------------


def test_multipart_all_fields_must_pass():
    gold = [("corr", "0.21"), ("relationship_type", "linear")]
    ok = verify_response(5, "@corr[0.21] @relationship_type[Linear]", gold)
    assert ok.all_correct
    assert ok.per_field == {"corr": True, "relationship_type": True}


def test_multipart_one_wrong_fails_whole():
    gold = [("corr", "0.21"), ("relationship_type", "linear")]
    bad = verify_response(5, "@corr[0.99] @relationship_type[linear]", gold)
    assert not bad.all_correct
    assert bad.n_correct_fields == 1


def test_multipart_missing_field_recorded():
    gold = [("a", "1"), ("b", "2")]
    r = verify_response(1, "@a[1]", gold)
    assert not r.all_correct
    assert r.missing_fields == ("b",)


# ---------------------------------------------------------------------------
# Faithfulness to the official benchmark (comparability)
# ---------------------------------------------------------------------------


def test_agrees_with_official_on_clean_numerics():
    for p, g in [("34.65", "34.65"), ("0.21", "0.21"), ("-0.5", "-0.5")]:
        assert compare_value(p, g) == official_is_equal(p, g) is True


def test_documents_divergence_from_official_on_case():
    # Official rejects case differences; we accept them by design.
    assert not official_is_equal("Linear", "linear")
    assert compare_value("Linear", "linear")
