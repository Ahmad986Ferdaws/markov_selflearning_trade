"""A labeled position followed by a sentence period is the most ordinary prose
form ("Set exposure to 0.8.") and must parse. The lookahead added to reject
`0.5oops`, `50%` and `0.5.3` also rejected any `.` after the number, so these
replies were silently converted into holds while the provenance histogram
blamed the model ("unparseable"). Every previously-held vector must still hold.
"""

import pytest

from app.services.agent_policy import _parse_position


@pytest.mark.parametrize("raw,expected", [
    ("Set exposure to 0.8.", 0.8),
    ("position: 0.7.", 0.7),
    ("Position: 0.7. Bear regime, staying cautious.", 0.7),
    ("I would set the target at 0.5. Nothing more.", 0.5),
    ("Allocation of 1.", 1.0),
    ("exposure to 0.8", 0.8),                      # no period, as before
])
def test_labeled_value_before_sentence_period_parses(raw, expected):
    assert _parse_position(raw, 0.3) == (expected, "salvaged_labeled")


@pytest.mark.parametrize("raw", [
    "position: 0.5.3",          # a second number glued on
    "position: 0.8.5",
    "position: 0..5",           # repeated dots: no backtracking to "0" (Codex review)
    "position: 0.5..3",
    "position: 0.5..",
    "position: 1..",            # "1." must not be taken as the number (review)
    "position: 0..",
    "position: 0.5oops",
    "position: 50%",
    "position: 0.5 + 0.2",
    "position: .5/.2",
])
def test_previously_rejected_forms_still_hold(raw):
    assert _parse_position(raw, 0.3) == (0.3, "unparseable_hold")


def test_json_reply_ending_in_a_period_still_takes_the_strict_path():
    assert _parse_position('{"position": 0.6}.', 0.3) == (0.6, "strict_json")


@pytest.mark.parametrize("raw", [
    "Bull regime has weight 0.6. Position: 0.4.",       # keyed stray number first
    "The bull-regime weight at 0.9. My position: 0.3.",
    "exposure to 0.8; target 0.2.",                     # two labels, two values
    "weight 0.6 and my position: 0.4.",                 # held before too
])
def test_two_different_labeled_values_are_ambiguous_and_hold(raw):
    assert _parse_position(raw, 0.3) == (0.3, "unparseable_hold")


def test_repeated_identical_labeled_value_still_parses():
    assert _parse_position("Position: 0.4. Yes, position 0.4.", 0.3) == (0.4, "salvaged_labeled")


def test_unkeyed_stray_numbers_before_the_label_still_parse():
    raw = "z-score 2.1, year 2026, window 200. position: 0.5."
    assert _parse_position(raw, 0.3) == (0.5, "salvaged_labeled")
