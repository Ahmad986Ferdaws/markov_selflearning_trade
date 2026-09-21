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
    "position: 0.5oops",
    "position: 50%",
    "position: 0.5 + 0.2",
    "position: .5/.2",
])
def test_previously_rejected_forms_still_hold(raw):
    assert _parse_position(raw, 0.3) == (0.3, "unparseable_hold")


def test_json_reply_ending_in_a_period_still_takes_the_strict_path():
    assert _parse_position('{"position": 0.6}.', 0.3) == (0.6, "strict_json")
