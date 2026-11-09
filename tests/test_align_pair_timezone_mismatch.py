"""kalman.data.align_pair names a naive-vs-aware timezone mismatch.

pandas intersects a tz-aware DatetimeIndex with a naive one to an empty
index, so two legs on IDENTICAL calendars used to fail as 'pair overlap too
short: 0 < 250', which is false. The mismatch is now reported for what it is;
two aware legs (same or different zones) and two naive legs align as before.
"""

from __future__ import annotations

import pytest

from kalman.data import align_pair, synthetic_pair


def _legs(n: int = 300):
    pair, _ = synthetic_pair(n=n)
    return pair.p1.copy(), pair.p2.copy()


def test_aware_second_leg_against_naive_first_leg_is_named_not_overlap():
    p1, p2 = _legs()
    p2.index = p2.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="tz-aware") as excinfo:
        align_pair(p1, p2, "A", "B")
    msg = str(excinfo.value)
    assert msg.startswith("A/B:")
    assert "None vs UTC" in msg
    assert "overlap too short" not in msg


def test_aware_first_leg_against_naive_second_leg_is_named_too():
    p1, p2 = _legs()
    p1.index = p1.index.tz_localize("UTC")
    with pytest.raises(ValueError, match=r"tz-aware .* \(UTC vs None\)"):
        align_pair(p1, p2, "A", "B")


def test_mismatch_is_reported_even_when_overlap_would_be_waived():
    p1, p2 = _legs()
    p2.index = p2.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="tz-aware"):
        align_pair(p1, p2, "A", "B", min_overlap=0)


def test_both_legs_localized_to_utc_align_every_row():
    p1, p2 = _legs()
    pair = align_pair(p1.tz_localize("UTC"), p2.tz_localize("UTC"), "A", "B")
    assert len(pair) == 300
    assert str(pair.index.tz) == "UTC"


def test_two_aware_legs_in_different_zones_are_left_to_pandas():
    p1, p2 = _legs()
    pair = align_pair(p1.tz_localize("UTC"), p2.tz_localize("UTC").tz_convert("America/New_York"),
                      "A", "B")
    assert len(pair) == 300                     # intersected by instant


def test_naive_legs_are_unchanged():
    p1, p2 = _legs()
    pair = align_pair(p1, p2, "A", "B")
    assert len(pair) == 300 and pair.index.tz is None
