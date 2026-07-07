"""Tests for the doom-loop repetition detector."""

from thoxa_antidoom.repetition import find_inner_repetition


def test_detects_simple_repetition():
    text = "Let me solve this. " + "Wait, let me reconsider. " * 6
    found, hit = find_inner_repetition(text)
    assert found
    assert hit is not None
    assert hit.repeats >= 4
    assert hit.period > 0


def test_no_repetition_returns_false():
    text = "This is a normal sentence with no repetition at all whatsoever."
    found, hit = find_inner_repetition(text)
    assert not found
    assert hit is None


def test_min_repeats_threshold():
    text = "blah " * 3
    found, hit = find_inner_repetition(text, min_repeats=4, min_total_repeated=10)
    assert not found


def test_min_total_repeated_threshold():
    text = "x " * 10
    found, hit = find_inner_repetition(text, min_total_repeated=60)
    assert not found


def test_repeat_start_offset():
    base = "Here is my answer. " * 10
    loop = "I need to reconsider this problem carefully now. " * 5
    text = base + loop
    found, hit = find_inner_repetition(text)
    assert found
    assert hit is not None
    assert hit.repeat_start == hit.start + hit.period


def test_empty_text():
    found, hit = find_inner_repetition("")
    assert not found


def test_short_text():
    found, hit = find_inner_repetition("ab")
    assert not found


def test_small_sample_len_catches_short_period_loops():
    # When the repeating unit is shorter than the default sample_len (16),
    # the fingerprint probe must be shrunk to <= the period to catch it.
    text = "prefix. " + "loop unit. " * 6
    found, hit = find_inner_repetition(
        text, sample_len=8, sample_interval=4, min_total_repeated=44
    )
    assert found
    assert hit is not None
    assert hit.repeats >= 4


def test_long_completion_doom_loop():
    # Simulates a realistic doom loop in a reasoning trace.
    prefix = "Let me work through this step by step. " * 5
    loop = "Wait, I need to reconsider my approach to this problem. " * 5
    text = prefix + loop
    found, hit = find_inner_repetition(text)
    assert found
    assert hit is not None
    assert hit.repeats >= 4
    assert hit.snippet
