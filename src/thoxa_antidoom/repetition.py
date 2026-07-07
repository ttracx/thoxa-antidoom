"""Doom-loop detector.

Scans model completions for degenerate inner repetition. A loop is flagged when
a span of text repeats at least ``min_repeats`` times and covers at least
``min_total_repeated`` characters. The detector returns the character offset
where the *second* occurrence begins (the repeat start) so callers can map it
back to token space and identify the first loop-starting token.

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepeatHit:
    """A verified repetition region inside a completion.

    Attributes:
        start: char offset where the *first* occurrence of the repeating
            pattern begins.
        end: char offset one past the *last* character of the final repeat.
        period: length in characters of the repeating unit.
        repeats: how many times the unit repeats.
        snippet: a short preview of the repeating pattern.
    """

    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self) -> int:
        """Char offset where the second occurrence begins (first loop token)."""
        return self.start + self.period


def _verify_repetition_at(
    text: str,
    start_pos: int,
    period: int,
    min_repeats: int,
    min_total_repeated: int,
) -> tuple[bool, RepeatHit | None]:
    """Confirm that ``text[start_pos : start_pos+period]`` repeats enough times.

    Walks forward and backward from ``start_pos`` to capture the full run of
    repetitions, then checks the combined count and character coverage.
    """
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return False, None

    pattern = text[start_pos : start_pos + period]

    # Walk forward from start_pos.
    reps = 0
    pos = start_pos
    while pos + period <= len(text) and text[pos : pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos

    # Walk backward to catch repetitions that begin before start_pos.
    pos = start_pos - period
    while pos >= 0 and text[pos : pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period

    total = reps * period
    if reps >= min_repeats and total >= min_total_repeated:
        snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
        return True, RepeatHit(start_pos, end_pos, period, reps, snippet)
    return False, None


def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,
    sample_len: int = 16,
    sample_interval: int = 128,
) -> tuple[bool, RepeatHit | None]:
    """Detect inner repetition in a completion string.

    Samples short fingerprints at regular intervals, looks for the same
    fingerprint elsewhere in the text, and if found, derives a candidate period
    (distance between the two occurrences) and verifies it with
    :func:`_verify_repetition_at`. This is faster than brute-forcing every
    possible period while still catching the vast majority of real loops.

    Args:
        text: the completion to scan.
        min_repeats: minimum number of full repetitions to qualify as a loop.
        max_period: upper bound on the repeating unit length in characters.
        min_period: lower bound on the repeating unit length.
        min_total_repeated: minimum total characters covered by all repeats.
        sample_len: character width of each fingerprint probe.
        sample_interval: stride between fingerprint probes.

    Returns:
        ``(True, RepeatHit)`` when a loop is found, else ``(False, None)``.
    """
    if not text or len(text) < min_total_repeated:
        return False, None

    n = len(text)
    for sample_pos in range(0, n - sample_len, sample_interval):
        fingerprint = text[sample_pos : sample_pos + sample_len]

        # Look for the same fingerprint later in the text.
        other_pos = text.find(fingerprint, sample_pos + sample_len)
        if other_pos != -1:
            candidate_period = other_pos - sample_pos
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(
                    text,
                    sample_pos,
                    candidate_period,
                    min_repeats=min_repeats,
                    min_total_repeated=min_total_repeated,
                )
                if found:
                    return True, hit

        # Also check for an earlier occurrence of the same fingerprint.
        other_pos = text.rfind(fingerprint, 0, sample_pos)
        if other_pos != -1:
            candidate_period = sample_pos - other_pos
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(
                    text,
                    other_pos,
                    candidate_period,
                    min_repeats=min_repeats,
                    min_total_repeated=min_total_repeated,
                )
                if found:
                    return True, hit

    return False, None
