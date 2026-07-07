"""Chosen-token selection at the loop-starting position.

Given the base model's top-k logprob alternatives at the position where a doom
loop begins, this module filters and selects plausible replacement tokens
("chosen" tokens) that are not the rejected loop-starting token. The selected
chosen tokens form the positive side of the FTPO preference pair.

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

import math

from thoxa_antidoom.tokens import decode_token


def _normalise(
    logprob_items: list[tuple[str, float]],
    temperature: float,
) -> list[tuple[str, float]]:
    """Convert logprobs to temperature-scaled probabilities summing to 1."""
    if not logprob_items:
        return []
    temp = max(float(temperature), 1e-6)
    vals = [
        (tok, math.exp(lp) ** (1.0 / temp))
        for tok, lp in logprob_items
        if lp is not None
    ]
    total = sum(v for _, v in vals)
    if total <= 0:
        return []
    return [(tok, val / total) for tok, val in vals]


def select_chosen_tokens(
    logprob_items: list[tuple[str, float]],
    *,
    rejected_token: str,
    temperature: float,
    min_p: float | None,
    top_k: int | None,
    max_tokens: int,
    min_decoded_chars: int = 1,
    require_alnum: bool = False,
    skip_raw_token_substrings: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Select up to ``max_tokens`` alternative tokens at the rejection point.

    Filters out the rejected token (by raw string or decoded match), applies
    optional min_p and top_k thresholds, then keeps tokens whose decoded surface
    form meets the minimum character / alnum requirements.

    Args:
        logprob_items: ``[(raw_token_string, logprob), ...]`` from the model.
        rejected_token: the raw token string that starts the loop.
        temperature: used for probability renormalisation.
        min_p: if set, drop alternatives below this fraction of the top prob.
        top_k: if set, keep only the top-k alternatives by probability.
        max_tokens: maximum number of chosen tokens to return.
        min_decoded_chars: minimum stripped decoded length to keep a token.
        require_alnum: if True, require at least one alphanumeric character.
        skip_raw_token_substrings: reject tokens whose raw string contains any
            of these substrings (e.g. known slop tokens).

    Returns:
        ``(raw_tokens, decoded_tokens)`` lists of the same length.
    """
    pairs = _normalise(logprob_items, temperature)
    if not pairs:
        return [], []

    rejected_decoded = decode_token(rejected_token).lower()

    def is_rejected(tok: str) -> bool:
        return tok == rejected_token or decode_token(tok).lower() == rejected_decoded

    pairs = [(tok, p) for tok, p in pairs if not is_rejected(tok)]
    if not pairs:
        return [], []

    if min_p is not None:
        floor = min_p * max(p for _, p in pairs)
        pairs = [(tok, p) for tok, p in pairs if p >= floor]
    if top_k is not None and len(pairs) > top_k:
        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]

    # Sort ascending so lower-probability alternatives are included first,
    # giving the trainer a broader spread of chosen tokens.
    pairs.sort(key=lambda item: item[1])
    raw: list[str] = []
    decoded: list[str] = []
    skip_raw_token_substrings = skip_raw_token_substrings or []
    for tok, _ in pairs:
        surf = decode_token(tok)
        if any(substring in tok for substring in skip_raw_token_substrings):
            continue
        if len(surf.strip()) < min_decoded_chars:
            continue
        if require_alnum and not any(c.isalnum() for c in surf):
            continue
        raw.append(tok)
        decoded.append(surf)
        if len(raw) >= max_tokens:
            break
    return raw, decoded
