"""Token-level helpers for mapping between character offsets and token indices.

The generation pipeline records raw token strings and per-position top-k
logprobs. This module provides a lightweight ``TokenState`` that accumulates
those records, builds a running decoded text buffer, and translates character
positions from the repetition detector back into token indices so the FTPO pair
builder can identify the exact rejected token.

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _build_u2b_full() -> dict[int, int]:
    """Build a Unicode-codepoint to Latin-1-byte mapping for mojibake repair."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {ord(chr(c)): b for b, c in zip(bs, cs)}
    for b in range(256):
        u2b[0x2500 + b] = b
    return u2b


_U2B = _build_u2b_full()


def fix_mojibake(text: str) -> str:
    """Repair common byte-level mojibake in raw token strings from tokenizers."""
    buf = bytearray()
    changed = False
    for ch in text:
        cp = ord(ch)
        if cp in _U2B:
            buf.append(_U2B[cp])
            changed = True
        else:
            buf.extend(ch.encode("utf-8"))
    if not changed:
        return text
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return text


def decode_token(token: str) -> str:
    """Decode a raw tokenizer token string into human-readable surface text."""
    if not token:
        return token
    token = fix_mojibake(token)
    return token.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")


@dataclass
class TokenState:
    """Accumulates generated tokens and their logprob alternatives.

    Maintains a parallel running text buffer so the repetition detector's
    character offsets can be mapped back to token indices.
    """

    prompt: str
    token_strings: list[str] = field(default_factory=list)
    decoded_lens: list[int] = field(default_factory=list)
    text: str = ""
    logprobs: dict[int, list[tuple[str, float]]] = field(default_factory=dict)

    def append(
        self,
        token_strings: list[str],
        logprobs: dict[int, list[tuple[str, float]]],
    ) -> None:
        """Append a batch of generated tokens and their logprob alternatives."""
        start = len(self.token_strings)
        for raw in token_strings:
            decoded = decode_token(raw)
            self.token_strings.append(raw)
            self.decoded_lens.append(len(decoded))
            self.text += decoded
        for rel_idx, alts in logprobs.items():
            abs_idx = start + rel_idx
            if abs_idx < len(self.token_strings):
                self.logprobs[abs_idx] = alts

    def char_to_token_index(self, char_pos: int) -> int | None:
        """Map a character offset in the running text to a token index."""
        if char_pos < 0 or char_pos >= len(self.text):
            return None
        running = 0
        for idx, length in enumerate(self.decoded_lens):
            running += length
            if char_pos < running:
                return idx
        return None

    def context_before(self, token_idx: int) -> str:
        """Return the full text (prompt + decoded tokens) up to ``token_idx``."""
        return self.prompt + "".join(
            decode_token(tok) for tok in self.token_strings[:token_idx]
        )
