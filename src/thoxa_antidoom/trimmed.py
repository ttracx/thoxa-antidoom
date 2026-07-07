"""Marker for trimmed source prompts.

Rows whose source prompt was truncated during generation are marked with a
prefix so the training filter can discard them (over-length prompts produce
unreliable preference signals).

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

from typing import Any

TRIMMED_PROMPT_PREFIX = "[TRIMMED_SOURCE_PROMPT]"


def row_has_trimmed_source_prompt(row: dict[str, Any]) -> bool:
    """Return True if the row's source prompt was trimmed during generation."""
    source_prompt = row.get("source_prompt") or ""
    return source_prompt.startswith(TRIMMED_PROMPT_PREFIX)
