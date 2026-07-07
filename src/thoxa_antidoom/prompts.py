"""Prompt dataset loading from HuggingFace or local JSONL.

Supports three prompt field shapes: plain strings, OpenAI-style chat message
lists, and ShareGPT-style conversation lists. The ``source`` field (if present)
is carried through to FTPO rows for source balancing.

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def _convert_sharegpt_to_chat(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert ShareGPT {from, value} messages to OpenAI {role, content}."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    result = []
    for msg in messages:
        role = role_map.get(msg.get("from", ""), "user")
        result.append({"role": role, "content": msg.get("value", "")})
    return result


def _parse_prompt_field(value: Any) -> tuple[str, list[dict[str, str]] | None]:
    """Parse a prompt field value into (plain_text, chat_messages_or_None).

    A plain string becomes ``(text, None)``. A list of {role, content} dicts
    becomes ``(None, messages)``. A list of {from, value} dicts is converted
    from ShareGPT to chat format.
    """
    if isinstance(value, str):
        return value, None

    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            if "role" in first:
                return "", value
            if "from" in first:
                return "", _convert_sharegpt_to_chat(value)

    return str(value), None


def load_prompts(
    *,
    hf_dataset: str | None = None,
    hf_split: str = "train",
    hf_source_field: str | None = None,
    hf_source_values: list[str] | None = None,
    input_jsonl: Path | None = None,
    prompt_field: str = "conversations",
    max_prompts: int | None = None,
    skip_eval_key_substrings: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield prompt dicts with keys: text, chat_messages, source.

    Reads from a HuggingFace dataset or a local JSONL file. Each yielded dict
    has ``text`` (plain prompt string) and/or ``chat_messages`` (OpenAI-style
    list), plus ``source`` for provenance tracking.
    """
    skip_eval_key_substrings = skip_eval_key_substrings or []

    if input_jsonl is not None:
        yield from _load_from_jsonl(
            input_jsonl, prompt_field, hf_source_field, max_prompts, skip_eval_key_substrings
        )
    elif hf_dataset is not None:
        yield from _load_from_hf(
            hf_dataset, hf_split, hf_source_field, hf_source_values,
            prompt_field, max_prompts, skip_eval_key_substrings
        )
    else:
        raise ValueError("Either hf_dataset or input_jsonl must be specified")


def _load_from_jsonl(
    path: Path,
    prompt_field: str,
    source_field: str | None,
    max_prompts: int | None,
    skip_substrings: list[str],
) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get(source_field, "")) if source_field else ""
            if _should_skip(source, skip_substrings):
                continue
            value = row.get(prompt_field, "")
            text, chat = _parse_prompt_field(value)
            yield {"text": text, "chat_messages": chat, "source": source}
            count += 1
            if max_prompts and count >= max_prompts:
                return


def _load_from_hf(
    dataset_name: str,
    split: str,
    source_field: str | None,
    source_values: list[str] | None,
    prompt_field: str,
    max_prompts: int | None,
    skip_substrings: list[str],
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    count = 0
    for row in ds:
        source = str(row.get(source_field, "")) if source_field else ""
        if source_values and source not in source_values:
            continue
        if _should_skip(source, skip_substrings):
            continue
        value = row.get(prompt_field, "")
        text, chat = _parse_prompt_field(value)
        yield {"text": text, "chat_messages": chat, "source": source}
        count += 1
        if max_prompts and count >= max_prompts:
            return


def _should_skip(source: str, skip_substrings: list[str]) -> bool:
    source_lower = source.lower()
    return any(sub in source_lower for sub in skip_substrings)
