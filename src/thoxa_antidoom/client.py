"""vLLM / OpenAI-compatible client for batch generation with logprobs.

Wraps a vLLM server (or any OpenAI-compatible /v1/completions endpoint) to
generate completions with top-k logprobs at each position. The generated tokens
and logprobs are accumulated into a :class:`~thoxa_antidoom.tokens.TokenState`
for downstream loop detection and FTPO pair extraction.

Copyright (c) 2026 Thox.ai LLC. All rights reserved.
CTO: Tommy Xaypanya | CEO: Craig Ross
"""

from __future__ import annotations

import logging
from typing import Any

from thoxa_antidoom.tokens import TokenState

logger = logging.getLogger(__name__)


class GenerationClient:
    """Minimal OpenAI-compatible completions client with logprobs support."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: int = 480):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def completions(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        top_p: float | None = 1.0,
        top_k: int | None = 50,
        min_p: float | None = 0.01,
        top_logprobs: int = 20,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send a single completions request and return the raw JSON response."""
        import httpx

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_logprobs": top_logprobs,
            "logprobs": top_logprobs,
        }
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if min_p is not None:
            body["min_p"] = min_p
        if stop:
            body["stop"] = stop

        resp = httpx.post(
            f"{self.base_url}/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def generate_state(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        top_p: float | None = 1.0,
        top_k: int | None = 50,
        min_p: float | None = 0.01,
        top_logprobs: int = 20,
        stop: list[str] | None = None,
    ) -> TokenState:
        """Generate a completion and return a populated :class:`TokenState`."""
        result = self.completions(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            top_logprobs=top_logprobs,
            stop=stop,
        )
        return self._parse_result(result, prompt, top_logprobs)

    def _parse_result(
        self, result: dict[str, Any], prompt: str, top_logprobs: int
    ) -> TokenState:
        """Parse an OpenAI completions response into a TokenState."""
        state = TokenState(prompt=prompt)
        choices = result.get("choices", [])
        if not choices:
            return state

        choice = choices[0]
        logprobs_data = choice.get("logprobs")
        if logprobs_data is None:
            # No logprobs; just decode the text.
            text = choice.get("text", "")
            # Tokenize naively by character is not ideal, but without logprobs
            # we cannot build proper token-level state. Return text-only state.
            state.text = text
            return state

        tokens = logprobs_data.get("tokens", [])
        top_logprobs_list = logprobs_data.get("top_logprobs", [])

        logprobs_dict: dict[int, list[tuple[str, float]]] = {}
        for idx, top_lp in enumerate(top_logprobs_list):
            if top_lp is None:
                continue
            alts = []
            for tok_str, lp in top_lp.items():
                alts.append((tok_str, float(lp)))
            if alts:
                logprobs_dict[idx] = alts

        state.append(tokens, logprobs_dict)
        return state
