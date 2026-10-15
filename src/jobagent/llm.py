"""The single entry point for every Claude call in the agent.

Every call:
- uses claude-opus-5-5 with an explicit effort (this model defaults to "medium"),
- opts into server-side refusal fallback (`fallbacks: "default"`),
- puts the system prompt in a cached block, so keep system prompts byte-stable
  (no dates, no per-job data) and put the volatile part (the JD) in `content`,
- appends token usage and estimated cost to data/logs/llm_usage.jsonl.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from jobagent import settings

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens for claude-opus-5-5 (cache write = 1.25x input on the 5-minute TTL).
PRICE_INPUT = 4.00
PRICE_OUTPUT = 20.00
PRICE_CACHE_WRITE = 5.00
PRICE_CACHE_READ = 0.20

USAGE_LOG = settings.LOG_DIR / "llm_usage.jsonl"

Effort = Literal["low", "medium", "high"]
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Claude returned no usable output (refusal, truncation, or missing key)."""


_client: anthropic.Anthropic | None = None


def api_key_configured() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool(key) and key != settings.API_KEY_PLACEHOLDER


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not api_key_configured():
            raise LLMError(
                f"ANTHROPIC_API_KEY is not set (or still the placeholder) in {settings.ROOT / '.env'}. "
                "Create one at console.anthropic.com → API keys."
            )
        _client = anthropic.Anthropic()
    return _client


def _system_blocks(system: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _check_stop(task: str, response: Any) -> None:
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMError(f"{task}: Claude declined the request (category={category}).")
    if response.stop_reason == "max_tokens":
        raise LLMError(f"{task}: output hit max_tokens; raise the limit for this task.")


def _log_usage(task: str, response: Any) -> float:
    u = response.usage
    cache_write = u.cache_creation_input_tokens or 0
    cache_read = u.cache_read_input_tokens or 0
    cost = (
        u.input_tokens * PRICE_INPUT
        + u.output_tokens * PRICE_OUTPUT
        + cache_write * PRICE_CACHE_WRITE
        + cache_read * PRICE_CACHE_READ
    ) / 1_000_000
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "cost_usd": round(cost, 6),
        "request_id": response._request_id,
    }
    with USAGE_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return cost


def parse(
    schema: type[T],
    *,
    task: str,
    system: str,
    content: str | list[dict[str, Any]],
    effort: Effort,
    max_tokens: int = 8000,
) -> T:
    """Structured call: returns a validated instance of `schema`."""
    try:
        response = _get_client().beta.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=_system_blocks(system),
            messages=[{"role": "user", "content": content}],
            output_config={"effort": effort},
            output_format=schema,
        )
    except ValidationError as e:
        # The SDK validates inside parse(), so a max_tokens cut-off surfaces here as invalid JSON.
        raise LLMError(
            f"{task}: output was not valid {schema.__name__} JSON (usually max_tokens={max_tokens} "
            "was too low)."
        ) from e
    _log_usage(task, response)
    _check_stop(task, response)
    if response.parsed_output is None:
        raise LLMError(f"{task}: response did not match the {schema.__name__} schema.")
    return response.parsed_output


def ask(
    *,
    task: str,
    system: str,
    content: str | list[dict[str, Any]],
    effort: Effort,
    max_tokens: int = 4000,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, float]:
    """Free-text call: returns (text, estimated cost in USD). `tools` is for server tools (web search)."""
    response = _get_client().beta.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=_system_blocks(system),
        messages=[{"role": "user", "content": content}],
        output_config={"effort": effort},
        **({"tools": tools} if tools else {}),
    )
    cost = _log_usage(task, response)
    _check_stop(task, response)
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, cost
