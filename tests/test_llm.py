"""Offline checks of the Claude wrapper: request shape, parsing, refusal handling, usage log."""

import json

import pytest
from pydantic import BaseModel

from jobagent import llm


class Score(BaseModel):
    score: int
    decision: str


def test_parse_sends_expected_request_and_returns_model(fake_api):
    fake_api.reply('{"score": 82, "decision": "apply"}')
    result = llm.parse(Score, task="score", system="rubric", content="JD text", effort="low")

    assert result == Score(score=82, decision="apply")
    body, headers = fake_api.sent[0]["body"], fake_api.sent[0]["headers"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert llm.FALLBACK_BETA in headers["anthropic-beta"]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in body  # thinking can't be disabled on this model; effort controls depth

    record = json.loads(fake_api.usage_log.read_text().strip())
    # 1000*4 + 500*20 + 2000*0.20 per million
    assert record["cost_usd"] == pytest.approx(0.0144)
    assert record["task"] == "score"


def test_refusal_raises(fake_api):
    fake_api.reply("", stop_reason="refusal")
    with pytest.raises(llm.LLMError, match="declined"):
        llm.ask(task="email", system="s", content="c", effort="low")
    with pytest.raises(llm.LLMError, match="declined"):
        llm.parse(Score, task="score", system="s", content="c", effort="low")


def test_truncation_raises(fake_api):
    fake_api.reply('{"score": 8', stop_reason="max_tokens")
    with pytest.raises(llm.LLMError, match="max_tokens"):
        llm.ask(task="tailor", system="s", content="c", effort="medium")
    with pytest.raises(llm.LLMError, match="max_tokens"):
        llm.parse(Score, task="score", system="s", content="c", effort="low")


def test_placeholder_key_is_not_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-REPLACE_ME")
    assert not llm.api_key_configured()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    assert llm.api_key_configured()
