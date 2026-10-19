"""Shared fixtures. `fake_api` routes the Anthropic SDK through a mock transport (no network, no key)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx2
import pytest

from jobagent import health, llm, settings, setup_status
from jobagent.agents import main, tracker
from jobagent.apply import browser, evidence, naukri, router
from jobagent.outreach import contacts, gmail, outbox

SAMPLE_PROFILE = settings.CONFIG_DIR / "profile.example.json"


def fake_message(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": llm.MODEL,
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": "x"}
        if stop_reason == "refusal"
        else None,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 2000,
        },
    }


@dataclass
class FakeAPI:
    usage_log: Path
    sent: list[dict] = field(default_factory=list)
    body: dict = field(default_factory=lambda: fake_message("{}"))
    routes: dict[str, dict] = field(default_factory=dict)

    def reply(self, text: str, stop_reason: str = "end_turn") -> None:
        self.body = fake_message(text, stop_reason)

    def reply_when(self, marker: str, text: str, stop_reason: str = "end_turn") -> None:
        """Answer differently for calls whose system prompt or user message contains `marker`.

        Later routes win, so a specific marker (a question's text) can override a broad one (the task).
        """
        self.routes[marker] = fake_message(text, stop_reason)

    def respond(self, body: dict) -> dict:
        system = body.get("system", [{}])[0].get("text", "")
        user = json.dumps(body.get("messages", [])[-1:])
        matches = [reply for marker, reply in self.routes.items() if marker in user or marker in system]
        return matches[-1] if matches else self.body


@pytest.fixture(autouse=True)
def isolate_from_real_setup(monkeypatch, tmp_path):
    """Every test: no real Chrome profile, email, inbox, DNS, locks or data/ writes, and never "live" because
    the user's setup is.

    Tests that need a browser or live mode opt in explicitly (see naukri_world / go_live).
    """

    def refuse(what: str):
        def _refuse(*_args, **_kwargs):
            raise AssertionError(f"tests must never {what}")
        return _refuse

    monkeypatch.setattr(browser, "naukri_context", refuse("open the agent's real Chrome profile"))
    monkeypatch.setattr(gmail, "send", refuse("send a real email"))
    monkeypatch.setattr(gmail, "Mailbox", refuse("open the real Gmail inbox"))
    monkeypatch.setattr(contacts, "has_mx", lambda domain: True)  # no DNS lookups in tests
    monkeypatch.setattr(router, "resolve", lambda url, client: url)  # never follow real short links
    monkeypatch.setattr(health, "online", lambda timeout=6.0: True)  # never depend on the real network
    monkeypatch.delenv("GMAIL_ADDRESS", raising=False)  # .env has the real Gmail: tests opt in with fake ones
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    monkeypatch.setattr(setup_status, "ready_for_live", lambda: False)
    monkeypatch.setattr(settings, "RESUME_DIR", tmp_path / "resumes")
    monkeypatch.setattr(evidence, "SHOTS_DIR", tmp_path / "shots")
    monkeypatch.setattr(tracker, "CSV_PATH", tmp_path / "applications.csv")
    monkeypatch.setattr(main, "LOCK_PATH", tmp_path / "main.lock")  # a real tick may hold the real lock
    monkeypatch.setattr(outbox, "LOCK_PATH", tmp_path / "outbox.lock")


@pytest.fixture
def fake_api(monkeypatch, tmp_path) -> FakeAPI:
    api = FakeAPI(usage_log=tmp_path / "usage.jsonl")

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        api.sent.append({"headers": dict(request.headers), "body": body})
        return httpx2.Response(200, json=api.respond(body))

    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "USAGE_LOG", api.usage_log)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # .env holds the placeholder, which disables scoring
    return api
